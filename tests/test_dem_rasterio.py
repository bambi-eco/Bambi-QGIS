# -*- coding: utf-8 -*-
"""Real-rasterio tests for the DEM processing chain.

Unlike tests/test_dem_downloader.py (which fakes or blocks rasterio), these
tests exercise the actual ``merge_and_clip`` / ``reproject_geotiff`` /
``generate_mesh`` bodies against tiny synthetic GeoTIFFs, including the
end-to-end conversion workers. Skipped automatically when rasterio/shapely
are not installed (they are part of the unit Docker image).
"""
import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")

from rasterio.crs import CRS  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from bambi_wildlife_detection.austria_dem_downloader import (  # noqa: E402
    BEV_CRS,
    BEV_CRS_PROJ4,
    BEVDownloader,
    BoundingBox,
    DEMDownloadWorker,
    DEMProcessor,
    GeoTIFFConversionWorker,
    GLTFMeshGenerator,
    get_proj4_for_crs,
)
from tests.fakes import SignalRecorder, parse_glb  # noqa: E402

# Small area near Golling, Austria (the BEV grid covers it)
BBOX = BoundingBox(47.5, 13.2, 47.51, 13.21)
UTM33_PROJ4 = get_proj4_for_crs("EPSG:32633")


def _write_geotiff(path, data, west, north, resolution, proj4):
    """Write a single-band float32 GeoTIFF with the given top-left corner."""
    with rasterio.open(
        str(path), "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=CRS.from_proj4(proj4),
        transform=from_origin(west, north, resolution, resolution),
    ) as dst:
        dst.write(data.astype("float32"), 1)
    return path


@pytest.fixture
def bev_extent():
    """The test bbox expressed in the BEV tile CRS (EPSG:3035)."""
    return BBOX.to_crs(BEV_CRS)


@pytest.fixture
def bev_tile(tmp_path, bev_extent):
    """One synthetic BEV tile fully covering the bbox (plus 1 km margin)."""
    min_x, min_y, max_x, max_y = bev_extent
    west, north = min_x - 1000, max_y + 1000
    width = int((max_x + 1000 - west) / 50) + 1
    height = int((north - (min_y - 1000)) / 50) + 1
    data = 400.0 + np.random.default_rng(42).random((height, width)) * 100.0
    return _write_geotiff(
        tmp_path / "tile.tif", data, west, north, 50.0, BEV_CRS_PROJ4)


class TestMergeAndClip:
    def test_single_tile_clipped_to_bbox(self, tmp_path, bev_tile, bev_extent):
        processor = DEMProcessor()
        out = tmp_path / "clipped.tif"
        result = processor.merge_and_clip([bev_tile], out, BBOX)
        assert result == out
        with rasterio.open(out) as src:
            min_x, min_y, max_x, max_y = bev_extent
            # Clipped to the bbox (within one 50 m pixel of tolerance)
            assert src.bounds.left == pytest.approx(min_x, abs=60)
            assert src.bounds.top == pytest.approx(max_y, abs=60)
            assert src.bounds.right < min_x + 3000  # much smaller than tile
            # CRS forced to EPSG:3035 regardless of tile tagging
            assert "laea" in src.crs.to_proj4()

    def test_two_tiles_are_merged(self, tmp_path, bev_extent):
        min_x, min_y, max_x, max_y = bev_extent
        mid_x = (min_x + max_x) / 2
        north = max_y + 1000
        height = int((north - (min_y - 1000)) / 50) + 1

        def tile(path, west, east):
            width = int((east - west) / 50) + 1
            data = np.full((height, width), 500.0)
            return _write_geotiff(path, data, west, north, 50.0, BEV_CRS_PROJ4)

        west_tile = tile(tmp_path / "w.tif", min_x - 1000, mid_x + 100)
        east_tile = tile(tmp_path / "e.tif", mid_x - 100, max_x + 1000)

        processor = DEMProcessor()
        out = tmp_path / "merged.tif"
        result = processor.merge_and_clip([west_tile, east_tile], out, BBOX)
        assert result == out
        with rasterio.open(out) as src:
            # Output spans both halves of the bbox
            assert src.bounds.left <= min_x + 60
            assert src.bounds.right >= max_x - 60

    @staticmethod
    def _surface(tmp_path, bev_extent, split):
        """One elevation surface around the bbox - as a single tile, or cut
        in two along the middle of the bbox, on one cell grid."""
        min_x, min_y, max_x, max_y = bev_extent
        res = 50.0
        west = np.floor((min_x - 5000) / res) * res
        north = np.ceil((max_y + 5000) / res) * res
        width = int((max_x + 5000 - west) / res) + 1
        height = int((north - (min_y - 5000)) / res) + 1
        rows, cols = np.mgrid[0:height, 0:width]
        surface = 300.0 + 7.0 * rows + 0.5 * cols          # every cell differs
        if not split:
            return [_write_geotiff(tmp_path / "whole.tif", surface, west,
                                   north, res, BEV_CRS_PROJ4)]
        cut = int(((min_x + max_x) / 2 - west) / res)
        return [
            _write_geotiff(tmp_path / "w.tif", surface[:, :cut], west, north,
                           res, BEV_CRS_PROJ4),
            _write_geotiff(tmp_path / "e.tif", surface[:, cut:],
                           west + cut * res, north, res, BEV_CRS_PROJ4)]

    def test_across_a_tile_edge_only_the_flights_area_is_read(
            self, tmp_path, bev_extent, monkeypatch):
        """The tiles were merged whole, in memory, and written out again
        before clipping - some 20 GB each way for two 1 m tiles."""
        import rasterio.merge

        tiles = self._surface(tmp_path, bev_extent, split=True)
        seen = {}
        real_merge = rasterio.merge.merge

        def merge(sources, *args, **kwargs):
            seen.update(kwargs)
            mosaic, transform = real_merge(sources, *args, **kwargs)
            seen["shape"] = mosaic.shape
            return mosaic, transform

        monkeypatch.setattr(rasterio.merge, "merge", merge)
        out = tmp_path / "clipped.tif"
        assert DEMProcessor().merge_and_clip(tiles, out, BBOX) == out

        min_x, min_y, max_x, max_y = bev_extent
        assert seen["bounds"][0] <= min_x and seen["bounds"][2] >= max_x
        # the mosaic is the flight's area, not the two tiles
        with rasterio.open(tiles[0]) as tile:
            assert seen["shape"][1] < tile.height / 3
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "clipped.tif", "e.tif", "w.tif"]          # no temp_merged.tif

    def test_the_result_does_not_depend_on_where_the_tiles_are_cut(
            self, tmp_path, bev_extent):
        """Cell for cell what one tile gives - the same cells, unshifted."""
        whole = tmp_path / "whole"
        halves = tmp_path / "halves"
        whole.mkdir()
        halves.mkdir()
        one = DEMProcessor().merge_and_clip(
            self._surface(whole, bev_extent, split=False),
            whole / "out.tif", BBOX)
        two = DEMProcessor().merge_and_clip(
            self._surface(halves, bev_extent, split=True),
            halves / "out.tif", BBOX)

        with rasterio.open(one) as a, rasterio.open(two) as b:
            assert a.transform == b.transform
            assert (a.width, a.height) == (b.width, b.height)
            assert np.array_equal(a.read(1), b.read(1))

    def test_bounds_are_widened_to_whole_cells(self):
        from bambi_wildlife_detection.austria_dem_downloader import (
            snap_bounds_outward)

        assert snap_bounds_outward(
            (1012.0, 2012.0, 1088.0, 2088.0), (1000.0, 3000.0),
            (50.0, 50.0)) == (1000.0, 2000.0, 1100.0, 2100.0)
        # ... and bounds already on cell edges stay where they are
        assert snap_bounds_outward(
            (1000.0, 2000.0, 1100.0, 2100.0), (1000.0, 3000.0),
            (50.0, -50.0)) == (1000.0, 2000.0, 1100.0, 2100.0)

    def test_the_tiles_are_closed_when_processing_fails(
            self, tmp_path, bev_extent, monkeypatch):
        import rasterio.merge

        tiles = self._surface(tmp_path, bev_extent, split=True)
        opened = []
        real_open = rasterio.open

        def tracking_open(*args, **kwargs):
            handle = real_open(*args, **kwargs)
            opened.append(handle)
            return handle

        def broken(*_args, **_kwargs):
            raise RuntimeError("no memory")

        monkeypatch.setattr(rasterio, "open", tracking_open)
        monkeypatch.setattr(rasterio.merge, "merge", broken)
        assert DEMProcessor().merge_and_clip(
            tiles, tmp_path / "o.tif", BBOX) is None
        assert opened and all(handle.closed for handle in opened)

    def test_empty_input_returns_none(self, tmp_path):
        logs = []
        processor = DEMProcessor(log_callback=logs.append)
        assert processor.merge_and_clip([], tmp_path / "o.tif", BBOX) is None
        assert any("No input files" in m for m in logs)


class TestReprojectGeotiff:
    def test_reprojects_to_utm(self, tmp_path, bev_tile):
        processor = DEMProcessor(output_crs="EPSG:32633")
        out = tmp_path / "utm.tif"
        result = processor.reproject_geotiff(bev_tile, out)
        assert result == out
        with rasterio.open(out) as src:
            proj4 = src.crs.to_proj4()
            assert "utm" in proj4 and "zone=33" in proj4
            # UTM 33N coordinates for this area: ~365-370 km east, ~5.26M north
            assert 5.2e6 < src.bounds.top < 5.4e6

    def test_unreadable_input_returns_none(self, tmp_path):
        logs = []
        processor = DEMProcessor(log_callback=logs.append)
        result = processor.reproject_geotiff(tmp_path / "nope.tif", tmp_path / "o.tif")
        assert result is None
        assert any("Failed to reproject" in m for m in logs)


class TestGenerateMeshReal:
    def test_mesh_from_utm_geotiff(self, tmp_path):
        data = np.linspace(400, 500, 6 * 6).reshape(6, 6)
        tif = _write_geotiff(
            tmp_path / "dem.tif", data, 365000.0, 5263000.0, 10.0, UTM33_PROJ4)
        gen = GLTFMeshGenerator()
        ok = gen.generate_mesh(
            tif, tmp_path / "dem.glb", tmp_path / "dem.json",
            source_crs="EPSG:32633")
        assert ok is True

        meta = json.loads((tmp_path / "dem.json").read_text())
        assert meta["origin"][0] == pytest.approx(365000.0)
        assert meta["origin"][1] == pytest.approx(5263000.0 - 6 * 10.0)
        assert 13.0 < meta["origin_wgs84"]["longitude"] < 13.5
        assert 47.3 < meta["origin_wgs84"]["latitude"] < 47.7

        gltf, _ = parse_glb(tmp_path / "dem.glb")
        assert gltf["accessors"][0]["count"] == 36

    def test_mislabeled_wgs84_dem_is_reprojected_to_metres(self, tmp_path):
        # The DJI Terra case: degree-unit raster tagged as UTM. With a
        # projected source_crs the mesh generator must detect the mislabel
        # and reproject in-memory so vertices are in metres.
        data = np.full((6, 6), 450.0)
        tif = _write_geotiff(
            tmp_path / "dem.tif", data, 13.2, 47.51, 0.0005, UTM33_PROJ4)
        logs = []
        gen = GLTFMeshGenerator(log_callback=logs.append)
        ok = gen.generate_mesh(
            tif, tmp_path / "dem.glb", tmp_path / "dem.json",
            source_crs="EPSG:32633")
        assert ok is True
        assert any("likely mislabeled" in m for m in logs)
        assert any("Reprojecting WGS84 geographic DEM" in m for m in logs)

        meta = json.loads((tmp_path / "dem.json").read_text())
        # Origin must be in UTM metres now, not degrees
        assert meta["origin"][0] > 100000
        assert meta["origin"][1] > 5.2e6
        assert meta["origin_wgs84"]["longitude"] == pytest.approx(13.2, abs=0.01)
        assert meta["origin_wgs84"]["latitude"] == pytest.approx(47.51 - 6 * 0.0005, abs=0.01)


class TestGeoTIFFConversionWorkerSourceCrs:
    """Source-CRS handling for files rasterio cannot read the CRS of.

    Dutch AHN tiles carry a compound "Amersfoort / RD New + NAP height" CRS
    that rasterio reports as ``crs = None``; the worker must fall back to
    GDAL identification (or the user override) instead of silently building
    a mesh from unreprojected coordinates labeled with the target CRS.
    """

    def _write_crsless_rd_tile(self, tmp_path):
        # 60 m tile at RD New (95000, 481250) - the corner of AHN tile 24HN2 -
        # written without any CRS tag so rasterio reads crs=None.
        data = np.linspace(-1.3, 8.0, 6 * 6).reshape(6, 6)
        path = tmp_path / "ahn.tif"
        with rasterio.open(
            str(path), "w",
            driver="GTiff", height=6, width=6, count=1, dtype="float32",
            transform=from_origin(95000.0, 481310.0, 10.0, 10.0),
        ) as dst:
            dst.write(data.astype("float32"), 1)
        return path

    def _run(self, worker):
        worker.finished = SignalRecorder()
        worker.progress = SignalRecorder()
        worker.log = SignalRecorder()
        worker.run()
        assert worker.finished.calls, "worker did not finish"
        return worker.finished.calls[-1]

    def test_unknown_source_crs_fails_instead_of_mislabeling(
            self, tmp_path, monkeypatch):
        from bambi_wildlife_detection import austria_dem_downloader as add
        monkeypatch.setattr(add, "detect_geotiff_epsg", lambda p: None)
        tile = self._write_crsless_rd_tile(tmp_path)
        worker = GeoTIFFConversionWorker(
            str(tile), str(tmp_path / "out"), output_crs="EPSG:32631",
            simplify_factor=1)
        success, message = self._run(worker)
        assert success is False
        assert "Could not determine the source CRS" in message

    def test_gdal_identified_rd_new_is_reprojected_to_utm31(
            self, tmp_path, monkeypatch):
        from bambi_wildlife_detection import austria_dem_downloader as add
        monkeypatch.setattr(add, "detect_geotiff_epsg", lambda p: "EPSG:28992")
        tile = self._write_crsless_rd_tile(tmp_path)
        worker = GeoTIFFConversionWorker(
            str(tile), str(tmp_path / "out"), output_crs="EPSG:32631",
            simplify_factor=1)
        success, message = self._run(worker)
        assert success is True, message

        meta = json.loads((tmp_path / "out" / "ahn.json").read_text())
        assert meta["crs"] == "EPSG:32631"
        # RD (95000, 481250) is UTM 31N (~602.6 km, ~5797.4 km), lat/lon
        # (52.317, 4.505) in the Dutch dunes - not raw RD numbers.
        assert 602000 < meta["origin"][0] < 603500
        assert 5.796e6 < meta["origin"][1] < 5.799e6
        assert meta["origin_wgs84"]["latitude"] == pytest.approx(52.317, abs=0.01)
        assert meta["origin_wgs84"]["longitude"] == pytest.approx(4.505, abs=0.01)

    def test_source_crs_override_on_crsless_file(self, tmp_path, monkeypatch):
        from bambi_wildlife_detection import austria_dem_downloader as add
        monkeypatch.setattr(add, "detect_geotiff_epsg", lambda p: None)
        tile = self._write_crsless_rd_tile(tmp_path)
        worker = GeoTIFFConversionWorker(
            str(tile), str(tmp_path / "out"), output_crs="EPSG:32631",
            simplify_factor=1, source_crs_override="EPSG:28992")
        success, message = self._run(worker)
        assert success is True, message
        meta = json.loads((tmp_path / "out" / "ahn.json").read_text())
        assert 602000 < meta["origin"][0] < 603500


class TestGeoTIFFConversionWorkerEndToEnd:
    def test_converts_bev_geotiff_to_utm_mesh(self, tmp_path, bev_tile):
        out_dir = tmp_path / "out"
        worker = GeoTIFFConversionWorker(
            str(bev_tile), str(out_dir), output_crs="EPSG:32633",
            simplify_factor=4)
        worker.finished = SignalRecorder()
        worker.progress = SignalRecorder()
        worker.log = SignalRecorder()

        worker.run()

        assert worker.finished.calls, "worker did not finish"
        success, message = worker.finished.calls[-1]
        assert success is True, message
        assert message.endswith(".glb")
        assert (out_dir / "tile.tif").exists()   # reprojected copy
        assert (out_dir / "tile.glb").exists()
        meta = json.loads((out_dir / "tile.json").read_text())
        assert meta["crs"] == "EPSG:32633"


class TestDEMDownloadWorkerEndToEnd:
    def test_full_pipeline_with_local_tile(self, tmp_path, bev_tile, monkeypatch):
        csv_path = tmp_path / "flight.csv"
        csv_path.write_text(
            "latitude,longitude\n47.5,13.2\n47.51,13.21\n", encoding="utf-8")

        # Replace only the network step: every requested tile resolves to
        # the local synthetic tile; the rest of the pipeline runs for real.
        monkeypatch.setattr(
            BEVDownloader, "download_tile",
            lambda self, tile_name, force=False, tile_progress_callback=None: bev_tile)

        out_dir = tmp_path / "out"
        worker = DEMDownloadWorker(
            str(csv_path), str(out_dir), padding=50.0,
            output_crs="EPSG:32633", simplify_factor=4)
        worker.finished = SignalRecorder()
        worker.progress = SignalRecorder()
        worker.log = SignalRecorder()

        worker.run()

        assert worker.finished.calls, "worker did not finish"
        success, message = worker.finished.calls[-1]
        assert success is True, message
        assert (out_dir / "dem.tif").exists()
        assert (out_dir / "dem.glb").exists()
        meta = json.loads((out_dir / "dem.json").read_text())
        assert meta["crs"] == "EPSG:32633"
        assert 13.1 < meta["origin_wgs84"]["longitude"] < 13.3
