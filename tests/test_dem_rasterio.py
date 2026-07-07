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
