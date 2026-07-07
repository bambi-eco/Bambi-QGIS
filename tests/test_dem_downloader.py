# -*- coding: utf-8 -*-
"""Unit tests for the Austria DEM downloader module.

Network access is mocked (fake requests session), rasterio-dependent code
paths are exercised either through their documented ImportError fallbacks
(by blocking the import) or through a minimal fake rasterio module.
"""
import json
import sys

import numpy as np
import pytest

pytest.importorskip("requests")  # BEVDownloader imports requests at init

from bambi_wildlife_detection.austria_dem_downloader import (  # noqa: E402
    BEV_DOWNLOAD_BASE,
    BEVDownloader,
    BEVTileCalculator,
    BoundingBox,
    DEMDownloadWorker,
    DEMProcessor,
    GeoTIFFConversionWorker,
    GLTFMeshGenerator,
    TILE_SIZE,
    get_proj4_for_crs,
)
from tests.fakes import (  # noqa: E402
    FakeAffine,
    FakeBounds,
    FakeRasterioDataset,
    FakeResponse,
    FakeSession,
    SignalRecorder,
    install_fake_rasterio,
    parse_glb,
)


# ---------------------------------------------------------------------------
# get_proj4_for_crs
# ---------------------------------------------------------------------------

class TestGetProj4ForCrs:
    def test_known_epsg_returns_proj4(self):
        assert "+proj=utm +zone=33" in get_proj4_for_crs("EPSG:32633")
        assert "+proj=longlat" in get_proj4_for_crs("EPSG:4326")

    def test_unknown_crs_passes_through(self):
        assert get_proj4_for_crs("EPSG:9999") == "EPSG:9999"


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------

class TestBoundingBoxFromPoints:
    def test_orders_coordinates(self):
        bbox = BoundingBox.from_points(47.9, 13.6, 47.5, 13.2)
        assert bbox.min_lat == 47.5
        assert bbox.max_lat == 47.9
        assert bbox.min_lon == 13.2
        assert bbox.max_lon == 13.6

    def test_no_padding_keeps_bounds_exact(self):
        bbox = BoundingBox.from_points(47.5, 13.2, 47.9, 13.6, padding_meters=0)
        assert (bbox.min_lat, bbox.max_lat) == (47.5, 47.9)

    def test_padding_expands_bounds(self):
        bbox = BoundingBox.from_points(47.5, 13.2, 47.9, 13.6, padding_meters=1113.2)
        lat_pad = 1113.2 / 111320
        assert bbox.min_lat == pytest.approx(47.5 - lat_pad)
        assert bbox.max_lat == pytest.approx(47.9 + lat_pad)
        # Longitude padding is larger than latitude padding at 47.7 deg north
        lon_pad = bbox.max_lon - 13.6
        assert lon_pad > lat_pad


class TestBoundingBoxFromAirdataCsv:
    def _write_csv(self, tmp_path, header, rows):
        path = tmp_path / "flight.csv"
        lines = [header] + rows
        path.write_text("\n".join(lines), encoding="utf-8")
        return str(path)

    def test_standard_columns(self, tmp_path):
        path = self._write_csv(
            tmp_path,
            "time,latitude,longitude,altitude(m)",
            ["1,47.5,13.2,100", "2,47.9,13.6,110", "3,47.7,13.4,105"],
        )
        bbox = BoundingBox.from_airdata_csv(path)
        assert bbox is not None
        assert bbox.min_lat == 47.5
        assert bbox.max_lat == 47.9
        assert bbox.min_lon == 13.2
        assert bbox.max_lon == 13.6

    def test_alternate_column_names(self, tmp_path):
        path = self._write_csv(
            tmp_path,
            "GPS Lat,GPS Lng",
            ["47.5,13.2", "47.9,13.6"],
        )
        bbox = BoundingBox.from_airdata_csv(path)
        assert bbox is not None
        assert bbox.max_lon == 13.6

    def test_invalid_and_zero_rows_are_skipped(self, tmp_path):
        path = self._write_csv(
            tmp_path,
            "latitude,longitude",
            ["47.5,13.2", "abc,def", "0,0", "999,13.3", "47.9,13.6"],
        )
        bbox = BoundingBox.from_airdata_csv(path)
        assert bbox is not None
        assert bbox.min_lat == 47.5
        assert bbox.max_lat == 47.9

    def test_missing_gps_columns_returns_none(self, tmp_path):
        path = self._write_csv(tmp_path, "time,altitude", ["1,100"])
        assert BoundingBox.from_airdata_csv(path) is None

    def test_no_valid_rows_returns_none(self, tmp_path):
        path = self._write_csv(tmp_path, "latitude,longitude", ["0,0", "abc,def"])
        assert BoundingBox.from_airdata_csv(path) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert BoundingBox.from_airdata_csv(str(tmp_path / "nope.csv")) is None

    def test_padding_is_forwarded(self, tmp_path):
        path = self._write_csv(
            tmp_path, "latitude,longitude", ["47.5,13.2", "47.9,13.6"])
        padded = BoundingBox.from_airdata_csv(path, padding_meters=500)
        unpadded = BoundingBox.from_airdata_csv(path)
        assert padded.min_lat < unpadded.min_lat
        assert padded.max_lon > unpadded.max_lon


class TestBoundingBoxToCrs:
    def test_austria_bbox_lands_in_bev_grid(self):
        pytest.importorskip("pyproj")
        bbox = BoundingBox(47.5, 13.2, 47.9, 13.6)
        min_x, min_y, max_x, max_y = bbox.to_crs("EPSG:3035")
        assert min_x < max_x
        assert min_y < max_y
        # EPSG:3035 coordinates of Austria fall within the BEV tile grid
        assert BEVTileCalculator.AUSTRIA_BOUNDS["min_x"] < min_x < BEVTileCalculator.AUSTRIA_BOUNDS["max_x"]
        assert BEVTileCalculator.AUSTRIA_BOUNDS["min_y"] < min_y < BEVTileCalculator.AUSTRIA_BOUNDS["max_y"]

    def test_utm_conversion_produces_metric_coords(self):
        pytest.importorskip("pyproj")
        bbox = BoundingBox(47.5, 13.2, 47.9, 13.6)
        min_x, min_y, max_x, max_y = bbox.to_crs("EPSG:32633")
        # UTM northing for ~47.5 deg north is around 5.26 million metres
        assert 5.2e6 < min_y < 5.4e6
        assert max_x - min_x > 1000


# ---------------------------------------------------------------------------
# BEVTileCalculator
# ---------------------------------------------------------------------------

class TestBEVTileCalculator:
    @pytest.fixture
    def calc(self):
        return BEVTileCalculator()

    def test_tile_bounds_roundtrip(self, calc):
        name = calc.get_tile_name(2650000, 4400000)
        assert name == "N2650000E4400000"
        bounds = calc.get_tile_bounds(name)
        assert bounds == (4400000, 2650000, 4400000 + TILE_SIZE, 2650000 + TILE_SIZE)

    def test_tile_bounds_invalid_name(self, calc):
        assert calc.get_tile_bounds("not-a-tile") is None

    def test_download_urls_cover_all_patterns(self, calc):
        urls = calc.get_download_urls("N2650000E4400000")
        assert len(urls) == 3
        assert all(u.startswith(BEV_DOWNLOAD_BASE) for u in urls)
        assert urls[0].endswith("20230915/ALS_DTM_CRS3035RES50000mN2650000E4400000.tif")

    def test_download_urls_invalid_name(self, calc):
        assert calc.get_download_urls("garbage") == []

    def test_required_tiles_single(self, calc, monkeypatch):
        monkeypatch.setattr(
            BoundingBox, "to_crs",
            lambda self, crs: (4400010.0, 2650010.0, 4400020.0, 2650020.0))
        tiles = calc.get_required_tiles(BoundingBox(0, 0, 0, 0))
        assert tiles == ["N2650000E4400000"]

    def test_required_tiles_spanning_boundary(self, calc, monkeypatch):
        monkeypatch.setattr(
            BoundingBox, "to_crs",
            lambda self, crs: (4440000.0, 2640000.0, 4460000.0, 2660000.0))
        tiles = calc.get_required_tiles(BoundingBox(0, 0, 0, 0))
        assert sorted(tiles) == [
            "N2600000E4400000", "N2600000E4450000",
            "N2650000E4400000", "N2650000E4450000",
        ]


# ---------------------------------------------------------------------------
# BEVDownloader (network mocked)
# ---------------------------------------------------------------------------

class TestBEVDownloader:
    TILE = "N2650000E4400000"

    @pytest.fixture
    def downloader(self, tmp_path):
        return BEVDownloader(cache_dir=tmp_path / "cache")

    def test_init_creates_cache_dir(self, tmp_path):
        cache = tmp_path / "brand_new"
        BEVDownloader(cache_dir=cache)
        assert cache.is_dir()

    def test_cached_tile_is_reused(self, downloader):
        cached = downloader._get_cache_path(self.TILE)
        cached.write_bytes(b"tif-data")
        progress = []
        result = downloader.download_tile(self.TILE, tile_progress_callback=progress.append)
        assert result == cached
        assert progress == [100]

    def test_cancelled_before_start_returns_none(self, downloader):
        downloader.cancel()
        assert downloader.download_tile(self.TILE) is None

    def test_successful_download_writes_file(self, downloader):
        chunks = [b"a" * 1024, b"b" * 1024]
        downloader.session = FakeSession([FakeResponse(chunks=chunks)])
        progress = []
        result = downloader.download_tile(self.TILE, tile_progress_callback=progress.append)
        assert result is not None
        assert result.read_bytes() == b"a" * 1024 + b"b" * 1024
        assert progress[-1] == 100

    def test_404_falls_back_to_next_url_pattern(self, downloader):
        downloader.session = FakeSession([
            FakeResponse(status_code=404),
            FakeResponse(chunks=[b"x" * 100]),
        ])
        result = downloader.download_tile(self.TILE)
        assert result is not None
        assert len(downloader.session.requested_urls) == 2

    def test_all_urls_failing_returns_none(self, downloader):
        downloader.session = FakeSession([
            FakeResponse(status_code=404),
            FakeResponse(status_code=500),
            RuntimeError("connection reset"),
        ])
        result = downloader.download_tile(self.TILE)
        assert result is None
        assert not downloader._get_cache_path(self.TILE).exists()

    def test_cancel_mid_download_removes_partial_file(self, downloader):
        def chunk_gen():
            yield b"a" * 1024
            downloader.cancel()
            yield b"b" * 1024

        response = FakeResponse(content_length=2048)
        response.iter_content = lambda chunk_size=None: chunk_gen()
        downloader.session = FakeSession([response])

        result = downloader.download_tile(self.TILE)
        assert result is None
        assert not downloader._get_cache_path(self.TILE).exists()

    def test_download_tiles_for_bbox_collects_successes(self, downloader, monkeypatch):
        progress = []
        downloader.progress_callback = progress.append
        monkeypatch.setattr(
            downloader.tile_calculator, "get_required_tiles", lambda bbox: ["A", "B"])

        def fake_download(tile_name, force=False, tile_progress_callback=None):
            if tile_progress_callback:
                tile_progress_callback(100)
            if tile_name == "A":
                return downloader._get_cache_path(tile_name)
            return None

        monkeypatch.setattr(downloader, "download_tile", fake_download)
        result = downloader.download_tiles_for_bbox(BoundingBox(0, 0, 0, 0))
        assert result == [downloader._get_cache_path("A")]
        # Download phase maps to the 0-50% band of overall progress
        assert progress and all(0 <= p <= 50 for p in progress)


# ---------------------------------------------------------------------------
# DEMProcessor — the rasterio-free fallback paths
# ---------------------------------------------------------------------------

class TestDEMProcessorWithoutRasterio:
    @pytest.fixture
    def processor(self, monkeypatch):
        # Setting a module to None in sys.modules makes its import fail,
        # which is exactly the environment the fallback handles.
        monkeypatch.setitem(sys.modules, "rasterio", None)
        logs = []
        proc = DEMProcessor(log_callback=logs.append)
        proc.test_logs = logs
        return proc

    def test_merge_and_clip_reports_missing_library(self, processor, tmp_path):
        result = processor.merge_and_clip(
            [tmp_path / "a.tif"], tmp_path / "out.tif", BoundingBox(47, 13, 48, 14))
        assert result is None
        assert any("Missing required library" in m for m in processor.test_logs)

    def test_reproject_reports_missing_library(self, processor, tmp_path):
        result = processor.reproject_geotiff(tmp_path / "in.tif", tmp_path / "out.tif")
        assert result is None
        assert any("Missing required library" in m for m in processor.test_logs)

    def test_cancel_sets_flag(self, processor):
        assert processor._cancelled is False
        processor.cancel()
        assert processor._cancelled is True


# ---------------------------------------------------------------------------
# GLTFMeshGenerator
# ---------------------------------------------------------------------------

class TestGLTFMeshGenerator:
    def test_simplify_factor_clamped_to_one(self):
        assert GLTFMeshGenerator(simplify_factor=0).simplify_factor == 1
        assert GLTFMeshGenerator(simplify_factor=-3).simplify_factor == 1
        assert GLTFMeshGenerator(simplify_factor=4).simplify_factor == 4

    def test_write_gltf_produces_valid_glb(self, tmp_path):
        gen = GLTFMeshGenerator()
        vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 5]
        normals = [0, 0, 1] * 4
        indices = [0, 1, 2, 1, 3, 2]
        out = tmp_path / "mesh.glb"
        gen._write_gltf(out, vertices, normals, indices)

        gltf, binary = parse_glb(out)
        assert gltf["accessors"][0]["count"] == 4      # vertices
        assert gltf["accessors"][1]["count"] == 4      # normals
        assert gltf["accessors"][2]["count"] == 6      # indices
        assert gltf["accessors"][0]["max"] == [1.0, 1.0, 5.0]
        assert gltf["buffers"][0]["byteLength"] == len(binary)
        # Vertex data round-trips through the binary chunk
        view = gltf["bufferViews"][0]
        raw = binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]
        assert np.frombuffer(raw, dtype=np.float32).tolist() == [float(v) for v in vertices]

    def test_generate_mesh_without_rasterio_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "rasterio", None)
        logs = []
        gen = GLTFMeshGenerator(log_callback=logs.append)
        ok = gen.generate_mesh(tmp_path / "dem.tif", tmp_path / "dem.glb")
        assert ok is False
        assert any("Missing required library" in m for m in logs)

    def test_generate_mesh_from_utm_dem(self, monkeypatch, tmp_path):
        pytest.importorskip("pyproj")
        elevation = np.arange(16, dtype=np.float64).reshape(4, 4) + 400.0
        dataset = FakeRasterioDataset(
            elevation,
            transform=FakeAffine(10.0, 0.0, 500000.0, 0.0, -10.0, 5265000.0),
            bounds=FakeBounds(500000.0, 5264960.0, 500040.0, 5265000.0),
            epsg=32633,
        )
        install_fake_rasterio(monkeypatch, dataset)

        gen = GLTFMeshGenerator()
        glb_path = tmp_path / "dem.glb"
        meta_path = tmp_path / "dem.json"
        ok = gen.generate_mesh(
            tmp_path / "dem.tif", glb_path, meta_path, source_crs="EPSG:32633")
        assert ok is True
        assert glb_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["crs"] == "EPSG:32633"
        assert meta["width"] == 4 and meta["height"] == 4
        assert meta["origin"] == [500000.0, 5264960.0, 400.0]
        # UTM 33N easting 500000 is the central meridian at 15 deg east
        assert meta["origin_wgs84"]["longitude"] == pytest.approx(15.0, abs=0.01)
        assert 47.0 < meta["origin_wgs84"]["latitude"] < 48.0

        gltf, _ = parse_glb(glb_path)
        assert gltf["accessors"][0]["count"] == 16          # 4x4 grid, no NaNs
        assert gltf["accessors"][2]["count"] == 3 * 2 * 9   # 3x3 quads, 2 tris each

    def test_generate_mesh_detects_mislabeled_wgs84_dem(self, monkeypatch, tmp_path):
        # DJI Terra exports degree-unit DEMs tagged as UTM: bounds in the
        # degree range with a UTM EPSG must be re-interpreted as EPSG:4326.
        pytest.importorskip("pyproj")
        elevation = np.full((3, 3), 500.0)
        dataset = FakeRasterioDataset(
            elevation,
            transform=FakeAffine(0.0001, 0.0, 13.4, 0.0, -0.0001, 47.5003),
            bounds=FakeBounds(13.4, 47.5, 13.4003, 47.5003),
            epsg=32643,
        )
        install_fake_rasterio(monkeypatch, dataset)

        logs = []
        gen = GLTFMeshGenerator(log_callback=logs.append)
        ok = gen.generate_mesh(
            tmp_path / "dem.tif", tmp_path / "dem.glb", tmp_path / "dem.json")
        assert ok is True
        assert any("likely mislabeled" in m for m in logs)

        # Origin is interpreted as WGS84 directly (no bogus UTM transform)
        meta = json.loads((tmp_path / "dem.json").read_text())
        assert meta["origin_wgs84"]["longitude"] == pytest.approx(13.4, abs=1e-6)
        assert meta["origin_wgs84"]["latitude"] == pytest.approx(47.5, abs=1e-6)

    def test_generate_mesh_all_nodata_fails(self, monkeypatch, tmp_path):
        elevation = np.full((3, 3), -9999.0)
        dataset = FakeRasterioDataset(
            elevation,
            transform=FakeAffine(10.0, 0.0, 500000.0, 0.0, -10.0, 5265000.0),
            bounds=FakeBounds(500000.0, 5264970.0, 500030.0, 5265000.0),
            epsg=32633,
            nodata=-9999.0,
        )
        install_fake_rasterio(monkeypatch, dataset)

        logs = []
        gen = GLTFMeshGenerator(log_callback=logs.append)
        ok = gen.generate_mesh(tmp_path / "dem.tif", tmp_path / "dem.glb")
        assert ok is False
        assert any("No valid elevation data" in m for m in logs)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class TestDEMDownloadWorker:
    def _make_worker(self, tmp_path, csv_path="missing.csv"):
        worker = DEMDownloadWorker(csv_path, str(tmp_path / "out"))
        worker.finished = SignalRecorder()
        worker.progress = SignalRecorder()
        worker.log = SignalRecorder()
        return worker

    def test_run_with_unreadable_csv_fails(self, tmp_path):
        worker = self._make_worker(tmp_path, str(tmp_path / "missing.csv"))
        worker.run()
        assert worker.finished.calls == [(False, "Failed to extract GPS coordinates from CSV")]

    def test_run_with_no_tiles_fails(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "flight.csv"
        csv_path.write_text("latitude,longitude\n47.5,13.2\n47.9,13.6\n", encoding="utf-8")
        worker = self._make_worker(tmp_path, str(csv_path))
        monkeypatch.setattr(
            BEVDownloader, "download_tiles_for_bbox", lambda self, bbox: [])
        worker.run()
        assert worker.finished.calls == [
            (False, "No tiles downloaded - area may be outside Austria")]

    def test_cancel_propagates_to_components(self, tmp_path):
        worker = self._make_worker(tmp_path)

        class FakeComponent:
            cancelled = False

            def cancel(self):
                self.cancelled = True

        worker._downloader = FakeComponent()
        worker._processor = FakeComponent()
        worker.cancel()
        assert worker._cancelled is True
        assert worker._downloader.cancelled is True
        assert worker._processor.cancelled is True


class TestGeoTIFFConversionWorker:
    def test_run_with_missing_input_fails(self, tmp_path):
        worker = GeoTIFFConversionWorker(
            str(tmp_path / "nope.tif"), str(tmp_path / "out"))
        worker.finished = SignalRecorder()
        worker.progress = SignalRecorder()
        worker.log = SignalRecorder()
        worker.run()
        assert len(worker.finished.calls) == 1
        success, message = worker.finished.calls[0]
        assert success is False
        assert "Input file not found" in message

    def test_cancel_propagates_to_processor(self, tmp_path):
        worker = GeoTIFFConversionWorker(str(tmp_path / "a.tif"), str(tmp_path))

        class FakeProcessor:
            cancelled = False

            def cancel(self):
                self.cancelled = True

        worker._processor = FakeProcessor()
        worker.cancel()
        assert worker._cancelled is True
        assert worker._processor.cancelled is True
