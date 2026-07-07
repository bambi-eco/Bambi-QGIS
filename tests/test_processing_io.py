# -*- coding: utf-8 -*-
"""Unit tests for georeferencing file output helpers (world/prj files) and
frame GeoTIFF collection."""
import os

import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor


def _read_world_file(path):
    with open(path) as fh:
        return [float(line.strip()) for line in fh]


class TestSaveWorldFile:
    BOUNDS = (100.0, 200.0, 110.0, 220.0)  # min_x, min_y, max_x, max_y

    def test_tfw_values(self, processor, tmp_path):
        target = str(tmp_path / "out.tif")
        processor._save_world_file(target, self.BOUNDS, width=100, height=200)

        values = _read_world_file(str(tmp_path / "out.tfw"))
        pixel_x, rot1, rot2, pixel_y, origin_x, origin_y = values
        assert pixel_x == pytest.approx(0.1)
        assert rot1 == 0.0
        assert rot2 == 0.0
        assert pixel_y == pytest.approx(-0.1)
        # Origin is the *centre* of the top-left pixel.
        assert origin_x == pytest.approx(100.05)
        assert origin_y == pytest.approx(219.95)

    @pytest.mark.parametrize("filename,world_name", [
        ("out.tif", "out.tfw"),
        ("out.tiff", "out.tfw"),
        ("out.png", "out.pgw"),
        ("out.jpg", "out.jgw"),
        ("out.jpeg", "out.jgw"),
        ("out.dat", "out.dat.wld"),
    ])
    def test_world_file_extension(self, processor, tmp_path, filename, world_name):
        processor._save_world_file(str(tmp_path / filename), self.BOUNDS, 10, 10)
        assert (tmp_path / world_name).exists()


class TestSavePrjFile:
    def test_writes_wkt_for_epsg(self, processor, tmp_path):
        pytest.importorskip("pyproj")
        target = str(tmp_path / "out.tif")
        processor._save_prj_file(target, 32633, log_fn=None)
        content = (tmp_path / "out.prj").read_text()
        assert "UTM zone 33N" in content

    def test_missing_pyproj_is_reported_not_raised(self, processor, tmp_path, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fail_pyproj(name, *args, **kwargs):
            if name == "pyproj":
                raise ImportError("pyproj disabled for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_pyproj)
        messages = []
        processor._save_prj_file(str(tmp_path / "out.tif"), 32633, messages.append)
        assert any("Could not save PRJ file" in m for m in messages)
        assert not (tmp_path / "out.prj").exists()


class TestCollectFrameGeotiffs:
    @pytest.fixture
    def geotiff_folder(self, tmp_path):
        for name in ["00000001.tiff", "00000002.tif", "00000005.tiff",
                     "orthomosaic.tif", "IMG.TIF", "notes.txt"]:
            (tmp_path / name).write_bytes(b"")
        return str(tmp_path)

    def test_all_frames_sorted_and_non_frames_skipped(self, geotiff_folder):
        result = BambiProcessor._collect_frame_geotiffs(geotiff_folder)
        names = [os.path.basename(p) for p in result]
        assert names == ["00000001.tiff", "00000002.tif", "00000005.tiff"]

    def test_range_filter(self, geotiff_folder):
        result = BambiProcessor._collect_frame_geotiffs(
            geotiff_folder, use_all_frames=False, start_frame=2, end_frame=4)
        names = [os.path.basename(p) for p in result]
        assert names == ["00000002.tif"]

    def test_open_ended_range(self, geotiff_folder):
        result = BambiProcessor._collect_frame_geotiffs(
            geotiff_folder, use_all_frames=False, start_frame=2, end_frame=None)
        names = [os.path.basename(p) for p in result]
        assert names == ["00000002.tif", "00000005.tiff"]


class TestDefaultModelDir:
    def test_falls_back_to_tempdir_without_qgis(self):
        # In the test environment qgis.core has no QgsApplication, so the
        # helper must take its documented temp-dir fallback.
        path = BambiProcessor._get_default_model_dir()
        assert path.endswith(os.path.join("bambi_deps", "models"))
