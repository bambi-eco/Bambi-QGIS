# -*- coding: utf-8 -*-
"""Unit tests for core.pipeline_outputs (the dock widget's layer-builder
readers) and the config-convention correction resolver."""
import pytest

from bambi_wildlife_detection.core.corrections import (
    correction_for_frame,
    correction_for_frame_config,
    read_correction,
)
from bambi_wildlife_detection.core.pipeline_outputs import (
    load_fov_polygons_3d,
    load_geo_tracks_by_id,
    load_georef_detections_by_frame,
)


class TestLoadGeoTracksById:
    def test_parses_and_groups_by_track(self, tmp_path):
        f = tmp_path / "tracks.csv"
        f.write_text(
            "# header\n"
            "0,7,100,200,5,101,201,6,0.9,1,0\n"
            "1,7,102,202,5,103,203,6,0.8,1,1\n"
            "0,8,110,210,5,111,211,6,0.7,2\n"
            "garbage,row\n")
        tracks = load_geo_tracks_by_id(str(f))
        assert sorted(tracks.keys()) == [7, 8]
        assert tracks[7][0]["z1"] == 5.0
        assert tracks[7][1]["interpolated"] == 1
        assert tracks[8][0]["interpolated"] == 0   # column absent -> 0
        assert tracks[8][0]["class_id"] == 2

    def test_missing_file_logs_and_returns_empty(self, tmp_path):
        logs = []
        tracks = load_geo_tracks_by_id(
            str(tmp_path / "nope.csv"), log_fn=logs.append)
        assert tracks == {}
        assert any("Error reading" in m for m in logs)


class TestLoadFovPolygons3d:
    def test_parses_xyz_triplets(self, tmp_path):
        f = tmp_path / "fov_polygons.txt"
        f.write_text(
            "# header\n"
            "0 3 10 20 400 30 20 401 30 40 402\n"
            "1 0\n"                                   # zero points -> skipped
            "2 2 1 2 3 4 5 6\n")
        polygons = load_fov_polygons_3d(str(f))
        assert polygons[0] == [(10.0, 20.0, 400.0), (30.0, 20.0, 401.0),
                               (30.0, 40.0, 402.0)]
        assert 1 not in polygons
        assert polygons[2] == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]

    def test_missing_file_logs(self, tmp_path):
        logs = []
        assert load_fov_polygons_3d(
            str(tmp_path / "nope.txt"), log_fn=logs.append) == {}
        assert any("Error reading FoV file" in m for m in logs)


class TestLoadGeorefDetectionsByFrame:
    def test_groups_by_frame_and_filters_invalid(self, tmp_path):
        f = tmp_path / "georeferenced.txt"
        f.write_text(
            "# header\n"
            "0 0 100 200 5 101 201 6 0.9 1\n"
            "1 0 -1 210 5 111 211 6 0.8 1\n"      # negative x1 -> skipped
            "2 3 120 220 5 121 221 6 0.7 2\n")
        dets = load_georef_detections_by_frame(str(f))
        assert sorted(dets.keys()) == [0, 3]
        assert len(dets[0]) == 1
        assert dets[0][0]["idx"] == 0
        assert dets[3][0]["class_id"] == 2

    def test_missing_file_logs(self, tmp_path):
        logs = []
        assert load_georef_detections_by_frame(
            str(tmp_path / "nope.txt"), log_fn=logs.append) == {}
        assert any("Error reading georeferenced file" in m for m in logs)


class TestCorrectionConventions:
    """The raw-file and pipeline-config conventions must resolve the same
    frame ranges to the same values."""

    RAW = {
        "translation": {"x": 0.0, "y": 0.0, "z": -2.825},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.011},
        "additional": [{
            "start": 100, "end": 200,
            "translation": {"x": 1.0, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.5},
        }],
    }

    @property
    def config(self):
        cfg = dict(self.RAW)
        cfg["additional_corrections"] = cfg.pop("additional")
        return cfg

    @pytest.mark.parametrize("frame,expect_ranged", [
        (0, False), (99, False), (100, True), (150, True),
        (200, True), (201, False),
    ])
    def test_conventions_agree(self, frame, expect_ranged):
        t, r = correction_for_frame(frame, self.RAW)
        combined = correction_for_frame_config(frame, self.config)
        assert combined["translation"] == t
        assert combined["rotation"] == r
        assert (t["x"] == 1.0) is expect_ranged

    def test_read_correction_prefers_explicit_path(self, tmp_path):
        explicit = tmp_path / "explicit.json"
        explicit.write_text('{"translation": {"x": 9}}')
        (tmp_path / "correction.json").write_text('{"translation": {"x": 1}}')
        assert read_correction(str(tmp_path), str(explicit))["translation"]["x"] == 9
        assert read_correction(str(tmp_path), "")["translation"]["x"] == 1
        assert read_correction(str(tmp_path / "empty"), "") == {}
