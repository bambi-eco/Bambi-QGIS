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
    read_dem_origin_xy,
)


class TestReadDemOriginXy:
    def _write(self, path, payload):
        import json
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_reads_the_json_next_to_the_mesh(self, tmp_path):
        self._write(tmp_path / "dem.json", {"origin": [500000.0, 5000000.0, 12.0]})
        assert read_dem_origin_xy(str(tmp_path / "dem.gltf")) == (500000.0, 5000000.0)
        # .glb meshes resolve to the same sidecar convention
        assert read_dem_origin_xy(str(tmp_path / "dem.glb")) == (500000.0, 5000000.0)

    def test_explicit_metadata_path_wins(self, tmp_path):
        self._write(tmp_path / "dem.json", {"origin": [1.0, 2.0, 0.0]})
        explicit = self._write(tmp_path / "other.json", {"origin": [9.0, 8.0, 0.0]})
        assert read_dem_origin_xy(str(tmp_path / "dem.gltf"), explicit) == (9.0, 8.0)

    def test_falls_back_to_the_sidecar_when_the_explicit_path_is_unusable(
            self, tmp_path):
        self._write(tmp_path / "dem.json", {"origin": [1.0, 2.0, 0.0]})
        missing = str(tmp_path / "nope.json")
        assert read_dem_origin_xy(str(tmp_path / "dem.gltf"), missing) == (1.0, 2.0)

    def test_missing_or_malformed_metadata_yields_the_world_origin(self, tmp_path):
        assert read_dem_origin_xy("", "") == (0.0, 0.0)
        assert read_dem_origin_xy(str(tmp_path / "absent.gltf")) == (0.0, 0.0)

        bad = self._write(tmp_path / "bad.json", {"no_origin": True})
        assert read_dem_origin_xy("", bad) == (0.0, 0.0)

        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        assert read_dem_origin_xy("", str(tmp_path / "broken.json")) == (0.0, 0.0)

        short = self._write(tmp_path / "short.json", {"origin": [1.0]})
        assert read_dem_origin_xy("", short) == (0.0, 0.0)


class TestTheTextFilesAreNotRead:
    """The readers take a legacy path only to locate the project's store.

    Their store behaviour is covered in ``test_pipeline_outputs_store.py``;
    what matters here is that a populated text file next to no store yields
    nothing, rather than a reconstruction that looks like an answer.
    """

    def test_tracks_csv_is_not_parsed(self, tmp_path):
        f = tmp_path / "tracks_t" / "tracks.csv"
        f.parent.mkdir()
        f.write_text("0,7,100,200,5,101,201,6,0.9,1,0")
        assert load_geo_tracks_by_id(str(f)) == {}

    def test_fov_polygons_txt_is_not_parsed(self, tmp_path):
        f = tmp_path / "fov_t" / "fov_polygons.txt"
        f.parent.mkdir()
        f.write_text("0 3 10 20 400 30 20 401 30 40 402")
        assert load_fov_polygons_3d(str(f)) == {}

    def test_georeferenced_txt_is_not_parsed(self, tmp_path):
        f = tmp_path / "georeferenced_t" / "georeferenced.txt"
        f.parent.mkdir()
        f.write_text("0 0 100 200 5 101 201 6 0.9 1")
        assert load_georef_detections_by_frame(str(f)) == {}

    def test_a_missing_file_is_not_an_error_either(self, tmp_path):
        """Nothing to show is a normal state — a step that has not run."""
        assert load_geo_tracks_by_id(str(tmp_path / "nope.csv")) == {}
        assert load_fov_polygons_3d(str(tmp_path / "nope.txt")) == {}
        assert load_georef_detections_by_frame(str(tmp_path / "no.txt")) == {}

    def test_the_reason_is_logged_when_a_log_is_offered(self, tmp_path):
        logs = []
        load_geo_tracks_by_id(str(tmp_path / "tracks_t" / "tracks.csv"),
                              log_fn=logs.append)
        assert any("Migrate" in m for m in logs)


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
