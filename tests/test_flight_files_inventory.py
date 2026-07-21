# -*- coding: utf-8 -*-
"""Unit tests for core.flight_files and core.output_inventory."""
import sys

from bambi_wildlife_detection.core.flight_files import (
    detect_common_files,
    extract_embedded_srts,
)
from bambi_wildlife_detection.core.output_inventory import (
    FOLDER_STATUS_STEPS,
    PERPENDICULAR_STEPS,
    check_existing_outputs,
)
from tests.fakes import make_module

ALL_T = {key: "_t" for key in (
    "extract", "flight_route", "detection", "tracking", "fov",
    "alfs", "geotiff", "ortho", "sam3")}


class TestDetectCommonFiles:
    def test_detects_flight_folder_conventions(self, tmp_path):
        (tmp_path / "flight_log.csv").write_text("")
        (tmp_path / "dem.glb").write_bytes(b"")
        (tmp_path / "dem.json").write_text("{}")
        (tmp_path / "T_calib.json").write_text("{}")
        (tmp_path / "W_calib.json").write_text("{}")
        (tmp_path / "correction.json").write_text("{}")

        found = detect_common_files(str(tmp_path))
        assert found["airdata"].endswith("flight_log.csv")
        assert found["dem"].endswith("dem.glb")
        assert found["dem_metadata"].endswith("dem.json")
        assert found["thermal_calibration"].endswith("T_calib.json")
        assert found["rgb_calibration"].endswith("W_calib.json")
        assert found["correction"].endswith("correction.json")
        assert found["target_folder"].endswith("qgis")

    def test_dem_metadata_suffix_fallbacks(self, tmp_path):
        (tmp_path / "terrain.gltf").write_bytes(b"")
        (tmp_path / "terrain_mesh.json").write_text("{}")
        found = detect_common_files(str(tmp_path))
        assert found["dem_metadata"].endswith("terrain_mesh.json")

    def test_empty_folder_only_proposes_target(self, tmp_path):
        found = detect_common_files(str(tmp_path))
        assert set(found.keys()) == {"target_folder"}

    def test_unlistable_folder_logs_warning(self, tmp_path):
        logs = []
        found = detect_common_files(
            str(tmp_path / "does_not_exist"), log_fn=logs.append)
        assert any("Could not list folder contents" in m for m in logs)
        assert set(found.keys()) == {"target_folder"}


class TestExtractEmbeddedSrts:
    def test_missing_ffmpeg_logs_error(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
        logs = []
        result = extract_embedded_srts(
            ["video.mp4"], str(tmp_path), log_fn=logs.append)
        assert result == []
        assert any("imageio-ffmpeg not available" in m for m in logs)

    def test_cached_srt_reused_without_ffmpeg_run(self, tmp_path, monkeypatch):
        fake_ffmpeg = make_module(
            "imageio_ffmpeg", get_ffmpeg_exe=lambda: "/usr/bin/ffmpeg")
        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_ffmpeg)

        import subprocess

        def explode(*args, **kwargs):
            raise AssertionError("ffmpeg must not run for cached SRTs")

        monkeypatch.setattr(subprocess, "run", explode)
        (tmp_path / "DJI_0001_W.srt").write_text("1\ncontent\n")
        result = extract_embedded_srts(
            ["/videos/DJI_0001_W.MP4"], str(tmp_path))
        assert result == [str(tmp_path / "DJI_0001_W.srt")]

    def test_ffmpeg_invoked_and_output_collected(self, tmp_path, monkeypatch):
        fake_ffmpeg = make_module(
            "imageio_ffmpeg", get_ffmpeg_exe=lambda: "ffmpeg-exe")
        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_ffmpeg)

        import subprocess
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            # ffmpeg writes the requested output file
            with open(cmd[-1], "w") as fh:
                fh.write("1\n00:00:00 --> 00:00:01\ntext\n")

            class R:
                stderr = ""
            return R()

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = extract_embedded_srts(["/videos/clip.MP4"], str(tmp_path))
        assert len(result) == 1
        assert calls[0][0] == "ffmpeg-exe"
        assert "-map" in calls[0]

    def test_no_subtitle_stream_warns(self, tmp_path, monkeypatch):
        fake_ffmpeg = make_module(
            "imageio_ffmpeg", get_ffmpeg_exe=lambda: "ffmpeg-exe")
        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_ffmpeg)

        import subprocess

        def fake_run(cmd, **kwargs):
            class R:
                stderr = "Stream map '0:s:0' matches no streams."
            return R()

        monkeypatch.setattr(subprocess, "run", fake_run)
        logs = []
        result = extract_embedded_srts(
            ["/videos/clip.MP4"], str(tmp_path), log_fn=logs.append)
        assert result == []
        assert any("No subtitle stream found" in m for m in logs)


class TestCheckExistingOutputs:
    def _make(self, tmp_path, *parts, content=b"x"):
        path = tmp_path.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def test_empty_or_missing_folder(self, tmp_path):
        assert check_existing_outputs(str(tmp_path / "nope"), ALL_T) == []
        assert check_existing_outputs("", ALL_T) == []
        assert check_existing_outputs(str(tmp_path), ALL_T) == []

    def test_frame_extraction_requires_frames_and_poses(self, tmp_path):
        self._make(tmp_path, "frames_t", "f0.jpg")
        assert check_existing_outputs(str(tmp_path), ALL_T) == []
        self._make(tmp_path, "poses_t.json")
        assert check_existing_outputs(str(tmp_path), ALL_T) == [
            "extract_thermal_frames"]

    def test_rgb_extraction_key_follows_camera(self, tmp_path):
        self._make(tmp_path, "frames_w", "f0.jpg")
        self._make(tmp_path, "poses_w.json")
        cameras = dict(ALL_T, extract="_w")
        assert check_existing_outputs(str(tmp_path), cameras) == [
            "extract_rgb_frames"]

    def test_folder_steps_respect_camera_suffix(self, tmp_path):
        self._make(tmp_path, "detections_t", "detections.txt")
        self._make(tmp_path, "tracks_w", "tracks.csv")
        completed = check_existing_outputs(str(tmp_path), ALL_T)
        assert completed == ["detection"]        # tracks_w ignored for "_t"
        completed = check_existing_outputs(
            str(tmp_path), dict(ALL_T, tracking="_w"))
        assert set(completed) == {"detection", "tracking"}

    def test_sam3_steps_need_marker_files(self, tmp_path):
        self._make(tmp_path, "segmentation_t", "masks", "m0.png")
        assert "sam3_segmentation" not in check_existing_outputs(
            str(tmp_path), ALL_T)
        self._make(tmp_path, "segmentation_t", "segmentation_pixel.json")
        completed = check_existing_outputs(str(tmp_path), ALL_T)
        assert "sam3_segmentation" in completed
        assert "sam3_georeference" not in completed
        self._make(tmp_path, "segmentation_georef.json")   # in target root
        assert "sam3_georeference" in check_existing_outputs(
            str(tmp_path), ALL_T)

    def test_perpendicular_files(self, tmp_path):
        self._make(tmp_path, "flight_route_t", "perpendicular_t.json")
        self._make(tmp_path, "flight_route_t", "perpendicular_tracks_w.json")
        completed = check_existing_outputs(
            str(tmp_path), dict(ALL_T, tracking="_w"))
        assert "perpendicular" in completed
        assert "track_perpendicular" in completed

    def test_step_constants_cover_all_status_keys(self):
        # every key the check can return must be resettable by the dialog
        resettable = set(FOLDER_STATUS_STEPS) | set(PERPENDICULAR_STEPS)
        possible = {
            "extract_thermal_frames", "extract_rgb_frames", "flight_route",
            "detection", "georeference", "tracking", "calculate_fov", "alfs",
            "export_geotiffs", "orthomosaic", "sam3_segmentation",
            "sam3_georeference", "trex_import", "perpendicular",
            "track_perpendicular"}
        assert possible <= resettable
