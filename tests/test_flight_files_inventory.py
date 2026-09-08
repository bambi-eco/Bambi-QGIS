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
            "track_perpendicular", "track_matching"}
        assert possible <= resettable


class TestStoreBackedOutputs:
    """A stage counts as done from its 6.0 store file too.

    Without this, a project running with legacy text output switched off would
    report every stage as never started (EXCHANGE_FORMAT_PLAN.md §7).
    """

    def test_store_file_alone_marks_the_stage_complete(self, tmp_path):
        from bambi_wildlife_detection.core import store

        root = str(tmp_path)
        path = store.stage_path(root, store.DETECTIONS, "t")
        store.open_store(path, store.DETECTIONS, "t").close()

        assert "detection" in check_existing_outputs(root, ALL_T)

    def test_legacy_folder_alone_still_works(self, tmp_path):
        import os

        root = str(tmp_path)
        folder = os.path.join(root, "detections_t")
        os.makedirs(folder)
        with open(os.path.join(folder, "detections.txt"), "w") as fh:
            fh.write("x")

        assert "detection" in check_existing_outputs(root, ALL_T)

    def test_neither_means_not_started(self, tmp_path):
        assert "detection" not in check_existing_outputs(str(tmp_path), ALL_T)

    def test_store_file_for_the_other_modality_does_not_count(self, tmp_path):
        from bambi_wildlife_detection.core import store

        root = str(tmp_path)
        store.open_store(
            store.stage_path(root, store.DETECTIONS, "w"),
            store.DETECTIONS, "w").close()

        assert "detection" not in check_existing_outputs(root, ALL_T)


class TestCrossModalOutputs:
    """Track matching is recorded against both cameras at once, in a store
    beside project.gpkg - which the per-camera scan used to walk straight
    past, so a reopened project showed the step as never run."""

    def test_matches_store_marks_track_matching_complete(self, tmp_path):
        from bambi_wildlife_detection.core import store

        root = str(tmp_path)
        store.open_store(store.matches_path(root), store.MATCHES).close()

        assert "track_matching" in check_existing_outputs(root, ALL_T)
        # ... whichever camera any step is set to.
        all_w = {key: "_w" for key in ALL_T}
        assert "track_matching" in check_existing_outputs(root, all_w)

    def test_without_the_store_it_is_not_started(self, tmp_path):
        assert "track_matching" not in check_existing_outputs(str(tmp_path), ALL_T)


class TestClassificationStates:
    """Classification steps are store-backed and per modality; the folder
    scan never saw them, so a reopened project showed them as never run
    (2026-09-07). They also know when they are out of date."""

    @staticmethod
    def _project(root):
        from bambi_wildlife_detection.core import (classification_store, stages,
                                                   store)
        store.open_store(store.project_path(root), store.PROJECT).close()
        for modality in ("t", "w"):
            classification_store.start_embedding_run(
                root, modality, backbone="b", dim=4, crop_size=8,
                padding=0.0, projection="non_geo", thermal_anchored=False,
                folder=f"embeddings_{modality}/non_geo", plugin_version="x")
            stages.mark_complete(root, "embeddings", modality)
        classification_store.record_frame_predictions(root, "t", "occlusion", [
            {"detection_id": 1, "label": "clear", "class_index": 0, "prob": 0.9}])
        classification_store.record_frame_predictions(root, "w", "occlusion", [
            {"detection_id": 1, "label": "clear", "class_index": 0, "prob": 0.9}])
        classification_store.record_track_predictions(root, "t", "species", [
            {"track_id": 1, "label": "red_deer", "votes": 1, "n": 1, "fraction": 1.0}])
        stages.mark_complete(root, "classification", "t")
        stages.mark_complete(root, "classification", "w")
        return stages

    def test_reads_the_configured_modalities(self, tmp_path):
        from bambi_wildlife_detection.core.output_inventory import classification_states
        root = str(tmp_path)
        self._project(root)
        states = classification_states(root, ("t", "w"), {
            "occlusion": ("t", "w"), "species": ("t",), "sex": ("t",),
            "life_stage": ("t",)})
        assert states["embeddings"] == "complete"
        assert states["classify_occlusion"] == "complete"
        assert states["classify_species"] == "complete"
        assert states["classify_sex"] == "pending"
        assert states["life_stage"] == "pending"

    def test_a_task_missing_on_one_configured_modality_is_pending(self, tmp_path):
        from bambi_wildlife_detection.core.output_inventory import classification_states
        root = str(tmp_path)
        self._project(root)
        states = classification_states(root, ("w",), {"species": ("t", "w")})
        assert states["classify_species"] == "pending"     # w has no species
        assert states["embeddings"] == "complete"

    def test_stale_shows_through(self, tmp_path):
        from bambi_wildlife_detection.core.output_inventory import classification_states
        root = str(tmp_path)
        stages = self._project(root)
        stages.mark_dependents_stale(root, "tracking", "t")   # re-ran tracking
        states = classification_states(root, ("t", "w"), {"occlusion": ("t", "w")})
        assert states["embeddings"] == "stale"
        assert states["classify_occlusion"] == "stale"
        states = classification_states(root, ("w",), {"occlusion": ("w",)})
        assert states["embeddings"] == "complete"
        assert states["classify_occlusion"] == "complete"

    def test_empty_project_is_all_pending(self, tmp_path):
        from bambi_wildlife_detection.core.output_inventory import classification_states
        states = classification_states(str(tmp_path), ("t",), {"occlusion": ("t",)})
        assert states == {"embeddings": "pending", "classify_occlusion": "pending",
                          "classify_species": "pending", "classify_sex": "pending",
                          "life_stage": "pending"}


class TestCrossModalStates:
    def test_reads_the_match_stage_state(self, tmp_path):
        from bambi_wildlife_detection.core import stages, store
        from bambi_wildlife_detection.core.output_inventory import cross_modal_states
        root = str(tmp_path)
        store.open_store(store.project_path(root), store.PROJECT).close()
        assert cross_modal_states(root) == {"track_matching": "pending"}
        store.open_store(store.matches_path(root), store.MATCHES).close()
        assert cross_modal_states(root) == {"track_matching": "complete"}
        stages.mark_complete(root, "track_matching", stages.CROSS_MODAL)
        stages.mark_complete(root, "tracking", "w")
        stages.mark_complete(root, "tracking", "w")        # a re-run
        assert cross_modal_states(root) == {"track_matching": "stale"}
