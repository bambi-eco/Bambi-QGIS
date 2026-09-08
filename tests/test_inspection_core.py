# -*- coding: utf-8 -*-
"""Unit tests for core.inspection (click-tool loaders, frame assembly and
map-point projection). The alfspy/trimesh stack is faked; the projection
happy path uses the identity camera from tests.fakes."""
import json
import sys

import pytest

from bambi_wildlife_detection.core import (
    detection_store, inspection, track_store)
from tests.fakes import install_fake_render_stack, make_module


def _detected(root, tracked=False):
    """Two detections in the store, optionally grouped into one track."""
    detection_store.record_detections(root, "t", [
        {"frame": 0, "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0,
         "confidence": 0.9, "source_class": "0"},
        {"frame": 1, "x1": 5.0, "y1": 6.0, "x2": 7.0, "y2": 8.0,
         "confidence": 0.8, "source_class": "0"},
    ])
    ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
    if tracked:
        track_store.record_tracks(root, "t", [
            {"track_id": 7, "detection_id": ids[0]},
            {"track_id": 7, "detection_id": ids[1]},
        ])
    return ids


IMG_W, IMG_H = 640, 512
ORIGIN = (1000.0, 2000.0, 300.0)


class TestSimpleLoaders:
    def test_load_pixel_detections(self, tmp_path):
        root = str(tmp_path)
        _detected(root)
        result = inspection.load_pixel_detections(root, "t")
        assert [d["frame"] for d in result] == [0, 1]
        assert result[0]["x1"] == 1.0
        # The detection_id is what the layer was built from, so the clicked
        # box needs no matching back by confidence (§1.2a).
        assert result[0]["detection_id"]

    def test_pixel_detections_ignore_the_text_file(self, tmp_path):
        det = tmp_path / "detections_t"
        det.mkdir()
        (det / "detections.txt").write_text("0 1 2 3 4 0.9 2")
        assert inspection.load_pixel_detections(str(tmp_path), "t") == []

    def test_pixel_detections_for_an_unknown_modality(self, tmp_path):
        assert inspection.load_pixel_detections(str(tmp_path), "x") == []

    def test_load_pixel_tracks(self, tmp_path):
        root = str(tmp_path)
        _detected(root, tracked=True)
        result = inspection.load_pixel_tracks(root, "t")
        assert len(result) == 1
        boxes = list(result.values())[0]
        assert [b["frame"] for b in boxes] == [0, 1]
        assert boxes[0]["conf"] == 0.9

    def test_pixel_tracks_ignore_the_csv(self, tmp_path):
        trk = tmp_path / "tracks_t"
        trk.mkdir()
        (trk / "tracks_pixel.csv").write_text("0,7,1,2,3,4,0.9,1")
        assert inspection.load_pixel_tracks(str(tmp_path), "t") == {}

    def test_resolve_image_paths(self, tmp_path):
        (tmp_path / "frames_t").mkdir()
        (tmp_path / "frames_t" / "f0.jpg").write_bytes(b"")
        (tmp_path / "poses_t.json").write_text(json.dumps(
            {"images": [{"imagefile": "f0.jpg"}]}))
        path_t, path_w = inspection.resolve_image_paths(str(tmp_path), 0)
        assert path_t.endswith("f0.jpg")
        assert path_w == ""      # no RGB poses at all
        assert inspection.resolve_image_paths(str(tmp_path), 5) == ("", "")

    def test_find_dem_mesh_path(self, tmp_path):
        glb = tmp_path / "dem.glb"
        glb.write_bytes(b"")
        assert inspection.find_dem_mesh_path(str(glb), None) == str(glb)
        # derived from the metadata JSON path
        assert inspection.find_dem_mesh_path(
            "", str(tmp_path / "dem.json")) == str(glb)
        assert inspection.find_dem_mesh_path("", None) is None


class TestFillInterpolatedBoxes:
    def _frame(self, idx, boxes):
        return {"frame_idx": idx, "boxes_green": boxes, "boxes_blue": []}

    def test_interpolates_between_neighbours(self):
        frames = [
            self._frame(0, [(0, 0, 10, 10, 0.9, 1, 0)]),
            self._frame(5, []),
            self._frame(10, [(100, 0, 110, 10, 0.9, 1, 0)]),
        ]
        inspection.fill_interpolated_boxes(frames)
        box = frames[1]["boxes_green"][0]
        assert box[:4] == (50.0, 0.0, 60.0, 10.0)
        assert box[6] == 1   # flagged as interpolated

    def test_single_sided_copies_nearest(self):
        frames = [
            self._frame(0, [(0, 0, 10, 10, 0.7, 2, 0)]),
            self._frame(1, []),
        ]
        inspection.fill_interpolated_boxes(frames)
        box = frames[1]["boxes_green"][0]
        assert box[:4] == (0.0, 0.0, 10.0, 10.0)
        assert (box[4], box[5], box[6]) == (0.7, 2, 1)

    def test_no_reference_boxes_leaves_empty(self):
        frames = [self._frame(0, []), self._frame(1, [])]
        inspection.fill_interpolated_boxes(frames)
        assert frames[0]["boxes_green"] == []


class TestBuildFrames:
    def test_from_pixel_tracks(self, tmp_path):
        track_dets = [{"frame": 3, "x1": 1, "y1": 2, "x2": 3, "y2": 4,
                       "conf": 0.9, "cls": 1, "interpolated": 1}]
        all_tracks = {
            5: track_dets,
            6: [{"frame": 3, "x1": 9, "y1": 9, "x2": 10, "y2": 10,
                 "conf": 0.5, "cls": 2}],
        }
        frames = inspection.build_frames_from_pixel_tracks(
            track_dets, all_tracks, 5, str(tmp_path), "t")
        assert len(frames) == 1
        assert frames[0]["boxes_green"] == [(1, 2, 3, 4, 0.9, 1, 1)]
        assert frames[0]["boxes_blue"] == [(9, 9, 10, 10, 0.5, 2)]
        assert frames[0]["boxes_modality"] == "t"


@pytest.fixture
def fake_mesh_stack(monkeypatch):
    """alfspy.render.read_gltf + trimesh returning a controllable ray-cast."""
    class _MeshData:
        vertices = [(0, 0, 0)]
        indices = [(0, 0, 0)]

    calls = {"read_gltf": 0}

    render_render = make_module("alfspy.render.render")

    def read_gltf(path):
        calls["read_gltf"] += 1
        return _MeshData(), None

    render_render.read_gltf = read_gltf

    class _Ray:
        def intersects_location(self, origins, directions):
            import numpy as np
            x, y = origins[0][0], origins[0][1]
            return np.array([[x, y, 450.0], [x, y, 430.0]]), None, None

    class Trimesh:
        def __init__(self, vertices=None, faces=None):
            self.ray = _Ray()

    alfspy = make_module("alfspy")
    alfspy.__path__ = []
    render_pkg = make_module("alfspy.render")
    render_pkg.__path__ = []
    render_pkg.render = render_render
    alfspy.render = render_pkg
    monkeypatch.setitem(sys.modules, "alfspy", alfspy)
    monkeypatch.setitem(sys.modules, "alfspy.render", render_pkg)
    monkeypatch.setitem(sys.modules, "alfspy.render.render", render_render)
    monkeypatch.setitem(
        sys.modules, "trimesh", make_module("trimesh", Trimesh=Trimesh))
    return calls


class TestRayCastDemZ:
    def test_returns_highest_intersection(self, fake_mesh_stack, tmp_path):
        mesh = tmp_path / "dem.glb"
        mesh.write_bytes(b"")
        cache = {}
        z = inspection.ray_cast_dem_z((10.0, 20.0), str(mesh), cache)
        assert z == 450.0

    def test_mesh_cached_between_calls(self, fake_mesh_stack, tmp_path):
        mesh = tmp_path / "dem.glb"
        mesh.write_bytes(b"")
        cache = {}
        inspection.ray_cast_dem_z((0, 0), str(mesh), cache)
        inspection.ray_cast_dem_z((1, 1), str(mesh), cache)
        assert fake_mesh_stack["read_gltf"] == 1

    def test_no_mesh_path(self):
        assert inspection.ray_cast_dem_z((0, 0), None, {}) is None


class TestProjectMapPoint:
    def _setup_folder(self, tmp_path):
        (tmp_path / "dem.json").write_text(json.dumps({"origin": list(ORIGIN)}))
        poses = {"images": [{
            "imagefile": "f0.jpg",
            "location": [0.0, 0.0, 10.0],
            "rotation": [0.0, 0.0, 0.0],
            "fovy": [50.0],
        }]}
        (tmp_path / "poses_w.json").write_text(json.dumps(poses))
        return str(tmp_path)

    def test_projects_click_into_frame(self, tmp_path, monkeypatch):
        install_fake_render_stack(monkeypatch)
        folder = self._setup_folder(tmp_path)
        logs = []

        # No mesh on disk -> ray-cast unavailable -> injected sampler is used
        result = inspection.project_map_point(
            (ORIGIN[0] - 0.5, ORIGIN[1] - 0.25), 0, folder,
            str(tmp_path / "dem.glb"), "", "w", mesh_cache={},
            sample_elevation=lambda xy, origin, dem_json: 0.0,
            log_fn=lambda msg, level="info": logs.append((level, msg)),
        )
        # Local point (-0.5, -0.25, 0) through the identity camera
        assert result == (pytest.approx(160.0), pytest.approx(320.0))
        assert any("Projected pixel" in msg for _, msg in logs)
        assert all(level == "info" for level, _ in logs)

    def test_out_of_frame_returns_none(self, tmp_path, monkeypatch):
        install_fake_render_stack(monkeypatch)
        folder = self._setup_folder(tmp_path)
        result = inspection.project_map_point(
            (ORIGIN[0] + 50.0, ORIGIN[1]), 0, folder, "", "", "w",
            mesh_cache={}, sample_elevation=lambda *a: 0.0)
        assert result is None

    def test_missing_poses_logs_warning(self, tmp_path, monkeypatch):
        install_fake_render_stack(monkeypatch)
        logs = []
        result = inspection.project_map_point(
            (0, 0), 0, str(tmp_path), "", "", "t", mesh_cache={},
            log_fn=lambda msg, level="info": logs.append((level, msg)))
        assert result is None
        assert logs and logs[0][0] == "warning"
        assert "poses file not found" in logs[0][1]

    def test_frame_index_out_of_range(self, tmp_path, monkeypatch):
        install_fake_render_stack(monkeypatch)
        folder = self._setup_folder(tmp_path)
        result = inspection.project_map_point(
            (0, 0), 7, folder, "", "", "w", mesh_cache={})
        assert result is None

    def test_missing_render_stack_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyrr", None)
        folder = self._setup_folder(tmp_path)
        result = inspection.project_map_point(
            (0, 0), 0, folder, "", "", "w", mesh_cache={})
        assert result is None


class TestPartnerFrames:
    """The other camera's frame for a moment is found on the shared capture
    clock, not by index: the cameras run at different rates, so the same
    index is a different moment (and, for the shorter stream, may not exist)."""

    @staticmethod
    def _poses(root, modality, stamps, with_files=True):
        images = []
        for i, stamp in enumerate(stamps):
            name = f"{modality}{i}.jpg"
            images.append({"imagefile": name, "timestamp": stamp})
            if with_files:
                (root / f"frames_{modality}").mkdir(exist_ok=True)
                (root / f"frames_{modality}" / name).write_bytes(b"")
        (root / f"poses_{modality}.json").write_text(json.dumps({"images": images}))

    def test_partner_is_the_frame_taken_at_the_same_moment(self, tmp_path):
        # Thermal at 10 Hz, RGB at 5 Hz starting half a second later.
        self._poses(tmp_path, "t", [f"2026-06-26T08:03:{32 + i // 10:02d}.{i % 10}00000+00:00"
                                    for i in range(20)])
        self._poses(tmp_path, "w", [f"2026-06-26T08:03:{32 + (5 + 2 * i) // 10:02d}.{(5 + 2 * i) % 10}00000+00:00"
                                    for i in range(6)])
        assert inspection.partner_frame(str(tmp_path), 5, "t") == 0     # 32.5 s
        assert inspection.partner_frame(str(tmp_path), 9, "t") == 2     # 32.9 s -> 32.9 s
        assert inspection.partner_frame(str(tmp_path), 0, "w") == 5     # back the other way
        found = inspection.frame_images(str(tmp_path), 9, "t")
        assert found["frame_idx_t"] == 9 and found["frame_idx_w"] == 2
        assert found["image_path_t"].endswith("t9.jpg")
        assert found["image_path_w"].endswith("w2.jpg")

    def test_resolve_image_paths_uses_the_partner(self, tmp_path):
        self._poses(tmp_path, "t", [f"2026-06-26T08:03:32.{i}00000+00:00" for i in range(10)])
        self._poses(tmp_path, "w", ["2026-06-26T08:03:32.900000+00:00"])
        path_t, path_w = inspection.resolve_image_paths(str(tmp_path), 9, "t")
        assert path_t.endswith("t9.jpg")
        assert path_w.endswith("w0.jpg")           # index 9 does not exist on RGB
        path_t, path_w = inspection.resolve_image_paths(str(tmp_path), 0, "w")
        assert path_w.endswith("w0.jpg") and path_t.endswith("t9.jpg")

    def test_without_timestamps_the_same_index_is_kept(self, tmp_path):
        self._poses(tmp_path, "t", [""] * 3)
        self._poses(tmp_path, "w", [""] * 3)
        assert inspection.partner_frame(str(tmp_path), 1, "t") == 1
        assert inspection.partner_frame(str(tmp_path), 5, "t") is None

    def test_no_other_camera_means_no_partner(self, tmp_path):
        self._poses(tmp_path, "t", ["2026-06-26T08:03:32+00:00"])
        assert inspection.partner_frame(str(tmp_path), 0, "t") is None
        found = inspection.frame_images(str(tmp_path), 0, "t")
        assert found["frame_idx_w"] is None and found["image_path_w"] == ""

    def test_track_frames_carry_both_indices(self, tmp_path):
        self._poses(tmp_path, "t", [f"2026-06-26T08:03:32.{i}00000+00:00" for i in range(10)])
        self._poses(tmp_path, "w", ["2026-06-26T08:03:32.300000+00:00"])
        dets = [{"frame": 3, "x1": 1, "y1": 2, "x2": 3, "y2": 4, "conf": 0.9, "cls": 0}]
        frames = inspection.build_frames_from_pixel_tracks(dets, {1: dets}, 1, str(tmp_path), "t")
        assert frames[0]["frame_idx_t"] == 3 and frames[0]["frame_idx_w"] == 0


class TestOtherCameraBoxes:
    """On the other camera the viewer showed only projected boxes, never
    what that camera itself detected at the same moment (2026-09-07)."""

    @staticmethod
    def _flight(root):
        from bambi_wildlife_detection.core import match_store
        detection_store.record_detections(root, "t", [
            {"frame": 3, "x1": 1, "y1": 1, "x2": 2, "y2": 2, "confidence": 0.9, "source_class": "0"},
            {"frame": 3, "x1": 5, "y1": 5, "x2": 6, "y2": 6, "confidence": 0.8, "source_class": "0"},
        ])
        detection_store.record_detections(root, "w", [
            {"frame": 0, "x1": 10, "y1": 10, "x2": 20, "y2": 20, "confidence": 0.7, "source_class": "animal"},
            {"frame": 0, "x1": 50, "y1": 50, "x2": 60, "y2": 60, "confidence": 0.6, "source_class": "animal"},
            {"frame": 1, "x1": 90, "y1": 90, "x2": 99, "y2": 99, "confidence": 0.5, "source_class": "animal"},
        ])
        ids_t = [d["detection_id"] for d in track_store.load_detections(root, "t")]
        ids_w = [d["detection_id"] for d in track_store.load_detections(root, "w")]
        match_store.record_matches(root, [{
            "track_id_t": 1, "track_id_w": 1, "shared": 1, "median_dist": 0.3,
            "conf_t": 0.9, "conf_w": 0.7,
            "pairs": [{"frame_t": 3, "frame_w": 0, "detection_id_t": ids_t[0],
                       "detection_id_w": ids_w[0], "dist": 0.3}],
        }])
        return ids_t, ids_w

    def test_matched_partner_is_green_and_the_rest_blue(self, tmp_path):
        root = str(tmp_path)
        ids_t, _ids_w = self._flight(root)
        green, blue = inspection.other_camera_boxes(root, "t", 0, [ids_t[0]])
        assert [b[:4] for b in green] == [(10, 10, 20, 20)]
        assert [b[:4] for b in blue] == [(50, 50, 60, 60)]
        assert green[0][4] == 0.7 and green[0][5] == 0      # conf, species_id

    def test_only_that_frame_of_the_other_camera(self, tmp_path):
        root = str(tmp_path)
        self._flight(root)
        green, blue = inspection.other_camera_boxes(root, "t", 1, [])
        assert green == [] and [b[:4] for b in blue] == [(90, 90, 99, 99)]

    def test_unhighlighted_or_unmatched_is_all_blue(self, tmp_path):
        root = str(tmp_path)
        ids_t, _ = self._flight(root)
        _, blue = inspection.other_camera_boxes(root, "t", 0, [])
        assert len(blue) == 2
        green, blue = inspection.other_camera_boxes(root, "t", 0, [ids_t[1]])
        assert green == [] and len(blue) == 2

    def test_works_from_the_rgb_side_too(self, tmp_path):
        root = str(tmp_path)
        _, ids_w = self._flight(root)
        green, blue = inspection.other_camera_boxes(root, "w", 3, [ids_w[0]])
        assert [b[:4] for b in green] == [(1, 1, 2, 2)]
        assert [b[:4] for b in blue] == [(5, 5, 6, 6)]

    def test_no_partner_frame_means_no_boxes(self, tmp_path):
        root = str(tmp_path)
        self._flight(root)
        assert inspection.other_camera_boxes(root, "t", None, [1]) == ([], [])

    def test_without_a_match_run_everything_is_blue(self, tmp_path):
        root = str(tmp_path)
        detection_store.record_detections(root, "w", [
            {"frame": 0, "x1": 10, "y1": 10, "x2": 20, "y2": 20, "confidence": 0.7, "source_class": "animal"}])
        green, blue = inspection.other_camera_boxes(root, "t", 0, [1, 2, 3])
        assert green == [] and len(blue) == 1

    def test_track_frames_carry_the_other_camera(self, tmp_path):
        root = str(tmp_path)
        ids_t, _ = self._flight(root)
        (tmp_path / "poses_t.json").write_text(json.dumps({"images": [
            {"imagefile": f"t{i}.jpg", "timestamp": f"2026-06-26T08:03:32.{i}00000+00:00"}
            for i in range(5)]}))
        (tmp_path / "poses_w.json").write_text(json.dumps({"images": [
            {"imagefile": "w0.jpg", "timestamp": "2026-06-26T08:03:32.300000+00:00"}]}))
        dets = [{"detection_id": ids_t[0], "frame": 3, "x1": 1, "y1": 1, "x2": 2, "y2": 2,
                 "conf": 0.9, "cls": 0}]
        frames = inspection.build_frames_from_pixel_tracks(dets, {1: dets}, 1, root, "t")
        assert frames[0]["frame_idx_w"] == 0
        assert [b[:4] for b in frames[0]["boxes_green_other"]] == [(10, 10, 20, 20)]
        assert len(frames[0]["boxes_blue_other"]) == 1
