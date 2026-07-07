# -*- coding: utf-8 -*-
"""Unit tests for core.inspection (click-tool loaders, frame assembly and
map-point projection). The alfspy/trimesh stack is faked; the projection
happy path uses the identity camera from tests.fakes."""
import json
import sys

import pytest

from bambi_wildlife_detection.core import inspection
from tests.fakes import install_fake_render_stack, make_module

IMG_W, IMG_H = 640, 512
ORIGIN = (1000.0, 2000.0, 300.0)


class TestLoadGeorefTrackDets:
    def _write(self, tmp_path, fname, lines):
        folder = tmp_path / "tracks_t"
        folder.mkdir(exist_ok=True)
        (folder / fname).write_text("\n".join(lines) + "\n")

    def test_filters_by_track_id_and_sorts(self, tmp_path):
        self._write(tmp_path, "tracks.csv", [
            "# header",
            "5,2,0,0,0,1,1,1,0.9,1,0",
            "3,2,0,0,0,1,1,1,0.8,1,1",
            "4,9,0,0,0,1,1,1,0.7,2,0",
        ])
        result = inspection.load_georef_track_dets(str(tmp_path), 2, "t")
        assert [d["frame"] for d in result] == [3, 5]
        assert result[0]["interpolated"] == 1
        assert result[1]["confidence"] == 0.9

    def test_skips_pixel_csvs(self, tmp_path):
        self._write(tmp_path, "tracks_pixel.csv", ["0,2,0,0,0,1,1,1,0.9,1"])
        assert inspection.load_georef_track_dets(str(tmp_path), 2, "t") == []

    def test_missing_folder(self, tmp_path):
        assert inspection.load_georef_track_dets(str(tmp_path), 1, "w") == []


class TestSimpleLoaders:
    def test_load_pixel_detections(self, tmp_path):
        f = tmp_path / "detections.txt"
        f.write_text("# h\n0 1 2 3 4 0.9 2\n1 5 6 7 8 0.8\n")
        result = inspection.load_pixel_detections(str(f))
        assert result[0]["class_id"] == 2
        assert result[1]["class_id"] == 0    # missing class defaults to 0
        assert inspection.load_pixel_detections(str(tmp_path / "x")) == []

    def test_load_pixel_tracks(self, tmp_path):
        f = tmp_path / "tracks_pixel.csv"
        f.write_text("# h\n0,7,1,2,3,4,0.9,1\n1,7,5,6,7,8,0.8,1\n")
        result = inspection.load_pixel_tracks(str(f))
        assert len(result[7]) == 2
        assert result[7][0]["conf"] == 0.9

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

    def test_from_georef_matches_by_confidence_and_class(self, tmp_path):
        georef_dets = [
            {"frame": 0, "confidence": 0.9, "class_id": 1, "interpolated": 0},
            {"frame": 1, "confidence": 0.9, "class_id": 1, "interpolated": 1},
        ]
        pixel_dets = [
            {"frame": 0, "x1": 1, "y1": 1, "x2": 2, "y2": 2,
             "confidence": 0.9001, "class_id": 1},
            {"frame": 0, "x1": 5, "y1": 5, "x2": 6, "y2": 6,
             "confidence": 0.5, "class_id": 1},
        ]
        frames = inspection.build_frames_from_georef(
            georef_dets, pixel_dets, str(tmp_path), "w")
        # frame 0: matched within confidence tolerance, other box is blue
        assert frames[0]["boxes_green"][0][:4] == (1, 1, 2, 2)
        assert frames[0]["boxes_blue"][0][:4] == (5, 5, 6, 6)
        # frame 1: no pixel match -> interpolated (copied) box, dashed flag
        assert frames[1]["boxes_green"][0][6] == 1


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
