# -*- coding: utf-8 -*-
"""Headless logic of the 3D Flight Viewer (core/flight_viewer.py)."""
import json
import math

import numpy as np
import pytest

from bambi_wildlife_detection.core import flight_viewer as fv

try:  # alfspy / pyrr are optional in the unit image
    import alfspy  # noqa: F401
    import pyrr  # noqa: F401
    HAVE_ALFSPY = True
except ImportError:
    HAVE_ALFSPY = False

needs_alfspy = pytest.mark.skipif(not HAVE_ALFSPY, reason="alfspy not installed")

POSES = [
    {"location": [0.0, 0.0, 100.0], "rotation": [0.0, 0.0, 0.0]},
    {"location": [5.0, -3.0, 90.0], "rotation": [0.0, 0.0, 73.0]},
    {"location": [-4.0, 8.0, 110.0], "rotation": [20.0, 0.0, 73.0]},
    {"location": [1.0, 2.0, 80.0], "rotation": [45.0, 3.0, 200.0]},
]
CORRECTIONS = [
    ({}, {}),
    ({"x": 1.5, "y": -2.0, "z": 0.5}, {"x": 0.02, "y": -0.01, "z": 0.05}),
]


# -- matrix helpers -----------------------------------------------------------

def test_look_at_maps_eye_to_origin_and_target_forward():
    view = fv.look_at([10, 0, 5], [0, 0, 0])
    eye = fv.transform_points(view, [[10, 0, 5]])[0]
    np.testing.assert_allclose(eye, 0, atol=1e-9)
    target = fv.transform_points(view, [[0, 0, 0]])[0]
    assert target[2] < 0 and abs(target[0]) < 1e-9


def test_perspective_maps_near_and_far_planes():
    proj = fv.perspective(60, 1.5, 1.0, 100.0)
    near = fv.transform_points(proj, [[0, 0, -1.0]])[0]
    far = fv.transform_points(proj, [[0, 0, -100.0]])[0]
    assert near[2] == pytest.approx(-1.0)
    assert far[2] == pytest.approx(1.0)


def test_orthographic_maps_box_to_ndc_cube():
    proj = fv.orthographic(-10, 30, 0, 20, 1, 11)
    pts = fv.transform_points(proj, [[-10, 0, -1], [30, 20, -11]])
    np.testing.assert_allclose(pts, [[-1, -1, -1], [1, 1, 1]], atol=1e-12)


# -- cameras -------------------------------------------------------------------

def _shot_reference(meta, t, r, aspect=1.0):
    """World -> clip of the shot ALFS renders the frame with (row-vector):
    the shared pose, no shot correction."""
    from alfspy.core.rendering import Camera
    from bambi_wildlife_detection.core.camera_pose import shot_pose
    pos, rot = shot_pose(meta, t, r)
    cam = Camera(meta.get("fovy", 50.0), aspect, position=pos, rotation=rot)
    v = np.asarray(cam.get_view(dtype="f8"), dtype=np.float64)
    p = np.asarray(cam.get_proj(dtype="f8"), dtype=np.float64)
    return v @ p


@needs_alfspy
@pytest.mark.parametrize("meta", POSES)
@pytest.mark.parametrize("t,r", CORRECTIONS)
def test_viewer_drapes_frames_like_the_alfs_shot(meta, t, r):
    cam = fv.camera_for_pose(meta, t, r, aspect=1.25)
    ref = _shot_reference(meta, t, r, aspect=1.25)
    pts = np.array([[0, 0, 0], [12, -7, 3], [-20, 15, -2]], dtype=float)
    homo = np.hstack([pts, np.ones((3, 1))])
    expected = homo @ ref
    expected = expected[:, :3] / expected[:, 3:]
    got = fv.transform_points(cam.shot_matrix, pts)
    np.testing.assert_allclose(got[:, :2], expected[:, :2], atol=1e-4)


@needs_alfspy
@pytest.mark.parametrize("meta", POSES)
@pytest.mark.parametrize("t,r", CORRECTIONS)
def test_viewer_camera_casts_the_rays_of_build_camera(meta, t, r):
    trimesh = pytest.importorskip("trimesh")
    from alfspy.core.convert.convert import pixel_to_world_coord
    from bambi_wildlife_detection.core.camera_pose import build_camera

    ground = trimesh.Trimesh(
        vertices=[[-2000, -2000, 0], [2000, -2000, 0], [2000, 2000, 0],
                  [-2000, 2000, 0]], faces=[[0, 1, 2], [0, 2, 3]])
    w, h = 640, 512
    pixels = np.array([[320, 256], [10, 20], [600, 400]], dtype=float)
    ref_cam = build_camera(dict(meta, fovy=50.0), t or fv._ZERO, r or fv._ZERO,
                           aspect_ratio=w / h)
    expected = np.reshape(pixel_to_world_coord(
        pixels[:, 0], pixels[:, 1], w, h, ground, ref_cam,
        include_misses=False), (-1, 3))

    cam = fv.camera_for_pose(dict(meta, fovy=50.0), t, r, aspect=w / h)
    got = fv.cast_rays(ground, cam.position[None, :],
                       fv.pixel_rays(cam, w, h, pixels))
    np.testing.assert_allclose(got, expected, atol=1e-3)


@needs_alfspy
def test_nadir_camera_looks_down_from_its_location():
    cam = fv.camera_for_pose({"location": [1, 2, 50], "rotation": [0, 0, 0]},
                             {}, {})
    np.testing.assert_allclose(cam.position, [1, 2, 50], atol=1e-5)
    np.testing.assert_allclose(cam.forward, [0, 0, -1], atol=1e-5)


@needs_alfspy
def test_georef_translation_correction_moves_the_camera_in_world_space():
    meta = {"location": [0, 0, 100], "rotation": [0, 0, 90]}
    cam = fv.camera_for_pose(meta, {"x": 2, "y": 0, "z": -1}, {})
    np.testing.assert_allclose(cam.position, [2, 0, 99], atol=1e-5)


def test_quaternion_rotation_is_accepted():
    if not HAVE_ALFSPY:
        pytest.skip("alfspy not installed")
    meta = {"location": [0, 0, 10], "rotation": [1.0, 0.0, 0.0, 0.0]}
    cam = fv.camera_for_pose(meta, {"x": 1}, {})
    assert np.all(np.isfinite(cam.view))
    np.testing.assert_allclose(cam.position, [1, 0, 10], atol=1e-6)


@needs_alfspy
def test_frustum_corners_of_nadir_camera_on_flat_ground():
    trimesh = pytest.importorskip("trimesh")
    ground = trimesh.Trimesh(
        vertices=[[-500, -500, 0], [500, -500, 0], [500, 500, 0],
                  [-500, 500, 0]], faces=[[0, 1, 2], [0, 2, 3]])
    cam = fv.camera_for_pose({"location": [0, 0, 100], "rotation": [0, 0, 0],
                              "fovy": 60.0}, {}, {}, aspect=2.0)
    corners, outline = fv.frustum_ground(cam, ground, samples_per_edge=4)
    half_h = 100 * math.tan(math.radians(30))
    np.testing.assert_allclose(np.abs(corners[:, 1]), half_h, atol=1e-3)
    np.testing.assert_allclose(np.abs(corners[:, 0]), 2 * half_h, atol=1e-3)
    np.testing.assert_allclose(corners[:, 2], 0, atol=1e-6)
    assert outline.shape == (16, 3)
    # Top-left of the image is the camera's up/left: +y / -x at heading 0.
    assert corners[0, 0] < 0 and corners[0, 1] > 0


def test_frustum_ground_falls_back_to_far_distance_on_a_miss():
    cam = fv.FrameCamera(fv.look_at([0, 0, 10], [0, 0, 20], up=(0, 1, 0)),
                         fv.perspective(40, 1, 0.1, 100), 40.0, 1.0)
    corners, _ = fv.frustum_ground(cam, None, far=50.0)
    np.testing.assert_allclose(np.linalg.norm(corners - [0, 0, 10], axis=1),
                               50.0)


def test_pixel_ray_through_image_centre_is_the_forward_axis():
    cam = fv.FrameCamera(fv.look_at([0, 0, 10], [5, 0, 0]),
                         fv.perspective(40, 1.5, 0.1, 100), 40.0, 1.5)
    ray = fv.pixel_rays(cam, 300, 200, [[150, 100]])[0]
    np.testing.assert_allclose(ray, cam.forward, atol=1e-9)
    up_ray = fv.pixel_rays(cam, 300, 200, [[150, 0]])[0]
    assert np.dot(up_ray, cam.up) > 0      # image row 0 is the top


# -- windows and stepping ------------------------------------------------------

def test_alfs_window_spans_n_frames_around_the_current_one():
    assert fv.alfs_window(10, 100, 12, 3) == [4, 7, 10, 13, 16]
    assert fv.alfs_window(10, 100, 4, 1) == [8, 9, 10, 11, 12]
    assert fv.alfs_window(10, 100, 0, 1) == [10]


def test_alfs_window_is_clipped_at_flight_edges():
    assert fv.alfs_window(1, 5, 4, 1) == [0, 1, 2, 3]
    assert fv.alfs_window(4, 5, 12, 2) == [0, 2, 4]
    assert fv.alfs_window(0, 0, 6, 1) == []


def test_step_index_clamps_or_wraps():
    assert fv.step_index(8, 5, 10) == 9
    assert fv.step_index(1, -5, 10) == 0
    assert fv.step_index(8, 5, 10, loop=True) == 3


def test_playback_delays_follow_timestamps():
    assert fv.playback_delays([0.0, 0.5, 1.5], 10.0) == [0.5, 1.0, 0.1]
    assert fv.playback_delays([0.0, 0.5, None], 4.0, speed=2.0) == \
        [0.25, 0.125, 0.125]


# -- frame matching ------------------------------------------------------------

def _timed(n, start, step):
    return [{"timestamp": f"2025-01-01T10:00:{start + i * step:06.3f}+00:00"}
            for i in range(n)]


def test_matching_frame_uses_capture_time():
    thermal = _timed(30, 0.0, 1 / 30)
    rgb = _timed(10, 0.0, 0.1)
    assert fv.matching_frame(thermal, rgb, 15) == 5
    assert fv.matching_frame(rgb, thermal, 5) == 15


def test_matching_frame_without_timestamps_keeps_the_index():
    assert fv.matching_frame([{}] * 20, [{}] * 10, 15) == 9


def test_timeline_is_relative_to_first_frame():
    times = fv.timeline(_timed(3, 2.0, 0.5))
    np.testing.assert_allclose(times, [0.0, 0.5, 1.0])


# -- annotations ---------------------------------------------------------------

TRACKS = {
    1: [{"frame": f, "x1": f, "y1": 0, "x2": f + 2, "y2": 2} for f in range(0, 6)],
    2: [{"frame": f, "x1": 0, "y1": 0, "x2": 1, "y2": 1} for f in (3, 4)],
}


def test_tracks_at_returns_history_and_current_box():
    states = {s.track_id: s for s in fv.tracks_at(TRACKS, 4)}
    assert [b["frame"] for b in states[1].history] == [0, 1, 2, 3, 4]
    assert states[1].current["frame"] == 4
    assert states[2].current["frame"] == 4


def test_tracks_at_limits_history_and_drops_lost_tracks():
    states = {s.track_id: s for s in fv.tracks_at(TRACKS, 5, history_frames=2)}
    assert [b["frame"] for b in states[1].history] == [4, 5]
    assert 2 not in states
    lingering = {s.track_id: s for s in fv.tracks_at(TRACKS, 5, linger_frames=1)}
    assert lingering[2].current is None


def test_boxes_on_frame_and_centres():
    dets = [{"frame": 1, "x1": 0, "y1": 0, "x2": 4, "y2": 2},
            {"frame": 2, "x1": 0, "y1": 0, "x2": 1, "y2": 1}]
    on = fv.boxes_on_frame(dets, 1)
    np.testing.assert_allclose(fv.box_centers(on), [[2, 1]])


# -- DEM and scene -------------------------------------------------------------

def test_vertex_normals_of_a_flat_grid_point_up():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    f = np.array([[0, 2, 1], [0, 3, 2]])          # wound downward
    np.testing.assert_allclose(fv.vertex_normals(v, f),
                               [[0, 0, 1]] * 4, atol=1e-6)


def test_correction_from_config_uses_the_file_layout():
    corr = fv.correction_from_config({
        "translation": {"x": 1, "y": 2, "z": 3},
        "rotation": {"x": 0, "y": 0, "z": 0.1},
        "additional_corrections": [{"start": 5, "end": 9,
                                    "translation": {"x": 9, "y": 9, "z": 9},
                                    "rotation": {"x": 0, "y": 0, "z": 0}}]})
    assert fv.correction_for(6, corr)[0]["x"] == 9
    assert fv.correction_for(2, corr)[0]["x"] == 1


@needs_alfspy
def test_scene_loads_poses_and_applies_frame_range_corrections(tmp_path):
    images = [{"imagefile": f"{i:04d}.png", "location": [i, 0, 100],
               "rotation": [0, 0, 0]} for i in range(5)]
    (tmp_path / "poses_w.json").write_text(json.dumps({"images": images}))
    corr = {"translation": {"x": 0, "y": 0, "z": 0},
            "rotation": {"x": 0, "y": 0, "z": 0},
            "additional": [{"start": 3, "end": 4,
                            "translation": {"x": 0, "y": 0, "z": -10},
                            "rotation": {"x": 0, "y": 0, "z": 0}}]}
    scene = fv.FlightScene.load(str(tmp_path), "w", corr,
                                image_size=(640, 512))
    assert len(scene) == 5 and scene.aspect == pytest.approx(1.25)
    pos = scene.positions()
    np.testing.assert_allclose(pos[:, 2], [100, 100, 100, 90, 90], atol=1e-4)
    assert scene.frame_path(2).endswith("frames_w" + __import__("os").sep
                                        + "0002.png")
    scene.set_correction({})
    np.testing.assert_allclose(scene.positions()[:, 2], 100, atol=1e-4)


# -- alfspy interop -------------------------------------------------------------

@needs_alfspy
@pytest.mark.parametrize("meta", POSES)
def test_alfs_camera_reproduces_the_viewer_camera(meta):
    cam = fv.camera_for_pose(meta, CORRECTIONS[1][0], CORRECTIONS[1][1],
                             aspect=1.25)
    alfs = fv.alfs_camera(cam)
    view = fv.to_row(np.asarray(alfs.get_view(dtype="f8"), dtype=float))
    np.testing.assert_allclose(view, cam.view, atol=1e-4)


@needs_alfspy
def test_ortho_camera_looks_down_with_north_up():
    cam = fv.alfs_ortho_camera([0, 0], [100, 50], top=80, bottom=-5)
    mat = np.asarray(cam.get_view(dtype="f8")) @ np.asarray(cam.get_proj(dtype="f8"))
    pts = np.array([[0, 0, 0, 1], [100, 50, 0, 1]], float) @ mat
    ndc = pts[:, :2] / pts[:, 3:]
    np.testing.assert_allclose(ndc, [[-1, -1], [1, 1]], atol=1e-5)


def test_window_extent_covers_the_footprints():
    cams = [fv.FrameCamera(fv.look_at([x, 0, 100], [x, 0, 0], up=(0, 1, 0)),
                           fv.perspective(60, 1, 0.1, 1000), 60.0, 1.0)
            for x in (0.0, 50.0)]
    trimesh = pytest.importorskip("trimesh")
    ground = trimesh.Trimesh(vertices=[[-1e3, -1e3, 0], [1e3, -1e3, 0],
                                       [1e3, 1e3, 0], [-1e3, 1e3, 0]],
                             faces=[[0, 1, 2], [0, 2, 3]])
    lo, hi = fv.window_extent(cams, ground, pad_fraction=0.0)
    half = 100 * math.tan(math.radians(30))
    np.testing.assert_allclose(lo, [-half - 1, -half - 1], atol=1e-3)
    np.testing.assert_allclose(hi, [50 + half + 1, half + 1], atol=1e-3)
    assert fv.window_extent(cams, None) is None


def test_integral_to_rgba_normalises_and_thresholds():
    accum = np.array([[[2.0, 1.0, 0.5, 2.0], [0.4, 0.4, 0.4, 1.0]]])
    cov = np.array([[2.0, 1.0]])
    out = fv.integral_to_rgba(accum, cov, min_coverage=2)
    np.testing.assert_array_equal(out[0, 0], [255, 127, 63, 255])
    assert out[0, 1, 3] == 0


# -- DEM detail ----------------------------------------------------------------

def _grid_dem(n=101, relief=lambda x, y: 0.02 * x + 3 * np.sin(x / 40) * np.cos(y / 30)):
    xs = np.linspace(-200, 200, n)
    X, Y = np.meshgrid(xs, xs)
    v = np.column_stack([X.ravel(), Y.ravel(), relief(X, Y).ravel()]).astype(np.float32)
    a = np.arange(n * n).reshape(n, n)
    f = np.concatenate([
        np.column_stack([a[:-1, :-1].ravel(), a[:-1, 1:].ravel(), a[1:, 1:].ravel()]),
        np.column_stack([a[:-1, :-1].ravel(), a[1:, 1:].ravel(), a[1:, :-1].ravel()])])
    uv = np.column_stack([(v[:, 0] + 200) / 400, (v[:, 1] + 200) / 400]).astype(np.float32)
    tex = np.zeros((4, 4, 3), np.uint8)
    return fv.DemData(v, f.astype(np.uint32), fv.vertex_normals(v, f), uv, tex)


@pytest.mark.parametrize("fraction", [0.66, 0.33])
def test_simplify_dem_keeps_the_share_of_triangles(fraction):
    pytest.importorskip("trimesh")
    dem = _grid_dem()
    s = fv.simplify_dem(dem, fraction)
    assert s.fraction == pytest.approx(fraction, rel=0.1)
    assert 0 <= s.rms_error <= s.max_error < 0.5      # smooth relief
    lo, hi = s.dem.bounds
    np.testing.assert_allclose(lo[:2], [-200, -200], atol=0.01)
    np.testing.assert_allclose(hi[:2], [200, 200], atol=0.01)
    # uv mapping carried over linearly, so the DEM texture still fits
    np.testing.assert_allclose(s.dem.uvs, (s.dem.vertices[:, :2] + 200) / 400,
                               atol=1e-5)
    assert s.dem.texture is not None


def test_simplify_dem_reports_the_error_of_rough_terrain():
    pytest.importorskip("trimesh")
    rough = _grid_dem(relief=lambda x, y: np.where((x // 4 + y // 4) % 2 == 0, 5.0, 0.0))
    s = fv.simplify_dem(rough, 0.33)
    assert s.max_error > 1.0


def test_full_detail_is_the_dem_itself():
    dem = _grid_dem(11)
    s = fv.simplify_dem(dem, 1.0)
    assert s.dem is dem and s.max_error == 0.0


# -- undistortion masks ---------------------------------------------------------

def test_find_mask_prefers_the_camera_specific_file(tmp_path):
    (tmp_path / "poses_w.json").write_text(json.dumps({"images": [], "mask": "m.png"}))
    assert fv.find_mask(str(tmp_path), "w") == ""
    (tmp_path / "m.png").write_bytes(b"x")
    assert fv.find_mask(str(tmp_path), "w").endswith("m.png")
    (tmp_path / "mask_W.png").write_bytes(b"x")
    assert fv.find_mask(str(tmp_path), "w").endswith("mask_W.png")
    assert fv.find_mask(str(tmp_path), "t") == ""


def test_apply_mask_makes_the_border_transparent():
    rgba = np.full((40, 60, 4), 200, np.uint8)
    mask = np.zeros((20, 30), np.uint8)
    mask[2:18, 3:27] = 255
    out = fv.apply_mask(rgba, mask)
    assert out[0, 0, 3] == 0 and out[20, 30, 3] == 200
    np.testing.assert_array_equal(out[..., :3], rgba[..., :3])
    assert fv.apply_mask(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4)))[0, 0, 3] == 255


def test_erode_mask_shrinks_the_valid_area():
    m = np.zeros((10, 10), bool)
    m[2:8, 2:8] = True
    e = fv.erode_mask(m, 1)
    assert e.sum() == 16 and e[3:7, 3:7].all()
    assert fv.erode_mask(m, 0).sum() == 36
    assert fv.erode_mask(m, 5).sum() == 0


def test_apply_mask_erodes_at_texture_size():
    rgba = np.full((20, 20, 4), 255, np.uint8)
    mask = np.zeros((40, 40), np.uint8)
    mask[4:36, 4:36] = 255                 # valid: rows 2..17 at 20 px
    plain = fv.apply_mask(rgba, mask)
    eroded = fv.apply_mask(rgba, mask, erode_px=2)
    assert plain[2, 2, 3] == 255 and eroded[2, 2, 3] == 0
    assert eroded[4, 4, 3] == 255


def test_pick_nearest_takes_the_first_surface_along_the_ray():
    trimesh = pytest.importorskip("trimesh")

    def plane(z):
        return trimesh.Trimesh(vertices=[[-50, -50, z], [50, -50, z], [50, 50, z],
                                         [-50, 50, z]], faces=[[0, 1, 2], [0, 2, 3]])
    dem, canopy = plane(0.0), plane(12.0)
    p, name = fv.pick_nearest([1, 2, 100], [0, 0, -1], [("dem", dem),
                                                         ("surface", canopy)])
    assert name == "surface" and p[2] == pytest.approx(12.0)
    p, name = fv.pick_nearest([1, 2, 100], [0, 0, -1], [("dem", dem),
                                                         ("surface", None)])
    assert name == "dem" and p[2] == pytest.approx(0.0)
    assert fv.pick_nearest([0, 0, 100], [0, 0, 1], [("dem", dem)]) == (None, None)
