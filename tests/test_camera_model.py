# -*- coding: utf-8 -*-
"""One camera model for the whole plugin (9.0.0).

Geo-referencing (``build_camera``) and rendering (ALFS, GeoTIFF export, the
Correction Wizard preview, the 3D Flight Viewer - all through ``shot_pose``)
must put a frame's pixel at the same place on the ground. Before 9.0.0 the
renderers built their own rotation (``quaternion_from_eulers(e, 'zyx')``)
and applied the correction as alfspy's shot transform (in camera space), so
tilted frames were up to ~30 degrees off and corrected flights metres off.
"""
import ast
import math
from pathlib import Path

import numpy as np
import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "bambi_wildlife_detection"

POSES = [
    {"location": [0.0, 0.0, 100.0], "rotation": [0.0, 0.0, 0.0], "fovy": 50.0},
    {"location": [5.0, -3.0, 90.0], "rotation": [0.0, 0.0, 73.0], "fovy": 50.0},
    {"location": [-4.0, 8.0, 110.0], "rotation": [20.0, 0.0, 73.0], "fovy": 45.0},
    {"location": [1.0, 2.0, 80.0], "rotation": [45.0, 3.0, 200.0], "fovy": 60.0},
]
CORRECTIONS = [
    ({}, {}),
    ({"x": 1.5, "y": -2.0, "z": 0.5}, {"x": 0.02, "y": -0.01, "z": 0.05}),
]


@pytest.mark.parametrize("meta", POSES)
@pytest.mark.parametrize("t,r", CORRECTIONS)
def test_rendered_pixel_lands_where_georeferencing_puts_it(meta, t, r):
    """Project ground points with the render shot, cast the pixels back
    with the geo-referencing camera: the same points come out."""
    pytest.importorskip("alfspy")
    trimesh = pytest.importorskip("trimesh")
    from alfspy.core.convert.convert import pixel_to_world_coord
    from alfspy.core.rendering import Camera
    from bambi_wildlife_detection.core.camera_pose import build_camera, shot_pose

    w, h = 640, 512
    aspect = w / h
    pos, rot = shot_pose(meta, t, r)
    shot_cam = Camera(meta["fovy"], aspect, position=pos, rotation=rot)
    mat = (np.asarray(shot_cam.get_view(dtype="f8"), float)
           @ np.asarray(shot_cam.get_proj(dtype="f8"), float))

    # Ground points inside the frame: shoot rays of the shot camera at z=0.
    to_world = np.linalg.inv(np.asarray(shot_cam.get_view(dtype="f8"), float))
    tan = math.tan(math.radians(meta["fovy"]) / 2)
    local = np.array([[x * tan * aspect, y * tan, -1.0]
                      for x in (-0.6, 0.0, 0.5) for y in (-0.5, 0.0, 0.6)])
    dirs = local @ to_world[:3, :3]
    origin = to_world[3, :3]
    ground = origin + dirs * (-origin[2] / dirs[:, 2])[:, None]

    clip = np.hstack([ground, np.ones((len(ground), 1))]) @ mat
    ndc = clip[:, :2] / clip[:, 3:]
    px = (ndc[:, 0] + 1) * w / 2
    py = h - (ndc[:, 1] + 1) * h / 2

    plane = trimesh.Trimesh(vertices=[[-5e3, -5e3, 0], [5e3, -5e3, 0],
                                      [5e3, 5e3, 0], [-5e3, 5e3, 0]],
                            faces=[[0, 1, 2], [0, 2, 3]])
    geo_cam = build_camera(meta, t or {}, r or {}, aspect_ratio=aspect)
    hits = np.reshape(pixel_to_world_coord(px, py, w, h, plane, geo_cam,
                                           include_misses=False), (-1, 3))
    np.testing.assert_allclose(hits, ground, atol=1e-2)


def test_translation_correction_is_in_world_axes():
    pytest.importorskip("alfspy")
    from bambi_wildlife_detection.core.camera_pose import shot_pose
    for heading in (0.0, 90.0, 180.0):
        pos, _ = shot_pose({"location": [10, 20, 100], "rotation": [0, 0, heading]},
                           {"x": 2.0, "y": -1.0, "z": 0.5}, {})
        np.testing.assert_allclose(np.asarray(pos), [12, 19, 100.5], atol=1e-5)


def _calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            owner = f.value.id if isinstance(f, ast.Attribute) and \
                isinstance(f.value, ast.Name) else ""
            yield node, owner, name


def test_only_camera_pose_builds_camera_rotations():
    """No module may turn pose eulers into a rotation on its own."""
    offenders = []
    for path in PLUGIN_DIR.rglob("*.py"):
        if path.name == "camera_pose.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node, owner, name in _calls(tree):
            if name in ("quaternion_from_eulers", "quaternion_from_drone_pose",
                        "label_camera_for") or (owner == "Quaternion"
                                                and name == "from_eulers"):
                offenders.append(f"{path.relative_to(PLUGIN_DIR)}:{node.lineno}")
    assert not offenders, (
        "Camera rotations must come from core.camera_pose "
        f"(corrected_pose / shot_pose / build_camera): {offenders}")


def test_shots_never_carry_a_correction_transform():
    """The correction is part of the pose; a CtxShot correction would apply
    it a second time, in camera space."""
    offenders = []
    for path in PLUGIN_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node, _, name in _calls(tree):
            if name != "CtxShot":
                continue
            corr = node.args[6] if len(node.args) > 6 else next(
                (k.value for k in node.keywords if k.arg == "correction"), None)
            if corr is not None and not (isinstance(corr, ast.Constant)
                                         and corr.value is None):
                offenders.append(f"{path.relative_to(PLUGIN_DIR)}:{node.lineno}")
    assert not offenders, offenders
