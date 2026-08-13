# -*- coding: utf-8 -*-
"""Camera pointing and the alfspy ray convention, against the real pyrr/alfspy.

Two things need proving here that the unit tier (which runs on fakes) cannot:

* the camera actually points where the gimbal pointed - heading about world up,
  applied after the tilt.  The older ``quaternion_from_eulers(e, 'zyx')``
  spelling applied the heading about the camera's own optical axis, which is
  exact at nadir and up to 128 degrees wrong at the horizon;
* both branches of :func:`core.camera_pose.ray_convention` cast the *same* world
  ray, so the plugin works against a legacy alfspy (rays via ``R33.T``) and a
  fixed one (rays via ``R33``) alike.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import numpy as np
import pytest

pyrr = pytest.importorskip("pyrr")
pytest.importorskip("alfspy")

from alfspy.core.geo.transform import Transform  # noqa: E402
from alfspy.core.util.pyrrs import quaternion_from_drone_pose  # noqa: E402
from pyrr import Vector3  # noqa: E402

from bambi_wildlife_detection.core import camera_pose  # noqa: E402


# [tilt off nadir, roll, heading clockwise from north]
POSES = [
    [0.0, 0.0, 51.6],       # nadir, yawed
    [90.0, 0.0, 51.6],      # horizon, north-east
    [45.0, 0.0, 120.0],     # oblique
    [7.3, 0.0, 15.4],       # near nadir
    [60.0, 0.0, -140.0],    # oblique, negative heading
    [76.4, 0.0, 50.0],      # the lion flight's dominant oblique pose
]


def expected_forward(tilt, heading):
    """ENU direction of a gimbal tilted *tilt* off nadir on *heading*."""
    t, h = np.radians(tilt), np.radians(heading)
    return np.array([np.sin(t) * np.sin(h), np.sin(t) * np.cos(h), -np.cos(t)])


def angle_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def camera_forward(camera, convention):
    """World-space forward vector of *camera* under the given ray convention."""
    rot = np.asarray(camera.transform.rotation.matrix33, dtype=float)
    local = np.array([0.0, 0.0, -1.0])
    return local @ (rot.T if convention == "legacy" else rot)


@pytest.mark.parametrize("rot", POSES)
@pytest.mark.parametrize("convention", ["legacy", "fixed"])
def test_camera_points_where_the_gimbal_pointed(rot, convention, monkeypatch):
    monkeypatch.setattr(camera_pose, "_RAY_CONVENTION", convention)
    meta = {"location": [0.0, 0.0, 100.0], "rotation": rot, "fovy": [50.0]}
    camera = camera_pose.build_camera(meta, {}, {})
    got = camera_forward(camera, convention)
    assert angle_between(got, expected_forward(rot[0], rot[2])) < 1e-3


@pytest.mark.parametrize("rot", POSES)
def test_both_conventions_cast_the_same_world_ray(rot, monkeypatch):
    meta = {"location": [0.0, 0.0, 100.0], "rotation": rot, "fovy": [50.0]}
    tan_fov = np.tan(np.deg2rad(50.0) / 2.0)
    # Off-centre, so a roll error about the optical axis would show up too.
    local = np.array([0.6 * tan_fov, -0.35 * tan_fov, -1.0])
    local /= np.linalg.norm(local)

    rays = {}
    for convention in ("legacy", "fixed"):
        monkeypatch.setattr(camera_pose, "_RAY_CONVENTION", convention)
        rot33 = np.asarray(
            camera_pose.build_camera(meta, {}, {}).transform.rotation.matrix33,
            dtype=float)
        ray = local @ (rot33.T if convention == "legacy" else rot33)
        rays[convention] = ray / np.linalg.norm(ray)

    assert np.allclose(rays["legacy"], rays["fixed"], atol=1e-6)


def test_heading_steers_the_tilt(monkeypatch):
    """The regression: two headings at the same tilt must differ."""
    monkeypatch.setattr(camera_pose, "_RAY_CONVENTION", "fixed")
    base = {"location": [0.0, 0.0, 100.0], "fovy": [50.0]}
    north = camera_forward(
        camera_pose.build_camera(dict(base, rotation=[45.0, 0.0, 0.0]), {}, {}),
        "fixed")
    east = camera_forward(
        camera_pose.build_camera(dict(base, rotation=[45.0, 0.0, 90.0]), {}, {}),
        "fixed")
    assert angle_between(north, east) == pytest.approx(60.0, abs=1e-2)
    assert north[0] == pytest.approx(0.0, abs=1e-5)
    assert east[1] == pytest.approx(0.0, abs=1e-5)


def test_correction_is_applied_once_and_in_radians(monkeypatch):
    """A 1 degree heading correction must move the camera by 1 degree, not 2."""
    monkeypatch.setattr(camera_pose, "_RAY_CONVENTION", "fixed")
    meta = {"location": [0.0, 0.0, 100.0], "rotation": [90.0, 0.0, 0.0],
            "fovy": [50.0]}
    plain = camera_forward(camera_pose.build_camera(meta, {}, {}), "fixed")
    nudged = camera_forward(
        camera_pose.build_camera(meta, {}, {"x": 0.0, "y": 0.0,
                                            "z": np.deg2rad(1.0)}), "fixed")
    assert angle_between(plain, nudged) == pytest.approx(1.0, abs=1e-3)


def test_probe_matches_the_installed_alfspy():
    pytest.importorskip("trimesh")
    camera_pose._RAY_CONVENTION = None
    convention = camera_pose.ray_convention(force_probe=True)
    assert convention in ("legacy", "fixed")

    import inspect

    from alfspy.core.convert.convert import pixel_to_world_coord

    source = inspect.getsource(pixel_to_world_coord)
    assert convention == ("legacy" if "matrix33.T" in source else "fixed")


def test_probe_result_is_cached():
    camera_pose._RAY_CONVENTION = None
    first = camera_pose.ray_convention()
    camera_pose._RAY_CONVENTION = "sentinel"
    assert camera_pose.ray_convention() == "sentinel"
    camera_pose._RAY_CONVENTION = None
    assert camera_pose.ray_convention() == first
