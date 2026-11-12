# -*- coding: utf-8 -*-
"""Orbit camera of the 3D Flight Viewer (core/view_camera.py)."""
import numpy as np
import pytest

from bambi_wildlife_detection.core import flight_viewer as fv
from bambi_wildlife_detection.core.view_camera import OrbitCamera


def test_eye_is_distance_away_from_target():
    cam = OrbitCamera(target=(1, 2, 3), distance=10, azimuth=0, elevation=0)
    np.testing.assert_allclose(cam.eye(), [11, 2, 3])
    cam = OrbitCamera(target=(0, 0, 0), distance=10, azimuth=90, elevation=90)
    cam.orbit(0, 10)                       # clamped below the pole
    assert cam.elevation == pytest.approx(89.0)


def test_view_maps_target_onto_the_optical_axis():
    cam = OrbitCamera(target=(5, 5, 0), distance=20, azimuth=30, elevation=40)
    p = fv.transform_points(cam.view(), [cam.target])[0]
    np.testing.assert_allclose(p[:2], 0, atol=1e-9)
    assert p[2] == pytest.approx(-20)


def test_pan_moves_target_in_the_screen_plane():
    cam = OrbitCamera(distance=100, azimuth=-90, elevation=89.0)
    before = cam.target.copy()
    cam.pan(50, 0, 500)
    moved = cam.target - before
    assert abs(moved[2]) < 1e-6
    assert moved[0] < 0                    # dragging right moves the world right


def test_zoom_towards_point_keeps_that_point_fixed_on_screen():
    cam = OrbitCamera(target=(0, 0, 0), distance=100, azimuth=0, elevation=45)
    point = np.array([10.0, 5.0, 0.0])
    before = fv.transform_points(cam.proj(1.0) @ cam.view(), [point])[0]
    cam.zoom(0.5, towards=point)
    after = fv.transform_points(cam.proj(1.0) @ cam.view(), [point])[0]
    np.testing.assert_allclose(before[:2], after[:2], atol=1e-9)
    assert cam.distance == pytest.approx(50)


def test_fit_frames_the_box():
    cam = OrbitCamera(fovy=60)
    cam.fit([0, 0, 0], [100, 100, 0])
    np.testing.assert_allclose(cam.target, [50, 50, 0])
    assert cam.distance == pytest.approx(np.hypot(100, 100) / 2 / 0.5)


def test_adopt_keeps_looking_where_the_drone_camera_looks():
    drone = fv.FrameCamera(fv.look_at([0, 0, 100], [30, 40, 0]),
                           fv.perspective(50, 1, 0.1, 1000), 50.0, 1.0)
    cam = OrbitCamera()
    cam.adopt(drone, 50.0)
    np.testing.assert_allclose(cam.eye(), drone.position, atol=1e-6)
    view_dir = cam.target - cam.eye()
    view_dir /= np.linalg.norm(view_dir)
    np.testing.assert_allclose(view_dir, drone.forward, atol=1e-9)


def test_ray_through_viewport_centre_hits_the_target():
    cam = OrbitCamera(target=(3, 4, 0), distance=50, azimuth=10, elevation=60)
    origin, direction = cam.ray(400, 300, 800, 600)
    np.testing.assert_allclose(origin + direction * 50, cam.target, atol=1e-9)
