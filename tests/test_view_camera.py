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


# ---------------------------------------------------------------------------
# Which way the view looks (the compass of the 3D view)
# ---------------------------------------------------------------------------

class TestBearing:
    """x is east and y north, as in the map the scene is drawn in."""

    def looking(self, towards, up=(0.0, 0.0, 1.0), eye=(0.0, 0.0, 100.0)):
        from bambi_wildlife_detection.core.flight_viewer import look_at

        eye = np.array(eye, dtype=float)
        return look_at(eye, eye + np.array(towards, dtype=float), up)

    @pytest.mark.parametrize("towards, bearing", [
        ((0, 1, 0), 0.0), ((1, 0, 0), 90.0), ((0, -1, 0), 180.0),
        ((-1, 0, 0), 270.0), ((1, 1, 0), 45.0), ((-1, 1, 0), 315.0)])
    def test_a_level_view_looks_where_it_is_pointed(self, towards, bearing):
        from bambi_wildlife_detection.core.view_camera import view_bearing

        assert view_bearing(self.looking(towards)) == pytest.approx(bearing)

    def test_a_view_down_at_a_slant_looks_the_same_way(self):
        from bambi_wildlife_detection.core.view_camera import view_bearing

        assert view_bearing(self.looking((1, 1, -3))) == pytest.approx(45.0)
        assert view_bearing(self.looking((0, -1, -0.01))) == pytest.approx(
            180.0)

    def test_straight_down_it_is_what_is_up_on_the_screen(self):
        from bambi_wildlife_detection.core.view_camera import view_bearing

        north_up = self.looking((0, 0, -1), up=(0, 1, 0))
        east_up = self.looking((0, 0, -1), up=(1, 0, 0))

        assert view_bearing(north_up) == pytest.approx(0.0)
        assert view_bearing(east_up) == pytest.approx(90.0)

    def test_the_orbit_camera_as_the_viewer_opens_it(self):
        """Azimuth -90: the eye is south of the target, looking north."""
        from bambi_wildlife_detection.core.view_camera import (
            OrbitCamera, view_bearing)

        camera = OrbitCamera(azimuth=-90.0, elevation=50.0)
        assert view_bearing(camera.view()) == pytest.approx(0.0, abs=1e-6)
        camera.orbit(90.0, 0.0)          # the eye goes round to the east
        assert view_bearing(camera.view()) == pytest.approx(270.0)
        camera.orbit(0.0, 39.0)          # nearly overhead: still west
        assert view_bearing(camera.view()) == pytest.approx(270.0)

    def test_a_matrix_that_says_nothing(self):
        from bambi_wildlife_detection.core.view_camera import view_bearing

        assert view_bearing(np.zeros((4, 4))) == 0.0
        assert view_bearing(np.full((4, 4), np.nan)) == 0.0

    @pytest.mark.parametrize("bearing, point", [
        (0, "N"), (22.4, "N"), (22.6, "NE"), (47, "NE"), (90, "E"),
        (135, "SE"), (180, "S"), (225, "SW"), (270, "W"), (315, "NW"),
        (337.6, "N"), (359.9, "N"), (360, "N"), (-90, "W"), (450, "E")])
    def test_the_nearest_of_eight_points(self, bearing, point):
        from bambi_wildlife_detection.core.view_camera import compass_point

        assert compass_point(bearing) == point


class TestScale:
    def test_a_pixel_covers_more_the_farther_the_terrain_is(self):
        from bambi_wildlife_detection.core.view_camera import metres_per_pixel

        # 90 degrees: the view is twice as high as the terrain is far
        assert metres_per_pixel(100.0, 90.0, 400) == pytest.approx(0.5)
        assert metres_per_pixel(200.0, 90.0, 400) == pytest.approx(1.0)
        assert metres_per_pixel(100.0, 90.0, 800) == pytest.approx(0.25)

    @pytest.mark.parametrize("distance, fovy, height", [
        (0.0, 45.0, 400), (100.0, 0.0, 400), (100.0, 45.0, 0),
        (float("nan"), 45.0, 400), (100.0, 180.0, 400)])
    def test_no_scale_where_there_is_no_saying(self, distance, fovy, height):
        from bambi_wildlife_detection.core.view_camera import metres_per_pixel

        assert metres_per_pixel(distance, fovy, height) == 0.0

    @pytest.mark.parametrize("per_pixel, length", [
        (1.0, 100.0), (0.5, 50.0), (0.3, 20.0), (0.2, 20.0), (0.19, 10.0),
        (12.0, 1000.0), (60.0, 5000.0), (0.004, 0.2), (0.01, 1.0)])
    def test_the_bar_is_the_largest_round_length_that_fits(
            self, per_pixel, length):
        from bambi_wildlife_detection.core.view_camera import scale_length

        found = scale_length(per_pixel, 100)

        assert found == pytest.approx(length)
        # never wider than allowed, never less than two fifths of it
        assert 40.0 - 1e-6 <= found / per_pixel <= 100.0 + 1e-6

    def test_no_bar_without_a_scale(self):
        from bambi_wildlife_detection.core.view_camera import scale_length

        assert scale_length(0.0, 100) == 0.0
        assert scale_length(float("inf"), 100) == 0.0
        assert scale_length(1.0, 0) == 0.0

    @pytest.mark.parametrize("length, text", [
        (50.0, "50 m"), (1.0, "1 m"), (1000.0, "1 km"), (5000.0, "5 km"),
        (0.2, "20 cm"), (0.01, "1 cm"), (0.005, "5 mm")])
    def test_its_length_is_said_without_decimals(self, length, text):
        from bambi_wildlife_detection.core.view_camera import scale_text

        assert scale_text(length) == text
