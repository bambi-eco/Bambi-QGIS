# -*- coding: utf-8 -*-
"""Unit tests for the flat-earth geometry helpers in BambiProcessor.

These helpers drive the line-transect distance sampling (perpendicular
distance computation) and FOV footprint estimation, so they are tested
against hand-computed expectations.
"""
import math

from bambi_wildlife_detection.bambi_processing import BambiProcessor

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
# L-shaped (concave) polygon: the notch is the square (2..4, 2..4).
L_SHAPE = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.0, 2.0), (2.0, 4.0), (0.0, 4.0)]


class TestPointInPolygon:
    def test_inside_square(self):
        assert BambiProcessor._point_in_polygon(5.0, 5.0, SQUARE)

    def test_outside_square(self):
        assert not BambiProcessor._point_in_polygon(15.0, 5.0, SQUARE)
        assert not BambiProcessor._point_in_polygon(-1.0, -1.0, SQUARE)

    def test_concave_polygon(self):
        # (1, 3) is in the vertical arm of the L, (3, 3) is in the notch.
        assert BambiProcessor._point_in_polygon(1.0, 3.0, L_SHAPE)
        assert not BambiProcessor._point_in_polygon(3.0, 3.0, L_SHAPE)


class TestSegmentsIntersect:
    def test_proper_crossing(self):
        assert BambiProcessor._segments_intersect(0, 0, 10, 10, 0, 10, 10, 0)

    def test_parallel_segments(self):
        assert not BambiProcessor._segments_intersect(0, 0, 10, 0, 0, 5, 10, 5)

    def test_disjoint_segments(self):
        assert not BambiProcessor._segments_intersect(0, 0, 1, 0, 5, 5, 6, 5)

    def test_shared_endpoint_is_not_proper_crossing(self):
        # The helper deliberately tests for *proper* crossings only.
        assert not BambiProcessor._segments_intersect(0, 0, 10, 0, 10, 0, 10, 10)


class TestSegmentIntersectsPolygon:
    def test_segment_fully_inside(self):
        assert BambiProcessor._segment_intersects_polygon(2, 2, 8, 8, SQUARE)

    def test_segment_crossing_edge(self):
        assert BambiProcessor._segment_intersects_polygon(5, 5, 15, 5, SQUARE)

    def test_segment_through_polygon_endpoints_outside(self):
        assert BambiProcessor._segment_intersects_polygon(-5, 5, 15, 5, SQUARE)

    def test_segment_far_away(self):
        assert not BambiProcessor._segment_intersects_polygon(20, 20, 30, 30, SQUARE)


class TestNearestOnLinestring:
    def test_perpendicular_foot_on_segment(self):
        route = [[0.0, 0.0], [10.0, 0.0]]
        fx, fy, dist = BambiProcessor._nearest_on_linestring(route, 5.0, 3.0)
        assert (fx, fy) == (5.0, 0.0)
        assert dist == 3.0

    def test_clamps_beyond_segment_end(self):
        route = [[0.0, 0.0], [10.0, 0.0]]
        fx, fy, dist = BambiProcessor._nearest_on_linestring(route, 15.0, 4.0)
        assert (fx, fy) == (10.0, 0.0)
        assert dist == math.sqrt(25 + 16)

    def test_degenerate_zero_length_segment(self):
        route = [[2.0, 2.0], [2.0, 2.0]]
        fx, fy, dist = BambiProcessor._nearest_on_linestring(route, 5.0, 6.0)
        assert (fx, fy) == (2.0, 2.0)
        assert dist == 5.0

    def test_picks_nearest_of_multiple_segments(self):
        route = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
        fx, fy, dist = BambiProcessor._nearest_on_linestring(route, 9.0, 8.0)
        assert (fx, fy) == (10.0, 8.0)
        assert dist == 1.0


class TestNearestOnFovLinestring:
    """Tests for the transect-selection logic used by distance sampling."""

    # Two parallel transects (y=0 and y=50) joined by a connector at x=100.
    ROUTE = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]]

    def test_camera_position_selects_flown_transect(self):
        # Detection at y=26 is *closer* to the top transect (24 m) than to the
        # bottom one (26 m), but the drone was flying the bottom transect —
        # the camera position must win over raw proximity.
        fx, fy, dist = BambiProcessor._nearest_on_fov_linestring(
            self.ROUTE, [], 50.0, 26.0, cam_x=50.0, cam_y=1.0)
        assert (fx, fy) == (50.0, 0.0)
        assert dist == 26.0

    def test_camera_path_uses_unclamped_perpendicular_foot(self):
        # Detection beyond the transect end: the foot must stay perpendicular
        # to the transect line (x=120) instead of snapping to the endpoint.
        fx, fy, dist = BambiProcessor._nearest_on_fov_linestring(
            self.ROUTE, [], 120.0, 30.0, cam_x=50.0, cam_y=1.0)
        assert (fx, fy) == (120.0, 0.0)
        assert dist == 30.0

    def test_legacy_fov_filter_without_camera(self):
        # FOV polygon covers only a part of the bottom transect.
        fov = [(40.0, -10.0), (60.0, -10.0), (60.0, 10.0), (40.0, 10.0)]
        fx, fy, dist = BambiProcessor._nearest_on_fov_linestring(
            self.ROUTE, fov, 50.0, 26.0)
        assert (fx, fy) == (50.0, 0.0)
        assert dist == 26.0

    def test_no_visible_segment_falls_back_to_nearest(self):
        fov = [(1000.0, 1000.0), (1001.0, 1000.0), (1001.0, 1001.0), (1000.0, 1001.0)]
        fx, fy, dist = BambiProcessor._nearest_on_fov_linestring(
            self.ROUTE, fov, 50.0, 20.0)
        # Fallback: plain nearest point on the whole route (bottom transect).
        assert (fx, fy) == (50.0, 0.0)
        assert dist == 20.0


class TestComputeFrameFovPolygon:
    @staticmethod
    def _metadata(z=100.0, fovy=60.0, yaw=0.0):
        return {
            "location": [10.0, 20.0, z],
            "fovy": [fovy],
            "rotation": [0.0, 0.0, yaw],
        }

    def test_no_yaw_axis_aligned_footprint(self):
        poly = BambiProcessor._compute_frame_fov_polygon(
            self._metadata(), x_offset=1000.0, y_offset=2000.0)
        cam_x, cam_y = 1010.0, 2020.0
        altitude = 100.0
        half_h = altitude * math.tan(math.radians(30.0))
        # tan(atan(k)) == k, so half_w = altitude * aspect * tan(fovy/2)
        half_w = altitude * (4.0 / 3.0) * math.tan(math.radians(30.0))

        xs = sorted(p[0] for p in poly)
        ys = sorted(p[1] for p in poly)
        assert xs[0] == xs[1]  # axis-aligned rectangle
        assert math.isclose(xs[0], cam_x - half_w, rel_tol=1e-12)
        assert math.isclose(xs[-1], cam_x + half_w, rel_tol=1e-12)
        assert math.isclose(ys[0], cam_y - half_h, rel_tol=1e-12)
        assert math.isclose(ys[-1], cam_y + half_h, rel_tol=1e-12)

    def test_yaw_rotates_footprint_counterclockwise(self):
        poly0 = BambiProcessor._compute_frame_fov_polygon(
            self._metadata(yaw=0.0), 0.0, 0.0)
        poly90 = BambiProcessor._compute_frame_fov_polygon(
            self._metadata(yaw=90.0), 0.0, 0.0)
        cam_x, cam_y = 10.0, 20.0
        # Rotating (lx, ly) by +90° maps it onto (-ly, lx).
        for (x0, y0), (x90, y90) in zip(poly0, poly90):
            lx, ly = x0 - cam_x, y0 - cam_y
            assert math.isclose(x90 - cam_x, -ly, abs_tol=1e-9)
            assert math.isclose(y90 - cam_y, lx, abs_tol=1e-9)

    def test_scalar_fovy_and_altitude_floor(self):
        meta = {"location": [0.0, 0.0, 0.5], "fovy": 60.0, "rotation": [0, 0, 0]}
        poly = BambiProcessor._compute_frame_fov_polygon(meta, 0.0, 0.0)
        # Altitude is floored at 1 m, so half_h = tan(30°).
        half_h = math.tan(math.radians(30.0))
        ys = sorted(p[1] for p in poly)
        assert math.isclose(ys[-1], half_h, rel_tol=1e-12)

    def test_aspect_ratio_changes_width_only(self):
        wide = BambiProcessor._compute_frame_fov_polygon(
            self._metadata(), 0.0, 0.0, aspect_ratio=16.0 / 9.0)
        narrow = BambiProcessor._compute_frame_fov_polygon(
            self._metadata(), 0.0, 0.0, aspect_ratio=4.0 / 3.0)
        width = max(p[0] for p in wide) - min(p[0] for p in wide)
        width_n = max(p[0] for p in narrow) - min(p[0] for p in narrow)
        height = max(p[1] for p in wide) - min(p[1] for p in wide)
        height_n = max(p[1] for p in narrow) - min(p[1] for p in narrow)
        assert width > width_n
        assert math.isclose(height, height_n, rel_tol=1e-12)
