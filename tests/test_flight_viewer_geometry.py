# -*- coding: utf-8 -*-
"""Geometry and overlays of the 3D Flight Viewer (core/flight_viewer_geometry.py)."""
import numpy as np
import pytest

from bambi_wildlife_detection.core import flight_viewer as fv
from bambi_wildlife_detection.core import flight_viewer_geometry as geo


def test_route_splits_into_flown_and_remaining_part():
    pts = np.array([[i, 0, 100] for i in range(5)], float)
    seg, col = geo.route_segments(pts, 2)
    assert seg.shape == (4, 2, 3)
    np.testing.assert_allclose(col[:2, 0], [geo.ROUTE_PAST] * 2)
    np.testing.assert_allclose(col[2:, 0], [geo.ROUTE_FUTURE] * 2)


def test_polyline_skips_gaps():
    pts = [[0, 0, 0], [1, 0, 0], [np.nan, 0, 0], [3, 0, 0], [4, 0, 0]]
    seg, _ = geo.polyline_segments(pts, (1, 1, 1, 1))
    assert len(seg) == 2


def test_pose_markers_are_thinned_and_skip_current():
    pts = np.zeros((1000, 3))
    out, col = geo.pose_markers(pts, 0, max_markers=100)
    assert len(out) == 99 and col.shape == (99, 4)


def test_frustum_has_edges_rectangle_and_outline():
    seg, col = geo.frustum_segments([0, 0, 10], np.ones((4, 3)),
                                    outline=np.zeros((8, 3)))
    assert seg.shape == (8 + 8, 2, 3) and col.shape == (16, 2, 4)


def test_box_overlay_draws_outline_and_light_fill():
    img = geo.box_overlay((100, 50), [{"x1": 20, "y1": 10, "x2": 40, "y2": 30}],
                          scale=0.5, thickness=1)
    assert img.shape == (50, 100, 4)
    assert img[5, 10, 3] == 255           # top-left corner, scaled
    assert img[10, 15, 3] == 40           # inside: light fill
    assert img[0, 0, 3] == 0              # outside: transparent
    np.testing.assert_array_equal(img[5, 10, :3], geo.DETECTION_COLOR)


def test_box_overlay_colours_tracks_and_clips_to_image():
    img = geo.box_overlay((20, 20), [{"x1": -5, "y1": -5, "x2": 5, "y2": 5,
                                      "track_id": 3}], fill_alpha=0)
    np.testing.assert_array_equal(img[5, 2, :3], geo.track_color(3))


def test_scaled_size_keeps_aspect():
    assert geo.scaled_size(4000, 3000, 1000) == (1000, 750)
    assert geo.scaled_size(640, 512, 1024) == (640, 512)


def test_ortho_matrix_maps_extent_to_ndc_with_north_up():
    m = geo.ortho_matrix_for_bounds([0, 0], [100, 50])
    pts = fv.transform_points(m, [[0, 0, 300], [100, 50, -20]])
    np.testing.assert_allclose(pts[:, :2], [[-1, -1], [1, 1]], atol=1e-12)


def test_track_segments_colour_by_id():
    seg, col = geo.track_segments({5: np.array([[0, 0, 0], [1, 0, 0]], float)})
    assert seg.shape == (1, 2, 3)
    np.testing.assert_allclose(col[0, 0, :3], np.array(geo.track_color(5)) / 255)
    assert seg[0, 0, 2] > 0                # lifted off the ground


def test_box_edge_pixels_walk_round_the_box():
    px = geo.box_edge_pixels([{"x1": 0, "y1": 0, "x2": 4, "y2": 2}], 2)
    assert px.shape == (1, 8, 2)
    np.testing.assert_allclose(px[0, ::2], [[0, 0], [4, 0], [4, 2], [0, 2]])


def test_dashed_segments_cover_half_the_length():
    ring = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], float)
    seg, col = geo.dashed_segments(ring, (1, 1, 1, 1), dash=1.0)
    length = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1).sum()
    assert length == pytest.approx(20.0, abs=1e-6)       # half of 40 m
    assert col.shape == (len(seg), 2, 4)
    solid, _ = geo.ring_segments(ring, (1, 1, 1, 1))
    assert len(solid) == 4


def test_dashes_skip_missing_points():
    ring = np.array([[0, 0, 0], [4, 0, 0], [np.nan] * 3, [0, 4, 0]], float)
    seg, _ = geo.dashed_segments(ring, (1, 1, 1, 1), dash=1.0, closed=False)
    assert len(seg) == 2


def test_nearest_screen_point_picks_within_tolerance():
    view = fv.look_at([0, 0, 100], [0, 0, 0], up=(0, 1, 0))
    mvp = fv.perspective(60, 1.0, 1, 1000) @ view
    pts = np.array([[0, 0, 0], [20, 0, 0], [0, 0, 200]], float)  # last: behind
    assert geo.nearest_screen_point(pts, mvp, 400, 400, 200, 200, 5) == 0
    x = 200 + 20 / (100 * np.tan(np.radians(30))) * 200
    assert geo.nearest_screen_point(pts, mvp, 400, 400, x + 3, 201, 6) == 1
    assert geo.nearest_screen_point(pts, mvp, 400, 400, 10, 10, 6) is None
