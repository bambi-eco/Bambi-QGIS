# -*- coding: utf-8 -*-
"""Detection on orthorectified GeoTIFFs: raster-pixel boxes are brought
into the camera frame (core.ortho_detection, 9.0.0)."""

import pytest

from bambi_wildlife_detection.core import ortho_detection as od


class Affine:
    """A north-up geotransform: 0.5 m pixels, upper-left at (1000, 2000)."""

    def __mul__(self, point):
        column, row = point
        return (1000.0 + 0.5 * column, 2000.0 - 0.5 * row)


def _project(points):
    """A nadir camera at local (10, -10) with 10 px per metre, frame 200 x
    200, image y pointing south."""
    xs = [100.0 + (x - 10.0) * 10.0 for x, _y, _z in points]
    ys = [100.0 - (y + 10.0) * 10.0 for _x, y, _z in points]
    return xs, ys


def _flat(points):
    return [50.0 for _ in points]


ORIGIN = (995.0, 2005.0)        # local = world - origin


def test_raster_pixels_map_to_world():
    assert od.raster_to_world(Affine(), 0, 0) == (1000.0, 2000.0)
    assert od.raster_to_world(Affine(), 10, 20) == (1005.0, 1990.0)
    assert od.raster_corners((1, 2, 3, 4)) == [(1.0, 2.0), (3.0, 2.0),
                                               (3.0, 4.0), (1.0, 4.0)]


def test_a_raster_box_lands_in_the_frame():
    # raster (8, 28)-(12, 32): world x 1004..1006, y 1986..1984
    # local x 9..11, y -19..-21  -> frame x 90..110, y 190..210 (clipped)
    boxes = od.boxes_to_frame([(8, 28, 12, 32)], Affine(), ORIGIN, _flat,
                              _project, 200, 200)
    assert boxes == [pytest.approx((90.0, 190.0, 110.0, 200.0))]


def test_boxes_keep_their_order_and_missing_ones_are_none():
    seen = []

    def heights(points):
        seen.append(len(points))
        # the second box has a corner outside the DEM
        return [50.0] * 4 + [50.0, None, 50.0, 50.0] + [50.0] * 4

    boxes = od.boxes_to_frame(
        [(8, 8, 12, 12), (0, 0, 2, 2), (1000, 1000, 1004, 1004)],
        Affine(), ORIGIN, heights, _project, 200, 200)
    assert seen == [12]                       # one DEM lookup for all boxes
    assert boxes[0] is not None
    assert boxes[1] is None                   # no ground height
    assert boxes[2] is None                   # far outside the frame


def test_the_ground_height_reaches_the_camera():
    got = []

    def project(points):
        got.extend(z for _x, _y, z in points)
        return _project(points)

    od.boxes_to_frame([(8, 8, 12, 12)], Affine(), ORIGIN,
                      lambda pts: [61.5] * len(pts), project, 200, 200)
    assert got == [61.5] * 4


def test_no_boxes_no_lookup():
    assert od.boxes_to_frame([], Affine(), ORIGIN, None, None, 10, 10) == []
