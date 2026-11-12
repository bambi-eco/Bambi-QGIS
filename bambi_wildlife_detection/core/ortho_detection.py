# -*- coding: utf-8 -*-
"""Detection on the orthorectified per-frame GeoTIFFs (9.0.0).

The detector then looks at the raster instead of the camera frame, so its
boxes are in **raster pixels**. The world position is one affine transform
away - that part is free. What still has to be worked out is where the box
is in the *camera frame*: a detection is stored as a box in the extracted
frame, because that is the picture the labelling tool, the inspector, the
video creator, the crops of the classifiers and the image exports all draw
it on. So each box goes

    raster pixels --(GeoTIFF transform)--> world --(frame camera)--> frame pixels

and from there on it is a detection like any other: the geo-referencing
step, tracking and everything downstream work unchanged.

Everything here is pure (the DEM lookup and the camera projection are
passed in), so it is testable without the rendering stack.
"""

from typing import Callable, List, Optional, Sequence, Tuple

#: Values of the detection "Projection" setting.
PERSPECTIVE = "perspective"
GEOTIFF = "geotiff"
PROJECTIONS = (PERSPECTIVE, GEOTIFF)


def raster_corners(box: Sequence[float]) -> List[Tuple[float, float]]:
    """The four corners of a raster-pixel box ``(x1, y1, x2, y2)``."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def raster_to_world(transform, column: float, row: float
                    ) -> Tuple[float, float]:
    """Raster pixel -> world ``(x, y)`` in the GeoTIFF's CRS (*transform*
    is its affine geotransform)."""
    x, y = transform * (float(column), float(row))
    return float(x), float(y)


def boxes_to_frame(boxes: Sequence[Sequence[float]], transform,
                   origin_xy: Tuple[float, float],
                   heights: Callable[[List[Tuple[float, float]]],
                                     List[Optional[float]]],
                   project: Callable[[List[Tuple[float, float, float]]],
                                     Tuple[Sequence[float], Sequence[float]]],
                   width: int, height: int,
                   min_side: float = 1.0
                   ) -> List[Optional[Tuple[float, float, float, float]]]:
    """Raster-pixel boxes as boxes in the camera frame.

    :param boxes: ``(x1, y1, x2, y2)`` in raster pixels
    :param transform: the GeoTIFF's affine geotransform
    :param origin_xy: the DEM origin - the world coordinates of the raster
        minus this are the mesh-local coordinates the poses are in
    :param heights: mesh-local ``[(x, y)] -> [ground z or None]``
    :param project: mesh-local ``[(x, y, z)] -> (pixel xs, pixel ys)`` in
        the frame of *width* x *height*
    :return: per box the frame box clipped to the frame, or None when it has
        no ground height (outside the DEM) or falls outside the frame
    """
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    local: List[Tuple[float, float]] = []
    for box in boxes:
        for column, row in raster_corners(box):
            x, y = raster_to_world(transform, column, row)
            local.append((x - ox, y - oy))
    if not local:
        return []
    ground = list(heights(local))

    out: List[Optional[Tuple[float, float, float, float]]] = []
    for index in range(len(boxes)):
        corners = local[index * 4:index * 4 + 4]
        zs = ground[index * 4:index * 4 + 4]
        if any(z is None for z in zs):
            out.append(None)
            continue
        xs, ys = project([(x, y, float(z))
                          for (x, y), z in zip(corners, zs)])
        x1, x2 = float(min(xs)), float(max(xs))
        y1, y2 = float(min(ys)), float(max(ys))
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 - x1 < min_side or y2 - y1 < min_side:
            out.append(None)                    # outside the frame
            continue
        out.append((x1, y1, x2, y2))
    return out
