# -*- coding: utf-8 -*-
"""What a cell of a survey raster says: a value, a zero, or nothing.

The density heatmap and the coverage map are rasters over the survey. Three
things can be true of a cell, and a raster has to keep them apart:

* **a value** - animals per hectare, frames that saw the ground;
* **zero** - the ground was looked at and there was nothing: no animal, or
  (coverage, inside the survey area) no frame. That is a result;
* **no data** - outside what the survey is about, or not known.

Both maps used to write the zeros as no data. Ground that was surveyed and
empty then looked like ground that was never surveyed - transparent on the
map - and every statistic taken from the raster (QGIS' zonal statistics, the
layer's own mean) left it out: the mean density of a survey area came out as
the mean of the places where animals were, and a gap in the coverage of a
survey area was indistinguishable from the world outside it.

``numpy`` is imported where it is needed, so the module imports without it.
"""

from typing import Optional, Tuple

#: No-data value of the coverage map (unsigned 16 bit; 0 is a count).
COVERAGE_NODATA = 65535
#: No-data value of the density heatmap (32 bit float; 0.0 is a density).
DENSITY_NODATA = -9999.0


def snap_bounds(min_x: float, max_y: float, width: int, height: int,
                cell_size: float) -> Tuple[float, float, float, float]:
    """``(min_x, min_y, max_x, max_y)`` of a grid of *width* x *height*
    cells of *cell_size*, counted from its top left corner.

    The bounds a raster is written with have to be the grid's own. Written
    with the extent of the data instead - which is not a whole number of
    cells - every cell came out a little smaller than the one the animals
    had been counted into, and slid up to a cell towards the top left.
    """
    return (min_x, max_y - height * cell_size,
            min_x + width * cell_size, max_y)


def density_raster(density, signal, looked_at=None):
    """The density heatmap as it is written.

    :param density: animals per hectare, per cell
    :param signal: where the kernel reaches (a density above "nothing")
    :param looked_at: boolean raster of the ground known to have been
        surveyed - the survey area, the cameras' footprints - or ``None``
        when that is not known
    :return: float32 raster. With *looked_at*: the density inside, zeros
        included, and no data outside. Without: the density where there is
        a signal and no data elsewhere - "no animal" cannot be told from
        "not looked at" then, and is not claimed.
    """
    import numpy as np

    density = np.asarray(density, dtype=np.float64)
    if looked_at is None:
        keep = np.asarray(signal, dtype=bool)
    else:
        keep = np.asarray(looked_at, dtype=bool)
    return np.where(keep, np.maximum(density, 0.0),
                    DENSITY_NODATA).astype(np.float32)


def coverage_raster(count, inside=None, excluded=None):
    """The coverage map as it is written.

    :param count: frames that saw each cell
    :param inside: boolean raster of the survey plot (minus its exclusion
        zones), or ``None`` when no plot is drawn
    :param excluded: boolean raster of ground that is no part of the survey
        although no plot is drawn (exclusion zones only), or ``None``
    :return: unsigned 16 bit raster. With a plot: the count inside it - **0
        where no frame saw the ground, a gap in the survey** - and no data
        outside. Without: the count where a frame saw the ground and no
        data elsewhere, since nothing says that ground was to be surveyed.
    """
    import numpy as np

    count = np.clip(np.asarray(count), 0, COVERAGE_NODATA - 1)
    if inside is not None:
        keep = np.asarray(inside, dtype=bool)
    else:
        keep = count > 0
        if excluded is not None:
            keep &= ~np.asarray(excluded, dtype=bool)
    return np.where(keep, count, COVERAGE_NODATA).astype(np.uint16)


def grow_to_bounds(array, transform, bounds, fill=0):
    """*array* and its transform, extended by whole cells until *bounds*
    ``(min_x, min_y, max_x, max_y)`` lie inside. Nothing is cut off: an
    array that already covers them comes back as it is.

    The coverage map is laid out over the frames. A survey area reaching
    beyond them has ground no frame saw - which is what the map is for, and
    has to be on it.
    """
    import math

    import numpy as np

    array = np.asarray(array)
    height, width = array.shape
    cell_x, cell_y = transform.a, -transform.e
    left, top = transform.c, transform.f
    right, bottom = left + width * cell_x, top - height * cell_y
    min_x, min_y, max_x, max_y = bounds

    def cells(distance: float, cell: float) -> int:
        # a hair of tolerance: bounds that are the grid's own add nothing
        return max(0, int(math.ceil(distance / cell - 1e-9)))

    west = cells(left - min_x, cell_x)
    east = cells(max_x - right, cell_x)
    north = cells(max_y - top, cell_y)
    south = cells(bottom - min_y, cell_y)
    if not (west or east or north or south):
        return array, transform

    grown = np.full((height + north + south, width + west + east), fill,
                    dtype=array.dtype)
    grown[north:north + height, west:west + width] = array
    moved = type(transform)(
        transform.a, transform.b, left - west * cell_x,
        transform.d, transform.e, top + north * cell_y)
    return grown, moved


def looked_at_geometry(geometries) -> Optional[object]:
    """The ground a survey is known to have looked at, from what is known
    for each of its flights - or ``None`` as soon as that is unknown for one
    of them (its animals would lie on ground the map calls unsurveyed)."""
    import shapely

    geometries = list(geometries)
    if not geometries or any(g is None for g in geometries):
        return None
    union = shapely.unary_union([g for g in geometries if not g.is_empty])
    return None if union.is_empty else union
