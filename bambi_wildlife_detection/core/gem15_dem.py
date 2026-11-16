# -*- coding: utf-8 -*-
"""Bare ground at 15 m for most of the world: Google's GEM-15.

The Copernicus DEM (:mod:`core.copernicus_dem`) is a *surface* model: over
forest it lies in the canopy, a tree's height above the animal that stands
under it. GEM-15 (Google Research, 2026) is the first openly readable
*terrain* model with near-global cover: the bare ground at 15 m, predicted
from sub-metre stereo satellite imagery, published as cloud-optimised
GeoTIFFs that need no account.

**What it is good for.** Against Austria's laser-scanned 1 m terrain model,
on eight areas of 2 km by 2 km (fields, villages, hill and mountain
forest; November 2026), the heights were off by, root mean square:

==========================  ========  ===========
                              GEM-15   Copernicus
==========================  ========  ===========
all eight, as published        6.1 m        9.1 m
all eight, levelled            5.6 m            -
under trees (seven areas)   6 - 10 m    11 - 19 m
flat open fields, levelled     1.3 m        1.6 m
==========================  ========  ===========

**What it is not.** It is experimental, and three things have to be known:

* It is *low*. In all eight areas the terrain model lay 1.4 to 5.3 m below
  the laser scan - the publishers name regional offsets of -4.8 to +1.9 m
  themselves and say "debias against local control". The control used here
  is the Copernicus DEM, which is level with the ground wherever nothing
  stands on it: on *open ground* - cells where GEM-15's own surface and
  terrain models agree to within a metre - the median difference between
  the two is taken for GEM-15's offset and added (:func:`level_offset`).
  That brought the offset down from 3.0 m to 0.8 m on average, and to
  within 0.9 m in seven of the eight areas. It cannot help where the
  terrain model itself still has canopy in it: in the eighth, a closed
  deciduous forest, levelling left the model 3.7 m high.
* It has *gaps*: water with a margin of 9 m, snow and ice, and other
  surfaces a stereo matcher finds nothing on are left without a value.
  They are filled from the Copernicus DEM.
* It does not cover *everything*: 60 deg S to 84 deg N, without Russia,
  China, Mongolia, Greenland and Antarctica. Where it has nothing, the
  caller falls back to the Copernicus DEM (:class:`NoTerrain`).

Heights are metres above the EGM96 geoid as published; once levelled they
are on the Copernicus DEM's EGM2008, which differs by decimetres.

**How it is read.** One file per degree cell, in the cell's UTM zone, read
through HTTP range requests onto the flight's own grid like every other
remote raster (:func:`core.env_remote.read_to_grid`).

numpy and requests are imported inside the functions that need them. Free
of QGIS and Qt.
"""

import math
from typing import Callable, List, Optional, Tuple

from . import copernicus_dem, env_remote
from .environment import Grid, check_cancelled, grid_bounds_wgs84

#: Where the cloud-optimised GeoTIFFs are.
BASE_URL = "https://storage.googleapis.com/climate-and-sustainability/gem-15"
#: The layers of a cell: the bare ground, and the surface with what stands
#: on it.
TERRAIN, SURFACE = "dtm", "dsm"

#: The attribution the licence (CC BY 4.0) asks for, and the paper.
ATTRIBUTION = (
    "GEM-15: This dataset is produced by Google (CC BY 4.0). Batchu, V. V., "
    "Neumann, M., Li, T., Allen, B., Zhang, J., & Van Arsdale, C. (2026): "
    "GEM-15: Global surface and terrain elevation models (15 meters) from "
    "sub-meter optical stereo imagery. EarthArXiv, "
    "https://doi.org/10.31223/X55N75")
PRODUCT = "Google GEM-15 terrain model"
VERTICAL_DATUM = "EGM96"

#: The cell size of the model, metres.
RESOLUTION_M = 15.0
#: Ground kept around the flight, at least (ten cells).
MIN_PADDING_M = 150.0

#: Open ground: where the surface model is no more than this above the
#: terrain model, metres - nothing stands there.
OPEN_GROUND_M = 1.0
#: An offset is taken from at least this many cells of open ground...
MIN_OPEN_CELLS = 50
#: ...and is believed up to this size, metres. The publishers name offsets
#: of a few metres; ten times that is something else gone wrong.
MAX_OFFSET_M = 25.0


class NoTerrain(copernicus_dem.NoElevation):
    """GEM-15 has no heights where they were asked for."""


# ---------------------------------------------------------------------------
# Where a cell's file is
# ---------------------------------------------------------------------------

def _signed(value: int) -> str:
    """``14`` or ``m91``: the publishers write a minus as an ``m``."""
    return f"m{abs(value)}" if value < 0 else str(value)


def group_id(lat: int, lon: int) -> str:
    """The folder of the ten-degree square a cell lies in, e.g.
    ``p40_p10`` for 47 N 14 E and ``m10_m100`` for 1 S 91 W."""
    south = int(math.floor(lat / 10.0)) * 10
    west = int(math.floor(lon / 10.0)) * 10
    return (f"{'m' if south < 0 else 'p'}{abs(south)}_"
            f"{'m' if west < 0 else 'p'}{abs(west)}")


def cell_id(lat: int, lon: int) -> str:
    """The name of the cell whose south-west corner is at *lat*, *lon*
    (whole degrees), e.g. ``w14s47`` and ``wm91sm1``."""
    return f"w{_signed(int(lon))}s{_signed(int(lat))}"


def tile_url(lat: int, lon: int, layer: str = TERRAIN) -> str:
    """Where a layer of the cell with its south-west corner at *lat*,
    *lon* is."""
    return (f"{BASE_URL}/{group_id(lat, lon)}/{cell_id(lat, lon)}/{layer}/"
            "zall/mosaic.tif")


def exists(url: str) -> bool:
    """Whether a cell's file is there (see :func:`core.env_remote.exists`)."""
    return env_remote.exists(url, "Google's GEM-15 elevation model")


# ---------------------------------------------------------------------------
# Levelling
# ---------------------------------------------------------------------------

def level_offset(terrain, surface, reference) -> Tuple[Optional[float], int]:
    """What has to be added to GEM-15 to bring it level with *reference*
    (the Copernicus DEM, on the same cells): ``(offset in metres, cells of
    open ground it was taken from)``.

    On open ground - where *surface* is no more than :data:`OPEN_GROUND_M`
    above *terrain* - both models and the reference describe the same
    thing, the ground, and what is left between them is GEM-15's regional
    offset. The offset is the median over those cells, of *reference*
    against *surface*.

    ``(None, n)`` where there is too little open ground to tell
    (:data:`MIN_OPEN_CELLS`), or the answer is beyond belief
    (:data:`MAX_OFFSET_M`).
    """
    import numpy as np

    terrain = np.asarray(terrain, dtype=np.float64)
    surface = np.asarray(surface, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    known = np.isfinite(terrain) & np.isfinite(surface)
    known &= np.isfinite(reference)
    with np.errstate(invalid="ignore"):
        open_ground = known & ((surface - terrain) < OPEN_GROUND_M)
    cells = int(open_ground.sum())
    if cells < MIN_OPEN_CELLS:
        return None, cells
    offset = float(np.median((reference - surface)[open_ground]))
    if not math.isfinite(offset) or abs(offset) > MAX_OFFSET_M:
        return None, cells
    return offset, cells


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _mosaic(grid: Grid, corners, layer: str, read: Callable,
            there: Callable[[str], bool], cancelled, log=None,
            done: Optional[Callable[[], None]] = None):
    """The cells of *layer* that are there, put together on *grid*:
    ``(heights with NaN where nothing is, names of the cells used)``."""
    import numpy as np

    heights = np.full(grid.shape, np.nan, dtype=np.float32)
    used: List[str] = []
    for lat, lon in corners:
        check_cancelled(cancelled)
        url = tile_url(lat, lon, layer)
        if there(url):
            if log:
                log(f"Reading GEM-15 cell {cell_id(lat, lon)} ({layer})")
            tile, _fill = read(url, grid, "bilinear", fill=float("nan"))
            tile = np.asarray(tile, dtype=np.float32)
            if np.any(np.isfinite(tile)):
                heights = np.where(np.isnan(heights), tile, heights)
                used.append(cell_id(lat, lon))
        if done:
            done()
    return heights, used


def fetch(grid: Grid, read: Callable = env_remote.read_to_grid,
          there: Callable[[str], bool] = exists,
          reference: Optional[Callable] = None,
          log: Optional[Callable[[str], None]] = None,
          cancelled: Optional[Callable[[], bool]] = None,
          progress: Optional[Callable[[int], None]] = None):
    """The ground under *grid*: ``(heights, metadata)``.

    GEM-15's terrain model, levelled to the Copernicus DEM on open ground
    and with its gaps filled from it.

    :param read: ``read(url, grid, "bilinear", fill=nan) -> (array, fill)``
    :param there: ``there(url) -> bool`` - whether a cell's file exists
    :param reference: ``reference(grid, log=, cancelled=) -> (heights,
        metadata)`` - the model GEM-15 is levelled to and filled from;
        :func:`core.copernicus_dem.fetch` unless given. It may raise
        :class:`core.copernicus_dem.NoElevation`: then nothing is levelled
        and gaps are closed from the nearest cell that has a height.
    :return: a float32 array of ``grid.shape`` without gaps, and
        ``{"product", "tiles", "min_m", "max_m", "attribution",
        "vertical_datum", "surface_model", "resolution_m",
        "levelled_by_m", "open_ground_cells", "filled_share",
        "filled_from", "sea_share"}``
    :raises NoTerrain: GEM-15 holds nothing of the grid
    """
    import numpy as np

    corners = copernicus_dem.tiles_for(grid_bounds_wgs84(grid))
    steps = {"done": 0, "of": 2 * len(corners) + 1}

    def step() -> None:
        steps["done"] += 1
        if progress:
            progress(int(90 * steps["done"] / steps["of"]))

    terrain, used = _mosaic(grid, corners, TERRAIN, read, there, cancelled,
                            log, step)
    if not used:
        raise NoTerrain(
            "GEM-15 has no heights where the flight is: outside 60 deg S "
            "to 84 deg N, in a country it leaves out (Russia, China, "
            "Mongolia, Greenland), or over open water.")
    surface, _names = _mosaic(grid, corners, SURFACE, read, there, cancelled,
                              None, step)

    check_cancelled(cancelled)
    reference = reference or copernicus_dem.fetch
    other, other_meta = None, {}
    try:
        other, other_meta = reference(grid, log=None, cancelled=cancelled)
        other = np.asarray(other, dtype=np.float32)
    except copernicus_dem.NoElevation:
        if log:
            log("The Copernicus DEM has nothing here to level GEM-15 by: "
                "its heights are taken as published")
    step()

    offset, open_cells = None, 0
    if other is not None:
        offset, open_cells = level_offset(terrain, surface, other)
    heights = terrain + np.float32(offset or 0.0)

    gaps = ~np.isfinite(heights)
    filled_from = None
    if gaps.any():
        if other is not None:
            heights = np.where(gaps, other, heights)
            filled_from = other_meta.get("product") or "Copernicus DEM"
        else:
            heights = _nearest_filled(heights)
            filled_from = "the nearest cells with a height"
    heights = heights.astype(np.float32)

    levelled = offset is not None
    product, attribution, datum = PRODUCT, ATTRIBUTION, VERTICAL_DATUM
    if levelled:
        product += ", levelled to the Copernicus DEM"
        datum = other_meta.get("vertical_datum", copernicus_dem.VERTICAL_DATUM)
    # whoever's heights are in the model is named with it
    from_the_other = levelled or (other is not None and bool(gaps.any()))
    if from_the_other and other_meta.get("attribution"):
        attribution += "; " + other_meta["attribution"]
    metadata = {
        "product": product,
        "tiles": used,
        "min_m": float(heights.min()),
        "max_m": float(heights.max()),
        "attribution": attribution,
        "vertical_datum": datum,
        "surface_model": False,
        "resolution_m": RESOLUTION_M,
        "levelled_by_m": None if offset is None else round(offset, 2),
        "open_ground_cells": open_cells,
        "filled_share": round(float(gaps.mean()), 4),
        "filled_from": filled_from,
        "sea_share": float(other_meta.get("sea_share") or 0.0),
    }
    if log:
        log(f"{PRODUCT}: {metadata['min_m']:.0f} to {metadata['max_m']:.0f} "
            "m above sea level")
        if levelled:
            log(f"Levelled to the Copernicus DEM: {offset:+.1f} m, from "
                f"{open_cells} cells of open ground (GEM-15 is known to lie "
                "a few metres off, region by region)")
        elif other is not None:
            log("Not levelled: too little open ground under the flight to "
                "tell GEM-15's offset from the Copernicus DEM "
                f"({open_cells} cells). Its heights are as published, and "
                "may lie a few metres low or high.")
        if filled_from:
            log(f"{metadata['filled_share'] * 100:.0f} % of the cells have "
                "no GEM-15 height (water, snow, surfaces without texture): "
                f"taken from {filled_from}")
    return heights, metadata


def _nearest_filled(heights):
    """*heights* with every gap given the value of the nearest cell that
    has one."""
    import numpy as np
    from scipy import ndimage

    gaps = ~np.isfinite(heights)
    if not gaps.any() or gaps.all():
        return heights
    _distance, (rows, cols) = ndimage.distance_transform_edt(
        gaps, return_indices=True)
    return heights[rows, cols]


def grid_for(bounds, epsg: int, padding_m: float = MIN_PADDING_M) -> Grid:
    """The grid the heights of a flight are put on: cells of 15 m.

    :param bounds: ``(min x, min y, max x, max y)`` of the flight's
        positions in the CRS *epsg*
    :param padding_m: ground kept around them; never less than
        :data:`MIN_PADDING_M`
    """
    from .environment import make_grid

    return make_grid(bounds, epsg, max(float(padding_m), MIN_PADDING_M),
                     RESOLUTION_M)
