# -*- coding: utf-8 -*-
"""Elevation for anywhere: the Copernicus DEM.

The pipeline places every detection on an elevation model. Where there is a
national LiDAR model - Austria's is fetched automatically - that is the one
to use. Everywhere else there is the Copernicus DEM: a model of the whole
land surface of the earth at 30 m, published as cloud-optimised GeoTIFFs
that need no account.

**What it is.** The *GLO-30* edition has cells of one arc second (about
30 m); a few countries are missing from its public release and are covered
by *GLO-90* (three arc seconds), which is used for a tile GLO-30 does not
have. Heights are metres above the EGM2008 geoid - above sea level, as a
drone's log gives its own.

**What it is not.** It is a *surface* model, measured by radar from space:
over forest it lies somewhere in the canopy, over a village on the roofs,
and its heights are good to a few metres, not to decimetres. An animal on
the forest floor is placed on a surface up to a tree's height above it, and
what is seen at an angle shifts sideways by that height times the tangent
of the angle. Open ground is far better served than forest - say so in the
position uncertainty (*Terrain model*), and prefer a terrain model where
there is one.

**How it is read.** Only the part under the flight is transferred: the
tiles are read through HTTP range requests onto the flight's own grid
(:func:`core.env_remote.read_to_grid`). A tile that does not exist is open
sea; where land and sea share the flight's grid the sea is at height
nought, and where there is no tile at all there is nothing to build.

numpy and requests are imported inside the functions that need them. Free
of QGIS and Qt.
"""

import math
from typing import Callable, List, Optional, Tuple

from . import env_remote
from .environment import (EnvironmentFailure, Grid, check_cancelled,
                          grid_bounds_wgs84)

#: The editions, the finer first: ``(name, address, arc seconds in the
#: tile's name, size of a cell in metres)``.
EDITIONS = (
    ("GLO-30", "https://copernicus-dem-30m.s3.amazonaws.com", 10, 30.0),
    ("GLO-90", "https://copernicus-dem-90m.s3.amazonaws.com", 30, 90.0),
)
ATTRIBUTION = (
    "Copernicus DEM GLO-30 / GLO-90 (c) DLR e.V. 2010-2014 and (c) Airbus "
    "Defence and Space GmbH 2014-2018, provided under COPERNICUS by the "
    "European Union and ESA; all rights reserved.")
VERTICAL_DATUM = "EGM2008"

#: The cell size the heights are put on, metres: that of the finer edition.
RESOLUTION_M = 30.0
#: Ground kept around the flight so that what a camera sees beyond the
#: route still meets the model: at least this, whatever padding is asked.
MIN_PADDING_M = 150.0
#: A tile's edges lie half a cell off the whole degree; a box is widened by
#: this before its tiles are named, so that none is missed.
_EDGE_DEGREES = 0.001


class NoElevation(EnvironmentFailure):
    """There is no elevation data where it was asked for."""


def tile_stem(lat: int, lon: int, arcseconds: int = 10) -> str:
    """The name of the tile whose south-west corner is at *lat*, *lon*
    (whole degrees), e.g. ``Copernicus_DSM_COG_10_N47_00_E013_00_DEM``."""
    north = "N" if lat >= 0 else "S"
    east = "E" if lon >= 0 else "W"
    return (f"Copernicus_DSM_COG_{arcseconds}_{north}{abs(lat):02d}_00_"
            f"{east}{abs(lon):03d}_00_DEM")


def tile_url(lat: int, lon: int, edition: int = 0) -> str:
    """Where the tile of an edition (index into :data:`EDITIONS`) is."""
    _name, base, arcseconds, _size = EDITIONS[edition]
    stem = tile_stem(lat, lon, arcseconds)
    return f"{base}/{stem}/{stem}.tif"


def tiles_for(bounds_wgs84) -> List[Tuple[int, int]]:
    """The south-west corners (whole degrees) of every tile a
    ``(west, south, east, north)`` box touches - or all but touches."""
    west, south, east, north = bounds_wgs84
    west, east = west - _EDGE_DEGREES, east + _EDGE_DEGREES
    south, north = south - _EDGE_DEGREES, north + _EDGE_DEGREES
    corners = []
    for lat in range(int(math.floor(south)), int(math.floor(north)) + 1):
        if not -90 <= lat < 90:
            continue
        for lon in range(int(math.floor(west)), int(math.floor(east)) + 1):
            wrapped = (lon + 180) % 360 - 180
            if (lat, wrapped) not in corners:
                corners.append((lat, wrapped))
    return corners


def exists(url: str) -> bool:
    """Whether a tile is there. A tile that is not is open sea - or a
    country an edition leaves out; anything else that goes wrong is the
    connection's, and raised.

    :raises core.env_remote.SourceUnavailable: no answer, or one that is
        neither "here it is" nor "there is none"
    """
    return env_remote.exists(url, "the Copernicus DEM")


def fetch(grid: Grid, read: Callable = env_remote.read_to_grid,
          there: Callable[[str], bool] = exists,
          log: Optional[Callable[[str], None]] = None,
          cancelled: Optional[Callable[[], bool]] = None,
          progress: Optional[Callable[[int], None]] = None):
    """The heights under *grid*: ``(heights, metadata)``.

    :param read: ``read(url, grid, "bilinear", fill=nan) -> (array, fill)``
    :param there: ``there(url) -> bool`` - whether a tile exists
    :return: a float32 array of ``grid.shape`` - sea, where the grid
        reaches it, at height nought - and ``{"product", "editions",
        "tiles", "sea_share", "min_m", "max_m", "attribution",
        "vertical_datum", "surface_model"}``
    :raises NoElevation: no tile holds any of the grid
    """
    import numpy as np

    corners = tiles_for(grid_bounds_wgs84(grid))
    heights = np.full(grid.shape, np.nan, dtype=np.float32)
    used, editions = [], []
    for index, (lat, lon) in enumerate(corners):
        check_cancelled(cancelled)
        for edition, (name, _base, arcseconds, _size) in enumerate(EDITIONS):
            url = tile_url(lat, lon, edition)
            if not there(url):
                continue
            stem = tile_stem(lat, lon, arcseconds)
            if log:
                log(f"Reading Copernicus DEM tile {stem} ({name})")
            tile, _fill = read(url, grid, "bilinear", fill=float("nan"))
            tile = np.asarray(tile, dtype=np.float32)
            heights = np.where(np.isnan(heights), tile, heights)
            if np.any(np.isfinite(tile)):
                used.append(stem)
                if name not in editions:
                    editions.append(name)
            break
        if progress:
            progress(int(90 * (index + 1) / len(corners)))
    if not np.any(np.isfinite(heights)):
        raise NoElevation(
            "The Copernicus DEM has no heights where the flight is: open "
            "sea, or a position that is not the flight's. Over water use "
            "'Flat Surface' instead.")
    sea = np.isnan(heights)
    heights = np.where(sea, np.float32(0.0), heights).astype(np.float32)
    metadata = {
        "product": "Copernicus DEM " + " + ".join(editions),
        "editions": editions,
        "tiles": used,
        "sea_share": float(sea.mean()),
        "min_m": float(heights.min()),
        "max_m": float(heights.max()),
        "attribution": ATTRIBUTION,
        "vertical_datum": VERTICAL_DATUM,
        "surface_model": True,
    }
    if log:
        note = ""
        if metadata["sea_share"] > 0:
            note = (f", {metadata['sea_share'] * 100:.0f} % of it without "
                    "data - taken as sea, at height nought")
        log(f"{metadata['product']}: {metadata['min_m']:.0f} to "
            f"{metadata['max_m']:.0f} m above sea level{note}")
        if "GLO-90" in editions:
            log("GLO-30 has no tile here: the 90 m edition is used, which "
                "is coarser")
    return heights, metadata


def grid_for(bounds, epsg: int, padding_m: float = MIN_PADDING_M) -> Grid:
    """The grid the heights of a flight are put on.

    :param bounds: ``(min x, min y, max x, max y)`` of the flight's
        positions in the CRS *epsg*
    :param padding_m: ground kept around them; never less than
        :data:`MIN_PADDING_M`, which is five cells
    """
    from .environment import make_grid

    return make_grid(bounds, epsg, max(float(padding_m), MIN_PADDING_M),
                     RESOLUTION_M)
