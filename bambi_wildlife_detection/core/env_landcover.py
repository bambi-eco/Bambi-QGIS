# -*- coding: utf-8 -*-
"""Land cover and tree cover from ESA WorldCover.

WorldCover is a global 10 m land-cover map (2021, v200) published as
cloud-optimised GeoTIFFs in 3 x 3 degree tiles that need no account. It is
static - one map for the year, not for the flight's date - which is what a
habitat description wants: meadow, forest, cropland and water do not move
between flights.

Two layers come out of it on the flight's grid (:mod:`core.environment`):

``landcover``
    The WorldCover class of every cell (see :data:`CLASSES`).
``tree_cover``
    The share of tree-covered cells within :data:`TREE_COVER_RADIUS_M` of
    each cell, in percent. A map of 10 m cells is either tree or not; the
    neighbourhood share is what tells closed forest from a hedge, and what
    the canopy does to detectability.
"""

import math
from typing import Callable, Dict, List, Optional, Tuple

from . import env_remote
from .environment import (EnvironmentFailure, Grid, check_cancelled,
                          grid_bounds_wgs84)

VERSION = "v200"
YEAR = 2021
BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
ATTRIBUTION = ("ESA WorldCover 10 m 2021 v200 (c) ESA WorldCover project, "
               "CC BY 4.0. Contains modified Copernicus Sentinel data.")

TREE_CLASS = 10
NO_DATA = 0

#: ``class value -> (name, (r, g, b))`` - the map's own legend.
CLASSES: Dict[int, Tuple[str, Tuple[int, int, int]]] = {
    10: ("Tree cover", (0, 100, 0)),
    20: ("Shrubland", (255, 187, 34)),
    30: ("Grassland", (255, 255, 76)),
    40: ("Cropland", (240, 150, 255)),
    50: ("Built-up", (250, 0, 0)),
    60: ("Bare / sparse vegetation", (180, 180, 180)),
    70: ("Snow and ice", (240, 240, 240)),
    80: ("Permanent water bodies", (0, 100, 200)),
    90: ("Herbaceous wetland", (0, 150, 160)),
    95: ("Mangroves", (0, 207, 117)),
    100: ("Moss and lichen", (250, 230, 160)),
}

#: Neighbourhood of the tree-cover layer, in metres.
TREE_COVER_RADIUS_M = 50.0

TILE_DEGREES = 3


def class_name(value) -> str:
    try:
        return CLASSES[int(value)][0]
    except (KeyError, TypeError, ValueError):
        return "No data" if value in (None, NO_DATA) else f"Class {value}"


def colormap() -> Dict[int, Tuple[int, int, int, int]]:
    """The legend as a GeoTIFF colour table."""
    table = {value: (r, g, b, 255) for value, (_n, (r, g, b)) in CLASSES.items()}
    table[NO_DATA] = (0, 0, 0, 0)
    return table


def tile_name(lat: float, lon: float) -> str:
    """Name of the 3-degree tile holding a position, e.g. ``N48E012``."""
    south = int(math.floor(lat / TILE_DEGREES) * TILE_DEGREES)
    west = int(math.floor(lon / TILE_DEGREES) * TILE_DEGREES)
    return (f"{'N' if south >= 0 else 'S'}{abs(south):02d}"
            f"{'E' if west >= 0 else 'W'}{abs(west):03d}")


def tiles_for(bounds_wgs84) -> List[str]:
    """Every tile a ``(west, south, east, north)`` box touches."""
    west, south, east, north = bounds_wgs84
    names = []
    lat = math.floor(south / TILE_DEGREES) * TILE_DEGREES
    while lat <= north:
        lon = math.floor(west / TILE_DEGREES) * TILE_DEGREES
        while lon <= east:
            name = tile_name(lat + 0.5, lon + 0.5)
            if name not in names:
                names.append(name)
            lon += TILE_DEGREES
        lat += TILE_DEGREES
    return names


def tile_url(name: str) -> str:
    return (f"{BASE_URL}/{VERSION}/{YEAR}/map/"
            f"ESA_WorldCover_10m_{YEAR}_{VERSION}_{name}_Map.tif")


def focal_fraction(mask, radius_cells: int):
    """Share of true cells in the square of side ``2 * radius + 1`` around
    each cell, counting only cells inside the array."""
    import numpy as np

    mask = np.asarray(mask, dtype=np.float64)
    r = max(0, int(radius_cells))
    if r == 0:
        return mask.copy()

    def box_sum(values):
        padded = np.pad(values, r, mode="constant")
        summed = np.cumsum(np.cumsum(padded, axis=0), axis=1)
        summed = np.pad(summed, ((1, 0), (1, 0)), mode="constant")
        size = 2 * r + 1
        below = summed[size:, size:] - summed[:-size, size:]
        left = summed[size:, :-size] - summed[:-size, :-size]
        return below - left

    counts = box_sum(np.ones_like(mask))
    return box_sum(mask) / counts


def tree_cover(classes, resolution: float,
               radius_m: float = TREE_COVER_RADIUS_M):
    """Tree share around every cell in percent; NaN where the map is empty."""
    import numpy as np

    classes = np.asarray(classes)
    valid = classes != NO_DATA
    radius = int(round(radius_m / resolution))
    trees = focal_fraction((classes == TREE_CLASS) & valid, radius)
    known = focal_fraction(valid, radius)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(known > 0, trees / known * 100.0, np.nan)
    return np.where(valid, share, np.nan).astype(np.float32)


def composition(classes) -> Dict[str, float]:
    """``class name -> share`` of the mapped cells, largest first."""
    import numpy as np

    classes = np.asarray(classes)
    values, counts = np.unique(classes[classes != NO_DATA], return_counts=True)
    total = float(counts.sum())
    if not total:
        return {}
    shares = {class_name(v): round(float(c) / total, 4)
              for v, c in zip(values, counts)}
    return dict(sorted(shares.items(), key=lambda kv: -kv[1]))


def fetch(grid: Grid, read: Callable = env_remote.read_to_grid,
          log: Optional[Callable[[str], None]] = None,
          cancelled: Optional[Callable[[], bool]] = None,
          progress: Optional[Callable[[int], None]] = None,
          ) -> Tuple[Dict[str, "object"], dict]:
    """``({"landcover": classes, "tree_cover": percent}, metadata)``."""
    import numpy as np

    names = tiles_for(grid_bounds_wgs84(grid))
    classes = np.zeros(grid.shape, dtype=np.uint8)
    for index, name in enumerate(names):
        check_cancelled(cancelled)
        if log:
            log(f"Reading WorldCover tile {name}")
        tile, _nodata = read(tile_url(name), grid, "nearest")
        tile = np.asarray(tile).astype(np.uint8)
        classes = np.where(classes == NO_DATA, tile, classes)
        if progress:
            progress(int(90 * (index + 1) / len(names)))
    if not np.any(classes != NO_DATA):
        raise EnvironmentFailure(
            "WorldCover has no data over the flight (open sea, or outside "
            "the map).")
    shares = composition(classes)
    metadata = {
        "product": f"ESA WorldCover 10 m {YEAR} {VERSION}",
        "attribution": ATTRIBUTION,
        "tiles": names,
        "tree_cover_radius_m": TREE_COVER_RADIUS_M,
        "composition": shares,
        "classes": {str(value): name for value, (name, _c) in CLASSES.items()},
    }
    if log:
        log("Land cover: " + ", ".join(
            f"{name} {share * 100:.0f}%" for name, share in
            list(shares.items())[:5]))
    return {"landcover": classes,
            "tree_cover": tree_cover(classes, grid.resolution)}, metadata
