# -*- coding: utf-8 -*-
"""Environmental context of a flight: what the ground and the weather were.

Publicly available data for the area a flight monitored, on (or near) the day
it flew - the things that may explain where animals were and how well a
camera could see them. Four independent *sources*, each its own pipeline
sub-step that can be run, re-run and fail on its own:

``sentinel2``
    Greenness (NDVI), vegetation moisture (NDMI), open water (NDWI) and snow
    (NDSI) from the Sentinel-2 scene nearest to the flight
    (:mod:`core.env_sentinel2`).
``landcover``
    ESA WorldCover land-cover classes and the tree cover derived from them
    (:mod:`core.env_landcover`).
``terrain``
    Topographic wetness index from the flight's own DEM
    (:mod:`core.env_terrain`).
``weather``
    Hourly weather at flight time and over the 24 hours before it
    (:mod:`core.env_weather`).

Everything lands in ``<flight>/environment/``: one GeoTIFF per layer in the
flight's CRS on a shared 10 m grid, and one ``<source>.json`` per source
saying what was fetched, from where and for which extent. A source counts as
done when its JSON and every file it names are there - the files win, as
everywhere else in the plugin.

The environment belongs to the flight, not to a camera: thermal and RGB look
at the same meadow.

This module holds what the sources share: the extent, the grid, the flight's
time, the layer catalogue and the metadata files. No network access here.
"""

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

FOLDER = "environment"

SENTINEL2 = "sentinel2"
LANDCOVER = "landcover"
TERRAIN = "terrain"
WEATHER = "weather"
SOURCES = (SENTINEL2, LANDCOVER, TERRAIN, WEATHER)

SOURCE_TITLES = {
    SENTINEL2: "Satellite indices (Sentinel-2)",
    LANDCOVER: "Land cover (ESA WorldCover)",
    TERRAIN: "Terrain wetness (DEM)",
    WEATHER: "Weather (Open-Meteo)",
}

#: Cell size of every environment raster, in metres. Sentinel-2's finest
#: bands and WorldCover are both 10 m; anything finer would only resample.
RESOLUTION_M = 10.0

DEFAULT_BUFFER_M = 200.0
DEFAULT_RADIUS_M = 30.0

CONTINUOUS = "continuous"
CATEGORICAL = "categorical"


@dataclass(frozen=True)
class Layer:
    """One environment raster: where it is and how to read it."""

    key: str
    source: str
    title: str
    kind: str = CONTINUOUS
    unit: str = ""
    #: Value range a chart or a colour ramp should span; ``None`` = the data's.
    value_range: Optional[Tuple[float, float]] = None
    description: str = ""

    @property
    def filename(self) -> str:
        return f"{self.key}.tif"


LAYERS: Tuple[Layer, ...] = (
    Layer("ndvi", SENTINEL2, "NDVI (greenness)", CONTINUOUS, "",
          (-0.2, 1.0),
          "Normalised difference vegetation index: dense green vegetation "
          "is high, bare ground and water are low."),
    Layer("ndmi", SENTINEL2, "NDMI (vegetation moisture)", CONTINUOUS, "",
          (-0.6, 0.8),
          "Normalised difference moisture index: water content of the "
          "vegetation canopy. A proxy for how wet a patch is, not air "
          "humidity; under closed forest it says little about the ground."),
    Layer("ndwi", SENTINEL2, "NDWI (open water)", CONTINUOUS, "",
          (-1.0, 1.0),
          "Normalised difference water index: positive over open water."),
    Layer("ndsi", SENTINEL2, "NDSI (snow)", CONTINUOUS, "", (-1.0, 1.0),
          "Normalised difference snow index: above about 0.4 over snow."),
    Layer("landcover", LANDCOVER, "Land cover", CATEGORICAL, "", None,
          "ESA WorldCover 2021 class per 10 m cell."),
    Layer("tree_cover", LANDCOVER, "Tree cover", CONTINUOUS, "%",
          (0.0, 100.0),
          "Share of tree-covered cells within 50 m, from the WorldCover "
          "tree class."),
    Layer("twi", TERRAIN, "Topographic wetness index", CONTINUOUS, "",
          None,
          "Where water tends to collect, from the terrain alone: high in "
          "flat hollows and valley floors, low on ridges and steep slopes."),
)

LAYERS_BY_KEY: Dict[str, Layer] = {layer.key: layer for layer in LAYERS}


class EnvironmentFailure(RuntimeError):
    """An environment source cannot be produced as asked."""


class Cancelled(Exception):
    """The user stopped a source while it was running."""


def check_cancelled(cancelled) -> None:
    """Raise :class:`Cancelled` when the *cancelled* callback says so."""
    if cancelled is not None and cancelled():
        raise Cancelled()


# --------------------------------------------------------------------------- #
# Places
# --------------------------------------------------------------------------- #

def folder(target_folder: str) -> str:
    return os.path.join(target_folder, FOLDER)


def ensure_folder(target_folder: str) -> str:
    path = folder(target_folder)
    os.makedirs(path, exist_ok=True)
    return path


def layer_path(target_folder: str, key: str) -> str:
    return os.path.join(folder(target_folder), LAYERS_BY_KEY[key].filename)


def metadata_path(target_folder: str, source: str) -> str:
    return os.path.join(folder(target_folder), f"{source}.json")


def layers_of(source: str) -> List[Layer]:
    return [layer for layer in LAYERS if layer.source == source]


# --------------------------------------------------------------------------- #
# Metadata - one JSON per source
# --------------------------------------------------------------------------- #

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_metadata(target_folder: str, source: str, data: dict) -> str:
    """Record what *source* produced. Written last, so its presence means
    the source finished."""
    ensure_folder(target_folder)
    document = {"source": source, "created": utc_now()}
    document.update(data)
    path = metadata_path(target_folder, source)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2, default=str)
    os.replace(temporary, path)
    return path


def read_metadata(target_folder: str, source: str) -> Optional[dict]:
    path = metadata_path(target_folder, source)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_complete(target_folder: str, source: str) -> bool:
    """True when *source* finished and every raster it wrote is still there."""
    metadata = read_metadata(target_folder, source)
    if metadata is None:
        return False
    for name in metadata.get("files", []) or []:
        if not os.path.isfile(os.path.join(folder(target_folder), name)):
            return False
    return True


def status(target_folder: str) -> Dict[str, bool]:
    """``source -> done`` for every source."""
    if not target_folder or not os.path.isdir(target_folder):
        return {source: False for source in SOURCES}
    return {source: is_complete(target_folder, source) for source in SOURCES}


def clear(target_folder: str, source: str) -> List[str]:
    """Remove what *source* wrote, so a failed re-run leaves no stale mix of
    old rasters and new metadata. Returns the removed paths."""
    removed = []
    names = [layer.filename for layer in layers_of(source)]
    names += [f"{source}.json", f"{source}.json.tmp"]
    if source == WEATHER:
        names.append("weather_hourly.csv")
    for name in names:
        path = os.path.join(folder(target_folder), name)
        for candidate in (path, path + ".aux.xml"):
            if os.path.isfile(candidate):
                os.remove(candidate)
                removed.append(candidate)
    return removed


def available_layers(target_folder: str) -> List[Layer]:
    """The rasters that exist for a flight, in catalogue order."""
    return [layer for layer in LAYERS
            if os.path.isfile(layer_path(target_folder, layer.key))]


# --------------------------------------------------------------------------- #
# Extent and grid
# --------------------------------------------------------------------------- #

Bounds = Tuple[float, float, float, float]


def _union(bounds: Iterable[Optional[Bounds]]) -> Optional[Bounds]:
    found = [b for b in bounds if b is not None]
    if not found:
        return None
    return (min(b[0] for b in found), min(b[1] for b in found),
            max(b[2] for b in found), max(b[3] for b in found))


def _points_bounds(points: Iterable[Sequence[float]]) -> Optional[Bounds]:
    xs, ys = [], []
    for point in points:
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def fov_bounds(target_folder: str) -> Optional[Bounds]:
    """Bounds of every field-of-view footprint of the flight (both cameras)."""
    from . import fov_store, store

    found = []
    for modality in store.MODALITIES:
        if not os.path.isfile(
                store.stage_path(target_folder, store.FOV, modality)):
            continue
        try:
            polygons = fov_store.load_fov(target_folder, modality)
        except Exception:  # nosec B112 - an unreadable store is no footprint
            continue
        found.append(_points_bounds(
            point for points in polygons.values() for point in points))
    return _union(found)


def pose_bounds(target_folder: str, origin=None) -> Optional[Bounds]:
    """Bounds of the camera positions, in the project CRS.

    The poses are stored relative to the DEM mesh origin, which is added
    back; without a known origin there is no absolute position to give.
    """
    from . import poses

    if origin is None:
        return None
    found = []
    for modality in ("t", "w"):
        images = poses.load_images(target_folder, modality)
        local = _points_bounds(
            image.get("location") for image in images
            if image.get("location"))
        if local is not None:
            found.append((local[0] + origin[0], local[1] + origin[1],
                          local[2] + origin[0], local[3] + origin[1]))
    return _union(found)


def flight_footprint(target_folder: str, origin=None
                     ) -> Tuple[Optional[Bounds], str]:
    """``(bounds, how it was found)`` of the ground a flight covered.

    The field-of-view footprints when that step has run - they are the
    ground the cameras actually saw - otherwise the flight path itself.
    """
    bounds = fov_bounds(target_folder)
    if bounds is not None:
        return bounds, "field-of-view footprints"
    bounds = pose_bounds(target_folder, origin)
    if bounds is not None:
        return bounds, "flight path (no field-of-view footprints yet)"
    return None, ""


@dataclass(frozen=True)
class Grid:
    """A north-up raster grid in a metric CRS."""

    epsg: int
    min_x: float
    max_y: float
    width: int
    height: int
    resolution: float = RESOLUTION_M

    @property
    def bounds(self) -> Bounds:
        return (self.min_x, self.max_y - self.height * self.resolution,
                self.min_x + self.width * self.resolution, self.max_y)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @property
    def geotransform(self) -> Tuple[float, float, float, float, float, float]:
        """Affine coefficients ``(a, b, c, d, e, f)`` as rasterio wants them."""
        return (self.resolution, 0.0, self.min_x,
                0.0, -self.resolution, self.max_y)

    def cell_centres(self):
        """``(xs, ys)`` - x of every column and y of every row."""
        import numpy as np

        xs = self.min_x + (np.arange(self.width) + 0.5) * self.resolution
        ys = self.max_y - (np.arange(self.height) + 0.5) * self.resolution
        return xs, ys

    def expanded(self, cells: int) -> "Grid":
        """The same grid with *cells* more on every side."""
        return Grid(self.epsg, self.min_x - cells * self.resolution,
                    self.max_y + cells * self.resolution,
                    self.width + 2 * cells, self.height + 2 * cells,
                    self.resolution)

    def to_dict(self) -> dict:
        return {"epsg": self.epsg, "bounds": list(self.bounds),
                "width": self.width, "height": self.height,
                "resolution_m": self.resolution}

    @classmethod
    def from_dict(cls, data: dict) -> "Grid":
        bounds = data["bounds"]
        return cls(int(data["epsg"]), float(bounds[0]), float(bounds[3]),
                   int(data["width"]), int(data["height"]),
                   float(data.get("resolution_m", RESOLUTION_M)))


#: Largest side of an environment grid, in cells (60 km at 10 m). A flight
#: is a few kilometres; anything near this is a wrong CRS or a stray point.
MAX_GRID_CELLS = 6000


def make_grid(bounds: Bounds, epsg: int, buffer_m: float = DEFAULT_BUFFER_M,
              resolution: float = RESOLUTION_M) -> Grid:
    """The grid covering *bounds* plus *buffer_m*, snapped to whole cells.

    Snapping to multiples of the cell size puts the grid on Sentinel-2's own
    pixel grid whenever the flight shares its UTM zone, so its 10 m bands
    are copied rather than resampled.
    """
    min_x, min_y, max_x, max_y = bounds
    if not all(math.isfinite(v) for v in bounds) or max_x < min_x \
            or max_y < min_y:
        raise EnvironmentFailure(f"The flight extent {bounds} is not valid.")
    buffer_m = max(0.0, float(buffer_m))
    snapped_min_x = math.floor((min_x - buffer_m) / resolution) * resolution
    snapped_min_y = math.floor((min_y - buffer_m) / resolution) * resolution
    snapped_max_x = math.ceil((max_x + buffer_m) / resolution) * resolution
    snapped_max_y = math.ceil((max_y + buffer_m) / resolution) * resolution
    width = max(1, int(round((snapped_max_x - snapped_min_x) / resolution)))
    height = max(1, int(round((snapped_max_y - snapped_min_y) / resolution)))
    if width > MAX_GRID_CELLS or height > MAX_GRID_CELLS:
        raise EnvironmentFailure(
            f"The flight extent is {width * resolution / 1000:.0f} x "
            f"{height * resolution / 1000:.0f} km, which is too large to be "
            "one flight. Check the target CRS and the DEM origin.")
    return Grid(int(epsg), snapped_min_x, snapped_max_y, width, height,
                resolution)


def flight_grid(target_folder: str, epsg: Optional[int], origin=None,
                buffer_m: float = DEFAULT_BUFFER_M,
                area_bounds: Optional[Bounds] = None
                ) -> Tuple[Grid, str]:
    """``(grid, description of the extent)`` for a flight's environment.

    The extent is the flight's footprint, widened to include the survey
    area when one is given, plus a buffer so the surrounding habitat shows.
    """
    if not epsg:
        raise EnvironmentFailure(
            "The flight has no target CRS. Set it on the Input tab.")
    footprint, how = flight_footprint(target_folder, origin)
    bounds = _union([footprint, area_bounds])
    if bounds is None:
        raise EnvironmentFailure(
            "The flight's extent is unknown: run 'Calculate Field of View', "
            "or extract the frames with a DEM whose metadata names its "
            "origin.")
    parts = []
    if footprint is not None:
        parts.append(how)
    if area_bounds is not None:
        parts.append("survey area")
    description = " + ".join(parts) + f", {buffer_m:g} m buffer"
    return make_grid(bounds, epsg, buffer_m), description


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #

def parse_time(value) -> Optional[datetime]:
    """A stored capture time (ISO text or epoch seconds) as aware UTC."""
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except (ValueError, OverflowError, OSError):
        return None
    if parsed.tzinfo is None:
        # The pose extractors write the flight log's clock, which is UTC.
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def flight_time(target_folder: str) -> Optional[Tuple[datetime, datetime]]:
    """``(first, last)`` capture time of the flight in UTC, or ``None``."""
    from . import poses

    times = []
    for modality in ("t", "w"):
        for image in poses.load_images(target_folder, modality):
            parsed = parse_time(image.get("timestamp") or image.get("epoch"))
            if parsed is not None:
                times.append(parsed)
    if not times:
        return None
    return min(times), max(times)


def require_flight_time(target_folder: str) -> Tuple[datetime, datetime]:
    found = flight_time(target_folder)
    if found is None:
        raise EnvironmentFailure(
            "The flight's date is unknown: its frames carry no capture "
            "time. Run 'Extract Frames' first.")
    return found


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Rasters
# --------------------------------------------------------------------------- #

def rasterio_crs(epsg: int):
    """A rasterio CRS for *epsg*.

    Through pyproj's WKT first: inside QGIS rasterio's own PROJ database can
    be a different version from the one QGIS loaded, and then
    ``CRS.from_epsg`` fails where the WKT route still works.
    """
    from rasterio.crs import CRS

    try:
        from pyproj import CRS as PyprojCRS
        return CRS.from_wkt(PyprojCRS.from_epsg(int(epsg)).to_wkt())
    except Exception:  # nosec B110 - no pyproj: rasterio's own database
        return CRS.from_epsg(int(epsg))


def grid_bounds_wgs84(grid: Grid) -> Bounds:
    """*grid*'s bounds as ``(west, south, east, north)`` in degrees."""
    from rasterio.warp import transform_bounds

    west, south, east, north = transform_bounds(
        rasterio_crs(grid.epsg), "EPSG:4326", *grid.bounds, densify_pts=21)
    return (west, south, east, north)


def grid_centre_wgs84(grid: Grid) -> Tuple[float, float]:
    """``(lat, lon)`` of the grid's centre."""
    from rasterio.warp import transform

    min_x, min_y, max_x, max_y = grid.bounds
    lon, lat = transform(rasterio_crs(grid.epsg), "EPSG:4326",
                         [(min_x + max_x) / 2.0], [(min_y + max_y) / 2.0])
    return float(lat[0]), float(lon[0])


def write_raster(path: str, array, grid: Grid, description: str = "",
                 nodata=None, colormap: Optional[dict] = None) -> str:
    """Write *array* (``grid.shape``) as a single-band GeoTIFF.

    Floats are written as float32 with NaN as no-data; integer arrays keep
    their type and take *nodata* and an optional colour table.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import Affine

    array = np.asarray(array)
    if array.shape != grid.shape:
        raise ValueError(f"raster is {array.shape}, grid is {grid.shape}")
    floating = np.issubdtype(array.dtype, np.floating)
    if floating:
        array = array.astype(np.float32)
        nodata = float("nan")
    profile = {
        "driver": "GTiff", "dtype": str(array.dtype), "count": 1,
        "width": grid.width, "height": grid.height,
        "transform": Affine(*grid.geotransform),
        "crs": rasterio_crs(grid.epsg), "compress": "deflate",
        "nodata": nodata,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp.tif"
    with rasterio.open(temporary, "w", **profile) as dst:
        dst.write(array, 1)
        if description:
            dst.set_band_description(1, description)
        if colormap:
            dst.write_colormap(1, colormap)
    os.replace(temporary, path)
    return path


def read_raster(path: str):
    """``(array, grid)`` of an environment raster; floats carry NaN where
    there is no data."""
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        array = src.read(1)
        transform = src.transform
        epsg = src.crs.to_epsg() if src.crs else None
        nodata = src.nodata
    if np.issubdtype(array.dtype, np.floating) and nodata is not None \
            and not (isinstance(nodata, float) and math.isnan(nodata)):
        array = np.where(array == nodata, np.nan, array)
    grid = Grid(int(epsg or 0), float(transform.c), float(transform.f),
                int(array.shape[1]), int(array.shape[0]), float(transform.a))
    return array, grid
