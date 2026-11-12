# -*- coding: utf-8 -*-
"""Habitat context of the animals: environment values where they were.

Joins a flight's environment (:mod:`core.environment`) with its tracks and
answers the question the layers alone cannot: *were the animals somewhere
special?* Three things are produced per survey:

**Per track** - the mean of every environment layer within a radius of the
animal's position, its land-cover class, and the flight's weather. A 10 m
satellite cell says little about the spot an animal stands on; the
neighbourhood describes the patch it was in.

**Used against available** - for each layer, how its values are distributed
at the animals compared with the whole monitored ground. Greener, wetter or
more wooded than the area at large is the signal; a value on its own is not.
*Available* is the ground the cameras actually saw (the field-of-view
footprints, inside the survey area when one is set): an animal could only be
found where somebody looked.

**Per transect** - the mean of each layer over a transect's monitored area,
as covariates for whatever model the user fits outside the plugin.

Nothing here reads the network; it only reads what the environment sub-steps
left in each flight's folder.
"""

import csv
import json
import os
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import environment
from .environment import CATEGORICAL, CONTINUOUS, Grid

HISTOGRAM_BINS = 20

#: Flight-level weather values copied onto every track row, so a pooled table
#: of several flights can be analysed without a join:
#: ``column -> (block of weather.json, key in it)``.
WEATHER_COLUMNS = {
    "air_temperature_c": ("at_flight", "temperature_2m"),
    "soil_temperature_c": ("at_flight", "soil_temperature"),
    "relative_humidity_pct": ("at_flight", "relative_humidity_2m"),
    "dew_point_spread_c": ("at_flight", "dew_point_spread"),
    "wind_speed_ms": ("at_flight", "wind_speed_10m"),
    "cloud_cover_pct": ("at_flight", "cloud_cover"),
    "shortwave_radiation_wm2": ("at_flight", "shortwave_radiation"),
    "snow_depth_m": ("at_flight", "snow_depth"),
    "rain_24h_mm": ("previous_24h", "precipitation_sum"),
    "sunshine_24h_h": ("previous_24h", "sunshine_hours"),
}

_LAYER_COLUMNS = [layer.key for layer in environment.LAYERS]

TRACK_COLUMNS = [
    "flight", "track_id", "species", "sex", "age", "x", "y", "crs",
    *_LAYER_COLUMNS, "landcover_share", "scene_offset_days",
    *WEATHER_COLUMNS,
]

TRANSECT_COLUMNS = [
    "flight", "transect_id", "name", "area_ha", "count",
    *_LAYER_COLUMNS, "landcover_share",
]


class HabitatError(RuntimeError):
    """The habitat summary cannot be produced as asked."""


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def _window(grid: Grid, x: float, y: float, radius: float):
    """``(row slice, col slice, inside mask)`` of the cells whose centre lies
    within *radius* of a point, or ``None`` when the point is off the grid.

    With a radius smaller than half a cell no centre may be inside; the cell
    holding the point is then the answer.
    """
    import numpy as np

    res = grid.resolution
    col = (x - grid.min_x) / res
    row = (grid.max_y - y) / res
    if not (0 <= col < grid.width and 0 <= row < grid.height):
        return None
    reach = radius / res
    row_0 = max(0, int(np.floor(row - reach)))
    row_1 = min(grid.height, int(np.ceil(row + reach)) + 1)
    col_0 = max(0, int(np.floor(col - reach)))
    col_1 = min(grid.width, int(np.ceil(col + reach)) + 1)
    rows = np.arange(row_0, row_1) + 0.5 - row
    cols = np.arange(col_0, col_1) + 0.5 - col
    inside = (rows[:, None] ** 2 + cols[None, :] ** 2) <= reach ** 2
    if not inside.any():
        inside = np.zeros_like(inside)
        inside[int(row) - row_0, int(col) - col_0] = True
    return slice(row_0, row_1), slice(col_0, col_1), inside


def sample_mean(array, grid: Grid, points: Sequence[Tuple[float, float]],
                radius: float) -> List[Optional[float]]:
    """Mean of the valid cells within *radius* of each point.

    ``None`` for a point off the raster or with no valid cell around it.
    """
    import numpy as np

    result: List[Optional[float]] = []
    for x, y in points:
        window = _window(grid, x, y, radius)
        if window is None:
            result.append(None)
            continue
        rows, cols, inside = window
        values = np.asarray(array[rows, cols], dtype=np.float64)[inside]
        values = values[np.isfinite(values)]
        result.append(float(values.mean()) if values.size else None)
    return result


def sample_classes(array, grid: Grid, points: Sequence[Tuple[float, float]],
                   radius: float, nodata=0
                   ) -> List[Optional[Tuple[int, float]]]:
    """``(most common class, its share)`` within *radius* of each point."""
    import numpy as np

    result: List[Optional[Tuple[int, float]]] = []
    for x, y in points:
        window = _window(grid, x, y, radius)
        if window is None:
            result.append(None)
            continue
        rows, cols, inside = window
        values = np.asarray(array[rows, cols])[inside]
        values = values[values != nodata]
        if not values.size:
            result.append(None)
            continue
        classes, counts = np.unique(values, return_counts=True)
        best = int(np.argmax(counts))
        result.append((int(classes[best]),
                       float(counts[best]) / float(counts.sum())))
    return result


def geometry_mask(grid: Grid, geometry):
    """Boolean array, true on the cells whose centre lies in *geometry*
    (everything when *geometry* is ``None``)."""
    import numpy as np

    if geometry is None:
        return np.ones(grid.shape, dtype=bool)
    if geometry.is_empty:
        return np.zeros(grid.shape, dtype=bool)
    from rasterio.features import geometry_mask as rasterio_mask
    from rasterio.transform import Affine

    return rasterio_mask([geometry], out_shape=grid.shape,
                         transform=Affine(*grid.geotransform), invert=True)


def bin_edges(layer: environment.Layer, arrays: Iterable) -> List[float]:
    """Edges of the histogram of *layer*: its documented range when it has
    one, otherwise the central range of the data in *arrays*."""
    import numpy as np

    if layer.value_range is not None:
        low, high = layer.value_range
    else:
        finite = [np.asarray(a)[np.isfinite(a)] for a in arrays]
        finite = [a for a in finite if a.size]
        if not finite:
            return []
        # Between the 1st and 99th percentile: a few extreme cells (a
        # stream bed, in the wetness index) would otherwise squeeze the
        # whole distribution into two bars. Values beyond still count, in
        # the outer bins.
        pooled = np.concatenate(finite)
        low, high = (float(v) for v in np.percentile(pooled, [1.0, 99.0]))
        if high <= low:
            high = low + 1.0
    return [float(v) for v in np.linspace(low, high, HISTOGRAM_BINS + 1)]


def histogram(values, edges: Sequence[float]) -> List[int]:
    """Counts per bin; values outside the edges go into the outer bins, so
    nothing that was measured disappears from the chart."""
    import numpy as np

    if not edges:
        return []
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return [0] * (len(edges) - 1)
    clipped = np.clip(values, edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=np.asarray(edges))
    return [int(c) for c in counts]


def _stats(values) -> dict:
    import numpy as np

    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"n": 0, "mean": None, "median": None}
    return {"n": int(values.size),
            "mean": round(float(values.mean()), 4),
            "median": round(float(np.median(values)), 4)}


# --------------------------------------------------------------------------- #
# One flight
# --------------------------------------------------------------------------- #

def flight_tracks(target_folder: str, modality: str,
                  species_ids: Optional[Sequence[int]] = None,
                  include_manual: bool = True, area=None
                  ) -> List[dict]:
    """``[{track_id, species, sex, age, x, y}]`` - one per counted track.

    The same animals as the census: the track inventory decides which
    tracks there are and what species, sex and age each is (so a species
    vote of the classifier counts, not only the detector's class), and the
    population filter of :mod:`core.analytics_source` - false positives
    excluded, one tracker run plus the manual one - gives each its position,
    the centroid of its box centres. A track that was never placed on the
    ground has no environment and is left out.
    """
    from . import analytics_source, track_inventory

    inventory = track_inventory.build_inventory(
        target_folder, modality, species_ids=species_ids)
    rows, _provenance = analytics_source.load_rows(
        target_folder, modality, include_manual=include_manual)
    centroids, _untracked = analytics_source.track_centroids(rows)

    tracks = []
    for entry in inventory:
        position = centroids.get(entry["track_id"])
        if position is None:
            continue
        x, y = position
        if area is not None and not area.contains(x, y):
            continue
        tracks.append({
            "track_id": int(entry["track_id"]),
            "species": entry.get("species") or "",
            "sex": entry.get("sex"), "age": entry.get("age"),
            "x": float(x), "y": float(y),
        })
    return tracks


def monitored_geometry(target_folder: str, modality: str, area=None):
    """The ground a flight's camera saw (inside *area*), or ``None`` when no
    field-of-view footprints exist - then the whole raster is "available"."""
    from . import fov_store, store
    from .population import merged_fov_area

    if not os.path.isfile(
            store.stage_path(target_folder, store.FOV, modality)):
        return area.geometry if area is not None else None
    polygons = fov_store.load_fov(target_folder, modality)
    _area_m2, geometry = merged_fov_area(polygons, sorted(polygons))
    if geometry is None:
        return area.geometry if area is not None else None
    if area is not None:
        return area.clip(geometry)
    return geometry


def load_layers(target_folder: str) -> Dict[str, Tuple[object, Grid]]:
    """``layer key -> (array, grid)`` of every environment raster there is."""
    layers = {}
    for layer in environment.available_layers(target_folder):
        layers[layer.key] = environment.read_raster(
            environment.layer_path(target_folder, layer.key))
    return layers


def weather_values(target_folder: str) -> Dict[str, Optional[float]]:
    """The :data:`WEATHER_COLUMNS` of a flight (``None`` where unknown)."""
    document = environment.read_metadata(
        target_folder, environment.WEATHER) or {}
    return {column: (document.get(block) or {}).get(key)
            for column, (block, key) in WEATHER_COLUMNS.items()}


def _class_name(value) -> str:
    from . import env_landcover
    return env_landcover.class_name(value)


def sample_points(layers: Dict[str, Tuple[object, Grid]],
                  points: Sequence[Tuple[float, float]],
                  radius: float) -> List[dict]:
    """One dict per point with every layer's value around it."""
    rows: List[dict] = [{} for _ in points]
    for key, (array, grid) in layers.items():
        layer = environment.LAYERS_BY_KEY[key]
        if layer.kind == CATEGORICAL:
            for row, found in zip(rows, sample_classes(
                    array, grid, points, radius)):
                row[key] = None if found is None else _class_name(found[0])
                row[f"{key}_share"] = (
                    None if found is None else round(found[1], 3))
        else:
            for row, value in zip(rows, sample_mean(
                    array, grid, points, radius)):
                row[key] = None if value is None else round(value, 4)
    return rows


def zonal(layers: Dict[str, Tuple[object, Grid]], geometry) -> dict:
    """Every layer's mean (or most common class) over a polygon."""
    import numpy as np

    result: dict = {}
    for key, (array, grid) in layers.items():
        layer = environment.LAYERS_BY_KEY[key]
        mask = geometry_mask(grid, geometry)
        values = np.asarray(array)[mask]
        if layer.kind == CATEGORICAL:
            values = values[values != 0]
            if not values.size:
                result[key], result[f"{key}_share"] = None, None
                continue
            classes, counts = np.unique(values, return_counts=True)
            best = int(np.argmax(counts))
            result[key] = _class_name(classes[best])
            result[f"{key}_share"] = round(
                float(counts[best]) / float(counts.sum()), 3)
        else:
            values = values[np.isfinite(values)]
            result[key] = (round(float(values.mean()), 4)
                           if values.size else None)
    return result


def transect_rows(target_folder: str, modality: str,
                  layers: Dict[str, Tuple[object, Grid]]) -> List[dict]:
    """Environment means per transect, when the population estimate has
    written the flight's transect areas; ``[]`` otherwise."""
    from shapely.geometry import shape

    path = os.path.join(target_folder, f"analytics_{modality}",
                        "transect_areas.geojson")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            features = json.load(fh).get("features") or []
    except (OSError, ValueError):
        return []
    rows = []
    for feature in features:
        if not feature.get("geometry"):
            continue
        properties = feature.get("properties") or {}
        row = {"transect_id": properties.get("transect_id"),
               "name": properties.get("name"),
               "area_ha": properties.get("area_ha"),
               "count": properties.get("count")}
        row.update(zonal(layers, shape(feature["geometry"])))
        rows.append(row)
    return rows


def summarise_flight(name: str, target_folder: str, modality: str,
                     radius: float,
                     species_ids: Optional[Sequence[int]] = None,
                     include_manual: bool = True, area=None,
                     epsg: Optional[int] = None) -> dict:
    """Everything the survey summary needs from one flight.

    :return: ``{"flight", "tracks": [row], "transects": [row],
        "available": {layer: values inside the monitored ground},
        "sources": {…}, "missing": [source titles]}``
    """
    import numpy as np

    layers = load_layers(target_folder)
    done = environment.status(target_folder)
    tracks = flight_tracks(target_folder, modality, species_ids,
                           include_manual, area)
    sampled = sample_points(layers, [(t["x"], t["y"]) for t in tracks],
                            radius)
    weather = weather_values(target_folder)
    scene = environment.read_metadata(
        target_folder, environment.SENTINEL2) or {}

    rows = []
    for track, values in zip(tracks, sampled):
        row = {"flight": name, "track_id": track["track_id"],
               "species": track["species"],
               "sex": track["sex"], "age": track["age"],
               "x": round(track["x"], 2), "y": round(track["y"], 2),
               "crs": f"EPSG:{epsg}" if epsg else ""}
        row.update(values)
        row["scene_offset_days"] = scene.get("offset_days")
        row.update(weather)
        rows.append(row)

    geometry = monitored_geometry(target_folder, modality, area)
    available = {}
    for key, (array, grid) in layers.items():
        values = np.asarray(array)[geometry_mask(grid, geometry)]
        if environment.LAYERS_BY_KEY[key].kind == CATEGORICAL:
            available[key] = values[values != 0]
        else:
            available[key] = values[np.isfinite(values)]

    transects = [dict(row, flight=name)
                 for row in transect_rows(target_folder, modality, layers)]

    weather_document = environment.read_metadata(
        target_folder, environment.WEATHER) or {}
    return {
        "flight": name,
        "tracks": rows,
        "transects": transects,
        "available": available,
        "missing": [environment.SOURCE_TITLES[source]
                    for source in environment.SOURCES if not done[source]],
        "sources": {
            "sentinel2": {key: scene.get(key) for key in (
                "scene", "scene_time", "offset_days", "clear_fraction")}
            if scene else None,
            "weather": {
                "at_flight": weather_document.get("at_flight"),
                "previous_24h": weather_document.get("previous_24h"),
                "flight_start": weather_document.get("flight_start"),
            } if weather_document else None,
        },
        "available_basis": ("monitored ground (field-of-view footprints)"
                            if geometry is not None else "whole raster"),
    }


# --------------------------------------------------------------------------- #
# The survey
# --------------------------------------------------------------------------- #

def selection_ratio(used_share: Optional[float],
                    available_share: Optional[float]) -> Optional[float]:
    """Share of the animals in a class over the share of the ground it
    covers: above 1 the class is used more than its area would suggest."""
    if used_share is None or not available_share:
        return None
    return round(used_share / available_share, 3)


def build_summary(flights: Sequence[dict], radius: float, modality: str,
                  species: Optional[Sequence[str]] = None) -> dict:
    """Pool the per-flight results into the survey's habitat document."""
    import numpy as np

    tracks = [row for flight in flights for row in flight["tracks"]]
    document_layers = {}
    for layer in environment.LAYERS:
        if layer.kind != CONTINUOUS:
            continue
        available = [flight["available"][layer.key] for flight in flights
                     if layer.key in flight["available"]]
        if not available:
            continue
        used = [row[layer.key] for row in tracks
                if row.get(layer.key) is not None]
        edges = bin_edges(layer, available + [np.asarray(used, dtype=float)])
        pooled = np.concatenate(available) if available else np.array([])
        document_layers[layer.key] = {
            "title": layer.title, "unit": layer.unit,
            "description": layer.description,
            "bin_edges": [round(edge, 4) for edge in edges],
            "available_counts": histogram(pooled, edges),
            "used_counts": histogram(used, edges),
            "available": _stats(pooled),
            "used": _stats(used),
        }

    landcover = None
    cover = [flight["available"]["landcover"] for flight in flights
             if "landcover" in flight["available"]]
    if cover:
        from . import env_landcover

        pooled = np.concatenate(cover)
        total = float(pooled.size) or 1.0
        used = Counter(row["landcover"] for row in tracks
                       if row.get("landcover"))
        n_used = sum(used.values())
        classes = []
        for value, (name, colour) in env_landcover.CLASSES.items():
            cells = int(np.count_nonzero(pooled == value))
            count = used.get(name, 0)
            if not cells and not count:
                continue
            available_share = cells / total
            used_share = count / n_used if n_used else None
            classes.append({
                "name": name, "colour": list(colour),
                "available_share": round(available_share, 4),
                "n_tracks": count,
                "used_share": (None if used_share is None
                               else round(used_share, 4)),
                "selection_ratio": selection_ratio(
                    used_share, available_share),
            })
        landcover = {"n_tracks": n_used, "classes": classes}

    return {
        "created": environment.utc_now(),
        "modality": modality,
        "radius_m": radius,
        "species": None if species is None else list(species),
        "n_tracks": len(tracks),
        "flights": [{
            "flight": flight["flight"],
            "n_tracks": len(flight["tracks"]),
            "n_transects": len(flight["transects"]),
            "missing": flight["missing"],
            "available_basis": flight["available_basis"],
            "sources": flight["sources"],
        } for flight in flights],
        "layers": document_layers,
        "landcover": landcover,
        "note": (
            "Track values are the mean of each layer within radius_m of the "
            "track's position; 'available' is the monitored ground of the "
            "flights. Satellite cells are 10-20 m: the values describe the "
            "patch an animal was in, not the spot it stood on."),
    }


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #

def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(round(value, 6))
    return str(value)


def write_table(rows: Sequence[dict], columns: Sequence[str],
                path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row.get(column)) for column in columns])
    return path


def write_json(document, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2, default=str)
    return path


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


TRACKS_CSV = "habitat_tracks.csv"
TRACKS_JSON = "habitat_tracks.json"
TRANSECTS_CSV = "habitat_transects.csv"
SUMMARY_JSON = "habitat.json"


def write_outputs(analytics_folder: str, summary: dict,
                  flights: Sequence[dict]) -> Dict[str, str]:
    """Write the survey's habitat files; returns ``what -> path``."""
    tracks = [row for flight in flights for row in flight["tracks"]]
    transects = [row for flight in flights for row in flight["transects"]]
    paths = {
        "summary": write_json(
            summary, os.path.join(analytics_folder, SUMMARY_JSON)),
        "tracks_csv": write_table(
            tracks, TRACK_COLUMNS,
            os.path.join(analytics_folder, TRACKS_CSV)),
        "tracks_json": write_json(
            tracks, os.path.join(analytics_folder, TRACKS_JSON)),
    }
    transect_path = os.path.join(analytics_folder, TRANSECTS_CSV)
    if transects:
        paths["transects_csv"] = write_table(
            transects, TRANSECT_COLUMNS, transect_path)
    elif os.path.isfile(transect_path):
        # Left over from a run that had transects: it would describe a
        # different selection of flights.
        os.remove(transect_path)
    return paths


def filter_tracks(tracks: Sequence[dict],
                  species: Optional[str] = None) -> List[dict]:
    """The track rows of one species (all of them for ``None``/``""``)."""
    if not species:
        return list(tracks)
    return [row for row in tracks if row.get("species") == species]


def used_histogram(tracks: Sequence[dict], key: str,
                   edges: Sequence[float]) -> List[int]:
    """Histogram of a layer at the given tracks, on the summary's bins - so
    the charts can be redrawn per species without recomputing anything."""
    return histogram([row[key] for row in tracks
                      if row.get(key) is not None], edges)


def describe(summary: dict) -> List[str]:
    """A few log lines: where the animals stood out from the area."""
    lines = [f"{summary['n_tracks']} track(s) over "
             f"{len(summary['flights'])} flight(s), values averaged within "
             f"{summary['radius_m']:g} m"]
    for key, layer in summary["layers"].items():
        used, available = layer["used"], layer["available"]
        if used["mean"] is None or available["mean"] is None:
            continue
        lines.append(
            f"{layer['title']}: at the animals {used['mean']:.2f}"
            f"{layer['unit']}, monitored ground {available['mean']:.2f}"
            f"{layer['unit']}")
    cover = summary.get("landcover")
    if cover and cover["n_tracks"]:
        ranked = sorted((c for c in cover["classes"] if c["n_tracks"]),
                        key=lambda c: -c["n_tracks"])
        lines.append("Land cover at the animals: " + ", ".join(
            f"{c['name']} {c['used_share'] * 100:.0f}% "
            f"(ground {c['available_share'] * 100:.0f}%)"
            for c in ranked[:4]))
    return lines
