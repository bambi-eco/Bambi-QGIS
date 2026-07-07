# -*- coding: utf-8 -*-
"""Headless flight-planner logic.

Moved verbatim from ``bambi_flight_planner.py`` (which re-exports every name
here): route-GeoJSON post-processing, the parameter defaults and the
QtCore-only worker that runs an evaluation flight strategy.
"""

import io
import json
import sys

from qgis.PyQt.QtCore import QObject, pyqtSignal

_SETTINGS_PREFIX = "bambi/flight_planner/"

_DEFAULTS = {
    "strategy": "random",
    "grid_size": 400.0,
    "max_start_and_stop_distance": 3000.0,
    "min_transects": 40,
    "max_transects_enabled": False,
    "max_transects": 100,
    "max_distance": 2000.0,
    "min_transect_overlap": 0.75,
    "number_of_retries": 100,
    "target_crs_epsg": 32633,
    "min_transects_per_route": 3,
    "x_offset": 0.0,
    "y_offset": 0.0,
    "padding_north": 0,
    "padding_east": 0,
    "padding_south": 0,
    "padding_west": 0,
    "seed": "",
    "max_number_of_overlapping_transects": 0,
    "max_number_of_flights": 100,
    "random_search": True,
    "number_of_retries_per_route": 50,
    "target_folder": "",
}


def _fix_std_streams():
    # QGIS sets sys.stderr/stdout to None; numpy/geopandas crash when they
    # try to write deprecation or error messages through those streams.
    if sys.stderr is None:
        sys.stderr = io.StringIO()
    if sys.stdout is None:
        sys.stdout = io.StringIO()


def _to_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes")


def _write_route_line(src_path, out_path):
    """Write a clean single-feature LineString GeoJSON containing only the
    'total-route' feature from a mixed-geometry route GeoJSON.  Returns
    *out_path* on success or None if the feature is not found."""
    try:
        with open(src_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    total = next(
        (feat for feat in data.get("features", [])
         if feat.get("properties", {}).get("name") == "total-route"),
        None,
    )
    if total is None:
        return None
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": [total]}, f)
    return out_path


def _write_route_transects(src_path, out_path):
    """Extract individual survey transect LineStrings from a route GeoJSON,
    tag each with a 1-based ``transect_no`` property, and write a clean
    single-geometry-type GeoJSON to *out_path*.  Returns *out_path* on
    success or None if no transects are found."""
    _SKIP = {"arrival", "departure", "total-route"}
    try:
        with open(src_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    transects = []
    no = 1
    for feat in data.get("features", []):
        if feat.get("geometry", {}).get("type") != "LineString":
            continue
        if feat.get("properties", {}).get("name") in _SKIP:
            continue
        feat = json.loads(json.dumps(feat))  # deep copy
        feat.setdefault("properties", {})["transect_no"] = no
        transects.append(feat)
        no += 1
    if not transects:
        return None
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": transects}, f)
    return out_path


class _FlightPlanWorker(QObject):
    finished = pyqtSignal(bool, str)

    def __init__(self, strategy_cls, strategy_kwargs,
                 area_path, start_path, target_path, invalid_path):
        super().__init__()
        self._strategy_cls = strategy_cls
        self._strategy_kwargs = strategy_kwargs
        self._area_path = area_path
        self._start_path = start_path
        self._target_path = target_path
        self._invalid_path = invalid_path

    def run(self):
        _fix_std_streams()
        try:
            try:
                from fiona.drvsupport import supported_drivers
                supported_drivers["kml"] = "rw"
                supported_drivers["KML"] = "rw"
                supported_drivers["LIBKML"] = "rw"
            except Exception:  # nosec B110
                pass

            strategy = self._strategy_cls(**self._strategy_kwargs)
            routes = strategy.create_routes(
                area_path=self._area_path,
                start_points_path=self._start_path,
                target_path=self._target_path,
                invalid_areas_path=self._invalid_path,
            )
            count = len(routes) if routes else 0
            self.finished.emit(True, str(count))
        except Exception:
            import traceback
            self.finished.emit(False, traceback.format_exc())
