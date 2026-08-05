# -*- coding: utf-8 -*-
"""Flights: configuration in the store, several flights per QGIS project.

Implements EXCHANGE_FORMAT_PLAN.md §10.2. Two halves:

* **Configuration lives in the flight's own ``project.gpkg``**, beside the
  species, enums and stage state that describe the same outputs. That makes a
  flight folder self-describing — hand someone the folder and the settings that
  produced it come with it. The mirror image is the honest cost: lose the
  folder and the configuration goes with it, where today it survives in the
  ``.qgz``.
* **A QGIS project holds a list of flights**, each one a name and a target
  folder, with exactly one active. Everything downstream keeps reading a single
  configuration, so the processing stages need no notion of flights at all.

**One flight is one target folder.** This is a constraint rather than a
convention: ``bambi_{m}/detections.gpkg`` and its siblings are per folder and
modality with no flight dimension, so two flights sharing a folder would
overwrite each other's detections. :func:`validate_folder` is what makes that
guaranteed rather than merely likely.
"""

import os
from typing import Dict, List, Optional

from . import store


class FlightError(ValueError):
    """A flight cannot be added or renamed as requested."""


# ---------------------------------------------------------------------------
# Configuration in the store
# ---------------------------------------------------------------------------

def save_config(target_folder: str, values: Dict[str, object],
                log_fn=None) -> int:
    """Write this flight's form values into its ``project.gpkg``.

    Values are stored as text, exactly as the QGIS project stored them; the
    schema in :mod:`core.config_schema` remains the single description of what
    each key means, so nothing here has to know a type.
    """
    conn = store.open_store(store.project_path(target_folder), store.PROJECT)
    try:
        with store.transaction(conn):
            conn.execute("DELETE FROM config")
            conn.executemany(
                "INSERT INTO config (key, value) VALUES (?, ?)",
                [(key, "" if value is None else str(value))
                 for key, value in sorted(values.items())])
    finally:
        conn.close()

    if log_fn:
        log_fn(f"Configuration saved to {os.path.basename(target_folder)}")
    return len(values)


def load_config(target_folder: str) -> Dict[str, str]:
    """Read this flight's stored form values, or ``{}`` when it has none."""
    path = store.project_path(target_folder)
    if not os.path.isfile(path):
        return {}
    conn = store.open_store(path, store.PROJECT)
    try:
        return {row["key"]: row["value"]
                for row in conn.execute("SELECT key, value FROM config")}
    finally:
        conn.close()


def has_config(target_folder: str) -> bool:
    return bool(load_config(target_folder))


def config_readers(values: Dict[str, str]):
    """Reader callables over *values*, in the shape ``load_config_entries`` wants.

    ``core.config_schema`` already takes injected readers and writers, so
    moving the configuration out of the QGIS project is a matter of passing
    these instead — every entry comes along unchanged.
    """
    def read_str(key: str, default: str = "") -> str:
        value = values.get(key)
        return default if value is None or value == "" else value

    def read_double(key: str, default: float = 0.0) -> float:
        try:
            return float(values[key])
        except (KeyError, TypeError, ValueError):
            return default

    def read_bool(key: str, default: bool = False) -> bool:
        value = values.get(key)
        if value is None or value == "":
            return default
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    return read_str, read_double, read_bool


def config_writers(target: Dict[str, object]):
    """Writer callables that collect into *target*, for ``save_config_entries``."""
    def write_str(key: str, value) -> None:
        target[key] = value

    return write_str, write_str, write_str


# ---------------------------------------------------------------------------
# The flight list
# ---------------------------------------------------------------------------

def normalise(folder: str) -> str:
    return os.path.normcase(os.path.abspath(folder)) if folder else ""


def default_name(target_folder: str) -> str:
    """A readable name for a flight, derived from its folder.

    The default target folder is ``<input>/qgis``, so the folder name alone is
    usually just "qgis" — the parent is what identifies the flight.
    """
    if not (target_folder or "").strip():
        return "Flight"
    folder = os.path.normpath(target_folder)
    base = os.path.basename(folder)
    if base.lower() in ("qgis", "") and os.path.dirname(folder):
        base = os.path.basename(os.path.dirname(folder))
    return base or "Flight"


def validate_folder(folder: str, flights: List[dict],
                    ignore_index: Optional[int] = None) -> None:
    """Raise :class:`FlightError` when *folder* cannot host a new flight.

    Two flights sharing a folder would overwrite each other's detections, so
    this is refused rather than warned about.
    """
    if not folder or not folder.strip():
        raise FlightError("A flight needs a target folder.")

    target = normalise(folder)
    for index, flight in enumerate(flights):
        if index == ignore_index:
            continue
        if normalise(flight.get("target_folder", "")) == target:
            raise FlightError(
                f"'{flight.get('name')}' already uses this folder. Each flight "
                "needs its own, because the stage files are per folder — two "
                "flights sharing one would overwrite each other's detections.")


def validate_name(name: str, flights: List[dict],
                  ignore_index: Optional[int] = None) -> str:
    """Return the cleaned flight name, or raise when it clashes."""
    name = (name or "").strip()
    if not name:
        raise FlightError("A flight needs a name.")
    for index, flight in enumerate(flights):
        if index == ignore_index:
            continue
        if flight.get("name", "").strip().lower() == name.lower():
            raise FlightError(f"A flight called '{name}' already exists.")
    return name


def add_flight(flights: List[dict], name: str, target_folder: str) -> List[dict]:
    """Return *flights* with a new one appended, validating both fields."""
    name = validate_name(name, flights)
    validate_folder(target_folder, flights)
    return list(flights) + [{"name": name, "target_folder": target_folder}]


def rename_flight(flights: List[dict], index: int, name: str) -> List[dict]:
    """Return *flights* with the flight at *index* renamed."""
    if not 0 <= index < len(flights):
        raise FlightError("No such flight.")
    name = validate_name(name, flights, ignore_index=index)
    updated = [dict(flight) for flight in flights]
    updated[index]["name"] = name
    return updated


def set_folder(flights: List[dict], index: int, folder: str) -> List[dict]:
    """Return *flights* with the flight at *index* pointed at *folder*."""
    if not 0 <= index < len(flights):
        raise FlightError("No such flight.")
    validate_folder(folder, flights, ignore_index=index)
    updated = [dict(flight) for flight in flights]
    updated[index]["target_folder"] = folder
    return updated


def find_by_folder(flights: List[dict], folder: str) -> Optional[int]:
    """Index of the flight using *folder*, or ``None``."""
    target = normalise(folder)
    for index, flight in enumerate(flights):
        if normalise(flight.get("target_folder", "")) == target:
            return index
    return None


#: Config key holding a flight's DEM metadata JSON, needed by the analytics
#: that place transects.
DEM_METADATA_KEY = "Input/DemMetadataPath"
DEM_PATH_KEY = "Input/DemPath"


def dem_metadata_path(target_folder: str) -> str:
    """The DEM metadata JSON a flight was processed with, or ``""``.

    Read from the flight's own stored configuration, which is the point of
    putting it there: adding a flight to a multi-project analysis no longer
    means remembering which ``dem.json`` went with which folder (§10.2).
    Falls back to the mesh path with its extension swapped, the convention the
    rest of the plugin already uses.
    """
    config = load_config(target_folder)
    explicit = (config.get(DEM_METADATA_KEY) or "").strip()
    if explicit:
        return explicit

    mesh = (config.get(DEM_PATH_KEY) or "").strip()
    if not mesh:
        return ""
    guess = mesh.replace(".gltf", ".json").replace(".glb", ".json")
    return guess if guess != mesh and os.path.isfile(guess) else ""


def analysis_entries(flights: List[dict], exclude: Optional[str] = None
                     ) -> List[dict]:
    """Flights as ``{name, target_folder, dem}`` rows for the analytics pickers.

    *exclude* drops one folder — normally the active flight, which those tools
    add separately via their "Add current project" tick.
    """
    skip = normalise(exclude) if exclude else ""
    entries = []
    for flight in flights:
        folder = flight.get("target_folder", "")
        if not folder or (skip and normalise(folder) == skip):
            continue
        entries.append({
            "name": flight.get("name") or default_name(folder),
            "target_folder": folder,
            "dem": dem_metadata_path(folder),
        })
    return entries


def group_name(flight: dict) -> str:
    """QGIS layer-group name for a flight — its name, and nothing derived.

    Renaming a flight renames the group, so the name is the only thing tying
    them together (§10.2). Deriving it from the folder instead would leave a
    stale group behind on every rename.
    """
    return (flight.get("name") or "").strip() or "Flight"
