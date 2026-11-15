# -*- coding: utf-8 -*-
"""What the animals look like on the map: a colour per species, a short
indicator for sex and age, and the attributes a layer needs to show both.

**Colour means species.** Detections and tracks are drawn in the colour of
their species, so a glance at the map says *what* is where - a colour per
track said only that two boxes were different animals. Every species has a
default colour: the seeded ones a fixed one, any other a colour derived from
its name, so "roe deer" looks the same in every flight without anyone having
set it. A project can override any of them in the Project Schema dialog;
the overrides are kept in the flight's ``project.gpkg`` (``bambi_meta``),
keyed by species id so that renaming a species keeps its colour.

**Sex and age are a short text beside the pin** - "♀", "♂ juv" - because
a third and fourth colour dimension would be unreadable. Unknown is not
shown at all: no indicator means nobody knows, not "neither".

No Qt in here; :mod:`bambi_map_symbols` turns this into QGIS symbology.
"""

import json
import re
import zlib
from typing import Dict, Iterable, List, Optional

from . import store

META_KEY = "species_colours"

#: Map scale (1:n) at which the boxes give way to pins. At 1:500 a
#: one-metre animal is two millimetres on screen - about where an outline
#: stops being a shape.
DEFAULT_SWITCH_SCALE = 500

#: Fixed colours of the base classes and the seeded species. Chosen to be
#: told apart on an orthophoto and on a thermal mosaic alike, and by people
#: who cannot tell red from green: no two neighbours differ in that alone.
SEEDED_COLOURS = {
    "animal": "#ffd92f",           # undetermined animal: plain yellow
    "unknown": "#bdbdbd",
    "not-an-animal": "#636363",
    "roe deer": "#e6550d",
    "red deer": "#a50f15",
    "fallow deer": "#f781bf",
    "wild boar": "#54278f",
    "chamois": "#1f78b4",
    "fox": "#ff7f00",
    "hare": "#b2df8a",
    "bird": "#17becf",
    "other": "#8c6d31",
}

#: Colours for species the user added, picked by name.
PALETTE = (
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02",
    "#a6761d", "#377eb8", "#984ea3", "#4daf4a", "#f768a1", "#00bfc4",
)

#: Colour of anything without a species.
FALLBACK_COLOUR = SEEDED_COLOURS["animal"]

#: Attributes a styled layer carries, in field order.
FIELDS = ("species", "sex", "age", "indicator")

_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")

#: Sex labels that have a symbol of their own.
_SEX_SYMBOLS = {"female": "♀", "male": "♂"}
#: Age labels that have a customary abbreviation.
_AGE_SHORT = {"adult": "ad", "juvenile": "juv", "subadult": "sub",
              "sub-adult": "sub", "yearling": "yrl", "fawn": "fawn",
              "calf": "calf"}
#: Labels that mean "nobody knows", shown as nothing.
_UNKNOWN = {"", "unknown", "none", "n/a", "na", "?"}


def normalise_colour(value) -> str:
    """*value* as ``#rrggbb`` (lower case).

    :raises ValueError: when it is not a six-digit hex colour
    """
    match = _HEX.match(str(value).strip()) if value is not None else None
    if match is None:
        raise ValueError(f"Not a colour: {value!r} (expected #rrggbb).")
    return "#" + match.group(1).lower()


def default_colour(name: str) -> str:
    """The colour a species has until the project says otherwise.

    The same name always gives the same colour, on every machine: it is
    looked up, or taken from the palette by a checksum of the name.
    """
    key = (name or "").strip().lower()
    if key in SEEDED_COLOURS:
        return SEEDED_COLOURS[key]
    if not key:
        return FALLBACK_COLOUR
    return PALETTE[zlib.crc32(key.encode("utf-8")) % len(PALETTE)]


def stored_colours(conn) -> Dict[int, str]:
    """``species_id -> colour`` of the overrides in a project store."""
    raw = store.get_meta(conn, META_KEY)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    colours: Dict[int, str] = {}
    for key, value in (data.items() if isinstance(data, dict) else ()):
        try:
            colours[int(key)] = normalise_colour(value)
        except (TypeError, ValueError):
            continue
    return colours


def store_colour(conn, species_id: int, colour: Optional[str]) -> None:
    """Set (or, with ``None``, clear) the colour override of a species.

    Written into the caller's connection without committing, so it is part
    of whatever the caller commits or rolls back.
    """
    colours = stored_colours(conn)
    if colour is None:
        colours.pop(int(species_id), None)
    else:
        colours[int(species_id)] = normalise_colour(colour)
    store.set_meta(conn, META_KEY, json.dumps(
        {str(key): value for key, value in sorted(colours.items())}))


def colours_for(species: Iterable[dict],
                overrides: Dict[int, str]) -> Dict[str, str]:
    """``name -> colour`` for *species* rows (``species_id``, ``name``)."""
    colours = {}
    for row in species:
        own = overrides.get(int(row["species_id"]))
        colours[row["name"]] = own or default_colour(row["name"])
    return colours


def species_colours(target_folder: str) -> Dict[str, str]:
    """``species name -> colour`` of a flight: its overrides over the
    defaults. Empty when the flight has no store yet - callers then fall
    back on :func:`default_colour` name by name."""
    import os

    path = store.project_path(target_folder) if target_folder else ""
    if not path or not os.path.isfile(path):
        return {}
    conn = store.open_store(path, store.PROJECT)
    try:
        rows = [dict(row) for row in conn.execute(
            "SELECT species_id, name FROM species")]
        return colours_for(rows, stored_colours(conn))
    finally:
        conn.close()


def colour_of(colours: Dict[str, str], name: Optional[str]) -> str:
    """The colour of species *name*, known to *colours* or not."""
    return colours.get(name or "") or default_colour(name or "")


# --------------------------------------------------------------------------- #
# Sex and age
# --------------------------------------------------------------------------- #

def _known(label) -> str:
    text = "" if label is None else str(label).strip()
    return "" if text.lower() in _UNKNOWN else text


def sex_indicator(label) -> str:
    """"♀" / "♂" for female / male; the first letter of any other label a
    project defined; nothing for unknown."""
    text = _known(label)
    if not text:
        return ""
    return _SEX_SYMBOLS.get(text.lower()) or text[:1].upper()


def age_indicator(label) -> str:
    """A short form of an age class ("juv", "ad"); nothing for unknown."""
    text = _known(label)
    if not text:
        return ""
    return _AGE_SHORT.get(text.lower()) or text[:4].lower()


def indicator(sex=None, age=None) -> str:
    """Sex and age as one short text, e.g. ``"♀ juv"``; empty when neither
    is known."""
    return " ".join(part for part in (sex_indicator(sex), age_indicator(age))
                    if part)


def attributes(species=None, sex=None, age=None) -> Dict[str, str]:
    """The :data:`FIELDS` of one animal, as a layer stores them."""
    return {"species": species or "", "sex": _known(sex), "age": _known(age),
            "indicator": indicator(sex, age)}


# --------------------------------------------------------------------------- #
# From the stores
# --------------------------------------------------------------------------- #

def track_attributes(target_folder: str, modality: str,
                     track_ids: Optional[Iterable[int]] = None
                     ) -> Dict[int, Dict[str, str]]:
    """``track_id -> attributes`` for the tracks of a flight.

    Species, sex and age as the track inventory reports them - the track's
    own species or the classifier's vote, the labelling tool's annotations -
    so the map agrees with every table. Empty when the flight has no tracks.
    """
    from . import track_inventory

    try:
        inventory = track_inventory.build_inventory(
            target_folder, modality, track_ids=track_ids)
    except Exception:  # noqa: BLE001 - an unreadable store is "no tracks"
        return {}
    return {int(row["track_id"]): attributes(
        row.get("species"), row.get("sex"), row.get("age"))
        for row in inventory}


def detection_attributes(target_folder: str, modality: str
                         ) -> Dict[int, Dict[str, str]]:
    """``detection_id -> attributes`` for the detections of a flight.

    A detection that belongs to a track is what its track is - the animal
    was identified as a whole, not box by box. One that belongs to none has
    the species the detector's class maps to, and no sex or age.
    """
    from .exporters import common

    try:
        rows = common.load_detections(
            target_folder, modality, include_not_an_animal=True,
            include_geo=False)
        names = common.load_vocabulary(target_folder)["species"]
    except Exception:  # noqa: BLE001 - nothing stored yet
        return {}
    tracks = track_attributes(target_folder, modality)
    result: Dict[int, Dict[str, str]] = {}
    for row in rows:
        track = tracks.get(row.get("track_id")) \
            if row.get("track_id") is not None else None
        result[int(row["detection_id"])] = track or attributes(
            names.get(row.get("species_id"), ""))
    return result


def species_present(values: Iterable) -> List[str]:
    """Distinct, non-empty species names of *values*, sorted."""
    return sorted({str(v) for v in values if v not in (None, "")})
