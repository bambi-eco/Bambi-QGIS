# -*- coding: utf-8 -*-
"""What the animals look like on the map: a colour per species, a short
indicator for sex and age, and the attributes a layer needs to show both.

Two scales, two questions. **Zoomed in**, the question is *which animal is
this* - so a track's box and path keep a colour of their own per individual,
and two roe deer side by side can be told apart. **Zoomed out**, where the
boxes have become pins, it is *what is where* - so a pin is drawn in the
colour of its species.

Every species has a default colour from the colour-blind-safe Okabe-Ito
palette: the seeded ones a fixed one (an undetermined animal is black), any
other a colour derived from its name, so "roe deer" looks the same in every
flight without anyone having set it. A project can override any of them in the
Project Schema dialog; the overrides are kept in the flight's
``project.gpkg`` (``bambi_meta``), keyed by species id so that renaming a
species keeps its colour.

**Sex and age are part of the pin**, not text beside it - with several pins
close together a label next to one reads as belonging to its neighbour. The
sex is the mark in the middle of the pin ("♀", "♂", and "?" when it is not
known), the age a small badge on its lower right ("J", "A"), left off when
it is not known.

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

#: The Okabe-Ito palette: eight colours that stay distinguishable with the
#: common forms of colour blindness (Okabe & Ito 2008, "Color Universal
#: Design").
OKABE_ITO = {
    "orange": "#e69f00",
    "sky blue": "#56b4e9",
    "bluish green": "#009e73",
    "yellow": "#f0e442",
    "blue": "#0072b2",
    "vermillion": "#d55e00",
    "reddish purple": "#cc79a7",
    "black": "#000000",
}

#: Fixed colours of the base classes and the seeded species, from the
#: Okabe-Ito palette. An animal whose species is not determined is black;
#: the two other base classes are greys, so that nothing that is not a
#: species competes with the species for a colour. The palette has seven
#: colours besides black and the seeded species are nine, so the last two
#: take colours from Paul Tol's colour-blind-safe "muted" scheme.
SEEDED_COLOURS = {
    "animal": OKABE_ITO["black"],
    "unknown": "#4d4d4d",
    "not-an-animal": "#bdbdbd",
    "red deer": OKABE_ITO["orange"],
    "roe deer": OKABE_ITO["blue"],
    "wild boar": OKABE_ITO["bluish green"],
    "fallow deer": OKABE_ITO["vermillion"],
    "chamois": OKABE_ITO["sky blue"],
    "fox": OKABE_ITO["reddish purple"],
    "hare": OKABE_ITO["yellow"],
    "bird": "#882255",
    "other": "#999933",
}

#: Colours for species the user added, picked by name: the Okabe-Ito
#: colours (black stays the colour of the undetermined), then Tol's.
PALETTE = tuple(colour for name, colour in OKABE_ITO.items()
                if name != "black") + (
    "#882255", "#999933", "#332288", "#44aa99", "#aa4499")

#: Colour of anything without a species.
FALLBACK_COLOUR = SEEDED_COLOURS["animal"]

#: Colours of the individuals: a track's box and path, by its track id.
INDIVIDUAL_PALETTE = (
    "#ff0000", "#009600", "#0064ff", "#ffa500", "#800080", "#00c8c8",
    "#ff69b4", "#8b4513", "#00008b", "#228b22",
)

#: Box colour of a detection that belongs to no track.
UNTRACKED_COLOUR = "#ffd92f"

#: Attributes a styled layer carries, in field order: what the animal is,
#: the two marks its pin shows, and the colour of its box.
FIELDS = ("species", "sex", "age", "sex_mark", "age_mark", "box_colour")

_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")

#: Sex labels that have a symbol of their own.
_SEX_SYMBOLS = {"female": "♀", "male": "♂"}
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


def sex_mark(label) -> str:
    """The mark in the middle of a pin: "♀" / "♂" for female / male, the
    first letter of any other label a project defined, nothing for unknown
    (the pin then shows :data:`UNKNOWN_SEX_MARK`)."""
    text = _known(label)
    if not text:
        return ""
    return _SEX_SYMBOLS.get(text.lower()) or text[:1].upper()


#: What a pin shows where the sex is not known.
UNKNOWN_SEX_MARK = "?"


def age_mark(label) -> str:
    """The letter in a pin's age badge: the first letter of the age class
    ("J", "A", "S"), nothing for unknown."""
    text = _known(label)
    return text[:1].upper() if text else ""


MARK_LIGHT = "#ffffff"
MARK_DARK = "#202020"


def _luminance(colour: str) -> float:
    """Relative luminance of a colour, 0 (black) to 1 (white), as WCAG
    defines it."""
    value = normalise_colour(colour)

    def channel(start: int) -> float:
        share = int(value[start:start + 2], 16) / 255.0
        return (share / 12.92 if share <= 0.03928
                else ((share + 0.055) / 1.055) ** 2.4)

    return 0.2126 * channel(1) + 0.7152 * channel(3) + 0.0722 * channel(5)


def contrast(first: str, second: str) -> float:
    """WCAG contrast ratio of two colours, 1 (none) to 21."""
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def reads_on(colour: str) -> str:
    """The colour a mark is drawn in on top of *colour*: white or
    near-black, whichever contrasts more - white on the blue of a roe deer,
    dark on the orange of a red deer."""
    light = contrast(colour, MARK_LIGHT)
    return MARK_LIGHT if light >= contrast(colour, MARK_DARK) else MARK_DARK


def individual_colour(track_id) -> str:
    """The colour of one individual's box and path.

    From the track id, so a track has the same colour on every layer it
    appears on - its own, and the detection layers - and keeps it when the
    layers are added again.
    """
    if track_id is None:
        return UNTRACKED_COLOUR
    return INDIVIDUAL_PALETTE[int(track_id) % len(INDIVIDUAL_PALETTE)]


def attributes(species=None, sex=None, age=None,
               track_id=None) -> Dict[str, str]:
    """The :data:`FIELDS` of one animal, as a layer stores them."""
    return {"species": species or "", "sex": _known(sex), "age": _known(age),
            "sex_mark": sex_mark(sex), "age_mark": age_mark(age),
            "box_colour": individual_colour(track_id)}


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
        row.get("species"), row.get("sex"), row.get("age"),
        track_id=row["track_id"])
        for row in inventory}


def detection_attributes(target_folder: str, modality: str
                         ) -> Dict[int, Dict[str, str]]:
    """``detection_id -> attributes`` for the detections of a flight.

    A detection that belongs to a track is what its track is - the animal
    was identified as a whole, not box by box - and has its track's box
    colour. One that belongs to none has the species the detector's class
    maps to, no sex or age, and the neutral box colour.
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
