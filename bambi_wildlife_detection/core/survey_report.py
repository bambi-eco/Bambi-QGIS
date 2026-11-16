# -*- coding: utf-8 -*-
"""The survey in one page.

Every analytic of a survey writes its own files - a census here, a grid
there, a density with its interval in a third place. A report is what is
handed on: to the land owner, the authority, the file. This module puts
what a survey's analytics found into **one HTML file that needs nothing
else** - no images beside it, no style sheet, no script, no connection:
the tables, the charts and the map are in the file itself, and it prints.

**What is in it.** What was flown, when and in what weather; a map of the
flights and the animals; the counts by species, sex and age and the groups
the animals formed; every density the survey has, each with the method it
came from and its interval; where the animals were against the ground that
was looked at; what changed between visits; how the animals took the drone;
how far off the places may be; and the settings it was all made with.

**What is not.** The report computes nothing. It reads the result files
the analytics wrote (:func:`collect`) and says of each when it was written -
an analytic that was not run is listed as not run, and one that was run
before the survey was changed is as old as its date says.

Free of QGIS and Qt, and of numpy.
"""

import json
import math
import os
from collections import OrderedDict
from datetime import datetime, timezone
from html import escape
from typing import Dict, List, Optional, Sequence, Tuple

from . import census as census_core

#: The result files of a survey: ``key -> (file name with {source}, what it
#: is called)``. Distance sampling and the population estimate may come per
#: species (``..._<species>.json``), see :func:`collect`.
RESULTS = OrderedDict([
    ("census", ("census_summary_{source}.json", "Census summary")),
    ("groups", ("groups_{source}.json", "Groups")),
    ("grid", ("grid_density_{source}.json", "Grid density")),
    ("distance", ("distance_sampling_{source}.json", "Distance sampling")),
    ("population", ("population_estimate.json",
                    "Population estimation (transects)")),
    ("habitat", ("habitat.json", "Habitat context")),
    ("trend", ("trend_{source}.json", "Compare visits")),
    ("reaction", ("drone_reaction.json", "Reaction to the drone")),
    ("position", ("position_uncertainty_{source}.json",
                  "Position uncertainty")),
])
PER_SPECIES = ("distance", "population")

FILE_NAME = "survey_report.html"

#: Colours of a species nobody chose a colour for, in turn.
PALETTE = ("#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9",
           "#d55e00", "#f0e442", "#000000")
GROUND = "#9a9a9a"
ANIMALS = "#d6601c"


# ---------------------------------------------------------------------------
# Reading what the analytics wrote
# ---------------------------------------------------------------------------

def _read(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _written(path: str) -> str:
    """When a file was last written, as it is said in a report."""
    try:
        moment = datetime.fromtimestamp(os.path.getmtime(path))
    except OSError:
        return ""
    return moment.strftime("%Y-%m-%d %H:%M")


def collect(analytics_folder: str, source: str = "tracks") -> dict:
    """The results a survey has: ``{"found": {key: [entry, ...]},
    "missing": [what is called]}``.

    An entry is ``{"species", "document", "file", "written"}``; *species*
    is ``""`` for a result of all animals. Distance sampling and the
    population estimate are listed once per species where they were run
    species by species.
    """
    found: Dict[str, List[dict]] = OrderedDict()
    missing: List[str] = []
    try:
        names = sorted(os.listdir(analytics_folder))
    except OSError:
        names = []
    for key, (pattern, title) in RESULTS.items():
        name = pattern.format(source=source)
        entries = []
        path = os.path.join(analytics_folder, name)
        document = _read(path) if name in names else None
        if document is not None:
            entries.append({"species": "", "document": document,
                            "file": name, "written": _written(path)})
        if key in PER_SPECIES:
            stem = name[:-len(".json")]
            for other in names:
                per_species = other.startswith(stem + "_")
                if not per_species or not other.endswith(".json"):
                    continue
                path = os.path.join(analytics_folder, other)
                document = _read(path)
                if document is None:
                    continue
                species = document.get("species") or other[
                    len(stem) + 1:-len(".json")]
                entries.append({"species": str(species),
                                "document": document, "file": other,
                                "written": _written(path)})
        if entries:
            found[key] = entries
        else:
            missing.append(title)
    return {"found": found, "missing": missing}


def first(results: dict, key: str) -> Optional[dict]:
    """The document of a result that is not per species, or ``None``."""
    entries = (results.get("found") or {}).get(key) or []
    for entry in entries:
        if not entry["species"]:
            return entry["document"]
    return None


def flights_of(document: dict) -> Optional[List[str]]:
    """The flights a result says it was made from, or ``None`` where it
    does not say. Most results list their flights by name, the habitat
    summary one entry per flight, and the comparison of visits the flights
    of each visit."""
    names: List[str] = []
    flights = document.get("flights")
    if isinstance(flights, list):
        for entry in flights:
            if isinstance(entry, dict):
                entry = entry.get("flight", entry.get("name"))
            if entry is not None:
                names.append(str(entry))
    elif isinstance(document.get("visits"), list):
        for visit in document["visits"]:
            if isinstance(visit, dict):
                names += [str(name) for name in visit.get("flights") or []]
    return names or None


def _species_key(names) -> Optional[Tuple[str, ...]]:
    if not names or not isinstance(names, (list, tuple)):
        return None
    return tuple(sorted(str(name).lower() for name in names))


def not_of_survey(document: dict, flights: Sequence[str], species=None,
                  per_species: bool = False) -> str:
    """Why a result is not of a survey with these *flights* and this
    *species* filter (``None``: all) - ``""`` when it is, or does not say
    what it was made from. A result made for one species (*per_species*)
    is not held to the survey's filter."""
    mine = sorted(str(name) for name in flights)
    theirs = flights_of(document)
    if mine and theirs is not None and sorted(theirs) != mine:
        return "made from other flights: " + ", ".join(theirs)
    if per_species or "species_filter" not in document:
        return ""
    made_for = document["species_filter"]
    if _species_key(made_for) != _species_key(species):
        named = ", ".join(str(name) for name in made_for or [])
        return "made for other species: " + (named or "all")
    return ""


def out_of_step(report: dict) -> List[Tuple[str, str]]:
    """The results that are not of the survey the report describes:
    ``[(what the result is called, why)]``.

    A result is read as it was written - and it may have been written when
    other flights were ticked, or other species chosen. Its figures are
    then right for that survey and wrong beside the rest of the page.
    Results that do not say what they were made from are not judged.
    """
    flights = [flight.get("name") for flight in report.get("flights") or []]
    found = (report.get("results") or {}).get("found") or {}
    out = []
    for key, entries in found.items():
        for entry in entries:
            who = f" ({entry['species']})" if entry["species"] else ""
            why = not_of_survey(entry["document"], flights,
                                report.get("species_filter"),
                                per_species=bool(entry["species"]))
            if why:
                out.append((RESULTS[key][1] + who, why))
    return out


def cautions(report: dict) -> List[str]:
    """What a reader has to know before believing the page - said at its
    top, in whole sentences."""
    odd = out_of_step(report)
    if not odd:
        return []
    names = ", ".join(title for title, _why in odd)
    one = len(odd) == 1
    return [
        f"Not of one piece: {names} {'was' if one else 'were'} made when "
        "the survey had other flights or species than this report lists "
        "(see 'What the report was made from' under Settings). Run "
        f"{'it' if one else 'them'} again and write the report anew, for "
        "figures that belong together."]


# ---------------------------------------------------------------------------
# Small things a page is made of
# ---------------------------------------------------------------------------

def _esc(value) -> str:
    return escape("" if value is None else str(value), quote=True)


def _num(value, digits: int = 1, dash: str = "&ndash;") -> str:
    if value is None:
        return dash
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    if not math.isfinite(number):
        return dash
    if digits == 0:
        return f"{number:,.0f}"
    return f"{number:,.{digits}f}"


def _interval(value, low, high, digits: int = 1) -> str:
    if value is None:
        return "&ndash;"
    text = _num(value, digits)
    if low is not None and high is not None:
        text += f" ({_num(low, digits)}&ndash;{_num(high, digits)})"
    return text


def _table(headers: Sequence[str], rows: Sequence[Sequence],
           numeric: Sequence[int] = (), bold_first: bool = False,
           raw: Sequence[int] = ()) -> str:
    """An HTML table. Cells are escaped unless their column is in *raw*
    (already HTML); columns in *numeric* are right-aligned."""
    numeric, raw = set(numeric), set(raw)
    out = ["<table>"]
    if any(str(header) for header in headers):
        out.append("<thead><tr>")
        for index, header in enumerate(headers):
            cls = " class='n'" if index in numeric else ""
            out.append(f"<th{cls}>{_esc(header)}</th>")
        out.append("</tr></thead>")
    out.append("<tbody>")
    for position, row in enumerate(rows):
        cls = " class='total'" if bold_first and position == 0 else ""
        out.append(f"<tr{cls}>")
        for index, cell in enumerate(row):
            text = ("" if cell is None else str(cell)) if index in raw \
                else _esc(cell)
            align = " class='n'" if index in numeric else ""
            out.append(f"<td{align}>{text}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _note(text: str) -> str:
    return f"<p class='note'>{_esc(text)}</p>" if text else ""


def _details(title: str, text: str) -> str:
    if not text:
        return ""
    return (f"<details><summary>{_esc(title)}</summary>"
            f"<p class='note'>{_esc(text)}</p></details>")


def species_colours(names: Sequence[str],
                    chosen: Optional[Dict[str, str]] = None
                    ) -> Dict[str, str]:
    """A colour per species: the project's own where it has one, else the
    next of :data:`PALETTE`."""
    chosen = {str(key).lower(): value for key, value in (chosen or {}).items()}
    colours, spare = OrderedDict(), 0
    for name in names:
        colour = chosen.get(str(name).lower())
        if not colour:
            colour = PALETTE[spare % len(PALETTE)]
            spare += 1
        colours[name] = colour
    return colours


# ---------------------------------------------------------------------------
# Charts, as SVG
# ---------------------------------------------------------------------------

def _svg_open(width: int, height: int, cls: str = "") -> str:
    """The opening tag of a chart: drawn at its own size, and smaller
    where the page is narrower."""
    name = f" class='{cls}'" if cls else ""
    return (f"<svg viewBox='0 0 {width} {height}' width='{width}' "
            f"height='{height}' role='img'{name} "
            "xmlns='http://www.w3.org/2000/svg'>")


def _nice_step(top: float, most: int = 5) -> float:
    if top <= 0:
        return 1.0
    raw = top / most
    power = 10.0 ** math.floor(math.log10(raw))
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        if factor * power >= raw:
            return factor * power
    return 10.0 * power


def _tick(value: float, step: float) -> str:
    return f"{value:,.0f}" if step >= 1 and float(step).is_integer() \
        else f"{value:g}"


def svg_bars(groups: Sequence[str], series: Sequence[Tuple[str, Sequence[float], str]],
             unit: str = "", width: int = 640, height: int = 240,
             percent: bool = False) -> str:
    """A grouped bar chart: one group of bars per entry of *groups*, one
    bar per series ``(name, values, colour)``."""
    if not groups or not series:
        return ""
    left, right, top, bottom = 52, 12, 14, 46
    plot_w, plot_h = width - left - right, height - top - bottom
    highest = max([1e-9, *(max(values) if values else 0
                           for _n, values, _c in series)])
    step = _nice_step(highest * 1.05)
    ceiling = step * math.ceil(highest * 1.05 / step)
    out = [_svg_open(width, height)]
    marks = int(round(ceiling / step))
    for mark in range(marks + 1):
        value = step * mark
        y = top + plot_h - plot_h * value / ceiling
        out.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width - right}' "
                   f"y2='{y:.1f}' class='grid'/>")
        label = f"{value * 100:.0f}%" if percent else _tick(value, step)
        out.append(f"<text x='{left - 6}' y='{y + 4:.1f}' "
                   f"class='tick end'>{label}</text>")
    slot = plot_w / len(groups)
    bar = min(46.0, slot * 0.8 / len(series))
    for g, group in enumerate(groups):
        start = left + slot * g + (slot - bar * len(series)) / 2.0
        for s, (_name, values, colour) in enumerate(series):
            value = values[g] if g < len(values) and values[g] else 0.0
            h = plot_h * value / ceiling
            out.append(
                f"<rect x='{start + bar * s:.1f}' "
                f"y='{top + plot_h - h:.1f}' width='{bar - 1.5:.1f}' "
                f"height='{h:.1f}' fill='{_esc(colour)}'/>")
        out.append(f"<text x='{left + slot * (g + 0.5):.1f}' "
                   f"y='{top + plot_h + 15}' class='tick mid'>"
                   f"{_esc(group)}</text>")
    out.append(f"<line x1='{left}' y1='{top + plot_h}' x2='{width - right}' "
               f"y2='{top + plot_h}' class='axis'/>")
    x = left
    for name, _values, colour in series:
        out.append(f"<rect x='{x}' y='{height - 14}' width='10' height='10' "
                   f"fill='{_esc(colour)}'/>")
        out.append(f"<text x='{x + 14}' y='{height - 5}' class='tick'>"
                   f"{_esc(name)}</text>")
        x += 22 + 6.4 * len(str(name))
    if unit:
        out.append(f"<text x='{width - right}' y='{height - 5}' "
                   f"class='tick end'>{_esc(unit)}</text>")
    out.append("</svg>")
    return "".join(out)


def _shares(counts: Sequence[float]) -> List[float]:
    total = float(sum(counts))
    return [count / total if total else 0.0 for count in counts]


def svg_histogram(layer: dict, width: int = 420, height: int = 190) -> str:
    """One environment layer: how its values are spread over the ground
    that was looked at (grey) and at the animals (colour)."""
    edges = layer.get("bin_edges") or []
    if len(edges) < 2:
        return ""
    ground = _shares(layer.get("available_counts") or [])
    used = _shares(layer.get("used_counts") or [])
    left, right, top, bottom = 40, 10, 10, 30
    plot_w, plot_h = width - left - right, height - top - bottom
    ceiling = max(ground + used + [0.01]) * 1.1
    low, span = edges[0], (edges[-1] - edges[0]) or 1.0
    out = [_svg_open(width, height)]

    def x_of(value: float) -> float:
        return left + (value - low) / span * plot_w

    for index, share in enumerate(ground):
        x0, x1 = x_of(edges[index]), x_of(edges[index + 1])
        h = plot_h * share / ceiling
        out.append(f"<rect x='{x0 + 0.5:.1f}' y='{top + plot_h - h:.1f}' "
                   f"width='{max(0.5, x1 - x0 - 1):.1f}' height='{h:.1f}' "
                   f"fill='{GROUND}'/>")
    for index, share in enumerate(used):
        if index + 1 >= len(edges) or share <= 0:
            continue
        x0, x1 = x_of(edges[index]), x_of(edges[index + 1])
        inset = (x1 - x0) * 0.28
        h = plot_h * share / ceiling
        out.append(f"<rect x='{x0 + inset:.1f}' y='{top + plot_h - h:.1f}' "
                   f"width='{max(0.5, x1 - x0 - 2 * inset):.1f}' "
                   f"height='{h:.1f}' fill='{ANIMALS}'/>")
    out.append(f"<line x1='{left}' y1='{top + plot_h}' x2='{width - right}' "
               f"y2='{top + plot_h}' class='axis'/>")
    for mark in range(5):
        value = low + span * mark / 4.0
        text = f"{value:.2f}".rstrip("0").rstrip(".") or "0"
        out.append(f"<text x='{x_of(value):.1f}' y='{top + plot_h + 14}' "
                   f"class='tick mid'>{text}</text>")
    for mark in (0, 2, 4):
        share = ceiling * mark / 4.0
        y = top + plot_h - plot_h * mark / 4.0
        out.append(f"<text x='{left - 5}' y='{y + 4:.1f}' class='tick end'>"
                   f"{share * 100:.0f}%</text>")
    out.append("</svg>")
    return "".join(out)


def svg_trend(points: dict, width: int = 640, height: int = 240) -> str:
    """The visits of one species: a point per visit with its interval and
    the fitted rate of change. *points* is ``{"days", "labels", "values",
    "lows", "highs", "fitted", "unit"}``."""
    values = [v for v in points.get("values") or [] if v is not None]
    if not values:
        return ""
    days = points["days"]
    fitted = points.get("fitted") or []
    highs = [h for h in points.get("highs") or [] if h is not None]
    peak = max(values + [f for f in fitted if f is not None])
    reach = min(max(highs), 3.0 * peak) if highs else peak
    ceiling_raw = max(peak, reach, 1e-9) * 1.08
    step = _nice_step(ceiling_raw)
    ceiling = step * math.ceil(ceiling_raw / step)
    left, right, top, bottom = 52, 20, 24, 34
    plot_w, plot_h = width - left - right, height - top - bottom
    start, span = min(days), (max(days) - min(days)) or 1.0

    def x_of(day: float) -> float:
        return left + (0.06 + 0.88 * (day - start) / span) * plot_w

    def y_of(value: float) -> float:
        return top + plot_h - min(value, ceiling) / ceiling * plot_h

    out = [_svg_open(width, height)]
    for mark in range(int(round(ceiling / step)) + 1):
        value = step * mark
        out.append(f"<line x1='{left}' y1='{y_of(value):.1f}' "
                   f"x2='{width - right}' y2='{y_of(value):.1f}' "
                   "class='grid'/>")
        out.append(f"<text x='{left - 6}' y='{y_of(value) + 4:.1f}' "
                   f"class='tick end'>{_tick(value, step)}</text>")
    out.append(f"<line x1='{left}' y1='{top + plot_h}' x2='{width - right}' "
               f"y2='{top + plot_h}' class='axis'/>")
    last_right = -1e9
    for day, label in zip(days, points["labels"]):
        text = str(label).split(" to ")[0]
        x = x_of(day)
        half = 3.1 * len(text)
        if x - half < last_right + 8:
            continue
        out.append(f"<text x='{x:.1f}' y='{top + plot_h + 15}' "
                   f"class='tick mid'>{_esc(text)}</text>")
        last_right = x + half
    if len(fitted) == len(days) and all(f is not None for f in fitted):
        path = " ".join(f"{x_of(day):.1f},{y_of(value):.1f}"
                        for day, value in zip(days, fitted))
        out.append(f"<polyline points='{path}' class='fit'/>")
    for day, value, low, high in zip(days, points["values"],
                                     points["lows"], points["highs"]):
        if value is None:
            continue
        x = x_of(day)
        if low is not None and high is not None:
            out.append(f"<line x1='{x:.1f}' y1='{y_of(low):.1f}' "
                       f"x2='{x:.1f}' y2='{y_of(high):.1f}' class='bar'/>")
        out.append(f"<circle cx='{x:.1f}' cy='{y_of(value):.1f}' r='4' "
                   "class='dot'/>")
    out.append(f"<text x='{width - right}' y='12' class='tick end'>"
               f"{_esc(points.get('unit', ''))}</text>")
    out.append("</svg>")
    return "".join(out)


def _ring_path(ring, x_of, y_of) -> str:
    return "M" + " L".join(f"{x_of(x):.1f},{y_of(y):.1f}"
                           for x, y in ring) + " Z"


def svg_map(data: dict, width: int = 760, max_height: int = 620) -> str:
    """The survey from above: the survey area, the ground the flights saw,
    the routes flown and the animals, each in the colour of its species.

    *data* is ``{"area": [rings], "exclusions": [rings], "seen": [rings],
    "routes": [{"flight", "points": [(x, y)]}], "animals": [{"x", "y",
    "species"}], "colours": {species: colour}}`` in the project's
    coordinates (metres); ``""`` where there is nothing to draw.
    (``"routes_missing"``, the flights whose route could not be placed, is
    said beside the map - see :func:`section_map`.)
    """
    xs, ys = [], []
    for key in ("area", "exclusions", "seen"):
        for ring in data.get(key) or []:
            xs += [point[0] for point in ring]
            ys += [point[1] for point in ring]
    for route in data.get("routes") or []:
        xs += [point[0] for point in route["points"]]
        ys += [point[1] for point in route["points"]]
    for animal in data.get("animals") or []:
        xs.append(animal["x"])
        ys.append(animal["y"])
    if not xs:
        return ""
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    margin = 0.04 * max(span_x, span_y)
    min_x, min_y = min_x - margin, min_y - margin
    span_x, span_y = span_x + 2 * margin, span_y + 2 * margin
    scale = min(width / span_x, max_height / span_y)
    w, h = span_x * scale, span_y * scale

    def x_of(x: float) -> float:
        return (x - min_x) * scale

    def y_of(y: float) -> float:
        return h - (y - min_y) * scale

    out = [_svg_open(round(w), round(h), "map"),
           f"<rect width='{w:.0f}' height='{h:.0f}' class='land'/>"]
    for ring in data.get("seen") or []:
        out.append(f"<path d='{_ring_path(ring, x_of, y_of)}' class='seen'/>")
    for ring in data.get("area") or []:
        out.append(f"<path d='{_ring_path(ring, x_of, y_of)}' class='plot'/>")
    for ring in data.get("exclusions") or []:
        out.append(f"<path d='{_ring_path(ring, x_of, y_of)}' "
                   "class='excluded'/>")
    for route in data.get("routes") or []:
        if len(route["points"]) < 2:
            continue
        path = " ".join(f"{x_of(x):.1f},{y_of(y):.1f}"
                        for x, y in route["points"])
        out.append(f"<polyline points='{path}' class='route'>"
                   f"<title>{_esc(route.get('flight', ''))}</title>"
                   "</polyline>")
    colours = data.get("colours") or {}
    for animal in data.get("animals") or []:
        colour = colours.get(animal.get("species"), "#000000")
        out.append(f"<circle cx='{x_of(animal['x']):.1f}' "
                   f"cy='{y_of(animal['y']):.1f}' r='3.6' "
                   f"fill='{_esc(colour)}' class='animal'>"
                   f"<title>{_esc(animal.get('species', ''))}</title>"
                   "</circle>")

    # a scale bar of a round length, about a fifth of the width
    length = _nice_step(span_x, 5)
    bar = length * scale
    out.append(f"<line x1='14' y1='{h - 14:.0f}' x2='{14 + bar:.1f}' "
               f"y2='{h - 14:.0f}' class='scale'/>")
    label = f"{length / 1000:g} km" if length >= 1000 else f"{length:g} m"
    out.append(f"<text x='{14 + bar / 2:.1f}' y='{h - 20:.0f}' "
               f"class='tick mid'>{label}</text>")
    out.append(f"<path d='M{w - 22:.0f},{34} l6,16 l-6,-4 l-6,4 Z' "
               "class='north'/>")
    out.append(f"<text x='{w - 22:.0f}' y='28' class='tick mid'>N</text>")
    out.append("</svg>")
    return "".join(out)


def _legend(colours: Dict[str, str], extra: Sequence[Tuple[str, str]] = ()
            ) -> str:
    items = [f"<span><i style='background:{_esc(colour)}'></i>"
             f"{_esc(name)}</span>" for name, colour in colours.items()]
    items += [f"<span><i class='{_esc(cls)}'></i>{_esc(name)}</span>"
              for name, cls in extra]
    return f"<p class='legend'>{''.join(items)}</p>" if items else ""


# ---------------------------------------------------------------------------
# The sections
# ---------------------------------------------------------------------------

def _species_title(name) -> str:
    return "All animals" if name in (census_core.ALL_SPECIES, "all",
                                     "All species") else str(name)


#: The flights looked at this many times the survey area or more: they saw
#: it more than once.
REPEATED = 1.2


def seen_more_than_once(census: dict) -> bool:
    """Whether the flights of a census looked at more ground than the
    survey area has - the same ground on several days, or twice on one. The
    counts of such flights do not add up to one count of the area."""
    plot = census.get("plot_ha")
    if not plot:
        return False
    for row in _census_totals(census):
        if row["species"] == census_core.ALL_SPECIES:
            covered = row.get("covered_ha")
            return covered is not None and covered > REPEATED * plot
    return False


def _all_first(rows: Sequence[dict]) -> List[dict]:
    """*rows* with the one for all animals in front."""
    total = [row for row in rows
             if row["species"] == census_core.ALL_SPECIES]
    return total + [row for row in rows
                    if row["species"] != census_core.ALL_SPECIES]


def _census_totals(census: dict) -> List[dict]:
    """The rows of the whole survey, one per species and one for all."""
    rows = census.get("rows") or []
    several = len(census.get("flights") or []) > 1
    wanted = census_core.ALL_FLIGHTS if several else None
    return [row for row in rows
            if (row["flight"] == wanted if several else True)]


def flying_days(report: dict) -> int:
    """On how many different days the flights of a report were flown."""
    return len({flight.get("date") for flight in report.get("flights") or []
                if flight.get("date")})


def section_survey(report: dict) -> str:
    flights = report.get("flights") or []
    out = ["<h2>The survey</h2>"]
    facts = [("Camera", report.get("camera")),
             ("What is counted", "tracked animals, each once"
              if report.get("source") == "tracks"
              else "boxes - an animal counts once per frame it is on"),
             ("Species", ", ".join(report["species_filter"])
              if report.get("species_filter") else "all"),
             ("Coordinate system", report.get("crs"))]
    area = report.get("survey_area")
    if area:
        size = area.get("adjusted_ha")
        text = "exclusion zones only" if size is None \
            else f"{_num(size, 2, '–')} ha"
        if area.get("exclusions"):
            text += f", {len(area['exclusions'])} exclusion zone(s) taken off"
        facts.append(("Survey area", text))
    else:
        facts.append(("Survey area", "none drawn: the whole flights count"))
    out.append(_table(["", ""], facts))
    if flights:
        weather_keys = [("air_temperature_c", "Air °C", 1),
                        ("wind_speed_ms", "Wind m/s", 1),
                        ("cloud_cover_pct", "Cloud %", 0),
                        ("rain_24h_mm", "Rain 24 h mm", 1)]
        shown = [key for key in weather_keys
                 if any((flight.get("weather") or {}).get(key[0]) is not None
                        for flight in flights)]
        headers = ["Flight", "Date", "Time", "Frames", "Flown (km)",
                   "Animals"] + [label for _k, label, _d in shown]
        rows = []
        for flight in flights:
            row = [flight.get("name"), flight.get("date") or "–",
                   flight.get("time") or "–", flight.get("frames"),
                   _num(flight.get("route_km"), 2, "–"),
                   flight.get("animals")]
            row += [_num((flight.get("weather") or {}).get(key), digits, "–")
                    for key, _label, digits in shown]
            rows.append(row)
        out.append(_table(headers, rows,
                          numeric=range(3, len(headers))))
        if not shown:
            out.append(_note(
                "No weather is known for these flights: run 'Fetch "
                "Weather' of the Environmental Context to have it here."))
    return "".join(out)


def section_map(report: dict) -> str:
    data = report.get("map") or {}
    drawing = svg_map(data)
    canvas = report.get("canvas_png")
    if not drawing and not canvas:
        return ""
    out = ["<h2>Map</h2>"]
    if drawing:
        out.append(f"<figure>{drawing}")
        extra = []
        if any(len(route["points"]) > 1
               for route in data.get("routes") or []):
            extra.append(("route flown", "route"))
        if data.get("seen"):
            extra.append(("ground the cameras saw", "seen"))
        if data.get("area"):
            extra.append(("survey area", "plot"))
        if data.get("exclusions"):
            extra.append(("exclusion zone", "excluded"))
        out.append(_legend(data.get("colours") or {}, extra))
        out.append("<figcaption>The flights and the animals counted, in "
                   "the project's coordinates. North is up.</figcaption>"
                   "</figure>")
        unplaced = data.get("routes_missing") or []
        if unplaced:
            out.append(_note(
                f"Not on the map: the route of {', '.join(unplaced)}. A "
                "flight's camera positions are placed by its elevation "
                "model, which is not set for it."))
    if canvas:
        out.append("<figure><img alt='The map as it was shown in QGIS' "
                   f"src='data:image/png;base64,{_esc(canvas)}'/>"
                   "<figcaption>The map as it was shown in QGIS when the "
                   "report was written.</figcaption></figure>")
    return "".join(out)


def section_counts(report: dict) -> str:
    census = first(report.get("results") or {}, "census")
    if not census:
        return ""
    rows = _census_totals(census)
    if not rows:
        return ""
    with_area = census.get("plot_ha") is not None
    counted = "In the survey area" if with_area else "Counted"
    out = ["<h2>Counts</h2>"]
    total = [row for row in rows
             if row["species"] == census_core.ALL_SPECIES]
    species = [row for row in rows
               if row["species"] != census_core.ALL_SPECIES]
    ordered = total + species
    headers = ["Species", counted, "Approved", "Groups", "Mean group size",
               "Largest group"]
    if with_area:
        headers[2:2] = ["Outside", "Not placed"]
    table = []
    for row in ordered:
        cells = [_species_title(row["species"]), row["inside"],
                 row["approved"],
                 "–" if row.get("groups") is None else row["groups"],
                 _num(row.get("mean_group_size"), 2, "–"),
                 "–" if row.get("largest_group") is None
                 else row["largest_group"]]
        if with_area:
            cells[2:2] = [row["outside"], row["unplaced"]]
        table.append(cells)
    out.append(_table(headers, table, numeric=range(1, len(headers)),
                      bold_first=bool(total)))

    if len(species) > 1:
        names = [row["species"] for row in species]
        out.append("<figure>" + svg_bars(
            names, [(counted, [row["inside"] for row in species], ANIMALS)],
            unit="animals") + "</figure>")

    for key, title in (("sex", "Sex"), ("age", "Age")):
        labels = census.get(f"{key}_labels") or []
        known = [label for label in labels if label != census_core.UNKNOWN]
        if not known:
            continue
        out.append(f"<h3>{title}</h3>")
        out.append(_table(
            ["Species", *labels],
            [[_species_title(row["species"]),
              *(row[key].get(label, 0) for label in labels)]
             for row in ordered],
            numeric=range(1, len(labels) + 1), bold_first=bool(total)))
        # what is not known is no kind of its own: grey, not a colour
        palette = species_colours(labels, {census_core.UNKNOWN: GROUND})
        if species:
            out.append("<figure>" + svg_bars(
                [row["species"] for row in species],
                [(label, [row[key].get(label, 0) for row in species],
                  palette[label]) for label in labels],
                unit="animals") + "</figure>")
    if len(census.get("flights") or []) > 1:
        out.append("<h3>Per flight</h3>")
        of_all = [row for row in census["rows"]
                  if row["species"] == census_core.ALL_SPECIES]
        per_flight = [row for row in of_all
                      if row["flight"] != census_core.ALL_FLIGHTS]
        out.append(_table(
            ["Flight", counted, "Ground seen (ha)", "Per km² of ground seen"],
            [[row["flight"], row["inside"],
              _num(row.get("covered_ha"), 2, "–"),
              _num(row.get("per_km2_covered"), 2, "–")]
             for row in per_flight], numeric=(1, 2, 3)))
    days = flying_days(report)
    if days > 1:
        out.append(_note(
            f"The flights were flown on {days} different days, and these "
            "are the animals of all of them together: an animal that was "
            "there each time is counted each time. The flights are listed "
            "one by one above; 'Over time' compares the visits."))
    elif seen_more_than_once(census):
        out.append(_note(
            "The flights looked at more ground than the survey area has: "
            "part of it was seen more than once, and an animal there may "
            "be counted more than once."))
    out.append(_note(
        "Minimum counts: an animal that was there and not found is not in "
        "them. 'Approved' are the animals a person looked at and "
        "confirmed."))
    return "".join(out)


def density_rows(results: dict) -> List[list]:
    """Every density a survey has, one row per method and species:
    ``[method, species, animals per km² (HTML), number in the survey area
    (HTML), what it rests on]``."""
    rows = []
    census = first(results, "census")
    if census:
        totals = _all_first(_census_totals(census))
        for row in totals:
            if row.get("per_km2_covered") is not None:
                rows.append([
                    "Census, over the ground seen",
                    _species_title(row["species"]),
                    _num(row["per_km2_covered"], 2), "&ndash;",
                    f"{_esc(row['inside'])} animals on "
                    f"{_num(row.get('covered_ha'), 1)} ha looked at; no "
                    "interval"])
        if not seen_more_than_once(census):
            for row in totals:
                if row.get("per_km2_plot") is not None:
                    rows.append([
                        "Census, over the survey area",
                        _species_title(row["species"]),
                        _num(row["per_km2_plot"], 2),
                        _num(row["inside"], 0),
                        "assumes the flights saw all of the area, once; no "
                        "interval"])
    grid = first(results, "grid")
    if grid:
        for name, stats in (grid.get("statistics") or {}).items():
            density = stats.get("density_per_km2") or {}
            if density.get("estimate") is None:
                continue
            total = stats.get("total_in_survey_area") or {}
            cells = (grid.get("summary") or {}).get("counted", 0)
            rows.append([
                "Grid density", _species_title(name),
                _interval(density["estimate"], density.get("low"),
                          density.get("high"), 2),
                _interval(total.get("estimate"), total.get("low"),
                          total.get("high"), 0),
                f"mean over {_esc(cells)} cells, interval from resampling "
                "them"])
    for entry in (results.get("found") or {}).get("distance") or []:
        document = entry["document"]
        if document.get("density_per_km2") is None:
            continue
        ci = document.get("density_ci95") or [None, None]
        area = document.get("abundance_survey_area_ci95") or [None, None]
        rows.append([
            "Distance sampling", entry["species"] or "All animals",
            _interval(document["density_per_km2"], ci[0], ci[1], 2),
            _interval(document.get("abundance_in_survey_area"), area[0],
                      area[1], 0),
            f"{_esc(document.get('n', 0))} observations, "
            f"{_esc(document.get('best_model', ''))}, effective strip "
            f"half-width {_num(document.get('effective_strip_width_m'), 1)} "
            "m"])
    for entry in (results.get("found") or {}).get("population") or []:
        document = entry["document"]
        for method, estimate in (document.get("estimates") or {}).items():
            density = estimate.get("density_per_100ha")
            if density is None:
                continue
            ci = estimate.get("ci95") or [None, None]
            low = None if ci[0] is None else ci[0]
            high = None if ci[1] is None else ci[1]
            rows.append([
                f"Transects ({_esc(method)})",
                entry["species"] or "All animals",
                _interval(density, low, high, 2), "&ndash;",
                f"{_esc(document.get('n_transects', 0))} transects, "
                f"{_num(document.get('total_ha'), 1)} ha; per km², which "
                "is per 100 ha"])
    return rows


def section_density(report: dict) -> str:
    rows = density_rows(report.get("results") or {})
    if not rows:
        return ""
    out = ["<h2>Density</h2>",
           _table(["Method", "Species", "Animals per km² (95 % interval)",
                   "In the survey area", "Rests on"], rows,
                  numeric=(2, 3), raw=(0, 2, 3, 4)),
           _note("Each method answers a slightly different question. The "
                 "census figures are what was counted over an area, with "
                 "nothing allowed for animals that were missed. Distance "
                 "sampling allows for animals missed further from the "
                 "line; the grid and the transects measure how much the "
                 "counts vary over the ground. Where they differ, the "
                 "method notes below say what each assumes.")]
    return "".join(out)


def section_where(report: dict) -> str:
    results = report.get("results") or {}
    grid = first(results, "grid")
    habitat = first(results, "habitat")
    if not grid and not habitat:
        return ""
    out = ["<h2>Where the animals were</h2>"]
    if grid:
        summary = grid.get("summary") or {}
        rows = []
        for name, stats in (grid.get("statistics") or {}).items():
            dispersion = stats.get("dispersion") or {}
            occupancy = stats.get("occupancy") or {}
            share = occupancy.get("share")
            rows.append([
                "All animals" if name == "all" else name,
                stats.get("animals", 0),
                _num(dispersion.get("index"), 2, "–"),
                dispersion.get("pattern") or "–",
                "–" if share is None else f"{share * 100:.0f} %",
                stats.get("hot_spots", 0)])
        out.append("<h3>Over the grid</h3>")
        out.append(_note(
            f"{grid.get('grid_text', 'Grid')}: {summary.get('counted', 0)} "
            f"of {summary.get('cells', 0)} cells seen well enough to "
            "count."))
        out.append(_table(
            ["Species", "Animals", "Variance-to-mean", "Pattern",
             "Cells with animals", "Hot-spot cells"], rows,
            numeric=(1, 2, 4, 5)))
    if habitat:
        layers = habitat.get("layers") or {}
        charts = []
        for layer in layers.values():
            drawing = svg_histogram(layer)
            if not drawing:
                continue
            unit = layer.get("unit") or ""
            ground = (layer.get("available") or {}).get("mean")
            used = (layer.get("used") or {}).get("mean")
            charts.append(
                f"<figure class='small'>{drawing}<figcaption>"
                f"<b>{_esc(layer.get('title', ''))}</b><br>ground: mean "
                f"{_num(ground, 2)}{_esc(unit)} &middot; animals: mean "
                f"{_num(used, 2)}{_esc(unit)}</figcaption></figure>")
        cover = habitat.get("landcover") or {}
        classes = cover.get("classes") or []
        if charts or classes:
            out.append("<h3>Habitat</h3>")
            what = "boxes" if habitat.get("source") == "detections" \
                else "animals"
            out.append(_note(
                f"What the {habitat.get('n_tracks', 0)} {what} had around "
                f"them (within {_num(habitat.get('radius_m'), 0)} m) "
                "against the ground the flights looked at."))
        if classes:
            names = [entry["name"] for entry in classes]
            bars = svg_bars(
                names,
                [("ground looked at",
                  [entry.get("available_share") or 0 for entry in classes],
                  GROUND),
                 (what, [entry.get("used_share") or 0 for entry in classes],
                  ANIMALS)], percent=True)
            out.append(f"<figure>{bars}<figcaption>Land cover: its share "
                       f"of the ground and of the {_esc(what)}."
                       "</figcaption></figure>")
            out.append(_table(
                ["Land cover", "Share of the ground", f"Share of the {what}",
                 "Used / available"],
                [[entry["name"],
                  f"{(entry.get('available_share') or 0) * 100:.0f} %",
                  "–" if entry.get("used_share") is None
                  else f"{entry['used_share'] * 100:.0f} %",
                  _num(entry.get("selection_ratio"), 2, "–")]
                 for entry in classes], numeric=(1, 2, 3)))
        if charts:
            out.append("<div class='charts'>" + "".join(charts) + "</div>")
            out.append(_legend({"ground looked at": GROUND, what: ANIMALS}))
    return "".join(out)


def section_trend(report: dict) -> str:
    trend = first(report.get("results") or {}, "trend")
    if not trend:
        return ""
    from . import survey_trend

    out = ["<h2>Over time</h2>"]
    visits = trend.get("visits") or []
    by_density = trend.get("basis") == survey_trend.BASIS_GROUND
    keys = (("density_per_km2", "density_low", "density_high")
            if by_density else ("animals", "animals_low", "animals_high"))
    level = trend.get("level", survey_trend.LEVEL)
    labels = {visit["visit"]: visit["label"] for visit in visits}
    names = trend.get("species") or []
    ordered = [name for name in names if name == survey_trend.ALL] + [
        name for name in names if name != survey_trend.ALL]
    for species in ordered:
        entries = (trend.get("series") or {}).get(species) or []
        if not entries:
            continue
        name = "All animals" if species == survey_trend.ALL else species
        fit = (trend.get("trends") or {}).get(species) or {}
        out.append(f"<h3>{_esc(name)}</h3>")
        if fit:
            out.append(f"<p><b>{_esc(survey_trend.trend_sentence(fit, level))}"
                       "</b></p>")
        out.append("<figure>" + svg_trend({
            "days": [visit["day"] for visit in visits],
            "labels": [visit["label"] for visit in visits],
            "values": [entry[keys[0]] for entry in entries],
            "lows": [entry[keys[1]] for entry in entries],
            "highs": [entry[keys[2]] for entry in entries],
            "fitted": fit.get("fitted"),
            "unit": "animals / km²" if by_density else "animals counted",
        }) + "</figure>")
        headers = ["Visit", "Date", "Flights", "Animals"]
        if by_density:
            headers += ["Ground looked at (ha)", "Per km² (95 %)"]
        rows = []
        for visit, entry in zip(visits, entries):
            row = [visit["visit"], visit["label"],
                   ", ".join(str(f) for f in visit["flights"]),
                   entry["animals"]]
            if by_density:
                row += [_num(visit.get("looked_at_ha"), 1),
                        _interval(entry["density_per_km2"],
                                  entry["density_low"],
                                  entry["density_high"], 1)]
            rows.append(row)
        raw = (4, 5) if by_density else ()
        out.append(_table(headers, rows, numeric=range(3, len(headers)),
                          raw=raw))
        changes = (trend.get("changes") or {}).get(species) or []
        if changes:
            out.append("<ul>" + "".join(
                f"<li>{_esc(survey_trend.change_sentence(change, labels))}"
                "</li>" for change in changes) + "</ul>")
    return "".join(out)


def section_reaction(report: dict) -> str:
    reaction = first(report.get("results") or {}, "reaction")
    if not reaction:
        return ""
    out = ["<h2>Reaction to the drone</h2>"]
    rows = []
    for name, summary in (reaction.get("summary") or {}).items():
        title = _species_title(name)
        out.append(f"<p><b>{_esc(title)}:</b> "
                   f"{_esc(summary.get('sentence', ''))}</p>")
        directions = summary.get("directions") or {}
        share = summary.get("moved_share")
        rows.append([
            title,
            summary.get("judged", 0),
            "–" if share is None
            else f"{summary.get('moved', 0)} ({share * 100:.0f} %)",
            directions.get("away from the line", 0),
            directions.get("towards the line", 0),
            _num(summary.get("median_displacement_m"), 1, "–"),
            _num(summary.get("median_speed_ms"), 2, "–")])
    out.append(_table(
        ["", "In view long enough", "Moved", "Away from the line",
         "Towards the line", "Went (m, median)", "Speed (m/s, median)"],
        rows, numeric=range(1, 7)))
    return "".join(out)


def section_position(report: dict) -> str:
    position = first(report.get("results") or {}, "position")
    if not position:
        return ""
    from . import position_uncertainty

    summary = position.get("summary") or {}
    out = ["<h2>How good the places are</h2>",
           f"<p>{_esc(position_uncertainty.sentence(summary))}</p>"]
    accuracy = position.get("accuracy") or {}
    names = (("gps_horizontal_m", "Camera position", "m"),
             ("gps_vertical_m", "Camera height", "m"),
             ("camera_angle_deg", "Camera angle", "°"),
             ("terrain_m", "Terrain model", "m"),
             ("pixel_px", "Pixel", "px"), ("sync_s", "Timing", "s"))
    known = [(label, f"{accuracy[key]:g} {unit}")
             for key, label, unit in names if key in accuracy]
    if known:
        out.append(_table(["Taken as known to (one sigma)", ""], known))
    return "".join(out)


def section_settings(report: dict) -> str:
    out = ["<h2>Settings</h2>"]
    results = report.get("results") or {}
    facts = []
    census = first(results, "census")
    distances = (census or {}).get("group_distances_m") or {}
    said = OrderedDict()
    for flight, of_flight in distances.items():
        if of_flight:
            said[flight] = ", ".join(f"{name} {metres:g} m" for name, metres
                                     in sorted(of_flight.items()))
    if len(set(said.values())) == 1:
        facts.append(("Group distance", next(iter(said.values()))))
    else:
        facts += [(f"Group distance ({flight})", text)
                  for flight, text in said.items()]
    grid = first(results, "grid")
    if grid:
        share = grid.get("min_seen")
        text = grid.get("grid_text", "")
        if share is not None:
            text += f"; a cell counts when {share * 100:.0f} % of it was seen"
        facts.append(("Grid", text))
    for entry in (results.get("found") or {}).get("distance") or []:
        document = entry["document"]
        who = f" ({entry['species']})" if entry["species"] else ""
        facts.append((
            f"Distance sampling{who}",
            f"truncated at {_num(document.get('truncation_m'), 1)} m, "
            f"model {document.get('best_model', '')}"))
    trend = first(results, "trend")
    if trend:
        gap = trend.get("gap_days") or 0
        facts.append(("Visits", "the flights of one day" if not gap
                      else f"flights up to {gap} day(s) apart"))
    reaction = first(results, "reaction")
    if reaction:
        minimum = (reaction.get("settings") or {}).get("min_displacement_m")
        if minimum is not None:
            facts.append(("Moved", f"went at least {minimum:g} m while in "
                                   "view"))
    if facts:
        out.append(_table(["Of the survey", ""], facts))
    for flight in report.get("flights") or []:
        settings = flight.get("settings") or {}
        if not settings:
            continue
        out.append(f"<h3>{_esc(flight.get('name'))}</h3>")
        out.append(_table(["", ""], list(settings.items())))
    listed = []
    odd = dict(out_of_step(report))
    for key, entries in (results.get("found") or {}).items():
        for entry in entries:
            who = f" ({entry['species']})" if entry["species"] else ""
            title = RESULTS[key][1] + who
            listed.append([title, entry["file"], entry["written"],
                           odd.get(title, "")])
    if listed:
        out.append("<h3>What the report was made from</h3>")
        headers = ["Result", "File", "Written"]
        if odd:
            headers.append("Not of this survey")
        else:
            listed = [row[:3] for row in listed]
        out.append(_table(headers, listed))
    missing = results.get("missing") or []
    if missing:
        named = ", ".join(missing)
        out.append(_note(f"Not run, and so not in this report: {named}."))
    return "".join(out)


def section_notes(report: dict) -> str:
    out = []
    found = (report.get("results") or {}).get("found") or {}
    for key, entries in found.items():
        for entry in entries:
            document = entry["document"]
            text = ""
            if isinstance(document, dict):
                text = document.get("notes") or document.get("note")
            who = f" ({entry['species']})" if entry["species"] else ""
            out.append(_details(RESULTS[key][1] + who, text or ""))
    body = "".join(out)
    return f"<h2>How the figures were made</h2>{body}" if body else ""


def headline(report: dict) -> List[Tuple[str, str]]:
    """The few figures a reader looks for first: ``(figure, what it is)``."""
    results = report.get("results") or {}
    flights = report.get("flights") or []
    cards = []
    census = first(results, "census")
    if census:
        totals = [row for row in _census_totals(census)
                  if row["species"] == census_core.ALL_SPECIES]
        if totals:
            where = ""
            if census.get("plot_ha") is not None:
                where = " in the survey area"
            days = flying_days(report)
            if days > 1:
                where += f", over {days} days of flying"
            cards.append((f"{totals[0]['inside']:,}",
                          f"animals counted{where}"))
            species = [row for row in _census_totals(census)
                       if row["inside"]
                       if row["species"] != census_core.ALL_SPECIES]
            if species:
                cards.append((str(len(species)), "species"))
    for entry in (results.get("found") or {}).get("distance") or []:
        document = entry["document"]
        if entry["species"] or document.get("density_per_km2") is None:
            continue
        ci = document.get("density_ci95") or [None, None]
        between = ""
        if ci[0] is not None:
            between = f" ({_num(ci[0], 1)}–{_num(ci[1], 1)})"
        cards.append((_num(document["density_per_km2"], 1),
                      f"animals per km² by distance sampling{between}"))
        break
    else:
        grid = first(results, "grid")
        density = ((grid or {}).get("statistics") or {}).get("all", {}).get(
            "density_per_km2") or {}
        if density.get("estimate") is not None:
            cards.append((_num(density["estimate"], 1),
                          "animals per km² over the grid"
                          f" ({_num(density.get('low'), 1)}–"
                          f"{_num(density.get('high'), 1)})"))
    if flights:
        cards.append((str(len(flights)),
                      "flight" if len(flights) == 1 else "flights"))
        dates = sorted({flight["date"] for flight in flights
                        if flight.get("date")})
        if dates:
            when = dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}"
            cards.append((when, "flown"))
    area = report.get("survey_area") or {}
    if area.get("adjusted_ha") is not None:
        cards.append((f"{_num(area['adjusted_ha'], 1)} ha", "survey area"))
    return cards


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

CSS = """
body { font: 14px/1.45 -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
  sans-serif; color: #222; max-width: 980px; margin: 24px auto;
  padding: 0 18px; }
h1 { font-size: 26px; margin: 0 0 2px 0; }
h2 { font-size: 19px; margin: 34px 0 8px 0; padding-bottom: 4px;
  border-bottom: 2px solid #d9d9d9; }
h3 { font-size: 15px; margin: 18px 0 6px 0; }
p.sub { color: #666; margin: 0 0 16px 0; }
table { border-collapse: collapse; margin: 6px 0 12px 0; }
th, td { border: 1px solid #cfcfcf; padding: 3px 9px; vertical-align: top; }
th { background: #f0f0f0; text-align: left; font-weight: 600; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
tr.total td { font-weight: 600; }
p.note { color: #666; font-size: 12.5px; margin: 4px 0 10px 0; }
p.warn { border-left: 4px solid #d6601c; background: #fdf3ec;
  padding: 7px 12px; margin: 12px 0 4px 0; }
.cards { display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0 6px 0; }
.card { border: 1px solid #d9d9d9; border-radius: 6px; padding: 8px 14px;
  min-width: 120px; }
.card b { display: block; font-size: 22px; }
.card span { color: #666; font-size: 12.5px; }
figure { margin: 10px 0 14px 0; }
figure svg, figure img { max-width: 100%; height: auto; }
figure.small { display: inline-block; width: 420px; max-width: 100%;
  margin: 6px 12px 10px 0; vertical-align: top; }
figcaption { color: #666; font-size: 12.5px; }
.charts { margin-top: 4px; }
svg { font: 11px sans-serif; }
svg .grid { stroke: #e4e4e4; stroke-width: 1; }
svg .axis { stroke: #555; stroke-width: 1; }
svg .tick { fill: #444; }
svg .end { text-anchor: end; }
svg .mid { text-anchor: middle; }
svg .fit { fill: none; stroke: #d6601c; stroke-width: 1.6;
  stroke-dasharray: 6 4; }
svg .bar { stroke: #1f77b4; stroke-width: 1.4; }
svg .dot { fill: #1f77b4; stroke: #17567f; }
svg.map { border: 1px solid #cfcfcf; }
svg .land { fill: #fbfbf8; }
svg .seen { fill: #e3ecf5; stroke: none; }
svg .plot { fill: none; stroke: #2a7f2a; stroke-width: 1.6; }
svg .excluded { fill: #f3d6d6; stroke: #b33; stroke-width: 1;
  stroke-dasharray: 4 3; }
svg .route { fill: none; stroke: #7a7a7a; stroke-width: 1; }
svg .animal { stroke: #fff; stroke-width: 0.8; }
svg .scale { stroke: #222; stroke-width: 3; }
svg .north { fill: #222; }
p.legend span { margin-right: 16px; white-space: nowrap; }
p.legend i { display: inline-block; width: 11px; height: 11px;
  margin-right: 5px; vertical-align: -1px; }
p.legend i.route { height: 0; border-top: 2px solid #7a7a7a;
  vertical-align: 4px; }
p.legend i.seen { background: #e3ecf5; border: 1px solid #c5d3e2; }
p.legend i.plot { border: 2px solid #2a7f2a; box-sizing: border-box; }
p.legend i.excluded { background: #f3d6d6; border: 1px dashed #b33;
  box-sizing: border-box; }
details { margin: 4px 0; }
summary { cursor: pointer; font-weight: 600; }
footer { margin-top: 40px; padding-top: 8px; border-top: 1px solid #d9d9d9;
  color: #777; font-size: 12px; }
@media print { body { margin: 0; max-width: none; }
  h2 { page-break-after: avoid; } table, figure { page-break-inside: avoid; }
  details > p { display: block; } }
"""

#: The parts of the page, in order: what each is called and what makes it.
SECTIONS = (("The survey", section_survey), ("Map", section_map),
            ("Counts", section_counts), ("Density", section_density),
            ("Where the animals were", section_where),
            ("Over time", section_trend),
            ("Reaction to the drone", section_reaction),
            ("How good the places are", section_position),
            ("Settings", section_settings),
            ("How the figures were made", section_notes))

#: What making a part ends in when a result file is not as this version of
#: the plugin writes it - an older one, or one changed by hand.
NOT_AS_WRITTEN = (KeyError, TypeError, ValueError, AttributeError,
                  IndexError)


def html(report: dict, problems: Optional[List[Tuple[str, str]]] = None
         ) -> str:
    """The report as one self-contained HTML page.

    *report* is ``{"title", "created", "plugin_version", "camera",
    "source", "crs", "species_filter", "survey_area", "flights": [{"name",
    "date", "time", "frames", "route_km", "animals", "weather",
    "settings"}], "map": {...}, "canvas_png", "results": collect(...)}``.

    A part whose result file cannot be read as this version writes it does
    not cost the page: it says so in its place, and ``(part, why)`` is
    added to *problems* where a list is given.
    """
    title = report.get("title") or "Survey report"
    created = report.get("created") or datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC")

    def failed(part: str, exc: Exception) -> None:
        if problems is not None:
            problems.append((part, f"{type(exc).__name__}: {exc}"))

    try:
        figures = headline(report)
    except NOT_AS_WRITTEN as exc:
        figures = []
        failed("The figures at the top", exc)
    cards = "".join(f"<div class='card'><b>{_esc(figure)}</b>"
                    f"<span>{_esc(what)}</span></div>"
                    for figure, what in figures)
    try:
        warned = cautions(report)
    except NOT_AS_WRITTEN as exc:
        warned = []
        failed("What the results were made from", exc)
    warnings = "".join(f"<p class='warn'>{_esc(text)}</p>"
                       for text in warned)
    parts = []
    for part, section in SECTIONS:
        try:
            parts.append(section(report))
        except NOT_AS_WRITTEN as exc:
            failed(part, exc)
            parts.append(f"<h2>{_esc(part)}</h2>" + _note(
                "This part could not be made: a result file it is read "
                "from is not as this version of the plugin writes it. Run "
                "its analysis again, and write the report anew."))
    body = "".join(parts)
    version = report.get("plugin_version")
    made_by = "BAMBI - QGIS Integration" + (f" {version}" if version else "")
    return (
        "<!DOCTYPE html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, "
        "initial-scale=1'>"
        f"<title>{_esc(title)}</title><style>{CSS}</style></head><body>"
        f"<h1>{_esc(title)}</h1>"
        f"<p class='sub'>{_esc(report.get('camera') or '')} camera &middot; "
        f"written {_esc(created)}</p>"
        f"<div class='cards'>{cards}</div>{warnings}{body}"
        f"<footer>Made with {_esc(made_by)} on {_esc(created)}. The report "
        "computes nothing itself: every figure is read from the result "
        "files listed under Settings.</footer></body></html>\n")


def write(report: dict, path: str,
          problems: Optional[List[Tuple[str, str]]] = None) -> str:
    """Write the page to *path*; returns *path*. See :func:`html` for
    *problems*."""
    page = html(report, problems)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(page)
    return path
