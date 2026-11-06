# -*- coding: utf-8 -*-
"""Census summary: counts and densities per species, flight and survey.

The plain census figures a survey report starts from - how many of each
species were counted, how many of them inside the survey plot, how many per
km², and how they split by sex and age. No detection-function model is
involved; these are minimum counts, and the densities assume everything
inside the surveyed ground was seen.

Two area bases are reported, because which one is right depends on the
survey:

* **plot** - the survey plot minus its exclusion zones. The right basis when
  the flights together cover the plot completely (one census). Across several
  flights the total is ``sum(counts) / plot area``: adjacent flights add up to
  one census of the plot.
* **covered** - the ground the flight's frames actually saw inside the plot
  (the union of its field-of-view footprints, clipped to the plot). It does
  not assume full coverage, and across flights the total is
  ``sum(counts) / sum(covered areas)``, which is right for repeat flights over
  the same plot as well as for flights over neighbouring ground.

A record that cannot be placed (no geo-referenced position) is counted, but
cannot be tested against the plot: with a survey area it is reported as
*unplaced* and not counted inside.
"""

import csv
import json
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence

ALL_SPECIES = "All species"
ALL_FLIGHTS = "All flights"
UNKNOWN = "unknown"

#: Fixed leading columns of the CSV; the sex/age breakdown follows.
COLUMNS = [
    "flight", "species", "counted", "inside", "outside", "unplaced",
    "approved", "plot_ha", "covered_ha", "per_km2_plot", "per_km2_covered",
]


def _label(value) -> str:
    text = "" if value is None else str(value).strip()
    return text.lower() if text else UNKNOWN


def tally(records: Sequence[dict], area=None) -> Dict[str, dict]:
    """Per-species counts of one flight: ``{species: counts}``.

    *records* are ``{species, sex, age, x, y, approved}``; ``x``/``y`` may be
    ``None`` for an animal that was never geo-referenced.
    """
    out: Dict[str, dict] = OrderedDict()
    for record in records:
        species = record.get("species") or UNKNOWN
        counts = out.setdefault(species, {
            "counted": 0, "inside": 0, "outside": 0, "unplaced": 0,
            "approved": 0, "sex": {}, "age": {}})
        counts["counted"] += 1
        x, y = record.get("x"), record.get("y")
        if area is None:
            inside = True
            if x is None or y is None:
                counts["unplaced"] += 1
        elif x is None or y is None:
            inside = False
            counts["unplaced"] += 1
        else:
            inside = area.contains(float(x), float(y))
            if not inside:
                counts["outside"] += 1
        if not inside:
            continue
        counts["inside"] += 1
        if record.get("approved"):
            counts["approved"] += 1
        for key in ("sex", "age"):
            label = _label(record.get(key))
            counts[key][label] = counts[key].get(label, 0) + 1
    return out


def _merge(target: dict, source: dict) -> dict:
    for key in ("counted", "inside", "outside", "unplaced", "approved"):
        target[key] = target.get(key, 0) + source.get(key, 0)
    for key in ("sex", "age"):
        bucket = target.setdefault(key, {})
        for label, n in source.get(key, {}).items():
            bucket[label] = bucket.get(label, 0) + n
    return target


def _empty() -> dict:
    return {"counted": 0, "inside": 0, "outside": 0, "unplaced": 0,
            "approved": 0, "sex": {}, "age": {}}


def _density(count: int, area_m2: Optional[float]) -> Optional[float]:
    if not area_m2:
        return None
    return count / (area_m2 / 1e6)


def build(flights: Sequence[dict], area=None,
          species_order: Optional[Sequence[str]] = None) -> dict:
    """The census of a survey.

    :param flights: ``{name, records, covered_m2}`` per flight; *covered_m2*
        is the ground its frames saw inside the area (``None`` when unknown).
    :param area: the survey area (``None`` = no plot: densities on the
        covered basis only).
    :param species_order: display order of the species (others follow).
    :return: ``{"rows": [...], "sex_labels": [...], "age_labels": [...],
        "plot_ha", "flights": [...]}`` - one row per flight and species, a
        per-flight "All species" row, and "All flights" totals.
    """
    plot_m2 = None if area is None else area.adjusted_m2
    per_flight = []
    species_seen: List[str] = list(species_order or [])
    for flight in flights:
        counts = tally(flight.get("records") or [], area)
        for name in counts:
            if name not in species_seen:
                species_seen.append(name)
        per_flight.append((flight["name"], counts, flight.get("covered_m2")))

    def row(flight_name, species, counts, covered_m2, plot_area_m2):
        return {
            "flight": flight_name,
            "species": species,
            "counted": counts["counted"],
            "inside": counts["inside"],
            "outside": counts["outside"],
            "unplaced": counts["unplaced"],
            "approved": counts["approved"],
            "plot_ha": None if plot_area_m2 is None else plot_area_m2 / 1e4,
            "covered_ha": None if covered_m2 is None else covered_m2 / 1e4,
            "per_km2_plot": _density(counts["inside"], plot_area_m2),
            "per_km2_covered": _density(counts["inside"], covered_m2),
            "sex": dict(counts["sex"]),
            "age": dict(counts["age"]),
        }

    rows = []
    totals: Dict[str, dict] = OrderedDict()
    covered_values = [c for _n, _c, c in per_flight]
    total_covered = (sum(covered_values)
                     if covered_values and all(c is not None
                                               for c in covered_values)
                     else None)
    for flight_name, counts, covered_m2 in per_flight:
        everything = _empty()
        for species in species_seen:
            if species not in counts:
                continue
            rows.append(row(flight_name, species, counts[species],
                            covered_m2, plot_m2))
            _merge(everything, counts[species])
            _merge(totals.setdefault(species, _empty()), counts[species])
        rows.append(row(flight_name, ALL_SPECIES, everything, covered_m2,
                        plot_m2))
    if len(per_flight) > 1:
        grand = _empty()
        for species in species_seen:
            if species in totals:
                rows.append(row(ALL_FLIGHTS, species, totals[species],
                                total_covered, plot_m2))
                _merge(grand, totals[species])
        rows.append(row(ALL_FLIGHTS, ALL_SPECIES, grand, total_covered,
                        plot_m2))

    sex_labels = _labels(rows, "sex")
    age_labels = _labels(rows, "age")
    return {
        "rows": rows,
        "sex_labels": sex_labels,
        "age_labels": age_labels,
        "plot_ha": None if plot_m2 is None else plot_m2 / 1e4,
        "flights": [name for name, _c, _a in per_flight],
    }


def _labels(rows: Iterable[dict], key: str) -> List[str]:
    labels = sorted({label for r in rows for label in r[key]} - {UNKNOWN})
    if any(UNKNOWN in r[key] for r in rows):
        labels.append(UNKNOWN)
    return labels


def flat_rows(census: dict) -> List[OrderedDict]:
    """Rows with the sex/age breakdown spread into ``sex_*``/``age_*``."""
    out = []
    for r in census["rows"]:
        flat = OrderedDict((key, r[key]) for key in COLUMNS)
        for label in census["sex_labels"]:
            flat[f"sex_{label}"] = r["sex"].get(label, 0)
        for label in census["age_labels"]:
            flat[f"age_{label}"] = r["age"].get(label, 0)
        out.append(flat)
    return out


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 4)
    return value


def write_csv(census: dict, path: str) -> str:
    rows = flat_rows(census)
    header = list(rows[0].keys()) if rows else list(COLUMNS)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in rows:
            writer.writerow([_cell(r[key]) for key in header])
    return path


def write_json(census: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(census, fh, indent=2)
    return path


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
