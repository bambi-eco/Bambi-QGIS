# -*- coding: utf-8 -*-
"""Trends across visits: the same ground, surveyed again and again.

One survey of an area is a count. Several, weeks or years apart, say what
is happening to the animals - if the visits are compared fairly. This module
does that for the flights of a survey.

**Visits.** A *visit* is one occasion on which the ground was surveyed: the
flights flown on the same day or - with a gap of *n* days - the flights
flown no more than *n* days after the one before (:func:`group_visits`). The
day is the local one the frames were stamped with. A night survey that runs
past midnight stays one visit: flights less than :data:`SAME_NIGHT_HOURS`
apart are never separated.

**What a visit counted** (:func:`analyse`). Per species: the animals, the
groups they formed, sex and age, and the density - the animals over the
ground the cameras looked at during the visit, summed over its flights, so
that two flights over the same ground on one day give the mean of the two
and two flights side by side one count of both. The interval around a
density takes a group of animals as one sighting (the gamma interval of Fay
and Feuer for a weighted sum of Poisson counts; for animals seen one by one
it is the exact Poisson interval): ten deer in one herd are one piece of
luck, not ten.

**The ground compared.** All the ground each visit saw, or only the ground
*every* visit saw (:data:`COMMON`) - the like-for-like comparison where the
visits did not cover the same part of the area.

**From visit to visit** (:func:`compare`): the ratio of two densities, an
exact interval for it, and how likely a difference at least as large is
between two counts of one and the same density (conditional binomial test;
clumped animals count for fewer sightings than there are animals).

**The trend** (:func:`log_linear`), from two visits on: a constant rate of
change fitted to the counts (Poisson regression with the ground looked at
as exposure). Its uncertainty is the largest of what counting alone
explains, what the groups add and - from three visits on - how far the
visits lie from the line.

A count is of the animals *seen*. Canopy, season, time of day and weather
change how many that are, so a trend in counts is a trend in the animals
only as far as the visits were flown alike - which is why every visit is
listed with its time of day and its weather.

numpy, scipy and shapely are imported inside the functions that need them.
Free of QGIS and Qt.
"""

import csv
import json
import math
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from . import census

#: Flights closer in time than this belong to one visit whatever the
#: calendar says: a night survey does not end at midnight.
SAME_NIGHT_HOURS = 6.0
MAX_GAP_DAYS = 3650

#: The ground compared.
SEEN = "seen"
COMMON = "common"
GROUNDS = (SEEN, COMMON)
GROUND_LABELS = {
    SEEN: "All the ground each visit saw",
    COMMON: "Only the ground every visit saw",
}

#: What a density is taken over: the ground looked at - or, where that is
#: not known for a flight, nothing: the counts are compared as they are.
BASIS_GROUND = "ground"
BASIS_COUNT = "count"

INCREASE = "increase"
DECREASE = "decrease"
NO_CHANGE = "no clear change"

LEVEL = 0.95
ALL = census.ALL_SPECIES

#: A rate per year is given from this span on; below it a year is further
#: than the visits can see.
YEAR_DAYS = 365.25

#: Leading columns of the table of visits; sex, age and weather follow.
VISIT_COLUMNS = [
    "visit", "date", "start", "end", "flights", "species", "animals",
    "looked_at_ha", "density_per_km2", "density_low", "density_high",
    "number", "number_low", "number_high", "groups", "mean_group_size",
    "largest_group", "change_pct", "change_p_value",
]


class TrendError(RuntimeError):
    """The visits cannot be compared as asked."""


# ---------------------------------------------------------------------------
# When a flight was flown
# ---------------------------------------------------------------------------

def _aware(moment: datetime) -> datetime:
    """A stamp without an offset is the flight log's clock: UTC."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def flight_span(target_folder: str, modality: str
                ) -> Optional[Tuple[datetime, datetime]]:
    """First and last capture time of a flight, each with the offset it was
    stamped with - so that its date is the local one. The other camera's
    frames are asked where this one's carry no time. ``None`` when neither
    does."""
    from .exporters import common

    for camera in (modality, "w" if modality == "t" else "t"):
        try:
            frames = common.load_frames(target_folder, camera)
        except (OSError, ValueError, KeyError):
            continue
        times = [common.capture_time(frame.get("epoch")) for frame in frames]
        times = [_aware(moment) for moment in times if moment is not None]
        if times:
            return min(times), max(times)
    return None


# ---------------------------------------------------------------------------
# Visits
# ---------------------------------------------------------------------------

def group_visits(flights: Sequence[dict], gap_days: int = 0) -> List[dict]:
    """The flights divided into visits, in order of time.

    :param flights: ``{"name", "start", "end"}`` - aware datetimes
    :param gap_days: a flight flown no more than this many days after the
        one before belongs to the same visit; 0 = the same day
    :return: ``{"visit": 1.., "flights": [names], "start", "end", "mid",
        "dates": [local dates], "label"}`` per visit
    :raises TrendError: for a flight without a date
    """
    try:
        gap = int(gap_days)
    except (TypeError, ValueError) as exc:
        raise TrendError(f"'{gap_days}' is not a number of days.") from exc
    if not 0 <= gap <= MAX_GAP_DAYS:
        raise TrendError(
            f"The gap between visits is 0 to {MAX_GAP_DAYS} days.")
    undated = [str(flight["name"]) for flight in flights
               if flight.get("start") is None]
    if undated:
        raise TrendError(
            "It is not known when " + ", ".join(undated) + " was flown: "
            "the frames carry no capture time. Extract the frames again "
            "with the flight log, or leave the flight out of the survey.")

    ordered = sorted(flights, key=lambda flight: (
        _aware(flight["start"]), str(flight["name"])))
    visits: List[dict] = []
    for flight in ordered:
        start = _aware(flight["start"])
        end = _aware(flight.get("end") or flight["start"])
        current = visits[-1] if visits else None
        if current is not None:
            days = (start.date() - current["_day"]).days
            hours = (start - current["end"]).total_seconds() / 3600.0
            if days <= gap or hours < SAME_NIGHT_HOURS:
                current["flights"].append(flight["name"])
                current["end"] = max(current["end"], end)
                current["_day"] = max(current["_day"], start.date())
                current["dates"].add(start.date())
                continue
        visits.append({"flights": [flight["name"]], "start": start,
                       "end": end, "_day": start.date(),
                       "dates": {start.date()}})
    for number, visit in enumerate(visits, start=1):
        del visit["_day"]
        dates = sorted(visit["dates"])
        visit["dates"] = dates
        visit["visit"] = number
        visit["mid"] = visit["start"] + (visit["end"] - visit["start"]) / 2
        visit["label"] = dates[0].isoformat() if len(dates) == 1 else (
            f"{dates[0].isoformat()} to {dates[-1].isoformat()}")
    return visits


# ---------------------------------------------------------------------------
# Counts with their uncertainty
# ---------------------------------------------------------------------------

def count_interval(total: float, variance: float, largest: float = 1.0,
                   level: float = LEVEL) -> Tuple[float, float]:
    """The interval around a count that came in groups.

    The animals of a visit are a sum of group sizes. Groups turn up by
    chance - a Poisson number of them - so the variance of the sum is the
    sum of the squared sizes (*variance*), and the gamma interval of Fay
    and Feuer (1997) covers what the count could as well have been. For
    animals seen one by one (*variance* = *total*) it is the exact Poisson
    interval.

    :param largest: the largest group; it sets how much one more sighting
        could have added
    """
    from scipy import stats

    alpha = 1.0 - level
    weight = max(1.0, float(largest or 1.0))
    if total <= 0 or variance <= 0:
        low = 0.0
    else:
        low = float(stats.gamma.ppf(alpha / 2.0, a=total ** 2 / variance,
                                    scale=variance / total))
    more, spread = total + weight, variance + weight ** 2
    high = float(stats.gamma.ppf(1.0 - alpha / 2.0, a=more ** 2 / spread,
                                 scale=spread / more))
    return low, high


def _binomial_p(successes: int, trials: int, share: float) -> float:
    """Two-sided exact binomial test: the chance of an outcome no more
    likely than the one seen."""
    import numpy as np
    from scipy import stats

    chances = stats.binom.pmf(np.arange(trials + 1), trials, share)
    seen = chances[successes]
    return float(min(1.0, chances[chances <= seen * (1.0 + 1e-7)].sum()))


def compare(count_a: float, variance_a: float, exposure_a: float,
            count_b: float, variance_b: float, exposure_b: float,
            level: float = LEVEL) -> dict:
    """Visit *b* against visit *a*: has the density changed?

    Given how many animals the two visits counted together, the share that
    fell to *b* is binomial if the density was the same - with *b*'s share
    of the ground looked at as its chance. That is exact for animals seen
    one by one. Clumped animals are worth fewer sightings than there are
    animals: the counts are divided by the dispersion - the mean squared
    group size over the mean group size - and rounded down.

    :return: ``{"ratio", "ratio_low", "ratio_high", "change_pct",
        "p_value", "verdict", "dispersion"}``; ``ratio`` and ``change_pct``
        are ``None`` where *a* counted nothing, ``ratio_high`` where the
        ratio has no upper limit, everything where neither counted anything
    """
    from scipy import stats

    out = {"ratio": None, "ratio_low": None, "ratio_high": None,
           "change_pct": None, "p_value": None, "verdict": NO_CHANGE,
           "dispersion": None}
    total = count_a + count_b
    if total <= 0 or exposure_a <= 0 or exposure_b <= 0:
        return out
    dispersion = max(1.0, (variance_a + variance_b) / total)

    def worth(count: float) -> int:
        if count <= 0:
            return 0
        if dispersion <= 1.0:
            return int(round(count))
        return max(1, int(count / dispersion))

    a, b = worth(count_a), worth(count_b)
    alpha = 1.0 - level
    share = exposure_b / (exposure_a + exposure_b)
    p_value = _binomial_p(b, a + b, share)
    low = 0.0 if b == 0 else float(stats.beta.ppf(alpha / 2.0, b, a + 1))
    high = 1.0 if a == 0 else float(
        stats.beta.ppf(1.0 - alpha / 2.0, b + 1, a))
    scale = exposure_a / exposure_b
    density_a, density_b = count_a / exposure_a, count_b / exposure_b
    out.update({
        "ratio_low": low / (1.0 - low) * scale if low < 1.0 else None,
        "ratio_high": high / (1.0 - high) * scale if high < 1.0 else None,
        "p_value": p_value,
        "dispersion": dispersion,
    })
    if count_a > 0:
        out["ratio"] = density_b / density_a
        out["change_pct"] = (out["ratio"] - 1.0) * 100.0
    if p_value < alpha and density_b != density_a:
        out["verdict"] = INCREASE if density_b > density_a else DECREASE
    return out


def log_linear(days: Sequence[float], counts: Sequence[float],
               exposures: Sequence[float], variances: Sequence[float],
               level: float = LEVEL) -> dict:
    """A constant rate of change through the visits.

    Poisson regression of the counts on time, with the ground looked at as
    exposure: ``count = exposure * exp(a + b * day)``. The standard error of
    *b* is scaled by the dispersion - the largest of 1 (counting alone),
    what the groups add (the sum of squared group sizes over the animals),
    and from three visits on the scatter of the visits about the line
    (Pearson's chi-square over its degrees of freedom; the interval is then
    Student's).

    :return: ``{"visits", "span_days", "slope_per_day", "slope_se",
        "dispersion", "dispersion_from", "change_over_span_pct" (+ ``_low``,
        ``_high``), "change_per_year_pct" (+ ``_low``, ``_high``; ``None``
        below a year), "p_value", "verdict", "fitted" (per unit of exposure,
        per visit), "reason"}`` - with a *reason* nothing was fitted
    """
    import numpy as np
    from scipy import stats

    t = np.asarray(days, dtype=float)
    y = np.asarray(counts, dtype=float)
    exposure = np.asarray(exposures, dtype=float)
    n = int(len(y))
    out = {
        "visits": n,
        "span_days": float(t.max() - t.min()) if n else 0.0,
        "slope_per_day": None, "slope_se": None, "dispersion": None,
        "dispersion_from": None, "change_over_span_pct": None,
        "change_over_span_pct_low": None, "change_over_span_pct_high": None,
        "change_per_year_pct": None, "change_per_year_pct_low": None,
        "change_per_year_pct_high": None, "p_value": None,
        "verdict": NO_CHANGE, "fitted": None, "reason": None,
    }
    if n < 2 or out["span_days"] <= 0:
        out["reason"] = "A trend needs at least two visits."
        return out
    if y.sum() <= 0:
        out["reason"] = "No animals were counted on any visit."
        return out
    with_animals = t[y > 0]
    if with_animals.min() == with_animals.max():
        end = None
        if with_animals[0] == t.max():
            end = "last"
        elif with_animals[0] == t.min():
            end = "first"
        if end is not None:
            out["reason"] = (
                f"All the animals were counted on the {end} visit. A "
                "constant rate of change cannot be fitted to a series that "
                f"{'starts' if end == 'last' else 'ends'} with nothing; "
                "see the change from visit to visit.")
            return out

    centred = t - t.mean()
    design = np.column_stack([np.ones(n), centred])
    offset = np.log(exposure)
    beta = np.array([math.log(y.sum() / exposure.sum()), 0.0])
    settled = False
    for _ in range(200):
        eta = np.clip(design @ beta + offset, -700.0, 700.0)
        mu = np.exp(eta)
        working = (eta - offset) + (y - mu) / mu
        weighted = design.T * mu
        try:
            new = np.linalg.solve(weighted @ design, weighted @ working)
        except np.linalg.LinAlgError:
            break
        step = float(np.max(np.abs(new - beta)))
        beta = new
        if not np.all(np.isfinite(beta)):
            break
        if step < 1e-10:
            settled = True
            break
    if not settled:
        out["reason"] = ("A constant rate of change could not be fitted to "
                         "these counts.")
        return out

    mu = np.exp(design @ beta + offset)
    covariance = np.linalg.inv((design.T * mu) @ design)
    slope = float(beta[1])
    from_groups = max(1.0, float(sum(variances)) / float(y.sum()))
    from_visits = None
    if n > 2:
        from_visits = float(np.sum((y - mu) ** 2 / mu)) / (n - 2)
    alpha = 1.0 - level
    if from_visits is not None and from_visits > from_groups:
        dispersion, source = from_visits, "visits"
        quantile = float(stats.t.ppf(1.0 - alpha / 2.0, n - 2))
    else:
        dispersion = from_groups
        source = "groups" if from_groups > 1.0 else "counting"
        quantile = float(stats.norm.ppf(1.0 - alpha / 2.0))
    error = math.sqrt(dispersion * float(covariance[1, 1]))
    score = abs(slope) / error if error > 0 else float("inf")
    if source == "visits":
        p_value = float(2.0 * stats.t.sf(score, n - 2))
    else:
        p_value = float(2.0 * stats.norm.sf(score))
    low, high = slope - quantile * error, slope + quantile * error

    def percent(rate: float, over_days: float) -> float:
        return (math.exp(min(700.0, rate * over_days)) - 1.0) * 100.0

    span = out["span_days"]
    out.update({
        "slope_per_day": slope, "slope_se": error,
        "dispersion": dispersion, "dispersion_from": source,
        "change_over_span_pct": percent(slope, span),
        "change_over_span_pct_low": percent(low, span),
        "change_over_span_pct_high": percent(high, span),
        "p_value": p_value,
        "fitted": [float(value) for value in np.exp(beta[0] + slope * centred)],
    })
    if span >= YEAR_DAYS:
        out.update({
            "change_per_year_pct": percent(slope, YEAR_DAYS),
            "change_per_year_pct_low": percent(low, YEAR_DAYS),
            "change_per_year_pct_high": percent(high, YEAR_DAYS),
        })
    if low > 0:
        out["verdict"] = INCREASE
    elif high < 0:
        out["verdict"] = DECREASE
    return out


# ---------------------------------------------------------------------------
# The survey, visit by visit
# ---------------------------------------------------------------------------

def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def _number(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _mean(values) -> Optional[float]:
    known = [number for number in map(_number, values) if number is not None]
    return sum(known) / len(known) if known else None


def _common_ground(visits: Sequence[dict], seen: Dict[str, object]):
    """What every visit saw: the ground of each visit - its flights' taken
    together - cut with that of all the others."""
    import shapely

    common = None
    for visit in visits:
        of_visit = shapely.unary_union([seen[name] for name in visit["flights"]
                                        if not seen[name].is_empty])
        common = of_visit if common is None else common.intersection(of_visit)
    if common is None or common.is_empty or common.area <= 0:
        raise TrendError(
            "The visits have no ground in common: no part of the survey "
            "was seen on every one of them. Compare all the ground each "
            "visit saw instead.")
    return common


def _within(records: Sequence[dict], ground) -> List[dict]:
    """The records that lie on *ground*; one without a place cannot be said
    to."""
    import shapely

    placed = [record for record in records
              if record.get("x") is not None and record.get("y") is not None]
    if not placed:
        return []
    inside = shapely.contains_xy(
        ground, [float(record["x"]) for record in placed],
        [float(record["y"]) for record in placed])
    return [record for record, kept in zip(placed, inside) if kept]


def analyse(flights: Sequence[dict], area=None, gap_days: int = 0,
            ground: str = SEEN, species_order: Optional[Sequence[str]] = None,
            level: float = LEVEL) -> dict:
    """The survey visit by visit, and what changed between the visits.

    :param flights: ``{"name", "start", "end", "records", "seen",
        "weather"}`` per flight - *records* as the census counts them
        (:func:`core.census.tally`, with ``group``), *seen* the ground its
        camera saw inside the survey area as a geometry (``None`` = not
        known), *weather* what is known of the conditions (numbers by name)
    :param area: the survey area, or ``None``
    :param ground: :data:`SEEN` or :data:`COMMON`
    :return: the document :func:`write_json` stores - ``visits``, per
        species the ``series`` (one entry per visit), the ``changes`` (from
        one visit to the next, and from the first to the last) and the
        ``trends``
    :raises TrendError: for fewer than two visits, a flight without a date,
        or no ground in common
    """
    if ground not in GROUNDS:
        raise TrendError(f"'{ground}' is not a ground to compare.")
    flights = list(flights)
    if not flights:
        raise TrendError("The survey has no flights.")
    visits = group_visits(flights, gap_days)
    if len(visits) < 2:
        only = visits[0]
        raise TrendError(
            f"The {len(flights)} flight(s) of the survey were all flown on "
            f"{only['label']}: that is one visit, and a comparison needs "
            "two. Tick the flights of the other surveys of this area as "
            "well" + (" - or allow fewer days between two visits."
                      if int(gap_days) > 0 else "."))
    by_name = {flight["name"]: flight for flight in flights}
    seen = {name: flight.get("seen") for name, flight in by_name.items()}
    known = all(geometry is not None for geometry in seen.values())
    common = None
    if ground == COMMON:
        if not known:
            unknown = sorted(str(name) for name, geometry in seen.items()
                             if geometry is None)
            raise TrendError(
                "The ground " + ", ".join(unknown) + " saw is not known: "
                "run 'Calculate Field of View' for this camera, or draw a "
                "survey area - which then counts as seen.")
        common = _common_ground(visits, seen)
    basis = BASIS_GROUND if known else BASIS_COUNT
    plot_m2 = None if area is None else area.adjusted_m2

    species_seen: List[str] = list(species_order or [])
    totals: List[Dict[str, dict]] = []
    looked_at: List[Optional[float]] = []
    conditions: List[dict] = []
    for visit in visits:
        of_visit: Dict[str, dict] = OrderedDict()
        ground_m2 = 0.0
        for name in visit["flights"]:
            flight = by_name[name]
            records = flight.get("records") or []
            if common is not None:
                records = _within(records, common)
            for species, counts in census.tally(records, area).items():
                census.merge(of_visit.setdefault(species, census.empty()),
                             counts)
                if species not in species_seen:
                    species_seen.append(species)
            if known:
                geometry = seen[name]
                if common is not None:
                    geometry = geometry.intersection(common)
                ground_m2 += float(geometry.area)
        everything = census.empty()
        for counts in of_visit.values():
            census.merge(everything, counts)
        of_visit[ALL] = everything
        totals.append(of_visit)
        looked_at.append(ground_m2 if known else None)
        weather: Dict[str, Optional[float]] = OrderedDict()
        for name in visit["flights"]:
            for key in (by_name[name].get("weather") or {}):
                weather.setdefault(key, None)
        for key in weather:
            weather[key] = _mean((by_name[name].get("weather") or {}).get(key)
                                 for name in visit["flights"])
        conditions.append(weather)

    first = visits[0]["mid"]
    days = [(visit["mid"] - first).total_seconds() / 86400.0
            for visit in visits]
    if basis == BASIS_GROUND:
        if any(not value for value in looked_at):
            empty = [visit["label"] for visit, value in zip(visits, looked_at)
                     if not value]
            raise TrendError(
                "On " + ", ".join(empty) + " the cameras saw none of the "
                "ground that is compared. Leave that visit out, or compare "
                "all the ground each visit saw.")
        exposures = [value / 1e6 for value in looked_at]       # km²
    else:
        exposures = [1.0] * len(visits)

    species_seen = [name for name in species_seen
                    if any(name in of_visit for of_visit in totals)]
    names = species_seen + [ALL]
    series: Dict[str, List[dict]] = OrderedDict()
    changes: Dict[str, List[dict]] = OrderedDict()
    trends: Dict[str, dict] = OrderedDict()
    for name in names:
        entries = []
        animals, variances = [], []
        for index, visit in enumerate(visits):
            counts = totals[index].get(name) or census.empty()
            sizes = list(counts.get("group_sizes") or [])
            sizes += [1] * int(counts.get("ungrouped") or 0)
            total = int(counts["inside"])
            variance = float(sum(size * size for size in sizes))
            largest = max(sizes) if sizes else 1
            low, high = count_interval(total, variance, largest, level)
            entry = {
                "visit": visit["visit"],
                "animals": total,
                "counted": counts["counted"],
                "outside": counts["outside"],
                "unplaced": counts["unplaced"],
                "approved": counts["approved"],
                "groups": len(sizes),
                "mean_group_size": total / len(sizes) if sizes else None,
                "largest_group": max(sizes) if sizes else None,
                "animals_low": low, "animals_high": high,
                "density_per_km2": None, "density_low": None,
                "density_high": None, "number": None, "number_low": None,
                "number_high": None,
                "sex": dict(counts["sex"]), "age": dict(counts["age"]),
            }
            if basis == BASIS_GROUND:
                km2 = exposures[index]
                entry.update({"density_per_km2": total / km2,
                              "density_low": low / km2,
                              "density_high": high / km2})
                if plot_m2 and common is None:
                    scale = plot_m2 / 1e6 / km2
                    entry.update({"number": total * scale,
                                  "number_low": low * scale,
                                  "number_high": high * scale})
            entries.append(entry)
            animals.append(total)
            variances.append(variance)
        series[name] = entries

        pairs = [(index, index + 1) for index in range(len(visits) - 1)]
        if len(visits) > 2:
            pairs.append((0, len(visits) - 1))
        of_species = []
        for a, b in pairs:
            found = compare(animals[a], variances[a], exposures[a],
                            animals[b], variances[b], exposures[b], level)
            found.update({"from": visits[a]["visit"], "to": visits[b]["visit"],
                          "days": days[b] - days[a]})
            of_species.append(found)
        changes[name] = of_species
        trends[name] = log_linear(days, animals, exposures, variances, level)

    described = []
    for index, visit in enumerate(visits):
        described.append({
            "visit": visit["visit"],
            "label": visit["label"],
            "start": _iso(visit["start"]),
            "end": _iso(visit["end"]),
            "mid": _iso(visit["mid"]),
            "day": days[index],
            "time_of_day": (f"{visit['start']:%H:%M}-{visit['end']:%H:%M}"),
            "flights": list(visit["flights"]),
            "looked_at_ha": None if looked_at[index] is None
            else looked_at[index] / 1e4,
            "weather": conditions[index],
        })
    rows = [entry for entries in series.values() for entry in entries]
    return {
        "gap_days": int(gap_days),
        "ground": ground,
        "basis": basis,
        "level": level,
        "plot_ha": None if plot_m2 is None else plot_m2 / 1e4,
        "common_ha": None if common is None else float(common.area) / 1e4,
        "visits": described,
        "species": names,
        "series": series,
        "changes": changes,
        "trends": trends,
        "sex_labels": _labels(rows, "sex"),
        "age_labels": _labels(rows, "age"),
    }


def _labels(rows: Sequence[dict], key: str) -> List[str]:
    found = {label for row in rows for label in row[key]}
    labels = sorted(found - {census.UNKNOWN})
    if any(census.UNKNOWN in row[key] for row in rows):
        labels.append(census.UNKNOWN)
    return labels


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------

def flat_rows(document: dict) -> List[OrderedDict]:
    """One row per visit and species, for a table: the columns of
    :data:`VISIT_COLUMNS`, then sex and age, then the weather."""
    visits = {visit["visit"]: visit for visit in document["visits"]}
    weather: List[str] = []
    for visit in document["visits"]:
        for key in visit.get("weather") or {}:
            if key not in weather:
                weather.append(key)
    rows = []
    for species in document["species"]:
        previous = {change["to"]: change
                    for change in document["changes"].get(species, [])
                    if change["to"] - change["from"] == 1}
        for entry in document["series"][species]:
            visit = visits[entry["visit"]]
            change = previous.get(entry["visit"]) or {}
            values = dict(entry, date=visit["label"], start=visit["start"],
                          end=visit["end"],
                          flights="; ".join(str(name)
                                            for name in visit["flights"]),
                          species=species,
                          looked_at_ha=visit["looked_at_ha"],
                          change_pct=change.get("change_pct"),
                          change_p_value=change.get("p_value"))
            row = OrderedDict((key, values.get(key)) for key in VISIT_COLUMNS)
            row["time_of_day"] = visit["time_of_day"]
            for label in document["sex_labels"]:
                row[f"sex_{label}"] = entry["sex"].get(label, 0)
            for label in document["age_labels"]:
                row[f"age_{label}"] = entry["age"].get(label, 0)
            for key in weather:
                row[key] = (visit.get("weather") or {}).get(key)
            rows.append(row)
    return rows


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 4)
    return value


def write_table(rows: Sequence[dict], path: str) -> str:
    header = list(rows[0].keys()) if rows else list(VISIT_COLUMNS)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([_cell(row.get(key)) for key in header])
    return path


def write_json(document: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
    return path


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# In words
# ---------------------------------------------------------------------------

def signed(percent: Optional[float], nothing: str = "no change") -> str:
    """A change in percent as it is read: ``+12 %``, ``-40 %`` - and
    *nothing* for a change of nought."""
    if percent is None:
        return "n/a"
    if abs(percent) < 0.05:
        return nothing
    if percent > 9999:
        return "more than +9,999 %"
    if abs(percent) >= 1000:
        return f"{percent:+,.0f} %"
    return f"{percent:+.0f} %" if abs(percent) >= 10 else f"{percent:+.1f} %"


def span_text(days: float) -> str:
    """A length of time as it is said: days, weeks, months, years."""
    days = abs(float(days))
    if days < 1.5:
        return "1 day" if days >= 0.5 else "less than a day"
    if days < 21:
        return f"{days:.0f} days"
    if days < 90:
        return f"{days / 7.0:.0f} weeks"
    if days < 1.5 * YEAR_DAYS:
        return f"{days / (YEAR_DAYS / 12.0):.0f} months"
    return f"{days / YEAR_DAYS:.1f} years"


def trend_sentence(trend: dict, level: float = LEVEL) -> str:
    """What the fitted trend of one species says, in a sentence."""
    if trend.get("reason"):
        return trend["reason"]
    share = int(round(level * 100))
    over = span_text(trend["span_days"])
    text = (f"{signed(trend['change_over_span_pct'])} over {over} "
            f"({share} % interval "
            f"{signed(trend['change_over_span_pct_low'], '0 %')} to "
            f"{signed(trend['change_over_span_pct_high'], '0 %')})")
    if trend.get("change_per_year_pct") is not None:
        text += f", {signed(trend['change_per_year_pct'])} per year"
    if trend["verdict"] == INCREASE:
        return text + ": an increase."
    if trend["verdict"] == DECREASE:
        return text + ": a decrease."
    return text + ": no clear trend - the interval includes no change."


def change_sentence(change: dict, labels: Dict[int, str]) -> str:
    """One change from visit to visit, in a sentence."""
    head = f"{labels[change['from']]} to {labels[change['to']]}: "
    if change.get("p_value") is None:
        return head + "nothing counted on either visit."
    if change.get("ratio") is None:
        body = "from none to some"
    elif change["ratio"] == 0:
        body = "from some to none"
    else:
        body = signed(change["change_pct"])
    chance = change["p_value"]
    odds = f"p = {chance:.3f}" if chance >= 0.001 else "p < 0.001"
    if change["verdict"] == NO_CHANGE:
        return head + f"{body} ({odds}): within what chance explains."
    article = "an" if change["verdict"] == INCREASE else "a"
    return head + f"{body} ({odds}): {article} {change['verdict']}."


def describe(document: dict) -> List[str]:
    """The result in a few lines, for the log."""
    visits = document["visits"]
    lines = [f"{len(visits)} visits: " + ", ".join(
        visit["label"] + (f" ({len(visit['flights'])} flights)"
                          if len(visit["flights"]) > 1 else "")
        for visit in visits)]
    by_density = document["basis"] == BASIS_GROUND
    for species in document["species"]:
        entries = document["series"][species]
        if by_density:
            run = " -> ".join(f"{entry['density_per_km2']:.2f}"
                              for entry in entries) + " /km²"
        else:
            run = " -> ".join(str(entry["animals"]) for entry in entries)
        said = trend_sentence(document["trends"][species],
                              document.get("level", LEVEL))
        lines.append(f"{species}: {run}. {said}")
    return lines
