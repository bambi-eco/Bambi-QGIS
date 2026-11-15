# -*- coding: utf-8 -*-
"""Life stage from body size, per flight.

Implements §3.3 (Eq. 4) of *When One Modality Is Not Enough*. Appearance
cannot separate a juvenile from an adult female at survey resolution - that is
exactly why the sex classifier's second class is ``female_juvenile`` - so life
stage is read from size instead.

An individual is called a juvenile only when **both** hold:

* it sits far below the median of its cohort, by a robust
  (median-absolute-deviation) z-score;
* a size gap wider than a multiple of the cohort's interquartile range
  separates it from the rest of the herd - directly above it, or above
  another animal that is as far below the median as it is.

The second condition is what stops the rule firing on the merely smallest adult
of a herd - in a group of similar animals someone is always smallest, and that
is not evidence of anything. It used to ask for the gap *directly above* the
individual, which fails for twins: of two fawns of a size, the smaller one is
a hair below its sibling and was called an adult, and only the larger one - the
one next to the gap - a juvenile.

**Comparison is strictly within one flight.** Box tightness varies between
recordings, and the paper's own data shows why: the ``B`` juvenile's box lands
inside ``A2``'s female range despite being the smallest animal in its own
flight. There is deliberately no global threshold to configure.

**... and within one species** (:func:`cohorts`). A roe deer is half the size
of a red deer at any age: measured against a herd of red deer an adult roe
deer is exactly the low outlier the rule looks for. The flight's animals used
to be one cohort whatever they were.

Pure arithmetic - no torch, no models, no embeddings. The size cue works on any
flight that has been tracked.
"""

from typing import Dict, Iterable, List, NamedTuple, Sequence

#: Scale factor turning a median absolute deviation into a standard-deviation
#: equivalent for normally distributed data, so the threshold reads like a
#: z-score.
MAD_SCALE = 0.6745

ADULT = "adult"
JUVENILE = "juvenile"

#: Where the areas came from. Orthorectified areas are metric and therefore
#: the better cue; perspective pixels still work within one flight.
AREA_GEO = "orthorectified"
AREA_PIXEL = "perspective"


class Config(NamedTuple):
    """Thresholds for the juvenile test."""

    #: How far below the flight median an individual must sit.
    z_threshold: float = -2.0
    #: Multiple of the interquartile range the size gap must exceed.
    iqr_factor: float = 0.5
    #: Fewest individuals for the test to run at all. A robust z-score over
    #: three animals is not robust.
    min_individuals: int = 4


class Assessment(NamedTuple):
    """One individual's size, and what it implies."""

    track_id: int
    area: float
    z: float
    gap: float
    label: str
    frames: int
    #: Whether the z-score alone put this individual below the flight. Kept
    #: apart from ``label`` so a near miss can be reported: the gap condition
    #: is deliberately hard to clear on a small flight - the candidate sits in
    #: the lower half and so inflates the very interquartile range it is
    #: tested against - and a user seeing "no juvenile" deserves to know the
    #: difference between "nothing was small" and "something was small but the
    #: cohort was too thin to confirm it".
    low_outlier: bool = False
    #: The gap that separates this individual from the herd: its own ``gap``,
    #: or - for the smaller of several juveniles - the one above the largest
    #: of them. ``0`` for an individual no such gap lies above.
    separation: float = 0.0


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _quartiles(values: Sequence[float]):
    """``(q1, q3)`` by the same halving rule as :func:`_median`."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    lower = ordered[:middle]
    upper = ordered[middle + 1:] if len(ordered) % 2 else ordered[middle:]
    if not lower or not upper:
        return ordered[0], ordered[-1]
    return _median(lower), _median(upper)


def median_absolute_deviation(values: Sequence[float]) -> float:
    centre = _median(values)
    return _median([abs(value - centre) for value in values])


def track_areas(rows: Iterable[dict], geo: bool = False) -> Dict[int, dict]:
    """Median box area per track, with the frame count behind it.

    The *median* over the track rather than the mean: a box that briefly
    swallows a neighbour, or clips the animal at the frame edge, should not
    move an individual's recorded size.
    """
    per_track: Dict[int, List[float]] = {}
    for row in rows:
        if geo:
            width = abs(float(row["gx2"]) - float(row["gx1"]))
            height = abs(float(row["gy2"]) - float(row["gy1"]))
        else:
            width = abs(float(row["x2"]) - float(row["x1"]))
            height = abs(float(row["y2"]) - float(row["y1"]))
        area = width * height
        if area > 0:
            per_track.setdefault(int(row["track_id"]), []).append(area)

    return {track_id: {"area": _median(areas), "frames": len(areas)}
            for track_id, areas in per_track.items() if areas}


def cohorts(areas: Dict[int, dict], cohort_of: Dict[int, object]
            ) -> Dict[object, Dict[int, dict]]:
    """*areas* split into the groups that are compared among themselves.

    :param cohort_of: ``track_id -> species`` (any hashable); a track it
        does not name goes into the cohort ``""`` - the animals nobody
        identified, which are compared with each other
    """
    groups: Dict[object, Dict[int, dict]] = {}
    for track_id, entry in areas.items():
        groups.setdefault(cohort_of.get(track_id) or "", {})[track_id] = entry
    return groups


def assess(areas: Dict[int, dict], config: Config = Config(),
           adults: Sequence[int] = ()) -> List[Assessment]:
    """Classify every individual of one cohort by size.

    *adults* names tracks already known to be adult - a male called by the sex
    classifier is an adult whatever its box says, because the cue that marked
    him is antlers.

    Everything that is not a low outlier comes back ``adult`` rather than
    unknown: the test is a low-outlier flag by design, and in a surveyed herd
    every non-outlier is an adult by construction.
    """
    if len(areas) < max(1, int(config.min_individuals)):
        return []

    values = [entry["area"] for entry in areas.values()]
    centre = _median(values)
    deviation = median_absolute_deviation(values)
    q1, q3 = _quartiles(values)
    spread = q3 - q1

    ordered = sorted(areas.items(),
                     key=lambda item: (item[1]["area"], item[0]))
    known_adults = {int(track_id) for track_id in adults}

    scores, gaps = [], []
    for position, (_track_id, entry) in enumerate(ordered):
        area = entry["area"]
        # A zero MAD means every individual is the same size, so nothing is an
        # outlier - not that everything is infinitely far from the median.
        scores.append((MAD_SCALE * (area - centre) / deviation)
                      if deviation else 0.0)
        # The gap upwards: how far this individual sits below the next-smallest
        # one. The largest individual has nothing above it, and cannot be a low
        # outlier anyway.
        gaps.append(ordered[position + 1][1]["area"] - area
                    if position + 1 < len(ordered) else 0.0)

    # Where the herd begins: above the largest individual that is far below
    # the median *and* has a wide enough gap over it. Everything as far below
    # the median and no larger sits under that same gap - the twin as much as
    # its sibling.
    below = -1
    for position, z in enumerate(scores):
        if z < config.z_threshold and \
                gaps[position] > config.iqr_factor * spread:
            below = position

    results = []
    for position, (track_id, entry) in enumerate(ordered):
        z = scores[position]
        low = z < config.z_threshold and track_id not in known_adults
        juvenile = low and position <= below
        results.append(Assessment(
            track_id=int(track_id), area=entry["area"], z=z,
            gap=gaps[position],
            label=JUVENILE if juvenile else ADULT, frames=entry["frames"],
            low_outlier=low,
            separation=gaps[below] if position <= below else 0.0))

    return sorted(results, key=lambda item: item.track_id)


def explain(assessments: Sequence[Assessment], config: Config = Config(),
            source: str = AREA_PIXEL, cohort: str = "",
            size: int = 0) -> str:
    """A one-line summary for the run log.

    :param cohort: the species these individuals were compared within, for
        a flight with several; empty for the flight as a whole
    :param size: how many individuals the cohort has - named when there are
        too few of them to compare
    """
    juveniles = [item for item in assessments if item.label == JUVENILE]
    head = f"Life stage ({cohort})" if cohort else "Life stage"
    where = "of this species" if cohort else "in this flight"
    if not assessments:
        count = f" ({size})" if size else ""
        return (f"{head}: too few individuals {where}{count} for a robust "
                f"size comparison (fewer than {config.min_individuals}).")
    if not juveniles:
        near = [item for item in assessments if item.low_outlier]
        if near:
            names = ", ".join(f"track {item.track_id} (z={item.z:.1f})"
                              for item in near)
            return (f"{head}: no juvenile called among "
                    f"{len(assessments)} individual(s), though {names} sits "
                    "well below the flight median - its size gap to the next "
                    "animal is not wide enough to separate it from the herd. "
                    "On a small flight the outlier inflates the very spread "
                    "it is measured against, so this is the cautious answer "
                    f"rather than a confident one (areas from the {source} "
                    "boxes).")
        return (f"{head}: no juvenile found among {len(assessments)} "
                f"individual(s) - the smallest sits on a smooth size "
                "continuum with the rest.")
    named = ", ".join(f"track {item.track_id} (z={item.z:.1f})"
                      for item in juveniles)
    return (f"{head}: {len(juveniles)} juvenile(s) of "
            f"{len(assessments)} individual(s) - {named}; areas measured on "
            f"the {source} boxes.")


def cohort_ratio(assessments: Sequence[Assessment]) -> Dict[int, float]:
    """Each individual's size as a fraction of the flight median.

    The figure the paper quotes for its juveniles (0.53x, 0.60x), and the most
    directly interpretable number to show a user.
    """
    if not assessments:
        return {}
    centre = _median([item.area for item in assessments])
    if not centre:
        return {}
    return {item.track_id: item.area / centre for item in assessments}
