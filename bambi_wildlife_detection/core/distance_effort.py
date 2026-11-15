# -*- coding: utf-8 -*-
"""Effort and encounter rate for line-transect distance sampling.

Distance sampling turns ``n`` animals seen along ``L`` metres of line into a
density, ``n / (2 * ESW * L)``. Two things about ``L`` and ``n`` decide
whether that is right:

**What counts as line.** Only line that was searched, and only the animals
seen from it. The whole flight log used to count - take-off, the transit to
the first transect, the turns, the way home - which makes ``L`` too long and
the density too low. Now:

* where a flight has **transects** (Transect Splitting Tool), its lines are
  the transect legs - the camera's path between each transect's first and
  last frame - and an animal counts when its frame lies on one of them;
* otherwise the line is the camera's path **from the first to the last
  recorded frame**: nothing was searched before recording started or after
  it stopped;
* with a **survey area**, a line is as long as the part of it inside the
  area.

**How sure the encounter rate is.** ``n / L`` varies from line to line by
more than chance alone: animals are clumped, lines cross different ground.
A Poisson variance (``CV = 1 / sqrt(n)``) ignores that and is too small.
With two or more lines the variance is measured between them (Buckland et
al. 2001, eq. 3.4; the "R2" estimator of Fewster et al. 2009)::

    var(n / L) = k / (L^2 (k - 1)) * sum_i  l_i^2 (n_i / l_i - n / L)^2

and only a single line falls back on the Poisson figure.

**What is observed.** Animals that were together (``core.groups``) are one
observation: ``n`` counts groups, the detection function is fitted to the
distance of each group's middle, and the density of animals is the density
of groups times the mean group size - whose own uncertainty goes into the
interval (Buckland et al. 2001, section 3.5). Twelve deer in one meadow
counted as twelve sightings would claim a precision the survey does not
have.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .transects import cumulative_distances, flight_positions, path_length

#: How the lines of a flight were found.
TRANSECTS = "transects"
RECORDED_ROUTE = "recorded route"
FLIGHT_ROUTE = "flight route"

#: How the encounter-rate variance was obtained.
BETWEEN_LINES = "between lines"
POISSON = "poisson"

EFFORT_TEXT = {
    TRANSECTS: "the transect legs",
    RECORDED_ROUTE: "the route between the first and the last recorded frame",
    FLIGHT_ROUTE: ("the whole flight route (no frame positions to tell where "
                   "recording started and stopped)"),
}


def _path(positions, first: int, last: int,
          offset: Tuple[float, float]) -> List[Tuple[float, float]]:
    """The camera's ground path over the frames *first*..*last*, in world
    coordinates; frames without a position are skipped."""
    first = max(0, min(first, len(positions) - 1))
    last = max(0, min(last, len(positions) - 1))
    if first > last:
        first, last = last, first
    return [(pos[0] + offset[0], pos[1] + offset[1])
            for pos in positions[first:last + 1] if pos is not None]


def flight_lines(images: Sequence[dict], transects: Sequence = (),
                 area=None, offset: Tuple[float, float] = (0.0, 0.0)
                 ) -> Tuple[List[dict], str]:
    """The surveyed lines of one flight.

    :param images: the ``images`` of the camera's poses file
    :param transects: the flight's transects (objects with ``first_frame``,
        ``last_frame`` and ``display_name``); empty for none
    :param area: a :class:`core.survey_area.SurveyArea`, or ``None``
    :param offset: mesh-local -> world shift of the positions (the DEM
        origin); only matters with *area*
    :return: ``(lines, kind)`` - each line ``{"name", "first_frame",
        "last_frame", "length_m"}``, *kind* one of :data:`TRANSECTS` /
        :data:`RECORDED_ROUTE`. A line outside the survey area has length 0.
    """
    positions = flight_positions(list(images))
    if not positions:
        return [], RECORDED_ROUTE
    cumulative = cumulative_distances(positions)

    if transects:
        spans = [(t.display_name, t.first_frame, t.last_frame)
                 for t in transects]
        kind = TRANSECTS
    else:
        spans = [("recorded route", 0, len(positions) - 1)]
        kind = RECORDED_ROUTE

    lines = []
    for name, first, last in spans:
        if area is None:
            length = path_length(cumulative, first, last)
        else:
            length = area.clipped_length(_path(positions, first, last, offset))
        lines.append({"name": name,
                      "first_frame": min(first, last),
                      "last_frame": max(first, last),
                      "length_m": float(length)})
    return lines, kind


def line_of_frame(lines: Sequence[dict], frame) -> Optional[int]:
    """Index of the line whose frames hold *frame*; ``None`` when the frame
    lies on none - seen in transit, not on effort. Where two lines share a
    frame the first one has it."""
    if frame is None:
        return None
    try:
        frame = int(frame)
    except (TypeError, ValueError):
        return None
    for index, line in enumerate(lines):
        if line["first_frame"] <= frame <= line["last_frame"]:
            return index
    return None


def encounter_rate_cv(counts: Sequence[float], lengths: Sequence[float]
                      ) -> Tuple[Optional[float], str]:
    """Coefficient of variation of the encounter rate ``n / L``.

    :param counts: animals per line (after truncation)
    :param lengths: length of each line, metres
    :return: ``(cv, method)``: the between-line figure with two or more
        lines that have length, else the Poisson ``1 / sqrt(n)``; ``None``
        when nothing was seen.
    """
    pairs = [(float(n), float(length))
             for n, length in zip(counts, lengths) if float(length) > 0]
    total_n = sum(n for n, _length in pairs)
    total_length = sum(length for _n, length in pairs)
    if total_n <= 0 or total_length <= 0:
        return None, POISSON
    if len(pairs) < 2:
        return 1.0 / math.sqrt(total_n), POISSON

    k = len(pairs)
    rate = total_n / total_length
    spread = sum(length * length * (n / length - rate) ** 2
                 for n, length in pairs)
    variance = k / (total_length * total_length * (k - 1)) * spread
    return math.sqrt(variance) / rate, BETWEEN_LINES


# ---------------------------------------------------------------------------
# What is observed: groups
# ---------------------------------------------------------------------------

def group_observations(animals: Sequence[dict], group_of=None
                       ) -> Tuple[List[dict], int]:
    """The observations of a distance-sampling survey: one per group.

    Animals that were together (``core.groups``) are not seen one by one -
    whoever sees one sees the herd - so the unit a detection function is
    fitted to is the group, at the distance of its middle. An animal alone
    is a group of one, and with nothing but those this is the list of
    animals.

    :param animals: ``{"id", "distance_m", "line", "centre", "foot"}`` per
        animal seen: its distance from the line, the index of the line it
        was seen from (``None``: seen in transit), and - for placing the
        middle of a group - its own place and the point of the line nearest
        to it, as ``(x, y)`` or ``None``
    :param group_of: ``group_of(id)`` - the group an animal is in; ``None``
        (or an animal it does not know) leaves the animal alone
    :return: ``(observations, in transit)`` - ``{"group", "size",
        "distance_m", "line"}`` per group seen from a line, and how many
        animals were seen in transit only. A group seen from a line counts
        with all its members; it is on the line most of them were seen
        from.
    """
    members: Dict[object, List[dict]] = {}
    order: List[object] = []
    for position, animal in enumerate(animals):
        group = None
        if group_of is not None and animal.get("id") is not None:
            group = group_of(animal["id"])
        # an animal nobody can say anything about is alone - also when
        # several come without an id
        key = ("alone", position) if group is None else ("group", group)
        if key not in members:
            members[key] = []
            order.append(key)
        members[key].append(animal)

    observations, in_transit = [], 0
    for key in order:
        of_group = members[key]
        lines = [animal["line"] for animal in of_group
                 if animal.get("line") is not None]
        if not lines:
            in_transit += len(of_group)
            continue
        counts: Dict[int, int] = {}
        for line in lines:
            counts[line] = counts.get(line, 0) + 1
        line = min(counts, key=lambda index: (-counts[index], index))
        observations.append({
            "group": key[1] if key[0] == "group" else of_group[0].get("id"),
            "size": len(of_group),
            "distance_m": _group_distance(of_group),
            "line": line,
        })
    return observations, in_transit


def _group_distance(animals: Sequence[dict]) -> float:
    """How far the middle of a group is from the line.

    Each animal is some way off the line, to one side or the other: the
    middle of the group is at the mean of those offsets - two animals
    either side of the line have their middle on it. Where the offsets are
    not known, the mean of the distances has to do.
    """
    if len(animals) == 1:
        return float(animals[0]["distance_m"])
    offsets = []
    for animal in animals:
        centre, foot = animal.get("centre"), animal.get("foot")
        if centre is None or foot is None:
            offsets = None
            break
        offsets.append((float(centre[0]) - float(foot[0]),
                        float(centre[1]) - float(foot[1])))
    if offsets is None:
        return sum(float(a["distance_m"]) for a in animals) / len(animals)
    mean_x = sum(offset[0] for offset in offsets) / len(offsets)
    mean_y = sum(offset[1] for offset in offsets) / len(offsets)
    return math.hypot(mean_x, mean_y)


def group_size(sizes: Sequence[int]) -> dict:
    """The mean group size and how sure it is::

        {"mean", "se", "cv", "largest"}

    The density of animals is the density of groups times this mean, and
    its uncertainty is part of the density's.
    """
    sizes = [int(size) for size in sizes]
    n = len(sizes)
    if not n:
        return {"mean": None, "se": None, "cv": None, "largest": 0}
    mean = sum(sizes) / n
    if n > 1:
        variance = sum((size - mean) ** 2 for size in sizes) / (n - 1)
        se = math.sqrt(variance / n)
    else:
        se = 0.0
    return {"mean": mean, "se": se, "cv": se / mean if mean else None,
            "largest": max(sizes)}


#: A size bias is suspected below this one-sided p (Buckland et al. 2001,
#: section 3.5.4, test at alpha = 0.15).
SIZE_BIAS_ALPHA = 0.15


def size_bias(distances: Sequence[float], sizes: Sequence[int]
              ) -> Optional[dict]:
    """Whether the larger groups were the ones seen from further away.

    Near the limit of the strip a large group is more likely to be noticed
    than a single animal. The groups in the sample are then larger than the
    groups out there, and the mean group size - and with it the density of
    animals - comes out too high.

    :return: ``{"correlation", "p_value", "suspected"}`` - the correlation
        of log group size with distance and its one-sided p; ``None`` with
        fewer than five groups or where all are the same size
    """
    pairs = [(float(d), math.log(int(s))) for d, s in zip(distances, sizes)
             if int(s) > 0]
    n = len(pairs)
    if n < 5:
        return None
    mean_d = sum(p[0] for p in pairs) / n
    mean_s = sum(p[1] for p in pairs) / n
    var_d = sum((p[0] - mean_d) ** 2 for p in pairs)
    var_s = sum((p[1] - mean_s) ** 2 for p in pairs)
    if var_d <= 0 or var_s <= 0:
        return None
    r = sum((p[0] - mean_d) * (p[1] - mean_s) for p in pairs) / math.sqrt(
        var_d * var_s)
    r = max(-1.0, min(1.0, r))
    if abs(r) >= 1.0:
        p_value = 0.0 if r > 0 else 1.0
    else:
        t = r * math.sqrt((n - 2) / (1.0 - r * r))
        try:
            from scipy import stats
            p_value = float(stats.t.sf(t, n - 2))
        except ImportError:
            # the normal distribution instead of Student's: close enough
            # for a warning, and exact as the sample grows
            p_value = 0.5 * math.erfc(t / math.sqrt(2.0))
    return {"correlation": r, "p_value": p_value,
            "suspected": bool(r > 0 and p_value < SIZE_BIAS_ALPHA)}
