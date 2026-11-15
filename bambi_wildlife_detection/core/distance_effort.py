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
"""

import math
from typing import List, Optional, Sequence, Tuple

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
