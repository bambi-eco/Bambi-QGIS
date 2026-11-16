# -*- coding: utf-8 -*-
"""Reaction to the drone: did the animals move, and where to?

A drone that drives the animals it counts away is counting something else.
Every tracked animal is a short film of where it was, frame by frame; this
module reads what the animal did while it was in view - how far it went,
how fast, and in which direction relative to the line the drone flew - and
puts the animals of a survey together to see whether they fled.

**What an animal did** (:func:`fit_motion`, :func:`describe_track`). The
positions of a track jitter: the camera's attitude is not known exactly,
and a box does not sit on the same pixel of the animal twice. Adding up the
steps from frame to frame therefore gives every animal a path, standing or
not. Instead a constant velocity is fitted through all positions of the
track; what is left over is the jitter, and it says how well the velocity
is known. An animal **moved** when the distance that velocity carries it
during its time in view is at least the set minimum (3 m by default) and
three times its own standard error. A track of too few positions or too
short a time is not judged.

**Relative to the line.** The direction the drone flew while the animal was
in view is the line. What the animal covered is split into the part *away
from the line* (to the side the animal was on; negative: towards it) and
the part *along* it (in the direction of flight). The part across the line
is the one to trust: a warm spot that is not at the height the terrain
model has for it - on a bush, under a slope - seems to slide along the line
as the camera passes over it, and never across.

**The animals together** (:func:`summarise`). How many moved; of those
that went sideways, how many away from the line and how many towards it -
without a reaction as many one way as the other; the mean speed away from
the line over all animals, with an interval; and whether the animals
nearer the line were the faster ones.

If the animals do move away from the line, the perpendicular distances a
distance-sampling estimate works with are too large, and its density too
low.

numpy and scipy are imported inside the functions that need them. Free of
QGIS and Qt.
"""

import csv
import json
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

#: An animal moved when it covered at least this while in view ...
MIN_DISPLACEMENT_M = 3.0
#: ... and that is this many of its standard errors.
SIGMAS = 3.0
#: Tracks shorter than this, or of fewer positions, are not judged.
MIN_DURATION_S = 2.0
MIN_POSITIONS = 4
#: Nearer the line than this an animal is under it: it has no side.
UNDER_THE_LINE_M = 2.0
#: A drone that covered less than this while the animal was in view was
#: hovering: there is no line then, only the drone.
HOVER_M = 1.0

MOVED = "moved"
STAYED = "stayed"
NOT_JUDGED = "not judged"

AWAY = "away from the line"
TOWARDS = "towards the line"
AHEAD = "ahead, along the line"
BACK = "back, along the line"

ALL = "All species"
LEVEL = 0.95

TRACK_COLUMNS = [
    "flight", "track_id", "species", "sex", "age", "status", "direction",
    "n_positions", "duration_s", "displacement_m", "displacement_se_m",
    "speed_ms", "speed_se_ms", "heading_deg", "position_noise_m",
    "away_m", "along_m", "away_ms", "line_distance_m", "drone_distance_m",
    "start_x", "start_y", "end_x", "end_y",
]


class ReactionError(RuntimeError):
    """The reaction cannot be read as asked."""


# ---------------------------------------------------------------------------
# One animal
# ---------------------------------------------------------------------------

def fit_motion(points: Sequence[Tuple[float, float, float]]) -> Optional[dict]:
    """A constant velocity through the positions of a track.

    :param points: ``(seconds, x, y)`` per position, in any order
    :return: ``{"n", "first_s", "duration_s", "velocity": (vx, vy), "speed_ms",
        "speed_se_ms", "displacement_m", "displacement_se_m",
        "position_noise_m", "start": (x, y), "end": (x, y)}`` - the start
        and the end are where the fitted motion has the animal at the first
        and the last moment, the noise is what the positions lie off it by
        (per axis). ``None`` for fewer than three positions or no time
        between them.
    """
    points = sorted((float(t), float(x), float(y)) for t, x, y in points)
    n = len(points)
    if n < 3:
        return None
    mean_t = sum(p[0] for p in points) / n
    mean_x = sum(p[1] for p in points) / n
    mean_y = sum(p[2] for p in points) / n
    spread = sum((p[0] - mean_t) ** 2 for p in points)
    duration = points[-1][0] - points[0][0]
    if spread <= 0 or duration <= 0:
        return None
    vx = sum((p[0] - mean_t) * (p[1] - mean_x) for p in points) / spread
    vy = sum((p[0] - mean_t) * (p[2] - mean_y) for p in points) / spread
    left_over = 0.0
    for moment, x, y in points:
        off_x = x - mean_x - vx * (moment - mean_t)
        off_y = y - mean_y - vy * (moment - mean_t)
        left_over += off_x * off_x + off_y * off_y
    noise = math.sqrt(left_over / (2 * n - 4))
    velocity_se = noise / math.sqrt(spread)
    speed = math.hypot(vx, vy)

    def at(moment: float) -> Tuple[float, float]:
        return (mean_x + vx * (moment - mean_t),
                mean_y + vy * (moment - mean_t))

    return {
        "n": n,
        "first_s": points[0][0],
        "duration_s": duration,
        "velocity": (vx, vy),
        "speed_ms": speed,
        "speed_se_ms": velocity_se,
        "displacement_m": speed * duration,
        "displacement_se_m": velocity_se * duration,
        "position_noise_m": noise,
        "start": at(points[0][0]),
        "end": at(points[-1][0]),
    }


def _heading(vx: float, vy: float) -> float:
    """Compass bearing of a movement: 0 = north, 90 = east."""
    return math.degrees(math.atan2(vx, vy)) % 360.0


def describe_track(points: Sequence[Tuple[float, float, float]],
                   drone: Optional[Sequence[Tuple[float, float, float]]] = None,
                   min_displacement: float = MIN_DISPLACEMENT_M,
                   min_duration: float = MIN_DURATION_S,
                   min_positions: int = MIN_POSITIONS) -> dict:
    """What one animal did while it was in view.

    :param points: ``(seconds, x, y)`` per position of the animal
    :param drone: ``(seconds, x, y)`` of the drone at the same moments (any
        that lie in the animal's time in view); ``None`` = not known
    :return: the keys of :data:`TRACK_COLUMNS` that are about movement.
        ``status`` is :data:`MOVED`, :data:`STAYED` or :data:`NOT_JUDGED`;
        ``direction`` is set for an animal that moved and whose line is
        known.
    """
    out = {key: None for key in TRACK_COLUMNS[5:]}
    out.update({"status": NOT_JUDGED, "n_positions": len(points)})
    motion = fit_motion(points)
    if motion is None:
        return out
    vx, vy = motion["velocity"]
    out.update({
        "duration_s": motion["duration_s"],
        "displacement_m": motion["displacement_m"],
        "displacement_se_m": motion["displacement_se_m"],
        "speed_ms": motion["speed_ms"],
        "speed_se_ms": motion["speed_se_ms"],
        "position_noise_m": motion["position_noise_m"],
        "start_x": motion["start"][0], "start_y": motion["start"][1],
        "end_x": motion["end"][0], "end_y": motion["end"][1],
    })
    long_enough = motion["duration_s"] >= min_duration
    if motion["n"] >= min_positions and long_enough:
        covered_m = motion["displacement_m"]
        far_enough = covered_m >= min_displacement
        sure_enough = covered_m >= SIGMAS * motion["displacement_se_m"]
        out["status"] = MOVED if far_enough and sure_enough else STAYED
        if out["status"] == MOVED:
            out["heading_deg"] = _heading(vx, vy)

    # ---- relative to the drone and its line ------------------------------ #
    drone = sorted((float(t), float(x), float(y))
                   for t, x, y in drone or ())
    if not drone:
        return out
    first, last = drone[0], drone[-1]
    start = motion["start"]
    offset = (start[0] - first[1], start[1] - first[2])
    # the nearest the drone came, over the ground, while the animal was in
    # view - the animal where the fitted motion has it at that moment
    out["drone_distance_m"] = min(
        math.hypot(start[0] + vx * (t - motion["first_s"]) - x,
                   start[1] + vy * (t - motion["first_s"]) - y)
        for t, x, y in drone)
    flown = (last[1] - first[1], last[2] - first[2])
    flown_m = math.hypot(*flown)
    covered = (vx * motion["duration_s"], vy * motion["duration_s"])
    if flown_m >= HOVER_M:
        along = (flown[0] / flown_m, flown[1] / flown_m)
        forward = offset[0] * along[0] + offset[1] * along[1]
        side = (offset[0] - forward * along[0], offset[1] - forward * along[1])
        out["along_m"] = covered[0] * along[0] + covered[1] * along[1]
    else:
        # hovering: "away" is away from the drone itself
        along = None
        side = offset
    side_m = math.hypot(*side)
    out["line_distance_m"] = side_m
    sideways = None
    if side_m >= UNDER_THE_LINE_M:
        outward = (side[0] / side_m, side[1] / side_m)
        out["away_m"] = covered[0] * outward[0] + covered[1] * outward[1]
        out["away_ms"] = out["away_m"] / motion["duration_s"]
        sideways = abs(out["away_m"])
    elif along is not None:
        # under the line an animal has no side: whichever way it goes
        # across, it goes away. (Its speed away is left open - counted as
        # always positive it would make standing animals look like fleeing
        # ones.)
        sideways = abs(covered[1] * along[0] - covered[0] * along[1])
    if out["status"] == MOVED:
        lengthwise = abs(out["along_m"]) if out["along_m"] is not None else 0.0
        if sideways is not None and sideways >= lengthwise:
            towards = out["away_m"] is not None and out["away_m"] < 0
            out["direction"] = TOWARDS if towards else AWAY
        elif out["along_m"] is not None:
            out["direction"] = AHEAD if out["along_m"] > 0 else BACK
    return out


# ---------------------------------------------------------------------------
# The animals together
# ---------------------------------------------------------------------------

def _median(values: Sequence[float]) -> Optional[float]:
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def _summary_of(rows: Sequence[dict], level: float = LEVEL) -> dict:
    """What the animals of *rows* did, taken together."""
    from . import spatial_stats

    judged = [row for row in rows if row["status"] != NOT_JUDGED]
    moved = [row for row in judged if row["status"] == MOVED]
    low, high = spatial_stats.wilson(len(moved), len(judged))
    summary = {
        "tracks": len(rows),
        "judged": len(judged),
        "not_judged": len(rows) - len(judged),
        "moved": len(moved),
        "moved_share": len(moved) / len(judged) if judged else None,
        "moved_share_low": low, "moved_share_high": high,
        "median_displacement_m": _median(
            [row["displacement_m"] for row in moved]),
        "median_speed_ms": _median([row["speed_ms"] for row in moved]),
        "directions": OrderedDict(
            (name, sum(1 for row in moved if row["direction"] == name))
            for name in (AWAY, TOWARDS, AHEAD, BACK)),
        "away_p_value": None,
        "mean_away_ms": None, "mean_away_ms_low": None,
        "mean_away_ms_high": None, "away_animals": 0,
        "nearer_faster": None,
    }

    # ---- sideways: as many away as towards, unless the drone drives them - #
    away = summary["directions"][AWAY]
    towards = summary["directions"][TOWARDS]
    if away + towards > 0:
        summary["away_p_value"] = _binomial_p(away, away + towards, 0.5)

    # ---- the speed away from the line, over every animal judged ---------- #
    speeds = [row["away_ms"] for row in judged if row["away_ms"] is not None]
    summary["away_animals"] = len(speeds)
    if speeds:
        mean = sum(speeds) / len(speeds)
        summary["mean_away_ms"] = mean
        if len(speeds) > 1:
            from scipy import stats

            variance = sum((s - mean) ** 2 for s in speeds) / (len(speeds) - 1)
            error = math.sqrt(variance / len(speeds))
            half = float(stats.t.ppf(0.5 + level / 2.0,
                                     len(speeds) - 1)) * error
            summary["mean_away_ms_low"] = mean - half
            summary["mean_away_ms_high"] = mean + half

    # ---- nearer the line, faster? ---------------------------------------- #
    pairs = [(row["line_distance_m"], row["speed_ms"]) for row in judged
             if None not in (row["line_distance_m"], row["speed_ms"])]
    distances = {distance for distance, _speed in pairs}
    paces = {speed for _distance, speed in pairs}
    if len(pairs) >= 5 and len(distances) > 1 and len(paces) > 1:
        from scipy import stats

        found = stats.spearmanr([d for d, _s in pairs],
                                [s for _d, s in pairs])
        rho, chance = float(found[0]), float(found[1])
        if math.isfinite(rho) and math.isfinite(chance):
            summary["nearer_faster"] = {
                "correlation": rho, "p_value": chance, "animals": len(pairs)}
    summary["verdict"], summary["sentence"] = verdict(summary)
    return summary


def _binomial_p(successes: int, trials: int, share: float) -> float:
    """Two-sided exact binomial test."""
    import numpy as np
    from scipy import stats

    chances = stats.binom.pmf(np.arange(trials + 1), trials, share)
    return float(min(1.0, chances[
        chances <= chances[successes] * (1.0 + 1e-7)].sum()))


#: What the animals of a survey did, in a word.
FLED = "moved away from the line"
MOVED_ABOUT = "moved, in no particular direction"
QUIET = "no sign of a reaction"
UNKNOWN = "too little to tell"

ALPHA = 0.05


def verdict(summary: dict) -> Tuple[str, str]:
    """``(verdict, sentence)`` for a summary of :func:`summarise`."""
    judged, moved = summary["judged"], summary["moved"]
    if judged < 5:
        return UNKNOWN, (
            f"{judged} animal(s) were in view long enough to tell: too few "
            "to say how the animals took the drone.")
    kind, sentence = _verdict(summary)
    ahead = summary["directions"][AHEAD]
    back = summary["directions"][BACK]
    lengthwise = ahead + back
    mostly = moved >= 5 and lengthwise >= 0.6 * moved
    if mostly and max(ahead, back) >= 0.8 * lengthwise:
        sentence += (
            f" Note: {lengthwise} of the {moved} that moved went along the "
            f"line, {max(ahead, back)} of them the same way. That is what a "
            "terrain model a few metres off looks like - a warm spot above "
            "or below the ground it is placed on seems to slide along the "
            "line as the camera passes - rather than animals.")
    return kind, sentence


def _verdict(summary: dict) -> Tuple[str, str]:
    judged, moved = summary["judged"], summary["moved"]
    share = f"{moved} of {judged} animals moved while in view"
    low, high = summary["mean_away_ms_low"], summary["mean_away_ms_high"]
    away = summary["directions"][AWAY]
    towards = summary["directions"][TOWARDS]
    drifting = low is not None and low > 0
    chance = summary["away_p_value"]
    sided = chance is not None and chance < ALPHA and away > towards
    if drifting or sided:
        parts = []
        if away + towards:
            parts.append(f"{away} went away from the line and {towards} "
                         "towards it")
        if summary["mean_away_ms"] is not None and low is not None:
            parts.append(
                f"on average the animals moved {summary['mean_away_ms']:.2f} "
                f"m/s away from the line ({low:.2f} to {high:.2f})")
        return FLED, (f"{share}; " + ", and ".join(parts) + ". The animals "
                      "moved away from the line the drone flew: a reaction "
                      "to it.")
    if summary["moved_share"] is not None and summary["moved_share"] >= 0.5:
        return MOVED_ABOUT, (
            f"{share}, as many towards the line as away from it: they were "
            "on the move, but not from the drone.")
    return QUIET, (f"{share}, and those that did went no particular way: "
                   "no sign of a reaction to the drone.")


def summarise(rows: Sequence[dict], species_order: Sequence[str] = (),
              level: float = LEVEL) -> "OrderedDict[str, dict]":
    """Per species, and for all animals: what they did.

    ``{"tracks", "judged", "not_judged", "moved", "moved_share" (+ ``_low``,
    ``_high``), "median_displacement_m", "median_speed_ms", "directions",
    "away_p_value", "mean_away_ms" (+ ``_low``, ``_high``), "away_animals",
    "nearer_faster", "verdict", "sentence"}``
    """
    names = [name for name in species_order
             if any(row.get("species") == name for row in rows)]
    for row in rows:
        if row.get("species") not in names:
            names.append(row.get("species"))
    out: "OrderedDict[str, dict]" = OrderedDict()
    out[ALL] = _summary_of(rows, level)
    if len(names) > 1:
        for name in names:
            out[name] = _summary_of(
                [row for row in rows if row.get("species") == name], level)
    return out


# ---------------------------------------------------------------------------
# A survey
# ---------------------------------------------------------------------------

def flight_rows(flight_name, animals: Sequence[dict],
                positions: Dict[int, Sequence[Tuple[int, float, float]]],
                seconds: Dict[int, float],
                drone: Optional[Dict[int, Tuple[float, float]]] = None,
                min_displacement: float = MIN_DISPLACEMENT_M,
                min_duration: float = MIN_DURATION_S) -> List[dict]:
    """One row per counted animal of a flight.

    :param animals: the survey's animals of the flight
        (:func:`core.analytics_source.load_animals`, tracks)
    :param positions: ``track id -> (frame, x, y)`` of its placed boxes
    :param seconds: ``frame -> capture time`` in seconds
    :param drone: ``frame -> (x, y)`` of the camera, in the same
        coordinates; ``None`` = not known
    """
    rows = []
    for animal in animals:
        track_id = animal["track_id"]
        timed = [(seconds[frame], x, y)
                 for frame, x, y in positions.get(track_id, ())
                 if frame in seconds]
        frames = [frame for frame, _x, _y in positions.get(track_id, ())
                  if frame in seconds]
        above = None
        if drone is not None:
            above = [(seconds[frame], drone[frame][0], drone[frame][1])
                     for frame in frames if frame in drone]
        row = OrderedDict([
            ("flight", flight_name), ("track_id", track_id),
            ("species", animal.get("species")), ("sex", animal.get("sex")),
            ("age", animal.get("age"))])
        row.update(describe_track(timed, above, min_displacement,
                                  min_duration))
        rows.append(OrderedDict((key, row.get(key)) for key in TRACK_COLUMNS))
    return rows


def describe(document: dict) -> List[str]:
    """The result in a few lines, for the log."""
    lines = []
    for name, summary in document["summary"].items():
        lines.append(f"{name}: {summary['sentence']}")
        if summary.get("median_displacement_m") is not None:
            lines.append(
                f"  those that moved went {summary['median_displacement_m']:.1f}"
                f" m at {summary['median_speed_ms']:.2f} m/s (medians)")
        nearer = summary.get("nearer_faster")
        if nearer and nearer["p_value"] < ALPHA and nearer["correlation"] < 0:
            lines.append(
                "  the nearer the line, the faster (rank correlation "
                f"{nearer['correlation']:.2f}, p = {nearer['p_value']:.3f})")
    return lines


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 3)
    return value


def write_table(rows: Sequence[dict], path: str) -> str:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRACK_COLUMNS)
        for row in rows:
            writer.writerow([_cell(row.get(key)) for key in TRACK_COLUMNS])
    return path


def write_json(document: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
    return path


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
