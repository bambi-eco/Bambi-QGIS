# -*- coding: utf-8 -*-
"""How far off the place of a detection may be.

A detection's place on the map is where the line of sight through its pixel
meets the terrain model. It is as good as what goes into that: where the
camera was, which way it looked, how high the terrain really is, which
pixel the box sits on, and whether the frame and the camera's position are
of the same moment. None of these is known exactly, and how much each
matters depends on the geometry of the one detection: straight below the
drone a degree of camera angle is 1.4 m on the ground from 80 m up, at 45
degrees to the side it is twice that along the line of sight, and on a
slope that falls away from the camera more again.

**The budget** (:func:`budget`). With ``H`` the height of the camera above
the animal's ground, ``theta`` the angle of the line of sight from straight
down, ``R = H / cos(theta)`` its length, and ``k = 1 / (1 + s tan(theta))``
what a terrain that rises (``s`` > 0) or falls (``s`` < 0) away from the
camera does to it, the one-sigma error of the place is, along the line from
the camera to the animal (*radial*) and across it (*cross*):

=====================  =============================  =====================
what is not known      radial                         cross
=====================  =============================  =====================
camera position (xy)   ``sigma * k``                  ``sigma``
camera height          ``sigma * tan(theta) * k``     -
terrain height         ``sigma * tan(theta) * k``     -
camera angle           ``sigma * R^2 / H * k^2``      ``sigma * R``
the pixel              as the angle, ``sigma`` in pixels times the angle one
                       pixel spans
frame against pose     the drone's speed times ``sigma`` (seconds), along
                       its direction of flight
=====================  =============================  =====================

The parts are independent, so their squares add. ``error_m`` is the
root-mean-square distance of the true place from the mapped one (about two
out of three detections lie within it); ``error95_m`` is twice that, within
which 95 to 98 in a hundred lie.

**What it is not.** The accuracies are the ones set, not measured ones: the
result says what follows *if* the camera's angle is good to a degree, and
which of the parts matters most - which is the one to improve. A constant
offset that the Correction Wizard would remove is not in it either.

Free of QGIS, Qt and - but for the percentiles of a summary - numpy.
"""

import csv
import json
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: The parts of the budget, as they are named in results.
GPS = "camera position"
HEIGHT = "camera height"
TERRAIN = "terrain model"
ANGLE = "camera angle"
PIXEL = "pixel"
TIMING = "timing"
SOURCES = (GPS, HEIGHT, TERRAIN, ANGLE, PIXEL, TIMING)

#: Beyond this the terrain all but runs along the line of sight: the factor
#: is held there, and the detection marked.
MAX_TERRAIN_FACTOR = 10.0
#: A line of sight flatter than this many degrees from the horizontal is
#: not one a place can be read from.
MAX_OFF_NADIR_DEG = 85.0

BOX_COLUMNS = [
    "flight", "detection_id", "track_id", "frame", "x", "y", "height_m",
    "off_nadir_deg", "range_m", "slope_deg", "radial_m", "cross_m",
    "error_m", "error95_m", "azimuth_deg", "main_source", "grazing",
]
TRACK_COLUMNS = [
    "flight", "track_id", "species", "x", "y", "boxes", "error_m",
    "error95_m", "best_error_m", "main_source",
]


class UncertaintyError(RuntimeError):
    """The uncertainty cannot be worked out as asked."""


@dataclass
class Accuracy:
    """How well each thing that goes into a place is known (one sigma)."""

    #: Horizontal position of the camera, metres.
    gps_horizontal_m: float = 1.5
    #: Height of the camera, metres.
    gps_vertical_m: float = 1.0
    #: Direction the camera looked in, degrees.
    camera_angle_deg: float = 1.0
    #: Height of the terrain model, metres.
    terrain_m: float = 1.0
    #: Where on the animal the middle of the box sits, pixels.
    pixel_px: float = 2.0
    #: Frame against camera position, seconds.
    sync_s: float = 0.1

    def validate(self) -> "Accuracy":
        for name, value in asdict(self).items():
            number = isinstance(value, (int, float)) and not isinstance(
                value, bool)
            if not number or not math.isfinite(value) or value < 0:
                raise UncertaintyError(
                    f"'{value}' is not an accuracy ({name}): a number of "
                    "nought or more is.")
        return self

    @classmethod
    def from_config(cls, config: dict) -> "Accuracy":
        """From the ``position_*`` settings of a configuration."""
        defaults = cls()
        values = {}
        for name in asdict(defaults):
            raw = config.get(f"position_{name}")
            if raw is None:
                values[name] = getattr(defaults, name)
                continue
            try:
                values[name] = float(raw)
            except (TypeError, ValueError) as exc:
                raise UncertaintyError(
                    f"'{raw}' is not an accuracy ({name}).") from exc
        return cls(**values).validate()


#: Ready-made accuracies: a drone that knows its place from satellites
#: alone, and one corrected by a base station.
PRESETS = OrderedDict([
    ("Standard GNSS", Accuracy()),
    ("RTK", Accuracy(gps_horizontal_m=0.05, gps_vertical_m=0.1)),
])


def pixel_angle(fovy_deg: Optional[float], image_height: Optional[int]
                ) -> Optional[float]:
    """The angle one pixel spans (radians) at the middle of a frame."""
    if not fovy_deg or not image_height:
        return None
    return 2.0 * math.tan(math.radians(float(fovy_deg)) / 2.0) / float(
        image_height)


def budget(height_m: float, offset_m: float, accuracy: Accuracy,
           slope: float = 0.0, pixel_rad: Optional[float] = None,
           speed_ms: float = 0.0, flight_angle_rad: float = 0.0) -> dict:
    """The error budget of one detection.

    :param height_m: the camera's height above the animal's ground
    :param offset_m: how far the animal is from the point below the camera
    :param slope: the terrain's rise per metre away from the camera, at the
        animal (negative: falling away)
    :param pixel_rad: the angle one pixel spans; ``None`` leaves the pixel
        out
    :param speed_ms: the drone's speed over the ground
    :param flight_angle_rad: the angle between the drone's direction of
        flight and the line from the camera to the animal
    :return: ``{"off_nadir_deg", "range_m", "terrain_factor", "grazing",
        "radial_m", "cross_m", "error_m", "error95_m", "parts": {source:
        (radial, cross)}, "main_source"}``
    :raises UncertaintyError: for a camera that is not above the animal, or
        a line of sight that is all but level
    """
    if not (math.isfinite(height_m) and height_m > 0):
        raise UncertaintyError(
            "The camera is not above the ground the detection lies on.")
    offset_m = abs(float(offset_m))
    off_nadir = math.atan2(offset_m, height_m)
    if math.degrees(off_nadir) > MAX_OFF_NADIR_DEG:
        raise UncertaintyError(
            "The line of sight is all but level: no place can be read "
            "from it.")
    tangent = offset_m / height_m
    reach = math.hypot(offset_m, height_m)
    lean = 1.0 + float(slope) * tangent
    grazing = lean < 1.0 / MAX_TERRAIN_FACTOR
    factor = MAX_TERRAIN_FACTOR if grazing else 1.0 / lean

    angle = math.radians(accuracy.camera_angle_deg)
    pixel = accuracy.pixel_px * pixel_rad if pixel_rad else 0.0
    drift = float(speed_ms) * accuracy.sync_s
    swing = reach * reach / height_m * factor * factor
    parts = OrderedDict([
        (GPS, (accuracy.gps_horizontal_m * factor,
               accuracy.gps_horizontal_m)),
        (HEIGHT, (accuracy.gps_vertical_m * tangent * factor, 0.0)),
        (TERRAIN, (accuracy.terrain_m * tangent * factor, 0.0)),
        (ANGLE, (angle * swing, angle * reach)),
        (PIXEL, (pixel * swing, pixel * reach)),
        (TIMING, (drift * abs(math.cos(flight_angle_rad)) * factor,
                  drift * abs(math.sin(flight_angle_rad)))),
    ])
    radial = math.sqrt(sum(part[0] ** 2 for part in parts.values()))
    cross = math.sqrt(sum(part[1] ** 2 for part in parts.values()))
    error = math.hypot(radial, cross)
    main = max(parts, key=lambda name: math.hypot(*parts[name]))
    return {
        "off_nadir_deg": math.degrees(off_nadir),
        "range_m": reach,
        "terrain_factor": factor,
        "grazing": grazing,
        "radial_m": radial,
        "cross_m": cross,
        "error_m": error,
        "error95_m": 2.0 * error,
        "parts": parts,
        "main_source": main if error > 0 else None,
    }


def describe_box(camera: Tuple[float, float, float],
                 ground: Tuple[float, float, float], accuracy: Accuracy,
                 gradient: Optional[Tuple[float, float]] = None,
                 pixel_rad: Optional[float] = None,
                 velocity: Tuple[float, float] = (0.0, 0.0)
                 ) -> Optional[dict]:
    """The budget of a detection from where the camera and the animal were.

    :param camera: the camera ``(x, y, z)``
    :param ground: the detection's place ``(x, y, z)``, same coordinates
    :param gradient: the terrain's rise per metre east and north at the
        animal; ``None`` = level
    :param velocity: the drone's velocity over the ground ``(east, north)``
    :return: :func:`budget` plus ``height_m``, ``slope_deg`` (along the
        line of sight; positive rising away) and ``azimuth_deg`` (compass
        bearing from the camera to the animal) - or ``None`` where the
        geometry gives no place (a camera below the animal, a level line
        of sight)
    """
    east, north = ground[0] - camera[0], ground[1] - camera[1]
    height = camera[2] - ground[2]
    offset = math.hypot(east, north)
    if offset > 0:
        along = (east / offset, north / offset)
    else:
        along = (0.0, 1.0)              # straight below: any direction
    slope = 0.0
    if gradient is not None:
        slope = gradient[0] * along[0] + gradient[1] * along[1]
    speed = math.hypot(*velocity)
    flight_angle = 0.0
    if speed > 0:
        cosine = (velocity[0] * along[0] + velocity[1] * along[1]) / speed
        flight_angle = math.acos(max(-1.0, min(1.0, cosine)))
    try:
        found = budget(height, offset, accuracy, slope, pixel_rad, speed,
                       flight_angle)
    except UncertaintyError:
        return None
    found.update({
        "height_m": height,
        "slope_deg": math.degrees(math.atan(slope)),
        "azimuth_deg": math.degrees(math.atan2(along[0], along[1])) % 360.0,
    })
    return found


# ---------------------------------------------------------------------------
# A flight, and a survey
# ---------------------------------------------------------------------------

def box_rows(flight_name, boxes: Sequence[dict],
             cameras: Dict[int, Tuple[float, float, float]],
             accuracy: Accuracy, pixel_rad: Optional[float] = None,
             gradient_of=None, seconds: Optional[Dict[int, float]] = None
             ) -> Tuple[List[OrderedDict], int]:
    """One row per detection that has a place.

    :param boxes: ``{"detection_id", "track_id", "frame", "x", "y", "z"}``
    :param cameras: ``frame -> (x, y, z)`` of the camera, in the same
        coordinates as the boxes
    :param gradient_of: ``gradient_of(x, y) -> (east, north)`` rise per
        metre, or ``None`` where the terrain is not known; ``None`` = level
        everywhere
    :param seconds: ``frame -> capture time``; with it the drone's speed is
        read from the frames either side, without it the timing counts for
        nothing
    :return: ``(rows, left out)`` - the rows as :data:`BOX_COLUMNS`, and
        how many boxes had no camera or a geometry that gives no place
    """
    velocities = _velocities(cameras, seconds or {})
    rows, left_out = [], 0
    for box in boxes:
        frame = box.get("frame")
        camera = cameras.get(frame)
        if camera is None or None in (box.get("x"), box.get("y"),
                                      box.get("z")):
            left_out += 1
            continue
        gradient = None
        if gradient_of is not None:
            gradient = gradient_of(float(box["x"]), float(box["y"]))
        found = describe_box(
            camera, (float(box["x"]), float(box["y"]), float(box["z"])),
            accuracy, gradient, pixel_rad, velocities.get(frame, (0.0, 0.0)))
        if found is None:
            left_out += 1
            continue
        values = dict(found, flight=flight_name,
                      detection_id=box.get("detection_id"),
                      track_id=box.get("track_id"), frame=frame,
                      x=float(box["x"]), y=float(box["y"]))
        rows.append(OrderedDict((key, values.get(key))
                                for key in BOX_COLUMNS))
    return rows, left_out


def _velocities(cameras: Dict[int, Tuple[float, float, float]],
                seconds: Dict[int, float]
                ) -> Dict[int, Tuple[float, float]]:
    """The drone's velocity over the ground at each frame, from the frames
    either side that have a time."""
    timed = sorted(frame for frame in cameras if frame in seconds)
    found = {}
    for index, frame in enumerate(timed):
        before = timed[max(0, index - 1)]
        after = timed[min(len(timed) - 1, index + 1)]
        elapsed = seconds[after] - seconds[before]
        if elapsed <= 0:
            continue
        found[frame] = (
            (cameras[after][0] - cameras[before][0]) / elapsed,
            (cameras[after][1] - cameras[before][1]) / elapsed)
    return found


def _percentile(values: Sequence[float], share: float) -> Optional[float]:
    values = sorted(values)
    if not values:
        return None
    position = share * (len(values) - 1)
    low = int(math.floor(position))
    high = min(len(values) - 1, low + 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def track_rows(rows: Sequence[dict], animals: Sequence[dict]
               ) -> List[OrderedDict]:
    """One row per counted animal: how far off its place may be.

    An animal's place is the middle of its boxes. What its boxes have in
    common - where the drone took itself to be, a terrain model that is
    off - does not average out, so the animal is given the median of its
    boxes' errors rather than something smaller; ``best_error_m`` is its
    best view.

    :param animals: ``{"flight", "track_id", "species", "x", "y"}``
    """
    by_track: Dict[Tuple[object, object], List[dict]] = {}
    for row in rows:
        if row.get("track_id") is not None:
            by_track.setdefault((row["flight"], row["track_id"]),
                                []).append(row)
    out = []
    for animal in animals:
        of_animal = by_track.get((animal.get("flight"),
                                  animal.get("track_id")), [])
        errors = [row["error_m"] for row in of_animal]
        sources: Dict[str, int] = {}
        for row in of_animal:
            if row.get("main_source"):
                sources[row["main_source"]] = sources.get(
                    row["main_source"], 0) + 1
        median = _percentile(errors, 0.5)
        out.append(OrderedDict([
            ("flight", animal.get("flight")),
            ("track_id", animal.get("track_id")),
            ("species", animal.get("species")),
            ("x", animal.get("x")), ("y", animal.get("y")),
            ("boxes", len(of_animal)),
            ("error_m", median),
            ("error95_m", None if median is None else 2.0 * median),
            ("best_error_m", min(errors) if errors else None),
            ("main_source", max(sources, key=lambda name: (
                sources[name], -SOURCES.index(name))) if sources else None),
        ]))
    return out


def summarise(rows: Sequence[dict]) -> dict:
    """The detections of *rows* taken together::

        {"boxes", "median_m", "p90_m", "max_m", "median95_m",
         "within": {metres: share}, "main_sources": {source: share},
         "grazing", "median_height_m", "median_off_nadir_deg"}
    """
    errors = [row["error_m"] for row in rows]
    n = len(errors)
    main: Dict[str, int] = OrderedDict((name, 0) for name in SOURCES)
    for row in rows:
        if row.get("main_source") in main:
            main[row["main_source"]] += 1
    median = _percentile(errors, 0.5)
    return {
        "boxes": n,
        "median_m": median,
        "p90_m": _percentile(errors, 0.9),
        "max_m": max(errors) if errors else None,
        "median95_m": None if median is None else 2.0 * median,
        "within": OrderedDict(
            (str(limit), sum(1 for e in errors if e <= limit) / n if n
             else None) for limit in (1, 2, 5, 10)),
        "main_sources": OrderedDict(
            (name, count / n) for name, count in main.items() if n and count),
        "grazing": sum(1 for row in rows if row.get("grazing")),
        "median_height_m": _percentile(
            [row["height_m"] for row in rows], 0.5),
        "median_off_nadir_deg": _percentile(
            [row["off_nadir_deg"] for row in rows], 0.5),
    }


def sentence(summary: dict) -> str:
    """A summary as it is said."""
    if not summary.get("boxes"):
        return "No detection has a place to judge."
    sources = summary.get("main_sources") or {}
    text = (f"{summary['boxes']} detections: typically "
            f"{summary['median_m']:.1f} m off (median; nine in ten within "
            f"{summary['p90_m']:.1f} m, the worst {summary['max_m']:.1f} m)")
    if sources:
        name = max(sources, key=lambda key: sources[key])
        text += (f". What matters most is the {name} - for "
                 f"{sources[name] * 100:.0f} % of them")
    if summary.get("grazing"):
        text += (f". {summary['grazing']} lie where the terrain all but "
                 "runs along the line of sight: their place is hardly "
                 "known")
    return text + "."


def describe(document: dict) -> List[str]:
    """The result in a few lines, for the log."""
    lines = [sentence(document["summary"])]
    for name, summary in (document.get("by_flight") or {}).items():
        if summary.get("boxes"):
            lines.append(f"  {name}: median {summary['median_m']:.1f} m, "
                         f"nine in ten within {summary['p90_m']:.1f} m")
    return lines


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------

def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else ""
    if isinstance(value, float):
        return round(value, 3)
    return value


def write_table(rows: Sequence[dict], path: str,
                columns: Sequence[str] = tuple(BOX_COLUMNS)) -> str:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row.get(key)) for key in columns])
    return path


def write_json(document: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
    return path


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
