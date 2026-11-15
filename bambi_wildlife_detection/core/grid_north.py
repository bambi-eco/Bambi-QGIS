# -*- coding: utf-8 -*-
"""The map's north and the drone's.

A drone reports where its camera looks as a heading from **true north** -
the direction of the meridian it is on. Everything the plugin computes lies
in a projected map: positions are metres east and north in the project's
CRS, and that grid's north is true north on the projection's central
meridian only. Away from it the two differ by the *grid convergence*: about
``(longitude - central meridian) * sin(latitude)`` - 1.4 degrees at Salzburg
and 3.9 degrees in Vorarlberg in UTM zone 33, up to 2.2 degrees at the edge
of a UTM zone at Austria's latitude.

The heading was used as if it were a map bearing. A frame was therefore
turned by that angle about the point below the drone: 1.4 m at the corner
of a frame taken from 80 m up for two degrees, 2.8 m for four - unless the
rotation correction had been calibrated by hand, which then absorbed it
together with the compass's own error.

Since the angle is known, it is now taken from the heading: at frame
extraction every pose gets the convergence at its own position
(:data:`KEY`), and the plugin's one camera model
(``core.camera_pose.corrected_pose``) subtracts it. Poses extracted before
carry no such entry and are used as they always were - with a rotation
correction that may already contain the angle, they must not change by
themselves.

``pyproj`` is imported where it is needed.
"""

import json
import math
import os
from typing import Optional

#: Pose entry: degrees by which the map's north lies east of true north
#: where the frame was taken. ``heading - convergence`` is the map bearing.
KEY = "grid_convergence"
#: Poses-file entry saying what was done to the headings.
HEADINGS = "headings"
#: Angles below this are not worth a line in a log.
_NOTEWORTHY = 0.1


def convergence(latitude, longitude, epsg: int):
    """Degrees by which the north of the map (EPSG:*epsg*) lies east -
    clockwise - of true north at a place; west is negative.

    Worked out from the projection itself rather than from a formula for one
    kind of it: where a short step due north ends up on the map. Takes
    numbers or arrays.

    A true bearing becomes a map bearing by subtracting this.
    """
    import numpy as np
    from pyproj import CRS, Transformer

    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_epsg(int(epsg)), always_xy=True)
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    step = 1.0e-4                                   # of a degree: 11 m
    east, north = transformer.transform(longitude, latitude)
    east_n, north_n = transformer.transform(longitude, latitude + step)
    # on the map, true north points along this step; the map's own north is
    # that many degrees clockwise of it
    angle = -np.degrees(np.arctan2(np.asarray(east_n) - np.asarray(east),
                                   np.asarray(north_n) - np.asarray(north)))
    return float(angle) if angle.ndim == 0 else angle


def of_pose(meta: dict) -> float:
    """The convergence recorded for a pose; 0 for one extracted before."""
    try:
        value = float((meta or {}).get(KEY) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def map_heading(meta: dict) -> float:
    """A pose's heading as a bearing on the map, degrees."""
    rotation = list((meta or {}).get("rotation") or [0.0, 0.0, 0.0])
    heading = float(rotation[2]) if len(rotation) >= 3 else 0.0
    return (heading - of_pose(meta)) % 360.0


def _position(image: dict):
    latitude, longitude = image.get("lat"), image.get("lng")
    try:
        latitude, longitude = float(latitude), float(longitude)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(latitude) and math.isfinite(longitude)):
        return None
    if latitude == 0.0 and longitude == 0.0:
        return None                     # no position was recorded
    return latitude, longitude


def here(poses: dict, epsg: int) -> Optional[float]:
    """The convergence where a flight took place (its first located frame),
    or ``None`` when no frame says where it was taken."""
    for image in (poses or {}).get("images") or []:
        position = _position(image)
        if position is not None:
            return convergence(position[0], position[1], epsg)
    return None


def recorded(poses: dict) -> Optional[dict]:
    """What a poses file says was done to its headings, or ``None``."""
    entry = (poses or {}).get(HEADINGS)
    return entry if isinstance(entry, dict) and entry.get("north") == "map" \
        else None


def record(path: str, epsg: int) -> Optional[float]:
    """Give every pose of the poses file at *path* the convergence at its
    position, and say so in the file.

    :return: the angle at the flight (the median over its frames), or
        ``None`` when no frame says where it was taken - nothing is written
        then
    """
    import numpy as np

    with open(path, "r", encoding="utf-8") as handle:
        poses = json.load(handle)
    images = poses.get("images") or []
    located = [(image, _position(image)) for image in images]
    located = [(image, position) for image, position in located
               if position is not None]
    if not located:
        return None
    angles = np.atleast_1d(convergence(
        [position[0] for _image, position in located],
        [position[1] for _image, position in located], epsg))
    typical = float(np.median(angles))
    for (image, _position_), angle in zip(located, angles):
        image[KEY] = round(float(angle), 5)
    for image in images:
        # a frame without a position of its own: the flight's angle
        image.setdefault(KEY, round(typical, 5))
    poses[HEADINGS] = {"north": "map", "epsg": int(epsg),
                       KEY: round(typical, 5)}

    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(poses, handle)
    os.replace(temporary, path)
    return typical


def describe(angle: float, epsg: int) -> str:
    """The convergence in words, for a log."""
    side = "east" if angle >= 0 else "west"
    return (f"The map's north (EPSG:{int(epsg)}) lies {abs(angle):.2f} "
            f"degrees {side} of true north here")


def note_for_older_poses(poses: dict, epsg: int) -> Optional[str]:
    """A sentence for the log about poses whose headings are still taken as
    map bearings - ``None`` when they are not, or when it does not matter
    here."""
    if recorded(poses) is not None:
        return None
    try:
        angle = here(poses, epsg)
    except Exception:  # noqa: BLE001 - a note, not a requirement
        return None
    if angle is None or abs(angle) < _NOTEWORTHY:
        return None
    return (
        f"Note: {describe(angle, epsg)}. The frames of this flight were "
        "extracted with version 9.0 or earlier, which used the drone's "
        "heading - measured from true north - as a bearing on the map: "
        "everything is turned by that angle about the point below the "
        "drone, unless the rotation correction (z) was calibrated and took "
        "it up. Extracting the frames again takes the angle into account; "
        "a calibrated rotation correction has to be checked afterwards.")


def correction_hint(z_radians: float, angle: float) -> Optional[str]:
    """What to say at extraction when the flight has a rotation correction
    about the vertical: it was found while the convergence was not taken
    into account, and may contain it."""
    try:
        z = math.degrees(float(z_radians))
    except (TypeError, ValueError):
        return None
    if abs(z) < 1.0e-6 or abs(angle) < _NOTEWORTHY:
        return None
    return (
        f"Note: this flight has a rotation correction of z = {z:.2f} "
        "degrees. If it was calibrated on frames extracted with 9.0 or "
        f"earlier, it contains the {abs(angle):.2f} degrees between the "
        "map's north and true north, which are now taken from the heading "
        f"itself: it should then be about {z - angle:.2f} degrees. Check "
        "it in the Correction Wizard.")
