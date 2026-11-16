# -*- coding: utf-8 -*-
"""A flight's recordings, out of an archive and in a folder.

The BAMBI project publishes its flights as one ZIP archive each - the
videos as they came off the drone, their SRT files, the AirData flight log
and the two camera calibrations (:mod:`core.bambi_dataset` knows which
flights there are and fetches them). This module is what happens to such
an archive once it is there, and to the folder it is unpacked into:

**Unpacking** (:func:`extract`). The archives are flat. Members that would
land outside the folder, and an archive that unpacks to far more than it
should, are refused.

**Finding** (:func:`flight_files`). Which of the files in a folder are the
thermal and the RGB recordings, their SRT files, the flight log and the
calibrations - what the Input tab is filled with - and whether that is a
flight the pipeline can start on (:func:`is_complete`).

**Where it was flown** (:func:`recording_bounds`), from the flight log: the
ground an elevation model is needed for.

Free of QGIS and Qt.
"""

import csv
import hashlib
import os
import zipfile
from typing import Callable, Dict, List, Optional, Tuple

from . import checked_download

#: An archive that unpacks to more than this many times its own size is not
#: the flight.
MAX_EXPANSION = 20


class ArchiveError(RuntimeError):
    """An archive is not one a flight can be unpacked from."""


# ---------------------------------------------------------------------------
# The archive
# ---------------------------------------------------------------------------

def sha256_of(path: str, cancelled: Optional[Callable[[], bool]] = None
              ) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            if cancelled and cancelled():
                raise checked_download.Cancelled("cancelled")
            digest.update(block)
    return digest.hexdigest()


def extract(archive: str, folder: str,
            log: Optional[Callable[[str], None]] = None) -> List[str]:
    """Unpack the archive into *folder*; returns the files written.

    :raises ArchiveError: an archive that is none, a member that would
        land outside *folder*, or contents far larger than the archive
    """
    target = os.path.realpath(folder)
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = [info for info in bundle.infolist()
                       if not info.is_dir()]
            total = sum(info.file_size for info in members)
            limit = MAX_EXPANSION * max(os.path.getsize(archive), 1)
            if total > limit:
                raise ArchiveError(
                    f"{os.path.basename(archive)} unpacks to "
                    f"{total / 1e6:.0f} MB, which a flight of this size "
                    "does not: it is not unpacked.")
            for info in members:
                landing = os.path.realpath(
                    os.path.join(target, info.filename))
                if os.path.commonpath([target, landing]) != target:
                    raise ArchiveError(
                        f"{os.path.basename(archive)} holds a file that "
                        f"would be written outside the folder "
                        f"({info.filename}): it is not unpacked.")
            written = []
            os.makedirs(target, exist_ok=True)
            for info in members:
                written.append(bundle.extract(info, target))
    except zipfile.BadZipFile as exc:
        raise ArchiveError(
            f"{os.path.basename(archive)} is not a ZIP archive that can be "
            f"read ({exc}).") from exc
    if log:
        log(f"Unpacked {len(written)} file(s) into {folder}")
    return written


# ---------------------------------------------------------------------------
# The flight in the folder
# ---------------------------------------------------------------------------

def _camera(name: str) -> str:
    """``"t"`` or ``"w"`` for a DJI recording's name, ``""`` for neither:
    ``DJI_..._0003_T.MP4`` is the thermal camera's, ``..._W.MP4`` the wide
    one's."""
    stem = os.path.splitext(name)[0].upper()
    if stem.endswith("_T"):
        return "t"
    if stem.endswith("_W") or stem.endswith("_V"):
        return "w"
    return ""


def flight_files(folder: str) -> Dict[str, object]:
    """What the Input tab is filled with, found in *folder*::

        {"thermal_videos", "thermal_srts", "rgb_videos", "rgb_srts",
         "airdata", "thermal_calibration", "rgb_calibration"}

    Lists are in the order the recordings were made; what is not there is
    an empty list or ``""``.
    """
    found: Dict[str, object] = {
        "thermal_videos": [], "thermal_srts": [], "rgb_videos": [],
        "rgb_srts": [], "airdata": "", "thermal_calibration": "",
        "rgb_calibration": ""}
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return found
    kinds = {"t": "thermal", "w": "rgb"}
    for name in names:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        lower = name.lower()
        camera = kinds.get(_camera(name))
        if lower.endswith(".mp4") and camera:
            found[f"{camera}_videos"].append(path)
        elif lower.endswith(".srt") and camera:
            found[f"{camera}_srts"].append(path)
        elif lower == "t_calib.json":
            found["thermal_calibration"] = path
        elif lower == "w_calib.json":
            found["rgb_calibration"] = path
        elif lower.endswith(".csv") and not found["airdata"]:
            found["airdata"] = path
    return found


def is_complete(files: Dict[str, object]) -> bool:
    """Whether a folder holds a flight the pipeline can start on: a thermal
    recording with its SRT file, and the flight log."""
    videos = files.get("thermal_videos") or []
    subtitles = files.get("thermal_srts") or []
    return bool(videos) and len(subtitles) == len(videos) and bool(
        files.get("airdata"))


def recording_bounds(airdata_csv: str
                     ) -> Optional[Tuple[float, float, float, float]]:
    """``(west, south, east, north)`` of where the drone was while it
    recorded, from an AirData log - of the whole log where it does not say
    when it recorded. ``None`` where the log holds no position."""
    recording, everything = [], []
    try:
        with open(airdata_csv, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            names = {(name or "").strip().lower(): name
                     for name in reader.fieldnames or []}
            lat, lon = names.get("latitude"), names.get("longitude")
            video = names.get("isvideo")
            if not lat or not lon:
                return None
            for row in reader:
                try:
                    place = (float(row[lon]), float(row[lat]))
                except (TypeError, ValueError):
                    continue
                on_earth = -90 <= place[1] <= 90 and -180 <= place[0] <= 180
                if place == (0.0, 0.0) or not on_earth:
                    continue
                everything.append(place)
                if video and (row.get(video) or "").strip() == "1":
                    recording.append(place)
    except OSError:
        return None
    places = recording or everything
    if not places:
        return None
    return (min(p[0] for p in places), min(p[1] for p in places),
            max(p[0] for p in places), max(p[1] for p in places))
