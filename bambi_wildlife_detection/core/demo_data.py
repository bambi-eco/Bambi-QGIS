# -*- coding: utf-8 -*-
"""A public example flight, to try everything with.

Nobody learns a pipeline from its settings. The BAMBI project publishes its
raw recordings - the videos as they came off the drone, with their flight
logs and calibrations - and one of the shortest of them is small enough to
fetch at a click: flight 306, a nadir flight of a DJI M30T over forest and
meadow in Styria on 19 November 2024, thermal and RGB, 122 MB.

**What is fetched.** One ZIP archive from Zenodo, named by what it is - its
size and SHA-256, as this release was tested with it. It arrives under
``<name>.part``, is continued where a download broke off, and is taken for
the flight only when it is the pinned file (:mod:`core.checked_download`).

**What is unpacked.** The archive is flat - two videos, their SRT files,
the AirData log and the two calibrations. Members that would land outside
the folder, and an archive that unpacks to far more than it should, are
refused.

Free of QGIS and Qt; ``requests`` is passed in as the session to use.
"""

import csv
import hashlib
import os
import zipfile
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

from . import checked_download, quiet


class DemoFlight(NamedTuple):
    """One published flight, as this release was tested with it."""

    key: str
    #: What the flight is called in the project.
    name: str
    #: The archive, where it is, how large and which file exactly.
    filename: str
    url: str
    size: int
    sha256: str
    #: Where it was published, and under which terms.
    doi: str
    licence: str
    citation: str
    #: What a user is told it is.
    description: str


DEMO = DemoFlight(
    key="bambi_flight_306",
    name="BAMBI demo flight 306",
    filename="flight_306.zip",
    url=("https://zenodo.org/api/records/19047570/files/flight_306.zip/"
         "content"),
    size=121842908,
    sha256="3743c9dedb9cdf2f0377a0c040f087721e0155aecb2707fa5725170282b5cb91",
    doi="10.5281/zenodo.19047570",
    licence="CC BY 4.0",
    citation=(
        "Praschl, C., Schedl, D. C., et al. (2026): The BAMBI Raw Dataset: "
        "Unprocessed Nadir UAV-Recordings of Forest Wildlife (Part 27). "
        "Zenodo. https://doi.org/10.5281/zenodo.19047570 (CC BY 4.0)"),
    description=(
        "A nadir flight of a DJI M30T over forest and meadow in Styria, "
        "Austria, on 19 November 2024: thermal and RGB video with their "
        "SRT files, the AirData flight log and both camera calibrations."),
)

#: The folder a demo flight is put in, inside the one the user chose.
FOLDER_NAME = "BAMBI_demo_flight"
#: An archive that unpacks to more than this many times its own size is not
#: the flight.
MAX_EXPANSION = 20


class DemoError(RuntimeError):
    """The demo flight could not be made ready."""


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


def archive_problem(path: str, demo: DemoFlight = DEMO) -> Optional[str]:
    """Why the file at *path* is not the demo's archive; ``None`` when it
    is."""
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return f"it cannot be read ({exc})"
    if size != demo.size:
        return f"it is {size} bytes long, not {demo.size}"
    found = sha256_of(path)
    if found != demo.sha256:
        return ("its contents are not the published ones (SHA-256 "
                f"{found[:12]}... instead of {demo.sha256[:12]}...)")
    return None


def download(folder: str, session, demo: DemoFlight = DEMO,
             progress: Optional[Callable[[int], None]] = None,
             cancelled: Optional[Callable[[], bool]] = None,
             log: Optional[Callable[[str], None]] = None, **fetching) -> str:
    """Fetch the demo's archive into *folder*; returns its path.

    An archive that is already there and is the pinned file is used as it
    is; one that is not is fetched again. A download that broke off is
    continued.

    :param session: something with ``get(url, stream=True, ...)`` like a
        ``requests.Session``
    :raises DemoError: the archive could not be fetched, or what arrived is
        not the published file
    :raises core.checked_download.Cancelled: stopped by the user (what
        arrived is kept)
    """
    say = log or (lambda message: None)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, demo.filename)
    if os.path.isfile(path):
        problem = archive_problem(path, demo)
        if problem is None:
            say(f"{demo.filename} is already there")
            if progress:
                progress(100)
            return path
        say(f"The {demo.filename} that is there cannot be used - {problem} "
            "- and is downloaded again")
        os.remove(path)
    say(f"Downloading {demo.filename} ({demo.size / 1e6:.0f} MB) from "
        "Zenodo...")
    try:
        checked_download.fetch(
            session, demo.url, path,
            check=lambda part: archive_problem(part, demo),
            cancelled=cancelled, progress=progress, log=log, **fetching)
    except checked_download.Cancelled:
        raise
    except checked_download.NotThere as exc:
        raise DemoError(
            f"The demo flight is not where it was published ({demo.url}). "
            "Look for a newer version of the plugin.") from exc
    except checked_download.DownloadError as exc:
        raise DemoError(str(exc)) from exc
    return path


def extract(archive: str, folder: str,
            log: Optional[Callable[[str], None]] = None) -> List[str]:
    """Unpack the archive into *folder*; returns the files written.

    :raises DemoError: an archive that is none, a member that would land
        outside *folder*, or contents far larger than the archive
    """
    target = os.path.realpath(folder)
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = [info for info in bundle.infolist()
                       if not info.is_dir()]
            total = sum(info.file_size for info in members)
            limit = MAX_EXPANSION * max(os.path.getsize(archive), 1)
            if total > limit:
                raise DemoError(
                    f"{os.path.basename(archive)} unpacks to "
                    f"{total / 1e6:.0f} MB, which a flight of this size "
                    "does not: it is not unpacked.")
            for info in members:
                landing = os.path.realpath(
                    os.path.join(target, info.filename))
                if os.path.commonpath([target, landing]) != target:
                    raise DemoError(
                        f"{os.path.basename(archive)} holds a file that "
                        f"would be written outside the folder "
                        f"({info.filename}): it is not unpacked.")
            written = []
            os.makedirs(target, exist_ok=True)
            for info in members:
                written.append(bundle.extract(info, target))
    except zipfile.BadZipFile as exc:
        raise DemoError(
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


def prepare(folder: str, session, demo: DemoFlight = DEMO,
            progress: Optional[Callable[[int], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None,
            log: Optional[Callable[[str], None]] = None,
            keep_archive: bool = False, **fetching) -> Dict[str, object]:
    """Make the demo flight ready in *folder*: fetch its archive, unpack
    it, and say what is there (:func:`flight_files`).

    A folder that already holds the flight is taken as it is. The archive
    is removed once it is unpacked unless *keep_archive* says otherwise.
    """
    files = flight_files(folder)
    if is_complete(files) and files["rgb_videos"]:
        if log:
            log(f"The demo flight is already in {folder}")
        if progress:
            progress(100)
        return files

    def scaled(value: int) -> None:
        if progress:
            progress(int(value * 0.9))

    archive = download(folder, session, demo, scaled, cancelled, log,
                       **fetching)
    if cancelled and cancelled():
        raise checked_download.Cancelled("cancelled")
    if log:
        log("Unpacking...")
    extract(archive, folder, log)
    if not keep_archive:
        try:
            os.remove(archive)
        except OSError:
            quiet.detail(f"{archive} could not be removed: it is left "
                         "where it is, in the way of nothing")
    files = flight_files(folder)
    if not is_complete(files):
        raise DemoError(
            f"{demo.filename} was unpacked into {folder}, and what is "
            "there is not a flight: a thermal video with its SRT file and "
            "the flight log are needed.")
    if progress:
        progress(100)
    return files
