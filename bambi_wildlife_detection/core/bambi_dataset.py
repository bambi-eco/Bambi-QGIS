# -*- coding: utf-8 -*-
"""Flights of the BAMBI dataset, by their number.

The BAMBI project publishes its flights on Zenodo, in two forms that share
the flights' numbers:

* the **raw dataset** - ``flight_<n>.zip`` with the videos as they came off
  the drone, their SRT files, the AirData flight log and the two camera
  calibrations: what this plugin's pipeline starts from. 400 flights, a
  hundred megabytes to nineteen gigabytes each;
* the **labelled dataset** - per flight a video that is already undistorted
  and synchronised, and beside it the *poses* of its frames, the animals
  *labelled* by hand on key frames, and the *correction* of the camera's
  position and angle that was found for the flight. 386 of the flights.

Which flights there are, and where each one lies, is the dataset's own
business: its repository (github.com/bambi-eco/Dataset) keeps a summary per
version - ``flight_metadata/zenodo_upload_summary.json`` for the labelled
flights, ``..._raw.json`` for the recordings - and a script,
``download_from_zenodo.py``, that downloads by it. This module reads those
summaries where the script reads them (:func:`fetch_catalogue`) and fetches
as the script does: Zenodo is asked for a record's files, a flight's
recordings are its whole archive (:func:`fetch_recordings`), and "the
annotations only" are every file of the labelled archive that is no video
and no picture, read out of it in place (:func:`fetch_published`). Nothing
about the dataset is kept in the plugin, so a new part or a corrected label
arrives without a new version of it.

What was published is then fitted to the flight as the plugin has it:

**Poses** (:func:`flight_poses`). Published poses say where a frame was
taken as latitude, longitude and height above sea level, and relative to an
origin of the dataset's own. The plugin's poses are relative to the origin
of the flight's elevation model, in the project's CRS.

**Correction** (:func:`flight_correction`). The published correction was
found while a frame's heading - measured from true north - was used as a
bearing on the map, so it contains the angle between the two
(``core.grid_north``). The plugin takes that angle from the heading itself,
so it is taken out of the correction: the same camera, said the plugin's
way.

**Labels** (:func:`label_tracks`). A label names a frame of the published
video by its number and a box by its pixels in that video. The plugin's own
frames of the same flight are not numbered the same - the published video
lacks frames the two cameras did not share - and are not the same picture:
the same view, cut closer. So a frame is found by the moment it was taken
(:func:`frame_mapping`), and a box is scaled about the middle of the
picture by the ratio of the two focal lengths (:func:`pixel_transform`).
On flight 306 that put the published boxes on the plugin's frames to
within half a pixel.

Free of QGIS and Qt. ``requests`` is passed in as the session to use;
pyproj is imported where it is needed.
"""

import csv
import hashlib
import io
import json
import math
import os
import re
from datetime import datetime
from typing import (Callable, Dict, Iterable, List, NamedTuple, Optional,
                    Sequence, Tuple)

from . import checked_download, flight_archive, quiet, remote_zip

ZENODO = "https://zenodo.org"
#: The dataset's repository: where its summaries are read from.
REPOSITORY = "https://raw.githubusercontent.com/bambi-eco/Dataset/main"
#: The summaries ``download_from_zenodo.py`` downloads by, per version of
#: the dataset: the labelled flights (its ``base``) and the recordings.
LABELLED, RAW = "labelled", "raw"
SUMMARIES = {
    LABELLED: f"{REPOSITORY}/flight_metadata/zenodo_upload_summary.json",
    RAW: f"{REPOSITORY}/flight_metadata/zenodo_upload_summary_raw.json",
}
SPLITS = f"{REPOSITORY}/flight_metadata/splits.json"
#: As the script names a flight's archive, unless a part says otherwise.
ZIP_PREFIX = "flight_"
#: What "annotations only" leaves on Zenodo, as the script has it: anything
#: that is imagery.
MEDIA_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".png", ".jpg",
                            ".jpeg", ".tif", ".tiff", ".bmp"})

#: The flight that is ticked when the list opens: one of the shortest.
EXAMPLE = 306
#: While a flight is unpacked, its archive and its files are both there.
SPACE_FACTOR = 2.0

#: The folder inside a flight's target folder that holds what was fetched
#: of the labelled dataset, and what was done with it.
FOLDER = "bambi_dataset"
STATE_FILE = "dataset.json"

#: The small files of a flight in the labelled dataset: what they are
#: called here, and in the archive (after the flight's number).
POSES, LABELS, CORRECTION, METADATA, CLASSES = (
    "poses", "labels", "correction", "metadata", "classes")
PUBLISHED_FILES = {
    POSES: "{id}_matched_poses.json",
    LABELS: "{id}_gt.txt",
    CORRECTION: "{id}_correction.json",
    METADATA: "{id}_metadata.json",
    CLASSES: "{id}_track_mapping.json",
}
#: None of them is larger than this.
MAX_PUBLISHED_BYTES = 64 << 20

#: The thermal picture of the published video: its left half, in pixels.
PUBLISHED_SIZE = (1024, 1024)
#: Two frames are the same moment when their times are no further apart
#: than this (a frame lasts 33 ms).
SAME_MOMENT_S = 0.005

#: How the dataset writes sex and age, in this plugin's words - as its
#: documentation has it (docs/annotation-format.md: gender 0 = unknown,
#: 1 = male, 2 = female; age 0 = unknown, 1 = juvenile, 2 = adult).
SEX = {0: "unknown", 1: "male", 2: "female"}
AGE = {0: "unknown", 1: "juvenile", 2: "adult"}
#: The custom field an imported label track carries: the number the track
#: has in the dataset. It is how an earlier import is found again.
TRACK_FIELD = "bambi_track"

_SPECIES = re.compile(r"^\s*(?P<scientific>[^()]+?)\s*\((?P<common>[^()]+)\)\s*$")


class DatasetError(RuntimeError):
    """Something of the dataset could not be had, or is not what it should
    be."""


# ---------------------------------------------------------------------------
# The flights, as the dataset's repository lists them
# ---------------------------------------------------------------------------

class Archive(NamedTuple):
    """One archive on Zenodo. The summary says which record it is in;
    its size, checksum and link are Zenodo's, once asked
    (:func:`resolve`)."""

    record: int
    filename: str
    size: int = 0
    md5: str = ""
    link: str = ""

    @property
    def url(self) -> str:
        return self.link or (f"{ZENODO}/api/records/{self.record}/files/"
                             f"{self.filename}/content")

    @property
    def doi(self) -> str:
        return f"10.5281/zenodo.{self.record}"


class Flight(NamedTuple):
    """One flight of the dataset."""

    id: int
    #: The recordings, and the labelled flight - ``None`` where the dataset
    #: has none.
    raw: Optional[Archive]
    labelled: Optional[Archive]
    #: ``train``, ``val`` or ``test``; ``""`` for a flight in none.
    split: str = ""

    @property
    def name(self) -> str:
        """What the flight is called in a project."""
        return f"BAMBI flight {self.id}"

    @property
    def folder(self) -> str:
        """The folder a flight is put in, inside the one the user chose."""
        return f"flight_{self.id}"


def _parts(summary, what: str) -> list:
    if not isinstance(summary, list) or not summary:
        raise DatasetError(
            f"The dataset's summary of its {what} flights is empty or not "
            "what it used to be.")
    return summary


def catalogue_from(labelled, raw, splits=None) -> dict:
    """The flights two upload summaries list - laid out as
    ``download_from_zenodo.py`` reads them: per part a ``deposition_id``
    and the ``flights`` in it, each in ``<zip_prefix><flight>.zip``.

    :return: ``{"flights": {number: Flight}, "records": {record: {"kind",
        "title", "doi"}}}``
    :raises DatasetError: a summary is not one
    """
    archives: Dict[str, Dict[int, Archive]] = {LABELLED: {}, RAW: {}}
    records: Dict[int, dict] = {}
    try:
        for kind, summary in ((LABELLED, labelled), (RAW, raw)):
            for part in _parts(summary, kind):
                record = int(part["deposition_id"])
                records[record] = {"kind": kind,
                                   "title": str(part.get("title") or ""),
                                   "doi": str(part.get("doi") or "")}
                prefix = part.get("zip_prefix", ZIP_PREFIX)
                for number in part["flights"]:
                    if not str(number).isdigit():
                        continue
                    archives[kind][int(number)] = Archive(
                        record, f"{prefix}{number}.zip")
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetError(
            "The dataset's summary of its flights is not what it used to "
            f"be ({exc!r}). Look for a newer version of the plugin.") from exc
    split_of: Dict[int, str] = {}
    for name, numbers in (splits or {}).items() if isinstance(
            splits, dict) else ():
        for number in numbers or []:
            if str(number).isdigit():
                split_of[int(number)] = str(name)
    known = {number: Flight(number, archives[RAW].get(number),
                            archives[LABELLED].get(number),
                            split_of.get(number, ""))
             for number in set(archives[RAW]) | set(archives[LABELLED])}
    return {"flights": known, "records": records}


def _get_json(session, url: str, timeout: float):
    try:
        response = session.get(url, timeout=timeout)
        if response.status_code != 200:
            raise DatasetError(
                f"{url} could not be read (HTTP {response.status_code}).")
        return response.json()
    except DatasetError:
        raise
    except Exception as exc:  # noqa: BLE001 - whatever the connection says
        raise DatasetError(f"{url} could not be read ({exc}).") from exc


def fetch_catalogue(session, timeout: float = 30.0) -> dict:
    """The dataset's flights, read from its repository now
    (:func:`catalogue_from`). The splits are an extra: without them the
    flights are listed all the same.

    :raises DatasetError: the repository could not be read
    """
    labelled = _get_json(session, SUMMARIES[LABELLED], timeout)
    raw = _get_json(session, SUMMARIES[RAW], timeout)
    try:
        splits = _get_json(session, SPLITS, timeout)
    except DatasetError:
        quiet.detail("The dataset's splits could not be read")
        splits = None
    return catalogue_from(labelled, raw, splits)


def flights_of(known: dict) -> List[Flight]:
    """Every flight of a catalogue, by its number."""
    return [known["flights"][number] for number in sorted(known["flights"])]


def deposition_files(session, record: int, timeout: float = 60.0
                     ) -> Dict[str, Tuple[str, int, str]]:
    """What Zenodo says a record holds: ``{file name: (link, size,
    md5)}`` - asked as ``download_from_zenodo.py`` asks.

    :raises DatasetError: Zenodo could not be asked
    """
    data = _get_json(session, f"{ZENODO}/api/records/{int(record)}", timeout)
    files: Dict[str, Tuple[str, int, str]] = {}
    for entry in (data.get("files") or []) if isinstance(data, dict) else []:
        name = entry.get("filename") or entry.get("key")
        links = entry.get("links") or {}
        link = links.get("download") or links.get("self")
        if not name or not link:
            continue
        checksum = str(entry.get("checksum") or "")
        md5 = checksum[4:] if checksum.startswith("md5:") else ""
        files[str(name)] = (str(link), int(entry.get("size") or 0), md5)
    return files


def resolve(archive: Archive, session,
            listings: Optional[Dict[int, dict]] = None) -> Archive:
    """An archive with its link, size and checksum as Zenodo gives them.

    :param listings: ``{record: its files}`` kept between calls, so that a
        record is asked once for all its flights
    :raises DatasetError: Zenodo could not be asked, or the record does
        not hold the archive
    """
    if archive.link:
        return archive
    listings = {} if listings is None else listings
    if archive.record not in listings:
        listings[archive.record] = deposition_files(session, archive.record)
    found = listings[archive.record].get(archive.filename)
    if found is None:
        raise DatasetError(
            f"{archive.filename} is not in the Zenodo record the dataset "
            f"names for it ({archive.doi}).")
    link, size, md5 = found
    return archive._replace(link=link, size=size, md5=md5)


def common_name(species: str) -> str:
    """``"roe deer"`` for ``"Capreolus capreolus (Roe deer)"`` - what the
    plugin calls a species; a name without brackets, lower-cased."""
    found = _SPECIES.match(species or "")
    return (found.group("common") if found else (species or "")).strip().lower()


def scientific_name(species: str) -> str:
    """``"Capreolus capreolus"`` for ``"Capreolus capreolus (Roe deer)"``,
    ``""`` for a name without one."""
    found = _SPECIES.match(species or "")
    return found.group("scientific").strip() if found else ""


# ---------------------------------------------------------------------------
# The recordings
# ---------------------------------------------------------------------------

def md5_of(path: str, cancelled: Optional[Callable[[], bool]] = None) -> str:
    digest = hashlib.md5()  # nosec B324 - Zenodo's checksum of a transfer
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            if cancelled and cancelled():
                raise checked_download.Cancelled("cancelled")
            digest.update(block)
    return digest.hexdigest()


def archive_problem(path: str, archive: Archive,
                    cancelled: Optional[Callable[[], bool]] = None
                    ) -> Optional[str]:
    """Why the file at *path* is not the published archive; ``None`` when
    it is: as long as Zenodo says, and with the checksum Zenodo gives -
    each where Zenodo gave one."""
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return f"it cannot be read ({exc})"
    if archive.size and size != archive.size:
        return f"it is {size} bytes long, not {archive.size}"
    if archive.md5:
        found = md5_of(path, cancelled)
        if found != archive.md5:
            return ("its contents are not the published ones (MD5 "
                    f"{found[:12]}... instead of {archive.md5[:12]}...)")
    return None


def space_problem(folder: str, archive: Archive, there: int = 0,
                  free: Optional[int] = None) -> Optional[str]:
    """Why a flight does not fit where it should go; ``None`` when it
    does, or when nobody can say how much room there is.

    :param there: bytes of the archive that have arrived already
    :param free: the free bytes, for a caller that knows them
    """
    if free is None:
        import shutil

        place = folder
        while place and not os.path.isdir(place):
            parent = os.path.dirname(place)
            if parent == place:
                break
            place = parent
        try:
            free = shutil.disk_usage(place or ".").free
        except OSError:
            return None
    needed = int(SPACE_FACTOR * archive.size) - int(there)
    if free >= needed:
        return None
    return (f"{_megabytes(needed)} are needed in {folder} - the archive, "
            f"and what is unpacked from it - and {_megabytes(free)} are "
            "free")


def fetch_recordings(entry: Flight, folder: str, session,
                     progress: Optional[Callable[[int], None]] = None,
                     cancelled: Optional[Callable[[], bool]] = None,
                     log: Optional[Callable[[str], None]] = None,
                     keep_archive: bool = False,
                     listings: Optional[Dict[int, dict]] = None, **fetching
                     ) -> Dict[str, object]:
    """Make a flight's recordings ready in *folder*: fetch the archive,
    check it, unpack it - and say what is there
    (:func:`core.flight_archive.flight_files`).

    A folder that already holds the flight is taken as it is; an archive
    that is there and is the published file is not fetched again; a
    download that broke off is continued.

    :raises DatasetError: the dataset has no recordings of the flight, the
        archive could not be fetched, or it is not the published file
    :raises core.checked_download.Cancelled: stopped by the user (what
        arrived is kept)
    """
    say = log or (lambda message: None)
    files = flight_archive.flight_files(folder)
    if flight_archive.is_complete(files):
        say(f"Flight {entry.id} is already in {folder}")
        if progress:
            progress(100)
        return files
    if entry.raw is None:
        raise DatasetError(
            f"The BAMBI raw dataset has no recordings of flight {entry.id}.")
    archive = resolve(entry.raw, session, listings)
    _archive_into(entry, archive, folder, session, archive.filename,
                  progress, cancelled, log, keep_archive, **fetching)
    files = flight_archive.flight_files(folder)
    if not flight_archive.is_complete(files):
        raise DatasetError(
            f"{archive.filename} was unpacked into {folder}, and what is "
            "there is not a flight: a thermal video with its SRT file and "
            "the flight log are needed.")
    if progress:
        progress(100)
    return files


def _archive_into(entry: Flight, archive: Archive, folder: str, session,
                  saved_as: str, progress, cancelled, log, keep_archive,
                  **fetching) -> None:
    """Fetch an archive into *folder*, check it and unpack it there."""
    say = log or (lambda message: None)

    def scaled(value: int) -> None:
        if progress:
            progress(int(value * 0.9))

    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, saved_as)
    have = False
    if os.path.isfile(path):
        problem = archive_problem(path, archive, cancelled)
        if problem is None:
            say(f"{archive.filename} is already there")
            have = True
        else:
            say(f"The {archive.filename} that is there cannot be used - "
                f"{problem} - and is downloaded again")
            os.remove(path)
    if not have:
        arrived = 0
        if os.path.isfile(path + ".part"):
            arrived = os.path.getsize(path + ".part")
        tight = space_problem(folder, archive, arrived)
        if tight:
            raise DatasetError(
                f"Flight {entry.id} does not fit: {tight}.")
        say(f"Downloading {archive.filename} "
            f"({_megabytes(archive.size)}) from Zenodo...")
        try:
            checked_download.fetch(
                session, archive.url, path,
                check=lambda part: archive_problem(part, archive, cancelled),
                cancelled=cancelled, progress=scaled, log=log, **fetching)
        except checked_download.Cancelled:
            raise
        except checked_download.NotThere as exc:
            raise DatasetError(
                f"Flight {entry.id} is not where it was published "
                f"({archive.url}). Look for a newer version of the "
                "plugin.") from exc
        except checked_download.DownloadError as exc:
            raise DatasetError(str(exc)) from exc
    if cancelled and cancelled():
        raise checked_download.Cancelled("cancelled")
    say("Unpacking...")
    try:
        flight_archive.extract(path, folder, log)
    except flight_archive.ArchiveError as exc:
        raise DatasetError(str(exc)) from exc
    if not keep_archive:
        try:
            os.remove(path)
        except OSError:
            quiet.detail(f"{path} could not be removed: it is left where "
                         "it is, in the way of nothing")


#: The dataset's two versions of a flight's video: the labelled dataset's -
#: undistorted, synchronised, thermal and RGB side by side - and the raw
#: dataset's, as it came off the drone (``download_from_zenodo.py
#: --version base`` and ``raw``).
PROCESSED, RECORDINGS = "processed", "raw"
VERSIONS = (PROCESSED, RECORDINGS)
PROCESSED_VIDEO = "{id}_matched_processed.mp4"


def processed_video(folder: str, flight_id: int) -> str:
    """Where a flight's processed video is, once fetched."""
    return os.path.join(folder, PROCESSED_VIDEO.format(id=int(flight_id)))


def fetch_processed(entry: Flight, folder: str, target_folder: str, session,
                    progress: Optional[Callable[[int], None]] = None,
                    cancelled: Optional[Callable[[], bool]] = None,
                    log: Optional[Callable[[str], None]] = None,
                    keep_archive: bool = False,
                    listings: Optional[Dict[int, dict]] = None, **fetching
                    ) -> str:
    """Have the labelled flight whole in *folder* - its processed video,
    its masks, its annotations - as ``download_from_zenodo.py --unzip``
    leaves it, and its annotations where :func:`fetch_published` would
    have put them. Returns the video's path.

    A folder that holds the video already is taken as it is.

    :raises DatasetError: the dataset has no labelled version of the
        flight, or its archive could not be fetched
    :raises core.checked_download.Cancelled: stopped by the user
    """
    video = processed_video(folder, entry.id)
    if os.path.isfile(video):
        if log:
            log(f"The processed video of flight {entry.id} is already in "
                f"{folder}")
    else:
        if entry.labelled is None:
            raise DatasetError(
                f"The labelled BAMBI dataset has no flight {entry.id}: "
                "there is no processed video of it.")
        archive = resolve(entry.labelled, session, listings)
        # the recordings' archive has the same name
        _archive_into(entry, archive, folder, session,
                      f"processed_{archive.filename}", progress, cancelled,
                      log, keep_archive, **fetching)
        if not os.path.isfile(video):
            raise DatasetError(
                f"{archive.filename} was unpacked into {folder}, and holds "
                f"no {os.path.basename(video)}.")
    # the annotations, where the rest of this module looks for them
    import shutil

    os.makedirs(published_folder(target_folder), exist_ok=True)
    for kind, name in PUBLISHED_FILES.items():
        source = os.path.join(folder, name.format(id=entry.id))
        there = published_path(target_folder, entry.id, kind)
        if os.path.isfile(source) and not os.path.isfile(there):
            shutil.copyfile(source, there)
    if progress:
        progress(100)
    return video


def _megabytes(size: int) -> str:
    return f"{size / 1e9:.1f} GB" if size >= 1e9 else f"{size / 1e6:.0f} MB"


# ---------------------------------------------------------------------------
# What is published beside the labelled video
# ---------------------------------------------------------------------------

def published_folder(target_folder: str) -> str:
    return os.path.join(target_folder, FOLDER)


def published_path(target_folder: str, flight_id: int, kind: str) -> str:
    """Where a published file of a flight is kept."""
    return os.path.join(published_folder(target_folder),
                        PUBLISHED_FILES[kind].format(id=int(flight_id)))


def _published(target_folder: str, flight_id: int) -> Dict[str, str]:
    """``{kind: path}`` of the published files a flight's folder has."""
    there = {kind: published_path(target_folder, flight_id, kind)
             for kind in PUBLISHED_FILES}
    return {kind: path for kind, path in there.items()
            if os.path.isfile(path)}


def fetch_published(entry: Flight, target_folder: str, session,
                    cancelled: Optional[Callable[[], bool]] = None,
                    log: Optional[Callable[[str], None]] = None,
                    listings: Optional[Dict[int, dict]] = None
                    ) -> Dict[str, str]:
    """Have the annotations of a labelled flight in the flight's target
    folder: ``{kind: path}`` of those this module knows by name.

    As ``download_from_zenodo.py --annotations-only`` does it: every file
    of the archive that is no video and no picture is read out of it in
    place - a megabyte or two out of an archive of several hundred
    (:mod:`core.remote_zip`) - and a flight whose poses and metadata are
    there is not fetched again.

    :raises DatasetError: the dataset has no labelled version of the
        flight, or its archive could not be read
    :raises core.remote_zip.Cancelled: stopped by the user
    """
    there = _published(target_folder, entry.id)
    if POSES in there and METADATA in there:
        return there
    if entry.labelled is None:
        raise DatasetError(
            f"The labelled BAMBI dataset has no flight {entry.id}: "
            "neither poses nor labels were published for it.")
    archive = resolve(entry.labelled, session, listings)
    try:
        members = remote_zip.list_members(
            session, archive.url, size=archive.size or None,
            cancelled=cancelled)
        wanted = sorted({
            os.path.basename(name) for name, _size in members
            if os.path.splitext(name)[1].lower() not in MEDIA_SUFFIXES})
        found = remote_zip.read_members(
            session, archive.url, wanted, size=archive.size or None,
            max_bytes=MAX_PUBLISHED_BYTES, cancelled=cancelled, log=log)
    except remote_zip.Cancelled:
        raise
    except remote_zip.NotThere as exc:
        raise DatasetError(
            f"The labelled flight {entry.id} is not where it was "
            f"published ({archive.url}).") from exc
    except remote_zip.RemoteArchiveError as exc:
        raise DatasetError(
            f"The published files of flight {entry.id} could not be "
            f"read: {exc}") from exc
    folder = published_folder(target_folder)
    os.makedirs(folder, exist_ok=True)
    for name, data in found.items():
        path = os.path.join(folder, os.path.basename(name))
        temporary = path + ".part"
        with open(temporary, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    return _published(target_folder, entry.id)


def read_state(target_folder: str) -> dict:
    """What a flight's target folder says was taken from the dataset:
    ``{}`` for a flight that is not of it."""
    path = os.path.join(published_folder(target_folder), STATE_FILE)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def write_state(target_folder: str, **changes) -> dict:
    """Note in a flight's target folder what was taken from the dataset."""
    state = read_state(target_folder)
    state.update(changes)
    os.makedirs(published_folder(target_folder), exist_ok=True)
    path = os.path.join(published_folder(target_folder), STATE_FILE)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    os.replace(temporary, path)
    return state


# ---------------------------------------------------------------------------
# Poses
# ---------------------------------------------------------------------------

def read_poses(path: str) -> dict:
    """A published poses file.

    :raises DatasetError: it is not one
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            poses = json.load(handle)
        images = poses["images"]
        if not isinstance(images, list) or not images:
            raise ValueError("it lists no frames")
        for image in (images[0], images[-1]):
            float(image["lat"])
            float(image["lng"])
            east, north, up = image["location"][:3]
            float(east), float(north), float(up)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise DatasetError(
            f"{os.path.basename(path)} is not a poses file of the BAMBI "
            f"dataset ({exc}).") from exc
    return poses


def _origin(document: dict) -> Tuple[float, float, float]:
    origin = document.get("origin") or {}
    return (float(origin.get("latitude", 0.0)),
            float(origin.get("longitude", 0.0)),
            float(origin.get("altitude", 0.0) or 0.0))


def flight_poses(published: dict, origin_wgs84: dict, epsg: int,
                 source: Optional[dict] = None) -> dict:
    """Published poses as the plugin's own: every frame's place relative
    to the origin of the flight's elevation model, in metres of the CRS
    *epsg*.

    The place is worked out from the frame's own latitude and longitude;
    its height is the published one, which is above sea level once the
    height of the dataset's origin is added. Angles, field of view and the
    time a frame was taken are the published ones.

    :param origin_wgs84: ``{"latitude", "longitude", "altitude"}`` of the
        elevation model's origin (its metadata's ``origin_wgs84``)
    :param source: said in the result under ``"source"`` - where the poses
        are from
    """
    from pyproj import Transformer

    to_map = Transformer.from_crs(4326, int(epsg), always_xy=True)
    east0, north0 = to_map.transform(float(origin_wgs84["longitude"]),
                                     float(origin_wgs84["latitude"]))
    up0 = float(origin_wgs84.get("altitude", 0.0) or 0.0)
    _lat, _lon, published_up = _origin(published)

    images = []
    for image in published["images"]:
        east, north = to_map.transform(float(image["lng"]),
                                       float(image["lat"]))
        height = float(image["location"][2]) + published_up
        entry = dict(image)
        entry["location"] = [east - east0, north - north0, height - up0]
        images.append(entry)
    poses = {key: value for key, value in published.items()
             if key not in ("images", "origin")}
    poses["images"] = images
    poses["origin"] = {"latitude": float(origin_wgs84["latitude"]),
                       "longitude": float(origin_wgs84["longitude"]),
                       "altitude": up0}
    poses["source"] = dict(source or {}, frames="published")
    return poses


def write_poses(target_folder: str, poses: dict, modality: str = "t") -> str:
    """Write poses as a flight's own (``poses_t.json``); returns the
    path."""
    os.makedirs(target_folder, exist_ok=True)
    path = os.path.join(target_folder, f"poses_{modality}.json")
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(poses, handle)
    os.replace(temporary, path)
    return path


def are_published(poses: dict) -> bool:
    """Whether poses are the dataset's own frame list - not yet replaced
    by an extraction of the plugin's."""
    source = (poses or {}).get("source")
    return isinstance(source, dict) and source.get("frames") == "published"


# ---------------------------------------------------------------------------
# The correction
# ---------------------------------------------------------------------------

def _xyz(values) -> Dict[str, float]:
    values = values or {}
    return {axis: float(values.get(axis, 0.0) or 0.0) for axis in "xyz"}


def flight_correction(published: dict, convergence_deg: float = 0.0,
                      mapping: Optional[Dict[int, int]] = None) -> dict:
    """A published correction in the plugin's form, with the angle between
    the map's north and true north taken out of its rotation.

    The dataset found its corrections while headings were used as bearings
    on the map: the rotation about the vertical then holds the grid
    convergence besides the compass's own error. The plugin's camera takes
    the convergence from the heading, so here it leaves the correction -
    ``z - convergence`` - and the camera looks where it did.

    Ranges of frames that say nothing but what holds for the whole flight
    are left out. The others name frames of the published video: with
    *mapping* (:func:`frame_mapping`) they are renumbered to the flight's
    own frames, and a range none of whose frames the flight has is left
    out.

    :param convergence_deg: degrees by which the map's north lies east of
        true north at the flight (``core.grid_north``); 0 leaves the
        rotation as published
    """
    turn = math.radians(float(convergence_deg or 0.0))

    def rotation(values) -> Dict[str, float]:
        result = _xyz(values)
        result["z"] -= turn
        return result

    whole = {"translation": _xyz(published.get("translation")),
             "rotation": rotation(published.get("rotation"))}
    ranges = []
    for entry in published.get("fine_corrections") or []:
        part = {
            "start": int(entry.get("start_frame", entry.get("start", 0))),
            "end": int(entry.get("end_frame", entry.get("end", 0))),
            "translation": _xyz(entry.get(
                "translation", published.get("translation"))),
            "rotation": rotation(entry.get(
                "rotation", published.get("rotation"))),
        }
        same = all(
            math.isclose(part[what][axis], whole[what][axis], abs_tol=1e-9)
            for what in ("translation", "rotation") for axis in "xyz")
        if same:
            continue
        if mapping is not None:
            own = [mapping[number]
                   for number in range(part["start"], part["end"] + 1)
                   if number in mapping]
            if not own:
                continue
            part["start"], part["end"] = min(own), max(own)
        ranges.append(part)
    return dict(whole, additional=ranges)


def read_correction(path: str) -> dict:
    """A published correction file.

    :raises DatasetError: it is not one
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            correction = json.load(handle)
        if not isinstance(correction, dict):
            raise ValueError("it is no JSON object")
        _xyz(correction.get("translation"))
        _xyz(correction.get("rotation"))
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise DatasetError(
            f"{os.path.basename(path)} is not a correction file of the "
            f"BAMBI dataset ({exc}).") from exc
    return correction


def write_correction(folder: str, correction: dict,
                     name: str = "correction.json") -> str:
    """Write a correction in the plugin's form; returns the path."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(correction, handle, indent=4)
    os.replace(temporary, path)
    return path


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

class Label(NamedTuple):
    """One labelled box: an animal on a key frame of the published video."""

    frame: int
    track: int
    x1: float
    y1: float
    x2: float
    y2: float
    species: str
    sex: str
    age: str
    occluded: bool


def _code(value: str, words: Dict[int, str]) -> str:
    try:
        return words.get(int(float(value)), "unknown")
    except (TypeError, ValueError):
        return "unknown"


def parse_labels(text: str) -> List[Label]:
    """The rows of a published label file: ``frame, track, left, top,
    width, height, confidence, class, visibility, species, sex, age,
    propagated``. A visibility below 1 is an animal partly hidden.

    Rows that cannot be read are left out; class numbers are not used -
    they mean something else in every flight.
    """
    labels: List[Label] = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 6 or not row[0].strip() or row[0].lstrip().startswith("#"):
            continue
        try:
            frame, track = int(float(row[0])), int(float(row[1]))
            left, top = float(row[2]), float(row[3])
            width, height = float(row[4]), float(row[5])
            visibility = float(row[8]) if len(row) > 8 and row[8] else 1.0
        except ValueError:
            continue
        if width <= 0 or height <= 0 or frame < 0:
            continue
        labels.append(Label(
            frame=frame, track=track, x1=left, y1=top, x2=left + width,
            y2=top + height,
            species=row[9].strip() if len(row) > 9 else "",
            sex=_code(row[10], SEX) if len(row) > 10 else "unknown",
            age=_code(row[11], AGE) if len(row) > 11 else "unknown",
            occluded=visibility < 1.0))
    return labels


def read_labels(path: str) -> List[Label]:
    """A published label file (:func:`parse_labels`).

    :raises DatasetError: it cannot be read
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return parse_labels(handle.read())
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(
            f"{os.path.basename(path)} cannot be read ({exc}).") from exc


def _moment(image: dict) -> Optional[float]:
    """When a frame was taken, in seconds since 1970; ``None`` for a frame
    that does not say."""
    stamp = (image or {}).get("timestamp")
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(str(stamp)).timestamp()
    except ValueError:
        return None


def frame_mapping(published: dict, own: dict,
                  tolerance_s: float = SAME_MOMENT_S) -> Dict[int, int]:
    """Which frame of a flight's own poses each published frame is:
    ``{published number: own number}``, by the moment the frames were
    taken. A published frame the flight does not have - not extracted, or
    dropped - is not in the result.

    Poses that are the published ones themselves map onto themselves.
    """
    theirs = published.get("images") or []
    mine = own.get("images") or []
    if are_published(own) and len(mine) == len(theirs):
        return {index: index for index in range(len(theirs))}
    moments = sorted((moment, index) for index, moment
                     in ((index, _moment(image))
                         for index, image in enumerate(mine))
                     if moment is not None)
    if not moments:
        return {}
    import bisect

    times = [moment for moment, _index in moments]
    mapping: Dict[int, int] = {}
    for number, image in enumerate(theirs):
        moment = _moment(image)
        if moment is None:
            continue
        place = bisect.bisect_left(times, moment)
        best = None
        for candidate in (place - 1, place):
            if 0 <= candidate < len(times):
                apart = abs(times[candidate] - moment)
                if apart <= tolerance_s and (best is None or apart < best[0]):
                    best = (apart, moments[candidate][1])
        if best is not None:
            mapping[number] = best[1]
    return mapping


def _fovy(poses: dict) -> Optional[float]:
    for image in poses.get("images") or []:
        value = image.get("fovy")
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if 1.0 < value < 179.0:
            return value
    return None


def pixel_transform(published: dict, own: dict,
                    own_size: Tuple[int, int],
                    published_size: Tuple[int, int] = PUBLISHED_SIZE
                    ) -> Callable[[float, float], Tuple[float, float]]:
    """A function that puts a pixel of the published video's thermal
    picture onto a flight's own frame of the same moment.

    Both are the same undistorted view about the same middle, cut to
    another field of view: a pixel moves along the line from the middle,
    by the ratio of the two focal lengths.

    :param own_size: ``(width, height)`` of the flight's frames
    :raises DatasetError: one of the two does not say its field of view
    """
    theirs, mine = _fovy(published), _fovy(own)
    if theirs is None or mine is None:
        raise DatasetError(
            "The field of view of the frames is not known: the labels "
            "cannot be put on them.")
    focal_theirs = (published_size[1] / 2.0) / math.tan(math.radians(theirs) / 2.0)
    focal_mine = (own_size[1] / 2.0) / math.tan(math.radians(mine) / 2.0)
    scale = focal_mine / focal_theirs
    cx, cy = published_size[0] / 2.0, published_size[1] / 2.0
    ox, oy = own_size[0] / 2.0, own_size[1] / 2.0

    def transform(x: float, y: float) -> Tuple[float, float]:
        return ox + (x - cx) * scale, oy + (y - cy) * scale

    return transform


def label_tracks(labels: Sequence[Label], mapping: Dict[int, int],
                 transform: Callable[[float, float], Tuple[float, float]],
                 size: Tuple[int, int]) -> List[dict]:
    """The labels as tracks of key frames on a flight's own frames:
    ``[{"track", "species", "sex", "age", "keyframes": {frame: {"box":
    (x1, y1, x2, y2), "occluded"}}}]``, by the dataset's track number.

    A key frame the flight does not have is left out, and so is a box that
    lies wholly outside the flight's (smaller) picture; one that reaches
    over its edge is cut there. A track is of the species, sex and age
    most of its key frames carry - of those that say one: an animal that
    was recognised on one key frame is not unknown for the others.
    """
    width, height = size
    by_track: Dict[int, dict] = {}
    for label in labels:
        frame = mapping.get(label.frame)
        if frame is None:
            continue
        x1, y1 = transform(label.x1, label.y1)
        x2, y2 = transform(label.x2, label.y2)
        if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
            continue
        box = (max(0.0, x1), max(0.0, y1), min(float(width), x2),
               min(float(height), y2))
        if box[2] - box[0] < 1.0 or box[3] - box[1] < 1.0:
            continue
        track = by_track.setdefault(label.track, {
            "track": label.track, "keyframes": {}, "votes": {
                "species": {}, "sex": {}, "age": {}}})
        track["keyframes"][frame] = {"box": box, "occluded": label.occluded}
        for what in ("species", "sex", "age"):
            value = getattr(label, what)
            track["votes"][what][value] = track["votes"][what].get(value, 0) + 1
    tracks = []
    for number in sorted(by_track):
        track = by_track[number]
        votes = track.pop("votes")
        for what in ("species", "sex", "age"):
            # of the values that say something, the most frequent
            track[what] = max(
                votes[what], key=lambda value: (
                    value.strip().lower() not in ("", "unknown"),
                    votes[what][value], value))
        tracks.append(track)
    return tracks


# ---------------------------------------------------------------------------
# Into a flight
# ---------------------------------------------------------------------------

#: The dataset's two labels that are no species, in the plugin's words.
NO_SPECIES = {"unknown": "unknown", "no-animal": "not-an-animal"}
#: Inside the folder of a flight's recordings: where its results go (as
#: ``core.flight_files.detect_common_files`` has it).
TARGET_SUBFOLDER = "qgis"


def _rank(scientific: str) -> str:
    """The rank a scientific name is of, where its form says so: two words
    name a species, three a subspecies. A cross of two is left without."""
    words = scientific.split()
    if "x" in words:
        return ""
    return {2: "species", 3: "subspecies"}.get(len(words), "")


def ensure_species(target_folder: str, species: Iterable[str],
                   log: Optional[Callable[[str], None]] = None
                   ) -> Dict[str, str]:
    """Have every species the dataset names in *species* in the flight's
    own list: ``{the dataset's name: the flight's}``.

    A species is found by its scientific name first - a flight may call
    *Cervus elaphus* something else than "red deer" - then by its common
    name. One the flight's list does not have is added to it, with its
    scientific name. What the dataset calls "Unknown" is the plugin's
    ``unknown``, its "No-animal" the plugin's ``not-an-animal``.

    :raises DatasetError: the flight's list could not be changed
    """
    from .schema_editor import SchemaEditor, SchemaError

    names: Dict[str, str] = {}
    added: List[str] = []
    try:
        with SchemaEditor(target_folder) as editor:
            listed = editor.species()
            by_latin = {str(row["scientific_name"]).strip().lower(): row["name"]
                        for row in listed if row["scientific_name"]}
            by_name = {str(row["name"]).strip().lower(): row["name"]
                       for row in listed}
            for name in species:
                common, latin = common_name(name), scientific_name(name)
                if not latin and (not common or common in NO_SPECIES):
                    names[name] = NO_SPECIES.get(common, "unknown")
                elif latin and latin.lower() in by_latin:
                    names[name] = by_latin[latin.lower()]
                elif common in by_name:
                    names[name] = by_name[common]
                else:
                    species_id = editor.add_species(common)
                    if latin:
                        editor.set_taxonomy(species_id, latin, _rank(latin))
                        by_latin[latin.lower()] = common
                    by_name[common] = common
                    added.append(common)
                    names[name] = common
            if added:
                editor.commit()
    except SchemaError as exc:
        raise DatasetError(
            "The species of the labels could not be added to the flight's "
            f"list ({exc}).") from exc
    if added and log:
        log(f"Added to the flight's species: {', '.join(sorted(added))}")
    return names


def frame_size(target_folder: str, poses: dict, modality: str = "t"
               ) -> Tuple[int, int]:
    """``(width, height)`` of the frames a flight's poses are of: the
    published picture for published poses, else what the first frame on
    disk measures.

    :raises DatasetError: the flight's frames cannot be measured
    """
    if are_published(poses):
        return PUBLISHED_SIZE
    folder = os.path.join(target_folder, f"frames_{modality}")
    for image in poses.get("images") or []:
        path = os.path.join(folder, str(image.get("imagefile") or ""))
        if not os.path.isfile(path):
            continue
        try:
            from PIL import Image

            with Image.open(path) as picture:
                return int(picture.width), int(picture.height)
        except Exception:  # noqa: BLE001 - the next frame, then
            quiet.detail(f"{path} could not be measured")
    raise DatasetError(
        "The flight has no frames yet whose size could be read: extract "
        "the frames first.")


def frames_fingerprint(poses: dict) -> str:
    """What tells one list of frames from another: their names and the
    moments they were taken."""
    digest = hashlib.sha256()
    for image in (poses or {}).get("images") or []:
        digest.update(f"{image.get('imagefile', '')}|"
                      f"{image.get('timestamp', '')}\n".encode("utf-8"))
    return digest.hexdigest()[:16]


def import_labels(target_folder: str, tracks: Sequence[dict],
                  modality: str = "t",
                  log: Optional[Callable[[str], None]] = None) -> dict:
    """Put label tracks (:func:`label_tracks`) into a flight's labelling
    store, as hand labels, and make the boxes between their key frames -
    through the labelling tool's own code, so that they are what a label
    drawn there and added to the flight's detections is.

    What an earlier import put there (tracks that carry
    :data:`TRACK_FIELD`) is replaced, each track under the number it had;
    labels drawn by hand stay.

    :return: ``{"tracks", "key_frames", "boxes", "species"}``
    """
    from .labelling import OCCLUSION_LEVELS, CustomField, LabelStore, LabelTrack

    names = ensure_species(
        target_folder, sorted({track["species"] for track in tracks}), log)
    labels = LabelStore(target_folder, modality)
    labels.load()
    if not any(field.name == TRACK_FIELD and field.scope == "track"
               for field in labels.custom_fields):
        labels.custom_fields.append(CustomField(TRACK_FIELD, "int", "track"))
        labels.schema_defined = True
    earlier: Dict[int, int] = {}
    for track_id in sorted(labels.tracks):
        if TRACK_FIELD in labels.tracks[track_id].attributes:
            number = labels.tracks.pop(track_id).attributes[TRACK_FIELD]
            try:
                earlier.setdefault(int(number), track_id)
            except (TypeError, ValueError):
                continue
    taken = set(labels.tracks) | set(earlier.values())

    key_frames = 0
    for track in tracks:
        if not track["keyframes"]:
            continue
        track_id = earlier.get(int(track["track"]))
        if track_id is None or track_id in labels.tracks:
            track_id = max(taken, default=0) + 1
        taken.add(track_id)
        entry = LabelTrack(
            track_id, species=names[track["species"]], sex=track["sex"],
            age=track["age"], attributes={TRACK_FIELD: int(track["track"])})
        for frame in sorted(track["keyframes"]):
            key = track["keyframes"][frame]
            entry.set_keyframe(
                int(frame), key["box"],
                occlusion=OCCLUSION_LEVELS[1 if key["occluded"] else 0])
            key_frames += 1
        labels.tracks[track_id] = entry
    labels.save()
    _path, boxes = labels.export_to_detections()
    result = {"tracks": sum(1 for track in tracks if track["keyframes"]),
              "key_frames": key_frames, "boxes": int(boxes),
              "species": sorted({names[track["species"]] for track in tracks
                                 if track["keyframes"]})}
    if log:
        log(f"Labels: {result['tracks']} animal(s) on {key_frames} key "
            f"frame(s), {boxes} boxes with the frames between them "
            f"({', '.join(result['species']) or 'none'})")
    return result


def _own_poses(target_folder: str, modality: str) -> Optional[dict]:
    path = os.path.join(target_folder, f"poses_{modality}.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            poses = json.load(handle)
    except (OSError, ValueError):
        return None
    return poses if isinstance(poses, dict) and poses.get("images") else None


def place_labels(target_folder: str, flight_id: int, modality: str = "t",
                 log: Optional[Callable[[str], None]] = None
                 ) -> Optional[dict]:
    """Put a flight's published labels on the frames the flight has now,
    and note in the flight's folder which frames those were.

    Needs the published poses and labels in the flight's target folder
    (:func:`fetch_published`) and poses of the flight's own: the published
    ones (:func:`import_route`), or those of an extraction. Without poses
    there is nothing to put the labels on yet - ``None``, and the labels
    are placed when the frames have been extracted
    (:func:`after_extraction`).

    :return: what :func:`import_labels` returns, with ``"on"`` -
        ``"published"`` or ``"extracted"`` - and ``"left_out"``, the
        labelled boxes the flight's frames do not have
    :raises DatasetError: the published files are not there, or the
        flight's frames cannot be measured
    """
    labels_path = published_path(target_folder, flight_id, LABELS)
    poses_path = published_path(target_folder, flight_id, POSES)
    if not os.path.isfile(labels_path) or not os.path.isfile(poses_path):
        raise DatasetError(
            f"The published labels and poses of flight {flight_id} are "
            "not in the flight's folder: fetch them first.")
    own = _own_poses(target_folder, modality)
    if own is None:
        return None
    published = read_poses(poses_path)
    labels = read_labels(labels_path)
    mapping = frame_mapping(published, own)
    size = frame_size(target_folder, own, modality)
    transform = pixel_transform(published, own, size)
    tracks = label_tracks(labels, mapping, transform, size)
    result = import_labels(target_folder, tracks, modality, log)
    result["on"] = "published" if are_published(own) else "extracted"
    result["left_out"] = len(labels) - result["key_frames"]
    if result["left_out"] and log:
        log(f"{result['left_out']} labelled box(es) are on frames the "
            "flight does not have, or outside their picture: left out")
    state = read_state(target_folder)
    write_state(target_folder, flight=int(flight_id), labels=dict(
        state.get("labels") or {}, wanted=True, on=result["on"],
        frames=frames_fingerprint(own), tracks=result["tracks"],
        boxes=result["boxes"], left_out=result["left_out"]))
    return result


def import_route(target_folder: str, flight_id: int, dem_metadata_path: str,
                 epsg: int, modality: str = "t", force: bool = False,
                 log: Optional[Callable[[str], None]] = None) -> dict:
    """Make a flight's published poses its own, so that its route is known
    before any frame is extracted.

    Poses of an extraction are not replaced (unless *force*): they are the
    same places, of frames that exist.

    :param dem_metadata_path: the metadata of the flight's elevation model
        - the poses are relative to its origin
    :return: ``{"poses": path, or None where the flight kept its own,
        "frames": published frames, "replaced": bool}``
    :raises DatasetError: the published poses are not in the flight's
        folder, or the elevation model's origin is not known
    """
    from . import grid_north

    poses_path = published_path(target_folder, flight_id, POSES)
    if not os.path.isfile(poses_path):
        raise DatasetError(
            f"The published poses of flight {flight_id} are not in the "
            "flight's folder: fetch them first.")
    published = read_poses(poses_path)
    try:
        with open(dem_metadata_path, "r", encoding="utf-8") as handle:
            origin = json.load(handle)["origin_wgs84"]
        float(origin["latitude"]), float(origin["longitude"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DatasetError(
            "The origin of the flight's elevation model is not known "
            f"({exc}): the published poses cannot be placed.") from exc

    result = {"poses": None, "frames": len(published["images"]),
              "replaced": False}
    own = _own_poses(target_folder, modality)
    if own is not None and not are_published(own) and not force:
        if log:
            log("The flight has poses of its own extraction: the published "
                "ones are not put in their place")
        return result
    source = {"dataset": "BAMBI", "flight": int(flight_id)}
    if read_state(target_folder).get("doi"):
        source["doi"] = str(read_state(target_folder)["doi"])
    poses = flight_poses(published, origin, epsg, source)
    result["poses"] = write_poses(target_folder, poses, modality)
    result["replaced"] = own is not None
    # as an extraction does: the angle between the map's north and true
    # north goes with every pose
    grid_north.record(result["poses"], epsg)
    state = read_state(target_folder)
    if state.get("frames") == PROCESSED and modality == "t":
        # the RGB frames of the processed video have the same poses
        grid_north.record(write_poses(target_folder, poses, "w"), epsg)
    write_state(target_folder, flight=int(flight_id), route=dict(
        state.get("route") or {}, frames=result["frames"]))
    if log:
        log(f"Flight route: {len(poses['images'])} poses as published, "
            "placed on the flight's elevation model")
    return result


def import_correction(target_folder: str, flight_id: int, epsg: int,
                      position: Optional[Tuple[float, float]] = None,
                      modality: str = "t",
                      log: Optional[Callable[[str], None]] = None
                      ) -> Optional[dict]:
    """A flight's published correction in the plugin's form
    (:func:`flight_correction`), written beside the published one - its
    ranges of frames numbered as the flight's frames are now.

    :param position: ``(latitude, longitude)`` of the flight, for the
        angle between the map's north and true north there - not needed
        where the published poses are in the flight's folder
    :return: ``{"correction", "path", "convergence"}``, or ``None`` for a
        flight whose folder has no published correction
    :raises DatasetError: where the flight took place is not known
    """
    from . import grid_north

    correction_path = published_path(target_folder, flight_id, CORRECTION)
    if not os.path.isfile(correction_path):
        return None
    poses_path = published_path(target_folder, flight_id, POSES)
    published = read_poses(poses_path) if os.path.isfile(poses_path) else None
    convergence = None
    if published is not None:
        convergence = grid_north.here(published, epsg)
    if convergence is None and position is not None:
        convergence = float(grid_north.convergence(
            position[0], position[1], epsg))
    if convergence is None:
        raise DatasetError(
            f"Where flight {flight_id} took place is not known: its "
            "published correction cannot be fitted to the map.")
    # frames of an extraction are numbered their own way
    own = _own_poses(target_folder, modality)
    mapping = None
    if published is not None and own is not None and not are_published(own):
        mapping = frame_mapping(published, own)
    correction = flight_correction(read_correction(correction_path),
                                   convergence, mapping)
    path = write_correction(published_folder(target_folder), correction,
                            f"{int(flight_id)}_correction_qgis.json")
    state = read_state(target_folder)
    write_state(target_folder, flight=int(flight_id), correction=dict(
        state.get("correction") or {}, epsg=int(epsg),
        ranges=len(correction["additional"]),
        frames=frames_fingerprint(own) if mapping is not None else ""))
    if log:
        turn = math.degrees(correction["rotation"]["z"])
        log("Correction as published, the "
            f"{abs(convergence):.2f} degrees between the map's north and "
            f"true north taken out of its rotation: z = {turn:.2f} "
            f"degrees, {correction['translation']['z']:+.2f} m in height")
    return {"correction": correction, "path": path,
            "convergence": float(convergence)}


def after_extraction(target_folder: str, modality: str = "t",
                     log: Optional[Callable[[str], None]] = None
                     ) -> Optional[dict]:
    """After a flight's frames were extracted - for a flight of the
    dataset, and where the frames are not the ones dealt with already:

    * its published labels, where they were asked for, are put on the
      frames it has now;
    * a published correction that changes over ranges of frames is written
      anew with the ranges numbered as the frames are now, and the log
      says where it is.

    Whatever goes wrong is said and costs the extraction nothing.

    :return: what :func:`place_labels` returned, or ``None``
    """
    state = read_state(target_folder)
    if modality != "t" or "flight" not in state:
        return None
    own = _own_poses(target_folder, modality)
    if own is None:
        return None
    frames = frames_fingerprint(own)
    try:
        flight_id = int(state["flight"])
    except (TypeError, ValueError):
        return None
    _renumber_correction(target_folder, flight_id, state, frames, log)
    asked = state.get("labels") or {}
    if not asked.get("wanted") or asked.get("frames") == frames:
        return None
    try:
        result = place_labels(target_folder, flight_id, modality, log)
    except Exception as exc:  # noqa: BLE001 - the frames are there all the same
        quiet.problem("The published labels could not be put on the "
                      "extracted frames")
        if log:
            log("Warning: the labels of the BAMBI dataset could not be put "
                f"on the extracted frames ({exc})")
        return None
    if result is not None and log:
        log("The labels of the BAMBI dataset are on the extracted frames "
            "now (found by the moment a frame was taken, scaled to its "
            "picture)")
    return result


def _renumber_correction(target_folder: str, flight_id: int, state: dict,
                         frames: str, log) -> None:
    """Write a correction with ranges of frames anew, for frames that were
    just extracted - and say where it is: the flight's settings are the
    user's, and are not changed behind their back."""
    noted = state.get("correction") or {}
    if not noted.get("ranges") or noted.get("frames") == frames:
        return
    try:
        again = import_correction(target_folder, flight_id,
                                  int(noted.get("epsg") or 0))
    except Exception as exc:  # noqa: BLE001 - the frames are there all the same
        quiet.problem("The published correction could not be renumbered "
                      "to the extracted frames")
        if log:
            log("Warning: the frame ranges of the published correction "
                f"could not be renumbered to the extracted frames ({exc})")
        return
    if again is not None and log:
        log("The published correction changes over "
            f"{len(again['correction']['additional'])} range(s) of frames, "
            "which named frames of the published video. They are numbered "
            "as the extracted frames in "
            f"{again['path']}: load that file under Configuration > "
            "Correction to use them.")


# ---------------------------------------------------------------------------
# A flight, ready
# ---------------------------------------------------------------------------

def _epsg_of(metadata_path: str) -> Optional[int]:
    """The UTM CRS an elevation model's metadata names."""
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            found = re.search(r"(\d{4,6})\s*$",
                              str(json.load(handle).get("crs") or ""))
    except (OSError, ValueError, AttributeError):
        return None
    return int(found.group(1)) if found else None


#: The masks of the labelled flight, and what an extraction calls them.
MASKS = (("{id}_mask_t.png", "mask_T.png"), ("{id}_mask_w.png", "mask_W.png"))


def extract_processed_frames(video: str, target_folder: str,
                             progress: Optional[Callable[[int], None]] = None,
                             cancelled: Optional[Callable[[], bool]] = None,
                             log: Optional[Callable[[str], None]] = None
                             ) -> int:
    """What 'Extract Frames' does, for a processed video: its frames are
    undistorted and matched already, thermal on the left and RGB on the
    right, and frame *k* is pose *k* of the published poses. Every frame
    is cut in two and written as ``frames_t/<imagefile>`` and
    ``frames_w/<imagefile>``; the published poses, which have to be the
    flight's own already (:func:`import_route`), become those of the RGB
    frames as well.

    Frames that are there are not written again.

    :return: the number of frames the flight has now
    :raises DatasetError: the flight has no published poses of its own,
        the video cannot be read, or it has fewer frames than poses
    :raises core.checked_download.Cancelled: stopped by the user
    """
    poses = _own_poses(target_folder, "t")
    if poses is None or not are_published(poses):
        raise DatasetError(
            "The published poses are not the flight's own: the frames of "
            "the processed video cannot be told apart without them.")
    names = [str(image.get("imagefile") or "") for image in poses["images"]]
    folders = {camera: os.path.join(target_folder, f"frames_{camera}")
               for camera in ("t", "w")}
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    missing = [index for index, name in enumerate(names)
               if not all(os.path.isfile(os.path.join(folder, name))
                          for folder in folders.values())]
    if missing:
        try:
            import cv2
        except ImportError as exc:
            raise DatasetError(
                "OpenCV is needed to cut the processed video into frames "
                f"({exc}). Install it with the Dependency Manager.") from exc
        capture = cv2.VideoCapture(video)
        try:
            if not capture.isOpened():
                raise DatasetError(
                    f"{os.path.basename(video)} cannot be read as a video.")
            wanted, last = set(missing), max(missing)
            for index in range(last + 1):
                if cancelled and cancelled():
                    raise checked_download.Cancelled("cancelled")
                if index in wanted:
                    ok, frame = capture.read()
                else:
                    ok, frame = capture.grab(), None
                if not ok:
                    raise DatasetError(
                        f"{os.path.basename(video)} ends at frame {index}, "
                        f"and the poses name {len(names)} frames.")
                if frame is not None:
                    half = frame.shape[1] // 2
                    for camera, picture in (("t", frame[:, :half]),
                                            ("w", frame[:, half:])):
                        cv2.imwrite(
                            os.path.join(folders[camera], names[index]),
                            picture)
                if progress and index % 20 == 0:
                    progress(int(100.0 * index / (last + 1)))
        finally:
            capture.release()
    # the same camera took both halves: the poses are those of both
    from . import grid_north

    rgb = write_poses(target_folder, poses, "w")
    epsg = (grid_north.recorded(poses) or {}).get("epsg")
    if epsg:
        grid_north.record(rgb, int(epsg))
    # the masks, under the names an extraction gives them
    import shutil

    flight_folder = os.path.dirname(video)
    number = os.path.basename(video).split("_")[0]
    for published, own in MASKS:
        source = os.path.join(flight_folder, published.format(id=number))
        if os.path.isfile(source):
            shutil.copyfile(source, os.path.join(target_folder, own))
    write_state(target_folder, frames=PROCESSED)
    if progress:
        progress(100)
    if log:
        written = f"{len(missing)} written" if missing else "all there"
        log(f"Frames from the processed video: {len(names)} thermal and "
            f"{len(names)} RGB ({written}), in frames_t/ and frames_w/")
    return len(names)


def poses_bounds(poses: dict) -> Optional[Tuple[float, float, float, float]]:
    """``(west, south, east, north)`` of where published poses were
    taken; ``None`` where none says."""
    places = []
    for image in (poses or {}).get("images") or []:
        try:
            places.append((float(image["lng"]), float(image["lat"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not places:
        return None
    return (min(p[0] for p in places), min(p[1] for p in places),
            max(p[0] for p in places), max(p[1] for p in places))


def prepare_flight(entry: Flight, folder: str, session,
                   route: bool = True, labels: bool = True,
                   correction: bool = True, videos: bool = True,
                   version: str = PROCESSED,
                   elevation: Optional[Callable[..., dict]] = None,
                   make_route: Optional[Callable[[dict], None]] = None,
                   progress: Optional[Callable[[int], None]] = None,
                   cancelled: Optional[Callable[[], bool]] = None,
                   log: Optional[Callable[[str], None]] = None,
                   listings: Optional[Dict[int, dict]] = None,
                   **fetching) -> dict:
    """Make one flight of the dataset ready in *folder*: its recordings
    (with *videos*), its elevation model, and - as asked - its published
    route, labels and camera correction.

    *version* says which video: the labelled dataset's processed one
    (the default; the annotations come with its archive, and its frames
    are extracted at once - :func:`extract_processed_frames`) or the raw
    recordings, which 'Extract Frames' works on.

    Without *videos* only the annotations are fetched - poses, labels,
    correction, a megabyte or two: enough for the route and the labelled
    animals, a fast overview. The recordings can be fetched later by
    asking for the flight again.

    :param elevation: makes the elevation model - called with ``(bounds,
        folder, crs, log=, progress=, cancelled=)``, returns ``{"mesh",
        "metadata"}`` (``copernicus_dem_downloader.build_dem``); without
        it a flight gets none
    :param make_route: called with the result once the published poses are
        the flight's own - to write the route's files
    :param listings: ``{record: its files}`` kept between flights
    :return: ``{"flight", "name", "folder", "target_folder", "files",
        "epsg", "dem", "dem_metadata", "route", "correction", "labels",
        "labels_wanted", "videos" (the raw recordings are there),
        "processed_video" (its path, or ""), "frames" (extracted from
        it), "notes"}`` - *route*, *correction* and
        *labels* are what :func:`import_route`, :func:`import_correction`
        and :func:`place_labels` returned, ``None`` where nothing was
        done; *notes* is what did not work, in sentences
    :raises DatasetError: neither recordings nor annotations could be had
    :raises core.checked_download.Cancelled, core.remote_zip.Cancelled,
        core.environment.Cancelled: stopped by the user
    """
    from . import environment, flight_files
    from .crs_utils import utm_epsg_for_position

    say = log or (lambda message: None)
    listings = {} if listings is None else listings

    def part(low: int, high: int) -> Optional[Callable[[int], None]]:
        if not progress:
            return None
        return lambda value: progress(int(low + (high - low) * value / 100.0))

    if version not in VERSIONS:
        raise DatasetError(f"The dataset has no version '{version}'.")
    target = os.path.join(folder, TARGET_SUBFOLDER)
    video = ""
    if videos and version == RECORDINGS:
        files = fetch_recordings(
            entry, folder, session, progress=part(0, 80),
            cancelled=cancelled, log=log, listings=listings, **fetching)
    elif videos:
        video = fetch_processed(
            entry, folder, target, session, progress=part(0, 80),
            cancelled=cancelled, log=log, listings=listings, **fetching)
        files = flight_archive.flight_files(folder)
    else:
        if entry.labelled is None:
            raise DatasetError(
                f"Flight {entry.id} is not in the labelled dataset: without "
                "its videos there is nothing of it to fetch.")
        files = flight_archive.flight_files(folder)
    os.makedirs(target, exist_ok=True)
    notes: List[str] = []
    prepared = {"flight": entry.id, "name": entry.name, "folder": folder,
                "target_folder": target, "files": files, "epsg": None,
                "dem": "", "dem_metadata": "", "route": None,
                "correction": None, "labels": None, "labels_wanted": False,
                "videos": flight_archive.is_complete(files),
                "processed_video": video or (
                    processed_video(folder, entry.id) if os.path.isfile(
                        processed_video(folder, entry.id)) else ""),
                "frames": 0, "notes": notes}

    # -- what the labelled dataset has of the flight -------------------------
    wanted = route or labels or correction or not videos or bool(video)
    there: Dict[str, str] = {}
    if wanted and entry.labelled is None:
        notes.append("The flight is not in the labelled dataset: neither "
                     "a route nor labels nor a correction were published "
                     "for it.")
    elif wanted:
        try:
            there = fetch_published(entry, target, session, cancelled, log,
                                    listings)
        except DatasetError as exc:
            if not videos or version == PROCESSED:
                raise
            notes.append(str(exc))
    if there:
        asked = dict(read_state(target).get("labels") or {})
        asked["wanted"] = bool(labels or asked.get("wanted"))
        write_state(target, flight=entry.id, doi=entry.labelled.doi,
                    labels=asked)
        prepared["labels_wanted"] = bool(labels)
    else:
        write_state(target, flight=entry.id)
    if progress:
        progress(84)

    # -- the elevation model: under the flight log, or under the poses ------
    bounds = flight_archive.recording_bounds(str(files.get("airdata") or ""))
    under = "recording"
    if bounds is None and POSES in there:
        under = "poses"
        try:
            bounds = poses_bounds(read_poses(there[POSES]))
        except DatasetError as exc:
            notes.append(str(exc))
    if bounds is not None:
        prepared["epsg"] = utm_epsg_for_position(
            (bounds[1] + bounds[3]) / 2.0, (bounds[0] + bounds[2]) / 2.0)
    found = flight_files.detect_common_files(folder)
    # a model made for the published frames alone is too small for the
    # whole recording, once that is there
    small = read_state(target).get("elevation") == "poses"
    outgrown = small and under == "recording" and bounds is not None
    if ("dem" not in found or outgrown) and elevation is not None:
        if bounds is None:
            notes.append("Where the flight took place is not known: no "
                         "elevation model was made.")
        else:
            try:
                elevation(bounds, folder, f"EPSG:{prepared['epsg']}",
                          log=log, progress=part(84, 94),
                          cancelled=cancelled)
            except environment.Cancelled:
                raise
            except environment.EnvironmentFailure as exc:
                notes.append(
                    f"The elevation model could not be fetched ({exc}). "
                    "Fetch it with 'Download DEM (worldwide)' on the Input "
                    "tab.")
            else:
                write_state(target, elevation=under)
            found = flight_files.detect_common_files(folder)
    prepared["dem"] = found.get("dem", "")
    prepared["dem_metadata"] = found.get("dem_metadata", "")
    named = _epsg_of(prepared["dem_metadata"])
    if named:
        prepared["epsg"] = named
    if progress:
        progress(94)

    # frames of the processed video are the published frames: their poses
    # are part of them, asked for or not - as an extraction writes both
    with_frames = bool(video)
    route = route or with_frames
    if there and (route or labels or correction):
        middle = None
        if bounds is not None:
            middle = ((bounds[1] + bounds[3]) / 2.0,
                      (bounds[0] + bounds[2]) / 2.0)
        _into_flight(entry, prepared, there, route, labels, correction,
                     middle, make_route, log,
                     video=video, cancelled=cancelled,
                     progress=part(94, 100))
    if progress:
        progress(100)
    for note in notes:
        say(f"Flight {entry.id}: {note}")
    return prepared


def _into_flight(entry: Flight, prepared: dict, there: Dict[str, str],
                 route: bool, labels: bool, correction: bool, position,
                 make_route, log, video: str = "", cancelled=None,
                 progress=None) -> None:
    """The last part of :func:`prepare_flight`: poses, correction and
    labels into the flight. What fails is noted - what was fetched is
    there all the same."""
    target, notes = prepared["target_folder"], prepared["notes"]
    if prepared["epsg"] is None:
        notes.append("Where the flight took place is not known: what was "
                     "published of it was fetched, and not put into the "
                     "flight.")
        return
    try:
        if correction and CORRECTION not in there:
            notes.append("No correction was published for the flight.")
        elif correction:
            prepared["correction"] = import_correction(
                target, entry.id, prepared["epsg"], position, log=log)
        if (route or labels) and POSES not in there:
            notes.append("The labelled flight has no poses file: neither "
                         "its route nor its labels could be taken.")
            return
        if route and not prepared["dem_metadata"]:
            notes.append("Without an elevation model the published route "
                         "cannot be placed. It was fetched, and is put "
                         "into the flight when the flight is asked for "
                         "again.")
        elif route:
            prepared["route"] = import_route(
                target, entry.id, prepared["dem_metadata"], prepared["epsg"],
                log=log)
        if video and (prepared["route"] or {}).get("poses"):
            prepared["frames"] = extract_processed_frames(
                video, target, progress, cancelled, log)
        elif video:
            notes.append("The frames of the processed video were not "
                         "extracted: the published poses are not the "
                         "flight's own.")
        if labels and LABELS not in there:
            notes.append("No labels were published for the flight.")
        elif labels:
            prepared["labels"] = place_labels(target, entry.id, log=log)
            if prepared["labels"] is None and log:
                log("The labels are put on the frames once they are "
                    "extracted ('Extract Frames')")
    except checked_download.Cancelled:
        raise
    except DatasetError as exc:
        notes.append(str(exc))
    except Exception as exc:  # noqa: BLE001 - said; what was fetched stays
        quiet.problem(f"What was published of flight {entry.id} could not "
                      "be put into the flight")
        notes.append("What was published of the flight could not be put "
                     f"into it ({exc}).")
    if make_route is not None and (prepared["route"] or {}).get("poses"):
        try:
            make_route(prepared)
        except Exception as exc:  # noqa: BLE001 - said; the poses are there
            quiet.problem(f"The route of flight {entry.id} could not be "
                          "written")
            notes.append(f"The route's files could not be written ({exc}): "
                         "run 'Generate Flight Route'.")


def flight_configuration(prepared: dict, defaults: Dict[str, object],
                         custom_calibration: str = "Custom file..."
                         ) -> Dict[str, object]:
    """The settings a prepared flight (:func:`prepare_flight`) starts
    with, by their keys in ``core.config_schema``: a new flight's - the
    form as it is when the plugin opens - with the flight's recordings,
    its elevation model, its CRS and, where it has one, its published
    correction.

    :param defaults: what a new form shows, by schema key
    :param custom_calibration: what the calibration lists call "a file of
        my own"
    """
    from . import flights as flights_core

    files = prepared["files"]
    values = flights_core.new_flight_values({}, False, defaults=defaults)

    def paths(key: str) -> str:
        return ", ".join(files.get(key) or [])

    values.update({
        "Input/VideoMode": True,
        "Input/EmbeddedSrt": False,
        "Input/ThermalVideoPaths": paths("thermal_videos"),
        "Input/ThermalSrtPaths": paths("thermal_srts"),
        "Input/RgbVideoPaths": paths("rgb_videos"),
        "Input/RgbSrtPaths": paths("rgb_srts"),
        "Input/AirdataPath": str(files.get("airdata") or ""),
        "Input/DemPath": prepared.get("dem") or "",
        "Input/DemMetadataPath": prepared.get("dem_metadata") or "",
        "Input/TargetFolder": prepared["target_folder"],
    })
    # the calibrations the flight came with, not a preset
    for camera, key in (("Thermal", "thermal_calibration"),
                        ("Rgb", "rgb_calibration")):
        if files.get(key):
            values[f"Input/{camera}CalibrationPreset"] = custom_calibration
            values[f"Input/{camera}CalibrationPath"] = str(files[key])
    if prepared.get("epsg"):
        values["Input/TargetCrs"] = f"EPSG:{int(prepared['epsg'])}"
    values.update(correction_settings(prepared.get("correction")))
    return values


#: The settings that name a flight's recordings: what a flight that was
#: fetched without its videos gets when they are fetched later.
RECORDING_KEYS = frozenset({
    "Input/VideoMode", "Input/EmbeddedSrt", "Input/ThermalVideoPaths",
    "Input/ThermalSrtPaths", "Input/RgbVideoPaths", "Input/RgbSrtPaths",
    "Input/AirdataPath", "Input/ThermalCalibrationPreset",
    "Input/ThermalCalibrationPath", "Input/RgbCalibrationPreset",
    "Input/RgbCalibrationPath"})


def correction_settings(imported: Optional[dict]) -> Dict[str, object]:
    """A correction (:func:`import_correction`) as the settings of a
    flight: its values in radians, its ranges of frames, and the file they
    are from. ``{}`` for none."""
    if not imported:
        return {}
    correction = imported["correction"]
    return {
        "Input/CorrectionPath": imported.get("path") or "",
        "Correction/RotationUnit": 0,               # radians
        "Correction/TransX": correction["translation"]["x"],
        "Correction/TransY": correction["translation"]["y"],
        "Correction/TransZ": correction["translation"]["z"],
        "Correction/RotX": correction["rotation"]["x"],
        "Correction/RotY": correction["rotation"]["y"],
        "Correction/RotZ": correction["rotation"]["z"],
        "Correction/AdditionalCorrections": json.dumps(
            correction.get("additional") or [], sort_keys=True),
    }


def has_correction(stored: Dict[str, object]) -> bool:
    """Whether a flight's stored settings hold a correction of their own:
    a value that is not nought, or a range of frames."""
    for key in ("TransX", "TransY", "TransZ", "RotX", "RotY", "RotZ"):
        try:
            if abs(float(stored.get(f"Correction/{key}") or 0.0)) > 1e-12:
                return True
        except (TypeError, ValueError):
            continue
    text = stored.get("Correction/AdditionalCorrections") or "[]"
    try:
        ranges = json.loads(text)
    except (TypeError, ValueError):
        ranges = []
    return bool(ranges)
