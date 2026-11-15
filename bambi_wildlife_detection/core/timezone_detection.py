# -*- coding: utf-8 -*-
"""Timezone-offset detection for flight inputs.

Moved from the dock widget (whose methods delegate here). SRT and
EXIF timestamps are local wall-clock time while the AirData log is UTC; the
offset between them is the timezone offset the extraction stages must apply.

**The offset is found from the date-times, not from the times of day.** It
used to be the difference between the mean *time of day* of the two, rounded
to whole hours - which goes wrong exactly when wildlife is flown: at night. A
video from 23:30 to 00:30 has a mean time of day of about noon, and the
answer was UTC-10 where UTC+2 was right; one starting at 00:30 local is still
"yesterday, 22:30" in UTC, and came out as UTC-22. The frames were then
matched to the wrong part of the flight log, or to none of it.

:func:`best_offset` instead looks for the offset - any quarter of an hour
between UTC-12 and UTC+14 - under which the local times fall on the log's
flagged rows. Only where the dates cannot be used (a camera clock that was
never set) is the time of day compared, and then on the clock face, where
23:30 and 00:30 are an hour apart.
"""

import math
import os
from datetime import datetime
from typing import Iterable, List, Optional, Sequence

_EPOCH = datetime(1970, 1, 1)

#: The offsets time zones have: quarter hours from UTC-12 to UTC+14.
_OFFSETS_MIN = tuple(range(-12 * 60, 14 * 60 + 1, 15))


def timezone_offset_hours(tz_name: str) -> Optional[float]:
    """Return the current UTC offset in hours for a given IANA timezone name."""
    try:
        import datetime
        from zoneinfo import ZoneInfo
        now = datetime.datetime.now(ZoneInfo(tz_name))
        return now.utcoffset().total_seconds() / 3600
    except Exception:
        try:
            import datetime
            from dateutil import tz as dateutil_tz
            zone = dateutil_tz.gettz(tz_name)
            if zone is None:
                return None
            now = datetime.datetime.now(zone)
            return now.utcoffset().total_seconds() / 3600
        except Exception:
            return None


def _hours(moments: Iterable[datetime]) -> list:
    """The time of day, in hours, of each of *moments*."""
    return [dt.hour + dt.minute / 60.0 + dt.second / 3600.0
            for dt in moments]


def srt_local_datetimes(srt_paths: List[str]) -> List[datetime]:
    """The local date-time of every SRT frame timestamp."""
    import re

    found: List[datetime] = []
    _dt_re = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    for srt_path in srt_paths:
        try:
            with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = _dt_re.search(line)
                    if m:
                        found.append(datetime.strptime(
                            m.group(1), "%Y-%m-%d %H:%M:%S"))
        except Exception:  # nosec B112
            continue
    return found


def srt_local_hours(srt_paths: List[str]) -> list:
    """Collect the local time-of-day (hours) of every SRT frame timestamp."""
    return _hours(srt_local_datetimes(srt_paths))


def exif_photo_datetimes(photo_dir: str) -> List[datetime]:
    """The local date-time of every photo's EXIF timestamp."""
    import glob as glob_mod

    image_paths: list = []
    for ext in ("*.jpg", "*.jpeg", "*.tiff", "*.tif", "*.png",
                "*.JPG", "*.JPEG", "*.TIFF", "*.TIF", "*.PNG"):
        image_paths.extend(glob_mod.glob(os.path.join(photo_dir, ext)))
    if not image_paths:
        return []

    try:
        from PIL import Image
    except ImportError:
        return []

    found: List[datetime] = []
    for p in sorted(set(image_paths)):
        try:
            with Image.open(p) as img:
                exif = img._getexif()
                if exif is None:
                    continue
                dt_str = exif.get(36867)  # DateTimeOriginal
                if not dt_str:
                    continue
                found.append(datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S"))
        except Exception:  # nosec B112
            continue
    return found


def exif_photo_hours(photo_dir: str) -> list:
    """Collect the local time-of-day (hours) of every photo's EXIF timestamp."""
    return _hours(exif_photo_datetimes(photo_dir))


def airdata_utc_datetimes(airdata_path: str, flag_column_lower: str
                          ) -> Optional[List[datetime]]:
    """The UTC date-times of the AirData rows whose flag column (lowercased
    name) is truthy; ``None`` when the file or the columns are not there, or
    no row is flagged."""
    import csv

    if not airdata_path or not os.path.exists(airdata_path):
        return None

    found: List[datetime] = []
    try:
        with open(airdata_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

            flag_col = next(
                (h for h in headers if h.strip().lower() == flag_column_lower), None)
            datetime_col = next(
                (h for h in headers if "datetime" in h.lower() and "utc" in h.lower()),
                None)
            if datetime_col is None:
                datetime_col = next(
                    (h for h in headers if "datetime" in h.lower()), None)
            if not flag_col or not datetime_col:
                return None

            for row in reader:
                val = row.get(flag_col, "").strip()
                if not val or val == "0" or val.lower() == "false":
                    continue
                dt_str = row.get(datetime_col, "").strip()
                for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                            "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                            "%Y-%m-%d %H:%M:%S.%f"):
                    try:
                        found.append(datetime.strptime(dt_str, fmt))
                        break
                    except ValueError:
                        continue
    except Exception:
        return None

    return found if found else None


def airdata_utc_hours(airdata_path: str, flag_column_lower: str) -> Optional[list]:
    """Read UTC hours from AirData rows where the given flag column (lowercased) is truthy."""
    found = airdata_utc_datetimes(airdata_path, flag_column_lower)
    return None if found is None else _hours(found)


def _minutes(moments: Iterable[datetime]) -> set:
    """Whole minutes since the epoch at which something of *moments* lies."""
    return {int((dt - _EPOCH).total_seconds() // 60) for dt in moments}


def _clock_offset(local_hours: Sequence[float],
                  utc_hours: Sequence[float]) -> float:
    """Hours from the mean UTC time of day to the mean local one, the short
    way round the clock face (-12 to +12), in whole hours."""
    def mean_angle(hours):
        angles = [h / 24.0 * 2.0 * math.pi for h in hours]
        return math.atan2(sum(math.sin(a) for a in angles),
                          sum(math.cos(a) for a in angles))

    turn = mean_angle(local_hours) - mean_angle(utc_hours)
    hours = (turn / (2.0 * math.pi) * 24.0 + 12.0) % 24.0 - 12.0
    return float(round(hours))


def best_offset(local: Sequence[datetime], utc: Sequence[datetime]
                ) -> Optional[float]:
    """The UTC offset, in hours, of the clock that wrote *local*.

    :param local: wall-clock date-times of the recording (SRT, EXIF)
    :param utc: UTC date-times of the flight log's rows flagged as that
        recording
    :return: the offset - a multiple of a quarter of an hour - under which
        most of the recording falls on flagged log rows; ``None`` when
        either side is empty

    The recording is usually part of what the log flags (one of several
    videos), so several offsets may put it inside: of those, a whole hour
    is preferred to a half and a half to a quarter, as time zones are. With
    no offset at all that fits - the dates of the two do not belong
    together - the times of day are compared on the clock face.
    """
    local, utc = list(local), list(utc)
    if not local or not utc:
        return None

    recorded = _minutes(local)
    logged = _minutes(utc)
    # a minute either side: the two clocks tick apart by seconds
    logged |= {minute + 1 for minute in logged} | \
        {minute - 1 for minute in logged}

    best, best_key = None, None
    for offset in _OFFSETS_MIN:
        hits = sum(1 for minute in recorded if minute - offset in logged)
        if not hits:
            continue
        key = (hits, offset % 60 == 0, offset % 30 == 0, -abs(offset))
        if best_key is None or key > best_key:
            best, best_key = offset, key
    if best is not None:
        return best / 60.0
    return _clock_offset(_hours(local), _hours(utc))


def offset_from_srt(srt_paths: List[str], airdata_path: str) -> Optional[float]:
    """Match SRT local timestamps against AirData isVideo UTC timestamps."""
    if not srt_paths:
        return None
    local = srt_local_datetimes(srt_paths)
    if not local:
        return None
    logged = airdata_utc_datetimes(airdata_path, "isvideo")
    return best_offset(local, logged or [])


def offset_from_exif(photo_dir: str, airdata_path: str) -> Optional[float]:
    """Match photo EXIF timestamps against AirData isPhoto UTC timestamps."""
    if not photo_dir or not os.path.isdir(photo_dir):
        return None
    local = exif_photo_datetimes(photo_dir)
    if not local:
        return None
    logged = airdata_utc_datetimes(airdata_path, "isphoto")
    return best_offset(local, logged or [])


def detect_offset(srt_paths: Sequence[str], photo_dirs: Sequence[str],
                  airdata_path: str, video_mode: bool = True
                  ) -> Optional[float]:
    """The UTC offset of a flight's recordings, from whichever of them is
    there: in video mode the SRT files (of either camera), else - or failing
    that - the photos' EXIF times. ``None`` when nothing can be compared."""
    if video_mode:
        offset = offset_from_srt(
            [path for path in srt_paths if path], airdata_path)
        if offset is not None:
            return offset
    for photo_dir in photo_dirs:
        offset = offset_from_exif(photo_dir, airdata_path)
        if offset is not None:
            return offset
    return None


def inputs_signature(paths: Iterable[str]) -> tuple:
    """What identifies the state of the files and folders in *paths*: each
    with its size and time of last change (a folder: of its newest entry).
    Equal signatures mean a detection would read the same data again."""
    signature = []
    for path in paths:
        if not path:
            continue
        try:
            if os.path.isdir(path):
                entries = sorted(os.listdir(path))
                newest = max((os.path.getmtime(os.path.join(path, name))
                              for name in entries), default=0.0)
                signature.append((path, len(entries), newest))
            else:
                signature.append(
                    (path, os.path.getsize(path), os.path.getmtime(path)))
        except OSError:
            signature.append((path, None, None))
    return tuple(signature)
