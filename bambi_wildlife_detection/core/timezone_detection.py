# -*- coding: utf-8 -*-
"""Timezone-offset detection for flight inputs.

Moved from ``bambi_dock_widget.py`` (whose methods delegate here). SRT and
EXIF timestamps are local wall-clock time while the AirData log is UTC; the
offset between their mean times of day (rounded to whole hours) is the
timezone offset the extraction stages must apply.
"""

import os
from typing import List, Optional


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


def srt_local_hours(srt_paths: List[str]) -> list:
    """Collect the local time-of-day (hours) of every SRT frame timestamp."""
    import re
    from datetime import datetime

    hours: list = []
    _dt_re = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    for srt_path in srt_paths:
        try:
            with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = _dt_re.search(line)
                    if m:
                        dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                        hours.append(dt.hour + dt.minute / 60.0 + dt.second / 3600.0)
        except Exception:  # nosec B112
            continue
    return hours


def exif_photo_hours(photo_dir: str) -> list:
    """Collect the local time-of-day (hours) of every photo's EXIF timestamp."""
    import glob as glob_mod
    from datetime import datetime

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

    hours: list = []
    for p in image_paths:
        try:
            with Image.open(p) as img:
                exif = img._getexif()
                if exif is None:
                    continue
                dt_str = exif.get(36867)  # DateTimeOriginal
                if not dt_str:
                    continue
                dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                hours.append(dt.hour + dt.minute / 60.0 + dt.second / 3600.0)
        except Exception:  # nosec B112
            continue
    return hours


def airdata_utc_hours(airdata_path: str, flag_column_lower: str) -> Optional[list]:
    """Read UTC hours from AirData rows where the given flag column (lowercased) is truthy."""
    import csv
    from datetime import datetime

    if not airdata_path or not os.path.exists(airdata_path):
        return None

    hours: list = []
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
                        dt = datetime.strptime(dt_str, fmt)
                        hours.append(dt.hour + dt.minute / 60.0 + dt.second / 3600.0)
                        break
                    except ValueError:
                        continue
    except Exception:
        return None

    return hours if hours else None


def offset_from_srt(srt_paths: List[str], airdata_path: str) -> Optional[float]:
    """Match SRT local timestamps against AirData isVideo UTC timestamps."""
    if not srt_paths:
        return None
    srt_hours = srt_local_hours(srt_paths)
    if not srt_hours:
        return None

    airdata_hours = airdata_utc_hours(airdata_path, "isvideo")
    if not airdata_hours:
        return None

    offset = round(sum(srt_hours) / len(srt_hours) - sum(airdata_hours) / len(airdata_hours))
    return float(offset)


def offset_from_exif(photo_dir: str, airdata_path: str) -> Optional[float]:
    """Match photo EXIF timestamps against AirData isPhoto UTC timestamps."""
    if not photo_dir or not os.path.isdir(photo_dir):
        return None

    photo_hours = exif_photo_hours(photo_dir)
    if not photo_hours:
        return None

    airdata_hours = airdata_utc_hours(airdata_path, "isphoto")
    if not airdata_hours:
        return None

    offset = round(sum(photo_hours) / len(photo_hours) - sum(airdata_hours) / len(airdata_hours))
    return float(offset)
