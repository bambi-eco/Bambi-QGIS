# -*- coding: utf-8 -*-
"""EPSG parsing and UTM-zone detection.

Moved from ``bambi_dock_widget.py`` (whose methods delegate here). The
pipeline requires a metric UTM CRS; the zone is derived from the flight's
first valid GPS position in the AirData log.
"""

import csv
import os
from typing import Callable, Optional


def parse_epsg_from_text(crs_text: str) -> Optional[int]:
    """Parse EPSG code from text input.

    Handles formats like ``EPSG:32633``, ``epsg:32633`` and ``32633``.
    Returns the EPSG code as integer, or None if invalid.
    """
    try:
        crs_text = crs_text.strip().upper()

        if crs_text.startswith("EPSG:"):
            code_str = crs_text[5:].strip()
        else:
            code_str = crs_text

        return int(code_str)
    except (ValueError, AttributeError):
        return None


def is_valid_utm_crs(crs_text: str) -> bool:
    """Check if the given CRS string is a valid UTM CRS.

    Valid UTM EPSG codes:
    - Northern Hemisphere: EPSG:32601 to EPSG:32660 (zones 1-60)
    - Southern Hemisphere: EPSG:32701 to EPSG:32760 (zones 1-60)
    """
    try:
        epsg = parse_epsg_from_text(crs_text)
        if epsg is None:
            return False

        is_northern = 32601 <= epsg <= 32660
        is_southern = 32701 <= epsg <= 32760

        return is_northern or is_southern
    except Exception:
        return False


def utm_epsg_for_position(lat: float, lon: float) -> int:
    """Return the UTM EPSG code containing the WGS84 position.

    UTM zones are 6 degrees wide; zone 1 starts at -180°. Northern
    hemisphere maps to 326xx, southern to 327xx.
    """
    utm_zone = int((lon + 180) / 6) + 1
    utm_zone = max(1, min(60, utm_zone))
    return (32600 if lat >= 0 else 32700) + utm_zone


def detect_utm_from_airdata(csv_path: str,
                            log_fn: Optional[Callable[[str], None]] = None,
                            ) -> Optional[str]:
    """Detect appropriate UTM zone from GPS coordinates in AirData CSV.

    Uses the first valid GPS position to determine the UTM zone.
    Returns a CRS string like ``"EPSG:32633"`` or None.
    """
    def log(message):
        if log_fn:
            log_fn(message)

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Find latitude/longitude columns
            lat_col = None
            lon_col = None

            fieldnames = reader.fieldnames
            if fieldnames is None:
                return None

            for col in fieldnames:
                col_lower = col.lower().strip()
                if lat_col is None and any(x in col_lower for x in ['latitude', 'lat']):
                    lat_col = col
                if lon_col is None and any(x in col_lower for x in ['longitude', 'lon', 'lng']):
                    lon_col = col

            if lat_col is None or lon_col is None:
                log(f"Could not find lat/lon columns in {os.path.basename(csv_path)}")
                return None

            # Get first valid GPS position
            for row in reader:
                try:
                    lat = float(row[lat_col])
                    lon = float(row[lon_col])

                    # Skip invalid values
                    if lat == 0 and lon == 0:
                        continue
                    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                        continue

                    epsg = utm_epsg_for_position(lat, lon)
                    utm_zone = epsg % 100

                    log(f"Detected UTM zone {utm_zone}{'N' if lat >= 0 else 'S'} "
                        f"from GPS: ({lat:.4f}, {lon:.4f})")

                    return f"EPSG:{epsg}"

                except (ValueError, KeyError):
                    continue

            log(f"No valid GPS coordinates found in {os.path.basename(csv_path)}")
            return None

    except Exception as e:
        log(f"Error reading AirData CSV for CRS detection: {e}")
        return None
