# -*- coding: utf-8 -*-
"""Weather at flight time and over the day before it.

Not satellite data, but for a thermal flight the strongest explanation of
how well animals could be seen: what the air and the ground were at when
the drone flew, and how much sun, rain and wind the hours before had put
into them. A warm, sunlit meadow hides a deer; a cold, overcast one after a
clear night shows it.

The values come from Open-Meteo, which needs no account or key: its archive
(the ERA5 reanalysis family, global, back to 1940) for flights more than a
few days old, and its forecast service - which keeps the recent past - for
newer ones, because the reanalysis lags a few days behind.

These are model values for a grid cell of one to a few tens of kilometres,
so they describe the flight as a whole, not a patch inside it. Two things
are kept (``weather.json``, and ``weather_hourly.csv`` with the hourly rows
they were derived from):

``at_flight``
    Every variable interpolated to the middle of the flight.
``previous_24h``
    Sums and extremes of the 24 hours before the first frame.
"""

import csv
import os
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence

from . import env_remote
from .environment import iso

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)."

#: Flights older than this are asked from the archive, newer ones from the
#: forecast service: the reanalysis is published with a delay of some days.
ARCHIVE_DELAY_DAYS = 7

#: Hours fetched before the flight. The summary needs 24; the rest lets
#: "hours since rain" look further back.
LOOKBACK_HOURS = 72

#: Variables both services know, with the unit they are asked for in.
CORE_VARIABLES = {
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "dew_point_2m": "°C",
    "precipitation": "mm",
    "snowfall": "cm",
    "snow_depth": "m",
    "cloud_cover": "%",
    "wind_speed_10m": "m/s",
    "wind_gusts_10m": "m/s",
    "wind_direction_10m": "°",
    "shortwave_radiation": "W/m²",
    "sunshine_duration": "s",
    "surface_pressure": "hPa",
}

#: Soil variables are named by depth, and the two services use different
#: layers: ``our name -> their name`` per service.
SOIL_VARIABLES = {
    ARCHIVE_URL: {"soil_temperature": "soil_temperature_0_to_7cm",
                  "soil_moisture": "soil_moisture_0_to_7cm"},
    FORECAST_URL: {"soil_temperature": "soil_temperature_0cm",
                   "soil_moisture": "soil_moisture_0_to_1cm"},
}
SOIL_UNITS = {"soil_temperature": "°C", "soil_moisture": "m³/m³"}

#: Variables that are a total over the hour ending at the timestamp, so they
#: are summed, never interpolated.
HOURLY_TOTALS = ("precipitation", "snowfall", "sunshine_duration")

#: Rain below this in an hour does not count as rain, in mm.
RAIN_THRESHOLD_MM = 0.1


def units() -> Dict[str, str]:
    result = dict(CORE_VARIABLES)
    result.update(SOIL_UNITS)
    return result


def service_for(flight_start: datetime,
                now: Optional[datetime] = None) -> str:
    """The Open-Meteo service that covers a flight's date."""
    now = now or datetime.now(timezone.utc)
    age = now - flight_start
    return ARCHIVE_URL if age > timedelta(days=ARCHIVE_DELAY_DAYS) \
        else FORECAST_URL


def request_params(lat: float, lon: float, start: datetime, end: datetime,
                   variables: Sequence[str]) -> dict:
    return {
        "latitude": round(float(lat), 5),
        "longitude": round(float(lon), 5),
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ",".join(variables),
        "timezone": "GMT",
        "wind_speed_unit": "ms",
    }


def parse_hourly(document: dict, rename: Optional[Dict[str, str]] = None
                 ) -> List[dict]:
    """The ``hourly`` block of an answer as one dict per hour, with ``time``
    an aware UTC datetime and the service's names mapped back to ours."""
    rename = rename or {}
    hourly = document.get("hourly") or {}
    times = hourly.get("time") or []
    rows = []
    for index, text in enumerate(times):
        try:
            moment = datetime.fromisoformat(str(text)).replace(
                tzinfo=timezone.utc)
        except ValueError:
            continue
        row = {"time": moment}
        for name, values in hourly.items():
            if name == "time":
                continue
            value = values[index] if index < len(values) else None
            row[rename.get(name, name)] = (
                None if value is None else float(value))
        rows.append(row)
    return rows


def _interpolate(rows: List[dict], moment: datetime, name: str
                 ) -> Optional[float]:
    """*name* at *moment*, linearly between the two hours around it."""
    known = [row for row in rows if row.get(name) is not None]
    before = [row for row in known if row["time"] <= moment]
    after = [row for row in known if row["time"] >= moment]
    if not before and not after:
        return None
    if not before:
        return after[0][name]
    if not after:
        return before[-1][name]
    low, high = before[-1], after[0]
    span = (high["time"] - low["time"]).total_seconds()
    if span <= 0:
        return low[name]
    share = (moment - low["time"]).total_seconds() / span
    if name == "wind_direction_10m":
        # Across north (350° -> 10°) the short way round is through 0.
        delta = ((high[name] - low[name] + 180.0) % 360.0) - 180.0
        return (low[name] + share * delta) % 360.0
    return low[name] + share * (high[name] - low[name])


def _window(rows: List[dict], start: datetime, end: datetime) -> List[dict]:
    """Hours whose total covers ``(start, end]`` - a row's totals are for the
    hour *ending* at its timestamp."""
    return [row for row in rows if start < row["time"] <= end]


def _values(rows: List[dict], name: str) -> List[float]:
    return [row[name] for row in rows if row.get(name) is not None]


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def summarise(rows: List[dict], flight_start: datetime,
              flight_end: datetime) -> dict:
    """``{"at_flight": …, "previous_24h": …}`` from hourly *rows*."""
    middle = flight_start + (flight_end - flight_start) / 2
    names = sorted({name for row in rows for name in row if name != "time"})

    at_flight = {}
    for name in names:
        if name in HOURLY_TOTALS:
            continue
        at_flight[name] = _round(_interpolate(rows, middle, name))
    # The hour the flight's middle falls in carries the totals for it.
    current = _window(rows, middle - timedelta(hours=1), middle)
    if not current:
        current = [row for row in rows if row["time"] >= middle][:1]
    for name in HOURLY_TOTALS:
        values = _values(current, name)
        at_flight[name] = _round(sum(values)) if values else None
    temperature = at_flight.get("temperature_2m")
    dew_point = at_flight.get("dew_point_2m")
    if temperature is not None and dew_point is not None:
        # A small spread means air close to saturation: dew, mist, fog.
        at_flight["dew_point_spread"] = _round(temperature - dew_point)

    day = _window(rows, flight_start - timedelta(hours=24), flight_start)
    previous = {"hours": len(day)}
    rain = _values(day, "precipitation")
    previous["precipitation_sum"] = _round(sum(rain)) if rain else None
    snow = _values(day, "snowfall")
    previous["snowfall_sum"] = _round(sum(snow)) if snow else None
    sunshine = _values(day, "sunshine_duration")
    previous["sunshine_hours"] = (
        _round(sum(sunshine) / 3600.0) if sunshine else None)
    for name, label in (("temperature_2m", "temperature"),
                        ("soil_temperature", "soil_temperature")):
        values = _values(day, name)
        previous[f"{label}_min"] = _round(min(values)) if values else None
        previous[f"{label}_max"] = _round(max(values)) if values else None
    for name, label in (("cloud_cover", "cloud_cover_mean"),
                        ("shortwave_radiation", "shortwave_radiation_mean"),
                        ("wind_speed_10m", "wind_speed_mean"),
                        ("relative_humidity_2m", "relative_humidity_mean")):
        values = _values(day, name)
        previous[label] = (
            _round(sum(values) / len(values)) if values else None)
    gusts = _values(day, "wind_gusts_10m")
    previous["wind_gusts_max"] = _round(max(gusts)) if gusts else None

    earlier = [row for row in rows if row["time"] <= flight_start]
    wet = [row for row in earlier
           if (row.get("precipitation") or 0.0) >= RAIN_THRESHOLD_MM]
    if wet:
        previous["hours_since_rain"] = _round(
            (flight_start - wet[-1]["time"]).total_seconds() / 3600.0, 1)
        previous["hours_since_rain_at_least"] = None
    else:
        # No rain in what was fetched: dry for at least that long.
        previous["hours_since_rain"] = None
        dry = (flight_start - earlier[0]["time"]) if earlier else None
        previous["hours_since_rain_at_least"] = (
            None if dry is None else _round(dry.total_seconds() / 3600.0, 1))
    return {"at_flight": at_flight, "previous_24h": previous}


def fetch(lat: float, lon: float, flight_start: datetime,
          flight_end: datetime,
          fetch_json: Callable = env_remote.fetch_json,
          now: Optional[datetime] = None,
          log: Optional[Callable[[str], None]] = None) -> dict:
    """The weather document for a flight at ``(lat, lon)``.

    :raises env_remote.SourceUnavailable: the service cannot be reached
    """
    url = service_for(flight_start, now)
    soil = SOIL_VARIABLES[url]
    start = flight_start - timedelta(hours=LOOKBACK_HOURS)
    wanted = list(CORE_VARIABLES) + list(soil.values())
    service = "archive" if url == ARCHIVE_URL else "forecast service"
    if log:
        log(f"Asking the Open-Meteo {service} for "
            f"{start:%Y-%m-%d} to {flight_end:%Y-%m-%d} at "
            f"{lat:.4f}, {lon:.4f}")
    notes = []
    try:
        document = fetch_json(
            url, params=request_params(lat, lon, start, flight_end, wanted),
            what=f"the Open-Meteo {service}")
    except env_remote.SourceUnavailable as exc:
        # A service that does not know one of the soil layers refuses the
        # whole request; the weather itself is still worth having.
        if "HTTP 400" not in str(exc):
            raise
        document = fetch_json(
            url, params=request_params(lat, lon, start, flight_end,
                                       list(CORE_VARIABLES)),
            what=f"the Open-Meteo {service}")
        notes.append("Soil temperature and moisture were not available "
                     f"from the {service}.")
        if log:
            log(notes[-1])

    rows = parse_hourly(document, {theirs: ours
                                   for ours, theirs in soil.items()})
    if not rows:
        raise env_remote.SourceUnavailable(
            f"The Open-Meteo {service} returned no hourly values for the "
            "flight's date.")
    covered = [row for row in rows
               if row.get("temperature_2m") is not None]
    if not covered or covered[-1]["time"] < flight_start:
        raise env_remote.SourceUnavailable(
            f"The Open-Meteo {service} has no values yet for "
            f"{flight_start:%Y-%m-%d %H:%M} UTC. Try again in a few days.")

    summary = summarise(rows, flight_start, flight_end)
    return {
        "service": url,
        "attribution": ATTRIBUTION,
        "latitude": document.get("latitude", lat),
        "longitude": document.get("longitude", lon),
        "model_elevation_m": document.get("elevation"),
        "flight_start": iso(flight_start),
        "flight_end": iso(flight_end),
        "units": units(),
        "notes": notes,
        "at_flight": summary["at_flight"],
        "previous_24h": summary["previous_24h"],
        "hourly": [dict(row, time=iso(row["time"])) for row in rows],
    }


def write_hourly_csv(document: dict, path: str) -> str:
    """The hourly rows of a weather document as a CSV."""
    rows = document.get("hourly") or []
    names = ["time"] + sorted({name for row in rows for name in row
                               if name != "time"})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(names)
        for row in rows:
            writer.writerow(["" if row.get(name) is None else row.get(name)
                             for name in names])
    return path


def describe(document: dict) -> List[str]:
    """A few log lines saying what the weather was."""
    at = document.get("at_flight") or {}
    before = document.get("previous_24h") or {}

    def text(value, unit, digits=1):
        return "n/a" if value is None else f"{value:.{digits}f}{unit}"

    lines = [
        "At flight time: "
        f"air {text(at.get('temperature_2m'), ' °C')}, "
        f"soil {text(at.get('soil_temperature'), ' °C')}, "
        f"humidity {text(at.get('relative_humidity_2m'), ' %', 0)}, "
        f"dew point {text(at.get('dew_point_2m'), ' °C')}, "
        f"wind {text(at.get('wind_speed_10m'), ' m/s')}, "
        f"cloud {text(at.get('cloud_cover'), ' %', 0)}",
        "Previous 24 h: "
        f"rain {text(before.get('precipitation_sum'), ' mm')}, "
        f"sunshine {text(before.get('sunshine_hours'), ' h')}, "
        f"air {text(before.get('temperature_min'), '')} to "
        f"{text(before.get('temperature_max'), ' °C')}",
    ]
    return lines
