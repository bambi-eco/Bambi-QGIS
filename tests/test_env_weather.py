# -*- coding: utf-8 -*-
"""Weather source (core/env_weather.py)."""
import csv
from datetime import datetime, timedelta, timezone

import pytest

from bambi_wildlife_detection.core import env_remote
from bambi_wildlife_detection.core import env_weather as weather

START = datetime(2026, 6, 10, 3, 10, tzinfo=timezone.utc)
END = datetime(2026, 6, 10, 3, 50, tzinfo=timezone.utc)
NOW = datetime(2026, 11, 12, tzinfo=timezone.utc)


def _document(url=weather.ARCHIVE_URL, hours=96, first=None, rain_at=()):
    """Hourly answer starting 72 h before the flight's day."""
    first = first or datetime(2026, 6, 7, 0, 0, tzinfo=timezone.utc)
    times = [first + timedelta(hours=h) for h in range(hours)]
    soil = weather.SOIL_VARIABLES[url]
    hourly = {
        "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
        "temperature_2m": [10.0 + t.hour for t in times],
        "relative_humidity_2m": [80.0] * hours,
        "dew_point_2m": [8.0] * hours,
        "precipitation": [2.0 if t in rain_at else 0.0 for t in times],
        "snowfall": [0.0] * hours,
        "snow_depth": [0.0] * hours,
        "cloud_cover": [50.0] * hours,
        "wind_speed_10m": [3.0] * hours,
        "wind_gusts_10m": [5.0 + (t.hour == 15) * 7.0 for t in times],
        "wind_direction_10m": [350.0 if t.hour % 2 else 10.0 for t in times],
        "shortwave_radiation": [100.0] * hours,
        "sunshine_duration": [1800.0] * hours,
        "surface_pressure": [980.0] * hours,
        soil["soil_temperature"]: [12.0] * hours,
        soil["soil_moisture"]: [0.3] * hours,
    }
    return {"latitude": 48.3, "longitude": 14.3, "elevation": 270.0,
            "hourly": hourly}


class TestService:
    def test_old_flights_use_the_archive_recent_ones_the_forecast(self):
        assert weather.service_for(START, NOW) == weather.ARCHIVE_URL
        recent = NOW - timedelta(days=2)
        assert weather.service_for(recent, NOW) == weather.FORECAST_URL

    def test_request_is_utc_metres_per_second_and_whole_days(self):
        params = weather.request_params(48.30814, 14.29187, START, END,
                                        ["temperature_2m", "precipitation"])
        assert params["start_date"] == params["end_date"] == "2026-06-10"
        assert params["timezone"] == "GMT"
        assert params["wind_speed_unit"] == "ms"
        assert params["hourly"] == "temperature_2m,precipitation"


class TestSummary:
    def test_values_are_interpolated_to_the_middle_of_the_flight(self):
        rows = weather.parse_hourly(_document())
        summary = weather.summarise(rows, START, END)
        # 03:30 lies halfway between 13 °C (03:00) and 14 °C (04:00)
        assert summary["at_flight"]["temperature_2m"] == pytest.approx(13.5)
        assert summary["at_flight"]["dew_point_spread"] == pytest.approx(5.5)
        assert summary["at_flight"]["relative_humidity_2m"] == 80.0

    def test_wind_direction_goes_the_short_way_round_north(self):
        rows = weather.parse_hourly(_document())
        direction = weather.summarise(rows, START, END)["at_flight"][
            "wind_direction_10m"]
        assert direction == pytest.approx(0.0, abs=1e-6) or \
            direction == pytest.approx(360.0, abs=1e-6)

    def test_previous_day_sums_and_extremes(self):
        rain = (datetime(2026, 6, 9, 12, tzinfo=timezone.utc),
                datetime(2026, 6, 9, 13, tzinfo=timezone.utc),
                # more than 24 h before the first frame: not in the sum
                datetime(2026, 6, 9, 3, tzinfo=timezone.utc))
        rows = weather.parse_hourly(_document(rain_at=rain))
        before = weather.summarise(rows, START, END)["previous_24h"]
        assert before["hours"] == 24
        assert before["precipitation_sum"] == pytest.approx(4.0)
        assert before["sunshine_hours"] == pytest.approx(12.0)
        assert before["temperature_min"] == 10.0
        assert before["temperature_max"] == 33.0
        assert before["wind_gusts_max"] == 12.0
        assert before["cloud_cover_mean"] == 50.0
        assert before["hours_since_rain"] == pytest.approx(14.2, abs=0.05)

    def test_no_rain_in_the_window_is_reported_as_at_least(self):
        rows = weather.parse_hourly(_document())
        before = weather.summarise(rows, START, END)["previous_24h"]
        assert before["hours_since_rain"] is None
        assert before["hours_since_rain_at_least"] == pytest.approx(75.2, abs=0.05)

    def test_soil_names_are_mapped_back(self):
        document = _document(weather.FORECAST_URL)
        soil = weather.SOIL_VARIABLES[weather.FORECAST_URL]
        rows = weather.parse_hourly(
            document, {theirs: ours for ours, theirs in soil.items()})
        assert rows[0]["soil_temperature"] == 12.0
        assert "soil_temperature_0cm" not in rows[0]

    def test_missing_values_stay_missing(self):
        document = _document()
        document["hourly"]["cloud_cover"] = [None] * 96
        summary = weather.summarise(weather.parse_hourly(document), START, END)
        assert summary["at_flight"]["cloud_cover"] is None
        assert summary["previous_24h"]["cloud_cover_mean"] is None


class TestFetch:
    def test_document_for_a_flight(self, tmp_path):
        calls = []

        def fetch_json(url, params=None, what=""):
            calls.append((url, params))
            return _document()

        log = []
        document = weather.fetch(48.3, 14.3, START, END, fetch_json=fetch_json,
                                 now=NOW, log=log.append)
        url, params = calls[0]
        assert url == weather.ARCHIVE_URL
        assert params["start_date"] == "2026-06-07"
        assert "soil_temperature_0_to_7cm" in params["hourly"]
        assert document["at_flight"]["soil_temperature"] == 12.0
        assert document["flight_start"] == "2026-06-10T03:10:00Z"
        assert document["hourly"][0]["time"] == "2026-06-07T00:00:00Z"
        assert document["notes"] == []
        assert "archive" in log[0]
        lines = weather.describe(document)
        assert "air 13.5 °C" in lines[0] and "rain 0.0 mm" in lines[1]

        path = weather.write_hourly_csv(document, str(tmp_path / "h.csv"))
        with open(path, encoding="utf-8") as fh:
            table = list(csv.DictReader(fh))
        assert len(table) == 96 and table[3]["temperature_2m"] == "13.0"

    def test_a_refused_soil_layer_costs_only_the_soil(self):
        calls = []

        def fetch_json(url, params=None, what=""):
            calls.append(params["hourly"])
            if "soil" in params["hourly"]:
                raise env_remote.SourceUnavailable(
                    "service answered HTTP 400: invalid variable.")
            document = _document()
            for name in list(document["hourly"]):
                if name.startswith("soil"):
                    del document["hourly"][name]
            return document

        document = weather.fetch(48.3, 14.3, START, END,
                                 fetch_json=fetch_json, now=NOW)
        assert len(calls) == 2 and "soil" not in calls[1]
        assert document["at_flight"]["temperature_2m"] == pytest.approx(13.5)
        assert document["at_flight"].get("soil_temperature") is None
        assert "Soil" in document["notes"][0]

    def test_an_unreachable_service_fails(self):
        def fetch_json(url, params=None, what=""):
            raise env_remote.SourceUnavailable(
                "the Open-Meteo archive answered HTTP 429: limit exceeded.")

        with pytest.raises(env_remote.SourceUnavailable, match="429"):
            weather.fetch(48.3, 14.3, START, END, fetch_json=fetch_json,
                          now=NOW)

    def test_a_date_the_service_has_not_reached_fails(self):
        early = _document(hours=24)   # ends three days before the flight
        with pytest.raises(env_remote.SourceUnavailable, match="no values yet"):
            weather.fetch(48.3, 14.3, START, END,
                          fetch_json=lambda *a, **k: early, now=NOW)

    def test_an_empty_answer_fails(self):
        with pytest.raises(env_remote.SourceUnavailable, match="no hourly"):
            weather.fetch(48.3, 14.3, START, END,
                          fetch_json=lambda *a, **k: {"hourly": {}}, now=NOW)
