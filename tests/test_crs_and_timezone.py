# -*- coding: utf-8 -*-
"""Unit tests for core.crs_utils and core.timezone_detection."""
import pytest

from bambi_wildlife_detection.core.crs_utils import (
    detect_utm_from_airdata,
    is_valid_utm_crs,
    parse_epsg_from_text,
    utm_epsg_for_position,
)
from bambi_wildlife_detection.core.timezone_detection import (
    airdata_utc_hours,
    offset_from_exif,
    offset_from_srt,
    srt_local_hours,
    timezone_offset_hours,
)


class TestParseEpsg:
    @pytest.mark.parametrize("text,expected", [
        ("EPSG:32633", 32633),
        ("epsg:32633", 32633),
        (" 32633 ", 32633),
        ("EPSG: 32633", 32633),
        ("not-a-code", None),
        ("", None),
    ])
    def test_formats(self, text, expected):
        assert parse_epsg_from_text(text) == expected


class TestIsValidUtm:
    @pytest.mark.parametrize("text,expected", [
        ("EPSG:32601", True), ("EPSG:32660", True),   # northern range
        ("EPSG:32701", True), ("EPSG:32760", True),   # southern range
        ("EPSG:32600", False), ("EPSG:32661", False),
        ("EPSG:4326", False), ("EPSG:3035", False),
        ("garbage", False),
    ])
    def test_ranges(self, text, expected):
        assert is_valid_utm_crs(text) is expected


class TestUtmEpsgForPosition:
    def test_austria_is_zone_33n(self):
        assert utm_epsg_for_position(47.5, 13.2) == 32633

    def test_southern_hemisphere(self):
        assert utm_epsg_for_position(-33.9, 151.2) == 32756  # Sydney

    def test_zone_boundaries_clamped(self):
        assert utm_epsg_for_position(0.0, -180.0) == 32601
        assert utm_epsg_for_position(0.0, 180.0) == 32660   # clamped from 61


class TestDetectUtmFromAirdata:
    def test_detects_from_first_valid_row(self, tmp_path):
        csv = tmp_path / "flight.csv"
        csv.write_text(
            "time,latitude,longitude\n"
            "1,0,0\n"                    # zero position skipped
            "2,abc,def\n"                # unparseable skipped
            "3,47.5,13.2\n")
        logs = []
        assert detect_utm_from_airdata(str(csv), log_fn=logs.append) == "EPSG:32633"
        assert any("Detected UTM zone 33N" in m for m in logs)

    def test_no_gps_columns(self, tmp_path):
        csv = tmp_path / "flight.csv"
        csv.write_text("time,altitude\n1,100\n")
        logs = []
        assert detect_utm_from_airdata(str(csv), log_fn=logs.append) is None
        assert any("Could not find lat/lon columns" in m for m in logs)

    def test_missing_file(self, tmp_path):
        logs = []
        assert detect_utm_from_airdata(
            str(tmp_path / "nope.csv"), log_fn=logs.append) is None
        assert any("Error reading" in m for m in logs)


class TestTimezoneOffsetHours:
    def test_known_zones(self):
        assert timezone_offset_hours("UTC") == 0.0
        vienna = timezone_offset_hours("Europe/Vienna")
        assert vienna in (1.0, 2.0)      # CET / CEST
        assert timezone_offset_hours("Not/AZone") is None


def _write_airdata(tmp_path, flag="isVideo", rows=None):
    lines = [f"time,{flag},datetime(utc)"]
    for flag_val, dt in rows:
        lines.append(f"1,{flag_val},{dt}")
    path = tmp_path / "airdata.csv"
    path.write_text("\n".join(lines))
    return str(path)


def _write_srt(tmp_path, timestamps):
    lines = []
    for i, ts in enumerate(timestamps):
        lines += [str(i + 1), "00:00:00,000 --> 00:00:00,033", ts, ""]
    path = tmp_path / "video.srt"
    path.write_text("\n".join(lines))
    return str(path)


class TestAirdataUtcHours:
    def test_reads_flagged_rows_only(self, tmp_path):
        path = _write_airdata(tmp_path, rows=[
            ("1", "2023-09-20T08:30:00Z"),
            ("0", "2023-09-20T09:00:00Z"),      # not flagged
            ("true", "2023-09-20 08:45:00"),    # alt truthy + alt format
            ("false", "2023-09-20T10:00:00Z"),
        ])
        hours = airdata_utc_hours(path, "isvideo")
        assert hours == [pytest.approx(8.5), pytest.approx(8.75)]

    def test_missing_flag_column(self, tmp_path):
        path = _write_airdata(tmp_path, flag="other", rows=[("1", "x")])
        assert airdata_utc_hours(path, "isvideo") is None

    def test_missing_file(self, tmp_path):
        assert airdata_utc_hours(str(tmp_path / "x.csv"), "isvideo") is None
        assert airdata_utc_hours("", "isvideo") is None


class TestOffsetFromSrt:
    def test_flight6_style_cest_offset(self, tmp_path):
        # SRT local time 10:30 (CEST), AirData UTC 08:30 -> UTC+2 (the
        # flight-6 configuration used by the integration suite)
        srt = _write_srt(tmp_path, ["2023-09-20 10:30:00", "2023-09-20 10:30:01"])
        airdata = _write_airdata(tmp_path, rows=[
            ("1", "2023-09-20T08:30:00Z"), ("1", "2023-09-20T08:30:01Z")])
        assert offset_from_srt([srt], airdata) == 2.0

    def test_srt_timestamps_extracted(self, tmp_path):
        srt = _write_srt(tmp_path, ["2023-09-20 10:30:00"])
        assert srt_local_hours([srt]) == [pytest.approx(10.5)]
        assert srt_local_hours([str(tmp_path / "missing.srt")]) == []

    def test_no_srt_paths(self, tmp_path):
        airdata = _write_airdata(tmp_path, rows=[("1", "2023-09-20T08:30:00Z")])
        assert offset_from_srt([], airdata) is None

    def test_no_airdata(self, tmp_path):
        srt = _write_srt(tmp_path, ["2023-09-20 10:30:00"])
        assert offset_from_srt([srt], str(tmp_path / "nope.csv")) is None


class TestOffsetFromExif:
    def test_missing_dir_returns_none(self, tmp_path):
        assert offset_from_exif(str(tmp_path / "photos"), "x.csv") is None
        assert offset_from_exif("", "x.csv") is None

    def test_dir_without_photos_returns_none(self, tmp_path):
        photos = tmp_path / "photos"
        photos.mkdir()
        airdata = _write_airdata(
            tmp_path, flag="isPhoto", rows=[("1", "2023-09-20T08:30:00Z")])
        assert offset_from_exif(str(photos), airdata) is None


# ---------------------------------------------------------------------------
# Night flights
# ---------------------------------------------------------------------------

class TestNightFlights:
    """The offset was the difference between the mean *times of day* of the
    video and of the log. That goes wrong exactly when wildlife is flown."""

    @staticmethod
    def _flight(tmp_path, local_start, utc_start, minutes=30, name="video"):
        """A video of *minutes* and the log rows flagged as it, a second
        apart at the start of every minute."""
        from datetime import datetime, timedelta

        local = datetime.strptime(local_start, "%Y-%m-%d %H:%M:%S")
        utc = datetime.strptime(utc_start, "%Y-%m-%d %H:%M:%S")
        stamps, rows = [], []
        for minute in range(minutes):
            for second in (0, 30):
                step = timedelta(minutes=minute, seconds=second)
                stamps.append((local + step).strftime("%Y-%m-%d %H:%M:%S"))
                rows.append(("1", (utc + step).strftime("%Y-%m-%dT%H:%M:%SZ")))
        folder = tmp_path / name
        folder.mkdir()
        return _write_srt(folder, stamps), _write_airdata(folder, rows=rows)

    def test_a_flight_across_midnight(self, tmp_path):
        """23:45 to 00:15 local: the mean time of day is about noon, and
        the answer was UTC-10."""
        srt, airdata = self._flight(
            tmp_path, "2026-06-14 23:45:00", "2026-06-14 21:45:00")
        assert offset_from_srt([srt], airdata) == 2.0

    def test_a_flight_after_midnight(self, tmp_path):
        """00:30 local is still yesterday, 22:30, in UTC: it came out as
        UTC-22."""
        srt, airdata = self._flight(
            tmp_path, "2026-06-15 00:30:00", "2026-06-14 22:30:00")
        assert offset_from_srt([srt], airdata) == 2.0

    def test_a_flight_before_dawn_west_of_greenwich(self, tmp_path):
        """22:00 local in Denver is 04:00 the next day in UTC."""
        srt, airdata = self._flight(
            tmp_path, "2026-06-14 22:00:00", "2026-06-15 04:00:00")
        assert offset_from_srt([srt], airdata) == -6.0

    @pytest.mark.parametrize("local, utc, offset", [
        ("2026-01-10 16:15:00", "2026-01-10 10:45:00", 5.5),     # India
        ("2026-01-10 16:30:00", "2026-01-10 10:45:00", 5.75),    # Nepal
        ("2026-01-10 21:15:00", "2026-01-10 10:45:00", 10.5),    # Adelaide
        ("2026-01-11 01:00:00", "2026-01-10 12:00:00", 13.0),    # Tonga
    ])
    def test_offsets_that_are_no_whole_hours(self, tmp_path, local, utc,
                                             offset):
        """Rounding to whole hours put such a flight half an hour off."""
        srt, airdata = self._flight(tmp_path, local, utc)
        assert offset_from_srt([srt], airdata) == offset

    def test_one_video_of_a_longer_log(self, tmp_path):
        """The log flags 40 minutes, the SRT at hand covers ten of them.
        Several offsets put it inside; the whole hour is the time zone."""
        from datetime import datetime, timedelta

        _srt, airdata = self._flight(
            tmp_path, "2026-06-14 23:40:00", "2026-06-14 21:40:00",
            minutes=40)
        start = datetime(2026, 6, 14, 23, 55)
        part = _write_srt(tmp_path, [
            (start + timedelta(seconds=30 * i)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(20)])
        assert offset_from_srt([part], airdata) == 2.0

    def test_a_camera_clock_on_another_date_goes_by_the_clock_face(
            self, tmp_path):
        """Nothing coincides - the camera thinks it is 2020. The times of
        day are then compared the short way round: 23:45 and 21:45 are two
        hours apart, also when both run across midnight."""
        srt, airdata = self._flight(
            tmp_path, "2020-01-01 23:45:00", "2026-06-14 21:45:00")
        assert offset_from_srt([srt], airdata) == 2.0

    def test_the_day_flight_is_what_it_was(self, tmp_path):
        srt, airdata = self._flight(
            tmp_path, "2023-09-20 10:30:00", "2023-09-20 08:30:00")
        assert offset_from_srt([srt], airdata) == 2.0

    def test_nothing_to_compare_is_no_answer(self):
        from bambi_wildlife_detection.core import timezone_detection as tz

        assert tz.best_offset([], []) is None
        assert tz.detect_offset([], [], "", True) is None


class TestDetectOffset:
    def test_either_cameras_srt_will_do(self, tmp_path):
        """Only the RGB files were read: a thermal-only flight had no
        detection at all."""
        from bambi_wildlife_detection.core import timezone_detection as tz

        srt, airdata = TestNightFlights._flight(
            tmp_path, "2026-06-15 00:30:00", "2026-06-14 22:30:00")
        assert tz.detect_offset([srt], [], airdata, video_mode=True) == 2.0
        assert tz.detect_offset(["", srt], [], airdata, True) == 2.0
        assert tz.detect_offset([srt], [], airdata, video_mode=False) is None

    def test_the_signature_changes_with_the_files(self, tmp_path):
        import os
        from bambi_wildlife_detection.core import timezone_detection as tz

        srt, airdata = TestNightFlights._flight(
            tmp_path, "2026-06-15 00:30:00", "2026-06-14 22:30:00")
        folder = str(tmp_path / "photos")
        os.makedirs(folder)
        before = tz.inputs_signature([srt, folder, airdata, "", "/nope"])
        assert before == tz.inputs_signature(
            [srt, folder, airdata, "", "/nope"])
        with open(srt, "a") as handle:
            handle.write("\n2026-06-15 01:10:00\n")
        assert tz.inputs_signature([srt, folder, airdata]) != before[:3]

