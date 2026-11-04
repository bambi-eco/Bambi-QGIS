# -*- coding: utf-8 -*-
"""Automatic flight log margin (core.log_margin)."""
import datetime

import pytest

from bambi_wildlife_detection.core import log_margin
from bambi_wildlife_detection.core.log_margin import (
    AUTO_MARGIN_BUFFER_MS, AUTO_MARGIN_MAX_MS, auto_margin_ms, measure_auto_margin_ms,
    resolve_margin_ms)

T0 = datetime.datetime(2025, 12, 8, 19, 38, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=-6)))


def _t(ms):
    return T0 + datetime.timedelta(milliseconds=ms)


@pytest.mark.parametrize("flags,video,overhang", [
    ((1100, 50000), (0, 50000), 1100),      # flag 1.1 s late (bat flight)
    ((0, 50000), (200, 49000), 0),          # flags cover the video
    ((0, 48500), (0, 50000), 1500),         # flag ends early
    ((1100, 48500), (0, 50000), 1500),      # the larger side counts
])
def test_margin_covers_the_overhang(flags, video, overhang):
    margin, measured = auto_margin_ms(_t(flags[0]), _t(flags[1]), _t(video[0]), _t(video[1]))
    assert measured == overhang
    assert margin == overhang + AUTO_MARGIN_BUFFER_MS


def test_margin_is_capped():
    margin, overhang = auto_margin_ms(_t(100_000), _t(200_000), _t(0), _t(200_000))
    assert (margin, overhang) == (AUTO_MARGIN_MAX_MS, 100_000)


class _Frame:
    def __init__(self, ms, attr):
        setattr(self, attr, _t(ms))


class _FakeExtractor:
    """TimedPoseExtractor's two static readers, as bambi_detection 1.1.0 has them."""
    calls = []

    @staticmethod
    def get_srt_frames(srt_files, timezone=None):
        _FakeExtractor.calls.append(("srt", tuple(srt_files), timezone))
        return [_Frame(0, "timestamp"), _Frame(50000, "timestamp")], 1, [0, 0]

    @staticmethod
    def get_air_data_frames(air_data_file, video_timestamp, timezone=None, padding_ms=0):
        _FakeExtractor.calls.append(("ad", air_data_file, video_timestamp, timezone, padding_ms))
        return None, [_Frame(1100, "datetime"), _Frame(50000, "datetime")]


def test_measure_reads_srt_and_flagged_log_rows():
    _FakeExtractor.calls = []
    assert measure_auto_margin_ms(_FakeExtractor, "ad.csv", ("a.srt",), "TZ") == (2100, 1100)
    assert _FakeExtractor.calls == [("srt", ("a.srt",), "TZ"), ("ad", "ad.csv", _t(0), "TZ", 0)]


def test_resolve_fixed_margin_reads_nothing():
    class Fails:
        @staticmethod
        def get_srt_frames(*a, **k):
            raise AssertionError("must not read files for a fixed margin")
    assert resolve_margin_ms({"airdata_video_padding_ms": 3000}, Fails, "ad.csv", [], None, "T") == 3000
    assert resolve_margin_ms({}, Fails, "ad.csv", [], None, "T") == 0


def test_resolve_auto_margin_is_logged():
    logs = []
    margin = resolve_margin_ms({"airdata_video_padding_auto": True, "airdata_video_padding_ms": 0},
                               _FakeExtractor, "ad.csv", ["a.srt"], None, "Thermal", log_fn=logs.append)
    assert margin == 2100
    assert logs and "Thermal" in logs[0] and "2.1 s" in logs[0] and "1.1 s" in logs[0]


def test_resolve_auto_margin_warns_when_capped(monkeypatch):
    monkeypatch.setattr(log_margin, "measure_auto_margin_ms",
                        lambda *a: (AUTO_MARGIN_MAX_MS, 90_000))
    logs = []
    assert resolve_margin_ms({"airdata_video_padding_auto": True}, _FakeExtractor, "ad.csv", [],
                             None, "RGB", log_fn=logs.append) == AUTO_MARGIN_MAX_MS
    assert "Warning (RGB)" in logs[0] and "timezone" in logs[0]


def test_measure_rejects_an_empty_srt():
    class Empty(_FakeExtractor):
        @staticmethod
        def get_srt_frames(srt_files, timezone=None):
            return [], 0, []
    with pytest.raises(ValueError, match="no frames"):
        measure_auto_margin_ms(Empty, "ad.csv", ["a.srt"], None)
