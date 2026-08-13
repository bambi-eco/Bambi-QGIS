# -*- coding: utf-8 -*-
"""Unit tests for core.calibration_check (wrong-calibration guard)."""
import pytest

from bambi_wildlife_detection.bambi_calibrations import (
    RGB_CALIBRATIONS, THERMAL_CALIBRATIONS)
from bambi_wildlife_detection.core import calibration_check as cc


def _calib(cx, cy, f=1000.0):
    return {"mtx": [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]],
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0]}


# ---------------------------------------------------------------------------
# Inferring the calibration's native resolution
# ---------------------------------------------------------------------------

def test_resolution_is_twice_the_principal_point():
    assert cc.infer_calibration_resolution(_calib(640.0, 512.0)) == (1280.0, 1024.0)


@pytest.mark.parametrize("bad", [
    {},
    {"mtx": None},
    {"mtx": [[1, 2], [3, 4]]},
    {"mtx": [[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]},
])
def test_unusable_matrices_infer_nothing(bad):
    assert cc.infer_calibration_resolution(bad) is None


# ---------------------------------------------------------------------------
# The check itself
# ---------------------------------------------------------------------------

def test_matching_calibration_is_ok():
    severity, message = cc.check_calibration_resolution(
        _calib(640.0, 512.0), 1280, 1024)
    assert severity == "ok"
    assert message == ""


def test_small_offset_is_tolerated():
    # 5 % off — real calibrations never land exactly on the centre.
    severity, _ = cc.check_calibration_resolution(
        _calib(608.0, 486.0), 1280, 1024)
    assert severity == "ok"


def test_moderate_offset_warns_without_blocking():
    severity, message = cc.check_calibration_resolution(
        _calib(544.0, 435.0), 1280, 1024)
    assert severity == "warn"
    assert "1088x870" in message


def test_the_lion_flight_mismatch_is_fatal():
    """M3T thermal calibration applied to an M30T 1280x1024 stream."""
    severity, message = cc.check_calibration_resolution(
        THERMAL_CALIBRATIONS["DJI M3T (T;Video)"], 1280, 1024,
        label="thermal calibration")
    assert severity == "error"
    assert "631x517" in message
    assert "1280x1024" in message


def test_unknown_when_the_matrix_is_unreadable():
    severity, _ = cc.check_calibration_resolution({}, 1280, 1024)
    assert severity == "unknown"


def test_unknown_when_the_media_size_is_missing():
    severity, _ = cc.check_calibration_resolution(_calib(640.0, 512.0), 0, 0)
    assert severity == "unknown"


# ---------------------------------------------------------------------------
# Every built-in preset must pass against the media it is meant for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,width,height", [
    ("DJI M30T (T;Video)", 1280, 1024),
    ("DJI M3T (T;Video)", 640, 512),
    ("DJI M3T (T;Photo)", 640, 512),
    ("DJI M4T (T;Video)", 1280, 1024),
])
def test_thermal_presets_pass_on_their_own_sensor(name, width, height):
    severity, message = cc.check_calibration_resolution(
        THERMAL_CALIBRATIONS[name], width, height)
    assert severity == "ok", message


@pytest.mark.parametrize("name,width,height", [
    ("DJI M30T (W;Video)", 3840, 2160),
    ("DJI M3T (V;Video)", 3840, 2160),
    ("DJI M3T (V;Photo)", 4000, 3000),
    ("DJI M4T (V;Video)", 3840, 2160),
])
def test_rgb_presets_pass_on_their_own_sensor(name, width, height):
    severity, message = cc.check_calibration_resolution(
        RGB_CALIBRATIONS[name], width, height)
    assert severity == "ok", message


def test_thermal_preset_on_an_rgb_stream_is_fatal():
    severity, _ = cc.check_calibration_resolution(
        THERMAL_CALIBRATIONS["DJI M30T (T;Video)"], 3840, 2160)
    assert severity == "error"


# ---------------------------------------------------------------------------
# enforce_calibration_resolution
# ---------------------------------------------------------------------------

def test_enforce_raises_on_a_gross_mismatch(monkeypatch):
    monkeypatch.setattr(cc, "media_resolution", lambda paths: (1280, 1024))
    with pytest.raises(cc.CalibrationMismatchError) as excinfo:
        cc.enforce_calibration_resolution(
            THERMAL_CALIBRATIONS["DJI M3T (T;Video)"], ["x.mp4"],
            "thermal calibration")
    assert "631x517" in str(excinfo.value)


def test_enforce_can_be_overridden(monkeypatch):
    monkeypatch.setattr(cc, "media_resolution", lambda paths: (1280, 1024))
    logged = []
    cc.enforce_calibration_resolution(
        THERMAL_CALIBRATIONS["DJI M3T (T;Video)"], ["x.mp4"],
        "thermal calibration", log_fn=logged.append, allow_mismatch=True)
    assert logged and "Warning" in logged[0]


def test_enforce_is_silent_when_the_calibration_fits(monkeypatch):
    monkeypatch.setattr(cc, "media_resolution", lambda paths: (1280, 1024))
    logged = []
    cc.enforce_calibration_resolution(
        THERMAL_CALIBRATIONS["DJI M30T (T;Video)"], ["x.mp4"],
        "thermal calibration", log_fn=logged.append)
    assert logged == []


def test_enforce_skips_when_the_media_cannot_be_read(monkeypatch):
    monkeypatch.setattr(cc, "media_resolution", lambda paths: None)
    cc.enforce_calibration_resolution(
        THERMAL_CALIBRATIONS["DJI M3T (T;Video)"], ["missing.mp4"], "thermal")
