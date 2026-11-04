# -*- coding: utf-8 -*-
"""Heading source of the extracted poses (core.heading)."""
import pytest

from bambi_wildlife_detection.core import heading
from bambi_wildlife_detection.core.heading import (
    DEFAULT_HEADING_SOURCE, HEADING_SOURCE_LABELS, heading_check_message,
    legacy_project_heading_source, make_timed_pose_extractor, photo_heading_source,
    resolve_heading_source)


def test_sources_match_bambi_detection():
    # bambi_detection 1.1.0 HEADING_SOURCES; the first one is the default.
    assert list(HEADING_SOURCE_LABELS) == [
        "airdata_compass", "airdata_gimbal", "srt_gimbal", "srt_drone"]
    assert DEFAULT_HEADING_SOURCE == "airdata_compass"


@pytest.mark.parametrize("config,expected", [
    ({}, "airdata_compass"),
    ({"heading_source": "srt_gimbal"}, "srt_gimbal"),
    ({"heading_source": "airdata_gimbal"}, "airdata_gimbal"),
    ({"heading_source": "bogus"}, "airdata_compass"),
    ({"heading_source": None}, "airdata_compass"),
])
def test_resolve_heading_source(config, expected):
    assert resolve_heading_source(config) == expected


@pytest.mark.parametrize("source,expected", [
    ("airdata_compass", "airdata_compass"),
    ("airdata_gimbal", "airdata_gimbal"),
    ("srt_gimbal", "airdata_gimbal"),     # photos have no SRT
    ("srt_drone", "airdata_compass"),
])
def test_photo_mode_uses_the_flight_log_sources(source, expected):
    # bambi_detection PHOTO_HEADING_SOURCES
    assert photo_heading_source(source) == expected


def test_legacy_project_checkbox():
    assert legacy_project_heading_source("1") == "airdata_gimbal"
    assert legacy_project_heading_source("0") == "airdata_compass"
    assert legacy_project_heading_source("") == "airdata_compass"


def _poses(prefer, source="airdata_compass"):
    return {"heading_source": source, "heading_check": {
        "compass_heading_std_deg": 2.43, "gimbal_heading_std_deg": 0.0,
        "prefer_gimbal_heading": prefer, "recommended_heading_source": "airdata_gimbal" if prefer else None}}


def test_message_when_the_compass_wobbled():
    msg = heading_check_message(_poses(True), "Thermal")
    assert "Thermal" in msg and "2.43" in msg
    assert HEADING_SOURCE_LABELS["airdata_gimbal"] in msg


@pytest.mark.parametrize("poses", [
    _poses(False),
    _poses(True, source="airdata_gimbal"),   # already on the gimbal
    _poses(True, source="srt_gimbal"),
    {"images": []},                          # poses.json from bambi_detection < 1.1
])
def test_no_message(poses):
    assert heading_check_message(poses, "RGB") is None


def test_srt_drone_in_use_still_warns():
    assert heading_check_message(_poses(True, source="srt_drone"), "Thermal")


class _NewExtractor:
    def __init__(self, accessor, heading_source="airdata_compass", airdata_video_padding_ms=0, **kw):
        self.accessor, self.heading_source = accessor, heading_source
        self.padding, self.kw = airdata_video_padding_ms, kw


class _OldExtractor:  # bambi_detection 1.0.x signature
    def __init__(self, accessor, rel_transformer=None, drone_name=None, camera_name=None,
                 use_gimbal_heading=False):  # noqa: ARG002 - the 1.0.x argument
        pass


def test_factory_passes_source_and_margin():
    ex = make_timed_pose_extractor(
        _NewExtractor, "acc", {"heading_source": "srt_gimbal", "airdata_video_padding_ms": 3000},
        camera_name="T")
    assert (ex.accessor, ex.heading_source, ex.padding, ex.kw) == ("acc", "srt_gimbal", 3000, {"camera_name": "T"})


def test_factory_defaults_keep_old_poses():
    ex = make_timed_pose_extractor(_NewExtractor, "acc", {})
    assert ex.heading_source == "airdata_compass" and ex.padding == 0


def test_factory_names_the_outdated_install():
    with pytest.raises(RuntimeError, match="Dependency Manager"):
        make_timed_pose_extractor(_OldExtractor, "acc", {})


def test_factory_does_not_hide_other_type_errors():
    class Broken:
        def __init__(self, *a, **k):
            raise TypeError("something else")
    with pytest.raises(TypeError, match="something else"):
        make_timed_pose_extractor(Broken, "acc", {})


def test_module_imports_without_bambi():
    # core/ must stay importable in the stubbed unit tier.
    assert heading.__name__.endswith("core.heading")


def test_factory_takes_the_measured_margin():
    ex = make_timed_pose_extractor(_NewExtractor, "acc", {"airdata_video_padding_ms": 3000},
                                   padding_ms=2100)
    assert ex.padding == 2100
