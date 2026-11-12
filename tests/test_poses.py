# -*- coding: utf-8 -*-
"""Duplicated (frozen thermal) frames: the shared helpers and the places
that must leave such frames out (core.poses, 9.0.0)."""

import json
import os

import pytest

from bambi_wildlife_detection.core import flight_viewer as fv
from bambi_wildlife_detection.core import frame_matching, poses
from bambi_wildlife_detection.core import segmentation as seg
from bambi_wildlife_detection.core.exporters import common


def _images(n, duplicated=(), start=0.0, step=1.0):
    out = []
    for i in range(n):
        entry = {"imagefile": f"{i:06d}.jpg", "location": [float(i), 0.0, 50.0],
                 "rotation": [0.0, 0.0, 0.0],
                 "timestamp": f"2026-05-01T10:00:{start + i * step:06.3f}+00:00"}
        if i in duplicated:
            entry["duplicated"] = True
        out.append(entry)
    return out


def _write(folder, modality, images, **extra):
    path = os.path.join(str(folder), f"poses_{modality}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(images=images, **extra), fh)
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_only_a_true_flag_marks_a_frame():
    assert poses.is_duplicated({"duplicated": True})
    assert not poses.is_duplicated({"duplicated": False})
    assert not poses.is_duplicated({"imagefile": "a.jpg"})
    assert not poses.is_duplicated(None)


def test_duplicated_frames_are_indices_into_the_full_list():
    images = _images(6, duplicated=(2, 3))
    assert poses.duplicated_frames(images) == {2, 3}
    assert len(images) == 6                      # never shortened
    assert poses.duplicated_frames(None) == set()


def test_without_duplicated_keeps_order_and_numbers():
    images = _images(6, duplicated=(2, 3))
    assert poses.without_duplicated([5, 3, 2, 0], images) == [5, 0]
    assert poses.without_duplicated([0, 99], images) == [0, 99]


def test_nearest_usable_frame():
    images = _images(6, duplicated=(2, 3))
    assert poses.nearest_usable(images, 0) == 0
    assert poses.nearest_usable(images, 2) == 1      # earlier one on a tie
    assert poses.nearest_usable(images, 3) == 4
    assert poses.nearest_usable(_images(2, duplicated=(0, 1)), 0) is None
    assert poses.nearest_usable([], 0) is None


def test_load_images_is_cached_and_follows_the_file(tmp_path):
    path = _write(tmp_path, "t", _images(3, duplicated=(1,)))
    first = poses.load_images(str(tmp_path), "t")
    assert poses.load_images(str(tmp_path), "t") is first
    assert poses.flagged(str(tmp_path), "t") == {1}

    _write(tmp_path, "t", _images(4))
    os.utime(path, (1, 1))                           # a different mtime
    assert len(poses.load_images(str(tmp_path), "t")) == 4
    assert poses.load_images(str(tmp_path), "w") == []


def test_alfspy_agrees_on_the_key():
    basic = pytest.importorskip("alfspy.core.util.basic")
    if not hasattr(basic, "is_duplicated"):
        pytest.skip("alfspy older than 3.1.0")
    for entry in ({"duplicated": True}, {"duplicated": False}, {}):
        assert basic.is_duplicated(entry) == poses.is_duplicated(entry)


# ---------------------------------------------------------------------------
# Extraction: the version guard and the recorded setting
# ---------------------------------------------------------------------------

def test_the_flag_is_passed_on_when_asked_for():
    seen = {}

    def extract(folder, *, flag_duplicates=False, skip=0):
        seen.update(folder=folder, flag=flag_duplicates, skip=skip)

    poses.extract_with_flag(extract, "frames", flag_duplicates=True, skip=2)
    assert seen == {"folder": "frames", "flag": True, "skip": 2}


def test_without_the_flag_an_old_extractor_still_works():
    def old_extract(folder, skip=0):
        return "ok"

    assert poses.extract_with_flag(
        old_extract, "frames", flag_duplicates=False) == "ok"


def test_an_old_extractor_gets_a_clear_message():
    def old_extract(folder, skip=0):
        return "ok"

    with pytest.raises(RuntimeError, match="1.2.0"):
        poses.extract_with_flag(old_extract, "frames", flag_duplicates=True)


def test_another_type_error_is_not_swallowed():
    def extract(folder, flag_duplicates=False):
        raise TypeError("something else")

    with pytest.raises(TypeError, match="something else"):
        poses.extract_with_flag(extract, "frames", flag_duplicates=True)


def test_the_setting_is_recorded_in_the_poses_file(tmp_path):
    path = _write(tmp_path, "t", _images(5, duplicated=(1, 4)), origin=[1, 2])
    assert poses.flag_setting(str(tmp_path), "t") is False   # pre-9.0.0 file

    assert poses.record_flag_setting(path, True) == 2
    assert poses.flag_setting(str(tmp_path), "t") is True
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["origin"] == [1, 2] and len(data["images"]) == 5

    assert poses.flag_setting(str(tmp_path), "w") is None    # no file


# ---------------------------------------------------------------------------
# Frame matching
# ---------------------------------------------------------------------------

def test_a_duplicated_frame_is_never_a_partner():
    rgb = _images(6)
    thermal = _images(6, duplicated=(2, 3))
    matcher = frame_matching.FrameMatcher(rgb, thermal)
    assert matcher.match(2)[0] == 1
    assert matcher.match(3)[0] == 4
    assert matcher.match(5)[0] == 5
    assert matcher.matches_within(2, 0.1) is None    # nothing close enough
    assert matcher.matches_within(1, 0.1) == 1


def test_matching_still_works_without_any_flag():
    matcher = frame_matching.FrameMatcher(_images(4), _images(4))
    assert [matcher.match(i)[0] for i in range(4)] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Consumers
# ---------------------------------------------------------------------------

def test_segmentation_frames_leave_the_excluded_out():
    assert seg.select_frames(10, step=2, exclude={2, 4}) == [0, 6, 8]
    assert seg.select_frames(5) == [0, 1, 2, 3, 4]
    assert seg.duplicated_frames(_images(4, duplicated=(3,))) == {3}


def test_export_frames_carry_the_flag_and_rows_on_them_are_dropped(tmp_path):
    _write(tmp_path, "t", _images(4, duplicated=(2,)))
    frames = {f["frame"]: f for f in common.load_frames(str(tmp_path), "t")}
    assert [frames[i]["duplicated"] for i in range(4)] == \
        [False, False, True, False]

    rows = [{"frame": 1, "id": "a"}, {"frame": 2, "id": "b"},
            {"frame": 9, "id": "c"}]
    assert [r["id"] for r in common.without_duplicated(rows, frames)] == \
        ["a", "c"]


def test_the_alfs_window_leaves_duplicated_frames_out():
    assert fv.alfs_window(5, 20, 4, 1) == [3, 4, 5, 6, 7]
    assert fv.alfs_window(5, 20, 4, 1, exclude={4, 5}) == [3, 6, 7]


def test_the_scene_knows_its_duplicated_frames(tmp_path):
    _write(tmp_path, "t", _images(6, duplicated=(2, 3)))
    scene = fv.FlightScene.load(str(tmp_path), "t")
    assert len(scene) == 6
    assert scene.duplicated() == {2, 3}
    assert scene.is_duplicated(3) and not scene.is_duplicated(4)
    assert scene.usable_frame(3) == 4
    assert scene.usable_frame(0) == 0


def test_the_viewer_links_cameras_past_a_frozen_thermal_frame():
    rgb = _images(6)
    thermal = _images(6, duplicated=(2,))
    assert fv.matching_frame(rgb, thermal, 2) in (1, 3)
    assert fv.matching_frame(rgb, thermal, 4) == 4


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------

def test_the_steps_drop_duplicated_frames_and_say_so():
    from bambi_wildlife_detection import bambi_processing

    log = []
    images = _images(6, duplicated=(2, 3))
    assert bambi_processing._drop_duplicated(
        [0, 2, 3, 4], images, log.append) == [0, 4]
    assert "2 duplicated" in log[0]

    log.clear()
    assert bambi_processing._drop_duplicated([0, 1], images, log.append) == [0, 1]
    assert log == []


def test_the_setting_is_part_of_the_configuration_and_on_by_default():
    from bambi_wildlife_detection.core import config_schema

    entry = next(e for e in config_schema.CONFIG_ENTRIES
                 if e.key == "Extraction/FlagDuplicates")
    assert entry.kind == "bool" and entry.default is True
    assert config_schema.WIDGET_BINDINGS["Extraction/FlagDuplicates"] == \
        ("flag_duplicates_check", "check")


def test_the_pins_cover_the_duplicated_flag():
    packaging = pytest.importorskip("packaging.version")
    from bambi_wildlife_detection.core import dependency_ops as ops

    assert packaging.Version(ops._VERSION_RANGES["bambi-detection"][0]) >= \
        packaging.Version("1.2.1")
    assert packaging.Version(ops._VERSION_RANGES["AlfsPy"][0]) >= \
        packaging.Version("3.1.0")
    assert ops.BAMBI_DETECTION_TAG == "v1.2.1"
    assert ops.ALFS_PY_TAG == "v3.1.0"
