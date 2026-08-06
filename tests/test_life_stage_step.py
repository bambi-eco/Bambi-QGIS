# -*- coding: utf-8 -*-
"""The C4 life-stage step, end to end.

Needs no models: size is geometry. What is checked here is that the step picks
the right areas (metric where they exist), honours a sex call, and records the
evidence behind each verdict.
"""
import os

import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import (
    classification_store, life_stage, store, track_store,
)

# One juvenile among a tightly clustered herd, in pixel areas.
HERD = [400, 1000, 1010, 1020, 1030, 1040, 1050]


@pytest.fixture
def flight(tmp_path):
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()

    conn = store.open_store(
        store.stage_path(root, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    with store.transaction(conn):
        for index, area in enumerate(HERD):
            side = area ** 0.5
            # Three detections per track, so the median is well defined.
            for _ in range(3):
                conn.execute(
                    "INSERT INTO detections (frame, x1, y1, x2, y2, "
                    "confidence, species_id, source_id) "
                    "VALUES (?, 0, 0, ?, ?, 0.9, 0, 1)",
                    (index, side, side))
    conn.close()

    members = []
    detection_id = 1
    for index in range(len(HERD)):
        for _ in range(3):
            members.append({"track_id": index + 1,
                            "detection_id": detection_id})
            detection_id += 1
    track_store.record_tracks(root, "t", members)
    return root


def _config(root, **overrides):
    config = {
        "target_folder": root,
        "classification_camera": "T",
        "life_stage_z": -2.0,
        "life_stage_iqr_factor": 2.0,
        "life_stage_min_individuals": 4,
    }
    config.update(overrides)
    return config


def _run(root, **overrides):
    logs = []
    BambiProcessor().run_life_stage(_config(root, **overrides),
                                    log_fn=logs.append)
    return logs


def _labels(root):
    return {row["track_id"]: row["label"] for row in
            classification_store.track_predictions(
                root, "t", classification_store.LIFE_STAGE)}


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def test_the_juvenile_is_flagged_and_the_rest_are_adults(flight):
    _run(flight)

    labels = _labels(flight)
    assert labels[1] == life_stage.JUVENILE
    assert all(labels[track] == life_stage.ADULT
               for track in range(2, len(HERD) + 1))


def test_every_individual_gets_a_row(flight):
    _run(flight)
    assert len(_labels(flight)) == len(HERD)


def test_the_evidence_behind_the_call_is_kept(flight):
    _run(flight)

    row = next(r for r in classification_store.track_predictions(
        flight, "t", classification_store.LIFE_STAGE) if r["track_id"] == 1)
    evidence = row["evidence"]
    for field in ("area", "z", "gap", "cohort_ratio", "frames", "source"):
        assert field in evidence
    assert evidence["z"] < -2.0
    assert evidence["cohort_ratio"] < 0.5


def test_size_is_one_measurement_not_a_vote(flight):
    """Recorded as a unanimous call of one, so the table stays uniform."""
    _run(flight)
    row = classification_store.track_predictions(
        flight, "t", classification_store.LIFE_STAGE)[0]
    assert (row["votes"], row["n"], row["fraction"]) == (1, 1, 1.0)


def test_the_log_names_the_juvenile(flight):
    logs = _run(flight)
    assert any("juvenile" in line and "track 1" in line for line in logs)


# ---------------------------------------------------------------------------
# Which areas
# ---------------------------------------------------------------------------

def test_pixel_areas_are_used_without_geo_referencing(flight):
    _run(flight)
    row = classification_store.track_predictions(
        flight, "t", classification_store.LIFE_STAGE)[0]
    assert row["evidence"]["source"] == life_stage.AREA_PIXEL


def test_metric_areas_are_preferred_when_available(flight):
    """Orthorectified areas are metric, so they are the better cue."""
    conn = store.open_store(
        store.stage_path(flight, store.GEOREFERENCED, "t"),
        store.GEOREFERENCED, "t")
    with store.transaction(conn):
        for detection_id in range(1, len(HERD) * 3 + 1):
            track = (detection_id - 1) // 3
            side = (HERD[track] ** 0.5) / 100.0
            conn.execute(
                "INSERT INTO detections_geo (detection_id, gx1, gy1, gz1, "
                "gx2, gy2, gz2) VALUES (?, 0, 0, 0, ?, ?, 0)",
                (detection_id, side, side))
    conn.close()

    _run(flight)
    row = classification_store.track_predictions(
        flight, "t", classification_store.LIFE_STAGE)[0]
    assert row["evidence"]["source"] == life_stage.AREA_GEO
    # Same verdict, different units.
    assert _labels(flight)[1] == life_stage.JUVENILE


# ---------------------------------------------------------------------------
# Interaction with sex
# ---------------------------------------------------------------------------

def test_a_male_is_an_adult_whatever_his_box_says(flight):
    """The cue that marked him is antlers, not size."""
    classification_store.record_track_predictions(flight, "t", "sex", [
        {"track_id": 1, "label": "male", "votes": 9, "n": 10,
         "fraction": 0.9}])
    _run(flight)

    assert _labels(flight)[1] == life_stage.ADULT


def test_a_female_call_does_not_protect_a_juvenile(flight):
    """'female_juvenile' is exactly the class size is meant to resolve."""
    classification_store.record_track_predictions(flight, "t", "sex", [
        {"track_id": 1, "label": "female_juvenile", "votes": 9, "n": 10,
         "fraction": 0.9}])
    _run(flight)

    assert _labels(flight)[1] == life_stage.JUVENILE


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_untracked_modality_is_refused(flight):
    with pytest.raises(ValueError, match="No RGB tracks"):
        BambiProcessor().run_life_stage(
            _config(flight, classification_camera="W"))


def test_too_few_individuals_writes_nothing_and_says_so(flight):
    logs = _run(flight, life_stage_min_individuals=99)

    assert classification_store.track_predictions(
        flight, "t", classification_store.LIFE_STAGE) == []
    assert any("too few individuals" in line for line in logs)


def test_thresholds_are_honoured(flight):
    _run(flight, life_stage_z=-99.0)
    assert all(label == life_stage.ADULT for label in _labels(flight).values())


def test_rerunning_replaces_rather_than_appends(flight):
    _run(flight)
    _run(flight)
    assert len(_labels(flight)) == len(HERD)


def test_the_verdict_reaches_the_track_age_field(flight):
    from bambi_wildlife_detection.core import label_store

    _run(flight)
    juvenile = label_store.vocabulary(flight)["enum_ids"]["age"]["juvenile"]

    conn = store.open_store(
        store.stage_path(flight, store.TRACKS, "t"), store.TRACKS, "t")
    try:
        attributes = {int(r["track_id"]): r["attributes"]
                      for r in conn.execute(
                          "SELECT track_id, attributes FROM tracks")}
    finally:
        conn.close()

    import json as _json
    assert _json.loads(attributes[1])["age"] == juvenile


def test_writing_can_be_switched_off(flight):
    _run(flight, classification_write_results=False)
    conn = store.open_store(
        store.stage_path(flight, store.TRACKS, "t"), store.TRACKS, "t")
    try:
        rows = [r["attributes"] for r in conn.execute(
            "SELECT attributes FROM tracks")]
    finally:
        conn.close()
    assert all(not row for row in rows)
    # Still recorded, just not projected.
    assert _labels(flight)[1] == life_stage.JUVENILE


def test_the_stage_is_recorded(flight):
    from bambi_wildlife_detection.core import stages

    _run(flight)
    assert stages.states(flight, "t")["life_stage"]["state"] == stages.COMPLETE
