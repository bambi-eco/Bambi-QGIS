# -*- coding: utf-8 -*-
"""Carrying labels across the match.

A stage labels the camera it reads, so the two sides can easily end up knowing
different things about the same animal. This is the bridge, and what matters is
what it refuses to do: it fills gaps, it never overrules, and it never invents
a classification run on the camera the label did not come from.
"""
import json

import pytest

from bambi_wildlife_detection.core import (
    classification_store, label_sync, match_store, store, track_store,
)

RED_DEER = 1
ROE_DEER = 2


@pytest.fixture
def flight(tmp_path):
    """Two animals, seen by both cameras and matched, with nothing labelled."""
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()

    for suffix in ("t", "w"):
        conn = store.open_store(
            store.stage_path(root, store.DETECTIONS, suffix),
            store.DETECTIONS, suffix)
        with store.transaction(conn):
            for detection_id in range(1, 5):
                conn.execute(
                    "INSERT INTO detections (frame, x1, y1, x2, y2, "
                    "confidence, species_id, source_id) "
                    "VALUES (?, 0, 0, 10, 10, 0.9, 0, 1)", (detection_id - 1,))
        conn.close()
        track_store.record_tracks(root, suffix, [
            {"track_id": 1 + (detection_id > 2), "detection_id": detection_id}
            for detection_id in range(1, 5)])

    match_store.record_matches(root, [
        {"track_id_t": track, "track_id_w": track, "shared": 2,
         "median_dist": 1.0, "conf_t": 0.9, "conf_w": 0.9,
         "pairs": [{"frame_t": d - 1, "frame_w": d - 1,
                    "detection_id_t": d, "detection_id_w": d, "dist": 1.0}
                   for d in ((1, 2) if track == 1 else (3, 4))]}
        for track in (1, 2)])
    return root


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_track(root, modality, track_id, species=None, attributes=None):
    path = store.stage_path(root, store.TRACKS, modality)
    conn = store.open_store(path, store.TRACKS, modality)
    try:
        with store.transaction(conn):
            if species is not None:
                conn.execute(
                    "UPDATE tracks SET species_id = ? WHERE track_id = ?",
                    (species, track_id))
            if attributes is not None:
                conn.execute(
                    "UPDATE tracks SET attributes = ? WHERE track_id = ?",
                    (json.dumps(attributes), track_id))
    finally:
        conn.close()


def _tracks(root, modality):
    path = store.stage_path(root, store.TRACKS, modality)
    conn = store.open_store(path, store.TRACKS, modality)
    try:
        return {int(row["track_id"]): (row["species_id"],
                                       json.loads(row["attributes"] or "{}"))
                for row in conn.execute(
                    "SELECT track_id, species_id, attributes FROM tracks")}
    finally:
        conn.close()


def _set_detection(root, modality, detection_id, species=None,
                   attributes=None):
    path = store.stage_path(root, store.DETECTIONS, modality)
    conn = store.open_store(path, store.DETECTIONS, modality)
    try:
        with store.transaction(conn):
            if species is not None:
                conn.execute(
                    "UPDATE detections SET species_id = ? "
                    "WHERE detection_id = ?", (species, detection_id))
            if attributes is not None:
                conn.execute(
                    "UPDATE detections SET attributes = ? "
                    "WHERE detection_id = ?",
                    (json.dumps(attributes), detection_id))
    finally:
        conn.close()


def _detections(root, modality):
    path = store.stage_path(root, store.DETECTIONS, modality)
    conn = store.open_store(path, store.DETECTIONS, modality)
    try:
        return {int(row["detection_id"]): (row["species_id"],
                                           json.loads(row["attributes"]
                                                      or "{}"))
                for row in conn.execute(
                    "SELECT detection_id, species_id, attributes "
                    "FROM detections")}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# What travels
# ---------------------------------------------------------------------------

def test_species_reaches_the_matched_partner(flight):
    _set_track(flight, "w", 1, species=RED_DEER)

    label_sync.sync_all(flight, "w", "t")

    assert _tracks(flight, "t")[1][0] == RED_DEER


def test_sex_and_age_travel_too(flight):
    _set_track(flight, "w", 1, attributes={"sex": 3, "age": 7})

    label_sync.sync_all(flight, "w", "t")

    assert _tracks(flight, "t")[1][1] == {"sex": 3, "age": 7}


def test_species_also_lands_on_the_detections(flight):
    """The exporters and the analytics filter on the detection's species, so a
    track-only answer would be invisible to them."""
    _set_detection(flight, "w", 1, species=RED_DEER)

    label_sync.sync_all(flight, "w", "t")

    assert _detections(flight, "t")[1][0] == RED_DEER


def test_occlusion_travels_per_frame(flight):
    _set_detection(flight, "w", 1, attributes={"occlusion": 1})

    label_sync.sync_all(flight, "w", "t")

    assert _detections(flight, "t")[1][1]["occlusion"] == 1


def test_it_works_in_both_directions(flight):
    _set_track(flight, "t", 2, species=ROE_DEER)

    label_sync.sync_all(flight, "t", "w")

    assert _tracks(flight, "w")[2][0] == ROE_DEER


# ---------------------------------------------------------------------------
# What it refuses to do
# ---------------------------------------------------------------------------

def test_an_existing_answer_is_never_replaced(flight):
    """Sync is for the gaps. Whatever is already there was put there by
    something that looked at this camera, which outranks a copy."""
    _set_track(flight, "w", 1, species=RED_DEER)
    _set_track(flight, "t", 1, species=ROE_DEER)

    label_sync.sync_all(flight, "w", "t")

    assert _tracks(flight, "t")[1][0] == ROE_DEER


def test_an_existing_attribute_is_never_replaced(flight):
    _set_track(flight, "w", 1, attributes={"sex": 3})
    _set_track(flight, "t", 1, attributes={"sex": 4})

    label_sync.sync_all(flight, "w", "t")

    assert _tracks(flight, "t")[1][1]["sex"] == 4


def test_unmatched_animals_are_left_alone(flight):
    """An animal one camera never saw has no partner to copy from."""
    track_store.record_tracks(flight, "t", [{"track_id": 9,
                                             "detection_id": 1}])
    _set_track(flight, "w", 1, species=RED_DEER)

    label_sync.sync_all(flight, "w", "t")

    assert _tracks(flight, "t").get(9, (0, {}))[0] in (0, None)


def test_nothing_is_written_into_the_classification_store(flight):
    """A synced label was not observed on this camera, so recording it as a
    prediction there would invent a run that never happened."""
    _set_track(flight, "w", 1, species=RED_DEER)

    label_sync.sync_all(flight, "w", "t")

    assert classification_store.track_predictions(
        flight, "t", "species") == []
    assert not classification_store.has_store(flight, "t")


def test_syncing_onto_the_same_camera_is_refused(flight):
    with pytest.raises(ValueError, match="not onto the camera"):
        label_sync.sync_all(flight, "t", "t")


def test_an_unknown_modality_is_refused(flight):
    with pytest.raises(ValueError, match="Unknown modality"):
        label_sync.sync_all(flight, "t", "x")


def test_a_flight_with_no_matches_syncs_nothing(tmp_path):
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()

    counts = label_sync.sync_all(root, "w", "t")

    assert not any(counts.values())


def test_it_is_idempotent(flight):
    _set_track(flight, "w", 1, species=RED_DEER)

    label_sync.sync_all(flight, "w", "t")
    second = label_sync.sync_all(flight, "w", "t")

    assert second["species"] == 0
    assert _tracks(flight, "t")[1][0] == RED_DEER


# ---------------------------------------------------------------------------
# What a reviewer said stands
# ---------------------------------------------------------------------------

def test_a_reviewer_who_said_not_known_is_not_overruled(flight):
    """Sync fills gaps - but "nobody can tell", said by a person who looked,
    is an answer and not a gap."""
    from bambi_wildlife_detection.core import track_review

    _set_track(flight, "w", 1, attributes={"sex": 3, "age": 7})
    track_review.set_attribute(flight, "t", 1, "sex", None)

    counts = label_sync.sync_all(flight, "w", "t")

    _species, attributes = _tracks(flight, "t")[1]
    assert "sex" not in attributes and attributes["age"] == 7
    assert (counts["sex"], counts["age"]) == (0, 1)


def test_a_species_left_open_by_hand_stays_open_on_track_and_boxes(flight):
    from bambi_wildlife_detection.core import track_review

    for track, species, boxes in ((1, RED_DEER, (1, 2)),
                                  (2, ROE_DEER, (3, 4))):
        _set_track(flight, "w", track, species=species)
        for box in boxes:
            _set_detection(flight, "w", box, species=species)
    track_review.set_species(flight, "t", 1, store.FALLBACK_SPECIES_ID)

    counts = label_sync.sync_all(flight, "w", "t")

    tracks = _tracks(flight, "t")
    assert tracks[1][0] == store.FALLBACK_SPECIES_ID
    assert tracks[2][0] == ROE_DEER              # nobody said anything of it
    boxes = _detections(flight, "t")
    assert [boxes[box][0] for box in (1, 2, 3, 4)] == [
        store.FALLBACK_SPECIES_ID, store.FALLBACK_SPECIES_ID,
        ROE_DEER, ROE_DEER]
    assert (counts["species"], counts["species_detections"]) == (1, 2)


def test_what_a_reviewer_said_travels_like_any_other_answer(flight):
    from bambi_wildlife_detection.core import track_review

    track_review.set_attribute(flight, "w", 1, "sex", 3)

    label_sync.sync_all(flight, "w", "t")

    # the answer goes across; the mark stays with the camera it was set on
    assert _tracks(flight, "t")[1][1] == {"sex": 3}


# ---------------------------------------------------------------------------
# The cameras number their boxes each for itself
# ---------------------------------------------------------------------------

@pytest.fixture
def lopsided(tmp_path):
    """The RGB detector found four things the thermal one did not, before
    the two animals: RGB boxes 5-8 are thermal boxes 1-4."""
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    for suffix, count in (("t", 4), ("w", 8)):
        conn = store.open_store(
            store.stage_path(root, store.DETECTIONS, suffix),
            store.DETECTIONS, suffix)
        with store.transaction(conn):
            for detection_id in range(1, count + 1):
                conn.execute(
                    "INSERT INTO detections (frame, x1, y1, x2, y2, "
                    "confidence, species_id, source_id) "
                    "VALUES (?, 0, 0, 10, 10, 0.9, 0, 1)", (detection_id - 1,))
        conn.close()
    track_store.record_tracks(root, "t", [
        {"track_id": 1 + (d > 2), "detection_id": d} for d in range(1, 5)])
    track_store.record_tracks(root, "w", [
        {"track_id": 1 + (d > 6), "detection_id": d} for d in range(5, 9)])
    match_store.record_matches(root, [
        {"track_id_t": track, "track_id_w": track, "shared": 2,
         "median_dist": 1.0, "conf_t": 0.9, "conf_w": 0.9,
         "pairs": [{"frame_t": d - 1, "frame_w": d + 3,
                    "detection_id_t": d, "detection_id_w": d + 4,
                    "dist": 1.0}
                   for d in ((1, 2) if track == 1 else (3, 4))]}
        for track in (1, 2)])
    return root


def test_a_box_gets_what_its_own_partner_carries(lopsided):
    """Up to 9.0 the pairs were looked up the wrong way round: right only
    where both cameras happened to number their boxes alike."""
    for box in (5, 6):                               # the first animal
        _set_detection(lopsided, "w", box, species=RED_DEER,
                       attributes={"occlusion": 1})
    for box in (1, 2, 3, 4):                         # nothing the other saw
        _set_detection(lopsided, "w", box, species=ROE_DEER,
                       attributes={"occlusion": 0})

    counts = label_sync.sync_detections(lopsided, "w", "t")

    boxes = _detections(lopsided, "t")
    assert [boxes[box] for box in (1, 2)] == [
        (RED_DEER, {"occlusion": 1})] * 2
    assert [boxes[box] for box in (3, 4)] == [(0, {})] * 2
    assert counts == {"occlusion": 2, "species": 2}


def test_and_from_the_other_side(lopsided):
    _set_detection(lopsided, "t", 3, species=ROE_DEER,
                   attributes={"occlusion": 1})

    counts = label_sync.sync_detections(lopsided, "t", "w")

    boxes = _detections(lopsided, "w")
    assert boxes[7] == (ROE_DEER, {"occlusion": 1})
    assert all(boxes[box] == (0, {}) for box in (1, 2, 3, 4, 5, 6, 8))
    assert counts == {"occlusion": 1, "species": 1}
