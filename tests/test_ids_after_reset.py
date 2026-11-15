# -*- coding: utf-8 -*-
"""A store that is created again does not hand out the old ids again.

Detection and track ids come from ``AUTOINCREMENT`` and are never reused -
within a file. "Reset Step" deletes the file (and so may the user, by hand),
keeps what depended on it and marks it out of date. The new file then started
at 1 again, and the kept results belonged to the new rows with the old
numbers: new boxes at the old map positions, in the old tracks, with the old
species and embeddings.
"""
import os
import re
import sqlite3

import pytest

from bambi_wildlife_detection.core import (
    detection_store, label_store, stages, store, track_store)


def _boxes(n, frame=0):
    return [{"frame": frame + i, "x1": 10.0 + i, "y1": 20.0, "x2": 30.0 + i,
             "y2": 40.0, "confidence": 0.9, "source_class": "0"}
            for i in range(n)]


def _detection_ids(root, modality="t"):
    return [row["detection_id"]
            for row in track_store.load_detections(root, modality)]


def _geo_reference(root, ids, modality="t"):
    track_store.record_georeference(root, modality, [
        {"detection_id": i, "gx1": 1.0, "gy1": 1.0, "gz1": 0.0,
         "gx2": 2.0, "gy2": 2.0, "gz2": 0.0} for i in ids])


@pytest.fixture
def flight(tmp_path):
    """Three boxes, geo-referenced and tracked as one animal."""
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    detection_store.record_detections(root, "t", _boxes(3))
    ids = _detection_ids(root)
    _geo_reference(root, ids)
    track_store.record_tracks(
        root, "t", [{"track_id": 1, "detection_id": i} for i in ids],
        kind="builtin")
    for stage in ("detection", "georeference", "tracking"):
        stages.mark_complete(root, stage, "t")
    return root


def _track_ids(root, modality="t"):
    conn = store.open_store(
        store.stage_path(root, store.TRACKS, modality), store.TRACKS, modality)
    try:
        return [row["track_id"] for row in conn.execute(
            "SELECT track_id FROM tracks ORDER BY track_id")]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------

def test_new_boxes_do_not_take_the_numbers_of_the_old_ones(flight):
    old = _detection_ids(flight)
    stages.reset(flight, "detection", "t")
    assert not os.path.exists(
        store.stage_path(flight, store.DETECTIONS, "t"))

    detection_store.record_detections(flight, "t", _boxes(2, frame=50))

    new = _detection_ids(flight)
    assert len(new) == 2
    assert min(new) > max(old)


def test_new_boxes_do_not_land_at_the_old_positions(flight):
    """The symptom: the kept geo-referencing joined the new boxes."""
    stages.reset(flight, "detection", "t")
    detection_store.record_detections(flight, "t", _boxes(2, frame=50))

    assert track_store.load_georeferenced(flight, "t") == []


def test_new_boxes_are_not_members_of_the_old_tracks(flight):
    stages.reset(flight, "detection", "t")
    detection_store.record_detections(flight, "t", _boxes(2, frame=50))

    assert track_store.load_pixel_tracks(flight, "t") == []
    assert stages.states(flight, "t")["tracking"]["state"] == stages.STALE


def test_a_file_deleted_by_hand_is_no_different(flight):
    old = _detection_ids(flight)
    os.remove(store.stage_path(flight, store.DETECTIONS, "t"))

    detection_store.record_detections(flight, "t", _boxes(1))

    assert _detection_ids(flight)[0] > max(old)


def test_labels_added_to_a_new_store_number_on_too(flight):
    """The labelling tool creates the detections store as well."""
    old = _detection_ids(flight)
    stages.reset(flight, "detection", "t")

    label_store.save_tracks(flight, "t", [{
        "label_track_id": 1, "species_id": 0,
        "keyframes": [{"frame": 1, "x1": 1.0, "y1": 1.0, "x2": 2.0,
                       "y2": 2.0}]}])
    label_store.materialise(flight, "t")

    assert min(_detection_ids(flight)) > max(old)


def test_a_first_store_starts_at_one(tmp_path):
    root = str(tmp_path)
    detection_store.record_detections(root, "t", _boxes(2))
    assert _detection_ids(root) == [1, 2]


def test_the_other_camera_is_not_counted_in(flight):
    """Each camera numbers its own boxes."""
    detection_store.record_detections(flight, "w", _boxes(2))
    assert _detection_ids(flight, "w") == [1, 2]


def test_rerunning_inside_the_same_file_is_unchanged(flight):
    old = _detection_ids(flight)
    detection_store.record_detections(flight, "t", _boxes(3))
    assert _detection_ids(flight) == [i + 3 for i in old]


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------

def test_new_tracks_do_not_take_the_numbers_of_the_old_ones(flight):
    """What refers to a track: its thermal/RGB match, its classification, the
    label that was imported from it."""
    (old,) = _track_ids(flight)
    label_store.save_tracks(flight, "t", [{
        "label_track_id": 1, "species_id": 0, "origin_track_id": old + 4,
        "keyframes": [{"frame": 1, "x1": 1.0, "y1": 1.0, "x2": 2.0,
                       "y2": 2.0}]}])
    stages.reset(flight, "tracking", "t")

    ids = _detection_ids(flight)
    track_store.record_tracks(
        flight, "t", [{"track_id": 1, "detection_id": ids[0]}],
        kind="builtin")

    (new,) = _track_ids(flight)
    assert new > old + 4


# ---------------------------------------------------------------------------
# Everything that refers to an id is looked at
# ---------------------------------------------------------------------------

def _write(path, kind, modality, statements):
    conn = store.open_store(path, kind, modality)
    try:
        with store.transaction(conn):
            for statement in statements:
                conn.execute(statement)
    finally:
        conn.close()


REFERRING = [
    (store.DETECTIONS, store.GEOREFERENCED,
     "INSERT INTO detections_geo (detection_id, gx1, gy1, gz1, gx2, gy2, gz2) "
     "VALUES (41, 0, 0, 0, 1, 1, 0)"),
    (store.DETECTIONS, store.GEOREFERENCED,
     "INSERT INTO georef_failures (detection_id, reason) VALUES (41, 'x')"),
    (store.DETECTIONS, store.LABELS,
     "INSERT INTO label_keyframes (label_track_id, frame, x1, y1, x2, y2, "
     "origin_detection_id) VALUES (1, 1, 0, 0, 1, 1, 41)"),
    (store.DETECTIONS, store.SEGMENTATION,
     "INSERT INTO segments (detection_id, frame) VALUES (41, 1)"),
    (store.DETECTIONS, store.CLASSIFICATION,
     "INSERT INTO embeddings (detection_id, run_id) VALUES (41, 1)"),
    (store.DETECTIONS, store.CLASSIFICATION,
     "INSERT INTO frame_predictions (detection_id, task, label, class_index, "
     "prob, modality_in) VALUES (41, 'sex', 'male', 0, 0.9, 'rgb')"),
    (store.TRACKS, store.LABELS,
     "INSERT INTO label_tracks (label_track_id, origin_track_id) "
     "VALUES (1, 41)"),
    (store.TRACKS, store.LABELS,
     "INSERT INTO label_tracks (label_track_id, track_id) VALUES (1, 41)"),
    (store.TRACKS, store.CLASSIFICATION,
     "INSERT INTO track_predictions (track_id, task, label, votes, n, "
     "fraction, modality_in) VALUES (41, 'sex', 'male', 1, 1, 1.0, 'rgb')"),
]


@pytest.mark.parametrize("numbered, referring, statement", REFERRING)
def test_each_reference_keeps_its_id_from_being_reused(
        tmp_path, numbered, referring, statement):
    root = str(tmp_path)
    prepare = []
    if "label_keyframes" in statement:
        prepare = ["INSERT INTO label_tracks (label_track_id) VALUES (1)"]
    if "embeddings" in statement:
        prepare = ["INSERT INTO embedding_runs (run_id) VALUES (1)"]
    _write(store.stage_path(root, referring, "t"), referring, "t",
           ["PRAGMA foreign_keys = OFF"] + prepare + [statement])

    path = store.stage_path(root, numbered, "t")
    assert store.highest_reference(path, numbered, "t") == 41
    assert store.highest_reference(
        store.stage_path(root, numbered, "w"), numbered, "w") == 0


def test_a_track_member_keeps_its_detection_id_from_being_reused(tmp_path):
    root = str(tmp_path)
    track_store.record_tracks(
        root, "t", [{"track_id": 1, "detection_id": 41}], kind="builtin")
    assert store.highest_reference(
        store.stage_path(root, store.DETECTIONS, "t"),
        store.DETECTIONS, "t") == 41


@pytest.mark.parametrize("modality, expected_detection, expected_track", [
    ("t", 41, 7), ("w", 52, 9)])
def test_the_match_store_counts_per_camera(tmp_path, modality,
                                           expected_detection, expected_track):
    root = str(tmp_path)
    _write(store.matches_path(root), store.MATCHES, "", [
        "INSERT INTO match_runs (run_id) VALUES (1)",
        "INSERT INTO track_matches (match_id, run_id, track_id_t, track_id_w, "
        "shared, median_dist) VALUES (1, 1, 7, 9, 3, 1.0)",
        "INSERT INTO detection_matches (match_id, frame_t, frame_w, "
        "detection_id_t, detection_id_w) VALUES (1, 1, 1, 41, 52)"])

    assert store.highest_reference(
        store.stage_path(root, store.DETECTIONS, modality),
        store.DETECTIONS, modality) == expected_detection
    assert store.highest_reference(
        store.stage_path(root, store.TRACKS, modality),
        store.TRACKS, modality) == expected_track


def test_an_unreadable_neighbour_does_not_stop_a_store_from_opening(tmp_path):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "bambi_t"))
    with open(store.stage_path(root, store.TRACKS, "t"), "w") as fh:
        fh.write("not a database")
    detection_store.record_detections(root, "t", _boxes(1))
    assert _detection_ids(root) == [1]


def test_every_column_that_refers_to_an_id_is_registered():
    """A table added later that refers to a detection or a track by its id
    has to be added to the register, or its rows could be taken over by a
    store created anew."""
    registered = set()
    for numbered, by_kind in store._REFERENCES.items():
        for kind, statements in by_kind.items():
            for statement in statements:
                column, table = re.match(
                    r"SELECT MAX\((\w+)\) FROM (\w+)$", statement).groups()
                registered.add((kind, table, column))
    for numbered, by_modality in store._MATCH_REFERENCES.items():
        for statement in by_modality.values():
            column, table = re.match(
                r"SELECT MAX\((\w+)\) FROM (\w+)$", statement).groups()
            registered.add((store.MATCHES, table, column))

    #: where the ids are handed out, and membership inside the same file
    own = {(store.DETECTIONS, "detections", "detection_id"),
           (store.TRACKS, "tracks", "track_id"),
           (store.TRACKS, "track_members", "track_id")}

    pattern = re.compile(r"^(origin_)?(detection|track)_id(_[tw])?$")
    found = set()
    for kind, ddl in store._DDL.items():
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(ddl)
            tables = [row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%'")]
            for table in tables:
                for column in conn.execute(f"PRAGMA table_info({table})"):
                    if pattern.match(column[1]):
                        found.add((kind, table, column[1]))
        finally:
            conn.close()

    assert found - own == registered
