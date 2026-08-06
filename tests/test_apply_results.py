# -*- coding: utf-8 -*-
"""Projecting classification results onto the canonical fields.

This is the step that makes the classifiers visible to everything else, so the
tests check where the values actually land — including at the detection level,
which is what the exporters and the analytics filter on.
"""
import json
import os

import pytest

from bambi_wildlife_detection.core import (
    apply_results, classification_store, label_store, store, track_store,
)


@pytest.fixture
def flight(tmp_path):
    """Two tracks of two detections each, un-identified by the detector."""
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()

    conn = store.open_store(
        store.stage_path(root, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    with store.transaction(conn):
        for frame in range(4):
            conn.execute(
                "INSERT INTO detections (frame, x1, y1, x2, y2, confidence, "
                "species_id, source_id, source_class) "
                "VALUES (?, 0, 0, 10, 10, 0.9, 0, 1, '0')", (frame,))
    conn.close()

    track_store.record_tracks(root, "t", [
        {"track_id": 1, "detection_id": 1},
        {"track_id": 1, "detection_id": 2},
        {"track_id": 2, "detection_id": 3},
        {"track_id": 2, "detection_id": 4},
    ])
    return root


def _vocabulary(root):
    return label_store.vocabulary(root)


def _species_id(root, name):
    return _vocabulary(root)["species_by_name"][name]


def _enum_id(root, enum, label):
    return _vocabulary(root)["enum_ids"][enum][label]


def _track_rows(root):
    conn = store.open_store(
        store.stage_path(root, store.TRACKS, "t"), store.TRACKS, "t")
    try:
        return {int(r["track_id"]): dict(r) for r in conn.execute(
            "SELECT track_id, species_id, attributes FROM tracks")}
    finally:
        conn.close()


def _detection_rows(root):
    conn = store.open_store(
        store.stage_path(root, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    try:
        return {int(r["detection_id"]): dict(r) for r in conn.execute(
            "SELECT detection_id, species_id, attributes FROM detections")}
    finally:
        conn.close()


def _models(root):
    return {
        "occlusion": {"class_labels": ["clear", "occluded"],
                      "labels": {"0": _enum_id(root, "occlusion", "clear"),
                                 "1": _enum_id(root, "occlusion",
                                               "occluded")}},
        "species": {"class_labels": ["roe deer", "red deer"],
                    "labels": {"0": _species_id(root, "roe deer"),
                               "1": _species_id(root, "red deer")}},
        "sex": {"class_labels": ["female_juvenile", "male"],
                "labels": {"0": _enum_id(root, "sex", "female"),
                           "1": _enum_id(root, "sex", "male")}},
        "life_stage": {},
    }


def _store_predictions(root):
    classification_store.start_embedding_run(
        root, "t", backbone="x", dim=4, crop_size=32, padding=0.1,
        projection="non_geo", folder="embeddings_t/non_geo")
    classification_store.record_frame_predictions(root, "t", "occlusion", [
        {"detection_id": 1, "label": "clear", "class_index": 0, "prob": 0.9},
        {"detection_id": 2, "label": "occluded", "class_index": 1,
         "prob": 0.8}])
    classification_store.record_track_predictions(root, "t", "species", [
        {"track_id": 1, "label": "red deer", "votes": 2, "n": 2,
         "fraction": 1.0}])
    classification_store.record_track_predictions(root, "t", "sex", [
        {"track_id": 1, "label": "male", "votes": 2, "n": 2,
         "fraction": 1.0}])
    classification_store.record_track_predictions(
        root, "t", classification_store.LIFE_STAGE, [
            {"track_id": 2, "label": "juvenile", "votes": 1, "n": 1,
             "fraction": 1.0}])


# ---------------------------------------------------------------------------
# Where the values land
# ---------------------------------------------------------------------------

class TestApplyAll:

    def test_species_reaches_the_track(self, flight):
        _store_predictions(flight)
        apply_results.apply_all(flight, "t", _models(flight))
        assert _track_rows(flight)[1]["species_id"] == \
            _species_id(flight, "red deer")

    def test_species_reaches_the_detections_too(self, flight):
        """The exporters and the analytics filter on the detection's species,
        so a track-only write would leave them blind to the result."""
        _store_predictions(flight)
        apply_results.apply_all(flight, "t", _models(flight))

        rows = _detection_rows(flight)
        assert rows[1]["species_id"] == _species_id(flight, "red deer")
        assert rows[2]["species_id"] == _species_id(flight, "red deer")
        # Track 2 was never called, so its detections are untouched.
        assert rows[3]["species_id"] == 0

    def test_sex_lands_on_the_track_attributes(self, flight):
        _store_predictions(flight)
        apply_results.apply_all(flight, "t", _models(flight))
        attributes = json.loads(_track_rows(flight)[1]["attributes"])
        assert attributes["sex"] == _enum_id(flight, "sex", "male")

    def test_life_stage_lands_in_the_age_field(self, flight):
        _store_predictions(flight)
        apply_results.apply_all(flight, "t", _models(flight))
        attributes = json.loads(_track_rows(flight)[2]["attributes"])
        assert attributes["age"] == _enum_id(flight, "age", "juvenile")

    def test_occlusion_lands_on_the_detections(self, flight):
        _store_predictions(flight)
        apply_results.apply_all(flight, "t", _models(flight))

        rows = _detection_rows(flight)
        assert json.loads(rows[1]["attributes"])["occlusion"] == \
            _enum_id(flight, "occlusion", "clear")
        assert json.loads(rows[2]["attributes"])["occlusion"] == \
            _enum_id(flight, "occlusion", "occluded")

    def test_only_the_requested_tasks_are_written(self, flight):
        _store_predictions(flight)
        apply_results.apply_all(flight, "t", _models(flight),
                                tasks=["species"])
        assert _detection_rows(flight)[1]["attributes"] in (None, "")

    def test_applying_twice_changes_nothing_the_second_time(self, flight):
        _store_predictions(flight)
        apply_results.apply_all(flight, "t", _models(flight))
        again = apply_results.apply_all(flight, "t", _models(flight))
        assert all(count == 0 for count in again.values())

    def test_it_can_be_re_run_after_something_else_overwrote_the_species(
            self, flight):
        """track_predictions stays the source of truth: re-applying a detector
        class mapping resets detection species, and this puts it back."""
        _store_predictions(flight)
        apply_results.apply_all(flight, "t", _models(flight))

        conn = store.open_store(
            store.stage_path(flight, store.DETECTIONS, "t"),
            store.DETECTIONS, "t")
        with store.transaction(conn):
            conn.execute("UPDATE detections SET species_id = 0")
        conn.close()

        apply_results.apply_all(flight, "t", _models(flight))
        assert _detection_rows(flight)[1]["species_id"] == \
            _species_id(flight, "red deer")


# ---------------------------------------------------------------------------
# What must not be overwritten
# ---------------------------------------------------------------------------

class TestGuards:

    def test_a_detector_identified_species_is_left_alone(self, flight):
        """Someone configured that class mapping; do not silently undo it."""
        boar = _species_id(flight, "wild boar")
        conn = store.open_store(
            store.stage_path(flight, store.DETECTIONS, "t"),
            store.DETECTIONS, "t")
        with store.transaction(conn):
            conn.execute(
                "UPDATE detections SET species_id = ? WHERE detection_id = 1",
                (boar,))
        conn.close()

        _store_predictions(flight)
        apply_results.apply_all(flight, "t", _models(flight))

        rows = _detection_rows(flight)
        assert rows[1]["species_id"] == boar          # kept
        assert rows[2]["species_id"] == _species_id(flight, "red deer")

    def test_the_detector_can_be_overridden_on_request(self, flight):
        boar = _species_id(flight, "wild boar")
        conn = store.open_store(
            store.stage_path(flight, store.DETECTIONS, "t"),
            store.DETECTIONS, "t")
        with store.transaction(conn):
            conn.execute(
                "UPDATE detections SET species_id = ? WHERE detection_id = 1",
                (boar,))
        conn.close()

        _store_predictions(flight)
        apply_results.apply_all(flight, "t", _models(flight),
                                overwrite_detections=True)
        assert _detection_rows(flight)[1]["species_id"] == \
            _species_id(flight, "red deer")

    def test_manual_tracks_are_never_touched(self, flight):
        """A hand annotation outranks a model."""
        from bambi_wildlife_detection.core import label_store as ls

        ls.save_tracks(flight, "t", [
            {"label_track_id": 1, "species_id": _species_id(flight, "fox"),
             "keyframes": [{"frame": 0, "x1": 0, "y1": 0, "x2": 5, "y2": 5}]}])
        ls.materialise(flight, "t")

        manual = track_store.manual_run(flight, "t")
        assert manual is not None

        conn = store.open_store(
            store.stage_path(flight, store.TRACKS, "t"), store.TRACKS, "t")
        try:
            manual_tracks = [int(r["track_id"]) for r in conn.execute(
                "SELECT track_id FROM tracks WHERE run_id = ?",
                (manual["run_id"],))]
        finally:
            conn.close()

        classification_store.record_track_predictions(flight, "t", "species", [
            {"track_id": track_id, "label": "red deer", "votes": 2, "n": 2,
             "fraction": 1.0} for track_id in manual_tracks])
        apply_results.apply_all(flight, "t", _models(flight),
                                tasks=["species"])

        rows = _track_rows(flight)
        for track_id in manual_tracks:
            assert rows[track_id]["species_id"] == _species_id(flight, "fox")

    def test_other_attributes_survive(self, flight):
        conn = store.open_store(
            store.stage_path(flight, store.TRACKS, "t"), store.TRACKS, "t")
        with store.transaction(conn):
            conn.execute(
                "UPDATE tracks SET attributes = ? WHERE track_id = 1",
                (json.dumps({"collar": "R-114"}),))
        conn.close()

        _store_predictions(flight)
        apply_results.apply_all(flight, "t", _models(flight))

        attributes = json.loads(_track_rows(flight)[1]["attributes"])
        assert attributes["collar"] == "R-114"
        assert "sex" in attributes

    def test_a_label_with_no_project_value_is_skipped(self, flight):
        """A class nobody mapped and whose name matches nothing is left out,
        rather than guessed at."""
        classification_store.start_embedding_run(
            flight, "t", backbone="x", dim=4, crop_size=32, padding=0.1,
            projection="non_geo", folder="embeddings_t/non_geo")
        classification_store.record_track_predictions(
            flight, "t", "species", [
                {"track_id": 1, "label": "brontosaurus", "votes": 2, "n": 2,
                 "fraction": 1.0}])

        models = _models(flight)
        models["species"] = {"class_labels": ["brontosaurus"], "labels": {}}
        apply_results.apply_all(flight, "t", models, tasks=["species"])
        assert _track_rows(flight)[1]["species_id"] == 0

    def test_a_project_without_a_vocabulary_writes_nothing(self, tmp_path):
        logs = []
        result = apply_results.apply_all(
            str(tmp_path), "t", {}, log_fn=logs.append)
        assert result == {}
        assert any("no vocabulary" in line for line in logs)


# ---------------------------------------------------------------------------
# Label resolution
# ---------------------------------------------------------------------------

class TestLabelValues:

    def test_the_configured_mapping_wins(self, flight):
        vocabulary = _vocabulary(flight)
        spec = {"class_labels": ["a", "b"],
                "labels": {"0": _species_id(flight, "red deer")}}
        values = apply_results.label_values(spec, vocabulary, "species")
        assert values["a"] == _species_id(flight, "red deer")

    def test_names_resolve_without_any_configuration(self, flight):
        """A model already speaking the project's language needs no clicks."""
        vocabulary = _vocabulary(flight)
        values = apply_results.label_values({}, vocabulary, "species")
        assert values["red deer"] == _species_id(flight, "red deer")

    def test_the_configured_mapping_beats_a_coincidental_name(self, flight):
        vocabulary = _vocabulary(flight)
        spec = {"class_labels": ["red deer"],
                "labels": {"0": _species_id(flight, "roe deer")}}
        values = apply_results.label_values(spec, vocabulary, "species")
        assert values["red deer"] == _species_id(flight, "roe deer")

    def test_life_stage_maps_onto_the_age_enum(self, flight):
        vocabulary = _vocabulary(flight)
        values = apply_results.label_values({}, vocabulary, "life_stage")
        assert values["juvenile"] == _enum_id(flight, "age", "juvenile")

    def test_matching_ignores_case(self, flight):
        vocabulary = _vocabulary(flight)
        values = apply_results.label_values(
            {"class_labels": ["Red Deer"]}, vocabulary, "species")
        assert values["Red Deer"] == _species_id(flight, "red deer")
