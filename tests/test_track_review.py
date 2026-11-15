# -*- coding: utf-8 -*-
"""A person's word on a track the pipeline made (core/track_review.py), and
that the classifiers leave it alone (core/apply_results.py)."""
import json

import pytest

from bambi_wildlife_detection.core import (
    analytics_source, apply_results, label_store, store, track_review,
    track_store)

from .survey_flights import make_flight, make_labelled_flight, species_id

TRACKS = [
    {"id": 1, "species": "roe deer", "at": (0, 0), "frames": (1, 5)},
    {"id": 2, "species": "roe deer", "at": (60, 0), "frames": (2, 6)},
]


@pytest.fixture
def flight(tmp_path):
    return make_flight(str(tmp_path / "flight"), TRACKS)


def _track(folder, track_id):
    conn = store.open_store(store.stage_path(folder, store.TRACKS, "t"),
                            store.TRACKS, "t")
    try:
        row = conn.execute("SELECT species_id, attributes FROM tracks "
                           "WHERE track_id = ?", (track_id,)).fetchone()
        return int(row["species_id"]), json.loads(row["attributes"] or "{}")
    finally:
        conn.close()


def _box_species(folder, track_id):
    return sorted({row["species_id"] for row in
                   track_store.load_pixel_tracks(folder, "t")
                   if row["track_id"] == track_id})


def _enum(folder, field, label):
    vocabulary = label_store.vocabulary(folder)
    return next(value["value_id"] for value in vocabulary["enums"][field]
                if value["label"] == label)


# ---------------------------------------------------------------------------
# Saying what an animal is
# ---------------------------------------------------------------------------

class TestSpecies:
    def test_the_species_goes_on_the_track_and_on_its_boxes(self, flight):
        red = species_id(flight, "red deer")

        assert track_review.set_species(flight, "t", 1, red) is True

        species, attributes = _track(flight, 1)
        assert species == red
        assert attributes[track_review.BY_HAND] == ["species"]
        assert _box_species(flight, 1) == [red]
        assert _box_species(flight, 2) != [red]

    def test_every_survey_tool_counts_it_as_that(self, flight):
        red = species_id(flight, "red deer")
        track_review.set_species(flight, "t", 1, red)

        animals, _facts = analytics_source.load_animals(flight, "t")

        assert {a["track_id"]: a["species"] for a in animals} == {
            1: "red deer", 2: "roe deer"}

    def test_not_an_animal_takes_it_out_of_every_count(self, flight):
        track_review.set_species(flight, "t", 2,
                                 store.NOT_AN_ANIMAL_SPECIES_ID)

        animals, facts = analytics_source.load_animals(flight, "t")

        assert [a["track_id"] for a in animals] == [1]
        assert facts["excluded_not_an_animal"] == 1

    def test_saying_the_same_again_changes_nothing(self, flight):
        red = species_id(flight, "red deer")
        track_review.set_species(flight, "t", 1, red)

        assert track_review.set_species(flight, "t", 1, red) is False

    def test_confirming_what_the_pipeline_said_marks_it_too(self, flight):
        """"Yes, a roe deer" is a person's word as much as a correction."""
        roe = species_id(flight, "roe deer")

        assert track_review.set_species(flight, "t", 1, roe) is True
        assert _track(flight, 1)[1][track_review.BY_HAND] == ["species"]

    def test_what_is_not_there_is_refused(self, flight):
        with pytest.raises(track_review.TrackReviewError, match="no track"):
            track_review.set_species(flight, "t", 99, 1)
        with pytest.raises(track_review.TrackReviewError, match="no species"):
            track_review.set_species(flight, "t", 1, 4242)
        with pytest.raises(track_review.TrackReviewError, match="no tracks"):
            track_review.set_species(flight, "w", 1, 1)

    def test_a_hand_labelled_track_is_changed_where_its_labels_are(
            self, tmp_path):
        folder = make_labelled_flight(str(tmp_path / "labelled"))
        manual = track_store.manual_run(folder, "t")
        conn = store.open_store(store.stage_path(folder, store.TRACKS, "t"),
                                store.TRACKS, "t")
        labelled = conn.execute(
            "SELECT track_id FROM tracks WHERE run_id = ?",
            (manual["run_id"],)).fetchone()["track_id"]
        conn.close()

        with pytest.raises(track_review.TrackReviewError,
                           match="Labelling Tool"):
            track_review.set_species(folder, "t", labelled, 1)
        with pytest.raises(track_review.TrackReviewError,
                           match="Labelling Tool"):
            track_review.set_attribute(folder, "t", labelled, "sex", None)


class TestSexAndAge:
    def test_sex_and_age_are_set_and_marked(self, flight):
        female = _enum(flight, "sex", "female")
        adult = _enum(flight, "age", "adult")

        assert track_review.set_attribute(flight, "t", 1, "sex", female)
        assert track_review.set_attribute(flight, "t", 1, "age", adult)

        _species, attributes = _track(flight, 1)
        assert (attributes["sex"], attributes["age"]) == (female, adult)
        assert attributes[track_review.BY_HAND] == ["sex", "age"]
        animals, _facts = analytics_source.load_animals(flight, "t")
        assert (animals[0]["sex"], animals[0]["age"]) == ("female", "adult")

    def test_not_known_is_a_persons_word_too(self, flight):
        female = _enum(flight, "sex", "female")
        track_store.set_track_attribute(flight, "t", 1, "sex", female)

        assert track_review.set_attribute(flight, "t", 1, "sex", None)

        _species, attributes = _track(flight, 1)
        assert "sex" not in attributes
        assert attributes[track_review.BY_HAND] == ["sex"]

    def test_the_same_again_changes_nothing(self, flight):
        female = _enum(flight, "sex", "female")
        track_review.set_attribute(flight, "t", 1, "sex", female)

        assert track_review.set_attribute(flight, "t", 1, "sex",
                                          female) is False

    def test_other_fields_are_not_a_reviewers(self, flight):
        with pytest.raises(track_review.TrackReviewError):
            track_review.set_attribute(flight, "t", 1, "approved", 1)

    def test_the_approved_tick_is_kept(self, flight):
        track_store.set_track_attribute(flight, "t", 1, "approved", True)
        track_review.set_attribute(flight, "t", 1, "sex",
                                   _enum(flight, "sex", "male"))

        assert _track(flight, 1)[1]["approved"] is True


def test_which_fields_a_person_set():
    assert track_review.by_hand({"by_hand": ["sex", "species"]}) == [
        "species", "sex"]
    assert track_review.by_hand('{"by_hand": ["age"]}') == ["age"]
    assert track_review.by_hand({"by_hand": "species"}) == []
    assert track_review.by_hand({"by_hand": ["approved"]}) == []
    assert track_review.by_hand("{not json") == []
    assert track_review.by_hand(None) == [] and track_review.by_hand({}) == []


# ---------------------------------------------------------------------------
# The classifiers leave it alone
# ---------------------------------------------------------------------------

def _call(track_id, label):
    return {"track_id": track_id, "label": label, "votes": 3, "n": 3,
            "fraction": 1.0, "model": "species.pt"}


class TestClassifiers:
    def test_a_species_set_by_hand_survives_a_classifier_run(self, flight):
        red = species_id(flight, "red deer")
        chamois = species_id(flight, "chamois")
        track_review.set_species(flight, "t", 1, red)

        apply_results.apply_track_species(
            flight, "t", [_call(1, "chamois"), _call(2, "chamois")],
            {"chamois": chamois}, overwrite_detections=True)

        assert _track(flight, 1)[0] == red              # the person's
        assert _box_species(flight, 1) == [red]
        assert _track(flight, 2)[0] == chamois          # the classifier's
        assert _box_species(flight, 2) == [chamois]

    def test_without_the_mark_the_classifier_would_have_overwritten_it(
            self, flight):
        """What happened to a species put on a track before 9.1."""
        red = species_id(flight, "red deer")
        chamois = species_id(flight, "chamois")
        track_review.set_species(flight, "t", 1, red)
        track_review.release(flight, "t", 1)

        apply_results.apply_track_species(
            flight, "t", [_call(1, "chamois")], {"chamois": chamois})

        assert _track(flight, 1)[0] == chamois

    def test_a_vote_that_goes_away_does_not_take_the_persons_word_along(
            self, flight):
        red = species_id(flight, "red deer")
        chamois = species_id(flight, "chamois")
        apply_results.apply_track_species(
            flight, "t", [_call(1, "chamois")], {"chamois": chamois})
        track_review.set_species(flight, "t", 1, red)

        apply_results.apply_track_species(flight, "t", [],
                                          {"chamois": chamois})

        assert _track(flight, 1)[0] == red

    def test_sex_and_age_set_by_hand_survive_too(self, flight):
        female, male = _enum(flight, "sex", "female"), _enum(
            flight, "sex", "male")
        track_review.set_attribute(flight, "t", 1, "sex", female)

        written = apply_results.apply_track_attribute(
            flight, "t", "sex",
            [dict(_call(1, "male")), dict(_call(2, "male"))], {"male": male})

        assert _track(flight, 1)[1]["sex"] == female
        assert _track(flight, 2)[1]["sex"] == male
        assert written == 1

    def test_only_the_field_that_was_set_is_kept_from_them(self, flight):
        red = species_id(flight, "red deer")
        male = _enum(flight, "sex", "male")
        track_review.set_species(flight, "t", 1, red)

        apply_results.apply_track_attribute(
            flight, "t", "sex", [_call(1, "male")], {"male": male})

        assert _track(flight, 1)[1]["sex"] == male
        assert _track(flight, 1)[0] == red

    def test_not_known_stays_not_known(self, flight):
        male = _enum(flight, "sex", "male")
        track_review.set_attribute(flight, "t", 1, "sex", None)

        apply_results.apply_track_attribute(
            flight, "t", "sex", [_call(1, "male")], {"male": male})

        assert "sex" not in _track(flight, 1)[1]


class TestRelease:
    def test_a_mark_can_be_taken_back(self, flight):
        red = species_id(flight, "red deer")
        track_review.set_species(flight, "t", 1, red)
        track_review.set_attribute(flight, "t", 1, "sex", None)

        assert track_review.release(flight, "t", 1, ["species"]) is True

        species, attributes = _track(flight, 1)
        assert species == red                     # the value stays
        assert attributes[track_review.BY_HAND] == ["sex"]
        assert track_review.release(flight, "t", 1) is True
        assert track_review.BY_HAND not in _track(flight, 1)[1]

    def test_nothing_to_take_back(self, flight):
        assert track_review.release(flight, "t", 1) is False
        assert track_review.release(flight, "t", 99) is False
        assert track_review.release(flight, "w", 1) is False
