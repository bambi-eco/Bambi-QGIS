# -*- coding: utf-8 -*-
"""A survey of only the animals both cameras saw (``Survey.confirmed``):
the tracks of the survey's camera that the cross-modal matching paired
with a track of the other one - in every tool, over the ground both
cameras looked at."""
import json
import os

import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import (
    analytics_source, habitat, match_store)
from bambi_wildlife_detection.core.survey import (
    Survey, SurveyError, SurveyFlight)

from .survey_flights import EPSG, make_flight

#: Thermal sees 100 m x 100 m; RGB the western half of it.
SEEN_T = [(0, 0, 100, 100)]
SEEN_W = [(0, 0, 50, 100)]


def matched(folder, pairs):
    """Record that thermal track *t* and RGB track *w* are one animal."""
    match_store.record_matches(folder, [
        {"track_id_t": t, "track_id_w": w, "shared": 2, "median_dist": 0.5,
         "conf_t": 0.9, "conf_w": 0.9, "pairs": []} for t, w in pairs])


@pytest.fixture
def flight(tmp_path):
    """Three thermal animals; RGB saw the first of them (as its track
    2), and one thing of its own (track 1). The store numbers the tracks
    of each camera from 1."""
    folder = make_flight(str(tmp_path / "Meadow"), [
        {"id": 1, "species": "roe deer", "at": (10, 10), "frames": (0, 4)},
        {"id": 2, "species": "roe deer", "at": (30, 10), "frames": (1, 5)},
        {"id": 3, "species": "red deer", "at": (70, 10), "frames": (2, 6)},
    ], seen=SEEN_T)
    make_flight(folder, [
        {"id": 1, "species": "roe deer", "at": (40, 40), "frames": (1, 5)},
        {"id": 2, "species": "roe deer", "at": (10, 11), "frames": (0, 4)},
    ], modality="w", seen=SEEN_W)
    matched(folder, [(1, 2)])
    return folder


def survey_of(tmp_path, *folders, **more):
    flights = [SurveyFlight(os.path.basename(folder), folder, EPSG)
               for folder in folders]
    return Survey(folder=str(tmp_path / "survey"), flights=flights, **more)


# ---------------------------------------------------------------------------
# The animals
# ---------------------------------------------------------------------------

class TestAnimals:
    def test_all_of_them_without_the_filter(self, flight):
        animals, facts = analytics_source.load_animals(flight, "t")

        assert [animal["track_id"] for animal in animals] == [1, 2, 3]
        assert facts["unconfirmed"] == 0

    def test_only_those_the_other_camera_saw(self, flight):
        animals, facts = analytics_source.load_animals(
            flight, "t", confirmed=True)

        assert [animal["track_id"] for animal in animals] == [1]
        assert facts["unconfirmed"] == 2

    def test_from_the_other_side_it_is_the_partner(self, flight):
        animals, facts = analytics_source.load_animals(
            flight, "w", confirmed=True)

        assert [animal["track_id"] for animal in animals] == [2]
        assert facts["unconfirmed"] == 1

    def test_boxes_count_when_their_animal_is_confirmed(self, flight):
        boxes, facts = analytics_source.load_animals(
            flight, "t", analytics_source.DETECTIONS, confirmed=True)

        assert {box["track_id"] for box in boxes} == {1}
        assert len(boxes) == 2
        assert facts["unconfirmed"] == 4

    def test_the_species_filter_comes_first(self, flight):
        from .survey_flights import species_id

        _animals, facts = analytics_source.load_animals(
            flight, "t", species_ids=[species_id(flight, "roe deer")],
            confirmed=True)

        # the red deer was never asked for: it is not "unconfirmed"
        assert facts["unconfirmed"] == 1

    def test_a_flight_never_matched_has_none(self, tmp_path):
        folder = make_flight(str(tmp_path / "Alone"), [
            {"id": 1, "species": "roe deer", "at": (10, 10),
             "frames": (0, 4)}])

        animals, _facts = analytics_source.load_animals(
            folder, "t", confirmed=True)

        assert animals == []
        assert analytics_source.confirmed_tracks(folder, "t") == set()

    def test_the_points_of_a_map(self, flight):
        points, provenance = analytics_source.load_points(
            flight, "t", analytics_source.TRACKS, confirmed=True)

        assert len(points) == 1
        assert provenance["confirmed_by_other_camera"] is True
        assert provenance["unconfirmed"] == 2
        assert "only what both cameras saw (2 unconfirmed left out)" in \
            analytics_source.describe_filter(provenance)

        boxes, provenance = analytics_source.load_points(
            flight, "t", analytics_source.DETECTIONS, confirmed=True)
        assert len(boxes) == 2
        assert provenance["unconfirmed"] == 4

    def test_what_a_distance_file_lists(self, flight):
        entries = [{"track_id": 1, "class_id": 1},
                   {"track_id": 2, "class_id": 1},
                   {"track_id": None, "class_id": 1}]
        now = analytics_source.current_species(flight, "t")

        kept, _dropped = analytics_source.select_entries(
            entries, "track_id", now,
            only=analytics_source.confirmed_ids(flight, "t"))

        assert [entry["track_id"] for entry in kept] == [1]
        everything, _dropped = analytics_source.select_entries(
            entries, "track_id", now)
        assert len(everything) == 3

    def test_the_ids_of_boxes(self, flight):
        ids = analytics_source.confirmed_ids(
            flight, "t", analytics_source.DETECTIONS)
        boxes, _facts = analytics_source.load_animals(
            flight, "t", analytics_source.DETECTIONS, confirmed=True)

        assert ids == {box["detection_id"] for box in boxes}


# ---------------------------------------------------------------------------
# The ground
# ---------------------------------------------------------------------------

class TestGround:
    def test_one_camera_saw_all_of_its_footprints(self, flight):
        assert habitat.monitored_geometry(flight, "t").area == \
            pytest.approx(100 * 100)

    def test_both_saw_where_they_overlap(self, flight):
        for modality in ("t", "w"):
            assert habitat.monitored_geometry(
                flight, modality, both=True).area == pytest.approx(50 * 100)

    def test_without_the_other_cameras_footprints_its_own_stand(
            self, tmp_path):
        folder = make_flight(str(tmp_path / "Half"), [
            {"id": 1, "species": "roe deer", "at": (10, 10),
             "frames": (0, 4)}], seen=SEEN_T)

        assert habitat.monitored_geometry(
            folder, "t", both=True).area == pytest.approx(100 * 100)


# ---------------------------------------------------------------------------
# The survey
# ---------------------------------------------------------------------------

class TestSurvey:
    def test_it_travels_with_the_survey(self, tmp_path, flight):
        survey = survey_of(tmp_path, flight, confirmed=True)

        again = Survey.from_config({"survey": survey.to_dict()})

        assert again.confirmed is True
        assert Survey.from_dict({"folder": "x"}).confirmed is False

    def test_it_says_what_was_counted(self, tmp_path, flight):
        assert survey_of(tmp_path, flight).counted_label == "Thermal"
        assert survey_of(tmp_path, flight, confirmed=True).counted_label \
            == "Thermal, confirmed by RGB"
        assert survey_of(tmp_path, flight, modality="w",
                         confirmed=True).counted_label \
            == "RGB, confirmed by Thermal"

    def test_a_flight_never_matched_is_refused(self, tmp_path, flight):
        alone = make_flight(str(tmp_path / "Alone"), [
            {"id": 1, "species": "roe deer", "at": (10, 10),
             "frames": (0, 4)}])
        survey = survey_of(tmp_path, flight, alone, confirmed=True)

        with pytest.raises(SurveyError, match="have not been matched.*Alone"):
            survey.validate()
        # without the filter nobody asks
        survey_of(tmp_path, flight, alone).validate()

    def test_a_flight_without_tracks_has_nothing_to_confirm(
            self, tmp_path, flight):
        empty = str(tmp_path / "Empty")
        os.makedirs(empty)

        survey = survey_of(tmp_path, flight, empty, confirmed=True)

        assert survey.unmatched_flights() == []
        survey.validate()


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------

def census_of(survey):
    BambiProcessor().run_census_summary({"survey": survey.to_dict()})
    with open(os.path.join(survey.analytics_folder,
                           "census_summary_tracks.json"),
              encoding="utf-8") as handle:
        return json.load(handle)


def by_species(census):
    return {row["species"]: row for row in census["rows"]
            if row["flight"] == "Meadow"}


class TestCensus:
    def test_everything_one_camera_saw(self, tmp_path, flight):
        census = census_of(survey_of(tmp_path, flight))

        assert census["camera"] == "Thermal"
        rows = by_species(census)
        assert rows["roe deer"]["counted"] == 2
        assert rows["red deer"]["counted"] == 1
        assert rows["roe deer"]["covered_ha"] == pytest.approx(1.0)

    def test_only_what_both_saw_over_the_ground_both_saw(
            self, tmp_path, flight):
        census = census_of(survey_of(tmp_path, flight, confirmed=True))

        assert census["camera"] == "Thermal, confirmed by RGB"
        rows = by_species(census)
        assert rows["roe deer"]["counted"] == 1
        assert "red deer" not in rows or rows["red deer"]["counted"] == 0
        # half the ground: one animal on half a hectare
        assert rows["roe deer"]["covered_ha"] == pytest.approx(0.5)
        assert rows["roe deer"]["per_km2_covered"] == pytest.approx(200.0)


class TestInventory:
    def test_it_lists_the_confirmed_tracks_only(self, tmp_path, flight):
        import csv

        survey = survey_of(tmp_path, flight, confirmed=True)
        BambiProcessor().run_track_inventory({"survey": survey.to_dict()})

        with open(os.path.join(flight, "analytics_t",
                               "track_inventory.csv"),
                  encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["track_id"] for row in rows] == ["1"]
        assert rows[0]["matched_track"] == "2"
