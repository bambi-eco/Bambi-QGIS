# -*- coding: utf-8 -*-
"""The tracks worth a human look (core/review_queue.py), and what the track
inventory knows that the queue asks (core/track_inventory.py)."""
import pytest

from bambi_wildlife_detection.core import review_queue as rq


def row(track_id=1, **more):
    """A plain track: nothing about it is in doubt."""
    return dict({
        "flight": "Meadow", "track_id": track_id, "camera": "thermal",
        "approved": False, "label_track_ids": "", "by_hand": "",
        "species": "roe deer", "species_vote_fraction": 0.95,
        "detector_species": "", "sex": "female", "sex_vote_fraction": 0.9,
        "mean_confidence": 0.85, "n_boxes": 30, "n_occluded": 2}, **more)


def kinds(found):
    return [kind for kind, _text in found]


# ---------------------------------------------------------------------------
# Why a track is in doubt
# ---------------------------------------------------------------------------

class TestReasons:
    def test_a_plain_track_is_not_in_doubt(self):
        assert rq.reasons(row()) == []

    def test_low_confidence(self):
        found = rq.reasons(row(mean_confidence=0.31))

        assert found == [("low_confidence",
                          "boxes of low confidence (mean 0.31)")]

    def test_very_short(self):
        assert rq.reasons(row(n_boxes=2, n_occluded=0)) == [
            ("short", "only 2 boxes")]
        assert rq.reasons(row(n_boxes=1, n_occluded=0)) == [
            ("short", "only 1 box")]
        assert rq.reasons(row(n_boxes=4)) == []

    def test_nobody_knows_what_it_is(self):
        for species in ("animal", "unknown", "", None):
            assert kinds(rq.reasons(row(
                species=species, species_vote_fraction=None))) == [
                "no_species"]

    def test_a_narrow_vote(self):
        found = rq.reasons(row(species_vote_fraction=0.55,
                               sex_vote_fraction=0.6))

        assert found == [
            ("species_vote", "roe deer by 55 % of the votes only"),
            ("sex_vote", "female by 60 % of the votes only")]

    def test_the_detector_says_another_species(self):
        found = rq.reasons(row(detector_species="red deer"))

        assert found == [("detector_species",
                          "the detector calls it red deer")]
        assert rq.reasons(row(detector_species="Roe Deer")) == []

    def test_the_other_camera_says_something_else(self):
        partner = row(2, camera="rgb", species="red deer", sex="male")

        found = rq.reasons(row(), partner, matched=True)

        assert found == [
            ("cameras_species", "the RGB camera says red deer"),
            ("cameras_sex", "the RGB camera says male")]

    def test_a_partner_that_agrees_or_does_not_know_is_no_doubt(self):
        assert rq.reasons(row(), row(2, camera="rgb"), matched=True) == []
        unsure = row(2, camera="rgb", species="animal", sex="")
        assert rq.reasons(row(), unsure, matched=True) == []

    def test_one_camera_only_where_the_cameras_were_matched(self):
        assert rq.reasons(row(), None, matched=True) == [
            ("no_partner", "not seen by the RGB camera")]
        assert rq.reasons(row(camera="rgb"), None, matched=True) == [
            ("no_partner", "not seen by the thermal camera")]
        assert rq.reasons(row(), None, matched=False) == []

    def test_mostly_hidden(self):
        assert rq.reasons(row(n_boxes=10, n_occluded=6)) == [
            ("occluded", "hidden on 6 of 10 frames")]
        assert rq.reasons(row(n_boxes=10, n_occluded=5)) == []

    def test_the_limits_can_be_set(self):
        strict = {"min_confidence": 0.9, "min_boxes": 40, "min_vote": 0.99}

        assert kinds(rq.reasons(row(), settings=strict)) == [
            "low_confidence", "short", "species_vote", "sex_vote"]
        assert rq.reasons(row(mean_confidence=0.2, n_boxes=2, n_occluded=0),
                          settings={"min_confidence": 0.1,
                                    "min_boxes": 1}) == []
        assert rq.reasons(row(), settings={"min_vote": None}) == []

    def test_what_a_person_said_is_not_asked_about_again(self):
        doubtful = dict(species_vote_fraction=0.5, detector_species="red deer",
                        sex_vote_fraction=0.5)
        partner = row(2, camera="rgb", species="red deer", sex="male")

        assert len(rq.reasons(row(**doubtful), partner, True)) == 5
        assert kinds(rq.reasons(row(by_hand="species", **doubtful),
                                partner, True)) == ["sex_vote",
                                                    "cameras_sex"]
        assert rq.reasons(row(by_hand="species; sex", **doubtful),
                          partner, True) == []

    def test_but_a_track_that_is_too_short_stays_in_doubt(self):
        found = rq.reasons(row(by_hand="species; sex; age", n_boxes=2,
                               n_occluded=0))

        assert kinds(found) == ["short"]

    def test_values_that_are_not_there_do_not_break_it(self):
        bare = {"track_id": 3, "species": "roe deer"}

        assert kinds(rq.reasons(bare)) == ["short"]


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------

class TestQueue:
    def test_only_tracks_in_doubt_are_in_it(self):
        queue = rq.build([row(1), row(2, mean_confidence=0.2), row(3)])

        assert [entry["track_id"] for entry in queue] == [2]
        assert queue[0]["reasons"] == ["boxes of low confidence (mean 0.20)"]
        assert queue[0]["reason_kinds"] == ["low_confidence"]
        assert queue[0]["priority"] == 3.0
        assert queue[0]["species"] == "roe deer"       # the row comes along

    def test_approved_and_hand_labelled_tracks_are_not(self):
        doubt = dict(mean_confidence=0.2, n_boxes=1)
        queue = rq.build([
            row(1, approved=True, **doubt),
            row(2, label_track_ids="7", **doubt),
            row(3, species="not-an-animal", **doubt),
            row(4, **doubt)])

        assert [entry["track_id"] for entry in queue] == [4]

    def test_the_most_doubtful_come_first(self):
        queue = rq.build([
            row(1, sex_vote_fraction=0.5),                     # 0.5
            row(2, mean_confidence=0.3, n_boxes=2,
                n_occluded=0),                                 # 6
            row(3, species="animal", species_vote_fraction=None),   # 1.5
            row(4, mean_confidence=0.2)])                      # 3

        assert [entry["track_id"] for entry in queue] == [2, 4, 3, 1]

    def test_of_equally_doubtful_the_fainter_first(self):
        queue = rq.build([row(1, mean_confidence=0.35),
                          row(2, mean_confidence=0.15),
                          row(3, mean_confidence=0.25)])

        assert [entry["track_id"] for entry in queue] == [2, 3, 1]

    def test_partners_are_found_by_flight_and_track(self):
        partner = row(9, camera="rgb", species="red deer")
        rows = [row(1), row(1, flight="Forest")]

        queue = rq.build(rows, partners={("Forest", 1): partner},
                         matched_flights=["Forest"])

        assert [(entry["flight"], entry["reasons"]) for entry in queue] == [
            ("Forest", ["the RGB camera says red deer"])]

    def test_a_flight_without_matching_does_not_miss_partners(self):
        rows = [row(1), row(2, flight="Forest")]

        queue = rq.build(rows, matched_flights=["Forest"])

        assert [(entry["flight"], entry["reason_kinds"])
                for entry in queue] == [("Forest", ["no_partner"])]

    def test_the_rows_are_not_changed(self):
        rows = [row(1, mean_confidence=0.2)]

        rq.build(rows)

        assert "reasons" not in rows[0]

    def test_nothing_to_queue(self):
        assert rq.build([]) == []
        assert rq.build([row(1), row(2)]) == []


class TestSummary:
    def test_the_queue_in_a_sentence(self):
        queue = rq.build([row(1, mean_confidence=0.2),
                          row(2, mean_confidence=0.3, n_boxes=2,
                              n_occluded=0),
                          row(3, species_vote_fraction=0.5)])

        assert rq.summarise(queue, 12) == (
            "3 of 12 track(s) are worth a look: 2 low confidence, 1 very "
            "short, 1 narrow species vote")

    def test_an_empty_queue(self):
        assert rq.summarise([], 12) == (
            "None of the 12 track(s) is in doubt - or they are all approved.")


def test_who_can_be_in_the_queue():
    assert rq.in_queue(row())
    assert not rq.in_queue(row(approved=True))
    assert not rq.in_queue(row(label_track_ids="3; 4"))
    assert not rq.in_queue(row(species="Not-An-Animal"))


# ---------------------------------------------------------------------------
# What the inventory knows
# ---------------------------------------------------------------------------

class TestInventory:
    @pytest.fixture
    def flight(self, tmp_path):
        from .survey_flights import make_flight
        return make_flight(str(tmp_path / "flight"), [
            {"id": 1, "species": "roe deer", "at": (0, 0), "frames": (1, 5)},
            {"id": 2, "species": "roe deer", "at": (60, 0), "frames": (2, 6)},
        ])

    def test_what_the_detector_called_it_stays_on_record(self, flight):
        """The classifier names the animal - and its boxes: what the
        detector's classes said is read from the classes themselves."""
        from bambi_wildlife_detection.core import (
            apply_results, track_inventory)
        from .survey_flights import species_id

        red = species_id(flight, "red deer")
        apply_results.apply_track_species(flight, "t", [
            {"track_id": 1, "label": "red deer", "votes": 3, "n": 3,
             "fraction": 1.0, "model": "species.pt"}], {"red deer": red},
            overwrite_detections=True)

        rows = {r["track_id"]: r
                for r in track_inventory.build_inventory(flight, "t")}

        assert rows[1]["species"] == "red deer"
        assert rows[1]["detector_species"] == "roe deer"
        assert rows[2]["detector_species"] == "roe deer"
        assert rq.reasons(dict(rows[1], n_boxes=30)) == [
            ("detector_species", "the detector calls it roe deer")]

    def test_a_detector_without_mapped_classes_says_nothing(self, tmp_path):
        from bambi_wildlife_detection.core import (
            detection_store, track_inventory, track_store)

        folder = str(tmp_path / "plain")
        detection_store.record_detections(folder, "t", [
            {"frame": f, "x1": 0.0, "y1": 0.0, "x2": 5.0, "y2": 5.0,
             "confidence": 0.9, "source_class": "0"} for f in range(3)])
        ids = [d["detection_id"]
               for d in track_store.load_detections(folder, "t")]
        track_store.record_tracks(folder, "t", [
            {"track_id": 1, "detection_id": i} for i in ids])

        rows = track_inventory.build_inventory(folder, "t")

        assert rows[0]["detector_species"] == ""

    def test_a_field_set_by_hand_says_so(self, flight):
        from bambi_wildlife_detection.core import (
            classification_store, track_inventory, track_review)
        from .survey_flights import species_id

        classification_store.record_track_predictions(flight, "t", "species", [
            {"track_id": 1, "label": "roe deer", "votes": 2, "n": 4,
             "fraction": 0.5, "model": "species.pt"},
            {"track_id": 2, "label": "roe deer", "votes": 2, "n": 4,
             "fraction": 0.5, "model": "species.pt"}])
        track_review.set_species(flight, "t", 1,
                                 species_id(flight, "red deer"))
        track_review.set_attribute(flight, "t", 1, "sex", None)

        rows = {r["track_id"]: r
                for r in track_inventory.build_inventory(flight, "t")}

        assert rows[1]["by_hand"] == "species; sex"
        assert rows[1]["species_model"] == "by hand"
        assert rows[1]["species_vote_fraction"] is None
        assert rows[1]["species_votes"] == ""
        assert rows[1]["sex_model"] == "by hand"
        assert rows[1]["attributes"] == ""          # the mark is not "other"
        assert (rows[2]["by_hand"], rows[2]["species_model"]) == (
            "", "species.pt")
        assert rows[2]["species_vote_fraction"] == 0.5
        # the vote was narrow on both: only the one nobody looked at is asked
        queue = rq.build(list(rows.values()), settings={"min_boxes": 1})
        assert [entry["track_id"] for entry in queue] == [2]

    def test_the_inspector_shows_the_new_columns_only_where_they_say_something(
            self, flight):
        from bambi_wildlife_detection.core import track_inventory

        assert "detector_species" in track_inventory.COLUMNS
        assert "by_hand" in track_inventory.COLUMNS
        rows = track_inventory.build_inventory(flight, "t")
        assert track_inventory.cell_text(rows[0]["by_hand"]) == ""


# ---------------------------------------------------------------------------
# The queue of a survey
# ---------------------------------------------------------------------------

class TestSurvey:
    TRACKS = [
        {"id": 1, "species": "roe deer", "at": (0, 0), "frames": (1, 5)},
        {"id": 2, "species": "roe deer", "at": (60, 0), "frames": (2, 6)},
        {"id": 3, "species": "red deer", "at": (90, 0), "frames": (3, 7)},
    ]
    LOOSE = {"min_boxes": 1}        # the tracks of the helper have two boxes

    @pytest.fixture
    def flight(self, tmp_path):
        """Both cameras: the RGB one calls track 1 a red deer, saw track 2
        as the thermal one did, and has no track for the third."""
        from bambi_wildlife_detection.core import match_store
        from .survey_flights import make_flight

        folder = make_flight(str(tmp_path / "Meadow"), self.TRACKS)
        make_flight(folder, [
            {"id": 1, "species": "roe deer", "at": (60, 0), "frames": (2, 6)},
            {"id": 2, "species": "red deer", "at": (0, 0),
             "frames": (1, 5)}], modality="w")
        match_store.record_matches(folder, [
            {"track_id_t": 1, "track_id_w": 2, "shared": 2,
             "median_dist": 1.0},
            {"track_id_t": 2, "track_id_w": 1, "shared": 2,
             "median_dist": 1.0}])
        return folder

    def test_the_other_camera_is_asked(self, flight):
        queue, total = rq.for_flights([("Meadow", flight, None)], "t",
                                      settings=self.LOOSE)

        assert total == 3
        assert [(entry["track_id"], entry["reason_kinds"])
                for entry in queue] == [
            (1, ["cameras_species"]), (3, ["no_partner"])]
        assert queue[0]["reasons"] == ["the RGB camera says red deer"]
        assert queue[0]["flight"] is None

    def test_from_the_other_side(self, flight):
        queue, total = rq.for_flights([("Meadow", flight, None)], "w",
                                      settings=self.LOOSE)

        assert total == 2
        assert [(entry["track_id"], entry["reason_kinds"])
                for entry in queue] == [(2, ["cameras_species"])]
        assert queue[0]["reasons"] == [
            "the thermal camera says roe deer"]

    def test_cameras_that_were_never_matched_are_not_held_against_a_track(
            self, tmp_path):
        from .survey_flights import make_flight

        folder = make_flight(str(tmp_path / "Forest"), self.TRACKS)

        queue, total = rq.for_flights([("Forest", folder, None)], "t",
                                      settings=self.LOOSE)

        assert (queue, total) == ([], 3)

    def test_a_survey_of_several_flights_names_the_flight(self, flight,
                                                          tmp_path):
        from .survey_flights import make_flight

        forest = make_flight(str(tmp_path / "Forest"), self.TRACKS[:1])

        queue, total = rq.for_flights(
            [("Meadow", flight, None), ("Forest", forest, None)], "t",
            several=True)

        assert total == 4
        assert sorted((entry["flight"], entry["track_id"])
                      for entry in queue) == [
            ("Forest", 1), ("Meadow", 1), ("Meadow", 2), ("Meadow", 3)]
        # the same track number in two flights: each asked its own camera
        by_key = {(entry["flight"], entry["track_id"]):
                  entry["reason_kinds"] for entry in queue}
        assert by_key[("Meadow", 1)] == ["short", "cameras_species"]
        assert by_key[("Forest", 1)] == ["short"]
        assert by_key[("Meadow", 3)] == ["short", "no_partner"]

    def test_the_species_of_the_survey(self, flight, tmp_path):
        from .survey_flights import make_flight, species_id

        forest = make_flight(str(tmp_path / "Forest"), self.TRACKS)

        queue, total = rq.for_flights(
            [("Meadow", flight, [species_id(flight, "red deer")]),
             ("Forest", forest, [])],            # none of them flew there
            "t", several=True)

        assert total == 1
        assert [(entry["flight"], entry["track_id"]) for entry in queue] == [
            ("Meadow", 3)]
