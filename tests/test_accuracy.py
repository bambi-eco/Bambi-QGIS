# -*- coding: utf-8 -*-
"""The pipeline's output against hand labels (core/accuracy.py)."""
import csv
import os

import pytest

from bambi_wildlife_detection.core import accuracy as acc

from .survey_flights import make_labelled_flight


def box_at(x, y, size=10.0):
    return (float(x), float(y), float(x) + size, float(y) + size)


def label(ident, frames, at=(100, 100), species="roe deer", **more):
    """A labelled animal standing still at *at* over *frames*."""
    return dict({"id": ident, "species": species, "sex": None, "age": None,
                 "imported": False,
                 "boxes": {frame: box_at(*at) for frame in frames}}, **more)


def found(ident, frame, at=(100, 100), confidence=0.9, track=None,
          species="roe deer", size=10.0):
    return {"id": ident, "frame": frame, "box": box_at(*at, size=size),
            "confidence": confidence, "species": species, "track": track}


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------

class TestOverlap:
    def test_the_same_box_overlaps_fully(self):
        assert acc.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_boxes_apart_do_not_overlap(self):
        assert acc.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
        assert acc.iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0   # touching

    def test_half_shifted(self):
        # 50 shared of 150 covered
        assert acc.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)

    def test_corners_in_any_order(self):
        assert acc.iou((10, 10, 0, 0), (0, 0, 10, 10)) == 1.0

    def test_a_box_without_size_overlaps_nothing(self):
        assert acc.iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


class TestPairs:
    def test_each_box_is_paired_once_the_best_pairs_first(self):
        labels = [box_at(0, 0), box_at(8, 0)]
        boxes = [box_at(7, 0), box_at(1, 0)]

        pairs = acc.pair_boxes(labels, boxes, 0.1)

        assert sorted((i, j) for i, j, _o in pairs) == [(0, 1), (1, 0)]

    def test_two_labels_cannot_share_a_box(self):
        labels = [box_at(0, 0), box_at(2, 0)]
        boxes = [box_at(1, 0)]

        pairs = acc.pair_boxes(labels, boxes, 0.1)

        assert len(pairs) == 1

    def test_a_pair_needs_the_overlap(self):
        labels, boxes = [box_at(0, 0)], [box_at(5, 0)]

        assert acc.pair_boxes(labels, boxes, 0.3) == [
            (0, 0, pytest.approx(1 / 3))]
        assert acc.pair_boxes(labels, boxes, 0.4) == []

    def test_nothing_to_pair(self):
        assert acc.pair_boxes([], [box_at(0, 0)], 0.3) == []
        assert acc.pair_boxes([box_at(0, 0)], [], 0.3) == []


def test_scores_from_the_counts():
    result = acc.scores(found=8, missed=2, false_alarms=8)

    assert result["precision"] == 0.5 and result["recall"] == 0.8
    assert result["f1"] == pytest.approx(2 * 0.5 * 0.8 / 1.3)
    nothing = acc.scores(0, 0, 0)
    assert nothing["precision"] is None and nothing["f1"] is None
    assert acc.scores(0, 3, 2)["f1"] is None


# ---------------------------------------------------------------------------
# Were the animals found
# ---------------------------------------------------------------------------

class TestDetection:
    def test_found_missed_and_false_alarms(self):
        labels = [label(1, range(0, 4))]
        boxes = [found(10, 0), found(11, 1), found(12, 3, at=(300, 300)),
                 found(13, 2, at=(300, 300))]

        result = acc.compare(labels, boxes)["detection"]

        assert (result["found"], result["missed"],
                result["false_alarms"]) == (2, 2, 2)
        assert result["precision"] == 0.5 and result["recall"] == 0.5
        assert result["mean_iou"] == 1.0

    def test_the_overlap_that_is_asked_for_decides(self):
        labels = [label(1, [0])]
        boxes = [found(10, 0, at=(105, 100))]            # overlap 1/3

        assert acc.compare(labels, boxes, threshold=0.3)[
            "detection"]["found"] == 1
        strict = acc.compare(labels, boxes, threshold=0.5)["detection"]
        assert (strict["found"], strict["missed"],
                strict["false_alarms"]) == (0, 1, 1)

    def test_only_the_frames_that_were_gone_through_count(self):
        """A box before the first label or after the last says nothing:
        nobody is known to have looked there."""
        labels = [label(1, range(10, 13))]
        boxes = [found(1, 5, at=(300, 300)), found(2, 10),
                 found(3, 40, at=(300, 300))]

        document = acc.compare(labels, boxes)

        assert document["frames"] == {
            "first": 10, "last": 12, "given": False, "with_labels": 3,
            "with_boxes": 1}
        assert document["detection"]["false_alarms"] == 0

    def test_the_frames_can_be_named(self):
        labels = [label(1, range(10, 13))]
        boxes = [found(1, 5, at=(300, 300)), found(2, 10),
                 found(3, 40, at=(300, 300))]

        document = acc.compare(labels, boxes, frames=(0, 50))

        assert document["frames"]["given"]
        assert document["detection"]["false_alarms"] == 2
        assert document["detection"]["found"] == 1

    def test_something_a_reviewer_ruled_out_is_a_confirmed_false_alarm(self):
        labels = [label(1, [0]),
                  label(2, [0, 1], at=(300, 300), species="not-an-animal")]
        boxes = [found(10, 0), found(11, 0, at=(300, 300))]

        document = acc.compare(labels, boxes)
        result = document["detection"]

        assert (result["found"], result["missed"]) == (1, 0)
        assert result["false_alarms"] == 1
        assert result["confirmed_false_alarms"] == 1
        # ... and it is no animal that could have been missed
        assert document["labels"] == {
            "animals": 1, "ruled_out": 1, "imported": 0, "boxes": 1}
        kinds = sorted(row["kind"] for row in document["rows"])
        assert kinds == ["found", "ruled out", "ruled out, no box"]

    def test_recall_per_labelled_species(self):
        labels = [label(1, [0, 1]), label(2, [0, 1], at=(300, 300),
                                          species="red deer")]
        boxes = [found(10, 0), found(11, 1), found(12, 0, at=(300, 300))]

        by_species = acc.compare(labels, boxes)["detection"]["by_species"]

        assert by_species["roe deer"] == {
            "found": 2, "missed": 0, "recall": 1.0}
        assert by_species["red deer"] == {
            "found": 1, "missed": 1, "recall": 0.5}

    def test_precision_per_species_the_detector_called(self):
        labels = [label(1, [0, 1])]
        boxes = [found(10, 0, species="chamois"), found(11, 1),
                 found(12, 1, at=(300, 300), species="chamois")]

        called = acc.compare(labels, boxes)["detection"]["by_called_species"]

        assert called["chamois"] == {
            "animals": 1, "false_alarms": 1, "precision": 0.5}
        assert called["roe deer"]["precision"] == 1.0

    def test_what_a_confidence_threshold_would_have_done(self):
        labels = [label(1, range(0, 4))]
        boxes = [found(10, 0, confidence=0.95), found(11, 1, confidence=0.6),
                 found(12, 2, confidence=0.35),
                 found(13, 3, at=(300, 300), confidence=0.15),
                 found(14, 0, at=(300, 300), confidence=0.25)]

        result = acc.compare(labels, boxes)["detection"]
        table = {row["confidence"]: row for row in result["by_confidence"]}

        assert (table[0.0]["found"], table[0.0]["false_alarms"]) == (3, 2)
        assert (table[0.3]["found"], table[0.3]["false_alarms"],
                table[0.3]["missed"]) == (3, 0, 1)
        assert (table[0.5]["found"], table[0.5]["missed"]) == (2, 2)
        assert table[0.9]["recall"] == 0.25
        assert result["best_confidence"] == 0.3

    def test_a_box_without_a_confidence_passes_every_threshold(self):
        labels = [label(1, [0])]
        boxes = [found(10, 0, confidence=None)]

        table = acc.compare(labels, boxes)["detection"]["by_confidence"]

        assert all(row["found"] == 1 for row in table)

    def test_labels_but_no_boxes(self):
        result = acc.compare([label(1, [0, 1])], [])["detection"]

        assert (result["found"], result["missed"]) == (0, 2)
        assert result["recall"] == 0.0 and result["precision"] is None
        assert result["best_confidence"] is None

    def test_without_labels_there_is_nothing_to_compare(self):
        with pytest.raises(acc.AccuracyError, match="no hand labels"):
            acc.compare([], [found(1, 0)])
        with pytest.raises(acc.AccuracyError, match="frames 50 to 60"):
            acc.compare([label(1, [0])], [], frames=(50, 60))


# ---------------------------------------------------------------------------
# Was each animal one track
# ---------------------------------------------------------------------------

class TestTracking:
    def test_an_animal_followed_by_one_track(self):
        labels = [label(1, range(0, 5))]
        boxes = [found(10 + f, f, track=7) for f in range(0, 4)]

        tracking = acc.compare(labels, boxes, {7: {"species": "roe deer"}})[
            "tracking"]

        assert tracking["animals_tracked"] == 1
        assert tracking["animals_in_pieces"] == 0
        assert tracking["tracks"] == tracking["tracks_on_animals"] == 1
        assert tracking["count_error"] == 0
        animal = tracking["per_animal"][0]
        assert (animal["frames"], animal["frames_with_box"],
                animal["frames_tracked"]) == (5, 4, 4)
        assert animal["share_tracked"] == 0.8
        assert animal["best_track"] == 7 and animal["pieces"] == 1

    def test_an_animal_in_two_pieces_is_counted_twice(self):
        labels = [label(1, range(0, 6))]
        boxes = [found(10 + f, f, track=7 if f < 3 else 8) for f in range(6)]

        tracking = acc.compare(labels, boxes, {7: {}, 8: {}})["tracking"]

        assert tracking["animals_in_pieces"] == 1
        assert tracking["per_animal"][0]["tracks"] == [7, 8]
        assert (tracking["tracks"], tracking["count_error"]) == (2, 1)

    def test_a_track_on_nothing_that_was_labelled(self):
        labels = [label(1, range(0, 3))]
        boxes = [found(10 + f, f, track=7) for f in range(3)]
        boxes += [found(20 + f, f, at=(300, 300), track=9) for f in range(3)]

        tracking = acc.compare(labels, boxes, {7: {}, 9: {}})["tracking"]

        assert tracking["tracks_on_nothing"] == 1
        assert tracking["count_error"] == 1
        assert [t["animals"] for t in tracking["per_track"]] == [[1], []]

    def test_one_track_over_two_animals(self):
        labels = [label(1, [0, 1]), label(2, [2, 3], at=(300, 300))]
        boxes = [found(10, 0, track=7), found(11, 1, track=7),
                 found(12, 2, at=(300, 300), track=7),
                 found(13, 3, at=(300, 300), track=7)]

        tracking = acc.compare(labels, boxes, {7: {}})["tracking"]

        assert tracking["tracks_on_several_animals"] == 1
        assert tracking["animals_tracked"] == 2
        assert tracking["count_error"] == -1

    def test_an_animal_with_boxes_that_no_tracker_joined(self):
        labels = [label(1, range(0, 3))]
        boxes = [found(10 + f, f) for f in range(3)]

        tracking = acc.compare(labels, boxes)["tracking"]

        assert tracking["animals_not_tracked"] == 1
        assert tracking["per_animal"][0]["frames_with_box"] == 3
        assert tracking["per_animal"][0]["best_track"] is None
        assert tracking["mean_share_of_time_tracked"] is None

    def test_a_track_the_pipeline_calls_no_animal_is_not_counted(self):
        labels = [label(1, range(0, 3)),
                  label(2, range(0, 3), at=(300, 300),
                        species="not-an-animal")]
        boxes = [found(10 + f, f, track=7) for f in range(3)]
        boxes += [found(20 + f, f, at=(300, 300), track=9) for f in range(3)]

        tracking = acc.compare(labels, boxes, {
            7: {"species": "roe deer"},
            9: {"species": "not-an-animal"}})["tracking"]

        assert tracking["tracks"] == 1 and tracking["count_error"] == 0
        on_rock = next(t for t in tracking["per_track"] if t["track"] == 9)
        assert on_rock["frames_on_ruled_out"] == 3

    def test_tracks_outside_the_frames_are_not_looked_at(self):
        labels = [label(1, [10, 11])]
        boxes = [found(1, 10, track=7), found(2, 90, at=(300, 300), track=8)]

        tracking = acc.compare(labels, boxes, {7: {}, 8: {}})["tracking"]

        assert [t["track"] for t in tracking["per_track"]] == [7]


# ---------------------------------------------------------------------------
# Was it named right
# ---------------------------------------------------------------------------

class TestNaming:
    def test_a_confusion_table(self):
        table = acc.confusion([
            ("roe deer", "roe deer"), ("roe deer", "roe deer"),
            ("roe deer", "red deer"), ("red deer", "red deer"),
            ("red deer", "undetermined")])

        assert table["labels"] == ["red deer", "roe deer"]
        assert table["called"] == ["red deer", "roe deer", "undetermined"]
        assert table["counts"]["roe deer"] == {"roe deer": 2, "red deer": 1}
        assert (table["n"], table["right"]) == (5, 3)
        assert table["accuracy"] == 0.6
        assert table["per_label"]["roe deer"] == {
            "n": 3, "right": 2, "recall": pytest.approx(2 / 3),
            "precision": 1.0}
        assert table["per_label"]["red deer"]["precision"] == 0.5

    def test_no_pairs_no_accuracy(self):
        table = acc.confusion([])

        assert table["n"] == 0 and table["accuracy"] is None
        assert table["labels"] == []

    def test_the_species_of_the_track_that_follows_the_animal(self):
        labels = [label(1, range(0, 4)),
                  label(2, range(0, 4), at=(300, 300), species="red deer"),
                  label(3, range(0, 4), at=(500, 500), species="red deer")]
        boxes = [found(10 + f, f, track=7) for f in range(4)]
        boxes += [found(20 + f, f, at=(300, 300), track=8) for f in range(4)]
        tracks = {7: {"species": "roe deer"}, 8: {"species": "roe deer"}}

        species = acc.compare(labels, boxes, tracks)["naming"]["species"]

        assert species["counts"] == {
            "roe deer": {"roe deer": 1},
            "red deer": {"roe deer": 1, "(no track)": 1}}
        assert species["accuracy"] == pytest.approx(1 / 3)

    def test_a_label_that_does_not_say_is_not_asked(self):
        labels = [label(1, [0], species="animal"),
                  label(2, [0], at=(300, 300), species="unknown")]
        boxes = [found(10, 0, track=7), found(11, 0, at=(300, 300), track=8)]

        naming = acc.compare(labels, boxes, {7: {"species": "roe deer"},
                                             8: {}})["naming"]

        assert naming["species"]["n"] == 0

    def test_a_track_nobody_named_is_undetermined(self):
        labels = [label(1, [0])]
        boxes = [found(10, 0, track=7)]

        species = acc.compare(labels, boxes, {7: {"species": "animal"}})[
            "naming"]["species"]

        assert species["counts"] == {"roe deer": {"undetermined": 1}}

    def test_the_track_with_most_of_the_animal_names_it(self):
        labels = [label(1, range(0, 5))]
        boxes = [found(10 + f, f, track=7 if f < 2 else 8) for f in range(5)]
        tracks = {7: {"species": "red deer"}, 8: {"species": "roe deer"}}

        species = acc.compare(labels, boxes, tracks)["naming"]["species"]

        assert species["counts"] == {"roe deer": {"roe deer": 1}}

    def test_sex_and_age(self):
        labels = [label(1, [0], sex="Female", age="adult"),
                  label(2, [0], at=(300, 300), sex="male", age=None)]
        boxes = [found(10, 0, track=7), found(11, 0, at=(300, 300), track=8)]
        tracks = {7: {"sex": "female", "age": "juvenile"},
                  8: {"sex": "female", "age": "adult"}}

        naming = acc.compare(labels, boxes, tracks)["naming"]

        assert naming["sex"]["counts"] == {
            "female": {"female": 1}, "male": {"female": 1}}
        assert naming["sex"]["accuracy"] == 0.5
        assert naming["age"]["counts"] == {"adult": {"juvenile": 1}}

    def test_what_each_classifier_model_said(self):
        labels = [label(1, [0]), label(2, [0], at=(300, 300)),
                  label(3, [0], at=(500, 500), species="red deer")]
        boxes = [found(10, 0, track=7), found(11, 0, at=(300, 300), track=8),
                 found(12, 0, at=(500, 500), track=9)]
        classifiers = {"species": {
            7: {"label": "roe deer", "model": "/models/species_v2.pt"},
            8: {"label": "red deer", "model": "/models/species_v2.pt"},
            9: {"label": "red deer", "model": "other.pt"}}}

        by_model = acc.compare(labels, boxes, {7: {}, 8: {}, 9: {}},
                               classifiers=classifiers)["naming"]["by_model"]

        assert list(by_model["species"]) == ["other.pt", "species_v2.pt"]
        assert by_model["species"]["species_v2.pt"]["accuracy"] == 0.5
        assert by_model["species"]["other.pt"]["accuracy"] == 1.0
        assert "sex" not in by_model


# ---------------------------------------------------------------------------
# In words, and on disk
# ---------------------------------------------------------------------------

@pytest.fixture
def document():
    labels = [label(1, range(0, 6), sex="female"),
              label(2, range(0, 6), at=(300, 300), species="red deer",
                    imported=True),
              label(3, [2, 3], at=(500, 500), species="not-an-animal")]
    boxes = [found(10 + f, f, track=7 if f < 3 else 8) for f in range(6)]
    boxes += [found(30 + f, f, at=(300, 300), track=9, confidence=0.4)
              for f in range(4)]
    boxes += [found(50, 2, at=(500, 500), track=11, confidence=0.2),
              found(51, 5, at=(700, 700), confidence=0.15)]
    tracks = {7: {"species": "roe deer", "sex": "female"},
              8: {"species": "roe deer", "sex": "male"},
              9: {"species": "roe deer"}, 11: {"species": "roe deer"}}
    return acc.compare(labels, boxes, tracks)


def test_the_result_in_a_few_lines(document):
    lines = acc.describe(document)

    assert lines[0] == ("Compared with 2 labelled animal(s) (12 boxes) in "
                        "the frames 0 to 5, at an overlap of at least 0.3")
    assert lines[1] == ("Boxes: 10 on an animal, 2 animals missed, 2 false "
                        "alarm(s) - precision 83 %, recall 83 %")
    assert lines[2].endswith("confidence of at least 0.3")
    assert lines[3] == (
        "Tracks: 2 of 2 animals are followed by a track (1 in several "
        "pieces); 1 of 4 track(s) follow nothing that was labelled. A "
        "census counts 4 animals - 2 more than were labelled")
    assert lines[4] == "Species: 1 of 2 named as labelled (50 %)"
    assert lines[5] == "Sex: 1 of 1 named as labelled (100 %)"
    assert "flatters the figures" in lines[6]


def test_a_census_that_got_it_right_is_said_so():
    labels = [label(1, range(0, 3))]
    boxes = [found(10 + f, f, track=7) for f in range(3)]

    lines = acc.describe(acc.compare(labels, boxes, {7: {}}))

    assert any("as many as were labelled" in line for line in lines)


def test_every_box_is_listed(document, tmp_path):
    path = acc.write_rows(document, str(tmp_path / "boxes.csv"))
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0]) == acc.ROW_COLUMNS
    kinds = [row["kind"] for row in rows]
    assert kinds.count("found") == 10 and kinds.count("missed") == 2
    assert kinds.count("false alarm") == 1 and kinds.count("ruled out") == 1
    assert kinds.count("ruled out, no box") == 1
    missed = [row for row in rows if row["kind"] == "missed"]
    assert {row["frame"] for row in missed} == {"4", "5"}
    assert missed[0]["box"] == "" and missed[0]["label_species"] == "red deer"
    first = rows[0]
    assert (first["frame"], first["label"], first["box"], first["track"],
            first["iou"]) == ("0", "1", "10", "7", "1.0")


def test_the_document_survives_being_written_and_read(document, tmp_path):
    path = acc.write_json(document, str(tmp_path / "accuracy.json"))
    again = acc.read_json(path)

    assert acc.describe(again) == acc.describe(document)
    assert again["detection"]["by_confidence"][3]["confidence"] == 0.3
    assert os.path.getsize(path) > 500


def test_a_percentage():
    assert acc.percent(0.826) == "83 %" and acc.percent(None) == "-"
    assert acc.percent(1.0) == "100 %"


# ---------------------------------------------------------------------------
# A flight: labels, detections, tracks and classifier calls from the stores
# ---------------------------------------------------------------------------

@pytest.fixture
def flight(tmp_path):
    return make_labelled_flight(str(tmp_path / "flight"))


class TestFlight:
    def test_the_labels_are_a_box_per_frame(self, flight):
        labels = {entry["id"]: entry for entry in acc.load_labels(flight, "t")}

        assert sorted(labels) == [1, 2, 3]
        assert sorted(labels[1]["boxes"]) == [0, 1, 2, 3, 4, 5]
        assert labels[1]["boxes"][3] == (100.0, 100.0, 110.0, 110.0)
        assert (labels[1]["species"], labels[2]["species"],
                labels[3]["species"]) == ("roe deer", "red deer",
                                          "not-an-animal")
        assert [labels[i]["imported"] for i in (1, 2, 3)] == [
            False, True, False]

    def test_the_pipelines_own_boxes_are_compared_not_the_hand_drawn(
            self, flight):
        from bambi_wildlife_detection.core import track_store

        pipeline = acc.load_pipeline(flight, "t")

        # the labels were written into the stores too: 14 more boxes
        assert len(track_store.load_detections(flight, "t")) == 26
        assert len(pipeline["boxes"]) == 12
        assert pipeline["sources"] == [
            {"kind": "detector", "model": "yolo_wildlife.pt", "boxes": 12}]
        assert pipeline["tracker"] == {
            "kind": "builtin", "tracker": "builtin", "tracks": 4}

    def test_a_track_a_label_has_replaced_is_still_compared(self, flight):
        """The red deer's label was made from track 3: the survey tools now
        read the label, but the report is about what the tracker did."""
        from bambi_wildlife_detection.core import track_store

        assert 3 in track_store.superseded_track_ids(flight, "t")
        pipeline = acc.load_pipeline(flight, "t")

        assert sorted(pipeline["tracks"]) == [1, 2, 3, 4]
        assert pipeline["tracks"][3]["species"] == "roe deer"
        assert sum(1 for box in pipeline["boxes"] if box["track"] == 3) == 4

    def test_the_whole_comparison(self, flight):
        document = acc.for_flight(flight, "t")

        assert document["labels"] == {
            "animals": 2, "ruled_out": 1, "imported": 1, "boxes": 12}
        found = document["detection"]
        assert (found["found"], found["missed"], found["false_alarms"],
                found["confirmed_false_alarms"]) == (10, 2, 2, 2)
        assert found["by_species"]["red deer"]["recall"] == \
            pytest.approx(4 / 6)
        tracking = document["tracking"]
        assert (tracking["animals_tracked"], tracking["animals_in_pieces"],
                tracking["tracks"], tracking["tracks_on_nothing"],
                tracking["count_error"]) == (2, 1, 4, 1, 2)
        species = document["naming"]["species"]
        assert species["counts"] == {"roe deer": {"roe deer": 1},
                                     "red deer": {"roe deer": 1}}
        assert document["naming"]["by_model"]["species"][
            "species_v3.pt"]["accuracy"] == 0.5
        assert document["classifier_models"] == {
            "species": ["species_v3.pt"]}
        assert document["modality"] == "t"

    def test_sex_as_the_track_has_it(self, flight):
        document = acc.for_flight(flight, "t")

        sex = document["naming"]["sex"]
        if sex["n"]:                    # the schema has a sex enum
            assert sex["counts"] == {"female": {"female": 1}}

    def test_labels_not_yet_written_into_the_stores_compare_the_same(
            self, tmp_path):
        with_boxes = acc.for_flight(
            make_labelled_flight(str(tmp_path / "a")), "t")
        without = acc.for_flight(
            make_labelled_flight(str(tmp_path / "b"), materialise=False), "t")

        assert with_boxes["detection"] == without["detection"]
        assert with_boxes["tracking"]["count_error"] == \
            without["tracking"]["count_error"]

    def test_a_stricter_overlap_and_other_frames(self, flight):
        document = acc.for_flight(flight, "t", threshold=0.9, frames=(0, 2))

        assert document["frames"]["last"] == 2
        assert document["iou_threshold"] == 0.9
        assert document["detection"]["found"] == 6

    def test_a_flight_without_detections(self, tmp_path):
        from bambi_wildlife_detection.core import label_store

        folder = str(tmp_path / "empty")
        os.makedirs(folder)
        label_store.vocabulary(folder, create=True)

        with pytest.raises(acc.AccuracyError, match="Detect Animals"):
            acc.for_flight(folder, "t")

    def test_a_flight_without_labels(self, tmp_path):
        from tests.survey_flights import make_flight

        folder = make_flight(str(tmp_path / "unlabelled"), [
            {"id": 1, "species": "roe deer", "at": (0, 0), "frames": (1, 5)}])

        with pytest.raises(acc.AccuracyError, match="no hand labels"):
            acc.for_flight(folder, "t")


# ---------------------------------------------------------------------------
# As a page
# ---------------------------------------------------------------------------

class TestPage:
    def test_the_page_answers_the_three_questions(self, document):
        page = acc.html_report(document)

        assert page.startswith("<!DOCTYPE html>")
        for heading in ("Were the animals found?",
                        "Was each animal one track?",
                        "Was it named right?"):
            assert f"<h3>{heading}</h3>" in page
        assert "<td class='n'>83 %</td>" in page
        assert "If only boxes of at least this confidence were kept" in page
        assert "<tr class='best'><td class='n'>0.3</td>" in page
        assert "Animals that are not exactly one track" in page
        assert "Tracks that follow nothing that was labelled" in page
        assert "A census counts" in page and "4 (+2)" in page
        assert "Species: 1 of 2 as labelled (50 %)" in page
        assert "flatters the figures" in page

    def test_the_confusion_table_marks_what_was_right(self, document):
        page = acc.html_report(document)

        assert "<th>labelled \\ called</th>" in page
        assert "<tr><th>roe deer</th><td class='n hit'>1</td>" in page

    def test_what_was_compared_is_said(self, flight):
        page = acc.html_report(acc.for_flight(flight, "t"),
                               title="Meadow, thermal")

        assert "<h2>Meadow, thermal</h2>" in page
        assert "Boxes of the detector" in page and "yolo_wildlife.pt" in page
        assert "builtin - 4 track(s)" in page
        assert "Classifier (species)" in page and "species_v3.pt" in page
        assert "The classifier's own call (species_v3.pt): 1 of 2" in page
        assert "0 to 5 (first to last labelled frame)" in page

    def test_names_are_escaped(self):
        labels = [label(1, [0], species="<b>deer</b>")]
        boxes = [found(10, 0, track=7, species="a & b")]

        page = acc.html_report(acc.compare(
            labels, boxes, {7: {"species": "x<y"}}))

        assert "<b>deer</b>" not in page
        assert "&lt;b&gt;deer&lt;/b&gt;" in page and "a &amp; b" in page

    def test_a_part_of_a_page(self, document):
        part = acc.html_report(document, page=False)

        assert part.startswith("<h2>") and "<html>" not in part

    def test_nothing_named_is_said(self):
        labels = [label(1, [0], species="animal")]
        page = acc.html_report(acc.compare(labels, [found(10, 0)]))

        assert "No label says a species, sex or age" in page
        assert "Animals that are not exactly one track" in page

    def test_long_lists_are_cut(self):
        labels = [label(i, [0], at=(20 * i, 0)) for i in range(1, 320)]
        page = acc.html_report(acc.compare(labels, []))

        assert "... and 19 more; all of them are in the JSON file." in page
