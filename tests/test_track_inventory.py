# -*- coding: utf-8 -*-
"""core.track_inventory - one row per tracked individual.

The fixture is a small flight that has been through every step: three
tracks, one geo-referenced end to end, one never placed on the ground, a
species vote, a sex call, per-frame occlusion, a cross-camera match and a
labelling-tool annotation - so every column has something to be checked
against.
"""
import csv
import json
import os

import pytest

from bambi_wildlife_detection.core import (
    apply_results, classification_store, detection_store, label_store,
    match_store, track_inventory, track_store)


@pytest.fixture
def flight(tmp_path):
    root = str(tmp_path)
    label_store.vocabulary(root, create=True)

    detection_store.record_detections(root, "t", [
        {"frame": 0, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0,
         "confidence": 0.9, "source_class": "0"},
        {"frame": 2, "x1": 12.0, "y1": 22.0, "x2": 32.0, "y2": 42.0,
         "confidence": 0.7, "source_class": "0"},
        {"frame": 4, "x1": 14.0, "y1": 24.0, "x2": 34.0, "y2": 44.0,
         "confidence": 0.8, "source_class": "0"},
        {"frame": 1, "x1": 50.0, "y1": 60.0, "x2": 70.0, "y2": 80.0,
         "confidence": 0.6, "source_class": "0"},
        {"frame": 3, "x1": 90.0, "y1": 90.0, "x2": 95.0, "y2": 95.0,
         "confidence": 0.5, "source_class": "0"},
    ])
    ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
    # Track 1 placed on the ground on every frame, track 2 never, track 3 once.
    track_store.record_georeference(root, "t", [
        {"detection_id": ids[0], "gx1": 500000.0, "gy1": 5300000.0, "gz1": 400.0,
         "gx2": 500002.0, "gy2": 5300002.0, "gz2": 400.0},
        {"detection_id": ids[1], "gx1": 500003.0, "gy1": 5300004.0, "gz1": 401.0,
         "gx2": 500005.0, "gy2": 5300006.0, "gz2": 401.0},
        {"detection_id": ids[2], "gx1": 500006.0, "gy1": 5300008.0, "gz1": 402.0,
         "gx2": 500008.0, "gy2": 5300010.0, "gz2": 402.0},
        {"detection_id": ids[4], "gx1": 500100.0, "gy1": 5300100.0, "gz1": 400.0,
         "gx2": 500102.0, "gy2": 5300102.0, "gz2": 400.0},
    ])
    track_store.record_tracks(root, "t", [
        {"track_id": 1, "detection_id": ids[0]},
        {"track_id": 1, "detection_id": ids[1]},
        {"track_id": 1, "detection_id": ids[2]},
        {"track_id": 2, "detection_id": ids[3]},
        {"track_id": 3, "detection_id": ids[4]},
    ])
    with open(os.path.join(root, "poses_t.json"), "w", encoding="utf-8") as fh:
        json.dump({"images": [
            {"imagefile": f"f{i}.jpg",
             "timestamp": f"2026-06-26T08:03:{32 + i:02d}+00:00"}
            for i in range(5)]}, fh)

    vocabulary = label_store.vocabulary(root)
    species = vocabulary["species_by_name"]
    enums = vocabulary["enum_ids"]
    classification_store.start_embedding_run(
        root, "t", backbone="b", dim=4, crop_size=8, padding=0.0,
        projection="non_geo", folder="embeddings_t/non_geo")
    classification_store.record_track_predictions(root, "t", "species", [
        {"track_id": 1, "label": "red_deer", "votes": 2, "n": 3,
         "fraction": 0.67, "model": "species_matched.pt"}])
    classification_store.record_track_predictions(root, "t", "sex", [
        {"track_id": 1, "label": "male", "votes": 3, "n": 3,
         "fraction": 1.0, "model": "sex_matched.pt"}])
    classification_store.record_track_predictions(
        root, "t", classification_store.LIFE_STAGE, [
            {"track_id": 1, "label": "adult", "votes": 1, "n": 1,
             "fraction": 1.0, "model": classification_store.SIZE_MODEL,
             "evidence": {"source": "size"}}])
    apply_results.apply_track_species(
        root, "t", [{"track_id": 1, "label": "red_deer"}],
        {"red_deer": species["red deer"]})
    apply_results.apply_track_attribute(
        root, "t", "sex", [{"track_id": 1, "label": "male"}],
        {"male": enums["sex"]["male"]})
    apply_results.apply_track_attribute(
        root, "t", "age", [{"track_id": 1, "label": "adult"}],
        {"adult": enums["age"]["adult"]})
    apply_results.apply_occlusion(
        root, "t", [{"detection_id": ids[1], "label": "occluded"},
                    {"detection_id": ids[2], "label": "clear"}],
        {"occluded": enums["occlusion"]["occluded"],
         "clear": enums["occlusion"]["clear"]})

    match_store.record_matches(root, [{
        "track_id_t": 1, "track_id_w": 7, "shared": 3, "median_dist": 0.4,
        "conf_t": 0.8, "conf_w": 0.7,
        "pairs": [{"frame_t": 0, "frame_w": 0, "detection_id_t": ids[0],
                   "detection_id_w": 1, "dist": 0.4}]}])

    # Imported from track 1 and annotated, but not materialised (track_id
    # None): the pipeline track stays the animal of record.
    label_store.save_tracks(root, "t", [{
        "label_track_id": 11, "track_id": None, "origin_track_id": 1,
        "species_id": species["red deer"],
        "attributes": {"sex": enums["sex"]["female"]},
        "keyframes": [{"frame": 0, "x1": 10, "y1": 20, "x2": 30, "y2": 40,
                       "stop": 0, "origin_detection_id": ids[0],
                       "attributes": {}}],
    }])

    route = os.path.join(root, "flight_route_t")
    os.makedirs(route)
    with open(os.path.join(route, "perpendicular_tracks_t.json"), "w") as fh:
        json.dump([{"track_id": 1, "distance": 12.5}], fh)
    return root


def _by_track(rows):
    return {row["track_id"]: row for row in rows}


class TestBuildInventory:

    def test_one_row_per_track(self, flight):
        rows = track_inventory.build_inventory(flight, "t", epsg=32633)
        assert [row["track_id"] for row in rows] == [1, 2, 3]
        assert all(row["camera"] == "thermal" for row in rows)

    def test_frames_times_and_boxes(self, flight):
        row = _by_track(track_inventory.build_inventory(flight, "t"))[1]
        assert (row["start_frame"], row["end_frame"]) == (0, 4)
        assert row["n_frames"] == 3 and row["frame_span"] == 5
        assert row["start_time"].startswith("2026-06-26T08:03:32")
        assert row["end_time"].startswith("2026-06-26T08:03:36")
        assert row["duration_s"] == pytest.approx(4.0)
        assert row["n_boxes"] == 3
        assert row["mean_confidence"] == pytest.approx(0.8)
        assert (row["min_confidence"], row["max_confidence"]) == (0.7, 0.9)
        assert row["mean_box_width_px"] == pytest.approx(20.0)

    def test_occlusion_counts(self, flight):
        row = _by_track(track_inventory.build_inventory(flight, "t"))[1]
        assert (row["n_occluded"], row["n_clear"], row["n_occlusion_unknown"]) == (1, 1, 1)

    def test_positions_in_the_project_crs_and_wgs84(self, flight):
        pytest.importorskip("pyproj")
        row = _by_track(track_inventory.build_inventory(flight, "t", epsg=32633))[1]
        assert (row["start_x"], row["start_y"], row["start_z"]) == (500001.0, 5300001.0, 400.0)
        assert (row["end_x"], row["end_y"]) == (500007.0, 5300009.0)
        assert row["crs"] == "EPSG:32633"
        assert row["displacement_m"] == pytest.approx((6 ** 2 + 8 ** 2) ** 0.5)
        assert row["path_length_m"] == pytest.approx(2 * 5.0)
        assert 47.0 < row["start_lat"] < 48.5 and 14.5 < row["start_lon"] < 15.5
        assert row["n_georeferenced"] == 3

    def test_an_unplaced_track_has_empty_positions(self, flight):
        row = _by_track(track_inventory.build_inventory(flight, "t", epsg=32633))[2]
        assert row["start_x"] is None and row["start_lat"] is None
        assert row["path_length_m"] is None and row["crs"] == ""
        assert row["n_georeferenced"] == 0

    def test_species_sex_and_age_with_their_votes(self, flight):
        rows = _by_track(track_inventory.build_inventory(flight, "t"))
        row = rows[1]
        assert row["species"] == "red deer"
        assert row["species_vote_fraction"] == pytest.approx(0.67)
        assert row["species_votes"] == "2/3"
        assert row["species_model"] == "species_matched.pt"
        assert row["sex"] == "male" and row["sex_vote_fraction"] == 1.0
        assert row["age"] == "adult" and row["age_source"] == "size"
        unclassified = rows[2]
        assert unclassified["species"] == "animal"
        assert unclassified["sex"] == "" and unclassified["species_votes"] == ""

    def test_match_labels_and_route_distance(self, flight):
        rows = _by_track(track_inventory.build_inventory(flight, "t"))
        row = rows[1]
        assert row["matched_track"] == 7 and row["matched_camera"] == "rgb"
        assert row["label_track_ids"] == "11"
        assert row["label_species"] == "red deer"
        assert row["label_sex"] == "female"          # the annotator disagreed
        assert row["perpendicular_distance_m"] == 12.5
        other = rows[3]
        assert other["matched_track"] is None and other["label_track_ids"] == ""
        assert other["perpendicular_distance_m"] is None

    def test_without_pyproj_the_wgs84_columns_stay_empty(self, flight, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "pyproj", None)
        row = _by_track(track_inventory.build_inventory(flight, "t", epsg=32633))[1]
        assert row["start_lat"] is None and row["start_x"] == 500001.0


class TestFiles:

    def test_csv_has_every_column_and_row(self, flight, tmp_path):
        rows = track_inventory.build_inventory(flight, "t", epsg=32633)
        path = track_inventory.write_csv(rows, str(tmp_path / "out" / "inv.csv"))
        with open(path, newline="", encoding="utf-8") as fh:
            table = list(csv.DictReader(fh))
        assert list(table[0].keys()) == track_inventory.COLUMNS
        assert [t["track_id"] for t in table] == ["1", "2", "3"]
        assert table[0]["species"] == "red deer"
        assert table[1]["start_x"] == ""              # None writes as empty
        assert table[0]["mean_confidence"] == "0.8"

    def test_json_round_trips(self, flight, tmp_path):
        rows = track_inventory.build_inventory(flight, "t", epsg=32633)
        path = track_inventory.write_json(rows, str(tmp_path / "inv.json"))
        assert track_inventory.read_json(path) == json.loads(json.dumps(rows))

    def test_summary_lines(self, flight):
        rows = track_inventory.build_inventory(flight, "t", epsg=32633)
        lines = track_inventory.summarise(rows)
        assert lines[0].startswith("3 tracked individual(s): ")
        assert "1 red deer" in lines[0]
        assert "1 sexed, 1 aged, 1 matched" in lines[1]
        assert track_inventory.summarise([]) == ["No tracks to list."]


class TestApproved:
    """The user's own verdict is the first column, written onto the track."""

    def test_unset_is_false_and_absent_from_attributes(self, flight):
        rows = _by_track(track_inventory.build_inventory(flight, "t"))
        assert rows[1]["approved"] is False
        assert track_inventory.COLUMNS[0] == "approved"

    def test_set_and_clear_round_trip(self, flight):
        assert track_store.set_track_attribute(flight, "t", 1, "approved", True)
        assert not track_store.set_track_attribute(flight, "t", 1, "approved", True)
        rows = _by_track(track_inventory.build_inventory(flight, "t"))
        assert rows[1]["approved"] is True and rows[2]["approved"] is False
        assert rows[1]["attributes"] == ""          # not repeated as an extra
        assert "1 of 3 approved" in track_inventory.summarise(list(rows.values()))[-1]
        assert track_store.set_track_attribute(flight, "t", 1, "approved", None)
        assert _by_track(track_inventory.build_inventory(flight, "t"))[1]["approved"] is False

    def test_unknown_track_or_missing_store_changes_nothing(self, flight, tmp_path):
        assert not track_store.set_track_attribute(flight, "t", 999, "approved", True)
        assert not track_store.set_track_attribute(str(tmp_path / "x"), "t", 1, "approved", True)

    def test_csv_writes_yes_no(self, flight, tmp_path):
        track_store.set_track_attribute(flight, "t", 1, "approved", True)
        rows = track_inventory.build_inventory(flight, "t")
        path = track_inventory.write_csv(rows, str(tmp_path / "inv.csv"))
        with open(path, newline="", encoding="utf-8") as fh:
            table = list(csv.DictReader(fh))
        assert [t["approved"] for t in table] == ["yes", "no", "no"]


class TestSpeciesFilter:
    """The Survey Analytics species filter applies to the inventory too."""

    def test_only_selected_species_are_listed(self, flight):
        from bambi_wildlife_detection.core import label_store
        red_deer = label_store.vocabulary(flight)["species_by_name"]["red deer"]
        rows = track_inventory.build_inventory(flight, "t", species_ids=[red_deer])
        assert [row["track_id"] for row in rows] == [1]
        assert track_inventory.build_inventory(flight, "t", species_ids=[]) == []

    def test_none_means_every_track(self, flight):
        assert len(track_inventory.build_inventory(flight, "t", species_ids=None)) == 3
