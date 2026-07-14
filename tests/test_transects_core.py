# -*- coding: utf-8 -*-
"""Unit tests for core.transects (transect tool data model & geometry)."""
import csv
import json
import os

import pytest

from bambi_wildlife_detection.core.transects import (
    Transect,
    TransectStore,
    cumulative_distances,
    flight_positions,
    frame_after_distance,
    path_length,
)


def _images_along_x(n, spacing=10.0):
    """Poses flying east: frame i sits at x = i * spacing metres."""
    return [
        {"imagefile": f"{i:06d}.png",
         "location": [i * spacing, 0.0, 100.0],
         "timestamp": f"2026-07-14T10:00:{i:02d}+00:00"}
        for i in range(n)
    ]


class TestFlightPositions:
    def test_extracts_xy_and_ignores_altitude(self):
        images = [{"location": [3.0, 4.0, 120.0]}]
        assert flight_positions(images) == [(3.0, 4.0)]

    def test_missing_or_malformed_location_yields_none(self):
        images = [{}, {"location": None}, {"location": [1.0]},
                  {"location": ["a", "b"]}, {"location": [1.0, 2.0]}]
        assert flight_positions(images) == [
            None, None, None, None, (1.0, 2.0)]


class TestCumulativeDistances:
    def test_straight_line(self):
        cum = cumulative_distances([(0, 0), (3, 4), (6, 8)])
        assert cum == [0.0, 5.0, 10.0]

    def test_none_entries_inherit_previous_distance(self):
        cum = cumulative_distances([(0, 0), None, (3, 4)])
        assert cum == [0.0, 0.0, 5.0]

    def test_leading_none_and_empty(self):
        assert cumulative_distances([]) == []
        assert cumulative_distances([None, (0, 0), (0, 1)]) == [0.0, 0.0, 1.0]


class TestPathLength:
    def test_order_insensitive(self):
        cum = [0.0, 5.0, 10.0]
        assert path_length(cum, 0, 2) == 10.0
        assert path_length(cum, 2, 0) == 10.0

    def test_clamps_out_of_range_frames(self):
        cum = [0.0, 5.0]
        assert path_length(cum, -3, 99) == 5.0
        assert path_length([], 0, 1) == 0.0


class TestFrameAfterDistance:
    def test_exact_and_between_frames(self):
        cum = cumulative_distances(
            flight_positions(_images_along_x(11)))  # 10 m per frame
        assert frame_after_distance(cum, 0, 30.0) == 3
        # 35 m falls between frame 3 and 4 -> first frame at >= 35 m is 4
        assert frame_after_distance(cum, 0, 35.0) == 4
        assert frame_after_distance(cum, 5, 20.0) == 7

    def test_zero_distance_returns_start(self):
        cum = [0.0, 10.0]
        assert frame_after_distance(cum, 0, 0.0) == 0

    def test_path_too_short_returns_none(self):
        cum = [0.0, 10.0, 20.0]
        assert frame_after_distance(cum, 1, 15.0) is None
        assert frame_after_distance([], 0, 1.0) is None
        assert frame_after_distance(cum, 0, -1.0) is None


class TestTransect:
    def test_display_name_defaults_to_counting_id(self):
        assert Transect(3).display_name == "Transect 3"
        assert Transect(3, name="Meadow").display_name == "Meadow"
        assert Transect(3, name="   ").display_name == "Transect 3"

    def test_first_last_normalize_reversed_range(self):
        t = Transect(1, start_frame=50, end_frame=10)
        assert t.first_frame == 10
        assert t.last_frame == 50

    def test_clamp(self):
        t = Transect(1, start_frame=-5, end_frame=500)
        t.clamp(100)
        assert (t.start_frame, t.end_frame) == (0, 99)

    def test_dict_roundtrip(self):
        t = Transect(2, "A", 5, 42)
        assert Transect.from_dict(t.to_dict()) == t


class TestTransectStore:
    def test_add_counts_ids_up(self, tmp_path):
        store = TransectStore(str(tmp_path), "t")
        assert store.add(0, 10).transect_id == 1
        assert store.add(20, 30).transect_id == 2
        store.remove(1)
        assert store.add(40, 50).transect_id == 3

    def test_ordered_by_first_frame(self, tmp_path):
        store = TransectStore(str(tmp_path), "t")
        store.add(50, 60)
        store.add(30, 0)   # reversed range, first_frame = 0
        assert [t.transect_id for t in store.ordered()] == [2, 1]

    def test_load_missing_file_returns_false(self, tmp_path):
        store = TransectStore(str(tmp_path), "t")
        assert store.load() is False
        assert store.transects == {}

    def test_save_load_roundtrip(self, tmp_path):
        store = TransectStore(str(tmp_path), "w")
        store.add(0, 5, name="First")
        store.add(7, 9)
        store.save(_images_along_x(10))

        fresh = TransectStore(str(tmp_path), "w")
        assert fresh.load() is True
        assert len(fresh.transects) == 2
        assert fresh.transects[1].name == "First"
        assert fresh.transects[1].start_frame == 0
        assert fresh.transects[1].end_frame == 5
        assert fresh.transects[2].name == ""

    def test_save_writes_metadata(self, tmp_path):
        store = TransectStore(str(tmp_path), "t")
        store.add(0, 5)
        store.save(_images_along_x(10))

        with open(store.json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        entry = data["transects"][0]
        assert data["modality"] == "t"
        assert entry["length_m"] == 50.0
        assert entry["start_time"] == "2026-07-14T10:00:00+00:00"
        assert entry["end_time"] == "2026-07-14T10:00:05+00:00"

    def test_save_without_images_leaves_metadata_empty(self, tmp_path):
        store = TransectStore(str(tmp_path), "t")
        store.add(0, 5)
        store.save()
        with open(store.json_path, encoding="utf-8") as fh:
            entry = json.load(fh)["transects"][0]
        assert entry["length_m"] is None
        assert entry["start_time"] == ""

    def test_csv_export(self, tmp_path):
        store = TransectStore(str(tmp_path), "t")
        store.add(0, 5, name="A")
        store.save(_images_along_x(10))

        with open(store.csv_path, encoding="utf-8", newline="") as fh:
            rows = list(csv.reader(fh))
        assert rows[0] == ["id", "name", "start_frame", "end_frame",
                           "start_time", "end_time", "length_m"]
        assert rows[1] == ["1", "A", "0", "5",
                           "2026-07-14T10:00:00+00:00",
                           "2026-07-14T10:00:05+00:00", "50.0"]

    def test_paths_are_modality_suffixed(self, tmp_path):
        store = TransectStore(str(tmp_path), "w")
        assert store.json_path.endswith(
            os.path.join("transects_w", "transects.json"))
        assert store.csv_path.endswith(
            os.path.join("transects_w", "transects.csv"))
