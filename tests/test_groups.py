# -*- coding: utf-8 -*-
"""Groups: animals of one species that were together (core/groups.py)."""
import math
import sqlite3

import pytest

from bambi_wildlife_detection.core import groups, store


def animal(ident, x, y, species="red deer", first=0, last=None, t=None,
           until=None):
    last = first if last is None else last
    return {"id": ident, "species": species, "x": x, "y": y,
            "first_frame": first, "last_frame": last,
            "first_time": t, "last_time": t if until is None else until}


DISTANCES = {"red deer": 50.0, "roe deer": 20.0, "fox": 0.0}


class TestPartition:
    def test_animals_within_the_distance_are_one_group(self):
        herd = [animal(3, 0, 0), animal(5, 30, 0), animal(9, 0, 40)]

        assert groups.partition(herd, DISTANCES) == [3, 3, 3]

    def test_animals_further_apart_are_not(self):
        two = [animal(1, 0, 0), animal(2, 50.1, 0)]

        assert groups.partition(two, DISTANCES) == [1, 2]

    def test_exactly_the_distance_still_belongs(self):
        two = [animal(1, 0, 0), animal(2, 50.0, 0)]

        assert groups.partition(two, DISTANCES) == [1, 1]

    def test_a_chain_is_one_group_although_its_ends_are_far_apart(self):
        chain = [animal(i, 45.0 * i, 0) for i in range(1, 8)]

        assert groups.partition(chain, DISTANCES) == [1] * 7
        assert chain[-1]["x"] - chain[0]["x"] > 250

    def test_each_species_has_its_own_distance(self):
        mixed = [animal(1, 0, 0, "red deer"), animal(2, 30, 0, "red deer"),
                 animal(3, 0, 0, "roe deer"), animal(4, 30, 0, "roe deer")]

        assert groups.partition(mixed, DISTANCES) == [1, 1, 3, 4]

    def test_species_are_never_mixed(self):
        side_by_side = [animal(1, 0, 0, "red deer"),
                        animal(2, 1, 0, "roe deer")]

        assert groups.partition(side_by_side, DISTANCES) == [1, 2]

    def test_a_distance_of_zero_means_never_grouped(self):
        foxes = [animal(1, 0, 0, "fox"), animal(2, 0, 0, "fox")]

        assert groups.partition(foxes, DISTANCES) == [1, 2]

    def test_a_species_without_a_distance_has_the_default(self):
        boar = [animal(1, 0, 0, "wild boar"), animal(2, 49, 0, "wild boar"),
                animal(3, 200, 0, "wild boar")]

        assert groups.DEFAULT_DISTANCE_M == 50.0
        assert groups.partition(boar, {}) == [1, 1, 3]

    def test_a_group_is_named_after_its_lowest_id(self):
        herd = [animal(42, 0, 0), animal(7, 10, 0), animal(19, 20, 0)]

        assert groups.partition(herd, DISTANCES) == [7, 7, 7]

    def test_an_animal_without_a_place_is_alone(self):
        some = [animal(1, None, None), animal(2, None, None),
                animal(3, 0, 0)]

        assert groups.partition(some, DISTANCES) == [1, 2, 3]

    def test_no_animals_no_groups(self):
        assert groups.partition([], DISTANCES) == []

    def test_many_animals_are_divided_as_pair_by_pair_would(self):
        """The grid of cells finds what comparing every pair finds."""
        import random

        rng = random.Random(7)
        many = [animal(i, rng.uniform(-400, 400), rng.uniform(-400, 400))
                for i in range(1, 301)]
        found = groups.partition(many, {"red deer": 35.0})

        parent = list(range(len(many)))

        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i

        for i, a in enumerate(many):
            for j in range(i + 1, len(many)):
                b = many[j]
                if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) <= 35.0:
                    parent[find(j)] = find(i)
        expected = {}
        for i, a in enumerate(many):
            expected.setdefault(find(i), []).append(a["id"])
        by_group = {}
        for a, group in zip(many, found):
            by_group.setdefault(group, []).append(a["id"])

        assert sorted(map(sorted, by_group.values())) == \
            sorted(map(sorted, expected.values()))
        assert all(group == min(ids) for group, ids in by_group.items())
        assert 1 < len(by_group) < len(many)

    def test_negative_coordinates_are_no_different(self):
        two = [animal(1, -0.5, -0.5), animal(2, 0.5, 0.5)]

        assert groups.partition(two, DISTANCES) == [1, 1]


class TestSeenTogether:
    def test_two_sightings_minutes_apart_are_not_a_group(self):
        """The same ground crossed on two legs: the animal counted on both
        is not a group of two."""
        twice = [animal(1, 0, 0, t=100.0, until=110.0),
                 animal(2, 3, 0, t=400.0, until=410.0)]

        assert groups.partition(twice, DISTANCES) == [1, 2]

    def test_seen_within_a_minute_is_seen_together(self):
        pair = [animal(1, 0, 0, t=100.0, until=110.0),
                animal(2, 30, 0, t=170.0, until=180.0)]

        assert groups.SAME_PASS_SECONDS == 60.0
        assert groups.partition(pair, DISTANCES) == [1, 1]

    def test_overlapping_in_time_is_together(self):
        pair = [animal(1, 0, 0, t=100.0, until=200.0),
                animal(2, 30, 0, t=150.0, until=160.0)]

        assert groups.partition(pair, DISTANCES) == [1, 1]

    def test_a_herd_passed_slowly_is_joined_member_by_member(self):
        herd = [animal(i, 40.0 * i, 0, t=100.0 + 50 * i, until=110.0 + 50 * i)
                for i in range(1, 6)]

        assert groups.partition(herd, DISTANCES) == [1] * 5
        assert herd[-1]["first_time"] - herd[0]["last_time"] > 60

    def test_without_times_distance_alone_decides(self):
        pair = [animal(1, 0, 0, first=10), animal(2, 3, 0, first=9000)]

        assert groups.partition(pair, DISTANCES) == [1, 1]

    def test_one_unknown_time_does_not_separate(self):
        pair = [animal(1, 0, 0, t=100.0), animal(2, 3, 0)]

        assert groups.partition(pair, DISTANCES) == [1, 1]

    def test_the_time_rule_can_be_left_out(self):
        twice = [animal(1, 0, 0, t=100.0), animal(2, 3, 0, t=4000.0)]

        assert groups.partition(twice, DISTANCES, max_gap_s=None) == [1, 1]


class TestBoxes:
    """A survey that counts boxes: only the boxes of one frame can be a
    group - every frame shows the same animal again."""

    def test_boxes_of_one_frame_are_grouped(self):
        boxes = [animal(1, 0, 0, first=5), animal(2, 20, 0, first=5),
                 animal(3, 300, 0, first=5)]

        assert groups.partition(boxes, DISTANCES, same_frame=True) == [1, 1, 3]

    def test_the_same_animal_in_the_next_frame_is_not_its_own_company(self):
        boxes = [animal(1, 0, 0, first=5), animal(2, 0.2, 0, first=6),
                 animal(3, 0.4, 0, first=7)]

        assert groups.partition(boxes, DISTANCES, same_frame=True) == [1, 2, 3]


class TestDescribe:
    def test_one_row_per_group(self):
        herd = [animal(3, 0, 0, first=10, last=20, t=1.0, until=2.0),
                animal(5, 30, 0, first=15, last=40, t=1.5, until=4.0),
                animal(9, 900, 0, first=80, last=90, t=8.0, until=9.0)]
        rows = groups.describe(herd, groups.partition(herd, DISTANCES))

        assert rows == [
            {"group": 3, "species": "red deer", "size": 2, "x": 15.0,
             "y": 0.0, "radius_m": 15.0, "first_frame": 10, "last_frame": 40,
             "first_time": 1.0, "last_time": 4.0, "members": [3, 5]},
            {"group": 9, "species": "red deer", "size": 1, "x": 900.0,
             "y": 0.0, "radius_m": 0.0, "first_frame": 80, "last_frame": 90,
             "first_time": 8.0, "last_time": 9.0, "members": [9]},
        ]

    def test_an_animal_without_a_place_has_none_as_a_group_either(self):
        rows = groups.describe([animal(4, None, None)], [4])

        assert rows[0]["x"] is None and rows[0]["radius_m"] is None
        assert rows[0]["size"] == 1


class TestSizeStats:
    def test_mean_and_its_standard_error(self):
        stats = groups.size_stats([1, 1, 4, 6])

        assert stats["groups"] == 4 and stats["animals"] == 12
        assert stats["mean_size"] == 3.0
        assert stats["largest"] == 6 and stats["alone"] == 2
        # variance 6, over n = 4
        assert stats["se_size"] == pytest.approx(math.sqrt(6.0 / 4))

    def test_one_group_has_no_standard_error(self):
        stats = groups.size_stats([5])

        assert stats["mean_size"] == 5.0 and stats["se_size"] is None

    def test_nothing(self):
        assert groups.size_stats([]) == {
            "groups": 0, "animals": 0, "mean_size": None, "se_size": None,
            "largest": 0, "alone": 0}

    def test_animals_all_alone(self):
        stats = groups.size_stats([1, 1, 1])

        assert stats["mean_size"] == 1.0 and stats["se_size"] == 0.0


class TestStoredDistances:
    @pytest.fixture
    def conn(self, tmp_path):
        made = store.open_store(str(tmp_path / "project.gpkg"), store.PROJECT)
        yield made
        made.close()

    def test_nothing_is_stored_at_first(self, conn):
        assert groups.stored_distances(conn) == {}

    def test_a_distance_is_kept_by_species_id(self, conn):
        groups.store_distance(conn, 4, 80)
        groups.store_distance(conn, 5, "12.5")

        assert groups.stored_distances(conn) == {4: 80.0, 5: 12.5}

    def test_none_goes_back_to_the_default(self, conn):
        groups.store_distance(conn, 4, 80)
        groups.store_distance(conn, 4, None)

        assert groups.stored_distances(conn) == {}

    def test_zero_is_a_distance(self, conn):
        groups.store_distance(conn, 4, 0)

        assert groups.stored_distances(conn) == {4: 0.0}

    @pytest.mark.parametrize("value", [-1, "far", float("nan"),
                                       float("inf"), 5001, None])
    def test_what_is_no_distance_is_refused(self, conn, value):
        with pytest.raises(ValueError):
            groups.normalise_distance(value)

    def test_a_damaged_entry_is_no_entry(self, conn):
        store.set_meta(conn, groups.META_KEY, "{not json")
        assert groups.stored_distances(conn) == {}
        store.set_meta(conn, groups.META_KEY, '{"4": "x", "5": 30, "y": 2}')
        assert groups.stored_distances(conn) == {5: 30.0}

    def test_names_get_the_default_unless_set(self):
        rows = [{"species_id": 4, "name": "red deer"},
                {"species_id": 5, "name": "roe deer"}]

        assert groups.distances_for(rows, {5: 20.0}) == {
            "red deer": 50.0, "roe deer": 20.0}
        assert groups.distance_of({"roe deer": 20.0}, "roe deer") == 20.0
        assert groups.distance_of({"roe deer": 20.0}, "lynx") == 50.0
        assert groups.distance_of({"fox": 0.0}, "fox") == 0.0
        assert groups.distance_of({}, None) == 50.0

    def test_a_flight_without_a_store_has_no_distances(self, tmp_path):
        assert groups.species_distances(str(tmp_path)) == {}
        assert groups.species_distances("") == {}

    def test_the_distances_of_a_flight(self, tmp_path):
        folder = str(tmp_path)
        conn = store.open_store(store.project_path(folder), store.PROJECT)
        try:
            row = conn.execute(
                "SELECT species_id FROM species WHERE name = 'roe deer'"
            ).fetchone()
            with store.transaction(conn):
                groups.store_distance(conn, row["species_id"], 15)
        finally:
            conn.close()

        distances = groups.species_distances(folder)
        assert distances["roe deer"] == 15.0
        assert distances["red deer"] == 50.0


def test_a_distance_differs_only_when_a_species_has_another_one():
    before = {"roe deer": 50.0, "fox": 0.0}

    assert not groups.differ(before, dict(before))
    assert not groups.differ(before, dict(before, wolf=50.0))    # new, default
    assert not groups.differ(dict(before, wolf=50.0), before)    # gone, default
    assert groups.differ(before, dict(before, wolf=300.0))
    assert groups.differ(before, {"roe deer": 20.0, "fox": 0.0})
    assert groups.differ(before, {"roe deer": 50.0})             # fox: 0 -> 50
    assert not groups.differ({}, {})


class TestCopyDistances:
    """A survey takes each flight's own distance; one that is meant for the
    whole project has to be in every flight."""

    @staticmethod
    def _flight(folder, **distances):
        conn = store.open_store(store.project_path(str(folder)), store.PROJECT)
        try:
            with store.transaction(conn):
                for name, metres in distances.items():
                    row = conn.execute(
                        "SELECT species_id FROM species WHERE name = ?",
                        (name.replace("_", " "),)).fetchone()
                    groups.store_distance(conn, row["species_id"], metres)
        finally:
            conn.close()
        return str(folder)

    def test_the_other_flights_get_the_distances_by_name(self, tmp_path):
        a = self._flight(tmp_path / "a", roe_deer=20, wild_boar=15)
        b = self._flight(tmp_path / "b")
        c = self._flight(tmp_path / "c", roe_deer=20, wild_boar=15)

        assert groups.copy_distances(a, [b, c]) == [b]
        assert groups.species_distances(b)["roe deer"] == 20.0
        assert groups.species_distances(b)["wild boar"] == 15.0
        assert groups.species_distances(b)["red deer"] == 50.0

    def test_a_default_in_the_source_is_a_default_in_the_target(self, tmp_path):
        a = self._flight(tmp_path / "a")
        b = self._flight(tmp_path / "b", roe_deer=20)

        assert groups.copy_distances(a, [b]) == [b]
        assert groups.species_distances(b)["roe deer"] == 50.0

    def test_ids_do_not_have_to_agree(self, tmp_path):
        """Species ids are assigned per flight: 'lynx' is one number here
        and another there."""
        a, b = str(tmp_path / "a"), str(tmp_path / "b")
        for folder, first in ((a, "wolf"), (b, "lynx")):
            conn = store.open_store(store.project_path(folder), store.PROJECT)
            with store.transaction(conn):
                for name in (first, "lynx" if first == "wolf" else "wolf"):
                    conn.execute(
                        "INSERT INTO species (species_id, name, protected) "
                        "VALUES ((SELECT MAX(species_id) + 1 FROM species), "
                        "?, 0)", (name,))
            conn.close()
        self._flight(a, lynx=120)

        groups.copy_distances(a, [b])

        assert groups.species_distances(b)["lynx"] == 120.0
        assert groups.species_distances(b)["wolf"] == 50.0

    def test_a_species_only_the_target_has_keeps_its_distance(self, tmp_path):
        a = self._flight(tmp_path / "a", roe_deer=20)
        b = str(tmp_path / "b")
        conn = store.open_store(store.project_path(b), store.PROJECT)
        with store.transaction(conn):
            conn.execute(
                "INSERT INTO species (species_id, name, protected) VALUES "
                "((SELECT MAX(species_id) + 1 FROM species), 'ibex', 0)")
            row = conn.execute(
                "SELECT species_id FROM species WHERE name = 'ibex'").fetchone()
            groups.store_distance(conn, row["species_id"], 70)
        conn.close()

        groups.copy_distances(a, [b])

        assert groups.species_distances(b)["ibex"] == 70.0
        assert groups.species_distances(b)["roe deer"] == 20.0

    def test_folders_without_a_store_are_left_alone(self, tmp_path):
        a = self._flight(tmp_path / "a", roe_deer=20)

        assert groups.copy_distances(a, ["", str(tmp_path / "nowhere")]) == []
        assert groups.copy_distances(str(tmp_path / "nowhere"), [a]) == []


# ---------------------------------------------------------------------------
# The groups of a flight
# ---------------------------------------------------------------------------

#: A herd of three red deer crossed at 05:00:30, the same meadow crossed
#: again five minutes later (two of them found again), two roe deer 30 m
#: apart, a lynx that was never placed.
HERD = [
    {"id": 1, "species": "red deer", "at": (10, 20), "frames": (30, 45)},
    {"id": 2, "species": "red deer", "at": (40, 20), "frames": (32, 50)},
    {"id": 3, "species": "red deer", "at": (40, 60), "frames": (40, 55)},
    {"id": 4, "species": "red deer", "at": (12, 22), "frames": (330, 345)},
    {"id": 5, "species": "red deer", "at": (41, 58), "frames": (335, 350)},
    {"id": 6, "species": "roe deer", "at": (300, 0), "frames": (100, 110)},
    {"id": 7, "species": "roe deer", "at": (330, 0), "frames": (102, 112)},
    {"id": 8, "species": "lynx", "at": None, "frames": (200, 201)},
]


@pytest.fixture
def herd(tmp_path):
    from tests.survey_flights import make_flight
    return make_flight(str(tmp_path / "flight"), HERD)


class TestForFlight:
    def test_animals_seen_together_are_a_group(self, herd):
        found = groups.for_flight(herd, "t")

        assert [(row["group"], row["size"], row["members"])
                for row in found.rows] == [
            (1, 3, [1, 2, 3]), (4, 2, [4, 5]), (6, 2, [6, 7]), (8, 1, [8])]
        assert found.timed and found.source == "tracks"
        assert len(found) == 4

    def test_every_animal_knows_its_group(self, herd):
        found = groups.for_flight(herd, "t")

        assert [found.of(track) for track in range(1, 9)] == [
            1, 1, 1, 4, 4, 6, 6, 8]
        assert found.size_of(1) == 3 and found.size_of(8) == 1
        assert found.of(99) is None and found.size_of(99) == 0

    def test_the_same_ground_crossed_again_is_another_group(self, herd):
        """Tracks 4 and 5 are where 1 and 3 were - five minutes later."""
        found = groups.for_flight(herd, "t")

        assert found.of(4) != found.of(1)
        again = next(row for row in found.rows if row["group"] == 4)
        first = next(row for row in found.rows if row["group"] == 1)
        assert again["first_time"] - first["last_time"] > 250

    def test_the_schemas_distance_decides(self, herd):
        conn = store.open_store(store.project_path(herd), store.PROJECT)
        with store.transaction(conn):
            for name, metres in (("roe deer", 20), ("red deer", 35)):
                row = conn.execute(
                    "SELECT species_id FROM species WHERE name = ?",
                    (name,)).fetchone()
                groups.store_distance(conn, row["species_id"], metres)
        conn.close()

        found = groups.for_flight(herd, "t")

        assert found.of(6) != found.of(7)             # 30 m apart, 20 allowed
        assert found.of(1) == found.of(2)             # 30 m
        assert found.of(3) != found.of(2)             # 40 m apart, 35 allowed
        assert found.distances["roe deer"] == 20.0

    def test_an_animal_that_was_never_placed_is_alone(self, herd):
        found = groups.for_flight(herd, "t")
        lynx = next(row for row in found.rows if row["group"] == 8)

        assert lynx["size"] == 1 and lynx["x"] is None

    def test_where_no_time_is_known_distance_alone_decides(self, tmp_path):
        from tests.survey_flights import make_flight
        folder = make_flight(str(tmp_path / "untimed"), HERD, timed=False)

        found = groups.for_flight(folder, "t")

        assert not found.timed
        assert found.of(4) == found.of(1) and found.size_of(1) == 5
        assert "capture time of a frame is not known" in found.describe_rule()

    def test_the_rule_in_a_sentence(self, herd):
        assert "seen within 60 s of each other" in \
            groups.for_flight(herd, "t").describe_rule()
        assert "in the same frame" in \
            groups.for_flight(herd, "t", "detections").describe_rule()

    def test_a_false_positive_is_in_no_group(self, tmp_path):
        from tests.survey_flights import make_flight
        folder = make_flight(str(tmp_path / "rocks"), [
            {"id": 1, "species": "red deer", "at": (0, 0), "frames": (1, 5)},
            {"id": 2, "species": "not-an-animal", "at": (5, 0),
             "frames": (1, 5)},
        ])

        found = groups.for_flight(folder, "t")

        assert found.of(1) == 1 and found.of(2) is None
        assert len(found) == 1

    def test_a_flight_without_detections_has_no_groups_to_ask_for(
            self, tmp_path):
        from bambi_wildlife_detection.core.analytics_source import (
            AnalyticsError)
        with pytest.raises(AnalyticsError):
            groups.for_flight(str(tmp_path), "t")


class TestBoxesOfAFlight:
    """A survey that counts boxes: a group is what one frame shows
    together."""

    @pytest.fixture
    def photos(self, tmp_path):
        from tests.survey_flights import make_flight
        return make_flight(str(tmp_path / "photos"), [], loose_boxes=[
            {"species": "red deer", "at": (0, 0), "frame": 3},
            {"species": "red deer", "at": (20, 0), "frame": 3},
            {"species": "red deer", "at": (20, 30), "frame": 3},
            {"species": "red deer", "at": (0, 1), "frame": 4},
            {"species": "roe deer", "at": (5, 5), "frame": 3},
            {"species": "red deer", "at": (400, 0), "frame": 3},
        ])

    def test_the_boxes_of_a_frame_are_grouped(self, photos):
        found = groups.for_flight(photos, "t", "detections")

        assert sorted(row["size"] for row in found.rows) == [1, 1, 1, 3]
        assert found.source == "detections"

    def test_a_box_is_known_by_its_detection_id(self, photos):
        from bambi_wildlife_detection.core import track_store

        found = groups.for_flight(photos, "t", "detections")
        stored = sorted(track_store.load_detections(photos, "t"),
                        key=lambda row: row["x1"])
        first_three = [found.of(row["detection_id"]) for row in stored[:3]]

        assert len(set(first_three)) == 1
        assert first_three[0] == min(row["detection_id"]
                                     for row in stored[:3])
        assert found.of(stored[3]["detection_id"]) not in first_three


def test_the_module_needs_no_qgis():
    import bambi_wildlife_detection.core.groups as module
    source = open(module.__file__, encoding="utf-8").read()
    assert "qgis" not in source.replace("QGIS", "")
    assert isinstance(sqlite3.sqlite_version, str)
