# -*- coding: utf-8 -*-
"""Reaction to the drone (core/drone_reaction.py): what one animal did while
it was in view, relative to the line the drone flew, and what the animals of
a survey did taken together."""
import math

import numpy as np
import pytest

from bambi_wildlife_detection.core import drone_reaction as dr

pytest.importorskip("scipy")


def walk(start=(0.0, 20.0), velocity=(0.0, 0.0), seconds=10, per_second=2,
         jitter=0.0, seed=1, from_second=0.0):
    """Positions of an animal: ``(t, x, y)``."""
    rng = np.random.default_rng(seed)
    points = []
    for step in range(int(seconds * per_second) + 1):
        t = step / per_second
        points.append((from_second + t,
                       start[0] + velocity[0] * t + rng.normal(0, jitter),
                       start[1] + velocity[1] * t + rng.normal(0, jitter)))
    return points


def drone_over(points, speed=5.0, y=0.0, start_x=-25.0):
    """The drone flying due east along *y*, at the moments of *points*."""
    first = points[0][0]
    return [(t, start_x + speed * (t - first), y) for t, _x, _y in points]


# ---------------------------------------------------------------------------
# A constant velocity through the positions
# ---------------------------------------------------------------------------

class TestFitMotion:
    def test_a_walking_animal(self):
        motion = dr.fit_motion(walk(velocity=(1.0, 0.5)))

        assert motion["velocity"] == pytest.approx((1.0, 0.5))
        assert motion["speed_ms"] == pytest.approx(math.hypot(1.0, 0.5))
        assert (motion["n"], motion["duration_s"]) == (21, 10.0)
        assert motion["displacement_m"] == pytest.approx(
            10 * math.hypot(1.0, 0.5))
        assert motion["position_noise_m"] == pytest.approx(0.0, abs=1e-9)
        assert motion["start"] == pytest.approx((0.0, 20.0))
        assert motion["end"] == pytest.approx((10.0, 25.0))
        assert motion["first_s"] == 0.0

    def test_a_standing_animal_has_a_path_and_goes_nowhere(self):
        """Positions jitter: added up from frame to frame they are a long
        way; the motion fitted through them is none."""
        points = walk(seconds=20, per_second=10, jitter=0.5)
        path = sum(math.hypot(b[1] - a[1], b[2] - a[2])
                   for a, b in zip(points, points[1:]))

        motion = dr.fit_motion(points)

        assert path > 100
        assert motion["displacement_m"] < 0.5
        assert motion["displacement_m"] < 3 * motion["displacement_se_m"]
        assert motion["position_noise_m"] == pytest.approx(0.5, rel=0.15)

    def test_how_sure_the_speed_is(self):
        points = walk(velocity=(0.6, 0.0), seconds=20, per_second=10,
                      jitter=0.5)
        spread = sum((t - 10.0) ** 2 for t, _x, _y in points)

        motion = dr.fit_motion(points)

        assert motion["speed_se_ms"] == pytest.approx(
            motion["position_noise_m"] / math.sqrt(spread))
        assert motion["speed_ms"] == pytest.approx(
            0.6, abs=4 * motion["speed_se_ms"])
        assert motion["displacement_se_m"] == pytest.approx(
            motion["speed_se_ms"] * 20.0)

    def test_the_order_of_the_positions_does_not_matter(self):
        points = walk(velocity=(1.0, 0.0), jitter=0.2)

        assert dr.fit_motion(points[::-1])["velocity"] == pytest.approx(
            dr.fit_motion(points)["velocity"])

    @pytest.mark.parametrize("points", [
        [], [(0, 0, 0), (1, 1, 1)], [(5, 0, 0), (5, 1, 1), (5, 2, 2)]])
    def test_too_little_to_fit(self, points):
        assert dr.fit_motion(points) is None


# ---------------------------------------------------------------------------
# One animal, and the line
# ---------------------------------------------------------------------------

class TestTrack:
    def test_an_animal_that_walked_off_moved(self):
        found = dr.describe_track(walk(velocity=(0.0, 0.8), jitter=0.3))

        assert found["status"] == dr.MOVED
        assert found["displacement_m"] == pytest.approx(8.0, abs=1.0)
        assert found["speed_ms"] == pytest.approx(0.8, abs=0.1)
        assert found["heading_deg"] == pytest.approx(0.0, abs=8.0) or \
            found["heading_deg"] > 352.0
        assert (found["n_positions"], found["duration_s"]) == (21, 10.0)
        assert found["start_y"] == pytest.approx(20.0, abs=0.5)
        assert found["end_y"] == pytest.approx(28.0, abs=0.5)

    def test_an_animal_that_stood_stayed(self):
        found = dr.describe_track(walk(jitter=0.5, per_second=10))

        assert found["status"] == dr.STAYED
        assert found["heading_deg"] is None and found["direction"] is None

    def test_a_couple_of_metres_is_not_moving(self):
        found = dr.describe_track(walk(velocity=(0.2, 0.0)))

        assert found["displacement_m"] == pytest.approx(2.0)
        assert found["status"] == dr.STAYED
        assert dr.describe_track(walk(velocity=(0.2, 0.0)),
                                 min_displacement=1.5)["status"] == dr.MOVED

    def test_far_enough_but_too_slow_is_not_moving(self):
        """Four metres - in twenty seconds: an animal that drifts."""
        slow = walk(velocity=(0.4, 0.0))
        found = dr.describe_track(slow, min_speed=0.5)

        assert found["displacement_m"] == pytest.approx(4.0)
        assert found["speed_ms"] == pytest.approx(0.4)
        assert found["status"] == dr.STAYED and found["heading_deg"] is None
        # at the threshold, and without one, it moved
        assert dr.describe_track(slow, min_speed=0.4)["status"] == dr.MOVED
        assert dr.describe_track(slow)["status"] == dr.MOVED
        assert dr.MIN_SPEED_MS == 0.0

    def test_fast_but_not_far_is_not_moving_either(self):
        found = dr.describe_track(walk(velocity=(0.2, 0.0)), min_speed=0.1)

        assert found["status"] == dr.STAYED            # 2 m of the 3

    def test_far_but_not_surely_is_not_moving(self):
        """Five metres in the fit - of positions that lie three metres off
        it: nobody can say this animal went anywhere."""
        found = dr.describe_track(
            walk(velocity=(0.5, 0.0), per_second=0.5, jitter=3.0, seed=4))

        assert found["displacement_m"] < 3 * found["displacement_se_m"]
        assert found["status"] == dr.STAYED

    @pytest.mark.parametrize("points", [
        walk(velocity=(1.0, 0.0), seconds=1.5),          # too short a time
        walk(velocity=(1.0, 0.0), seconds=10, per_second=0.2),   # 3 places
    ])
    def test_too_short_a_look_is_not_judged(self, points):
        found = dr.describe_track(points)

        assert found["status"] == dr.NOT_JUDGED
        assert found["speed_ms"] == pytest.approx(1.0)   # for what it is worth

    def test_nothing_to_fit(self):
        found = dr.describe_track([(0.0, 1.0, 1.0)])

        assert found["status"] == dr.NOT_JUDGED
        assert (found["n_positions"], found["speed_ms"]) == (1, None)

    @pytest.mark.parametrize("velocity, direction, away, along", [
        ((0.0, 0.8), dr.AWAY, 8.0, 0.0),
        ((0.0, -0.8), dr.TOWARDS, -8.0, 0.0),
        ((0.8, 0.0), dr.AHEAD, 0.0, 8.0),
        ((-0.8, 0.0), dr.BACK, 0.0, -8.0),
        ((0.5, 0.7), dr.AWAY, 7.0, 5.0),
        ((0.7, -0.5), dr.AHEAD, -5.0, 7.0),
    ])
    def test_where_it_went_relative_to_the_line(self, velocity, direction,
                                                away, along):
        points = walk(start=(0.0, 20.0), velocity=velocity)

        found = dr.describe_track(points, drone_over(points))

        assert found["direction"] == direction
        assert found["away_m"] == pytest.approx(away, abs=1e-6)
        assert found["along_m"] == pytest.approx(along, abs=1e-6)
        assert found["away_ms"] == pytest.approx(away / 10.0, abs=1e-6)
        assert found["line_distance_m"] == pytest.approx(20.0)

    def test_away_is_away_on_either_side(self):
        north = walk(start=(0.0, 20.0), velocity=(0.0, 0.8))
        south = walk(start=(0.0, -20.0), velocity=(0.0, -0.8))

        for points in (north, south):
            found = dr.describe_track(points, drone_over(points))
            assert found["direction"] == dr.AWAY
            assert found["away_m"] == pytest.approx(8.0)

    def test_the_line_is_the_way_the_drone_flew_whichever_that_is(self):
        """The drone flying north-east; the animal walking off at right
        angles to that."""
        points = walk(start=(10.0, -10.0), velocity=(0.5, -0.5))
        first = points[0][0]
        drone = [(t, 3.0 * (t - first), 3.0 * (t - first))
                 for t, _x, _y in points]

        found = dr.describe_track(points, drone)

        assert found["direction"] == dr.AWAY
        assert found["line_distance_m"] == pytest.approx(math.hypot(10, 10))
        assert found["along_m"] == pytest.approx(0.0, abs=1e-6)
        assert found["away_m"] == pytest.approx(10 * math.hypot(0.5, 0.5))

    def test_the_nearest_the_drone_came(self):
        points = walk(start=(0.0, 20.0))
        # the drone passes x = 0 five seconds in, 20 m to the side
        found = dr.describe_track(points, drone_over(points))

        assert found["drone_distance_m"] == pytest.approx(20.0)
        assert found["status"] == dr.STAYED and found["direction"] is None
        assert found["away_ms"] == pytest.approx(0.0, abs=1e-9)

    def test_under_the_line_either_way_across_is_away(self):
        points = walk(start=(0.0, 0.5), velocity=(0.0, -0.8))

        found = dr.describe_track(points, drone_over(points))

        assert found["line_distance_m"] == pytest.approx(0.5)
        assert found["direction"] == dr.AWAY
        # ... but its speed away is left open: it has no side
        assert found["away_m"] is None and found["away_ms"] is None
        assert found["along_m"] == pytest.approx(0.0, abs=1e-6)

    def test_under_the_line_and_along_it(self):
        points = walk(start=(0.0, 0.5), velocity=(1.0, 0.0))

        assert dr.describe_track(
            points, drone_over(points))["direction"] == dr.AHEAD

    def test_a_hovering_drone_has_no_line_only_itself(self):
        points = walk(start=(0.0, 20.0), velocity=(0.0, 0.8))
        drone = [(t, 0.0, 0.0) for t, _x, _y in points]

        found = dr.describe_track(points, drone)

        assert found["along_m"] is None
        assert found["away_m"] == pytest.approx(8.0)
        assert found["direction"] == dr.AWAY
        assert found["line_distance_m"] == pytest.approx(20.0)

    def test_without_the_drone_only_the_movement_is_known(self):
        found = dr.describe_track(walk(velocity=(0.0, 0.8)))

        assert found["status"] == dr.MOVED
        assert found["direction"] is None and found["away_m"] is None
        assert found["line_distance_m"] is None
        assert found["heading_deg"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The animals together
# ---------------------------------------------------------------------------

def row(status=dr.STAYED, direction=None, species="roe deer", away_ms=0.0,
        line=20.0, speed=0.05, displacement=0.5, **more):
    return dict({"flight": "Meadow", "track_id": 1, "species": species,
                 "status": status, "direction": direction,
                 "away_ms": away_ms, "line_distance_m": line,
                 "speed_ms": speed, "displacement_m": displacement}, **more)


def herd(moved_away=0, moved_towards=0, stayed=0, ahead=0, back=0,
         not_judged=0, species="roe deer", seed=1):
    """Animals as the summary reads them; those that stayed drift by what
    the jitter makes of them."""
    rng = np.random.default_rng(seed)
    rows = []
    for count, direction, sign in ((moved_away, dr.AWAY, 1.0),
                                   (moved_towards, dr.TOWARDS, -1.0),
                                   (ahead, dr.AHEAD, 0.0),
                                   (back, dr.BACK, 0.0)):
        for _ in range(count):
            speed = 0.8 + rng.uniform(0, 0.4)
            rows.append(row(dr.MOVED, direction, species,
                            away_ms=sign * speed + rng.normal(0, 0.02),
                            speed=speed, displacement=speed * 10,
                            line=float(rng.uniform(5, 25))))
    for _ in range(stayed):
        rows.append(row(species=species, away_ms=float(rng.normal(0, 0.03)),
                        speed=float(abs(rng.normal(0, 0.04))),
                        line=float(rng.uniform(20, 60))))
    for _ in range(not_judged):
        rows.append(row(dr.NOT_JUDGED, species=species, away_ms=None,
                        speed=None, displacement=None))
    return rows


class TestSummary:
    def test_how_many_moved(self):
        summary = dr.summarise(herd(moved_away=3, moved_towards=2,
                                    stayed=15, not_judged=4))[dr.ALL]

        assert (summary["tracks"], summary["judged"],
                summary["not_judged"], summary["moved"]) == (24, 20, 4, 5)
        assert summary["moved_share"] == 0.25
        assert summary["moved_share_low"] < 0.25 < summary["moved_share_high"]
        assert summary["directions"] == {dr.AWAY: 3, dr.TOWARDS: 2,
                                         dr.AHEAD: 0, dr.BACK: 0}
        assert 8.0 <= summary["median_displacement_m"] <= 12.0
        assert 0.8 <= summary["median_speed_ms"] <= 1.2

    def test_animals_that_fled(self):
        summary = dr.summarise(herd(moved_away=9, moved_towards=1,
                                    stayed=10))[dr.ALL]

        # nine of ten to one side
        assert summary["away_p_value"] == pytest.approx(22 / 1024)
        assert summary["mean_away_ms"] > 0.3
        assert summary["mean_away_ms_low"] > 0
        assert summary["away_animals"] == 20
        assert summary["verdict"] == dr.FLED
        assert summary["sentence"].startswith(
            "10 of 20 animals moved while in view; 9 went away from the "
            "line and 1 towards it, and on average the animals moved ")
        assert summary["sentence"].endswith(
            "The animals moved away from the line the drone flew: a "
            "reaction to it.")

    def test_animals_on_the_move_but_not_from_the_drone(self):
        summary = dr.summarise(herd(moved_away=6, moved_towards=6,
                                    stayed=4))[dr.ALL]

        assert summary["away_p_value"] == pytest.approx(1.0)
        assert summary["mean_away_ms_low"] < 0 < summary["mean_away_ms_high"]
        assert summary["verdict"] == dr.MOVED_ABOUT
        assert "not from the drone" in summary["sentence"]

    def test_animals_that_took_no_notice(self):
        summary = dr.summarise(herd(moved_away=1, moved_towards=1,
                                    stayed=28))[dr.ALL]

        assert summary["verdict"] == dr.QUIET
        assert summary["sentence"] == (
            "2 of 30 animals moved while in view, and those that did went "
            "no particular way: no sign of a reaction to the drone.")

    def test_a_handful_of_animals_says_nothing(self):
        summary = dr.summarise(herd(moved_away=3, not_judged=6))[dr.ALL]

        assert summary["verdict"] == dr.UNKNOWN
        assert summary["sentence"].startswith(
            "3 animal(s) were in view long enough to tell")

    def test_the_nearer_the_line_the_faster(self):
        rows = [row(speed=2.0 - 0.03 * line, line=float(line))
                for line in range(5, 65, 5)]

        nearer = dr.summarise(rows)[dr.ALL]["nearer_faster"]

        assert nearer["correlation"] == pytest.approx(-1.0)
        assert nearer["p_value"] < 0.001 and nearer["animals"] == 12

    def test_too_few_or_all_alike_say_nothing_of_that(self):
        assert dr.summarise(
            [row(line=float(d)) for d in (5, 10, 15)])[
            dr.ALL]["nearer_faster"] is None
        assert dr.summarise(
            [row(line=10.0, speed=0.1 * n) for n in range(8)])[
            dr.ALL]["nearer_faster"] is None

    def test_a_terrain_model_that_is_off_is_not_taken_for_fleeing(self):
        summary = dr.summarise(herd(back=9, ahead=1, moved_away=1,
                                    stayed=9))[dr.ALL]

        assert summary["verdict"] != dr.FLED
        assert ("Note: 10 of the 11 that moved went along the line, 9 of "
                "them the same way. That is what a terrain model a few "
                "metres off looks like") in summary["sentence"]

    def test_per_species_when_there_are_several(self):
        rows = herd(moved_away=9, moved_towards=1, stayed=5) + herd(
            stayed=12, species="red deer", seed=2)

        summary = dr.summarise(rows, species_order=["red deer", "roe deer"])

        assert list(summary) == [dr.ALL, "red deer", "roe deer"]
        assert summary["roe deer"]["verdict"] == dr.FLED
        assert summary["red deer"]["verdict"] == dr.QUIET
        assert summary[dr.ALL]["tracks"] == 27

    def test_one_species_is_all_animals(self):
        assert list(dr.summarise(herd(stayed=6))) == [dr.ALL]

    def test_no_animals(self):
        summary = dr.summarise([])[dr.ALL]

        assert (summary["tracks"], summary["moved_share"]) == (0, None)
        assert summary["verdict"] == dr.UNKNOWN


# ---------------------------------------------------------------------------
# A flight
# ---------------------------------------------------------------------------

class TestFlight:
    def test_one_row_per_animal(self):
        seconds = {frame: frame * 0.5 for frame in range(40)}
        positions = {
            1: [(f, 0.0, 20.0 + 0.4 * f) for f in range(21)],      # walks off
            2: [(f, 50.0, -30.0) for f in range(10, 31)],          # stands
            3: [(f, 80.0, 5.0) for f in (3, 4)],                   # a glimpse
        }
        drone = {frame: (-25.0 + 2.5 * frame, 0.0) for frame in range(40)}
        animals = [
            {"track_id": 1, "species": "roe deer", "sex": "female",
             "age": None},
            {"track_id": 2, "species": "red deer", "sex": None, "age": None},
            {"track_id": 3, "species": "roe deer", "sex": None, "age": None},
            {"track_id": 4, "species": "roe deer", "sex": None, "age": None},
        ]

        rows = dr.flight_rows("Meadow", animals, positions, seconds, drone)

        assert [list(r)[:5] for r in rows] == [dr.TRACK_COLUMNS[:5]] * 4
        assert [(r["track_id"], r["status"], r["direction"])
                for r in rows] == [
            (1, dr.MOVED, dr.AWAY), (2, dr.STAYED, None),
            (3, dr.NOT_JUDGED, None), (4, dr.NOT_JUDGED, None)]
        walker = rows[0]
        assert (walker["flight"], walker["species"], walker["sex"]) == (
            "Meadow", "roe deer", "female")
        assert walker["speed_ms"] == pytest.approx(0.8)
        assert walker["away_m"] == pytest.approx(8.0)
        assert rows[1]["line_distance_m"] == pytest.approx(30.0)
        assert rows[3]["n_positions"] == 0

    def test_frames_without_a_time_are_left_out(self):
        seconds = {frame: float(frame) for frame in range(0, 20, 2)}
        positions = {1: [(f, 0.0, 20.0 + f) for f in range(20)]}

        rows = dr.flight_rows("Meadow", [{"track_id": 1, "species": "roe"}],
                              positions, seconds)

        assert rows[0]["n_positions"] == 10
        assert rows[0]["speed_ms"] == pytest.approx(1.0)
        assert rows[0]["away_m"] is None          # no drone, no line


# ---------------------------------------------------------------------------
# Written down
# ---------------------------------------------------------------------------

class TestWritten:
    def test_the_table(self, tmp_path):
        import csv

        points = walk(velocity=(0.0, 0.8))
        found = dr.describe_track(points, drone_over(points))
        row_ = dict(found, flight="Meadow", track_id=7, species="roe deer",
                    sex=None, age="adult")

        path = dr.write_table([row_], str(tmp_path / "tracks.csv"))

        with open(path, encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert list(rows[0]) == dr.TRACK_COLUMNS
        assert (rows[0]["track_id"], rows[0]["status"], rows[0]["sex"]) == (
            "7", "moved", "")
        assert rows[0]["direction"] == "away from the line"
        assert float(rows[0]["away_m"]) == pytest.approx(8.0)

    def test_the_document_can_be_read_again(self, tmp_path):
        document = {"summary": dr.summarise(herd(moved_away=9, stayed=10)),
                    "tracks": []}

        again = dr.read_json(dr.write_json(document,
                                           str(tmp_path / "reaction.json")))

        assert again["summary"][dr.ALL]["verdict"] == dr.FLED
        assert list(again["summary"][dr.ALL]["directions"]) == [
            dr.AWAY, dr.TOWARDS, dr.AHEAD, dr.BACK]

    def test_in_a_few_lines(self):
        rows = herd(moved_away=9, moved_towards=1, stayed=10)
        for index, entry in enumerate(rows):       # the near ones the fast
            entry["line_distance_m"] = 60.0 - 50.0 * (
                entry["speed_ms"] > 0.5) + 0.01 * index
        lines = dr.describe({"summary": dr.summarise(rows)})

        assert lines[0].startswith(
            "All species: 10 of 20 animals moved while in view; ")
        assert lines[1].startswith("  those that moved went ")
        assert lines[1].endswith("m/s (medians)")
        assert lines[2].startswith(
            "  the nearer the line, the faster (rank correlation -")


# ---------------------------------------------------------------------------
# What "moved" is, species by species
# ---------------------------------------------------------------------------

class TestSpeciesThresholds:
    def test_a_species_has_its_own_or_everyones(self):
        own = dr.species_thresholds({
            "Roe Deer": {"min_displacement_m": 5, "min_speed_ms": "0.3"},
            "hare": {"min_displacement_m": "", "min_speed_ms": 1.5},
            "fox": {"min_displacement_m": None, "min_speed_ms": None}})

        assert own == {"roe deer": {"min_displacement_m": 5.0,
                                    "min_speed_ms": 0.3},
                       "hare": {"min_speed_ms": 1.5}}
        assert dr.thresholds_of("roe deer", own, 3.0, 0.0) == (5.0, 0.3)
        assert dr.thresholds_of("Hare", own, 3.0, 0.0) == (3.0, 1.5)
        assert dr.thresholds_of("fox", own, 3.0, 0.1) == (3.0, 0.1)
        assert dr.thresholds_of(None, own, 3.0, 0.1) == (3.0, 0.1)

    @pytest.mark.parametrize("raw", [None, {}, "", {"": {"min_speed_ms": 1}},
                                     {"fox": "fast"}])
    def test_nothing_given_is_nothing(self, raw):
        assert dr.species_thresholds(raw) == {}

    @pytest.mark.parametrize("values, said", [
        ({"min_displacement_m": 0}, "distance for 'fox' has to be more "
                                    "than nothing"),
        ({"min_displacement_m": -1}, "distance for 'fox'"),
        ({"min_speed_ms": -0.1}, "speed for 'fox' has to be nought or more"),
        ({"min_speed_ms": "fast"}, "speed for 'fox' is not a number"),
        ({"min_displacement_m": float("inf")}, "distance for 'fox'"),
    ])
    def test_a_value_that_is_none_is_refused(self, values, said):
        with pytest.raises(dr.ReactionError, match=said):
            dr.species_thresholds({"fox": values})
        with pytest.raises(dr.ReactionError, match="not a table"):
            dr.species_thresholds(["fox"])

    def test_a_zero_speed_is_a_speed(self):
        assert dr.species_thresholds({"fox": {"min_speed_ms": 0}}) == {
            "fox": {"min_speed_ms": 0.0}}

    def test_said_in_a_line(self):
        settings = {"min_displacement_m": 3.0, "min_speed_ms": 0.0}
        assert dr.describe_thresholds(settings) == "at least 3 m"
        settings["min_speed_ms"] = 0.2
        assert dr.describe_thresholds(settings) == (
            "at least 3 m at 0.2 m/s or faster")
        settings["per_species"] = {"roe deer": {"min_displacement_m": 5.0},
                                   "hare": {"min_speed_ms": 0.0}}
        assert dr.describe_thresholds(settings) == (
            "at least 3 m at 0.2 m/s or faster; hare: at least 3 m, "
            "roe deer: at least 5 m at 0.2 m/s or faster")
        assert dr.describe_thresholds({}) == ""
        assert dr.describe_thresholds(None) == ""
