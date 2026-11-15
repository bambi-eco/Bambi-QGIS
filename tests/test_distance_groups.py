# -*- coding: utf-8 -*-
"""Distance sampling takes a group of animals as one observation
(core.distance_effort, core.groups and the Distance Sampling step).

Twelve deer in one meadow are one thing the camera came across. Counted as
twelve sightings they make the detection function look better supported
than it is and the interval narrower than the survey can justify. So the
detection function is fitted to the groups, and the animals come back in
through the mean group size - with its own uncertainty.
"""
import json
import math
import os

import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import distance_effort as de
from bambi_wildlife_detection.core import groups, store
from bambi_wildlife_detection.core.survey import Survey, SurveyFlight

from .survey_flights import EPSG, ORIGIN, make_flight


def seen(ident, distance, line=0, side=1.0, along=0.0):
    """An animal *distance* m to one *side* of a line along the x axis."""
    return {"id": ident, "distance_m": float(distance), "line": line,
            "centre": (along, side * distance), "foot": (along, 0.0)}


# ---------------------------------------------------------------------------
# What is observed
# ---------------------------------------------------------------------------

class TestObservations:
    def test_without_groups_the_animals_are_the_observations(self):
        animals = [seen(1, 4.0), seen(2, 9.0, line=1)]

        observations, in_transit = de.group_observations(animals)

        assert observations == [
            {"group": 1, "size": 1, "distance_m": 4.0, "line": 0},
            {"group": 2, "size": 1, "distance_m": 9.0, "line": 1}]
        assert in_transit == 0

    def test_a_group_is_one_observation_at_the_distance_of_its_middle(self):
        herd = [seen(1, 10.0), seen(2, 20.0), seen(3, 30.0, along=15.0)]

        observations, _n = de.group_observations(herd, {1: 1, 2: 1, 3: 1}.get)

        assert observations == [
            {"group": 1, "size": 3, "distance_m": pytest.approx(20.0),
             "line": 0}]

    def test_a_group_either_side_of_the_line_has_its_middle_on_it(self):
        """The mean of their distances would put a herd the drone flew
        straight over 15 m off the line."""
        herd = [seen(1, 15.0, side=1.0), seen(2, 15.0, side=-1.0)]

        observations, _n = de.group_observations(herd, {1: 1, 2: 1}.get)

        assert observations[0]["distance_m"] == pytest.approx(0.0)
        assert observations[0]["size"] == 2

    def test_without_places_the_mean_distance_has_to_do(self):
        herd = [dict(seen(1, 10.0), centre=None), seen(2, 30.0)]

        observations, _n = de.group_observations(herd, {1: 1, 2: 1}.get)

        assert observations[0]["distance_m"] == pytest.approx(20.0)

    def test_a_group_is_on_the_line_most_of_it_was_seen_from(self):
        herd = [seen(1, 5.0, line=2), seen(2, 6.0, line=1),
                seen(3, 7.0, line=1)]

        observations, _n = de.group_observations(
            herd, {1: 1, 2: 1, 3: 1}.get)

        assert observations[0]["line"] == 1

    def test_a_group_seen_in_transit_only_is_not_counted(self):
        herd = [seen(1, 5.0, line=None), seen(2, 6.0, line=None),
                seen(3, 7.0, line=0)]

        observations, in_transit = de.group_observations(
            herd, {1: 1, 2: 1, 3: 3}.get)

        assert [(o["group"], o["size"]) for o in observations] == [(3, 1)]
        assert in_transit == 2

    def test_a_group_seen_from_a_line_counts_with_all_its_members(self):
        """The leg ended in the middle of the herd."""
        herd = [seen(1, 5.0, line=0), seen(2, 6.0, line=None)]

        observations, in_transit = de.group_observations(herd, {1: 1, 2: 1}.get)

        assert [(o["size"], o["line"]) for o in observations] == [(2, 0)]
        assert in_transit == 0

    def test_an_animal_the_groups_do_not_know_is_alone(self):
        animals = [seen(1, 5.0), seen(2, 6.0), seen(3, 7.0)]

        observations, _n = de.group_observations(animals, {1: 1, 2: 1}.get)

        assert [(o["group"], o["size"]) for o in observations] == [
            (1, 2), (3, 1)]

    def test_animals_without_an_id_are_not_one_group(self):
        animals = [dict(seen(None, 5.0)), dict(seen(None, 6.0))]

        observations, _n = de.group_observations(animals, {None: 1}.get)

        assert [o["size"] for o in observations] == [1, 1]

    def test_nothing_seen(self):
        assert de.group_observations([]) == ([], 0)


class TestGroupSize:
    def test_mean_standard_error_and_cv(self):
        size = de.group_size([1, 1, 4, 6])

        assert size["mean"] == 3.0 and size["largest"] == 6
        assert size["se"] == pytest.approx(math.sqrt(6.0 / 4))
        assert size["cv"] == pytest.approx(math.sqrt(6.0 / 4) / 3.0)

    def test_animals_all_alone_add_no_uncertainty(self):
        assert de.group_size([1, 1, 1]) == {
            "mean": 1.0, "se": 0.0, "cv": 0.0, "largest": 1}

    def test_one_group(self):
        assert de.group_size([4]) == {
            "mean": 4.0, "se": 0.0, "cv": 0.0, "largest": 4}

    def test_no_group(self):
        assert de.group_size([])["mean"] is None


class TestSizeBias:
    def test_larger_groups_further_away_are_suspected(self):
        pytest.importorskip("scipy")
        distances = [1.0, 3.0, 5.0, 8.0, 12.0, 15.0, 18.0, 20.0]
        sizes = [1, 1, 2, 2, 4, 5, 7, 9]

        bias = de.size_bias(distances, sizes)

        assert bias["correlation"] > 0.9
        assert bias["p_value"] < 0.01 and bias["suspected"]

    def test_sizes_that_have_nothing_to_do_with_distance_are_not(self):
        pytest.importorskip("scipy")
        distances = [1.0, 3.0, 5.0, 8.0, 12.0, 15.0, 18.0, 20.0]
        sizes = [3, 1, 4, 1, 3, 1, 4, 2]

        bias = de.size_bias(distances, sizes)

        assert abs(bias["correlation"]) < 0.3 and not bias["suspected"]

    def test_larger_groups_nearer_are_no_bias_of_this_kind(self):
        pytest.importorskip("scipy")
        distances = [1.0, 3.0, 5.0, 8.0, 12.0, 15.0]
        sizes = [9, 7, 5, 3, 2, 1]

        bias = de.size_bias(distances, sizes)

        assert bias["correlation"] < 0 and not bias["suspected"]

    def test_too_few_groups_say_nothing(self):
        assert de.size_bias([1.0, 2.0, 3.0, 4.0], [1, 2, 3, 4]) is None

    def test_groups_all_of_one_size_say_nothing(self):
        assert de.size_bias([1.0, 2.0, 3.0, 4.0, 5.0], [2] * 5) is None


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------

#: The drone flies due east along y = 0, one frame per second and 5 m per
#: frame: two legs of 500 m (frames 20-120 and 140-240) with a gap between.
#: (id, species, x along the line, y off it, frame it is under the drone)
ANIMALS = [
    # leg A: a herd of four red deer, two of them either side of the line
    (1, "red deer", 150, 12), (2, "red deer", 160, 20),
    (3, "red deer", 170, -6), (4, "red deer", 180, 14),
    # ... a pair, and three on their own
    (5, "red deer", 300, 30), (6, "red deer", 320, 34),
    (7, "red deer", 420, 5), (8, "red deer", 500, 45),
    (9, "red deer", 560, 18),
    # in transit between the legs
    (10, "red deer", 650, 9),
    # leg B: a herd of three, a pair, two on their own
    (11, "red deer", 800, 8), (12, "red deer", 815, 16),
    (13, "red deer", 830, 12),
    (14, "red deer", 950, 26), (15, "red deer", 965, 22),
    (16, "red deer", 1050, 3), (17, "red deer", 1150, 38),
    # roe deer are never grouped in this project (distance 0)
    (18, "roe deer", 400, 10), (19, "roe deer", 405, 12),
]


def _flight(folder, never_grouped=("roe deer",)):
    tracks = []
    for ident, species, x, y in ANIMALS:
        frame = int(x / 5)
        tracks.append({"id": ident, "species": species, "at": (x, y),
                       "frames": (frame - 3, frame + 3)})
    make_flight(folder, tracks, n_frames=260)
    conn = store.open_store(store.project_path(folder), store.PROJECT)
    with store.transaction(conn):
        for name in never_grouped:
            row = conn.execute("SELECT species_id FROM species WHERE name = ?",
                               (name,)).fetchone()
            groups.store_distance(conn, row["species_id"], 0)
    conn.close()
    os.makedirs(os.path.join(folder, "transects_t"))
    with open(os.path.join(folder, "transects_t", "transects.json"),
              "w") as handle:
        json.dump({"version": 1, "modality": "t", "transects": [
            {"id": 1, "name": "A", "start_frame": 20, "end_frame": 120},
            {"id": 2, "name": "B", "start_frame": 140, "end_frame": 240}]},
            handle)
    route = os.path.join(folder, "flight_route_t")
    os.makedirs(route)
    entries = [{"track_id": ident, "class_id": 1, "distance_m": abs(y),
                "last_frame": int(x / 5) + 3,
                "detection_center": [ORIGIN[0] + x, ORIGIN[1] + y, 0.0],
                "foot_point": [ORIGIN[0] + x, ORIGIN[1]]}
               for ident, _species, x, y in ANIMALS]
    with open(os.path.join(route, "perpendicular_tracks_t.json"),
              "w") as handle:
        json.dump({"tracks": entries}, handle)
    return folder


def _run(tmp_path, folder, species=None, truncation=50.0):
    survey_folder = str(tmp_path / "survey")
    survey = Survey(folder=survey_folder, modality="t", species=species,
                    flights=[SurveyFlight(name="Meadow", target_folder=folder,
                                          epsg=EPSG)])
    log = []
    BambiProcessor().run_distance_sampling({
        "survey": survey.to_dict(), "target_epsg": EPSG,
        "ds_truncation": truncation}, log_fn=log.append)
    with open(os.path.join(survey_folder, "analytics_t",
                           "distance_sampling_tracks.json")) as handle:
        return json.load(handle), log


@pytest.fixture
def flight(tmp_path):
    return _flight(str(tmp_path / "flight"))


def test_groups_are_the_observations(tmp_path, flight):
    pytest.importorskip("scipy")
    result, log = _run(tmp_path, flight, species=["red deer"])

    # leg A: herd of 4, pair, three alone; leg B: herd of 3, pair, two alone
    assert result["n"] == result["n_groups"] == 9
    assert result["n_animals"] == 16
    assert result["n_off_effort"] == 1
    assert result["mean_group_size"] == pytest.approx(16 / 9)
    assert result["largest_group"] == 4
    assert [(line["name"], line["n"], line["animals"])
            for line in result["lines"]] == [("A", 5, 9), ("B", 4, 7)]
    assert result["projects"][0]["n"] == 9
    assert result["projects"][0]["n_animals"] == 16
    assert any("The 9 observations are groups of 16 animals" in line
               for line in log)
    assert any("9 group(s) of 16 animals on 2 line(s)" in line
               for line in log)
    assert "the detection function is fitted to the 9 groups" in \
        result["notes"]


def test_the_animals_come_back_in_through_the_mean_group_size(tmp_path,
                                                              flight):
    pytest.importorskip("scipy")
    result, _log = _run(tmp_path, flight, species=["red deer"])

    esw = result["effective_strip_width_m"]
    length = result["transect_length_m"]
    assert length == pytest.approx(1000.0)
    group_density = 9 / (2.0 * esw * length) * 1e6
    assert result["group_density_per_km2"] == pytest.approx(group_density)
    assert result["density_per_km2"] == pytest.approx(
        group_density * 16 / 9)
    assert result["abundance_in_covered_area"] == pytest.approx(
        result["density_per_km2"] * result["covered_area_km2"])


def test_the_group_sizes_uncertainty_is_part_of_the_interval(tmp_path,
                                                             flight):
    pytest.importorskip("scipy")
    result, _log = _run(tmp_path, flight, species=["red deer"])

    sizes = [4, 2, 1, 1, 1, 3, 2, 1, 1]
    expected = de.group_size(sizes)
    assert result["group_size_se"] == pytest.approx(expected["se"])
    assert result["group_size_cv"] == pytest.approx(expected["cv"])
    of_size = result["group_size_cv"] ** 2
    without = math.sqrt(result["cv_density"] ** 2 - of_size)
    encounter, _how = de.encounter_rate_cv([5, 4], [500.0, 500.0])
    assert without >= encounter
    assert result["cv_density"] > without


def test_a_herd_the_drone_flew_over_is_near_the_line(tmp_path, flight):
    """Tracks 1-4: 12, 20 and 14 m to one side, 6 m to the other - their
    middle is 10 m off, not the 13 m their distances average to."""
    from bambi_wildlife_detection.core import distance_effort

    entries = [{"id": i, "distance_m": abs(y), "line": 0,
                "centre": (x, y), "foot": (x, 0.0)}
               for i, _s, x, y in ANIMALS[:4]]
    observations, _n = distance_effort.group_observations(
        entries, lambda ident: 1)

    assert observations[0]["distance_m"] == pytest.approx(10.0)


def test_a_species_that_is_never_grouped_is_counted_animal_by_animal(
        tmp_path, flight):
    pytest.importorskip("scipy")
    result, _log = _run(tmp_path, flight, species=["roe deer"],
                        truncation=40.0)

    assert result["n"] == result["n_animals"] == 2
    assert result["mean_group_size"] == 1.0
    assert result["group_size_cv"] == 0.0
    assert result["group_size_bias"] is None
    assert "groups" not in result["notes"]


def test_with_nothing_grouped_the_result_is_what_it_was(tmp_path):
    """Every species at distance 0: the estimate is the one distance
    sampling made before it knew of groups."""
    pytest.importorskip("scipy")
    grouped, _log = _run(tmp_path / "a", _flight(str(tmp_path / "a" / "f")),
                         species=["red deer"])
    alone, _log = _run(
        tmp_path / "b",
        _flight(str(tmp_path / "b" / "f"),
                never_grouped=("roe deer", "red deer")),
        species=["red deer"])

    assert alone["n"] == alone["n_animals"] == 16
    assert alone["mean_group_size"] == 1.0
    assert alone["density_per_km2"] == pytest.approx(
        alone["group_density_per_km2"])
    esw = alone["effective_strip_width_m"]
    assert alone["density_per_km2"] == pytest.approx(
        16 / (2.0 * esw * 1000.0) * 1e6)
    encounter, _how = de.encounter_rate_cv([9, 7], [500.0, 500.0])
    assert alone["encounter_rate_cv"] == pytest.approx(encounter)
    # ... and the same animals, counted as the groups they were in, are
    # fewer observations
    assert grouped["n"] == 9 < alone["n"]


def test_the_result_says_when_larger_groups_were_seen_further_away(tmp_path):
    pytest.importorskip("scipy")
    folder = str(tmp_path / "bias" / "f")
    tracks, entries, ident = [], [], 1
    # one animal near the line, ever larger groups further out
    for size, distance, x in ((1, 2, 100), (1, 4, 180), (2, 8, 260),
                              (2, 12, 340), (3, 18, 420), (4, 24, 500),
                              (5, 30, 580), (6, 36, 660)):
        for k in range(size):
            frame = int(x / 5)
            tracks.append({"id": ident, "species": "red deer",
                           "at": (x + 3 * k, distance),
                           "frames": (frame - 2, frame + 2)})
            entries.append({
                "track_id": ident, "class_id": 1, "distance_m": distance,
                "last_frame": frame + 2,
                "detection_center": [ORIGIN[0] + x + 3 * k,
                                     ORIGIN[1] + distance, 0.0],
                "foot_point": [ORIGIN[0] + x + 3 * k, ORIGIN[1]]})
            ident += 1
    make_flight(folder, tracks, n_frames=160)
    route = os.path.join(folder, "flight_route_t")
    os.makedirs(route)
    with open(os.path.join(route, "perpendicular_tracks_t.json"),
              "w") as handle:
        json.dump({"tracks": entries}, handle)

    result, _log = _run(tmp_path / "bias", folder)

    assert result["n_groups"] == 8 and result["n_animals"] == 24
    assert result["group_size_bias"]["suspected"]
    assert result["group_size_bias"]["correlation"] > 0.9
    assert "Larger groups were seen from further away" in result["notes"]
