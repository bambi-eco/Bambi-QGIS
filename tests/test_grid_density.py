# -*- coding: utf-8 -*-
"""Animal density cell by cell (core/grid_density.py)."""
import csv
import json

import pytest

from bambi_wildlife_detection.core import grid_density as gd
from bambi_wildlife_detection.core import spatial_stats, survey_grid as sg

shapely = pytest.importorskip("shapely")
from shapely.geometry import box  # noqa: E402

SQUARES = sg.GridSpec(shape=sg.SQUARE, size_mode=sg.BY_EDGE, edge_m=100.0)
#: Four cells by three, a hectare each.
REGION = box(0, 0, 400, 300)


def animal(x, y, species="roe deer", group=None, sex=None, age=None):
    return {"species": species, "x": x, "y": y, "group": group, "sex": sex,
            "age": age}


def flight(name, seen, *animals):
    return {"name": name, "seen": seen, "records": list(animals)}


def cells(document):
    return {cell["cell"]: cell for cell in document["cells"]}


# ---------------------------------------------------------------------------
# Which cells there are, and what was seen of them
# ---------------------------------------------------------------------------

class TestCells:
    def test_the_grid_is_the_cells_on_the_surveys_ground(self):
        document = gd.analyse(SQUARES, [flight("A", REGION)], REGION)

        assert sorted(cells(document)) == sorted(
            f"c{col}r{row}" for col in range(4) for row in range(3))
        assert document["summary"]["cells"] == 12
        assert document["grid_text"] == "squares of 1 ha (edge 100.0 m)"

    def test_a_cell_that_only_touches_the_border_is_not_one(self):
        """The region ends at x = 400: the cells from 400 on share a line
        with it, and no ground."""
        document = gd.analyse(SQUARES, [flight("A", REGION)], REGION)

        assert "c4r0" not in cells(document)

    def test_a_cell_is_whole_also_where_the_area_cuts_it(self):
        region = box(0, 0, 250, 100)
        document = gd.analyse(SQUARES, [flight("A", region)], region)
        cut = cells(document)["c2r0"]

        assert cut["area_m2"] == 10000.0
        assert cut["in_area_m2"] == pytest.approx(5000.0)
        assert len(cut["ring"]) == 4 and cut["x"] == 250.0

    def test_what_was_seen_is_measured_inside_the_cell(self):
        seen = box(0, 0, 400, 130)
        document = gd.analyse(SQUARES, [flight("A", seen)], REGION)
        found = cells(document)

        assert found["c0r0"]["seen_share"] == pytest.approx(1.0)
        assert found["c0r1"]["seen_share"] == pytest.approx(0.3)
        assert found["c0r1"]["seen_m2"] == pytest.approx(3000.0)
        assert found["c0r2"]["seen_share"] == 0.0
        assert document["summary"]["seen_ha"] == pytest.approx(5.2)
        assert document["summary"]["seen_share"] == pytest.approx(5.2 / 12)

    def test_what_was_seen_outside_the_area_does_not_count(self):
        region = box(0, 0, 150, 100)
        document = gd.analyse(SQUARES, [flight("A", box(0, 0, 400, 100))],
                              region)

        assert cells(document)["c1r0"]["seen_m2"] == pytest.approx(5000.0)
        assert document["summary"]["seen_ha"] == pytest.approx(1.5)

    def test_cells_are_counted_too_little_seen_or_not_seen(self):
        seen = box(0, 0, 400, 130)
        document = gd.analyse(SQUARES, [flight("A", seen)], REGION)
        status = {name: cell["status"] for name, cell in
                  cells(document).items()}

        assert status["c0r0"] == gd.COUNTED
        assert status["c0r1"] == gd.TOO_LITTLE          # 30 % seen
        assert status["c0r2"] == gd.NOT_SEEN
        assert document["summary"]["counted"] == 4
        assert document["summary"]["too_little_seen"] == 4
        assert document["summary"]["not_seen"] == 4

    def test_the_share_that_has_to_be_seen_can_be_set(self):
        seen = box(0, 0, 400, 130)

        lenient = gd.analyse(SQUARES, [flight("A", seen)], REGION,
                             min_seen=0.25)
        strict = gd.analyse(SQUARES, [flight("A", box(0, 0, 400, 95))],
                            REGION, min_seen=1.0)

        assert lenient["summary"]["counted"] == 8
        assert strict["summary"]["counted"] == 0
        assert lenient["min_seen"] == 0.25

    def test_exactly_the_share_counts(self):
        document = gd.analyse(SQUARES, [flight("A", box(0, 0, 400, 150))],
                              REGION)

        assert cells(document)["c0r1"]["status"] == gd.COUNTED

    def test_a_sliver_left_by_the_survey_area_is_too_little(self):
        """The area ends 10 m into the last column: all of that was seen,
        and it is a tenth of a cell."""
        region = box(0, 0, 310, 100)
        document = gd.analyse(SQUARES, [flight("A", region)], region)

        assert cells(document)["c3r0"]["seen_share"] == pytest.approx(0.1)
        assert cells(document)["c3r0"]["status"] == gd.TOO_LITTLE


# ---------------------------------------------------------------------------
# Animals and density
# ---------------------------------------------------------------------------

class TestDensity:
    def test_density_is_animals_over_ground_seen(self):
        seen = box(0, 0, 400, 150)
        document = gd.analyse(SQUARES, [flight(
            "A", seen, animal(50, 50), animal(60, 40), animal(150, 120))],
            REGION)
        found = cells(document)

        assert found["c0r0"]["animals"] == 2
        assert found["c0r0"]["density_per_km2"] == pytest.approx(200.0)
        # one animal on the half of the cell that was seen
        assert found["c1r1"]["animals"] == 1
        assert found["c1r1"]["effort_m2"] == pytest.approx(5000.0)
        assert found["c1r1"]["density_per_km2"] == pytest.approx(200.0)
        assert found["c2r0"]["density_per_km2"] == 0.0

    def test_an_animal_is_in_one_cell_also_on_an_edge(self):
        document = gd.analyse(SQUARES, [flight(
            "A", REGION, animal(100.0, 50.0), animal(200.0, 200.0))], REGION)

        assert sum(cell["animals"] for cell in document["cells"]) == 2
        assert document["summary"]["animals_in_cells"] == 2

    def test_animals_outside_the_ground_and_without_a_place_are_in_no_cell(
            self):
        document = gd.analyse(SQUARES, [flight(
            "A", REGION, animal(50, 50), animal(950, 50),
            animal(None, None))], REGION)

        assert document["summary"]["animals_in_cells"] == 1
        assert document["summary"]["animals_outside"] == 1
        assert document["summary"]["animals_unplaced"] == 1

    def test_an_animal_on_ground_nobody_saw_is_counted_but_has_no_density(
            self):
        document = gd.analyse(SQUARES, [flight(
            "A", box(0, 0, 400, 100), animal(50, 250))], REGION)
        cell = cells(document)["c0r2"]

        assert cell["status"] == gd.NOT_SEEN and cell["animals"] == 1
        assert cell["density_per_km2"] is None
        assert document["summary"]["animals_in_counted_cells"] == 0

    def test_species_are_counted_apart(self):
        document = gd.analyse(SQUARES, [flight(
            "A", REGION, animal(50, 50), animal(60, 60),
            animal(70, 70, "red deer"), animal(350, 250, "red deer"))],
            REGION, species_order=["red deer", "roe deer", "fox"])
        cell = cells(document)["c0r0"]

        assert document["species"] == ["red deer", "roe deer"]
        assert cell["animals"] == 3
        assert cell["species"]["roe deer"]["animals"] == 2
        assert cell["species"]["roe deer"]["density_per_km2"] == \
            pytest.approx(200.0)
        assert cell["species"]["red deer"]["density_per_km2"] == \
            pytest.approx(100.0)
        # a species that is not in a cell is there with nought
        assert cells(document)["c1r0"]["species"]["roe deer"] == {
            "animals": 0, "effort_m2": 10000.0, "density_per_km2": 0.0,
            "gi_z": cells(document)["c1r0"]["species"]["roe deer"]["gi_z"],
            "spot": ""}

    def test_sex_and_age(self):
        document = gd.analyse(SQUARES, [flight(
            "A", REGION, animal(50, 50, sex="Female", age="adult"),
            animal(60, 60, sex="male"), animal(70, 70))], REGION)
        cell = cells(document)["c0r0"]

        assert cell["sex"] == {"female": 1, "male": 1, "unknown": 1}
        assert cell["age"] == {"adult": 1, "unknown": 2}
        assert document["sex_labels"] == ["female", "male", "unknown"]
        assert document["age_labels"] == ["adult", "unknown"]

    def test_the_groups_among_a_cells_animals(self):
        document = gd.analyse(SQUARES, [flight(
            "A", REGION, animal(50, 50, group=1), animal(60, 60, group=1),
            animal(70, 70, group=1), animal(20, 20, group=7),
            animal(150, 50, group=1))], REGION)
        found = cells(document)

        assert (found["c0r0"]["groups"],
                found["c0r0"]["mean_group_size"]) == (2, 2.0)
        # the herd reaches into the next cell: its member there is a group
        # of one, there
        assert (found["c1r0"]["groups"],
                found["c1r0"]["mean_group_size"]) == (1, 1.0)
        assert found["c2r0"]["groups"] == 0
        assert found["c2r0"]["mean_group_size"] is None

    def test_animals_without_a_group_leave_the_groups_empty(self):
        document = gd.analyse(SQUARES, [flight(
            "A", REGION, animal(50, 50), animal(60, 60, group=2))], REGION)

        assert cells(document)["c0r0"]["groups"] is None


# ---------------------------------------------------------------------------
# Ground flown more than once
# ---------------------------------------------------------------------------

class TestRepeats:
    #: Two flights over the same four cells; the second found more in the
    #: first cell and saw only half of the second.
    FLIGHTS = [
        flight("Morning", box(0, 0, 400, 100),
               animal(50, 50), animal(150, 50, "red deer"),
               animal(160, 50, "red deer")),
        flight("Evening", box(0, 0, 150, 100),
               animal(40, 40), animal(60, 60), animal(70, 70),
               animal(120, 50, "red deer")),
    ]
    REGION = box(0, 0, 400, 100)

    def test_average_is_all_sightings_over_all_ground_looked_at(self):
        document = gd.analyse(SQUARES, self.FLIGHTS, self.REGION,
                              repeat=gd.AVERAGE)
        found = cells(document)

        first = found["c0r0"]
        assert (first["animals"], first["flights"]) == (4, 2)
        assert first["effort_m2"] == pytest.approx(20000.0)
        assert first["density_per_km2"] == pytest.approx(200.0)
        assert first["seen_share"] == pytest.approx(1.0)
        assert first["best_flight"] is None
        second = found["c1r0"]
        assert second["animals"] == 3
        assert second["effort_m2"] == pytest.approx(15000.0)
        assert second["density_per_km2"] == pytest.approx(200.0)
        assert second["species"]["red deer"]["animals"] == 3
        assert found["c2r0"]["flights"] == 1

    def test_highest_is_the_flight_that_found_most(self):
        document = gd.analyse(SQUARES, self.FLIGHTS, self.REGION,
                              repeat=gd.HIGHEST)
        found = cells(document)

        first = found["c0r0"]
        assert (first["animals"], first["best_flight"]) == (3, "Evening")
        assert first["effort_m2"] == pytest.approx(10000.0)
        assert first["density_per_km2"] == pytest.approx(300.0)
        # the second cell: 2 on the whole of it against 1 on half - the
        # same density, and the flight that comes first keeps it
        second = found["c1r0"]
        assert (second["animals"], second["best_flight"]) == (2, "Morning")
        assert second["density_per_km2"] == pytest.approx(200.0)

    def test_highest_is_found_species_by_species(self):
        flights = [
            flight("Morning", box(0, 0, 100, 100), animal(50, 50),
                   animal(60, 60), animal(70, 70, "red deer")),
            flight("Evening", box(0, 0, 100, 100), animal(40, 40),
                   animal(20, 20, "red deer"), animal(30, 30, "red deer"),
                   animal(35, 35, "red deer")),
        ]
        document = gd.analyse(SQUARES, flights, box(0, 0, 100, 100),
                              repeat=gd.HIGHEST)
        cell = cells(document)["c0r0"]

        assert (cell["animals"], cell["best_flight"]) == (4, "Evening")
        assert cell["species"]["roe deer"]["animals"] == 2      # Morning
        assert cell["species"]["red deer"]["animals"] == 3      # Evening

    def test_a_flight_that_saw_too_little_of_a_cell_is_not_the_highest(self):
        """One animal on the tenth of a cell a flight clipped would be the
        highest density there ever was."""
        flights = [
            flight("Whole", box(0, 0, 100, 100), animal(50, 50),
                   animal(60, 60)),
            flight("Corner", box(0, 0, 100, 10), animal(50, 5)),
        ]
        document = gd.analyse(SQUARES, flights, box(0, 0, 100, 100),
                              repeat=gd.HIGHEST)
        cell = cells(document)["c0r0"]

        assert (cell["best_flight"], cell["animals"]) == ("Whole", 2)
        assert cell["density_per_km2"] == pytest.approx(200.0)

    def test_highest_needs_one_flight_that_saw_enough_on_its_own(self):
        """Two flights saw a third of the cell each - not the same third."""
        flights = [flight("West", box(0, 0, 33, 100), animal(10, 50)),
                   flight("East", box(67, 0, 100, 100))]
        region = box(0, 0, 100, 100)

        highest = gd.analyse(SQUARES, flights, region, repeat=gd.HIGHEST)
        average = gd.analyse(SQUARES, flights, region, repeat=gd.AVERAGE)

        assert cells(highest)["c0r0"]["status"] == gd.TOO_LITTLE
        assert cells(average)["c0r0"]["status"] == gd.COUNTED     # 66 %
        assert cells(average)["c0r0"]["seen_share"] == pytest.approx(0.66)

    def test_one_flight_reads_the_same_either_way(self):
        flights = [flight("A", box(0, 0, 400, 130), animal(50, 50),
                          animal(250, 110, "red deer"))]

        average = gd.analyse(SQUARES, flights, REGION, repeat=gd.AVERAGE)
        highest = gd.analyse(SQUARES, flights, REGION, repeat=gd.HIGHEST)

        for one, other in zip(average["cells"], highest["cells"]):
            assert one["status"] == other["status"]
            assert one["animals"] == other["animals"]
            assert one["density_per_km2"] == other["density_per_km2"]

    def test_every_flight_has_its_row_per_cell(self):
        document = gd.analyse(SQUARES, self.FLIGHTS, self.REGION)
        rows = {(row["cell"], row["flight"]): row
                for row in document["by_flight"]}

        assert sorted(rows) == [
            ("c0r0", "Evening"), ("c0r0", "Morning"), ("c1r0", "Evening"),
            ("c1r0", "Morning"), ("c2r0", "Morning"), ("c3r0", "Morning")]
        evening = rows[("c1r0", "Evening")]
        assert evening["animals"] == 1
        assert evening["seen_share"] == pytest.approx(0.5)
        assert evening["density_per_km2"] == pytest.approx(200.0)
        assert evening["species"] == {"red deer": 1}

    def test_an_unknown_way_to_read_repeats_is_refused(self):
        with pytest.raises(gd.GridDensityError, match="repeated flights"):
            gd.analyse(SQUARES, self.FLIGHTS, self.REGION, repeat="median")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _clumped():
    """A grid of 8 x 8 hectares, all seen: a herd in one corner, a few
    animals scattered over the rest."""
    region = box(0, 0, 800, 800)
    animals = []
    for col in range(3):
        for row in range(3):
            for k in range(6):
                animals.append(animal(100 * col + 10 + 12 * k,
                                      100 * row + 50, "red deer"))
    for col, row in ((5, 1), (6, 6), (2, 6), (7, 3)):
        animals.append(animal(100 * col + 50, 100 * row + 50))
    return region, [flight("A", region, *animals)]


class TestStatistics:
    def test_density_over_the_counted_cells_with_an_interval(self):
        region, flights = _clumped()
        document = gd.analyse(SQUARES, flights, region,
                              plot_m2=float(region.area))
        everything = document["statistics"][gd.ALL]

        assert everything["cells"] == 64 and everything["animals"] == 58
        density = everything["density_per_km2"]
        assert density["estimate"] == pytest.approx(58 / 0.64)
        assert density["low"] < density["estimate"] < density["high"]
        total = everything["total_in_survey_area"]
        assert total["estimate"] == pytest.approx(58.0)
        assert total["low"] == pytest.approx(density["low"] * 0.64)

    def test_no_survey_area_no_total(self):
        region, flights = _clumped()
        document = gd.analyse(SQUARES, flights, region)

        assert document["statistics"][gd.ALL]["total_in_survey_area"] is None
        assert document["summary"]["plot_ha"] is None

    def test_cells_that_do_not_count_are_not_in_the_statistics(self):
        seen = box(0, 0, 800, 430)
        region, flights = _clumped()
        flights[0]["seen"] = seen
        document = gd.analyse(SQUARES, flights, region)
        everything = document["statistics"][gd.ALL]

        assert everything["cells"] == 32                 # rows 0 to 3
        # rows 0-2 of the herd, and the two scattered animals in rows 1, 3
        assert everything["animals"] == 54 + 2
        assert everything["density_per_km2"]["estimate"] == pytest.approx(
            56 / 0.32)

    def test_a_herd_makes_the_pattern_clumped(self):
        region, flights = _clumped()
        document = gd.analyse(SQUARES, flights, region)
        red = document["statistics"]["red deer"]

        assert red["dispersion"]["pattern"] == spatial_stats.CLUSTERED
        assert red["dispersion"]["index"] > 3
        assert red["morans_i"]["pattern"] == spatial_stats.CLUSTERED
        assert red["morans_i"]["i"] > 0.4

    def test_a_few_scattered_animals_are_not(self):
        region, flights = _clumped()
        document = gd.analyse(SQUARES, flights, region)
        roe = document["statistics"]["roe deer"]

        assert roe["animals"] == 4
        assert roe["dispersion"]["pattern"] == spatial_stats.RANDOM

    def test_the_herds_cells_are_hot_spots(self):
        region, flights = _clumped()
        document = gd.analyse(SQUARES, flights, region)
        found = cells(document)

        assert found["c1r1"]["spot"] == "hot spot (99 %)"
        assert found["c1r1"]["gi_z"] > 2.58
        assert found["c1r1"]["species"]["red deer"]["spot"] == \
            "hot spot (99 %)"
        assert found["c6r6"]["spot"] == ""
        assert document["statistics"][gd.ALL]["hot_spots"] >= 4
        assert document["statistics"]["roe deer"]["hot_spots"] <= 2

    def test_only_counted_cells_are_neighbours(self):
        """A hot spot is not made by cells nobody looked at."""
        region, flights = _clumped()
        flights[0]["seen"] = box(0, 0, 800, 430)
        document = gd.analyse(SQUARES, flights, region)
        found = cells(document)

        assert found["c1r5"]["gi_z"] is None and found["c1r5"]["spot"] == ""
        assert found["c1r5"]["species"]["red deer"]["gi_z"] is None
        assert found["c1r1"]["gi_z"] is not None

    def test_the_share_of_cells_that_held_an_animal(self):
        region, flights = _clumped()
        document = gd.analyse(SQUARES, flights, region)
        occupancy = document["statistics"][gd.ALL]["occupancy"]

        assert (occupancy["occupied"], occupancy["cells"]) == (13, 64)
        assert occupancy["share"] == pytest.approx(13 / 64)
        assert occupancy["low"] < 13 / 64 < occupancy["high"]
        assert document["statistics"]["roe deer"]["occupancy"]["occupied"] == 4

    def test_the_same_survey_gives_the_same_numbers(self):
        region, flights = _clumped()

        first = gd.analyse(SQUARES, flights, region)
        again = gd.analyse(SQUARES, flights, region)

        assert json.dumps(first["statistics"]) == \
            json.dumps(again["statistics"])

    def test_statistics_of_the_highest_flight_use_that_flights_ground(self):
        flights = [
            flight("Morning", box(0, 0, 400, 100), animal(50, 50)),
            flight("Evening", box(0, 0, 400, 100), animal(40, 40),
                   animal(60, 60), animal(150, 50)),
        ]
        region = box(0, 0, 400, 100)

        highest = gd.analyse(SQUARES, flights, region, repeat=gd.HIGHEST)
        average = gd.analyse(SQUARES, flights, region, repeat=gd.AVERAGE)

        assert highest["statistics"][gd.ALL]["density_per_km2"][
            "estimate"] == pytest.approx(3 / 0.04)
        assert average["statistics"][gd.ALL]["density_per_km2"][
            "estimate"] == pytest.approx(4 / 0.08)


# ---------------------------------------------------------------------------
# Every shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [sg.HEXAGON, sg.TRIANGLE, sg.SQUARE,
                                   sg.RECTANGLE])
def test_every_shape_counts_every_animal_once(shape):
    import random

    spec = sg.GridSpec(shape=shape, area_m2=2500.0, width_m=80.0,
                       height_m=35.0)
    region = box(500013.0, 5300027.0, 500713.0, 5300527.0)
    rng = random.Random(9)
    animals = [animal(rng.uniform(500013, 500713),
                      rng.uniform(5300027, 5300527)) for _ in range(300)]
    document = gd.analyse(spec, [flight("A", region, *animals)], region,
                          plot_m2=float(region.area))

    assert sum(cell["animals"] for cell in document["cells"]) == 300
    assert document["summary"]["animals_outside"] == 0
    assert sum(cell["in_area_m2"] for cell in document["cells"]) == \
        pytest.approx(region.area)
    assert sum(cell["seen_m2"] for cell in document["cells"]) == \
        pytest.approx(region.area)
    counted = [cell for cell in document["cells"]
               if cell["status"] == gd.COUNTED]
    assert all(cell["seen_share"] >= 0.5 for cell in counted)
    # 300 animals on 35 ha: about 857 per km²
    assert document["statistics"][gd.ALL]["density_per_km2"][
        "estimate"] == pytest.approx(857.0, rel=0.08)


# ---------------------------------------------------------------------------
# What cannot be analysed
# ---------------------------------------------------------------------------

def test_without_ground_there_is_no_grid():
    with pytest.raises(gd.GridDensityError, match="no ground"):
        gd.analyse(SQUARES, [flight("A", REGION)], None)
    with pytest.raises(gd.GridDensityError, match="no ground"):
        gd.analyse(SQUARES, [flight("A", REGION)], box(0, 0, 0, 0))


def test_without_flights_there_is_nothing_to_analyse():
    with pytest.raises(gd.GridDensityError, match="No flight"):
        gd.analyse(SQUARES, [], REGION)


def test_a_grid_that_is_far_too_fine_is_refused():
    fine = sg.GridSpec(shape=sg.SQUARE, size_mode=sg.BY_EDGE, edge_m=1.0)

    with pytest.raises(gd.GridDensityError, match="Choose larger cells"):
        gd.analyse(fine, [flight("A", box(0, 0, 5000, 5000))],
                   box(0, 0, 5000, 5000))


def test_the_analysis_can_be_cancelled():
    with pytest.raises(InterruptedError):
        gd.analyse(SQUARES, [flight("A", REGION)], REGION,
                   cancelled=lambda: True)


def test_progress_is_reported():
    seen = []
    gd.analyse(SQUARES, [flight("A", REGION, animal(50, 50))], REGION,
               progress=seen.append)

    assert seen == sorted(seen) and seen[0] >= 5 and seen[-1] == 95


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

@pytest.fixture
def document():
    region, flights = _clumped()
    flights.append(flight("B", box(0, 0, 800, 100), animal(50, 50)))
    flights[0]["records"][0]["sex"] = "female"
    flights[0]["records"][1]["age"] = "juvenile"
    for record in flights[0]["records"][:6]:
        record["group"] = 1
    return gd.analyse(SQUARES, flights, region, plot_m2=float(region.area))


class TestTables:
    def test_a_row_per_cell_with_a_column_per_value(self, document):
        rows = gd.flat_cells(document)
        first = rows[0]

        assert len(rows) == 64
        assert list(first)[:6] == ["cell", "status", "animals",
                                   "density_per_km2", "groups",
                                   "mean_group_size"]
        assert first["cell"] == "c0r0" and first["animals"] == 7
        assert first["n_red_deer"] == 6 and first["n_roe_deer"] == 1
        assert first["per_km2_red_deer"] == pytest.approx(300.0)
        assert first["spot_red_deer"] == "hot spot (99 %)"
        assert (first["sex_female"], first["sex_unknown"]) == (1, 6)
        assert first["age_juvenile"] == 1
        assert (first["area_ha"], first["seen_ha"], first["effort_ha"]) == (
            1.0, 1.0, 2.0)
        assert first["flights"] == 2 and first["seen_share"] == 1.0
        assert (first["x"], first["y"]) == (50.0, 50.0)
        assert all(list(row) == list(first) for row in rows)

    def test_a_row_per_cell_and_flight(self, document):
        rows = gd.flat_flights(document)

        assert len(rows) == 64 + 8
        b = next(row for row in rows
                 if (row["cell"], row["flight"]) == ("c0r0", "B"))
        assert (b["animals"], b["n_roe_deer"], b["n_red_deer"]) == (1, 1, 0)
        assert b["seen_ha"] == 1.0 and b["density_per_km2"] == 100.0

    def test_the_tables_are_written_as_csv(self, document, tmp_path):
        path = gd.write_table(gd.flat_cells(document),
                              str(tmp_path / "cells.csv"))
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 64
        assert rows[0]["cell"] == "c0r0" and rows[0]["n_red_deer"] == "6"
        assert rows[0]["best_flight"] == ""            # none: empty, not None
        assert rows[63]["density_per_km2"] == "0.0"
        gd.write_table([], str(tmp_path / "empty.csv"))
        assert (tmp_path / "empty.csv").read_text() == ""

    def test_the_document_survives_being_written_and_read(self, document,
                                                          tmp_path):
        path = gd.write_json(document, str(tmp_path / "grid.json"))
        again = gd.read_json(path)

        assert gd.flat_cells(again) == gd.flat_cells(document)
        assert list(again["statistics"]) == [gd.ALL, "red deer", "roe deer"]
        assert gd.describe(again) == gd.describe(document)

    def test_the_result_in_a_few_lines(self, document):
        lines = gd.describe(document)

        assert lines[0] == ("Grid: squares of 1 ha (edge 100.0 m) - 64 "
                            "cell(s), 64 counted, 0 too little seen, 0 not "
                            "seen")
        assert lines[1] == "64.0 of 64.0 ha were seen (100 %)"
        red = next(line for line in lines if line.startswith("red deer:"))
        assert "54 in the counted cells" in red
        assert "/km²" in red and "in the survey area" in red
        assert "clumped (variance-to-mean" in red
        assert "in 9 of 64 cells" in red and "hot-spot cell(s)" in red

    def test_a_column_name_from_a_species_name(self):
        assert gd.slug("roe deer") == "roe_deer"
        assert gd.slug("Red-Deer (stag)") == "red_deer_stag"
        assert gd.slug("") == "species"


# ---------------------------------------------------------------------------
# Environment values
# ---------------------------------------------------------------------------

class TestEnvironment:
    @pytest.fixture
    def rasters(self):
        np = pytest.importorskip("numpy")
        pytest.importorskip("rasterio")
        from bambi_wildlife_detection.core.environment import Grid

        # 10 m pixels over 400 m x 300 m, north up
        grid = Grid(epsg=32633, min_x=0.0, max_y=300.0, width=40, height=30,
                    resolution=10.0)
        ndvi = np.full((30, 40), 0.2, dtype="float32")
        ndvi[20:, :10] = 0.8                 # the south-west cell: greener
        ndvi[0:10, 30:] = np.nan             # the north-east cell: no data
        land = np.full((30, 40), 30, dtype="uint8")       # grassland
        land[20:, :7] = 10                   # 70 % of the south-west: trees
        return grid, ndvi, land

    def test_every_cell_gets_the_mean_of_what_lies_in_it(self, rasters):
        grid, ndvi, _land = rasters
        document = gd.analyse(SQUARES, [flight("A", REGION)], REGION)

        values = gd.zonal_by_cells(
            ndvi, grid, [cell["ring"] for cell in document["cells"]], False)
        by_cell = dict(zip([cell["cell"] for cell in document["cells"]],
                           values))

        assert by_cell["c0r0"] == (pytest.approx(0.8),)
        assert by_cell["c1r0"] == (pytest.approx(0.2),)
        assert by_cell["c3r2"] is None

    def test_the_most_common_class_and_its_share(self, rasters):
        grid, _ndvi, land = rasters
        document = gd.analyse(SQUARES, [flight("A", REGION)], REGION)

        values = gd.zonal_by_cells(
            land, grid, [cell["ring"] for cell in document["cells"]], True)
        by_cell = dict(zip([cell["cell"] for cell in document["cells"]],
                           values))

        assert by_cell["c0r0"] == (10, pytest.approx(0.7))
        assert by_cell["c1r0"] == (30, pytest.approx(1.0))

    def test_cells_take_a_layer_from_the_first_flight_that_has_it(
            self, rasters):
        np = pytest.importorskip("numpy")
        grid, ndvi, land = rasters
        other = np.full((30, 40), 0.5, dtype="float32")
        document = gd.analyse(SQUARES, [flight("A", REGION)], REGION)

        keys = gd.add_environment(
            document,
            [{"ndvi": (ndvi, grid), "landcover": (land, grid)},
             {"ndvi": (other, grid)}],
            class_name={10: "Tree cover", 30: "Grassland"}.get)
        found = cells(document)

        assert keys == ["ndvi", "landcover"]
        assert found["c0r0"]["environment"] == {
            "ndvi": 0.8, "landcover": "Tree cover", "landcover_share": 0.7}
        # the first flight has no value there, the second has
        assert found["c3r2"]["environment"]["ndvi"] == 0.5
        rows = gd.flat_cells(document)
        assert rows[0]["ndvi"] == 0.8 and rows[0]["landcover"] == "Tree cover"
        assert rows[0]["landcover_share"] == 0.7
        assert "ndvi" in rows[5]

    def test_no_environment_no_columns(self):
        document = gd.analyse(SQUARES, [flight("A", REGION)], REGION)

        assert gd.add_environment(document, []) == []
        assert "ndvi" not in gd.flat_cells(document)[0]
