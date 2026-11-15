# -*- coding: utf-8 -*-
"""The grid a survey is counted in (core/survey_grid.py)."""
import math
import random

import pytest

from bambi_wildlife_detection.core import survey_grid as sg
from bambi_wildlife_detection.core.survey_grid import GridError, GridSpec

REGULAR = [sg.HEXAGON, sg.SQUARE, sg.TRIANGLE]
#: Somewhere in UTM 33N, not on a round number.
BOUNDS = (500013.0, 5300027.0, 500913.0, 5300727.0)


def _polygon(cell):
    shapely = pytest.importorskip("shapely.geometry")
    return shapely.Polygon(cell.ring)


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------

class TestSize:
    def test_a_hectare(self):
        assert sg.edge_for_area(sg.SQUARE, 10000.0) == pytest.approx(100.0)
        assert sg.edge_for_area(sg.HEXAGON, 10000.0) == pytest.approx(
            62.0403, abs=1e-3)
        assert sg.edge_for_area(sg.TRIANGLE, 10000.0) == pytest.approx(
            151.9671, abs=1e-3)

    @pytest.mark.parametrize("shape", REGULAR)
    def test_edge_and_area_are_two_ways_to_say_the_same(self, shape):
        for area in (25.0, 10000.0, 250000.0):
            assert sg.area_for_edge(shape, sg.edge_for_area(shape, area)) \
                == pytest.approx(area)

    def test_the_default_is_a_hexagon_of_a_hectare(self):
        spec = GridSpec()

        assert (spec.shape, spec.size_mode) == (sg.HEXAGON, sg.BY_AREA)
        assert spec.cell_area_m2 == pytest.approx(10000.0)
        assert spec.describe() == "hexagons of 1 ha (edge 62.0 m)"

    def test_size_by_edge(self):
        spec = GridSpec(shape=sg.SQUARE, size_mode=sg.BY_EDGE, edge_m=50.0,
                        area_m2=10000.0)

        assert spec.edge == 50.0 and spec.cell_area_m2 == 2500.0
        assert spec.describe() == "squares of 0.25 ha (edge 50.0 m)"

    def test_a_rectangle_has_its_own_sides(self):
        spec = GridSpec(shape=sg.RECTANGLE, width_m=200.0, height_m=50.0)

        assert spec.cell_area_m2 == 10000.0
        assert spec.describe() == "rectangles of 200 m x 50 m (1 ha)"

    def test_small_cells_are_described_in_square_metres(self):
        assert GridSpec(shape=sg.SQUARE, area_m2=400.0).describe() == \
            "squares of 400 m² (edge 20.0 m)"

    def test_from_the_settings(self):
        spec = GridSpec.from_config({
            "grid_shape": "Triangle", "grid_size_mode": "edge",
            "grid_cell_edge_m": 80, "grid_cell_area_ha": 2})

        assert (spec.shape, spec.size_mode, spec.edge) == (
            sg.TRIANGLE, sg.BY_EDGE, 80.0)
        assert GridSpec.from_config({}).to_dict()["cell_area_m2"] == \
            pytest.approx(10000.0)
        assert GridSpec.from_config(
            {"grid_cell_area_ha": 0.25}).cell_area_m2 == pytest.approx(2500.0)

    @pytest.mark.parametrize("spec", [
        GridSpec(shape="octagon"),
        GridSpec(size_mode="volume"),
        GridSpec(size_mode=sg.BY_EDGE, edge_m=0.5),
        GridSpec(area_m2=0.1),
        GridSpec(shape=sg.RECTANGLE, width_m=100.0, height_m=0.0),
        GridSpec(size_mode=sg.BY_EDGE, edge_m=float("nan")),
    ])
    def test_what_is_no_grid_is_refused(self, spec):
        with pytest.raises(GridError):
            spec.validate()

    def test_a_rectangle_does_not_care_about_the_size_mode(self):
        GridSpec(shape=sg.RECTANGLE, size_mode="whatever").validate()


# ---------------------------------------------------------------------------
# The cells
# ---------------------------------------------------------------------------

SPECS = [GridSpec(shape=sg.HEXAGON, area_m2=2500.0),
         GridSpec(shape=sg.SQUARE, area_m2=2500.0),
         GridSpec(shape=sg.TRIANGLE, area_m2=2500.0),
         GridSpec(shape=sg.RECTANGLE, width_m=80.0, height_m=35.0)]
IDS = [spec.shape for spec in SPECS]


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
class TestCells:
    def test_every_cell_has_the_size_that_was_asked_for(self, spec):
        grid = sg.cells(spec, BOUNDS)

        assert len(grid) > 50
        for cell in grid:
            assert sg.ring_area(cell.ring) == pytest.approx(spec.cell_area_m2)
        corners = {sg.HEXAGON: 6, sg.TRIANGLE: 3}.get(spec.shape, 4)
        assert {len(cell.ring) for cell in grid} == {corners}

    def test_the_cells_cover_the_ground_without_gaps_or_overlaps(self, spec):
        shapely = pytest.importorskip("shapely")
        from shapely.geometry import box
        from shapely.ops import unary_union

        grid = sg.cells(spec, BOUNDS)
        polygons = [_polygon(cell) for cell in grid]
        together = unary_union(polygons)

        assert together.covers(box(*BOUNDS))
        assert together.area == pytest.approx(
            sum(polygon.area for polygon in polygons), rel=1e-9)
        assert shapely.__version__

    def test_a_place_is_in_exactly_one_cell(self, spec):
        from shapely.geometry import Point

        polygons = [_polygon(cell) for cell in sg.cells(spec, BOUNDS)]
        rng = random.Random(3)
        for _ in range(200):
            point = Point(rng.uniform(BOUNDS[0], BOUNDS[2]),
                          rng.uniform(BOUNDS[1], BOUNDS[3]))
            assert sum(1 for polygon in polygons
                       if polygon.contains(point)) == 1

    def test_no_cells_are_made_far_from_the_extent(self, spec):
        from shapely.geometry import box

        extent = box(*BOUNDS)
        for cell in sg.cells(spec, BOUNDS):
            xs = [p[0] for p in cell.ring]
            ys = [p[1] for p in cell.ring]
            assert box(min(xs), min(ys), max(xs), max(ys)).intersects(extent)

    def test_the_same_ground_is_the_same_cell_whatever_is_analysed(self, spec):
        """The grid starts at the origin of the coordinate system, not at a
        corner of today's extent."""
        here = {cell.name: cell for cell in sg.cells(spec, BOUNDS)}
        shifted = (BOUNDS[0] + 137.3, BOUNDS[1] - 61.9,
                   BOUNDS[2] + 400.0, BOUNDS[3] + 13.0)
        there = {cell.name: cell for cell in sg.cells(spec, shifted)}
        shared = set(here) & set(there)

        assert len(shared) > 30
        for name in shared:
            assert here[name].ring == there[name].ring
            assert here[name].centre == there[name].centre

    def test_names_are_unique_and_cells_come_row_by_row(self, spec):
        grid = sg.cells(spec, BOUNDS)

        assert len({cell.name for cell in grid}) == len(grid)
        assert [(cell.row, cell.col) for cell in grid] == sorted(
            (cell.row, cell.col) for cell in grid)
        assert grid[0].name == f"c{grid[0].col}r{grid[0].row}"

    def test_the_middle_is_inside(self, spec):
        from shapely.geometry import Point

        for cell in sg.cells(spec, BOUNDS)[:40]:
            assert _polygon(cell).contains(Point(*cell.centre))

    def test_negative_coordinates_are_no_different(self, spec):
        grid = sg.cells(spec, (-450.0, -320.0, 380.0, 290.0))
        from shapely.ops import unary_union

        polygons = [_polygon(cell) for cell in grid]
        assert unary_union(polygons).area == pytest.approx(
            sum(polygon.area for polygon in polygons), rel=1e-9)
        assert len({cell.name for cell in grid}) == len(grid)


# ---------------------------------------------------------------------------
# Neighbours
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec,expected", [
    (SPECS[0], 6), (SPECS[1], 4), (SPECS[2], 3), (SPECS[3], 4)], ids=IDS)
def test_a_cell_in_the_middle_has_all_its_neighbours(spec, expected):
    grid = sg.cells(spec, BOUNDS)
    near = sg.neighbours(grid)
    counts = [len(near[cell.name]) for cell in grid]

    assert max(counts) == expected
    # most cells are not on the rim
    assert counts.count(expected) > len(grid) / 2
    assert min(counts) >= 1


def test_neighbours_share_an_edge_not_just_a_corner():
    grid = sg.cells(GridSpec(shape=sg.SQUARE, size_mode=sg.BY_EDGE,
                             edge_m=100.0), (0.0, 0.0, 299.0, 299.0))
    near = sg.neighbours(grid)

    assert near["c1r1"] == ["c0r1", "c1r0", "c1r2", "c2r1"]
    assert "c0r0" not in near["c1r1"]
    assert near["c0r0"] == ["c0r1", "c1r0"]


def test_neighbours_are_mutual():
    grid = sg.cells(SPECS[0], BOUNDS)
    near = sg.neighbours(grid)

    for name, others in near.items():
        for other in others:
            assert name in near[other]


def test_cells_left_out_are_nobodys_neighbours():
    grid = sg.cells(SPECS[1], BOUNDS)
    kept = grid[::2]
    near = sg.neighbours(kept)

    names = {cell.name for cell in kept}
    assert all(set(others) <= names for others in near.values())


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

def test_a_grid_of_millions_of_cells_is_refused():
    spec = GridSpec(shape=sg.SQUARE, size_mode=sg.BY_EDGE, edge_m=1.0)

    with pytest.raises(GridError, match="Choose larger cells"):
        sg.cells(spec, (0.0, 0.0, 5000.0, 5000.0))


def test_an_extent_that_is_none_is_refused():
    with pytest.raises(GridError, match="Not an extent"):
        sg.cells(GridSpec(), (10.0, 0.0, 0.0, 5.0))
    with pytest.raises(GridError, match="Not an extent"):
        sg.cells(GridSpec(), (0.0, 0.0, float("inf"), 5.0))


def test_an_extent_of_one_point_still_has_its_cell():
    grid = sg.cells(GridSpec(shape=sg.SQUARE), (150.0, 250.0, 150.0, 250.0))

    assert [cell.name for cell in grid] == ["c1r2"]


def test_square_cells_can_be_read_off_the_coordinates():
    spec = GridSpec(shape=sg.SQUARE, size_mode=sg.BY_EDGE, edge_m=100.0)

    assert sg.cell_at(spec, 150.0, 250.0) == "c1r2"
    assert sg.cell_at(spec, -1.0, -1.0) == "c-1r-1"
    assert sg.cell_at(GridSpec(shape=sg.RECTANGLE, width_m=200.0,
                               height_m=50.0), 450.0, 60.0) == "c2r1"
    assert sg.cell_at(GridSpec(), 1.0, 1.0) is None


def test_the_shoelace():
    assert sg.ring_area([(0, 0), (4, 0), (4, 3), (0, 3)]) == 12.0
    assert sg.ring_area([(0, 0), (4, 0), (0, 3)]) == 6.0
    assert math.isclose(sg.ring_area([(0, 3), (4, 0), (0, 0)]), 6.0)
