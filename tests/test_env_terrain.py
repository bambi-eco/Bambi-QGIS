# -*- coding: utf-8 -*-
"""Terrain wetness source (core/env_terrain.py)."""
import numpy as np
import pytest

from bambi_wildlife_detection.core import env_terrain as terrain
from bambi_wildlife_detection.core import environment as env


def _valley(rows=60, cols=60):
    """A V-shaped valley along the middle column, falling towards row 0...n."""
    y, x = np.mgrid[0:rows, 0:cols].astype(float)
    return np.abs(x - cols // 2) * 0.5 + (rows - y) * 0.05 + 100.0


class TestFill:
    def test_a_pit_is_raised_to_its_rim(self):
        dem = np.full((7, 7), 10.0)
        dem[3, 3] = 2.0
        filled = terrain.fill_depressions(dem, epsilon=0.0)
        assert filled[3, 3] == 10.0
        assert np.all(filled >= dem)

    def test_a_slope_is_left_alone(self):
        dem = _valley()
        assert np.allclose(terrain.fill_depressions(dem, epsilon=0.0), dem)

    def test_filled_flats_still_drain(self):
        dem = np.full((9, 9), 5.0)
        dem[4, 4] = 1.0
        filled = terrain.fill_depressions(dem)
        assert filled[4, 4] > filled[4, 0]   # rises away from the edge

    def test_no_data_acts_as_an_edge(self):
        dem = np.full((5, 5), 3.0)
        dem[2, 2] = np.nan
        filled = terrain.fill_depressions(dem)
        assert np.isnan(filled[2, 2]) and np.isfinite(filled[0, 0])


class TestAccumulation:
    def test_everything_drains_through_the_valley_outlet(self):
        dem = _valley()
        accumulation = terrain.flow_accumulation(dem)
        assert accumulation.max() == pytest.approx(dem.size)
        assert np.unravel_index(np.argmax(accumulation), dem.shape) == (59, 30)
        assert accumulation.min() == pytest.approx(1.0)

    def test_a_plane_passes_every_row_on_to_the_next(self):
        rows = np.arange(10, dtype=float)[::-1]
        dem = np.repeat(rows[:, None], 5, axis=1)   # falls towards row 9
        accumulation = terrain.flow_accumulation(dem)
        assert accumulation[0].tolist() == [1.0] * 5
        # Water spreads to the diagonal neighbours too, so a single column
        # is not exact - but nothing is lost or made between rows.
        assert accumulation.sum(axis=1) == pytest.approx(
            5.0 * np.arange(1, 11))

    def test_no_data_cells_stay_empty(self):
        dem = _valley(10, 10)
        dem[0, 0] = np.nan
        accumulation = terrain.flow_accumulation(dem)
        assert np.isnan(accumulation[0, 0])
        assert np.isfinite(accumulation[5, 5])


class TestWetness:
    def test_valley_floor_is_wetter_than_the_slopes(self):
        index = terrain.wetness_index(_valley(), 10.0)
        assert index[50, 30] > index[50, 5] + 3.0
        assert np.all(np.isfinite(index))

    def test_flatter_ground_is_wetter_for_the_same_catchment(self):
        steep = terrain.wetness_index(_valley() * 4.0, 10.0)
        gentle = terrain.wetness_index(_valley(), 10.0)
        assert gentle[30, 10] > steep[30, 10]

    def test_slope_tangent(self):
        rows = np.arange(6, dtype=float)
        dem = np.repeat(rows[:, None] * 5.0, 6, axis=1)   # 5 m per 10 m cell
        assert terrain.slope_tangent(dem, 10.0)[2, 2] == pytest.approx(0.5)


def _vertices(spacing=20.0, relief=True):
    xs, ys = np.meshgrid(np.arange(-800.0, 1200.0, spacing),
                         np.arange(-800.0, 1200.0, spacing))
    zs = (0.05 * np.abs(xs - 200.0) + 0.01 * ys) if relief \
        else np.zeros_like(xs)
    return np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])


class TestCompute:
    ORIGIN = (447000.0, 5350500.0, 300.0)
    GRID = env.make_grid((447000.0, 5350500.0, 447400.0, 5350800.0),
                         32633, 0.0)

    def test_grid_from_vertices_interpolates_and_adds_the_origin(self):
        dem = terrain.grid_from_vertices(_vertices(), self.ORIGIN, self.GRID)
        assert dem.shape == self.GRID.shape
        xs, ys = self.GRID.cell_centres()
        expected = 300.0 + 0.05 * abs(xs[3] - 447000.0 - 200.0) \
            + 0.01 * (ys[5] - 5350500.0)
        assert dem[5, 3] == pytest.approx(expected, abs=1e-6)

    def test_outside_the_mesh_is_no_data(self):
        far = env.make_grid((460000.0, 5350500.0, 460100.0, 5350600.0),
                            32633, 0.0)
        assert np.all(np.isnan(
            terrain.grid_from_vertices(_vertices(), self.ORIGIN, far)))

    def test_wetness_for_the_grid_with_metadata(self):
        layers, meta = terrain.compute(
            self.GRID, "dem.glb", self.ORIGIN,
            load_vertices=lambda path: _vertices())
        assert layers["twi"].shape == self.GRID.shape
        assert np.all(np.isfinite(layers["twi"]))
        assert meta["dem_coverage"] == 1.0 and meta["relief_m"] > 10
        # the valley line (x = origin + 200) is the wettest column
        assert int(np.argmax(layers["twi"].mean(axis=0))) in (19, 20)

    def test_a_flat_surface_is_refused(self):
        with pytest.raises(env.EnvironmentFailure, match="flat surface"):
            terrain.compute(self.GRID, "dem.glb", self.ORIGIN,
                            load_vertices=lambda path: _vertices(relief=False))

    def test_a_dem_elsewhere_is_refused(self):
        far = env.make_grid((460000.0, 5350500.0, 460100.0, 5350600.0),
                            32633, 0.0)
        with pytest.raises(env.EnvironmentFailure, match="covers only"):
            terrain.compute(far, "dem.glb", self.ORIGIN,
                            load_vertices=lambda path: _vertices())

    def test_no_mesh_or_no_origin_says_so(self):
        with pytest.raises(env.EnvironmentFailure, match="No DEM mesh"):
            terrain.compute(self.GRID, "", self.ORIGIN)
        with pytest.raises(env.EnvironmentFailure, match="origin"):
            terrain.compute(self.GRID, "dem.glb", None)

    def test_cancelling_stops_the_routing(self):
        with pytest.raises(env.Cancelled):
            terrain.compute(self.GRID, "dem.glb", self.ORIGIN,
                            load_vertices=lambda path: _vertices(),
                            cancelled=lambda: True)
