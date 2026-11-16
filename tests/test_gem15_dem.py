# -*- coding: utf-8 -*-
"""The bare ground from Google's GEM-15 (core/gem15_dem.py): where a cell's
file is, how its regional offset is taken out, and how its gaps are closed
- with the cells handed in, so that no network is needed."""
import numpy as np
import pytest

from bambi_wildlife_detection.core import (
    copernicus_dem, env_remote, environment, gem15_dem)
from bambi_wildlife_detection.core.environment import make_grid

pytest.importorskip("pyproj")
pytest.importorskip("scipy")

#: Around Hagenberg, Upper Austria, in UTM 33 north: one cell, w14s48.
HAGENBERG = (450000.0, 5357000.0, 451500.0, 5358500.0)
#: Across the 15th meridian: the cells w14s48 and w15s48.
TWO_CELLS = (499000.0, 5357000.0, 501000.0, 5358000.0)

COPERNICUS = {"product": "Copernicus DEM GLO-30", "sea_share": 0.0,
              "attribution": copernicus_dem.ATTRIBUTION,
              "vertical_datum": "EGM2008"}


def grid_of(bounds=HAGENBERG):
    return gem15_dem.grid_for(bounds, 32633)


def cells(terrain, surface=None, missing=(), seen=None):
    """``(read, there)`` over made-up cells: *terrain* and *surface* are
    ``f(grid, cell) -> array``; cells named in *missing* do not exist."""
    def name(url):
        return url.split("/")[-4]

    def there(url):
        return name(url) not in missing

    def read(url, grid, resampling, fill=None):
        if seen is not None:
            seen.append((name(url), url.split("/")[-3], resampling, fill))
        make = terrain if "/dtm/" in url else (surface or terrain)
        return np.asarray(make(grid, name(url)), dtype=np.float32), fill

    return read, there


def flat(height):
    return lambda grid, cell: np.full(grid.shape, height, dtype=np.float32)


def reference_of(make=flat(503.0), meta=None, seen=None):
    def reference(grid, log=None, cancelled=None):
        if seen is not None:
            seen.append(grid)
        return make(grid, "").astype(np.float32), dict(meta or COPERNICUS)
    return reference


def no_reference(grid, log=None, cancelled=None):
    raise copernicus_dem.NoElevation("nothing here")


# ---------------------------------------------------------------------------
# Where a cell's file is
# ---------------------------------------------------------------------------

class TestNames:
    """Checked against the published catalogue: 454 cells of nine
    ten-degree squares, among them these."""

    @pytest.mark.parametrize("lat, lon, group, cell", [
        (47, 14, "p40_p10", "w14s47"),
        (-1, -91, "m10_m100", "wm91sm1"),
        (4, -10, "p0_m10", "wm10s4"),
        (-2, 5, "m10_p0", "w5sm2"),
        (5, 0, "p0_p0", "w0s5"),
        (80, -81, "p80_m90", "wm81s80"),
        (-52, -61, "m60_m70", "wm61sm52"),
        (-1, 174, "m10_p170", "w174sm1"),
        (60, 20, "p60_p20", "w20s60"),
    ])
    def test_the_square_and_the_cell(self, lat, lon, group, cell):
        assert gem15_dem.group_id(lat, lon) == group
        assert gem15_dem.cell_id(lat, lon) == cell

    def test_where_the_terrain_and_the_surface_are(self):
        assert gem15_dem.tile_url(47, 14) == (
            "https://storage.googleapis.com/climate-and-sustainability/"
            "gem-15/p40_p10/w14s47/dtm/zall/mosaic.tif")
        assert gem15_dem.tile_url(47, 14, gem15_dem.SURFACE).endswith(
            "/p40_p10/w14s47/dsm/zall/mosaic.tif")

    def test_a_cell_that_is_not_there(self, monkeypatch):
        asked = []
        monkeypatch.setattr(
            env_remote, "exists",
            lambda url, what: asked.append((url, what)) or False)

        assert gem15_dem.exists(gem15_dem.tile_url(55, 100)) is False
        assert asked[0][1] == "Google's GEM-15 elevation model"

    def test_the_grid_has_cells_of_15_m_and_ground_around_the_flight(self):
        grid = grid_of()

        assert (grid.epsg, grid.resolution) == (32633, 15.0)
        west, south, east, north = grid.bounds
        assert 150 <= HAGENBERG[0] - west < 165
        assert 150 <= north - HAGENBERG[3] < 165
        wide = gem15_dem.grid_for(HAGENBERG, 32633, padding_m=600.0)
        assert wide.width == grid.width + 60


# ---------------------------------------------------------------------------
# Taking the regional offset out
# ---------------------------------------------------------------------------

class TestLevelling:
    def _scene(self, offset=-4.0, canopy=20.0, open_share=0.5, shape=(40, 40)):
        """Ground at 500 m. GEM-15 lies *offset* off it; one part is open,
        the other has trees of *canopy* m on it, which the reference - a
        surface model - sees half of."""
        rows, cols = shape
        trees = np.zeros(shape, dtype=bool)
        trees[:, int(cols * open_share):] = True
        terrain = np.full(shape, 500.0 + offset)
        surface = terrain + np.where(trees, canopy, 0.3)
        reference = 500.0 + np.where(trees, canopy / 2.0, 0.2)
        return terrain, surface, reference, trees

    def test_the_offset_is_read_on_open_ground(self):
        terrain, surface, reference, trees = self._scene(offset=-4.0)

        offset, cells = gem15_dem.level_offset(terrain, surface, reference)

        # the reference against GEM-15's surface, where nothing stands
        assert offset == pytest.approx(3.9)
        assert cells == int((~trees).sum())

    def test_what_stands_on_the_ground_does_not_enter(self):
        """Over the whole grid the reference is 7 m above GEM-15's terrain
        on average: the trees, not the offset."""
        terrain, surface, reference, _trees = self._scene(
            offset=-2.0, canopy=30.0, open_share=0.2)

        offset, _cells = gem15_dem.level_offset(terrain, surface, reference)

        assert offset == pytest.approx(1.9)
        assert float(np.median(reference - terrain)) > 10.0

    def test_too_little_open_ground_to_tell(self):
        terrain, surface, reference, _trees = self._scene(
            open_share=0.025, shape=(40, 40))      # one column: 40 cells

        offset, cells = gem15_dem.level_offset(terrain, surface, reference)

        assert offset is None and cells == 40

    def test_cells_without_a_height_do_not_count(self):
        terrain, surface, reference, trees = self._scene()
        terrain[:10] = np.nan
        reference[10:20] = np.nan

        _offset, cells = gem15_dem.level_offset(terrain, surface, reference)

        assert cells == int((~trees)[20:].sum())

    def test_an_offset_beyond_belief_is_not_applied(self):
        terrain, surface, reference, _trees = self._scene()

        offset, cells = gem15_dem.level_offset(terrain, surface,
                                               reference + 100.0)

        assert offset is None and cells > 0

    def test_nothing_at_all(self):
        nothing = np.full((5, 5), np.nan)

        assert gem15_dem.level_offset(nothing, nothing, nothing) == (None, 0)


# ---------------------------------------------------------------------------
# Reading the heights under a grid
# ---------------------------------------------------------------------------

class TestFetch:
    def test_the_terrain_levelled_to_the_reference(self):
        grid, seen, log = grid_of(), [], []
        read, there = cells(flat(500.0), flat(500.4), seen=seen)

        heights, meta = gem15_dem.fetch(
            grid, read, there, reference_of(flat(503.0)), log=log.append)

        assert heights.shape == grid.shape and heights.dtype == np.float32
        # 503.0 against a surface of 500.4: GEM-15 lies 2.6 m low
        assert np.allclose(heights, 502.6)
        assert meta["levelled_by_m"] == pytest.approx(2.6, abs=1e-4)
        assert meta["open_ground_cells"] == grid.width * grid.height
        assert meta["product"] == ("Google GEM-15 terrain model, levelled "
                                   "to the Copernicus DEM")
        assert meta["surface_model"] is False
        assert meta["resolution_m"] == 15.0
        assert meta["vertical_datum"] == "EGM2008"
        assert meta["tiles"] == ["w14s48"]
        assert (meta["filled_share"], meta["filled_from"]) == (0.0, None)
        assert meta["attribution"].startswith(
            "GEM-15: This dataset is produced by Google (CC BY 4.0).")
        assert meta["attribution"].endswith(copernicus_dem.ATTRIBUTION)
        assert seen == [("w14s48", "dtm", "bilinear", pytest.approx(
            float("nan"), nan_ok=True)), ("w14s48", "dsm", "bilinear",
                                          pytest.approx(float("nan"),
                                                        nan_ok=True))]
        assert log == [
            "Reading GEM-15 cell w14s48 (dtm)",
            "Google GEM-15 terrain model: 503 to 503 m above sea level",
            f"Levelled to the Copernicus DEM: +2.6 m, from "
            f"{grid.width * grid.height} cells of open ground (GEM-15 is "
            "known to lie a few metres off, region by region)"]

    def test_gaps_are_closed_from_the_reference(self):
        """Water is left without a value, with a margin around it."""
        grid, log = grid_of(), []

        def with_a_lake(grid, cell):
            heights = np.full(grid.shape, 500.0, dtype=np.float32)
            heights[:20, :30] = np.nan
            return heights

        def reference(grid, cell):
            heights = np.full(grid.shape, 501.0, dtype=np.float32)
            heights[:20, :30] = 480.0                 # the lake's surface
            return heights

        read, there = cells(with_a_lake)
        heights, meta = gem15_dem.fetch(
            grid, read, there, reference_of(reference), log=log.append)

        assert np.isfinite(heights).all()
        assert np.allclose(heights[:20, :30], 480.0)
        assert np.allclose(heights[20:], 501.0)       # levelled by +1.0
        assert meta["filled_share"] == pytest.approx(
            600 / (grid.width * grid.height), abs=1e-4)
        assert meta["filled_from"] == "Copernicus DEM GLO-30"
        assert log[-1].endswith(
            "% of the cells have no GEM-15 height (water, snow, surfaces "
            "without texture): taken from Copernicus DEM GLO-30")

    def test_cells_are_put_together_across_a_meridian(self):
        grid, seen = grid_of(TWO_CELLS), []

        def half(grid, cell):
            """Each cell holds its own side of the 15th meridian."""
            heights = np.full(grid.shape, np.nan, dtype=np.float32)
            centres = grid.min_x + (np.arange(grid.width) + 0.5) * 15.0
            mine = centres < 500000.0 if cell == "w14s48" \
                else centres >= 500000.0
            heights[:, mine] = 400.0 if cell == "w14s48" else 410.0
            return heights

        read, there = cells(half, seen=seen)
        heights, meta = gem15_dem.fetch(grid, read, there, no_reference)

        assert meta["tiles"] == ["w14s48", "w15s48"]
        assert sorted({float(v) for v in np.unique(heights)}) == [400.0,
                                                                  410.0]
        assert [(cell, layer) for cell, layer, *_ in seen] == [
            ("w14s48", "dtm"), ("w15s48", "dtm"),
            ("w14s48", "dsm"), ("w15s48", "dsm")]

    def test_a_cell_that_does_not_exist_is_not_read(self):
        grid, seen = grid_of(TWO_CELLS), []
        read, there = cells(flat(400.0), missing=("w15s48",), seen=seen)

        _heights, meta = gem15_dem.fetch(grid, read, there, no_reference)

        assert meta["tiles"] == ["w14s48"]
        assert {cell for cell, *_ in seen} == {"w14s48"}

    def test_nothing_there_is_said_so_that_the_caller_can_look_elsewhere(
            self):
        read, there = cells(flat(400.0), missing=("w14s48",))

        with pytest.raises(gem15_dem.NoTerrain) as raised:
            gem15_dem.fetch(grid_of(), read, there, reference_of())

        assert "Russia, China, Mongolia, Greenland" in str(raised.value)
        # it is "no elevation" to whoever asks for no more than that
        assert isinstance(raised.value, copernicus_dem.NoElevation)

    def test_a_cell_that_holds_nothing_of_the_grid_is_nothing(self):
        """The file is there; the flight lies in its masked part."""
        read, there = cells(flat(float("nan")))

        with pytest.raises(gem15_dem.NoTerrain):
            gem15_dem.fetch(grid_of(), read, there, reference_of())

    def test_without_a_reference_the_heights_are_as_published(self):
        grid, log = grid_of(), []

        def with_a_gap(grid, cell):
            heights = np.full(grid.shape, 500.0, dtype=np.float32)
            heights[0, 0] = np.nan
            heights[5:8, 5:8] = np.nan
            return heights

        read, there = cells(with_a_gap)
        heights, meta = gem15_dem.fetch(grid, read, there, no_reference,
                                        log=log.append)

        assert np.allclose(heights, 500.0)            # the nearest height
        assert meta["levelled_by_m"] is None
        assert meta["vertical_datum"] == "EGM96"
        assert meta["product"] == "Google GEM-15 terrain model"
        assert meta["filled_from"] == "the nearest cells with a height"
        assert meta["attribution"] == gem15_dem.ATTRIBUTION
        assert ("The Copernicus DEM has nothing here to level GEM-15 by: "
                "its heights are taken as published") in log

    def test_under_closed_forest_nothing_is_levelled_and_it_is_said(self):
        grid, log = grid_of(), []
        read, there = cells(flat(500.0), flat(525.0))      # trees everywhere

        heights, meta = gem15_dem.fetch(
            grid, read, there, reference_of(flat(515.0)), log=log.append)

        assert np.allclose(heights, 500.0)
        assert (meta["levelled_by_m"], meta["open_ground_cells"]) == (None,
                                                                      0)
        assert meta["vertical_datum"] == "EGM96"
        assert meta["attribution"] == gem15_dem.ATTRIBUTION
        assert log[-1] == (
            "Not levelled: too little open ground under the flight to tell "
            "GEM-15's offset from the Copernicus DEM (0 cells). Its heights "
            "are as published, and may lie a few metres low or high.")

    def test_the_sea_of_the_reference_is_passed_on(self):
        read, there = cells(flat(3.0))

        _heights, meta = gem15_dem.fetch(
            grid_of(), read, there, reference_of(
                flat(3.0), dict(COPERNICUS, sea_share=0.25)))

        assert meta["sea_share"] == 0.25

    def test_progress_runs_through_both_layers_and_the_reference(self):
        told = []
        read, there = cells(flat(400.0))

        gem15_dem.fetch(grid_of(TWO_CELLS), read, there, reference_of(),
                        progress=told.append)

        assert told == [18, 36, 54, 72, 90]

    def test_it_can_be_cancelled_between_cells(self):
        read, there = cells(flat(400.0))
        asked = []

        def cancelled():
            asked.append(1)
            return len(asked) > 2

        with pytest.raises(environment.Cancelled):
            gem15_dem.fetch(grid_of(TWO_CELLS), read, there, reference_of(),
                            cancelled=cancelled)

    def test_the_reference_is_asked_for_the_same_grid(self):
        grid, grids = grid_of(), []
        read, there = cells(flat(400.0))

        gem15_dem.fetch(grid, read, there, reference_of(seen=grids))

        assert grids == [grid]


def test_gaps_take_the_height_of_the_nearest_cell():
    heights = np.array([[1.0, np.nan, np.nan, 9.0, 9.0]], dtype=np.float32)

    filled = gem15_dem._nearest_filled(heights)

    assert filled.tolist() == [[1.0, 1.0, 9.0, 9.0, 9.0]]
    nothing = np.full((2, 2), np.nan, dtype=np.float32)
    assert np.isnan(gem15_dem._nearest_filled(nothing)).all()


def test_the_grid_is_a_plain_one():
    assert grid_of() == make_grid(HAGENBERG, 32633, 150.0, 15.0)
