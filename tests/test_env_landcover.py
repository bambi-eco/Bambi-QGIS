# -*- coding: utf-8 -*-
"""Land-cover source (core/env_landcover.py)."""
import numpy as np
import pytest

from bambi_wildlife_detection.core import env_landcover as lc
from bambi_wildlife_detection.core import environment as env

GRID = env.make_grid((447000.0, 5350500.0, 447200.0, 5350700.0), 32633, 0.0)


class TestTiles:
    @pytest.mark.parametrize("lat,lon,name", [
        (48.3, 14.3, "N48E012"), (47.9, 11.9, "N45E009"),
        (-0.5, -0.5, "S03W003"), (0.5, 0.5, "N00E000"),
        (-33.9, 151.2, "S36E150"),
    ])
    def test_tile_name(self, lat, lon, name):
        assert lc.tile_name(lat, lon) == name

    def test_a_box_across_a_corner_touches_four_tiles(self):
        assert lc.tiles_for((11.9, 47.9, 12.1, 48.1)) == [
            "N45E009", "N45E012", "N48E009", "N48E012"]
        assert lc.tiles_for((14.2, 48.2, 14.3, 48.3)) == ["N48E012"]

    def test_tile_url(self):
        assert lc.tile_url("N48E012") == (
            "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/"
            "map/ESA_WorldCover_10m_2021_v200_N48E012_Map.tif")


class TestTreeCover:
    def test_focal_fraction_counts_only_cells_inside(self):
        mask = np.zeros((5, 5))
        mask[0, 0] = 1
        share = lc.focal_fraction(mask, 1)
        assert share[0, 0] == pytest.approx(1 / 4)   # corner: 4 cells
        assert share[0, 1] == pytest.approx(1 / 6)   # edge: 6 cells
        assert share[1, 1] == pytest.approx(1 / 9)
        assert share[3, 3] == 0.0
        assert lc.focal_fraction(mask, 0).tolist() == mask.tolist()

    def test_forest_edge_grades_from_full_to_none(self):
        classes = np.full((20, 40), 30, dtype=np.uint8)
        classes[:, :20] = lc.TREE_CLASS
        cover = lc.tree_cover(classes, 10.0, radius_m=50.0)
        assert cover[10, 2] == pytest.approx(100.0)
        assert cover[10, 37] == pytest.approx(0.0)
        assert 40.0 < cover[10, 20] < 60.0

    def test_unmapped_cells_have_no_tree_cover(self):
        classes = np.full((6, 6), lc.TREE_CLASS, dtype=np.uint8)
        classes[0, 0] = lc.NO_DATA
        cover = lc.tree_cover(classes, 10.0, radius_m=10.0)
        assert np.isnan(cover[0, 0])
        assert cover[1, 1] == pytest.approx(100.0)  # the gap is not "no tree"


class TestFetch:
    def test_classes_and_tree_cover_on_the_grid(self):
        tile = np.full(GRID.shape, 30, dtype=np.uint8)
        tile[:, :5] = 10
        urls = []

        def read(href, grid, resampling):
            urls.append((href, resampling))
            return tile, 0

        layers, meta = lc.fetch(GRID, read=read)
        assert urls == [(lc.tile_url("N48E012"), "nearest")]
        assert layers["landcover"].dtype == np.uint8
        assert layers["landcover"].tolist() == tile.tolist()
        assert layers["tree_cover"].shape == GRID.shape
        assert meta["composition"] == {"Grassland": 0.75, "Tree cover": 0.25}
        assert meta["tiles"] == ["N48E012"]

    def test_a_second_tile_fills_only_the_gaps(self, monkeypatch):
        first = np.zeros(GRID.shape, dtype=np.uint8)
        first[:, :10] = 40
        second = np.full(GRID.shape, 80, dtype=np.uint8)
        monkeypatch.setattr(lc, "tiles_for", lambda bounds: ["A", "B"])
        tiles = iter([first, second])
        layers, _meta = lc.fetch(GRID, read=lambda *a: (next(tiles), 0))
        assert layers["landcover"][0, 0] == 40
        assert layers["landcover"][0, 15] == 80

    def test_open_sea_fails(self):
        with pytest.raises(env.EnvironmentFailure, match="no data"):
            lc.fetch(GRID, read=lambda *a: (
                np.zeros(GRID.shape, dtype=np.uint8), 0))

    def test_names_and_colours(self):
        assert lc.class_name(10) == "Tree cover"
        assert lc.class_name(0) == "No data"
        assert lc.class_name(33) == "Class 33"
        table = lc.colormap()
        assert table[80] == (0, 100, 200, 255) and table[0][3] == 0
