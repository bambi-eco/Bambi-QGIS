# -*- coding: utf-8 -*-
"""Elevation for anywhere (core/copernicus_dem.py): which tiles a flight
needs, where they are, and how their heights are put on the flight's grid -
without a network: the tiles are stand-ins."""
import numpy as np
import pytest

from bambi_wildlife_detection.core import copernicus_dem as cd, env_remote
from bambi_wildlife_detection.core.environment import (
    EnvironmentFailure, Grid, grid_bounds_wgs84)

GLO30 = "https://copernicus-dem-30m.s3.amazonaws.com"
GLO90 = "https://copernicus-dem-90m.s3.amazonaws.com"


# ---------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------

class TestTiles:
    @pytest.mark.parametrize("lat, lon, stem", [
        (47, 13, "Copernicus_DSM_COG_10_N47_00_E013_00_DEM"),
        (-35, 18, "Copernicus_DSM_COG_10_S35_00_E018_00_DEM"),
        (30, -40, "Copernicus_DSM_COG_10_N30_00_W040_00_DEM"),
        (0, 0, "Copernicus_DSM_COG_10_N00_00_E000_00_DEM"),
        (-1, -1, "Copernicus_DSM_COG_10_S01_00_W001_00_DEM"),
        (69, 118, "Copernicus_DSM_COG_10_N69_00_E118_00_DEM"),
    ])
    def test_a_tile_is_named_by_its_south_west_corner(self, lat, lon, stem):
        assert cd.tile_stem(lat, lon) == stem

    def test_where_a_tile_is(self):
        assert cd.tile_url(47, 13) == (
            f"{GLO30}/Copernicus_DSM_COG_10_N47_00_E013_00_DEM/"
            "Copernicus_DSM_COG_10_N47_00_E013_00_DEM.tif")
        assert cd.tile_url(40, 45, edition=1) == (
            f"{GLO90}/Copernicus_DSM_COG_30_N40_00_E045_00_DEM/"
            "Copernicus_DSM_COG_30_N40_00_E045_00_DEM.tif")

    def test_a_flight_inside_a_tile_needs_that_tile(self):
        assert cd.tiles_for((14.50, 48.36, 14.52, 48.38)) == [(48, 14)]

    def test_a_flight_across_a_line_needs_both(self):
        assert cd.tiles_for((13.99, 47.89, 14.01, 47.91)) == [
            (47, 13), (47, 14)]
        assert cd.tiles_for((13.99, 47.99, 14.01, 48.01)) == [
            (47, 13), (47, 14), (48, 13), (48, 14)]

    def test_a_tiles_edge_is_half_a_cell_off_the_whole_degree(self):
        """Tile E013 ends 0.00014 degrees short of 14: a flight that
        starts just there still needs it - the neighbour is asked too."""
        assert cd.tiles_for((14.0002, 47.5, 14.01, 47.51)) == [
            (47, 13), (47, 14)]

    def test_south_and_west_of_nought(self):
        assert cd.tiles_for((-0.5, -0.5, -0.4, -0.4)) == [(-1, -1)]
        assert cd.tiles_for((18.4, -34.4, 18.5, -34.3)) == [(-35, 18)]

    def test_across_the_date_line(self):
        assert cd.tiles_for((179.995, 10.2, 180.005, 10.3)) == [
            (10, 179), (10, -180)]

    def test_the_poles_have_no_tile_beyond_them(self):
        assert cd.tiles_for((10.2, 89.9995, 10.3, 90.0)) == [(89, 10)]


# ---------------------------------------------------------------------------
# Is a tile there?
# ---------------------------------------------------------------------------

class Answer:
    def __init__(self, status):
        self.status_code = status


class TestExists:
    @pytest.mark.parametrize("status, there", [
        (200, True), (404, False), (403, False)])
    def test_a_tile_is_there_or_it_is_not(self, monkeypatch, status, there):
        import requests

        asked = []

        def head(url, **more):
            asked.append((url, more))
            return Answer(status)

        monkeypatch.setattr(requests, "head", head)

        assert cd.exists(cd.tile_url(47, 13)) is there
        assert asked[0][0] == cd.tile_url(47, 13)
        assert asked[0][1]["headers"] == {"User-Agent": "bambi-qgis-plugin"}
        assert asked[0][1]["timeout"] == env_remote.TIMEOUT_S

    def test_anything_else_is_the_services_fault(self, monkeypatch):
        import requests

        monkeypatch.setattr(requests, "head", lambda url, **more: Answer(503))

        with pytest.raises(env_remote.SourceUnavailable, match="answered 503"):
            cd.exists(cd.tile_url(47, 13))

    def test_no_connection_says_so(self, monkeypatch):
        import requests

        def head(url, **more):
            raise requests.ConnectionError("no route")

        monkeypatch.setattr(requests, "head", head)

        with pytest.raises(env_remote.SourceUnavailable,
                           match="could not be reached"):
            cd.exists(cd.tile_url(47, 13))

    def test_only_https(self):
        with pytest.raises(env_remote.SourceUnavailable):
            cd.exists("http://copernicus-dem-30m.s3.amazonaws.com/x.tif")


# ---------------------------------------------------------------------------
# The heights of a flight
# ---------------------------------------------------------------------------

def _grid(west_x=499700.0, north_y=5306000.0, width=20, height=10):
    """A grid in UTM 33N; x = 500 000 is the meridian of 15 degrees east,
    which the grid lies across unless told otherwise."""
    return Grid(32633, west_x, north_y, width, height, 30.0)


class Tiles:
    """Stand-ins for the bucket: which tiles there are, and what each
    gives - a constant height on the columns it covers."""

    def __init__(self, there, covers=None):
        self.there = set(there)             # urls
        self.covers = covers or {}          # url -> (first column, last + 1)
        self.read = []
        self.asked = []

    def exists(self, url):
        self.asked.append(url)
        return url in self.there

    def __call__(self, url, grid, resampling, fill=None):
        self.read.append((url, resampling))
        assert fill is not None and np.isnan(fill)
        array = np.full(grid.shape, np.nan, dtype=np.float32)
        first, last = self.covers.get(url, (0, grid.width))
        array[:, first:last] = 100.0 * (len(self.read))
        return array, fill


class TestFetch:
    def test_the_heights_of_a_flight_inside_one_tile(self):
        grid = _grid(west_x=520000.0)
        url = cd.tile_url(47, 15)
        tiles = Tiles([url])
        log = []

        heights, metadata = cd.fetch(grid, read=tiles, there=tiles.exists,
                                     log=log.append)

        assert heights.shape == grid.shape and heights.dtype == np.float32
        assert np.all(heights == 100.0)
        assert tiles.read == [(url, "bilinear")]
        assert metadata["product"] == "Copernicus DEM GLO-30"
        assert metadata["tiles"] == [cd.tile_stem(47, 15)]
        assert (metadata["sea_share"], metadata["min_m"],
                metadata["max_m"]) == (0.0, 100.0, 100.0)
        assert metadata["surface_model"] is True
        assert metadata["vertical_datum"] == "EGM2008"
        assert "COPERNICUS" in metadata["attribution"]
        assert log == [
            f"Reading Copernicus DEM tile {cd.tile_stem(47, 15)} (GLO-30)",
            "Copernicus DEM GLO-30: 100 to 100 m above sea level"]

    def test_a_flight_across_a_line_is_put_together(self):
        grid = _grid()                    # 600 m across the 15th meridian
        west, east = cd.tile_url(47, 14), cd.tile_url(47, 15)
        tiles = Tiles([west, east], {west: (0, 10), east: (10, 20)})

        heights, metadata = cd.fetch(grid, read=tiles, there=tiles.exists)

        assert np.all(heights[:, :10] == 100.0)
        assert np.all(heights[:, 10:] == 200.0)
        assert metadata["tiles"] == [cd.tile_stem(47, 14),
                                     cd.tile_stem(47, 15)]

    def test_where_tiles_overlap_the_first_one_stands(self):
        grid = _grid()
        west, east = cd.tile_url(47, 14), cd.tile_url(47, 15)
        tiles = Tiles([west, east], {west: (0, 12), east: (8, 20)})

        heights, _metadata = cd.fetch(grid, read=tiles, there=tiles.exists)

        assert np.all(heights[:, :12] == 100.0)
        assert np.all(heights[:, 12:] == 200.0)

    def test_a_tile_the_fine_edition_lacks_comes_from_the_coarse_one(self):
        grid = _grid(west_x=520000.0)
        coarse = cd.tile_url(47, 15, edition=1)
        tiles = Tiles([coarse])
        log = []

        heights, metadata = cd.fetch(grid, read=tiles, there=tiles.exists,
                                     log=log.append)

        assert tiles.asked == [cd.tile_url(47, 15), coarse]
        assert tiles.read == [(coarse, "bilinear")]
        assert metadata["product"] == "Copernicus DEM GLO-90"
        assert metadata["tiles"] == [cd.tile_stem(47, 15, 30)]
        assert log[0].endswith("Copernicus_DSM_COG_30_N47_00_E015_00_DEM "
                               "(GLO-90)")
        assert log[-1] == ("GLO-30 has no tile here: the 90 m edition is "
                           "used, which is coarser")

    def test_the_fine_edition_is_not_looked_past(self):
        grid = _grid(west_x=520000.0)
        tiles = Tiles([cd.tile_url(47, 15), cd.tile_url(47, 15, edition=1)])

        cd.fetch(grid, read=tiles, there=tiles.exists)

        assert tiles.asked == [cd.tile_url(47, 15)]

    def test_both_editions_side_by_side(self):
        grid = _grid()
        west, east = cd.tile_url(47, 14), cd.tile_url(47, 15, edition=1)
        tiles = Tiles([west, east], {west: (0, 10), east: (10, 20)})

        _heights, metadata = cd.fetch(grid, read=tiles, there=tiles.exists)

        assert metadata["product"] == "Copernicus DEM GLO-30 + GLO-90"
        assert metadata["editions"] == ["GLO-30", "GLO-90"]

    def test_where_the_land_ends_the_sea_is_at_nought(self):
        grid = _grid()
        west = cd.tile_url(47, 14)
        tiles = Tiles([west], {west: (0, 10)})        # no tile to the east
        log = []

        heights, metadata = cd.fetch(grid, read=tiles, there=tiles.exists,
                                     log=log.append)

        assert np.all(heights[:, :10] == 100.0)
        assert np.all(heights[:, 10:] == 0.0)
        assert not np.any(np.isnan(heights))
        assert metadata["sea_share"] == 0.5
        assert (metadata["min_m"], metadata["max_m"]) == (0.0, 100.0)
        assert log[-1] == (
            "Copernicus DEM GLO-30: 0 to 100 m above sea level, 50 % of it "
            "without data - taken as sea, at height nought")

    def test_open_sea_has_nothing_to_build(self):
        tiles = Tiles([])

        with pytest.raises(cd.NoElevation, match="open sea"):
            cd.fetch(_grid(), read=tiles, there=tiles.exists)

        assert tiles.read == []
        assert issubclass(cd.NoElevation, EnvironmentFailure)

    def test_a_tile_that_is_there_and_holds_none_of_the_grid(self):
        """The neighbour that was asked to be sure: it gives nothing, and
        is not listed."""
        grid = _grid()
        west, east = cd.tile_url(47, 14), cd.tile_url(47, 15)
        tiles = Tiles([west, east], {west: (0, 20), east: (0, 0)})

        _heights, metadata = cd.fetch(grid, read=tiles, there=tiles.exists)

        assert metadata["tiles"] == [cd.tile_stem(47, 14)]

    def test_it_can_be_cancelled_between_tiles(self):
        grid = _grid()
        tiles = Tiles([cd.tile_url(47, 14), cd.tile_url(47, 15)])
        calls = []

        def cancelled():
            calls.append(1)
            return len(calls) > 1

        from bambi_wildlife_detection.core.environment import Cancelled

        with pytest.raises(Cancelled):
            cd.fetch(grid, read=tiles, there=tiles.exists,
                     cancelled=cancelled)

        assert len(tiles.read) == 1

    def test_progress_tile_by_tile(self):
        grid = _grid()
        tiles = Tiles([cd.tile_url(47, 14), cd.tile_url(47, 15)])
        seen = []

        cd.fetch(grid, read=tiles, there=tiles.exists, progress=seen.append)

        assert seen == [45, 90]


class TestGrid:
    def test_the_ground_around_a_flight_is_kept(self):
        grid = cd.grid_for((500010.0, 5300010.0, 500590.0, 5300290.0), 32633)

        assert grid.resolution == 30.0 and grid.epsg == 32633
        west, south, east, north = grid.bounds
        # at least 150 m around, on whole cells
        assert west <= 500010.0 - 150 and east >= 500590.0 + 150
        assert south <= 5300010.0 - 150 and north >= 5300290.0 + 150
        assert west % 30 == 0 and north % 30 == 0

    def test_more_can_be_asked_for_not_less(self):
        bounds = (500000.0, 5300000.0, 500300.0, 5300300.0)

        # 300 m of flight and 600 m either side, on whole cells of 30 m
        assert cd.grid_for(bounds, 32633, 600.0).width == 51
        assert cd.grid_for(bounds, 32633, 10.0).width == \
            cd.grid_for(bounds, 32633).width == 21

    def test_the_tiles_of_a_grid(self):
        west, south, east, north = grid_bounds_wgs84(_grid())

        assert west < 15.0 < east
        assert cd.tiles_for((west, south, east, north)) == [
            (47, 14), (47, 15)]


# ---------------------------------------------------------------------------
# A raster that names no no-data value
# ---------------------------------------------------------------------------

class TestFill:
    @pytest.fixture
    def heights_tif(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_origin

        path = str(tmp_path / "heights.tif")
        data = np.full((40, 40), 250.0, dtype=np.float32)
        data[:, :20] = 0.0                       # a lake, at height nought
        with rasterio.open(
                path, "w", driver="GTiff", height=40, width=40, count=1,
                dtype="float32", crs="EPSG:32633",
                transform=from_origin(500000.0, 5301200.0, 30.0, 30.0)) as dst:
            dst.write(data, 1)
        return path

    def test_what_a_source_does_not_cover_is_marked(self, heights_tif):
        # the grid reaches 300 m beyond the raster to the east
        grid = Grid(32633, 500900.0, 5301200.0, 20, 10, 30.0)

        array, fill = env_remote.read_to_grid(heights_tif, grid, "bilinear",
                                              fill=float("nan"))

        assert np.isnan(fill)
        assert np.all(array[:, :10] == 250.0)
        assert np.all(np.isnan(array[:, 10:]))

    def test_nought_is_a_height_like_any_other(self, heights_tif):
        grid = Grid(32633, 500000.0, 5301200.0, 10, 10, 30.0)

        array, _fill = env_remote.read_to_grid(heights_tif, grid, "bilinear",
                                               fill=float("nan"))

        assert np.all(array == 0.0) and not np.any(np.isnan(array))

    def test_without_a_fill_it_is_as_it_was(self, heights_tif):
        grid = Grid(32633, 500900.0, 5301200.0, 20, 10, 30.0)

        array, nodata = env_remote.read_to_grid(heights_tif, grid)

        assert nodata is None
        assert np.all(array[:, 10:] == 0.0)

    def test_the_heights_of_a_real_file(self, heights_tif):
        """The whole way with a file in place of the bucket."""
        grid = Grid(32633, 500900.0, 5301200.0, 20, 10, 30.0)

        heights, metadata = cd.fetch(
            grid, read=lambda url, grid, how, fill=None:
            env_remote.read_to_grid(heights_tif, grid, how, fill=fill),
            there=lambda url: url == cd.tile_url(47, 15))

        assert np.all(heights[:, :10] == 250.0)
        assert np.all(heights[:, 10:] == 0.0)
        assert metadata["sea_share"] == 0.5
