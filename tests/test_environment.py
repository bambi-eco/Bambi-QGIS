# -*- coding: utf-8 -*-
"""Shared parts of the environmental context (core/environment.py): the
extent and grid, the flight's time, the per-source metadata and the raster
round trip."""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pytest

from bambi_wildlife_detection.core import environment as env
from bambi_wildlife_detection.core import fov_store

ORIGIN = (447000.0, 5350500.0, 260.0)


def _poses(root, modality="t", count=10, stamp="2026-06-10T03:10:{:02d}"):
    images = [{"imagefile": f"{i}.jpg", "location": [i * 10.0, i * 5.0, 80.0],
               "timestamp": stamp.format(i)} for i in range(count)]
    with open(os.path.join(root, f"poses_{modality}.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"images": images}, fh)


class TestGrid:
    def test_snaps_outwards_to_whole_cells(self):
        grid = env.make_grid((1003.0, 2004.0, 1096.0, 2051.0), 32633,
                             buffer_m=0.0)
        assert grid.bounds == (1000.0, 2000.0, 1100.0, 2060.0)
        assert (grid.width, grid.height) == (10, 6)

    def test_buffer_is_added_on_every_side(self):
        grid = env.make_grid((1000.0, 2000.0, 1100.0, 2100.0), 32633,
                             buffer_m=200.0)
        assert grid.bounds == (800.0, 1800.0, 1300.0, 2300.0)

    def test_a_single_point_still_gets_a_cell(self):
        grid = env.make_grid((1005.0, 2005.0, 1005.0, 2005.0), 32633, 0.0)
        assert grid.width >= 1 and grid.height >= 1

    def test_an_absurd_extent_is_refused(self):
        with pytest.raises(env.EnvironmentFailure, match="too large"):
            env.make_grid((0.0, 0.0, 500000.0, 100.0), 32633)

    def test_a_broken_extent_is_refused(self):
        with pytest.raises(env.EnvironmentFailure):
            env.make_grid((0.0, 0.0, float("nan"), 100.0), 32633)

    def test_cell_centres_and_round_trip(self):
        grid = env.make_grid((1000.0, 2000.0, 1030.0, 2020.0), 32633, 0.0)
        xs, ys = grid.cell_centres()
        assert list(xs) == [1005.0, 1015.0, 1025.0]
        assert list(ys) == [2015.0, 2005.0]
        assert env.Grid.from_dict(grid.to_dict()) == grid

    def test_expanded_grows_on_every_side(self):
        grid = env.make_grid((1000.0, 2000.0, 1030.0, 2020.0), 32633, 0.0)
        wide = grid.expanded(2)
        assert wide.bounds == (980.0, 1980.0, 1050.0, 2040.0)


class TestExtent:
    def test_footprints_win_over_the_flight_path(self, tmp_path):
        root = str(tmp_path)
        _poses(root)
        fov_store.record_fov(root, "t", {
            0: [(500.0, 600.0, 0.0), (700.0, 600.0, 0.0), (700.0, 900.0, 0.0)]})
        bounds, how = env.flight_footprint(root, ORIGIN)
        assert bounds == (500.0, 600.0, 700.0, 900.0)
        assert "field-of-view" in how

    def test_flight_path_needs_the_dem_origin(self, tmp_path):
        root = str(tmp_path)
        _poses(root)
        assert env.flight_footprint(root, None) == (None, "")
        bounds, how = env.flight_footprint(root, ORIGIN)
        assert bounds == (447000.0, 5350500.0, 447090.0, 5350545.0)
        assert "flight path" in how

    def test_survey_area_widens_the_extent(self, tmp_path):
        root = str(tmp_path)
        _poses(root)
        grid, description = env.flight_grid(
            root, 32633, ORIGIN, buffer_m=0.0,
            area_bounds=(446000.0, 5350000.0, 446500.0, 5350100.0))
        assert grid.bounds[0] == 446000.0 and grid.bounds[2] == 447090.0
        assert "survey area" in description

    def test_no_crs_or_no_extent_says_what_to_do(self, tmp_path):
        root = str(tmp_path)
        with pytest.raises(env.EnvironmentFailure, match="target CRS"):
            env.flight_grid(root, None)
        with pytest.raises(env.EnvironmentFailure, match="Field of View"):
            env.flight_grid(root, 32633)


class TestTime:
    @pytest.mark.parametrize("value,expected", [
        ("2026-06-10T03:10:00", datetime(2026, 6, 10, 3, 10, tzinfo=timezone.utc)),
        ("2026-06-10T05:10:00+02:00", datetime(2026, 6, 10, 3, 10, tzinfo=timezone.utc)),
        ("2026-06-10T03:10:00Z", datetime(2026, 6, 10, 3, 10, tzinfo=timezone.utc)),
        (0, datetime(1970, 1, 1, tzinfo=timezone.utc)),
        ("", None), (None, None), ("yesterday", None),
    ])
    def test_parse_time(self, value, expected):
        assert env.parse_time(value) == expected

    def test_flight_time_spans_both_cameras(self, tmp_path):
        root = str(tmp_path)
        _poses(root, "t", 3, "2026-06-10T03:10:{:02d}")
        _poses(root, "w", 3, "2026-06-10T03:20:{:02d}")
        start, end = env.flight_time(root)
        assert env.iso(start) == "2026-06-10T03:10:00Z"
        assert env.iso(end) == "2026-06-10T03:20:02Z"

    def test_without_capture_times_the_date_is_unknown(self, tmp_path):
        assert env.flight_time(str(tmp_path)) is None
        with pytest.raises(env.EnvironmentFailure, match="date is unknown"):
            env.require_flight_time(str(tmp_path))


class TestMetadata:
    def test_nothing_is_done_in_an_empty_folder(self, tmp_path):
        assert env.status(str(tmp_path)) == {s: False for s in env.SOURCES}
        assert env.status("") == {s: False for s in env.SOURCES}

    def test_a_source_is_done_when_its_files_are_there(self, tmp_path):
        root = str(tmp_path)
        grid = env.make_grid((0.0, 0.0, 30.0, 20.0), 32633, 0.0)
        env.write_raster(env.layer_path(root, "ndvi"),
                         np.zeros(grid.shape, dtype=np.float32), grid)
        env.write_metadata(root, env.SENTINEL2, {"files": ["ndvi.tif"]})
        assert env.is_complete(root, env.SENTINEL2)
        assert env.read_metadata(root, env.SENTINEL2)["source"] == "sentinel2"
        os.remove(env.layer_path(root, "ndvi"))
        assert not env.is_complete(root, env.SENTINEL2)

    def test_clear_removes_only_that_source(self, tmp_path):
        root = str(tmp_path)
        grid = env.make_grid((0.0, 0.0, 30.0, 20.0), 32633, 0.0)
        for key in ("ndvi", "twi"):
            env.write_raster(env.layer_path(root, key),
                             np.zeros(grid.shape, dtype=np.float32), grid)
        env.write_metadata(root, env.SENTINEL2, {"files": ["ndvi.tif"]})
        env.write_metadata(root, env.TERRAIN, {"files": ["twi.tif"]})
        env.clear(root, env.SENTINEL2)
        assert not os.path.exists(env.layer_path(root, "ndvi"))
        assert env.read_metadata(root, env.SENTINEL2) is None
        assert env.is_complete(root, env.TERRAIN)
        assert [layer.key for layer in env.available_layers(root)] == ["twi"]

    def test_a_broken_metadata_file_is_not_done(self, tmp_path):
        root = str(tmp_path)
        env.ensure_folder(root)
        with open(env.metadata_path(root, env.WEATHER), "w") as fh:
            fh.write("{not json")
        assert env.read_metadata(root, env.WEATHER) is None
        assert not env.is_complete(root, env.WEATHER)

    def test_every_layer_belongs_to_a_source(self):
        assert {layer.source for layer in env.LAYERS} <= set(env.SOURCES)
        assert len({layer.key for layer in env.LAYERS}) == len(env.LAYERS)


class TestRasters:
    def test_float_round_trip_keeps_nan(self, tmp_path):
        grid = env.make_grid((447000.0, 5350500.0, 447050.0, 5350530.0),
                             32633, 0.0)
        values = np.arange(15, dtype=np.float32).reshape(grid.shape)
        values[0, 0] = np.nan
        path = env.write_raster(str(tmp_path / "a.tif"), values, grid, "x")
        read, read_grid = env.read_raster(path)
        assert read_grid == grid
        assert np.isnan(read[0, 0]) and read[1, 2] == values[1, 2]

    def test_classes_keep_their_type_and_colours(self, tmp_path):
        import rasterio

        grid = env.make_grid((0.0, 0.0, 30.0, 20.0), 32633, 0.0)
        classes = np.array([[10, 30, 80], [0, 10, 50]], dtype=np.uint8)
        path = env.write_raster(str(tmp_path / "c.tif"), classes, grid,
                                nodata=0, colormap={10: (0, 100, 0, 255)})
        read, _grid = env.read_raster(path)
        assert read.dtype == np.uint8 and read.tolist() == classes.tolist()
        with rasterio.open(path) as src:
            assert src.colormap(1)[10] == (0, 100, 0, 255)

    def test_a_wrong_shape_is_refused(self, tmp_path):
        grid = env.make_grid((0.0, 0.0, 30.0, 20.0), 32633, 0.0)
        with pytest.raises(ValueError):
            env.write_raster(str(tmp_path / "x.tif"), np.zeros((3, 3)), grid)

    def test_grid_in_degrees(self):
        grid = env.make_grid((447000.0, 5350500.0, 448000.0, 5351000.0),
                             32633, 0.0)
        west, south, east, north = env.grid_bounds_wgs84(grid)
        assert 14.27 < west < east < 14.31 and 48.30 < south < north < 48.32
        lat, lon = env.grid_centre_wgs84(grid)
        assert south < lat < north and west < lon < east
