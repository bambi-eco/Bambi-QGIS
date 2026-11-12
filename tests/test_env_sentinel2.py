# -*- coding: utf-8 -*-
"""Sentinel-2 source (core/env_sentinel2.py): reading the catalogue, picking
the scene that is clear over the flight, and the indices."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from bambi_wildlife_detection.core import env_sentinel2 as s2
from bambi_wildlife_detection.core import environment as env

FLIGHT = datetime(2026, 6, 10, 3, 15, tzinfo=timezone.utc)
GRID = env.make_grid((447000.0, 5350500.0, 447040.0, 5350530.0), 32633, 0.0)


def _item(scene_id, when, cloud=10.0, offset_applied=True, drop=None):
    assets = {}
    for key in ("green", "red", "nir", "nir08", "swir16"):
        assets[key] = {"href": f"https://x/{scene_id}/{key}.tif",
                       "raster:bands": [{"scale": 0.0001, "offset": -0.1}]}
    assets["scl"] = {"href": f"https://x/{scene_id}/scl.tif"}
    if drop:
        del assets[drop]
    return {"id": scene_id, "assets": assets, "properties": {
        "datetime": when, "eo:cloud_cover": cloud,
        "platform": "sentinel-2a", "grid:code": "MGRS-33UVP",
        "earthsearch:boa_offset_applied": offset_applied}}


def _scene(scene_id, days, cloud=10.0):
    when = env.iso(FLIGHT + timedelta(days=days))
    return s2._scene_from_item(_item(scene_id, when, cloud), s2.EARTH_SEARCH)


class TestCatalogue:
    def test_offset_already_applied_is_not_applied_again(self):
        scene = s2._scene_from_item(
            _item("a", "2026-06-08T10:17:35Z"), s2.EARTH_SEARCH)
        assert scene.assets["red"] == ("https://x/a/red.tif", 0.0001, 0.0)
        assert scene.assets["scl"][0].endswith("scl.tif")
        assert scene.tile == "MGRS-33UVP"

    def test_offset_still_in_the_pixels_is_applied(self):
        scene = s2._scene_from_item(
            _item("a", "2026-06-08T10:17:35Z", offset_applied=False),
            s2.EARTH_SEARCH)
        assert scene.assets["nir"][2] == -0.1

    def test_an_item_missing_a_band_is_skipped(self):
        assert s2._scene_from_item(
            _item("a", "2026-06-08T10:17:35Z", drop="swir16"),
            s2.EARTH_SEARCH) is None

    def test_search_follows_the_next_link(self):
        pages = [
            {"features": [_item("a", "2026-06-08T10:00:00Z")],
             "links": [{"rel": "next", "href": "https://x/search",
                        "body": {"next": "tok"}, "merge": True}]},
            {"features": [_item("b", "2026-06-11T10:00:00Z")], "links": []},
        ]
        bodies = []

        def fetch_json(url, body=None, what=""):
            bodies.append(body)
            return pages[len(bodies) - 1]

        scenes = s2.search_scenes((14.2, 48.2, 14.3, 48.3), FLIGHT, FLIGHT,
                                  fetch_json=fetch_json)
        assert [scene.scene_id for scene in scenes] == ["a", "b"]
        assert bodies[0]["collections"] == ["sentinel-2-l2a"]
        assert bodies[1]["next"] == "tok" and bodies[1]["bbox"]

    def test_nearest_first_then_less_cloud(self):
        scenes = [_scene("late", 3), _scene("before", -1, 60.0),
                  _scene("after", 1, 20.0), _scene("far", -6)]
        ranked = s2.rank(scenes, FLIGHT)
        assert [s.scene_id for s in ranked] == ["after", "before", "late", "far"]
        assert ranked[1].offset_days(FLIGHT) == pytest.approx(-1.0)


class TestIndices:
    def test_usable_classes(self):
        scl = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
        assert s2.usable_mask(scl).tolist() == [
            [False, False, True, False], [True, True, True, True],
            [False, False, False, True]]
        assert s2.clear_fraction(scl) == pytest.approx(6 / 12)
        assert s2.clear_fraction(np.zeros((0, 0))) == 0.0
        assert s2.scl_composition(np.array([[4, 4, 9]]))["vegetation"] == \
            pytest.approx(0.6667)

    def test_reflectance_scales_and_masks_no_data(self):
        values = s2.reflectance(np.array([[0, 1000, 50]], dtype=np.uint16),
                                0.0001, -0.1)
        assert np.isnan(values[0, 0])
        assert values[0, 1] == pytest.approx(0.0)
        assert values[0, 2] == 0.0  # clipped, not negative

    def test_normalised_difference(self):
        a = np.array([0.5, 0.0, np.nan], dtype=np.float32)
        b = np.array([0.1, 0.0, 0.2], dtype=np.float32)
        result = s2.normalised_difference(a, b)
        assert result[0] == pytest.approx(0.6667, abs=1e-3)
        assert np.isnan(result[1]) and np.isnan(result[2])

    def test_vegetation_water_and_cloud(self):
        # columns: dense vegetation, open water, a cloudy cell
        bands = {
            "green": np.array([[0.05, 0.06, 0.5]], dtype=np.float32),
            "red": np.array([[0.04, 0.04, 0.5]], dtype=np.float32),
            "nir": np.array([[0.45, 0.02, 0.5]], dtype=np.float32),
            "nir08": np.array([[0.45, 0.02, 0.5]], dtype=np.float32),
            "swir16": np.array([[0.20, 0.01, 0.5]], dtype=np.float32),
        }
        usable = np.array([[True, True, False]])
        indices = s2.compute_indices(bands, usable)
        assert indices["ndvi"][0, 0] > 0.8 and indices["ndvi"][0, 1] < 0
        assert indices["ndwi"][0, 1] > 0.4 and indices["ndwi"][0, 0] < 0
        assert indices["ndmi"][0, 0] > 0.3
        assert all(np.isnan(values[0, 2]) for values in indices.values())
        assert s2.means(indices)["ndvi"] == pytest.approx(
            float(np.nanmean(indices["ndvi"])), abs=1e-4)
        assert s2.means({"x": np.full((2, 2), np.nan)})["x"] is None


class _Reader:
    """Serves a scene classification per scene and constant band values."""

    def __init__(self, scl_by_scene):
        self.scl_by_scene = scl_by_scene
        self.read = []

    def __call__(self, href, grid, resampling):
        self.read.append(href)
        scene_id, name = href.split("/")[-2:]
        if name == "scl.tif":
            return np.full(grid.shape, self.scl_by_scene[scene_id],
                           dtype=np.uint8), 0
        value = {"green": 600, "red": 400, "nir": 4000, "nir08": 4000,
                 "swir16": 2000}[name[:-4]]
        return np.full(grid.shape, value, dtype=np.uint16), 0


class TestFetch:
    def test_takes_the_nearest_scene_that_is_clear(self):
        scenes = [_scene("cloudy", 1), _scene("clear", -2), _scene("far", 5)]
        reader = _Reader({"cloudy": 9, "clear": 4, "far": 4})
        log = []
        indices, meta = s2.fetch(GRID, FLIGHT, search=lambda *a: scenes,
                                 read=reader, log=log.append)
        assert meta["scene"] == "clear"
        assert meta["offset_days"] == pytest.approx(-2.0)
        assert meta["clear_fraction"] == 1.0
        assert [c["scene"] for c in meta["considered"]] == ["cloudy", "clear"]
        assert not any("far" in href for href in reader.read)
        assert set(indices) == {"ndvi", "ndmi", "ndwi", "ndsi"}
        assert indices["ndvi"].shape == GRID.shape
        assert indices["ndvi"][0, 0] == pytest.approx(0.8182, abs=1e-3)
        assert any("before the flight" in line for line in log)

    def test_cloudy_cells_of_the_chosen_scene_are_empty(self):
        scl = np.full(GRID.shape, 4, dtype=np.uint8)
        scl[0, 0] = 9

        def read(href, grid, resampling):
            if href.endswith("scl.tif"):
                return scl, 0
            return np.full(grid.shape, 3000, dtype=np.uint16), 0

        indices, meta = s2.fetch(GRID, FLIGHT, min_clear=0.5,
                                 search=lambda *a: [_scene("s", 0)], read=read)
        assert np.isnan(indices["ndvi"][0, 0])
        assert np.isfinite(indices["ndvi"][1, 1])
        assert 0.5 < meta["clear_fraction"] < 1.0

    def test_no_clear_scene_fails_and_names_the_best(self):
        scl = np.full(GRID.shape, 9, dtype=np.uint8)
        scl[0, :2] = 4

        def read(href, grid, resampling):
            return (scl if "half" in href else
                    np.full(grid.shape, 9, dtype=np.uint8)), 0

        with pytest.raises(env.EnvironmentFailure) as error:
            s2.fetch(GRID, FLIGHT,
                     search=lambda *a: [_scene("none", 1), _scene("half", 2)],
                     read=read)
        assert "half" in str(error.value) and "80%" in str(error.value)

    def test_no_scene_at_all_fails(self):
        with pytest.raises(env.EnvironmentFailure, match="no Sentinel-2"):
            s2.fetch(GRID, FLIGHT, search=lambda *a: [], read=None)

    def test_the_search_window_surrounds_the_flight(self):
        seen = {}

        def search(bounds, start, end, catalogue):
            seen.update(start=start, end=end, bounds=bounds)
            return []

        with pytest.raises(env.EnvironmentFailure):
            s2.fetch(GRID, FLIGHT, window_days=4, search=search, read=None)
        assert seen["end"] - seen["start"] == timedelta(days=8)
        assert 14.2 < seen["bounds"][0] < 14.4

    def test_cancelling_stops_between_scenes(self):
        with pytest.raises(env.Cancelled):
            s2.fetch(GRID, FLIGHT, search=lambda *a: [_scene("a", 0)],
                     read=_Reader({"a": 4}), cancelled=lambda: True)
