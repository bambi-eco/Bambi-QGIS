# -*- coding: utf-8 -*-
"""Which open model a flight's elevation is built from
(copernicus_dem_downloader.build_dem): the terrain model where it has the
flight, the surface model where it has not - or the one that was asked
for. Both are handed in, so that no network is needed."""
import json
import os

import numpy as np
import pytest

from bambi_wildlife_detection import copernicus_dem_downloader as cdd
from bambi_wildlife_detection.core import (
    copernicus_dem, environment, gem15_dem)

pytest.importorskip("rasterio")
pytest.importorskip("pyproj")

FLIGHT = (14.508, 48.365, 14.518, 48.372)
SURFACE = {"product": "Copernicus DEM GLO-30", "editions": ["GLO-30"],
           "tiles": ["Copernicus_DSM_COG_10_N48_00_E014_00_DEM"],
           "attribution": copernicus_dem.ATTRIBUTION,
           "vertical_datum": "EGM2008", "surface_model": True,
           "sea_share": 0.0}
TERRAIN = {"product": "Google GEM-15 terrain model, levelled to the "
                      "Copernicus DEM",
           "tiles": ["w14s48"], "attribution": gem15_dem.ATTRIBUTION,
           "vertical_datum": "EGM2008", "surface_model": False,
           "resolution_m": 15.0, "levelled_by_m": 3.1,
           "open_ground_cells": 3590, "filled_share": 0.005,
           "filled_from": "Copernicus DEM GLO-30", "sea_share": 0.0,
           "min_m": 352.0, "max_m": 465.0}


class Readers:
    """A terrain model and a surface model that note who asked for what."""

    def __init__(self, terrain=True):
        self.has_terrain = terrain
        self.asked = []

    def surface(self, grid, log=None, cancelled=None, progress=None):
        self.asked.append(("surface", grid.resolution))
        return np.full(grid.shape, 480.0, dtype=np.float32), dict(SURFACE)

    def terrain(self, grid, reference=None, log=None, cancelled=None,
                progress=None):
        self.asked.append(("terrain", grid.resolution))
        self.reference = reference
        if not self.has_terrain:
            raise gem15_dem.NoTerrain("GEM-15 has no heights where the "
                                      "flight is.")
        if progress:
            progress(90)
        return np.full(grid.shape, 460.0, dtype=np.float32), dict(TERRAIN)

    def build(self, folder, source=None, **more):
        if source is not None:
            more["source"] = source
        return cdd.build_dem(FLIGHT, str(folder), "EPSG:32633",
                             fetch=self.surface, fetch_terrain=self.terrain,
                             **more)


def test_the_terrain_model_is_taken_where_there_is_one(tmp_path):
    readers, log = Readers(), []

    built = readers.build(tmp_path / "out", log=log.append)

    assert readers.asked == [("terrain", 15.0)]
    assert built["kind"] == cdd.GEM15
    assert sorted(os.listdir(tmp_path / "out")) == [
        "gem15_dem.glb", "gem15_dem.json", "gem15_dem.tif"]
    assert built["mesh"].endswith("gem15_dem.glb")
    assert built["grid"].resolution == 15.0
    assert log[0].endswith(
        "km around the flight, in cells of 15 m, EPSG:32633")
    # the surface model is what the terrain model is levelled by
    assert readers.reference == readers.surface


def test_what_the_terrain_model_is_goes_with_it(tmp_path):
    built = Readers().build(tmp_path)

    with open(built["metadata"], encoding="utf-8") as handle:
        source = json.load(handle)["source"]
    assert source == {
        "product": TERRAIN["product"], "tiles": ["w14s48"],
        "attribution": gem15_dem.ATTRIBUTION, "vertical_datum": "EGM2008",
        "surface_model": False, "sea_share": 0.0, "levelled_by_m": 3.1,
        "open_ground_cells": 3590, "filled_share": 0.005,
        "filled_from": "Copernicus DEM GLO-30", "resolution_m": 15.0}


def test_where_the_terrain_model_has_nothing_the_surface_model_is_taken(
        tmp_path):
    readers, log = Readers(terrain=False), []

    built = readers.build(tmp_path / "out", log=log.append)

    assert readers.asked == [("terrain", 15.0), ("surface", 30.0)]
    assert built["kind"] == cdd.COPERNICUS
    assert sorted(os.listdir(tmp_path / "out")) == [
        "copernicus_dem.glb", "copernicus_dem.json", "copernicus_dem.tif"]
    assert log[0] == (
        "GEM-15 has no heights where the flight is. The Copernicus DEM is "
        "used instead: a surface model at 30 m, with the canopy of a "
        "forest in it.")
    with open(built["metadata"], encoding="utf-8") as handle:
        assert json.load(handle)["source"]["surface_model"] is True


def test_only_the_terrain_model_and_it_has_nothing(tmp_path):
    readers = Readers(terrain=False)

    with pytest.raises(gem15_dem.NoTerrain):
        readers.build(tmp_path / "out", source=cdd.GEM15)

    assert readers.asked == [("terrain", 15.0)]
    assert not os.path.exists(tmp_path / "out")


def test_only_the_terrain_model(tmp_path):
    readers = Readers()

    built = readers.build(tmp_path, source=cdd.GEM15)

    assert built["kind"] == cdd.GEM15 and readers.asked == [("terrain", 15.0)]


def test_only_the_surface_model(tmp_path):
    readers = Readers()

    built = readers.build(tmp_path, source=cdd.COPERNICUS)

    assert readers.asked == [("surface", 30.0)]
    assert built["kind"] == cdd.COPERNICUS
    assert built["mesh"].endswith("copernicus_dem.glb")


def test_a_source_nobody_knows(tmp_path):
    with pytest.raises(environment.EnvironmentFailure) as raised:
        Readers().build(tmp_path, source="lidar")

    assert str(raised.value) == (
        "'lidar' is no elevation source: auto, gem15 or copernicus.")


def test_another_failure_of_the_terrain_model_is_not_hidden(tmp_path):
    """A service that does not answer is not "no terrain here": falling
    back would quietly give a flight a model with the trees in it."""
    readers = Readers()

    def broken(grid, **more):
        raise environment.EnvironmentFailure("storage.googleapis.com "
                                             "answered 503")

    with pytest.raises(environment.EnvironmentFailure, match="503"):
        cdd.build_dem(FLIGHT, str(tmp_path), "EPSG:32633",
                      fetch=readers.surface, fetch_terrain=broken)

    assert readers.asked == []


def test_the_sources_a_user_can_choose_from():
    assert [key for key, _title in cdd.SOURCES] == [
        cdd.AUTO, cdd.GEM15, cdd.COPERNICUS]
    assert cdd.STEMS == {cdd.GEM15: "gem15_dem",
                         cdd.COPERNICUS: "copernicus_dem"}
    assert cdd.STEM == "copernicus_dem"


def test_the_terrain_models_own_reader_is_given_the_surface_reader(
        tmp_path, monkeypatch):
    """Without anything handed in, GEM-15 is levelled by the Copernicus
    DEM's own reader."""
    seen = {}

    def terrain(grid, reference=None, **more):
        seen["reference"] = reference
        return np.full(grid.shape, 460.0, dtype=np.float32), dict(TERRAIN)

    monkeypatch.setattr(gem15_dem, "fetch", terrain)

    cdd.build_dem(FLIGHT, str(tmp_path), "EPSG:32633")

    assert seen["reference"] is copernicus_dem.fetch


# ---------------------------------------------------------------------------
# What a built model is, in words
# ---------------------------------------------------------------------------

class TestDescribe:
    def _metadata(self, tmp_path, source):
        path = tmp_path / "dem.json"
        path.write_text(json.dumps({"crs": "EPSG:32633", "source": source}),
                        encoding="utf-8")
        return str(path)

    def test_a_terrain_model(self, tmp_path):
        text = cdd.describe(self._metadata(tmp_path, dict(
            TERRAIN, resolution_m=15.0, filled_share=0.0)))

        assert text == (
            "It was made from the Google GEM-15 terrain model, levelled to "
            "the Copernicus DEM: the bare ground at 15 m. Its heights are "
            "good to a few metres, under trees less so: set what you take "
            "them to be under 'Position Uncertainty' (Terrain model).")

    def test_one_that_could_not_be_levelled_and_has_water_in_it(
            self, tmp_path):
        text = cdd.describe(self._metadata(tmp_path, dict(
            TERRAIN, product="Google GEM-15 terrain model",
            resolution_m=15.0, levelled_by_m=None, filled_share=0.31)))

        assert ("It could not be levelled - too little open ground under "
                "the flight - so its heights may lie a few metres low or "
                "high.") in text
        assert ("31 % of it had no value (water, snow) and is taken from "
                "Copernicus DEM GLO-30.") in text

    def test_a_surface_model(self, tmp_path):
        text = cdd.describe(self._metadata(tmp_path, dict(
            SURFACE, resolution_m=30.0)))

        assert text == (
            "It was made from the Copernicus DEM GLO-30: a surface model at "
            "30 m. Over forest it lies in the canopy - allow for that under "
            "'Position Uncertainty' (Terrain model).")

    @pytest.mark.parametrize("content", [
        "{ not json", "[]", '{"crs": "EPSG:32633"}', '{"source": {}}'])
    def test_metadata_that_does_not_say(self, tmp_path, content):
        path = tmp_path / "dem.json"
        path.write_text(content, encoding="utf-8")

        assert cdd.describe(str(path)) == ""
        assert cdd.describe(str(tmp_path / "none.json")) == ""

    def test_what_a_real_build_writes_can_be_described(self, tmp_path):
        built = Readers().build(tmp_path)

        assert cdd.describe(built["metadata"]).startswith(
            "It was made from the Google GEM-15 terrain model, levelled to "
            "the Copernicus DEM: the bare ground at 15 m.")
