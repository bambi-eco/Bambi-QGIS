# -*- coding: utf-8 -*-
"""The environment sub-steps and the habitat summary as the processor runs
them (bambi_environment.py): what lands on disk, and that one source failing
leaves the others alone."""
import json
import os

import numpy as np
import pytest

from bambi_wildlife_detection.bambi_processing import CancelledException
from bambi_wildlife_detection.core import (
    env_landcover, env_remote, env_sentinel2, env_terrain, env_weather,
    environment as env, habitat, survey_area)
from bambi_wildlife_detection.core.survey import Survey, SurveyFlight
from tests.test_habitat import WEST, EAST, _flight

ORIGIN = [447000.0, 5350500.0, 260.0]


@pytest.fixture
def config(tmp_path):
    root = str(tmp_path / "flight")
    os.makedirs(root)
    images = [{"imagefile": f"{i}.jpg", "location": [i * 20.0, i * 10.0, 80.0],
               "timestamp": f"2026-06-10T03:{10 + i}:00"} for i in range(20)]
    with open(os.path.join(root, "poses_t.json"), "w") as fh:
        json.dump({"images": images}, fh)
    dem_json = os.path.join(root, "dem.json")
    with open(dem_json, "w") as fh:
        json.dump({"origin": ORIGIN}, fh)
    mesh = os.path.join(root, "dem.glb")
    open(mesh, "w").close()
    return {"target_folder": root, "target_epsg": 32633, "dem_path": mesh,
            "alfs_dem_metadata_path": dem_json, "environment_buffer_m": 100}


def _fake_sentinel(monkeypatch, fail=None):
    def fetch(grid, flight_time, **kwargs):
        if fail is not None:
            raise fail
        values = np.full(grid.shape, 0.5, dtype=np.float32)
        return ({key: values for key in ("ndvi", "ndmi", "ndwi", "ndsi")},
                {"scene": "S2X", "offset_days": -1.0, "clear_fraction": 1.0,
                 "scene_time": "2026-06-09T10:00:00Z",
                 "window": kwargs["window_days"],
                 "min_clear": kwargs["min_clear"],
                 "flight": env.iso(flight_time)})
    monkeypatch.setattr(env_sentinel2, "fetch", fetch)


def _fake_weather(monkeypatch, fail=None):
    def fetch(lat, lon, start, end, **kwargs):
        if fail is not None:
            raise fail
        return {"at_flight": {"temperature_2m": 9.0}, "previous_24h": {},
                "hourly": [{"time": "2026-06-10T03:00:00Z",
                            "temperature_2m": 9.0}],
                "latitude": lat, "longitude": lon,
                "flight_start": env.iso(start)}
    monkeypatch.setattr(env_weather, "fetch", fetch)


class TestSources:
    def test_sentinel2_writes_four_rasters_and_its_metadata(
            self, processor, config, monkeypatch):
        _fake_sentinel(monkeypatch)
        config.update(environment_window_days=6, environment_min_clear=0.6)
        log = []
        processor.run_env_sentinel2(config, None, log.append, lambda: False)
        root = config["target_folder"]
        meta = env.read_metadata(root, env.SENTINEL2)
        assert meta["files"] == ["ndvi.tif", "ndmi.tif", "ndwi.tif", "ndsi.tif"]
        assert meta["window"] == 6 and meta["min_clear"] == 0.6
        assert meta["flight"] == "2026-06-10T03:19:30Z"   # middle of flight
        assert meta["means"]["ndvi"] == 0.5
        assert "flight path" in meta["extent"]
        array, grid = env.read_raster(env.layer_path(root, "ndvi"))
        # flight path 380 x 190 m plus 100 m on every side
        assert grid.bounds == (446900.0, 5350400.0, 447480.0, 5350790.0)
        assert array.shape == grid.shape and grid.epsg == 32633
        assert env.status(root)[env.SENTINEL2]
        assert any("Extent:" in line for line in log)

    def test_landcover_keeps_classes_and_the_legend(
            self, processor, config, monkeypatch):
        import rasterio

        def fetch(grid, **kwargs):
            classes = np.full(grid.shape, 30, dtype=np.uint8)
            return ({"landcover": classes,
                     "tree_cover": np.zeros(grid.shape, dtype=np.float32)},
                    {"composition": {"Grassland": 1.0}})
        monkeypatch.setattr(env_landcover, "fetch", fetch)
        processor.run_env_landcover(config, None, None, None)
        root = config["target_folder"]
        with rasterio.open(env.layer_path(root, "landcover")) as src:
            assert src.dtypes[0] == "uint8" and src.nodata == 0
            assert src.colormap(1)[30] == (255, 255, 76, 255)
        meta = env.read_metadata(root, env.LANDCOVER)
        assert meta["files"] == ["landcover.tif", "tree_cover.tif"]
        assert not any(key.startswith("_") for key in meta)

    def test_terrain_uses_the_flights_dem(self, processor, config,
                                          monkeypatch):
        seen = {}

        def compute(grid, mesh_path, origin, **kwargs):
            seen.update(mesh=mesh_path, origin=origin)
            return ({"twi": np.full(grid.shape, 7.0, dtype=np.float32)},
                    {"relief_m": 12.0})
        monkeypatch.setattr(env_terrain, "compute", compute)
        processor.run_env_terrain(config, None, None, None)
        assert seen["mesh"] == config["dem_path"]
        assert seen["origin"] == tuple(ORIGIN)
        assert env.status(config["target_folder"])[env.TERRAIN]

    def test_weather_writes_the_document_and_the_hourly_table(
            self, processor, config, monkeypatch):
        _fake_weather(monkeypatch)
        log = []
        processor.run_env_weather(config, None, log.append, None)
        root = config["target_folder"]
        meta = env.read_metadata(root, env.WEATHER)
        assert meta["files"] == [] and meta["at_flight"]["temperature_2m"] == 9.0
        assert 48.30 < meta["latitude"] < 48.32
        assert meta["flight_start"] == "2026-06-10T03:10:00Z"
        assert os.path.isfile(os.path.join(env.folder(root),
                                           "weather_hourly.csv"))
        assert any("At flight time" in line for line in log)


class TestIndependence:
    def test_a_failing_source_leaves_the_others_untouched(
            self, processor, config, monkeypatch):
        _fake_sentinel(monkeypatch)
        _fake_weather(monkeypatch, fail=env_remote.SourceUnavailable(
            "the Open-Meteo archive answered HTTP 429."))
        processor.run_env_sentinel2(config, None, None, None)
        with pytest.raises(RuntimeError) as error:
            processor.run_env_weather(config, None, None, None)
        assert "Weather (Open-Meteo)" in str(error.value)
        assert "429" in str(error.value)
        assert env.status(config["target_folder"]) == {
            env.SENTINEL2: True, env.LANDCOVER: False,
            env.TERRAIN: False, env.WEATHER: False}

    def test_a_failed_rerun_removes_that_sources_old_output(
            self, processor, config, monkeypatch):
        root = config["target_folder"]
        _fake_sentinel(monkeypatch)
        processor.run_env_sentinel2(config, None, None, None)
        _fake_sentinel(monkeypatch, fail=env.EnvironmentFailure(
            "No Sentinel-2 scene is clear."))
        with pytest.raises(RuntimeError, match="No Sentinel-2 scene"):
            processor.run_env_sentinel2(config, None, None, None)
        assert not env.status(root)[env.SENTINEL2]
        assert env.available_layers(root) == []

    def test_cancelling_is_not_an_error(self, processor, config, monkeypatch):
        def fetch(*args, **kwargs):
            raise env.Cancelled()
        monkeypatch.setattr(env_sentinel2, "fetch", fetch)
        with pytest.raises(CancelledException):
            processor.run_env_sentinel2(config, None, None, None)

    def test_an_unknown_extent_fails_before_anything_is_removed(
            self, processor, config, monkeypatch):
        root = config["target_folder"]
        _fake_sentinel(monkeypatch)
        processor.run_env_sentinel2(config, None, None, None)
        os.remove(config["alfs_dem_metadata_path"])
        with pytest.raises(RuntimeError, match="extent is unknown"):
            processor.run_env_sentinel2(config, None, None, None)
        assert env.status(root)[env.SENTINEL2]

    def test_a_missing_folder_fails(self, processor):
        with pytest.raises(RuntimeError, match="target folder"):
            processor.run_env_weather({"target_folder": "/nope"},
                                      None, None, None)


class TestSurveyArea:
    def _survey(self, tmp_path, enabled=True):
        from shapely.geometry import box

        folder = str(tmp_path / "survey")
        os.makedirs(folder)
        survey_area.create(folder, 32633, plots=[
            (box(446000.0, 5350000.0, 446300.0, 5350200.0), "plot")])
        return Survey(folder=folder, area_enabled=enabled).to_dict()

    def test_the_survey_area_widens_the_extent(self, processor, config,
                                               monkeypatch, tmp_path):
        _fake_sentinel(monkeypatch)
        config["survey"] = self._survey(tmp_path)
        processor.run_env_sentinel2(config, None, None, None)
        meta = env.read_metadata(config["target_folder"], env.SENTINEL2)
        assert meta["grid"]["bounds"][0] == 445900.0
        assert "survey area" in meta["extent"]

    @pytest.mark.parametrize("include,area_enabled", [
        (False, True),    # the Environment tab's tick box is off
        (True, False),    # the survey area itself is switched off
    ])
    def test_not_when_switched_off(self, processor, config, monkeypatch,
                                   tmp_path, include, area_enabled):
        _fake_sentinel(monkeypatch)
        config["survey"] = self._survey(tmp_path, enabled=area_enabled)
        config["environment_include_survey_area"] = include
        processor.run_env_sentinel2(config, None, None, None)
        meta = env.read_metadata(config["target_folder"], env.SENTINEL2)
        assert meta["grid"]["bounds"][0] == 446900.0


class TestHabitatSummary:
    def _config(self, tmp_path, flights, **survey):
        folder = str(tmp_path / "survey")
        described = Survey(
            folder=folder, modality="t",
            flights=[SurveyFlight(name, root, 32633) for name, root in flights],
            **survey)
        return {"survey": described.to_dict(), "environment_radius_m": 30.0}

    def test_writes_the_survey_files(self, processor, tmp_path):
        a = _flight(str(tmp_path / "a"), [WEST, WEST, EAST])
        b = _flight(str(tmp_path / "b"), [WEST], species="red deer")
        config = self._config(tmp_path, [("A", a), ("B", b)])
        log = []
        summary = processor.run_habitat_summary(
            config, None, log.append, lambda: False)
        folder = os.path.join(str(tmp_path / "survey"), "analytics_t")
        assert sorted(os.listdir(folder)) == [
            "habitat.json", "habitat_tracks.csv", "habitat_tracks.json"]
        assert summary["n_tracks"] == 4 and summary["survey_area"] is None
        assert habitat.read_json(
            os.path.join(folder, "habitat.json"))["radius_m"] == 30.0
        assert any("A: 3 track(s)" in line for line in log)
        assert any("Terrain wetness" in line for line in log)   # not run

    def test_the_species_filter_applies(self, processor, tmp_path):
        a = _flight(str(tmp_path / "a"), [WEST, EAST])
        b = _flight(str(tmp_path / "b"), [WEST], species="red deer")
        config = self._config(tmp_path, [("A", a), ("B", b)],
                              species=["red deer"])
        summary = processor.run_habitat_summary(config, None, None, None)
        assert summary["n_tracks"] == 1
        assert [f["n_tracks"] for f in summary["flights"]] == [0, 1]

    def test_a_flight_without_environment_is_named(self, processor, tmp_path):
        a = _flight(str(tmp_path / "a"), [WEST])
        b = _flight(str(tmp_path / "b"), [WEST])
        for source in env.SOURCES:
            env.clear(b, source)
        config = self._config(tmp_path, [("A", a), ("B", b)])
        with pytest.raises(RuntimeError) as error:
            processor.run_habitat_summary(config, None, None, None)
        assert "No environment data for: B" in str(error.value)

    def test_tracks_are_required(self, processor, tmp_path):
        root = str(tmp_path / "empty")
        os.makedirs(root)
        env.write_metadata(root, env.WEATHER, {"files": []})
        config = self._config(tmp_path, [("E", root)])
        with pytest.raises(FileNotFoundError, match="Track Animals"):
            processor.run_habitat_summary(config, None, None, None)
