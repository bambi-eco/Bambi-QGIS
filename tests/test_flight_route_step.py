# -*- coding: utf-8 -*-
"""Unit tests for BambiProcessor.run_flight_route.

The poses file is optional: without it only the AirData-based route line is
generated (quick flight overview before any frame extraction). The ``bambi``
AirData parser is faked; the real pyproj handles the WGS84→UTM projection.
"""
import json
import os
import sys

import pytest

from tests.fakes import make_module


class _Frame:
    def __init__(self, latitude, longitude, altitude=100.0):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude_above_seaLevel = altitude
        self.altitude = altitude


def _install_fake_airdata(monkeypatch, frames):
    class FakeAirDataParser:
        def parse_yield(self, path):
            yield from frames

    bambi = make_module("bambi")
    bambi.__path__ = []
    airdata = make_module("bambi.airdata")
    airdata.__path__ = []
    parser_mod = make_module(
        "bambi.airdata.air_data_parser", AirDataParser=FakeAirDataParser)
    bambi.airdata = airdata
    airdata.air_data_parser = parser_mod
    monkeypatch.setitem(sys.modules, "bambi", bambi)
    monkeypatch.setitem(sys.modules, "bambi.airdata", airdata)
    monkeypatch.setitem(sys.modules, "bambi.airdata.air_data_parser", parser_mod)


def _write_poses(folder, n=3):
    poses = {"images": [
        {"location": [float(i), float(i * 2), 50.0],
         "imagefile": f"img_{i}.jpg", "timestamp": ""}
        for i in range(n)
    ]}
    with open(os.path.join(folder, "poses_t.json"), "w") as f:
        json.dump(poses, f)


class TestRunFlightRoute:
    def test_missing_poses_and_airdata_raises(self, processor, tmp_path):
        config = {"target_folder": str(tmp_path), "target_epsg": 32633}
        with pytest.raises(FileNotFoundError):
            processor.run_flight_route(config)

    def test_airdata_only_generates_route_line_without_camera_positions(
            self, processor, tmp_path, monkeypatch):
        airdata = tmp_path / "airdata.csv"
        airdata.write_text("stub")
        _install_fake_airdata(
            monkeypatch, [_Frame(48.0, 15.0), _Frame(48.001, 15.001)])
        logs = []
        config = {"target_folder": str(tmp_path), "target_epsg": 32633,
                  "airdata_path": str(airdata), "flight_route_camera": "T"}
        processor.run_flight_route(config, log_fn=logs.append)

        route_folder = tmp_path / "flight_route_t"
        route_file = route_folder / "flight_route.geojson"
        assert route_file.exists()
        geo = json.loads(route_file.read_text())
        assert geo["features"][0]["geometry"]["type"] == "LineString"
        assert len(geo["features"][0]["geometry"]["coordinates"]) == 2

        assert not (route_folder / "camera_positions.geojson").exists()
        assert any("poses_t.json not found" in m for m in logs)

    def test_airdata_with_no_usable_gps_raises(
            self, processor, tmp_path, monkeypatch):
        airdata = tmp_path / "airdata.csv"
        airdata.write_text("stub")
        # A single (0, 0) fix is dropped by the GPS-origin filter, leaving
        # neither a route line nor camera positions.
        _install_fake_airdata(monkeypatch, [_Frame(0.0, 0.0)])
        config = {"target_folder": str(tmp_path), "target_epsg": 32633,
                  "airdata_path": str(airdata)}
        with pytest.raises(RuntimeError):
            processor.run_flight_route(config)

    def test_poses_only_generates_camera_positions(self, processor, tmp_path):
        _write_poses(str(tmp_path))
        config = {"target_folder": str(tmp_path), "target_epsg": 32633,
                  "flight_route_camera": "T"}
        processor.run_flight_route(config)

        route_folder = tmp_path / "flight_route_t"
        cam_file = route_folder / "camera_positions.geojson"
        assert cam_file.exists()
        geo = json.loads(cam_file.read_text())
        assert len(geo["features"]) == 3
        assert not (route_folder / "flight_route.geojson").exists()
