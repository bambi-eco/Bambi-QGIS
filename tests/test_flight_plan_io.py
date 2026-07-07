# -*- coding: utf-8 -*-
"""Unit tests for core.flight_plan_io (flight planner I/O + worker)."""
import json

import pytest

from bambi_wildlife_detection.core.flight_plan_io import (
    _DEFAULTS,
    _FlightPlanWorker,
    _to_bool,
    _write_route_line,
    _write_route_transects,
)
from tests.fakes import SignalRecorder


def _route_geojson(tmp_path, features):
    path = tmp_path / "route.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return str(path)


def _line(name, coords=((0, 0), (1, 1))):
    return {"type": "Feature", "properties": {"name": name},
            "geometry": {"type": "LineString", "coordinates": [list(c) for c in coords]}}


def _point(name):
    return {"type": "Feature", "properties": {"name": name},
            "geometry": {"type": "Point", "coordinates": [0, 0]}}


class TestToBool:
    @pytest.mark.parametrize("value,expected", [
        (True, True), (False, False),
        ("true", True), ("True", True), ("1", True), ("yes", True),
        ("false", False), ("0", False), ("no", False), ("", False),
    ])
    def test_coercion(self, value, expected):
        assert _to_bool(value) is expected


class TestDefaults:
    def test_contains_every_strategy_parameter(self):
        # keys the dialog binds to widgets — a removal would silently break
        # settings persistence
        for key in ("strategy", "grid_size", "min_transects", "max_distance",
                    "target_crs_epsg", "seed", "random_search", "target_folder"):
            assert key in _DEFAULTS


class TestWriteRouteLine:
    def test_extracts_total_route(self, tmp_path):
        src = _route_geojson(tmp_path, [
            _point("0"), _line("t1"), _line("total-route", ((0, 0), (2, 2)))])
        out = tmp_path / "line.geojson"
        assert _write_route_line(src, str(out)) == str(out)
        data = json.loads(out.read_text())
        assert len(data["features"]) == 1
        assert data["features"][0]["properties"]["name"] == "total-route"

    def test_missing_total_route_returns_none(self, tmp_path):
        src = _route_geojson(tmp_path, [_line("t1")])
        assert _write_route_line(src, str(tmp_path / "o.geojson")) is None

    def test_unreadable_source_returns_none(self, tmp_path):
        assert _write_route_line(
            str(tmp_path / "missing.geojson"), str(tmp_path / "o.geojson")) is None


class TestWriteRouteTransects:
    def test_numbers_transects_and_skips_non_survey_lines(self, tmp_path):
        src = _route_geojson(tmp_path, [
            _point("0"),
            _line("arrival"),
            _line("3_4_3_5"),
            _line("3_5_4_5"),
            _line("departure"),
            _line("total-route"),
        ])
        out = tmp_path / "transects.geojson"
        assert _write_route_transects(src, str(out)) == str(out)
        data = json.loads(out.read_text())
        names = [f["properties"]["name"] for f in data["features"]]
        numbers = [f["properties"]["transect_no"] for f in data["features"]]
        assert names == ["3_4_3_5", "3_5_4_5"]
        assert numbers == [1, 2]

    def test_no_transects_returns_none(self, tmp_path):
        src = _route_geojson(tmp_path, [_line("total-route"), _point("0")])
        assert _write_route_transects(src, str(tmp_path / "o.geojson")) is None


class TestFlightPlanWorker:
    class _RecordingStrategy:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            type(self).instances.append(self)

        def create_routes(self, area_path, start_points_path, target_path,
                          invalid_areas_path=None):
            self.call = dict(area_path=area_path,
                             start_points_path=start_points_path,
                             target_path=target_path,
                             invalid_areas_path=invalid_areas_path)
            return [[1], [2]]

    def test_runs_strategy_and_reports_route_count(self):
        worker = _FlightPlanWorker(
            self._RecordingStrategy, {"grid_size": 100.0},
            "area.geojson", "starts.geojson", "/target", None)
        worker.finished = SignalRecorder()
        worker.run()
        assert worker.finished.calls == [(True, "2")]
        strategy = self._RecordingStrategy.instances[-1]
        assert strategy.kwargs == {"grid_size": 100.0}
        assert strategy.call["area_path"] == "area.geojson"
        assert strategy.call["invalid_areas_path"] is None

    def test_reports_traceback_on_failure(self):
        class ExplodingStrategy:
            def __init__(self, **kwargs):
                pass

            def create_routes(self, **kwargs):
                raise ValueError("bad area")

        worker = _FlightPlanWorker(
            ExplodingStrategy, {}, "a", "s", "t", None)
        worker.finished = SignalRecorder()
        worker.run()
        assert len(worker.finished.calls) == 1
        success, message = worker.finished.calls[0]
        assert success is False
        assert "bad area" in message
        assert "Traceback" in message
