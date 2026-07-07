# -*- coding: utf-8 -*-
"""Unit tests for the evaluation flight strategy (transect grid + routes).

The module is Qt-free; it needs the geo stack (geopandas/fiona/shapely/
simplekml), which is part of the unit Docker image. All geometry is built
around a synthetic 1 km x 1 km flight area in UTM 33N.
"""
import json

import pytest

pytest.importorskip("geopandas")
pytest.importorskip("fiona")
pytest.importorskip("shapely")
pytest.importorskip("simplekml")
pytest.importorskip("pyproj")

import geopandas as gpd  # noqa: E402
import shapely  # noqa: E402
from pyproj import CRS, Transformer  # noqa: E402

from bambi_wildlife_detection.evaluation_flight_strategy import (  # noqa: E402
    EvaluationFlightStrategy,
    RandomStrategy,
)

# 1 km x 1 km flight area in UTM 33N metres
AREA_MIN_X, AREA_MIN_Y = 400000.0, 5260000.0
AREA_MAX_X, AREA_MAX_Y = 401000.0, 5261000.0


class _ConcreteStrategy(EvaluationFlightStrategy):
    """Expose the abstract base's concrete helpers for testing."""

    def create_routes(self, *args, **kwargs):
        raise NotImplementedError


@pytest.fixture
def strategy():
    return _ConcreteStrategy()


@pytest.fixture
def transformer():
    # Same construction as create_routes uses
    return Transformer.from_crs(CRS.from_epsg(32633), CRS.from_epsg(4326))


@pytest.fixture
def flight_area():
    return shapely.geometry.box(AREA_MIN_X, AREA_MIN_Y, AREA_MAX_X, AREA_MAX_Y)


def _utm_to_lonlat_ring(corners):
    to_wgs = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
    return [list(to_wgs.transform(x, y)) for x, y in corners]


def _write_geojson(path, features):
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return str(path)


@pytest.fixture
def area_path(tmp_path):
    ring = _utm_to_lonlat_ring([
        (AREA_MIN_X, AREA_MIN_Y), (AREA_MAX_X, AREA_MIN_Y),
        (AREA_MAX_X, AREA_MAX_Y), (AREA_MIN_X, AREA_MAX_Y),
        (AREA_MIN_X, AREA_MIN_Y),
    ])
    return _write_geojson(tmp_path / "area.geojson", [{
        "type": "Feature", "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }])


@pytest.fixture
def start_points_path(tmp_path):
    center = _utm_to_lonlat_ring([((AREA_MIN_X + AREA_MAX_X) / 2,
                                   (AREA_MIN_Y + AREA_MAX_Y) / 2)])[0]
    return _write_geojson(tmp_path / "starts.geojson", [{
        "type": "Feature", "properties": {"name": "s0"},
        "geometry": {"type": "Point", "coordinates": center},
    }])


class TestGetCrs:
    def test_metric_crs_accepted(self, strategy):
        crs = strategy.get_crs(32633)
        assert crs.to_epsg() == 32633

    def test_degree_crs_rejected(self, strategy):
        with pytest.raises(Exception, match="metres"):
            strategy.get_crs(4326)


class TestReadData:
    def test_reads_and_reprojects_to_target_crs(
            self, strategy, area_path, start_points_path):
        target_crs = strategy.get_crs(32633)
        areas, starts, invalid = strategy.read_data(
            area_path, start_points_path, target_crs)
        assert invalid is None
        assert areas.crs.to_epsg() == 32633
        assert starts.crs.to_epsg() == 32633
        # Round-trip through WGS84 keeps the metric footprint (~1 km square)
        minx, miny, maxx, maxy = areas.geometry[0].bounds
        assert maxx - minx == pytest.approx(1000, abs=5)
        assert maxy - miny == pytest.approx(1000, abs=5)

    def test_reads_invalid_areas_when_given(
            self, strategy, area_path, start_points_path):
        target_crs = strategy.get_crs(32633)
        _, _, invalid = strategy.read_data(
            area_path, start_points_path, target_crs,
            invalid_areas_path=area_path)
        assert invalid is not None
        assert invalid.crs.to_epsg() == 32633


class TestGetRouteGeojson:
    ROUTE = [
        (AREA_MIN_X, AREA_MIN_Y),
        (AREA_MIN_X + 250, AREA_MIN_Y),
        (AREA_MIN_X + 250, AREA_MIN_Y + 250),
    ]

    def test_structure(self, strategy, transformer):
        geojson = strategy.get_route_geojson(self.ROUTE, ["t1", "t2"], transformer)
        features = geojson["features"]
        points = [f for f in features if f["geometry"]["type"] == "Point"]
        lines = [f for f in features if f["geometry"]["type"] == "LineString"]
        assert len(points) == 3
        # one LineString per transect + the total route
        assert len(lines) == 3
        names = [f["properties"]["name"] for f in lines]
        assert names == ["t1", "t2", "total-route"]
        total = lines[-1]
        assert len(total["geometry"]["coordinates"]) == 3


class TestSaveTotalRouteKml:
    def test_kml_written_with_start_end(self, strategy, transformer, tmp_path):
        out = tmp_path / "route.kml"
        strategy.save_total_route_kml(
            TestGetRouteGeojson.ROUTE, ["t1", "t2"], transformer, str(out))
        content = out.read_text()
        assert "Total Route" in content
        assert "<name>Start</name>" in content
        assert "<name>End</name>" in content
        assert "transects-t1-t2" in content

    def test_kml_without_start_end_markers(self, strategy, transformer, tmp_path):
        out = tmp_path / "route.kml"
        strategy.save_total_route_kml(
            TestGetRouteGeojson.ROUTE, ["t1", "t2"], transformer, str(out),
            add_start_end=False)
        content = out.read_text()
        assert "<name>Start</name>" not in content
        assert "<name>End</name>" not in content


class TestCalculateGrid:
    GRID_SIZE = 250.0

    def test_grid_dimensions_and_valid_nodes(
            self, strategy, flight_area, transformer):
        grid, valid_nodes, transects, width, height = strategy.calculate_grid(
            flight_area, self.GRID_SIZE, None, transformer,
            min_transect_overlap=0.75)
        # 0..1000 in 250 m steps plus one overshoot row/column
        assert (width, height) == (6, 6)
        # Only strictly interior points are valid: {250,500,750}^2
        assert len(valid_nodes) == 9
        for node in valid_nodes:
            assert node.latitude in (AREA_MIN_X + 250, AREA_MIN_X + 500, AREA_MIN_X + 750)

    def test_transect_validity_against_area(
            self, strategy, flight_area, transformer):
        _, _, transects, _, _ = strategy.calculate_grid(
            flight_area, self.GRID_SIZE, None, transformer,
            min_transect_overlap=0.75)
        # 6x6 grid: 30 horizontal + 30 vertical candidate transects
        assert len(transects) == 60
        valid = [k for k, v in transects.items() if v.is_valid]
        # Only segments between coordinates <= 1000 m lie inside the area:
        # 5x4 per orientation
        assert len(valid) == 40
        # An interior transect is valid, one reaching the overshoot row is not
        assert transects["1_1_1_2"].is_valid is True
        assert transects["5_0_5_1"].is_valid is False

    def test_no_go_area_invalidates_crossing_transects(
            self, strategy, flight_area, transformer):
        blocker = gpd.GeoDataFrame(
            geometry=[shapely.geometry.box(
                AREA_MIN_X + 240, AREA_MIN_Y + 240,
                AREA_MIN_X + 260, AREA_MIN_Y + 510)],
            crs="EPSG:32633")
        _, _, transects, _, _ = strategy.calculate_grid(
            flight_area, self.GRID_SIZE, blocker, transformer,
            min_transect_overlap=0.75)
        assert transects["1_1_1_2"].is_valid is False

    def test_grid_files_written(self, strategy, flight_area, transformer, tmp_path):
        strategy.calculate_grid(
            flight_area, self.GRID_SIZE, None, transformer,
            min_transect_overlap=0.75, target_path=str(tmp_path))
        for name in ("grid.geojson", "grid_filtered.geojson",
                     "transects.geojson", "transects_valids.geojson"):
            data = json.loads((tmp_path / name).read_text())
            assert data["type"] == "FeatureCollection"
        filtered = json.loads((tmp_path / "grid_filtered.geojson").read_text())
        assert len(filtered["features"]) == 9


class TestCreateStartpointsGeojson:
    def test_writes_startpoints_file(self, strategy, transformer, tmp_path):
        out_dir = tmp_path / "plan"
        strategy.create_startpoints_geojson(
            [[0, AREA_MIN_X, AREA_MIN_Y], [1, AREA_MAX_X, AREA_MAX_Y]],
            transformer, str(out_dir))
        data = json.loads((out_dir / "startpoints.geojson").read_text())
        assert [f["properties"]["name"] for f in data["features"]] == \
            ["start-0", "start-1"]


class TestRandomStrategyEndToEnd:
    def test_create_routes_produces_valid_plan(
            self, area_path, start_points_path, tmp_path):
        target = tmp_path / "plan"
        target.mkdir()
        strategy = RandomStrategy(
            grid_size=250.0,
            max_start_and_stop_distance=10000.0,
            min_transects=2,
            max_distance=2000.0,
            min_transect_overlap=0.75,
            number_of_retries=50,
            max_number_of_flights=4,
            target_crs_epsg=32633,
            min_transects_per_route=1,
            seed=1234,
        )
        routes = strategy.create_routes(area_path, start_points_path, str(target))

        assert isinstance(routes, list)
        assert len(routes) >= 1
        # Route points are UTM metre positions inside the (padded) grid
        for route in routes:
            for point in route:
                assert AREA_MIN_X - 300 <= point[0] <= AREA_MAX_X + 300

        # All documented outputs exist
        assert (target / "startpoints.geojson").exists()
        assert (target / "grid.geojson").exists()
        assert (target / "transects_valids.geojson").exists()
        assert (target / "log.txt").exists()
        valid_routes = list((target / "routes" / "valid").glob("route_*.geojson"))
        assert valid_routes
        kml_routes = list((target / "routes" / "valid_kml").glob("route_*.kml"))
        assert len(kml_routes) == len(valid_routes)
        # Selected routes were augmented with arrival/departure to the start
        selected = json.loads(valid_routes[0].read_text())
        line_names = [f["properties"]["name"] for f in selected["features"]
                      if f["geometry"]["type"] == "LineString"]
        assert "arrival" in line_names
        assert "departure" in line_names

    def test_seeded_runs_are_reproducible(
            self, area_path, start_points_path, tmp_path):
        def run(folder):
            target = tmp_path / folder
            target.mkdir()
            strategy = RandomStrategy(
                grid_size=250.0, max_start_and_stop_distance=10000.0,
                min_transects=2, max_distance=2000.0,
                number_of_retries=50, max_number_of_flights=4,
                target_crs_epsg=32633, min_transects_per_route=1, seed=99)
            return strategy.create_routes(
                area_path, start_points_path, str(target))

        assert run("a") == run("b")
