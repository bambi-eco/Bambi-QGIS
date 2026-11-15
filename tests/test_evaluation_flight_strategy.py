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
    RandomLoopStrategy,
    RandomStrategy,
    TransectStatus,
    clear_previous_plan,
    clear_routes,
    meets_no_fly,
    wgs84_transformer,
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
    return wgs84_transformer(CRS.from_epsg(32633))


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


def _a_flight_folder(folder):
    """A target folder that already holds somebody's data: a flight, notes,
    and a ``routes`` folder with files that are not the planner's."""
    (folder / "frames_t").mkdir(parents=True)
    (folder / "frames_t" / "00000001.jpg").write_bytes(b"jpeg")
    (folder / "project.gpkg").write_bytes(b"sqlite")
    (folder / "notes.txt").write_text("field notes")
    (folder / "routes" / "valid").mkdir(parents=True)
    (folder / "routes" / "valid" / "my_route.gpx").write_text("<gpx/>")
    (folder / "routes" / "archive").mkdir()
    (folder / "routes" / "archive" / "route_1.geojson").write_text("{}")
    return {path: path.read_bytes() for path in folder.rglob("*")
            if path.is_file()}


def _plan(strategy_cls, area_path, start_points_path, target, seed=1234,
          **overrides):
    parameters = dict(
        grid_size=250.0, max_start_and_stop_distance=10000.0, min_transects=2,
        max_distance=2000.0, number_of_retries=50, target_crs_epsg=32633,
        min_transects_per_route=1, seed=seed)
    parameters.update(overrides)
    return strategy_cls(**parameters).create_routes(
        area_path, start_points_path, str(target))


class TestTheTargetFolderIsNotThePlanners:
    """A plan used to start by deleting its target folder - with a flight's
    folder chosen as the target, the flight. It may only replace what an
    earlier plan wrote."""

    @pytest.mark.parametrize("strategy_cls, extra", [
        (RandomStrategy, {"max_number_of_flights": 4}),
        (RandomLoopStrategy, {}),
    ])
    def test_planning_keeps_everything_else_in_the_folder(
            self, strategy_cls, extra, area_path, start_points_path, tmp_path):
        target = tmp_path / "flight"
        before = _a_flight_folder(target)

        _plan(strategy_cls, area_path, start_points_path, target, **extra)

        for path, content in before.items():
            assert path.is_file(), f"{path.name} was deleted"
            assert path.read_bytes() == content
        assert (target / "startpoints.geojson").is_file()
        assert (target / "log.txt").is_file()

    def test_a_new_plan_replaces_the_routes_of_the_old_one(
            self, area_path, start_points_path, tmp_path):
        target = tmp_path / "plan"
        target.mkdir()
        _plan(RandomStrategy, area_path, start_points_path, target,
              max_number_of_flights=4)
        valid = target / "routes" / "valid"
        # routes the next plan will not write again
        for folder in ("valid", "all", "valid_kml", "valid_lines", "7"):
            (target / "routes" / folder).mkdir(exist_ok=True)
        (valid / "route_99.geojson").write_text("{}")
        (target / "routes" / "all" / "route_99.geojson").write_text("{}")
        (target / "routes" / "valid_kml" / "route_99.kml").write_text("")
        (target / "routes" / "valid_lines" / "route_99.geojson").write_text("{}")
        (target / "routes" / "valid_lines" / "route_99_transects.geojson"
         ).write_text("{}")
        (target / "routes" / "7" / "route_99.geojson").write_text("{}")

        _plan(RandomStrategy, area_path, start_points_path, target,
              max_number_of_flights=4, seed=5)

        left = sorted(str(path.relative_to(target))
                      for path in target.rglob("route_99*"))
        assert left == []
        assert not (target / "routes" / "7").exists()
        assert list(valid.glob("route_*.geojson"))

    def test_clearing_a_folder_without_a_plan_does_nothing(self, tmp_path):
        before = _a_flight_folder(tmp_path)
        clear_previous_plan(str(tmp_path))
        clear_routes(str(tmp_path / "missing"))
        assert {path: path.read_bytes() for path in tmp_path.rglob("*")
                if path.is_file()} == before

    def test_only_the_planners_own_files_go(self, tmp_path):
        before = _a_flight_folder(tmp_path)
        for name in ("startpoints.geojson", "grid.geojson",
                     "grid_filtered.geojson", "transects.geojson",
                     "transects_valids.geojson", "log.txt"):
            (tmp_path / name).write_text("{}")
        (tmp_path / "routes" / "valid" / "route_3.geojson").write_text("{}")
        (tmp_path / "routes" / "0").mkdir()
        (tmp_path / "routes" / "0" / "route_3.geojson").write_text("{}")
        # the planner's naming, but not where the planner writes
        (tmp_path / "route_3.geojson").write_text("{}")
        before[tmp_path / "route_3.geojson"] = b"{}"

        clear_previous_plan(str(tmp_path))

        assert {path: path.read_bytes() for path in tmp_path.rglob("*")
                if path.is_file()} == before
        assert not (tmp_path / "routes" / "0").exists()
        assert (tmp_path / "routes" / "valid").is_dir()     # still holds a file



# ---------------------------------------------------------------------------
# Routing: where the plan is written, what blocks a line, when planning ends
# ---------------------------------------------------------------------------

def _lonlat_bounds(path):
    """Bounds of every coordinate in a GeoJSON file, as written."""
    def pairs(node):
        if isinstance(node, (list, tuple)) and len(node) >= 2 and \
                all(isinstance(value, (int, float)) for value in node[:2]):
            yield node[0], node[1]
        elif isinstance(node, (list, tuple)):
            for child in node:
                yield from pairs(child)

    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    found = [pair for feature in document["features"]
             for pair in pairs(feature["geometry"]["coordinates"])]
    xs, ys = [p[0] for p in found], [p[1] for p in found]
    return min(xs), min(ys), max(xs), max(ys)


def _plan_routes(tmp_path, strategy_cls, area_path, start_points_path, **options):
    out = tmp_path / "plan"
    out.mkdir(exist_ok=True)
    invalid = options.pop("invalid_areas_path", None)
    options.setdefault("seed", 3)
    strategy = strategy_cls(**options)
    routes = strategy.create_routes(
        area_path, start_points_path, str(out), invalid)
    return routes, out


class TestAxisOrder:
    """A CRS has an official axis order, and for many national grids it is
    northing first. The planner handed pyproj (easting, northing) as
    geopandas gives them and let it apply the official order: with MGI /
    Austria Lambert, the Gauss-Kruger zones or ETRS89 LAEA the plan was
    written about 200 km from the flight area. UTM is easting first."""

    #: lon/lat of the 1 km square (UTM 33N 400000..401000 / 5260000..5261000)
    AREA = (13.67, 47.48, 13.70, 47.50)

    def test_the_transformer_gives_longitude_then_latitude(self):
        to_utm = Transformer.from_crs(4326, 32633, always_xy=True)
        to_lambert = Transformer.from_crs(4326, 31287, always_xy=True)
        lon, lat = 13.68, 47.49

        for crs, forward in ((32633, to_utm), (31287, to_lambert)):
            x, y = forward.transform(lon, lat)       # easting, northing
            assert wgs84_transformer(CRS.from_epsg(crs)).transform(x, y) == \
                pytest.approx((lon, lat), abs=1e-6)

    @pytest.mark.parametrize("epsg", [32633, 31287, 31256, 3035])
    @pytest.mark.parametrize("strategy_cls",
                             [RandomStrategy, RandomLoopStrategy])
    def test_the_plan_is_written_where_the_area_is(
            self, tmp_path, area_path, start_points_path, strategy_cls, epsg):
        _routes, out = _plan_routes(
            tmp_path, strategy_cls, area_path, start_points_path,
            grid_size=250.0, target_crs_epsg=epsg, min_transects=4,
            min_transects_per_route=2, max_distance=2000.0)

        west, south, east, north = self.AREA
        for name in ("grid.geojson", "transects.geojson",
                     "startpoints.geojson"):
            min_x, min_y, max_x, max_y = _lonlat_bounds(str(out / name))
            assert west - 0.01 < min_x and max_x < east + 0.01, (name, epsg)
            assert south - 0.01 < min_y and max_y < north + 0.01, (name, epsg)

    def test_the_routes_and_the_kml_are_there_too(
            self, tmp_path, area_path, start_points_path):
        _routes, out = _plan_routes(
            tmp_path, RandomStrategy, area_path, start_points_path,
            grid_size=250.0, target_crs_epsg=31287, min_transects=4,
            min_transects_per_route=2, max_distance=2000.0,
            max_number_of_flights=10)
        west, south, east, north = self.AREA

        route_files = sorted((out / "routes" / "valid").glob("*.geojson"))
        assert route_files
        for path in route_files:
            min_x, min_y, max_x, max_y = _lonlat_bounds(str(path))
            assert west - 0.01 < min_x and max_x < east + 0.01
            assert south - 0.01 < min_y and max_y < north + 0.01
        kml = sorted((out / "routes" / "valid_kml").glob("*.kml"))[0]
        first = kml.read_text().split("<coordinates>")[1].split(",")
        assert west - 0.01 < float(first[0]) < east + 0.01       # longitude
        assert south - 0.01 < float(first[1]) < north + 0.01     # latitude


class TestNoFly:
    LINE = shapely.geometry.LineString([(0, 0), (100, 0)])

    def _areas(self, *geometries):
        return gpd.GeoDataFrame(geometry=list(geometries), crs="EPSG:32633")

    def test_nothing_blocks_without_no_fly_geometries(self):
        assert meets_no_fly(self.LINE, None) is False
        assert meets_no_fly(self.LINE, self._areas()) is False

    def test_an_area_blocks_a_line_through_it(self):
        assert meets_no_fly(self.LINE, self._areas(
            shapely.geometry.box(40, -10, 60, 10)))

    def test_an_area_met_in_a_corner_does_not_block(self):
        assert not meets_no_fly(self.LINE, self._areas(
            shapely.geometry.box(100, 0, 120, 20)))

    def test_an_area_beside_the_line_does_not_block(self):
        assert not meets_no_fly(self.LINE, self._areas(
            shapely.geometry.box(40, 5, 60, 10)))

    def test_a_no_fly_line_blocks_what_crosses_it(self):
        """A power line drawn as a line: crossing it has no length, and
        the length of the overlap was the test."""
        assert meets_no_fly(self.LINE, self._areas(
            shapely.geometry.LineString([(50, -10), (50, 10)])))

    def test_a_no_fly_point_on_the_line_blocks(self):
        assert meets_no_fly(self.LINE, self._areas(
            shapely.geometry.Point(50, 0)))
        assert not meets_no_fly(self.LINE, self._areas(
            shapely.geometry.Point(50, 3)))

    def test_any_one_of_several_blocks(self):
        assert meets_no_fly(self.LINE, self._areas(
            shapely.geometry.box(40, 5, 60, 10),
            shapely.geometry.LineString([(70, -1), (70, 1)])))

    def test_a_no_fly_line_takes_the_transects_it_crosses(
            self, strategy, flight_area, transformer):
        power_line = self._areas(shapely.geometry.LineString([
            (AREA_MIN_X - 50, AREA_MIN_Y + 375),
            (AREA_MAX_X + 50, AREA_MIN_Y + 375)]))
        _, _, transects, _, _ = strategy.calculate_grid(
            flight_area, 250.0, power_line, transformer,
            min_transect_overlap=0.75)

        # the transects from y = 250 up to y = 500 cross it; those along
        # y = 250 and y = 500 do not
        assert transects["1_1_1_2"].is_valid is False
        assert transects["2_1_2_2"].is_valid is False
        assert transects["1_1_2_1"].is_valid is True
        assert transects["1_2_2_2"].is_valid is True

    # -- the way to the route and back ---------------------------------------

    START = [7, AREA_MIN_X - 500.0, AREA_MIN_Y + 500.0]     # [index, x, y]
    FIRST = (AREA_MIN_X + 250.0, AREA_MIN_Y + 500.0, 1, 2)
    LAST = (AREA_MIN_X + 250.0, AREA_MIN_Y + 750.0, 1, 3)

    def test_a_leg_is_tested_from_where_the_start_point_is(self, strategy):
        """The loop strategy took the start point's index for its x and its
        x for its y: the tested line began near the origin of the CRS."""
        wall = self._areas(shapely.geometry.box(
            AREA_MIN_X - 250, AREA_MIN_Y, AREA_MIN_X - 200, AREA_MAX_Y))
        assert strategy.blocked_leg(
            self.START, self.FIRST, self.LAST, wall) == "start"

    def test_the_way_back_is_tested_as_well(self, strategy):
        # only the line to the route's last point passes through this one
        patch = self._areas(shapely.geometry.box(
            AREA_MIN_X - 150, AREA_MIN_Y + 620, AREA_MIN_X - 100,
            AREA_MIN_Y + 700))
        assert strategy.blocked_leg(
            self.START, self.FIRST, self.LAST, patch) == "end"

    def test_a_free_way_is_not_blocked(self, strategy):
        far = self._areas(shapely.geometry.box(
            AREA_MAX_X + 500, AREA_MAX_Y + 500, AREA_MAX_X + 600,
            AREA_MAX_Y + 600))
        assert strategy.blocked_leg(self.START, self.FIRST, self.LAST,
                                    far) == ""
        assert strategy.blocked_leg(self.START, self.FIRST, self.LAST,
                                    None) == ""

    @pytest.mark.parametrize("strategy_cls",
                             [RandomStrategy, RandomLoopStrategy])
    def test_no_route_is_reached_across_a_no_fly_wall(
            self, tmp_path, area_path, strategy_cls):
        """The only start point lies west of the area, and a no-fly strip
        runs between the two: no route can be flown from it."""
        start = _write_geojson(tmp_path / "west.geojson", [{
            "type": "Feature", "properties": {},
            "geometry": {"type": "Point", "coordinates": _utm_to_lonlat_ring(
                [(AREA_MIN_X - 500, AREA_MIN_Y + 500)])[0]}}])
        ring = _utm_to_lonlat_ring([
            (AREA_MIN_X - 250, AREA_MIN_Y - 2000),
            (AREA_MIN_X - 200, AREA_MIN_Y - 2000),
            (AREA_MIN_X - 200, AREA_MAX_Y + 2000),
            (AREA_MIN_X - 250, AREA_MAX_Y + 2000),
            (AREA_MIN_X - 250, AREA_MIN_Y - 2000)])
        wall = _write_geojson(tmp_path / "wall.geojson", [{
            "type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [ring]}}])

        _routes, out = _plan_routes(
            tmp_path, strategy_cls, area_path, start,
            invalid_areas_path=wall, grid_size=250.0, min_transects=4,
            min_transects_per_route=2, max_distance=2000.0,
            max_start_and_stop_distance=5000.0, number_of_retries=3)

        valid = out / "routes" / "valid"
        assert not valid.is_dir() or not list(valid.glob("*.geojson"))
        assert "intersection with no-fly area" in (out / "log.txt").read_text()


class TestTheWholeArea:
    def test_every_polygon_of_the_layer_is_planned_for(self, strategy):
        """Only the first feature was used."""
        areas = gpd.GeoDataFrame(geometry=[
            shapely.geometry.box(0, 0, 1000, 1000),
            shapely.geometry.box(3000, 0, 4000, 1000)], crs="EPSG:32633")
        whole = strategy.whole_area(areas)

        assert whole.bounds == (0.0, 0.0, 4000.0, 1000.0)
        assert shapely.geometry.Point(3500, 500).within(whole)
        assert not shapely.geometry.Point(2000, 500).within(whole)

    def test_one_polygon_is_taken_as_it_is(self, strategy, flight_area):
        areas = gpd.GeoDataFrame(geometry=[flight_area], crs="EPSG:32633")
        assert strategy.whole_area(areas).equals(flight_area)

    def test_an_empty_area_file_is_said_to_be_empty(
            self, strategy, tmp_path, start_points_path):
        empty = _write_geojson(tmp_path / "empty.geojson", [])
        with pytest.raises(ValueError, match="no geometry"):
            strategy.read_data(empty, start_points_path,
                               strategy.get_crs(32633))


def _run_plan(strategy_name, options, area, start, out):
    """Module level, so it can run in a process of its own."""
    from bambi_wildlife_detection import evaluation_flight_strategy as module

    getattr(module, strategy_name)(**options).create_routes(area, start, out)


def _finishes(strategy_name, options, area, start, out, seconds=60):
    """Whether planning returns within *seconds* - run in its own process,
    so one that does not is stopped instead of running on."""
    import multiprocessing

    process = multiprocessing.get_context("spawn").Process(
        target=_run_plan, args=(strategy_name, options, area, start, out))
    process.start()
    process.join(seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        return False
    return process.exitcode == 0


class TestPlanningEnds:
    def test_random_planning_ends_when_no_route_is_long_enough(
            self, tmp_path, area_path, start_points_path):
        """Every route is drawn on a fresh grid, so there was always a point
        to start the next try from - and nothing but a valid route ended the
        loop. It ran until the disk was full of log."""
        out = tmp_path / "plan"
        out.mkdir()
        assert _finishes("RandomStrategy", dict(
            grid_size=400.0, min_transects=5, min_transects_per_route=50,
            max_number_of_flights=5, number_of_retries=20, seed=1),
            area_path, start_points_path, str(out))

        log = (out / "log.txt").read_text()
        assert "Stopping: 20 routes in a row" in log
        assert "Could not create a valid plan" in log
        assert len(log) < 200_000

    def test_loop_planning_ends_when_the_distance_is_too_short(
            self, tmp_path, area_path, start_points_path):
        """A maximum distance that ends every route before it has the
        minimum number of transects: such a try did not count as a failed
        one."""
        out = tmp_path / "plan"
        out.mkdir()
        assert _finishes("RandomLoopStrategy", dict(
            grid_size=250.0, min_transects=5, min_transects_per_route=4,
            max_distance=600.0, number_of_retries=2,
            number_of_retries_per_route=10, seed=1),
            area_path, start_points_path, str(out))
        assert "too short" in (out / "log.txt").read_text()

    def test_a_discarded_try_gives_its_transects_back(self, strategy):
        """They were "freed" in the pristine dictionary and stayed used in
        the copy the plan is drawn on."""
        pristine = {"0_0_0_1": TransectStatus(True, False),
                    "0_1_0_2": TransectStatus(True, False)}
        working = {key: status._replace(already_visited=True)
                   for key, status in pristine.items()}
        grid = [[[0.0, 0.0, True, True], [0.0, 1.0, True, True],
                 [0.0, 2.0, True, True]]]

        strategy.release_try(grid, working, [[0, 0], [0, 1]], ["0_0_0_1"])

        assert working["0_0_0_1"].already_visited is False
        assert working["0_1_0_2"].already_visited is True     # not its own
        assert [node[3] for node in grid[0]] == [False, False, True]

    def test_a_plan_still_comes_out_of_a_normal_area(
            self, tmp_path, area_path, start_points_path):
        routes, out = _plan_routes(
            tmp_path, RandomLoopStrategy, area_path, start_points_path,
            grid_size=250.0, min_transects=6, min_transects_per_route=2,
            max_distance=2000.0)
        assert routes
        assert list((out / "routes" / "valid").glob("*.geojson"))
