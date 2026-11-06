# -*- coding: utf-8 -*-
"""Survey plot and exclusion zones (core.survey_area) and the analytics that
honour them.

A census is reported for a defined area, not for whatever ground the flight
happened to see. These tests pin what "inside" means, that the area figures
are plot minus exclusions, and that each analytic both drops the animals
outside and shrinks its surveyed area to match - counting one without the
other would bias the density.
"""
import json
import os

import pytest

pytest.importorskip("shapely")

from shapely.geometry import Polygon, box  # noqa: E402

from bambi_wildlife_detection.bambi_processing import BambiProcessor  # noqa: E402
from bambi_wildlife_detection.core import (  # noqa: E402
    analytics_source, gpkg, survey_area)
from bambi_wildlife_detection.core.survey import (  # noqa: E402
    Survey, SurveyFlight)

from .test_analytics_source import survey as source_survey  # noqa: E402,F401
from .test_population_step import flight  # noqa: E402,F401

EPSG = 32633


def _area(folder, plots=(), exclusions=(), epsg=EPSG):
    survey_area.create(folder, epsg, plots=plots, exclusions=exclusions)
    return survey_area.load(folder, epsg)


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------

class TestFile:
    def test_no_file_means_no_area(self, tmp_path):
        assert survey_area.load(str(tmp_path), EPSG) is None

    def test_an_empty_file_means_no_area(self, tmp_path):
        """'Draw plot' pressed and cancelled must not switch anything on."""
        survey_area.create(str(tmp_path), EPSG)
        assert survey_area.load(str(tmp_path), EPSG) is None

    def test_the_file_is_a_feature_geopackage(self, tmp_path):
        path = survey_area.create(str(tmp_path), EPSG)
        assert gpkg.is_geopackage(path)
        conn = gpkg.connect(path, create=False)
        try:
            rows = {r["table_name"]: (r["data_type"], r["srs_id"])
                    for r in conn.execute(
                        "SELECT table_name, data_type, srs_id "
                        "FROM gpkg_contents")}
            geometry = {r["table_name"]: r["geometry_type_name"]
                        for r in conn.execute(
                            "SELECT * FROM gpkg_geometry_columns")}
        finally:
            conn.close()
        assert rows == {"plot": ("features", EPSG),
                        "exclusion_zones": ("features", EPSG)}
        assert geometry == {"plot": "MULTIPOLYGON",
                            "exclusion_zones": "MULTIPOLYGON"}

    def test_create_keeps_existing_polygons(self, tmp_path):
        """It runs before every 'Draw' and must never wipe the plot."""
        folder = str(tmp_path)
        survey_area.create(folder, EPSG, plots=[(box(0, 0, 10, 10), "A")])
        survey_area.create(folder, EPSG)
        assert survey_area.load(folder, EPSG).plot_m2 == pytest.approx(100.0)

    def test_geometry_blob_round_trip(self):
        shape = Polygon([(0, 0), (4, 0), (4, 3)])
        blob = survey_area.encode_geometry(shape, EPSG)
        assert blob[:2] == b"GP"
        assert survey_area.decode_geometry(blob).equals(shape)

    def test_names_and_reasons_are_kept(self, tmp_path):
        area = _area(str(tmp_path), plots=[(box(0, 0, 10, 10), "District")],
                     exclusions=[(box(1, 1, 2, 2), "Wood", "thick forest")])
        info = area.describe()
        assert info["plots"] == ["District"]
        assert info["exclusions"] == [{"name": "Wood",
                                       "reason": "thick forest"}]

    def test_polygons_in_another_crs_are_reprojected(self, tmp_path):
        pytest.importorskip("pyproj")
        folder = str(tmp_path)
        # 1 km square in UTM 33N, read back in UTM 32N: same area, moved.
        plot = box(500000, 5300000, 501000, 5301000)
        survey_area.create(folder, 32633, plots=[(plot, "")])
        area = survey_area.load(folder, 32632)
        assert area.plot_m2 == pytest.approx(1e6, rel=0.01)
        assert area.geometry.bounds[0] > 800000


# ---------------------------------------------------------------------------
# What counts, and how big the area is
# ---------------------------------------------------------------------------

class TestArea:
    @pytest.fixture
    def area(self, tmp_path):
        return _area(str(tmp_path), plots=[(box(0, 0, 100, 100), "")],
                     exclusions=[(box(10, 10, 30, 30), "", "forest"),
                                 (box(90, 90, 120, 120), "", "outside")])

    def test_adjusted_is_plot_minus_the_exclusions_inside_it(self, area):
        assert area.plot_m2 == pytest.approx(10000.0)
        # The second zone sticks out of the plot: only its 10x10 inside counts.
        assert area.excluded_m2 == pytest.approx(400.0 + 100.0)
        assert area.adjusted_m2 == pytest.approx(9500.0)

    def test_points_inside_outside_and_excluded(self, area):
        inside = area.contains_xy([50, 20, 150, 95], [50, 20, 50, 95])
        assert inside.tolist() == [True, False, False, False]

    def test_a_point_on_the_plot_boundary_counts(self, area):
        assert area.contains(100.0, 50.0)

    def test_filter_reports_what_it_dropped(self, area):
        kept, dropped = area.filter_points([(50, 50), (20, 20), (500, 500)])
        assert kept == [(50, 50)]
        assert dropped == 2

    def test_clipped_length(self, area):
        # y = 50 from x = -50 to 150: 100 m inside the plot, no exclusion hit.
        assert area.clipped_length([(-50, 50), (150, 50)]) == \
            pytest.approx(100.0)
        # y = 20 crosses the forest zone: 100 - 20 m.
        assert area.clipped_length([(-50, 20), (150, 20)]) == \
            pytest.approx(80.0)

    def test_exclusions_without_a_plot_cut_out_of_the_whole_flight(
            self, tmp_path):
        area = _area(str(tmp_path),
                     exclusions=[(box(0, 0, 10, 10), "", "")])
        assert area.geometry is None
        assert area.adjusted_m2 is None
        assert area.contains_xy([5, 50], [5, 50]).tolist() == [False, True]
        # An animal right on the zone's edge is not inside it.
        assert area.contains(10.0, 5.0)

    def test_repeat_flights_over_one_plot_count_its_area_once(self, tmp_path):
        first = _area(str(tmp_path / "a"), plots=[(box(0, 0, 10, 10), "")])
        again = _area(str(tmp_path / "b"), plots=[(box(0, 0, 10, 10), "")])
        beside = _area(str(tmp_path / "c"), plots=[(box(10, 0, 20, 10), "")])
        assert survey_area.union_adjusted_m2([first, again]) == \
            pytest.approx(100.0)
        assert survey_area.union_adjusted_m2([first, beside]) == \
            pytest.approx(200.0)
        assert survey_area.union_adjusted_m2([first, None]) is None


class TestSeveralSurveyAreas:
    """Any number of survey areas: inside any of them and outside every
    exclusion zone counts."""

    @pytest.fixture
    def area(self, tmp_path):
        return _area(str(tmp_path),
                     plots=[(box(0, 0, 100, 100), "North"),
                            (box(200, 0, 300, 100), "South"),
                            # Overlaps North: shared ground counts once.
                            (box(50, 0, 150, 100), "Middle")],
                     exclusions=[(box(10, 10, 30, 30), "", "forest"),
                                 (box(210, 10, 230, 30), "", "village")])

    def test_animals_in_any_area_count_but_not_in_an_exclusion(self, area):
        xs = [50, 250, 120, 175, 20, 220, 500]
        ys = [50, 50, 50, 50, 20, 20, 50]
        assert area.contains_xy(xs, ys).tolist() == \
            [True, True, True, False, False, False, False]

    def test_the_area_is_the_union_minus_the_exclusions(self, area):
        # North+Middle = 150 x 100, South = 100 x 100.
        assert area.plot_m2 == pytest.approx(25000.0)
        assert area.excluded_m2 == pytest.approx(800.0)
        assert area.adjusted_m2 == pytest.approx(24200.0)

    def test_every_area_is_listed(self, area):
        assert area.describe()["plots"] == ["North", "South", "Middle"]
        assert area.summary().startswith("3 survey area(s), 2.50 ha")

    def test_census_counts_across_the_areas(self, area):
        from bambi_wildlife_detection.core import census

        records = [{"species": "roe deer", "x": x, "y": y}
                   for x, y in [(50, 50), (250, 50), (175, 50), (20, 20)]]
        result = census.build([{"name": "F1", "records": records,
                                "covered_m2": None}], area)
        row = next(r for r in result["rows"] if r["species"] == "roe deer")
        assert (row["counted"], row["inside"], row["outside"]) == (4, 2, 2)
        assert row["per_km2_plot"] == pytest.approx(2 / 0.0242)


# ---------------------------------------------------------------------------
# analytics_source
# ---------------------------------------------------------------------------

class TestAnalyticsSource:
    # Detection centres of the fixture: (1, 1) and (11, 21) for track 1 (roe
    # deer), (21, 41) for track 2 (chamois).
    @pytest.fixture
    def area(self, source_survey):  # noqa: F811
        return _area(source_survey, plots=[(box(0, 0, 15, 30), "")])

    def test_detections_outside_are_dropped(self, source_survey, area):  # noqa: F811
        points, provenance = analytics_source.load_points(
            source_survey, "t", area=area)
        assert sorted(points) == [(1.0, 1.0), (11.0, 21.0)]
        assert provenance["outside_survey_area"] == 1
        assert provenance["survey_area"]["plot_ha"] == pytest.approx(0.045)

    def test_a_track_counts_by_its_centroid(self, source_survey, area):  # noqa: F811
        points, provenance = analytics_source.load_points(
            source_survey, "t", source="tracks", area=area)
        assert points == [(6.0, 11.0)]
        assert provenance["n_tracks"] == 1

    def test_without_an_area_nothing_changes(self, source_survey):  # noqa: F811
        points, provenance = analytics_source.load_points(source_survey, "t")
        assert len(points) == 3
        assert "survey_area" not in provenance


# ---------------------------------------------------------------------------
# The analytics: the survey area belongs to the survey, not to a flight
# ---------------------------------------------------------------------------

def _survey(survey_folder, *flight_folders, area_enabled=True):
    return Survey(folder=survey_folder, modality="t", area_enabled=area_enabled,
                  flights=[SurveyFlight(name=os.path.basename(f),
                                        target_folder=f, epsg=EPSG)
                           for f in flight_folders])


def test_the_filter_is_off_unless_asked_for(tmp_path):
    _area(str(tmp_path), plots=[(box(0, 0, 1, 1), "")])
    survey = _survey(str(tmp_path), str(tmp_path), area_enabled=False)
    assert BambiProcessor()._survey_area(survey) is None


def test_exclusions_covering_the_plot_are_an_error(tmp_path):
    _area(str(tmp_path), plots=[(box(0, 0, 10, 10), "")],
          exclusions=[(box(-1, -1, 11, 11), "", "")])
    with pytest.raises(RuntimeError, match="whole survey area"):
        BambiProcessor()._survey_area(_survey(str(tmp_path), str(tmp_path)))


def test_the_area_lives_in_the_survey_folder(tmp_path):
    """A plot drawn in a flight's folder does not apply to the survey."""
    folder = str(tmp_path / "flight")
    _area(folder, plots=[(box(0, 0, 10, 10), "")])
    survey = _survey(str(tmp_path / "survey"), folder)
    assert BambiProcessor()._survey_area(survey) is None


def test_density_heatmap_is_masked_to_the_area(source_survey,  # noqa: F811
                                               tmp_path):
    rasterio = pytest.importorskip("rasterio")
    import numpy as np

    survey_folder = str(tmp_path / "survey")
    _area(survey_folder, plots=[(box(0, 0, 15, 30), "")])
    BambiProcessor().run_density_heatmap({
        "survey": _survey(survey_folder, source_survey).to_dict(),
        "target_epsg": EPSG, "density_source": "detections",
        "density_cell_size": 1.0, "density_bandwidth": 5.0})

    folder = os.path.join(survey_folder, "analytics_t")
    with open(os.path.join(folder, "density_detections.json"),
              encoding="utf-8") as fh:
        stats = json.load(fh)
    assert stats["n_points"] == 2
    assert stats["points_per_ha_in_survey_area"] == pytest.approx(2 / 0.045)
    assert stats["population_filter"]["outside_survey_area"] == 1

    with rasterio.open(os.path.join(folder, "density_detections.tif")) as src:
        data = src.read(1)
        transform = src.transform
    rows, cols = np.nonzero(data != -9999.0)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    assert max(xs) <= 15.0 and max(ys) <= 30.0
    assert min(xs) >= 0.0 and min(ys) >= 0.0


def _distance_sampling_project(folder):
    """A 200 m route along y = 0 with five detections at known distances."""
    route = os.path.join(folder, "flight_route_t")
    os.makedirs(route, exist_ok=True)
    with open(os.path.join(route, "flight_route.geojson"), "w",
              encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": [{
            "type": "Feature", "properties": {},
            "geometry": {"type": "LineString",
                         "coordinates": [[0, 0], [200, 0]]}}]}, fh)
    entries = [{"class_id": 1, "detection_center": [x, d, 0],
                "distance_m": float(d)}
               for x, d in ((20, 2), (40, 5), (60, 9), (150, 4), (170, 7))]
    with open(os.path.join(route, "perpendicular_t.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"perpendiculas": entries}, fh)


def test_distance_sampling_clips_observations_and_effort(tmp_path):
    folder = str(tmp_path / "flight")
    _distance_sampling_project(folder)
    survey_folder = str(tmp_path / "survey")
    # The plot covers the first 100 m of route and 50 m either side.
    _area(survey_folder, plots=[(box(0, -50, 100, 50), "")])

    survey = _survey(survey_folder, folder)
    distances, length = BambiProcessor()._load_distance_sampling_project(
        survey.flights[0], survey, "detections",
        area=survey_area.load(survey_folder, EPSG))
    assert sorted(distances.tolist()) == [2.0, 5.0, 9.0]
    assert length == pytest.approx(100.0)


def test_distance_sampling_reports_abundance_in_the_survey_area(tmp_path):
    pytest.importorskip("scipy")
    folder = str(tmp_path / "flight")
    _distance_sampling_project(folder)
    survey_folder = str(tmp_path / "survey")
    _area(survey_folder, plots=[(box(0, -50, 100, 50), "")])

    BambiProcessor().run_distance_sampling({
        "survey": _survey(survey_folder, folder).to_dict(),
        "target_epsg": EPSG, "ds_source": "detections",
        "ds_truncation": 10.0})
    with open(os.path.join(survey_folder, "analytics_t",
                           "distance_sampling_detections.json"),
              encoding="utf-8") as fh:
        result = json.load(fh)
    assert result["transect_length_m"] == pytest.approx(100.0)
    assert result["survey_area_km2"] == pytest.approx(0.01)
    assert result["abundance_in_survey_area"] == pytest.approx(
        result["density_per_km2"] * 0.01)
    assert result["projects"][0]["outside_survey_area"] == 2


def test_distance_sampling_pools_flights_over_one_plot(tmp_path):
    """Two flights, one survey area: both clipped, effort summed."""
    pytest.importorskip("scipy")
    first = str(tmp_path / "a")
    second = str(tmp_path / "b")
    _distance_sampling_project(first)
    _distance_sampling_project(second)
    survey_folder = str(tmp_path / "survey")
    _area(survey_folder, plots=[(box(0, -50, 100, 50), "")])

    BambiProcessor().run_distance_sampling({
        "survey": _survey(survey_folder, first, second).to_dict(),
        "target_epsg": EPSG, "ds_source": "detections",
        "ds_truncation": 10.0})
    with open(os.path.join(survey_folder, "analytics_t",
                           "distance_sampling_detections.json"),
              encoding="utf-8") as fh:
        result = json.load(fh)
    assert result["n"] == 6
    assert result["transect_length_m"] == pytest.approx(200.0)
    assert result["flights"] == ["a", "b"]
    # The plot is counted once, not once per flight.
    assert result["survey_area_km2"] == pytest.approx(0.01)


class TestPopulation:
    """Fixture flight: transect A along y = 0, transect B along y = 100
    (world: origin + ...), tracks 1-3 near A, 4 near B, 5 far north."""

    @staticmethod
    def _run(config, **extra):
        config = dict(config, **extra)
        config["survey"] = dict(config["survey"], area_enabled=True)
        BambiProcessor().run_population_estimation(config)
        with open(os.path.join(config["survey_folder"], "analytics_t",
                               "population_estimate.json"),
                  encoding="utf-8") as fh:
            return json.load(fh)

    def test_tracks_and_ground_outside_the_plot_are_not_counted(
            self, flight):  # noqa: F811
        ox, oy = 500000.0, 5000000.0
        # Only the west half of transect A: x in [-10, 100), y in [-10, 10].
        _area(flight["survey_folder"],
              plots=[(box(ox - 10, oy - 10, ox + 100, oy + 10), "")])
        result = self._run(flight)

        by_name = {t["name"]: t for t in result["transects"]}
        north = by_name["North"]
        assert north["count"] == 2          # tracks 1 and 2; 3 is at x = 120
        assert north["area_m2"] == pytest.approx(110.0 * 20.0)
        assert result["n_tracks_outside_survey_area"] == 3

    def test_the_survey_area_can_be_the_study_area(self, flight):  # noqa: F811
        ox, oy = 500000.0, 5000000.0
        _area(flight["survey_folder"],
              plots=[(box(ox - 10, oy - 10, ox + 200, oy + 110), "")],
              exclusions=[(box(ox - 10, oy + 20, ox + 200, oy + 90), "",
                           "field")])
        result = self._run(flight, pop_study_area_survey=True)
        assert result["study_area_source"] == "survey_area"
        assert result["study_area_ha"] == pytest.approx(
            (210 * 120 - 210 * 70) / 10000.0)
