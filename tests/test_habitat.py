# -*- coding: utf-8 -*-
"""Habitat context (core/habitat.py): sampling the environment at the
tracks, the monitored ground it is compared with, and the survey files."""
import csv
import json
import os

import numpy as np
import pytest

from bambi_wildlife_detection.core import (
    detection_store, environment as env, fov_store, habitat, label_store,
    store, track_store)

GRID = env.make_grid((500000.0, 5300000.0, 500200.0, 5300200.0), 32633, 0.0)


class TestSampling:
    def test_mean_within_a_radius(self):
        values = np.zeros(GRID.shape, dtype=np.float32)
        values[:, 10:] = 1.0     # east half
        west, edge, east, off = habitat.sample_mean(
            values, GRID, [(500030.0, 5300100.0), (500100.0, 5300100.0),
                           (500170.0, 5300100.0), (499000.0, 5300100.0)], 30.0)
        assert west == 0.0 and east == 1.0 and off is None
        assert edge == pytest.approx(0.5)

    def test_zero_radius_takes_the_cell_under_the_point(self):
        values = np.arange(GRID.width * GRID.height,
                           dtype=np.float32).reshape(GRID.shape)
        # column 3, row 1 (rows count from the top)
        found = habitat.sample_mean(
            values, GRID, [(500037.0, 5300183.0)], 0.0)
        assert found == [values[1, 3]]

    def test_no_data_is_ignored_and_all_no_data_is_none(self):
        values = np.full(GRID.shape, np.nan, dtype=np.float32)
        values[10, 10] = 4.0
        centre = (500105.0, 5300095.0)
        assert habitat.sample_mean(values, GRID, [centre], 30.0) == [4.0]
        assert habitat.sample_mean(
            values, GRID, [(500005.0, 5300005.0)], 10.0) == [None]

    def test_most_common_class_and_its_share(self):
        classes = np.full(GRID.shape, 30, dtype=np.uint8)
        classes[:, 10:] = 10
        classes[0, 0] = 0
        (west, edge) = habitat.sample_classes(
            classes, GRID, [(500030.0, 5300100.0), (500095.0, 5300100.0)], 30.0)
        assert west == (30, 1.0)
        assert edge[0] == 30 and 0.5 < edge[1] < 1.0
        assert habitat.sample_classes(
            np.zeros(GRID.shape, dtype=np.uint8), GRID,
            [(500030.0, 5300100.0)], 20.0) == [None]

    def test_histogram_keeps_out_of_range_values(self):
        edges = [0.0, 0.5, 1.0]
        assert habitat.histogram([-3.0, 0.2, 0.7, 9.0, float("nan")],
                                 edges) == [2, 2]
        assert habitat.histogram([], edges) == [0, 0]
        assert habitat.histogram([1.0], []) == []

    def test_bin_edges_use_the_documented_range_or_the_data(self):
        ndvi = env.LAYERS_BY_KEY["ndvi"]
        edges = habitat.bin_edges(ndvi, [np.array([0.3])])
        assert edges[0] == -0.2 and edges[-1] == 1.0
        assert len(edges) == habitat.HISTOGRAM_BINS + 1
        twi = env.LAYERS_BY_KEY["twi"]
        values = np.concatenate([np.linspace(4.0, 12.0, 1000), [90.0]])
        edges = habitat.bin_edges(twi, [values])
        assert edges[0] == pytest.approx(4.08, abs=0.01)
        assert edges[-1] < 12.1          # the outlier does not set the range
        assert habitat.histogram(values, edges)[-1] >= 1   # but is counted
        assert habitat.bin_edges(twi, [np.array([np.nan])]) == []

    def test_geometry_mask(self):
        from shapely.geometry import box

        mask = habitat.geometry_mask(
            GRID, box(500000.0, 5300000.0, 500100.0, 5300200.0))
        assert mask[:, :10].all() and not mask[:, 10:].any()
        assert habitat.geometry_mask(GRID, None).all()

    def test_selection_ratio(self):
        assert habitat.selection_ratio(0.6, 0.2) == 3.0
        assert habitat.selection_ratio(None, 0.2) is None
        assert habitat.selection_ratio(0.5, 0.0) is None

    def test_selection_ratio_standard_error(self):
        # Manly: sqrt(o (1 - o) / (n pi^2)), the ground's share taken as known
        assert habitat.selection_ratio_se(0.6, 0.2, 100) == pytest.approx(
            (0.6 * 0.4 / (100 * 0.04)) ** 0.5, abs=1e-3)
        # four times the animals, half the uncertainty
        assert habitat.selection_ratio_se(0.6, 0.2, 400) == pytest.approx(
            habitat.selection_ratio_se(0.6, 0.2, 100) / 2.0, abs=1e-3)
        assert habitat.selection_ratio_se(None, 0.2, 10) is None
        assert habitat.selection_ratio_se(0.5, 0.0, 10) is None
        assert habitat.selection_ratio_se(0.5, 0.2, 0) is None


class TestSameGrain:
    """The ground is measured the way the animals are: every cell by its
    surroundings within the radius."""

    def _patchy(self):
        rng = np.random.default_rng(7)
        values = rng.normal(0.4, 0.2, GRID.shape).astype(np.float32)
        values[3, 4] = np.nan
        classes = np.full(GRID.shape, 30, dtype=np.uint8)
        classes[:, 12:] = 10
        classes[5:8, 2:5] = 20
        classes[:, 9] = 50            # a road one cell wide
        classes[0, 0] = 0
        return values, classes

    def test_disc_offsets(self):
        assert habitat.disc_offsets(0.0) == [(0, 0)]
        assert habitat.disc_offsets(0.4) == [(0, 0)]
        assert sorted(habitat.disc_offsets(1.0)) == [
            (-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
        assert len(habitat.disc_offsets(3.0)) == 29

    @pytest.mark.parametrize("reach", [0.0, 1.0, 2.5, 3.0, 7.3])
    def test_disc_sum_adds_up_the_cells_of_the_disc(self, reach):
        rng = np.random.default_rng(3)
        values = rng.integers(0, 9, (11, 17)).astype(np.int32)
        expected = np.zeros_like(values)
        for d_row, d_col in habitat.disc_offsets(reach):
            for row in range(values.shape[0]):
                for col in range(values.shape[1]):
                    r, c = row + d_row, col + d_col
                    if 0 <= r < values.shape[0] and 0 <= c < values.shape[1]:
                        expected[row, col] += values[r, c]
        assert np.array_equal(habitat.disc_sum(values, reach), expected)

    def test_a_radius_wider_than_the_raster(self):
        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        assert np.allclose(habitat.focal_mean(values, 40.0), values.mean())
        classes = np.array([[10, 10, 30], [10, 30, 30]], dtype=np.uint8)
        # three cells each: the lower class wins the tie everywhere
        assert (habitat.focal_majority(classes, 40.0) == 10).all()

    def test_the_largest_radius_stays_quick(self):
        import time
        rng = np.random.default_rng(5)
        classes = rng.choice([10, 20, 30, 40, 50, 60, 80, 90],
                             (400, 400)).astype(np.uint8)
        start = time.perf_counter()
        habitat.focal_majority(classes, 50.0)      # 500 m on a 10 m grid
        habitat.focal_mean(classes.astype(np.float32), 50.0)
        assert time.perf_counter() - start < 20.0

    @pytest.mark.parametrize("radius", [0.0, 10.0, 30.0, 45.0])
    def test_a_cell_gets_what_an_animal_at_its_centre_gets(self, radius):
        values, classes = self._patchy()
        xs, ys = GRID.cell_centres()
        mean = habitat.focal_mean(values, radius / GRID.resolution)
        majority = habitat.focal_majority(classes, radius / GRID.resolution)
        for row, col in [(0, 0), (3, 4), (6, 3), (10, 9), (19, 19), (8, 12)]:
            point = [(float(xs[col]), float(ys[row]))]
            (at_animal,) = habitat.sample_mean(values, GRID, point, radius)
            if at_animal is None:
                assert np.isnan(mean[row, col])
            else:
                assert mean[row, col] == pytest.approx(at_animal)
            (found,) = habitat.sample_classes(classes, GRID, point, radius)
            assert majority[row, col] == (0 if found is None else found[0])

    def test_animals_placed_at_random_show_no_selection(self, tmp_path,
                                                        monkeypatch):
        """The check on the whole comparison: with animals scattered
        uniformly over the monitored ground every class has to come out near
        1. Measured against single cells of the ground, small patches and
        narrow strips looked avoided (0.6 and 0.0) whatever the animals did -
        they are rarely the most common class around a point."""
        from bambi_wildlife_detection.core import env_landcover

        grid = env.make_grid((500000.0, 5300000.0, 502000.0, 5302000.0),
                             32633, 0.0)
        rng = np.random.default_rng(11)
        classes = np.full(grid.shape, 30, dtype=np.uint8)       # grassland
        classes[:, 120:] = 10                                    # forest
        for _ in range(60):                                      # shrub patches
            row, col = rng.integers(0, 190), rng.integers(0, 110)
            classes[row:row + 4, col:col + 4] = 20
        classes[:, 60] = 50                                      # a 10 m road
        classes[100, :] = 80                                     # a 10 m stream
        root = str(tmp_path)
        env.write_raster(env.layer_path(root, "landcover"), classes, grid,
                         nodata=0)
        env.write_metadata(root, env.LANDCOVER, {"files": ["landcover.tif"]})

        points = rng.uniform([500000.0, 5300000.0], [502000.0, 5302000.0],
                             size=(6000, 2))
        monkeypatch.setattr(habitat, "flight_tracks", lambda *a, **k: [
            {"track_id": i, "species": "roe deer", "sex": "", "age": "",
             "x": float(x), "y": float(y)} for i, (x, y) in enumerate(points)])
        monkeypatch.setattr(habitat, "monitored_geometry",
                            lambda *a, **k: None)
        flight = habitat.summarise_flight("A", root, "t", 30.0)
        summary = habitat.build_summary([flight], 30.0, "t")

        rows = {c["name"]: c for c in summary["landcover"]["classes"]}
        assert {"Grassland", "Tree cover", "Shrubland"} <= set(rows)
        for name, row in rows.items():
            if row["available_share"] < 0.01:
                continue          # too rare at this grain to say anything
            ratio, error = row["selection_ratio"], row["selection_ratio_se"]
            assert abs(ratio - 1.0) < max(0.1, 4.0 * error), (name, row)
        # the one-cell road and stream are no patch's most common class - on
        # either side, so they do not show up as "avoided"
        for name in (env_landcover.class_name(50), env_landcover.class_name(80)):
            assert name not in rows or rows[name]["available_share"] < 0.01


def _flight(root, positions, species="roe deer"):
    """A flight with one track per position, an FoV covering the west half
    and environment rasters: NDVI 0.8 in the west, 0.2 in the east; forest
    in the west, grassland in the east."""
    os.makedirs(root, exist_ok=True)
    label_store.vocabulary(root, create=True)
    names = label_store.vocabulary(root)["species_by_name"]
    # Map the detector's class to the species, so it resolves on the way in.
    project = store.open_store(store.project_path(root), store.PROJECT)
    source_id = detection_store.ensure_source(project, detection_store.DETECTOR)
    project.execute(
        "INSERT INTO class_mapping (source_id, source_class, species_id) "
        "VALUES (?, ?, ?)", (source_id, "0", names[species]))
    project.commit()
    project.close()
    detection_store.record_detections(root, "t", [
        {"frame": i, "x1": 1.0, "y1": 1.0, "x2": 5.0, "y2": 5.0,
         "confidence": 0.9, "source_class": "0"}
        for i in range(len(positions))])
    ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
    track_store.record_georeference(root, "t", [
        {"detection_id": det, "gx1": x - 1, "gy1": y - 1, "gz1": 0.0,
         "gx2": x + 1, "gy2": y + 1, "gz2": 0.0}
        for det, (x, y) in zip(ids, positions)])
    track_store.record_tracks(root, "t", [
        {"track_id": i + 1, "detection_id": det} for i, det in enumerate(ids)],
        species_of=lambda _track: names[species])
    fov_store.record_fov(root, "t", {0: [
        (500000.0, 5300000.0, 0.0), (500150.0, 5300000.0, 0.0),
        (500150.0, 5300200.0, 0.0), (500000.0, 5300200.0, 0.0)]})

    ndvi = np.full(GRID.shape, 0.2, dtype=np.float32)
    ndvi[:, :10] = 0.8
    classes = np.full(GRID.shape, 30, dtype=np.uint8)
    classes[:, :10] = 10
    env.write_raster(env.layer_path(root, "ndvi"), ndvi, GRID)
    env.write_raster(env.layer_path(root, "landcover"), classes, GRID,
                     nodata=0)
    env.write_metadata(root, env.SENTINEL2, {
        "files": ["ndvi.tif"], "scene": "S2X", "offset_days": -1.5,
        "scene_time": "2026-06-08T10:17:35Z", "clear_fraction": 0.97})
    env.write_metadata(root, env.LANDCOVER, {"files": ["landcover.tif"]})
    env.write_metadata(root, env.WEATHER, {
        "files": [], "flight_start": "2026-06-10T03:10:00Z",
        "at_flight": {"temperature_2m": 9.5, "soil_temperature": 12.0,
                      "relative_humidity_2m": 91.0, "wind_speed_10m": 1.2},
        "previous_24h": {"precipitation_sum": 3.4, "sunshine_hours": 6.0}})
    return root


WEST = (500040.0, 5300100.0)
EAST = (500130.0, 5300100.0)
OUTSIDE_FOV = (500180.0, 5300100.0)


class TestFlight:
    def test_one_row_per_track_with_layers_and_weather(self, tmp_path):
        root = _flight(str(tmp_path), [WEST, WEST, EAST])
        result = habitat.summarise_flight("A", root, "t", 30.0, epsg=32633)
        rows = result["tracks"]
        assert [row["track_id"] for row in rows] == [1, 2, 3]
        assert rows[0]["ndvi"] == pytest.approx(0.8)
        assert rows[2]["ndvi"] == pytest.approx(0.2)
        assert rows[0]["landcover"] == "Tree cover"
        assert rows[2]["landcover"] == "Grassland"
        assert rows[0]["species"] == "roe deer"
        assert rows[0]["crs"] == "EPSG:32633"
        assert rows[0]["air_temperature_c"] == 9.5
        assert rows[0]["rain_24h_mm"] == 3.4
        assert rows[0]["scene_offset_days"] == -1.5
        assert "twi" not in rows[0]      # the terrain step has not run
        assert result["missing"] == [env.SOURCE_TITLES[env.TERRAIN]]

    def test_available_ground_is_what_the_camera_saw(self, tmp_path):
        root = _flight(str(tmp_path), [WEST])
        result = habitat.summarise_flight("A", root, "t", 30.0)
        ndvi = result["available"]["ndvi"]
        # FoV = columns 0-14: ten at 0.8, five at 0.2
        assert ndvi.size == 15 * 20
        assert float(ndvi.mean()) == pytest.approx((10 * 0.8 + 5 * 0.2) / 15)
        assert "field-of-view" in result["available_basis"]

    def test_species_filter(self, tmp_path):
        root = _flight(str(tmp_path), [WEST, EAST])
        names = label_store.vocabulary(root)["species_by_name"]
        other = next(i for name, i in names.items() if name != "roe deer")
        assert habitat.summarise_flight(
            "A", root, "t", 30.0, species_ids=[other])["tracks"] == []
        assert len(habitat.summarise_flight(
            "A", root, "t", 30.0,
            species_ids=[names["roe deer"]])["tracks"]) == 2

    def test_transects_get_zonal_means(self, tmp_path):
        root = _flight(str(tmp_path), [WEST])
        folder = os.path.join(root, "analytics_t")
        os.makedirs(folder)
        square = [[500000.0, 5300000.0], [500200.0, 5300000.0],
                  [500200.0, 5300200.0], [500000.0, 5300200.0],
                  [500000.0, 5300000.0]]
        with open(os.path.join(folder, "transect_areas.geojson"), "w") as fh:
            json.dump({"type": "FeatureCollection", "features": [{
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [square]},
                "properties": {"transect_id": 7, "name": "T7",
                               "area_ha": 4.0, "count": 1}}]}, fh)
        result = habitat.summarise_flight("A", root, "t", 30.0)
        (row,) = result["transects"]
        assert row["transect_id"] == 7 and row["flight"] == "A"
        assert row["ndvi"] == pytest.approx(0.5)
        assert row["landcover"] in ("Tree cover", "Grassland")
        assert row["landcover_share"] == 0.5


class TestSurvey:
    def _summary(self, tmp_path):
        a = _flight(str(tmp_path / "a"), [WEST, WEST, WEST, EAST])
        b = _flight(str(tmp_path / "b"), [WEST], species="red deer")
        flights = [habitat.summarise_flight("A", a, "t", 30.0, epsg=32633),
                   habitat.summarise_flight("B", b, "t", 30.0, epsg=32633)]
        return flights, habitat.build_summary(flights, 30.0, "t")

    def test_used_against_available(self, tmp_path):
        _flights, summary = self._summary(tmp_path)
        assert summary["n_tracks"] == 5
        ndvi = summary["layers"]["ndvi"]
        assert ndvi["used"]["n"] == 5
        assert ndvi["used"]["mean"] == pytest.approx((4 * 0.8 + 0.2) / 5)
        assert ndvi["available"]["mean"] == pytest.approx(0.6)
        assert sum(ndvi["used_counts"]) == 5
        assert sum(ndvi["available_counts"]) == 2 * 15 * 20
        assert len(ndvi["bin_edges"]) == habitat.HISTOGRAM_BINS + 1
        assert "landcover" not in summary["layers"]   # it is not a measure

    def test_land_cover_selection(self, tmp_path):
        _flights, summary = self._summary(tmp_path)
        classes = {c["name"]: c for c in summary["landcover"]["classes"]}
        assert set(classes) == {"Tree cover", "Grassland"}
        assert classes["Tree cover"]["available_share"] == pytest.approx(
            2 / 3, abs=1e-3)
        assert classes["Tree cover"]["used_share"] == 0.8
        assert classes["Tree cover"]["selection_ratio"] == pytest.approx(1.2)
        assert classes["Grassland"]["selection_ratio"] == pytest.approx(0.6)

    def test_flights_carry_their_scene_and_weather(self, tmp_path):
        _flights, summary = self._summary(tmp_path)
        first = summary["flights"][0]
        assert first["flight"] == "A" and first["n_tracks"] == 4
        assert first["sources"]["sentinel2"]["offset_days"] == -1.5
        assert first["sources"]["weather"]["at_flight"]["temperature_2m"] == 9.5

    def test_files(self, tmp_path):
        flights, summary = self._summary(tmp_path)
        folder = str(tmp_path / "survey" / "analytics_t")
        paths = habitat.write_outputs(folder, summary, flights)
        with open(paths["tracks_csv"], encoding="utf-8") as fh:
            table = list(csv.DictReader(fh))
        assert list(table[0]) == habitat.TRACK_COLUMNS
        assert [row["flight"] for row in table] == ["A"] * 4 + ["B"]
        assert table[4]["species"] == "red deer"
        assert table[0]["twi"] == ""
        assert habitat.read_json(paths["summary"])["n_tracks"] == 5
        assert len(habitat.read_json(paths["tracks_json"])) == 5
        assert "transects_csv" not in paths

    def test_a_stale_transect_table_is_removed(self, tmp_path):
        flights, summary = self._summary(tmp_path)
        folder = str(tmp_path / "analytics_t")
        os.makedirs(folder)
        stale = os.path.join(folder, habitat.TRANSECTS_CSV)
        open(stale, "w").close()
        habitat.write_outputs(folder, summary, flights)
        assert not os.path.exists(stale)

    def test_charts_can_be_redrawn_per_species(self, tmp_path):
        flights, summary = self._summary(tmp_path)
        tracks = [row for flight in flights for row in flight["tracks"]]
        red = habitat.filter_tracks(tracks, "red deer")
        assert len(red) == 1 and len(habitat.filter_tracks(tracks, "")) == 5
        edges = summary["layers"]["ndvi"]["bin_edges"]
        assert sum(habitat.used_histogram(red, "ndvi", edges)) == 1

    def test_describe_says_where_the_animals_stood_out(self, tmp_path):
        _flights, summary = self._summary(tmp_path)
        lines = habitat.describe(summary)
        assert lines[0].startswith("5 track(s) over 2 flight(s)")
        assert any("NDVI" in line and "0.68" in line for line in lines)
        assert any("Tree cover 80%" in line for line in lines)

    def test_no_layers_gives_an_empty_but_valid_summary(self, tmp_path):
        root = _flight(str(tmp_path), [WEST])
        for source in (env.SENTINEL2, env.LANDCOVER):
            env.clear(root, source)
        flights = [habitat.summarise_flight("A", root, "t", 30.0)]
        summary = habitat.build_summary(flights, 30.0, "t")
        assert summary["layers"] == {} and summary["landcover"] is None
        assert summary["n_tracks"] == 1
