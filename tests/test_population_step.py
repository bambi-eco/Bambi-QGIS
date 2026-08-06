# -*- coding: utf-8 -*-
"""End-to-end test of BambiProcessor.run_population_estimation.

Builds a synthetic flight on disk — poses, transect definitions, FoV
footprints and track perpendicular distances — and checks that the step wires
them together into the per-transect count/area table and the estimate files.
"""
import csv
import json
import os

import pytest

from bambi_wildlife_detection.core import fov_store, store


def _square(cx, cy, half=10.0):
    """A 20x20 m footprint centred on (cx, cy), as the FoV step records it."""
    return [(cx - half, cy - half, 0.0), (cx + half, cy - half, 0.0),
            (cx + half, cy + half, 0.0), (cx - half, cy + half, 0.0)]


@pytest.fixture
def flight(tmp_path):
    """A 2-transect flight: frames 0-19 fly east, frames 20-39 fly east 100 m north.

    The DEM origin is non-zero so the mesh-local -> world shift is exercised.
    """
    folder = tmp_path / "target"
    folder.mkdir()
    origin = [500000.0, 5000000.0, 0.0]

    # ---- poses (mesh-local: world minus origin) --------------------------
    images = []
    for i in range(20):                      # transect A: y = 0
        images.append({"imagefile": f"a{i:03d}.png",
                       "location": [i * 10.0, 0.0, 100.0],
                       "timestamp": f"2026-07-14T10:00:{i:02d}+00:00"})
    for i in range(20):                      # transect B: y = 100
        images.append({"imagefile": f"b{i:03d}.png",
                       "location": [i * 10.0, 100.0, 100.0],
                       "timestamp": f"2026-07-14T10:01:{i:02d}+00:00"})
    (folder / "poses_t.json").write_text(json.dumps({"images": images}))

    # ---- DEM metadata carrying the origin --------------------------------
    dem_json = folder / "dem.json"
    dem_json.write_text(json.dumps({"origin": origin}))

    # ---- transects -------------------------------------------------------
    (folder / "transects_t").mkdir()
    (folder / "transects_t" / "transects.json").write_text(json.dumps({
        "version": 1, "modality": "t",
        "transects": [
            {"id": 1, "name": "North", "start_frame": 0, "end_frame": 19},
            {"id": 2, "name": "", "start_frame": 20, "end_frame": 39},
        ],
    }))

    # ---- FoV footprints: a 20x20 m square under every frame (world CRS) ---
    fov_store.record_fov(str(folder), "t", {
        idx: _square(img["location"][0] + origin[0],
                     img["location"][1] + origin[1])
        for idx, img in enumerate(images)})

    # ---- track perpendicular distances (world CRS) ------------------------
    # 3 tracks near transect A, 1 near transect B, 1 far from both (600 m north)
    tracks = [
        {"track_id": 1, "last_frame": 5, "class_id": 0,
         "detection_center": [origin[0] + 50.0, origin[1] + 5.0, 0.0]},
        {"track_id": 2, "last_frame": 8, "class_id": 0,
         "detection_center": [origin[0] + 80.0, origin[1] - 3.0, 0.0]},
        {"track_id": 3, "last_frame": 12, "class_id": 1,
         "detection_center": [origin[0] + 120.0, origin[1] + 8.0, 0.0]},
        {"track_id": 4, "last_frame": 25, "class_id": 0,
         "detection_center": [origin[0] + 60.0, origin[1] + 104.0, 0.0]},
        {"track_id": 5, "last_frame": 30, "class_id": 0,
         "detection_center": [origin[0] + 60.0, origin[1] + 600.0, 0.0]},
    ]
    (folder / "flight_route_t").mkdir()
    (folder / "flight_route_t" / "perpendicular_tracks_t.json").write_text(
        json.dumps({"crs": "EPSG:32633", "total_tracks": len(tracks),
                    "tracks": tracks}))

    return {
        "target_folder": str(folder),
        "target_epsg": 32633,
        "dem_path": str(folder / "dem.gltf"),
        "pop_camera": "T",
        "flight_route_camera": "T",
        "pop_methods": ["naive", "bootstrap"],
        "pop_n_boot": 100,
        "pop_seed": 42,
    }


def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class TestRunPopulationEstimation:
    def test_assigns_tracks_and_measures_areas(self, processor, flight):
        processor.run_population_estimation(flight)

        analytics = os.path.join(flight["target_folder"], "analytics_t")
        result = _read_json(os.path.join(analytics, "population_estimate.json"))

        assert result["n_transects"] == 2
        assert result["n_tracks"] == 5
        # Track 5 sits 600 m north of transect 2's line, far outside its 20 m
        # wide footprint: it was never monitored, so it is not counted.
        assert result["n_tracks_assigned"] == 4
        assert result["n_tracks_outside_fov"] == 1
        assert result["total_count"] == 4

        by_name = {t["name"]: t for t in result["transects"]}
        assert by_name["North"]["count"] == 3
        # the unnamed transect falls back to its counting id
        assert by_name["Transect 2"]["count"] == 1

        # 20 frames at x = 0, 10, …, 190, each covering a 20x20 m square, so
        # the union spans x ∈ [-10, 200] and y ∈ [-10, 10]: 210 m x 20 m.
        # Overlapping squares must be unioned, not summed (which would give
        # 20 x 400 = 8000 m²).
        assert by_name["North"]["area_m2"] == pytest.approx(4200.0, rel=1e-6)
        assert by_name["North"]["area_ha"] == pytest.approx(0.42, rel=1e-6)
        assert by_name["North"]["length_m"] == pytest.approx(190.0)
        assert by_name["North"]["n_frames_with_fov"] == 20

    def test_track_outside_every_footprint_is_not_counted(self, processor, flight):
        """The density's denominator is the monitored area — so must its numerator.

        Counting an animal that fell outside every field of view (merely
        because some transect's centre line was nearest) would spread it over
        ground that was never surveyed.
        """
        processor.run_population_estimation(flight)

        analytics = os.path.join(flight["target_folder"], "analytics_t")
        rows = _read_csv(os.path.join(analytics, "population_tracks.csv"))
        far = next(r for r in rows if r["track_id"] == "5")

        assert far["transect_id"] == ""
        assert far["outside_fov"] == "1"
        assert far["truncated"] == "0"
        # No transect took it, so there is no distance *to its transect* — but
        # the distance to the nearest centre line is kept, so the exclusion can
        # be diagnosed instead of just disappearing.
        assert far["distance_m"] == ""
        assert float(far["nearest_distance_m"]) == pytest.approx(500.0)

    def test_truncation_drops_a_track_inside_the_footprint(self, processor,
                                                           flight):
        # Track 3 lies 8 m off transect 1's line but inside its footprint;
        # a 5 m truncation must still discard it.
        flight["pop_truncation"] = 5.0
        processor.run_population_estimation(flight)

        analytics = os.path.join(flight["target_folder"], "analytics_t")
        result = _read_json(os.path.join(analytics, "population_estimate.json"))

        assert result["n_tracks_truncated"] == 1
        assert result["n_tracks_outside_fov"] == 1     # the far track
        assert result["total_count"] == 3

        rows = _read_csv(os.path.join(analytics, "population_tracks.csv"))
        near = next(r for r in rows if r["track_id"] == "3")
        assert near["transect_id"] == ""
        assert near["truncated"] == "1"
        assert near["outside_fov"] == "0"

    def test_naive_density_is_count_over_area(self, processor, flight):
        processor.run_population_estimation(flight)
        result = _read_json(os.path.join(
            flight["target_folder"], "analytics_t", "population_estimate.json"))

        # 4 counted animals on 2 x 0.42 ha = 0.84 ha -> 4 / 0.84 * 100
        assert result["total_ha"] == pytest.approx(0.84, rel=1e-6)
        assert result["estimates"]["naive"]["density_per_100ha"] == pytest.approx(
            4 / 0.84 * 100.0, rel=1e-6)

    def test_study_area_yields_an_abundance(self, processor, flight):
        flight["pop_study_area_ha"] = 200.0
        processor.run_population_estimation(flight)
        result = _read_json(os.path.join(
            flight["target_folder"], "analytics_t", "population_estimate.json"))

        naive = result["estimates"]["naive"]
        assert result["study_area_source"] == "manual"
        assert naive["abundance_study_area"] == pytest.approx(
            naive["density_per_100ha"] * 2.0, rel=1e-9)


class TestStudyArea:
    """The study area is the region extrapolated to, not the searched strips."""

    def test_flight_fov_coverage_is_always_reported(self, processor, flight):
        processor.run_population_estimation(flight)
        result = _read_json(os.path.join(
            flight["target_folder"], "analytics_t", "population_estimate.json"))

        # Two transects, 4200 m² each, 100 m apart: their footprints (y in
        # [-10, 10] and [90, 110]) do not touch, so the union equals the sum.
        assert result["flight_fov_area_ha"] == pytest.approx(0.84, rel=1e-6)
        assert result["total_ha"] == pytest.approx(0.84, rel=1e-6)
        # Reported even when no abundance was asked for
        assert result["study_area_source"] == "none"
        assert "abundance_study_area" not in result["estimates"]["naive"]

    def test_auto_uses_the_flight_fov_coverage(self, processor, flight):
        flight["pop_study_area_auto"] = True
        flight["pop_study_area_ha"] = 999.0      # ignored when auto is on
        processor.run_population_estimation(flight)
        result = _read_json(os.path.join(
            flight["target_folder"], "analytics_t", "population_estimate.json"))

        assert result["study_area_source"] == "flight_fov"
        assert result["study_area_ha"] == pytest.approx(
            result["flight_fov_area_ha"], rel=1e-9)

        naive = result["estimates"]["naive"]
        scale = result["study_area_ha"] / 100.0
        assert naive["abundance_study_area"] == pytest.approx(
            naive["density_per_100ha"] * scale, rel=1e-9)

    def test_overlapping_transects_make_the_two_areas_differ(self, processor,
                                                             flight, tmp_path):
        # Move transect B onto transect A's ground (both at y = 0): the summed
        # transect areas still count that ground twice (each transect is its
        # own sample), the flight FoV union counts it once.
        folder = flight["target_folder"]
        poses_path = os.path.join(folder, "poses_t.json")
        poses = _read_json(poses_path)
        for img in poses["images"][20:]:
            img["location"][1] = 0.0
        with open(poses_path, "w", encoding="utf-8") as fh:
            json.dump(poses, fh)

        origin = 5000000.0
        fov_store.record_fov(folder, "t", {
            idx: _square(img["location"][0] + 500000.0,
                         img["location"][1] + origin)
            for idx, img in enumerate(poses["images"])})

        processor.run_population_estimation(flight)
        result = _read_json(os.path.join(
            folder, "analytics_t", "population_estimate.json"))

        # Both transects now cover the same 4200 m² strip.
        assert result["total_ha"] == pytest.approx(0.84, rel=1e-6)
        assert result["flight_fov_area_ha"] == pytest.approx(0.42, rel=1e-6)

    def test_track_csv_flags_frame_range_agreement(self, processor, flight):
        processor.run_population_estimation(flight)
        rows = _read_csv(os.path.join(
            flight["target_folder"], "analytics_t", "population_tracks.csv"))

        by_id = {r["track_id"]: r for r in rows}
        # track 1 was filmed in frames of transect 1 and counted towards it
        assert by_id["1"]["transect_id"] == "1"
        assert by_id["1"]["in_frame_range"] == "1"
        # track 5 was filmed during transect 2 but landed outside its footprint,
        # so it has no transect and hence no frame-range verdict
        assert by_id["5"]["transect_id"] == ""
        assert by_id["5"]["in_frame_range"] == ""
        assert by_id["5"]["outside_fov"] == "1"

    def test_transect_areas_geojson_is_written(self, processor, flight):
        processor.run_population_estimation(flight)
        gj = _read_json(os.path.join(
            flight["target_folder"], "analytics_t", "transect_areas.geojson"))

        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) == 2
        props = {f["properties"]["name"] for f in gj["features"]}
        assert props == {"North", "Transect 2"}
        for feature in gj["features"]:
            assert feature["geometry"]["type"] == "Polygon"
            ring = feature["geometry"]["coordinates"][0]
            assert ring[0] == ring[-1]          # closed
            # world CRS, not mesh-local
            assert ring[0][0] > 400000.0

    def test_transect_routes_geojson_is_written(self, processor, flight):
        processor.run_population_estimation(flight)
        gj = _read_json(os.path.join(
            flight["target_folder"], "analytics_t", "transect_routes.geojson"))

        assert len(gj["features"]) == 2
        by_name = {f["properties"]["name"]: f for f in gj["features"]}
        north = by_name["North"]
        assert north["geometry"]["type"] == "LineString"
        assert north["properties"]["start_frame"] == 0
        assert north["properties"]["end_frame"] == 19
        assert north["properties"]["length_m"] == pytest.approx(190.0)

        # One vertex per frame of the transect, in the world CRS (origin added)
        coords = north["geometry"]["coordinates"]
        assert len(coords) == 20
        assert coords[0] == pytest.approx([500000.0, 5000000.0])
        assert coords[-1] == pytest.approx([500190.0, 5000000.0])

        # The second transect's route runs 100 m north of the first
        assert by_name["Transect 2"]["geometry"]["coordinates"][0] == pytest.approx(
            [500000.0, 5000100.0])

    def test_missing_transects_is_a_clear_error(self, processor, flight):
        os.remove(os.path.join(
            flight["target_folder"], "transects_t", "transects.json"))
        with pytest.raises(FileNotFoundError, match="Transect Splitting Tool"):
            processor.run_population_estimation(flight)

    def test_missing_fov_is_a_clear_error(self, processor, flight):
        """The store is what the step reads; the text file is not consulted."""
        os.remove(store.stage_path(
            flight["target_folder"], store.FOV, "t"))
        with pytest.raises(FileNotFoundError, match="Calculate Field of View"):
            processor.run_population_estimation(flight)

    def test_missing_track_perpendicular_is_a_clear_error(self, processor, flight):
        os.remove(os.path.join(
            flight["target_folder"], "flight_route_t",
            "perpendicular_tracks_t.json"))
        with pytest.raises(FileNotFoundError,
                           match="Calculate Track Perpendicular"):
            processor.run_population_estimation(flight)


def _write_second_flight(folder, origin):
    """A 2-transect flight like the ``flight`` fixture, at a different origin.

    Writes its own ``dem.json`` (carrying the origin) and returns its path, so
    the project can be added with an explicit DEM. Returns ``(dem_path,)``.
    """
    os.makedirs(folder, exist_ok=True)

    dem_path = os.path.join(folder, "dem.json")
    with open(dem_path, "w", encoding="utf-8") as fh:
        json.dump({"origin": list(origin) + [0.0]}, fh)
    images = []
    for i in range(20):
        images.append({"imagefile": f"a{i:03d}.png",
                       "location": [i * 10.0, 0.0, 100.0]})
    for i in range(20):
        images.append({"imagefile": f"b{i:03d}.png",
                       "location": [i * 10.0, 100.0, 100.0]})
    with open(os.path.join(folder, "poses_t.json"), "w", encoding="utf-8") as fh:
        json.dump({"images": images}, fh)

    os.makedirs(os.path.join(folder, "transects_t"), exist_ok=True)
    with open(os.path.join(folder, "transects_t", "transects.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "modality": "t", "transects": [
            {"id": 1, "name": "South", "start_frame": 0, "end_frame": 19},
            {"id": 2, "name": "", "start_frame": 20, "end_frame": 39},
        ]}, fh)

    fov_store.record_fov(folder, "t", {
        idx: _square(img["location"][0] + origin[0],
                     img["location"][1] + origin[1])
        for idx, img in enumerate(images)})

    # 3 tracks near transect A, 1 near transect B, 1 far from both.
    tracks = [
        {"track_id": 1, "last_frame": 5, "class_id": 0,
         "detection_center": [origin[0] + 50.0, origin[1] + 5.0, 0.0]},
        {"track_id": 2, "last_frame": 8, "class_id": 0,
         "detection_center": [origin[0] + 80.0, origin[1] - 3.0, 0.0]},
        {"track_id": 3, "last_frame": 12, "class_id": 1,
         "detection_center": [origin[0] + 120.0, origin[1] + 8.0, 0.0]},
        {"track_id": 4, "last_frame": 25, "class_id": 0,
         "detection_center": [origin[0] + 60.0, origin[1] + 104.0, 0.0]},
        {"track_id": 5, "last_frame": 30, "class_id": 0,
         "detection_center": [origin[0] + 60.0, origin[1] + 600.0, 0.0]},
    ]
    os.makedirs(os.path.join(folder, "flight_route_t"), exist_ok=True)
    with open(os.path.join(folder, "flight_route_t",
                           "perpendicular_tracks_t.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"crs": "EPSG:32633", "tracks": tracks}, fh)

    # World-CRS flight route (poses shifted by the origin) for origin recovery.
    route = [[img["location"][0] + origin[0], img["location"][1] + origin[1]]
             for img in images]
    with open(os.path.join(folder, "flight_route_t", "flight_route.geojson"),
              "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {},
             "geometry": {"type": "LineString", "coordinates": route}},
        ]}, fh)
    return dem_path


class TestMultiProject:
    """Pooling several flights into one combined estimate."""

    def test_pools_two_projects(self, processor, flight, tmp_path):
        folder2 = tmp_path / "target2"
        dem2 = _write_second_flight(str(folder2), origin=(600000.0, 5100000.0))

        config = dict(flight)
        config["pop_project_folders"] = [
            {"target": flight["target_folder"], "dem": ""},
            {"target": str(folder2), "dem": dem2},
        ]
        processor.run_population_estimation(config)

        result = _read_json(os.path.join(
            flight["target_folder"], "analytics_t", "population_estimate.json"))

        # Two projects, two transects and five tracks each.
        assert result["n_projects"] == 2
        assert len(result["projects"]) == 2
        assert result["n_transects"] == 4
        assert result["n_tracks"] == 10
        # Four counted per flight (the far track is never monitored).
        assert result["total_count"] == 8
        assert result["n_tracks_outside_fov"] == 2
        # 8 counted animals on 4 x 0.42 ha = 1.68 ha.
        assert result["total_ha"] == pytest.approx(1.68, rel=1e-6)
        assert result["estimates"]["naive"]["density_per_100ha"] == pytest.approx(
            8 / 1.68 * 100.0, rel=1e-6)

        # Names are project-prefixed so the two flights' transects stay distinct.
        names = [t["name"] for t in result["transects"]]
        assert any(n.endswith("/ North") for n in names)
        assert any(n.endswith("/ South") for n in names)

        # The current project used the configured DEM; the second used the
        # dem.json supplied with it.
        sources = {p["name"]: p["dem_origin_source"] for p in result["projects"]}
        assert sources[os.path.basename(flight["target_folder"])] == "config"
        assert sources["target2"] == "provided"

    def test_added_project_without_dem_is_an_error(self, processor, flight, tmp_path):
        folder2 = tmp_path / "target2"
        _write_second_flight(str(folder2), origin=(600000.0, 5100000.0))
        config = dict(flight)
        # The added project carries no dem.json, so its origin cannot be resolved.
        config["pop_project_folders"] = [
            {"target": flight["target_folder"], "dem": ""},
            {"target": str(folder2), "dem": ""},
        ]
        with pytest.raises(FileNotFoundError, match="dem.json"):
            processor.run_population_estimation(config)

    def test_single_project_via_folders_matches_default(self, processor, flight):
        config = dict(flight)
        config["pop_project_folders"] = [{"target": flight["target_folder"], "dem": ""}]
        processor.run_population_estimation(config)
        result = _read_json(os.path.join(
            flight["target_folder"], "analytics_t", "population_estimate.json"))

        assert result["n_projects"] == 1
        assert result["n_transects"] == 2
        assert result["total_count"] == 4
        # A lone project keeps its plain transect names (no project prefix).
        assert {t["name"] for t in result["transects"]} == {"North", "Transect 2"}
