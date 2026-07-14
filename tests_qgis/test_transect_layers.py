# -*- coding: utf-8 -*-
"""Real-QGIS test of 'Add Transect Areas to QGIS'.

Drives the dock widget's loader against a prepared analytics folder and
inspects the layer tree it builds — the styling and labelling calls only fail
inside a real QGIS, which the unit suite (with its qgis stub) cannot catch.
"""
import json
import os

import pytest

from qgis.core import QgsProject, QgsWkbTypes


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    """Turn a swallowed error into a test failure, not a hung modal.

    ``add_transect_areas_to_qgis`` reports failures with QMessageBox.critical.
    Left alone, that opens a modal dialog which aborts the headless process —
    the error itself never reaches the report. Raising instead surfaces it.
    """
    from bambi_wildlife_detection import bambi_dock_widget as mod

    def fail(_parent, title, text, *args, **kwargs):
        raise AssertionError(f"unexpected error dialog: {title} — {text}")

    monkeypatch.setattr(mod.QMessageBox, "critical", fail)


@pytest.fixture
def analytics(tmp_path):
    """A target folder with the two geojsons the estimation step writes.

    Everything lives in the world CRS the poses fixture implies (DEM origin
    1000/2000): transect 1's footprint straddles y = 2000, transect 2's
    y = 2100. The containment test only means anything if the footprints and
    the track positions share a coordinate system.
    """
    folder = tmp_path / "target"
    (folder / "analytics_t").mkdir(parents=True)

    def ring(x0, y0, x1, y1):
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]

    areas = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "Polygon",
                          "coordinates": [ring(990, 1990, 1030, 2010)]},
             "properties": {"transect_id": 1, "name": "North meadow",
                            "area_ha": 0.4, "count": 3}},
            {"type": "Feature",
             "geometry": {"type": "Polygon",
                          "coordinates": [ring(990, 2090, 1030, 2110)]},
             "properties": {"transect_id": 2, "name": "Transect 2",
                            "area_ha": 0.4, "count": 1}},
        ],
    }
    routes = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "LineString",
                          "coordinates": [[1000, 2000], [1010, 2000],
                                          [1020, 2000]]},
             "properties": {"transect_id": 1, "name": "North meadow",
                            "start_frame": 0, "end_frame": 19,
                            "length_m": 200.0, "count": 3}},
            {"type": "Feature",
             "geometry": {"type": "LineString",
                          "coordinates": [[1000, 2100], [1020, 2100]]},
             "properties": {"transect_id": 2, "name": "Transect 2",
                            "start_frame": 20, "end_frame": 39,
                            "length_m": 200.0, "count": 1}},
        ],
    }
    (folder / "analytics_t" / "transect_areas.geojson").write_text(
        json.dumps(areas), encoding="utf-8")
    (folder / "analytics_t" / "transect_routes.geojson").write_text(
        json.dumps(routes), encoding="utf-8")
    return folder


def _write_poses_and_transects(folder):
    """The inputs the loader rebuilds the routes from, with a non-zero origin.

    Poses are mesh-local (world CRS minus the DEM origin), so a rebuilt route
    is only correct if the loader adds the origin back.
    """
    images = [{"imagefile": f"{i:03d}.png", "location": [i * 10.0, 0.0, 100.0]}
              for i in range(3)]
    images += [{"imagefile": f"b{i:03d}.png", "location": [i * 10.0, 100.0, 100.0]}
               for i in range(3)]
    (folder / "poses_t.json").write_text(
        json.dumps({"images": images}), encoding="utf-8")

    (folder / "dem.json").write_text(
        json.dumps({"origin": [1000.0, 2000.0, 0.0]}), encoding="utf-8")

    (folder / "transects_t").mkdir(exist_ok=True)
    (folder / "transects_t" / "transects.json").write_text(json.dumps({
        "version": 1, "modality": "t",
        "transects": [
            {"id": 1, "name": "North meadow", "start_frame": 0, "end_frame": 2},
            {"id": 2, "name": "", "start_frame": 3, "end_frame": 5},
        ],
    }), encoding="utf-8")


def _configure(dock, folder):
    dock.target_folder_edit.setText(str(folder))
    dock.pop_camera_combo.setCurrentIndex(0)          # Thermal
    dock.target_crs_edit.setText("EPSG:32633")
    dock.dem_metadata_path_edit.setText(str(folder / "dem.json"))


def _find_group(name):
    return QgsProject.instance().layerTreeRoot().findGroup(name)


def _layers_of(group):
    """``{layer name: layer}`` of one transect subgroup, in tree order."""
    return {node.layer().name(): node.layer() for node in group.findLayers()}


class TestAddTransectAreas:
    def test_builds_a_group_per_transect(self, dock, analytics):
        _configure(dock, analytics)
        dock.add_transect_areas_to_qgis()

        main = _find_group("BAMBI Transect Areas (Thermal)")
        assert main is not None

        subgroups = main.findGroups()
        assert [g.name() for g in subgroups] == ["North meadow", "Transect 2"]

        for group in subgroups:
            layers = [node.layer() for node in group.findLayers()]
            # Tree order = draw order: the points and line must sit above the
            # translucent field-of-view fill, not under it.
            assert [lyr.name() for lyr in layers] == ["Flight Route",
                                                      "Field of View"]
            route, fov = layers
            assert fov.geometryType() == QgsWkbTypes.PolygonGeometry
            assert route.geometryType() == QgsWkbTypes.LineGeometry
            assert fov.featureCount() == 1
            assert route.featureCount() == 1
            assert fov.isValid() and route.isValid()

    def test_layers_carry_the_transect_attributes(self, dock, analytics):
        _configure(dock, analytics)
        dock.add_transect_areas_to_qgis()

        main = _find_group("BAMBI Transect Areas (Thermal)")
        north = next(g for g in main.findGroups() if g.name() == "North meadow")
        layers = _layers_of(north)

        fov_feat = next(layers["Field of View"].getFeatures())
        assert fov_feat["transect_id"] == 1
        assert fov_feat["name"] == "North meadow"
        assert fov_feat["count"] == 3
        assert fov_feat["area_ha"] == pytest.approx(0.4)
        assert layers["Field of View"].labelsEnabled()

        route_feat = next(layers["Flight Route"].getFeatures())
        assert route_feat["start_frame"] == 0
        assert route_feat["end_frame"] == 19
        assert route_feat["length_m"] == pytest.approx(200.0)

    def test_each_transect_gets_its_own_colour(self, dock, analytics):
        _configure(dock, analytics)
        dock.add_transect_areas_to_qgis()

        main = _find_group("BAMBI Transect Areas (Thermal)")
        colours = [
            _layers_of(group)["Field of View"].renderer().symbol().color().name()
            for group in main.findGroups()
        ]
        assert len(set(colours)) == 2

    def test_routes_are_rebuilt_when_the_file_is_missing(self, dock, analytics):
        """An estimation run from before transect_routes.geojson existed.

        The route is only the poses of the transect's frame range, so the
        loader rebuilds it instead of dropping the layer — the user must not
        have to re-run the estimation to get it.
        """
        (analytics / "analytics_t" / "transect_routes.geojson").unlink()
        _write_poses_and_transects(analytics)
        _configure(dock, analytics)
        dock.add_transect_areas_to_qgis()

        main = _find_group("BAMBI Transect Areas (Thermal)")
        assert main is not None
        for group in main.findGroups():
            assert list(_layers_of(group)) == ["Flight Route", "Field of View"]

        north = next(g for g in main.findGroups() if g.name() == "North meadow")
        route = _layers_of(north)["Flight Route"]
        feat = next(route.getFeatures())
        assert feat["start_frame"] == 0
        assert feat["end_frame"] == 2
        # Poses are mesh-local; the rebuilt route must be shifted to the world
        # CRS by the DEM origin, like the estimation step's own output.
        geom = feat.geometry().asPolyline()
        assert geom[0].x() == pytest.approx(1000.0)
        assert geom[0].y() == pytest.approx(2000.0)
        assert geom[-1].x() == pytest.approx(1020.0)

    def test_transect_without_a_fov_area_still_shows_its_route(self, dock,
                                                               analytics):
        """The FoV step may not have covered a transect's frame range.

        That transect has no area feature — it must still reach the map with
        its route rather than disappear from the group.
        """
        areas_file = analytics / "analytics_t" / "transect_areas.geojson"
        areas = json.loads(areas_file.read_text())
        areas["features"] = [f for f in areas["features"]
                             if f["properties"]["transect_id"] != 2]
        areas_file.write_text(json.dumps(areas), encoding="utf-8")

        _configure(dock, analytics)
        dock.add_transect_areas_to_qgis()

        main = _find_group("BAMBI Transect Areas (Thermal)")
        assert [g.name() for g in main.findGroups()] == ["North meadow",
                                                         "Transect 2"]
        assert list(_layers_of(main.findGroups()[0])) == ["Flight Route",
                                                          "Field of View"]
        assert list(_layers_of(main.findGroups()[1])) == ["Flight Route"]

    def test_areas_load_when_no_poses_exist_to_rebuild_from(self, dock, analytics):
        (analytics / "analytics_t" / "transect_routes.geojson").unlink()
        _configure(dock, analytics)
        dock.add_transect_areas_to_qgis()

        main = _find_group("BAMBI Transect Areas (Thermal)")
        assert main is not None
        for group in main.findGroups():
            assert list(_layers_of(group)) == ["Field of View"]

    def test_no_tracks_layer_without_perpendicular_distances(self, dock,
                                                             analytics):
        """No perpendicular distances: the track stage is skipped, not failed."""
        _configure(dock, analytics)
        dock.add_transect_areas_to_qgis()

        main = _find_group("BAMBI Transect Areas (Thermal)")
        for group in main.findGroups():
            assert "Tracks" not in _layers_of(group)

    def test_missing_output_warns_instead_of_raising(self, dock, tmp_path,
                                                     monkeypatch):
        from bambi_wildlife_detection import bambi_dock_widget as mod

        warned = []
        monkeypatch.setattr(mod.QMessageBox, "warning",
                            lambda *a, **k: warned.append(a))
        _configure(dock, tmp_path)
        dock.add_transect_areas_to_qgis()

        assert warned
        assert _find_group("BAMBI Transect Areas (Thermal)") is None


class TestReAdding:
    """Clicking the button again must replace the layers, not fail on them."""

    def _groups_named(self, name):
        root = QgsProject.instance().layerTreeRoot()
        return [g for g in root.findGroups() if g.name() == name]

    def test_second_run_replaces_the_group(self, dock, analytics):
        _configure(dock, analytics)
        dock.add_transect_areas_to_qgis()
        dock.add_transect_areas_to_qgis()

        # Not two stacked copies of the same group
        assert len(self._groups_named("BAMBI Transect Areas (Thermal)")) == 1

        main = _find_group("BAMBI Transect Areas (Thermal)")
        assert [g.name() for g in main.findGroups()] == ["North meadow",
                                                         "Transect 2"]
        for group in main.findGroups():
            for layer in _layers_of(group).values():
                assert layer.isValid()
                assert layer.featureCount() == 1

    def test_second_run_survives_a_locked_geopackage(self, dock, analytics,
                                                     monkeypatch):
        """Windows keeps the .gpkg open via GDAL's dataset pool.

        The delete then fails with WinError 32. A GeoPackage is SQLite, so the
        layer inside it is rewritten in place instead — the run must succeed
        rather than fall back to a temporary memory layer.
        """
        _configure(dock, analytics)
        dock.add_transect_areas_to_qgis()

        real_remove = os.remove

        def locked(path, *args, **kwargs):
            if str(path).endswith(".gpkg"):
                raise OSError(32, "The process cannot access the file because "
                                  "it is being used by another process")
            return real_remove(path, *args, **kwargs)

        monkeypatch.setattr(os, "remove", locked)
        dock.add_transect_areas_to_qgis()      # must not raise a dialog

        main = _find_group("BAMBI Transect Areas (Thermal)")
        north = next(g for g in main.findGroups() if g.name() == "North meadow")
        for layer in _layers_of(north).values():
            assert layer.isValid()
            assert layer.featureCount() == 1
            # persisted to the GeoPackage, not silently downgraded to memory
            assert layer.dataProvider().name() == "ogr"

    def test_layers_are_persisted_to_geopackages(self, dock, analytics):
        _configure(dock, analytics)
        dock.add_transect_areas_to_qgis()

        gpkg_dir = analytics / "transect_layers"
        written = sorted(p.name for p in gpkg_dir.glob("*.gpkg"))
        assert written == [
            "Transect1_FoV_t.gpkg", "Transect1_Route_t.gpkg",
            "Transect2_FoV_t.gpkg", "Transect2_Route_t.gpkg",
        ]


def _write_population_tracks_csv(folder):
    """The per-track assignment the estimation step writes."""
    rows = [
        # track_id,last_frame,x,y,class_id,transect_id,transect_name,
        # distance_m,in_frame_range,truncated
        "track_id,last_frame,x,y,class_id,transect_id,transect_name,"
        "distance_m,in_frame_range,truncated",
        "1,1,1005.0,2003.0,0,1,North meadow,3.0,1,0",
        "2,2,1015.0,1998.0,1,1,North meadow,2.0,1,0",
        "3,4,1010.0,2098.0,0,2,Transect 2,2.0,1,0",
        # unassigned (beyond the truncation distance) — must not be shown
        "4,5,9999.0,9999.0,0,,,,,1",
    ]
    (folder / "analytics_t" / "population_tracks.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8")


def _write_perpendicular_tracks(folder):
    """The raw perpendicular distances the assignment can be recomputed from."""
    (folder / "flight_route_t").mkdir(exist_ok=True)
    tracks = [
        {"track_id": 1, "last_frame": 1, "class_id": 0,
         "detection_center": [1005.0, 2003.0, 0.0]},
        {"track_id": 2, "last_frame": 2, "class_id": 1,
         "detection_center": [1015.0, 1998.0, 0.0]},
        {"track_id": 3, "last_frame": 4, "class_id": 0,
         "detection_center": [1010.0, 2098.0, 0.0]},
    ]
    (folder / "flight_route_t" / "perpendicular_tracks_t.json").write_text(
        json.dumps({"crs": "EPSG:32633", "tracks": tracks}), encoding="utf-8")


class TestTransectTracks:
    def test_tracks_come_from_the_estimation_csv(self, dock, analytics):
        _write_population_tracks_csv(analytics)
        _configure(dock, analytics)
        dock.add_transect_areas_to_qgis()

        main = _find_group("BAMBI Transect Areas (Thermal)")
        north = next(g for g in main.findGroups() if g.name() == "North meadow")
        layers = _layers_of(north)
        # Points on top of the line, line on top of the fill
        assert list(layers) == ["Tracks", "Flight Route", "Field of View"]

        tracks = layers["Tracks"]
        assert tracks.geometryType() == QgsWkbTypes.PointGeometry
        assert tracks.featureCount() == 2
        assert tracks.labelsEnabled()

        feats = {f["track_id"]: f for f in tracks.getFeatures()}
        assert set(feats) == {1, 2}
        assert feats[1]["transect_id"] == 1
        assert feats[1]["transect"] == "North meadow"
        assert feats[1]["distance_m"] == pytest.approx(3.0)
        assert feats[1]["in_frame_range"] == 1
        assert feats[2]["class_id"] == 1
        pt = feats[1].geometry().asPoint()
        assert (pt.x(), pt.y()) == pytest.approx((1005.0, 2003.0))

        second = next(g for g in main.findGroups() if g.name() == "Transect 2")
        assert _layers_of(second)["Tracks"].featureCount() == 1

    def test_unassigned_tracks_are_not_shown(self, dock, analytics):
        _write_population_tracks_csv(analytics)
        _configure(dock, analytics)
        dock.add_transect_areas_to_qgis()

        main = _find_group("BAMBI Transect Areas (Thermal)")
        shown = {
            f["track_id"]
            for group in main.findGroups()
            for f in _layers_of(group)["Tracks"].getFeatures()
        }
        # Track 4 was truncated away and never counted, so it is not on the map
        assert shown == {1, 2, 3}

    def test_assignment_is_recomputed_without_the_csv(self, dock, analytics):
        """No estimation run yet — assign_tracks is reused on the raw distances."""
        _write_poses_and_transects(analytics)
        _write_perpendicular_tracks(analytics)
        _configure(dock, analytics)
        dock.add_transect_areas_to_qgis()

        main = _find_group("BAMBI Transect Areas (Thermal)")
        north = next(g for g in main.findGroups() if g.name() == "North meadow")
        second = next(g for g in main.findGroups() if g.name() == "Transect 2")

        # Tracks 1 & 2 sit near the y=2000 line, track 3 near the y=2100 one
        assert {f["track_id"] for f in _layers_of(north)["Tracks"].getFeatures()} \
            == {1, 2}
        assert {f["track_id"] for f in _layers_of(second)["Tracks"].getFeatures()} \
            == {3}

    def test_recomputed_assignment_honours_the_truncation(self, dock, analytics):
        _write_poses_and_transects(analytics)
        _write_perpendicular_tracks(analytics)
        _configure(dock, analytics)
        # Track 1 is 3 m off its transect, track 2 only 2 m
        dock.pop_truncation_spin.setValue(2.5)
        dock.add_transect_areas_to_qgis()

        main = _find_group("BAMBI Transect Areas (Thermal)")
        north = next(g for g in main.findGroups() if g.name() == "North meadow")
        assert {f["track_id"] for f in _layers_of(north)["Tracks"].getFeatures()} \
            == {2}

    def test_recomputed_assignment_excludes_tracks_outside_the_footprint(
            self, dock, analytics):
        """The containment rule must hold on the QGIS side too.

        The loader's predicate is built on QgsGeometry (shapely is not
        guaranteed to exist in a QGIS install), so it needs its own proof that
        a track outside every footprint is not drawn into a transect.
        """
        _write_poses_and_transects(analytics)

        # Track 9 sits 60 m north of transect 1's line — the areas in this
        # fixture only span y ∈ [0, 20] and y ∈ [100, 120], so nothing saw it.
        tracks = [
            {"track_id": 1, "last_frame": 1, "class_id": 0,
             "detection_center": [1005.0, 2003.0, 0.0]},
            {"track_id": 9, "last_frame": 1, "class_id": 0,
             "detection_center": [1005.0, 2060.0, 0.0]},
        ]
        (analytics / "flight_route_t").mkdir(exist_ok=True)
        (analytics / "flight_route_t" / "perpendicular_tracks_t.json").write_text(
            json.dumps({"tracks": tracks}), encoding="utf-8")

        _configure(dock, analytics)
        dock.add_transect_areas_to_qgis()

        main = _find_group("BAMBI Transect Areas (Thermal)")
        shown = {
            f["track_id"]
            for group in main.findGroups()
            for name, layer in _layers_of(group).items() if name == "Tracks"
            for f in layer.getFeatures()
        }
        assert shown == {1}
