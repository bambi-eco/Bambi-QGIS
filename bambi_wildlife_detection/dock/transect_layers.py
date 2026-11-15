# -*- coding: utf-8 -*-
"""Survey Analytics > Add Transect Areas: the transects on the map.

For every flight of the survey: the ground each transect covered, the route
flown along it, and the tracks that were counted on it.
"""

import json
import os
from collections import defaultdict

from qgis.PyQt.QtWidgets import QMessageBox
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsCoordinateReferenceSystem, QgsFeature,
    QgsGeometry, QgsPointXY, QgsField,
)

from ..qgis_compat import FieldType


class TransectLayersMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def add_transect_areas_to_qgis(self):
        """Add each survey flight's transect areas into that flight's group.

        Transects belong to one flight - their frame ranges index its poses -
        so each flight gets its own 'BAMBI Transect Areas' group, written by
        the population estimation into the flight's own analytics folder.
        """
        from ..core.survey import route_suffix

        survey = self.current_survey()
        base = self.get_config()
        m = survey.modality
        missing, added = [], 0
        for flight in survey.flights:
            fr = route_suffix(flight.target_folder, m,
                              f"perpendicular_tracks_{m}.json") or m
            config = dict(
                base, target_folder=flight.target_folder,
                dem_path=flight.dem_path,
                alfs_dem_metadata_path=flight.dem or None,
                flight_route_camera="T" if fr == "t" else "W",
                pop_camera="T" if m == "t" else "W",
                target_epsg=survey.epsg or base.get("target_epsg"))
            with self._layers_for_flight(flight.target_folder):
                if self._add_transect_areas_for(config, flight.name):
                    added += 1
                else:
                    missing.append(flight.name)
        if not added:
            QMessageBox.warning(
                self, "Missing Data",
                "The transect areas have not been generated.\n"
                "Please run 'Estimate Population (Transects)' first.")
        elif missing:
            self.log("No transect areas yet for: " + ", ".join(missing))

    def _add_transect_areas_for(self, config: dict, flight_name: str) -> bool:
        """Add one layer group per transect, holding its route and its FoV area.

        Layer tree::

            BAMBI Transect Areas (Thermal)
              ├─ North meadow
              │    ├─ Field of View   (the merged frame footprints)
              │    └─ Flight Route    (the sub-route, start frame → end frame)
              └─ Transect 2
                   └─ …

        Each transect keeps the colour the Transect Splitting Tool gives it, so
        the map and the tool's overview agree.
        """
        suffix = "t" if config.get("pop_camera", "T") == "T" else "w"
        camera_label = "Thermal" if suffix == "t" else "RGB"
        analytics = os.path.join(config["target_folder"], f"analytics_{suffix}")

        areas_file = os.path.join(analytics, "transect_areas.geojson")
        routes_file = os.path.join(analytics, "transect_routes.geojson")
        if not os.path.exists(areas_file):
            return False

        try:
            self.update_status("add_transect_areas", "🟡 Loading...")
            areas = self._read_geojson_features(areas_file)
            if not areas:
                raise RuntimeError("The transect areas file holds no features.")

            # A run from before the routes existed wrote no transect_routes.geojson.
            # The routes are just the poses of each transect's frame range, so
            # rebuild them rather than making the user re-run the estimation.
            if os.path.exists(routes_file):
                routes = self._read_geojson_features(routes_file)
            else:
                routes = self._transect_routes_from_poses(config, suffix)
                if routes:
                    self.log("transect_routes.geojson not found - the sub-flight "
                             "routes were rebuilt from the poses and the transect "
                             "definitions.")

            # The tracks assigned to each transect. Absent when no perpendicular
            # distances exist - that stage is simply skipped. The footprints are
            # passed along because a track only counts inside the area that saw it.
            tracks_by_id = self._transect_tracks(config, suffix, areas)

            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{config['target_epsg']}")
            group_name = f"BAMBI Transect Areas ({camera_label})"
            # Drop a previous run's group first: it would otherwise pile up a
            # duplicate, and its layers keep the GeoPackages open, which stops
            # them being rewritten on Windows.
            self._remove_layer_group(group_name)
            main_group = self._create_layer_group(group_name)

            areas_by_id = defaultdict(list)
            for feature in areas:
                areas_by_id[feature["properties"].get("transect_id")].append(feature)
            routes_by_id = defaultdict(list)
            for feature in routes:
                routes_by_id[feature["properties"].get("transect_id")].append(feature)

            # Key off both: a transect whose frame range the FoV step never
            # covered has no area feature, and it must still appear on the map
            # (with its route alone) rather than vanish silently.
            n_added = 0
            all_ids = sorted(set(areas_by_id) | set(routes_by_id),
                             key=lambda v: (v is None, v))
            for transect_id in all_ids:
                group_features = areas_by_id.get(transect_id, [])
                route_features = routes_by_id.get(transect_id, [])
                source = (group_features or route_features)[0]["properties"]
                name = source.get("name") or f"Transect {transect_id}"
                color = self._transect_color_str(transect_id)
                transect_group = main_group.addGroup(name)

                # Added top-down: the layer tree draws its first child on top,
                # so the points and the line stay visible over the translucent
                # field-of-view fill.
                tracks_layer = self._build_transect_tracks_layer(
                    tracks_by_id.get(transect_id, []), transect_id, name,
                    target_crs, color, suffix)
                if tracks_layer is not None:
                    transect_group.addLayer(tracks_layer)

                route_layer = self._build_transect_route_layer(
                    route_features, transect_id, name, target_crs, color, suffix)
                if route_layer is not None:
                    transect_group.addLayer(route_layer)

                fov_layer = self._build_transect_fov_layer(
                    group_features, transect_id, name, target_crs, color, suffix)
                if fov_layer is not None:
                    transect_group.addLayer(fov_layer)
                else:
                    self.log(f"Note: '{name}' has no field-of-view footprint - "
                             "its group holds no area layer.")

                transect_group.setExpanded(False)
                n_added += 1

            main_group.setExpanded(True)
            self.update_status("add_transect_areas", "🟢 Added")
            self.iface.mapCanvas().refresh()
            self.log(f"{flight_name}: added {n_added} transect group(s) "
                     f"under 'BAMBI Transect Areas ({camera_label})'")
            if not routes:
                self.log("Warning: no sub-flight routes could be built - check "
                         f"that poses_{suffix}.json and transects_{suffix}/"
                         "transects.json exist. Only the field-of-view areas "
                         "were added.")
        except Exception as e:
            self.update_status("add_transect_areas", "🔴 Error")
            self.log(f"Error adding transect areas of {flight_name}: {e}")
            QMessageBox.critical(
                self, "Error",
                f"Failed to add the transect areas of {flight_name}: {e}")
        return True

    @staticmethod
    def _read_geojson_features(path: str) -> list:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f).get("features", [])

    def _transect_tracks(self, config: dict, suffix: str,
                         area_features: list) -> dict:
        """The tracks assigned to each transect: ``{transect_id: [track, …]}``.

        Prefers ``population_tracks.csv`` - that is the very assignment the
        density numbers were computed from, so the map cannot disagree with
        them. Without it (no estimation run yet) the assignment is recomputed
        from the track perpendicular distances with the same
        :func:`core.population.assign_tracks` the step uses, including the
        containment test against the monitored areas in *area_features*.
        Returns ``{}`` when no perpendicular distances exist at all, so the
        caller skips the track layers instead of failing.
        """
        target_folder = config.get("target_folder", "")
        by_id = defaultdict(list)

        tracks_csv = os.path.join(
            target_folder, f"analytics_{suffix}", "population_tracks.csv")
        if os.path.isfile(tracks_csv):
            try:
                import csv as _csv
                with open(tracks_csv, 'r', encoding='utf-8', newline='') as f:
                    for row in _csv.DictReader(f):
                        # Blank transect_id: unassigned or beyond the truncation
                        # distance - it was not counted, so it is not shown.
                        if not row.get("transect_id"):
                            continue
                        by_id[int(row["transect_id"])].append({
                            # empty for a box of a survey that counts boxes
                            "track_id": (int(row["track_id"])
                                         if row.get("track_id") else None),
                            "last_frame": int(row["last_frame"] or 0),
                            "x": float(row["x"]),
                            "y": float(row["y"]),
                            "class_id": int(row["class_id"] or 0),
                            "distance_m": float(row["distance_m"] or 0.0),
                            "in_frame_range": int(row.get("in_frame_range") or 0),
                        })
                return dict(by_id)
            except (ValueError, KeyError, OSError) as e:
                self.log(f"Could not read {os.path.basename(tracks_csv)} "
                         f"({e}) - recomputing the track assignment.")
                by_id = defaultdict(list)

        return self._assign_tracks_to_transects(config, suffix, area_features)

    def _fov_predicate(self, area_features: list):
        """``(transect_id, x, y) -> bool``: does that transect's area cover it?

        Built on ``QgsGeometry`` rather than shapely: this runs inside the QGIS
        process, where QGIS' own geometry engine is always present while
        shapely is only an optional dependency.
        """
        geoms = {}
        for feature in area_features:
            rings = feature.get("geometry", {}).get("coordinates", [])
            if not rings:
                continue
            transect_id = feature["properties"].get("transect_id")
            points = [QgsPointXY(float(x), float(y)) for x, y in rings[0]]
            geom = QgsGeometry.fromPolygonXY([points])
            if not geom.isGeosValid():
                geom = geom.makeValid()
            # A transect's area can arrive as several rings (a footprint broken
            # up by gaps in the frame sampling) - all of them count.
            geoms[transect_id] = (geom if transect_id not in geoms
                                  else geoms[transect_id].combine(geom))

        def contains(transect_id, x: float, y: float) -> bool:
            geom = geoms.get(transect_id)
            if geom is None or geom.isEmpty():
                return False
            # intersects() counts a point on the boundary as inside, so a track
            # sitting right at the edge of the footprint is not lost.
            return geom.intersects(QgsGeometry.fromPointXY(QgsPointXY(x, y)))

        return contains

    def _assign_tracks_to_transects(self, config: dict, suffix: str,
                                    area_features: list) -> dict:
        """Recompute the transect assignment from the perpendicular distances."""
        from ..core.pipeline_outputs import read_dem_origin_xy
        from ..core.population import assign_tracks, transect_centerline
        from ..core.transects import TransectStore

        target_folder = config.get("target_folder", "")
        fr_suffix = "t" if config.get("flight_route_camera", "T") == "T" else "w"
        perp_file = os.path.join(
            target_folder, f"flight_route_{fr_suffix}",
            f"perpendicular_tracks_{suffix}.json")
        poses_path = os.path.join(target_folder, f"poses_{suffix}.json")
        if not os.path.isfile(perp_file) or not os.path.isfile(poses_path):
            self.log("No track perpendicular distances found - the transect "
                     "groups are added without their tracks.")
            return {}

        try:
            with open(perp_file, 'r', encoding='utf-8') as f:
                tracks = json.load(f).get("tracks", [])
            with open(poses_path, 'r', encoding='utf-8') as f:
                images = json.load(f).get("images", [])
        except (ValueError, OSError):
            return {}
        if not tracks or not images:
            return {}

        store = TransectStore(target_folder, suffix)
        try:
            if not store.load():
                return {}
        except (ValueError, OSError):
            return {}

        x_offset, y_offset = read_dem_origin_xy(
            config.get("dem_path", ""),
            config.get("alfs_dem_metadata_path") or "")
        centerlines = {}
        for transect in store.ordered():
            transect.clamp(len(images))
            centerlines[transect.transect_id] = transect_centerline(
                images, transect.first_frame, transect.last_frame,
                x_offset, y_offset)

        ranges = {t.transect_id: (t.first_frame, t.last_frame)
                  for t in store.ordered()}
        truncation = float(config.get("pop_truncation", 0.0) or 0.0)

        by_id = defaultdict(list)
        for a in assign_tracks(tracks, centerlines, truncation,
                               contains=self._fov_predicate(area_features)):
            transect_id = a["transect_id"]
            if transect_id is None:
                continue
            lo, hi = ranges[transect_id]
            frame = a["last_frame"]
            by_id[transect_id].append({
                "track_id": a["track_id"],
                "last_frame": frame if frame is not None else 0,
                "x": a["x"],
                "y": a["y"],
                "class_id": a["class_id"] if a["class_id"] is not None else 0,
                "distance_m": a["distance_m"],
                "in_frame_range": int(
                    frame is not None and lo <= frame <= hi),
            })
        self.log("No population_tracks.csv found - the track assignment was "
                 "recomputed from the perpendicular distances.")
        return dict(by_id)

    def _build_transect_tracks_layer(self, tracks, transect_id, name, target_crs,
                                     color, suffix):
        """Point layer with the tracks assigned to one transect."""
        from qgis.core import QgsMarkerSymbol, QgsSingleSymbolRenderer

        if not tracks:
            return None

        layer = QgsVectorLayer(
            "Point?crs=" + target_crs.authid(), "Tracks", "memory")
        provider = layer.dataProvider()
        provider.addAttributes([
            QgsField("track_id", FieldType.Int),
            QgsField("transect_id", FieldType.Int),
            QgsField("transect", FieldType.String),
            QgsField("last_frame", FieldType.Int),
            QgsField("class_id", FieldType.Int),
            QgsField("distance_m", FieldType.Double),
            QgsField("in_frame_range", FieldType.Int),
        ])
        layer.updateFields()

        features = []
        for track in tracks:
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPointXY(
                QgsPointXY(float(track["x"]), float(track["y"]))))
            feat.setAttributes([
                track["track_id"], transect_id, name, track["last_frame"],
                track["class_id"],
                round(float(track["distance_m"] or 0.0), 3),
                track["in_frame_range"],
            ])
            features.append(feat)
        provider.addFeatures(features)
        layer.updateExtents()

        def style(target):
            target.setRenderer(QgsSingleSymbolRenderer(
                QgsMarkerSymbol.createSimple({
                    'name': 'circle',
                    'color': f'{color},255',
                    'outline_color': '255,255,255,255',
                    'outline_width': '0.2',
                    'size': '2.4',
                })))
            self._label_transect_tracks(target)

        style(layer)
        layer = self._persist_memory_layer(
            layer, f"Transect{transect_id}_Tracks_{suffix}", "transect_layers",
            display_name="Tracks")
        style(layer)
        QgsProject.instance().addMapLayer(layer, False)
        return layer

    def _label_transect_tracks(self, layer):
        """Label each track point with its id."""
        try:
            from qgis.core import (
                QgsPalLayerSettings, QgsTextFormat, QgsVectorLayerSimpleLabeling,
            )
            settings = QgsPalLayerSettings()
            settings.fieldName = "track_id"
            settings.placement = QgsPalLayerSettings.Placement.AroundPoint
            text_format = QgsTextFormat()
            text_format.setSize(8)
            settings.setFormat(text_format)
            layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
            layer.setLabelsEnabled(True)
            layer.triggerRepaint()
        except Exception as e:  # nosec B110
            self.log(f"Warning: could not label transect tracks layer: {e}")

    def _transect_routes_from_poses(self, config: dict, suffix: str) -> list:
        """Rebuild the transects' sub-flight routes from poses + definitions.

        Same shape as ``transect_routes.geojson`` - used when that file is
        missing because the estimation was last run before it existed. The
        route of a transect is nothing but the camera positions of its frame
        range, so it needs no output of the estimation step.
        """
        from ..core.pipeline_outputs import read_dem_origin_xy
        from ..core.population import transect_centerline
        from ..core.transects import (
            TransectStore, cumulative_distances, flight_positions, path_length,
        )

        target_folder = config.get("target_folder", "")
        poses_path = os.path.join(target_folder, f"poses_{suffix}.json")
        if not os.path.isfile(poses_path):
            return []
        try:
            with open(poses_path, 'r', encoding='utf-8') as f:
                images = json.load(f).get("images", [])
        except (ValueError, OSError):
            return []
        if not images:
            return []

        store = TransectStore(target_folder, suffix)
        try:
            if not store.load():
                return []
        except (ValueError, OSError):
            return []

        x_offset, y_offset = read_dem_origin_xy(
            config.get("dem_path", ""),
            config.get("alfs_dem_metadata_path") or "")
        cum = cumulative_distances(flight_positions(images))

        features = []
        for transect in store.ordered():
            transect.clamp(len(images))
            line = transect_centerline(
                images, transect.first_frame, transect.last_frame,
                x_offset, y_offset)
            if len(line) < 2:
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString",
                             "coordinates": [[x, y] for x, y in line]},
                "properties": {
                    "transect_id": transect.transect_id,
                    "name": transect.display_name,
                    "start_frame": transect.start_frame,
                    "end_frame": transect.end_frame,
                    "length_m": round(path_length(
                        cum, transect.first_frame, transect.last_frame), 2),
                },
            })
        return features

    @staticmethod
    def _transect_color_str(transect_id) -> str:
        """The colour the Transect Splitting Tool paints this transect with."""
        from ..core.labelling import track_color_rgb

        rgb = track_color_rgb(int(transect_id or 0))
        return f"{rgb[0]},{rgb[1]},{rgb[2]}"

    def _build_transect_fov_layer(self, features, transect_id, name, target_crs,
                                  color, suffix):
        """Polygon layer with the merged FoV footprint(s) of one transect."""
        from qgis.core import QgsFillSymbol, QgsSingleSymbolRenderer

        layer = QgsVectorLayer(
            "Polygon?crs=" + target_crs.authid(), "Field of View", "memory")
        provider = layer.dataProvider()
        provider.addAttributes([
            QgsField("transect_id", FieldType.Int),
            QgsField("name", FieldType.String),
            QgsField("area_ha", FieldType.Double),
            QgsField("count", FieldType.Int),
        ])
        layer.updateFields()

        qgs_features = []
        for feature in features:
            rings = feature.get("geometry", {}).get("coordinates", [])
            if not rings:
                continue
            points = [[QgsPointXY(float(x), float(y)) for x, y in ring]
                      for ring in rings]
            props = feature.get("properties", {})
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPolygonXY(points))
            feat.setAttributes([
                transect_id, name,
                props.get("area_ha", 0.0), props.get("count", 0),
            ])
            qgs_features.append(feat)
        if not qgs_features:
            return None

        provider.addFeatures(qgs_features)
        layer.updateExtents()

        def style(target):
            target.setRenderer(QgsSingleSymbolRenderer(
                QgsFillSymbol.createSimple({
                    'color': f'{color},60',
                    'outline_color': f'{color},255',
                    'outline_width': '0.5',
                })))
            self._label_transect_layer(target)

        style(layer)
        layer = self._persist_memory_layer(
            layer, f"Transect{transect_id}_FoV_{suffix}", "transect_layers",
            display_name="Field of View")
        style(layer)
        QgsProject.instance().addMapLayer(layer, False)
        return layer

    def _build_transect_route_layer(self, features, transect_id, name, target_crs,
                                    color, suffix):
        """Line layer with the sub-flight route of one transect."""
        from qgis.core import QgsLineSymbol, QgsSingleSymbolRenderer

        qgs_features = []
        for feature in features:
            coords = feature.get("geometry", {}).get("coordinates", [])
            if len(coords) < 2:
                continue
            props = feature.get("properties", {})
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPolylineXY(
                [QgsPointXY(float(x), float(y)) for x, y in coords]))
            feat.setAttributes([
                transect_id, name,
                props.get("start_frame", 0), props.get("end_frame", 0),
                props.get("length_m", 0.0),
            ])
            qgs_features.append(feat)
        if not qgs_features:
            return None

        layer = QgsVectorLayer(
            "LineString?crs=" + target_crs.authid(), "Flight Route", "memory")
        provider = layer.dataProvider()
        provider.addAttributes([
            QgsField("transect_id", FieldType.Int),
            QgsField("name", FieldType.String),
            QgsField("start_frame", FieldType.Int),
            QgsField("end_frame", FieldType.Int),
            QgsField("length_m", FieldType.Double),
        ])
        layer.updateFields()
        provider.addFeatures(qgs_features)
        layer.updateExtents()

        def style(target):
            target.setRenderer(QgsSingleSymbolRenderer(
                QgsLineSymbol.createSimple({'color': color, 'width': '0.8'})))

        style(layer)
        layer = self._persist_memory_layer(
            layer, f"Transect{transect_id}_Route_{suffix}", "transect_layers",
            display_name="Flight Route")
        style(layer)
        QgsProject.instance().addMapLayer(layer, False)
        return layer

    def _label_transect_layer(self, layer):
        """Label the FoV polygon with 'name (count)'."""
        try:
            from qgis.core import (
                QgsPalLayerSettings, QgsTextFormat, QgsVectorLayerSimpleLabeling,
            )
            settings = QgsPalLayerSettings()
            settings.fieldName = '"name" || \' (\' || "count" || \')\''
            settings.isExpression = True
            settings.placement = QgsPalLayerSettings.Placement.OverPoint
            text_format = QgsTextFormat()
            text_format.setSize(9)
            settings.setFormat(text_format)
            layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
            layer.setLabelsEnabled(True)
            layer.triggerRepaint()
        except Exception as e:  # nosec B110
            self.log(f"Warning: could not label transect layer: {e}")
