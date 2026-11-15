# -*- coding: utf-8 -*-
"""Results on the map: the "Add ... to QGIS" buttons of the steps.

Detections per frame, tracks with their paths, fields of view, light-field
renders, GeoTIFFs, the orthomosaic, matched tracks and perpendicular
distances - read from the flight's folder and added to the flight's layer
group, coloured by the label schema.
"""

import json
import os
from typing import Optional, Dict

from qgis.PyQt.QtWidgets import QMessageBox, QDialog
from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsCoordinateReferenceSystem, QgsFeature,
    QgsGeometry, QgsPointXY, QgsField, QgsRasterLayer, QgsLineSymbol,
)

from ..qgis_compat import FieldType
from ..bambi_map_symbols import (
    animal_fields, animal_values, restyle, style_boxes, style_path,
)
from ..core import store as _store_kinds
from .range_dialogs import FrameRangeDialog


def _camera_suffix(camera_label: str) -> str:
    """``"t"``/``"w"`` for a "Thermal"/"RGB" label: the suffix that keeps the
    two cameras' persisted layer files apart."""
    return "w" if camera_label == "RGB" else "t"


class ResultLayersMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def add_orthomosaic_to_qgis(self):
        """Add the generated orthomosaic to QGIS as a raster layer."""
        config = self.get_config()
        camera = config.get("ortho_camera", "T")
        suffix = "t" if camera == "T" else "w"
        camera_label = "Thermal" if camera == "T" else "RGB"
        ortho_file = os.path.join(
            config["target_folder"], f"orthomosaic_{suffix}", "orthomosaic.tif"
        )

        if not os.path.exists(ortho_file):
            QMessageBox.warning(
                self,
                "Missing Orthomosaic",
                f"{camera_label} orthomosaic has not been generated.\n"
                f"Please run 'Generate Orthomosaic' first."
            )
            return

        try:
            self.log(f"Adding {camera_label} orthomosaic to QGIS...")
            self.update_status("add_orthomosaic", "🟡 Loading...")

            layer = QgsRasterLayer(ortho_file, f"BAMBI Orthomosaic ({camera_label})")
            if not layer.isValid():
                raise RuntimeError(f"Failed to load raster: {ortho_file}")
            self._add_layer_to_flight(layer)

            self.update_status("add_orthomosaic", "🟢 Added")
            self.iface.mapCanvas().refresh()
            self.log(f"Added {camera_label} orthomosaic layer to QGIS")

        except Exception as e:
            self.log(f"Error adding orthomosaic: {str(e)}")
            self.update_status("add_orthomosaic", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add orthomosaic: {str(e)}")

    def add_matches_to_qgis(self):
        """Draw each matched pair as a line between the two animals' positions."""
        from ..core import match_store

        config = self.get_config()
        target_folder = config["target_folder"]

        matches = match_store.track_matches(target_folder)
        if not matches:
            QMessageBox.warning(
                self, "No Matches",
                "No matched track pairs were found. Run 'Match RGB ↔ Thermal "
                "Tracks' first."
            )
            return

        try:
            self.log("Adding matched track pairs to QGIS...")
            self.update_status("add_matches", "🟡 Loading...")

            centroids_t = self._track_centroids(target_folder, "t")
            centroids_w = self._track_centroids(target_folder, "w")

            epsg = config.get("target_epsg", 32633)
            layer = QgsVectorLayer(
                f"LineString?crs=EPSG:{epsg}", "BAMBI Matched Pairs", "memory")
            provider = layer.dataProvider()
            provider.addAttributes([
                QgsField("track_t", FieldType.Int),
                QgsField("track_w", FieldType.Int),
                QgsField("shared", FieldType.Int),
                QgsField("median_dist", FieldType.Double),
                QgsField("conf_t", FieldType.Double),
                QgsField("conf_w", FieldType.Double),
            ])
            layer.updateFields()

            features, skipped = [], 0
            for match in matches:
                start = centroids_t.get(match["track_id_t"])
                end = centroids_w.get(match["track_id_w"])
                if start is None or end is None:
                    # One side was never geo-referenced, so there is no world
                    # position to draw from. The match itself is still valid.
                    skipped += 1
                    continue
                feature = QgsFeature()
                feature.setGeometry(QgsGeometry.fromPolylineXY(
                    [QgsPointXY(*start), QgsPointXY(*end)]))
                feature.setAttributes([
                    match["track_id_t"], match["track_id_w"],
                    match["shared"], match["median_dist"],
                    match["conf_t"], match["conf_w"],
                ])
                features.append(feature)

            provider.addFeatures(features)
            layer.updateExtents()

            # Persist beside the other derived layers, so the pairs survive a
            # project reload instead of living only in this QGIS session.
            layer = self._persist_memory_layer(
                layer, "Matched_Pairs", "matches_layers",
                display_name="BAMBI Matched Pairs")
            # The file-backed layer comes back with a default renderer, so
            # the styling is applied after persisting, as the track layers do.
            symbol = QgsLineSymbol.createSimple(
                {"color": "255,140,0,255", "width": "0.8"})
            layer.renderer().setSymbol(symbol)
            layer.setCustomProperty("bambi_layer_type", "matched_pairs")
            layer.setCustomProperty("bambi_target_folder", target_folder)

            self._add_layer_to_flight(layer)
            self.update_status("add_matches", "🟢 Added")
            self.iface.mapCanvas().refresh()

            note = f", {skipped} without a world position" if skipped else ""
            self.log(f"Added {len(features)} matched pair(s) to QGIS{note}")

        except Exception as exc:
            self.log(f"Error adding matched pairs: {exc}")
            self.update_status("add_matches", "🔴 Error")
            QMessageBox.critical(
                self, "Error", f"Failed to add matched pairs: {exc}")

    @staticmethod
    def _track_centroids(target_folder: str, modality: str) -> Dict[int, tuple]:
        """Mean world position of each track, for drawing it on the map."""
        from ..core import track_store

        # Geo rows are per detection, so they are joined onto their tracks
        # through the pixel-track membership.
        sums: Dict[int, list] = {}
        geo = {row["detection_id"]: row for row
               in track_store.load_georeferenced(target_folder, modality)}
        for row in track_store.load_pixel_tracks(target_folder, modality):
            placed = geo.get(row["detection_id"])
            if placed is None:
                continue
            cx = (placed["gx1"] + placed["gx2"]) / 2.0
            cy = (placed["gy1"] + placed["gy2"]) / 2.0
            entry = sums.setdefault(row["track_id"], [0.0, 0.0, 0])
            entry[0] += cx
            entry[1] += cy
            entry[2] += 1

        return {track_id: (x / n, y / n)
                for track_id, (x, y, n) in sums.items() if n}

    def add_perpendicular_to_qgis(self):
        """Add perpendicular lines (detection → nearest flight route point) to QGIS."""
        config = self.get_config()
        fr_camera = config.get("flight_route_camera", "T")
        fr_suffix = "t" if fr_camera == "T" else "w"
        det_camera = config.get("detection_camera", "T")
        det_suffix = "t" if det_camera == "T" else "w"
        camera_label = "Thermal" if det_camera == "T" else "RGB"
        perp_file = os.path.join(config["target_folder"], f"flight_route_{fr_suffix}",
                                 f"perpendicular_{det_suffix}.json")

        if not os.path.exists(perp_file):
            QMessageBox.warning(
                self, "Missing Data",
                "Perpendicular distances have not been calculated.\n"
                "Please run 'Calculate Perpendicular' first."
            )
            return

        try:
            self.log(f"Adding {camera_label} perpendicular lines to QGIS...")
            self.update_status("add_perpendicular", "🟡 Loading...")

            with open(perp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            perpendiculas = data.get("perpendiculas", [])
            if not perpendiculas:
                QMessageBox.warning(self, "No Data", "No perpendicular results found.")
                self.update_status("add_perpendicular", "🔴 No data")
                return

            frame_vals = sorted(e["frame"] for e in perpendiculas)
            dlg = FrameRangeDialog(
                self, frame_vals[0], frame_vals[-1], len(set(frame_vals)), "frames"
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.update_status("add_perpendicular", "⚪ Cancelled")
                return

            start, end = dlg.selected_range()
            perpendiculas = [e for e in perpendiculas if start <= e["frame"] <= end]

            if not perpendiculas:
                QMessageBox.warning(self, "No Data", "No perpendicular results in selected range.")
                self.update_status("add_perpendicular", "🔴 No data")
                return

            if len(perpendiculas) > 100:
                reply = QMessageBox.question(
                    self,
                    "Many Lines",
                    f"Selected range contains {len(perpendiculas)} perpendicular lines.\n"
                    "Loading all may slow down QGIS.\n\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.update_status("add_perpendicular", "⚪ Cancelled")
                    return

            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{config['target_epsg']}")

            # Create a line layer: one line per detection (foot point → detection center)
            layer = QgsVectorLayer(
                "LineString?crs=" + target_crs.authid(),
                "Perpendicular Lines",
                "memory"
            )
            provider = layer.dataProvider()
            provider.addAttributes([
                QgsField("det_idx", FieldType.Int),
                QgsField("frame", FieldType.Int),
                QgsField("confidence", FieldType.Double),
                QgsField("class_id", FieldType.Int),
                QgsField("distance_m", FieldType.Double),
            ])
            layer.updateFields()

            features = []
            for entry in perpendiculas:
                foot = entry["foot_point"]
                center = entry["detection_center"]

                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromPolylineXY([
                    QgsPointXY(foot[0], foot[1]),
                    QgsPointXY(center[0], center[1])
                ]))
                feat.setAttributes([
                    entry["det_idx"],
                    entry["frame"],
                    round(entry["confidence"], 4),
                    entry["class_id"],
                    round(entry["distance_m"], 4),
                ])
                features.append(feat)

            provider.addFeatures(features)
            layer.updateExtents()

            # Style: thin orange lines
            symbol = QgsLineSymbol.createSimple({
                'color': '#ff7800',
                'width': '0.6',
                'capstyle': 'round',
            })
            layer.renderer().setSymbol(symbol)

            # Persist and add to project
            layer = self._persist_memory_layer(
                layer, f"Perpendicular_Lines_{det_suffix}", "flight_route_layers")
            symbol = QgsLineSymbol.createSimple({
                'color': '#ff7800',
                'width': '0.6',
                'capstyle': 'round',
            })
            layer.renderer().setSymbol(symbol)

            group = self._create_layer_group(f"BAMBI Perpendicular ({camera_label})")
            QgsProject.instance().addMapLayer(layer, False)
            group.addLayer(layer)

            self.log(f"Added {len(features)} {camera_label} perpendicular lines to QGIS")
            self.update_status("add_perpendicular", "🟢 Completed")
            self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding perpendicular lines: {str(e)}")
            self.update_status("add_perpendicular", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add perpendicular lines: {str(e)}")

    def add_track_perpendicular_to_qgis(self):
        """Add track perpendicular lines (last detection → nearest flight route point) to QGIS."""
        config = self.get_config()
        fr_camera = config.get("flight_route_camera", "T")
        fr_suffix = "t" if fr_camera == "T" else "w"
        trk_camera = config.get("tracking_camera", "T")
        trk_suffix = "t" if trk_camera == "T" else "w"
        camera_label = "Thermal" if trk_camera == "T" else "RGB"
        perp_file = os.path.join(config["target_folder"], f"flight_route_{fr_suffix}",
                                 f"perpendicular_tracks_{trk_suffix}.json")

        if not os.path.exists(perp_file):
            QMessageBox.warning(
                self, "Missing Data",
                "Track perpendicular distances have not been calculated.\n"
                "Please run 'Calculate Track Perpendicular' first."
            )
            return

        try:
            self.log(f"Adding {camera_label} track perpendicular lines to QGIS...")
            self.update_status("add_track_perpendicular", "🟡 Loading...")

            with open(perp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            track_entries = data.get("tracks", [])
            if not track_entries:
                QMessageBox.warning(self, "No Data", "No track perpendicular results found.")
                self.update_status("add_track_perpendicular", "🔴 No data")
                return

            frame_vals = sorted(e["last_frame"] for e in track_entries)
            dlg = FrameRangeDialog(
                self, frame_vals[0], frame_vals[-1], len(set(frame_vals)), "frames"
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.update_status("add_track_perpendicular", "⚪ Cancelled")
                return

            start, end = dlg.selected_range()
            track_entries = [e for e in track_entries if start <= e["last_frame"] <= end]

            if not track_entries:
                QMessageBox.warning(self, "No Data", "No track perpendicular results in selected range.")
                self.update_status("add_track_perpendicular", "🔴 No data")
                return

            if len(track_entries) > 100:
                reply = QMessageBox.question(
                    self,
                    "Many Lines",
                    f"Selected range contains {len(track_entries)} track perpendicular lines.\n"
                    "Loading all may slow down QGIS.\n\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.update_status("add_track_perpendicular", "⚪ Cancelled")
                    return

            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{config['target_epsg']}")

            layer = QgsVectorLayer(
                "LineString?crs=" + target_crs.authid(),
                "Track Perpendicular Lines",
                "memory"
            )
            provider = layer.dataProvider()
            provider.addAttributes([
                QgsField("track_id", FieldType.Int),
                QgsField("last_frame", FieldType.Int),
                QgsField("last_image", FieldType.String),
                QgsField("confidence", FieldType.Double),
                QgsField("class_id", FieldType.Int),
                QgsField("distance_m", FieldType.Double),
            ])
            layer.updateFields()

            features = []
            for entry in track_entries:
                foot = entry["foot_point"]
                center = entry["detection_center"]

                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromPolylineXY([
                    QgsPointXY(foot[0], foot[1]),
                    QgsPointXY(center[0], center[1])
                ]))
                feat.setAttributes([
                    entry["track_id"],
                    entry["last_frame"],
                    entry.get("last_image", ""),
                    round(entry["confidence"], 4),
                    entry["class_id"],
                    round(entry["distance_m"], 4),
                ])
                features.append(feat)

            provider.addFeatures(features)
            layer.updateExtents()

            # Style: slightly thicker purple lines to distinguish from detection perpendiculars
            symbol = QgsLineSymbol.createSimple({
                'color': '#9b30ff',
                'width': '0.8',
                'capstyle': 'round',
            })
            layer.renderer().setSymbol(symbol)

            layer = self._persist_memory_layer(
                layer, f"Track_Perpendicular_Lines_{trk_suffix}", "flight_route_layers")
            symbol = QgsLineSymbol.createSimple({
                'color': '#9b30ff',
                'width': '0.8',
                'capstyle': 'round',
            })
            layer.renderer().setSymbol(symbol)

            group = self._create_layer_group(f"BAMBI Track Perpendicular ({camera_label})")
            QgsProject.instance().addMapLayer(layer, False)
            group.addLayer(layer)

            self.log(f"Added {len(features)} {camera_label} track perpendicular lines to QGIS")
            self.update_status("add_track_perpendicular", "🟢 Completed")
            self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding track perpendicular lines: {str(e)}")
            self.update_status("add_track_perpendicular", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add track perpendicular lines: {str(e)}")

    def add_tracks_to_qgis(self):
        """Add tracked animals as individual layer groups to QGIS.

        Each track gets its own subgroup containing:
        - A polyline showing the movement path
        - The final bounding box as a polygon

        This allows users to show/hide individual animals.
        """
        config = self.get_config()
        trk_camera = config.get("tracking_camera", "T")
        trk_suffix = "t" if trk_camera == "T" else "w"
        camera_label = "Thermal" if trk_camera == "T" else "RGB"

        # The tracks are in the store. This used to look for tracks.csv -
        # which is only written while "Also write legacy text outputs" is
        # on, and never for animals that were labelled by hand - and gave
        # up without it, although it read nothing from the file.
        if not self.has_stage(_store_kinds.TRACKS, trk_suffix,
                              config["target_folder"]):
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_label} tracking has not been completed.\nPlease run step A2 (Track Animals Or Import) first."
            )
            return

        try:
            self.log(f"Adding {camera_label} tracks to QGIS...")
            self.update_status("add_layers", "🟡 Loading...")

            # Get target CRS
            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{config['target_epsg']}")

            # The geo-referenced tracks of the store: track id -> points.
            # (Every .csv in the legacy folder used to stand for the whole
            # store, so a second file there doubled every track.)
            from ..core.pipeline_outputs import geo_tracks
            all_tracks = {
                track_id: detections for track_id, detections in
                geo_tracks(config["target_folder"], trk_suffix).items()
                if detections}

            if not all_tracks:
                QMessageBox.warning(
                    self, "No Tracks",
                    f"No {camera_label} track has a position on the map "
                    "yet.\nRun Geo-Reference Detections (under step A1) "
                    "and then Track Animals.")
                self.update_status("add_layers", "🔴 No valid tracks")
                return

            # Ask for a frame range - a track is kept when its final frame
            # falls inside the range (its earlier history is not considered)
            final_frames = {
                key: max(d['frame'] for d in detections)
                for key, detections in all_tracks.items()
            }
            dlg = FrameRangeDialog(
                self, min(final_frames.values()), max(final_frames.values()),
                len(all_tracks), "track end frames"
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.update_status("add_layers", "⚪ Cancelled")
                return

            start, end = dlg.selected_range()
            all_tracks = {
                key: detections for key, detections in all_tracks.items()
                if start <= final_frames[key] <= end
            }

            if not all_tracks:
                QMessageBox.warning(
                    self, "No Tracks",
                    "No tracks end within the selected frame range."
                )
                self.update_status("add_layers", "🔴 No valid tracks")
                return

            # Check if there are many tracks - warn user
            num_tracks = len(all_tracks)
            if num_tracks > 50:
                reply = QMessageBox.question(
                    self,
                    "Many Tracks",
                    f"Found {num_tracks} tracks. Creating individual layers for each may slow down QGIS.\n\n"
                    "Continue with individual layers?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                if reply == QMessageBox.StandardButton.No:
                    self.update_status("add_layers", "⚪ Cancelled")
                    return

            # Create main group for all tracks at top of layer tree
            main_group = self._create_layer_group(f"BAMBI Wildlife Tracks ({camera_label})")

            # Each track keeps a colour of its own for its box and path;
            # the pin it turns into when zoomed out is in the colour of its
            # species and carries its sex and age (core.map_symbols).
            from ..core import map_symbols
            animals = map_symbols.track_attributes(
                config["target_folder"], trk_suffix)
            style_settings = self._animal_style_settings(config)

            total_tracks = 0

            for idx, (track_id, detections) in enumerate(all_tracks.items()):
                # Sort detections by frame
                detections_sorted = sorted(detections, key=lambda d: d['frame'])
                known = animals.get(int(track_id))
                animal = animal_values(
                    known if known else
                    map_symbols.attributes(track_id=track_id))

                # Create subgroup for this track
                track_name = f"Track {track_id}"
                track_group = main_group.addGroup(track_name)

                # Create path layer (polyline) - only when there are 2+ detections
                path_layer = None
                if len(detections_sorted) >= 2:
                    path_layer = QgsVectorLayer(
                        "LineString?crs=" + target_crs.authid(),
                        "Path",
                        "memory"
                    )
                    path_provider = path_layer.dataProvider()
                    path_provider.addAttributes([
                        QgsField("track_id", FieldType.Int),
                        QgsField("start_frame", FieldType.Int),
                        QgsField("end_frame", FieldType.Int),
                        QgsField("num_detections", FieldType.Int),
                        QgsField("avg_confidence", FieldType.Double)
                    ] + animal_fields())
                    path_layer.updateFields()

                    center_points = [
                        QgsPointXY((det['x1'] + det['x2']) / 2.0, (det['y1'] + det['y2']) / 2.0)
                        for det in detections_sorted
                    ]

                    path_feat = QgsFeature()
                    path_feat.setGeometry(QgsGeometry.fromPolylineXY(center_points))
                    avg_conf = sum(d['confidence'] for d in detections_sorted) / len(detections_sorted)
                    path_feat.setAttributes([
                        track_id,
                        detections_sorted[0]['frame'],
                        detections_sorted[-1]['frame'],
                        len(detections_sorted),
                        round(avg_conf, 4)
                    ] + animal)
                    path_provider.addFeatures([path_feat])
                    path_layer.updateExtents()

                # Create final position layer (polygon)
                bbox_layer = QgsVectorLayer(
                    "Polygon?crs=" + target_crs.authid(),
                    "Final Position",
                    "memory"
                )
                bbox_provider = bbox_layer.dataProvider()
                bbox_provider.addAttributes([
                    QgsField("track_id", FieldType.Int),
                    QgsField("frame", FieldType.Int),
                    QgsField("confidence", FieldType.Double),
                    QgsField("class_id", FieldType.Int)
                ] + animal_fields())
                bbox_layer.updateFields()

                # Create final bbox feature
                final_det = detections_sorted[-1]
                bbox_feat = QgsFeature()
                points = [
                    QgsPointXY(final_det['x1'], final_det['y1']),
                    QgsPointXY(final_det['x2'], final_det['y1']),
                    QgsPointXY(final_det['x2'], final_det['y2']),
                    QgsPointXY(final_det['x1'], final_det['y2']),
                    QgsPointXY(final_det['x1'], final_det['y1'])
                ]
                bbox_feat.setGeometry(QgsGeometry.fromPolygonXY([points]))
                bbox_feat.setAttributes([
                    track_id,
                    final_det['frame'],
                    final_det['confidence'],
                    final_det['class_id']
                ] + animal)
                bbox_provider.addFeatures([bbox_feat])
                bbox_layer.updateExtents()

                # Persist layers to GeoPackage (using unique file names, keeping display names)
                bbox_layer = self._persist_memory_layer(
                    bbox_layer, f"Track{track_id}_FinalPosition_{trk_suffix}", "tracks_layers",
                    display_name="Final Position"
                )
                self._style_animal_boxes(bbox_layer, style_settings)
                # Tag layer so the inspector tool can identify and handle it
                bbox_layer.setCustomProperty("bambi_layer_type", "track_final")
                bbox_layer.setCustomProperty("bambi_target_folder", config["target_folder"])
                # The tracks were built for the *tracking* camera; tagging the
                # detection combo's camera sent the inspector to the other
                # store whenever the two combos disagreed.
                bbox_layer.setCustomProperty("bambi_detection_camera", trk_camera)
                bbox_layer.setCustomProperty("bambi_dem_path", config.get("dem_path", ""))
                bbox_layer.setCustomProperty("bambi_correction_path", config.get("correction_path", ""))
                QgsProject.instance().addMapLayer(bbox_layer, False)
                track_group.addLayer(bbox_layer)

                if path_layer is not None:
                    path_layer = self._persist_memory_layer(
                        path_layer, f"Track{track_id}_Path_{trk_suffix}", "tracks_layers",
                        display_name="Path"
                    )
                    self._style_animal_path(path_layer)
                    # Tag layer so the inspector tool can identify and handle it
                    path_layer.setCustomProperty("bambi_layer_type", "track_path")
                    path_layer.setCustomProperty("bambi_target_folder", config["target_folder"])
                    path_layer.setCustomProperty("bambi_detection_camera", trk_camera)
                    path_layer.setCustomProperty("bambi_dem_path", config.get("dem_path", ""))
                    path_layer.setCustomProperty("bambi_correction_path", config.get("correction_path", ""))
                    QgsProject.instance().addMapLayer(path_layer, False)
                    track_group.addLayer(path_layer)

                # Collapse the track subgroup by default
                track_group.setExpanded(False)

                total_tracks += 1

            # Collapse main group
            main_group.setExpanded(True)  # Keep main group expanded to show tracks

            self.log(f"Added {total_tracks} individual track layers to QGIS")
            self.update_status("add_layers", "🟢 Completed")

            # Refresh canvas
            if total_tracks > 0:
                self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding layers: {str(e)}")
            self.update_status("add_layers", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add tracks: {str(e)}")

    # ------------------------------------------------------------------ #
    # Animal symbology: colour by species, pins when zoomed out
    # ------------------------------------------------------------------ #
    def _animal_style_settings(self, config: Optional[dict] = None):
        """``(species colours, switch-over scale, pins on)`` for the layers
        of the flight in *config* (the active one by default)."""
        from ..core import map_symbols

        config = config if config is not None else self.get_config()
        colours = map_symbols.species_colours(config.get("target_folder", ""))
        return (colours,
                float(config.get("map_pin_scale",
                                 map_symbols.DEFAULT_SWITCH_SCALE)),
                bool(config.get("map_pins_enabled", True)))

    def _style_animal_boxes(self, layer, settings=None):
        """Style a layer of animal boxes (detections, a track's final
        position): species colours, and pins beyond the switch-over scale
        (:mod:`bambi_map_symbols`).

        *settings* is what :meth:`_animal_style_settings` returned. Pass it
        when styling many layers: reading it afresh per layer means reading
        the whole form and the flight's store every time.
        """
        try:
            colours, scale, pins = (
                settings if settings is not None
                else self._animal_style_settings())
            style_boxes(layer, colours, scale, pins)
        except Exception as exc:  # nosec B110 - an unstyled layer still shows
            self.log(f"Note: could not style {layer.name()}: {exc}")

    def _style_animal_path(self, layer):
        """Style a track's path in the colour of its individual."""
        try:
            style_path(layer)
        except Exception as exc:  # nosec B110
            self.log(f"Note: could not style {layer.name()}: {exc}")

    def restyle_animal_layers(self, quiet: bool = False) -> int:
        """Apply the current species colours and pin settings to the animal
        layers already in the project - after a colour was changed in the
        Project Schema, or the switch-over scale on the Project tab.

        Every flight's layers take their own flight's colours. Layers added
        by an earlier version have no species to colour by and are left as
        they are; adding them again brings them up to date.
        """
        from ..bambi_map_symbols import STYLE_PROPERTY
        from ..core import map_symbols

        config = self.get_config()
        scale = float(config.get("map_pin_scale",
                                 map_symbols.DEFAULT_SWITCH_SCALE))
        pins = bool(config.get("map_pins_enabled", True))
        by_folder: Dict[str, list] = {}
        for layer in QgsProject.instance().mapLayers().values():
            if not layer.customProperty(STYLE_PROPERTY, ""):
                continue
            folder = layer.customProperty("bambi_target_folder", "") or \
                config.get("target_folder", "")
            by_folder.setdefault(folder, []).append(layer)
        count = 0
        for folder, layers in by_folder.items():
            count += restyle(layers, map_symbols.species_colours(folder),
                             scale, pins)
        if count:
            self.iface.mapCanvas().refresh()
        if not quiet:
            self.log(f"Restyled {count} animal layer(s)" if count else
                     "No animal layers to restyle - add detections or "
                     "tracks to QGIS first.")
        return count

    def add_fov_to_qgis(self):
        """Add Field of View polygons as QGIS layers."""
        config = self.get_config()
        fov_camera = config.get("fov_camera", "T")
        fov_suffix = "t" if fov_camera == "T" else "w"
        camera_label = "Thermal" if fov_camera == "T" else "RGB"
        fov_folder = os.path.join(config["target_folder"], f"fov_{fov_suffix}")
        fov_file = os.path.join(fov_folder, "fov_polygons.txt")

        if not self.has_stage(_store_kinds.FOV, fov_suffix,
                              config["target_folder"]):
            QMessageBox.warning(
                self,
                "Missing FoV Data",
                f"{camera_label} FoV calculation has not been completed.\n"
                f"Please run Step 5 (Calculate Field of View) first."
            )
            return

        try:
            self.log(f"Adding {camera_label} FoV layers to QGIS...")
            self.update_status("add_fov", "🟡 Loading...")

            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{config['target_epsg']}")
            fov_polygons = self.load_fov_polygons(fov_file)

            if not fov_polygons:
                QMessageBox.warning(self, "No FoV Data", "No valid FoV polygons found.")
                self.update_status("add_fov", "🔴 No data")
                return

            frame_keys = sorted(fov_polygons.keys())
            dlg = FrameRangeDialog(
                self, frame_keys[0], frame_keys[-1], len(frame_keys), "frames"
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.update_status("add_fov", "⚪ Cancelled")
                return

            start, end = dlg.selected_range()
            fov_polygons = {k: v for k, v in fov_polygons.items() if start <= k <= end}

            if not fov_polygons:
                QMessageBox.warning(self, "No FoV Data", "No FoV polygons in selected range.")
                self.update_status("add_fov", "🔴 No data")
                return

            num_frames = len(fov_polygons)
            if num_frames > 100:
                reply = QMessageBox.question(
                    self,
                    "Layer Mode",
                    f"Selected range contains {num_frames} frames.\n\n"
                    "- Yes: one layer per frame (separate)\n"
                    "- No: all polygons in a single combined layer",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
                )
                if reply == QMessageBox.StandardButton.Cancel:
                    self.update_status("add_fov", "⚪ Cancelled")
                    return
                elif reply == QMessageBox.StandardButton.Yes:
                    self._add_fov_separate_layers(fov_polygons, target_crs, camera_label)
                else:
                    self._add_fov_combined_layer(fov_polygons, target_crs, camera_label)
            else:
                self._add_fov_separate_layers(fov_polygons, target_crs, camera_label)

            self.log("Added FoV layers to QGIS")
            self.update_status("add_fov", "🟢 Completed")
            self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding FoV layers: {str(e)}")
            self.update_status("add_fov", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add FoV layers: {str(e)}")

    def _add_fov_separate_layers(self, fov_polygons: Dict[int, list], target_crs,
                                 camera_label: str = "Thermal"):
        """Add FoV polygons as separate layers for each frame."""
        # Create a group for the layers at top of layer tree
        group = self._create_layer_group(f"BAMBI FoV Polygons ({camera_label})")

        for frame_idx, points in fov_polygons.items():
            if len(points) < 3:
                continue

            # Create polygon layer for this frame
            layer_name = f"FoV_Frame_{frame_idx:06d}"
            layer = QgsVectorLayer("Polygon?crs=" + target_crs.authid(), layer_name, "memory")

            provider = layer.dataProvider()

            # Add fields
            provider.addAttributes([
                QgsField("frame", FieldType.Int),
                QgsField("num_points", FieldType.Int)
            ])
            layer.updateFields()

            # Create feature
            feat = QgsFeature()

            # Create polygon from points (only x, y coordinates)
            qgs_points = [QgsPointXY(p[0], p[1]) for p in points]
            qgs_points.append(qgs_points[0])  # Close the polygon

            feat.setGeometry(QgsGeometry.fromPolygonXY([qgs_points]))
            feat.setAttributes([frame_idx, len(points)])

            provider.addFeatures([feat])
            layer.updateExtents()

            # Persist layer to GeoPackage (camera-suffixed, so the thermal and
            # RGB layers live side by side instead of replacing each other)
            layer = self._persist_memory_layer(
                layer, f"{layer_name}_{_camera_suffix(camera_label)}", "fov_layers")

            # Tag layer so the FoV inspector tool can identify and handle it
            layer.setCustomProperty("bambi_layer_type", "fov")
            layer.setCustomProperty("bambi_target_folder", self.target_folder_edit.text().strip())
            # The polygon's frame index belongs to the FoV camera's poses, so
            # the inspector has to read that camera's detections for it.
            layer.setCustomProperty(
                "bambi_detection_camera",
                "T" if self.fov_camera_combo.currentIndex() == 0 else "W")
            layer.setCustomProperty("bambi_dem_path", self.dem_path_edit.text().strip())
            layer.setCustomProperty("bambi_correction_path", self.correction_path_edit.text().strip())

            # Add layer to project and group
            QgsProject.instance().addMapLayer(layer, False)
            group.addLayer(layer)

        # Collapse the group
        group.setExpanded(False)

    def _add_fov_combined_layer(self, fov_polygons: Dict[int, list], target_crs,
                                camera_label: str = "Thermal"):
        """Add all FoV polygons as a single combined layer."""
        # Create a single layer with all polygons
        layer_name = f"BAMBI FoV Polygons - Combined ({camera_label})"
        layer = QgsVectorLayer("Polygon?crs=" + target_crs.authid(), layer_name, "memory")

        provider = layer.dataProvider()

        # Add fields
        provider.addAttributes([
            QgsField("frame", FieldType.Int),
            QgsField("num_points", FieldType.Int)
        ])
        layer.updateFields()

        features = []
        for frame_idx, points in fov_polygons.items():
            if len(points) < 3:
                continue

            feat = QgsFeature()

            # Create polygon from points
            qgs_points = [QgsPointXY(p[0], p[1]) for p in points]
            qgs_points.append(qgs_points[0])  # Close the polygon

            feat.setGeometry(QgsGeometry.fromPolygonXY([qgs_points]))
            feat.setAttributes([frame_idx, len(points)])
            features.append(feat)

        provider.addFeatures(features)
        layer.updateExtents()

        # Persist layer to GeoPackage
        layer = self._persist_memory_layer(
            layer, f"FoV_Combined_{_camera_suffix(camera_label)}", "fov_layers")

        # Tag layer so the FoV inspector tool can identify and handle it
        layer.setCustomProperty("bambi_layer_type", "fov")
        layer.setCustomProperty("bambi_target_folder", self.target_folder_edit.text().strip())
        # The polygon's frame index belongs to the FoV camera's poses, so
        # the inspector has to read that camera's detections for it.
        layer.setCustomProperty(
            "bambi_detection_camera",
            "T" if self.fov_camera_combo.currentIndex() == 0 else "W")
        layer.setCustomProperty("bambi_dem_path", self.dem_path_edit.text().strip())
        layer.setCustomProperty("bambi_correction_path", self.correction_path_edit.text().strip())

        # Add layer to project
        self._add_layer_to_flight(layer)

    def add_merged_fov_to_qgis(self):
        """Add merged (union) Field of View polygon as single QGIS layer.

        This creates a single polygon/multipolygon from the union of all
        individual FoV polygons, useful for calculating total surveyed area.
        """
        config = self.get_config()
        fov_camera = config.get("fov_camera", "T")
        fov_suffix = "t" if fov_camera == "T" else "w"
        camera_label = "Thermal" if fov_camera == "T" else "RGB"
        fov_folder = os.path.join(config["target_folder"], f"fov_{fov_suffix}")
        fov_file = os.path.join(fov_folder, "fov_polygons.txt")

        if not self.has_stage(_store_kinds.FOV, fov_suffix,
                              config["target_folder"]):
            QMessageBox.warning(
                self,
                "Missing FoV Data",
                f"{camera_label} FoV calculation has not been completed.\n"
                f"Please run Step 5 (Calculate Field of View) first."
            )
            return

        try:
            self.log(f"Creating merged {camera_label} FoV layer...")
            self.update_status("add_merged_fov", "🟡 Processing...")

            # Get target CRS
            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{config['target_epsg']}")

            # Load FoV polygons
            fov_polygons = self.load_fov_polygons(fov_file)

            if not fov_polygons:
                QMessageBox.warning(self, "No FoV Data", "No valid FoV polygons found.")
                self.update_status("add_merged_fov", "🔴 No data")
                return

            self.log(f"Merging {len(fov_polygons)} FoV polygons...")

            # Collect all polygon geometries
            geometries = []
            for frame_idx, points in fov_polygons.items():
                if len(points) < 3:
                    continue

                # Create polygon from points
                qgs_points = [QgsPointXY(p[0], p[1]) for p in points]
                qgs_points.append(qgs_points[0])  # Close the polygon

                geom = QgsGeometry.fromPolygonXY([qgs_points])
                if geom.isGeosValid():
                    geometries.append(geom)
                else:
                    # Try to fix invalid geometry
                    fixed_geom = geom.makeValid()
                    if fixed_geom.isGeosValid():
                        geometries.append(fixed_geom)

            if not geometries:
                QMessageBox.warning(self, "No Valid Polygons", "No valid polygons to merge.")
                self.update_status("add_merged_fov", "🔴 No valid data")
                return

            # Merge all geometries using unaryUnion
            self.log("Computing union of all FoV polygons...")
            merged_geom = QgsGeometry.unaryUnion(geometries)

            if merged_geom.isEmpty():
                QMessageBox.warning(self, "Merge Failed", "Failed to merge FoV polygons.")
                self.update_status("add_merged_fov", "🔴 Merge failed")
                return

            # Create layer for merged polygon
            layer_name = f"BAMBI FoV Coverage ({camera_label})"
            layer = QgsVectorLayer("Polygon?crs=" + target_crs.authid(), layer_name, "memory")

            provider = layer.dataProvider()

            # Add fields
            provider.addAttributes([
                QgsField("num_frames", FieldType.Int),
                QgsField("area_m2", FieldType.Double),
                QgsField("area_ha", FieldType.Double)
            ])
            layer.updateFields()

            # Create feature
            feat = QgsFeature()
            feat.setGeometry(merged_geom)

            # Calculate area
            area_m2 = merged_geom.area()
            area_ha = area_m2 / 10000.0

            feat.setAttributes([len(fov_polygons), area_m2, area_ha])

            provider.addFeatures([feat])
            layer.updateExtents()

            # Style the layer with semi-transparent fill
            symbol = layer.renderer().symbol()
            symbol.setColor(QColor(0, 150, 255, 50))  # Light blue with transparency
            symbol.symbolLayer(0).setStrokeColor(QColor(0, 100, 200))
            symbol.symbolLayer(0).setStrokeWidth(0.5)

            # Persist layer to GeoPackage
            layer = self._persist_memory_layer(
                layer, f"FoV_Coverage_Merged_{fov_suffix}", "fov_layers")

            # Re-apply style after persistence
            symbol = layer.renderer().symbol()
            symbol.setColor(QColor(0, 150, 255, 50))
            symbol.symbolLayer(0).setStrokeColor(QColor(0, 100, 200))
            symbol.symbolLayer(0).setStrokeWidth(0.5)

            # Add layer to project
            self._add_layer_to_flight(layer)

            self.log("Merged FoV layer added to QGIS")
            self.log(f"  Total coverage area: {area_m2:.2f} m² ({area_ha:.4f} ha)")
            self.update_status("add_merged_fov", "🟢 Completed")
            self.iface.mapCanvas().refresh()

            # Show info to user
            QMessageBox.information(
                self,
                "Merged FoV Created",
                f"Merged FoV polygon created from {len(fov_polygons)} frames.\n\n"
                "Total coverage area:\n"
                f"  {area_m2:,.2f} m²\n"
                f"  {area_ha:.4f} ha\n\n"
                "You can also use QGIS Field Calculator for precise area calculation."
            )

        except Exception as e:
            self.log(f"Error creating merged FoV layer: {str(e)}")
            self.update_status("add_merged_fov", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to create merged FoV layer: {str(e)}")

    def load_fov_polygons(self, fov_file: str) -> Dict[int, list]:
        """Load FoV polygons from file (core.pipeline_outputs).

        :param fov_file: Path to FoV polygons file
        :return: Dictionary mapping frame index to list of (x, y, z) points
        """
        from ..core.pipeline_outputs import load_fov_polygons_3d
        return load_fov_polygons_3d(fov_file, log_fn=self.log)

    def add_frame_detections_to_qgis(self):
        """Add geo-referenced detections as QGIS layers (one layer per frame)."""
        config = self.get_config()
        det_camera = config.get("detection_camera", "T")
        det_suffix = "t" if det_camera == "T" else "w"
        camera_label = "Thermal" if det_camera == "T" else "RGB"
        georef_folder = os.path.join(config["target_folder"], f"georeferenced_{det_suffix}")
        georef_file = os.path.join(georef_folder, "georeferenced.txt")

        if not self.has_stage(_store_kinds.GEOREFERENCED, det_suffix,
                              config["target_folder"]):
            QMessageBox.warning(
                self,
                "Missing Data",
                f"{camera_label} geo-referencing has not been completed.\n"
                f"Please run Geo-Reference Detections (under step A1) first."
            )
            return

        try:
            self.log(f"Adding per-frame {camera_label} detection layers to QGIS...")
            self.update_status("add_frame_detections", "🟡 Loading...")

            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{config['target_epsg']}")
            frame_detections = self.load_detections_by_frame(georef_file)

            if not frame_detections:
                QMessageBox.warning(self, "No Detections", "No valid detections found.")
                self.update_status("add_frame_detections", "🔴 No data")
                return

            frame_keys = sorted(frame_detections.keys())
            dlg = FrameRangeDialog(
                self, frame_keys[0], frame_keys[-1], len(frame_keys), "frames"
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.update_status("add_frame_detections", "⚪ Cancelled")
                return

            start, end = dlg.selected_range()
            frame_detections = {k: v for k, v in frame_detections.items() if start <= k <= end}

            if not frame_detections:
                QMessageBox.warning(self, "No Detections", "No detections in selected range.")
                self.update_status("add_frame_detections", "🔴 No data")
                return

            num_frames = len(frame_detections)
            if num_frames > 100:
                reply = QMessageBox.question(
                    self,
                    "Layer Mode",
                    f"Selected range contains {num_frames} frames.\n\n"
                    "- Yes: one layer per frame (separate)\n"
                    "- No: all detections in a single combined layer",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
                )
                if reply == QMessageBox.StandardButton.Cancel:
                    self.update_status("add_frame_detections", "⚪ Cancelled")
                    return
                elif reply == QMessageBox.StandardButton.Yes:
                    self._add_detections_separate_layers(frame_detections, target_crs, camera_label)
                else:
                    self._add_detections_combined_layer(frame_detections, target_crs, camera_label)
            else:
                self._add_detections_separate_layers(frame_detections, target_crs, camera_label)

            self.log("Added detection layers to QGIS")
            self.update_status("add_frame_detections", "🟢 Completed")
            self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding detection layers: {str(e)}")
            self.update_status("add_frame_detections", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add detection layers: {str(e)}")

    def _detection_animals(self, camera_label: str) -> Dict[int, dict]:
        """``detection_id -> species / sex / age`` of the camera's
        detections, for the layers' attributes (core.map_symbols)."""
        from ..core import map_symbols
        return map_symbols.detection_attributes(
            self.target_folder_edit.text().strip(),
            _camera_suffix(camera_label))

    def _add_detections_separate_layers(self, frame_detections: Dict[int, list], target_crs,
                                        camera_label: str = "Thermal"):
        """Add detections as separate layers for each frame."""
        # Create a group for the layers at top of layer tree
        group = self._create_layer_group(f"BAMBI Frame Detections ({camera_label})")
        animals = self._detection_animals(camera_label)
        style_settings = self._animal_style_settings()

        for frame_idx, detections in frame_detections.items():
            if not detections:
                continue

            # Create polygon layer for this frame
            layer_name = f"Detections_Frame_{frame_idx:06d}"
            layer = QgsVectorLayer("Polygon?crs=" + target_crs.authid(), layer_name, "memory")

            provider = layer.dataProvider()

            # Add fields
            provider.addAttributes([
                QgsField("det_id", FieldType.Int),
                QgsField("frame", FieldType.Int),
                QgsField("confidence", FieldType.Double),
                QgsField("class_id", FieldType.Int)
            ] + animal_fields())
            layer.updateFields()

            # Add features
            features = []
            for det in detections:
                feat = QgsFeature()

                # Create polygon from bounding box
                points = [
                    QgsPointXY(det['x1'], det['y1']),
                    QgsPointXY(det['x2'], det['y1']),
                    QgsPointXY(det['x2'], det['y2']),
                    QgsPointXY(det['x1'], det['y2']),
                    QgsPointXY(det['x1'], det['y1'])  # Close the polygon
                ]
                feat.setGeometry(QgsGeometry.fromPolygonXY([points]))
                feat.setAttributes([
                    det.get('detection_id', det['idx']),
                    det['frame'],
                    det['confidence'],
                    det['class_id']
                ] + animal_values(animals.get(det.get('detection_id'))))
                features.append(feat)

            provider.addFeatures(features)
            layer.updateExtents()

            # Persist layer to GeoPackage
            layer = self._persist_memory_layer(
                layer, f"{layer_name}_{_camera_suffix(camera_label)}", "detection_layers")
            self._style_animal_boxes(layer, style_settings)

            # Tag layer so the inspector tool can identify and handle it
            layer.setCustomProperty("bambi_layer_type", "detection")
            layer.setCustomProperty("bambi_target_folder", self.target_folder_edit.text().strip())
            layer.setCustomProperty(
                "bambi_detection_camera",
                "T" if self.detection_camera_combo.currentIndex() == 0 else "W")
            layer.setCustomProperty("bambi_dem_path", self.dem_path_edit.text().strip())
            layer.setCustomProperty("bambi_correction_path", self.correction_path_edit.text().strip())

            # Add layer to project and group
            QgsProject.instance().addMapLayer(layer, False)
            group.addLayer(layer)

        # Collapse the group
        group.setExpanded(False)

    def _add_detections_combined_layer(self, frame_detections: Dict[int, list], target_crs,
                                       camera_label: str = "Thermal"):
        """Add all detections in a single combined layer."""
        layer_name = f"BAMBI Detections - All Frames ({camera_label})"
        layer = QgsVectorLayer("Polygon?crs=" + target_crs.authid(), layer_name, "memory")
        animals = self._detection_animals(camera_label)

        provider = layer.dataProvider()

        # Add fields
        provider.addAttributes([
            QgsField("det_id", FieldType.Int),
            QgsField("frame", FieldType.Int),
            QgsField("confidence", FieldType.Double),
            QgsField("class_id", FieldType.Int)
        ] + animal_fields())
        layer.updateFields()

        features = []
        for frame_idx, detections in frame_detections.items():
            for det in detections:
                feat = QgsFeature()

                # Create polygon from bounding box
                points = [
                    QgsPointXY(det['x1'], det['y1']),
                    QgsPointXY(det['x2'], det['y1']),
                    QgsPointXY(det['x2'], det['y2']),
                    QgsPointXY(det['x1'], det['y2']),
                    QgsPointXY(det['x1'], det['y1'])
                ]
                feat.setGeometry(QgsGeometry.fromPolygonXY([points]))
                feat.setAttributes([
                    det.get('detection_id', det['idx']),
                    det['frame'],
                    det['confidence'],
                    det['class_id']
                ] + animal_values(animals.get(det.get('detection_id'))))
                features.append(feat)

        provider.addFeatures(features)
        layer.updateExtents()

        # Persist layer to GeoPackage
        layer = self._persist_memory_layer(
            layer, f"Detections_AllFrames_{_camera_suffix(camera_label)}", "detection_layers")
        self._style_animal_boxes(layer)

        # Tag layer so the inspector tool can identify and handle it
        layer.setCustomProperty("bambi_layer_type", "detection")
        layer.setCustomProperty("bambi_target_folder", self.target_folder_edit.text().strip())
        layer.setCustomProperty(
            "bambi_detection_camera",
            "T" if self.detection_camera_combo.currentIndex() == 0 else "W")
        layer.setCustomProperty("bambi_dem_path", self.dem_path_edit.text().strip())
        layer.setCustomProperty("bambi_correction_path", self.correction_path_edit.text().strip())

        # Add layer to project
        self._add_layer_to_flight(layer)

    def load_detections_by_frame(self, georef_file: str) -> Dict[int, list]:
        """Load geo-referenced detections grouped by frame (core.pipeline_outputs).

        :param georef_file: Path to georeferenced detections file
        :return: Dictionary mapping frame index to list of detections
        """
        from ..core.pipeline_outputs import load_georef_detections_by_frame
        return load_georef_detections_by_frame(georef_file, log_fn=self.log)

    def add_alfs_to_qgis(self):
        """Add all GeoTIFFs in the alfs folder to QGIS as raster layers."""
        config = self.get_config()
        alfs_camera = config.get("alfs_camera", "T")
        alfs_suffix = "t" if alfs_camera == "T" else "w"
        camera_label = "Thermal" if alfs_camera == "T" else "RGB"
        alfs_folder = os.path.join(config["target_folder"], f"alfs_{alfs_suffix}")

        if not os.path.exists(alfs_folder):
            QMessageBox.warning(
                self,
                "Missing ALFS",
                f"{camera_label} alfs has not been generated.\nPlease run Step 6 (Generate ALFS) first."
            )
            return

        try:
            self.log(f"Adding {camera_label} alfs to QGIS...")
            self.update_status("add_alfs", "🟡 Loading...")

            tif_files = sorted([
                os.path.join(alfs_folder, f)
                for f in os.listdir(alfs_folder)
                if f.lower().endswith((".tif", ".tiff"))
            ])

            if not tif_files:
                QMessageBox.warning(
                    self,
                    "Missing ALFS",
                    "ALFS has not been generated.\nPlease run Step 6 (Generate ALFS) first."
                )
                self.update_status("add_alfs", "🔴 No files")
                return

            if len(tif_files) == 1:
                layer_name = f"BAMBI ALFS ({camera_label})"
                layer = QgsRasterLayer(tif_files[0], layer_name)
                if not layer.isValid():
                    raise RuntimeError(f"Failed to load raster: {tif_files[0]}")
                self._add_layer_to_flight(layer)
                self.log(f"Added {camera_label} alfs layer to QGIS")
            else:
                group = self._create_layer_group(f"BAMBI ALFS ({camera_label})")
                loaded_count = 0
                for tif_path in tif_files:
                    layer_name = os.path.splitext(os.path.basename(tif_path))[0]
                    layer = QgsRasterLayer(tif_path, layer_name)
                    if layer.isValid():
                        QgsProject.instance().addMapLayer(layer, False)
                        group.addLayer(layer)
                        loaded_count += 1
                    else:
                        self.log(f"Warning: Could not load {os.path.basename(tif_path)}")
                self.log(f"Added {loaded_count} alfs file(s) to QGIS")

            self.update_status("add_alfs", "🟢 Added")
            self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding alfs: {str(e)}")
            self.update_status("add_alfs", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add alfs: {str(e)}")

    def add_geotiffs_to_qgis(self):
        """Add exported frame GeoTIFFs to QGIS as raster layers in a group."""
        config = self.get_config()
        gt_camera = config.get("geotiff_camera", "T")
        gt_suffix = "t" if gt_camera == "T" else "w"
        camera_label = "Thermal" if gt_camera == "T" else "RGB"
        geotiff_folder = os.path.join(config["target_folder"], f"geotiffs_{gt_suffix}")

        if not os.path.exists(geotiff_folder):
            QMessageBox.warning(
                self,
                "Missing GeoTIFFs",
                f"{camera_label} frame GeoTIFFs have not been exported.\n"
                f"Please run Step 7 (Export Frames as GeoTIFF) first."
            )
            return

        try:
            self.log(f"Adding {camera_label} frame GeoTIFFs to QGIS...")
            self.update_status("add_geotiffs", "🟡 Loading...")

            # Collect files and parse frame indices from filenames ({idx:08d}.tiff)
            geotiff_files = []
            for f in sorted(os.listdir(geotiff_folder)):
                if not f.lower().endswith(('.tif', '.tiff')):
                    continue
                stem = os.path.splitext(f)[0]
                try:
                    frame_idx = int(stem)
                except ValueError:
                    continue
                geotiff_files.append((frame_idx, os.path.join(geotiff_folder, f)))

            if not geotiff_files:
                QMessageBox.warning(self, "No GeoTIFFs", "No GeoTIFF files found.")
                self.update_status("add_geotiffs", "🔴 No files")
                return

            min_idx = geotiff_files[0][0]
            max_idx = geotiff_files[-1][0]
            dlg = FrameRangeDialog(
                self, min_idx, max_idx, len(geotiff_files), "frames"
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.update_status("add_geotiffs", "⚪ Cancelled")
                return

            start, end = dlg.selected_range()
            geotiff_files = [(idx, p) for idx, p in geotiff_files if start <= idx <= end]

            if not geotiff_files:
                QMessageBox.warning(self, "No GeoTIFFs", "No GeoTIFF files in selected range.")
                self.update_status("add_geotiffs", "🔴 No files")
                return

            if len(geotiff_files) > 100:
                reply = QMessageBox.question(
                    self,
                    "Many Files",
                    f"Selected range contains {len(geotiff_files)} GeoTIFF files.\n"
                    "Loading all may slow down QGIS.\n\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.update_status("add_geotiffs", "⚪ Cancelled")
                    return

            group = self._create_layer_group(f"BAMBI Frame GeoTIFFs ({camera_label})")
            loaded_count = 0

            for frame_idx, geotiff_path in geotiff_files:
                layer_name = f"Frame {frame_idx:08d}"
                layer = QgsRasterLayer(geotiff_path, layer_name)
                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer, False)
                    group.addLayer(layer)
                    loaded_count += 1
                else:
                    self.log(f"Warning: Could not load {os.path.basename(geotiff_path)}")

            self.log(f"Added {loaded_count} GeoTIFF layers to QGIS")
            self.update_status("add_geotiffs", "🟢 Added")
            group.setExpanded(False)
            self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding GeoTIFFs: {str(e)}")
            self.update_status("add_geotiffs", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add GeoTIFFs: {str(e)}")
