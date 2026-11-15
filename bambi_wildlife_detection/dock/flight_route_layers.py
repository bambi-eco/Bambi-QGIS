# -*- coding: utf-8 -*-
"""The flight route on the map, and its markers.

The line the drone flew, and along it - as the Flight Route settings say -
a marker per frame, per distance or per time, and the frame numbers.
"""

import json
import os
from typing import Optional

from qgis.PyQt.QtWidgets import QMessageBox
from qgis.PyQt.QtGui import QFont, QColor
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY,
    QgsLineSymbol, QgsMarkerSymbol, QgsPalLayerSettings, QgsTextFormat,
    QgsVectorLayerSimpleLabeling, QgsTextBufferSettings,
)


class FlightRouteLayersMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def add_flight_route_to_qgis(self):
        """Add flight route layers (polyline, points, and frame markers) to QGIS."""
        config = self.get_config()
        fr_camera = config.get("flight_route_camera", "T")
        fr_suffix = "t" if fr_camera == "T" else "w"
        camera_label = "Thermal" if fr_camera == "T" else "RGB"
        route_folder = os.path.join(config["target_folder"], f"flight_route_{fr_suffix}")

        if not os.path.exists(route_folder):
            QMessageBox.warning(
                self,
                "Missing Flight Route",
                f"{camera_label} flight route has not been generated.\n"
                f"Please run Step 2 (Generate Flight Route) first."
            )
            return

        try:
            self.log(f"Adding {camera_label} flight route to QGIS...")
            self.update_status("add_flight_route", "🟡 Loading...")

            # Create a group for the layers at top of layer tree
            group = self._create_layer_group(f"BAMBI Flight Route ({camera_label})")

            loaded_count = 0

            # Add flight route line
            route_line_file = os.path.join(route_folder, "flight_route.geojson")
            if os.path.exists(route_line_file):
                layer = QgsVectorLayer(route_line_file, "Flight Route", "ogr")
                if layer.isValid():
                    # Set a nice style for the line
                    symbol = QgsLineSymbol.createSimple({
                        'color': '#3388ff',
                        'width': '1.5',
                        'capstyle': 'round',
                        'joinstyle': 'round'
                    })
                    layer.renderer().setSymbol(symbol)

                    QgsProject.instance().addMapLayer(layer, False)
                    group.addLayer(layer)
                    loaded_count += 1
                    self.log("Added flight route line layer")
                else:
                    self.log("Warning: Could not load flight route line")

            # Add camera positions
            camera_points_file = os.path.join(route_folder, "camera_positions.geojson")
            if os.path.exists(camera_points_file):
                layer = QgsVectorLayer(camera_points_file, "Camera Positions", "ogr")
                if layer.isValid():
                    # Set a nice style for the points
                    symbol = QgsMarkerSymbol.createSimple({
                        'name': 'circle',
                        'color': '#ff7800',
                        'outline_color': '#ffffff',
                        'outline_width': '0.5',
                        'size': '3'
                    })
                    layer.renderer().setSymbol(symbol)

                    QgsProject.instance().addMapLayer(layer, False)
                    group.addLayer(layer)
                    loaded_count += 1
                    self.log("Added camera positions layer")
                else:
                    self.log("Warning: Could not load camera positions")

            # Add frame markers if enabled
            if self.frame_markers_enabled_check.isChecked():
                markers_layer = self._create_frame_markers_layer(
                    camera_points_file, config
                )
                if markers_layer and markers_layer.isValid():
                    # Persist the layer
                    markers_layer = self._persist_memory_layer(
                        markers_layer, f"Frame_Markers_{fr_suffix}", "flight_route_layers"
                    )
                    # Re-apply styling after persistence (GeoPackage doesn't store QGIS styling)
                    self._style_frame_markers_layer(markers_layer)
                    QgsProject.instance().addMapLayer(markers_layer, False)
                    group.addLayer(markers_layer)
                    loaded_count += 1
                    self.log("Added frame markers layer")

            # Add distance markers if enabled
            if self.distance_markers_enabled_check.isChecked():
                distance_layer = self._create_distance_markers_layer(
                    camera_points_file, config
                )
                if distance_layer and distance_layer.isValid():
                    # Persist the layer
                    distance_layer = self._persist_memory_layer(
                        distance_layer, f"Distance_Markers_{fr_suffix}", "flight_route_layers"
                    )
                    # Re-apply styling after persistence (GeoPackage doesn't store QGIS styling)
                    self._style_distance_markers_layer(distance_layer)
                    QgsProject.instance().addMapLayer(distance_layer, False)
                    group.addLayer(distance_layer)
                    loaded_count += 1
                    self.log("Added distance markers layer")

            # Add time markers if enabled
            if self.time_markers_enabled_check.isChecked():
                time_layer = self._create_time_markers_layer(
                    camera_points_file, config
                )
                if time_layer and time_layer.isValid():
                    time_layer = self._persist_memory_layer(
                        time_layer, f"Time_Markers_{fr_suffix}", "flight_route_layers"
                    )
                    self._style_time_markers_layer(time_layer)
                    QgsProject.instance().addMapLayer(time_layer, False)
                    group.addLayer(time_layer)
                    loaded_count += 1
                    self.log("Added time markers layer")

            # Add image labels if enabled
            if self.image_labels_enabled_check.isChecked():
                image_labels_layer = self._create_image_labels_layer(
                    camera_points_file, config
                )
                if image_labels_layer and image_labels_layer.isValid():
                    image_labels_layer = self._persist_memory_layer(
                        image_labels_layer, f"Image_Labels_{fr_suffix}", "flight_route_layers"
                    )
                    self._style_image_labels_layer(image_labels_layer)
                    QgsProject.instance().addMapLayer(image_labels_layer, False)
                    group.addLayer(image_labels_layer)
                    loaded_count += 1
                    self.log("Added image labels layer")

            if loaded_count == 0:
                QMessageBox.warning(self, "No Layers", "No flight route layers found.")
                self.update_status("add_flight_route", "🔴 No files")
                # Remove empty group
                QgsProject.instance().layerTreeRoot().removeChildNode(group)
                return

            self.log(f"Added {loaded_count} flight route layers to QGIS")
            self.update_status("add_flight_route", "🟢 Added")

            # Refresh canvas
            self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding flight route: {str(e)}")
            self.update_status("add_flight_route", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add flight route: {str(e)}")

    def _create_frame_markers_layer(self, camera_points_file: str, config: dict) -> Optional[QgsVectorLayer]:
        """Create a vector layer with frame markers at every N-th position.

        :param camera_points_file: Path to the camera positions GeoJSON file
        :param config: Configuration dictionary
        :return: QgsVectorLayer with frame markers, or None if creation fails
        """
        if not os.path.exists(camera_points_file):
            self.log("Warning: Camera positions file not found for frame markers")
            return None

        try:
            # Read camera positions
            with open(camera_points_file, 'r', encoding='utf-8') as f:
                camera_data = json.load(f)

            features = camera_data.get("features", [])
            if not features:
                self.log("Warning: No camera positions found for frame markers")
                return None

            # Get marker settings
            marker_interval = self.frame_marker_interval_spin.value()
            include_zero = self.frame_marker_include_zero_check.isChecked()

            # Get CRS from config
            target_epsg = config.get("target_epsg", 32633)

            # Create memory layer for markers
            layer_uri = f"Point?crs=EPSG:{target_epsg}&field=frame_idx:integer&field=label:string"
            markers_layer = QgsVectorLayer(layer_uri, "Frame Markers", "memory")

            if not markers_layer.isValid():
                self.log("Warning: Could not create frame markers layer")
                return None

            # Add features for every N-th frame
            marker_features = []
            for feature in features:
                frame_idx = feature.get("properties", {}).get("frame_idx", 0)

                # Check if this frame should have a marker
                should_add_marker = False
                if frame_idx == 0 and include_zero:
                    should_add_marker = True
                elif frame_idx > 0 and frame_idx % marker_interval == 0:
                    should_add_marker = True

                if should_add_marker:
                    coords = feature.get("geometry", {}).get("coordinates", [0, 0])
                    point = QgsPointXY(coords[0], coords[1])

                    qgs_feature = QgsFeature(markers_layer.fields())
                    qgs_feature.setGeometry(QgsGeometry.fromPointXY(point))
                    qgs_feature.setAttributes([frame_idx, str(frame_idx)])
                    marker_features.append(qgs_feature)

            if not marker_features:
                self.log("Warning: No frame markers to add (interval may be larger than total frames)")
                return None

            # Add features to layer
            markers_layer.dataProvider().addFeatures(marker_features)

            # Style the marker points
            symbol = QgsMarkerSymbol.createSimple({
                'name': 'diamond',
                'color': '#e31a1c',
                'outline_color': '#ffffff',
                'outline_width': '0.8',
                'size': '4'
            })
            markers_layer.renderer().setSymbol(symbol)

            # Configure labeling
            label_settings = QgsPalLayerSettings()
            label_settings.fieldName = 'label'
            label_settings.enabled = True

            # Text format
            text_format = QgsTextFormat()
            text_format.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            text_format.setSize(10)
            text_format.setColor(QColor('#000000'))

            # Add text buffer (halo) for better readability
            buffer_settings = QgsTextBufferSettings()
            buffer_settings.setEnabled(True)
            buffer_settings.setSize(1.5)
            buffer_settings.setColor(QColor('#ffffff'))
            text_format.setBuffer(buffer_settings)

            label_settings.setFormat(text_format)

            # Position labels above the marker
            label_settings.placement = QgsPalLayerSettings.Placement.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantPosition.QuadrantAbove

            # Apply labeling to layer
            labeling = QgsVectorLayerSimpleLabeling(label_settings)
            markers_layer.setLabelsEnabled(True)
            markers_layer.setLabeling(labeling)

            self.log(f"Created {len(marker_features)} frame markers (interval: {marker_interval})")
            return markers_layer

        except Exception as e:
            self.log(f"Error creating frame markers: {str(e)}")
            return None

    def _create_distance_markers_layer(self, camera_points_file: str, config: dict) -> Optional[QgsVectorLayer]:
        """Create a vector layer with markers at regular distance intervals along the flight path.

        Markers show both the cumulative distance and the associated frame number,
        e.g., "100m (243)" meaning 100 meters traveled, reached at frame 243.

        :param camera_points_file: Path to the camera positions GeoJSON file
        :param config: Configuration dictionary
        :return: QgsVectorLayer with distance markers, or None if creation fails
        """
        import math

        if not os.path.exists(camera_points_file):
            self.log("Warning: Camera positions file not found for distance markers")
            return None

        try:
            # Read camera positions
            with open(camera_points_file, 'r', encoding='utf-8') as f:
                camera_data = json.load(f)

            features = camera_data.get("features", [])
            if len(features) < 2:
                self.log("Warning: Need at least 2 camera positions for distance markers")
                return None

            # Get marker settings
            distance_interval = self.distance_marker_interval_spin.value()
            include_start = self.distance_marker_include_start_check.isChecked()

            # Get CRS from config
            target_epsg = config.get("target_epsg", 32633)

            # Calculate cumulative distances and find marker positions
            # Each entry: (cumulative_distance, frame_idx, x, y)
            positions = []
            cumulative_distance = 0.0

            for i, feature in enumerate(features):
                coords = feature.get("geometry", {}).get("coordinates", [0, 0])
                frame_idx = feature.get("properties", {}).get("frame_idx", i)
                x, y = coords[0], coords[1]

                if i == 0:
                    positions.append((0.0, frame_idx, x, y))
                else:
                    prev_coords = features[i - 1].get("geometry", {}).get("coordinates", [0, 0])
                    dx = x - prev_coords[0]
                    dy = y - prev_coords[1]
                    segment_distance = math.sqrt(dx * dx + dy * dy)
                    cumulative_distance += segment_distance
                    positions.append((cumulative_distance, frame_idx, x, y))

            total_distance = positions[-1][0]
            self.log(f"Total flight distance: {total_distance:.1f}m")

            # Create memory layer for distance markers
            layer_uri = (
                f"Point?crs=EPSG:{target_epsg}"
                "&field=distance:double&field=frame_idx:integer&field=label:string"
            )
            markers_layer = QgsVectorLayer(layer_uri, "Distance Markers", "memory")

            if not markers_layer.isValid():
                self.log("Warning: Could not create distance markers layer")
                return None

            # Find positions for each distance interval
            marker_features = []

            # Optionally include start marker
            if include_start:
                start_pos = positions[0]
                point = QgsPointXY(start_pos[2], start_pos[3])
                qgs_feature = QgsFeature(markers_layer.fields())
                qgs_feature.setGeometry(QgsGeometry.fromPointXY(point))
                label = f"0m ({start_pos[1]})"
                qgs_feature.setAttributes([0.0, start_pos[1], label])
                marker_features.append(qgs_feature)

            # Find positions at each distance interval
            target_distance = distance_interval

            while target_distance <= total_distance:
                # Find the segment containing this distance
                for i in range(1, len(positions)):
                    prev_dist, prev_frame, prev_x, prev_y = positions[i - 1]
                    curr_dist, curr_frame, curr_x, curr_y = positions[i]

                    if prev_dist <= target_distance <= curr_dist:
                        # Interpolate position within this segment
                        segment_length = curr_dist - prev_dist
                        if segment_length > 0:
                            # How far along the segment is our target distance
                            ratio = (target_distance - prev_dist) / segment_length
                            interp_x = prev_x + ratio * (curr_x - prev_x)
                            interp_y = prev_y + ratio * (curr_y - prev_y)
                        else:
                            interp_x, interp_y = curr_x, curr_y

                        # Use the frame index of the end of this segment
                        # (the first frame that exceeds the target distance)
                        point = QgsPointXY(interp_x, interp_y)
                        qgs_feature = QgsFeature(markers_layer.fields())
                        qgs_feature.setGeometry(QgsGeometry.fromPointXY(point))
                        label = f"{int(target_distance)}m ({curr_frame})"
                        qgs_feature.setAttributes([target_distance, curr_frame, label])
                        marker_features.append(qgs_feature)
                        break

                target_distance += distance_interval

            if not marker_features:
                self.log(
                    "Warning: No distance markers created "
                    f"(interval {distance_interval}m > total {total_distance:.1f}m)"
                )
                return None

            # Add features to layer
            markers_layer.dataProvider().addFeatures(marker_features)

            # Style the marker points (different color from frame markers)
            symbol = QgsMarkerSymbol.createSimple({
                'name': 'triangle',
                'color': '#2ca02c',  # Green color
                'outline_color': '#ffffff',
                'outline_width': '0.8',
                'size': '4.5'
            })
            markers_layer.renderer().setSymbol(symbol)

            # Configure labeling
            label_settings = QgsPalLayerSettings()
            label_settings.fieldName = 'label'
            label_settings.enabled = True

            # Text format
            text_format = QgsTextFormat()
            text_format.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            text_format.setSize(9)
            text_format.setColor(QColor('#1a5e1a'))  # Dark green

            # Add text buffer (halo) for better readability
            buffer_settings = QgsTextBufferSettings()
            buffer_settings.setEnabled(True)
            buffer_settings.setSize(1.5)
            buffer_settings.setColor(QColor('#ffffff'))
            text_format.setBuffer(buffer_settings)

            label_settings.setFormat(text_format)

            # Position labels above the marker
            label_settings.placement = QgsPalLayerSettings.Placement.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantPosition.QuadrantAbove

            # Apply labeling to layer
            labeling = QgsVectorLayerSimpleLabeling(label_settings)
            markers_layer.setLabelsEnabled(True)
            markers_layer.setLabeling(labeling)

            self.log(f"Created {len(marker_features)} distance markers (interval: {distance_interval}m)")
            return markers_layer

        except Exception as e:
            self.log(f"Error creating distance markers: {str(e)}")
            return None

    def _create_time_markers_layer(self, camera_points_file: str, config: dict) -> Optional[QgsVectorLayer]:
        """Create a vector layer with markers at regular time intervals along the flight path.

        Timestamps are read directly from the AirData CSV (first column = relative ms,
        datetime(utc) column = absolute UTC time).  Positions are interpolated along the
        AirData GPS track projected to the target CRS.

        Marker labels:
          Relative mode  →  "30s", "60s", …
          Absolute mode  →  "10:30:45", "10:31:15", …

        :param camera_points_file: Unused; kept for API consistency with the other marker methods.
        :param config: Configuration dictionary
        :return: QgsVectorLayer with time markers, or None if creation fails
        """
        import csv
        from datetime import datetime, timezone

        airdata_path = config.get("airdata_path", "")
        if not airdata_path or not os.path.exists(airdata_path):
            self.log("Warning: No AirData file configured – time markers require an AirData log")
            return None

        time_interval = self.time_marker_interval_spin.value()
        include_start = self.time_marker_include_start_check.isChecked()
        use_relative = self.time_marker_type_combo.currentIndex() == 0
        target_epsg = config.get("target_epsg", 32633)

        try:
            from pyproj import Transformer, CRS as PyprojCRS

            wgs84_proj4 = "+proj=longlat +datum=WGS84 +no_defs"
            if 32601 <= target_epsg <= 32660:
                zone = target_epsg - 32600
                utm_proj4 = f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"
            elif 32701 <= target_epsg <= 32760:
                zone = target_epsg - 32700
                utm_proj4 = f"+proj=utm +zone={zone} +south +datum=WGS84 +units=m +no_defs"
            else:
                from bambi_wildlife_detection.austria_dem_downloader import get_proj4_for_crs, WGS84_PROJ4
                wgs84_proj4 = WGS84_PROJ4
                utm_proj4 = get_proj4_for_crs(f"EPSG:{target_epsg}")

            transformer = Transformer.from_crs(
                PyprojCRS.from_proj4(wgs84_proj4),
                PyprojCRS.from_proj4(utm_proj4),
                always_xy=True
            )

            # ── Read AirData CSV ──────────────────────────────────────────────
            def _find_col(headers, *keywords):
                """Return the first header whose lower-case form contains all keywords."""
                for h in headers:
                    hl = h.lower().strip()
                    if all(k in hl for k in keywords):
                        return h
                return None

            def _parse_utc(s):
                if not s:
                    return None
                for fmt in (
                    "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M:%S.%f",
                ):
                    try:
                        return datetime.strptime(s.strip(), fmt).replace(
                            tzinfo=timezone.utc).timestamp()
                    except ValueError:
                        pass
                return None

            raw_rows = []   # (time_ms, utc_unix, lat, lon)

            with open(airdata_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []

                col_ms = _find_col(headers, "time", "millisecond")
                col_utc = _find_col(headers, "datetime", "utc") or _find_col(headers, "datetime")
                col_lat = _find_col(headers, "latitude") or _find_col(headers, "lat")
                col_lon = _find_col(headers, "longitude") or _find_col(headers, "lon")

                if col_lat is None or col_lon is None:
                    self.log("Warning: Could not find lat/lon columns in AirData CSV")
                    return None
                if col_ms is None and col_utc is None:
                    self.log("Warning: Could not find timestamp columns in AirData CSV")
                    return None

                for row in reader:
                    try:
                        lat = float(row[col_lat])
                        lon = float(row[col_lon])
                        if lat == 0.0 and lon == 0.0:
                            continue
                        ts_ms = float(row[col_ms]) if col_ms else None
                        ts_utc = _parse_utc(row[col_utc]) if col_utc else None
                        raw_rows.append((ts_ms, ts_utc, lat, lon))
                    except (ValueError, KeyError):
                        continue

            if len(raw_rows) < 2:
                self.log("Warning: Not enough valid GPS rows in AirData for time markers")
                return None

            # ── Project to target CRS ─────────────────────────────────────────
            # Build (elapsed_sec, unix_ts_or_None, x, y)
            first_ms = raw_rows[0][0]
            first_utc = raw_rows[0][1]

            positions = []
            for ts_ms, ts_utc, lat, lon in raw_rows:
                x, y = transformer.transform(lon, lat)
                if use_relative:
                    if ts_ms is None or first_ms is None:
                        continue
                    elapsed = (ts_ms - first_ms) / 1000.0
                    positions.append((elapsed, None, x, y))
                else:
                    if ts_utc is None or first_utc is None:
                        continue
                    elapsed = ts_utc - first_utc
                    positions.append((elapsed, ts_utc, x, y))

            if len(positions) < 2:
                self.log("Warning: Not enough timestamped rows in AirData for time markers")
                return None

            total_elapsed = positions[-1][0]
            if total_elapsed <= 0:
                self.log("Warning: AirData timestamps do not increase – cannot build time markers")
                return None

            self.log(f"AirData time range: {total_elapsed:.1f}s ({len(positions)} GPS points)")

            # ── Label helper ─────────────────────────────────────────────────
            def make_label(elapsed_sec, unix_ts):
                if use_relative:
                    return f"{elapsed_sec:.0f}s"
                if unix_ts is not None:
                    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%H:%M:%S")
                return f"{elapsed_sec:.0f}s"

            # ── Create memory layer ───────────────────────────────────────────
            layer_uri = (
                f"Point?crs=EPSG:{target_epsg}"
                "&field=elapsed_sec:double&field=label:string"
            )
            markers_layer = QgsVectorLayer(layer_uri, "Time Markers", "memory")
            if not markers_layer.isValid():
                self.log("Warning: Could not create time markers memory layer")
                return None

            marker_features = []

            if include_start:
                p = positions[0]
                feat = QgsFeature(markers_layer.fields())
                feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(p[2], p[3])))
                feat.setAttributes([0.0, make_label(0.0, p[1])])
                marker_features.append(feat)

            target_sec = float(time_interval)
            while target_sec <= total_elapsed:
                for i in range(1, len(positions)):
                    e_prev, ts_prev, xp, yp = positions[i - 1]
                    e_curr, ts_curr, xc, yc = positions[i]
                    if e_prev <= target_sec <= e_curr:
                        seg = e_curr - e_prev
                        ratio = (target_sec - e_prev) / seg if seg > 0 else 0.0
                        ix = xp + ratio * (xc - xp)
                        iy = yp + ratio * (yc - yp)
                        interp_ts = (
                            None if (ts_prev is None or ts_curr is None)
                            else ts_prev + ratio * (ts_curr - ts_prev)
                        )
                        feat = QgsFeature(markers_layer.fields())
                        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(ix, iy)))
                        feat.setAttributes([target_sec, make_label(target_sec, interp_ts)])
                        marker_features.append(feat)
                        break
                target_sec += time_interval

            if not marker_features:
                self.log(
                    "Warning: No time markers created "
                    f"(interval {time_interval}s > total {total_elapsed:.1f}s)")
                return None

            markers_layer.dataProvider().addFeatures(marker_features)
            self.log(
                f"Created {len(marker_features)} time markers "
                f"(interval: {time_interval}s, "
                f"{'relative' if use_relative else 'absolute'})"
            )
            return markers_layer

        except Exception as e:
            self.log(f"Error creating time markers: {str(e)}")
            return None

    def _style_time_markers_layer(self, layer: QgsVectorLayer):
        """Apply styling and labeling to the time markers layer."""
        try:
            symbol = QgsMarkerSymbol.createSimple({
                'name': 'diamond',
                'color': '#9b59b6',        # Purple
                'outline_color': '#ffffff',
                'outline_width': '0.8',
                'size': '4.5'
            })
            layer.renderer().setSymbol(symbol)

            label_settings = QgsPalLayerSettings()
            label_settings.fieldName = 'label'
            label_settings.enabled = True

            text_format = QgsTextFormat()
            text_format.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            text_format.setSize(9)
            text_format.setColor(QColor('#6c3483'))  # Dark purple

            buffer_settings = QgsTextBufferSettings()
            buffer_settings.setEnabled(True)
            buffer_settings.setSize(1.5)
            buffer_settings.setColor(QColor('#ffffff'))
            text_format.setBuffer(buffer_settings)

            label_settings.setFormat(text_format)
            label_settings.placement = QgsPalLayerSettings.Placement.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantPosition.QuadrantAbove

            layer.setLabelsEnabled(True)
            layer.setLabeling(QgsVectorLayerSimpleLabeling(label_settings))
            layer.triggerRepaint()
        except Exception as e:
            self.log(f"Warning: Could not style time markers layer: {str(e)}")

    def _create_image_labels_layer(self, camera_points_file: str, config: dict) -> Optional[QgsVectorLayer]:
        """Create a vector layer with image filename labels at every N-th camera position.

        :param camera_points_file: Path to the camera positions GeoJSON file
        :param config: Configuration dictionary
        :return: QgsVectorLayer with image labels, or None if creation fails
        """
        if not os.path.exists(camera_points_file):
            self.log("Warning: Camera positions file not found for image labels")
            return None

        try:
            with open(camera_points_file, 'r', encoding='utf-8') as f:
                camera_data = json.load(f)

            features = camera_data.get("features", [])
            if not features:
                self.log("Warning: No camera positions found for image labels")
                return None

            label_interval = self.image_label_interval_spin.value()
            target_epsg = config.get("target_epsg", 32633)

            layer_uri = (
                f"Point?crs=EPSG:{target_epsg}"
                "&field=frame_idx:integer&field=imagefile:string&field=label:string"
            )
            labels_layer = QgsVectorLayer(layer_uri, "Image Labels", "memory")

            if not labels_layer.isValid():
                self.log("Warning: Could not create image labels layer")
                return None

            label_features = []
            for feature in features:
                props = feature.get("properties", {})
                frame_idx = props.get("frame_idx", 0)

                if frame_idx % label_interval != 0:
                    continue

                coords = feature.get("geometry", {}).get("coordinates", [0, 0])
                imagefile = props.get("imagefile", "")
                label = os.path.basename(imagefile) if imagefile else str(frame_idx)

                point = QgsPointXY(coords[0], coords[1])
                qgs_feature = QgsFeature(labels_layer.fields())
                qgs_feature.setGeometry(QgsGeometry.fromPointXY(point))
                qgs_feature.setAttributes([frame_idx, imagefile, label])
                label_features.append(qgs_feature)

            if not label_features:
                self.log("Warning: No image labels to add (interval may be larger than total frames)")
                return None

            labels_layer.dataProvider().addFeatures(label_features)
            self.log(f"Created {len(label_features)} image labels (interval: {label_interval})")
            return labels_layer

        except Exception as e:
            self.log(f"Error creating image labels: {str(e)}")
            return None

    def _style_image_labels_layer(self, layer: QgsVectorLayer):
        """Apply styling and labeling to the image labels layer."""
        try:
            # Invisible marker - label text only
            symbol = QgsMarkerSymbol.createSimple({
                'name': 'circle',
                'color': '0,0,0,0',
                'outline_style': 'no',
                'size': '1'
            })
            layer.renderer().setSymbol(symbol)

            label_settings = QgsPalLayerSettings()
            label_settings.fieldName = 'label'
            label_settings.enabled = True

            text_format = QgsTextFormat()
            text_format.setFont(QFont("Arial", 8))
            text_format.setSize(8)
            text_format.setColor(QColor('#1a1a1a'))

            buffer_settings = QgsTextBufferSettings()
            buffer_settings.setEnabled(True)
            buffer_settings.setSize(1.0)
            buffer_settings.setColor(QColor('#ffffff'))
            text_format.setBuffer(buffer_settings)

            label_settings.setFormat(text_format)
            label_settings.placement = QgsPalLayerSettings.Placement.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantPosition.QuadrantBelow

            layer.setLabelsEnabled(True)
            layer.setLabeling(QgsVectorLayerSimpleLabeling(label_settings))
            layer.triggerRepaint()
        except Exception as e:
            self.log(f"Warning: Could not style image labels layer: {str(e)}")

    def _style_frame_markers_layer(self, layer: QgsVectorLayer):
        """Apply styling and labeling to frame markers layer.

        :param layer: The frame markers layer to style
        """
        try:
            # Style the marker points
            symbol = QgsMarkerSymbol.createSimple({
                'name': 'diamond',
                'color': '#e31a1c',
                'outline_color': '#ffffff',
                'outline_width': '0.8',
                'size': '4'
            })
            layer.renderer().setSymbol(symbol)

            # Configure labeling
            label_settings = QgsPalLayerSettings()
            label_settings.fieldName = 'label'
            label_settings.enabled = True

            # Text format
            text_format = QgsTextFormat()
            text_format.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            text_format.setSize(10)
            text_format.setColor(QColor('#000000'))

            # Add text buffer (halo) for better readability
            buffer_settings = QgsTextBufferSettings()
            buffer_settings.setEnabled(True)
            buffer_settings.setSize(1.5)
            buffer_settings.setColor(QColor('#ffffff'))
            text_format.setBuffer(buffer_settings)

            label_settings.setFormat(text_format)

            # Position labels above the marker
            label_settings.placement = QgsPalLayerSettings.Placement.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantPosition.QuadrantAbove

            # Apply labeling to layer
            labeling = QgsVectorLayerSimpleLabeling(label_settings)
            layer.setLabelsEnabled(True)
            layer.setLabeling(labeling)
        except Exception as e:
            self.log(f"Warning: Could not style frame markers layer: {str(e)}")

    def _style_distance_markers_layer(self, layer: QgsVectorLayer):
        """Apply styling and labeling to distance markers layer.

        :param layer: The distance markers layer to style
        """
        try:
            # Style the marker points (different color from frame markers)
            symbol = QgsMarkerSymbol.createSimple({
                'name': 'triangle',
                'color': '#2ca02c',  # Green color
                'outline_color': '#ffffff',
                'outline_width': '0.8',
                'size': '4.5'
            })
            layer.renderer().setSymbol(symbol)

            # Configure labeling
            label_settings = QgsPalLayerSettings()
            label_settings.fieldName = 'label'
            label_settings.enabled = True

            # Text format
            text_format = QgsTextFormat()
            text_format.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            text_format.setSize(9)
            text_format.setColor(QColor('#1a5e1a'))  # Dark green

            # Add text buffer (halo) for better readability
            buffer_settings = QgsTextBufferSettings()
            buffer_settings.setEnabled(True)
            buffer_settings.setSize(1.5)
            buffer_settings.setColor(QColor('#ffffff'))
            text_format.setBuffer(buffer_settings)

            label_settings.setFormat(text_format)

            # Position labels above the marker
            label_settings.placement = QgsPalLayerSettings.Placement.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantPosition.QuadrantAbove

            # Apply labeling to layer
            labeling = QgsVectorLayerSimpleLabeling(label_settings)
            layer.setLabelsEnabled(True)
            layer.setLabeling(labeling)
        except Exception as e:
            self.log(f"Warning: Could not style distance markers layer: {str(e)}")
