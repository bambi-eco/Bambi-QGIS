# -*- coding: utf-8 -*-
"""
BAMBI Click Tool
================

QgsMapToolIdentify subclass that intercepts left-clicks on BAMBI detection /
track layers and opens the FeatureViewerDialog with the corresponding frame
image and annotated bounding boxes.

Layer custom properties used
----------------------------
``bambi_layer_type``       : "detection" | "track_final" | "track_path"
``bambi_target_folder``    : absolute path to the plugin output root folder
``bambi_detection_camera`` : "T" (thermal) | "W" (RGB) — modality used for detection
``bambi_dem_path``         : absolute path to the DEM GLTF/GLB file
``bambi_correction_path``  : absolute path to correction.json (may be empty)

Data read (relative to *target_folder*, ``{m}`` = ``t`` thermal / ``w`` RGB,
selected from the layer's ``bambi_detection_camera`` property)
---------------------------------------------
``bambi_{m}/detections.gpkg``  — pixel-space detections and their species
``bambi_{m}/tracks.gpkg``      — which detections belong to which track

Both through :mod:`core.inspection`. The clicked box is found by the
``detection_id`` the layer was built from; the 5.x text files carry no such id,
so they were matched back on confidence and class to four decimal places and
are no longer read at all.

``poses_t.json`` / ``poses_w.json`` — frame → image file mapping
    structure: ``{"images": [{"imagefile": "...", ...}, ...]}``
    images stored under ``frames_t/`` or ``frames_w/`` respectively
"""

import os
from typing import Dict, List, Optional, Tuple

from qgis.gui import QgsMapToolIdentify
from qgis.core import (
    QgsVectorLayer, QgsProject, QgsMessageLog, Qgis,
    QgsCoordinateTransform, QgsPointXY,
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QCursor
from qgis.PyQt.QtWidgets import QMessageBox

from .bambi_feature_viewer import FeatureViewerDialog
from .core import inspection

TRACK_TYPES = {"track_final", "track_path"}
DETECTION_TYPE = "detection"
FOV_TYPE = "fov"


class BambiClickTool(QgsMapToolIdentify):
    """Map tool that opens FeatureViewerDialog on BAMBI layer feature clicks.

    Parameters
    ----------
    mode : str
        ``"detection_track"`` — identifies detection and track layers (default).
        ``"fov"``             — identifies FoV polygon layers (simple viewer, no geo-referencing).
        ``"fov_georef"``      — identifies FoV polygon layers and projects the click position
                                into each frame's image space using DEM elevation data.
    """

    def __init__(self, iface, mode: str = "detection_track"):
        super().__init__(iface.mapCanvas())
        self.iface = iface
        self.mode = mode
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        # DEM mesh cache: keyed by absolute mesh path so repeated FoV clicks
        # on the same DEM don't reload and re-build the BVH every time.
        self._dem_mesh_cache: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # QgsMapTool overrides
    # ------------------------------------------------------------------

    def canvasReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return

        bambi_layers = self._get_bambi_layers()
        if not bambi_layers:
            if self.mode in ("fov", "fov_georef"):
                QMessageBox.warning(
                    None,
                    "No Field of View Layers",
                    "No Field of View layers were found in the layer hierarchy.\n\n"
                    "Please add individual Field of View layers to QGIS before "
                    "using this tool.",
                )
            else:
                QMessageBox.warning(
                    None,
                    "No Detection / Track Layers",
                    "No active detection or track layers were found in the layer hierarchy.\n\n"
                    "Please add/activate individual detection or track layers to QGIS before "
                    "using this tool.",
                )
            return

        # Collect ALL results so we can apply mode-specific priority.
        results = self.identify(
            event.x(),
            event.y(),
            bambi_layers,
            QgsMapToolIdentify.IdentifyMode.TopDownAll,
        )
        if not results:
            return

        if self.mode in ("fov", "fov_georef"):
            # FoV mode: collect ALL FoV features at the clicked position so the
            # viewer can cycle through them with prev/next navigation.
            fov_results = [
                r for r in results
                if r.mLayer.customProperty("bambi_layer_type", "") == FOV_TYPE
            ]
            if fov_results:
                if self.mode == "fov_georef":
                    if not self._dem_mesh_cache:
                        reply = QMessageBox.question(
                            None,
                            "Load Digital Elevation Model",
                            "The geo-referenced FoV inspector needs to load the "
                            "digital elevation model.\n\n"
                            "This may take some time on the first click. Continue?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.Yes,
                        )
                        if reply != QMessageBox.StandardButton.Yes:
                            return
                    # Convert canvas pixel → map coordinate for click projection.
                    map_pt = self.canvas().getCoordinateTransform().toMapCoordinates(
                        event.x(), event.y()
                    )
                    click_xy = (map_pt.x(), map_pt.y())
                else:
                    click_xy = None
                self._handle_fov_click(fov_results, click_xy=click_xy)
            return

        # Detection/track mode: honour the layer hierarchy — whichever BAMBI
        # layer sits higher in the layer tree wins.  Build an ordered list of
        # layer IDs from the tree (top → bottom) and pick the result whose
        # layer has the smallest index.
        layer_order = self._get_layer_tree_order()

        def _tree_rank(result):
            try:
                return layer_order.index(result.mLayer.id())
            except ValueError:
                return len(layer_order)  # not found → lowest priority

        chosen = min(results, key=_tree_rank, default=None)

        if chosen is None:
            return

        layer = chosen.mLayer
        feature = chosen.mFeature
        layer_type = layer.customProperty("bambi_layer_type", "")
        target_folder = layer.customProperty("bambi_target_folder", "")
        dem_path = layer.customProperty("bambi_dem_path", "")
        correction_path = layer.customProperty("bambi_correction_path", "")
        # "T" = thermal, "W" = RGB; determines which pixel space the boxes are in
        detection_camera = layer.customProperty("bambi_detection_camera", "T")
        boxes_modality = "t" if detection_camera == "T" else "w"

        if not target_folder:
            return

        if layer_type == DETECTION_TYPE:
            self._handle_detection_click(
                feature, target_folder, boxes_modality, dem_path, correction_path)
        elif layer_type == "track_final":
            self._handle_track_click(
                feature, target_folder, boxes_modality, dem_path, correction_path,
                start_at_last=True)
        elif layer_type == "track_path":
            self._handle_track_click(
                feature, target_folder, boxes_modality, dem_path, correction_path,
                start_at_last=False)

    def deactivate(self):
        super().deactivate()
        self.iface.mapCanvas().setCursor(Qt.CursorShape.ArrowCursor)

    # ------------------------------------------------------------------
    # Click handlers
    # ------------------------------------------------------------------

    def _handle_detection_click(self, feature, target_folder: str,
                                boxes_modality: str, dem_path: str,
                                correction_path: str):
        """Show the frame for a clicked detection bounding box."""
        frame_idx = feature["frame"]
        det_conf = float(feature["confidence"])
        det_class = int(feature["class_id"])

        all_dets = self._load_pixel_detections(target_folder, boxes_modality)

        same_frame = [d for d in all_dets if d["frame"] == frame_idx]

        # Identify the clicked detection by matching confidence + class.
        # Confidence is stored with 4 decimal places in both files.
        clicked = [
            d for d in same_frame
            if d["class_id"] == det_class and abs(d["confidence"] - det_conf) < 0.0015
        ]
        not_clicked = [d for d in same_frame if d not in clicked]

        green_boxes = [
            (d["x1"], d["y1"], d["x2"], d["y2"], d["confidence"], d["class_id"])
            for d in clicked
        ]
        blue_boxes = [
            (d["x1"], d["y1"], d["x2"], d["y2"], d["confidence"], d["class_id"])
            for d in not_clicked
        ]

        image_path_t, image_path_w = self._resolve_image_paths(target_folder, frame_idx)
        title = (
            f"Detection — Frame {frame_idx}"
            f"   |   conf: {det_conf:.3f}"
            f"   |   cls: {det_class}"
        )

        viewer = FeatureViewerDialog.get_instance(self.iface.mainWindow())
        viewer.show_detection(
            title, green_boxes, blue_boxes,
            image_path_t=image_path_t, image_path_w=image_path_w,
            boxes_modality=boxes_modality,
            target_folder=target_folder, dem_path=dem_path,
            correction_path=correction_path,
            frame_idx=frame_idx,
        )

    def _handle_fov_click(self, fov_results, click_xy: Optional[Tuple[float, float]] = None):
        """Build a navigable frame list from all FoV features at the clicked position.

        Each overlapping FoV becomes one entry in the frame list so the user can
        cycle through all of them using the viewer's prev/next buttons.

        *click_xy* — (map_x, map_y) geographic coordinate of the click.  When
        provided it is projected into each FoV's image space (thermal and RGB)
        and stored as ``click_point_t`` / ``click_point_w`` in the frame dict
        so the viewer can draw a crosshair at the clicked location.
        """
        frames = []
        for result in fov_results:
            layer = result.mLayer
            feature = result.mFeature

            try:
                frame_idx = int(feature["frame"])
            except (TypeError, ValueError):
                continue

            target_folder = layer.customProperty("bambi_target_folder", "")
            dem_path = layer.customProperty("bambi_dem_path", "")
            correction_path = layer.customProperty("bambi_correction_path", "")
            detection_camera = layer.customProperty("bambi_detection_camera", "T")
            boxes_modality = "t" if detection_camera == "T" else "w"

            if not target_folder:
                continue

            # The DEM / correction paths may be missing on FoV layers that
            # were created before those fields were filled in the UI.  Fall
            # back to any other BAMBI layer that shares the same target
            # folder and has the property set.
            if not dem_path or not correction_path:
                dem_path, correction_path = self._resolve_missing_paths(
                    target_folder, dem_path, correction_path
                )

            all_dets = self._load_pixel_detections(
                target_folder, boxes_modality)
            same_frame = [d for d in all_dets if d["frame"] == frame_idx]

            # All detections on this frame are shown in green — there is no
            # "clicked" detection to single out.
            green_boxes = [
                (d["x1"], d["y1"], d["x2"], d["y2"], d["confidence"], d["class_id"])
                for d in same_frame
            ]
            image_path_t, image_path_w = self._resolve_image_paths(target_folder, frame_idx)

            frame_dict = {
                "frame_idx": frame_idx,
                "image_path_t": image_path_t,
                "image_path_w": image_path_w,
                "boxes_modality": boxes_modality,
                "boxes_green": green_boxes,
                "boxes_blue": [],
                # Per-frame projection context (may differ across layers).
                "target_folder": target_folder,
                "dem_path": dem_path,
                "correction_path": correction_path,
            }

            # Project the clicked map position into this frame's image space.
            # Failures are non-fatal — the frame is still shown, just without
            # the crosshair.
            if click_xy is not None:
                try:
                    # Transform click from canvas CRS → layer CRS if needed.
                    layer_xy = click_xy
                    canvas_crs = self.canvas().mapSettings().destinationCrs()
                    layer_crs = layer.crs()
                    if canvas_crs.isValid() and layer_crs.isValid() \
                            and canvas_crs != layer_crs:
                        xform = QgsCoordinateTransform(
                            canvas_crs, layer_crs, QgsProject.instance())
                        pt = xform.transform(QgsPointXY(*click_xy))
                        layer_xy = (pt.x(), pt.y())

                    frame_dict["click_point_t"] = self._project_map_point(
                        layer_xy, frame_idx, image_path_t,
                        target_folder, dem_path, correction_path, "t",
                    )
                    frame_dict["click_point_w"] = self._project_map_point(
                        layer_xy, frame_idx, image_path_w,
                        target_folder, dem_path, correction_path, "w",
                    )
                except Exception:  # nosec B110
                    pass

            frames.append(frame_dict)

        if not frames:
            return

        frames.sort(key=lambda f: f["frame_idx"])

        n_det = sum(len(f["boxes_green"]) for f in frames)
        title = (
            f"FoV — {len(frames)} field(s) of view"
            f"   |   {n_det} detection(s)"
        )
        viewer = FeatureViewerDialog.get_instance(self.iface.mainWindow())
        viewer.show_track(
            title, frames, start_idx=0,
            target_folder=frames[0]["target_folder"],
            dem_path=frames[0]["dem_path"],
            correction_path=frames[0]["correction_path"],
        )

    def _handle_track_click(self, feature, target_folder: str, boxes_modality: str,
                            dem_path: str, correction_path: str, start_at_last: bool):
        """Show the navigable frame sequence for a clicked track."""
        # GeoPackage may store integers as LongLong; cast explicitly.
        try:
            track_id = int(feature["track_id"])
        except (TypeError, ValueError):
            return

        # Reference frame differs between "Final Position" and "Path" layers.
        field_names = [f.name() for f in feature.fields()]
        ref_frame = None
        if "frame" in field_names:
            try:
                ref_frame = int(feature["frame"])
            except (TypeError, ValueError):
                pass
        elif "start_frame" in field_names:
            try:
                ref_frame = int(feature["start_frame"])
            except (TypeError, ValueError):
                pass

        # The store holds each track's boxes together with the detection they
        # came from, so there is one path rather than a primary and a fallback
        # that matched geo rows back to pixel rows by confidence and class.
        all_tracks = self._load_pixel_tracks(target_folder, boxes_modality)
        track_dets_pixel = sorted(
            all_tracks.get(track_id, []), key=lambda d: d["frame"])

        if not track_dets_pixel:
            QMessageBox.warning(
                None,
                "BAMBI Inspector",
                f"No frame data found for track {track_id}.\n\n"
                "The store holds no boxes for it. Re-run tracking, or use "
                "'Migrate 5.x…' on the Input tab for an older project.",
            )
            return

        frames = self._build_frames_from_pixel_tracks(
            track_dets_pixel, all_tracks, track_id, target_folder,
            boxes_modality
        )

        if not frames:
            return

        # Determine starting position in the frame list.
        if start_at_last:
            start_idx = len(frames) - 1
        elif ref_frame is not None:
            frame_list = [f["frame_idx"] for f in frames]
            start_idx = frame_list.index(ref_frame) if ref_frame in frame_list else 0
        else:
            start_idx = 0

        title = f"Track {track_id}   |   {len(frames)} frame(s)"
        viewer = FeatureViewerDialog.get_instance(self.iface.mainWindow())
        viewer.show_track(
            title, frames, start_idx,
            target_folder=target_folder, dem_path=dem_path,
            correction_path=correction_path,
        )

    # ------------------------------------------------------------------
    # Frame-list builders
    # ------------------------------------------------------------------

    def _build_frames_from_pixel_tracks(
        self, track_dets, all_tracks, track_id, target_folder, boxes_modality: str
    ) -> List[dict]:
        """Build viewer frame list from pixel-space track data (core.inspection)."""
        return inspection.build_frames_from_pixel_tracks(
            track_dets, all_tracks, track_id, target_folder, boxes_modality)

    def _fill_interpolated_boxes(self, frames: List[dict]) -> None:
        inspection.fill_interpolated_boxes(frames)

    def _qgis_log(self, message: str, level: str = "info") -> None:
        QgsMessageLog.logMessage(
            message, "BAMBI",
            Qgis.MessageLevel.Warning if level == "warning" else Qgis.MessageLevel.Info)

    def _project_map_point(
        self,
        xy: Tuple[float, float],
        frame_idx: int,
        image_path: str,
        target_folder: str,
        dem_path: str,
        correction_path: str,
        modality: str,
    ) -> Optional[Tuple[float, float]]:
        """Project a map coordinate into image pixel space (core.inspection).

        The QgsRasterLayer elevation fallback and the QGIS message log are
        injected; the maths lives in :func:`core.inspection.project_map_point`.
        """
        return inspection.project_map_point(
            xy, frame_idx, target_folder, dem_path, correction_path, modality,
            mesh_cache=self._dem_mesh_cache,
            sample_elevation=self._sample_dem_elevation,
            log_fn=self._qgis_log,
        )

    def _sample_dem_elevation(
        self,
        xy: Tuple[float, float],
        origin: Tuple[float, float, float],
        dem_json_path: Optional[str],
    ) -> float:
        """Return the terrain elevation at *xy* in local z coordinates.

        Looks for a raster DEM (GeoTIFF) alongside *dem_json_path*: first
        a same-name ``.tif`` / ``.tiff``, then any ``.tif`` / ``.tiff`` in
        the same directory.  Uses QgsRasterLayer to sample the elevation and
        subtracts the DEM origin z so the result is in the same local
        coordinate space as the camera poses.

        Returns ``0.0`` when no raster is found or sampling fails.
        """
        if not dem_json_path:
            return 0.0
        try:
            from qgis.core import QgsRasterLayer, QgsPointXY

            dem_dir = os.path.dirname(dem_json_path)
            dem_base = os.path.splitext(dem_json_path)[0]

            # Prefer same-name TIF; fall back to any TIF in the directory.
            candidates: List[str] = []
            for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
                p = dem_base + ext
                if os.path.isfile(p):
                    candidates.append(p)
            for fname in os.listdir(dem_dir):
                if fname.lower().endswith((".tif", ".tiff")):
                    p = os.path.join(dem_dir, fname)
                    if p not in candidates:
                        candidates.append(p)

            point = QgsPointXY(xy[0], xy[1])
            for raster_path in candidates:
                layer = QgsRasterLayer(raster_path, "_bambi_dem_tmp", "gdal")
                if not layer.isValid():
                    continue
                value, ok = layer.dataProvider().sample(point, 1)
                if ok and value is not None:
                    return float(value) - origin[2]  # convert to local z

        except Exception:  # nosec B110
            pass
        return 0.0

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _load_pixel_detections(self, target_folder: str,
                               modality: str) -> List[dict]:
        return inspection.load_pixel_detections(target_folder, modality)

    def _load_pixel_tracks(self, target_folder: str,
                           modality: str) -> Dict[int, List[dict]]:
        return inspection.load_pixel_tracks(target_folder, modality)

    def _resolve_image_paths(self, target_folder: str, frame_idx: int) -> tuple:
        return inspection.resolve_image_paths(target_folder, frame_idx)

    # ------------------------------------------------------------------
    # Layer helpers
    # ------------------------------------------------------------------

    def _find_dem_mesh_path(
        self,
        dem_path: str,
        dem_json_path: Optional[str],
    ) -> Optional[str]:
        return inspection.find_dem_mesh_path(dem_path, dem_json_path)

    def _ray_cast_dem_z(
        self,
        local_xy: Tuple[float, float],
        mesh_path: Optional[str],
    ) -> Optional[float]:
        return inspection.ray_cast_dem_z(local_xy, mesh_path, self._dem_mesh_cache)

    def _resolve_missing_paths(
        self,
        target_folder: str,
        dem_path: str,
        correction_path: str,
    ) -> Tuple[str, str]:
        """Fill empty *dem_path* / *correction_path* from sibling BAMBI layers.

        FoV layers created before the DEM or correction fields were set in the
        UI will have empty custom properties.  Detection and track layers from
        the same flight (same ``bambi_target_folder``) carry the correct paths,
        so we borrow them when needed.
        """
        all_types = {DETECTION_TYPE} | TRACK_TYPES | {FOV_TYPE}
        for layer in QgsProject.instance().mapLayers().values():
            if not isinstance(layer, QgsVectorLayer):
                continue
            if layer.customProperty("bambi_layer_type", "") not in all_types:
                continue
            if layer.customProperty("bambi_target_folder", "") != target_folder:
                continue
            if not dem_path:
                candidate = layer.customProperty("bambi_dem_path", "")
                if candidate:
                    dem_path = candidate
            if not correction_path:
                candidate = layer.customProperty("bambi_correction_path", "")
                if candidate:
                    correction_path = candidate
            if dem_path and correction_path:
                break
        return dem_path, correction_path

    def _get_layer_tree_order(self) -> List[str]:
        """Return layer IDs in top-to-bottom order as they appear in the layer tree."""
        order: List[str] = []

        def _walk(node):
            from qgis.core import QgsLayerTreeLayer
            if isinstance(node, QgsLayerTreeLayer):
                order.append(node.layerId())
            else:
                for child in node.children():
                    _walk(child)

        _walk(QgsProject.instance().layerTreeRoot())
        return order

    def _get_bambi_layers(self) -> List[QgsVectorLayer]:
        """Return BAMBI layers relevant to the current mode.

        Detection/track mode: only visible layers are considered (so the user
        can hide a layer to exclude it from clicking).

        FoV mode: all FoV layers are considered regardless of visibility,
        because the user may want to click a FoV polygon that sits beneath a
        visible GeoTIFF raster.
        """
        valid_types = (
            {FOV_TYPE} if self.mode in ("fov", "fov_georef")
            else {DETECTION_TYPE} | TRACK_TYPES
        )
        root = QgsProject.instance().layerTreeRoot()
        layers = []
        for layer in QgsProject.instance().mapLayers().values():
            if not isinstance(layer, QgsVectorLayer):
                continue
            if layer.customProperty("bambi_layer_type", "") not in valid_types:
                continue
            if self.mode not in ("fov", "fov_georef"):
                node = root.findLayer(layer.id())
                if not (node and node.isVisible()):
                    continue
            layers.append(layer)
        return layers
