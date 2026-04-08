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

Data files read (relative to *target_folder*)
---------------------------------------------
``detections/detections.txt``  — pixel-space detections
    format: ``frame x1 y1 x2 y2 confidence class_id`` (space-separated)
    header line starts with ``#``

``tracks/tracks_pixel.csv``    — pixel-space track detections
    format: ``frame,track_id,x1,y1,x2,y2,conf,cls[,interpolated]``
    header line starts with ``#``

``poses_t.json`` / ``poses_w.json`` — frame → image file mapping
    structure: ``{"images": [{"imagefile": "...", ...}, ...]}``
    images stored under ``frames_t/`` or ``frames_w/`` respectively
"""

import os
import json
from typing import Dict, List, Optional, Tuple

from qgis.gui import QgsMapToolIdentify
from qgis.core import QgsVectorLayer, QgsProject, QgsMessageLog, Qgis
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QCursor
from qgis.PyQt.QtWidgets import QMessageBox

from .bambi_feature_viewer import FeatureViewerDialog

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
        self.setCursor(QCursor(Qt.CrossCursor))
        # DEM mesh cache: keyed by absolute mesh path so repeated FoV clicks
        # on the same DEM don't reload and re-build the BVH every time.
        self._dem_mesh_cache: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # QgsMapTool overrides
    # ------------------------------------------------------------------

    def canvasReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
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
            QgsMapToolIdentify.TopDownAll,
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
        layer_type        = layer.customProperty("bambi_layer_type", "")
        target_folder     = layer.customProperty("bambi_target_folder", "")
        dem_path          = layer.customProperty("bambi_dem_path", "")
        correction_path   = layer.customProperty("bambi_correction_path", "")
        # "T" = thermal, "W" = RGB; determines which pixel space the boxes are in
        detection_camera  = layer.customProperty("bambi_detection_camera", "T")
        boxes_modality    = "t" if detection_camera == "T" else "w"

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
        self.iface.mapCanvas().setCursor(Qt.ArrowCursor)

    # ------------------------------------------------------------------
    # Click handlers
    # ------------------------------------------------------------------

    def _handle_detection_click(self, feature, target_folder: str,
                                boxes_modality: str, dem_path: str,
                                correction_path: str):
        """Show the frame for a clicked detection bounding box."""
        frame_idx = feature["frame"]
        det_conf  = float(feature["confidence"])
        det_class = int(feature["class_id"])

        det_file  = os.path.join(target_folder, "detections", "detections.txt")
        all_dets  = self._load_pixel_detections(det_file)

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
            layer   = result.mLayer
            feature = result.mFeature

            try:
                frame_idx = int(feature["frame"])
            except (TypeError, ValueError):
                continue

            target_folder    = layer.customProperty("bambi_target_folder", "")
            dem_path         = layer.customProperty("bambi_dem_path", "")
            correction_path  = layer.customProperty("bambi_correction_path", "")
            detection_camera = layer.customProperty("bambi_detection_camera", "T")
            boxes_modality   = "t" if detection_camera == "T" else "w"

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

            det_file   = os.path.join(target_folder, "detections", "detections.txt")
            all_dets   = self._load_pixel_detections(det_file)
            same_frame = [d for d in all_dets if d["frame"] == frame_idx]

            # All detections on this frame are shown in green — there is no
            # "clicked" detection to single out.
            green_boxes = [
                (d["x1"], d["y1"], d["x2"], d["y2"], d["confidence"], d["class_id"])
                for d in same_frame
            ]
            image_path_t, image_path_w = self._resolve_image_paths(target_folder, frame_idx)

            frame_dict = {
                "frame_idx":      frame_idx,
                "image_path_t":   image_path_t,
                "image_path_w":   image_path_w,
                "boxes_modality": boxes_modality,
                "boxes_green":    green_boxes,
                "boxes_blue":     [],
                # Per-frame projection context (may differ across layers).
                "target_folder":  target_folder,
                "dem_path":       dem_path,
                "correction_path": correction_path,
            }

            # Project the clicked map position into this frame's image space.
            # Failures are non-fatal — the frame is still shown, just without
            # the crosshair.
            if click_xy is not None:
                try:
                    frame_dict["click_point_t"] = self._project_map_point(
                        click_xy, frame_idx, image_path_t,
                        target_folder, dem_path, correction_path, "t",
                    )
                    frame_dict["click_point_w"] = self._project_map_point(
                        click_xy, frame_idx, image_path_w,
                        target_folder, dem_path, correction_path, "w",
                    )
                except Exception:
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

        # Load all pixel-space detections once — needed by both paths below.
        det_file = os.path.join(target_folder, "detections", "detections.txt")
        all_pixel_dets = self._load_pixel_detections(det_file)

        # --- Primary path: tracks_pixel.csv (built-in / BoxMOT trackers) ---
        tracks_file = os.path.join(target_folder, "tracks", "tracks_pixel.csv")
        track_dets_pixel = []
        if os.path.isfile(tracks_file):
            all_tracks = self._load_pixel_tracks(tracks_file)
            track_dets_pixel = sorted(
                all_tracks.get(track_id, []), key=lambda d: d["frame"]
            )

        if track_dets_pixel:
            frames = self._build_frames_from_pixel_tracks(
                track_dets_pixel, all_tracks, track_id, target_folder, boxes_modality
            )
        else:
            # --- Fallback: geo-referenced CSVs + detections.txt ---
            # Used when pixel tracks don't exist (geo-referenced native tracker)
            # or when the track_id is absent from tracks_pixel.csv.
            georef_dets = self._load_georef_track_dets(target_folder, track_id)
            if not georef_dets:
                QMessageBox.warning(
                    None,
                    "BAMBI Inspector",
                    f"No frame data found for track {track_id}.\n\n"
                    "Neither tracks_pixel.csv nor any geo-referenced CSV "
                    "contains entries for this track.",
                )
                return
            frames = self._build_frames_from_georef(
                georef_dets, all_pixel_dets, target_folder, boxes_modality
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
        """Build viewer frame list from pixel-space track data.

        The ``interpolated`` flag from tracks_pixel.csv is forwarded as the
        7th element of each green box tuple so the viewer can draw dashed lines.
        """
        frames = []
        for det in track_dets:
            fi = det["frame"]
            is_interp = det.get("interpolated", 0)
            other_on_frame = [
                (d["x1"], d["y1"], d["x2"], d["y2"], d["conf"], d["cls"])
                for tid, dets in all_tracks.items()
                if tid != track_id
                for d in dets
                if d["frame"] == fi
            ]
            path_t, path_w = self._resolve_image_paths(target_folder, fi)
            frames.append({
                "frame_idx":      fi,
                "image_path_t":   path_t,
                "image_path_w":   path_w,
                "boxes_modality": boxes_modality,
                "boxes_green": [
                    (det["x1"], det["y1"], det["x2"], det["y2"],
                     det["conf"], det["cls"], is_interp)
                ],
                "boxes_blue": other_on_frame,
            })
        return frames

    def _build_frames_from_georef(
        self, georef_dets, all_pixel_dets, target_folder, boxes_modality: str
    ) -> List[dict]:
        """Build viewer frame list by matching geo-referenced track detections
        back to pixel detections via (frame, confidence, class_id).

        Frames where no pixel match is found (interpolated frames whose bbox
        was never written to detections.txt) get their bounding box
        interpolated linearly from neighbouring actual detections and are
        flagged with is_interpolated=1 so the viewer draws a dashed border.
        """
        frames = []
        for gd in georef_dets:
            fi        = gd["frame"]
            conf      = gd["confidence"]
            cls       = gd["class_id"]
            is_interp = gd.get("interpolated", 0)

            same_frame = [d for d in all_pixel_dets if d["frame"] == fi]

            # Match by confidence ± tolerance and class
            matched = [
                d for d in same_frame
                if d["class_id"] == cls and abs(d["confidence"] - conf) < 0.0015
            ]
            others = [d for d in same_frame if d not in matched]

            path_t, path_w = self._resolve_image_paths(target_folder, fi)
            frames.append({
                "frame_idx":      fi,
                "image_path_t":   path_t,
                "image_path_w":   path_w,
                "boxes_modality": boxes_modality,
                "boxes_green": [
                    (d["x1"], d["y1"], d["x2"], d["y2"],
                     d["confidence"], d["class_id"], is_interp)
                    for d in matched
                ],
                "boxes_blue": [
                    (d["x1"], d["y1"], d["x2"], d["y2"],
                     d["confidence"], d["class_id"])
                    for d in others
                ],
            })

        # For frames where the pixel match failed (interpolated frames that
        # were never written to detections.txt), interpolate from neighbours.
        self._fill_interpolated_boxes(frames)
        return frames

    def _fill_interpolated_boxes(self, frames: List[dict]) -> None:
        """Fill empty ``boxes_green`` entries by linear interpolation.

        Operates in-place.  Interpolated boxes are flagged with
        ``is_interpolated=1`` so the viewer draws them with a dashed border.
        """
        n = len(frames)
        for i, frame in enumerate(frames):
            if frame["boxes_green"]:
                continue  # already has a real or previously-interpolated box

            # Find nearest frames before and after that have actual boxes
            j_before = next(
                (j for j in range(i - 1, -1, -1) if frames[j]["boxes_green"]),
                None,
            )
            j_after = next(
                (j for j in range(i + 1, n) if frames[j]["boxes_green"]),
                None,
            )

            if j_before is None and j_after is None:
                continue  # no reference boxes anywhere — leave empty

            if j_before is not None and j_after is not None:
                b1 = frames[j_before]["boxes_green"][0]
                b2 = frames[j_after]["boxes_green"][0]
                fi1, fi2, fi = (
                    frames[j_before]["frame_idx"],
                    frames[j_after]["frame_idx"],
                    frame["frame_idx"],
                )
                alpha = (fi - fi1) / (fi2 - fi1) if fi2 != fi1 else 0.5
            else:
                # Only one side available — copy the nearest box unchanged
                src = frames[j_before if j_before is not None else j_after]
                b1 = b2 = src["boxes_green"][0]
                alpha = 0.0

            x1 = b1[0] + alpha * (b2[0] - b1[0])
            y1 = b1[1] + alpha * (b2[1] - b1[1])
            x2 = b1[2] + alpha * (b2[2] - b1[2])
            y2 = b1[3] + alpha * (b2[3] - b1[3])
            conf = float(b1[4]) if len(b1) > 4 else 0.0
            cls  = int(b1[5])   if len(b1) > 5 else 0

            frame["boxes_green"] = [(x1, y1, x2, y2, conf, cls, 1)]

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
        """Project a geographic map coordinate into image pixel space.

        Uses the same camera model as :class:`~.bambi_box_projector.BoxProjectionWorker`.
        The z component of the world point is set to the DEM datum elevation
        (i.e. ``local_z = 0`` after subtracting the DEM origin), which is a
        good approximation for ground-level points.

        Returns ``(pixel_x, pixel_y)`` when the projection falls within the
        image bounds, or ``None`` on failure / out-of-frame.
        """
        poses_path = os.path.join(target_folder, f"poses_{modality}.json")
        if not os.path.isfile(poses_path):
            return None

        try:
            import numpy as np
            from pyrr import Vector3, Quaternion
            from alfspy.core.rendering import Camera
            from .bambi_box_projector import (
                _read_correction, _correction_for_frame, _world_to_pixel,
            )
        except ImportError:
            return None

        try:
            with open(poses_path, "r", encoding="utf-8") as fh:
                poses_data = json.load(fh)
            images = poses_data.get("images", [])
            if frame_idx >= len(images):
                return None

            # ---- DEM origin (mirrors BoxProjectionWorker fallback chain) ----
            def _try_load_origin(json_path):
                if not json_path or not os.path.isfile(json_path):
                    return None
                try:
                    with open(json_path, "r", encoding="utf-8") as fh:
                        d = json.load(fh)
                    o = d.get("origin")
                    if o and len(o) >= 3:
                        return (float(o[0]), float(o[1]), float(o[2]))
                except Exception:
                    pass
                return None

            # Derive the candidate JSON paths from dem_path, handling both
            # mesh files (.gltf/.glb) and the case where dem_path already
            # points directly to the JSON metadata file.
            origin = (0.0, 0.0, 0.0)
            dem_json_found = None      # path of the JSON that provided the origin
            dem_json_tried = "N/A"
            if dem_path:
                if dem_path.lower().endswith(".json"):
                    dem_json_tried = dem_path
                else:
                    dem_json_tried = (
                        dem_path.replace(".gltf", ".json").replace(".glb", ".json")
                    )
                o = _try_load_origin(dem_json_tried)
                if o:
                    origin = o
                    dem_json_found = dem_json_tried

            if origin == (0.0, 0.0, 0.0):
                for search_dir in [target_folder, os.path.dirname(target_folder)]:
                    if not search_dir or not os.path.isdir(search_dir):
                        continue
                    for fname in os.listdir(search_dir):
                        if not fname.lower().endswith(".json"):
                            continue
                        json_path = os.path.join(search_dir, fname)
                        o = _try_load_origin(json_path)
                        if o and (o[0] != 0.0 or o[1] != 0.0):
                            origin = o
                            dem_json_found = json_path
                            break
                    if origin != (0.0, 0.0, 0.0):
                        break

            # ---- Terrain elevation at click position ---------------------
            # Local (x, y) after removing DEM origin offset — needed for mesh
            # ray-casting which operates in local coordinate space.
            local_xy = (xy[0] - origin[0], xy[1] - origin[1])

            # Try mesh ray-cast first (sub-metre accuracy); fall back to
            # raster DEM sampling if the mesh is unavailable.
            mesh_path = self._find_dem_mesh_path(dem_path, dem_json_found)
            local_z = self._ray_cast_dem_z(local_xy, mesh_path)
            elev_method = "mesh" if local_z is not None else "raster"
            if local_z is None:
                local_z = self._sample_dem_elevation(xy, origin, dem_json_found)

            QgsMessageLog.logMessage(
                f"[FoV click projection | {modality}]  Origin lookup\n"
                f"  dem_path     : {dem_path!r}\n"
                f"  JSON tried   : {dem_json_tried!r}  exists={os.path.isfile(dem_json_tried)}\n"
                f"  origin found : {origin}\n"
                f"  local_z      : {local_z:.3f}  (via {elev_method})",
                "BAMBI", Qgis.Info,
            )

            # ---- Correction ---------------------------------------------
            corr = _read_correction(target_folder, correction_path)
            t_corr, r_corr = _correction_for_frame(frame_idx, corr)

            # ---- Camera -------------------------------------------------
            meta     = images[frame_idx]
            fovy     = meta.get("fovy", 50)
            if isinstance(fovy, list):
                fovy = fovy[0]
            position = Vector3(meta["location"])
            rot_vals = meta["rotation"]

            cor_t = Vector3(
                [t_corr.get("x", 0.0), t_corr.get("y", 0.0), t_corr.get("z", 0.0)],
                dtype="f4",
            )
            cor_r = Vector3(
                [r_corr.get("x", 0.0), r_corr.get("y", 0.0), r_corr.get("z", 0.0)],
                dtype="f4",
            )
            # Wrap to (-180, +180] before converting to radians — matches the
            # correction wizard's _wrap_deg() to avoid edge cases near ±180°.
            rotation_eulers = (
                Vector3([np.deg2rad(((v + 180.0) % 360.0) - 180.0) for v in rot_vals]) - cor_r
            ) * -1
            position       = position + cor_t
            rotation_quat  = Quaternion.from_eulers(rotation_eulers)
            camera = Camera(fovy=fovy, aspect_ratio=1.0, position=position,
                            rotation=rotation_quat)

            # ---- Image dimensions ---------------------------------------
            img_width, img_height = 640, 512  # sensible fallback
            if image_path:
                from qgis.PyQt.QtGui import QImage
                qimg = QImage(image_path)
                if not qimg.isNull():
                    img_width  = qimg.width()
                    img_height = qimg.height()

            # ---- Project ------------------------------------------------
            # Convert to local coords by subtracting the DEM origin.
            local = np.array(
                [[xy[0] - origin[0], xy[1] - origin[1], local_z]],
                dtype=np.float64,
            )

            canvas_crs = self.canvas().mapSettings().destinationCrs().authid()
            QgsMessageLog.logMessage(
                f"[FoV click projection | {modality}]\n"
                f"  Canvas CRS   : {canvas_crs}\n"
                f"  Map click    : x={xy[0]:.3f}  y={xy[1]:.3f}\n"
                f"  DEM origin   : x={origin[0]:.3f}  y={origin[1]:.3f}  z={origin[2]:.3f}\n"
                f"  Local point  : x={local[0,0]:.3f}  y={local[0,1]:.3f}  z={local[0,2]:.3f}\n"
                f"  Cam position : {list(meta.get('location', []))}\n"
                f"  Correction t : {t_corr}\n"
                f"  Correction r : {r_corr}\n"
                f"  Image size   : {img_width} x {img_height}",
                "BAMBI", Qgis.Info,
            )

            pxs, pys = _world_to_pixel(local, img_width, img_height, camera)
            px, py = float(pxs[0]), float(pys[0])

            QgsMessageLog.logMessage(
                f"[FoV click projection | {modality}]  "
                f"Projected pixel: px={px:.1f}  py={py:.1f}  "
                f"(in bounds: {0 <= px <= img_width and 0 <= py <= img_height})",
                "BAMBI", Qgis.Info,
            )

            if 0 <= px <= img_width and 0 <= py <= img_height:
                return (px, py)
            return None

        except Exception as exc:
            QgsMessageLog.logMessage(
                f"[FoV click projection | {modality}]  Exception: {exc}",
                "BAMBI", Qgis.Warning,
            )
            return None

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

            dem_dir  = os.path.dirname(dem_json_path)
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

        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _load_georef_track_dets(self, target_folder: str, track_id: int) -> List[dict]:
        """Return frame/confidence/class_id entries for *track_id* from the
        geo-referenced track CSVs (``tracks/*.csv``, excluding ``_pixel.csv``).

        Format: ``frame,track_id,x1,y1,z1,x2,y2,z2,confidence,class_id[,interpolated]``
        """
        tracks_folder = os.path.join(target_folder, "tracks")
        if not os.path.isdir(tracks_folder):
            return []

        result: List[dict] = []
        try:
            for fname in os.listdir(tracks_folder):
                if not fname.endswith(".csv") or fname.endswith("_pixel.csv"):
                    continue
                csv_path = os.path.join(tracks_folder, fname)
                with open(csv_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split(",")
                        if len(parts) >= 10 and int(parts[1]) == track_id:
                            result.append({
                                "frame":        int(parts[0]),
                                "confidence":   float(parts[8]),
                                "class_id":     int(parts[9]),
                                "interpolated": int(parts[10]) if len(parts) > 10 else 0,
                            })
                if result:
                    break  # found in this file, no need to search further
        except Exception:
            pass

        return sorted(result, key=lambda d: d["frame"])

    def _load_pixel_detections(self, det_file: str) -> List[dict]:
        """Parse ``detections/detections.txt``.

        Format (space-separated, comment header with #):
        ``frame x1 y1 x2 y2 confidence class_id``
        """
        result: List[dict] = []
        if not os.path.isfile(det_file):
            return result
        try:
            with open(det_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 6:
                        result.append({
                            "frame":      int(parts[0]),
                            "x1":         float(parts[1]),
                            "y1":         float(parts[2]),
                            "x2":         float(parts[3]),
                            "y2":         float(parts[4]),
                            "confidence": float(parts[5]),
                            "class_id":   int(parts[6]) if len(parts) > 6 else 0,
                        })
        except Exception:
            pass
        return result

    def _load_pixel_tracks(self, tracks_file: str) -> Dict[int, List[dict]]:
        """Parse ``tracks/tracks_pixel.csv``.

        Format (comma-separated, comment header with #):
        ``frame,track_id,x1,y1,x2,y2,conf,cls[,interpolated]``
        """
        result: Dict[int, List[dict]] = {}
        if not os.path.isfile(tracks_file):
            return result
        try:
            with open(tracks_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    if len(parts) >= 8:
                        tid = int(parts[1])
                        entry = {
                            "frame": int(parts[0]),
                            "x1":    float(parts[2]),
                            "y1":    float(parts[3]),
                            "x2":    float(parts[4]),
                            "y2":    float(parts[5]),
                            "conf":  float(parts[6]),
                            "cls":   int(parts[7]),
                        }
                        result.setdefault(tid, []).append(entry)
        except Exception:
            pass
        return result

    def _resolve_image_paths(self, target_folder: str, frame_idx: int) -> tuple:
        """Return ``(path_t, path_w)`` for frame *frame_idx*.

        Each element is the filesystem path to the thermal / RGB frame image,
        or an empty string when that frame type has not been extracted.
        """
        paths = []
        for poses_name, frames_dir in [
            ("poses_t.json", "frames_t"),
            ("poses_w.json", "frames_w"),
        ]:
            poses_path = os.path.join(target_folder, poses_name)
            found = ""
            if os.path.isfile(poses_path):
                try:
                    with open(poses_path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    images = data.get("images", [])
                    if 0 <= frame_idx < len(images):
                        imagefile = images[frame_idx].get("imagefile", "")
                        candidate = os.path.join(target_folder, frames_dir, imagefile)
                        if imagefile and os.path.isfile(candidate):
                            found = candidate
                except Exception:
                    pass
            paths.append(found)
        return paths[0], paths[1]

    # ------------------------------------------------------------------
    # Layer helpers
    # ------------------------------------------------------------------

    def _find_dem_mesh_path(
        self,
        dem_path: str,
        dem_json_path: Optional[str],
    ) -> Optional[str]:
        """Return the path to the DEM mesh file (.glb / .gltf), or None."""
        if dem_path and dem_path.lower().endswith((".glb", ".gltf")):
            if os.path.isfile(dem_path):
                return dem_path
        if dem_json_path:
            base = os.path.splitext(dem_json_path)[0]
            for ext in (".glb", ".gltf"):
                candidate = base + ext
                if os.path.isfile(candidate):
                    return candidate
        return None

    def _ray_cast_dem_z(
        self,
        local_xy: Tuple[float, float],
        mesh_path: Optional[str],
    ) -> Optional[float]:
        """Return local z of the DEM mesh at *(local_x, local_y)* via a
        downward vertical ray-cast.

        The mesh and its BVH are cached by path so subsequent calls for the
        same DEM are cheap.  Returns ``None`` on failure or if no mesh is
        available.
        """
        if not mesh_path:
            return None
        try:
            import numpy as np
            from alfspy.render.render import read_gltf
            from trimesh import Trimesh

            if mesh_path not in self._dem_mesh_cache:
                mesh_data, _ = read_gltf(mesh_path)
                tri_mesh = Trimesh(
                    vertices=mesh_data.vertices,
                    faces=mesh_data.indices,
                )
                try:
                    _ = tri_mesh.triangles_tree   # pre-build BVH
                except Exception:
                    pass
                self._dem_mesh_cache[mesh_path] = tri_mesh

            tri_mesh = self._dem_mesh_cache[mesh_path]

            ray_origins    = np.array([[local_xy[0], local_xy[1], 10_000.0]])
            ray_directions = np.array([[0.0, 0.0, -1.0]])
            locations, _, _ = tri_mesh.ray.intersects_location(
                ray_origins, ray_directions
            )
            if len(locations) > 0:
                return float(np.max(locations[:, 2]))
        except Exception:
            pass
        return None

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
