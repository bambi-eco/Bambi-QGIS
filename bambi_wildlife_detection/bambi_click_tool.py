# -*- coding: utf-8 -*-
"""
BAMBI Click Tool
================

QgsMapToolIdentify subclass that intercepts left-clicks on BAMBI detection /
track layers and opens the FeatureViewerDialog with the corresponding frame
image and annotated bounding boxes.

Layer custom properties used
----------------------------
``bambi_layer_type``    : "detection" | "track_final" | "track_path"
``bambi_target_folder`` : absolute path to the plugin output root folder

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
from typing import Dict, List, Optional

from qgis.gui import QgsMapToolIdentify
from qgis.core import QgsVectorLayer, QgsProject
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QCursor
from qgis.PyQt.QtWidgets import QMessageBox

from .bambi_feature_viewer import FeatureViewerDialog

TRACK_TYPES = {"track_final", "track_path"}
DETECTION_TYPE = "detection"


class BambiClickTool(QgsMapToolIdentify):
    """Map tool that opens FeatureViewerDialog on BAMBI layer feature clicks."""

    def __init__(self, iface):
        super().__init__(iface.mapCanvas())
        self.iface = iface
        self.setCursor(QCursor(Qt.CrossCursor))

    # ------------------------------------------------------------------
    # QgsMapTool overrides
    # ------------------------------------------------------------------

    def canvasReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        bambi_layers = self._get_bambi_layers()
        if not bambi_layers:
            return

        # Collect ALL results so we can prefer track features over detection
        # features even when detection layers render on top of track layers.
        results = self.identify(
            event.x(),
            event.y(),
            bambi_layers,
            QgsMapToolIdentify.TopDownAll,
        )
        if not results:
            return

        # Prefer any track result over detection results, because:
        # - Track "Final Position" polygons geographically overlap detection
        #   polygons (same bbox projected to the same coordinates).
        # - Detection layers are typically added (and therefore rendered on top)
        #   after track layers, so TopDownStopAtFirst would always pick the
        #   detection. We therefore scan all results and pick a track first.
        track_result = None
        detection_result = None
        for r in results:
            ltype = r.mLayer.customProperty("bambi_layer_type", "")
            if ltype in TRACK_TYPES and track_result is None:
                track_result = r
            elif ltype == DETECTION_TYPE and detection_result is None:
                detection_result = r

        chosen = track_result if track_result is not None else detection_result
        if chosen is None:
            return

        layer = chosen.mLayer
        feature = chosen.mFeature
        layer_type = layer.customProperty("bambi_layer_type", "")
        target_folder = layer.customProperty("bambi_target_folder", "")

        if not target_folder:
            return

        if layer_type == DETECTION_TYPE:
            self._handle_detection_click(feature, target_folder)
        elif layer_type == "track_final":
            self._handle_track_click(feature, target_folder, start_at_last=True)
        elif layer_type == "track_path":
            self._handle_track_click(feature, target_folder, start_at_last=False)

    def deactivate(self):
        super().deactivate()
        self.iface.mapCanvas().setCursor(Qt.ArrowCursor)

    # ------------------------------------------------------------------
    # Click handlers
    # ------------------------------------------------------------------

    def _handle_detection_click(self, feature, target_folder: str):
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

        image_path = self._resolve_image_path(target_folder, frame_idx)
        title = (
            f"Detection — Frame {frame_idx}"
            f"   |   conf: {det_conf:.3f}"
            f"   |   cls: {det_class}"
        )

        viewer = FeatureViewerDialog.get_instance(self.iface.mainWindow())
        viewer.show_detection(title, image_path, green_boxes, blue_boxes)

    def _handle_track_click(self, feature, target_folder: str, start_at_last: bool):
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
                track_dets_pixel, all_tracks, track_id, target_folder
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
                georef_dets, all_pixel_dets, target_folder
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
        viewer.show_track(title, frames, start_idx)

    # ------------------------------------------------------------------
    # Frame-list builders
    # ------------------------------------------------------------------

    def _build_frames_from_pixel_tracks(
        self, track_dets, all_tracks, track_id, target_folder
    ) -> List[dict]:
        """Build viewer frame list from pixel-space track data."""
        frames = []
        for det in track_dets:
            fi = det["frame"]
            other_on_frame = [
                (d["x1"], d["y1"], d["x2"], d["y2"], d["conf"], d["cls"])
                for tid, dets in all_tracks.items()
                if tid != track_id
                for d in dets
                if d["frame"] == fi
            ]
            frames.append({
                "frame_idx":   fi,
                "image_path":  self._resolve_image_path(target_folder, fi),
                "boxes_green": [
                    (det["x1"], det["y1"], det["x2"], det["y2"],
                     det["conf"], det["cls"])
                ],
                "boxes_blue": other_on_frame,
            })
        return frames

    def _build_frames_from_georef(
        self, georef_dets, all_pixel_dets, target_folder
    ) -> List[dict]:
        """Build viewer frame list by matching geo-referenced track detections
        back to pixel detections via (frame, confidence, class_id)."""
        frames = []
        for gd in georef_dets:
            fi   = gd["frame"]
            conf = gd["confidence"]
            cls  = gd["class_id"]

            same_frame = [d for d in all_pixel_dets if d["frame"] == fi]

            # Match by confidence ± tolerance and class
            matched = [
                d for d in same_frame
                if d["class_id"] == cls and abs(d["confidence"] - conf) < 0.0015
            ]
            others = [d for d in same_frame if d not in matched]

            frames.append({
                "frame_idx":  fi,
                "image_path": self._resolve_image_path(target_folder, fi),
                "boxes_green": [
                    (d["x1"], d["y1"], d["x2"], d["y2"], d["confidence"], d["class_id"])
                    for d in matched
                ],
                "boxes_blue": [
                    (d["x1"], d["y1"], d["x2"], d["y2"], d["confidence"], d["class_id"])
                    for d in others
                ],
            })
        return frames

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
                                "frame":      int(parts[0]),
                                "confidence": float(parts[8]),
                                "class_id":   int(parts[9]),
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

    def _resolve_image_path(self, target_folder: str, frame_idx: int) -> str:
        """Return the filesystem path for frame *frame_idx*.

        Tries thermal (``poses_t.json`` / ``frames_t/``) first, then RGB.
        Returns an empty string if the image cannot be found.
        """
        for poses_name, frames_dir in [
            ("poses_t.json", "frames_t"),
            ("poses_w.json", "frames_w"),
        ]:
            poses_path = os.path.join(target_folder, poses_name)
            if not os.path.isfile(poses_path):
                continue
            try:
                with open(poses_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                images = data.get("images", [])
                if 0 <= frame_idx < len(images):
                    imagefile = images[frame_idx].get("imagefile", "")
                    candidate = os.path.join(target_folder, frames_dir, imagefile)
                    if imagefile and os.path.isfile(candidate):
                        return candidate
            except Exception:
                continue
        return ""

    # ------------------------------------------------------------------
    # Layer helpers
    # ------------------------------------------------------------------

    def _get_bambi_layers(self) -> List[QgsVectorLayer]:
        """Return BAMBI vector layers that are currently visible in the layer panel."""
        valid_types = {DETECTION_TYPE} | TRACK_TYPES
        root = QgsProject.instance().layerTreeRoot()
        layers = []
        for layer in QgsProject.instance().mapLayers().values():
            if not isinstance(layer, QgsVectorLayer):
                continue
            if layer.customProperty("bambi_layer_type", "") not in valid_types:
                continue
            node = root.findLayer(layer.id())
            if node and node.isVisible():
                layers.append(layer)
        return layers
