# -*- coding: utf-8 -*-
"""
BAMBI Video Creator
===================
Dialog for turning processed BAMBI results into an MP4 video.

The user composes a video from up to three side-by-side panels:

  * one or two *video* panels (RGB / Thermal) built from either the extracted
    frames (``frames_{t,w}/``) or the orthographic projection (per-frame
    GeoTIFFs ``geotiffs_{t,w}/`` or the ALFS mosaic ``alfs_{t,w}/``), optionally
    with detection or track overlays; and
  * one *map* panel rendered from the live QGIS project (flight path, current
    field of view, detections/tracks, perpendicular distances, a merged-FoV
    background and/or a frame-by-frame accumulating FoV that shows the area
    monitored so far).

At least one video modality *or* the map panel is required.

Rendering runs on the GUI thread (QGIS map rendering must happen there) and
keeps the UI responsive via ``QApplication.processEvents()``; a Cancel button
aborts cleanly.
"""

import json
import os

from qgis.PyQt.QtCore import Qt, QSize, QVariant
from qgis.PyQt.QtGui import QColor, QImage
from qgis.PyQt.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QRadioButton, QScrollArea, QSpinBox,
    QTextEdit, QVBoxLayout, QWidget,
)
from qgis.core import (
    QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsFeature,
    QgsField, QgsFillSymbol, QgsGeometry, QgsLineSymbol, QgsMapRendererParallelJob,
    QgsMapSettings, QgsPointXY, QgsProject, QgsRectangle, QgsSingleSymbolRenderer,
    QgsVectorLayer,
)


# Distinct colors reused for per-track overlay boxes / markers.
_TRACK_COLORS = [
    (229, 57, 53), (142, 36, 170), (30, 136, 229), (0, 172, 193),
    (67, 160, 71), (251, 140, 0), (109, 76, 65), (3, 155, 229),
    (216, 27, 96), (94, 53, 177), (0, 137, 123), (192, 202, 51),
]


def _track_color_bgr(track_id):
    r, g, b = _TRACK_COLORS[int(track_id) % len(_TRACK_COLORS)]
    return (b, g, r)


class _Cancelled(Exception):
    """Raised internally when the user cancels the render."""


class VideoCreatorDialog(QDialog):
    """Non-modal dialog that composes an MP4 from processed BAMBI results."""

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self._cancel = False
        self._rendering = False
        self.setWindowTitle("BAMBI Video Creator")
        self.resize(640, 820)
        self._build_ui()
        self._update_enabled_state()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)
        sv = QVBoxLayout(content)
        sv.setSpacing(8)
        outer.addWidget(scroll, stretch=1)

        sv.addWidget(self._build_source_group())
        sv.addWidget(self._build_frame_source_group())
        sv.addWidget(self._build_modality_group())
        sv.addWidget(self._build_overlay_group())
        sv.addWidget(self._build_map_group())
        sv.addWidget(self._build_output_group())
        sv.addStretch()

        self._hint_label = QLabel("")
        self._hint_label.setStyleSheet("color: #c0392b;")
        self._hint_label.setWordWrap(True)
        outer.addWidget(self._hint_label)

        btn_row = QHBoxLayout()
        self._create_btn = QPushButton("Create Video")
        self._create_btn.clicked.connect(self._on_create_clicked)
        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.close)
        btn_row.addWidget(self._create_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._close_btn)
        outer.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        outer.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(140)
        outer.addWidget(self._log)

    def _build_source_group(self):
        grp = QGroupBox("Processed Data")
        fl = QFormLayout(grp)
        self._target_edit = QLineEdit()
        self._target_edit.setPlaceholderText("Processing target folder (contains frames_t / frames_w …)")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_target)
        row = QHBoxLayout()
        row.addWidget(self._target_edit)
        row.addWidget(browse)
        w = QWidget()
        w.setLayout(row)
        fl.addRow("Target Folder:", w)

        self._data_epsg_edit = QLineEdit()
        crs = QgsProject.instance().crs()
        self._data_epsg_edit.setText(crs.authid() if crs.isValid() else "EPSG:32633")
        self._data_epsg_edit.setToolTip(
            "CRS of the geo-referenced products (FoV / perpendicular). Used to place "
            "map overlays. Defaults to the current project CRS.")
        fl.addRow("Data CRS:", self._data_epsg_edit)
        return grp

    def _build_frame_source_group(self):
        grp = QGroupBox("1. Frame Source")
        vl = QVBoxLayout(grp)
        self._src_extracted = QRadioButton("Extracted frames")
        self._src_ortho = QRadioButton("Orthographic projection")
        self._src_extracted.setChecked(True)
        self._src_group = QButtonGroup(self)
        self._src_group.addButton(self._src_extracted)
        self._src_group.addButton(self._src_ortho)
        vl.addWidget(self._src_extracted)

        ortho_row = QHBoxLayout()
        ortho_row.addWidget(self._src_ortho)
        self._ortho_kind = QComboBox()
        self._ortho_kind.addItem("Per-frame GeoTIFFs", "geotiffs")
        self._ortho_kind.addItem("ALFS mosaic (static)", "alfs")
        self._ortho_kind.setToolTip(
            "Per-frame GeoTIFFs animate one orthorectified image per frame; "
            "the ALFS mosaic shows the single merged image on every frame.")
        ortho_row.addWidget(self._ortho_kind)
        ortho_row.addStretch()
        vl.addLayout(ortho_row)

        self._src_ortho.toggled.connect(self._update_enabled_state)
        return grp

    def _build_modality_group(self):
        grp = QGroupBox("2. Modality (video panels)")
        hl = QHBoxLayout(grp)
        self._mod_rgb = QRadioButton("RGB")
        self._mod_thermal = QRadioButton("Thermal")
        self._mod_both = QRadioButton("Both")
        self._mod_none = QRadioButton("None")
        self._mod_rgb.setChecked(True)
        self._mod_group = QButtonGroup(self)
        for b in (self._mod_rgb, self._mod_thermal, self._mod_both, self._mod_none):
            self._mod_group.addButton(b)
            hl.addWidget(b)
            b.toggled.connect(self._update_enabled_state)
        hl.addStretch()
        return grp

    def _build_overlay_group(self):
        grp = QGroupBox("3. Overlay on video panels")
        hl = QHBoxLayout(grp)
        self._ov_none = QRadioButton("None")
        self._ov_det = QRadioButton("Detections")
        self._ov_trk = QRadioButton("Tracks")
        self._ov_none.setChecked(True)
        self._ov_group = QButtonGroup(self)
        for b in (self._ov_none, self._ov_det, self._ov_trk):
            self._ov_group.addButton(b)
            hl.addWidget(b)
        hl.addStretch()
        grp.setToolTip(
            "Overlays are drawn in extracted-frame pixel space and therefore only "
            "appear on the 'Extracted frames' source.")
        return grp

    def _build_map_group(self):
        grp = QGroupBox("4. Map View")
        vl = QVBoxLayout(grp)
        self._map_enable = QCheckBox("Include map view panel")
        self._map_enable.toggled.connect(self._update_enabled_state)
        vl.addWidget(self._map_enable)

        self._map_options = QWidget()
        ol = QVBoxLayout(self._map_options)
        ol.setContentsMargins(18, 0, 0, 0)
        self._map_flight = QCheckBox("Flight path")
        self._map_fov = QCheckBox("Current field of view")
        self._map_dettrk = QCheckBox("Detections / tracks")
        self._map_perp = QCheckBox("Perpendicular distances")
        self._map_merged_bg = QCheckBox("Merged FoV as background")
        self._map_accumulate = QCheckBox("Accumulate FoV frame-by-frame (monitored area)")
        self._map_follow = QCheckBox("Follow drone (track current FoV)")
        self._map_flight.setChecked(True)
        self._map_fov.setChecked(True)
        self._map_follow.setChecked(True)
        for c in (self._map_flight, self._map_fov, self._map_dettrk, self._map_perp,
                  self._map_merged_bg, self._map_accumulate, self._map_follow):
            ol.addWidget(c)
        vl.addWidget(self._map_options)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("FoV source camera:"))
        self._map_camera = QComboBox()
        self._map_camera.addItem("Thermal", "t")
        self._map_camera.addItem("RGB", "w")
        cam_row.addWidget(self._map_camera)
        cam_row.addStretch()
        self._map_options.layout().addLayout(cam_row)
        return grp

    def _build_output_group(self):
        grp = QGroupBox("5. Output")
        fl = QFormLayout(grp)
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Output .mp4 file")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_output)
        row = QHBoxLayout()
        row.addWidget(self._out_edit)
        row.addWidget(browse)
        w = QWidget()
        w.setLayout(row)
        fl.addRow("Video File:", w)

        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 60)
        self._fps_spin.setValue(10)
        fl.addRow("Frames per second:", self._fps_spin)

        self._height_spin = QSpinBox()
        self._height_spin.setRange(240, 2160)
        self._height_spin.setSingleStep(120)
        self._height_spin.setValue(720)
        fl.addRow("Panel height (px):", self._height_spin)
        return grp

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _browse_target(self):
        path = QFileDialog.getExistingDirectory(self, "Select Target Folder",
                                                self._target_edit.text().strip())
        if path:
            self._target_edit.setText(path)
            if not self._out_edit.text().strip():
                self._out_edit.setText(os.path.join(path, "bambi_video.mp4"))

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Video As", self._out_edit.text().strip(), "MP4 Video (*.mp4)")
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self._out_edit.setText(path)

    def _modality(self):
        if self._mod_rgb.isChecked():
            return "rgb"
        if self._mod_thermal.isChecked():
            return "thermal"
        if self._mod_both.isChecked():
            return "both"
        return "none"

    def _video_suffixes(self):
        """Camera suffixes for the requested video panels, left-to-right."""
        mod = self._modality()
        if mod == "rgb":
            return ["w"]
        if mod == "thermal":
            return ["t"]
        if mod == "both":
            return ["t", "w"]
        return []

    def _overlay(self):
        if self._ov_det.isChecked():
            return "detections"
        if self._ov_trk.isChecked():
            return "tracks"
        return "none"

    def _update_enabled_state(self):
        self._ortho_kind.setEnabled(self._src_ortho.isChecked())
        self._map_options.setEnabled(self._map_enable.isChecked())

        has_video = self._modality() != "none"
        has_map = self._map_enable.isChecked()
        ok = has_video or has_map
        self._create_btn.setEnabled(ok and not self._rendering)
        if not ok:
            self._hint_label.setText(
                "Select at least one video modality (RGB / Thermal / Both) or enable the map view.")
        else:
            self._hint_label.setText("")

    def _log_msg(self, text):
        self._log.append(text)
        QApplication.processEvents()

    # ------------------------------------------------------------------
    # Create / cancel
    # ------------------------------------------------------------------

    def _on_create_clicked(self):
        if self._rendering:
            self._cancel = True
            self._log_msg("Cancelling…")
            return
        try:
            params = self._collect_params()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Configuration", str(exc))
            return

        self._rendering = True
        self._cancel = False
        self._create_btn.setText("Cancel")
        self._close_btn.setEnabled(False)
        self._progress.setValue(0)
        self._log.clear()
        try:
            out = self._render(params)
            if out:
                self._log_msg(f"Done. Saved: {out}")
                QMessageBox.information(self, "Video Created", f"Video saved to:\n{out}")
        except _Cancelled:
            self._log_msg("Cancelled by user.")
        except Exception as exc:  # noqa: BLE001
            import traceback
            self._log_msg("FAILED:\n" + traceback.format_exc())
            QMessageBox.critical(self, "Render Failed", str(exc))
        finally:
            self._rendering = False
            self._create_btn.setText("Create Video")
            self._close_btn.setEnabled(True)
            self._update_enabled_state()

    def _collect_params(self):
        target = self._target_edit.text().strip()
        if not target or not os.path.isdir(target):
            raise ValueError("Please select a valid target folder.")
        out = self._out_edit.text().strip()
        if not out:
            raise ValueError("Please choose an output .mp4 file.")
        if not out.lower().endswith(".mp4"):
            out += ".mp4"
        if not (self._modality() != "none" or self._map_enable.isChecked()):
            raise ValueError("Select at least one video modality or enable the map view.")

        crs = QgsCoordinateReferenceSystem(self._data_epsg_edit.text().strip())
        if not crs.isValid():
            raise ValueError("Data CRS is not a valid EPSG identifier.")

        return {
            "target": target,
            "out": out,
            "data_crs": crs,
            "source": "ortho" if self._src_ortho.isChecked() else "extracted",
            "ortho_kind": self._ortho_kind.currentData(),
            "video_suffixes": self._video_suffixes(),
            "overlay": self._overlay(),
            "map": self._map_enable.isChecked(),
            "map_flight": self._map_flight.isChecked(),
            "map_fov": self._map_fov.isChecked(),
            "map_dettrk": self._map_dettrk.isChecked(),
            "map_perp": self._map_perp.isChecked(),
            "map_merged_bg": self._map_merged_bg.isChecked(),
            "map_accumulate": self._map_accumulate.isChecked(),
            "map_follow": self._map_follow.isChecked(),
            "map_camera": self._map_camera.currentData(),
            "fps": self._fps_spin.value(),
            "height": self._height_spin.value(),
        }

    def _check_cancel(self):
        QApplication.processEvents()
        if self._cancel:
            raise _Cancelled()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_poses(self, target, suffix):
        """Return the ordered imagefile list for a camera, or [] if absent."""
        path = os.path.join(target, f"poses_{suffix}.json")
        if not os.path.exists(path):
            return []
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return [img.get("imagefile", "") for img in data.get("images", [])]
        except Exception as exc:  # noqa: BLE001
            self._log_msg(f"Warning: could not read {path}: {exc}")
            return []

    def _load_pixel_detections(self, target, suffix):
        """frame -> list of (x1, y1, x2, y2, class_id)."""
        path = os.path.join(target, f"detections_{suffix}", "detections.txt")
        out = {}
        if not os.path.exists(path):
            return out
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = line.split()
                if len(p) < 5:
                    continue
                frame = int(float(p[0]))
                out.setdefault(frame, []).append(
                    (float(p[1]), float(p[2]), float(p[3]), float(p[4]),
                     p[6] if len(p) > 6 else "0"))
        return out

    def _load_pixel_tracks(self, target, suffix):
        """frame -> list of (track_id, x1, y1, x2, y2)."""
        path = os.path.join(target, f"tracks_pixel_{suffix}", "tracks_pixel.csv")
        out = {}
        if not os.path.exists(path):
            return out
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = line.split(",")
                if len(p) < 6:
                    continue
                frame = int(float(p[0]))
                out.setdefault(frame, []).append(
                    (int(float(p[1])), float(p[2]), float(p[3]),
                     float(p[4]), float(p[5])))
        return out

    def _load_fov_polygons(self, target, suffix):
        """frame -> list of (x, y). Geo coordinates in the data CRS."""
        path = os.path.join(target, f"fov_{suffix}", "fov_polygons.txt")
        out = {}
        if not os.path.exists(path):
            return out
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = line.split()
                if len(p) < 2:
                    continue
                frame = int(p[0])
                n = int(p[1])
                pts = []
                for i in range(n):
                    idx = 2 + i * 3
                    if idx + 2 < len(p):
                        pts.append((float(p[idx]), float(p[idx + 1])))
                if len(pts) >= 3:
                    out[frame] = pts
        return out

    def _load_perpendicular(self, target, suffix):
        """frame -> list of ((cx, cy), (fx, fy), distance)."""
        path = os.path.join(target, f"flight_route_{suffix}", "perpendicular.json")
        out = {}
        if not os.path.exists(path):
            return out
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001
            return out
        for r in data.get("perpendiculas", []):
            frame = int(r.get("frame", -1))
            c = r.get("detection_center") or []
            foot = r.get("foot_point") or []
            if len(c) >= 2 and len(foot) >= 2:
                out.setdefault(frame, []).append(
                    ((c[0], c[1]), (foot[0], foot[1]), r.get("distance", 0.0)))
        return out

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, params):
        import numpy as np  # noqa: F401
        import cv2

        target = params["target"]
        panel_h = params["height"]

        # ---- Determine timeline (frame count) -----------------------------
        n_frames = 0
        poses = {}
        for suffix in params["video_suffixes"]:
            imgs = self._load_poses(target, suffix)
            poses[suffix] = imgs
            n_frames = max(n_frames, len(imgs))

        fov_polys = {}
        perp = {}
        if params["map"]:
            cam = params["map_camera"]
            fov_polys = self._load_fov_polygons(target, cam)
            if fov_polys:
                n_frames = max(n_frames, max(fov_polys.keys()) + 1)
            if not poses:
                imgs = self._load_poses(target, cam)
                if imgs:
                    n_frames = max(n_frames, len(imgs))
            if params["map_perp"]:
                perp = self._load_perpendicular(target, cam)

        if n_frames == 0:
            raise ValueError(
                "No frames found. Check that the target folder contains processed "
                "poses/FoV data for the selected options.")

        # ---- Overlay data --------------------------------------------------
        det_data = {}
        trk_data = {}
        if params["overlay"] == "detections":
            for s in params["video_suffixes"]:
                det_data[s] = self._load_pixel_detections(target, s)
        elif params["overlay"] == "tracks":
            for s in params["video_suffixes"]:
                trk_data[s] = self._load_pixel_tracks(target, s)

        # ---- ALFS static image (loaded once) -------------------------------
        alfs_cache = {}
        if params["source"] == "ortho" and params["ortho_kind"] == "alfs":
            for s in params["video_suffixes"]:
                img = self._read_geo_image(os.path.join(target, f"alfs_{s}", "alfs.tif"))
                alfs_cache[s] = img

        # ---- Map static setup ---------------------------------------------
        map_ctx = None
        if params["map"]:
            map_ctx = self._prepare_map_context(params, fov_polys)

        # ---- Determine composite geometry from the first renderable frame --
        self._log_msg(f"Rendering {n_frames} frames…")
        writer = None
        composite_size = None
        panel_widths = None

        try:
            for idx in range(n_frames):
                self._check_cancel()
                panels = []

                for s in params["video_suffixes"]:
                    panels.append(self._render_video_panel(
                        params, s, idx, poses.get(s, []),
                        det_data.get(s, {}), trk_data.get(s, {}),
                        alfs_cache.get(s), panel_h, cv2))

                if params["map"]:
                    panels.append(self._render_map_panel(
                        params, map_ctx, idx, fov_polys, perp, panel_h))

                # Establish fixed per-panel widths on the first frame.
                if panel_widths is None:
                    panel_widths = []
                    for pimg in panels:
                        if pimg is None:
                            panel_widths.append(panel_h)  # square placeholder
                        else:
                            h, w = pimg.shape[:2]
                            panel_widths.append(max(1, round(panel_h * w / h)))

                fitted = [self._fit(panels[i], panel_widths[i], panel_h, cv2)
                          for i in range(len(panels))]
                composite = fitted[0] if len(fitted) == 1 else cv2.hconcat(fitted)

                if writer is None:
                    composite_size = (composite.shape[1], composite.shape[0])
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(params["out"], fourcc, params["fps"],
                                             composite_size)
                    if not writer.isOpened():
                        raise RuntimeError(
                            "Could not open the output video for writing "
                            f"({params['out']}). Check the path and codec availability.")
                elif (composite.shape[1], composite.shape[0]) != composite_size:
                    composite = cv2.resize(composite, composite_size)

                writer.write(composite)

                if idx % 5 == 0 or idx == n_frames - 1:
                    self._progress.setValue(int((idx + 1) / n_frames * 100))
                    if idx % 25 == 0:
                        self._log_msg(f"  frame {idx + 1}/{n_frames}")
        finally:
            if writer is not None:
                writer.release()

        self._progress.setValue(100)
        return params["out"]

    # ---- Video panels -----------------------------------------------------

    def _render_video_panel(self, params, suffix, idx, imgs, dets, trks,
                            alfs_img, panel_h, cv2):
        import numpy as np

        img = None
        if params["source"] == "extracted":
            if idx < len(imgs) and imgs[idx]:
                path = os.path.join(params["target"], f"frames_{suffix}", imgs[idx])
                if os.path.exists(path):
                    img = cv2.imread(path)
        elif params["ortho_kind"] == "alfs":
            img = None if alfs_img is None else alfs_img.copy()
        else:  # per-frame geotiffs
            path = os.path.join(params["target"], f"geotiffs_{suffix}", f"{idx:08d}.tiff")
            img = self._read_geo_image(path)

        if img is None:
            img = np.zeros((panel_h, panel_h, 3), dtype=np.uint8)

        # Overlays only make sense in extracted-frame pixel space.
        if params["source"] == "extracted":
            if params["overlay"] == "detections":
                for (x1, y1, x2, y2, _cls) in dets.get(idx, []):
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 0, 255), 2)
            elif params["overlay"] == "tracks":
                for (tid, x1, y1, x2, y2) in trks.get(idx, []):
                    color = _track_color_bgr(tid)
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(img, f"#{tid}", (int(x1), max(0, int(y1) - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Camera label banner.
        label = "Thermal" if suffix == "t" else "RGB"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return img

    def _read_geo_image(self, path):
        """Read a (possibly geo-referenced) TIFF into a BGR uint8 array."""
        if not path or not os.path.exists(path):
            return None
        import numpy as np
        try:
            import rasterio
            with rasterio.open(path) as src:
                count = min(src.count, 3)
                bands = [src.read(i + 1) for i in range(count)]
                arr = np.dstack(bands) if count > 1 else bands[0]
        except Exception:  # noqa: BLE001 - fall back to OpenCV
            import cv2
            return cv2.imread(path)
        if arr.dtype != np.uint8:
            amax = float(arr.max()) if arr.size else 0.0
            arr = (arr / amax * 255.0).astype(np.uint8) if amax > 0 else arr.astype(np.uint8)
        if arr.ndim == 2:
            import cv2
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        # rasterio yields RGB order; OpenCV expects BGR.
        return arr[:, :, ::-1].copy()

    # ---- Map panel --------------------------------------------------------

    def _prepare_map_context(self, params, fov_polys):
        """Assemble static layers, the merged-FoV background and base extent."""
        ctx = {
            "data_crs": params["data_crs"],
            "static_layers": [],
            "merged_layer": None,
            "accum_geom": None,
        }
        # Live project layers reused as static base content.
        if params["map_dettrk"]:
            ctx["static_layers"] += self._collect_layers("BAMBI Wildlife Tracks")
            ctx["static_layers"] += self._collect_layers("BAMBI Frame Detections")
        if params["map_flight"]:
            ctx["static_layers"] += self._collect_layers("BAMBI Flight Route")
        ctx["static_layers"] += self._collect_layers("BAMBI Orthomosaic")

        # Merged FoV background: union of every frame's polygon, built once.
        if params["map_merged_bg"] and fov_polys:
            geoms = [self._polygon_geom(pts) for pts in fov_polys.values()]
            merged = QgsGeometry.unaryUnion([g for g in geoms if g])
            ctx["merged_layer"] = self._poly_layer(
                "merged_fov", params["data_crs"], merged,
                fill="180,180,180,90", stroke="120,120,120,160")

        # Base extent (data CRS) for the non-following case.
        pts = [p for poly in fov_polys.values() for p in poly]
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            rect = QgsRectangle(min(xs), min(ys), max(xs), max(ys))
            rect.scale(1.15)
            ctx["base_extent"] = rect
        else:
            ctx["base_extent"] = None
        return ctx

    def _render_map_panel(self, params, ctx, idx, fov_polys, perp, panel_h):
        import numpy as np

        data_crs = params["data_crs"]
        dyn_layers = []

        # Perpendicular distance lines for this frame (drawn on top).
        if params["map_perp"] and idx in perp:
            lines = []
            for (c, foot, _d) in perp[idx]:
                lines.append(QgsGeometry.fromPolylineXY(
                    [QgsPointXY(c[0], c[1]), QgsPointXY(foot[0], foot[1])]))
            if lines:
                dyn_layers.append(self._line_layer(
                    "perp", data_crs, lines, color="230,80,20,255", width=0.6))

        # Current FoV outline.
        cur_pts = fov_polys.get(idx)
        if params["map_fov"] and cur_pts:
            dyn_layers.append(self._poly_layer(
                "cur_fov", data_crs, self._polygon_geom(cur_pts),
                fill="0,0,0,0", stroke="255,215,0,255", stroke_width=0.8))

        # Accumulated FoV (monitored area so far).
        if params["map_accumulate"] and cur_pts:
            g = self._polygon_geom(cur_pts)
            if ctx["accum_geom"] is None:
                ctx["accum_geom"] = g
            else:
                ctx["accum_geom"] = ctx["accum_geom"].combine(g)
            dyn_layers.append(self._poly_layer(
                "accum_fov", data_crs, QgsGeometry(ctx["accum_geom"]),
                fill="67,160,71,70", stroke="67,160,71,150"))

        layers = list(dyn_layers)
        if ctx["merged_layer"] is not None:
            layers.append(ctx["merged_layer"])
        layers += ctx["static_layers"]

        # Extent: follow the current FoV or use the fixed base extent.
        extent = None
        if params["map_follow"] and cur_pts:
            xs = [p[0] for p in cur_pts]
            ys = [p[1] for p in cur_pts]
            extent = QgsRectangle(min(xs), min(ys), max(xs), max(ys))
            extent.scale(1.6)
        elif ctx["base_extent"] is not None:
            extent = QgsRectangle(ctx["base_extent"])

        return self._render_map_image(layers, extent, data_crs, panel_h, panel_h)

    def _render_map_image(self, layers, extent_data_crs, data_crs, width, height):
        import numpy as np

        project = QgsProject.instance()
        dest_crs = project.crs() if project.crs().isValid() else data_crs

        ms = QgsMapSettings()
        ms.setDestinationCrs(dest_crs)
        ms.setLayers([l for l in layers if l is not None])
        ms.setOutputSize(QSize(width, height))
        ms.setBackgroundColor(QColor(30, 30, 30))

        if extent_data_crs is not None:
            extent = extent_data_crs
            if data_crs != dest_crs:
                xform = QgsCoordinateTransform(data_crs, dest_crs, project)
                try:
                    extent = xform.transformBoundingBox(extent_data_crs)
                except Exception:  # noqa: BLE001
                    extent = extent_data_crs
            ms.setExtent(extent)
        elif ms.layers():
            ms.setExtent(ms.fullExtent())

        job = QgsMapRendererParallelJob(ms)
        job.start()
        job.waitForFinished()
        qimg = job.renderedImage().convertToFormat(QImage.Format_RGB888)

        w, h = qimg.width(), qimg.height()
        ptr = qimg.constBits()
        ptr.setsize(qimg.byteCount() if hasattr(qimg, "byteCount") else qimg.sizeInBytes())
        bpl = qimg.bytesPerLine()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))[:, : w * 3]
        arr = arr.reshape((h, w, 3))
        return arr[:, :, ::-1].copy()  # RGB -> BGR

    # ---- QGIS layer builders ---------------------------------------------

    def _collect_layers(self, prefix):
        """Live project layers whose own name or containing group starts with *prefix*."""
        found = []
        project = QgsProject.instance()
        root = project.layerTreeRoot()
        for grp in root.findGroups():
            if grp.name().startswith(prefix):
                for node in grp.findLayers():
                    lyr = node.layer()
                    if lyr is not None:
                        found.append(lyr)
        for lyr in project.mapLayers().values():
            if lyr.name().startswith(prefix) and lyr not in found:
                found.append(lyr)
        return found

    @staticmethod
    def _polygon_geom(pts):
        ring = [QgsPointXY(x, y) for (x, y) in pts]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        return QgsGeometry.fromPolygonXY([ring])

    def _poly_layer(self, name, crs, geom, fill, stroke, stroke_width=0.4):
        layer = QgsVectorLayer(f"Polygon?crs={crs.authid()}", name, "memory")
        prov = layer.dataProvider()
        prov.addAttributes([QgsField("id", QVariant.Int)])
        layer.updateFields()
        feat = QgsFeature()
        feat.setGeometry(geom)
        feat.setAttributes([0])
        prov.addFeatures([feat])
        layer.updateExtents()
        sym = QgsFillSymbol.createSimple({
            "color": fill, "outline_color": stroke,
            "outline_width": str(stroke_width)})
        layer.setRenderer(QgsSingleSymbolRenderer(sym))
        return layer

    def _line_layer(self, name, crs, geoms, color, width=0.5):
        layer = QgsVectorLayer(f"LineString?crs={crs.authid()}", name, "memory")
        prov = layer.dataProvider()
        prov.addAttributes([QgsField("id", QVariant.Int)])
        layer.updateFields()
        feats = []
        for i, g in enumerate(geoms):
            f = QgsFeature()
            f.setGeometry(g)
            f.setAttributes([i])
            feats.append(f)
        prov.addFeatures(feats)
        layer.updateExtents()
        sym = QgsLineSymbol.createSimple({"color": color, "width": str(width)})
        layer.setRenderer(QgsSingleSymbolRenderer(sym))
        return layer

    # ---- Compositing helpers ---------------------------------------------

    @staticmethod
    def _fit(img, width, height, cv2):
        """Resize *img* to fit within width x height (letterboxed), exact size."""
        import numpy as np
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if img is None or img.size == 0:
            return canvas
        h, w = img.shape[:2]
        scale = min(width / w, height / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (nw, nh))
        x0 = (width - nw) // 2
        y0 = (height - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._rendering:
            self._cancel = True
        super().closeEvent(event)
