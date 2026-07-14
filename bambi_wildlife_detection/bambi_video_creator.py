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

import os

from qgis.PyQt.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QRadioButton, QScrollArea, QSpinBox,
    QTextEdit, QVBoxLayout, QWidget,
)
from qgis.core import (
    QgsCoordinateReferenceSystem, QgsGeometry, QgsPointXY, QgsProject,
)


from .gui_utils import fit_to_screen
from .core.tile_math import latlon_to_tile, tile_to_latlon
# Track colours and all data loaders moved to core.video_export;
# re-exported under their old names for compatibility.
from .core.video_export import (  # noqa: F401 — re-exported API
    TRACK_COLORS as _TRACK_COLORS,
    id_to_color as _id_to_color,
    track_color_bgr as _track_color_bgr,
)
from .core import video_export

# Fixed colour for the drone marker / look-point trail (BGR).
_DRONE_COLOR = (0, 215, 255)


class _MapTileProvider:
    """Downloads and stitches web-map tiles for a UTM extent (satellite/OSM
    background). Adapted from TRexConnector's visualiser; requires requests +
    pyproj, which are probed lazily."""

    ESRI_SATELLITE = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                      "World_Imagery/MapServer/tile/{z}/{y}/{x}")
    OPENSTREETMAP = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

    def __init__(self, tile_url, utm_epsg, cache_dir=None):
        import math  # noqa: F401
        from pyproj import CRS, Transformer
        self.tile_url = tile_url
        self.cache_dir = cache_dir
        self.transformer = Transformer.from_crs(
            CRS.from_epsg(utm_epsg), CRS.from_epsg(4326), always_xy=True)
        self.headers = {"User-Agent": "BambiVideoCreator/1.0"}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _utm_to_latlon(self, x, y):
        lon, lat = self.transformer.transform(x, y)
        return lat, lon

    @staticmethod
    def _latlon_to_tile(lat, lon, zoom):
        return latlon_to_tile(lat, lon, zoom)

    @staticmethod
    def _tile_to_latlon(x, y, zoom):
        return tile_to_latlon(x, y, zoom)

    def _download_tile(self, x, y, zoom):
        import cv2
        import hashlib
        import numpy as np
        import requests
        cache_path = None
        if self.cache_dir:
            # MD5 only fingerprints the tile URL for cache file names — not security
            h = hashlib.md5(self.tile_url.encode(), usedforsecurity=False).hexdigest()[:8]
            cache_path = os.path.join(self.cache_dir, f"{h}_{zoom}_{x}_{y}.png")
            if os.path.exists(cache_path):
                return cv2.imread(cache_path)
        url = self.tile_url.format(z=zoom, x=x, y=y)
        try:
            resp = requests.get(url, headers=self.headers, timeout=5)
            if resp.status_code == 200:
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if cache_path and img is not None:
                    cv2.imwrite(cache_path, img)
                return img
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return None

    def render(self, extent, width, height, margin):
        """Return a (height, width, 3) BGR image with the tiled background placed
        inside *margin*, or None on failure."""
        import cv2
        import numpy as np
        min_x, max_x, min_y, max_y = extent
        min_lat, min_lon = self._utm_to_latlon(min_x, min_y)
        max_lat, max_lon = self._utm_to_latlon(max_x, max_y)
        if min_lat > max_lat:
            min_lat, max_lat = max_lat, min_lat
        if min_lon > max_lon:
            min_lon, max_lon = max_lon, min_lon

        zoom = 16
        for z in range(19, 11, -1):
            x1, y1 = self._latlon_to_tile(max_lat, min_lon, z)
            x2, y2 = self._latlon_to_tile(min_lat, max_lon, z)
            if (abs(x2 - x1) + 1) * 256 > width and (abs(y2 - y1) + 1) * 256 > height:
                zoom = z
                break

        tx1, ty1 = self._latlon_to_tile(max_lat, min_lon, zoom)
        tx2, ty2 = self._latlon_to_tile(min_lat, max_lon, zoom)
        stitch_w = (tx2 - tx1 + 1) * 256
        stitch_h = (ty2 - ty1 + 1) * 256
        if stitch_w <= 0 or stitch_h <= 0 or stitch_w * stitch_h > 60_000_000:
            return None
        stitch = np.zeros((stitch_h, stitch_w, 3), dtype=np.uint8)
        for y in range(ty1, ty2 + 1):
            for x in range(tx1, tx2 + 1):
                t = self._download_tile(x, y, zoom)
                if t is not None:
                    py, px = (y - ty1) * 256, (x - tx1) * 256
                    stitch[py:py + 256, px:px + 256] = t

        top_lat, left_lon = self._tile_to_latlon(tx1, ty1, zoom)
        btm_lat, rgt_lon = self._tile_to_latlon(tx2 + 1, ty2 + 1, zoom)

        def ll2px(lat, lon):
            px = (lon - left_lon) / (rgt_lon - left_lon) * stitch_w
            py = (top_lat - lat) / (top_lat - btm_lat) * stitch_h
            return int(px), int(py)

        px1, py1 = ll2px(max_lat, min_lon)
        px2, py2 = ll2px(min_lat, max_lon)
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(stitch_w, px2), min(stitch_h, py2)
        if px2 <= px1 or py2 <= py1:
            return None
        crop = stitch[py1:py2, px1:px2]

        iw, ih = width - 2 * margin, height - 2 * margin
        final = np.zeros((height, width, 3), dtype=np.uint8)
        try:
            resized = cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_AREA)
            final[margin:margin + ih, margin:margin + iw] = resized
        except Exception:  # noqa: BLE001
            return None
        return final


class _Cancelled(Exception):
    """Raised internally when the user cancels the render."""


class VideoCreatorDialog(QDialog):
    """Non-modal dialog that composes an MP4 from processed BAMBI results."""

    def __init__(self, iface, dock_widget=None, parent=None):
        super().__init__(parent)
        self.iface = iface
        self._dock_widget = dock_widget
        self._cancel = False
        self._rendering = False
        self.setWindowTitle("BAMBI Video Creator")
        self._build_ui()
        fit_to_screen(self, 640, 820)
        self.apply_dock_defaults()
        self._update_enabled_state()

    def apply_dock_defaults(self):
        """Initialise the target folder and data CRS from the dock widget's
        input fields, without clobbering values the user has already entered."""
        dock = self._dock_widget
        if not self._target_edit.text().strip():
            folder = ""
            if dock is not None and hasattr(dock, "target_folder_edit"):
                folder = dock.target_folder_edit.text().strip()
            if folder:
                self._target_edit.setText(folder)
                if not self._out_edit.text().strip():
                    self._out_edit.setText(os.path.join(folder, "bambi_video.mp4"))

        if not self._data_epsg_edit.text().strip():
            crs_text = ""
            if dock is not None and hasattr(dock, "target_crs_edit"):
                crs_text = dock.target_crs_edit.text().strip()
            if not crs_text:
                crs = QgsProject.instance().crs()
                crs_text = crs.authid() if crs.isValid() else "EPSG:32633"
            self._data_epsg_edit.setText(crs_text)

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
        sv.addWidget(self._build_info_group())
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
        self._data_epsg_edit.setToolTip(
            "CRS of the geo-referenced products (FoV / perpendicular). Used to place "
            "map overlays. Initialised from the BAMBI panel's target CRS.")
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
        self._map_det = QCheckBox("Detections")
        self._map_trk = QCheckBox("Tracks")
        self._map_perp = QCheckBox("Perpendicular distances (detections)")
        self._map_merged_bg = QCheckBox("Merged FoV as background")
        self._map_accumulate = QCheckBox("Accumulate FoV frame-by-frame (monitored area)")
        self._map_satellite = QCheckBox("Satellite / OSM background (needs internet)")
        self._map_flight.setChecked(True)
        self._map_fov.setChecked(True)
        for c in (self._map_flight, self._map_fov, self._map_det, self._map_trk,
                  self._map_perp, self._map_merged_bg, self._map_accumulate,
                  self._map_satellite):
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

    def _build_info_group(self):
        grp = QGroupBox("5. Info Panel (bottom bar)")
        vl = QVBoxLayout(grp)
        self._info_enable = QCheckBox("Include info panel")
        self._info_enable.toggled.connect(self._update_enabled_state)
        vl.addWidget(self._info_enable)

        self._info_options = QWidget()
        ol = QVBoxLayout(self._info_options)
        ol.setContentsMargins(18, 0, 0, 0)
        self._info_frame = QCheckBox("Current frame number")
        self._info_dets = QCheckBox("Number of detections (current frame)")
        self._info_tracks = QCheckBox("Number of tracks (current frame)")
        self._info_area = QCheckBox("Monitored area (observed / total)")
        self._info_frame.setChecked(True)
        for c in (self._info_frame, self._info_dets, self._info_tracks, self._info_area):
            ol.addWidget(c)
        vl.addWidget(self._info_options)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Statistics source camera:"))
        self._info_camera = QComboBox()
        self._info_camera.addItem("Thermal", "t")
        self._info_camera.addItem("RGB", "w")
        cam_row.addWidget(self._info_camera)
        cam_row.addStretch()
        ol.addLayout(cam_row)
        return grp

    def _build_output_group(self):
        grp = QGroupBox("6. Output")
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
        self._fps_spin.setValue(30)
        fl.addRow("Frames per second:", self._fps_spin)

        self._height_spin = QSpinBox()
        self._height_spin.setRange(240, 2160)
        self._height_spin.setSingleStep(120)
        self._height_spin.setValue(720)
        fl.addRow("Panel height (px):", self._height_spin)

        self._all_frames_check = QCheckBox("Render all frames")
        self._all_frames_check.setChecked(True)
        self._all_frames_check.toggled.connect(self._update_enabled_state)
        fl.addRow("", self._all_frames_check)

        range_row = QHBoxLayout()
        self._start_frame_spin = QSpinBox()
        self._start_frame_spin.setRange(0, 999999)
        self._start_frame_spin.setValue(0)
        self._end_frame_spin = QSpinBox()
        self._end_frame_spin.setRange(0, 999999)
        self._end_frame_spin.setValue(999999)
        self._end_frame_spin.setToolTip("Inclusive; clamped to the last available frame.")
        range_row.addWidget(QLabel("Start:"))
        range_row.addWidget(self._start_frame_spin)
        range_row.addWidget(QLabel("End:"))
        range_row.addWidget(self._end_frame_spin)
        range_row.addStretch()
        self._frame_range_widget = QWidget()
        self._frame_range_widget.setLayout(range_row)
        fl.addRow("Frame range:", self._frame_range_widget)
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
        self._info_options.setEnabled(self._info_enable.isChecked())
        self._frame_range_widget.setEnabled(not self._all_frames_check.isChecked())

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

        # Warn up-front about every selected item whose data is missing, so the
        # user can fix the selection before the (possibly long) render instead
        # of discovering the gaps in the finished video.
        warns = self._availability_warnings(params)
        if warns:
            msg = ("The following selected options have no data in the target "
                   "folder and will be skipped or shown empty:\n\n  • "
                   + "\n  • ".join(warns)  # noqa: W503, W504
                   + "\n\nCreate the video anyway?")  # noqa: W503, W504
            if QMessageBox.warning(
                    self, "Missing Data", msg,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes) != QMessageBox.StandardButton.Yes:
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
            "map_det": self._map_det.isChecked(),
            "map_trk": self._map_trk.isChecked(),
            "map_perp": self._map_perp.isChecked(),
            "map_merged_bg": self._map_merged_bg.isChecked(),
            "map_accumulate": self._map_accumulate.isChecked(),
            "map_satellite": self._map_satellite.isChecked(),
            "map_camera": self._map_camera.currentData(),
            "info": self._info_enable.isChecked(),
            "info_frame": self._info_frame.isChecked(),
            "info_dets": self._info_dets.isChecked(),
            "info_tracks": self._info_tracks.isChecked(),
            "info_area": self._info_area.isChecked(),
            "info_camera": self._info_camera.currentData(),
            "fps": self._fps_spin.value(),
            "height": self._height_spin.value(),
            "all_frames": self._all_frames_check.isChecked(),
            "start_frame": self._start_frame_spin.value(),
            "end_frame": self._end_frame_spin.value(),
        }

    def _pixel_tracks_available(self, target, suffix):
        return video_export.pixel_tracks_available(target, suffix)

    def _availability_warnings(self, params):
        """Warnings for selected options with missing data (core.video_export)."""
        return video_export.availability_warnings(params)

    def _check_cancel(self):
        QApplication.processEvents()
        if self._cancel:
            raise _Cancelled()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_poses(self, target, suffix):
        return video_export.load_poses(target, suffix, log_fn=self._log_msg)

    def _load_pixel_detections(self, target, suffix):
        return video_export.load_pixel_detections(target, suffix)

    def _load_detection_rows(self, target, suffix):
        return video_export.load_detection_rows(target, suffix)

    def _load_track_id_rows(self, target, suffix):
        return video_export.load_track_id_rows(target, suffix)

    def _load_georef_rows(self, target, suffix):
        return video_export.load_georef_rows(target, suffix)

    @staticmethod
    def _parse_pixel_tracks_csv(path, interp_col):
        return video_export.parse_pixel_tracks_csv(path, interp_col)

    @staticmethod
    def _coord_key(coords):
        return video_export.coord_key(coords)

    def _load_pixel_tracks(self, target, suffix):
        return video_export.load_pixel_tracks(target, suffix, log_fn=self._log_msg)

    def _pair_tracks_via_geo(self, target, suffix):
        return video_export.pair_tracks_via_geo(target, suffix)

    def _pair_tracks_by_line_index(self, target, suffix):
        return video_export.pair_tracks_by_line_index(
            target, suffix, log_fn=self._log_msg)

    def _load_fov_polygons(self, target, suffix):
        return video_export.load_fov_polygons(target, suffix)

    def _load_perpendicular(self, target, suffix):
        return video_export.load_perpendicular(target, suffix)
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

        # ---- Frame range (subsequence) ------------------------------------
        if params["all_frames"]:
            start_idx, end_idx = 0, n_frames - 1
        else:
            start_idx = max(0, params["start_frame"])
            end_idx = min(n_frames - 1, params["end_frame"])
            if start_idx > end_idx:
                raise ValueError(
                    f"Empty frame range: start {start_idx} > last available "
                    f"frame {end_idx} (total {n_frames}).")

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

        # ---- Info-panel statistics sources --------------------------------
        info_dets = {}
        info_trks = {}
        info_area_fov = {}
        info_bar_h = 0
        if params["info"]:
            cam = params["info_camera"]
            if params["info_dets"]:
                info_dets = self._load_pixel_detections(target, cam)
            if params["info_tracks"]:
                info_trks = self._load_pixel_tracks(target, cam)
            if params["info_area"]:
                # Reuse already-loaded FoV if it is the same camera.
                if fov_polys and params.get("map_camera") == cam:
                    info_area_fov = fov_polys
                else:
                    info_area_fov = self._load_fov_polygons(target, cam)
            info_bar_h = max(48, panel_h // 8)
        # Totals across the whole dataset (the "n" in "x / n").
        info_dets_total = sum(len(v) for v in info_dets.values())
        info_trks_total = len({r[0] for v in info_trks.values() for r in v})
        info_accum_geom = None
        # Total monitored area = union of every FoV polygon over the whole flight.
        info_area_total = 0.0
        if info_area_fov:
            geoms = [self._polygon_geom(pts) for pts in info_area_fov.values() if pts]
            total_geom = QgsGeometry.unaryUnion(geoms) if geoms else None
            if total_geom is not None and not total_geom.isEmpty():
                info_area_total = total_geom.area()

        # ---- Determine composite geometry from the first renderable frame --
        total_out = end_idx - start_idx + 1
        self._log_msg(
            f"Rendering frames {start_idx}–{end_idx} ({total_out} of {n_frames})…")
        writer = None
        composite_size = None
        panel_widths = None

        try:
            for out_i, idx in enumerate(range(start_idx, end_idx + 1)):
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

                # ---- Bottom info panel ------------------------------------
                if params["info"]:
                    if params["info_area"]:
                        pts = info_area_fov.get(idx)
                        if pts:
                            g = self._polygon_geom(pts)
                            info_accum_geom = (g if info_accum_geom is None
                                               else info_accum_geom.combine(g))
                    area_m2 = info_accum_geom.area() if info_accum_geom is not None else 0.0
                    bar = self._render_info_panel(
                        params, composite.shape[1], info_bar_h, idx, n_frames,
                        len(info_dets.get(idx, [])), info_dets_total,
                        len({r[0] for r in info_trks.get(idx, [])}), info_trks_total,
                        area_m2, info_area_total, cv2)
                    composite = cv2.vconcat([composite, bar])

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

                if out_i % 5 == 0 or idx == end_idx:
                    self._progress.setValue(int((out_i + 1) / total_out * 100))
                    if out_i % 25 == 0:
                        self._log_msg(f"  frame {idx} ({out_i + 1}/{total_out})")
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

    # ---- Map panel (OpenCV renderer) --------------------------------------

    def _prepare_map_context(self, params, fov_polys):
        """Load geo data, compute a fixed extent + canvas mapping, and prepare
        the (optional) satellite background and accumulation state. All drawing
        happens with OpenCV in world->canvas pixel space (see the TRex tool
        ``visualize_trex_video_and_map.py``)."""
        import numpy as np

        target = params["target"]
        cam = params["map_camera"]
        size = 900
        margin = 60

        flight_pos = self._load_camera_positions(target, cam) if params["map_flight"] else {}
        geo_tracks = self._load_geo_tracks(target, cam) if params["map_trk"] else {}
        geo_dets = self._load_geo_detections(target, cam) if params["map_det"] else {}

        # Global extent over everything to be drawn (whole flight, so the map does
        # not jump while the path is revealed frame by frame).
        xs, ys = [], []
        for poly in fov_polys.values():
            for (x, y) in poly:
                xs.append(x)
                ys.append(y)
        for (x, y) in flight_pos.values():
            xs.append(x)
            ys.append(y)
        for group in (geo_tracks, geo_dets):
            for dets in group.values():
                for d in dets:
                    xs += [d["x1"], d["x2"]]
                    ys += [d["y1"], d["y2"]]

        if not xs:
            return {"empty": True, "size": size}

        extent = (min(xs), max(xs), min(ys), max(ys))
        extent = self._pad_extent_to_aspect(extent, size, size, margin)
        # Grow slightly so content is not flush against the axes.
        min_x, max_x, min_y, max_y = extent
        pad = 0.06 * max(max_x - min_x, max_y - min_y)
        extent = (min_x - pad, max_x + pad, min_y - pad, max_y + pad)
        cfg = self._make_canvas_cfg(extent, size, size, margin)

        # Optional satellite / OSM background (built once for the fixed extent).
        background = None
        if params["map_satellite"]:
            try:
                epsg = params["data_crs"].postgisSrid()
                url = _MapTileProvider.ESRI_SATELLITE
                prov = _MapTileProvider(url, epsg,
                                        cache_dir=os.path.join(target, "_map_tiles"))
                bg = prov.render(extent, size, size, margin)
                if bg is not None:
                    background = (bg * 0.55).astype(np.uint8)
                    self._log_msg("Satellite background downloaded.")
                else:
                    self._log_msg("Satellite background unavailable (offline?); using dark canvas.")
            except Exception as exc:  # noqa: BLE001
                self._log_msg(f"Satellite background skipped: {exc}")

        # Merged FoV background: rasterise every frame's polygon once.
        merged_mask = None
        if params["map_merged_bg"] and fov_polys:
            merged_mask = np.zeros((size, size), dtype=np.uint8)
            for poly in fov_polys.values():
                pix = np.array([self._world_to_canvas(x, y, cfg) for (x, y) in poly],
                               dtype=np.int32)
                cv2_fillable = pix.reshape((-1, 1, 2))
                import cv2
                cv2.fillPoly(merged_mask, [cv2_fillable], 1)

        return {
            "empty": False,
            "size": size,
            "margin": margin,
            "cfg": cfg,
            "extent": extent,
            "background": background,
            "flight_pos": flight_pos,
            "flight_trail": [],
            "geo_tracks": geo_tracks,
            "geo_dets": geo_dets,
            "merged_mask": merged_mask,
            "accum_mask": np.zeros((size, size), dtype=np.uint8),
            "track_history": {},
            "epsg": params["data_crs"].postgisSrid(),
        }

    def _render_map_panel(self, params, ctx, idx, fov_polys, perp, panel_h):
        import cv2
        import numpy as np

        size = ctx["size"]
        if ctx.get("empty"):
            img = np.full((size, size, 3), 30, dtype=np.uint8)
            cv2.putText(img, "No geo data for map", (30, size // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
            return img

        cfg = ctx["cfg"]
        img = (ctx["background"].copy() if ctx["background"] is not None
               else np.full((size, size, 3), 30, dtype=np.uint8))

        def to_px(x, y):
            return self._world_to_canvas(x, y, cfg)

        # --- Merged FoV background (whole monitored area) ---
        if ctx["merged_mask"] is not None:
            m = ctx["merged_mask"] > 0
            img[m] = (img[m] * 0.6 + np.array((150, 150, 150)) * 0.4).astype(np.uint8)

        # --- Accumulated FoV so far ---
        cur_pts = fov_polys.get(idx)
        if params["map_accumulate"] and cur_pts:
            pix = np.array([to_px(x, y) for (x, y) in cur_pts], dtype=np.int32)
            cv2.fillPoly(ctx["accum_mask"], [pix.reshape((-1, 1, 2))], 1)
        if params["map_accumulate"]:
            m = ctx["accum_mask"] > 0
            img[m] = (img[m] * 0.6 + np.array((67, 160, 71)) * 0.4).astype(np.uint8)

        # --- Flight path (revealed live, frame by frame from the poses) ---
        if params["map_flight"]:
            pos = ctx["flight_pos"].get(idx)
            if pos is not None:
                ctx["flight_trail"].append(to_px(pos[0], pos[1]))
            if len(ctx["flight_trail"]) > 1:
                cv2.polylines(img, [np.array(ctx["flight_trail"], dtype=np.int32)],
                              False, (255, 170, 60), 2, cv2.LINE_AA)

        # --- Current FoV polygon (the mask-aware calculated FoV) ---
        if params["map_fov"] and cur_pts:
            pix = np.array([to_px(x, y) for (x, y) in cur_pts], dtype=np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pix.reshape((-1, 1, 2))], (0, 200, 255))
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
            cv2.polylines(img, [pix], True, (0, 215, 255), 2, cv2.LINE_AA)

        # --- Geo detections (current frame only, single colour, no id/trail) ---
        if params["map_det"] and ctx["geo_dets"]:
            for d in ctx["geo_dets"].get(idx, []):
                p1 = to_px(d["x1"], d["y1"])
                p2 = to_px(d["x2"], d["y2"])
                cv2.rectangle(img, p1, p2, (60, 60, 235), 2)

        # --- Geo tracks (per-id colour + id label + trajectory trail) ---
        if params["map_trk"] and ctx["geo_tracks"]:
            hist = ctx["track_history"]
            for d in ctx["geo_tracks"].get(idx, []):
                tid = d["tid"]
                color = _id_to_color(tid)
                p1 = to_px(d["x1"], d["y1"])
                p2 = to_px(d["x2"], d["y2"])
                cv2.rectangle(img, p1, p2, color, 2)
                cv2.putText(img, f"ID {tid}", (p1[0], max(0, min(p1[1], p2[1]) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                cx = (d["x1"] + d["x2"]) / 2.0
                cy = (d["y1"] + d["y2"]) / 2.0
                hist.setdefault(tid, []).append(to_px(cx, cy))
            for tid, trail in hist.items():
                if len(trail) > 1:
                    cv2.polylines(img, [np.array(trail, dtype=np.int32)], False,
                                  _id_to_color(tid), 1, cv2.LINE_AA)

        # --- Perpendicular distance lines for this frame ---
        if params["map_perp"] and idx in perp:
            for (c, foot, dist) in perp[idx]:
                pc = to_px(c[0], c[1])
                pf = to_px(foot[0], foot[1])
                cv2.line(img, pc, pf, (40, 120, 240), 1, cv2.LINE_AA)
                cv2.circle(img, pf, 3, (40, 120, 240), -1)
                mid = ((pc[0] + pf[0]) // 2, (pc[1] + pf[1]) // 2)
                cv2.putText(img, f"{dist:.0f}m", mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 120, 240), 1, cv2.LINE_AA)

        # --- Drone position (actual camera position; FoV centroid as fallback) ---
        drone_xy = ctx["flight_pos"].get(idx) if params["map_flight"] else None
        if drone_xy is None and cur_pts:
            drone_xy = (sum(p[0] for p in cur_pts) / len(cur_pts),
                        sum(p[1] for p in cur_pts) / len(cur_pts))
        if drone_xy is not None:
            px, py = to_px(drone_xy[0], drone_xy[1])
            cv2.circle(img, (px, py), 5, _DRONE_COLOR, -1)
            cv2.circle(img, (px, py), 5, (0, 0, 0), 1)

        self._draw_axes(img, cfg)
        cv2.putText(img, f"Map (EPSG:{ctx['epsg']})", (10, size - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    # ---- Map geometry / geo data helpers ---------------------------------

    @staticmethod
    def _pad_extent_to_aspect(extent, width, height, margin):
        return video_export.pad_extent_to_aspect(extent, width, height, margin)

    @staticmethod
    def _make_canvas_cfg(extent, width, height, margin):
        return video_export.make_canvas_cfg(extent, width, height, margin)

    @staticmethod
    def _world_to_canvas(x, y, cfg):
        return video_export.world_to_canvas(x, y, cfg)

    def _draw_axes(self, img, cfg, num_ticks=4):
        import cv2
        axis_color = (200, 200, 200)
        bl = self._world_to_canvas(cfg["min_x"], cfg["min_y"], cfg)
        br = self._world_to_canvas(cfg["max_x"], cfg["min_y"], cfg)
        tl = self._world_to_canvas(cfg["min_x"], cfg["max_y"], cfg)
        cv2.line(img, bl, br, axis_color, 1)
        cv2.line(img, bl, tl, axis_color, 1)
        for i in range(num_ticks):
            t = i / (num_ticks - 1)
            vx = cfg["min_x"] + t * (cfg["max_x"] - cfg["min_x"])
            px, py = self._world_to_canvas(vx, cfg["min_y"], cfg)
            cv2.line(img, (px, py), (px, py + 5), axis_color, 1)
            cv2.putText(img, f"{int(vx)}", (px - 24, py + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            vy = cfg["min_y"] + t * (cfg["max_y"] - cfg["min_y"])
            px, py = self._world_to_canvas(cfg["min_x"], vy, cfg)
            cv2.line(img, (px - 5, py), (px, py), axis_color, 1)
            cv2.putText(img, f"{int(vy)}", (px - 58, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    def _load_camera_positions(self, target, suffix):
        return video_export.load_camera_positions(target, suffix)

    def _load_geo_tracks(self, target, suffix):
        return video_export.load_geo_tracks(target, suffix)

    def _load_geo_detections(self, target, suffix):
        return video_export.load_geo_detections(target, suffix)

    @staticmethod
    def _polygon_geom(pts):
        """QgsGeometry polygon (used only for accumulated-area measurement)."""
        ring = [QgsPointXY(x, y) for (x, y) in pts]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        return QgsGeometry.fromPolygonXY([ring])

    # ---- Compositing helpers ---------------------------------------------

    @staticmethod
    def _fmt_area(area_m2):
        return video_export.fmt_area(area_m2)

    @staticmethod
    def _fmt_area_ratio(area_m2, total_m2):
        return video_export.fmt_area_ratio(area_m2, total_m2)

    def _render_info_panel(self, params, width, bar_h, idx, n_frames,
                           n_dets, total_dets, n_tracks, total_tracks,
                           area_m2, total_area, cv2):
        """Render the full-width bottom statistics bar."""
        import numpy as np

        bar = np.zeros((bar_h, width, 3), dtype=np.uint8)
        bar[:] = (40, 40, 40)

        parts = []
        if params["info_frame"]:
            parts.append(f"Frame: {idx + 1}/{n_frames}")
        if params["info_dets"]:
            parts.append(f"Detections: {n_dets} / {total_dets}")
        if params["info_tracks"]:
            parts.append(f"Tracks: {n_tracks} / {total_tracks}")
        if params["info_area"]:
            parts.append(f"Monitored: {self._fmt_area_ratio(area_m2, total_area)}")
        if not parts:
            return bar

        text = "     ".join(parts)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = bar_h / 42.0
        thickness = max(1, int(round(scale * 1.5)))
        # Shrink the scale until the line fits within the available width.
        while scale > 0.3:
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            if tw <= width - 24:
                break
            scale -= 0.05
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = 12
        y = (bar_h + th) // 2
        cv2.putText(bar, text, (x, y), font, scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)
        return bar

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
