# -*- coding: utf-8 -*-
"""3D Flight Viewer: the active flight in 3D, replayed frame by frame.

A separate window with the DEM mesh, the flight route and the corrected
camera poses. The replay drapes each frame onto the DEM (projective texture
mapping, the same mapping ALFS uses), optionally with a trail of earlier
frames, the projected detections or tracks, and a live ALFS integral. The
view either follows the drone camera or can be moved freely while the replay
runs. Further tabs adjust the correction factors and build a 3D
reconstruction from the frames.

The headless parts live in :mod:`core.flight_viewer`,
:mod:`core.flight_viewer_geometry` and :mod:`core.view_camera`; the OpenGL
part in :mod:`viewer3d`.
"""

import math
import os
from typing import Dict, List, Optional

import numpy as np
from qgis.PyQt.QtCore import QSettings, Qt, QThread, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QImage, QKeySequence, QPixmap
from qgis.PyQt.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QSplitter, QStyle, QTabWidget, QToolButton,
    QVBoxLayout, QWidget,
)

try:
    from qgis.PyQt.QtWidgets import QShortcut
except ImportError:  # Qt6
    from qgis.PyQt.QtGui import QShortcut

from .core import flight_viewer as fv
from .core import flight_viewer_geometry as geo

SETTINGS_GROUP = "bambi/flight_viewer"

ANNOTATION_NONE = "none"
ANNOTATION_BOXES = "boxes"
ANNOTATION_TRACKS = "tracks"

DRAPE_FRAME = "frame"      # the current frame alone ("pinhole")
DRAPE_ALFS = "alfs"
DRAPE_NONE = "none"

SPEEDS = [("0.25×", 0.25), ("0.5×", 0.5), ("1× (real time)", 1.0),
          ("2×", 2.0), ("4×", 4.0), ("8×", 8.0), ("Every frame, fast", 0.0)]


def _settings():
    try:
        from qgis.core import QgsSettings
        return QgsSettings()
    except Exception:  # noqa: BLE001 - outside QGIS
        return QSettings()


class _LoadWorker(QThread):
    """Reads the DEM, builds the ray caster, poses and annotations."""
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, folder, dem_mesh, origin, correction, parent=None):
        super().__init__(parent)
        self.folder, self.dem_mesh, self.origin = folder, dem_mesh, origin
        self.correction = correction

    def run(self):
        try:
            from .viewer3d.frame_cache import image_size
            from .core import inspection
            result = {"scenes": {}, "detections": {}, "tracks": {}}
            self.progress.emit("Reading the DEM mesh…")
            dem = fv.load_dem(self.dem_mesh, self.origin)
            result["dem"] = dem
            self.progress.emit("Building the ray caster…")
            result["caster"] = fv.make_caster(dem)
            for modality in fv.MODALITIES:
                scene = fv.FlightScene.load(self.folder, modality,
                                            self.correction)
                if not len(scene):
                    continue
                size = None
                for i in range(min(len(scene), 20)):
                    path = scene.frame_path(i)
                    if path and os.path.isfile(path):
                        size = image_size(path)
                        break
                if size:
                    scene.image_size = size
                    scene.aspect = size[0] / float(size[1])
                result["scenes"][modality] = scene
                self.progress.emit(
                    f"Loading {fv.MODALITY_LABELS[modality]} annotations…")
                try:
                    result["detections"][modality] = \
                        inspection.load_pixel_detections(self.folder, modality)
                except Exception:  # noqa: BLE001 - no store yet
                    result["detections"][modality] = []
                try:
                    result["tracks"][modality] = \
                        inspection.load_pixel_tracks(self.folder, modality)
                except Exception:  # noqa: BLE001
                    result["tracks"][modality] = {}
            if not result["scenes"]:
                raise RuntimeError(
                    "No poses_t.json / poses_w.json in the target folder. "
                    "Run the frame extraction first.")
            self.done.emit(result)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.failed.emit(str(exc))


def _scrolled(widget):
    """*widget* in a vertical scroll area, so its forms keep their natural
    height instead of being squeezed into the window."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(widget)
    return area


class _SimplifyWorker(QThread):
    """Resamples the DEM to a display detail level (see fv.simplify_dem)."""
    done = pyqtSignal(float, object)
    failed = pyqtSignal(str)

    def __init__(self, dem, fraction, caster, parent=None):
        super().__init__(parent)
        self.dem, self.fraction, self.caster = dem, fraction, caster

    def run(self):
        try:
            self.done.emit(self.fraction,
                           fv.simplify_dem(self.dem, self.fraction, self.caster))
        except Exception as exc:  # noqa: BLE001 - shown in the panel
            self.failed.emit(str(exc))


class _OrthoWorker(QThread):
    """Reads a camera's orthomosaic (downsampled) off the GUI thread."""
    done = pyqtSignal(str, object)        # modality, (rgba, lower, upper)
    failed = pyqtSignal(str, str)

    def __init__(self, modality, path, max_edge, parent=None):
        super().__init__(parent)
        self.modality, self.path, self.max_edge = modality, path, max_edge

    def run(self):
        try:
            from .core.flight_viewer_io import read_raster_rgba
            self.done.emit(self.modality,
                           read_raster_rgba(self.path, self.max_edge))
        except Exception as exc:  # noqa: BLE001 - shown next to the checkbox
            self.failed.emit(self.modality, str(exc))


class _OutlineWorker(QThread):
    """Projects box outlines onto the DEM for a whole camera, in the
    background, so boxes seen earlier can be drawn wherever the replay is."""
    chunk = pyqtSignal(int, object)       # generation, {key: ring}
    progress = pyqtSignal(int, float)

    def __init__(self, generation, jobs, images, correction, aspect, size,
                 caster, parent=None):
        super().__init__(parent)
        self.generation, self.jobs = generation, jobs
        self.images, self.correction = images, correction
        self.aspect, self.size, self.caster = aspect, size, caster
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        out = {}
        for n, (frame, boxes) in enumerate(self.jobs):
            if self._stop:
                return
            try:
                t, r = fv.correction_for(frame, self.correction)
                cam = fv.camera_for_pose(self.images[frame], t, r, self.aspect)
                out.update(_project_outlines(cam, self.size, frame, boxes,
                                             self.caster))
            except Exception:  # nosec B112 - a frame without outlines
                continue
            if len(out) >= 400 or n == len(self.jobs) - 1:
                self.chunk.emit(self.generation, out)
                self.progress.emit(self.generation, (n + 1) / len(self.jobs))
                out = {}
        if out:
            self.chunk.emit(self.generation, out)


OUTLINE_SAMPLES = 8


def _outline_key(frame, box):
    return (int(frame), round(float(box["x1"]), 1), round(float(box["y1"]), 1),
            round(float(box["x2"]), 1), round(float(box["y2"]), 1))


def _project_outlines(camera, size, frame, boxes, caster):
    """``{key: (4*samples, 3) ring on the DEM}`` for the boxes of one frame."""
    if not boxes:
        return {}
    w, h = size
    px = geo.box_edge_pixels(boxes, OUTLINE_SAMPLES)
    dirs = fv.pixel_rays(camera, w, h, px.reshape(-1, 2))
    hits = fv.cast_rays(caster, camera.position[None, :], dirs)
    rings = geo.lift(hits, 0.25).reshape(len(boxes), -1, 3)
    return {_outline_key(frame, b): ring for b, ring in zip(boxes, rings)}


class _FallbackView(QLabel):
    """Shown instead of the 3D view when OpenGL 3.3 / PyOpenGL is missing."""

    def __init__(self, message: str, parent=None):
        super().__init__(parent)
        self.message = message
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setText(message)
        self._image = None

    def show_frame(self, rgba: Optional[np.ndarray]):
        if rgba is None:
            self.setText(self.message)
            return
        h, w = rgba.shape[:2]
        self._image = np.ascontiguousarray(rgba)
        img = QImage(self._image.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
        self.setPixmap(QPixmap.fromImage(img).scaled(
            self.width(), self.height(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))


class FlightViewerDialog(QDialog):
    """The 3D Flight Viewer window."""

    def __init__(self, iface, dock_widget=None, parent=None):
        super().__init__(parent)
        self.iface = iface
        self._dock_widget = dock_widget
        self.setWindowTitle("BAMBI - 3D Flight Viewer")
        buttons = Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowMinimizeButtonHint
        self.setWindowFlags(self.windowFlags() | buttons)
        self.resize(1400, 860)

        self.config: dict = {}
        self.folder = ""
        self.dem = None              # full DEM: ray casting, extents
        self.display_dem = None      # what is drawn and integrated (detail)
        self._detail_cache: Dict[float, object] = {}
        self._detail_worker = None
        self._scan_last, self._scan_queue = -1, []
        self._trail_ready = False
        self._trail_matrix = None
        self._ortho: Dict[str, np.ndarray] = {}      # modality -> drape matrix
        self._ortho_worker = None
        self._ortho_text: Dict[str, str] = {}
        self._outlines: Dict[tuple, np.ndarray] = {}
        self._outline_worker = None
        self._outline_generation = 0
        self._mask_paths: Dict[str, str] = {}
        self._mask_images: Dict[str, object] = {}
        self.caster = None
        self.scenes: Dict[str, fv.FlightScene] = {}
        self.detections: Dict[str, list] = {}
        self.tracks: Dict[str, dict] = {}
        self.modality = "w"
        self.index = 0
        self.playing = False
        self.marker = None
        self._worker = None
        self._ground_cache: Dict[tuple, Optional[np.ndarray]] = {}
        self._wait_since = None

        self._build_ui()
        self._restore_settings()

        from .viewer3d.frame_cache import FrameCache
        self.cache = FrameCache(max_edge=self.texture_combo.currentData())
        self.cache.loaded.connect(self._on_frame_loaded)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._tick)

        self.apply_dock_defaults()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.flight_label = QLabel("No flight loaded")
        self.flight_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        top.addWidget(self.flight_label, 1)
        top.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        top.addWidget(self.camera_combo)
        self.reload_btn = QPushButton("Reload")
        self.reload_btn.setToolTip("Reload the active flight from the BAMBI panel")
        self.reload_btn.clicked.connect(self.reload)
        top.addWidget(self.reload_btn)
        root.addLayout(top)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.view_host = QWidget()
        self.view_layout = QVBoxLayout(self.view_host)
        self.view_layout.setContentsMargins(0, 0, 0, 0)
        self.viewport = None
        self._create_viewport()
        self.splitter.addWidget(self.view_host)

        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(320)
        self.tabs.addTab(_scrolled(self._build_view_tab()), "View")
        self._extra_tabs()
        self.splitter.addWidget(self.tabs)
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)
        root.addWidget(self.splitter, 1)

        root.addLayout(self._build_transport())

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray;")
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: gray;")
        bottom = QHBoxLayout()
        bottom.addWidget(self.status_label, 1)
        bottom.addWidget(self.info_label)
        root.addLayout(bottom)

        self._shortcuts()

    def _extra_tabs(self):
        """ALFS, corrections and reconstruction tabs (see their modules)."""
        from .viewer3d.alfs_panel import AlfsPanel
        from .viewer3d.corrections_panel import CorrectionsPanel
        from .viewer3d.reconstruction_panel import ReconstructionPanel
        self.alfs_panel = AlfsPanel(self)
        self.tabs.addTab(_scrolled(self.alfs_panel), "ALFS")
        self.corrections_panel = CorrectionsPanel(self)
        self.tabs.addTab(_scrolled(self.corrections_panel), "Corrections")
        self.recon_panel = ReconstructionPanel(self)
        self.tabs.addTab(_scrolled(self.recon_panel), "Reconstruction")

    def _create_viewport(self):
        from .viewer3d import missing_dependencies
        missing = missing_dependencies()
        if missing:
            self._use_fallback(
                "The interactive 3D view needs " + " and ".join(missing)
                + ".\nInstall it with the BAMBI Dependency Manager "  # noqa: W503
                "(\"3D Flight Viewer\"), then reopen this window.\n\n"
                "Until then the frames are shown here without the 3D scene.")
            return
        from .viewer3d.viewport import FlightViewport
        self.viewport = FlightViewport()
        self.viewport.failed.connect(self._on_gl_failed)
        self.viewport.navigated.connect(self._on_navigated)
        self.viewport.hovered.connect(self._on_hover)
        self.viewport.picked.connect(self._on_pick)
        self.viewport.clicked.connect(self._on_click)
        self.viewport.moved.connect(self._on_move)
        self.viewport.set_pick_function(self._cast)
        self.view_layout.addWidget(self.viewport)

    def _use_fallback(self, message):
        self.viewport = None
        self.fallback = _FallbackView(message)
        self.view_layout.addWidget(self.fallback)

    def _on_gl_failed(self, message):
        if self.viewport is not None:
            self.viewport.setParent(None)
            self.viewport.deleteLater()
        self._use_fallback(
            "The 3D view could not start OpenGL 3.3:\n" + message
            + "\n\nThis happens on remote desktops, virtual machines and very "  # noqa: W503
            "old graphics drivers. The frames are shown here without the 3D "
            "scene.")
        self._refresh()

    def _build_view_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        drape = QGroupBox("On the DEM")
        f = QFormLayout(drape)
        self.drape_combo = QComboBox()
        self.drape_combo.addItem("Current frame (pinhole)", DRAPE_FRAME)
        self.drape_combo.addItem("ALFS integral", DRAPE_ALFS)
        self.drape_combo.addItem("Nothing", DRAPE_NONE)
        self.drape_combo.setToolTip(
            "What is projected onto the DEM: the current frame alone, the "
            "ALFS integral of the frames around it (see the ALFS tab), or "
            "nothing.")
        self.drape_combo.currentIndexChanged.connect(self._on_drape_changed)
        f.addRow("Show", self.drape_combo)
        self.trail_check = QCheckBox("Keep trail (continuous orthophoto)")
        self.trail_check.setToolTip(
            "Every frame shown is painted into one top-down texture that "
            "stays on the DEM - like a scanner sliding along the route. "
            "Newer frames cover older ones. Stepping back before the start "
            "of the trail starts it anew.")
        self.trail_check.toggled.connect(self._on_trail_toggled)
        self.trail_show_check = QCheckBox("Show")
        self.trail_show_check.setChecked(True)
        self.trail_show_check.setToolTip(
            "Draw the trail. Unticked, the trail is hidden but kept - and "
            "keeps growing while the replay runs - so it reappears complete.")
        self.trail_show_check.toggled.connect(self._refresh_layers)
        self.trail_show_check.setEnabled(False)
        row = QHBoxLayout()
        row.addWidget(self.trail_check, 1)
        row.addWidget(self.trail_show_check)
        f.addRow(row)
        self.mask_check = QCheckBox("Hide the undistortion border (mask)")
        self.mask_check.setChecked(True)
        self.mask_check.setToolTip(
            "Use the camera's undistortion mask (mask_T.png / mask_W.png in the "
            "target folder) so the black border of undistorted frames is not "
            "draped, integrated by ALFS or used by the reconstruction.")
        self.mask_check.toggled.connect(self._on_mask_toggled)
        f.addRow(self.mask_check)
        self.erosion_spin = QSpinBox()
        self.erosion_spin.setRange(0, 20)
        self.erosion_spin.setValue(2)
        self.erosion_spin.setSuffix(" px")
        self.erosion_spin.setToolTip(
            "Shrink the mask by this many pixels: the pixels right at the "
            "mask edge blend image and black border once sampled and leave a "
            "dark rim. Same idea as the GeoTIFF export's edge erosion, whose "
            "value it starts from.")
        self.erosion_spin.valueChanged.connect(self._on_mask_toggled)
        f.addRow("Mask edge erosion", self.erosion_spin)
        self.trail_step_spin = QSpinBox()
        self.trail_step_spin.setRange(1, 200)
        self.trail_step_spin.setValue(2)
        self.trail_step_spin.setToolTip(
            "When the replay jumps ahead (step > 1, slider, click), the frames "
            "in between are painted into the trail every k-th frame.")
        f.addRow("Trail: fill every k-th frame", self.trail_step_spin)
        row = QHBoxLayout()
        self.trail_res_combo = QComboBox()
        for size in (2048, 4096, 8192):
            self.trail_res_combo.addItem(f"{size} px", size)
        self.trail_res_combo.setCurrentIndex(1)
        self.trail_res_combo.setToolTip("Size of the trail texture over the "
                                        "flight area (longest edge).")
        self.trail_res_combo.currentIndexChanged.connect(self._reset_trail)
        row.addWidget(self.trail_res_combo, 1)
        b = QPushButton("Clear")
        b.setToolTip("Start the trail anew from the current frame")
        b.clicked.connect(self._reset_trail)
        row.addWidget(b)
        f.addRow("Trail resolution", row)
        self.trail_info = QLabel("")
        self.trail_info.setStyleSheet("color: gray;")
        f.addRow(self.trail_info)
        self.ortho_check = QCheckBox("Show orthomosaic")
        self.ortho_check.setToolTip(
            "Drape the flight's orthomosaic (pipeline step \"Orthomosaic\", "
            "orthomosaic_t / orthomosaic_w in the target folder) onto the DEM "
            "- a static alternative to the trail. The current frame is drawn "
            "on top of it.")
        self.ortho_check.toggled.connect(self._on_ortho_toggled)
        f.addRow(self.ortho_check)
        self.ortho_info = QLabel("")
        self.ortho_info.setStyleSheet("color: gray;")
        self.ortho_info.setWordWrap(True)
        f.addRow(self.ortho_info)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._refresh)
        f.addRow("Opacity", self.opacity_slider)
        self.texture_combo = QComboBox()
        for size in (512, 1024, 2048, 4096):
            self.texture_combo.addItem(f"{size} px", size)
        self.texture_combo.setCurrentIndex(1)
        self.texture_combo.setToolTip(
            "Longest edge of the frame textures. Larger is sharper but uses "
            "more memory and loads slower.")
        self.texture_combo.currentIndexChanged.connect(self._on_texture_size)
        f.addRow("Texture size", self.texture_combo)
        lay.addWidget(drape)

        cam = QGroupBox("Camera")
        f = QFormLayout(cam)
        self.follow_check = QCheckBox("Follow camera")
        self.follow_check.setToolTip(
            "Lock the view to the drone camera of the current frame (F). "
            "Dragging the view leaves follow mode.")
        self.follow_check.toggled.connect(self._on_follow_toggled)
        f.addRow(self.follow_check)
        self.frustum_check = QCheckBox("Show frustum")
        self.frustum_check.setChecked(True)
        self.frustum_check.toggled.connect(self._refresh)
        f.addRow(self.frustum_check)
        self.route_check = QCheckBox("Show route")
        self.route_check.setChecked(True)
        self.route_check.toggled.connect(self._refresh)
        f.addRow(self.route_check)
        self.poses_check = QCheckBox("Show camera positions")
        self.poses_check.setChecked(True)
        self.poses_check.toggled.connect(self._refresh)
        f.addRow(self.poses_check)
        self.line_spin = QSpinBox()
        self.line_spin.setRange(1, 20)
        self.line_spin.setValue(4)
        self.line_spin.setSuffix(" px")
        self.line_spin.setToolTip("Width of the route, frustum and track lines.")
        self.line_spin.valueChanged.connect(self._refresh)
        f.addRow("Line width", self.line_spin)
        self.marker_spin = QSpinBox()
        self.marker_spin.setRange(2, 60)
        self.marker_spin.setValue(10)
        self.marker_spin.setSuffix(" px")
        self.marker_spin.setToolTip(
            "Size of the camera position markers; the current camera is drawn "
            "larger.")
        self.marker_spin.valueChanged.connect(self._refresh)
        f.addRow("Camera marker size", self.marker_spin)
        lay.addWidget(cam)

        ann = QGroupBox("Annotations")
        f = QFormLayout(ann)
        self.annotation_combo = QComboBox()
        self.annotation_combo.addItem("No annotations", ANNOTATION_NONE)
        self.annotation_combo.addItem("Show bounding boxes", ANNOTATION_BOXES)
        self.annotation_combo.addItem("Show tracks", ANNOTATION_TRACKS)
        self.annotation_combo.currentIndexChanged.connect(self._on_annotation_mode)
        f.addRow(self.annotation_combo)
        self.history_spin = QSpinBox()
        self.history_spin.setRange(0, 100000)
        self.history_spin.setSpecialValueText("whole track")
        self.history_spin.setToolTip("Frames of track history to draw (0 = all).")
        self.history_spin.valueChanged.connect(self._refresh)
        f.addRow("Track history", self.history_spin)
        self.seen_check = QCheckBox("Keep boxes seen earlier (dashed)")
        self.seen_check.setChecked(True)
        self.seen_check.setToolTip(
            "Also draw boxes from earlier frames, dashed, after the camera has "
            "moved on - an overview of what has been seen. Boxes of the "
            "current frame are solid. For tracks: the last box of every track "
            "seen so far.")
        self.seen_check.toggled.connect(self._refresh)
        f.addRow(self.seen_check)
        self.seen_step_spin = QSpinBox()
        self.seen_step_spin.setRange(1, 1000)
        self.seen_step_spin.setValue(10)
        self.seen_step_spin.setToolTip(
            "Bounding boxes mode: keep the boxes of every k-th frame only, so "
            "an animal seen in hundreds of frames does not pile up hundreds "
            "of outlines.")
        self.seen_step_spin.valueChanged.connect(self._on_annotation_mode)
        f.addRow("Seen boxes: every k-th frame", self.seen_step_spin)
        self.ann_width_spin = QSpinBox()
        self.ann_width_spin.setRange(1, 20)
        self.ann_width_spin.setValue(3)
        self.ann_width_spin.setSuffix(" px")
        self.ann_width_spin.valueChanged.connect(self._refresh)
        f.addRow("Box line width", self.ann_width_spin)
        self.ann_info = QLabel("")
        self.ann_info.setStyleSheet("color: gray;")
        self.ann_info.setWordWrap(True)
        f.addRow(self.ann_info)
        lay.addWidget(ann)

        base = QGroupBox("DEM")
        f = QFormLayout(base)
        self.base_combo = QComboBox()
        self.base_combo.addItem("Shaded relief", 0)
        self.base_combo.addItem("DEM texture", 1)
        self.base_combo.currentIndexChanged.connect(self._on_base_changed)
        f.addRow("Surface", self.base_combo)
        self.dem_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.dem_opacity_slider.setRange(10, 100)
        self.dem_opacity_slider.setValue(100)
        self.dem_opacity_slider.valueChanged.connect(self._on_base_changed)
        f.addRow("Opacity", self.dem_opacity_slider)
        self.detail_combo = QComboBox()
        for label, fraction in fv.DEM_DETAIL:
            self.detail_combo.addItem(label, fraction)
        self.detail_combo.setToolTip(
            "Triangles of the DEM drawn and used for the draped frames and "
            "live ALFS here. Fewer triangles render faster - every draped or "
            "integrated frame draws the whole DEM once - at the cost of height "
            "accuracy on rough terrain. Ray casting, frustums and the "
            "reconstruction always use the full DEM; the plugin's own ALFS "
            "and GeoTIFF products are not affected.")
        self.detail_combo.currentIndexChanged.connect(self._on_detail_changed)
        f.addRow("Detail", self.detail_combo)
        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet("color: gray;")
        self.detail_label.setWordWrap(True)
        f.addRow(self.detail_label)
        self.wire_check = QCheckBox("Wireframe")
        self.wire_check.toggled.connect(self._on_base_changed)
        f.addRow(self.wire_check)
        lay.addWidget(base)

        row = QHBoxLayout()
        b = QPushButton("Reset view")
        b.setToolTip("Show the whole flight (Home)")
        b.clicked.connect(self.reset_view)
        row.addWidget(b)
        b = QPushButton("Screenshot…")
        b.clicked.connect(self._save_screenshot)
        row.addWidget(b)
        lay.addLayout(row)
        lay.addStretch(1)
        return w

    def _build_transport(self):
        row = QHBoxLayout()
        style = self.style()

        def tool(icon, tip, slot):
            btn = QToolButton()
            btn.setIcon(style.standardIcon(icon))
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            row.addWidget(btn)
            return btn

        SP = QStyle.StandardPixmap
        tool(SP.SP_MediaSkipBackward, "First frame", lambda: self.set_frame(0))
        tool(SP.SP_MediaSeekBackward, "Previous frame (←, Shift = 10)",
             lambda: self.step(-1))
        self.play_btn = tool(SP.SP_MediaPlay, "Play / pause (Space)",
                             self.toggle_play)
        tool(SP.SP_MediaSeekForward, "Next frame (→, Shift = 10)",
             lambda: self.step(1))
        tool(SP.SP_MediaSkipForward, "Last frame",
             lambda: self.set_frame(self._count() - 1))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(lambda v: self.set_frame(v))
        row.addWidget(self.slider, 1)
        self.frame_spin = QSpinBox()
        self.frame_spin.setKeyboardTracking(False)
        self.frame_spin.valueChanged.connect(lambda v: self.set_frame(v))
        row.addWidget(self.frame_spin)
        self.count_label = QLabel("/ 0")
        row.addWidget(self.count_label)
        row.addWidget(QLabel("Step"))
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 100)
        self.step_spin.setToolTip(
            "Frames to advance per step and per playback tick "
            "(jump in stride)")
        row.addWidget(self.step_spin)
        self.speed_combo = QComboBox()
        for label, value in SPEEDS:
            self.speed_combo.addItem(label, value)
        self.speed_combo.setCurrentIndex(2)
        row.addWidget(self.speed_combo)
        self.loop_check = QCheckBox("Loop")
        row.addWidget(self.loop_check)
        self.time_label = QLabel("")
        self.time_label.setMinimumWidth(150)
        row.addWidget(self.time_label)
        return row

    def _shortcuts(self):
        def sc(key, slot):
            s = QShortcut(QKeySequence(key), self)
            s.activated.connect(slot)

        sc("Space", self.toggle_play)
        for key, d in (("Right", 1), ("Left", -1), ("J", 1), ("H", -1)):
            sc(key, lambda d=d: self.step(d))
            sc("Shift+" + key, lambda d=d: self.step(10 * d))
        sc("F", self.follow_check.toggle)
        sc("Home", self.reset_view)
        sc("Return", lambda: self.alfs_panel.render_now())
        sc("Enter", lambda: self.alfs_panel.render_now())

    # -------------------------------------------------------------- settings

    _SAVED = ("trail_check", "trail_show_check", "ortho_check", "trail_step_spin", "opacity_slider", "frustum_check",
              "route_check", "poses_check", "history_spin", "step_spin",
              "loop_check", "dem_opacity_slider", "mask_check", "line_spin",
              "marker_spin", "seen_check", "seen_step_spin", "ann_width_spin",
              "erosion_spin")
    _SAVED_COMBOS = ("drape_combo", "annotation_combo", "speed_combo",
                     "texture_combo", "base_combo", "detail_combo", "trail_res_combo")

    def _restore_settings(self):
        s = _settings()
        s.beginGroup(SETTINGS_GROUP)
        for name in self._SAVED:
            w = getattr(self, name)
            val = s.value(name)
            if val is None:
                continue
            try:
                if isinstance(w, QCheckBox):
                    w.setChecked(str(val).lower() in ("1", "true"))
                else:
                    w.setValue(int(val))
            except (TypeError, ValueError):
                pass
        for name in self._SAVED_COMBOS:
            w = getattr(self, name)
            val = s.value(name)
            if val is not None:
                i = w.findText(str(val))
                if i >= 0:
                    w.setCurrentIndex(i)
        s.endGroup()

    def _save_settings(self):
        s = _settings()
        s.beginGroup(SETTINGS_GROUP)
        for name in self._SAVED:
            w = getattr(self, name)
            s.setValue(name, w.isChecked() if isinstance(w, QCheckBox) else w.value())
        for name in self._SAVED_COMBOS:
            s.setValue(name, getattr(self, name).currentText())
        s.endGroup()
        for panel in ("alfs_panel", "corrections_panel", "recon_panel"):
            p = getattr(self, panel, None)
            if p is not None and hasattr(p, "save_settings"):
                p.save_settings()

    # ------------------------------------------------------------- loading

    def apply_dock_defaults(self):
        """Load the flight that is active in the BAMBI panel."""
        if self._dock_widget is None:
            return
        try:
            config = self._dock_widget.get_config()
        except Exception:  # noqa: BLE001
            return
        if config.get("target_folder") != self.folder or not self.scenes:
            self.load(config)

    def reload(self):
        self.config, self.folder = {}, ""
        self.apply_dock_defaults()

    def _dem_paths(self, config):
        from .core.agl_profile import dem_origin
        from .core.inspection import find_dem_mesh_path
        dem_path = config.get("dem_path", "") or ""
        dem_json = config.get("dem_metadata_path", "") or ""
        if not dem_json and dem_path.lower().endswith(".json"):
            dem_json = dem_path
        mesh = find_dem_mesh_path(dem_path, dem_json)
        origin = dem_origin(dem_path, dem_json, config.get("target_folder", ""))
        return mesh, origin

    def load(self, config: dict):
        folder = config.get("target_folder", "")
        if not folder or not os.path.isdir(folder):
            self.status("Choose a target folder in the BAMBI panel first.")
            return
        mesh, origin = self._dem_paths(config)
        if not mesh:
            self.status("No DEM mesh (.glb/.gltf) configured for this flight. "
                        "Import or download a DEM in the BAMBI panel.")
            return
        self.stop()
        self.config = dict(config)
        self.folder = folder
        if "geotiff_edge_erosion_px" in config and not getattr(
                self, "_erosion_from_dock", False):
            self.erosion_spin.setValue(int(config["geotiff_edge_erosion_px"]))
            self._erosion_from_dock = True
        self.flight_label.setText(f"Flight: {folder}")
        correction = self.corrections_panel.initial_correction(config)
        self._worker = _LoadWorker(folder, mesh, origin, correction, self)
        self._worker.progress.connect(self.status)
        self._worker.failed.connect(self._on_load_failed)
        self._worker.done.connect(self._on_loaded)
        self.reload_btn.setEnabled(False)
        self.status("Loading flight…")
        self._worker.start()

    def _on_load_failed(self, message):
        self.reload_btn.setEnabled(True)
        self.status("")
        QMessageBox.warning(self, "3D Flight Viewer",
                            f"The flight could not be loaded:\n\n{message}")

    def _on_loaded(self, result):
        self.reload_btn.setEnabled(True)
        self.dem = result["dem"]
        self.display_dem = self.dem
        self._trail_ready = False
        self._scan_last, self._scan_queue = -1, []
        self._detail_cache = {1.0: fv.DemSimplification(self.dem, 1.0, 0.0, 0.0)}
        self.caster = result["caster"]
        self.scenes = result["scenes"]
        self.detections = result["detections"]
        self.tracks = result["tracks"]
        self._ground_cache.clear()
        self.cache.clear()
        if self.viewport is not None:
            self.viewport.set_dem(self.dem)
            self.base_combo.model().item(1).setEnabled(self.dem.texture is not None)
        self._on_detail_changed()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for m in ("w", "t"):
            if m in self.scenes:
                self.camera_combo.addItem(fv.MODALITY_LABELS[m], m)
        self.camera_combo.blockSignals(False)
        self.modality = self.camera_combo.currentData()
        self._mask_paths, self._mask_images = {}, {}
        if self.viewport is not None:
            for m in list(self._ortho):
                self.viewport.drop_texture(("ortho", m))
        self._ortho, self._ortho_text = {}, {}
        self._update_mask_check()
        self._update_ortho_check()
        self._on_annotation_info()
        self._restart_outlines()
        self._on_base_changed()
        self._setup_range()
        self.reset_view()
        for panel in (self.alfs_panel, self.corrections_panel, self.recon_panel):
            panel.flight_loaded()
        self.status(f"Loaded {len(self.scene)} {fv.MODALITY_LABELS[self.modality]} "
                    f"frames and a DEM with {len(self.dem.faces):,} triangles.")
        self.set_frame(0, force=True)

    # --------------------------------------------------------------- state

    @property
    def scene(self) -> Optional[fv.FlightScene]:
        return self.scenes.get(self.modality)

    def _count(self) -> int:
        return len(self.scene) if self.scene is not None else 0

    def _setup_range(self):
        n = max(self._count() - 1, 0)
        for w in (self.slider, self.frame_spin):
            w.blockSignals(True)
            w.setRange(0, n)
            w.blockSignals(False)
        self.count_label.setText(f"/ {n}")

    def status(self, text: str):
        self.status_label.setText(text)

    def origin(self) -> np.ndarray:
        if self.dem is not None and self.dem.origin is not None:
            return np.asarray(self.dem.origin, dtype=float)
        return np.zeros(3)

    def _pick_casters(self):
        """What the mouse can hit: the DEM (when drawn) and the reconstructed
        surface (when it is there and shown)."""
        casters = []
        vp = self.viewport
        if vp is None or (vp.dem_visible and "dem" not in vp.hidden):
            casters.append(("dem", self.caster))
        recon = getattr(self, "recon_panel", None)
        if recon is not None:
            casters.append(("surface", recon.pick_caster()))
        if not any(c is not None for _, c in casters):
            casters = [("dem", self.caster)]    # DEM hidden, nothing else shown
        return casters

    def _cast(self, origin, direction):
        """Nearest hit for mouse hover, double click and zooming."""
        hit, kind = fv.pick_nearest(origin, direction, self._pick_casters())
        self._last_hit_kind = kind
        return hit

    def set_correction(self, correction: dict):
        """Apply edited correction factors to every camera (both cameras)."""
        for scene in self.scenes.values():
            scene.set_correction(correction)
        self._ground_cache.clear()
        self._restart_outlines()
        self._reset_trail()
        self._refresh()
        self.alfs_panel.cameras_changed()

    # ----------------------------------------------------------- navigation

    def set_frame(self, index: int, force: bool = False):
        if self.scene is None:
            return
        index = min(max(int(index), 0), self._count() - 1)
        if index == self.index and not force:
            return
        self.index = index
        for w in (self.slider, self.frame_spin):
            w.blockSignals(True)
            w.setValue(index)
            w.blockSignals(False)
        self._refresh()
        self.alfs_panel.frame_changed()
        self.corrections_panel.frame_changed()

    def step(self, delta: int):
        if self.scene is None:
            return
        self.set_frame(fv.step_index(self.index, delta * self.step_spin.value(),
                                     self._count(), self.loop_check.isChecked()))

    def reset_view(self):
        if self.viewport is None or self.dem is None:
            return
        lo, hi = self.dem.bounds
        if self.scene is not None and len(self.scene):
            pos = self.scene.positions()
            lo = np.minimum(lo, np.nanmin(pos, axis=0))
            hi = np.maximum(hi, np.nanmax(pos, axis=0))
        self.viewport.orbit.azimuth, self.viewport.orbit.elevation = -90.0, 50.0
        self.viewport.fit(lo, hi)
        if self.follow_check.isChecked():
            self.follow_check.setChecked(False)

    def _on_camera_changed(self):
        new = self.camera_combo.currentData()
        if not new or new == self.modality or self.scene is None:
            self.modality = new or self.modality
            return
        other = self.scenes.get(new)
        index = fv.matching_frame(self.scene.images, other.images, self.index) \
            if other is not None else 0
        self.modality = new
        self._trail_ready = False
        self._scan_last, self._scan_queue = -1, []
        self._update_mask_check()
        self._update_ortho_check()
        self._on_annotation_info()
        self._restart_outlines()
        self._setup_range()
        self.alfs_panel.cameras_changed()
        self.set_frame(index, force=True)

    def _on_drape_changed(self):
        # The trail is kept (and continued) when coming back to the frames.
        self._refresh()
        if self.drape_combo.currentData() == DRAPE_ALFS:
            self.alfs_panel.frame_changed()     # render right away

    def _on_follow_toggled(self, on):
        if self.viewport is None:
            return
        if on:
            self._refresh()
        else:
            self.viewport.leave_follow()

    def _on_navigated(self):
        if self.viewport is not None and self.viewport.follow is None \
                and self.follow_check.isChecked():
            self.follow_check.blockSignals(True)
            self.follow_check.setChecked(False)
            self.follow_check.blockSignals(False)
        self.alfs_panel.view_changed()

    def _on_texture_size(self):
        self.cache.clear(self.texture_combo.currentData())
        if self.viewport is not None:
            for key in list(self.viewport._ready_textures):
                if isinstance(key, tuple) and key[0] in ("frame", "boxes"):
                    self.viewport.drop_texture(key)
        self._refresh()

    def _on_base_changed(self):
        if self.viewport is None:
            return
        self.viewport.dem_mode = self.base_combo.currentData()
        self.viewport.dem_opacity = self.dem_opacity_slider.value() / 100.0
        self.viewport.wireframe = self.wire_check.isChecked()
        self.viewport.update()

    # ------------------------------------------------------------ DEM detail

    def _on_detail_changed(self):
        if self.dem is None:
            return
        fraction = self.detail_combo.currentData()
        cached = self._detail_cache.get(fraction)
        if cached is not None:
            self._apply_detail(cached)
            return
        if self._detail_worker is not None:      # finishes, then re-checks
            return
        self.detail_label.setText("Reducing the DEM…")
        self._detail_worker = _SimplifyWorker(self.dem, fraction, self.caster, self)
        self._detail_worker.done.connect(self._on_detail_ready)
        self._detail_worker.failed.connect(self._on_detail_failed)
        self._detail_worker.start()

    def _finish_detail_worker(self):
        if self._detail_worker is not None:
            self._detail_worker.wait(2000)
        self._detail_worker = None

    def _on_detail_ready(self, fraction, result):
        self._finish_detail_worker()
        self._detail_cache[fraction] = result
        self._on_detail_changed()   # applies it, or starts the level now wanted

    def _on_detail_failed(self, message):
        self._finish_detail_worker()
        self.detail_label.setText("Could not reduce the DEM: " + message)

    def _apply_detail(self, result):
        self.display_dem = result.dem
        n = len(result.dem.faces)
        if result.fraction >= 0.999:
            self.detail_label.setText(f"{n:,} triangles (full DEM)")
        else:
            self.detail_label.setText(
                f"{n:,} triangles ({result.fraction * 100:.0f} %), height "
                f"error max {result.max_error:.2f} m, RMS {result.rms_error:.2f} m")
        if self.viewport is not None:
            self.viewport.set_dem(result.dem)
            self.base_combo.model().item(1).setEnabled(result.dem.texture is not None)
            if result.dem.texture is None and self.base_combo.currentData() == 1:
                self.base_combo.setCurrentIndex(0)
        self.alfs_panel.cameras_changed()

    def _on_annotation_mode(self):
        self._on_annotation_info()
        self._restart_outlines()
        self._refresh()

    def _on_annotation_info(self):
        mode = self.annotation_combo.currentData()
        dets = self.detections.get(self.modality, [])
        tracks = self.tracks.get(self.modality, {})
        if mode == ANNOTATION_BOXES:
            self.ann_info.setText(f"{len(dets):,} detections on this camera."
                                  if dets else "No detections for this camera.")
        elif mode == ANNOTATION_TRACKS:
            self.ann_info.setText(f"{len(tracks):,} tracks on this camera."
                                  if tracks else "No tracks for this camera.")
        else:
            self.ann_info.setText("")

    def toggle_play(self):
        if self.playing:
            self.stop()
        else:
            self.play()

    def play(self):
        if self.scene is None:
            return
        if self.index >= self._count() - 1 and not self.loop_check.isChecked():
            self.set_frame(0)
        self.playing = True
        self.play_btn.setIcon(self.style().standardIcon(
            QStyle.StandardPixmap.SP_MediaPause))
        self._schedule()

    def stop(self):
        self.playing = False
        if hasattr(self, "timer"):
            self.timer.stop()
        self.play_btn.setIcon(self.style().standardIcon(
            QStyle.StandardPixmap.SP_MediaPlay))

    def _schedule(self):
        speed = self.speed_combo.currentData()
        if speed <= 0:
            delay = 0.0
        else:
            times = self.scene.times()
            i, j = self.index, min(self.index + self.step_spin.value(),
                                   self._count() - 1)
            a, b = times[i], times[j]
            delay = (b - a) / speed if a is not None and b is not None and b > a \
                else self.step_spin.value() / (30.0 * speed)
        self.timer.start(int(max(delay, 0.0) * 1000))

    def _tick(self):
        if not self.playing or self.scene is None:
            return
        last = self._count() - 1
        if self.index >= last and not self.loop_check.isChecked():
            self.stop()
            return
        nxt = fv.step_index(self.index, self.step_spin.value(), self._count(),
                            self.loop_check.isChecked())
        # Show a frame only once its texture is there, so playback never
        # flashes an empty DEM; give up waiting after two seconds.
        if self._needs_frame_texture() and not self._texture_ready(nxt):
            import time
            now = time.monotonic()
            if self._wait_since is None:
                self._wait_since = now
            if now - self._wait_since < 2.0:
                self._request_frame(nxt)
                self.timer.start(15)
                return
        self._wait_since = None
        self.set_frame(nxt)
        self._schedule()

    # ------------------------------------------------------------ textures

    # ----------------------------------------------------------------- masks

    def mask_path(self, modality=None) -> str:
        modality = modality or self.modality
        if modality not in self._mask_paths:
            self._mask_paths[modality] = fv.find_mask(self.folder, modality) \
                if self.folder else ""
        return self._mask_paths[modality]

    def use_mask(self, modality=None) -> bool:
        return self.mask_check.isChecked() and bool(self.mask_path(modality))

    def _mask_image(self, modality):
        if modality not in self._mask_images:
            from .viewer3d.frame_cache import read_rgba
            path = self.mask_path(modality)
            self._mask_images[modality] = read_rgba(path, 4096) if path else None
        return self._mask_images[modality]

    def _upload_frame(self, key, arr):
        """Hand a frame texture to the viewport, masked if asked to."""
        if self.viewport is None:
            return
        if self.use_mask(key[1]):
            mask = self._mask_image(key[1])
            if mask is not None:
                arr = fv.apply_mask(arr, mask, self.erosion_spin.value())
        self.viewport.set_texture(key, arr)

    def mask_erosion(self) -> int:
        return self.erosion_spin.value()

    def _update_mask_check(self):
        has = bool(self.mask_path())
        self.mask_check.setEnabled(has)
        self.erosion_spin.setEnabled(has)
        self.mask_check.setToolTip(
            ("Use the camera's undistortion mask ({}) so the black border of "
             "undistorted frames is not draped, integrated by ALFS or used by "
             "the reconstruction.").format(os.path.basename(self.mask_path()))
            if has else "No undistortion mask (mask_T.png / mask_W.png) for "
                        "this camera in the target folder.")

    def _on_mask_toggled(self):
        self._reset_trail()
        if self.viewport is not None:
            for key in list(self.viewport._ready_textures) + \
                    list(self.viewport._pending_textures):
                if isinstance(key, tuple) and key[0] == "frame":
                    self.viewport.drop_texture(key)
        self._refresh()
        self.alfs_panel.cameras_changed()

    def _needs_frame_texture(self) -> bool:
        return (self.drape_combo.currentData() == DRAPE_FRAME or self.viewport is None)

    def _frame_key(self, index, modality=None):
        return ("frame", modality or self.modality, int(index))

    def _texture_ready(self, index) -> bool:
        key = self._frame_key(index)
        if self.viewport is None:
            return self.cache.contains(key)
        return self.viewport.has_texture(key) or self.cache.contains(key)

    def _request_frame(self, index, modality=None) -> Optional[np.ndarray]:
        modality = modality or self.modality
        scene = self.scenes.get(modality)
        if scene is None or not 0 <= index < len(scene):
            return None
        key = self._frame_key(index, modality)
        if self.cache.request(key, scene.frame_path(index)):
            return self.cache.get(key)
        return None

    def _on_frame_loaded(self, key, arr):
        if arr is None:
            return
        if self.viewport is not None:
            if self._scan_queue and key == self._frame_key(self._scan_queue[0]):
                self._process_scan()
            if key in self._wanted_textures():
                self._upload_frame(key, arr)
                self._refresh_layers()
        elif key == self._frame_key(self.index):
            self.fallback.show_frame(arr)

    def _wanted_textures(self) -> List[tuple]:
        keys = [self._frame_key(i) for i in self._scan_queue[:8]]
        if self.drape_combo.currentData() == DRAPE_FRAME:
            keys.append(self._frame_key(self.index))
        keys += self.corrections_panel.wanted_textures()
        return keys

    # ----------------------------------------------------------- trail

    def _trail_active(self) -> bool:
        return all((self.trail_check.isChecked(), self.viewport is not None,
                    self.drape_combo.currentData() == DRAPE_FRAME,
                    self.scene is not None, self.dem is not None))

    def _trail_extent(self):
        """XY extent of the flight's ground footprints (clipped to the DEM)."""
        lo_dem, hi_dem = self.dem.bounds
        pos = self.scene.positions()
        height = float(np.nanmax(pos[:, 2]) - lo_dem[2])
        fovy = max(fv.pose_fovy(m) for m in self.scene.images[:50]) \
            if self.scene.images else 50.0
        reach = height * math.tan(math.radians(fovy) / 2.0) \
            * max(self.scene.aspect, 1.0) * 1.5
        lo = np.maximum(np.nanmin(pos[:, :2], axis=0) - reach, lo_dem[:2])
        hi = np.minimum(np.nanmax(pos[:, :2], axis=0) + reach, hi_dem[:2])
        if np.any(hi <= lo):
            lo, hi = lo_dem[:2], hi_dem[:2]
        return lo, hi

    def _setup_trail(self):
        lo, hi = self._trail_extent()
        size = self.trail_res_combo.currentData()
        w, h = geo.scaled_size(int((hi[0] - lo[0]) * 1000),
                               int((hi[1] - lo[1]) * 1000), size)
        self._trail_matrix = geo.ortho_matrix_for_bounds(lo, hi)
        self.viewport.set_canvas(w, h, self._trail_matrix)
        self._trail_ready = True
        self._scan_last, self._scan_queue = -1, []
        self.trail_info.setText(
            f"{w}×{h} px, {(hi[0] - lo[0]) / w:.2f} m per pixel")

    def _reset_trail(self, *_):
        """Start the trail anew (it is rebuilt from the current frame)."""
        self._scan_last, self._scan_queue = -1, []
        if self.viewport is None:
            return
        if self._trail_active():
            self._setup_trail()
            self._refresh()
        else:
            self._trail_ready = False

    def _on_trail_toggled(self, on):
        self.trail_show_check.setEnabled(bool(on))
        self._reset_trail()
        self._refresh()

    def _scan_to(self, index: int):
        """Queue the frames from the end of the trail up to *index*."""
        if not self._trail_active():
            return
        if not self._trail_ready:
            self._setup_trail()
        if index < self._scan_last:                 # went back: start anew
            self.viewport.clear_canvas()
            self._scan_last, self._scan_queue = -1, []
        if index == self._scan_last:
            return
        k = self.trail_step_spin.value()
        if self._scan_last < 0:
            frames = [index]
        else:
            frames = list(range(self._scan_last + k, index, k)) + [index]
        # a frozen thermal image is not painted: it was taken elsewhere
        frozen = self.scene.duplicated()
        self._scan_queue.extend(f for f in frames if f not in frozen)
        self._scan_last = index
        self._process_scan()

    def _process_scan(self):
        """Paint queued frames in order as their textures become available."""
        while self._scan_queue and self.viewport is not None:
            i = self._scan_queue[0]
            key = self._frame_key(i)
            if not self.viewport.has_texture(key):
                arr = self._request_frame(i)
                if arr is None:
                    break                            # wait for the loader
                self._upload_frame(key, arr)
            self.viewport.paint_canvas(key, self.scene.camera(i).shot_matrix)
            self._scan_queue.pop(0)
        for i in self._scan_queue[:8]:              # prefetch the next ones
            self._request_frame(i)

    def _ensure_textures(self):
        """Hand cached frames to the viewport and request the missing ones."""
        if self.scene is None:
            return
        for key in self._wanted_textures():
            if self.viewport is not None and self.viewport.has_texture(key):
                continue
            arr = self._request_frame(key[2])
            if arr is not None and self.viewport is not None:
                self._upload_frame(key, arr)
        # Prefetch ahead of the replay.
        step = self.step_spin.value()
        for k in range(1, 9):
            self._request_frame(min(self.index + k * step, self._count() - 1))

    # ------------------------------------------------------------- drawing

    def _refresh(self):
        if self.scene is None:
            return
        self._update_time_label()
        if self.viewport is None:
            arr = self._request_frame(self.index)
            if arr is not None:
                self.fallback.show_frame(arr)
            return
        cam = self.scene.camera(self.index)
        if self.follow_check.isChecked():
            self.viewport.set_follow(cam)
        self._update_geometry(cam)
        self._scan_to(self.index)
        self._ensure_textures()
        self._refresh_layers()

    def _update_time_label(self):
        times = self.scene.times()
        t = times[self.index] if self.index < len(times) else None
        prev = times[self.index - 1] if self.index > 0 else None
        frozen = ""
        if self.scene.is_duplicated(self.index):
            from .core.poses import FROZEN_LABEL
            frozen = f"  ❄ {FROZEN_LABEL}"
        if t is None:
            self.time_label.setText(frozen.strip())
            return
        m, s = divmod(t, 60.0)
        h, m = divmod(int(m), 60)
        delta = f" (+{(t - prev) * 1000:.0f} ms)" if prev is not None else ""
        self.time_label.setText(
            f"{h:02d}:{m:02d}:{s:06.3f}{delta}{frozen}")

    def _update_geometry(self, cam: fv.FrameCamera):
        from .viewer3d.specs import LineSpec, PointSpec
        vp = self.viewport
        positions = self.scene.positions()
        if self.route_check.isChecked():
            seg, col = geo.route_segments(positions, self.index)
            vp.set_lines("route", LineSpec(seg, col, width=self.line_spin.value()))
        else:
            vp.set_lines("route", None)
        if self.poses_check.isChecked():
            pts, col = geo.pose_markers(positions, self.index, max_markers=60)
            vp.set_points("poses", PointSpec(pts, col, size=self.marker_spin.value()))
        else:
            vp.set_points("poses", None)
        vp.set_points("current", PointSpec(cam.position[None, :],
                                           np.array([geo.CURRENT_COLOR]),
                                           self.marker_spin.value() * 1.8))
        if self.frustum_check.isChecked() and vp.follow is None:
            corners, outline = fv.frustum_ground(cam, self.caster)
            seg, col = geo.frustum_segments(cam.position, corners,
                                            geo.lift(outline, 0.2))
            vp.set_lines("frustum", LineSpec(seg, col,
                                             width=max(1.0, self.line_spin.value() * 0.75)))
        else:
            vp.set_lines("frustum", None)
        self._update_annotations()
        if self.marker is not None:
            vp.set_points("marker", PointSpec(self.marker[None, :],
                                              np.array([[0.2, 0.9, 1.0, 1.0]]),
                                              self.marker_spin.value() * 1.6))

    def _refresh_layers(self):
        from .viewer3d.specs import Layer
        if self.viewport is None or self.scene is None:
            return
        layers = []
        alpha = self.opacity_slider.value() / 100.0
        mode = self.drape_combo.currentData()
        ortho = self._ortho.get(self.modality)
        if self.ortho_check.isChecked() and ortho is not None:
            layers.append(Layer(("ortho", self.modality), ortho, alpha))
        if mode == DRAPE_FRAME:
            if self._trail_active() and self._trail_ready \
                    and self.trail_show_check.isChecked():
                layers.append(Layer(self.viewport.CANVAS_KEY,
                                    self._trail_matrix, alpha))
            layers.append(Layer(self._frame_key(self.index),
                                self.scene.camera(self.index).shot_matrix, alpha))
        elif mode == DRAPE_ALFS:
            layers += self.alfs_panel.layers(alpha)
        layers += self.corrections_panel.layers()
        self.viewport.set_layers(layers)

    # ---------------------------------------------------------- orthomosaic

    def _update_ortho_check(self):
        """Enable the checkbox when this camera has an orthomosaic."""
        from .core.flight_viewer_io import find_orthomosaic
        path = find_orthomosaic(self.folder, self.modality) if self.folder else ""
        self.ortho_check.setEnabled(bool(path))
        if not path:
            self.ortho_info.setText(
                "No orthomosaic for this camera yet - run \"Export GeoTIFFs\" "
                "and \"Orthomosaic\" in the BAMBI panel.")
        else:
            self.ortho_info.setText(self._ortho_text.get(self.modality, ""))
        if path and self.ortho_check.isChecked():
            self._load_ortho()

    def _on_ortho_toggled(self, on):
        if on:
            self._load_ortho()
        self._refresh_layers()

    def _load_ortho(self):
        from .core.flight_viewer_io import find_orthomosaic
        if self.viewport is None or self.modality in self._ortho \
                or self._ortho_worker is not None:
            return
        path = find_orthomosaic(self.folder, self.modality)
        if not path:
            return
        self.ortho_info.setText("Loading the orthomosaic…")
        self._ortho_worker = _OrthoWorker(self.modality, path, 8192, self)
        self._ortho_worker.done.connect(self._on_ortho_loaded)
        self._ortho_worker.failed.connect(self._on_ortho_failed)
        self._ortho_worker.start()

    def _finish_ortho_worker(self):
        if self._ortho_worker is not None:
            self._ortho_worker.wait(2000)
        self._ortho_worker = None

    def _on_ortho_loaded(self, modality, data):
        self._finish_ortho_worker()
        rgba, lower, upper = data
        origin = self.origin()
        lo = np.asarray(lower, float) - origin[:2]      # DEM-local metres
        hi = np.asarray(upper, float) - origin[:2]
        if self.viewport is not None:
            self.viewport.set_texture(("ortho", modality), rgba)
        self._ortho[modality] = geo.ortho_matrix_for_bounds(lo, hi)
        self._ortho_text[modality] = (
            f"Orthomosaic {rgba.shape[1]}×{rgba.shape[0]} px, "
            f"{(hi[0] - lo[0]) / rgba.shape[1]:.2f} m per pixel")
        if modality == self.modality:
            self.ortho_info.setText(self._ortho_text[modality])
            self._refresh_layers()
        else:
            self._update_ortho_check()       # the camera changed meanwhile

    def _on_ortho_failed(self, modality, message):
        self._finish_ortho_worker()
        if modality == self.modality:
            self.ortho_info.setText("Could not read the orthomosaic: " + message)

    # ---------------------------------------------------------- annotations

    def _track_states(self):
        history = self.history_spin.value() or None
        return fv.tracks_at(self.tracks.get(self.modality, {}), self.index,
                            history_frames=history)

    def _seen_jobs(self):
        """(frame, boxes) whose outlines the background worker projects."""
        mode = self.annotation_combo.currentData()
        k = self.seen_step_spin.value()
        by_frame: Dict[int, list] = {}
        if mode == ANNOTATION_BOXES:
            for d in self.detections.get(self.modality, []):
                f = int(d["frame"])
                if f % k == 0:
                    by_frame.setdefault(f, []).append(d)
        elif mode == ANNOTATION_TRACKS:
            for track_id, boxes in self.tracks.get(self.modality, {}).items():
                frames = sorted(int(b["frame"]) for b in boxes)
                keep = set(frames[::k]) | {frames[-1]}
                for b in boxes:
                    if int(b["frame"]) in keep:
                        by_frame.setdefault(int(b["frame"]), []).append(b)
        return sorted(by_frame.items())

    def _restart_outlines(self):
        """Forget projected outlines and project the seen boxes again."""
        self._outline_generation += 1
        self._outlines = {}
        if self._outline_worker is not None:
            self._outline_worker.stop()
            self._outline_worker.wait(2000)
            self._outline_worker = None
        if self.scene is None or self.scene.image_size is None \
                or self.annotation_combo.currentData() == ANNOTATION_NONE \
                or self.caster is None:
            return
        jobs = self._seen_jobs()
        if not jobs:
            return
        self._outline_worker = _OutlineWorker(
            self._outline_generation, jobs, self.scene.images,
            dict(self.scene.correction), self.scene.aspect,
            self.scene.image_size, self.caster, self)
        self._outline_worker.chunk.connect(self._on_outlines)
        self._outline_worker.progress.connect(self._on_outline_progress)
        self._outline_worker.start()

    def _on_outlines(self, generation, rings):
        if generation != self._outline_generation:
            return
        self._outlines.update(rings)
        self._update_annotations()

    def _on_outline_progress(self, generation, fraction):
        if generation != self._outline_generation:
            return
        if fraction >= 1.0:
            self._outline_worker = None
            self._on_annotation_info()
        else:
            self.ann_info.setText(f"Projecting earlier boxes onto the DEM… "
                                  f"{fraction * 100:.0f} %")

    def _outline(self, frame, box):
        key = _outline_key(frame, box)
        ring = self._outlines.get(key)
        if ring is None and frame == self.index:     # current frame: right now
            self._outlines.update(_project_outlines(
                self.scene.camera(frame), self.scene.image_size, frame, [box],
                self.caster))
            ring = self._outlines.get(key)
        return ring

    def _ground_point(self, frame: int, box: dict) -> Optional[np.ndarray]:
        key = (self.modality, frame,
               round(box["x1"], 1), round(box["y1"], 1),
               round(box["x2"], 1), round(box["y2"], 1))
        if key not in self._ground_cache:
            pt = None
            if self.scene.image_size is not None and 0 <= frame < self._count():
                cam = self.scene.camera(frame)
                w, h = self.scene.image_size
                ray = fv.pixel_rays(cam, w, h, fv.box_centers([box]))
                hit = fv.cast_rays(self.caster, cam.position[None, :], ray)[0]
                pt = hit if np.all(np.isfinite(hit)) else None
            self._ground_cache[key] = pt
        return self._ground_cache[key]

    @staticmethod
    def _dash_length(ring):
        pts = ring[np.all(np.isfinite(ring), axis=1)]
        if len(pts) < 2:
            return 0.5
        perimeter = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0),
                                   axis=1).sum()
        return max(0.05, perimeter / 24.0)

    def _update_annotations(self):
        """Box outlines (solid now, dashed earlier) and track lines in 3D."""
        from .viewer3d.specs import LineSpec
        vp = self.viewport
        if vp is None or self.scene is None:
            return
        mode = self.annotation_combo.currentData()
        if mode == ANNOTATION_NONE or self.scene.image_size is None:
            for name in ("ann_now", "ann_seen", "tracks"):
                vp.set_lines(name, None)
            return
        width = self.ann_width_spin.value()
        keep_seen = self.seen_check.isChecked()
        now_s, now_c, seen_s, seen_c = [], [], [], []

        def add(ring, rgb, current):
            if ring is None:
                return
            if current:
                s, c = geo.ring_segments(ring, (*rgb, 1.0))
                now_s.append(s), now_c.append(c)
            else:
                s, c = geo.dashed_segments(ring, (*rgb, 0.9),
                                           self._dash_length(ring))
                seen_s.append(s), seen_c.append(c)

        if mode == ANNOTATION_BOXES:
            yellow = np.array(geo.DETECTION_COLOR) / 255.0
            for d in fv.boxes_on_frame(self.detections.get(self.modality, []),
                                       self.index):
                add(self._outline(self.index, d), yellow, True)
            if keep_seen:
                for key, ring in self._outlines.items():
                    if key[0] < self.index:
                        add(ring, yellow, False)
            self.viewport.set_lines("tracks", None)
        else:
            tracks = self.tracks.get(self.modality, {})
            ground = {}
            states = fv.tracks_at(tracks, self.index,
                                  history_frames=self.history_spin.value() or None,
                                  linger_frames=10 ** 9 if keep_seen else 0)
            for state in states:
                rgb = np.array(geo.track_color(state.track_id)) / 255.0
                if state.current is not None:
                    add(self._outline(self.index, state.current), rgb, True)
                elif keep_seen:
                    # The newest projected box of the track up to now.
                    for b in reversed(state.history):
                        ring = self._outlines.get(_outline_key(b["frame"], b))
                        if ring is not None:
                            add(ring, rgb, False)
                            break
                pts = [self._ground_point(int(b["frame"]), b)
                       for b in state.history]
                pts = [p for p in pts if p is not None]
                if len(pts) >= 2:
                    ground[state.track_id] = np.array(pts)
            seg, col = geo.track_segments(ground)
            vp.set_lines("tracks", LineSpec(seg, col, width=self.line_spin.value()))

        def lines(name, segs, cols):
            if segs:
                vp.set_lines(name, LineSpec(np.concatenate(segs),
                                            np.concatenate(cols), width=width))
            else:
                vp.set_lines(name, None)
        lines("ann_now", now_s, now_c)
        lines("ann_seen", seen_s if keep_seen else [], seen_c)

    # ------------------------------------------------------------- picking

    def _on_hover(self, point):
        if point is None:
            self.info_label.setText("")
            return
        world = np.asarray(point) + self.origin()
        text = f"E {world[0]:.2f}  N {world[1]:.2f}  Z {world[2]:.2f}"
        if getattr(self, "_last_hit_kind", None) == "surface" and \
                self.caster is not None:
            p = np.asarray(point, dtype=float)
            g = fv.cast_rays(self.caster, [[p[0], p[1], p[2] + 1e4]],
                             [[0.0, 0.0, -1.0]])[0]
            above = f", {p[2] - g[2]:.2f} m above DEM" if np.isfinite(g[2]) else ""
            text += f"  (surface{above})"
        epsg = self.config.get("target_epsg")
        if epsg:
            text += f"  (EPSG:{epsg})"
            ll = self._to_wgs84(world[0], world[1], epsg)
            if ll is not None:
                text += f"  {ll[1]:.6f}, {ll[0]:.6f}"
        if self.marker is not None:
            d = np.linalg.norm(np.asarray(point) - self.marker)
            text += f"  |  {d:.2f} m to marker"
        self.info_label.setText(text)

    def _to_wgs84(self, x, y, epsg):
        try:
            from qgis.core import (QgsCoordinateReferenceSystem,
                                   QgsCoordinateTransform, QgsPointXY, QgsProject)
            if getattr(self, "_xform_epsg", None) != epsg:
                self._xform = QgsCoordinateTransform(
                    QgsCoordinateReferenceSystem(f"EPSG:{epsg}"),
                    QgsCoordinateReferenceSystem("EPSG:4326"),
                    QgsProject.instance())
                self._xform_epsg = epsg
            p = self._xform.transform(QgsPointXY(x, y))
            return p.x(), p.y()
        except Exception:  # noqa: BLE001 - coordinates without lat/lon
            return None

    def _camera_at(self, x, y) -> Optional[int]:
        """Frame whose camera position is drawn at widget pixel (x, y)."""
        if self.viewport is None or self.scene is None or not len(self.scene):
            return None
        view, proj = self.viewport.view_projection()
        tolerance = max(8.0, self.marker_spin.value() * 0.8)
        return geo.nearest_screen_point(
            self.scene.positions(), proj @ view, self.viewport.width(),
            self.viewport.height(), x, y, tolerance)

    def _on_click(self, x, y):
        """A click on (or next to) a camera position jumps to that frame."""
        index = self._camera_at(x, y)
        if index is not None:
            self.set_frame(index)
            self.status(f"Frame {index}")

    def _on_move(self, x, y):
        index = self._camera_at(x, y)
        cursor = Qt.CursorShape.PointingHandCursor if index is not None \
            else Qt.CursorShape.ArrowCursor
        self.viewport.setCursor(cursor)
        if index is not None:
            self.viewport.setToolTip(f"Frame {index} - click to jump there")
        else:
            self.viewport.setToolTip("")

    def _on_pick(self, point):
        """Double click: set the marker the hover readout measures from."""
        from .viewer3d.specs import PointSpec
        self.marker = np.asarray(point, dtype=float)
        if self.viewport is not None:
            self.viewport.set_points("marker", PointSpec(
                self.marker[None, :], np.array([[0.2, 0.9, 1.0, 1.0]]),
                self.marker_spin.value() * 1.6))

    # ----------------------------------------------------------- utilities

    def _save_screenshot(self):
        if self.viewport is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save view", os.path.join(self.folder or "",
                                            f"view_{self.index:05d}.png"),
            "PNG image (*.png)")
        if path:
            self.viewport.grabFramebuffer().save(path)
            self.status(f"Saved {path}")

    def closeEvent(self, event):
        self.stop()
        self._save_settings()
        if self._detail_worker is not None:
            self._detail_worker.wait(10000)
        if self._outline_worker is not None:
            self._outline_worker.stop()
            self._outline_worker.wait(5000)
        if self._ortho_worker is not None:
            self._ortho_worker.wait(10000)
        for panel in ("alfs_panel", "corrections_panel", "recon_panel"):
            p = getattr(self, panel, None)
            if p is not None and hasattr(p, "shutdown"):
                p.shutdown()
        if hasattr(self, "cache"):
            self.cache.shutdown()
        if self.viewport is not None:
            self.viewport.dispose()
        super().closeEvent(event)
