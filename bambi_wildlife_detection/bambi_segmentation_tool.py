# -*- coding: utf-8 -*-
"""Segmentation tool: prompt SAM3 / SAM 3.1 on the extracted frames.

A window of its own, opened from the toolbar, replacing the SAM3 tab and the
S1 step the Processing panel used to carry. It shows the extracted frames of
one camera (thermal or RGB), takes a prompt, runs a backend and draws the
masks over the frame:

* **Text prompts** - "deer", "wild boar" - find every instance of a concept.
* **Point prompts** - left click on an object, right click beside it - define
  one object each; several objects can be clicked on one frame.
* **Image mode** segments each selected frame on its own; **sequence mode**
  hands the frames to the video tracker as one clip, so an object keeps its
  identity from frame to frame (SAM3's video tracker, or SAM 3.1's
  multiplexed tracker through Meta's ``sam3`` package).
* **Backends** - Roboflow's API (text on images), ``facebook/sam3`` through
  transformers (everything, CPU or GPU), or Meta's official package (SAM 3
  and SAM 3.1, GPU only).

Results are written to ``segmentation_{t|w}/segmentation_pixel.json`` - the
same file the pipeline always wrote, so geo-referencing, the QGIS layers and
the GeoJSON export are unchanged - and the three follow-up actions live here
too: geo-reference onto the DEM, add as QGIS layers, export as GeoJSON.

The model behind the window is :mod:`core.segmentation`; this module only
draws, collects clicks and runs the work on a thread.
"""
import os
from typing import Dict, List, Optional

from qgis.PyQt.QtCore import QPointF, QRectF, QSettings, Qt, QThread, pyqtSignal
from qgis.PyQt.QtGui import (QBrush, QColor, QFont, QPainter, QPen, QPixmap,
                             QPolygonF)
from qgis.PyQt.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGraphicsEllipseItem, QGraphicsItem, QGraphicsPixmapItem,
    QGraphicsPolygonItem, QGraphicsScene, QGraphicsSimpleTextItem,
    QGraphicsView, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QRadioButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter, QTextEdit,
    QVBoxLayout, QWidget,
)

from .core import segmentation as seg
from .gui_utils import fit_to_screen, read_hf_token, write_hf_token

_SETTINGS_PREFIX = "bambi/segmentation_tool/"
_SIDE_PANEL_WIDTH = 360

_PALETTE = [
    (230, 60, 60), (40, 170, 70), (50, 110, 240), (250, 160, 30),
    (150, 60, 190), (30, 190, 190), (240, 100, 180), (150, 90, 30),
    (110, 200, 40), (90, 90, 220), (220, 200, 40), (40, 140, 130),
]

_POSITIVE_COLOR = QColor(60, 220, 60)
_NEGATIVE_COLOR = QColor(240, 60, 60)


def _prompt_colors(names) -> Dict[str, tuple]:
    return {name: _PALETTE[i % len(_PALETTE)]
            for i, name in enumerate(sorted(set(names)))}


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

class _SegCanvas(QGraphicsView):
    """Zoom / pan view of one frame with mask polygons and click markers.

    In point mode a left click is a positive click and a right click a
    negative one; both are reported as image pixel coordinates, because the
    pixmap sits unscaled at the scene origin.
    """

    pointClicked = pyqtSignal(float, float, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing
                            | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(30, 30, 30))
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._pix_item = QGraphicsPixmapItem()
        self._pix_item.setZValue(-1)
        self.scene().addItem(self._pix_item)
        self._overlay_items = []
        self._point_mode = False
        self._pan_last = None
        self._fitted = False

    # -- content -------------------------------------------------------------

    def set_frame(self, pixmap: QPixmap) -> None:
        first = self._pix_item.pixmap().isNull() or (
            self._pix_item.pixmap().size() != pixmap.size())
        self._pix_item.setPixmap(pixmap)
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        if first or not self._fitted:
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._fitted = True

    def image_size(self):
        pix = self._pix_item.pixmap()
        return (pix.width(), pix.height()) if not pix.isNull() else (0, 0)

    def clear_overlay(self) -> None:
        for item in self._overlay_items:
            self.scene().removeItem(item)
        self._overlay_items = []

    def add_polygon(self, ring, color: tuple, label: str = "") -> None:
        polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in ring])
        item = QGraphicsPolygonItem(polygon)
        pen = QPen(QColor(*color), 2)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(QBrush(QColor(color[0], color[1], color[2], 70)))
        item.setZValue(1)
        self.scene().addItem(item)
        self._overlay_items.append(item)
        if label and ring:
            self._add_text(label, ring[0][0], ring[0][1], color)

    def add_point(self, x: float, y: float, positive: bool,
                  label: str = "") -> None:
        radius = max(3.0, self.image_size()[0] / 200.0)
        color = _POSITIVE_COLOR if positive else _NEGATIVE_COLOR
        item = QGraphicsEllipseItem(x - radius, y - radius,
                                    2 * radius, 2 * radius)
        pen = QPen(QColor(255, 255, 255), 1.5)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(QBrush(color))
        item.setZValue(3)
        self.scene().addItem(item)
        self._overlay_items.append(item)
        if label:
            self._add_text(label, x + radius, y - radius,
                           (color.red(), color.green(), color.blue()))

    def _add_text(self, text: str, x: float, y: float, color: tuple) -> None:
        item = QGraphicsSimpleTextItem(text)
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        item.setFont(font)
        item.setBrush(QBrush(QColor(*color)))
        item.setPen(QPen(QColor(0, 0, 0, 160), 0.5))
        item.setPos(float(x), float(y))
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        item.setZValue(4)
        self.scene().addItem(item)
        self._overlay_items.append(item)

    def set_point_mode(self, enabled: bool) -> None:
        self._point_mode = enabled
        self.viewport().setCursor(Qt.CursorShape.CrossCursor if enabled
                                  else Qt.CursorShape.ArrowCursor)

    def fit(self) -> None:
        if not self._pix_item.pixmap().isNull():
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    # -- interaction -----------------------------------------------------------

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_last = event.pos()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if self._point_mode and event.button() in (
                Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            width, height = self.image_size()
            if width and height:
                pos = self.mapToScene(event.pos())
                x = min(max(pos.x(), 0.0), float(width - 1))
                y = min(max(pos.y(), 0.0), float(height - 1))
                self.pointClicked.emit(
                    x, y, event.button() == Qt.MouseButton.LeftButton)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_last is not None:
            delta = event.pos() - self._pan_last
            self._pan_last = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton and self._pan_last is not None:
            self._pan_last = None
            self.viewport().setCursor(
                Qt.CursorShape.CrossCursor if self._point_mode
                else Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        # The right button is a negative click in point mode, never a menu.
        event.accept()


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class _SegmentationWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, request: seg.SegmentationRequest, parent=None):
        super().__init__(parent)
        self.request = request
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            results = seg.run_segmentation(
                self.request, progress_fn=self.progress.emit,
                log_fn=self.log.emit, cancel_check=lambda: self._cancel)
        except seg.CancelledError:
            self.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(results)


class _GeoRefWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            from .bambi_processing import BambiProcessor
            BambiProcessor().run_sam3_georeference(
                self.config, progress_fn=self.progress.emit,
                log_fn=self.log.emit, cancel_check=lambda: self._cancel)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit()


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class SegmentationToolDialog(QDialog):
    """Non-modal window: frames, prompts, backend, results, follow-ups."""

    def __init__(self, iface, dock_widget=None, parent=None):
        super().__init__(parent)
        self.iface = iface
        self._dock_widget = dock_widget
        self._target_folder = ""
        self._modality = "t"
        self._images: List[dict] = []
        self._frame = 0
        self._results: List[dict] = []
        self._by_frame: Dict[int, dict] = {}
        self._colors: Dict[str, tuple] = {}
        self._points: List[seg.PointPrompt] = []
        self._object_names: Dict[int, str] = {}
        self._next_object_id = 1
        self._worker = None
        self._georef_worker = None
        self._updating_ui = False

        self.setWindowTitle("Segmentation - SAM3 / SAM 3.1 on the extracted frames")
        self.setWindowFlags(Qt.WindowType.Window
                            | Qt.WindowType.WindowCloseButtonHint
                            | Qt.WindowType.WindowMinimizeButtonHint
                            | Qt.WindowType.WindowMaximizeButtonHint)
        self._setup_ui()
        fit_to_screen(self, 1280, 820)
        self._restore_settings()
        self._on_backend_changed()
        self._on_prompt_kind_changed()
        self._on_scope_changed()
        self.apply_dock_defaults()

    # ------------------------------------------------------------------ UI --

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Target folder:"))
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText(
            "Plugin output folder with frames_t / frames_w and poses_t.json / poses_w.json")
        folder_row.addWidget(self.folder_edit, 1)
        browse = QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self._on_browse_folder)
        folder_row.addWidget(browse)
        folder_row.addSpacing(8)
        folder_row.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Thermal", "t")
        self.camera_combo.addItem("RGB", "w")
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        folder_row.addWidget(self.camera_combo)
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self.load)
        folder_row.addWidget(self.load_btn)
        layout.addLayout(folder_row)

        dem_row = QHBoxLayout()
        dem_row.addWidget(QLabel("DEM mesh:"))
        self.dem_edit = QLineEdit()
        self.dem_edit.setPlaceholderText(
            "dem_mesh.glb - needed to geo-reference the masks (its .json beside it gives the origin)")
        dem_row.addWidget(self.dem_edit, 1)
        browse_dem = QPushButton("…")
        browse_dem.setFixedWidth(30)
        browse_dem.clicked.connect(self._on_browse_dem)
        dem_row.addWidget(browse_dem)
        layout.addLayout(dem_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        # Left: canvas + navigation
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = _SegCanvas()
        self.canvas.pointClicked.connect(self._on_canvas_point)
        left_layout.addWidget(self.canvas, 1)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(lambda: self._goto_frame(self._frame - 1))
        nav.addWidget(self.prev_btn)
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 0)
        self.frame_spin.valueChanged.connect(self._on_frame_spin)
        nav.addWidget(self.frame_spin)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_frame_slider)
        nav.addWidget(self.frame_slider, 1)
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(lambda: self._goto_frame(self._frame + 1))
        nav.addWidget(self.next_btn)
        fit_btn = QPushButton("Fit")
        fit_btn.setToolTip("Fit the frame into the view")
        fit_btn.clicked.connect(self.canvas.fit)
        nav.addWidget(fit_btn)
        left_layout.addLayout(nav)

        self.frame_label = QLabel("No frames loaded.")
        self.frame_label.setStyleSheet("font-family: monospace;")
        left_layout.addWidget(self.frame_label)
        splitter.addWidget(left)

        # Right: side panel
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(4, 0, 0, 0)
        side_layout.addWidget(self._build_model_group())
        side_layout.addWidget(self._build_prompt_group())
        side_layout.addWidget(self._build_scope_group())
        side_layout.addWidget(self._build_run_group())
        side_layout.addWidget(self._build_results_group())
        side_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setMinimumWidth(_SIDE_PANEL_WIDTH)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(90)
        self.log_edit.setStyleSheet("font-family: monospace; font-size: 10px;")
        layout.addWidget(self.log_edit)

        self.status_label = QLabel("Choose a target folder and press Load.")
        self.status_label.setStyleSheet("color: #888;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _build_model_group(self) -> QGroupBox:
        group = QGroupBox("Model")
        form = QFormLayout(group)

        self.backend_combo = QComboBox()
        for key, label, _caps in seg.BACKENDS:
            self.backend_combo.addItem(label, key)
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        form.addRow("Backend:", self.backend_combo)

        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText(seg.sam3_local.DEFAULT_SAM3_REPO)
        self.model_edit.setToolTip(
            "Hugging Face repository of the SAM3 checkpoint. Leave empty for "
            f"{seg.sam3_local.DEFAULT_SAM3_REPO}; pin a revision with "
            "'repo@commit'.")
        self.model_row_label = QLabel("Model:")
        form.addRow(self.model_row_label, self.model_edit)

        self.version_combo = QComboBox()
        for version in seg.META_VERSIONS:
            self.version_combo.addItem(
                "SAM 3.1 (facebook/sam3.1, multiplexed video tracking)"
                if version == "sam3.1" else "SAM 3 (facebook/sam3)", version)
        self.version_row_label = QLabel("Version:")
        form.addRow(self.version_row_label, self.version_combo)

        checkpoint_row = QHBoxLayout()
        self.checkpoint_edit = QLineEdit()
        self.checkpoint_edit.setPlaceholderText(
            "optional local .pt - else downloaded from Hugging Face")
        checkpoint_row.addWidget(self.checkpoint_edit, 1)
        checkpoint_browse = QPushButton("…")
        checkpoint_browse.setFixedWidth(30)
        checkpoint_browse.clicked.connect(self._on_browse_checkpoint)
        checkpoint_row.addWidget(checkpoint_browse)
        self.checkpoint_row_widget = QWidget()
        self.checkpoint_row_widget.setLayout(checkpoint_row)
        self.checkpoint_row_label = QLabel("Checkpoint:")
        form.addRow(self.checkpoint_row_label, self.checkpoint_row_widget)

        key_row = QHBoxLayout()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("Roboflow API key")
        key_row.addWidget(self.api_key_edit, 1)
        self.show_key_check = QCheckBox("show")
        self.show_key_check.toggled.connect(
            lambda on: self.api_key_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        key_row.addWidget(self.show_key_check)
        self.api_key_row_widget = QWidget()
        self.api_key_row_widget.setLayout(key_row)
        self.api_key_row_label = QLabel("API key:")
        form.addRow(self.api_key_row_label, self.api_key_row_widget)

        token_row = QHBoxLayout()
        self.hf_token_edit = QLineEdit()
        self.hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_edit.setPlaceholderText(
            "hf_… read token (or leave empty for HF_TOKEN / hf auth login)")
        self.hf_token_edit.setToolTip(
            "Hugging Face read token with access to the gated SAM3 "
            "repository. The same token as in the Classification tab - "
            "entering it here enters it there, and the other way round.")
        self.hf_token_edit.editingFinished.connect(self._on_hf_token_edited)
        token_row.addWidget(self.hf_token_edit, 1)
        self.show_token_check = QCheckBox("show")
        self.show_token_check.toggled.connect(
            lambda on: self.hf_token_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        token_row.addWidget(self.show_token_check)
        self.hf_token_row_widget = QWidget()
        self.hf_token_row_widget.setLayout(token_row)
        self.hf_token_row_label = QLabel("HF token:")
        form.addRow(self.hf_token_row_label, self.hf_token_row_widget)

        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto (GPU when available)", "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("CUDA GPU", "cuda")
        self.device_row_label = QLabel("Device:")
        form.addRow(self.device_row_label, self.device_combo)

        access_row = QHBoxLayout()
        self.check_access_btn = QPushButton("Check access")
        self.check_access_btn.setToolTip(
            "Ask Hugging Face whether the token from the Classification tab "
            "may read the gated checkpoint")
        self.check_access_btn.clicked.connect(self._check_access)
        access_row.addWidget(self.check_access_btn)
        self.access_status = QLabel("")
        self.access_status.setWordWrap(True)
        access_row.addWidget(self.access_status, 1)
        form.addRow("", access_row)

        self.model_info = QLabel("")
        self.model_info.setWordWrap(True)
        self.model_info.setTextFormat(Qt.TextFormat.RichText)
        self.model_info.setOpenExternalLinks(True)
        self.model_info.setStyleSheet("color: gray; font-size: 10px;")
        form.addRow(self.model_info)
        return group

    def _build_prompt_group(self) -> QGroupBox:
        group = QGroupBox("Prompt")
        layout = QVBoxLayout(group)

        kind_row = QHBoxLayout()
        self.text_radio = QRadioButton("Text")
        self.text_radio.setChecked(True)
        self.text_radio.setToolTip(
            "Find every instance of a concept: one prompt per line")
        self.point_radio = QRadioButton("Points")
        self.point_radio.setToolTip(
            "Click objects on the frame: left = on the object, right = beside it")
        self.text_radio.toggled.connect(self._on_prompt_kind_changed)
        kind_row.addWidget(self.text_radio)
        kind_row.addWidget(self.point_radio)
        kind_row.addStretch()
        layout.addLayout(kind_row)

        self.text_widget = QWidget()
        text_layout = QVBoxLayout(self.text_widget)
        text_layout.setContentsMargins(0, 0, 0, 0)
        self.prompts_edit = QTextEdit()
        self.prompts_edit.setPlaceholderText("deer\nwild boar\nperson")
        self.prompts_edit.setMaximumHeight(70)
        self.prompts_edit.setToolTip("One text prompt per line")
        text_layout.addWidget(self.prompts_edit)
        layout.addWidget(self.text_widget)

        self.point_widget = QWidget()
        point_layout = QVBoxLayout(self.point_widget)
        point_layout.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Left click = on the object, right click = not the "
                      "object. One object at a time; press 'New object' "
                      "for the next one.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 10px;")
        point_layout.addWidget(hint)
        object_row = QHBoxLayout()
        object_row.addWidget(QLabel("Object:"))
        self.object_combo = QComboBox()
        self.object_combo.currentIndexChanged.connect(self._on_object_changed)
        object_row.addWidget(self.object_combo, 1)
        self.new_object_btn = QPushButton("New object")
        self.new_object_btn.clicked.connect(self._new_object)
        object_row.addWidget(self.new_object_btn)
        point_layout.addLayout(object_row)
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.object_name_edit = QLineEdit()
        self.object_name_edit.setPlaceholderText("e.g. deer 1 (optional)")
        self.object_name_edit.editingFinished.connect(self._on_object_named)
        name_row.addWidget(self.object_name_edit, 1)
        point_layout.addLayout(name_row)
        self.points_list = QListWidget()
        self.points_list.setMaximumHeight(90)
        point_layout.addWidget(self.points_list)
        points_btns = QHBoxLayout()
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected_point)
        points_btns.addWidget(remove_btn)
        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self._clear_points)
        points_btns.addWidget(clear_btn)
        point_layout.addLayout(points_btns)
        layout.addWidget(self.point_widget)
        return group

    def _build_scope_group(self) -> QGroupBox:
        group = QGroupBox("Frames")
        layout = QVBoxLayout(group)
        scope_row = QHBoxLayout()
        self.current_radio = QRadioButton("Current frame")
        self.current_radio.setChecked(True)
        self.range_radio = QRadioButton("Frame range")
        self.current_radio.toggled.connect(self._on_scope_changed)
        scope_row.addWidget(self.current_radio)
        scope_row.addWidget(self.range_radio)
        scope_row.addStretch()
        layout.addLayout(scope_row)

        self.range_widget = QWidget()
        range_form = QFormLayout(self.range_widget)
        range_form.setContentsMargins(0, 0, 0, 0)
        range_row = QHBoxLayout()
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, 999999)
        self.end_spin = QSpinBox()
        self.end_spin.setRange(0, 999999)
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 1000)
        self.step_spin.setValue(1)
        range_row.addWidget(QLabel("from"))
        range_row.addWidget(self.start_spin)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self.end_spin)
        range_row.addWidget(QLabel("step"))
        range_row.addWidget(self.step_spin)
        range_form.addRow(range_row)
        self.sequence_check = QCheckBox("Track across the sequence (video mode)")
        self.sequence_check.setToolTip(
            "Hand the frames to the video tracker as one clip: objects keep "
            "their identity between frames, and a clicked object is followed "
            "onto the other frames. Off: every frame is segmented on its own.")
        range_form.addRow(self.sequence_check)
        layout.addWidget(self.range_widget)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confidence threshold:"))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setValue(0.5)
        conf_row.addWidget(self.confidence_spin)
        conf_row.addStretch()
        layout.addLayout(conf_row)
        return group

    def _build_run_group(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.run_btn = QPushButton("▶ Run segmentation")
        self.run_btn.setStyleSheet("font-weight: bold;")
        self.run_btn.clicked.connect(self._run)
        row.addWidget(self.run_btn, 1)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        row.addWidget(self.cancel_btn)
        layout.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)
        return widget

    def _build_results_group(self) -> QGroupBox:
        group = QGroupBox("Results")
        layout = QVBoxLayout(group)
        self.results_label = QLabel("No results for this camera yet.")
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.results_label)
        self.show_masks_check = QCheckBox("Show masks on the frame")
        self.show_masks_check.setChecked(True)
        self.show_masks_check.toggled.connect(lambda _on: self._render_frame())
        layout.addWidget(self.show_masks_check)

        self.georef_btn = QPushButton("Geo-reference onto the DEM")
        self.georef_btn.setToolTip(
            "Project every mask polygon through the camera pose onto the DEM "
            "mesh, into world coordinates")
        self.georef_btn.clicked.connect(self._georeference)
        layout.addWidget(self.georef_btn)
        self.add_qgis_btn = QPushButton("Add to QGIS as layers")
        self.add_qgis_btn.clicked.connect(self._add_to_qgis)
        layout.addWidget(self.add_qgis_btn)
        self.export_btn = QPushButton("Export GeoJSON…")
        self.export_btn.setToolTip(
            "Write the geo-referenced masks as WGS84 polygons, one feature "
            "per object with prompt, frame, object id and confidence")
        self.export_btn.clicked.connect(self._export_geojson)
        layout.addWidget(self.export_btn)
        self.delete_btn = QPushButton("Delete results of this camera…")
        self.delete_btn.clicked.connect(self._delete_results)
        layout.addWidget(self.delete_btn)
        return group

    # ------------------------------------------------------------- settings --

    def _restore_settings(self):
        settings = QSettings()
        backend = settings.value(_SETTINGS_PREFIX + "backend", "", type=str)
        idx = self.backend_combo.findData(backend)
        if idx >= 0:
            self.backend_combo.setCurrentIndex(idx)
        self.model_edit.setText(
            settings.value(_SETTINGS_PREFIX + "model", "", type=str))
        version = settings.value(_SETTINGS_PREFIX + "meta_version", "", type=str)
        idx = self.version_combo.findData(version)
        if idx >= 0:
            self.version_combo.setCurrentIndex(idx)
        self.checkpoint_edit.setText(
            settings.value(_SETTINGS_PREFIX + "meta_checkpoint", "", type=str))
        self.api_key_edit.setText(
            settings.value(_SETTINGS_PREFIX + "roboflow_api_key", "", type=str))
        device = settings.value(_SETTINGS_PREFIX + "device", "", type=str)
        idx = self.device_combo.findData(device)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        self.prompts_edit.setPlainText(
            settings.value(_SETTINGS_PREFIX + "prompts", "", type=str))
        self.confidence_spin.setValue(
            settings.value(_SETTINGS_PREFIX + "confidence", 0.5, type=float))
        self.sequence_check.setChecked(
            settings.value(_SETTINGS_PREFIX + "sequence", True, type=bool))

    def _save_settings(self):
        settings = QSettings()
        settings.setValue(_SETTINGS_PREFIX + "backend",
                          self.backend_combo.currentData())
        settings.setValue(_SETTINGS_PREFIX + "model", self.model_edit.text())
        settings.setValue(_SETTINGS_PREFIX + "meta_version",
                          self.version_combo.currentData())
        settings.setValue(_SETTINGS_PREFIX + "meta_checkpoint",
                          self.checkpoint_edit.text())
        settings.setValue(_SETTINGS_PREFIX + "roboflow_api_key",
                          self.api_key_edit.text())
        settings.setValue(_SETTINGS_PREFIX + "device",
                          self.device_combo.currentData())
        settings.setValue(_SETTINGS_PREFIX + "prompts",
                          self.prompts_edit.toPlainText())
        settings.setValue(_SETTINGS_PREFIX + "confidence",
                          self.confidence_spin.value())
        settings.setValue(_SETTINGS_PREFIX + "sequence",
                          self.sequence_check.isChecked())

    # ----------------------------------------------------------------- dock --

    def _dock_config(self) -> dict:
        if self._dock_widget is None:
            return {}
        try:
            return self._dock_widget.get_config() or {}
        except Exception:  # nosec B110 - a half-built dock has no config
            return {}

    def apply_dock_defaults(self):
        """Seed folder, DEM and the Hugging Face token from the dock, then load."""
        self._sync_token_from_dock()
        config = self._dock_config()
        changed = False
        folder = config.get("target_folder", "")
        if folder and not self.folder_edit.text():
            self.folder_edit.setText(folder)
            changed = True
        dem = config.get("dem_path", "")
        if dem and not self.dem_edit.text():
            self.dem_edit.setText(dem)
        if changed and os.path.isdir(self.folder_edit.text()):
            self.load()

    # -- the Hugging Face token: one credential, two places ------------------

    def _sync_token_from_dock(self):
        """Show the token the Classification tab holds (or the stored one).

        The dock's field is authoritative while the dock exists, because a
        token typed there after this window opened must not be overwritten
        by an older stored value when the window is re-opened.
        """
        dock = self._dock_widget
        edit = getattr(dock, "hf_token_edit", None) if dock is not None else None
        try:
            token = edit.text().strip() if edit is not None else read_hf_token()
        except Exception:  # nosec B110 - a torn-down dock or no QGIS settings
            token = ""
        if token != self.hf_token_edit.text().strip():
            self.hf_token_edit.setText(token)

    def _on_hf_token_edited(self):
        """Persist the token and hand it to the Classification tab."""
        token = self.hf_token_edit.text().strip()
        dock = self._dock_widget
        if dock is not None and hasattr(dock, "set_hf_token"):
            try:
                dock.set_hf_token(token)
                return
            except Exception:  # nosec B110 - fall through to the settings
                pass
        try:
            write_hf_token(token)
        except Exception:  # nosec B110 - no QGIS settings (tests)
            pass

    def _hf_token(self) -> str:
        """The token a run should use: the field, else environment / CLI."""
        from .core import hf_access
        token, _source = hf_access.resolve_token(self.hf_token_edit.text())
        return token

    def _epsg(self) -> Optional[int]:
        epsg = self._dock_config().get("target_epsg")
        try:
            return int(epsg) if epsg else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _models_dir() -> str:
        from .bambi_processing import BambiProcessor
        return BambiProcessor._get_default_model_dir()

    # -------------------------------------------------------------- browse --

    def _on_browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select plugin output folder", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)
            self.load()

    def _on_browse_dem(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select DEM mesh", self.dem_edit.text(),
            "DEM mesh (*.glb *.gltf);;All files (*)")
        if path:
            self.dem_edit.setText(path)

    def _on_browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SAM 3 / SAM 3.1 checkpoint", self.checkpoint_edit.text(),
            "PyTorch checkpoint (*.pt *.pth);;All files (*)")
        if path:
            self.checkpoint_edit.setText(path)

    # ------------------------------------------------------------- loading --

    def _on_camera_changed(self, _index):
        self._modality = self.camera_combo.currentData() or "t"
        if self._target_folder:
            self.load()

    def load(self):
        folder = self.folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            self.status_label.setText("Choose a target folder first.")
            return
        self._target_folder = folder
        self._modality = self.camera_combo.currentData() or "t"
        camera = self.camera_combo.currentText()
        try:
            self._images = seg.pose_images(folder, self._modality)
        except (FileNotFoundError, seg.SegmentationError) as exc:
            self._images = []
            self.canvas.clear_overlay()
            self.frame_spin.setRange(0, 0)
            self.frame_slider.setRange(0, 0)
            self.frame_label.setText("No frames loaded.")
            self.status_label.setText(
                f"No {camera} frames in {folder}: {exc}")
            self._clear_points()
            self._reload_results()
            return
        last = max(0, len(self._images) - 1)
        self._updating_ui = True
        self.frame_spin.setRange(0, last)
        self.frame_slider.setRange(0, last)
        self.start_spin.setRange(0, last)
        self.end_spin.setRange(0, last)
        self.end_spin.setValue(last)
        self._updating_ui = False
        self._clear_points()
        self._reload_results()
        self._goto_frame(min(self._frame, last), force=True)
        self.status_label.setText(
            f"{camera}: {len(self._images)} frames from {folder}")

    def _reload_results(self):
        store = self._store()
        self._results = store.load_pixel() if store else []
        self._by_frame = {int(r.get("frame_idx", 0)): r for r in self._results}
        self._colors = _prompt_colors(
            p.get("prompt", "") for r in self._results
            for p in r.get("prompts", []))
        self._update_results_label()

    def _store(self) -> Optional[seg.SegmentationStore]:
        if not self._target_folder:
            return None
        return seg.SegmentationStore(self._target_folder, self._modality)

    def _update_results_label(self):
        store = self._store()
        if not self._results:
            self.results_label.setText("No results for this camera yet.")
        else:
            summary = seg.prompt_summary(self._results)
            lines = [f"{len(self._results)} frame(s), "
                     f"{seg.count_predictions(self._results)} mask(s):"]
            for name, (frames, count) in sorted(summary.items()):
                lines.append(f"  • {name}: {count} mask(s) on {frames} frame(s)")
            if store and store.has_georef():
                lines.append("Geo-referenced" + (
                    " (stale - re-run the geo-referencing)"
                    if store.georef_is_stale() else ""))
            else:
                lines.append("Not geo-referenced yet")
            self.results_label.setText("\n".join(lines))
        has = bool(self._results)
        geo = bool(store and store.has_georef())
        self.georef_btn.setEnabled(has)
        self.add_qgis_btn.setEnabled(geo)
        self.export_btn.setEnabled(geo)
        self.delete_btn.setEnabled(has or geo)

    # ---------------------------------------------------------------- frames --

    def _goto_frame(self, frame: int, force: bool = False):
        if not self._images:
            return
        frame = max(0, min(frame, len(self._images) - 1))
        if frame == self._frame and not force:
            return
        self._frame = frame
        self._updating_ui = True
        self.frame_spin.setValue(frame)
        self.frame_slider.setValue(frame)
        self._updating_ui = False
        self._render_frame()

    def _on_frame_spin(self, value):
        if not self._updating_ui:
            self._goto_frame(value)

    def _on_frame_slider(self, value):
        if not self._updating_ui:
            self._goto_frame(value)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Left:
            self._goto_frame(self._frame - 1)
        elif key == Qt.Key.Key_Right:
            self._goto_frame(self._frame + 1)
        elif key == Qt.Key.Key_PageUp:
            self._goto_frame(self._frame - 10)
        elif key == Qt.Key.Key_PageDown:
            self._goto_frame(self._frame + 10)
        elif key == Qt.Key.Key_Escape:
            event.ignore()
        else:
            super().keyPressEvent(event)

    def _frame_image_path(self, frame: int) -> str:
        if not (0 <= frame < len(self._images)):
            return ""
        path = seg.frame_path(self._target_folder, self._modality,
                              self._images[frame])
        return path if os.path.isfile(path) else ""

    def _render_frame(self):
        self.canvas.clear_overlay()
        if not self._images:
            return
        path = self._frame_image_path(self._frame)
        pixmap = QPixmap(path) if path else QPixmap()
        if pixmap.isNull():
            pixmap = QPixmap(640, 512)
            pixmap.fill(QColor(40, 40, 40))
        self.canvas.set_frame(pixmap)

        masks = 0
        entry = self._by_frame.get(self._frame)
        if entry and self.show_masks_check.isChecked():
            for prompt in entry.get("prompts", []):
                name = prompt.get("prompt", "")
                color = self._colors.get(name, (200, 200, 200))
                for prediction in prompt.get("predictions", []):
                    object_id = prediction.get("object_id",
                                               prompt.get("object_id"))
                    label = name
                    if object_id is not None and prompt.get(
                            "prompt_type") != seg.PROMPT_POINT:
                        label = f"{name} #{object_id}"
                    label += f" {prediction.get('confidence', 0):.2f}"
                    for index, ring in enumerate(prediction.get("polygons", [])):
                        if len(ring) < 3:
                            continue
                        self.canvas.add_polygon(
                            ring, color, label if index == 0 else "")
                        masks += 1

        for point in self._points:
            if point.frame_idx == self._frame:
                self.canvas.add_point(
                    point.x, point.y, point.positive,
                    seg.object_name(point.object_id, self._object_names))

        name = self._images[self._frame].get("imagefile", "")
        stamp = self._images[self._frame].get("timestamp", "") or ""
        self.frame_label.setText(
            f"frame {self._frame:5d} / {len(self._images) - 1}  {name}"
            f"{'  ' + str(stamp) if stamp else ''}  |  {masks} mask(s) shown"
            + ("" if path else "  |  image file missing"))
        self._refresh_points_list()

    # ---------------------------------------------------------------- points --

    def _on_prompt_kind_changed(self, *_args):
        points = self.point_radio.isChecked()
        self.text_widget.setVisible(not points)
        self.point_widget.setVisible(points)
        self.canvas.set_point_mode(points)
        if points and self.object_combo.count() == 0:
            self._new_object()

    def _new_object(self):
        object_id = self._next_object_id
        self._next_object_id += 1
        self.object_combo.addItem(seg.object_name(object_id), object_id)
        self.object_combo.setCurrentIndex(self.object_combo.count() - 1)

    def _current_object_id(self) -> int:
        if self.object_combo.count() == 0:
            self._new_object()
        return int(self.object_combo.currentData())

    def _on_object_changed(self, _index):
        object_id = self.object_combo.currentData()
        self.object_name_edit.setText(
            self._object_names.get(object_id, "") if object_id is not None else "")
        self._refresh_points_list()

    def _on_object_named(self):
        object_id = self.object_combo.currentData()
        if object_id is None:
            return
        name = self.object_name_edit.text().strip()
        if name:
            self._object_names[int(object_id)] = name
        else:
            self._object_names.pop(int(object_id), None)
        self.object_combo.setItemText(
            self.object_combo.currentIndex(),
            seg.object_name(int(object_id), self._object_names))
        self._render_frame()

    def _on_canvas_point(self, x: float, y: float, positive: bool):
        if not self._images:
            return
        self._points.append(seg.PointPrompt(
            self._current_object_id(), self._frame, x, y, positive))
        self._render_frame()

    def _refresh_points_list(self):
        self.points_list.clear()
        for index, point in enumerate(self._points):
            item = QListWidgetItem(
                f"{seg.object_name(point.object_id, self._object_names)}  "
                f"frame {point.frame_idx}  ({point.x:.0f}, {point.y:.0f})  "
                f"{'+' if point.positive else '−'}")
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.points_list.addItem(item)

    def _remove_selected_point(self):
        rows = sorted({item.data(Qt.ItemDataRole.UserRole)
                       for item in self.points_list.selectedItems()},
                      reverse=True)
        for row in rows:
            if 0 <= row < len(self._points):
                del self._points[row]
        self._render_frame()

    def _clear_points(self):
        self._points = []
        self._object_names = {}
        self._next_object_id = 1
        self.object_combo.blockSignals(True)
        self.object_combo.clear()
        self.object_combo.blockSignals(False)
        self.object_name_edit.clear()
        if self.point_radio.isChecked():
            self._new_object()
        if self._images:
            self._render_frame()
        else:
            self._refresh_points_list()

    # --------------------------------------------------------------- backend --

    def _on_backend_changed(self, *_args):
        key = self.backend_combo.currentData()
        caps = seg.backend_capabilities(key) if key else frozenset()
        transformers = key == seg.BACKEND_TRANSFORMERS
        meta = key == seg.BACKEND_META
        roboflow = key == seg.BACKEND_ROBOFLOW
        for widget in (self.model_row_label, self.model_edit):
            widget.setVisible(transformers)
        for widget in (self.version_row_label, self.version_combo,
                       self.checkpoint_row_label, self.checkpoint_row_widget):
            widget.setVisible(meta)
        for widget in (self.api_key_row_label, self.api_key_row_widget):
            widget.setVisible(roboflow)
        for widget in (self.hf_token_row_label, self.hf_token_row_widget):
            widget.setVisible(not roboflow)
        for widget in (self.device_row_label, self.device_combo):
            widget.setVisible(transformers)
        self.check_access_btn.setVisible(not roboflow)
        self.access_status.setVisible(not roboflow)
        self.access_status.setText("")
        self.point_radio.setEnabled(seg.CAP_POINTS in caps)
        if seg.CAP_POINTS not in caps and self.point_radio.isChecked():
            self.text_radio.setChecked(True)
        self.sequence_check.setEnabled(seg.CAP_SEQUENCE in caps)
        if transformers:
            self.model_info.setText(
                'Meta\'s <a href="https://huggingface.co/facebook/sam3">'
                "facebook/sam3</a> through transformers (>= 5.0): text and "
                "point prompts on images, and both across a sequence with "
                "the video trackers. The repository is gated - request "
                "access there and enter a read token above (shared with the "
                "Classification tab). The checkpoint (~3.4 GB) is shared with the classifiers' "
                "model cache. Works on CPU, much faster on a GPU.")
        elif meta:
            self.model_info.setText(
                'Meta\'s official <a href="https://github.com/facebookresearch/sam3">'
                "sam3</a> package - the only way to run <b>SAM 3.1</b> "
                '(<a href="https://huggingface.co/facebook/sam3.1">'
                "facebook/sam3.1</a>, ~7x faster many-object video tracking "
                "through object multiplexing; Hugging Face hosts it as a "
                "bare checkpoint without a transformers integration). Needs "
                "a CUDA GPU, Python >= 3.12 and PyTorch >= 2.7: "
                "<code>pip install git+https://github.com/facebookresearch/sam3.git</code>. "
                "Both checkpoints are gated; the token above (shared with "
                "the Classification tab) is used for the download.")
        else:
            self.model_info.setText(
                "Roboflow's hosted SAM3 endpoint: text prompts on single "
                "frames, one request per frame. No local model; needs a "
                "Roboflow API key and a network connection.")

    def _check_access(self):
        from .core import hf_access
        from .core.classification import split_revision

        key = self.backend_combo.currentData()
        if key == seg.BACKEND_META:
            repo = f"facebook/{self.version_combo.currentData()}"
        else:
            repo, _rev = split_revision(
                self.model_edit.text().strip() or seg.sam3_local.DEFAULT_SAM3_REPO)
        token = self._hf_token()
        self.access_status.setText("🟡 Checking…")
        self.check_access_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            result = hf_access.check_repo_access(repo, token)
        finally:
            self.check_access_btn.setEnabled(True)
        icons = {hf_access.ACCESS_GRANTED: "🟢", hf_access.ACCESS_GATED: "🔴",
                 hf_access.ACCESS_NO_TOKEN: "🔴", hf_access.ACCESS_MISSING: "🔴",
                 hf_access.ACCESS_UNAVAILABLE: "🟠", hf_access.ACCESS_ERROR: "🟠"}
        icon = icons.get(result["status"], "⚪")
        message = result["message"]
        if result["status"] in (hf_access.ACCESS_NO_TOKEN, hf_access.ACCESS_GATED):
            message += " Enter a read token in the HF token field above."
        self.access_status.setText(f"{icon} {message}")
        self._log(f"Hugging Face access check for {repo}: {result['status']} - "
                  f"{result['message']}")

    # ------------------------------------------------------------------ run --

    def _on_scope_changed(self, *_args):
        self.range_widget.setEnabled(self.range_radio.isChecked())

    def _selected_frames(self) -> List[int]:
        if not self._images:
            return []
        if self.current_radio.isChecked():
            frames = [self._frame]
        else:
            frames = seg.select_frames(
                len(self._images), False, self.start_spin.value(),
                self.end_spin.value(), self.step_spin.value())
        if self.point_radio.isChecked():
            # A clicked frame is always part of the run, whatever the step.
            clicked = {p.frame_idx for p in self._points}
            frames = sorted(set(frames) | clicked)
        return frames

    def _build_request(self) -> seg.SegmentationRequest:
        key = self.backend_combo.currentData()
        texts = []
        points = []
        if self.text_radio.isChecked():
            texts = [line.strip() for line in
                     self.prompts_edit.toPlainText().splitlines() if line.strip()]
        else:
            points = list(self._points)
        mode = (seg.MODE_SEQUENCE
                if self.range_radio.isChecked() and self.sequence_check.isChecked()
                and self.sequence_check.isEnabled() else seg.MODE_IMAGE)
        model = (self.version_combo.currentData() if key == seg.BACKEND_META
                 else self.model_edit.text().strip())
        return seg.SegmentationRequest(
            target_folder=self._target_folder, modality=self._modality,
            backend=key, mode=mode, frames=self._selected_frames(),
            texts=texts, points=points, object_names=dict(self._object_names),
            confidence=self.confidence_spin.value(), model=model,
            checkpoint_path=self.checkpoint_edit.text().strip(),
            api_key=self.api_key_edit.text().strip(),
            hf_token=self._hf_token(),
            device=self.device_combo.currentData() or "auto",
            models_dir=self._models_dir())

    def _run(self):
        if self._worker is not None:
            return
        if not self._images:
            QMessageBox.warning(self, "Segmentation",
                                "Load a target folder with extracted frames first.")
            return
        self._save_settings()
        self._on_hf_token_edited()
        request = self._build_request()
        try:
            request.validate()
        except seg.SegmentationError as exc:
            QMessageBox.warning(self, "Segmentation", str(exc))
            return
        if (request.mode == seg.MODE_SEQUENCE and len(request.frames) > 300
                and request.backend == seg.BACKEND_TRANSFORMERS):
            reply = QMessageBox.question(
                self, "Long sequence",
                f"The clip has {len(request.frames)} frames. The video "
                "tracker keeps every frame in memory; a few hundred frames "
                "at 1008 px take several GB. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._set_running(True)
        self._worker = _SegmentationWorker(request, self)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.log.connect(self._log)
        self._worker.finished_ok.connect(self._on_run_finished)
        self._worker.failed.connect(self._on_run_failed)
        self._worker.cancelled.connect(self._on_run_cancelled)
        self._worker.finished.connect(self._on_worker_done)
        self._run_prompt_names = request.prompt_names()
        self._log(f"Running on {len(request.frames)} frame(s)…")
        self._worker.start()

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self.status_label.setText("Cancelling…")
        if self._georef_worker is not None:
            self._georef_worker.cancel()

    def _set_running(self, running: bool):
        for widget in (self.run_btn, self.load_btn, self.camera_combo,
                       self.georef_btn, self.add_qgis_btn, self.export_btn,
                       self.delete_btn, self.backend_combo):
            widget.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        if running:
            self.progress.setValue(0)
        else:
            self._update_results_label()

    def _on_run_finished(self, results: list):
        store = self._store()
        if store is None:
            return
        merged = seg.merge_results(store.load_pixel(), results,
                                   self._run_prompt_names)
        store.save_pixel(merged)
        if store.has_georef():
            self._log("The geo-referenced file is now older than the masks - "
                      "re-run 'Geo-reference onto the DEM'.")
        self._reload_results()
        if self.point_radio.isChecked():
            # The clicks became masks; a fresh object for the next round.
            self._clear_points()
        self._render_frame()
        found = seg.count_predictions(results)
        self.status_label.setText(
            f"Done: {found} mask(s) on {len(results)} frame(s), saved to "
            f"{store.pixel_path}")

    def _on_run_failed(self, message: str):
        self._log(f"Error: {message}")
        self.status_label.setText("Segmentation failed - see the log.")
        QMessageBox.critical(self, "Segmentation failed", message)

    def _on_run_cancelled(self):
        self._log("Cancelled.")
        self.status_label.setText("Cancelled.")

    def _on_worker_done(self):
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_running(False)

    # ---------------------------------------------------------- follow-ups --

    def _georeference(self):
        store = self._store()
        if store is None or not store.has_pixel():
            QMessageBox.warning(self, "Geo-reference", "Run a segmentation first.")
            return
        if self._georef_worker is not None:
            return
        config = dict(self._dock_config())
        dem = self.dem_edit.text().strip() or config.get("dem_path", "")
        if not dem or not os.path.isfile(dem):
            QMessageBox.warning(
                self, "Geo-reference",
                "Choose the DEM mesh (.glb / .gltf) first - the masks are "
                "projected through the camera pose onto it.")
            return
        config["dem_path"] = dem
        config["target_folder"] = self._target_folder
        config["sam3_camera"] = "T" if self._modality == "t" else "W"
        self._set_running(True)
        self._log("Geo-referencing the masks onto the DEM…")
        self._georef_worker = _GeoRefWorker(config, self)
        self._georef_worker.progress.connect(self.progress.setValue)
        self._georef_worker.log.connect(self._log)
        self._georef_worker.finished_ok.connect(self._on_georef_finished)
        self._georef_worker.failed.connect(self._on_run_failed)
        self._georef_worker.finished.connect(self._on_georef_done)
        self._georef_worker.start()

    def _on_georef_finished(self):
        self._reload_results()
        self.status_label.setText("Geo-referenced. Add the layers to QGIS or "
                                  "export them as GeoJSON.")

    def _on_georef_done(self):
        worker = self._georef_worker
        self._georef_worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_running(False)

    def _export_geojson(self):
        store = self._store()
        if store is None or not store.has_georef():
            QMessageBox.warning(self, "Export GeoJSON",
                                "Geo-reference the masks first.")
            return
        epsg = self._epsg()
        if not epsg:
            QMessageBox.warning(
                self, "Export GeoJSON",
                "GeoJSON publishes latitude/longitude, so the target CRS "
                "must be set in the plugin panel before exporting.")
            return
        camera = "thermal" if self._modality == "t" else "rgb"
        suggested = os.path.join(self._target_folder,
                                 f"segmentations_{camera}.geojson")
        output, _filter = QFileDialog.getSaveFileName(
            self, "Export segmentation GeoJSON", suggested, "GeoJSON (*.geojson)")
        if not output:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            counts = seg.export_geojson(
                store.load_georef(), output, epsg, self._modality,
                log_fn=self._log)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Export GeoJSON", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.status_label.setText(
            f"GeoJSON written: {counts['features']} polygon(s) to {output}")

    def _delete_results(self):
        store = self._store()
        if store is None:
            return
        reply = QMessageBox.question(
            self, "Delete results",
            f"Delete the {self.camera_combo.currentText()} segmentation "
            f"results in\n{store.folder}\n(the pixel masks and the "
            "geo-referenced file)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        for path in (store.pixel_path, store.georef_path):
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError as exc:
                self._log(f"Could not delete {path}: {exc}")
        self._reload_results()
        self._render_frame()
        self.status_label.setText("Results deleted.")

    # ---------------------------------------------------------------- QGIS --

    def _add_to_qgis(self):
        """One group per frame, one polygon layer per prompt, per object id."""
        store = self._store()
        if store is None or not store.has_georef():
            QMessageBox.warning(self, "Add to QGIS", "Geo-reference the masks first.")
            return
        epsg = self._epsg()
        if not epsg:
            QMessageBox.warning(self, "Add to QGIS",
                                "Set the target CRS in the plugin panel first.")
            return
        from qgis.PyQt.QtCore import QVariant
        from qgis.core import (QgsCoordinateReferenceSystem, QgsFeature,
                               QgsField, QgsFillSymbol, QgsGeometry,
                               QgsLayerTreeGroup, QgsPointXY, QgsProject,
                               QgsSingleSymbolRenderer, QgsVectorLayer)

        georef = store.load_georef()
        if not georef:
            QMessageBox.warning(self, "Add to QGIS", "No geo-referenced masks.")
            return
        frames = sorted(int(r.get("frame_idx", 0)) for r in georef)
        if len(frames) > 50:
            reply = QMessageBox.question(
                self, "Many frames",
                f"{len(frames)} frames carry masks. One layer group per "
                "frame may slow QGIS down. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes)
            if reply != QMessageBox.StandardButton.Yes:
                return

        camera_label = "Thermal" if self._modality == "t" else "RGB"
        group_name = f"SAM3 Segmentation ({camera_label})"
        colors = _prompt_colors(p.get("prompt", "") for r in georef
                                for p in r.get("prompts", []))
        crs = QgsCoordinateReferenceSystem(f"EPSG:{epsg}")
        dock = self._dock_widget
        if dock is not None and hasattr(dock, "_remove_layer_group"):
            dock._remove_layer_group(group_name)
            main_group = dock._create_layer_group(group_name)
        else:
            root = QgsProject.instance().layerTreeRoot()
            main_group = QgsLayerTreeGroup(group_name)
            root.insertChildNode(0, main_group)

        def style(layer, color):
            rgb = f"{color[0]},{color[1]},{color[2]}"
            symbol = QgsFillSymbol.createSimple({
                "color": f"{rgb},80", "outline_color": f"{rgb},255",
                "outline_width": "0.8"})
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            layer.triggerRepaint()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        total = 0
        try:
            for entry in georef:
                frame_idx = int(entry.get("frame_idx", 0))
                prompts = [p for p in entry.get("prompts", [])
                           if p.get("predictions")]
                if not prompts:
                    continue
                frame_group = main_group.addGroup(f"Frame {frame_idx:04d}")
                for prompt in prompts:
                    name = prompt.get("prompt", "unknown")
                    layer = QgsVectorLayer(f"Polygon?crs={crs.authid()}",
                                           name, "memory")
                    provider = layer.dataProvider()
                    provider.addAttributes([
                        QgsField("prompt", QVariant.String),
                        QgsField("prompt_type", QVariant.String),
                        QgsField("object_id", QVariant.Int),
                        QgsField("frame", QVariant.Int),
                        QgsField("confidence", QVariant.Double),
                        QgsField("polygon_idx", QVariant.Int),
                    ])
                    layer.updateFields()
                    features = []
                    for prediction in prompt.get("predictions", []):
                        object_id = prediction.get("object_id",
                                                   prompt.get("object_id"))
                        for poly_idx, polygon in enumerate(
                                prediction.get("world_polygons", [])):
                            if len(polygon) < 3:
                                continue
                            points = [QgsPointXY(p[0], p[1]) for p in polygon]
                            if points[0] != points[-1]:
                                points.append(points[0])
                            feature = QgsFeature()
                            feature.setGeometry(QgsGeometry.fromPolygonXY([points]))
                            feature.setAttributes([
                                name, prompt.get("prompt_type", seg.PROMPT_TEXT),
                                int(object_id) if object_id is not None else None,
                                frame_idx,
                                round(float(prediction.get("confidence", 0)), 4),
                                poly_idx])
                            features.append(feature)
                    if not features:
                        continue
                    provider.addFeatures(features)
                    layer.updateExtents()
                    color = colors.get(name, (120, 120, 120))
                    style(layer, color)
                    if dock is not None and hasattr(dock, "_persist_memory_layer"):
                        layer = dock._persist_memory_layer(
                            layer, f"SAM3_{camera_label}_Frame{frame_idx:04d}_{name}",
                            "sam3_layers")
                        style(layer, color)
                    QgsProject.instance().addMapLayer(layer, False)
                    frame_group.addLayer(layer)
                    total += len(features)
                frame_group.setExpanded(False)
            main_group.setExpanded(True)
        finally:
            QApplication.restoreOverrideCursor()
        if total:
            self.iface.mapCanvas().refresh()
        self._log(f"Added {total} polygon(s) to QGIS under '{group_name}'")
        self.status_label.setText(f"{total} polygon(s) added to QGIS.")

    # ------------------------------------------------------------------ misc --

    def _log(self, message: str):
        self.log_edit.append(message)
        if self._dock_widget is not None and hasattr(self._dock_widget, "log"):
            try:
                self._dock_widget.log(f"[Segmentation] {message}")
            except Exception:  # nosec B110
                pass

    def showEvent(self, event):
        self._sync_token_from_dock()
        super().showEvent(event)

    def closeEvent(self, event):
        if self._worker is not None or self._georef_worker is not None:
            reply = QMessageBox.question(
                self, "Segmentation running",
                "A run is still in progress. Cancel it and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._cancel()
            for worker in (self._worker, self._georef_worker):
                if worker is not None:
                    worker.wait(5000)
        self._save_settings()
        super().closeEvent(event)
