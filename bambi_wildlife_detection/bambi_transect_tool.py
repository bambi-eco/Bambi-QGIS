# -*- coding: utf-8 -*-
"""
BAMBI Transect Splitting Tool
=============================

Non-modal dialog for splitting a processed flight into transects — named,
contiguous frame ranges of the extracted frames.

Features
--------
* Frame viewer with the same navigation as the labelling tool (prev/next,
  step buttons, spin box, slider, arrow keys / PageUp / PageDown).
* Overview map: the flight route (per-frame camera ground positions from the
  poses file) with the current frame's position highlighted; the defined
  transects are drawn in their colours. Clicking the map jumps to the nearest
  frame.
* Transects are defined by a start and an end frame (set to the current
  frame with a button or shortcut). The length between them is measured
  along the flight-path polyline (metres, from the metric mesh-local pose
  coordinates).
* Instead of picking the end frame manually, "end after distance" places the
  end at the first frame at least X metres along the flight path.
* An optional name can be given; unnamed transects show ``Transect {id}``
  with a counting-up id.
* Saving writes ``transects_{m}/transects.json`` (+ ``transects.csv``) to
  the output folder; existing definitions are loaded automatically.
"""

import os
import json
from typing import List, Optional, Tuple

from qgis.PyQt.QtCore import Qt, QPointF, QRectF, pyqtSignal
from qgis.PyQt.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QWidget, QGroupBox,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QSizePolicy, QScrollArea, QFrame,
)

from .gui_utils import fit_to_screen
from .core.labelling import track_color_rgb
from .core.transects import (
    Transect,
    TransectStore,
    cumulative_distances,
    flight_positions,
    frame_after_distance,
    path_length,
)

#: Width of the controls next to the canvas (the scroll area adds the bar).
_SIDE_PANEL_WIDTH = 300

_CURRENT_COLOR = QColor(255, 220, 0)     # current frame marker
_ROUTE_COLOR = QColor(150, 150, 150)     # flight route polyline


def _transect_color(transect_id: int) -> QColor:
    return QColor(*track_color_rgb(transect_id))


# ---------------------------------------------------------------------------
# Frame view
# ---------------------------------------------------------------------------

class _FrameView(QGraphicsView):
    """Zoomable / pannable graphics view holding the current frame image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(30, 30, 30))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Middle-button panning like the labelling canvas; left drag scrolls.
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        self._pix_item = QGraphicsPixmapItem()
        self.scene().addItem(self._pix_item)
        self._fitted = False

    def set_frame(self, pixmap: QPixmap) -> None:
        first = self._pix_item.pixmap().isNull() or (
            self._pix_item.pixmap().size() != pixmap.size())
        self._pix_item.setPixmap(pixmap)
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        if first or not self._fitted:
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._fitted = True

    def fit(self) -> None:
        if not self._pix_item.pixmap().isNull():
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)


# ---------------------------------------------------------------------------
# Overview map
# ---------------------------------------------------------------------------

class _FlightMapWidget(QWidget):
    """Small overview map: flight route, transects and the current position.

    Draws in the mesh-local metric coordinates of the poses file (north up,
    so the y axis is flipped for painting). Clicking emits the frame whose
    camera position is nearest to the click.
    """

    frameClicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._positions: List[Optional[Tuple[float, float]]] = []
        self._current = 0
        self._transects: List[Transect] = []
        self._selected_id: Optional[int] = None

    def set_route(self, positions: List[Optional[Tuple[float, float]]]) -> None:
        self._positions = positions
        self.update()

    def set_state(self, current: int, transects: List[Transect],
                  selected_id: Optional[int]) -> None:
        self._current = current
        self._transects = transects
        self._selected_id = selected_id
        self.update()

    # -- coordinate mapping ------------------------------------------------

    def _world_to_widget(self):
        """Return ``(scale, offset_x, offset_y)`` fitting the route, or None."""
        pts = [p for p in self._positions if p is not None]
        if len(pts) < 2:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        margin = 14
        avail_w = max(1, self.width() - 2 * margin)
        avail_h = max(1, self.height() - 2 * margin)
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        scale = min(avail_w / span_x, avail_h / span_y)
        # Centre the route in the widget; y is flipped (north up).
        off_x = margin + (avail_w - span_x * scale) / 2 - min_x * scale
        off_y = margin + (avail_h - span_y * scale) / 2 + max_y * scale
        return scale, off_x, off_y

    def _to_point(self, pos, mapping) -> QPointF:
        scale, off_x, off_y = mapping
        return QPointF(pos[0] * scale + off_x, off_y - pos[1] * scale)

    def mousePressEvent(self, event):
        mapping = self._world_to_widget()
        if mapping is None:
            return
        best_frame, best_d2 = None, float("inf")
        for idx, pos in enumerate(self._positions):
            if pos is None:
                continue
            pt = self._to_point(pos, mapping)
            d2 = (pt.x() - event.pos().x()) ** 2 + (pt.y() - event.pos().y()) ** 2
            if d2 < best_d2:
                best_frame, best_d2 = idx, d2
        if best_frame is not None:
            self.frameClicked.emit(best_frame)

    # -- painting ------------------------------------------------------------

    def _draw_polyline(self, painter, frames, mapping) -> None:
        path = QPainterPath()
        started = False
        for idx in frames:
            pos = self._positions[idx]
            if pos is None:
                continue
            pt = self._to_point(pos, mapping)
            if not started:
                path.moveTo(pt)
                started = True
            else:
                path.lineTo(pt)
        painter.drawPath(path)

    def _draw_scale_bar(self, painter, scale) -> None:
        """A round-metre scale bar in the lower-left corner."""
        target_px = self.width() / 4
        meters = target_px / scale
        # Round down to 1/2/5 * 10^n
        magnitude = 10 ** max(0, len(str(int(max(meters, 1)))) - 1)
        for factor in (5, 2, 1):
            if factor * magnitude <= meters:
                meters = factor * magnitude
                break
        else:
            meters = magnitude
        bar_px = int(meters * scale)
        x0, y0 = 8, self.height() - 10
        painter.setPen(QPen(QColor(220, 220, 220), 1))
        painter.drawLine(x0, y0, x0 + bar_px, y0)
        painter.drawLine(x0, y0 - 3, x0, y0 + 3)
        painter.drawLine(x0 + bar_px, y0 - 3, x0 + bar_px, y0 + 3)
        painter.drawText(x0 + 4, y0 - 4, f"{meters:g} m")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(35, 35, 35))

        mapping = self._world_to_widget()
        if mapping is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No flight route\n(poses without positions)")
            painter.end()
            return

        n = len(self._positions)

        # Whole flight route
        pen = QPen(_ROUTE_COLOR, 1)
        painter.setPen(pen)
        self._draw_polyline(painter, range(n), mapping)

        # Transect segments on top, selected one emphasized
        for transect in self._transects:
            selected = transect.transect_id == self._selected_id
            pen = QPen(_transect_color(transect.transect_id),
                       4 if selected else 2)
            painter.setPen(pen)
            lo = max(0, min(transect.first_frame, n - 1))
            hi = max(0, min(transect.last_frame, n - 1))
            self._draw_polyline(painter, range(lo, hi + 1), mapping)
            if selected:
                for frame, label in ((lo, "S"), (hi, "E")):
                    pos = self._positions[frame]
                    if pos is None:
                        continue
                    pt = self._to_point(pos, mapping)
                    painter.setBrush(_transect_color(transect.transect_id))
                    painter.drawEllipse(pt, 4, 4)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawText(QPointF(pt.x() + 5, pt.y() - 5), label)

        # Current frame position
        if 0 <= self._current < n and self._positions[self._current] is not None:
            pt = self._to_point(self._positions[self._current], mapping)
            painter.setPen(QPen(_CURRENT_COLOR, 2))
            painter.setBrush(_CURRENT_COLOR)
            painter.drawEllipse(pt, 3, 3)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(pt, 7, 7)

        # North arrow (mesh-local y axis is north)
        painter.setPen(QPen(QColor(220, 220, 220), 1))
        painter.drawText(self.width() - 18, 16, "N")
        painter.drawLine(self.width() - 13, 20, self.width() - 13, 34)
        painter.drawLine(self.width() - 13, 20, self.width() - 16, 26)
        painter.drawLine(self.width() - 13, 20, self.width() - 10, 26)

        self._draw_scale_bar(painter, mapping[0])
        painter.end()


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

class _TransectTimeline(QWidget):
    """Slim timeline bar showing the transect spans over all frames."""

    frameClicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(24)
        self._total = 1
        self._current = 0
        self._transects: List[Transect] = []
        self._selected_id: Optional[int] = None

    def set_data(self, total: int, current: int, transects: List[Transect],
                 selected_id: Optional[int]) -> None:
        self._total = max(1, total)
        self._current = current
        self._transects = transects
        self._selected_id = selected_id
        self.update()

    def _frame_to_x(self, frame: int) -> float:
        return frame / max(1, self._total - 1) * (self.width() - 1)

    def mousePressEvent(self, event):
        frac = event.pos().x() / max(1, self.width() - 1)
        self.frameClicked.emit(int(round(frac * (self._total - 1))))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(45, 45, 45))

        for transect in self._transects:
            selected = transect.transect_id == self._selected_id
            x0 = int(self._frame_to_x(transect.first_frame))
            x1 = int(self._frame_to_x(transect.last_frame))
            color = _transect_color(transect.transect_id)
            if not selected:
                color = QColor(color.red(), color.green(), color.blue(), 140)
            painter.fillRect(x0, 4, max(2, x1 - x0), self.height() - 8, color)
            if selected:
                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.drawLine(x0, 2, x0, self.height() - 2)
                painter.drawLine(x1, 2, x1, self.height() - 2)

        painter.setPen(QPen(_CURRENT_COLOR, 2))
        x = int(self._frame_to_x(self._current))
        painter.drawLine(x, 0, x, self.height())
        painter.end()


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class TransectToolDialog(QDialog):
    """Non-modal tool for splitting a flight into transects."""

    def __init__(self, iface, dock_widget=None, parent=None):
        super().__init__(parent)
        self.iface = iface
        self._dock_widget = dock_widget

        self.setWindowTitle("BAMBI Transect Splitting Tool")
        flags = Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint
        flags |= Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint
        self.setWindowFlags(flags)

        # Project state
        self._target_folder = ""
        self._modality = "t"
        self._images: List[dict] = []          # poses images of the modality
        self._positions: List[Optional[Tuple[float, float]]] = []
        self._cum: List[float] = []            # cumulative path length [m]
        self._store: Optional[TransectStore] = None

        # UI state
        self._current_frame = 0
        self._selected_id: Optional[int] = None
        self._dirty = False
        self._updating_ui = False

        self._setup_ui()
        fit_to_screen(self, 1200, 800)
        self.apply_dock_defaults()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ---- top row: project inputs --------------------------------------
        top = QHBoxLayout()
        top.addWidget(QLabel("Target folder:"))
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText(
            "Plugin output folder (contains frames_t / frames_w)")
        top.addWidget(self.folder_edit, 1)
        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(30)
        browse_btn.clicked.connect(self._on_browse_folder)
        top.addWidget(browse_btn)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._on_load_clicked)
        top.addWidget(load_btn)

        top.addSpacing(12)
        top.addWidget(QLabel("Modality:"))
        self.modality_combo = QComboBox()
        self.modality_combo.addItem("Thermal", "t")
        self.modality_combo.addItem("RGB", "w")
        self.modality_combo.currentIndexChanged.connect(self._on_modality_changed)
        top.addWidget(self.modality_combo)

        top.addStretch()
        self.save_btn = QPushButton("Save Transects")
        self.save_btn.setToolTip(
            "Write transects_{t,w}/transects.json and transects.csv to the "
            "output folder. Existing definitions are loaded automatically.")
        self.save_btn.clicked.connect(self._on_save)
        top.addWidget(self.save_btn)
        layout.addLayout(top)

        # ---- center: canvas + side panel ----------------------------------
        center = QHBoxLayout()
        self.canvas = _FrameView()
        # NoFocus so arrow keys reach the dialog's keyPressEvent for frame
        # navigation instead of scrolling the graphics view.
        self.canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        center.addWidget(self.canvas, 1)
        center.addWidget(self._build_side_panel())
        layout.addLayout(center, 1)

        # ---- bottom: navigation --------------------------------------------
        nav = QHBoxLayout()
        self.nav_step_spin = QSpinBox()
        self.nav_step_spin.setRange(1, 500)
        self.nav_step_spin.setValue(10)
        self.nav_step_spin.setToolTip(
            "Frame stride of the step navigation buttons (and PageUp / "
            "PageDown).")
        self.prev_step_btn = QPushButton("<< Step")
        self.prev_step_btn.clicked.connect(
            lambda: self._go_relative(-self.nav_step_spin.value()))
        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(lambda: self._go_relative(-1))
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(lambda: self._go_relative(1))
        self.next_step_btn = QPushButton("Step >>")
        self.next_step_btn.clicked.connect(
            lambda: self._go_relative(self.nav_step_spin.value()))

        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 0)
        self.frame_spin.valueChanged.connect(self._on_frame_spin)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_frame_slider)

        nav.addWidget(self.prev_step_btn)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.frame_spin)
        nav.addWidget(self.frame_slider, 1)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.next_step_btn)
        nav.addWidget(QLabel("Step:"))
        nav.addWidget(self.nav_step_spin)
        layout.addLayout(nav)

        self.timeline = _TransectTimeline()
        self.timeline.frameClicked.connect(self._goto_frame)
        layout.addWidget(self.timeline)

        self.status_label = QLabel("No project loaded.")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(_SIDE_PANEL_WIDTH)
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)

        # Overview map
        map_group = QGroupBox("Flight overview")
        mg = QVBoxLayout(map_group)
        self.map_widget = _FlightMapWidget()
        self.map_widget.setFixedHeight(260)
        self.map_widget.setToolTip(
            "Flight route with the current frame's position (yellow) and "
            "the defined transects. Click to jump to the nearest frame.")
        self.map_widget.frameClicked.connect(self._goto_frame)
        mg.addWidget(self.map_widget)
        vbox.addWidget(map_group)

        # Transect list
        list_group = QGroupBox("Transects")
        lg = QVBoxLayout(list_group)
        self.transect_list = QListWidget()
        self.transect_list.currentItemChanged.connect(self._on_list_selection)
        lg.addWidget(self.transect_list)
        btn_row = QHBoxLayout()
        new_btn = QPushButton("New Transect (N)")
        new_btn.setToolTip(
            "Start a new transect at the current frame (start = end = "
            "current frame; then set the end frame).")
        new_btn.clicked.connect(self._on_new_transect)
        del_btn = QPushButton("Delete")
        del_btn.clicked.connect(self._on_delete_transect)
        btn_row.addWidget(new_btn)
        btn_row.addWidget(del_btn)
        lg.addLayout(btn_row)
        vbox.addWidget(list_group)

        # Selected transect
        self.edit_group = QGroupBox("Selected transect")
        eg = QVBoxLayout(self.edit_group)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("optional — default: Transect {id}")
        self.name_edit.textEdited.connect(self._on_name_edited)
        name_row.addWidget(self.name_edit, 1)
        eg.addLayout(name_row)

        start_row = QHBoxLayout()
        self.start_label = QLabel("Start: –")
        start_row.addWidget(self.start_label, 1)
        goto_start_btn = QPushButton("Go to")
        goto_start_btn.setToolTip("Jump to the transect's start frame.")
        goto_start_btn.clicked.connect(
            lambda: self._goto_transect_frame("start_frame"))
        start_row.addWidget(goto_start_btn)
        set_start_btn = QPushButton("Set (S)")
        set_start_btn.setToolTip("Set the start to the current frame.")
        set_start_btn.clicked.connect(self._on_set_start)
        start_row.addWidget(set_start_btn)
        eg.addLayout(start_row)

        end_row = QHBoxLayout()
        self.end_label = QLabel("End: –")
        end_row.addWidget(self.end_label, 1)
        goto_end_btn = QPushButton("Go to")
        goto_end_btn.setToolTip("Jump to the transect's end frame.")
        goto_end_btn.clicked.connect(
            lambda: self._goto_transect_frame("end_frame"))
        end_row.addWidget(goto_end_btn)
        set_end_btn = QPushButton("Set (E)")
        set_end_btn.setToolTip("Set the end to the current frame.")
        set_end_btn.clicked.connect(self._on_set_end)
        end_row.addWidget(set_end_btn)
        eg.addLayout(end_row)

        dist_row = QHBoxLayout()
        dist_row.addWidget(QLabel("End after:"))
        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(0.1, 1_000_000.0)
        self.distance_spin.setDecimals(1)
        self.distance_spin.setValue(100.0)
        self.distance_spin.setSuffix(" m")
        dist_row.addWidget(self.distance_spin, 1)
        end_after_btn = QPushButton("Set end")
        end_after_btn.setToolTip(
            "Place the end at the first frame at least this many metres "
            "along the flight path after the start frame.")
        end_after_btn.clicked.connect(self._on_end_after_distance)
        dist_row.addWidget(end_after_btn)
        eg.addLayout(dist_row)

        self.length_label = QLabel("Length: –")
        self.length_label.setToolTip(
            "Flight-path length between start and end frame (measured along "
            "the polyline of the per-frame camera positions).")
        eg.addWidget(self.length_label)

        self.edit_group.setEnabled(False)
        vbox.addWidget(self.edit_group)

        vbox.addStretch()
        fit_btn = QPushButton("Fit view")
        fit_btn.clicked.connect(self.canvas.fit)
        vbox.addWidget(fit_btn)

        # The panel is taller than the dialog on small screens, so it scrolls
        # instead of forcing a minimum height on the whole window.
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(
            _SIDE_PANEL_WIDTH + scroll.verticalScrollBar().sizeHint().width())
        return scroll

    # ------------------------------------------------------------------
    # Shortcuts
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Right:
            self._go_relative(1)
        elif key == Qt.Key.Key_Left:
            self._go_relative(-1)
        elif key == Qt.Key.Key_PageUp:
            self._go_relative(self.nav_step_spin.value())
        elif key == Qt.Key.Key_PageDown:
            self._go_relative(-self.nav_step_spin.value())
        elif key == Qt.Key.Key_N:
            self._on_new_transect()
        elif key == Qt.Key.Key_S:
            self._on_set_start()
        elif key == Qt.Key.Key_E:
            self._on_set_end()
        elif key == Qt.Key.Key_Delete:
            self._on_delete_transect()
        elif key == Qt.Key.Key_Escape:
            pass  # do not close the dialog on Escape
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Project loading
    # ------------------------------------------------------------------

    def apply_dock_defaults(self):
        """Seed the target folder from the plugin dock widget."""
        if self._dock_widget is None:
            return
        try:
            config = self._dock_widget.get_config()
        except Exception:  # nosec B110
            return
        folder = config.get("target_folder", "")
        if not self.folder_edit.text() and folder:
            self.folder_edit.setText(folder)
            # Only auto-load quietly when extracted frames actually exist;
            # otherwise the user picks / fixes the folder manually.
            if os.path.isfile(os.path.join(folder, "poses_t.json")) or \
                    os.path.isfile(os.path.join(folder, "poses_w.json")):
                self._on_load_clicked()

    def _on_browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select plugin output folder", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)
            self._on_load_clicked()

    def _on_load_clicked(self):
        folder = self.folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "BAMBI Transect Tool",
                                "Please select a valid target folder.")
            return
        if not self._maybe_save_dirty():
            return

        self._target_folder = folder

        # Determine available modalities from the poses files
        has_t = os.path.isfile(os.path.join(folder, "poses_t.json"))
        has_w = os.path.isfile(os.path.join(folder, "poses_w.json"))
        if not has_t and not has_w:
            QMessageBox.warning(
                self, "BAMBI Transect Tool",
                "No poses_t.json / poses_w.json found in the target folder.\n\n"
                "Please run the frame extraction step first.")
            return

        self._updating_ui = True
        self.modality_combo.model().item(0).setEnabled(has_t)
        self.modality_combo.model().item(1).setEnabled(has_w)
        if self.modality_combo.currentData() == "t" and not has_t:
            self.modality_combo.setCurrentIndex(1)
        elif self.modality_combo.currentData() == "w" and not has_w:
            self.modality_combo.setCurrentIndex(0)
        self._updating_ui = False

        self._load_modality()

    def _on_modality_changed(self):
        if self._updating_ui or not self._target_folder:
            return
        if not self._maybe_save_dirty():
            return
        self._load_modality()

    def _load_modality(self):
        """(Re)load poses, flight route and transect store for the modality."""
        self._modality = self.modality_combo.currentData() or "t"
        m = self._modality

        poses_path = os.path.join(self._target_folder, f"poses_{m}.json")
        self._images = []
        if os.path.isfile(poses_path):
            try:
                with open(poses_path, "r", encoding="utf-8") as fh:
                    self._images = json.load(fh).get("images", [])
            except Exception:  # nosec B110
                pass
        if not self._images:
            QMessageBox.warning(
                self, "BAMBI Transect Tool",
                f"Could not read frames from {poses_path}.")
            return

        self._positions = flight_positions(self._images)
        self._cum = cumulative_distances(self._positions)

        self._store = TransectStore(self._target_folder, m)
        loaded = False
        try:
            loaded = self._store.load()
        except Exception as exc:
            QMessageBox.warning(
                self, "BAMBI Transect Tool",
                f"Could not load existing transects:\n{exc}")
        # Definitions from an earlier run may exceed the current frame count
        # (e.g. re-extracted frames): keep them but clamp into range.
        for transect in self._store.transects.values():
            transect.clamp(len(self._images))
        self._dirty = False
        self._selected_id = None

        total = len(self._images)
        self._updating_ui = True
        self.frame_spin.setRange(0, total - 1)
        self.frame_slider.setRange(0, total - 1)
        self._updating_ui = False

        self.map_widget.set_route(self._positions)
        self._current_frame = min(self._current_frame, total - 1)
        self._refresh_transect_list()
        self._goto_frame(self._current_frame, force=True)
        if loaded:
            self.status_label.setText(
                f"Loaded {len(self._store.transects)} transect(s) from "
                f"{self._store.json_path}")

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go_relative(self, delta: int):
        self._goto_frame(self._current_frame + delta)

    def _goto_frame(self, frame: int, force: bool = False):
        if not self._images:
            return
        frame = max(0, min(frame, len(self._images) - 1))
        if frame == self._current_frame and not force:
            return
        self._current_frame = frame
        self._updating_ui = True
        self.frame_spin.setValue(frame)
        self.frame_slider.setValue(frame)
        self._updating_ui = False
        self._render_frame()

    def _on_frame_spin(self, value: int):
        if not self._updating_ui:
            self._goto_frame(value)

    def _on_frame_slider(self, value: int):
        if not self._updating_ui:
            self._goto_frame(value)

    def _goto_transect_frame(self, attr: str):
        transect = self._selected_transect()
        if transect is not None:
            self._goto_frame(getattr(transect, attr))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _frame_image_path(self, frame: int) -> str:
        if not (0 <= frame < len(self._images)):
            return ""
        imagefile = self._images[frame].get("imagefile", "")
        if not imagefile:
            return ""
        path = os.path.join(
            self._target_folder, f"frames_{self._modality}", imagefile)
        return path if os.path.isfile(path) else ""

    def _render_frame(self):
        if not self._images:
            return
        path = self._frame_image_path(self._current_frame)
        pixmap = QPixmap(path) if path else QPixmap()
        if pixmap.isNull():
            pixmap = QPixmap(640, 512)
            pixmap.fill(QColor(40, 40, 40))
        self.canvas.set_frame(pixmap)
        self._update_views()

    def _update_views(self):
        """Refresh map, timeline, selected-transect info and status line."""
        transects = self._store.ordered() if self._store else []
        self.map_widget.set_state(
            self._current_frame, transects, self._selected_id)
        self.timeline.set_data(
            len(self._images), self._current_frame, transects,
            self._selected_id)
        self._update_edit_group()
        self._update_status()

    def _update_edit_group(self):
        transect = self._selected_transect()
        self.edit_group.setEnabled(transect is not None)
        self._updating_ui = True
        if transect is None:
            self.name_edit.clear()
            self.start_label.setText("Start: –")
            self.end_label.setText("End: –")
            self.length_label.setText("Length: –")
        else:
            if self.name_edit.text() != transect.name:
                self.name_edit.setText(transect.name)
            self.start_label.setText(f"Start: frame {transect.start_frame}")
            self.end_label.setText(f"End: frame {transect.end_frame}")
            length = path_length(
                self._cum, transect.first_frame, transect.last_frame)
            self.length_label.setText(
                f"Length: {length:.1f} m "
                f"({transect.last_frame - transect.first_frame + 1} frames)")
        self._updating_ui = False

    def _update_status(self):
        if not self._images:
            self.status_label.setText("No project loaded.")
            return
        parts = [f"Frame {self._current_frame} / {len(self._images) - 1}"]
        if self._cum:
            parts.append(
                f"along flight path: {self._cum[self._current_frame]:.1f} m")
        if self._store:
            parts.append(f"{len(self._store.transects)} transect(s)")
        if self._dirty:
            parts.append("unsaved changes")
        self.status_label.setText("   |   ".join(parts))

    # ------------------------------------------------------------------
    # Transect list
    # ------------------------------------------------------------------

    def _selected_transect(self) -> Optional[Transect]:
        if self._store is None or self._selected_id is None:
            return None
        return self._store.transects.get(self._selected_id)

    def _refresh_transect_list(self):
        self._updating_ui = True
        self.transect_list.clear()
        current_item = None
        for transect in (self._store.ordered() if self._store else []):
            length = path_length(
                self._cum, transect.first_frame, transect.last_frame)
            item = QListWidgetItem(
                f"{transect.display_name}  "
                f"[{transect.first_frame}–{transect.last_frame}]  "
                f"{length:.1f} m")
            item.setData(Qt.ItemDataRole.UserRole, transect.transect_id)
            item.setForeground(_transect_color(transect.transect_id))
            self.transect_list.addItem(item)
            if transect.transect_id == self._selected_id:
                current_item = item
        if current_item is not None:
            self.transect_list.setCurrentItem(current_item)
        elif self._selected_id is not None:
            self._selected_id = None
        self._updating_ui = False

    def _on_list_selection(self, item, _prev):
        if self._updating_ui:
            return
        self._selected_id = item.data(Qt.ItemDataRole.UserRole) if item else None
        self._update_views()

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def _require_store(self) -> bool:
        if self._store is None:
            QMessageBox.information(
                self, "BAMBI Transect Tool", "Load a target folder first.")
            return False
        return True

    def _on_new_transect(self):
        if not self._require_store():
            return
        transect = self._store.add(self._current_frame, self._current_frame)
        self._selected_id = transect.transect_id
        self._mark_dirty()
        self._refresh_transect_list()
        self._update_views()
        self.name_edit.setFocus()

    def _on_delete_transect(self):
        transect = self._selected_transect()
        if transect is None:
            return
        reply = QMessageBox.question(
            self, "Delete Transect",
            f"Delete {transect.display_name} "
            f"(frames {transect.first_frame}–{transect.last_frame})?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._store.remove(transect.transect_id)
        self._selected_id = None
        self._mark_dirty()
        self._refresh_transect_list()
        self._update_views()

    def _on_name_edited(self, text: str):
        transect = self._selected_transect()
        if transect is None or self._updating_ui:
            return
        transect.name = text.strip()
        self._mark_dirty()
        self._refresh_transect_list()

    def _on_set_start(self):
        transect = self._selected_transect()
        if transect is None:
            return
        transect.start_frame = self._current_frame
        self._mark_dirty()
        self._refresh_transect_list()
        self._update_views()

    def _on_set_end(self):
        transect = self._selected_transect()
        if transect is None:
            return
        transect.end_frame = self._current_frame
        self._mark_dirty()
        self._refresh_transect_list()
        self._update_views()

    def _on_end_after_distance(self):
        transect = self._selected_transect()
        if transect is None or not self._cum:
            return
        meters = self.distance_spin.value()
        frame = frame_after_distance(self._cum, transect.start_frame, meters)
        if frame is None:
            frame = len(self._images) - 1
            remaining = path_length(self._cum, transect.start_frame, frame)
            QMessageBox.information(
                self, "BAMBI Transect Tool",
                f"The flight path after frame {transect.start_frame} is only "
                f"{remaining:.1f} m long — the end was set to the last frame "
                f"({frame}).")
        transect.end_frame = frame
        self._mark_dirty()
        self._refresh_transect_list()
        self._update_views()
        self._goto_frame(frame)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _mark_dirty(self):
        self._dirty = True

    def _on_save(self):
        if self._store is None:
            return
        try:
            self._store.save(self._images)
        except Exception as exc:
            QMessageBox.warning(
                self, "BAMBI Transect Tool",
                f"Could not save transects:\n{exc}")
            return
        self._dirty = False
        self._update_status()
        self.status_label.setText(
            f"Saved {len(self._store.transects)} transect(s) to "
            f"{self._store.json_path}")

    def _maybe_save_dirty(self) -> bool:
        """Ask the user about unsaved changes. Returns False on cancel."""
        if not self._dirty or self._store is None:
            return True
        reply = QMessageBox.question(
            self, "Unsaved Transects",
            "There are unsaved transect changes. Save them now?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save)
        if reply == QMessageBox.StandardButton.Cancel:
            return False
        if reply == QMessageBox.StandardButton.Save:
            self._on_save()
        else:
            self._dirty = False
        return True

    def closeEvent(self, event):
        if not self._maybe_save_dirty():
            event.ignore()
            return
        super().closeEvent(event)
