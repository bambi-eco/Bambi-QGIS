# -*- coding: utf-8 -*-
"""AGL tool: the flight's altitude over the relief it flew across.

A window with one plot. The ground below the route is a filled relief, the
aircraft's altitude a line above it, both in metres above sea level; the x
axis is the distance flown or the time since the first frame, switchable.
Hovering shows the frame under the cursor with its altitude, the ground
height and the height above ground level. The camera whose poses are drawn
is selectable, because thermal and RGB were extracted separately and can
differ in frame count and timing.

The numbers come from :mod:`core.agl_profile`; this module only draws.
"""
import os
from typing import List, Optional

from qgis.PyQt.QtCore import QPointF, QRectF, Qt, QThread, pyqtSignal
from qgis.PyQt.QtGui import (QBrush, QColor, QFont, QFontMetrics, QPainter,
                             QPainterPath, QPen, QPolygonF)
from qgis.PyQt.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QRadioButton, QSizePolicy, QVBoxLayout, QWidget,
)

from .core import agl_profile


class ProfileCanvas(QWidget):
    """Draws the profile and reports the frame under the cursor."""

    MARGIN_LEFT = 64
    MARGIN_RIGHT = 16
    MARGIN_TOP = 16
    MARGIN_BOTTOM = 44

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points: List[dict] = []
        self._axis = "distance_m"
        self._hover: Optional[int] = None
        self._on_hover = None
        self._message = ""
        self.setMouseTracking(True)
        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    # -- data --------------------------------------------------------------

    def set_points(self, points: List[dict]):
        self._points = [p for p in points
                        if p.get(self._axis) is not None] if points else []
        self._all_points = list(points or [])
        self._hover = None
        self.update()

    def set_message(self, message: str):
        """Text shown instead of the empty-state hint while there is no profile."""
        self._message = message
        self.update()

    def set_axis(self, axis: str):
        """``"distance_m"`` or ``"time_s"``."""
        self._axis = axis
        self.set_points(getattr(self, "_all_points", self._points))

    def set_hover_callback(self, callback):
        self._on_hover = callback

    def hovered_point(self) -> Optional[dict]:
        if self._hover is None or self._hover >= len(self._points):
            return None
        return self._points[self._hover]

    # -- geometry ----------------------------------------------------------

    def _plot_rect(self) -> QRectF:
        return QRectF(self.MARGIN_LEFT, self.MARGIN_TOP,
                      max(10, self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT),
                      max(10, self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM))

    def _ranges(self):
        xs = [p[self._axis] for p in self._points]
        ys = [p["altitude"] for p in self._points]
        ys += [p["ground"] for p in self._points if p["ground"] is not None]
        if not xs or not ys:
            return (0.0, 1.0), (0.0, 1.0)
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if x1 - x0 < 1e-9:
            x1 = x0 + 1.0
        pad = max(1.0, (y1 - y0) * 0.08)
        return (x0, x1), (y0 - pad, y1 + pad)

    def _to_pixel(self, x, y, rect, xr, yr) -> QPointF:
        px = rect.left() + (x - xr[0]) / (xr[1] - xr[0]) * rect.width()
        py = rect.bottom() - (y - yr[0]) / (yr[1] - yr[0]) * rect.height()
        return QPointF(px, py)

    @staticmethod
    def _ticks(low, high, count=6):
        span = high - low
        if span <= 0:
            return [low]
        raw = span / count
        magnitude = 10 ** int(math_floor(math_log10(raw)))
        for step in (1, 2, 2.5, 5, 10):
            if raw <= step * magnitude:
                step = step * magnitude
                break
        first = math_ceil(low / step) * step
        ticks = []
        value = first
        while value <= high + 1e-9:
            ticks.append(value)
            value += step
        return ticks

    # -- painting ----------------------------------------------------------

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(250, 250, 250))
        rect = self._plot_rect()
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        metrics = QFontMetrics(font)

        if not self._points:
            painter.setPen(QColor(120, 120, 120))
            hint = ("No profile loaded - choose a target folder with "
                    "extracted frames and a DEM, then press Load.")
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             self._message or hint)
            return

        xr, yr = self._ranges()

        # Grid and axes
        painter.setPen(QPen(QColor(225, 225, 225), 1))
        for y in self._ticks(yr[0], yr[1]):
            p = self._to_pixel(xr[0], y, rect, xr, yr)
            painter.drawLine(QPointF(rect.left(), p.y()),
                             QPointF(rect.right(), p.y()))
        for x in self._ticks(xr[0], xr[1], 8):
            p = self._to_pixel(x, yr[0], rect, xr, yr)
            painter.drawLine(QPointF(p.x(), rect.top()),
                             QPointF(p.x(), rect.bottom()))

        # Ground relief: filled down to the plot bottom, gaps where unknown.
        ground_pen = QPen(QColor(120, 85, 50), 1.5)
        ground_brush = QBrush(QColor(190, 160, 120, 160))
        segment: List[QPointF] = []

        def _flush():
            if len(segment) >= 2:
                polygon = QPolygonF(segment)
                polygon.append(QPointF(segment[-1].x(), rect.bottom()))
                polygon.append(QPointF(segment[0].x(), rect.bottom()))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(ground_brush)
                painter.drawPolygon(polygon)
                painter.setPen(ground_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPolyline(QPolygonF(segment))
            segment.clear()

        for p in self._points:
            if p["ground"] is None:
                _flush()
                continue
            segment.append(self._to_pixel(p[self._axis], p["ground"], rect, xr, yr))
        _flush()

        # Flight altitude
        path = QPainterPath()
        for i, p in enumerate(self._points):
            pt = self._to_pixel(p[self._axis], p["altitude"], rect, xr, yr)
            if i == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        painter.setPen(QPen(QColor(30, 100, 200), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        # Axes frame and tick labels
        painter.setPen(QPen(QColor(90, 90, 90), 1))
        painter.drawRect(rect)
        for y in self._ticks(yr[0], yr[1]):
            p = self._to_pixel(xr[0], y, rect, xr, yr)
            label = f"{y:,.0f}"
            painter.drawText(QPointF(rect.left() - 6 - metrics.horizontalAdvance(label),
                                     p.y() + metrics.ascent() / 2 - 1), label)
        unit = "m" if self._axis == "distance_m" else "s"
        for x in self._ticks(xr[0], xr[1], 8):
            p = self._to_pixel(x, yr[0], rect, xr, yr)
            label = f"{x:,.0f}"
            painter.drawText(QPointF(p.x() - metrics.horizontalAdvance(label) / 2,
                                     rect.bottom() + 4 + metrics.ascent()), label)
        axis_title = ("Distance from start (m)" if self._axis == "distance_m"
                      else "Time since start (s)")
        painter.drawText(QPointF(rect.center().x() - metrics.horizontalAdvance(axis_title) / 2,
                                 rect.bottom() + 8 + 2 * metrics.ascent()), axis_title)
        painter.save()
        painter.translate(12, rect.center().y())
        painter.rotate(-90)
        title = "Altitude above sea level (m)"
        painter.drawText(QPointF(-metrics.horizontalAdvance(title) / 2, 0), title)
        painter.restore()

        # Legend
        painter.setPen(QPen(QColor(30, 100, 200), 2))
        painter.drawLine(QPointF(rect.left() + 12, rect.top() + 12),
                         QPointF(rect.left() + 32, rect.top() + 12))
        painter.setPen(QColor(60, 60, 60))
        painter.drawText(QPointF(rect.left() + 38, rect.top() + 12 + metrics.ascent() / 2 - 1),
                         "flight altitude")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(ground_brush)
        painter.drawRect(QRectF(rect.left() + 130, rect.top() + 6, 20, 12))
        painter.setPen(QColor(60, 60, 60))
        painter.drawText(QPointF(rect.left() + 156, rect.top() + 12 + metrics.ascent() / 2 - 1),
                         "ground (DEM)")

        # Hover marker
        point = self.hovered_point()
        if point is not None:
            px = self._to_pixel(point[self._axis], point["altitude"], rect, xr, yr)
            painter.setPen(QPen(QColor(200, 60, 60), 1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(px.x(), rect.top()), QPointF(px.x(), rect.bottom()))
            painter.setPen(QPen(QColor(200, 60, 60), 2))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(px, 4, 4)
            if point["ground"] is not None:
                pg = self._to_pixel(point[self._axis], point["ground"], rect, xr, yr)
                painter.drawEllipse(pg, 4, 4)
                painter.setPen(QPen(QColor(200, 60, 60), 1))
                painter.drawLine(QPointF(px.x(), px.y() + 4), QPointF(pg.x(), pg.y() - 4))
                agl_label = f"{point['agl']:.1f} m AGL"
                painter.drawText(QPointF(px.x() + 8, (px.y() + pg.y()) / 2 + metrics.ascent() / 2),
                                 agl_label)

    # -- mouse -------------------------------------------------------------

    def mouseMoveEvent(self, event):
        if not self._points:
            return
        rect = self._plot_rect()
        xr, _yr = self._ranges()
        pos = event.position() if hasattr(event, "position") else event.localPos()
        x = xr[0] + (pos.x() - rect.left()) / rect.width() * (xr[1] - xr[0])
        nearest = min(range(len(self._points)),
                      key=lambda i: abs(self._points[i][self._axis] - x))
        if nearest != self._hover:
            self._hover = nearest
            self.update()
            if self._on_hover is not None:
                self._on_hover(self._points[nearest])

    def leaveEvent(self, _event):
        self._hover = None
        self.update()
        if self._on_hover is not None:
            self._on_hover(None)


def math_floor(value):
    import math
    return math.floor(value)


def math_ceil(value):
    import math
    return math.ceil(value)


def math_log10(value):
    import math
    return math.log10(value) if value > 0 else 0.0


class _ProfileWorker(QThread):
    """Builds the profile off the GUI thread.

    Reading the DEM mesh and casting a ray per pose takes seconds on a
    large flight, and the window used to do that inside its constructor - so
    nothing appeared until it was done. The window now opens at once, says
    it is loading, and fills in when this finishes.
    """

    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, folder, modality, dem_mesh, dem_json, parent=None):
        super().__init__(parent)
        self.args = (folder, modality, dem_mesh, dem_json)

    def run(self):
        folder, modality, dem_mesh, dem_json = self.args
        try:
            profile = agl_profile.build_profile(
                folder, modality, dem_path=dem_mesh, dem_json_path=dem_json)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(profile)


class AglToolDialog(QDialog):
    """Non-modal window with the AGL profile of the loaded flight."""

    def __init__(self, iface, dock_widget=None, parent=None):
        super().__init__(parent)
        self.iface = iface
        self._dock_widget = dock_widget
        self._profile: Optional[dict] = None
        self._worker: Optional[_ProfileWorker] = None
        self._pending: Optional[tuple] = None
        self.setWindowTitle("AGL - altitude above ground along the flight")
        self.setMinimumSize(900, 520)
        self.setWindowFlags(self.windowFlags()
                            | Qt.WindowType.WindowMaximizeButtonHint
                            | Qt.WindowType.WindowMinimizeButtonHint)

        layout = QVBoxLayout(self)

        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Target folder:"))
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Plugin output folder with poses_t.json / poses_w.json")
        folder_row.addWidget(self.folder_edit, 1)
        browse = QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self._on_browse_folder)
        folder_row.addWidget(browse)
        layout.addLayout(folder_row)

        dem_row = QHBoxLayout()
        dem_row.addWidget(QLabel("DEM mesh:"))
        self.dem_edit = QLineEdit()
        self.dem_edit.setPlaceholderText("dem_mesh.glb (its .json beside it gives the origin)")
        dem_row.addWidget(self.dem_edit, 1)
        browse_dem = QPushButton("…")
        browse_dem.setFixedWidth(30)
        browse_dem.clicked.connect(self._on_browse_dem)
        dem_row.addWidget(browse_dem)
        layout.addLayout(dem_row)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Thermal", "t")
        self.camera_combo.addItem("RGB", "w")
        self.camera_combo.currentIndexChanged.connect(lambda _i: self.load())
        controls.addWidget(self.camera_combo)
        controls.addSpacing(16)
        controls.addWidget(QLabel("X axis:"))
        self.distance_radio = QRadioButton("Distance from start")
        self.distance_radio.setChecked(True)
        self.time_radio = QRadioButton("Time since start")
        self.distance_radio.toggled.connect(self._on_axis_changed)
        controls.addWidget(self.distance_radio)
        controls.addWidget(self.time_radio)
        controls.addStretch()
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self.load)
        controls.addWidget(self.load_btn)
        layout.addLayout(controls)

        self.canvas = ProfileCanvas()
        self.canvas.set_hover_callback(self._on_hover)
        layout.addWidget(self.canvas, 1)

        self.hover_label = QLabel("Hover over the profile to read the altitude above ground.")
        self.hover_label.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.hover_label)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.apply_dock_defaults()

    # -- dock ----------------------------------------------------------------

    def apply_dock_defaults(self):
        """Seed folder and DEM from the plugin dock widget, then load."""
        if self._dock_widget is None:
            return
        try:
            config = self._dock_widget.get_config()
        except Exception:  # nosec B110
            return
        changed = False
        folder = config.get("target_folder", "")
        if folder and not self.folder_edit.text():
            self.folder_edit.setText(folder)
            changed = True
        dem = config.get("dem_path", "") or config.get("dem_metadata_path", "")
        if dem and not self.dem_edit.text():
            self.dem_edit.setText(dem)
            changed = True
        self._dem_json = config.get("dem_metadata_path", "")
        if changed and os.path.isdir(self.folder_edit.text()):
            self.load()

    def _on_browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select plugin output folder", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)
            self.load()

    def _on_browse_dem(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select DEM mesh", self.dem_edit.text(),
            "DEM mesh (*.glb *.gltf);;DEM metadata (*.json);;All files (*)")
        if path:
            self.dem_edit.setText(path)
            self.load()

    # -- behaviour -----------------------------------------------------------

    def _axis(self) -> str:
        return "distance_m" if self.distance_radio.isChecked() else "time_s"

    def _on_axis_changed(self, _checked):
        self.canvas.set_axis(self._axis())
        if self._profile and self._axis() == "time_s" and not any(
                p["time_s"] is not None for p in self._profile["points"]):
            self.status_label.setText(
                "The poses carry no timestamps, so there is no time axis for "
                "this flight - re-extract the frames with SRT files present.")

    def load(self):
        """Start building the profile; the window stays responsive meanwhile.

        A request that arrives while one is running (the camera switched, a
        new folder picked) is kept and started when the current one ends,
        so the profile shown is always the one last asked for.
        """
        folder = self.folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            self.canvas.set_message("")
            self.canvas.set_points([])
            self.status_label.setText("Choose a target folder first.")
            return
        modality = self.camera_combo.currentData()
        dem = self.dem_edit.text().strip()
        dem_json = getattr(self, "_dem_json", "") or (
            dem if dem.lower().endswith(".json") else "")
        dem_mesh = "" if dem.lower().endswith(".json") else dem
        args = (folder, modality, dem_mesh, dem_json)
        if self._worker is not None:
            self._pending = args
            return
        self._start_worker(args)

    def _start_worker(self, args):
        self._pending = None
        camera = self.camera_combo.currentText()
        self.load_btn.setEnabled(False)
        self.canvas.set_points([])
        self.canvas.set_message(
            f"Loading the {camera} profile - reading the DEM mesh and "
            "casting a ray for every pose…")
        self.status_label.setText(f"Loading {camera} profile from {args[0]}…")
        self._worker = _ProfileWorker(*args, parent=self)
        self._worker.finished_ok.connect(self._on_profile_ready)
        self._worker.failed.connect(self._on_profile_failed)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()

    def _on_worker_done(self):
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self.load_btn.setEnabled(True)
        if self._pending is not None:
            self._start_worker(self._pending)

    def _on_profile_failed(self, message: str):
        if self._pending is not None:
            return  # superseded; the next run reports its own outcome
        self._profile = None
        self.canvas.set_message("")
        self.canvas.set_points([])
        self.status_label.setText(f"Could not build the profile: {message}")

    def _on_profile_ready(self, profile: dict):
        if self._pending is not None:
            return  # superseded by a newer request
        self._profile = profile
        folder = self.folder_edit.text().strip()
        self.canvas.set_message("")
        self.canvas.set_axis(self._axis())
        self.canvas.set_points(self._profile["points"])
        camera = self.camera_combo.currentText()
        lines = agl_profile.summarise(self._profile)
        if not self._profile["points"]:
            lines.insert(0, f"No {camera} poses in {folder} - extract the "
                            f"{camera} frames first.")
        self.status_label.setText(f"{camera}: " + " | ".join(lines))
        self.hover_label.setText(
            "Hover over the profile to read the altitude above ground.")

    def closeEvent(self, event):
        worker = self._worker
        if worker is not None:
            self._pending = None
            # The mesh read cannot be interrupted; it is short and the
            # thread is daemon-like to Qt, so a bounded wait is enough.
            worker.wait(3000)
        super().closeEvent(event)

    def _on_hover(self, point: Optional[dict]):
        if point is None:
            self.hover_label.setText(
                "Hover over the profile to read the altitude above ground.")
            return
        time_part = (f"  t = {point['time_s']:7.1f} s" if point["time_s"] is not None
                     else "")
        ground = (f"{point['ground']:8.1f} m" if point["ground"] is not None
                  else "    n/a")
        agl = (f"{point['agl']:6.1f} m" if point["agl"] is not None else "   n/a")
        self.hover_label.setText(
            f"frame {point['frame']:5d}  d = {point['distance_m']:8.1f} m{time_part}"
            f"  |  altitude {point['altitude']:8.1f} m ASL   ground {ground}"
            f"   AGL {agl}  |  E {point['east']:.1f}  N {point['north']:.1f}")
