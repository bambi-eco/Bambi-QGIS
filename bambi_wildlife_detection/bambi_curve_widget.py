# -*- coding: utf-8 -*-
"""Interactive tone-curve editor for thermal imagery.

Provides three levels of reuse:

``CurveWidget``
    The bare canvas: histogram backdrop, grid, monotone curve and draggable
    control points. Left-click adds/drags a point, right-click removes one
    (endpoints can only be moved vertically).

``CurveEditorPanel``
    CurveWidget plus the temperature-domain controls: lower/upper °C spin
    boxes, an "Auto Detect" button that scans a flight's images for the
    actual temperature range (via ``core.thermal_curve.scan_temperatures``)
    and a reset button. Embedded directly in the Thermal Image Viewer.

``CurveEditorDialog``
    The panel wrapped in an OK/Cancel dialog, used by the photo-extraction
    configuration in the dock widget.

The heavy work (curve math, image scanning) lives in
``core.thermal_curve``; this module is Qt-only glue and painting.
"""

from typing import Callable, List, Optional, Sequence, Tuple

from qgis.PyQt.QtCore import Qt, pyqtSignal, QPointF, QRectF
from qgis.PyQt.QtGui import QColor, QPainter, QPainterPath, QPen
from qgis.PyQt.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QHBoxLayout, QLabel, QMessageBox, QProgressDialog, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

from .core.thermal_curve import (
    DEFAULT_POINTS, MIN_POINT_SEPARATION, ThermalCurve, scan_temperatures,
)

_POINT_HIT_RADIUS_PX = 9.0
_HIST_BARS = 128


def _event_xy(event) -> Tuple[float, float]:
    """Mouse event position compatible with both Qt5 and Qt6."""
    if hasattr(event, "position"):
        p = event.position()
    else:
        p = event.pos()
    return float(p.x()), float(p.y())


class CurveWidget(QWidget):
    """Canvas that renders and edits normalized curve control points."""

    curveChanged = pyqtSignal()
    hoverInfo = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(280, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self._points: List[Tuple[float, float]] = list(DEFAULT_POINTS)
        self._domain = (0.0, 40.0)
        self._hist: Optional[List[float]] = None   # normalized 0..1 bars
        self._drag_index: Optional[int] = None
        self._hover_index: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def points(self) -> List[Tuple[float, float]]:
        return list(self._points)

    def set_points(self, points: Sequence[Sequence[float]]):
        self._points = ThermalCurve(0.0, 1.0, points).points
        self.update()

    def set_domain(self, lo: float, hi: float):
        self._domain = (float(lo), float(hi))
        self.update()

    def set_histogram(self, bars: Optional[Sequence[float]]):
        """Set the backdrop histogram as pre-normalized bar heights (0..1)."""
        self._hist = list(bars) if bars is not None else None
        self.update()

    def reset(self):
        self._points = list(DEFAULT_POINTS)
        self.update()
        self.curveChanged.emit()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _plot_rect(self):
        return self.rect().adjusted(8, 8, -8, -22)

    def _to_px(self, x: float, y: float) -> Tuple[float, float]:
        r = self._plot_rect()
        return (r.left() + x * r.width(), r.bottom() - y * r.height())

    def _to_norm(self, px: float, py: float) -> Tuple[float, float]:
        r = self._plot_rect()
        w = max(r.width(), 1)
        h = max(r.height(), 1)
        x = (px - r.left()) / w
        y = (r.bottom() - py) / h
        return (min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0))

    def _point_at(self, px: float, py: float) -> Optional[int]:
        for i, (x, y) in enumerate(self._points):
            cx, cy = self._to_px(x, y)
            if (cx - px) ** 2 + (cy - py) ** 2 <= _POINT_HIT_RADIUS_PX ** 2:
                return i
        return None

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self._plot_rect()

        painter.fillRect(self.rect(), QColor(26, 26, 26))
        painter.fillRect(r, QColor(38, 38, 38))

        # Histogram backdrop
        if self._hist:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(90, 130, 190, 110))
            n = len(self._hist)
            bar_w = r.width() / n
            for i, v in enumerate(self._hist):
                if v <= 0.0:
                    continue
                h = v * r.height()
                painter.drawRect(QRectF(r.left() + i * bar_w, r.bottom() - h,
                                        max(bar_w, 1.0), h))

        # Grid (quarters) + diagonal reference
        grid_pen = QPen(QColor(70, 70, 70))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)
        for f in (0.25, 0.5, 0.75):
            painter.drawLine(QPointF(*self._to_px(f, 0.0)),
                             QPointF(*self._to_px(f, 1.0)))
            painter.drawLine(QPointF(*self._to_px(0.0, f)),
                             QPointF(*self._to_px(1.0, f)))
        diag_pen = QPen(QColor(90, 90, 90))
        diag_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(diag_pen)
        painter.drawLine(QPointF(*self._to_px(0.0, 0.0)),
                         QPointF(*self._to_px(1.0, 1.0)))

        # Border
        painter.setPen(QPen(QColor(110, 110, 110)))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(r)

        # Curve
        curve = ThermalCurve(0.0, 1.0, self._points)
        lut = curve.lut(160)
        path = QPainterPath()
        for i, v in enumerate(lut):
            px, py = self._to_px(i / (len(lut) - 1), float(v))
            if i == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)
        curve_pen = QPen(QColor(235, 235, 235))
        curve_pen.setWidthF(1.6)
        painter.setPen(curve_pen)
        painter.drawPath(path)

        # Control points
        for i, (x, y) in enumerate(self._points):
            px, py = self._to_px(x, y)
            active = i == self._drag_index or i == self._hover_index
            painter.setBrush(QColor(255, 170, 60) if active
                             else QColor(230, 230, 230))
            painter.setPen(QPen(QColor(20, 20, 20)))
            painter.drawRect(QRectF(px - 3.5, py - 3.5, 7.0, 7.0))

        # Domain labels (°C)
        painter.setPen(QPen(QColor(170, 170, 170)))
        lo, hi = self._domain
        painter.drawText(r.left(), self.rect().bottom() - 6, f"{lo:.1f} °C")
        hi_text = f"{hi:.1f} °C"
        metrics = painter.fontMetrics()
        painter.drawText(r.right() - metrics.horizontalAdvance(hi_text),
                         self.rect().bottom() - 6, hi_text)
        painter.end()

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        px, py = _event_xy(event)
        idx = self._point_at(px, py)
        if event.button() == Qt.MouseButton.RightButton:
            if idx is not None and 0 < idx < len(self._points) - 1:
                del self._points[idx]
                self._drag_index = None
                self.update()
                self.curveChanged.emit()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        if idx is None:
            x, y = self._to_norm(px, py)
            idx = self._insert_point(x, y)
        self._drag_index = idx
        self.update()
        if idx is not None:
            self.curveChanged.emit()

    def mouseMoveEvent(self, event):
        px, py = _event_xy(event)
        if self._drag_index is not None:
            self._move_dragged_point(px, py)
            return
        hover = self._point_at(px, py)
        if hover != self._hover_index:
            self._hover_index = hover
            self.update()
        x, y = self._to_norm(px, py)
        lo, hi = self._domain
        temp = lo + x * (hi - lo)
        curve = ThermalCurve(0.0, 1.0, self._points)
        out = float(curve.lut(256)[min(int(x * 255), 255)])
        self.hoverInfo.emit(f"{temp:.1f} °C  →  {out * 100.0:.0f} %")

    def mouseReleaseEvent(self, event):
        if self._drag_index is not None:
            self._drag_index = None
            self.update()
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self._hover_index = None
        self.hoverInfo.emit("")
        self.update()
        super().leaveEvent(event)

    def _insert_point(self, x: float, y: float) -> Optional[int]:
        """Insert a control point at normalized (x, y); returns its index."""
        for i, (ex, _ey) in enumerate(self._points):
            if abs(ex - x) < MIN_POINT_SEPARATION:
                return i  # too close to an existing column: drag that one
        idx = next((i for i, (ex, _ey) in enumerate(self._points) if ex > x),
                   len(self._points))
        self._points.insert(idx, (x, y))
        return idx

    def _move_dragged_point(self, px: float, py: float):
        i = self._drag_index
        x, y = self._to_norm(px, py)
        last = len(self._points) - 1
        if i == 0:
            x = 0.0
        elif i == last:
            x = 1.0
        else:
            x = min(max(x, self._points[i - 1][0] + MIN_POINT_SEPARATION),
                    self._points[i + 1][0] - MIN_POINT_SEPARATION)
        self._points[i] = (x, y)
        lo, hi = self._domain
        self.hoverInfo.emit(
            f"{lo + x * (hi - lo):.1f} °C  →  {y * 100.0:.0f} %")
        self.update()
        self.curveChanged.emit()


class CurveEditorPanel(QWidget):
    """Curve canvas plus temperature-domain controls and auto-detection.

    :param image_paths_provider: Callable returning the list of image paths
        the "Auto Detect" scan should cover (may be None to hide the button).
    :param parse_factory: Callable returning ``(parse_fn, close_fn)`` where
        *parse_fn* maps a file path to a °C array and *close_fn* releases
        any resources afterwards. Exceptions raised by the factory are shown
        to the user.
    """

    curveChanged = pyqtSignal()

    def __init__(self, image_paths_provider: Optional[Callable] = None,
                 parse_factory: Optional[Callable] = None, parent=None):
        super().__init__(parent)
        self._image_paths_provider = image_paths_provider
        self._parse_factory = parse_factory
        self._scan = None            # TemperatureScan from Auto Detect
        self._temp_array = None      # single-image histogram source
        self._build_ui()

    def _build_ui(self):
        import numpy as np  # noqa: F401 — ensures availability early

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Range:"))
        self.lo_spin = QDoubleSpinBox()
        self.hi_spin = QDoubleSpinBox()
        for spin in (self.lo_spin, self.hi_spin):
            spin.setRange(-200.0, 3000.0)
            spin.setDecimals(1)
            spin.setSuffix(" °C")
        self.lo_spin.setValue(0.0)
        self.hi_spin.setValue(40.0)
        self.lo_spin.setToolTip(
            "Temperature mapped to the left edge of the curve")
        self.hi_spin.setToolTip(
            "Temperature mapped to the right edge of the curve")
        range_row.addWidget(self.lo_spin)
        range_row.addWidget(QLabel("–"))
        range_row.addWidget(self.hi_spin)
        self.auto_detect_btn = QPushButton("Auto Detect…")
        self.auto_detect_btn.setToolTip(
            "Scan the flight's thermal images and set the range to the "
            "measured minimum/maximum temperature")
        self.auto_detect_btn.clicked.connect(self._auto_detect)
        range_row.addWidget(self.auto_detect_btn)
        self.robust_check = QCheckBox("Ignore 1% outliers")
        self.robust_check.setToolTip(
            "Use the 1st/99th percentile instead of the exact extremes, "
            "suppressing dead or hot pixels")
        range_row.addWidget(self.robust_check)
        range_row.addStretch()
        root.addLayout(range_row)

        self.curve_widget = CurveWidget()
        root.addWidget(self.curve_widget, 1)

        bottom_row = QHBoxLayout()
        self.hover_label = QLabel("")
        self.hover_label.setStyleSheet("color: grey;")
        bottom_row.addWidget(self.hover_label, 1)
        self.reset_btn = QPushButton("Reset Curve")
        self.reset_btn.clicked.connect(self.curve_widget.reset)
        bottom_row.addWidget(self.reset_btn)
        root.addLayout(bottom_row)

        self.curve_widget.curveChanged.connect(self.curveChanged)
        self.curve_widget.hoverInfo.connect(self.hover_label.setText)
        self.lo_spin.valueChanged.connect(self._on_domain_changed)
        self.hi_spin.valueChanged.connect(self._on_domain_changed)
        self._on_domain_changed()

        if self._image_paths_provider is None:
            self.auto_detect_btn.setVisible(False)
            self.robust_check.setVisible(False)

    # ------------------------------------------------------------------
    # Curve accessors
    # ------------------------------------------------------------------

    def curve(self) -> ThermalCurve:
        return ThermalCurve(self.lo_spin.value(), self.hi_spin.value(),
                            self.curve_widget.points())

    def set_curve(self, curve: ThermalCurve):
        for spin, value in ((self.lo_spin, curve.domain_lo),
                            (self.hi_spin, curve.domain_hi)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.curve_widget.set_points(curve.points)
        self._on_domain_changed()

    def set_temperature_array(self, arr):
        """Use a single image as the histogram source (viewer preview).

        Replaces a previous "Auto Detect" scan as the histogram source —
        the detected temperature range itself stays in the spin boxes.
        """
        self._temp_array = arr
        self._scan = None
        self._refresh_histogram()

    def seed_domain_if_default(self, lo: float, hi: float):
        """Initialise the range from an image when still at the defaults."""
        if (self.lo_spin.value(), self.hi_spin.value()) == (0.0, 40.0):
            self.lo_spin.blockSignals(True)
            self.hi_spin.blockSignals(True)
            self.lo_spin.setValue(round(lo, 1))
            self.hi_spin.setValue(round(hi, 1))
            self.lo_spin.blockSignals(False)
            self.hi_spin.blockSignals(False)
            self._on_domain_changed()

    # ------------------------------------------------------------------
    # Domain / histogram
    # ------------------------------------------------------------------

    def _on_domain_changed(self):
        self.curve_widget.set_domain(self.lo_spin.value(),
                                     self.hi_spin.value())
        self._refresh_histogram()
        self.curveChanged.emit()

    def _refresh_histogram(self):
        import numpy as np

        lo = self.lo_spin.value()
        hi = self.hi_spin.value()
        if hi <= lo:
            self.curve_widget.set_histogram(None)
            return
        counts = None
        if self._scan is not None and int(self._scan.counts.sum()) > 0:
            centers = self._scan.bin_edges[:-1] + (
                self._scan.bin_edges[1] - self._scan.bin_edges[0]) / 2.0
            counts, _ = np.histogram(centers, bins=_HIST_BARS,
                                     range=(lo, hi),
                                     weights=self._scan.counts)
        elif self._temp_array is not None:
            arr = np.asarray(self._temp_array)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                counts, _ = np.histogram(finite, bins=_HIST_BARS,
                                         range=(lo, hi))
        if counts is None or counts.max() <= 0:
            self.curve_widget.set_histogram(None)
            return
        # sqrt scaling keeps strongly peaked thermal histograms readable
        bars = np.sqrt(counts.astype(np.float64))
        self.curve_widget.set_histogram((bars / bars.max()).tolist())

    # ------------------------------------------------------------------
    # Auto detection
    # ------------------------------------------------------------------

    def _auto_detect(self):
        paths = []
        try:
            paths = list(self._image_paths_provider() or [])
        except Exception as exc:
            QMessageBox.warning(self, "Auto Detect",
                                f"Could not list images:\n{exc}")
            return
        if not paths:
            QMessageBox.information(
                self, "Auto Detect",
                "No thermal images found. Configure the thermal photo "
                "directory (or open a folder) first.")
            return
        try:
            parse_fn, close_fn = self._parse_factory()
        except Exception as exc:
            QMessageBox.warning(
                self, "Auto Detect",
                f"Thermal parser unavailable:\n{exc}\n\n"
                "Make sure the DJI Thermal SDK is installed "
                "(Dependency Manager).")
            return

        progress = QProgressDialog(
            "Scanning thermal images…", "Cancel", 0, len(paths), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        def on_progress(i, total, path):
            progress.setValue(i)
            progress.setLabelText(
                f"Scanning thermal images… ({i + 1}/{total})")
            QApplication.processEvents()

        try:
            scan = scan_temperatures(
                paths, parse_fn,
                progress_cb=on_progress,
                cancel_cb=progress.wasCanceled,
            )
        finally:
            try:
                close_fn()
            except Exception:
                pass
            progress.close()

        if scan.n_images == 0:
            QMessageBox.warning(
                self, "Auto Detect",
                "None of the images could be parsed as radiometric "
                "thermal data.")
            return

        self._scan = scan
        self._temp_array = None
        domain = scan.suggested_domain(robust=self.robust_check.isChecked())
        if domain is not None:
            lo, hi = domain
            self.lo_spin.blockSignals(True)
            self.hi_spin.blockSignals(True)
            self.lo_spin.setValue(round(lo, 1))
            self.hi_spin.setValue(round(hi, 1))
            self.lo_spin.blockSignals(False)
            self.hi_spin.blockSignals(False)
        self._on_domain_changed()

        note = " (scan cancelled — partial result)" if scan.cancelled else ""
        skipped = (f", {scan.n_errors} unreadable image(s) skipped"
                   if scan.n_errors else "")
        QMessageBox.information(
            self, "Auto Detect",
            f"Scanned {scan.n_images} image(s){note}{skipped}.\n"
            f"Temperature range: {scan.minimum:.1f} – "
            f"{scan.maximum:.1f} °C\n"
            f"1st–99th percentile: {scan.percentile(1):.1f} – "
            f"{scan.percentile(99):.1f} °C")


class CurveEditorDialog(QDialog):
    """Modal curve editor used by the photo-extraction configuration."""

    def __init__(self, curve: Optional[ThermalCurve] = None,
                 image_paths_provider: Optional[Callable] = None,
                 parse_factory: Optional[Callable] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Thermal Curve Mapping")
        self.resize(460, 420)

        layout = QVBoxLayout(self)
        hint = QLabel(
            "Map flight temperatures (x-axis, °C) to display intensity "
            "(y-axis). Left-click adds or drags a control point, "
            "right-click removes one. \"Auto Detect\" scans the configured "
            "thermal photo directory for the actual temperature range.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: grey;")
        layout.addWidget(hint)

        self.panel = CurveEditorPanel(
            image_paths_provider=image_paths_provider,
            parse_factory=parse_factory, parent=self)
        if curve is not None:
            self.panel.set_curve(curve)
        layout.addWidget(self.panel, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def curve(self) -> ThermalCurve:
        return self.panel.curve()
