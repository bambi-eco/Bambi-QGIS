# -*- coding: utf-8 -*-
"""
BAMBI Labelling Tool
====================

Non-modal dialog for reviewing and creating track annotations on the
extracted frames of a processed flight.

Features
--------
* Visualises existing pipeline detections (``detections_{m}/detections.txt``)
  and tracks (``tracks_{m}/tracks_pixel.csv``) as read-only overlays.
* Modality selector: label either on thermal or RGB frames — never mixed.
  Label tracks are stored per modality in ``labels_{m}/labels.json``.
* Key-frame based annotation: bounding boxes are only stored on key frames
  (e.g. every 10th frame); all frames in between are linearly interpolated.
* Stop frames: a key frame can be flagged as "stop" when the animal
  disappears — no interpolation happens between a stop frame and the next
  key frame (where the track resumes), leaving a gap in the track.
* New tracks are drawn as bounding boxes and carry a species class, sex
  class, age class (per track) and an occlusion level (per key frame).
* Custom fields: the settings dialog (gear button) defines additional
  attributes — a name, a data type (int / float / string / bool / datetime)
  and a scope (per track, like species, or per key frame, like occlusion).
  Each one adds an input widget to the side panel and is stored in
  ``labels.json`` only (see below).  The configuration can be exported to a
  standalone JSON file and imported again, so a labelling setup can be shared.
* Existing label boxes can be moved/resized (which writes a key frame at the
  current frame) and their classes edited at any time.
* Existing pipeline tracks can be imported as editable label tracks.
* Geo-referenced propagation: the current bounding box is ray-cast onto the
  DEM (pixel → world) with the camera pose of the current frame and
  back-projected (world → pixel) with the pose of an offset frame — the
  result usually only needs small size adaptions.  A sampling rate adds
  intermediate key frames on the way to the target frame, so the linear
  interpolation between key frames never has to bridge a long span of curved
  drone ego-motion.
* Cross-modality copy: the label tracks of the other modality (RGB ↔
  thermal) are projected onto this modality's frames — frames are matched by
  capture time, each box is ray-cast onto the DEM with the source camera and
  back-projected with this modality's camera — and added as new, editable
  label tracks that typically need small manual adaptions.

Files written (relative to *target_folder*, ``{m}`` = ``t`` / ``w``)
--------------------------------------------------------------------
``labels_{m}/labels.json``  — key-frame source of truth, including the custom
    field schema (``custom_fields``) and their values (``attributes``)
``labels_{m}/labels.csv``   — per-frame interpolated export
    format: ``frame,track_id,x1,y1,x2,y2,species,sex,age,occlusion,keyframe``

Custom field values live in ``labels.json`` alone: the CSV export and
``detections.txt`` keep their fixed column layouts because the rest of the
pipeline parses them positionally.

"Add detections to project" additionally merges the interpolated label boxes
into ``detections_{m}/detections.txt`` in the exact format of the "Detect
Animals" stage (``frame x1 y1 x2 y2 confidence class_id``) so the rest of the
pipeline (geo-referencing, tracking, …) can consume them.  The exported block
is delimited by a marker comment and replaced on re-export.

"Replace detections in project" is the destructive sibling: after a modal
confirmation it overwrites ``detections_{m}/detections.txt`` with the label
boxes alone (detector output is discarded) and deletes the derived tracking
outputs (``tracks_{m}/``, ``tracks_pixel_{m}/``), so re-running
geo-referencing and tracking rebuilds everything from the labels.
"""

import os
import json
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from qgis.PyQt.QtCore import (
    Qt, QRectF, QPointF, QDateTime, QSettings, QTimer, pyqtSignal,
)
from qgis.PyQt.QtGui import (
    QColor, QPen, QFont, QPixmap, QPainter, QCursor, QPainterPath,
)
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QDateTimeEdit, QCheckBox, QSlider,
    QWidget, QGroupBox, QFormLayout, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialogButtonBox,
    QFileDialog, QMessageBox, QApplication,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsPixmapItem,
    QGraphicsSimpleTextItem, QSizePolicy, QScrollArea, QFrame,
)


# ---------------------------------------------------------------------------
# Headless data model & geometry — moved to core.labelling; re-exported here
# so existing imports keep working.
# ---------------------------------------------------------------------------

from .gui_utils import fit_to_screen

from .core.labelling import (  # noqa: F401 — re-exported API
    AGE_CLASSES,
    FIELD_SCOPES,
    FIELD_TYPES,
    CustomField,
    LabelStore,
    LabelTrack,
    OCCLUSION_LEVELS,
    RESERVED_FIELD_NAMES,
    SEX_CLASSES,
    SPECIES_CLASSES,
    TRACK_COLORS_RGB,
    _FrameMatcher,
    _GeoPropagator,
    _load_detections_by_frame,
    _load_pixel_tracks,
    _pose_epochs,
    box_in_valid_area,
    custom_fields_from_dicts,
    keyframe_window,
    load_valid_mask,
    propagation_frames,
    read_custom_fields,
    track_color_rgb,
    validate_custom_fields,
    write_custom_fields,
)

#: Where the last configured field schema is remembered, so a newly labelled
#: flight (or the other modality) starts out with the same extra fields.
_FIELDS_SETTING = "bambi/labelling_tool/custom_fields"

#: Folder the import/export file dialogs open in (last one used).
_FIELDS_DIR_SETTING = "bambi/labelling_tool/custom_fields_dir"

#: Width of the controls next to the canvas (the scroll area adds the bar).
_SIDE_PANEL_WIDTH = 300

_FIELDS_FILE_FILTER = "Labelling field schema (*.json);;All files (*)"

# Distinct colours cycled per label track id
_TRACK_COLORS = [QColor(*rgb) for rgb in TRACK_COLORS_RGB]

_DETECTION_COLOR = QColor(80, 140, 255)   # read-only detection overlay
_TRACK_OVERLAY_COLOR = QColor(0, 190, 190)  # read-only track overlay


def _track_color(track_id: int) -> QColor:
    return QColor(*track_color_rgb(track_id))


# ---------------------------------------------------------------------------
# Canvas items
# ---------------------------------------------------------------------------

class _BoxItem(QGraphicsRectItem):
    """A bounding-box rectangle in image pixel coordinates.

    Editable items support moving (drag inside) and resizing (drag edges /
    corners); the final geometry is committed through *on_committed*.
    Non-editable items only report clicks through *on_clicked*.
    """

    def __init__(self, track_id: int, rect: QRectF, color: QColor,
                 label: str, font_px: int, editable: bool = False,
                 dashed: bool = False, emphasized: bool = False,
                 on_committed=None, on_clicked=None):
        super().__init__(rect)
        self._track_id = track_id
        self._editable = editable
        self._on_committed = on_committed
        self._on_clicked = on_clicked
        self._drag = None  # (zone, press_pos, press_rect)

        pen = QPen(color, 3 if emphasized else 2,
                   Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
        pen.setCosmetic(True)
        self.setPen(pen)

        if label:
            text = QGraphicsSimpleTextItem(label, self)
            text.setBrush(color)
            font = QFont("Arial")
            font.setPixelSize(font_px)
            text.setFont(font)
            text.setPos(rect.left() + 2,
                        max(0.0, rect.top() - font_px - 4))

        self.setAcceptHoverEvents(editable)
        if editable or on_clicked is not None:
            self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        else:
            self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    # -- geometry helpers --------------------------------------------------

    def _margin(self) -> float:
        """Hit margin for edge handles, in scene units (zoom independent)."""
        scale = 1.0
        if self.scene() and self.scene().views():
            scale = max(1e-6, self.scene().views()[0].transform().m11())
        return 8.0 / scale

    def _hit_zone(self, pos: QPointF) -> Optional[str]:
        r = self.rect()
        m = self._margin()
        left = abs(pos.x() - r.left()) <= m
        right = abs(pos.x() - r.right()) <= m
        top = abs(pos.y() - r.top()) <= m
        bottom = abs(pos.y() - r.bottom()) <= m
        in_x = r.left() - m <= pos.x() <= r.right() + m
        in_y = r.top() - m <= pos.y() <= r.bottom() + m
        if not (in_x and in_y):
            return None
        zone = ""
        if top:
            zone += "t"
        elif bottom:
            zone += "b"
        if left:
            zone += "l"
        elif right:
            zone += "r"
        return zone or "move"

    def boundingRect(self) -> QRectF:
        m = self._margin()
        return self.rect().adjusted(-m, -m, m, m)

    def shape(self):
        m = self._margin()
        path = QPainterPath()
        path.addRect(self.rect().adjusted(-m, -m, m, m))
        if not self._editable:
            # non-editable boxes are only clickable on their outline so they
            # don't swallow clicks meant for boxes drawn below them
            inner = QPainterPath()
            inner.addRect(self.rect().adjusted(m, m, -m, -m))
            path = path.subtracted(inner)
        return path

    # -- interaction --------------------------------------------------------

    _CURSORS = {
        "move": Qt.CursorShape.SizeAllCursor,
        "t": Qt.CursorShape.SizeVerCursor, "b": Qt.CursorShape.SizeVerCursor,
        "l": Qt.CursorShape.SizeHorCursor, "r": Qt.CursorShape.SizeHorCursor,
        "tl": Qt.CursorShape.SizeFDiagCursor, "br": Qt.CursorShape.SizeFDiagCursor,
        "tr": Qt.CursorShape.SizeBDiagCursor, "bl": Qt.CursorShape.SizeBDiagCursor,
    }

    def hoverMoveEvent(self, event):
        zone = self._hit_zone(event.pos())
        self.setCursor(QCursor(self._CURSORS.get(zone, Qt.CursorShape.ArrowCursor)))
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if self._on_clicked is not None:
            self._on_clicked(self._track_id)
        if self._editable:
            zone = self._hit_zone(event.pos())
            if zone:
                self._drag = (zone, event.pos(), QRectF(self.rect()))
                event.accept()
                return
        event.ignore()

    def mouseMoveEvent(self, event):
        if not self._drag:
            return
        zone, press_pos, press_rect = self._drag
        d = event.pos() - press_pos
        r = QRectF(press_rect)
        if zone == "move":
            r.translate(d)
        else:
            if "t" in zone:
                r.setTop(press_rect.top() + d.y())
            if "b" in zone:
                r.setBottom(press_rect.bottom() + d.y())
            if "l" in zone:
                r.setLeft(press_rect.left() + d.x())
            if "r" in zone:
                r.setRight(press_rect.right() + d.x())
            r = r.normalized()
        if r.width() >= 2 and r.height() >= 2:
            self.prepareGeometryChange()
            self.setRect(r)

    def mouseReleaseEvent(self, event):
        if self._drag:
            self._drag = None
            if self._on_committed is not None:
                self._on_committed(self._track_id, QRectF(self.rect()))
        event.accept()


class _LabelCanvas(QGraphicsView):
    """Zoomable / pannable graphics view holding the frame + box items."""

    boxDrawn = pyqtSignal(QRectF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(30, 30, 30))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._pix_item = QGraphicsPixmapItem()
        self._pix_item.setZValue(-1)
        self.scene().addItem(self._pix_item)

        self._box_items: List[_BoxItem] = []
        self._draw_mode = False
        self._rubber_item = None
        self._rubber_start = None
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

    def clear_boxes(self) -> None:
        for item in self._box_items:
            self.scene().removeItem(item)
        self._box_items = []

    def add_box(self, item: _BoxItem) -> None:
        self.scene().addItem(item)
        self._box_items.append(item)

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = enabled
        self.viewport().setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

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
        if self._draw_mode and event.button() == Qt.MouseButton.LeftButton:
            self._rubber_start = self.mapToScene(event.pos())
            self._rubber_item = QGraphicsRectItem(
                QRectF(self._rubber_start, self._rubber_start))
            pen = QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._rubber_item.setPen(pen)
            self.scene().addItem(self._rubber_item)
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
        if self._rubber_item is not None:
            rect = QRectF(self._rubber_start,
                          self.mapToScene(event.pos())).normalized()
            self._rubber_item.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton and self._pan_last is not None:
            self._pan_last = None
            self.viewport().setCursor(
                Qt.CursorShape.CrossCursor if self._draw_mode else Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if self._rubber_item is not None and event.button() == Qt.MouseButton.LeftButton:
            rect = self._rubber_item.rect().normalized()
            self.scene().removeItem(self._rubber_item)
            self._rubber_item = None
            self._rubber_start = None
            if rect.width() >= 3 and rect.height() >= 3:
                self.boxDrawn.emit(rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _TimelineWidget(QWidget):
    """Slim timeline bar showing the selected track's range and key frames."""

    frameClicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(24)
        self._total = 1
        self._current = 0
        self._keyframes: List[int] = []
        self._segments: List[Tuple[int, int]] = []
        self._stops: List[int] = []

    def set_data(self, total: int, current: int,
                 keyframes: List[int],
                 segments: List[Tuple[int, int]],
                 stops: List[int]) -> None:
        self._total = max(1, total)
        self._current = current
        self._keyframes = keyframes
        self._segments = segments
        self._stops = stops
        self.update()

    def _frame_to_x(self, frame: int) -> float:
        return frame / max(1, self._total - 1) * (self.width() - 1)

    def mousePressEvent(self, event):
        frac = event.pos().x() / max(1, self.width() - 1)
        self.frameClicked.emit(
            int(round(frac * (self._total - 1))))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(45, 45, 45))

        # Selected track's visible segments (gaps after stop frames stay dark)
        for seg in self._segments:
            x0 = self._frame_to_x(seg[0])
            x1 = self._frame_to_x(seg[1])
            painter.fillRect(
                int(x0), 4, max(2, int(x1 - x0)), self.height() - 8,
                QColor(70, 110, 70))

        # Key frames (stop frames in red)
        stops = set(self._stops)
        painter.setPen(QPen(QColor(0, 220, 0), 2))
        for kf in self._keyframes:
            if kf in stops:
                continue
            x = int(self._frame_to_x(kf))
            painter.drawLine(x, 2, x, self.height() - 2)
        painter.setPen(QPen(QColor(230, 60, 60), 2))
        for kf in self._stops:
            x = int(self._frame_to_x(kf))
            painter.drawLine(x, 2, x, self.height() - 2)

        # Current frame cursor
        painter.setPen(QPen(QColor(255, 220, 0), 2))
        x = int(self._frame_to_x(self._current))
        painter.drawLine(x, 0, x, self.height())
        painter.end()


# ---------------------------------------------------------------------------
# Custom field widgets & settings dialog
# ---------------------------------------------------------------------------

#: ISO-8601 spelling used by the datetime editor, matching ``Qt.ISODate`` and
#: :meth:`CustomField.coerce`'s ``datetime.fromisoformat`` validation.
_DATETIME_FORMAT = "yyyy-MM-ddTHH:mm:ss"

#: Human-readable scope names for the settings table.
_SCOPE_LABELS = {"track": "Track", "keyframe": "Key frame"}


def _make_field_widget(field: CustomField, on_changed) -> QWidget:
    """Build the input widget for *field* and wire it to *on_changed*."""
    if field.type == "int":
        widget = QSpinBox()
        widget.setRange(-2_147_483_648, 2_147_483_647)
        widget.valueChanged.connect(on_changed)
    elif field.type == "float":
        widget = QDoubleSpinBox()
        widget.setDecimals(6)
        widget.setRange(-1e12, 1e12)
        widget.valueChanged.connect(on_changed)
    elif field.type == "bool":
        widget = QCheckBox()
        widget.toggled.connect(on_changed)
    elif field.type == "datetime":
        widget = QDateTimeEdit()
        widget.setDisplayFormat(_DATETIME_FORMAT)
        widget.setCalendarPopup(True)
        widget.dateTimeChanged.connect(on_changed)
    else:  # string
        widget = QLineEdit()
        widget.textChanged.connect(on_changed)
    widget.setToolTip(
        f"Custom {_SCOPE_LABELS[field.scope].lower()} field "
        f"'{field.name}' ({field.type}). Stored in labels.json only.")
    return widget


def _field_widget_value(field: CustomField, widget: QWidget) -> Any:
    if field.type in ("int", "float"):
        return widget.value()
    if field.type == "bool":
        return widget.isChecked()
    if field.type == "datetime":
        return widget.dateTime().toString(Qt.DateFormat.ISODate)
    return widget.text()


def _set_field_widget_value(field: CustomField, widget: QWidget,
                            value: Any) -> None:
    """Show *value* in *widget* without disturbing an in-progress edit.

    Every setter is guarded by an equality check: re-rendering the frame
    refreshes the whole side panel after each keystroke, and re-setting a
    line edit's text would move the caret back to the end.
    """
    try:
        value = field.coerce(value)
    except (ValueError, TypeError):
        value = field.default
    if field.type in ("int", "float"):
        if widget.value() != value:
            widget.setValue(value)
    elif field.type == "bool":
        if widget.isChecked() != value:
            widget.setChecked(value)
    elif field.type == "datetime":
        stamp = QDateTime.fromString(value, Qt.DateFormat.ISODate)
        if not stamp.isValid():
            stamp = QDateTime.currentDateTime()
        if widget.dateTime() != stamp:
            widget.setDateTime(stamp)
    else:
        if widget.text() != value:
            widget.setText(value)


class _CustomFieldsDialog(QDialog):
    """Editor for the user-defined attribute schema (name / type / scope).

    The schema can be written to a standalone JSON file and read back, so a
    labelling setup can be shared with colleagues working on other flights.
    """

    _COLUMNS = ("Name", "Data type", "Scope")

    def __init__(self, fields: List[CustomField], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Labelling Tool Settings")
        self.resize(560, 400)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Additional label attributes. <b>Track</b> fields hold one value "
            "per label track (like species); <b>key frame</b> fields hold one "
            "value per key frame (like occlusion) and are inherited by the "
            "interpolated frames that follow.<br><br>"
            "Values are stored in <code>labels.json</code> only — the CSV "
            "export and <code>detections.txt</code> keep their fixed columns "
            "so the rest of the pipeline can still read them.<br><br>"
            "<b>Export</b> writes this configuration to a JSON file you can "
            "pass on; <b>Import</b> reads such a file (or a colleague's "
            "<code>labels.json</code>) back.")
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        intro.setStyleSheet("color: #888;")
        layout.addWidget(intro)

        self.table = QTableWidget(0, len(self._COLUMNS))
        self.table.setHorizontalHeaderLabels(self._COLUMNS)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)

        for field in fields:
            self._append_row(field)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add field")
        add_btn.clicked.connect(lambda: self._append_row(None))
        self.remove_btn = QPushButton("Remove field")
        self.remove_btn.clicked.connect(self._on_remove_row)
        self.import_btn = QPushButton("Import…")
        self.import_btn.setToolTip(
            "Replace the configuration below with the one from a schema file "
            "(or a labels.json).")
        self.import_btn.clicked.connect(self._on_import)
        self.export_btn = QPushButton("Export…")
        self.export_btn.setToolTip(
            "Write the configuration below to a JSON file that can be shared "
            "and imported elsewhere.")
        self.export_btn.clicked.connect(self._on_export)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(self.remove_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.import_btn)
        btn_row.addWidget(self.export_btn)
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _append_row(self, field: Optional[CustomField]) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(
            row, 0, QTableWidgetItem(field.name if field else ""))

        type_combo = QComboBox()
        type_combo.addItems(FIELD_TYPES)
        if field:
            type_combo.setCurrentText(field.type)
        self.table.setCellWidget(row, 1, type_combo)

        scope_combo = QComboBox()
        for scope in FIELD_SCOPES:
            scope_combo.addItem(_SCOPE_LABELS[scope], scope)
        if field:
            scope_combo.setCurrentIndex(FIELD_SCOPES.index(field.scope))
        self.table.setCellWidget(row, 2, scope_combo)

        if field is None:
            self.table.setCurrentCell(row, 0)
            self.table.editItem(self.table.item(row, 0))

    def _on_remove_row(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def set_fields(self, fields: List[CustomField]) -> None:
        """Replace the table's contents with *fields*."""
        self.table.setRowCount(0)
        for field in fields:
            self._append_row(field)

    # -- sharing ---------------------------------------------------------

    @staticmethod
    def _last_dir() -> str:
        return QSettings().value(_FIELDS_DIR_SETTING, "", type=str)

    @staticmethod
    def _remember_dir(path: str) -> None:
        QSettings().setValue(_FIELDS_DIR_SETTING, os.path.dirname(path))

    def _on_import(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, "Import label field configuration",
            self._last_dir(), _FIELDS_FILE_FILTER)
        if not path:
            return
        try:
            fields = read_custom_fields(path)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(
                self, "Import Failed",
                f"Could not read the field configuration:\n\n{exc}")
            return

        if self.table.rowCount():
            reply = QMessageBox.question(
                self, "Import Fields",
                f"Replace the current {self.table.rowCount()} field(s) with "
                f"the {len(fields)} field(s) from this file?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._remember_dir(path)
        self.set_fields(fields)
        # Nothing is applied to the labels yet — OK still has to be pressed,
        # which is where the data-loss check for the new schema runs.
        QMessageBox.information(
            self, "Import Fields",
            f"Imported {len(fields)} field(s). Press OK to apply them.")

    def _on_export(self):
        try:
            fields = self.fields()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Field", str(exc))
            return
        if not fields:
            QMessageBox.information(
                self, "Export Fields", "There are no fields to export.")
            return

        path, _filter = QFileDialog.getSaveFileName(
            self, "Export label field configuration",
            os.path.join(self._last_dir(), "labelling_fields.json"),
            _FIELDS_FILE_FILTER)
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".json"
        try:
            write_custom_fields(path, fields)
        except OSError as exc:
            QMessageBox.warning(
                self, "Export Failed",
                f"Could not write the field configuration:\n\n{exc}")
            return
        self._remember_dir(path)
        QMessageBox.information(
            self, "Export Fields",
            f"Wrote {len(fields)} field(s) to:\n{path}")

    def fields(self) -> List[CustomField]:
        """Return the edited schema. Raises ``ValueError`` when invalid."""
        result: List[CustomField] = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            result.append(CustomField(
                item.text() if item else "",
                self.table.cellWidget(row, 1).currentText(),
                self.table.cellWidget(row, 2).currentData(),
            ))
        validate_custom_fields(result)
        return result

    def _on_accept(self) -> None:
        try:
            self.fields()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Field", str(exc))
            return
        self.accept()


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class LabellingToolDialog(QDialog):
    """Non-modal key-frame labelling tool for extracted frames."""

    def __init__(self, iface, dock_widget=None, parent=None):
        super().__init__(parent)
        self.iface = iface
        self._dock_widget = dock_widget

        self.setWindowTitle("BAMBI Labelling Tool")
        flags = Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint
        flags |= Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint
        self.setWindowFlags(flags)

        # Project state
        self._target_folder = ""
        self._dem_path = ""
        self._correction_path = ""
        self._modality = "t"
        self._images: List[dict] = []          # poses images of the modality
        self._detections: Dict[int, List[tuple]] = {}
        self._pixel_tracks: Dict[int, List[dict]] = {}
        self._store: Optional[LabelStore] = None
        self._propagator: Optional[_GeoPropagator] = None
        self._img_size: Optional[Tuple[int, int]] = None

        # UI state
        self._current_frame = 0
        self._selected_track: Optional[int] = None
        self._dirty = False
        self._updating_ui = False

        # Custom field editors, rebuilt whenever the schema changes
        self._fields_by_name: Dict[str, CustomField] = {}
        self._track_field_widgets: Dict[str, QWidget] = {}
        self._kf_field_widgets: Dict[str, QWidget] = {}

        # Autosave debounce: rapid edits (dragging, typing a species name)
        # collapse into one save shortly after the last change.
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(1000)
        self._autosave_timer.timeout.connect(self._autosave_now)

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
        self.folder_edit.setPlaceholderText("Plugin output folder (contains frames_t / frames_w)")
        top.addWidget(self.folder_edit, 1)
        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(30)
        browse_btn.clicked.connect(self._on_browse_folder)
        top.addWidget(browse_btn)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._on_load_clicked)
        top.addWidget(load_btn)

        top.addSpacing(12)
        top.addWidget(QLabel("Label modality:"))
        self.modality_combo = QComboBox()
        self.modality_combo.addItem("Thermal", "t")
        self.modality_combo.addItem("RGB", "w")
        self.modality_combo.currentIndexChanged.connect(self._on_modality_changed)
        top.addWidget(self.modality_combo)

        top.addStretch()
        self.autosave_check = QCheckBox("Autosave")
        self.autosave_check.setToolTip(
            "Automatically save the labels (labels.json / labels.csv) "
            "shortly after every change.")
        self.autosave_check.setChecked(
            QSettings().value("bambi/labelling_tool/autosave", False, type=bool))
        self.autosave_check.toggled.connect(self._on_autosave_toggled)
        top.addWidget(self.autosave_check)
        self.save_btn = QPushButton("Save Labels")
        self.save_btn.clicked.connect(self._on_save)
        top.addWidget(self.save_btn)

        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setFixedWidth(30)
        self.settings_btn.setToolTip(
            "Settings: configure additional label attributes (per track or "
            "per key frame).")
        self.settings_btn.clicked.connect(self._on_open_settings)
        top.addWidget(self.settings_btn)
        layout.addLayout(top)

        # ---- center: canvas + side panel ----------------------------------
        center = QHBoxLayout()

        self.canvas = _LabelCanvas()
        # NoFocus so arrow keys reach the dialog's keyPressEvent for frame
        # navigation instead of scrolling the graphics view.
        self.canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.canvas.boxDrawn.connect(self._on_box_drawn)
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
        self.prev_step_btn.clicked.connect(lambda: self._go_relative(-self.nav_step_spin.value()))
        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(lambda: self._go_relative(-1))
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(lambda: self._go_relative(1))
        self.next_step_btn = QPushButton("Step >>")
        self.next_step_btn.clicked.connect(lambda: self._go_relative(self.nav_step_spin.value()))

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

        self.timeline = _TimelineWidget()
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

        # Overlays
        overlay_group = QGroupBox("Overlays (read-only)")
        og = QVBoxLayout(overlay_group)
        self.show_det_check = QCheckBox("Show detections")
        self.show_det_check.setChecked(True)
        self.show_det_check.stateChanged.connect(lambda _: self._render_frame())
        self.show_tracks_check = QCheckBox("Show existing tracks")
        self.show_tracks_check.setChecked(True)
        self.show_tracks_check.stateChanged.connect(lambda _: self._render_frame())
        og.addWidget(self.show_det_check)
        og.addWidget(self.show_tracks_check)

        import_row = QHBoxLayout()
        self.import_track_combo = QComboBox()
        self.import_track_combo.setToolTip("Existing pipeline track id")
        import_btn = QPushButton("Import as label track")
        import_btn.setToolTip(
            "Convert the selected pipeline track into an editable label "
            "track.")
        import_btn.clicked.connect(self._on_import_track)
        import_row.addWidget(self.import_track_combo)
        import_row.addWidget(import_btn)
        og.addLayout(import_row)

        resample_row = QHBoxLayout()
        resample_row.addWidget(QLabel("Resample:"))
        self.import_resample_spin = QSpinBox()
        self.import_resample_spin.setRange(1, 500)
        self.import_resample_spin.setValue(1)
        self.import_resample_spin.setToolTip(
            "Keep a key frame only at every N-th frame of the imported "
            "track (first and last frame are always kept). 1 = keep every "
            "frame as a key frame (no simplification).")
        resample_row.addWidget(self.import_resample_spin)
        resample_row.addStretch()
        og.addLayout(resample_row)

        # Cross-modality import: project the other modality's label tracks
        # (RGB <-> thermal) into this modality via the DEM and add them as
        # new, editable label tracks for manual refinement.
        self.copy_from_btn = QPushButton("Copy labels from other modality")
        self.copy_from_btn.setToolTip(
            "Project the label tracks of the other modality (RGB ↔ "
            "thermal) onto this modality's frames via the DEM and add them "
            "as new label tracks. Frames are matched by capture time; the "
            "projected boxes usually need small manual adaptions.")
        self.copy_from_btn.clicked.connect(self._on_copy_from_other_modality)
        og.addWidget(self.copy_from_btn)
        vbox.addWidget(overlay_group)

        # Label tracks
        tracks_group = QGroupBox("Label tracks")
        tg = QVBoxLayout(tracks_group)
        self.track_list = QListWidget()
        self.track_list.currentItemChanged.connect(self._on_track_list_selection)
        tg.addWidget(self.track_list)

        btn_row = QHBoxLayout()
        self.new_track_btn = QPushButton("New Track (N)")
        self.new_track_btn.setCheckable(True)
        self.new_track_btn.setToolTip(
            "Draw a bounding box on the canvas to start a new track at the "
            "current frame.")
        self.new_track_btn.toggled.connect(self._on_new_track_toggled)
        del_track_btn = QPushButton("Delete Track")
        del_track_btn.clicked.connect(self._on_delete_track)
        btn_row.addWidget(self.new_track_btn)
        btn_row.addWidget(del_track_btn)
        tg.addLayout(btn_row)
        vbox.addWidget(tracks_group)

        # Track-level attributes (identity of the animal; per track)
        attr_group = QGroupBox("Track classes")
        ag = QVBoxLayout(attr_group)

        def combo_row(label, combo):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(combo, 1)
            ag.addLayout(row)

        self.species_combo = QComboBox()
        self.species_combo.setEditable(True)
        self.species_combo.addItems(SPECIES_CLASSES)
        # editable combo: react on both selection and manual edit
        self.species_combo.currentTextChanged.connect(self._on_attributes_changed)
        combo_row("Species:", self.species_combo)

        self.sex_combo = QComboBox()
        self.sex_combo.addItems(SEX_CLASSES)
        self.sex_combo.currentTextChanged.connect(self._on_attributes_changed)
        combo_row("Sex:", self.sex_combo)

        self.age_combo = QComboBox()
        self.age_combo.addItems(AGE_CLASSES)
        self.age_combo.currentTextChanged.connect(self._on_attributes_changed)
        combo_row("Age:", self.age_combo)

        # Rows for the user-defined track-scope fields (see the gear button)
        self.track_fields_form = QFormLayout()
        self.track_fields_form.setContentsMargins(0, 0, 0, 0)
        ag.addLayout(self.track_fields_form)
        vbox.addWidget(attr_group)

        # Key frames
        kf_group = QGroupBox("Key frames")
        kg = QVBoxLayout(kf_group)
        self.kf_info_label = QLabel("–")
        self.kf_info_label.setWordWrap(True)
        self.kf_info_label.setTextFormat(Qt.TextFormat.RichText)
        self.kf_info_label.setToolTip(
            "Click a frame number to jump to that key frame.")
        self.kf_info_label.linkActivated.connect(self._on_kf_link_clicked)
        kg.addWidget(self.kf_info_label)
        kf_row = QHBoxLayout()
        add_kf_btn = QPushButton("Set key frame (K)")
        add_kf_btn.setToolTip(
            "Store the currently shown (interpolated) box as a key frame on "
            "this frame. Outside the track's range the nearest key frame's "
            "box is copied here (extending the track).")
        add_kf_btn.clicked.connect(self._on_add_keyframe)
        del_kf_btn = QPushButton("Delete key frame")
        del_kf_btn.clicked.connect(self._on_delete_keyframe)
        kf_row.addWidget(add_kf_btn)
        kf_row.addWidget(del_kf_btn)
        kg.addLayout(kf_row)
        self.draw_kf_btn = QPushButton("Draw key frame (B)")
        self.draw_kf_btn.setCheckable(True)
        self.draw_kf_btn.setToolTip(
            "Draw the selected track's bounding box on the current frame — "
            "also on frames outside the track's current range, extending "
            "the track without geo-propagation.")
        self.draw_kf_btn.toggled.connect(self._on_draw_kf_toggled)
        kg.addWidget(self.draw_kf_btn)
        self.whole_track_check = QCheckBox("Move/resize whole track")
        self.whole_track_check.setToolTip(
            "Apply moving or resizing of the box to the whole track: every "
            "key frame's box is shifted and scaled by the same amount. On "
            "interpolated frames the track is transformed without creating "
            "a new key frame. Unchecked: only the current frame's box "
            "changes (promoting it to a key frame).")
        kg.addWidget(self.whole_track_check)
        # Per-key-frame attribute: occlusion can change along the track
        # (visible on one key frame, occluded on the next).
        occ_row = QHBoxLayout()
        occ_row.addWidget(QLabel("Occlusion:"))
        self.occlusion_combo = QComboBox()
        self.occlusion_combo.addItems(OCCLUSION_LEVELS)
        self.occlusion_combo.setToolTip(
            "Occlusion level of the box at the current frame — stored per "
            "key frame, so it can change along the track. Interpolated "
            "frames inherit the previous key frame's occlusion; changing it "
            "on an interpolated frame promotes that frame to a key frame.")
        self.occlusion_combo.currentTextChanged.connect(self._on_occlusion_changed)
        occ_row.addWidget(self.occlusion_combo, 1)
        kg.addLayout(occ_row)
        self.stop_check = QCheckBox("Stop frame (S)")
        self.stop_check.setToolTip(
            "Mark this key frame as the last sighting before the animal "
            "disappears: no boxes are interpolated between a stop frame and "
            "the next key frame, where the track resumes. Shown red in the "
            "timeline.")
        self.stop_check.toggled.connect(self._on_stop_toggled)
        kg.addWidget(self.stop_check)

        # Rows for the user-defined key-frame-scope fields (see the gear
        # button); like occlusion they promote an interpolated frame to a key
        # frame when edited.
        self.kf_fields_form = QFormLayout()
        self.kf_fields_form.setContentsMargins(0, 0, 0, 0)
        kg.addLayout(self.kf_fields_form)
        vbox.addWidget(kf_group)

        # Geo propagation
        prop_group = QGroupBox("Geo-referenced propagation")
        pg = QVBoxLayout(prop_group)
        prop_row = QHBoxLayout()
        prop_row.addWidget(QLabel("Frame offset:"))
        self.prop_offset_spin = QSpinBox()
        self.prop_offset_spin.setRange(-5000, 5000)
        self.prop_offset_spin.setValue(10)
        self.prop_offset_spin.setToolTip(
            "Frames between the current frame and the propagation target "
            "(negative propagates backwards).")
        prop_row.addWidget(self.prop_offset_spin)
        pg.addLayout(prop_row)
        sample_row = QHBoxLayout()
        sample_row.addWidget(QLabel("Sample every:"))
        self.prop_sample_spin = QSpinBox()
        self.prop_sample_spin.setRange(0, 5000)
        self.prop_sample_spin.setValue(10)
        self.prop_sample_spin.setSuffix(" frames")
        self.prop_sample_spin.setSpecialValueText("off (target only)")
        self.prop_sample_spin.setToolTip(
            "Also create intermediate key frames at this spacing between the "
            "current frame and the target. Boxes are interpolated linearly "
            "between key frames, which only holds while the drone flies "
            "straight — sampling shortens the interpolated spans so curved "
            "ego-motion is followed. 0 propagates to the target frame only.")
        sample_row.addWidget(self.prop_sample_spin)
        pg.addLayout(sample_row)
        self.propagate_btn = QPushButton("Propagate box (geo)")
        self.propagate_btn.setToolTip(
            "Ray-cast the current box onto the DEM and back-project it into "
            "the offset frame — and, if sampling is enabled, into the "
            "intermediate frames; a key frame is created at each. Usually "
            "only small size adaptions are needed afterwards.")
        self.propagate_btn.clicked.connect(self._on_propagate)
        pg.addWidget(self.propagate_btn)
        prop_hint = QLabel(
            "Projection quality depends on calibration and correction "
            "accuracy.")
        prop_hint.setWordWrap(True)
        prop_hint.setStyleSheet("color: #888; font-style: italic;")
        pg.addWidget(prop_hint)
        vbox.addWidget(prop_group)

        # Pipeline export
        export_group = QGroupBox("Pipeline export")
        eg = QVBoxLayout(export_group)
        self.export_det_btn = QPushButton("Add detections to project")
        self.export_det_btn.setToolTip(
            "Save the labels and merge the interpolated label boxes into "
            "detections_{t,w}/detections.txt in the same format as the "
            "'Detect Animals' stage, so geo-referencing and tracking can "
            "use them. Re-exporting replaces the previously added block.")
        self.export_det_btn.clicked.connect(self._on_add_detections_to_project)
        eg.addWidget(self.export_det_btn)
        self.replace_det_btn = QPushButton("Replace detections in project")
        self.replace_det_btn.setToolTip(
            "Save the labels and REPLACE detections_{t,w}/detections.txt "
            "with the interpolated label boxes — the detector output is "
            "discarded and the derived tracking outputs (tracks_{t,w}/, "
            "tracks_pixel_{t,w}/) are deleted. Asks for confirmation "
            "before touching anything.")
        self.replace_det_btn.clicked.connect(
            self._on_replace_detections_in_project)
        eg.addWidget(self.replace_det_btn)
        vbox.addWidget(export_group)

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
        # Reserve the scrollbar width so the controls keep their width whether
        # or not the bar is shown.
        scroll.setFixedWidth(
            _SIDE_PANEL_WIDTH + scroll.verticalScrollBar().sizeHint().width())
        return scroll

    # ------------------------------------------------------------------
    # Custom fields (settings)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_custom_fields() -> List[CustomField]:
        """The last configured schema, used for stores that have none yet."""
        raw = QSettings().value(_FIELDS_SETTING, "", type=str)
        if not raw:
            return []
        try:
            return custom_fields_from_dicts(json.loads(raw))
        except (ValueError, TypeError):
            return []

    def _on_open_settings(self):
        if self._store is None:
            QMessageBox.information(
                self, "BAMBI Labelling Tool",
                "Load a target folder first — custom fields are stored in the "
                "flight's labels.json.")
            return

        dialog = _CustomFieldsDialog(self._store.custom_fields, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        fields = dialog.fields()
        if fields == self._store.custom_fields:
            return

        losses = self._describe_data_loss(fields)
        if losses:
            details = "\n".join(f"• {msg}" for msg in losses)
            reply = QMessageBox.question(
                self, "Change Custom Fields",
                "The new configuration affects labels you already have:\n\n"
                f"{details}\n\nApply anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._store.set_custom_fields(fields)
        QSettings().setValue(
            _FIELDS_SETTING, json.dumps([f.to_dict() for f in fields]))
        self._rebuild_custom_field_widgets()
        self._mark_dirty()
        self._render_frame()

    def _describe_data_loss(self, fields: List[CustomField]) -> List[str]:
        """Warn about stored values the new schema *fields* would discard."""
        if self._store is None:
            return []
        by_name = {f.name: f for f in fields}
        messages: List[str] = []
        for old in self._store.custom_fields:
            count = self._store.count_values(old)
            if not count:
                continue
            new = by_name.get(old.name)
            if new is None:
                messages.append(
                    f"'{old.name}' is removed — its {count} stored value(s) "
                    "will be deleted.")
            elif new.scope != old.scope:
                messages.append(
                    f"'{old.name}' moves from {_SCOPE_LABELS[old.scope]} to "
                    f"{_SCOPE_LABELS[new.scope]} scope — its {count} stored "
                    "value(s) will be deleted.")
            elif new.type != old.type:
                messages.append(
                    f"'{old.name}' becomes a {new.type} — its {count} stored "
                    f"{old.type} value(s) will be converted where possible "
                    "and deleted otherwise.")
        return messages

    def _rebuild_custom_field_widgets(self):
        """Recreate the side-panel editors for the current field schema."""
        self._updating_ui = True
        self._fields_by_name = {}
        self._track_field_widgets = {}
        self._kf_field_widgets = {}
        for form in (self.track_fields_form, self.kf_fields_form):
            while form.rowCount():
                form.removeRow(0)

        for field in (self._store.custom_fields if self._store else []):
            self._fields_by_name[field.name] = field
            if field.scope == "track":
                widget = _make_field_widget(
                    field, partial(self._on_custom_track_changed, field.name))
                self._track_field_widgets[field.name] = widget
                self.track_fields_form.addRow(f"{field.name}:", widget)
            else:
                widget = _make_field_widget(
                    field, partial(self._on_custom_kf_changed, field.name))
                self._kf_field_widgets[field.name] = widget
                self.kf_fields_form.addRow(f"{field.name}:", widget)
        self._updating_ui = False

    def _on_custom_track_changed(self, name: str, *_args):
        """A track-scope custom field was edited — store it on the track.

        Only the edited field is written, so fields the user never touched
        keep their "unset" state instead of picking up a widget default.
        """
        if self._updating_ui:
            return
        track = self._current_track()
        field = self._fields_by_name.get(name)
        if track is None or field is None:
            return
        track.attributes[name] = _field_widget_value(
            field, self._track_field_widgets[name])
        self._mark_dirty()
        self._update_status()

    def _on_custom_kf_changed(self, name: str, *_args):
        """A key-frame-scope custom field was edited.

        Like occlusion the value lives on a key frame, so editing it on an
        interpolated frame promotes that frame to a key frame.
        """
        if self._updating_ui:
            return
        track = self._current_track()
        field = self._fields_by_name.get(name)
        if track is None or field is None:
            return
        res = track.box_at(self._current_frame)
        if res is None:
            return
        attributes = track.attributes_at(self._current_frame) or {}
        attributes[name] = _field_widget_value(
            field, self._kf_field_widgets[name])
        track.set_keyframe(self._current_frame, res[0], occlusion=res[2],
                           attributes=attributes)
        self._mark_dirty()
        self._refresh_track_list()
        self._render_frame()

    def keyPressEvent(self, event):
        """Frame navigation / labelling shortcuts.

        Handled at the dialog level so widgets that need the keys themselves
        (line edits, spin boxes, the editable species combo) keep working —
        keys only arrive here when the focused widget did not consume them.
        The canvas has ``Qt.NoFocus`` so arrow keys navigate frames instead
        of scrolling the view.
        """
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
            self.new_track_btn.toggle()
        elif key == Qt.Key.Key_B:
            self.draw_kf_btn.toggle()
        elif key == Qt.Key.Key_K:
            self._on_add_keyframe()
        elif key == Qt.Key.Key_S:
            if self.stop_check.isEnabled():
                self.stop_check.toggle()
        elif key == Qt.Key.Key_Delete:
            self._on_delete_keyframe()
        elif key == Qt.Key.Key_Escape:
            # Do not close the dialog on Escape — just cancel draw modes.
            if self.new_track_btn.isChecked():
                self.new_track_btn.setChecked(False)
            if self.draw_kf_btn.isChecked():
                self.draw_kf_btn.setChecked(False)
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Project loading
    # ------------------------------------------------------------------

    def apply_dock_defaults(self):
        """Seed folder / DEM / correction from the plugin dock widget."""
        if self._dock_widget is None:
            return
        try:
            config = self._dock_widget.get_config()
        except Exception:  # nosec B110
            return
        folder = config.get("target_folder", "")
        if not self.folder_edit.text() and folder:
            self.folder_edit.setText(folder)
            self._dem_path = config.get("dem_path", "")
            self._correction_path = config.get("correction_path", "")
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
            QMessageBox.warning(self, "BAMBI Labelling Tool",
                                "Please select a valid target folder.")
            return
        if not self._maybe_save_dirty():
            return

        self._target_folder = folder
        if not self._dem_path or not self._correction_path:
            self._resolve_paths_from_layers()

        # Determine available modalities from the poses files
        has_t = os.path.isfile(os.path.join(folder, "poses_t.json"))
        has_w = os.path.isfile(os.path.join(folder, "poses_w.json"))
        if not has_t and not has_w:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
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

    def _resolve_paths_from_layers(self):
        """Borrow DEM / correction paths from BAMBI layers of this folder."""
        try:
            from qgis.core import QgsProject, QgsVectorLayer
            for layer in QgsProject.instance().mapLayers().values():
                if not isinstance(layer, QgsVectorLayer):
                    continue
                if layer.customProperty("bambi_target_folder", "") != self._target_folder:
                    continue
                if not self._dem_path:
                    self._dem_path = layer.customProperty("bambi_dem_path", "")
                if not self._correction_path:
                    self._correction_path = layer.customProperty(
                        "bambi_correction_path", "")
                if self._dem_path and self._correction_path:
                    break
        except Exception:  # nosec B110
            pass
        # Common fallbacks inside the folder
        if not self._correction_path:
            candidate = os.path.join(self._target_folder, "correction.json")
            if os.path.isfile(candidate):
                self._correction_path = candidate

    def _on_modality_changed(self):
        if self._updating_ui or not self._target_folder:
            return
        if not self._maybe_save_dirty():
            return
        self._load_modality()

    def _load_modality(self):
        """(Re)load poses, overlays and label store for the current modality."""
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
                self, "BAMBI Labelling Tool",
                f"Could not read frames from {poses_path}.")
            return

        self._detections = _load_detections_by_frame(
            os.path.join(self._target_folder, f"detections_{m}", "detections.txt"))
        self._pixel_tracks = _load_pixel_tracks(
            os.path.join(self._target_folder, f"tracks_{m}", "tracks_pixel.csv"))

        self._store = LabelStore(self._target_folder, m)
        try:
            self._store.load()
        except Exception as exc:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                f"Could not load existing labels:\n{exc}")
        if not self._store.schema_defined:
            # Labels written before / without a schema: start from the fields
            # the user configured last instead of showing none.
            self._store.set_custom_fields(self._default_custom_fields())
        self._rebuild_custom_field_widgets()
        self._dirty = False
        self._propagator = None  # modality-specific, rebuild lazily
        self._img_size = None
        self._selected_track = None

        # Navigation ranges
        total = len(self._images)
        self._updating_ui = True
        self.frame_spin.setRange(0, total - 1)
        self.frame_slider.setRange(0, total - 1)
        self.import_track_combo.clear()
        for tid in sorted(self._pixel_tracks.keys()):
            self.import_track_combo.addItem(f"Track {tid}", tid)
        other_name = "RGB" if m == "t" else "Thermal"
        self.copy_from_btn.setText(f"Copy labels from {other_name}")
        self._updating_ui = False

        self._current_frame = min(self._current_frame, total - 1)
        self._refresh_track_list()
        self._goto_frame(self._current_frame, force=True)

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
        if not self._images or self._store is None:
            return
        frame = self._current_frame

        path = self._frame_image_path(frame)
        pixmap = QPixmap(path) if path else QPixmap()
        if pixmap.isNull():
            pixmap = QPixmap(640, 512)
            pixmap.fill(QColor(40, 40, 40))
        if self._img_size is None and path:
            self._img_size = (pixmap.width(), pixmap.height())
        self.canvas.set_frame(pixmap)
        font_px = max(10, pixmap.width() // 80)

        self.canvas.clear_boxes()

        # ---- read-only overlays -------------------------------------------
        if self.show_det_check.isChecked():
            for det in self._detections.get(frame, []):
                item = _BoxItem(
                    -1, QRectF(det[0], det[1], det[2] - det[0], det[3] - det[1]),
                    _DETECTION_COLOR, f"det {det[4]:.2f}", font_px)
                item.setZValue(1)
                self.canvas.add_box(item)

        if self.show_tracks_check.isChecked():
            for tid, entries in self._pixel_tracks.items():
                for d in entries:
                    if d["frame"] == frame:
                        item = _BoxItem(
                            -1, QRectF(d["x1"], d["y1"],
                                       d["x2"] - d["x1"], d["y2"] - d["y1"]),
                            _TRACK_OVERLAY_COLOR, f"T{tid}", font_px)
                        item.setZValue(1)
                        self.canvas.add_box(item)

        # ---- label tracks ---------------------------------------------------
        for track in self._store.tracks.values():
            res = track.box_at(frame)
            if res is None:
                continue
            (x1, y1, x2, y2), is_kf, _occ = res
            selected = track.track_id == self._selected_track
            label = f"L{track.track_id} {track.species}"
            if is_kf and track.is_stop(frame):
                label += " [stop]"
            item = _BoxItem(
                track.track_id,
                QRectF(x1, y1, x2 - x1, y2 - y1),
                _track_color(track.track_id),
                label,
                font_px,
                editable=selected,
                dashed=not is_kf,
                emphasized=selected,
                on_committed=self._on_box_committed,
                on_clicked=self._select_track,
            )
            item.setZValue(3 if selected else 2)
            self.canvas.add_box(item)

        self._update_side_panel()
        self._update_timeline()
        self._update_status()

    def _update_timeline(self):
        keyframes: List[int] = []
        segments: List[Tuple[int, int]] = []
        stops: List[int] = []
        if self._selected_track is not None and self._store:
            track = self._store.tracks.get(self._selected_track)
            if track:
                keyframes = track.frames()
                segments = track.visible_segments()
                stops = [f for f in keyframes if track.is_stop(f)]
        self.timeline.set_data(
            len(self._images), self._current_frame, keyframes, segments, stops)

    def _update_status(self):
        parts = [f"Frame {self._current_frame} / {len(self._images) - 1}"]
        if self._selected_track is not None and self._store:
            track = self._store.tracks.get(self._selected_track)
            if track:
                res = track.box_at(self._current_frame)
                rng = track.frame_range()
                if res is None:
                    if rng and rng[0] <= self._current_frame <= rng[1]:
                        parts.append(
                            f"track L{track.track_id}: gap (after stop frame)")
                    else:
                        parts.append(
                            f"track L{track.track_id}: outside key-frame range")
                elif res[1]:
                    suffix = " (stop)" if track.is_stop(self._current_frame) else ""
                    parts.append(f"track L{track.track_id}: KEY FRAME{suffix}")
                    parts.append(f"occlusion: {res[2]}")
                else:
                    parts.append(f"track L{track.track_id}: interpolated")
                    parts.append(f"occlusion: {res[2]}")
        if self._dirty:
            parts.append("unsaved changes")
        self.status_label.setText("   |   ".join(parts))

    def _update_side_panel(self):
        """Sync attribute editors with the selected track / current frame."""
        self._updating_ui = True
        track = None
        if self._selected_track is not None and self._store:
            track = self._store.tracks.get(self._selected_track)

        # Track-level classes follow the selection …
        enabled = track is not None
        for w in (self.species_combo, self.sex_combo, self.age_combo):
            w.setEnabled(enabled)
        for w in self._track_field_widgets.values():
            w.setEnabled(enabled)

        if track is not None:
            self.species_combo.setCurrentText(track.species)
            self.sex_combo.setCurrentText(track.sex)
            self.age_combo.setCurrentText(track.age)
            for name, widget in self._track_field_widgets.items():
                field = self._fields_by_name[name]
                _set_field_widget_value(
                    field, widget, track.attributes.get(name, field.default))
            # … while occlusion and the stop flag are per key frame: they
            # need a box on the current frame.
            res = track.box_at(self._current_frame)
            if res is not None:
                self.occlusion_combo.setCurrentText(res[2])
            self.occlusion_combo.setEnabled(res is not None)
            self.stop_check.setEnabled(res is not None)
            self.stop_check.setChecked(track.is_stop(self._current_frame))
            kf_attributes = track.attributes_at(self._current_frame) or {}
            for name, widget in self._kf_field_widgets.items():
                field = self._fields_by_name[name]
                widget.setEnabled(res is not None)
                _set_field_widget_value(
                    field, widget, kf_attributes.get(name, field.default))
            kfs = track.frames()

            def _kf_anchor(f: int) -> str:
                # Anchor per key frame — clicking jumps to that frame.
                # Stop frames are red, matching the timeline; the current
                # frame is bold.
                text = str(f)
                if track.is_stop(f):
                    text = f'<span style="color:#e63c3c;">{text}</span>'
                if f == self._current_frame:
                    text = f"<b>{text}</b>"
                return f'<a href="{f}">{text}</a>'

            # Long lists are elided around the current frame, keeping the
            # track's first and last key frame anchored.
            kf_list = ", ".join(
                "…" if f is None else _kf_anchor(f)
                for f in keyframe_window(kfs, self._current_frame))
            self.kf_info_label.setText(f"{len(kfs)} key frame(s): {kf_list}")
        else:
            self.kf_info_label.setText("–")
            self.occlusion_combo.setEnabled(False)
            self.stop_check.setEnabled(False)
            self.stop_check.setChecked(False)
            for widget in self._kf_field_widgets.values():
                widget.setEnabled(False)
        self._updating_ui = False

    def _refresh_track_list(self):
        self._updating_ui = True
        self.track_list.clear()
        if self._store:
            for track in sorted(self._store.tracks.values(),
                                key=lambda t: t.track_id):
                rng = track.frame_range()
                rng_str = f"{rng[0]}–{rng[1]}" if rng else "empty"
                item = QListWidgetItem(
                    f"L{track.track_id}  {track.species}  [{rng_str}]")
                item.setData(Qt.ItemDataRole.UserRole, track.track_id)
                item.setForeground(_track_color(track.track_id))
                self.track_list.addItem(item)
                if track.track_id == self._selected_track:
                    self.track_list.setCurrentItem(item)
        self._updating_ui = False

    # ------------------------------------------------------------------
    # Track selection / creation / deletion
    # ------------------------------------------------------------------

    def _select_track(self, track_id: Optional[int]):
        if track_id == -1:  # read-only overlay clicked
            return
        if track_id == self._selected_track:
            return
        self._selected_track = track_id
        self._refresh_track_list()
        # Deferred: this is reached from a _BoxItem mouse event — re-rendering
        # immediately would delete the item that is still handling the event.
        QTimer.singleShot(0, self._render_frame)

    def _on_track_list_selection(self, current, _previous):
        if self._updating_ui or current is None:
            return
        tid = current.data(Qt.ItemDataRole.UserRole)
        if tid != self._selected_track:
            self._selected_track = tid
            # Jump into the track's range so the user sees the box
            track = self._store.tracks.get(tid) if self._store else None
            if track:
                rng = track.frame_range()
                if rng and not (rng[0] <= self._current_frame <= rng[1]):
                    self._goto_frame(rng[0], force=True)
                    return
            self._render_frame()

    def _on_new_track_toggled(self, checked: bool):
        if checked and self.draw_kf_btn.isChecked():
            self.draw_kf_btn.setChecked(False)
        self._update_draw_mode()

    def _on_draw_kf_toggled(self, checked: bool):
        if checked:
            if self._current_track() is None:
                QMessageBox.information(
                    self, "BAMBI Labelling Tool",
                    "Please select a label track first.")
                self.draw_kf_btn.setChecked(False)
                return
            if self.new_track_btn.isChecked():
                self.new_track_btn.setChecked(False)
        self._update_draw_mode()

    def _update_draw_mode(self):
        self.canvas.set_draw_mode(
            self.new_track_btn.isChecked() or self.draw_kf_btn.isChecked())

    def _clamp_rect(self, rect: QRectF) -> QRectF:
        """Clamp *rect* to the frame image area (scene rect)."""
        scene_rect = self.canvas.scene().sceneRect()
        clamped = rect.intersected(scene_rect)
        return clamped if not clamped.isEmpty() else rect

    def _on_box_drawn(self, rect: QRectF):
        """A box was drawn on the canvas (New Track / Draw Key Frame mode)."""
        if self._store is None:
            return
        rect = self._clamp_rect(rect)

        if self.draw_kf_btn.isChecked():
            # Key frame for the selected track at the current frame — also
            # valid outside the track's current range (extends the track).
            self.draw_kf_btn.setChecked(False)
            track = self._current_track()
            if track is None:
                return
            track.set_keyframe(
                self._current_frame,
                (rect.left(), rect.top(), rect.right(), rect.bottom()))
            self._mark_dirty()
            self._refresh_track_list()
            self._render_frame()
            return

        self.new_track_btn.setChecked(False)
        track = LabelTrack(self._store.next_track_id())
        track.species = self.species_combo.currentText() or "unknown"
        track.set_keyframe(
            self._current_frame,
            (rect.left(), rect.top(), rect.right(), rect.bottom()),
            occlusion="none")
        self._store.tracks[track.track_id] = track
        self._selected_track = track.track_id
        self._mark_dirty()
        self._refresh_track_list()
        self._render_frame()

    def _on_delete_track(self):
        track = self._current_track()
        if track is None:
            return
        reply = QMessageBox.question(
            self, "Delete Track",
            f"Delete label track L{track.track_id} "
            f"({len(track.keyframes)} key frame(s))?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        del self._store.tracks[track.track_id]
        self._selected_track = None
        self._mark_dirty()
        self._refresh_track_list()
        self._render_frame()

    def _on_import_track(self):
        """Convert the selected pipeline track into an editable label track."""
        if self._store is None or self.import_track_combo.currentIndex() < 0:
            return
        src_tid = self.import_track_combo.currentData()
        entries = self._pixel_tracks.get(src_tid, [])
        if not entries:
            return

        step = self.import_resample_spin.value()
        track = LabelTrack(self._store.next_track_id())
        frames = [d["frame"] for d in entries]
        first, last = frames[0], frames[-1]
        for d in entries:
            f = d["frame"]
            if f == first or f == last or (f - first) % step == 0:
                track.set_keyframe(
                    f, (d["x1"], d["y1"], d["x2"], d["y2"]), occlusion="none")

        self._store.tracks[track.track_id] = track
        self._selected_track = track.track_id
        self._mark_dirty()
        self._refresh_track_list()
        self._goto_frame(first, force=True)

    def _current_track(self) -> Optional[LabelTrack]:
        if self._store is None or self._selected_track is None:
            return None
        return self._store.tracks.get(self._selected_track)

    # ------------------------------------------------------------------
    # Cross-modality label import (RGB <-> thermal)
    # ------------------------------------------------------------------

    def _modality_image_size(
            self, modality: str, images: List[dict]) -> Optional[Tuple[int, int]]:
        """Return the pixel size of *modality*'s frames, or ``None``.

        Read from the first extracted frame image that exists on disk, since
        the poses file does not store the resolution.
        """
        frames_dir = os.path.join(self._target_folder, f"frames_{modality}")
        for img in images:
            imagefile = img.get("imagefile", "")
            if not imagefile:
                continue
            path = os.path.join(frames_dir, imagefile)
            if os.path.isfile(path):
                pix = QPixmap(path)
                if not pix.isNull():
                    return pix.width(), pix.height()
        return None

    def _on_copy_from_other_modality(self):
        """Project the other modality's label tracks onto this modality.

        Frames are matched across modalities by capture time; each source
        key-frame box is ray-cast onto the DEM with the source camera and
        back-projected with this modality's camera at the matched frame. The
        results are added as new label tracks (manual refinement expected).
        """
        if self._store is None or not self._images:
            QMessageBox.information(
                self, "BAMBI Labelling Tool",
                "Load a target folder first.")
            return

        other = "w" if self._modality == "t" else "t"
        other_name = "RGB" if other == "w" else "Thermal"
        this_name = "RGB" if self._modality == "w" else "Thermal"

        # ---- source poses -------------------------------------------------
        other_poses = os.path.join(self._target_folder, f"poses_{other}.json")
        if not os.path.isfile(other_poses):
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                f"No {other_name} poses found ({os.path.basename(other_poses)}).\n"
                f"Process the {other_name} camera first.")
            return
        try:
            with open(other_poses, "r", encoding="utf-8") as fh:
                other_images = json.load(fh).get("images", [])
        except Exception as exc:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                f"Could not read {other_name} poses:\n{exc}")
            return
        if not other_images:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                f"{other_name} poses contain no frames.")
            return

        # ---- source labels ------------------------------------------------
        other_store = LabelStore(self._target_folder, other)
        try:
            other_store.load()
        except Exception as exc:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                f"Could not load {other_name} labels:\n{exc}")
            return
        src_tracks = [t for t in other_store.tracks.values() if t.keyframes]
        if not src_tracks:
            QMessageBox.information(
                self, "BAMBI Labelling Tool",
                f"No {other_name} label tracks found to copy.\n"
                f"Create labels on the {other_name} modality first.")
            return

        # ---- frame correspondence by capture time -------------------------
        matcher = _FrameMatcher(other_images, self._images)
        if not matcher.usable:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                "Cannot match frames across modalities: the poses files carry "
                "no usable capture timestamps.")
            return

        # ---- resolutions --------------------------------------------------
        src_size = self._modality_image_size(other, other_images)
        if src_size is None:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                f"Could not read a {other_name} frame image to determine its "
                f"resolution (frames_{other}/). Extract the {other_name} "
                "frames first.")
            return
        if self._img_size is None:
            self._img_size = self._modality_image_size(
                self._modality, self._images)
        if self._img_size is None:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                f"Could not determine this modality's frame resolution "
                f"(frames_{self._modality}/).")
            return

        # ---- DEM propagator ----------------------------------------------
        if self._propagator is None:
            if not self._dem_path:
                self._resolve_paths_from_layers()
            self._propagator = _GeoPropagator(
                self._target_folder, self._dem_path,
                self._correction_path, self._modality)
        if not self._propagator.is_loaded:
            reply = QMessageBox.question(
                self, "Load Digital Elevation Model",
                f"Copying labels from {other_name} needs to load the DEM "
                "mesh.\n\nThis may take some time on the first use. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
            if reply != QMessageBox.StandardButton.Yes:
                return

        # ---- project every source key frame -------------------------------
        src_w, src_h = src_size
        dst_w, dst_h = self._img_size
        added_tracks = 0
        added_keyframes = 0
        skipped_keyframes = 0
        first_new_frame: Optional[int] = None

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if not self._propagator.is_loaded:
                self._propagator.load()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                f"Could not load the DEM mesh:\n{exc}")
            return

        try:
            for src in sorted(src_tracks, key=lambda t: t.track_id):
                new_track = LabelTrack(
                    self._store.next_track_id(),
                    species=src.species, sex=src.sex, age=src.age,
                    attributes=dict(src.attributes))
                for sf in src.frames():
                    kf = src.keyframes[sf]
                    m = matcher.match(sf)
                    if m is None:
                        skipped_keyframes += 1
                        continue
                    dst_frame, _dt = m
                    try:
                        new_box = self._propagator.propagate_between(
                            (kf["x1"], kf["y1"], kf["x2"], kf["y2"]),
                            sf, other_images, src_w, src_h,
                            dst_frame, self._images, dst_w, dst_h)
                    except Exception:
                        skipped_keyframes += 1
                        continue
                    new_track.set_keyframe(
                        dst_frame, new_box,
                        occlusion=kf.get("occlusion", "none"),
                        stop=bool(kf.get("stop", False)),
                        attributes=dict(kf.get("attributes", {})))
                    added_keyframes += 1
                    if first_new_frame is None:
                        first_new_frame = dst_frame
                if new_track.keyframes:
                    self._store.tracks[new_track.track_id] = new_track
                    added_tracks += 1
            # The source modality may define other custom fields — keep only
            # the values this modality's schema knows about.
            self._store.set_custom_fields(self._store.custom_fields)
        finally:
            QApplication.restoreOverrideCursor()

        if added_tracks == 0:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                f"None of the {other_name} label boxes could be projected onto "
                f"the {this_name} frames (no DEM intersection or all outside "
                "the frames / the valid mask area).")
            return

        self._mark_dirty()
        self._refresh_track_list()
        if first_new_frame is not None:
            self._goto_frame(first_new_frame, force=True)
        else:
            self._render_frame()

        msg = (f"Copied {added_tracks} track(s) from {other_name} "
               f"({added_keyframes} key frame(s)).")
        if skipped_keyframes:
            msg += (f"\n\n{skipped_keyframes} key frame(s) were skipped "
                    "(no DEM intersection, outside the target frame, or no "
                    "time-matched frame).")
        msg += "\n\nThe projected boxes were added as new tracks — review and "
        msg += "adjust them before exporting."
        QMessageBox.information(self, "BAMBI Labelling Tool", msg)

    # ------------------------------------------------------------------
    # Key frame editing
    # ------------------------------------------------------------------

    def _on_box_committed(self, track_id: int, rect: QRectF):
        """Selected box was moved/resized — write a key frame here.

        In "Move/resize whole track" mode the same shift and scaling is
        applied to all key frames of the track instead.
        """
        track = self._store.tracks.get(track_id) if self._store else None
        if track is None:
            return
        rect = self._clamp_rect(rect)
        new_box = (rect.left(), rect.top(), rect.right(), rect.bottom())
        old = (track.box_at(self._current_frame)
               if self.whole_track_check.isChecked() else None)
        if old is not None:
            scene_rect = self.canvas.scene().sceneRect()
            track.transform_keyframes(
                old[0], new_box,
                bounds=(scene_rect.width(), scene_rect.height()))
        else:
            track.set_keyframe(self._current_frame, new_box)
        self._mark_dirty()
        self._refresh_track_list()
        # Deferred: reached from the _BoxItem's mouseReleaseEvent — see
        # _select_track.
        QTimer.singleShot(0, self._render_frame)

    def _on_add_keyframe(self):
        """Freeze the interpolated box at the current frame as a key frame."""
        track = self._current_track()
        if track is None:
            return
        res = track.box_at(self._current_frame)
        if res is None:
            # Outside the track's range: extend it by copying the nearest
            # key frame's box here; the user then adjusts it by dragging.
            fs = track.frames()
            nearest = min(fs, key=lambda f: abs(f - self._current_frame))
            kf = track.keyframes[nearest]
            track.set_keyframe(
                self._current_frame,
                (kf["x1"], kf["y1"], kf["x2"], kf["y2"]),
                occlusion=kf.get("occlusion", "none"))
        elif res[1]:
            return  # already a key frame
        else:
            track.set_keyframe(self._current_frame, res[0], occlusion=res[2])
        self._mark_dirty()
        self._refresh_track_list()
        self._render_frame()

    def _on_delete_keyframe(self):
        track = self._current_track()
        if track is None:
            return
        if not track.remove_keyframe(self._current_frame):
            return
        if not track.keyframes:
            del self._store.tracks[track.track_id]
            self._selected_track = None
        self._mark_dirty()
        self._refresh_track_list()
        self._render_frame()

    def _on_attributes_changed(self):
        if self._updating_ui:
            return
        track = self._current_track()
        if track is None:
            return
        track.species = self.species_combo.currentText()
        track.sex = self.sex_combo.currentText()
        track.age = self.age_combo.currentText()
        self._mark_dirty()
        self._refresh_track_list()
        self._render_frame()

    def _on_occlusion_changed(self):
        if self._updating_ui:
            return
        track = self._current_track()
        if track is None:
            return
        res = track.box_at(self._current_frame)
        if res is None:
            return
        # Occlusion is stored per key frame; setting it on an interpolated
        # frame promotes that frame to a key frame.
        track.set_keyframe(
            self._current_frame, res[0],
            occlusion=self.occlusion_combo.currentText())
        self._mark_dirty()
        self._render_frame()

    def _on_kf_link_clicked(self, href: str):
        """A key-frame anchor in the overview label was clicked — jump there."""
        try:
            self._goto_frame(int(href))
        except ValueError:
            pass

    def _on_stop_toggled(self, checked: bool):
        if self._updating_ui:
            return
        track = self._current_track()
        if track is None:
            return
        res = track.box_at(self._current_frame)
        if res is None:
            return
        # Like occlusion, the stop flag lives on a key frame; toggling it on
        # an interpolated frame promotes that frame to a key frame.
        track.set_keyframe(
            self._current_frame, res[0], occlusion=res[2], stop=checked)
        self._mark_dirty()
        self._refresh_track_list()
        self._render_frame()

    # ------------------------------------------------------------------
    # Geo-referenced propagation
    # ------------------------------------------------------------------

    def _on_propagate(self):
        track = self._current_track()
        if track is None:
            QMessageBox.information(
                self, "BAMBI Labelling Tool",
                "Please select a label track first.")
            return
        res = track.box_at(self._current_frame)
        if res is None:
            QMessageBox.information(
                self, "BAMBI Labelling Tool",
                "The selected track has no box on the current frame.")
            return
        offset = self.prop_offset_spin.value()
        if offset == 0:
            return
        dst_frame = self._current_frame + offset
        if not (0 <= dst_frame < len(self._images)):
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                f"Target frame {dst_frame} is outside the frame range.")
            return

        if self._propagator is None:
            if not self._dem_path:
                self._resolve_paths_from_layers()
            self._propagator = _GeoPropagator(
                self._target_folder, self._dem_path,
                self._correction_path, self._modality)

        if not self._propagator.is_loaded:
            reply = QMessageBox.question(
                self, "Load Digital Elevation Model",
                "Geo-referenced propagation needs to load the DEM mesh.\n\n"
                "This may take some time on the first use. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
            if reply != QMessageBox.StandardButton.Yes:
                return

        if self._img_size is None:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                "Frame image size unknown — cannot project.")
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            boxes, failures = self._propagator.propagate_series(
                res[0], self._current_frame, dst_frame, self._images,
                self._img_size[0], self._img_size[1],
                step=self.prop_sample_spin.value())
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Propagation Failed", str(exc))
            return
        QApplication.restoreOverrideCursor()

        if not boxes:
            QMessageBox.warning(
                self, "Propagation Failed",
                failures[0][1] if failures else
                "No frames to propagate to.")
            return

        for frame, new_box in boxes:
            track.set_keyframe(frame, new_box, occlusion=res[2])
        self._mark_dirty()
        self._refresh_track_list()
        self._goto_frame(boxes[-1][0], force=True)

        if failures:
            skipped = ", ".join(str(f) for f, _ in failures)
            QMessageBox.information(
                self, "BAMBI Labelling Tool",
                f"Created {len(boxes)} key frame(s).\n\n"
                f"Skipped frame(s) {skipped}: the projected box lies outside "
                "the frame or the valid (white) mask area.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _mark_dirty(self):
        self._dirty = True
        if self.autosave_check.isChecked():
            self._autosave_timer.start()  # (re)start the debounce interval

    def _on_autosave_toggled(self, checked: bool):
        QSettings().setValue("bambi/labelling_tool/autosave", checked)
        if checked and self._dirty:
            self._autosave_now()
        elif not checked:
            self._autosave_timer.stop()

    def _autosave_now(self):
        """Debounced autosave — silent except for the status line."""
        if not self._dirty or self._store is None:
            return
        try:
            self._store.save()
        except Exception as exc:
            # Do not nag once per edit: disable autosave and tell the user.
            self.autosave_check.setChecked(False)
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                f"Autosave failed — autosave has been disabled:\n{exc}")
            return
        self._dirty = False
        self._update_status()
        self.status_label.setText(
            self.status_label.text() + "   |   autosaved")

    def _on_save(self):
        if self._store is None:
            return
        self._autosave_timer.stop()  # manual save supersedes a pending one
        try:
            self._store.save()
        except Exception as exc:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool", f"Could not save labels:\n{exc}")
            return
        self._dirty = False
        self._update_status()
        self.status_label.setText(
            f"Saved {len(self._store.tracks)} track(s) to {self._store.json_path}")

    def _on_add_detections_to_project(self):
        """Merge the label boxes into the pipeline's detections.txt."""
        if self._store is None:
            return
        if not any(t.keyframes for t in self._store.tracks.values()):
            QMessageBox.information(
                self, "BAMBI Labelling Tool",
                "There are no label tracks with key frames to export.")
            return
        try:
            # Keep labels.json/csv in sync with what is exported.
            self._store.save()
            self._dirty = False
            det_file, n_boxes = self._store.export_to_detections()
        except Exception as exc:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                f"Could not export detections:\n{exc}")
            return

        # Refresh the read-only overlay so the exported boxes show up.
        self._detections = _load_detections_by_frame(det_file)
        self._render_frame()

        QMessageBox.information(
            self, "BAMBI Labelling Tool",
            f"Added {n_boxes} bounding box(es) from "
            f"{len(self._store.tracks)} label track(s) to:\n{det_file}\n\n"
            "Re-run 'Geo-Reference Detections' (and tracking, if needed) in "
            "the plugin panel to update the QGIS layers.")

    def _on_replace_detections_in_project(self):
        """Replace the pipeline's detections (and derived tracks) with the labels."""
        if self._store is None:
            return
        if not any(t.keyframes for t in self._store.tracks.values()):
            QMessageBox.information(
                self, "BAMBI Labelling Tool",
                "There are no label tracks with key frames to export.")
            return
        m = self._modality
        modality_name = "RGB" if m == "w" else "Thermal"
        reply = QMessageBox.warning(
            self, "Replace Detections in Project",
            f"This replaces the {modality_name} pipeline results with the "
            "label tracks:\n\n"
            f"• detections_{m}/detections.txt is overwritten with the "
            "interpolated label boxes — all detector output is discarded\n"
            f"• the derived tracking outputs (tracks_{m}/, "
            f"tracks_pixel_{m}/) are deleted\n\n"
            "This cannot be undone. Replace them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            # Keep labels.json/csv in sync with what is exported.
            self._store.save()
            self._dirty = False
            det_file, n_boxes, removed = self._store.replace_detections()
        except Exception as exc:
            QMessageBox.warning(
                self, "BAMBI Labelling Tool",
                f"Could not replace detections:\n{exc}")
            return

        # Refresh the read-only overlays: the detections now show the label
        # boxes and the deleted pipeline tracks disappear.
        self._detections = _load_detections_by_frame(det_file)
        self._pixel_tracks = _load_pixel_tracks(
            os.path.join(self._target_folder, f"tracks_{m}", "tracks_pixel.csv"))
        self._render_frame()

        removed_note = ""
        if removed:
            removed_note = "Deleted stale tracking output:\n" + "\n".join(
                removed) + "\n\n"
        QMessageBox.information(
            self, "BAMBI Labelling Tool",
            f"Replaced the detections with {n_boxes} bounding box(es) from "
            f"{len(self._store.tracks)} label track(s) in:\n{det_file}\n\n"
            f"{removed_note}"
            "Re-run 'Geo-Reference Detections' and 'Track Animals' in the "
            "plugin panel to rebuild the QGIS layers from the labels.")

    def _maybe_save_dirty(self) -> bool:
        """Ask the user about unsaved changes. Returns False on cancel."""
        if self._dirty and self._store is not None \
                and self.autosave_check.isChecked():
            # Autosave mode: flush pending changes without prompting.
            self._autosave_timer.stop()
            self._autosave_now()
        if not self._dirty or self._store is None:
            return True
        reply = QMessageBox.question(
            self, "Unsaved Labels",
            "There are unsaved label changes. Save them now?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save)
        if reply == QMessageBox.StandardButton.Cancel:
            return False
        if reply == QMessageBox.StandardButton.Save:
            self._on_save()
        else:
            # Discard: restore the on-disk state so the in-memory store does
            # not silently keep (and later save) the discarded changes.
            self._dirty = False
            self._selected_track = None
            try:
                self._store.load()
            except Exception:  # nosec B110
                self._store.tracks = {}
            if not self._store.schema_defined:
                self._store.set_custom_fields(self._default_custom_fields())
            self._rebuild_custom_field_widgets()
            self._refresh_track_list()
            self._render_frame()
        return True

    def closeEvent(self, event):
        if not self._maybe_save_dirty():
            event.ignore()
            return
        super().closeEvent(event)
