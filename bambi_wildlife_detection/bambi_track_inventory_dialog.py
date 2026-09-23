# -*- coding: utf-8 -*-
"""The track inventory as a report window.

The CSV is for spreadsheets and the point layer for the map; this is for
reading and checking. One row per tracked individual, the columns the user
chose (all of them by default), sortable by any of them, with a text filter
for finding one animal and the summary lines on top. The first column is
the user's own verdict - an "Approved" checkmark that is written back to the
track, so it survives, exports and shows up in every later inventory. Copy
puts the table on the clipboard as tab-separated text, which pastes straight
into a spreadsheet or a document.
"""
import csv
import io
from typing import Callable, List, Optional, Sequence

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QBrush, QColor, QFont
from qgis.PyQt.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QDialogButtonBox, QFileDialog,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from .core import track_inventory

#: Columns shown first; the rest follow in ``track_inventory.COLUMNS`` order.
_LEAD = ("approved", "track_id", "species", "sex", "age", "start_frame",
         "end_frame", "duration_s", "n_boxes", "n_occluded", "start_lat",
         "start_lon", "end_lat", "end_lon", "matched_track")

#: Columns that stay whatever the user chose: the verdict and the identity.
ALWAYS_SHOWN = ("approved", "track_id")

#: What a column holds, for the column chooser.
COLUMN_HINTS = {
    "approved": "Your own checkmark: the row has been checked and is correct",
    "track_id": "Track number of the pipeline",
    "camera": "Which camera the track comes from",
    "species": "Species on the track (classifier, annotation or label sync)",
    "species_vote_fraction": "Share of the voting frames that agreed",
    "species_votes": "Votes for the winning species / voting frames",
    "species_model": "Classifier file the species came from",
    "sex": "Sex on the track", "sex_vote_fraction": "Share of frames agreeing",
    "sex_model": "Classifier file the sex came from",
    "age": "Life stage on the track", "age_source": "Model or size estimate",
    "n_boxes": "Detections in the track",
    "n_occluded": "Detections the occlusion classifier called occluded",
    "n_clear": "Detections called clear",
    "n_occlusion_unknown": "Detections without an occlusion call",
    "n_interpolated": "Boxes the tracker interpolated",
    "n_georeferenced": "Detections with a ground position",
    "mean_confidence": "Mean detector confidence",
    "min_confidence": "Lowest detector confidence",
    "max_confidence": "Highest detector confidence",
    "start_frame": "First frame (opens the inspector)",
    "end_frame": "Last frame (opens the inspector)",
    "n_frames": "Frames with a detection", "frame_span": "Last - first + 1",
    "start_time": "Capture time of the first frame",
    "end_time": "Capture time of the last frame",
    "duration_s": "Seconds between first and last frame",
    "start_x": "First ground position, project CRS",
    "start_y": "First ground position, project CRS",
    "start_z": "Ground height at the first position",
    "end_x": "Last ground position, project CRS",
    "end_y": "Last ground position, project CRS",
    "end_z": "Ground height at the last position",
    "crs": "Coordinate system of the x/y columns",
    "start_lat": "First position, WGS84", "start_lon": "First position, WGS84",
    "end_lat": "Last position, WGS84", "end_lon": "Last position, WGS84",
    "path_length_m": "Length of the ground path between positions",
    "displacement_m": "Straight-line distance first to last position",
    "mean_box_width_px": "Mean box width in pixels",
    "mean_box_height_px": "Mean box height in pixels",
    "matched_track": "Track of the same animal on the other camera",
    "matched_camera": "The other camera",
    "label_track_ids": "Labelling-tool tracks made from this one",
    "label_species": "Species set in the labelling tool",
    "label_sex": "Sex set in the labelling tool",
    "label_age": "Age set in the labelling tool",
    "perpendicular_distance_m": "Distance to the flight line",
    "attributes": "Any other track attributes, as JSON",
}


def report_columns(selected: Optional[Sequence[str]] = None) -> List[str]:
    """The inventory's columns, the most-asked ones first.

    *selected* narrows them to a user's choice; the verdict and the track id
    are kept regardless, and unknown names are ignored.
    """
    rest = [c for c in track_inventory.COLUMNS if c not in _LEAD]
    ordered = [c for c in _LEAD if c in track_inventory.COLUMNS] + rest
    if not selected:
        return ordered
    wanted = set(selected) | set(ALWAYS_SHOWN)
    return [c for c in ordered if c in wanted]


class _NumericItem(QTableWidgetItem):
    """A cell that sorts by its number when it holds one."""

    def __init__(self, value):
        super().__init__(track_inventory.cell_text(value))
        self._sort_key = value if isinstance(value, (int, float)) \
            and not isinstance(value, bool) else None
        self.setFlags(self.flags() & ~Qt.ItemFlag.ItemIsEditable)
        if self._sort_key is not None:
            self.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def __lt__(self, other):
        mine = self._sort_key
        theirs = getattr(other, "_sort_key", None)
        if mine is not None and theirs is not None:
            return mine < theirs
        if mine is None and theirs is None:
            return self.text() < other.text()
        # Numbers sort before empty cells.
        return theirs is None


class _CheckItem(QTableWidgetItem):
    """The Approved cell: a checkbox that sorts checked-first."""

    def __init__(self, checked: bool):
        super().__init__("")
        self.setFlags((self.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                      & ~Qt.ItemFlag.ItemIsEditable)
        self.setCheckState(Qt.CheckState.Checked if checked
                           else Qt.CheckState.Unchecked)
        self.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

    def __lt__(self, other):
        return (self.checkState() == Qt.CheckState.Checked) > \
            (other.checkState() == Qt.CheckState.Checked)


class BambiInventoryColumnsDialog(QDialog):
    """Choose which columns the report shows."""

    def __init__(self, selected: Optional[Sequence[str]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Track Inventory - columns")
        self.setMinimumSize(420, 520)
        layout = QVBoxLayout(self)
        info = QLabel(
            "Tick the columns the report shows for each animal. The "
            "Approved checkmark and the track id are always shown; the "
            "written CSV and JSON keep every column regardless.")
        info.setWordWrap(True)
        info.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(info)

        self.list = QListWidget()
        chosen = set(selected) if selected else set(track_inventory.COLUMNS)
        for column in report_columns():
            item = QListWidgetItem(column)
            item.setToolTip(COLUMN_HINTS.get(column, ""))
            if column in ALWAYS_SHOWN:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable
                              & ~Qt.ItemFlag.ItemIsEnabled)
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if column in chosen
                                   else Qt.CheckState.Unchecked)
            self.list.addItem(item)
        layout.addWidget(self.list, 1)

        row = QHBoxLayout()
        all_btn = QPushButton("All")
        all_btn.clicked.connect(lambda: self._set_all(True))
        none_btn = QPushButton("None")
        none_btn.clicked.connect(lambda: self._set_all(False))
        row.addWidget(all_btn)
        row.addWidget(none_btn)
        row.addStretch()
        layout.addLayout(row)

        standard = QDialogButtonBox.StandardButton
        box = QDialogButtonBox(standard.Ok | standard.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

    def _set_all(self, on: bool):
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.text() in ALWAYS_SHOWN:
                continue
            item.setCheckState(Qt.CheckState.Checked if on
                               else Qt.CheckState.Unchecked)

    def selected_columns(self) -> List[str]:
        """The ticked columns, in report order. Every column when all are."""
        chosen = [self.list.item(i).text() for i in range(self.list.count())
                  if self.list.item(i).checkState() == Qt.CheckState.Checked]
        return chosen


class BambiTrackInventoryDialog(QDialog):
    """Show an inventory as a filterable, sortable table with a verdict."""

    #: Columns that open the inspector at that frame when clicked.
    LINK_COLUMNS = ("start_frame", "end_frame")

    def __init__(self, rows: List[dict], camera_label: str = "",
                 csv_path: Optional[str] = None,
                 open_track: Optional[Callable] = None,
                 set_approved: Optional[Callable] = None,
                 columns: Optional[Sequence[str]] = None,
                 on_columns_changed: Optional[Callable] = None,
                 parent=None):
        """
        :param open_track: ``callable(track_id, frame) -> bool`` that shows
            the animal in the Detection/Track inspector at *frame*; a click
            on a start or end frame cell, or a double-click on the row,
            calls it. ``None`` leaves the cells plain.
        :param set_approved: ``callable(track_id, bool)`` that records the
            verdict; ``None`` makes the checkmarks read-only.
        :param columns: the columns to show (``None`` = all).
        :param on_columns_changed: ``callable(list)`` told the new choice
            when the user changes it here, so it can be saved.
        """
        super().__init__(parent)
        self.rows = list(rows)
        self.csv_path = csv_path
        self._open_track = open_track
        self._set_approved = set_approved
        self._on_columns_changed = on_columns_changed
        self._selected = list(columns) if columns else None
        self.columns: List[str] = []
        self._filling = False
        self.setWindowTitle(
            f"Track Inventory{' - ' + camera_label if camera_label else ''}")
        self.setMinimumSize(960, 560)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)

        layout = QVBoxLayout(self)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText(
            "Show only rows containing this text (any column)")
        self.filter_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self.filter_edit, 1)
        self.count_label = QLabel("")
        filter_row.addWidget(self.count_label)
        columns_btn = QPushButton("Columns…")
        columns_btn.setToolTip("Choose which columns the report shows.")
        columns_btn.clicked.connect(self.choose_columns)
        filter_row.addWidget(columns_btn)
        layout.addLayout(filter_row)

        self.table = QTableWidget(0, 0)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.cellClicked.connect(self._on_cell_clicked)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        copy_btn = QPushButton("Copy table")
        copy_btn.setToolTip("Copy the visible rows and columns to the "
                            "clipboard as tab-separated text - pastes into a "
                            "spreadsheet.")
        copy_btn.clicked.connect(self.copy_table)
        buttons.addWidget(copy_btn)
        save_btn = QPushButton("Save CSV as…")
        save_btn.setToolTip("Save the visible rows and columns.")
        save_btn.clicked.connect(self.save_csv)
        buttons.addWidget(save_btn)
        if csv_path:
            path_label = QLabel(f"Written to: {csv_path}")
            path_label.setStyleSheet("color: gray; font-size: 10px;")
            path_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            buttons.addWidget(path_label, 1)
        else:
            buttons.addStretch()
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        box.accepted.connect(self.accept)
        buttons.addWidget(box)
        layout.addLayout(buttons)

        self._fill()

    # -- contents --------------------------------------------------------

    def _fill(self):
        """(Re)build the table for the current column choice."""
        header = self.table.horizontalHeader()
        sorted_by = None
        if self.columns and self.table.isSortingEnabled():
            section = header.sortIndicatorSection()
            if 0 <= section < len(self.columns):
                sorted_by = (self.columns[section], header.sortIndicatorOrder())
        self._filling = True
        try:
            self.table.setSortingEnabled(False)
            self.columns = report_columns(self._selected)
            self.table.clear()
            self.table.setRowCount(len(self.rows))
            self.table.setColumnCount(len(self.columns))
            self.table.setHorizontalHeaderLabels(
                ["Approved" if c == "approved" else c for c in self.columns])
            for c, column in enumerate(self.columns):
                self.table.horizontalHeaderItem(c).setToolTip(
                    COLUMN_HINTS.get(column, ""))
            link_font = QFont()
            link_font.setUnderline(True)
            for r, row in enumerate(self.rows):
                for c, column in enumerate(self.columns):
                    if column == "approved":
                        item = _CheckItem(bool(row.get("approved")))
                        if self._set_approved is None:
                            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                        item.setToolTip(COLUMN_HINTS["approved"])
                    else:
                        item = _NumericItem(row.get(column))
                        if self._open_track is not None \
                                and column in self.LINK_COLUMNS \
                                and row.get(column) is not None:
                            item.setFont(link_font)
                            item.setForeground(QBrush(QColor(30, 100, 200)))
                            item.setToolTip(
                                f"Open track {row.get('track_id')} in the "
                                f"inspector at frame {row.get(column)}")
                    self.table.setItem(r, c, item)
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            header.setStretchLastSection(True)
            self.table.resizeColumnsToContents()
            for c in range(len(self.columns)):
                self.table.setColumnWidth(c, min(self.table.columnWidth(c), 220))
            self.table.setSortingEnabled(True)
            # A rebuild (columns changed, a row updated from the inspector)
            # keeps the order the user sorted into.
            if sorted_by is not None and sorted_by[0] in self.columns:
                self.table.sortItems(self.columns.index(sorted_by[0]),
                                     sorted_by[1])
        finally:
            self._filling = False
        self.summary.setText("\n".join(track_inventory.summarise(self.rows)))
        self._apply_filter(self.filter_edit.text())

    # -- kept in step with the inspector ---------------------------------

    def _row_index(self, track_id: int) -> Optional[int]:
        for i, row in enumerate(self.rows):
            if row.get("track_id") == track_id:
                return i
        return None

    def remove_track(self, track_id: int) -> bool:
        """Drop a track the reviewer deleted in the inspector."""
        index = self._row_index(int(track_id))
        if index is None:
            return False
        del self.rows[index]
        self._fill()
        return True

    def update_track(self, row: dict) -> None:
        """Replace a track's row with a freshly built one (a box was deleted,
        so its counts, times and positions changed); a new track is appended."""
        index = self._row_index(int(row.get("track_id")))
        if index is None:
            self.rows.append(dict(row))
        else:
            self.rows[index] = dict(row)
        self._fill()

    def set_track_approved(self, track_id: int, approved: bool) -> None:
        """Show a verdict recorded elsewhere, without recording it again."""
        index = self._row_index(int(track_id))
        if index is None:
            return
        self.rows[index]["approved"] = bool(approved)
        if "approved" not in self.columns:
            return
        column = self.columns.index("approved")
        self._filling = True
        try:
            for r in range(self.table.rowCount()):
                if self._track_at(r)[0] == int(track_id):
                    item = self.table.item(r, column)
                    if isinstance(item, _CheckItem):
                        item.setCheckState(Qt.CheckState.Checked if approved
                                           else Qt.CheckState.Unchecked)
        finally:
            self._filling = False
        self.summary.setText("\n".join(track_inventory.summarise(self.rows)))

    def choose_columns(self):
        dialog = BambiInventoryColumnsDialog(self._selected, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = dialog.selected_columns()
        self._selected = None if set(chosen) >= set(track_inventory.COLUMNS) \
            else chosen
        self._fill()
        if self._on_columns_changed is not None:
            self._on_columns_changed(list(self._selected or []))

    # -- behaviour -------------------------------------------------------

    def _track_at(self, row: int):
        """``(track_id, start_frame, end_frame)`` of a table row, by its
        cells - the table may be sorted, so the row index says nothing."""
        def _int(column):
            if column not in self.columns:
                return None
            item = self.table.item(row, self.columns.index(column))
            try:
                return int(item.text()) if item and item.text() else None
            except ValueError:
                return None
        return _int("track_id"), _int("start_frame"), _int("end_frame")

    def _show(self, track_id, frame):
        if track_id is None or self._open_track is None:
            return
        try:
            shown = self._open_track(track_id, frame)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.warning(self, "Track Inventory",
                                f"Could not open track {track_id}: {exc}")
            return
        if shown is False:
            QMessageBox.information(
                self, "Track Inventory",
                f"The store holds no frames for track {track_id}.")

    def _on_cell_clicked(self, row: int, column: int):
        name = self.columns[column]
        if name not in self.LINK_COLUMNS:
            return
        track_id, start, end = self._track_at(row)
        self._show(track_id, start if name == "start_frame" else end)

    def _on_cell_double_clicked(self, row: int, column: int):
        if self.columns[column] == "approved":
            return
        track_id, start, _end = self._track_at(row)
        self._show(track_id, start)

    def _on_item_changed(self, item: QTableWidgetItem):
        """A checkmark toggled: record the verdict and keep the rows in step."""
        if self._filling or not isinstance(item, _CheckItem):
            return
        track_id, _start, _end = self._track_at(item.row())
        if track_id is None:
            return
        approved = item.checkState() == Qt.CheckState.Checked
        for row in self.rows:
            if row.get("track_id") == track_id:
                row["approved"] = approved
        if self._set_approved is not None:
            try:
                self._set_approved(track_id, approved)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                QMessageBox.warning(
                    self, "Track Inventory",
                    f"Could not record the verdict for track {track_id}: {exc}")
        self.summary.setText("\n".join(track_inventory.summarise(self.rows)))

    def _apply_filter(self, text: str):
        wanted = (text or "").strip().lower()
        shown = 0
        for r in range(self.table.rowCount()):
            visible = not wanted or any(
                wanted in (self.table.item(r, c).text().lower())
                for c in range(self.table.columnCount())
                if self.table.item(r, c) is not None)
            self.table.setRowHidden(r, not visible)
            shown += int(visible)
        self.count_label.setText(f"{shown} of {self.table.rowCount()} shown")

    def _cell_text(self, r: int, c: int) -> str:
        item = self.table.item(r, c)
        if item is None:
            return ""
        if isinstance(item, _CheckItem):
            return "yes" if item.checkState() == Qt.CheckState.Checked else "no"
        return item.text()

    def visible_rows(self) -> List[List[str]]:
        """The table as it is shown: filtered, in the current sort order."""
        return [[self._cell_text(r, c) for c in range(self.table.columnCount())]
                for r in range(self.table.rowCount())
                if not self.table.isRowHidden(r)]

    def copy_table(self):
        lines = ["\t".join(self.columns)]
        lines += ["\t".join(cells) for cells in self.visible_rows()]
        QApplication.clipboard().setText("\n".join(lines))
        self.count_label.setText(
            f"{len(lines) - 1} row(s) copied to the clipboard")

    def save_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save track inventory", self.csv_path or "track_inventory.csv",
            "CSV files (*.csv)")
        if not path:
            return
        try:
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerow(self.columns)
            writer.writerows(self.visible_rows())
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(buffer.getvalue())
        except OSError as exc:
            QMessageBox.warning(self, "Track Inventory",
                                f"Could not write {path}: {exc}")
            return
        self.count_label.setText(f"Saved to {path}")
