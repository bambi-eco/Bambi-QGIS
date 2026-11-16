# -*- coding: utf-8 -*-
"""Download Example Flights: the flights of the BAMBI dataset, to tick.

A window over the list of flights that ships with the plugin
(:mod:`core.bambi_dataset`): every flight by its number, with the day it
was flown, the animals that were labelled on it and how large its
recordings are. The flights that are ticked are fetched into a folder; two
further ticks say what else is taken for them out of the labelled dataset -
the route and the labels - and a third whether the camera correction that
was published for a flight is used.

This module only asks; :mod:`bambi_dataset_downloader` fetches.
"""

import os
import shutil
from typing import Iterable, List, Optional, Sequence

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from .core import bambi_dataset

_SORT = Qt.ItemDataRole.UserRole
_FLIGHT, _FLOWN, _LABELLED, _SIZE = range(4)
_HEADERS = ("Flight", "Flown", "Labelled animals", "Recordings")


def size_text(size: int) -> str:
    """``122 MB``, ``4.7 GB``, ``2.15 TB`` - a number of bytes to read."""
    if size >= 1e12:
        return f"{size / 1e12:.2f} TB"
    if size >= 1e9:
        return f"{size / 1e9:.1f} GB"
    return f"{size / 1e6:.0f} MB"


def labelled_text(flight) -> str:
    """What was labelled on a flight: ``2 roe deer``, ``29 wild boar, 2
    unknown`` - ``none`` for a flight without an animal, and ``not
    labelled`` for one the labelled dataset does not have."""
    if flight.labelled is None:
        return "not labelled"
    if not flight.species:
        return "none"
    return ", ".join(f"{count} {bambi_dataset.common_name(name)}"
                     for name, count in flight.species)


def free_bytes(folder: str) -> Optional[int]:
    """The free bytes where *folder* is, or would be made; ``None`` when
    nobody can say."""
    place = os.path.abspath(folder) if folder else ""
    while place and not os.path.isdir(place):
        parent = os.path.dirname(place)
        if parent == place:
            return None
        place = parent
    if not place:
        return None
    try:
        return shutil.disk_usage(place).free
    except OSError:
        return None


class _Sorted(QTableWidgetItem):
    """A cell that is put in order by a value of its own, not by its text:
    flight 9 before flight 10, 900 MB before 1.2 GB."""

    def __init__(self, text: str, key):
        super().__init__(text)
        self.setData(_SORT, key)

    def __lt__(self, other) -> bool:
        mine, theirs = self.data(_SORT), other.data(_SORT)
        try:
            return mine < theirs
        except TypeError:
            return str(mine) < str(theirs)


class DatasetFlightsDialog(QDialog):
    """Which flights of the BAMBI dataset to fetch, where to, and what
    else to take for them."""

    def __init__(self, flights: Sequence, parent=None, folder: str = "",
                 ticked: Iterable[int] = (bambi_dataset.EXAMPLE,),
                 route: bool = True, labels: bool = False,
                 correction: bool = True):
        super().__init__(parent)
        self.setWindowTitle("Download Example Flights")
        self.resize(660, 720)
        self._flights = {flight.id: flight for flight in flights}
        #: The flights that can be ticked: those with videos to fetch.
        self._fetchable = sum(1 for flight in flights
                              if flight.raw is not None and flight.videos)
        self._filling = False

        layout = QVBoxLayout(self)
        intro = QLabel(
            "The BAMBI project publishes its flights on Zenodo (CC BY 4.0): "
            "thermal and RGB video as it came off the drone, with SRT "
            "files, flight log and camera calibrations - and for most "
            "flights the animals that were labelled by hand. Tick the "
            "flights to fetch; each gets its elevation model and is added "
            "to the project, ready for 'Extract Frames'.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        search = QHBoxLayout()
        search.addWidget(QLabel("Show:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText(
            "all flights - or type: roe deer, 2024, mavic, a number")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)
        search.addWidget(self.filter_edit)
        layout.addLayout(search)

        self.table = QTableWidget(0, len(_HEADERS))
        self.table.setHorizontalHeaderLabels(list(_HEADERS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(
            _LABELLED, QHeaderView.ResizeMode.Stretch)
        for column in (_FLIGHT, _FLOWN, _SIZE):
            header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents)
        self._fill(flights, set(int(number) for number in ticked))
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table, 1)

        choose = QHBoxLayout()
        self.select_all_btn = QPushButton("Select all")
        self.select_all_btn.setToolTip(
            "Tick every flight that is shown - all of them, or those the "
            "line above leaves.")
        self.select_all_btn.clicked.connect(self.select_all)
        choose.addWidget(self.select_all_btn)
        self.select_none_btn = QPushButton("Select none")
        self.select_none_btn.setToolTip("Untick every flight.")
        self.select_none_btn.clicked.connect(self.select_none)
        choose.addWidget(self.select_none_btn)
        self.summary_label = QLabel("")
        choose.addWidget(self.summary_label, 1)
        layout.addLayout(choose)

        extras = QGroupBox("Take from the labelled dataset as well")
        extras_layout = QVBoxLayout(extras)
        self.route_check = QCheckBox("Import flight route")
        self.route_check.setChecked(bool(route))
        self.route_check.setToolTip(
            "Fetch a flight's published poses file and make it the "
            "flight's own: where the camera was for every frame of the "
            "published video. The route is on the map at once, before a "
            "frame is extracted. 'Extract Frames' later replaces the poses "
            "with those of the frames it writes - the same places.")
        extras_layout.addWidget(self.route_check)
        self.labels_check = QCheckBox("Import labels")
        self.labels_check.setChecked(bool(labels))
        self.labels_check.setToolTip(
            "Fetch the animals that were labelled by hand - a box on "
            "every key frame, with species, sex and age - put them on the "
            "flight's frames as label tracks, and fill in the boxes on the "
            "frames between the key frames. They are in the labelling tool "
            "and among the flight's detections, as labels drawn there and "
            "added to the detections are. Without the route they wait "
            "until the frames are extracted.")
        extras_layout.addWidget(self.labels_check)
        self.correction_check = QCheckBox("Use the published correction")
        self.correction_check.setChecked(bool(correction))
        self.correction_check.setToolTip(
            "Set the correction of the camera's position and angle that "
            "was published for a flight as the flight's correction "
            "(Configuration > Correction). Without it the positions and "
            "heights of the flight log are used as they are, which can put "
            "detections metres off.")
        extras_layout.addWidget(self.correction_check)
        layout.addWidget(extras)

        where = QHBoxLayout()
        where.addWidget(QLabel("Into:"))
        self.folder_edit = QLineEdit(folder)
        self.folder_edit.setPlaceholderText(
            "the folder the flights are put in, each in one of its own")
        self.folder_edit.textChanged.connect(self._refresh_summary)
        where.addWidget(self.folder_edit)
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse)
        where.addWidget(self.browse_btn)
        layout.addLayout(where)

        self.buttons = QDialogButtonBox()
        self.download_btn = self.buttons.addButton(
            QDialogButtonBox.StandardButton.Ok)
        self.download_btn.setText("Download")
        self.buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.table.sortItems(_FLIGHT, Qt.SortOrder.AscendingOrder)
        self.table.setSortingEnabled(True)
        self._refresh_summary()

    def showEvent(self, event):
        """The flight that is ticked is in view when the list opens."""
        super().showEvent(event)
        self._show_first_ticked()

    # -- the list -------------------------------------------------------------

    def _fill(self, flights: Sequence, ticked: set) -> None:
        self._filling = True
        self.table.setRowCount(len(flights))
        for row, flight in enumerate(flights):
            number = _Sorted(str(flight.id), flight.id)
            size = flight.raw.size if flight.raw is not None else 0
            fetchable = bool(size) and flight.videos
            if fetchable:
                number.setFlags(
                    number.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                number.setCheckState(
                    Qt.CheckState.Checked if flight.id in ticked
                    else Qt.CheckState.Unchecked)
            else:
                # a cell of a table can be ticked unless it is told not to
                number.setFlags(
                    number.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            cells = (
                number,
                _Sorted(flight.date or "–", flight.date or ""),
                _Sorted(labelled_text(flight), (-flight.animals,
                                                labelled_text(flight))),
                _Sorted(size_text(size) if fetchable else "no video", size),
            )
            tip = self._tooltip(flight)
            for column, cell in enumerate(cells):
                cell.setToolTip(tip)
                if not fetchable:
                    # there, and not to be had: no box, and greyed
                    cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(row, column, cell)
            cells[_SIZE].setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._filling = False

    @staticmethod
    def _tooltip(flight) -> str:
        lines = [f"BAMBI flight {flight.id}"]
        if flight.date:
            drone = f" with a {flight.drone}" if flight.drone else ""
            lines.append(f"Flown on {flight.date}{drone}")
        if not flight.videos:
            lines.append("Published without its videos - the flight log and "
                         "the calibrations only: it cannot be fetched.")
        if flight.labelled is None:
            lines.append("Not in the labelled dataset: recordings only - "
                         "no route, no labels, no correction.")
        elif flight.animals:
            lines.append(f"{flight.animals} labelled animal(s), "
                         f"{flight.boxes} boxes on key frames "
                         f"({flight.split or 'no'} split)")
        else:
            lines.append("In the labelled dataset, without an animal on it.")
        for archive, what in ((flight.raw, "Recordings"),
                              (flight.labelled, "Labelled")):
            if archive is not None:
                lines.append(f"{what}: https://doi.org/{archive.doi}")
        return "\n".join(lines)

    def _number_item(self, row: int) -> QTableWidgetItem:
        return self.table.item(row, _FLIGHT)

    def _row_flight(self, row: int):
        return self._flights[int(self._number_item(row).data(_SORT))]

    def _apply_filter(self, text: str) -> None:
        """Show the flights that have every word of *text* somewhere - in
        their number, their day, what was labelled, the drone, the split."""
        words = [word for word in (text or "").lower().split() if word]
        for row in range(self.table.rowCount()):
            flight = self._row_flight(row)
            about = " ".join((
                str(flight.id), flight.date, labelled_text(flight),
                flight.drone, flight.split,
                " ".join(name for name, _count in flight.species))).lower()
            self.table.setRowHidden(
                row, not all(word in about for word in words))
        self._refresh_summary()

    def _set_all(self, state, shown_only: bool) -> None:
        self._filling = True
        for row in range(self.table.rowCount()):
            if shown_only and self.table.isRowHidden(row):
                continue
            item = self._number_item(row)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(state)
        self._filling = False
        self._refresh_summary()

    def select_all(self) -> None:
        """Tick every flight that is shown."""
        self._set_all(Qt.CheckState.Checked, shown_only=True)

    def select_none(self) -> None:
        """Untick every flight, shown or not."""
        self._set_all(Qt.CheckState.Unchecked, shown_only=False)

    def _on_item_changed(self, item) -> None:
        if not self._filling and item.column() == _FLIGHT:
            self._refresh_summary()

    def _show_first_ticked(self) -> None:
        for row in range(self.table.rowCount()):
            if self._number_item(row).checkState() == Qt.CheckState.Checked:
                self.table.scrollToItem(
                    self._number_item(row),
                    QAbstractItemView.ScrollHint.PositionAtCenter)
                return

    # -- what was chosen ----------------------------------------------------

    def selected_ids(self) -> List[int]:
        """The numbers of the flights that are ticked, in order."""
        items = [self._number_item(row)
                 for row in range(self.table.rowCount())]
        return sorted(int(item.data(_SORT)) for item in items
                      if item.checkState() == Qt.CheckState.Checked)

    def selected_size(self) -> int:
        """The bytes of the recordings that are ticked."""
        return sum(self._flights[number].raw.size
                   for number in self.selected_ids()
                   if self._flights[number].raw is not None)

    def folder(self) -> str:
        return self.folder_edit.text().strip()

    def import_route(self) -> bool:
        return self.route_check.isChecked()

    def import_labels(self) -> bool:
        return self.labels_check.isChecked()

    def use_correction(self) -> bool:
        return self.correction_check.isChecked()

    def _refresh_summary(self, *_args) -> None:
        if self._filling:
            return
        ticked = self.selected_ids()
        size = self.selected_size()
        shown = sum(1 for row in range(self.table.rowCount())
                    if not self.table.isRowHidden(row))
        text = (f"{len(ticked)} of {self._fetchable} flight(s) ticked: "
                f"{size_text(size)}")
        if shown != self.table.rowCount():
            text += f" - {shown} shown"
        tight = False
        free = free_bytes(self.folder())
        if free is not None and ticked:
            text += f" ({size_text(free)} free)"
            # an archive and what is unpacked from it are there together
            largest = max(self._flights[number].raw.size
                          for number in ticked
                          if self._flights[number].raw is not None)
            tight = free < size + largest
        self.summary_label.setText(text)
        self.summary_label.setStyleSheet("color: #b00020;" if tight else "")
        self.summary_label.setToolTip(
            "The flights that are ticked do not all fit into the folder: "
            "they are fetched one after the other, and those that no "
            "longer fit are left out." if tight else "")
        self.download_btn.setEnabled(bool(ticked) and bool(self.folder()))

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Where to put the flights",
            self.folder() or os.path.expanduser("~"))
        if chosen:
            self.folder_edit.setText(chosen)
