# -*- coding: utf-8 -*-
"""Download Example Flights: the flights of the BAMBI dataset, to tick.

A window over the flights the dataset's repository lists
(:mod:`core.bambi_dataset`): every flight by its number, with the split it
belongs to and what the dataset has of it - the recordings, the labelled
flight. The flights that are ticked are fetched into a folder; a tick says
whether their videos are fetched or only their annotations, and three more
what is done with those: the route, the labels, the camera correction.

This module only asks; :mod:`bambi_dataset_downloader` fetches.
"""

import os
from typing import Iterable, List, Optional, Sequence

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from .core import bambi_dataset

_SORT = Qt.ItemDataRole.UserRole
_FLIGHT, _SPLIT, _LABELLED, _RECORDINGS = range(4)
_HEADERS = ("Flight", "Split", "Labelled", "Recordings")


def _has(archive) -> str:
    return "yes" if archive is not None else "–"


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
                 correction: bool = True, videos: bool = True,
                 records: Optional[dict] = None):
        super().__init__(parent)
        self.setWindowTitle("Download Example Flights")
        self.resize(600, 720)
        self._flights = {flight.id: flight for flight in flights}
        #: ``{record: {"title", "doi"}}``, for what a row says of itself.
        self._records = dict(records or {})
        self._filling = False

        layout = QVBoxLayout(self)
        intro = QLabel(
            "The BAMBI project publishes its flights on Zenodo (CC BY 4.0): "
            "thermal and RGB video as it came off the drone, with SRT "
            "files, flight log and camera calibrations - and for most "
            "flights the poses and the animals that were labelled by hand. "
            "The list is the one the dataset's repository keeps. Tick the "
            "flights to fetch; each gets its elevation model and is added "
            "to the project.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        search = QHBoxLayout()
        search.addWidget(QLabel("Show:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText(
            "all flights - or type: a number, train, val, test")
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
        for column in range(len(_HEADERS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
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

        self.videos_check = QCheckBox("Download video data")
        self.videos_check.setChecked(bool(videos))
        self.videos_check.setToolTip(
            "Fetch a flight's recordings - thermal and RGB video, SRT "
            "files, flight log, calibrations; hundreds of megabytes to "
            "many gigabytes - which every step from 'Extract Frames' on "
            "needs. Unticked, only the poses and labels of a flight are "
            "fetched, a megabyte or two: its route and its labelled "
            "animals for a fast overview in QGIS. The videos can be "
            "fetched later by asking for the flight again.")
        self.videos_check.toggled.connect(self._refresh_summary)
        layout.addWidget(self.videos_check)

        extras = QGroupBox("From the labelled dataset")
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
            "until the frames are extracted. Sex and age are read as the "
            "dataset documents them.")
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
            number.setFlags(number.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            number.setCheckState(Qt.CheckState.Checked if flight.id in ticked
                                 else Qt.CheckState.Unchecked)
            cells = (
                number,
                _Sorted(flight.split or "–", flight.split or "~"),
                _Sorted(_has(flight.labelled), flight.labelled is None),
                _Sorted(_has(flight.raw), flight.raw is None),
            )
            tip = self._tooltip(flight)
            for column, cell in enumerate(cells):
                cell.setToolTip(tip)
                self.table.setItem(row, column, cell)
        self._filling = False

    def _tooltip(self, flight) -> str:
        lines = [f"BAMBI flight {flight.id}"]
        for archive, what in ((flight.raw, "Recordings"),
                              (flight.labelled, "Labelled flight")):
            if archive is None:
                lines.append(f"{what}: not in the dataset")
                continue
            record = self._records.get(archive.record) or {}
            title = record.get("title") or "Zenodo"
            lines.append(f"{what}: {title}, https://doi.org/"
                         f"{record.get('doi') or archive.doi}")
        return "\n".join(lines)

    def _number_item(self, row: int) -> QTableWidgetItem:
        return self.table.item(row, _FLIGHT)

    def _row_flight(self, row: int):
        return self._flights[int(self._number_item(row).data(_SORT))]

    def _apply_filter(self, text: str) -> None:
        """Show the flights that have every word of *text* in their number
        or their split."""
        words = [word for word in (text or "").lower().split() if word]
        for row in range(self.table.rowCount()):
            flight = self._row_flight(row)
            about = f"{flight.id} {flight.split}".lower()
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

    def selected_flights(self) -> list:
        """The flights that are ticked, as the list was given them."""
        return [self._flights[number] for number in self.selected_ids()]

    def folder(self) -> str:
        return self.folder_edit.text().strip()

    def import_route(self) -> bool:
        return self.route_check.isChecked()

    def import_labels(self) -> bool:
        return self.labels_check.isChecked()

    def use_correction(self) -> bool:
        return self.correction_check.isChecked()

    def download_videos(self) -> bool:
        return self.videos_check.isChecked()

    def _refresh_summary(self, *_args) -> None:
        if self._filling:
            return
        ticked = self.selected_flights()
        shown = sum(1 for row in range(self.table.rowCount())
                    if not self.table.isRowHidden(row))
        text = f"{len(ticked)} of {self.table.rowCount()} flight(s) ticked"
        if shown != self.table.rowCount():
            text += f" - {shown} shown"
        # what the dataset does not have of a ticked flight
        if self.download_videos():
            lacking = [flight.id for flight in ticked if flight.raw is None]
            what = "recordings"
        else:
            lacking = [flight.id for flight in ticked
                       if flight.labelled is None]
            what = "poses or labels"
        if lacking:
            text += f"; {len(lacking)} without {what}"
        self.summary_label.setText(text)
        self.summary_label.setStyleSheet("color: #b00020;" if lacking else "")
        numbers = ", ".join(str(number) for number in lacking[:20])
        self.summary_label.setToolTip(
            f"The dataset has no {what} of flight(s) {numbers}: they are "
            "left out." if lacking else "")
        self.download_btn.setEnabled(bool(ticked) and bool(self.folder()))

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Where to put the flights",
            self.folder() or os.path.expanduser("~"))
        if chosen:
            self.folder_edit.setText(chosen)
