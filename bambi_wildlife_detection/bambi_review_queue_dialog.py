# -*- coding: utf-8 -*-
"""The review queue: the tracks worth a human look, one after another.

A window on :mod:`core.review_queue`. It lists the tracks that are in doubt,
the most doubtful first, each with why. Selecting one opens it in the
inspector; the reviewer then says what it is - approve it as it stands, rule
it out as not an animal, or correct its species, sex or age - and goes on to
the next. What is said here is written to the track at once and stands: a
later classifier run leaves it alone (:mod:`core.track_review`).

The window knows nothing of the stores: it is handed the functions that
load the queue and write the reviewer's word.
"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QKeySequence
from qgis.PyQt.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)
try:                                    # Qt 5
    from qgis.PyQt.QtWidgets import QShortcut
except ImportError:                     # Qt 6 moved it
    from qgis.PyQt.QtGui import QShortcut

from .core import review_queue

_RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
#: What the sex and age selectors show for "nobody knows".
NOT_KNOWN = "(not known)"
NOT_AN_ANIMAL = "not-an-animal"


class ReviewQueueDialog(QDialog):
    """Go through the tracks that are in doubt.

    :param load: ``load(settings) -> (queue, total)`` - the queue of
        :func:`core.review_queue.for_flights` for limits given as
        ``{"min_confidence", "min_boxes", "min_vote"}``
    :param vocabulary: ``vocabulary(flight) -> {"species": [(id, name)],
        "sex": [(id, label)], "age": [(id, label)]}`` of a flight
    :param open_track: ``open_track(row)`` - show the track in the inspector
    :param approve: ``approve(row)`` - mark the track as approved
    :param set_species: ``set_species(row, species id)``
    :param set_attribute: ``set_attribute(row, "sex" | "age", id or None)``
    :param several: whether the survey has more than one flight
    """

    def __init__(self, load, vocabulary, open_track, approve, set_species,
                 set_attribute, camera_label: str = "", several: bool = False,
                 parent=None):
        super().__init__(parent)
        self._load = load
        self._vocabulary = vocabulary
        self._open_track = open_track
        self._approve = approve
        self._set_species = set_species
        self._set_attribute = set_attribute
        self._several = several
        self._queue = []
        self._total = 0
        self._done = 0
        self._filling = False
        self.setWindowTitle(f"Review Queue ({camera_label})" if camera_label
                            else "Review Queue")
        self.resize(980, 560)
        layout = QVBoxLayout(self)

        intro = QLabel(
            "The tracks that are in doubt, the most doubtful first. Select "
            "one to see it in the inspector, then say what it is: approve "
            "it, rule it out, or correct its species, sex or age. What you "
            "say is written at once, and no classifier run changes it.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        limits = QHBoxLayout()
        limits.addWidget(QLabel("In doubt: confidence below"))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(review_queue.MIN_CONFIDENCE)
        self.confidence_spin.setToolTip(
            "A track whose boxes have a mean detector confidence below this "
            "is in doubt.")
        limits.addWidget(self.confidence_spin)
        limits.addWidget(QLabel("fewer boxes than"))
        self.boxes_spin = QSpinBox()
        self.boxes_spin.setRange(0, 100000)
        self.boxes_spin.setValue(review_queue.MIN_BOXES)
        self.boxes_spin.setToolTip(
            "A track of fewer boxes than this is very short - which is what "
            "a false alarm looks like.")
        limits.addWidget(self.boxes_spin)
        limits.addWidget(QLabel("votes below"))
        self.vote_spin = QSpinBox()
        self.vote_spin.setRange(0, 100)
        self.vote_spin.setSuffix(" %")
        self.vote_spin.setValue(int(round(review_queue.MIN_VOTE * 100)))
        self.vote_spin.setToolTip(
            "A species or sex that a classifier called by less than this "
            "share of its votes is in doubt.")
        limits.addWidget(self.vote_spin)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip(
            "Build the queue again from the project as it is now.")
        self.refresh_btn.clicked.connect(self.refresh)
        limits.addWidget(self.refresh_btn)
        limits.addStretch()
        layout.addLayout(limits)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self._columns = (["Flight"] if several else []) + [
            "Track", "Species", "Sex", "Age", "Why", "Confidence", "Boxes"]
        self.table = QTableWidget(0, len(self._columns))
        self.table.setHorizontalHeaderLabels(self._columns)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.itemSelectionChanged.connect(self._on_row_changed)
        layout.addWidget(self.table)

        act = QHBoxLayout()
        self.approve_btn = QPushButton("✓ Approve (A)")
        self.approve_btn.setToolTip(
            "The track is right as it stands. It leaves the queue and is "
            "ticked as approved in the track inventory.")
        self.approve_btn.clicked.connect(self.approve_current)
        act.addWidget(self.approve_btn)
        self.reject_btn = QPushButton("✗ Not an animal (X)")
        self.reject_btn.setToolTip(
            "Rule the track out: it becomes 'not-an-animal', which no "
            "survey tool counts. Nothing is deleted.")
        self.reject_btn.clicked.connect(self.reject_current)
        act.addWidget(self.reject_btn)
        act.addWidget(QLabel("Species:"))
        self.species_combo = QComboBox()
        self.species_combo.activated.connect(self._on_species_chosen)
        act.addWidget(self.species_combo)
        act.addWidget(QLabel("Sex:"))
        self.sex_combo = QComboBox()
        self.sex_combo.activated.connect(
            lambda _i: self._on_attribute_chosen("sex", self.sex_combo))
        act.addWidget(self.sex_combo)
        act.addWidget(QLabel("Age:"))
        self.age_combo = QComboBox()
        self.age_combo.activated.connect(
            lambda _i: self._on_attribute_chosen("age", self.age_combo))
        act.addWidget(self.age_combo)
        self.skip_btn = QPushButton("Skip (N)")
        self.skip_btn.setToolTip("Leave this track in the queue and go on.")
        self.skip_btn.clicked.connect(self.select_next)
        act.addWidget(self.skip_btn)
        act.addStretch()
        layout.addLayout(act)

        foot = QHBoxLayout()
        self.inspector_check = QCheckBox("Show the selected track in the "
                                         "inspector")
        self.inspector_check.setChecked(True)
        foot.addWidget(self.inspector_check)
        foot.addStretch()
        self.progress_label = QLabel("")
        foot.addWidget(self.progress_label)
        layout.addLayout(foot)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        for key, slot in (("A", self.approve_current),
                          ("X", self.reject_current),
                          ("N", self.select_next)):
            QShortcut(QKeySequence(key), self, activated=slot)
        self.refresh()

    # ------------------------------------------------------------------ #
    # The queue
    # ------------------------------------------------------------------ #

    def settings(self) -> dict:
        return {"min_confidence": self.confidence_spin.value(),
                "min_boxes": self.boxes_spin.value(),
                "min_vote": self.vote_spin.value() / 100.0}

    def refresh(self) -> None:
        """Build the queue again from the project as it is now."""
        self._queue, self._total = self._load(self.settings())
        self._done = 0
        self.summary_label.setText(
            review_queue.summarise(self._queue, self._total))
        self._fill()
        if self._queue:
            self.table.selectRow(0)
        else:
            self._on_row_changed()

    def rows(self) -> list:
        """The tracks still in the queue, in its order."""
        return list(self._queue)

    def _cells(self, row: dict) -> list:
        confidence = row.get("mean_confidence")
        cells = [row.get("flight") or ""] if self._several else []
        return cells + [
            row.get("track_id"), row.get("species") or "",
            row.get("sex") or "", row.get("age") or "",
            "; ".join(row.get("reasons") or []),
            "" if confidence is None else f"{confidence:.2f}",
            row.get("n_boxes")]

    def _fill(self) -> None:
        self._filling = True
        try:
            self.table.setRowCount(len(self._queue))
            for r, row in enumerate(self._queue):
                self._fill_row(r, row)
            self.table.resizeColumnsToContents()
        finally:
            self._filling = False
        self._show_progress()

    def _fill_row(self, r: int, row: dict) -> None:
        numbers = {self._columns.index(name)
                   for name in ("Track", "Confidence", "Boxes")}
        for c, value in enumerate(self._cells(row)):
            item = QTableWidgetItem("" if value is None else str(value))
            if c in numbers:
                item.setTextAlignment(_RIGHT)
            self.table.setItem(r, c, item)

    def _show_progress(self) -> None:
        left = len(self._queue)
        self.progress_label.setText(
            f"{left} left" + (f", {self._done} done" if self._done else ""))
        for widget in (self.approve_btn, self.reject_btn, self.species_combo,
                       self.sex_combo, self.age_combo, self.skip_btn):
            widget.setEnabled(self.current() is not None)

    # ------------------------------------------------------------------ #
    # The selected track
    # ------------------------------------------------------------------ #

    def current(self):
        """The selected track's row, or ``None``."""
        index = self.table.currentRow()
        selected = self.table.selectionModel().hasSelection() \
            if self.table.selectionModel() else False
        if not selected or not 0 <= index < len(self._queue):
            return None
        return self._queue[index]

    def _on_row_changed(self) -> None:
        if self._filling:
            return
        row = self.current()
        self._show_progress()
        if row is None:
            return
        self._fill_choices(row)
        if self.inspector_check.isChecked():
            self._open_track(row)

    def _fill_choices(self, row: dict) -> None:
        """Offer what this track's flight knows, on what the track is."""
        choices = self._vocabulary(row.get("flight")) or {}
        self.species_combo.clear()
        for species_id, name in choices.get("species") or []:
            self.species_combo.addItem(name, species_id)
        self.species_combo.setCurrentIndex(
            self.species_combo.findText(row.get("species") or ""))
        for combo, field in ((self.sex_combo, "sex"), (self.age_combo, "age")):
            combo.clear()
            combo.addItem(NOT_KNOWN, None)
            for value_id, label in choices.get(field) or []:
                combo.addItem(label, value_id)
            index = combo.findText(str(row.get(field) or ""))
            combo.setCurrentIndex(index if index > 0 else 0)

    def select_next(self) -> None:
        """Go on to the next track; from the last one back to the first."""
        if not self._queue:
            return
        index = self.table.currentRow() + 1
        self.table.selectRow(index if index < len(self._queue) else 0)

    def _leave(self) -> None:
        """Take the selected track out of the queue and select what comes
        next."""
        index = self.table.currentRow()
        if not 0 <= index < len(self._queue):
            return
        del self._queue[index]
        self._done += 1
        self._filling = True
        try:
            self.table.removeRow(index)
        finally:
            self._filling = False
        if self._queue:
            self.table.selectRow(min(index, len(self._queue) - 1))
            self._on_row_changed()
        else:
            self.table.clearSelection()
            self._show_progress()

    def remove_track(self, track_id, flight=None) -> bool:
        """Take a track out of the queue that was dealt with elsewhere -
        approved or deleted in the inspector."""
        for index, row in enumerate(self._queue):
            if row.get("track_id") == track_id and \
                    (row.get("flight") or None) == (flight or None):
                selected = index == self.table.currentRow()
                if selected:
                    self._leave()
                else:
                    del self._queue[index]
                    self._done += 1
                    self._filling = True
                    try:
                        self.table.removeRow(index)
                    finally:
                        self._filling = False
                    self._show_progress()
                return True
        return False

    # ------------------------------------------------------------------ #
    # The reviewer's word
    # ------------------------------------------------------------------ #

    def approve_current(self) -> None:
        row = self.current()
        if row is None:
            return
        if self._approve(row) is False:
            return
        self._leave()

    def reject_current(self) -> None:
        row = self.current()
        if row is None:
            return
        choices = self._vocabulary(row.get("flight")) or {}
        ruled_out = next((species_id for species_id, name in
                          choices.get("species") or []
                          if name == NOT_AN_ANIMAL), None)
        if ruled_out is None or self._set_species(row, ruled_out) is False:
            return
        self._leave()

    def _on_species_chosen(self, index: int) -> None:
        row = self.current()
        if row is None or index < 0:
            return
        name = self.species_combo.itemText(index)
        if name == (row.get("species") or ""):
            return
        if self._set_species(row, self.species_combo.itemData(index)) is False:
            self._fill_choices(row)
            return
        if name == NOT_AN_ANIMAL:
            self._leave()
            return
        row["species"] = name
        self._refill_current()

    def _on_attribute_chosen(self, field: str, combo) -> None:
        row = self.current()
        if row is None:
            return
        value = combo.currentData()
        label = "" if value is None else combo.currentText()
        if label == str(row.get(field) or ""):
            return
        if self._set_attribute(row, field, value) is False:
            self._fill_choices(row)
            return
        row[field] = label
        self._refill_current()

    def _refill_current(self) -> None:
        index = self.table.currentRow()
        if 0 <= index < len(self._queue):
            self._filling = True
            try:
                self._fill_row(index, self._queue[index])
            finally:
                self._filling = False
