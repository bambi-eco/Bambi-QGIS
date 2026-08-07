# -*- coding: utf-8 -*-
"""Choose a classifier per species, for the demographic tasks.

Sex and life stage are not one problem across animals. The published sex
classifier reads antlers on red deer; nothing about it transfers to a wild
boar, and a model applied to a species it was never fitted for would return
confident nonsense rather than nothing. So the choice is made per species, and
a species with no classifier is simply left uncalled — for life stage the
box-area estimate then fills the gap, if it is switched on.

That is also why the sex table has exactly one entry switched on: red deer is
the only species with a released model today.
"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from .core import hf_access, label_store

#: Species the published sex classifier was fitted on.
DEFAULT_SPECIES = "red deer"

_SOURCES = (("Off", "off"), ("Default", "default"), ("Custom…", "custom"))

#: Life stage's default is a measurement rather than a model, and takes the
#: "Default" slot because that is what it is: no life-stage head has been
#: published, size needs none, and it is what runs unless you say otherwise.
#: It sits in the same column as the models because it answers the same
#: question — "what decides this?" is one decision, not a model choice plus a
#: separate switch somewhere else.
SIZE_SOURCE = ("Default (size-based)", "size")


def sources_for(task: str):
    """Model options offered for *task*."""
    if task == "life_stage":
        # No published head to offer, so the plain "Default" would only fail
        # later; the size estimate is the default that actually runs.
        off, _default, *rest = _SOURCES
        return (off, SIZE_SOURCE) + tuple(rest)
    return _SOURCES


def default_source_for(task: str) -> str:
    """The option *task* starts on, and falls back to after a bad load."""
    from .core import hf_access

    if hf_access.has_default_head(task):
        return "default"
    return "size" if task == "life_stage" else "off"


class BambiClassificationModelDialog(QDialog):
    """Per-species classifier selection for one demographic task."""

    def __init__(self, spec: dict, target_folder: str, task: str = "sex",
                 parent=None):
        super().__init__(parent)
        self.spec = dict(spec or {})
        self.target_folder = target_folder
        self.task = task
        self._label = hf_access.TASK_LABELS.get(task, task).lower()

        self.setWindowTitle(f"{self._label.capitalize()} classifiers per species")
        self.setMinimumSize(560, 380)

        layout = QVBoxLayout(self)

        info = QLabel(self._describe())
        info.setWordWrap(True)
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(info)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Species", "Model", "File"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.table)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status)

        standard = QDialogButtonBox.StandardButton
        box = QDialogButtonBox(standard.Ok | standard.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

        self._populate()

    # -- contents --------------------------------------------------------

    def _describe(self) -> str:
        """What this task reads, and what happens to a species left off."""
        if self.task == "life_stage":
            return (
                "Choose what decides life stage for each species.\n\n"
                "<b>Default (size-based)</b> needs no model: a juvenile sits "
                "far below its cohort, measured within this flight. It is the "
                "default because no life-stage classifier has been published "
                "yet, and it is what step C5 runs. A species left <b>Off</b> "
                "is not called at all."
            )
        return (
            "Choose which classifier decides sex for each species.\n\n"
            "Sex is read from a species-specific cue — antlers, for red "
            "deer — so a model fitted on one species says nothing useful "
            "about another. A species left <b>Off</b> is simply not sexed, "
            "which is the honest answer rather than a guess."
        )

    def _species(self):
        """Concrete species from the project, base classes excluded.

        ``animal`` / ``unknown`` / ``not-an-animal`` are not species anyone has
        a classifier for.
        """
        vocabulary = label_store.vocabulary(self.target_folder)
        rows = vocabulary.get("species", []) if vocabulary else []
        names = [row["name"] for row in rows if not row.get("protected")]
        if not names:
            names = [DEFAULT_SPECIES]
        return names

    def _populate(self):
        per_species = self.spec.get("species") or {}
        names = self._species()
        self.table.setRowCount(len(names))

        for row, name in enumerate(names):
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, item)

            entry = per_species.get(name) or {}
            source = entry.get("model")
            if source is None:
                # Nothing saved yet: offer the published model for the one
                # species it was fitted on, and leave the rest alone.
                fitted = name == DEFAULT_SPECIES
                published = hf_access.has_default_head(self.task)
                if fitted and published:
                    source = "default"
                elif self.task == "life_stage":
                    # No life-stage model exists, and size needs none — so it
                    # is the useful default rather than "off".
                    source = "size"
                else:
                    source = "off"

            combo = QComboBox()
            for label, value in sources_for(self.task):
                combo.addItem(label, value)
            found = combo.findData(source)
            combo.setCurrentIndex(max(0, found))
            combo.currentIndexChanged.connect(
                lambda _index, r=row: self._on_source_changed(r))
            self.table.setCellWidget(row, 1, combo)

            self.table.setItem(row, 2, QTableWidgetItem(entry.get("path", "")))
            self._on_source_changed(row)

        if not hf_access.has_default_head(self.task):
            extra = ("" if self.task != "life_stage" else
                     " The size-based estimate needs none, and is what these "
                     "default to.")
            self.status.setText(
                f"No {self._label} classifier has been published yet; choose "
                f"a custom model for any species you want called by one."
                f"{extra}")
        else:
            self.status.setText(
                f"The published model covers {DEFAULT_SPECIES}. For anything "
                "else, supply a custom model or leave it Off.")

    def _on_source_changed(self, row: int):
        """Only a custom model has a file to choose."""
        combo = self.table.cellWidget(row, 1)
        source = combo.currentData() if combo else "off"
        item = self.table.item(row, 2)
        if item is None:
            return
        if source == "custom":
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            if not item.text():
                item.setText("(double-click to choose a file)")
        else:
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if source == "default":
                species = self.table.item(row, 0).text()
                item.setText(
                    hf_access.default_head_repo(self.task) or ""
                    if species == DEFAULT_SPECIES else "")
            elif source == "size":
                item.setText("measured from the box area, per flight")
            else:
                item.setText("")

    def _on_item_double_clicked(self, item):
        """Double-clicking a custom model's file cell opens a file chooser."""
        if item.column() != 2:
            return
        combo = self.table.cellWidget(item.row(), 1)
        if combo is None or combo.currentData() != "custom":
            return
        self.browse_for(item.row())

    def browse_for(self, row: int):
        """Pick a custom model file for one species."""
        path, _filter = QFileDialog.getOpenFileName(
            self, f"Choose a {self._label} classifier", "",
            "TorchScript model (*.pt *.pth);;All files (*)")
        if path:
            self.table.item(row, 2).setText(path)

    # -- result ----------------------------------------------------------

    def result_spec(self) -> dict:
        """The task's spec with its per-species selection updated."""
        per_species = {}
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text()
            combo = self.table.cellWidget(row, 1)
            source = combo.currentData() if combo else "off"
            if source == "off":
                continue      # absent means "not called", which is the default
            entry = {"model": source}
            if source == "custom":
                entry["path"] = (self.table.item(row, 2).text() or "").strip()
            per_species[name] = entry

        spec = dict(self.spec)
        spec["species"] = per_species
        return spec
