# -*- coding: utf-8 -*-
"""Detector class mapping — raw model classes onto the project's species.

A detector reports whatever classes its weights were trained on: `0`, `1`, `2`,
or names of its own. The project has its own species list, and the two are not
the same thing — which is why detections store the **raw class** alongside the
resolved species (EXCHANGE_FORMAT_PLAN.md §3.1).

This dialog is where the two are connected. It belongs with the model, not with
the project vocabulary: the mapping is a property of the weights being used, so
a different model needs a different mapping even though the species are the
same.

Anything unmapped resolves to `animal`, so a species-agnostic detector needs no
mapping at all and nothing has to be configured before a first run.

Applying a change **re-resolves the detections already stored** from the raw
class they kept, rather than requiring the detector to run again.
"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
    QHeaderView, QLabel, QMessageBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
)

from .core.schema_editor import SchemaEditor, SchemaError


class BambiClassMappingDialog(QDialog):
    """Map one producer's raw classes onto project species."""

    def __init__(self, target_folder: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detector Class Mapping")
        self.setMinimumSize(520, 380)
        self.editor = SchemaEditor(target_folder)
        self._species = [s for s in self.editor.species()]

        layout = QVBoxLayout(self)
        info = QLabel(
            "Map the classes your detector reports onto this project's "
            "species. Anything left unmapped counts as 'animal', so a "
            "single-class detector needs nothing here.\n\n"
            "Changing a mapping re-reads the detections already stored — the "
            "detector does not have to run again."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(info)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Detector:"))
        self.source_combo = QComboBox()
        self.source_combo.currentIndexChanged.connect(self._reload)
        source_row.addWidget(self.source_combo, 1)
        layout.addLayout(source_row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            ["Detector class", "Detections", "Species"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status)

        standard = QDialogButtonBox.StandardButton
        buttons = QDialogButtonBox(standard.Ok | standard.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_sources()

    # -- lifecycle -------------------------------------------------------

    def accept(self):
        """Write the mappings, then re-resolve the stored detections."""
        source_id = self._current_source_id()
        if source_id is None:
            self.editor.close()
            super().accept()
            return

        try:
            for row in range(self.table.rowCount()):
                combo = self.table.cellWidget(row, 2)
                raw = self.table.item(row, 0).text()
                self.editor.set_class_mapping(
                    source_id, raw, combo.currentData())
            self.editor.commit()
            changed = self.editor.reapply_class_mapping(source_id)
        except SchemaError as exc:
            QMessageBox.warning(self, "Class Mapping", str(exc))
            return

        self.editor.close()
        self._changed = changed
        super().accept()

    def reject(self):
        self.editor.close()
        super().reject()

    @property
    def rows_changed(self) -> int:
        """Detections whose species changed when the mapping was applied."""
        return getattr(self, "_changed", 0)

    # -- contents --------------------------------------------------------

    def _load_sources(self):
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        for source in self.editor.sources():
            label = source["kind"]
            if source["model"]:
                label += f" — {source['model']}"
            self.source_combo.addItem(label, source["source_id"])
        self.source_combo.blockSignals(False)
        self._reload()

    def _current_source_id(self):
        index = self.source_combo.currentIndex()
        return None if index < 0 else self.source_combo.itemData(index)

    def _reload(self):
        self.table.setRowCount(0)
        source_id = self._current_source_id()
        if source_id is None:
            self.status.setText(
                "No detector has run in this project yet, so there are no "
                "classes to map. Run detection first.")
            return

        observed = self.editor.observed_source_classes(source_id)
        self.table.setRowCount(len(observed))
        for row, entry in enumerate(observed):
            name = QTableWidgetItem(entry["source_class"])
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, name)

            count = QTableWidgetItem(str(entry["detections"]))
            count.setFlags(count.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, count)

            combo = QComboBox()
            for species in self._species:
                combo.addItem(species["name"], species["species_id"])
            index = combo.findData(entry["species_id"])
            combo.setCurrentIndex(max(0, index))
            self.table.setCellWidget(row, 2, combo)

        unmapped = sum(1 for entry in observed if not entry["mapped"])
        if not observed:
            self.status.setText(
                "This detector has produced no detections yet.")
        elif unmapped:
            self.status.setText(
                f"{unmapped} of {len(observed)} class(es) are unmapped and "
                "currently count as 'animal'.")
        else:
            self.status.setText(f"{len(observed)} class(es) mapped.")
