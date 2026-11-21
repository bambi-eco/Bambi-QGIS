# -*- coding: utf-8 -*-
"""Survey Analytics > Dataset Export: the flights of the survey as one
dataset, each flight in one split.

The group on the *Export* sub tab - the format, a table that says for
every ticked flight whether it is trained on, validated on, tested on or
left out, the shares to deal the flights by, and the button that writes
the dataset (``core.dataset_export``). The export of one flight stays on
the Processing tab.
"""

import json
import os

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout,
)

from ..core import dataset_export, exporters, flights as flights_core
from ..gui_utils import PLUGIN_SCOPE

#: Where the export's settings are kept in the QGIS project.
ENTRY_DATASET_EXPORT = "Survey/DatasetExport"
_FOLDER_ROLE = Qt.ItemDataRole.UserRole


class DatasetExportMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _build_dataset_export_group(self) -> QGroupBox:
        """The Dataset Export group of the Survey Analytics tab."""
        group = QGroupBox("Dataset Export")
        layout = QVBoxLayout(group)
        description = QLabel(
            "The ticked flights as one dataset - for training a detector, "
            "a tracker or a classifier, or for publishing. Every flight "
            "goes into one split whole: frames of one flight are seconds "
            "apart and show the same animals, so a flight on both sides of "
            "a split would only test how well a model remembers. Exported "
            "is what the flights hold for the survey's camera - with 'Only "
            "animals confirmed by the other camera' ticked in the Survey "
            "box, only what both cameras saw. The species filter and the "
            "survey area do not apply. One flight alone is exported on the "
            "Processing tab.")
        description.setWordWrap(True)
        description.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(description)

        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Format:"))
        self.dataset_format_combo = QComboBox()
        for key, (label, _fn, _folder) in exporters.EXPORTERS.items():
            if key == "geojson_segmentation":
                continue
            self.dataset_format_combo.addItem(label, key)
        self.dataset_format_combo.setToolTip(
            "COCO and YOLO are joined over the flights: the classes are "
            "named alike everywhere (species ids are a flight's own), "
            "COCO gets one annotations.json per split and YOLO one "
            "data.yaml with the image folders of every flight. The other "
            "formats are written as a folder per flight inside its split.")
        self.dataset_format_combo.currentIndexChanged.connect(
            self._on_dataset_format_changed)
        format_row.addWidget(self.dataset_format_combo, 1)
        layout.addLayout(format_row)

        # ----- which flight goes where -----
        self._dataset_splits = {}        # normalised folder -> split
        self.dataset_flights_table = QTableWidget(0, 2)
        self.dataset_flights_table.setHorizontalHeaderLabels(
            ["Flight", "Split"])
        self.dataset_flights_table.verticalHeader().setVisible(False)
        header = self.dataset_flights_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        self.dataset_flights_table.setMaximumHeight(170)
        self.dataset_flights_table.setToolTip(
            "The flights ticked in the Survey box, and the split each one "
            "goes into as a whole. 'Leave out' keeps a flight in the "
            "survey and out of the dataset.")
        layout.addWidget(self.dataset_flights_table)

        deal_row = QHBoxLayout()
        deal_row.addWidget(QLabel("Train / validation / test:"))
        self.dataset_share_spins = []
        for share in dataset_export.DEFAULT_SHARES:
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setSuffix(" %")
            spin.setValue(share)
            self.dataset_share_spins.append(spin)
            deal_row.addWidget(spin)
        deal_row.addWidget(QLabel("Seed:"))
        self.dataset_seed_spin = QSpinBox()
        self.dataset_seed_spin.setRange(0, 999999)
        self.dataset_seed_spin.setToolTip(
            "The same seed deals the same flights into the same splits "
            "every time; another seed deals them another way.")
        deal_row.addWidget(self.dataset_seed_spin)
        self.dataset_deal_btn = QPushButton("Assign")
        self.dataset_deal_btn.setToolTip(
            "Deal every listed flight into train, validation and test at "
            "random by these shares of the flights - whole flights, so "
            "with few flights the shares are met only roughly. Change "
            "single flights in the table afterwards.")
        self.dataset_deal_btn.clicked.connect(self.assign_dataset_splits)
        deal_row.addWidget(self.dataset_deal_btn)
        deal_row.addStretch()
        layout.addLayout(deal_row)

        self.dataset_splits_label = QLabel("")
        self.dataset_splits_label.setStyleSheet(
            "color: gray; font-size: 10px;")
        layout.addWidget(self.dataset_splits_label)

        self.dataset_false_positives_check = QCheckBox(
            "Include detections labelled 'not-an-animal'")
        self.dataset_false_positives_check.setToolTip(
            "Survey formats keep them as a record of what was looked at "
            "and rejected; training formats drop them. Darwin Core always "
            "excludes them.")
        layout.addWidget(self.dataset_false_positives_check)
        self.dataset_images_check = QCheckBox("Include images")
        self.dataset_images_check.setChecked(True)
        self.dataset_images_check.setToolTip(
            "Copy the frames the export refers to into the dataset. The "
            "frames are the heaviest part of a flight; without them the "
            "dataset holds the annotations only. Formats that reference "
            "no image (GeoJSON, TRex) ignore it.")
        layout.addWidget(self.dataset_images_check)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Folder:"))
        self.dataset_output_edit = QLineEdit()
        self.dataset_output_edit.setPlaceholderText(
            "Where the dataset is written (empty = 'dataset' in the "
            "survey folder)")
        output_row.addWidget(self.dataset_output_edit, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_dataset_output)
        output_row.addWidget(browse)
        layout.addLayout(output_row)

        run_row = QHBoxLayout()
        self.dataset_export_btn = QPushButton("→ Export Dataset")
        self.dataset_export_btn.setToolTip(
            "Write <folder>/train, /val and /test with a folder per "
            "flight, splits.json (which flight is in which split) and "
            "dataset.json. A folder that holds an earlier export in which "
            "a flight was in another split is refused.")
        self.dataset_export_btn.clicked.connect(self.run_dataset_export)
        self.dataset_export_status = QLabel("⚪")
        run_row.addWidget(self.dataset_export_btn)
        run_row.addWidget(self.dataset_export_status)
        run_row.addStretch()
        layout.addLayout(run_row)

        for widget in (self.dataset_seed_spin, *self.dataset_share_spins):
            widget.valueChanged.connect(
                lambda _value: self._dataset_export_changed())
        for check in (self.dataset_false_positives_check,
                      self.dataset_images_check):
            check.toggled.connect(lambda _on: self._dataset_export_changed())
        self.dataset_output_edit.editingFinished.connect(
            self._dataset_export_changed)
        self._on_dataset_format_changed()
        return group

    # ------------------------------------------------------------------ #
    # The flights and their splits
    # ------------------------------------------------------------------ #

    def _export_flights(self):
        """``[(name, normalised folder)]`` of the ticked flights."""
        if not hasattr(self, "survey_panel"):
            return []
        listed = []
        for flight in self.survey_panel.selected_flights():
            folder = flight.get("target_folder", "")
            listed.append((
                flight.get("name") or flights_core.default_name(folder),
                flights_core.normalise(folder)))
        return listed

    def _refresh_dataset_flights(self):
        """Show the ticked flights with the split each one has. A flight
        nobody has placed yet is trained on."""
        if not hasattr(self, "dataset_flights_table"):
            return
        table = self.dataset_flights_table
        flights = self._export_flights()
        table.setRowCount(len(flights))
        for row, (name, folder) in enumerate(flights):
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setData(_FOLDER_ROLE, folder)
            table.setItem(row, 0, item)
            combo = QComboBox()
            for split in (*dataset_export.SPLITS, dataset_export.LEFT_OUT):
                combo.addItem(dataset_export.SPLIT_LABELS[split], split)
            combo.setCurrentIndex(max(0, combo.findData(
                self._dataset_splits.get(folder, dataset_export.TRAIN))))
            combo.currentIndexChanged.connect(
                lambda _index, key=folder, box=combo:
                self._on_dataset_split_chosen(key, box.currentData()))
            table.setCellWidget(row, 1, combo)
        self._update_dataset_splits_label()

    def _on_dataset_split_chosen(self, folder: str, split: str):
        self._dataset_splits[folder] = split
        self._update_dataset_splits_label()
        self._dataset_export_changed()

    def dataset_splits(self) -> dict:
        """``{flight name: split}`` of the ticked flights, as the table
        shows them."""
        return {name: self._dataset_splits.get(folder, dataset_export.TRAIN)
                for name, folder in self._export_flights()}

    def _update_dataset_splits_label(self):
        splits = self.dataset_splits()
        self.dataset_splits_label.setText(
            dataset_export.describe(splits) if splits else
            "Tick the flights of the dataset in the Survey box.")

    def assign_dataset_splits(self):
        """Deal the listed flights into the splits by the shares."""
        flights = self._export_flights()
        if not flights:
            return
        shares = [spin.value() for spin in self.dataset_share_spins]
        try:
            dealt = dataset_export.assign(
                [folder for _name, folder in flights], shares,
                self.dataset_seed_spin.value())
        except dataset_export.DatasetExportError as exc:
            QMessageBox.warning(self, "Dataset Export", str(exc))
            return
        self._dataset_splits.update(dealt)
        self._refresh_dataset_flights()
        self._dataset_export_changed()

    # ------------------------------------------------------------------ #
    # Settings
    # ------------------------------------------------------------------ #

    def _on_dataset_format_changed(self):
        """Offer what the chosen format can carry, set as it does by
        default."""
        key = self.dataset_format_combo.currentData()
        self.dataset_images_check.setEnabled(
            key in exporters.SUPPORTS_IMAGES)
        self.dataset_false_positives_check.setEnabled(key != "dwca")
        blocked = self.dataset_false_positives_check.blockSignals(True)
        self.dataset_false_positives_check.setChecked(
            key not in exporters.TRAINING_FORMATS and key != "dwca")
        self.dataset_false_positives_check.blockSignals(blocked)
        self._dataset_export_changed()

    def _browse_dataset_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Dataset Export - choose a folder",
            self.dataset_output_edit.text().strip())
        if folder:
            self.dataset_output_edit.setText(folder)
            self._dataset_export_changed()

    def _dataset_output_folder(self) -> str:
        """The folder entered, else ``dataset`` in the survey folder."""
        entered = self.dataset_output_edit.text().strip()
        if entered:
            return entered
        folder = self.survey_panel.folder() if hasattr(
            self, "survey_panel") else ""
        return os.path.join(folder, "dataset") if folder else ""

    def dataset_export_settings(self) -> dict:
        """What the step is given as ``config["dataset_export"]``."""
        if not hasattr(self, "dataset_format_combo"):
            return {}
        return {
            "format": self.dataset_format_combo.currentData(),
            "output": self._dataset_output_folder(),
            "splits": self.dataset_splits(),
            "include_images": self.dataset_images_check.isChecked(),
            "include_not_an_animal": (
                self.dataset_false_positives_check.isChecked()
                if self.dataset_false_positives_check.isEnabled() else None),
        }

    def _dataset_export_changed(self):
        if getattr(self, "_loading_config", False):
            return
        from qgis.core import QgsProject
        try:
            self._save_dataset_export_settings(QgsProject.instance())
        except Exception:  # noqa: BLE001 - saved again with the project
            from ..core import quiet
            quiet.problem("The dataset export settings could not be saved "
                          "to the project now; they are saved again with "
                          "the project")

    def _save_dataset_export_settings(self, project) -> None:
        project.writeEntry(PLUGIN_SCOPE, ENTRY_DATASET_EXPORT, json.dumps({
            "format": self.dataset_format_combo.currentData(),
            "output": self.dataset_output_edit.text().strip(),
            "splits": dict(self._dataset_splits),
            "shares": [spin.value() for spin in self.dataset_share_spins],
            "seed": self.dataset_seed_spin.value(),
            "include_images": self.dataset_images_check.isChecked(),
            "include_not_an_animal":
                self.dataset_false_positives_check.isChecked(),
        }))

    def _load_dataset_export_settings(self, project) -> None:
        raw, ok = project.readEntry(PLUGIN_SCOPE, ENTRY_DATASET_EXPORT, "")
        try:
            values = json.loads(raw) if ok and raw else {}
        except ValueError:
            values = {}
        self._set_dataset_export_values(
            values if isinstance(values, dict) else {})

    def _reset_dataset_export_settings(self) -> None:
        self._set_dataset_export_values({})

    def _set_dataset_export_values(self, values: dict) -> None:
        widgets = [self.dataset_format_combo, self.dataset_output_edit,
                   self.dataset_seed_spin, self.dataset_images_check,
                   self.dataset_false_positives_check,
                   *self.dataset_share_spins]
        blocked = [widget.blockSignals(True) for widget in widgets]
        try:
            self.dataset_format_combo.setCurrentIndex(max(
                0, self.dataset_format_combo.findData(
                    values.get("format") or "coco")))
            self._on_dataset_format_changed_quietly()
            self.dataset_output_edit.setText(str(values.get("output") or ""))
            shares = values.get("shares") or dataset_export.DEFAULT_SHARES
            for spin, share in zip(self.dataset_share_spins, shares):
                spin.setValue(int(share))
            self.dataset_seed_spin.setValue(int(values.get("seed") or 0))
            self.dataset_images_check.setChecked(
                bool(values.get("include_images", True)))
            if "include_not_an_animal" in values:
                self.dataset_false_positives_check.setChecked(
                    bool(values["include_not_an_animal"]))
            splits = values.get("splits") or {}
            self._dataset_splits = {
                str(folder): split for folder, split in splits.items()
                if split in (*dataset_export.SPLITS, dataset_export.LEFT_OUT)}
        finally:
            for widget, was in zip(widgets, blocked):
                widget.blockSignals(was)
        self._refresh_dataset_flights()

    def _on_dataset_format_changed_quietly(self):
        loading = getattr(self, "_loading_config", False)
        self._loading_config = True
        try:
            self._on_dataset_format_changed()
        finally:
            self._loading_config = loading

    # ------------------------------------------------------------------ #
    # Running it
    # ------------------------------------------------------------------ #

    def run_dataset_export(self):
        """Export the ticked flights as a dataset (core.dataset_export)."""
        survey = self._checked_survey(check_species=False)
        if survey is None:
            return
        settings = self.dataset_export_settings()
        key = settings["format"]
        label = exporters.EXPORTERS[key][0]
        if key in exporters.NEEDS_CRS and not survey.epsg:
            QMessageBox.warning(
                self, "Dataset Export",
                f"{label} publishes latitude/longitude, so the target CRS "
                "must be set before exporting.")
            return
        if not settings["output"]:
            QMessageBox.warning(
                self, "Dataset Export", "Choose the folder to export into.")
            return
        if not any(split in dataset_export.SPLITS
                   for split in settings["splits"].values()):
            QMessageBox.warning(
                self, "Dataset Export",
                "No flight is in a split. Put the flights into train, "
                "validation or test first.")
            return
        try:
            dataset_export.check_output(settings["output"],
                                        settings["splits"])
        except dataset_export.DatasetExportError as exc:
            QMessageBox.warning(self, "Dataset Export", str(exc))
            return
        self.start_worker("dataset_export")
