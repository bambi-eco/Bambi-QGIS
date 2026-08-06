# -*- coding: utf-8 -*-
"""Map a classifier's output classes onto the project's own vocabulary.

A head returns ``probs[i]``; what class ``i`` *means* is a property of the
model, and what the project calls that thing is a property of the project. This
dialog connects the two, and is the reason a third-party classifier is a
configuration change rather than a code change.

**The mapping is keyed on the class index, not the label.** Position is the
only thing a head guarantees; the names are for the user's benefit and are
stored with each prediction verbatim. That also means renaming a label here can
never silently re-target a mapping.

The class list is discovered where possible and defined by hand where not (see
the plan, §5.3): a head may not carry names, and one that has not been
downloaded yet cannot be asked at all.
"""

import os

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
    QHeaderView, QLabel, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout,
)

from .core import classification, hf_access
from .core import label_store

#: Which project vocabulary each task maps onto.
TASK_VOCABULARY = {
    "occlusion": "occlusion",   # an enum
    "sex": "sex",               # an enum
    "species": "species",       # the species list itself
}


class BambiLabelMappingDialog(QDialog):
    """Map one head's classes onto project values."""

    def __init__(self, task: str, spec: dict, target_folder: str,
                 models_dir: str = "", projection: str = "non_geo",
                 parent=None):
        super().__init__(parent)
        self.task = task
        self.spec = dict(spec or {})
        self.target_folder = target_folder
        self.models_dir = models_dir
        self.projection = projection

        self.setWindowTitle(f"{task.capitalize()} — class mapping")
        self.setMinimumSize(560, 420)

        layout = QVBoxLayout(self)

        info = QLabel(
            "Connect the classes this model returns to the values used in "
            "this project.\n\n"
            "The mapping follows the class <b>order</b>, not the names — a "
            "model returns positions, and the names are only there to read. "
            "Renaming one here never re-points the mapping."
        )
        info.setWordWrap(True)
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(info)

        self.model_label = QLabel(self._describe_model())
        self.model_label.setWordWrap(True)
        self.model_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.model_label)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            ["#", "Model class", "This project"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

        buttons_row = QHBoxLayout()
        self.detect_btn = QPushButton("Read classes from the model")
        self.detect_btn.setToolTip(
            "Load the model and take its class list. Falls back to asking it "
            "how many classes it returns when it carries no names."
        )
        self.detect_btn.clicked.connect(self.detect_classes)
        buttons_row.addWidget(self.detect_btn)

        add_btn = QPushButton("Add class")
        add_btn.setToolTip(
            "Define a class by hand — for a model that is not downloaded yet, "
            "or one whose own class list is wrong."
        )
        add_btn.clicked.connect(lambda: self._add_row("", None))
        buttons_row.addWidget(add_btn)

        remove_btn = QPushButton("Remove class")
        remove_btn.clicked.connect(self._remove_selected)
        buttons_row.addWidget(remove_btn)
        buttons_row.addStretch()
        layout.addLayout(buttons_row)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status)

        standard = QDialogButtonBox.StandardButton
        box = QDialogButtonBox(standard.Ok | standard.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

        self._values = self._load_vocabulary()
        self._populate()

    # -- vocabulary ------------------------------------------------------

    def _load_vocabulary(self):
        """``[(label, id)]`` the classes may be mapped onto."""
        vocabulary = label_store.vocabulary(self.target_folder)
        if not vocabulary:
            return []

        if self.task == "species":
            return [(row["name"], row["species_id"])
                    for row in vocabulary.get("species", [])]

        enum_name = TASK_VOCABULARY.get(self.task, self.task)
        return [(row["label"], row["value_id"])
                for row in vocabulary.get("enums", {}).get(enum_name, [])]

    def _describe_model(self):
        source = self.spec.get("model", "default")
        if source == "custom":
            path = self.spec.get("path", "")
            return f"Custom model: {path or '(none chosen)'}"
        if source == "off":
            return "This classifier is switched off."
        repo = hf_access.default_head_repo(self.task)
        if not repo:
            return "No default model has been published for this task yet."
        modality = self.spec.get("modality", "matched")
        return (f"Default model: {repo}/"
                f"{hf_access.head_repo_path(self.task, self.projection, modality)}")

    # -- contents --------------------------------------------------------

    def _populate(self):
        """Fill the table from the saved mapping, if there is one."""
        labels = self.spec.get("class_labels") or []
        mapping = self.spec.get("labels") or {}
        self.table.setRowCount(0)
        for index, name in enumerate(labels):
            self._add_row(name, mapping.get(str(index)))
        if not labels:
            self.status.setText(
                "No classes are defined yet. Read them from the model, or "
                "add them by hand if it is not available.")

    def _add_row(self, name: str, value_id):
        row = self.table.rowCount()
        self.table.insertRow(row)

        number = QTableWidgetItem(str(row))
        number.setFlags(number.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, 0, number)

        # Editable: a probed head has no names, and a wrong one is worth
        # correcting.
        self.table.setItem(row, 1, QTableWidgetItem(name))

        combo = QComboBox()
        combo.addItem("— not mapped —", None)
        for label, identifier in self._values:
            combo.addItem(label, identifier)
        if value_id is not None:
            found = combo.findData(value_id)
            if found >= 0:
                combo.setCurrentIndex(found)
        elif name:
            # Default to an exact name match, so a model that speaks the
            # project's language needs no clicks at all.
            found = combo.findText(name, Qt.MatchFlag.MatchFixedString)
            if found >= 0:
                combo.setCurrentIndex(found)
        self.table.setCellWidget(row, 2, combo)

    def _remove_selected(self):
        rows = sorted({index.row() for index in
                       self.table.selectionModel().selectedRows()},
                      reverse=True)
        for row in rows:
            self.table.removeRow(row)
        self._renumber()

    def _renumber(self):
        for row in range(self.table.rowCount()):
            self.table.item(row, 0).setText(str(row))

    # -- discovery -------------------------------------------------------

    def detect_classes(self):
        """Read the class list off the model, by whichever route works."""
        path = self._model_path()
        if not path:
            QMessageBox.information(
                self, "Class Mapping",
                "The model is not available locally yet.\n\n"
                "Either choose a custom model file, or run the classifier "
                "once so the default is downloaded — or define the classes "
                "by hand with 'Add class'.")
            return

        modality = self.spec.get("modality", "matched")
        dim = hf_access.feature_dim(modality)
        head = classification.Head(path, feature_dim=dim)
        try:
            names = head.classes
            source = head.class_source
        except classification.HeadError as exc:
            QMessageBox.warning(self, "Class Mapping", str(exc))
            return

        if not names:
            self.status.setText(
                "The model does not report its classes and could not be "
                "asked how many it returns. Add them by hand.")
            return

        existing = {row: self._value_at(row)
                    for row in range(self.table.rowCount())}
        self.table.setRowCount(0)
        for index, name in enumerate(names):
            self._add_row(name, existing.get(index))

        self.status.setText({
            "classes": f"{len(names)} class(es) read from the model.",
            "probe": (f"The model returns {len(names)} classes but does not "
                      "name them — fill the names in if you want them "
                      "readable; the mapping works either way."),
        }.get(source, f"{len(names)} class(es) found."))

    def _model_path(self):
        source = self.spec.get("model", "default")
        if source == "custom":
            path = (self.spec.get("path") or "").strip()
            return path if path and os.path.isfile(path) else ""
        if source == "off" or not self.models_dir:
            return ""
        path = hf_access.head_local_path(
            self.models_dir, self.task, self.projection,
            self.spec.get("modality", "matched"))
        return path if os.path.isfile(path) else ""

    # -- result ----------------------------------------------------------

    def _value_at(self, row):
        combo = self.table.cellWidget(row, 2)
        return None if combo is None else combo.currentData()

    def result_spec(self) -> dict:
        """The task spec with its mapping updated."""
        labels, mapping = [], {}
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 1)
            labels.append(item.text() if item else "")
            value = self._value_at(row)
            if value is not None:
                mapping[str(row)] = value

        spec = dict(self.spec)
        spec["class_labels"] = labels
        spec["labels"] = mapping

        # Occlusion additionally drives frame selection, so which classes mean
        # "usable" has to be recorded rather than guessed from a name.
        if self.task == "occlusion":
            clear = self._clear_selection(labels, mapping)
            spec["clear_labels"] = clear["labels"]
            spec["clear_values"] = clear["values"]
        return spec

    def _clear_selection(self, labels, mapping):
        """Which classes count as 'this frame is usable'.

        Taken from whichever class maps onto the project's first occlusion
        value — that is the "nothing is wrong with this frame" end of the enum
        by construction, in both the 6.1 vocabulary and a project that kept an
        older one.
        """
        if not self._values:
            return {"labels": ["clear"], "values": [0]}
        clear_id = self._values[0][1]
        chosen = [labels[int(row)] for row, value in mapping.items()
                  if value == clear_id and int(row) < len(labels)]
        return {"labels": chosen or ["clear"], "values": [clear_id]}
