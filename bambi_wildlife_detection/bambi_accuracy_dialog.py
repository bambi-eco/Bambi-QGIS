# -*- coding: utf-8 -*-
"""The accuracy report: the pipeline's output against hand labels.

A window on the comparison of :mod:`core.accuracy`: which camera, how much
two boxes have to overlap, which frames were gone through - and the result
as a page that answers three questions in turn: were the animals found, was
each one track, was it named right.

Every comparison is also written into the flight's folder
(``analytics_{camera}/accuracy.html``, ``accuracy.json`` and
``accuracy_boxes.csv`` - every box with what became of it).
"""

import os

from qgis.PyQt.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QHBoxLayout, QLabel, QPushButton, QSpinBox, QTextBrowser, QVBoxLayout,
)

from .core import accuracy

_CAMERAS = (("Thermal", "t"), ("RGB", "w"))


class AccuracyDialog(QDialog):
    """Compare a flight's detections, tracks and names with its labels."""

    def __init__(self, target_folder: str, flight_name: str = "",
                 modality: str = "t", parent=None, log_fn=None):
        super().__init__(parent)
        self._folder = target_folder
        self._flight = flight_name or os.path.basename(
            os.path.normpath(target_folder))
        self._log = log_fn
        #: The last result (``None`` until a comparison worked).
        self.document = None
        self.setWindowTitle(f"Accuracy against hand labels - {self._flight}")
        self.resize(900, 640)
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Compares what the pipeline found on this flight with the "
            "animals labelled in the Labelling Tool: the detector's boxes "
            "(hand-drawn ones are left out), the tracker's tracks and the "
            "species, sex and age the survey tools count them as. A label "
            "of the species not-an-animal says \"this is none\".")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        row = QHBoxLayout()
        row.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        for label, key in _CAMERAS:
            self.camera_combo.addItem(label, key)
        self.camera_combo.setCurrentIndex(1 if modality == "w" else 0)
        row.addWidget(self.camera_combo)

        row.addWidget(QLabel("Overlap (IoU) at least:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setDecimals(2)
        self.iou_spin.setRange(accuracy.MIN_IOU, accuracy.MAX_IOU)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(accuracy.DEFAULT_IOU)
        self.iou_spin.setToolTip(
            "How much a box of the pipeline and a labelled box have to "
            "overlap to be the same animal: the area they share over the "
            "area they cover together. Labels are drawn on key frames and "
            "filled in between, and a thermal animal is a few pixels "
            "across - 0.3 allows for that; benchmark data sets use 0.5.")
        row.addWidget(self.iou_spin)

        self.auto_frames_check = QCheckBox("First to last labelled frame")
        self.auto_frames_check.setChecked(True)
        self.auto_frames_check.setToolTip(
            "Which frames were gone through. A frame without a label only "
            "says that there is no animal if somebody looked at it: by "
            "default the frames from the first to the last label count. "
            "Untick to name the frames - the whole flight, if all of it "
            "was gone through.")
        row.addWidget(self.auto_frames_check)
        self.first_spin = QSpinBox()
        self.first_spin.setRange(0, 10_000_000)
        self.first_spin.setPrefix("frames ")
        self.last_spin = QSpinBox()
        self.last_spin.setRange(0, 10_000_000)
        self.last_spin.setPrefix("to ")
        row.addWidget(self.first_spin)
        row.addWidget(self.last_spin)
        self.auto_frames_check.toggled.connect(self._sync_frames)

        self.compare_btn = QPushButton("Compare")
        self.compare_btn.setDefault(True)
        self.compare_btn.clicked.connect(self.compare)
        row.addWidget(self.compare_btn)
        row.addStretch()
        layout.addLayout(row)

        self.page = QTextBrowser()
        self.page.setOpenExternalLinks(False)
        layout.addWidget(self.page)

        self.saved_label = QLabel("")
        self.saved_label.setStyleSheet("color: gray; font-size: 10px;")
        self.saved_label.setWordWrap(True)
        layout.addWidget(self.saved_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._sync_frames()

    def _sync_frames(self) -> None:
        named = not self.auto_frames_check.isChecked()
        self.first_spin.setEnabled(named)
        self.last_spin.setEnabled(named)

    def modality(self) -> str:
        return self.camera_combo.currentData()

    def output_folder(self) -> str:
        return os.path.join(self._folder, f"analytics_{self.modality()}")

    def compare(self) -> bool:
        """Run the comparison and show it. ``False`` when there is nothing
        to compare - the page then says why."""
        frames = None
        if not self.auto_frames_check.isChecked():
            frames = (self.first_spin.value(), self.last_spin.value())
        camera = self.camera_combo.currentText()
        try:
            document = accuracy.for_flight(
                self._folder, self.modality(), self.iou_spin.value(), frames)
        except accuracy.AccuracyError as exc:
            self.document = None
            self.page.setHtml(f"<p><b>Nothing to compare.</b></p><p>{exc}</p>")
            self.saved_label.setText("")
            return False
        self.document = document
        title = f"Accuracy against hand labels - {self._flight} ({camera})"
        self.page.setHtml(accuracy.html_report(document, title))
        if self.auto_frames_check.isChecked():
            # show which frames it was
            self.first_spin.setValue(document["frames"]["first"])
            self.last_spin.setValue(document["frames"]["last"])

        folder = self.output_folder()
        try:
            os.makedirs(folder, exist_ok=True)
            accuracy.write_json(document,
                                os.path.join(folder, "accuracy.json"))
            accuracy.write_rows(document,
                                os.path.join(folder, "accuracy_boxes.csv"))
            with open(os.path.join(folder, "accuracy.html"), "w",
                      encoding="utf-8") as handle:
                handle.write(accuracy.html_report(document, title))
            self.saved_label.setText(
                f"Saved to {folder}: accuracy.html, accuracy.json, "
                "accuracy_boxes.csv (every box with what became of it)")
        except OSError as exc:
            self.saved_label.setText(f"Could not be saved to {folder}: {exc}")
        if self._log:
            self._log(f"Accuracy against hand labels ({self._flight}, "
                      f"{camera}):")
            for line in accuracy.describe(document):
                self._log(f"  {line}")
        return True
