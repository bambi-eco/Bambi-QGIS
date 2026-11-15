# -*- coding: utf-8 -*-
"""Processing > Accuracy: the pipeline's output against hand labels.

One button: it opens the accuracy report (``bambi_accuracy_dialog``) for the
flight that is selected, on the camera the animals are tracked on.
"""

import os

from qgis.PyQt.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout,
)


class AccuracyMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _build_accuracy_group(self) -> QGroupBox:
        """The Accuracy group of the Processing tab."""
        group = QGroupBox("Accuracy")
        layout = QVBoxLayout(group)
        note = QLabel(
            "How right was the pipeline? On a flight you went through in "
            "the Labelling Tool, compare its detections, tracks and names "
            "with your labels: precision and recall, the confidence worth "
            "asking for, animals split into several tracks, and which "
            "species are mistaken for which.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(note)
        row = QHBoxLayout()
        self.accuracy_btn = QPushButton("→ Compare with Hand Labels…")
        self.accuracy_btn.setToolTip(
            "Open the accuracy report of this flight. Nothing is changed; "
            "the result is also written to analytics_{camera}/accuracy.html, "
            ".json and accuracy_boxes.csv in the flight's folder.")
        self.accuracy_btn.clicked.connect(self.open_accuracy_report)
        row.addWidget(self.accuracy_btn)
        row.addStretch()
        layout.addLayout(row)
        return group

    def open_accuracy_report(self):
        """Open the accuracy report for the active flight and compare."""
        from ..bambi_accuracy_dialog import AccuracyDialog

        folder = self.target_folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(
                self, "Accuracy",
                "Select a flight first - the report compares one flight's "
                "results with its labels.")
            return None
        flight = self._active_flight() or {}
        modality = "t" if self.tracking_camera_combo.currentIndex() == 0 \
            else "w"
        dialog = AccuracyDialog(folder, flight.get("name") or "", modality,
                                parent=self, log_fn=self.log)
        dialog.setModal(False)
        dialog.show()
        dialog.compare()
        self._accuracy_dialog = dialog
        return dialog
