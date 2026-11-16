# -*- coding: utf-8 -*-
"""Position Uncertainty: how far off the places of a survey's detections
may be, as a table.

A window over the survey's ``position_uncertainty_{source}.json``
(:mod:`core.position_uncertainty`): one row for all detections and - for a
survey of several flights - one per flight, the accuracies the figures
follow from, and what they say in words.

This module only shows; every number comes from the file.
"""

from typing import List, Tuple

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QDialogButtonBox, QLabel,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from .core import position_uncertainty

_RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

HEADERS = ["", "Detections", "Typically (m)", "9 in 10 within (m)",
           "Worst (m)", "Within 2 m", "Within 5 m", "Height (m)",
           "To the side (°)", "Matters most"]

#: The accuracies, as they are named to the user: ``(key, label, unit)``.
ACCURACY_LABELS = (
    ("gps_horizontal_m", "camera position", "m"),
    ("gps_vertical_m", "camera height", "m"),
    ("camera_angle_deg", "camera angle", "°"),
    ("terrain_m", "terrain model", "m"),
    ("pixel_px", "pixel", "px"),
    ("sync_s", "timing", "s"),
)


def _number(value, digits: int = 1) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _share(value) -> str:
    return "–" if value is None else f"{value * 100:.0f} %"


def _row(name: str, summary: dict) -> List[str]:
    within = summary.get("within") or {}
    sources = summary.get("main_sources") or {}
    main = "–"
    if sources:
        key = max(sources, key=lambda source: sources[source])
        main = f"{key} ({sources[key] * 100:.0f} %)"
    return [
        name,
        str(summary.get("boxes", 0)),
        _number(summary.get("median_m")),
        _number(summary.get("p90_m")),
        _number(summary.get("max_m")),
        _share(within.get("2")),
        _share(within.get("5")),
        _number(summary.get("median_height_m"), 0),
        _number(summary.get("median_off_nadir_deg"), 0),
        main,
    ]


def uncertainty_rows(document: dict) -> Tuple[List[str], List[List[str]]]:
    """``(headers, rows)`` of the table: all detections, then each flight
    of a survey of several - every cell a string."""
    rows = [_row("All detections", document.get("summary") or {})]
    for flight, summary in (document.get("by_flight") or {}).items():
        rows.append(_row(f"Flight {flight}", summary))
    return list(HEADERS), rows


def accuracy_text(document: dict) -> str:
    """The accuracies the figures follow from, in a line."""
    accuracy = document.get("accuracy") or {}
    parts = [f"{label} {accuracy[key]:g} {unit}"
             for key, label, unit in ACCURACY_LABELS if key in accuracy]
    return "Taken as known to (one sigma): " + ", ".join(parts) if parts \
        else ""


class PositionDialog(QDialog):
    """What the position uncertainty of a survey came to."""

    def __init__(self, document: dict, json_path: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Position Uncertainty ({document.get('camera', '')}, "
            f"{document.get('source', '')})")
        self.resize(1000, 400)
        layout = QVBoxLayout(self)

        self.accuracy_label = QLabel(accuracy_text(document))
        self.accuracy_label.setWordWrap(True)
        layout.addWidget(self.accuracy_label)

        headers, rows = uncertainty_rows(document)
        table = QTableWidget(len(rows), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        bold = None
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                item = QTableWidgetItem(text)
                if 0 < c < len(headers) - 1:
                    item.setTextAlignment(_RIGHT)
                if r == 0:
                    if bold is None:
                        bold = item.font()
                        bold.setBold(True)
                    item.setFont(bold)
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        self.table = table
        layout.addWidget(table)

        self.sentence_label = QLabel(position_uncertainty.sentence(
            document.get("summary") or {}))
        self.sentence_label.setWordWrap(True)
        layout.addWidget(self.sentence_label)

        notes = QLabel(document.get("notes", ""))
        notes.setWordWrap(True)
        notes.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(notes)
        if json_path:
            path_label = QLabel(
                f"Saved to: {json_path} (and _boxes.csv, _tracks.csv "
                "beside it)")
            path_label.setStyleSheet("color: gray; font-size: 10px;")
            layout.addWidget(path_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy_btn = buttons.addButton(
            "Copy", QDialogButtonBox.ButtonRole.ActionRole)
        copy_btn.clicked.connect(self.copy_table)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def copy_table(self):
        """Copy the table as tab-separated text (pastes into a
        spreadsheet)."""
        table = self.table
        lines = ["\t".join(table.horizontalHeaderItem(c).text()
                           for c in range(table.columnCount()))]
        for r in range(table.rowCount()):
            lines.append("\t".join(table.item(r, c).text()
                                   for c in range(table.columnCount())))
        QApplication.clipboard().setText("\n".join(lines))
