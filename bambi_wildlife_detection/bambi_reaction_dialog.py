# -*- coding: utf-8 -*-
"""Reaction to the Drone: what the tracked animals did, as a table.

A window over the survey's ``drone_reaction.json``
(:mod:`core.drone_reaction`): one row for all animals, one per species and -
for a survey of several flights - one per flight, and under the table what
the rows say in words.

This module only shows; every number comes from the file.
"""

from typing import List, Tuple

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QDialogButtonBox, QLabel,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from .core import drone_reaction

_RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

HEADERS = ["", "Animals", "Judged", "Moved", "Went (m)", "Speed (m/s)",
           "Away", "Towards", "Along the line", "Mean speed away (m/s)",
           "Nearer = faster", "In a word"]


def _number(value, digits: int = 1) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _row(name: str, summary: dict) -> List[str]:
    moved = "–"
    if summary.get("moved_share") is not None:
        moved = (f"{summary['moved']} ({summary['moved_share'] * 100:.0f} %, "
                 f"{summary['moved_share_low'] * 100:.0f}–"
                 f"{summary['moved_share_high'] * 100:.0f} %)")
    directions = summary.get("directions") or {}
    away = "–"
    if summary.get("mean_away_ms") is not None:
        away = f"{summary['mean_away_ms']:+.2f}"
        if summary.get("mean_away_ms_low") is not None:
            away += (f" ({summary['mean_away_ms_low']:+.2f} to "
                     f"{summary['mean_away_ms_high']:+.2f})")
    nearer = summary.get("nearer_faster")
    related = "–"
    if nearer:
        chance = nearer["p_value"]
        odds = "p < 0.001" if chance < 0.001 else f"p = {chance:.3f}"
        related = f"{nearer['correlation']:+.2f} ({odds})"
    lengthwise = sum(directions.get(name, 0) for name in (
        drone_reaction.AHEAD, drone_reaction.BACK))
    return [
        "All animals" if name == drone_reaction.ALL else str(name),
        str(summary.get("tracks", 0)),
        str(summary.get("judged", 0)),
        moved,
        _number(summary.get("median_displacement_m")),
        _number(summary.get("median_speed_ms"), 2),
        str(directions.get(drone_reaction.AWAY, 0)),
        str(directions.get(drone_reaction.TOWARDS, 0)),
        str(lengthwise),
        away,
        related,
        summary.get("verdict") or "",
    ]


def reaction_rows(document: dict) -> Tuple[List[str], List[List[str]]]:
    """``(headers, rows)`` of the table: all animals, each species, each
    flight of a survey of several - every cell a string."""
    rows = [_row(name, summary)
            for name, summary in (document.get("summary") or {}).items()]
    for flight, summary in (document.get("by_flight") or {}).items():
        rows.append(_row(f"Flight {flight}", summary))
    return list(HEADERS), rows


def sentences(document: dict) -> str:
    """What the animals did, in words: all of them, then species by
    species."""
    lines = []
    for name, summary in (document.get("summary") or {}).items():
        title = "All animals" if name == drone_reaction.ALL else str(name)
        lines.append(f"{title}: {summary.get('sentence', '')}")
    return "\n".join(lines)


class ReactionDialog(QDialog):
    """What the reading of the reaction found."""

    def __init__(self, document: dict, json_path: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Reaction to the Drone ({document.get('camera', '')})")
        self.resize(1080, 420)
        layout = QVBoxLayout(self)

        settings = document.get("settings") or {}
        facts = [f"Flights: {', '.join(document.get('flights', []))}"]
        if settings.get("min_displacement_m") is not None:
            pace = ""
            if settings.get("min_speed_ms"):
                pace = f" at {settings['min_speed_ms']:g} m/s or faster"
            facts.append("moved = at least "
                         f"{settings['min_displacement_m']:g} m{pace} while "
                         "in view")
        header = QLabel(" · ".join(facts))
        header.setWordWrap(True)
        layout.addWidget(header)

        headers, rows = reaction_rows(document)
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

        self.sentences_label = QLabel(sentences(document))
        self.sentences_label.setWordWrap(True)
        layout.addWidget(self.sentences_label)

        notes = QLabel(document.get("notes", ""))
        notes.setWordWrap(True)
        notes.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(notes)
        if json_path:
            path_label = QLabel(
                f"Saved to: {json_path} (and _tracks.csv beside it)")
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
