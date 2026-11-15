# -*- coding: utf-8 -*-
"""The grid of the Grid Density analysis: its settings, and its result.

* :class:`GridSettingsPanel` - the shape of a cell, its size, how much of a
  cell has to have been seen, and how ground flown more than once is read.
  Kept with the QGIS project: a grid is only worth comparing with itself.
* :class:`GridStatisticsDialog` - what the analysis found, as a table: one
  row for all animals and one per species.

The analysis itself is :mod:`core.grid_density`.
"""

import json

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .core import grid_density, survey_grid
from .gui_utils import PLUGIN_SCOPE

#: QGIS project entry the settings are kept in, as JSON.
ENTRY_GRID = "Survey/Grid"

_RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

DEFAULTS = {
    "grid_shape": survey_grid.HEXAGON,
    "grid_size_mode": survey_grid.BY_AREA,
    "grid_cell_area_ha": 1.0,
    "grid_cell_edge_m": 100.0,
    "grid_cell_width_m": 100.0,
    "grid_cell_height_m": 100.0,
    "grid_min_seen_pct": 50.0,
    "grid_repeat": grid_density.AVERAGE,
}


class GridSettingsPanel(QWidget):
    """The controls that say how the grid is cut and read."""

    #: Emitted when the user changes a setting (not when one is loaded).
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loading = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        cells_row = QHBoxLayout()
        cells_row.addWidget(QLabel("Cells:"))
        self.shape_combo = QComboBox()
        for shape in survey_grid.SHAPES:
            self.shape_combo.addItem(survey_grid.SHAPE_LABELS[shape], shape)
        self.shape_combo.setToolTip(
            "The shape of a cell. Hexagons are the default: each of a "
            "hexagon's neighbours shares a whole edge with it and is "
            "equally far away, which is what the hot-spot statistics "
            "assume. Rectangles have a width and a height of their own.")
        cells_row.addWidget(self.shape_combo)

        self.size_mode_label = QLabel("Size by:")
        cells_row.addWidget(self.size_mode_label)
        self.size_mode_combo = QComboBox()
        self.size_mode_combo.addItem("Area", survey_grid.BY_AREA)
        self.size_mode_combo.addItem("Edge length", survey_grid.BY_EDGE)
        self.size_mode_combo.setToolTip(
            "Give a cell's size by its area or by the length of its edge - "
            "two ways to say the same: switching keeps the cells as they "
            "are.")
        cells_row.addWidget(self.size_mode_combo)

        self.area_spin = QDoubleSpinBox()
        self.area_spin.setDecimals(4)
        self.area_spin.setRange(0.0001, 100000.0)
        self.area_spin.setValue(DEFAULTS["grid_cell_area_ha"])
        self.area_spin.setSuffix(" ha")
        self.area_spin.setToolTip("Area of one cell in hectares")
        cells_row.addWidget(self.area_spin)

        self.edge_spin = QDoubleSpinBox()
        self.edge_spin.setDecimals(2)
        self.edge_spin.setRange(survey_grid.MIN_EDGE_M, 100000.0)
        self.edge_spin.setValue(DEFAULTS["grid_cell_edge_m"])
        self.edge_spin.setSuffix(" m")
        self.edge_spin.setToolTip("Length of one edge of a cell in metres")
        cells_row.addWidget(self.edge_spin)

        self.width_label = QLabel("Width:")
        cells_row.addWidget(self.width_label)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setDecimals(2)
        self.width_spin.setRange(survey_grid.MIN_EDGE_M, 100000.0)
        self.width_spin.setValue(DEFAULTS["grid_cell_width_m"])
        self.width_spin.setSuffix(" m")
        self.width_spin.setToolTip("East-west side of a rectangle")
        cells_row.addWidget(self.width_spin)
        self.height_label = QLabel("Height:")
        cells_row.addWidget(self.height_label)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setDecimals(2)
        self.height_spin.setRange(survey_grid.MIN_EDGE_M, 100000.0)
        self.height_spin.setValue(DEFAULTS["grid_cell_height_m"])
        self.height_spin.setSuffix(" m")
        self.height_spin.setToolTip("North-south side of a rectangle")
        cells_row.addWidget(self.height_spin)
        cells_row.addStretch()
        layout.addLayout(cells_row)

        read_row = QHBoxLayout()
        read_row.addWidget(QLabel("Seen at least:"))
        self.min_seen_spin = QDoubleSpinBox()
        self.min_seen_spin.setDecimals(0)
        self.min_seen_spin.setRange(1.0, 100.0)
        self.min_seen_spin.setSingleStep(5.0)
        self.min_seen_spin.setValue(DEFAULTS["grid_min_seen_pct"])
        self.min_seen_spin.setSuffix(" %")
        self.min_seen_spin.setToolTip(
            "A cell counts when at least this share of it was seen by the "
            "cameras. Cells below it stay on the map as 'too little seen' "
            "and are left out of the statistics: one animal on the corner "
            "of a cell that was seen says little about the cell.")
        read_row.addWidget(self.min_seen_spin)
        read_row.addWidget(QLabel("Repeated flights:"))
        self.repeat_combo = QComboBox()
        for mode in grid_density.REPEAT_MODES:
            self.repeat_combo.addItem(grid_density.REPEAT_LABELS[mode], mode)
        self.repeat_combo.setToolTip(
            "How a cell that several flights saw is read.\n"
            "Average over the flights: all sightings over all the ground "
            "looked at - the mean density over the visits.\n"
            "Highest flight: the flight with the highest density, among "
            "those that saw enough of the cell on their own - what was "
            "there at least.")
        read_row.addWidget(self.repeat_combo)
        read_row.addStretch()
        layout.addLayout(read_row)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.summary_label)

        self.shape_combo.currentIndexChanged.connect(self._on_shape_changed)
        self.size_mode_combo.currentIndexChanged.connect(
            self._on_size_mode_changed)
        for spin in (self.area_spin, self.edge_spin, self.width_spin,
                     self.height_spin, self.min_seen_spin):
            spin.valueChanged.connect(self._on_value_changed)
        self.repeat_combo.currentIndexChanged.connect(self._on_value_changed)
        self._sync()

    # ------------------------------------------------------------------ #
    # The settings
    # ------------------------------------------------------------------ #

    def values(self) -> dict:
        """The settings as the step takes them (``grid_*`` config keys)."""
        return {
            "grid_shape": self.shape_combo.currentData(),
            "grid_size_mode": self.size_mode_combo.currentData(),
            "grid_cell_area_ha": self.area_spin.value(),
            "grid_cell_edge_m": self.edge_spin.value(),
            "grid_cell_width_m": self.width_spin.value(),
            "grid_cell_height_m": self.height_spin.value(),
            "grid_min_seen_pct": self.min_seen_spin.value(),
            "grid_repeat": self.repeat_combo.currentData(),
        }

    def set_values(self, values: dict) -> None:
        """Show *values* (missing ones: the defaults) without reporting a
        change."""
        merged = dict(DEFAULTS)
        merged.update({key: value for key, value in (values or {}).items()
                       if key in DEFAULTS and value is not None})
        self._loading = True
        try:
            for combo, key in ((self.shape_combo, "grid_shape"),
                               (self.size_mode_combo, "grid_size_mode"),
                               (self.repeat_combo, "grid_repeat")):
                index = combo.findData(str(merged[key]))
                combo.setCurrentIndex(index if index >= 0 else 0)
            for spin, key in ((self.area_spin, "grid_cell_area_ha"),
                              (self.edge_spin, "grid_cell_edge_m"),
                              (self.width_spin, "grid_cell_width_m"),
                              (self.height_spin, "grid_cell_height_m"),
                              (self.min_seen_spin, "grid_min_seen_pct")):
                try:
                    spin.setValue(float(merged[key]))
                except (TypeError, ValueError):
                    spin.setValue(float(DEFAULTS[key]))
        finally:
            self._loading = False
        self._sync()

    def spec(self) -> survey_grid.GridSpec:
        """The grid the controls describe."""
        return survey_grid.GridSpec.from_config(self.values())

    # ------------------------------------------------------------------ #
    # Keeping the controls in step
    # ------------------------------------------------------------------ #

    def _sync(self) -> None:
        """Show the controls that apply, and say what the grid is."""
        rectangle = self.shape_combo.currentData() == survey_grid.RECTANGLE
        by_area = self.size_mode_combo.currentData() == survey_grid.BY_AREA
        for widget in (self.size_mode_label, self.size_mode_combo):
            widget.setVisible(not rectangle)
        self.area_spin.setVisible(not rectangle and by_area)
        self.edge_spin.setVisible(not rectangle and not by_area)
        for widget in (self.width_label, self.width_spin,
                       self.height_label, self.height_spin):
            widget.setVisible(rectangle)
        try:
            text = self.spec().describe()
            text = text[0].upper() + text[1:]
        except survey_grid.GridError as exc:
            text = str(exc)
        self.summary_label.setText(
            f"{text}. A cell counts when at least "
            f"{self.min_seen_spin.value():.0f} % of it was seen; its "
            "density is its animals over the ground that was seen in it.")

    def _on_shape_changed(self, _index=None) -> None:
        self._on_value_changed()

    def _on_size_mode_changed(self, _index=None) -> None:
        """Area and edge are two ways to say the same: the one that is now
        shown takes the size the other one had, so the cells stay as they
        are."""
        shape = self.shape_combo.currentData()
        if not self._loading and shape != survey_grid.RECTANGLE:
            self._loading = True
            try:
                if self.size_mode_combo.currentData() == survey_grid.BY_EDGE:
                    self.edge_spin.setValue(survey_grid.edge_for_area(
                        shape, self.area_spin.value() * 1e4))
                else:
                    self.area_spin.setValue(survey_grid.area_for_edge(
                        shape, self.edge_spin.value()) / 1e4)
            finally:
                self._loading = False
        self._on_value_changed()

    def _on_value_changed(self, _value=None) -> None:
        self._sync()
        if not self._loading:
            self.changed.emit()

    # ------------------------------------------------------------------ #
    # Persistence (QGIS project)
    # ------------------------------------------------------------------ #

    def save(self, project) -> None:
        project.writeEntry(PLUGIN_SCOPE, ENTRY_GRID,
                           json.dumps(self.values()))

    def load(self, project) -> None:
        raw, ok = project.readEntry(PLUGIN_SCOPE, ENTRY_GRID, "")
        try:
            values = json.loads(raw) if ok and raw else {}
        except ValueError:
            values = {}
        self.set_values(values if isinstance(values, dict) else {})

    def reset(self) -> None:
        self.set_values({})


# --------------------------------------------------------------------------- #
# The result
# --------------------------------------------------------------------------- #

def _number(value, digits: int = 1) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _interval(entry, digits: int = 1) -> str:
    if not entry or entry.get("estimate") is None:
        return "–"
    text = _number(entry["estimate"], digits)
    if entry.get("low") is not None:
        text += (f" ({_number(entry['low'], digits)}–"
                 f"{_number(entry['high'], digits)})")
    return text


def _p(value) -> str:
    if value is None:
        return ""
    return "p < 0.001" if value < 0.001 else f"p = {value:.3f}"


def statistics_rows(document: dict):
    """``(headers, rows)`` of the statistics table: one row for all animals
    and one per species, every cell a string."""
    headers = ["", "Animals", "Density /km² (95 %)", "In the survey area",
               "Variance-to-mean", "Pattern", "Moran's I",
               "Cells with animals", "Hot spots", "Cold spots"]
    rows = []
    for name, stats in document.get("statistics", {}).items():
        dispersion = stats.get("dispersion") or {}
        moran = stats.get("morans_i") or {}
        occupancy = stats.get("occupancy") or {}
        index = dispersion.get("index")
        spread = "–" if index is None else (
            f"{index:.2f} ({_p(dispersion.get('p_clustered'))})")
        alike = "–" if moran.get("i") is None else (
            f"{moran['i']:.2f} ({_p(moran.get('p_value'))})")
        share = "–"
        if occupancy.get("share") is not None:
            share = (f"{occupancy['occupied']} of {occupancy['cells']} "
                     f"({occupancy['share'] * 100:.0f} %, "
                     f"{occupancy['low'] * 100:.0f}–"
                     f"{occupancy['high'] * 100:.0f} %)")
        rows.append([
            "All animals" if name == grid_density.ALL else name,
            str(stats.get("animals", 0)),
            _interval(stats.get("density_per_km2")),
            _interval(stats.get("total_in_survey_area"), 0),
            spread,
            grid_density.PATTERN_TEXT.get(dispersion.get("pattern"), "–"),
            alike,
            share,
            str(stats.get("hot_spots", 0)),
            str(stats.get("cold_spots", 0)),
        ])
    return headers, rows


class GridStatisticsDialog(QDialog):
    """What the grid analysis found: one row for all animals, one per
    species."""

    def __init__(self, document: dict, json_path: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Grid Density ({document.get('camera', '')}, "
            f"{document.get('source', '')})")
        self.resize(1040, 380)
        layout = QVBoxLayout(self)

        summary = document.get("summary") or {}
        facts = [
            f"Flights: {', '.join(document.get('flights', []))}",
            f"grid: {document.get('grid_text', '')}",
            f"{summary.get('counted', 0)} of {summary.get('cells', 0)} "
            f"cells counted ({summary.get('too_little_seen', 0)} too little "
            f"seen, {summary.get('not_seen', 0)} not seen)",
        ]
        if summary.get("seen_share") is not None:
            facts.append(f"{summary['seen_share'] * 100:.0f} % of "
                         f"{summary.get('region_ha', 0):.1f} ha seen")
        facts.append("repeated flights: " + grid_density.REPEAT_LABELS.get(
            document.get("repeat"), "").lower())
        header = QLabel(" · ".join(facts))
        header.setWordWrap(True)
        layout.addWidget(header)

        headers, rows = statistics_rows(document)
        table = QTableWidget(len(rows), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        bold = None
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                item = QTableWidgetItem(text)
                if c in (1, 8, 9):
                    item.setTextAlignment(_RIGHT)
                if r == 0:
                    if bold is None:
                        bold = item.font()
                        bold.setBold(True)
                    item.setFont(bold)
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        self._table = table
        layout.addWidget(table)

        notes = QLabel(document.get("notes", ""))
        notes.setWordWrap(True)
        notes.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(notes)
        if json_path:
            path_label = QLabel(
                f"Saved to: {json_path} (and _cells.csv, _by_flight.csv "
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
        table = self._table
        lines = ["\t".join(table.horizontalHeaderItem(c).text()
                           for c in range(table.columnCount()))]
        for r in range(table.rowCount()):
            lines.append("\t".join(table.item(r, c).text()
                                   for c in range(table.columnCount())))
        QApplication.clipboard().setText("\n".join(lines))
