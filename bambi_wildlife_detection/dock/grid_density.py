# -*- coding: utf-8 -*-
"""Survey Analytics > Grid Density: the survey counted cell by cell.

The group on the Survey Analytics tab - how the grid is cut
(``bambi_grid_panel.GridSettingsPanel``), the button that runs the analysis
(``core.grid_density``), the table of what it found - and the grid on the
map: every cell with all its figures, drawn by density, with a second style
for the hot spots.
"""

import os

from qgis.PyQt.QtGui import QColor, QFont
from qgis.PyQt.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout,
)
from qgis.core import (
    QgsCategorizedSymbolRenderer, QgsFeature, QgsField, QgsFillSymbol,
    QgsGeometry, QgsPalLayerSettings, QgsPointXY, QgsRendererCategory,
    QgsTextBufferSettings, QgsTextFormat, QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)

from ..bambi_grid_panel import GridSettingsPanel, GridStatisticsDialog
from ..core import store as _store_kinds
from ..qgis_compat import FieldType

#: Names of the two styles the grid layer comes with.
STYLE_DENSITY = "Density"
STYLE_HOT_SPOTS = "Hot spots"

#: The attributes the two styles draw by: the class a cell is in. They are
#: worked out when the layer is made - a style that only names a value per
#: class is one QGIS shows, saves and restores without surprises.
DENSITY_CLASS = "density_class"
SPOT_CLASS = "spot_class"
NO_ANIMALS = "no animals"
NOTHING_STANDS_OUT = "nothing stands out"

#: Fill colours of the density classes, lowest first (ColorBrewer YlOrRd).
_DENSITY_COLOURS = ("#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026")
#: Colour of a hot or cold spot, by what it is called.
_SPOT_COLOURS = {
    "hot spot (99 %)": "#b2182b", "hot spot (95 %)": "#ef8a62",
    "hot spot (90 %)": "#fddbc7",
    "cold spot (90 %)": "#d1e5f0", "cold spot (95 %)": "#67a9cf",
    "cold spot (99 %)": "#2166ac",
}


def density_breaks(values, classes: int = 5):
    """Upper limits of up to *classes* classes for the densities above
    nought, by rank - each class holds about as many cells - rounded to two
    figures so that a legend can be read."""
    found = sorted(float(v) for v in values if v is not None and v > 0)
    if not found:
        return []

    def rounded(value: float) -> float:
        if value <= 0:
            return 0.0
        from math import floor, log10
        digits = 1 - int(floor(log10(value)))
        return round(value, digits)

    limits = []
    for step in range(1, classes + 1):
        index = min(len(found) - 1,
                    max(0, int(round(step * len(found) / classes)) - 1))
        limit = rounded(found[index])
        if step == classes:
            limit = max(limit, found[-1])
        if not limits or limit > limits[-1]:
            limits.append(limit)
    return limits


def density_classes(limits) -> list:
    """The labels of the density classes for *limits*
    (:func:`density_breaks`), lowest first."""
    labels, lower = [], 0.0
    for upper in limits:
        labels.append(f"{lower:g} - {upper:g} /km²")
        lower = upper
    return labels


def density_class(cell: dict, limits) -> str:
    """The class a cell is drawn in by density: its status where it has no
    density, :data:`NO_ANIMALS`, or the class its density falls in."""
    if cell.get("status") != "counted":
        return cell.get("status") or ""
    density = cell.get("density_per_km2") or 0.0
    if density <= 0:
        return NO_ANIMALS
    labels = density_classes(limits)
    for label, upper in zip(labels, limits):
        if density <= upper:
            return label
    return labels[-1] if labels else NO_ANIMALS


def spot_class(cell: dict) -> str:
    """The class a cell is drawn in by what stands out."""
    if cell.get("status") != "counted":
        return cell.get("status") or ""
    return cell.get("spot") or NOTHING_STANDS_OUT


class GridDensityMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _build_grid_group(self) -> QGroupBox:
        """The Grid Density group of the Survey Analytics tab."""
        group = QGroupBox("Grid Density")
        layout = QVBoxLayout(group)
        description = QLabel(
            "The survey's ground cut into equal cells - the survey area, or "
            "without one the ground the flights saw - and counted cell by "
            "cell: animals and density per species, sex and age, groups, "
            "and the environment each cell lies in. Over the cells: the "
            "mean density with an interval, the total for the survey area, "
            "how clumped the animals are, and where the hot spots are. "
            "Needs the field-of-view footprints of every flight.")
        description.setWordWrap(True)
        description.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(description)

        self.grid_panel = GridSettingsPanel(self)
        self.grid_panel.changed.connect(self._on_grid_settings_changed)
        layout.addWidget(self.grid_panel)

        run_row = QHBoxLayout()
        self.grid_btn = QPushButton("→ Analyse Grid Density")
        self.grid_btn.clicked.connect(self.run_grid_density)
        self.grid_btn.setToolTip(
            "Write analytics_{camera}/grid_density_{source}.json, "
            "_cells.csv and _by_flight.csv into the survey folder and show "
            "the statistics.")
        self.grid_status = QLabel("⚪")
        run_row.addWidget(self.grid_btn)
        run_row.addWidget(self.grid_status)
        layout.addLayout(run_row)

        show_row = QHBoxLayout()
        self.grid_show_btn = QPushButton("→ Show Grid Statistics")
        self.grid_show_btn.clicked.connect(self.show_grid_statistics)
        show_row.addWidget(self.grid_show_btn)
        show_row.addStretch()
        layout.addLayout(show_row)

        add_row = QHBoxLayout()
        self.add_grid_btn = QPushButton("→ Add Grid to QGIS")
        self.add_grid_btn.clicked.connect(self.add_grid_to_qgis)
        self.add_grid_btn.setToolTip(
            "The cells as a layer with every figure as an attribute, "
            "coloured by density. Cells that were too little seen are "
            "hatched, cells nobody saw left empty. A second style, 'Hot "
            "spots', is in the layer's style menu.")
        self.add_grid_status = QLabel("⚪")
        add_row.addWidget(self.add_grid_btn)
        add_row.addWidget(self.add_grid_status)
        layout.addLayout(add_row)
        return group

    def _on_grid_settings_changed(self) -> None:
        """A grid setting was changed by the user: it is kept with the
        project, like the survey's flights."""
        if getattr(self, "_loading_config", False):
            return
        self.save_config_to_project()

    # ------------------------------------------------------------------ #
    # Running it, and its result
    # ------------------------------------------------------------------ #

    def _grid_result_path(self, survey=None) -> str:
        survey = survey or self.current_survey()
        return os.path.join(
            survey.analytics_folder,
            f"grid_density_{self._survey_source()}.json")

    def run_grid_density(self):
        """Count the survey cell by cell (core.grid_density)."""
        from ..core import survey_grid

        survey = self._checked_survey()
        if survey is None:
            return
        try:
            self.grid_panel.spec()
        except survey_grid.GridError as exc:
            QMessageBox.warning(self, "Grid Density", str(exc))
            return
        source = self._survey_source()
        if source == "tracks":
            checks = self._tracks_check(survey)
        else:
            def checks(flight):
                if self.has_stage(_store_kinds.DETECTIONS, survey.modality,
                                  flight.target_folder):
                    return []
                return [f"{survey.camera_label} detections (run 'Detect "
                        "Animals' first)"]
        if self._survey_prerequisites(survey, checks):
            self.start_worker("grid_density")

    def show_grid_statistics(self):
        """Show what the last grid analysis of the survey found."""
        from ..core import grid_density

        path = self._grid_result_path()
        if not os.path.isfile(path):
            QMessageBox.warning(
                self, "Missing Data",
                "The grid has not been analysed yet.\n"
                "Run 'Analyse Grid Density' first.")
            return
        try:
            document = grid_density.read_json(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Grid Density",
                                 f"Could not read {path}: {exc}")
            return
        dialog = GridStatisticsDialog(document, json_path=path, parent=self)
        dialog.setModal(False)
        dialog.show()
        self._grid_dialog = dialog

    # ------------------------------------------------------------------ #
    # The grid on the map
    # ------------------------------------------------------------------ #

    def add_grid_to_qgis(self):
        """Add the cells of the last grid analysis as a layer."""
        from ..core import grid_density

        survey = self.current_survey()
        path = self._grid_result_path(survey)
        if not os.path.isfile(path):
            QMessageBox.warning(
                self, "Missing Data",
                "The grid has not been analysed yet.\n"
                "Run 'Analyse Grid Density' first.")
            return

        name = f"BAMBI Grid Density ({survey.camera_label})"
        try:
            self.update_status("add_grid", "🟡 Loading...")
            document = grid_density.read_json(path)
            rows = grid_density.flat_cells(document)
            epsg = survey.epsg or self.get_config().get("target_epsg", 32633)
            layer = QgsVectorLayer(f"Polygon?crs=EPSG:{epsg}", name, "memory")
            provider = layer.dataProvider()
            columns = list(rows[0].keys()) if rows else ["cell"]
            limits = density_breaks(
                cell.get("density_per_km2") for cell in document["cells"]
                if cell.get("status") == "counted")

            def field(column):
                sample = next((row[column] for row in rows
                               if row.get(column) is not None), None)
                if isinstance(sample, bool):
                    return QgsField(column, FieldType.Bool)
                if isinstance(sample, int):
                    return QgsField(column, FieldType.LongLong)
                if isinstance(sample, float):
                    return QgsField(column, FieldType.Double)
                return QgsField(column, FieldType.String)

            fields = [field(column) for column in columns]
            fields.append(QgsField(DENSITY_CLASS, FieldType.String))
            fields.append(QgsField(SPOT_CLASS, FieldType.String))
            provider.addAttributes(fields)
            layer.updateFields()

            features = []
            for row, cell in zip(rows, document["cells"]):
                feature = QgsFeature(layer.fields())
                feature.setGeometry(QgsGeometry.fromPolygonXY([[
                    QgsPointXY(float(x), float(y)) for x, y in cell["ring"]]]))
                values = [row.get(column) for column in columns]
                values.append(density_class(cell, limits))
                values.append(spot_class(cell))
                feature.setAttributes(values)
                features.append(feature)
            provider.addFeatures(features)
            layer.updateExtents()

            layer = self._persist_memory_layer(
                layer,
                f"Grid_Density_{survey.modality}_{self._survey_source()}",
                "analytics_layers", display_name=name,
                base_folder=survey.folder)
            layer.setCustomProperty("bambi_layer_type", "grid_density")
            self._style_grid_layer(layer, limits)
            self._add_layer_to_survey(layer)
            self.update_status("add_grid", "🟢 Added")
            self.iface.mapCanvas().refresh()
            summary = document.get("summary") or {}
            self.log(f"Added the grid: {len(features)} cell(s), "
                     f"{summary.get('counted', 0)} counted. The layer's "
                     f"style menu has '{STYLE_DENSITY}' and "
                     f"'{STYLE_HOT_SPOTS}'.")
        except Exception as e:
            self.update_status("add_grid", "🔴 Error")
            self.log(f"Error adding the grid: {e}")
            QMessageBox.critical(self, "Error",
                                 f"Failed to add the grid: {e}")

    @staticmethod
    def _grid_categories(attribute: str, classes) -> object:
        """A renderer that draws each class of *attribute* its own way;
        *classes* are ``(value, fill properties)``."""
        categories = [
            QgsRendererCategory(value, QgsFillSymbol.createSimple(properties),
                                value)
            for value, properties in classes]
        return QgsCategorizedSymbolRenderer(attribute, categories)

    def _style_grid_layer(self, layer, limits) -> None:
        """Two styles: the cells by density (the one that is shown), and by
        what stands out (Getis-Ord Gi*)."""
        outline = {"outline_color": "#6e6e6e", "outline_width": "0.15"}
        # the two kinds of cell that carry no density
        not_counted = [
            ("too little seen", {
                "style": "b_diagonal", "color": "#8c8c8c",
                "outline_color": "#8c8c8c", "outline_width": "0.2",
                "outline_style": "dash"}),
            ("not seen", {
                "style": "no", "outline_color": "#bdbdbd",
                "outline_width": "0.15", "outline_style": "dot"}),
        ]

        spots = [(spot, dict(outline, color=colour))
                 for spot, colour in _SPOT_COLOURS.items()]
        spots.append((NOTHING_STANDS_OUT, dict(outline, color="#f7f7f7")))
        layer.setRenderer(self._grid_categories(SPOT_CLASS,
                                                spots + not_counted))
        layer.styleManager().addStyleFromLayer(STYLE_HOT_SPOTS)

        labels = density_classes(limits)
        first = len(_DENSITY_COLOURS) - len(labels)
        by_density = [(NO_ANIMALS, dict(outline, color="#f7f7f7"))]
        for index, label in enumerate(labels):
            colour = _DENSITY_COLOURS[
                min(len(_DENSITY_COLOURS) - 1, max(0, first + index))]
            by_density.append((label, dict(outline, color=colour)))
        layer.setRenderer(self._grid_categories(DENSITY_CLASS,
                                                by_density + not_counted))

        settings = QgsPalLayerSettings()
        settings.fieldName = (
            "CASE WHEN \"animals\" > 0 THEN \"animals\" END")
        settings.isExpression = True
        text = QgsTextFormat()
        font = QFont()
        font.setBold(True)
        text.setFont(font)
        text.setSize(9)
        text.setColor(QColor("#202020"))
        halo = QgsTextBufferSettings()
        halo.setEnabled(True)
        halo.setSize(0.8)
        halo.setColor(QColor("#ffffff"))
        text.setBuffer(halo)
        settings.setFormat(text)
        layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
        layer.setLabelsEnabled(True)
        layer.styleManager().addStyleFromLayer(STYLE_DENSITY)
        layer.styleManager().setCurrentStyle(STYLE_DENSITY)
        layer.triggerRepaint()
