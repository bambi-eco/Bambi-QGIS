# -*- coding: utf-8 -*-
"""Survey Area panel of the Survey Analytics tab.

Lets the user define the census plot and its exclusion zones in QGIS - drawn
with the ordinary digitising tools or copied from an existing polygon layer -
and switches the survey analytics to work only inside that area.

The polygons belong to the survey, not to one flight: they are stored in
``<survey folder>/survey_area/survey_area.gpkg`` (see
:mod:`core.survey_area`), which the analytics read without QGIS. This module
only edits that file through QGIS layers and shows its status.
"""

import os

from qgis.PyQt.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout,
)
from qgis.core import (
    QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsFeature,
    QgsFillSymbol, QgsGeometry, QgsProject, QgsSingleSymbolRenderer,
    QgsVectorLayer, QgsWkbTypes,
)

from .core import survey_area

#: Layer names in the QGIS layer tree, per GeoPackage table.
LAYER_TITLES = {
    survey_area.PLOT_LAYER: "Survey Plot",
    survey_area.EXCLUSION_LAYER: "Survey Exclusion Zones",
}

_STYLES = {
    survey_area.PLOT_LAYER: {
        "color": "46,160,67,35", "outline_color": "46,160,67",
        "outline_width": "0.8", "style": "solid"},
    survey_area.EXCLUSION_LAYER: {
        "color": "215,25,28,110", "outline_color": "215,25,28",
        "outline_width": "0.5", "style": "b_diagonal"},
}


_Buttons = QMessageBox.StandardButton
_SAVE_IGNORE_CANCEL = _Buttons.Save | _Buttons.Ignore | _Buttons.Cancel
_DialogButtons = QDialogButtonBox.StandardButton
_OK_CANCEL = _DialogButtons.Ok | _DialogButtons.Cancel


class SurveyAreaPanel(QGroupBox):
    """The "Survey Area" group box. *dock* is the BAMBI dock widget."""

    def __init__(self, dock):
        super().__init__("Survey Area")
        self.dock = dock
        self._had_area = False

        layout = QVBoxLayout(self)
        desc = QLabel(
            "Optional census plot with exclusion zones (e.g. thick forest "
            "cover). When applied, the density heatmap, coverage map, "
            "distance sampling and population estimation count only animals "
            "inside the plot and outside the exclusions, and use only that "
            "ground as the surveyed area. Stored in the survey folder "
            "(survey_area/survey_area.gpkg) - draw or edit the polygons with "
            "the normal QGIS editing tools.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(desc)

        self.enabled_check = QCheckBox("Limit analytics to the survey area")
        self.enabled_check.setToolTip(
            "Off: the analytics cover the whole flight, as before.\n"
            "On: only the plot minus its exclusion zones is analysed, in "
            "every flight of the survey.")
        layout.addWidget(self.enabled_check)

        buttons = QHBoxLayout()
        self.draw_plot_btn = QPushButton("Draw Plot")
        self.draw_plot_btn.setToolTip(
            "Add the Survey Plot layer, switch it to editing and start the "
            "polygon tool. Click the vertices, right-click to finish, then "
            "save the layer edits.")
        self.draw_plot_btn.clicked.connect(
            lambda: self.draw(survey_area.PLOT_LAYER))
        self.draw_exclusion_btn = QPushButton("Draw Exclusion Zone")
        self.draw_exclusion_btn.setToolTip(
            "Like 'Draw Plot', for a zone to cut out of the plot. Give it a "
            "reason in the attribute form (e.g. 'thick forest cover').")
        self.draw_exclusion_btn.clicked.connect(
            lambda: self.draw(survey_area.EXCLUSION_LAYER))
        self.import_btn = QPushButton("Import…")
        self.import_btn.setToolTip(
            "Copy polygons from a layer already in the project - a shapefile "
            "of the hunting district, a KML, a GeoJSON - into the plot or the "
            "exclusion zones.")
        self.import_btn.clicked.connect(self.import_from_layer)
        self.show_btn = QPushButton("Add to QGIS")
        self.show_btn.setToolTip("Load both survey-area layers into the map.")
        self.show_btn.clicked.connect(self.show_layers)
        for button in (self.draw_plot_btn, self.draw_exclusion_btn,
                       self.import_btn, self.show_btn):
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status_label)
        # Refreshed by the dock once the whole UI exists (it reads the
        # target folder and CRS widgets of other tabs).

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #

    def _target(self):
        """``(survey folder, epsg)`` or ``(None, None)`` when not set up."""
        panel = getattr(self.dock, "survey_panel", None)
        if panel is None:
            return None, None
        folder = panel.folder()
        if not folder:
            return None, None
        return folder, panel.epsg()

    def is_enabled(self) -> bool:
        return self.enabled_check.isEnabled() and self.enabled_check.isChecked()

    def refresh(self):
        """Re-read the file and update the status line and the checkbox."""
        folder, epsg = self._target()
        if folder is None:
            self._set_status("Set the survey folder to define a survey area.",
                             has_area=False)
            return
        try:
            area = survey_area.load(folder, epsg)
        except Exception as exc:  # unreadable file, missing shapely, ...
            self._set_status(f"Survey area cannot be read: {exc}",
                             has_area=False)
            return
        if area is None:
            self._set_status(
                "No survey area defined - the analytics cover the whole "
                "flights.", has_area=False)
            return
        self._set_status(area.summary(), has_area=True)

    def _set_status(self, text: str, has_area: bool):
        self.status_label.setText(text)
        self.enabled_check.setEnabled(has_area)
        if has_area and not self._had_area:
            # A newly drawn area is meant to be used.
            self.enabled_check.setChecked(True)
        elif not has_area:
            self.enabled_check.setChecked(False)
        self._had_area = has_area

    # ------------------------------------------------------------------ #
    # Layers
    # ------------------------------------------------------------------ #

    def _ensure_file(self):
        folder, epsg = self._target()
        if folder is None:
            QMessageBox.warning(self, "Survey Area",
                                "Please set the survey folder first.")
            return None
        if not epsg:
            QMessageBox.warning(
                self, "Survey Area",
                "The survey has no target CRS yet - tick a processed flight "
                "or set the target CRS first.")
            return None
        return survey_area.create(folder, int(epsg))

    @staticmethod
    def _source(file_path: str, table: str) -> str:
        return f"{file_path}|layername={table}"

    def _find_layer(self, file_path: str, table: str):
        wanted = os.path.normcase(os.path.abspath(file_path))
        for layer in QgsProject.instance().mapLayers().values():
            if not isinstance(layer, QgsVectorLayer):
                continue
            source = layer.source()
            path_part, _sep, rest = source.partition("|")
            if os.path.normcase(os.path.abspath(path_part)) != wanted:
                continue
            if f"layername={table}" in rest:
                return layer
        return None

    def _layer(self, table: str, create: bool = True):
        """The QGIS layer for *table*, loaded into the flight group if needed."""
        file_path = self._ensure_file() if create else None
        if file_path is None:
            folder, _epsg = self._target()
            if folder is None:
                return None
            file_path = survey_area.path(folder)
            if not os.path.isfile(file_path):
                return None
        layer = self._find_layer(file_path, table)
        if layer is not None:
            return layer
        layer = QgsVectorLayer(self._source(file_path, table),
                               LAYER_TITLES[table], "ogr")
        if not layer.isValid():
            self.dock.log(f"Could not open {file_path} ({table}).")
            return None
        layer.setRenderer(QgsSingleSymbolRenderer(
            QgsFillSymbol.createSimple(_STYLES[table])))
        layer.afterCommitChanges.connect(self.refresh)
        layer.editingStopped.connect(self.refresh)
        self.dock._add_layer_to_survey(layer)
        return layer

    def show_layers(self):
        for table in survey_area.LAYERS:
            self._layer(table)
        self.refresh()

    def draw(self, table: str):
        """Start digitising a polygon into *table*."""
        layer = self._layer(table)
        if layer is None:
            return
        iface = self.dock.iface
        iface.setActiveLayer(layer)
        if not layer.isEditable():
            layer.startEditing()
        iface.actionAddFeature().trigger()
        what = ("survey plot" if table == survey_area.PLOT_LAYER
                else "exclusion zone")
        self.dock.log(
            f"Drawing the {what}: click the corners on the map, right-click to "
            "finish, fill in the name (and reason) and then 'Save Layer "
            "Edits'. The analytics read the saved polygons.")

    def flush_edits(self) -> bool:
        """Ask to save unsaved survey-area edits before an analytic runs.

        Returns False when the user cancels the run.
        """
        folder, _epsg = self._target()
        if folder is None:
            return True
        file_path = survey_area.path(folder)
        for table in survey_area.LAYERS:
            layer = self._find_layer(file_path, table)
            if layer is None or not layer.isEditable() or \
                    not layer.isModified():
                continue
            answer = QMessageBox.question(
                self, "Unsaved Survey Area",
                f"'{layer.name()}' has unsaved edits. The analytics only "
                "see saved polygons.\n\nSave them now?",
                _SAVE_IGNORE_CANCEL)
            if answer == QMessageBox.StandardButton.Cancel:
                return False
            if answer == QMessageBox.StandardButton.Save:
                if not layer.commitChanges(False):
                    QMessageBox.warning(
                        self, "Survey Area",
                        "Saving failed:\n" + "\n".join(layer.commitErrors()))
                    return False
        self.refresh()
        return True

    # ------------------------------------------------------------------ #
    # Import
    # ------------------------------------------------------------------ #

    def import_from_layer(self):
        """Copy polygons of another project layer into the survey area."""
        candidates = [
            layer for layer in QgsProject.instance().mapLayers().values()
            if self._importable(layer)]
        if not candidates:
            QMessageBox.information(
                self, "Import Survey Area",
                "No polygon layer in the project to import from. Add the "
                "shapefile / GeoJSON / KML of the area to QGIS first.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Import Survey Area")
        form = QFormLayout(dialog)
        layer_combo = QComboBox()
        for layer in candidates:
            layer_combo.addItem(layer.name(), layer.id())
        active = self.dock.iface.activeLayer()
        if active is not None:
            index = layer_combo.findData(active.id())
            if index >= 0:
                layer_combo.setCurrentIndex(index)
        form.addRow("From layer:", layer_combo)
        role_combo = QComboBox()
        role_combo.addItem("Survey plot", survey_area.PLOT_LAYER)
        role_combo.addItem("Exclusion zones", survey_area.EXCLUSION_LAYER)
        form.addRow("Import as:", role_combo)
        selected_check = QCheckBox("Selected features only")
        form.addRow("", selected_check)
        replace_check = QCheckBox("Replace the existing polygons")
        replace_check.setChecked(True)
        form.addRow("", replace_check)
        box = QDialogButtonBox(_OK_CANCEL)
        box.accepted.connect(dialog.accept)
        box.rejected.connect(dialog.reject)
        form.addRow(box)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        source = QgsProject.instance().mapLayer(layer_combo.currentData())
        table = role_combo.currentData()
        try:
            count = self._copy_features(
                source, table, selected_check.isChecked(),
                replace_check.isChecked())
        except RuntimeError as exc:
            QMessageBox.warning(self, "Import Survey Area", str(exc))
            return
        self.dock.log(f"Imported {count} polygon(s) from '{source.name()}' "
                      f"into the {LAYER_TITLES[table]}.")
        self.refresh()

    def _importable(self, layer) -> bool:
        """A polygon layer that is not one of the survey-area layers."""
        if not isinstance(layer, QgsVectorLayer):
            return False
        if layer.geometryType() != QgsWkbTypes.GeometryType.PolygonGeometry:
            return False
        return not self._table_of(layer)

    @staticmethod
    def _table_of(layer) -> str:
        _path, _sep, rest = layer.source().partition("|")
        if os.path.basename(_path) != survey_area.FILENAME:
            return ""
        for table in survey_area.LAYERS:
            if f"layername={table}" in rest:
                return table
        return ""

    def _copy_features(self, source, table, selected_only, replace) -> int:
        target = self._layer(table)
        if target is None:
            raise RuntimeError("The survey-area file could not be opened.")
        if target.isEditable() and target.isModified():
            raise RuntimeError(
                f"'{target.name()}' has unsaved edits. Save or discard them "
                "first.")

        _folder, epsg = self._target()
        transform = QgsCoordinateTransform(
            source.crs(), QgsCoordinateReferenceSystem(f"EPSG:{int(epsg)}"),
            QgsProject.instance())
        features = (source.selectedFeatures() if selected_only
                    else list(source.getFeatures()))
        if not features:
            raise RuntimeError("The layer has no (selected) features.")

        names = {field.name().lower(): field.name()
                 for field in source.fields()}
        name_field = next((names[key] for key in ("name", "label", "title")
                           if key in names), None)
        reason_field = names.get("reason")

        new_features = []
        for feature in features:
            geometry = QgsGeometry(feature.geometry())
            if geometry.isEmpty():
                continue
            geometry.transform(transform)
            geometry.convertToMultiType()
            out = QgsFeature(target.fields())
            out.setGeometry(geometry)
            value = feature[name_field] if name_field else None
            out["name"] = source.name() if value is None else str(value)
            if reason_field and feature[reason_field] is not None:
                out["reason"] = str(feature[reason_field])
            new_features.append(out)
        if not new_features:
            raise RuntimeError("None of the features has a polygon.")

        was_editing = target.isEditable()
        if was_editing:
            target.rollBack()
        provider = target.dataProvider()
        if replace:
            ids = [f.id() for f in target.getFeatures()]
            if ids and not provider.deleteFeatures(ids):
                raise RuntimeError("Could not remove the existing polygons.")
        ok, _added = provider.addFeatures(new_features)
        if not ok:
            raise RuntimeError("Could not write the polygons.")
        target.updateExtents()
        target.triggerRepaint()
        if was_editing:
            target.startEditing()
        return len(new_features)
