# -*- coding: utf-8 -*-
"""Survey Analytics > Position Uncertainty: how far off is a place?

The group on the Survey Analytics tab - how well the camera's position and
angle, the terrain model, the pixel and the timing are taken to be known,
the button that works the uncertainty out (``core.position_uncertainty``),
the table of what it came to (``bambi_position_dialog``) - and the
uncertainty on the map: a circle around every animal within which its true
place almost certainly lies.
"""

import json
import math
import os
from dataclasses import asdict

from qgis.PyQt.QtWidgets import (
    QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QVBoxLayout,
)
from qgis.core import (
    QgsFeature, QgsField, QgsFillSymbol, QgsGeometry, QgsPointXY,
    QgsSingleSymbolRenderer, QgsVectorLayer,
)

from ..core import position_uncertainty, store as _store_kinds
from ..gui_utils import PLUGIN_SCOPE
from ..qgis_compat import FieldType

#: Where the accuracies are kept in the QGIS project.
ENTRY_POSITION = "Survey/Position"
CUSTOM = "Custom"

#: The spin boxes: ``(setting, widget, label, unit, decimals, maximum,
#: step, tooltip)``.
_SPINS = (
    ("gps_horizontal_m", "position_gps_spin", "Camera position", " m", 2,
     100.0, 0.1,
     "How well the drone knows where it is over the ground. About 1.5 m "
     "from satellites alone, a few centimetres with RTK."),
    ("gps_vertical_m", "position_height_spin", "Camera height", " m", 2,
     100.0, 0.1,
     "How well the drone knows how high it is. It only matters for what "
     "is seen to the side."),
    ("camera_angle_deg", "position_angle_spin", "Camera angle", " °", 2,
     45.0, 0.1,
     "How well the direction the camera looked in is known - gimbal, "
     "compass and what is left after the Correction Wizard. A degree is "
     "1.4 m on the ground from 80 m up, and more to the side."),
    ("terrain_m", "position_terrain_spin", "Terrain model", " m", 2, 100.0,
     0.1,
     "How far the terrain model may be off in height where the animal "
     "stood - and how far above it the warm spot was."),
    ("pixel_px", "position_pixel_spin", "Pixel", " px", 1, 100.0, 0.5,
     "How well the middle of a box marks the animal, in pixels."),
    ("sync_s", "position_sync_spin", "Timing", " s", 2, 10.0, 0.05,
     "How far a frame and the camera position given for it may be apart "
     "in time. The drone covers its speed times this in between."),
)
_CIRCLE_POINTS = 48


class PositionMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _build_position_group(self) -> QGroupBox:
        """The Position Uncertainty group of the Survey Analytics tab."""
        group = QGroupBox("Position Uncertainty")
        layout = QVBoxLayout(group)
        description = QLabel(
            "How far off may the place of an animal be? For every "
            "detection of the ticked flights, from how high the camera was "
            "above it, how far to the side it looked and the slope the "
            "animal stood on - and from how well the things below are "
            "known. The result also says which of them matters most.")
        description.setWordWrap(True)
        description.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(description)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Known to (one sigma):"))
        self.position_preset_combo = QComboBox()
        for name in position_uncertainty.PRESETS:
            self.position_preset_combo.addItem(name)
        self.position_preset_combo.addItem(CUSTOM)
        self.position_preset_combo.setToolTip(
            "Ready-made accuracies for a drone that knows its place from "
            "satellites alone and for one with RTK; any value below can be "
            "changed.")
        self.position_preset_combo.currentTextChanged.connect(
            self._on_position_preset)
        preset_row.addWidget(self.position_preset_combo)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        grid = QGridLayout()
        defaults = asdict(position_uncertainty.Accuracy())
        for index, (key, attribute, label, unit, decimals, maximum, step,
                    tip) in enumerate(_SPINS):
            spin = QDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setRange(0.0, maximum)
            spin.setSingleStep(step)
            spin.setSuffix(unit)
            spin.setValue(defaults[key])
            spin.setToolTip(tip)
            spin.valueChanged.connect(self._on_position_value_changed)
            setattr(self, attribute, spin)
            row, column = divmod(index, 2)
            grid.addWidget(QLabel(label + ":"), row, 2 * column)
            grid.addWidget(spin, row, 2 * column + 1)
        grid.setColumnStretch(4, 1)
        layout.addLayout(grid)

        run_row = QHBoxLayout()
        self.position_btn = QPushButton("→ Estimate Position Uncertainty")
        self.position_btn.clicked.connect(self.run_position_uncertainty)
        self.position_btn.setToolTip(
            "Write analytics_{camera}/position_uncertainty_{source}.json, "
            "_boxes.csv and _tracks.csv into the survey folder and show "
            "the result. Needs geo-referenced detections and the flight's "
            "DEM.")
        self.position_status = QLabel("⚪")
        run_row.addWidget(self.position_btn)
        run_row.addWidget(self.position_status)
        layout.addLayout(run_row)

        show_row = QHBoxLayout()
        self.position_show_btn = QPushButton("→ Show Position Uncertainty")
        self.position_show_btn.clicked.connect(self.show_position_uncertainty)
        show_row.addWidget(self.position_show_btn)
        show_row.addStretch()
        layout.addLayout(show_row)

        add_row = QHBoxLayout()
        self.add_position_btn = QPushButton("→ Add Uncertainty to QGIS")
        self.add_position_btn.clicked.connect(self.add_position_to_qgis)
        self.add_position_btn.setToolTip(
            "A circle around every counted animal within which its true "
            "place almost certainly lies (twice the typical error: 95 to "
            "98 in a hundred).")
        self.add_position_status = QLabel("⚪")
        add_row.addWidget(self.add_position_btn)
        add_row.addWidget(self.add_position_status)
        layout.addLayout(add_row)
        return group

    # ------------------------------------------------------------------ #
    # The accuracies
    # ------------------------------------------------------------------ #

    def _position_accuracy(self) -> dict:
        """The accuracies as they stand in the form, by setting name."""
        return {key: getattr(self, attribute).value()
                for key, attribute, *_rest in _SPINS}

    def _position_values(self) -> dict:
        """The accuracies as the step reads them (``position_*``)."""
        if not hasattr(self, "position_gps_spin"):
            return {}
        return {f"position_{key}": value
                for key, value in self._position_accuracy().items()}

    def _set_position_values(self, values: dict) -> None:
        defaults = asdict(position_uncertainty.Accuracy())
        self._position_filling = True
        try:
            for key, attribute, *_rest in _SPINS:
                value = (values or {}).get(f"position_{key}")
                try:
                    value = defaults[key] if value is None else float(value)
                except (TypeError, ValueError):
                    value = defaults[key]
                getattr(self, attribute).setValue(value)
        finally:
            self._position_filling = False
        self._show_position_preset()

    def _show_position_preset(self) -> None:
        """Name the preset the values are - or Custom."""
        current = self._position_accuracy()
        name = CUSTOM
        for preset, accuracy in position_uncertainty.PRESETS.items():
            if all(math.isclose(current[key], value, abs_tol=1e-9)
                   for key, value in asdict(accuracy).items()):
                name = preset
                break
        self.position_preset_combo.blockSignals(True)
        self.position_preset_combo.setCurrentText(name)
        self.position_preset_combo.blockSignals(False)

    def _on_position_preset(self, name: str) -> None:
        accuracy = position_uncertainty.PRESETS.get(name)
        if accuracy is None:
            return                      # Custom: the values stay
        self._set_position_values(
            {f"position_{key}": value
             for key, value in asdict(accuracy).items()})
        self._on_position_settings_changed()

    def _on_position_value_changed(self, *_args) -> None:
        if getattr(self, "_position_filling", False):
            return
        self._show_position_preset()
        self._on_position_settings_changed()

    def _on_position_settings_changed(self) -> None:
        """An accuracy was changed by the user: it is kept with the
        project, like the survey's flights."""
        if getattr(self, "_loading_config", False):
            return
        self.save_config_to_project()

    def _save_position_settings(self, project) -> None:
        project.writeEntry(PLUGIN_SCOPE, ENTRY_POSITION,
                           json.dumps(self._position_values()))

    def _load_position_settings(self, project) -> None:
        raw, ok = project.readEntry(PLUGIN_SCOPE, ENTRY_POSITION, "")
        try:
            values = json.loads(raw) if ok and raw else {}
        except ValueError:
            values = {}
        self._set_position_values(values if isinstance(values, dict) else {})

    def _reset_position_settings(self) -> None:
        self._set_position_values({})

    # ------------------------------------------------------------------ #
    # Running it, and its result
    # ------------------------------------------------------------------ #

    def _position_result_path(self, survey=None) -> str:
        survey = survey or self.current_survey()
        return os.path.join(
            survey.analytics_folder,
            f"position_uncertainty_{self._survey_source()}.json")

    def run_position_uncertainty(self):
        """Work out the position uncertainty of what the survey counts
        (core.position_uncertainty)."""
        survey = self._checked_survey()
        if survey is None:
            return
        if self._survey_source() == "tracks":
            checks = self._tracks_check(survey)
        else:
            def checks(flight):
                if self.has_stage(_store_kinds.DETECTIONS, survey.modality,
                                  flight.target_folder):
                    return []
                return [f"{survey.camera_label} detections (run 'Detect "
                        "Animals' first)"]
        if self._survey_prerequisites(survey, checks):
            self.start_worker("position_uncertainty")

    def _read_position(self, path: str):
        """The stored result, or ``None`` after saying why not."""
        if not os.path.isfile(path):
            QMessageBox.warning(
                self, "Missing Data",
                "The position uncertainty has not been worked out yet.\n"
                "Run 'Estimate Position Uncertainty' first.")
            return None
        try:
            return position_uncertainty.read_json(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Position Uncertainty",
                                 f"Could not read {path}: {exc}")
            return None

    def show_position_uncertainty(self):
        """Show what the last run came to."""
        from ..bambi_position_dialog import PositionDialog

        path = self._position_result_path()
        document = self._read_position(path)
        if document is None:
            return
        dialog = PositionDialog(document, json_path=path, parent=self)
        dialog.setModal(False)
        dialog.show()
        self._position_dialog = dialog

    # ------------------------------------------------------------------ #
    # The uncertainty on the map
    # ------------------------------------------------------------------ #

    def add_position_to_qgis(self):
        """Add a circle around every animal: where its true place almost
        certainly is."""
        survey = self.current_survey()
        document = self._read_position(self._position_result_path(survey))
        if document is None:
            return
        animals = [row for row in document.get("tracks") or []
                   if None not in (row.get("x"), row.get("y"),
                                   row.get("error95_m"))]
        if not animals:
            QMessageBox.warning(
                self, "Position Uncertainty",
                "The circles are drawn around animals, and the survey "
                "counts boxes - or none of its animals has a place.\n"
                "Set the survey to count animals and run 'Estimate "
                "Position Uncertainty' again.")
            return

        name = f"BAMBI Position Uncertainty ({survey.camera_label})"
        try:
            self.update_status("add_position", "🟡 Loading...")
            epsg = survey.epsg or self.get_config().get("target_epsg", 32633)
            layer = QgsVectorLayer(f"Polygon?crs=EPSG:{epsg}", name,
                                   "memory")
            provider = layer.dataProvider()
            provider.addAttributes([
                QgsField("flight", FieldType.String),
                QgsField("track_id", FieldType.LongLong),
                QgsField("species", FieldType.String),
                QgsField("boxes", FieldType.LongLong),
                QgsField("error_m", FieldType.Double),
                QgsField("error95_m", FieldType.Double),
                QgsField("best_error_m", FieldType.Double),
                QgsField("main_source", FieldType.String),
            ])
            layer.updateFields()

            features = []
            for row in animals:
                x, y = float(row["x"]), float(row["y"])
                radius = float(row["error95_m"])
                ring = [QgsPointXY(
                    x + radius * math.cos(2 * math.pi * n / _CIRCLE_POINTS),
                    y + radius * math.sin(2 * math.pi * n / _CIRCLE_POINTS))
                    for n in range(_CIRCLE_POINTS)]
                feature = QgsFeature(layer.fields())
                feature.setGeometry(QgsGeometry.fromPolygonXY([ring]))
                feature.setAttributes([
                    row.get("flight"), row.get("track_id"),
                    row.get("species"), row.get("boxes"), row.get("error_m"),
                    row.get("error95_m"), row.get("best_error_m"),
                    row.get("main_source")])
                features.append(feature)
            provider.addFeatures(features)
            layer.updateExtents()

            layer = self._persist_memory_layer(
                layer,
                f"Position_Uncertainty_{survey.modality}",
                "analytics_layers", display_name=name,
                base_folder=survey.folder)
            layer.setCustomProperty("bambi_layer_type",
                                    "position_uncertainty")
            layer.setRenderer(QgsSingleSymbolRenderer(
                QgsFillSymbol.createSimple({
                    "color": "230,85,13,35", "outline_color": "#e6550d",
                    "outline_width": "0.3"})))
            self._add_layer_to_survey(layer)
            self.update_status("add_position", "🟢 Added")
            self.iface.mapCanvas().refresh()
            summary = document.get("summary") or {}
            typical = summary.get("median_m")
            note = "" if typical is None else (
                f" The detections are typically {typical:.1f} m off.")
            self.log(f"Added the uncertainty of {len(features)} animal(s): "
                     "each circle holds the true place 95 to 98 times in a "
                     f"hundred.{note}")
        except Exception as e:
            self.update_status("add_position", "🔴 Error")
            self.log(f"Error adding the position uncertainty: {e}")
            QMessageBox.critical(
                self, "Error",
                f"Failed to add the position uncertainty: {e}")
