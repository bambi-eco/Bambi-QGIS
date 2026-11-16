# -*- coding: utf-8 -*-
"""Survey Analytics > Reaction to the Drone: did the animals move, and
where to?

The group on the Survey Analytics tab - how far an animal has to have gone
to count as having moved, the button that reads the tracks
(``core.drone_reaction``), the table of what it found
(``bambi_reaction_dialog``) - and the movements on the map: an arrow per
animal that moved, coloured by where it went relative to the line the drone
flew.
"""

import json
import os

from qgis.PyQt.QtWidgets import (
    QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QVBoxLayout,
)
from qgis.core import (
    QgsArrowSymbolLayer, QgsCategorizedSymbolRenderer, QgsFeature, QgsField,
    QgsFillSymbol, QgsGeometry, QgsLineSymbol, QgsPointXY,
    QgsRendererCategory, QgsVectorLayer,
)

from ..core import drone_reaction
from ..gui_utils import PLUGIN_SCOPE
from ..qgis_compat import FieldType

#: What the arrows are drawn by, and in which colour: away from the line
#: red, towards it blue, along it grey - and dark where the line is not
#: known.
DIRECTION_COLOURS = (
    (drone_reaction.AWAY, "#d7301f"),
    (drone_reaction.TOWARDS, "#2b8cbe"),
    (drone_reaction.AHEAD, "#8c8c8c"),
    (drone_reaction.BACK, "#8c8c8c"),
    ("", "#404040"),
)
_TEXT_COLUMNS = ("flight", "species", "sex", "age", "status", "direction")
_WHOLE_COLUMNS = ("track_id", "n_positions")


#: Where the two thresholds are kept in the QGIS project.
ENTRY_REACTION = "Survey/Reaction"


class ReactionMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _build_reaction_group(self) -> QGroupBox:
        """The Reaction to the Drone group of the Survey Analytics tab."""
        group = QGroupBox("Reaction to the Drone")
        layout = QVBoxLayout(group)
        description = QLabel(
            "Did the animals flee? From every track of the ticked flights: "
            "how far the animal went while it was in view, how fast, and "
            "where to relative to the line the drone flew. Then all "
            "animals together - how many moved, whether more of them away "
            "from the line than towards it, and whether the ones nearer "
            "the line were the faster. Animals that move away from the "
            "line make a distance-sampling density too low.")
        description.setWordWrap(True)
        description.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(description)

        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Moved = went at least"))
        self.reaction_min_spin = QDoubleSpinBox()
        self.reaction_min_spin.setRange(0.5, 500.0)
        self.reaction_min_spin.setDecimals(1)
        self.reaction_min_spin.setSingleStep(0.5)
        self.reaction_min_spin.setValue(drone_reaction.MIN_DISPLACEMENT_M)
        self.reaction_min_spin.setSuffix(" m")
        self.reaction_min_spin.setToolTip(
            "How far an animal has to have gone while it was in view to "
            "count as having moved. Positions jitter by a metre or two, "
            "and a terrain model that is a little off makes a standing "
            "animal seem to slide along the line: below a few metres "
            "nobody can tell. The distance also has to be three times its "
            "own standard error.")
        params_row.addWidget(self.reaction_min_spin)
        params_row.addWidget(QLabel("at"))
        self.reaction_speed_spin = QDoubleSpinBox()
        self.reaction_speed_spin.setRange(0.0, 20.0)
        self.reaction_speed_spin.setDecimals(2)
        self.reaction_speed_spin.setSingleStep(0.05)
        self.reaction_speed_spin.setValue(drone_reaction.MIN_SPEED_MS)
        self.reaction_speed_spin.setSuffix(" m/s or faster")
        self.reaction_speed_spin.setSpecialValueText("any speed")
        self.reaction_speed_spin.setToolTip(
            "How fast an animal has to have gone, at the least, to count "
            "as having moved - the speed of the constant velocity fitted "
            "through its track. Over a long track an animal that drifts "
            "covers the distance too: 3 m in a minute is 0.05 m/s. A "
            "grazing animal ambles at 0.1 to 0.3 m/s, a walking one does "
            "about 1 m/s. At nought the distance alone decides.")
        params_row.addWidget(self.reaction_speed_spin)
        params_row.addStretch()
        layout.addLayout(params_row)

        run_row = QHBoxLayout()
        self.reaction_btn = QPushButton("→ Read Reaction to the Drone")
        self.reaction_btn.clicked.connect(self.run_drone_reaction)
        self.reaction_btn.setToolTip(
            "Write analytics_{camera}/drone_reaction.json and "
            "drone_reaction_tracks.csv into the survey folder and show the "
            "result. Needs the tracks of every flight, and the flight's "
            "DEM for the line the drone flew.")
        self.reaction_status = QLabel("⚪")
        run_row.addWidget(self.reaction_btn)
        run_row.addWidget(self.reaction_status)
        layout.addLayout(run_row)

        show_row = QHBoxLayout()
        self.reaction_show_btn = QPushButton("→ Show Reaction")
        self.reaction_show_btn.clicked.connect(self.show_reaction)
        show_row.addWidget(self.reaction_show_btn)
        show_row.addStretch()
        layout.addLayout(show_row)

        add_row = QHBoxLayout()
        self.add_reaction_btn = QPushButton("→ Add Movements to QGIS")
        self.add_reaction_btn.clicked.connect(self.add_reaction_to_qgis)
        self.add_reaction_btn.setToolTip(
            "An arrow per animal that moved, from where it was first seen "
            "to where it was last seen: red away from the line the drone "
            "flew, blue towards it, grey along it.")
        self.add_reaction_status = QLabel("⚪")
        add_row.addWidget(self.add_reaction_btn)
        add_row.addWidget(self.add_reaction_status)
        layout.addLayout(add_row)
        return group

    # ------------------------------------------------------------------ #
    # Running it, and its result
    # ------------------------------------------------------------------ #

    def _reaction_result_path(self, survey=None) -> str:
        survey = survey or self.current_survey()
        return os.path.join(survey.analytics_folder, "drone_reaction.json")

    def run_drone_reaction(self):
        """Read the reaction to the drone from the survey's tracks
        (core.drone_reaction)."""
        survey = self._checked_survey()
        if survey is None:
            return
        if self._survey_prerequisites(survey, self._tracks_check(survey)):
            self.start_worker("drone_reaction")

    def _read_reaction(self, path: str):
        """The stored result, or ``None`` after saying why not."""
        if not os.path.isfile(path):
            QMessageBox.warning(
                self, "Missing Data",
                "The reaction to the drone has not been read yet.\n"
                "Run 'Read Reaction to the Drone' first.")
            return None
        try:
            return drone_reaction.read_json(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Reaction to the Drone",
                                 f"Could not read {path}: {exc}")
            return None

    def show_reaction(self):
        """Show what the last reading of the reaction found."""
        from ..bambi_reaction_dialog import ReactionDialog

        path = self._reaction_result_path()
        document = self._read_reaction(path)
        if document is None:
            return
        dialog = ReactionDialog(document, json_path=path, parent=self)
        dialog.setModal(False)
        dialog.show()
        self._reaction_dialog = dialog

    # ------------------------------------------------------------------ #
    # The movements on the map
    # ------------------------------------------------------------------ #

    def add_reaction_to_qgis(self):
        """Add an arrow per animal that moved."""
        survey = self.current_survey()
        document = self._read_reaction(self._reaction_result_path(survey))
        if document is None:
            return

        name = f"BAMBI Movements ({survey.camera_label})"
        try:
            self.update_status("add_reaction", "🟡 Loading...")
            epsg = survey.epsg or self.get_config().get("target_epsg", 32633)
            layer = QgsVectorLayer(f"LineString?crs=EPSG:{epsg}", name,
                                   "memory")
            provider = layer.dataProvider()
            columns = list(drone_reaction.TRACK_COLUMNS)

            def field(column):
                if column in _TEXT_COLUMNS:
                    return QgsField(column, FieldType.String)
                if column in _WHOLE_COLUMNS:
                    return QgsField(column, FieldType.LongLong)
                return QgsField(column, FieldType.Double)

            provider.addAttributes([field(column) for column in columns])
            layer.updateFields()

            features = []
            for row in document.get("tracks", []):
                if row.get("status") != drone_reaction.MOVED:
                    continue
                ends = [row.get(key) for key in
                        ("start_x", "start_y", "end_x", "end_y")]
                if None in ends:
                    continue
                feature = QgsFeature(layer.fields())
                feature.setGeometry(QgsGeometry.fromPolylineXY([
                    QgsPointXY(float(ends[0]), float(ends[1])),
                    QgsPointXY(float(ends[2]), float(ends[3]))]))
                values = [row.get(column) for column in columns]
                # the layer is drawn by the direction: none is a class too
                position = columns.index("direction")
                values[position] = values[position] or ""
                feature.setAttributes(values)
                features.append(feature)
            provider.addFeatures(features)
            layer.updateExtents()

            layer = self._persist_memory_layer(
                layer, f"Movements_{survey.modality}", "analytics_layers",
                display_name=name, base_folder=survey.folder)
            layer.setCustomProperty("bambi_layer_type", "drone_reaction")
            self._style_reaction_layer(layer)
            self._add_layer_to_survey(layer)
            self.update_status("add_reaction", "🟢 Added")
            self.iface.mapCanvas().refresh()
            everything = (document.get("summary") or {}).get(
                drone_reaction.ALL) or {}
            self.log(f"Added {len(features)} movement(s) of "
                     f"{everything.get('judged', 0)} animal(s) that were in "
                     "view long enough to tell: red away from the line, "
                     "blue towards it, grey along it.")
        except Exception as e:
            self.update_status("add_reaction", "🔴 Error")
            self.log(f"Error adding the movements: {e}")
            QMessageBox.critical(self, "Error",
                                 f"Failed to add the movements: {e}")

    @staticmethod
    def _style_reaction_layer(layer) -> None:
        """Arrows, coloured by where the animal went."""
        categories = []
        for value, colour in DIRECTION_COLOURS:
            arrow = QgsArrowSymbolLayer.create({
                "arrow_width": "0.5", "arrow_start_width": "0.5",
                "head_length": "2.2", "head_thickness": "1.4",
                "is_curved": "0", "is_repeated": "0"})
            arrow.setSubSymbol(QgsFillSymbol.createSimple({
                "color": colour, "outline_style": "no"}))
            symbol = QgsLineSymbol()
            symbol.changeSymbolLayer(0, arrow)
            categories.append(QgsRendererCategory(
                value, symbol, value or "line not known"))
        layer.setRenderer(
            QgsCategorizedSymbolRenderer("direction", categories))

    # ------------------------------------------------------------------ #
    # The thresholds, kept with the project
    # ------------------------------------------------------------------ #

    def _save_reaction_settings(self, project) -> None:
        project.writeEntry(PLUGIN_SCOPE, ENTRY_REACTION, json.dumps({
            "min_displacement_m": self.reaction_min_spin.value(),
            "min_speed_ms": self.reaction_speed_spin.value()}))

    def _load_reaction_settings(self, project) -> None:
        raw, ok = project.readEntry(PLUGIN_SCOPE, ENTRY_REACTION, "")
        try:
            values = json.loads(raw) if ok and raw else {}
        except ValueError:
            values = {}
        self._set_reaction_values(values if isinstance(values, dict) else {})

    def _reset_reaction_settings(self) -> None:
        self._set_reaction_values({})

    def _set_reaction_values(self, values: dict) -> None:
        for spin, key, default in (
                (self.reaction_min_spin, "min_displacement_m",
                 drone_reaction.MIN_DISPLACEMENT_M),
                (self.reaction_speed_spin, "min_speed_ms",
                 drone_reaction.MIN_SPEED_MS)):
            try:
                spin.setValue(float(values.get(key, default)))
            except (TypeError, ValueError):
                spin.setValue(default)
