# -*- coding: utf-8 -*-
"""Dock-widget side of the environmental context (see
:mod:`bambi_environment` for the processing and :mod:`core.environment` for
the data).

Mixed into :class:`BambiDockWidget`; it adds

* the **Environment** configuration sub-tab - how far from the flight a
  satellite scene may be, how clear it has to be, how much ground around the
  flight to fetch;
* the **Environmental Context** group on the Pre-Processing tab - one button
  and one status per source, because each is its own sub-step, plus a button
  that runs them one after the other and one that adds the layers;
* the **Habitat Context** group on the Survey Analytics tab - the summary,
  its charts and the per-track layer.

Kept out of ``bambi_dock_widget.py`` so the feature can be read in one
place.
"""

import os
from typing import Dict, List, Optional

from qgis.core import (
    QgsFeature, QgsField, QgsGeometry, QgsPointXY, QgsRasterLayer,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from .bambi_environment import ENVIRONMENT_STEPS
from .core import environment
from .core import store as _store

ENVIRONMENT_GROUP = "BAMBI Environment"

#: Step -> (button label, tooltip).
_STEP_TEXT = {
    "env_sentinel2": (
        "E1. Fetch Satellite Indices (Sentinel-2)",
        "NDVI (greenness), NDMI (vegetation moisture), NDWI (open water) "
        "and NDSI (snow) from the Sentinel-2 scene nearest to the flight "
        "that is clear over it. Needs an internet connection."),
    "env_landcover": (
        "E2. Fetch Land Cover (ESA WorldCover)",
        "10 m land-cover classes and the tree cover derived from them. "
        "Static (2021 map). Needs an internet connection."),
    "env_terrain": (
        "E3. Compute Terrain Wetness (DEM)",
        "Topographic wetness index from the flight's own DEM: where water "
        "tends to collect. Works under forest and needs no download."),
    "env_weather": (
        "E4. Fetch Weather (Open-Meteo)",
        "Hourly model weather at flight time and over the 24 hours before "
        "it: air and soil temperature, humidity, dew point, wind, cloud, "
        "rain, sunshine. Needs an internet connection."),
}

#: Colour ramps of the continuous layers: ``(position 0-1, (r, g, b))``.
_RAMPS = {
    "ndvi": [(0.0, (166, 97, 26)), (0.25, (223, 194, 125)),
             (0.5, (255, 255, 191)), (0.75, (120, 198, 121)),
             (1.0, (0, 104, 55))],
    "ndmi": [(0.0, (140, 81, 10)), (0.35, (223, 194, 125)),
             (0.5, (245, 245, 245)), (0.7, (128, 205, 193)),
             (1.0, (1, 102, 94))],
    "ndwi": [(0.0, (247, 251, 255)), (0.5, (198, 219, 239)),
             (0.75, (66, 146, 198)), (1.0, (8, 48, 107))],
    "ndsi": [(0.0, (90, 90, 90)), (0.5, (200, 200, 200)),
             (0.7, (220, 240, 250)), (1.0, (255, 255, 255))],
    "tree_cover": [(0.0, (255, 255, 229)), (0.5, (120, 198, 121)),
                   (1.0, (0, 69, 41))],
    "twi": [(0.0, (255, 255, 204)), (0.5, (65, 182, 196)),
            (1.0, (12, 44, 132))],
}


class EnvironmentDockMixin:
    """Environment and habitat UI of the dock widget."""

    # ------------------------------------------------------------------ #
    # Configuration sub-tab
    # ------------------------------------------------------------------ #

    def _build_environment_config_tab(self, config_sub_tabs) -> None:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        config_sub_tabs.addTab(tab, "Environment")

        group = QGroupBox("Environmental Context")
        form = QFormLayout(group)

        info = QLabel(
            "Public data for the ground a flight monitored: satellite "
            "indices, land cover, terrain wetness and weather. Each source "
            "is its own step on the Pre-Processing tab and writes into the "
            "flight's 'environment' folder.")
        info.setWordWrap(True)
        form.addRow(info)

        self.environment_window_spin = QSpinBox()
        self.environment_window_spin.setRange(1, 90)
        self.environment_window_spin.setValue(10)
        self.environment_window_spin.setSuffix(" days")
        self.environment_window_spin.setToolTip(
            "How far before or after the flight a Sentinel-2 scene may have "
            "been taken. The nearest scene that is clear enough is used; "
            "its distance from the flight is recorded.")
        form.addRow("Scene window (±):", self.environment_window_spin)

        self.environment_clear_spin = QSpinBox()
        self.environment_clear_spin.setRange(10, 100)
        self.environment_clear_spin.setValue(80)
        self.environment_clear_spin.setSuffix(" %")
        self.environment_clear_spin.setToolTip(
            "Share of the flight's extent that must be free of cloud, cloud "
            "shadow and missing data for a scene to be used. Cloudy cells "
            "of the chosen scene are left empty.")
        form.addRow("Required clear share:", self.environment_clear_spin)

        self.environment_buffer_spin = QSpinBox()
        self.environment_buffer_spin.setRange(0, 5000)
        self.environment_buffer_spin.setSingleStep(50)
        self.environment_buffer_spin.setValue(200)
        self.environment_buffer_spin.setSuffix(" m")
        self.environment_buffer_spin.setToolTip(
            "Ground fetched around the flight's footprint, so the habitat "
            "next to it is visible too.")
        form.addRow("Buffer around the flight:", self.environment_buffer_spin)

        self.environment_area_check = QCheckBox(
            "Extend to the survey area when one is defined")
        self.environment_area_check.setChecked(True)
        self.environment_area_check.setToolTip(
            "With the survey area switched on (Survey Analytics tab), the "
            "layers cover the whole area and not only the strip flown.")
        form.addRow("", self.environment_area_check)

        tab_layout.addWidget(group)
        tab_layout.addStretch()

    # ------------------------------------------------------------------ #
    # Pre-Processing: the four sources
    # ------------------------------------------------------------------ #

    def _build_environment_steps_group(self) -> QGroupBox:
        group = QGroupBox("Environmental Context")
        layout = QVBoxLayout(group)

        note = QLabel(
            "Independent of the camera and of each other: a source that "
            "cannot be reached fails its own step only.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(note)

        self._environment_buttons: Dict[str, QPushButton] = {}
        self._environment_status_labels: Dict[str, QLabel] = {}
        for step, (text, tooltip) in _STEP_TEXT.items():
            row = QHBoxLayout()
            button = QPushButton(text)
            button.setToolTip(tooltip)
            button.clicked.connect(
                lambda _checked=False, s=step: self.run_environment_step(s))
            label = QLabel("⚪ Not started")
            row.addWidget(button)
            row.addWidget(label)
            layout.addLayout(row)
            self._environment_buttons[step] = button
            self._environment_status_labels[step] = label

        all_row = QHBoxLayout()
        self.environment_all_btn = QPushButton("   → Run All Four")
        self.environment_all_btn.setToolTip(
            "Run E1 to E4 one after the other. A source that fails does not "
            "stop the ones after it.")
        self.environment_all_btn.clicked.connect(self.run_environment_all)
        all_row.addWidget(self.environment_all_btn)
        layout.addLayout(all_row)

        add_row = QHBoxLayout()
        self.add_environment_btn = QPushButton(
            "   → Add Environment Layers to QGIS")
        self.add_environment_btn.setToolTip(
            "Add every environment raster of the flight as a styled layer. "
            "One is shown at a time.")
        self.add_environment_btn.clicked.connect(self.add_environment_to_qgis)
        self.add_environment_status = QLabel("⚪")
        add_row.addWidget(self.add_environment_btn)
        add_row.addWidget(self.add_environment_status)
        layout.addLayout(add_row)

        self._environment_queue: List[str] = []
        return group

    def _environment_config(self) -> dict:
        """The environment settings, as config keys for the processor.

        ``get_config`` can be asked while the tabs are still being built,
        so a widget that is not there yet yields its default.
        """
        def value(attr, default):
            widget = getattr(self, attr, None)
            return default if widget is None else widget.value()

        area_check = getattr(self, "environment_area_check", None)
        return {
            "environment_window_days": value("environment_window_spin", 10),
            "environment_min_clear":
                value("environment_clear_spin", 80) / 100.0,
            "environment_buffer_m": value("environment_buffer_spin", 200),
            "environment_include_survey_area":
                True if area_check is None else area_check.isChecked(),
            "environment_radius_m": value(
                "habitat_radius_spin", environment.DEFAULT_RADIUS_M),
        }

    def _environment_problems(self, step: str, config: dict) -> List[str]:
        """What stops *step* from running on the active flight."""
        from .core import agl_profile, inspection

        problems = []
        target_folder = config.get("target_folder", "")
        if not target_folder or not os.path.isdir(target_folder):
            return ["The flight has no target folder yet (Input tab)."]
        if not config.get("target_epsg"):
            problems.append("The target CRS is not set (Input tab).")
        origin = agl_profile.dem_origin(
            config.get("dem_path", "") or "",
            config.get("alfs_dem_metadata_path", "") or "", target_folder)
        footprint, _how = environment.flight_footprint(target_folder, origin)
        if footprint is None:
            problems.append(
                "The flight's extent is unknown: run 'Calculate Field of "
                "View', or extract the frames with a DEM whose metadata "
                "names its origin.")
        if step in ("env_sentinel2", "env_weather") \
                and environment.flight_time(target_folder) is None:
            problems.append(
                "The flight's date is unknown: run 'Extract Frames' first.")
        if step == "env_terrain":
            dem_path = config.get("dem_path", "") or ""
            mesh = inspection.find_dem_mesh_path(
                dem_path, config.get("alfs_dem_metadata_path", "") or "")
            if not mesh and not os.path.isfile(dem_path):
                problems.append("No DEM mesh is selected (Input tab).")
            elif origin is None:
                problems.append(
                    "The DEM's metadata JSON names no origin.")
        return problems

    def run_environment_step(self, step: str) -> bool:
        """Start one environment sub-step; False when it could not start."""
        problems = self._environment_problems(step, self.get_config())
        if problems:
            reasons = "\n".join(f"  • {p}" for p in problems)
            QMessageBox.warning(
                self, "Missing Prerequisites",
                f"{_STEP_TEXT[step][0]} cannot run:\n\n{reasons}")
            return False
        self.start_worker(step)
        return self.worker is not None

    def run_environment_all(self) -> None:
        """Run the four sources one after the other."""
        if self.worker is not None:
            QMessageBox.warning(self, "Processing",
                                "Another process is already running.")
            return
        config = self.get_config()
        runnable, skipped = [], []
        for step in ENVIRONMENT_STEPS:
            problems = self._environment_problems(step, config)
            (skipped if problems else runnable).append((step, problems))
        if not runnable:
            QMessageBox.warning(
                self, "Missing Prerequisites",
                "None of the environment sources can run:\n\n" + "\n".join(
                    f"  • {p}" for p in dict.fromkeys(
                        p for _s, ps in skipped for p in ps)))
            return
        for step, problems in skipped:
            self.log(f"Skipping {_STEP_TEXT[step][0]}: " + " ".join(problems))
            self.update_status(step, "⚪ Skipped")
        self._environment_queue = [step for step, _p in runnable]
        self._run_next_environment_step()

    def _run_next_environment_step(self) -> None:
        while self._environment_queue:
            step = self._environment_queue.pop(0)
            if self.run_environment_step(step):
                return

    def _on_environment_step_finished(self, step: str) -> None:
        """After a source ended - however it ended - show what is on disk
        and carry on with the next queued one."""
        target_folder = self.get_config().get("target_folder", "")
        done = environment.status(target_folder)
        if done.get(ENVIRONMENT_STEPS[step]):
            self._set_environment_status(step, target_folder)
        self._run_next_environment_step()

    def _set_environment_status(self, step: str, target_folder: str) -> None:
        """Mark *step* completed, with what it found as the tooltip."""
        source = ENVIRONMENT_STEPS[step]
        metadata = environment.read_metadata(target_folder, source) or {}
        label = self._environment_status_labels[step]
        text, tip = "🟢 Completed", ""
        if source == environment.SENTINEL2 and metadata.get("scene_time"):
            offset = metadata.get("offset_days") or 0.0
            text = f"🟢 {str(metadata['scene_time'])[:10]} ({offset:+.1f} d)"
            tip = (f"Scene {metadata.get('scene')}, "
                   f"{(metadata.get('clear_fraction') or 0) * 100:.0f}% clear "
                   "over the flight")
        elif source == environment.WEATHER:
            at = metadata.get("at_flight") or {}
            if at.get("temperature_2m") is not None:
                text = f"🟢 {at['temperature_2m']:.0f} °C at flight"
        label.setText(text)
        label.setToolTip(tip)

    def _refresh_environment_status(self, target_folder: str) -> None:
        """Set the four status labels from what is in the flight's folder."""
        if not hasattr(self, "_environment_status_labels"):
            return
        done = environment.status(target_folder)
        for step, source in ENVIRONMENT_STEPS.items():
            label = self._environment_status_labels[step]
            label.setToolTip("")
            if done.get(source):
                self._set_environment_status(step, target_folder)
            else:
                label.setText("⚪ Not started")

    def _set_environment_buttons_enabled(self, enabled: bool) -> None:
        if not hasattr(self, "_environment_buttons"):
            return
        for button in self._environment_buttons.values():
            button.setEnabled(enabled)
        self.environment_all_btn.setEnabled(enabled)
        self.add_environment_btn.setEnabled(enabled)
        self.habitat_btn.setEnabled(enabled)
        self.habitat_charts_btn.setEnabled(enabled)
        self.add_habitat_btn.setEnabled(enabled)

    # ------------------------------------------------------------------ #
    # Layers
    # ------------------------------------------------------------------ #

    def _style_environment_layer(self, layer: QgsRasterLayer,
                                 spec: environment.Layer) -> None:
        """A colour ramp for a measurement, the legend for land cover."""
        from qgis.core import (
            QgsColorRampShader, QgsPalettedRasterRenderer, QgsRasterShader,
            QgsSingleBandPseudoColorRenderer,
        )
        try:
            if spec.kind == environment.CATEGORICAL:
                from .core import env_landcover
                classes = [
                    QgsPalettedRasterRenderer.Class(value, QColor(*colour),
                                                    name)
                    for value, (name, colour) in env_landcover.CLASSES.items()]
                layer.setRenderer(QgsPalettedRasterRenderer(
                    layer.dataProvider(), 1, classes))
            else:
                if spec.value_range is not None:
                    low, high = spec.value_range
                else:
                    # Between the 2nd and 98th percentile: a few extreme
                    # cells (a stream bed, in the wetness index) would
                    # otherwise flatten the ramp over everything else.
                    low, high = layer.dataProvider().cumulativeCut(
                        1, 0.02, 0.98)
                if high <= low:
                    high = low + 1.0
                ramp = QgsColorRampShader(low, high)
                ramp.setColorRampType(QgsColorRampShader.Type.Interpolated)
                ramp.setColorRampItemList([
                    QgsColorRampShader.ColorRampItem(
                        low + position * (high - low), QColor(*colour),
                        f"{low + position * (high - low):.2f}")
                    for position, colour in _RAMPS.get(
                        spec.key, _RAMPS["twi"])])
                shader = QgsRasterShader()
                shader.setRasterShaderFunction(ramp)
                layer.setRenderer(QgsSingleBandPseudoColorRenderer(
                    layer.dataProvider(), 1, shader))
            layer.triggerRepaint()
        except Exception as exc:  # nosec B110 - an unstyled layer still shows
            self.log(f"Warning: could not style {layer.name()}: {exc}")

    def add_environment_to_qgis(self) -> None:
        """Add the flight's environment rasters as one layer group.

        The group goes below the flight's camera groups - these are base
        layers - and shows one raster at a time: stacked opaque rasters
        would only hide each other.
        """
        target_folder = self.get_config().get("target_folder", "")
        layers = environment.available_layers(target_folder) \
            if target_folder else []
        if not layers:
            QMessageBox.warning(
                self, "Missing Data",
                "This flight has no environment layers yet.\n"
                "Run the Environmental Context steps first.")
            return
        try:
            self.update_status("add_environment", "🟡 Loading...")
            self._remove_layer_group(ENVIRONMENT_GROUP)
            group = self._create_layer_group(ENVIRONMENT_GROUP, at_top=False)
            group.setIsMutuallyExclusive(True)
            from qgis.core import QgsProject

            for index, spec in enumerate(layers):
                layer = QgsRasterLayer(
                    environment.layer_path(target_folder, spec.key),
                    f"{spec.title}")
                if not layer.isValid():
                    raise RuntimeError(
                        f"Failed to load {spec.filename}")
                self._style_environment_layer(layer, spec)
                layer.setCustomProperty("bambi_layer_type",
                                        f"environment_{spec.key}")
                QgsProject.instance().addMapLayer(layer, False)
                node = group.addLayer(layer)
                node.setItemVisibilityChecked(index == 0)
                node.setExpanded(False)
            self.update_status("add_environment", "🟢 Added")
            self.iface.mapCanvas().refresh()
            titles = ", ".join(spec.title for spec in layers)
            self.log(f"Added {len(layers)} environment layer(s): {titles}")
        except Exception as exc:
            self.update_status("add_environment", "🔴 Error")
            self.log(f"Error adding the environment layers: {exc}")
            QMessageBox.critical(
                self, "Error",
                f"Failed to add the environment layers: {exc}")

    # ------------------------------------------------------------------ #
    # Survey Analytics: habitat context
    # ------------------------------------------------------------------ #

    def _build_habitat_group(self) -> QGroupBox:
        group = QGroupBox("Habitat Context")
        layout = QVBoxLayout(group)

        note = QLabel(
            "Environment values where the animals were, against the ground "
            "the flights monitored. Needs the Environmental Context steps "
            "(Pre-Processing tab) on each flight, and tracks.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(note)

        params = QHBoxLayout()
        params.addWidget(QLabel("Radius around each animal:"))
        self.habitat_radius_spin = QDoubleSpinBox()
        self.habitat_radius_spin.setRange(0.0, 500.0)
        self.habitat_radius_spin.setSingleStep(10.0)
        self.habitat_radius_spin.setDecimals(0)
        self.habitat_radius_spin.setValue(30.0)
        self.habitat_radius_spin.setSuffix(" m")
        self.habitat_radius_spin.setToolTip(
            "Each layer is averaged over the cells within this distance of "
            "a track's position. 0 takes the single cell under it.")
        params.addWidget(self.habitat_radius_spin)
        params.addStretch()
        layout.addLayout(params)

        run_row = QHBoxLayout()
        self.habitat_btn = QPushButton("→ Summarise Habitat")
        self.habitat_btn.setToolTip(
            "Write analytics_{camera}/habitat_tracks.csv (one row per "
            "track), habitat_transects.csv (when the population estimate "
            "has run) and habitat.json.")
        self.habitat_btn.clicked.connect(self.run_habitat_summary)
        self.habitat_status = QLabel("⚪ Not started")
        run_row.addWidget(self.habitat_btn)
        run_row.addWidget(self.habitat_status)
        layout.addLayout(run_row)

        charts_row = QHBoxLayout()
        self.habitat_charts_btn = QPushButton("   → Show Habitat Charts")
        self.habitat_charts_btn.clicked.connect(self.show_habitat_charts)
        charts_row.addWidget(self.habitat_charts_btn)
        layout.addLayout(charts_row)

        add_row = QHBoxLayout()
        self.add_habitat_btn = QPushButton("   → Add Habitat Tracks to QGIS")
        self.add_habitat_btn.setToolTip(
            "One point per track with its environment values as attributes, "
            "to filter, style and export.")
        self.add_habitat_btn.clicked.connect(self.add_habitat_to_qgis)
        self.add_habitat_status = QLabel("⚪")
        add_row.addWidget(self.add_habitat_btn)
        add_row.addWidget(self.add_habitat_status)
        layout.addLayout(add_row)
        return group

    def run_habitat_summary(self) -> None:
        survey = self._checked_survey()
        if survey is None:
            return

        def checks(flight):
            missing = []
            if not self.has_stage(_store.TRACKS, survey.modality,
                                  flight.target_folder):
                missing.append(f"{survey.camera_label} tracks "
                               "(run 'Track Animals' first)")
            if not any(environment.status(flight.target_folder).values()):
                missing.append("Environment data (run the Environmental "
                               "Context steps on the Pre-Processing tab)")
            return missing

        if self._survey_prerequisites(survey, checks):
            self.start_worker("habitat_summary")

    def _habitat_files(self):
        """``(summary path, tracks path)`` of the current survey."""
        from .core import habitat

        folder = self.current_survey().analytics_folder
        return (os.path.join(folder, habitat.SUMMARY_JSON),
                os.path.join(folder, habitat.TRACKS_JSON))

    def show_habitat_charts(self) -> None:
        from .bambi_habitat_dialog import BambiHabitatDialog
        from .core import habitat

        summary_path, tracks_path = self._habitat_files()
        if not os.path.isfile(summary_path):
            QMessageBox.warning(
                self, "Missing Data",
                "The habitat summary has not been built yet.\n"
                "Run 'Summarise Habitat' first.")
            return
        try:
            summary = habitat.read_json(summary_path)
            tracks = (habitat.read_json(tracks_path)
                      if os.path.isfile(tracks_path) else [])
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Habitat Context",
                                 f"Could not read {summary_path}: {exc}")
            return
        dialog = BambiHabitatDialog(summary, tracks, parent=self)
        dialog.setModal(False)
        dialog.show()
        self._habitat_dialog = dialog

    def add_habitat_to_qgis(self) -> None:
        """The per-track habitat table as a point layer in the survey group."""
        from .core import habitat

        survey = self.current_survey()
        _summary_path, tracks_path = self._habitat_files()
        if not os.path.isfile(tracks_path):
            QMessageBox.warning(
                self, "Missing Data",
                "The habitat summary has not been built yet.\n"
                "Run 'Summarise Habitat' first.")
            return
        columns = list(habitat.TRACK_COLUMNS)
        name = f"BAMBI Habitat Tracks ({survey.camera_label})"
        try:
            self.update_status("add_habitat", "🟡 Loading...")
            rows = habitat.read_json(tracks_path)
            epsg = survey.epsg or self.get_config().get("target_epsg", 32633)
            layer = QgsVectorLayer(f"Point?crs=EPSG:{epsg}", name, "memory")
            provider = layer.dataProvider()

            def _field(column):
                sample = next((r[column] for r in rows
                               if r.get(column) is not None), None)
                if isinstance(sample, bool):
                    return QgsField(column, QVariant.Bool)
                if isinstance(sample, int):
                    return QgsField(column, QVariant.LongLong)
                if isinstance(sample, float):
                    return QgsField(column, QVariant.Double)
                return QgsField(column, QVariant.String)

            provider.addAttributes([_field(column) for column in columns])
            layer.updateFields()
            features = []
            for row in rows:
                feature = QgsFeature(layer.fields())
                if row.get("x") is not None and row.get("y") is not None:
                    feature.setGeometry(QgsGeometry.fromPointXY(
                        QgsPointXY(row["x"], row["y"])))
                feature.setAttributes([row.get(c) for c in columns])
                features.append(feature)
            provider.addFeatures(features)
            layer.updateExtents()

            layer = self._persist_memory_layer(
                layer, f"Habitat_Tracks_{survey.modality}",
                "analytics_layers", display_name=name,
                base_folder=survey.folder)
            layer.setCustomProperty("bambi_layer_type", "habitat_tracks")
            self._add_layer_to_survey(layer)
            self.update_status("add_habitat", "🟢 Added")
            self.iface.mapCanvas().refresh()
            self.log(f"Added habitat layer with {len(features)} track(s)")
        except Exception as exc:
            self.update_status("add_habitat", "🔴 Error")
            self.log(f"Error adding the habitat tracks: {exc}")
            QMessageBox.critical(
                self, "Error", f"Failed to add the habitat tracks: {exc}")

    def _refresh_habitat_status(self) -> None:
        if not hasattr(self, "habitat_status"):
            return
        try:
            summary_path, _tracks = self._habitat_files()
        except Exception:  # nosec B110 - no survey configured yet
            return
        self.habitat_status.setText(
            "🟢 Completed" if os.path.isfile(summary_path)
            else "⚪ Not started")

    def _environment_status_map(self) -> Dict[str, Optional[QLabel]]:
        """Status labels of this mixin, for ``update_status``."""
        labels: Dict[str, Optional[QLabel]] = dict(
            getattr(self, "_environment_status_labels", {}))
        labels["add_environment"] = getattr(
            self, "add_environment_status", None)
        labels["habitat_summary"] = getattr(self, "habitat_status", None)
        labels["add_habitat"] = getattr(self, "add_habitat_status", None)
        return labels
