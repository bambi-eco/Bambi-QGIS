# -*- coding: utf-8 -*-
"""
BAMBI Random Flight Strategy Planner
=====================================
Dialog for configuring and running RandomStrategy or RandomLoopStrategy
from the drone_flight_simulation package, then importing the results as
persistent QGIS layers.
"""

import os
import sys
import tempfile

from qgis.PyQt.QtCore import Qt, QThread, QObject, pyqtSignal, QSettings
from qgis.PyQt.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QProgressBar, QPushButton, QRadioButton,
    QScrollArea, QSizePolicy, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)
from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsLineSymbol, QgsMarkerSymbol, QgsProject,
    QgsSingleSymbolRenderer, QgsVectorFileWriter, QgsVectorLayer,
    QgsWkbTypes,
)

_ROUTE_COLORS = [
    "#E53935", "#8E24AA", "#1E88E5", "#00ACC1",
    "#43A047", "#FB8C00", "#6D4C41", "#039BE5",
]


class _FlightPlanWorker(QObject):
    finished = pyqtSignal(bool, str)

    def __init__(self, strategy_cls, strategy_kwargs,
                 area_path, start_path, target_path, invalid_path):
        super().__init__()
        self._strategy_cls = strategy_cls
        self._strategy_kwargs = strategy_kwargs
        self._area_path = area_path
        self._start_path = start_path
        self._target_path = target_path
        self._invalid_path = invalid_path

    def run(self):
        try:
            try:
                from fiona.drvsupport import supported_drivers
                supported_drivers["kml"] = "rw"
                supported_drivers["KML"] = "rw"
                supported_drivers["LIBKML"] = "rw"
            except Exception:
                pass

            strategy = self._strategy_cls(**self._strategy_kwargs)
            strategy.create_routes(
                area_path=self._area_path,
                start_points_path=self._start_path,
                target_path=self._target_path,
                invalid_areas_path=self._invalid_path,
            )
            self.finished.emit(True, "")
        except Exception:
            import traceback
            self.finished.emit(False, traceback.format_exc())


class FlightPlannerDialog(QDialog):
    """Dialog for configuring and running the random flight strategy planner."""

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self._worker = None
        self._thread = None
        self._temp_files = []
        self._target_path = ""
        self.setWindowTitle("Random Flight Strategy Planner")
        self.resize(720, 860)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        sv_widget = QWidget()
        scroll.setWidget(sv_widget)
        sv = QVBoxLayout(sv_widget)
        sv.setSpacing(8)
        outer.addWidget(scroll, stretch=1)

        sv.addWidget(self._build_module_group())
        sv.addWidget(self._build_inputs_group())
        sv.addWidget(self._build_strategy_group())
        sv.addWidget(self._build_params_group())
        sv.addWidget(self._build_output_group())
        sv.addStretch()

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.clicked.connect(self._run)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(self._run_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setVisible(False)
        outer.addWidget(self._progress_bar)

        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setFixedHeight(150)
        outer.addWidget(self._log_edit)

    def _build_module_group(self):
        grp = QGroupBox("Strategy Module")
        fl = QFormLayout(grp)
        self._module_path_edit = QLineEdit()
        self._module_path_edit.setPlaceholderText(
            r"Path to drone_flight_simulation\src"
        )
        saved = QSettings().value("bambi/flight_planner/module_path", "")
        if not saved:
            candidate = r"C:\D\Projects\drone_flight_simulation\src"
            if os.path.isdir(candidate):
                saved = candidate
        self._module_path_edit.setText(saved)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_module_path)
        row = QHBoxLayout()
        row.addWidget(self._module_path_edit)
        row.addWidget(browse)
        fl.addRow("Module path:", row)
        return grp

    def _build_inputs_group(self):
        grp = QGroupBox("Input Data")
        fl = QFormLayout(grp)

        area_w, self._area_combo, self._area_file_edit = self._make_geo_row("Polygon")
        fl.addRow("Monitoring Area *:", area_w)

        start_w, self._start_combo, self._start_file_edit = self._make_geo_row("Point")
        fl.addRow("Start Positions *:", start_w)

        inv_w, self._invalid_combo, self._invalid_file_edit = self._make_geo_row("Polygon")
        fl.addRow("Invalid Areas (opt.):", inv_w)

        return grp

    def _build_strategy_group(self):
        grp = QGroupBox("Strategy")
        hl = QHBoxLayout(grp)
        self._random_radio = QRadioButton("Random Strategy")
        self._random_loop_radio = QRadioButton("Random Loop Strategy")
        self._random_radio.setChecked(True)
        self._strat_btn_group = QButtonGroup(self)
        self._strat_btn_group.addButton(self._random_radio)
        self._strat_btn_group.addButton(self._random_loop_radio)
        hl.addWidget(self._random_radio)
        hl.addWidget(self._random_loop_radio)
        hl.addStretch()
        self._random_radio.toggled.connect(self._on_strategy_changed)
        return grp

    def _build_params_group(self):
        grp = QGroupBox("Parameters")
        fl = QFormLayout(grp)

        self._grid_size_spin = QDoubleSpinBox()
        self._grid_size_spin.setRange(1.0, 100_000.0)
        self._grid_size_spin.setValue(400.0)
        self._grid_size_spin.setSingleStep(50.0)
        self._grid_size_spin.setSuffix(" m")
        fl.addRow("Grid Size:", self._grid_size_spin)

        self._max_start_dist_spin = QDoubleSpinBox()
        self._max_start_dist_spin.setRange(1.0, 1_000_000.0)
        self._max_start_dist_spin.setValue(3000.0)
        self._max_start_dist_spin.setSingleStep(100.0)
        self._max_start_dist_spin.setSuffix(" m")
        fl.addRow("Max Start/Stop Distance:", self._max_start_dist_spin)

        self._min_transects_spin = QSpinBox()
        self._min_transects_spin.setRange(1, 100_000)
        self._min_transects_spin.setValue(40)
        fl.addRow("Min Transects (plan):", self._min_transects_spin)

        max_t_w = QWidget()
        max_t_hl = QHBoxLayout(max_t_w)
        max_t_hl.setContentsMargins(0, 0, 0, 0)
        self._max_transects_spin = QSpinBox()
        self._max_transects_spin.setRange(1, 100_000)
        self._max_transects_spin.setValue(100)
        self._max_transects_spin.setEnabled(False)
        self._max_transects_none_chk = QCheckBox("No upper limit")
        self._max_transects_none_chk.setChecked(True)
        self._max_transects_none_chk.toggled.connect(
            lambda checked: self._max_transects_spin.setEnabled(not checked)
        )
        max_t_hl.addWidget(self._max_transects_spin)
        max_t_hl.addWidget(self._max_transects_none_chk)
        fl.addRow("Max Transects (plan):", max_t_w)

        self._max_distance_spin = QDoubleSpinBox()
        self._max_distance_spin.setRange(1.0, 1_000_000.0)
        self._max_distance_spin.setValue(2000.0)
        self._max_distance_spin.setSingleStep(100.0)
        self._max_distance_spin.setSuffix(" m")
        fl.addRow("Max Route Distance:", self._max_distance_spin)

        self._min_overlap_spin = QDoubleSpinBox()
        self._min_overlap_spin.setRange(0.0, 1.0)
        self._min_overlap_spin.setValue(0.75)
        self._min_overlap_spin.setSingleStep(0.05)
        self._min_overlap_spin.setDecimals(2)
        fl.addRow("Min Transect Overlap:", self._min_overlap_spin)

        self._num_retries_spin = QSpinBox()
        self._num_retries_spin.setRange(1, 1_000_000)
        self._num_retries_spin.setValue(100)
        fl.addRow("Number of Retries:", self._num_retries_spin)

        self._epsg_spin = QSpinBox()
        self._epsg_spin.setRange(1, 999_999)
        self._epsg_spin.setValue(32633)
        fl.addRow("Target CRS EPSG:", self._epsg_spin)

        self._min_per_route_spin = QSpinBox()
        self._min_per_route_spin.setRange(1, 10_000)
        self._min_per_route_spin.setValue(3)
        fl.addRow("Min Transects per Route:", self._min_per_route_spin)

        self._x_offset_spin = QDoubleSpinBox()
        self._x_offset_spin.setRange(-1_000_000.0, 1_000_000.0)
        self._x_offset_spin.setValue(0.0)
        self._x_offset_spin.setSuffix(" m")
        fl.addRow("X Offset (E/W):", self._x_offset_spin)

        self._y_offset_spin = QDoubleSpinBox()
        self._y_offset_spin.setRange(-1_000_000.0, 1_000_000.0)
        self._y_offset_spin.setValue(0.0)
        self._y_offset_spin.setSuffix(" m")
        fl.addRow("Y Offset (N/S):", self._y_offset_spin)

        pad_w = QWidget()
        pad_hl = QHBoxLayout(pad_w)
        pad_hl.setContentsMargins(0, 0, 0, 0)
        self._padding_north_spin = self._make_pad_spin()
        self._padding_east_spin = self._make_pad_spin()
        self._padding_south_spin = self._make_pad_spin()
        self._padding_west_spin = self._make_pad_spin()
        for lbl, spin in [("N:", self._padding_north_spin), ("E:", self._padding_east_spin),
                           ("S:", self._padding_south_spin), ("W:", self._padding_west_spin)]:
            pad_hl.addWidget(QLabel(lbl))
            pad_hl.addWidget(spin)
        fl.addRow("Grid Padding:", pad_w)

        self._seed_edit = QLineEdit()
        self._seed_edit.setPlaceholderText("Leave empty for random seed")
        fl.addRow("Seed:", self._seed_edit)

        # --- RandomStrategy-specific ---
        self._random_only_widget = QGroupBox("Random Strategy Parameters")
        rand_fl = QFormLayout(self._random_only_widget)

        self._max_overlapping_spin = QSpinBox()
        self._max_overlapping_spin.setRange(0, 100_000)
        self._max_overlapping_spin.setValue(0)
        rand_fl.addRow("Max Overlapping Transects:", self._max_overlapping_spin)

        self._max_flights_spin = QSpinBox()
        self._max_flights_spin.setRange(1, 1_000_000)
        self._max_flights_spin.setValue(100)
        rand_fl.addRow("Max Number of Flights:", self._max_flights_spin)

        self._random_search_chk = QCheckBox("Random search (uncheck for sorted by length)")
        self._random_search_chk.setChecked(True)
        rand_fl.addRow("", self._random_search_chk)

        fl.addRow(self._random_only_widget)

        # --- RandomLoopStrategy-specific ---
        self._loop_only_widget = QGroupBox("Random Loop Strategy Parameters")
        loop_fl = QFormLayout(self._loop_only_widget)

        self._retries_per_route_spin = QSpinBox()
        self._retries_per_route_spin.setRange(1, 1_000_000)
        self._retries_per_route_spin.setValue(50)
        loop_fl.addRow("Retries per Route:", self._retries_per_route_spin)

        fl.addRow(self._loop_only_widget)
        self._loop_only_widget.setVisible(False)

        return grp

    def _make_pad_spin(self):
        s = QSpinBox()
        s.setRange(0, 100)
        s.setFixedWidth(60)
        return s

    def _build_output_group(self):
        grp = QGroupBox("Output")
        fl = QFormLayout(grp)
        self._target_edit = QLineEdit()
        self._target_edit.setPlaceholderText("Target folder for output files")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_target)
        row = QHBoxLayout()
        row.addWidget(self._target_edit)
        row.addWidget(browse)
        fl.addRow("Target Folder *:", row)
        return grp

    # ------------------------------------------------------------------
    # Geo row helpers
    # ------------------------------------------------------------------

    def _make_geo_row(self, geom_type):
        """Returns (widget, combo, file_edit) with layer-picker + file-picker rows."""
        w = QWidget()
        vb = QVBoxLayout(w)
        vb.setContentsMargins(0, 2, 0, 2)
        vb.setSpacing(2)

        lr = QHBoxLayout()
        lr.setContentsMargins(0, 0, 0, 0)
        combo = QComboBox()
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(26)
        refresh_btn.setToolTip("Refresh layer list")
        refresh_btn.clicked.connect(lambda: self._populate_layer_combo(combo, geom_type))
        lr.addWidget(QLabel("Layer:"))
        lr.addWidget(combo)
        lr.addWidget(refresh_btn)
        vb.addLayout(lr)

        fr = QHBoxLayout()
        fr.setContentsMargins(0, 0, 0, 0)
        file_edit = QLineEdit()
        file_edit.setPlaceholderText("— or select a file (.shp / .kml / .geojson) —")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(lambda: self._browse_geo_file(file_edit))
        fr.addWidget(QLabel("File:"))
        fr.addWidget(file_edit)
        fr.addWidget(browse_btn)
        vb.addLayout(fr)

        self._populate_layer_combo(combo, geom_type)
        return w, combo, file_edit

    def _populate_layer_combo(self, combo, geom_type):
        combo.clear()
        combo.addItem("-- Select layer --", None)
        for lid, layer in QgsProject.instance().mapLayers().items():
            if not isinstance(layer, QgsVectorLayer):
                continue
            gt = layer.geometryType()
            if geom_type == "Polygon" and gt == QgsWkbTypes.PolygonGeometry:
                combo.addItem(layer.name(), lid)
            elif geom_type == "Point" and gt == QgsWkbTypes.PointGeometry:
                combo.addItem(layer.name(), lid)

    def _browse_geo_file(self, edit):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Geographic File", "",
            "Geo Files (*.shp *.kml *.KML *.geojson *.gpkg);;All Files (*)"
        )
        if path:
            edit.setText(path)

    def _browse_module_path(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select drone_flight_simulation/src directory"
        )
        if folder:
            self._module_path_edit.setText(folder)
            QSettings().setValue("bambi/flight_planner/module_path", folder)

    def _browse_target(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Target Folder")
        if folder:
            self._target_edit.setText(folder)

    # ------------------------------------------------------------------
    # Strategy switching
    # ------------------------------------------------------------------

    def _on_strategy_changed(self, is_random):
        self._random_only_widget.setVisible(is_random)
        self._loop_only_widget.setVisible(not is_random)

    # ------------------------------------------------------------------
    # Geo input resolution
    # ------------------------------------------------------------------

    def _resolve_geo_input(self, combo, file_edit, optional=False):
        """Return a file path usable by geopandas, exporting from QGIS layer if needed."""
        file_path = file_edit.text().strip()
        if file_path:
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            return file_path

        layer_id = combo.currentData()
        if layer_id is None:
            if optional:
                return None
            raise ValueError("No layer or file selected.")

        layer = QgsProject.instance().mapLayer(layer_id)
        if layer is None:
            raise ValueError("Selected layer no longer exists in the project.")

        uri = layer.dataProvider().dataSourceUri()
        uri_path = uri.split("|")[0]
        if os.path.exists(uri_path):
            return uri_path

        # Memory or unsupported provider — export to temp GeoJSON
        tmp = tempfile.mktemp(suffix=".geojson")
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GeoJSON"
        options.fileEncoding = "utf-8"
        result = QgsVectorFileWriter.writeAsVectorFormatV3(
            layer, tmp, QgsProject.instance().transformContext(), options
        )
        error_code = result[0]
        error_msg = result[1] if len(result) > 1 else ""
        if error_code != QgsVectorFileWriter.NoError:
            raise ValueError(f"Failed to export layer '{layer.name()}': {error_msg}")
        self._temp_files.append(tmp)
        return tmp

    # ------------------------------------------------------------------
    # Parameter collection
    # ------------------------------------------------------------------

    def _collect_kwargs(self):
        max_t = (None if self._max_transects_none_chk.isChecked()
                 else self._max_transects_spin.value())
        seed_text = self._seed_edit.text().strip()
        seed = int(seed_text) if seed_text else None

        kwargs = {
            "grid_size": self._grid_size_spin.value(),
            "max_start_and_stop_distance": self._max_start_dist_spin.value(),
            "min_transects": self._min_transects_spin.value(),
            "max_transects": max_t,
            "max_distance": self._max_distance_spin.value(),
            "min_transect_overlap": self._min_overlap_spin.value(),
            "number_of_retries": self._num_retries_spin.value(),
            "target_crs_epsg": self._epsg_spin.value(),
            "min_transects_per_route": self._min_per_route_spin.value(),
            "x_offset": self._x_offset_spin.value(),
            "y_offset": self._y_offset_spin.value(),
            "padding_north": self._padding_north_spin.value(),
            "padding_east": self._padding_east_spin.value(),
            "padding_south": self._padding_south_spin.value(),
            "padding_west": self._padding_west_spin.value(),
            "seed": seed,
        }

        if self._random_radio.isChecked():
            kwargs["max_number_of_overlapping_transects"] = self._max_overlapping_spin.value()
            kwargs["max_number_of_flights"] = self._max_flights_spin.value()
            kwargs["random_search"] = self._random_search_chk.isChecked()
        else:
            kwargs["number_of_retries_per_route"] = self._retries_per_route_spin.value()

        return kwargs

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _run(self):
        module_path = self._module_path_edit.text().strip()
        if module_path and module_path not in sys.path:
            sys.path.insert(0, module_path)
        if module_path:
            QSettings().setValue("bambi/flight_planner/module_path", module_path)

        try:
            from drone_survey_planner.evaluation_flight_strategy import (
                RandomStrategy, RandomLoopStrategy,
            )
        except ImportError as exc:
            QMessageBox.critical(
                self, "Import Error",
                f"Cannot import strategy module:\n{exc}\n\n"
                "Set the correct 'Module path' to the drone_flight_simulation/src directory."
            )
            return

        try:
            area_path = self._resolve_geo_input(self._area_combo, self._area_file_edit)
            start_path = self._resolve_geo_input(self._start_combo, self._start_file_edit)
            invalid_path = self._resolve_geo_input(
                self._invalid_combo, self._invalid_file_edit, optional=True
            )
        except ValueError as exc:
            QMessageBox.critical(self, "Input Error", str(exc))
            return

        target_path = self._target_edit.text().strip()
        if not target_path:
            QMessageBox.critical(self, "Input Error", "Please select a target folder.")
            return
        os.makedirs(target_path, exist_ok=True)
        self._target_path = target_path

        try:
            kwargs = self._collect_kwargs()
        except Exception as exc:
            QMessageBox.critical(self, "Parameter Error", str(exc))
            return

        strategy_cls = (RandomStrategy if self._random_radio.isChecked()
                        else RandomLoopStrategy)

        self._run_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._log_edit.clear()
        self._log_edit.append("Starting flight plan generation…")

        self._worker = _FlightPlanWorker(
            strategy_cls, kwargs, area_path, start_path, target_path, invalid_path
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_finished(self, success, error_msg):
        self._run_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        if not success:
            self._log_edit.append("FAILED.\n\n" + error_msg)
            QMessageBox.critical(self, "Planning Failed",
                                 "Flight planning failed. See the log below for details.")
            return
        self._log_edit.append("Completed successfully!")
        self._load_results_to_qgis(self._target_path)

    # ------------------------------------------------------------------
    # Load results into QGIS as persistent layers
    # ------------------------------------------------------------------

    def _load_results_to_qgis(self, target_path):
        root = QgsProject.instance().layerTreeRoot()
        folder_name = os.path.basename(os.path.normpath(target_path))
        group_name = f"Flight Plan – {folder_name}"
        group = root.insertGroup(0, group_name)

        def add_layer(path, name, parent=None):
            if not os.path.exists(path):
                return None
            lyr = QgsVectorLayer(path, name, "ogr")
            if not lyr.isValid():
                return None
            QgsProject.instance().addMapLayer(lyr, False)
            (parent or group).addLayer(lyr)
            return lyr

        grid_lyr = add_layer(
            os.path.join(target_path, "grid_filtered.geojson"), "Grid (Filtered)"
        )
        if grid_lyr:
            sym = QgsMarkerSymbol.createSimple(
                {"name": "circle", "color": "#BDBDBD", "size": "1.5",
                 "outline_style": "no"}
            )
            grid_lyr.setRenderer(QgsSingleSymbolRenderer(sym))

        transect_lyr = add_layer(
            os.path.join(target_path, "transects_valids.geojson"), "Transects (Valid)"
        )
        if transect_lyr:
            sym = QgsLineSymbol.createSimple(
                {"color": "#9E9E9E", "width": "0.3"}
            )
            transect_lyr.setRenderer(QgsSingleSymbolRenderer(sym))

        start_lyr = add_layer(
            os.path.join(target_path, "startpoints.geojson"), "Start Points"
        )
        if start_lyr:
            sym = QgsMarkerSymbol.createSimple(
                {"name": "triangle", "color": "#1565C0", "size": "5",
                 "outline_color": "#0D47A1", "outline_width": "0.4"}
            )
            start_lyr.setRenderer(QgsSingleSymbolRenderer(sym))

        routes_dir = os.path.join(target_path, "routes", "valid")
        if os.path.isdir(routes_dir):
            routes_group = group.addGroup("Flight Routes")
            route_files = sorted(
                f for f in os.listdir(routes_dir) if f.endswith(".geojson")
            )
            for idx, fname in enumerate(route_files):
                label = os.path.splitext(fname)[0].replace("_", " ").title()
                lyr = add_layer(os.path.join(routes_dir, fname), label, routes_group)
                if lyr:
                    color = _ROUTE_COLORS[idx % len(_ROUTE_COLORS)]
                    sym = QgsLineSymbol.createSimple({"color": color, "width": "0.8"})
                    lyr.setRenderer(QgsSingleSymbolRenderer(sym))

        self.iface.mapCanvas().refresh()
        QMessageBox.information(
            self, "Layers Imported",
            f"Results have been imported into QGIS group:\n'{group_name}'"
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        for tmp in self._temp_files:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
        self._temp_files.clear()
        super().closeEvent(event)
