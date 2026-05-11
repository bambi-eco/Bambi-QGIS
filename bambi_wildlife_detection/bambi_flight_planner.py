# -*- coding: utf-8 -*-
"""
BAMBI Random Flight Strategy Planner
=====================================
Dialog for configuring and running RandomStrategy or RandomLoopStrategy
from the drone_flight_simulation package, then importing the results as
persistent QGIS layers.
"""

import io
import json
import os
import sys
import tempfile


def _fix_package_priority():
    """Permanently move user site-packages to the end of sys.path.

    QGIS embeds its own pyproj linked against the PROJ DLLs that are already
    loaded in the process.  When the user has also pip-installed pyproj into
    their user site-packages, Python finds that version first.  Its compiled
    extension (_context.pyd) looks for PROJ DLLs that are not on the search
    path → "DLL load failed".

    Geopandas checks pyproj availability at *import time* and caches the
    result in HAS_PYPROJ.  The fix must therefore run before geopandas is
    first imported (which happens indirectly through evaluation_flight_strategy
    on first use).  Applying the fix at module-load time here ensures the
    correct order is in place for the whole QGIS session.
    """
    import site as _site
    try:
        user_site = _site.getusersitepackages()
    except Exception:
        return
    if user_site not in sys.path:
        return
    # Demote user site-packages to the tail so QGIS-bundled packages win.
    sys.path[:] = [p for p in sys.path if p != user_site] + [user_site]
    # Remove any half-imported pyproj left by a prior failed attempt so the
    # next import starts fresh from the now-prioritised QGIS location.
    for _k in [k for k in sys.modules if k == 'pyproj' or k.startswith('pyproj.')]:
        del sys.modules[_k]


_fix_package_priority()

from qgis.PyQt.QtCore import QSettings, QThread, QObject, pyqtSignal
from qgis.PyQt.QtGui import QColor, QFont
from qgis.PyQt.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QProgressBar, QPushButton, QRadioButton,
    QScrollArea, QSizePolicy, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)
from qgis.core import (
    QgsFillSymbol, QgsLineSymbol, QgsMarkerSymbol, QgsPalLayerSettings,
    QgsProject, QgsSingleSymbolRenderer, QgsTextBufferSettings, QgsTextFormat,
    QgsVectorFileWriter, QgsVectorLayer, QgsVectorLayerSimpleLabeling,
    QgsWkbTypes,
)

def _fix_std_streams():
    # QGIS sets sys.stderr/stdout to None; numpy/geopandas crash when they
    # try to write deprecation or error messages through those streams.
    if sys.stderr is None:
        sys.stderr = io.StringIO()
    if sys.stdout is None:
        sys.stdout = io.StringIO()


def _write_route_line(src_path, out_path):
    """Write a clean single-feature LineString GeoJSON containing only the
    'total-route' feature from a mixed-geometry route GeoJSON.  Returns
    *out_path* on success or None if the feature is not found."""
    try:
        with open(src_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    total = next(
        (feat for feat in data.get("features", [])
         if feat.get("properties", {}).get("name") == "total-route"),
        None,
    )
    if total is None:
        return None
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": [total]}, f)
    return out_path


def _write_route_transects(src_path, out_path):
    """Extract individual survey transect LineStrings from a route GeoJSON,
    tag each with a 1-based ``transect_no`` property, and write a clean
    single-geometry-type GeoJSON to *out_path*.  Returns *out_path* on
    success or None if no transects are found."""
    _SKIP = {"arrival", "departure", "total-route"}
    try:
        with open(src_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    transects = []
    no = 1
    for feat in data.get("features", []):
        if feat.get("geometry", {}).get("type") != "LineString":
            continue
        if feat.get("properties", {}).get("name") in _SKIP:
            continue
        feat = json.loads(json.dumps(feat))  # deep copy
        feat.setdefault("properties", {})["transect_no"] = no
        transects.append(feat)
        no += 1
    if not transects:
        return None
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": transects}, f)
    return out_path


def _enable_transect_labels(layer, color_hex):
    """Configure *layer* to display its ``transect_no`` field as map labels."""
    fmt = QgsTextFormat()
    font = QFont("Arial", 7)
    font.setBold(True)
    fmt.setFont(font)
    fmt.setSize(7)
    fmt.setColor(QColor(color_hex))

    buf = QgsTextBufferSettings()
    buf.setEnabled(True)
    buf.setSize(0.8)
    buf.setColor(QColor("white"))
    fmt.setBuffer(buf)

    settings = QgsPalLayerSettings()
    settings.fieldName = "transect_no"
    settings.placement = QgsPalLayerSettings.Line
    settings.setFormat(fmt)

    labeling = QgsVectorLayerSimpleLabeling(settings)
    layer.setLabeling(labeling)
    layer.setLabelsEnabled(True)


_ROUTE_COLORS = [
    "#E53935", "#8E24AA", "#1E88E5", "#00ACC1",
    "#43A047", "#FB8C00", "#6D4C41", "#039BE5",
]

_SETTINGS_PREFIX = "bambi/flight_planner/"

_DEFAULTS = {
    "strategy":                          "random",
    "grid_size":                         400.0,
    "max_start_and_stop_distance":       3000.0,
    "min_transects":                     40,
    "max_transects_enabled":             False,
    "max_transects":                     100,
    "max_distance":                      2000.0,
    "min_transect_overlap":              0.75,
    "number_of_retries":                 100,
    "target_crs_epsg":                   32633,
    "min_transects_per_route":           3,
    "x_offset":                          0.0,
    "y_offset":                          0.0,
    "padding_north":                     0,
    "padding_east":                      0,
    "padding_south":                     0,
    "padding_west":                      0,
    "seed":                              "",
    "max_number_of_overlapping_transects": 0,
    "max_number_of_flights":             100,
    "random_search":                     True,
    "number_of_retries_per_route":       50,
    "target_folder":                     "",
}


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
        _fix_std_streams()
        try:
            try:
                from fiona.drvsupport import supported_drivers
                supported_drivers["kml"] = "rw"
                supported_drivers["KML"] = "rw"
                supported_drivers["LIBKML"] = "rw"
            except Exception:
                pass

            strategy = self._strategy_cls(**self._strategy_kwargs)
            routes = strategy.create_routes(
                area_path=self._area_path,
                start_points_path=self._start_path,
                target_path=self._target_path,
                invalid_areas_path=self._invalid_path,
            )
            count = len(routes) if routes else 0
            self.finished.emit(True, str(count))
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
        self._area_external_path = None
        self._invalid_external_path = None
        self.setWindowTitle("Random Flight Strategy Planner")
        self.resize(720, 800)
        self._build_ui()
        self._load_settings()

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

        sv.addWidget(self._build_inputs_group())
        sv.addWidget(self._build_strategy_group())
        sv.addWidget(self._build_params_group())
        sv.addWidget(self._build_output_group())
        sv.addStretch()

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.clicked.connect(self._run)
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(lambda: self._apply_settings(_DEFAULTS))
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(self._run_btn)
        btn_row.addStretch()
        btn_row.addWidget(reset_btn)
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
    # Settings persistence
    # ------------------------------------------------------------------

    def _apply_settings(self, values):
        """Apply a values dict to every parameter widget."""
        if values.get("strategy") == "loop":
            self._random_loop_radio.setChecked(True)
        else:
            self._random_radio.setChecked(True)

        self._grid_size_spin.setValue(float(values["grid_size"]))
        self._max_start_dist_spin.setValue(float(values["max_start_and_stop_distance"]))
        self._min_transects_spin.setValue(int(values["min_transects"]))

        max_t_enabled = self._to_bool(values["max_transects_enabled"])
        self._max_transects_none_chk.setChecked(not max_t_enabled)
        self._max_transects_spin.setValue(int(values["max_transects"]))

        self._max_distance_spin.setValue(float(values["max_distance"]))
        self._min_overlap_spin.setValue(float(values["min_transect_overlap"]))
        self._num_retries_spin.setValue(int(values["number_of_retries"]))
        self._epsg_spin.setValue(int(values["target_crs_epsg"]))
        self._min_per_route_spin.setValue(int(values["min_transects_per_route"]))
        self._x_offset_spin.setValue(float(values["x_offset"]))
        self._y_offset_spin.setValue(float(values["y_offset"]))
        self._padding_north_spin.setValue(int(values["padding_north"]))
        self._padding_east_spin.setValue(int(values["padding_east"]))
        self._padding_south_spin.setValue(int(values["padding_south"]))
        self._padding_west_spin.setValue(int(values["padding_west"]))
        self._seed_edit.setText(str(values["seed"]))
        self._max_overlapping_spin.setValue(int(values["max_number_of_overlapping_transects"]))
        self._max_flights_spin.setValue(int(values["max_number_of_flights"]))
        self._random_search_chk.setChecked(self._to_bool(values["random_search"]))
        self._retries_per_route_spin.setValue(int(values["number_of_retries_per_route"]))
        self._target_edit.setText(str(values["target_folder"]))

    @staticmethod
    def _to_bool(v):
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("true", "1", "yes")

    def _load_settings(self):
        s = QSettings()
        values = {
            key: s.value(_SETTINGS_PREFIX + key, default)
            for key, default in _DEFAULTS.items()
        }
        self._apply_settings(values)

    def _save_settings(self):
        s = QSettings()
        values = {
            "strategy":    "random" if self._random_radio.isChecked() else "loop",
            "grid_size":   self._grid_size_spin.value(),
            "max_start_and_stop_distance": self._max_start_dist_spin.value(),
            "min_transects":               self._min_transects_spin.value(),
            "max_transects_enabled":       not self._max_transects_none_chk.isChecked(),
            "max_transects":               self._max_transects_spin.value(),
            "max_distance":                self._max_distance_spin.value(),
            "min_transect_overlap":        self._min_overlap_spin.value(),
            "number_of_retries":           self._num_retries_spin.value(),
            "target_crs_epsg":             self._epsg_spin.value(),
            "min_transects_per_route":     self._min_per_route_spin.value(),
            "x_offset":                    self._x_offset_spin.value(),
            "y_offset":                    self._y_offset_spin.value(),
            "padding_north":               self._padding_north_spin.value(),
            "padding_east":                self._padding_east_spin.value(),
            "padding_south":               self._padding_south_spin.value(),
            "padding_west":                self._padding_west_spin.value(),
            "seed":                        self._seed_edit.text().strip(),
            "max_number_of_overlapping_transects": self._max_overlapping_spin.value(),
            "max_number_of_flights":       self._max_flights_spin.value(),
            "random_search":               self._random_search_chk.isChecked(),
            "number_of_retries_per_route": self._retries_per_route_spin.value(),
            "target_folder":               self._target_edit.text().strip(),
        }
        for key, value in values.items():
            s.setValue(_SETTINGS_PREFIX + key, value)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _run(self):
        _fix_std_streams()
        from .evaluation_flight_strategy import RandomStrategy, RandomLoopStrategy

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
        self._area_external_path = self._area_file_edit.text().strip() or None
        self._invalid_external_path = self._invalid_file_edit.text().strip() or None

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

    def _on_finished(self, success, msg):
        self._run_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        if not success:
            self._log_edit.append("FAILED.\n\n" + msg)
            QMessageBox.critical(self, "Planning Failed",
                                 "Flight planning failed. See the log below for details.")
            return
        count = int(msg) if msg.isdigit() else 0
        self._log_edit.append(
            f"Completed successfully! Generated {count} valid flight route(s)."
        )
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

        # --- Grid (full) ---
        grid_all_lyr = add_layer(
            os.path.join(target_path, "grid.geojson"), "Grid (All)"
        )
        if grid_all_lyr:
            sym = QgsMarkerSymbol.createSimple(
                {"name": "circle", "color": "#E0E0E0", "size": "1.0",
                 "outline_style": "no"}
            )
            grid_all_lyr.setRenderer(QgsSingleSymbolRenderer(sym))

        # --- Grid (filtered) ---
        grid_lyr = add_layer(
            os.path.join(target_path, "grid_filtered.geojson"), "Grid (Filtered)"
        )
        if grid_lyr:
            sym = QgsMarkerSymbol.createSimple(
                {"name": "circle", "color": "#BDBDBD", "size": "1.5",
                 "outline_style": "no"}
            )
            grid_lyr.setRenderer(QgsSingleSymbolRenderer(sym))

        # --- Transects (valid) ---
        transect_lyr = add_layer(
            os.path.join(target_path, "transects_valids.geojson"), "Transects (Valid)"
        )
        if transect_lyr:
            sym = QgsLineSymbol.createSimple({"color": "#9E9E9E", "width": "0.3"})
            transect_lyr.setRenderer(QgsSingleSymbolRenderer(sym))

        # --- Start Points ---
        start_lyr = add_layer(
            os.path.join(target_path, "startpoints.geojson"), "Start Points"
        )
        if start_lyr:
            sym = QgsMarkerSymbol.createSimple(
                {"name": "triangle", "color": "#1565C0", "size": "5",
                 "outline_color": "#0D47A1", "outline_width": "0.4"}
            )
            start_lyr.setRenderer(QgsSingleSymbolRenderer(sym))

        # --- Flight Routes ---
        # Route GeoJSONs contain mixed geometry (Points + LineStrings).  OGR
        # picks the first geometry type (Point) and silently drops the rest, so
        # we extract the "total-route" LineString and the individual survey
        # transects into separate clean files and load them as sub-layers.
        routes_dir = os.path.join(target_path, "routes", "valid")
        if os.path.isdir(routes_dir):
            lines_dir = os.path.join(target_path, "routes", "valid_lines")
            os.makedirs(lines_dir, exist_ok=True)
            routes_group = group.addGroup("Flight Routes")
            route_files = sorted(
                f for f in os.listdir(routes_dir) if f.endswith(".geojson")
            )
            for idx, fname in enumerate(route_files):
                stem = os.path.splitext(fname)[0]
                src = os.path.join(routes_dir, fname)
                color = _ROUTE_COLORS[idx % len(_ROUTE_COLORS)]
                route_label = stem.replace("_", " ").title()
                route_sub = routes_group.addGroup(route_label)

                # Total-route line
                line_path = _write_route_line(
                    src, os.path.join(lines_dir, fname)
                )
                if line_path:
                    lyr = add_layer(line_path, "Route", route_sub)
                    if lyr:
                        sym = QgsLineSymbol.createSimple(
                            {"color": color, "width": "0.8"}
                        )
                        lyr.setRenderer(QgsSingleSymbolRenderer(sym))

                # Numbered survey transects
                transect_path = _write_route_transects(
                    src,
                    os.path.join(lines_dir, f"{stem}_transects.geojson"),
                )
                if transect_path:
                    t_lyr = add_layer(transect_path, "Transects", route_sub)
                    if t_lyr:
                        sym = QgsLineSymbol.createSimple(
                            {"color": color, "width": "0.4",
                             "line_style": "dash"}
                        )
                        t_lyr.setRenderer(QgsSingleSymbolRenderer(sym))
                        _enable_transect_labels(t_lyr, color)

        # --- External input files (monitoring area / invalid areas) ---
        if self._area_external_path and os.path.exists(self._area_external_path):
            area_lyr = add_layer(self._area_external_path, "Monitoring Area")
            if area_lyr:
                sym = QgsFillSymbol.createSimple(
                    {"color": "33,150,243,40", "outline_color": "#1565C0",
                     "outline_width": "0.8"}
                )
                area_lyr.setRenderer(QgsSingleSymbolRenderer(sym))

        if self._invalid_external_path and os.path.exists(self._invalid_external_path):
            inv_lyr = add_layer(self._invalid_external_path, "Invalid Areas")
            if inv_lyr:
                sym = QgsFillSymbol.createSimple(
                    {"color": "244,67,54,60", "outline_color": "#B71C1C",
                     "outline_width": "0.8"}
                )
                inv_lyr.setRenderer(QgsSingleSymbolRenderer(sym))

        self.iface.mapCanvas().refresh()
        QMessageBox.information(
            self, "Layers Imported",
            f"Results have been imported into QGIS group:\n'{group_name}'"
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._save_settings()
        for tmp in self._temp_files:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
        self._temp_files.clear()
        super().closeEvent(event)
