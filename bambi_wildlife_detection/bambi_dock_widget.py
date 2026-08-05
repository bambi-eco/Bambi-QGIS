# -*- coding: utf-8 -*-
"""
BAMBI Wildlife Detection - Dock Widget
=======================================

This module contains the main dock widget UI for the plugin.
"""

import os
import json
import math
import tempfile
from collections import defaultdict
from typing import Optional, Dict, Any, List
from qgis.PyQt.QtCore import Qt, QThread
from qgis.PyQt.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QFileDialog, QLabel, QProgressBar, QTextEdit, QComboBox,
    QCheckBox, QTabWidget, QMessageBox, QScrollArea,
    QFrame, QListWidget, QListWidgetItem, QDialog,
    QDialogButtonBox, QGridLayout, QToolButton,
    QRadioButton, QButtonGroup, QInputDialog
)
from qgis.PyQt.QtGui import QFont, QColor
from qgis.core import (
    QgsProject, QgsSettings, QgsVectorLayer, QgsCoordinateReferenceSystem,
    QgsFeature, QgsGeometry, QgsPointXY, QgsField, QgsRasterLayer,
    QgsLineSymbol, QgsMarkerSymbol, QgsPalLayerSettings, QgsTextFormat,
    QgsVectorLayerSimpleLabeling, QgsTextBufferSettings, QgsLayerTreeGroup
)
from qgis.PyQt.QtCore import QVariant

from .bambi_processing import BambiProcessor, ProcessingWorker
from .austria_dem_downloader import DEMDownloadWorker, GeoTIFFConversionWorker
from .bambi_click_tool import BambiClickTool
# Built-in calibration presets live in bambi_calibrations.py (Qt-free module
# so they are importable without QGIS, e.g. by the integration tests).
from .bambi_calibrations import THERMAL_CALIBRATIONS, RGB_CALIBRATIONS  # noqa: F401

# Plugin scope for project settings storage
PLUGIN_SCOPE = "BambiWildlifeDetection"


class CorrectionRangeDialog(QDialog):
    """Dialog for adding/editing frame-range specific corrections."""

    def __init__(self, parent=None, correction_data: Optional[Dict[str, Any]] = None, use_degrees: bool = False):
        """Initialize the dialog.

        :param parent: Parent widget
        :param correction_data: Existing correction data for editing (rotations in radians), or None for new
        :param use_degrees: If True, display and accept rotation values in degrees
        """
        super().__init__(parent)
        self.setWindowTitle("Frame-Range Correction")
        self.setMinimumWidth(400)
        self._use_degrees = use_degrees

        self._setup_ui()

        # Load existing data if editing
        if correction_data:
            self._load_data(correction_data)

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # Frame range
        range_group = QGroupBox("Frame Range")
        range_layout = QHBoxLayout(range_group)

        range_layout.addWidget(QLabel("Start:"))
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(0, 999999)
        self.start_frame_spin.setValue(0)
        self.start_frame_spin.setToolTip("First frame index (inclusive)")
        range_layout.addWidget(self.start_frame_spin)

        range_layout.addWidget(QLabel("End:"))
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setRange(0, 999999)
        self.end_frame_spin.setValue(999999)
        self.end_frame_spin.setToolTip("Last frame index (inclusive)")
        range_layout.addWidget(self.end_frame_spin)

        layout.addWidget(range_group)

        # Translation
        trans_group = QGroupBox("Translation (x, y, z)")
        trans_layout = QGridLayout(trans_group)

        trans_layout.addWidget(QLabel("X:"), 0, 0)
        self.trans_x_spin = QDoubleSpinBox()
        self.trans_x_spin.setRange(-100, 100)
        self.trans_x_spin.setValue(0.0)
        self.trans_x_spin.setDecimals(3)
        trans_layout.addWidget(self.trans_x_spin, 0, 1)

        trans_layout.addWidget(QLabel("Y:"), 0, 2)
        self.trans_y_spin = QDoubleSpinBox()
        self.trans_y_spin.setRange(-100, 100)
        self.trans_y_spin.setValue(0.0)
        self.trans_y_spin.setDecimals(3)
        trans_layout.addWidget(self.trans_y_spin, 0, 3)

        trans_layout.addWidget(QLabel("Z:"), 0, 4)
        self.trans_z_spin = QDoubleSpinBox()
        self.trans_z_spin.setRange(-100, 100)
        self.trans_z_spin.setValue(0.0)
        self.trans_z_spin.setDecimals(3)
        trans_layout.addWidget(self.trans_z_spin, 0, 5)

        layout.addWidget(trans_group)

        # Rotation - configure based on unit
        unit_str = "deg" if self._use_degrees else "rad"
        rot_group = QGroupBox(f"Rotation (pitch, roll, yaw) [{unit_str}]")
        rot_layout = QGridLayout(rot_group)

        if self._use_degrees:
            rot_range = (-180, 180)
            rot_decimals = 3
            rot_step = 1.0
        else:
            rot_range = (-6.28319, 6.28319)  # -2π to 2π
            rot_decimals = 5
            rot_step = 0.01

        rot_layout.addWidget(QLabel("Pitch:"), 0, 0)
        self.rot_x_spin = QDoubleSpinBox()
        self.rot_x_spin.setRange(*rot_range)
        self.rot_x_spin.setValue(0.0)
        self.rot_x_spin.setDecimals(rot_decimals)
        self.rot_x_spin.setSingleStep(rot_step)
        rot_layout.addWidget(self.rot_x_spin, 0, 1)

        rot_layout.addWidget(QLabel("Roll:"), 0, 2)
        self.rot_y_spin = QDoubleSpinBox()
        self.rot_y_spin.setRange(*rot_range)
        self.rot_y_spin.setValue(0.0)
        self.rot_y_spin.setDecimals(rot_decimals)
        self.rot_y_spin.setSingleStep(rot_step)
        rot_layout.addWidget(self.rot_y_spin, 0, 3)

        rot_layout.addWidget(QLabel("Yaw:"), 0, 4)
        self.rot_z_spin = QDoubleSpinBox()
        self.rot_z_spin.setRange(*rot_range)
        self.rot_z_spin.setValue(0.0)
        self.rot_z_spin.setDecimals(rot_decimals)
        self.rot_z_spin.setSingleStep(rot_step)
        rot_layout.addWidget(self.rot_z_spin, 0, 5)

        layout.addWidget(rot_group)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _load_data(self, data: Dict[str, Any]):
        """Load existing correction data into the dialog.

        :param data: Correction data dictionary (rotations in radians)
        """
        self.start_frame_spin.setValue(data.get('start', 0))
        self.end_frame_spin.setValue(data.get('end', 999999))

        trans = data.get('translation', {})
        self.trans_x_spin.setValue(trans.get('x', 0.0))
        self.trans_y_spin.setValue(trans.get('y', 0.0))
        self.trans_z_spin.setValue(trans.get('z', 0.0))

        rot = data.get('rotation', {})
        rot_x = rot.get('x', 0.0)
        rot_y = rot.get('y', 0.0)
        rot_z = rot.get('z', 0.0)

        # Convert from radians to degrees for display if needed
        if self._use_degrees:
            rot_x = math.degrees(rot_x)
            rot_y = math.degrees(rot_y)
            rot_z = math.degrees(rot_z)

        self.rot_x_spin.setValue(rot_x)
        self.rot_y_spin.setValue(rot_y)
        self.rot_z_spin.setValue(rot_z)

    def get_correction_data(self) -> Dict[str, Any]:
        """Get the correction data from the dialog.

        :return: Dictionary with correction data (rotations always in radians)
        """
        rot_x = self.rot_x_spin.value()
        rot_y = self.rot_y_spin.value()
        rot_z = self.rot_z_spin.value()

        # Convert from degrees to radians if needed
        if self._use_degrees:
            rot_x = math.radians(rot_x)
            rot_y = math.radians(rot_y)
            rot_z = math.radians(rot_z)

        return {
            "start": self.start_frame_spin.value(),
            "end": self.end_frame_spin.value(),
            "translation": {
                "x": self.trans_x_spin.value(),
                "y": self.trans_y_spin.value(),
                "z": self.trans_z_spin.value()
            },
            "rotation": {
                "x": rot_x,
                "y": rot_y,
                "z": rot_z
            }
        }


class FrameRangeDialog(QDialog):
    """Dialog for selecting a frame index range before importing per-frame QGIS layers."""

    def __init__(self, parent, min_idx: int, max_idx: int, total: int,
                 item_label: str = "frames"):
        super().__init__(parent)
        self.setWindowTitle("Select Frame Range")
        self.setMinimumWidth(350)

        self._min_idx = min_idx
        self._max_idx = max_idx

        layout = QVBoxLayout(self)

        info = QLabel(f"Available: {item_label} {min_idx}–{max_idx}  ({total} total)")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)

        self._all_radio = QRadioButton("All frames")
        self._all_radio.setChecked(True)
        self._range_radio = QRadioButton("Frame range:")

        self._btn_group = QButtonGroup(self)
        self._btn_group.addButton(self._all_radio)
        self._btn_group.addButton(self._range_radio)

        layout.addWidget(self._all_radio)

        range_row = QHBoxLayout()
        range_row.addWidget(self._range_radio)
        self._start_spin = QSpinBox()
        self._start_spin.setRange(min_idx, max_idx)
        self._start_spin.setValue(min_idx)
        self._start_spin.setEnabled(False)
        self._end_spin = QSpinBox()
        self._end_spin.setRange(min_idx, max_idx)
        self._end_spin.setValue(max_idx)
        self._end_spin.setEnabled(False)
        range_row.addWidget(QLabel("from"))
        range_row.addWidget(self._start_spin)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self._end_spin)
        layout.addLayout(range_row)

        self._all_radio.toggled.connect(
            lambda checked: (
                self._start_spin.setEnabled(not checked),
                self._end_spin.setEnabled(not checked),
            )
        )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_range(self):
        """Return ``(start, end)`` — clamped and always start <= end."""
        if self._all_radio.isChecked():
            return self._min_idx, self._max_idx
        start = self._start_spin.value()
        end = self._end_spin.value()
        if start > end:
            start, end = end, start
        return start, end


class BambiDockWidget(QDockWidget):
    """Main dock widget for the BAMBI Wildlife Detection plugin."""

    def __init__(self, iface, parent=None):
        """Initialize the dock widget.

        :param iface: QGIS interface instance
        :param parent: Parent widget
        """
        super().__init__("Bambi - QGIS Integration", parent)
        self.iface = iface
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        # Processing state
        self.processor = BambiProcessor()
        self.worker = None
        self.worker_thread = None

        # Track if initial check has been done
        self._initial_check_done = False

        # Flag to prevent recursive saves during config loading
        self._loading_config = False

        # Inspector map tools (lazy-created on first activation)
        self._click_tool = None
        self._fov_click_tool = None
        self._fov_georef_click_tool = None

        # Toolbar QAction references — set via set_inspector_actions()
        self._inspector_action = None
        self._fov_inspector_action = None
        self._fov_georef_inspector_action = None

        # Setup UI
        self.setup_ui()

        # Snapshot pristine widget state so a new project can reset to defaults
        self._snapshot_default_config()

        # Connect to project signals for config persistence
        self._connect_project_signals()

        # Keep the inspector button in sync when the user switches to another map tool
        self.iface.mapCanvas().mapToolSet.connect(self._on_map_tool_changed)

        # Load config if project already has saved config
        self._loading_config = True
        try:
            self.load_config_from_project()
        finally:
            self._loading_config = False

    def showEvent(self, event):
        """Handle widget show event to check for existing layers."""
        super().showEvent(event)

        # Check for existing QGIS layers when first shown
        if not self._initial_check_done:
            self._initial_check_done = True
            self._check_existing_qgis_layers()

            # Also check target folder if already set
            target_folder = self.target_folder_edit.text().strip()
            if target_folder and os.path.isdir(target_folder):
                self._check_existing_outputs(target_folder)

    def setup_ui(self):
        """Setup the user interface."""
        # Main widget and layout
        main_widget = QWidget()
        self.setWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Create scrollable area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        # Create main tab widget
        main_tabs = QTabWidget()
        scroll_layout.addWidget(main_tabs)

        # =====================================================================
        # MAIN TAB 1: INPUT
        # =====================================================================
        # Thermal Video inputs
        input_tab = QWidget()
        input_layout = QVBoxLayout(input_tab)
        main_tabs.addTab(input_tab, "Input")

        # --- Flights ---
        # A QGIS project can hold several flights; exactly one is active, so
        # everything downstream still reads a single configuration (§10.2).
        flight_group = QGroupBox("Flight")
        flight_box = QVBoxLayout(flight_group)
        flight_row = QHBoxLayout()
        flight_row.addWidget(QLabel("Active:"))

        self.flight_combo = QComboBox()
        self.flight_combo.setToolTip(
            "Each flight has its own target folder, configuration and layer "
            "group. Switching flights loads that flight's settings.")
        self.flight_combo.currentIndexChanged.connect(self._on_flight_changed)
        flight_row.addWidget(self.flight_combo, 1)

        self.flight_add_btn = QPushButton("+")
        self.flight_add_btn.setFixedWidth(30)
        self.flight_add_btn.setToolTip("Add another flight to this project")
        self.flight_add_btn.clicked.connect(self.add_flight)
        flight_row.addWidget(self.flight_add_btn)

        self.flight_rename_btn = QPushButton("Rename…")
        self.flight_rename_btn.setToolTip(
            "Rename this flight. Its QGIS layer group is renamed with it.")
        self.flight_rename_btn.clicked.connect(self.rename_flight)
        flight_row.addWidget(self.flight_rename_btn)
        flight_box.addLayout(flight_row)
        input_layout.addWidget(flight_group)

        # --- Input mode toggle ---
        self.video_mode_check = QCheckBox("Video Mode")
        self.video_mode_check.setChecked(True)
        self.video_mode_check.setToolTip(
            "Checked: use video files + SRT files as input.\n"
            "Unchecked: use a folder of still photos as input.")

        self._input_mode_info_btn = QToolButton()
        self._input_mode_info_btn.setText("?")
        self._input_mode_info_btn.setFixedSize(20, 20)
        self._input_mode_info_btn.setStyleSheet(
            "QToolButton {"
            "  border-radius: 10px;"
            "  border: 1px solid palette(mid);"
            "  background: palette(button);"
            "  font-weight: bold;"
            "  font-size: 11px;"
            "}"
            "QToolButton:hover { background: palette(light); }"
            "QToolButton:pressed { background: palette(mid); }"
        )
        self._input_mode_info_btn.clicked.connect(self._show_input_mode_info)

        self.embedded_srt_check = QCheckBox("Embedded SRT")
        self.embedded_srt_check.setChecked(False)
        self.embedded_srt_check.setVisible(True)  # video mode is on by default
        self.embedded_srt_check.setToolTip(
            "Checked: SRT timing data is embedded in the video file (newer DJI drones).\n"
            "The SRT stream is extracted automatically — no separate .srt files needed.\n"
            "Unchecked: SRT files are provided as separate files alongside the video."
        )
        self.embedded_srt_check.stateChanged.connect(self._on_embedded_srt_changed)

        input_mode_row = QHBoxLayout()
        input_mode_row.setContentsMargins(0, 0, 0, 0)
        input_mode_row.addWidget(self.video_mode_check)
        input_mode_row.addWidget(self.embedded_srt_check)
        input_mode_row.addStretch()
        input_mode_row.addWidget(self._input_mode_info_btn)
        input_layout.addLayout(input_mode_row)

        # ── Video inputs container ─────────────────────────────────────────
        self.video_inputs_widget = QWidget()
        video_inputs_layout = QVBoxLayout(self.video_inputs_widget)
        video_inputs_layout.setContentsMargins(0, 0, 0, 0)
        video_inputs_layout.setSpacing(4)

        thermal_group = QGroupBox("Thermal Video Inputs")
        thermal_layout = QFormLayout(thermal_group)

        self.thermal_video_paths_edit = QLineEdit()
        self.thermal_video_paths_edit.setPlaceholderText("Comma-separated paths to thermal video files (_T_)")
        thermal_video_browse_btn = QPushButton("Browse...")
        thermal_video_browse_btn.clicked.connect(self.browse_thermal_videos)
        thermal_video_row = QHBoxLayout()
        thermal_video_row.addWidget(self.thermal_video_paths_edit)
        thermal_video_row.addWidget(thermal_video_browse_btn)
        thermal_layout.addRow("Videos:", thermal_video_row)

        self._thermal_srt_field = QWidget()
        _t_srt_lay = QHBoxLayout(self._thermal_srt_field)
        _t_srt_lay.setContentsMargins(0, 0, 0, 0)
        self.thermal_srt_paths_edit = QLineEdit()
        self.thermal_srt_paths_edit.setPlaceholderText("Comma-separated paths to thermal SRT files (_T_)")
        thermal_srt_browse_btn = QPushButton("Browse...")
        thermal_srt_browse_btn.clicked.connect(self.browse_thermal_srts)
        _t_srt_lay.addWidget(self.thermal_srt_paths_edit)
        _t_srt_lay.addWidget(thermal_srt_browse_btn)
        thermal_layout.addRow("SRT Files:", self._thermal_srt_field)
        self._thermal_srt_label = thermal_layout.labelForField(self._thermal_srt_field)

        thermal_calib_container = QWidget()
        thermal_calib_vbox = QVBoxLayout(thermal_calib_container)
        thermal_calib_vbox.setContentsMargins(0, 0, 0, 0)
        thermal_calib_vbox.setSpacing(2)

        self.thermal_calib_preset_combo = QComboBox()
        self.thermal_calib_preset_combo.addItem("Custom file...")
        for name in THERMAL_CALIBRATIONS:
            self.thermal_calib_preset_combo.addItem(name)
        thermal_calib_vbox.addWidget(self.thermal_calib_preset_combo)

        self.thermal_calib_file_row = QWidget()
        thermal_calib_file_layout = QHBoxLayout(self.thermal_calib_file_row)
        thermal_calib_file_layout.setContentsMargins(0, 0, 0, 0)
        self.thermal_calibration_path_edit = QLineEdit()
        self.thermal_calibration_path_edit.setPlaceholderText("Path to T_calib.json")
        thermal_calib_browse_btn = QPushButton("Browse...")
        thermal_calib_browse_btn.clicked.connect(self.browse_thermal_calibration)
        thermal_calib_file_layout.addWidget(self.thermal_calibration_path_edit)
        thermal_calib_file_layout.addWidget(thermal_calib_browse_btn)
        thermal_calib_vbox.addWidget(self.thermal_calib_file_row)

        self.thermal_calib_preset_combo.currentIndexChanged.connect(
            self._on_thermal_calib_preset_changed)
        thermal_layout.addRow("Calibration:", thermal_calib_container)

        video_inputs_layout.addWidget(thermal_group)

        # RGB Video inputs
        rgb_group = QGroupBox("RGB Video Inputs")
        rgb_layout = QFormLayout(rgb_group)

        self.rgb_video_paths_edit = QLineEdit()
        self.rgb_video_paths_edit.setPlaceholderText("Comma-separated paths to RGB video files (_W_ or _V_)")
        rgb_video_browse_btn = QPushButton("Browse...")
        rgb_video_browse_btn.clicked.connect(self.browse_rgb_videos)
        rgb_video_row = QHBoxLayout()
        rgb_video_row.addWidget(self.rgb_video_paths_edit)
        rgb_video_row.addWidget(rgb_video_browse_btn)
        rgb_layout.addRow("Videos:", rgb_video_row)

        self._rgb_srt_field = QWidget()
        _r_srt_lay = QHBoxLayout(self._rgb_srt_field)
        _r_srt_lay.setContentsMargins(0, 0, 0, 0)
        self.rgb_srt_paths_edit = QLineEdit()
        self.rgb_srt_paths_edit.setPlaceholderText("Comma-separated paths to RGB SRT files (_W_ or _V_)")
        rgb_srt_browse_btn = QPushButton("Browse...")
        rgb_srt_browse_btn.clicked.connect(self.browse_rgb_srts)
        _r_srt_lay.addWidget(self.rgb_srt_paths_edit)
        _r_srt_lay.addWidget(rgb_srt_browse_btn)
        rgb_layout.addRow("SRT Files:", self._rgb_srt_field)
        self._rgb_srt_label = rgb_layout.labelForField(self._rgb_srt_field)

        rgb_calib_container = QWidget()
        rgb_calib_vbox = QVBoxLayout(rgb_calib_container)
        rgb_calib_vbox.setContentsMargins(0, 0, 0, 0)
        rgb_calib_vbox.setSpacing(2)

        self.rgb_calib_preset_combo = QComboBox()
        self.rgb_calib_preset_combo.addItem("Custom file...")
        for name in RGB_CALIBRATIONS:
            self.rgb_calib_preset_combo.addItem(name)
        rgb_calib_vbox.addWidget(self.rgb_calib_preset_combo)

        self.rgb_calib_file_row = QWidget()
        rgb_calib_file_layout = QHBoxLayout(self.rgb_calib_file_row)
        rgb_calib_file_layout.setContentsMargins(0, 0, 0, 0)
        self.rgb_calibration_path_edit = QLineEdit()
        self.rgb_calibration_path_edit.setPlaceholderText("Path to W_calib.json")
        rgb_calib_browse_btn = QPushButton("Browse...")
        rgb_calib_browse_btn.clicked.connect(self.browse_rgb_calibration)
        rgb_calib_file_layout.addWidget(self.rgb_calibration_path_edit)
        rgb_calib_file_layout.addWidget(rgb_calib_browse_btn)
        rgb_calib_vbox.addWidget(self.rgb_calib_file_row)

        self.rgb_calib_preset_combo.currentIndexChanged.connect(
            self._on_rgb_calib_preset_changed)
        rgb_layout.addRow("Calibration:", rgb_calib_container)

        video_inputs_layout.addWidget(rgb_group)
        input_layout.addWidget(self.video_inputs_widget)

        # ── Photo inputs container ─────────────────────────────────────────
        self.photo_inputs_widget = QWidget()
        self.photo_inputs_widget.setVisible(False)
        photo_inputs_layout = QVBoxLayout(self.photo_inputs_widget)
        photo_inputs_layout.setContentsMargins(0, 0, 0, 0)
        photo_inputs_layout.setSpacing(4)

        # Thermal photo group
        thermal_photo_group = QGroupBox("Thermal Photo Inputs")
        thermal_photo_layout = QFormLayout(thermal_photo_group)

        self.thermal_photo_dir_edit = QLineEdit()
        self.thermal_photo_dir_edit.setPlaceholderText("Folder containing thermal photos")
        thermal_photo_browse_btn = QPushButton("Browse...")
        thermal_photo_browse_btn.clicked.connect(self.browse_thermal_photo_dir)
        thermal_photo_dir_row = QHBoxLayout()
        thermal_photo_dir_row.addWidget(self.thermal_photo_dir_edit)
        thermal_photo_dir_row.addWidget(thermal_photo_browse_btn)
        thermal_photo_layout.addRow("Photo Folder:", thermal_photo_dir_row)

        self.thermal_photo_filter_check = QCheckBox()
        self.thermal_photo_filter_check.setToolTip(
            "Only process images with _T_ or _T in the filename")
        thermal_photo_layout.addRow("Filter (_T_ / _T):", self.thermal_photo_filter_check)

        thermal_photo_calib_container = QWidget()
        thermal_photo_calib_vbox = QVBoxLayout(thermal_photo_calib_container)
        thermal_photo_calib_vbox.setContentsMargins(0, 0, 0, 0)
        thermal_photo_calib_vbox.setSpacing(2)
        self.thermal_photo_calib_preset_combo = QComboBox()
        self.thermal_photo_calib_preset_combo.addItem("Custom file...")
        for name in THERMAL_CALIBRATIONS:
            self.thermal_photo_calib_preset_combo.addItem(name)
        thermal_photo_calib_vbox.addWidget(self.thermal_photo_calib_preset_combo)
        self.thermal_photo_calib_file_row = QWidget()
        thermal_photo_calib_file_layout = QHBoxLayout(self.thermal_photo_calib_file_row)
        thermal_photo_calib_file_layout.setContentsMargins(0, 0, 0, 0)
        self.thermal_photo_calibration_path_edit = QLineEdit()
        self.thermal_photo_calibration_path_edit.setPlaceholderText("Path to T_calib.json")
        thermal_photo_calib_browse_btn = QPushButton("Browse...")
        thermal_photo_calib_browse_btn.clicked.connect(self.browse_thermal_photo_calibration)
        thermal_photo_calib_file_layout.addWidget(self.thermal_photo_calibration_path_edit)
        thermal_photo_calib_file_layout.addWidget(thermal_photo_calib_browse_btn)
        thermal_photo_calib_vbox.addWidget(self.thermal_photo_calib_file_row)
        self.thermal_photo_calib_preset_combo.currentIndexChanged.connect(
            self._on_thermal_photo_calib_preset_changed)
        thermal_photo_layout.addRow("Calibration:", thermal_photo_calib_container)
        photo_inputs_layout.addWidget(thermal_photo_group)

        # RGB photo group
        rgb_photo_group = QGroupBox("RGB Photo Inputs")
        rgb_photo_layout = QFormLayout(rgb_photo_group)

        self.rgb_photo_dir_edit = QLineEdit()
        self.rgb_photo_dir_edit.setPlaceholderText("Folder containing RGB photos")
        rgb_photo_browse_btn = QPushButton("Browse...")
        rgb_photo_browse_btn.clicked.connect(self.browse_rgb_photo_dir)
        rgb_photo_dir_row = QHBoxLayout()
        rgb_photo_dir_row.addWidget(self.rgb_photo_dir_edit)
        rgb_photo_dir_row.addWidget(rgb_photo_browse_btn)
        rgb_photo_layout.addRow("Photo Folder:", rgb_photo_dir_row)

        self.rgb_photo_filter_check = QCheckBox()
        self.rgb_photo_filter_check.setToolTip(
            "Only process images with _W_, _W, _V_ or _V in the filename")
        rgb_photo_layout.addRow("Filter (_W_ / _W / _V_ / _V):", self.rgb_photo_filter_check)

        rgb_photo_calib_container = QWidget()
        rgb_photo_calib_vbox = QVBoxLayout(rgb_photo_calib_container)
        rgb_photo_calib_vbox.setContentsMargins(0, 0, 0, 0)
        rgb_photo_calib_vbox.setSpacing(2)
        self.rgb_photo_calib_preset_combo = QComboBox()
        self.rgb_photo_calib_preset_combo.addItem("Custom file...")
        for name in RGB_CALIBRATIONS:
            self.rgb_photo_calib_preset_combo.addItem(name)
        rgb_photo_calib_vbox.addWidget(self.rgb_photo_calib_preset_combo)
        self.rgb_photo_calib_file_row = QWidget()
        rgb_photo_calib_file_layout = QHBoxLayout(self.rgb_photo_calib_file_row)
        rgb_photo_calib_file_layout.setContentsMargins(0, 0, 0, 0)
        self.rgb_photo_calibration_path_edit = QLineEdit()
        self.rgb_photo_calibration_path_edit.setPlaceholderText("Path to W_calib.json")
        rgb_photo_calib_browse_btn = QPushButton("Browse...")
        rgb_photo_calib_browse_btn.clicked.connect(self.browse_rgb_photo_calibration)
        rgb_photo_calib_file_layout.addWidget(self.rgb_photo_calibration_path_edit)
        rgb_photo_calib_file_layout.addWidget(rgb_photo_calib_browse_btn)
        rgb_photo_calib_vbox.addWidget(self.rgb_photo_calib_file_row)
        self.rgb_photo_calib_preset_combo.currentIndexChanged.connect(
            self._on_rgb_photo_calib_preset_changed)
        rgb_photo_layout.addRow("Calibration:", rgb_photo_calib_container)
        photo_inputs_layout.addWidget(rgb_photo_group)

        input_layout.addWidget(self.photo_inputs_widget)

        self.video_mode_check.stateChanged.connect(self._on_input_mode_changed)

        # Common inputs
        common_group = QGroupBox("Flight Data")
        common_layout = QFormLayout(common_group)

        self.airdata_path_edit = QLineEdit()
        self.airdata_path_edit.setPlaceholderText("Path to AirData CSV file")
        airdata_browse_btn = QPushButton("Browse...")
        airdata_browse_btn.clicked.connect(self.browse_airdata)

        airdata_label = QLabel(
            '<a href="https://www.airdata.com">https://www.airdata.com</a>'
        )
        airdata_label.setOpenExternalLinks(True)
        airdata_label.setWordWrap(True)
        airdata_label.setStyleSheet("color: blue; font-size: 10px;")

        self._airdata_exif_info_label = QLabel(
            "If no AirData CSV is provided, the plugin will attempt to reconstruct "
            "the flight log row by row from the GPS position, altitude and gimbal "
            "orientation stored in each image's EXIF / XMP data. "
            "An error is raised if neither source is available."
        )
        self._airdata_exif_info_label.setWordWrap(True)
        self._airdata_exif_info_label.setStyleSheet("color: gray; font-size: 10px;")
        self._airdata_exif_info_label.setVisible(False)

        airdata_field = QWidget()
        airdata_field_layout = QVBoxLayout(airdata_field)
        airdata_field_layout.setContentsMargins(0, 0, 0, 0)
        airdata_field_layout.setSpacing(0)
        airdata_row = QHBoxLayout()
        airdata_row.setContentsMargins(0, 0, 0, 0)
        airdata_row.addWidget(self.airdata_path_edit)
        airdata_row.addWidget(airdata_browse_btn)
        airdata_field_layout.addLayout(airdata_row)
        airdata_field_layout.addWidget(airdata_label)
        airdata_field_layout.addWidget(self._airdata_exif_info_label)
        common_layout.addRow("AirData CSV:", airdata_field)

        self.correction_path_edit = QLineEdit()
        self.correction_path_edit.setPlaceholderText("Path to correction.json (auto-detected)")
        correction_browse_btn = QPushButton("Browse...")
        correction_browse_btn.clicked.connect(self.browse_correction)
        correction_row = QHBoxLayout()
        correction_row.addWidget(self.correction_path_edit)
        correction_row.addWidget(correction_browse_btn)
        common_layout.addRow("Correction:", correction_row)

        input_layout.addWidget(common_group)

        # Output configuration
        output_group = QGroupBox("Output Configuration")
        output_layout = QFormLayout(output_group)

        self.target_folder_edit = QLineEdit()
        self.target_folder_edit.setPlaceholderText("Target folder for all outputs")
        self.target_folder_edit.editingFinished.connect(self._on_target_folder_changed)
        target_browse_btn = QPushButton("Browse...")
        target_browse_btn.clicked.connect(self.browse_target_folder)
        # Offered only for folders holding 5.x outputs that have not been
        # migrated yet; see _refresh_migrate_button.
        self.migrate_btn = QPushButton("Migrate 5.x…")
        self.migrate_btn.setToolTip(
            "Import the outputs of this folder into the 6.0 store.\n"
            "The existing files are only read, never modified."
        )
        self.migrate_btn.clicked.connect(self.migrate_legacy_project)
        self.migrate_btn.setVisible(False)
        target_row = QHBoxLayout()
        target_row.addWidget(self.target_folder_edit)
        target_row.addWidget(target_browse_btn)
        target_row.addWidget(self.migrate_btn)
        output_layout.addRow("Target Folder:", target_row)

        # 6.0 writes the GeoPackage store; the text files are kept alongside it
        # for one release. They are not decoration yet — several tools still
        # read them — so the tooltip says what stops working.
        self.legacy_text_outputs_check = QCheckBox(
            "Also write legacy text outputs (.txt / .csv)")
        self.legacy_text_outputs_check.setChecked(True)
        self.legacy_text_outputs_check.setToolTip(
            "6.0 stores detections, geo-referencing and tracks in "
            "bambi_{t,w}/*.gpkg. The 5.x text files are written alongside them "
            "so external scripts keep working.\n\n"
            "Turning this off also stops the Video Creator, the click tool and "
            "the QGIS layers from finding their input, until those read the "
            "store directly. Leave it on unless you know you do not need them."
        )
        output_layout.addRow("", self.legacy_text_outputs_check)

        self.target_crs_edit = QLineEdit()
        self.target_crs_edit.setPlaceholderText("EPSG:32633 (UTM CRS required)")
        self.target_crs_edit.setText("EPSG:32633")
        self.target_crs_edit.setToolTip(
            "Enter a UTM CRS (e.g., EPSG:32632, EPSG:32633).\n"
            "UTM zones 01-60 for Northern (EPSG:32601-32660) or Southern (EPSG:32701-32760) hemisphere.\n"
            "Click 'Auto' to detect from DEM metadata or AirData GPS coordinates."
        )
        self.target_crs_edit.editingFinished.connect(self._validate_crs_input)

        self.target_crs_auto_btn = QPushButton("Auto")
        self.target_crs_auto_btn.setToolTip(
            "Auto-detect CRS from DEM metadata (dem.json) or AirData CSV GPS coordinates"
        )
        self.target_crs_auto_btn.setFixedWidth(50)
        self.target_crs_auto_btn.clicked.connect(self._auto_detect_crs)

        crs_row = QHBoxLayout()
        crs_row.addWidget(self.target_crs_edit)
        crs_row.addWidget(self.target_crs_auto_btn)
        output_layout.addRow("Target CRS:", crs_row)

        crs_info_label = QLabel(
            "Only UTM CRS is supported (EPSG:32601-32660 N / EPSG:32701-32760 S). "
            "A metric CRS is required for the subsequent processing."
        )
        crs_info_label.setWordWrap(True)
        crs_info_label.setStyleSheet("color: gray; font-size: 10px;")
        output_layout.addRow("", crs_info_label)

        input_layout.addWidget(output_group)

        # Geo-referencing data with tabs for different input methods
        geo_group = QGroupBox("Geo-referencing Data")
        geo_main_layout = QVBoxLayout(geo_group)

        # Tabs for different input methods
        self.geo_input_tabs = QTabWidget()
        geo_main_layout.addWidget(self.geo_input_tabs)

        # ----- Tab 1: Mesh Input (manual GLTF/JSON selection) -----
        mesh_input_tab = QWidget()
        mesh_input_layout = QFormLayout(mesh_input_tab)

        mesh_info_label = QLabel(
            "Select existing GLTF/GLB mesh and JSON metadata files directly."
        )
        mesh_info_label.setWordWrap(True)
        mesh_input_layout.addRow(mesh_info_label)

        # DEM GLTF file selection
        self.dem_path_edit = QLineEdit()
        self.dem_path_edit.setPlaceholderText("Path to DEM GLTF file")
        dem_browse_btn = QPushButton("Browse...")
        dem_browse_btn.clicked.connect(self.browse_dem)
        dem_row = QHBoxLayout()
        dem_row.addWidget(self.dem_path_edit)
        dem_row.addWidget(dem_browse_btn)
        mesh_input_layout.addRow("DEM (GLTF):", dem_row)

        # DEM metadata JSON file selection
        self.dem_metadata_path_edit = QLineEdit()
        self.dem_metadata_path_edit.setPlaceholderText("Path to DEM metadata JSON (auto-detected from DEM)")
        dem_meta_browse_btn = QPushButton("Browse...")
        dem_meta_browse_btn.clicked.connect(self.browse_dem_metadata)
        dem_meta_row = QHBoxLayout()
        dem_meta_row.addWidget(self.dem_metadata_path_edit)
        dem_meta_row.addWidget(dem_meta_browse_btn)
        mesh_input_layout.addRow("DEM Metadata:", dem_meta_row)

        self.geo_input_tabs.addTab(mesh_input_tab, "Mesh Input")

        # ----- Tab 2: GeoTIFF Input (convert arbitrary GeoTIFF) -----
        geotiff_input_tab = QWidget()
        geotiff_input_layout = QFormLayout(geotiff_input_tab)

        # GeoTIFF file selection
        self.geotiff_input_path_edit = QLineEdit()
        self.geotiff_input_path_edit.setPlaceholderText("Select a GeoTIFF DEM file to convert")
        geotiff_browse_btn = QPushButton("Browse...")
        geotiff_browse_btn.clicked.connect(self.browse_geotiff_input)
        geotiff_browse_row = QHBoxLayout()
        geotiff_browse_row.addWidget(self.geotiff_input_path_edit)
        geotiff_browse_row.addWidget(geotiff_browse_btn)
        geotiff_input_layout.addRow("GeoTIFF File:", geotiff_browse_row)

        # Simplification factor
        self.geotiff_simplify_spin = QSpinBox()
        self.geotiff_simplify_spin.setRange(1, 20)
        self.geotiff_simplify_spin.setValue(2)
        self.geotiff_simplify_spin.setToolTip(
            "Mesh simplification factor. Higher values = smaller file size but less detail.\n"
            "1 = full resolution, 2 = half resolution, etc."
        )
        geotiff_input_layout.addRow("Simplify Factor:", self.geotiff_simplify_spin)

        # Source CRS override
        self.geotiff_source_crs_edit = QLineEdit()
        self.geotiff_source_crs_edit.setPlaceholderText("Auto-detect (e.g. EPSG:32633)")
        self.geotiff_source_crs_edit.setToolTip(
            "Override the input CRS of the GeoTIFF.\n"
            "Leave empty to use the CRS stored in the file.\n"
            "Use this when the file's embedded CRS metadata is wrong\n"
            "(e.g. file contains SWEREF99TM / UTM Zone 33 data but is\n"
            "labelled EPSG:32634)."
        )
        geotiff_input_layout.addRow("Source CRS:", self.geotiff_source_crs_edit)

        # Convert button
        convert_row = QHBoxLayout()
        self.geotiff_convert_btn = QPushButton("Convert GeoTIFF to Mesh")
        self.geotiff_convert_btn.setToolTip(
            "Convert the selected GeoTIFF to GLTF mesh format.\n"
            "Output files will be saved in the Target Folder."
        )
        self.geotiff_convert_btn.clicked.connect(self.convert_geotiff_to_mesh)
        convert_row.addWidget(self.geotiff_convert_btn)
        convert_row.addStretch()
        geotiff_input_layout.addRow("", convert_row)

        self.geo_input_tabs.addTab(geotiff_input_tab, "GeoTIFF Input")

        # ----- Tab 3: Auto-Download (Austria BEV service) -----
        auto_download_tab = QWidget()
        auto_download_layout = QFormLayout(auto_download_tab)

        # Padding setting
        self.dem_padding_spin = QSpinBox()
        self.dem_padding_spin.setRange(0, 500)
        self.dem_padding_spin.setValue(30)
        self.dem_padding_spin.setSuffix(" m")
        self.dem_padding_spin.setToolTip("Padding around flight area in meters")
        auto_download_layout.addRow("Padding:", self.dem_padding_spin)

        # Download button
        download_row = QHBoxLayout()
        self.dem_download_btn = QPushButton("Download DEM (Austria)")
        self.dem_download_btn.setToolTip(
            "Download DEM from Austrian BEV service based on AirData CSV GPS coordinates. "
            "Requires AirData CSV to be selected. Uses Austria-wide 1m ALS-DTM dataset. "
            "Note, that this may take some time, since individual tiles are 10GB+ "
            "and it may be necessary to download multiple map tiles. "
            "Downloaded to \"<user profile>/.cache/austria_dem\""
        )
        self.dem_download_btn.clicked.connect(self.download_austria_dem)
        download_row.addWidget(self.dem_download_btn)
        download_row.addStretch()
        auto_download_layout.addRow("", download_row)

        # Info label
        download_info_label = QLabel(
            "Downloads DEM tiles from the Austrian BEV service based on GPS coordinates\n"
            "in the AirData CSV file. The area is determined automatically with the\n"
            "specified padding around the flight path.\n"
            "Note, that this may take some time, since individual tiles are 10GB+ "
            "and it may be necessary to download multiple map tiles.\n"
            "Downloaded to \"<user profile>/.cache/austria_dem\""
        )
        download_info_label.setWordWrap(True)
        auto_download_layout.addRow("", download_info_label)

        self.geo_input_tabs.addTab(auto_download_tab, "Auto-Download")

        # ----- Tab 4: Flat Surface -----
        flat_surface_tab = QWidget()
        flat_surface_layout = QFormLayout(flat_surface_tab)

        flat_surf_info = QLabel(
            "Generates a flat GLB mesh and its companion JSON for aquatic/marine surveys "
            "(e.g. sharks near the water surface).\n\n"
            "Origin and extent are derived automatically from the AirData CSV or SRT files "
            "selected in the input section. The generated files are written to the Target "
            "Folder and set as the Mesh Input paths, so they are used like any other DEM."
        )
        flat_surf_info.setWordWrap(True)
        flat_surface_layout.addRow(flat_surf_info)

        flat_surf_elev_row = QHBoxLayout()
        self.flat_surface_spin = QDoubleSpinBox()
        self.flat_surface_spin.setRange(-20000.0, 20000.0)
        self.flat_surface_spin.setSingleStep(1.0)
        self.flat_surface_spin.setDecimals(1)
        self.flat_surface_spin.setValue(0.0)
        self.flat_surface_spin.setSuffix(" m MSL")
        self.flat_surface_spin.setToolTip(
            "Elevation of the flat projection surface in metres above mean sea level.\n"
            "Use 0.0 for sea-surface surveys (sharks, manta rays, etc.)."
        )
        flat_surf_elev_row.addWidget(self.flat_surface_spin)
        flat_surf_elev_row.addStretch()
        flat_surface_layout.addRow("Elevation (MSL):", flat_surf_elev_row)

        flat_surf_pad_row = QHBoxLayout()
        self.flat_surface_padding_auto_check = QCheckBox("Auto (50 m)")
        self.flat_surface_padding_auto_check.setChecked(True)
        self.flat_surface_padding_auto_check.setToolTip(
            "When checked, the default 50 m margin is added around the GPS bounding box\n"
            "to cover the camera footprint. Uncheck to set a custom padding."
        )
        self.flat_surface_padding_spin = QDoubleSpinBox()
        self.flat_surface_padding_spin.setRange(0.0, 10000.0)
        self.flat_surface_padding_spin.setSingleStep(5.0)
        self.flat_surface_padding_spin.setDecimals(1)
        self.flat_surface_padding_spin.setValue(50.0)
        self.flat_surface_padding_spin.setSuffix(" m")
        self.flat_surface_padding_spin.setEnabled(False)
        self.flat_surface_padding_spin.setToolTip(
            "Margin added around the GPS bounding box of the flight when sizing the mesh."
        )
        self.flat_surface_padding_auto_check.toggled.connect(
            lambda checked: self.flat_surface_padding_spin.setEnabled(not checked)
        )
        flat_surf_pad_row.addWidget(self.flat_surface_padding_auto_check)
        flat_surf_pad_row.addWidget(self.flat_surface_padding_spin)
        flat_surf_pad_row.addStretch()
        flat_surface_layout.addRow("Padding:", flat_surf_pad_row)

        flat_surf_gen_row = QHBoxLayout()
        flat_surface_generate_btn = QPushButton("Generate Flat Surface Mesh")
        flat_surface_generate_btn.setToolTip(
            "Reads GPS from the AirData CSV or SRT files, generates a flat GLB mesh\n"
            "centred on the flight area, and sets the result as the DEM input."
        )
        flat_surface_generate_btn.clicked.connect(self._generate_flat_surface_mesh)
        flat_surf_gen_row.addWidget(flat_surface_generate_btn)
        flat_surf_gen_row.addStretch()
        flat_surface_layout.addRow("", flat_surf_gen_row)

        self.geo_input_tabs.addTab(flat_surface_tab, "Flat Surface")

        input_layout.addWidget(geo_group)
        input_layout.addStretch()

        # =====================================================================
        # MAIN TAB 2: CONFIGURATION (with sub-tabs)
        # =====================================================================
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        main_tabs.addTab(config_tab, "Configuration")

        # Create sub-tabs for configuration
        config_sub_tabs = QTabWidget()
        config_layout.addWidget(config_sub_tabs)

        # ----- Sub-Tab: Extraction -----
        extraction_tab = QWidget()
        extraction_tab_layout = QVBoxLayout(extraction_tab)
        config_sub_tabs.addTab(extraction_tab, "Extraction")

        # DJI SDK info box
        try:
            from qgis.core import QgsApplication
            self._thermal_plugins_dir = os.path.join(
                QgsApplication.qgisSettingsDirPath(), 'bambi_deps'
            )
        except Exception:
            self._thermal_plugins_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'plugins'
            )

        # Frame extraction parameters
        frame_ext_group = QGroupBox("Frame Extraction")
        frame_ext_layout = QFormLayout(frame_ext_group)

        self.extract_skip_spin = QSpinBox()
        self.extract_skip_spin.setRange(0, 999999)
        self.extract_skip_spin.setValue(0)
        self.extract_skip_spin.setToolTip(
            "Skip this many matched images / frames at the start before processing begins.")
        frame_ext_layout.addRow("Skip first:", self.extract_skip_spin)

        limit_row = QHBoxLayout()
        self.extract_limit_check = QCheckBox("Enable")
        self.extract_limit_spin = QSpinBox()
        self.extract_limit_spin.setRange(1, 999999)
        self.extract_limit_spin.setValue(100)
        self.extract_limit_spin.setEnabled(False)
        self.extract_limit_spin.setToolTip("Maximum number of images / frames to process.")
        self.extract_limit_check.toggled.connect(self.extract_limit_spin.setEnabled)
        limit_row.addWidget(self.extract_limit_check)
        limit_row.addWidget(self.extract_limit_spin)
        limit_row.addStretch()
        frame_ext_layout.addRow("Limit:", limit_row)

        sampling_row = QHBoxLayout()
        self.extract_sampling_rate_check = QCheckBox("Enable")
        self.extract_sampling_rate_spin = QSpinBox()
        self.extract_sampling_rate_spin.setRange(1, 9999)
        self.extract_sampling_rate_spin.setValue(5)
        self.extract_sampling_rate_spin.setEnabled(False)
        self.extract_sampling_rate_spin.setToolTip(
            "Video mode only. Take every N-th frame (e.g. 5 = every 5th frame).")
        self.extract_sampling_rate_check.toggled.connect(self.extract_sampling_rate_spin.setEnabled)
        sampling_row.addWidget(self.extract_sampling_rate_check)
        sampling_row.addWidget(self.extract_sampling_rate_spin)
        sampling_row.addStretch()
        frame_ext_layout.addRow("Sampling rate (video only):", sampling_row)

        self.preserve_aspect_ratio_check = QCheckBox("Preserve original aspect ratio (otherwise square 1:1 format)")
        self.preserve_aspect_ratio_check.setChecked(False)
        self.preserve_aspect_ratio_check.setToolTip(
            "When enabled, extracted frames keep the original width/height ratio instead of being cropped to a square.")
        frame_ext_layout.addRow("Frame format:", self.preserve_aspect_ratio_check)

        self.use_gimbal_heading_check = QCheckBox("Use gimbal heading (instead of compass heading)")
        self.use_gimbal_heading_check.setChecked(False)
        self.use_gimbal_heading_check.setToolTip(
            "When enabled, uses the gimbal heading instead of the compass heading for the yaw rotation.")
        frame_ext_layout.addRow("Heading source:", self.use_gimbal_heading_check)

        self.no_images_check = QCheckBox("Only extract poses (no images)")
        self.no_images_check.setChecked(False)
        self.no_images_check.setToolTip(
            "When enabled, only the poses.json file is created and no image data is "
            "written to disk.")
        frame_ext_layout.addRow("Pose-only mode:", self.no_images_check)

        no_images_label = QLabel(
            "For debugging / previewing the flight route only. All subsequent stages "
            "(detection, tracking, correction, georeferencing, analytics) depend on the "
            "extracted images and will not work when this is enabled."
        )
        no_images_label.setWordWrap(True)
        no_images_label.setStyleSheet("color: gray; font-size: 10px;")
        frame_ext_layout.addRow("", no_images_label)

        extraction_tab_layout.addWidget(frame_ext_group)

        # Timezone (shared by video and photo modes)
        tz_group = QGroupBox("Timezone")
        tz_group_layout = QFormLayout(tz_group)

        self.tz_auto_check = QCheckBox("Auto-detect")
        self.tz_auto_check.setChecked(True)
        self.tz_auto_check.setToolTip(
            "Auto-detect the drone/camera clock timezone by comparing SRT timestamps "
            "(video mode) or photo EXIF timestamps (photo mode) with AirData UTC timestamps.")
        self.tz_auto_check.stateChanged.connect(self._on_tz_auto_changed)

        self.tz_auto_result_label = QLabel()
        self.tz_auto_result_label.setStyleSheet("color: grey; font-size: 10px;")
        self.tz_auto_result_label.setVisible(True)

        _tz_auto_row = QHBoxLayout()
        _tz_auto_row.addWidget(self.tz_auto_check)
        _tz_auto_row.addWidget(self.tz_auto_result_label)
        tz_group_layout.addRow("Offset:", _tz_auto_row)

        self.timezone_combo = QComboBox()
        self.timezone_combo.setEditable(True)
        self.timezone_combo.setToolTip(
            "Timezone of the drone/camera clock. The current UTC offset is shown on the right.")
        try:
            from zoneinfo import available_timezones
            _tz_names = sorted(available_timezones())
        except ImportError:
            _tz_names = [
                "UTC", "Europe/London", "Europe/Paris", "Europe/Berlin",
                "Europe/Vienna", "Europe/Rome", "Europe/Madrid",
                "Europe/Warsaw", "Europe/Helsinki", "Europe/Athens",
                "America/New_York", "America/Chicago", "America/Denver",
                "America/Los_Angeles", "America/Anchorage", "Pacific/Honolulu",
                "America/Sao_Paulo", "America/Buenos_Aires",
                "Asia/Dubai", "Asia/Kolkata", "Asia/Bangkok",
                "Asia/Colombo", "Asia/Male",
                "Asia/Shanghai", "Asia/Tokyo", "Australia/Sydney",
            ]
        self.timezone_combo.addItems(_tz_names)
        _tz_default_idx = self.timezone_combo.findText("Europe/Vienna")
        if _tz_default_idx >= 0:
            self.timezone_combo.setCurrentIndex(_tz_default_idx)

        self.timezone_offset_label = QLabel()
        self.timezone_offset_label.setStyleSheet("color: grey; font-size: 10px;")
        self._update_timezone_offset_label()

        self.tz_manual_widget = QWidget()
        self.tz_manual_widget.setVisible(False)
        _tz_manual_row = QHBoxLayout(self.tz_manual_widget)
        _tz_manual_row.setContentsMargins(0, 0, 0, 0)
        _tz_manual_row.addWidget(self.timezone_combo)
        _tz_manual_row.addWidget(self.timezone_offset_label)
        tz_group_layout.addRow("Timezone:", self.tz_manual_widget)

        self.timezone_combo.currentTextChanged.connect(self._update_timezone_offset_label)

        self.tz_manual_hint = QLabel(
            'Timezone of camera/drone clock when recordings were made '
            '(consider daylight saving time)!'
        )
        self.tz_manual_hint.setWordWrap(True)
        self.tz_manual_hint.setVisible(False)
        tz_group_layout.addWidget(self.tz_manual_hint)

        extraction_tab_layout.addWidget(tz_group)

        # Thermal visualisation (photo mode only — hidden in video mode)
        self.thermal_vis_group = QGroupBox("Thermal Visualisation")
        self.thermal_vis_group.setVisible(not self.video_mode_check.isChecked())
        thermal_vis_group = self.thermal_vis_group
        thermal_vis_layout = QFormLayout(thermal_vis_group)

        self.thermal_vis_cmap_combo = QComboBox()
        self.thermal_vis_cmap_combo.addItems([
            "(none)",
            "white-hotspot", "black-hotspot",
            "plasma", "inferno", "magma", "viridis", "jet",
        ])
        thermal_vis_layout.addRow("Colormap:", self.thermal_vis_cmap_combo)

        self.thermal_vis_mode_combo = QComboBox()
        self.thermal_vis_mode_combo.addItems([
            "Thresholds (lower/upper)",
            "Curve (custom mapping)",
        ])
        self.thermal_vis_mode_combo.setToolTip(
            "Thresholds: linear stretch with optional black clipping.\n"
            "Curve: fine-granular tone mapping over a fixed temperature "
            "range, like the Curves tool in image editors. The range can "
            "be auto-detected by scanning the flight's images.")
        self.thermal_vis_mode_combo.currentIndexChanged.connect(
            self._on_thermal_vis_mode_changed)
        thermal_vis_layout.addRow("Tone mapping:", self.thermal_vis_mode_combo)

        lo_row = QHBoxLayout()
        self.thermal_vis_lo_check = QCheckBox("Enable")
        self.thermal_vis_lo_spin = QDoubleSpinBox()
        self.thermal_vis_lo_spin.setRange(-200.0, 3000.0)
        self.thermal_vis_lo_spin.setDecimals(1)
        self.thermal_vis_lo_spin.setSuffix(" °C")
        self.thermal_vis_lo_spin.setEnabled(False)
        self.thermal_vis_lo_check.toggled.connect(self.thermal_vis_lo_spin.setEnabled)
        lo_row.addWidget(self.thermal_vis_lo_check)
        lo_row.addWidget(self.thermal_vis_lo_spin)
        lo_row.addStretch()
        thermal_vis_layout.addRow("Lower threshold (→ black):", lo_row)

        hi_row = QHBoxLayout()
        self.thermal_vis_hi_check = QCheckBox("Enable")
        self.thermal_vis_hi_spin = QDoubleSpinBox()
        self.thermal_vis_hi_spin.setRange(-200.0, 3000.0)
        self.thermal_vis_hi_spin.setDecimals(1)
        self.thermal_vis_hi_spin.setSuffix(" °C")
        self.thermal_vis_hi_spin.setEnabled(False)
        self.thermal_vis_hi_check.toggled.connect(self.thermal_vis_hi_spin.setEnabled)
        hi_row.addWidget(self.thermal_vis_hi_check)
        hi_row.addWidget(self.thermal_vis_hi_spin)
        hi_row.addStretch()
        thermal_vis_layout.addRow("Upper threshold (→ black):", hi_row)

        # Curve mapping (used when tone mapping = curve)
        self._thermal_vis_curve = None  # core.thermal_curve.ThermalCurve
        curve_row = QHBoxLayout()
        self.thermal_vis_curve_btn = QPushButton("Edit Curve…")
        self.thermal_vis_curve_btn.setToolTip(
            "Define the temperature-to-intensity mapping curve. "
            "\"Auto Detect\" inside the editor scans the thermal photo "
            "directory for the flight's actual temperature range.")
        self.thermal_vis_curve_btn.clicked.connect(self._edit_thermal_vis_curve)
        self.thermal_vis_curve_label = QLabel()
        self.thermal_vis_curve_label.setStyleSheet("color: grey;")
        curve_row.addWidget(self.thermal_vis_curve_btn)
        curve_row.addWidget(self.thermal_vis_curve_label, 1)
        thermal_vis_layout.addRow("Curve:", curve_row)
        self._on_thermal_vis_mode_changed(
            self.thermal_vis_mode_combo.currentIndex())
        self._update_thermal_vis_curve_label()

        sdk_row = QHBoxLayout()
        self._thermal_sdk_info_label = QLabel()
        self._thermal_sdk_info_label.setWordWrap(True)
        self._thermal_sdk_info_label.setTextFormat(Qt.TextFormat.RichText)
        sdk_row.addWidget(self._thermal_sdk_info_label)
        thermal_vis_layout.addRow(sdk_row)

        extraction_tab_layout.addWidget(thermal_vis_group)

        config_sub_tabs.currentChanged.connect(
            lambda _: self._refresh_thermal_sdk_status()
        )
        self._refresh_thermal_sdk_status()
        extraction_tab_layout.addStretch()

        # ----- Sub-Tab 7: Flight Route Visualization -----
        flight_route_tab = QWidget()
        flight_route_tab_layout = QVBoxLayout(flight_route_tab)
        config_sub_tabs.addTab(flight_route_tab, "Flight Route")

        flight_route_viz_group = QGroupBox("Flight Route Visualization")
        flight_route_viz_layout = QFormLayout(flight_route_viz_group)

        # Filter GPS origin (0, 0) checkbox
        self.filter_gps_origin_check = QCheckBox()
        self.filter_gps_origin_check.setChecked(True)
        self.filter_gps_origin_check.setToolTip(
            "Remove GPS entries at (0°, 0°) which occur when the drone has not yet acquired a GPS fix"
        )
        flight_route_viz_layout.addRow("Filter GPS Origin (0°, 0°):", self.filter_gps_origin_check)

        # Frame markers enable checkbox
        self.frame_markers_enabled_check = QCheckBox()
        self.frame_markers_enabled_check.setChecked(True)
        self.frame_markers_enabled_check.setToolTip(
            "Enable displaying text markers at every N-th camera position"
        )
        self.frame_markers_enabled_check.stateChanged.connect(self._toggle_frame_marker_interval)
        flight_route_viz_layout.addRow("Show Frame Markers:", self.frame_markers_enabled_check)

        # Frame marker interval
        self.frame_marker_interval_spin = QSpinBox()
        self.frame_marker_interval_spin.setRange(1, 10000)
        self.frame_marker_interval_spin.setValue(100)
        self.frame_marker_interval_spin.setToolTip(
            "Display a text marker every N frames (e.g., 100 shows markers at frame 100, 200, 300, ...)"
        )
        flight_route_viz_layout.addRow("Marker Interval (N):", self.frame_marker_interval_spin)

        # Include frame 0 checkbox
        self.frame_marker_include_zero_check = QCheckBox()
        self.frame_marker_include_zero_check.setChecked(False)
        self.frame_marker_include_zero_check.setToolTip(
            "Include a marker at frame 0 (the starting position)"
        )
        flight_route_viz_layout.addRow("Include Frame 0:", self.frame_marker_include_zero_check)

        # Image labels enable checkbox
        self.image_labels_enabled_check = QCheckBox()
        self.image_labels_enabled_check.setChecked(False)
        self.image_labels_enabled_check.setToolTip(
            "Enable displaying image filenames at every N-th camera position"
        )
        self.image_labels_enabled_check.stateChanged.connect(self._toggle_image_label_controls)
        flight_route_viz_layout.addRow("Show Image Labels:", self.image_labels_enabled_check)

        # Image label interval
        self.image_label_interval_spin = QSpinBox()
        self.image_label_interval_spin.setRange(1, 100000)
        self.image_label_interval_spin.setValue(100)
        self.image_label_interval_spin.setEnabled(False)
        self.image_label_interval_spin.setToolTip(
            "Display an image filename label every N frames (e.g., 100 shows labels at frame 100, 200, 300, ...)"
        )
        flight_route_viz_layout.addRow("Image Label Interval (N):", self.image_label_interval_spin)

        flight_route_tab_layout.addWidget(flight_route_viz_group)

        # Distance-based markers group
        distance_markers_group = QGroupBox("Distance Markers")
        distance_markers_layout = QFormLayout(distance_markers_group)

        # Distance markers enable checkbox
        self.distance_markers_enabled_check = QCheckBox()
        self.distance_markers_enabled_check.setChecked(False)
        self.distance_markers_enabled_check.setToolTip(
            "Enable displaying markers at regular distance intervals along the flight path"
        )
        self.distance_markers_enabled_check.stateChanged.connect(self._toggle_distance_marker_controls)
        distance_markers_layout.addRow("Show Distance Markers:", self.distance_markers_enabled_check)

        # Distance interval spinbox
        self.distance_marker_interval_spin = QSpinBox()
        self.distance_marker_interval_spin.setRange(1, 10000)
        self.distance_marker_interval_spin.setValue(100)
        self.distance_marker_interval_spin.setSuffix(" m")
        self.distance_marker_interval_spin.setEnabled(False)
        self.distance_marker_interval_spin.setToolTip(
            "Display a marker every N meters (e.g., 100 shows markers at 100m, 200m, 300m, ...)"
        )
        distance_markers_layout.addRow("Distance Interval:", self.distance_marker_interval_spin)

        # Include start (0m) checkbox
        self.distance_marker_include_start_check = QCheckBox()
        self.distance_marker_include_start_check.setChecked(False)
        self.distance_marker_include_start_check.setEnabled(False)
        self.distance_marker_include_start_check.setToolTip(
            "Include a marker at the starting position (0m)"
        )
        distance_markers_layout.addRow("Include Start (0m):", self.distance_marker_include_start_check)

        flight_route_tab_layout.addWidget(distance_markers_group)

        # Time-based markers group
        time_markers_group = QGroupBox("Time Markers")
        time_markers_layout = QFormLayout(time_markers_group)

        self.time_markers_enabled_check = QCheckBox()
        self.time_markers_enabled_check.setChecked(False)
        self.time_markers_enabled_check.setToolTip(
            "Enable displaying markers at regular time intervals along the flight path"
        )
        self.time_markers_enabled_check.stateChanged.connect(self._toggle_time_marker_controls)
        time_markers_layout.addRow("Show Time Markers:", self.time_markers_enabled_check)

        self.time_marker_type_combo = QComboBox()
        self.time_marker_type_combo.addItems(["Relative", "Absolute"])
        self.time_marker_type_combo.setEnabled(False)
        self.time_marker_type_combo.setToolTip(
            "Relative: elapsed seconds from first frame\n"
            "Absolute: UTC date/time from AirData log"
        )
        time_markers_layout.addRow("Time Type:", self.time_marker_type_combo)

        self.time_marker_interval_spin = QSpinBox()
        self.time_marker_interval_spin.setRange(1, 3600)
        self.time_marker_interval_spin.setValue(10)
        self.time_marker_interval_spin.setSuffix(" sec")
        self.time_marker_interval_spin.setEnabled(False)
        self.time_marker_interval_spin.setToolTip(
            "Place a marker every N seconds (e.g., 10 shows markers at 10s, 20s, 30s, ...)"
        )
        time_markers_layout.addRow("Interval:", self.time_marker_interval_spin)

        self.time_marker_include_start_check = QCheckBox()
        self.time_marker_include_start_check.setChecked(False)
        self.time_marker_include_start_check.setEnabled(False)
        self.time_marker_include_start_check.setToolTip(
            "Include a marker at the starting position (t=0)"
        )
        time_markers_layout.addRow("Include Start:", self.time_marker_include_start_check)

        flight_route_tab_layout.addWidget(time_markers_group)
        flight_route_tab_layout.addStretch()

        # ----- Sub-Tab 1: Detection -----
        detect_tab = QWidget()
        detect_tab_layout = QVBoxLayout(detect_tab)
        config_sub_tabs.addTab(detect_tab, "Detection")

        # Detection parameters
        detection_group = QGroupBox("Detection")
        detection_layout = QFormLayout(detection_group)

        self.thermal_model_path_edit = QLineEdit()
        self.thermal_model_path_edit.setPlaceholderText(
            "Leave empty for default HuggingFace model")
        thermal_model_browse_btn = QPushButton("Browse...")
        thermal_model_browse_btn.clicked.connect(self.browse_thermal_model)
        thermal_model_row = QHBoxLayout()
        thermal_model_row.addWidget(self.thermal_model_path_edit)
        thermal_model_row.addWidget(thermal_model_browse_btn)
        detection_layout.addRow("Thermal Model Path:", thermal_model_row)

        self.rgb_model_path_edit = QLineEdit()
        self.rgb_model_path_edit.setPlaceholderText(
            "Leave empty for default HuggingFace model")
        rgb_model_browse_btn = QPushButton("Browse...")
        rgb_model_browse_btn.clicked.connect(self.browse_rgb_model)
        rgb_model_row = QHBoxLayout()
        rgb_model_row.addWidget(self.rgb_model_path_edit)
        rgb_model_row.addWidget(rgb_model_browse_btn)
        detection_layout.addRow("RGB Model Path:", rgb_model_row)

        detection_label = QLabel(
            "The model matching the camera selected in the Detection step is "
            "used. Note, that the default BAMBI models were trained on "
            "white-hotspot thermal respectively RGB data showing roe deer, red "
            "deer and wild boar with an AGL between 30 to 60 m. So the "
            "applicability is limited to that scope. Additionally, the models "
            "are based on the Ultralytics framework so the utilization follows "
            "their license."
        )
        detection_label.setWordWrap(True)
        detection_label.setStyleSheet("color: gray; font-size: 10px;")
        detection_layout.addWidget(detection_label)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.5)
        self.confidence_spin.setToolTip("Minimum detection confidence")
        detection_layout.addRow("Min Confidence:", self.confidence_spin)

        # Detection frame filters
        det_filter_label = QLabel("Frame Range:")
        det_filter_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        detection_layout.addRow(det_filter_label)

        self.detect_all_frames_check = QCheckBox("Use all frames")
        self.detect_all_frames_check.setChecked(True)
        self.detect_all_frames_check.stateChanged.connect(self.toggle_detect_frame_range)
        detection_layout.addRow("", self.detect_all_frames_check)

        self.detect_frame_range_widget = QWidget()
        detect_frame_range_layout = QHBoxLayout(self.detect_frame_range_widget)
        detect_frame_range_layout.setContentsMargins(0, 0, 0, 0)

        self.detect_start_frame_spin = QSpinBox()
        self.detect_start_frame_spin.setRange(0, 999999)
        self.detect_start_frame_spin.setValue(0)
        self.detect_start_frame_spin.setToolTip("First frame to include (inclusive)")
        detect_frame_range_layout.addWidget(QLabel("Start:"))
        detect_frame_range_layout.addWidget(self.detect_start_frame_spin)

        self.detect_end_frame_spin = QSpinBox()
        self.detect_end_frame_spin.setRange(0, 999999)
        self.detect_end_frame_spin.setValue(999999)
        self.detect_end_frame_spin.setToolTip("Last frame to include (inclusive)")
        detect_frame_range_layout.addWidget(QLabel("End:"))
        detect_frame_range_layout.addWidget(self.detect_end_frame_spin)

        self.detect_frame_range_widget.setEnabled(False)
        detection_layout.addRow("", self.detect_frame_range_widget)

        self.detect_sample_rate_spin = QSpinBox()
        self.detect_sample_rate_spin.setRange(1, 100)
        self.detect_sample_rate_spin.setValue(1)
        self.detect_sample_rate_spin.setToolTip("Process every Nth frame (1 = all frames)")
        detection_layout.addRow("Sample Rate:", self.detect_sample_rate_spin)

        detect_tab_layout.addWidget(detection_group)

        # Project vocabulary. Species are a property of the survey rather than
        # of whichever tool is open, so they are configured here and only here
        # — that is what keeps class ids stable.
        vocabulary_group = QGroupBox("Project Schema")
        vocabulary_layout = QVBoxLayout(vocabulary_group)
        vocabulary_layout.addWidget(QLabel(
            "Species, enums and custom fields for this project. Every tool "
            "picks from these; none of them can invent new ones."))
        self.schema_editor_btn = QPushButton("Edit Project Schema…")
        self.schema_editor_btn.clicked.connect(self.open_schema_dialog)
        vocabulary_layout.addWidget(self.schema_editor_btn)
        detect_tab_layout.addWidget(vocabulary_group)

        detect_tab_layout.addStretch()

        # ----- Sub-Tab 2: Position Correction -----
        correction_tab = QWidget()
        correction_tab_layout = QVBoxLayout(correction_tab)
        config_sub_tabs.addTab(correction_tab, "Position Correction")

        # Rotation unit selection
        unit_group = QGroupBox("Rotation Unit")
        unit_layout = QHBoxLayout(unit_group)
        unit_layout.addWidget(QLabel("Display rotations in:"))
        self.rotation_unit_combo = QComboBox()
        self.rotation_unit_combo.addItems(["Radians", "Degrees"])
        self.rotation_unit_combo.setCurrentIndex(0)  # Default to radians
        self.rotation_unit_combo.currentIndexChanged.connect(self._on_rotation_unit_changed)
        self.rotation_unit_combo.setToolTip(
            "Select the unit for rotation values in the UI.\n"
            "Backend and saved files always use radians."
        )
        unit_layout.addWidget(self.rotation_unit_combo)
        unit_layout.addStretch()
        correction_tab_layout.addWidget(unit_group)

        correction_group = QGroupBox("Default Correction Factors")
        correction_layout = QFormLayout(correction_group)

        # Translation corrections
        trans_label = QLabel("Translation (x, y, z):")
        correction_layout.addRow(trans_label)

        trans_row = QHBoxLayout()
        self.trans_x_spin = QDoubleSpinBox()
        self.trans_x_spin.setRange(-100, 100)
        self.trans_x_spin.setValue(0.0)
        self.trans_x_spin.setDecimals(3)
        trans_row.addWidget(QLabel("X:"))
        trans_row.addWidget(self.trans_x_spin)

        self.trans_y_spin = QDoubleSpinBox()
        self.trans_y_spin.setRange(-100, 100)
        self.trans_y_spin.setValue(0.0)
        self.trans_y_spin.setDecimals(3)
        trans_row.addWidget(QLabel("Y:"))
        trans_row.addWidget(self.trans_y_spin)

        self.trans_z_spin = QDoubleSpinBox()
        self.trans_z_spin.setRange(-100, 100)
        self.trans_z_spin.setValue(0.0)
        self.trans_z_spin.setDecimals(3)
        trans_row.addWidget(QLabel("Z:"))
        trans_row.addWidget(self.trans_z_spin)
        correction_layout.addRow(trans_row)

        # Rotation corrections
        self.rot_label = QLabel("Rotation (pitch, roll, yaw) [rad]:")
        correction_layout.addRow(self.rot_label)

        rot_row = QHBoxLayout()
        self.rot_x_spin = QDoubleSpinBox()
        self.rot_x_spin.setRange(-6.28319, 6.28319)  # -2π to 2π
        self.rot_x_spin.setValue(0.0)
        self.rot_x_spin.setDecimals(5)
        self.rot_x_spin.setSingleStep(0.01)
        rot_row.addWidget(QLabel("Pitch:"))
        rot_row.addWidget(self.rot_x_spin)

        self.rot_y_spin = QDoubleSpinBox()
        self.rot_y_spin.setRange(-6.28319, 6.28319)  # -2π to 2π
        self.rot_y_spin.setValue(0.0)
        self.rot_y_spin.setDecimals(5)
        self.rot_y_spin.setSingleStep(0.01)
        rot_row.addWidget(QLabel("Roll:"))
        rot_row.addWidget(self.rot_y_spin)

        self.rot_z_spin = QDoubleSpinBox()
        self.rot_z_spin.setRange(-6.28319, 6.28319)  # -2π to 2π
        self.rot_z_spin.setValue(0.0)
        self.rot_z_spin.setDecimals(5)
        self.rot_z_spin.setSingleStep(0.01)
        rot_row.addWidget(QLabel("Yaw:"))
        rot_row.addWidget(self.rot_z_spin)
        correction_layout.addRow(rot_row)

        correction_tab_layout.addWidget(correction_group)

        # Frame-range specific corrections
        additional_group = QGroupBox("Frame-Range Specific Corrections")
        additional_layout = QVBoxLayout(additional_group)

        # Info label
        info_label = QLabel(
            "Define corrections for specific frame ranges. "
            "These override the default correction for frames within the specified range."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; font-size: 10px;")
        additional_layout.addWidget(info_label)

        # List widget for additional corrections
        self.additional_corrections_list = QListWidget()
        self.additional_corrections_list.setMinimumHeight(100)
        self.additional_corrections_list.setMaximumHeight(150)
        self.additional_corrections_list.setToolTip(
            "Double-click to edit, or select and use buttons below"
        )
        self.additional_corrections_list.itemDoubleClicked.connect(
            self._edit_additional_correction
        )
        additional_layout.addWidget(self.additional_corrections_list)

        # Buttons for managing additional corrections
        additional_btn_row = QHBoxLayout()
        self.add_correction_btn = QPushButton("Add...")
        self.add_correction_btn.setToolTip("Add a new frame-range specific correction")
        self.add_correction_btn.clicked.connect(self._add_additional_correction)
        additional_btn_row.addWidget(self.add_correction_btn)

        self.edit_correction_btn = QPushButton("Edit...")
        self.edit_correction_btn.setToolTip("Edit the selected correction")
        self.edit_correction_btn.clicked.connect(self._edit_selected_correction)
        additional_btn_row.addWidget(self.edit_correction_btn)

        self.remove_correction_btn = QPushButton("Remove")
        self.remove_correction_btn.setToolTip("Remove the selected correction")
        self.remove_correction_btn.clicked.connect(self._remove_additional_correction)
        additional_btn_row.addWidget(self.remove_correction_btn)

        additional_layout.addLayout(additional_btn_row)

        correction_tab_layout.addWidget(additional_group)

        # Store additional corrections data (always in radians internally)
        self._additional_corrections: List[Dict[str, Any]] = []

        # Save correction button
        save_correction_btn = QPushButton("Save Correction to JSON...")
        save_correction_btn.setToolTip(
            "Save the current translation and rotation values to a JSON file (rotations saved in radians)")
        save_correction_btn.clicked.connect(self.save_correction_values)
        correction_tab_layout.addWidget(save_correction_btn)

        correction_tab_layout.addStretch()

        # ----- Sub-Tab 3: Tracking -----
        tracking_tab = QWidget()
        tracking_tab_layout = QVBoxLayout(tracking_tab)
        config_sub_tabs.addTab(tracking_tab, "Tracking")

        tracking_group = QGroupBox("Tracking Parameters")
        tracking_layout = QFormLayout(tracking_group)

        # Tracker backend selection
        self.tracker_backend_combo = QComboBox()
        self.tracker_backend_combo.setToolTip(
            "Select tracking algorithm:\n"
            "• Built-in: Default tracker, no dependencies\n"
            "• BoxMOT: Advanced trackers (requires: pip install boxmot)\n"
            "• GeoRef: Geo-referenced tracking (requires: pip install georef-tracker)"
        )
        self._populate_tracker_backends()
        self.tracker_backend_combo.currentIndexChanged.connect(self._on_tracker_changed)

        tracker_row = QHBoxLayout()
        tracker_row.addWidget(self.tracker_backend_combo)
        self.refresh_trackers_btn = QPushButton("↻")
        self.refresh_trackers_btn.setFixedWidth(30)
        self.refresh_trackers_btn.setToolTip(
            "Refresh available trackers\n(Use after installing boxmot or georef-tracker)")
        self.refresh_trackers_btn.clicked.connect(self._refresh_trackers)
        tracker_row.addWidget(self.refresh_trackers_btn)
        tracking_layout.addRow("Tracker:", tracker_row)

        # Tracker info label
        self.tracker_info_label = QLabel()
        self.tracker_info_label.setWordWrap(True)
        self.tracker_info_label.setStyleSheet("color: gray; font-size: 10px;")
        tracking_layout.addRow("", self.tracker_info_label)

        # ReID model selection
        self.reid_widget = QWidget()
        reid_layout = QFormLayout(self.reid_widget)
        reid_layout.setContentsMargins(0, 0, 0, 0)

        self.reid_model_combo = QComboBox()
        self.reid_model_combo.addItems([
            "osnet - Standard OSNet model",
            "bambi-thermal-omni - BAMBI Thermal-Omni ReID (HuggingFace)",
            "custom - Custom weights file"
        ])
        self.reid_model_combo.setToolTip(
            "Select ReID model for appearance-based tracking:\n"
            "• osnet: BoxMOT's standard OSNet model (general purpose)\n"
            "• bambi-thermal-omni: BAMBI Thermal-Omni ReID model from HuggingFace\n"
            "• custom: Use your own ReID weights file"
        )
        self.reid_model_combo.currentIndexChanged.connect(self._on_reid_model_changed)
        reid_layout.addRow("ReID Model:", self.reid_model_combo)

        # Custom ReID weights path
        self.custom_reid_widget = QWidget()
        custom_reid_layout = QHBoxLayout(self.custom_reid_widget)
        custom_reid_layout.setContentsMargins(0, 0, 0, 0)
        self.custom_reid_path_edit = QLineEdit()
        self.custom_reid_path_edit.setPlaceholderText("Path to custom ReID weights (.pt)")
        custom_reid_browse_btn = QPushButton("Browse...")
        custom_reid_browse_btn.clicked.connect(self._browse_custom_reid)
        custom_reid_layout.addWidget(self.custom_reid_path_edit)
        custom_reid_layout.addWidget(custom_reid_browse_btn)
        reid_layout.addRow("Custom Weights:", self.custom_reid_widget)
        self.custom_reid_widget.setVisible(False)

        tracking_layout.addRow(self.reid_widget)
        self.reid_widget.setVisible(False)

        # JSON parameters
        self.tracker_params_widget = QWidget()
        params_inner_layout = QVBoxLayout(self.tracker_params_widget)
        params_inner_layout.setContentsMargins(0, 0, 0, 0)

        params_header = QHBoxLayout()
        params_label = QLabel("Advanced Parameters (JSON):")
        self.show_params_help_btn = QPushButton("?")
        self.show_params_help_btn.setFixedWidth(25)
        self.show_params_help_btn.setToolTip("Show available parameters for selected tracker")
        self.show_params_help_btn.clicked.connect(self._show_tracker_params_help)
        params_header.addWidget(params_label)
        params_header.addWidget(self.show_params_help_btn)
        params_header.addStretch()
        params_inner_layout.addLayout(params_header)

        self.tracker_params_edit = QTextEdit()
        self.tracker_params_edit.setPlaceholderText('{"param_name": value, ...}')
        self.tracker_params_edit.setMaximumHeight(80)
        self.tracker_params_edit.setToolTip(
            "JSON object with tracker-specific parameters.\n"
            "Only valid parameters for the selected tracker will be used.\n"
            "Example: {\"max_age\": 50, \"iou_threshold\": 0.4}"
        )
        params_inner_layout.addWidget(self.tracker_params_edit)
        tracking_layout.addRow(self.tracker_params_widget)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        tracking_layout.addRow(separator)

        # Built-in tracker parameters
        self.builtin_params_widget = QWidget()
        builtin_layout = QFormLayout(self.builtin_params_widget)
        builtin_layout.setContentsMargins(0, 0, 0, 0)

        self.iou_threshold_spin = QDoubleSpinBox()
        self.iou_threshold_spin.setRange(0.0, 1.0)
        self.iou_threshold_spin.setSingleStep(0.05)
        self.iou_threshold_spin.setValue(0.3)
        self.iou_threshold_spin.setToolTip("IoU threshold for track matching")
        builtin_layout.addRow("IoU Threshold:", self.iou_threshold_spin)

        self.max_age_spin = QSpinBox()
        self.max_age_spin.setRange(-1, 100)
        self.max_age_spin.setValue(-1)
        self.max_age_spin.setToolTip("Max frames without detection (-1 = unlimited)")
        builtin_layout.addRow("Max Age:", self.max_age_spin)

        self.max_center_dist_spin = QDoubleSpinBox()
        self.max_center_dist_spin.setRange(0.0, 100.0)
        self.max_center_dist_spin.setValue(0.2)
        self.max_center_dist_spin.setToolTip("Max center distance for matching (meters)")
        builtin_layout.addRow("Max Center Distance:", self.max_center_dist_spin)

        self.tracker_mode_combo = QComboBox()
        self.tracker_mode_combo.addItems([
            "GREEDY",
            "HUNGARIAN",
            "CENTER",
            "HUNGARIAN_CENTER"
        ])
        self.tracker_mode_combo.setCurrentIndex(1)
        builtin_layout.addRow("Tracker Mode:", self.tracker_mode_combo)

        self.class_aware_check = QCheckBox()
        self.class_aware_check.setChecked(True)
        builtin_layout.addRow("Class Aware:", self.class_aware_check)

        self.interpolate_check = QCheckBox()
        self.interpolate_check.setChecked(True)
        builtin_layout.addRow("Interpolate Missing:", self.interpolate_check)

        tracking_layout.addRow(self.builtin_params_widget)

        tracking_tab_layout.addWidget(tracking_group)

        # Initialize tracker UI state
        self._on_tracker_changed(0)

        # ----- TRex Tracklet Import -----
        # When an NPZ folder is specified here, the "Track Animals Or Import" step
        # imports the pre-computed TRex tracklets instead of running the tracker.
        trex_group = QGroupBox("TRex Tracklet Import")
        trex_layout = QFormLayout(trex_group)

        self.trex_npz_dir_edit = QLineEdit()
        self.trex_npz_dir_edit.setPlaceholderText("Folder containing TRex *.npz tracklet files")
        trex_npz_browse_btn = QPushButton("Browse...")
        trex_npz_browse_btn.clicked.connect(self.browse_trex_npz_dir)
        trex_npz_row = QHBoxLayout()
        trex_npz_row.addWidget(self.trex_npz_dir_edit)
        trex_npz_row.addWidget(trex_npz_browse_btn)
        trex_layout.addRow("NPZ Folder:", trex_npz_row)

        self.trex_undistorted_check = QCheckBox("Labels already in undistorted frame space")
        self.trex_undistorted_check.setChecked(False)
        self.trex_undistorted_check.setToolTip(
            "Check if TRex was run on already-undistorted BAMBI frames.\n"
            "Leave unchecked (default) if TRex was run on the original raw video."
        )
        trex_layout.addRow("", self.trex_undistorted_check)

        tracking_tab_layout.addWidget(trex_group)

        tracking_tab_layout.addStretch()

        # ----- Sub-Tab 4: Field of View -----
        fov_tab = QWidget()
        fov_tab_layout = QVBoxLayout(fov_tab)
        config_sub_tabs.addTab(fov_tab, "Field of View")

        fov_group = QGroupBox("Field of View Calculation")
        fov_layout = QFormLayout(fov_group)

        self.use_fov_mask_check = QCheckBox()
        self.use_fov_mask_check.setChecked(False)
        self.use_fov_mask_check.setToolTip(
            "Use a custom binary mask image to define the FoV shape.\n"
            "If unchecked, image corners will be used."
        )
        self.use_fov_mask_check.stateChanged.connect(self._toggle_fov_mask)
        fov_layout.addRow("Use Custom Mask:", self.use_fov_mask_check)

        self.fov_mask_widget = QWidget()
        fov_mask_layout = QHBoxLayout(self.fov_mask_widget)
        fov_mask_layout.setContentsMargins(0, 0, 0, 0)

        self.fov_mask_path_edit = QLineEdit()
        self.fov_mask_path_edit.setPlaceholderText("Path to binary mask PNG")
        self.fov_mask_path_edit.setToolTip("Binary mask image where white pixels define the FoV")
        fov_mask_layout.addWidget(self.fov_mask_path_edit)

        fov_mask_browse_btn = QPushButton("Browse...")
        fov_mask_browse_btn.clicked.connect(self.browse_fov_mask)
        fov_mask_layout.addWidget(fov_mask_browse_btn)

        self.fov_mask_widget.setEnabled(False)
        fov_layout.addRow("Mask Path:", self.fov_mask_widget)

        self.mask_simplify_spin = QDoubleSpinBox()
        self.mask_simplify_spin.setRange(0.0, 20.0)
        self.mask_simplify_spin.setSingleStep(0.5)
        self.mask_simplify_spin.setValue(2.0)
        self.mask_simplify_spin.setToolTip("Polygon simplification factor (higher = fewer points)")
        fov_layout.addRow("Simplify Epsilon:", self.mask_simplify_spin)

        # FoV frame filters
        fov_filter_label = QLabel("Frame Range:")
        fov_filter_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        fov_layout.addRow(fov_filter_label)

        self.fov_all_frames_check = QCheckBox("Use all frames")
        self.fov_all_frames_check.setChecked(True)
        self.fov_all_frames_check.stateChanged.connect(self.toggle_fov_frame_range)
        fov_layout.addRow("", self.fov_all_frames_check)

        self.fov_frame_range_widget = QWidget()
        fov_frame_range_layout = QHBoxLayout(self.fov_frame_range_widget)
        fov_frame_range_layout.setContentsMargins(0, 0, 0, 0)

        self.fov_start_frame_spin = QSpinBox()
        self.fov_start_frame_spin.setRange(0, 999999)
        self.fov_start_frame_spin.setValue(0)
        self.fov_start_frame_spin.setToolTip("First frame to include (inclusive)")
        fov_frame_range_layout.addWidget(QLabel("Start:"))
        fov_frame_range_layout.addWidget(self.fov_start_frame_spin)

        self.fov_end_frame_spin = QSpinBox()
        self.fov_end_frame_spin.setRange(0, 999999)
        self.fov_end_frame_spin.setValue(999999)
        self.fov_end_frame_spin.setToolTip("Last frame to include (inclusive)")
        fov_frame_range_layout.addWidget(QLabel("End:"))
        fov_frame_range_layout.addWidget(self.fov_end_frame_spin)

        self.fov_frame_range_widget.setEnabled(False)
        fov_layout.addRow("", self.fov_frame_range_widget)

        self.fov_sample_rate_spin = QSpinBox()
        self.fov_sample_rate_spin.setRange(1, 100)
        self.fov_sample_rate_spin.setValue(1)
        self.fov_sample_rate_spin.setToolTip("Calculate FoV for every Nth frame (1 = all frames)")
        fov_layout.addRow("Sample Rate:", self.fov_sample_rate_spin)

        fov_tab_layout.addWidget(fov_group)
        fov_tab_layout.addStretch()

        # ----- Sub-Tab 5: ALFS -----
        alfs_tab = QWidget()
        alfs_tab_layout = QVBoxLayout(alfs_tab)
        config_sub_tabs.addTab(alfs_tab, "ALFS/GeoTiff")

        alfs_group = QGroupBox("Airborne Light Field Sampling (ALFS) / GeoTIFF Generation")
        alfs_layout = QFormLayout(alfs_group)

        self.alfs_resolution_spin = QDoubleSpinBox()
        self.alfs_resolution_spin.setRange(0.001, 1.0)
        self.alfs_resolution_spin.setSingleStep(0.01)
        self.alfs_resolution_spin.setValue(0.05)
        self.alfs_resolution_spin.setDecimals(3)
        self.alfs_resolution_spin.setSuffix(" m/px")
        self.alfs_resolution_spin.setToolTip("Ground resolution in meters per pixel")
        alfs_layout.addRow("Ground Resolution:", self.alfs_resolution_spin)

        frame_range_label = QLabel("Frame Range:")
        alfs_layout.addRow(frame_range_label)

        self.alfs_all_frames_check = QCheckBox("Use all frames")
        self.alfs_all_frames_check.setChecked(True)
        self.alfs_all_frames_check.stateChanged.connect(self.toggle_alfs_frame_range)
        alfs_layout.addRow("", self.alfs_all_frames_check)

        self.alfs_frame_range_widget = QWidget()
        frame_range_layout = QHBoxLayout(self.alfs_frame_range_widget)
        frame_range_layout.setContentsMargins(0, 0, 0, 0)

        self.alfs_start_frame_spin = QSpinBox()
        self.alfs_start_frame_spin.setRange(0, 999999)
        self.alfs_start_frame_spin.setValue(0)
        self.alfs_start_frame_spin.setToolTip("First frame to include (inclusive)")
        frame_range_layout.addWidget(QLabel("Start:"))
        frame_range_layout.addWidget(self.alfs_start_frame_spin)

        self.alfs_end_frame_spin = QSpinBox()
        self.alfs_end_frame_spin.setRange(0, 999999)
        self.alfs_end_frame_spin.setValue(999999)
        self.alfs_end_frame_spin.setToolTip("Last frame to include (inclusive)")
        frame_range_layout.addWidget(QLabel("End:"))
        frame_range_layout.addWidget(self.alfs_end_frame_spin)

        self.alfs_frame_range_widget.setEnabled(False)
        alfs_layout.addRow("", self.alfs_frame_range_widget)

        self.preview_frames_btn = QPushButton("Detect Frame Count")
        self.preview_frames_btn.clicked.connect(self.detect_frame_count)
        self.preview_frames_btn.setToolTip("Load poses.json to detect available frames")
        alfs_layout.addRow("", self.preview_frames_btn)

        self.frame_count_label = QLabel("Frame count: Unknown")
        alfs_layout.addRow("", self.frame_count_label)

        self.alfs_crop_check = QCheckBox()
        self.alfs_crop_check.setChecked(True)
        self.alfs_crop_check.setToolTip("Crop output to content area")
        alfs_layout.addRow("Crop to Content:", self.alfs_crop_check)

        self.alfs_overviews_check = QCheckBox()
        self.alfs_overviews_check.setChecked(True)
        self.alfs_overviews_check.setToolTip("Create overview pyramids for faster GIS viewing")
        alfs_layout.addRow("Create Overviews:", self.alfs_overviews_check)

        self.alfs_tile_size_spin = QSpinBox()
        self.alfs_tile_size_spin.setRange(1024, 16384)
        self.alfs_tile_size_spin.setValue(8192)
        self.alfs_tile_size_spin.setSingleStep(1024)
        self.alfs_tile_size_spin.setToolTip("Maximum tile size for processing large images")
        alfs_layout.addRow("Max Tile Size:", self.alfs_tile_size_spin)

        # ALFS frame step (skip)
        self.alfs_frame_step_spin = QSpinBox()
        self.alfs_frame_step_spin.setRange(1, 100)
        self.alfs_frame_step_spin.setValue(1)
        self.alfs_frame_step_spin.setToolTip("Process every Nth frame (1 = all frames, 2 = every 2nd frame, etc.)")
        alfs_layout.addRow("Frame Step:", self.alfs_frame_step_spin)

        # Sampling mode
        self.alfs_sampling_check = QCheckBox("Enable")
        self.alfs_sampling_check.setChecked(False)
        self.alfs_sampling_check.setToolTip(
            "Create multiple small integral ALFS along the flight path instead of one large one"
        )
        self.alfs_sampling_check.stateChanged.connect(self.toggle_alfs_sampling)
        alfs_layout.addRow("Sampling Mode:", self.alfs_sampling_check)

        self.alfs_sampling_widget = QWidget()
        sampling_layout = QHBoxLayout(self.alfs_sampling_widget)
        sampling_layout.setContentsMargins(0, 0, 0, 0)

        self.alfs_sampling_rate_spin = QSpinBox()
        self.alfs_sampling_rate_spin.setRange(1, 9999)
        self.alfs_sampling_rate_spin.setValue(10)
        self.alfs_sampling_rate_spin.setToolTip("One central frame is picked every N frames")
        sampling_layout.addWidget(QLabel("Rate (frames):"))
        sampling_layout.addWidget(self.alfs_sampling_rate_spin)

        self.alfs_sampling_range_spin = QSpinBox()
        self.alfs_sampling_range_spin.setRange(0, 999)
        self.alfs_sampling_range_spin.setValue(5)
        self.alfs_sampling_range_spin.setToolTip(
            "Frames on each side of the central frame to include in each alfs"
        )
        sampling_layout.addWidget(QLabel("Range ±:"))
        sampling_layout.addWidget(self.alfs_sampling_range_spin)

        self.alfs_sampling_widget.setEnabled(False)
        alfs_layout.addRow("", self.alfs_sampling_widget)

        alfs_tab_layout.addWidget(alfs_group)
        alfs_tab_layout.addStretch()

        # ----- Sub-Tab 6: Orthomosaic (merge exported frame GeoTIFFs) -----
        ortho_tab = QWidget()
        ortho_tab_layout = QVBoxLayout(ortho_tab)
        config_sub_tabs.addTab(ortho_tab, "Orthomosaic")

        ortho_group = QGroupBox("Orthomosaic Generation")
        ortho_layout = QFormLayout(ortho_group)

        ortho_info = QLabel(
            "Merges the per-frame GeoTIFFs produced by 'Export Frames as GeoTIFF' "
            "into a single georeferenced orthomosaic. Export the frame GeoTIFFs for "
            "the selected camera first."
        )
        ortho_info.setWordWrap(True)
        ortho_layout.addRow(ortho_info)

        # Overlap resolution method ("mode")
        self.ortho_method_combo = QComboBox()
        self.ortho_method_combo.addItems([
            "FIRST - First frame wins",
            "LAST - Last frame wins",
            "MIN - Darkest pixel wins",
            "MAX - Brightest pixel wins",
            "AVERAGE - Mean of overlapping frames",
        ])
        self.ortho_method_combo.setCurrentIndex(4)
        self.ortho_method_combo.setToolTip(
            "How overlapping pixels from different frames are resolved"
        )
        ortho_layout.addRow("Merge Mode:", self.ortho_method_combo)

        # Frame range
        ortho_frame_range_label = QLabel("Frame Range:")
        ortho_layout.addRow(ortho_frame_range_label)

        self.ortho_all_frames_check = QCheckBox("Use all GeoTIFFs")
        self.ortho_all_frames_check.setChecked(True)
        self.ortho_all_frames_check.stateChanged.connect(self.toggle_ortho_frame_range)
        ortho_layout.addRow("", self.ortho_all_frames_check)

        self.ortho_frame_range_widget = QWidget()
        ortho_frame_range_layout = QHBoxLayout(self.ortho_frame_range_widget)
        ortho_frame_range_layout.setContentsMargins(0, 0, 0, 0)

        self.ortho_start_frame_spin = QSpinBox()
        self.ortho_start_frame_spin.setRange(0, 999999)
        self.ortho_start_frame_spin.setValue(0)
        self.ortho_start_frame_spin.setToolTip("First frame index to include (inclusive)")
        ortho_frame_range_layout.addWidget(QLabel("Start:"))
        ortho_frame_range_layout.addWidget(self.ortho_start_frame_spin)

        self.ortho_end_frame_spin = QSpinBox()
        self.ortho_end_frame_spin.setRange(0, 999999)
        self.ortho_end_frame_spin.setValue(999999)
        self.ortho_end_frame_spin.setToolTip("Last frame index to include (inclusive)")
        ortho_frame_range_layout.addWidget(QLabel("End:"))
        ortho_frame_range_layout.addWidget(self.ortho_end_frame_spin)

        self.ortho_frame_range_widget.setEnabled(False)
        ortho_layout.addRow("", self.ortho_frame_range_widget)

        ortho_tab_layout.addWidget(ortho_group)
        ortho_tab_layout.addStretch()

        # ----- Sub-Tab 7: SAM3 Segmentation -----
        sam3_tab = QWidget()
        sam3_tab_layout = QVBoxLayout(sam3_tab)
        config_sub_tabs.addTab(sam3_tab, "SAM3 Segmentation")

        sam3_api_group = QGroupBox("Roboflow API Configuration")
        sam3_api_layout = QFormLayout(sam3_api_group)

        self.sam3_api_key_edit = QLineEdit()
        self.sam3_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.sam3_api_key_edit.setPlaceholderText("Enter your Roboflow API key")
        self.sam3_api_key_edit.setToolTip("Your Roboflow API key for SAM3 inference")
        sam3_api_layout.addRow("API Key:", self.sam3_api_key_edit)

        # Toggle to show/hide API key
        self.show_api_key_check = QCheckBox("Show API key")
        self.show_api_key_check.stateChanged.connect(self._toggle_api_key_visibility)
        sam3_api_layout.addRow("", self.show_api_key_check)

        sam3_tab_layout.addWidget(sam3_api_group)

        sam3_prompts_group = QGroupBox("Segmentation Prompts")
        sam3_prompts_layout = QVBoxLayout(sam3_prompts_group)

        prompts_info = QLabel("Enter text prompts (one per line) for objects to segment:")
        prompts_info.setWordWrap(True)
        sam3_prompts_layout.addWidget(prompts_info)

        self.sam3_prompts_edit = QTextEdit()
        self.sam3_prompts_edit.setPlaceholderText("deer\nwild boar\nperson\ncar")
        self.sam3_prompts_edit.setMaximumHeight(100)
        self.sam3_prompts_edit.setToolTip("Text prompts for SAM3 segmentation. One prompt per line.")
        sam3_prompts_layout.addWidget(self.sam3_prompts_edit)

        sam3_tab_layout.addWidget(sam3_prompts_group)

        sam3_params_group = QGroupBox("Segmentation Parameters")
        sam3_params_layout = QFormLayout(sam3_params_group)

        self.sam3_confidence_spin = QDoubleSpinBox()
        self.sam3_confidence_spin.setRange(0.0, 1.0)
        self.sam3_confidence_spin.setSingleStep(0.05)
        self.sam3_confidence_spin.setValue(0.5)
        self.sam3_confidence_spin.setDecimals(2)
        self.sam3_confidence_spin.setToolTip("Minimum confidence threshold for segmentation masks")
        sam3_params_layout.addRow("Confidence Threshold:", self.sam3_confidence_spin)

        # SAM3 frame filters
        sam3_filter_label = QLabel("Frame Range:")
        sam3_filter_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        sam3_params_layout.addRow(sam3_filter_label)

        self.sam3_all_frames_check = QCheckBox("Use all frames")
        self.sam3_all_frames_check.setChecked(True)
        self.sam3_all_frames_check.stateChanged.connect(self.toggle_sam3_frame_range)
        sam3_params_layout.addRow("", self.sam3_all_frames_check)

        self.sam3_frame_range_widget = QWidget()
        sam3_frame_range_layout = QHBoxLayout(self.sam3_frame_range_widget)
        sam3_frame_range_layout.setContentsMargins(0, 0, 0, 0)

        self.sam3_start_frame_spin = QSpinBox()
        self.sam3_start_frame_spin.setRange(0, 999999)
        self.sam3_start_frame_spin.setValue(0)
        self.sam3_start_frame_spin.setToolTip("First frame to include (inclusive)")
        sam3_frame_range_layout.addWidget(QLabel("Start:"))
        sam3_frame_range_layout.addWidget(self.sam3_start_frame_spin)

        self.sam3_end_frame_spin = QSpinBox()
        self.sam3_end_frame_spin.setRange(0, 999999)
        self.sam3_end_frame_spin.setValue(999999)
        self.sam3_end_frame_spin.setToolTip("Last frame to include (inclusive)")
        sam3_frame_range_layout.addWidget(QLabel("End:"))
        sam3_frame_range_layout.addWidget(self.sam3_end_frame_spin)

        self.sam3_frame_range_widget.setEnabled(False)
        sam3_params_layout.addRow("", self.sam3_frame_range_widget)

        self.sam3_step_spin = QSpinBox()
        self.sam3_step_spin.setRange(1, 100)
        self.sam3_step_spin.setValue(1)
        self.sam3_step_spin.setToolTip("Process every Nth frame (1 = all frames, 2 = every 2nd frame, etc.)")
        sam3_params_layout.addRow("Frame Step:", self.sam3_step_spin)

        sam3_tab_layout.addWidget(sam3_params_group)
        sam3_tab_layout.addStretch()

        # =====================================================================
        # MAIN TAB 3: PROCESSING
        # =====================================================================
        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)
        main_tabs.addTab(processing_tab, "Pre-Processing")

        # Info button row
        self._processing_info_btn = QToolButton()
        self._processing_info_btn.setText("?")
        self._processing_info_btn.setFixedSize(20, 20)
        self._processing_info_btn.setStyleSheet(
            "QToolButton {"
            "  border-radius: 10px;"
            "  border: 1px solid palette(mid);"
            "  background: palette(button);"
            "  font-weight: bold;"
            "  font-size: 11px;"
            "}"
            "QToolButton:hover { background: palette(light); }"
            "QToolButton:pressed { background: palette(mid); }"
        )
        self._processing_info_btn.clicked.connect(self._show_processing_info)
        processing_info_row = QHBoxLayout()
        processing_info_row.setContentsMargins(0, 0, 0, 0)
        processing_info_row.addStretch()
        processing_info_row.addWidget(self._processing_info_btn)
        processing_layout.addLayout(processing_info_row)

        # Step buttons. The split mirrors the dependency graph of
        # EXCHANGE_FORMAT_PLAN.md §7: everything here derives from the poses and
        # the DEM and is independent of any animal, while the Processing tab
        # holds everything that depends on detections. Presenting them as one
        # numbered list implied a sequence that does not exist.
        steps_group = QGroupBox("Pre-Processing Steps")
        steps_btn_layout = QVBoxLayout(steps_group)

        proc_steps_group = QGroupBox("Processing Steps")
        proc_steps_layout = QVBoxLayout(proc_steps_group)

        # ----- P1: Extract Frames -----
        step1_row = QHBoxLayout()
        self.extract_btn = QPushButton("P1. Extract Frames")
        self.extract_btn.clicked.connect(self.run_extract_frames)
        self.extract_btn.setToolTip(
            "Extract frames from the selected camera's videos:\n"
            "Thermal (_T_) → frames_t/, RGB (_W_/_V_) → frames_w/"
        )
        self.extract_camera_combo = QComboBox()
        self.extract_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.extract_camera_combo.setFixedWidth(100)
        self.extract_camera_combo.setToolTip("Select camera source for frame extraction")
        self.extract_status = QLabel("⚪ Not started")
        step1_row.addWidget(self.extract_btn)
        step1_row.addWidget(self.extract_camera_combo)
        step1_row.addWidget(self.extract_status)
        steps_btn_layout.addLayout(step1_row)

        correction_info_label = QLabel(
            "If not already done: Use the correction tooling to determine "
            "positional and orientation errors after frame extraction."
        )
        correction_info_label.setWordWrap(True)
        correction_info_label.setStyleSheet("color: gray; font-size: 10px;")
        steps_btn_layout.addWidget(correction_info_label)

        # Separator after extraction
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        steps_btn_layout.addWidget(separator1)

        # ----- P2: Generate Flight Route -----
        step2_row = QHBoxLayout()
        self.flight_route_btn = QPushButton("P2. Generate Flight Route")
        self.flight_route_btn.clicked.connect(self.run_flight_route)
        self.flight_route_btn.setToolTip("Generate flight route polyline from camera positions")
        self.flight_route_camera_combo = QComboBox()
        self.flight_route_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.flight_route_camera_combo.setFixedWidth(100)
        self.flight_route_camera_combo.setToolTip("Select camera source for poses")
        self.flight_route_status = QLabel("⚪ Not started")
        step2_row.addWidget(self.flight_route_btn)
        step2_row.addWidget(self.flight_route_camera_combo)
        step2_row.addWidget(self.flight_route_status)
        steps_btn_layout.addLayout(step2_row)

        # -> Add Flight Route to QGIS
        add_flight_route_row = QHBoxLayout()
        self.add_flight_route_btn = QPushButton("   → Add Flight Route to QGIS")
        self.add_flight_route_btn.clicked.connect(self.add_flight_route_to_qgis)
        self.add_flight_route_status = QLabel("⚪")
        add_flight_route_row.addWidget(self.add_flight_route_btn)
        add_flight_route_row.addWidget(self.add_flight_route_status)
        steps_btn_layout.addLayout(add_flight_route_row)

        # ----- A1: Detect Animals -----
        step3_row = QHBoxLayout()
        self.detect_btn = QPushButton("A1. Detect Animals")
        self.detect_btn.clicked.connect(self.run_detection)
        self.detection_camera_combo = QComboBox()
        self.detection_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.detection_camera_combo.setFixedWidth(100)
        self.detection_camera_combo.setToolTip("Select camera source for frames")
        self.detect_status = QLabel("⚪ Not started")
        step3_row.addWidget(self.detect_btn)
        step3_row.addWidget(self.detection_camera_combo)
        step3_row.addWidget(self.detect_status)
        proc_steps_layout.addLayout(step3_row)

        # -> Geo-Reference Detections (sub-step of detection)
        step4_row = QHBoxLayout()
        self.georef_btn = QPushButton("   → Geo-Reference Detections")
        self.georef_btn.clicked.connect(self.run_georeference)
        self.georef_status = QLabel("⚪ Not started")
        step4_row.addWidget(self.georef_btn)
        step4_row.addWidget(self.georef_status)
        proc_steps_layout.addLayout(step4_row)

        # -> Add Detections to QGIS
        add_detections_row = QHBoxLayout()
        self.add_frame_detections_btn = QPushButton("   → Add Detections to QGIS")
        self.add_frame_detections_btn.clicked.connect(self.add_frame_detections_to_qgis)
        self.add_frame_detections_btn.setToolTip("Add geo-referenced bounding boxes as layers (one per frame)")
        self.frame_detections_status = QLabel("⚪")
        add_detections_row.addWidget(self.add_frame_detections_btn)
        add_detections_row.addWidget(self.frame_detections_status)
        proc_steps_layout.addLayout(add_detections_row)

        # -> Calculate Perpendicular
        perp_calc_row = QHBoxLayout()
        self.perpendicular_btn = QPushButton("   → Calculate Perpendicular")
        self.perpendicular_btn.clicked.connect(self.run_perpendicular)
        self.perpendicular_btn.setToolTip(
            "For each detection, find the nearest point on the flight route and\n"
            "calculate the perpendicular distance. Requires both flight route and\n"
            "geo-referenced detections."
        )
        self.perpendicular_status = QLabel("⚪")
        perp_calc_row.addWidget(self.perpendicular_btn)
        perp_calc_row.addWidget(self.perpendicular_status)
        proc_steps_layout.addLayout(perp_calc_row)

        # -> Add Perpendicular Lines to QGIS
        perp_add_row = QHBoxLayout()
        self.add_perpendicular_btn = QPushButton("   → Add Perpendicular Lines to QGIS")
        self.add_perpendicular_btn.clicked.connect(self.add_perpendicular_to_qgis)
        self.add_perpendicular_btn.setToolTip(
            "Add line layers connecting each detection to its nearest point on the flight route."
        )
        self.add_perpendicular_status = QLabel("⚪")
        perp_add_row.addWidget(self.add_perpendicular_btn)
        perp_add_row.addWidget(self.add_perpendicular_status)
        proc_steps_layout.addLayout(perp_add_row)

        # ----- A2: Track Animals Or Import -----
        step5_row = QHBoxLayout()
        self.track_btn = QPushButton("A2. Track Animals Or Import")
        self.track_btn.setToolTip(
            "Track geo-referenced detections.\n"
            "If a TRex NPZ folder is set (Config → Tracking tab), the pre-computed\n"
            "TRex tracklets are imported and geo-referenced instead."
        )
        self.track_btn.clicked.connect(self.run_tracking)
        self.tracking_camera_combo = QComboBox()
        self.tracking_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.tracking_camera_combo.setFixedWidth(100)
        self.tracking_camera_combo.setToolTip("Select camera source for geo-referenced detections")
        self.track_status = QLabel("⚪ Not started")
        step5_row.addWidget(self.track_btn)
        step5_row.addWidget(self.tracking_camera_combo)
        step5_row.addWidget(self.track_status)
        proc_steps_layout.addLayout(step5_row)

        # -> Add Tracks to QGIS
        add_tracks_row = QHBoxLayout()
        self.add_layers_btn = QPushButton("   → Add Tracks to QGIS")
        self.add_layers_btn.clicked.connect(self.add_tracks_to_qgis)
        self.layers_status = QLabel("⚪")
        add_tracks_row.addWidget(self.add_layers_btn)
        add_tracks_row.addWidget(self.layers_status)
        proc_steps_layout.addLayout(add_tracks_row)

        # -> Calculate Track Perpendicular
        track_perp_calc_row = QHBoxLayout()
        self.track_perpendicular_btn = QPushButton("   → Calculate Track Perpendicular")
        self.track_perpendicular_btn.clicked.connect(self.run_track_perpendicular)
        self.track_perpendicular_btn.setToolTip(
            "For each track, find the nearest point on the flight route to the\n"
            "last bounding box and calculate the perpendicular distance.\n"
            "Requires both flight route and tracked animals."
        )
        self.track_perpendicular_status = QLabel("⚪")
        track_perp_calc_row.addWidget(self.track_perpendicular_btn)
        track_perp_calc_row.addWidget(self.track_perpendicular_status)
        proc_steps_layout.addLayout(track_perp_calc_row)

        # -> Add Track Perpendicular Lines to QGIS
        track_perp_add_row = QHBoxLayout()
        self.add_track_perpendicular_btn = QPushButton("   → Add Track Perpendicular Lines to QGIS")
        self.add_track_perpendicular_btn.clicked.connect(self.add_track_perpendicular_to_qgis)
        self.add_track_perpendicular_btn.setToolTip(
            "Add line layers connecting the last detection of each track\n"
            "to its nearest point on the flight route."
        )
        self.add_track_perpendicular_status = QLabel("⚪")
        track_perp_add_row.addWidget(self.add_track_perpendicular_btn)
        track_perp_add_row.addWidget(self.add_track_perpendicular_status)
        proc_steps_layout.addLayout(track_perp_add_row)

        # ----- P3: Calculate Field of View -----
        step6_row = QHBoxLayout()
        self.calculate_fov_btn = QPushButton("P3. Calculate Field of View")
        self.calculate_fov_btn.clicked.connect(self.run_calculate_fov)
        self.calculate_fov_btn.setToolTip("Calculate and save camera FoV footprints for each frame")
        self.fov_camera_combo = QComboBox()
        self.fov_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.fov_camera_combo.setFixedWidth(100)
        self.fov_camera_combo.setToolTip("Select camera source for poses and calibration")
        self.calculate_fov_status = QLabel("⚪ Not started")
        step6_row.addWidget(self.calculate_fov_btn)
        step6_row.addWidget(self.fov_camera_combo)
        step6_row.addWidget(self.calculate_fov_status)
        steps_btn_layout.addLayout(step6_row)

        # -> Add FoV Layers to QGIS
        add_fov_row = QHBoxLayout()
        self.add_fov_btn = QPushButton("   → Add FoV Layers to QGIS")
        self.add_fov_btn.clicked.connect(self.add_fov_to_qgis)
        self.add_fov_btn.setToolTip("Add FoV polygons as QGIS layers (one per frame)")
        self.add_fov_status = QLabel("⚪")
        add_fov_row.addWidget(self.add_fov_btn)
        add_fov_row.addWidget(self.add_fov_status)
        steps_btn_layout.addLayout(add_fov_row)

        # -> Add Merged FoV to QGIS
        add_merged_fov_row = QHBoxLayout()
        self.add_merged_fov_btn = QPushButton("   → Add Merged FoV to QGIS")
        self.add_merged_fov_btn.clicked.connect(self.add_merged_fov_to_qgis)
        self.add_merged_fov_btn.setToolTip("Add union of all FoV polygons as single layer (for area calculation)")
        self.add_merged_fov_status = QLabel("⚪")
        add_merged_fov_row.addWidget(self.add_merged_fov_btn)
        add_merged_fov_row.addWidget(self.add_merged_fov_status)
        steps_btn_layout.addLayout(add_merged_fov_row)

        # ----- P4: Generate ALFS -----
        step7_row = QHBoxLayout()
        self.alfs_btn = QPushButton("P4. Generate ALFS")
        self.alfs_btn.clicked.connect(self.run_alfs)
        self.alfs_camera_combo = QComboBox()
        self.alfs_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.alfs_camera_combo.setFixedWidth(100)
        self.alfs_camera_combo.setToolTip("Select camera source for frames and poses")
        self.alfs_status = QLabel("⚪ Not started")
        step7_row.addWidget(self.alfs_btn)
        step7_row.addWidget(self.alfs_camera_combo)
        step7_row.addWidget(self.alfs_status)
        steps_btn_layout.addLayout(step7_row)

        # -> Add ALFS to QGIS
        add_alfs_row = QHBoxLayout()
        self.add_alfs_btn = QPushButton("   → Add ALFS to QGIS")
        self.add_alfs_btn.clicked.connect(self.add_alfs_to_qgis)
        self.add_alfs_status = QLabel("⚪")
        add_alfs_row.addWidget(self.add_alfs_btn)
        add_alfs_row.addWidget(self.add_alfs_status)
        steps_btn_layout.addLayout(add_alfs_row)

        # ----- P5: Export Frames as GeoTIFF -----
        step8_row = QHBoxLayout()
        self.export_geotiffs_btn = QPushButton("P5. Export Frames as GeoTIFF")
        self.export_geotiffs_btn.clicked.connect(self.run_export_geotiffs)
        self.geotiff_camera_combo = QComboBox()
        self.geotiff_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.geotiff_camera_combo.setFixedWidth(100)
        self.geotiff_camera_combo.setToolTip("Select camera source for frames and poses")
        self.geotiff_edge_erosion_spin = QSpinBox()
        self.geotiff_edge_erosion_spin.setRange(0, 10)
        self.geotiff_edge_erosion_spin.setValue(2)
        self.geotiff_edge_erosion_spin.setFixedWidth(55)
        self.geotiff_edge_erosion_spin.setToolTip(
            "Shrink each frame's valid footprint by this many pixels before\n"
            "saving, to remove the dark antialiased rim at the image border\n"
            "that otherwise shows as seams in the orthomosaic. 0 disables it."
        )
        self.export_geotiffs_status = QLabel("⚪ Not started")
        step8_row.addWidget(self.export_geotiffs_btn)
        step8_row.addWidget(self.geotiff_camera_combo)
        step8_row.addWidget(QLabel("Erode px:"))
        step8_row.addWidget(self.geotiff_edge_erosion_spin)
        step8_row.addWidget(self.export_geotiffs_status)
        steps_btn_layout.addLayout(step8_row)

        # -> Add Frame GeoTIFFs to QGIS
        add_geotiffs_row = QHBoxLayout()
        self.add_geotiffs_btn = QPushButton("   → Add Frame GeoTIFFs to QGIS")
        self.add_geotiffs_btn.clicked.connect(self.add_geotiffs_to_qgis)
        self.add_geotiffs_status = QLabel("⚪")
        add_geotiffs_row.addWidget(self.add_geotiffs_btn)
        add_geotiffs_row.addWidget(self.add_geotiffs_status)
        steps_btn_layout.addLayout(add_geotiffs_row)

        # ----- P6: Generate Orthomosaic (merge exported GeoTIFFs) -----
        ortho_step_row = QHBoxLayout()
        self.orthomosaic_btn = QPushButton("P6. Generate Orthomosaic")
        self.orthomosaic_btn.setToolTip(
            "Merge the exported frame GeoTIFFs (P5) into a single orthomosaic"
        )
        self.orthomosaic_btn.clicked.connect(self.run_orthomosaic)
        self.ortho_camera_combo = QComboBox()
        self.ortho_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.ortho_camera_combo.setFixedWidth(100)
        self.ortho_camera_combo.setToolTip("Select camera source (GeoTIFFs) for the orthomosaic")
        self.orthomosaic_status = QLabel("⚪ Not started")
        ortho_step_row.addWidget(self.orthomosaic_btn)
        ortho_step_row.addWidget(self.ortho_camera_combo)
        ortho_step_row.addWidget(self.orthomosaic_status)
        steps_btn_layout.addLayout(ortho_step_row)

        # -> Add Orthomosaic to QGIS
        add_ortho_row = QHBoxLayout()
        self.add_orthomosaic_btn = QPushButton("   → Add Orthomosaic to QGIS")
        self.add_orthomosaic_btn.clicked.connect(self.add_orthomosaic_to_qgis)
        self.add_orthomosaic_status = QLabel("⚪")
        add_ortho_row.addWidget(self.add_orthomosaic_btn)
        add_ortho_row.addWidget(self.add_orthomosaic_status)
        steps_btn_layout.addLayout(add_ortho_row)

        # ----- A3: Run SAM3 Segmentation -----
        step9_row = QHBoxLayout()
        self.sam3_segment_btn = QPushButton("A3. Run SAM3 Segmentation")
        self.sam3_segment_btn.clicked.connect(self.run_sam3_segmentation)
        self.sam3_segment_btn.setToolTip("Run SAM3 segmentation on extracted frames using Roboflow API")
        self.sam3_camera_combo = QComboBox()
        self.sam3_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.sam3_camera_combo.setFixedWidth(100)
        self.sam3_camera_combo.setToolTip("Select camera source for frames and poses")
        self.sam3_segment_status = QLabel("⚪ Not started")
        step9_row.addWidget(self.sam3_segment_btn)
        step9_row.addWidget(self.sam3_camera_combo)
        step9_row.addWidget(self.sam3_segment_status)
        proc_steps_layout.addLayout(step9_row)

        # ----- A4: Geo-Reference Segmentation -----
        step10_row = QHBoxLayout()
        self.sam3_georef_btn = QPushButton("A4. Geo-Reference Segmentation")
        self.sam3_georef_btn.clicked.connect(self.run_sam3_georeference)
        self.sam3_georef_btn.setToolTip("Convert pixel segmentation masks to world coordinates")
        self.sam3_georef_status = QLabel("⚪ Not started")
        step10_row.addWidget(self.sam3_georef_btn)
        step10_row.addWidget(self.sam3_georef_status)
        proc_steps_layout.addLayout(step10_row)

        # -> Add SAM3 Segmentation to QGIS
        add_sam3_row = QHBoxLayout()
        self.add_sam3_btn = QPushButton("   → Add Segmentation to QGIS")
        self.add_sam3_btn.clicked.connect(self.add_sam3_to_qgis)
        self.add_sam3_btn.setToolTip("Add geo-referenced segmentation masks as QGIS layers")
        self.add_sam3_status = QLabel("⚪")
        add_sam3_row.addWidget(self.add_sam3_btn)
        add_sam3_row.addWidget(self.add_sam3_status)
        proc_steps_layout.addLayout(add_sam3_row)

        # Re-evaluate step statuses when a camera selection changes so the
        # indicators always reflect the selected modality's outputs.
        for camera_combo in (self.extract_camera_combo, self.flight_route_camera_combo,
                             self.detection_camera_combo, self.tracking_camera_combo,
                             self.fov_camera_combo, self.alfs_camera_combo,
                             self.geotiff_camera_combo, self.ortho_camera_combo,
                             self.sam3_camera_combo):
            camera_combo.currentIndexChanged.connect(self._on_step_camera_changed)

        processing_layout.addWidget(steps_group)
        processing_layout.addStretch()

        # =====================================================================
        # MAIN TAB 3b: PROCESSING (animal-specific steps)
        # =====================================================================
        animal_tab = QWidget()
        animal_layout = QVBoxLayout(animal_tab)
        main_tabs.addTab(animal_tab, "Processing")

        animal_info = QLabel(
            "Steps that work on animals: detection, geo-referencing, tracking "
            "and segmentation. Everything here depends on the detections, so "
            "re-running an earlier step marks these out of date."
        )
        animal_info.setWordWrap(True)
        animal_info.setStyleSheet("color: gray; font-size: 10px;")
        animal_layout.addWidget(animal_info)
        animal_layout.addWidget(proc_steps_group)

        # ----- Export -----
        from .core import exporters as _exporters

        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        export_layout.addWidget(QLabel(
            "Write the detections, tracks and their attributes out in a "
            "standard format. Custom fields travel with them where the format "
            "has room; species names and enum values are resolved, so nothing "
            "receives an internal id."
        ))

        export_form = QHBoxLayout()
        export_form.addWidget(QLabel("Format:"))
        self.export_format_combo = QComboBox()
        for key, (label, _fn, _folder) in _exporters.EXPORTERS.items():
            self.export_format_combo.addItem(label, key)
        self.export_format_combo.currentIndexChanged.connect(
            self._on_export_format_changed)
        export_form.addWidget(self.export_format_combo, 1)

        self.export_camera_combo = QComboBox()
        self.export_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.export_camera_combo.setFixedWidth(100)
        export_form.addWidget(self.export_camera_combo)
        export_layout.addLayout(export_form)

        self.export_false_positives_check = QCheckBox(
            "Include detections labelled 'not-an-animal'")
        self.export_false_positives_check.setToolTip(
            "Survey formats keep them as a record of what was looked at and "
            "rejected; training formats drop them. Darwin Core always excludes "
            "them — a rejected detection is not an occurrence of anything."
        )
        export_layout.addWidget(self.export_false_positives_check)

        self.export_btn = QPushButton("Export…")
        self.export_btn.clicked.connect(self.run_export)
        export_layout.addWidget(self.export_btn)

        animal_layout.addWidget(export_group)
        self._on_export_format_changed()
        animal_layout.addStretch()

        # ---------------------------------------------------------------------
        # Shared run panel — below the tabs, so progress and the log are visible
        # whichever tab a step was started from.
        # ---------------------------------------------------------------------
        run_panel = QWidget()
        run_layout = QVBoxLayout(run_panel)
        run_layout.setContentsMargins(0, 0, 0, 0)

        # Progress bar with abort button
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar, stretch=1)

        self.abort_btn = QPushButton("Abort")
        self.abort_btn.setToolTip("Cancel the current running process")
        self.abort_btn.setEnabled(False)
        self.abort_btn.setFixedWidth(70)
        self.abort_btn.clicked.connect(self._abort_current_process)
        progress_layout.addWidget(self.abort_btn)

        run_layout.addLayout(progress_layout)

        # Refresh / Reset
        status_row = QHBoxLayout()
        self.refresh_status_btn = QPushButton("🔄 Refresh Status")
        self.refresh_status_btn.setToolTip("Check for existing outputs and QGIS layers to update status indicators")
        self.refresh_status_btn.clicked.connect(self._refresh_all_statuses)
        status_row.addWidget(self.refresh_status_btn)

        self.reset_stage_btn = QPushButton("Reset Step…")
        self.reset_stage_btn.setToolTip(
            "Delete a step's outputs so it re-runs from scratch, and flag "
            "everything that depended on it as out of date."
        )
        self.reset_stage_btn.clicked.connect(self.reset_stage)
        status_row.addWidget(self.reset_stage_btn)
        run_layout.addLayout(status_row)

        # Log output
        log_group = QGroupBox("Log Output")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        font = QFont("Courier")
        font.setPointSize(9)
        self.log_text.setFont(font)
        log_layout.addWidget(self.log_text)

        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.log_text.clear)
        log_layout.addWidget(clear_log_btn)

        run_layout.addWidget(log_group)
        scroll_layout.addWidget(run_panel)

        # =====================================================================
        # MAIN TAB 4: SURVEY ANALYTICS
        # =====================================================================
        analytics_tab = QWidget()
        analytics_tab_layout = QVBoxLayout(analytics_tab)
        main_tabs.addTab(analytics_tab, "Survey Analytics")

        analytics_info = QLabel(
            "Population-level products derived from geo-referenced detections or "
            "tracks. Tracks count each animal once; detections use every box. "
            "The run log is shown below the tabs."
        )
        analytics_info.setWordWrap(True)
        analytics_info.setStyleSheet("color: gray; font-size: 10px;")
        analytics_tab_layout.addWidget(analytics_info)

        # ----- Density Heatmap -----
        density_group = QGroupBox("Density Heatmap")
        density_layout = QVBoxLayout(density_group)

        density_desc = QLabel(
            "Kernel-density estimate raster (points per hectare) of animal "
            "locations. Requires geo-referenced detections or tracks."
        )
        density_desc.setWordWrap(True)
        density_desc.setStyleSheet("color: gray; font-size: 10px;")
        density_layout.addWidget(density_desc)

        density_params_row = QHBoxLayout()
        density_params_row.addWidget(QLabel("Source:"))
        self.density_source_combo = QComboBox()
        self.density_source_combo.addItems(["Detections", "Tracks"])
        self.density_source_combo.setToolTip(
            "Detections: every geo-referenced box.\n"
            "Tracks: one point per track (centroid), so each animal counts once."
        )
        density_params_row.addWidget(self.density_source_combo)
        density_params_row.addWidget(QLabel("Cell (m):"))
        self.density_cell_spin = QDoubleSpinBox()
        self.density_cell_spin.setRange(0.5, 500.0)
        self.density_cell_spin.setValue(5.0)
        self.density_cell_spin.setSingleStep(1.0)
        self.density_cell_spin.setToolTip("Output raster cell size in metres")
        density_params_row.addWidget(self.density_cell_spin)
        density_params_row.addWidget(QLabel("Bandwidth (m):"))
        self.density_bandwidth_spin = QDoubleSpinBox()
        self.density_bandwidth_spin.setRange(1.0, 2000.0)
        self.density_bandwidth_spin.setValue(25.0)
        self.density_bandwidth_spin.setSingleStep(5.0)
        self.density_bandwidth_spin.setToolTip(
            "Gaussian kernel bandwidth (smoothing radius) in metres")
        density_params_row.addWidget(self.density_bandwidth_spin)
        density_params_row.addStretch()
        density_layout.addLayout(density_params_row)

        density_run_row = QHBoxLayout()
        self.density_btn = QPushButton("→ Generate Density Heatmap")
        self.density_btn.clicked.connect(self.run_density_heatmap)
        self.density_btn.setToolTip(
            "Kernel-density estimate raster (points per hectare) of animal locations."
        )
        self.density_status = QLabel("⚪")
        density_run_row.addWidget(self.density_btn)
        density_run_row.addWidget(self.density_status)
        density_layout.addLayout(density_run_row)

        density_add_row = QHBoxLayout()
        self.add_density_btn = QPushButton("→ Add Density Heatmap to QGIS")
        self.add_density_btn.clicked.connect(self.add_density_heatmap_to_qgis)
        self.add_density_btn.setToolTip("Load the density raster with a graduated colour ramp.")
        self.add_density_status = QLabel("⚪")
        density_add_row.addWidget(self.add_density_btn)
        density_add_row.addWidget(self.add_density_status)
        density_layout.addLayout(density_add_row)

        analytics_tab_layout.addWidget(density_group)

        # ----- Distance Sampling -----
        ds_group = QGroupBox("Distance Sampling")
        ds_layout = QVBoxLayout(ds_group)

        ds_desc = QLabel(
            "Line-transect density/abundance with 95% confidence intervals, fitted "
            "from the perpendicular distances. Run 'Calculate Perpendicular' (or "
            "'Calculate Track Perpendicular') on the Processing tab first."
        )
        ds_desc.setWordWrap(True)
        ds_desc.setStyleSheet("color: gray; font-size: 10px;")
        ds_layout.addWidget(ds_desc)

        ds_layout.addWidget(self._build_projects_group(
            "ds",
            "BAMBI target folders to pool for the distance-sampling estimate. "
            "Their perpendicular distances are combined and the flight-route "
            "lengths summed into the total effort L."))

        ds_params_row = QHBoxLayout()
        ds_params_row.addWidget(QLabel("Source:"))
        self.ds_source_combo = QComboBox()
        self.ds_source_combo.addItems(["Detections", "Tracks"])
        self.ds_source_combo.setToolTip(
            "Which perpendicular distances to analyse. Tracks (one observation per\n"
            "animal) are usually preferred for distance sampling."
        )
        ds_params_row.addWidget(self.ds_source_combo)
        ds_params_row.addWidget(QLabel("Truncation (m):"))
        self.ds_truncation_spin = QDoubleSpinBox()
        self.ds_truncation_spin.setRange(0.0, 100000.0)
        self.ds_truncation_spin.setValue(0.0)
        self.ds_truncation_spin.setSingleStep(5.0)
        self.ds_truncation_spin.setSpecialValueText("auto (95th pct)")
        self.ds_truncation_spin.setToolTip(
            "Discard observations beyond this distance. 0 = automatic "
            "(95th percentile of distances).")
        ds_params_row.addWidget(self.ds_truncation_spin)
        ds_params_row.addStretch()
        ds_layout.addLayout(ds_params_row)

        ds_run_row = QHBoxLayout()
        self.distance_sampling_btn = QPushButton("→ Estimate Density (Distance Sampling)")
        self.distance_sampling_btn.clicked.connect(self.run_distance_sampling)
        self.distance_sampling_btn.setToolTip(
            "Fit half-normal & hazard-rate detection functions to the perpendicular\n"
            "distances and estimate density/abundance with 95% confidence intervals.\n"
            "Requires 'Calculate Perpendicular' (or track perpendicular) first."
        )
        self.distance_sampling_status = QLabel("⚪")
        ds_run_row.addWidget(self.distance_sampling_btn)
        ds_run_row.addWidget(self.distance_sampling_status)
        ds_layout.addLayout(ds_run_row)

        analytics_tab_layout.addWidget(ds_group)

        # ----- Coverage Map -----
        coverage_group = QGroupBox("Coverage Map")
        coverage_layout = QVBoxLayout(coverage_group)

        coverage_desc = QLabel(
            "Combines the exported frame GeoTIFFs like the orthomosaic, but "
            "counts how many frames cover each pixel (survey effort/overlap). "
            "Requires 'Export Frames as GeoTIFF' on the Processing tab."
        )
        coverage_desc.setWordWrap(True)
        coverage_desc.setStyleSheet("color: gray; font-size: 10px;")
        coverage_layout.addWidget(coverage_desc)

        coverage_params_row = QHBoxLayout()
        coverage_params_row.addWidget(QLabel("Camera:"))
        self.coverage_camera_combo = QComboBox()
        self.coverage_camera_combo.addItems(["Thermal", "RGB"])
        self.coverage_camera_combo.setToolTip(
            "Which camera's exported frame GeoTIFFs to combine.")
        coverage_params_row.addWidget(self.coverage_camera_combo)
        coverage_params_row.addWidget(QLabel("Cell (m):"))
        self.coverage_cell_spin = QDoubleSpinBox()
        self.coverage_cell_spin.setRange(0.0, 100.0)
        self.coverage_cell_spin.setValue(1.0)
        self.coverage_cell_spin.setSingleStep(0.5)
        self.coverage_cell_spin.setSpecialValueText("native")
        self.coverage_cell_spin.setToolTip(
            "Output raster cell size in metres.\n"
            "0 = native resolution of the exported GeoTIFFs (larger output).")
        coverage_params_row.addWidget(self.coverage_cell_spin)
        coverage_params_row.addStretch()
        coverage_layout.addLayout(coverage_params_row)

        coverage_run_row = QHBoxLayout()
        self.coverage_btn = QPushButton("→ Generate Coverage Map")
        self.coverage_btn.clicked.connect(self.run_coverage_map)
        self.coverage_btn.setToolTip(
            "Count, per ground pixel, the number of overlapping frame GeoTIFFs.\n"
            "1 = seen once, N = seen in N frames, transparent = never covered."
        )
        self.coverage_status = QLabel("⚪")
        coverage_run_row.addWidget(self.coverage_btn)
        coverage_run_row.addWidget(self.coverage_status)
        coverage_layout.addLayout(coverage_run_row)

        coverage_add_row = QHBoxLayout()
        self.add_coverage_btn = QPushButton("→ Add Coverage Map to QGIS")
        self.add_coverage_btn.clicked.connect(self.add_coverage_map_to_qgis)
        self.add_coverage_btn.setToolTip(
            "Load the coverage raster with a graduated colour ramp.")
        self.add_coverage_status = QLabel("⚪")
        coverage_add_row.addWidget(self.add_coverage_btn)
        coverage_add_row.addWidget(self.add_coverage_status)
        coverage_layout.addLayout(coverage_add_row)

        analytics_tab_layout.addWidget(coverage_group)

        # ----- Population Estimation (Transects) -----
        pop_group = QGroupBox("Population Estimation (Transects)")
        pop_layout = QVBoxLayout(pop_group)

        pop_desc = QLabel(
            "Density from the transects of a split flight: each track is assigned "
            "to the transect nearest in perpendicular distance, and the monitored "
            "area of a transect is the union of its frames' field-of-view "
            "footprints. Requires transects (Transect Splitting Tool), "
            "'Calculate Field of View' and 'Calculate Track Perpendicular'."
        )
        pop_desc.setWordWrap(True)
        pop_desc.setStyleSheet("color: gray; font-size: 10px;")
        pop_layout.addWidget(pop_desc)

        pop_layout.addWidget(self._build_projects_group(
            "pop",
            "BAMBI projects to combine. Every project's transects are pooled "
            "into one count/area table and the estimators run on all of them "
            "together; each added project supplies its own dem.json so its "
            "transects are georeferenced with the right DEM origin.",
            with_dem=True))

        pop_params_row = QHBoxLayout()
        pop_params_row.addWidget(QLabel("Camera:"))
        self.pop_camera_combo = QComboBox()
        self.pop_camera_combo.addItems(["Thermal", "RGB"])
        self.pop_camera_combo.setToolTip(
            "Modality of the transects, FoV footprints and tracks. Frame indices\n"
            "are modality-specific, so all three must be the same camera.")
        pop_params_row.addWidget(self.pop_camera_combo)
        pop_params_row.addWidget(QLabel("Truncation (m):"))
        self.pop_truncation_spin = QDoubleSpinBox()
        self.pop_truncation_spin.setRange(0.0, 100000.0)
        self.pop_truncation_spin.setValue(0.0)
        self.pop_truncation_spin.setSingleStep(5.0)
        self.pop_truncation_spin.setSpecialValueText("off (assign all)")
        self.pop_truncation_spin.setToolTip(
            "Maximum perpendicular distance between a track and its transect.\n"
            "Tracks farther from every transect are not counted.\n"
            "0 = assign every track to its nearest transect.")
        pop_params_row.addWidget(self.pop_truncation_spin)
        pop_params_row.addStretch()
        pop_layout.addLayout(pop_params_row)

        # Study area: the region the density is extrapolated to. It is NOT the
        # monitored area (the searched strips, i.e. the density's denominator)
        # — setting it to that just returns the animals already counted.
        pop_area_row = QHBoxLayout()
        pop_area_row.addWidget(QLabel("Study area (ha):"))
        self.pop_study_area_spin = QDoubleSpinBox()
        self.pop_study_area_spin.setRange(0.0, 10_000_000.0)
        self.pop_study_area_spin.setValue(0.0)
        self.pop_study_area_spin.setSingleStep(10.0)
        self.pop_study_area_spin.setSpecialValueText("off (density only)")
        self.pop_study_area_spin.setToolTip(
            "Size of the region the density is extrapolated to an abundance for.\n"
            "This is not the monitored area (the searched strips) — extrapolating\n"
            "to that would just return the animals already counted.\n"
            "0 = report densities only.")
        pop_area_row.addWidget(self.pop_study_area_spin)
        self.pop_study_area_auto_check = QCheckBox("Use flight FoV area")
        self.pop_study_area_auto_check.setToolTip(
            "Derive the study area from the flight's total field-of-view coverage:\n"
            "the union of every frame's footprint. Unlike the summed transect areas\n"
            "it counts ground seen by two transects only once, and it also includes\n"
            "frames that belong to no transect.")
        self.pop_study_area_auto_check.toggled.connect(
            self.pop_study_area_spin.setDisabled)
        pop_area_row.addWidget(self.pop_study_area_auto_check)
        pop_area_row.addStretch()
        pop_layout.addLayout(pop_area_row)

        pop_methods_row = QHBoxLayout()
        pop_methods_row.addWidget(QLabel("Methods:"))
        self.pop_naive_check = QCheckBox("Naive")
        self.pop_naive_check.setChecked(True)
        self.pop_naive_check.setToolTip(
            "Total animals divided by the total monitored area "
            "(sum(count) / sum(ha) × 100).")
        self.pop_bootstrap_check = QCheckBox("Bootstrap")
        self.pop_bootstrap_check.setChecked(True)
        self.pop_bootstrap_check.setToolTip(
            "Resample the transects with replacement, recompute the naive\n"
            "density each time and report its mean, SE and percentile 95% CI.")
        self.pop_zinb_check = QCheckBox("ZINB")
        self.pop_zinb_check.setChecked(True)
        self.pop_zinb_check.setToolTip(
            "Zero-inflated negative binomial regression (count ~ ha, constant\n"
            "zero-inflation), for transect counts with many zeros and\n"
            "overdispersion. Needs at least 4 transects of differing area.")
        pop_methods_row.addWidget(self.pop_naive_check)
        pop_methods_row.addWidget(self.pop_bootstrap_check)
        pop_methods_row.addWidget(self.pop_zinb_check)
        pop_methods_row.addSpacing(12)
        pop_methods_row.addWidget(QLabel("Bootstrap iterations:"))
        self.pop_n_boot_spin = QSpinBox()
        self.pop_n_boot_spin.setRange(10, 100000)
        self.pop_n_boot_spin.setValue(999)
        self.pop_n_boot_spin.setSingleStep(100)
        self.pop_n_boot_spin.setToolTip("Number of bootstrap resamples.")
        pop_methods_row.addWidget(self.pop_n_boot_spin)
        pop_methods_row.addWidget(QLabel("Seed:"))
        self.pop_seed_spin = QSpinBox()
        self.pop_seed_spin.setRange(0, 2_147_483_647)
        self.pop_seed_spin.setValue(42)
        self.pop_seed_spin.setToolTip(
            "Random seed of the bootstrap, so a run is reproducible.")
        pop_methods_row.addWidget(self.pop_seed_spin)
        pop_methods_row.addStretch()
        pop_layout.addLayout(pop_methods_row)

        pop_run_row = QHBoxLayout()
        self.population_btn = QPushButton("→ Estimate Population (Transects)")
        self.population_btn.clicked.connect(self.run_population_estimation)
        self.population_btn.setToolTip(
            "Assign the tracks to the transects, measure each transect's monitored\n"
            "area from its FoV footprints and estimate the density with the\n"
            "selected methods."
        )
        self.population_status = QLabel("⚪")
        pop_run_row.addWidget(self.population_btn)
        pop_run_row.addWidget(self.population_status)
        pop_layout.addLayout(pop_run_row)

        pop_add_row = QHBoxLayout()
        self.add_transect_areas_btn = QPushButton("→ Add Transect Areas to QGIS")
        self.add_transect_areas_btn.clicked.connect(self.add_transect_areas_to_qgis)
        self.add_transect_areas_btn.setToolTip(
            "Load the monitored area of every transect (the merged FoV footprint) "
            "as a labelled polygon layer.")
        self.add_transect_areas_status = QLabel("⚪")
        pop_add_row.addWidget(self.add_transect_areas_btn)
        pop_add_row.addWidget(self.add_transect_areas_status)
        pop_layout.addLayout(pop_add_row)

        analytics_tab_layout.addWidget(pop_group)

        # Progress mirror for analytics runs (main log lives on the Processing tab)
        self.analytics_progress_bar = QProgressBar()
        self.analytics_progress_bar.setRange(0, 100)
        self.analytics_progress_bar.setValue(0)
        analytics_tab_layout.addWidget(self.analytics_progress_bar)

        analytics_tab_layout.addStretch()

    def _populate_tracker_backends(self):
        """Populate the tracker backend dropdown with available trackers."""
        from .tracker_manager import get_tracker_manager

        self._tracker_manager = get_tracker_manager()
        trackers = self._tracker_manager.get_available_trackers()

        self._tracker_ids = []
        for tracker_id, info in trackers.items():
            self.tracker_backend_combo.addItem(info.name)
            self._tracker_ids.append(tracker_id)

        # Store import errors for later display (log widget may not exist yet)
        self._tracker_import_errors = self._tracker_manager.get_import_errors()

    def _refresh_trackers(self):
        """Refresh the available trackers list."""
        # Clear current items
        self.tracker_backend_combo.clear()
        self._tracker_ids = []

        # Refresh the tracker manager
        self._tracker_manager.refresh_available_trackers()

        # Re-populate
        trackers = self._tracker_manager.get_available_trackers()

        for tracker_id, info in trackers.items():
            self.tracker_backend_combo.addItem(info.name)
            self._tracker_ids.append(tracker_id)

        # Check for import errors
        errors = self._tracker_manager.get_import_errors()
        if errors:
            error_msg = "Import errors:\n"
            for pkg, error in errors.items():
                error_msg += f"• {pkg}: {error}\n"
                self.log(f"Import error for {pkg}: {error}")
            QMessageBox.warning(self, "Tracker Import Errors", error_msg)
        else:
            count = len(trackers)
            self.log(f"Refreshed trackers: {count} available")

        # Update UI for selected tracker
        if self._tracker_ids:
            self._on_tracker_changed(0)

    def _on_tracker_changed(self, index: int):
        """Handle tracker selection change."""
        if index < 0 or index >= len(self._tracker_ids):
            return

        tracker_id = self._tracker_ids[index]
        trackers = self._tracker_manager.get_available_trackers()
        info = trackers.get(tracker_id)

        if info:
            # Update info label
            self.tracker_info_label.setText(info.description)

            # Show/hide ReID options
            self.reid_widget.setVisible(info.requires_reid)

            # Show/hide built-in parameters
            is_builtin = tracker_id == "builtin"
            self.builtin_params_widget.setVisible(is_builtin)

    def _on_reid_model_changed(self, index: int):
        """Handle ReID model selection change."""
        # Show custom path field only when "custom" is selected
        self.custom_reid_widget.setVisible(index == 2)

    def _browse_custom_reid(self):
        """Browse for custom ReID weights file."""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select ReID Weights", "", "PyTorch Model (*.pt *.pth)")
        if file:
            self.custom_reid_path_edit.setText(file)

    def _show_tracker_params_help(self):
        """Show help dialog with available parameters for the selected tracker."""
        if not hasattr(self, '_tracker_ids') or not self._tracker_ids:
            return

        index = self.tracker_backend_combo.currentIndex()
        if index < 0 or index >= len(self._tracker_ids):
            return

        tracker_id = self._tracker_ids[index]
        params = self._tracker_manager.get_tracker_constructor_params(tracker_id)

        if not params:
            QMessageBox.information(
                self,
                "Tracker Parameters",
                "No configurable parameters available for this tracker.\n\n"
                "The tracker uses its default settings."
            )
            return

        # Build help text
        help_text = f"Available parameters for {tracker_id}:\n\n"
        for name, (param_type, default, desc) in params.items():
            type_name = param_type.__name__ if hasattr(param_type, '__name__') else str(param_type)
            default_str = f" = {default}" if default is not None else ""
            help_text += f"• {name} ({type_name}){default_str}\n  {desc}\n\n"

        help_text += "\nExample JSON:\n"
        example = {}
        for name, (param_type, default, _) in list(params.items())[:3]:
            if default is not None:
                example[name] = default
        help_text += json.dumps(example, indent=2)

        QMessageBox.information(self, "Tracker Parameters", help_text)

    def _get_selected_tracker_id(self) -> str:
        """Get the currently selected tracker ID."""
        if not hasattr(self, '_tracker_ids') or not self._tracker_ids:
            return "builtin"
        index = self.tracker_backend_combo.currentIndex()
        if 0 <= index < len(self._tracker_ids):
            return self._tracker_ids[index]
        return "builtin"

    def _get_selected_reid_model(self) -> str:
        """Get the currently selected ReID model."""
        index = self.reid_model_combo.currentIndex()
        return ["osnet", "bambi", "custom"][index]

    def log(self, message: str):
        """Add a message to the log."""
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration from UI elements."""
        # Parse CRS from text input
        crs_text = self.target_crs_edit.text().strip().upper()
        epsg = self._parse_epsg_from_text(crs_text)

        # Get rotation values - convert to radians if UI is in degrees mode
        rot_x = self.rot_x_spin.value()
        rot_y = self.rot_y_spin.value()
        rot_z = self.rot_z_spin.value()
        if self._is_using_degrees():
            rot_x = math.radians(rot_x)
            rot_y = math.radians(rot_y)
            rot_z = math.radians(rot_z)

        video_mode = self.video_mode_check.isChecked()
        embedded_srt = video_mode and self.embedded_srt_check.isChecked()

        thermal_video_paths = [p.strip() for p in self.thermal_video_paths_edit.text().split(",") if p.strip()]
        rgb_video_paths = [p.strip() for p in self.rgb_video_paths_edit.text().split(",") if p.strip()]

        # Thermal visualisation: the lo/hi spin boxes apply in the threshold
        # mode, the curve in the curve mode — the other one is sent as None.
        threshold_mode = self.thermal_vis_mode_combo.currentIndex() == 0
        curve_mode = self.thermal_vis_mode_combo.currentIndex() == 1

        return {
            # Input mode
            "input_mode": "video" if video_mode else "photo",

            # Video inputs
            "thermal_video_paths": thermal_video_paths,
            "thermal_srt_paths": (
                self._resolve_embedded_srts(thermal_video_paths)
                if embedded_srt
                else [p.strip() for p in self.thermal_srt_paths_edit.text().split(",") if p.strip()]
            ),
            "thermal_calibration_path": (
                self.thermal_calibration_path_edit.text()
                if self.thermal_calib_preset_combo.currentIndex() == 0 else ""
            ),
            "thermal_calibration_data": (
                THERMAL_CALIBRATIONS.get(self.thermal_calib_preset_combo.currentText())
                if self.thermal_calib_preset_combo.currentIndex() > 0 else None
            ),
            "rgb_video_paths": rgb_video_paths,
            "rgb_srt_paths": (
                self._resolve_embedded_srts(rgb_video_paths)
                if embedded_srt
                else [p.strip() for p in self.rgb_srt_paths_edit.text().split(",") if p.strip()]
            ),
            "rgb_calibration_path": (
                self.rgb_calibration_path_edit.text()
                if self.rgb_calib_preset_combo.currentIndex() == 0 else ""
            ),
            "rgb_calibration_data": (
                RGB_CALIBRATIONS.get(self.rgb_calib_preset_combo.currentText())
                if self.rgb_calib_preset_combo.currentIndex() > 0 else None
            ),
            # Photo inputs
            "thermal_photo_dir": self.thermal_photo_dir_edit.text(),
            "thermal_photo_filter": self.thermal_photo_filter_check.isChecked(),
            "thermal_photo_calibration_path": (
                self.thermal_photo_calibration_path_edit.text()
                if self.thermal_photo_calib_preset_combo.currentIndex() == 0 else ""
            ),
            "thermal_photo_calibration_data": (
                THERMAL_CALIBRATIONS.get(self.thermal_photo_calib_preset_combo.currentText())
                if self.thermal_photo_calib_preset_combo.currentIndex() > 0 else None
            ),
            "rgb_photo_dir": self.rgb_photo_dir_edit.text(),
            "rgb_photo_filter": self.rgb_photo_filter_check.isChecked(),
            "rgb_photo_calibration_path": (
                self.rgb_photo_calibration_path_edit.text()
                if self.rgb_photo_calib_preset_combo.currentIndex() == 0 else ""
            ),
            "rgb_photo_calibration_data": (
                RGB_CALIBRATIONS.get(self.rgb_photo_calib_preset_combo.currentText())
                if self.rgb_photo_calib_preset_combo.currentIndex() > 0 else None
            ),
            "timezone_offset_hours": self._get_timezone_offset(),

            # Frame extraction
            "extract_skip": self.extract_skip_spin.value(),
            "extract_limit": (
                self.extract_limit_spin.value()
                if self.extract_limit_check.isChecked() else None
            ),
            "extract_sampling_rate": (
                self.extract_sampling_rate_spin.value()
                if self.extract_sampling_rate_check.isChecked() else None
            ),
            "preserve_aspect_ratio": self.preserve_aspect_ratio_check.isChecked(),
            "use_gimbal_heading": self.use_gimbal_heading_check.isChecked(),
            "no_images": self.no_images_check.isChecked(),

            # Thermal visualisation
            "thermal_photo_colormap": (
                self.thermal_vis_cmap_combo.currentText()
                if self.thermal_vis_cmap_combo.currentText() != "(none)" else None
            ),
            "thermal_photo_lo_threshold": (
                self.thermal_vis_lo_spin.value()
                if threshold_mode and self.thermal_vis_lo_check.isChecked() else None
            ),
            "thermal_photo_hi_threshold": (
                self.thermal_vis_hi_spin.value()
                if threshold_mode and self.thermal_vis_hi_check.isChecked() else None
            ),
            "thermal_photo_curve": (
                self._thermal_vis_curve.to_dict()
                if curve_mode and self._thermal_vis_curve is not None else None
            ),

            # Common inputs
            "airdata_path": self.airdata_path_edit.text(),
            "dem_path": self.dem_path_edit.text(),
            "correction_path": self.correction_path_edit.text(),

            # Output
            "target_folder": self.target_folder_edit.text(),
            "target_epsg": epsg,
            "write_legacy_text_outputs": (
                self.legacy_text_outputs_check.isChecked()
                if hasattr(self, 'legacy_text_outputs_check') else True),

            # Detection
            "thermal_model_path": self.thermal_model_path_edit.text() or None,
            "rgb_model_path": self.rgb_model_path_edit.text() or None,
            "min_confidence": self.confidence_spin.value(),
            "detect_use_all_frames": (
                self.detect_all_frames_check.isChecked()
                if hasattr(self, 'detect_all_frames_check') else True),
            "detect_start_frame": (
                self.detect_start_frame_spin.value()
                if hasattr(self, 'detect_start_frame_spin') else 0),
            "detect_end_frame": (
                self.detect_end_frame_spin.value()
                if hasattr(self, 'detect_end_frame_spin') else 999999),
            "detect_sample_rate": (
                self.detect_sample_rate_spin.value()
                if hasattr(self, 'detect_sample_rate_spin') else 1),

            # Correction factors (rotations always in radians)
            "translation": {
                "x": self.trans_x_spin.value(),
                "y": self.trans_y_spin.value(),
                "z": self.trans_z_spin.value()
            },
            "rotation": {
                "x": rot_x,
                "y": rot_y,
                "z": rot_z
            },
            "additional_corrections": self._additional_corrections.copy(),

            # Tracking
            "tracker_id": self._get_selected_tracker_id(),
            "reid_model": self._get_selected_reid_model(),
            "custom_reid_path": self.custom_reid_path_edit.text() if hasattr(self, 'custom_reid_path_edit') else "",
            "tracker_params_json": self.tracker_params_edit.toPlainText() if hasattr(self,
                                                                                     'tracker_params_edit') else "",
            "iou_threshold": self.iou_threshold_spin.value(),
            "max_age": self.max_age_spin.value(),
            "max_center_distance": self.max_center_dist_spin.value(),
            "tracker_mode": self.tracker_mode_combo.currentText(),
            "class_aware": self.class_aware_check.isChecked(),
            "interpolate": self.interpolate_check.isChecked(),

            # ALFS
            "alfs_ground_resolution": self.alfs_resolution_spin.value(),
            "alfs_dem_metadata_path": self.dem_metadata_path_edit.text() or None,
            "alfs_use_all_frames": self.alfs_all_frames_check.isChecked(),
            "alfs_start_frame": (
                self.alfs_start_frame_spin.value()
                if not self.alfs_all_frames_check.isChecked() else None),
            "alfs_end_frame": (
                self.alfs_end_frame_spin.value()
                if not self.alfs_all_frames_check.isChecked() else None),
            "alfs_crop_to_content": self.alfs_crop_check.isChecked(),
            "alfs_create_overviews": self.alfs_overviews_check.isChecked(),
            "alfs_max_tile_size": self.alfs_tile_size_spin.value(),
            "alfs_frame_step": (
                self.alfs_frame_step_spin.value()
                if hasattr(self, 'alfs_frame_step_spin') else 1),
            "alfs_sampling_mode": (
                self.alfs_sampling_check.isChecked()
                if hasattr(self, 'alfs_sampling_check') else False),
            "alfs_sampling_rate": (
                self.alfs_sampling_rate_spin.value()
                if hasattr(self, 'alfs_sampling_rate_spin') else 10),
            "alfs_sampling_range": (
                self.alfs_sampling_range_spin.value()
                if hasattr(self, 'alfs_sampling_range_spin') else 5),

            # Field of View
            "use_fov_mask": self.use_fov_mask_check.isChecked(),
            "fov_mask_path": self.fov_mask_path_edit.text() if hasattr(self, 'fov_mask_path_edit') else "",
            "mask_simplify_epsilon": self.mask_simplify_spin.value() if hasattr(self, 'mask_simplify_spin') else 2.0,
            "fov_use_all_frames": self.fov_all_frames_check.isChecked() if hasattr(self,
                                                                                   'fov_all_frames_check') else True,
            "fov_start_frame": self.fov_start_frame_spin.value() if hasattr(self, 'fov_start_frame_spin') else 0,
            "fov_end_frame": self.fov_end_frame_spin.value() if hasattr(self, 'fov_end_frame_spin') else 999999,
            "fov_sample_rate": self.fov_sample_rate_spin.value() if hasattr(self, 'fov_sample_rate_spin') else 1,

            # SAM3 Segmentation
            "sam3_api_key": self.sam3_api_key_edit.text() if hasattr(self, 'sam3_api_key_edit') else "",
            "sam3_prompts": [p.strip() for p in self.sam3_prompts_edit.toPlainText().split("\n") if
                             p.strip()] if hasattr(self, 'sam3_prompts_edit') else [],
            "sam3_confidence": self.sam3_confidence_spin.value() if hasattr(self, 'sam3_confidence_spin') else 0.5,
            "sam3_use_all_frames": (
                self.sam3_all_frames_check.isChecked()
                if hasattr(self, 'sam3_all_frames_check') else True),
            "sam3_start_frame": (
                self.sam3_start_frame_spin.value()
                if hasattr(self, 'sam3_start_frame_spin') else 0),
            "sam3_end_frame": (
                self.sam3_end_frame_spin.value()
                if hasattr(self, 'sam3_end_frame_spin') else 999999),
            "sam3_step": (
                self.sam3_step_spin.value()
                if hasattr(self, 'sam3_step_spin') else 1),

            # Flight Route options
            "filter_gps_origin": (
                self.filter_gps_origin_check.isChecked()
                if hasattr(self, 'filter_gps_origin_check') else True),

            # TRex import
            "trex_npz_dir": self.trex_npz_dir_edit.text() if hasattr(self, 'trex_npz_dir_edit') else "",
            "trex_already_undistorted": (
                self.trex_undistorted_check.isChecked()
                if hasattr(self, 'trex_undistorted_check') else False),

            # Camera selections for processing steps
            "flight_route_camera": "T" if self.flight_route_camera_combo.currentIndex() == 0 else "W",
            "detection_camera": "T" if self.detection_camera_combo.currentIndex() == 0 else "W",
            "georeference_camera": "T" if self.detection_camera_combo.currentIndex() == 0 else "W",
            "tracking_camera": "T" if self.tracking_camera_combo.currentIndex() == 0 else "W",
            "fov_camera": "T" if self.fov_camera_combo.currentIndex() == 0 else "W",
            "alfs_camera": "T" if self.alfs_camera_combo.currentIndex() == 0 else "W",
            "geotiff_camera": "T" if self.geotiff_camera_combo.currentIndex() == 0 else "W",
            "geotiff_edge_erosion_px": (
                self.geotiff_edge_erosion_spin.value()
                if hasattr(self, 'geotiff_edge_erosion_spin') else 2),
            "sam3_camera": "T" if self.sam3_camera_combo.currentIndex() == 0 else "W",
            # Orthomosaic (merge of exported frame GeoTIFFs)
            "ortho_camera": "T" if self.ortho_camera_combo.currentIndex() == 0 else "W",
            "ortho_method": self.ortho_method_combo.currentText().split(" - ")[0].lower(),
            "ortho_use_all_frames": self.ortho_all_frames_check.isChecked(),
            "ortho_start_frame": (
                self.ortho_start_frame_spin.value()
                if not self.ortho_all_frames_check.isChecked() else None),
            "ortho_end_frame": (
                self.ortho_end_frame_spin.value()
                if not self.ortho_all_frames_check.isChecked() else None),

            # Survey analytics: density heatmap
            "density_source": (
                self.density_source_combo.currentText().lower()
                if hasattr(self, 'density_source_combo') else "detections"),
            "density_cell_size": (
                self.density_cell_spin.value()
                if hasattr(self, 'density_cell_spin') else 5.0),
            "density_bandwidth": (
                self.density_bandwidth_spin.value()
                if hasattr(self, 'density_bandwidth_spin') else 25.0),

            # Survey analytics: coverage map
            "coverage_camera": (
                ("T" if self.coverage_camera_combo.currentIndex() == 0 else "W")
                if hasattr(self, 'coverage_camera_combo') else "T"),
            "coverage_cell_size": (
                self.coverage_cell_spin.value()
                if hasattr(self, 'coverage_cell_spin') else 1.0),

            # Survey analytics: distance sampling
            "ds_source": (
                self.ds_source_combo.currentText().lower()
                if hasattr(self, 'ds_source_combo') else "detections"),
            "ds_truncation": (
                self.ds_truncation_spin.value()
                if hasattr(self, 'ds_truncation_spin') else 0.0),
            "ds_project_folders": (
                [e["target"] for e in self._resolve_project_entries("ds")]
                if hasattr(self, 'ds_projects_list') else []),

            # Survey analytics: transect population estimation
            "pop_camera": (
                ("T" if self.pop_camera_combo.currentIndex() == 0 else "W")
                if hasattr(self, 'pop_camera_combo') else "T"),
            "pop_truncation": (
                self.pop_truncation_spin.value()
                if hasattr(self, 'pop_truncation_spin') else 0.0),
            "pop_study_area_ha": (
                self.pop_study_area_spin.value()
                if hasattr(self, 'pop_study_area_spin') else 0.0),
            "pop_study_area_auto": (
                self.pop_study_area_auto_check.isChecked()
                if hasattr(self, 'pop_study_area_auto_check') else False),
            "pop_methods": self._selected_population_methods(),
            "pop_n_boot": (
                self.pop_n_boot_spin.value()
                if hasattr(self, 'pop_n_boot_spin') else 999),
            "pop_seed": (
                self.pop_seed_spin.value()
                if hasattr(self, 'pop_seed_spin') else 42),
            "pop_project_folders": (
                self._resolve_project_entries("pop")
                if hasattr(self, 'pop_projects_list') else []),
        }

    def _selected_population_methods(self) -> list:
        """The population estimators ticked on the Survey Analytics tab."""
        checks = (
            ("naive", "pop_naive_check"),
            ("bootstrap", "pop_bootstrap_check"),
            ("zinb", "pop_zinb_check"),
        )
        methods = [
            name for name, attr in checks
            if not hasattr(self, attr) or getattr(self, attr).isChecked()
        ]
        return methods

    # ------------------------------------------------------------------ #
    # Multi-project selector (shared by distance sampling & population)
    # ------------------------------------------------------------------ #

    def _build_projects_group(self, prefix: str, tooltip: str,
                              with_dem: bool = False) -> QGroupBox:
        """A project list + add/remove + 'Add current project' selector.

        Lets a Survey Analytics tool run over several BAMBI target folders at
        once and combine the results. Each list entry stores its target folder
        and (when *with_dem*) the project's DEM metadata JSON, under the item's
        ``Qt.UserRole`` data. The widgets are stored on ``self`` under
        ``{prefix}_projects_list`` and ``{prefix}_include_current_check`` so
        :meth:`_resolve_project_entries` can read them back.

        :param with_dem: population estimation needs each added project's
            ``dem.json`` to place its transects, so its Add button opens a
            two-picker dialog; distance sampling has no DEM dependency and just
            picks a folder.
        """
        box = QGroupBox("Projects")
        layout = QVBoxLayout(box)

        description = (
            "Analyse one or more BAMBI target folders together — each project "
            "is processed on its own and the results are combined. Leave the "
            "list empty and keep 'Add current project' ticked to analyse only "
            "the active project (same as before)."
        )
        if with_dem:
            description += (
                " Added projects need their DEM metadata (dem.json) so their "
                "transects can be georeferenced; the current project reuses "
                "the DEM configured on the Processing tab."
            )
        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(desc)

        lst = QListWidget()
        lst.setToolTip(tooltip)
        lst.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        lst.setMaximumHeight(90)
        setattr(self, f"{prefix}_projects_list", lst)
        layout.addWidget(lst)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add Project…" if with_dem else "+ Add Folder…")
        add_btn.setToolTip(
            "Add a project (target folder + dem.json) to analyse." if with_dem
            else "Add a BAMBI target folder to analyse.")
        if with_dem:
            add_btn.clicked.connect(lambda: self._add_project_with_dem(prefix))
        else:
            add_btn.clicked.connect(lambda: self._add_project_folder(prefix))
        remove_btn = QPushButton("− Remove Selected")
        remove_btn.setToolTip("Remove the selected project(s) from the list.")
        remove_btn.clicked.connect(lambda: self._remove_project_folders(prefix))
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch()

        include_current = QCheckBox("Add current project")
        include_current.setChecked(True)
        include_current.setToolTip(
            "Include the active project's target folder in the analysis, in "
            "addition to any projects listed above.")
        setattr(self, f"{prefix}_include_current_check", include_current)
        btn_row.addWidget(include_current)
        layout.addLayout(btn_row)
        return box

    def _project_item_role(self):
        """Qt.UserRole, spelled for both PyQt5 and PyQt6."""
        return Qt.ItemDataRole.UserRole

    def _append_project_item(self, prefix: str, target: str, dem: str = ""):
        """Add one project entry to a tool's list, de-duplicated by target."""
        lst = getattr(self, f"{prefix}_projects_list", None)
        if lst is None:
            return
        key = os.path.normcase(os.path.abspath(target))
        for i in range(lst.count()):
            data = lst.item(i).data(self._project_item_role()) or {}
            if os.path.normcase(os.path.abspath(data.get("target", ""))) == key:
                return
        label = target
        if dem:
            label = f"{target}    —    DEM: {os.path.basename(dem)}"
        item = QListWidgetItem(label)
        item.setData(self._project_item_role(), {"target": target, "dem": dem})
        item.setToolTip(f"Target: {target}" + (f"\nDEM: {dem}" if dem else ""))
        lst.addItem(item)

    def _add_project_folder(self, prefix: str):
        """Append a chosen BAMBI target folder (no DEM) to a tool's list."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select BAMBI Target Folder")
        if folder:
            self._append_project_item(prefix, folder)

    def _add_project_with_dem(self, prefix: str):
        """Prompt for a target folder + its dem.json, then add the project."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Add Project")
        dlg.setMinimumWidth(520)
        layout = QVBoxLayout(dlg)
        form = QFormLayout()

        target_edit = QLineEdit()
        target_edit.setPlaceholderText("BAMBI target folder")
        target_browse = QPushButton("Browse…")

        def _pick_target():
            folder = QFileDialog.getExistingDirectory(
                dlg, "Select BAMBI Target Folder")
            if folder:
                target_edit.setText(folder)
                # Offer the DEM sitting next to the mesh if it is obvious.
                if not dem_edit.text():
                    for name in ("dem.json", "dem_mesh.json"):
                        guess = os.path.join(folder, name)
                        if os.path.isfile(guess):
                            dem_edit.setText(guess)
                            break
        target_browse.clicked.connect(_pick_target)
        target_row = QHBoxLayout()
        target_row.addWidget(target_edit)
        target_row.addWidget(target_browse)
        form.addRow("Target folder:", target_row)

        dem_edit = QLineEdit()
        dem_edit.setPlaceholderText("DEM metadata JSON (dem.json)")
        dem_browse = QPushButton("Browse…")

        def _pick_dem():
            start = target_edit.text().strip() or ""
            file, _ = QFileDialog.getOpenFileName(
                dlg, "Select DEM Metadata JSON", start, "JSON files (*.json)")
            if file:
                dem_edit.setText(file)
        dem_browse.clicked.connect(_pick_dem)
        dem_row = QHBoxLayout()
        dem_row.addWidget(dem_edit)
        dem_row.addWidget(dem_browse)
        form.addRow("DEM metadata:", dem_row)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        target = target_edit.text().strip()
        dem = dem_edit.text().strip()
        if not target:
            QMessageBox.warning(self, "Add Project",
                                "Please choose a target folder.")
            return
        if not dem or not os.path.isfile(dem):
            QMessageBox.warning(
                self, "Add Project",
                "Please choose the project's DEM metadata JSON (dem.json).")
            return
        self._append_project_item(prefix, target, dem)

    def _remove_project_folders(self, prefix: str):
        """Remove the selected project(s) from a tool's list."""
        lst = getattr(self, f"{prefix}_projects_list", None)
        if lst is None:
            return
        for item in lst.selectedItems():
            lst.takeItem(lst.row(item))

    def _resolve_project_entries(self, prefix: str) -> List[Dict[str, str]]:
        """Ordered, de-duplicated ``{"target", "dem"}`` entries for a tool.

        The projects explicitly listed, followed by the active project (with an
        empty DEM, so the worker reuses the configured DEM) when 'Add current
        project' is ticked. Blank targets and duplicates (by normalised target
        path) are dropped; existence is not checked here — the run handlers
        report missing folders/files to the user.
        """
        entries: List[Dict[str, str]] = []
        lst = getattr(self, f"{prefix}_projects_list", None)
        if lst is not None:
            for i in range(lst.count()):
                data = lst.item(i).data(self._project_item_role()) or {}
                entries.append({"target": data.get("target", ""),
                                "dem": data.get("dem", "")})
        check = getattr(self, f"{prefix}_include_current_check", None)
        if check is not None and check.isChecked():
            current = self.target_folder_edit.text().strip()
            if current:
                entries.append({"target": current, "dem": ""})

        seen = set()
        resolved: List[Dict[str, str]] = []
        for entry in entries:
            target = (entry.get("target") or "").strip()
            if not target:
                continue
            key = os.path.normcase(os.path.abspath(target))
            if key in seen:
                continue
            seen.add(key)
            resolved.append({"target": target, "dem": (entry.get("dem") or "").strip()})
        return resolved

    def _warn_missing_project_files(self, problems: Dict[str, List[str]]):
        """Report which required files are missing in which project(s)."""
        blocks = []
        for folder, missing in problems.items():
            name = os.path.basename(os.path.normpath(folder)) or folder
            bullets = "\n".join(f"      • {m}" for m in missing)
            blocks.append(f"  {name}\n    ({folder})\n{bullets}")
        body = "\n\n".join(blocks)
        QMessageBox.warning(
            self, "Missing Prerequisites",
            "The analysis was aborted — required files are missing in the "
            "following project(s):\n\n" + body)

    def validate_inputs(self, required_fields: list) -> bool:
        """Validate that required input fields are filled."""
        config = self.get_config()
        missing = []

        field_labels = {
            "thermal_video_paths": "Thermal video files",
            "thermal_srt_paths": "Thermal SRT files",
            "thermal_calibration_path": "Thermal calibration file",
            "rgb_video_paths": "RGB video files",
            "rgb_srt_paths": "RGB SRT files",
            "rgb_calibration_path": "RGB calibration file",
            "airdata_path": "AirData CSV",
            "dem_path": "DEM file",
            "target_folder": "Target folder"
        }

        for field in required_fields:
            value = config.get(field)
            if not value or (isinstance(value, list) and len(value) == 0):
                missing.append(field_labels.get(field, field))

        if missing:
            QMessageBox.warning(
                self,
                "Missing Inputs",
                "Please provide the following required inputs:\n\n• " + "\n• ".join(missing)
            )
            return False
        return True

    # Browse functions
    def browse_thermal_videos(self):
        """Browse for thermal video files and auto-detect related files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Thermal Video Files", "", "Video Files (*.mp4 *.MP4 *.avi *.mov)")
        if files:
            # Filter for thermal videos (containing _T_)
            thermal_files = [f for f in files if '_T_' in os.path.basename(f).upper()]
            if not thermal_files:
                thermal_files = files  # Use all if no _T_ pattern found

            self.thermal_video_paths_edit.setText(", ".join(thermal_files))

            # Get the folder of the first video
            video_folder = os.path.dirname(thermal_files[0])

            # Auto-populate thermal SRT paths (skipped in embedded SRT mode)
            if not self.embedded_srt_check.isChecked():
                srts = [f.replace(".MP4", ".SRT").replace(".mp4", ".srt") for f in thermal_files]
                existing_srts = [s for s in srts if os.path.exists(s)]
                if existing_srts:
                    self.thermal_srt_paths_edit.setText(", ".join(existing_srts))
                    self.log(f"Auto-detected {len(existing_srts)} thermal SRT file(s)")

            # Auto-detect T_calib.json (only when using custom file mode)
            if (self.thermal_calib_preset_combo.currentIndex() == 0 and not self.thermal_calibration_path_edit.text()):
                t_calib_path = os.path.join(video_folder, "T_calib.json")
                if os.path.exists(t_calib_path):
                    self.thermal_calibration_path_edit.setText(t_calib_path)
                    self.log("Auto-detected thermal calibration: T_calib.json")

            # Auto-detect common files if not already set
            self._auto_detect_common_files(video_folder)

    def browse_rgb_videos(self):
        """Browse for RGB video files and auto-detect related files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select RGB Video Files", "", "Video Files (*.mp4 *.MP4 *.avi *.mov)")
        if files:
            # Filter for RGB videos (containing _W_ or _V_)
            rgb_files = [f for f in files if
                         '_W_' in os.path.basename(f).upper() or '_V_' in os.path.basename(f).upper()]
            if not rgb_files:
                rgb_files = files  # Use all if no _W_ or _V_ pattern found

            self.rgb_video_paths_edit.setText(", ".join(rgb_files))

            # Get the folder of the first video
            video_folder = os.path.dirname(rgb_files[0])

            # Auto-populate RGB SRT paths (skipped in embedded SRT mode)
            if not self.embedded_srt_check.isChecked():
                srts = [f.replace(".MP4", ".SRT").replace(".mp4", ".srt") for f in rgb_files]
                existing_srts = [s for s in srts if os.path.exists(s)]
                if existing_srts:
                    self.rgb_srt_paths_edit.setText(", ".join(existing_srts))
                    self.log(f"Auto-detected {len(existing_srts)} RGB SRT file(s)")

            # Auto-detect W_calib.json (only when using custom file mode)
            if (self.rgb_calib_preset_combo.currentIndex() == 0 and not self.rgb_calibration_path_edit.text()):
                w_calib_path = os.path.join(video_folder, "W_calib.json")
                if os.path.exists(w_calib_path):
                    self.rgb_calibration_path_edit.setText(w_calib_path)
                    self.log("Auto-detected RGB calibration: W_calib.json")

            # Auto-detect common files if not already set
            self._auto_detect_common_files(video_folder)

    def _auto_detect_common_files(self, video_folder: str):
        """Auto-detect common input files from video folder (core.flight_files).

        :param video_folder: Folder containing video files
        """
        from .core.flight_files import detect_common_files
        found = detect_common_files(video_folder, log_fn=self.log)

        # Auto-detect AirData CSV (first CSV in folder)
        if not self.airdata_path_edit.text() and "airdata" in found:
            self.airdata_path_edit.setText(found["airdata"])
            self.log(f"Auto-detected AirData CSV: {os.path.basename(found['airdata'])}")

        # Auto-detect DEM GLTF/GLB (+ metadata JSON with matching name)
        if not self.dem_path_edit.text() and "dem" in found:
            self.dem_path_edit.setText(found["dem"])
            self.log(f"Auto-detected DEM: {os.path.basename(found['dem'])}")
            if "dem_metadata" in found:
                self.dem_metadata_path_edit.setText(found["dem_metadata"])
                self.log(
                    "Auto-detected DEM metadata: "
                    f"{os.path.basename(found['dem_metadata'])}")

        # Auto-detect thermal calibration if not set (only in custom file mode)
        if (self.thermal_calib_preset_combo.currentIndex() == 0 and not self.thermal_calibration_path_edit.text()):
            if "thermal_calibration" in found:
                self.thermal_calibration_path_edit.setText(found["thermal_calibration"])
                self.log("Auto-detected thermal calibration: T_calib.json")

        # Auto-detect RGB calibration if not set (only in custom file mode)
        if (self.rgb_calib_preset_combo.currentIndex() == 0 and not self.rgb_calibration_path_edit.text()):
            if "rgb_calibration" in found:
                self.rgb_calibration_path_edit.setText(found["rgb_calibration"])
                self.log("Auto-detected RGB calibration: W_calib.json")

        # Auto-detect correction.json and load values
        if not self.correction_path_edit.text() and "correction" in found:
            self.correction_path_edit.setText(found["correction"])
            self.load_correction_values(found["correction"])

        # Auto-set target folder to "qgis" subfolder
        if not self.target_folder_edit.text():
            qgis_folder = found["target_folder"]
            self.target_folder_edit.setText(qgis_folder)
            self.log("Auto-set target folder: qgis/")
            # Create the folder if it doesn't exist
            if not os.path.exists(qgis_folder):
                try:
                    os.makedirs(qgis_folder, exist_ok=True)
                    self.log(f"Created target folder: {qgis_folder}")
                except Exception as e:
                    self.log(f"Warning: Could not create qgis folder: {e}")
            else:
                # Check for existing outputs if folder already exists
                self._check_existing_outputs(qgis_folder)

        # Try to auto-detect CRS from available sources
        self._try_auto_detect_crs_silent()

    def _resolve_embedded_srts(self, video_paths: List[str]) -> List[str]:
        """Extract embedded SRT streams via ffmpeg (core.flight_files).

        Uses a temp directory as a persistent cache — if a .srt file for a
        video already exists there it is reused without re-running ffmpeg.
        """
        from .core.flight_files import extract_embedded_srts
        if not hasattr(self, "_srt_tmpdir") or not os.path.isdir(self._srt_tmpdir):
            self._srt_tmpdir = tempfile.mkdtemp(prefix="bambi_srt_")
        return extract_embedded_srts(video_paths, self._srt_tmpdir, log_fn=self.log)

    def browse_thermal_srts(self):
        """Browse for thermal SRT files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Thermal SRT Files", "", "SRT Files (*.srt *.SRT)")
        if files:
            self.thermal_srt_paths_edit.setText(", ".join(files))

    def browse_rgb_srts(self):
        """Browse for RGB SRT files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select RGB SRT Files", "", "SRT Files (*.srt *.SRT)")
        if files:
            self.rgb_srt_paths_edit.setText(", ".join(files))

    @staticmethod
    def _compute_timezone_offset(tz_name: str) -> Optional[float]:
        """Return the current UTC offset in hours for a given IANA timezone name."""
        from .core.timezone_detection import timezone_offset_hours
        return timezone_offset_hours(tz_name)

    def _update_timezone_offset_label(self):
        """Update the UTC offset label next to the manual timezone combo."""
        tz_name = self.timezone_combo.currentText().strip()
        offset = self._compute_timezone_offset(tz_name)
        if offset is None:
            self.timezone_offset_label.setText("(unknown timezone)")
        else:
            sign = "+" if offset >= 0 else ""
            self.timezone_offset_label.setText(f"UTC{sign}{offset:g}h")

    def _on_tz_auto_changed(self, state: int):
        """Toggle between auto-detected and manual timezone."""
        auto = bool(state)
        self.tz_manual_widget.setVisible(not auto)
        self.tz_manual_hint.setVisible(not auto)
        self.tz_auto_result_label.setVisible(auto)
        if auto:
            self._refresh_auto_timezone_offset()

    def _refresh_auto_timezone_offset(self):
        """Recompute the auto-detected offset and update the result label."""
        offset = self._compute_auto_timezone_offset()
        if offset is not None:
            sign = "+" if offset >= 0 else ""
            self.tz_auto_result_label.setText(f"Detected offset: UTC{sign}{offset:g}h")
        else:
            self.tz_auto_result_label.setText(
                "(could not detect — check SRT/photo dir & AirData)")

    def _get_timezone_offset(self) -> float:
        """Return the timezone offset in hours, auto-detected or manual."""
        if self.tz_auto_check.isChecked():
            offset = self._compute_auto_timezone_offset()
            if offset is not None:
                return offset
            self.log("Warning: Auto timezone detection failed, falling back to manual timezone")
        return self._compute_timezone_offset(self.timezone_combo.currentText().strip()) or 0.0

    def _compute_auto_timezone_offset(self) -> Optional[float]:
        """Auto-detect timezone offset.

        In video mode: compares SRT frame timestamps with AirData isVideo UTC timestamps.
        In photo mode (or as fallback): compares photo EXIF timestamps with AirData isPhoto UTC timestamps.
        Returns the offset in whole hours or None if detection fails.
        """
        if self.video_mode_check.isChecked():
            offset = self._compute_auto_tz_from_srt()
            if offset is not None:
                return offset
        return self._compute_auto_tz_from_exif()

    def _compute_auto_tz_from_srt(self) -> Optional[float]:
        """Match SRT local timestamps against AirData isVideo UTC timestamps."""
        from .core.timezone_detection import offset_from_srt
        srt_text = self.rgb_srt_paths_edit.text().strip()
        srt_paths = [p.strip() for p in srt_text.split(",") if p.strip()]
        return offset_from_srt(srt_paths, self.airdata_path_edit.text().strip())

    def _compute_auto_tz_from_exif(self) -> Optional[float]:
        """Match photo EXIF timestamps against AirData isPhoto UTC timestamps."""
        from .core.timezone_detection import offset_from_exif
        photo_dir = (self.thermal_photo_dir_edit.text().strip() or self.rgb_photo_dir_edit.text().strip())
        return offset_from_exif(photo_dir, self.airdata_path_edit.text().strip())

    def _collect_airdata_hours(self, flag_column_lower: str) -> Optional[list]:
        """Read UTC hours from AirData rows where the flag column is truthy."""
        from .core.timezone_detection import airdata_utc_hours
        return airdata_utc_hours(
            self.airdata_path_edit.text().strip(), flag_column_lower)

    def _on_input_mode_changed(self, state: int):
        """Toggle between video and photo input panels."""
        video_mode = bool(state)
        self.video_inputs_widget.setVisible(video_mode)
        self.photo_inputs_widget.setVisible(not video_mode)
        self.thermal_vis_group.setVisible(not video_mode)
        self.embedded_srt_check.setVisible(video_mode)
        self._airdata_exif_info_label.setVisible(not video_mode)
        if not video_mode:
            self.embedded_srt_check.setChecked(False)
        self.extract_sampling_rate_check.setEnabled(video_mode)
        if not video_mode:
            self.extract_sampling_rate_check.setChecked(False)

    def _on_embedded_srt_changed(self, state: int):
        """Show/hide the SRT file input rows depending on embedded SRT mode."""
        show_fields = not bool(state)
        self._thermal_srt_field.setVisible(show_fields)
        self._thermal_srt_label.setVisible(show_fields)
        self._rgb_srt_field.setVisible(show_fields)
        self._rgb_srt_label.setVisible(show_fields)

    def _show_input_mode_info(self):
        """Show an info popup describing the currently active input mode."""
        if self.video_mode_check.isChecked():
            title = "Video Mode"
            text = (
                "<b>Video Mode</b> is active.<br><br>"
                "The plugin processes continuous drone video recordings together with "
                "SRT subtitle files, which provide per-frame timestamps used to match "
                "each frame to a high-precision GPS position from the AirData flight log.<br><br>"
                "Note: When using newer drones like DJI’s M4T SRT is embedded in "
                "the videos. In such a case, please activate the ‘Embedded SRT’ "
                "checkbox<br><br>"
                "Practical tips: To ensure high-quality and consistent video "
                "recordings, configure your drone mission with stable flight "
                "parameters and a fixed camera setup. Plan the flight at a constant "
                "altitude between 30 m and 60 m above ground, depending on the "
                "terrain and desired coverage, and maintain a steady speed of "
                "3–7 m/s throughout the mission. "
                "Although the exact heading is not critical, the drone’s orientation "
                "should remain constant for the entire flight, and yaw rotations "
                "should be avoided. A practical approach is to configure each "
                "waypoint so that the drone faces north, ensuring a stable and "
                "repeatable camera perspective. "
                "Set the gimbal pitch to −90° so the camera is pointing "
                "straight down (nadir). This provides a consistent top-down view "
                "and simplifies later processing of the video data. "
                "Don’t forget the subtitle feature in the video settings, otherwise "
                "no SRT file is created/embedded!"
                "Finally, start video recording at the first waypoint and stop "
                "recording at the last waypoint to capture the full survey area in "
                "one continuous sequence while avoiding unnecessary footage. "
            )
        else:
            title = "Photo Mode"
            text = (
                "<b>Photo Mode</b> is active.<br><br>"
                "The plugin processes a series of still images captured during a mapping "
                "flight. GPS positions and global orientation are matched to images by comparing image EXIF "
                "timestamps with the AirData flight log.<br><br>"
                "Note: In photo mode, images are associated with the AirData file "
                "in a fixed sequential order for cases with same timestamps due to "
                "temporal ambiguities caused by second-level precision and the lack "
                "of sub-second information. Consequently, the complete dataset must "
                "always be provided. If individual images are missing, "
                "georeferencing will fail."
            )
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()

    def _show_processing_info(self):
        """Show an info popup describing the processing pipeline."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Processing Pipeline")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(
            "<b>Processing Pipeline</b><br><br>"
            "The steps are split across two tabs, following what they depend "
            "on. <b>Pre-Processing</b> derives from the drone poses and the "
            "DEM and is independent of any animal; <b>Processing</b> depends "
            "on the detections. The two branches can be run independently — "
            "re-running detection marks the Processing steps out of date and "
            "leaves the Pre-Processing ones alone.<br><br>"
            "<b>Pre-Processing</b><br><br>"
            "<b>P1 — Extract Frames</b><br>"
            "Decodes and undistorts frames for the selected camera (thermal or RGB); "
            "matches GPS positions from the AirData log via SRT timestamps (video) or "
            "EXIF timestamps (photo). Run once per camera you need.<br><br>"
            "<b>P2 — Generate Flight Route</b><br>"
            "Creates a GPS flight-path line layer and per-frame camera-position points "
            "from the extracted pose data.<br><br>"
            "<b>P3 — Calculate Field of View</b><br>"
            "Computes per-frame camera footprint polygons on the ground using the DEM.<br><br>"
            "<b>P4 — Generate ALFS</b><br>"
            "Projects all frames onto the DEM surface and blends them into a "
            "georeferenced GeoTIFF mosaic.<br><br>"
            "<b>P5 — Export Frames as GeoTIFF</b><br>"
            "Exports individual frames as separate georeferenced GeoTIFFs.<br><br>"
            "<b>P6 — Generate Orthomosaic</b><br>"
            "Merges the exported frame GeoTIFFs (P5) into a single true "
            "orthomosaic, using all frames or a selected range and a configurable "
            "overlap merge mode.<br><br>"
            "<b>Processing</b><br><br>"
            "<b>A1 — Detect Animals</b><br>"
            "Runs YOLO-based detection on every extracted frame. The thermal model is "
            "downloaded automatically on first use.<br><br>"
            "<b>→ Geo-Reference Detections</b><br>"
            "Projects pixel-space bounding boxes to UTM coordinates by ray-casting "
            "against the DEM mesh. A detection that cannot be placed is recorded "
            "with the reason rather than dropped.<br><br>"
            "<b>→ Calculate Perpendicular</b><br>"
            "Measures the perpendicular distance from each geo-referenced detection "
            "to the flight route line — useful for transect-based surveys.<br><br>"
            "<b>A2 — Track Animals Or Import</b><br>"
            "Links detections across frames into continuous tracks using the selected "
            "tracking backend. With a TRex tracklet folder configured this imports "
            "those instead, and runs no tracker.<br><br>"
            "<b>A3 / A4 — Object Segmentation</b><br>"
            "Segments detected objects using Roboflow SAM3 and projects masks to "
            "world coordinates.<br><br>"

            "The BAMBI plugin is not intended to process multiple flights within "
            "the same QGIS project (and the same output folder), since the result "
            "files in the output folder will be replaced after additional runs. "
            "If you want to combine multiple results, it is recommended to run "
            "processing independently with (1) different output folders (don't "
            "foregt to add all layers to QGIS before changing the output folder, "
            "otherwise you will have to change it back again!) or (2) group all "
            "layers of interest, export them as 'Layer Definition File' and import "
            "it again to QGIS."
        )
        msg.exec()

    def _on_thermal_calib_preset_changed(self, index: int):
        """Show/hide custom file row depending on combo selection."""
        self.thermal_calib_file_row.setVisible(index == 0)

    def _on_rgb_calib_preset_changed(self, index: int):
        """Show/hide custom file row depending on combo selection."""
        self.rgb_calib_file_row.setVisible(index == 0)

    def _on_thermal_photo_calib_preset_changed(self, index: int):
        """Show/hide custom file row for thermal photo calibration."""
        self.thermal_photo_calib_file_row.setVisible(index == 0)

    def _on_rgb_photo_calib_preset_changed(self, index: int):
        """Show/hide custom file row for RGB photo calibration."""
        self.rgb_photo_calib_file_row.setVisible(index == 0)

    def browse_thermal_calibration(self):
        """Browse for thermal calibration JSON file."""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Thermal Calibration JSON", "", "JSON Files (*.json)")
        if file:
            self.thermal_calibration_path_edit.setText(file)

    def browse_rgb_calibration(self):
        """Browse for RGB calibration JSON file."""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select RGB Calibration JSON", "", "JSON Files (*.json)")
        if file:
            self.rgb_calibration_path_edit.setText(file)

    def browse_thermal_photo_dir(self):
        """Browse for thermal photo directory."""
        folder = QFileDialog.getExistingDirectory(self, "Select Thermal Photo Folder")
        if folder:
            self.thermal_photo_dir_edit.setText(folder)

    def browse_rgb_photo_dir(self):
        """Browse for RGB photo directory."""
        folder = QFileDialog.getExistingDirectory(self, "Select RGB Photo Folder")
        if folder:
            self.rgb_photo_dir_edit.setText(folder)

    def browse_thermal_photo_calibration(self):
        """Browse for thermal photo calibration JSON file."""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Thermal Photo Calibration JSON", "", "JSON Files (*.json)")
        if file:
            self.thermal_photo_calibration_path_edit.setText(file)

    def browse_rgb_photo_calibration(self):
        """Browse for RGB photo calibration JSON file."""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select RGB Photo Calibration JSON", "", "JSON Files (*.json)")
        if file:
            self.rgb_photo_calibration_path_edit.setText(file)

    def browse_airdata(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select AirData CSV", "", "CSV Files (*.csv)")
        if file:
            self.airdata_path_edit.setText(file)
            # Try to auto-detect CRS from GPS coordinates (only if no DEM metadata)
            self._try_auto_detect_crs_silent()

    def _generate_flat_surface_mesh(self):
        """Generate flat_surface_dem.glb + .json from AirData/SRT GPS and set as DEM input."""
        config = self.get_config()
        target_folder = config.get("target_folder", "")
        if not target_folder:
            QMessageBox.warning(self, "Missing Target Folder", "Please set the target folder first.")
            return

        flat_msl = self.flat_surface_spin.value()
        lats, lons = [], []

        # Primary: AirData CSV (video rows only)
        airdata_path = config.get("airdata_path", "")
        if airdata_path and os.path.exists(airdata_path):
            try:
                import csv as _csv
                with open(airdata_path, 'r', encoding='utf-8-sig') as f:
                    reader = _csv.DictReader(f)
                    stripped = {k.strip(): k for k in (reader.fieldnames or [])}
                    lat_key = next((stripped[k] for k in stripped if 'latitude' in k.lower()), None)
                    lon_key = next((stripped[k] for k in stripped if 'longitude' in k.lower()), None)
                    vid_key = next((stripped[k] for k in stripped if k.strip().lower() == 'isvideo'), None)
                    for row in reader:
                        if vid_key and row.get(vid_key, '').strip() != '1':
                            continue
                        try:
                            la, lo = float(row[lat_key]), float(row[lon_key])
                            if -90 <= la <= 90 and -180 <= lo <= 180 and la != 0 and lo != 0:
                                lats.append(la)
                                lons.append(lo)
                        except (ValueError, TypeError, KeyError):
                            continue
            except Exception as e:
                self.log(f"Warning: could not read AirData GPS: {e}")

        # Fallback: SRT files (thermal first, then RGB)
        if not lats:
            srt_paths = config.get("thermal_srt_paths", []) or config.get("rgb_srt_paths", [])
            if srt_paths:
                try:
                    from bambi.srt.srt_parser import SrtParser
                    parser = SrtParser()
                    for p in srt_paths:
                        if os.path.exists(p):
                            for frame in parser.parse(p):
                                if frame.latitude and frame.longitude:
                                    lats.append(frame.latitude)
                                    lons.append(frame.longitude)
                except Exception as e:
                    self.log(f"Warning: could not read SRT GPS: {e}")

        if not lats:
            QMessageBox.warning(self, "Missing GPS Data",
                                "Could not determine the coordinate origin.\n\n"
                                "Please select an AirData CSV or SRT files in the input section.")
            return

        import math as _math
        lat = (min(lats) + max(lats)) / 2
        lon = (min(lons) + max(lons)) / 2
        delta_y = (max(lats) - min(lats)) * 111320.0
        delta_x = (max(lons) - min(lons)) * 111320.0 * _math.cos(_math.radians(lat))
        # half-diagonal of the GPS bounding box + margin for camera footprint
        # (default 50 m, or user-defined when the Auto checkbox is unchecked)
        if self.flat_surface_padding_auto_check.isChecked():
            padding_m = 50.0
        else:
            padding_m = self.flat_surface_padding_spin.value()
        half_diag = _math.sqrt((delta_x / 2) ** 2 + (delta_y / 2) ** 2)
        extent_m = max(half_diag + padding_m, padding_m, 1.0)

        output_glb = os.path.join(target_folder, "flat_surface_dem.glb")
        output_json = os.path.join(target_folder, "flat_surface_dem.json")

        target_epsg = config.get("target_epsg", 0)

        try:
            self.log(f"Generating flat surface mesh: {flat_msl:.1f} m MSL, "
                     f"extent ±{extent_m:.0f} m (padding {padding_m:.0f} m), "
                     f"origin ({lat:.5f}, {lon:.5f}), "
                     f"CRS EPSG:{target_epsg or 'auto'} …")
            from .bambi_processing import BambiProcessor
            BambiProcessor.generate_flat_surface_mesh(
                lat, lon, flat_msl, extent_m, output_glb, output_json, epsg=target_epsg
            )
            self.dem_path_edit.setText(output_glb)
            self.dem_metadata_path_edit.setText(output_json)
            self.geo_input_tabs.setCurrentIndex(0)
            self.log(f"Flat surface mesh ready: {output_glb}")
        except Exception as e:
            self.log(f"Error generating flat surface mesh: {e}")
            QMessageBox.critical(self, "Error", f"Failed to generate flat surface mesh:\n{e}")

    def browse_dem(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select DEM GLTF File", "", "GLTF Files (*.gltf *.glb)")
        if file:
            self.dem_path_edit.setText(file)
            # Auto-initialize DEM metadata path by changing suffix to .json
            json_path = file.replace(".gltf", ".json").replace(".glb", ".json")
            metadata_found = False
            if os.path.exists(json_path):
                self.dem_metadata_path_edit.setText(json_path)
                self.log(f"Auto-detected DEM metadata: {json_path}")
                metadata_found = True
            else:
                # Try common naming patterns
                for suffix in ["_mesh.json", "_dem.json", "_metadata.json"]:
                    base = file.rsplit(".", 1)[0]
                    alt_path = base + suffix
                    if os.path.exists(alt_path):
                        self.dem_metadata_path_edit.setText(alt_path)
                        self.log(f"Auto-detected DEM metadata: {alt_path}")
                        metadata_found = True
                        break

            # Try to auto-detect CRS from the metadata
            if metadata_found:
                self._try_auto_detect_crs_silent()

    def download_austria_dem(self):
        """Download DEM from Austrian BEV service based on AirData CSV GPS coordinates."""
        # Check if AirData CSV is selected
        airdata_path = self.airdata_path_edit.text()
        if not airdata_path or not os.path.exists(airdata_path):
            QMessageBox.warning(
                self, "Missing Input",
                "Please select an AirData CSV file first.\n"
                "The GPS coordinates from this file will be used to determine the DEM area."
            )
            return

        # Determine output folder (parent of AirData CSV)
        output_folder = os.path.dirname(airdata_path)
        if not output_folder:
            QMessageBox.warning(
                self, "Invalid Path",
                "Could not determine output folder from AirData CSV path."
            )
            return

        # Get padding value
        padding = self.dem_padding_spin.value()

        # Get output CRS from the target CRS text input
        output_crs = self.target_crs_edit.text().strip().upper()
        if not output_crs.startswith("EPSG:"):
            output_crs = f"EPSG:{output_crs}"
        # Validate it's a UTM CRS
        if not self._is_valid_utm_crs(output_crs):
            QMessageBox.warning(
                self, "Invalid CRS",
                f"The CRS '{output_crs}' is not a valid UTM CRS.\n"
                "Please enter a UTM CRS (EPSG:32601-32660 for N hemisphere, "
                "EPSG:32701-32760 for S hemisphere)."
            )
            return

        # Confirm download
        reply = QMessageBox.question(
            self, "Download DEM",
            "Download DEM from Austrian BEV service?\n\n"
            f"AirData CSV: {os.path.basename(airdata_path)}\n"
            f"Padding: {padding} m\n"
            f"Output CRS: {output_crs}\n"
            f"Output folder: {output_folder}\n\n"
            "This will download the Austria-wide 1m ALS-DTM dataset.\n"
            "Large areas may take several minutes.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Disable button during download
        self.dem_download_btn.setEnabled(False)
        self.dem_download_btn.setText("Downloading...")

        # Create and start worker thread
        self.dem_download_thread = QThread()
        self.dem_download_worker = DEMDownloadWorker(
            csv_path=airdata_path,
            output_folder=output_folder,
            padding=float(padding),
            output_crs=output_crs,
            simplify_factor=2
        )

        self.dem_download_worker.moveToThread(self.dem_download_thread)
        self.dem_download_thread.started.connect(self.dem_download_worker.run)
        self.dem_download_worker.finished.connect(self._on_dem_download_finished)
        self.dem_download_worker.progress.connect(self._on_dem_download_progress)
        self.dem_download_worker.log.connect(self.log)
        self.dem_download_worker.finished.connect(self.dem_download_thread.quit)
        self.dem_download_worker.finished.connect(self.dem_download_worker.deleteLater)
        self.dem_download_thread.finished.connect(self.dem_download_thread.deleteLater)

        self.dem_download_thread.start()
        self.log("Starting Austria DEM download...")

    def _on_dem_download_progress(self, percent: int):
        """Handle DEM download progress updates."""
        self.dem_download_btn.setText(f"Downloading... {percent}%")

    def _on_dem_download_finished(self, success: bool, message: str):
        """Handle DEM download completion."""
        self.dem_download_btn.setEnabled(True)
        self.dem_download_btn.setText("Download DEM (Austria)")

        if success:
            # Set the DEM path to the downloaded file
            mesh_path = message  # message contains the path on success
            self.dem_path_edit.setText(mesh_path)

            # Auto-detect metadata
            json_path = mesh_path.replace(".glb", ".json").replace(".gltf", ".json")
            if os.path.exists(json_path):
                self.dem_metadata_path_edit.setText(json_path)
                self.log(f"Auto-detected DEM metadata: {json_path}")
                # Also try to auto-detect CRS from the new metadata
                self._try_auto_detect_crs_silent()

            QMessageBox.information(
                self, "DEM Download Complete",
                "DEM downloaded successfully!\n\n"
                f"Mesh: {mesh_path}\n"
                f"Metadata: {json_path}"
            )
        else:
            QMessageBox.warning(
                self, "DEM Download Failed",
                f"Failed to download DEM:\n{message}"
            )

    # =========================================================================
    # GeoTIFF to Mesh Conversion Methods
    # =========================================================================

    def browse_geotiff_input(self):
        """Browse for a GeoTIFF file to convert."""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select GeoTIFF DEM File", "",
            "GeoTIFF Files (*.tif *.tiff *.TIF *.TIFF);;All Files (*.*)"
        )
        if file:
            self.geotiff_input_path_edit.setText(file)
            self.log(f"Selected GeoTIFF: {file}")

    def convert_geotiff_to_mesh(self):
        """Convert the selected GeoTIFF to GLTF mesh format."""
        # Check if GeoTIFF is selected
        geotiff_path = self.geotiff_input_path_edit.text()
        if not geotiff_path or not os.path.exists(geotiff_path):
            QMessageBox.warning(
                self, "Missing Input",
                "Please select a GeoTIFF file to convert."
            )
            return

        # Determine output folder
        output_folder = self.target_folder_edit.text().strip()
        if not output_folder:
            # Use same folder as GeoTIFF if no target folder specified
            output_folder = os.path.dirname(geotiff_path)

        if not output_folder:
            QMessageBox.warning(
                self, "Invalid Path",
                "Could not determine output folder. Please set a Target Folder."
            )
            return

        # Get simplification factor
        simplify_factor = self.geotiff_simplify_spin.value()

        # Determine output CRS
        output_crs = None
        output_crs = self.target_crs_edit.text().strip().upper()
        if not output_crs.startswith("EPSG:"):
            output_crs = f"EPSG:{output_crs}"
        # Validate it's a UTM CRS
        if not self._is_valid_utm_crs(output_crs):
            QMessageBox.warning(
                self, "Invalid CRS",
                f"The CRS '{output_crs}' is not a valid UTM CRS.\n"
                "Please enter a UTM CRS (EPSG:32601-32660 for N hemisphere, "
                "EPSG:32701-32760 for S hemisphere)."
            )
            return

        # Determine source CRS override (optional)
        source_crs_override = self.geotiff_source_crs_edit.text().strip().upper()
        if source_crs_override:
            if not source_crs_override.startswith("EPSG:"):
                source_crs_override = f"EPSG:{source_crs_override}"
        else:
            source_crs_override = None

        # Read file CRS for display in confirmation dialog.
        # Avoid to_epsg() — it uses the PROJ database which may be QGIS's outdated version.
        file_crs_label = "Unknown"
        try:
            import rasterio
            import re
            with rasterio.open(geotiff_path) as src:
                if src.crs:
                    # Try WKT regex first (no DB lookup needed)
                    matches = re.findall(r'ID\["EPSG",(\d+)\]', str(src.crs))
                    if matches:
                        file_crs_label = f"EPSG:{matches[-1]}"
                    else:
                        try:
                            epsg = src.crs.to_epsg()
                            if epsg:
                                file_crs_label = f"EPSG:{epsg}"
                        except Exception:  # nosec B110
                            pass
        except Exception:  # nosec B110
            pass
        if file_crs_label == "Unknown":
            # rasterio reports crs=None for e.g. compound CRS files; GDAL's OSR
            # can often still identify the horizontal EPSG code.
            from .austria_dem_downloader import detect_geotiff_epsg
            detected = detect_geotiff_epsg(geotiff_path)
            if detected:
                file_crs_label = detected
        input_crs_label = source_crs_override if source_crs_override else file_crs_label

        # Confirm conversion
        crs_info = f"\nInput CRS: {input_crs_label}"
        crs_info += f"\nOutput CRS: {output_crs}" if output_crs else "\n(Using original CRS)"
        reply = QMessageBox.question(
            self, "Convert GeoTIFF",
            "Convert GeoTIFF to mesh?\n\n"
            f"Input: {os.path.basename(geotiff_path)}\n"
            f"Output folder: {output_folder}\n"
            f"Simplification: {simplify_factor}x{crs_info}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Disable button during conversion
        self.geotiff_convert_btn.setEnabled(False)
        self.geotiff_convert_btn.setText("Converting...")

        # Create and start worker thread
        self.geotiff_convert_thread = QThread()
        self.geotiff_convert_worker = GeoTIFFConversionWorker(
            geotiff_path=geotiff_path,
            output_folder=output_folder,
            output_crs=output_crs,
            simplify_factor=simplify_factor,
            source_crs_override=source_crs_override
        )

        self.geotiff_convert_worker.moveToThread(self.geotiff_convert_thread)
        self.geotiff_convert_thread.started.connect(self.geotiff_convert_worker.run)
        self.geotiff_convert_worker.finished.connect(self._on_geotiff_convert_finished)
        self.geotiff_convert_worker.progress.connect(self._on_geotiff_convert_progress)
        self.geotiff_convert_worker.log.connect(self.log)
        self.geotiff_convert_worker.finished.connect(self.geotiff_convert_thread.quit)
        self.geotiff_convert_worker.finished.connect(self.geotiff_convert_worker.deleteLater)
        self.geotiff_convert_thread.finished.connect(self.geotiff_convert_thread.deleteLater)

        self.geotiff_convert_thread.start()
        self.log("Starting GeoTIFF to mesh conversion...")

    def _on_geotiff_convert_progress(self, percent: int):
        """Handle GeoTIFF conversion progress updates."""
        self.geotiff_convert_btn.setText(f"Converting... {percent}%")

    def _on_geotiff_convert_finished(self, success: bool, message: str):
        """Handle GeoTIFF conversion completion."""
        self.geotiff_convert_btn.setEnabled(True)
        self.geotiff_convert_btn.setText("Convert GeoTIFF to Mesh")

        if success:
            # Set the DEM path to the converted file
            mesh_path = message  # message contains the path on success
            self.dem_path_edit.setText(mesh_path)

            # Auto-detect metadata
            json_path = mesh_path.replace(".glb", ".json").replace(".gltf", ".json")
            if os.path.exists(json_path):
                self.dem_metadata_path_edit.setText(json_path)
                self.log(f"Auto-detected DEM metadata: {json_path}")
                # Also try to auto-detect CRS from the new metadata
                self._try_auto_detect_crs_silent()

            QMessageBox.information(
                self, "Conversion Complete",
                "GeoTIFF converted successfully!\n\n"
                f"Mesh: {mesh_path}\n"
                f"Metadata: {json_path}"
            )
        else:
            QMessageBox.warning(
                self, "Conversion Failed",
                f"Failed to convert GeoTIFF:\n{message}"
            )

    # =========================================================================
    # CRS Validation and Auto-Detection Methods
    # =========================================================================

    def _is_valid_utm_crs(self, crs_text: str) -> bool:
        """True when *crs_text* is a UTM EPSG code (core.crs_utils)."""
        from .core.crs_utils import is_valid_utm_crs
        return is_valid_utm_crs(crs_text)

    def _parse_epsg_from_text(self, crs_text: str) -> Optional[int]:
        """Parse an EPSG code from text input (core.crs_utils)."""
        from .core.crs_utils import parse_epsg_from_text
        return parse_epsg_from_text(crs_text)

    def _validate_crs_input(self):
        """Validate the CRS text input and show warning if invalid."""
        crs_text = self.target_crs_edit.text().strip()

        if not crs_text:
            return  # Empty is ok, will use default

        if not self._is_valid_utm_crs(crs_text):
            epsg = self._parse_epsg_from_text(crs_text)
            if epsg is None:
                msg = f"'{crs_text}' is not a valid EPSG code format."
            else:
                msg = f"EPSG:{epsg} is not a UTM CRS."

            QMessageBox.warning(
                self, "Invalid CRS",
                f"{msg}\n\n"
                "The algorithms in this plugin require a UTM CRS.\n"
                "Valid ranges:\n"
                "  • Northern Hemisphere: EPSG:32601 - EPSG:32660\n"
                "  • Southern Hemisphere: EPSG:32701 - EPSG:32760\n\n"
                "Click 'Auto' to detect the correct UTM zone automatically."
            )
        else:
            # Normalize format
            epsg = self._parse_epsg_from_text(crs_text)
            self.target_crs_edit.setText(f"EPSG:{epsg}")

    def _auto_detect_crs(self):
        """
        Auto-detect the target CRS from available sources.

        Priority:
        1. DEM metadata JSON file (if available and contains 'crs' field)
        2. First GPS position from AirData CSV (calculate UTM zone)
        """
        detected_crs = None
        source = None

        # Try 1: DEM metadata JSON
        dem_metadata_path = self.dem_metadata_path_edit.text()
        if dem_metadata_path and os.path.exists(dem_metadata_path):
            try:
                with open(dem_metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                if 'crs' in metadata:
                    crs_value = metadata['crs']
                    if self._is_valid_utm_crs(crs_value):
                        detected_crs = crs_value.upper()
                        source = f"DEM metadata ({os.path.basename(dem_metadata_path)})"
                    else:
                        self.log(f"DEM metadata has CRS '{crs_value}' but it's not a UTM CRS")
            except Exception as e:
                self.log(f"Could not read DEM metadata: {e}")

        # Try 2: AirData CSV GPS coordinates
        if detected_crs is None:
            airdata_path = self.airdata_path_edit.text()
            if airdata_path and os.path.exists(airdata_path):
                detected_crs = self._detect_utm_from_airdata(airdata_path)
                if detected_crs:
                    source = f"AirData CSV ({os.path.basename(airdata_path)})"

        # Apply result
        if detected_crs:
            self.target_crs_edit.setText(detected_crs)
            self.log(f"Auto-detected CRS: {detected_crs} from {source}")
            QMessageBox.information(
                self, "CRS Auto-Detected",
                f"Detected CRS: {detected_crs}\n"
                f"Source: {source}"
            )
        else:
            # No sources available
            sources_checked = []
            if dem_metadata_path:
                sources_checked.append("DEM metadata")
            if self.airdata_path_edit.text():
                sources_checked.append("AirData CSV")

            if not sources_checked:
                QMessageBox.warning(
                    self, "Cannot Auto-Detect",
                    "No data sources available for CRS auto-detection.\n\n"
                    "Please provide either:\n"
                    "  • A DEM metadata JSON file (with 'crs' field)\n"
                    "  • An AirData CSV file with GPS coordinates"
                )
            else:
                QMessageBox.warning(
                    self, "Auto-Detection Failed",
                    f"Could not detect CRS from: {', '.join(sources_checked)}\n\n"
                    "Please enter the target CRS manually."
                )

    def _try_auto_detect_crs_silent(self):
        """
        Silently try to auto-detect CRS from available sources.

        This is called automatically when files are selected, and only
        updates the CRS field without showing message boxes. It logs
        the result to the output panel.
        """
        detected_crs = None
        source = None

        # Try 1: DEM metadata JSON
        dem_metadata_path = self.dem_metadata_path_edit.text()
        if dem_metadata_path and os.path.exists(dem_metadata_path):
            try:
                with open(dem_metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                if 'crs' in metadata:
                    crs_value = metadata['crs']
                    if self._is_valid_utm_crs(crs_value):
                        detected_crs = crs_value.upper()
                        source = "DEM metadata"
            except Exception:  # nosec B110
                pass

        # Try 2: AirData CSV GPS coordinates
        if detected_crs is None:
            airdata_path = self.airdata_path_edit.text()
            if airdata_path and os.path.exists(airdata_path):
                detected_crs = self._detect_utm_from_airdata(airdata_path)
                if detected_crs:
                    source = "AirData GPS"

        # Apply result silently (just log, no message boxes)
        if detected_crs:
            self.target_crs_edit.setText(detected_crs)
            self.log(f"Auto-detected CRS: {detected_crs} (from {source})")

    def _detect_utm_from_airdata(self, csv_path: str) -> Optional[str]:
        """Detect the UTM zone from the AirData GPS track (core.crs_utils)."""
        from .core.crs_utils import detect_utm_from_airdata
        return detect_utm_from_airdata(csv_path, log_fn=self.log)

    def cancel_dem_download(self):
        """Cancel ongoing DEM download."""
        if hasattr(self, 'dem_download_worker') and self.dem_download_worker:
            self.dem_download_worker.cancel()
            self.log("Cancelling DEM download...")

    def browse_correction(self):
        """Browse for correction.json file and load its values."""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Correction JSON", "", "JSON Files (*.json)")
        if file:
            self.correction_path_edit.setText(file)
            self.load_correction_values(file)

    def load_correction_values(self, correction_path: str):
        """Load correction values from JSON file and populate the spin boxes.

        :param correction_path: Path to the correction.json file (rotations stored in radians)
        """
        try:
            with open(correction_path, 'r') as f:
                correction = json.load(f)

            # Load translation values
            if 'translation' in correction:
                trans = correction['translation']
                if 'x' in trans:
                    self.trans_x_spin.setValue(float(trans['x']))
                if 'y' in trans:
                    self.trans_y_spin.setValue(float(trans['y']))
                if 'z' in trans:
                    self.trans_z_spin.setValue(float(trans['z']))

            # Load rotation values (stored in radians, convert to display unit if needed)
            if 'rotation' in correction:
                rot = correction['rotation']
                rot_x = float(rot.get('x', 0))
                rot_y = float(rot.get('y', 0))
                rot_z = float(rot.get('z', 0))

                # Convert from radians to degrees if UI is in degrees mode
                if self._is_using_degrees():
                    rot_x = math.degrees(rot_x)
                    rot_y = math.degrees(rot_y)
                    rot_z = math.degrees(rot_z)

                self.rot_x_spin.setValue(rot_x)
                self.rot_y_spin.setValue(rot_y)
                self.rot_z_spin.setValue(rot_z)

            # Load additional corrections (already in radians, stored as-is internally)
            self._additional_corrections = []
            self.additional_corrections_list.clear()
            if 'additional' in correction:
                for add_corr in correction['additional']:
                    self._add_correction_to_list(add_corr)

            self.log(f"Loaded correction values from: {os.path.basename(correction_path)}")

        except Exception as e:
            self.log(f"Warning: Could not load correction.json: {e}")

    def save_correction_values(self):
        """Save correction values to a JSON file.

        Saves the current translation and rotation values from the UI
        to a user-selected JSON file.
        """
        # Get save file path
        default_path = ""
        target_folder = self.target_folder_edit.text()
        if target_folder and os.path.exists(target_folder):
            default_path = os.path.join(target_folder, "correction.json")
        elif self.correction_path_edit.text():
            default_path = self.correction_path_edit.text()

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Correction JSON",
            default_path,
            "JSON Files (*.json)"
        )

        if not file_path:
            return  # User cancelled

        # Ensure .json extension
        if not file_path.lower().endswith('.json'):
            file_path += '.json'

        # Get rotation values and convert to radians if UI is in degrees mode
        rot_x = self.rot_x_spin.value()
        rot_y = self.rot_y_spin.value()
        rot_z = self.rot_z_spin.value()
        if self._is_using_degrees():
            rot_x = math.radians(rot_x)
            rot_y = math.radians(rot_y)
            rot_z = math.radians(rot_z)

        # Build correction data (rotations always saved in radians)
        correction_data = {
            "translation": {
                "x": self.trans_x_spin.value(),
                "y": self.trans_y_spin.value(),
                "z": self.trans_z_spin.value()
            },
            "rotation": {
                "x": rot_x,
                "y": rot_y,
                "z": rot_z
            }
        }

        # Add additional corrections if any (already in radians internally)
        if self._additional_corrections:
            correction_data["additional"] = self._additional_corrections

        try:
            with open(file_path, 'w') as f:
                json.dump(correction_data, f, indent=4)

            # Update the correction path field
            self.correction_path_edit.setText(file_path)

            self.log(f"Saved correction values to: {os.path.basename(file_path)}")
            QMessageBox.information(
                self,
                "Success",
                f"Correction values saved to:\n{file_path}\n\n(Rotations saved in radians)"
            )

        except Exception as e:
            self.log(f"Error saving correction.json: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save correction file:\n{str(e)}"
            )

    def _add_correction_to_list(self, correction_data: Dict[str, Any]):
        """Add a correction entry to the internal list and UI list.

        :param correction_data: Dictionary with correction data including
                               translation, rotation, start, and end
        """
        self._additional_corrections.append(correction_data)
        self._update_corrections_list_ui()

    def _is_using_degrees(self) -> bool:
        """Check if the UI is set to display rotations in degrees.

        :return: True if using degrees, False if using radians
        """
        return self.rotation_unit_combo.currentIndex() == 1

    def _on_rotation_unit_changed(self, index: int):
        """Handle rotation unit change - convert displayed values.

        :param index: 0 for radians, 1 for degrees
        """
        # Block signals to prevent recursion
        self.rot_x_spin.blockSignals(True)
        self.rot_y_spin.blockSignals(True)
        self.rot_z_spin.blockSignals(True)

        # Get current values
        rot_x = self.rot_x_spin.value()
        rot_y = self.rot_y_spin.value()
        rot_z = self.rot_z_spin.value()

        if index == 1:  # Changed to degrees
            # Convert from radians to degrees
            rot_x = math.degrees(rot_x)
            rot_y = math.degrees(rot_y)
            rot_z = math.degrees(rot_z)
            # Update spinbox ranges and precision
            self.rot_x_spin.setRange(-180, 180)
            self.rot_y_spin.setRange(-180, 180)
            self.rot_z_spin.setRange(-180, 180)
            self.rot_x_spin.setDecimals(3)
            self.rot_y_spin.setDecimals(3)
            self.rot_z_spin.setDecimals(3)
            self.rot_x_spin.setSingleStep(1.0)
            self.rot_y_spin.setSingleStep(1.0)
            self.rot_z_spin.setSingleStep(1.0)
            self.rot_label.setText("Rotation (pitch, roll, yaw) [deg]:")
        else:  # Changed to radians
            # Convert from degrees to radians
            rot_x = math.radians(rot_x)
            rot_y = math.radians(rot_y)
            rot_z = math.radians(rot_z)
            # Update spinbox ranges and precision
            self.rot_x_spin.setRange(-6.28319, 6.28319)
            self.rot_y_spin.setRange(-6.28319, 6.28319)
            self.rot_z_spin.setRange(-6.28319, 6.28319)
            self.rot_x_spin.setDecimals(5)
            self.rot_y_spin.setDecimals(5)
            self.rot_z_spin.setDecimals(5)
            self.rot_x_spin.setSingleStep(0.01)
            self.rot_y_spin.setSingleStep(0.01)
            self.rot_z_spin.setSingleStep(0.01)
            self.rot_label.setText("Rotation (pitch, roll, yaw) [rad]:")

        # Set converted values
        self.rot_x_spin.setValue(rot_x)
        self.rot_y_spin.setValue(rot_y)
        self.rot_z_spin.setValue(rot_z)

        # Re-enable signals
        self.rot_x_spin.blockSignals(False)
        self.rot_y_spin.blockSignals(False)
        self.rot_z_spin.blockSignals(False)

        # Update the list display
        self._update_corrections_list_ui()

    def _update_corrections_list_ui(self):
        """Update the list widget to reflect the current additional corrections."""
        self.additional_corrections_list.clear()
        use_degrees = self._is_using_degrees()
        unit_str = "deg" if use_degrees else "rad"

        for i, corr in enumerate(self._additional_corrections):
            start = corr.get('start', 0)
            end = corr.get('end', 999999)
            trans = corr.get('translation', {})
            rot = corr.get('rotation', {})

            # Get rotation values (stored in radians)
            rot_x = rot.get('x', 0)
            rot_y = rot.get('y', 0)
            rot_z = rot.get('z', 0)

            # Convert to degrees for display if needed
            if use_degrees:
                rot_x = math.degrees(rot_x)
                rot_y = math.degrees(rot_y)
                rot_z = math.degrees(rot_z)

            # Create summary text
            trans_str = f"T({trans.get('x', 0):.1f}, {trans.get('y', 0):.1f}, {trans.get('z', 0):.1f})"
            rot_str = f"R({rot_x:.2f}, {rot_y:.2f}, {rot_z:.2f}){unit_str}"
            item_text = f"Frames {start}-{end}: {trans_str} {rot_str}"

            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, i)  # Store index
            self.additional_corrections_list.addItem(item)

    def _add_additional_correction(self):
        """Show dialog to add a new frame-range specific correction."""
        dialog = CorrectionRangeDialog(self, use_degrees=self._is_using_degrees())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            correction_data = dialog.get_correction_data()
            self._add_correction_to_list(correction_data)
            self.log(f"Added correction for frames {correction_data['start']}-{correction_data['end']}")

    def _edit_additional_correction(self, item: QListWidgetItem):
        """Edit an additional correction when double-clicked.

        :param item: The list item that was double-clicked
        """
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is not None and 0 <= index < len(self._additional_corrections):
            correction_data = self._additional_corrections[index]
            dialog = CorrectionRangeDialog(self, correction_data, use_degrees=self._is_using_degrees())
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self._additional_corrections[index] = dialog.get_correction_data()
                self._update_corrections_list_ui()
                self.log(f"Updated correction at index {index}")

    def _edit_selected_correction(self):
        """Edit the currently selected additional correction."""
        selected_items = self.additional_corrections_list.selectedItems()
        if selected_items:
            self._edit_additional_correction(selected_items[0])
        else:
            QMessageBox.information(
                self, "No Selection",
                "Please select a correction to edit."
            )

    def _remove_additional_correction(self):
        """Remove the currently selected additional correction."""
        selected_items = self.additional_corrections_list.selectedItems()
        if not selected_items:
            QMessageBox.information(
                self, "No Selection",
                "Please select a correction to remove."
            )
            return

        index = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if index is not None and 0 <= index < len(self._additional_corrections):
            removed = self._additional_corrections.pop(index)
            self._update_corrections_list_ui()
            self.log(f"Removed correction for frames {removed.get('start', 0)}-{removed.get('end', 0)}")

    @staticmethod
    def get_correction_for_frame(frame_idx: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get the appropriate correction factors for a given frame index.

        Single source: :func:`core.corrections.correction_for_frame_config`.
        """
        from .core.corrections import correction_for_frame_config
        return correction_for_frame_config(frame_idx, config)

    def browse_target_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Target Folder")
        if folder:
            self.target_folder_edit.setText(folder)
            self._check_existing_outputs(folder)
            self._adopt_target_folder(folder)
            self._refresh_migrate_button()

    def browse_trex_npz_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select TRex NPZ Folder")
        if folder:
            self.trex_npz_dir_edit.setText(folder)

    def run_trex_import(self):
        """Import TRex tracklets and geo-reference them against the DEM."""
        config = self.get_config()
        camera = config.get("tracking_camera", "T")
        target_folder = config.get("target_folder", "")
        npz_dir = config.get("trex_npz_dir", "")

        if not npz_dir or not os.path.isdir(npz_dir):
            QMessageBox.warning(
                self, "Missing Input",
                "Please specify a valid NPZ folder in the Tracking config tab."
            )
            return

        poses_suffix = "t" if camera == "T" else "w"
        poses_file = os.path.join(target_folder, f"poses_{poses_suffix}.json")
        if not os.path.isfile(poses_file):
            camera_name = "Thermal" if camera == "T" else "RGB"
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_name} frame extraction has not been completed.\n"
                f"Please run Step 1 (Extract Frames) for {camera_name} first."
            )
            return

        if not self.validate_inputs(["dem_path"]):
            return

        self.start_worker("trex_import")

    def _check_existing_outputs(self, target_folder: str):
        """Check for existing output subfolders and update status labels.

        Each camera-selectable step is checked against the camera currently
        selected in its combo box (core.output_inventory does the folder
        checks), so the status always reflects the selected modality.

        :param target_folder: Path to the target folder
        """
        from .core.output_inventory import (
            FOLDER_STATUS_STEPS, PERPENDICULAR_STEPS, check_existing_outputs)

        if not target_folder or not os.path.isdir(target_folder):
            return

        # Reset all folder-based statuses before re-checking so that deleted
        # outputs are correctly reflected as "Not started" after a refresh.
        for step in FOLDER_STATUS_STEPS:
            self.update_status(step, "⚪ Not started")
        for step in PERPENDICULAR_STEPS:
            self.update_status(step, "⚪")

        def combo_suffix(camera_combo) -> str:
            return "_t" if camera_combo.currentIndex() == 0 else "_w"

        cameras = {
            "extract": combo_suffix(self.extract_camera_combo),
            "flight_route": combo_suffix(self.flight_route_camera_combo),
            "detection": combo_suffix(self.detection_camera_combo),
            "tracking": combo_suffix(self.tracking_camera_combo),
            "fov": combo_suffix(self.fov_camera_combo),
            "alfs": combo_suffix(self.alfs_camera_combo),
            "geotiff": combo_suffix(self.geotiff_camera_combo),
            "ortho": combo_suffix(self.ortho_camera_combo),
            "sam3": combo_suffix(self.sam3_camera_combo),
        }

        completed = check_existing_outputs(target_folder, cameras)
        for step in completed:
            self.update_status(step, "🟢 Completed")

        if completed:
            self.log(f"Detected {len(completed)} completed processing step(s) in target folder")

        # Also check for existing QGIS layers
        self._check_existing_qgis_layers()

    def _check_existing_qgis_layers(self):
        """Check for existing BAMBI layers in QGIS and update status labels accordingly.

        This method checks if specific BAMBI layers or layer groups already exist
        in the current QGIS project and marks the corresponding 'Add to QGIS' steps
        as completed.
        """
        # Reset all QGIS-layer statuses before re-checking.
        for step in ("add_flight_route", "add_frame_detections", "add_layers",
                     "add_fov", "add_merged_fov", "add_alfs", "add_geotiffs",
                     "add_orthomosaic", "add_sam3", "add_perpendicular",
                     "add_track_perpendicular"):
            self.update_status(step, "⚪")

        root = QgsProject.instance().layerTreeRoot()

        # Get all existing layer group names
        existing_groups = set()
        for child in root.children():
            if hasattr(child, 'name'):
                existing_groups.add(child.name())

        # Get all existing layer names
        existing_layers = set()
        for layer in QgsProject.instance().mapLayers().values():
            existing_layers.add(layer.name())

        # Define mapping: (check_type, name_prefix, status_key, camera_combo)
        # Group/layer names carry the camera label of the modality they were
        # created for (e.g. "BAMBI Wildlife Tracks (Thermal)"), so only names
        # matching the camera currently selected for the step count as added.
        layer_status_mapping = [
            ("group", "BAMBI Flight Route", "add_flight_route", self.flight_route_camera_combo),
            ("group", "BAMBI Frame Detections", "add_frame_detections", self.detection_camera_combo),
            ("group", "BAMBI Wildlife Tracks", "add_layers", self.tracking_camera_combo),
            ("group", "BAMBI FoV Polygons", "add_fov", self.fov_camera_combo),
            ("layer", "BAMBI FoV Coverage", "add_merged_fov", self.fov_camera_combo),
            ("layer", "BAMBI ALFS", "add_alfs", self.alfs_camera_combo),
            ("group", "BAMBI Frame GeoTIFFs", "add_geotiffs", self.geotiff_camera_combo),
            ("layer", "BAMBI Orthomosaic", "add_orthomosaic", self.ortho_camera_combo),
            ("group", "SAM3 Segmentation", "add_sam3", self.sam3_camera_combo),
            # Perpendicular groups are labeled with the camera of the
            # detections/tracks the distances were computed for.
            ("group", "BAMBI Perpendicular", "add_perpendicular", self.detection_camera_combo),
            ("group", "BAMBI Track Perpendicular", "add_track_perpendicular", self.tracking_camera_combo),
        ]

        added_count = 0

        for check_type, name_prefix, status_key, camera_combo in layer_status_mapping:
            camera_label = "Thermal" if camera_combo.currentIndex() == 0 else "RGB"
            names = existing_groups if check_type == "group" else existing_layers
            if any(n.startswith(name_prefix) and n.endswith(f"({camera_label})") for n in names):
                self.update_status(status_key, "🟢 Added")
                added_count += 1

        if added_count > 0:
            self.log(f"Detected {added_count} existing BAMBI layer(s) in QGIS project")

    def _on_target_folder_changed(self):
        """Handle target folder path change from manual text editing."""
        folder = self.target_folder_edit.text().strip()
        if folder and os.path.isdir(folder):
            self._check_existing_outputs(folder)
        if not getattr(self, "_switching_flight", False):
            self._adopt_target_folder(folder)
        self._refresh_migrate_button()

    def _refresh_migrate_button(self):
        """Show 'Migrate 5.x…' only where there is something to migrate."""
        from .core import migration

        folder = self.target_folder_edit.text().strip()
        show = bool(folder) and os.path.isdir(folder) and \
            migration.has_legacy_outputs(folder) and \
            not migration.is_migrated(folder)
        self.migrate_btn.setVisible(show)

    def migrate_legacy_project(self):
        """Import this folder's 5.x outputs into the 6.0 store.

        Additive and read-only with respect to the legacy files, so it is safe
        to retry: delete the generated stores and run it again.
        """
        from qgis.PyQt.QtWidgets import QApplication
        from .core import migration

        folder = self.target_folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Migrate", "Select a target folder first.")
            return

        answer = QMessageBox.question(
            self, "Migrate to the 6.0 format",
            "Import this project's existing outputs into the new store?\n\n"
            "The current files are only read — nothing is modified or deleted, "
            "so the project keeps working exactly as it does now.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok)
        if answer != QMessageBox.StandardButton.Ok:
            return

        self.log("Migrating 5.x outputs into the 6.0 store...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            report = migration.migrate_project(folder, log_fn=self.log)
        except Exception as exc:  # noqa: BLE001 — surfaced to the user below
            QApplication.restoreOverrideCursor()
            self.log(f"Migration failed: {exc}")
            QMessageBox.critical(
                self, "Migration failed",
                f"The migration could not be completed:\n\n{exc}\n\n"
                "The existing outputs are unchanged.")
            return
        finally:
            QApplication.restoreOverrideCursor()

        summary = "\n".join(
            f"  {key}: {value}" for key, value in sorted(report.counts.items()))
        text = f"Migration finished.\n\n{summary or '  (nothing to import)'}"
        if report.warnings:
            text += "\n\nWarnings:\n" + "\n".join(
                f"  • {w}" for w in report.warnings)
        QMessageBox.information(self, "Migration finished", text)
        self._refresh_migrate_button()

    #: Human-readable names for the stage keys of ``core.stages``.
    STAGE_LABELS = {
        "flight_route": "Generate Flight Route",
        "detection": "Detect Animals",
        "trex_import": "Import TRex Tracklets",
        "georeference": "Geo-Reference Detections",
        "tracking": "Track Animals",
        "calculate_fov": "Calculate Field of View",
        "alfs": "Generate ALFS",
        "export_geotiffs": "Export Frames as GeoTIFF",
        "orthomosaic": "Generate Orthomosaic",
        "sam3_segmentation": "SAM3 Segmentation",
        "sam3_georeference": "Geo-Reference Segmentation",
        "perpendicular": "Perpendicular Distances",
        "track_perpendicular": "Track Perpendicular Distances",
        "population": "Population Estimate",
        "density": "Density Map",
        "coverage": "Coverage Map",
    }

    # ------------------------------------------------------------------
    # Flights (§10.2)
    # ------------------------------------------------------------------

    def _restore_flights(self, project) -> None:
        """Read the flight list from the QGIS project.

        A project written before 6.0 has none, so the target folder it stored
        becomes the first flight — that keeps a single-flight project working
        exactly as before, with the list simply describing what was already
        true.
        """
        from .core import flights as flights_core

        raw, _ok = project.readEntry(PLUGIN_SCOPE, "Flights/List", "")
        try:
            restored = json.loads(raw) if raw else []
        except ValueError:
            restored = []
        restored = [f for f in restored
                    if isinstance(f, dict) and f.get("target_folder")]

        if not restored:
            folder, _ok = project.readEntry(
                PLUGIN_SCOPE, "Input/TargetFolder", "")
            folder = (folder or "").strip()
            restored = [{"name": flights_core.default_name(folder),
                         "target_folder": folder}] if folder else []

        index_text, _ok = project.readEntry(PLUGIN_SCOPE, "Flights/Active", "0")
        try:
            index = int(index_text)
        except (TypeError, ValueError):
            index = 0

        self._flight_list = restored
        if restored:
            self._active_flight_index = max(0, min(index, len(restored) - 1))
        else:
            self._active_flight_index = 0
        self._refresh_flight_combo()

    def _flights(self) -> List[dict]:
        return list(getattr(self, "_flight_list", []) or [])

    def _active_flight(self) -> Optional[dict]:
        flights = self._flights()
        index = getattr(self, "_active_flight_index", 0)
        if 0 <= index < len(flights):
            return flights[index]
        return None

    def _refresh_flight_combo(self):
        from .core import flights as flights_core

        blocked = self.flight_combo.blockSignals(True)
        self.flight_combo.clear()
        for flight in self._flights():
            self.flight_combo.addItem(flights_core.group_name(flight))
        index = getattr(self, "_active_flight_index", 0)
        if 0 <= index < self.flight_combo.count():
            self.flight_combo.setCurrentIndex(index)
        self.flight_combo.blockSignals(blocked)

        several = self.flight_combo.count() > 1
        self.flight_combo.setEnabled(several)
        self.flight_rename_btn.setEnabled(self.flight_combo.count() > 0)

    def _adopt_target_folder(self, folder: str):
        """Keep the flight list in step with a target folder chosen by hand.

        The folder is the flight's identity, so setting one on a project with
        no flights creates the first, and changing it moves the active flight
        rather than silently detaching it from its outputs.
        """
        from .core import flights as flights_core

        folder = (folder or "").strip()
        if not folder:
            return
        existing = self._flights()

        if not existing:
            self._flight_list = [{
                "name": flights_core.default_name(folder),
                "target_folder": folder,
            }]
            self._active_flight_index = 0
            self._refresh_flight_combo()
            return

        found = flights_core.find_by_folder(existing, folder)
        if found is not None:
            self._active_flight_index = found
            self._refresh_flight_combo()
            return

        index = getattr(self, "_active_flight_index", 0)
        try:
            self._flight_list = flights_core.set_folder(existing, index, folder)
        except flights_core.FlightError as exc:
            QMessageBox.warning(self, "Flight", str(exc))
            return
        self._refresh_flight_combo()

    def _on_flight_changed(self, index: int):
        """Switch flights: save the current configuration, load the new one."""
        if index < 0 or index >= len(self._flights()):
            return
        if getattr(self, "_switching_flight", False):
            return
        if getattr(self, "current_worker", None) is not None:
            QMessageBox.information(
                self, "Flight",
                "A step is running. Wait for it to finish before switching "
                "flights — otherwise the log and status would describe a "
                "flight you are no longer looking at.")
            self._refresh_flight_combo()
            return

        self._switching_flight = True
        try:
            self.save_config_to_project()
            self._active_flight_index = index
            self.load_config_from_project(restore_flights=False)
        finally:
            self._switching_flight = False

        flight = self._active_flight()
        if flight:
            self.log(f"Switched to flight '{flight['name']}'")
            self._check_existing_outputs(flight.get("target_folder", ""))

    def add_flight(self):
        """Add another flight to this project and switch to it."""
        from .core import flights as flights_core

        folder = QFileDialog.getExistingDirectory(
            self, "Target folder for the new flight")
        if not folder:
            return

        name, ok = QInputDialog.getText(
            self, "New flight", "Flight name:",
            text=flights_core.default_name(folder))
        if not ok:
            return

        try:
            updated = flights_core.add_flight(self._flights(), name, folder)
        except flights_core.FlightError as exc:
            QMessageBox.warning(self, "Flight", str(exc))
            return

        self.save_config_to_project()
        self._flight_list = updated
        self._active_flight_index = len(updated) - 1
        self._refresh_flight_combo()

        self._switching_flight = True
        try:
            self.target_folder_edit.setText(folder)
            self.load_config_from_project(restore_flights=False)
            self.target_folder_edit.setText(folder)
        finally:
            self._switching_flight = False

        self.log(f"Added flight '{updated[-1]['name']}' → {folder}")
        self._check_existing_outputs(folder)

    def rename_flight(self):
        """Rename the active flight, and its QGIS layer group with it."""
        from .core import flights as flights_core

        flight = self._active_flight()
        if flight is None:
            QMessageBox.information(
                self, "Flight", "There is no flight to rename yet.")
            return

        previous = flights_core.group_name(flight)
        name, ok = QInputDialog.getText(
            self, "Rename flight", "Flight name:", text=flight["name"])
        if not ok:
            return

        try:
            updated = flights_core.rename_flight(
                self._flights(), self._active_flight_index, name)
        except flights_core.FlightError as exc:
            QMessageBox.warning(self, "Flight", str(exc))
            return

        self._flight_list = updated
        self._refresh_flight_combo()

        # Rename the group rather than leaving a stale one behind.
        group = QgsProject.instance().layerTreeRoot().findGroup(previous)
        if group is not None:
            group.setName(flights_core.group_name(updated[
                self._active_flight_index]))
        self.log(f"Renamed flight to '{updated[self._active_flight_index]['name']}'")

    def _on_export_format_changed(self):
        """Reflect the chosen format's own defaults in the options."""
        from .core import exporters

        key = self.export_format_combo.currentData()
        if key is None:
            return
        # Darwin Core has no say in this, so the checkbox is disabled rather
        # than silently ignored.
        is_dwca = key == "dwca"
        self.export_false_positives_check.setEnabled(not is_dwca)
        self.export_false_positives_check.setChecked(
            (not is_dwca) and key not in exporters.TRAINING_FORMATS)

    def run_export(self):
        """Write the current project out in the selected format (§8.1)."""
        from qgis.PyQt.QtWidgets import QApplication

        from .core import exporters

        folder = self.target_folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Export", "Select a target folder first.")
            return

        key = self.export_format_combo.currentData()
        label, _function, is_folder = exporters.EXPORTERS[key]
        modality = "t" if self.export_camera_combo.currentIndex() == 0 else "w"

        epsg = None
        if key in exporters.NEEDS_CRS:
            epsg = self._parse_epsg_from_text(
                self.target_crs_edit.text().strip().upper())
            if not epsg:
                QMessageBox.warning(
                    self, "Export",
                    f"{label} publishes latitude/longitude, so the target CRS "
                    "must be set before exporting.")
                return

        if is_folder:
            output = QFileDialog.getExistingDirectory(
                self, f"Export {label} — choose a folder", folder)
        else:
            suggested = os.path.join(
                folder, exporters.DEFAULT_FILENAME.get(key, "export.json"))
            output, _filter = QFileDialog.getSaveFileName(
                self, f"Export {label}", suggested)
        if not output:
            return

        self.log(f"Exporting {label} ({'thermal' if modality == 't' else 'RGB'})…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            exporters.run_export(
                key, folder, modality, output, epsg=epsg,
                include_not_an_animal=(
                    self.export_false_positives_check.isChecked()
                    if self.export_false_positives_check.isEnabled() else None),
                log_fn=self.log)
        except exporters.ExportError as exc:
            QApplication.restoreOverrideCursor()
            self.log(f"Export failed: {exc}")
            QMessageBox.warning(self, "Export", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 — surfaced to the user below
            QApplication.restoreOverrideCursor()
            self.log(f"Export failed: {exc}")
            QMessageBox.critical(
                self, "Export", f"The export could not be completed:\n\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        QMessageBox.information(
            self, "Export", f"{label} written to:\n{output}")

    def reset_stage(self):
        """Delete one step's outputs and flag what depended on it.

        Replaces deleting a folder in Explorer: the cascade is applied, so the
        results further downstream are marked out of date instead of quietly
        surviving as stale files (§7).
        """
        from .core import stages

        folder = self.target_folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Reset Step",
                                "Select a target folder first.")
            return

        modality = self.detection_camera_combo.currentText()[:1].lower() \
            if hasattr(self, "detection_camera_combo") else "t"
        modality = "w" if modality == "r" else "t"

        present = [s for s in stages.known_stages()
                   if stages.has_output(folder, s, modality)]
        if not present:
            QMessageBox.information(
                self, "Reset Step",
                "This project has no step outputs to reset.")
            return

        labels = [f"{self.STAGE_LABELS.get(s, s)}" for s in present]
        choice, ok = QInputDialog.getItem(
            self, "Reset Step", "Step to reset:", labels, 0, False)
        if not ok:
            return
        stage = present[labels.index(choice)]

        affected = stages.dependents(stage)
        affected_present = [s for s in affected
                            if stages.has_output(folder, s, modality)]
        message = (f"Delete the outputs of '{choice}'?\n\n"
                   "The files are removed and the step runs from scratch "
                   "next time.")
        if affected_present:
            names = "\n".join(
                f"  • {self.STAGE_LABELS.get(s, s)}" for s in affected_present)
            message += ("\n\nThese depend on it and will be marked out of "
                        f"date (their files are kept):\n{names}")

        standard = QMessageBox.StandardButton
        if QMessageBox.question(
                self, "Reset Step", message,
                standard.Ok | standard.Cancel, standard.Cancel) != standard.Ok:
            return

        try:
            result = stages.reset(folder, stage, modality, log_fn=self.log)
        except stages.StageLockedError as exc:
            QMessageBox.warning(self, "Reset Step", str(exc))
            return

        self._check_existing_outputs(folder)
        stale_note = (
            f"{len(result['stale'])} dependent step(s) marked out of date."
            if result["stale"] else "Nothing depended on it.")
        QMessageBox.information(
            self, "Reset Step",
            f"Removed {len(result['removed'])} output(s).\n{stale_note}")

    def open_schema_dialog(self, initial_tab: int = 0) -> bool:
        """Open the Project Schema editor for the current target folder.

        Returns True when the user accepted, so callers that show a species or
        enum combo can repopulate it.
        """
        from .bambi_schema_dialog import BambiSchemaDialog

        folder = self.target_folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(
                self, "Project Schema",
                "Select a target folder first — the schema belongs to a "
                "project.")
            return False

        dialog = BambiSchemaDialog(folder, parent=self, initial_tab=initial_tab)
        return dialog.exec() == QDialog.DialogCode.Accepted

    def open_correction_wizard(self) -> None:
        """Open the correction calibration wizard as a modal dialog."""
        from .bambi_correction_wizard import BambiCorrectionWizard
        config = self.get_config()
        wizard = BambiCorrectionWizard(self.iface, config, parent=self)
        wizard.correctionFileSaved.connect(self.load_correction_values)
        wizard.exec()

    def _refresh_thermal_sdk_status(self):
        """Re-check DJI Thermal SDK availability and update the Thermal Visualisation group."""
        plugins_dir = self._thermal_plugins_dir
        available = (
            os.path.isdir(plugins_dir) and  # noqa: W503, W504
            any(
                e.startswith("dji_thermal_sdk_v") and  # noqa: W503, W504
                os.path.isdir(os.path.join(plugins_dir, e))
                for e in os.listdir(plugins_dir)
            )
        )
        self.thermal_vis_group.setEnabled(available)
        if available:
            self._thermal_sdk_info_label.setVisible(False)
        else:
            self._thermal_sdk_info_label.setText(
                "⚠ DJI Thermal SDK not found – install it via the Dependency Manager."
            )
            self._thermal_sdk_info_label.setVisible(True)

    def _on_thermal_vis_mode_changed(self, index):
        """Enable either the threshold widgets or the curve editor row."""
        curve_mode = index == 1
        self.thermal_vis_lo_check.setEnabled(not curve_mode)
        self.thermal_vis_hi_check.setEnabled(not curve_mode)
        self.thermal_vis_lo_spin.setEnabled(
            not curve_mode and self.thermal_vis_lo_check.isChecked())
        self.thermal_vis_hi_spin.setEnabled(
            not curve_mode and self.thermal_vis_hi_check.isChecked())
        self.thermal_vis_curve_btn.setEnabled(curve_mode)
        self.thermal_vis_curve_label.setEnabled(curve_mode)

    def _update_thermal_vis_curve_label(self):
        if self._thermal_vis_curve is None:
            self.thermal_vis_curve_label.setText("(not defined)")
        else:
            self.thermal_vis_curve_label.setText(
                self._thermal_vis_curve.describe())

    def _thermal_photo_image_paths(self):
        """List the thermal photos the curve auto-detection should scan."""
        import glob as _glob
        folder = self.thermal_photo_dir_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            return []
        _ext_suffixes = (".JPG", ".jpg", ".jpeg", ".JPEG",
                         ".tiff", ".TIFF", ".png", ".PNG")
        if self.thermal_photo_filter_check.isChecked():
            patterns = tuple(f"{p}{e}" for p in ("*_T_*", "*_T")
                             for e in _ext_suffixes)
        else:
            patterns = tuple(f"*{e}" for e in _ext_suffixes)
        return sorted(set(
            p for pattern in patterns
            for p in _glob.glob(os.path.join(folder, pattern))
        ))

    def _thermal_parse_factory(self):
        """Create a Thermal parser for the curve auto-detection scan.

        Returns ``(parse_fn, close_fn)``; the caller must invoke *close_fn*
        when done so the SDK DLLs are released.
        """
        import numpy as np
        from .bambi_thermal import Thermal
        thermal = Thermal(dtype=np.float32)
        return thermal.parse, thermal.close

    def _edit_thermal_vis_curve(self):
        from .bambi_curve_widget import CurveEditorDialog
        from .core.thermal_curve import ThermalCurve
        curve = self._thermal_vis_curve or ThermalCurve(0.0, 40.0)
        dlg = CurveEditorDialog(
            curve=curve,
            image_paths_provider=self._thermal_photo_image_paths,
            parse_factory=self._thermal_parse_factory,
            colormap=self.thermal_vis_cmap_combo.currentText(),
            parent=self,
        )
        if dlg.exec():
            self._thermal_vis_curve = dlg.curve()
            self._update_thermal_vis_curve_label()

    def set_inspector_actions(self, inspector_action, fov_inspector_action,
                              fov_georef_inspector_action=None):
        """Receive the toolbar QAction references from the main plugin class.

        Called once, immediately after the dock widget is created.  The dock
        widget uses these to keep the toolbar button checked-states in sync.
        """
        self._inspector_action = inspector_action
        self._fov_inspector_action = fov_inspector_action
        self._fov_georef_inspector_action = fov_georef_inspector_action

    def _toggle_inspector(self, checked: bool):
        """Activate or deactivate the detection/track inspector map tool."""
        canvas = self.iface.mapCanvas()
        if checked:
            self._set_fov_inspector_off()
            self._set_fov_georef_inspector_off()
            if self._click_tool is None:
                self._click_tool = BambiClickTool(self.iface, mode="detection_track")
            canvas.setMapTool(self._click_tool)
        else:
            if self._click_tool is not None:
                canvas.unsetMapTool(self._click_tool)
        if self._inspector_action is not None:
            self._inspector_action.blockSignals(True)
            self._inspector_action.setChecked(checked)
            self._inspector_action.blockSignals(False)

    def _toggle_fov_inspector(self, checked: bool):
        """Activate or deactivate the FoV inspector map tool (simple, no geo-referencing)."""
        canvas = self.iface.mapCanvas()
        if checked:
            self._set_inspector_off()
            self._set_fov_georef_inspector_off()
            if self._fov_click_tool is None:
                self._fov_click_tool = BambiClickTool(self.iface, mode="fov")
            canvas.setMapTool(self._fov_click_tool)
        else:
            if self._fov_click_tool is not None:
                canvas.unsetMapTool(self._fov_click_tool)
        if self._fov_inspector_action is not None:
            self._fov_inspector_action.blockSignals(True)
            self._fov_inspector_action.setChecked(checked)
            self._fov_inspector_action.blockSignals(False)

    def _toggle_fov_georef_inspector(self, checked: bool):
        """Activate or deactivate the FoV geo-referenced inspector map tool."""
        canvas = self.iface.mapCanvas()
        if checked:
            self._set_inspector_off()
            self._set_fov_inspector_off()
            if self._fov_georef_click_tool is None:
                self._fov_georef_click_tool = BambiClickTool(self.iface, mode="fov_georef")
            canvas.setMapTool(self._fov_georef_click_tool)
        else:
            if self._fov_georef_click_tool is not None:
                canvas.unsetMapTool(self._fov_georef_click_tool)
        if self._fov_georef_inspector_action is not None:
            self._fov_georef_inspector_action.blockSignals(True)
            self._fov_georef_inspector_action.setChecked(checked)
            self._fov_georef_inspector_action.blockSignals(False)

    def _set_inspector_off(self):
        """Silently deactivate the detection/track inspector."""
        if self._inspector_action is not None:
            self._inspector_action.blockSignals(True)
            self._inspector_action.setChecked(False)
            self._inspector_action.blockSignals(False)

    def _set_fov_inspector_off(self):
        """Silently deactivate the FoV inspector."""
        if self._fov_inspector_action is not None:
            self._fov_inspector_action.blockSignals(True)
            self._fov_inspector_action.setChecked(False)
            self._fov_inspector_action.blockSignals(False)

    def _set_fov_georef_inspector_off(self):
        """Silently deactivate the FoV geo-referenced inspector."""
        if self._fov_georef_inspector_action is not None:
            self._fov_georef_inspector_action.blockSignals(True)
            self._fov_georef_inspector_action.setChecked(False)
            self._fov_georef_inspector_action.blockSignals(False)

    def _on_map_tool_changed(self, new_tool, old_tool):
        """Keep toolbar action states in sync when the user switches map tools."""
        if self._click_tool is not None and new_tool is not self._click_tool:
            self._set_inspector_off()
        if self._fov_click_tool is not None and new_tool is not self._fov_click_tool:
            self._set_fov_inspector_off()
        if self._fov_georef_click_tool is not None and new_tool is not self._fov_georef_click_tool:
            self._set_fov_georef_inspector_off()

    def _on_step_camera_changed(self):
        """Re-evaluate step statuses when a step's camera selection changes."""
        if self.worker is not None:
            return  # don't clobber the status of a currently running step
        target_folder = self.target_folder_edit.text().strip()
        if target_folder and os.path.isdir(target_folder):
            self._check_existing_outputs(target_folder)

    def _refresh_all_statuses(self):
        """Refresh all status indicators by checking outputs and QGIS layers."""
        self.log("Refreshing status indicators...")

        # Check target folder outputs
        target_folder = self.target_folder_edit.text().strip()
        if target_folder and os.path.isdir(target_folder):
            self._check_existing_outputs(target_folder)
        else:
            # Still check QGIS layers even without target folder
            self._check_existing_qgis_layers()

    def browse_thermal_model(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Thermal YOLO Model", "", "Model Files (*.pt *.onnx)")
        if file:
            self.thermal_model_path_edit.setText(file)

    def browse_rgb_model(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select RGB YOLO Model", "", "Model Files (*.pt *.onnx)")
        if file:
            self.rgb_model_path_edit.setText(file)

    def browse_dem_metadata(self):
        """Browse for DEM metadata JSON file."""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select DEM Metadata JSON", "", "JSON Files (*.json)")
        if file:
            self.dem_metadata_path_edit.setText(file)

    def browse_fov_mask(self):
        """Browse for FoV binary mask image."""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select FoV Mask Image", "", "Image Files (*.png *.PNG)")
        if file:
            self.fov_mask_path_edit.setText(file)

    def _toggle_fov_mask(self, state):
        """Toggle the FoV mask path widget based on checkbox state."""
        self.fov_mask_widget.setEnabled(state)

    def toggle_alfs_frame_range(self, state):
        """Toggle the frame range controls based on checkbox state."""
        self.alfs_frame_range_widget.setEnabled(not state)

    def toggle_ortho_frame_range(self, state):
        """Toggle the orthomosaic frame range controls based on checkbox state."""
        self.ortho_frame_range_widget.setEnabled(not state)

    def toggle_alfs_sampling(self, state):
        """Toggle sampling mode controls."""
        self.alfs_sampling_widget.setEnabled(bool(state))

    def toggle_detect_frame_range(self, state):
        """Toggle the detection frame range controls based on checkbox state."""
        self.detect_frame_range_widget.setEnabled(not state)

    def toggle_fov_frame_range(self, state):
        """Toggle the FoV frame range controls based on checkbox state."""
        self.fov_frame_range_widget.setEnabled(not state)

    def toggle_sam3_frame_range(self, state):
        """Toggle the SAM3 frame range controls based on checkbox state."""
        self.sam3_frame_range_widget.setEnabled(not state)

    def _toggle_api_key_visibility(self, state):
        """Toggle visibility of SAM3 API key."""
        if state:
            self.sam3_api_key_edit.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self.sam3_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)

    def _toggle_frame_marker_interval(self, state):
        """Toggle the frame marker interval spinbox based on checkbox state."""
        self.frame_marker_interval_spin.setEnabled(state)
        self.frame_marker_include_zero_check.setEnabled(state)

    def _toggle_image_label_controls(self, state):
        """Toggle image label interval spinbox based on checkbox state."""
        self.image_label_interval_spin.setEnabled(state)

    def _toggle_distance_marker_controls(self, state):
        """Toggle distance marker controls based on checkbox state."""
        self.distance_marker_interval_spin.setEnabled(state)
        self.distance_marker_include_start_check.setEnabled(state)

    def _toggle_time_marker_controls(self, state):
        """Toggle time marker controls based on checkbox state."""
        self.time_marker_type_combo.setEnabled(state)
        self.time_marker_interval_spin.setEnabled(state)
        self.time_marker_include_start_check.setEnabled(state)

    def detect_frame_count(self):
        """Detect the number of available frames from poses.json."""
        config = self.get_config()
        target_folder = config.get("target_folder", "")

        if not target_folder:
            QMessageBox.warning(self, "Error", "Please set the target folder first.")
            return

        camera = config.get("alfs_camera", "T")
        poses_file = os.path.join(target_folder, f"poses_{'t' if camera == 'T' else 'w'}.json")

        if not os.path.exists(poses_file):
            QMessageBox.warning(
                self, "Error",
                "poses.json not found. Please run Step 1 (Extract Frames) first."
            )
            return

        try:
            with open(poses_file, 'r') as f:
                poses = json.load(f)

            images = poses.get("images", [])
            frame_count = len(images)

            self.frame_count_label.setText(f"Frame count: {frame_count}")

            # Update spinbox ranges for all frame range widgets
            if frame_count > 0:
                # ALFS
                self.alfs_start_frame_spin.setRange(0, frame_count - 1)
                self.alfs_end_frame_spin.setRange(0, frame_count - 1)
                self.alfs_end_frame_spin.setValue(frame_count - 1)

                # Detection
                if hasattr(self, 'detect_start_frame_spin'):
                    self.detect_start_frame_spin.setRange(0, frame_count - 1)
                    self.detect_end_frame_spin.setRange(0, frame_count - 1)
                    self.detect_end_frame_spin.setValue(frame_count - 1)

                # Field of View
                if hasattr(self, 'fov_start_frame_spin'):
                    self.fov_start_frame_spin.setRange(0, frame_count - 1)
                    self.fov_end_frame_spin.setRange(0, frame_count - 1)
                    self.fov_end_frame_spin.setValue(frame_count - 1)

                # SAM3 Segmentation
                if hasattr(self, 'sam3_start_frame_spin'):
                    self.sam3_start_frame_spin.setRange(0, frame_count - 1)
                    self.sam3_end_frame_spin.setRange(0, frame_count - 1)
                    self.sam3_end_frame_spin.setValue(frame_count - 1)

            self.log(f"Detected {frame_count} frames available for processing")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to read poses.json: {str(e)}")

    # Processing functions
    def run_extract_frames(self):
        """Run frame extraction for the camera selected in the step's combo."""
        if self.extract_camera_combo.currentIndex() == 0:
            self.run_extract_thermal_frames()
        else:
            self.run_extract_rgb_frames()

    def run_extract_thermal_frames(self):
        """Run frame extraction for thermal modality."""
        config = self.get_config()

        if config["input_mode"] == "photo":
            # Photo mode validation
            thermal_calib_ok = (
                config.get("thermal_photo_calibration_data") is not None or  # noqa: W503, W504
                config.get("thermal_photo_calibration_path")
            )
            if not config.get("thermal_photo_dir") or not thermal_calib_ok:
                QMessageBox.warning(
                    self,
                    "Missing Thermal Photo Inputs",
                    "Please provide thermal photo inputs:\n\n"
                    "• Thermal photo folder\n"
                    "• Thermal calibration (preset or custom file)"
                )
                return
        else:
            # Video mode validation
            thermal_calib_ok = (
                config.get("thermal_calibration_data") is not None or  # noqa: W503, W504
                config.get("thermal_calibration_path")
            )
            srt_ok = self.embedded_srt_check.isChecked() or bool(config.get("thermal_srt_paths"))
            if not (config.get("thermal_video_paths") and srt_ok and thermal_calib_ok):
                QMessageBox.warning(
                    self,
                    "Missing Thermal Inputs",
                    "Please provide thermal video inputs:\n\n"
                    "• Thermal video files\n" +  # noqa: W503, W504
                    ("" if self.embedded_srt_check.isChecked() else "• Thermal SRT files\n") +  # noqa: W503, W504
                    "• Thermal calibration (preset or custom file)"
                )
                return

        # Validate common inputs (airdata optional in photo mode — EXIF fallback)
        required = ["target_folder"]
        if config["input_mode"] != "photo":
            required.append("airdata_path")
        if not self.validate_inputs(required):
            return

        self.start_worker("extract_thermal_frames")

    def run_extract_rgb_frames(self):
        """Run frame extraction for RGB modality."""
        config = self.get_config()

        if config["input_mode"] == "photo":
            # Photo mode validation
            rgb_calib_ok = (
                config.get("rgb_photo_calibration_data") is not None or  # noqa: W503, W504
                config.get("rgb_photo_calibration_path")
            )
            if not config.get("rgb_photo_dir") or not rgb_calib_ok:
                QMessageBox.warning(
                    self,
                    "Missing RGB Photo Inputs",
                    "Please provide RGB photo inputs:\n\n"
                    "• RGB photo folder\n"
                    "• RGB calibration (preset or custom file)"
                )
                return
        else:
            # Video mode validation
            rgb_calib_ok = (
                config.get("rgb_calibration_data") is not None or  # noqa: W503, W504
                config.get("rgb_calibration_path")
            )
            srt_ok = self.embedded_srt_check.isChecked() or bool(config.get("rgb_srt_paths"))
            if not (config.get("rgb_video_paths") and srt_ok and rgb_calib_ok):
                QMessageBox.warning(
                    self,
                    "Missing RGB Inputs",
                    "Please provide RGB video inputs:\n\n"
                    "• RGB video files\n" +  # noqa: W503, W504
                    ("" if self.embedded_srt_check.isChecked() else "• RGB SRT files\n") +  # noqa: W503, W504
                    "• RGB calibration (preset or custom file)"
                )
                return

        # Validate common inputs (airdata optional in photo mode — EXIF fallback)
        required = ["target_folder"]
        if config["input_mode"] != "photo":
            required.append("airdata_path")
        if not self.validate_inputs(required):
            return

        self.start_worker("extract_rgb_frames")

    def _confirm_ultralytics_license(self) -> bool:
        """Show the Ultralytics license notice before running detection.

        Returns True if the user accepted (or previously chose to skip the
        notice), False if they cancelled. The skip decision is only stored
        when the user accepts, so cancelling can never silently disable
        detection in future runs.
        """
        settings = QgsSettings()
        settings_key = f"{PLUGIN_SCOPE}/skipUltralyticsLicenseNotice"
        if settings.value(settings_key, False, type=bool):
            return True

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Ultralytics License Notice")
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setText(
            "Wildlife detection uses the Ultralytics framework, which is "
            "licensed under AGPL-3.0."
        )
        msg_box.setInformativeText(
            "Commercial use requires a separate Ultralytics Enterprise "
            "License. Please make sure your usage complies with the license "
            "terms before proceeding.\n\n"
            "See https://www.ultralytics.com/license for details."
        )
        msg_box.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QMessageBox.StandardButton.Ok)
        remember_check = QCheckBox("Remember my decision")
        remember_check.setChecked(False)
        msg_box.setCheckBox(remember_check)

        msg_box.exec()
        # exec() returns a plain int under PyQt6, so resolve the clicked
        # button to a StandardButton for a binding-agnostic comparison.
        clicked = msg_box.standardButton(msg_box.clickedButton())
        if clicked != QMessageBox.StandardButton.Ok:
            return False

        if remember_check.isChecked():
            settings.setValue(settings_key, True)
        return True

    def run_detection(self):
        """Run animal detection step."""
        config = self.get_config()
        camera = config.get("detection_camera", "T")

        # Check if frames exist for selected camera
        target_folder = config["target_folder"]
        frames_folder = os.path.join(target_folder, f"frames_{'t' if camera == 'T' else 'w'}")
        poses_file = os.path.join(target_folder, f"poses_{'t' if camera == 'T' else 'w'}.json")

        if not os.path.exists(poses_file) or not os.path.exists(frames_folder):
            camera_name = "Thermal" if camera == "T" else "RGB"
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_name} frame extraction has not been completed.\n"
                f"Please run Step 1 (Extract Frames) for {camera_name} first."
            )
            return

        if not self._confirm_ultralytics_license():
            return

        self.start_worker("detection")

    def run_georeference(self):
        """Run geo-referencing step."""
        config = self.get_config()
        det_camera = config.get("detection_camera", "T")
        det_suffix = "t" if det_camera == "T" else "w"

        # Check if detections exist (camera-specific folder)
        detections_folder = os.path.join(config["target_folder"], f"detections_{det_suffix}")

        if not os.path.exists(detections_folder):
            camera_name = "Thermal" if det_camera == "T" else "RGB"
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_name} detection has not been completed.\nPlease run Step 3 (Detect Animals) first."
            )
            return

        if not self.validate_inputs(["dem_path"]):
            return

        self.start_worker("georeference")

    def run_calculate_fov(self):
        """Run Field of View calculation step."""
        config = self.get_config()
        camera = config.get("fov_camera", "T")

        # Check if poses exist for selected camera
        target_folder = config["target_folder"]
        poses_file = os.path.join(target_folder, f"poses_{'t' if camera == 'T' else 'w'}.json")

        if not os.path.exists(poses_file):
            camera_name = "Thermal" if camera == "T" else "RGB"
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_name} frame extraction has not been completed.\n"
                f"Please run Step 1 (Extract Frames) for {camera_name} first."
            )
            return

        # Check DEM path
        if not self.validate_inputs(["dem_path"]):
            return

        self.start_worker("calculate_fov")

    def run_tracking(self):
        """Run tracking step, or import TRex tracklets if a tracklet folder is set."""
        config = self.get_config()

        # If a TRex NPZ folder is configured, import the pre-computed tracklets
        # instead of running the classic tracker.
        if config.get("trex_npz_dir", "").strip():
            self.run_trex_import()
            return

        trk_camera = config.get("tracking_camera", "T")
        trk_suffix = "t" if trk_camera == "T" else "w"

        # Check if georeferenced detections exist (camera-specific folder)
        georef_folder = os.path.join(config["target_folder"], f"georeferenced_{trk_suffix}")

        if not os.path.exists(georef_folder):
            camera_name = "Thermal" if trk_camera == "T" else "RGB"
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_name} geo-referencing has not been completed.\n"
                f"Please run Geo-Reference Detections (under step A1) first."
            )
            return

        self.start_worker("tracking")

    def run_alfs(self):
        """Run alfs generation step."""
        config = self.get_config()
        camera = config.get("alfs_camera", "T")

        # Check if frames exist for selected camera
        target_folder = config["target_folder"]
        poses_file = os.path.join(target_folder, f"poses_{'t' if camera == 'T' else 'w'}.json")
        frames_folder = os.path.join(target_folder, f"frames_{'t' if camera == 'T' else 'w'}")

        if not os.path.exists(poses_file) or not os.path.exists(frames_folder):
            camera_name = "Thermal" if camera == "T" else "RGB"
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_name} frame extraction has not been completed.\n"
                f"Please run Step 1 (Extract Frames) for {camera_name} first."
            )
            return

        # Check DEM path
        if not self.validate_inputs(["dem_path"]):
            return

        self.start_worker("alfs")

    def run_export_geotiffs(self):
        """Run frame GeoTIFF export step."""
        config = self.get_config()
        camera = config.get("geotiff_camera", "T")

        # Check if frames exist for selected camera
        target_folder = config["target_folder"]
        poses_file = os.path.join(target_folder, f"poses_{'t' if camera == 'T' else 'w'}.json")
        frames_folder = os.path.join(target_folder, f"frames_{'t' if camera == 'T' else 'w'}")

        if not os.path.exists(poses_file) or not os.path.exists(frames_folder):
            camera_name = "Thermal" if camera == "T" else "RGB"
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_name} frame extraction has not been completed.\n"
                f"Please run Step 1 (Extract Frames) for {camera_name} first."
            )
            return

        # Check DEM path
        if not self.validate_inputs(["dem_path"]):
            return

        self.start_worker("export_geotiffs")

    def run_orthomosaic(self):
        """Merge exported frame GeoTIFFs into a single orthomosaic."""
        config = self.get_config()
        camera = config.get("ortho_camera", "T")
        suffix = "t" if camera == "T" else "w"
        camera_name = "Thermal" if camera == "T" else "RGB"

        geotiff_folder = os.path.join(config["target_folder"], f"geotiffs_{suffix}")
        has_geotiffs = os.path.isdir(geotiff_folder) and any(
            f.lower().endswith((".tif", ".tiff")) for f in os.listdir(geotiff_folder)
        )
        if not has_geotiffs:
            QMessageBox.warning(
                self,
                "Missing GeoTIFFs",
                f"{camera_name} frame GeoTIFFs have not been exported.\n"
                f"Please run Step 7 (Export Frames as GeoTIFF) first."
            )
            return

        self.start_worker("orthomosaic")

    def add_orthomosaic_to_qgis(self):
        """Add the generated orthomosaic to QGIS as a raster layer."""
        config = self.get_config()
        camera = config.get("ortho_camera", "T")
        suffix = "t" if camera == "T" else "w"
        camera_label = "Thermal" if camera == "T" else "RGB"
        ortho_file = os.path.join(
            config["target_folder"], f"orthomosaic_{suffix}", "orthomosaic.tif"
        )

        if not os.path.exists(ortho_file):
            QMessageBox.warning(
                self,
                "Missing Orthomosaic",
                f"{camera_label} orthomosaic has not been generated.\n"
                f"Please run 'Generate Orthomosaic' first."
            )
            return

        try:
            self.log(f"Adding {camera_label} orthomosaic to QGIS...")
            self.update_status("add_orthomosaic", "🟡 Loading...")

            layer = QgsRasterLayer(ortho_file, f"BAMBI Orthomosaic ({camera_label})")
            if not layer.isValid():
                raise RuntimeError(f"Failed to load raster: {ortho_file}")
            QgsProject.instance().addMapLayer(layer)

            self.update_status("add_orthomosaic", "🟢 Added")
            self.iface.mapCanvas().refresh()
            self.log(f"Added {camera_label} orthomosaic layer to QGIS")

        except Exception as e:
            self.log(f"Error adding orthomosaic: {str(e)}")
            self.update_status("add_orthomosaic", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add orthomosaic: {str(e)}")

    def run_sam3_segmentation(self):
        """Run SAM3 segmentation step."""
        config = self.get_config()
        camera = config.get("sam3_camera", "T")

        # Check if frames exist for selected camera
        target_folder = config["target_folder"]
        poses_file = os.path.join(target_folder, f"poses_{'t' if camera == 'T' else 'w'}.json")
        frames_folder = os.path.join(target_folder, f"frames_{'t' if camera == 'T' else 'w'}")

        if not os.path.exists(poses_file) or not os.path.exists(frames_folder):
            camera_name = "Thermal" if camera == "T" else "RGB"
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_name} frame extraction has not been completed.\n"
                f"Please run Step 1 (Extract Frames) for {camera_name} first."
            )
            return

        # Check API key
        if not config.get("sam3_api_key"):
            QMessageBox.warning(
                self,
                "Missing API Key",
                "Please enter your Roboflow API key in the SAM3 Segmentation configuration tab."
            )
            return

        # Check prompts
        if not config.get("sam3_prompts"):
            QMessageBox.warning(
                self,
                "Missing Prompts",
                "Please enter at least one text prompt in the SAM3 Segmentation configuration tab."
            )
            return

        self.start_worker("sam3_segmentation")

    def run_sam3_georeference(self):
        """Run SAM3 geo-referencing step."""
        config = self.get_config()
        sam3_suffix = "t" if config.get("sam3_camera", "T") == "T" else "w"

        # Check if pixel segmentation exists (camera-specific folder)
        segmentation_file = os.path.join(
            config["target_folder"], f"segmentation_{sam3_suffix}", "segmentation_pixel.json"
        )

        if not os.path.exists(segmentation_file):
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                "SAM3 segmentation has not been completed.\nPlease run Step 9 (Run SAM3 Segmentation) first."
            )
            return

        # Check DEM path
        if not self.validate_inputs(["dem_path"]):
            return

        self.start_worker("sam3_georeference")

    def add_sam3_to_qgis(self):
        """Add SAM3 segmentation masks as QGIS layers.

        Creates a group per frame, with one layer per prompt within each frame.
        This allows enabling/disabling individual frames.
        """
        config = self.get_config()
        sam3_camera = config.get("sam3_camera", "T")
        sam3_suffix = "t" if sam3_camera == "T" else "w"
        camera_label = "Thermal" if sam3_camera == "T" else "RGB"
        segmentation_folder = os.path.join(config["target_folder"], f"segmentation_{sam3_suffix}")
        georef_file = os.path.join(segmentation_folder, "segmentation_georef.json")

        if not os.path.exists(georef_file):
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_label} SAM3 geo-referencing has not been completed.\n"
                f"Please run Step 10 (Geo-Reference Segmentation) first."
            )
            return

        try:
            self.log(f"Adding {camera_label} SAM3 segmentation to QGIS...")
            self.update_status("add_sam3", "🟡 Loading...")

            with open(georef_file, 'r', encoding='utf-8') as f:
                georef_results = json.load(f)

            if not georef_results:
                QMessageBox.warning(self, "No Results", "No geo-referenced segmentation found.")
                self.update_status("add_sam3", "🔴 No data")
                return

            frame_indices = sorted(
                r.get('frame_idx', 0) for r in georef_results
            )
            dlg = FrameRangeDialog(
                self, frame_indices[0], frame_indices[-1], len(frame_indices), "frames"
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.update_status("add_sam3", "⚪ Cancelled")
                return

            start, end = dlg.selected_range()
            georef_results = [r for r in georef_results if start <= r.get('frame_idx', 0) <= end]

            if not georef_results:
                QMessageBox.warning(self, "No Results", "No segmentation in selected range.")
                self.update_status("add_sam3", "🔴 No data")
                return

            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{config['target_epsg']}")

            all_prompts = set()
            for frame_result in georef_results:
                for prompt_data in frame_result.get('prompts', []):
                    all_prompts.add(prompt_data.get('prompt', 'unknown'))

            prompt_colors = {}
            colors = [
                (255, 0, 0),
                (0, 150, 0),
                (0, 100, 255),
                (255, 165, 0),
                (128, 0, 128),
                (0, 200, 200),
                (255, 105, 180),
                (139, 69, 19),
            ]
            for idx, prompt in enumerate(sorted(all_prompts)):
                prompt_colors[prompt] = colors[idx % len(colors)]

            if len(georef_results) > 50:
                reply = QMessageBox.question(
                    self,
                    "Many Frames",
                    f"Selected range contains {len(georef_results)} frames with segmentation.\n"
                    "Creating individual layer groups for each may slow down QGIS.\n\n"
                    "Continue with individual frame groups?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                if reply == QMessageBox.StandardButton.No:
                    self.update_status("add_sam3", "⚪ Cancelled")
                    return

            main_group = self._create_layer_group(f"SAM3 Segmentation ({camera_label})")

            total_polygons = 0
            total_frames_added = 0

            # Process each frame
            for frame_result in georef_results:
                frame_idx = frame_result.get('frame_idx', 0)
                prompts_data = frame_result.get('prompts', [])

                if not prompts_data:
                    continue

                # Create subgroup for this frame
                frame_group = main_group.addGroup(f"Frame {frame_idx:04d}")

                # Create a layer for each prompt in this frame
                for prompt_data in prompts_data:
                    prompt = prompt_data.get('prompt', 'unknown')
                    predictions = prompt_data.get('predictions', [])

                    if not predictions:
                        continue

                    # Create layer for this prompt
                    layer = QgsVectorLayer(
                        "Polygon?crs=" + target_crs.authid(),
                        prompt,
                        "memory"
                    )
                    provider = layer.dataProvider()
                    provider.addAttributes([
                        QgsField("prompt", QVariant.String),
                        QgsField("frame", QVariant.Int),
                        QgsField("confidence", QVariant.Double),
                        QgsField("polygon_idx", QVariant.Int)
                    ])
                    layer.updateFields()

                    features = []

                    for pred_idx, pred in enumerate(predictions):
                        confidence = pred.get('confidence', 0)

                        for poly_idx, world_polygon in enumerate(pred.get('world_polygons', [])):
                            if len(world_polygon) < 3:
                                continue

                            # Create polygon points (use only x, y)
                            points = [QgsPointXY(pt[0], pt[1]) for pt in world_polygon]

                            # Close the polygon if not closed
                            if points[0] != points[-1]:
                                points.append(points[0])

                            feat = QgsFeature()
                            feat.setGeometry(QgsGeometry.fromPolygonXY([points]))
                            feat.setAttributes([
                                prompt,
                                frame_idx,
                                round(confidence, 4),
                                poly_idx
                            ])
                            features.append(feat)
                            total_polygons += 1

                    if features:
                        provider.addFeatures(features)
                        layer.updateExtents()

                        # Style the layer
                        color = prompt_colors.get(prompt, (100, 100, 100))
                        self._style_sam3_layer(layer, color)

                        # Persist the layer to GeoPackage (use unique name with frame index)
                        layer_filename = f"SAM3_Frame{frame_idx:04d}_{prompt}"
                        layer = self._persist_memory_layer(layer, layer_filename, "sam3_layers")

                        # Re-apply style after persistence (style is lost when saving)
                        self._style_sam3_layer(layer, color)

                        # Add to project and frame group
                        QgsProject.instance().addMapLayer(layer, False)
                        frame_group.addLayer(layer)

                # Collapse frame group by default
                frame_group.setExpanded(False)
                total_frames_added += 1

            # Keep main group expanded
            main_group.setExpanded(True)

            self.log(f"Added SAM3 segmentation to QGIS: {total_frames_added} frames, {total_polygons} polygons")
            self.update_status("add_sam3", "🟢 Completed")

            # Refresh canvas
            if total_polygons > 0:
                self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding SAM3 layers: {str(e)}")
            self.update_status("add_sam3", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add SAM3 segmentation: {str(e)}")

    def _style_sam3_layer(self, layer, color: tuple):
        """Apply styling to a SAM3 segmentation layer.

        :param layer: Polygon layer to style
        :param color: RGB tuple like (255, 0, 0)
        """
        from qgis.core import QgsFillSymbol, QgsSingleSymbolRenderer

        try:
            color_str = f"{color[0]},{color[1]},{color[2]}"
            symbol = QgsFillSymbol.createSimple({
                'color': f"{color_str},80",  # Semi-transparent fill
                'outline_color': f"{color_str},255",
                'outline_width': '0.8'
            })
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            layer.triggerRepaint()
        except Exception:  # nosec B110
            pass

    def run_flight_route(self):
        """Run flight route generation step."""
        config = self.get_config()
        camera = config.get("flight_route_camera", "T")

        # Poses are optional: without them only the AirData route line is
        # generated (no camera positions / frame markers / image labels)
        target_folder = config["target_folder"]
        poses_file = os.path.join(target_folder, f"poses_{'t' if camera == 'T' else 'w'}.json")
        airdata_path = config.get("airdata_path", "")

        if not os.path.exists(poses_file):
            camera_name = "Thermal" if camera == "T" else "RGB"
            if not (airdata_path and os.path.exists(airdata_path)):
                QMessageBox.warning(
                    self,
                    "Missing Prerequisites",
                    f"{camera_name} frame extraction has not been completed and "
                    f"no AirData CSV is selected.\n"
                    f"Run Step 1 (Extract Frames) for {camera_name}, or select an "
                    f"AirData file to generate the flight route line only."
                )
                return
            QMessageBox.warning(
                self,
                "Camera Positions Not Available",
                f"{camera_name} frame extraction has not been completed.\n"
                f"Only the flight route line (from AirData) will be generated — "
                f"camera positions, frame/distance markers and image labels "
                f"will not be available.\n\n"
                f"Run Step 1 (Extract Frames) for {camera_name} and re-run this "
                f"step later to add them."
            )

        self.start_worker("flight_route")

    # ------------------------------------------------------------------ #
    # Survey analytics
    # ------------------------------------------------------------------ #

    def run_density_heatmap(self):
        """Run the density-heatmap generation step."""
        config = self.get_config()
        target_folder = config.get("target_folder", "")
        if not target_folder or not os.path.isdir(target_folder):
            QMessageBox.warning(self, "Missing Target Folder",
                                "Please set a valid target folder first.")
            return

        source = config.get("density_source", "detections")
        if source == "tracks":
            suffix = "t" if config.get("tracking_camera", "T") == "T" else "w"
            required = os.path.join(target_folder, f"tracks_{suffix}")
            ok = os.path.isdir(required)
            msg = "Geo-referenced tracks (run 'Track Animals' first)"
        else:
            suffix = "t" if config.get("detection_camera", "T") == "T" else "w"
            required = os.path.join(target_folder, f"georeferenced_{suffix}", "georeferenced.txt")
            ok = os.path.exists(required)
            msg = "Geo-referenced detections (run 'Geo-Reference Detections' first)"

        if not ok:
            QMessageBox.warning(self, "Missing Prerequisites",
                                f"The following are required:\n\n• {msg}")
            return

        self.start_worker("density_heatmap")

    def add_density_heatmap_to_qgis(self):
        """Load the generated density-heatmap raster with a graduated colour ramp."""
        config = self.get_config()
        source = config.get("density_source", "detections")
        if source == "tracks":
            suffix = "t" if config.get("tracking_camera", "T") == "T" else "w"
        else:
            suffix = "t" if config.get("detection_camera", "T") == "T" else "w"
        camera_label = "Thermal" if suffix == "t" else "RGB"

        raster_file = os.path.join(config["target_folder"], f"analytics_{suffix}",
                                   f"density_{source}.tif")
        if not os.path.exists(raster_file):
            QMessageBox.warning(
                self, "Missing Data",
                "Density heatmap has not been generated.\n"
                "Please run 'Generate Density Heatmap' first."
            )
            return

        try:
            self.update_status("add_density", "🟡 Loading...")
            layer_name = f"BAMBI Density {source.capitalize()} ({camera_label})"
            layer = QgsRasterLayer(raster_file, layer_name)
            if not layer.isValid():
                raise RuntimeError(f"Failed to load raster: {raster_file}")
            self._apply_density_style(layer)
            QgsProject.instance().addMapLayer(layer)
            self.update_status("add_density", "🟢 Added")
            self.iface.mapCanvas().refresh()
            self.log(f"Added density heatmap layer: {layer_name}")
        except Exception as e:
            self.update_status("add_density", "🔴 Error")
            self.log(f"Error adding density heatmap: {e}")
            QMessageBox.critical(self, "Error", f"Failed to add density heatmap: {e}")

    def _apply_density_style(self, layer):
        """Apply a single-band pseudocolor (blue→red) ramp scaled to the data."""
        try:
            from qgis.core import (
                QgsSingleBandPseudoColorRenderer, QgsColorRampShader,
                QgsRasterShader,
            )
            stats = layer.dataProvider().bandStatistics(1)
            vmin, vmax = stats.minimumValue, stats.maximumValue
            if vmax <= vmin:
                vmax = vmin + 1.0

            ramp = QgsColorRampShader(vmin, vmax)
            ramp.setColorRampType(QgsColorRampShader.Type.Interpolated)
            steps = [
                (0.0, QColor(44, 123, 182)),
                (0.25, QColor(171, 217, 233)),
                (0.5, QColor(255, 255, 191)),
                (0.75, QColor(253, 174, 97)),
                (1.0, QColor(215, 25, 28)),
            ]
            items = [
                QgsColorRampShader.ColorRampItem(
                    vmin + frac * (vmax - vmin), color, f"{vmin + frac * (vmax - vmin):.2f}")
                for frac, color in steps
            ]
            ramp.setColorRampItemList(items)

            shader = QgsRasterShader()
            shader.setRasterShaderFunction(ramp)
            renderer = QgsSingleBandPseudoColorRenderer(
                layer.dataProvider(), 1, shader)
            layer.setRenderer(renderer)
            layer.triggerRepaint()
        except Exception as e:  # nosec B110
            self.log(f"Warning: could not style density layer: {e}")

    def run_coverage_map(self):
        """Run the coverage-map generation step."""
        config = self.get_config()
        target_folder = config.get("target_folder", "")
        if not target_folder or not os.path.isdir(target_folder):
            QMessageBox.warning(self, "Missing Target Folder",
                                "Please set a valid target folder first.")
            return

        suffix = "t" if config.get("coverage_camera", "T") == "T" else "w"
        camera_label = "Thermal" if suffix == "t" else "RGB"
        geotiff_folder = os.path.join(target_folder, f"geotiffs_{suffix}")
        if not os.path.isdir(geotiff_folder):
            QMessageBox.warning(
                self, "Missing Prerequisites",
                "The following are required:\n\n"
                f"• {camera_label} frame GeoTIFFs "
                "(run 'Export Frames as GeoTIFF' first)"
            )
            return

        self.start_worker("coverage_map")

    def add_coverage_map_to_qgis(self):
        """Load the generated coverage-map raster with a graduated colour ramp."""
        config = self.get_config()
        suffix = "t" if config.get("coverage_camera", "T") == "T" else "w"
        camera_label = "Thermal" if suffix == "t" else "RGB"

        raster_file = os.path.join(config["target_folder"], f"analytics_{suffix}",
                                   "coverage_map.tif")
        if not os.path.exists(raster_file):
            QMessageBox.warning(
                self, "Missing Data",
                "Coverage map has not been generated.\n"
                "Please run 'Generate Coverage Map' first."
            )
            return

        try:
            self.update_status("add_coverage", "🟡 Loading...")
            layer_name = f"BAMBI Coverage Map ({camera_label})"
            layer = QgsRasterLayer(raster_file, layer_name)
            if not layer.isValid():
                raise RuntimeError(f"Failed to load raster: {raster_file}")
            self._apply_density_style(layer)
            QgsProject.instance().addMapLayer(layer)
            self.update_status("add_coverage", "🟢 Added")
            self.iface.mapCanvas().refresh()
            self.log(f"Added coverage map layer: {layer_name}")
        except Exception as e:
            self.update_status("add_coverage", "🔴 Error")
            self.log(f"Error adding coverage map: {e}")
            QMessageBox.critical(self, "Error", f"Failed to add coverage map: {e}")

    def run_distance_sampling(self):
        """Run the distance-sampling estimation step (one or more projects)."""
        config = self.get_config()
        folders = config.get("ds_project_folders", [])
        if not folders:
            QMessageBox.warning(
                self, "No Projects",
                "At least one project must be available. Add one or more BAMBI "
                "target folders, or tick 'Add current project'.")
            return

        source = config.get("ds_source", "detections")
        fr_suffix = "t" if config.get("flight_route_camera", "T") == "T" else "w"
        if source == "tracks":
            trk_suffix = "t" if config.get("tracking_camera", "T") == "T" else "w"
            perp_name = f"perpendicular_tracks_{trk_suffix}.json"
            perp_msg = ("Track perpendicular distances "
                        "(run 'Calculate Track Perpendicular')")
        else:
            det_suffix = "t" if config.get("detection_camera", "T") == "T" else "w"
            perp_name = f"perpendicular_{det_suffix}.json"
            perp_msg = "Perpendicular distances (run 'Calculate Perpendicular')"

        problems: Dict[str, List[str]] = {}
        for folder in folders:
            missing = []
            if not os.path.isdir(folder):
                problems[folder] = ["The project folder does not exist."]
                continue
            fr_dir = os.path.join(folder, f"flight_route_{fr_suffix}")
            if not os.path.exists(os.path.join(fr_dir, perp_name)):
                missing.append(perp_msg)
            if not os.path.exists(os.path.join(fr_dir, "flight_route.geojson")):
                missing.append("Flight route (run 'Generate Flight Route')")
            if missing:
                problems[folder] = missing

        if problems:
            self._warn_missing_project_files(problems)
            return

        self.start_worker("distance_sampling")

    def _show_distance_sampling_results(self):
        """Read the distance-sampling JSON and show a summary dialog."""
        config = self.get_config()
        source = config.get("ds_source", "detections")
        if source == "tracks":
            suffix = "t" if config.get("tracking_camera", "T") == "T" else "w"
        else:
            suffix = "t" if config.get("detection_camera", "T") == "T" else "w"
        result_file = os.path.join(config["target_folder"], f"analytics_{suffix}",
                                   f"distance_sampling_{source}.json")
        if not os.path.exists(result_file):
            return
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                r = json.load(f)
        except Exception as e:
            self.log(f"Could not read distance-sampling results: {e}")
            return

        d_ci = r.get("density_ci95", [0, 0])
        n_ci = r.get("abundance_ci95", [0, 0])
        rows = "".join(
            f"<tr><td>{m['name']}</td><td align='right'>{m['esw_m']:.2f}</td>"
            f"<td align='right'>{m['aic']:.2f}</td></tr>"
            for m in r.get("models", [])
        )

        projects = r.get("projects", [])
        projects_section = ""
        if len(projects) > 1:
            proj_rows = "".join(
                f"<tr><td>{os.path.basename(os.path.normpath(p['target_folder']))}</td>"
                f"<td align='right'>{p.get('n', 0)}</td>"
                f"<td align='right'>{p.get('transect_length_m', 0):.0f}</td></tr>"
                for p in projects
            )
            projects_section = f"""
        <h4>Projects pooled ({len(projects)})</h4>
        <table cellpadding='4'>
          <tr><th align='left'>Project</th><th align='right'>Observations</th>
              <th align='right'>Route length (m)</th></tr>
          {proj_rows}
        </table>
        """

        html = f"""
        <h3>Distance-Sampling Estimate ({r.get('source', '')})</h3>
        <table cellpadding='4'>
          <tr><td><b>Observations (n)</b></td><td align='right'>{r.get('n', 0)}
              (of {r.get('n_before_truncation', 0)})</td></tr>
          <tr><td><b>Transect length (L)</b></td>
              <td align='right'>{r.get('transect_length_m', 0):.0f} m</td></tr>
          <tr><td><b>Truncation (w)</b></td>
              <td align='right'>{r.get('truncation_m', 0):.2f} m</td></tr>
          <tr><td><b>Best model</b></td><td align='right'>{r.get('best_model', '')}</td></tr>
          <tr><td><b>Effective strip width</b></td>
              <td align='right'>{r.get('effective_strip_width_m', 0):.2f} m</td></tr>
          <tr><td><b>Detection probability (p)</b></td>
              <td align='right'>{r.get('detection_probability', 0):.3f}</td></tr>
          <tr><td><b>Density</b></td>
              <td align='right'>{r.get('density_per_km2', 0):.3f} /km²
              (95% CI {d_ci[0]:.3f}–{d_ci[1]:.3f})</td></tr>
          <tr><td><b>Abundance in covered area</b></td>
              <td align='right'>{r.get('abundance_in_covered_area', 0):.1f}
              (95% CI {n_ci[0]:.1f}–{n_ci[1]:.1f})</td></tr>
          <tr><td><b>Covered area</b></td>
              <td align='right'>{r.get('covered_area_km2', 0):.3f} km²</td></tr>
        </table>
        <h4>Model comparison</h4>
        <table cellpadding='4'>
          <tr><th align='left'>Model</th><th align='right'>ESW (m)</th>
              <th align='right'>AIC</th></tr>
          {rows}
        </table>
        {projects_section}
        <p style='color:gray;font-size:11px'>{r.get('notes', '')}</p>
        <p style='color:gray;font-size:11px'>Saved to: {result_file}</p>
        """

        dlg = QDialog(self)
        dlg.setWindowTitle("Distance-Sampling Results")
        dlg.setMinimumWidth(480)
        layout = QVBoxLayout(dlg)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml(html)
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec()

    def run_population_estimation(self):
        """Run the transect-based population estimation (one or more projects)."""
        config = self.get_config()
        entries = config.get("pop_project_folders", [])
        if not entries:
            QMessageBox.warning(
                self, "No Projects",
                "At least one project must be available. Add one or more BAMBI "
                "projects, or tick 'Add current project'.")
            return

        if not config.get("pop_methods"):
            QMessageBox.warning(
                self, "No Method Selected",
                "Please tick at least one estimation method "
                "(Naive, Bootstrap or ZINB).")
            return

        suffix = "t" if config.get("pop_camera", "T") == "T" else "w"
        camera_label = "Thermal" if suffix == "t" else "RGB"
        fr_suffix = "t" if config.get("flight_route_camera", "T") == "T" else "w"

        current = self.target_folder_edit.text().strip()
        current_key = (os.path.normcase(os.path.abspath(current))
                       if current else None)

        problems: Dict[str, List[str]] = {}
        for entry in entries:
            folder = entry.get("target", "")
            dem = entry.get("dem", "")
            if not os.path.isdir(folder):
                problems[folder] = ["The project folder does not exist."]
                continue
            missing = []
            if not os.path.exists(os.path.join(
                    folder, f"transects_{suffix}", "transects.json")):
                missing.append(
                    f"{camera_label} transect definitions "
                    "(define them with the Transect Splitting Tool)")
            if not os.path.exists(os.path.join(
                    folder, f"fov_{suffix}", "fov_polygons.txt")):
                missing.append(
                    f"{camera_label} FoV footprints "
                    "(run 'Calculate Field of View')")
            if not os.path.exists(os.path.join(
                    folder, f"flight_route_{fr_suffix}",
                    f"perpendicular_tracks_{suffix}.json")):
                missing.append(
                    f"{camera_label} track perpendicular distances "
                    "(run 'Calculate Track Perpendicular')")
            if not os.path.exists(os.path.join(folder, f"poses_{suffix}.json")):
                missing.append(
                    f"{camera_label} camera poses (poses_{suffix}.json — "
                    "run frame extraction)")
            # The active project uses the DEM configured on the Processing tab;
            # every added project must supply its own dem.json.
            folder_key = os.path.normcase(os.path.abspath(folder))
            is_current = current_key is not None and folder_key == current_key
            if not is_current:
                if not dem:
                    missing.append("DEM metadata (dem.json) — add it for this project")
                elif not os.path.isfile(dem):
                    missing.append(f"DEM metadata not found: {dem}")
            if missing:
                problems[folder] = missing

        if problems:
            self._warn_missing_project_files(problems)
            return

        self.start_worker("population_estimation")

    def _show_population_results(self):
        """Read the population-estimate JSON and show a summary dialog."""
        config = self.get_config()
        suffix = "t" if config.get("pop_camera", "T") == "T" else "w"
        result_file = os.path.join(config["target_folder"], f"analytics_{suffix}",
                                   "population_estimate.json")
        if not os.path.exists(result_file):
            return
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                r = json.load(f)
        except Exception as e:
            self.log(f"Could not read population-estimation results: {e}")
            return

        study_ha = r.get("study_area_ha", 0) or 0
        method_labels = {"naive": "Naive", "bootstrap": "Bootstrap", "zinb": "ZINB"}
        est_rows = ""
        for key, est in r.get("estimates", {}).items():
            label = method_labels.get(key, key)
            density = est.get("density_per_100ha")
            if density is None:
                est_rows += (
                    f"<tr><td>{label}</td><td colspan='3' style='color:#c33'>"
                    f"failed — {est.get('error', 'unknown error')}</td></tr>")
                continue
            ci = est.get("ci95")
            ci_txt = f"{ci[0]:.2f} – {ci[1]:.2f}" if ci else "–"
            abundance = est.get("abundance_study_area")
            ab_txt = f"{abundance:.1f}" if abundance is not None else "–"
            note = (f"<br><span style='color:#c80'>{est['error']}</span>"
                    if est.get("error") else "")
            est_rows += (
                f"<tr><td>{label}{note}</td>"
                f"<td align='right'>{density:.2f}</td>"
                f"<td align='right'>{ci_txt}</td>"
                f"<td align='right'>{ab_txt}</td></tr>")

        transect_rows = "".join(
            f"<tr><td>{t['name']}</td>"
            f"<td align='right'>{t['start_frame']}–{t['end_frame']}</td>"
            f"<td align='right'>{t['length_m']:.0f}</td>"
            f"<td align='right'>{t['area_ha']:.2f}</td>"
            f"<td align='right'>{t['count']}</td></tr>"
            for t in r.get("transects", [])
        )

        source_labels = {
            "flight_fov": "from the flight's FoV coverage",
            "manual": "entered manually",
        }
        source = source_labels.get(r.get("study_area_source", ""), "")
        study_row = (
            f"<tr><td><b>Study area</b></td>"
            f"<td align='right'>{study_ha:.1f} ha"
            f"<span style='color:gray'> ({source})</span></td></tr>"
            if study_ha > 0 else "")
        flight_ha = r.get("flight_fov_area_ha")
        flight_row = (
            f"<tr><td><b>Flight FoV coverage</b></td>"
            f"<td align='right'>{flight_ha:.2f} ha</td></tr>"
            if flight_ha else "")
        no_area = r.get("n_transects_without_area", 0)
        no_area_row = (
            f"<tr><td><b>Transects without area</b></td>"
            f"<td align='right' style='color:#c80'>{no_area} (excluded)</td></tr>"
            if no_area else "")

        projects = r.get("projects", [])
        projects_section = ""
        if len(projects) > 1:
            proj_rows = "".join(
                f"<tr><td>{p.get('name', '')}</td>"
                f"<td align='right'>{p.get('n_transects', 0)}</td>"
                f"<td align='right'>{int(p.get('total_count', 0))}</td>"
                f"<td align='right'>{p.get('total_ha', 0):.2f}</td>"
                f"<td align='right'>{p.get('dem_origin_source', '')}</td></tr>"
                for p in projects
            )
            projects_section = f"""
        <h4>Projects pooled ({len(projects)})</h4>
        <table cellpadding='4'>
          <tr><th align='left'>Project</th><th align='right'>Transects</th>
              <th align='right'>Counted</th><th align='right'>Area (ha)</th>
              <th align='right'>DEM origin</th></tr>
          {proj_rows}
        </table>
        <p style='color:gray;font-size:11px'>DEM origin: "config" = the active
        project's configured DEM; "provided" = the dem.json supplied when the
        project was added.</p>
        """

        html = f"""
        <h3>Transect Population Estimate ({r.get('camera', '')})</h3>
        <table cellpadding='4'>
          <tr><td><b>Transects</b></td>
              <td align='right'>{r.get('n_transects', 0)}
              ({r.get('n_zero_transects', 0)} with zero counts)</td></tr>
          <tr><td><b>Animals (tracks)</b></td>
              <td align='right'>{int(r.get('total_count', 0))} counted
              (of {r.get('n_tracks', 0)})</td></tr>
          <tr><td><b>Not counted</b></td>
              <td align='right'>
              {r.get('n_tracks_outside_fov', 0)} outside every field of view,
              {r.get('n_tracks_truncated', 0)} beyond the truncation</td></tr>
          <tr><td><b>Monitored area</b></td>
              <td align='right'>{r.get('total_ha', 0):.2f} ha
              <span style='color:gray'>(sum of the transects)</span></td></tr>
          {flight_row}
          <tr><td><b>Truncation</b></td>
              <td align='right'>{'off' if not r.get('truncation_m') else f"{r['truncation_m']:.1f} m"}</td></tr>
          {study_row}
          {no_area_row}
        </table>
        <h4>Estimates</h4>
        <table cellpadding='4'>
          <tr><th align='left'>Method</th>
              <th align='right'>Density (/100 ha)</th>
              <th align='right'>95% CI</th>
              <th align='right'>{'Abundance' if study_ha > 0 else '—'}</th></tr>
          {est_rows}
        </table>
        {projects_section}
        <h4>Per transect</h4>
        <table cellpadding='4'>
          <tr><th align='left'>Name</th><th align='right'>Frames</th>
              <th align='right'>Length (m)</th><th align='right'>Area (ha)</th>
              <th align='right'>Count</th></tr>
          {transect_rows}
        </table>
        <p style='color:gray;font-size:11px'>{r.get('notes', '')}</p>
        <p style='color:gray;font-size:11px'>Saved to: {result_file}</p>
        """

        dlg = QDialog(self)
        dlg.setWindowTitle("Population-Estimation Results")
        dlg.setMinimumWidth(560)
        dlg.setMinimumHeight(520)
        layout = QVBoxLayout(dlg)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml(html)
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec()

    def add_transect_areas_to_qgis(self):
        """Add one layer group per transect, holding its route and its FoV area.

        Layer tree::

            BAMBI Transect Areas (Thermal)
              ├─ North meadow
              │    ├─ Field of View   (the merged frame footprints)
              │    └─ Flight Route    (the sub-route, start frame → end frame)
              └─ Transect 2
                   └─ …

        Each transect keeps the colour the Transect Splitting Tool gives it, so
        the map and the tool's overview agree.
        """
        config = self.get_config()
        suffix = "t" if config.get("pop_camera", "T") == "T" else "w"
        camera_label = "Thermal" if suffix == "t" else "RGB"
        analytics = os.path.join(config["target_folder"], f"analytics_{suffix}")

        areas_file = os.path.join(analytics, "transect_areas.geojson")
        routes_file = os.path.join(analytics, "transect_routes.geojson")
        if not os.path.exists(areas_file):
            QMessageBox.warning(
                self, "Missing Data",
                "The transect areas have not been generated.\n"
                "Please run 'Estimate Population (Transects)' first."
            )
            return

        try:
            self.update_status("add_transect_areas", "🟡 Loading...")
            areas = self._read_geojson_features(areas_file)
            if not areas:
                raise RuntimeError("The transect areas file holds no features.")

            # A run from before the routes existed wrote no transect_routes.geojson.
            # The routes are just the poses of each transect's frame range, so
            # rebuild them rather than making the user re-run the estimation.
            if os.path.exists(routes_file):
                routes = self._read_geojson_features(routes_file)
            else:
                routes = self._transect_routes_from_poses(config, suffix)
                if routes:
                    self.log("transect_routes.geojson not found — the sub-flight "
                             "routes were rebuilt from the poses and the transect "
                             "definitions.")

            # The tracks assigned to each transect. Absent when no perpendicular
            # distances exist — that stage is simply skipped. The footprints are
            # passed along because a track only counts inside the area that saw it.
            tracks_by_id = self._transect_tracks(config, suffix, areas)

            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{config['target_epsg']}")
            group_name = f"BAMBI Transect Areas ({camera_label})"
            # Drop a previous run's group first: it would otherwise pile up a
            # duplicate, and its layers keep the GeoPackages open, which stops
            # them being rewritten on Windows.
            self._remove_layer_group(group_name)
            main_group = self._create_layer_group(group_name)

            areas_by_id = defaultdict(list)
            for feature in areas:
                areas_by_id[feature["properties"].get("transect_id")].append(feature)
            routes_by_id = defaultdict(list)
            for feature in routes:
                routes_by_id[feature["properties"].get("transect_id")].append(feature)

            # Key off both: a transect whose frame range the FoV step never
            # covered has no area feature, and it must still appear on the map
            # (with its route alone) rather than vanish silently.
            n_added = 0
            all_ids = sorted(set(areas_by_id) | set(routes_by_id),
                             key=lambda v: (v is None, v))
            for transect_id in all_ids:
                group_features = areas_by_id.get(transect_id, [])
                route_features = routes_by_id.get(transect_id, [])
                source = (group_features or route_features)[0]["properties"]
                name = source.get("name") or f"Transect {transect_id}"
                color = self._transect_color_str(transect_id)
                transect_group = main_group.addGroup(name)

                # Added top-down: the layer tree draws its first child on top,
                # so the points and the line stay visible over the translucent
                # field-of-view fill.
                tracks_layer = self._build_transect_tracks_layer(
                    tracks_by_id.get(transect_id, []), transect_id, name,
                    target_crs, color, suffix)
                if tracks_layer is not None:
                    transect_group.addLayer(tracks_layer)

                route_layer = self._build_transect_route_layer(
                    route_features, transect_id, name, target_crs, color, suffix)
                if route_layer is not None:
                    transect_group.addLayer(route_layer)

                fov_layer = self._build_transect_fov_layer(
                    group_features, transect_id, name, target_crs, color, suffix)
                if fov_layer is not None:
                    transect_group.addLayer(fov_layer)
                else:
                    self.log(f"Note: '{name}' has no field-of-view footprint — "
                             "its group holds no area layer.")

                transect_group.setExpanded(False)
                n_added += 1

            main_group.setExpanded(True)
            self.update_status("add_transect_areas", "🟢 Added")
            self.iface.mapCanvas().refresh()
            self.log(f"Added {n_added} transect group(s) under "
                     f"'BAMBI Transect Areas ({camera_label})'")
            if not routes:
                self.log("Warning: no sub-flight routes could be built — check "
                         f"that poses_{suffix}.json and transects_{suffix}/"
                         "transects.json exist. Only the field-of-view areas "
                         "were added.")
        except Exception as e:
            self.update_status("add_transect_areas", "🔴 Error")
            self.log(f"Error adding transect areas: {e}")
            QMessageBox.critical(self, "Error", f"Failed to add transect areas: {e}")

    @staticmethod
    def _read_geojson_features(path: str) -> list:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f).get("features", [])

    def _transect_tracks(self, config: dict, suffix: str,
                         area_features: list) -> dict:
        """The tracks assigned to each transect: ``{transect_id: [track, …]}``.

        Prefers ``population_tracks.csv`` — that is the very assignment the
        density numbers were computed from, so the map cannot disagree with
        them. Without it (no estimation run yet) the assignment is recomputed
        from the track perpendicular distances with the same
        :func:`core.population.assign_tracks` the step uses, including the
        containment test against the monitored areas in *area_features*.
        Returns ``{}`` when no perpendicular distances exist at all, so the
        caller skips the track layers instead of failing.
        """
        target_folder = config.get("target_folder", "")
        by_id = defaultdict(list)

        tracks_csv = os.path.join(
            target_folder, f"analytics_{suffix}", "population_tracks.csv")
        if os.path.isfile(tracks_csv):
            try:
                import csv as _csv
                with open(tracks_csv, 'r', encoding='utf-8', newline='') as f:
                    for row in _csv.DictReader(f):
                        # Blank transect_id: unassigned or beyond the truncation
                        # distance — it was not counted, so it is not shown.
                        if not row.get("transect_id"):
                            continue
                        by_id[int(row["transect_id"])].append({
                            "track_id": int(row["track_id"]),
                            "last_frame": int(row["last_frame"] or 0),
                            "x": float(row["x"]),
                            "y": float(row["y"]),
                            "class_id": int(row["class_id"] or 0),
                            "distance_m": float(row["distance_m"] or 0.0),
                            "in_frame_range": int(row.get("in_frame_range") or 0),
                        })
                return dict(by_id)
            except (ValueError, KeyError, OSError) as e:
                self.log(f"Could not read {os.path.basename(tracks_csv)} "
                         f"({e}) — recomputing the track assignment.")
                by_id = defaultdict(list)

        return self._assign_tracks_to_transects(config, suffix, area_features)

    def _fov_predicate(self, area_features: list):
        """``(transect_id, x, y) -> bool``: does that transect's area cover it?

        Built on ``QgsGeometry`` rather than shapely: this runs inside the QGIS
        process, where QGIS' own geometry engine is always present while
        shapely is only an optional dependency.
        """
        geoms = {}
        for feature in area_features:
            rings = feature.get("geometry", {}).get("coordinates", [])
            if not rings:
                continue
            transect_id = feature["properties"].get("transect_id")
            points = [QgsPointXY(float(x), float(y)) for x, y in rings[0]]
            geom = QgsGeometry.fromPolygonXY([points])
            if not geom.isGeosValid():
                geom = geom.makeValid()
            # A transect's area can arrive as several rings (a footprint broken
            # up by gaps in the frame sampling) — all of them count.
            geoms[transect_id] = (geom if transect_id not in geoms
                                  else geoms[transect_id].combine(geom))

        def contains(transect_id, x: float, y: float) -> bool:
            geom = geoms.get(transect_id)
            if geom is None or geom.isEmpty():
                return False
            # intersects() counts a point on the boundary as inside, so a track
            # sitting right at the edge of the footprint is not lost.
            return geom.intersects(QgsGeometry.fromPointXY(QgsPointXY(x, y)))

        return contains

    def _assign_tracks_to_transects(self, config: dict, suffix: str,
                                    area_features: list) -> dict:
        """Recompute the transect assignment from the perpendicular distances."""
        from .core.pipeline_outputs import read_dem_origin_xy
        from .core.population import assign_tracks, transect_centerline
        from .core.transects import TransectStore

        target_folder = config.get("target_folder", "")
        fr_suffix = "t" if config.get("flight_route_camera", "T") == "T" else "w"
        perp_file = os.path.join(
            target_folder, f"flight_route_{fr_suffix}",
            f"perpendicular_tracks_{suffix}.json")
        poses_path = os.path.join(target_folder, f"poses_{suffix}.json")
        if not os.path.isfile(perp_file) or not os.path.isfile(poses_path):
            self.log("No track perpendicular distances found — the transect "
                     "groups are added without their tracks.")
            return {}

        try:
            with open(perp_file, 'r', encoding='utf-8') as f:
                tracks = json.load(f).get("tracks", [])
            with open(poses_path, 'r', encoding='utf-8') as f:
                images = json.load(f).get("images", [])
        except (ValueError, OSError):
            return {}
        if not tracks or not images:
            return {}

        store = TransectStore(target_folder, suffix)
        try:
            if not store.load():
                return {}
        except (ValueError, OSError):
            return {}

        x_offset, y_offset = read_dem_origin_xy(
            config.get("dem_path", ""),
            config.get("alfs_dem_metadata_path") or "")
        centerlines = {}
        for transect in store.ordered():
            transect.clamp(len(images))
            centerlines[transect.transect_id] = transect_centerline(
                images, transect.first_frame, transect.last_frame,
                x_offset, y_offset)

        ranges = {t.transect_id: (t.first_frame, t.last_frame)
                  for t in store.ordered()}
        truncation = float(config.get("pop_truncation", 0.0) or 0.0)

        by_id = defaultdict(list)
        for a in assign_tracks(tracks, centerlines, truncation,
                               contains=self._fov_predicate(area_features)):
            transect_id = a["transect_id"]
            if transect_id is None:
                continue
            lo, hi = ranges[transect_id]
            frame = a["last_frame"]
            by_id[transect_id].append({
                "track_id": a["track_id"],
                "last_frame": frame if frame is not None else 0,
                "x": a["x"],
                "y": a["y"],
                "class_id": a["class_id"] if a["class_id"] is not None else 0,
                "distance_m": a["distance_m"],
                "in_frame_range": int(
                    frame is not None and lo <= frame <= hi),
            })
        self.log("No population_tracks.csv found — the track assignment was "
                 "recomputed from the perpendicular distances.")
        return dict(by_id)

    def _build_transect_tracks_layer(self, tracks, transect_id, name, target_crs,
                                     color, suffix):
        """Point layer with the tracks assigned to one transect."""
        from qgis.core import QgsMarkerSymbol, QgsSingleSymbolRenderer

        if not tracks:
            return None

        layer = QgsVectorLayer(
            "Point?crs=" + target_crs.authid(), "Tracks", "memory")
        provider = layer.dataProvider()
        provider.addAttributes([
            QgsField("track_id", QVariant.Int),
            QgsField("transect_id", QVariant.Int),
            QgsField("transect", QVariant.String),
            QgsField("last_frame", QVariant.Int),
            QgsField("class_id", QVariant.Int),
            QgsField("distance_m", QVariant.Double),
            QgsField("in_frame_range", QVariant.Int),
        ])
        layer.updateFields()

        features = []
        for track in tracks:
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPointXY(
                QgsPointXY(float(track["x"]), float(track["y"]))))
            feat.setAttributes([
                track["track_id"], transect_id, name, track["last_frame"],
                track["class_id"],
                round(float(track["distance_m"] or 0.0), 3),
                track["in_frame_range"],
            ])
            features.append(feat)
        provider.addFeatures(features)
        layer.updateExtents()

        def style(target):
            target.setRenderer(QgsSingleSymbolRenderer(
                QgsMarkerSymbol.createSimple({
                    'name': 'circle',
                    'color': f'{color},255',
                    'outline_color': '255,255,255,255',
                    'outline_width': '0.2',
                    'size': '2.4',
                })))
            self._label_transect_tracks(target)

        style(layer)
        layer = self._persist_memory_layer(
            layer, f"Transect{transect_id}_Tracks_{suffix}", "transect_layers",
            display_name="Tracks")
        style(layer)
        QgsProject.instance().addMapLayer(layer, False)
        return layer

    def _label_transect_tracks(self, layer):
        """Label each track point with its id."""
        try:
            from qgis.core import (
                QgsPalLayerSettings, QgsTextFormat, QgsVectorLayerSimpleLabeling,
            )
            settings = QgsPalLayerSettings()
            settings.fieldName = "track_id"
            settings.placement = QgsPalLayerSettings.Placement.AroundPoint
            text_format = QgsTextFormat()
            text_format.setSize(8)
            settings.setFormat(text_format)
            layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
            layer.setLabelsEnabled(True)
            layer.triggerRepaint()
        except Exception as e:  # nosec B110
            self.log(f"Warning: could not label transect tracks layer: {e}")

    def _transect_routes_from_poses(self, config: dict, suffix: str) -> list:
        """Rebuild the transects' sub-flight routes from poses + definitions.

        Same shape as ``transect_routes.geojson`` — used when that file is
        missing because the estimation was last run before it existed. The
        route of a transect is nothing but the camera positions of its frame
        range, so it needs no output of the estimation step.
        """
        from .core.pipeline_outputs import read_dem_origin_xy
        from .core.population import transect_centerline
        from .core.transects import (
            TransectStore, cumulative_distances, flight_positions, path_length,
        )

        target_folder = config.get("target_folder", "")
        poses_path = os.path.join(target_folder, f"poses_{suffix}.json")
        if not os.path.isfile(poses_path):
            return []
        try:
            with open(poses_path, 'r', encoding='utf-8') as f:
                images = json.load(f).get("images", [])
        except (ValueError, OSError):
            return []
        if not images:
            return []

        store = TransectStore(target_folder, suffix)
        try:
            if not store.load():
                return []
        except (ValueError, OSError):
            return []

        x_offset, y_offset = read_dem_origin_xy(
            config.get("dem_path", ""),
            config.get("alfs_dem_metadata_path") or "")
        cum = cumulative_distances(flight_positions(images))

        features = []
        for transect in store.ordered():
            transect.clamp(len(images))
            line = transect_centerline(
                images, transect.first_frame, transect.last_frame,
                x_offset, y_offset)
            if len(line) < 2:
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString",
                             "coordinates": [[x, y] for x, y in line]},
                "properties": {
                    "transect_id": transect.transect_id,
                    "name": transect.display_name,
                    "start_frame": transect.start_frame,
                    "end_frame": transect.end_frame,
                    "length_m": round(path_length(
                        cum, transect.first_frame, transect.last_frame), 2),
                },
            })
        return features

    @staticmethod
    def _transect_color_str(transect_id) -> str:
        """The colour the Transect Splitting Tool paints this transect with."""
        from .core.labelling import track_color_rgb

        rgb = track_color_rgb(int(transect_id or 0))
        return f"{rgb[0]},{rgb[1]},{rgb[2]}"

    def _build_transect_fov_layer(self, features, transect_id, name, target_crs,
                                  color, suffix):
        """Polygon layer with the merged FoV footprint(s) of one transect."""
        from qgis.core import QgsFillSymbol, QgsSingleSymbolRenderer

        layer = QgsVectorLayer(
            "Polygon?crs=" + target_crs.authid(), "Field of View", "memory")
        provider = layer.dataProvider()
        provider.addAttributes([
            QgsField("transect_id", QVariant.Int),
            QgsField("name", QVariant.String),
            QgsField("area_ha", QVariant.Double),
            QgsField("count", QVariant.Int),
        ])
        layer.updateFields()

        qgs_features = []
        for feature in features:
            rings = feature.get("geometry", {}).get("coordinates", [])
            if not rings:
                continue
            points = [[QgsPointXY(float(x), float(y)) for x, y in ring]
                      for ring in rings]
            props = feature.get("properties", {})
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPolygonXY(points))
            feat.setAttributes([
                transect_id, name,
                props.get("area_ha", 0.0), props.get("count", 0),
            ])
            qgs_features.append(feat)
        if not qgs_features:
            return None

        provider.addFeatures(qgs_features)
        layer.updateExtents()

        def style(target):
            target.setRenderer(QgsSingleSymbolRenderer(
                QgsFillSymbol.createSimple({
                    'color': f'{color},60',
                    'outline_color': f'{color},255',
                    'outline_width': '0.5',
                })))
            self._label_transect_layer(target)

        style(layer)
        layer = self._persist_memory_layer(
            layer, f"Transect{transect_id}_FoV_{suffix}", "transect_layers",
            display_name="Field of View")
        style(layer)
        QgsProject.instance().addMapLayer(layer, False)
        return layer

    def _build_transect_route_layer(self, features, transect_id, name, target_crs,
                                    color, suffix):
        """Line layer with the sub-flight route of one transect."""
        from qgis.core import QgsLineSymbol, QgsSingleSymbolRenderer

        qgs_features = []
        for feature in features:
            coords = feature.get("geometry", {}).get("coordinates", [])
            if len(coords) < 2:
                continue
            props = feature.get("properties", {})
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPolylineXY(
                [QgsPointXY(float(x), float(y)) for x, y in coords]))
            feat.setAttributes([
                transect_id, name,
                props.get("start_frame", 0), props.get("end_frame", 0),
                props.get("length_m", 0.0),
            ])
            qgs_features.append(feat)
        if not qgs_features:
            return None

        layer = QgsVectorLayer(
            "LineString?crs=" + target_crs.authid(), "Flight Route", "memory")
        provider = layer.dataProvider()
        provider.addAttributes([
            QgsField("transect_id", QVariant.Int),
            QgsField("name", QVariant.String),
            QgsField("start_frame", QVariant.Int),
            QgsField("end_frame", QVariant.Int),
            QgsField("length_m", QVariant.Double),
        ])
        layer.updateFields()
        provider.addFeatures(qgs_features)
        layer.updateExtents()

        def style(target):
            target.setRenderer(QgsSingleSymbolRenderer(
                QgsLineSymbol.createSimple({'color': color, 'width': '0.8'})))

        style(layer)
        layer = self._persist_memory_layer(
            layer, f"Transect{transect_id}_Route_{suffix}", "transect_layers",
            display_name="Flight Route")
        style(layer)
        QgsProject.instance().addMapLayer(layer, False)
        return layer

    def _label_transect_layer(self, layer):
        """Label the FoV polygon with 'name (count)'."""
        try:
            from qgis.core import (
                QgsPalLayerSettings, QgsTextFormat, QgsVectorLayerSimpleLabeling,
            )
            settings = QgsPalLayerSettings()
            settings.fieldName = '"name" || \' (\' || "count" || \')\''
            settings.isExpression = True
            settings.placement = QgsPalLayerSettings.Placement.OverPoint
            text_format = QgsTextFormat()
            text_format.setSize(9)
            settings.setFormat(text_format)
            layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
            layer.setLabelsEnabled(True)
            layer.triggerRepaint()
        except Exception as e:  # nosec B110
            self.log(f"Warning: could not label transect layer: {e}")

    def start_worker(self, step: str):
        """Start a background worker for the given step."""
        if self.worker is not None:
            QMessageBox.warning(self, "Processing", "Another process is already running.")
            return

        config = self.get_config()

        self.worker_thread = QThread()
        self.worker = ProcessingWorker(self.processor, step, config)
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error.connect(self.on_worker_error)
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.log.connect(self.log)

        # Cleanup connections
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        # Update UI
        self.set_buttons_enabled(False)
        self.abort_btn.setEnabled(True)  # Enable abort button during processing
        self.update_status(step, "🟡 Running...")
        self.progress_bar.setValue(0)
        if hasattr(self, 'analytics_progress_bar'):
            self.analytics_progress_bar.setValue(0)

        self.log(f"Starting {step}...")
        self.worker_thread.start()

    def on_worker_finished(self, step: str, success: bool):
        """Handle worker completion."""
        self.set_buttons_enabled(True)
        self.abort_btn.setEnabled(False)  # Disable abort button when done
        self.worker = None

        if success:
            self.update_status(step, "🟢 Completed")
            self.log(f"{step} completed successfully!")
            self.progress_bar.setValue(100)
            if hasattr(self, 'analytics_progress_bar'):
                self.analytics_progress_bar.setValue(100)

            # Show the distance-sampling summary dialog on completion.
            if step == "distance_sampling":
                self._show_distance_sampling_results()

            # Show the population-estimation summary dialog on completion.
            if step == "population_estimation":
                self._show_population_results()

            # In photo mode, warn if 0 images were matched (likely a timezone issue)
            if step in ("extract_thermal_frames", "extract_rgb_frames"):
                config = self.get_config()
                if config.get("input_mode") == "photo":
                    suffix = "t" if step == "extract_thermal_frames" else "w"
                    poses_path = os.path.join(
                        config.get("target_folder", ""), f"poses_{suffix}.json"
                    )
                    try:
                        with open(poses_path, "r", encoding="utf-8") as fh:
                            poses = json.load(fh)
                        if len(poses.get("images", [])) == 0:
                            QMessageBox.warning(
                                self,
                                "No Photos Matched",
                                "0 photos could be matched to the flight log.\n\n"
                                "This is often caused by an incorrect Timezone setting.\n"
                                "Please check the Timezone in the Photo Settings and try again."
                            )
                    except Exception:  # nosec B110
                        pass

                    # If the airdata field was empty, fill it with the
                    # EXIF-reconstructed CSV so the user can see it and reuse it.
                    if not self.airdata_path_edit.text().strip():
                        synthetic_csv = os.path.join(
                            config.get("target_folder", ""),
                            f"airdata_from_exif_{suffix}.csv",
                        )
                        if os.path.exists(synthetic_csv):
                            self.airdata_path_edit.setText(synthetic_csv)
        else:
            # Check if it was cancelled (worker would have logged it)
            self.update_status(step, "🔴 Cancelled/Failed")

    def on_worker_error(self, step: str, error_msg: str):
        """Handle worker error."""
        self.set_buttons_enabled(True)
        self.abort_btn.setEnabled(False)  # Disable abort button on error
        self.worker = None
        self.update_status(step, "🔴 Error")
        self.log(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Processing Error", error_msg)

    def on_worker_progress(self, value: int):
        """Update progress bar."""
        self.progress_bar.setValue(value)
        if hasattr(self, 'analytics_progress_bar'):
            self.analytics_progress_bar.setValue(value)

    def _abort_current_process(self):
        """Abort the currently running process."""
        if self.worker is not None:
            self.log("Aborting current process...")
            self.abort_btn.setEnabled(False)  # Prevent multiple clicks
            self.worker.cancel()
            # The worker will emit finished signal when it detects cancellation

    def update_status(self, step: str, status: str):
        """Update the status label for a step."""
        status_map = {
            # Both extraction modalities run under the single "Extract Frames"
            # step, so they share one status label.
            "extract_thermal_frames": self.extract_status,
            "extract_rgb_frames": self.extract_status,
            "detection": self.detect_status,
            "georeference": self.georef_status,
            "add_frame_detections": self.frame_detections_status,
            "tracking": self.track_status,
            "add_layers": self.layers_status,
            "calculate_fov": self.calculate_fov_status,
            "add_fov": self.add_fov_status,
            "add_merged_fov": self.add_merged_fov_status,
            "alfs": self.alfs_status,
            "export_geotiffs": self.export_geotiffs_status,
            "orthomosaic": self.orthomosaic_status,
            "flight_route": self.flight_route_status,
            "add_alfs": self.add_alfs_status,
            "add_geotiffs": self.add_geotiffs_status,
            "add_orthomosaic": self.add_orthomosaic_status,
            "add_flight_route": self.add_flight_route_status,
            "perpendicular": self.perpendicular_status,
            "add_perpendicular": self.add_perpendicular_status,
            "track_perpendicular": self.track_perpendicular_status,
            "add_track_perpendicular": self.add_track_perpendicular_status,
            "sam3_segmentation": self.sam3_segment_status,
            "sam3_georeference": self.sam3_georef_status,
            "add_sam3": self.add_sam3_status,
            # TRex import runs under the "Track Animals Or Import" step, so its
            # progress is reported on the tracking status label.
            "trex_import": self.track_status,
            "density_heatmap": self.density_status,
            "add_density": self.add_density_status,
            "distance_sampling": self.distance_sampling_status,
            "coverage_map": self.coverage_status,
            "add_coverage": self.add_coverage_status,
            "population_estimation": self.population_status,
            "add_transect_areas": self.add_transect_areas_status,
        }
        if step in status_map:
            status_map[step].setText(status)

    def set_buttons_enabled(self, enabled: bool):
        """Enable or disable all processing buttons."""
        self.extract_btn.setEnabled(enabled)
        self.detect_btn.setEnabled(enabled)
        self.georef_btn.setEnabled(enabled)
        self.add_frame_detections_btn.setEnabled(enabled)
        self.track_btn.setEnabled(enabled)
        self.add_layers_btn.setEnabled(enabled)
        self.calculate_fov_btn.setEnabled(enabled)
        self.add_fov_btn.setEnabled(enabled)
        self.add_merged_fov_btn.setEnabled(enabled)
        self.alfs_btn.setEnabled(enabled)
        self.export_geotiffs_btn.setEnabled(enabled)
        self.orthomosaic_btn.setEnabled(enabled)
        self.flight_route_btn.setEnabled(enabled)
        self.add_alfs_btn.setEnabled(enabled)
        self.add_geotiffs_btn.setEnabled(enabled)
        self.add_orthomosaic_btn.setEnabled(enabled)
        self.add_flight_route_btn.setEnabled(enabled)
        self.perpendicular_btn.setEnabled(enabled)
        self.add_perpendicular_btn.setEnabled(enabled)
        self.track_perpendicular_btn.setEnabled(enabled)
        self.add_track_perpendicular_btn.setEnabled(enabled)
        self.refresh_status_btn.setEnabled(enabled)
        self.sam3_segment_btn.setEnabled(enabled)
        self.sam3_georef_btn.setEnabled(enabled)
        self.add_sam3_btn.setEnabled(enabled)
        if hasattr(self, 'density_btn'):
            self.density_btn.setEnabled(enabled)
            self.add_density_btn.setEnabled(enabled)
            self.distance_sampling_btn.setEnabled(enabled)
        if hasattr(self, 'coverage_btn'):
            self.coverage_btn.setEnabled(enabled)
            self.add_coverage_btn.setEnabled(enabled)
        if hasattr(self, 'population_btn'):
            self.population_btn.setEnabled(enabled)
            self.add_transect_areas_btn.setEnabled(enabled)

    def run_perpendicular(self):
        """Run perpendicular distance calculation step."""
        config = self.get_config()
        target_folder = config["target_folder"]
        fr_suffix = "t" if config.get("flight_route_camera", "T") == "T" else "w"
        det_suffix = "t" if config.get("detection_camera", "T") == "T" else "w"

        route_file = os.path.join(target_folder, f"flight_route_{fr_suffix}", "flight_route.geojson")
        georef_file = os.path.join(target_folder, f"georeferenced_{det_suffix}", "georeferenced.txt")

        missing = []
        if not os.path.exists(route_file):
            missing.append("Flight route (run Generate Flight Route first)")
        if not os.path.exists(georef_file):
            missing.append("Geo-referenced detections (run Geo-Reference Detections first)")

        if missing:
            QMessageBox.warning(
                self, "Missing Prerequisites",
                "The following are required:\n\n" + "\n".join(f"• {m}" for m in missing)
            )
            return

        self.start_worker("perpendicular")

    def add_perpendicular_to_qgis(self):
        """Add perpendicular lines (detection → nearest flight route point) to QGIS."""
        config = self.get_config()
        fr_camera = config.get("flight_route_camera", "T")
        fr_suffix = "t" if fr_camera == "T" else "w"
        det_camera = config.get("detection_camera", "T")
        det_suffix = "t" if det_camera == "T" else "w"
        camera_label = "Thermal" if det_camera == "T" else "RGB"
        perp_file = os.path.join(config["target_folder"], f"flight_route_{fr_suffix}",
                                 f"perpendicular_{det_suffix}.json")

        if not os.path.exists(perp_file):
            QMessageBox.warning(
                self, "Missing Data",
                "Perpendicular distances have not been calculated.\n"
                "Please run 'Calculate Perpendicular' first."
            )
            return

        try:
            self.log(f"Adding {camera_label} perpendicular lines to QGIS...")
            self.update_status("add_perpendicular", "🟡 Loading...")

            with open(perp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            perpendiculas = data.get("perpendiculas", [])
            if not perpendiculas:
                QMessageBox.warning(self, "No Data", "No perpendicular results found.")
                self.update_status("add_perpendicular", "🔴 No data")
                return

            frame_vals = sorted(e["frame"] for e in perpendiculas)
            dlg = FrameRangeDialog(
                self, frame_vals[0], frame_vals[-1], len(set(frame_vals)), "frames"
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.update_status("add_perpendicular", "⚪ Cancelled")
                return

            start, end = dlg.selected_range()
            perpendiculas = [e for e in perpendiculas if start <= e["frame"] <= end]

            if not perpendiculas:
                QMessageBox.warning(self, "No Data", "No perpendicular results in selected range.")
                self.update_status("add_perpendicular", "🔴 No data")
                return

            if len(perpendiculas) > 100:
                reply = QMessageBox.question(
                    self,
                    "Many Lines",
                    f"Selected range contains {len(perpendiculas)} perpendicular lines.\n"
                    "Loading all may slow down QGIS.\n\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.update_status("add_perpendicular", "⚪ Cancelled")
                    return

            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{config['target_epsg']}")

            # Create a line layer: one line per detection (foot point → detection center)
            layer = QgsVectorLayer(
                "LineString?crs=" + target_crs.authid(),
                "Perpendicular Lines",
                "memory"
            )
            provider = layer.dataProvider()
            provider.addAttributes([
                QgsField("det_idx", QVariant.Int),
                QgsField("frame", QVariant.Int),
                QgsField("confidence", QVariant.Double),
                QgsField("class_id", QVariant.Int),
                QgsField("distance_m", QVariant.Double),
            ])
            layer.updateFields()

            features = []
            for entry in perpendiculas:
                foot = entry["foot_point"]
                center = entry["detection_center"]

                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromPolylineXY([
                    QgsPointXY(foot[0], foot[1]),
                    QgsPointXY(center[0], center[1])
                ]))
                feat.setAttributes([
                    entry["det_idx"],
                    entry["frame"],
                    round(entry["confidence"], 4),
                    entry["class_id"],
                    round(entry["distance_m"], 4),
                ])
                features.append(feat)

            provider.addFeatures(features)
            layer.updateExtents()

            # Style: thin orange lines
            symbol = QgsLineSymbol.createSimple({
                'color': '#ff7800',
                'width': '0.6',
                'capstyle': 'round',
            })
            layer.renderer().setSymbol(symbol)

            # Persist and add to project
            layer = self._persist_memory_layer(layer, "Perpendicular_Lines", "flight_route_layers")
            symbol = QgsLineSymbol.createSimple({
                'color': '#ff7800',
                'width': '0.6',
                'capstyle': 'round',
            })
            layer.renderer().setSymbol(symbol)

            group = self._create_layer_group(f"BAMBI Perpendicular ({camera_label})")
            QgsProject.instance().addMapLayer(layer, False)
            group.addLayer(layer)

            self.log(f"Added {len(features)} {camera_label} perpendicular lines to QGIS")
            self.update_status("add_perpendicular", "🟢 Completed")
            self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding perpendicular lines: {str(e)}")
            self.update_status("add_perpendicular", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add perpendicular lines: {str(e)}")

    def run_track_perpendicular(self):
        """Run track perpendicular distance calculation step."""
        config = self.get_config()
        target_folder = config["target_folder"]
        fr_suffix = "t" if config.get("flight_route_camera", "T") == "T" else "w"
        trk_suffix = "t" if config.get("tracking_camera", "T") == "T" else "w"

        route_file = os.path.join(target_folder, f"flight_route_{fr_suffix}", "flight_route.geojson")
        tracks_folder = os.path.join(target_folder, f"tracks_{trk_suffix}")

        missing = []
        if not os.path.exists(route_file):
            missing.append("Flight route (run Generate Flight Route first)")
        if not os.path.exists(tracks_folder):
            missing.append("Tracked animals (run Track Animals first)")

        if missing:
            QMessageBox.warning(
                self, "Missing Prerequisites",
                "The following are required:\n\n" + "\n".join(f"• {m}" for m in missing)
            )
            return

        self.start_worker("track_perpendicular")

    def add_track_perpendicular_to_qgis(self):
        """Add track perpendicular lines (last detection → nearest flight route point) to QGIS."""
        config = self.get_config()
        fr_camera = config.get("flight_route_camera", "T")
        fr_suffix = "t" if fr_camera == "T" else "w"
        trk_camera = config.get("tracking_camera", "T")
        trk_suffix = "t" if trk_camera == "T" else "w"
        camera_label = "Thermal" if trk_camera == "T" else "RGB"
        perp_file = os.path.join(config["target_folder"], f"flight_route_{fr_suffix}",
                                 f"perpendicular_tracks_{trk_suffix}.json")

        if not os.path.exists(perp_file):
            QMessageBox.warning(
                self, "Missing Data",
                "Track perpendicular distances have not been calculated.\n"
                "Please run 'Calculate Track Perpendicular' first."
            )
            return

        try:
            self.log(f"Adding {camera_label} track perpendicular lines to QGIS...")
            self.update_status("add_track_perpendicular", "🟡 Loading...")

            with open(perp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            track_entries = data.get("tracks", [])
            if not track_entries:
                QMessageBox.warning(self, "No Data", "No track perpendicular results found.")
                self.update_status("add_track_perpendicular", "🔴 No data")
                return

            frame_vals = sorted(e["last_frame"] for e in track_entries)
            dlg = FrameRangeDialog(
                self, frame_vals[0], frame_vals[-1], len(set(frame_vals)), "frames"
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.update_status("add_track_perpendicular", "⚪ Cancelled")
                return

            start, end = dlg.selected_range()
            track_entries = [e for e in track_entries if start <= e["last_frame"] <= end]

            if not track_entries:
                QMessageBox.warning(self, "No Data", "No track perpendicular results in selected range.")
                self.update_status("add_track_perpendicular", "🔴 No data")
                return

            if len(track_entries) > 100:
                reply = QMessageBox.question(
                    self,
                    "Many Lines",
                    f"Selected range contains {len(track_entries)} track perpendicular lines.\n"
                    "Loading all may slow down QGIS.\n\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.update_status("add_track_perpendicular", "⚪ Cancelled")
                    return

            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{config['target_epsg']}")

            layer = QgsVectorLayer(
                "LineString?crs=" + target_crs.authid(),
                "Track Perpendicular Lines",
                "memory"
            )
            provider = layer.dataProvider()
            provider.addAttributes([
                QgsField("track_id", QVariant.Int),
                QgsField("last_frame", QVariant.Int),
                QgsField("last_image", QVariant.String),
                QgsField("confidence", QVariant.Double),
                QgsField("class_id", QVariant.Int),
                QgsField("distance_m", QVariant.Double),
            ])
            layer.updateFields()

            features = []
            for entry in track_entries:
                foot = entry["foot_point"]
                center = entry["detection_center"]

                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromPolylineXY([
                    QgsPointXY(foot[0], foot[1]),
                    QgsPointXY(center[0], center[1])
                ]))
                feat.setAttributes([
                    entry["track_id"],
                    entry["last_frame"],
                    entry.get("last_image", ""),
                    round(entry["confidence"], 4),
                    entry["class_id"],
                    round(entry["distance_m"], 4),
                ])
                features.append(feat)

            provider.addFeatures(features)
            layer.updateExtents()

            # Style: slightly thicker purple lines to distinguish from detection perpendiculars
            symbol = QgsLineSymbol.createSimple({
                'color': '#9b30ff',
                'width': '0.8',
                'capstyle': 'round',
            })
            layer.renderer().setSymbol(symbol)

            layer = self._persist_memory_layer(layer, "Track_Perpendicular_Lines", "flight_route_layers")
            symbol = QgsLineSymbol.createSimple({
                'color': '#9b30ff',
                'width': '0.8',
                'capstyle': 'round',
            })
            layer.renderer().setSymbol(symbol)

            group = self._create_layer_group(f"BAMBI Track Perpendicular ({camera_label})")
            QgsProject.instance().addMapLayer(layer, False)
            group.addLayer(layer)

            self.log(f"Added {len(features)} {camera_label} track perpendicular lines to QGIS")
            self.update_status("add_track_perpendicular", "🟢 Completed")
            self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding track perpendicular lines: {str(e)}")
            self.update_status("add_track_perpendicular", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add track perpendicular lines: {str(e)}")

    def add_tracks_to_qgis(self):
        """Add tracked animals as individual layer groups to QGIS.

        Each track gets its own subgroup containing:
        - A polyline showing the movement path
        - The final bounding box as a polygon

        This allows users to show/hide individual animals.
        """
        config = self.get_config()
        trk_camera = config.get("tracking_camera", "T")
        trk_suffix = "t" if trk_camera == "T" else "w"
        camera_label = "Thermal" if trk_camera == "T" else "RGB"
        tracks_folder = os.path.join(config["target_folder"], f"tracks_{trk_suffix}")

        if not os.path.exists(tracks_folder):
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_label} tracking has not been completed.\nPlease run step A2 (Track Animals Or Import) first."
            )
            return

        try:
            self.log(f"Adding {camera_label} tracks to QGIS...")
            self.update_status("add_layers", "🟡 Loading...")

            # Get target CRS
            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{config['target_epsg']}")

            # Find all track files
            track_files = []
            for root, dirs, files in os.walk(tracks_folder):
                for f in files:
                    if f.endswith(".csv") and not f.endswith("_pixel.csv"):
                        track_files.append(os.path.join(root, f))

            if not track_files:
                QMessageBox.warning(self, "No Tracks", "No track files found.")
                self.update_status("add_layers", "🔴 No files")
                return

            # Collect all tracks from all files
            all_tracks = {}  # (file_basename, track_id) -> detections

            for track_file in track_files:
                tracks = self.load_tracks_from_csv(track_file)
                if not tracks:
                    continue
                file_basename = os.path.basename(track_file).replace(".csv", "")
                for track_id, detections in tracks.items():
                    if detections:
                        all_tracks[(file_basename, track_id)] = detections

            if not all_tracks:
                QMessageBox.warning(self, "No Tracks", "No valid tracks found.")
                self.update_status("add_layers", "🔴 No valid tracks")
                return

            # Ask for a frame range — a track is kept when its final frame
            # falls inside the range (its earlier history is not considered)
            final_frames = {
                key: max(d['frame'] for d in detections)
                for key, detections in all_tracks.items()
            }
            dlg = FrameRangeDialog(
                self, min(final_frames.values()), max(final_frames.values()),
                len(all_tracks), "track end frames"
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.update_status("add_layers", "⚪ Cancelled")
                return

            start, end = dlg.selected_range()
            all_tracks = {
                key: detections for key, detections in all_tracks.items()
                if start <= final_frames[key] <= end
            }

            if not all_tracks:
                QMessageBox.warning(
                    self, "No Tracks",
                    "No tracks end within the selected frame range."
                )
                self.update_status("add_layers", "🔴 No valid tracks")
                return

            # Check if there are many tracks - warn user
            num_tracks = len(all_tracks)
            if num_tracks > 50:
                reply = QMessageBox.question(
                    self,
                    "Many Tracks",
                    f"Found {num_tracks} tracks. Creating individual layers for each may slow down QGIS.\n\n"
                    "Continue with individual layers?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                if reply == QMessageBox.StandardButton.No:
                    self.update_status("add_layers", "⚪ Cancelled")
                    return

            # Create main group for all tracks at top of layer tree
            main_group = self._create_layer_group(f"BAMBI Wildlife Tracks ({camera_label})")

            # Generate colors for tracks (cycle through a palette)
            colors = [
                (255, 0, 0),  # Red
                (0, 150, 0),  # Green
                (0, 100, 255),  # Blue
                (255, 165, 0),  # Orange
                (128, 0, 128),  # Purple
                (0, 200, 200),  # Cyan
                (255, 105, 180),  # Pink
                (139, 69, 19),  # Brown
                (0, 0, 139),  # Dark Blue
                (34, 139, 34),  # Forest Green
            ]

            total_tracks = 0

            for idx, ((file_basename, track_id), detections) in enumerate(all_tracks.items()):
                # Sort detections by frame
                detections_sorted = sorted(detections, key=lambda d: d['frame'])

                # Get color for this track
                color = colors[idx % len(colors)]
                color_str = f"{color[0]},{color[1]},{color[2]}"

                # Create subgroup for this track
                track_name = f"Track {track_id}"
                if len(track_files) > 1:
                    track_name = f"Track {file_basename}_{track_id}"
                track_group = main_group.addGroup(track_name)

                # Create path layer (polyline) — only when there are 2+ detections
                path_layer = None
                if len(detections_sorted) >= 2:
                    path_layer = QgsVectorLayer(
                        "LineString?crs=" + target_crs.authid(),
                        "Path",
                        "memory"
                    )
                    path_provider = path_layer.dataProvider()
                    path_provider.addAttributes([
                        QgsField("track_id", QVariant.Int),
                        QgsField("start_frame", QVariant.Int),
                        QgsField("end_frame", QVariant.Int),
                        QgsField("num_detections", QVariant.Int),
                        QgsField("avg_confidence", QVariant.Double)
                    ])
                    path_layer.updateFields()

                    center_points = [
                        QgsPointXY((det['x1'] + det['x2']) / 2.0, (det['y1'] + det['y2']) / 2.0)
                        for det in detections_sorted
                    ]

                    path_feat = QgsFeature()
                    path_feat.setGeometry(QgsGeometry.fromPolylineXY(center_points))
                    avg_conf = sum(d['confidence'] for d in detections_sorted) / len(detections_sorted)
                    path_feat.setAttributes([
                        track_id,
                        detections_sorted[0]['frame'],
                        detections_sorted[-1]['frame'],
                        len(detections_sorted),
                        round(avg_conf, 4)
                    ])
                    path_provider.addFeatures([path_feat])
                    path_layer.updateExtents()

                    self._style_path_layer(path_layer, color_str)

                # Create final position layer (polygon)
                bbox_layer = QgsVectorLayer(
                    "Polygon?crs=" + target_crs.authid(),
                    "Final Position",
                    "memory"
                )
                bbox_provider = bbox_layer.dataProvider()
                bbox_provider.addAttributes([
                    QgsField("track_id", QVariant.Int),
                    QgsField("frame", QVariant.Int),
                    QgsField("confidence", QVariant.Double),
                    QgsField("class_id", QVariant.Int)
                ])
                bbox_layer.updateFields()

                # Create final bbox feature
                final_det = detections_sorted[-1]
                bbox_feat = QgsFeature()
                points = [
                    QgsPointXY(final_det['x1'], final_det['y1']),
                    QgsPointXY(final_det['x2'], final_det['y1']),
                    QgsPointXY(final_det['x2'], final_det['y2']),
                    QgsPointXY(final_det['x1'], final_det['y2']),
                    QgsPointXY(final_det['x1'], final_det['y1'])
                ]
                bbox_feat.setGeometry(QgsGeometry.fromPolygonXY([points]))
                bbox_feat.setAttributes([
                    track_id,
                    final_det['frame'],
                    final_det['confidence'],
                    final_det['class_id']
                ])
                bbox_provider.addFeatures([bbox_feat])
                bbox_layer.updateExtents()

                # Style bbox layer
                self._style_bbox_layer(bbox_layer, color_str)

                # Persist layers to GeoPackage (using unique file names, keeping display names)
                bbox_layer = self._persist_memory_layer(
                    bbox_layer, f"Track{track_id}_FinalPosition", "tracks_layers",
                    display_name="Final Position"
                )
                self._style_bbox_layer(bbox_layer, color_str)
                # Tag layer so the inspector tool can identify and handle it
                bbox_layer.setCustomProperty("bambi_layer_type", "track_final")
                bbox_layer.setCustomProperty("bambi_target_folder", config["target_folder"])
                bbox_layer.setCustomProperty("bambi_detection_camera", config["detection_camera"])
                bbox_layer.setCustomProperty("bambi_dem_path", config.get("dem_path", ""))
                bbox_layer.setCustomProperty("bambi_correction_path", config.get("correction_path", ""))
                QgsProject.instance().addMapLayer(bbox_layer, False)
                track_group.addLayer(bbox_layer)

                if path_layer is not None:
                    path_layer = self._persist_memory_layer(
                        path_layer, f"Track{track_id}_Path", "tracks_layers",
                        display_name="Path"
                    )
                    self._style_path_layer(path_layer, color_str)
                    # Tag layer so the inspector tool can identify and handle it
                    path_layer.setCustomProperty("bambi_layer_type", "track_path")
                    path_layer.setCustomProperty("bambi_target_folder", config["target_folder"])
                    path_layer.setCustomProperty("bambi_detection_camera", config["detection_camera"])
                    path_layer.setCustomProperty("bambi_dem_path", config.get("dem_path", ""))
                    path_layer.setCustomProperty("bambi_correction_path", config.get("correction_path", ""))
                    QgsProject.instance().addMapLayer(path_layer, False)
                    track_group.addLayer(path_layer)

                # Collapse the track subgroup by default
                track_group.setExpanded(False)

                total_tracks += 1

            # Collapse main group
            main_group.setExpanded(True)  # Keep main group expanded to show tracks

            self.log(f"Added {total_tracks} individual track layers to QGIS")
            self.update_status("add_layers", "🟢 Completed")

            # Refresh canvas
            if total_tracks > 0:
                self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding layers: {str(e)}")
            self.update_status("add_layers", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add tracks: {str(e)}")

    def _style_path_layer(self, layer, color_str: str):
        """Apply styling to a track path layer.

        :param layer: Line layer to style
        :param color_str: RGB color string like "255,0,0"
        """
        from qgis.core import QgsLineSymbol, QgsSingleSymbolRenderer

        try:
            symbol = QgsLineSymbol.createSimple({
                'color': f"{color_str},255",
                'width': '1.2',
                'capstyle': 'round',
                'joinstyle': 'round'
            })
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            layer.triggerRepaint()
        except Exception:  # nosec B110
            pass

    def _style_bbox_layer(self, layer, color_str: str):
        """Apply styling to a final position (bbox) layer.

        :param layer: Polygon layer to style
        :param color_str: RGB color string like "255,0,0"
        """
        from qgis.core import QgsFillSymbol, QgsSingleSymbolRenderer

        try:
            symbol = QgsFillSymbol.createSimple({
                'color': f"{color_str},80",  # Semi-transparent fill
                'outline_color': f"{color_str},255",
                'outline_width': '0.8'
            })
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            layer.triggerRepaint()
        except Exception:  # nosec B110
            pass

    def _style_track_layers(self, final_pos_layer, paths_layer):
        """Apply default styling to track visualization layers (legacy method).

        :param final_pos_layer: Polygon layer with final bounding boxes
        :param paths_layer: Line layer with track paths
        """
        from qgis.core import (
            QgsFillSymbol, QgsLineSymbol,
            QgsSingleSymbolRenderer
        )

        try:
            # Style final positions (semi-transparent red fill)
            final_symbol = QgsFillSymbol.createSimple({
                'color': '255,100,100,100',  # Semi-transparent red
                'outline_color': '255,0,0,255',  # Solid red outline
                'outline_width': '0.5'
            })
            final_pos_layer.setRenderer(QgsSingleSymbolRenderer(final_symbol))

            # Style track paths (bold colored line)
            path_symbol = QgsLineSymbol.createSimple({
                'color': '0,100,255,255',  # Blue
                'width': '1.0',
                'capstyle': 'round',
                'joinstyle': 'round'
            })
            paths_layer.setRenderer(QgsSingleSymbolRenderer(path_symbol))

            # Trigger repaint
            final_pos_layer.triggerRepaint()
            paths_layer.triggerRepaint()

        except Exception as e:
            # Styling is optional, don't fail if it doesn't work
            self.log(f"Note: Could not apply default styling: {e}")

    def load_tracks_from_csv(self, csv_path: str) -> Dict[int, list]:
        """Load tracks from a CSV file (core.pipeline_outputs)."""
        from .core.pipeline_outputs import load_geo_tracks_by_id
        return load_geo_tracks_by_id(csv_path, log_fn=self.log)

    def add_fov_to_qgis(self):
        """Add Field of View polygons as QGIS layers."""
        config = self.get_config()
        fov_camera = config.get("fov_camera", "T")
        fov_suffix = "t" if fov_camera == "T" else "w"
        camera_label = "Thermal" if fov_camera == "T" else "RGB"
        fov_folder = os.path.join(config["target_folder"], f"fov_{fov_suffix}")
        fov_file = os.path.join(fov_folder, "fov_polygons.txt")

        if not os.path.exists(fov_file):
            QMessageBox.warning(
                self,
                "Missing FoV Data",
                f"{camera_label} FoV calculation has not been completed.\n"
                f"Please run Step 5 (Calculate Field of View) first."
            )
            return

        try:
            self.log(f"Adding {camera_label} FoV layers to QGIS...")
            self.update_status("add_fov", "🟡 Loading...")

            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{config['target_epsg']}")
            fov_polygons = self.load_fov_polygons(fov_file)

            if not fov_polygons:
                QMessageBox.warning(self, "No FoV Data", "No valid FoV polygons found.")
                self.update_status("add_fov", "🔴 No data")
                return

            frame_keys = sorted(fov_polygons.keys())
            dlg = FrameRangeDialog(
                self, frame_keys[0], frame_keys[-1], len(frame_keys), "frames"
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.update_status("add_fov", "⚪ Cancelled")
                return

            start, end = dlg.selected_range()
            fov_polygons = {k: v for k, v in fov_polygons.items() if start <= k <= end}

            if not fov_polygons:
                QMessageBox.warning(self, "No FoV Data", "No FoV polygons in selected range.")
                self.update_status("add_fov", "🔴 No data")
                return

            num_frames = len(fov_polygons)
            if num_frames > 100:
                reply = QMessageBox.question(
                    self,
                    "Layer Mode",
                    f"Selected range contains {num_frames} frames.\n\n"
                    "- Yes: one layer per frame (separate)\n"
                    "- No: all polygons in a single combined layer",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
                )
                if reply == QMessageBox.StandardButton.Cancel:
                    self.update_status("add_fov", "⚪ Cancelled")
                    return
                elif reply == QMessageBox.StandardButton.Yes:
                    self._add_fov_separate_layers(fov_polygons, target_crs, camera_label)
                else:
                    self._add_fov_combined_layer(fov_polygons, target_crs, camera_label)
            else:
                self._add_fov_separate_layers(fov_polygons, target_crs, camera_label)

            self.log("Added FoV layers to QGIS")
            self.update_status("add_fov", "🟢 Completed")
            self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding FoV layers: {str(e)}")
            self.update_status("add_fov", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add FoV layers: {str(e)}")

    def _add_fov_separate_layers(self, fov_polygons: Dict[int, list], target_crs,
                                 camera_label: str = "Thermal"):
        """Add FoV polygons as separate layers for each frame."""
        # Create a group for the layers at top of layer tree
        group = self._create_layer_group(f"BAMBI FoV Polygons ({camera_label})")

        for frame_idx, points in fov_polygons.items():
            if len(points) < 3:
                continue

            # Create polygon layer for this frame
            layer_name = f"FoV_Frame_{frame_idx:06d}"
            layer = QgsVectorLayer("Polygon?crs=" + target_crs.authid(), layer_name, "memory")

            provider = layer.dataProvider()

            # Add fields
            provider.addAttributes([
                QgsField("frame", QVariant.Int),
                QgsField("num_points", QVariant.Int)
            ])
            layer.updateFields()

            # Create feature
            feat = QgsFeature()

            # Create polygon from points (only x, y coordinates)
            qgs_points = [QgsPointXY(p[0], p[1]) for p in points]
            qgs_points.append(qgs_points[0])  # Close the polygon

            feat.setGeometry(QgsGeometry.fromPolygonXY([qgs_points]))
            feat.setAttributes([frame_idx, len(points)])

            provider.addFeatures([feat])
            layer.updateExtents()

            # Persist layer to GeoPackage
            layer = self._persist_memory_layer(layer, layer_name, "fov_layers")

            # Tag layer so the FoV inspector tool can identify and handle it
            layer.setCustomProperty("bambi_layer_type", "fov")
            layer.setCustomProperty("bambi_target_folder", self.target_folder_edit.text().strip())
            layer.setCustomProperty(
                "bambi_detection_camera",
                "T" if self.detection_camera_combo.currentIndex() == 0 else "W")
            layer.setCustomProperty("bambi_dem_path", self.dem_path_edit.text().strip())
            layer.setCustomProperty("bambi_correction_path", self.correction_path_edit.text().strip())

            # Add layer to project and group
            QgsProject.instance().addMapLayer(layer, False)
            group.addLayer(layer)

        # Collapse the group
        group.setExpanded(False)

    def _add_fov_combined_layer(self, fov_polygons: Dict[int, list], target_crs,
                                camera_label: str = "Thermal"):
        """Add all FoV polygons as a single combined layer."""
        # Create a single layer with all polygons
        layer_name = f"BAMBI FoV Polygons - Combined ({camera_label})"
        layer = QgsVectorLayer("Polygon?crs=" + target_crs.authid(), layer_name, "memory")

        provider = layer.dataProvider()

        # Add fields
        provider.addAttributes([
            QgsField("frame", QVariant.Int),
            QgsField("num_points", QVariant.Int)
        ])
        layer.updateFields()

        features = []
        for frame_idx, points in fov_polygons.items():
            if len(points) < 3:
                continue

            feat = QgsFeature()

            # Create polygon from points
            qgs_points = [QgsPointXY(p[0], p[1]) for p in points]
            qgs_points.append(qgs_points[0])  # Close the polygon

            feat.setGeometry(QgsGeometry.fromPolygonXY([qgs_points]))
            feat.setAttributes([frame_idx, len(points)])
            features.append(feat)

        provider.addFeatures(features)
        layer.updateExtents()

        # Persist layer to GeoPackage
        layer = self._persist_memory_layer(layer, "FoV_Combined", "fov_layers")

        # Tag layer so the FoV inspector tool can identify and handle it
        layer.setCustomProperty("bambi_layer_type", "fov")
        layer.setCustomProperty("bambi_target_folder", self.target_folder_edit.text().strip())
        layer.setCustomProperty(
            "bambi_detection_camera",
            "T" if self.detection_camera_combo.currentIndex() == 0 else "W")
        layer.setCustomProperty("bambi_dem_path", self.dem_path_edit.text().strip())
        layer.setCustomProperty("bambi_correction_path", self.correction_path_edit.text().strip())

        # Add layer to project
        QgsProject.instance().addMapLayer(layer)

    def add_merged_fov_to_qgis(self):
        """Add merged (union) Field of View polygon as single QGIS layer.

        This creates a single polygon/multipolygon from the union of all
        individual FoV polygons, useful for calculating total surveyed area.
        """
        config = self.get_config()
        fov_camera = config.get("fov_camera", "T")
        fov_suffix = "t" if fov_camera == "T" else "w"
        camera_label = "Thermal" if fov_camera == "T" else "RGB"
        fov_folder = os.path.join(config["target_folder"], f"fov_{fov_suffix}")
        fov_file = os.path.join(fov_folder, "fov_polygons.txt")

        if not os.path.exists(fov_file):
            QMessageBox.warning(
                self,
                "Missing FoV Data",
                f"{camera_label} FoV calculation has not been completed.\n"
                f"Please run Step 5 (Calculate Field of View) first."
            )
            return

        try:
            self.log(f"Creating merged {camera_label} FoV layer...")
            self.update_status("add_merged_fov", "🟡 Processing...")

            # Get target CRS
            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{config['target_epsg']}")

            # Load FoV polygons
            fov_polygons = self.load_fov_polygons(fov_file)

            if not fov_polygons:
                QMessageBox.warning(self, "No FoV Data", "No valid FoV polygons found.")
                self.update_status("add_merged_fov", "🔴 No data")
                return

            self.log(f"Merging {len(fov_polygons)} FoV polygons...")

            # Collect all polygon geometries
            geometries = []
            for frame_idx, points in fov_polygons.items():
                if len(points) < 3:
                    continue

                # Create polygon from points
                qgs_points = [QgsPointXY(p[0], p[1]) for p in points]
                qgs_points.append(qgs_points[0])  # Close the polygon

                geom = QgsGeometry.fromPolygonXY([qgs_points])
                if geom.isGeosValid():
                    geometries.append(geom)
                else:
                    # Try to fix invalid geometry
                    fixed_geom = geom.makeValid()
                    if fixed_geom.isGeosValid():
                        geometries.append(fixed_geom)

            if not geometries:
                QMessageBox.warning(self, "No Valid Polygons", "No valid polygons to merge.")
                self.update_status("add_merged_fov", "🔴 No valid data")
                return

            # Merge all geometries using unaryUnion
            self.log("Computing union of all FoV polygons...")
            merged_geom = QgsGeometry.unaryUnion(geometries)

            if merged_geom.isEmpty():
                QMessageBox.warning(self, "Merge Failed", "Failed to merge FoV polygons.")
                self.update_status("add_merged_fov", "🔴 Merge failed")
                return

            # Create layer for merged polygon
            layer_name = f"BAMBI FoV Coverage ({camera_label})"
            layer = QgsVectorLayer("Polygon?crs=" + target_crs.authid(), layer_name, "memory")

            provider = layer.dataProvider()

            # Add fields
            provider.addAttributes([
                QgsField("num_frames", QVariant.Int),
                QgsField("area_m2", QVariant.Double),
                QgsField("area_ha", QVariant.Double)
            ])
            layer.updateFields()

            # Create feature
            feat = QgsFeature()
            feat.setGeometry(merged_geom)

            # Calculate area
            area_m2 = merged_geom.area()
            area_ha = area_m2 / 10000.0

            feat.setAttributes([len(fov_polygons), area_m2, area_ha])

            provider.addFeatures([feat])
            layer.updateExtents()

            # Style the layer with semi-transparent fill
            symbol = layer.renderer().symbol()
            symbol.setColor(QColor(0, 150, 255, 50))  # Light blue with transparency
            symbol.symbolLayer(0).setStrokeColor(QColor(0, 100, 200))
            symbol.symbolLayer(0).setStrokeWidth(0.5)

            # Persist layer to GeoPackage
            layer = self._persist_memory_layer(layer, "FoV_Coverage_Merged", "fov_layers")

            # Re-apply style after persistence
            symbol = layer.renderer().symbol()
            symbol.setColor(QColor(0, 150, 255, 50))
            symbol.symbolLayer(0).setStrokeColor(QColor(0, 100, 200))
            symbol.symbolLayer(0).setStrokeWidth(0.5)

            # Add layer to project
            QgsProject.instance().addMapLayer(layer)

            self.log("Merged FoV layer added to QGIS")
            self.log(f"  Total coverage area: {area_m2:.2f} m² ({area_ha:.4f} ha)")
            self.update_status("add_merged_fov", "🟢 Completed")
            self.iface.mapCanvas().refresh()

            # Show info to user
            QMessageBox.information(
                self,
                "Merged FoV Created",
                f"Merged FoV polygon created from {len(fov_polygons)} frames.\n\n"
                "Total coverage area:\n"
                f"  {area_m2:,.2f} m²\n"
                f"  {area_ha:.4f} ha\n\n"
                "You can also use QGIS Field Calculator for precise area calculation."
            )

        except Exception as e:
            self.log(f"Error creating merged FoV layer: {str(e)}")
            self.update_status("add_merged_fov", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to create merged FoV layer: {str(e)}")

    def load_fov_polygons(self, fov_file: str) -> Dict[int, list]:
        """Load FoV polygons from file (core.pipeline_outputs).

        :param fov_file: Path to FoV polygons file
        :return: Dictionary mapping frame index to list of (x, y, z) points
        """
        from .core.pipeline_outputs import load_fov_polygons_3d
        return load_fov_polygons_3d(fov_file, log_fn=self.log)

    def add_frame_detections_to_qgis(self):
        """Add geo-referenced detections as QGIS layers (one layer per frame)."""
        config = self.get_config()
        det_camera = config.get("detection_camera", "T")
        det_suffix = "t" if det_camera == "T" else "w"
        camera_label = "Thermal" if det_camera == "T" else "RGB"
        georef_folder = os.path.join(config["target_folder"], f"georeferenced_{det_suffix}")
        georef_file = os.path.join(georef_folder, "georeferenced.txt")

        if not os.path.exists(georef_file):
            QMessageBox.warning(
                self,
                "Missing Data",
                f"{camera_label} geo-referencing has not been completed.\n"
                f"Please run Geo-Reference Detections (under step A1) first."
            )
            return

        try:
            self.log(f"Adding per-frame {camera_label} detection layers to QGIS...")
            self.update_status("add_frame_detections", "🟡 Loading...")

            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{config['target_epsg']}")
            frame_detections = self.load_detections_by_frame(georef_file)

            if not frame_detections:
                QMessageBox.warning(self, "No Detections", "No valid detections found.")
                self.update_status("add_frame_detections", "🔴 No data")
                return

            frame_keys = sorted(frame_detections.keys())
            dlg = FrameRangeDialog(
                self, frame_keys[0], frame_keys[-1], len(frame_keys), "frames"
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.update_status("add_frame_detections", "⚪ Cancelled")
                return

            start, end = dlg.selected_range()
            frame_detections = {k: v for k, v in frame_detections.items() if start <= k <= end}

            if not frame_detections:
                QMessageBox.warning(self, "No Detections", "No detections in selected range.")
                self.update_status("add_frame_detections", "🔴 No data")
                return

            num_frames = len(frame_detections)
            if num_frames > 100:
                reply = QMessageBox.question(
                    self,
                    "Layer Mode",
                    f"Selected range contains {num_frames} frames.\n\n"
                    "- Yes: one layer per frame (separate)\n"
                    "- No: all detections in a single combined layer",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
                )
                if reply == QMessageBox.StandardButton.Cancel:
                    self.update_status("add_frame_detections", "⚪ Cancelled")
                    return
                elif reply == QMessageBox.StandardButton.Yes:
                    self._add_detections_separate_layers(frame_detections, target_crs, camera_label)
                else:
                    self._add_detections_combined_layer(frame_detections, target_crs, camera_label)
            else:
                self._add_detections_separate_layers(frame_detections, target_crs, camera_label)

            self.log("Added detection layers to QGIS")
            self.update_status("add_frame_detections", "🟢 Completed")
            self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding detection layers: {str(e)}")
            self.update_status("add_frame_detections", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add detection layers: {str(e)}")

    def _add_detections_separate_layers(self, frame_detections: Dict[int, list], target_crs,
                                        camera_label: str = "Thermal"):
        """Add detections as separate layers for each frame."""
        # Create a group for the layers at top of layer tree
        group = self._create_layer_group(f"BAMBI Frame Detections ({camera_label})")

        for frame_idx, detections in frame_detections.items():
            if not detections:
                continue

            # Create polygon layer for this frame
            layer_name = f"Detections_Frame_{frame_idx:06d}"
            layer = QgsVectorLayer("Polygon?crs=" + target_crs.authid(), layer_name, "memory")

            provider = layer.dataProvider()

            # Add fields
            provider.addAttributes([
                QgsField("det_id", QVariant.Int),
                QgsField("frame", QVariant.Int),
                QgsField("confidence", QVariant.Double),
                QgsField("class_id", QVariant.Int)
            ])
            layer.updateFields()

            # Add features
            features = []
            for det in detections:
                feat = QgsFeature()

                # Create polygon from bounding box
                points = [
                    QgsPointXY(det['x1'], det['y1']),
                    QgsPointXY(det['x2'], det['y1']),
                    QgsPointXY(det['x2'], det['y2']),
                    QgsPointXY(det['x1'], det['y2']),
                    QgsPointXY(det['x1'], det['y1'])  # Close the polygon
                ]
                feat.setGeometry(QgsGeometry.fromPolygonXY([points]))
                feat.setAttributes([
                    det['idx'],
                    det['frame'],
                    det['confidence'],
                    det['class_id']
                ])
                features.append(feat)

            provider.addFeatures(features)
            layer.updateExtents()

            # Persist layer to GeoPackage
            layer = self._persist_memory_layer(layer, layer_name, "detection_layers")

            # Tag layer so the inspector tool can identify and handle it
            layer.setCustomProperty("bambi_layer_type", "detection")
            layer.setCustomProperty("bambi_target_folder", self.target_folder_edit.text().strip())
            layer.setCustomProperty(
                "bambi_detection_camera",
                "T" if self.detection_camera_combo.currentIndex() == 0 else "W")
            layer.setCustomProperty("bambi_dem_path", self.dem_path_edit.text().strip())
            layer.setCustomProperty("bambi_correction_path", self.correction_path_edit.text().strip())

            # Add layer to project and group
            QgsProject.instance().addMapLayer(layer, False)
            group.addLayer(layer)

        # Collapse the group
        group.setExpanded(False)

    def _add_detections_combined_layer(self, frame_detections: Dict[int, list], target_crs,
                                       camera_label: str = "Thermal"):
        """Add all detections in a single combined layer."""
        layer_name = f"BAMBI Detections - All Frames ({camera_label})"
        layer = QgsVectorLayer("Polygon?crs=" + target_crs.authid(), layer_name, "memory")

        provider = layer.dataProvider()

        # Add fields
        provider.addAttributes([
            QgsField("det_id", QVariant.Int),
            QgsField("frame", QVariant.Int),
            QgsField("confidence", QVariant.Double),
            QgsField("class_id", QVariant.Int)
        ])
        layer.updateFields()

        features = []
        for frame_idx, detections in frame_detections.items():
            for det in detections:
                feat = QgsFeature()

                # Create polygon from bounding box
                points = [
                    QgsPointXY(det['x1'], det['y1']),
                    QgsPointXY(det['x2'], det['y1']),
                    QgsPointXY(det['x2'], det['y2']),
                    QgsPointXY(det['x1'], det['y2']),
                    QgsPointXY(det['x1'], det['y1'])
                ]
                feat.setGeometry(QgsGeometry.fromPolygonXY([points]))
                feat.setAttributes([
                    det['idx'],
                    det['frame'],
                    det['confidence'],
                    det['class_id']
                ])
                features.append(feat)

        provider.addFeatures(features)
        layer.updateExtents()

        # Persist layer to GeoPackage
        layer = self._persist_memory_layer(layer, "Detections_AllFrames", "detection_layers")

        # Tag layer so the inspector tool can identify and handle it
        layer.setCustomProperty("bambi_layer_type", "detection")
        layer.setCustomProperty("bambi_target_folder", self.target_folder_edit.text().strip())
        layer.setCustomProperty(
            "bambi_detection_camera",
            "T" if self.detection_camera_combo.currentIndex() == 0 else "W")
        layer.setCustomProperty("bambi_dem_path", self.dem_path_edit.text().strip())
        layer.setCustomProperty("bambi_correction_path", self.correction_path_edit.text().strip())

        # Add layer to project
        QgsProject.instance().addMapLayer(layer)

    def load_detections_by_frame(self, georef_file: str) -> Dict[int, list]:
        """Load geo-referenced detections grouped by frame (core.pipeline_outputs).

        :param georef_file: Path to georeferenced detections file
        :return: Dictionary mapping frame index to list of detections
        """
        from .core.pipeline_outputs import load_georef_detections_by_frame
        return load_georef_detections_by_frame(georef_file, log_fn=self.log)

    def add_alfs_to_qgis(self):
        """Add all GeoTIFFs in the alfs folder to QGIS as raster layers."""
        config = self.get_config()
        alfs_camera = config.get("alfs_camera", "T")
        alfs_suffix = "t" if alfs_camera == "T" else "w"
        camera_label = "Thermal" if alfs_camera == "T" else "RGB"
        alfs_folder = os.path.join(config["target_folder"], f"alfs_{alfs_suffix}")

        if not os.path.exists(alfs_folder):
            QMessageBox.warning(
                self,
                "Missing ALFS",
                f"{camera_label} alfs has not been generated.\nPlease run Step 6 (Generate ALFS) first."
            )
            return

        try:
            self.log(f"Adding {camera_label} alfs to QGIS...")
            self.update_status("add_alfs", "🟡 Loading...")

            tif_files = sorted([
                os.path.join(alfs_folder, f)
                for f in os.listdir(alfs_folder)
                if f.lower().endswith((".tif", ".tiff"))
            ])

            if not tif_files:
                QMessageBox.warning(
                    self,
                    "Missing ALFS",
                    "ALFS has not been generated.\nPlease run Step 6 (Generate ALFS) first."
                )
                self.update_status("add_alfs", "🔴 No files")
                return

            if len(tif_files) == 1:
                layer_name = f"BAMBI ALFS ({camera_label})"
                layer = QgsRasterLayer(tif_files[0], layer_name)
                if not layer.isValid():
                    raise RuntimeError(f"Failed to load raster: {tif_files[0]}")
                QgsProject.instance().addMapLayer(layer)
                self.log(f"Added {camera_label} alfs layer to QGIS")
            else:
                group = self._create_layer_group(f"BAMBI ALFS ({camera_label})")
                loaded_count = 0
                for tif_path in tif_files:
                    layer_name = os.path.splitext(os.path.basename(tif_path))[0]
                    layer = QgsRasterLayer(tif_path, layer_name)
                    if layer.isValid():
                        QgsProject.instance().addMapLayer(layer, False)
                        group.addLayer(layer)
                        loaded_count += 1
                    else:
                        self.log(f"Warning: Could not load {os.path.basename(tif_path)}")
                self.log(f"Added {loaded_count} alfs file(s) to QGIS")

            self.update_status("add_alfs", "🟢 Added")
            self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding alfs: {str(e)}")
            self.update_status("add_alfs", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add alfs: {str(e)}")

    def add_geotiffs_to_qgis(self):
        """Add exported frame GeoTIFFs to QGIS as raster layers in a group."""
        config = self.get_config()
        gt_camera = config.get("geotiff_camera", "T")
        gt_suffix = "t" if gt_camera == "T" else "w"
        camera_label = "Thermal" if gt_camera == "T" else "RGB"
        geotiff_folder = os.path.join(config["target_folder"], f"geotiffs_{gt_suffix}")

        if not os.path.exists(geotiff_folder):
            QMessageBox.warning(
                self,
                "Missing GeoTIFFs",
                f"{camera_label} frame GeoTIFFs have not been exported.\n"
                f"Please run Step 7 (Export Frames as GeoTIFF) first."
            )
            return

        try:
            self.log(f"Adding {camera_label} frame GeoTIFFs to QGIS...")
            self.update_status("add_geotiffs", "🟡 Loading...")

            # Collect files and parse frame indices from filenames ({idx:08d}.tiff)
            geotiff_files = []
            for f in sorted(os.listdir(geotiff_folder)):
                if not f.lower().endswith(('.tif', '.tiff')):
                    continue
                stem = os.path.splitext(f)[0]
                try:
                    frame_idx = int(stem)
                except ValueError:
                    continue
                geotiff_files.append((frame_idx, os.path.join(geotiff_folder, f)))

            if not geotiff_files:
                QMessageBox.warning(self, "No GeoTIFFs", "No GeoTIFF files found.")
                self.update_status("add_geotiffs", "🔴 No files")
                return

            min_idx = geotiff_files[0][0]
            max_idx = geotiff_files[-1][0]
            dlg = FrameRangeDialog(
                self, min_idx, max_idx, len(geotiff_files), "frames"
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.update_status("add_geotiffs", "⚪ Cancelled")
                return

            start, end = dlg.selected_range()
            geotiff_files = [(idx, p) for idx, p in geotiff_files if start <= idx <= end]

            if not geotiff_files:
                QMessageBox.warning(self, "No GeoTIFFs", "No GeoTIFF files in selected range.")
                self.update_status("add_geotiffs", "🔴 No files")
                return

            if len(geotiff_files) > 100:
                reply = QMessageBox.question(
                    self,
                    "Many Files",
                    f"Selected range contains {len(geotiff_files)} GeoTIFF files.\n"
                    "Loading all may slow down QGIS.\n\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.update_status("add_geotiffs", "⚪ Cancelled")
                    return

            group = self._create_layer_group(f"BAMBI Frame GeoTIFFs ({camera_label})")
            loaded_count = 0

            for frame_idx, geotiff_path in geotiff_files:
                layer_name = f"Frame {frame_idx:08d}"
                layer = QgsRasterLayer(geotiff_path, layer_name)
                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer, False)
                    group.addLayer(layer)
                    loaded_count += 1
                else:
                    self.log(f"Warning: Could not load {os.path.basename(geotiff_path)}")

            self.log(f"Added {loaded_count} GeoTIFF layers to QGIS")
            self.update_status("add_geotiffs", "🟢 Added")
            group.setExpanded(False)
            self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding GeoTIFFs: {str(e)}")
            self.update_status("add_geotiffs", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add GeoTIFFs: {str(e)}")

    def add_flight_route_to_qgis(self):
        """Add flight route layers (polyline, points, and frame markers) to QGIS."""
        config = self.get_config()
        fr_camera = config.get("flight_route_camera", "T")
        fr_suffix = "t" if fr_camera == "T" else "w"
        camera_label = "Thermal" if fr_camera == "T" else "RGB"
        route_folder = os.path.join(config["target_folder"], f"flight_route_{fr_suffix}")

        if not os.path.exists(route_folder):
            QMessageBox.warning(
                self,
                "Missing Flight Route",
                f"{camera_label} flight route has not been generated.\n"
                f"Please run Step 2 (Generate Flight Route) first."
            )
            return

        try:
            self.log(f"Adding {camera_label} flight route to QGIS...")
            self.update_status("add_flight_route", "🟡 Loading...")

            # Create a group for the layers at top of layer tree
            group = self._create_layer_group(f"BAMBI Flight Route ({camera_label})")

            loaded_count = 0

            # Add flight route line
            route_line_file = os.path.join(route_folder, "flight_route.geojson")
            if os.path.exists(route_line_file):
                layer = QgsVectorLayer(route_line_file, "Flight Route", "ogr")
                if layer.isValid():
                    # Set a nice style for the line
                    symbol = QgsLineSymbol.createSimple({
                        'color': '#3388ff',
                        'width': '1.5',
                        'capstyle': 'round',
                        'joinstyle': 'round'
                    })
                    layer.renderer().setSymbol(symbol)

                    QgsProject.instance().addMapLayer(layer, False)
                    group.addLayer(layer)
                    loaded_count += 1
                    self.log("Added flight route line layer")
                else:
                    self.log("Warning: Could not load flight route line")

            # Add camera positions
            camera_points_file = os.path.join(route_folder, "camera_positions.geojson")
            if os.path.exists(camera_points_file):
                layer = QgsVectorLayer(camera_points_file, "Camera Positions", "ogr")
                if layer.isValid():
                    # Set a nice style for the points
                    symbol = QgsMarkerSymbol.createSimple({
                        'name': 'circle',
                        'color': '#ff7800',
                        'outline_color': '#ffffff',
                        'outline_width': '0.5',
                        'size': '3'
                    })
                    layer.renderer().setSymbol(symbol)

                    QgsProject.instance().addMapLayer(layer, False)
                    group.addLayer(layer)
                    loaded_count += 1
                    self.log("Added camera positions layer")
                else:
                    self.log("Warning: Could not load camera positions")

            # Add frame markers if enabled
            if self.frame_markers_enabled_check.isChecked():
                markers_layer = self._create_frame_markers_layer(
                    camera_points_file, config
                )
                if markers_layer and markers_layer.isValid():
                    # Persist the layer
                    markers_layer = self._persist_memory_layer(
                        markers_layer, "Frame_Markers", "flight_route_layers"
                    )
                    # Re-apply styling after persistence (GeoPackage doesn't store QGIS styling)
                    self._style_frame_markers_layer(markers_layer)
                    QgsProject.instance().addMapLayer(markers_layer, False)
                    group.addLayer(markers_layer)
                    loaded_count += 1
                    self.log("Added frame markers layer")

            # Add distance markers if enabled
            if self.distance_markers_enabled_check.isChecked():
                distance_layer = self._create_distance_markers_layer(
                    camera_points_file, config
                )
                if distance_layer and distance_layer.isValid():
                    # Persist the layer
                    distance_layer = self._persist_memory_layer(
                        distance_layer, "Distance_Markers", "flight_route_layers"
                    )
                    # Re-apply styling after persistence (GeoPackage doesn't store QGIS styling)
                    self._style_distance_markers_layer(distance_layer)
                    QgsProject.instance().addMapLayer(distance_layer, False)
                    group.addLayer(distance_layer)
                    loaded_count += 1
                    self.log("Added distance markers layer")

            # Add time markers if enabled
            if self.time_markers_enabled_check.isChecked():
                time_layer = self._create_time_markers_layer(
                    camera_points_file, config
                )
                if time_layer and time_layer.isValid():
                    time_layer = self._persist_memory_layer(
                        time_layer, "Time_Markers", "flight_route_layers"
                    )
                    self._style_time_markers_layer(time_layer)
                    QgsProject.instance().addMapLayer(time_layer, False)
                    group.addLayer(time_layer)
                    loaded_count += 1
                    self.log("Added time markers layer")

            # Add image labels if enabled
            if self.image_labels_enabled_check.isChecked():
                image_labels_layer = self._create_image_labels_layer(
                    camera_points_file, config
                )
                if image_labels_layer and image_labels_layer.isValid():
                    image_labels_layer = self._persist_memory_layer(
                        image_labels_layer, "Image_Labels", "flight_route_layers"
                    )
                    self._style_image_labels_layer(image_labels_layer)
                    QgsProject.instance().addMapLayer(image_labels_layer, False)
                    group.addLayer(image_labels_layer)
                    loaded_count += 1
                    self.log("Added image labels layer")

            if loaded_count == 0:
                QMessageBox.warning(self, "No Layers", "No flight route layers found.")
                self.update_status("add_flight_route", "🔴 No files")
                # Remove empty group
                QgsProject.instance().layerTreeRoot().removeChildNode(group)
                return

            self.log(f"Added {loaded_count} flight route layers to QGIS")
            self.update_status("add_flight_route", "🟢 Added")

            # Refresh canvas
            self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding flight route: {str(e)}")
            self.update_status("add_flight_route", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add flight route: {str(e)}")

    def _create_frame_markers_layer(self, camera_points_file: str, config: dict) -> Optional[QgsVectorLayer]:
        """Create a vector layer with frame markers at every N-th position.

        :param camera_points_file: Path to the camera positions GeoJSON file
        :param config: Configuration dictionary
        :return: QgsVectorLayer with frame markers, or None if creation fails
        """
        if not os.path.exists(camera_points_file):
            self.log("Warning: Camera positions file not found for frame markers")
            return None

        try:
            # Read camera positions
            with open(camera_points_file, 'r', encoding='utf-8') as f:
                camera_data = json.load(f)

            features = camera_data.get("features", [])
            if not features:
                self.log("Warning: No camera positions found for frame markers")
                return None

            # Get marker settings
            marker_interval = self.frame_marker_interval_spin.value()
            include_zero = self.frame_marker_include_zero_check.isChecked()

            # Get CRS from config
            target_epsg = config.get("target_epsg", 32633)

            # Create memory layer for markers
            layer_uri = f"Point?crs=EPSG:{target_epsg}&field=frame_idx:integer&field=label:string"
            markers_layer = QgsVectorLayer(layer_uri, "Frame Markers", "memory")

            if not markers_layer.isValid():
                self.log("Warning: Could not create frame markers layer")
                return None

            # Add features for every N-th frame
            marker_features = []
            for feature in features:
                frame_idx = feature.get("properties", {}).get("frame_idx", 0)

                # Check if this frame should have a marker
                should_add_marker = False
                if frame_idx == 0 and include_zero:
                    should_add_marker = True
                elif frame_idx > 0 and frame_idx % marker_interval == 0:
                    should_add_marker = True

                if should_add_marker:
                    coords = feature.get("geometry", {}).get("coordinates", [0, 0])
                    point = QgsPointXY(coords[0], coords[1])

                    qgs_feature = QgsFeature(markers_layer.fields())
                    qgs_feature.setGeometry(QgsGeometry.fromPointXY(point))
                    qgs_feature.setAttributes([frame_idx, str(frame_idx)])
                    marker_features.append(qgs_feature)

            if not marker_features:
                self.log("Warning: No frame markers to add (interval may be larger than total frames)")
                return None

            # Add features to layer
            markers_layer.dataProvider().addFeatures(marker_features)

            # Style the marker points
            symbol = QgsMarkerSymbol.createSimple({
                'name': 'diamond',
                'color': '#e31a1c',
                'outline_color': '#ffffff',
                'outline_width': '0.8',
                'size': '4'
            })
            markers_layer.renderer().setSymbol(symbol)

            # Configure labeling
            label_settings = QgsPalLayerSettings()
            label_settings.fieldName = 'label'
            label_settings.enabled = True

            # Text format
            text_format = QgsTextFormat()
            text_format.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            text_format.setSize(10)
            text_format.setColor(QColor('#000000'))

            # Add text buffer (halo) for better readability
            buffer_settings = QgsTextBufferSettings()
            buffer_settings.setEnabled(True)
            buffer_settings.setSize(1.5)
            buffer_settings.setColor(QColor('#ffffff'))
            text_format.setBuffer(buffer_settings)

            label_settings.setFormat(text_format)

            # Position labels above the marker
            label_settings.placement = QgsPalLayerSettings.Placement.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantPosition.QuadrantAbove

            # Apply labeling to layer
            labeling = QgsVectorLayerSimpleLabeling(label_settings)
            markers_layer.setLabelsEnabled(True)
            markers_layer.setLabeling(labeling)

            self.log(f"Created {len(marker_features)} frame markers (interval: {marker_interval})")
            return markers_layer

        except Exception as e:
            self.log(f"Error creating frame markers: {str(e)}")
            return None

    def _create_distance_markers_layer(self, camera_points_file: str, config: dict) -> Optional[QgsVectorLayer]:
        """Create a vector layer with markers at regular distance intervals along the flight path.

        Markers show both the cumulative distance and the associated frame number,
        e.g., "100m (243)" meaning 100 meters traveled, reached at frame 243.

        :param camera_points_file: Path to the camera positions GeoJSON file
        :param config: Configuration dictionary
        :return: QgsVectorLayer with distance markers, or None if creation fails
        """
        import math

        if not os.path.exists(camera_points_file):
            self.log("Warning: Camera positions file not found for distance markers")
            return None

        try:
            # Read camera positions
            with open(camera_points_file, 'r', encoding='utf-8') as f:
                camera_data = json.load(f)

            features = camera_data.get("features", [])
            if len(features) < 2:
                self.log("Warning: Need at least 2 camera positions for distance markers")
                return None

            # Get marker settings
            distance_interval = self.distance_marker_interval_spin.value()
            include_start = self.distance_marker_include_start_check.isChecked()

            # Get CRS from config
            target_epsg = config.get("target_epsg", 32633)

            # Calculate cumulative distances and find marker positions
            # Each entry: (cumulative_distance, frame_idx, x, y)
            positions = []
            cumulative_distance = 0.0

            for i, feature in enumerate(features):
                coords = feature.get("geometry", {}).get("coordinates", [0, 0])
                frame_idx = feature.get("properties", {}).get("frame_idx", i)
                x, y = coords[0], coords[1]

                if i == 0:
                    positions.append((0.0, frame_idx, x, y))
                else:
                    prev_coords = features[i - 1].get("geometry", {}).get("coordinates", [0, 0])
                    dx = x - prev_coords[0]
                    dy = y - prev_coords[1]
                    segment_distance = math.sqrt(dx * dx + dy * dy)
                    cumulative_distance += segment_distance
                    positions.append((cumulative_distance, frame_idx, x, y))

            total_distance = positions[-1][0]
            self.log(f"Total flight distance: {total_distance:.1f}m")

            # Create memory layer for distance markers
            layer_uri = (
                f"Point?crs=EPSG:{target_epsg}"
                "&field=distance:double&field=frame_idx:integer&field=label:string"
            )
            markers_layer = QgsVectorLayer(layer_uri, "Distance Markers", "memory")

            if not markers_layer.isValid():
                self.log("Warning: Could not create distance markers layer")
                return None

            # Find positions for each distance interval
            marker_features = []

            # Optionally include start marker
            if include_start:
                start_pos = positions[0]
                point = QgsPointXY(start_pos[2], start_pos[3])
                qgs_feature = QgsFeature(markers_layer.fields())
                qgs_feature.setGeometry(QgsGeometry.fromPointXY(point))
                label = f"0m ({start_pos[1]})"
                qgs_feature.setAttributes([0.0, start_pos[1], label])
                marker_features.append(qgs_feature)

            # Find positions at each distance interval
            target_distance = distance_interval

            while target_distance <= total_distance:
                # Find the segment containing this distance
                for i in range(1, len(positions)):
                    prev_dist, prev_frame, prev_x, prev_y = positions[i - 1]
                    curr_dist, curr_frame, curr_x, curr_y = positions[i]

                    if prev_dist <= target_distance <= curr_dist:
                        # Interpolate position within this segment
                        segment_length = curr_dist - prev_dist
                        if segment_length > 0:
                            # How far along the segment is our target distance
                            ratio = (target_distance - prev_dist) / segment_length
                            interp_x = prev_x + ratio * (curr_x - prev_x)
                            interp_y = prev_y + ratio * (curr_y - prev_y)
                        else:
                            interp_x, interp_y = curr_x, curr_y

                        # Use the frame index of the end of this segment
                        # (the first frame that exceeds the target distance)
                        point = QgsPointXY(interp_x, interp_y)
                        qgs_feature = QgsFeature(markers_layer.fields())
                        qgs_feature.setGeometry(QgsGeometry.fromPointXY(point))
                        label = f"{int(target_distance)}m ({curr_frame})"
                        qgs_feature.setAttributes([target_distance, curr_frame, label])
                        marker_features.append(qgs_feature)
                        break

                target_distance += distance_interval

            if not marker_features:
                self.log(
                    "Warning: No distance markers created "
                    f"(interval {distance_interval}m > total {total_distance:.1f}m)"
                )
                return None

            # Add features to layer
            markers_layer.dataProvider().addFeatures(marker_features)

            # Style the marker points (different color from frame markers)
            symbol = QgsMarkerSymbol.createSimple({
                'name': 'triangle',
                'color': '#2ca02c',  # Green color
                'outline_color': '#ffffff',
                'outline_width': '0.8',
                'size': '4.5'
            })
            markers_layer.renderer().setSymbol(symbol)

            # Configure labeling
            label_settings = QgsPalLayerSettings()
            label_settings.fieldName = 'label'
            label_settings.enabled = True

            # Text format
            text_format = QgsTextFormat()
            text_format.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            text_format.setSize(9)
            text_format.setColor(QColor('#1a5e1a'))  # Dark green

            # Add text buffer (halo) for better readability
            buffer_settings = QgsTextBufferSettings()
            buffer_settings.setEnabled(True)
            buffer_settings.setSize(1.5)
            buffer_settings.setColor(QColor('#ffffff'))
            text_format.setBuffer(buffer_settings)

            label_settings.setFormat(text_format)

            # Position labels above the marker
            label_settings.placement = QgsPalLayerSettings.Placement.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantPosition.QuadrantAbove

            # Apply labeling to layer
            labeling = QgsVectorLayerSimpleLabeling(label_settings)
            markers_layer.setLabelsEnabled(True)
            markers_layer.setLabeling(labeling)

            self.log(f"Created {len(marker_features)} distance markers (interval: {distance_interval}m)")
            return markers_layer

        except Exception as e:
            self.log(f"Error creating distance markers: {str(e)}")
            return None

    def _create_time_markers_layer(self, camera_points_file: str, config: dict) -> Optional[QgsVectorLayer]:
        """Create a vector layer with markers at regular time intervals along the flight path.

        Timestamps are read directly from the AirData CSV (first column = relative ms,
        datetime(utc) column = absolute UTC time).  Positions are interpolated along the
        AirData GPS track projected to the target CRS.

        Marker labels:
          Relative mode  →  "30s", "60s", …
          Absolute mode  →  "10:30:45", "10:31:15", …

        :param camera_points_file: Unused; kept for API consistency with the other marker methods.
        :param config: Configuration dictionary
        :return: QgsVectorLayer with time markers, or None if creation fails
        """
        import csv
        from datetime import datetime, timezone

        airdata_path = config.get("airdata_path", "")
        if not airdata_path or not os.path.exists(airdata_path):
            self.log("Warning: No AirData file configured – time markers require an AirData log")
            return None

        time_interval = self.time_marker_interval_spin.value()
        include_start = self.time_marker_include_start_check.isChecked()
        use_relative = self.time_marker_type_combo.currentIndex() == 0
        target_epsg = config.get("target_epsg", 32633)

        try:
            from pyproj import Transformer, CRS as PyprojCRS

            wgs84_proj4 = "+proj=longlat +datum=WGS84 +no_defs"
            if 32601 <= target_epsg <= 32660:
                zone = target_epsg - 32600
                utm_proj4 = f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"
            elif 32701 <= target_epsg <= 32760:
                zone = target_epsg - 32700
                utm_proj4 = f"+proj=utm +zone={zone} +south +datum=WGS84 +units=m +no_defs"
            else:
                from bambi_wildlife_detection.austria_dem_downloader import get_proj4_for_crs, WGS84_PROJ4
                wgs84_proj4 = WGS84_PROJ4
                utm_proj4 = get_proj4_for_crs(f"EPSG:{target_epsg}")

            transformer = Transformer.from_crs(
                PyprojCRS.from_proj4(wgs84_proj4),
                PyprojCRS.from_proj4(utm_proj4),
                always_xy=True
            )

            # ── Read AirData CSV ──────────────────────────────────────────────
            def _find_col(headers, *keywords):
                """Return the first header whose lower-case form contains all keywords."""
                for h in headers:
                    hl = h.lower().strip()
                    if all(k in hl for k in keywords):
                        return h
                return None

            def _parse_utc(s):
                if not s:
                    return None
                for fmt in (
                    "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M:%S.%f",
                ):
                    try:
                        return datetime.strptime(s.strip(), fmt).replace(
                            tzinfo=timezone.utc).timestamp()
                    except ValueError:
                        pass
                return None

            raw_rows = []   # (time_ms, utc_unix, lat, lon)

            with open(airdata_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []

                col_ms = _find_col(headers, "time", "millisecond")
                col_utc = _find_col(headers, "datetime", "utc") or _find_col(headers, "datetime")
                col_lat = _find_col(headers, "latitude") or _find_col(headers, "lat")
                col_lon = _find_col(headers, "longitude") or _find_col(headers, "lon")

                if col_lat is None or col_lon is None:
                    self.log("Warning: Could not find lat/lon columns in AirData CSV")
                    return None
                if col_ms is None and col_utc is None:
                    self.log("Warning: Could not find timestamp columns in AirData CSV")
                    return None

                for row in reader:
                    try:
                        lat = float(row[col_lat])
                        lon = float(row[col_lon])
                        if lat == 0.0 and lon == 0.0:
                            continue
                        ts_ms = float(row[col_ms]) if col_ms else None
                        ts_utc = _parse_utc(row[col_utc]) if col_utc else None
                        raw_rows.append((ts_ms, ts_utc, lat, lon))
                    except (ValueError, KeyError):
                        continue

            if len(raw_rows) < 2:
                self.log("Warning: Not enough valid GPS rows in AirData for time markers")
                return None

            # ── Project to target CRS ─────────────────────────────────────────
            # Build (elapsed_sec, unix_ts_or_None, x, y)
            first_ms = raw_rows[0][0]
            first_utc = raw_rows[0][1]

            positions = []
            for ts_ms, ts_utc, lat, lon in raw_rows:
                x, y = transformer.transform(lon, lat)
                if use_relative:
                    if ts_ms is None or first_ms is None:
                        continue
                    elapsed = (ts_ms - first_ms) / 1000.0
                    positions.append((elapsed, None, x, y))
                else:
                    if ts_utc is None or first_utc is None:
                        continue
                    elapsed = ts_utc - first_utc
                    positions.append((elapsed, ts_utc, x, y))

            if len(positions) < 2:
                self.log("Warning: Not enough timestamped rows in AirData for time markers")
                return None

            total_elapsed = positions[-1][0]
            if total_elapsed <= 0:
                self.log("Warning: AirData timestamps do not increase – cannot build time markers")
                return None

            self.log(f"AirData time range: {total_elapsed:.1f}s ({len(positions)} GPS points)")

            # ── Label helper ─────────────────────────────────────────────────
            def make_label(elapsed_sec, unix_ts):
                if use_relative:
                    return f"{elapsed_sec:.0f}s"
                if unix_ts is not None:
                    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%H:%M:%S")
                return f"{elapsed_sec:.0f}s"

            # ── Create memory layer ───────────────────────────────────────────
            layer_uri = (
                f"Point?crs=EPSG:{target_epsg}"
                "&field=elapsed_sec:double&field=label:string"
            )
            markers_layer = QgsVectorLayer(layer_uri, "Time Markers", "memory")
            if not markers_layer.isValid():
                self.log("Warning: Could not create time markers memory layer")
                return None

            marker_features = []

            if include_start:
                p = positions[0]
                feat = QgsFeature(markers_layer.fields())
                feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(p[2], p[3])))
                feat.setAttributes([0.0, make_label(0.0, p[1])])
                marker_features.append(feat)

            target_sec = float(time_interval)
            while target_sec <= total_elapsed:
                for i in range(1, len(positions)):
                    e_prev, ts_prev, xp, yp = positions[i - 1]
                    e_curr, ts_curr, xc, yc = positions[i]
                    if e_prev <= target_sec <= e_curr:
                        seg = e_curr - e_prev
                        ratio = (target_sec - e_prev) / seg if seg > 0 else 0.0
                        ix = xp + ratio * (xc - xp)
                        iy = yp + ratio * (yc - yp)
                        interp_ts = (
                            None if (ts_prev is None or ts_curr is None)
                            else ts_prev + ratio * (ts_curr - ts_prev)
                        )
                        feat = QgsFeature(markers_layer.fields())
                        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(ix, iy)))
                        feat.setAttributes([target_sec, make_label(target_sec, interp_ts)])
                        marker_features.append(feat)
                        break
                target_sec += time_interval

            if not marker_features:
                self.log(
                    "Warning: No time markers created "
                    f"(interval {time_interval}s > total {total_elapsed:.1f}s)")
                return None

            markers_layer.dataProvider().addFeatures(marker_features)
            self.log(
                f"Created {len(marker_features)} time markers "
                f"(interval: {time_interval}s, "
                f"{'relative' if use_relative else 'absolute'})"
            )
            return markers_layer

        except Exception as e:
            self.log(f"Error creating time markers: {str(e)}")
            return None

    def _style_time_markers_layer(self, layer: QgsVectorLayer):
        """Apply styling and labeling to the time markers layer."""
        try:
            symbol = QgsMarkerSymbol.createSimple({
                'name': 'diamond',
                'color': '#9b59b6',        # Purple
                'outline_color': '#ffffff',
                'outline_width': '0.8',
                'size': '4.5'
            })
            layer.renderer().setSymbol(symbol)

            label_settings = QgsPalLayerSettings()
            label_settings.fieldName = 'label'
            label_settings.enabled = True

            text_format = QgsTextFormat()
            text_format.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            text_format.setSize(9)
            text_format.setColor(QColor('#6c3483'))  # Dark purple

            buffer_settings = QgsTextBufferSettings()
            buffer_settings.setEnabled(True)
            buffer_settings.setSize(1.5)
            buffer_settings.setColor(QColor('#ffffff'))
            text_format.setBuffer(buffer_settings)

            label_settings.setFormat(text_format)
            label_settings.placement = QgsPalLayerSettings.Placement.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantPosition.QuadrantAbove

            layer.setLabelsEnabled(True)
            layer.setLabeling(QgsVectorLayerSimpleLabeling(label_settings))
            layer.triggerRepaint()
        except Exception as e:
            self.log(f"Warning: Could not style time markers layer: {str(e)}")

    def _create_image_labels_layer(self, camera_points_file: str, config: dict) -> Optional[QgsVectorLayer]:
        """Create a vector layer with image filename labels at every N-th camera position.

        :param camera_points_file: Path to the camera positions GeoJSON file
        :param config: Configuration dictionary
        :return: QgsVectorLayer with image labels, or None if creation fails
        """
        if not os.path.exists(camera_points_file):
            self.log("Warning: Camera positions file not found for image labels")
            return None

        try:
            with open(camera_points_file, 'r', encoding='utf-8') as f:
                camera_data = json.load(f)

            features = camera_data.get("features", [])
            if not features:
                self.log("Warning: No camera positions found for image labels")
                return None

            label_interval = self.image_label_interval_spin.value()
            target_epsg = config.get("target_epsg", 32633)

            layer_uri = (
                f"Point?crs=EPSG:{target_epsg}"
                "&field=frame_idx:integer&field=imagefile:string&field=label:string"
            )
            labels_layer = QgsVectorLayer(layer_uri, "Image Labels", "memory")

            if not labels_layer.isValid():
                self.log("Warning: Could not create image labels layer")
                return None

            label_features = []
            for feature in features:
                props = feature.get("properties", {})
                frame_idx = props.get("frame_idx", 0)

                if frame_idx % label_interval != 0:
                    continue

                coords = feature.get("geometry", {}).get("coordinates", [0, 0])
                imagefile = props.get("imagefile", "")
                label = os.path.basename(imagefile) if imagefile else str(frame_idx)

                point = QgsPointXY(coords[0], coords[1])
                qgs_feature = QgsFeature(labels_layer.fields())
                qgs_feature.setGeometry(QgsGeometry.fromPointXY(point))
                qgs_feature.setAttributes([frame_idx, imagefile, label])
                label_features.append(qgs_feature)

            if not label_features:
                self.log("Warning: No image labels to add (interval may be larger than total frames)")
                return None

            labels_layer.dataProvider().addFeatures(label_features)
            self.log(f"Created {len(label_features)} image labels (interval: {label_interval})")
            return labels_layer

        except Exception as e:
            self.log(f"Error creating image labels: {str(e)}")
            return None

    def _style_image_labels_layer(self, layer: QgsVectorLayer):
        """Apply styling and labeling to the image labels layer."""
        try:
            # Invisible marker — label text only
            symbol = QgsMarkerSymbol.createSimple({
                'name': 'circle',
                'color': '0,0,0,0',
                'outline_style': 'no',
                'size': '1'
            })
            layer.renderer().setSymbol(symbol)

            label_settings = QgsPalLayerSettings()
            label_settings.fieldName = 'label'
            label_settings.enabled = True

            text_format = QgsTextFormat()
            text_format.setFont(QFont("Arial", 8))
            text_format.setSize(8)
            text_format.setColor(QColor('#1a1a1a'))

            buffer_settings = QgsTextBufferSettings()
            buffer_settings.setEnabled(True)
            buffer_settings.setSize(1.0)
            buffer_settings.setColor(QColor('#ffffff'))
            text_format.setBuffer(buffer_settings)

            label_settings.setFormat(text_format)
            label_settings.placement = QgsPalLayerSettings.Placement.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantPosition.QuadrantBelow

            layer.setLabelsEnabled(True)
            layer.setLabeling(QgsVectorLayerSimpleLabeling(label_settings))
            layer.triggerRepaint()
        except Exception as e:
            self.log(f"Warning: Could not style image labels layer: {str(e)}")

    def _style_frame_markers_layer(self, layer: QgsVectorLayer):
        """Apply styling and labeling to frame markers layer.

        :param layer: The frame markers layer to style
        """
        try:
            # Style the marker points
            symbol = QgsMarkerSymbol.createSimple({
                'name': 'diamond',
                'color': '#e31a1c',
                'outline_color': '#ffffff',
                'outline_width': '0.8',
                'size': '4'
            })
            layer.renderer().setSymbol(symbol)

            # Configure labeling
            label_settings = QgsPalLayerSettings()
            label_settings.fieldName = 'label'
            label_settings.enabled = True

            # Text format
            text_format = QgsTextFormat()
            text_format.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            text_format.setSize(10)
            text_format.setColor(QColor('#000000'))

            # Add text buffer (halo) for better readability
            buffer_settings = QgsTextBufferSettings()
            buffer_settings.setEnabled(True)
            buffer_settings.setSize(1.5)
            buffer_settings.setColor(QColor('#ffffff'))
            text_format.setBuffer(buffer_settings)

            label_settings.setFormat(text_format)

            # Position labels above the marker
            label_settings.placement = QgsPalLayerSettings.Placement.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantPosition.QuadrantAbove

            # Apply labeling to layer
            labeling = QgsVectorLayerSimpleLabeling(label_settings)
            layer.setLabelsEnabled(True)
            layer.setLabeling(labeling)
        except Exception as e:
            self.log(f"Warning: Could not style frame markers layer: {str(e)}")

    def _style_distance_markers_layer(self, layer: QgsVectorLayer):
        """Apply styling and labeling to distance markers layer.

        :param layer: The distance markers layer to style
        """
        try:
            # Style the marker points (different color from frame markers)
            symbol = QgsMarkerSymbol.createSimple({
                'name': 'triangle',
                'color': '#2ca02c',  # Green color
                'outline_color': '#ffffff',
                'outline_width': '0.8',
                'size': '4.5'
            })
            layer.renderer().setSymbol(symbol)

            # Configure labeling
            label_settings = QgsPalLayerSettings()
            label_settings.fieldName = 'label'
            label_settings.enabled = True

            # Text format
            text_format = QgsTextFormat()
            text_format.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            text_format.setSize(9)
            text_format.setColor(QColor('#1a5e1a'))  # Dark green

            # Add text buffer (halo) for better readability
            buffer_settings = QgsTextBufferSettings()
            buffer_settings.setEnabled(True)
            buffer_settings.setSize(1.5)
            buffer_settings.setColor(QColor('#ffffff'))
            text_format.setBuffer(buffer_settings)

            label_settings.setFormat(text_format)

            # Position labels above the marker
            label_settings.placement = QgsPalLayerSettings.Placement.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantPosition.QuadrantAbove

            # Apply labeling to layer
            labeling = QgsVectorLayerSimpleLabeling(label_settings)
            layer.setLabelsEnabled(True)
            layer.setLabeling(labeling)
        except Exception as e:
            self.log(f"Warning: Could not style distance markers layer: {str(e)}")

    # =========================================================================
    # PROJECT CONFIGURATION PERSISTENCE
    # =========================================================================

    def _connect_project_signals(self):
        """Connect to QGIS project signals for automatic config persistence."""
        project = QgsProject.instance()

        # Save config before project is written (saved)
        project.writeProject.connect(self._on_project_write)

        # Load config when a project is read (opened)
        project.readProject.connect(self._on_project_read)

        # Handle project cleared (new project)
        project.cleared.connect(self._on_project_cleared)

    def disconnect_project_signals(self):
        """Disconnect from QGIS project signals. Call this when unloading the plugin."""
        try:
            project = QgsProject.instance()
            project.writeProject.disconnect(self._on_project_write)
            project.readProject.disconnect(self._on_project_read)
            project.cleared.disconnect(self._on_project_cleared)
        except Exception:  # nosec B110
            pass  # Ignore errors if signals weren't connected

    def _on_project_write(self, doc):
        """Called when project is about to be saved."""
        if not self._loading_config:
            self.save_config_to_project()

    def _on_project_read(self, doc):
        """Called when a project is opened."""
        self._loading_config = True
        try:
            self.load_config_from_project()
        finally:
            self._loading_config = False

    def _on_project_cleared(self):
        """Called when project is cleared (new project).

        Also fires when an existing project is opened (QGIS clears before
        reading), in which case readProject re-loads the saved config
        right after this reset.
        """
        self._loading_config = True
        try:
            self.reset_config_to_defaults()
        finally:
            self._loading_config = False

    def _snapshot_default_config(self):
        """Capture the pristine post-construction state of all config widgets.

        Must be called once right after setup_ui(), before any project
        config is loaded. reset_config_to_defaults() restores this state.
        """
        self._default_config = {}
        for name, widget in vars(self).items():
            if name == "log_text":
                continue  # keep log history across projects
            if isinstance(widget, QLineEdit):
                self._default_config[name] = widget.text()
            elif isinstance(widget, QCheckBox):
                self._default_config[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                self._default_config[name] = (widget.currentIndex(),
                                              widget.currentText())
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                self._default_config[name] = widget.value()
            elif isinstance(widget, QTextEdit):
                self._default_config[name] = widget.toPlainText()

    def reset_config_to_defaults(self):
        """Reset all configuration widgets to their default (startup) state."""
        for name, value in self._default_config.items():
            widget = getattr(self, name, None)
            if widget is None:
                continue
            if isinstance(widget, QLineEdit):
                widget.setText(value)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(value)
            elif isinstance(widget, QComboBox):
                index, text = value
                if widget.isEditable():
                    widget.setCurrentText(text)
                elif 0 <= index < widget.count():
                    widget.setCurrentIndex(index)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QTextEdit):
                widget.setPlainText(value)

        self._additional_corrections = []
        self._update_corrections_list_ui()

        self._thermal_vis_curve = None
        self._update_thermal_vis_curve_label()

        self.log("Configuration reset to defaults")

    def _config_widget_value(self, attr: str, role: str):
        """Read a config value from the bound widget (core.config_schema role)."""
        widget = getattr(self, attr)
        if role in ("line", "crs"):
            return widget.text()
        if role == "text":
            return widget.toPlainText()
        if role == "check":
            return widget.isChecked()
        if role == "spin":
            return widget.value()
        if role in ("combo_text", "combo_text_editable"):
            return widget.currentText()
        if role == "combo_index":
            return widget.currentIndex()
        raise ValueError(f"Unknown config role {role!r}")

    def _apply_config_value(self, attr: str, role: str, value) -> None:
        """Apply a loaded config value to the bound widget (config_schema role)."""
        widget = getattr(self, attr)
        if role == "line":
            widget.setText(value)
        elif role == "crs":
            widget.setText(value if value else "EPSG:32633")
        elif role == "text":
            widget.setPlainText(value)
        elif role == "check":
            widget.setChecked(value)
        elif role == "spin":
            widget.setValue(value)
        elif role == "combo_text":
            idx = widget.findText(value)
            if idx >= 0:
                widget.setCurrentIndex(idx)
        elif role == "combo_text_editable":
            if value:
                idx = widget.findText(value)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                else:
                    widget.setCurrentText(value)
        elif role == "combo_index":
            if 0 <= value < widget.count():
                widget.setCurrentIndex(value)
        else:
            raise ValueError(f"Unknown config role {role!r}")

    def save_config_to_project(self):
        """Save the configuration of the active flight (§10.2).

        The form values go to the flight's own ``project.gpkg``, so the folder
        is self-describing; the QGIS project keeps only the flight list and
        which one is active.
        """
        from .core import flights as flights_core
        from .core.config_schema import WIDGET_BINDINGS, save_config_entries
        project = QgsProject.instance()

        values = {key: self._config_widget_value(attr, role)
                  for key, (attr, role) in WIDGET_BINDINGS.items()}

        target_folder = self.target_folder_edit.text().strip()
        if target_folder and os.path.isdir(target_folder):
            stored = {}
            write = flights_core.config_writers(stored)
            save_config_entries(values, *write)
            try:
                flights_core.save_config(target_folder, stored)
            except Exception as exc:  # noqa: BLE001 — never lose a save
                self.log(f"Warning: could not save configuration to the "
                         f"store: {exc}")
        else:
            # No folder yet, so there is nowhere to put it — the QGIS project
            # carries it until one is chosen.
            save_config_entries(
                values,
                lambda k, v: project.writeEntry(PLUGIN_SCOPE, k, v),
                lambda k, v: project.writeEntryDouble(PLUGIN_SCOPE, k, v),
                lambda k, v: project.writeEntryBool(PLUGIN_SCOPE, k, v),
            )

        project.writeEntry(PLUGIN_SCOPE, "Flights/List",
                           json.dumps(self._flights()))
        project.writeEntry(PLUGIN_SCOPE, "Flights/Active",
                           str(getattr(self, "_active_flight_index", 0)))

        # Additional corrections are bound to a list widget, not a value
        # widget, so they are saved separately. The SAM3 API key is
        # intentionally never saved.
        corrections_data = []
        for i in range(self.additional_corrections_list.count()):
            item = self.additional_corrections_list.item(i)
            data = item.data(Qt.ItemDataRole.UserRole)
            if data:
                corrections_data.append(data)
        project.writeEntry(PLUGIN_SCOPE, "Correction/AdditionalCorrections",
                           json.dumps(corrections_data))

        # Thermal curve mapping is bound to a dialog, not a value widget,
        # so it is likewise saved separately (empty string = no curve).
        project.writeEntry(
            PLUGIN_SCOPE, "Input/ThermalVisCurve",
            json.dumps(self._thermal_vis_curve.to_dict())
            if self._thermal_vis_curve is not None else "")

        # Mark project as modified so user is prompted to save
        project.setDirty(True)

        self.log("Configuration saved to project")

    def load_config_from_project(self, restore_flights: bool = True):
        """Load the active flight's configuration (core.config_schema).

        *restore_flights* is False when switching or adding a flight: the list
        in memory is already newer than the one in the QGIS project, which is
        only written on save.
        """
        from .core.config_schema import WIDGET_BINDINGS, load_config_entries
        project = QgsProject.instance()

        # Which flight is active decides where the configuration is read from
        # (§10.2), so the list has to be settled first.
        if restore_flights:
            self._restore_flights(project)

        # Prefer the active flight's own store; fall back to the QGIS project,
        # which is where projects written before 6.0 keep everything.
        from .core import flights as flights_core

        flight = self._active_flight()
        folder = (flight or {}).get("target_folder", "")
        stored = flights_core.load_config(folder) if folder else {}
        migrated_from_project = False

        if stored:
            read_str, read_double, read_bool = flights_core.config_readers(stored)
        else:
            _test_value, has_config = project.readEntry(
                PLUGIN_SCOPE, "Input/TargetFolder", "")
            if not has_config:
                return  # No saved config anywhere
            migrated_from_project = bool(folder)

            def read_str(key: str, default: str = "") -> str:
                value, ok = project.readEntry(PLUGIN_SCOPE, key, default)
                return value if ok else default

            def read_double(key: str, default: float = 0.0) -> float:
                value, ok = project.readDoubleEntry(PLUGIN_SCOPE, key, default)
                return value if ok else default

            def read_bool(key: str, default: bool = False) -> bool:
                value, ok = project.readBoolEntry(PLUGIN_SCOPE, key, default)
                return value if ok else default

        for key, value in load_config_entries(read_str, read_double, read_bool):
            attr, role = WIDGET_BINDINGS[key]
            self._apply_config_value(attr, role, value)

        # Legacy migration: projects saved before the model path was split
        # per modality stored a single Detection/ModelPath (thermal model).
        legacy_model_path = read_str("Detection/ModelPath", "")
        if legacy_model_path and not self.thermal_model_path_edit.text():
            self.thermal_model_path_edit.setText(legacy_model_path)

        # A project written before 6.0 keeps its configuration in the .qgz;
        # move it into the flight's store once, so the folder becomes
        # self-describing from here on (§10.2).
        if migrated_from_project:
            self.save_config_to_project()
            self.log("Configuration moved into the flight's project.gpkg")

        # Additional corrections (bound to a list widget, loaded separately)
        corrections_json = read_str("Correction/AdditionalCorrections", "[]")
        try:
            corrections_data = json.loads(corrections_json)
            self._additional_corrections = []
            self._update_corrections_list_ui()
            for corr in corrections_data:
                self._add_correction_to_list(corr)
        except json.JSONDecodeError:
            pass

        # Thermal curve mapping (bound to a dialog, loaded separately)
        curve_json = read_str("Input/ThermalVisCurve", "")
        if curve_json:
            from .core.thermal_curve import ThermalCurve
            try:
                self._thermal_vis_curve = ThermalCurve.from_dict(
                    json.loads(curve_json))
            except (json.JSONDecodeError, TypeError, ValueError):
                self._thermal_vis_curve = None
        else:
            self._thermal_vis_curve = None
        self._update_thermal_vis_curve_label()

        self.log("Configuration loaded from project")

    def _flight_group(self, create: bool = True):
        """The layer group of the active flight, or the tree root.

        Layers are grouped per flight so it is obvious which outputs belong
        together (§10.2). Renaming a flight renames this group rather than
        leaving a stale one behind, which is why it is looked up by the current
        name every time instead of being cached.
        """
        from .core import flights

        root = QgsProject.instance().layerTreeRoot()
        flight = self._active_flight()
        if not flight:
            return root

        name = flights.group_name(flight)
        group = root.findGroup(name)
        if group is None and create:
            group = root.insertGroup(0, name)
        return group or root

    def _remove_layer_group(self, group_name: str) -> bool:
        """Remove a layer group and every layer in it from the project.

        Re-adding a group would otherwise duplicate it, and its layers hold
        their GeoPackages open — which on Windows blocks rewriting them.
        """
        root = self._flight_group(create=False)
        group = root.findGroup(group_name)
        if group is None:
            return False

        layer_ids = [node.layerId() for node in group.findLayers()
                     if node.layerId()]
        if layer_ids:
            QgsProject.instance().removeMapLayers(layer_ids)
        root.removeChildNode(group)
        return True

    def _create_layer_group(self, group_name: str, at_top: bool = True) -> QgsLayerTreeGroup:
        """Create a layer group, optionally at the top of the layer tree.

        :param group_name: Name for the new group
        :param at_top: If True, insert at top of layer tree; if False, append at bottom
        :return: The created QgsLayerTreeGroup
        """
        root = self._flight_group()
        group = QgsLayerTreeGroup(group_name)

        if at_top:
            root.insertChildNode(0, group)
        else:
            root.addChildNode(group)

        return group

    def _persist_memory_layer(self, mem_layer: QgsVectorLayer, file_name: str,
                              subfolder: str = "qgis_layers",
                              display_name: Optional[str] = None) -> QgsVectorLayer:
        """Save a memory layer to GeoPackage file and return the file-based layer.

        If no target folder is set, returns the original memory layer unchanged.

        :param mem_layer: The memory layer to persist
        :param file_name: Unique name for the GeoPackage file (without extension)
        :param subfolder: Subfolder within target folder for storing layers
        :param display_name: Display name for the layer in QGIS (defaults to mem_layer's name)
        :return: File-based QgsVectorLayer if saved successfully, otherwise original memory layer
        """
        from qgis.core import QgsVectorFileWriter, QgsCoordinateTransformContext

        # Use memory layer's name as display name if not specified
        if display_name is None:
            display_name = mem_layer.name()

        # Check if target folder is set
        target_folder = self.target_folder_edit.text().strip()
        if not target_folder or not os.path.isdir(target_folder):
            # No target folder, return memory layer as-is
            self.log(f"Warning: No target folder set, layer '{display_name}' will be temporary")
            return mem_layer

        try:
            # Create output folder
            gpkg_folder = os.path.join(target_folder, subfolder)
            os.makedirs(gpkg_folder, exist_ok=True)

            # Sanitize file name for filesystem
            safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in file_name)
            gpkg_path = os.path.join(gpkg_folder, f"{safe_name}.gpkg")

            # If file already exists, release any QGIS layers holding it open
            # before deleting — on Windows, open file handles cause WinError 32.
            exists = os.path.exists(gpkg_path)
            if exists:
                norm_path = os.path.normcase(os.path.abspath(gpkg_path))
                layers_to_remove = [
                    lid for lid, lyr in QgsProject.instance().mapLayers().items()
                    if isinstance(lyr, QgsVectorLayer) and  # noqa: W503, W504
                    os.path.normcase(os.path.abspath(lyr.source().split("|")[0])) == norm_path
                ]
                if layers_to_remove:
                    QgsProject.instance().removeMapLayers(layers_to_remove)
                try:
                    os.remove(gpkg_path)
                    exists = False
                except OSError as e:
                    # GDAL pools its datasets, so the handle can outlive the
                    # layer and Windows then refuses the delete (WinError 32).
                    # No need to delete: a GeoPackage is SQLite, so the layer
                    # inside it can be rewritten in place while a reader holds
                    # the file open.
                    self.log(f"Note: could not replace "
                             f"{os.path.basename(gpkg_path)} ({e}); "
                             "overwriting the layer inside it instead.")

            # Write to GeoPackage
            options = QgsVectorFileWriter.SaveVectorOptions()
            options.driverName = "GPKG"
            options.fileEncoding = "UTF-8"
            options.layerName = safe_name
            if exists:
                options.actionOnExistingFile = (
                    QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer)

            error = QgsVectorFileWriter.writeAsVectorFormatV3(
                mem_layer,
                gpkg_path,
                QgsCoordinateTransformContext(),
                options
            )

            if error[0] != QgsVectorFileWriter.WriterError.NoError:
                self.log(f"Warning: Could not save layer '{display_name}' to GeoPackage: {error[1]}")
                return mem_layer

            # Load the persisted layer with the display name
            persisted_layer = QgsVectorLayer(gpkg_path, display_name, "ogr")

            if not persisted_layer.isValid():
                self.log(f"Warning: Persisted layer '{display_name}' is not valid, using memory layer")
                return mem_layer

            return persisted_layer

        except Exception as e:
            self.log(f"Warning: Failed to persist layer '{display_name}': {str(e)}")
            return mem_layer
