# -*- coding: utf-8 -*-
"""
BAMBI Wildlife Detection - Dock Widget
=======================================

This module contains the main dock widget UI for the plugin.
"""

import os
import json
import math
import csv
from typing import Optional, Dict, Any, List
import sys

from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QFileDialog, QLabel, QProgressBar, QTextEdit, QComboBox,
    QCheckBox, QTabWidget, QMessageBox, QScrollArea, QSlider,
    QFrame, QListWidget, QListWidgetItem, QSizePolicy, QDialog,
    QDialogButtonBox, QGridLayout, QToolButton
)
from qgis.PyQt.QtGui import QFont, QColor
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsCoordinateReferenceSystem,
    QgsFeature, QgsGeometry, QgsPointXY, QgsField, QgsRasterLayer,
    QgsLineSymbol, QgsMarkerSymbol, QgsPalLayerSettings, QgsTextFormat,
    QgsVectorLayerSimpleLabeling, QgsTextBufferSettings, QgsLayerTreeGroup
)
from qgis.PyQt.QtCore import QVariant

from .bambi_processing import BambiProcessor, ProcessingWorker
from .austria_dem_downloader import DEMDownloadWorker, GeoTIFFConversionWorker
from .bambi_click_tool import BambiClickTool
from .bambi_feature_viewer import FeatureViewerDialog

# Plugin scope for project settings storage
PLUGIN_SCOPE = "BambiWildlifeDetection"

# ---------------------------------------------------------------------------
# Built-in calibration presets
# Add your calibration presets here as:
#   "Preset Name": { ... calibration JSON dict ... }
# ---------------------------------------------------------------------------
THERMAL_CALIBRATIONS: Dict[str, Any] = {
    "DJI M30T (T;Video)": {"ret": 7.245051860689403, "mtx": [[1520.3107832839369, 0.0, 626.3977414191858], [0.0, 1485.995024483995, 503.3510429805098], [0.0, 0.0, 1.0]], "dist": [[-0.3394678906712876, 0.17727664454650988, 0.0009308512048568154, 0.0004890617951054967, 0]]},
    "DJI M3T (T;Video)": {"ret": None, "mtx": [[762.1973876953125, 0.0, 315.58062744140625], [0.0, 745.9588012695312, 258.6109619140625], [0.0, 0.0, 1.0]], "dist": [-0.3735257089138031, 0.21459056437015533, -0.0011555751552805305, 0.0010070821736007929, -0.020840825513005257]},
}

RGB_CALIBRATIONS: Dict[str, Any] = {
    "DJI M30T (W;Video)": {"ret": None, "mtx": [[2888.178324915765, 0.0, 1929.0179499431672], [0.0, 2819.3160759551897, 1070.7804294784548], [0.0, 0.0, 1.0]], "dist": [[0.13853585738444427, -0.25508562557544706, 0.00020336601637138113, -0.0009057472383396885, 0.0]]},
    "DJI M3T (V;Video)": {"ret": None, "mtx": [[2985.574462890625, 0.0, 1933.8876953125], [0.0, 2916.530517578125, 1082.4605712890625], [0.0, 0.0, 1.0]], "dist": [0.1394435465335846, -0.26558583974838257, -0.0008494078647345304, -0.0012549880193546414, 0.0]},
}


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
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
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


class BambiDockWidget(QDockWidget):
    """Main dock widget for the BAMBI Wildlife Detection plugin."""

    def __init__(self, iface, parent=None):
        """Initialize the dock widget.

        :param iface: QGIS interface instance
        :param parent: Parent widget
        """
        super().__init__("Bambi - QGIS Integration", parent)
        self.iface = iface
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

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

        input_mode_row = QHBoxLayout()
        input_mode_row.setContentsMargins(0, 0, 0, 0)
        input_mode_row.addWidget(self.video_mode_check)
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

        self.thermal_srt_paths_edit = QLineEdit()
        self.thermal_srt_paths_edit.setPlaceholderText("Comma-separated paths to thermal SRT files (_T_)")
        thermal_srt_browse_btn = QPushButton("Browse...")
        thermal_srt_browse_btn.clicked.connect(self.browse_thermal_srts)
        thermal_srt_row = QHBoxLayout()
        thermal_srt_row.addWidget(self.thermal_srt_paths_edit)
        thermal_srt_row.addWidget(thermal_srt_browse_btn)
        thermal_layout.addRow("SRT Files:", thermal_srt_row)

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

        self.rgb_srt_paths_edit = QLineEdit()
        self.rgb_srt_paths_edit.setPlaceholderText("Comma-separated paths to RGB SRT files (_W_ or _V_)")
        rgb_srt_browse_btn = QPushButton("Browse...")
        rgb_srt_browse_btn.clicked.connect(self.browse_rgb_srts)
        rgb_srt_row = QHBoxLayout()
        rgb_srt_row.addWidget(self.rgb_srt_paths_edit)
        rgb_srt_row.addWidget(rgb_srt_browse_btn)
        rgb_layout.addRow("SRT Files:", rgb_srt_row)

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

        # Photo settings
        photo_settings_group = QGroupBox("Photo Settings")
        photo_settings_layout = QFormLayout(photo_settings_group)

        # Auto-detect checkbox
        self.photo_tz_auto_check = QCheckBox("Auto")
        self.photo_tz_auto_check.setChecked(True)
        self.photo_tz_auto_check.setToolTip(
            "Automatically determine the UTC offset by comparing average photo "
            "timestamps (EXIF) with average AirData isPhoto timestamps.")
        self.photo_tz_auto_check.stateChanged.connect(self._on_photo_tz_auto_changed)

        # Auto-offset result label (shown when Auto is checked)
        self.photo_tz_auto_result_label = QLabel()
        self.photo_tz_auto_result_label.setStyleSheet("color: grey; font-size: 10px;")
        self.photo_tz_auto_result_label.setVisible(True)

        auto_row = QHBoxLayout()
        auto_row.addWidget(self.photo_tz_auto_check)
        auto_row.addWidget(self.photo_tz_auto_result_label)
        photo_settings_layout.addRow("Offset:", auto_row)

        # Manual timezone widgets
        self.photo_timezone_combo = QComboBox()
        self.photo_timezone_combo.setEditable(True)
        self.photo_timezone_combo.setToolTip(
            "Timezone of the camera clock. The current UTC offset is shown on the right.")
        try:
            from zoneinfo import available_timezones
            tz_names = sorted(available_timezones())
        except ImportError:
            tz_names = [
                "UTC", "Europe/London", "Europe/Paris", "Europe/Berlin",
                "Europe/Vienna", "Europe/Rome", "Europe/Madrid",
                "Europe/Warsaw", "Europe/Helsinki", "Europe/Athens",
                "America/New_York", "America/Chicago", "America/Denver",
                "America/Los_Angeles", "America/Anchorage", "Pacific/Honolulu",
                "America/Sao_Paulo", "America/Buenos_Aires",
                "Asia/Dubai", "Asia/Kolkata", "Asia/Bangkok",
                "Asia/Shanghai", "Asia/Tokyo", "Australia/Sydney",
            ]
        self.photo_timezone_combo.addItems(tz_names)
        default_tz = "Europe/Vienna"
        default_idx = self.photo_timezone_combo.findText(default_tz)
        if default_idx >= 0:
            self.photo_timezone_combo.setCurrentIndex(default_idx)

        self.photo_timezone_offset_label = QLabel()
        self.photo_timezone_offset_label.setStyleSheet("color: grey; font-size: 10px;")
        self._update_timezone_offset_label()

        self.photo_tz_manual_widget = QWidget()
        self.photo_tz_manual_widget.setVisible(False)
        tz_row = QHBoxLayout(self.photo_tz_manual_widget)
        tz_row.setContentsMargins(0, 0, 0, 0)
        tz_row.addWidget(self.photo_timezone_combo)
        tz_row.addWidget(self.photo_timezone_offset_label)
        photo_settings_layout.addRow("Timezone:", self.photo_tz_manual_widget)

        self.photo_timezone_combo.currentTextChanged.connect(self._update_timezone_offset_label)
        photo_inputs_layout.addWidget(photo_settings_group)

        self.photo_tz_manual_hint = QLabel(
            'Timezone of camera, when photos were created not current timezone (consider daylight saving time)!'
        )
        self.photo_tz_manual_hint.setWordWrap(True)
        self.photo_tz_manual_hint.setVisible(False)
        photo_settings_layout.addWidget(self.photo_tz_manual_hint)

        input_layout.addWidget(self.photo_inputs_widget)

        self.video_mode_check.stateChanged.connect(self._on_input_mode_changed)

        # Common inputs
        common_group = QGroupBox("Flight Data")
        common_layout = QFormLayout(common_group)

        self.airdata_path_edit = QLineEdit()
        self.airdata_path_edit.setPlaceholderText("Path to AirData CSV file")
        airdata_browse_btn = QPushButton("Browse...")
        airdata_browse_btn.clicked.connect(self.browse_airdata)
        airdata_row = QHBoxLayout()
        airdata_row.addWidget(self.airdata_path_edit)
        airdata_row.addWidget(airdata_browse_btn)
        common_layout.addRow("AirData CSV:", airdata_row)
        airdata_label = QLabel(
            '<a href="https://www.airdata.com">https://www.airdata.com</a>'
        )
        airdata_label.setOpenExternalLinks(True)
        airdata_label.setWordWrap(True)
        airdata_label.setStyleSheet("color: blue; font-size: 10px;")
        common_layout.addWidget(airdata_label)

        self.correction_path_edit = QLineEdit()
        self.correction_path_edit.setPlaceholderText("Path to correction.json (auto-detected)")
        correction_browse_btn = QPushButton("Browse...")
        correction_browse_btn.clicked.connect(self.browse_correction)
        correction_row = QHBoxLayout()
        correction_row.addWidget(self.correction_path_edit)
        correction_row.addWidget(correction_browse_btn)
        common_layout.addRow("Correction:", correction_row)

        input_layout.addWidget(common_group)

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
            "Note, that this may take some time, since individual tiles are 10GB+ and it may be necessary to download multiple map tiles. "
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
            "Note, that this may take some time, since individual tiles are 10GB+ and it may be necessary to download multiple map tiles.\n"
            "Downloaded to \"<user profile>/.cache/austria_dem\""
        )
        download_info_label.setWordWrap(True)
        auto_download_layout.addRow("", download_info_label)

        self.geo_input_tabs.addTab(auto_download_tab, "Auto-Download")

        input_layout.addWidget(geo_group)

        # Output configuration
        output_group = QGroupBox("Output Configuration")
        output_layout = QFormLayout(output_group)

        self.target_folder_edit = QLineEdit()
        self.target_folder_edit.setPlaceholderText("Target folder for all outputs")
        self.target_folder_edit.editingFinished.connect(self._on_target_folder_changed)
        target_browse_btn = QPushButton("Browse...")
        target_browse_btn.clicked.connect(self.browse_target_folder)
        target_row = QHBoxLayout()
        target_row.addWidget(self.target_folder_edit)
        target_row.addWidget(target_browse_btn)
        output_layout.addRow("Target Folder:", target_row)

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

        input_layout.addWidget(output_group)
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

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Leave empty for default HuggingFace model")
        model_browse_btn = QPushButton("Browse...")
        model_browse_btn.clicked.connect(self.browse_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_path_edit)
        model_row.addWidget(model_browse_btn)
        detection_layout.addRow("Model Path:", model_row)
        detection_label = QLabel(
            "Note, that the default BAMBI model was trained on white-hotspot thermal data showing roe deer, red deer and wild boar with an AGL between 30 to 60 m. So the applicability is limited to that scope. Additionally, the model is based on the Ultralytics framework so the utilization follows their license."
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
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
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

        # ----- Sub-Tab 5: Orthomosaic -----
        ortho_tab = QWidget()
        ortho_tab_layout = QVBoxLayout(ortho_tab)
        config_sub_tabs.addTab(ortho_tab, "Orthomosaic/GeoTiff")

        ortho_group = QGroupBox("Orthomosaic/GeoTIFF Generation")
        ortho_layout = QFormLayout(ortho_group)

        self.ortho_resolution_spin = QDoubleSpinBox()
        self.ortho_resolution_spin.setRange(0.001, 1.0)
        self.ortho_resolution_spin.setSingleStep(0.01)
        self.ortho_resolution_spin.setValue(0.05)
        self.ortho_resolution_spin.setDecimals(3)
        self.ortho_resolution_spin.setSuffix(" m/px")
        self.ortho_resolution_spin.setToolTip("Ground resolution in meters per pixel")
        ortho_layout.addRow("Ground Resolution:", self.ortho_resolution_spin)

        self.blend_mode_combo = QComboBox()
        self.blend_mode_combo.addItems([
            "INTEGRAL - Average all overlapping",
            "FIRST - First pixel wins",
            "LAST - Last pixel wins",
            "CENTER - Center priority"
        ])
        self.blend_mode_combo.setCurrentIndex(0)
        self.blend_mode_combo.setToolTip("How to blend overlapping frames")
        ortho_layout.addRow("Blend Mode:", self.blend_mode_combo)

        frame_range_label = QLabel("Frame Range:")
        ortho_layout.addRow(frame_range_label)

        self.ortho_all_frames_check = QCheckBox("Use all frames")
        self.ortho_all_frames_check.setChecked(True)
        self.ortho_all_frames_check.stateChanged.connect(self.toggle_ortho_frame_range)
        ortho_layout.addRow("", self.ortho_all_frames_check)

        self.ortho_frame_range_widget = QWidget()
        frame_range_layout = QHBoxLayout(self.ortho_frame_range_widget)
        frame_range_layout.setContentsMargins(0, 0, 0, 0)

        self.ortho_start_frame_spin = QSpinBox()
        self.ortho_start_frame_spin.setRange(0, 999999)
        self.ortho_start_frame_spin.setValue(0)
        self.ortho_start_frame_spin.setToolTip("First frame to include (inclusive)")
        frame_range_layout.addWidget(QLabel("Start:"))
        frame_range_layout.addWidget(self.ortho_start_frame_spin)

        self.ortho_end_frame_spin = QSpinBox()
        self.ortho_end_frame_spin.setRange(0, 999999)
        self.ortho_end_frame_spin.setValue(999999)
        self.ortho_end_frame_spin.setToolTip("Last frame to include (inclusive)")
        frame_range_layout.addWidget(QLabel("End:"))
        frame_range_layout.addWidget(self.ortho_end_frame_spin)

        self.ortho_frame_range_widget.setEnabled(False)
        ortho_layout.addRow("", self.ortho_frame_range_widget)

        self.preview_frames_btn = QPushButton("Detect Frame Count")
        self.preview_frames_btn.clicked.connect(self.detect_frame_count)
        self.preview_frames_btn.setToolTip("Load poses.json to detect available frames")
        ortho_layout.addRow("", self.preview_frames_btn)

        self.frame_count_label = QLabel("Frame count: Unknown")
        ortho_layout.addRow("", self.frame_count_label)

        self.ortho_crop_check = QCheckBox()
        self.ortho_crop_check.setChecked(True)
        self.ortho_crop_check.setToolTip("Crop output to content area")
        ortho_layout.addRow("Crop to Content:", self.ortho_crop_check)

        self.ortho_overviews_check = QCheckBox()
        self.ortho_overviews_check.setChecked(True)
        self.ortho_overviews_check.setToolTip("Create overview pyramids for faster GIS viewing")
        ortho_layout.addRow("Create Overviews:", self.ortho_overviews_check)

        self.ortho_tile_size_spin = QSpinBox()
        self.ortho_tile_size_spin.setRange(1024, 16384)
        self.ortho_tile_size_spin.setValue(8192)
        self.ortho_tile_size_spin.setSingleStep(1024)
        self.ortho_tile_size_spin.setToolTip("Maximum tile size for processing large images")
        ortho_layout.addRow("Max Tile Size:", self.ortho_tile_size_spin)

        # Orthomosaic frame step (skip)
        self.ortho_frame_step_spin = QSpinBox()
        self.ortho_frame_step_spin.setRange(1, 100)
        self.ortho_frame_step_spin.setValue(1)
        self.ortho_frame_step_spin.setToolTip("Process every Nth frame (1 = all frames, 2 = every 2nd frame, etc.)")
        ortho_layout.addRow("Frame Step:", self.ortho_frame_step_spin)

        ortho_tab_layout.addWidget(ortho_group)
        ortho_tab_layout.addStretch()

        # ----- Sub-Tab 6: SAM3 Segmentation -----
        sam3_tab = QWidget()
        sam3_tab_layout = QVBoxLayout(sam3_tab)
        config_sub_tabs.addTab(sam3_tab, "SAM3 Segmentation")

        sam3_api_group = QGroupBox("Roboflow API Configuration")
        sam3_api_layout = QFormLayout(sam3_api_group)

        self.sam3_api_key_edit = QLineEdit()
        self.sam3_api_key_edit.setEchoMode(QLineEdit.Password)
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
        main_tabs.addTab(processing_tab, "Processing")

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

        # Step buttons
        steps_group = QGroupBox("Processing Steps")
        steps_btn_layout = QVBoxLayout(steps_group)

        # ----- Step 1a: Extract Thermal Frames -----
        step1a_row = QHBoxLayout()
        self.extract_thermal_btn = QPushButton("1a. Extract Thermal Frames")
        self.extract_thermal_btn.clicked.connect(self.run_extract_thermal_frames)
        self.extract_thermal_btn.setToolTip("Extract frames from thermal videos (_T_) to frames_t/")
        self.extract_thermal_status = QLabel("⚪ Not started")
        step1a_row.addWidget(self.extract_thermal_btn)
        step1a_row.addWidget(self.extract_thermal_status)
        steps_btn_layout.addLayout(step1a_row)

        # ----- Step 1b: Extract RGB Frames -----
        step1b_row = QHBoxLayout()
        self.extract_rgb_btn = QPushButton("1b. Extract RGB Frames")
        self.extract_rgb_btn.clicked.connect(self.run_extract_rgb_frames)
        self.extract_rgb_btn.setToolTip("Extract frames from RGB videos (_W_/_V_) to frames_w/")
        self.extract_rgb_status = QLabel("⚪ Not started")
        step1b_row.addWidget(self.extract_rgb_btn)
        step1b_row.addWidget(self.extract_rgb_status)
        steps_btn_layout.addLayout(step1b_row)

        correction_info_label = QLabel(
            "If not already done: Use the correction tooling to determine positional and orientation errors after frame extraction."
        )
        correction_info_label.setWordWrap(True)
        correction_info_label.setStyleSheet("color: gray; font-size: 10px;")
        steps_btn_layout.addWidget(correction_info_label)

        # Separator after extraction
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        steps_btn_layout.addWidget(separator1)

        # ----- Step 2: Generate Flight Route -----
        step2_row = QHBoxLayout()
        self.flight_route_btn = QPushButton("2. Generate Flight Route")
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

        # ----- Step 3: Detect Animals -----
        step3_row = QHBoxLayout()
        self.detect_btn = QPushButton("3. Detect Animals")
        self.detect_btn.clicked.connect(self.run_detection)
        self.detection_camera_combo = QComboBox()
        self.detection_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.detection_camera_combo.setFixedWidth(100)
        self.detection_camera_combo.setToolTip("Select camera source for frames")
        self.detect_status = QLabel("⚪ Not started")
        step3_row.addWidget(self.detect_btn)
        step3_row.addWidget(self.detection_camera_combo)
        step3_row.addWidget(self.detect_status)
        steps_btn_layout.addLayout(step3_row)

        # ----- Step 4: Geo-Reference Detections -----
        step4_row = QHBoxLayout()
        self.georef_btn = QPushButton("4. Geo-Reference Detections")
        self.georef_btn.clicked.connect(self.run_georeference)
        self.georef_status = QLabel("⚪ Not started")
        step4_row.addWidget(self.georef_btn)
        step4_row.addWidget(self.georef_status)
        steps_btn_layout.addLayout(step4_row)

        # -> Add Detections to QGIS
        add_detections_row = QHBoxLayout()
        self.add_frame_detections_btn = QPushButton("   → Add Detections to QGIS")
        self.add_frame_detections_btn.clicked.connect(self.add_frame_detections_to_qgis)
        self.add_frame_detections_btn.setToolTip("Add geo-referenced bounding boxes as layers (one per frame)")
        self.frame_detections_status = QLabel("⚪")
        add_detections_row.addWidget(self.add_frame_detections_btn)
        add_detections_row.addWidget(self.frame_detections_status)
        steps_btn_layout.addLayout(add_detections_row)

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
        steps_btn_layout.addLayout(perp_calc_row)

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
        steps_btn_layout.addLayout(perp_add_row)

        # ----- Step 5: Track Animals -----
        step5_row = QHBoxLayout()
        self.track_btn = QPushButton("5. Track Animals")
        self.track_btn.clicked.connect(self.run_tracking)
        self.track_status = QLabel("⚪ Not started")
        step5_row.addWidget(self.track_btn)
        step5_row.addWidget(self.track_status)
        steps_btn_layout.addLayout(step5_row)

        # -> Add Tracks to QGIS
        add_tracks_row = QHBoxLayout()
        self.add_layers_btn = QPushButton("   → Add Tracks to QGIS")
        self.add_layers_btn.clicked.connect(self.add_tracks_to_qgis)
        self.layers_status = QLabel("⚪")
        add_tracks_row.addWidget(self.add_layers_btn)
        add_tracks_row.addWidget(self.layers_status)
        steps_btn_layout.addLayout(add_tracks_row)

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
        steps_btn_layout.addLayout(track_perp_calc_row)

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
        steps_btn_layout.addLayout(track_perp_add_row)

        # ----- Step 6: Calculate Field of View -----
        step6_row = QHBoxLayout()
        self.calculate_fov_btn = QPushButton("6. Calculate Field of View")
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

        # ----- Step 7: Generate Orthomosaic -----
        step7_row = QHBoxLayout()
        self.ortho_btn = QPushButton("7. Generate Orthomosaic")
        self.ortho_btn.clicked.connect(self.run_orthomosaic)
        self.ortho_camera_combo = QComboBox()
        self.ortho_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.ortho_camera_combo.setFixedWidth(100)
        self.ortho_camera_combo.setToolTip("Select camera source for frames and poses")
        self.ortho_status = QLabel("⚪ Not started")
        step7_row.addWidget(self.ortho_btn)
        step7_row.addWidget(self.ortho_camera_combo)
        step7_row.addWidget(self.ortho_status)
        steps_btn_layout.addLayout(step7_row)

        # -> Add Orthomosaic to QGIS
        add_ortho_row = QHBoxLayout()
        self.add_ortho_btn = QPushButton("   → Add Orthomosaic to QGIS")
        self.add_ortho_btn.clicked.connect(self.add_orthomosaic_to_qgis)
        self.add_ortho_status = QLabel("⚪")
        add_ortho_row.addWidget(self.add_ortho_btn)
        add_ortho_row.addWidget(self.add_ortho_status)
        steps_btn_layout.addLayout(add_ortho_row)

        # ----- Step 8: Export Frames as GeoTIFF -----
        step8_row = QHBoxLayout()
        self.export_geotiffs_btn = QPushButton("8. Export Frames as GeoTIFF")
        self.export_geotiffs_btn.clicked.connect(self.run_export_geotiffs)
        self.geotiff_camera_combo = QComboBox()
        self.geotiff_camera_combo.addItems(["T - Thermal", "W - RGB"])
        self.geotiff_camera_combo.setFixedWidth(100)
        self.geotiff_camera_combo.setToolTip("Select camera source for frames and poses")
        self.export_geotiffs_status = QLabel("⚪ Not started")
        step8_row.addWidget(self.export_geotiffs_btn)
        step8_row.addWidget(self.geotiff_camera_combo)
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
        # ----- Step 9: Run SAM3 Segmentation -----
        step9_row = QHBoxLayout()
        self.sam3_segment_btn = QPushButton("9. Run SAM3 Segmentation")
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
        steps_btn_layout.addLayout(step9_row)

        # ----- Step 10: Geo-Reference Segmentation -----
        step10_row = QHBoxLayout()
        self.sam3_georef_btn = QPushButton("10. Geo-Reference Segmentation")
        self.sam3_georef_btn.clicked.connect(self.run_sam3_georeference)
        self.sam3_georef_btn.setToolTip("Convert pixel segmentation masks to world coordinates")
        self.sam3_georef_status = QLabel("⚪ Not started")
        step10_row.addWidget(self.sam3_georef_btn)
        step10_row.addWidget(self.sam3_georef_status)
        steps_btn_layout.addLayout(step10_row)

        # -> Add SAM3 Segmentation to QGIS
        add_sam3_row = QHBoxLayout()
        self.add_sam3_btn = QPushButton("   → Add Segmentation to QGIS")
        self.add_sam3_btn.clicked.connect(self.add_sam3_to_qgis)
        self.add_sam3_btn.setToolTip("Add geo-referenced segmentation masks as QGIS layers")
        self.add_sam3_status = QLabel("⚪")
        add_sam3_row.addWidget(self.add_sam3_btn)
        add_sam3_row.addWidget(self.add_sam3_status)
        steps_btn_layout.addLayout(add_sam3_row)

        processing_layout.addWidget(steps_group)

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

        processing_layout.addLayout(progress_layout)

        # Refresh Status button
        self.refresh_status_btn = QPushButton("🔄 Refresh Status")
        self.refresh_status_btn.setToolTip("Check for existing outputs and QGIS layers to update status indicators")
        self.refresh_status_btn.clicked.connect(self._refresh_all_statuses)
        processing_layout.addWidget(self.refresh_status_btn)

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

        processing_layout.addWidget(log_group)
        processing_layout.addStretch()

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
        from .tracker_manager import get_tracker_manager

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
                f"No configurable parameters available for this tracker.\n\n"
                f"The tracker uses its default settings."
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
        return {
            # Input mode
            "input_mode": "video" if video_mode else "photo",

            # Video inputs
            "thermal_video_paths": [p.strip() for p in self.thermal_video_paths_edit.text().split(",") if p.strip()],
            "thermal_srt_paths": [p.strip() for p in self.thermal_srt_paths_edit.text().split(",") if p.strip()],
            "thermal_calibration_path": (
                self.thermal_calibration_path_edit.text()
                if self.thermal_calib_preset_combo.currentIndex() == 0 else ""
            ),
            "thermal_calibration_data": (
                THERMAL_CALIBRATIONS.get(self.thermal_calib_preset_combo.currentText())
                if self.thermal_calib_preset_combo.currentIndex() > 0 else None
            ),
            "rgb_video_paths": [p.strip() for p in self.rgb_video_paths_edit.text().split(",") if p.strip()],
            "rgb_srt_paths": [p.strip() for p in self.rgb_srt_paths_edit.text().split(",") if p.strip()],
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
            "photo_timezone_offset": self._get_photo_timezone_offset(),

            # Common inputs
            "airdata_path": self.airdata_path_edit.text(),
            "dem_path": self.dem_path_edit.text(),
            "correction_path": self.correction_path_edit.text(),

            # Output
            "target_folder": self.target_folder_edit.text(),
            "target_epsg": epsg,

            # Detection
            "model_path": self.model_path_edit.text() or None,
            "min_confidence": self.confidence_spin.value(),
            "detect_use_all_frames": self.detect_all_frames_check.isChecked() if hasattr(self,
                                                                                         'detect_all_frames_check') else True,
            "detect_start_frame": self.detect_start_frame_spin.value() if hasattr(self,
                                                                                  'detect_start_frame_spin') else 0,
            "detect_end_frame": self.detect_end_frame_spin.value() if hasattr(self,
                                                                              'detect_end_frame_spin') else 999999,
            "detect_sample_rate": self.detect_sample_rate_spin.value() if hasattr(self,
                                                                                  'detect_sample_rate_spin') else 1,

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

            # Orthomosaic
            "ortho_ground_resolution": self.ortho_resolution_spin.value(),
            "ortho_dem_metadata_path": self.dem_metadata_path_edit.text() or None,
            "ortho_blend_mode": self.blend_mode_combo.currentText().split(" - ")[0].lower(),
            "ortho_use_all_frames": self.ortho_all_frames_check.isChecked(),
            "ortho_start_frame": self.ortho_start_frame_spin.value() if not self.ortho_all_frames_check.isChecked() else None,
            "ortho_end_frame": self.ortho_end_frame_spin.value() if not self.ortho_all_frames_check.isChecked() else None,
            "ortho_crop_to_content": self.ortho_crop_check.isChecked(),
            "ortho_create_overviews": self.ortho_overviews_check.isChecked(),
            "ortho_max_tile_size": self.ortho_tile_size_spin.value(),
            "ortho_frame_step": self.ortho_frame_step_spin.value() if hasattr(self, 'ortho_frame_step_spin') else 1,

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
            "sam3_use_all_frames": self.sam3_all_frames_check.isChecked() if hasattr(self,
                                                                                     'sam3_all_frames_check') else True,
            "sam3_start_frame": self.sam3_start_frame_spin.value() if hasattr(self, 'sam3_start_frame_spin') else 0,
            "sam3_end_frame": self.sam3_end_frame_spin.value() if hasattr(self, 'sam3_end_frame_spin') else 999999,
            "sam3_step": self.sam3_step_spin.value() if hasattr(self, 'sam3_step_spin') else 1,

            # Flight Route options
            "filter_gps_origin": self.filter_gps_origin_check.isChecked() if hasattr(self, 'filter_gps_origin_check') else True,

            # Camera selections for processing steps
            "flight_route_camera": "T" if self.flight_route_camera_combo.currentIndex() == 0 else "W",
            "detection_camera": "T" if self.detection_camera_combo.currentIndex() == 0 else "W",
            "fov_camera": "T" if self.fov_camera_combo.currentIndex() == 0 else "W",
            "ortho_camera": "T" if self.ortho_camera_combo.currentIndex() == 0 else "W",
            "geotiff_camera": "T" if self.geotiff_camera_combo.currentIndex() == 0 else "W",
            "sam3_camera": "T" if self.sam3_camera_combo.currentIndex() == 0 else "W",
        }

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
                f"Please provide the following required inputs:\n\n• " + "\n• ".join(missing)
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

            # Auto-populate thermal SRT paths
            srts = [f.replace(".MP4", ".SRT").replace(".mp4", ".srt") for f in thermal_files]
            existing_srts = [s for s in srts if os.path.exists(s)]
            if existing_srts:
                self.thermal_srt_paths_edit.setText(", ".join(existing_srts))
                self.log(f"Auto-detected {len(existing_srts)} thermal SRT file(s)")

            # Auto-detect T_calib.json (only when using custom file mode)
            if (self.thermal_calib_preset_combo.currentIndex() == 0
                    and not self.thermal_calibration_path_edit.text()):
                t_calib_path = os.path.join(video_folder, "T_calib.json")
                if os.path.exists(t_calib_path):
                    self.thermal_calibration_path_edit.setText(t_calib_path)
                    self.log(f"Auto-detected thermal calibration: T_calib.json")

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

            # Auto-populate RGB SRT paths
            srts = [f.replace(".MP4", ".SRT").replace(".mp4", ".srt") for f in rgb_files]
            existing_srts = [s for s in srts if os.path.exists(s)]
            if existing_srts:
                self.rgb_srt_paths_edit.setText(", ".join(existing_srts))
                self.log(f"Auto-detected {len(existing_srts)} RGB SRT file(s)")

            # Auto-detect W_calib.json (only when using custom file mode)
            if (self.rgb_calib_preset_combo.currentIndex() == 0
                    and not self.rgb_calibration_path_edit.text()):
                w_calib_path = os.path.join(video_folder, "W_calib.json")
                if os.path.exists(w_calib_path):
                    self.rgb_calibration_path_edit.setText(w_calib_path)
                    self.log(f"Auto-detected RGB calibration: W_calib.json")

            # Auto-detect common files if not already set
            self._auto_detect_common_files(video_folder)

    def _auto_detect_common_files(self, video_folder: str):
        """Auto-detect common input files from video folder.

        :param video_folder: Folder containing video files
        """
        # Get list of all files in folder for auto-detection
        try:
            folder_files = os.listdir(video_folder)
        except Exception as e:
            self.log(f"Warning: Could not list folder contents: {e}")
            folder_files = []

        # Auto-detect AirData CSV (first CSV in folder)
        if not self.airdata_path_edit.text():
            csv_files = [f for f in folder_files if f.lower().endswith('.csv')]
            if csv_files:
                csv_path = os.path.join(video_folder, csv_files[0])
                self.airdata_path_edit.setText(csv_path)
                self.log(f"Auto-detected AirData CSV: {csv_files[0]}")

        # Auto-detect DEM GLTF/GLB (first GLTF or GLB in folder)
        if not self.dem_path_edit.text():
            dem_files = [f for f in folder_files if f.lower().endswith(('.gltf', '.glb'))]
            if dem_files:
                dem_path = os.path.join(video_folder, dem_files[0])
                self.dem_path_edit.setText(dem_path)
                self.log(f"Auto-detected DEM: {dem_files[0]}")

                # Also auto-detect DEM metadata JSON (same name with .json suffix)
                dem_base = os.path.splitext(dem_files[0])[0]
                json_name = dem_base + ".json"
                json_path = os.path.join(video_folder, json_name)
                if os.path.exists(json_path):
                    self.dem_metadata_path_edit.setText(json_path)
                    self.log(f"Auto-detected DEM metadata: {json_name}")
                else:
                    # Try common naming patterns
                    for suffix in ["_mesh.json", "_dem.json", "_metadata.json"]:
                        alt_path = os.path.join(video_folder, dem_base + suffix)
                        if os.path.exists(alt_path):
                            self.dem_metadata_path_edit.setText(alt_path)
                            self.log(f"Auto-detected DEM metadata: {dem_base + suffix}")
                            break

        # Auto-detect thermal calibration if not set (only in custom file mode)
        if (self.thermal_calib_preset_combo.currentIndex() == 0
                and not self.thermal_calibration_path_edit.text()):
            t_calib_path = os.path.join(video_folder, "T_calib.json")
            if os.path.exists(t_calib_path):
                self.thermal_calibration_path_edit.setText(t_calib_path)
                self.log(f"Auto-detected thermal calibration: T_calib.json")

        # Auto-detect RGB calibration if not set (only in custom file mode)
        if (self.rgb_calib_preset_combo.currentIndex() == 0
                and not self.rgb_calibration_path_edit.text()):
            w_calib_path = os.path.join(video_folder, "W_calib.json")
            if os.path.exists(w_calib_path):
                self.rgb_calibration_path_edit.setText(w_calib_path)
                self.log(f"Auto-detected RGB calibration: W_calib.json")

        # Auto-detect correction.json and load values
        if not self.correction_path_edit.text():
            correction_path = os.path.join(video_folder, "correction.json")
            if os.path.exists(correction_path):
                self.correction_path_edit.setText(correction_path)
                self.load_correction_values(correction_path)

        # Auto-set target folder to "qgis" subfolder
        if not self.target_folder_edit.text():
            qgis_folder = os.path.join(video_folder, "qgis")
            self.target_folder_edit.setText(qgis_folder)
            self.log(f"Auto-set target folder: qgis/")
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
        try:
            import datetime
            from zoneinfo import ZoneInfo
            now = datetime.datetime.now(ZoneInfo(tz_name))
            return now.utcoffset().total_seconds() / 3600
        except Exception:
            try:
                import datetime
                from dateutil import tz as dateutil_tz
                zone = dateutil_tz.gettz(tz_name)
                if zone is None:
                    return None
                now = datetime.datetime.now(zone)
                return now.utcoffset().total_seconds() / 3600
            except Exception:
                return None

    def _update_timezone_offset_label(self):
        """Update the UTC offset label next to the timezone combo."""
        tz_name = self.photo_timezone_combo.currentText().strip()
        offset = self._compute_timezone_offset(tz_name)
        if offset is None:
            self.photo_timezone_offset_label.setText("(unknown timezone)")
        else:
            sign = "+" if offset >= 0 else ""
            self.photo_timezone_offset_label.setText(f"UTC{sign}{offset:g}h")

    def _on_photo_tz_auto_changed(self, state: int):
        """Toggle between auto-detected and manual timezone offset."""
        auto = bool(state)
        self.photo_tz_manual_widget.setVisible(not auto)
        self.photo_tz_manual_hint.setVisible(not auto)
        self.photo_tz_auto_result_label.setVisible(auto)
        if auto:
            self._refresh_auto_timezone_offset()

    def _refresh_auto_timezone_offset(self):
        """Recompute the auto-detected offset and update the result label."""
        offset = self._compute_auto_timezone_offset()
        if offset is not None:
            sign = "+" if offset >= 0 else ""
            self.photo_tz_auto_result_label.setText(
                f"Detected offset: UTC{sign}{offset:g}h")
        else:
            self.photo_tz_auto_result_label.setText(
                "(could not detect — check photo dir & AirData)")

    def _get_photo_timezone_offset(self) -> float:
        """Return the timezone offset in hours, auto-detected or manual."""
        if self.photo_tz_auto_check.isChecked():
            offset = self._compute_auto_timezone_offset()
            if offset is not None:
                return offset
            self.log("Warning: Auto timezone detection failed, falling back to manual timezone")
        return self._compute_timezone_offset(
            self.photo_timezone_combo.currentText().strip()) or 0.0

    def _compute_auto_timezone_offset(self) -> Optional[float]:
        """Determine the UTC offset by comparing photo EXIF hours with AirData isPhoto hours.

        Reads all EXIF ``DateTimeOriginal`` timestamps from the thermal (or RGB)
        photo directory and all ``datetime(utc)`` values from AirData rows where
        ``isPhoto`` is truthy.  The offset is ``mean(photo hours) - mean(airdata hours)``
        rounded to the nearest integer.
        """
        import csv
        import glob as glob_mod
        from datetime import datetime

        # ── Collect photo timestamps (EXIF) ──────────────────────────────
        photo_dir = (self.thermal_photo_dir_edit.text().strip()
                     or self.rgb_photo_dir_edit.text().strip())
        if not photo_dir or not os.path.isdir(photo_dir):
            return None

        photo_hours: list = []
        _IMG_EXTS = ("*.jpg", "*.jpeg", "*.tiff", "*.tif", "*.png",
                     "*.JPG", "*.JPEG", "*.TIFF", "*.TIF", "*.PNG")
        image_paths: list = []
        for ext in _IMG_EXTS:
            image_paths.extend(glob_mod.glob(os.path.join(photo_dir, ext)))
        if not image_paths:
            return None

        try:
            from PIL import Image
        except ImportError:
            return None

        for p in image_paths:
            try:
                with Image.open(p) as img:
                    exif = img._getexif()
                    if exif is None:
                        continue
                    # Tag 36867 = DateTimeOriginal
                    dt_str = exif.get(36867)
                    if not dt_str:
                        continue
                    dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                    photo_hours.append(dt.hour + dt.minute / 60.0 + dt.second / 3600.0)
            except Exception:
                continue

        if not photo_hours:
            return None

        # ── Collect AirData isPhoto timestamps (UTC) ─────────────────────
        airdata_path = self.airdata_path_edit.text().strip()
        if not airdata_path or not os.path.exists(airdata_path):
            return None

        airdata_hours: list = []
        try:
            with open(airdata_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []

                # Find isPhoto column (case-insensitive)
                is_photo_col = None
                for h in headers:
                    if h.strip().lower() == "isphoto":
                        is_photo_col = h
                        break
                if is_photo_col is None:
                    return None

                # Find datetime(utc) column
                datetime_col = None
                for h in headers:
                    hl = h.lower().strip()
                    if "datetime" in hl and "utc" in hl:
                        datetime_col = h
                        break
                if datetime_col is None:
                    # Fall back to any datetime column
                    for h in headers:
                        if "datetime" in h.lower().strip():
                            datetime_col = h
                            break
                if datetime_col is None:
                    return None

                for row in reader:
                    val = row.get(is_photo_col, "").strip()
                    if not val or val == "0" or val.lower() == "false":
                        continue
                    dt_str = row.get(datetime_col, "").strip()
                    if not dt_str:
                        continue
                    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                                "%Y-%m-%d %H:%M:%S.%f"):
                        try:
                            dt = datetime.strptime(dt_str, fmt)
                            airdata_hours.append(
                                dt.hour + dt.minute / 60.0 + dt.second / 3600.0)
                            break
                        except ValueError:
                            continue
        except Exception:
            return None

        if not airdata_hours:
            return None

        avg_photo = sum(photo_hours) / len(photo_hours)
        avg_airdata = sum(airdata_hours) / len(airdata_hours)
        offset = round(avg_photo - avg_airdata)
        return float(offset)

    def _on_input_mode_changed(self, state: int):
        """Toggle between video and photo input panels."""
        video_mode = bool(state)
        self.video_inputs_widget.setVisible(video_mode)
        self.photo_inputs_widget.setVisible(not video_mode)

    def _show_input_mode_info(self):
        """Show an info popup describing the currently active input mode."""
        if self.video_mode_check.isChecked():
            title = "Video Mode"
            text = (
                "<b>Video Mode</b> is active.<br><br>"
                "The plugin processes continuous drone video recordings together with "
                "SRT subtitle files, which provide per-frame timestamps used to match "
                "each frame to a high-precision GPS position from the AirData flight log.<br><br>"
                "Practical tips: To ensure high-quality and consistent video recordings, configure your drone mission with stable flight parameters and a fixed camera setup. Plan the flight at a constant altitude between 30 m and 60 m above ground, depending on the terrain and desired coverage, and maintain a steady speed of 3–7 m/s throughout the mission. "
                "Although the exact heading is not critical, the drone’s orientation should remain constant for the entire flight, and yaw rotations should be avoided. A practical approach is to configure each waypoint so that the drone faces north, ensuring a stable and repeatable camera perspective. "
                "Set the gimbal pitch to −90° so the camera is pointing straight down (nadir). This provides a consistent top-down view and simplifies later processing of the video data. "
                "Finally, start video recording at the first waypoint and stop recording at the last waypoint to capture the full survey area in one continuous sequence while avoiding unnecessary footage. "
            )
        else:
            title = "Photo Mode"
            text = (
                "<b>Photo Mode</b> is active.<br><br>"
                "The plugin processes a series of still images captured during a mapping "
                "flight. GPS positions and global orientation are matched to images by comparing image EXIF "
                "timestamps with the AirData flight log.<br><br>"
                "Note: In photo mode, images are associated with the AirData file in a fixed sequential order for cases with same timestamps due to temporal ambiguities caused by second-level precision and the lack of sub-second information. Consequently, the complete dataset must always be provided. If individual images are missing, georeferencing will fail."
            )
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def _show_processing_info(self):
        """Show an info popup describing the processing pipeline."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Processing Pipeline")
        msg.setIcon(QMessageBox.Information)
        msg.setText(
            "<b>Processing Pipeline</b><br><br>"
            "Execute the steps below in order. Each step builds on the outputs of the "
            "previous one.<br><br>"
            "<b>1a / 1b — Extract Frames</b><br>"
            "Decodes and undistorts thermal and RGB frames; matches GPS positions from "
            "the AirData log via SRT timestamps (video) or EXIF timestamps (photo).<br><br>"
            "<b>2 — Generate Flight Route</b><br>"
            "Creates a GPS flight-path line layer and per-frame camera-position points "
            "from the extracted pose data.<br><br>"
            "<b>3 — Detect Animals</b><br>"
            "Runs YOLO-based detection on every extracted frame. The thermal model is "
            "downloaded automatically on first use.<br><br>"
            "<b>4 — Geo-Reference Detections</b><br>"
            "Projects pixel-space bounding boxes to UTM coordinates by ray-casting "
            "against the DEM mesh.<br><br>"
            "<b>→ Calculate Perpendicular</b><br>"
            "Measures the perpendicular distance from each geo-referenced detection "
            "to the flight route line — useful for transect-based surveys.<br><br>"
            "<b>5 — Track Animals</b><br>"
            "Links detections across frames into continuous tracks using the selected "
            "tracking backend.<br><br>"
            "<b>6 — Calculate Field of View</b><br>"
            "Computes per-frame camera footprint polygons on the ground using the DEM.<br><br>"
            "<b>7 — Generate Orthomosaic</b><br>"
            "Projects all frames onto the DEM surface and blends them into a "
            "georeferenced GeoTIFF mosaic.<br><br>"
            "<b>8 — Export Frames as GeoTIFF</b><br>"
            "Exports individual frames as separate georeferenced GeoTIFFs.<br><br>"
            "<b>9 / 10 — Object Segmentation</b><br>"
            "Segments detected objects using Roboflow SAM3 and projects masks to "
            "world coordinates.<br><br>"

            "The BAMBI plugin is not intended to process multiple flights within the same QGIS project (and the same output folder), since the result files in the output folder will be replaced after additional runs. If you want to combine multiple results, it is recommended to run processing independently with (1) different output folders (don't foregt to add all layers to QGIS before changing the output folder, otherwise you will have to change it back again!) or (2) group all layers of interest, export them as 'Layer Definition File' and import it again to QGIS."
        )
        msg.exec_()

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
            f"Download DEM from Austrian BEV service?\n\n"
            f"AirData CSV: {os.path.basename(airdata_path)}\n"
            f"Padding: {padding} m\n"
            f"Output CRS: {output_crs}\n"
            f"Output folder: {output_folder}\n\n"
            f"This will download the Austria-wide 1m ALS-DTM dataset.\n"
            f"Large areas may take several minutes.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply != QMessageBox.Yes:
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
                f"DEM downloaded successfully!\n\n"
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
            import rasterio, re
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
                        except Exception:
                            pass
        except Exception:
            pass
        input_crs_label = source_crs_override if source_crs_override else file_crs_label

        # Confirm conversion
        crs_info = f"\nInput CRS: {input_crs_label}"
        crs_info += f"\nOutput CRS: {output_crs}" if output_crs else "\n(Using original CRS)"
        reply = QMessageBox.question(
            self, "Convert GeoTIFF",
            f"Convert GeoTIFF to mesh?\n\n"
            f"Input: {os.path.basename(geotiff_path)}\n"
            f"Output folder: {output_folder}\n"
            f"Simplification: {simplify_factor}x{crs_info}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply != QMessageBox.Yes:
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
                f"GeoTIFF converted successfully!\n\n"
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
        """
        Check if the given CRS string is a valid UTM CRS.

        Valid UTM EPSG codes:
        - Northern Hemisphere: EPSG:32601 to EPSG:32660 (zones 1-60)
        - Southern Hemisphere: EPSG:32701 to EPSG:32760 (zones 1-60)

        :param crs_text: CRS string like "EPSG:32633" or "32633"
        :return: True if valid UTM CRS
        """
        try:
            epsg = self._parse_epsg_from_text(crs_text)
            if epsg is None:
                return False

            # Check if in UTM range
            is_northern = 32601 <= epsg <= 32660
            is_southern = 32701 <= epsg <= 32760

            return is_northern or is_southern
        except:
            return False

    def _parse_epsg_from_text(self, crs_text: str) -> Optional[int]:
        """
        Parse EPSG code from text input.

        Handles formats like:
        - "EPSG:32633"
        - "epsg:32633"
        - "32633"

        :param crs_text: CRS string
        :return: EPSG code as integer, or None if invalid
        """
        try:
            crs_text = crs_text.strip().upper()

            if crs_text.startswith("EPSG:"):
                code_str = crs_text[5:].strip()
            else:
                code_str = crs_text

            return int(code_str)
        except (ValueError, AttributeError):
            return None

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
                        source = f"DEM metadata"
            except Exception:
                pass

        # Try 2: AirData CSV GPS coordinates
        if detected_crs is None:
            airdata_path = self.airdata_path_edit.text()
            if airdata_path and os.path.exists(airdata_path):
                detected_crs = self._detect_utm_from_airdata(airdata_path)
                if detected_crs:
                    source = f"AirData GPS"

        # Apply result silently (just log, no message boxes)
        if detected_crs:
            self.target_crs_edit.setText(detected_crs)
            self.log(f"Auto-detected CRS: {detected_crs} (from {source})")

    def _detect_utm_from_airdata(self, csv_path: str) -> Optional[str]:
        """
        Detect appropriate UTM zone from GPS coordinates in AirData CSV.

        Uses the first valid GPS position to determine the UTM zone.

        :param csv_path: Path to AirData CSV file
        :return: CRS string like "EPSG:32633" or None
        """
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                # Find latitude/longitude columns
                lat_col = None
                lon_col = None

                fieldnames = reader.fieldnames
                if fieldnames is None:
                    return None

                for col in fieldnames:
                    col_lower = col.lower().strip()
                    if lat_col is None and any(x in col_lower for x in ['latitude', 'lat']):
                        lat_col = col
                    if lon_col is None and any(x in col_lower for x in ['longitude', 'lon', 'lng']):
                        lon_col = col

                if lat_col is None or lon_col is None:
                    self.log(f"Could not find lat/lon columns in {os.path.basename(csv_path)}")
                    return None

                # Get first valid GPS position
                for row in reader:
                    try:
                        lat = float(row[lat_col])
                        lon = float(row[lon_col])

                        # Skip invalid values
                        if lat == 0 and lon == 0:
                            continue
                        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                            continue

                        # Calculate UTM zone from longitude
                        # UTM zones are 6 degrees wide, zone 1 starts at -180°
                        utm_zone = int((lon + 180) / 6) + 1

                        # Clamp to valid range 1-60
                        utm_zone = max(1, min(60, utm_zone))

                        # Determine hemisphere
                        if lat >= 0:
                            epsg = 32600 + utm_zone  # Northern hemisphere
                        else:
                            epsg = 32700 + utm_zone  # Southern hemisphere

                        self.log(f"Detected UTM zone {utm_zone}{'N' if lat >= 0 else 'S'} "
                                 f"from GPS: ({lat:.4f}, {lon:.4f})")

                        return f"EPSG:{epsg}"

                    except (ValueError, KeyError):
                        continue

                self.log(f"No valid GPS coordinates found in {os.path.basename(csv_path)}")
                return None

        except Exception as e:
            self.log(f"Error reading AirData CSV for CRS detection: {e}")
            return None

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
            item.setData(Qt.UserRole, i)  # Store index
            self.additional_corrections_list.addItem(item)

    def _add_additional_correction(self):
        """Show dialog to add a new frame-range specific correction."""
        dialog = CorrectionRangeDialog(self, use_degrees=self._is_using_degrees())
        if dialog.exec_() == QDialog.Accepted:
            correction_data = dialog.get_correction_data()
            self._add_correction_to_list(correction_data)
            self.log(f"Added correction for frames {correction_data['start']}-{correction_data['end']}")

    def _edit_additional_correction(self, item: QListWidgetItem):
        """Edit an additional correction when double-clicked.

        :param item: The list item that was double-clicked
        """
        index = item.data(Qt.UserRole)
        if index is not None and 0 <= index < len(self._additional_corrections):
            correction_data = self._additional_corrections[index]
            dialog = CorrectionRangeDialog(self, correction_data, use_degrees=self._is_using_degrees())
            if dialog.exec_() == QDialog.Accepted:
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

        index = selected_items[0].data(Qt.UserRole)
        if index is not None and 0 <= index < len(self._additional_corrections):
            removed = self._additional_corrections.pop(index)
            self._update_corrections_list_ui()
            self.log(f"Removed correction for frames {removed.get('start', 0)}-{removed.get('end', 0)}")

    @staticmethod
    def get_correction_for_frame(frame_idx: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get the appropriate correction factors for a given frame index.

        This method checks if there's a frame-range specific correction that
        applies to the given frame index. If found, returns that correction;
        otherwise returns the default correction.

        :param frame_idx: The frame index to get correction for
        :param config: The configuration dictionary containing correction factors
        :return: Dictionary with 'translation' and 'rotation' keys
        """
        # Get default correction
        default_correction = {
            "translation": config.get("translation", {"x": 0, "y": 0, "z": 0}),
            "rotation": config.get("rotation", {"x": 0, "y": 0, "z": 0})
        }

        # Check additional corrections
        additional = config.get("additional_corrections", [])
        for add_corr in additional:
            start = add_corr.get("start", 0)
            end = add_corr.get("end", float('inf'))
            if start <= frame_idx <= end:
                return {
                    "translation": add_corr.get("translation", {"x": 0, "y": 0, "z": 0}),
                    "rotation": add_corr.get("rotation", {"x": 0, "y": 0, "z": 0})
                }

        return default_correction

    def browse_target_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Target Folder")
        if folder:
            self.target_folder_edit.setText(folder)
            self._check_existing_outputs(folder)

    def _check_existing_outputs(self, target_folder: str):
        """Check for existing output subfolders and update status labels accordingly.

        This method checks if the target folder and its expected subfolders exist,
        and marks the corresponding processing steps as 'Completed' if outputs are found.

        :param target_folder: Path to the target folder
        """
        if not target_folder or not os.path.isdir(target_folder):
            return

        # Reset all folder-based statuses before re-checking so that deleted
        # outputs are correctly reflected as "Not started" after a refresh.
        for step in ("extract_thermal_frames", "extract_rgb_frames", "detection",
                     "georeference", "tracking", "calculate_fov", "flight_route",
                     "orthomosaic", "export_geotiffs", "sam3_segmentation",
                     "sam3_georeference"):
            self.update_status(step, "⚪ Not started")

        completed_count = 0

        # Check for thermal frame extraction
        frames_t_path = os.path.join(target_folder, "frames_t")
        poses_t_path = os.path.join(target_folder, "poses_t.json")

        if os.path.isdir(frames_t_path) and os.listdir(frames_t_path) and os.path.isfile(poses_t_path):
            self.update_status("extract_thermal_frames", "🟢 Completed")
            completed_count += 1

        # Check for RGB frame extraction
        frames_w_path = os.path.join(target_folder, "frames_w")
        poses_w_path = os.path.join(target_folder, "poses_w.json")

        if os.path.isdir(frames_w_path) and os.listdir(frames_w_path) and os.path.isfile(poses_w_path):
            self.update_status("extract_rgb_frames", "🟢 Completed")
            completed_count += 1

        # Define the mapping between subfolders and their corresponding status updates
        # Format: (subfolder_name, status_step_key, additional_check_file)
        # additional_check_file is optional - if specified, the file must also exist
        folder_status_mapping = [
            ("flight_route", "flight_route", None),
            ("detections", "detection", None),
            ("georeferenced", "georeference", None),
            ("tracks", "tracking", None),
            ("fov", "calculate_fov", None),
            ("orthomosaic", "orthomosaic", None),
            ("geotiffs", "export_geotiffs", None),
            ("segmentation", "sam3_segmentation", "segmentation_pixel.json"),
            ("segmentation", "sam3_georeference", "segmentation_georef.json"),
        ]

        for subfolder, status_key, check_file in folder_status_mapping:
            subfolder_path = os.path.join(target_folder, subfolder)

            # Check if subfolder exists
            if os.path.isdir(subfolder_path):
                # If there's an additional file to check, verify it exists
                if check_file:
                    # Check file can be in subfolder or in target folder
                    check_path_subfolder = os.path.join(subfolder_path, check_file)
                    check_path_target = os.path.join(target_folder, check_file)

                    if not os.path.isfile(check_path_subfolder) and not os.path.isfile(check_path_target):
                        continue

                # Check if subfolder has any content (not empty)
                if os.listdir(subfolder_path):
                    self.update_status(status_key, "🟢 Completed")
                    completed_count += 1

        if completed_count > 0:
            self.log(f"Detected {completed_count} completed processing step(s) in target folder")

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
                     "add_fov", "add_merged_fov", "add_ortho", "add_geotiffs",
                     "add_sam3"):
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

        # Define mapping: (check_type, check_name, status_key)
        # check_type: "group" to check layer groups, "layer" to check layer names
        layer_status_mapping = [
            ("group", "BAMBI Flight Route", "add_flight_route"),
            ("group", "BAMBI Frame Detections", "add_frame_detections"),
            ("group", "BAMBI Wildlife Tracks", "add_layers"),
            ("group", "BAMBI FoV Polygons", "add_fov"),
            ("layer", "BAMBI FoV Coverage (Merged)", "add_merged_fov"),
            ("layer", "BAMBI Orthomosaic", "add_ortho"),
            ("group", "BAMBI Frame GeoTIFFs", "add_geotiffs"),
        ]

        added_count = 0

        for check_type, check_name, status_key in layer_status_mapping:
            if check_type == "group":
                if check_name in existing_groups:
                    self.update_status(status_key, "🟢 Added")
                    added_count += 1
            elif check_type == "layer":
                if check_name in existing_layers:
                    self.update_status(status_key, "🟢 Added")
                    added_count += 1

        if added_count > 0:
            self.log(f"Detected {added_count} existing BAMBI layer(s) in QGIS project")

    def _on_target_folder_changed(self):
        """Handle target folder path change from manual text editing."""
        folder = self.target_folder_edit.text().strip()
        if folder and os.path.isdir(folder):
            self._check_existing_outputs(folder)

    def open_correction_wizard(self) -> None:
        """Open the correction calibration wizard as a modal dialog."""
        from .bambi_correction_wizard import BambiCorrectionWizard
        config = self.get_config()
        wizard = BambiCorrectionWizard(self.iface, config, parent=self)
        wizard.correctionFileSaved.connect(self.load_correction_values)
        wizard.exec_()

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

    def browse_model(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "", "Model Files (*.pt *.onnx)")
        if file:
            self.model_path_edit.setText(file)

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

    def toggle_ortho_frame_range(self, state):
        """Toggle the frame range controls based on checkbox state."""
        self.ortho_frame_range_widget.setEnabled(not state)

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
            self.sam3_api_key_edit.setEchoMode(QLineEdit.Normal)
        else:
            self.sam3_api_key_edit.setEchoMode(QLineEdit.Password)

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

        camera = config.get("ortho_camera", "T")
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
                # Orthomosaic
                self.ortho_start_frame_spin.setRange(0, frame_count - 1)
                self.ortho_end_frame_spin.setRange(0, frame_count - 1)
                self.ortho_end_frame_spin.setValue(frame_count - 1)

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
    def run_extract_thermal_frames(self):
        """Run frame extraction for thermal modality."""
        config = self.get_config()

        if config["input_mode"] == "photo":
            # Photo mode validation
            thermal_calib_ok = (
                config.get("thermal_photo_calibration_data") is not None
                or config.get("thermal_photo_calibration_path")
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
                config.get("thermal_calibration_data") is not None
                or config.get("thermal_calibration_path")
            )
            if not (config.get("thermal_video_paths") and
                    config.get("thermal_srt_paths") and
                    thermal_calib_ok):
                QMessageBox.warning(
                    self,
                    "Missing Thermal Inputs",
                    "Please provide thermal video inputs:\n\n"
                    "• Thermal video files\n"
                    "• Thermal SRT files\n"
                    "• Thermal calibration (preset or custom file)"
                )
                return

        # Validate common inputs
        if not self.validate_inputs(["airdata_path", "target_folder"]):
            return

        self.start_worker("extract_thermal_frames")

    def run_extract_rgb_frames(self):
        """Run frame extraction for RGB modality."""
        config = self.get_config()

        if config["input_mode"] == "photo":
            # Photo mode validation
            rgb_calib_ok = (
                config.get("rgb_photo_calibration_data") is not None
                or config.get("rgb_photo_calibration_path")
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
                config.get("rgb_calibration_data") is not None
                or config.get("rgb_calibration_path")
            )
            if not (config.get("rgb_video_paths") and
                    config.get("rgb_srt_paths") and
                    rgb_calib_ok):
                QMessageBox.warning(
                    self,
                    "Missing RGB Inputs",
                    "Please provide RGB video inputs:\n\n"
                    "• RGB video files\n"
                    "• RGB SRT files\n"
                    "• RGB calibration (preset or custom file)"
                )
                return

        # Validate common inputs
        if not self.validate_inputs(["airdata_path", "target_folder"]):
            return

        self.start_worker("extract_rgb_frames")

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
                f"Please run Step 1{'a' if camera == 'T' else 'b'} first."
            )
            return

        self.start_worker("detection")

    def run_georeference(self):
        """Run geo-referencing step."""
        config = self.get_config()

        # Check if detections exist
        detections_folder = os.path.join(config["target_folder"], "detections")

        if not os.path.exists(detections_folder):
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                "Detection has not been completed.\nPlease run Step 2 first."
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
                f"Please run Step 1{'a' if camera == 'T' else 'b'} first."
            )
            return

        # Check DEM path
        if not self.validate_inputs(["dem_path"]):
            return

        self.start_worker("calculate_fov")

    def run_tracking(self):
        """Run tracking step."""
        config = self.get_config()

        # Check if georeferenced detections exist
        georef_folder = os.path.join(config["target_folder"], "georeferenced")

        if not os.path.exists(georef_folder):
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                "Geo-referencing has not been completed.\nPlease run Step 3 first."
            )
            return

        self.start_worker("tracking")

    def run_orthomosaic(self):
        """Run orthomosaic generation step."""
        config = self.get_config()
        camera = config.get("ortho_camera", "T")

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
                f"Please run Step 1{'a' if camera == 'T' else 'b'} first."
            )
            return

        # Check DEM path
        if not self.validate_inputs(["dem_path"]):
            return

        self.start_worker("orthomosaic")

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
                f"Please run Step 1{'a' if camera == 'T' else 'b'} first."
            )
            return

        # Check DEM path
        if not self.validate_inputs(["dem_path"]):
            return

        self.start_worker("export_geotiffs")

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
                f"Please run Step 1{'a' if camera == 'T' else 'b'} first."
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

        # Check if pixel segmentation exists
        segmentation_file = os.path.join(config["target_folder"], "segmentation", "segmentation_pixel.json")

        if not os.path.exists(segmentation_file):
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                "SAM3 segmentation has not been completed.\nPlease run Step 9 first."
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
        segmentation_folder = os.path.join(config["target_folder"], "segmentation")
        georef_file = os.path.join(segmentation_folder, "segmentation_georef.json")

        if not os.path.exists(georef_file):
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                "SAM3 geo-referencing has not been completed.\nPlease run Step 10 first."
            )
            return

        try:
            self.log("Adding SAM3 segmentation to QGIS...")
            self.update_status("add_sam3", "🟡 Loading...")

            # Load geo-referenced results
            with open(georef_file, 'r', encoding='utf-8') as f:
                georef_results = json.load(f)

            if not georef_results:
                QMessageBox.warning(self, "No Results", "No geo-referenced segmentation found.")
                self.update_status("add_sam3", "🔴 No data")
                return

            # Get target CRS
            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{config['target_epsg']}")

            # Collect all unique prompts for consistent coloring
            all_prompts = set()
            for frame_result in georef_results:
                for prompt_data in frame_result.get('prompts', []):
                    all_prompts.add(prompt_data.get('prompt', 'unknown'))

            # Generate colors for prompts
            prompt_colors = {}
            colors = [
                (255, 0, 0),  # Red
                (0, 150, 0),  # Green
                (0, 100, 255),  # Blue
                (255, 165, 0),  # Orange
                (128, 0, 128),  # Purple
                (0, 200, 200),  # Cyan
                (255, 105, 180),  # Pink
                (139, 69, 19),  # Brown
            ]
            for idx, prompt in enumerate(sorted(all_prompts)):
                prompt_colors[prompt] = colors[idx % len(colors)]

            # Check if there are many frames - warn user
            num_frames = len(georef_results)
            if num_frames > 50:
                reply = QMessageBox.question(
                    self,
                    "Many Frames",
                    f"Found {num_frames} frames with segmentation. Creating individual layer groups "
                    f"for each may slow down QGIS.\n\nContinue with individual frame groups?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply == QMessageBox.No:
                    self.update_status("add_sam3", "⚪ Cancelled")
                    return

            # Create main group at top of layer tree
            main_group = self._create_layer_group("SAM3 Segmentation")

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
        except Exception:
            pass

    def run_flight_route(self):
        """Run flight route generation step."""
        config = self.get_config()
        camera = config.get("flight_route_camera", "T")

        # Check if poses exist for selected camera
        target_folder = config["target_folder"]
        poses_file = os.path.join(target_folder, f"poses_{'t' if camera == 'T' else 'w'}.json")

        if not os.path.exists(poses_file):
            camera_name = "Thermal" if camera == "T" else "RGB"
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                f"{camera_name} frame extraction has not been completed.\n"
                f"Please run Step 1{'a' if camera == 'T' else 'b'} first."
            )
            return

        self.start_worker("flight_route")

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
                    except Exception:
                        pass
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
            "extract_thermal_frames": self.extract_thermal_status,
            "extract_rgb_frames": self.extract_rgb_status,
            "detection": self.detect_status,
            "georeference": self.georef_status,
            "add_frame_detections": self.frame_detections_status,
            "tracking": self.track_status,
            "add_layers": self.layers_status,
            "calculate_fov": self.calculate_fov_status,
            "add_fov": self.add_fov_status,
            "add_merged_fov": self.add_merged_fov_status,
            "orthomosaic": self.ortho_status,
            "export_geotiffs": self.export_geotiffs_status,
            "flight_route": self.flight_route_status,
            "add_ortho": self.add_ortho_status,
            "add_geotiffs": self.add_geotiffs_status,
            "add_flight_route": self.add_flight_route_status,
            "perpendicular": self.perpendicular_status,
            "add_perpendicular": self.add_perpendicular_status,
            "track_perpendicular": self.track_perpendicular_status,
            "add_track_perpendicular": self.add_track_perpendicular_status,
            "sam3_segmentation": self.sam3_segment_status,
            "sam3_georeference": self.sam3_georef_status,
            "add_sam3": self.add_sam3_status
        }
        if step in status_map:
            status_map[step].setText(status)

    def set_buttons_enabled(self, enabled: bool):
        """Enable or disable all processing buttons."""
        self.extract_thermal_btn.setEnabled(enabled)
        self.extract_rgb_btn.setEnabled(enabled)
        self.detect_btn.setEnabled(enabled)
        self.georef_btn.setEnabled(enabled)
        self.add_frame_detections_btn.setEnabled(enabled)
        self.track_btn.setEnabled(enabled)
        self.add_layers_btn.setEnabled(enabled)
        self.calculate_fov_btn.setEnabled(enabled)
        self.add_fov_btn.setEnabled(enabled)
        self.add_merged_fov_btn.setEnabled(enabled)
        self.ortho_btn.setEnabled(enabled)
        self.export_geotiffs_btn.setEnabled(enabled)
        self.flight_route_btn.setEnabled(enabled)
        self.add_ortho_btn.setEnabled(enabled)
        self.add_geotiffs_btn.setEnabled(enabled)
        self.add_flight_route_btn.setEnabled(enabled)
        self.perpendicular_btn.setEnabled(enabled)
        self.add_perpendicular_btn.setEnabled(enabled)
        self.track_perpendicular_btn.setEnabled(enabled)
        self.add_track_perpendicular_btn.setEnabled(enabled)
        self.refresh_status_btn.setEnabled(enabled)
        self.sam3_segment_btn.setEnabled(enabled)
        self.sam3_georef_btn.setEnabled(enabled)
        self.add_sam3_btn.setEnabled(enabled)

    def run_perpendicular(self):
        """Run perpendicular distance calculation step."""
        config = self.get_config()
        target_folder = config["target_folder"]

        route_file = os.path.join(target_folder, "flight_route", "flight_route.geojson")
        georef_file = os.path.join(target_folder, "georeferenced", "georeferenced.txt")

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
        perp_file = os.path.join(config["target_folder"], "flight_route", "perpendicular.json")

        if not os.path.exists(perp_file):
            QMessageBox.warning(
                self, "Missing Data",
                "Perpendicular distances have not been calculated.\n"
                "Please run 'Calculate Perpendicular' first."
            )
            return

        try:
            self.log("Adding perpendicular lines to QGIS...")
            self.update_status("add_perpendicular", "🟡 Loading...")

            with open(perp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            perpendiculas = data.get("perpendiculas", [])
            if not perpendiculas:
                QMessageBox.warning(self, "No Data", "No perpendicular results found.")
                self.update_status("add_perpendicular", "🔴 No data")
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

            group = self._create_layer_group("BAMBI Perpendicular")
            QgsProject.instance().addMapLayer(layer, False)
            group.addLayer(layer)

            self.log(f"Added {len(features)} perpendicular lines to QGIS")
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

        route_file = os.path.join(target_folder, "flight_route", "flight_route.geojson")
        tracks_folder = os.path.join(target_folder, "tracks")

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
        perp_file = os.path.join(config["target_folder"], "flight_route", "perpendicular_tracks.json")

        if not os.path.exists(perp_file):
            QMessageBox.warning(
                self, "Missing Data",
                "Track perpendicular distances have not been calculated.\n"
                "Please run 'Calculate Track Perpendicular' first."
            )
            return

        try:
            self.log("Adding track perpendicular lines to QGIS...")
            self.update_status("add_track_perpendicular", "🟡 Loading...")

            with open(perp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            track_entries = data.get("tracks", [])
            if not track_entries:
                QMessageBox.warning(self, "No Data", "No track perpendicular results found.")
                self.update_status("add_track_perpendicular", "🔴 No data")
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

            group = self._create_layer_group("BAMBI Track Perpendicular")
            QgsProject.instance().addMapLayer(layer, False)
            group.addLayer(layer)

            self.log(f"Added {len(features)} track perpendicular lines to QGIS")
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
        tracks_folder = os.path.join(config["target_folder"], "tracks")

        if not os.path.exists(tracks_folder):
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                "Tracking has not been completed.\nPlease run Step 4 first."
            )
            return

        try:
            self.log("Adding tracks to QGIS...")
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

            # Check if there are many tracks - warn user
            num_tracks = len(all_tracks)
            if num_tracks > 50:
                reply = QMessageBox.question(
                    self,
                    "Many Tracks",
                    f"Found {num_tracks} tracks. Creating individual layers for each may slow down QGIS.\n\n"
                    f"Continue with individual layers?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply == QMessageBox.No:
                    self.update_status("add_layers", "⚪ Cancelled")
                    return

            # Create main group for all tracks at top of layer tree
            main_group = self._create_layer_group("BAMBI Wildlife Tracks")

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
        except Exception:
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
        except Exception:
            pass

    def _style_track_layers(self, final_pos_layer, paths_layer):
        """Apply default styling to track visualization layers (legacy method).

        :param final_pos_layer: Polygon layer with final bounding boxes
        :param paths_layer: Line layer with track paths
        """
        from qgis.core import (
            QgsSymbol, QgsFillSymbol, QgsLineSymbol,
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
        """Load tracks from a CSV file."""
        tracks = {}

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split(',')
                    if len(parts) >= 10:
                        try:
                            frame = int(parts[0])
                            track_id = int(parts[1])
                            x1 = float(parts[2])
                            y1 = float(parts[3])
                            z1 = float(parts[4])
                            x2 = float(parts[5])
                            y2 = float(parts[6])
                            z2 = float(parts[7])
                            conf = float(parts[8])
                            cls = int(parts[9])
                            interpolated = int(parts[10]) if len(parts) > 10 else 0

                            if track_id not in tracks:
                                tracks[track_id] = []

                            tracks[track_id].append({
                                'frame': frame,
                                'x1': x1, 'y1': y1, 'z1': z1,
                                'x2': x2, 'y2': y2, 'z2': z2,
                                'confidence': conf,
                                'class_id': cls,
                                'interpolated': interpolated
                            })
                        except (ValueError, IndexError):
                            continue

        except Exception as e:
            self.log(f"Error reading {csv_path}: {str(e)}")

        return tracks

    def add_fov_to_qgis(self):
        """Add Field of View polygons as QGIS layers."""
        config = self.get_config()
        fov_folder = os.path.join(config["target_folder"], "fov")
        fov_file = os.path.join(fov_folder, "fov_polygons.txt")

        if not os.path.exists(fov_file):
            QMessageBox.warning(
                self,
                "Missing FoV Data",
                "FoV calculation has not been completed.\nPlease run Step 6 first."
            )
            return

        try:
            self.log("Adding FoV layers to QGIS...")
            self.update_status("add_fov", "🟡 Loading...")

            # Get target CRS
            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{config['target_epsg']}")

            # Load FoV polygons
            fov_polygons = self.load_fov_polygons(fov_file)

            if not fov_polygons:
                QMessageBox.warning(self, "No FoV Data", "No valid FoV polygons found.")
                self.update_status("add_fov", "🔴 No data")
                return

            # Ask user how to add layers
            num_frames = len(fov_polygons)
            if num_frames > 100:
                reply = QMessageBox.question(
                    self,
                    "Many Frames",
                    f"Found {num_frames} FoV polygons.\n"
                    f"Loading all as separate layers may slow down QGIS.\n\n"
                    f"Options:\n"
                    f"- Yes: Load first 100 frames as separate layers\n"
                    f"- No: Load all frames in a single combined layer\n"
                    f"- Cancel: Cancel operation",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
                )
                if reply == QMessageBox.Cancel:
                    self.update_status("add_fov", "⚪ Cancelled")
                    return
                elif reply == QMessageBox.Yes:
                    # Limit to 100 frames as separate layers
                    fov_polygons = dict(list(fov_polygons.items())[:100])
                    self._add_fov_separate_layers(fov_polygons, target_crs)
                else:
                    # Combined layer
                    self._add_fov_combined_layer(fov_polygons, target_crs)
            else:
                self._add_fov_separate_layers(fov_polygons, target_crs)

            self.log(f"Added FoV layers to QGIS")
            self.update_status("add_fov", "🟢 Completed")
            self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding FoV layers: {str(e)}")
            self.update_status("add_fov", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add FoV layers: {str(e)}")

    def _add_fov_separate_layers(self, fov_polygons: Dict[int, list], target_crs):
        """Add FoV polygons as separate layers for each frame."""
        # Create a group for the layers at top of layer tree
        group = self._create_layer_group("BAMBI FoV Polygons")

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
            layer.setCustomProperty("bambi_detection_camera", "T" if self.detection_camera_combo.currentIndex() == 0 else "W")
            layer.setCustomProperty("bambi_dem_path", self.dem_path_edit.text().strip())
            layer.setCustomProperty("bambi_correction_path", self.correction_path_edit.text().strip())

            # Add layer to project and group
            QgsProject.instance().addMapLayer(layer, False)
            group.addLayer(layer)

        # Collapse the group
        group.setExpanded(False)

    def _add_fov_combined_layer(self, fov_polygons: Dict[int, list], target_crs):
        """Add all FoV polygons as a single combined layer."""
        # Create a single layer with all polygons
        layer_name = "BAMBI FoV Polygons (Combined)"
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
        layer.setCustomProperty("bambi_detection_camera", "T" if self.detection_camera_combo.currentIndex() == 0 else "W")
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
        fov_folder = os.path.join(config["target_folder"], "fov")
        fov_file = os.path.join(fov_folder, "fov_polygons.txt")

        if not os.path.exists(fov_file):
            QMessageBox.warning(
                self,
                "Missing FoV Data",
                "FoV calculation has not been completed.\nPlease run Step 6 first."
            )
            return

        try:
            self.log("Creating merged FoV layer...")
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
            layer_name = "BAMBI FoV Coverage (Merged)"
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

            self.log(f"Merged FoV layer added to QGIS")
            self.log(f"  Total coverage area: {area_m2:.2f} m² ({area_ha:.4f} ha)")
            self.update_status("add_merged_fov", "🟢 Completed")
            self.iface.mapCanvas().refresh()

            # Show info to user
            QMessageBox.information(
                self,
                "Merged FoV Created",
                f"Merged FoV polygon created from {len(fov_polygons)} frames.\n\n"
                f"Total coverage area:\n"
                f"  {area_m2:,.2f} m²\n"
                f"  {area_ha:.4f} ha\n\n"
                f"You can also use QGIS Field Calculator for precise area calculation."
            )

        except Exception as e:
            self.log(f"Error creating merged FoV layer: {str(e)}")
            self.update_status("add_merged_fov", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to create merged FoV layer: {str(e)}")

    def load_fov_polygons(self, fov_file: str) -> Dict[int, list]:
        """Load FoV polygons from file.

        :param fov_file: Path to FoV polygons file
        :return: Dictionary mapping frame index to list of (x, y, z) points
        """
        polygons = {}

        try:
            with open(fov_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    if len(parts) < 2:
                        continue

                    frame_idx = int(parts[0])
                    num_points = int(parts[1])

                    if num_points == 0:
                        continue

                    # Parse points (x y z triplets)
                    points = []
                    for i in range(num_points):
                        idx = 2 + i * 3
                        if idx + 2 < len(parts):
                            x = float(parts[idx])
                            y = float(parts[idx + 1])
                            z = float(parts[idx + 2])
                            points.append((x, y, z))

                    if points:
                        polygons[frame_idx] = points

        except Exception as e:
            self.log(f"Error reading FoV file: {str(e)}")

        return polygons

    def add_frame_detections_to_qgis(self):
        """Add geo-referenced detections as QGIS layers (one layer per frame)."""
        config = self.get_config()
        georef_folder = os.path.join(config["target_folder"], "georeferenced")
        georef_file = os.path.join(georef_folder, "georeferenced.txt")

        if not os.path.exists(georef_file):
            QMessageBox.warning(
                self,
                "Missing Data",
                "Geo-referencing has not been completed.\nPlease run Step 3 first."
            )
            return

        try:
            self.log("Adding per-frame detection layers to QGIS...")
            self.update_status("add_frame_detections", "🟡 Loading...")

            # Get target CRS
            target_crs = QgsCoordinateReferenceSystem(f"EPSG:{config['target_epsg']}")

            # Load detections grouped by frame
            frame_detections = self.load_detections_by_frame(georef_file)

            if not frame_detections:
                QMessageBox.warning(self, "No Detections", "No valid detections found.")
                self.update_status("add_frame_detections", "🔴 No data")
                return

            num_frames = len(frame_detections)
            total_dets = sum(len(dets) for dets in frame_detections.values())

            if num_frames > 100:
                reply = QMessageBox.question(
                    self,
                    "Many Frames",
                    f"Found {total_dets} detections in {num_frames} frames.\n"
                    f"Loading all as separate layers may slow down QGIS.\n\n"
                    f"Options:\n"
                    f"- Yes: Load first 100 frames as separate layers\n"
                    f"- No: Load all in a single combined layer\n"
                    f"- Cancel: Cancel operation",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
                )
                if reply == QMessageBox.Cancel:
                    self.update_status("add_frame_detections", "⚪ Cancelled")
                    return
                elif reply == QMessageBox.Yes:
                    frame_detections = dict(list(frame_detections.items())[:100])
                    self._add_detections_separate_layers(frame_detections, target_crs)
                else:
                    self._add_detections_combined_layer(frame_detections, target_crs)
            else:
                self._add_detections_separate_layers(frame_detections, target_crs)

            self.log(f"Added detection layers to QGIS")
            self.update_status("add_frame_detections", "🟢 Completed")
            self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding detection layers: {str(e)}")
            self.update_status("add_frame_detections", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add detection layers: {str(e)}")

    def _add_detections_separate_layers(self, frame_detections: Dict[int, list], target_crs):
        """Add detections as separate layers for each frame."""
        # Create a group for the layers at top of layer tree
        group = self._create_layer_group("BAMBI Frame Detections")

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
            layer.setCustomProperty("bambi_detection_camera", "T" if self.detection_camera_combo.currentIndex() == 0 else "W")
            layer.setCustomProperty("bambi_dem_path", self.dem_path_edit.text().strip())
            layer.setCustomProperty("bambi_correction_path", self.correction_path_edit.text().strip())

            # Add layer to project and group
            QgsProject.instance().addMapLayer(layer, False)
            group.addLayer(layer)

        # Collapse the group
        group.setExpanded(False)

    def _add_detections_combined_layer(self, frame_detections: Dict[int, list], target_crs):
        """Add all detections in a single combined layer."""
        layer_name = "BAMBI Detections (All Frames)"
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
        layer.setCustomProperty("bambi_detection_camera", "T" if self.detection_camera_combo.currentIndex() == 0 else "W")
        layer.setCustomProperty("bambi_dem_path", self.dem_path_edit.text().strip())
        layer.setCustomProperty("bambi_correction_path", self.correction_path_edit.text().strip())

        # Add layer to project
        QgsProject.instance().addMapLayer(layer)

    def load_detections_by_frame(self, georef_file: str) -> Dict[int, list]:
        """Load geo-referenced detections grouped by frame.

        :param georef_file: Path to georeferenced detections file
        :return: Dictionary mapping frame index to list of detections
        """
        from collections import defaultdict

        frame_detections = defaultdict(list)

        try:
            with open(georef_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    if len(parts) >= 10:
                        try:
                            idx = int(parts[0])
                            frame = int(parts[1])
                            x1 = float(parts[2])
                            y1 = float(parts[3])
                            z1 = float(parts[4])
                            x2 = float(parts[5])
                            y2 = float(parts[6])
                            z2 = float(parts[7])
                            conf = float(parts[8])
                            cls = int(parts[9])

                            # Skip invalid detections
                            if x1 < 0 or y1 < 0:
                                continue

                            frame_detections[frame].append({
                                'idx': idx,
                                'frame': frame,
                                'x1': x1, 'y1': y1, 'z1': z1,
                                'x2': x2, 'y2': y2, 'z2': z2,
                                'confidence': conf,
                                'class_id': cls
                            })
                        except (ValueError, IndexError):
                            continue

        except Exception as e:
            self.log(f"Error reading georeferenced file: {str(e)}")

        return dict(frame_detections)

    def add_orthomosaic_to_qgis(self):
        """Add the generated orthomosaic to QGIS as a raster layer."""
        config = self.get_config()
        ortho_folder = os.path.join(config["target_folder"], "orthomosaic")
        ortho_file = os.path.join(ortho_folder, "orthomosaic.tif")

        if not os.path.exists(ortho_file):
            QMessageBox.warning(
                self,
                "Missing Orthomosaic",
                "Orthomosaic has not been generated.\nPlease run Step 6 first."
            )
            return

        try:
            self.log("Adding orthomosaic to QGIS...")
            self.update_status("add_ortho", "🟡 Loading...")

            # Create raster layer
            layer_name = "BAMBI Orthomosaic"
            layer = QgsRasterLayer(ortho_file, layer_name)

            if not layer.isValid():
                raise RuntimeError(f"Failed to load raster: {ortho_file}")

            # Add layer to project
            QgsProject.instance().addMapLayer(layer)

            self.log(f"Added orthomosaic layer: {layer_name}")
            self.update_status("add_ortho", "🟢 Added")

            # Zoom to layer extent
            # self.iface.mapCanvas().setExtent(layer.extent())
            # self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding orthomosaic: {str(e)}")
            self.update_status("add_ortho", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add orthomosaic: {str(e)}")

    def add_geotiffs_to_qgis(self):
        """Add exported frame GeoTIFFs to QGIS as raster layers in a group."""
        config = self.get_config()
        geotiff_folder = os.path.join(config["target_folder"], "geotiffs")

        if not os.path.exists(geotiff_folder):
            QMessageBox.warning(
                self,
                "Missing GeoTIFFs",
                "Frame GeoTIFFs have not been exported.\nPlease run Step 7 first."
            )
            return

        try:
            self.log("Adding frame GeoTIFFs to QGIS...")
            self.update_status("add_geotiffs", "🟡 Loading...")

            # Find all GeoTIFF files
            geotiff_files = []
            for f in os.listdir(geotiff_folder):
                if f.lower().endswith(('.tif', '.tiff')):
                    geotiff_files.append(os.path.join(geotiff_folder, f))

            if not geotiff_files:
                QMessageBox.warning(self, "No GeoTIFFs", "No GeoTIFF files found.")
                self.update_status("add_geotiffs", "🔴 No files")
                return

            # Sort files by name (which is frame index)
            geotiff_files.sort()

            # Limit number of layers to avoid performance issues
            max_layers = 100
            if len(geotiff_files) > max_layers:
                reply = QMessageBox.question(
                    self,
                    "Many Files",
                    f"Found {len(geotiff_files)} GeoTIFF files.\n"
                    f"Loading all may slow down QGIS.\n\n"
                    f"Load only first {max_layers} files?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
                )
                if reply == QMessageBox.Cancel:
                    self.update_status("add_geotiffs", "⚪ Cancelled")
                    return
                elif reply == QMessageBox.Yes:
                    geotiff_files = geotiff_files[:max_layers]

            # Create a group for the layers at top of layer tree
            group = self._create_layer_group("BAMBI Frame GeoTIFFs")

            loaded_count = 0

            for geotiff_path in geotiff_files:
                filename = os.path.basename(geotiff_path)
                layer_name = f"Frame {os.path.splitext(filename)[0]}"

                layer = QgsRasterLayer(geotiff_path, layer_name)

                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer, False)
                    group.addLayer(layer)
                    loaded_count += 1
                else:
                    self.log(f"Warning: Could not load {filename}")

            self.log(f"Added {loaded_count} GeoTIFF layers to QGIS")
            self.update_status("add_geotiffs", "🟢 Added")

            # Collapse the group by default to improve performance
            group.setExpanded(False)

            # Refresh canvas
            self.iface.mapCanvas().refresh()

        except Exception as e:
            self.log(f"Error adding GeoTIFFs: {str(e)}")
            self.update_status("add_geotiffs", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add GeoTIFFs: {str(e)}")

    def add_flight_route_to_qgis(self):
        """Add flight route layers (polyline, points, and frame markers) to QGIS."""
        config = self.get_config()
        route_folder = os.path.join(config["target_folder"], "flight_route")

        if not os.path.exists(route_folder):
            QMessageBox.warning(
                self,
                "Missing Flight Route",
                "Flight route has not been generated.\nPlease run Step 9 first."
            )
            return

        try:
            self.log("Adding flight route to QGIS...")
            self.update_status("add_flight_route", "🟡 Loading...")

            # Create a group for the layers at top of layer tree
            group = self._create_layer_group("BAMBI Flight Route")

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
                    self.log(f"Warning: Could not load flight route line")

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
                    self.log(f"Warning: Could not load camera positions")

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
                root.removeChildNode(group)
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
            text_format.setFont(QFont("Arial", 10, QFont.Bold))
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
            label_settings.placement = QgsPalLayerSettings.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantAbove

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
            layer_uri = f"Point?crs=EPSG:{target_epsg}&field=distance:double&field=frame_idx:integer&field=label:string"
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
                    f"Warning: No distance markers created (interval {distance_interval}m > total {total_distance:.1f}m)")
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
            text_format.setFont(QFont("Arial", 9, QFont.Bold))
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
            label_settings.placement = QgsPalLayerSettings.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantAbove

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
                    "%Y-%m-%dT%H:%M:%S",     "%Y-%m-%d %H:%M:%S",
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

                col_ms  = _find_col(headers, "time", "millisecond")
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
                        ts_ms  = float(row[col_ms])  if col_ms  else None
                        ts_utc = _parse_utc(row[col_utc]) if col_utc else None
                        raw_rows.append((ts_ms, ts_utc, lat, lon))
                    except (ValueError, KeyError):
                        continue

            if len(raw_rows) < 2:
                self.log("Warning: Not enough valid GPS rows in AirData for time markers")
                return None

            # ── Project to target CRS ─────────────────────────────────────────
            # Build (elapsed_sec, unix_ts_or_None, x, y)
            first_ms  = raw_rows[0][0]
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
                f"&field=elapsed_sec:double&field=label:string"
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
                    f"Warning: No time markers created "
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
            text_format.setFont(QFont("Arial", 9, QFont.Bold))
            text_format.setSize(9)
            text_format.setColor(QColor('#6c3483'))  # Dark purple

            buffer_settings = QgsTextBufferSettings()
            buffer_settings.setEnabled(True)
            buffer_settings.setSize(1.5)
            buffer_settings.setColor(QColor('#ffffff'))
            text_format.setBuffer(buffer_settings)

            label_settings.setFormat(text_format)
            label_settings.placement = QgsPalLayerSettings.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantAbove

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
                f"&field=frame_idx:integer&field=imagefile:string&field=label:string"
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
            label_settings.placement = QgsPalLayerSettings.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantBelow

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
            text_format.setFont(QFont("Arial", 10, QFont.Bold))
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
            label_settings.placement = QgsPalLayerSettings.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantAbove

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
            text_format.setFont(QFont("Arial", 9, QFont.Bold))
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
            label_settings.placement = QgsPalLayerSettings.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantAbove

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
        except Exception:
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
        """Called when project is cleared (new project)."""
        # Optionally reset fields to defaults when creating new project
        pass

    def save_config_to_project(self):
        """Save all configuration to the QGIS project."""
        project = QgsProject.instance()

        # ===== Input Mode =====
        project.writeEntry(PLUGIN_SCOPE, "Input/VideoMode",
                           "1" if self.video_mode_check.isChecked() else "0")

        # ===== Video Input Paths =====
        project.writeEntry(PLUGIN_SCOPE, "Input/ThermalVideoPaths",
                           self.thermal_video_paths_edit.text())
        project.writeEntry(PLUGIN_SCOPE, "Input/ThermalSrtPaths",
                           self.thermal_srt_paths_edit.text())
        project.writeEntry(PLUGIN_SCOPE, "Input/ThermalCalibrationPath",
                           self.thermal_calibration_path_edit.text())
        project.writeEntry(PLUGIN_SCOPE, "Input/ThermalCalibrationPreset",
                           self.thermal_calib_preset_combo.currentText())
        project.writeEntry(PLUGIN_SCOPE, "Input/RgbVideoPaths",
                           self.rgb_video_paths_edit.text())
        project.writeEntry(PLUGIN_SCOPE, "Input/RgbSrtPaths",
                           self.rgb_srt_paths_edit.text())
        project.writeEntry(PLUGIN_SCOPE, "Input/RgbCalibrationPath",
                           self.rgb_calibration_path_edit.text())
        project.writeEntry(PLUGIN_SCOPE, "Input/RgbCalibrationPreset",
                           self.rgb_calib_preset_combo.currentText())

        # ===== Photo Input Paths =====
        project.writeEntry(PLUGIN_SCOPE, "Input/ThermalPhotoDir",
                           self.thermal_photo_dir_edit.text())
        project.writeEntry(PLUGIN_SCOPE, "Input/ThermalPhotoCalibrationPath",
                           self.thermal_photo_calibration_path_edit.text())
        project.writeEntry(PLUGIN_SCOPE, "Input/ThermalPhotoCalibrationPreset",
                           self.thermal_photo_calib_preset_combo.currentText())
        project.writeEntry(PLUGIN_SCOPE, "Input/RgbPhotoDir",
                           self.rgb_photo_dir_edit.text())
        project.writeEntry(PLUGIN_SCOPE, "Input/RgbPhotoCalibrationPath",
                           self.rgb_photo_calibration_path_edit.text())
        project.writeEntry(PLUGIN_SCOPE, "Input/RgbPhotoCalibrationPreset",
                           self.rgb_photo_calib_preset_combo.currentText())
        project.writeEntry(PLUGIN_SCOPE, "Input/ThermalPhotoFilter",
                           "1" if self.thermal_photo_filter_check.isChecked() else "0")
        project.writeEntry(PLUGIN_SCOPE, "Input/RgbPhotoFilter",
                           "1" if self.rgb_photo_filter_check.isChecked() else "0")
        project.writeEntry(PLUGIN_SCOPE, "Input/PhotoTimezone",
                           self.photo_timezone_combo.currentText())
        project.writeEntry(PLUGIN_SCOPE, "Input/PhotoTimezoneAuto",
                           "1" if self.photo_tz_auto_check.isChecked() else "0")
        project.writeEntry(PLUGIN_SCOPE, "Input/AirdataPath",
                           self.airdata_path_edit.text())
        project.writeEntry(PLUGIN_SCOPE, "Input/DemPath",
                           self.dem_path_edit.text())
        project.writeEntry(PLUGIN_SCOPE, "Input/DemMetadataPath",
                           self.dem_metadata_path_edit.text())
        project.writeEntryDouble(PLUGIN_SCOPE, "Input/DemPadding",
                                 self.dem_padding_spin.value())
        project.writeEntry(PLUGIN_SCOPE, "Input/GeotiffInputPath",
                           self.geotiff_input_path_edit.text())
        project.writeEntryDouble(PLUGIN_SCOPE, "Input/GeotiffSimplifyFactor",
                                 self.geotiff_simplify_spin.value())
        project.writeEntry(PLUGIN_SCOPE, "Input/CorrectionPath",
                           self.correction_path_edit.text())
        project.writeEntry(PLUGIN_SCOPE, "Input/TargetFolder",
                           self.target_folder_edit.text())
        project.writeEntry(PLUGIN_SCOPE, "Input/TargetCrs",
                           self.target_crs_edit.text())

        # ===== Detection Settings =====
        project.writeEntry(PLUGIN_SCOPE, "Detection/ModelPath",
                           self.model_path_edit.text())
        project.writeEntryDouble(PLUGIN_SCOPE, "Detection/Confidence",
                                 self.confidence_spin.value())
        project.writeEntryBool(PLUGIN_SCOPE, "Detection/AllFrames",
                               self.detect_all_frames_check.isChecked())
        project.writeEntryDouble(PLUGIN_SCOPE, "Detection/StartFrame",
                                 self.detect_start_frame_spin.value())
        project.writeEntryDouble(PLUGIN_SCOPE, "Detection/EndFrame",
                                 self.detect_end_frame_spin.value())
        project.writeEntryDouble(PLUGIN_SCOPE, "Detection/SampleRate",
                                 self.detect_sample_rate_spin.value())

        # ===== Correction Settings =====
        project.writeEntryDouble(PLUGIN_SCOPE, "Correction/RotationUnit",
                                 self.rotation_unit_combo.currentIndex())
        project.writeEntryDouble(PLUGIN_SCOPE, "Correction/TransX",
                                 self.trans_x_spin.value())
        project.writeEntryDouble(PLUGIN_SCOPE, "Correction/TransY",
                                 self.trans_y_spin.value())
        project.writeEntryDouble(PLUGIN_SCOPE, "Correction/TransZ",
                                 self.trans_z_spin.value())
        project.writeEntryDouble(PLUGIN_SCOPE, "Correction/RotX",
                                 self.rot_x_spin.value())
        project.writeEntryDouble(PLUGIN_SCOPE, "Correction/RotY",
                                 self.rot_y_spin.value())
        project.writeEntryDouble(PLUGIN_SCOPE, "Correction/RotZ",
                                 self.rot_z_spin.value())

        # Save additional corrections as JSON
        corrections_data = []
        for i in range(self.additional_corrections_list.count()):
            item = self.additional_corrections_list.item(i)
            data = item.data(Qt.UserRole)
            if data:
                corrections_data.append(data)
        project.writeEntry(PLUGIN_SCOPE, "Correction/AdditionalCorrections",
                           json.dumps(corrections_data))

        # ===== Tracking Settings =====
        project.writeEntryDouble(PLUGIN_SCOPE, "Tracking/TrackerType",
                                 self.tracker_backend_combo.currentIndex())
        project.writeEntryDouble(PLUGIN_SCOPE, "Tracking/ReidModel",
                                 self.reid_model_combo.currentIndex())
        project.writeEntry(PLUGIN_SCOPE, "Tracking/CustomReidPath",
                           self.custom_reid_path_edit.text())
        project.writeEntry(PLUGIN_SCOPE, "Tracking/TrackerParams",
                           self.tracker_params_edit.toPlainText())
        project.writeEntryDouble(PLUGIN_SCOPE, "Tracking/IouThreshold",
                                 self.iou_threshold_spin.value())
        project.writeEntryDouble(PLUGIN_SCOPE, "Tracking/MaxAge",
                                 self.max_age_spin.value())
        project.writeEntryDouble(PLUGIN_SCOPE, "Tracking/MaxCenterDist",
                                 self.max_center_dist_spin.value())
        project.writeEntryDouble(PLUGIN_SCOPE, "Tracking/TrackerMode",
                                 self.tracker_mode_combo.currentIndex())
        project.writeEntryBool(PLUGIN_SCOPE, "Tracking/ClassAware",
                               self.class_aware_check.isChecked())
        project.writeEntryBool(PLUGIN_SCOPE, "Tracking/Interpolate",
                               self.interpolate_check.isChecked())

        # ===== Field of View Settings =====
        project.writeEntryBool(PLUGIN_SCOPE, "FoV/UseMask",
                               self.use_fov_mask_check.isChecked())
        project.writeEntry(PLUGIN_SCOPE, "FoV/MaskPath",
                           self.fov_mask_path_edit.text())
        project.writeEntryDouble(PLUGIN_SCOPE, "FoV/MaskSimplify",
                                 self.mask_simplify_spin.value())
        project.writeEntryBool(PLUGIN_SCOPE, "FoV/AllFrames",
                               self.fov_all_frames_check.isChecked())
        project.writeEntryDouble(PLUGIN_SCOPE, "FoV/StartFrame",
                                 self.fov_start_frame_spin.value())
        project.writeEntryDouble(PLUGIN_SCOPE, "FoV/EndFrame",
                                 self.fov_end_frame_spin.value())
        project.writeEntryDouble(PLUGIN_SCOPE, "FoV/SampleRate",
                                 self.fov_sample_rate_spin.value())

        # ===== Orthomosaic Settings =====
        project.writeEntryDouble(PLUGIN_SCOPE, "Ortho/Resolution",
                                 self.ortho_resolution_spin.value())
        project.writeEntryDouble(PLUGIN_SCOPE, "Ortho/BlendMode",
                                 self.blend_mode_combo.currentIndex())
        project.writeEntryBool(PLUGIN_SCOPE, "Ortho/AllFrames",
                               self.ortho_all_frames_check.isChecked())
        project.writeEntryDouble(PLUGIN_SCOPE, "Ortho/StartFrame",
                                 self.ortho_start_frame_spin.value())
        project.writeEntryDouble(PLUGIN_SCOPE, "Ortho/EndFrame",
                                 self.ortho_end_frame_spin.value())
        project.writeEntryBool(PLUGIN_SCOPE, "Ortho/Crop",
                               self.ortho_crop_check.isChecked())
        project.writeEntryBool(PLUGIN_SCOPE, "Ortho/Overviews",
                               self.ortho_overviews_check.isChecked())
        project.writeEntryDouble(PLUGIN_SCOPE, "Ortho/TileSize",
                                 self.ortho_tile_size_spin.value())
        project.writeEntryDouble(PLUGIN_SCOPE, "Ortho/FrameStep",
                                 self.ortho_frame_step_spin.value())

        # ===== SAM3 Settings =====
        # Note: API key is intentionally NOT saved for security
        project.writeEntry(PLUGIN_SCOPE, "SAM3/Prompts",
                           self.sam3_prompts_edit.toPlainText())
        project.writeEntryDouble(PLUGIN_SCOPE, "SAM3/Confidence",
                                 self.sam3_confidence_spin.value())
        project.writeEntryBool(PLUGIN_SCOPE, "SAM3/AllFrames",
                               self.sam3_all_frames_check.isChecked())
        project.writeEntryDouble(PLUGIN_SCOPE, "SAM3/StartFrame",
                                 self.sam3_start_frame_spin.value())
        project.writeEntryDouble(PLUGIN_SCOPE, "SAM3/EndFrame",
                                 self.sam3_end_frame_spin.value())
        project.writeEntryDouble(PLUGIN_SCOPE, "SAM3/FrameStep",
                                 self.sam3_step_spin.value())

        # ===== Flight Route Settings =====
        project.writeEntryBool(PLUGIN_SCOPE, "FlightRoute/FrameMarkersEnabled",
                               self.frame_markers_enabled_check.isChecked())
        project.writeEntryDouble(PLUGIN_SCOPE, "FlightRoute/FrameMarkerInterval",
                                 self.frame_marker_interval_spin.value())
        project.writeEntryBool(PLUGIN_SCOPE, "FlightRoute/IncludeFrameZero",
                               self.frame_marker_include_zero_check.isChecked())
        project.writeEntryBool(PLUGIN_SCOPE, "FlightRoute/ImageLabelsEnabled",
                               self.image_labels_enabled_check.isChecked())
        project.writeEntryDouble(PLUGIN_SCOPE, "FlightRoute/ImageLabelInterval",
                                 self.image_label_interval_spin.value())
        project.writeEntryBool(PLUGIN_SCOPE, "FlightRoute/DistanceMarkersEnabled",
                               self.distance_markers_enabled_check.isChecked())
        project.writeEntryDouble(PLUGIN_SCOPE, "FlightRoute/DistanceMarkerInterval",
                                 self.distance_marker_interval_spin.value())
        project.writeEntryBool(PLUGIN_SCOPE, "FlightRoute/IncludeDistanceStart",
                               self.distance_marker_include_start_check.isChecked())
        project.writeEntryBool(PLUGIN_SCOPE, "FlightRoute/TimeMarkersEnabled",
                               self.time_markers_enabled_check.isChecked())
        project.writeEntryDouble(PLUGIN_SCOPE, "FlightRoute/TimeMarkerInterval",
                                 self.time_marker_interval_spin.value())
        project.writeEntryDouble(PLUGIN_SCOPE, "FlightRoute/TimeMarkerTypeIndex",
                                 self.time_marker_type_combo.currentIndex())
        project.writeEntryBool(PLUGIN_SCOPE, "FlightRoute/IncludeTimeStart",
                               self.time_marker_include_start_check.isChecked())
        project.writeEntryDouble(PLUGIN_SCOPE, "FlightRoute/CameraComboIndex",
                                 self.flight_route_camera_combo.currentIndex())

        # Mark project as modified so user is prompted to save
        project.setDirty(True)

        self.log("Configuration saved to project")

    def load_config_from_project(self):
        """Load all configuration from the QGIS project."""
        project = QgsProject.instance()

        # Helper functions
        def read_str(key: str, default: str = "") -> str:
            value, ok = project.readEntry(PLUGIN_SCOPE, key, default)
            return value if ok else default

        def read_double(key: str, default: float = 0.0) -> float:
            value, ok = project.readDoubleEntry(PLUGIN_SCOPE, key, default)
            return value if ok else default

        def read_bool(key: str, default: bool = False) -> bool:
            value, ok = project.readBoolEntry(PLUGIN_SCOPE, key, default)
            return value if ok else default

        def read_int(key: str, default: int = 0) -> int:
            return int(read_double(key, float(default)))

        # Check if there's any saved config
        test_value, has_config = project.readEntry(PLUGIN_SCOPE, "Input/TargetFolder", "")
        if not has_config:
            return  # No saved config in this project

        # ===== Input Mode =====
        video_mode_val = read_str("Input/VideoMode")
        if video_mode_val != "":
            self.video_mode_check.setChecked(video_mode_val != "0")

        # ===== Video Input Paths =====
        self.thermal_video_paths_edit.setText(read_str("Input/ThermalVideoPaths"))
        self.thermal_srt_paths_edit.setText(read_str("Input/ThermalSrtPaths"))
        self.thermal_calibration_path_edit.setText(read_str("Input/ThermalCalibrationPath"))
        thermal_preset = read_str("Input/ThermalCalibrationPreset")
        t_idx = self.thermal_calib_preset_combo.findText(thermal_preset)
        if t_idx >= 0:
            self.thermal_calib_preset_combo.setCurrentIndex(t_idx)
        self.rgb_video_paths_edit.setText(read_str("Input/RgbVideoPaths"))
        self.rgb_srt_paths_edit.setText(read_str("Input/RgbSrtPaths"))
        self.rgb_calibration_path_edit.setText(read_str("Input/RgbCalibrationPath"))
        rgb_preset = read_str("Input/RgbCalibrationPreset")
        r_idx = self.rgb_calib_preset_combo.findText(rgb_preset)
        if r_idx >= 0:
            self.rgb_calib_preset_combo.setCurrentIndex(r_idx)

        # ===== Photo Input Paths =====
        self.thermal_photo_dir_edit.setText(read_str("Input/ThermalPhotoDir"))
        self.thermal_photo_calibration_path_edit.setText(
            read_str("Input/ThermalPhotoCalibrationPath"))
        tp_preset = read_str("Input/ThermalPhotoCalibrationPreset")
        tp_idx = self.thermal_photo_calib_preset_combo.findText(tp_preset)
        if tp_idx >= 0:
            self.thermal_photo_calib_preset_combo.setCurrentIndex(tp_idx)
        self.rgb_photo_dir_edit.setText(read_str("Input/RgbPhotoDir"))
        self.rgb_photo_calibration_path_edit.setText(
            read_str("Input/RgbPhotoCalibrationPath"))
        rp_preset = read_str("Input/RgbPhotoCalibrationPreset")
        rp_idx = self.rgb_photo_calib_preset_combo.findText(rp_preset)
        if rp_idx >= 0:
            self.rgb_photo_calib_preset_combo.setCurrentIndex(rp_idx)
        self.thermal_photo_filter_check.setChecked(
            read_str("Input/ThermalPhotoFilter") == "1")
        self.rgb_photo_filter_check.setChecked(
            read_str("Input/RgbPhotoFilter") == "1")
        saved_tz = read_str("Input/PhotoTimezone")
        if saved_tz:
            tz_idx = self.photo_timezone_combo.findText(saved_tz)
            if tz_idx >= 0:
                self.photo_timezone_combo.setCurrentIndex(tz_idx)
            else:
                self.photo_timezone_combo.setCurrentText(saved_tz)
        self.photo_tz_auto_check.setChecked(
            read_str("Input/PhotoTimezoneAuto") != "0")
        self.airdata_path_edit.setText(read_str("Input/AirdataPath"))
        self.dem_path_edit.setText(read_str("Input/DemPath"))
        self.dem_metadata_path_edit.setText(read_str("Input/DemMetadataPath"))
        self.dem_padding_spin.setValue(read_int("Input/DemPadding", 30))
        self.geotiff_input_path_edit.setText(read_str("Input/GeotiffInputPath"))
        self.geotiff_simplify_spin.setValue(read_int("Input/GeotiffSimplifyFactor", 2))
        self.correction_path_edit.setText(read_str("Input/CorrectionPath"))
        self.target_folder_edit.setText(read_str("Input/TargetFolder"))

        target_crs = read_str("Input/TargetCrs")
        if target_crs:
            self.target_crs_edit.setText(target_crs)
        else:
            self.target_crs_edit.setText("EPSG:32633")  # Default

        # ===== Detection Settings =====
        self.model_path_edit.setText(read_str("Detection/ModelPath"))
        self.confidence_spin.setValue(read_double("Detection/Confidence", 0.5))
        self.detect_all_frames_check.setChecked(read_bool("Detection/AllFrames", True))
        self.detect_start_frame_spin.setValue(read_int("Detection/StartFrame", 0))
        self.detect_end_frame_spin.setValue(read_int("Detection/EndFrame", 999999))
        self.detect_sample_rate_spin.setValue(read_int("Detection/SampleRate", 1))

        # ===== Correction Settings =====
        rot_unit_idx = read_int("Correction/RotationUnit", 0)
        if 0 <= rot_unit_idx < self.rotation_unit_combo.count():
            self.rotation_unit_combo.setCurrentIndex(rot_unit_idx)

        self.trans_x_spin.setValue(read_double("Correction/TransX", 0.0))
        self.trans_y_spin.setValue(read_double("Correction/TransY", 0.0))
        self.trans_z_spin.setValue(read_double("Correction/TransZ", 0.0))
        self.rot_x_spin.setValue(read_double("Correction/RotX", 0.0))
        self.rot_y_spin.setValue(read_double("Correction/RotY", 0.0))
        self.rot_z_spin.setValue(read_double("Correction/RotZ", 0.0))

        # Load additional corrections
        corrections_json = read_str("Correction/AdditionalCorrections", "[]")
        try:
            corrections_data = json.loads(corrections_json)
            self.additional_corrections_list.clear()
            for corr in corrections_data:
                self._add_correction_to_list(corr)
        except json.JSONDecodeError:
            pass

        # ===== Tracking Settings =====
        tracker_idx = read_int("Tracking/TrackerType", 0)
        if 0 <= tracker_idx < self.tracker_backend_combo.count():
            self.tracker_backend_combo.setCurrentIndex(tracker_idx)

        reid_model_idx = read_int("Tracking/ReidModel", 0)
        if 0 <= reid_model_idx < self.reid_model_combo.count():
            self.reid_model_combo.setCurrentIndex(reid_model_idx)

        self.custom_reid_path_edit.setText(read_str("Tracking/CustomReidPath"))
        self.tracker_params_edit.setPlainText(read_str("Tracking/TrackerParams"))
        self.iou_threshold_spin.setValue(read_double("Tracking/IouThreshold", 0.3))
        self.max_age_spin.setValue(read_int("Tracking/MaxAge", -1))
        self.max_center_dist_spin.setValue(read_double("Tracking/MaxCenterDist", 0.2))

        tracker_mode_idx = read_int("Tracking/TrackerMode", 1)
        if 0 <= tracker_mode_idx < self.tracker_mode_combo.count():
            self.tracker_mode_combo.setCurrentIndex(tracker_mode_idx)

        self.class_aware_check.setChecked(read_bool("Tracking/ClassAware", True))
        self.interpolate_check.setChecked(read_bool("Tracking/Interpolate", True))

        # ===== Field of View Settings =====
        self.use_fov_mask_check.setChecked(read_bool("FoV/UseMask", False))
        self.fov_mask_path_edit.setText(read_str("FoV/MaskPath"))
        self.mask_simplify_spin.setValue(read_double("FoV/MaskSimplify", 2.0))
        self.fov_all_frames_check.setChecked(read_bool("FoV/AllFrames", True))
        self.fov_start_frame_spin.setValue(read_int("FoV/StartFrame", 0))
        self.fov_end_frame_spin.setValue(read_int("FoV/EndFrame", 999999))
        self.fov_sample_rate_spin.setValue(read_int("FoV/SampleRate", 1))

        # ===== Orthomosaic Settings =====
        self.ortho_resolution_spin.setValue(read_double("Ortho/Resolution", 0.05))

        blend_mode_idx = read_int("Ortho/BlendMode", 0)
        if 0 <= blend_mode_idx < self.blend_mode_combo.count():
            self.blend_mode_combo.setCurrentIndex(blend_mode_idx)

        self.ortho_all_frames_check.setChecked(read_bool("Ortho/AllFrames", True))
        self.ortho_start_frame_spin.setValue(read_int("Ortho/StartFrame", 0))
        self.ortho_end_frame_spin.setValue(read_int("Ortho/EndFrame", 999999))
        self.ortho_crop_check.setChecked(read_bool("Ortho/Crop", True))
        self.ortho_overviews_check.setChecked(read_bool("Ortho/Overviews", True))
        self.ortho_tile_size_spin.setValue(read_int("Ortho/TileSize", 8192))
        self.ortho_frame_step_spin.setValue(read_int("Ortho/FrameStep", 1))

        # ===== SAM3 Settings =====
        self.sam3_prompts_edit.setPlainText(read_str("SAM3/Prompts"))
        self.sam3_confidence_spin.setValue(read_double("SAM3/Confidence", 0.5))
        self.sam3_all_frames_check.setChecked(read_bool("SAM3/AllFrames", True))
        self.sam3_start_frame_spin.setValue(read_int("SAM3/StartFrame", 0))
        self.sam3_end_frame_spin.setValue(read_int("SAM3/EndFrame", 999999))
        self.sam3_step_spin.setValue(read_int("SAM3/FrameStep", 1))

        # ===== Flight Route Settings =====
        self.frame_markers_enabled_check.setChecked(read_bool("FlightRoute/FrameMarkersEnabled", True))
        self.frame_marker_interval_spin.setValue(read_int("FlightRoute/FrameMarkerInterval", 100))
        self.frame_marker_include_zero_check.setChecked(read_bool("FlightRoute/IncludeFrameZero", False))
        self.image_labels_enabled_check.setChecked(read_bool("FlightRoute/ImageLabelsEnabled", False))
        self.image_label_interval_spin.setValue(read_int("FlightRoute/ImageLabelInterval", 100))
        self.distance_markers_enabled_check.setChecked(read_bool("FlightRoute/DistanceMarkersEnabled", False))
        self.distance_marker_interval_spin.setValue(read_int("FlightRoute/DistanceMarkerInterval", 100))
        self.distance_marker_include_start_check.setChecked(read_bool("FlightRoute/IncludeDistanceStart", False))
        self.time_markers_enabled_check.setChecked(read_bool("FlightRoute/TimeMarkersEnabled", False))
        self.time_marker_interval_spin.setValue(read_int("FlightRoute/TimeMarkerInterval", 10))
        time_type_idx = read_int("FlightRoute/TimeMarkerTypeIndex", 0)
        if 0 <= time_type_idx < self.time_marker_type_combo.count():
            self.time_marker_type_combo.setCurrentIndex(time_type_idx)
        self.time_marker_include_start_check.setChecked(read_bool("FlightRoute/IncludeTimeStart", False))

        camera_combo_idx = read_int("FlightRoute/CameraComboIndex", 0)
        if 0 <= camera_combo_idx < self.flight_route_camera_combo.count():
            self.flight_route_camera_combo.setCurrentIndex(camera_combo_idx)

        self.log("Configuration loaded from project")

    def _create_layer_group(self, group_name: str, at_top: bool = True) -> QgsLayerTreeGroup:
        """Create a layer group, optionally at the top of the layer tree.

        :param group_name: Name for the new group
        :param at_top: If True, insert at top of layer tree; if False, append at bottom
        :return: The created QgsLayerTreeGroup
        """
        root = QgsProject.instance().layerTreeRoot()
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
            if os.path.exists(gpkg_path):
                norm_path = os.path.normcase(os.path.abspath(gpkg_path))
                layers_to_remove = [
                    lid for lid, lyr in QgsProject.instance().mapLayers().items()
                    if isinstance(lyr, QgsVectorLayer)
                    and os.path.normcase(os.path.abspath(lyr.source().split("|")[0])) == norm_path
                ]
                if layers_to_remove:
                    QgsProject.instance().removeMapLayers(layers_to_remove)
                os.remove(gpkg_path)

            # Write to GeoPackage
            options = QgsVectorFileWriter.SaveVectorOptions()
            options.driverName = "GPKG"
            options.fileEncoding = "UTF-8"

            error = QgsVectorFileWriter.writeAsVectorFormatV3(
                mem_layer,
                gpkg_path,
                QgsCoordinateTransformContext(),
                options
            )

            if error[0] != QgsVectorFileWriter.NoError:
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