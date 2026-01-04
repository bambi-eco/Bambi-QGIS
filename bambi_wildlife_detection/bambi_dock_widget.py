# -*- coding: utf-8 -*-
"""
BAMBI Wildlife Detection - Dock Widget
=======================================

This module contains the main dock widget UI for the plugin.
"""

import os
import json
from typing import Optional, Dict, Any

from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QFileDialog, QLabel, QProgressBar, QTextEdit, QComboBox,
    QCheckBox, QTabWidget, QMessageBox, QScrollArea, QSlider,
    QFrame
)
from qgis.PyQt.QtGui import QFont, QColor
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsCoordinateReferenceSystem,
    QgsFeature, QgsGeometry, QgsPointXY, QgsField, QgsRasterLayer,
    QgsLineSymbol, QgsMarkerSymbol
)
from qgis.PyQt.QtCore import QVariant

from .bambi_processing import BambiProcessor, ProcessingWorker


class BambiDockWidget(QDockWidget):
    """Main dock widget for the BAMBI Wildlife Detection plugin."""
    
    def __init__(self, iface, parent=None):
        """Initialize the dock widget.
        
        :param iface: QGIS interface instance
        :param parent: Parent widget
        """
        super().__init__("BAMBI Wildlife Detection", parent)
        self.iface = iface
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # Processing state
        self.processor = BambiProcessor()
        self.worker = None
        self.worker_thread = None
        
        # Setup UI
        self.setup_ui()
        
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
        input_tab = QWidget()
        input_layout = QVBoxLayout(input_tab)
        main_tabs.addTab(input_tab, "Input")
        
        # Video inputs
        video_group = QGroupBox("Video Inputs")
        video_layout = QFormLayout(video_group)
        
        self.video_paths_edit = QLineEdit()
        self.video_paths_edit.setPlaceholderText("Comma-separated paths to video files")
        video_browse_btn = QPushButton("Browse...")
        video_browse_btn.clicked.connect(self.browse_videos)
        video_row = QHBoxLayout()
        video_row.addWidget(self.video_paths_edit)
        video_row.addWidget(video_browse_btn)
        video_layout.addRow("Videos:", video_row)
        
        self.srt_paths_edit = QLineEdit()
        self.srt_paths_edit.setPlaceholderText("Comma-separated paths to SRT files")
        srt_browse_btn = QPushButton("Browse...")
        srt_browse_btn.clicked.connect(self.browse_srts)
        srt_row = QHBoxLayout()
        srt_row.addWidget(self.srt_paths_edit)
        srt_row.addWidget(srt_browse_btn)
        video_layout.addRow("SRT Files:", srt_row)
        
        self.airdata_path_edit = QLineEdit()
        self.airdata_path_edit.setPlaceholderText("Path to AirData CSV file")
        airdata_browse_btn = QPushButton("Browse...")
        airdata_browse_btn.clicked.connect(self.browse_airdata)
        airdata_row = QHBoxLayout()
        airdata_row.addWidget(self.airdata_path_edit)
        airdata_row.addWidget(airdata_browse_btn)
        video_layout.addRow("AirData CSV:", airdata_row)
        
        input_layout.addWidget(video_group)
        
        # Geo-referencing data
        geo_group = QGroupBox("Geo-referencing Data")
        geo_layout = QFormLayout(geo_group)
        
        self.dem_path_edit = QLineEdit()
        self.dem_path_edit.setPlaceholderText("Path to DEM GLTF file")
        dem_browse_btn = QPushButton("Browse...")
        dem_browse_btn.clicked.connect(self.browse_dem)
        dem_row = QHBoxLayout()
        dem_row.addWidget(self.dem_path_edit)
        dem_row.addWidget(dem_browse_btn)
        geo_layout.addRow("DEM (GLTF):", dem_row)
        
        self.dem_metadata_path_edit = QLineEdit()
        self.dem_metadata_path_edit.setPlaceholderText("Path to DEM metadata JSON (auto-detected from DEM)")
        dem_meta_browse_btn = QPushButton("Browse...")
        dem_meta_browse_btn.clicked.connect(self.browse_dem_metadata)
        dem_meta_row = QHBoxLayout()
        dem_meta_row.addWidget(self.dem_metadata_path_edit)
        dem_meta_row.addWidget(dem_meta_browse_btn)
        geo_layout.addRow("DEM Metadata:", dem_meta_row)
        
        self.calibration_path_edit = QLineEdit()
        self.calibration_path_edit.setPlaceholderText("Path to calibration JSON file")
        calib_browse_btn = QPushButton("Browse...")
        calib_browse_btn.clicked.connect(self.browse_calibration)
        calib_row = QHBoxLayout()
        calib_row.addWidget(self.calibration_path_edit)
        calib_row.addWidget(calib_browse_btn)
        geo_layout.addRow("Calibration:", calib_row)
        
        self.correction_path_edit = QLineEdit()
        self.correction_path_edit.setPlaceholderText("Path to correction.json (auto-detected)")
        correction_browse_btn = QPushButton("Browse...")
        correction_browse_btn.clicked.connect(self.browse_correction)
        correction_row = QHBoxLayout()
        correction_row.addWidget(self.correction_path_edit)
        correction_row.addWidget(correction_browse_btn)
        geo_layout.addRow("Correction:", correction_row)
        
        input_layout.addWidget(geo_group)
        
        # Output configuration
        output_group = QGroupBox("Output Configuration")
        output_layout = QFormLayout(output_group)
        
        self.target_folder_edit = QLineEdit()
        self.target_folder_edit.setPlaceholderText("Target folder for all outputs")
        target_browse_btn = QPushButton("Browse...")
        target_browse_btn.clicked.connect(self.browse_target_folder)
        target_row = QHBoxLayout()
        target_row.addWidget(self.target_folder_edit)
        target_row.addWidget(target_browse_btn)
        output_layout.addRow("Target Folder:", target_row)
        
        self.target_crs_combo = QComboBox()
        self.target_crs_combo.addItems([
            "EPSG:32632 - UTM zone 32N",
            "EPSG:32633 - UTM zone 33N", 
            "EPSG:32634 - UTM zone 34N",
            "EPSG:4326 - WGS 84"
        ])
        self.target_crs_combo.setCurrentIndex(1)  # Default to 32633
        output_layout.addRow("Target CRS:", self.target_crs_combo)
        
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
        
        # ----- Sub-Tab 1: Extraction & Detection -----
        extract_detect_tab = QWidget()
        extract_detect_layout = QVBoxLayout(extract_detect_tab)
        config_sub_tabs.addTab(extract_detect_tab, "Extraction & Detection")
        
        # Frame extraction parameters
        frame_group = QGroupBox("Frame Extraction")
        frame_layout = QFormLayout(frame_group)
        
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(1, 100)
        self.sample_rate_spin.setValue(10)
        self.sample_rate_spin.setToolTip("Extract every Nth frame")
        frame_layout.addRow("Sample Rate:", self.sample_rate_spin)
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["T - Thermal", "W - Wide"])
        frame_layout.addRow("Camera:", self.camera_combo)
        
        extract_detect_layout.addWidget(frame_group)
        
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
        
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.5)
        self.confidence_spin.setToolTip("Minimum detection confidence")
        detection_layout.addRow("Min Confidence:", self.confidence_spin)
        
        extract_detect_layout.addWidget(detection_group)
        extract_detect_layout.addStretch()
        
        # ----- Sub-Tab 2: Position Correction -----
        correction_tab = QWidget()
        correction_tab_layout = QVBoxLayout(correction_tab)
        config_sub_tabs.addTab(correction_tab, "Position Correction")
        
        correction_group = QGroupBox("Position Correction Factors")
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
        rot_label = QLabel("Rotation (pitch, roll, yaw):")
        correction_layout.addRow(rot_label)
        
        rot_row = QHBoxLayout()
        self.rot_x_spin = QDoubleSpinBox()
        self.rot_x_spin.setRange(-180, 180)
        self.rot_x_spin.setValue(0.0)
        self.rot_x_spin.setDecimals(3)
        rot_row.addWidget(QLabel("Pitch:"))
        rot_row.addWidget(self.rot_x_spin)
        
        self.rot_y_spin = QDoubleSpinBox()
        self.rot_y_spin.setRange(-180, 180)
        self.rot_y_spin.setValue(0.0)
        self.rot_y_spin.setDecimals(3)
        rot_row.addWidget(QLabel("Roll:"))
        rot_row.addWidget(self.rot_y_spin)
        
        self.rot_z_spin = QDoubleSpinBox()
        self.rot_z_spin.setRange(-180, 180)
        self.rot_z_spin.setValue(0.0)
        self.rot_z_spin.setDecimals(3)
        rot_row.addWidget(QLabel("Yaw:"))
        rot_row.addWidget(self.rot_z_spin)
        correction_layout.addRow(rot_row)
        
        correction_tab_layout.addWidget(correction_group)
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
        self.refresh_trackers_btn.setToolTip("Refresh available trackers\n(Use after installing boxmot or georef-tracker)")
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
        
        fov_tab_layout.addWidget(fov_group)
        fov_tab_layout.addStretch()
        
        # ----- Sub-Tab 5: Orthomosaic -----
        ortho_tab = QWidget()
        ortho_tab_layout = QVBoxLayout(ortho_tab)
        config_sub_tabs.addTab(ortho_tab, "Orthomosaic")
        
        ortho_group = QGroupBox("Orthomosaic Generation")
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
        
        ortho_tab_layout.addWidget(ortho_group)
        ortho_tab_layout.addStretch()
        
        # =====================================================================
        # MAIN TAB 3: PROCESSING
        # =====================================================================
        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)
        main_tabs.addTab(processing_tab, "Processing")
        
        # Step buttons
        steps_group = QGroupBox("Processing Steps")
        steps_btn_layout = QVBoxLayout(steps_group)
        
        # ----- Step 1: Extract Frames -----
        step1_row = QHBoxLayout()
        self.extract_btn = QPushButton("1. Extract Frames")
        self.extract_btn.clicked.connect(self.run_extract_frames)
        self.extract_status = QLabel("⚪ Not started")
        step1_row.addWidget(self.extract_btn)
        step1_row.addWidget(self.extract_status)
        steps_btn_layout.addLayout(step1_row)
        
        # ----- Step 2: Generate Flight Route -----
        step2_row = QHBoxLayout()
        self.flight_route_btn = QPushButton("2. Generate Flight Route")
        self.flight_route_btn.clicked.connect(self.run_flight_route)
        self.flight_route_btn.setToolTip("Generate flight route polyline from camera positions")
        self.flight_route_status = QLabel("⚪ Not started")
        step2_row.addWidget(self.flight_route_btn)
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
        self.detect_status = QLabel("⚪ Not started")
        step3_row.addWidget(self.detect_btn)
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
        
        # ----- Step 6: Calculate Field of View -----
        step6_row = QHBoxLayout()
        self.calculate_fov_btn = QPushButton("6. Calculate Field of View")
        self.calculate_fov_btn.clicked.connect(self.run_calculate_fov)
        self.calculate_fov_btn.setToolTip("Calculate and save camera FoV footprints for each frame")
        self.calculate_fov_status = QLabel("⚪ Not started")
        step6_row.addWidget(self.calculate_fov_btn)
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
        self.ortho_status = QLabel("⚪ Not started")
        step7_row.addWidget(self.ortho_btn)
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
        self.export_geotiffs_status = QLabel("⚪ Not started")
        step8_row.addWidget(self.export_geotiffs_btn)
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
        
        processing_layout.addWidget(steps_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        processing_layout.addWidget(self.progress_bar)
        
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
        # Parse CRS
        crs_text = self.target_crs_combo.currentText()
        epsg = int(crs_text.split(":")[1].split(" ")[0])
        
        return {
            # Inputs
            "video_paths": [p.strip() for p in self.video_paths_edit.text().split(",") if p.strip()],
            "srt_paths": [p.strip() for p in self.srt_paths_edit.text().split(",") if p.strip()],
            "airdata_path": self.airdata_path_edit.text(),
            "dem_path": self.dem_path_edit.text(),
            "calibration_path": self.calibration_path_edit.text(),
            "correction_path": self.correction_path_edit.text(),
            
            # Output
            "target_folder": self.target_folder_edit.text(),
            "target_epsg": epsg,
            
            # Frame extraction
            "sample_rate": self.sample_rate_spin.value(),
            "camera": "T" if self.camera_combo.currentIndex() == 0 else "W",
            
            # Detection
            "model_path": self.model_path_edit.text() or None,
            "min_confidence": self.confidence_spin.value(),
            
            # Correction factors
            "translation": {
                "x": self.trans_x_spin.value(),
                "y": self.trans_y_spin.value(),
                "z": self.trans_z_spin.value()
            },
            "rotation": {
                "x": self.rot_x_spin.value(),
                "y": self.rot_y_spin.value(),
                "z": self.rot_z_spin.value()
            },
            
            # Tracking
            "tracker_id": self._get_selected_tracker_id(),
            "reid_model": self._get_selected_reid_model(),
            "custom_reid_path": self.custom_reid_path_edit.text() if hasattr(self, 'custom_reid_path_edit') else "",
            "tracker_params_json": self.tracker_params_edit.toPlainText() if hasattr(self, 'tracker_params_edit') else "",
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
            
            # Field of View
            "use_fov_mask": self.use_fov_mask_check.isChecked(),
            "fov_mask_path": self.fov_mask_path_edit.text() if hasattr(self, 'fov_mask_path_edit') else "",
            "mask_simplify_epsilon": self.mask_simplify_spin.value() if hasattr(self, 'mask_simplify_spin') else 2.0
        }
        
    def validate_inputs(self, required_fields: list) -> bool:
        """Validate that required input fields are filled."""
        config = self.get_config()
        missing = []
        
        field_labels = {
            "video_paths": "Video files",
            "srt_paths": "SRT files",
            "airdata_path": "AirData CSV",
            "dem_path": "DEM file",
            "calibration_path": "Calibration file",
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
    def browse_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "", "Video Files (*.mp4 *.MP4 *.avi *.mov)")
        if files:
            self.video_paths_edit.setText(", ".join(files))
            
            # Get the folder of the first video
            video_folder = os.path.dirname(files[0])
            
            # Auto-populate SRT paths
            srts = [f.replace(".MP4", ".SRT").replace(".mp4", ".srt") for f in files]
            existing_srts = [s for s in srts if os.path.exists(s)]
            if existing_srts:
                self.srt_paths_edit.setText(", ".join(existing_srts))
                self.log(f"Auto-detected {len(existing_srts)} SRT file(s)")
            
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
            
            # Auto-detect calibration JSON (look for file with "calib" in name)
            if not self.calibration_path_edit.text():
                json_files = [f for f in folder_files if f.lower().endswith('.json')]
                calib_files = [f for f in json_files if 'calib' in f.lower()]
                if calib_files:
                    calib_path = os.path.join(video_folder, calib_files[0])
                    self.calibration_path_edit.setText(calib_path)
                    self.log(f"Auto-detected calibration: {calib_files[0]}")
                else:
                    # Fallback: check JSON files for calibration structure
                    dem_metadata = os.path.basename(self.dem_metadata_path_edit.text()) if self.dem_metadata_path_edit.text() else ""
                    for json_file in json_files:
                        # Skip DEM metadata and correction files
                        if json_file == dem_metadata or json_file.lower() == 'correction.json':
                            continue
                        json_path = os.path.join(video_folder, json_file)
                        try:
                            with open(json_path, 'r') as f:
                                data = json.load(f)
                            if 'mtx' in data or 'camera_matrix' in data or 'dist' in data:
                                self.calibration_path_edit.setText(json_path)
                                self.log(f"Auto-detected calibration: {json_file}")
                                break
                        except:
                            continue
            
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
                
    def browse_srts(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select SRT Files", "", "SRT Files (*.srt *.SRT)")
        if files:
            self.srt_paths_edit.setText(", ".join(files))
            
    def browse_airdata(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select AirData CSV", "", "CSV Files (*.csv)")
        if file:
            self.airdata_path_edit.setText(file)
            
    def browse_dem(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select DEM GLTF File", "", "GLTF Files (*.gltf *.glb)")
        if file:
            self.dem_path_edit.setText(file)
            # Auto-initialize DEM metadata path by changing suffix to .json
            json_path = file.replace(".gltf", ".json").replace(".glb", ".json")
            if os.path.exists(json_path):
                self.dem_metadata_path_edit.setText(json_path)
                self.log(f"Auto-detected DEM metadata: {json_path}")
            else:
                # Try common naming patterns
                for suffix in ["_mesh.json", "_dem.json", "_metadata.json"]:
                    base = file.rsplit(".", 1)[0]
                    alt_path = base + suffix
                    if os.path.exists(alt_path):
                        self.dem_metadata_path_edit.setText(alt_path)
                        self.log(f"Auto-detected DEM metadata: {alt_path}")
                        break
            
    def browse_calibration(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Calibration JSON", "", "JSON Files (*.json)")
        if file:
            self.calibration_path_edit.setText(file)
            
    def browse_correction(self):
        """Browse for correction.json file and load its values."""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Correction JSON", "", "JSON Files (*.json)")
        if file:
            self.correction_path_edit.setText(file)
            self.load_correction_values(file)
            
    def load_correction_values(self, correction_path: str):
        """Load correction values from JSON file and populate the spin boxes.
        
        :param correction_path: Path to the correction.json file
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
            
            # Load rotation values
            if 'rotation' in correction:
                rot = correction['rotation']
                if 'x' in rot:
                    self.rot_x_spin.setValue(float(rot['x']))
                if 'y' in rot:
                    self.rot_y_spin.setValue(float(rot['y']))
                if 'z' in rot:
                    self.rot_z_spin.setValue(float(rot['z']))
            
            self.log(f"Loaded correction values from: {os.path.basename(correction_path)}")
            
        except Exception as e:
            self.log(f"Warning: Could not load correction.json: {e}")
            
    def browse_target_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Target Folder")
        if folder:
            self.target_folder_edit.setText(folder)
            
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
        
    def detect_frame_count(self):
        """Detect the number of available frames from poses.json."""
        config = self.get_config()
        target_folder = config.get("target_folder", "")
        
        if not target_folder:
            QMessageBox.warning(self, "Error", "Please set the target folder first.")
            return
            
        poses_file = os.path.join(target_folder, "poses.json")
        
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
            
            # Update spinbox ranges
            if frame_count > 0:
                self.ortho_start_frame_spin.setRange(0, frame_count - 1)
                self.ortho_end_frame_spin.setRange(0, frame_count - 1)
                self.ortho_end_frame_spin.setValue(frame_count - 1)
                
            self.log(f"Detected {frame_count} frames available for orthomosaic")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to read poses.json: {str(e)}")
            
    # Processing functions
    def run_extract_frames(self):
        """Run frame extraction step."""
        if not self.validate_inputs(["video_paths", "srt_paths", "airdata_path", 
                                     "calibration_path", "target_folder"]):
            return
            
        self.start_worker("extract_frames")
        
    def run_detection(self):
        """Run animal detection step."""
        config = self.get_config()
        
        # Check if frames exist
        frames_folder = config["target_folder"]
        poses_file = os.path.join(frames_folder, "poses.json")
        
        if not os.path.exists(poses_file):
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                "Frame extraction has not been completed.\nPlease run Step 1 first."
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
        
        # Check if poses exist
        poses_file = os.path.join(config["target_folder"], "poses.json")
        
        if not os.path.exists(poses_file):
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                "Frame extraction has not been completed.\nPlease run Step 1 first."
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
        
        # Check if frames exist
        poses_file = os.path.join(config["target_folder"], "poses.json")
        
        if not os.path.exists(poses_file):
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                "Frame extraction has not been completed.\nPlease run Step 1 first."
            )
            return
            
        # Check DEM path
        if not self.validate_inputs(["dem_path"]):
            return
            
        self.start_worker("orthomosaic")
        
    def run_export_geotiffs(self):
        """Run frame GeoTIFF export step."""
        config = self.get_config()
        
        # Check if frames exist
        poses_file = os.path.join(config["target_folder"], "poses.json")
        
        if not os.path.exists(poses_file):
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                "Frame extraction has not been completed.\nPlease run Step 1 first."
            )
            return
            
        # Check DEM path
        if not self.validate_inputs(["dem_path"]):
            return
            
        self.start_worker("export_geotiffs")
    
    def run_flight_route(self):
        """Run flight route generation step."""
        config = self.get_config()
        
        # Check if frames exist
        poses_file = os.path.join(config["target_folder"], "poses.json")
        
        if not os.path.exists(poses_file):
            QMessageBox.warning(
                self,
                "Missing Prerequisites",
                "Frame extraction has not been completed.\nPlease run Step 1 first."
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
        self.update_status(step, "🟡 Running...")
        self.progress_bar.setValue(0)
        
        self.log(f"Starting {step}...")
        self.worker_thread.start()
        
    def on_worker_finished(self, step: str, success: bool):
        """Handle worker completion."""
        self.set_buttons_enabled(True)
        self.worker = None
        
        if success:
            self.update_status(step, "🟢 Completed")
            self.log(f"{step} completed successfully!")
            self.progress_bar.setValue(100)
        else:
            self.update_status(step, "🔴 Failed")
            
    def on_worker_error(self, step: str, error_msg: str):
        """Handle worker error."""
        self.set_buttons_enabled(True)
        self.worker = None
        self.update_status(step, "🔴 Error")
        self.log(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Processing Error", error_msg)
        
    def on_worker_progress(self, value: int):
        """Update progress bar."""
        self.progress_bar.setValue(value)
        
    def update_status(self, step: str, status: str):
        """Update the status label for a step."""
        status_map = {
            "extract_frames": self.extract_status,
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
            "add_flight_route": self.add_flight_route_status
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
        self.ortho_btn.setEnabled(enabled)
        self.export_geotiffs_btn.setEnabled(enabled)
        self.flight_route_btn.setEnabled(enabled)
        self.add_ortho_btn.setEnabled(enabled)
        self.add_geotiffs_btn.setEnabled(enabled)
        self.add_flight_route_btn.setEnabled(enabled)
        
    def add_tracks_to_qgis(self):
        """Add tracked animals as polygon layers to QGIS."""
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
                    if f.endswith(".csv"):
                        track_files.append(os.path.join(root, f))
                        
            if not track_files:
                QMessageBox.warning(self, "No Tracks", "No track files found.")
                self.update_status("add_layers", "🔴 No files")
                return
                
            # Create a group for the layers
            root = QgsProject.instance().layerTreeRoot()
            group = root.addGroup("BAMBI Wildlife Tracks")
            
            total_tracks = 0
            
            for track_file in track_files:
                tracks = self.load_tracks_from_csv(track_file)
                
                if not tracks:
                    continue
                    
                file_basename = os.path.basename(track_file).replace(".csv", "")
                
                for track_id, detections in tracks.items():
                    if len(detections) < 2:
                        continue  # Skip single-detection tracks
                        
                    # Create polygon layer for this track
                    layer_name = f"Track_{file_basename}_{track_id}"
                    layer = QgsVectorLayer("Polygon?crs=" + target_crs.authid(), layer_name, "memory")
                    
                    provider = layer.dataProvider()
                    
                    # Add fields
                    provider.addAttributes([
                        QgsField("track_id", QVariant.Int),
                        QgsField("frame", QVariant.Int),
                        QgsField("confidence", QVariant.Double),
                        QgsField("class_id", QVariant.Int),
                        QgsField("interpolated", QVariant.Int)
                    ])
                    layer.updateFields()
                    
                    # Add features (bounding boxes as polygons)
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
                            track_id,
                            det['frame'],
                            det['confidence'],
                            det['class_id'],
                            det.get('interpolated', 0)
                        ])
                        features.append(feat)
                        
                    provider.addFeatures(features)
                    layer.updateExtents()
                    
                    # Add layer to project and group
                    QgsProject.instance().addMapLayer(layer, False)
                    group.addLayer(layer)
                    total_tracks += 1
                    
            self.log(f"Added {total_tracks} track layers to QGIS")
            self.update_status("add_layers", "🟢 Completed")
            
            # Zoom to tracks extent
            if total_tracks > 0:
                self.iface.mapCanvas().refresh()
                
        except Exception as e:
            self.log(f"Error adding layers: {str(e)}")
            self.update_status("add_layers", "🔴 Error")
            QMessageBox.critical(self, "Error", f"Failed to add tracks: {str(e)}")
            
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
        # Create a group for the layers
        root = QgsProject.instance().layerTreeRoot()
        group = root.addGroup("BAMBI FoV Polygons")
        
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
        # Create a group for the layers
        root = QgsProject.instance().layerTreeRoot()
        group = root.addGroup("BAMBI Frame Detections")
        
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
            self.iface.mapCanvas().setExtent(layer.extent())
            self.iface.mapCanvas().refresh()
            
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
            
            # Create a group for the layers
            root = QgsProject.instance().layerTreeRoot()
            group = root.addGroup("BAMBI Frame GeoTIFFs")
            
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
        """Add flight route layers (polyline and points) to QGIS."""
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
            
            # Create a group for the layers
            root = QgsProject.instance().layerTreeRoot()
            group = root.addGroup("BAMBI Flight Route")
            
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
