# -*- coding: utf-8 -*-
"""The Configuration tab: one sub-tab of settings per kind of step.

Project, Extraction, Flight Route, Detection, Field of View, Light Field and
Orthomosaic are built here, with what their controls do: the time zone of
the flight log, how thermal images are shown, which model detects, which
frames a step takes, the user's own implementations.

Three sub-tabs have a module of their own - ``correction_settings``,
``tracking_settings`` and ``classification_settings`` - and the Environment
sub-tab is in ``bambi_environment_panel``.
"""

import json
import os
from typing import Optional

from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QPushButton,
    QLineEdit, QSpinBox, QDoubleSpinBox, QFileDialog, QLabel, QComboBox,
    QCheckBox, QMessageBox, QDialog,
)

from ..bambi_processing import BambiProcessor
from ..gui_utils import FitTabWidget


class ConfigurationTabMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _build_configuration_tab(self, main_tabs):
        """The Configuration tab: one sub-tab per group of settings."""
        # =====================================================================
        # MAIN TAB 2: CONFIGURATION (with sub-tabs)
        # =====================================================================
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        main_tabs.addTab(config_tab, "Configuration")

        # Create sub-tabs for configuration
        config_sub_tabs = FitTabWidget()
        self.config_sub_tabs = config_sub_tabs
        config_layout.addWidget(config_sub_tabs)

        self._build_project_settings(config_sub_tabs)
        self._build_extraction_settings(config_sub_tabs)
        self._build_flight_route_settings(config_sub_tabs)
        self._build_detection_settings(config_sub_tabs)
        self._build_correction_settings(config_sub_tabs)
        self._build_tracking_settings(config_sub_tabs)
        self._build_classification_settings(config_sub_tabs)
        self._build_fov_settings(config_sub_tabs)
        self._build_alfs_settings(config_sub_tabs)
        self._build_orthomosaic_settings(config_sub_tabs)
        # ----- Sub-Tab: Environment (bambi_environment_panel) -----
        self._build_environment_config_tab(config_sub_tabs)

    def _build_project_settings(self, config_sub_tabs):
        """Configuration > Project."""
        # ----- Sub-Tab: Project -----
        # Settings that describe the survey rather than one processing step,
        # so they come before the steps that consume them.
        project_tab = QWidget()
        project_tab_layout = QVBoxLayout(project_tab)
        config_sub_tabs.addTab(project_tab, "Project")

        # Species are a property of the survey rather than of whichever tool is
        # open, and they are edited here and only here - that is what keeps
        # class ids stable.
        vocabulary_group = QGroupBox("Project Schema")
        vocabulary_layout = QVBoxLayout(vocabulary_group)
        vocabulary_info = QLabel(
            "Species, enums and custom fields for this project. Every tool "
            "picks from these; none of them can invent new ones.")
        vocabulary_info.setWordWrap(True)
        vocabulary_info.setStyleSheet("color: gray; font-size: 10px;")
        vocabulary_layout.addWidget(vocabulary_info)
        self.schema_editor_btn = QPushButton("Edit Project Schema…")
        self.schema_editor_btn.clicked.connect(self.open_schema_dialog)
        vocabulary_layout.addWidget(self.schema_editor_btn)

        project_tab_layout.addWidget(vocabulary_group)

        # How detections and tracks are drawn on the map.
        symbols_group = QGroupBox("Map Symbols")
        symbols_layout = QFormLayout(symbols_group)
        symbols_info = QLabel(
            "Zoomed in, every tracked animal's bounding box has a colour of "
            "its own. Zoomed out, where a box is too small to see, a pin "
            "takes its place - in the colour of the species (set in the "
            "Project Schema), with the sex inside it and the age as a "
            "badge.")
        symbols_info.setWordWrap(True)
        symbols_info.setStyleSheet("color: gray; font-size: 10px;")
        symbols_layout.addRow(symbols_info)

        self.map_pins_check = QCheckBox("Show pins when zoomed out")
        self.map_pins_check.setChecked(True)
        self.map_pins_check.setToolTip(
            "Off: the bounding boxes are drawn at every scale.")
        symbols_layout.addRow("", self.map_pins_check)

        self.map_pin_scale_spin = QSpinBox()
        self.map_pin_scale_spin.setRange(10, 1000000)
        self.map_pin_scale_spin.setSingleStep(100)
        self.map_pin_scale_spin.setValue(500)
        self.map_pin_scale_spin.setPrefix("1:")
        self.map_pin_scale_spin.setToolTip(
            "Map scale at which the boxes give way to pins. At 1:500 a "
            "one-metre animal is two millimetres on screen. The scale can "
            "also be changed per layer in the layer's Symbology.")
        self.map_pins_check.toggled.connect(
            self.map_pin_scale_spin.setEnabled)
        symbols_layout.addRow("Pins beyond:", self.map_pin_scale_spin)

        self.map_restyle_btn = QPushButton("Apply to Layers in QGIS")
        self.map_restyle_btn.setToolTip(
            "Apply the species colours and these settings to the detection "
            "and track layers already added. New layers take them anyway.")
        self.map_restyle_btn.clicked.connect(
            lambda _checked=False: self.restyle_animal_layers())
        symbols_layout.addRow("", self.map_restyle_btn)

        project_tab_layout.addWidget(symbols_group)
        project_tab_layout.addStretch()

    def _build_extraction_settings(self, config_sub_tabs):
        """Configuration > Extraction."""
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
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'plugins'
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

        self.flag_duplicates_check = QCheckBox(
            "Keep duplicated thermal frames (flagged)")
        self.flag_duplicates_check.setChecked(True)
        self.flag_duplicates_check.setToolTip(
            "Thermal video only. A thermal camera closes its shutter now and "
            "then to recalibrate; while it is closed the video repeats the "
            "previous image.\n\n"
            "Ticked: these frozen frames stay in the frame list, marked "
            "\"duplicated\", and every image-based step skips them. The frame "
            "index then equals the video's frame number (one video, no skip "
            "or sampling) - which is what TRex tracklets and anything else "
            "that refers to the video by frame number need.\n\n"
            "Unticked: they are dropped, as before 9.0.0, and all later "
            "frames move up.\n\n"
            "Changing this renumbers the frames: re-run the steps that were "
            "computed from the previous extraction.")
        frame_ext_layout.addRow("Duplicates (thermal video):",
                                self.flag_duplicates_check)

        self.grid_convergence_check = QCheckBox(
            "Turn headings from true north to the map's north")
        self.grid_convergence_check.setChecked(True)
        self.grid_convergence_check.setToolTip(
            "A drone reports where its camera looks as a heading from true "
            "north. The north of the map - of the project's CRS - is true "
            "north on the projection's central meridian only; away from it "
            "the two differ by the grid convergence: 1.4 degrees at "
            "Salzburg and 3.9 degrees in Vorarlberg in UTM zone 33.\n\n"
            "Ticked: every frame's pose gets the angle at its position, "
            "and its heading is used as what it is. The log of the "
            "extraction says how large the angle is.\n\n"
            "Unticked: the heading is used as a bearing on the map, as up "
            "to 9.0 - everything is then turned by that angle about the "
            "point below the drone (1.4 m at the corner of a frame from "
            "80 m up, for two degrees).\n\n"
            "A rotation correction (z) that was calibrated on frames "
            "extracted without this contains the angle and has to be "
            "reduced by it; the extraction's log names the value.\n\n"
            "Takes effect when frames are extracted; frames extracted "
            "before keep their headings.")
        frame_ext_layout.addRow("Headings:", self.grid_convergence_check)

        self.preserve_aspect_ratio_check = QCheckBox("Preserve original aspect ratio (otherwise square 1:1 format)")
        self.preserve_aspect_ratio_check.setChecked(False)
        self.preserve_aspect_ratio_check.setToolTip(
            "When enabled, extracted frames keep the original width/height ratio instead of being cropped to a square.")
        frame_ext_layout.addRow("Frame format:", self.preserve_aspect_ratio_check)

        from ..core.heading import HEADING_SOURCE_LABELS
        self.heading_source_combo = QComboBox()
        for source, label in HEADING_SOURCE_LABELS.items():
            self.heading_source_combo.addItem(label, source)
        self.heading_source_combo.setToolTip(
            "Where the camera yaw of every frame comes from.\n"
            "Compass (flight log): the drone body's heading - the historic default.\n"
            "Gimbal (flight log): use this when the gimbal holds its heading while the\n"
            "drone yaws, e.g. on fast, agile flights; otherwise frames rotate with the body.\n"
            "Gimbal (SRT): the gimbal yaw written for each video frame (video only).\n"
            "Drone (SRT): only some DJI models write it; extraction stops if it is missing.\n"
            "Flight log and SRT can disagree by a constant offset - check against a basemap.\n"
            "Photo mode has no SRT: the SRT options fall back to the flight-log values.")
        frame_ext_layout.addRow("Heading source:", self.heading_source_combo)

        self.airdata_padding_spin = QDoubleSpinBox()
        self.airdata_padding_spin.setRange(0.0, 30.0)
        self.airdata_padding_spin.setSingleStep(0.5)
        self.airdata_padding_spin.setDecimals(1)
        self.airdata_padding_spin.setSuffix(" s")
        self.airdata_padding_spin.setValue(0.0)
        self.airdata_padding_spin.setToolTip(
            "Video mode only. Flight log kept before and after the rows AirData flags as\n"
            "recording. DJI can set that flag about 1 s after the video starts, so the first\n"
            "frames' positions are extrapolated (metres off on fast flights). 3 s is\n"
            "recommended; 0 keeps the poses of earlier versions. A larger margin gives\n"
            "the same poses, it only has to cover the delay.")
        self.airdata_padding_auto_check = QCheckBox("Auto")
        # New projects measure the margin; projects saved without this
        # setting load it unticked and keep their fixed value (see schema).
        self.airdata_padding_auto_check.setChecked(True)
        self.airdata_padding_auto_check.setToolTip(
            "Measure the margin per camera before extraction: the time the video (SRT)\n"
            "extends beyond the flight log rows flagged as recording, plus 1 s.\n"
            "The value used is written to the log and to poses.json.")
        self.airdata_padding_auto_check.toggled.connect(
            lambda auto: self.airdata_padding_spin.setEnabled(not auto))
        self.airdata_padding_spin.setEnabled(False)
        airdata_padding_row = QHBoxLayout()
        airdata_padding_row.addWidget(self.airdata_padding_spin, 1)
        airdata_padding_row.addWidget(self.airdata_padding_auto_check)
        self._airdata_padding_info_btn = self._make_info_button(
            self._show_airdata_padding_info)
        airdata_padding_row.addWidget(self._airdata_padding_info_btn)
        frame_ext_layout.addRow("Flight log margin around recording:", airdata_padding_row)

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

        # The detected offset follows the files it is read from. It was
        # only worked out when "auto" was ticked, so the label kept showing
        # the previous flight's offset after other files were chosen.
        self._tz_refresh_timer = QTimer(self)
        self._tz_refresh_timer.setSingleShot(True)
        self._tz_refresh_timer.setInterval(400)
        self._tz_refresh_timer.timeout.connect(
            self._refresh_auto_timezone_offset)
        for edit in (self.rgb_srt_paths_edit, self.thermal_srt_paths_edit,
                     self.airdata_path_edit, self.thermal_photo_dir_edit,
                     self.rgb_photo_dir_edit):
            edit.textChanged.connect(
                lambda _text: self._tz_refresh_timer.start())
        self.video_mode_check.toggled.connect(
            lambda _on: self._tz_refresh_timer.start())

        self.tz_manual_hint = QLabel(
            'Timezone of camera/drone clock when recordings were made '
            '(consider daylight saving time)!'
        )
        self.tz_manual_hint.setWordWrap(True)
        self.tz_manual_hint.setVisible(False)
        tz_group_layout.addWidget(self.tz_manual_hint)

        extraction_tab_layout.addWidget(tz_group)

        # Thermal visualisation (photo mode only - hidden in video mode)
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

    def _build_flight_route_settings(self, config_sub_tabs):
        """Configuration > Flight Route."""
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

    def _build_detection_settings(self, config_sub_tabs):
        """Configuration > Detection."""
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

        self.detection_projection_combo = QComboBox()
        self.detection_projection_combo.addItems([
            "Perspective (camera frames)",
            "Orthorectified (GeoTIFF)",
            "ALFS (light-field integrals)",
        ])
        self.detection_projection_combo.setToolTip(
            "Which imagery the detector looks at - the built-in one and a "
            "custom one alike.\n\n"
            "Perspective: the extracted camera frames, as always.\n\n"
            "Orthorectified: the per-frame GeoTIFFs, for models trained on "
            "orthophotos. 'Export Frames as GeoTIFF' has to have run for "
            "the camera (frames without a GeoTIFF are skipped), and the DEM "
            "is needed. The detector's boxes are raster pixels; each is "
            "placed in its camera frame through the GeoTIFF's "
            "georeference, the DEM and the frame's camera, because a "
            "detection is stored as a box in the extracted frame - the "
            "picture the labelling tool, the inspector, the video creator "
            "and the classifiers' crops use. Everything after detection "
            "works as before.\n\n"
            "ALFS: the light-field integrals of ALFS sampling mode "
            "('Generate ALFS' with sampling enabled), in which occluding "
            "vegetation is blurred away - for detectors trained on ALFS "
            "imagery. An integral exists per central frame, so only those "
            "frames are searched; the boxes are placed in the central "
            "frame's camera image the same way.\n\n"
            "The default BAMBI models were trained on perspective frames.")
        detection_layout.addRow("Projection:", self.detection_projection_combo)

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

        # Custom detectors (core.custom_models, 9.0.0): a class from the
        # user's own .py file in place of the Ultralytics model, per camera.
        from ..bambi_custom_impl_widget import (
            DEPENDENCY_NOTE, FOLDER_NOTE, CustomImplementationBox)
        from ..core import custom_models as _custom_models

        self._custom_implementations = {}
        self._custom_boxes = {}
        custom_det_group = QGroupBox("Custom detector")
        custom_det_layout = QVBoxLayout(custom_det_group)
        custom_det_info = QLabel(
            "Any detector can stand in for the built-in Ultralytics YOLO "
            "model: write a class with <code>setup(context)</code> and "
            "<code>run(image, frame)</code> in a .py file and select it "
            "here. The sample implementations show the interface - the "
            "built-in detector itself, Microsoft's Overhead Wildlife "
            "Locator (OWL) and DroneMegaDetector. " + FOLDER_NOTE + " "
            + DEPENDENCY_NOTE)
        custom_det_info.setWordWrap(True)
        custom_det_info.setTextFormat(Qt.TextFormat.RichText)
        custom_det_info.setStyleSheet("color: gray; font-size: 10px;")
        custom_det_layout.addWidget(custom_det_info)
        for key, title, camera in (
                ("detector_t", "Thermal detector", "thermal"),
                ("detector_w", "RGB detector", "rgb")):
            box = CustomImplementationBox(
                title, _custom_models.DETECTOR, camera)
            box.changed.connect(self._on_custom_implementation_changed)
            self._custom_boxes[key] = box
            custom_det_layout.addWidget(box)
        self.custom_samples_btn = QPushButton(
            "Download sample implementations…")
        self.custom_samples_btn.setToolTip(
            "Save ready-to-edit sample files for detectors, trackers and "
            "classifiers into a folder of your choice.")
        self.custom_samples_btn.clicked.connect(self.download_custom_samples)
        custom_det_layout.addWidget(self.custom_samples_btn)
        detect_tab_layout.addWidget(custom_det_group)

        # The mapping belongs with the model rather than with the project's
        # species: it describes what these weights report, so a different model
        # needs a different mapping even for the same species.
        mapping_group = QGroupBox("Class Mapping")
        mapping_layout = QVBoxLayout(mapping_group)
        mapping_info = QLabel(
            "Map the classes this detector reports onto the project's species. "
            "Unmapped classes count as 'animal', so a single-class model needs "
            "nothing here.")
        mapping_info.setWordWrap(True)
        mapping_info.setStyleSheet("color: gray; font-size: 10px;")
        mapping_layout.addWidget(mapping_info)

        self.class_mapping_btn = QPushButton("Edit Class Mapping…")
        self.class_mapping_btn.clicked.connect(self.open_class_mapping_dialog)
        mapping_layout.addWidget(self.class_mapping_btn)
        detect_tab_layout.addWidget(mapping_group)

        detect_tab_layout.addStretch()

    def _build_fov_settings(self, config_sub_tabs):
        """Configuration > Field of View."""
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

    def _build_alfs_settings(self, config_sub_tabs):
        """Configuration > ALFS/GeoTiff."""
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

        # Alternative to the ground resolution: give the raster size directly.
        # With a large DEM a fine m/px explodes into an unusable pixel count,
        # and per-frame GeoTIFFs are easier to consume on a fixed canvas.
        self.alfs_fixed_size_check = QCheckBox("Use fixed render size instead")
        self.alfs_fixed_size_check.setChecked(False)
        self.alfs_fixed_size_check.setToolTip(
            "Ignore the ground resolution and render into a raster of exactly\n"
            "the given pixel size. The ground resolution then follows from the\n"
            "extent being covered instead of the other way round."
        )
        self.alfs_fixed_size_check.stateChanged.connect(self.toggle_alfs_fixed_size)
        alfs_layout.addRow("Render Size:", self.alfs_fixed_size_check)

        self.alfs_render_size_widget = QWidget()
        render_size_layout = QHBoxLayout(self.alfs_render_size_widget)
        render_size_layout.setContentsMargins(0, 0, 0, 0)

        self.alfs_render_width_spin = QSpinBox()
        self.alfs_render_width_spin.setRange(16, 16384)
        self.alfs_render_width_spin.setValue(2048)
        self.alfs_render_width_spin.setSingleStep(256)
        self.alfs_render_width_spin.setToolTip("Output raster width in pixels")
        render_size_layout.addWidget(QLabel("Width:"))
        render_size_layout.addWidget(self.alfs_render_width_spin)

        self.alfs_render_height_spin = QSpinBox()
        self.alfs_render_height_spin.setRange(16, 16384)
        self.alfs_render_height_spin.setValue(2048)
        self.alfs_render_height_spin.setSingleStep(256)
        self.alfs_render_height_spin.setToolTip("Output raster height in pixels")
        render_size_layout.addWidget(QLabel("Height:"))
        render_size_layout.addWidget(self.alfs_render_height_spin)

        self.alfs_render_size_widget.setEnabled(False)
        alfs_layout.addRow("", self.alfs_render_size_widget)

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

        # The GeoTIFF export's only setting; it sat on the step row, which is
        # where the run buttons live rather than their configuration.
        geotiff_group = QGroupBox("Frame GeoTIFF Export")
        geotiff_layout = QFormLayout(geotiff_group)

        self.geotiff_edge_erosion_spin = QSpinBox()
        self.geotiff_edge_erosion_spin.setRange(0, 10)
        self.geotiff_edge_erosion_spin.setValue(2)
        self.geotiff_edge_erosion_spin.setFixedWidth(55)
        self.geotiff_edge_erosion_spin.setToolTip(
            "Shrink each frame's valid footprint by this many pixels before\n"
            "saving, to remove the dark antialiased rim at the image border\n"
            "that otherwise shows as seams in the orthomosaic. 0 disables it."
        )
        geotiff_layout.addRow("Edge erosion (px):", self.geotiff_edge_erosion_spin)

        alfs_tab_layout.addWidget(geotiff_group)
        alfs_tab_layout.addStretch()

    def _build_orthomosaic_settings(self, config_sub_tabs):
        """Configuration > Orthomosaic."""
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

    @staticmethod
    def _compute_timezone_offset(tz_name: str) -> Optional[float]:
        """Return the current UTC offset in hours for a given IANA timezone name."""
        from ..core.timezone_detection import timezone_offset_hours
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
        if not self.tz_auto_check.isChecked():
            return
        offset = self._compute_auto_timezone_offset()
        if offset is not None:
            sign = "+" if offset >= 0 else ""
            self.tz_auto_result_label.setText(f"Detected offset: UTC{sign}{offset:g}h")
        else:
            self.tz_auto_result_label.setText(
                "(could not detect - check SRT/photo dir & AirData)")

    def _get_timezone_offset(self) -> float:
        """Return the timezone offset in hours, auto-detected or manual."""
        if self.tz_auto_check.isChecked():
            offset = self._compute_auto_timezone_offset()
            if offset is not None:
                return offset
            self.log("Warning: Auto timezone detection failed, falling back to manual timezone")
        return self._compute_timezone_offset(self.timezone_combo.currentText().strip()) or 0.0

    def _timezone_inputs(self):
        """``(srt paths, photo folders, flight log, video mode)`` - what the
        offset is detected from. The SRT files of *either* camera: only the
        RGB ones used to be read, so a thermal-only flight had no detection
        at all."""
        def paths(edit):
            return [p.strip() for p in edit.text().split(",") if p.strip()]

        srt = paths(self.rgb_srt_paths_edit) + paths(self.thermal_srt_paths_edit)
        photos = [folder for folder in (
            self.thermal_photo_dir_edit.text().strip(),
            self.rgb_photo_dir_edit.text().strip()) if folder]
        return (srt, photos, self.airdata_path_edit.text().strip(),
                self.video_mode_check.isChecked())

    def _compute_auto_timezone_offset(self) -> Optional[float]:
        """Auto-detect timezone offset.

        In video mode: compares SRT frame timestamps with AirData isVideo UTC timestamps.
        In photo mode (or as fallback): compares photo EXIF timestamps with AirData isPhoto UTC timestamps.
        Returns the offset in hours or None if detection fails.

        The answer is kept until one of the files it was read from changes:
        every query of the form asks for it, and reading all SRT files and
        the flight log each time made the form sluggish.
        """
        from ..core import timezone_detection

        srt, photos, airdata, video_mode = self._timezone_inputs()
        signature = (video_mode,) + timezone_detection.inputs_signature(
            srt + photos + [airdata])
        cached = getattr(self, "_timezone_cache", None)
        if cached is None or cached[0] != signature:
            cached = (signature, timezone_detection.detect_offset(
                srt, photos, airdata, video_mode))
            self._timezone_cache = cached
        return cached[1]

    def _collect_airdata_hours(self, flag_column_lower: str) -> Optional[list]:
        """Read UTC hours from AirData rows where the flag column is truthy."""
        from ..core.timezone_detection import airdata_utc_hours
        return airdata_utc_hours(
            self.airdata_path_edit.text().strip(), flag_column_lower)

    def _show_airdata_padding_info(self):
        """Info popup for the flight log margin (video mode)."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Flight log margin around recording")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(
            "<b>Flight log margin around recording</b> (video mode only)<br><br>"
            "<b>Why it is needed</b><br>"
            "The position and orientation of every video frame are interpolated from "
            "the flight log (AirData). Only the part of the log marked as recording is "
            "used. DJI often sets that mark about a second after the video has started, "
            "so the first frames lie before the log that is kept. Their positions are "
            "then extrapolated, which can be several metres off on fast flights. The "
            "same can happen at the end of the video.<br><br>"
            "<b>What it does</b><br>"
            "The margin keeps that much extra log before and after the recording, so "
            "every frame lies between two real log rows. It does not add frames, and a "
            "larger margin gives the same result, so it only has to cover the delay.<br><br>"
            "<b>Auto</b> (recommended)<br>"
            "Before extraction, the plugin compares the video's timestamps (SRT) with "
            "the recording mark in the flight log, and uses the delay plus 1 s. The "
            "value used is written to the log panel.<br><br>"
            "<b>Fixed value</b><br>"
            "Untick Auto to set the margin yourself; 3 s covers the usual delay. "
            "0 s gives the results of plugin versions before 7.1, which is why projects "
            "saved with them keep it."
        )
        msg.exec()

    def open_class_mapping_dialog(self) -> bool:
        """Map the detector's raw classes onto the project's species."""
        from ..bambi_class_mapping_dialog import BambiClassMappingDialog

        folder = self.target_folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(
                self, "Class Mapping",
                "Select a target folder first - the mapping belongs to a "
                "project.")
            return False

        dialog = BambiClassMappingDialog(folder, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False

        if dialog.rows_changed:
            # Boxes did not move, so nothing upstream is invalidated; only the
            # species-dependent analytics are now out of date.
            self.log(f"Class mapping applied: {dialog.rows_changed} detection(s) "
                     "re-assigned to a different species")
            QMessageBox.information(
                self, "Class Mapping",
                f"{dialog.rows_changed} stored detection(s) were re-assigned. "
                "The boxes are unchanged, so geo-referencing and tracking stay "
                "valid; re-run the survey analytics to pick up the new species.")
        return True

    def open_schema_dialog(self, initial_tab: int = 0) -> bool:
        """Open the Project Schema editor for the current target folder.

        Returns True when the user accepted, so callers that show a species or
        enum combo can repopulate it.
        """
        from ..bambi_schema_dialog import BambiSchemaDialog

        folder = self.target_folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(
                self, "Project Schema",
                "Select a target folder first - the schema belongs to a "
                "project.")
            return False

        from ..core import groups

        distances_before = groups.species_distances(folder)
        dialog = BambiSchemaDialog(folder, parent=self, initial_tab=initial_tab)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        if accepted:
            # A species may have a new colour: the layers on the map follow.
            self.restyle_animal_layers(quiet=True)
            if groups.differ(distances_before,
                             groups.species_distances(folder)):
                self._offer_group_distances_to_other_flights(folder)
        return accepted

    def _offer_group_distances_to_other_flights(self, folder: str) -> None:
        """The schema belongs to a flight, and a survey takes every
        flight's own group distances - so one that was just changed here
        applies to this flight only, unless it is carried over."""
        import sqlite3
        from ..core import flights as flights_core
        from ..core import groups

        here = flights_core.normalise(folder)
        others = []
        for flight in self._flights():
            other = flight.get("target_folder") or ""
            if not other or flights_core.normalise(other) == here:
                continue
            if os.path.isdir(other):
                others.append(other)
        if not others:
            return
        answer = QMessageBox.question(
            self, "Group distances",
            "The group distances of this flight were changed. A survey takes "
            "each flight's own.\n\nUse them for the other "
            f"{len(others)} flight(s) of this project as well?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes)
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            changed = groups.copy_distances(folder, others)
        except sqlite3.OperationalError as exc:
            QMessageBox.warning(
                self, "Group distances",
                "A flight's files are in use - a step is probably writing "
                f"its results - so not every flight was changed ({exc}). "
                "Try again in a moment.")
            return
        self.log(f"Group distances carried over to {len(changed)} other "
                 "flight(s).")

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
        from ..bambi_thermal import Thermal
        thermal = Thermal(dtype=np.float32)
        return thermal.parse, thermal.close

    def _edit_thermal_vis_curve(self):
        from ..bambi_curve_widget import CurveEditorDialog
        from ..core.thermal_curve import ThermalCurve
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

    def toggle_alfs_fixed_size(self, state):
        """Swap between the ground-resolution and fixed-pixel-size inputs.

        The two are alternatives, so only the active one stays editable -
        a greyed-out m/px field is what tells the user it no longer applies.
        """
        fixed = bool(state)
        self.alfs_render_size_widget.setEnabled(fixed)
        self.alfs_resolution_spin.setEnabled(not fixed)

    def toggle_detect_frame_range(self, state):
        """Toggle the detection frame range controls based on checkbox state."""
        self.detect_frame_range_widget.setEnabled(not state)

    def toggle_fov_frame_range(self, state):
        """Toggle the FoV frame range controls based on checkbox state."""
        self.fov_frame_range_widget.setEnabled(not state)

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

            self.log(f"Detected {frame_count} frames available for processing")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to read poses.json: {str(e)}")

    # ------------------------------------------------------------------
    # Custom implementations (core.custom_models)
    # ------------------------------------------------------------------
    def _on_custom_implementation_changed(self):
        """Mirror the boxes into the saved mapping."""
        self._custom_implementations = {
            key: box.spec() for key, box in
            getattr(self, "_custom_boxes", {}).items()}
        if "tracker" in self._custom_implementations and hasattr(
                self, "custom_tracker_interpolate_check"):
            self._custom_implementations["tracker"]["interpolate"] = \
                self.custom_tracker_interpolate_check.isChecked()

    def _apply_custom_implementations(self):
        """Push the saved mapping back into the boxes after a config load."""
        saved = getattr(self, "_custom_implementations", None) or {}
        if not isinstance(saved, dict):
            saved = {}
        for key, box in getattr(self, "_custom_boxes", {}).items():
            box.set_model_dir(BambiProcessor._get_default_model_dir())
            box.set_spec(saved.get(key) or {})
        if hasattr(self, "custom_tracker_interpolate_check"):
            check = self.custom_tracker_interpolate_check
            blocked = check.blockSignals(True)
            check.setChecked(bool(
                (saved.get("tracker") or {}).get("interpolate", True)))
            check.blockSignals(blocked)
        self._on_custom_implementation_changed()

    def download_custom_samples(self):
        from ..bambi_custom_impl_widget import download_samples
        written = download_samples(self)
        if written:
            self.log(f"Sample implementations written: "
                     f"{os.path.dirname(written[0])}")

    @staticmethod
    def _custom_specs_for_step(step: str, config: dict) -> list:
        """The custom implementations a step would run."""
        custom = config.get("custom_implementations") or {}
        if step == "detection":
            camera = "t" if config.get("detection_camera", "T") == "T" else "w"
            return [custom.get(f"detector_{camera}")]
        if step in ("tracking", "trex_import"):
            return [custom.get("tracker")]
        if step.startswith("classif") or step == "life_stage":
            specs = []
            for spec in (config.get("classification_models") or {}).values():
                specs.append(spec)
                specs.extend((spec.get("species") or {}).values())
            return specs
        return []
