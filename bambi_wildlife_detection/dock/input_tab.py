# -*- coding: utf-8 -*-
"""The Input tab: what a flight is made of.

The videos or photos of the two cameras, their flight logs and calibrations,
the elevation model and the map projection - and what the buttons next to
them do: picking the files, finding the ones that belong together, making an
elevation mesh (downloaded, converted from a GeoTIFF, or flat), and working
out the projection from where the flight took place.

The flight selector at the top of the tab is built here; what it does is in
``flights``.
"""

import json
import os
import tempfile
from typing import Optional, List

from qgis.PyQt.QtCore import QThread
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QPushButton,
    QLineEdit, QSpinBox, QDoubleSpinBox, QFileDialog, QLabel, QComboBox,
    QCheckBox, QMessageBox, QToolButton,
)

from ..austria_dem_downloader import DEMDownloadWorker, GeoTIFFConversionWorker
from ..bambi_calibrations import THERMAL_CALIBRATIONS, RGB_CALIBRATIONS
from ..gui_utils import FitTabWidget
from ..core import quiet


class InputTabMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _build_input_tab(self, main_tabs):
        """The Input tab: videos or photos, flight log, calibrations, the
        elevation model and the target folder."""
        # =====================================================================
        # MAIN TAB 1: INPUT
        # =====================================================================
        # Thermal Video inputs
        input_tab = QWidget()
        input_layout = QVBoxLayout(input_tab)
        main_tabs.addTab(input_tab, "Input")

        # --- Flights ---
        # A QGIS project can hold several flights; exactly one is active, so
        # everything downstream still reads a single configuration.
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

        self.flight_rename_btn = QPushButton("Rename…")
        self.flight_rename_btn.setToolTip(
            "Rename this flight. Its QGIS layer group is renamed with it.")
        self.flight_rename_btn.clicked.connect(self.rename_flight)
        flight_row.addWidget(self.flight_rename_btn)

        self.flight_remove_btn = QPushButton("🗑")
        self.flight_remove_btn.setFixedWidth(30)
        self.flight_remove_btn.setToolTip(
            "Remove this flight from the project. Its target folder and "
            "everything in it are left untouched.")
        self.flight_remove_btn.clicked.connect(self.remove_flight)
        flight_row.addWidget(self.flight_remove_btn)
        flight_box.addLayout(flight_row)

        # Every flight has a name the user gave it (9.0.0). A new flight
        # is added by name; its target folder is chosen below afterwards.
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.flight_name_edit = QLineEdit()
        self.flight_name_edit.setPlaceholderText(
            "Name of a new flight - e.g. area and date")
        self.flight_name_edit.setToolTip(
            "Type the name of a new flight and press + (or Enter) to add it. "
            "Every flight needs a name, also a single one: it titles the "
            "flight's layer group in QGIS and identifies the flight in the "
            "survey tools. Names are unique within the project.")
        self.flight_name_edit.returnPressed.connect(self.add_flight)
        name_row.addWidget(self.flight_name_edit, 1)

        self.flight_add_btn = QPushButton("+")
        self.flight_add_btn.setFixedWidth(30)
        self.flight_add_btn.setToolTip(
            "Add a new flight with this name. Its target folder is then set "
            "under 'Target folder' below.")
        self.flight_add_btn.clicked.connect(self.add_flight)
        name_row.addWidget(self.flight_add_btn)

        self.flight_browse_add_btn = QPushButton("Browse and Add…")
        self.flight_browse_add_btn.setToolTip(
            "Add an existing flight: pick a target folder that has been "
            "processed before. It is added back with its stored "
            "configuration and results.")
        self.flight_browse_add_btn.clicked.connect(self.browse_and_add_flight)
        name_row.addWidget(self.flight_browse_add_btn)
        flight_box.addLayout(name_row)

        flight_info = QLabel(
            "A flight needs a name and its own output target folder. Add a "
            "new flight by name with +, then set its target folder below. "
            "\"Browse and Add\" adds a folder that has already been "
            "processed, with its stored configuration."
        )
        flight_info.setWordWrap(True)
        flight_info.setStyleSheet("color: gray; font-size: 10px;")
        flight_box.addWidget(flight_info)

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
            "The SRT stream is extracted automatically - no separate .srt files needed.\n"
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
        self.target_folder_edit.textChanged.connect(
            lambda _text: self._mark_flight_name())
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
        # for external scripts only. Nothing in the plugin reads them any more,
        # so turning this off no longer changes what the plugin can do.
        self.legacy_text_outputs_check = QCheckBox(
            "Also write legacy text outputs "
            "(.txt / .csv; will be removed in future)")
        self.legacy_text_outputs_check.setChecked(True)
        self.legacy_text_outputs_check.setToolTip(
            "6.0 stores detections, geo-referencing, tracks, field of view and "
            "labels in bambi_{t,w}/*.gpkg, and every step, layer, overlay and "
            "analytic reads them from there.\n\n"
            "The 5.x text files are written alongside so external scripts of "
            "your own keep working. Nothing in the plugin reads them, so "
            "turning this off costs you nothing but disk writes - and it is "
            "how you check whether your own scripts still depend on them.\n\n"
            "They will be removed in a future release."
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
        self.geo_input_tabs = FitTabWidget()
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

        # Anywhere else: the Copernicus DEM (copernicus_dem_downloader.py)
        worldwide_row = QHBoxLayout()
        self.copernicus_dem_btn = QPushButton("Download DEM (worldwide)")
        self.copernicus_dem_btn.setToolTip(
            "Fetch the Copernicus DEM (30 m) under the flight - anywhere on "
            "earth, no account needed - and make the mesh and its metadata "
            "from it. Only the part under the flight is transferred. The "
            "flight's position is read from the AirData CSV or the SRT "
            "files.\n"
            "It is a surface model at 30 m: over forest it lies in the "
            "canopy, and its heights are good to a few metres. Where there "
            "is a terrain model, use that.")
        self.copernicus_dem_btn.clicked.connect(self.download_copernicus_dem)
        worldwide_row.addWidget(self.copernicus_dem_btn)
        worldwide_row.addStretch()
        auto_download_layout.addRow("", worldwide_row)

        worldwide_info_label = QLabel(
            "Outside Austria: the Copernicus DEM, a model of the earth's "
            "land surface at 30 m. A few megabytes at most; written next to "
            "the AirData CSV (or into the target folder) as "
            "copernicus_dem.glb / .json / .tif. At least 150 m of ground is "
            "kept around the flight.")
        worldwide_info_label.setWordWrap(True)
        auto_download_layout.addRow("", worldwide_info_label)

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
        from ..core.flight_files import detect_common_files
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
            # The flight list follows, as when the folder is typed in: a
            # folder that has been processed before is offered with its
            # stored configuration instead of being written over.
            self._on_target_folder_changed()

        # Try to auto-detect CRS from available sources
        self._try_auto_detect_crs_silent()

    def _resolve_embedded_srts(self, video_paths: List[str]) -> List[str]:
        """Extract embedded SRT streams via ffmpeg (core.flight_files).

        Uses a temp directory as a persistent cache - if a .srt file for a
        video already exists there it is reused without re-running ffmpeg.
        """
        from ..core.flight_files import extract_embedded_srts
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
        # only thermal video has frozen frames; the setting itself is kept
        self.flag_duplicates_check.setEnabled(video_mode)

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

    def _flight_gps_positions(self, config=None):
        """``(latitudes, longitudes)`` of where the flight was: from the
        AirData CSV (the rows while a video was recorded), else from the
        SRT files - thermal first, then RGB. Empty where neither tells."""
        config = config or self.get_config()
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
        return lats, lons

    def _generate_flat_surface_mesh(self):
        """Generate flat_surface_dem.glb + .json from AirData/SRT GPS and set as DEM input."""
        config = self.get_config()
        target_folder = config.get("target_folder", "")
        if not target_folder:
            QMessageBox.warning(self, "Missing Target Folder", "Please set the target folder first.")
            return

        flat_msl = self.flat_surface_spin.value()
        lats, lons = self._flight_gps_positions(config)

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
            from ..bambi_processing import BambiProcessor
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
    # Worldwide elevation: the Copernicus DEM
    # =========================================================================
    def download_copernicus_dem(self):
        """Build the flight's elevation model from the Copernicus DEM
        (copernicus_dem_downloader.py)."""
        from ..core.crs_utils import utm_epsg_for_position

        config = self.get_config()
        lats, lons = self._flight_gps_positions(config)
        if not lats:
            QMessageBox.warning(
                self, "Missing GPS Data",
                "Nobody knows where the flight was yet.\n\n"
                "Please select an AirData CSV or SRT files in the input "
                "section first.")
            return
        bounds = (min(lons), min(lats), max(lons), max(lats))

        # Where the files go: beside the AirData CSV, like the Austrian
        # model - or into the target folder where there is no CSV.
        airdata_path = self.airdata_path_edit.text()
        if airdata_path and os.path.exists(airdata_path):
            output_folder = os.path.dirname(airdata_path)
        else:
            output_folder = config.get("target_folder", "")
        if not output_folder:
            QMessageBox.warning(
                self, "Missing Target Folder",
                "Please set the target folder first: the elevation model "
                "is written into it.")
            return

        # The mesh is in metres, in the UTM zone the flight lies in.
        zone = utm_epsg_for_position(
            (bounds[1] + bounds[3]) / 2.0, (bounds[0] + bounds[2]) / 2.0)
        output_crs = self.target_crs_edit.text().strip().upper()
        if output_crs and not output_crs.startswith("EPSG:"):
            output_crs = f"EPSG:{output_crs}"
        if output_crs != f"EPSG:{zone}":
            current = output_crs if self._is_valid_utm_crs(output_crs) \
                else None
            keep = (f"\n\nNo keeps {current}." if current else
                    "\n\nNo cancels: the elevation model needs a UTM CRS.")
            reply = QMessageBox.question(
                self, "Target CRS",
                f"The flight lies in the UTM zone EPSG:{zone}; the target "
                f"CRS is {output_crs or 'not set'}.\n\n"
                f"Set the target CRS to EPSG:{zone}?{keep}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes)
            if reply == QMessageBox.StandardButton.Yes:
                output_crs = f"EPSG:{zone}"
                self.target_crs_edit.setText(output_crs)
            elif current is None:
                return

        padding = float(self.dem_padding_spin.value())
        reply = QMessageBox.question(
            self, "Download DEM",
            "Fetch the Copernicus DEM (30 m) under the flight?\n\n"
            f"Flight: {bounds[1]:.5f}, {bounds[0]:.5f} to "
            f"{bounds[3]:.5f}, {bounds[2]:.5f}\n"
            f"Padding: {max(padding, 150.0):.0f} m\n"
            f"Output CRS: {output_crs}\n"
            f"Output folder: {output_folder}\n\n"
            "It is a surface model: over forest it lies in the canopy, and "
            "its heights are good to a few metres. Where there is a "
            "terrain model of the area, use that.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes)
        if reply != QMessageBox.StandardButton.Yes:
            return

        from ..copernicus_dem_downloader import CopernicusDEMWorker

        self.copernicus_dem_btn.setEnabled(False)
        self.copernicus_dem_btn.setText("Downloading...")
        self.copernicus_dem_thread = QThread()
        self.copernicus_dem_worker = CopernicusDEMWorker(
            bounds, output_folder, output_crs, padding)
        self.copernicus_dem_worker.moveToThread(self.copernicus_dem_thread)
        self.copernicus_dem_thread.started.connect(
            self.copernicus_dem_worker.run)
        self.copernicus_dem_worker.finished.connect(
            self._on_copernicus_dem_finished)
        self.copernicus_dem_worker.progress.connect(
            self._on_copernicus_dem_progress)
        self.copernicus_dem_worker.log.connect(self.log)
        self.copernicus_dem_worker.finished.connect(
            self.copernicus_dem_thread.quit)
        self.copernicus_dem_worker.finished.connect(
            self.copernicus_dem_worker.deleteLater)
        self.copernicus_dem_thread.finished.connect(
            self.copernicus_dem_thread.deleteLater)
        self.copernicus_dem_thread.start()
        self.log("Starting the Copernicus DEM download...")

    def _on_copernicus_dem_progress(self, percent: int):
        self.copernicus_dem_btn.setText(f"Downloading... {percent}%")

    def _on_copernicus_dem_finished(self, success: bool, message: str):
        """The worldwide elevation model is there - or it is not."""
        self.copernicus_dem_btn.setEnabled(True)
        self.copernicus_dem_btn.setText("Download DEM (worldwide)")
        if not success:
            QMessageBox.warning(
                self, "DEM Download Failed",
                f"The elevation model could not be made:\n{message}")
            return
        mesh_path = message
        self.dem_path_edit.setText(mesh_path)
        json_path = os.path.splitext(mesh_path)[0] + ".json"
        if os.path.exists(json_path):
            self.dem_metadata_path_edit.setText(json_path)
            self.log(f"Auto-detected DEM metadata: {json_path}")
            self._try_auto_detect_crs_silent()
        QMessageBox.information(
            self, "DEM Download Complete",
            "The elevation model was made from the Copernicus DEM.\n\n"
            f"Mesh: {mesh_path}\n"
            f"Metadata: {json_path}\n\n"
            "It is a surface model at 30 m: allow for that under "
            "'Position Uncertainty' (Terrain model), above all in forest.")

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
        # Avoid to_epsg() - it uses the PROJ database which may be QGIS's outdated version.
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
                        except Exception:
                            quiet.detail("The GeoTIFF's CRS gave no EPSG code")
        except Exception:
            quiet.detail("The GeoTIFF's CRS could not be read for the confirmation dialog")
        if file_crs_label == "Unknown":
            # rasterio reports crs=None for e.g. compound CRS files; GDAL's OSR
            # can often still identify the horizontal EPSG code.
            from ..austria_dem_downloader import detect_geotiff_epsg
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
        from ..core.crs_utils import is_valid_utm_crs
        return is_valid_utm_crs(crs_text)

    def _parse_epsg_from_text(self, crs_text: str) -> Optional[int]:
        """Parse an EPSG code from text input (core.crs_utils)."""
        from ..core.crs_utils import parse_epsg_from_text
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
            except Exception:
                quiet.problem(f"The DEM metadata {dem_metadata_path} could not be read to find the CRS")

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
        from ..core.crs_utils import detect_utm_from_airdata
        return detect_utm_from_airdata(csv_path, log_fn=self.log)

    def cancel_dem_download(self):
        """Cancel ongoing DEM download."""
        if hasattr(self, 'dem_download_worker') and self.dem_download_worker:
            self.dem_download_worker.cancel()
            self.log("Cancelling DEM download...")

    def browse_dem_metadata(self):
        """Browse for DEM metadata JSON file."""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select DEM Metadata JSON", "", "JSON Files (*.json)")
        if file:
            self.dem_metadata_path_edit.setText(file)
