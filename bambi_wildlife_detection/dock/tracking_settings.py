# -*- coding: utf-8 -*-
"""Configuration > Tracking: which tracker follows the animals.

The built-in tracker and its thresholds, the trackers of the BoxMOT package
with their parameters and re-identification models, tracklets imported from
TRex, and a tracker of the user's own.
"""

import json

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QPushButton,
    QLineEdit, QSpinBox, QDoubleSpinBox, QFileDialog, QLabel, QTextEdit,
    QComboBox, QCheckBox, QMessageBox, QFrame,
)

from .common import _cap_vertical_growth


class TrackingSettingsMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _build_tracking_settings(self, config_sub_tabs):
        """Configuration > Tracking."""
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
        _cap_vertical_growth(self.tracker_params_edit)
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
        self.iou_threshold_spin.setToolTip(
            "Smallest overlap (IoU) of a box with a track's last box for "
            "the two to be joined.\nUsed by the modes GREEDY, HUNGARIAN and "
            "HUNGARIAN_CENTER.")
        builtin_layout.addRow("IoU Threshold:", self.iou_threshold_spin)

        self.max_age_spin = QSpinBox()
        self.max_age_spin.setRange(-1, 100)
        self.max_age_spin.setValue(-1)
        self.max_age_spin.setToolTip(
            "Frames in which a track may go unmatched before it ends "
            "(-1 = never).\nOnly frames that have boxes count: a stretch "
            "without any box - the thermal camera's shutter - does not end "
            "a track.")
        builtin_layout.addRow("Max Age:", self.max_age_spin)

        self.max_center_dist_spin = QDoubleSpinBox()
        self.max_center_dist_spin.setRange(0.0, 100.0)
        self.max_center_dist_spin.setSingleStep(0.5)
        self.max_center_dist_spin.setValue(2.0)
        self.max_center_dist_spin.setSuffix(" m")
        self.max_center_dist_spin.setToolTip(
            "Largest distance between the centre of a box and of a track's "
            "last box for the two to be joined, in metres on the ground.\n"
            "Used by the modes CENTER and HUNGARIAN_CENTER.")
        builtin_layout.addRow("Max Center Distance:", self.max_center_dist_spin)

        from ..core import builtin_tracker
        self.tracker_mode_combo = QComboBox()
        self.tracker_mode_combo.addItems(list(builtin_tracker.MODES))
        self.tracker_mode_combo.setCurrentIndex(
            builtin_tracker.MODES.index(builtin_tracker.DEFAULT_MODE))
        self.tracker_mode_combo.setToolTip(
            "How the boxes of a frame are matched to the tracks so far:\n"
            "GREEDY - by overlap, the best pair first\n"
            "HUNGARIAN - by overlap, the assignment with the most overlap "
            "in all\n"
            "CENTER - by the distance between the box centres, nearest "
            "first\n"
            "HUNGARIAN_CENTER - HUNGARIAN, then CENTER for what overlap "
            "could not pair\n\n"
            "The centre modes join an animal whose consecutive boxes do not "
            "overlap: a small animal, a low frame rate, a jump in the "
            "position.")
        self.tracker_mode_combo.currentTextChanged.connect(
            self._on_tracker_mode_changed)
        builtin_layout.addRow("Tracker Mode:", self.tracker_mode_combo)
        self._on_tracker_mode_changed(self.tracker_mode_combo.currentText())

        self.class_aware_check = QCheckBox()
        self.class_aware_check.setChecked(True)
        builtin_layout.addRow("Class Aware:", self.class_aware_check)

        self.interpolate_check = QCheckBox()
        self.interpolate_check.setChecked(True)
        builtin_layout.addRow("Interpolate Missing:", self.interpolate_check)

        tracking_layout.addRow(self.builtin_params_widget)

        tracking_tab_layout.addWidget(tracking_group)

        from ..bambi_custom_impl_widget import CustomImplementationBox
        from ..core import custom_models as _custom_models

        custom_trk_box = CustomImplementationBox(
            "Custom tracker", _custom_models.TRACKER)
        custom_trk_box.setToolTip(
            "A class with setup(context) and update(frame, detections) - or "
            "run(detections, frames) for the whole flight at once - that "
            "says which track each detection belongs to. When ticked it "
            "replaces the tracker selected above (and the TRex import does "
            "not apply).\n\nOther .py files in the same folder can be "
            "imported by it, and files next to it (weights, data) are "
            "reached through os.path.dirname(__file__).")
        self.custom_tracker_interpolate_check = QCheckBox(
            "Interpolate gaps in the tracks")
        self.custom_tracker_interpolate_check.setChecked(True)
        self.custom_tracker_interpolate_check.setToolTip(
            "Fill the frames a track skips with linearly interpolated boxes "
            "(in the track files, as for the other trackers). Only applied "
            "where the tracker says it is applicable - its setup() may "
            "return {\"interpolate\": false}, e.g. for a tracker that "
            "already fills gaps itself or tracks re-identified animals "
            "across long absences.")
        self.custom_tracker_interpolate_check.toggled.connect(
            self._on_custom_implementation_changed)
        custom_trk_box.body.layout().addRow(
            "", self.custom_tracker_interpolate_check)
        custom_trk_box.changed.connect(self._on_custom_implementation_changed)
        self._custom_boxes["tracker"] = custom_trk_box
        tracking_tab_layout.addWidget(custom_trk_box)

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

        # ----- Custom import (core.custom_models, kind "importer") -----
        # What the TRex import does for TRex, for any other tool: the
        # user's own code reads the file, the plugin stores, undistorts
        # and places what it returns.
        import_box = CustomImplementationBox(
            "Custom import (detections or tracks)",
            _custom_models.IMPORTER, checkable=False)
        import_box.setToolTip(
            "Read detections - or whole tracks - made by another tool with "
            "a class of your own: setup(context) and run(source, frames), "
            "returning one row per box (frame, x1, y1, x2, y2, confidence, "
            "class, track_id).")
        import_info = QLabel(
            "Detections or tracks made by another tool can be imported "
            "with your own code, as the TRex import does for TRex: write "
            "a class with <code>setup(context)</code> and "
            "<code>run(source, frames)</code> that returns one row per box "
            "- with a <code>track_id</code> to import tracks, without for "
            "detections only. The sample <code>importer_csv.py</code> "
            "reads a CSV file. The import uses the camera of the Track "
            "step and replaces what an earlier import brought in; boxes "
            "of the detector stay.")
        import_info.setWordWrap(True)
        import_info.setTextFormat(Qt.TextFormat.RichText)
        import_info.setStyleSheet("color: gray; font-size: 10px;")
        import_box.layout().insertWidget(0, import_info)

        self.custom_import_source_edit = QLineEdit()
        self.custom_import_source_edit.setPlaceholderText(
            "The file or folder the importer reads")
        self.custom_import_source_edit.setToolTip(
            "Handed to the importer's run() as 'source': the other tool's "
            "output - a file, or a folder of files.")
        source_file_btn = QPushButton("File...")
        source_file_btn.clicked.connect(self.browse_custom_import_file)
        source_folder_btn = QPushButton("Folder...")
        source_folder_btn.clicked.connect(self.browse_custom_import_folder)
        source_row = QHBoxLayout()
        source_row.addWidget(self.custom_import_source_edit)
        source_row.addWidget(source_file_btn)
        source_row.addWidget(source_folder_btn)
        import_box.body.layout().insertRow(0, "Source:", source_row)

        self.custom_import_btn = QPushButton("Import now")
        self.custom_import_btn.setToolTip(
            "Run the importer on the source for the camera of the Track "
            "step: the boxes are stored, undistorted where they are on "
            "the raw video, and placed on the ground (Geo-Reference).")
        self.custom_import_btn.clicked.connect(self.run_custom_import)
        import_row = QHBoxLayout()
        import_row.addWidget(self.custom_import_btn)
        import_row.addStretch()
        import_box.body.layout().addRow("", import_row)
        import_box.changed.connect(self._on_custom_implementation_changed)
        self._custom_boxes["importer"] = import_box
        self.custom_import_box = import_box
        tracking_tab_layout.addWidget(import_box)

        # --- Cross-modal matching ---
        # Beside tracking rather than classification: it is a statement about
        # the tracks themselves - which two of them are one animal - and the
        # classifiers are only one of the things that read the answer.
        match_group = QGroupBox("Cross-Modal Track Matching")
        match_layout = QFormLayout(match_group)

        match_info = QLabel(
            "Decides which thermal track and which RGB track are the same "
            "animal. Useful on its own - an animal both cameras saw is a "
            "confirmed animal, which is what keeps a census free of tracks "
            "that fired on shadow - and required by the <i>matched</i> "
            "classifiers and by syncing labels between the cameras."
        )
        match_info.setWordWrap(True)
        match_info.setTextFormat(Qt.TextFormat.RichText)
        match_info.setStyleSheet("color: gray; font-size: 10px;")
        match_layout.addRow(match_info)

        self.match_min_shared_spin = QSpinBox()
        self.match_min_shared_spin.setRange(1, 1000)
        self.match_min_shared_spin.setValue(8)
        self.match_min_shared_spin.setToolTip(
            "Frames two tracks must have in common before they can be "
            "matched. Below this the median distance is not a median of much."
        )
        match_layout.addRow("Min shared frames:", self.match_min_shared_spin)

        self.match_space_combo = QComboBox()
        self.match_space_combo.addItem(
            "Automatic (ground when both cameras are geo-referenced)", "auto")
        self.match_space_combo.addItem("Ground distance (metres)", "world")
        self.match_space_combo.addItem(
            "Image pixels through an estimated transform", "pixel")
        self.match_space_combo.setToolTip(
            "Where the distance between a thermal and an RGB track is "
            "measured.\n\n"
            "On the ground, the geo-referenced boxes of both cameras are "
            "compared directly in metres - nothing has to be estimated, and "
            "each frame's own pose absorbs the aircraft's motion between the "
            "two capture instants.\n\n"
            "In image pixels (the published recipe) an RGB-to-thermal "
            "transform is first bootstrapped from the detections themselves; "
            "on a real flight that has paired neighbours instead of "
            "partners. Use it only when one camera has not been "
            "geo-referenced."
        )
        match_layout.addRow("Measure distance:", self.match_space_combo)

        self.match_gate_m_spin = QDoubleSpinBox()
        self.match_gate_m_spin.setRange(0.1, 1000.0)
        self.match_gate_m_spin.setDecimals(2)
        self.match_gate_m_spin.setSingleStep(0.25)
        self.match_gate_m_spin.setValue(1.5)
        self.match_gate_m_spin.setSuffix(" m")
        self.match_gate_m_spin.setToolTip(
            "Largest median distance on the ground still counted as the same "
            "animal, when matching by ground distance.\n\n"
            "Boxes the two cameras place on one animal typically land within "
            "half a metre of each other; two animals in a herd stand more "
            "than a body length apart."
        )
        match_layout.addRow("Ground distance gate:", self.match_gate_m_spin)

        self.match_gate_spin = QDoubleSpinBox()
        self.match_gate_spin.setRange(0.1, 10000.0)
        self.match_gate_spin.setDecimals(1)
        self.match_gate_spin.setSingleStep(1.0)
        self.match_gate_spin.setValue(28.0)
        self.match_gate_spin.setSuffix(" px")
        self.match_gate_spin.setToolTip(
            "Largest median inter-centre distance still counted as the same "
            "animal, in thermal pixels.\n\n"
            "The default of 28 px is the value published for 1024×1024 "
            "inference - raise it if your frames are larger, and check the "
            "run log, which reports how far the closest rejected candidate "
            "was."
        )
        match_layout.addRow("Pixel distance gate:", self.match_gate_spin)

        self.match_min_confidence_spin = QDoubleSpinBox()
        self.match_min_confidence_spin.setRange(0.0, 1.0)
        self.match_min_confidence_spin.setDecimals(2)
        self.match_min_confidence_spin.setSingleStep(0.05)
        self.match_min_confidence_spin.setValue(0.20)
        self.match_min_confidence_spin.setToolTip(
            "Mean detection confidence both tracks must reach. Stops a match "
            "being built on a track that fired on shadow."
        )
        match_layout.addRow("Min mean confidence:",
                            self.match_min_confidence_spin)

        self.match_max_time_offset_spin = QDoubleSpinBox()
        self.match_max_time_offset_spin.setRange(0.001, 10.0)
        self.match_max_time_offset_spin.setDecimals(3)
        self.match_max_time_offset_spin.setSingleStep(0.01)
        self.match_max_time_offset_spin.setValue(0.10)
        self.match_max_time_offset_spin.setSuffix(" s")
        self.match_max_time_offset_spin.setToolTip(
            "How far apart two frames may be on the shared capture clock and "
            "still count as the same moment. The two cameras record at "
            "different rates, so frame numbers never line up; at the ends of "
            "a flight the nearest frame in time can be seconds away, and "
            "pairing across such a gap would invent correspondences."
        )
        match_layout.addRow("Max frame time offset:",
                            self.match_max_time_offset_spin)

        self.match_thermal_anchored_check = QCheckBox(
            "Size RGB crops from the thermal box")
        self.match_thermal_anchored_check.setChecked(True)
        self.match_thermal_anchored_check.setToolTip(
            "For a matched pair, take the crop size from the thermal box - "
            "the looser of the two, so it more reliably encloses the whole "
            "animal, antlers included.\n\n"
            "Affects only the crop the classifier sees. The stored detections "
            "are never rewritten, so geo-referencing and tracking stay valid."
        )
        match_layout.addRow("", self.match_thermal_anchored_check)

        tracking_tab_layout.addWidget(match_group)

        tracking_tab_layout.addStretch()

    def _populate_tracker_backends(self):
        """Populate the tracker backend dropdown with available trackers."""
        from ..tracker_manager import get_tracker_manager

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

    def browse_custom_import_file(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, "Select the file to import",
            self.custom_import_source_edit.text())
        if path:
            self.custom_import_source_edit.setText(path)

    def browse_custom_import_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select the folder to import",
            self.custom_import_source_edit.text())
        if folder:
            self.custom_import_source_edit.setText(folder)

    def browse_trex_npz_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select TRex NPZ Folder")
        if folder:
            self.trex_npz_dir_edit.setText(folder)

    def _on_tracker_mode_changed(self, mode: str) -> None:
        """Enable the threshold each built-in tracker mode actually uses."""
        from ..core import builtin_tracker

        self.iou_threshold_spin.setEnabled(
            mode in builtin_tracker.OVERLAP_MODES)
        self.max_center_dist_spin.setEnabled(
            mode in builtin_tracker.CENTER_MODES)
