# -*- coding: utf-8 -*-
"""The settings: what the form says, and where it is kept.

``get_config`` reads the whole form into the dictionary a step runs with.
The rest keeps the form in the QGIS project and in each flight's folder:
stored when the project is saved or the flight is left, shown again when
the project is opened or the flight is entered, and put back to the
defaults for a new project.
"""

import json
import math
import os
from typing import Dict, Any

from qgis.PyQt.QtWidgets import (
    QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QComboBox, QCheckBox,
)
from qgis.core import QgsProject

from ..bambi_calibrations import THERMAL_CALIBRATIONS, RGB_CALIBRATIONS
from ..gui_utils import PLUGIN_SCOPE, read_alfs_selection
from ..core import quiet

# Combo index -> the value the processing steps use. Kept beside the combo
# rather than derived from its label so renaming a label cannot silently
# re-target a setting.
_CLASSIFICATION_PROJECTIONS = ("non_geo", "geo_1k", "geo_2k", "alfs")

#: Detection "Projection" values, in the order of its combo
#: (core.raster_sources).
_DETECTION_PROJECTIONS = ("perspective", "geotiff", "alfs")
_CLASSIFICATION_DEVICES = ("auto", "cpu", "cuda")
_CLASSIFICATION_UNMATCHED = ("skip", "rgb", "thermal")


class SettingsMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration from UI elements."""
        # The render engine and ray caster have no widget on this tab: they
        # are chosen in the Dependency Manager, next to the install that makes
        # each choice possible, and live in the QGIS project.
        _alfs_engine, _alfs_raycaster = read_alfs_selection()

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
        # mode, the curve in the curve mode - the other one is sent as None.
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
            "heading_source": self.heading_source_combo.currentData(),
            "airdata_video_padding_ms": int(round(
                self.airdata_padding_spin.value() * 1000)),
            "airdata_video_padding_auto": self.airdata_padding_auto_check.isChecked(),
            "no_images": self.no_images_check.isChecked(),
            "flag_duplicate_frames": (
                self.flag_duplicates_check.isChecked() and self.video_mode_check.isChecked()),
            "correct_grid_convergence": self.grid_convergence_check.isChecked(),

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
            "detection_projection": _DETECTION_PROJECTIONS[max(0, min(
                self.detection_projection_combo.currentIndex(),
                len(_DETECTION_PROJECTIONS) - 1))],
            "custom_implementations": {
                key: dict(spec) for key, spec in
                (getattr(self, "_custom_implementations", None) or {}).items()},
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
            # Which alfspy engine and ray caster to render with. Chosen in the
            # Dependency Manager, because the choice is only meaningful
            # alongside the extras that make it work, and stored in the QGIS
            # project rather than the flight folder - it describes this
            # machine's install, not the flight.
            "alfs_engine": _alfs_engine,
            "alfs_raycaster": _alfs_raycaster,
            "alfs_ground_resolution": self.alfs_resolution_spin.value(),
            "alfs_fixed_render_size": (
                self.alfs_fixed_size_check.isChecked()
                if hasattr(self, 'alfs_fixed_size_check') else False),
            "alfs_render_width": (
                self.alfs_render_width_spin.value()
                if hasattr(self, 'alfs_render_width_spin') else 2048),
            "alfs_render_height": (
                self.alfs_render_height_spin.value()
                if hasattr(self, 'alfs_render_height_spin') else 2048),
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

            # Classification. The token is resolved here rather than stored in
            # the config, so it never reaches the project file; an empty field
            # falls back to the environment or the 'hf auth login' token.
            "hf_token": (
                self._resolve_hf_token()
                if hasattr(self, 'hf_token_edit') else ""),
            "classification_backbone": self._classification_backbone(),
            "classification_projection": (
                _CLASSIFICATION_PROJECTIONS[
                    self.classification_projection_combo.currentIndex()]
                if hasattr(self, 'classification_projection_combo')
                else _CLASSIFICATION_PROJECTIONS[0]),
            "classification_device": (
                _CLASSIFICATION_DEVICES[
                    self.classification_device_combo.currentIndex()]
                if hasattr(self, 'classification_device_combo')
                else _CLASSIFICATION_DEVICES[0]),
            "classification_batch_size": (
                self.classification_batch_spin.value()
                if hasattr(self, 'classification_batch_spin') else 16),
            "classification_fp16": (
                self.classification_fp16_check.isChecked()
                if hasattr(self, 'classification_fp16_check') else True),
            "classification_models": dict(
                getattr(self, '_classification_models', None) or {}),
            "classification_crop_padding": (
                self.classification_padding_spin.value()
                if hasattr(self, 'classification_padding_spin') else 0.25),
            "classification_crop_size": (
                self.classification_crop_size_spin.value()
                if hasattr(self, 'classification_crop_size_spin') else 224),
            "classification_letterbox": (
                self.classification_letterbox_check.isChecked()
                if hasattr(self, 'classification_letterbox_check') else True),
            "embeddings_camera": (
                self.embeddings_camera_combo.currentData()
                if hasattr(self, 'embeddings_camera_combo') else "matched"),
            "classification_only": list(
                getattr(self, '_classification_only', ()) or ()),
            "classification_frame_selection": self._combo_choice(
                'classification_frames_combo', ("visible", "all")),
            "classification_quorum": (
                self.classification_quorum_spin.value()
                if hasattr(self, 'classification_quorum_spin') else 0.5),
            "classification_min_frames": (
                self.classification_min_frames_spin.value()
                if hasattr(self, 'classification_min_frames_spin') else 1),
            "classification_unmatched": self._combo_choice(
                'classification_unmatched_combo', _CLASSIFICATION_UNMATCHED),
            "life_stage_z": (
                self.life_stage_z_spin.value()
                if hasattr(self, 'life_stage_z_spin') else -2.0),
            "life_stage_iqr_factor": (
                self.life_stage_iqr_spin.value()
                if hasattr(self, 'life_stage_iqr_spin') else 0.5),
            "life_stage_min_individuals": (
                self.life_stage_min_spin.value()
                if hasattr(self, 'life_stage_min_spin') else 4),
            "classification_overwrite_species": (
                self.classification_overwrite_check.isChecked()
                if hasattr(self, 'classification_overwrite_check') else False),

            # Cross-modal matching
            "match_min_shared": (
                self.match_min_shared_spin.value()
                if hasattr(self, 'match_min_shared_spin') else 8),
            "match_gate_px": (
                self.match_gate_spin.value()
                if hasattr(self, 'match_gate_spin') else 28.0),
            "match_space": (
                self.match_space_combo.currentData()
                if hasattr(self, 'match_space_combo') else "auto"),
            "match_gate_m": (
                self.match_gate_m_spin.value()
                if hasattr(self, 'match_gate_m_spin') else 1.5),
            "match_min_confidence": (
                self.match_min_confidence_spin.value()
                if hasattr(self, 'match_min_confidence_spin') else 0.20),
            "match_max_time_offset": (
                self.match_max_time_offset_spin.value()
                if hasattr(self, 'match_max_time_offset_spin') else 0.10),
            "match_thermal_anchored": (
                self.match_thermal_anchored_check.isChecked()
                if hasattr(self, 'match_thermal_anchored_check') else True),

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

            # Survey analytics: the flights, camera, folder, species and
            # survey area travel as config["survey"] (core.survey), added by
            # start_worker for the survey steps - see _survey_config().
            # What the survey's tools count - animals or boxes - is one
            # choice for all of them, made in the Survey box. It is passed
            # under each tool's own key as well, which is what the tools and
            # the result files are named by.
            "census_source": self._survey_source(),

            # Survey analytics: grid density (the grid_* settings follow
            # below, from the grid panel)
            "grid_source": self._survey_source(),

            # Survey analytics: the visits compared (dock/trend.py)
            "trend_source": self._survey_source(),
            **self._trend_values(),

            # Survey analytics: density heatmap
            "density_source": self._survey_source(),
            "density_cell_size": (
                self.density_cell_spin.value()
                if hasattr(self, 'density_cell_spin') else 5.0),
            "density_bandwidth": (
                self.density_bandwidth_spin.value()
                if hasattr(self, 'density_bandwidth_spin') else 25.0),

            # Survey analytics: coverage map
            "inventory_columns": list(
                getattr(self, '_inventory_columns', None) or []),
            "coverage_cell_size": (
                self.coverage_cell_spin.value()
                if hasattr(self, 'coverage_cell_spin') else 1.0),

            # Survey analytics: distance sampling
            "ds_source": self._survey_source(),
            "ds_truncation": (
                self.ds_truncation_spin.value()
                if hasattr(self, 'ds_truncation_spin') else 0.0),
            "ds_covariates": (
                self.ds_covariates_check.isChecked()
                if hasattr(self, 'ds_covariates_check') else False),

            # Survey analytics: transect population estimation
            "pop_truncation": (
                self.pop_truncation_spin.value()
                if hasattr(self, 'pop_truncation_spin') else 0.0),
            "pop_study_area_ha": (
                self.pop_study_area_spin.value()
                if hasattr(self, 'pop_study_area_spin') else 0.0),
            "pop_study_area_auto": (
                self.pop_study_area_auto_check.isChecked()
                if hasattr(self, 'pop_study_area_auto_check') else False),
            "pop_study_area_survey": (
                self.pop_study_area_survey_check.isChecked()
                if hasattr(self, 'pop_study_area_survey_check') else False),
            "pop_methods": self._selected_population_methods(),
            "pop_n_boot": (
                self.pop_n_boot_spin.value()
                if hasattr(self, 'pop_n_boot_spin') else 999),
            "pop_seed": (
                self.pop_seed_spin.value()
                if hasattr(self, 'pop_seed_spin') else 42),

            # Environmental context and habitat summary
            **self._environment_config(),

            # Grid density: how the grid is cut and read
            **(self.grid_panel.values() if hasattr(self, 'grid_panel')
               else {}),

            # Map symbols of detections and tracks
            "map_pins_enabled": (
                self.map_pins_check.isChecked()
                if hasattr(self, 'map_pins_check') else True),
            "map_pin_scale": (
                self.map_pin_scale_spin.value()
                if hasattr(self, 'map_pin_scale_spin') else 500),
        }

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
            quiet.detail("The project's signals were not connected")

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

        The flights go with the project they belong to. They used to stay:
        the form was emptied, but the selector still listed the previous
        project's flights with one of them active - so the target folder
        entered for the new project pointed *that* flight at it, and saving
        the new project wrote the old one's flights into it.
        """
        self._loading_config = True
        try:
            self._flight_list = []
            self._active_flight_index = 0
            self._refresh_flight_combo()
            self.reset_config_to_defaults()
            self.flight_name_edit.clear()
            # nothing is processed in a project without a flight
            self._check_existing_outputs("")
            self._refresh_migrate_button()
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

        # Bound through the schema's ``json`` role, so it is an attribute
        # rather than a widget and the widget sweep above does not reach it.
        # The Hugging Face token is deliberately *not* reset: it is a user
        # credential in QSettings, not part of this project's configuration.
        self._classification_models = {}
        self._inventory_columns = []
        if hasattr(self, 'survey_panel'):
            self.survey_panel.reset()
        if hasattr(self, 'grid_panel'):
            self.grid_panel.reset()
        if hasattr(self, 'trend_gap_spin'):
            self._reset_trend_settings()

        self.log("Configuration reset to defaults")

    # -- structured settings, as the schema's ``json`` role reads and writes
    # them. Both used to be saved by hand - into the QGIS project, while a
    # load read the flight's store - so they were gone after every load.
    @property
    def _stored_additional_corrections(self) -> list:
        """The frame-range corrections, as stored with the flight."""
        return list(self._additional_corrections)

    @_stored_additional_corrections.setter
    def _stored_additional_corrections(self, value) -> None:
        # Anything but a list of corrections loads as "none": a missing
        # entry, or the row numbers that versions before 9.1 wrote instead of
        # the corrections themselves.
        entries = value if isinstance(value, list) else []
        self._additional_corrections = [
            dict(entry) for entry in entries if isinstance(entry, dict)]
        self._update_corrections_list_ui()

    @property
    def _stored_thermal_vis_curve(self) -> dict:
        """The thermal curve as a plain dict (empty: no curve)."""
        curve = self._thermal_vis_curve
        return curve.to_dict() if curve is not None else {}

    @_stored_thermal_vis_curve.setter
    def _stored_thermal_vis_curve(self, value) -> None:
        curve = None
        if isinstance(value, dict) and value:
            from ..core.thermal_curve import ThermalCurve
            try:
                curve = ThermalCurve.from_dict(value)
            except (TypeError, ValueError):
                self.log("Warning: ignoring an unreadable thermal curve")
        self._thermal_vis_curve = curve
        self._update_thermal_vis_curve_label()

    def _current_config_values(self) -> Dict[str, Any]:
        """Every setting of the form, by its key in ``core.config_schema``."""
        from ..core.config_schema import WIDGET_BINDINGS

        return {key: self._config_widget_value(attr, role)
                for key, (attr, role) in WIDGET_BINDINGS.items()}

    def _config_widget_value(self, attr: str, role: str):
        """Read a config value from the bound widget (core.config_schema role)."""
        if role == "json":
            # Not a widget: a plain attribute holding a dict/list, serialised
            # into a string entry. Structured settings live behind a table or
            # dialog rather than one value widget.
            return json.dumps(getattr(self, attr, None) or {}, sort_keys=True)
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
        if role == "combo_data":
            data = widget.currentData()
            return "" if data is None else str(data)
        raise ValueError(f"Unknown config role {role!r}")

    def _apply_config_value(self, attr: str, role: str, value) -> None:
        """Apply a loaded config value to the bound widget (config_schema role)."""
        if role == "json":
            try:
                setattr(self, attr, json.loads(value) if value else {})
            except (ValueError, TypeError):
                # A corrupted entry must not stop the rest of the
                # configuration loading; the setting reverts to its default.
                setattr(self, attr, {})
                self.log(f"Warning: ignoring unreadable setting for {attr}")
            return
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
        elif role == "combo_data":
            idx = widget.findData(value) if value else -1
            # Missing or unknown (a project from an older version): the first
            # item is the default, so a stale choice of the previous flight
            # does not survive a flight switch.
            widget.setCurrentIndex(idx if idx >= 0 else 0)
        else:
            raise ValueError(f"Unknown config role {role!r}")

    def save_config_to_project(self):
        """Save the configuration of the active flight.

        The form values go to the flight's own ``project.gpkg``, so the folder
        is self-describing; the QGIS project keeps only the flight list and
        which one is active.

        Only the active flight's own folder is written to. A folder that is
        merely in the target folder field - no flight has it yet - keeps
        what it has stored: this is also what runs when the QGIS project is
        saved, and it replaced a processed folder's configuration with
        whatever the form showed.
        """
        from ..core import flights as flights_core
        from ..core.config_schema import save_config_entries
        project = QgsProject.instance()

        values = self._current_config_values()

        target_folder = self.target_folder_edit.text().strip()
        name = (self._active_flight() or {}).get("name") or ""
        if target_folder and os.path.isdir(target_folder) and \
                self._flight_owns(target_folder):
            if self._store_form_config(target_folder):
                # they have their home now; the project need not keep them
                self._keep_in_project(project, name, None)
        else:
            # No folder yet, so there is nowhere to put it - the QGIS project
            # keeps it for this flight until there is. The same for a folder
            # in the field that is not the active flight's: a folder's
            # stored configuration is only ever replaced by its own flight.
            kept = {}
            save_config_entries(values, *flights_core.config_writers(kept))
            self._keep_in_project(project, name, kept)

        project.writeEntry(PLUGIN_SCOPE, "Flights/List",
                           json.dumps(self._flights()))
        if hasattr(self, 'survey_panel'):
            self.survey_panel.save(project)
        if hasattr(self, 'grid_panel'):
            self.grid_panel.save(project)
        if hasattr(self, 'trend_gap_spin'):
            self._save_trend_settings(project)
        project.writeEntry(PLUGIN_SCOPE, "Flights/Active",
                           str(getattr(self, "_active_flight_index", 0)))

        # Credentials are never saved here.

        # Mark project as modified so user is prompted to save
        project.setDirty(True)

        self.log("Configuration saved to project")

    def load_config_from_project(self, restore_flights: bool = True):
        """Load the active flight's configuration (core.config_schema).

        *restore_flights* is False when switching or adding a flight: the list
        in memory is already newer than the one in the QGIS project, which is
        only written on save.
        """
        from ..core.config_schema import WIDGET_BINDINGS, load_config_entries
        project = QgsProject.instance()

        # Which flight is active decides where the configuration is read
        # from, so the list has to be settled first.
        if restore_flights:
            self._restore_flights(project)
            # The survey selection is a project-level setting beside the
            # flight list, not part of any one flight's configuration.
            if hasattr(self, 'survey_panel'):
                self.survey_panel.load(project)
            if hasattr(self, 'grid_panel'):
                self.grid_panel.load(project)
            if hasattr(self, 'trend_gap_spin'):
                self._load_trend_settings(project)

        # Prefer the active flight's own store; fall back to the QGIS project,
        # which is where projects written before 6.0 keep everything.
        from ..core import flights as flights_core

        flight = self._active_flight()
        folder = (flight or {}).get("target_folder", "")
        stored, unreadable = {}, ""
        if folder:
            try:
                stored = flights_core.load_config(folder)
            except Exception as exc:  # noqa: BLE001 - shown to the user below
                unreadable = str(exc) or type(exc).__name__
        migrated_from_project = False

        if not stored:
            # what the QGIS project keeps for a flight without a folder to
            # put it in - this flight's own, and only its own
            stored = self._kept_in_project(project).get(
                (flight or {}).get("name") or "", {})
        if stored:
            read_str, read_double, read_bool = flights_core.config_readers(stored)
        elif not self._legacy_entries_are_for(project, flight):
            if flight is not None:
                # Nothing is stored for this flight. The form must not go on
                # showing the flight that was looked at before.
                self._show_flight_without_settings(flight, unreadable)
            return  # No saved config anywhere
        else:
            # never "moved" over a store that could not be read
            migrated_from_project = bool(folder) and not unreadable

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

        if folder:
            # The flight's folder is where this was read from. The path
            # stored with the settings is the one the folder had when they
            # were saved - another one once it has been moved, copied or
            # handed on - and steps would run in that old place.
            self._set_target_folder_text(folder)

        # Legacy migration: before 7.1 the heading was a checkbox
        # (Extraction/UseGimbalHeading); map it onto the heading source.
        if not read_str("Extraction/HeadingSource", ""):
            from ..core.heading import legacy_project_heading_source
            legacy_source = legacy_project_heading_source(
                read_str("Extraction/UseGimbalHeading", ""))
            idx = self.heading_source_combo.findData(legacy_source)
            if idx >= 0:
                self.heading_source_combo.setCurrentIndex(idx)

        # Legacy migration: projects saved before the model path was split
        # per modality stored a single Detection/ModelPath (thermal model).
        legacy_model_path = read_str("Detection/ModelPath", "")
        if legacy_model_path and not self.thermal_model_path_edit.text():
            self.thermal_model_path_edit.setText(legacy_model_path)

        # A project written before 6.0 keeps its configuration in the .qgz;
        # move it into the flight's store once, so the folder becomes
        # self-describing from here on.
        if migrated_from_project:
            self.save_config_to_project()
            self.log("Configuration moved into the flight's project.gpkg")

        # The classifier mapping is bound through the schema's ``json`` role,
        # so the attribute is already loaded - the table it is edited in has to
        # be brought back into step with it.
        self._apply_classification_models_to_table()
        self._sync_stage_cameras()
        self._apply_custom_implementations()

        self.log("Configuration loaded from project")

    #: QGIS project entry: the settings of flights that have no folder to
    #: keep them in (yet, or just now), as ``{flight name: {key: value}}``.
    ENTRY_KEPT_SETTINGS = "Flights/Pending"

    def _kept_in_project(self, project) -> dict:
        """``{flight name: settings}`` the QGIS project keeps for flights
        whose folder cannot take them: a flight added by name and waiting
        for its folder, one whose folder is not there. ``""`` is the form of
        a project that has no flight yet.

        One slot per flight. There used to be a single one - the project's
        plain entries - written by whichever flight was saved last and read
        for any flight whose own store had nothing: a flight whose drive was
        not connected was shown, and then given, another flight's settings.
        """
        raw, _ok = project.readEntry(PLUGIN_SCOPE, self.ENTRY_KEPT_SETTINGS, "")
        try:
            kept = json.loads(raw) if raw else {}
        except ValueError:
            return {}
        if not isinstance(kept, dict):
            return {}
        return {str(name): values for name, values in kept.items()
                if isinstance(values, dict)}

    def _keep_in_project(self, project, name: str, values) -> None:
        """Set (or, with ``None``, drop) what the project keeps for the
        flight called *name*."""
        kept = self._kept_in_project(project)
        if values is None:
            if name not in kept:
                return
            del kept[name]
        else:
            kept[name] = {key: "" if value is None else str(value)
                          for key, value in values.items()}
        project.writeEntry(PLUGIN_SCOPE, self.ENTRY_KEPT_SETTINGS,
                           json.dumps(kept, sort_keys=True))

    def _legacy_entries_are_for(self, project, flight) -> bool:
        """Whether the plain settings entries of the QGIS project - where
        versions before 9.1 kept the settings of a flight without a folder,
        and versions before 6.0 all settings - are *flight*'s.

        Nothing says whose they are, so they are taken for the only flight
        of the project (written before 6.0) or for a flight that still has
        no folder; never for one flight among several that has a folder.
        """
        _value, has_config = project.readEntry(
            PLUGIN_SCOPE, "Input/TargetFolder", "")
        if not has_config:
            return False
        return len(self._flights()) <= 1 or \
            not ((flight or {}).get("target_folder") or "").strip()

    def _show_flight_without_settings(self, flight: dict, unreadable: str = ""):
        """Show a flight nothing is stored for: the form as a new project
        shows it, in the flight's folder - and say why.

        The form used to stay as it was, so the flight appeared with the
        folder, the recordings and every setting of the flight selected
        before it (and a store that could not be read ended the switch with
        an error half way).
        """
        folder = (flight.get("target_folder") or "").strip()
        name = flight.get("name") or ""
        self._apply_flight_values(self._pristine_config)
        self._set_target_folder_text(folder)
        self._apply_classification_models_to_table()
        self._sync_stage_cameras()
        self._apply_custom_implementations()

        if unreadable:
            text = (f"The settings of flight '{name}' could not be read from "
                    f"{folder} ({unreadable}). The form shows the default "
                    "settings; nothing in the folder was changed.")
        elif folder and not os.path.isdir(folder):
            text = (f"The folder of flight '{name}' is not there: {folder} - "
                    "a drive that is not connected, or a folder that was "
                    "moved? The form shows the default settings.")
        else:
            self.log(f"No settings are stored for flight '{name}' yet - the "
                     "form shows the defaults.")
            return
        self.log(f"WARNING: {text}")
        bar = getattr(self.iface, "messageBar", None)
        if callable(bar):
            bar().pushWarning("BAMBI", text)
