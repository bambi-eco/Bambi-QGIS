# -*- coding: utf-8 -*-
"""What has been done: the status next to every step.

Which results a flight's folder holds, which of them are on the map, the
line that says so next to each step - and Reset Step, which removes a
step's results again.
"""

import os

from qgis.PyQt.QtWidgets import QMessageBox, QInputDialog
from qgis.core import QgsProject, QgsLayerTree

from ..core import store as _store_kinds


class StepStatusMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _check_existing_outputs(self, target_folder: str):
        """Show what has been processed in *target_folder*.

        A folder whose stores cannot be read - damaged, or written by a
        newer version - is shown as not processed, with the reason in the
        log. The error used to end whatever had asked for the refresh half
        way: switching to such a flight left the selector on it and
        everything else on the flight before.
        """
        try:
            self._read_existing_outputs(target_folder)
        except Exception as exc:  # noqa: BLE001 - said in the log
            self._read_existing_outputs("")
            self.log("WARNING: what has been processed in "
                     f"{target_folder} could not be read ({exc}); the steps "
                     "are shown as not started.")

    def _read_existing_outputs(self, target_folder: str):
        """Check for existing output subfolders and update status labels.

        Each camera-selectable step is checked against the camera currently
        selected in its combo box (core.output_inventory does the folder
        checks), so the status always reflects the selected modality.

        :param target_folder: Path to the target folder
        """
        from ..core.output_inventory import (
            FOLDER_STATUS_STEPS, PERPENDICULAR_STEPS, check_existing_outputs,
            stale_steps)

        if not target_folder or not os.path.isdir(target_folder):
            # A flight without a (valid) target folder has processed
            # nothing: show that, rather than whatever the previously
            # shown flight left in the labels.
            from ..core.output_inventory import CLASSIFICATION_STATUS_STEPS
            for step in FOLDER_STATUS_STEPS:
                self.update_status(step, "⚪ Not started")
            for step in PERPENDICULAR_STEPS:
                self.update_status(step, "⚪")
            for step in CLASSIFICATION_STATUS_STEPS:
                self.update_status(step, "⚪ Not started")
            self._refresh_environment_status(target_folder)
            self._check_existing_qgis_layers()
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
            # The Segmentation tool has its own camera choice; the panel
            # reports the thermal results and the tool shows the rest.
            "sam3": "_t",
            # Each flight keeps its own copy of the survey's inventory.
            "inventory": (f"_{self.survey_panel.modality()}"
                          if hasattr(self, "survey_panel") else "_t"),
        }

        completed = check_existing_outputs(target_folder, cameras)
        for step in completed:
            self.update_status(step, "🟢 Completed")
        # ... unless something it was computed from has run again since.
        for step in stale_steps(target_folder, cameras, completed):
            self.update_status(step, "🟠 Out of date")

        # Classification steps: store-backed, per modality, and aware of
        # being out of date - read from the stage states, as configured.
        from ..core import classification
        from ..core.output_inventory import (
            CLASSIFICATION_STATUS_STEPS, CLASSIFICATION_TASKS,
            classification_states)

        for step in CLASSIFICATION_STATUS_STEPS:
            self.update_status(step, "⚪ Not started")
        embeddings_targets = classification.INPUT_TARGETS.get(
            self.embeddings_camera_combo.currentData()
            if hasattr(self, "embeddings_camera_combo") else "matched",
            ("t", "w"))
        task_targets = {
            task: classification.targets_of(self._classification_spec(task))
            for task in CLASSIFICATION_TASKS
        }
        states = classification_states(
            target_folder, embeddings_targets, task_targets)
        # Cross-modal matching: the file scan found it; the stage state says
        # whether a re-run of either camera's tracking has outdated it.
        from ..core.output_inventory import cross_modal_states
        states.update(cross_modal_states(target_folder))
        labels = {"complete": "🟢 Completed", "stale": "🟠 Out of date"}
        classified = 0
        for step, state in states.items():
            if state in labels:
                self.update_status(step, labels[state])
                classified += 1

        # Environment sources: flight-level, read from their metadata files.
        self._refresh_environment_status(target_folder)
        self._refresh_habitat_status()

        if completed or classified:
            self.log(f"Detected {len(completed) + classified} completed "
                     "processing step(s) in target folder")

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
                     "add_orthomosaic", "add_perpendicular",
                     "add_track_perpendicular", "add_matches"):
            self.update_status(step, "⚪")

        # Only the active flight's own group counts: every flight has
        # layers of the same names, so a look at the whole project would
        # report another flight's orthomosaic as this one's. Groups and
        # layers are collected at any depth (camera subgroups, 9.0.0).
        scope = self._flight_group(create=False)
        existing_groups = set()
        existing_layers = set()

        def collect(node):
            for child in node.children():
                if QgsLayerTree.isGroup(child):
                    existing_groups.add(child.name())
                    collect(child)
                elif hasattr(child, "name"):
                    existing_layers.add(child.name())

        # A flight whose group does not exist yet has nothing added - the
        # root (what _flight_group falls back to) holds the other flights.
        root = QgsProject.instance().layerTreeRoot()
        if scope is not root or self._active_flight() is None:
            collect(scope)

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

        # The matched-pairs layer spans both cameras, so its name carries no
        # camera label to check against.
        if "BAMBI Matched Pairs" in existing_layers:
            self.update_status("add_matches", "🟢 Added")
            added_count += 1

        if added_count > 0:
            self.log(f"Detected {added_count} existing BAMBI layer(s) in QGIS project")

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
        "track_matching": "Match RGB ↔ Thermal Tracks",
        "embeddings": "DINOv3 Embeddings",
        "classification": "Classification",
        "life_stage": "Life Stage",
        "sam3_segmentation": "SAM3 Segmentation",
        "sam3_georeference": "Geo-Reference Segmentation",
        "perpendicular": "Perpendicular Distances",
        "track_perpendicular": "Track Perpendicular Distances",
        "population": "Population Estimate",
        "density": "Density Map",
        "coverage": "Coverage Map",
        "track_inventory": "Track Inventory",
    }

    def has_stage(self, kind: str, modality: str, folder: str = "") -> bool:
        """True when a stage has results for this modality.

        On the store, not on the 5.x text file beside it: those are written for
        external scripts and can be switched off, so their presence says
        nothing about whether a step has run.
        """
        folder = folder or self.target_folder_edit.text().strip()
        if not folder or modality not in _store_kinds.MODALITIES:
            return False
        return os.path.isfile(
            _store_kinds.stage_path(folder, kind, modality))

    def _resettable_steps(self, folder: str):
        """``(label, stage, modality)`` of every step that has output in
        *folder*, per camera; a step that belongs to both cameras once."""
        from ..core import stages

        entries = []
        for stage in stages.known_stages():
            name = self.STAGE_LABELS.get(stage, stage)
            if stage in stages.STAGE_SHARED_STORE_KIND:
                if stages.has_output(folder, stage, stages.CROSS_MODAL):
                    entries.append((f"{name} - both cameras", stage,
                                    stages.CROSS_MODAL))
                continue
            for modality, camera in (("t", "Thermal"), ("w", "RGB")):
                if stages.has_output(folder, stage, modality):
                    entries.append((f"{name} - {camera}", stage, modality))
        return entries

    def reset_stage(self):
        """Delete one step's outputs and flag what depended on it.

        Replaces deleting a folder in Explorer: the cascade is applied, so the
        results further downstream are marked out of date instead of quietly
        surviving as stale files.
        """
        from ..core import stages

        folder = self.target_folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Reset Step",
                                "Select a target folder first.")
            return

        # Every step with output, for each camera: the camera used to be
        # read off a combo box in a way that always gave "thermal", so the
        # RGB results could not be reset at all - and a thermal step was
        # reset when the RGB one was meant.
        entries = self._resettable_steps(folder)
        if not entries:
            QMessageBox.information(
                self, "Reset Step",
                "This project has no step outputs to reset.")
            return

        labels = [label for label, _stage, _modality in entries]
        choice, ok = QInputDialog.getItem(
            self, "Reset Step", "Step to reset:", labels, 0, False)
        if not ok or choice not in labels:
            return
        _label, stage, modality = entries[labels.index(choice)]

        depending = set(stages.dependents(stage))
        cameras = (modality, stages.CROSS_MODAL)
        affected_present = [
            label for label, other, other_modality in entries
            if other in depending if other_modality in cameras]
        message = (f"Delete the outputs of '{choice}'?\n\n"
                   "The files are removed and the step runs from scratch "
                   "next time.")
        if affected_present:
            names = "\n".join(f"  • {label}" for label in affected_present)
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
            "track_matching": self.track_matching_status,
            "add_matches": self.add_matches_status,
            "embeddings": self.embeddings_status,
            "classification": self.classify_occlusion_status,
            "classify_occlusion": self.classify_occlusion_status,
            "classify_species": self.classify_species_status,
            "classify_sex": self.classify_sex_status,
            "life_stage": self.life_stage_status,
            "sync_labels": self.sync_labels_status,
            "sam3_segmentation": self.sam3_segment_status,
            # TRex import runs under the "Track Animals Or Import" step, so its
            # progress is reported on the tracking status label.
            "trex_import": self.track_status,
            "density_heatmap": self.density_status,
            "add_density": self.add_density_status,
            "distance_sampling": self.distance_sampling_status,
            "coverage_map": self.coverage_status,
            "add_coverage": self.add_coverage_status,
            "survey_orthomosaic": self.survey_ortho_status,
            "add_survey_orthomosaic": self.add_survey_ortho_status,
            "track_inventory": self.track_inventory_status,
            "census_summary": self.census_status,
            "add_groups": self.add_groups_status,
            "grid_density": self.grid_status,
            "add_grid": self.add_grid_status,
            "survey_trend": self.trend_status,
            "drone_reaction": self.reaction_status,
            "add_reaction": self.add_reaction_status,
            "position_uncertainty": self.position_status,
            "add_position": self.add_position_status,
            "survey_report": self.report_status,
            "add_track_inventory": self.add_track_inventory_status,
            "population_estimation": self.population_status,
            "add_transect_areas": self.add_transect_areas_status,
        }
        status_map.update({key: label for key, label in
                           self._environment_status_map().items()
                           if label is not None})
        if step in status_map:
            status_map[step].setText(status)
