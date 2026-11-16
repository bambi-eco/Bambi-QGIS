# -*- coding: utf-8 -*-
"""Running a step, as the panel shows it: start, progress, the log.

Whichever tab a step is started from, it runs one at a time, with the
progress bar and Abort above the tabs and the log below them. This part
gets the step ready - the settings, the questions that have to be asked
first - and hands it to the step runner (``step_runner``, here
``self.steps``); it shows how the step goes and what it writes, and puts
the panel back when the step has ended or was aborted.
"""

import json
import os

from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QFileDialog,
    QTextEdit, QMessageBox,
)
from qgis.PyQt.QtGui import QFont

from ..bambi_environment import ENVIRONMENT_STEPS
from ..core import quiet


class RunPanelMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _build_run_panel(self, scroll_layout):
        """What is below the tabs whichever tab is in front: refresh and
        reset, and the log."""
        # ---------------------------------------------------------------------
        # Shared run panel - below the tabs, so the log is visible whichever
        # tab a step was started from (the progress bar sits above the tabs).
        # ---------------------------------------------------------------------
        run_panel = QWidget()
        run_layout = QVBoxLayout(run_panel)
        run_layout.setContentsMargins(0, 0, 0, 0)

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

        log_btn_row = QHBoxLayout()
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.log_text.clear)
        log_btn_row.addWidget(clear_log_btn)

        self.export_log_btn = QPushButton("Export Log…")
        self.export_log_btn.setToolTip("Save the log output to a text file")
        self.export_log_btn.clicked.connect(self._export_log)
        log_btn_row.addWidget(self.export_log_btn)
        log_layout.addLayout(log_btn_row)

        run_layout.addWidget(log_group)
        scroll_layout.addWidget(run_panel)

    def log(self, message: str):
        """Add a message to the log."""
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _export_log(self):
        """Save the log output to a text file chosen by the user."""
        from datetime import datetime
        folder = self.target_folder_edit.text().strip()
        name = f"bambi_log_{datetime.now():%Y%m%d_%H%M%S}.txt"
        default = os.path.join(folder, name) if os.path.isdir(folder) else name
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Log", default, "Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.log_text.toPlainText())
        except OSError as e:
            QMessageBox.warning(self, "Export Log", f"Could not write log:\n{e}")
            return
        self.log(f"Log exported to {path}")

    @property
    def worker(self):
        """The worker of the running step, ``None`` while no step runs -
        the step runner's, under the name the dock has always had for it."""
        return self.steps.worker

    @worker.setter
    def worker(self, value):
        self.steps.worker = value

    @property
    def worker_thread(self):
        """The thread of the step that was started last (the step
        runner's)."""
        return self.steps.thread

    @worker_thread.setter
    def worker_thread(self, value):
        self.steps.thread = value

    def _step_is_running(self) -> bool:
        """True while a background step runs.

        The flight guards used to look at an attribute nothing ever set, so
        they never held: flights could be switched, added and removed under
        a running step, whose log and status then described a flight no
        longer shown.
        """
        return self.steps.is_running()

    #: Survey Analytics steps: they run on config["survey"] (core.survey).
    _SURVEY_STEPS = frozenset({
        "census_summary", "grid_density", "survey_trend", "track_inventory",
        "density_heatmap",
        "coverage_map", "survey_orthomosaic", "distance_sampling",
        "population_estimation", "habitat_summary"})

    def start_worker(self, step: str):
        """Start a background worker for the given step."""
        if self.steps.is_running():
            QMessageBox.warning(self, "Processing", "Another process is already running.")
            return

        config = self.get_config()
        # Custom implementations are the user's own Python files: ask
        # before one runs here for the first time (or changed since).
        from ..bambi_custom_impl_widget import confirm_specs
        if not confirm_specs(self, self._custom_specs_for_step(step, config)):
            self.log("Not started: the custom implementation was not "
                     "approved to run.")
            return
        if step in self._SURVEY_STEPS:
            if self.survey_area_panel.is_enabled() and \
                    not self.survey_area_panel.flush_edits():
                return
            config["survey"] = self._survey_config()
        elif step in ENVIRONMENT_STEPS and \
                self.survey_area_panel.is_enabled():
            # The survey area widens the extent the environment is fetched
            # for; a survey that is not set up yet is simply no area.
            if not self.survey_area_panel.flush_edits():
                return
            try:
                config["survey"] = self._survey_config()
            except Exception:  # no survey, no area
                quiet.detail("No survey is set up: the step runs without a survey area")

        self.steps.start(
            self.processor, step, config,
            finished=self.on_worker_finished, error=self.on_worker_error,
            progress=self.on_worker_progress, log=self.log,
            starting=lambda: self._show_step_running(step))

    def _show_step_running(self, step: str) -> None:
        """The panel while a step runs: no other can be started, this one
        can be aborted."""
        # Update UI
        self.set_buttons_enabled(False)
        self.abort_btn.setEnabled(True)  # Enable abort button during processing
        self.update_status(step, "🟡 Running...")
        self.progress_bar.setValue(0)

        self.log(f"Starting {step}...")

    def shut_down(self) -> None:
        """Before the dock is removed (the plugin is unloaded or reloaded):
        cancel the running step and let its thread end by itself
        (``StepRunner.shut_down``)."""
        self.steps.shut_down()

    def on_worker_finished(self, step: str, success: bool):
        """Handle worker completion."""
        self.set_buttons_enabled(True)
        self.abort_btn.setEnabled(False)  # Disable abort button when done
        self.steps.ended()

        if success:
            self.update_status(step, "🟢 Completed")
            self.log(f"{step} completed successfully!")
            self.progress_bar.setValue(100)

            # Show the distance-sampling summary dialog on completion.
            if step == "distance_sampling":
                self._show_distance_sampling_results()

            # Show the population-estimation summary dialog on completion.
            if step == "population_estimation":
                self._show_population_results()

            if step == "census_summary":
                self.show_census_table()

            if step == "grid_density":
                self.show_grid_statistics()

            if step == "survey_trend":
                self.show_trend()

            if step == "habitat_summary":
                self.show_habitat_charts()

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
                        quiet.detail(f"{poses_path} could not be checked for matched photos")

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

        # An environment source that failed must not stop the others: show
        # what is on disk and go on with the next queued one.
        if step in ENVIRONMENT_STEPS:
            self._on_environment_step_finished(step)

    def on_worker_error(self, step: str, error_msg: str):
        """Handle worker error."""
        self.set_buttons_enabled(True)
        self.abort_btn.setEnabled(False)  # Disable abort button on error
        self.steps.ended()
        self.update_status(step, "🔴 Error")
        self.log(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Processing Error", error_msg)

    def on_worker_progress(self, value: int):
        """Update progress bar."""
        self.progress_bar.setValue(value)

    def _abort_current_process(self):
        """Abort the currently running process."""
        if self.steps.is_running():
            self.log("Aborting current process...")
            self.abort_btn.setEnabled(False)  # Prevent multiple clicks
            # Aborting stops the whole "Run All Four" queue, not one source.
            self._environment_queue = []
            self.steps.cancel()
            # The worker will emit finished signal when it detects cancellation

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
        if hasattr(self, 'density_btn'):
            self.density_btn.setEnabled(enabled)
            self.add_density_btn.setEnabled(enabled)
            self.distance_sampling_btn.setEnabled(enabled)
        if hasattr(self, 'coverage_btn'):
            self.coverage_btn.setEnabled(enabled)
            self.add_coverage_btn.setEnabled(enabled)
        if hasattr(self, 'survey_ortho_btn'):
            self.survey_ortho_btn.setEnabled(enabled)
            self.add_survey_ortho_btn.setEnabled(enabled)
        if hasattr(self, 'population_btn'):
            self.population_btn.setEnabled(enabled)
            self.add_transect_areas_btn.setEnabled(enabled)
        self._set_environment_buttons_enabled(enabled)
