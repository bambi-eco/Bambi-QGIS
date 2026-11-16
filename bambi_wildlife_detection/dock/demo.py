# -*- coding: utf-8 -*-
"""Input > Download Demo Flight: a public example flight at a click.

The row at the top of the Input tab, the question where to put the flight,
the download (``demo_flight_downloader``), and - in a project that has no
flight yet - filling the Input tab with what arrived, so that the first
step can be run at once.
"""

import os

from qgis.PyQt.QtCore import QThread
from qgis.PyQt.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
)

from ..core import demo_data


class DemoMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _build_demo_row(self) -> QHBoxLayout:
        """The row that offers the demo flight."""
        row = QHBoxLayout()
        hint = QLabel("New here?")
        hint.setStyleSheet("color: gray;")
        row.addWidget(hint)
        self.demo_btn = QPushButton("Download Demo Flight…")
        self.demo_btn.setToolTip(
            "Fetch a public example flight (122 MB, from the BAMBI raw "
            "dataset on Zenodo, CC BY 4.0) with its elevation model, and "
            "fill this tab with it - to try every step without data of "
            "your own.")
        self.demo_btn.clicked.connect(self.download_demo_flight)
        row.addWidget(self.demo_btn)
        row.addStretch()
        return row

    @staticmethod
    def _demo_folder(chosen: str) -> str:
        """The folder the flight goes into: one of its own inside the one
        chosen - unless that one is it already."""
        chosen = os.path.normpath(chosen)
        if os.path.basename(chosen) == demo_data.FOLDER_NAME:
            return chosen
        if demo_data.is_complete(demo_data.flight_files(chosen)):
            return chosen
        return os.path.join(chosen, demo_data.FOLDER_NAME)

    def download_demo_flight(self):
        """Ask where to put the demo flight, and fetch it."""
        if self._step_is_running():
            QMessageBox.information(
                self, "Demo Flight",
                "A step is running. Wait for it to finish first.")
            return
        demo = demo_data.DEMO
        chosen = QFileDialog.getExistingDirectory(
            self, "Where to put the demo flight", os.path.expanduser("~"))
        if not chosen:
            return
        folder = self._demo_folder(chosen)
        reply = QMessageBox.question(
            self, "Demo Flight",
            f"{demo.description}\n\n"
            f"Download: {demo.size / 1e6:.0f} MB from Zenodo, and its "
            "elevation model (the bare ground at 15 m from Google's GEM-15, "
            "a megabyte or two).\n"
            f"Into: {folder}\n\n"
            f"Licence: {demo.licence}. If you publish anything made with "
            f"it, cite:\n{demo.citation}\n\n"
            "Download it now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes)
        if reply != QMessageBox.StandardButton.Yes:
            return

        from ..demo_flight_downloader import DemoFlightWorker

        self.demo_btn.setEnabled(False)
        self.demo_btn.setText("Downloading...")
        self.demo_thread = QThread()
        self.demo_worker = DemoFlightWorker(folder)
        self.demo_worker.moveToThread(self.demo_thread)
        self.demo_thread.started.connect(self.demo_worker.run)
        self.demo_worker.finished.connect(self._on_demo_finished)
        self.demo_worker.progress.connect(self._on_demo_progress)
        self.demo_worker.log.connect(self.log)
        self.demo_worker.finished.connect(self.demo_thread.quit)
        self.demo_worker.finished.connect(self.demo_worker.deleteLater)
        self.demo_thread.finished.connect(self.demo_thread.deleteLater)
        self.demo_thread.start()
        self.log(f"Fetching the demo flight into {folder}...")

    def _on_demo_progress(self, percent: int):
        self.demo_btn.setText(f"Downloading... {percent}%")

    def _on_demo_finished(self, success: bool, message: str):
        """The demo flight is there - or it is not."""
        self.demo_btn.setEnabled(True)
        self.demo_btn.setText("Download Demo Flight…")
        if not success:
            if message != "Cancelled":
                QMessageBox.warning(
                    self, "Demo Flight",
                    f"The demo flight could not be made ready:\n{message}\n\n"
                    "What arrived is kept: pressing the button again "
                    "continues from there.")
            return
        folder = message
        worker = getattr(self, "demo_worker", None)
        no_model = getattr(worker, "dem_problem", "")
        note = ""
        if no_model:
            note = ("\n\nIts elevation model could not be fetched "
                    f"({no_model}). Fetch it with 'Download DEM "
                    "(worldwide)' below before the first step.")
        if self._flights():
            # the form is another flight's: nothing of it is touched
            QMessageBox.information(
                self, "Demo Flight",
                f"The demo flight is in\n{folder}\n\n"
                "This project has flights already, and the Input tab shows "
                "one of them. To work with the demo flight beside them: "
                "type a name under 'Name', press +, and choose the "
                "thermal videos in that folder - the other files are found "
                f"by themselves.{note}")
            return
        self._load_demo_flight(folder)
        QMessageBox.information(
            self, "Demo Flight",
            f"The demo flight is ready: '{demo_data.DEMO.name}', in\n"
            f"{folder}\n\n"
            "The Input tab is filled in. Go on with Pre-Processing > "
            "Extract Frames, then the steps of the Processing tab from top "
            f"to bottom.{note}")

    def _load_demo_flight(self, folder: str) -> None:
        """Fill the Input tab with the flight in *folder* - as the first
        flight of the project, under the demo's name."""
        files = demo_data.flight_files(folder)
        # the first flight takes the name that stands in the name field
        self.flight_name_edit.setText(demo_data.DEMO.name)
        self.video_mode_check.setChecked(True)
        self.embedded_srt_check.setChecked(False)
        self.thermal_video_paths_edit.setText(
            ", ".join(files["thermal_videos"]))
        self.thermal_srt_paths_edit.setText(", ".join(files["thermal_srts"]))
        self.rgb_video_paths_edit.setText(", ".join(files["rgb_videos"]))
        self.rgb_srt_paths_edit.setText(", ".join(files["rgb_srts"]))
        # the calibrations the flight came with, not a preset
        if files["thermal_calibration"]:
            self.thermal_calib_preset_combo.setCurrentIndex(0)
            self.thermal_calibration_path_edit.setText(
                files["thermal_calibration"])
        if files["rgb_calibration"]:
            self.rgb_calib_preset_combo.setCurrentIndex(0)
            self.rgb_calibration_path_edit.setText(files["rgb_calibration"])
        # flight log, elevation model, target folder and CRS: as for any
        # folder of recordings
        self._auto_detect_common_files(folder)
        self.log(f"The Input tab shows the demo flight in {folder}")
