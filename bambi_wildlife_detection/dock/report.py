# -*- coding: utf-8 -*-
"""Survey Analytics > Survey Report: the survey in one HTML file.

The group at the end of the Survey Analytics tab - a title for the report,
whether a picture of the map as QGIS shows it goes in, the button that
writes the file (``core.survey_report``) and the one that opens it.
"""

import json
import os

from qgis.PyQt.QtCore import Qt, QUrl
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QCheckBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QVBoxLayout,
)

from ..core import quiet, store as _store_kinds, survey_report
from ..gui_utils import PLUGIN_SCOPE

#: Where the title and the tick box are kept in the QGIS project.
ENTRY_REPORT = "Survey/Report"

#: A picture of the map wider than this is made smaller before it goes in.
CANVAS_WIDTH = 1400
#: What the picture is called while the report is written. It is removed
#: when the step has ended: it is in the page by then.
CANVAS_FILE = "survey_report_map.png"


class ReportMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _build_report_group(self) -> QGroupBox:
        """The Survey Report group of the Survey Analytics tab."""
        group = QGroupBox("Survey Report")
        layout = QVBoxLayout(group)
        description = QLabel(
            "Everything the tools above found for the ticked flights, in "
            "one HTML file that needs nothing else: what was flown and in "
            "what weather, a map, the counts by species, sex and age, "
            "every density with its interval, where the animals were, what "
            "changed between visits, and the settings. It reads the "
            "results that are there - run the tools you want in it first - "
            "and says of each when it was made.")
        description.setWordWrap(True)
        description.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(description)

        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("Title:"))
        self.report_title_edit = QLineEdit()
        self.report_title_edit.setPlaceholderText(
            "Survey report - the name of the survey folder")
        self.report_title_edit.setToolTip(
            "What the report is headed. Empty: 'Survey report' and the "
            "name of the survey folder.")
        title_row.addWidget(self.report_title_edit, 1)
        layout.addLayout(title_row)

        self.report_canvas_check = QCheckBox(
            "Include the map as it is shown in QGIS")
        self.report_canvas_check.setChecked(True)
        self.report_canvas_check.setToolTip(
            "Beside the map the report draws itself - routes, survey area "
            "and animals - a picture of the map canvas as it is now: with "
            "your base map, orthomosaic and layers.")
        layout.addWidget(self.report_canvas_check)

        run_row = QHBoxLayout()
        self.report_btn = QPushButton("→ Write Survey Report")
        self.report_btn.clicked.connect(self.run_survey_report)
        self.report_btn.setToolTip(
            "Write analytics_{camera}/survey_report.html into the survey "
            "folder and open it. The census is built first where it is "
            "missing.")
        self.report_status = QLabel("⚪")
        run_row.addWidget(self.report_btn)
        run_row.addWidget(self.report_status)
        layout.addLayout(run_row)

        open_row = QHBoxLayout()
        self.report_open_btn = QPushButton("→ Open Report")
        self.report_open_btn.clicked.connect(self.open_survey_report)
        self.report_open_btn.setToolTip(
            "Open the report that was written last in the web browser.")
        open_row.addWidget(self.report_open_btn)
        open_row.addStretch()
        layout.addLayout(open_row)
        return group

    def _report_values(self) -> dict:
        """The settings of the report, as the step reads them."""
        if not hasattr(self, "report_title_edit"):
            return {}
        return {
            "report_source": self._survey_source(),
            "report_title": self.report_title_edit.text().strip(),
            "report_canvas_png": getattr(self, "_report_canvas_path", ""),
        }

    # ------------------------------------------------------------------ #
    # Its settings, kept with the QGIS project
    # ------------------------------------------------------------------ #

    def _set_report_settings(self, values: dict) -> None:
        title = values.get("title")
        self.report_title_edit.setText(title if isinstance(title, str)
                                       else "")
        # the picture goes in unless somebody unticked it
        self.report_canvas_check.setChecked(values.get("canvas") is not False)

    def _save_report_settings(self, project) -> None:
        project.writeEntry(PLUGIN_SCOPE, ENTRY_REPORT, json.dumps({
            "title": self.report_title_edit.text().strip(),
            "canvas": self.report_canvas_check.isChecked()}))

    def _load_report_settings(self, project) -> None:
        raw, ok = project.readEntry(PLUGIN_SCOPE, ENTRY_REPORT, "")
        try:
            values = json.loads(raw) if ok and raw else {}
        except ValueError:
            values = {}
        self._set_report_settings(values if isinstance(values, dict) else {})

    def _reset_report_settings(self) -> None:
        self._set_report_settings({})

    # ------------------------------------------------------------------ #
    # Writing it, and opening it
    # ------------------------------------------------------------------ #

    def _report_path(self, survey=None) -> str:
        survey = survey or self.current_survey()
        return os.path.join(survey.analytics_folder, survey_report.FILE_NAME)

    def _grab_report_canvas(self, survey) -> str:
        """Save a picture of the map canvas for the report; ``""`` where
        that is not wanted or does not work."""
        if not self.report_canvas_check.isChecked():
            return ""
        try:
            picture = self.iface.mapCanvas().grab()
            if picture.isNull():
                return ""
            if picture.width() > CANVAS_WIDTH:
                picture = picture.scaledToWidth(
                    CANVAS_WIDTH, Qt.TransformationMode.SmoothTransformation)
            folder = survey.ensure_analytics_folder()
            path = os.path.join(folder, CANVAS_FILE)
            return path if picture.save(path, "PNG") else ""
        except Exception as exc:  # noqa: BLE001 - the report does without
            self.log(f"The map could not be pictured for the report "
                     f"({exc}): it has the drawn map only")
            return ""

    def _forget_report_canvas(self):
        """Remove the picture of the map that was made for the report. It
        is in the page now - beside it, it would look as if the page needed
        it."""
        path = getattr(self, "_report_canvas_path", "")
        self._report_canvas_path = ""
        if not path or not os.path.isfile(path):
            return
        try:
            os.remove(path)
        except OSError:
            quiet.detail(f"{path} could not be removed: it is left beside "
                         "the report, which does not need it")

    def run_survey_report(self):
        """Write the survey's report (core.survey_report)."""
        survey = self._checked_survey()
        if survey is None:
            return
        if self._survey_source() == "tracks":
            checks = self._tracks_check(survey)
        else:
            def checks(flight):
                if self.has_stage(_store_kinds.DETECTIONS, survey.modality,
                                  flight.target_folder):
                    return []
                return [f"{survey.camera_label} detections (run 'Detect "
                        "Animals' first)"]
        if not self._survey_prerequisites(survey, checks):
            return
        self._report_canvas_path = self._grab_report_canvas(survey)
        self.start_worker("survey_report")

    def open_survey_report(self):
        """Open the report in the web browser."""
        path = self._report_path()
        if not os.path.isfile(path):
            QMessageBox.warning(
                self, "Missing Data",
                "The survey report has not been written yet.\n"
                "Run 'Write Survey Report' first.")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
            QMessageBox.information(
                self, "Survey Report",
                f"The report is at\n{path}\n\nOpen it in a web browser.")
        self.log(f"Survey report: {path}")
