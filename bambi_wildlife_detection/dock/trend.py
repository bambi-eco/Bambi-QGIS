# -*- coding: utf-8 -*-
"""Survey Analytics > Compare Visits: the same ground, surveyed again.

The group on the Survey Analytics tab - how many days may lie between two
flights of one visit, which ground is compared, the button that runs the
comparison (``core.survey_trend``) - and the window that shows what it
found (``bambi_trend_dialog``).
"""

import json
import os

from qgis.PyQt.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSpinBox, QVBoxLayout,
)

from ..core import store as _store_kinds, survey_trend
from ..gui_utils import PLUGIN_SCOPE

#: Where the two settings are kept in the QGIS project.
ENTRY_TREND = "Survey/Trend"
TREND_DEFAULTS = {"trend_gap_days": 0, "trend_ground": survey_trend.SEEN}


class TrendMixin:
    """Mixed into :class:`BambiDockWidget`."""

    def _build_trend_group(self) -> QGroupBox:
        """The Compare Visits group of the Survey Analytics tab."""
        group = QGroupBox("Compare Visits")
        layout = QVBoxLayout(group)
        description = QLabel(
            "The same ground surveyed again - weeks or years apart. Tick "
            "the flights of every visit above: they are divided into "
            "visits by the day they were flown, and each visit is counted "
            "- animals, density with an interval, groups, sex and age per "
            "species. Then what changed from visit to visit, and the rate "
            "of change through all of them.")
        description.setWordWrap(True)
        description.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(description)

        gap_row = QHBoxLayout()
        gap_row.addWidget(QLabel("One visit: flights up to"))
        self.trend_gap_spin = QSpinBox()
        self.trend_gap_spin.setRange(0, survey_trend.MAX_GAP_DAYS)
        self.trend_gap_spin.setValue(TREND_DEFAULTS["trend_gap_days"])
        self.trend_gap_spin.setSuffix(" day(s) apart")
        self.trend_gap_spin.setToolTip(
            "A flight flown no more than this many days after the one "
            "before belongs to the same visit. 0 = the flights of one day. "
            "Flights less than six hours apart are always one visit, so a "
            "night survey does not end at midnight.")
        self.trend_gap_spin.valueChanged.connect(
            self._on_trend_settings_changed)
        gap_row.addWidget(self.trend_gap_spin)
        gap_row.addStretch()
        layout.addLayout(gap_row)

        ground_row = QHBoxLayout()
        ground_row.addWidget(QLabel("Ground compared:"))
        self.trend_ground_combo = QComboBox()
        for key in survey_trend.GROUNDS:
            self.trend_ground_combo.addItem(
                survey_trend.GROUND_LABELS[key], key)
        self.trend_ground_combo.setToolTip(
            "All the ground each visit saw: every visit is taken as it was "
            "flown, its animals over its ground. Only the ground every "
            "visit saw: the like-for-like comparison where the visits did "
            "not cover the same part of the area - animals elsewhere are "
            "left out.")
        self.trend_ground_combo.currentIndexChanged.connect(
            self._on_trend_settings_changed)
        ground_row.addWidget(self.trend_ground_combo)
        ground_row.addStretch()
        layout.addLayout(ground_row)

        run_row = QHBoxLayout()
        self.trend_btn = QPushButton("→ Compare Visits")
        self.trend_btn.clicked.connect(self.run_survey_trend)
        self.trend_btn.setToolTip(
            "Write analytics_{camera}/trend_{source}.json and "
            "_visits.csv into the survey folder and show the result.")
        self.trend_status = QLabel("⚪")
        run_row.addWidget(self.trend_btn)
        run_row.addWidget(self.trend_status)
        layout.addLayout(run_row)

        show_row = QHBoxLayout()
        self.trend_show_btn = QPushButton("→ Show Visits and Trend")
        self.trend_show_btn.clicked.connect(self.show_trend)
        show_row.addWidget(self.trend_show_btn)
        show_row.addStretch()
        layout.addLayout(show_row)
        return group

    # ------------------------------------------------------------------ #
    # The two settings
    # ------------------------------------------------------------------ #

    def _trend_values(self) -> dict:
        """The settings of the comparison, as the step reads them."""
        if not hasattr(self, "trend_gap_spin"):
            return dict(TREND_DEFAULTS)
        ground = self.trend_ground_combo.currentData()
        return {
            "trend_gap_days": self.trend_gap_spin.value(),
            "trend_ground": ground or survey_trend.SEEN,
        }

    def _set_trend_values(self, values: dict) -> None:
        merged = dict(TREND_DEFAULTS)
        merged.update({key: value for key, value in (values or {}).items()
                       if key in TREND_DEFAULTS and value is not None})
        try:
            gap = int(merged["trend_gap_days"])
        except (TypeError, ValueError):
            gap = TREND_DEFAULTS["trend_gap_days"]
        blocked = [self.trend_gap_spin.blockSignals(True),
                   self.trend_ground_combo.blockSignals(True)]
        try:
            self.trend_gap_spin.setValue(gap)
            index = self.trend_ground_combo.findData(merged["trend_ground"])
            self.trend_ground_combo.setCurrentIndex(max(0, index))
        finally:
            self.trend_gap_spin.blockSignals(blocked[0])
            self.trend_ground_combo.blockSignals(blocked[1])

    def _on_trend_settings_changed(self, *_args) -> None:
        """A setting was changed by the user: it is kept with the project,
        like the survey's flights."""
        if getattr(self, "_loading_config", False):
            return
        self.save_config_to_project()

    def _save_trend_settings(self, project) -> None:
        project.writeEntry(PLUGIN_SCOPE, ENTRY_TREND,
                           json.dumps(self._trend_values()))

    def _load_trend_settings(self, project) -> None:
        raw, ok = project.readEntry(PLUGIN_SCOPE, ENTRY_TREND, "")
        try:
            values = json.loads(raw) if ok and raw else {}
        except ValueError:
            values = {}
        self._set_trend_values(values if isinstance(values, dict) else {})

    def _reset_trend_settings(self) -> None:
        self._set_trend_values({})

    # ------------------------------------------------------------------ #
    # Running it, and its result
    # ------------------------------------------------------------------ #

    def _trend_result_path(self, survey=None) -> str:
        survey = survey or self.current_survey()
        return os.path.join(
            survey.analytics_folder, f"trend_{self._survey_source()}.json")

    def run_survey_trend(self):
        """Compare the visits of the survey (core.survey_trend)."""
        survey = self._checked_survey()
        if survey is None:
            return
        if len(survey.flights) < 2:
            QMessageBox.warning(
                self, "Compare Visits",
                "A comparison needs the flights of at least two visits.\n"
                "Tick the flights of every survey of this area in the "
                "Survey box above.")
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
        if self._survey_prerequisites(survey, checks):
            self.start_worker("survey_trend")

    def show_trend(self):
        """Show what the last comparison of the survey's visits found."""
        from ..bambi_trend_dialog import TrendDialog

        path = self._trend_result_path()
        if not os.path.isfile(path):
            QMessageBox.warning(
                self, "Missing Data",
                "The visits have not been compared yet.\n"
                "Run 'Compare Visits' first.")
            return
        try:
            document = survey_trend.read_json(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Compare Visits",
                                 f"Could not read {path}: {exc}")
            return
        dialog = TrendDialog(document, json_path=path, parent=self)
        dialog.setModal(False)
        dialog.show()
        self._trend_dialog = dialog
