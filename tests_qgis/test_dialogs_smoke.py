# -*- coding: utf-8 -*-
"""Real-QGIS smoke tests: every plugin dialog must construct without error.

These catch the most common breakage the unit and integration suites miss —
a renamed widget attribute or a signal wired to a deleted slot surfaces the
moment the dialog's ``__init__`` / ``_setup_ui`` runs. They do not exercise
behaviour; the assertion is simply "constructs, then closes cleanly".
"""
import pytest

from qgis.PyQt.QtWidgets import QDialog


def _close(dialog):
    dialog.close()
    dialog.deleteLater()


class TestDialogsConstruct:
    def test_labelling_tool(self, iface, dock):
        from bambi_wildlife_detection.bambi_labelling_tool import LabellingToolDialog
        dlg = LabellingToolDialog(iface, dock)
        assert isinstance(dlg, QDialog)
        _close(dlg)

    def test_video_creator(self, iface, dock):
        from bambi_wildlife_detection.bambi_video_creator import VideoCreatorDialog
        dlg = VideoCreatorDialog(iface, dock)
        assert isinstance(dlg, QDialog)
        _close(dlg)

    def test_flight_planner(self, iface):
        from bambi_wildlife_detection.bambi_flight_planner import FlightPlannerDialog
        dlg = FlightPlannerDialog(iface)
        assert isinstance(dlg, QDialog)
        _close(dlg)

    def test_camera_calibration_wizard(self):
        from bambi_wildlife_detection.bambi_camera_calibration import (
            CameraCalibrationWizard)
        dlg = CameraCalibrationWizard()
        assert isinstance(dlg, QDialog)
        _close(dlg)

    def test_dependency_manager(self):
        from bambi_wildlife_detection.bambi_dependency_manager import (
            DependencyManagerDialog)
        dlg = DependencyManagerDialog()
        assert isinstance(dlg, QDialog)
        _close(dlg)

    def test_thermal_viewer(self):
        from bambi_wildlife_detection.bambi_thermal_viewer import ThermalViewerDialog
        dlg = ThermalViewerDialog()
        assert isinstance(dlg, QDialog)
        _close(dlg)

    def test_feature_viewer(self):
        from bambi_wildlife_detection.bambi_feature_viewer import FeatureViewerDialog
        dlg = FeatureViewerDialog()
        assert isinstance(dlg, QDialog)
        _close(dlg)

    def test_correction_wizard_invalid_config_rejects_cleanly(self, iface):
        # With an empty config the wizard validates, schedules a deferred
        # rejection and returns from __init__ without building the UI or
        # raising — the construction path (imports, super().__init__) still
        # runs, which is what we are smoke-testing.
        from bambi_wildlife_detection.bambi_correction_wizard import (
            BambiCorrectionWizard)
        dlg = BambiCorrectionWizard(iface, {})
        assert isinstance(dlg, QDialog)
        _close(dlg)
