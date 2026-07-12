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

    def test_custom_fields_settings(self):
        from bambi_wildlife_detection.bambi_labelling_tool import (
            CustomField, _CustomFieldsDialog)
        fields = [CustomField("weight", "float", "track"),
                  CustomField("blurry", "bool", "keyframe")]
        dlg = _CustomFieldsDialog(fields)
        assert isinstance(dlg, QDialog)
        # the table round-trips the schema it was given
        assert dlg.fields() == fields
        # … and the one a shared schema file is imported into
        dlg.set_fields([CustomField("tag", "string", "track")])
        assert [f.name for f in dlg.fields()] == ["tag"]
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


class TestDialogGeometry:
    """The big dialogs must open inside the screen, however small it is."""

    def _make_labelling_tool(self, iface, dock):
        from bambi_wildlife_detection.bambi_labelling_tool import (
            LabellingToolDialog)
        return LabellingToolDialog(iface, dock)

    def _make_video_creator(self, iface, dock):
        from bambi_wildlife_detection.bambi_video_creator import (
            VideoCreatorDialog)
        return VideoCreatorDialog(iface, dock)

    @pytest.mark.parametrize("factory", ["_make_labelling_tool",
                                         "_make_video_creator"])
    def test_fits_on_the_available_screen(self, factory, iface, dock):
        from qgis.PyQt.QtWidgets import QApplication
        avail = QApplication.primaryScreen().availableGeometry()
        dlg = getattr(self, factory)(iface, dock)
        assert dlg.width() <= avail.width()
        assert dlg.height() <= avail.height()
        # Their content scrolls, so no layout minimum forces the window back
        # above the screen height.
        assert dlg.minimumSizeHint().height() <= avail.height()
        _close(dlg)


class TestCustomFieldWidgets:
    """The custom-field editors must survive a value → widget → value trip."""

    @pytest.mark.parametrize("type_, value", [
        ("int", 42),
        ("float", 1.5),
        ("string", "roe deer"),
        ("bool", True),
        ("datetime", "2023-09-20T10:00:00"),
    ])
    def test_value_round_trip(self, type_, value):
        from bambi_wildlife_detection.bambi_labelling_tool import (
            CustomField, _field_widget_value, _make_field_widget,
            _set_field_widget_value)
        field = CustomField("f", type_)
        widget = _make_field_widget(field, lambda *a: None)
        _set_field_widget_value(field, widget, value)
        assert _field_widget_value(field, widget) == value
        widget.deleteLater()

    def test_unparseable_value_falls_back_to_the_default(self):
        from bambi_wildlife_detection.bambi_labelling_tool import (
            CustomField, _field_widget_value, _make_field_widget,
            _set_field_widget_value)
        field = CustomField("f", "int")
        widget = _make_field_widget(field, lambda *a: None)
        _set_field_widget_value(field, widget, "not a number")
        assert _field_widget_value(field, widget) == 0
        widget.deleteLater()
