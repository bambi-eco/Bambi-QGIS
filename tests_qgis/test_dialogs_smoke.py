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

    def test_transect_tool(self, iface, dock):
        from bambi_wildlife_detection.bambi_transect_tool import TransectToolDialog
        dlg = TransectToolDialog(iface, dock)
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
        # both tone-mapping modes must switch without error
        dlg._tone_mode_box.setCurrentIndex(1)
        dlg._tone_mode_box.setCurrentIndex(0)
        _close(dlg)

    def test_curve_editor_dialog(self):
        from bambi_wildlife_detection.bambi_curve_widget import CurveEditorDialog
        from bambi_wildlife_detection.core.thermal_curve import ThermalCurve
        curve = ThermalCurve(5.0, 35.0, [(0.0, 0.0), (0.4, 0.7), (1.0, 1.0)])
        dlg = CurveEditorDialog(curve=curve)
        assert isinstance(dlg, QDialog)
        # the editor round-trips the curve it was seeded with
        out = dlg.curve()
        assert out.domain_lo == pytest.approx(5.0)
        assert out.domain_hi == pytest.approx(35.0)
        assert out.points == curve.points
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

    def _make_transect_tool(self, iface, dock):
        from bambi_wildlife_detection.bambi_transect_tool import (
            TransectToolDialog)
        return TransectToolDialog(iface, dock)

    @pytest.mark.parametrize("factory", ["_make_labelling_tool",
                                         "_make_video_creator",
                                         "_make_transect_tool"])
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


def _dialog_with_tracks(iface, dock, tmp_path):
    """A labelling dialog holding three tracks with disjoint frame ranges."""
    from bambi_wildlife_detection.bambi_labelling_tool import (
        LabelStore, LabelTrack, LabellingToolDialog)
    dlg = LabellingToolDialog(iface, dock)
    dlg._store = LabelStore(str(tmp_path), "t")
    for track_id, frames in ((1, (0, 10)), (2, (40, 50)), (3, (80, 90))):
        track = LabelTrack(track_id)
        for f in frames:
            track.set_keyframe(f, (0, 0, 10, 10))
        dlg._store.tracks[track_id] = track
    dlg._refresh_track_list()
    return dlg


class TestTrackMerging:
    """Merging / splitting through the dialog's list selection."""

    def _dialog_with_tracks(self, iface, dock, tmp_path):
        return _dialog_with_tracks(iface, dock, tmp_path)

    def test_selection_ids_follow_the_list(self, iface, dock, tmp_path):
        dlg = self._dialog_with_tracks(iface, dock, tmp_path)
        for row in (0, 2):
            dlg.track_list.item(row).setSelected(True)
        assert dlg._selected_track_ids() == [1, 3]
        _close(dlg)

    def test_merge_replaces_the_sources_with_one_track(
            self, iface, dock, tmp_path):
        dlg = self._dialog_with_tracks(iface, dock, tmp_path)
        assert dlg._apply_merge([1, 3], mark_gaps=True) == 1
        assert sorted(dlg._store.tracks) == [1, 2]
        assert dlg._store.tracks[1].frames() == [0, 10, 80, 90]
        assert dlg._store.tracks[1].is_stop(10) is True
        _close(dlg)

    def test_the_multi_selection_survives_a_list_refresh(
            self, iface, dock, tmp_path):
        dlg = self._dialog_with_tracks(iface, dock, tmp_path)
        dlg._selected_track = 2
        for row in (0, 1):
            dlg.track_list.item(row).setSelected(True)
        dlg._refresh_track_list()
        assert dlg._selected_track_ids() == [1, 2]
        _close(dlg)


class TestPropagationSelection:
    """The geo-propagation acts on the list selection (several tracks)."""

    def test_every_selected_track_is_propagated(self, iface, dock, tmp_path):
        dlg = _dialog_with_tracks(iface, dock, tmp_path)
        for row in (0, 2):
            dlg.track_list.item(row).setSelected(True)
        assert [t.track_id for t in dlg._propagation_tracks()] == [1, 3]
        assert "2 tracks" in dlg.propagate_btn.text()
        _close(dlg)

    def test_it_falls_back_to_the_current_track(self, iface, dock, tmp_path):
        dlg = _dialog_with_tracks(iface, dock, tmp_path)
        dlg.track_list.clearSelection()
        dlg._selected_track = 2
        assert [t.track_id for t in dlg._propagation_tracks()] == [2]
        dlg._update_propagate_button()
        assert dlg.propagate_btn.text() == "Propagate box (geo)"
        _close(dlg)

    def test_the_report_lists_the_tracks_that_were_skipped(
            self, iface, dock, tmp_path):
        dlg = _dialog_with_tracks(iface, dock, tmp_path)
        track = dlg._store.tracks[1]
        report = dlg._propagation_report(
            [(track, ((0, 0, 10, 10), True, False), [(20, (1, 1, 2, 2))],
              [(30, "outside")])],
            missing=[2], errors=[(3, "no DEM hit")], src_frame=10)
        assert "Created 1 key frame(s) in 1 track(s)" in report
        assert "L1" in report and "30" in report      # partially skipped
        assert "L2: no box on frame 10" in report     # nothing to project
        assert "L3: no DEM hit" in report             # ray-cast failed
        _close(dlg)
