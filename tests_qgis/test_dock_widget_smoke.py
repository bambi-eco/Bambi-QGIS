# -*- coding: utf-8 -*-
"""Real-QGIS smoke tests for the dock widget.

The config round-trip test is the important one: it drives the real
schema-bound widgets and a real ``QgsProject`` through
save → reset → load, which is exactly the manual check the config-schema
refactor (core.config_schema) otherwise requires.
"""
import pytest
from qgis.core import QgsSettings
from qgis.PyQt.QtWidgets import QMessageBox

from bambi_wildlife_detection.bambi_dock_widget import PLUGIN_SCOPE
from bambi_wildlife_detection.core.config_schema import WIDGET_BINDINGS

LICENSE_KEY = f"{PLUGIN_SCOPE}/skipUltralyticsLicenseNotice"


class TestConstruction:
    def test_dock_constructs(self, dock):
        # A handful of schema-bound widgets must exist after setup_ui().
        assert dock.target_folder_edit is not None
        assert dock.confidence_spin is not None
        assert dock.tracker_backend_combo is not None

    def test_every_bound_widget_exists(self, dock):
        """The config schema references only widgets that really exist.

        This is the anti-drift guarantee the unit suite cannot give: a
        binding pointing at a renamed/removed widget fails here.
        """
        missing = [(key, attr) for key, (attr, _role) in WIDGET_BINDINGS.items()
                   if not hasattr(dock, attr)]
        assert missing == [], f"WIDGET_BINDINGS reference missing widgets: {missing}"

    def test_get_config_returns_dict(self, dock):
        config = dock.get_config()
        assert isinstance(config, dict)
        # a few keys the pipeline relies on
        for key in ("input_mode", "thermal_video_paths", "target_epsg"):
            assert key in config


def _save_as_an_old_version_did(dock):
    """The form as plain entries of the QGIS project - where versions before
    6.0 kept every setting, and versions before 9.1 those of a flight that
    had no folder to keep them in."""
    from qgis.core import QgsProject
    from bambi_wildlife_detection.core.config_schema import save_config_entries

    project = QgsProject.instance()
    save_config_entries(
        dock._current_config_values(),
        lambda key, value: project.writeEntry(PLUGIN_SCOPE, key, value),
        lambda key, value: project.writeEntryDouble(PLUGIN_SCOPE, key, value),
        lambda key, value: project.writeEntryBool(PLUGIN_SCOPE, key, value))


class TestConfigRoundTrip:
    """save_config_to_project → reset → load_config_from_project fidelity."""

    def _set_sample_values(self, dock):
        """Set a non-default value on one widget per role, return expectations."""
        dock.target_folder_edit.setText("/tmp/bambi_out")          # line
        dock.target_crs_edit.setText("EPSG:32610")                 # crs
        dock.confidence_spin.setValue(0.77)                        # spin (double)
        dock.detect_start_frame_spin.setValue(123)                 # spin (int)
        dock.detect_all_frames_check.setChecked(False)             # bool check
        dock.video_mode_check.setChecked(True)                     # bool01 check
        dock.tracker_params_edit.setPlainText('{"a": 1}')          # text
        # combo by index (roles combo_index)
        if dock.tracker_backend_combo.count() > 0:
            dock.tracker_backend_combo.setCurrentIndex(0)
        # combo by text (preset)
        if dock.thermal_calib_preset_combo.count() > 1:
            dock.thermal_calib_preset_combo.setCurrentIndex(1)
        preset_text = dock.thermal_calib_preset_combo.currentText()
        return {
            "target_folder": "/tmp/bambi_out",
            "target_crs": "EPSG:32610",
            "confidence": 0.77,
            "start_frame": 123,
            "all_frames": False,
            "video_mode": True,
            "tracker_params": '{"a": 1}',
            "preset_text": preset_text,
        }

    def test_roundtrip_preserves_values(self, dock):
        expected = self._set_sample_values(dock)

        dock.save_config_to_project()

        # Wipe the widgets back to their pristine construction state.
        dock.reset_config_to_defaults()
        assert dock.confidence_spin.value() != 0.77
        assert dock.target_folder_edit.text() != "/tmp/bambi_out"

        # Reload from the project and verify every kind survived.
        dock.load_config_from_project()
        assert dock.target_folder_edit.text() == expected["target_folder"]
        assert dock.target_crs_edit.text() == expected["target_crs"]
        assert dock.confidence_spin.value() == expected["confidence"]
        assert dock.detect_start_frame_spin.value() == expected["start_frame"]
        assert dock.detect_all_frames_check.isChecked() is expected["all_frames"]
        assert dock.video_mode_check.isChecked() is expected["video_mode"]
        assert dock.tracker_params_edit.toPlainText() == expected["tracker_params"]
        assert dock.thermal_calib_preset_combo.currentText() == expected["preset_text"]

    def test_empty_crs_loads_default(self, dock, clean_project):
        # A saved project with a blank CRS must load as the EPSG:32610-style
        # default (crs role behaviour) rather than an empty field.
        dock.target_folder_edit.setText("/tmp/has_config")
        dock.target_crs_edit.setText("")
        dock.save_config_to_project()
        dock.target_crs_edit.setText("EPSG:9999")   # dirty it
        dock.load_config_from_project()
        assert dock.target_crs_edit.text() == "EPSG:32633"

    def test_heading_source_and_margin_roundtrip(self, dock, clean_project):
        # combo_data role: the stable key is stored, not the label.
        dock.target_folder_edit.setText("/tmp/heading_roundtrip")
        dock.heading_source_combo.setCurrentIndex(
            dock.heading_source_combo.findData("srt_gimbal"))
        dock.airdata_padding_auto_check.setChecked(False)
        dock.airdata_padding_spin.setValue(3.0)
        dock.save_config_to_project()
        dock.reset_config_to_defaults()
        assert dock.heading_source_combo.currentData() == "airdata_compass"
        assert dock.airdata_padding_spin.value() == 0.0
        assert dock.airdata_padding_auto_check.isChecked()   # new projects: auto
        assert not dock.airdata_padding_spin.isEnabled()
        dock.load_config_from_project()
        assert dock.heading_source_combo.currentData() == "srt_gimbal"
        assert dock.airdata_padding_spin.value() == 3.0
        assert not dock.airdata_padding_auto_check.isChecked()
        assert dock.airdata_padding_spin.isEnabled()
        config = dock.get_config()
        assert config["heading_source"] == "srt_gimbal"
        assert "use_gimbal_heading" not in config
        assert config["airdata_video_padding_ms"] == 3000
        assert config["airdata_video_padding_auto"] is False

    def test_auto_margin_roundtrip(self, dock, clean_project):
        dock.target_folder_edit.setText("/tmp/margin_auto")
        assert dock.airdata_padding_auto_check.isChecked()
        dock.save_config_to_project()
        dock.airdata_padding_auto_check.setChecked(False)
        dock.load_config_from_project()
        assert dock.airdata_padding_auto_check.isChecked()
        assert dock.get_config()["airdata_video_padding_auto"] is True

    def test_project_without_auto_setting_keeps_fixed_margin(self, dock, clean_project):
        # Saved before "Auto" existed: no Extraction/AirdataPaddingAuto entry.
        from qgis.core import QgsProject
        dock.target_folder_edit.setText("/tmp/margin_legacy")
        _save_as_an_old_version_did(dock)
        QgsProject.instance().removeEntry(PLUGIN_SCOPE, "Extraction/AirdataPaddingAuto")
        dock.load_config_from_project()
        assert not dock.airdata_padding_auto_check.isChecked()
        assert dock.get_config()["airdata_video_padding_ms"] == 0

    def test_project_from_7_0_with_gimbal_checkbox(self, dock, clean_project):
        # Before 7.1 the heading was a checkbox (Extraction/UseGimbalHeading).
        from qgis.core import QgsProject
        dock.target_folder_edit.setText("/tmp/heading_legacy")
        _save_as_an_old_version_did(dock)
        project = QgsProject.instance()
        project.removeEntry(PLUGIN_SCOPE, "Extraction/HeadingSource")
        project.writeEntry(PLUGIN_SCOPE, "Extraction/UseGimbalHeading", "1")
        dock.load_config_from_project()
        assert dock.heading_source_combo.currentData() == "airdata_gimbal"

    def test_load_without_saved_config_is_noop(self, dock):
        # A fresh project (no Input/TargetFolder) must not raise or clobber.
        dock.confidence_spin.setValue(0.42)
        dock.load_config_from_project()   # nothing saved -> early return
        assert dock.confidence_spin.value() == 0.42


class TestUltralyticsLicenseNotice:
    """_confirm_ultralytics_license gating before the detection stage."""

    @pytest.fixture(autouse=True)
    def _clean_license_setting(self):
        settings = QgsSettings()
        settings.remove(LICENSE_KEY)
        yield
        settings.remove(LICENSE_KEY)

    def test_stored_skip_bypasses_dialog(self, dock, monkeypatch):
        QgsSettings().setValue(LICENSE_KEY, True)

        def fail_exec(box):
            raise AssertionError("dialog must not be shown when skip is stored")

        monkeypatch.setattr(QMessageBox, "exec", fail_exec)
        assert dock._confirm_ultralytics_license() is True

    def test_ok_with_remember_persists_skip(self, dock, monkeypatch):
        def fake_exec(box):
            box.checkBox().setChecked(True)
            box.button(QMessageBox.StandardButton.Ok).click()

        monkeypatch.setattr(QMessageBox, "exec", fake_exec)
        assert dock._confirm_ultralytics_license() is True
        assert QgsSettings().value(LICENSE_KEY, False, type=bool) is True

    def test_ok_without_remember_stores_nothing(self, dock, monkeypatch):
        def fake_exec(box):
            box.button(QMessageBox.StandardButton.Ok).click()

        monkeypatch.setattr(QMessageBox, "exec", fake_exec)
        assert dock._confirm_ultralytics_license() is True
        assert QgsSettings().value(LICENSE_KEY, False, type=bool) is False

    def test_cancel_blocks_and_never_persists(self, dock, monkeypatch):
        # Even with "Remember my decision" ticked, cancelling must not be
        # stored - otherwise detection would be silently blocked forever.
        def fake_exec(box):
            box.checkBox().setChecked(True)
            box.button(QMessageBox.StandardButton.Cancel).click()

        monkeypatch.setattr(QMessageBox, "exec", fake_exec)
        assert dock._confirm_ultralytics_license() is False
        assert QgsSettings().value(LICENSE_KEY, False, type=bool) is False


class TestResetToDefaults:
    def test_reset_restores_startup_state(self, dock):
        default_conf = dock.confidence_spin.value()
        dock.confidence_spin.setValue(0.9)
        dock.target_folder_edit.setText("/changed")
        dock.reset_config_to_defaults()
        assert dock.confidence_spin.value() == default_conf
        assert dock.target_folder_edit.text() != "/changed"


def test_the_legacy_switch_says_it_is_going_away(dock):
    """It is the one place a user learns the .txt/.csv files are temporary."""
    label = dock.legacy_text_outputs_check.text()
    assert ".txt" in label and ".csv" in label
    assert "removed" in label.lower()


def test_the_legacy_switch_no_longer_claims_tools_depend_on_it(dock):
    """They did until the readers were moved to the store; a tooltip that
    still said so would keep people from ever turning it off."""
    tip = dock.legacy_text_outputs_check.toolTip()
    assert "Video Creator" not in tip
    assert "Nothing in the plugin reads them" in tip


def test_flight_log_margin_info_explains_why_and_auto(dock, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box: shown.append(box.text()))
    dock._airdata_padding_info_btn.click()
    assert len(shown) == 1
    text = shown[0]
    assert "Why it is needed" in text and "extrapolated" in text
    assert "Auto" in text and "plus 1 s" in text


# ---------------------------------------------------------------------------
# The detected time zone
# ---------------------------------------------------------------------------

def _night_flight(tmp_path):
    """A thermal video from 00:30 local (UTC+2) and its flight log."""
    from datetime import datetime, timedelta

    local, utc = datetime(2026, 6, 15, 0, 30), datetime(2026, 6, 14, 22, 30)
    srt, rows = [], ["time,isVideo,datetime(utc)"]
    for second in range(0, 600, 20):
        step = timedelta(seconds=second)
        srt += [str(second), "00:00:00,000 --> 00:00:00,033",
                (local + step).strftime("%Y-%m-%d %H:%M:%S"), ""]
        rows.append(f"1,1,{(utc + step).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    srt_path = tmp_path / "DJI_0001_T.SRT"
    srt_path.write_text("\n".join(srt))
    log_path = tmp_path / "airdata.csv"
    log_path.write_text("\n".join(rows))
    return str(srt_path), str(log_path)


def test_the_time_zone_of_a_thermal_night_flight_is_detected(dock, tmp_path):
    """Only the RGB SRT files were read, and the times of day compared:
    a thermal-only flight had no detection, and after midnight it was
    UTC-22."""
    srt, log = _night_flight(tmp_path)
    dock.thermal_srt_paths_edit.setText(srt)
    dock.airdata_path_edit.setText(log)

    assert dock._compute_auto_timezone_offset() == 2.0
    assert dock.get_config()["timezone_offset_hours"] == 2.0
    dock._refresh_auto_timezone_offset()
    assert dock.tz_auto_result_label.text() == "Detected offset: UTC+2h"


def test_the_detection_is_not_repeated_for_every_query(dock, tmp_path,
                                                       monkeypatch):
    """Every get_config() read all SRT files and the flight log again."""
    from bambi_wildlife_detection.core import timezone_detection

    srt, log = _night_flight(tmp_path)
    dock.thermal_srt_paths_edit.setText(srt)
    dock.airdata_path_edit.setText(log)
    calls = []
    real = timezone_detection.detect_offset
    monkeypatch.setattr(
        timezone_detection, "detect_offset",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    for _ in range(5):
        assert dock.get_config()["timezone_offset_hours"] == 2.0
    assert len(calls) == 1

    with open(srt, "a") as handle:                   # the file changes
        handle.write("\n999\n00:00:00,000 --> 00:00:00,033\n"
                     "2026-06-15 00:45:00\n")
    dock.get_config()
    assert len(calls) == 2


def test_headings_are_turned_to_the_maps_north_unless_unticked(dock):
    """The extraction setting for the grid convergence (core.grid_north):
    ticked for a new flight, and part of what a step is given."""
    assert dock.grid_convergence_check.isChecked()
    assert dock.get_config()["correct_grid_convergence"] is True

    dock.grid_convergence_check.setChecked(False)
    try:
        assert dock.get_config()["correct_grid_convergence"] is False
    finally:
        dock.grid_convergence_check.setChecked(True)
