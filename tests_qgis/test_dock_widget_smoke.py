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
        # stored — otherwise detection would be silently blocked forever.
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
