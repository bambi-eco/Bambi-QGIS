# -*- coding: utf-8 -*-
"""The Classification configuration tab, in a real QGIS.

Two things are worth pinning here that the unit suite cannot reach: that the
tab actually builds, and that the Hugging Face token goes to the QGIS settings
rather than into the project file — a project gets shared, and a credential
must not travel with it.
"""
import json

import pytest

from qgis.PyQt.QtWidgets import QLineEdit, QTabWidget
from qgis.core import QgsSettings

from bambi_wildlife_detection.bambi_dock_widget import (
    _HF_TOKEN_SETTING, HF_DEFAULT_BACKBONE,
)
from bambi_wildlife_detection.core import config_schema, hf_access


def _config_tabs(dock):
    for widget in dock.findChildren(QTabWidget):
        names = [widget.tabText(i) for i in range(widget.count())]
        if "Tracking" in names:
            return widget
    raise AssertionError("configuration sub-tab widget not found")


@pytest.fixture(autouse=True)
def _clean_token():
    """Never leave a token behind in the test runner's QGIS settings."""
    QgsSettings().remove(_HF_TOKEN_SETTING)
    yield
    QgsSettings().remove(_HF_TOKEN_SETTING)


# ---------------------------------------------------------------------------
# The tab exists and is placed where the plan says
# ---------------------------------------------------------------------------

def test_classification_tab_exists(dock):
    names = [t.tabText(i) for t in [_config_tabs(dock)]
             for i in range(t.count())]
    assert "Classification" in names


def test_classification_follows_tracking(dock):
    tabs = _config_tabs(dock)
    names = [tabs.tabText(i) for i in range(tabs.count())]
    assert names.index("Classification") == names.index("Tracking") + 1


@pytest.mark.parametrize("attribute", [
    "hf_token_edit",
    "hf_show_token_check",
    "hf_check_access_btn",
    "hf_access_status",
    "classification_backbone_edit",
    "classification_projection_combo",
    "classification_device_combo",
    "classification_batch_spin",
    "classification_fp16_check",
])
def test_widget_exists(dock, attribute):
    assert getattr(dock, attribute) is not None


# ---------------------------------------------------------------------------
# The token is a credential, not project configuration
# ---------------------------------------------------------------------------

def test_token_is_masked_by_default(dock):
    assert dock.hf_token_edit.echoMode() == QLineEdit.EchoMode.Password


def test_show_token_reveals_and_hides(dock):
    dock.hf_show_token_check.setChecked(True)
    assert dock.hf_token_edit.echoMode() == QLineEdit.EchoMode.Normal
    dock.hf_show_token_check.setChecked(False)
    assert dock.hf_token_edit.echoMode() == QLineEdit.EchoMode.Password


def test_token_round_trips_through_qgis_settings(dock):
    dock.hf_token_edit.setText("hf_secret_value")
    dock._save_hf_token()
    assert QgsSettings().value(_HF_TOKEN_SETTING, "", type=str) == \
        "hf_secret_value"

    dock.hf_token_edit.setText("")
    dock._load_hf_token()
    assert dock.hf_token_edit.text() == "hf_secret_value"


def test_token_never_reaches_the_saved_configuration(dock, tmp_path):
    """The whole point of QSettings: saving a project must not leak the token."""
    dock.hf_token_edit.setText("hf_secret_value")
    dock._save_hf_token()
    dock.target_folder_edit.setText(str(tmp_path))
    dock.save_config_to_project()

    from bambi_wildlife_detection.core import flights as flights_core

    stored = flights_core.load_config(str(tmp_path))
    serialised = json.dumps(stored)
    assert "hf_secret_value" not in serialised
    assert not any("token" in key.lower() for key in stored)


def test_token_survives_a_configuration_reset(dock):
    """Resetting this project's settings must not log the user out."""
    dock.hf_token_edit.setText("hf_secret_value")
    dock._save_hf_token()
    dock.reset_config_to_defaults()
    assert QgsSettings().value(_HF_TOKEN_SETTING, "", type=str) == \
        "hf_secret_value"


# ---------------------------------------------------------------------------
# Config round-trip, including the json role
# ---------------------------------------------------------------------------

def test_backbone_falls_back_to_the_default_when_empty(dock):
    dock.classification_backbone_edit.setText("")
    assert dock.get_config()["classification_backbone"] == HF_DEFAULT_BACKBONE
    assert HF_DEFAULT_BACKBONE == hf_access.DEFAULT_BACKBONE


def test_projection_reports_the_repo_folder_not_the_label(dock):
    for index, expected in enumerate(("non_geo", "geo_1k", "geo_2k")):
        dock.classification_projection_combo.setCurrentIndex(index)
        assert dock.get_config()["classification_projection"] == expected


def test_device_reports_a_stable_value(dock):
    for index, expected in enumerate(("auto", "cpu", "cuda")):
        dock.classification_device_combo.setCurrentIndex(index)
        assert dock.get_config()["classification_device"] == expected


def test_projection_combo_covers_every_published_variant(dock):
    assert dock.classification_projection_combo.count() == \
        len(hf_access.PROJECTIONS)


def test_model_mapping_round_trips_through_the_json_role(dock):
    mapping = {"sex": {"modality": "matched", "model": "default",
                       "class_labels": ["female_juvenile", "male"],
                       "labels": {"0": 1, "1": 2}}}
    dock._classification_models = mapping
    # A folder that does not exist keeps both directions on the QGIS-project
    # path, which is what a dock with no registered flight reads back from.
    dock.target_folder_edit.setText("/tmp/bambi_classification_roundtrip")
    dock.save_config_to_project()

    dock._classification_models = {}
    dock.load_config_from_project()

    # Loading refreshes the editing table, which normalises every task into
    # the mapping — so the check is that nothing was *lost*, not that the dict
    # is byte-identical.
    restored = dock._classification_models["sex"]
    assert restored["modality"] == "matched"
    assert restored["model"] == "default"
    assert restored["class_labels"] == ["female_juvenile", "male"]
    assert restored["labels"] == {"0": 1, "1": 2}


def test_loading_does_not_lose_a_label_mapping(dock):
    """The table only knows about input/model/file, so it must leave the
    class mapping it does not edit alone."""
    dock._classification_models = {
        "occlusion": {"modality": "rgb", "model": "custom",
                      "path": "/m/occ.pt",
                      "class_labels": ["visible", "hidden"],
                      "labels": {"0": 0, "1": 1},
                      "clear_labels": ["visible"], "clear_values": [0]}}
    dock.target_folder_edit.setText("/tmp/bambi_labels_preserved")
    dock.save_config_to_project()
    dock._classification_models = {}
    dock.load_config_from_project()

    restored = dock._classification_models["occlusion"]
    assert restored["clear_labels"] == ["visible"]
    assert restored["clear_values"] == [0]
    assert restored["class_labels"] == ["visible", "hidden"]


def test_unreadable_model_mapping_falls_back_to_empty(dock):
    """A corrupted entry must not stop the rest of the configuration loading."""
    attr, role = config_schema.WIDGET_BINDINGS["Classification/Models"]
    dock._apply_config_value(attr, role, "{not json")
    assert dock._classification_models == {}


def test_classification_settings_survive_a_project_round_trip(dock):
    dock.classification_backbone_edit.setText("facebook/custom-backbone")
    dock.classification_projection_combo.setCurrentIndex(2)
    dock.classification_device_combo.setCurrentIndex(1)
    dock.classification_batch_spin.setValue(64)
    dock.classification_fp16_check.setChecked(False)
    dock.target_folder_edit.setText("/tmp/bambi_classification_settings")
    dock.save_config_to_project()

    dock.classification_backbone_edit.setText("")
    dock.classification_projection_combo.setCurrentIndex(0)
    dock.classification_device_combo.setCurrentIndex(0)
    dock.classification_batch_spin.setValue(16)
    dock.classification_fp16_check.setChecked(True)

    dock.load_config_from_project()
    assert dock.classification_backbone_edit.text() == "facebook/custom-backbone"
    assert dock.classification_projection_combo.currentIndex() == 2
    assert dock.classification_device_combo.currentIndex() == 1
    assert dock.classification_batch_spin.value() == 64
    assert dock.classification_fp16_check.isChecked() is False


# ---------------------------------------------------------------------------
# Access check
# ---------------------------------------------------------------------------

def test_matching_settings_survive_a_project_round_trip(dock):
    dock.match_min_shared_spin.setValue(12)
    dock.match_gate_spin.setValue(64.0)
    dock.match_min_confidence_spin.setValue(0.35)
    dock.match_max_time_offset_spin.setValue(0.25)
    dock.match_thermal_anchored_check.setChecked(False)
    dock.target_folder_edit.setText("/tmp/bambi_matching_settings")
    dock.save_config_to_project()

    dock.match_min_shared_spin.setValue(8)
    dock.match_gate_spin.setValue(28.0)
    dock.match_thermal_anchored_check.setChecked(True)

    dock.load_config_from_project()
    assert dock.match_min_shared_spin.value() == 12
    assert dock.match_gate_spin.value() == 64.0
    assert dock.match_min_confidence_spin.value() == 0.35
    assert dock.match_max_time_offset_spin.value() == 0.25
    assert dock.match_thermal_anchored_check.isChecked() is False


def test_matching_defaults_match_the_published_gates(dock):
    from bambi_wildlife_detection.core.track_matching import MatchConfig

    defaults = MatchConfig()
    config = dock.get_config()
    assert config["match_min_shared"] == defaults.min_shared
    assert config["match_gate_px"] == defaults.gate_px
    assert config["match_min_confidence"] == defaults.min_confidence


def test_matching_step_and_layer_buttons_exist(dock):
    assert dock.track_matching_btn.text().strip().startswith("C1.")
    assert dock.add_matches_btn is not None
    assert dock.track_matching_status is not None


def test_matching_needs_both_modalities(dock, tmp_path, monkeypatch):
    """It is the one step that is inherently about the pair, so there is no
    camera combo — and it must say so rather than half-run."""
    warned = {}

    def _warn(_parent, title, text):
        warned["title"] = title
        warned["text"] = text

    monkeypatch.setattr(
        "bambi_wildlife_detection.bambi_dock_widget.QMessageBox.warning",
        _warn)
    started = []
    monkeypatch.setattr(dock, "start_worker", started.append)

    dock.target_folder_edit.setText(str(tmp_path))
    dock.run_track_matching()

    assert started == []
    assert "Prerequisites" in warned["title"]
    assert "both" in warned["text"]


def test_crop_settings_survive_a_project_round_trip(dock):
    dock.classification_padding_spin.setValue(0.25)
    dock.classification_crop_size_spin.setValue(336)
    dock.classification_letterbox_check.setChecked(False)
    dock.target_folder_edit.setText("/tmp/bambi_crop_settings")
    dock.save_config_to_project()

    dock.classification_padding_spin.setValue(0.10)
    dock.classification_crop_size_spin.setValue(224)
    dock.classification_letterbox_check.setChecked(True)

    dock.load_config_from_project()
    assert dock.classification_padding_spin.value() == 0.25
    assert dock.classification_crop_size_spin.value() == 336
    assert dock.classification_letterbox_check.isChecked() is False


def test_crop_defaults_match_the_core_defaults(dock):
    from bambi_wildlife_detection.core.classification import CropConfig

    defaults = CropConfig()
    config = dock.get_config()
    assert config["classification_crop_padding"] == defaults.padding
    assert config["classification_crop_size"] == defaults.size
    assert config["classification_letterbox"] == defaults.letterbox


def test_embedding_step_row_exists(dock):
    assert dock.embeddings_btn.text().strip().startswith("C2.")
    assert dock.embeddings_camera_combo.count() == 2
    assert dock.embeddings_status is not None


def test_embeddings_needs_tracking_first(dock, tmp_path, monkeypatch):
    warned = {}
    monkeypatch.setattr(
        "bambi_wildlife_detection.bambi_dock_widget.QMessageBox.warning",
        lambda _p, title, text: warned.update(title=title, text=text))
    started = []
    monkeypatch.setattr(dock, "start_worker", started.append)

    dock.target_folder_edit.setText(str(tmp_path))
    dock.run_embeddings()

    assert started == []
    assert "tracking" in warned["text"].lower()


def test_result_writing_is_on_by_default(dock):
    """Otherwise the classifications would be invisible everywhere else."""
    assert dock.classification_write_check.isChecked() is True
    assert dock.get_config()["classification_write_results"] is True


def test_the_detector_species_is_protected_by_default(dock):
    """A configured class mapping must not be quietly undone."""
    assert dock.classification_overwrite_check.isChecked() is False
    assert dock.get_config()["classification_overwrite_species"] is False


def test_the_apply_action_exists(dock):
    assert dock.apply_results_btn is not None
    assert dock.apply_results_status is not None


def test_applying_without_results_says_so(dock, tmp_path, monkeypatch):
    from bambi_wildlife_detection.core import store

    warned = {}
    monkeypatch.setattr(
        "bambi_wildlife_detection.bambi_dock_widget.QMessageBox.warning",
        lambda _p, title, text: warned.update(title=title, text=text))

    store.open_store(store.project_path(str(tmp_path)),
                     store.PROJECT).close()
    dock.target_folder_edit.setText(str(tmp_path))
    dock.apply_classification_results()

    assert "No Results" in warned["title"]


def test_voting_settings_survive_a_project_round_trip(dock):
    dock.classification_quorum_spin.setValue(0.75)
    dock.classification_min_frames_spin.setValue(5)
    dock.classification_frames_combo.setCurrentIndex(1)
    dock.classification_unmatched_combo.setCurrentIndex(2)
    dock.classification_write_check.setChecked(False)
    dock.life_stage_z_spin.setValue(-3.5)
    dock.target_folder_edit.setText("/tmp/bambi_voting_settings")
    dock.save_config_to_project()

    dock.classification_quorum_spin.setValue(0.5)
    dock.classification_min_frames_spin.setValue(1)
    dock.classification_frames_combo.setCurrentIndex(0)
    dock.classification_write_check.setChecked(True)
    dock.life_stage_z_spin.setValue(-2.0)

    dock.load_config_from_project()
    assert dock.classification_quorum_spin.value() == 0.75
    assert dock.classification_min_frames_spin.value() == 5
    assert dock.get_config()["classification_frame_selection"] == "all"
    assert dock.get_config()["classification_unmatched"] == "thermal"
    assert dock.classification_write_check.isChecked() is False
    assert dock.life_stage_z_spin.value() == -3.5


def test_check_access_reports_without_raising(dock, monkeypatch):
    """The button answers even when huggingface_hub is absent or offline."""
    monkeypatch.setattr(
        hf_access, "check_repo_access",
        lambda repo, token: {"repo": repo, "status": hf_access.ACCESS_GRANTED,
                             "message": "ok"})
    dock.hf_token_edit.setText("hf_x")
    dock.check_hf_access()
    assert "granted" in dock.hf_access_status.text().lower()
    assert dock.hf_check_access_btn.isEnabled()


def test_check_access_reenables_the_button_after_a_failure(dock, monkeypatch):
    def _boom(repo, token):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(hf_access, "check_repo_access", _boom)
    dock.hf_token_edit.setText("hf_x")
    with pytest.raises(RuntimeError):
        dock.check_hf_access()
    # A raising check must not leave the button dead for the rest of the session.
    assert dock.hf_check_access_btn.isEnabled()
