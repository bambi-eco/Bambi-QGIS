# -*- coding: utf-8 -*-
"""The Reset Step action.

The cascade itself is covered headlessly in ``tests/test_stages.py``; what needs
a real QGIS is the wiring - that the button exists, refuses sensibly, and that
a locked file produces an explanation rather than a traceback.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from qgis.PyQt.QtWidgets import QInputDialog, QMessageBox

from bambi_wildlife_detection.core import stages, store


@pytest.fixture
def project_folder(tmp_path):
    root = str(tmp_path / "flight")
    os.makedirs(root, exist_ok=True)
    store.open_store(store.project_path(root), store.PROJECT).close()
    for kind in (store.DETECTIONS, store.GEOREFERENCED, store.TRACKS):
        store.open_store(
            store.stage_path(root, kind, "t"), kind, "t").close()
    return root


def _choose(monkeypatch, label):
    """Answer the step picker with *label*; returns what it offered."""
    offered = []

    def get_item(_parent, _title, _prompt, items, *_args, **_kwargs):
        offered.extend(items)
        return label, True

    monkeypatch.setattr(QInputDialog, "getItem", get_item)
    return offered


@pytest.fixture
def quiet(monkeypatch):
    """Stub the modal dialogs - they abort a headless run."""
    shown = {"info": [], "warn": [], "question": []}
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: shown["info"].append(a))
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: shown["warn"].append(a))
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **k: (shown["question"].append(a),
                         QMessageBox.StandardButton.Ok)[1])
    return shown


def test_button_exists(dock):
    assert dock.reset_stage_btn is not None


def test_refuses_without_a_target_folder(dock, quiet):
    dock.target_folder_edit.setText("")
    dock.reset_stage()
    assert quiet["warn"], "the user should be told why nothing happened"


def test_reports_when_there_is_nothing_to_reset(dock, quiet, tmp_path):
    empty = str(tmp_path / "empty")
    os.makedirs(empty, exist_ok=True)
    dock.target_folder_edit.setText(empty)
    dock.reset_stage()
    assert quiet["info"], "the user should be told there is nothing to reset"


def test_resets_the_chosen_stage(dock, quiet, monkeypatch, project_folder):
    _choose(monkeypatch, "Track Animals - Thermal")

    dock.target_folder_edit.setText(project_folder)
    dock.reset_stage()

    assert not stages.has_output(project_folder, "tracking", "t")
    assert stages.has_output(project_folder, "detection", "t")


@pytest.fixture
def both_cameras(project_folder):
    for kind in (store.DETECTIONS, store.TRACKS):
        store.open_store(
            store.stage_path(project_folder, kind, "w"), kind, "w").close()
    store.open_store(
        store.matches_path(project_folder), store.MATCHES).close()
    return project_folder


def test_the_picker_offers_each_cameras_steps(dock, quiet, monkeypatch,
                                              both_cameras):
    offered = _choose(monkeypatch, "nothing of that name")

    dock.target_folder_edit.setText(both_cameras)
    dock.reset_stage()

    assert offered == [
        "Detect Animals - Thermal", "Detect Animals - RGB",
        "Geo-Reference Detections - Thermal",
        "Match RGB ↔ Thermal Tracks - both cameras",
        "Track Animals - Thermal", "Track Animals - RGB"]


@pytest.mark.parametrize("camera", ["t", "w"])
def test_the_rgb_camera_can_be_reset(dock, quiet, monkeypatch, both_cameras,
                                     camera):
    """The camera was read off a combo box in a way that always gave
    thermal: the RGB results could not be reset, and with the RGB camera
    selected the thermal ones were."""
    dock.detection_camera_combo.setCurrentIndex(1 if camera == "w" else 0)
    _choose(monkeypatch, "Track Animals - RGB")

    dock.target_folder_edit.setText(both_cameras)
    dock.reset_stage()

    assert not stages.has_output(both_cameras, "tracking", "w")
    assert stages.has_output(both_cameras, "tracking", "t")
    assert stages.has_output(both_cameras, "detection", "w")


def test_the_question_names_what_depends_on_the_step(dock, quiet, monkeypatch,
                                                     both_cameras):
    _choose(monkeypatch, "Detect Animals - RGB")

    dock.target_folder_edit.setText(both_cameras)
    dock.reset_stage()

    (question,) = quiet["question"]
    text = question[2]
    assert "'Detect Animals - RGB'" in text
    assert "Track Animals - RGB" in text
    assert "Match RGB ↔ Thermal Tracks - both cameras" in text
    assert "Thermal\n" not in text and not text.endswith("Thermal")


def test_the_match_of_both_cameras_is_reset_once(dock, quiet, monkeypatch,
                                                 both_cameras):
    stages.set_state(both_cameras, "track_matching", stages.CROSS_MODAL,
                     stages.COMPLETE)
    _choose(monkeypatch, "Match RGB ↔ Thermal Tracks - both cameras")

    dock.target_folder_edit.setText(both_cameras)
    dock.reset_stage()

    assert not os.path.exists(store.matches_path(both_cameras))
    assert stages.states(both_cameras, stages.CROSS_MODAL)[
        "track_matching"]["state"] == stages.PENDING
    assert stages.has_output(both_cameras, "tracking", "t")


def test_dependents_are_kept_but_flagged(dock, quiet, monkeypatch,
                                         project_folder):
    stages.mark_complete(project_folder, "tracking", "t")
    _choose(monkeypatch, "Detect Animals - Thermal")

    dock.target_folder_edit.setText(project_folder)
    dock.reset_stage()

    assert not stages.has_output(project_folder, "detection", "t")
    assert stages.has_output(project_folder, "tracking", "t")   # kept
    assert stages.states(project_folder, "t")["tracking"]["state"] == \
        stages.STALE


def test_cancelling_the_picker_changes_nothing(dock, quiet, monkeypatch,
                                               project_folder):
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("", False))
    dock.target_folder_edit.setText(project_folder)
    dock.reset_stage()
    assert stages.has_output(project_folder, "tracking", "t")


def test_a_locked_file_is_explained(dock, quiet, monkeypatch, project_folder):
    """Windows locks a file open in the QGIS Browser."""
    _choose(monkeypatch, "Track Animals - Thermal")

    def refuse(_path):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(os, "remove", refuse)
    dock.target_folder_edit.setText(project_folder)
    dock.reset_stage()

    assert quiet["warn"], "a locked file must be explained, not raised"
    assert any("QGIS Browser" in str(args) for args in quiet["warn"])
