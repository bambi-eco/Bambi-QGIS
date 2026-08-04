# -*- coding: utf-8 -*-
"""The Reset Step action (EXCHANGE_FORMAT_PLAN.md §7, §10 Phase 5).

The cascade itself is covered headlessly in ``tests/test_stages.py``; what needs
a real QGIS is the wiring — that the button exists, refuses sensibly, and that
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


@pytest.fixture
def quiet(monkeypatch):
    """Stub the modal dialogs — they abort a headless run."""
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
    monkeypatch.setattr(
        QInputDialog, "getItem",
        lambda *a, **k: (dock.STAGE_LABELS["tracking"], True))

    dock.target_folder_edit.setText(project_folder)
    dock.reset_stage()

    assert not stages.has_output(project_folder, "tracking", "t")
    assert stages.has_output(project_folder, "detection", "t")


def test_dependents_are_kept_but_flagged(dock, quiet, monkeypatch,
                                         project_folder):
    stages.mark_complete(project_folder, "tracking", "t")
    monkeypatch.setattr(
        QInputDialog, "getItem",
        lambda *a, **k: (dock.STAGE_LABELS["detection"], True))

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
    """Windows locks a file open in the QGIS Browser (§13, Q1)."""
    monkeypatch.setattr(
        QInputDialog, "getItem",
        lambda *a, **k: (dock.STAGE_LABELS["tracking"], True))

    def refuse(_path):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(os, "remove", refuse)
    dock.target_folder_edit.setText(project_folder)
    dock.reset_stage()

    assert quiet["warn"], "a locked file must be explained, not raised"
    assert any("QGIS Browser" in str(args) for args in quiet["warn"])
