# -*- coding: utf-8 -*-
"""The detector class mapping dialog (EXCHANGE_FORMAT_PLAN.md §3.1).

The rules live headlessly in ``tests/test_schema_editor.py``; what needs a real
QGIS is that the table lists what the detector actually reported, that applying
a change re-resolves the detections already stored, and that cancelling leaves
them alone.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from qgis.PyQt.QtWidgets import QMessageBox

from bambi_wildlife_detection.bambi_class_mapping_dialog import (
    BambiClassMappingDialog)
from bambi_wildlife_detection.core import detection_store, store


@pytest.fixture
def project_folder(tmp_path):
    """A project whose detector reported two classes, neither mapped."""
    root = str(tmp_path / "flight")
    os.makedirs(root, exist_ok=True)
    detection_store.record_detections(root, "t", [
        {"frame": 0, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.9, "source_class": "0"},
        {"frame": 1, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.9, "source_class": "0"},
        {"frame": 2, "x1": 3.0, "y1": 3.0, "x2": 4.0, "y2": 4.0,
         "confidence": 0.8, "source_class": "3"},
    ])
    return root


@pytest.fixture
def quiet(monkeypatch):
    shown = {"info": [], "warn": []}
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: shown["info"].append(a))
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: shown["warn"].append(a))
    return shown


def _species_of(folder, modality="t"):
    conn = store.open_store(
        store.stage_path(folder, store.DETECTIONS, modality),
        store.DETECTIONS, modality)
    try:
        return [row["species_id"] for row in conn.execute(
            "SELECT species_id FROM detections ORDER BY detection_id")]
    finally:
        conn.close()


def _set_row(dialog, raw_class, species_name):
    for row in range(dialog.table.rowCount()):
        if dialog.table.item(row, 0).text() == raw_class:
            combo = dialog.table.cellWidget(row, 2)
            combo.setCurrentIndex(combo.findText(species_name))
            return
    raise AssertionError(f"class {raw_class} not offered")


# ---------------------------------------------------------------------------
# What the table shows
# ---------------------------------------------------------------------------

def test_the_table_lists_what_the_detector_reported(project_folder):
    """Not what someone already mapped — otherwise it starts empty."""
    dialog = BambiClassMappingDialog(project_folder)
    try:
        classes = [dialog.table.item(row, 0).text()
                   for row in range(dialog.table.rowCount())]
        assert classes == ["0", "3"]
    finally:
        dialog.reject()


def test_the_table_shows_how_often_each_class_was_seen(project_folder):
    dialog = BambiClassMappingDialog(project_folder)
    try:
        counts = {dialog.table.item(row, 0).text():
                  dialog.table.item(row, 1).text()
                  for row in range(dialog.table.rowCount())}
        assert counts == {"0": "2", "3": "1"}
    finally:
        dialog.reject()


def test_unmapped_classes_are_reported(project_folder):
    dialog = BambiClassMappingDialog(project_folder)
    try:
        assert "unmapped" in dialog.status.text()
    finally:
        dialog.reject()


def test_the_species_choices_are_the_projects_own(project_folder):
    dialog = BambiClassMappingDialog(project_folder)
    try:
        combo = dialog.table.cellWidget(0, 2)
        names = [combo.itemText(i) for i in range(combo.count())]
        assert "animal" in names and "roe deer" in names
    finally:
        dialog.reject()


def test_unmapped_classes_default_to_animal(project_folder):
    """A species-agnostic detector needs no configuration at all."""
    assert _species_of(project_folder) == [0, 0, 0]


def test_a_project_without_a_detector_says_so(tmp_path):
    root = str(tmp_path / "empty")
    os.makedirs(root, exist_ok=True)
    store.open_store(store.project_path(root), store.PROJECT).close()

    dialog = BambiClassMappingDialog(root)
    try:
        assert dialog.table.rowCount() == 0
        assert "no detector" in dialog.status.text().lower()
    finally:
        dialog.reject()


# ---------------------------------------------------------------------------
# Applying re-resolves what is already stored
# ---------------------------------------------------------------------------

def test_applying_reassigns_the_stored_detections(project_folder):
    """The raw class was kept, so the detector need not run again (§3.1)."""
    dialog = BambiClassMappingDialog(project_folder)
    _set_row(dialog, "0", "roe deer")
    _set_row(dialog, "3", "chamois")
    dialog.accept()

    assert _species_of(project_folder) == [1, 1, 5]
    assert dialog.rows_changed == 3


def test_cancelling_changes_nothing(project_folder):
    dialog = BambiClassMappingDialog(project_folder)
    _set_row(dialog, "0", "roe deer")
    dialog.reject()

    assert _species_of(project_folder) == [0, 0, 0]


def test_reopening_shows_the_saved_mapping(project_folder):
    dialog = BambiClassMappingDialog(project_folder)
    _set_row(dialog, "0", "roe deer")
    dialog.accept()

    dialog = BambiClassMappingDialog(project_folder)
    try:
        combo = dialog.table.cellWidget(0, 2)
        assert combo.currentText() == "roe deer"
        assert "unmapped" not in dialog.status.text()
    finally:
        dialog.reject()


def test_mapping_back_to_animal_is_allowed(project_folder):
    dialog = BambiClassMappingDialog(project_folder)
    _set_row(dialog, "0", "roe deer")
    dialog.accept()

    dialog = BambiClassMappingDialog(project_folder)
    _set_row(dialog, "0", "animal")
    dialog.accept()
    assert _species_of(project_folder)[0] == 0


# ---------------------------------------------------------------------------
# Reached from the Detection tab
# ---------------------------------------------------------------------------

def test_the_button_lives_on_the_detection_tab(dock):
    """It describes the model, not the survey vocabulary."""
    assert dock.class_mapping_btn.parent() is not \
        dock.schema_editor_btn.parent()


def test_the_dock_refuses_without_a_target_folder(dock, quiet):
    dock.target_folder_edit.setText("")
    assert dock.open_class_mapping_dialog() is False
    assert quiet["warn"]
