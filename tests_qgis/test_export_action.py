# -*- coding: utf-8 -*-
"""The Export action on the Processing tab (EXCHANGE_FORMAT_PLAN.md §8.1).

The exporters themselves are covered headlessly in ``tests/test_exporters.py``;
what needs a real QGIS is the wiring — that every format is offered, that the
per-format defaults are reflected rather than silently applied, and that a
format needing the CRS refuses before opening a file dialog.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox

from bambi_wildlife_detection.core import (
    detection_store, exporters, store, track_store)


@pytest.fixture
def project_folder(tmp_path):
    root = str(tmp_path / "flight")
    os.makedirs(root, exist_ok=True)
    detection_store.record_detections(root, "t", [
        {"frame": 0, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0,
         "confidence": 0.9, "source_class": "1"},
    ])
    ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
    track_store.record_georeference(root, "t", [
        {"detection_id": ids[0], "gx1": 500000.0, "gy1": 5300000.0,
         "gz1": 400.0, "gx2": 500010.0, "gy2": 5300010.0, "gz2": 400.0}])
    track_store.record_tracks(
        root, "t", [{"track_id": 1, "detection_id": ids[0]}])
    return root


@pytest.fixture
def quiet(monkeypatch):
    shown = {"info": [], "warn": [], "critical": []}
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: shown["info"].append(a))
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: shown["warn"].append(a))
    monkeypatch.setattr(QMessageBox, "critical",
                        lambda *a, **k: shown["critical"].append(a))
    return shown


# ---------------------------------------------------------------------------
# The widget
# ---------------------------------------------------------------------------

def test_every_format_is_offered(dock):
    keys = {dock.export_format_combo.itemData(i)
            for i in range(dock.export_format_combo.count())}
    assert keys == set(exporters.EXPORTERS)


def test_the_export_button_lives_with_the_animal_steps(dock):
    """Export describes detections and tracks, so it belongs in Processing."""
    assert dock.export_btn.parent() is not dock.extract_btn.parent()


@pytest.mark.parametrize("key", sorted(exporters.TRAINING_FORMATS))
def test_training_formats_default_to_dropping_false_positives(dock, key):
    index = [dock.export_format_combo.itemData(i)
             for i in range(dock.export_format_combo.count())].index(key)
    dock.export_format_combo.setCurrentIndex(index)
    assert not dock.export_false_positives_check.isChecked()


@pytest.mark.parametrize("key", ["geojson", "camtrap"])
def test_survey_formats_default_to_keeping_false_positives(dock, key):
    index = [dock.export_format_combo.itemData(i)
             for i in range(dock.export_format_combo.count())].index(key)
    dock.export_format_combo.setCurrentIndex(index)
    assert dock.export_false_positives_check.isChecked()


def test_darwin_core_disables_the_option_rather_than_ignoring_it(dock):
    index = [dock.export_format_combo.itemData(i)
             for i in range(dock.export_format_combo.count())].index("dwca")
    dock.export_format_combo.setCurrentIndex(index)
    assert not dock.export_false_positives_check.isEnabled()


# ---------------------------------------------------------------------------
# Refusals happen before a file dialog opens
# ---------------------------------------------------------------------------

def _select(dock, key):
    index = [dock.export_format_combo.itemData(i)
             for i in range(dock.export_format_combo.count())].index(key)
    dock.export_format_combo.setCurrentIndex(index)


def test_export_without_a_target_folder_is_refused(dock, quiet):
    dock.target_folder_edit.setText("")
    dock.run_export()
    assert quiet["warn"]


def test_a_geo_format_without_a_crs_is_refused(dock, quiet, monkeypatch,
                                               project_folder):
    """Refuse before the file dialog, not after writing wrong coordinates."""
    opened = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a, **k: opened.append(a) or "")
    dock.target_folder_edit.setText(project_folder)
    dock.target_crs_edit.setText("")
    _select(dock, "camtrap")

    dock.run_export()
    assert quiet["warn"]
    assert not opened, "the file dialog must not open when the CRS is missing"


def test_cancelling_the_dialog_writes_nothing(dock, quiet, monkeypatch,
                                              project_folder):
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    dock.target_folder_edit.setText(project_folder)
    _select(dock, "coco")
    dock.run_export()
    assert not quiet["info"]


# ---------------------------------------------------------------------------
# Running an export
# ---------------------------------------------------------------------------

def test_geojson_export_writes_the_file(dock, quiet, monkeypatch,
                                        project_folder, tmp_path):
    output = str(tmp_path / "out.geojson")
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: (output, ""))
    dock.target_folder_edit.setText(project_folder)
    dock.target_crs_edit.setText("EPSG:32633")
    _select(dock, "geojson")

    dock.run_export()
    assert os.path.isfile(output)
    assert quiet["info"]


def test_a_failed_export_is_explained_not_raised(dock, quiet, monkeypatch,
                                                 project_folder, tmp_path):
    """COCO needs the frame size; without extracted frames it must say so."""
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(tmp_path / "coco.json"), ""))
    dock.target_folder_edit.setText(project_folder)
    _select(dock, "coco")

    dock.run_export()
    assert quiet["warn"]
    assert any("frame size" in str(args) for args in quiet["warn"])


def test_export_of_an_empty_project_is_explained(dock, quiet, monkeypatch,
                                                 tmp_path):
    empty = str(tmp_path / "empty")
    os.makedirs(empty, exist_ok=True)
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(tmp_path / "x.geojson"), ""))
    dock.target_folder_edit.setText(empty)
    dock.target_crs_edit.setText("EPSG:32633")
    _select(dock, "geojson")

    dock.run_export()
    assert quiet["warn"]


def test_the_export_is_logged(dock, quiet, monkeypatch, project_folder,
                              tmp_path):
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(tmp_path / "out.geojson"), ""))
    dock.target_folder_edit.setText(project_folder)
    dock.target_crs_edit.setText("EPSG:32633")
    _select(dock, "geojson")

    dock.run_export()
    assert "Exporting" in dock.log_text.toPlainText()


def test_the_modality_selector_is_honoured(dock, quiet, monkeypatch,
                                           project_folder, tmp_path):
    """RGB has no detections here, so the export must refuse rather than
    silently write the thermal ones."""
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(tmp_path / "out.geojson"), ""))
    dock.target_folder_edit.setText(project_folder)
    dock.target_crs_edit.setText("EPSG:32633")
    _select(dock, "geojson")
    dock.export_camera_combo.setCurrentIndex(1)      # RGB

    dock.run_export()
    assert quiet["warn"]


def test_the_store_is_untouched_by_exporting(dock, quiet, monkeypatch,
                                             project_folder, tmp_path):
    """Exports are one-directional; nothing writes back."""
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(tmp_path / "out.geojson"), ""))
    dock.target_folder_edit.setText(project_folder)
    dock.target_crs_edit.setText("EPSG:32633")
    _select(dock, "geojson")
    dock.run_export()

    conn = store.open_store(
        store.stage_path(project_folder, store.DETECTIONS, "t"),
        store.DETECTIONS, "t")
    try:
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM detections").fetchone()["n"] == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# "Include images"
# ---------------------------------------------------------------------------

def test_the_images_option_is_on_by_default(dock):
    """An export that references images is incomplete without them."""
    assert dock.export_images_check.isChecked() is True


def test_the_option_is_disabled_for_formats_without_images(dock):
    """Disabled rather than hidden: it should read as "this format has no
    images", not as a missing option."""
    from bambi_wildlife_detection.core import exporters

    for index in range(dock.export_format_combo.count()):
        dock.export_format_combo.setCurrentIndex(index)
        key = dock.export_format_combo.currentData()
        assert dock.export_images_check.isEnabled() == (
            key in exporters.SUPPORTS_IMAGES), key


def test_the_choice_reaches_the_exporter(dock, quiet, monkeypatch,
                                         project_folder, tmp_path):
    seen = {}
    monkeypatch.setattr(exporters, "run_export",
                        lambda *a, **k: seen.update(k) or str(tmp_path))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a, **k: str(tmp_path / "out"))
    dock.target_folder_edit.setText(project_folder)
    _select(dock, "yolo")

    dock.export_images_check.setChecked(False)
    dock.run_export()
    assert seen["include_images"] is False

    dock.export_images_check.setChecked(True)
    dock.run_export()
    assert seen["include_images"] is True
