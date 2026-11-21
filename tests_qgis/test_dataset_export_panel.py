# -*- coding: utf-8 -*-
"""Survey Analytics > Export > Dataset Export: the flights' table, the
splits and what is handed to the step.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from qgis.PyQt.QtWidgets import QFileDialog, QInputDialog, QMessageBox
from qgis.core import QgsProject

from bambi_wildlife_detection.core import dataset_export, flights as fc


NAMES = ("Alder", "Birch", "Cedar", "Dogwood")


@pytest.fixture
def flights(dock, tmp_path, monkeypatch):
    """Four flights of the project, all ticked for the survey."""
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(type(dock), "_ask_new_flight_configuration",
                        lambda self: True)
    made = []
    for name in NAMES:
        folder = tmp_path / name / "qgis"
        folder.mkdir(parents=True)
        monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                            lambda *a, _f=str(folder), **k: _f)
        monkeypatch.setattr(QInputDialog, "getText",
                            lambda *a, _n=name, **k: (_n, True))
        dock.browse_and_add_flight()
        dock.target_crs_edit.setText("EPSG:32633")
        dock.save_config_to_project()
        made.append({"name": name, "target_folder": str(folder)})
    dock.survey_panel.refresh()
    dock.survey_panel._tick_all(True)
    dock.survey_panel.folder_edit.setText(str(tmp_path / "survey"))
    dock._refresh_dataset_flights()
    return made


@pytest.fixture
def warned(monkeypatch):
    shown = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        lambda _parent, _title, text, *a, **k: shown.append(text))
    return shown


@pytest.fixture
def started(dock, monkeypatch):
    steps = []
    monkeypatch.setattr(dock, "start_worker", steps.append)
    return steps


def rows(dock):
    table = dock.dataset_flights_table
    return [(table.item(row, 0).text(),
             table.cellWidget(row, 1).currentData())
            for row in range(table.rowCount())]


def test_the_ticked_flights_are_listed_and_trained_on(dock, flights):
    assert rows(dock) == [(name, "train") for name in NAMES]
    assert dock.dataset_splits_label.text() == "4 train, 0 validation, 0 test"


def test_a_flight_is_given_its_split_in_the_table(dock, flights):
    combo = dock.dataset_flights_table.cellWidget(1, 1)
    combo.setCurrentIndex(combo.findData("val"))
    combo = dock.dataset_flights_table.cellWidget(3, 1)
    combo.setCurrentIndex(combo.findData(dataset_export.LEFT_OUT))

    assert dock.dataset_splits() == {
        "Alder": "train", "Birch": "val", "Cedar": "train", "Dogwood": ""}
    assert dock.dataset_splits_label.text() == \
        "2 train, 1 validation, 0 test, 1 left out"


def test_assign_deals_whole_flights_by_the_shares(dock, flights):
    for spin, share in zip(dock.dataset_share_spins, (50, 25, 25)):
        spin.setValue(share)
    dock.dataset_seed_spin.setValue(3)

    dock.assign_dataset_splits()

    splits = dock.dataset_splits()
    assert sorted(splits.values()) == ["test", "train", "train", "val"]
    # the same seed deals the same way
    dock.assign_dataset_splits()
    assert dock.dataset_splits() == splits
    assert rows(dock) == [(name, splits[name]) for name, _s in rows(dock)]


def test_unticking_a_flight_takes_it_off_the_table(dock, flights):
    from qgis.PyQt.QtCore import Qt

    # the survey box says so itself (dock._on_survey_changed)
    dock.survey_panel.flights_list.item(0).setCheckState(
        Qt.CheckState.Unchecked)

    assert [name for name, _split in rows(dock)] == [
        "Birch", "Cedar", "Dogwood"]


def test_the_step_is_given_format_folder_and_splits(dock, flights, tmp_path):
    dock.dataset_format_combo.setCurrentIndex(
        dock.dataset_format_combo.findData("yolo"))

    settings = dock.get_config()["dataset_export"]

    assert settings["format"] == "yolo"
    assert settings["output"] == os.path.join(
        str(tmp_path / "survey"), "dataset")
    assert settings["splits"] == dict.fromkeys(NAMES, "train")
    assert settings["include_images"] is True
    assert settings["include_not_an_animal"] is False


def test_a_format_offers_what_it_can_carry(dock):
    combo = dock.dataset_format_combo
    combo.setCurrentIndex(combo.findData("geojson"))
    assert not dock.dataset_images_check.isEnabled()
    assert dock.dataset_false_positives_check.isChecked()

    combo.setCurrentIndex(combo.findData("dwca"))
    assert not dock.dataset_false_positives_check.isEnabled()
    assert dock.dataset_export_settings()["include_not_an_animal"] is None

    assert combo.findData("geojson_segmentation") == -1


def test_the_splits_are_saved_with_the_project(dock, flights):
    combo = dock.dataset_flights_table.cellWidget(2, 1)
    combo.setCurrentIndex(combo.findData("test"))
    dock.dataset_seed_spin.setValue(11)
    dock.dataset_output_edit.setText("/data/set")
    dock._save_dataset_export_settings(QgsProject.instance())

    dock._reset_dataset_export_settings()
    assert dock.dataset_splits()["Cedar"] == "train"
    assert dock.dataset_seed_spin.value() == 0

    dock._load_dataset_export_settings(QgsProject.instance())
    assert dock.dataset_splits()["Cedar"] == "test"
    assert dock.dataset_seed_spin.value() == 11
    assert dock.dataset_output_edit.text() == "/data/set"
    # kept by the flight's folder, so that renaming a flight keeps its split
    assert dock._dataset_splits[fc.normalise(
        flights[2]["target_folder"])] == "test"


def test_export_starts_the_step(dock, flights, warned, started):
    dock.run_dataset_export()

    assert started == ["dataset_export"]
    assert warned == []


def test_nothing_in_a_split_is_said_before_anything_runs(
        dock, flights, warned, started):
    for row in range(dock.dataset_flights_table.rowCount()):
        combo = dock.dataset_flights_table.cellWidget(row, 1)
        combo.setCurrentIndex(combo.findData(dataset_export.LEFT_OUT))

    dock.run_dataset_export()

    assert started == []
    assert "No flight is in a split" in warned[-1]


def test_a_folder_with_another_export_is_refused_up_front(
        dock, flights, tmp_path, warned, started):
    import json

    output = tmp_path / "dataset"
    output.mkdir()
    (output / dataset_export.SPLITS_FILE).write_text(
        json.dumps({"train": [], "val": ["Alder"], "test": []}))
    dock.dataset_output_edit.setText(str(output))

    dock.run_dataset_export()

    assert started == []
    assert "Alder (val -> train)" in warned[-1]
