# -*- coding: utf-8 -*-
"""The Survey box on the Survey Analytics tab (core.survey).

The survey tools are not tied to the flight selected for processing: the
user ticks the flights of the survey. What needs a real QGIS is the wiring -
that the list follows the project's flights, that the active flight is the
default, that each flight brings its own CRS and DEM, that the selection is
saved with the project, and that the census table opens.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import json
import os

import pytest

from qgis.core import QgsProject
from qgis.PyQt.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from bambi_wildlife_detection.core import flights


@pytest.fixture
def quiet(monkeypatch):
    shown = {"info": [], "warn": []}
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: shown["info"].append(a))
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: shown["warn"].append(a))
    return shown


@pytest.fixture
def two_flights(dock, quiet, monkeypatch, tmp_path):
    a = str(tmp_path / "meadow" / "qgis")
    b = str(tmp_path / "forest" / "qgis")
    os.makedirs(a)
    os.makedirs(b)
    dock.target_folder_edit.setText(a)
    dock.target_crs_edit.setText("EPSG:32633")
    dock._on_target_folder_changed()

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a_, **k: b)
    monkeypatch.setattr(QInputDialog, "getText",
                        lambda *a_, **k: ("Forest", True))
    monkeypatch.setattr(type(dock), "_ask_new_flight_configuration",
                        lambda self: True)
    dock.add_flight()
    dock.save_config_to_project()      # the forest flight's own store
    dock.flight_combo.setCurrentIndex(0)
    return a, b


def _names(panel):
    return [panel.flights_list.item(i).text()
            for i in range(panel.flights_list.count())]


def test_the_box_sits_at_the_top_of_the_tab(dock):
    assert dock.survey_panel is not None
    assert dock.survey_panel.title() == "Survey"


def test_the_list_follows_the_projects_flights(dock, two_flights):
    assert len(_names(dock.survey_panel)) == 2


def test_the_active_flight_is_ticked_by_default(dock, two_flights):
    a, _b = two_flights
    chosen = dock.survey_panel.selected_flights()
    assert [f["target_folder"] for f in chosen] == [a]


def test_switching_flights_does_not_change_the_survey(dock, two_flights):
    """That is the point: the survey is its own selection."""
    a, _b = two_flights
    dock.flight_combo.setCurrentIndex(1)
    chosen = dock.survey_panel.selected_flights()
    assert [f["target_folder"] for f in chosen] == [a]


def test_all_flights_go_into_the_survey(dock, two_flights):
    dock.survey_panel._tick_all(True)
    survey = dock.current_survey()
    assert [f.name for f in survey.flights] == [
        f.get("name") or flights.default_name(f["target_folder"])
        for f in dock._flights()]
    assert all(f.epsg == 32633 for f in survey.flights)
    assert survey.epsg == 32633


def test_the_camera_is_the_surveys(dock, two_flights):
    dock.survey_panel.camera_combo.setCurrentIndex(1)
    assert dock.current_survey().modality == "w"


def test_the_folder_defaults_next_to_the_flights(dock, two_flights):
    folder = dock.current_survey().folder
    assert folder.endswith("bambi_survey")


def test_an_explicit_folder_wins(dock, two_flights, tmp_path):
    dock.survey_panel.folder_edit.setText(str(tmp_path / "census2026"))
    assert dock.current_survey().folder == str(tmp_path / "census2026")


def test_the_selection_is_saved_with_the_project(dock, two_flights,
                                                 tmp_path):
    a, b = two_flights
    dock.survey_panel._tick_all(True)
    dock.survey_panel.camera_combo.setCurrentIndex(1)
    dock.survey_panel.folder_edit.setText(str(tmp_path / "s"))
    dock.survey_panel.save(QgsProject.instance())

    dock.survey_panel.reset()
    assert len(dock.survey_panel.selected_flights()) == 1
    dock.survey_panel.load(QgsProject.instance())
    assert {f["target_folder"] for f in
            dock.survey_panel.selected_flights()} == {a, b}
    assert dock.survey_panel.modality() == "w"
    assert dock.survey_panel.folder() == str(tmp_path / "s")


def test_running_without_a_flight_is_refused(dock, quiet, two_flights):
    dock.survey_panel._tick_all(False)
    dock.run_census_summary()
    assert any("No flight" in str(a) for a in quiet["warn"])


def test_missing_prerequisites_name_the_flight(dock, quiet, two_flights):
    dock.survey_panel._tick_all(True)
    dock.run_track_inventory()
    text = " ".join(str(a) for a in quiet["warn"])
    assert "Forest" in text


def test_the_per_tool_project_pickers_are_gone(dock):
    for name in ("ds_projects_list", "pop_projects_list",
                 "ds_add_flight_btn", "pop_add_flight_btn",
                 "pop_camera_combo", "coverage_camera_combo",
                 "inventory_camera_combo"):
        assert not hasattr(dock, name), name


def test_the_census_table_opens(dock, two_flights, tmp_path):
    from bambi_wildlife_detection.bambi_survey_panel import CensusTableDialog
    from bambi_wildlife_detection.core import census

    result = census.build([
        {"name": "Meadow", "records": [
            {"species": "roe deer", "sex": "female", "age": "adult",
             "x": 1.0, "y": 1.0, "approved": True}], "covered_m2": 1e4},
        {"name": "Forest", "records": [
            {"species": "roe deer", "sex": "", "age": "juvenile",
             "x": 2.0, "y": 2.0, "approved": False}], "covered_m2": 2e4},
    ])
    result.update({"camera": "Thermal", "source": "tracks", "notes": ""})
    dialog = CensusTableDialog(result)
    table = dialog._table
    assert table.rowCount() == len(result["rows"])
    headers = [table.horizontalHeaderItem(c).text()
               for c in range(table.columnCount())]
    assert "Flight" in headers and "/km² (covered)" in headers

    dialog.copy_table()
    from qgis.PyQt.QtWidgets import QApplication
    assert "roe deer" in QApplication.clipboard().text()


def test_show_census_reads_the_survey_folder(dock, two_flights, tmp_path,
                                             monkeypatch):
    from bambi_wildlife_detection.core import census

    folder = tmp_path / "s"
    dock.survey_panel.folder_edit.setText(str(folder))
    out = folder / "analytics_t"
    out.mkdir(parents=True)
    result = census.build([{"name": "Meadow", "records": [],
                            "covered_m2": None}])
    result.update({"camera": "Thermal", "source": "tracks", "notes": ""})
    (out / "census_summary_tracks.json").write_text(json.dumps(result))

    dock.show_census_table()
    assert dock._census_dialog.isVisible()
    dock._census_dialog.close()
