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
    dock.browse_and_add_flight()
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


# ---------------------------------------------------------------------------
# Survey orthomosaic (9.0.0)
# ---------------------------------------------------------------------------

def test_the_survey_orthomosaic_controls_exist(dock):
    assert dock.survey_ortho_btn is not None
    assert dock.add_survey_ortho_btn is not None
    dock.update_status("survey_orthomosaic", "🟢 Completed")
    assert "Completed" in dock.survey_ortho_status.text()
    assert "survey_orthomosaic" in dock._SURVEY_STEPS


def test_adding_the_survey_orthomosaic_needs_it_generated(dock, monkeypatch):
    from qgis.PyQt.QtWidgets import QMessageBox
    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: warned.append(a[2]))
    dock.add_survey_orthomosaic_to_qgis()
    assert any("Generate Survey Orthomosaic" in text for text in warned)


# ---------------------------------------------------------------------------
# What the survey counts: one setting for every tool
# ---------------------------------------------------------------------------

def test_the_survey_counts_animals_by_default(dock, two_flights):
    assert dock.survey_panel.source() == "tracks"
    assert dock.survey_panel.source_combo.currentText() == "Animals (tracks)"
    config = dock.get_config()
    assert (config["census_source"], config["density_source"],
            config["ds_source"]) == ("tracks", "tracks", "tracks")
    assert dock._survey_config()["source"] == "tracks"


def test_one_selector_decides_for_every_tool(dock, two_flights):
    """The census used to count animals while the heat map and distance
    sampling counted boxes, each with a selector of its own."""
    panel = dock.survey_panel
    panel.source_combo.setCurrentIndex(panel.source_combo.findData("detections"))

    config = dock.get_config()
    assert (config["census_source"], config["density_source"],
            config["ds_source"]) == ("detections",) * 3
    assert dock._survey_config()["source"] == "detections"


def test_the_separate_selectors_are_gone(dock):
    for name in ("census_source_combo", "density_source_combo",
                 "ds_source_combo"):
        assert not hasattr(dock, name), name


def test_what_is_counted_is_saved_with_the_project(dock, two_flights):
    panel = dock.survey_panel
    panel.source_combo.setCurrentIndex(panel.source_combo.findData("detections"))
    dock.save_config_to_project()

    panel.reset()
    assert panel.source() == "tracks"

    panel.load(QgsProject.instance())
    assert panel.source() == "detections"


def test_a_project_saved_before_the_setting_counts_animals(dock, two_flights):
    from bambi_wildlife_detection.bambi_survey_panel import ENTRY_SOURCE
    from bambi_wildlife_detection.gui_utils import PLUGIN_SCOPE

    dock.save_config_to_project()
    QgsProject.instance().removeEntry(PLUGIN_SCOPE, ENTRY_SOURCE)
    dock.survey_panel.load(QgsProject.instance())
    assert dock.survey_panel.source() == "tracks"


# ---------------------------------------------------------------------------
# ... and what each tool therefore needs
# ---------------------------------------------------------------------------

def _count_boxes(dock):
    panel = dock.survey_panel
    panel.source_combo.setCurrentIndex(panel.source_combo.findData("detections"))


def _warnings(quiet):
    return " ".join(str(argument) for call in quiet["warn"]
                    for argument in call)


def test_the_population_estimate_asks_for_the_distances_it_reads(
        dock, quiet, two_flights):
    dock.run_population_estimation()
    assert "Calculate Track Perpendicular" in _warnings(quiet)

    quiet["warn"].clear()
    _count_boxes(dock)
    dock.run_population_estimation()
    text = _warnings(quiet)
    assert "'Calculate Perpendicular'" in text
    assert "Track Perpendicular" not in text


def test_a_flight_without_animals_needs_no_distances(
        dock, quiet, two_flights):
    """It has none and can have none - and counts with zero animals. The
    step pools such a flight; the check before it used to refuse it."""
    from bambi_wildlife_detection.core import detection_store

    a, _b = two_flights
    detection_store.record_detections(a, "t", [])
    dock.run_population_estimation()
    text = _warnings(quiet)
    assert "transect definitions" in text          # still needed
    assert "perpendicular" not in text.lower()


def test_habitat_use_asks_for_what_the_survey_counts(dock, quiet,
                                                     two_flights):
    dock.run_habitat_summary()
    assert "Track Animals" in _warnings(quiet)

    quiet["warn"].clear()
    _count_boxes(dock)
    dock.run_habitat_summary()
    text = _warnings(quiet)
    assert "Geo-Reference Detections" in text
    assert "Track Animals" not in text


# ---------------------------------------------------------------------------
# One survey, one folder
# ---------------------------------------------------------------------------

def _write_a_result(folder):
    os.makedirs(os.path.join(folder, "analytics_t"), exist_ok=True)
    with open(os.path.join(folder, "analytics_t",
                           "census_summary_tracks.csv"), "w") as handle:
        handle.write("species,count\n")


def _save_project(tmp_path, name="projects"):
    home = str(tmp_path / name)
    os.makedirs(home, exist_ok=True)
    assert QgsProject.instance().write(os.path.join(home, "survey.qgz"))
    return home


def test_saving_the_project_does_not_move_a_survey_with_results(
        dock, two_flights, tmp_path):
    """Until the QGIS project is saved the default is next to the first
    flight; saved, it is next to the project. The results written before
    stayed behind and the tools went on in an empty folder."""
    panel = dock.survey_panel
    before = panel.folder()
    assert before == os.path.join(str(tmp_path / "meadow"), "bambi_survey")
    _write_a_result(before)

    _save_project(tmp_path)

    assert panel.folder() == before
    assert panel.folder_edit.text() == before
    stored, _ok = QgsProject.instance().readEntry(
        "BambiWildlifeDetection", "Survey/Folder", "")
    assert stored == before
    assert "Survey folder kept" in dock.log_text.toPlainText()


def test_a_result_written_just_before_saving_is_enough(dock, two_flights,
                                                       tmp_path):
    """The folder is empty when a tool asks for it and full when it has
    finished - nothing asks again before the project is saved."""
    panel = dock.survey_panel
    before = dock.current_survey().folder      # what a tool is given
    _write_a_result(before)

    _save_project(tmp_path)

    assert dock.current_survey().folder == before


def test_a_survey_without_results_follows_the_project(dock, two_flights,
                                                      tmp_path):
    panel = dock.survey_panel
    assert panel.folder().startswith(str(tmp_path / "meadow"))

    home = _save_project(tmp_path)

    assert panel.folder() == os.path.join(home, "bambi_survey")
    assert panel.folder_edit.text() == ""


def test_ticking_another_flight_first_does_not_move_the_survey(
        dock, two_flights, tmp_path):
    panel = dock.survey_panel
    before = panel.folder()
    _write_a_result(before)

    panel._tick_all(True)
    panel.flights_list.item(0).setCheckState(0)      # only "Forest" is left

    assert [f["name"] for f in panel.selected_flights()] == ["Forest"]
    assert panel.folder() == before


def test_saving_the_project_elsewhere_keeps_the_survey_folder(
        dock, two_flights, tmp_path):
    panel = dock.survey_panel
    first_home = _save_project(tmp_path, "first")
    survey_folder = panel.folder()
    assert survey_folder == os.path.join(first_home, "bambi_survey")
    _write_a_result(survey_folder)

    _save_project(tmp_path, "second")

    assert panel.folder() == survey_folder


def test_emptying_the_field_goes_back_to_the_default(dock, two_flights,
                                                     tmp_path):
    """The folder was kept for the user, not against them."""
    panel = dock.survey_panel
    before = panel.folder()
    _write_a_result(before)
    home = _save_project(tmp_path)
    assert panel.folder() == before

    panel.folder_edit.setText("")
    panel._on_folder_edited()

    assert panel.folder() == os.path.join(home, "bambi_survey")
    assert panel.folder_edit.text() == ""


def test_a_new_project_does_not_inherit_the_kept_folder(dock, two_flights,
                                                        tmp_path):
    panel = dock.survey_panel
    _write_a_result(panel.folder())

    QgsProject.instance().clear()

    assert panel.folder() == ""
    assert panel.folder_edit.text() == ""


# ---------------------------------------------------------------------------
# Only what both cameras saw
# ---------------------------------------------------------------------------

def test_every_track_counts_by_default(dock, two_flights):
    assert dock.survey_panel.confirmed() is False
    assert dock._survey_config()["confirmed"] is False


def test_the_filter_reaches_every_tool(dock, two_flights):
    dock.survey_panel.confirmed_check.setChecked(True)

    assert dock._survey_config()["confirmed"] is True


def test_the_filter_is_saved_with_the_project(dock, two_flights):
    panel = dock.survey_panel
    panel.confirmed_check.setChecked(True)
    dock.save_config_to_project()

    panel.reset()
    assert panel.confirmed() is False

    panel.load(QgsProject.instance())
    assert panel.confirmed() is True


def test_a_project_saved_before_the_filter_counts_every_track(
        dock, two_flights):
    from bambi_wildlife_detection.bambi_survey_panel import ENTRY_CONFIRMED
    from bambi_wildlife_detection.gui_utils import PLUGIN_SCOPE

    dock.survey_panel.confirmed_check.setChecked(True)
    dock.save_config_to_project()
    QgsProject.instance().removeEntry(PLUGIN_SCOPE, ENTRY_CONFIRMED)
    dock.survey_panel.load(QgsProject.instance())
    assert dock.survey_panel.confirmed() is False
