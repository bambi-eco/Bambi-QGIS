# -*- coding: utf-8 -*-
"""Survey Analytics > Position Uncertainty: the group on the tab with its
accuracies, the table, and the circles on the map (dock/position.py,
bambi_position_dialog.py).

The budget itself is covered without QGIS in
tests/test_position_uncertainty.py and tests/test_position_step.py.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from qgis.PyQt.QtWidgets import QMessageBox
from qgis.core import QgsProject

from bambi_wildlife_detection.bambi_position_dialog import (
    HEADERS, PositionDialog, accuracy_text, uncertainty_rows)
from bambi_wildlife_detection.core import position_uncertainty as pu

CAMERAS = {frame: (500000.0 + 5.0 * frame, 5300000.0, 380.0)
           for frame in range(40)}


def _boxes(track, x, y, frames):
    return [{"detection_id": 100 * track + n, "track_id": track,
             "frame": frame, "x": 500000.0 + x, "y": 5300000.0 + y,
             "z": 300.0} for n, frame in enumerate(frames)]


def _document(by_flight=False, source="tracks"):
    boxes = [*_boxes(1, 50.0, 0.0, (8, 9, 10, 11, 12)),
             *_boxes(2, 120.0, 80.0, (22, 23, 24, 25, 26))]
    rows, _left_out = pu.box_rows("Meadow", boxes, CAMERAS, pu.Accuracy())
    animals = [
        {"flight": "Meadow", "track_id": 1, "species": "roe deer",
         "x": 500050.0, "y": 5300000.0},
        {"flight": "Meadow", "track_id": 2, "species": "red deer",
         "x": 500120.0, "y": 5300080.0},
        {"flight": "Meadow", "track_id": 3, "species": "roe deer",
         "x": None, "y": None},
    ]
    from dataclasses import asdict
    document = {
        "source": source, "camera": "Thermal", "crs": "EPSG:32633",
        "flights": ["Meadow"], "accuracy": asdict(pu.Accuracy()),
        "summary": pu.summarise(rows), "by_flight": {},
        "tracks": pu.track_rows(rows, animals) if source == "tracks" else [],
        "left_out": 0, "level_terrain": [],
        "notes": "What the figures mean."}
    if by_flight:
        document["by_flight"] = {"Meadow": pu.summarise(rows[:5]),
                                 "Forest": pu.summarise(rows[5:])}
    return document


def _circles():
    """The layer of circles in the project."""
    found = _layers()
    assert len(found) == 1
    return found[0]


def _layers():
    kind = "position_uncertainty"
    return [layer for layer in QgsProject.instance().mapLayers().values()
            if layer.customProperty("bambi_layer_type") == kind]


@pytest.fixture
def survey_with_positions(dock, tmp_path, monkeypatch):
    flight = str(tmp_path / "meadow")
    os.makedirs(flight)
    dock.target_folder_edit.setText(flight)
    dock.target_crs_edit.setText("EPSG:32633")
    dock._on_target_folder_changed()
    folder = tmp_path / "survey"
    dock.survey_panel.folder_edit.setText(str(folder))
    out = folder / "analytics_t"
    out.mkdir(parents=True)
    pu.write_json(_document(),
                  str(out / "position_uncertainty_tracks.json"))
    shown = []
    monkeypatch.setattr(QMessageBox, "critical",
                        lambda *a, **k: shown.append(a[1:]))
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: shown.append(a[1:]))
    return shown


# ---------------------------------------------------------------------------
# The group on the tab
# ---------------------------------------------------------------------------

class TestGroup:
    def test_the_tab_has_the_group(self, dock):
        assert dock.position_btn.text() == "→ Estimate Position Uncertainty"
        assert dock.position_show_btn.text() == (
            "→ Show Position Uncertainty")
        assert dock.add_position_btn.text() == "→ Add Uncertainty to QGIS"
        assert dock.position_status.text() == "⚪"
        assert "position_uncertainty" in dock._SURVEY_STEPS
        dock.update_status("position_uncertainty", "🟢 Completed")
        dock.update_status("add_position", "🟢 Added")
        assert "Completed" in dock.position_status.text()
        assert "Added" in dock.add_position_status.text()

    def test_it_starts_with_a_drone_without_a_base_station(self, dock):
        assert [dock.position_preset_combo.itemText(i)
                for i in range(dock.position_preset_combo.count())] == [
            "Standard GNSS", "RTK", "Custom"]
        assert dock.position_preset_combo.currentText() == "Standard GNSS"
        assert dock._position_values() == {
            "position_gps_horizontal_m": 1.5, "position_gps_vertical_m": 1.0,
            "position_camera_angle_deg": 1.0, "position_terrain_m": 1.0,
            "position_pixel_px": 2.0, "position_sync_s": 0.1}
        assert dock.position_angle_spin.suffix() == " °"

    def test_a_preset_sets_what_it_is_about(self, dock):
        dock.position_angle_spin.setValue(0.4)       # the user's own gimbal

        dock.position_preset_combo.setCurrentText("RTK")

        values = dock._position_values()
        assert values["position_gps_horizontal_m"] == 0.05
        assert values["position_gps_vertical_m"] == 0.1
        # ... and with it everything a preset says: the angle too
        assert values["position_camera_angle_deg"] == 1.0
        assert dock.position_preset_combo.currentText() == "RTK"

    def test_a_value_of_ones_own_is_custom(self, dock):
        dock.position_angle_spin.setValue(0.4)

        assert dock.position_preset_combo.currentText() == "Custom"
        dock.position_angle_spin.setValue(1.0)
        assert dock.position_preset_combo.currentText() == "Standard GNSS"

    def test_choosing_custom_leaves_the_values(self, dock):
        dock.position_preset_combo.setCurrentText("RTK")

        dock.position_preset_combo.setCurrentText("Custom")

        assert dock._position_values()["position_gps_horizontal_m"] == 0.05

    def test_the_accuracies_reach_the_step(self, dock):
        dock.position_preset_combo.setCurrentText("RTK")
        dock.position_sync_spin.setValue(0.25)

        config = dock.get_config()

        assert config["position_gps_horizontal_m"] == 0.05
        assert config["position_sync_s"] == 0.25
        assert config["position_source"] == config["census_source"]
        assert pu.Accuracy.from_config(config).sync_s == 0.25

    def test_they_are_saved_and_restored_with_the_project(self, dock):
        dock.position_preset_combo.setCurrentText("RTK")
        dock.position_terrain_spin.setValue(2.5)
        dock.save_config_to_project()

        dock._reset_position_settings()
        assert dock.position_preset_combo.currentText() == "Standard GNSS"
        dock.load_config_from_project()

        values = dock._position_values()
        assert values["position_gps_horizontal_m"] == 0.05
        assert values["position_terrain_m"] == 2.5
        assert dock.position_preset_combo.currentText() == "Custom"

    def test_a_new_project_starts_with_the_defaults(self, dock):
        dock.position_preset_combo.setCurrentText("RTK")

        dock.reset_config_to_defaults()

        assert dock.position_preset_combo.currentText() == "Standard GNSS"

    def test_values_that_make_no_sense_are_not_taken(self, dock):
        dock._set_position_values({"position_gps_horizontal_m": "good",
                                   "position_pixel_px": None,
                                   "position_sync_s": 0.3})

        values = dock._position_values()
        assert values["position_gps_horizontal_m"] == 1.5
        assert values["position_pixel_px"] == 2.0
        assert values["position_sync_s"] == 0.3

    def test_the_button_starts_the_step(self, dock, monkeypatch):
        started = []
        survey = dock.current_survey()
        monkeypatch.setattr(dock, "_checked_survey", lambda: survey)
        monkeypatch.setattr(dock, "_survey_prerequisites",
                            lambda survey, checks: True)
        monkeypatch.setattr(dock, "start_worker", started.append)

        dock.position_btn.click()

        assert started == ["position_uncertainty"]

    def test_nothing_starts_without_a_survey(self, dock, monkeypatch):
        started = []
        monkeypatch.setattr(dock, "_checked_survey", lambda: None)
        monkeypatch.setattr(dock, "start_worker", started.append)

        dock.run_position_uncertainty()

        assert started == []

    def test_when_the_step_has_ended_the_table_is_shown(
            self, dock, survey_with_positions):
        dock.on_worker_finished("position_uncertainty", True)

        assert dock._position_dialog.isVisible()
        dock._position_dialog.close()

    def test_without_a_result_there_is_nothing_to_show(self, dock, tmp_path,
                                                       monkeypatch):
        warned = []
        monkeypatch.setattr(QMessageBox, "warning",
                            lambda *a, **k: warned.append(a[1:]))
        dock.survey_panel.folder_edit.setText(str(tmp_path / "empty"))

        dock.show_position_uncertainty()
        dock.add_position_to_qgis()

        assert warned == [("Missing Data",
                           "The position uncertainty has not been worked "
                           "out yet.\nRun 'Estimate Position Uncertainty' "
                           "first.")] * 2


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

class TestTable:
    def test_all_detections_in_a_row(self):
        headers, rows = uncertainty_rows(_document())

        assert headers == HEADERS
        assert len(rows) == 1
        row = rows[0]
        assert row[:2] == ["All detections", "10"]
        assert float(row[2]) < float(row[3]) <= float(row[4])
        assert row[5].endswith(" %") and row[6].endswith(" %")
        assert row[7] == "80"
        assert row[9] in ("camera position (50 %)", "camera angle (50 %)")

    def test_a_survey_of_several_flights_has_a_row_per_flight(self):
        _headers, rows = uncertainty_rows(_document(by_flight=True))

        assert [row[0] for row in rows] == [
            "All detections", "Flight Meadow", "Flight Forest"]
        assert rows[1][9] == "camera position (100 %)"
        assert rows[2][9] == "camera angle (100 %)"
        assert float(rows[2][2]) > float(rows[1][2])

    def test_the_accuracies_in_a_line(self):
        assert accuracy_text(_document()) == (
            "Taken as known to (one sigma): camera position 1.5 m, camera "
            "height 1 m, camera angle 1 °, terrain model 1 m, pixel 2 px, "
            "timing 0.1 s")
        assert accuracy_text({}) == ""

    def test_the_dialog_shows_them(self, dock):
        dialog = PositionDialog(_document(by_flight=True),
                                json_path="/x/u.json", parent=dock)

        assert dialog.windowTitle() == (
            "Position Uncertainty (Thermal, tracks)")
        assert dialog.table.rowCount() == 3
        assert dialog.table.item(0, 0).font().bold()
        assert dialog.sentence_label.text().startswith(
            "10 detections: typically ")
        assert dialog.accuracy_label.text().startswith("Taken as known to")
        dialog.copy_table()
        from qgis.PyQt.QtWidgets import QApplication
        copied = QApplication.clipboard().text().splitlines()
        assert copied[0].split("\t")[1] == "Detections"
        assert copied[1].split("\t")[0] == "All detections"

    def test_a_result_with_nothing_in_it_does_not_break_it(self, dock):
        dialog = PositionDialog({"summary": pu.summarise([])}, parent=dock)

        assert dialog.table.item(0, 2).text() == "–"
        assert dialog.sentence_label.text() == (
            "No detection has a place to judge.")


# ---------------------------------------------------------------------------
# The circles on the map
# ---------------------------------------------------------------------------

class TestLayer:
    def test_a_circle_around_every_animal_that_has_a_place(
            self, dock, survey_with_positions):
        dock.add_position_btn.click()

        assert survey_with_positions == []
        layer = _circles()
        assert layer.name() == "BAMBI Position Uncertainty (Thermal)"
        assert layer.featureCount() == 2
        features = {f["track_id"]: f for f in layer.getFeatures()}
        below, beside = features[1], features[2]
        assert (below["species"], below["boxes"]) == ("roe deer", 5)
        box = below.geometry().boundingBox()
        # as wide as twice what almost never is exceeded
        assert box.width() == pytest.approx(2 * below["error95_m"], rel=0.01)
        assert box.center().x() == pytest.approx(500050.0)
        assert box.center().y() == pytest.approx(5300000.0)
        assert beside["error95_m"] > below["error95_m"]
        assert below.geometry().area() == pytest.approx(
            3.14159 * below["error95_m"] ** 2, rel=0.01)
        assert "Added" in dock.add_position_status.text()
        assert "Added the uncertainty of 2 animal(s)" in \
            dock.log_text.toPlainText()

    def test_the_circles_are_one_plain_style(self, dock,
                                             survey_with_positions):
        dock.add_position_to_qgis()

        renderer = _circles().renderer()
        assert renderer.type() == "singleSymbol"
        assert renderer.symbol().symbolLayer(0).strokeColor().name() == (
            "#e6550d")

    def test_a_survey_that_counts_boxes_has_no_animals_to_circle(
            self, dock, survey_with_positions):
        path = dock._position_result_path()
        pu.write_json(_document(source="detections"), path)

        dock.add_position_to_qgis()

        assert survey_with_positions[0][0] == "Position Uncertainty"
        assert "the survey counts boxes" in survey_with_positions[0][1]
        assert _layers() == []
