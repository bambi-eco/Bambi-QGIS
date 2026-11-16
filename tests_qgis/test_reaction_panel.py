# -*- coding: utf-8 -*-
"""Survey Analytics > Reaction to the Drone: the group on the tab, the table
of what the animals did, and their movements on the map (dock/reaction.py,
bambi_reaction_dialog.py).

The reading itself is covered without QGIS in tests/test_drone_reaction.py
and tests/test_reaction_step.py.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from qgis.PyQt.QtWidgets import QMessageBox
from qgis.core import QgsProject

from bambi_wildlife_detection.bambi_reaction_dialog import (
    HEADERS, ReactionDialog, reaction_rows, sentences)
from bambi_wildlife_detection.core import drone_reaction as dr


def _track(track_id, status=dr.STAYED, direction=None, species="roe deer",
           away=0.0, flight="Meadow", line=20.0, **more):
    x = 500000.0 + 100.0 * track_id
    moved = status == dr.MOVED
    covered = (away or 0.0) * 19.0
    return dict({
        "flight": flight, "track_id": track_id, "species": species,
        "sex": None, "age": None, "status": status, "direction": direction,
        "n_positions": 20, "duration_s": 19.0,
        "displacement_m": 11.4 if moved else 0.4,
        "displacement_se_m": 0.3, "speed_ms": 0.6 if moved else 0.02,
        "speed_se_ms": 0.02, "heading_deg": 0.0 if moved else None,
        "position_noise_m": 0.3, "away_m": covered, "along_m": 0.0,
        "away_ms": away, "line_distance_m": line, "drone_distance_m": line,
        "start_x": x, "start_y": 5300000.0 + line,
        "end_x": x, "end_y": 5300000.0 + line + covered}, **more)


def _document(by_flight=False):
    tracks = [_track(n, dr.MOVED, dr.AWAY, away=0.6, line=10.0 + n)
              for n in range(1, 8)]
    tracks.append(_track(8, dr.MOVED, dr.TOWARDS, away=-0.5))
    tracks += [_track(n, line=40.0 + n) for n in range(9, 15)]
    tracks += [_track(n, species="red deer", flight="Forest", line=30.0 + n)
               for n in range(15, 22)]
    tracks.append(_track(22, dr.NOT_JUDGED, species="red deer",
                         flight="Forest", away=None, away_m=None,
                         away_ms=None, start_x=None, end_x=None))
    document = {
        "source": "tracks", "camera": "Thermal", "crs": "EPSG:32633",
        "flights": ["Meadow", "Forest"],
        "settings": {"min_displacement_m": 3.0, "min_duration_s": 2.0,
                     "min_positions": 4, "sigmas": 3.0},
        "summary": dr.summarise(tracks, ["roe deer", "red deer"]),
        "by_flight": {}, "tracks": tracks,
        "notes": "What the figures mean."}
    if by_flight:
        for name in document["flights"]:
            document["by_flight"][name] = dr.summarise(
                [t for t in tracks if t["flight"] == name])[dr.ALL]
    return document


def _movements():
    """The layer of movements in the project."""
    found = [layer for layer in QgsProject.instance().mapLayers().values()
             if layer.customProperty("bambi_layer_type") == "drone_reaction"]
    assert len(found) == 1
    return found[0]


@pytest.fixture
def survey_with_reaction(dock, tmp_path, monkeypatch):
    flight = str(tmp_path / "meadow")
    os.makedirs(flight)
    dock.target_folder_edit.setText(flight)
    dock.target_crs_edit.setText("EPSG:32633")
    dock._on_target_folder_changed()
    folder = tmp_path / "survey"
    dock.survey_panel.folder_edit.setText(str(folder))
    out = folder / "analytics_t"
    out.mkdir(parents=True)
    dr.write_json(_document(), str(out / "drone_reaction.json"))
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
        assert dock.reaction_btn.text() == "→ Read Reaction to the Drone"
        assert dock.reaction_show_btn.text() == "→ Show Reaction"
        assert dock.add_reaction_btn.text() == "→ Add Movements to QGIS"
        assert dock.reaction_min_spin.value() == 3.0
        assert dock.reaction_min_spin.suffix() == " m"
        assert dock.reaction_status.text() == "⚪"
        assert "drone_reaction" in dock._SURVEY_STEPS
        dock.update_status("drone_reaction", "🟢 Completed")
        dock.update_status("add_reaction", "🟢 Added")
        assert "Completed" in dock.reaction_status.text()
        assert "Added" in dock.add_reaction_status.text()

    def test_the_setting_reaches_the_step(self, dock):
        dock.reaction_min_spin.setValue(7.5)

        assert dock.get_config()["reaction_min_displacement_m"] == 7.5

    def test_the_speed_is_a_threshold_too(self, dock):
        assert dock.reaction_speed_spin.value() == 0.0
        assert dock.reaction_speed_spin.text() == "any speed"
        assert dock.get_config()["reaction_min_speed_ms"] == 0.0

        dock.reaction_speed_spin.setValue(0.25)

        assert dock.reaction_speed_spin.text() == "0.25 m/s or faster"
        assert dock.get_config()["reaction_min_speed_ms"] == 0.25

    def test_a_species_can_have_thresholds_of_its_own(self, dock,
                                                      monkeypatch):
        from qgis.PyQt.QtWidgets import QDialog

        from bambi_wildlife_detection.dock import reaction

        assert dock.reaction_species_btn.text() == "Per Species…"
        assert dock.reaction_species_label.text() == (
            "every species by the two values above")
        assert dock.get_config()["reaction_species_thresholds"] == {}
        monkeypatch.setattr(dock, "_reaction_species_names",
                            lambda: ["roe deer", "red deer"])
        shown = []

        def answer(dialog):
            shown.append([dialog.table.item(row, 0).text()
                          for row in range(dialog.table.rowCount())])
            dialog.table.item(0, 1).setText("5")
            dialog.table.item(0, 2).setText("0,3")
            dialog.table.item(1, 2).setText("1")
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(reaction.SpeciesThresholdsDialog, "exec", answer)

        dock.reaction_species_btn.click()

        assert shown == [["roe deer", "red deer"]]
        assert dock.get_config()["reaction_species_thresholds"] == {
            "roe deer": {"min_displacement_m": 5.0, "min_speed_ms": 0.3},
            "red deer": {"min_speed_ms": 1.0}}
        assert dock.reaction_species_label.text() == (
            "red deer: at least 3 m at 1 m/s or faster, roe deer: at least "
            "5 m at 0.3 m/s or faster")

    def test_the_table_shows_what_a_species_has_and_refuses_nonsense(self):
        from bambi_wildlife_detection.dock import reaction

        dialog = reaction.SpeciesThresholdsDialog(
            ["roe deer", "fox"], {"roe deer": {"min_displacement_m": 5.0}},
            3.0, 0.0)

        assert dialog.table.item(0, 1).text() == "5"
        assert dialog.table.item(0, 2).text() == ""
        assert dialog.thresholds() == {
            "roe deer": {"min_displacement_m": 5.0}}

        dialog.table.item(1, 2).setText("fast")
        dialog.accept()

        assert dialog.result() != dialog.DialogCode.Accepted
        assert dialog.problem_label.text() == (
            "The speed for 'fox' is not a number ('fast').")
        dialog.deleteLater()

    def test_both_thresholds_are_kept_with_the_project(self, dock):
        """Neither was: a project opened again asked for 3 m at any
        speed, whatever its results had been read with."""
        dock.reaction_min_spin.setValue(7.5)
        dock.reaction_speed_spin.setValue(0.25)
        dock._reaction_species = {"hare": {"min_speed_ms": 2.0}}
        dock.save_config_to_project()
        dock.reaction_min_spin.setValue(3.0)
        dock.reaction_speed_spin.setValue(0.0)
        dock._reaction_species = {}

        dock.load_config_from_project()

        assert dock.reaction_min_spin.value() == 7.5
        assert dock.reaction_speed_spin.value() == 0.25
        assert dock._reaction_species == {"hare": {"min_speed_ms": 2.0}}
        assert dock.reaction_species_label.text().startswith("hare: ")

        dock._reset_reaction_settings()
        assert dock._reaction_species == {}
        assert dock.reaction_min_spin.value() == 3.0
        assert dock.reaction_speed_spin.value() == 0.0

    def test_the_button_starts_the_step(self, dock, monkeypatch):
        started = []
        survey = dock.current_survey()
        monkeypatch.setattr(dock, "_checked_survey", lambda: survey)
        monkeypatch.setattr(dock, "_survey_prerequisites",
                            lambda survey, checks: True)
        monkeypatch.setattr(dock, "start_worker", started.append)

        dock.reaction_btn.click()

        assert started == ["drone_reaction"]

    def test_it_needs_the_tracks_of_every_flight(self, dock, monkeypatch):
        asked = []
        survey = dock.current_survey()
        monkeypatch.setattr(dock, "_checked_survey", lambda: survey)
        monkeypatch.setattr(dock, "_tracks_check", lambda s: "tracks")
        monkeypatch.setattr(
            dock, "_survey_prerequisites",
            lambda survey, checks: asked.append(checks) or False)
        monkeypatch.setattr(dock, "start_worker", asked.append)

        dock.run_drone_reaction()

        assert asked == ["tracks"]            # asked for, and not started

    def test_nothing_starts_without_a_survey(self, dock, monkeypatch):
        started = []
        monkeypatch.setattr(dock, "_checked_survey", lambda: None)
        monkeypatch.setattr(dock, "start_worker", started.append)

        dock.run_drone_reaction()

        assert started == []

    def test_a_running_step_takes_the_buttons(self, dock):
        dock.set_buttons_enabled(False)
        assert not dock.reaction_btn.isEnabled()
        assert not dock.trend_btn.isEnabled()
        assert not dock.grid_btn.isEnabled()
        dock.set_buttons_enabled(True)
        assert dock.reaction_btn.isEnabled()

    def test_when_the_step_has_ended_the_table_is_shown(
            self, dock, survey_with_reaction):
        dock.on_worker_finished("drone_reaction", True)

        assert dock._reaction_dialog.isVisible()
        assert "Completed" in dock.reaction_status.text()
        dock._reaction_dialog.close()

    def test_without_a_result_there_is_nothing_to_show(self, dock, tmp_path,
                                                       monkeypatch):
        warned = []
        monkeypatch.setattr(QMessageBox, "warning",
                            lambda *a, **k: warned.append(a[1:]))
        dock.survey_panel.folder_edit.setText(str(tmp_path / "empty"))

        dock.show_reaction()
        dock.add_reaction_to_qgis()

        assert warned == [("Missing Data",
                           "The reaction to the drone has not been read "
                           "yet.\nRun 'Read Reaction to the Drone' "
                           "first.")] * 2

    def test_a_result_that_cannot_be_read_says_so(self, dock,
                                                  survey_with_reaction):
        with open(dock._reaction_result_path(), "w",
                  encoding="utf-8") as handle:
            handle.write("{ not json")

        dock.show_reaction()

        assert survey_with_reaction[0][0] == "Reaction to the Drone"
        assert "Could not read" in survey_with_reaction[0][1]


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

class TestTable:
    def test_a_row_for_all_animals_and_one_per_species(self):
        headers, rows = reaction_rows(_document())

        assert headers == HEADERS
        assert [row[0] for row in rows] == ["All animals", "roe deer",
                                            "red deer"]
        everything, roe, red = rows
        assert everything[1:3] == ["22", "21"]
        assert everything[3].startswith("8 (38 %, ")
        assert (everything[4], everything[5]) == ("11.4", "0.60")
        assert everything[6:9] == ["7", "1", "0"]
        assert roe[9].startswith("+0.26 (+")           # (7*0.6 - 0.5) / 14
        assert roe[11] == dr.FLED
        assert red[3].startswith("0 (0 %, ")
        assert (red[4], red[11]) == ("–", dr.QUIET)
        assert roe[10].startswith("-0.") and "(p " in roe[10]

    def test_a_survey_of_several_flights_has_a_row_per_flight(self):
        _headers, rows = reaction_rows(_document(by_flight=True))

        assert [row[0] for row in rows][3:] == ["Flight Meadow",
                                                "Flight Forest"]
        assert rows[3][3].startswith("8 (57 %, ")

    def test_in_words(self):
        lines = sentences(_document()).splitlines()

        assert len(lines) == 3
        assert lines[0].startswith(
            "All animals: 8 of 21 animals moved while in view; 7 went away "
            "from the line and 1 towards it")
        assert lines[2].startswith("red deer: 0 of 7 animals moved")

    def test_the_dialog_shows_them(self, dock):
        dialog = ReactionDialog(_document(), json_path="/x/reaction.json",
                                parent=dock)

        assert dialog.windowTitle() == "Reaction to the Drone (Thermal)"
        assert dialog.table.rowCount() == 3
        assert dialog.table.columnCount() == len(HEADERS)
        assert dialog.table.item(0, 0).font().bold()
        assert not dialog.table.item(1, 0).font().bold()
        assert "a reaction to it" in dialog.sentences_label.text()
        dialog.copy_table()
        from qgis.PyQt.QtWidgets import QApplication
        copied = QApplication.clipboard().text().splitlines()
        assert copied[0].split("\t")[1] == "Animals"
        assert copied[2].split("\t")[0] == "roe deer"

    def test_a_result_of_an_older_shape_does_not_break_it(self, dock):
        document = _document()
        for key in ("settings", "by_flight", "notes", "flights"):
            document.pop(key)

        assert ReactionDialog(document, parent=dock).table.rowCount() == 3


# ---------------------------------------------------------------------------
# The movements on the map
# ---------------------------------------------------------------------------

class TestLayer:
    def test_an_arrow_per_animal_that_moved(self, dock,
                                            survey_with_reaction):
        dock.add_reaction_btn.click()

        assert survey_with_reaction == []
        layer = _movements()
        assert layer.name() == "BAMBI Movements (Thermal)"
        assert layer.featureCount() == 8
        assert [field.name() for field in layer.fields()][-len(
            dr.TRACK_COLUMNS):] == dr.TRACK_COLUMNS
        features = {f["track_id"]: f for f in layer.getFeatures()}
        first = features[1]
        line = first.geometry().asPolyline()
        assert (line[0].x(), line[0].y()) == (500100.0, 5300011.0)
        assert line[1].y() == pytest.approx(5300011.0 + 0.6 * 19)
        assert (first["direction"], first["species"]) == (dr.AWAY,
                                                          "roe deer")
        assert first["speed_ms"] == pytest.approx(0.6)
        assert features[8]["direction"] == dr.TOWARDS
        assert "Added" in dock.add_reaction_status.text()
        assert "Added 8 movement(s) of 21 animal(s)" in \
            dock.log_text.toPlainText()

    def test_the_arrows_are_coloured_by_where_the_animal_went(
            self, dock, survey_with_reaction):
        dock.add_reaction_to_qgis()

        layer = _movements()
        renderer = layer.renderer()
        assert renderer.type() == "categorizedSymbol"
        assert renderer.classAttribute() == "direction"
        categories = {c.value(): c for c in renderer.categories()}
        assert set(categories) == {dr.AWAY, dr.TOWARDS, dr.AHEAD, dr.BACK, ""}
        assert categories[""].label() == "line not known"
        away = categories[dr.AWAY].symbol().symbolLayer(0)
        assert away.layerType() == "ArrowLine"
        assert away.subSymbol().color().name() == "#d7301f"
        assert categories[dr.TOWARDS].symbol().symbolLayer(
            0).subSymbol().color().name() == "#2b8cbe"

    def test_an_animal_whose_line_is_not_known_is_drawn_too(
            self, dock, survey_with_reaction):
        path = dock._reaction_result_path()
        document = dr.read_json(path)
        document["tracks"][0]["direction"] = None
        dr.write_json(document, path)

        dock.add_reaction_to_qgis()

        layer = _movements()
        directions = sorted(f["direction"] for f in layer.getFeatures())
        assert directions[0] == "" and len(directions) == 8
