# -*- coding: utf-8 -*-
"""Survey Analytics > Compare Visits: the group on the tab and the window
that shows the visits (dock/trend.py, bambi_trend_dialog.py).

The comparison itself is covered without QGIS in tests/test_survey_trend.py
and tests/test_trend_step.py.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
from datetime import datetime, timedelta, timezone

import pytest

from qgis.PyQt.QtGui import QImage
from qgis.PyQt.QtWidgets import QMessageBox

from bambi_wildlife_detection.bambi_trend_dialog import (
    TrendChart, TrendDialog, axis_step, changes_text, chart_points,
    overall_text, visit_rows)
from bambi_wildlife_detection.core import survey_trend

MAY = datetime(2024, 5, 1, 5, 0, tzinfo=timezone(timedelta(hours=2)))


def _animals(species, count, group=None, **more):
    return [dict({"species": species, "x": 100.0 + 20 * n, "y": 100.0,
                  "sex": None, "age": None, "approved": False,
                  "group": group}, **more) for n in range(count)]


def _flights(seen=True):
    from shapely.geometry import box

    ground = box(0, 0, 1000, 1000) if seen else None
    return [
        {"name": "May", "start": MAY, "end": MAY + timedelta(minutes=40),
         "seen": ground, "weather": {"air_temperature_c": 9.5},
         "records": [*_animals("roe deer", 4, sex="female"),
                     *_animals("red deer", 3, group=1)]},
        {"name": "June", "start": MAY + timedelta(days=31),
         "end": MAY + timedelta(days=31, minutes=40), "seen": ground,
         "records": _animals("roe deer", 6, age="juvenile")},
        {"name": "Next May", "start": MAY + timedelta(days=366),
         "end": MAY + timedelta(days=366, minutes=40), "seen": ground,
         "records": _animals("roe deer", 40)},
    ]


def _document(seen=True, **more):
    document = survey_trend.analyse(_flights(seen), **more)
    document.update({"source": "tracks", "camera": "Thermal",
                     "crs": "EPSG:32633", "notes": "What the figures mean."})
    return document


def _column(table, name):
    headers = [table.horizontalHeaderItem(c).text()
               for c in range(table.columnCount())]
    return [table.item(r, headers.index(name)).text()
            for r in range(table.rowCount())]


@pytest.fixture
def survey_with_trend(dock, tmp_path, monkeypatch):
    import os

    flight = str(tmp_path / "meadow")
    os.makedirs(flight)
    dock.target_folder_edit.setText(flight)
    dock.target_crs_edit.setText("EPSG:32633")
    dock._on_target_folder_changed()
    folder = tmp_path / "survey"
    dock.survey_panel.folder_edit.setText(str(folder))
    out = folder / "analytics_t"
    out.mkdir(parents=True)
    survey_trend.write_json(_document(), str(out / "trend_tracks.json"))
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
        assert dock.trend_btn.text() == "→ Compare Visits"
        assert dock.trend_show_btn.text() == "→ Show Visits and Trend"
        assert dock.trend_gap_spin.value() == 0
        assert dock.trend_gap_spin.suffix() == " day(s) apart"
        assert [dock.trend_ground_combo.itemText(i) for i in range(2)] == [
            "All the ground each visit saw",
            "Only the ground every visit saw"]
        assert dock.trend_status.text() == "⚪"
        assert "survey_trend" in dock._SURVEY_STEPS
        dock.update_status("survey_trend", "🟢 Completed")
        assert "Completed" in dock.trend_status.text()

    def test_the_settings_reach_the_step(self, dock):
        dock.trend_gap_spin.setValue(3)
        dock.trend_ground_combo.setCurrentIndex(1)

        config = dock.get_config()

        assert config["trend_gap_days"] == 3
        assert config["trend_ground"] == "common"
        assert config["trend_source"] == config["census_source"]

    def test_the_settings_are_saved_and_restored_with_the_project(self, dock):
        dock.trend_gap_spin.setValue(14)
        dock.trend_ground_combo.setCurrentIndex(1)
        dock.save_config_to_project()

        dock._reset_trend_settings()
        assert dock._trend_values() == {"trend_gap_days": 0,
                                        "trend_ground": "seen"}
        dock.load_config_from_project()

        assert dock._trend_values() == {"trend_gap_days": 14,
                                        "trend_ground": "common"}

    def test_a_new_project_starts_with_the_same_day(self, dock):
        dock.trend_gap_spin.setValue(9)
        dock.trend_ground_combo.setCurrentIndex(1)

        dock.reset_config_to_defaults()

        assert dock._trend_values() == {"trend_gap_days": 0,
                                        "trend_ground": "seen"}

    def test_settings_that_make_no_sense_are_not_taken(self, dock):
        dock._set_trend_values({"trend_gap_days": "soon",
                                "trend_ground": "everywhere",
                                "something": 1})

        assert dock._trend_values() == {"trend_gap_days": 0,
                                        "trend_ground": "seen"}

    def test_the_button_starts_the_step(self, dock, monkeypatch):
        from bambi_wildlife_detection.core.survey import SurveyFlight

        started = []
        survey = dock.current_survey()
        survey.flights = [SurveyFlight(name=name, target_folder=f"/x/{name}")
                          for name in ("May", "June")]
        monkeypatch.setattr(dock, "_checked_survey", lambda: survey)
        monkeypatch.setattr(dock, "_survey_prerequisites",
                            lambda survey, checks: True)
        monkeypatch.setattr(dock, "start_worker", started.append)

        dock.trend_btn.click()

        assert started == ["survey_trend"]

    def test_one_flight_is_nothing_to_compare(self, dock, monkeypatch):
        from bambi_wildlife_detection.core.survey import SurveyFlight

        started, warned = [], []
        survey = dock.current_survey()
        survey.flights = [SurveyFlight(name="May", target_folder="/x/May")]
        monkeypatch.setattr(dock, "_checked_survey", lambda: survey)
        monkeypatch.setattr(dock, "start_worker", started.append)
        monkeypatch.setattr(QMessageBox, "warning",
                            lambda *a, **k: warned.append(a[1:]))

        dock.run_survey_trend()

        assert started == []
        assert warned[0][0] == "Compare Visits"
        assert "at least two visits" in warned[0][1]

    def test_nothing_starts_without_a_survey(self, dock, monkeypatch):
        started = []
        monkeypatch.setattr(dock, "_checked_survey", lambda: None)
        monkeypatch.setattr(dock, "start_worker", started.append)

        dock.run_survey_trend()

        assert started == []

    def test_when_the_step_has_ended_the_visits_are_shown(
            self, dock, survey_with_trend):
        dock.on_worker_finished("survey_trend", True)

        assert dock._trend_dialog.isVisible()
        assert "Completed" in dock.trend_status.text()
        dock._trend_dialog.close()

    def test_show_reads_the_surveys_result(self, dock, survey_with_trend):
        dock.trend_show_btn.click()

        assert survey_with_trend == []
        assert dock._trend_dialog.isVisible()
        assert dock._trend_dialog.table.rowCount() == 3
        dock._trend_dialog.close()

    def test_without_a_result_there_is_nothing_to_show(self, dock, tmp_path,
                                                       monkeypatch):
        warned = []
        monkeypatch.setattr(QMessageBox, "warning",
                            lambda *a, **k: warned.append(a[1:]))
        dock.survey_panel.folder_edit.setText(str(tmp_path / "empty"))

        dock.show_trend()

        assert warned == [("Missing Data",
                           "The visits have not been compared yet.\n"
                           "Run 'Compare Visits' first.")]

    def test_a_result_that_cannot_be_read_says_so(self, dock, tmp_path,
                                                  survey_with_trend):
        path = dock._trend_result_path()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{ not json")

        dock.show_trend()

        assert survey_with_trend[0][0] == "Compare Visits"
        assert "Could not read" in survey_with_trend[0][1]


# ---------------------------------------------------------------------------
# The table of visits
# ---------------------------------------------------------------------------

class TestRows:
    def test_one_row_per_visit(self):
        headers, rows = visit_rows(_document(), "roe deer")

        assert headers == [
            "Visit", "Date", "Time", "Flights", "Looked at (ha)", "Animals",
            "Density /km² (95 %)", "Groups", "Largest group",
            "Since the visit before", "Sex", "Age", "Air °C"]
        assert [row[:6] for row in rows] == [
            ["1", "2024-05-01", "05:00-05:40", "May", "100.0", "4"],
            ["2", "2024-06-01", "05:00-05:40", "June", "100.0", "6"],
            ["3", "2025-05-02", "05:00-05:40", "Next May", "100.0", "40"]]
        assert rows[0][6] == "4.0 (1.1–10.2)"
        assert [row[9] for row in rows] == [
            "", "+50 % (p = 0.754)", "+567 % (p < 0.001)"]
        assert [row[10] for row in rows] == ["female 4", "", ""]
        assert [row[11] for row in rows] == ["", "juvenile 6", ""]
        assert [row[12] for row in rows] == ["9.5", "–", "–"]

    def test_a_species_that_came_and_went(self):
        _headers, rows = visit_rows(_document(), "red deer")

        assert [row[5] for row in rows] == ["3", "0", "0"]
        assert [row[7] for row in rows] == ["1", "0", "0"]      # groups
        assert [row[8] for row in rows] == ["3", "–", "–"]
        assert rows[1][9].startswith("to none (p = ")
        assert rows[2][9] == "–"                     # nothing on either

    def test_columns_nobody_has_anything_for_are_left_out(self):
        headers, _rows = visit_rows(_document(), "red deer")

        assert "Sex" not in headers and "Age" not in headers

    def test_a_survey_area_adds_the_number_in_it(self):
        class Plot:
            adjusted_m2 = 2e6

            @staticmethod
            def contains(x, y):
                return True

        headers, rows = visit_rows(_document(area=Plot()), "roe deer")

        assert headers[7] == "In the survey area"
        assert rows[0][7] == "8 (2–20)"

    def test_without_the_ground_the_counts_are_listed(self):
        headers, rows = visit_rows(_document(seen=False), "roe deer")

        assert headers[4] == "Animals (95 %)"
        assert "Looked at (ha)" not in headers
        assert rows[0][4] == "4.0 (1.1–10.2)"

    def test_the_changes_in_words(self):
        text = changes_text(_document(), "roe deer").splitlines()

        assert text == [
            "2024-05-01 to 2024-06-01: +50 % (p = 0.754): within what "
            "chance explains.",
            "2024-06-01 to 2025-05-02: +567 % (p < 0.001): an increase.",
            "2024-05-01 to 2025-05-02: +900 % (p < 0.001): an increase."]

    def test_the_first_visit_against_the_last(self):
        assert overall_text(_document(), "roe deer") == (
            "First against last visit - 2024-05-01 to 2025-05-02: +900 % "
            "(p < 0.001): an increase.")
        two = survey_trend.analyse(_flights()[:2])
        assert overall_text(two, "roe deer").startswith(
            "First against last visit - 2024-05-01 to 2024-06-01: +50 %")


# ---------------------------------------------------------------------------
# The chart
# ---------------------------------------------------------------------------

class TestChart:
    @pytest.mark.parametrize("top, step", [
        (89.0, 20.0), (43.2, 10.0), (11.0, 2.5), (7.3, 2.0), (4.4, 1.0),
        (0.9, 0.2), (1240.0, 250.0), (0.0, 1.0)])
    def test_the_marks_of_the_axis_are_round_numbers(self, top, step):
        assert axis_step(top) == pytest.approx(step)
        assert top / axis_step(top) <= 5 + 1e-9

    def test_what_is_drawn(self):
        points = chart_points(_document(), "roe deer")

        assert points["days"] == pytest.approx([0.0, 31.0, 366.0])
        assert points["labels"] == ["2024-05-01", "2024-06-01", "2025-05-02"]
        assert points["values"] == [4.0, 6.0, 40.0]
        assert points["lows"][0] < 4.0 < points["highs"][0]
        assert len(points["fitted"]) == 3
        assert points["unit"] == "animals / km²"

    def test_counts_where_the_ground_is_not_known(self):
        points = chart_points(_document(seen=False), "roe deer")

        assert points["values"] == [4, 6, 40]
        assert points["unit"] == "animals counted"

    def test_no_line_where_no_rate_was_fitted(self):
        # the red deer were all counted on the first visit
        assert chart_points(_document(), "red deer")["fitted"] is None

    def test_the_axis_has_room_for_the_points_and_the_line(self):
        chart = TrendChart()
        chart.set_points(chart_points(_document(), "roe deer"), "roe deer")

        assert chart.top_value() >= 40.0
        assert chart.points()["values"][-1] == 40.0

    def test_an_interval_far_above_the_points_does_not_flatten_them(self):
        chart = TrendChart()
        chart.set_points({"days": [0, 10], "labels": ["a", "b"],
                          "values": [1.0, 2.0], "lows": [0.1, 0.2],
                          "highs": [50.0, 4.0], "fitted": None, "unit": ""})

        assert chart.top_value() == pytest.approx(2.0 * 3.0 * 1.08)

    @pytest.mark.parametrize("points", [
        None,
        {"days": [], "labels": [], "values": [], "lows": [], "highs": [],
         "fitted": None, "unit": ""},
        {"days": [0.0, 5.0], "labels": ["2024-05-01", "2024-05-06"],
         "values": [0.0, 0.0], "lows": [0.0, 0.0], "highs": [3.7, 3.7],
         "fitted": None, "unit": "animals / km²"},
        {"days": [0.0], "labels": ["2024-05-01 to 2024-05-03"],
         "values": [None], "lows": [None], "highs": [None],
         "fitted": None, "unit": ""},
    ])
    def test_it_draws_whatever_it_is_given(self, points):
        chart = TrendChart()
        chart.resize(600, 260)
        if points is not None:
            chart.set_points(points, "roe deer")
        image = QImage(600, 260, QImage.Format.Format_ARGB32)

        chart.render(image)

        assert not image.isNull()

    def test_it_draws_the_visits(self):
        chart = TrendChart()
        chart.resize(640, 280)
        chart.set_points(chart_points(_document(), "roe deer"), "roe deer")
        image = QImage(640, 280, QImage.Format.Format_ARGB32)
        image.fill(0)

        chart.render(image)

        colours = {image.pixelColor(x, y).name()
                   for x in range(0, 640, 2) for y in range(0, 280, 2)}
        assert "#1f77b4" in colours             # the points
        assert len(colours) > 4


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------

class TestDialog:
    def test_it_opens_on_all_animals(self, dock):
        dialog = TrendDialog(_document(), json_path="/x/trend_tracks.json",
                             parent=dock)

        assert dialog.windowTitle() == "Compare Visits (Thermal, tracks)"
        assert [dialog.species_combo.itemText(i)
                for i in range(dialog.species_combo.count())] == [
            "All animals", "roe deer", "red deer"]
        assert dialog.species() == survey_trend.ALL
        assert _column(dialog.table, "Animals") == ["7", "6", "40"]
        assert dialog.trend_label.text().startswith("All animals: +")
        assert dialog.trend_label.font().bold()
        assert dialog.chart.points()["values"] == [7.0, 6.0, 40.0]

    def test_choosing_a_species_shows_that_species(self, dock):
        dialog = TrendDialog(_document(), parent=dock)

        dialog.species_combo.setCurrentIndex(
            dialog.species_combo.findText("red deer"))

        assert dialog.species() == "red deer"
        assert _column(dialog.table, "Animals") == ["3", "0", "0"]
        assert dialog.trend_label.text() == (
            "red deer: All the animals were counted on the first visit. A "
            "constant rate of change cannot be fitted to a series that "
            "ends with nothing; see the change from visit to visit.")
        assert dialog.chart.points()["fitted"] is None
        # visit against visit is in the table; below it the first against
        # the last, and every pair when the mouse rests on it
        assert dialog.changes_label.text().startswith(
            "First against last visit - 2024-05-01 to 2025-05-02: from "
            "some to none")
        assert dialog.changes_label.toolTip().splitlines()[0].startswith(
            "2024-05-01 to 2024-06-01: from some to none")

    def test_the_header_says_what_was_compared(self, dock):
        from qgis.PyQt.QtWidgets import QLabel

        dialog = TrendDialog(_document(gap_days=2), parent=dock)
        header = dialog.findChildren(QLabel)[0].text()

        assert header == ("3 visits · over 12 months · ground compared: all "
                          "the ground each visit saw · one visit = flights "
                          "up to 2 day(s) apart")

    def test_the_table_can_be_copied(self, dock):
        from qgis.PyQt.QtWidgets import QApplication

        dialog = TrendDialog(_document(), parent=dock)
        dialog.species_combo.setCurrentIndex(1)

        dialog.copy_table()

        copied = QApplication.clipboard().text().splitlines()
        assert copied[0].split("\t")[:2] == ["Visit", "Date"]
        assert copied[1].split("\t")[:2] == ["1", "2024-05-01"]
        assert len(copied) == 4

    def test_a_result_of_an_older_shape_does_not_break_it(self, dock):
        document = _document()
        for key in ("level", "common_ha", "plot_ha", "gap_days", "notes"):
            document.pop(key, None)
        for visit in document["visits"]:
            visit.pop("weather", None)

        dialog = TrendDialog(document, parent=dock)

        assert dialog.table.rowCount() == 3
