# -*- coding: utf-8 -*-
"""Survey Analytics > Survey Report: the group at the end of the tab, the
picture of the map it takes, and the report opened (dock/report.py).

The page and the step are covered without QGIS in
tests/test_survey_report.py and tests/test_report_step.py.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtGui import QColor, QImage, QPixmap
from qgis.PyQt.QtWidgets import QGroupBox, QMessageBox

from bambi_wildlife_detection.core import store
from bambi_wildlife_detection.dock import report as report_dock


class Canvas:
    """A map canvas that shows one colour, *width* x *height* pixels."""

    def __init__(self, width=300, height=200, fails=False):
        self.width, self.height, self.fails = width, height, fails

    def grab(self):
        if self.fails:
            raise RuntimeError("no screen")
        if not self.width:
            return QPixmap()
        picture = QPixmap(self.width, self.height)
        picture.fill(QColor("#336699"))
        return picture


@pytest.fixture
def survey(dock, tmp_path, monkeypatch):
    """The dock on a flight and a survey folder; returns the survey's
    analytics folder (not made yet)."""
    flight = str(tmp_path / "meadow")
    os.makedirs(flight)
    dock.target_folder_edit.setText(flight)
    dock.target_crs_edit.setText("EPSG:32633")
    dock._on_target_folder_changed()
    folder = tmp_path / "survey"
    dock.survey_panel.folder_edit.setText(str(folder))
    return str(folder / "analytics_t")


@pytest.fixture
def started(dock, survey, monkeypatch):
    """Whatever the dock starts, with a survey that is ready for it."""
    steps = []
    checked = dock.current_survey()
    monkeypatch.setattr(dock, "_checked_survey", lambda: checked)
    monkeypatch.setattr(dock, "_survey_prerequisites",
                        lambda survey, checks: True)
    monkeypatch.setattr(dock, "start_worker", steps.append)
    return steps


@pytest.fixture
def opened(monkeypatch):
    """What would have been opened in the web browser."""
    urls = []
    monkeypatch.setattr(
        report_dock.QDesktopServices, "openUrl",
        staticmethod(lambda url: urls.append(url) or True))
    return urls


@pytest.fixture
def told(monkeypatch):
    """The message boxes shown: ``(kind, title, text)``."""
    shown = []
    for kind in ("warning", "information", "critical"):
        monkeypatch.setattr(
            QMessageBox, kind,
            lambda *a, _kind=kind, **k: shown.append((_kind, *a[1:3])))
    return shown


def _write_report(folder, text="<!DOCTYPE html><html></html>"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "survey_report.html")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


# ---------------------------------------------------------------------------
# The group on the tab
# ---------------------------------------------------------------------------

class TestGroup:
    def test_the_tab_has_the_group(self, dock):
        assert dock.report_btn.text() == "→ Write Survey Report"
        assert dock.report_open_btn.text() == "→ Open Report"
        assert dock.report_status.text() == "⚪"
        assert dock.report_title_edit.text() == ""
        assert dock.report_canvas_check.isChecked()
        assert "survey_report" in dock._SURVEY_STEPS
        dock.update_status("survey_report", "🟢 Completed")
        assert "Completed" in dock.report_status.text()

    def test_it_is_the_last_of_the_tab(self, dock):
        """A report is written when everything else has been run."""
        group = dock.report_btn.parentWidget()
        assert isinstance(group, QGroupBox)
        assert group.title() == "Survey Report"
        layout = group.parentWidget().layout()
        widgets = [layout.itemAt(index).widget()
                   for index in range(layout.count())
                   if layout.itemAt(index).widget() is not None]
        assert widgets[-1] is group
        assert "Habitat" in widgets[-2].title()

    def test_the_settings_reach_the_step(self, dock):
        dock.report_title_edit.setText("  Roe deer in the Hintertal  ")

        config = dock.get_config()

        assert config["report_title"] == "Roe deer in the Hintertal"
        assert config["report_source"] == config["census_source"]
        assert config["report_canvas_png"] == ""

    def test_the_title_and_the_tick_are_kept_with_the_project(self, dock):
        """A report is written again whenever the survey has moved on."""
        dock.report_title_edit.setText("Roe deer in the Hintertal, 2024")
        dock.report_canvas_check.setChecked(False)
        dock.save_config_to_project()

        dock._reset_report_settings()
        assert dock.report_title_edit.text() == ""
        assert dock.report_canvas_check.isChecked()
        dock.load_config_from_project()

        assert dock.report_title_edit.text() == (
            "Roe deer in the Hintertal, 2024")
        assert not dock.report_canvas_check.isChecked()

    def test_a_new_project_starts_without_a_title(self, dock):
        dock.report_title_edit.setText("Last year's survey")
        dock.report_canvas_check.setChecked(False)

        dock.reset_config_to_defaults()

        assert dock.report_title_edit.text() == ""
        assert dock.report_canvas_check.isChecked()

    def test_a_project_that_never_wrote_a_report_has_the_defaults(
            self, dock, clean_project):
        dock.report_title_edit.setText("Something typed")

        dock._load_report_settings(clean_project)

        assert dock.report_title_edit.text() == ""
        assert dock.report_canvas_check.isChecked()

    def test_settings_that_make_no_sense_are_not_taken(self, dock,
                                                       clean_project):
        from bambi_wildlife_detection.gui_utils import PLUGIN_SCOPE

        for stored in ("{ not json", "[1, 2]", '{"title": 7, "canvas": "x"}'):
            clean_project.writeEntry(PLUGIN_SCOPE, report_dock.ENTRY_REPORT,
                                     stored)
            dock.report_title_edit.setText("Something typed")

            dock._load_report_settings(clean_project)

            assert dock.report_title_edit.text() == ""
            assert dock.report_canvas_check.isChecked()

    def test_the_button_is_off_while_a_step_runs(self, dock):
        dock.set_buttons_enabled(False)
        assert not dock.report_btn.isEnabled()

        dock.set_buttons_enabled(True)
        assert dock.report_btn.isEnabled()


# ---------------------------------------------------------------------------
# Writing the report
# ---------------------------------------------------------------------------

class TestRun:
    def test_the_button_starts_the_step(self, dock, started):
        dock.report_btn.click()

        assert started == ["survey_report"]

    def test_the_map_is_pictured_for_the_report(self, dock, started, survey,
                                                monkeypatch):
        monkeypatch.setattr(dock.iface, "mapCanvas", lambda: Canvas(300, 200))

        dock.run_survey_report()

        path = os.path.join(survey, "survey_report_map.png")
        assert dock.get_config()["report_canvas_png"] == path
        picture = QImage(path)
        assert (picture.width(), picture.height()) == (300, 200)
        assert picture.pixelColor(10, 10).name() == "#336699"

    def test_a_wide_map_is_made_smaller(self, dock, started, survey,
                                        monkeypatch):
        """A 4k canvas as it is would make a report of many megabytes."""
        monkeypatch.setattr(dock.iface, "mapCanvas",
                            lambda: Canvas(2800, 1000))

        dock.run_survey_report()

        picture = QImage(os.path.join(survey, "survey_report_map.png"))
        assert (picture.width(), picture.height()) == (1400, 500)

    def test_the_real_canvas_can_be_pictured(self, dock, started, survey):
        dock.iface.mapCanvas().resize(320, 240)

        dock.run_survey_report()

        path = dock.get_config()["report_canvas_png"]
        assert path == os.path.join(survey, "survey_report_map.png")
        assert QImage(path).width() == 320

    def test_without_the_tick_there_is_no_picture(self, dock, started,
                                                  survey, monkeypatch):
        monkeypatch.setattr(dock.iface, "mapCanvas", lambda: Canvas())
        dock.report_canvas_check.setChecked(False)

        dock.run_survey_report()

        assert started == ["survey_report"]
        assert dock.get_config()["report_canvas_png"] == ""
        assert not os.path.exists(os.path.join(survey,
                                               "survey_report_map.png"))

    def test_the_picture_of_an_earlier_report_is_not_taken_again(
            self, dock, started, survey, monkeypatch):
        monkeypatch.setattr(dock.iface, "mapCanvas", lambda: Canvas())
        dock.run_survey_report()
        assert dock.get_config()["report_canvas_png"]

        dock.report_canvas_check.setChecked(False)
        dock.run_survey_report()

        assert dock.get_config()["report_canvas_png"] == ""

    def test_a_map_that_cannot_be_pictured_does_not_stop_the_report(
            self, dock, started, monkeypatch):
        monkeypatch.setattr(dock.iface, "mapCanvas",
                            lambda: Canvas(fails=True))

        dock.run_survey_report()

        assert started == ["survey_report"]
        assert dock.get_config()["report_canvas_png"] == ""
        assert ("The map could not be pictured for the report (no screen): "
                "it has the drawn map only") in dock.log_text.toPlainText()

    def test_a_map_of_no_size_is_no_picture(self, dock, started,
                                            monkeypatch):
        monkeypatch.setattr(dock.iface, "mapCanvas", lambda: Canvas(0, 0))

        dock.run_survey_report()

        assert started == ["survey_report"]
        assert dock.get_config()["report_canvas_png"] == ""

    def test_nothing_starts_without_a_survey(self, dock, monkeypatch):
        steps = []
        monkeypatch.setattr(dock, "_checked_survey", lambda: None)
        monkeypatch.setattr(dock, "start_worker", steps.append)

        dock.run_survey_report()

        assert steps == []

    def test_nothing_starts_and_nothing_is_pictured_where_data_is_missing(
            self, dock, survey, monkeypatch, told):
        """The flight of the fixture has no tracks."""
        steps = []
        monkeypatch.setattr(dock.iface, "mapCanvas", lambda: Canvas())
        monkeypatch.setattr(dock, "start_worker", steps.append)

        dock.run_survey_report()

        assert steps == []
        assert not os.path.exists(os.path.join(survey,
                                               "survey_report_map.png"))
        assert len(told) == 1 and told[0][0] == "warning"
        assert "Track Animals" in told[0][2]

    def test_a_report_of_boxes_asks_for_boxes(self, dock, survey,
                                              monkeypatch):
        asked = []
        checked = dock.current_survey()
        monkeypatch.setattr(dock, "_checked_survey", lambda: checked)
        monkeypatch.setattr(dock, "_survey_source", lambda: "detections")
        monkeypatch.setattr(
            dock, "_survey_prerequisites",
            lambda survey, checks: asked.append(checks) or False)
        has = []
        monkeypatch.setattr(
            dock, "has_stage",
            lambda kind, modality, folder="": (kind, modality) in has)

        dock.run_survey_report()

        flight = checked.flights[0]
        assert asked[0](flight) == [
            "Thermal detections (run 'Detect Animals' first)"]
        has.append((store.DETECTIONS, "t"))
        assert asked[0](flight) == []


# ---------------------------------------------------------------------------
# Opening it
# ---------------------------------------------------------------------------

class TestOpen:
    def test_when_the_step_has_ended_the_report_is_opened(
            self, dock, survey, opened):
        path = _write_report(survey)

        dock.on_worker_finished("survey_report", True)

        assert opened == [QUrl.fromLocalFile(path)]
        assert f"Survey report: {path}" in dock.log_text.toPlainText()

    def test_a_step_that_failed_opens_nothing(self, dock, survey, opened,
                                              told):
        _write_report(survey)

        dock.on_worker_finished("survey_report", False)

        assert opened == []

    @pytest.mark.parametrize("ending", ["written", "failed", "error"])
    def test_the_picture_made_for_the_report_does_not_stay_beside_it(
            self, dock, survey, started, opened, told, monkeypatch, ending):
        """It is in the page. Beside it, it looks as if the page needed
        it - and the next step's settings would still name it."""
        monkeypatch.setattr(dock.iface, "mapCanvas", lambda: Canvas())
        dock.run_survey_report()
        picture = os.path.join(survey, "survey_report_map.png")
        assert os.path.isfile(picture)
        _write_report(survey)

        if ending == "error":
            dock.on_worker_error("survey_report", "No tracks.")
        else:
            dock.on_worker_finished("survey_report", ending == "written")

        assert not os.path.exists(picture)
        assert dock.get_config()["report_canvas_png"] == ""
        assert os.path.isfile(os.path.join(survey, "survey_report.html"))

    def test_another_step_leaves_the_picture_alone(self, dock, survey,
                                                   started, monkeypatch):
        monkeypatch.setattr(dock.iface, "mapCanvas", lambda: Canvas())
        monkeypatch.setattr(dock, "show_census_table", lambda: None)
        dock.run_survey_report()

        dock.on_worker_finished("census_summary", True)

        assert os.path.isfile(os.path.join(survey, "survey_report_map.png"))

    def test_the_report_written_last_can_be_opened_again(
            self, dock, survey, opened):
        path = _write_report(survey)

        dock.report_open_btn.click()

        assert opened == [QUrl.fromLocalFile(path)]

    def test_without_a_report_there_is_nothing_to_open(self, dock, survey,
                                                       opened, told):
        dock.open_survey_report()

        assert opened == []
        assert told == [("warning", "Missing Data",
                         "The survey report has not been written yet.\n"
                         "Run 'Write Survey Report' first.")]

    def test_where_nothing_opens_it_the_user_is_told_where_it_is(
            self, dock, survey, told, monkeypatch):
        path = _write_report(survey)
        monkeypatch.setattr(report_dock.QDesktopServices, "openUrl",
                            staticmethod(lambda url: False))

        dock.open_survey_report()

        assert told == [("information", "Survey Report",
                         f"The report is at\n{path}\n\nOpen it in a web "
                         "browser.")]

    def test_each_camera_has_a_report_of_its_own(self, dock, survey, opened,
                                                 told):
        """The thermal survey's report is not the RGB survey's."""
        _write_report(survey)
        assert dock._report_path() == os.path.join(survey,
                                                   "survey_report.html")

        dock.survey_panel.camera_combo.setCurrentIndex(1)

        assert dock._report_path() == os.path.join(
            os.path.dirname(survey), "analytics_w", "survey_report.html")
        dock.open_survey_report()
        assert opened == [] and told[0][1] == "Missing Data"


# ---------------------------------------------------------------------------
# From the button to the page
# ---------------------------------------------------------------------------

class TestWholeWay:
    """The step is run with what the dock hands it - as the worker would,
    in this thread."""

    @pytest.fixture
    def meadow(self, dock, tmp_path, monkeypatch):
        from bambi_wildlife_detection.bambi_processing import BambiProcessor
        from tests.survey_flights import make_flight

        folder = make_flight(str(tmp_path / "Meadow"), [
            {"id": 1, "species": "roe deer", "at": (20, 40),
             "frames": (1, 5)},
            {"id": 2, "species": "red deer", "at": (80, 40),
             "frames": (2, 6)},
        ], seen=[(0, 0, 200, 100)])
        dock.target_folder_edit.setText(folder)
        dock.target_crs_edit.setText("EPSG:32633")
        dock._on_target_folder_changed()
        dock.survey_panel.folder_edit.setText(str(tmp_path / "survey"))

        def run_now(step):
            config = dock.get_config()
            config["survey"] = dock._survey_config()
            try:
                getattr(BambiProcessor(), f"run_{step}")(
                    config, log_fn=dock.log)
            except Exception as exc:  # noqa: BLE001 - as the worker tells it
                dock.on_worker_error(step, str(exc))
            else:
                dock.on_worker_finished(step, True)

        monkeypatch.setattr(dock, "start_worker", run_now)
        return str(tmp_path / "survey" / "analytics_t")

    def test_the_button_writes_the_report_and_opens_it(
            self, dock, meadow, opened, told, monkeypatch):
        monkeypatch.setattr(dock.iface, "mapCanvas", lambda: Canvas(300, 200))
        dock.report_title_edit.setText("Deer on the meadow")

        dock.report_btn.click()

        path = os.path.join(meadow, "survey_report.html")
        assert told == []
        assert opened == [QUrl.fromLocalFile(path)]
        with open(path, encoding="utf-8") as handle:
            page = handle.read()
        assert "<h1>Deer on the meadow</h1>" in page
        assert "<td>What is counted</td><td>tracked animals, each once</td>" \
            in page
        assert "<tr class='total'><td>All animals</td><td class='n'>2</td>" \
            in page
        # the picture of the canvas is in the page, and nowhere else
        assert "src='data:image/png;base64," in page
        assert os.listdir(meadow).count("survey_report_map.png") == 0
        assert "Completed" in dock.report_status.text()
        log = dock.log_text.toPlainText()
        assert "The census has not been built for this survey yet" in log
        assert f"Survey report written to {path}" in log

    def test_without_the_tick_the_page_has_the_drawn_map_only(
            self, dock, meadow, opened, told):
        dock.report_canvas_check.setChecked(False)

        dock.report_btn.click()

        with open(os.path.join(meadow, "survey_report.html"),
                  encoding="utf-8") as handle:
            page = handle.read()
        assert "<img" not in page
        assert "<h2>Map</h2><figure><svg" in page
        assert "<h1>Survey report - survey</h1>" in page
