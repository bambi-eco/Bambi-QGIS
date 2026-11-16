# -*- coding: utf-8 -*-
"""Input > "Download Demo Flight": a public example flight at a click
(dock/demo.py).

What is fetched and unpacked is covered without QGIS in
tests/test_demo_data.py and tests/test_demo_worker.py; here the worker is a
stand-in and nothing reaches the network.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import json
import os

import pytest

from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox

from bambi_wildlife_detection import demo_flight_downloader
from bambi_wildlife_detection.core import demo_data
from bambi_wildlife_detection.dock import demo as demo_dock
from tests.test_demo_data import MEMBERS


class Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in list(self.slots):
            slot(*args)


class Thread:
    """Runs what is started on it there and then."""

    def __init__(self):
        self.started, self.finished = Signal(), Signal()

    def start(self):
        self.started.emit()

    def quit(self, *_args):
        pass

    def deleteLater(self, *_args):
        pass


class Worker:
    """Stands for the download: puts the flight's files into the folder
    and ends as the test says."""

    made = []
    outcome = "there"            # or a message: why it failed
    dem_problem = ""

    def __init__(self, folder):
        self.folder = folder
        self.finished, self.progress, self.log = Signal(), Signal(), Signal()
        self.dem_problem = Worker.dem_problem
        Worker.made.append(self)

    def moveToThread(self, thread):
        pass

    def deleteLater(self, *_args):
        pass

    def run(self):
        self.progress.emit(40)
        if Worker.outcome != "there":
            self.finished.emit(False, Worker.outcome)
            return
        _put_flight(self.folder, with_dem=not self.dem_problem)
        self.log.emit("Unpacked 7 file(s)")
        self.finished.emit(True, self.folder)


def _put_flight(folder, with_dem=True):
    os.makedirs(folder, exist_ok=True)
    for name, data in MEMBERS.items():
        with open(os.path.join(folder, name), "wb") as handle:
            handle.write(data)
    if with_dem:
        with open(os.path.join(folder, "copernicus_dem.glb"), "wb") as handle:
            handle.write(b"glTF")
        with open(os.path.join(folder, "copernicus_dem.json"), "w") as handle:
            json.dump({"crs": "EPSG:32633",
                       "origin": [573000.0, 5206000.0, 400.0]}, handle)


@pytest.fixture
def asked(dock, tmp_path, monkeypatch):
    """The dialogs answered - the folder chosen, "yes" - and the download
    stood in for."""
    said = []
    answers = {"folder": str(tmp_path / "Documents"),
               "question": QMessageBox.StandardButton.Yes}
    os.makedirs(answers["folder"])

    def question(_parent, title, text, *args, **kwargs):
        said.append(("question", title, text))
        return answers["question"]

    def told(kind):
        return lambda _parent, title, text, *a, **k: said.append(
            (kind, title, text))

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a, **k: answers["folder"])
    monkeypatch.setattr(QMessageBox, "question", question)
    monkeypatch.setattr(QMessageBox, "warning", told("warning"))
    monkeypatch.setattr(QMessageBox, "information", told("information"))
    monkeypatch.setattr(demo_dock, "QThread", Thread)
    monkeypatch.setattr(demo_flight_downloader, "DemoFlightWorker", Worker)
    Worker.made, Worker.outcome, Worker.dem_problem = [], "there", ""
    return {"said": said, "answers": answers,
            "folder": os.path.join(answers["folder"], "BAMBI_demo_flight")}


def _names(text):
    return [os.path.basename(path) for path in text.split(", ") if path]


def test_the_input_tab_offers_the_demo_flight(dock):
    assert dock.demo_btn.text() == "Download Demo Flight…"
    assert "122 MB" in dock.demo_btn.toolTip()
    assert "CC BY 4.0" in dock.demo_btn.toolTip()


def test_it_goes_into_a_folder_of_its_own(dock, asked):
    dock.demo_btn.click()

    assert Worker.made[0].folder == asked["folder"]
    assert os.path.basename(asked["folder"]) == "BAMBI_demo_flight"


def test_a_folder_that_is_the_demos_is_taken_as_it_is(dock, asked, tmp_path):
    named = str(tmp_path / "BAMBI_demo_flight")
    holding = str(tmp_path / "my flight")
    _put_flight(holding)

    assert dock._demo_folder(named) == named
    assert dock._demo_folder(holding) == holding
    assert dock._demo_folder(str(tmp_path)) == named


def test_before_it_starts_it_says_what_it_is_and_whose(dock, asked):
    dock.demo_btn.click()

    kind, title, text = asked["said"][0]
    assert (kind, title) == ("question", "Demo Flight")
    assert "122 MB from Zenodo" in text
    assert f"Into: {asked['folder']}" in text
    assert "Licence: CC BY 4.0" in text
    assert demo_data.DEMO.citation in text


def test_the_input_tab_is_filled_with_the_flight(dock, asked):
    dock.demo_btn.click()

    folder = asked["folder"]
    assert _names(dock.thermal_video_paths_edit.text()) == [
        "DJI_20241119101935_0003_T.MP4"]
    assert _names(dock.thermal_srt_paths_edit.text()) == [
        "DJI_20241119101935_0003_T.SRT"]
    assert _names(dock.rgb_video_paths_edit.text()) == [
        "DJI_20241119101935_0003_W.MP4"]
    assert _names(dock.rgb_srt_paths_edit.text()) == [
        "DJI_20241119101935_0003_W.SRT"]
    assert dock.airdata_path_edit.text() == os.path.join(folder,
                                                         "air_data.csv")
    # the calibrations it came with, not a preset
    assert dock.thermal_calib_preset_combo.currentIndex() == 0
    assert dock.thermal_calibration_path_edit.text() == os.path.join(
        folder, "T_calib.json")
    assert dock.rgb_calibration_path_edit.text() == os.path.join(
        folder, "W_calib.json")
    assert dock.dem_path_edit.text() == os.path.join(folder,
                                                     "copernicus_dem.glb")
    assert dock.dem_metadata_path_edit.text() == os.path.join(
        folder, "copernicus_dem.json")
    assert dock.target_folder_edit.text() == os.path.join(folder, "qgis")
    assert dock.target_crs_edit.text() == "EPSG:32633"
    assert dock.video_mode_check.isChecked()
    assert not dock.embedded_srt_check.isChecked()


def test_it_is_the_projects_first_flight_under_its_own_name(dock, asked):
    dock.demo_btn.click()

    flights = dock._flights()
    assert [flight["name"] for flight in flights] == [
        "BAMBI demo flight 306"]
    assert flights[0]["target_folder"] == os.path.join(asked["folder"],
                                                       "qgis")
    assert dock.flight_name_edit.text() == ""       # taken, not left over
    done = asked["said"][-1]
    assert (done[0], done[1]) == ("information", "Demo Flight")
    assert "The demo flight is ready: 'BAMBI demo flight 306'" in done[2]
    assert "Pre-Processing > Extract Frames" in done[2]
    assert dock.demo_btn.isEnabled()
    assert dock.demo_btn.text() == "Download Demo Flight…"
    assert "Unpacked 7 file(s)" in dock.log_text.toPlainText()


def test_a_project_with_flights_keeps_its_input_tab(dock, asked, tmp_path):
    mine = tmp_path / "my flight" / "qgis"
    mine.mkdir(parents=True)
    dock.flight_name_edit.setText("My own flight")
    dock.target_folder_edit.setText(str(mine))
    dock._on_target_folder_changed()
    dock.thermal_video_paths_edit.setText("/data/mine_T.MP4")

    dock.demo_btn.click()

    assert [flight["name"] for flight in dock._flights()] == [
        "My own flight"]
    assert dock.thermal_video_paths_edit.text() == "/data/mine_T.MP4"
    assert dock.target_folder_edit.text() == str(mine)
    done = asked["said"][-1]
    assert done[0] == "information"
    assert f"The demo flight is in\n{asked['folder']}" in done[2]
    assert "type a name under 'Name', press +" in done[2]
    # ... and it was fetched all the same
    assert os.path.isfile(os.path.join(asked["folder"], "air_data.csv"))


def test_a_flight_without_its_model_says_what_to_do(dock, asked):
    Worker.dem_problem = "the Copernicus DEM could not be reached"

    dock.demo_btn.click()

    done = asked["said"][-1]
    assert ("Its elevation model could not be fetched (the Copernicus DEM "
            "could not be reached). Fetch it with 'Download DEM "
            "(worldwide)'") in done[2]
    assert dock.dem_path_edit.text() == ""
    assert _names(dock.thermal_video_paths_edit.text()) == [
        "DJI_20241119101935_0003_T.MP4"]


def test_when_it_fails_it_says_why_and_can_be_tried_again(dock, asked):
    Worker.outcome = "flight_306.zip could not be downloaded: reset"

    dock.demo_btn.click()

    failed = asked["said"][-1]
    assert (failed[0], failed[1]) == ("warning", "Demo Flight")
    assert "flight_306.zip could not be downloaded: reset" in failed[2]
    assert "pressing the button again continues" in failed[2]
    assert dock.thermal_video_paths_edit.text() == ""
    assert dock._flights() == []
    assert dock.demo_btn.isEnabled()


def test_cancelling_is_not_a_failure(dock, asked):
    Worker.outcome = "Cancelled"

    dock.demo_btn.click()

    assert [entry[0] for entry in asked["said"]] == ["question"]
    assert dock.demo_btn.isEnabled()


def test_no_folder_chosen_nothing_happens(dock, asked):
    asked["answers"]["folder"] = ""

    dock.demo_btn.click()

    assert Worker.made == [] and asked["said"] == []


def test_declining_starts_nothing(dock, asked):
    asked["answers"]["question"] = QMessageBox.StandardButton.No

    dock.demo_btn.click()

    assert Worker.made == []
    assert not os.path.exists(asked["folder"])


def test_not_while_a_step_is_running(dock, asked, monkeypatch):
    monkeypatch.setattr(dock, "_step_is_running", lambda: True)

    dock.demo_btn.click()

    assert Worker.made == []
    assert asked["said"] == [("information", "Demo Flight",
                              "A step is running. Wait for it to finish "
                              "first.")]
