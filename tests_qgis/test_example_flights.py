# -*- coding: utf-8 -*-
"""Input > "Download Example Flights": flights of the BAMBI dataset, by
their number (dock/example_flights.py).

What is fetched and how it is fitted to a flight is covered without QGIS
in tests/test_bambi_dataset*.py and tests/test_dataset_worker.py; the list
itself in test_dataset_dialog.py. Here the list is answered, the worker is
a stand-in, and nothing reaches the network.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import json
import os

import pytest

from qgis.PyQt.QtWidgets import QDialog, QMessageBox

from bambi_wildlife_detection import (
    bambi_dataset_dialog, bambi_dataset_downloader)
from bambi_wildlife_detection.core import bambi_dataset
from bambi_wildlife_detection.core import flights as flights_core
from bambi_wildlife_detection.dock import example_flights
from tests.test_flight_archive import MEMBERS

CORRECTION = {"translation": {"x": 0.0, "y": 0.0, "z": 46.65},
              "rotation": {"x": 0.0, "y": 0.0, "z": 0.0205},
              "additional": []}


class Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in list(self.slots):
            slot(*args)


class Thread:
    """Runs what is started on it there and then - or, when a test wants
    to act while the download runs, when the test says."""

    waiting = None               # a list: threads that were started
    made = []

    def __init__(self):
        self.started, self.finished = Signal(), Signal()
        Thread.made.append(self)

    def start(self):
        if Thread.waiting is None:
            self.started.emit()
        else:
            Thread.waiting.append(self)

    def quit(self, *_args):
        pass

    def deleteLater(self, *_args):
        pass


def put_flight(folder, with_dem=True, recordings=True):
    """The recordings of a flight in *folder*, as a download leaves them."""
    os.makedirs(os.path.join(folder, "qgis"), exist_ok=True)
    for name, data in MEMBERS.items() if recordings else ():
        with open(os.path.join(folder, name), "wb") as handle:
            handle.write(data)
    if with_dem:
        with open(os.path.join(folder, "gem15_dem.glb"), "wb") as handle:
            handle.write(b"glTF")
        with open(os.path.join(folder, "gem15_dem.json"), "w") as handle:
            json.dump({"crs": "EPSG:32633",
                       "origin": [573000.0, 5206000.0, 400.0]}, handle)


def prepared_flight(folder, number, route=True, correction=True,
                    labels=None, notes=(), videos=True, processed=False):
    """What ``core.bambi_dataset.prepare_flight`` returns for a flight."""
    from bambi_wildlife_detection.core import flight_archive

    flight = os.path.join(folder, f"flight_{number}")
    put_flight(flight, recordings=videos)
    target = os.path.join(flight, "qgis")
    imported = None
    if correction:
        path = os.path.join(target, f"{number}_correction_qgis.json")
        with open(path, "w") as handle:
            json.dump(CORRECTION, handle)
        imported = {"correction": CORRECTION, "path": path,
                    "convergence": 0.71}
    if route:
        os.makedirs(os.path.join(target, "flight_route_t"), exist_ok=True)
    return {
        "flight": number, "name": f"BAMBI flight {number}", "folder": flight,
        "target_folder": target,
        "files": flight_archive.flight_files(flight), "epsg": 32633,
        "dem": os.path.join(flight, "gem15_dem.glb"),
        "dem_metadata": os.path.join(flight, "gem15_dem.json"),
        "route": {"poses": os.path.join(target, "poses_t.json"),
                  "frames": 12, "replaced": False} if route else None,
        "correction": imported, "labels": labels,
        "labels_wanted": labels is not None, "notes": list(notes),
        "videos": flight_archive.is_complete(
            flight_archive.flight_files(flight)),
        "processed_video": os.path.join(
            flight, f"{number}_matched_processed.mp4") if processed else ""}


class Worker:
    """Stands for the download: leaves the flights' files and ends as the
    test says."""

    made = []
    outcome = "there"            # or a message: why it ended
    fails = {}                   # flight: why it could not be made ready
    notes = {}                   # flight: what there is to know
    labels = None                # what was placed, for every flight

    def __init__(self, flights, folder, route=True, labels=True,
                 correction=True, videos=True, version="processed"):
        self.flight_ids = [flight.id for flight in flights]
        self.folder, self.videos, self.version = folder, videos, version
        self.route, self.labels, self.correction = route, labels, correction
        self.finished, self.progress, self.log = Signal(), Signal(), Signal()
        self.flight_started = Signal()
        self.prepared, self.failures = [], []
        self.cancelled = False
        Worker.made.append(self)

    def moveToThread(self, thread):
        pass

    def deleteLater(self, *_args):
        pass

    def cancel(self):
        self.cancelled = True

    def run(self):
        count = len(self.flight_ids)
        for index, number in enumerate(self.flight_ids):
            if self.cancelled:
                self.finished.emit(False, "Cancelled")
                return
            self.flight_started.emit(number, index + 1, count)
            self.progress.emit(int(100 * (index + 0.4) / count))
            if number in Worker.fails:
                self.failures.append((number, Worker.fails[number]))
                continue
            self.prepared.append(prepared_flight(
                self.folder, number, route=self.route,
                videos=self.videos and self.version == "raw",
                processed=self.videos and self.version == "processed",
                correction=self.correction,
                labels=Worker.labels if self.labels else None,
                notes=Worker.notes.get(number, ())))
            self.log.emit(f"Unpacked 7 file(s) of flight {number}")
        if Worker.outcome != "there":
            self.finished.emit(False, Worker.outcome)
        elif self.prepared:
            self.finished.emit(True, self.folder)
        else:
            self.finished.emit(False, "\n".join(
                f"Flight {number}: {why}" for number, why in self.failures))


class Settings:
    """QSettings that forget when the test ends."""

    kept = {}

    def value(self, key, default=None, type=None):
        return Settings.kept.get(key, default)

    def setValue(self, key, value):
        Settings.kept[key] = value


@pytest.fixture
def asked(dock, tmp_path, monkeypatch):
    """The list answered - which flights, where to, what else - and the
    download stood in for. ``answers["accept"] = False`` closes the list
    without a download."""
    said, shown, routes = [], [], []
    answers = {"flights": [306], "folder": str(tmp_path / "flights"),
               "route": True, "labels": False, "correction": True,
               "videos": True, "version": "raw", "accept": True}
    listed = bambi_dataset.catalogue_from(
        [{"part": 1, "title": "The BAMBI Dataset (Part 1)",
          "deposition_id": 222, "doi": "10.5281/zenodo.222",
          "flights": ["7", "284", "306"]}],
        [{"part": 1, "title": "The BAMBI Raw Dataset (Part 1)",
          "deposition_id": 111, "doi": "10.5281/zenodo.111",
          "flights": ["7", "23", "284", "306"]}], {"val": [306]})
    monkeypatch.setattr(type(dock), "_dataset_flights",
                        staticmethod(lambda: listed))

    class Dialog:
        def __init__(self, flights, parent=None, folder="", records=None):
            shown.append({"flights": len(flights), "folder": folder,
                          "records": sorted(records or {})})

        def exec(self):
            return (QDialog.DialogCode.Accepted if answers["accept"]
                    else QDialog.DialogCode.Rejected)

        def selected_flights(self):
            return [listed["flights"][number]
                    for number in answers["flights"]]

        def download_videos(self):
            return answers["videos"]

        def version(self):
            return answers["version"]

        def folder(self):
            return answers["folder"]

        def import_route(self):
            return answers["route"]

        def import_labels(self):
            return answers["labels"]

        def use_correction(self):
            return answers["correction"]

    def told(kind):
        return lambda _parent, title, text, *a, **k: said.append(
            (kind, title, text))

    monkeypatch.setattr(bambi_dataset_dialog, "DatasetFlightsDialog", Dialog)
    monkeypatch.setattr(bambi_dataset_downloader, "DatasetFlightsWorker",
                        Worker)
    monkeypatch.setattr(example_flights, "QThread", Thread)
    monkeypatch.setattr(example_flights, "QSettings", Settings)
    monkeypatch.setattr(QMessageBox, "warning", told("warning"))
    monkeypatch.setattr(QMessageBox, "information", told("information"))
    monkeypatch.setattr(type(dock), "add_flight_route_to_qgis",
                        lambda self: routes.append(
                            self.target_folder_edit.text()))
    Worker.made, Worker.outcome, Worker.fails = [], "there", {}
    Worker.notes, Worker.labels = {}, None
    Thread.made, Thread.waiting = [], None
    Settings.kept = {}
    return {"said": said, "shown": shown, "answers": answers,
            "routes": routes, "folder": answers["folder"]}


def _names(text):
    return [os.path.basename(path) for path in text.split(", ") if path]


def _target(asked, number):
    return os.path.join(asked["folder"], f"flight_{number}", "qgis")


# ---------------------------------------------------------------------------
# The button, the list, the download
# ---------------------------------------------------------------------------

def test_the_input_tab_offers_the_flights(dock):
    assert dock.examples_btn.text() == "Download Example Flights…"
    assert "BAMBI dataset" in dock.examples_btn.toolTip()
    assert "CC BY 4.0" in dock.examples_btn.toolTip()
    assert "dataset's repository" in dock.examples_btn.toolTip()


def test_the_list_is_the_one_the_datasets_repository_keeps(dock, asked):
    dock.examples_btn.click()

    assert asked["shown"] == [{"flights": 4, "folder": "",
                               "records": [111, 222]}]


def test_what_was_ticked_is_what_is_fetched(dock, asked):
    asked["answers"].update(flights=[7, 306], labels=True, correction=False)

    dock.examples_btn.click()

    (worker,) = Worker.made
    assert worker.flight_ids == [7, 306]
    assert worker.folder == asked["folder"]
    assert (worker.route, worker.labels, worker.correction,
            worker.videos, worker.version) == (True, True, False, True,
                                               "raw")
    assert (f"Fetching 2 flight(s) of the BAMBI dataset with their route, "
            f"labels (raw video) into {asked['folder']}...") in \
        dock.log_text.toPlainText()


def test_the_folder_is_offered_again_the_next_time(dock, asked):
    dock.examples_btn.click()
    dock.examples_btn.click()

    assert asked["shown"][1]["folder"] == asked["folder"]


def test_closing_the_list_starts_nothing(dock, asked):
    asked["answers"]["accept"] = False

    dock.examples_btn.click()

    assert Worker.made == [] and asked["said"] == []
    assert not os.path.exists(asked["folder"])
    assert dock.examples_btn.text() == "Download Example Flights…"


def test_not_while_a_step_is_running(dock, asked, monkeypatch):
    monkeypatch.setattr(dock, "_step_is_running", lambda: True)

    dock.examples_btn.click()

    assert Worker.made == [] and asked["shown"] == []
    assert asked["said"] == [("information", "Example Flights",
                              "A step is running. Wait for it to finish "
                              "first.")]


def test_a_list_that_cannot_be_read_says_so(dock, asked, monkeypatch):
    def broken():
        raise bambi_dataset.DatasetError("no route to host")

    monkeypatch.setattr(type(dock), "_dataset_flights", staticmethod(broken))

    dock.examples_btn.click()

    ((kind, title, text),) = asked["said"]
    assert (kind, title) == ("warning", "Example Flights")
    assert text.startswith("The list of flights could not be read from the "
                           "dataset's repository:\nno route to host")
    assert asked["shown"] == []


# ---------------------------------------------------------------------------
# The flights in the project
# ---------------------------------------------------------------------------

def test_a_flight_becomes_a_flight_of_the_project(dock, asked):
    dock.examples_btn.click()

    flights = dock._flights()
    assert [flight["name"] for flight in flights] == ["BAMBI flight 306"]
    assert flights[0]["target_folder"] == _target(asked, 306)
    assert dock.flight_combo.currentText() == "BAMBI flight 306"
    done = asked["said"][-1]
    assert (done[0], done[1]) == ("information", "Example Flights")
    assert ("1 flight(s) are ready and in the project: 'BAMBI flight 306'. "
            "The Input tab shows 'BAMBI flight 306'.") in done[2]
    assert "Pre-Processing > Extract Frames" in done[2]
    assert dock.examples_btn.isEnabled()
    assert dock.examples_btn.text() == "Download Example Flights…"
    assert "Unpacked 7 file(s) of flight 306" in dock.log_text.toPlainText()
    assert "Added flight 'BAMBI flight 306'" in dock.log_text.toPlainText()


def test_the_input_tab_is_filled_with_the_flight(dock, asked):
    dock.examples_btn.click()

    folder = os.path.join(asked["folder"], "flight_306")
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
    assert dock.rgb_calib_preset_combo.currentIndex() == 0
    assert dock.rgb_calibration_path_edit.text() == os.path.join(
        folder, "W_calib.json")
    assert dock.dem_path_edit.text() == os.path.join(folder, "gem15_dem.glb")
    assert dock.dem_metadata_path_edit.text() == os.path.join(
        folder, "gem15_dem.json")
    assert dock.target_folder_edit.text() == os.path.join(folder, "qgis")
    assert dock.target_crs_edit.text() == "EPSG:32633"
    assert dock.video_mode_check.isChecked()
    assert not dock.embedded_srt_check.isChecked()


def test_the_published_correction_is_the_flights(dock, asked):
    dock.examples_btn.click()

    assert dock.rotation_unit_combo.currentIndex() == 0        # radians
    assert dock.trans_z_spin.value() == pytest.approx(46.65)
    assert dock.rot_z_spin.value() == pytest.approx(0.0205)
    assert dock.correction_path_edit.text() == os.path.join(
        _target(asked, 306), "306_correction_qgis.json")
    # ... and what the steps are run with
    assert dock.get_config()["translation"]["z"] == pytest.approx(46.65)


def test_without_it_the_correction_stays_at_nought(dock, asked):
    asked["answers"]["correction"] = False

    dock.examples_btn.click()

    assert dock.trans_z_spin.value() == 0.0
    assert dock.rot_z_spin.value() == 0.0
    assert dock.correction_path_edit.text() == ""


def test_the_settings_are_kept_in_the_flights_folder(dock, asked):
    dock.examples_btn.click()

    stored = flights_core.load_config(_target(asked, 306))
    assert stored["Input/TargetCrs"] == "EPSG:32633"
    assert stored["Input/AirdataPath"].endswith("air_data.csv")
    assert float(stored["Correction/TransZ"]) == pytest.approx(46.65)


def test_the_route_is_put_on_the_map(dock, asked):
    dock.examples_btn.click()

    assert asked["routes"] == [_target(asked, 306)]
    assert "The published route of 1 flight(s) is the flight's own and on " \
        "the map." in asked["said"][-1][2]


def test_no_route_no_layers(dock, asked):
    asked["answers"]["route"] = False

    dock.examples_btn.click()

    assert asked["routes"] == []
    assert "published route" not in asked["said"][-1][2]


def test_several_flights_are_all_added_and_the_first_is_shown(dock, asked):
    asked["answers"]["flights"] = [7, 284, 306]

    dock.examples_btn.click()

    assert [flight["name"] for flight in dock._flights()] == [
        "BAMBI flight 7", "BAMBI flight 284", "BAMBI flight 306"]
    assert dock.flight_combo.count() == 3
    assert dock.flight_combo.currentText() == "BAMBI flight 7"
    assert dock.target_folder_edit.text() == _target(asked, 7)
    # every route is on the map; the flight that stays shown came last
    assert sorted(asked["routes"]) == sorted(
        _target(asked, number) for number in (7, 284, 306))
    assert asked["routes"][-1] == _target(asked, 7)
    assert "3 flight(s) are ready and in the project" in asked["said"][-1][2]
    # each has its own settings
    for number in (7, 284, 306):
        stored = flights_core.load_config(_target(asked, number))
        assert stored["Input/TargetFolder"] == _target(asked, number)
        assert f"flight_{number}" in stored["Input/ThermalVideoPaths"]


def test_the_other_flights_can_be_switched_to(dock, asked):
    asked["answers"]["flights"] = [7, 306]
    dock.examples_btn.click()

    dock.flight_combo.setCurrentIndex(1)

    assert dock.target_folder_edit.text() == _target(asked, 306)
    assert "flight_306" in dock.thermal_video_paths_edit.text()
    assert dock.trans_z_spin.value() == pytest.approx(46.65)


def test_very_many_routes_are_not_all_put_on_the_map(dock, asked,
                                                     monkeypatch):
    monkeypatch.setattr(example_flights, "ROUTES_SHOWN", 2)
    asked["answers"]["flights"] = [7, 284, 306]

    dock.examples_btn.click()

    assert sorted(asked["routes"]) == sorted(
        _target(asked, number) for number in (7, 284))
    assert ("3 routes were imported; those of the first 2 flights are put "
            "on the map.") in dock.log_text.toPlainText()
    assert dock.flight_combo.currentText() == "BAMBI flight 7"


def test_a_project_with_flights_keeps_them(dock, asked, tmp_path):
    mine = tmp_path / "my flight" / "qgis"
    mine.mkdir(parents=True)
    dock.flight_name_edit.setText("My own flight")
    dock.target_folder_edit.setText(str(mine))
    dock._on_target_folder_changed()
    dock.thermal_video_paths_edit.setText("/data/mine_T.MP4")

    dock.examples_btn.click()

    assert [flight["name"] for flight in dock._flights()] == [
        "My own flight", "BAMBI flight 306"]
    assert dock.flight_combo.currentText() == "BAMBI flight 306"
    # what was typed for the other flight is with that flight
    assert flights_core.load_config(str(mine))[
        "Input/ThermalVideoPaths"] == "/data/mine_T.MP4"
    dock.flight_combo.setCurrentIndex(0)
    assert dock.thermal_video_paths_edit.text() == "/data/mine_T.MP4"
    assert dock.target_folder_edit.text() == str(mine)
    assert dock.trans_z_spin.value() == 0.0


def test_a_flight_that_is_in_the_project_is_not_added_twice(dock, asked):
    dock.examples_btn.click()
    dock.confidence_spin.setValue(0.33)       # set up since

    dock.examples_btn.click()

    assert [flight["name"] for flight in dock._flights()] == [
        "BAMBI flight 306"]
    assert dock.confidence_spin.value() == pytest.approx(0.33)
    assert float(flights_core.load_config(_target(asked, 306))[
        "Detection/Confidence"]) == pytest.approx(0.33)


def test_another_flight_of_that_name_is_told_apart(dock, asked, tmp_path):
    dock.examples_btn.click()
    asked["answers"]["folder"] = str(tmp_path / "elsewhere")

    dock.examples_btn.click()

    assert [flight["name"] for flight in dock._flights()] == [
        "BAMBI flight 306", "BAMBI flight 306 (2)"]
    assert dock._flights()[1]["target_folder"] == os.path.join(
        str(tmp_path / "elsewhere"), "flight_306", "qgis")


def test_a_flight_fetched_before_gets_the_correction_it_lacks(dock, asked):
    asked["answers"]["correction"] = False
    dock.examples_btn.click()
    dock.confidence_spin.setValue(0.33)
    assert dock.trans_z_spin.value() == 0.0

    asked["answers"]["correction"] = True
    dock.examples_btn.click()

    assert dock.trans_z_spin.value() == pytest.approx(46.65)
    assert dock.rot_z_spin.value() == pytest.approx(0.0205)
    assert dock.confidence_spin.value() == pytest.approx(0.33)


def test_a_correction_of_the_users_own_is_left_alone(dock, asked):
    asked["answers"]["correction"] = False
    dock.examples_btn.click()
    dock.trans_z_spin.setValue(12.0)

    asked["answers"]["correction"] = True
    dock.examples_btn.click()

    assert dock.trans_z_spin.value() == pytest.approx(12.0)
    assert dock.rot_z_spin.value() == 0.0


# ---------------------------------------------------------------------------
# Without the videos: a fast overview
# ---------------------------------------------------------------------------

def test_without_the_videos_the_flight_is_there_for_an_overview(dock, asked):
    asked["answers"]["videos"] = False

    dock.examples_btn.click()

    assert Worker.made[0].videos is False
    assert [flight["name"] for flight in dock._flights()] == [
        "BAMBI flight 306"]
    assert dock.thermal_video_paths_edit.text() == ""
    assert dock.airdata_path_edit.text() == ""
    assert dock.dem_path_edit.text().endswith("gem15_dem.glb")
    assert dock.target_crs_edit.text() == "EPSG:32633"
    assert dock.trans_z_spin.value() == pytest.approx(46.65)
    assert asked["routes"] == [_target(asked, 306)]
    done = asked["said"][-1][2]
    assert ("1 flight(s) are there without their raw recordings: for the "
            "steps from 'Extract Frames' on, ask for them again with "
            "'Download video data' ticked and the version 'Raw'.") in done
    assert "Go on with Pre-Processing" not in done
    assert "(without the videos)" in dock.log_text.toPlainText()


def test_the_processed_video_is_fetched_and_said(dock, asked):
    asked["answers"]["version"] = "processed"

    dock.examples_btn.click()

    assert Worker.made[0].version == "processed"
    assert dock.thermal_video_paths_edit.text() == ""
    done = asked["said"][-1][2]
    assert "The processed video of 1 flight(s) is in the flight's folder." \
        in done
    assert "without their raw recordings" in done
    assert "(processed video)" in dock.log_text.toPlainText()


def test_videos_fetched_later_are_filled_in(dock, asked):
    asked["answers"]["videos"] = False
    dock.examples_btn.click()
    dock.confidence_spin.setValue(0.33)             # set up since

    asked["answers"]["videos"] = True
    dock.examples_btn.click()

    assert [flight["name"] for flight in dock._flights()] == [
        "BAMBI flight 306"]
    assert _names(dock.thermal_video_paths_edit.text()) == [
        "DJI_20241119101935_0003_T.MP4"]
    assert dock.airdata_path_edit.text().endswith("air_data.csv")
    assert dock.thermal_calibration_path_edit.text().endswith("T_calib.json")
    assert dock.video_mode_check.isChecked()
    assert dock.confidence_spin.value() == pytest.approx(0.33)
    assert "Go on with Pre-Processing" in asked["said"][-1][2]


# ---------------------------------------------------------------------------
# What the user is told
# ---------------------------------------------------------------------------

def test_labels_that_were_placed_are_counted(dock, asked):
    asked["answers"]["labels"] = True
    Worker.labels = {"tracks": 2, "key_frames": 32, "boxes": 324,
                     "species": ["roe deer"], "on": "published",
                     "left_out": 0}

    dock.examples_btn.click()

    assert ("2 labelled animal(s) with 324 boxes are in the labelling tool "
            "and among the detections: go on with Processing > "
            "Geo-Reference Detections") in asked["said"][-1][2]


def test_labels_that_wait_for_the_frames_are_said(dock, asked, monkeypatch):
    asked["answers"].update(labels=True, route=False)
    monkeypatch.setattr(
        Worker, "run", lambda self: (
            self.prepared.append(dict(
                prepared_flight(self.folder, 306, route=False),
                labels=None, labels_wanted=True)),
            self.finished.emit(True, self.folder)))

    dock.examples_btn.click()

    assert ("The labels of 1 flight(s) are put on the frames when they are "
            "extracted.") in asked["said"][-1][2]


def test_what_did_not_work_for_a_flight_is_said(dock, asked):
    Worker.notes = {306: ["The elevation model could not be fetched (no "
                          "connection). Fetch it with 'Download DEM "
                          "(worldwide)' on the Input tab."]}

    dock.examples_btn.click()

    done = asked["said"][-1]
    assert done[0] == "information"
    assert ("To know:\nFlight 306: The elevation model could not be fetched "
            "(no connection).") in done[2]


def test_flights_that_failed_are_named_beside_those_that_are_ready(dock,
                                                                   asked):
    asked["answers"]["flights"] = [7, 306]
    Worker.fails = {7: "Flight 7 does not fit: 9.4 GB are needed"}

    dock.examples_btn.click()

    failed = asked["said"][-1]
    assert (failed[0], failed[1]) == ("warning", "Example Flights")
    assert "1 flight(s) are ready and in the project: 'BAMBI flight 306'" \
        in failed[2]
    assert ("Not ready:\nFlight 7: Flight 7 does not fit: 9.4 GB are "
            "needed") in failed[2]
    assert [flight["name"] for flight in dock._flights()] == [
        "BAMBI flight 306"]


def test_when_nothing_is_ready_it_says_why_and_can_be_tried_again(dock,
                                                                  asked):
    Worker.fails = {306: "flight_306.zip could not be downloaded: reset"}

    dock.examples_btn.click()

    failed = asked["said"][-1]
    assert (failed[0], failed[1]) == ("warning", "Example Flights")
    assert failed[2] == ("No flight could be made ready:\nFlight 306: "
                         "flight_306.zip could not be downloaded: reset")
    assert dock._flights() == []
    assert dock.thermal_video_paths_edit.text() == ""
    assert dock.examples_btn.isEnabled()
    assert dock.examples_btn.text() == "Download Example Flights…"


# ---------------------------------------------------------------------------
# While it runs
# ---------------------------------------------------------------------------

def test_the_button_says_how_far_it_is(dock, asked):
    asked["answers"]["flights"] = [7, 306]
    Thread.waiting = []
    dock.examples_btn.click()
    worker = Worker.made[0]

    assert dock.examples_btn.text() == "Cancel Download"
    assert dock.examples_btn.isEnabled()

    worker.flight_started.emit(306, 2, 2)
    worker.progress.emit(70)

    assert dock.examples_btn.text() == (
        "Cancel Download (flight 306, 2 of 2: 70%)")


def test_the_same_button_stops_it(dock, asked):
    asked["answers"]["flights"] = [7, 306]
    Thread.waiting = []
    dock.examples_btn.click()
    worker = Worker.made[0]

    dock.examples_btn.click()

    assert worker.cancelled is True
    assert len(Worker.made) == 1 and len(asked["shown"]) == 1
    assert dock.examples_btn.text() == "Stopping..."
    assert not dock.examples_btn.isEnabled()
    worker.progress.emit(80)                 # still arriving: not shown
    assert dock.examples_btn.text() == "Stopping..."

    Thread.waiting[0].started.emit()         # the worker notices

    assert dock.examples_btn.isEnabled()
    assert dock.examples_btn.text() == "Download Example Flights…"
    assert dock._flights() == []
    stopped = asked["said"][-1]
    assert stopped[0] == "information"
    assert stopped[2].startswith("The download was stopped. What arrived is "
                                 "kept")


def test_flights_that_were_ready_when_it_was_stopped_are_added(dock, asked,
                                                               monkeypatch):
    asked["answers"]["flights"] = [7, 306]

    def run(self):                            # stopped after the first
        self.prepared.append(prepared_flight(self.folder, 7))
        self.finished.emit(False, "Cancelled")

    monkeypatch.setattr(Worker, "run", run)

    dock.examples_btn.click()

    assert [flight["name"] for flight in dock._flights()] == [
        "BAMBI flight 7"]
    stopped = asked["said"][-1]
    assert stopped[0] == "information"
    assert "1 flight(s) are ready and in the project" in stopped[2]
    assert "The download was stopped." in stopped[2]
