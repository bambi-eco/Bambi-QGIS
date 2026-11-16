# -*- coding: utf-8 -*-
"""Input > Auto-Download > "Download DEM (worldwide)": the button that makes
a flight's elevation model from the Copernicus DEM (dock/input_tab.py).

What is fetched and built is covered without QGIS in
tests/test_copernicus_dem.py and tests/test_copernicus_build.py; here the
worker is a stand-in and nothing reaches the network.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from qgis.PyQt.QtWidgets import QMessageBox

from bambi_wildlife_detection import copernicus_dem_downloader
from bambi_wildlife_detection.dock import input_tab


class Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in list(self.slots):
            slot(*args)


class Thread:
    """Runs what is started on it there and then. Like Qt's own slots, its
    methods take whatever a signal hands them."""

    def __init__(self):
        self.started, self.finished = Signal(), Signal()

    def start(self):
        self.started.emit()

    def quit(self, *_args):
        pass

    def deleteLater(self, *_args):
        pass


class Worker:
    """Stands for the download: remembers what it was asked for and ends
    as the test says."""

    made = []
    outcome = (True, "")

    def __init__(self, bounds, output_folder, output_crs, padding,
                 source=None):
        self.asked = (tuple(bounds), output_folder, output_crs, padding)
        self.source = source
        self.finished, self.progress, self.log = Signal(), Signal(), Signal()
        Worker.made.append(self)

    def moveToThread(self, thread):
        pass

    def deleteLater(self, *_args):
        pass

    def run(self):
        self.progress.emit(40)
        self.log.emit("Reading Copernicus DEM tile ...")
        self.finished.emit(*Worker.outcome)


@pytest.fixture
def flight(dock, tmp_path, monkeypatch):
    """A flight in Kenya - UTM zone 37 south - known by its AirData CSV,
    with the dialogs answered "yes" and the download stood in for."""
    csv_path = tmp_path / "flight" / "airdata.csv"
    csv_path.parent.mkdir()
    rows = ["latitude,longitude,isVideo",
            "-1.4000,36.9000,0",               # before the recording
            "-1.2900,36.8200,1",
            "-1.2850,36.8260,1",
            "0,0,1",                           # no fix
            "-1.2880,36.8230,1"]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    dock.airdata_path_edit.setText(str(csv_path))
    dock.target_crs_edit.setText("EPSG:32633")

    asked = []
    answers = {"question": QMessageBox.StandardButton.Yes}

    def question(_parent, title, text, *args, **kwargs):
        asked.append(("question", title, text))
        return answers["question"]

    def said(kind):
        return lambda _parent, title, text, *a, **k: asked.append(
            (kind, title, text))

    monkeypatch.setattr(QMessageBox, "question", question)
    monkeypatch.setattr(QMessageBox, "warning", said("warning"))
    monkeypatch.setattr(QMessageBox, "information", said("information"))
    monkeypatch.setattr(input_tab, "QThread", Thread)
    monkeypatch.setattr(copernicus_dem_downloader, "CopernicusDEMWorker",
                        Worker)
    Worker.made = []
    mesh = csv_path.parent / "copernicus_dem.glb"
    Worker.outcome = (True, str(mesh))
    return {"csv": csv_path, "asked": asked, "answers": answers,
            "mesh": mesh}


def test_the_auto_download_tab_has_the_button(dock):
    assert dock.copernicus_dem_btn.text() == "Download DEM (worldwide)"
    assert "surface model" in dock.copernicus_dem_btn.toolTip()
    assert dock.dem_download_btn.text() == "Download DEM (Austria)"


def test_where_the_flight_was_is_read_from_the_recording(dock, flight):
    lats, lons = dock._flight_gps_positions()

    # the rows of the recording that have a fix
    assert lats == [-1.29, -1.285, -1.288]
    assert lons == [36.82, 36.826, 36.823]


def test_it_fetches_the_ground_under_the_flight(dock, flight):
    dock.dem_padding_spin.setValue(250)

    dock.copernicus_dem_btn.click()

    worker = Worker.made[0]
    bounds, folder, crs, padding = worker.asked
    assert bounds == (36.82, -1.29, 36.826, -1.285)
    assert folder == str(flight["csv"].parent)
    assert (crs, padding) == ("EPSG:32737", 250.0)


def test_the_target_crs_becomes_the_flights_zone(dock, flight):
    dock.copernicus_dem_btn.click()

    kind, title, text = flight["asked"][0]
    assert (kind, title) == ("question", "Target CRS")
    assert "The flight lies in the UTM zone EPSG:32737" in text
    assert "the target CRS is EPSG:32633" in text
    assert "No keeps EPSG:32633" in text
    assert dock.target_crs_edit.text() == "EPSG:32737"


def test_a_crs_that_is_the_flights_zone_is_not_asked_about(dock, flight):
    dock.target_crs_edit.setText("epsg:32737")

    dock.copernicus_dem_btn.click()

    assert [entry[1] for entry in flight["asked"]
            if entry[0] == "question"] == ["Download DEM"]
    assert Worker.made[0].asked[2] == "EPSG:32737"


def test_before_it_starts_it_says_what_it_is(dock, flight):
    dock.copernicus_dem_btn.click()

    confirm = [entry for entry in flight["asked"]
               if entry[1] == "Download DEM"][0]
    assert "Fetch the elevation under the flight?" in confirm[2]
    assert ("From: the terrain model GEM-15 (15 m) where it has the flight, "
            "else the Copernicus DEM (30 m)") in confirm[2]
    assert "Padding: 150 m" in confirm[2]           # never less
    assert "Output CRS: EPSG:32737" in confirm[2]
    assert "GEM-15 is the bare ground" in confirm[2]
    assert "Where there is a national terrain model of the area" in confirm[2]


def test_the_terrain_model_is_asked_for_unless_told_otherwise(dock, flight):
    from bambi_wildlife_detection.copernicus_dem_downloader import (
        AUTO, COPERNICUS, GEM15)

    combo = dock.world_dem_source_combo
    assert [combo.itemData(i) for i in range(combo.count())] == [
        AUTO, GEM15, COPERNICUS]
    assert combo.currentData() == AUTO
    assert "Best available" in combo.currentText()

    dock.copernicus_dem_btn.click()

    assert Worker.made[0].source == AUTO


def test_the_surface_model_can_be_asked_for(dock, flight):
    from bambi_wildlife_detection.copernicus_dem_downloader import COPERNICUS

    combo = dock.world_dem_source_combo
    combo.setCurrentIndex(combo.findData(COPERNICUS))

    dock.copernicus_dem_btn.click()

    confirm = [entry for entry in flight["asked"]
               if entry[1] == "Download DEM"][0]
    assert "From: the Copernicus DEM (30 m)" in confirm[2]
    assert "The Copernicus DEM is a surface model" in confirm[2]
    assert "GEM-15 is the bare ground" not in confirm[2]
    assert Worker.made[0].source == COPERNICUS


def test_only_the_terrain_model_can_be_asked_for(dock, flight):
    from bambi_wildlife_detection.copernicus_dem_downloader import GEM15

    combo = dock.world_dem_source_combo
    combo.setCurrentIndex(combo.findData(GEM15))

    dock.copernicus_dem_btn.click()

    confirm = [entry for entry in flight["asked"]
               if entry[1] == "Download DEM"][0]
    assert "From: the terrain model GEM-15 (15 m)\n" in confirm[2]
    assert Worker.made[0].source == GEM15


def test_when_it_is_done_the_model_is_the_flights_dem(dock, flight):
    flight["mesh"].write_bytes(b"glTF")
    flight["mesh"].with_suffix(".json").write_text(
        '{"crs": "EPSG:32737", "origin": [257000.0, 9857000.0, 1600.0], '
        '"source": {"product": "Copernicus DEM GLO-30", '
        '"surface_model": true, "resolution_m": 30.0}}')

    dock.copernicus_dem_btn.click()

    assert dock.dem_path_edit.text() == str(flight["mesh"])
    assert dock.dem_metadata_path_edit.text() == str(
        flight["mesh"].with_suffix(".json"))
    done = flight["asked"][-1]
    assert (done[0], done[1]) == ("information", "DEM Download Complete")
    assert done[2].startswith("The elevation model is there.\n\nMesh: ")
    assert ("It was made from the Copernicus DEM GLO-30: a surface model at "
            "30 m. Over forest it lies in the canopy - allow for that under "
            "'Position Uncertainty' (Terrain model).") in done[2]
    assert dock.copernicus_dem_btn.isEnabled()
    assert dock.copernicus_dem_btn.text() == "Download DEM (worldwide)"
    assert "Reading Copernicus DEM tile ..." in dock.log_text.toPlainText()


def test_when_it_fails_it_says_why_and_can_be_tried_again(dock, flight):
    Worker.outcome = (False, "The Copernicus DEM has no heights where the "
                             "flight is: open sea.")

    dock.copernicus_dem_btn.click()

    failed = flight["asked"][-1]
    assert (failed[0], failed[1]) == ("warning", "DEM Download Failed")
    assert "open sea" in failed[2]
    assert dock.dem_path_edit.text() == ""
    assert dock.copernicus_dem_btn.isEnabled()
    assert dock.copernicus_dem_btn.text() == "Download DEM (worldwide)"


def test_declining_starts_nothing(dock, flight):
    dock.target_crs_edit.setText("EPSG:32737")
    flight["answers"]["question"] = QMessageBox.StandardButton.No

    dock.copernicus_dem_btn.click()

    assert Worker.made == []
    assert dock.copernicus_dem_btn.isEnabled()


def test_keeping_another_utm_zone_is_the_users_choice(dock, flight,
                                                      monkeypatch):
    answers = iter([QMessageBox.StandardButton.No,       # keep 32633
                    QMessageBox.StandardButton.Yes])     # go ahead
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: next(answers))

    dock.copernicus_dem_btn.click()

    assert dock.target_crs_edit.text() == "EPSG:32633"
    assert Worker.made[0].asked[2] == "EPSG:32633"


def test_without_a_utm_crs_no_cancels(dock, flight):
    dock.target_crs_edit.setText("EPSG:4326")
    flight["answers"]["question"] = QMessageBox.StandardButton.No

    dock.copernicus_dem_btn.click()

    assert "No cancels" in flight["asked"][0][2]
    assert Worker.made == []
    assert len(flight["asked"]) == 1


def test_nobody_knows_where_the_flight_was(dock, flight):
    dock.airdata_path_edit.setText("")

    dock.copernicus_dem_btn.click()

    assert flight["asked"] == [(
        "warning", "Missing GPS Data",
        "Nobody knows where the flight was yet.\n\nPlease select an "
        "AirData CSV or SRT files in the input section first.")]
    assert Worker.made == []


def test_without_a_csv_the_model_goes_into_the_target_folder(
        dock, flight, tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir()
    dock.target_folder_edit.setText(str(target))
    dock._on_target_folder_changed()
    dock.airdata_path_edit.setText("")
    monkeypatch.setattr(dock, "_flight_gps_positions",
                        lambda config=None: ([48.365, 48.372],
                                             [14.508, 14.518]))
    dock.target_crs_edit.setText("EPSG:32633")

    dock.copernicus_dem_btn.click()

    assert Worker.made[0].asked[1] == str(target)
    assert Worker.made[0].asked[2] == "EPSG:32633"
    assert os.path.isdir(target)
