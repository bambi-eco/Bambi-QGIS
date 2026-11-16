# -*- coding: utf-8 -*-
"""The demo flight made ready off the main thread
(demo_flight_downloader.py): fetched, unpacked, and given its elevation
model - with the fetching stood in for."""
import os

import pytest

from bambi_wildlife_detection import (
    copernicus_dem_downloader, demo_flight_downloader)
from bambi_wildlife_detection.core import (
    checked_download, demo_data, environment)

from .fakes import SignalRecorder

BOUNDS = (15.9695, 47.0021, 15.9785, 47.0084)


@pytest.fixture
def made(tmp_path, monkeypatch):
    """A worker whose flight "arrives" at once: the flight log is there,
    and the elevation model is built by a stand-in."""
    folder = tmp_path / "demo"
    calls = {"prepare": [], "dem": []}

    def prepare(folder, session, progress=None, cancelled=None, log=None):
        calls["prepare"].append((folder, session.headers["User-Agent"]))
        os.makedirs(folder, exist_ok=True)
        log("Unpacking...")
        progress(100)
        airdata = os.path.join(folder, "air_data.csv")
        with open(airdata, "w", encoding="utf-8") as handle:
            handle.write("latitude,longitude,isVideo\n"
                         f"{BOUNDS[1]},{BOUNDS[0]},1\n"
                         f"{BOUNDS[3]},{BOUNDS[2]},1\n")
        return {"airdata": airdata}

    def build_dem(bounds, folder, crs, log=None, progress=None,
                  cancelled=None):
        calls["dem"].append((bounds, folder, crs))
        progress(100)
        return {"mesh": os.path.join(folder, "copernicus_dem.glb")}

    monkeypatch.setattr(demo_data, "prepare", prepare)
    monkeypatch.setattr(copernicus_dem_downloader, "build_dem", build_dem)

    def make(**more):
        worker = demo_flight_downloader.DemoFlightWorker(str(folder), **more)
        worker.finished = SignalRecorder()
        worker.progress = SignalRecorder()
        worker.log = SignalRecorder()
        return worker

    return make, calls, str(folder)


def test_the_flight_and_its_elevation_model(made):
    make, calls, folder = made
    worker = make()

    worker.run()

    assert calls["prepare"] == [(folder, "bambi-qgis-plugin")]
    # the ground under where the drone recorded, in the flight's UTM zone
    assert calls["dem"] == [(BOUNDS, folder, "EPSG:32633")]
    assert worker.finished.calls == [(True, folder)]
    assert worker.dem_problem == ""
    shown = [call[0] for call in worker.progress.calls]
    assert shown == sorted(shown) and shown[-1] == 100
    assert shown[0] == 85                 # the flight is most of the way
    assert ("Unpacking...",) in worker.log.calls


def test_a_model_that_cannot_be_fetched_does_not_cost_the_flight(
        made, monkeypatch):
    make, _calls, folder = made

    def build_dem(*args, **more):
        raise environment.EnvironmentFailure("no connection to the bucket")

    monkeypatch.setattr(copernicus_dem_downloader, "build_dem", build_dem)
    worker = make()

    worker.run()

    assert worker.finished.calls == [(True, folder)]
    assert worker.dem_problem == "no connection to the bucket"
    assert worker.log.calls[-1][0].startswith(
        "The elevation model could not be fetched (no connection to the "
        "bucket). The flight is there all the same")


def test_a_folder_that_has_its_model_keeps_it(made):
    make, calls, folder = made
    os.makedirs(folder)
    with open(os.path.join(folder, "copernicus_dem.glb"), "wb") as handle:
        handle.write(b"glTF")
    worker = make()

    worker.run()

    assert calls["dem"] == []
    assert worker.finished.calls == [(True, folder)]


def test_without_the_model(made):
    make, calls, folder = made
    worker = make(with_dem=False)

    worker.run()

    assert calls["dem"] == [] and worker.finished.calls == [(True, folder)]


def test_a_log_without_a_position_gives_no_model(made, monkeypatch):
    make, calls, folder = made
    monkeypatch.setattr(demo_data, "recording_bounds", lambda path: None)
    worker = make()

    worker.run()

    assert calls["dem"] == []
    assert worker.dem_problem == "the flight log holds no position"
    assert worker.finished.calls == [(True, folder)]


def test_a_flight_that_cannot_be_fetched_says_why(made, monkeypatch):
    make, calls, _folder = made

    def prepare(*args, **more):
        raise demo_data.DemoError("flight_306.zip could not be downloaded")

    monkeypatch.setattr(demo_data, "prepare", prepare)
    worker = make()

    worker.run()

    assert worker.finished.calls == [
        (False, "flight_306.zip could not be downloaded")]
    assert calls["dem"] == []


@pytest.mark.parametrize("stopped", [
    checked_download.Cancelled("cancelled"), environment.Cancelled()])
def test_cancelled_while_fetching_or_building(made, monkeypatch, stopped):
    make, _calls, _folder = made

    def prepare(*args, **more):
        raise stopped

    monkeypatch.setattr(demo_data, "prepare", prepare)
    worker = make()
    worker.cancel()

    worker.run()

    assert worker.finished.calls == [(False, "Cancelled")]


def test_cancelling_reaches_what_is_fetching(made, monkeypatch):
    make, _calls, _folder = made
    asked = []

    def prepare(folder, session, progress=None, cancelled=None, log=None):
        asked.append(cancelled())
        raise checked_download.Cancelled("cancelled")

    monkeypatch.setattr(demo_data, "prepare", prepare)
    worker = make()
    worker.cancel()

    worker.run()

    assert asked == [True]


def test_anything_else_is_told_whole(made, monkeypatch):
    make, _calls, _folder = made

    def prepare(*args, **more):
        raise OSError("disk full")

    monkeypatch.setattr(demo_data, "prepare", prepare)
    worker = make()

    worker.run()

    assert worker.finished.calls == [
        (False, "The demo flight could not be made ready: disk full")]
