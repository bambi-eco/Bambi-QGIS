# -*- coding: utf-8 -*-
"""Flights of the BAMBI dataset made ready off the main thread
(bambi_dataset_downloader.py): one after the other, each into a folder of
its own - with the making of a flight stood in for."""
import os

import pytest

from bambi_wildlife_detection import (
    bambi_dataset_downloader, bambi_processing, copernicus_dem_downloader)
from bambi_wildlife_detection.core import (
    bambi_dataset, checked_download, environment, remote_zip)

from .fakes import SignalRecorder


@pytest.fixture
def made(tmp_path, monkeypatch):
    """A worker whose flights "arrive" at once; ``calls`` is what was asked
    for, ``fate`` what becomes of a flight: nothing (it is ready) or the
    exception it ends with."""
    folder = str(tmp_path / "flights")
    calls, fate = [], {}

    def prepare_flight(entry, into, session, route=True, labels=True,
                       correction=True, elevation=None, make_route=None,
                       progress=None, cancelled=None, log=None):
        calls.append({"flight": entry.id, "into": into, "route": route,
                      "labels": labels, "correction": correction,
                      "elevation": elevation, "make_route": make_route,
                      "agent": session.headers["User-Agent"],
                      "cancelled": cancelled()})
        log(f"Downloading flight_{entry.id}.zip...")
        progress(50)
        if entry.id in fate:
            raise fate[entry.id]
        progress(100)
        return {"flight": entry.id, "name": entry.name, "folder": into,
                "target_folder": os.path.join(into, "qgis"), "notes": []}

    monkeypatch.setattr(bambi_dataset, "prepare_flight", prepare_flight)

    def make(flights=(306,), **more):
        worker = bambi_dataset_downloader.DatasetFlightsWorker(
            flights, folder, **more)
        for signal in ("finished", "progress", "log", "flight_started"):
            setattr(worker, signal, SignalRecorder())
        return worker

    return make, calls, fate, folder


def test_a_flight_goes_into_a_folder_of_its_own(made):
    make, calls, _fate, folder = made
    worker = make()

    worker.run()

    (call,) = calls
    assert call["flight"] == 306
    assert call["into"] == os.path.join(folder, "flight_306")
    assert call["agent"] == "bambi-qgis-plugin"
    assert worker.finished.calls == [(True, folder)]
    assert [flight["flight"] for flight in worker.prepared] == [306]
    assert worker.failures == []
    assert ("--- BAMBI flight 306 (1 of 1) ---",) in worker.log.calls
    assert ("Downloading flight_306.zip...",) in worker.log.calls


def test_what_was_ticked_is_what_is_taken(made):
    make, calls, _fate, _folder = made

    make(route=False, labels=True, correction=False).run()
    make().run()

    assert [(call["route"], call["labels"], call["correction"])
            for call in calls] == [(False, True, False), (True, True, True)]


def test_several_flights_one_after_the_other(made):
    make, calls, _fate, folder = made
    worker = make((306, 7, 284))

    worker.run()

    assert [call["flight"] for call in calls] == [306, 7, 284]
    assert worker.flight_started.calls == [(306, 1, 3), (7, 2, 3),
                                           (284, 3, 3)]
    assert worker.finished.calls == [(True, folder)]
    shown = [call[0] for call in worker.progress.calls]
    assert shown == sorted(shown) and shown[-1] == 100
    # a flight is a third of the way: half of the second is half-way
    assert 50 in shown and shown[0] == 0


def test_a_flight_that_fails_does_not_cost_the_others(made):
    make, calls, fate, folder = made
    fate[7] = bambi_dataset.DatasetError("Flight 7 does not fit: no room.")
    fate[284] = ValueError("bad archive")
    worker = make((306, 7, 284, 146))

    worker.run()

    assert [call["flight"] for call in calls] == [306, 7, 284, 146]
    assert [flight["flight"] for flight in worker.prepared] == [306, 146]
    assert worker.failures == [(7, "Flight 7 does not fit: no room."),
                               (284, "bad archive")]
    assert worker.finished.calls == [(True, folder)]
    assert ("Flight 7: Flight 7 does not fit: no room.",) in worker.log.calls
    assert ("Flight 284 could not be made ready: bad archive",) in \
        worker.log.calls


def test_when_no_flight_is_ready_it_says_why(made):
    make, _calls, fate, _folder = made
    fate[306] = bambi_dataset.DatasetError("flight_306.zip could not be "
                                           "downloaded: reset")
    worker = make()

    worker.run()

    assert worker.finished.calls == [
        (False, "Flight 306: flight_306.zip could not be downloaded: reset")]
    assert worker.prepared == []


def test_a_flight_the_dataset_does_not_have(made):
    make, calls, _fate, _folder = made
    worker = make((99999,))

    worker.run()

    assert calls == []
    assert worker.finished.calls == [
        (False, "Flight 99999: The BAMBI dataset has no flight 99999.")]


def test_nothing_asked_for(made):
    make, _calls, _fate, _folder = made
    worker = make(())

    worker.run()

    assert worker.finished.calls == [(False, "No flight was asked for.")]


@pytest.mark.parametrize("stopped", [
    checked_download.Cancelled("cancelled"), environment.Cancelled(),
    remote_zip.Cancelled("cancelled")])
def test_stopped_in_the_middle_keeps_what_is_ready(made, stopped):
    make, calls, fate, _folder = made
    fate[7] = stopped
    worker = make((306, 7, 284))

    worker.run()

    assert [call["flight"] for call in calls] == [306, 7]
    assert worker.finished.calls == [(False, "Cancelled")]
    assert [flight["flight"] for flight in worker.prepared] == [306]
    assert worker.failures == []


def test_stopped_between_two_flights(made):
    make, calls, _fate, _folder = made
    worker = make((306, 7))
    worker.cancel()

    worker.run()

    assert calls == [] and worker.is_cancelled()
    assert worker.finished.calls == [(False, "Cancelled")]


def test_the_flight_is_told_when_to_stop(made):
    make, calls, _fate, _folder = made
    worker = make()

    worker.run()
    worker.cancel()

    assert calls[0]["cancelled"] is False
    assert worker.is_cancelled()


def test_the_elevation_model_is_the_worldwide_one(made, monkeypatch):
    make, calls, _fate, _folder = made
    asked = []

    def build_dem(bounds, folder, crs, log=None, progress=None,
                  cancelled=None):
        asked.append((bounds, folder, crs))
        return {"mesh": os.path.join(folder, "gem15_dem.glb")}

    monkeypatch.setattr(copernicus_dem_downloader, "build_dem", build_dem)
    make().run()

    built = calls[0]["elevation"]((15.0, 47.0, 15.1, 47.1), "/there",
                                  "EPSG:32633")

    assert asked == [((15.0, 47.0, 15.1, 47.1), "/there", "EPSG:32633")]
    assert built == {"mesh": "/there/gem15_dem.glb"}


def test_without_an_elevation_model(made):
    make, calls, _fate, _folder = made

    make(with_dem=False).run()

    assert calls[0]["elevation"] is None


def test_the_routes_files_are_what_the_step_writes(made, monkeypatch):
    make, calls, _fate, _folder = made
    asked = []

    class Processor:
        def run_flight_route(self, config, log_fn=None):
            asked.append(config)
            log_fn("Flight route generation complete")

    monkeypatch.setattr(bambi_processing, "BambiProcessor", Processor)
    worker = make()
    worker.run()

    calls[0]["make_route"]({
        "files": {"airdata": "/there/air_data.csv"}, "epsg": 32633,
        "target_folder": "/there/qgis", "dem": "/there/gem15_dem.glb",
        "dem_metadata": "/there/gem15_dem.json"})

    assert asked == [{
        "flight_route_camera": "T", "target_folder": "/there/qgis",
        "target_epsg": 32633, "airdata_path": "/there/air_data.csv",
        "alfs_dem_metadata_path": "/there/gem15_dem.json",
        "dem_path": "/there/gem15_dem.glb", "filter_gps_origin": True}]
    assert ("Flight route generation complete",) in worker.log.calls
