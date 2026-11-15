# -*- coding: utf-8 -*-
"""A step whose input has run again is shown as out of date.

The panel read the stage states for classification and track matching only.
Every other step said "Completed" for as long as its files existed, whatever
had run again before it - and several steps (flight route, ALFS, GeoTIFFs,
orthomosaic, the perpendicular distances, segmentation, BoxMOT tracking)
never recorded that they ran, so they could not be marked out of date at all.
"""
import os

import pytest

from bambi_wildlife_detection import bambi_processing
from bambi_wildlife_detection.bambi_processing import ProcessingWorker
from bambi_wildlife_detection.core import stages, store
from bambi_wildlife_detection.core.output_inventory import (
    check_existing_outputs, stale_steps)
from tests.fakes import SignalRecorder

ALL_T = {key: "_t" for key in (
    "extract", "flight_route", "detection", "tracking", "fov",
    "alfs", "geotiff", "ortho", "sam3", "inventory")}


def _folder(root, name):
    os.makedirs(os.path.join(root, name))
    with open(os.path.join(root, name, "output"), "w") as fh:
        fh.write("x")


@pytest.fixture
def flight(tmp_path):
    """A flight on which every step has run for the thermal camera."""
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    for name in ("detections_t", "georeferenced_t", "tracks_t", "fov_t",
                 "flight_route_t", "alfs_t", "geotiffs_t", "orthomosaic_t"):
        _folder(root, name)
    for name in ("perpendicular_t.json", "perpendicular_tracks_t.json"):
        with open(os.path.join(root, "flight_route_t", name), "w") as fh:
            fh.write("{}")
    for stage in ("flight_route", "detection", "georeference", "tracking",
                  "calculate_fov", "alfs", "export_geotiffs", "orthomosaic",
                  "perpendicular", "track_perpendicular"):
        stages.set_state(root, stage, "t", stages.COMPLETE)
    return root


def _stale(root, cameras=None):
    cameras = cameras or ALL_T
    return stale_steps(root, cameras, check_existing_outputs(root, cameras))


# ---------------------------------------------------------------------------
# What the panel is told
# ---------------------------------------------------------------------------

def test_nothing_is_out_of_date_on_a_flight_that_ran_through(flight):
    assert _stale(flight) == []


def test_detecting_again_outdates_what_follows(flight):
    stages.mark_complete(flight, "detection", "t")

    assert set(_stale(flight)) == {
        "georeference", "tracking", "perpendicular", "track_perpendicular"}


def test_extracting_again_outdates_every_step(flight):
    stages.mark_dependents_stale(flight, stages.FRAMES, "t")

    assert set(_stale(flight)) == {
        "flight_route", "detection", "georeference", "tracking",
        "calculate_fov", "alfs", "export_geotiffs", "orthomosaic",
        "perpendicular", "track_perpendicular"}


def test_running_the_step_again_brings_it_up_to_date(flight):
    stages.mark_complete(flight, "detection", "t")
    stages.mark_complete(flight, "georeference", "t")

    assert "georeference" not in _stale(flight)
    assert "tracking" in _stale(flight)


def test_the_other_cameras_steps_are_not_touched(flight):
    stages.mark_complete(flight, "detection", "w")
    assert _stale(flight) == []


def test_the_selected_camera_decides(flight):
    """Each step is looked up under the camera chosen for it."""
    _folder(flight, "tracks_w")
    stages.set_state(flight, "tracking", "w", stages.COMPLETE)
    stages.mark_complete(flight, "georeference", "t")

    assert "tracking" in _stale(flight)
    assert "tracking" not in _stale(flight, dict(ALL_T, tracking="_w"))


def test_a_reset_step_is_not_shown_as_out_of_date(flight):
    """It has no output any more: it is "Not started"."""
    stages.reset(flight, "georeference", "t")

    stale = _stale(flight)
    assert "georeference" not in stale
    assert "tracking" in stale


def test_a_folder_without_a_project_store(tmp_path):
    _folder(str(tmp_path), "tracks_t")
    assert _stale(str(tmp_path)) == []
    assert stale_steps("", ALL_T, ["tracking"]) == []


# ---------------------------------------------------------------------------
# Steps that used not to be on record
# ---------------------------------------------------------------------------

class _Processor:
    def __getattr__(self, name):
        return lambda config, progress_fn, log_fn, cancel_check: None


def _run(step, config):
    worker = ProcessingWorker(_Processor(), step, config)
    worker.finished = SignalRecorder()
    worker.error = SignalRecorder()
    worker.progress = SignalRecorder()
    worker.log = SignalRecorder()
    worker.run()
    assert worker.finished.calls == [(step, True)], worker.error.calls
    return worker


@pytest.mark.parametrize("step, camera_key", sorted(
    bambi_processing._STEPS_RECORDED_BY_WORKER.items()))
@pytest.mark.parametrize("camera, modality", [("T", "t"), ("W", "w")])
def test_the_step_is_recorded_when_it_succeeds(tmp_path, step, camera_key,
                                               camera, modality):
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()

    _run(step, {"target_folder": root, camera_key: camera})

    recorded = stages.states(root, modality)
    assert recorded[step]["state"] == stages.COMPLETE
    other = "w" if modality == "t" else "t"
    assert step not in stages.states(root, other)


def test_a_recorded_step_can_go_out_of_date(tmp_path):
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    _run("alfs", {"target_folder": root, "alfs_camera": "T"})
    _run("export_geotiffs", {"target_folder": root, "geotiff_camera": "T"})
    _run("orthomosaic", {"target_folder": root, "ortho_camera": "T"})

    assert set(stages.mark_dependents_stale(root, stages.FRAMES, "t")) == {
        "alfs", "export_geotiffs", "orthomosaic"}


def test_a_step_that_fails_is_not_recorded(tmp_path):
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()

    class Failing:
        def __getattr__(self, name):
            def step(config, progress_fn, log_fn, cancel_check):
                raise RuntimeError("no DEM")
            return step

    worker = ProcessingWorker(Failing(), "alfs",
                              {"target_folder": root, "alfs_camera": "T"})
    for name in ("finished", "error", "progress", "log"):
        setattr(worker, name, SignalRecorder())
    worker.run()

    assert worker.finished.calls == [("alfs", False)]
    assert "alfs" not in stages.states(root, "t")


def test_a_cancelled_step_is_not_recorded(tmp_path):
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    worker = ProcessingWorker(_Processor(), "alfs",
                              {"target_folder": root, "alfs_camera": "T"})
    for name in ("finished", "error", "progress", "log"):
        setattr(worker, name, SignalRecorder())
    worker.cancel()
    worker.run()
    assert "alfs" not in stages.states(root, "t")


def test_recording_never_fails_the_step(tmp_path, monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(bambi_processing, "_record_stage", broken)
    worker = _run("alfs", {"target_folder": str(tmp_path), "alfs_camera": "T"})
    assert any("could not record" in args[0] for args in worker.log.calls)


def test_steps_that_record_themselves_are_left_to_it():
    """Recording them here as well would overwrite the row count they
    record."""
    assert not set(bambi_processing._STEPS_RECORDED_BY_WORKER) & {
        "detection", "georeference", "tracking", "calculate_fov",
        "embeddings", "track_matching", "life_stage"}
