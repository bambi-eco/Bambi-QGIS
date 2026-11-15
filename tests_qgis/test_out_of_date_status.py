# -*- coding: utf-8 -*-
"""The panel shows a step as out of date when its input has run again.

Only classification and track matching used to show it; detection,
geo-referencing, tracking and the rest stayed "Completed" for as long as
their files existed.

Run via the QGIS image only::

    docker compose run --rm qgis-tests
"""
import os

import pytest

from bambi_wildlife_detection.core import stages, store

DONE = "🟢 Completed"
OUTDATED = "🟠 Out of date"


@pytest.fixture
def flight(tmp_path):
    root = str(tmp_path / "flight")
    os.makedirs(root)
    store.open_store(store.project_path(root), store.PROJECT).close()
    for name in ("detections_t", "georeferenced_t", "tracks_t", "alfs_t"):
        os.makedirs(os.path.join(root, name))
        with open(os.path.join(root, name, "output"), "w") as fh:
            fh.write("x")
    for stage in ("detection", "georeference", "tracking", "alfs"):
        stages.set_state(root, stage, "t", stages.COMPLETE)
    return root


def _shown(dock):
    return {"detection": dock.detect_status.text(),
            "georeference": dock.georef_status.text(),
            "tracking": dock.track_status.text(),
            "alfs": dock.alfs_status.text()}


def test_a_flight_that_ran_through_is_completed(dock, flight):
    dock.target_folder_edit.setText(flight)
    dock._on_target_folder_changed()
    dock._check_existing_outputs(flight)

    assert set(_shown(dock).values()) == {DONE}


def test_detecting_again_shows_what_follows_as_out_of_date(dock, flight):
    dock.target_folder_edit.setText(flight)
    dock._on_target_folder_changed()
    stages.mark_complete(flight, "detection", "t")

    dock._check_existing_outputs(flight)

    assert _shown(dock) == {"detection": DONE, "georeference": OUTDATED,
                            "tracking": OUTDATED, "alfs": DONE}


def test_extracting_again_shows_every_step_as_out_of_date(dock, flight):
    dock.target_folder_edit.setText(flight)
    dock._on_target_folder_changed()
    stages.mark_dependents_stale(flight, stages.FRAMES, "t")

    dock._check_existing_outputs(flight)

    assert set(_shown(dock).values()) == {OUTDATED}


def test_running_the_step_again_clears_it(dock, flight):
    dock.target_folder_edit.setText(flight)
    dock._on_target_folder_changed()
    stages.mark_complete(flight, "detection", "t")
    stages.mark_complete(flight, "georeference", "t")

    dock._check_existing_outputs(flight)

    assert _shown(dock)["georeference"] == DONE
    assert _shown(dock)["tracking"] == OUTDATED
