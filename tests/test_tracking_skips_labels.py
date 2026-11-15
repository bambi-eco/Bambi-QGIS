# -*- coding: utf-8 -*-
"""A tracker does not track the hand-labelled boxes a second time.

"Add detections to project" writes the label boxes as detections *and* the
label tracks as tracks, under the labelling tool's own run, which every
reader pools with the tracker's run. The trackers took every detection as
input, the label boxes included - so a labelled animal was a manual track
and a tracker track at once: counted twice, exported twice.
"""
import os

import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import (
    detection_store, label_store, store, track_store)


@pytest.fixture
def processor():
    return BambiProcessor()


def _geo_reference_everything(root):
    track_store.record_georeference(root, "t", [
        {"detection_id": row["detection_id"],
         "gx1": 500000.0 + row["x1"], "gy1": 5300000.0 + row["y1"],
         "gz1": 400.0,
         "gx2": 500000.0 + row["x2"], "gy2": 5300000.0 + row["y2"],
         "gz2": 400.0}
        for row in track_store.load_detections(root, "t")])


def _label_track(label_track_id=1, x=200.0):
    return {"label_track_id": label_track_id, "species_id": 0,
            "attributes": {},
            "keyframes": [
                {"frame": 0, "x1": x, "y1": 200.0, "x2": x + 20, "y2": 220.0},
                {"frame": 2, "x1": x + 2, "y1": 202.0, "x2": x + 22,
                 "y2": 222.0}]}


@pytest.fixture
def project(tmp_path):
    """One animal the detector found on three frames, and one that was
    labelled by hand on the same three frames, somewhere else."""
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    detection_store.record_detections(root, "t", [
        {"frame": frame, "x1": 10.0 + frame, "y1": 20.0, "x2": 30.0 + frame,
         "y2": 40.0, "confidence": 0.9, "source_class": "0"}
        for frame in range(3)])
    label_store.save_tracks(root, "t", [_label_track()])
    label_store.materialise(root, "t")
    _geo_reference_everything(root)
    return root


def _config(root):
    return {"target_folder": root, "tracking_camera": "T",
            "tracker_mode": "HUNGARIAN", "write_legacy_text_outputs": True}


def _ids(root, labelled):
    conn = store.open_store(
        store.stage_path(root, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    try:
        return {row["detection_id"] for row in conn.execute(
            "SELECT detection_id FROM detections WHERE label_track_id IS "
            + ("NOT NULL" if labelled else "NULL"))}
    finally:
        conn.close()


def _tracks_by_run(root):
    """``{kind: {track_id: {detection ids}}}`` of the runs that are read."""
    conn = store.open_store(
        store.stage_path(root, store.TRACKS, "t"), store.TRACKS, "t")
    try:
        runs = track_store.analysis_runs(root, "t")
        result = {}
        for row in conn.execute(
                "SELECT r.kind AS kind, t.track_id AS track_id, "
                "m.detection_id AS detection_id FROM tracks t "
                "JOIN track_runs r ON r.run_id = t.run_id "
                "LEFT JOIN track_members m ON m.track_id = t.track_id"):
            run = conn.execute(
                "SELECT run_id FROM tracks WHERE track_id = ?",
                (row["track_id"],)).fetchone()["run_id"]
            if run not in runs:
                continue
            members = result.setdefault(row["kind"], {}).setdefault(
                row["track_id"], set())
            if row["detection_id"] is not None:
                members.add(row["detection_id"])
        return result
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# What a tracker is given
# ---------------------------------------------------------------------------

def test_the_loaders_can_leave_the_labelled_boxes_out(project):
    detector, labelled = _ids(project, False), _ids(project, True)
    assert len(detector) == 3 and len(labelled) == 3

    everything = {r["detection_id"]
                  for r in track_store.load_detections(project, "t")}
    assert everything == detector | labelled
    assert {r["detection_id"] for r in track_store.load_detections(
        project, "t", for_tracking=True)} == detector

    assert {r["detection_id"] for r in track_store.load_georeferenced(
        project, "t")} == detector | labelled
    assert {r["detection_id"] for r in track_store.load_georeferenced(
        project, "t", for_tracking=True)} == detector

    assert track_store.labelled_boxes(project, "t") == 3
    assert track_store.labelled_boxes(project, "w") == 0


# ---------------------------------------------------------------------------
# The built-in tracker
# ---------------------------------------------------------------------------

def test_a_labelled_animal_is_one_track(processor, project):
    log = []
    processor._run_builtin_tracking(_config(project), log_fn=log.append)

    tracks = _tracks_by_run(project)
    assert sorted(tracks) == ["builtin", "manual"]
    (tracked,) = tracks["builtin"].values()
    (labelled,) = tracks["manual"].values()
    assert tracked == _ids(project, False)
    assert labelled == _ids(project, True)
    assert any("3 hand-labelled box(es)" in line for line in log)


def test_no_box_belongs_to_two_tracks(processor, project):
    """What the analytics and the exports see."""
    processor._run_builtin_tracking(_config(project))

    rows = track_store.load_pixel_tracks(project, "t")
    by_detection = {}
    for row in rows:
        by_detection.setdefault(row["detection_id"], set()).add(row["track_id"])
    assert all(len(tracks) == 1 for tracks in by_detection.values())
    assert len({row["track_id"] for row in rows}) == 2


def test_only_labelled_boxes_is_no_error(processor, project):
    """After "Replace detections in project" the labels are all there is.
    The tracker has nothing to do then - and its earlier run, of boxes that
    are gone, must stop being read."""
    processor._run_builtin_tracking(_config(project))
    label_store.clear_detector_detections(project, "t")

    log = []
    processor._run_builtin_tracking(_config(project), log_fn=log.append)

    tracks = _tracks_by_run(project)
    assert sorted(tracks) == ["manual"]
    assert any("nothing for the tracker to associate" in line for line in log)
    with open(os.path.join(project, "tracks_t", "tracks.csv")) as fh:
        assert fh.read() == ""


def test_no_boxes_at_all_is_still_an_error(processor, tmp_path):
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    detection_store.record_detections(root, "t", [])
    track_store.record_georeference(root, "t", [])

    with pytest.raises(RuntimeError, match="No geo-referenced detections"):
        processor._run_builtin_tracking(_config(root))
