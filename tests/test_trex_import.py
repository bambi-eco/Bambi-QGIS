# -*- coding: utf-8 -*-
"""A TRex import ends up in the store like any other flight's results.

Up to 9.0 the import projected the tracklets itself and wrote their
positions to ``georeferenced.txt`` and ``tracks.csv`` only. Nothing reads
those files since 6.0: an imported flight had boxes and tracks in the store
but no position for any of them, so "Add Tracks to QGIS" had nothing to add
and every analysis counted no animal. Beside a detector run or hand labels
not even the tracks were recorded, and boxes made on the raw video were
stored in the video's pixels where every reader expects the frame's.

These tests run the whole step on a small flight over flat ground.
"""
import json
import os

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("trimesh")
pytest.importorskip("alfspy")
pytest.importorskip("bambi")
pytest.importorskip("gltflib")

from bambi_wildlife_detection.bambi_processing import BambiProcessor  # noqa: E402
from bambi_wildlife_detection.core import (  # noqa: E402
    analytics_source, detection_store, pipeline_outputs, stages, store,
    track_store, trex_tracklets)

WIDTH, HEIGHT = 640, 512
N_FRAMES = 3
HEIGHT_AGL = 100.0
FOVY = 50.0
#: Metres on the ground per pixel, straight down from 100 m.
GSD = 2 * HEIGHT_AGL * np.tan(np.radians(FOVY / 2)) / HEIGHT

MATRIX = [[520.0, 0.0, 322.0], [0.0, 518.0, 254.0], [0.0, 0.0, 1.0]]
DISTORTION = [-0.25, 0.08, 0.001, -0.0005, 0.0]


def _frames(root, width=WIDTH, height=HEIGHT, count=N_FRAMES):
    import cv2

    folder = os.path.join(root, "frames_t")
    os.makedirs(folder, exist_ok=True)
    for index in range(count):
        cv2.imwrite(os.path.join(folder, f"frame_{index:06d}.png"),
                    np.zeros((height, width, 3), dtype=np.uint8))


def _tracklet(folder, track_id, frames, left, top=250.0, size=20.0,
              video_size=(WIDTH, HEIGHT)):
    os.makedirs(folder, exist_ok=True)
    left = np.asarray(left, dtype=float)
    count = len(frames)
    np.savez(
        os.path.join(folder, f"video_id{track_id}.npz"),
        id=np.array([track_id]), frame=np.asarray(frames),
        detection_p=np.linspace(0.9, 0.7, count),
        detection_class=np.zeros(count),
        poseX0=left, poseY0=np.full(count, top),
        poseX1=left + size, poseY1=np.full(count, top + size),
        video_size=np.asarray(video_size))


@pytest.fixture
def flight(tmp_path):
    """Three thermal frames from a camera 100 m above flat ground, looking
    straight down, and two tracklets seen on each of them."""
    root = str(tmp_path / "flight")
    os.makedirs(root)
    _frames(root)
    glb = os.path.join(root, "dem.glb")
    BambiProcessor.generate_flat_surface_mesh(
        47.0, 14.0, 500.0, 500.0, glb, os.path.join(root, "dem.json"),
        epsg=32633)
    with open(os.path.join(root, "poses_t.json"), "w") as handle:
        json.dump({"images": [
            {"imagefile": f"frame_{index:06d}.png",
             "location": [float(index), 0.0, HEIGHT_AGL],
             "rotation": [0.0, 0.0, 0.0], "fovy": [FOVY]}
            for index in range(N_FRAMES)]}, handle)

    tracklets = os.path.join(root, "trex")
    frames = list(range(N_FRAMES))
    _tracklet(tracklets, 3, frames, [200.0, 202.0, 204.0])
    _tracklet(tracklets, 9, frames, [300.0, 302.0, 304.0])
    return {
        "root": root,
        "config": {"tracking_camera": "T", "trex_already_undistorted": True,
                   "trex_npz_dir": tracklets, "target_folder": root,
                   "dem_path": glb, "target_epsg": 32633},
    }


def _import(flight, **overrides):
    log = []
    BambiProcessor().run_trex_import(
        dict(flight["config"], **overrides), log_fn=log.append)
    return log


def _centre(point):
    return ((point["x1"] + point["x2"]) / 2, (point["y1"] + point["y2"]) / 2)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def test_the_tracklets_have_positions_in_the_store(flight):
    _import(flight)
    root = flight["root"]

    assert os.path.isfile(store.stage_path(root, store.GEOREFERENCED, "t"))
    tracks = pipeline_outputs.geo_tracks(root, "t")
    assert len(tracks) == 2
    assert all(len(points) == N_FRAMES for points in tracks.values())


def test_they_are_where_the_camera_saw_them(flight):
    """100 px apart on the frame is 100 ground-sample distances apart."""
    _import(flight)
    first, second = pipeline_outputs.geo_tracks(flight["root"], "t").values()

    for a, b in zip(first, second):
        (ax, ay), (bx, by) = _centre(a), _centre(b)
        assert abs(bx - ax) == pytest.approx(100 * GSD, abs=0.02)
        assert by == pytest.approx(ay, abs=0.02)
    # 2 px to the right per frame, and the drone a metre further each time
    (x0, _y0), (x1, _y1) = _centre(first[0]), _centre(first[1])
    assert x1 - x0 == pytest.approx(1.0 + 2 * GSD, abs=0.02)


def test_the_survey_tools_count_the_animals(flight):
    _import(flight)
    records, facts = analytics_source.load_animals(flight["root"], "t")

    assert len(records) == 2
    assert all(record["x"] is not None and record["y"] is not None
               for record in records)
    assert facts["unplaced"] == 0


def test_every_box_is_accounted_for(flight):
    """A tracklet frame the extraction has no pose for is recorded as not
    placed, with the reason - not dropped."""
    _tracklet(flight["config"]["trex_npz_dir"], 12, [1, 7], [100.0, 100.0])
    _import(flight)
    root = flight["root"]

    report = track_store.accounting(root, "t")
    assert report["detections"] == 8
    assert report["resolved"] == 7 and report["failed"] == 1
    assert report["unaccounted"] == [] and report["both"] == []
    assert track_store.failure_summary(root, "t") == {"no_pose": 1}


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------

def _stored_boxes(root):
    rows = track_store.load_detections(root, "t")
    return [(row["frame"], row["x1"], row["y1"], row["x2"], row["y2"])
            for row in rows]


def test_boxes_made_on_the_frames_are_stored_as_they_are(flight):
    _import(flight)
    assert _stored_boxes(flight["root"])[:2] == [
        (0, 200.0, 250.0, 220.0, 270.0), (0, 300.0, 250.0, 320.0, 270.0)]


def test_boxes_made_on_the_video_are_stored_in_the_frames_pixels(flight):
    """A 640x512 video, 512x512 frames: the store held the video's pixels,
    and the classifier's crops and every overlay were taken beside the
    animal."""
    root = flight["root"]
    for name in os.listdir(os.path.join(root, "frames_t")):
        os.remove(os.path.join(root, "frames_t", name))
    _frames(root, 512, 512)

    _import(flight, trex_already_undistorted=False,
            thermal_calibration_data={"mtx": MATRIX, "dist": DISTORTION})

    undistorter = trex_tracklets.Undistorter(
        MATRIX, DISTORTION, (WIDTH, HEIGHT), (512, 512))
    expected = undistorter.box(200.0, 250.0, 220.0, 270.0)
    frame, *box = _stored_boxes(root)[0]
    assert frame == 0
    assert box == pytest.approx([round(value, 2) for value in expected])
    assert abs(box[0] - 200.0) > 30            # not the video's pixel


def test_the_positions_follow_from_the_stored_boxes(flight):
    """One projection for every box: placing the TRex boxes again with
    "Geo-Reference Detections" changes nothing."""
    _import(flight)
    root = flight["root"]
    before = pipeline_outputs.geo_tracks(root, "t")

    BambiProcessor().run_georeference(
        dict(flight["config"], georeference_camera="T"))

    assert pipeline_outputs.geo_tracks(root, "t") == before


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------

def test_a_tracklet_is_a_track(flight):
    _import(flight)
    root = flight["root"]
    conn = store.open_store(store.stage_path(root, store.TRACKS, "t"),
                            store.TRACKS, "t")
    try:
        runs = [dict(row) for row in conn.execute(
            "SELECT kind, is_active FROM track_runs")]
        sizes = [row["n"] for row in conn.execute(
            "SELECT COUNT(*) AS n FROM track_members GROUP BY track_id")]
    finally:
        conn.close()

    assert runs == [{"kind": "trex", "is_active": 1}]
    assert sizes == [N_FRAMES, N_FRAMES]


def test_the_tracks_are_recorded_beside_a_detector_run(flight):
    """Every detection of the camera was counted against the import's, so
    with anything else in the store no track was recorded at all."""
    root = flight["root"]
    detection_store.record_detections(root, "t", [
        {"frame": frame, "x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 30.0,
         "confidence": 0.5, "source_class": "0"} for frame in range(5)])

    log = _import(flight)

    assert not any("could not record TRex tracks" in line for line in log)
    tracks = pipeline_outputs.geo_tracks(root, "t")
    assert len(tracks) == 2
    for points in tracks.values():
        assert [point["frame"] for point in points] == [0, 1, 2]
        assert all(point["confidence"] >= 0.7 for point in points)
    assert detection_store.detection_counts(root, "t") == \
        {"detector": 5, "trex": 6}


def test_each_box_is_in_its_own_tracklet(flight):
    _import(flight)
    first, second = pipeline_outputs.geo_tracks(flight["root"], "t").values()
    # the tracklet further left stays further left, frame by frame
    assert all(_centre(a)[0] < _centre(b)[0] for a, b in zip(first, second))


def test_importing_again_replaces_the_tracklets(flight):
    _import(flight)
    _import(flight)
    root = flight["root"]

    assert detection_store.detection_counts(root, "t") == {"trex": 6}
    assert len(pipeline_outputs.geo_tracks(root, "t")) == 2
    assert len(analytics_source.load_animals(root, "t")[0]) == 2


def test_the_steps_are_done_and_none_is_out_of_date(flight):
    _import(flight)
    root = flight["root"]

    done = stages.completed_stages(root, "t")
    assert {"detection", "georeference", "tracking"} <= set(done)
    assert stages.stale_stages(root, "t") == []


# ---------------------------------------------------------------------------
# The legacy text files
# ---------------------------------------------------------------------------

LEGACY = ("detections_t/detections.txt", "georeferenced_t/georeferenced.txt",
          "tracks_t/tracks.csv", "tracks_pixel_t/tracks_pixel.csv",
          "tracks_pixel_t/tracks_pixel_undistorted.csv")


def test_the_text_files_are_written_while_asked_for(flight):
    _import(flight)
    root = flight["root"]
    for name in LEGACY:
        assert os.path.isfile(os.path.join(root, *name.split("/"))), name

    with open(os.path.join(root, "tracks_t", "tracks.csv")) as handle:
        rows = [line.split(",") for line in handle.read().splitlines()]
    assert len(rows) == 6
    assert {row[1] for row in rows} == {"3", "9"}      # the tracklets' ids
    stored = pipeline_outputs.geo_tracks(root, "t")
    first = next(iter(stored.values()))[0]
    assert float(rows[0][2]) == pytest.approx(first["x1"], abs=1e-5)


def test_the_text_files_are_not_written_when_switched_off(flight):
    """They were written whatever the switch said."""
    _import(flight, write_legacy_text_outputs=False)
    root = flight["root"]

    for name in LEGACY:
        assert not os.path.exists(os.path.join(root, *name.split("/"))), name
    assert len(pipeline_outputs.geo_tracks(root, "t")) == 2


# ---------------------------------------------------------------------------
# What cannot be imported says so
# ---------------------------------------------------------------------------

def test_tracklets_without_a_single_box_are_an_error(flight):
    folder = flight["config"]["trex_npz_dir"]
    for name in os.listdir(folder):
        os.remove(os.path.join(folder, name))
    nan = float("nan")
    _tracklet(folder, 1, [0, 1], [nan, nan])

    with pytest.raises(RuntimeError, match="no box to import"):
        _import(flight)
    assert detection_store.detection_counts(flight["root"], "t") == {}


def test_raw_boxes_need_the_video_size(flight):
    folder = flight["config"]["trex_npz_dir"]
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        data = {key: value for key, value in np.load(path).items()
                if key != "video_size"}
        np.savez(path, **data)

    with pytest.raises(ValueError, match="Raw video size unknown"):
        _import(flight, trex_already_undistorted=False,
                thermal_calibration_data={"mtx": MATRIX, "dist": DISTORTION})


# ---------------------------------------------------------------------------
# Out and back in
# ---------------------------------------------------------------------------

def test_exported_tracklets_can_be_imported_again(flight, tmp_path):
    """The export wrote arrays the import does not read."""
    from bambi_wildlife_detection.core import exporters

    _import(flight)
    root = flight["root"]
    before = sorted(
        [(p["frame"], round(p["x1"], 4), round(p["y1"], 4)) for p in points]
        for points in pipeline_outputs.geo_tracks(root, "t").values())

    exported = str(tmp_path / "exported")
    exporters.export_trex_npz(root, "t", exported)
    _import(flight, trex_npz_dir=exported)

    after = sorted(
        [(p["frame"], round(p["x1"], 4), round(p["y1"], 4)) for p in points]
        for points in pipeline_outputs.geo_tracks(root, "t").values())
    assert after == before
    assert detection_store.detection_counts(root, "t") == {"trex": 6}
