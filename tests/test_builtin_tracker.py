# -*- coding: utf-8 -*-
"""The built-in tracker and its settings.

"Tracker Mode" offered four modes and "Max Center Distance" a threshold, and
neither was read: the tracker matched by box overlap with the Hungarian
assignment whatever was selected. Abort was not passed to the trackers
either - the run finished, was written, and was then reported as cancelled.
"""
import json
import os

import numpy as np
import pytest

from bambi_wildlife_detection import bambi_processing
from bambi_wildlife_detection.bambi_processing import (
    BambiProcessor, CancelledException)
from bambi_wildlife_detection.core import (
    builtin_tracker as bt, detection_store, stages, store, track_store)


def box(x, y, size=2.0):
    return (x, y, x + size, y + size)


def ids(frames, **arguments):
    result = bt.track(frames, **arguments)
    return [result[frame] for frame in sorted(frames)]


# ---------------------------------------------------------------------------
# The tracker as it was: Hungarian on overlap, written out frame by frame
# ---------------------------------------------------------------------------

def _as_it_was(frames, classes, iou_threshold, max_age, class_aware):
    from scipy.optimize import linear_sum_assignment

    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        union = ((a[2] - a[0]) * (a[3] - a[1])
                 + (b[2] - b[0]) * (b[3] - b[1]) - inter)
        return inter / union if union > 0 else 0.0

    tracks, next_id, result = [], 1, {}
    for frame in sorted(frames):
        dets = frames[frame]
        det_classes = classes[frame]
        nd, nt = len(dets), len(tracks)
        matches, det_used, trk_used = [], set(), set()
        if nd and nt:
            overlap = np.zeros((nd, nt), dtype=np.float32)
            for di, d in enumerate(dets):
                for ti, t in enumerate(tracks):
                    if class_aware and det_classes[di] != t["cls"]:
                        continue
                    overlap[di, ti] = iou(d, t["box"])
            cost = 1.0 - overlap
            cost[overlap < iou_threshold] = 1e6
            for di, ti in zip(*linear_sum_assignment(cost)):
                if overlap[di, ti] >= iou_threshold and cost[di, ti] < 1e6:
                    matches.append((di, ti))
                    det_used.add(di)
                    trk_used.add(ti)
        frame_ids = [0] * nd
        for di, ti in matches:
            tracks[ti]["box"] = dets[di]
            tracks[ti]["age"] = 0
            frame_ids[di] = tracks[ti]["id"]
        for di in range(nd):
            if di in det_used:
                continue
            tracks.append({"id": next_id, "box": dets[di], "age": 0,
                           "cls": det_classes[di]})
            frame_ids[di] = next_id
            next_id += 1
        survivors = []
        for ti, t in enumerate(tracks):
            if ti < nt and ti not in trk_used:
                t["age"] += 1
            if max_age < 0 or t["age"] <= max_age:
                survivors.append(t)
        tracks = survivors
        result[frame] = frame_ids
    return result


def _scene(rng):
    frames, classes = {}, {}
    animals = int(rng.integers(1, 6))
    position = rng.uniform(0, 60, (animals, 2))
    species = rng.integers(0, 3, animals)
    for frame in range(int(rng.integers(3, 25))):
        if rng.random() < 0.15:
            continue                        # a frame without any box
        boxes, kinds = [], []
        for animal in range(animals):
            position[animal] += rng.normal(0, rng.choice([0.3, 2.5]), 2)
            if rng.random() < 0.2:
                continue                    # missed in this frame
            width, height = rng.uniform(1, 3, 2)
            boxes.append((position[animal, 0], position[animal, 1],
                          position[animal, 0] + width,
                          position[animal, 1] + height))
            kinds.append(int(species[animal]))
        if boxes:
            frames[frame], classes[frame] = boxes, kinds
    return frames, classes


@pytest.mark.parametrize("class_aware", [True, False])
@pytest.mark.parametrize("max_age", [-1, 0, 2])
def test_the_default_mode_tracks_as_before(class_aware, max_age):
    """HUNGARIAN is what always ran: choosing it must change nothing."""
    rng = np.random.default_rng(42)
    for _ in range(120):
        frames, classes = _scene(rng)
        if not frames:
            continue
        assert bt.track(frames, classes, mode="HUNGARIAN", iou_threshold=0.3,
                        max_age=max_age, class_aware=class_aware) == \
            _as_it_was(frames, classes, 0.3, max_age, class_aware)


def test_it_tracks_like_the_detection_framework():
    """The same matching as ``bambi.tracking.iou.track_boxes``, in every
    mode. Skipped where that package is not installed."""
    iou = pytest.importorskip("bambi.tracking.iou")

    rng = np.random.default_rng(7)
    for _ in range(60):
        frames, classes = _scene(rng)
        if not frames:
            continue
        flat_frames = [f for f in sorted(frames) for _ in frames[f]]
        flat_boxes = np.array([b for f in sorted(frames) for b in frames[f]])
        flat_classes = [c for f in sorted(frames) for c in classes[f]]
        for mode in bt.MODES:
            for distance in (0.5, 3.0):
                mine = bt.track(frames, classes, mode=mode,
                                max_center_distance=distance, max_age=2)
                theirs = iou.track_boxes(
                    flat_frames, flat_boxes, flat_classes, mode=mode.lower(),
                    max_age=2, max_center_distance=distance)
                assert [i for f in sorted(frames) for i in mine[f]] == \
                    list(theirs), mode


# ---------------------------------------------------------------------------
# What each mode does
# ---------------------------------------------------------------------------

def test_overlap_modes_do_not_join_boxes_that_do_not_overlap():
    frames = {0: [box(0, 0)], 1: [box(3, 0)]}       # one metre apart
    for mode in ("GREEDY", "HUNGARIAN"):
        assert ids(frames, mode=mode) == [[1], [2]]


def test_center_joins_boxes_that_do_not_overlap():
    """What the mode is for: a small animal whose boxes miss each other."""
    frames = {0: [box(0, 0)], 1: [box(3, 0)], 2: [box(6, 0)]}
    assert ids(frames, mode="CENTER", max_center_distance=3.5) == [
        [1], [1], [1]]


def test_the_center_distance_is_a_limit():
    frames = {0: [box(0, 0)], 1: [box(3, 0)]}
    assert ids(frames, mode="CENTER", max_center_distance=3.0) == [[1], [1]]
    assert ids(frames, mode="CENTER", max_center_distance=2.9) == [[1], [2]]


def test_center_takes_the_nearest_first():
    frames = {0: [box(0, 0), box(10, 0)],
              1: [box(9, 0), box(1, 0)]}
    assert ids(frames, mode="CENTER", max_center_distance=5.0) == [
        [1, 2], [2, 1]]


def test_center_ignores_the_overlap_threshold():
    frames = {0: [box(0, 0)], 1: [box(1.9, 0)]}     # barely touching
    assert ids(frames, mode="HUNGARIAN", iou_threshold=0.3) == [[1], [2]]
    assert ids(frames, mode="CENTER", iou_threshold=0.99,
               max_center_distance=2.0) == [[1], [1]]


def test_hungarian_center_falls_back_for_what_overlap_did_not_pair():
    frames = {0: [box(0, 0), box(20, 0)],
              1: [box(0.2, 0), box(23, 0)]}         # second one jumped
    assert ids(frames, mode="HUNGARIAN") == [[1, 2], [1, 3]]
    assert ids(frames, mode="HUNGARIAN_CENTER",
               max_center_distance=4.0) == [[1, 2], [1, 2]]
    assert ids(frames, mode="HUNGARIAN_CENTER",
               max_center_distance=1.0) == [[1, 2], [1, 3]]


def test_hungarian_center_keeps_overlap_first():
    """A box overlapping one track and nearer to the centre of another goes
    with the one it overlaps."""
    frames = {0: [(0.0, 0.0, 10.0, 10.0), (11.0, 4.0, 12.0, 5.0)],
              1: [(4.0, 0.0, 14.0, 10.0)]}
    assert ids(frames, mode="HUNGARIAN_CENTER", iou_threshold=0.3,
               max_center_distance=50.0) == [[1, 2], [1]]


def test_greedy_takes_the_best_pair_and_hungarian_the_best_whole():
    """Two boxes, two tracks: the first box overlaps both tracks, the second
    only the first track. Greedy gives the first box its best track and the
    second box is left over; the assignment pairs both."""
    frames = {0: [(0.0, 0.0, 10.0, 10.0), (6.0, 0.0, 16.0, 10.0)],
              1: [(1.0, 0.0, 11.0, 10.0), (-4.0, 0.0, 6.0, 10.0)]}
    greedy = ids(frames, mode="GREEDY", iou_threshold=0.3)
    hungarian = ids(frames, mode="HUNGARIAN", iou_threshold=0.3)
    assert greedy == [[1, 2], [1, 3]]
    assert hungarian == [[1, 2], [2, 1]]


@pytest.mark.parametrize("mode", bt.MODES)
def test_class_aware_keeps_species_apart(mode):
    frames = {0: [box(0, 0)], 1: [box(0.1, 0)]}
    classes = {0: [1], 1: [2]}
    assert bt.track(frames, classes, mode=mode, class_aware=True,
                    max_center_distance=5.0) == {0: [1], 1: [2]}
    assert bt.track(frames, classes, mode=mode, class_aware=False,
                    max_center_distance=5.0) == {0: [1], 1: [1]}


@pytest.mark.parametrize("mode", bt.MODES)
def test_max_age_ends_a_track_that_goes_unmatched(mode):
    frames = {0: [box(0, 0)], 1: [box(50, 50)], 2: [box(50, 50)],
              3: [box(0, 0), box(50, 50)]}
    kwargs = dict(mode=mode, max_center_distance=1.0)
    assert ids(frames, max_age=-1, **kwargs)[3] == [1, 2]
    assert ids(frames, max_age=2, **kwargs)[3] == [1, 2]
    assert ids(frames, max_age=1, **kwargs)[3] == [3, 2]


def test_frames_without_any_box_do_not_age_a_track():
    """The thermal camera's shutter: nothing is seen, so nothing is lost."""
    frames = {0: [box(0, 0)], 40: [box(0.1, 0)]}
    assert ids(frames, max_age=0) == [[1], [1]]


def test_every_box_gets_an_id_in_its_place():
    frames = {5: [box(0, 0), box(30, 0), box(60, 0)], 2: [box(30, 0)]}
    result = bt.track(frames)
    assert result == {2: [1], 5: [2, 1, 3]}


def test_no_boxes_at_all():
    assert bt.track({}) == {}
    assert bt.track({3: []}) == {3: []}


def test_an_unknown_mode_is_refused():
    """Not quietly run as the default - that is how two modes went missing."""
    with pytest.raises(ValueError, match="Unknown tracker mode"):
        bt.track({0: [box(0, 0)]}, mode="NEAREST")
    assert bt.normalise_mode("hungarian center") == "HUNGARIAN_CENTER"
    assert bt.normalise_mode(None) == "HUNGARIAN"


def test_abort_stops_between_two_frames():
    asked = []

    def cancel():
        asked.append(1)
        return len(asked) > 2

    with pytest.raises(bt.Cancelled):
        bt.track({f: [box(0, 0)] for f in range(10)}, cancel_check=cancel)
    assert len(asked) == 3


def test_progress_runs_to_the_end():
    seen = []
    bt.track({f: [box(0, 0)] for f in range(4)}, progress=seen.append)
    assert seen == [0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# The tracking step
# ---------------------------------------------------------------------------

@pytest.fixture
def processor():
    return BambiProcessor()


@pytest.fixture
def flight(tmp_path):
    """One animal on frames 0, 1 and 3, three metres further on each frame:
    its boxes (two metres wide) never overlap."""
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    detection_store.record_detections(root, "t", [
        {"frame": frame, "x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 20.0,
         "confidence": 0.9, "source_class": "0"} for frame in (0, 1, 3)])
    rows = track_store.load_detections(root, "t")
    track_store.record_georeference(root, "t", [
        {"detection_id": row["detection_id"],
         "gx1": 3.0 * row["frame"], "gy1": 0.0, "gz1": 400.0,
         "gx2": 3.0 * row["frame"] + 2.0, "gy2": 2.0, "gz2": 400.0}
        for row in rows])
    return root


def _config(root, **settings):
    config = {"target_folder": root, "tracking_camera": "T",
              "write_legacy_text_outputs": True}
    config.update(settings)
    return config


def _tracks(root):
    rows = track_store.load_pixel_tracks(root, "t")
    by_track = {}
    for row in rows:
        by_track.setdefault(row["track_id"], []).append(row["frame"])
    return sorted(sorted(frames) for frames in by_track.values())


def test_the_step_tracks_by_overlap_by_default(processor, flight):
    processor._run_builtin_tracking(_config(flight))
    assert _tracks(flight) == [[0], [1], [3]]


def test_the_step_honours_the_mode_and_the_distance(processor, flight):
    log = []
    processor._run_builtin_tracking(
        _config(flight, tracker_mode="CENTER", max_center_distance=7.0),
        log_fn=log.append)

    assert _tracks(flight) == [[0, 1, 3]]
    assert any("mode: CENTER (centres within 7 m)" in line for line in log)

    processor._run_builtin_tracking(
        _config(flight, tracker_mode="CENTER", max_center_distance=4.0))
    assert _tracks(flight) == [[0, 1], [3]]

    processor._run_builtin_tracking(
        _config(flight, tracker_mode="HUNGARIAN_CENTER",
                max_center_distance=7.0))
    assert _tracks(flight) == [[0, 1, 3]]


def test_the_mode_is_recorded_with_the_run(processor, flight):
    processor._run_builtin_tracking(_config(flight, tracker_mode="GREEDY"))
    assert track_store.active_run(flight, "t")["tracker"] == "GREEDY"


def test_an_unknown_mode_stops_the_step(processor, flight):
    with pytest.raises(ValueError, match="Unknown tracker mode"):
        processor._run_builtin_tracking(_config(flight, tracker_mode="X"))
    assert track_store.active_run(flight, "t") is None


def test_an_interpolated_box_is_not_recorded_as_a_detection(processor, flight):
    """Frame 2 is filled in between 1 and 3. It is no detection and has no
    id; it was recorded as member "-1" of the track, a row that refers to
    nothing."""
    log = []
    processor._run_builtin_tracking(
        _config(flight, tracker_mode="CENTER", max_center_distance=7.0,
                interpolate=True), log_fn=log.append)

    conn = store.open_store(
        store.stage_path(flight, store.TRACKS, "t"), store.TRACKS, "t")
    try:
        members = [row["detection_id"] for row in conn.execute(
            "SELECT detection_id FROM track_members")]
    finally:
        conn.close()
    assert len(members) == 3 and min(members) > 0
    assert track_store.track_orphans(flight, "t") == []
    assert not any("no longer exists" in line for line in log)
    # the interpolated box is still in the text output
    with open(os.path.join(flight, "tracks_t", "tracks.csv")) as fh:
        assert [line.split(",")[0] for line in fh] == [
            "00000000", "00000001", "00000002", "00000003"]


def test_abort_leaves_the_previous_tracks(processor, flight):
    processor._run_builtin_tracking(_config(flight))
    before = track_store.active_run(flight, "t")
    with open(os.path.join(flight, "tracks_t", "tracks.csv")) as fh:
        text = fh.read()

    with pytest.raises(CancelledException):
        processor._run_builtin_tracking(
            _config(flight, tracker_mode="CENTER", max_center_distance=7.0),
            cancel_check=lambda: True)

    assert track_store.active_run(flight, "t") == before
    assert _tracks(flight) == [[0], [1], [3]]
    with open(os.path.join(flight, "tracks_t", "tracks.csv")) as fh:
        assert fh.read() == text


def test_abort_reaches_the_built_in_tracker_through_the_step(
        processor, flight, monkeypatch):
    monkeypatch.setattr(BambiProcessor, "_get_default_model_dir",
                        lambda self: flight)
    with pytest.raises(CancelledException):
        processor.run_tracking(_config(flight, tracker_id="builtin"),
                               cancel_check=lambda: True)
    assert track_store.active_run(flight, "t") is None
    assert "tracking" not in stages.states(flight, "t")


# ---------------------------------------------------------------------------
# The other trackers
# ---------------------------------------------------------------------------

class _Backend:
    value = "boxmot"


class _Tracker:
    """Stands in for a BoxMOT tracker: every box is its own track."""

    def __init__(self):
        self.frames = 0

    def update(self, dets, img):
        self.frames += 1
        return np.array([[d[0], d[1], d[2], d[3], index + 1, d[4], d[5], index]
                         for index, d in enumerate(dets)])


class _Manager:
    def __init__(self):
        self.tracker = _Tracker()

    def create_tracker(self, **_arguments):
        return self.tracker, _Backend()


@pytest.fixture
def advanced(flight, monkeypatch, processor):
    with open(os.path.join(flight, "poses_t.json"), "w") as fh:
        json.dump({"images": [{"imagefile": f"{i}.jpg"} for i in range(4)]},
                  fh)
    monkeypatch.setattr(BambiProcessor, "run_georeference_tracks",
                        lambda self, *args, **kwargs: None)
    manager = _Manager()

    def run(cancel_check=None):
        processor._run_advanced_tracking(
            _config(flight, tracker_id="bytetrack"), manager,
            cancel_check=cancel_check)

    return run, manager.tracker


def test_abort_stops_the_other_trackers_before_anything_is_written(
        advanced, flight):
    run, tracker = advanced
    asked = []

    def cancel():
        asked.append(1)
        return len(asked) > 1

    with pytest.raises(CancelledException):
        run(cancel)

    assert tracker.frames == 1                       # stopped after one
    assert track_store.active_run(flight, "t") is None
    assert not os.path.exists(
        os.path.join(flight, "tracks_t", "tracks_pixel.csv"))


def test_the_other_trackers_are_on_record_when_they_finish(advanced, flight):
    """Without a record of the step, nothing that depends on the tracks was
    ever marked out of date after this tracker ran."""
    run, _tracker = advanced
    stages.set_state(flight, "track_inventory", "t", stages.COMPLETE)

    run()

    states = stages.states(flight, "t")
    assert states["tracking"]["state"] == stages.COMPLETE
    assert states["tracking"]["row_count"] == 1
    assert states["track_inventory"]["state"] == stages.STALE
    assert bambi_processing._is_detection_id(-1) is False
