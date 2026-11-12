# -*- coding: utf-8 -*-
"""Custom implementations driving the real steps: detection, tracking and
classification (frame / track level, embeddings / crops). No torch needed -
the custom classes here are plain Python (core.custom_models, 9.0.0)."""

import json
import os
import textwrap

import numpy as np
import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import (
    classification_store, embedding_files, store, track_store,
)

cv2 = pytest.importorskip("cv2")

DIM = 4

#: detection_id -> (track, occlusion, species). One detection per frame.
PLAN = {
    1: (1, 0, 1), 2: (1, 0, 1), 3: (1, 0, 1), 4: (1, 0, 0),
    5: (2, 0, 0), 6: (2, 0, 0), 7: (2, 0, 0), 8: (2, 0, 1),
}


def _impl(root, name, source):
    path = os.path.join(root, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(textwrap.dedent(source))
    return path


def _frames(root, suffix, n, bright=()):
    """Frame images: bright ones are 'species 1' for the crop classifier."""
    folder = os.path.join(root, f"frames_{suffix}")
    os.makedirs(folder, exist_ok=True)
    for i in range(n):
        image = np.full((40, 60, 3), 220 if i in bright else 20, np.uint8)
        cv2.imwrite(os.path.join(folder, f"f{i:03d}.png"), image)
    with open(os.path.join(root, f"poses_{suffix}.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"images": [{"imagefile": f"f{i:03d}.png"}
                              for i in range(n)]}, fh)


@pytest.fixture
def flight(tmp_path):
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    conn = store.open_store(
        store.stage_path(root, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    with store.transaction(conn):
        for detection_id in sorted(PLAN):
            conn.execute(
                "INSERT INTO detections (frame, x1, y1, x2, y2, confidence, "
                "species_id, source_id) VALUES (?, 10, 10, 30, 30, 0.9, 0, 1)",
                (detection_id - 1,))
    conn.close()
    track_store.record_tracks(root, "t", [
        {"track_id": spec[0], "detection_id": detection_id}
        for detection_id, spec in sorted(PLAN.items())])
    _frames(root, "t", len(PLAN),
            bright={d - 1 for d, spec in PLAN.items() if spec[2] == 1})
    return root


def _embed(root):
    run_id = classification_store.start_embedding_run(
        root, "t", backbone="fake", dim=DIM, crop_size=32, padding=0.1,
        projection="non_geo", folder="embeddings_t/non_geo")
    for detection_id, (_track, occ, species) in PLAN.items():
        frame = detection_id - 1
        embedding_files.write_frame(
            embedding_files.frame_path(root, "t", "non_geo", frame,
                                       f"f{frame:03d}.png"),
            {detection_id: np.array([occ, species, 0.0, 0.0], np.float32)})
    classification_store.record_embedded(root, "t", run_id, list(PLAN))


def _classify(root, models, **overrides):
    config = {
        "target_folder": root, "classification_models": models,
        "classification_backbone_dim": DIM, "classification_quorum": 0.5,
        "classification_min_frames": 1,
        "classification_frame_selection": "all",
        "classification_unmatched": "skip", "hf_token": "",
    }
    config.update(overrides)
    logs = []
    BambiProcessor().run_classification(config, log_fn=logs.append)
    return logs


def _species(root):
    return {row["track_id"]: row["label"] for row in
            classification_store.track_predictions(root, "t", "species")}


EMBEDDING_FRAMES = """
    class OnEmbeddings:
        def setup(self, context):
            return {"input": "embeddings", "class_names": ["roe", "red"]}
        def run(self, batch, items):
            assert batch.shape[1] == 4 and len(items) == len(batch)
            assert all("detection_id" in i and "track_id" in i for i in items)
            return [(int(row[1]), 0.9) for row in batch]
"""

CROP_FRAMES = """
    class OnCrops:
        def setup(self, context):
            return {"input": "crops", "class_names": ["roe", "red"],
                    "crop_size": 32, "cameras": ["thermal"]}
        def run(self, batch, items):
            assert all(c.shape == (32, 32, 3) for c in batch)
            return [(1 if c.mean() > 100 else 0, 0.8) for c in batch]
"""

CROP_TRACKS = """
    class PerTrack:
        def setup(self, context):
            return {"input": "crops", "level": "track",
                    "class_names": ["roe", "red"], "crop_size": 16}
        def run_track(self, batch, items):
            if items[0]["track_id"] == 2:
                return None                       # no call for this animal
            bright = sum(1 for c in batch if c.mean() > 100)
            return (1 if bright * 2 > len(batch) else 0, 0.7)
"""


def _spec(path, **extra):
    return dict({"modality": "thermal", "model": "implementation",
                 "path": path}, **extra)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_a_frame_level_classifier_on_embeddings_is_voted(flight):
    _embed(flight)
    path = _impl(flight, "emb.py", EMBEDDING_FRAMES)
    logs = _classify(flight, {"species": _spec(path)})

    assert _species(flight) == {1: "red", 2: "roe"}
    frames = classification_store.frame_predictions(flight, "t", "species")
    assert len(frames) == 8
    assert frames[0]["model"].startswith("custom:emb.py:OnEmbeddings@")
    assert any("custom species classifier" in line for line in logs)


def test_a_crop_classifier_needs_no_embeddings(flight):
    path = _impl(flight, "crops.py", CROP_FRAMES)
    _classify(flight, {"species": _spec(path)})
    assert _species(flight) == {1: "red", 2: "roe"}


def test_an_embedding_classifier_without_embeddings_says_so(flight):
    path = _impl(flight, "emb.py", EMBEDDING_FRAMES)
    with pytest.raises(ValueError, match="Compute DINOv3"):
        _classify(flight, {"species": _spec(path)})


def test_a_track_level_classifier_makes_the_call_itself(flight):
    path = _impl(flight, "tracks.py", CROP_TRACKS)
    _classify(flight, {"species": _spec(path)})

    rows = {row["track_id"]: row for row in
            classification_store.track_predictions(flight, "t", "species")}
    assert set(rows) == {1}                 # track 2 was left uncalled
    assert rows[1]["label"] == "red"
    assert rows[1]["fraction"] == pytest.approx(0.7)
    # no per-frame calls: the classifier never made any
    assert classification_store.frame_predictions(
        flight, "t", "species") == []


def test_occlusion_cannot_be_decided_per_track(flight):
    path = _impl(flight, "tracks.py", CROP_TRACKS)
    with pytest.raises(ValueError, match="per-frame"):
        _classify(flight, {"occlusion": _spec(path)})


def test_a_classifier_is_refused_on_a_camera_it_does_not_support(flight):
    path = _impl(flight, "crops.py", CROP_FRAMES)
    with pytest.raises(Exception, match="does not support"):
        _classify(flight, {"species": _spec(path, modality="matched")})


def test_frozen_frames_give_no_crop(flight):
    poses = os.path.join(flight, "poses_t.json")
    with open(poses, encoding="utf-8") as fh:
        data = json.load(fh)
    for i in (0, 1, 2):                      # track 1's bright frames
        data["images"][i]["duplicated"] = True
    with open(poses, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    os.utime(poses, (5, 5))

    path = _impl(flight, "crops.py", CROP_FRAMES)
    _classify(flight, {"species": _spec(path)})
    assert _species(flight)[1] == "roe"      # only the dark frame 3 voted


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

DETECTOR = """
    class Det:
        def setup(self, context):
            self.margin = context.options["margin"]
            return {"class_names": ["roe deer", "boar"]}
        def run(self, image, frame):
            if image.mean() < 100:
                return []
            h, w = image.shape[:2]
            m = self.margin
            return [(m, m, w - m, h - m, 0.9, 0), (0, 0, 4, 4, 0.2, 1)]
"""


def test_a_custom_detector_fills_the_detection_store(tmp_path):
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    _frames(root, "t", 4, bright={1, 3})
    path = _impl(root, "det.py", DETECTOR)
    logs = []
    BambiProcessor().run_detection({
        "target_folder": root, "detection_camera": "T",
        "min_confidence": 0.5,
        "custom_implementations": {"detector_t": {
            "source": "implementation", "path": path,
            "options": '{"margin": 5}'}},
    }, log_fn=logs.append)

    rows = track_store.load_detections(root, "t")
    assert [r["frame"] for r in rows] == [1, 3]
    assert rows[0]["source_class"] == "roe_deer"
    assert (rows[0]["x1"], rows[0]["x2"]) == (5.0, 55.0)
    assert any("custom detector custom:det.py:Det@" in line for line in logs)
    assert not any("mismatch" in line for line in logs)


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

TRACKER = """
    class Trk:
        def setup(self, context):
            return {"space": "pixel"}
        def run(self, detections, frames):
            assert len(frames) == 8 and frames[0].path.endswith("f000.png")
            return {d["detection_id"]: (7 if d["frame"] < 4 else 9)
                    for d in detections if d["frame"] != 7}
"""


def test_a_custom_tracker_records_a_track_run(flight):
    path = _impl(flight, "trk.py", TRACKER)
    logs = []
    BambiProcessor().run_tracking({
        "target_folder": flight, "tracking_camera": "T",
        "custom_implementations": {"tracker": {
            "source": "implementation", "path": path}},
    }, log_fn=logs.append)

    rows = track_store.load_pixel_tracks(flight, "t")
    by_track = {}
    for row in rows:
        by_track.setdefault(row["track_id"], []).append(row["frame"])
    assert sorted(len(v) for v in by_track.values()) == [3, 4]
    assert track_store.active_run(flight, "t")["kind"] == "custom"
    assert any("1 left untracked" in line for line in logs)


def test_a_geo_tracker_needs_geo_referenced_detections(flight):
    path = _impl(flight, "trk.py", TRACKER.replace('"pixel"', '"geo"'))
    with pytest.raises(RuntimeError, match="Geo-Reference"):
        BambiProcessor().run_tracking({
            "target_folder": flight, "tracking_camera": "T",
            "custom_implementations": {"tracker": {
                "source": "implementation", "path": path}},
        })


# ---------------------------------------------------------------------------
# Tracker interpolation
# ---------------------------------------------------------------------------

GAPPY = """
    class Gappy:
        def setup(self, context):
            return {"space": "pixel", "interpolate": %s}
        def run(self, detections, frames):
            # one track over frames 0, 1 and 5: frames 2-4 are a gap
            return {d["detection_id"]: 1 for d in detections
                    if d["frame"] in (0, 1, 5)}
"""


def _pixel_file(flight):
    path = os.path.join(flight, "tracks_t", "tracks_pixel.csv")
    with open(path, encoding="utf-8") as fh:
        return [line.strip().split(",") for line in fh
                if not line.startswith("#")]


def _track(flight, source, **spec):
    path = _impl(flight, "gappy.py", source)
    logs = []
    BambiProcessor().run_tracking({
        "target_folder": flight, "tracking_camera": "T",
        "custom_implementations": {"tracker": dict(
            {"source": "implementation", "path": path}, **spec)},
    }, log_fn=logs.append)
    return logs


def test_gaps_are_interpolated_in_the_track_files(flight):
    logs = _track(flight, GAPPY % "True")
    rows = _pixel_file(flight)
    assert [int(r[0]) for r in rows] == [0, 1, 2, 3, 4, 5]
    assert [int(r[8]) for r in rows] == [0, 0, 1, 1, 1, 0]
    # the store holds the real detections only
    assert len(track_store.load_pixel_tracks(flight, "t")) == 3
    assert any("3 gap frame(s) interpolated" in line for line in logs)


def test_the_configuration_can_switch_interpolation_off(flight):
    _track(flight, GAPPY % "True", interpolate=False)
    assert [int(r[0]) for r in _pixel_file(flight)] == [0, 1, 5]


def test_a_tracker_can_declare_interpolation_not_applicable(flight):
    logs = _track(flight, GAPPY % "False", interpolate=True)
    assert [int(r[0]) for r in _pixel_file(flight)] == [0, 1, 5]
    assert any("not applicable" in line for line in logs)


# ---------------------------------------------------------------------------
# Crops from the orthorectified imagery
# ---------------------------------------------------------------------------

def test_crops_follow_the_projection(monkeypatch):
    from bambi_wildlife_detection.core import classification

    frame = np.zeros((40, 60, 3), np.uint8)
    raster = np.full((100, 100, 3), 200, np.uint8)
    boxes = {7: (0, (10, 10, 30, 30))}

    perspective = classification.CropResolver(
        "t", boxes, {}, {}, lambda m, f: frame)
    assert perspective.resolve(7, "thermal", size=16).mean() < 50

    seen = []

    def to_pixel(transform, world):
        seen.append((transform, world))
        return (20, 20, 60, 60)

    ortho = classification.CropResolver(
        "t", boxes, {}, {}, lambda m, f, w: (raster, "affine"),
        world_boxes={"t": {7: (1.0, 2.0, 3.0, 4.0)}}, to_pixel=to_pixel)
    crop = ortho.resolve(7, "thermal", size=16)
    assert crop.shape == (16, 16, 3) and crop.mean() > 150
    assert seen == [("affine", (1.0, 2.0, 3.0, 4.0))]

    # no world coordinates for the detection, or no raster: no crop
    missing = classification.CropResolver(
        "t", boxes, {}, {}, lambda m, f, w: (raster, "affine"),
        world_boxes={"t": {}}, to_pixel=to_pixel)
    assert missing.resolve(7, "thermal", size=16) is None
    gone = classification.CropResolver(
        "t", boxes, {}, {}, lambda m, f, w: (None, None),
        world_boxes={"t": {7: (1, 2, 3, 4)}}, to_pixel=to_pixel)
    assert gone.resolve(7, "thermal", size=16) is None


def test_the_context_tells_a_classifier_the_projection(flight):
    path = _impl(flight, "proj.py", """
        class P:
            def setup(self, context):
                context.log("projection=" + context.projection)
                return {"input": "crops", "class_names": ["a", "b"],
                        "crop_size": 16}
            def run(self, batch, items):
                return [(0, 1.0) for _ in items]
    """)
    logs = _classify(flight, {"species": _spec(path)})
    assert "projection=perspective" in logs


# ---------------------------------------------------------------------------
# Camera extrinsics reach every kind
# ---------------------------------------------------------------------------

def _locate(root, suffix="t"):
    """Give the frames a position: frame i at x = 10 * i, 60 m up."""
    path = os.path.join(root, f"poses_{suffix}.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    for i, image in enumerate(data["images"]):
        image.update(location=[10.0 * i, 5.0, 60.0],
                     rotation=[0.0, 0.0, 90.0], fovy=50.0)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    os.utime(path, (9, 9))


POSE_DETECTOR = """
    class Det:
        def setup(self, context):
            self.log = context.log
        def run(self, image, frame):
            p = frame.pose
            self.log("pose %d %.1f %.1f %s" % (
                frame.index, p.position[0], p.fovy, p.rotation[2]))
            return []
"""


def test_a_detector_is_told_where_the_camera_was(tmp_path):
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    _frames(root, "t", 3)
    _locate(root)
    logs = []
    BambiProcessor().run_detection({
        "target_folder": root, "detection_camera": "T",
        "custom_implementations": {"detector_t": {
            "source": "implementation",
            "path": _impl(root, "det.py", POSE_DETECTOR)}},
    }, log_fn=logs.append)
    assert "pose 2 20.0 50.0 90.0" in logs


POSE_TRACKER = """
    class Trk:
        def setup(self, context):
            return {"space": "pixel", "interpolate": False}
        def run(self, detections, frames):
            assert [f.pose.position[0] for f in frames[:3]] == [0.0, 10.0, 20.0]
            # geo-fence: only what was seen from x < 35 m is tracked
            return {d["detection_id"]: 1 for d in detections
                    if frames[d["frame"]].pose.position[0] < 35.0}
"""


def test_a_tracker_can_use_the_camera_positions(flight):
    _locate(flight)
    BambiProcessor().run_tracking({
        "target_folder": flight, "tracking_camera": "T",
        "custom_implementations": {"tracker": {
            "source": "implementation",
            "path": _impl(flight, "trk.py", POSE_TRACKER)}},
    })
    rows = track_store.load_pixel_tracks(flight, "t")
    assert sorted(r["frame"] for r in rows) == [0, 1, 2, 3]


POSE_CLASSIFIER = """
    class Fenced:
        def setup(self, context):
            return {"input": "crops", "class_names": ["outside", "inside"],
                    "crop_size": 16}
        def run(self, batch, items):
            assert all(i["box"] == (10.0, 10.0, 30.0, 30.0) for i in items)
            assert all(i["world"] is None for i in items)   # not geo-referenced
            # geo-fence on the camera position of each row's frame
            return [(1 if i["pose"].position[0] < 35.0 else 0, 1.0)
                    for i in items]
"""


def test_a_classifier_can_geo_fence_on_the_camera(flight):
    _locate(flight)
    _classify(flight, {"species": _spec(
        _impl(flight, "fence.py", POSE_CLASSIFIER))})
    # track 1 is frames 0-3 (x 0-30 m), track 2 frames 4-7 (x 40-70 m)
    assert _species(flight) == {1: "inside", 2: "outside"}


# ---------------------------------------------------------------------------
# Progress and batching
# ---------------------------------------------------------------------------

PROGRESS_TRACKER = """
    class Trk:
        def setup(self, context):
            self.progress = context.progress
            return {"space": "pixel", "interpolate": False}
        def run(self, detections, frames):
            for i in range(len(frames)):
                self.progress(i / len(frames))     # the total is known here
            self.progress("not a number")          # must not break the run
            return {d["detection_id"]: 1 for d in detections}
"""


def test_a_tracker_moves_the_progress_bar_itself(flight):
    seen = []
    BambiProcessor().run_tracking({
        "target_folder": flight, "tracking_camera": "T",
        "custom_implementations": {"tracker": {
            "source": "implementation",
            "path": _impl(flight, "trk.py", PROGRESS_TRACKER)}},
    }, progress_fn=seen.append)
    between = [p for p in seen if 10 < p < 85]
    assert len(set(between)) >= 5                  # several steps, not a jump
    assert seen == sorted(seen)
    assert seen[0] == 10 and 85 in seen


def test_detection_progress_counts_skipped_frames_too(tmp_path):
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    _frames(root, "t", 40)
    for i in range(5, 40):                         # most frames have no image
        os.remove(os.path.join(root, "frames_t", f"f{i:03d}.png"))
    seen = []
    BambiProcessor().run_detection({
        "target_folder": root, "detection_camera": "T",
        "custom_implementations": {"detector_t": {
            "source": "implementation",
            "path": _impl(root, "det.py", POSE_DETECTOR)}},
    }, progress_fn=seen.append)
    assert [p for p in seen if p < 100] == [0, 25, 50, 75]


BATCH_CLASSIFIER = """
    SIZES = []
    class Batched:
        def setup(self, context):
            return {"input": "%s", "class_names": ["clear", "occluded"],
                    "crop_size": 16}
        def run(self, batch, items):
            SIZES.append(len(items))
            return [(0, 1.0) for _ in items]
"""


@pytest.mark.parametrize("kind, batch", [("crops", 3), ("embeddings", 5)])
def test_a_stage_is_handed_over_in_batches(flight, monkeypatch, kind, batch):
    """Memory must not grow with the flight - for crops and embeddings."""
    if kind == "embeddings":
        _embed(flight)
    monkeypatch.setattr(BambiProcessor, "_HEAD_BATCH", {kind: batch})
    path = _impl(flight, "batched.py", BATCH_CLASSIFIER % kind)
    seen = []
    config = {
        "target_folder": flight,
        "classification_models": {"occlusion": _spec(path)},
        "classification_backbone_dim": DIM,
        "classification_frame_selection": "all",
        "classification_unmatched": "skip", "hf_token": "",
    }
    BambiProcessor().run_classification(config, progress_fn=seen.append)

    from bambi_wildlife_detection.core import custom_models
    sizes = custom_models.load_module(path).SIZES   # a fresh import: empty
    assert sizes == []
    assert len(classification_store.frame_predictions(
        flight, "t", "occlusion")) == 8
    # 8 detections in batches of 3 / 5 -> the bar moved inside the stage
    inside = sorted({p for p in seen if 5 < p <= 35})
    assert len(inside) >= 2


def test_vectors_are_read_on_demand(flight):
    _embed(flight)
    wanted = [{"detection_id": d, "frame": d - 1,
               "imagefile": f"f{d - 1:03d}.png"} for d in PLAN]
    lazy = embedding_files.LazyVectors(flight, "t", "non_geo", wanted,
                                       cache_frames=2)
    assert bool(lazy) is True
    assert lazy._cache == {}                        # nothing read yet
    assert lazy.get(3)[1] == 1.0
    assert lazy.get(5)[1] == 0.0
    assert lazy.get(7) is not None
    assert len(lazy._cache) == 2                    # only the last two frames
    assert lazy.get(999) is None and lazy.get(None) is None
    full = embedding_files.read_vectors(flight, "t", "non_geo", wanted)
    assert all((lazy.get(d) == full[d]).all() for d in PLAN)
    assert not embedding_files.LazyVectors(flight, "w", "non_geo", wanted)


# ---------------------------------------------------------------------------
# Online trackers: the loop is the plugin's
# ---------------------------------------------------------------------------

ONLINE_TRACKER = """
    class Online:
        def setup(self, context):
            self.log = context.log
            self.frames = 0
            return {"space": "pixel", "interpolate": False, "images": %s}
        def update(self, frame, detections):
            self.frames += 1
            if frame.index == 0:
                self.log("image=" + str(None if frame.image is None
                                        else frame.image.shape))
            # frame 2's animal joins track 1, the rest track 2
            return {d["detection_id"]: (1 if frame.index < 3 else 2)
                    for d in detections}
        def finish(self):
            self.log("frames=%%d" %% self.frames)
            return %s
"""


def _online(flight, images="False", finish="None", **kwargs):
    logs, seen = [], []
    BambiProcessor().run_tracking({
        "target_folder": flight, "tracking_camera": "T",
        "custom_implementations": {"tracker": {
            "source": "implementation",
            "path": _impl(flight, "online.py",
                          ONLINE_TRACKER % (images, finish))}},
    }, progress_fn=seen.append, log_fn=logs.append, **kwargs)
    return logs, seen


def _tracks_of(flight):
    by_track = {}
    for row in track_store.load_pixel_tracks(flight, "t"):
        by_track.setdefault(row["track_id"], []).append(row["frame"])
    return {t: sorted(f) for t, f in by_track.items()}


def test_an_online_tracker_is_called_once_per_frame(flight):
    logs, seen = _online(flight)
    assert "frames=8" in logs
    assert "image=None" in logs                    # not asked for
    assert sorted(len(f) for f in _tracks_of(flight).values()) == [3, 5]
    assert any("frame by frame" in line for line in logs)
    # the plugin knows frame i of N, so the bar moves without the tracker
    assert seen[0] == 10 and 85 in seen and seen == sorted(seen)


def test_an_online_tracker_can_ask_for_the_images(flight):
    logs, _seen = _online(flight, images="True")
    assert "image=(40, 60, 3)" in logs


def test_finish_can_revise_and_untrack(flight):
    # detection 1 (frame 0) is taken out, detection 8 moves to track 1
    _online(flight, finish="{1: None, 8: 1}")
    tracks = _tracks_of(flight)
    assert 0 not in [f for frames in tracks.values() for f in frames]
    assert 7 in next(f for f in tracks.values() if 1 in f)


def test_an_online_tracker_can_be_cancelled_between_frames(flight):
    from bambi_wildlife_detection.bambi_processing import CancelledException
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) > 3

    with pytest.raises(CancelledException):
        _online(flight, cancel_check=cancel)
