# -*- coding: utf-8 -*-
"""The C2 embedding step, end to end against a fake backbone.

The behaviour worth defending is resumption: the vectors are the expensive
part of classification, so a second run must embed only what is missing, and
an interrupted one must pick up exactly where it stopped.
"""
import json
import os

import numpy as np
import pytest

from bambi_wildlife_detection.bambi_processing import (
    BambiProcessor, CancelledException,
)
from bambi_wildlife_detection.core import (
    classification, classification_store, embedding_files, store, track_store,
)

DIM = 16


class FakeBackbone:
    """Counts crops so the tests can prove what was recomputed."""

    def __init__(self, *_args, **kwargs):
        self.model_id = kwargs.get("model_id") or "fake/backbone"
        self.dim = DIM
        self.embedded = 0
        self.loaded = 0
        FakeBackbone.last = self

    def load(self):
        self.loaded += 1

    def embed(self, crops):
        self.embedded += len(crops)
        return np.tile(
            np.arange(DIM, dtype=np.float32), (len(crops), 1))


@pytest.fixture
def flight(tmp_path, monkeypatch):
    """A flight with two thermal frames, three detections and two tracks."""
    import cv2

    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()

    frames_folder = os.path.join(root, "frames_t")
    os.makedirs(frames_folder)
    images = []
    for frame in range(2):
        name = f"frame_{frame:06d}.jpg"
        cv2.imwrite(os.path.join(frames_folder, name),
                    np.full((120, 160, 3), 100, dtype=np.uint8))
        images.append({"imagefile": name, "timestamp": ""})
    with open(os.path.join(root, "poses_t.json"), "w", encoding="utf-8") as fh:
        json.dump({"images": images}, fh)

    detections = [
        {"frame": 0, "x1": 10, "y1": 10, "x2": 30, "y2": 30, "confidence": 0.9},
        {"frame": 0, "x1": 60, "y1": 40, "x2": 90, "y2": 70, "confidence": 0.8},
        {"frame": 1, "x1": 12, "y1": 12, "x2": 32, "y2": 32, "confidence": 0.9},
    ]
    conn = store.open_store(
        store.stage_path(root, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    with store.transaction(conn):
        for row in detections:
            conn.execute(
                "INSERT INTO detections (frame, x1, y1, x2, y2, confidence, "
                "species_id, source_id) VALUES (?, ?, ?, ?, ?, ?, 0, 1)",
                (row["frame"], row["x1"], row["y1"], row["x2"], row["y2"],
                 row["confidence"]))
    conn.close()

    track_store.record_tracks(root, "t", [
        {"track_id": 1, "detection_id": 1},
        {"track_id": 2, "detection_id": 2},
        {"track_id": 1, "detection_id": 3},
    ])

    monkeypatch.setattr(classification, "Backbone", FakeBackbone)
    monkeypatch.setattr(
        "bambi_wildlife_detection.bambi_processing.BambiProcessor."
        "_get_default_model_dir", staticmethod(lambda: str(tmp_path / "models")))
    return root


def _config(root, **overrides):
    config = {
        "target_folder": root,
        "embeddings_camera": "T",
        "classification_projection": "non_geo",
        "classification_backbone": "fake/backbone",
        "classification_device": "cpu",
        "classification_batch_size": 2,
        "classification_crop_padding": 0.1,
        "classification_crop_size": 32,
        "classification_letterbox": True,
        "hf_token": "",
    }
    config.update(overrides)
    return config


def _run(root, **overrides):
    logs = []
    BambiProcessor().run_embeddings(
        _config(root, **overrides), log_fn=logs.append)
    return logs


# ---------------------------------------------------------------------------
# Producing vectors
# ---------------------------------------------------------------------------

def test_every_tracked_detection_is_embedded(flight):
    _run(flight)

    run = classification_store.active_embedding_run(flight, "t")
    assert classification_store.embedded_ids(flight, "t", run["run_id"]) == \
        {1, 2, 3}
    assert FakeBackbone.last.embedded == 3


def test_vectors_land_beside_the_frames(flight):
    _run(flight)

    folder = embedding_files.run_folder(flight, "t", "non_geo")
    assert sorted(os.listdir(folder)) == \
        ["frame_000000.npz", "frame_000001.npz"]


def test_a_frame_archive_holds_that_frames_detections(flight):
    _run(flight)

    archive = embedding_files.read_frame(embedding_files.frame_path(
        flight, "t", "non_geo", 0, "frame_000000.jpg"))
    assert set(archive) == {1, 2}
    assert archive[1].shape == (DIM,)


def test_no_sidecar_metadata_is_written(flight):
    """The run is described by embedding_runs alone; a second copy would be a
    second thing that can disagree."""
    _run(flight)

    folder = embedding_files.run_folder(flight, "t", "non_geo")
    assert all(name.endswith(".npz") for name in os.listdir(folder))


def test_the_run_records_what_produced_it(flight):
    _run(flight)

    run = classification_store.active_embedding_run(flight, "t")
    assert run["backbone"] == "fake/backbone"
    assert run["dim"] == DIM
    assert run["crop_size"] == 32
    assert run["projection"] == "non_geo"
    assert run["folder"] == "embeddings_t/non_geo"


def test_the_stage_is_recorded(flight):
    from bambi_wildlife_detection.core import stages

    _run(flight)
    assert "embeddings" in stages.completed_stages(flight, "t")


# ---------------------------------------------------------------------------
# Resuming — why membership is stored at all
# ---------------------------------------------------------------------------

def test_a_second_run_embeds_nothing(flight):
    _run(flight)
    logs = _run(flight)

    assert FakeBackbone.last.embedded == 0
    assert any("already embedded" in line for line in logs)


def test_only_the_missing_detections_are_recomputed(flight):
    _run(flight)

    # One frame's vectors are lost, as if the file had been deleted by hand.
    os.remove(embedding_files.frame_path(
        flight, "t", "non_geo", 1, "frame_000001.jpg"))
    logs = _run(flight)

    assert FakeBackbone.last.embedded == 1
    assert any("missing from disk" in line for line in logs)


def test_an_interrupted_run_keeps_what_it_finished(flight):
    calls = {"n": 0}

    def _cancel():
        calls["n"] += 1
        return calls["n"] > 1        # let the first frame through

    with pytest.raises(CancelledException):
        BambiProcessor().run_embeddings(_config(flight), cancel_check=_cancel)

    run = classification_store.active_embedding_run(flight, "t")
    finished = classification_store.embedded_ids(flight, "t", run["run_id"])
    assert finished == {1, 2}        # frame 0 only

    # Resuming does the remainder and nothing more.
    _run(flight)
    assert FakeBackbone.last.embedded == 1


def test_changing_the_crop_size_starts_a_new_run(flight):
    _run(flight)
    first = classification_store.active_embedding_run(flight, "t")["run_id"]

    _run(flight, classification_crop_size=64)
    second = classification_store.active_embedding_run(flight, "t")["run_id"]

    assert second != first
    assert FakeBackbone.last.embedded == 3


def test_the_previous_runs_vectors_are_not_discarded(flight):
    _run(flight)
    first = classification_store.active_embedding_run(flight, "t")["run_id"]
    _run(flight, classification_crop_size=64)

    # Switching back must not cost another pass over the flight.
    assert classification_store.embedded_ids(flight, "t", first) == {1, 2, 3}


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_untracked_modality_is_refused(flight):
    with pytest.raises(ValueError, match="No RGB tracks"):
        BambiProcessor().run_embeddings(_config(flight, embeddings_camera="W"))


def test_a_geo_projection_without_geotiffs_is_refused(flight):
    with pytest.raises(ValueError, match="No Thermal GeoTIFFs"):
        BambiProcessor().run_embeddings(
            _config(flight, classification_projection="geo_2k"))


def test_an_unreadable_frame_is_skipped_with_a_warning(flight):
    os.remove(os.path.join(flight, "frames_t", "frame_000001.jpg"))
    logs = _run(flight)

    assert any("could not be read" in line for line in logs)
    run = classification_store.active_embedding_run(flight, "t")
    # The other frame still produced its vectors.
    assert classification_store.embedded_ids(flight, "t", run["run_id"]) == {1, 2}


def test_the_backbone_is_loaded_before_the_work_starts(flight):
    """A bad token should fail in seconds, not after a long scan."""
    _run(flight)
    assert FakeBackbone.last.loaded >= 1


# ---------------------------------------------------------------------------
# Thermal-anchored RGB crops
# ---------------------------------------------------------------------------

def _add_rgb_side(root):
    """A minimal RGB modality with one detection matched to thermal id 1."""
    from bambi_wildlife_detection.core import match_store

    conn = store.open_store(
        store.stage_path(root, store.DETECTIONS, "w"), store.DETECTIONS, "w")
    with store.transaction(conn):
        conn.execute(
            "INSERT INTO detections (frame, x1, y1, x2, y2, confidence, "
            "species_id, source_id) VALUES (0, 100, 100, 110, 110, 0.9, 0, 1)")
    conn.close()
    track_store.record_tracks(root, "w", [
        {"track_id": 1, "detection_id": 1}])

    # Thermal is half the RGB scale, so one thermal pixel spans two RGB ones.
    match_store.record_matches(root, [{
        "track_id_t": 1, "track_id_w": 1, "shared": 10, "median_dist": 1.0,
        "conf_t": 0.9, "conf_w": 0.9,
        "pairs": [{"frame_t": 0, "frame_w": 0,
                   "detection_id_t": 1, "detection_id_w": 1, "dist": 1.0}],
    }], affine=([[0.5, 0.0], [0.0, 0.5]], [0.0, 0.0]))


def test_rgb_crops_are_sized_from_their_thermal_partner(flight):
    _add_rgb_side(flight)
    logs = []
    sizes = BambiProcessor()._thermal_anchor_sizes(
        _config(flight, embeddings_camera="W"), flight, "w", logs.append)

    # Thermal detection 1 is a 20x20 box; at half scale that is 40x40 in RGB —
    # much larger than the RGB box's own 10x10, which is the point: the
    # thermal box is the looser one and encloses the whole animal.
    assert sizes == {1: pytest.approx((40.0, 40.0))}
    assert any("thermal partner" in line for line in logs)


def test_thermal_crops_are_never_anchored(flight):
    """The anchoring runs one way: thermal is the reference, not the follower."""
    _add_rgb_side(flight)
    assert BambiProcessor()._thermal_anchor_sizes(
        _config(flight), flight, "t") == {}


def test_anchoring_can_be_switched_off(flight):
    _add_rgb_side(flight)
    assert BambiProcessor()._thermal_anchor_sizes(
        _config(flight, match_thermal_anchored=False), flight, "w") == {}


def test_without_a_matching_run_nothing_is_anchored(flight):
    assert BambiProcessor()._thermal_anchor_sizes(
        _config(flight), flight, "w") == {}


# ---------------------------------------------------------------------------
# Orthorectified crops
# ---------------------------------------------------------------------------

def _add_geo_side(root, width=2048, height=1024):
    """Geo-referenced detections plus the GeoTIFFs they map into."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from bambi_wildlife_detection.core import geo_crops

    origin_x, origin_y, pixel = 1000.0, 2000.0, 0.1
    for frame in (0, 1):
        path = geo_crops.geotiff_path(root, "t", frame)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with rasterio.open(
                path, "w", driver="GTiff", height=height, width=width,
                count=3, dtype="uint8",
                transform=from_origin(origin_x, origin_y, pixel, pixel),
                crs="EPSG:32633") as raster:
            raster.write(np.full((3, height, width), 120, dtype=np.uint8))

    conn = store.open_store(
        store.stage_path(root, store.GEOREFERENCED, "t"),
        store.GEOREFERENCED, "t")
    with store.transaction(conn):
        for detection_id in (1, 2, 3):
            offset = detection_id * 2.0
            conn.execute(
                "INSERT INTO detections_geo (detection_id, gx1, gy1, gz1, "
                "gx2, gy2, gz2) VALUES (?, ?, ?, 0, ?, ?, 0)",
                (detection_id, origin_x + offset, origin_y - offset - 1.0,
                 origin_x + offset + 1.0, origin_y - offset))
    conn.close()


def test_geo_crops_come_from_the_geotiffs(flight):
    _add_geo_side(flight)
    _run(flight, classification_projection="geo_2k")

    run = classification_store.active_embedding_run(flight, "t")
    assert run["projection"] == "geo_2k"
    assert classification_store.embedded_ids(flight, "t", run["run_id"]) == \
        {1, 2, 3}
    assert FakeBackbone.last.embedded == 3


def test_geo_vectors_are_kept_apart_from_perspective_ones(flight):
    _add_geo_side(flight)
    _run(flight)
    _run(flight, classification_projection="geo_2k")

    assert os.path.isdir(embedding_files.run_folder(flight, "t", "non_geo"))
    assert os.path.isdir(embedding_files.run_folder(flight, "t", "geo_2k"))


def test_a_geo_run_needs_geo_referenced_detections(flight):
    rasterio = pytest.importorskip("rasterio")  # noqa: F841
    from rasterio.transform import from_origin
    from bambi_wildlife_detection.core import geo_crops

    path = geo_crops.geotiff_path(flight, "t", 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=64, width=64,
                       count=3, dtype="uint8",
                       transform=from_origin(1000.0, 2000.0, 0.5, 0.5),
                       crs="EPSG:32633") as raster:
        raster.write(np.zeros((3, 64, 64), dtype=np.uint8))

    with pytest.raises(ValueError, match="world coordinates"):
        BambiProcessor().run_embeddings(
            _config(flight, classification_projection="geo_2k"))


def test_a_mismatched_export_resolution_is_flagged_once(flight):
    _add_geo_side(flight, width=300, height=200)
    logs = _run(flight, classification_projection="geo_1k")

    warnings = [line for line in logs if "were fitted around" in line]
    assert len(warnings) == 1


def test_thermal_anchoring_does_not_apply_to_geo_crops(flight):
    """An orthophoto is already in metric ground units, so both modalities'
    boxes describe the same footprint and there is nothing to borrow."""
    _add_geo_side(flight)
    _add_rgb_side(flight)
    _run(flight, classification_projection="geo_2k")

    run = classification_store.active_embedding_run(flight, "t")
    assert run["thermal_anchored"] == 0


def test_anchoring_does_not_rewrite_the_stored_detections(flight):
    """Only the crop is affected — rewriting boxes would invalidate
    geo-referencing and tracking upstream."""
    _add_rgb_side(flight)
    before = track_store.load_pixel_tracks(flight, "w")
    BambiProcessor()._thermal_anchor_sizes(
        _config(flight, embeddings_camera="W"), flight, "w")
    assert track_store.load_pixel_tracks(flight, "w") == before
