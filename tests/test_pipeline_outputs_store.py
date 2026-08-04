# -*- coding: utf-8 -*-
"""The QGIS layer readers preferring the store (EXCHANGE_FORMAT_PLAN.md §11).

The plugin never hands a stage file to QGIS as a layer source — layers are
built in memory from rows. These readers are where those rows come from, so
they read the store when there is one and the legacy text otherwise, returning
the same shape either way.
"""
import os

import pytest

from bambi_wildlife_detection.core import (
    detection_store, pipeline_outputs, store, track_store)


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


@pytest.fixture
def populated(tmp_path):
    """A project with two detections, geo-referenced and tracked."""
    root = str(tmp_path)
    detection_store.record_detections(root, "t", [
        {"frame": 1, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0,
         "confidence": 0.9, "source_class": "0"},
        {"frame": 2, "x1": 11.0, "y1": 21.0, "x2": 31.0, "y2": 41.0,
         "confidence": 0.8, "source_class": "0"},
    ])
    ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
    track_store.record_georeference(root, "t", [
        {"detection_id": ids[0], "gx1": 100.0, "gy1": 200.0, "gz1": 5.0,
         "gx2": 110.0, "gy2": 210.0, "gz2": 5.0},
        {"detection_id": ids[1], "gx1": 300.0, "gy1": 400.0, "gz1": 5.0,
         "gx2": 310.0, "gy2": 410.0, "gz2": 5.0},
    ])
    track_store.record_tracks(root, "t", [
        {"track_id": 7, "detection_id": ids[0]},
        {"track_id": 7, "detection_id": ids[1]},
    ])
    return root


# ---------------------------------------------------------------------------
# Path derivation
# ---------------------------------------------------------------------------

def test_store_source_found_for_a_legacy_path(populated):
    path = os.path.join(populated, "tracks_t", "tracks.csv")
    assert pipeline_outputs._store_source(path, "tracks") == (populated, "t")


def test_store_source_none_without_a_store(tmp_path):
    path = os.path.join(str(tmp_path), "tracks_t", "tracks.csv")
    assert pipeline_outputs._store_source(path, "tracks") is None


def test_store_source_none_for_an_unknown_modality(populated):
    path = os.path.join(populated, "tracks_x", "tracks.csv")
    assert pipeline_outputs._store_source(path, "tracks") is None


def test_store_source_none_for_a_folderless_path(tmp_path):
    assert pipeline_outputs._store_source("tracks.csv", "tracks") is None


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------

def test_geo_tracks_come_from_the_store(populated):
    path = os.path.join(populated, "tracks_t", "tracks.csv")
    tracks = pipeline_outputs.load_geo_tracks_by_id(path)
    assert len(tracks) == 1
    points = list(tracks.values())[0]
    assert [p["frame"] for p in points] == [1, 2]
    assert points[0]["x1"] == 100.0


def test_geo_tracks_shape_matches_the_text_reader(populated, tmp_path):
    """The layer builders must not care which source was used."""
    stored = pipeline_outputs.load_geo_tracks_by_id(
        os.path.join(populated, "tracks_t", "tracks.csv"))

    legacy_root = str(tmp_path / "legacy")
    legacy = os.path.join(legacy_root, "tracks_t", "tracks.csv")
    _write(legacy,
           "1,7,100.0,200.0,5.0,110.0,210.0,5.0,0.9,0,0\n"
           "2,7,300.0,400.0,5.0,310.0,410.0,5.0,0.8,0,0\n")
    from_text = pipeline_outputs.load_geo_tracks_by_id(legacy)

    assert set(list(stored.values())[0][0]) == set(list(from_text.values())[0][0])


def test_geo_tracks_fall_back_to_the_csv(tmp_path):
    path = os.path.join(str(tmp_path), "tracks_t", "tracks.csv")
    _write(path, "1,7,100.0,200.0,5.0,110.0,210.0,5.0,0.9,0,0\n")
    tracks = pipeline_outputs.load_geo_tracks_by_id(path)
    assert list(tracks) == [7]


def test_geo_tracks_use_the_active_run(populated):
    """A second run supersedes the first for the layers (§3.3)."""
    ids = [d["detection_id"] for d in track_store.load_detections(populated, "t")]
    track_store.record_tracks(populated, "t", [
        {"track_id": 1, "detection_id": ids[0]},
        {"track_id": 2, "detection_id": ids[1]},
    ], kind="boxmot")

    tracks = pipeline_outputs.load_geo_tracks_by_id(
        os.path.join(populated, "tracks_t", "tracks.csv"))
    assert len(tracks) == 2


def test_geo_tracks_without_an_active_run_fall_back(populated):
    """Manual tracks are recorded inactive, so labels alone show no layer.

    That is deliberate (§8.2): pooling manual and tracker output would
    double-count animals, so it has to be an explicit choice.
    """
    conn = store.open_store(
        store.stage_path(populated, store.TRACKS, "t"), store.TRACKS, "t")
    conn.execute("UPDATE track_runs SET is_active = 0")
    conn.commit()
    conn.close()

    path = os.path.join(populated, "tracks_t", "tracks.csv")
    _write(path, "1,9,1.0,2.0,3.0,4.0,5.0,6.0,0.9,0,0\n")
    assert list(pipeline_outputs.load_geo_tracks_by_id(path)) == [9]


# ---------------------------------------------------------------------------
# Geo-referenced detections
# ---------------------------------------------------------------------------

def test_georef_detections_come_from_the_store(populated):
    path = os.path.join(populated, "georeferenced_t", "georeferenced.txt")
    by_frame = pipeline_outputs.load_georef_detections_by_frame(path)
    assert sorted(by_frame) == [1, 2]
    assert by_frame[1][0]["x1"] == 100.0
    assert by_frame[1][0]["detection_id"]


def test_georef_detections_fall_back_to_the_text(tmp_path):
    path = os.path.join(str(tmp_path), "georeferenced_t", "georeferenced.txt")
    _write(path, "0 1 100.0 200.0 5.0 110.0 210.0 5.0 0.9 0\n")
    by_frame = pipeline_outputs.load_georef_detections_by_frame(path)
    assert by_frame[1][0]["x1"] == 100.0


def test_failed_detections_are_simply_absent(populated):
    """No negative-corner sentinel: a failure has no geo row at all (§3.2)."""
    ids = [d["detection_id"] for d in track_store.load_detections(populated, "t")]
    track_store.record_georeference(
        populated, "t",
        [{"detection_id": ids[0], "gx1": 1.0, "gy1": 2.0, "gz1": 3.0,
          "gx2": 4.0, "gy2": 5.0, "gz2": 6.0}],
        [{"detection_id": ids[1], "reason": "beyond_mesh"}])

    path = os.path.join(populated, "georeferenced_t", "georeferenced.txt")
    by_frame = pipeline_outputs.load_georef_detections_by_frame(path)
    assert sorted(by_frame) == [1]
    assert all(row["x1"] >= 0 for rows in by_frame.values() for row in rows)


# ---------------------------------------------------------------------------
# FoV
# ---------------------------------------------------------------------------

def test_fov_polygons_come_from_the_store(tmp_path):
    root = str(tmp_path)
    conn = store.open_store(
        store.stage_path(root, store.FOV, "t"), store.FOV, "t")
    conn.execute("INSERT INTO fov_polygons (frame, n_points) VALUES (1, 3)")
    conn.executemany(
        "INSERT INTO fov_vertices (frame, seq, x, y, z) VALUES (?, ?, ?, ?, ?)",
        [(1, 0, 0.0, 0.0, 0.0), (1, 1, 1.0, 0.0, 0.0), (1, 2, 1.0, 1.0, 0.0)])
    conn.commit()
    conn.close()

    polygons = pipeline_outputs.load_fov_polygons_3d(
        os.path.join(root, "fov_t", "fov_polygons.txt"))
    assert polygons == {1: [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]}


def test_fov_polygons_fall_back_to_the_text(tmp_path):
    path = os.path.join(str(tmp_path), "fov_t", "fov_polygons.txt")
    _write(path, "1 3 0.0 0.0 0.0 1.0 0.0 0.0 1.0 1.0 0.0\n")
    polygons = pipeline_outputs.load_fov_polygons_3d(path)
    assert list(polygons) == [1]
    assert len(polygons[1]) == 3


def test_empty_store_falls_back_rather_than_returning_nothing(tmp_path):
    """An empty stage file must not mask a populated legacy file."""
    root = str(tmp_path)
    store.open_store(
        store.stage_path(root, store.FOV, "t"), store.FOV, "t").close()
    path = os.path.join(root, "fov_t", "fov_polygons.txt")
    _write(path, "1 3 0.0 0.0 0.0 1.0 0.0 0.0 1.0 1.0 0.0\n")
    assert list(pipeline_outputs.load_fov_polygons_3d(path)) == [1]
