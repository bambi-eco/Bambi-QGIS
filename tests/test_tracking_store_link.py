# -*- coding: utf-8 -*-
"""Tracking has to land in the store, or nothing downstream can see it.

Tracks written only to ``tracks_{m}/tracks.csv`` look like a successful run —
the file is there, the log says "5 tracks" — while every export and analytic,
which read the store, report the detections as untracked. Two ways in:

* the built-in tracker falling back to ``georeferenced.txt`` because the geo
  store is missing, in which case membership cannot be recorded at all;
* the advanced (BoxMOT) backend, which used to record nothing either way.
"""
import os

import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import detection_store, store, track_store


@pytest.fixture
def processor():
    return BambiProcessor()


@pytest.fixture
def detected(tmp_path):
    """Detections in the store, plus a stale georeferenced.txt beside them."""
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    detection_store.record_detections(root, "t", [
        {"frame": 0, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0,
         "confidence": 0.9, "source_class": "0"},
        {"frame": 1, "x1": 12.0, "y1": 22.0, "x2": 32.0, "y2": 42.0,
         "confidence": 0.8, "source_class": "0"},
    ])

    folder = os.path.join(root, "georeferenced_t")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "georeferenced.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("# idx frame min_x min_y min_z max_x max_y max_z conf cls\n")
        fh.write("0 0 500000.0 5300000.0 400.0 500010.0 5300010.0 400.0 "
                 "0.9000 0\n")
        fh.write("1 1 500001.0 5300001.0 400.0 500011.0 5300011.0 400.0 "
                 "0.8000 0\n")
    return root


def _config(root):
    return {"target_folder": root, "tracking_camera": "T",
            "tracker_mode": "HUNGARIAN", "write_legacy_text_outputs": True}


# ---------------------------------------------------------------------------
# The built-in tracker
# ---------------------------------------------------------------------------

def test_tracking_refuses_the_stale_text_fallback(processor, detected):
    """georeferenced.txt carries no detection_id, so a run built from it
    could never be linked back to the detections it describes. It is not read
    at all, however recent it looks."""
    with pytest.raises(FileNotFoundError, match="Geo-Reference Detections"):
        processor._run_builtin_tracking(_config(detected))


def test_the_refusal_happens_before_anything_is_written(processor, detected):
    with pytest.raises((RuntimeError, OSError)):
        processor._run_builtin_tracking(_config(detected))

    assert not os.path.exists(
        os.path.join(detected, "tracks_t", "tracks.csv"))


def test_tracking_proceeds_once_the_geo_store_exists(processor, detected):
    ids = [d["detection_id"]
           for d in track_store.load_detections(detected, "t")]
    track_store.record_georeference(detected, "t", [
        {"detection_id": i, "gx1": 500000.0 + n, "gy1": 5300000.0 + n,
         "gz1": 400.0, "gx2": 500010.0 + n, "gy2": 5300010.0 + n,
         "gz2": 400.0} for n, i in enumerate(ids)])

    processor._run_builtin_tracking(_config(detected))

    assert track_store.analysis_runs(detected, "t")


def test_the_recorded_run_covers_the_detections(processor, detected):
    """The symptom that started this: tracks.csv written, store untouched."""
    ids = [d["detection_id"]
           for d in track_store.load_detections(detected, "t")]
    track_store.record_georeference(detected, "t", [
        {"detection_id": i, "gx1": 500000.0 + n, "gy1": 5300000.0 + n,
         "gz1": 400.0, "gx2": 500010.0 + n, "gy2": 5300010.0 + n,
         "gz2": 400.0} for n, i in enumerate(ids)])

    processor._run_builtin_tracking(_config(detected))

    tracked = [row for row in track_store.load_pixel_tracks(detected, "t")]
    assert {row["detection_id"] for row in tracked} == set(ids)


def test_a_project_without_a_store_is_told_to_migrate(processor, tmp_path):
    """A 5.x folder is not silently tracked from its text files — the result
    would exist only as tracks.csv and be invisible to everything else."""
    root = str(tmp_path / "legacy")
    folder = os.path.join(root, "georeferenced_t")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "georeferenced.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("0 0 500000.0 5300000.0 400.0 500010.0 5300010.0 400.0 0.9 0")

    with pytest.raises(FileNotFoundError, match="Migrate"):
        processor._run_builtin_tracking(_config(root))
    assert not os.path.exists(os.path.join(root, "tracks_t", "tracks.csv"))
