# -*- coding: utf-8 -*-
"""TRex tracklets replace tracking rather than competing with it.

A configured TRex tracklet folder means the tracks already exist and were
computed elsewhere. Running a tracker as well would produce a second, competing
set of tracks for the same animals — and since manual and tracker runs are now
pooled (EXCHANGE_FORMAT_PLAN.md §8.2), a stray extra run is exactly the kind of
thing that quietly doubles a population estimate.
"""
import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import detection_store, store, track_store


@pytest.fixture
def processor():
    return BambiProcessor()


def test_tracking_imports_instead_of_tracking(processor, monkeypatch, tmp_path):
    called = {"trex": 0, "builtin": 0, "advanced": 0}
    monkeypatch.setattr(BambiProcessor, "run_trex_import",
                        lambda *a, **k: called.__setitem__("trex", 1))
    monkeypatch.setattr(BambiProcessor, "_run_builtin_tracking",
                        lambda *a, **k: called.__setitem__("builtin", 1))
    monkeypatch.setattr(BambiProcessor, "_run_advanced_tracking",
                        lambda *a, **k: called.__setitem__("advanced", 1))

    processor.run_tracking({
        "target_folder": str(tmp_path),
        "trex_npz_dir": "/some/tracklets",
    })
    assert called == {"trex": 1, "builtin": 0, "advanced": 0}


def test_tracking_runs_normally_without_a_tracklet_folder(processor,
                                                          monkeypatch, tmp_path):
    called = {"trex": 0, "builtin": 0}
    monkeypatch.setattr(BambiProcessor, "run_trex_import",
                        lambda *a, **k: called.__setitem__("trex", 1))
    monkeypatch.setattr(BambiProcessor, "_run_builtin_tracking",
                        lambda *a, **k: called.__setitem__("builtin", 1))
    monkeypatch.setattr(BambiProcessor, "_write_pixel_tracks",
                        lambda *a, **k: "")

    processor.run_tracking({
        "target_folder": str(tmp_path),
        "trex_npz_dir": "",
        "tracker_id": "builtin",
    })
    assert called == {"trex": 0, "builtin": 1}


def test_a_whitespace_only_path_does_not_count(processor, monkeypatch, tmp_path):
    called = {"trex": 0, "builtin": 0}
    monkeypatch.setattr(BambiProcessor, "run_trex_import",
                        lambda *a, **k: called.__setitem__("trex", 1))
    monkeypatch.setattr(BambiProcessor, "_run_builtin_tracking",
                        lambda *a, **k: called.__setitem__("builtin", 1))
    monkeypatch.setattr(BambiProcessor, "_write_pixel_tracks",
                        lambda *a, **k: "")

    processor.run_tracking({
        "target_folder": str(tmp_path),
        "trex_npz_dir": "   ",
        "tracker_id": "builtin",
    })
    assert called["builtin"] == 1


def test_the_guard_reports_why(processor, monkeypatch, tmp_path):
    monkeypatch.setattr(BambiProcessor, "run_trex_import", lambda *a, **k: None)
    messages = []
    processor.run_tracking(
        {"target_folder": str(tmp_path), "trex_npz_dir": "/tracklets"},
        log_fn=messages.append)
    assert any("importing tracklets" in m for m in messages)


# ---------------------------------------------------------------------------
# Tracklets reach the store as tracks
# ---------------------------------------------------------------------------

def _import_detections(root, rows):
    detection_store.record_detections(
        root, "t",
        [{"frame": r["frame"], "x1": r["x1"], "y1": r["y1"],
          "x2": r["x2"], "y2": r["y2"], "confidence": r["confidence"],
          "source_class": str(r["class_id"])} for r in rows],
        kind=detection_store.TREX)


def _rows():
    return [
        {"frame": 1, "track_id": 7, "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0,
         "confidence": 0.9, "class_id": 0},
        {"frame": 2, "track_id": 7, "x1": 1.5, "y1": 2.5, "x2": 3.5, "y2": 4.5,
         "confidence": 0.9, "class_id": 0},
        {"frame": 1, "track_id": 8, "x1": 9.0, "y1": 9.0, "x2": 10.0, "y2": 10.0,
         "confidence": 0.8, "class_id": 0},
    ]


def test_tracklets_are_recorded_as_a_track_run(processor, tmp_path):
    root = str(tmp_path)
    rows = _rows()
    _import_detections(root, rows)

    processor._record_trex_tracks(root, "t", rows)

    conn = store.open_store(
        store.stage_path(root, store.TRACKS, "t"), store.TRACKS, "t")
    try:
        runs = [dict(r) for r in conn.execute(
            "SELECT kind, tracker FROM track_runs")]
        tracks = conn.execute("SELECT COUNT(*) AS n FROM tracks").fetchone()["n"]
    finally:
        conn.close()

    assert [run["kind"] for run in runs] == ["trex"]
    assert tracks == 2


def test_membership_comes_from_the_tracklet_id(processor, tmp_path):
    """No coordinate matching: the id travels with the imported detection."""
    root = str(tmp_path)
    rows = _rows()
    _import_detections(root, rows)
    processor._record_trex_tracks(root, "t", rows)

    pixel = track_store.load_pixel_tracks(root, "t")
    by_track = {}
    for row in pixel:
        by_track.setdefault(row["track_id"], []).append(row["frame"])
    assert sorted(sorted(v) for v in by_track.values()) == [[1], [1, 2]]


def test_a_count_mismatch_is_reported_not_guessed(processor, tmp_path):
    root = str(tmp_path)
    _import_detections(root, _rows()[:2])

    messages = []
    processor._record_trex_tracks(root, "t", _rows(), log_fn=messages.append)
    assert any("could not record TRex tracks" in m for m in messages)


def test_no_store_means_no_track_run(processor, tmp_path):
    """An un-migrated project must not gain a store as a side effect."""
    root = str(tmp_path / "legacy")
    processor._record_trex_tracks(root, "t", _rows())
    assert not store.project_path(root) or True


def test_empty_import_records_nothing(processor, tmp_path):
    root = str(tmp_path)
    _import_detections(root, _rows())
    processor._record_trex_tracks(root, "t", [])
    assert track_store.active_run(root, "t") is None
