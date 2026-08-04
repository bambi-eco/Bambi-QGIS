# -*- coding: utf-8 -*-
"""Geo-referencing and tracking against the store (EXCHANGE_FORMAT_PLAN.md §3.2, §3.3).

The two properties that matter here are the ones the text format could not
express: every detection is accounted for, and a detection that fails to
geo-reference costs only itself.
"""
import pytest

from bambi_wildlife_detection.core import detection_store, store, track_store


@pytest.fixture
def project(tmp_path):
    """Three detections on two frames, ready to geo-reference."""
    root = str(tmp_path)
    detection_store.record_detections(root, "t", [
        {"frame": 1, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0,
         "confidence": 0.9, "source_class": "0"},
        {"frame": 1, "x1": 50.0, "y1": 60.0, "x2": 70.0, "y2": 80.0,
         "confidence": 0.8, "source_class": "0"},
        {"frame": 2, "x1": 11.0, "y1": 21.0, "x2": 31.0, "y2": 41.0,
         "confidence": 0.7, "source_class": "0"},
    ])
    return root


def _ids(project):
    return [d["detection_id"] for d in track_store.load_detections(project, "t")]


def _geo(detection_id, offset=0.0):
    return {"detection_id": detection_id,
            "gx1": 100.0 + offset, "gy1": 200.0 + offset, "gz1": 5.0,
            "gx2": 110.0 + offset, "gy2": 210.0 + offset, "gz2": 5.0}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_has_store(tmp_path, project):
    assert track_store.has_store(project, "t")
    assert not track_store.has_store(str(tmp_path / "elsewhere"), "t")


def test_detections_come_back_with_their_ids(project):
    rows = track_store.load_detections(project, "t")
    assert len(rows) == 3
    assert all(row["detection_id"] for row in rows)
    assert [row["frame"] for row in rows] == [1, 1, 2]


def test_loading_from_an_empty_project(tmp_path):
    assert track_store.load_detections(str(tmp_path), "t") == []


# ---------------------------------------------------------------------------
# Geo-referencing
# ---------------------------------------------------------------------------

def test_records_resolved_boxes(project):
    ids = _ids(project)
    result = track_store.record_georeference(
        project, "t", [_geo(i) for i in ids])
    assert result == {"resolved": 3, "failed": 0}
    assert len(track_store.load_georeferenced(project, "t")) == 3


def test_records_failures_with_reasons(project):
    ids = _ids(project)
    track_store.record_georeference(
        project, "t", [_geo(ids[0])],
        [{"detection_id": ids[1], "reason": "above_horizon"},
         {"detection_id": ids[2], "reason": "beyond_mesh"}])
    assert track_store.failure_summary(project, "t") == \
        {"above_horizon": 1, "beyond_mesh": 1}


def test_geo_rows_join_back_to_their_detections(project):
    ids = _ids(project)
    track_store.record_georeference(project, "t", [_geo(i) for i in ids])
    rows = track_store.load_georeferenced(project, "t")
    assert [row["detection_id"] for row in rows] == ids
    assert [row["frame"] for row in rows] == [1, 1, 2]


def test_rerunning_replaces_previous_results(project):
    ids = _ids(project)
    track_store.record_georeference(project, "t", [_geo(i) for i in ids])
    track_store.record_georeference(project, "t", [_geo(ids[0])])
    assert len(track_store.load_georeferenced(project, "t")) == 1
    assert track_store.failure_summary(project, "t") == {}


def test_log_reports_the_split(project):
    ids = _ids(project)
    messages = []
    track_store.record_georeference(
        project, "t", [_geo(ids[0])],
        [{"detection_id": ids[1], "reason": "no_pose"}], log_fn=messages.append)
    assert any("1 detection(s) geo-referenced" in m for m in messages)


# ---------------------------------------------------------------------------
# Total accounting (§12.2) — the assertion the text format could not express
# ---------------------------------------------------------------------------

def test_everything_accounted_for_when_all_resolve(project):
    ids = _ids(project)
    track_store.record_georeference(project, "t", [_geo(i) for i in ids])
    report = track_store.accounting(project, "t")
    assert report["detections"] == 3
    assert report["resolved"] == 3
    assert report["unaccounted"] == []


def test_everything_accounted_for_with_failures(project):
    ids = _ids(project)
    track_store.record_georeference(
        project, "t", [_geo(ids[0])],
        [{"detection_id": ids[1], "reason": "above_horizon"},
         {"detection_id": ids[2], "reason": "beyond_mesh"}])
    report = track_store.accounting(project, "t")
    assert report["unaccounted"] == []
    assert report["resolved"] == 1 and report["failed"] == 2


def test_a_silently_dropped_detection_is_reported(project):
    """The 5.x failure mode: a detection that simply vanishes."""
    ids = _ids(project)
    track_store.record_georeference(project, "t", [_geo(ids[0])])
    report = track_store.accounting(project, "t")
    assert report["unaccounted"] == sorted(ids[1:])


def test_a_detection_cannot_be_both_resolved_and_failed(project):
    ids = _ids(project)
    track_store.record_georeference(
        project, "t", [_geo(ids[0])],
        [{"detection_id": ids[0], "reason": "no_pose"}])
    assert track_store.accounting(project, "t")["both"] == [ids[0]]


def test_accounting_before_georeferencing_ran(project):
    report = track_store.accounting(project, "t")
    assert report["detections"] == 3
    assert report["unaccounted"] == _ids(project)


def test_accounting_of_an_empty_project(tmp_path):
    assert track_store.accounting(str(tmp_path), "t")["detections"] == 0


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def test_records_a_run_with_its_members(project):
    ids = _ids(project)
    result = track_store.record_tracks(project, "t", [
        {"track_id": 7, "detection_id": ids[0]},
        {"track_id": 7, "detection_id": ids[2]},
        {"track_id": 8, "detection_id": ids[1]},
    ])
    assert result["tracks"] == 2
    assert result["members"] == 3


def test_pixel_tracks_are_a_join_not_a_reconstruction(project):
    """What replaces core/track_export.py's coordinate hashing."""
    ids = _ids(project)
    track_store.record_tracks(project, "t", [
        {"track_id": 7, "detection_id": ids[0]},
        {"track_id": 7, "detection_id": ids[2]},
        {"track_id": 8, "detection_id": ids[1]},
    ])
    rows = track_store.load_pixel_tracks(project, "t")
    assert len(rows) == 3
    by_track = {}
    for row in rows:
        by_track.setdefault(row["track_id"], []).append(row["frame"])
    assert sorted(sorted(v) for v in by_track.values()) == [[1], [1, 2]]
    assert all(row["x1"] is not None for row in rows)


def test_a_failed_geo_detection_does_not_cost_its_frame_mates(project):
    """The core/track_export.py regression, stated directly.

    Detections 0 and 1 share frame 1. Detection 1 fails to geo-reference; the
    old code discarded every pixel track in that frame, because its per-frame
    counts no longer lined up. Here detection 0 keeps its track.
    """
    ids = _ids(project)
    track_store.record_georeference(
        project, "t", [_geo(ids[0]), _geo(ids[2], 50.0)],
        [{"detection_id": ids[1], "reason": "beyond_mesh"}])
    track_store.record_tracks(project, "t", [
        {"track_id": 7, "detection_id": ids[0]},
        {"track_id": 7, "detection_id": ids[2]},
    ])

    rows = track_store.load_pixel_tracks(project, "t")
    frame_one = [row for row in rows if row["frame"] == 1]
    assert len(frame_one) == 1
    assert frame_one[0]["detection_id"] == ids[0]


def test_two_animals_at_the_same_place_stay_distinct(project):
    """Rounded coordinates collided; ids cannot."""
    root = project
    detection_store.record_detections(root, "w", [
        {"frame": 1, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0,
         "confidence": 0.9, "source_class": "0"},
        {"frame": 1, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0,
         "confidence": 0.9, "source_class": "0"},
    ])
    ids = [d["detection_id"] for d in track_store.load_detections(root, "w")]
    track_store.record_tracks(root, "w", [
        {"track_id": 1, "detection_id": ids[0]},
        {"track_id": 2, "detection_id": ids[1]},
    ])
    rows = track_store.load_pixel_tracks(root, "w")
    assert len({row["track_id"] for row in rows}) == 2


def test_interpolated_members_are_flagged(project):
    ids = _ids(project)
    track_store.record_tracks(project, "t", [
        {"track_id": 7, "detection_id": ids[0], "interpolated": 0},
        {"track_id": 7, "detection_id": ids[1], "interpolated": 1},
    ])
    rows = track_store.load_pixel_tracks(project, "t")
    assert sorted(row["interpolated"] for row in rows) == [0, 1]


def test_species_can_be_assigned_per_track(project):
    ids = _ids(project)
    track_store.record_tracks(
        project, "t",
        [{"track_id": 7, "detection_id": ids[0]}],
        species_of=lambda legacy: 3)
    conn = store.open_store(
        store.stage_path(project, store.TRACKS, "t"), store.TRACKS, "t")
    assert conn.execute(
        "SELECT species_id FROM tracks").fetchone()["species_id"] == 3
    conn.close()


# ---------------------------------------------------------------------------
# Runs coexist (§3.3)
# ---------------------------------------------------------------------------

def test_a_new_run_deactivates_the_previous_one(project):
    ids = _ids(project)
    first = track_store.record_tracks(
        project, "t", [{"track_id": 1, "detection_id": ids[0]}], kind="builtin")
    second = track_store.record_tracks(
        project, "t", [{"track_id": 1, "detection_id": ids[1]}], kind="boxmot")

    active = track_store.active_run(project, "t")
    assert active["run_id"] == second["run_id"] != first["run_id"]
    assert active["kind"] == "boxmot"


def test_previous_runs_are_kept_for_comparison(project):
    ids = _ids(project)
    first = track_store.record_tracks(
        project, "t", [{"track_id": 1, "detection_id": ids[0]}])
    track_store.record_tracks(
        project, "t", [{"track_id": 1, "detection_id": ids[1]}])

    assert len(track_store.load_pixel_tracks(
        project, "t", run_id=first["run_id"])) == 1


def test_generation_counts_per_tracker_kind(project):
    ids = _ids(project)
    a = track_store.record_tracks(
        project, "t", [{"track_id": 1, "detection_id": ids[0]}], kind="builtin")
    b = track_store.record_tracks(
        project, "t", [{"track_id": 1, "detection_id": ids[0]}], kind="builtin")
    c = track_store.record_tracks(
        project, "t", [{"track_id": 1, "detection_id": ids[0]}], kind="trex")
    assert (a["generation"], b["generation"], c["generation"]) == (1, 2, 1)


def test_track_ids_are_unique_across_runs(project):
    ids = _ids(project)
    track_store.record_tracks(
        project, "t", [{"track_id": 1, "detection_id": ids[0]}])
    track_store.record_tracks(
        project, "t", [{"track_id": 1, "detection_id": ids[1]}])

    conn = store.open_store(
        store.stage_path(project, store.TRACKS, "t"), store.TRACKS, "t")
    rows = [dict(r) for r in conn.execute("SELECT track_id, run_id FROM tracks")]
    conn.close()
    assert len({r["track_id"] for r in rows}) == 2


def test_no_active_run_before_tracking(project):
    assert track_store.active_run(project, "t") is None
    assert track_store.load_pixel_tracks(project, "t") == []


# ---------------------------------------------------------------------------
# Orphans
# ---------------------------------------------------------------------------

def test_no_orphans_in_a_healthy_project(project):
    ids = _ids(project)
    track_store.record_tracks(
        project, "t", [{"track_id": 1, "detection_id": ids[0]}])
    assert track_store.track_orphans(project, "t") == []


def test_orphans_are_detected(project):
    """Deleting detections under a tracking run must not go unnoticed."""
    ids = _ids(project)
    track_store.record_tracks(
        project, "t", [{"track_id": 1, "detection_id": ids[0]}])

    conn = store.open_store(
        store.stage_path(project, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    conn.execute("DELETE FROM detections WHERE detection_id = ?", (ids[0],))
    conn.commit()
    conn.close()

    assert track_store.track_orphans(project, "t") == [ids[0]]


def test_orphans_of_an_empty_project(tmp_path):
    assert track_store.track_orphans(str(tmp_path), "t") == []
