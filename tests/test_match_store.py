# -*- coding: utf-8 -*-
"""Cross-modal match persistence.

The invariant worth defending here is one-to-one assignment: a thermal track
has at most one RGB partner and the reverse, because a track matched twice
would count one animal as two in every survey analytic downstream.
"""
import sqlite3

import pytest

from bambi_wildlife_detection.core import match_store, store


@pytest.fixture
def root(tmp_path):
    return str(tmp_path)


def _match(track_t, track_w, shared=10, dist=3.5, pairs=()):
    return {
        "track_id_t": track_t, "track_id_w": track_w,
        "shared": shared, "median_dist": dist,
        "conf_t": 0.8, "conf_w": 0.7,
        "pairs": list(pairs),
    }


def _pair(frame_t, frame_w, det_t, det_w, dist=2.0):
    return {"frame_t": frame_t, "frame_w": frame_w,
            "detection_id_t": det_t, "detection_id_w": det_w, "dist": dist}


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

def test_the_store_sits_beside_the_project_not_in_a_modality(root):
    path = store.matches_path(root)
    assert path.endswith("matches.gpkg")
    # Not under bambi_t/ or bambi_w/: a match belongs to both modalities, and
    # filing it under one would make resetting that one destroy the other's.
    assert "bambi_t" not in path and "bambi_w" not in path


def test_absent_store_reads_as_empty(root):
    assert not match_store.has_store(root)
    assert match_store.active_run(root) is None
    assert match_store.track_matches(root) == []
    assert match_store.detection_pairs(root) == []
    assert match_store.partner_tracks(root, "t") == {}


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_matches_round_trip(root):
    result = match_store.record_matches(
        root,
        [_match(1, 11, pairs=[_pair(0, 0, 100, 200), _pair(1, 1, 101, 201)]),
         _match(2, 12, dist=1.0, pairs=[_pair(5, 5, 105, 205)])],
        affine=([[1.0, 0.0], [0.0, 1.0]], [3.0, -2.0]), affine_rmse=0.4)

    assert result["matches"] == 2
    assert result["pairs"] == 3

    matches = match_store.track_matches(root)
    assert [m["track_id_t"] for m in matches] == [2, 1]   # closest first
    assert matches[0]["median_dist"] == 1.0
    assert matches[1]["shared"] == 10
    assert matches[1]["conf_t"] == 0.8


def test_affine_round_trips_as_numbers(root):
    match_store.record_matches(
        root, [_match(1, 11)],
        affine=([[1.5, 0.25], [-0.25, 1.5]], [3.0, -2.0]), affine_rmse=0.75)
    run = match_store.active_run(root)
    linear, translation = run["affine"]
    assert linear == [[1.5, 0.25], [-0.25, 1.5]]
    assert translation == [3.0, -2.0]
    assert run["affine_rmse"] == 0.75


def test_a_run_without_an_affine_is_allowed(root):
    match_store.record_matches(root, [_match(1, 11)])
    assert match_store.active_run(root)["affine"] is None


# ---------------------------------------------------------------------------
# One-to-one
# ---------------------------------------------------------------------------

def test_a_thermal_track_cannot_be_matched_twice(root):
    with pytest.raises(sqlite3.IntegrityError):
        match_store.record_matches(root, [_match(1, 11), _match(1, 12)])


def test_an_rgb_track_cannot_be_matched_twice(root):
    with pytest.raises(sqlite3.IntegrityError):
        match_store.record_matches(root, [_match(1, 11), _match(2, 11)])


def test_a_rejected_run_leaves_nothing_behind(root):
    """The insert is one transaction, so a violation cannot half-commit."""
    with pytest.raises(sqlite3.IntegrityError):
        match_store.record_matches(root, [_match(1, 11), _match(1, 12)])
    assert match_store.track_matches(root) == []
    assert match_store.active_run(root) is None


def test_the_same_pair_may_recur_in_a_later_run(root):
    match_store.record_matches(root, [_match(1, 11)])
    match_store.record_matches(root, [_match(1, 11, dist=2.0)])
    assert len(match_store.track_matches(root)) == 1
    assert match_store.track_matches(root)[0]["median_dist"] == 2.0


# ---------------------------------------------------------------------------
# Runs are versioned
# ---------------------------------------------------------------------------

def test_a_new_run_supersedes_the_previous(root):
    first = match_store.record_matches(root, [_match(1, 11)])["run_id"]
    second = match_store.record_matches(
        root, [_match(1, 11), _match(2, 12)])["run_id"]

    assert second != first
    assert match_store.active_run(root)["run_id"] == second
    assert len(match_store.track_matches(root)) == 2


def test_a_superseded_run_is_still_readable(root):
    first = match_store.record_matches(root, [_match(1, 11)])["run_id"]
    match_store.record_matches(root, [_match(3, 13), _match(4, 14)])

    # Comparing two gate settings is the point of keeping it.
    assert len(match_store.track_matches(root, run_id=first)) == 1
    assert match_store.track_matches(root, run_id=first)[0]["track_id_t"] == 1


def test_pairs_are_scoped_to_their_run(root):
    first = match_store.record_matches(
        root, [_match(1, 11, pairs=[_pair(0, 0, 100, 200)])])["run_id"]
    match_store.record_matches(
        root, [_match(1, 11, pairs=[_pair(0, 0, 100, 200),
                                    _pair(1, 1, 101, 201)])])

    assert len(match_store.detection_pairs(root, run_id=first)) == 1
    assert len(match_store.detection_pairs(root)) == 2


# ---------------------------------------------------------------------------
# Partner lookups — what builds a matched feature vector
# ---------------------------------------------------------------------------

def test_partner_detections_both_directions(root):
    match_store.record_matches(
        root, [_match(1, 11, pairs=[_pair(0, 0, 100, 200),
                                    _pair(1, 1, 101, 201)])])

    assert match_store.partner_detections(root, "t") == {100: 200, 101: 201}
    assert match_store.partner_detections(root, "w") == {200: 100, 201: 101}


def test_a_detection_without_a_counterpart_is_simply_absent(root):
    """The matched heads cannot take a zero-filled substitute, so the caller
    has to be able to tell 'no partner' from 'partner is zero'."""
    match_store.record_matches(
        root, [_match(1, 11, pairs=[_pair(0, 0, 100, 200)])])
    partners = match_store.partner_detections(root, "t")
    assert 999 not in partners


def test_partner_tracks_both_directions(root):
    match_store.record_matches(root, [_match(1, 11), _match(2, 12)])
    assert match_store.partner_tracks(root, "t") == {1: 11, 2: 12}
    assert match_store.partner_tracks(root, "w") == {11: 1, 12: 2}


def test_unknown_modality_is_rejected(root):
    match_store.record_matches(root, [_match(1, 11)])
    with pytest.raises(ValueError):
        match_store.partner_detections(root, "rgb")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_summary_counts_like_the_papers_table_2(root):
    match_store.record_matches(
        root, [_match(1, 11), _match(2, 12), _match(3, 13)])
    counts = match_store.summary(root, tracks_t=51, tracks_w=43)

    assert counts["raw"] == 94
    assert counts["confirmed"] == 3
    # A confirmed pair consumes one track on each side.
    assert counts["unmatched"] == 94 - 6


def test_summary_without_any_match(root):
    counts = match_store.summary(root, tracks_t=5, tracks_w=4)
    assert counts["confirmed"] == 0
    assert counts["unmatched"] == 9
