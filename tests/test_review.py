# -*- coding: utf-8 -*-
"""core.review - deleting wrong detections and tracks from every store.

The fixture is the inventory test's flight: three tracks, geo-referencing,
classifier votes, per-frame occlusion, a cross-camera match and a labelling
annotation - every store that can reference a detection or a track, so a
deletion can be checked for what it left behind.
"""
import pytest

from bambi_wildlife_detection.core import (
    classification_store, detection_store, label_store, match_store,
    review, stages, store, track_store)


@pytest.fixture
def flight(tmp_path):
    root = str(tmp_path)
    label_store.vocabulary(root, create=True)
    detection_store.record_detections(root, "t", [
        {"frame": 0, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0,
         "confidence": 0.9, "source_class": "0"},
        {"frame": 2, "x1": 12.0, "y1": 22.0, "x2": 32.0, "y2": 42.0,
         "confidence": 0.7, "source_class": "0"},
        {"frame": 4, "x1": 14.0, "y1": 24.0, "x2": 34.0, "y2": 44.0,
         "confidence": 0.8, "source_class": "0"},
        {"frame": 1, "x1": 50.0, "y1": 60.0, "x2": 70.0, "y2": 80.0,
         "confidence": 0.6, "source_class": "0"},
        {"frame": 3, "x1": 90.0, "y1": 90.0, "x2": 95.0, "y2": 95.0,
         "confidence": 0.5, "source_class": "0"},
    ])
    ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
    track_store.record_georeference(root, "t", [
        {"detection_id": i, "gx1": 1.0, "gy1": 2.0, "gz1": 0.0,
         "gx2": 3.0, "gy2": 4.0, "gz2": 0.0} for i in ids[:3]],
        failures=[{"detection_id": ids[3], "reason": "beyond_mesh"},
                  {"detection_id": ids[4], "reason": "beyond_mesh"}])
    track_store.record_tracks(root, "t", [
        {"track_id": 1, "detection_id": ids[0]},
        {"track_id": 1, "detection_id": ids[1]},
        {"track_id": 1, "detection_id": ids[2]},
        {"track_id": 2, "detection_id": ids[3]},
        {"track_id": 3, "detection_id": ids[4]},
    ])
    run = classification_store.start_embedding_run(
        root, "t", backbone="b", dim=4, crop_size=8, padding=0.0,
        projection="non_geo", folder="embeddings_t/non_geo")
    classification_store.record_embedded(root, "t", run, ids)
    classification_store.record_frame_predictions(root, "t", "occlusion", [
        {"detection_id": i, "label": "clear", "class_index": 0, "prob": 0.9,
         "modality_in": "thermal", "model": "occ.pt"} for i in ids])
    classification_store.record_track_predictions(root, "t", "species", [
        {"track_id": 1, "label": "red_deer", "votes": 2, "n": 3,
         "fraction": 0.67, "model": "species.pt"},
        {"track_id": 3, "label": "roe_deer", "votes": 1, "n": 1,
         "fraction": 1.0, "model": "species.pt"}])
    match_store.record_matches(root, [
        {"track_id_t": 1, "track_id_w": 7, "shared": 3, "median_dist": 0.4,
         "conf_t": 0.8, "conf_w": 0.7,
         "pairs": [{"frame_t": 0, "frame_w": 0, "detection_id_t": ids[0],
                    "detection_id_w": 1, "dist": 0.4},
                   {"frame_t": 2, "frame_w": 2, "detection_id_t": ids[1],
                    "detection_id_w": 2, "dist": 0.5}]},
        {"track_id_t": 3, "track_id_w": 8, "shared": 1, "median_dist": 0.2,
         "conf_t": 0.5, "conf_w": 0.5,
         "pairs": [{"frame_t": 3, "frame_w": 3, "detection_id_t": ids[4],
                    "detection_id_w": 3, "dist": 0.2}]}])
    stages.mark_complete(root, "track_inventory", "t")
    stages.mark_complete(root, "classification", "t")
    return root


def _ids(root):
    return [d["detection_id"] for d in track_store.load_detections(root, "t")]


def _tracks(root):
    return sorted({m["track_id"] for m in track_store.load_pixel_tracks(root, "t")})


def _rows(root, kind, sql, modality="t"):
    path = (store.matches_path(root) if kind == store.MATCHES
            else store.stage_path(root, kind, modality))
    conn = store.open_store(path, kind, "" if kind == store.MATCHES else modality)
    try:
        return [dict(r) for r in conn.execute(sql)]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# One detection
# ---------------------------------------------------------------------------

class TestDeleteDetection:

    def test_removes_the_box_and_what_was_recorded_about_it(self, flight):
        ids = _ids(flight)
        gone = ids[1]
        result = review.delete_detections(flight, "t", [gone])
        assert result["detections"] == 1
        assert gone not in _ids(flight)
        assert track_store.accounting(flight, "t")["unaccounted"] == []
        assert not [r for r in _rows(flight, store.GEOREFERENCED,
                                     "SELECT * FROM detections_geo")
                    if r["detection_id"] == gone]
        assert not [r for r in _rows(flight, store.CLASSIFICATION,
                                     "SELECT * FROM embeddings")
                    if r["detection_id"] == gone]
        assert not [r for r in _rows(flight, store.CLASSIFICATION,
                                     "SELECT * FROM frame_predictions")
                    if r["detection_id"] == gone]
        pairs = match_store.detection_pairs(flight)
        assert gone not in {p["detection_id_t"] for p in pairs}
        assert len(pairs) == 2

    def test_the_track_keeps_its_other_members(self, flight):
        ids = _ids(flight)
        result = review.delete_detections(flight, "t", [ids[1]])
        assert result["tracks_kept"] == [1] and result["tracks_removed"] == []
        members = [m for m in track_store.load_pixel_tracks(flight, "t")
                   if m["track_id"] == 1]
        assert [m["frame"] for m in members] == [0, 4]
        assert track_store.track_orphans(flight, "t") == []
        # The animal is still there, so its votes and its match are too.
        assert classification_store.track_predictions(flight, "t", "species")
        assert 1 in match_store.partner_tracks(flight, "t")

    def test_a_track_emptied_by_its_last_box_goes_with_it(self, flight):
        ids = _ids(flight)
        result = review.delete_detections(flight, "t", [ids[4]])
        assert result["tracks_removed"] == [3]
        assert _tracks(flight) == [1, 2]
        votes = {p["track_id"] for p in
                 classification_store.track_predictions(flight, "t", "species")}
        assert votes == {1}
        assert 3 not in match_store.partner_tracks(flight, "t")
        assert track_store.track_orphans(flight, "t") == []

    def test_a_recorded_failure_is_removed_too(self, flight):
        ids = _ids(flight)
        review.delete_detections(flight, "t", [ids[3]])
        assert track_store.failure_summary(flight, "t") == {"beyond_mesh": 1}
        assert track_store.accounting(flight, "t")["unaccounted"] == []

    def test_unknown_ids_and_nothing_are_no_ops(self, flight):
        before = _ids(flight)
        assert review.delete_detections(flight, "t", [])["detections"] == 0
        assert review.delete_detections(flight, "t", [99999])["detections"] == 0
        assert _ids(flight) == before

    def test_refuses_a_box_of_a_label_track(self, flight):
        label_store.save_tracks(flight, "t", [{
            "label_track_id": 11, "track_id": None, "origin_track_id": None,
            "species_id": 0, "attributes": {},
            "keyframes": [{"frame": 5, "x1": 1, "y1": 1, "x2": 2, "y2": 2,
                           "stop": 0, "attributes": {}}]}])
        label_store.materialise(flight, "t")
        labelled = [d["detection_id"] for d in
                    track_store.load_detections(flight, "t")
                    if d["frame"] == 5]
        with pytest.raises(review.ReviewError, match="labelling tool"):
            review.delete_detections(flight, "t", labelled)
        assert labelled[0] in _ids(flight)

    def test_the_report_stages_go_stale(self, flight):
        ids = _ids(flight)
        review.delete_detections(flight, "t", [ids[0]])
        state = stages.states(flight, "t")
        assert state["track_inventory"]["state"] == stages.STALE
        # Per-animal results are cleaned row-wise, not flagged wholesale.
        assert state["classification"]["state"] == stages.COMPLETE

    def test_rejects_an_unknown_modality(self, flight):
        with pytest.raises(ValueError):
            review.delete_detections(flight, "x", [1])


# ---------------------------------------------------------------------------
# A whole track
# ---------------------------------------------------------------------------

class TestDeleteTrack:

    def test_takes_the_track_its_boxes_votes_and_match(self, flight):
        ids = _ids(flight)
        result = review.delete_tracks(flight, "t", [1])
        assert result["tracks"] == 1 and result["detections"] == 3
        assert _tracks(flight) == [2, 3]
        assert set(_ids(flight)) == set(ids[3:])
        assert track_store.track_orphans(flight, "t") == []
        assert track_store.accounting(flight, "t")["unaccounted"] == []
        votes = {p["track_id"] for p in
                 classification_store.track_predictions(flight, "t", "species")}
        assert votes == {3}
        assert match_store.partner_tracks(flight, "t") == {3: 8}
        assert {p["detection_id_t"] for p in match_store.detection_pairs(flight)} \
            == {ids[4]}

    def test_the_other_cameras_stores_are_untouched(self, flight):
        detection_store.record_detections(flight, "w", [
            {"frame": 0, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
             "confidence": 0.9, "source_class": "0"}])
        before = track_store.load_detections(flight, "w")
        review.delete_tracks(flight, "t", [1])
        assert track_store.load_detections(flight, "w") == before

    def test_describe_before_deleting(self, flight):
        facts = review.describe_track_deletion(flight, "t", 1)
        assert facts["exists"] and facts["detections"] == 3
        assert facts["matched"] and facts["label_track_ids"] == []
        assert review.describe_track_deletion(flight, "t", 42)["exists"] is False

    def test_a_materialised_label_track_is_deleted_with_its_track(self, flight):
        label_store.save_tracks(flight, "t", [{
            "label_track_id": 11, "track_id": None, "origin_track_id": None,
            "species_id": 0, "attributes": {},
            "keyframes": [{"frame": 5, "x1": 1, "y1": 1, "x2": 2, "y2": 2,
                           "stop": 0, "attributes": {}}]}])
        label_store.materialise(flight, "t")
        manual = [t for t in label_store.load_tracks(flight, "t")
                  if t["label_track_id"] == 11][0]
        assert manual["track_id"] is not None
        facts = review.describe_track_deletion(flight, "t", manual["track_id"])
        assert facts["label_track_ids"] == [11]

        result = review.delete_tracks(flight, "t", [manual["track_id"]])
        assert result["label_tracks"] == [11]
        assert not [t for t in label_store.load_tracks(flight, "t")
                    if t["label_track_id"] == 11]
        assert manual["track_id"] not in _tracks(flight)
        # Nothing comes back on the next save.
        assert label_store.materialise(flight, "t")["added"] == 0
        assert track_store.track_orphans(flight, "t") == []

    def test_unknown_track_is_a_no_op(self, flight):
        before = _ids(flight)
        result = review.delete_tracks(flight, "t", [42])
        assert result["detections"] == 0 and _ids(flight) == before

    def test_without_optional_stores(self, tmp_path):
        root = str(tmp_path)
        detection_store.record_detections(root, "t", [
            {"frame": 0, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
             "confidence": 0.9, "source_class": "0"}])
        ids = _ids(root)
        track_store.record_tracks(root, "t", [
            {"track_id": 1, "detection_id": ids[0]}])
        assert review.delete_tracks(root, "t", [1])["detections"] == 1
        assert _ids(root) == [] and _tracks(root) == []


class TestLookups:

    def test_track_of_detection(self, flight):
        ids = _ids(flight)
        assert review.track_of_detection(flight, "t", ids[1]) == 1
        assert review.track_of_detection(flight, "t", ids[4]) == 3
        assert review.track_of_detection(flight, "t", 99999) is None

    def test_track_members(self, flight):
        ids = _ids(flight)
        assert review.track_members(flight, "t", 1) == sorted(ids[:3])
        assert review.track_members(flight, "t", 42) == []
