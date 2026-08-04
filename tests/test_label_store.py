# -*- coding: utf-8 -*-
"""Label materialisation against the store (EXCHANGE_FORMAT_PLAN.md §6).

The upsert matrix of §6.2 is the contract: editing one key frame must not
re-mint every detection in the flight, or the labelling loop becomes unusable.
Everything else here follows from that.
"""
import json

import pytest

from bambi_wildlife_detection.core import (
    detection_store, label_store, store, track_store)


def _track(label_track_id=1, species_id=0, keyframes=None, attributes=None):
    return {
        "label_track_id": label_track_id,
        "species_id": species_id,
        "attributes": attributes or {},
        "keyframes": keyframes if keyframes is not None else [
            {"frame": 1, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
            {"frame": 3, "x1": 30.0, "y1": 40.0, "x2": 50.0, "y2": 60.0},
        ],
    }


def _detections(root, modality="t"):
    conn = store.open_store(
        store.stage_path(root, store.DETECTIONS, modality),
        store.DETECTIONS, modality)
    try:
        return [dict(r) for r in conn.execute(
            "SELECT detection_id, frame, x1, y1, x2, y2, species_id, "
            "label_track_id, attributes FROM detections "
            "ORDER BY label_track_id, frame")]
    finally:
        conn.close()


def _by_frame(root, modality="t"):
    return {row["frame"]: row["detection_id"] for row in _detections(root, modality)}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_tracks_round_trip(tmp_path):
    root = str(tmp_path)
    label_store.save_tracks(root, "t", [
        _track(1, 3, attributes={"collar_id": "R-114", "sex": 1})])
    loaded = label_store.load_tracks(root, "t")
    assert len(loaded) == 1
    assert loaded[0]["species_id"] == 3
    assert loaded[0]["attributes"] == {"collar_id": "R-114", "sex": 1}
    assert [kf["frame"] for kf in loaded[0]["keyframes"]] == [1, 3]


def test_keyframe_attributes_and_stop_round_trip(tmp_path):
    root = str(tmp_path)
    label_store.save_tracks(root, "t", [_track(1, keyframes=[
        {"frame": 1, "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0,
         "attributes": {"occlusion": 2}},
        {"frame": 5, "x1": 2.0, "y1": 2.0, "x2": 3.0, "y2": 3.0, "stop": True},
    ])])
    keyframes = label_store.load_tracks(root, "t")[0]["keyframes"]
    assert keyframes[0]["attributes"] == {"occlusion": 2}
    assert keyframes[1]["stop"] == 1


def test_saving_removes_deleted_tracks(tmp_path):
    root = str(tmp_path)
    label_store.save_tracks(root, "t", [_track(1), _track(2)])
    label_store.save_tracks(root, "t", [_track(1)])
    assert [t["label_track_id"] for t in label_store.load_tracks(root, "t")] == [1]


def test_loading_an_empty_project(tmp_path):
    assert label_store.load_tracks(str(tmp_path), "t") == []


# ---------------------------------------------------------------------------
# Interpolation — mirrors LabelTrack.box_at
# ---------------------------------------------------------------------------

def test_interpolation_fills_the_gap():
    rows = label_store.interpolate([
        {"frame": 1, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0},
        {"frame": 3, "x1": 20.0, "y1": 20.0, "x2": 30.0, "y2": 30.0},
    ])
    assert [row["frame"] for row in rows] == [1, 2, 3]
    assert rows[1]["x1"] == 10.0
    assert rows[1]["keyframe"] is False


def test_stop_frames_leave_a_hole():
    rows = label_store.interpolate([
        {"frame": 1, "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "stop": True},
        {"frame": 4, "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
    ])
    assert [row["frame"] for row in rows] == [1, 4]


def test_interpolated_frames_inherit_the_previous_attributes():
    rows = label_store.interpolate([
        {"frame": 1, "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0,
         "attributes": {"occlusion": 1}},
        {"frame": 3, "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0,
         "attributes": {"occlusion": 0}},
    ])
    assert rows[1]["attributes"] == {"occlusion": 1}


def test_interpolation_of_nothing():
    assert label_store.interpolate([]) == []


def test_single_keyframe_yields_one_box():
    rows = label_store.interpolate(
        [{"frame": 7, "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}])
    assert [row["frame"] for row in rows] == [7]


# ---------------------------------------------------------------------------
# The §6.2 upsert matrix
# ---------------------------------------------------------------------------

def test_first_materialisation_adds_every_frame(tmp_path):
    root = str(tmp_path)
    result = label_store.materialise(root, "t", [_track()])
    assert result["added"] == 3      # frames 1, 2, 3
    assert [row["frame"] for row in _detections(root)] == [1, 2, 3]


def test_unchanged_labels_keep_their_detection_ids(tmp_path):
    """The property the whole design turns on."""
    root = str(tmp_path)
    label_store.materialise(root, "t", [_track()])
    before = _by_frame(root)

    result = label_store.materialise(root, "t", [_track()])
    assert result == {"kept": 3, "moved": 0, "added": 0, "removed": 0,
                      "invalidated": 0, "tracks": 1}
    assert _by_frame(root) == before


def test_editing_one_keyframe_leaves_the_others_alone(tmp_path):
    root = str(tmp_path)
    label_store.materialise(root, "t", [_track()])
    before = _by_frame(root)

    edited = _track(keyframes=[
        {"frame": 1, "x1": 11.0, "y1": 20.0, "x2": 31.0, "y2": 40.0},
        {"frame": 3, "x1": 30.0, "y1": 40.0, "x2": 50.0, "y2": 60.0},
    ])
    result = label_store.materialise(root, "t", [edited])

    after = _by_frame(root)
    assert after == before                 # every id survived
    assert result["added"] == 0 and result["removed"] == 0
    assert result["moved"] == 2            # frames 1 and 2 (interpolated)
    assert result["kept"] == 1             # frame 3 untouched


def test_extending_a_track_mints_only_the_new_frames(tmp_path):
    root = str(tmp_path)
    label_store.materialise(root, "t", [_track()])
    before = _by_frame(root)

    extended = _track(keyframes=[
        {"frame": 1, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
        {"frame": 3, "x1": 30.0, "y1": 40.0, "x2": 50.0, "y2": 60.0},
        {"frame": 5, "x1": 50.0, "y1": 60.0, "x2": 70.0, "y2": 80.0},
    ])
    result = label_store.materialise(root, "t", [extended])
    after = _by_frame(root)

    assert result["added"] == 2            # frames 4 and 5
    for frame in (1, 2, 3):
        assert after[frame] == before[frame]


def test_shortening_a_track_removes_the_vanished_frames(tmp_path):
    root = str(tmp_path)
    label_store.materialise(root, "t", [_track()])
    shortened = _track(keyframes=[
        {"frame": 1, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
        {"frame": 2, "x1": 20.0, "y1": 30.0, "x2": 40.0, "y2": 50.0},
    ])
    result = label_store.materialise(root, "t", [shortened])
    assert result["removed"] == 1
    assert [row["frame"] for row in _detections(root)] == [1, 2]


def test_deleting_a_track_removes_its_detections(tmp_path):
    root = str(tmp_path)
    label_store.materialise(root, "t", [_track(1), _track(2)])
    label_store.materialise(root, "t", [_track(1)])
    assert {row["label_track_id"] for row in _detections(root)} == {1}


def test_changing_species_updates_without_re_minting(tmp_path):
    root = str(tmp_path)
    label_store.materialise(root, "t", [_track(species_id=0)])
    before = _by_frame(root)

    result = label_store.materialise(root, "t", [_track(species_id=3)])
    assert result["moved"] == 0 and result["added"] == 0
    assert _by_frame(root) == before
    assert {row["species_id"] for row in _detections(root)} == {3}


# ---------------------------------------------------------------------------
# Incremental geo-referencing (§6.2)
# ---------------------------------------------------------------------------

def _georeference_everything(root, modality="t"):
    rows = track_store.load_detections(root, modality)
    track_store.record_georeference(root, modality, [
        {"detection_id": row["detection_id"],
         "gx1": 1.0, "gy1": 2.0, "gz1": 3.0,
         "gx2": 4.0, "gy2": 5.0, "gz2": 6.0} for row in rows])


def test_a_moved_box_loses_its_geo_row(tmp_path):
    root = str(tmp_path)
    label_store.materialise(root, "t", [_track()])
    _georeference_everything(root)
    assert track_store.accounting(root, "t")["unaccounted"] == []

    label_store.materialise(root, "t", [_track(keyframes=[
        {"frame": 1, "x1": 99.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
        {"frame": 3, "x1": 30.0, "y1": 40.0, "x2": 50.0, "y2": 60.0},
    ])])

    # Exactly the moved detections now need re-projecting; the rest are done.
    pending = track_store.accounting(root, "t")["unaccounted"]
    assert len(pending) == 2


def test_an_untouched_label_keeps_its_geo_row(tmp_path):
    root = str(tmp_path)
    label_store.materialise(root, "t", [_track()])
    _georeference_everything(root)
    label_store.materialise(root, "t", [_track()])
    assert track_store.accounting(root, "t")["unaccounted"] == []


def test_a_removed_label_drops_its_geo_row(tmp_path):
    root = str(tmp_path)
    label_store.materialise(root, "t", [_track()])
    _georeference_everything(root)
    label_store.materialise(root, "t", [])
    report = track_store.accounting(root, "t")
    assert report["detections"] == 0 and report["resolved"] == 0


# ---------------------------------------------------------------------------
# Manual tracks (§6.5) and producer isolation (§4.1)
# ---------------------------------------------------------------------------

def test_label_tracks_become_real_tracks(tmp_path):
    root = str(tmp_path)
    label_store.materialise(root, "t", [_track(1), _track(2)])
    conn = store.open_store(
        store.stage_path(root, store.TRACKS, "t"), store.TRACKS, "t")
    try:
        runs = [dict(r) for r in conn.execute(
            "SELECT kind, is_active FROM track_runs")]
        tracks = conn.execute("SELECT COUNT(*) AS n FROM tracks").fetchone()["n"]
        members = conn.execute(
            "SELECT COUNT(*) AS n FROM track_members").fetchone()["n"]
    finally:
        conn.close()
    assert [run["kind"] for run in runs] == ["manual"]
    assert tracks == 2
    assert members == 6


def test_label_track_records_the_track_it_produced(tmp_path):
    root = str(tmp_path)
    label_store.save_tracks(root, "t", [_track(1)])
    label_store.materialise(root, "t")
    assert label_store.load_tracks(root, "t")[0]["track_id"] is not None


def test_rematerialising_replaces_only_the_manual_run(tmp_path):
    root = str(tmp_path)
    detection_store.record_detections(root, "t", [
        {"frame": 9, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.5, "source_class": "0"}])
    detector_ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
    track_store.record_tracks(
        root, "t", [{"track_id": 1, "detection_id": detector_ids[0]}],
        kind="builtin")

    label_store.materialise(root, "t", [_track()])
    label_store.materialise(root, "t", [_track()])

    conn = store.open_store(
        store.stage_path(root, store.TRACKS, "t"), store.TRACKS, "t")
    try:
        kinds = sorted(r["kind"] for r in conn.execute(
            "SELECT kind FROM track_runs"))
    finally:
        conn.close()
    assert kinds == ["builtin", "manual"]


def test_materialising_never_touches_detector_detections(tmp_path):
    root = str(tmp_path)
    detection_store.record_detections(root, "t", [
        {"frame": 9, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.5, "source_class": "0"}])
    label_store.materialise(root, "t", [_track()])
    label_store.materialise(root, "t", [])

    remaining = _detections(root)
    assert len(remaining) == 1
    assert remaining[0]["label_track_id"] is None


def test_a_detector_rerun_never_touches_manual_detections(tmp_path):
    root = str(tmp_path)
    label_store.materialise(root, "t", [_track()])
    manual_before = {row["detection_id"] for row in _detections(root)}

    detection_store.record_detections(root, "t", [
        {"frame": 9, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.5, "source_class": "0"}])

    manual_after = {row["detection_id"] for row in _detections(root)
                    if row["label_track_id"] is not None}
    assert manual_after == manual_before


def test_the_manual_run_is_not_a_tracker_run(tmp_path):
    """It is additive, not an alternative (§8.2).

    Tracker runs are alternatives — builtin, boxmot and TRex describe the same
    animals differently, so one is active at a time. The labelling tool's run
    is pooled alongside whichever that is, because its tracks are usually
    animals the detector missed.
    """
    root = str(tmp_path)
    label_store.materialise(root, "t", [_track()])
    assert track_store.active_run(root, "t") is None
    assert track_store.manual_run(root, "t") is not None
    assert track_store.analysis_runs(root, "t")


def test_clear_detector_detections_keeps_the_labels(tmp_path):
    """'Replace detections in project' becomes one delete plus the upsert."""
    root = str(tmp_path)
    detection_store.record_detections(root, "t", [
        {"frame": 9, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.5, "source_class": "0"}])
    label_store.materialise(root, "t", [_track()])

    assert label_store.clear_detector_detections(root, "t") == 1
    remaining = _detections(root)
    assert len(remaining) == 3
    assert all(row["label_track_id"] == 1 for row in remaining)


def test_clear_materialised_removes_the_labels_only(tmp_path):
    root = str(tmp_path)
    detection_store.record_detections(root, "t", [
        {"frame": 9, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.5, "source_class": "0"}])
    label_store.materialise(root, "t", [_track()])

    assert label_store.clear_materialised(root, "t") == 3
    assert len(_detections(root)) == 1


# ---------------------------------------------------------------------------
# Custom fields reach the detections (§5)
# ---------------------------------------------------------------------------

def test_keyframe_attributes_land_on_the_detections(tmp_path):
    """The original complaint: custom fields could not leave labels.json."""
    root = str(tmp_path)
    label_store.materialise(root, "t", [_track(keyframes=[
        {"frame": 1, "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0,
         "attributes": {"occlusion": 2, "posture": "standing"}},
        {"frame": 2, "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0,
         "attributes": {"occlusion": 0, "posture": "lying"}},
    ])])
    attributes = [json.loads(row["attributes"]) for row in _detections(root)]
    assert attributes[0] == {"occlusion": 2, "posture": "standing"}
    assert attributes[1] == {"occlusion": 0, "posture": "lying"}


def test_track_attributes_land_on_the_track(tmp_path):
    root = str(tmp_path)
    label_store.materialise(
        root, "t", [_track(attributes={"collar_id": "R-114"})])
    conn = store.open_store(
        store.stage_path(root, store.TRACKS, "t"), store.TRACKS, "t")
    try:
        attributes = json.loads(
            conn.execute("SELECT attributes FROM tracks").fetchone()["attributes"])
    finally:
        conn.close()
    assert attributes == {"collar_id": "R-114"}


def test_changing_an_attribute_does_not_move_the_box(tmp_path):
    root = str(tmp_path)
    base = _track(keyframes=[
        {"frame": 1, "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0,
         "attributes": {"occlusion": 0}}])
    label_store.materialise(root, "t", [base])
    _georeference_everything(root)

    changed = _track(keyframes=[
        {"frame": 1, "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0,
         "attributes": {"occlusion": 2}}])
    result = label_store.materialise(root, "t", [changed])

    assert result["moved"] == 0
    assert track_store.accounting(root, "t")["unaccounted"] == []
    assert json.loads(_detections(root)[0]["attributes"]) == {"occlusion": 2}


# ---------------------------------------------------------------------------
# Pixel tracks of manual data
# ---------------------------------------------------------------------------

def test_manual_tracks_are_readable_as_pixel_tracks(tmp_path):
    root = str(tmp_path)
    label_store.materialise(root, "t", [_track()])
    conn = store.open_store(
        store.stage_path(root, store.TRACKS, "t"), store.TRACKS, "t")
    try:
        run_id = conn.execute("SELECT run_id FROM track_runs").fetchone()["run_id"]
    finally:
        conn.close()
    rows = track_store.load_pixel_tracks(root, "t", run_id=run_id)
    assert [row["frame"] for row in rows] == [1, 2, 3]
    assert all(row["x1"] is not None for row in rows)


def test_no_orphans_after_materialising(tmp_path):
    root = str(tmp_path)
    label_store.materialise(root, "t", [_track()])
    label_store.materialise(root, "t", [_track(keyframes=[
        {"frame": 1, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0}])])
    assert track_store.track_orphans(root, "t") == []


@pytest.mark.parametrize("modality", ["t", "w"])
def test_modalities_are_independent(tmp_path, modality):
    root = str(tmp_path)
    label_store.materialise(root, modality, [_track()])
    other = "w" if modality == "t" else "t"
    assert label_store.load_tracks(root, other) == []


# ---------------------------------------------------------------------------
# The LabelStore adapter (core/labelling.py -> the store)
# ---------------------------------------------------------------------------

def _label_store(root, modality="t"):
    from bambi_wildlife_detection.core.labelling import LabelStore, LabelTrack

    store.open_store(store.project_path(root), store.PROJECT).close()
    ls = LabelStore(root, modality)
    track = LabelTrack(1, species="roe deer", sex="female", age="adult")
    track.set_keyframe(1, (10.0, 20.0, 30.0, 40.0), occlusion="partially")
    track.set_keyframe(3, (30.0, 40.0, 50.0, 60.0), occlusion="none")
    ls.tracks[1] = track
    return ls


def test_label_store_save_mirrors_into_the_store(tmp_path):
    root = str(tmp_path)
    _label_store(root).save()
    tracks = label_store.load_tracks(root, "t")
    assert len(tracks) == 1
    assert tracks[0]["species_id"] == 1          # 'roe deer' from the taxonomy
    assert [kf["frame"] for kf in tracks[0]["keyframes"]] == [1, 3]


def test_species_and_enums_are_stored_as_ids(tmp_path):
    """Names would break on rename; ids do not (§5.1)."""
    root = str(tmp_path)
    _label_store(root).save()
    track = label_store.load_tracks(root, "t")[0]
    assert track["attributes"]["sex"] == 1       # female
    assert track["attributes"]["age"] == 1       # adult
    assert track["keyframes"][0]["attributes"]["occlusion"] == 1   # partially


def test_export_to_detections_materialises(tmp_path):
    root = str(tmp_path)
    ls = _label_store(root)
    ls.save()
    ls.export_to_detections()

    rows = _detections(root)
    assert [row["frame"] for row in rows] == [1, 2, 3]
    assert all(row["label_track_id"] == 1 for row in rows)
    assert all(row["species_id"] == 1 for row in rows)


def test_export_is_idempotent_for_detection_ids(tmp_path):
    root = str(tmp_path)
    ls = _label_store(root)
    ls.save()
    ls.export_to_detections()
    before = _by_frame(root)
    ls.export_to_detections()
    assert _by_frame(root) == before


def test_replace_detections_drops_only_the_detector_rows(tmp_path):
    root = str(tmp_path)
    detection_store.record_detections(root, "t", [
        {"frame": 9, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.5, "source_class": "0"}])
    ls = _label_store(root)
    ls.save()
    ls.replace_detections()

    rows = _detections(root)
    assert len(rows) == 3
    assert all(row["label_track_id"] == 1 for row in rows)


def test_unknown_species_falls_back_to_animal(tmp_path):
    """The vocabulary is closed; an unmatched name resolves rather than fails."""
    from bambi_wildlife_detection.core.labelling import LabelTrack

    root = str(tmp_path)
    ls = _label_store(root)
    stray = LabelTrack(2, species="chupacabra")
    stray.set_keyframe(1, (0.0, 0.0, 1.0, 1.0))
    ls.tracks[2] = stray
    ls.save()

    tracks = {t["label_track_id"]: t for t in label_store.load_tracks(root, "t")}
    assert tracks[2]["species_id"] == store.FALLBACK_SPECIES_ID


def test_label_store_without_a_project_store_is_a_no_op(tmp_path):
    """The labelling tool must keep working on an un-migrated 5.x folder."""
    from bambi_wildlife_detection.core.labelling import LabelStore, LabelTrack

    root = str(tmp_path / "legacy")
    ls = LabelStore(root, "t")
    track = LabelTrack(1, species="roe deer")
    track.set_keyframe(1, (0.0, 0.0, 1.0, 1.0))
    ls.tracks[1] = track
    ls.save()
    ls.export_to_detections()

    assert not store.project_path(root) or not label_store.load_tracks(root, "t")
