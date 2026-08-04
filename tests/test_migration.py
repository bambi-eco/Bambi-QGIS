# -*- coding: utf-8 -*-
"""Golden-file tests for the 5.x → 6.0 migration (EXCHANGE_FORMAT_PLAN.md §9).

The fixtures below are written in the exact byte layout the 5.x pipeline
produced, so they pin the parsers as much as the mapping. The migration has no
second chance to be correct: it runs once per project, and afterwards the legacy
files are the only record of what the numbers meant.
"""
import json
import os

import pytest

from bambi_wildlife_detection.core import migration, store

MARKER = migration.DETECTIONS_MARKER


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


@pytest.fixture
def legacy(tmp_path):
    """A 5.x thermal target folder with every output kind present."""
    root = str(tmp_path / "flight")

    # Detector block, then the labelling tool's export below the marker.
    detector_block = (
        "# frame x1 y1 x2 y2 confidence class_id\n"
        "1 10.00 20.00 30.00 40.00 0.9000 0\n"
        "1 50.00 60.00 70.00 80.00 0.8000 1\n"
        "2 11.00 21.00 31.00 41.00 0.7000 0\n"
    )
    label_block = (
        "# class_id mapping: 0=unknown, 10=wolf\n"
        "3 12.00 22.00 32.00 42.00 1.0000 0\n"
        "3 13.00 23.00 33.00 43.00 1.0000 10\n"
    )
    _write(os.path.join(root, "detections_t", "detections.txt"),
           detector_block + MARKER + "\n" + label_block)

    # One row per detection, in the same per-frame order. The frame-2 row is
    # the "dropped detection" case: negative corners.
    _write(os.path.join(root, "georeferenced_t", "georeferenced.txt"),
           "0 1 100.0 200.0 5.0 110.0 210.0 5.0 0.9000 0\n"
           "1 1 300.0 400.0 5.0 310.0 410.0 5.0 0.8000 1\n"
           "2 2 -1.0 -1.0 0.0 -1.0 -1.0 0.0 0.7000 0\n"
           "3 3 500.0 600.0 5.0 510.0 610.0 5.0 1.0000 0\n"
           "4 3 700.0 800.0 5.0 710.0 810.0 5.0 1.0000 10\n")

    _write(os.path.join(root, "tracks_t", "tracks_pixel.csv"),
           "# frame,track_id,x1,y1,x2,y2,conf,cls,interpolated\n"
           "1,7,10.00,20.00,30.00,40.00,0.9000,0,0\n"
           "1,8,50.00,60.00,70.00,80.00,0.8000,1,0\n"
           "2,7,11.00,21.00,31.00,41.00,0.7000,0,0\n")

    _write(os.path.join(root, "tracks_t", "tracks.csv"),
           "1,7,100.0,200.0,5.0,110.0,210.0,5.0,0.9,0,0\n"
           "1,8,300.0,400.0,5.0,310.0,410.0,5.0,0.8,1,0\n")

    _write(os.path.join(root, "fov_t", "fov_polygons.txt"),
           "1 3 0.0 0.0 0.0 1.0 0.0 0.0 1.0 1.0 0.0\n"
           "2 0\n")

    _write(os.path.join(root, "labels_t", "labels.json"), json.dumps({
        "modality": "t",
        "custom_fields": [
            {"name": "collar_id", "type": "string", "scope": "track"},
            {"name": "posture", "type": "string", "scope": "keyframe"},
        ],
        "tracks": [
            {"track_id": 1, "species": "wolf", "sex": "female", "age": "adult",
             "attributes": {"collar_id": "R-114"},
             "keyframes": {
                 "3": {"x1": 12.0, "y1": 22.0, "x2": 32.0, "y2": 42.0,
                       "occlusion": "partially",
                       "attributes": {"posture": "standing"}},
                 "9": {"x1": 14.0, "y1": 24.0, "x2": 34.0, "y2": 44.0,
                       "occlusion": "none", "stop": True},
             }},
            {"track_id": 2, "species": "unknown", "sex": "unknown",
             "age": "unknown", "keyframes": {
                 "3": {"x1": 13.0, "y1": 23.0, "x2": 33.0, "y2": 43.0,
                       "occlusion": "none"}}},
        ],
    }))

    _write(os.path.join(root, "segmentation_t", "segmentation_pixel.json"),
           json.dumps([{"frame_idx": 1, "imagefile": "f1.jpg", "prompts": [
               {"prompt": "animal", "predictions": [
                   {"confidence": 0.77, "polygons": [[[0, 0], [1, 0], [1, 1]]]}]}]}]))
    _write(os.path.join(root, "segmentation_t", "segmentation_georef.json"),
           json.dumps([{"frame_idx": 1, "imagefile": "f1.jpg", "prompts": [
               {"prompt": "animal", "predictions": [
                   {"confidence": 0.77,
                    "world_polygons": [[10.0, 20.0, 11.0, 20.0, 11.0, 21.0]]}]}]}]))
    return root


def _rows(conn, sql, *args):
    return [dict(r) for r in conn.execute(sql, args)]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_detects_legacy_outputs(legacy):
    assert migration.has_legacy_outputs(legacy)
    assert migration.legacy_modalities(legacy) == ["t"]


def test_empty_folder_has_nothing_to_migrate(tmp_path):
    assert not migration.has_legacy_outputs(str(tmp_path))
    report = migration.migrate_project(str(tmp_path))
    assert report.counts == {}
    assert any("nothing to migrate" in w for w in report.warnings)


def test_is_migrated_reflects_the_project_store(legacy):
    assert not migration.is_migrated(legacy)
    migration.migrate_project(legacy)
    assert migration.is_migrated(legacy)


# ---------------------------------------------------------------------------
# The frozen species mapping (§1.3)
# ---------------------------------------------------------------------------

def test_legacy_species_class_ids_reproduces_the_5x_rule():
    mapping = migration.legacy_species_class_ids({"wolf", "badger", "roe deer"})
    assert mapping["unknown"] == 0
    assert mapping["roe deer"] == 1
    assert mapping["other"] == 9
    assert mapping["badger"] == 10   # custom, alphabetical
    assert mapping["wolf"] == 11


def test_custom_species_keep_the_id_5x_would_have_given_them(legacy):
    migration.migrate_project(legacy)
    conn = store.open_store(store.project_path(legacy), store.PROJECT)
    names = {r["name"]: r["species_id"] for r in _rows(
        conn, "SELECT species_id, name FROM species")}
    assert names["wolf"] == 10
    assert names["roe deer"] == 1
    conn.close()


def test_base_classes_survive_migration(legacy):
    migration.migrate_project(legacy)
    conn = store.open_store(store.project_path(legacy), store.PROJECT)
    names = {r["name"]: r["species_id"] for r in _rows(
        conn, "SELECT species_id, name FROM species WHERE protected = 1")}
    assert names == {"animal": 0, "unknown": -1, "not-an-animal": -2}
    conn.close()


# ---------------------------------------------------------------------------
# detections.txt, and the class_id-0 disambiguation (§9)
# ---------------------------------------------------------------------------

def test_detections_split_by_the_marker(legacy):
    detector, labels = migration.read_legacy_detections(
        os.path.join(legacy, "detections_t", "detections.txt"))
    assert len(detector) == 3
    assert len(labels) == 2
    assert detector[0]["frame"] == 1
    assert labels[0]["frame"] == 3


def test_detections_are_imported_with_their_source(legacy):
    migration.migrate_project(legacy)
    conn = store.open_store(
        store.stage_path(legacy, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    project = store.open_store(store.project_path(legacy), store.PROJECT)
    kinds = {r["source_id"]: r["kind"] for r in _rows(
        project, "SELECT source_id, kind FROM detection_sources")}
    rows = _rows(conn, "SELECT frame, source_id FROM detections ORDER BY detection_id")
    assert [kinds[r["source_id"]] for r in rows] == \
        ["detector", "detector", "detector", "manual", "manual"]
    conn.close()
    project.close()


def test_detection_ids_follow_file_order(legacy):
    migration.migrate_project(legacy)
    conn = store.open_store(
        store.stage_path(legacy, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    rows = _rows(conn, "SELECT detection_id, frame, x1 FROM detections "
                       "ORDER BY detection_id")
    assert [r["frame"] for r in rows] == [1, 1, 2, 3, 3]
    assert [r["x1"] for r in rows] == [10.0, 50.0, 11.0, 12.0, 13.0]
    conn.close()


def test_class_zero_above_the_marker_is_animal(legacy):
    """The detector's 0 means 'an animal' (§1.3)."""
    migration.migrate_project(legacy)
    conn = store.open_store(
        store.stage_path(legacy, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    rows = _rows(conn, "SELECT species_id FROM detections "
                       "WHERE frame IN (1, 2) ORDER BY detection_id")
    assert [r["species_id"] for r in rows] == [0, 1, 0]
    conn.close()


def test_class_zero_below_the_marker_is_unknown(legacy):
    """The labelling tool's 0 means 'not yet determined' — a different class."""
    migration.migrate_project(legacy)
    conn = store.open_store(
        store.stage_path(legacy, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    rows = _rows(conn, "SELECT species_id FROM detections WHERE frame = 3 "
                       "ORDER BY detection_id")
    assert [r["species_id"] for r in rows] == [-1, 10]
    conn.close()


def test_raw_class_is_kept_alongside_the_resolved_species(legacy):
    """So a corrected mapping can be re-applied without re-running anything."""
    migration.migrate_project(legacy)
    conn = store.open_store(
        store.stage_path(legacy, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    rows = _rows(conn, "SELECT source_class, species_id FROM detections "
                       "WHERE frame = 3 ORDER BY detection_id")
    assert rows[0]["source_class"] == "0" and rows[0]["species_id"] == -1
    conn.close()


def test_the_zero_decision_is_stored_as_mapping_rows(legacy):
    """Not a hardcoded branch — it stays visible and editable afterwards."""
    migration.migrate_project(legacy)
    project = store.open_store(store.project_path(legacy), store.PROJECT)
    rows = _rows(project,
                 "SELECT s.kind, m.source_class, m.species_id "
                 "FROM class_mapping m JOIN detection_sources s USING (source_id) "
                 "WHERE m.source_class = '0'")
    by_kind = {r["kind"]: r["species_id"] for r in rows}
    assert by_kind == {"detector": 0, "manual": -1}
    project.close()


# ---------------------------------------------------------------------------
# georeferenced.txt
# ---------------------------------------------------------------------------

def test_valid_geo_rows_are_linked_to_their_detections(legacy):
    migration.migrate_project(legacy)
    conn = store.open_store(
        store.stage_path(legacy, store.GEOREFERENCED, "t"),
        store.GEOREFERENCED, "t")
    rows = _rows(conn, "SELECT detection_id, gx1 FROM detections_geo "
                       "ORDER BY detection_id")
    assert [r["gx1"] for r in rows] == [100.0, 300.0, 500.0, 700.0]
    conn.close()


def test_invalid_geo_rows_become_failures_instead_of_vanishing(legacy):
    """5.x dropped these silently; they are now accounted for (§3.2)."""
    migration.migrate_project(legacy)
    conn = store.open_store(
        store.stage_path(legacy, store.GEOREFERENCED, "t"),
        store.GEOREFERENCED, "t")
    rows = _rows(conn, "SELECT detection_id, reason FROM georef_failures")
    assert len(rows) == 1
    assert rows[0]["reason"] == "legacy_invalid"
    conn.close()


def test_every_detection_is_accounted_for(legacy):
    """The assertion the old format could not express (§12.2)."""
    migration.migrate_project(legacy)
    det = store.open_store(
        store.stage_path(legacy, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    geo = store.open_store(
        store.stage_path(legacy, store.GEOREFERENCED, "t"),
        store.GEOREFERENCED, "t")
    all_ids = {r["detection_id"] for r in _rows(
        det, "SELECT detection_id FROM detections")}
    resolved = {r["detection_id"] for r in _rows(
        geo, "SELECT detection_id FROM detections_geo")}
    failed = {r["detection_id"] for r in _rows(
        geo, "SELECT detection_id FROM georef_failures")}
    assert all_ids == resolved | failed
    assert not (resolved & failed)
    det.close()
    geo.close()


def test_count_mismatch_is_reported_not_guessed(tmp_path):
    root = str(tmp_path / "flight")
    _write(os.path.join(root, "detections_t", "detections.txt"),
           "1 10.0 20.0 30.0 40.0 0.9 0\n")
    _write(os.path.join(root, "georeferenced_t", "georeferenced.txt"),
           "0 1 1.0 2.0 3.0 4.0 5.0 6.0 0.9 0\n"
           "1 1 7.0 8.0 9.0 10.0 11.0 12.0 0.9 0\n")
    report = migration.migrate_project(root)
    assert report.counts.get("georef_unlinked") == 2
    assert any("geo rows vs" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Tracks — the linkage recovered one final time
# ---------------------------------------------------------------------------

def test_tracks_are_linked_through_pixel_boxes(legacy):
    migration.migrate_project(legacy)
    tracks = store.open_store(
        store.stage_path(legacy, store.TRACKS, "t"), store.TRACKS, "t")
    det = store.open_store(
        store.stage_path(legacy, store.DETECTIONS, "t"), store.DETECTIONS, "t")

    members = _rows(tracks, "SELECT track_id, detection_id FROM track_members")
    assert len(members) == 3

    frames = {r["detection_id"]: r["frame"] for r in _rows(
        det, "SELECT detection_id, frame FROM detections")}
    per_track = {}
    for m in members:
        per_track.setdefault(m["track_id"], []).append(frames[m["detection_id"]])
    assert sorted(sorted(v) for v in per_track.values()) == [[1], [1, 2]]
    tracks.close()
    det.close()


def test_track_run_records_where_the_linkage_came_from(legacy):
    migration.migrate_project(legacy)
    conn = store.open_store(
        store.stage_path(legacy, store.TRACKS, "t"), store.TRACKS, "t")
    runs = _rows(conn, "SELECT kind, tracker, is_active FROM track_runs")
    assert len(runs) == 1
    assert runs[0]["is_active"] == 1
    assert "tracks_pixel.csv" in runs[0]["tracker"]
    conn.close()


def test_unmatched_track_point_does_not_cost_the_frame(tmp_path):
    """The core/track_export.py regression, in migration form (§9).

    5.x discarded every detection in a frame whose counts disagreed. Here the
    one unmatchable point is dropped and the rest of the frame survives.
    """
    root = str(tmp_path / "flight")
    _write(os.path.join(root, "detections_t", "detections.txt"),
           "1 10.0 20.0 30.0 40.0 0.9 0\n"
           "1 50.0 60.0 70.0 80.0 0.8 0\n")
    _write(os.path.join(root, "tracks_t", "tracks_pixel.csv"),
           "1,7,10.0,20.0,30.0,40.0,0.9,0,0\n"
           "1,7,99.0,99.0,99.0,99.0,0.5,0,0\n")   # matches nothing

    report = migration.migrate_project(root)
    conn = store.open_store(
        store.stage_path(root, store.TRACKS, "t"), store.TRACKS, "t")
    assert len(_rows(conn, "SELECT * FROM track_members")) == 1
    assert report.counts.get("track_members_unmatched") == 1
    assert any("could not be matched" in w for w in report.warnings)
    conn.close()


def test_geo_tracks_are_used_when_no_pixel_file_exists(tmp_path):
    root = str(tmp_path / "flight")
    _write(os.path.join(root, "detections_t", "detections.txt"),
           "1 10.0 20.0 30.0 40.0 0.9 0\n")
    _write(os.path.join(root, "georeferenced_t", "georeferenced.txt"),
           "0 1 100.0 200.0 5.0 110.0 210.0 5.0 0.9 0\n")
    _write(os.path.join(root, "tracks_t", "tracks.csv"),
           "1,7,100.0,200.0,5.0,110.0,210.0,5.0,0.9,0,0\n")

    migration.migrate_project(root)
    conn = store.open_store(
        store.stage_path(root, store.TRACKS, "t"), store.TRACKS, "t")
    assert len(_rows(conn, "SELECT * FROM track_members")) == 1
    assert "tracks.csv" in _rows(conn, "SELECT tracker FROM track_runs")[0]["tracker"]
    conn.close()


# ---------------------------------------------------------------------------
# FoV
# ---------------------------------------------------------------------------

def test_fov_polygons_and_vertices(legacy):
    migration.migrate_project(legacy)
    conn = store.open_store(
        store.stage_path(legacy, store.FOV, "t"), store.FOV, "t")
    polygons = _rows(conn, "SELECT frame, n_points FROM fov_polygons")
    assert polygons == [{"frame": 1, "n_points": 3}]   # the empty frame 2 is skipped
    vertices = _rows(conn, "SELECT seq, x, y, z FROM fov_vertices ORDER BY seq")
    assert [v["x"] for v in vertices] == [0.0, 1.0, 1.0]
    conn.close()


# ---------------------------------------------------------------------------
# Labels, enums and custom fields
# ---------------------------------------------------------------------------

def test_label_tracks_and_keyframes(legacy):
    migration.migrate_project(legacy)
    conn = store.open_store(
        store.stage_path(legacy, store.LABELS, "t"), store.LABELS, "t")
    tracks = _rows(conn, "SELECT label_track_id, species_id FROM label_tracks "
                         "ORDER BY label_track_id")
    assert tracks == [{"label_track_id": 1, "species_id": 10},
                      {"label_track_id": 2, "species_id": -1}]
    keyframes = _rows(conn, "SELECT frame, stop FROM label_keyframes "
                            "WHERE label_track_id = 1 ORDER BY frame")
    assert keyframes == [{"frame": 3, "stop": 0}, {"frame": 9, "stop": 1}]
    conn.close()


def test_sex_and_age_become_enum_value_ids(legacy):
    """Stored by id, so a later rename cannot orphan them (§5.1)."""
    migration.migrate_project(legacy)
    conn = store.open_store(
        store.stage_path(legacy, store.LABELS, "t"), store.LABELS, "t")
    attributes = json.loads(_rows(
        conn, "SELECT attributes FROM label_tracks WHERE label_track_id = 1"
    )[0]["attributes"])
    assert attributes["sex"] == 1     # female
    assert attributes["age"] == 1     # adult
    assert attributes["collar_id"] == "R-114"
    conn.close()


def test_occlusion_becomes_a_keyframe_enum_value(legacy):
    migration.migrate_project(legacy)
    conn = store.open_store(
        store.stage_path(legacy, store.LABELS, "t"), store.LABELS, "t")
    attributes = json.loads(_rows(
        conn, "SELECT attributes FROM label_keyframes "
              "WHERE label_track_id = 1 AND frame = 3")[0]["attributes"])
    assert attributes["occlusion"] == 1   # partially
    assert attributes["posture"] == "standing"
    conn.close()


def test_custom_fields_are_imported_with_mapped_scopes(legacy):
    migration.migrate_project(legacy)
    conn = store.open_store(store.project_path(legacy), store.PROJECT)
    fields = {r["name"]: r["scope"] for r in _rows(
        conn, "SELECT name, scope FROM field_schema")}
    assert fields["collar_id"] == "track"
    assert fields["posture"] == "detection"   # 5.x "keyframe" scope
    conn.close()


def test_unseen_enum_value_is_appended_not_dropped(tmp_path):
    root = str(tmp_path / "flight")
    _write(os.path.join(root, "labels_t", "labels.json"), json.dumps({
        "tracks": [{"track_id": 1, "species": "unknown", "sex": "hermaphrodite",
                    "age": "unknown", "keyframes": {
                        "1": {"x1": 0, "y1": 0, "x2": 1, "y2": 1,
                              "occlusion": "none"}}}]}))
    report = migration.migrate_project(root)

    conn = store.open_store(store.project_path(root), store.PROJECT)
    labels = [r["label"] for r in _rows(
        conn, "SELECT v.label FROM enum_values v JOIN enums e USING (enum_id) "
              "WHERE e.name = 'sex' ORDER BY v.value_id")]
    assert labels == ["unknown", "female", "male", "hermaphrodite"]
    assert report.counts.get("enum_values_appended") == 1
    conn.close()


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def test_segments_carry_pixel_and_world_polygons(legacy):
    migration.migrate_project(legacy)
    conn = store.open_store(
        store.stage_path(legacy, store.SEGMENTATION, "t"),
        store.SEGMENTATION, "t")
    rows = _rows(conn, "SELECT frame, polygon_px, polygon_geo, attributes "
                       "FROM segments")
    assert len(rows) == 1
    assert rows[0]["frame"] == 1
    assert json.loads(rows[0]["polygon_px"]) == [[[0, 0], [1, 0], [1, 1]]]
    assert json.loads(rows[0]["polygon_geo"])[0][0] == 10.0
    assert json.loads(rows[0]["attributes"])["prompt"] == "animal"
    conn.close()


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_migration_reports_what_it_did(legacy):
    report = migration.migrate_project(legacy)
    assert report.counts["detections_detector"] == 3
    assert report.counts["detections_manual"] == 2
    assert report.counts["detections_geo"] == 4
    assert report.counts["georef_failures"] == 1
    assert report.counts["tracks"] == 2
    assert report.counts["track_members"] == 3
    assert report.counts["label_tracks"] == 2
    assert report.counts["segments"] == 1


def test_migration_logs_progress(legacy):
    messages = []
    migration.migrate_project(legacy, log_fn=messages.append)
    assert any("Vocabulary" in m for m in messages)
    assert any("modality 't'" in m for m in messages)


def test_legacy_files_are_left_untouched(legacy):
    path = os.path.join(legacy, "detections_t", "detections.txt")
    with open(path, encoding="utf-8") as fh:
        before = fh.read()
    migration.migrate_project(legacy)
    with open(path, encoding="utf-8") as fh:
        assert fh.read() == before


def test_both_modalities_are_migrated(tmp_path):
    root = str(tmp_path / "flight")
    for modality in ("t", "w"):
        _write(os.path.join(root, f"detections_{modality}", "detections.txt"),
               "1 10.0 20.0 30.0 40.0 0.9 0\n")
    migration.migrate_project(root)
    for modality in ("t", "w"):
        conn = store.open_store(
            store.stage_path(root, store.DETECTIONS, modality),
            store.DETECTIONS, modality)
        assert len(_rows(conn, "SELECT * FROM detections")) == 1
        conn.close()


def test_trex_folder_is_flagged_as_indistinguishable(tmp_path):
    """5.x wrote TRex detections in the detector's own format (§9)."""
    root = str(tmp_path / "flight")
    _write(os.path.join(root, "detections_t", "detections.txt"),
           "1 10.0 20.0 30.0 40.0 0.9 0\n")
    os.makedirs(os.path.join(root, "tracks_pixel_t"), exist_ok=True)
    report = migration.migrate_project(root)
    assert any("TRex" in w for w in report.warnings)


def test_migration_refuses_when_a_store_already_exists(legacy):
    """Migration inserts; it never reconciles, so a second run would duplicate.

    The store may also have been written by the pipeline itself rather than by
    an earlier migration — that is the dangerous case, because the duplicate
    rows would sit alongside live ones.
    """
    first = migration.migrate_project(legacy)
    assert first.counts["detections_detector"] == 3

    second = migration.migrate_project(legacy)
    assert second.counts == {}
    assert any("already has a 6.0 store" in w for w in second.warnings)

    conn = store.open_store(
        store.stage_path(legacy, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    assert len(_rows(conn, "SELECT * FROM detections")) == 5   # not 10
    conn.close()


def test_migration_refuses_a_folder_the_pipeline_already_wrote(tmp_path):
    from bambi_wildlife_detection.core import detection_store

    root = str(tmp_path / "flight")
    _write(os.path.join(root, "detections_t", "detections.txt"),
           "1 10.0 20.0 30.0 40.0 0.9 0\n")
    detection_store.record_detections(root, "t", [
        {"frame": 1, "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0,
         "confidence": 0.9, "source_class": "0"}])

    report = migration.migrate_project(root)
    assert any("already has a 6.0 store" in w for w in report.warnings)

    conn = store.open_store(
        store.stage_path(root, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    assert len(_rows(conn, "SELECT * FROM detections")) == 1
    conn.close()


def test_existing_stores_lists_what_is_there(legacy):
    assert migration.existing_stores(legacy) == []
    migration.migrate_project(legacy)
    found = migration.existing_stores(legacy)
    assert "project.gpkg" in found
    assert "bambi_t/detections.gpkg" in found
