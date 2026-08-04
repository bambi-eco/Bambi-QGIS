# -*- coding: utf-8 -*-
"""Tests for the detection write path (EXCHANGE_FORMAT_PLAN.md §4.1, §10 Phase 2).

The load-bearing property here is per-source ownership: re-running one producer
must re-mint only its own rows. A global generation counter would mean any
producer re-running invalidated all the others, which is the behaviour 6.0 is
removing.
"""
import os

import pytest

from bambi_wildlife_detection.core import detection_store, store


def _rows(n=2, frame=1, cls="0"):
    return [{"frame": frame + i, "x1": 10.0 + i, "y1": 20.0, "x2": 30.0,
             "y2": 40.0, "confidence": 0.9, "source_class": cls}
            for i in range(n)]


def _stored(target_folder, modality="t"):
    conn = store.open_store(
        store.stage_path(target_folder, store.DETECTIONS, modality),
        store.DETECTIONS, modality)
    try:
        return [dict(r) for r in conn.execute(
            "SELECT detection_id, frame, species_id, source_id, source_class "
            "FROM detections ORDER BY detection_id")]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def test_records_rows_and_creates_the_stores(tmp_path):
    result = detection_store.record_detections(str(tmp_path), "t", _rows(3))
    assert result["written"] == 3
    assert os.path.isfile(store.project_path(str(tmp_path)))
    assert len(_stored(str(tmp_path))) == 3


def test_empty_input_is_allowed(tmp_path):
    assert detection_store.record_detections(str(tmp_path), "t", [])["written"] == 0


def test_unmapped_class_resolves_to_animal(tmp_path):
    detection_store.record_detections(str(tmp_path), "t", _rows(1, cls="7"))
    assert _stored(str(tmp_path))[0]["species_id"] == store.FALLBACK_SPECIES_ID


def test_raw_class_is_preserved(tmp_path):
    detection_store.record_detections(str(tmp_path), "t", _rows(1, cls="7"))
    assert _stored(str(tmp_path))[0]["source_class"] == "7"


def test_mapped_class_resolves_to_its_species(tmp_path):
    project = store.open_store(store.project_path(str(tmp_path)), store.PROJECT)
    source_id = detection_store.ensure_source(project, detection_store.DETECTOR)
    project.execute(
        "INSERT INTO class_mapping (source_id, source_class, species_id) "
        "VALUES (?, '7', 1)", (source_id,))
    project.commit()
    project.close()

    detection_store.record_detections(str(tmp_path), "t", _rows(1, cls="7"))
    assert _stored(str(tmp_path))[0]["species_id"] == 1


def test_modalities_are_independent(tmp_path):
    detection_store.record_detections(str(tmp_path), "t", _rows(2))
    detection_store.record_detections(str(tmp_path), "w", _rows(3))
    assert len(_stored(str(tmp_path), "t")) == 2
    assert len(_stored(str(tmp_path), "w")) == 3


def test_log_reports_what_was_written(tmp_path):
    messages = []
    detection_store.record_detections(
        str(tmp_path), "t", _rows(2), log_fn=messages.append)
    assert any("wrote 2 detection" in m for m in messages)


# ---------------------------------------------------------------------------
# Per-source ownership and generations (§4.1)
# ---------------------------------------------------------------------------

def test_rerunning_a_producer_replaces_only_its_own_rows(tmp_path):
    detection_store.record_detections(
        str(tmp_path), "t", _rows(2), kind=detection_store.DETECTOR)
    detection_store.record_detections(
        str(tmp_path), "t", _rows(3, frame=100), kind=detection_store.MANUAL)

    detection_store.record_detections(
        str(tmp_path), "t", _rows(1), kind=detection_store.DETECTOR)

    frames = sorted(r["frame"] for r in _stored(str(tmp_path)))
    assert frames == [1, 100, 101, 102]


def test_rerunning_reports_what_it_replaced(tmp_path):
    detection_store.record_detections(str(tmp_path), "t", _rows(4))
    result = detection_store.record_detections(str(tmp_path), "t", _rows(1))
    assert result["replaced"] == 4


def test_detection_ids_are_not_reused_across_runs(tmp_path):
    detection_store.record_detections(str(tmp_path), "t", _rows(2))
    first = [r["detection_id"] for r in _stored(str(tmp_path))]
    detection_store.record_detections(str(tmp_path), "t", _rows(2))
    second = [r["detection_id"] for r in _stored(str(tmp_path))]
    assert not set(first) & set(second)


def test_generation_advances_per_source(tmp_path):
    first = detection_store.record_detections(str(tmp_path), "t", _rows(1))
    second = detection_store.record_detections(str(tmp_path), "t", _rows(1))
    other = detection_store.record_detections(
        str(tmp_path), "t", _rows(1), kind=detection_store.TREX)
    assert second["generation"] == first["generation"] + 1
    assert other["generation"] == 1   # its own counter, untouched by the detector


def test_different_models_are_different_sources(tmp_path):
    detection_store.record_detections(str(tmp_path), "t", _rows(2), model="a.pt")
    detection_store.record_detections(str(tmp_path), "t", _rows(3), model="b.pt")
    assert len(_stored(str(tmp_path))) == 5


def test_detection_counts_by_kind(tmp_path):
    detection_store.record_detections(str(tmp_path), "t", _rows(2))
    detection_store.record_detections(
        str(tmp_path), "t", _rows(3), kind=detection_store.TREX)
    counts = detection_store.detection_counts(str(tmp_path), "t")
    assert counts == {"detector": 2, "trex": 3}


def test_detection_counts_of_an_empty_project(tmp_path):
    assert detection_store.detection_counts(str(tmp_path), "t") == {}


def test_ensure_source_is_idempotent(tmp_path):
    project = store.open_store(store.project_path(str(tmp_path)), store.PROJECT)
    try:
        first = detection_store.ensure_source(project, detection_store.DETECTOR)
        second = detection_store.ensure_source(project, detection_store.DETECTOR)
        assert first == second
    finally:
        project.close()


# ---------------------------------------------------------------------------
# Dual-write parity (§10 Phase 2 gate)
# ---------------------------------------------------------------------------

def _write_text(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# frame x1 y1 x2 y2 confidence class_id\n")
        for row in rows:
            fh.write(f"{row['frame']} {row['x1']:.2f} {row['y1']:.2f} "
                     f"{row['x2']:.2f} {row['y2']:.2f} "
                     f"{row['confidence']:.4f} {row['source_class']}\n")


def test_parity_holds_when_both_paths_saw_the_same_boxes(tmp_path):
    rows = _rows(5)
    path = os.path.join(str(tmp_path), "detections_t", "detections.txt")
    _write_text(path, rows)
    detection_store.record_detections(str(tmp_path), "t", rows)
    assert detection_store.compare_with_legacy_text(
        str(tmp_path), "t", path) is None


def test_parity_detects_a_missing_row(tmp_path):
    rows = _rows(5)
    path = os.path.join(str(tmp_path), "detections_t", "detections.txt")
    _write_text(path, rows)
    detection_store.record_detections(str(tmp_path), "t", rows[:4])
    assert "5 row(s)" in detection_store.compare_with_legacy_text(
        str(tmp_path), "t", path)


def test_parity_detects_a_moved_box(tmp_path):
    rows = _rows(2)
    path = os.path.join(str(tmp_path), "detections_t", "detections.txt")
    _write_text(path, rows)
    shifted = [dict(row) for row in rows]
    shifted[1]["x1"] += 5.0
    detection_store.record_detections(str(tmp_path), "t", shifted)
    assert "x1" in detection_store.compare_with_legacy_text(
        str(tmp_path), "t", path)


def test_parity_detects_a_changed_class(tmp_path):
    rows = _rows(2)
    path = os.path.join(str(tmp_path), "detections_t", "detections.txt")
    _write_text(path, rows)
    changed = [dict(row) for row in rows]
    changed[0]["source_class"] = "3"
    detection_store.record_detections(str(tmp_path), "t", changed)
    assert "class" in detection_store.compare_with_legacy_text(
        str(tmp_path), "t", path)


def test_parity_ignores_other_producers(tmp_path):
    rows = _rows(3)
    path = os.path.join(str(tmp_path), "detections_t", "detections.txt")
    _write_text(path, rows)
    detection_store.record_detections(str(tmp_path), "t", rows)
    detection_store.record_detections(
        str(tmp_path), "t", _rows(9, frame=500), kind=detection_store.MANUAL)
    assert detection_store.compare_with_legacy_text(
        str(tmp_path), "t", path) is None


def test_parity_on_an_empty_project_is_clean(tmp_path):
    path = os.path.join(str(tmp_path), "detections_t", "detections.txt")
    _write_text(path, [])
    assert detection_store.compare_with_legacy_text(
        str(tmp_path), "t", path) is None


def test_rows_from_legacy_text_skips_comments_and_junk(tmp_path):
    path = str(tmp_path / "detections.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# frame x1 y1 x2 y2 confidence class_id\n")
        fh.write("\n")
        fh.write("1 10.0 20.0 30.0 40.0 0.9 0\n")
        fh.write("not a row at all\n")
        fh.write("2 11.0 21.0 31.0 41.0 0.8 1\n")
    assert [r["frame"] for r in detection_store.rows_from_legacy_text(path)] == [1, 2]


@pytest.mark.parametrize("kind", [detection_store.DETECTOR,
                                  detection_store.TREX,
                                  detection_store.MANUAL])
def test_every_producer_kind_round_trips(tmp_path, kind):
    detection_store.record_detections(str(tmp_path), "t", _rows(2), kind=kind)
    assert detection_store.detection_counts(str(tmp_path), "t") == {kind: 2}
