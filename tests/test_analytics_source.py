# -*- coding: utf-8 -*-
"""Point selection for the survey analytics (EXCHANGE_FORMAT_PLAN.md §8.2).

The question every analytic starts from is *which animals count?*. 5.x answered
it implicitly by reading whatever files were present, so a density figure gave
no indication of what went into it. These tests pin the three rules that make
the answer explicit, and that the answer travels with the result.
"""
import pytest

from bambi_wildlife_detection.core import (
    analytics_source, detection_store, label_store, store, track_store)


@pytest.fixture
def survey(tmp_path):
    """Two roe deer detections on one track, one chamois, one false positive."""
    root = str(tmp_path)

    # Map the detector's raw classes first, so species resolve on the way in.
    project = store.open_store(store.project_path(root), store.PROJECT)
    source_id = detection_store.ensure_source(project, detection_store.DETECTOR)
    project.executemany(
        "INSERT INTO class_mapping (source_id, source_class, species_id) "
        "VALUES (?, ?, ?)",
        [(source_id, "1", 1), (source_id, "5", 5), (source_id, "0", -2)])
    project.commit()
    project.close()

    detection_store.record_detections(root, "t", [
        {"frame": 0, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.9, "source_class": "1"},
        {"frame": 1, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.9, "source_class": "1"},
        {"frame": 0, "x1": 3.0, "y1": 3.0, "x2": 4.0, "y2": 4.0,
         "confidence": 0.8, "source_class": "5"},
        {"frame": 1, "x1": 9.0, "y1": 9.0, "x2": 9.5, "y2": 9.5,
         "confidence": 0.2, "source_class": "0"},
    ])

    ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
    track_store.record_georeference(root, "t", [
        {"detection_id": i, "gx1": 10.0 * n, "gy1": 20.0 * n, "gz1": 0.0,
         "gx2": 10.0 * n + 2.0, "gy2": 20.0 * n + 2.0, "gz2": 0.0}
        for n, i in enumerate(ids)])
    track_store.record_tracks(root, "t", [
        {"track_id": 1, "detection_id": ids[0]},
        {"track_id": 1, "detection_id": ids[1]},
        {"track_id": 2, "detection_id": ids[2]},
        {"track_id": 3, "detection_id": ids[3]},
    ])
    return root


# ---------------------------------------------------------------------------
# Rule 1: not-an-animal never counts
# ---------------------------------------------------------------------------

def test_false_positives_are_excluded(survey):
    """A labelled false positive in a density estimate biases it upward."""
    rows, provenance = analytics_source.load_rows(survey, "t")
    assert all(row["species_id"] != store.NOT_AN_ANIMAL_SPECIES_ID
               for row in rows)
    assert provenance["excluded_not_an_animal"] == 1


def test_unknown_and_animal_still_count(survey):
    """The distinction is determinacy, not presence."""
    detection_store.record_detections(survey, "t", [
        {"frame": 5, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.5, "source_class": "?"}], kind=detection_store.MANUAL)
    ids = [d["detection_id"] for d in track_store.load_detections(survey, "t")]
    track_store.record_georeference(survey, "t", [
        {"detection_id": i, "gx1": 1.0, "gy1": 1.0, "gz1": 0.0,
         "gx2": 2.0, "gy2": 2.0, "gz2": 0.0} for i in ids])

    rows, _ = analytics_source.load_rows(survey, "t")
    assert any(row["species_id"] == store.FALLBACK_SPECIES_ID for row in rows)


def test_species_options_exclude_the_false_positive_class(survey):
    ids = {entry["species_id"] for entry in
           analytics_source.species_options(survey)}
    assert store.NOT_AN_ANIMAL_SPECIES_ID not in ids
    assert 0 in ids and -1 in ids       # animal and unknown are offerable


# ---------------------------------------------------------------------------
# Rule 2: one tracker run, plus the manual run
# ---------------------------------------------------------------------------

def test_the_active_tracker_run_is_recorded(survey):
    _, provenance = analytics_source.load_rows(survey, "t")
    assert provenance["tracker_run"]["kind"] == "builtin"


def test_manual_tracks_are_included_by_default(survey):
    """Labels are usually animals the detector missed (§8.2)."""
    label_store.save_tracks(survey, "t", [{
        "label_track_id": 1, "species_id": 1,
        "keyframes": [{"frame": 3, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0}],
    }])
    label_store.materialise(survey, "t")
    ids = [d["detection_id"] for d in track_store.load_detections(survey, "t")]
    track_store.record_georeference(survey, "t", [
        {"detection_id": i, "gx1": 1.0, "gy1": 1.0, "gz1": 0.0,
         "gx2": 2.0, "gy2": 2.0, "gz2": 0.0} for i in ids])

    points, provenance = analytics_source.load_points(survey, "t", "tracks")
    assert provenance["manual_run_included"] is True
    # Two detector tracks plus the manual one; the false-positive track is
    # excluded by rule 1, so it never reaches the count.
    assert len(points) == 3


def test_manual_tracks_can_be_excluded(survey):
    label_store.save_tracks(survey, "t", [{
        "label_track_id": 1, "species_id": 1,
        "keyframes": [{"frame": 3, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0}],
    }])
    label_store.materialise(survey, "t")

    _, provenance = analytics_source.load_rows(survey, "t", include_manual=False)
    assert provenance["manual_run_included"] is False


def test_superseded_tracks_are_reported(survey):
    """An imported label replaces the tracker track it came from."""
    conn = store.open_store(
        store.stage_path(survey, store.TRACKS, "t"), store.TRACKS, "t")
    original = conn.execute("SELECT MIN(track_id) AS t FROM tracks").fetchone()["t"]
    conn.close()

    label_store.save_tracks(survey, "t", [{
        "label_track_id": 1, "species_id": 1, "origin_track_id": original,
        "keyframes": [{"frame": 3, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0}],
    }])
    label_store.materialise(survey, "t")

    _, provenance = analytics_source.load_rows(survey, "t")
    assert provenance["superseded_tracks"] == [original]


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------

def test_detections_give_one_point_each(survey):
    points, provenance = analytics_source.load_points(survey, "t", "detections")
    assert len(points) == 3
    assert provenance["source"] == "detections"


def test_tracks_give_one_point_per_animal(survey):
    """An animal followed across many frames must count once."""
    points, provenance = analytics_source.load_points(survey, "t", "tracks")
    assert len(points) == 2          # the two-detection track collapses to one
    assert provenance["n_tracks"] == 2


def test_track_points_are_the_centroid(survey):
    points, _ = analytics_source.load_points(survey, "t", "tracks")
    xs = sorted(x for x, _ in points)
    assert xs[0] == pytest.approx(6.0)   # centres of (0,0)-(2,2) and (10,20)…


def test_untracked_detections_are_counted(survey):
    detection_store.record_detections(survey, "t", [
        {"frame": 7, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
         "confidence": 0.5, "source_class": "1"}], kind=detection_store.MANUAL)
    ids = [d["detection_id"] for d in track_store.load_detections(survey, "t")]
    track_store.record_georeference(survey, "t", [
        {"detection_id": i, "gx1": 1.0, "gy1": 1.0, "gz1": 0.0,
         "gx2": 2.0, "gy2": 2.0, "gz2": 0.0} for i in ids])

    _, provenance = analytics_source.load_points(survey, "t", "tracks")
    assert provenance["untracked_detections"] >= 1


def test_an_unknown_source_is_rejected(survey):
    with pytest.raises(analytics_source.AnalyticsError, match="source"):
        analytics_source.load_points(survey, "t", "vibes")


def test_a_project_without_detections_is_refused(tmp_path):
    with pytest.raises(analytics_source.AnalyticsError, match="No detections"):
        analytics_source.load_rows(str(tmp_path), "t")


# ---------------------------------------------------------------------------
# Species filtering and stratification
# ---------------------------------------------------------------------------

def test_species_filter_selects_one_species(survey):
    points, provenance = analytics_source.load_points(
        survey, "t", "detections", species_ids=[1])
    assert len(points) == 2
    assert provenance["species_filter"] == [1]


def test_stratification_groups_by_species_name(survey):
    """Possible only now that species_id has a stable meaning (§1.3)."""
    strata, provenance = analytics_source.load_points_by_species(
        survey, "t", "tracks")
    assert set(strata) == {"roe deer", "chamois"}
    assert provenance["strata"] == {"roe deer": 1, "chamois": 1}


def test_stratification_omits_species_with_no_points(survey):
    strata, _ = analytics_source.load_points_by_species(survey, "t")
    assert "wild boar" not in strata


# ---------------------------------------------------------------------------
# The filter travels with the result
# ---------------------------------------------------------------------------

def test_provenance_records_what_was_counted(survey):
    _, provenance = analytics_source.load_points(survey, "t", "tracks")
    assert provenance["modality"] == "t"
    assert provenance["n_points"] == 2
    assert "excluded_not_an_animal" in provenance
    assert "superseded_tracks" in provenance


def test_describe_filter_is_readable(survey):
    _, provenance = analytics_source.load_points(survey, "t", "tracks")
    text = analytics_source.describe_filter(provenance)
    assert "tracker run" in text
    assert "not-an-animal excluded" in text


def test_describe_filter_without_a_tracker_run(tmp_path):
    text = analytics_source.describe_filter({"tracker_run": None})
    assert "no tracker run" in text


# ---------------------------------------------------------------------------
# Multi-project pooling: match species by name, never by id (§8.2)
# ---------------------------------------------------------------------------

def _project_with(tmp_path, name, extra_species=()):
    from bambi_wildlife_detection.core.schema_editor import SchemaEditor

    root = str(tmp_path / name)
    with SchemaEditor(root) as editor:
        for species in extra_species:
            editor.add_species(species)
        editor.commit()
    return root


def test_species_names_resolve_to_this_projects_ids(tmp_path):
    a = _project_with(tmp_path, "a", ["wolf"])
    ids, missing = analytics_source.resolve_species_names(a, ["roe deer", "wolf"])
    assert ids == [1, 10]
    assert missing == []


def test_unknown_species_names_are_reported(tmp_path):
    a = _project_with(tmp_path, "a")
    ids, missing = analytics_source.resolve_species_names(a, ["roe deer", "wolf"])
    assert ids == [1]
    assert missing == ["wolf"]


def test_the_same_name_maps_to_different_ids_per_project(tmp_path):
    """Why pooling must translate rather than carry an id across."""
    a = _project_with(tmp_path, "a", ["wolf"])
    b = _project_with(tmp_path, "b", ["badger", "wolf"])

    a_ids, _ = analytics_source.resolve_species_names(a, ["wolf"])
    b_ids, _ = analytics_source.resolve_species_names(b, ["wolf"])
    assert a_ids != b_ids          # 10 in one project, 11 in the other


def test_vocabulary_comparison_reports_the_difference(tmp_path):
    a = _project_with(tmp_path, "a", ["wolf"])
    b = _project_with(tmp_path, "b", ["badger"])

    comparison = analytics_source.compare_vocabularies([a, b])
    assert "roe deer" in comparison["shared"]
    assert comparison["only_in"][a] == ["wolf"]
    assert comparison["only_in"][b] == ["badger"]


def test_identical_vocabularies_report_no_difference(tmp_path):
    a = _project_with(tmp_path, "a", ["wolf"])
    b = _project_with(tmp_path, "b", ["wolf"])
    assert analytics_source.compare_vocabularies([a, b])["only_in"] == {}


def test_vocabulary_comparison_ignores_the_false_positive_class(tmp_path):
    a = _project_with(tmp_path, "a")
    comparison = analytics_source.compare_vocabularies([a])
    assert "not-an-animal" not in comparison["all"]


def test_vocabulary_comparison_of_a_project_without_a_store(tmp_path):
    a = _project_with(tmp_path, "a")
    comparison = analytics_source.compare_vocabularies(
        [a, str(tmp_path / "never-run")])
    assert "roe deer" in comparison["shared"]
