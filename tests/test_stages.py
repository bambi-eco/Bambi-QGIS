# -*- coding: utf-8 -*-
"""Stage state and the dependency cascade (EXCHANGE_FORMAT_PLAN.md §7).

The cascade replaces the hand-maintained cleanup of latent bug 3: 5.x deleted
``tracks_{m}`` and ``tracks_pixel_{m}`` when labels replaced the detections, and
left the perpendicular distances, population estimates and exported media stale
and unmarked.
"""
import os

import pytest

from bambi_wildlife_detection.core import stages, store


@pytest.fixture
def project(tmp_path):
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    return root


def _make_output(root, stage, modality="t"):
    """Create whatever stages.has_output() looks for."""
    kind = stages.STAGE_STORE_KIND.get(stage)
    if kind:
        store.open_store(
            store.stage_path(root, kind, modality), kind, modality).close()
        return
    base = stages.STAGE_LEGACY_FOLDERS[stage][0]
    folder = os.path.join(root, f"{base}_{modality}")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "out.txt"), "w", encoding="utf-8") as fh:
        fh.write("x")


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------

def test_detection_dependents_reach_the_analytics():
    found = stages.dependents("detection")
    assert "georeference" in found
    assert "tracking" in found
    assert "population" in found
    assert "track_perpendicular" in found


def test_scene_products_do_not_depend_on_detections():
    """Pre-Processing is independent of any animal (§10.1)."""
    found = stages.dependents("detection")
    for stage in ("alfs", "export_geotiffs", "orthomosaic", "calculate_fov"):
        assert stage not in found


def test_frames_invalidate_everything_downstream():
    found = stages.dependents(stages.FRAMES)
    assert {"detection", "georeference", "tracking", "calculate_fov",
            "alfs", "orthomosaic"} <= set(found)


def test_orthomosaic_depends_on_the_geotiffs():
    assert "orthomosaic" in stages.dependents("export_geotiffs")


def test_a_leaf_has_no_dependents():
    assert stages.dependents("population") == []


def test_dependencies_are_reported():
    assert stages.dependencies("tracking") == ("georeference",)
    assert set(stages.dependencies("track_perpendicular")) == \
        {"tracking", "flight_route"}


def test_the_graph_has_no_cycles():
    """A cycle would deadlock the cascade — the reason labelling is not a node."""
    for stage in stages.known_stages():
        assert stage not in stages.dependents(stage)


# ---------------------------------------------------------------------------
# Recording state
# ---------------------------------------------------------------------------

def test_completing_a_stage_records_it(project):
    stages.mark_complete(project, "detection", "t", row_count=42)
    row = stages.states(project, "t")["detection"]
    assert row["state"] == stages.COMPLETE
    assert row["row_count"] == 42
    assert row["finished_at"]


def test_running_then_complete_keeps_the_start_time(project):
    stages.mark_running(project, "detection", "t")
    started = stages.states(project, "t")["detection"]["started_at"]
    stages.mark_complete(project, "detection", "t")
    assert stages.states(project, "t")["detection"]["started_at"] == started


def test_failure_is_recorded(project):
    stages.mark_failed(project, "detection", "t")
    assert stages.states(project, "t")["detection"]["state"] == stages.FAILED


def test_modalities_are_tracked_separately(project):
    stages.mark_complete(project, "detection", "t")
    assert "detection" not in stages.states(project, "w")


def test_states_of_a_project_without_a_store(tmp_path):
    assert stages.states(str(tmp_path)) == {}


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------

def test_completing_a_stage_marks_dependents_stale(project):
    stages.mark_complete(project, "tracking", "t")
    stages.mark_complete(project, "population", "t")

    affected = stages.mark_complete(project, "detection", "t")
    assert "tracking" in affected and "population" in affected
    assert stages.states(project, "t")["tracking"]["state"] == stages.STALE


def test_the_cascade_reaches_transitively(project):
    """5.x cleaned up two folders; everything further downstream was missed."""
    for stage in ("georeference", "tracking", "track_perpendicular",
                  "population"):
        stages.mark_complete(project, stage, "t")

    stages.mark_complete(project, "detection", "t")
    recorded = stages.states(project, "t")
    for stage in ("georeference", "tracking", "track_perpendicular",
                  "population"):
        assert recorded[stage]["state"] == stages.STALE


def test_unrelated_stages_are_not_marked_stale(project):
    stages.mark_complete(project, "alfs", "t")
    stages.mark_complete(project, "detection", "t")
    assert stages.states(project, "t")["alfs"]["state"] == stages.COMPLETE


def test_stale_results_are_flagged_not_deleted(project):
    _make_output(project, "tracking")
    stages.mark_complete(project, "tracking", "t")
    stages.mark_complete(project, "detection", "t")

    assert stages.states(project, "t")["tracking"]["state"] == stages.STALE
    assert stages.has_output(project, "tracking", "t")


def test_a_pending_stage_is_not_marked_stale(project):
    stages.mark_complete(project, "detection", "t")
    assert "tracking" not in stages.states(project, "t")


def test_stale_stages_are_listed(project):
    stages.mark_complete(project, "tracking", "t")
    stages.mark_complete(project, "detection", "t")
    assert stages.stale_stages(project, "t") == ["tracking"]


# ---------------------------------------------------------------------------
# Output detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stage", ["detection", "georeference", "tracking",
                                   "calculate_fov", "sam3_segmentation"])
def test_store_backed_stages_are_detected(project, stage):
    assert not stages.has_output(project, stage, "t")
    _make_output(project, stage)
    assert stages.has_output(project, stage, "t")


@pytest.mark.parametrize("stage", ["alfs", "export_geotiffs", "flight_route"])
def test_folder_backed_stages_are_detected(project, stage):
    assert not stages.has_output(project, stage, "t")
    _make_output(project, stage)
    assert stages.has_output(project, stage, "t")


def test_an_empty_legacy_folder_does_not_count(project):
    os.makedirs(os.path.join(project, "alfs_t"), exist_ok=True)
    assert not stages.has_output(project, "alfs", "t")


def test_completed_stages_are_listed(project):
    _make_output(project, "detection")
    _make_output(project, "alfs")
    assert stages.completed_stages(project, "t") == ["alfs", "detection"]


# ---------------------------------------------------------------------------
# Reconciliation — the files win (§7)
# ---------------------------------------------------------------------------

def test_deleting_a_stage_file_by_hand_is_noticed(project):
    """`rm bambi_t/tracks.gpkg` in Explorer must still give the right answer."""
    _make_output(project, "tracking")
    stages.mark_complete(project, "tracking", "t")

    os.remove(store.stage_path(project, store.TRACKS, "t"))
    changed = stages.reconcile(project, "t")

    assert changed == ["tracking"]
    assert stages.states(project, "t")["tracking"]["state"] == stages.PENDING


def test_reconciliation_cascades(project):
    _make_output(project, "georeference")
    _make_output(project, "tracking")
    stages.mark_complete(project, "georeference", "t")
    stages.mark_complete(project, "tracking", "t")

    os.remove(store.stage_path(project, store.GEOREFERENCED, "t"))
    stages.reconcile(project, "t")

    # tracking still has output, so it is stale rather than pending
    assert stages.states(project, "t")["tracking"]["state"] == stages.STALE


def test_reconciliation_demotes_an_output_less_dependent_to_pending(project):
    """'Stale' means out of date, not missing — a stage with no output is pending."""
    _make_output(project, "georeference")
    stages.mark_complete(project, "georeference", "t")
    stages.mark_complete(project, "tracking", "t")     # recorded, never wrote

    os.remove(store.stage_path(project, store.GEOREFERENCED, "t"))
    stages.reconcile(project, "t")
    assert stages.states(project, "t")["tracking"]["state"] == stages.PENDING


def test_reconciliation_leaves_intact_stages_alone(project):
    _make_output(project, "detection")
    stages.mark_complete(project, "detection", "t")
    assert stages.reconcile(project, "t") == []
    assert stages.states(project, "t")["detection"]["state"] == stages.COMPLETE


def test_reconciliation_without_a_store(tmp_path):
    assert stages.reconcile(str(tmp_path), "t") == []


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_removes_the_stage_output(project):
    _make_output(project, "tracking")
    stages.mark_complete(project, "tracking", "t")

    result = stages.reset(project, "tracking", "t")
    assert result["removed"]
    assert not stages.has_output(project, "tracking", "t")
    assert stages.states(project, "t")["tracking"]["state"] == stages.PENDING


def test_reset_removes_the_legacy_folders_too(project):
    _make_output(project, "tracking")
    os.makedirs(os.path.join(project, "tracks_t"), exist_ok=True)
    with open(os.path.join(project, "tracks_t", "tracks.csv"), "w") as fh:
        fh.write("x")

    stages.reset(project, "tracking", "t")
    assert not os.path.exists(os.path.join(project, "tracks_t"))


def test_reset_without_cascade_only_flags_dependents(project):
    _make_output(project, "detection")
    _make_output(project, "tracking")
    stages.mark_complete(project, "detection", "t")
    stages.mark_complete(project, "tracking", "t")

    result = stages.reset(project, "detection", "t")
    assert "tracking" in result["stale"]
    assert stages.has_output(project, "tracking", "t")   # kept, just flagged


def test_reset_with_cascade_removes_dependents(project):
    _make_output(project, "detection")
    _make_output(project, "georeference")
    _make_output(project, "tracking")

    stages.reset(project, "detection", "t", cascade=True)
    for stage in ("detection", "georeference", "tracking"):
        assert not stages.has_output(project, stage, "t")


def test_reset_of_a_stage_with_no_output(project):
    assert stages.reset(project, "tracking", "t")["removed"] == []


def test_reset_logs_what_it_did(project):
    _make_output(project, "tracking")
    messages = []
    stages.reset(project, "tracking", "t", log_fn=messages.append)
    assert any("Reset 'tracking'" in m for m in messages)


def test_reset_reports_a_locked_file(project, monkeypatch):
    """Windows locks a file open in the QGIS Browser (§13, Q1)."""
    _make_output(project, "tracking")

    def refuse(_path):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(os, "remove", refuse)
    with pytest.raises(stages.StageLockedError, match="QGIS Browser"):
        stages.reset(project, "tracking", "t")


# ---------------------------------------------------------------------------
# Fingerprints and description
# ---------------------------------------------------------------------------

def test_fingerprint_is_stable_for_the_same_inputs():
    config = {"min_confidence": 0.5, "unrelated": 1}
    assert stages.fingerprint(config, ["min_confidence"]) == \
        stages.fingerprint(config, ["min_confidence"])


def test_fingerprint_changes_with_a_relevant_setting():
    a = stages.fingerprint({"min_confidence": 0.5}, ["min_confidence"])
    b = stages.fingerprint({"min_confidence": 0.7}, ["min_confidence"])
    assert a != b


def test_fingerprint_ignores_irrelevant_settings():
    a = stages.fingerprint({"min_confidence": 0.5, "x": 1}, ["min_confidence"])
    b = stages.fingerprint({"min_confidence": 0.5, "x": 2}, ["min_confidence"])
    assert a == b


def test_fingerprint_tracks_upstream_generations():
    a = stages.fingerprint({}, [], upstream=["detection:1"])
    b = stages.fingerprint({}, [], upstream=["detection:2"])
    assert a != b


def test_describe_reports_every_stage(project):
    _make_output(project, "detection")
    stages.mark_complete(project, "detection", "t")
    summary = {row["stage"]: row for row in stages.describe(project, "t")}

    assert summary["detection"]["state"] == stages.COMPLETE
    assert summary["detection"]["on_disk"] is True
    assert summary["tracking"]["state"] == stages.PENDING
    assert "tracking" in summary["detection"]["dependents"]


def test_describe_prefers_the_disk_over_the_record(project):
    stages.mark_complete(project, "tracking", "t")   # recorded, but no output
    summary = {row["stage"]: row for row in stages.describe(project, "t")}
    assert summary["tracking"]["state"] == stages.PENDING


def test_describe_infers_completion_from_the_disk(project):
    """A 5.x folder has outputs but no recorded state."""
    _make_output(project, "alfs")
    summary = {row["stage"]: row for row in stages.describe(project, "t")}
    assert summary["alfs"]["state"] == stages.COMPLETE


# ---------------------------------------------------------------------------
# Stages recording their own completion
# ---------------------------------------------------------------------------

def test_record_stage_marks_complete_and_cascades(project):
    from bambi_wildlife_detection.bambi_processing import _record_stage

    stages.mark_complete(project, "tracking", "t")
    messages = []
    _record_stage({"target_folder": project}, "detection", "t",
                  row_count=7, log_fn=messages.append)

    recorded = stages.states(project, "t")
    assert recorded["detection"]["state"] == stages.COMPLETE
    assert recorded["detection"]["row_count"] == 7
    assert recorded["tracking"]["state"] == stages.STALE
    assert any("out of date" in m for m in messages)


def test_record_stage_is_a_no_op_without_a_store(tmp_path):
    """An un-migrated 5.x project must not gain a store as a side effect."""
    from bambi_wildlife_detection.bambi_processing import _record_stage

    root = str(tmp_path / "legacy")
    os.makedirs(root, exist_ok=True)
    _record_stage({"target_folder": root}, "detection", "t")
    assert not os.path.isfile(store.project_path(root))


def test_record_stage_without_a_target_folder():
    from bambi_wildlife_detection.bambi_processing import _record_stage

    _record_stage({}, "detection", "t")   # must not raise
