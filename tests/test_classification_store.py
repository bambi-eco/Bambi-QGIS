# -*- coding: utf-8 -*-
"""Embedding bookkeeping and classification results.

Two things carry their weight here: that a re-run embeds only what is missing
(the vectors cost hours of GPU time), and that resetting one head does not take
the embeddings — or the other heads — with it.
"""
import pytest

from bambi_wildlife_detection.core import classification_store as cs
from bambi_wildlife_detection.core import store


@pytest.fixture
def root(tmp_path):
    return str(tmp_path)


def _run(root, **overrides):
    settings = dict(
        backbone="facebook/dinov3-vith16plus-pretrain-lvd1689m",
        dim=1280, crop_size=224, padding=0.1,
        projection="non_geo", folder="embeddings_t/non_geo",
        thermal_anchored=False, plugin_version="6.1.0")
    settings.update(overrides)
    return cs.start_embedding_run(root, "t", **settings), settings


# ---------------------------------------------------------------------------
# Empty store
# ---------------------------------------------------------------------------

def test_absent_store_reads_as_empty(root):
    assert not cs.has_store(root, "t")
    assert cs.active_embedding_run(root, "t") is None
    assert cs.embedded_ids(root, "t", 1) == set()
    assert cs.frame_predictions(root, "t", "sex") == []
    assert cs.track_predictions(root, "t", "sex") == []
    assert cs.tasks_present(root, "t") == []


def test_the_store_is_per_modality(root):
    _run(root)
    assert cs.has_store(root, "t")
    assert not cs.has_store(root, "w")


# ---------------------------------------------------------------------------
# Embedding runs
# ---------------------------------------------------------------------------

def test_run_settings_round_trip(root):
    run_id, settings = _run(root)
    active = cs.active_embedding_run(root, "t")

    assert active["run_id"] == run_id
    assert active["backbone"] == settings["backbone"]
    assert active["dim"] == 1280
    assert active["projection"] == "non_geo"
    assert active["folder"] == "embeddings_t/non_geo"
    assert active["plugin_version"] == "6.1.0"


def test_the_run_is_the_only_description_of_itself(root):
    """No sidecar file: everything needed to interpret the .npz files is here."""
    run_id, _ = _run(root)
    active = cs.active_embedding_run(root, "t")
    for field in ("backbone", "dim", "crop_size", "padding", "projection",
                  "thermal_anchored", "folder", "created_at"):
        assert field in active


def test_a_new_run_supersedes_the_previous(root):
    first, _ = _run(root)
    second, _ = _run(root, projection="geo_2k", folder="embeddings_t/geo_2k")

    assert cs.active_embedding_run(root, "t")["run_id"] == second
    # The old vectors are not discarded — switching projection back must not
    # cost another hour of GPU time.
    assert cs.embedded_ids(root, "t", first) == set()


def test_membership_is_scoped_to_its_run(root):
    first, _ = _run(root)
    cs.record_embedded(root, "t", first, [1, 2, 3])
    second, _ = _run(root, projection="geo_1k")

    assert cs.embedded_ids(root, "t", first) == {1, 2, 3}
    assert cs.embedded_ids(root, "t", second) == set()


# ---------------------------------------------------------------------------
# Resuming — the reason membership is stored at all
# ---------------------------------------------------------------------------

def test_pending_reports_only_what_is_missing(root):
    run_id, _ = _run(root)
    cs.record_embedded(root, "t", run_id, [1, 2])
    assert cs.pending_ids(root, "t", run_id, [1, 2, 3, 4]) == [3, 4]


def test_pending_preserves_the_given_order(root):
    run_id, _ = _run(root)
    cs.record_embedded(root, "t", run_id, [5])
    assert cs.pending_ids(root, "t", run_id, [9, 5, 7, 1]) == [9, 7, 1]


def test_recording_the_same_detection_twice_is_harmless(root):
    """Batches are recorded as they finish, so a resumed run can overlap."""
    run_id, _ = _run(root)
    cs.record_embedded(root, "t", run_id, [1, 2])
    cs.record_embedded(root, "t", run_id, [2, 3])
    assert cs.embedded_ids(root, "t", run_id) == {1, 2, 3}


def test_an_interrupted_run_can_be_resumed(root):
    run_id, settings = _run(root)
    cs.record_embedded(root, "t", run_id, [1, 2])

    resumed = cs.reuse_embedding_run(
        root, "t", settings["backbone"], settings["dim"],
        settings["crop_size"], settings["padding"], settings["projection"])
    assert resumed is not None and resumed["run_id"] == run_id
    assert cs.pending_ids(root, "t", resumed["run_id"], [1, 2, 3]) == [3]


@pytest.mark.parametrize("changed", [
    {"backbone": "facebook/other"},
    {"dim": 768},
    {"crop_size": 336},
    {"padding": 0.25},
    {"projection": "geo_2k"},
    {"thermal_anchored": True},
])
def test_settings_that_change_the_vectors_prevent_reuse(root, changed):
    _run(root)
    settings = dict(backbone="facebook/dinov3-vith16plus-pretrain-lvd1689m",
                    dim=1280, crop_size=224, padding=0.1,
                    projection="non_geo", thermal_anchored=False)
    settings.update(changed)
    assert cs.reuse_embedding_run(root, "t", **settings) is None


def test_settings_that_do_not_change_the_vectors_still_reuse(root):
    """A newer plugin version writes the same numbers, so it resumes."""
    run_id, settings = _run(root)
    resumed = cs.reuse_embedding_run(
        root, "t", settings["backbone"], settings["dim"],
        settings["crop_size"], settings["padding"], settings["projection"])
    assert resumed["run_id"] == run_id


def test_activating_an_older_run_switches_back(root):
    first, _ = _run(root)
    _run(root, projection="geo_2k")
    cs.activate_embedding_run(root, "t", first)
    assert cs.active_embedding_run(root, "t")["run_id"] == first


def test_a_vector_deleted_by_hand_can_be_forgotten(root):
    """Files win: a membership row whose .npz is gone must not be believed."""
    run_id, _ = _run(root)
    cs.record_embedded(root, "t", run_id, [1, 2, 3])
    cs.forget_embedded(root, "t", run_id, [2])
    assert cs.embedded_ids(root, "t", run_id) == {1, 3}
    assert cs.pending_ids(root, "t", run_id, [1, 2, 3]) == [2]


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def _frames(task="sex"):
    return [
        {"detection_id": 1, "label": "male", "class_index": 1, "prob": 0.91,
         "modality_in": "matched", "model": "sex_matched.pt"},
        {"detection_id": 2, "label": "female_juvenile", "class_index": 0,
         "prob": 0.62, "modality_in": "matched", "model": "sex_matched.pt"},
    ]


def test_frame_predictions_round_trip(root):
    _run(root)
    cs.record_frame_predictions(root, "t", "sex", _frames())
    rows = cs.frame_predictions(root, "t", "sex")

    assert [r["detection_id"] for r in rows] == [1, 2]
    # The head's own label is kept verbatim; mapping onto the project
    # vocabulary happens later and must not destroy the original.
    assert rows[1]["label"] == "female_juvenile"
    assert rows[0]["class_index"] == 1
    assert rows[0]["prob"] == pytest.approx(0.91)


def test_rerunning_a_head_replaces_rather_than_appends(root):
    _run(root)
    cs.record_frame_predictions(root, "t", "sex", _frames())
    cs.record_frame_predictions(root, "t", "sex", [
        {"detection_id": 1, "label": "female_juvenile", "class_index": 0,
         "prob": 0.55, "modality_in": "rgb", "model": "sex_rgb.pt"}])

    rows = cs.frame_predictions(root, "t", "sex")
    assert len(rows) == 1
    assert rows[0]["label"] == "female_juvenile"


def test_tasks_do_not_disturb_each_other(root):
    _run(root)
    cs.record_frame_predictions(root, "t", "sex", _frames())
    cs.record_frame_predictions(root, "t", "occlusion", [
        {"detection_id": 1, "label": "clear", "class_index": 0, "prob": 0.8,
         "modality_in": "matched", "model": "occlusion_matched.pt"}])

    assert len(cs.frame_predictions(root, "t", "sex")) == 2
    assert len(cs.frame_predictions(root, "t", "occlusion")) == 1


def test_track_call_keeps_its_vote(root):
    _run(root)
    cs.record_track_predictions(root, "t", "sex", [
        {"track_id": 4, "label": "male", "votes": 106, "n": 115,
         "fraction": 106 / 115, "modality_in": "matched",
         "model": "sex_matched.pt", "evidence": {"frames": "occlusion-head"}}])

    row = cs.track_predictions(root, "t", "sex")[0]
    # The margin is the point: "male 106/115" is what makes a borderline
    # individual reviewable, and it allows re-voting at another quorum.
    assert (row["votes"], row["n"]) == (106, 115)
    assert row["fraction"] == pytest.approx(106 / 115)
    assert row["evidence"] == {"frames": "occlusion-head"}


def test_unreadable_evidence_degrades_to_empty(root):
    _run(root)
    cs.record_track_predictions(root, "t", "sex", [
        {"track_id": 1, "label": "male", "votes": 3, "n": 4, "fraction": 0.75}])
    assert cs.track_predictions(root, "t", "sex")[0]["evidence"] == {}


def test_life_stage_needs_no_frame_rows(root):
    """It is read from box area per flight, not from features per frame."""
    _run(root)
    cs.record_track_predictions(root, "t", cs.LIFE_STAGE, [
        {"track_id": 5, "label": "juvenile", "votes": 1, "n": 1,
         "fraction": 1.0, "evidence": {"area": 673.0, "z": -2.4}}])

    assert cs.frame_predictions(root, "t", cs.LIFE_STAGE) == []
    assert cs.track_predictions(root, "t", cs.LIFE_STAGE)[0]["label"] == \
        "juvenile"


def test_occlusion_is_not_a_voted_task(root):
    # It selects the frames the others vote over rather than describing the
    # animal, so it never produces a per-track call.
    assert "occlusion" not in cs.VOTED_TASKS
    assert set(cs.VOTED_TASKS) == {"species", "sex"}


# ---------------------------------------------------------------------------
# Clearing one task
# ---------------------------------------------------------------------------

def test_clearing_a_task_keeps_the_embeddings(root):
    run_id, _ = _run(root)
    cs.record_embedded(root, "t", run_id, [1, 2])
    cs.record_frame_predictions(root, "t", "sex", _frames())
    cs.record_track_predictions(root, "t", "sex", [
        {"track_id": 1, "label": "male", "votes": 2, "n": 2, "fraction": 1.0}])

    removed = cs.clear_task(root, "t", "sex")

    assert removed == {"frames": 2, "tracks": 1}
    assert cs.frame_predictions(root, "t", "sex") == []
    # Re-running one head must not cost the hours the vectors took.
    assert cs.embedded_ids(root, "t", run_id) == {1, 2}


def test_clearing_a_task_keeps_the_other_tasks(root):
    _run(root)
    cs.record_frame_predictions(root, "t", "sex", _frames())
    cs.record_frame_predictions(root, "t", "occlusion", [
        {"detection_id": 1, "label": "clear", "class_index": 0, "prob": 0.8}])

    cs.clear_task(root, "t", "sex")
    assert len(cs.frame_predictions(root, "t", "occlusion")) == 1


def test_clearing_an_absent_task_is_harmless(root):
    _run(root)
    assert cs.clear_task(root, "t", "species") == {"frames": 0, "tracks": 0}


def test_tasks_present_lists_only_voted_results(root):
    _run(root)
    cs.record_frame_predictions(root, "t", "occlusion", [
        {"detection_id": 1, "label": "clear", "class_index": 0, "prob": 0.8}])
    cs.record_track_predictions(root, "t", "sex", [
        {"track_id": 1, "label": "male", "votes": 2, "n": 2, "fraction": 1.0}])

    # occlusion has frame rows but no track call, so it is not "present"
    # as a per-animal answer.
    assert cs.tasks_present(root, "t") == ["sex"]


# ---------------------------------------------------------------------------
# Stage wiring
# ---------------------------------------------------------------------------

def test_classification_is_a_per_modality_stage_kind(root):
    path = store.stage_path(root, store.CLASSIFICATION, "t")
    assert path.endswith("classification.gpkg")
    assert "bambi_t" in path


def test_the_classification_stage_does_not_own_the_file(root):
    """Resetting one head must not delete the vectors every head reads."""
    from bambi_wildlife_detection.core import stages

    assert stages.STAGE_STORE_KIND.get("embeddings") == store.CLASSIFICATION
    assert "classification" not in stages.STAGE_STORE_KIND
