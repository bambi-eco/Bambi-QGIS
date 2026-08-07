# -*- coding: utf-8 -*-
"""The C3 classification step, end to end.

Built on real TorchScript heads whose weights are chosen so the answers are
known in advance — the point is to prove the *pipeline* does what §3.3 says:
occlusion gates the frames, species fixes what the animal is, and sex reuses
exactly the species frames and picks its model from the species call.
"""
import json
import os
from typing import List

import numpy as np
import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import (
    classification_store, embedding_files, match_store, store, track_store,
)

torch = pytest.importorskip("torch")

DIM = 4


class _Decider(torch.nn.Module):
    """A head that reads the answer straight out of the feature vector.

    ``features[:, channel]`` selects the class, so a test can decide what any
    given crop is called simply by what it wrote into that channel.
    """

    def __init__(self, classes, channel=0):
        super().__init__()
        self.channel = channel
        self.n = len(classes)
        self.classes = torch.jit.Attribute(list(classes), List[str])

    def forward(self, features):
        picked = features[:, self.channel].long().clamp(0, self.n - 1)
        probabilities = torch.nn.functional.one_hot(
            picked, num_classes=self.n).float()
        return features[:, :2], probabilities


def _write_head(path, classes, channel=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.jit.script(_Decider(classes, channel)).save(path)
    return path


#: detection_id -> (track, occlusion, species, sex). Channel 0 drives
#: occlusion, channel 1 species, channel 2 sex.
PLAN = {
    1: (1, 0, 1, 1), 2: (1, 0, 1, 1), 3: (1, 0, 1, 1), 4: (1, 0, 1, 0),
    5: (2, 0, 1, 0), 6: (2, 0, 1, 0), 7: (2, 0, 1, 0), 8: (2, 0, 1, 0),
    9: (3, 1, 1, 1), 10: (3, 1, 1, 1), 11: (3, 0, 0, 0), 12: (3, 1, 1, 1),
}


def _build_modality(root, suffix, plan=None):
    """Detections, tracks, poses and embeddings for one camera."""
    plan = plan or PLAN

    conn = store.open_store(
        store.stage_path(root, store.DETECTIONS, suffix),
        store.DETECTIONS, suffix)
    with store.transaction(conn):
        for detection_id in sorted(plan):
            conn.execute(
                "INSERT INTO detections (frame, x1, y1, x2, y2, confidence, "
                "species_id, source_id) VALUES (?, 0, 0, 10, 10, 0.9, 0, 1)",
                (detection_id - 1,))
    conn.close()

    track_store.record_tracks(root, suffix, [
        {"track_id": spec[0], "detection_id": detection_id}
        for detection_id, spec in sorted(plan.items())])

    poses = os.path.join(root, f"poses_{suffix}.json")
    with open(poses, "w", encoding="utf-8") as fh:
        json.dump({"images": [{"imagefile": f"f{i:03d}.jpg"}
                              for i in range(len(plan))]}, fh)

    run_id = classification_store.start_embedding_run(
        root, suffix, backbone="fake", dim=DIM, crop_size=32, padding=0.1,
        projection="non_geo", folder=f"embeddings_{suffix}/non_geo")
    for detection_id, (_track, occ, species, sex) in plan.items():
        frame = detection_id - 1
        embedding_files.write_frame(
            embedding_files.frame_path(root, suffix, "non_geo", frame,
                                       f"f{frame:03d}.jpg"),
            {detection_id: np.array([occ, species, sex, 0.0],
                                    dtype=np.float32)})
    classification_store.record_embedded(root, suffix, run_id, list(plan))


@pytest.fixture
def flight(tmp_path):
    """Three tracks: two red deer (one male-ish) and one with occluded frames."""
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    _build_modality(root, "t")

    heads = os.path.join(root, "heads")
    _write_head(os.path.join(heads, "occ.pt"), ("clear", "occluded"), 0)
    _write_head(os.path.join(heads, "spe.pt"), ("roe deer", "red deer"), 1)
    _write_head(os.path.join(heads, "sex.pt"), ("female_juvenile", "male"), 2)
    return root


@pytest.fixture
def paired_flight(flight):
    """The same flight seen by both cameras, with every track matched.

    The RGB side carries the same feature channels, so a matched head reading
    ``[RGB, thermal]`` sees each value twice and lands on the same class — what
    is being tested is which animals the answer reaches, not the answer.
    """
    _build_modality(flight, "w")
    match_store.record_matches(flight, [
        {"track_id_t": track, "track_id_w": track, "shared": 4,
         "median_dist": 1.0, "conf_t": 0.9, "conf_w": 0.9,
         "pairs": [{"frame_t": detection_id - 1, "frame_w": detection_id - 1,
                    "detection_id_t": detection_id,
                    "detection_id_w": detection_id, "dist": 1.0}
                   for detection_id, spec in sorted(PLAN.items())
                   if spec[0] == track]}
        for track in (1, 2, 3)])
    return flight


def _models(root, **overrides):
    heads = os.path.join(root, "heads")
    models = {
        "occlusion": {"modality": "thermal", "model": "custom",
                      "path": os.path.join(heads, "occ.pt")},
        "species": {"modality": "thermal", "model": "custom",
                    "path": os.path.join(heads, "spe.pt")},
        "sex": {"modality": "thermal", "model": "custom",
                "path": os.path.join(heads, "sex.pt"),
                "species": {"red deer": {"model": "custom",
                                         "path": os.path.join(heads, "sex.pt")}}},
    }
    for task, patch in overrides.items():
        if patch is None:
            models.pop(task, None)
        else:
            models[task].update(patch)
    return models


def _config(root, models=None, **overrides):
    config = {
        "target_folder": root,
        "classification_models": models or _models(root),
        "classification_backbone_dim": DIM,
        "classification_quorum": 0.5,
        "classification_min_frames": 1,
        "classification_frame_selection": "visible",
        "classification_unmatched": "skip",
        "hf_token": "",
    }
    config.update(overrides)
    return config


def _run(root, models=None, **overrides):
    logs = []
    BambiProcessor().run_classification(
        _config(root, models, **overrides), log_fn=logs.append)
    return logs


# ---------------------------------------------------------------------------
# The hierarchy
# ---------------------------------------------------------------------------

def test_occlusion_is_recorded_per_frame_and_never_voted(flight):
    _run(flight)

    frames = classification_store.frame_predictions(flight, "t", "occlusion")
    assert len(frames) == 12
    # It selects which frames the others may use; it does not describe the
    # animal, so it produces no per-track call.
    assert classification_store.track_predictions(
        flight, "t", "occlusion") == []


def test_species_and_sex_are_voted_per_track(flight):
    _run(flight)

    species = {row["track_id"]: row["label"] for row in
               classification_store.track_predictions(flight, "t", "species")}
    # Track 3 is occluded on three frames of four; its one clear frame is the
    # roe deer one, so that is what it is called — the occluded majority never
    # gets a say.
    assert species == {1: "red deer", 2: "red deer", 3: "roe deer"}


def test_the_vote_margin_is_kept(flight):
    _run(flight)

    sex = {row["track_id"]: row for row in
           classification_store.track_predictions(flight, "t", "sex")}
    # Track 1: three male frames of four clear ones.
    assert sex[1]["label"] == "male"
    assert (sex[1]["votes"], sex[1]["n"]) == (3, 4)
    assert sex[2]["label"] == "female_juvenile"


def test_occluded_frames_do_not_vote(flight):
    """Track 3 is occluded on three of four frames; only the clear one votes,
    and it says roe deer."""
    _run(flight)

    species = {row["track_id"]: row for row in
               classification_store.track_predictions(flight, "t", "species")}
    assert species[3]["n"] == 1


def test_sex_reuses_exactly_the_species_frames(flight):
    """The paper is explicit that sex reuses those frames, not its own set."""
    _run(flight)

    species = {r["track_id"]: r["n"] for r in
               classification_store.track_predictions(flight, "t", "species")}
    sex = {r["track_id"]: r["n"] for r in
           classification_store.track_predictions(flight, "t", "sex")}
    for track_id in sex:
        assert sex[track_id] == species[track_id]


def test_the_frame_source_is_recorded(flight):
    _run(flight)

    row = classification_store.track_predictions(flight, "t", "species")[0]
    assert row["evidence"]["frames"] == "occlusion-head"


# ---------------------------------------------------------------------------
# Occlusion is optional (§5.2a)
# ---------------------------------------------------------------------------

def test_species_and_sex_run_without_any_occlusion_model(flight):
    models = _models(flight)
    models["occlusion"]["model"] = "off"
    _run(flight, models)

    species = classification_store.track_predictions(flight, "t", "species")
    assert len(species) == 3
    # With nothing to filter by, every frame votes — and that is said out loud.
    assert species[0]["evidence"]["frames"] == "all-frames-unfiltered"


def test_all_frames_ignores_the_occlusion_head(flight):
    _run(flight, classification_frame_selection="all")

    species = {r["track_id"]: r for r in
               classification_store.track_predictions(flight, "t", "species")}
    assert species[3]["n"] == 4          # the occluded frames voted too
    assert species[3]["evidence"]["frames"] == "all-frames"


def test_stored_annotations_are_used_when_the_head_is_off(flight):
    """A hand annotation outranks a 78 %-accurate head."""
    conn = store.open_store(
        store.stage_path(flight, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    with store.transaction(conn):
        for detection_id in (1, 2):
            conn.execute(
                "UPDATE detections SET attributes = ? WHERE detection_id = ?",
                (json.dumps({"occlusion": 0}), detection_id))
        for detection_id in (3, 4):
            conn.execute(
                "UPDATE detections SET attributes = ? WHERE detection_id = ?",
                (json.dumps({"occlusion": 1}), detection_id))
    conn.close()

    models = _models(flight)
    models["occlusion"]["model"] = "off"
    _run(flight, models)

    species = {r["track_id"]: r for r in
               classification_store.track_predictions(flight, "t", "species")}
    assert species[1]["n"] == 2
    assert species[1]["evidence"]["frames"] == "annotations"


def test_the_log_says_where_the_voting_frames_came_from(flight):
    models = _models(flight)
    models["occlusion"]["model"] = "off"
    logs = _run(flight, models)
    assert any("no occlusion labels" in line for line in logs)


# ---------------------------------------------------------------------------
# Sex is per species
# ---------------------------------------------------------------------------

def test_a_species_without_a_sex_model_is_left_alone(flight):
    """Sex is only defined for species someone has a classifier for — red deer
    today, and nothing else."""
    models = _models(flight)
    models["sex"]["species"] = {
        "wild boar": models["sex"]["species"]["red deer"]}
    _run(flight, models)

    # Nothing votes wild boar, so no track is sexed at all.
    assert classification_store.track_predictions(flight, "t", "sex") == []


def test_only_the_species_with_a_model_is_sexed(flight):
    _run(flight)

    sexed = {row["track_id"] for row in
             classification_store.track_predictions(flight, "t", "sex")}
    # Tracks 1 and 2 are red deer; track 3 votes roe deer, which has no model.
    assert sexed == {1, 2}


def test_sex_needs_the_species_call_first(flight):
    """Without species there is no key to choose a sex model with."""
    models = _models(flight)
    models["species"]["model"] = "off"
    _run(flight, models)

    assert classification_store.track_predictions(flight, "t", "sex") == []


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

def test_a_track_below_quorum_is_left_undecided_not_dropped(flight):
    # A unanimous track clears any quorum below 1.0, so demand more than
    # unanimity to make every call abstain.
    _run(flight, classification_quorum=1.0)

    assert classification_store.track_predictions(flight, "t", "species") == []
    # The frames were still classified — only the per-animal call abstained,
    # which is the difference between "we do not know" and "we did not look".
    assert classification_store.frame_predictions(flight, "t", "species")


def test_sex_does_not_run_when_species_abstained(flight):
    """Sex is chosen by species, so an undecided species leaves no key."""
    _run(flight, classification_quorum=1.0)
    assert classification_store.track_predictions(flight, "t", "sex") == []


def test_a_track_with_too_few_clear_frames_abstains(flight):
    _run(flight, classification_min_frames=3)

    species = {r["track_id"] for r in
               classification_store.track_predictions(flight, "t", "species")}
    assert 3 not in species          # only one clear frame


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_no_enabled_task_is_refused(flight):
    models = {task: {"model": "off"} for task in
              ("occlusion", "species", "sex")}
    with pytest.raises(ValueError, match="No classifier is enabled"):
        BambiProcessor().run_classification(_config(flight, models))


def test_missing_embeddings_are_refused(flight, tmp_path):
    bare = str(tmp_path / "bare")
    os.makedirs(bare)
    store.open_store(store.project_path(bare), store.PROJECT).close()
    with pytest.raises(ValueError, match="No Thermal embeddings"):
        BambiProcessor().run_classification(_config(bare, _models(flight)))


def test_a_missing_custom_model_is_reported(flight):
    models = _models(flight)
    models["occlusion"]["path"] = os.path.join(flight, "nope.pt")
    with pytest.raises(FileNotFoundError):
        BambiProcessor().run_classification(_config(flight, models))


def test_a_custom_model_without_a_path_is_reported(flight):
    models = _models(flight)
    models["occlusion"]["path"] = ""
    with pytest.raises(ValueError, match="no file was chosen"):
        BambiProcessor().run_classification(_config(flight, models))


def test_species_has_no_default_model_to_fall_back_on(flight):
    models = _models(flight)
    models["species"]["model"] = "default"
    with pytest.raises(ValueError, match="no default species classifier"):
        BambiProcessor().run_classification(_config(flight, models))


def test_rerunning_replaces_rather_than_appends(flight):
    _run(flight)
    _run(flight)
    assert len(classification_store.frame_predictions(
        flight, "t", "occlusion")) == 12


# ---------------------------------------------------------------------------
# Reaching the rest of the plugin
# ---------------------------------------------------------------------------

def _track_species(root):
    conn = store.open_store(
        store.stage_path(root, store.TRACKS, "t"), store.TRACKS, "t")
    try:
        return {int(r["track_id"]): int(r["species_id"]) for r in conn.execute(
            "SELECT track_id, species_id FROM tracks")}
    finally:
        conn.close()


def _detection_species(root):
    conn = store.open_store(
        store.stage_path(root, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    try:
        return {int(r["detection_id"]): int(r["species_id"])
                for r in conn.execute(
                    "SELECT detection_id, species_id FROM detections")}
    finally:
        conn.close()


def test_results_reach_the_tracks_and_detections(flight):
    """Without this the classifications would be invisible to the exports,
    the layers, the analytics and the labelling tool."""
    from bambi_wildlife_detection.core import label_store

    _run(flight)
    red_deer = label_store.vocabulary(flight)["species_by_name"]["red deer"]

    assert _track_species(flight)[1] == red_deer
    # The exporters filter on the detection's species, so it has to land there
    # as well.
    assert _detection_species(flight)[1] == red_deer


def test_writing_cannot_be_switched_off(flight):
    """There is no such switch any more: a result nobody can see is not a
    result, and the store keeps the full record regardless."""
    _run(flight, classification_write_results=False)

    assert any(species for species in _track_species(flight).values())


# ---------------------------------------------------------------------------
# Which camera's animals a stage is about
# ---------------------------------------------------------------------------

def test_a_thermal_stage_labels_only_the_thermal_animals(flight):
    _run(flight)

    assert classification_store.track_predictions(flight, "t", "species")
    assert classification_store.track_predictions(flight, "w", "species") == []


def test_an_rgb_stage_labels_only_the_rgb_animals(paired_flight):
    root = paired_flight
    models = _models(root)
    for spec in models.values():
        spec["modality"] = "rgb"
    _run(root, models)

    assert classification_store.track_predictions(root, "t", "species") == []
    assert classification_store.track_predictions(root, "w", "species")


def test_a_matched_stage_labels_both_sides_of_the_pair(paired_flight):
    """They are one animal, so both carry the call."""
    root = paired_flight
    models = _models(root)
    for spec in models.values():
        spec["modality"] = "matched"
    _run(root, models)

    thermal = {row["track_id"]: row["label"] for row in
               classification_store.track_predictions(root, "t", "species")}
    rgb = {row["track_id"]: row["label"] for row in
           classification_store.track_predictions(root, "w", "species")}
    assert thermal and rgb
    # Concatenation is [RGB, thermal] whichever side is primary, so the two
    # passes see the same vector and must agree rather than merely coincide.
    partners = match_store.partner_tracks(root, "t")
    for track_t, label in thermal.items():
        if track_t in partners:
            assert rgb.get(partners[track_t]) == label


def test_stages_can_target_different_cameras(paired_flight):
    root = paired_flight
    models = _models(root)
    models["occlusion"]["modality"] = "rgb"
    models["species"]["modality"] = "thermal"
    _run(root, models)

    assert classification_store.frame_predictions(root, "w", "occlusion")
    assert classification_store.frame_predictions(root, "t", "occlusion") == []
    assert classification_store.track_predictions(root, "t", "species")


# ---------------------------------------------------------------------------
# One step per classifier
# ---------------------------------------------------------------------------

def test_a_single_task_can_be_run_alone(flight):
    _run(flight, classification_only=["occlusion"])

    assert classification_store.frame_predictions(flight, "t", "occlusion")
    assert classification_store.track_predictions(flight, "t", "species") == []


def test_a_later_step_reads_what_an_earlier_one_stored(flight):
    """The steps run separately, so species must reach sex through the store
    rather than through the same run."""
    _run(flight, classification_only=["occlusion"])
    _run(flight, classification_only=["species"])
    _run(flight, classification_only=["sex"])

    sexed = {row["track_id"] for row in
             classification_store.track_predictions(flight, "t", "sex")}
    assert sexed == {1, 2}      # the red deer, which have a sex model


def test_the_stored_occlusion_still_selects_the_voting_frames(flight):
    """Occlusion ran in its own step; species must still honour it."""
    _run(flight, classification_only=["occlusion"])
    logs = _run(flight, classification_only=["species"])

    species = {r["track_id"]: r for r in
               classification_store.track_predictions(flight, "t", "species")}
    assert species[3]["n"] == 1          # three of its four frames are occluded
    assert species[3]["evidence"]["frames"] == "occlusion-head"
    assert any("stored frame call" in line for line in logs)


def test_sex_without_a_species_call_says_what_is_missing(flight):
    logs = _run(flight, classification_only=["sex"])

    assert classification_store.track_predictions(flight, "t", "sex") == []
    assert any("Run the species classifier on this camera" in line
               for line in logs)


def test_a_species_synced_from_the_other_camera_counts(paired_flight):
    """Which is what makes syncing a way out rather than a dead end: the label
    lands on the animal, and the demographic head reads it from there."""
    from bambi_wildlife_detection.core import label_store, label_sync

    root = paired_flight
    _run(root, classification_only=["species"])       # thermal only
    label_sync.sync_all(root, "t", "w")

    models = _models(root)
    models["sex"]["modality"] = "rgb"
    _run(root, models, classification_only=["sex"])

    red_deer = label_store.vocabulary(root)["species_by_name"]["red deer"]
    assert track_store.track_species(root, "w")[1] == red_deer
    assert classification_store.track_predictions(root, "w", "sex")


def test_running_a_switched_off_task_is_refused_by_name(flight):
    models = _models(flight)
    models["species"]["model"] = "off"
    with pytest.raises(ValueError, match="species classifier is switched off"):
        BambiProcessor().run_classification(
            _config(flight, models, classification_only=["species"]))


def test_an_empty_restriction_runs_everything_enabled(flight):
    _run(flight, classification_only=[])
    assert classification_store.frame_predictions(flight, "t", "occlusion")
    assert classification_store.track_predictions(flight, "t", "species")


def test_the_stage_is_recorded(flight):
    from bambi_wildlife_detection.core import stages

    _run(flight)
    assert "classification" in [
        s for s, row in stages.states(flight, "t").items()
        if row["state"] == stages.COMPLETE]
