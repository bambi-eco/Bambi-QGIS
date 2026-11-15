# -*- coding: utf-8 -*-
"""The C4 life-stage step, end to end.

Needs no models: size is geometry. What is checked here is that the step picks
the right areas (metric where they exist), honours a sex call, and records the
evidence behind each verdict.
"""
import os

import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import (
    classification_store, life_stage, store, track_store,
)

# One juvenile among a tightly clustered herd, in pixel areas.
HERD = [400, 1000, 1010, 1020, 1030, 1040, 1050]


# The same herd as a camera without a size outlier sees it: nobody stands
# out, so the size test alone calls everyone adult.
FLAT_HERD = [980, 1000, 1010, 1020, 1030, 1040, 1050]


def _populate(root, modality, herd):
    """One track per herd entry, three boxes each, on *modality*."""
    conn = store.open_store(
        store.stage_path(root, store.DETECTIONS, modality),
        store.DETECTIONS, modality)
    with store.transaction(conn):
        for index, area in enumerate(herd):
            side = area ** 0.5
            # Three detections per track, so the median is well defined.
            for _ in range(3):
                conn.execute(
                    "INSERT INTO detections (frame, x1, y1, x2, y2, "
                    "confidence, species_id, source_id) "
                    "VALUES (?, 0, 0, ?, ?, 0.9, 0, 1)",
                    (index, side, side))
    conn.close()

    members = []
    detection_id = 1
    for index in range(len(herd)):
        for _ in range(3):
            members.append({"track_id": index + 1,
                            "detection_id": detection_id})
            detection_id += 1
    track_store.record_tracks(root, modality, members)


@pytest.fixture
def flight(tmp_path):
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    _populate(root, "t", HERD)
    return root


@pytest.fixture
def pair(tmp_path):
    """Both cameras tracked and matched track-for-track.

    The RGB herd has the juvenile at track 1; the thermal herd is the same
    animals seen without a size outlier, as a thermal cohort padded with warm
    blobs tends to be.
    """
    from bambi_wildlife_detection.core import match_store

    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    _populate(root, "t", FLAT_HERD)
    _populate(root, "w", HERD)
    match_store.record_matches(root, [
        {"track_id_t": index + 1, "track_id_w": index + 1, "shared": 3,
         "median_dist": 0.2, "conf_t": 0.9, "conf_w": 0.9}
        for index in range(len(HERD))])
    return root


def _config(root, **overrides):
    config = {
        "target_folder": root,
        "classification_models": {
            "life_stage": {"modality": "thermal",
                           "species": {name: {"model": "size"}
                                       for name in ("red deer", "roe deer",
                                                    "wild boar")}}},
        "life_stage_z": -2.0,
        "life_stage_iqr_factor": 0.5,
        "life_stage_min_individuals": 4,
    }
    config.update(overrides)
    return config


def _run(root, **overrides):
    logs = []
    BambiProcessor().run_life_stage(_config(root, **overrides),
                                    log_fn=logs.append)
    return logs


def _labels(root, modality="t"):
    return {row["track_id"]: row["label"] for row in
            classification_store.track_predictions(
                root, modality, classification_store.LIFE_STAGE)}


def _rows(root, modality):
    return {row["track_id"]: row for row in
            classification_store.track_predictions(
                root, modality, classification_store.LIFE_STAGE)}


MATCHED = {"life_stage": {"modality": "matched"}}


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def test_the_juvenile_is_flagged_and_the_rest_are_adults(flight):
    _run(flight)

    labels = _labels(flight)
    assert labels[1] == life_stage.JUVENILE
    assert all(labels[track] == life_stage.ADULT
               for track in range(2, len(HERD) + 1))


def test_every_individual_gets_a_row(flight):
    _run(flight)
    assert len(_labels(flight)) == len(HERD)


def test_the_evidence_behind_the_call_is_kept(flight):
    _run(flight)

    row = next(r for r in classification_store.track_predictions(
        flight, "t", classification_store.LIFE_STAGE) if r["track_id"] == 1)
    evidence = row["evidence"]
    for field in ("area", "z", "gap", "cohort_ratio", "frames", "source"):
        assert field in evidence
    assert evidence["z"] < -2.0
    assert evidence["cohort_ratio"] < 0.5


def test_size_is_one_measurement_not_a_vote(flight):
    """Recorded as a unanimous call of one, so the table stays uniform."""
    _run(flight)
    row = classification_store.track_predictions(
        flight, "t", classification_store.LIFE_STAGE)[0]
    assert (row["votes"], row["n"], row["fraction"]) == (1, 1, 1.0)


def test_the_log_names_the_juvenile(flight):
    logs = _run(flight)
    assert any("juvenile" in line and "track 1" in line for line in logs)


# ---------------------------------------------------------------------------
# Which areas
# ---------------------------------------------------------------------------

def test_pixel_areas_are_used_without_geo_referencing(flight):
    _run(flight)
    row = classification_store.track_predictions(
        flight, "t", classification_store.LIFE_STAGE)[0]
    assert row["evidence"]["source"] == life_stage.AREA_PIXEL


def test_the_metric_areas_are_what_actually_decide(flight):
    """Not just labelled 'orthorectified' - actually used.

    The pixel boxes and the world boxes are made to disagree about *which*
    animal is the outlier, so the verdict itself says which was measured. With
    proportional fixtures both answers coincide and the test proves nothing.
    """
    # By pixel area, track 7 is the small one (the fixture's last entry).
    # By world area, track 1 is - the opposite animal. The adults are given
    # slightly different sizes, because a herd of exactly equal animals has a
    # zero median-absolute-deviation and nothing can be an outlier of it.
    geo_areas = {1: 0.25, 2: 9.0, 3: 9.1, 4: 9.2, 5: 9.3, 6: 9.4, 7: 9.5}
    conn = store.open_store(
        store.stage_path(flight, store.GEOREFERENCED, "t"),
        store.GEOREFERENCED, "t")
    with store.transaction(conn):
        for detection_id in range(1, len(HERD) * 3 + 1):
            track = (detection_id - 1) // 3 + 1
            side = geo_areas[track] ** 0.5
            conn.execute(
                "INSERT INTO detections_geo (detection_id, gx1, gy1, gz1, "
                "gx2, gy2, gz2) VALUES (?, 0, 0, 0, ?, ?, 0)",
                (detection_id, side, side))
    conn.close()

    # Reverse the pixel areas so the pixel outlier is a *different* track.
    conn = store.open_store(
        store.stage_path(flight, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    with store.transaction(conn):
        for detection_id in range(1, len(HERD) * 3 + 1):
            track = (detection_id - 1) // 3
            side = HERD[len(HERD) - 1 - track] ** 0.5
            conn.execute(
                "UPDATE detections SET x2 = ?, y2 = ? WHERE detection_id = ?",
                (side, side, detection_id))
    conn.close()

    _run(flight)

    labels = _labels(flight)
    assert labels[1] == life_stage.JUVENILE      # the world-space outlier
    assert labels[len(HERD)] == life_stage.ADULT  # the pixel-space one


def test_metric_areas_are_preferred_when_available(flight):
    """Orthorectified areas are metric, so they are the better cue."""
    conn = store.open_store(
        store.stage_path(flight, store.GEOREFERENCED, "t"),
        store.GEOREFERENCED, "t")
    with store.transaction(conn):
        for detection_id in range(1, len(HERD) * 3 + 1):
            track = (detection_id - 1) // 3
            side = (HERD[track] ** 0.5) / 100.0
            conn.execute(
                "INSERT INTO detections_geo (detection_id, gx1, gy1, gz1, "
                "gx2, gy2, gz2) VALUES (?, 0, 0, 0, ?, ?, 0)",
                (detection_id, side, side))
    conn.close()

    _run(flight)
    row = classification_store.track_predictions(
        flight, "t", classification_store.LIFE_STAGE)[0]
    assert row["evidence"]["source"] == life_stage.AREA_GEO
    # Same verdict, different units.
    assert _labels(flight)[1] == life_stage.JUVENILE


# ---------------------------------------------------------------------------
# Interaction with sex
# ---------------------------------------------------------------------------

def test_a_male_is_an_adult_whatever_his_box_says(flight):
    """The cue that marked him is antlers, not size."""
    classification_store.record_track_predictions(flight, "t", "sex", [
        {"track_id": 1, "label": "male", "votes": 9, "n": 10,
         "fraction": 0.9}])
    _run(flight)

    assert _labels(flight)[1] == life_stage.ADULT


def test_a_female_call_does_not_protect_a_juvenile(flight):
    """'female_juvenile' is exactly the class size is meant to resolve."""
    classification_store.record_track_predictions(flight, "t", "sex", [
        {"track_id": 1, "label": "female_juvenile", "votes": 9, "n": 10,
         "fraction": 0.9}])
    _run(flight)

    assert _labels(flight)[1] == life_stage.JUVENILE


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_untracked_modality_is_refused(flight):
    """The stage's Input says which camera's animals it measures, so pointing
    it at one that was never tracked is an error rather than a silent skip."""
    config = _config(flight)
    config["classification_models"]["life_stage"]["modality"] = "rgb"
    with pytest.raises(ValueError, match="No RGB tracks"):
        BambiProcessor().run_life_stage(config)


def test_too_few_individuals_writes_nothing_and_says_so(flight):
    logs = _run(flight, life_stage_min_individuals=99)

    assert classification_store.track_predictions(
        flight, "t", classification_store.LIFE_STAGE) == []
    assert any("too few individuals" in line for line in logs)


def test_thresholds_are_honoured(flight):
    _run(flight, life_stage_z=-99.0)
    assert all(label == life_stage.ADULT for label in _labels(flight).values())


def test_rerunning_replaces_rather_than_appends(flight):
    _run(flight)
    _run(flight)
    assert len(_labels(flight)) == len(HERD)


def test_the_verdict_reaches_the_track_age_field(flight):
    from bambi_wildlife_detection.core import label_store

    _run(flight)
    juvenile = label_store.vocabulary(flight)["enum_ids"]["age"]["juvenile"]

    conn = store.open_store(
        store.stage_path(flight, store.TRACKS, "t"), store.TRACKS, "t")
    try:
        attributes = {int(r["track_id"]): r["attributes"]
                      for r in conn.execute(
                          "SELECT track_id, attributes FROM tracks")}
    finally:
        conn.close()

    import json as _json
    assert _json.loads(attributes[1])["age"] == juvenile


def test_writing_cannot_be_switched_off(flight):
    """The old switch is gone: the verdict always reaches the animal."""
    _run(flight, classification_write_results=False)
    conn = store.open_store(
        store.stage_path(flight, store.TRACKS, "t"), store.TRACKS, "t")
    try:
        rows = [r["attributes"] for r in conn.execute(
            "SELECT attributes FROM tracks")]
    finally:
        conn.close()
    assert any(row for row in rows)
    assert _labels(flight)[1] == life_stage.JUVENILE


# ---------------------------------------------------------------------------
# Size is the fallback, not the only route
# ---------------------------------------------------------------------------

def test_an_animal_a_classifier_called_is_left_alone(flight):
    """A model that looked at the animal beats a measurement of its box."""
    classification_store.record_track_predictions(
        flight, "t", classification_store.LIFE_STAGE, [
            {"track_id": 1, "label": "adult", "votes": 9, "n": 10,
             "fraction": 0.9, "model": "life_stage_matched.pt"}])
    logs = _run(flight)

    rows = {r["track_id"]: r for r in
            classification_store.track_predictions(
                flight, "t", classification_store.LIFE_STAGE)}
    # Track 1 is the size outlier, but the classifier already called it adult.
    assert rows[1]["label"] == life_stage.ADULT
    assert rows[1]["model"] == "life_stage_matched.pt"
    assert any("left as they are" in line for line in logs)


def test_size_still_fills_in_the_animals_no_model_called(flight):
    classification_store.record_track_predictions(
        flight, "t", classification_store.LIFE_STAGE, [
            {"track_id": 2, "label": "adult", "votes": 9, "n": 10,
             "fraction": 0.9, "model": "life_stage_matched.pt"}])
    _run(flight)

    rows = {r["track_id"]: r for r in
            classification_store.track_predictions(
                flight, "t", classification_store.LIFE_STAGE)}
    assert rows[1]["label"] == life_stage.JUVENILE
    assert rows[1]["model"] == classification_store.SIZE_MODEL
    assert rows[2]["model"] == "life_stage_matched.pt"


def test_a_called_animal_still_counts_towards_the_cohort(flight):
    """The comparison is against the whole herd, so excluding called animals
    from the statistics would shift everyone else's z-score."""
    classification_store.record_track_predictions(
        flight, "t", classification_store.LIFE_STAGE, [
            {"track_id": 4, "label": "adult", "votes": 9, "n": 10,
             "fraction": 0.9, "model": "m.pt"}])
    _run(flight)

    rows = {r["track_id"]: r for r in
            classification_store.track_predictions(
                flight, "t", classification_store.LIFE_STAGE)}
    assert rows[1]["label"] == life_stage.JUVENILE


def test_no_species_set_to_size_measures_nothing(flight):
    logs = _run(flight, classification_models={"life_stage": {"species": {}}})

    assert classification_store.track_predictions(
        flight, "t", classification_store.LIFE_STAGE) == []
    assert any("nothing to measure" in line for line in logs)


def _species(root, calls, modality="t"):
    """Record the species classifier's call for ``{track_id: species}``."""
    classification_store.record_track_predictions(root, modality, "species", [
        {"track_id": track_id, "label": label, "votes": 3, "n": 3,
         "fraction": 1.0} for track_id, label in calls.items()])


def test_only_the_species_set_to_size_are_measured(flight):
    """The choice is per species, in the same place the models are chosen."""
    _species(flight, {1: "red deer", 2: "red deer", 3: "red deer",
                      4: "red deer", 5: "wild boar", 6: "wild boar",
                      7: "wild boar"})
    _run(flight, life_stage_min_individuals=3, classification_models={
        "life_stage": {"modality": "thermal",
                       "species": {"red deer": {"model": "size"},
                                   "wild boar": {"model": "off"}}}})

    called = _labels(flight)
    assert {1, 2, 3, 4} <= set(called)            # red deer are measured
    assert not {5, 6, 7} & set(called)            # wild boar left uncalled


def test_the_stage_is_recorded(flight):
    from bambi_wildlife_detection.core import stages

    _run(flight)
    assert stages.states(flight, "t")["life_stage"]["state"] == stages.COMPLETE


def test_a_never_configured_selection_measures_by_default(flight):
    """What the Species… dialog shows before it is ever saved - size for
    every species - is what runs, instead of "nothing to measure"."""
    _species(flight, {1: "red deer", 2: "red deer", 3: "red deer",
                      4: "red deer", 5: "wild boar", 6: "wild boar",
                      7: "wild boar"})
    logs = _run(flight, life_stage_min_individuals=3, classification_models={
        "life_stage": {"modality": "thermal"}})

    assert not any("nothing to measure" in line for line in logs)
    called = _labels(flight)
    assert 1 in called and 5 in called


# ---------------------------------------------------------------------------
# Among its own kind
# ---------------------------------------------------------------------------

# Six red deer and four roe deer, all adults: a roe deer is half the size.
MIXED = [1000, 1010, 1020, 1030, 1040, 1050, 480, 490, 500, 510]
MIXED_SPECIES = {track: "red deer" if track <= 6 else "roe deer"
                 for track in range(1, 11)}


@pytest.fixture
def mixed(tmp_path):
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    _populate(root, "t", MIXED)
    return root


def test_an_adult_of_a_smaller_species_is_no_juvenile(mixed):
    """All animals of the flight were one cohort: among red deer every roe
    deer sits far below the median, and the largest of them - the one next
    to the gap - was called a juvenile."""
    _species(mixed, MIXED_SPECIES)
    logs = _run(mixed)

    labels = _labels(mixed)
    assert len(labels) == 10
    assert set(labels.values()) == {life_stage.ADULT}
    assert any("Life stage (red deer)" in line for line in logs)
    assert any("Life stage (roe deer)" in line for line in logs)


def test_without_species_the_flight_is_one_cohort_as_before(mixed):
    """Nothing says what the animals are, so they are compared with each
    other - which is all there is to go by."""
    _run(mixed)
    assert life_stage.JUVENILE in _labels(mixed).values()


def test_a_juvenile_is_found_among_its_own_species(tmp_path):
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    # red deer 1-6, roe deer 7-11 - of which 7 is a fawn
    _populate(root, "t", MIXED[:6] + [240, 480, 490, 500, 510])
    _species(root, {t: "red deer" if t <= 6 else "roe deer"
                    for t in range(1, 12)})
    _run(root)

    labels = _labels(root)
    assert [t for t, label in labels.items()
            if label == life_stage.JUVENILE] == [7]


def test_a_species_with_too_few_animals_is_not_judged(mixed):
    """Three animals are no herd to stand out from - and the other species'
    are no measure for them."""
    species = dict(MIXED_SPECIES)
    species[10] = "chamois"
    _species(mixed, species)
    logs = _run(mixed)

    labels = _labels(mixed)
    assert not {7, 8, 9, 10} & set(labels)       # 3 roe deer, 1 chamois
    assert set(labels) == {1, 2, 3, 4, 5, 6}
    assert any("Life stage (roe deer): too few individuals" in line
               for line in logs)


def test_the_species_the_detector_named_counts(mixed):
    """No classifier ran; the detector's class mapping called the boxes."""
    from bambi_wildlife_detection.core import label_store

    label_store.vocabulary(mixed, create=True)
    names = label_store.vocabulary(mixed)["species_by_name"]
    conn = store.open_store(
        store.stage_path(mixed, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    with store.transaction(conn):
        # frames 0-5 hold the red deer, 6-9 the roe deer (_populate)
        conn.execute("UPDATE detections SET species_id = ? WHERE frame < 6",
                     (names["red deer"],))
        conn.execute("UPDATE detections SET species_id = ? WHERE frame >= 6",
                     (names["roe deer"],))
    conn.close()
    _run(mixed)

    assert set(_labels(mixed).values()) == {life_stage.ADULT}


def test_a_track_that_is_no_animal_is_not_part_of_the_herd(flight):
    """Its box is a rock's: it is neither judged nor a measure for others.
    (It also used to count as "a species is known for this flight" - after
    which no animal without a species was measured at all.)"""
    conn = store.open_store(
        store.stage_path(flight, store.TRACKS, "t"), store.TRACKS, "t")
    conn.execute("UPDATE tracks SET species_id = ? WHERE track_id = 7",
                 (store.NOT_AN_ANIMAL_SPECIES_ID,))
    conn.commit()
    conn.close()
    _run(flight)

    labels = _labels(flight)
    assert 7 not in labels and len(labels) == 6
    assert labels[1] == life_stage.JUVENILE


def test_twins_are_both_juveniles(tmp_path):
    """Of two fawns of a size, the smaller one is a hair below its sibling,
    not below the herd - and was called an adult."""
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    _populate(root, "t", [400, 410, 1000, 1010, 1020, 1030, 1040, 1050])
    _run(root)

    labels = _labels(root)
    assert labels[1] == labels[2] == life_stage.JUVENILE
    assert set(list(labels.values())[2:]) == {life_stage.ADULT}
    evidence = {track: row["evidence"]
                for track, row in _rows(root, "t").items()}
    assert evidence[1]["separation"] == evidence[2]["separation"] == \
        pytest.approx(590.0)
    assert evidence[1]["gap"] == pytest.approx(10.0)
    assert evidence[1]["cohort_size"] == 8


# ---------------------------------------------------------------------------
# One animal, two cameras
# ---------------------------------------------------------------------------

def test_a_juvenile_found_on_one_camera_reaches_its_matched_track(pair):
    """Matching says the two tracks are one animal, so an outlier on the RGB
    side is evidence about the thermal track too."""
    logs = _run(pair, classification_models=MATCHED)

    assert _labels(pair, "w")[1] == life_stage.JUVENILE
    thermal = _labels(pair, "t")
    assert thermal[1] == life_stage.JUVENILE
    assert all(label == life_stage.ADULT
               for track_id, label in thermal.items() if track_id != 1)
    assert any("takes the juvenile call" in line for line in logs)


def test_the_carried_call_names_its_source(pair):
    _run(pair, classification_models=MATCHED)

    row = _rows(pair, "t")[1]
    # Still a size verdict, so a re-run recomputes rather than protects it.
    assert row["model"] == classification_store.SIZE_MODEL
    assert row["evidence"]["partner_track"] == 1
    assert row["evidence"]["partner_camera"] == "RGB"
    assert row["evidence"]["own_label"] == life_stage.ADULT
    assert row["evidence"]["source"] == "matched RGB track 1"
    # The other thermal animals keep the measurement as their source.
    assert _rows(pair, "t")[2]["evidence"]["source"] == life_stage.AREA_PIXEL


def test_the_carried_call_reaches_the_track_age_field(pair):
    import json as _json
    from bambi_wildlife_detection.core import label_store

    _run(pair, classification_models=MATCHED)
    juvenile = label_store.vocabulary(pair)["enum_ids"]["age"]["juvenile"]

    conn = store.open_store(
        store.stage_path(pair, store.TRACKS, "t"), store.TRACKS, "t")
    try:
        attributes = conn.execute(
            "SELECT attributes FROM tracks WHERE track_id = 1").fetchone()[0]
    finally:
        conn.close()
    assert _json.loads(attributes)["age"] == juvenile


def test_a_rerun_carries_the_call_again_rather_than_stacking_it(pair):
    _run(pair, classification_models=MATCHED)
    _run(pair, classification_models=MATCHED)

    rows = _rows(pair, "t")
    assert len(rows) == len(FLAT_HERD)
    assert rows[1]["label"] == life_stage.JUVENILE
    assert rows[1]["evidence"]["own_label"] == life_stage.ADULT


def test_a_classifier_call_beats_a_carried_size_call(pair):
    classification_store.record_track_predictions(
        pair, "t", classification_store.LIFE_STAGE, [
            {"track_id": 1, "label": "adult", "votes": 9, "n": 10,
             "fraction": 0.9, "model": "life_stage_matched.pt"}])
    _run(pair, classification_models=MATCHED)

    row = _rows(pair, "t")[1]
    assert row["label"] == life_stage.ADULT
    assert row["model"] == "life_stage_matched.pt"


def test_without_matches_nothing_is_carried(tmp_path):
    root = str(tmp_path)
    store.open_store(store.project_path(root), store.PROJECT).close()
    _populate(root, "t", FLAT_HERD)
    _populate(root, "w", HERD)
    _run(root, classification_models=MATCHED)

    assert _labels(root, "w")[1] == life_stage.JUVENILE
    assert _labels(root, "t")[1] == life_stage.ADULT


def test_an_unmeasured_camera_receives_nothing(pair):
    """The stage set to RGB only measures RGB; the thermal tracks were never
    assessed, so there is no size verdict to amend."""
    _run(pair, classification_models={"life_stage": {"modality": "rgb"}})

    assert _labels(pair, "w")[1] == life_stage.JUVENILE
    assert _labels(pair, "t") == {}
