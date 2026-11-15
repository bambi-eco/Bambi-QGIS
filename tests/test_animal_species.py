# -*- coding: utf-8 -*-
"""An animal is one thing in every survey tool.

The tools used to decide a tracked animal's species three ways: the census,
the track inventory and habitat use asked the track (``animal`` for every
track no classifier had named, whatever the detector called its boxes); the
density heatmap asked each box (an animal the detector called roe deer on
two frames and chamois on a third counted once for each, at two places);
distance sampling and the population estimate asked the last box. And a
track of nothing but boxes the detector's class mapping calls
``not-an-animal`` was counted by the census.

Now there is one answer (``track_store.animal_species``): the track's own
species where the animal was identified, else what most of its boxes were
called - and ``analytics_source.load_animals`` hands every tool the same
animals (or boxes) with it.
"""
import json
import os

import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor
from bambi_wildlife_detection.core import (
    analytics_source, detection_store, habitat, label_store, store,
    track_inventory, track_store)
from bambi_wildlife_detection.core.survey import Survey, SurveyFlight

ROE, RED, CHAMOIS = "roe deer", "red deer", "chamois"
ANIMAL, NOT_AN_ANIMAL = store.FALLBACK_SPECIES_ID, store.NOT_AN_ANIMAL_SPECIES_ID

#: (frame, detector class, track) - track ``None`` is a box of no track.
BOXES = [
    (0, "roe", 1), (1, "roe", 1), (2, "chamois", 1),    # flickering detector
    (0, "chamois", 2), (1, "chamois", 2),
    (0, "rock", 3), (1, "rock", 3),                     # a false positive
    (3, "roe", 4), (4, "roe", 4),                       # named red deer since
    (5, "something", 5),                                # an unmapped class
    (6, "roe", None),
]


@pytest.fixture
def flight(tmp_path):
    """Five tracks, of which the detector named three, a classifier one and
    nobody one - and one that is a rock."""
    root = str(tmp_path / "flight")
    os.makedirs(root)
    label_store.vocabulary(root, create=True)
    names = label_store.vocabulary(root)["species_by_name"]
    project = store.open_store(store.project_path(root), store.PROJECT)
    source_id = detection_store.ensure_source(project, detection_store.DETECTOR)
    project.executemany(
        "INSERT INTO class_mapping (source_id, source_class, species_id) "
        "VALUES (?, ?, ?)",
        [(source_id, "roe", names[ROE]), (source_id, "chamois", names[CHAMOIS]),
         (source_id, "rock", NOT_AN_ANIMAL)])
    project.commit()
    project.close()

    detection_store.record_detections(root, "t", [
        {"frame": frame, "x1": 10.0 * n, "y1": 1.0, "x2": 10.0 * n + 4,
         "y2": 5.0, "confidence": 0.9, "source_class": cls}
        for n, (frame, cls, _track) in enumerate(BOXES)])
    stored = sorted(track_store.load_detections(root, "t"),
                    key=lambda d: d["x1"])
    ids = [d["detection_id"] for d in stored]
    track_store.record_georeference(root, "t", [
        {"detection_id": i, "gx1": 500000.0 + 10 * n, "gy1": 5300000.0,
         "gz1": 0.0, "gx2": 500002.0 + 10 * n, "gy2": 5300002.0, "gz2": 0.0}
        for n, i in enumerate(ids)])
    track_store.record_tracks(
        root, "t",
        [{"track_id": track, "detection_id": i}
         for i, (_frame, _cls, track) in zip(ids, BOXES) if track is not None],
        species_of=lambda track: names[RED] if track == 4 else ANIMAL)
    track_store.set_track_attribute(root, "t", 1, "sex", "female")
    track_store.set_track_attribute(root, "t", 1, "approved", True)
    return root


def _names(root):
    return label_store.vocabulary(root)["species_by_name"]


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

class TestMajority:
    def test_most_boxes_decide(self):
        assert track_store.majority_species(
            [(1, 0.9), (1, 0.4), (5, 0.99)]) == 1

    def test_a_box_without_a_species_has_no_vote(self):
        assert track_store.majority_species(
            [(ANIMAL, 0.9), (ANIMAL, 0.9), (5, 0.3), (None, 0.9)]) == 5

    def test_no_named_box_leaves_the_animal_unnamed(self):
        assert track_store.majority_species([(ANIMAL, 0.9)]) == ANIMAL
        assert track_store.majority_species([]) == ANIMAL

    def test_equal_numbers_go_to_the_surer_boxes(self):
        assert track_store.majority_species([(1, 0.5), (5, 0.8)]) == 5
        assert track_store.majority_species([(1, 0.8), (5, 0.5)]) == 1

    def test_a_dead_heat_goes_to_the_animal(self):
        """A false positive needs a majority of its own."""
        assert track_store.majority_species(
            [(1, 0.5), (NOT_AN_ANIMAL, 0.5)]) == 1
        assert track_store.majority_species(
            [(store.FALLBACK_SPECIES_ID - 1, 0.5), (5, 0.5)]) == 5
        assert track_store.majority_species([(5, None), (1, None)]) == 1

    def test_a_false_positive_can_win(self):
        assert track_store.majority_species(
            [(1, 0.9), (NOT_AN_ANIMAL, 0.2), (NOT_AN_ANIMAL, 0.2)]
        ) == NOT_AN_ANIMAL


def test_each_animal_has_one_species(flight):
    names = _names(flight)
    assert track_store.animal_species(flight, "t") == {
        1: names[ROE],          # two of its three boxes
        2: names[CHAMOIS],
        3: NOT_AN_ANIMAL,
        4: names[RED],          # the track's own outranks its boxes
        5: ANIMAL,
    }


def test_no_tracks_no_animals(tmp_path):
    assert track_store.animal_species(str(tmp_path), "t") == {}


def test_a_replaced_track_is_no_animal(flight):
    names = _names(flight)
    label_store.save_tracks(flight, "t", [{
        "label_track_id": 1, "species_id": names[RED], "origin_track_id": 2,
        "keyframes": [{"frame": 0, "x1": 1.0, "y1": 1.0, "x2": 2.0,
                       "y2": 2.0}]}])
    label_store.materialise(flight, "t")

    species = track_store.animal_species(flight, "t")
    assert 2 not in species
    assert sorted(species.values()).count(names[RED]) == 2
    assert names[RED] not in track_store.animal_species(
        flight, "t", include_manual=False).values() or \
        list(track_store.animal_species(
            flight, "t", include_manual=False).values()).count(names[RED]) == 1


# ---------------------------------------------------------------------------
# The animals every tool gets
# ---------------------------------------------------------------------------

def test_animals_are_counted_once_and_for_one_species(flight):
    animals, facts = analytics_source.load_animals(flight, "t", "tracks")
    assert [(a["track_id"], a["species"], a["n_boxes"]) for a in animals] == [
        (1, ROE, 3), (2, CHAMOIS, 2), (4, RED, 2), (5, "animal", 1)]
    assert facts == {"excluded_not_an_animal": 1, "untracked_detections": 1,
                     "unplaced": 0}


def test_an_animal_is_where_all_its_boxes_are(flight):
    animals, _facts = analytics_source.load_animals(flight, "t", "tracks")
    first = animals[0]
    assert (first["x"], first["y"]) == pytest.approx((500011.0, 5300001.0))
    assert (first["first_frame"], first["last_frame"]) == (0, 2)


def test_sex_age_and_approval_are_the_tracks(flight):
    animals, _facts = analytics_source.load_animals(flight, "t", "tracks")
    assert [(a["sex"], a["approved"]) for a in animals] == [
        ("female", True), (None, False), (None, False), (None, False)]


def test_the_species_filter_selects_whole_animals(flight):
    names = _names(flight)
    for name, tracks in ((ROE, [1]), (CHAMOIS, [2]), (RED, [4])):
        animals, _facts = analytics_source.load_animals(
            flight, "t", "tracks", [names[name]])
        assert [a["track_id"] for a in animals] == tracks, name
    assert analytics_source.load_animals(flight, "t", "tracks", [])[0] == []


def _boxes(flight):
    """The counted boxes in the order of :data:`BOXES`."""
    boxes, facts = analytics_source.load_animals(flight, "t", "detections")
    return sorted(boxes, key=lambda box: box["detection_id"]), facts


def test_boxes_are_what_they_were_called(flight):
    boxes, facts = _boxes(flight)
    assert [b["species"] for b in boxes] == [
        ROE, ROE, CHAMOIS, CHAMOIS, CHAMOIS, ROE, ROE, "animal", ROE]
    assert facts["excluded_not_an_animal"] == 2
    assert facts["untracked_detections"] == 1
    assert all(b["detection_id"] is not None for b in boxes)


def test_a_box_of_a_known_female_is_a_box_of_a_female(flight):
    boxes, _facts = _boxes(flight)
    assert [b["sex"] for b in boxes[:4]] == ["female"] * 3 + [None]
    assert [b["approved"] for b in boxes[:4]] == [True] * 3 + [False]
    assert boxes[-1]["track_id"] is None


def test_what_cannot_be_placed_still_counts(flight):
    conn = store.open_store(
        store.stage_path(flight, store.GEOREFERENCED, "t"),
        store.GEOREFERENCED, "t")
    conn.execute("DELETE FROM detections_geo")
    conn.commit()
    conn.close()

    animals, facts = analytics_source.load_animals(flight, "t", "tracks")
    assert len(animals) == 4 and facts["unplaced"] == 4
    assert all(a["x"] is None for a in animals)


def test_an_unknown_source_is_refused(flight):
    with pytest.raises(analytics_source.AnalyticsError):
        analytics_source.load_animals(flight, "t", "frames")


# ---------------------------------------------------------------------------
# ... and the tools
# ---------------------------------------------------------------------------

def test_the_heat_map_places_an_animal_once(flight):
    """It was at two places, once for each species its boxes were called."""
    names = _names(flight)
    points, provenance = analytics_source.load_points(
        flight, "t", "tracks", [names[ROE]])
    assert points == [pytest.approx((500011.0, 5300001.0))]
    points, _p = analytics_source.load_points(
        flight, "t", "tracks", [names[CHAMOIS]])
    assert points == [pytest.approx((500036.0, 5300001.0))]       # not two
    everything, provenance = analytics_source.load_points(flight, "t", "tracks")
    assert len(everything) == 4
    assert provenance["excluded_not_an_animal_tracks"] == 1
    assert provenance["untracked_detections"] == 1


def test_the_strata_add_up_to_the_animals(flight):
    strata, _provenance = analytics_source.load_points_by_species(
        flight, "t", "tracks")
    assert {name: len(points) for name, points in strata.items()} == {
        "animal": 1, ROE: 1, RED: 1, CHAMOIS: 1}


def test_the_inventory_names_the_animal(flight):
    inventory = track_inventory.build_inventory(flight, "t")
    assert {row["track_id"]: row["species"] for row in inventory} == {
        1: ROE, 2: CHAMOIS, 3: "not-an-animal", 4: RED, 5: "animal"}
    roe = track_inventory.build_inventory(
        flight, "t", species_ids=[_names(flight)[ROE]])
    assert [row["track_id"] for row in roe] == [1]      # it listed none


def _survey(flight, source="tracks", species=None):
    survey = Survey(folder=os.path.join(os.path.dirname(flight), "survey"),
                    modality="t", source=source, species=species,
                    flights=[SurveyFlight("flight", flight, 32633)])
    return survey


def _census(flight, source="tracks"):
    survey = _survey(flight, source)
    BambiProcessor().run_census_summary({"survey": survey.to_dict()})
    with open(os.path.join(survey.analytics_folder,
                           f"census_summary_{source}.json")) as fh:
        return {row["species"]: row["counted"]
                for row in json.load(fh)["rows"]}


def test_the_census_counts_animals_by_their_species(flight):
    """Every track was "animal" - and the rock was one of them."""
    assert _census(flight) == {
        ROE: 1, CHAMOIS: 1, RED: 1, "animal": 1, "All species": 4}


def test_the_census_of_boxes_leaves_the_rock_out_too(flight):
    assert _census(flight, "detections") == {
        ROE: 5, CHAMOIS: 3, "animal": 1, "All species": 9}


def test_habitat_use_reads_the_same_animals(flight):
    names = _names(flight)
    tracks = habitat.flight_tracks(flight, "t")
    assert [(t["track_id"], t["species"], t["sex"]) for t in tracks] == [
        (1, ROE, "female"), (2, CHAMOIS, ""), (4, RED, ""), (5, "animal", "")]
    assert [t["track_id"] for t in habitat.flight_tracks(
        flight, "t", [names[ROE]])] == [1]
    boxes = habitat.flight_tracks(flight, "t", source="detections")
    assert len(boxes) == 9 and boxes[0]["detection_id"] is not None


def test_the_habitat_table_names_the_box_when_boxes_are_counted():
    rows = [{"track_id": 1, "detection_id": 7, "species": ROE}]
    columns = habitat.table_columns(rows)
    assert columns[columns.index("track_id") + 1] == "detection_id"
    assert habitat.table_columns([{"track_id": 1}]) == list(
        habitat.TRACK_COLUMNS)
    assert habitat.unit({"source": "detections"}, 3) == "3 box(es)"
    assert habitat.unit({}, 3) == "3 track(s)"


# ---------------------------------------------------------------------------
# Distances written earlier
# ---------------------------------------------------------------------------

def _entries(flight):
    """As "Calculate Track Perpendicular" wrote them: ``class_id`` is what
    the *last* box was called - chamois for the roe deer of track 1."""
    names = _names(flight)
    return [
        {"track_id": 1, "class_id": names[CHAMOIS], "distance_m": 3.0},
        {"track_id": 2, "class_id": names[CHAMOIS], "distance_m": 5.0},
        {"track_id": 4, "class_id": names[ROE], "distance_m": 7.0},
        {"track_id": 5, "class_id": ANIMAL, "distance_m": 9.0},
    ]


def test_a_distance_counts_for_the_animals_species(flight):
    names = _names(flight)
    now = analytics_source.current_species(flight, "t", "tracks")

    def distances(name):
        kept, _dropped = analytics_source.select_entries(
            _entries(flight), "track_id", now, [names[name]])
        return [entry["distance_m"] for entry in kept]

    assert distances(ROE) == [3.0]          # was 7.0 - the red deer's
    assert distances(RED) == [7.0]          # was nothing
    assert distances(CHAMOIS) == [5.0]      # was 3.0 and 5.0


def test_a_false_positive_marked_since_is_dropped(flight):
    entries = _entries(flight) + [
        {"track_id": 3, "class_id": ANIMAL, "distance_m": 1.0}]
    now = analytics_source.current_species(flight, "t", "tracks")
    kept, dropped = analytics_source.select_entries(entries, "track_id", now)
    assert dropped == 1
    assert [entry["track_id"] for entry in kept] == [1, 2, 4, 5]
    assert kept[0]["class_id"] == _names(flight)[ROE]
    assert entries[0]["class_id"] == _names(flight)[CHAMOIS]    # a copy


def test_an_entry_the_store_does_not_know_keeps_its_class(flight):
    """A file from before a re-run, or a flight without a store."""
    entries = [{"track_id": 99, "class_id": 5, "distance_m": 2.0},
               {"class_id": 5, "distance_m": 4.0},
               {"track_id": "x", "class_id": NOT_AN_ANIMAL, "distance_m": 6.0}]
    now = analytics_source.current_species(flight, "t", "tracks")
    kept, dropped = analytics_source.select_entries(
        entries, "track_id", now, [5])
    assert [entry["distance_m"] for entry in kept] == [2.0, 4.0]
    assert dropped == 1
    assert analytics_source.select_entries(entries, "track_id", {})[1] == 1


def test_boxes_are_looked_up_by_their_own_id(flight):
    now = analytics_source.current_species(flight, "t", "detections")
    assert len(now) == len(BOXES)
    rock = next(i for i, species in now.items() if species == NOT_AN_ANIMAL)
    kept, dropped = analytics_source.select_entries(
        [{"det_idx": rock, "class_id": ANIMAL, "distance_m": 1.0}],
        "det_idx", now)
    assert (kept, dropped) == ([], 1)
    assert analytics_source.current_species(
        os.path.dirname(flight), "t", "detections") == {}


def _distance_files(flight):
    """Poses and a route along the boxes, and the distances of the tracks."""
    with open(os.path.join(flight, "poses_t.json"), "w") as fh:
        json.dump({"images": [
            {"imagefile": f"{i:04d}.jpg",
             "location": [500000.0 + 20.0 * i, 5300000.0, 100.0]}
            for i in range(8)]}, fh)
    route = os.path.join(flight, "flight_route_t")
    os.makedirs(route, exist_ok=True)
    with open(os.path.join(route, "flight_route.geojson"), "w") as fh:
        json.dump({"type": "FeatureCollection", "features": [{
            "type": "Feature", "properties": {}, "geometry": {
                "type": "LineString",
                "coordinates": [[500000.0, 5300000.0],
                                [500140.0, 5300000.0]]}}]}, fh)
    with open(os.path.join(route, "perpendicular_tracks_t.json"), "w") as fh:
        json.dump({"tracks": [
            dict(entry, last_frame=2,
                 detection_center=[500010.0, 5300000.0 + entry["distance_m"],
                                   0.0])
            for entry in _entries(flight)]}, fh)


def test_distance_sampling_and_the_census_agree(flight):
    _distance_files(flight)
    survey = _survey(flight)
    processor = BambiProcessor()

    def distances(name):
        loaded = processor._load_distance_sampling_project(
            survey.flights[0], survey, "tracks", names=[name])
        return loaded["distances"].tolist()

    assert distances(ROE) == [3.0]
    assert distances(RED) == [7.0]
    assert distances(CHAMOIS) == [5.0]
    everything = processor._load_distance_sampling_project(
        survey.flights[0], survey, "tracks")
    assert sorted(everything["distances"].tolist()) == [3.0, 5.0, 7.0, 9.0]
