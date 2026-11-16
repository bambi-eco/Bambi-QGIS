# -*- coding: utf-8 -*-
"""What was published of a BAMBI flight, put into a flight of the plugin's
(core/bambi_dataset.py): its species into the flight's list, its labels
into the labelling store, its poses and its correction into the flight's
folder - and all of it in one go, for a flight that is fetched.

The parts are covered one by one in tests/test_bambi_dataset.py."""
import json
import math
import os

import pytest

from bambi_wildlife_detection.core import bambi_dataset as bd
from bambi_wildlife_detection.core import (
    checked_download, environment, label_store, remote_zip)
from bambi_wildlife_detection.core import flights as flights_core
from bambi_wildlife_detection.core.config_schema import (
    CONFIG_KEYS, save_config_entries)
from bambi_wildlife_detection.core.labelling import LabelStore, LabelTrack
from bambi_wildlife_detection.core.schema_editor import SchemaEditor

from . import dataset_samples as samples
from .test_flight_archive import MEMBERS

pytest.importorskip("pyproj")

#: The roe deer on frames 0-10, the boar on frames 2-8.
BOXES = 11 + 7
#: Between the map's north and true north at the made-up flight, degrees.
CONVERGENCE = 0.7146


def labels_of(target):
    store = LabelStore(target, "t")
    store.load()
    return store


def by_dataset_track(target):
    return {track.attributes.get(bd.TRACK_FIELD): track
            for track in labels_of(target).tracks.values()}


@pytest.fixture
def target(tmp_path):
    folder = tmp_path / "flight_7" / "qgis"
    folder.mkdir(parents=True)
    return str(folder)


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------

class TestSpecies:
    def test_species_the_plugin_knows_are_found_by_their_scientific_name(
            self, target):
        names = bd.ensure_species(target, [
            "Capreolus capreolus (Roe deer)", "Dama dama (Fallow Deer)",
            "Aves (Bird)"])

        assert names == {"Capreolus capreolus (Roe deer)": "roe deer",
                         "Dama dama (Fallow Deer)": "fallow deer",
                         "Aves (Bird)": "bird"}

    def test_what_is_no_species_is_the_plugins_base_class(self, target):
        log = []

        names = bd.ensure_species(target, ["Unknown", "No-animal", ""],
                                  log.append)

        assert names == {"Unknown": "unknown", "No-animal": "not-an-animal",
                         "": "unknown"}
        assert log == []

    def test_a_species_the_flight_does_not_have_is_added(self, target):
        log = []

        names = bd.ensure_species(target, [
            "Capra ibex (Alpine ibex)", "Canis lupus familiaris (Dog)",
            "Sus scrofa x Sus domesticus (Hybrid Pig)"], log.append)

        assert names["Capra ibex (Alpine ibex)"] == "alpine ibex"
        assert log == ["Added to the flight's species: alpine ibex, dog, "
                       "hybrid pig"]
        with SchemaEditor(target) as editor:
            ibex = editor.species_by_name("alpine ibex")
            dog = editor.species_by_name("dog")
            pig = editor.species_by_name("hybrid pig")
        assert (ibex["scientific_name"], ibex["taxon_rank"]) == (
            "Capra ibex", "species")
        assert (dog["scientific_name"], dog["taxon_rank"]) == (
            "Canis lupus familiaris", "subspecies")
        # a cross of two is no rank this plugin would name
        assert (pig["scientific_name"], pig["taxon_rank"]) == (
            "Sus scrofa x Sus domesticus", None)
        assert not ibex["protected"]

    def test_it_is_added_once(self, target):
        bd.ensure_species(target, ["Capra ibex (Alpine ibex)"])
        log = []

        names = bd.ensure_species(target, ["Capra ibex (Alpine ibex)"],
                                  log.append)

        assert names == {"Capra ibex (Alpine ibex)": "alpine ibex"} and not log
        known = label_store.vocabulary(target)["species_by_name"]
        assert sum(1 for name in known if name == "alpine ibex") == 1

    def test_a_flight_that_calls_a_species_its_own_way(self, target):
        with SchemaEditor(target) as editor:
            editor.rename_species(
                editor.species_by_name("red deer")["species_id"], "Rothirsch")
            editor.commit()

        names = bd.ensure_species(target, ["Cervus elaphus (Red deer)"])

        assert names == {"Cervus elaphus (Red deer)": "Rothirsch"}
        assert "red deer" not in label_store.vocabulary(target)[
            "species_by_name"]

    def test_a_name_without_a_scientific_one(self, target):
        names = bd.ensure_species(target, ["Fox", "Marten"])

        assert names == {"Fox": "fox", "Marten": "marten"}
        with SchemaEditor(target) as editor:
            assert editor.species_by_name("marten")["scientific_name"] is None


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def tracks_of(mapping=None, size=(1024, 1024)):
    if mapping is None:
        mapping = {frame: frame for frame in range(12)}
    return bd.label_tracks(bd.parse_labels(samples.GT), mapping,
                           lambda x, y: (x, y), size)


class TestImportLabels:
    def test_the_animals_are_label_tracks_of_the_flight(self, target):
        log = []

        result = bd.import_labels(target, tracks_of(), log=log.append)

        assert result == {"tracks": 2, "key_frames": 5, "boxes": BOXES,
                          "species": ["roe deer", "wild boar"]}
        deer, boar = (by_dataset_track(target)[number]
                      for number in (4668, 4669))
        assert (deer.species, deer.sex, deer.age) == (
            "roe deer", "female", "adult")
        assert (boar.species, boar.sex, boar.age) == (
            "wild boar", "male", "juvenile")
        assert sorted(deer.keyframes) == [0, 4, 10]
        assert deer.keyframes[0] == {"x1": 570.0, "y1": 223.0, "x2": 619.0,
                                     "y2": 299.0, "occlusion": "occluded"}
        assert deer.keyframes[4]["occlusion"] == "clear"
        assert log[-1] == (
            f"Labels: 2 animal(s) on 5 key frame(s), {BOXES} boxes with the "
            "frames between them (roe deer, wild boar)")

    def test_the_boxes_between_the_key_frames_are_made(self, target):
        bd.import_labels(target, tracks_of())

        with open(os.path.join(target, "detections_t",
                               "detections.txt")) as handle:
            rows = [line.split() for line in handle
                    if line.strip() and not line.startswith("#")]
        assert len(rows) == BOXES
        # half-way between the key frames 0 and 4 of the roe deer
        (between,) = [row for row in rows
                      if row[0] == "2" and float(row[1]) > 500]
        assert [float(value) for value in between[1:5]] == [
            568.0, 227.0, 617.0, 303.0]
        assert label_store.has_labels_store(target, "t")

    def test_the_tool_reads_them_as_its_own(self, target):
        bd.import_labels(target, tracks_of())

        store = labels_of(target)

        assert store.load_notes == []
        assert [(field.name, field.type, field.scope)
                for field in store.custom_fields] == [
            (bd.TRACK_FIELD, "int", "track")]
        assert sorted(os.listdir(os.path.join(target, "labels_t"))) == [
            "labels.csv", "labels.json"]

    def test_importing_again_replaces_what_was_imported(self, target):
        bd.import_labels(target, tracks_of())
        before = {number: track.track_id
                  for number, track in by_dataset_track(target).items()}

        # on other frames now: the flight was extracted in between
        shifted = {frame: frame + 3 for frame in range(12)}
        result = bd.import_labels(target, tracks_of(shifted))

        after = by_dataset_track(target)
        assert result["boxes"] == BOXES and len(after) == 2
        assert sorted(after[4668].keyframes) == [3, 7, 13]
        # each animal under the number it had
        assert {number: track.track_id
                for number, track in after.items()} == before

    def test_labels_drawn_by_hand_stay(self, target):
        own = LabelStore(target, "t")
        own.load()
        mine = LabelTrack(own.next_track_id(), species="fox")
        mine.set_keyframe(3, (1.0, 2.0, 30.0, 40.0))
        own.tracks[mine.track_id] = mine
        own.save()

        bd.import_labels(target, tracks_of())
        bd.import_labels(target, tracks_of())

        tracks = labels_of(target).tracks
        assert len(tracks) == 3
        assert tracks[mine.track_id].species == "fox"
        assert bd.TRACK_FIELD not in tracks[mine.track_id].attributes
        assert sorted(tracks[mine.track_id].keyframes) == [3]

    def test_an_animal_that_is_gone_from_the_labels_goes(self, target):
        bd.import_labels(target, tracks_of())

        result = bd.import_labels(target, tracks_of()[:1])

        assert result["tracks"] == 1 and result["boxes"] == 11
        assert list(by_dataset_track(target)) == [4668]

    def test_no_labels_is_no_error(self, target):
        assert bd.import_labels(target, []) == {
            "tracks": 0, "key_frames": 0, "boxes": 0, "species": []}


class TestPlaceLabels:
    def test_without_poses_there_is_nothing_to_put_them_on(self, target):
        samples.put_published(target)

        assert bd.place_labels(target, samples.FLIGHT) is None
        assert not os.path.exists(os.path.join(target, "labels_t"))

    def test_on_the_published_frames(self, target, tmp_path):
        samples.put_published(target)
        bd.import_route(target, samples.FLIGHT,
                        samples.dem_metadata(str(tmp_path / "dem")),
                        samples.EPSG)

        result = bd.place_labels(target, samples.FLIGHT)

        assert result == {"tracks": 2, "key_frames": 5, "boxes": BOXES,
                          "species": ["roe deer", "wild boar"],
                          "on": "published", "left_out": 0}
        deer = by_dataset_track(target)[4668]
        assert sorted(deer.keyframes) == [0, 4, 10]
        assert deer.keyframes[0]["x1"] == 570.0

    def test_on_the_frames_of_an_extraction(self, target):
        published = samples.put_published(target)
        samples.put_extraction(target, samples.own_poses(published, before=1))
        log = []

        result = bd.place_labels(target, samples.FLIGHT, log=log.append)

        assert (result["on"], result["tracks"], result["left_out"]) == (
            "extracted", 2, 0)
        deer = by_dataset_track(target)[4668]
        # a frame ahead of the published ones: every number is one more
        assert sorted(deer.keyframes) == [1, 5, 11]
        # ... and the picture is another: 640 x 512, cut closer
        scale = (256.0 / math.tan(math.radians(samples.OWN_FOVY) / 2)) / (
            512.0 / math.tan(math.radians(samples.PUBLISHED_FOVY) / 2))
        assert deer.keyframes[1]["x1"] == pytest.approx(
            320.0 + (570.0 - 512.0) * scale)
        assert deer.keyframes[1]["y1"] == pytest.approx(
            256.0 + (223.0 - 512.0) * scale)

    def test_key_frames_the_flight_does_not_have_are_said(self, target):
        published = samples.put_published(target)
        samples.put_extraction(target, samples.own_poses(
            published, before=0, dropped=(4, 8)))
        log = []

        result = bd.place_labels(target, samples.FLIGHT, log=log.append)

        assert result["key_frames"] == 3 and result["left_out"] == 2
        assert log[-1] == ("2 labelled box(es) are on frames the flight "
                           "does not have, or outside their picture: left "
                           "out")

    def test_what_they_were_put_on_is_noted(self, target):
        published = samples.put_published(target)
        own = samples.own_poses(published)
        samples.put_extraction(target, own)

        bd.place_labels(target, samples.FLIGHT)

        state = bd.read_state(target)
        assert state["flight"] == samples.FLIGHT
        assert state["labels"] == {
            "wanted": True, "on": "extracted", "tracks": 2, "boxes": BOXES,
            "left_out": 0, "frames": bd.frames_fingerprint(own)}

    def test_the_published_files_have_to_be_there(self, target):
        samples.put_published(target, kinds=(bd.POSES,))

        with pytest.raises(bd.DatasetError, match="published labels and "
                                                  "poses of flight 7"):
            bd.place_labels(target, samples.FLIGHT)

    def test_frames_that_cannot_be_measured(self, target):
        published = samples.put_published(target)
        own = samples.own_poses(published)
        with open(os.path.join(target, "poses_t.json"), "w") as handle:
            json.dump(own, handle)             # poses only: no images

        with pytest.raises(bd.DatasetError, match="extract the frames "
                                                  "first"):
            bd.place_labels(target, samples.FLIGHT)


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------

class TestImportRoute:
    def test_the_published_poses_become_the_flights_own(self, target,
                                                        tmp_path):
        published = samples.put_published(target)
        log = []

        result = bd.import_route(
            target, samples.FLIGHT,
            samples.dem_metadata(str(tmp_path / "dem")), samples.EPSG,
            log=log.append)

        path = os.path.join(target, "poses_t.json")
        assert result == {"poses": path, "frames": 12, "replaced": False}
        with open(path) as handle:
            poses = json.load(handle)
        assert [image["imagefile"] for image in poses["images"]] == [
            image["imagefile"] for image in published["images"]]
        assert bd.are_published(poses)
        assert poses["source"]["flight"] == samples.FLIGHT
        assert poses["source"]["dataset"] == "BAMBI"
        assert log == ["Flight route: 12 poses as published, placed on the "
                       "flight's elevation model"]

    def test_the_frames_are_where_the_published_ones_are(self, target,
                                                         tmp_path):
        from pyproj import Transformer

        published = samples.put_published(target)
        metadata = samples.dem_metadata(str(tmp_path / "dem"))

        bd.import_route(target, samples.FLIGHT, metadata, samples.EPSG)

        with open(os.path.join(target, "poses_t.json")) as handle:
            poses = json.load(handle)
        with open(metadata) as handle:
            origin = json.load(handle)["origin"]
        to_map = Transformer.from_crs(4326, samples.EPSG, always_xy=True)
        for image, theirs in zip(poses["images"], published["images"]):
            east, north = to_map.transform(theirs["lng"], theirs["lat"])
            # the model's origin and a frame's place add up to the map's
            assert origin[0] + image["location"][0] == pytest.approx(east)
            assert origin[1] + image["location"][1] == pytest.approx(north)
            assert origin[2] + image["location"][2] == pytest.approx(373.8)

    def test_headings_are_turned_to_the_maps_north_as_an_extraction_does(
            self, target, tmp_path):
        samples.put_published(target)

        bd.import_route(target, samples.FLIGHT,
                        samples.dem_metadata(str(tmp_path / "dem")),
                        samples.EPSG)

        with open(os.path.join(target, "poses_t.json")) as handle:
            poses = json.load(handle)
        assert poses["headings"]["north"] == "map"
        assert poses["headings"]["epsg"] == samples.EPSG
        assert poses["images"][0]["grid_convergence"] == pytest.approx(
            CONVERGENCE, abs=2e-3)

    def test_poses_of_an_extraction_are_not_replaced(self, target, tmp_path):
        published = samples.put_published(target)
        own = samples.own_poses(published)
        samples.put_extraction(target, own)
        log = []

        result = bd.import_route(
            target, samples.FLIGHT,
            samples.dem_metadata(str(tmp_path / "dem")), samples.EPSG,
            log=log.append)

        assert result == {"poses": None, "frames": 12, "replaced": False}
        with open(os.path.join(target, "poses_t.json")) as handle:
            assert json.load(handle) == own
        assert log == ["The flight has poses of its own extraction: the "
                       "published ones are not put in their place"]

    def test_unless_that_is_what_is_asked_for(self, target, tmp_path):
        published = samples.put_published(target)
        samples.put_extraction(target, samples.own_poses(published))

        result = bd.import_route(
            target, samples.FLIGHT,
            samples.dem_metadata(str(tmp_path / "dem")), samples.EPSG,
            force=True)

        assert result["replaced"] is True
        with open(result["poses"]) as handle:
            assert bd.are_published(json.load(handle))

    def test_published_poses_from_before_are_written_anew(self, target,
                                                          tmp_path):
        samples.put_published(target)
        metadata = samples.dem_metadata(str(tmp_path / "dem"))
        bd.import_route(target, samples.FLIGHT, metadata, samples.EPSG)

        result = bd.import_route(target, samples.FLIGHT, metadata,
                                 samples.EPSG)

        assert result["poses"] and result["replaced"] is True

    def test_it_is_noted_in_the_flights_folder(self, target, tmp_path):
        samples.put_published(target)

        bd.import_route(target, samples.FLIGHT,
                        samples.dem_metadata(str(tmp_path / "dem")),
                        samples.EPSG)

        assert bd.read_state(target) == {"flight": samples.FLIGHT,
                                         "route": {"frames": 12}}

    def test_the_poses_have_to_be_there(self, target, tmp_path):
        with pytest.raises(bd.DatasetError, match="published poses of "
                                                  "flight 7 are not"):
            bd.import_route(target, samples.FLIGHT,
                            samples.dem_metadata(str(tmp_path / "dem")),
                            samples.EPSG)

    @pytest.mark.parametrize("document", [
        None, "not json", "{}", '{"origin_wgs84": {"latitude": "north"}}'])
    def test_an_elevation_model_that_does_not_say_where_it_is(
            self, target, tmp_path, document):
        samples.put_published(target)
        metadata = tmp_path / "dem.json"
        if document is not None:
            metadata.write_text(document)

        with pytest.raises(bd.DatasetError, match="origin of the flight's "
                                                  "elevation model"):
            bd.import_route(target, samples.FLIGHT, str(metadata),
                            samples.EPSG)

        assert not os.path.exists(os.path.join(target, "poses_t.json"))


# ---------------------------------------------------------------------------
# The correction
# ---------------------------------------------------------------------------

RANGES = dict(samples.CORRECTION, fine_corrections=[
    {"start_frame": 2, "end_frame": 5, "translation": {"z": 2.0}}])


class TestImportCorrection:
    def test_it_is_fitted_to_the_map_where_the_flight_took_place(self,
                                                                 target):
        samples.put_published(target)
        log = []

        imported = bd.import_correction(target, samples.FLIGHT, samples.EPSG,
                                        log=log.append)

        assert imported["convergence"] == pytest.approx(CONVERGENCE, abs=2e-3)
        assert imported["correction"]["rotation"]["z"] == pytest.approx(
            0.033 - math.radians(imported["convergence"]))
        assert imported["correction"]["translation"]["z"] == 46.65
        assert imported["path"] == os.path.join(
            target, "bambi_dataset", "7_correction_qgis.json")
        with open(imported["path"]) as handle:
            assert json.load(handle) == imported["correction"]
        assert log == [
            "Correction as published, the 0.71 degrees between the map's "
            "north and true north taken out of its rotation: z = 1.18 "
            "degrees, +46.65 m in height"]

    def test_without_the_poses_the_flight_log_says_where(self, target):
        samples.put_published(target, kinds=(bd.CORRECTION,))

        imported = bd.import_correction(target, samples.FLIGHT, samples.EPSG,
                                        position=(47.005, 15.977))

        assert imported["convergence"] == pytest.approx(CONVERGENCE, abs=2e-3)

    def test_a_flight_that_is_nowhere(self, target):
        samples.put_published(target, kinds=(bd.CORRECTION,))

        with pytest.raises(bd.DatasetError, match="Where flight 7 took "
                                                  "place is not known"):
            bd.import_correction(target, samples.FLIGHT, samples.EPSG)

    def test_a_flight_without_a_published_correction(self, target):
        samples.put_published(target, kinds=(bd.POSES,))

        assert bd.import_correction(target, samples.FLIGHT,
                                    samples.EPSG) is None

    def test_the_published_file_is_left_as_it_is(self, target):
        samples.put_published(target)

        bd.import_correction(target, samples.FLIGHT, samples.EPSG)

        with open(bd.published_path(target, 7, bd.CORRECTION)) as handle:
            assert json.load(handle) == samples.CORRECTION

    def test_ranges_name_the_published_frames_while_those_are_the_flights(
            self, target, tmp_path):
        samples.put_published(target, correction=RANGES)
        bd.import_route(target, samples.FLIGHT,
                        samples.dem_metadata(str(tmp_path / "dem")),
                        samples.EPSG)

        imported = bd.import_correction(target, samples.FLIGHT, samples.EPSG)

        (part,) = imported["correction"]["additional"]
        assert (part["start"], part["end"]) == (2, 5)
        assert bd.read_state(target)["correction"] == {
            "epsg": samples.EPSG, "ranges": 1, "frames": ""}

    def test_ranges_follow_the_frames_of_an_extraction(self, target):
        published = samples.put_published(target, correction=RANGES)
        own = samples.own_poses(published, before=3)
        samples.put_extraction(target, own)

        imported = bd.import_correction(target, samples.FLIGHT, samples.EPSG)

        (part,) = imported["correction"]["additional"]
        assert (part["start"], part["end"]) == (5, 8)
        assert bd.read_state(target)["correction"]["frames"] == \
            bd.frames_fingerprint(own)


# ---------------------------------------------------------------------------
# After an extraction
# ---------------------------------------------------------------------------

class TestAfterExtraction:
    @pytest.fixture
    def imported(self, target, tmp_path):
        """A flight whose route and labels were imported; returns the
        published poses."""
        published = samples.put_published(target)
        bd.import_route(target, samples.FLIGHT,
                        samples.dem_metadata(str(tmp_path / "dem")),
                        samples.EPSG)
        bd.place_labels(target, samples.FLIGHT)
        return published

    def test_the_labels_follow_the_frames(self, target, imported):
        samples.put_extraction(target, samples.own_poses(imported, before=2))
        log = []

        result = bd.after_extraction(target, log=log.append)

        assert result["on"] == "extracted" and result["boxes"] == BOXES
        assert sorted(by_dataset_track(target)[4668].keyframes) == [2, 6, 12]
        assert log[-1] == (
            "The labels of the BAMBI dataset are on the extracted frames "
            "now (found by the moment a frame was taken, scaled to its "
            "picture)")
        assert bd.read_state(target)["labels"]["on"] == "extracted"

    def test_the_same_frames_again_change_nothing(self, target, imported):
        own = samples.own_poses(imported, before=2)
        samples.put_extraction(target, own)
        bd.after_extraction(target)
        # the user moves a box in the labelling tool
        store = labels_of(target)
        deer = [track for track in store.tracks.values()
                if track.attributes[bd.TRACK_FIELD] == 4668][0]
        deer.set_keyframe(2, (1.0, 2.0, 30.0, 40.0))
        store.save()

        assert bd.after_extraction(target) is None

        assert by_dataset_track(target)[4668].keyframes[2]["x1"] == 1.0

    def test_frames_of_another_extraction_get_them_anew(self, target,
                                                        imported):
        samples.put_extraction(target, samples.own_poses(imported, before=2))
        bd.after_extraction(target)

        samples.put_extraction(target, samples.own_poses(imported, before=0))
        result = bd.after_extraction(target)

        assert result is not None
        assert sorted(by_dataset_track(target)[4668].keyframes) == [0, 4, 10]

    def test_a_flight_that_is_not_of_the_dataset(self, target):
        published = samples.published_poses()
        samples.put_extraction(target, samples.own_poses(published))

        assert bd.after_extraction(target) is None
        assert sorted(os.listdir(target)) == ["frames_t", "poses_t.json"]

    def test_labels_nobody_asked_for_stay_where_they_are(self, target,
                                                         tmp_path):
        published = samples.put_published(target)
        bd.import_route(target, samples.FLIGHT,
                        samples.dem_metadata(str(tmp_path / "dem")),
                        samples.EPSG)
        samples.put_extraction(target, samples.own_poses(published))

        assert bd.after_extraction(target) is None
        assert not os.path.exists(os.path.join(target, "labels_t"))

    def test_labels_that_waited_for_the_frames(self, target):
        published = samples.put_published(target)
        bd.write_state(target, flight=samples.FLIGHT,
                       labels={"wanted": True})
        samples.put_extraction(target, samples.own_poses(published))

        result = bd.after_extraction(target)

        assert (result["on"], result["tracks"]) == ("extracted", 2)

    def test_only_the_thermal_frames_carry_them(self, target, imported):
        samples.put_extraction(target, samples.own_poses(imported, before=2))

        assert bd.after_extraction(target, "w") is None

    def test_what_goes_wrong_is_said_and_costs_nothing(self, target,
                                                       imported):
        samples.put_extraction(target, samples.own_poses(imported, before=2))
        os.remove(bd.published_path(target, samples.FLIGHT, bd.LABELS))
        log = []

        assert bd.after_extraction(target, log=log.append) is None

        assert log[-1].startswith(
            "Warning: the labels of the BAMBI dataset could not be put on "
            "the extracted frames (The published labels and poses of "
            "flight 7")

    def test_a_correction_with_ranges_is_numbered_anew(self, target,
                                                       tmp_path):
        published = samples.put_published(target, correction=RANGES)
        bd.import_route(target, samples.FLIGHT,
                        samples.dem_metadata(str(tmp_path / "dem")),
                        samples.EPSG)
        first = bd.import_correction(target, samples.FLIGHT, samples.EPSG)
        samples.put_extraction(target, samples.own_poses(published, before=3))
        log = []

        bd.after_extraction(target, log=log.append)

        with open(first["path"]) as handle:
            (part,) = json.load(handle)["additional"]
        assert (part["start"], part["end"]) == (5, 8)
        assert log == [
            "The published correction changes over 1 range(s) of frames, "
            "which named frames of the published video. They are numbered "
            f"as the extracted frames in {first['path']}: load that file "
            "under Configuration > Correction to use them."]
        # ... once
        log.clear()
        bd.after_extraction(target, log=log.append)
        assert log == []

    def test_a_correction_for_the_whole_flight_needs_no_word(self, target,
                                                             tmp_path):
        published = samples.put_published(target)
        bd.import_route(target, samples.FLIGHT,
                        samples.dem_metadata(str(tmp_path / "dem")),
                        samples.EPSG)
        bd.import_correction(target, samples.FLIGHT, samples.EPSG)
        samples.put_extraction(target, samples.own_poses(published))
        log = []

        bd.after_extraction(target, log=log.append)

        assert log == []


# ---------------------------------------------------------------------------
# A flight, ready
# ---------------------------------------------------------------------------

class Made:
    """What a prepared flight was made with: the server, the elevation
    model's maker, the route's writer."""

    def __init__(self, tmp_path, labelled=True, **archive):
        data = samples.labelled_archive(**archive) if labelled else None
        self.flight = samples.entry(labelled=data)
        self.session = samples.Zenodo(self.flight, labelled=data or b"")
        self.folder = str(tmp_path / "flights" / self.flight.folder)
        self.target = os.path.join(self.folder, "qgis")
        self.dems, self.routes, self.log, self.seen = [], [], [], []
        self.dem_fails = None

    def elevation(self, bounds, folder, crs, log=None, progress=None,
                  cancelled=None):
        self.dems.append((bounds, folder, crs))
        if self.dem_fails is not None:
            raise self.dem_fails
        if progress:
            progress(100)
        return {"metadata": samples.dem_metadata(folder)}

    def prepare(self, **more):
        more.setdefault("elevation", self.elevation)
        return bd.prepare_flight(
            self.flight, self.folder, self.session,
            make_route=self.routes.append, progress=self.seen.append,
            log=self.log.append, sleep=lambda seconds: None, **more)


@pytest.fixture
def made(tmp_path):
    return Made(tmp_path)


class TestPrepareFlight:
    def test_everything_that_was_published_of_it(self, made):
        prepared = made.prepare()

        assert prepared["flight"] == 7 and prepared["name"] == "BAMBI flight 7"
        assert prepared["folder"] == made.folder
        assert prepared["target_folder"] == made.target
        assert prepared["notes"] == []
        # the recordings
        assert set(MEMBERS) <= set(os.listdir(made.folder))
        assert prepared["files"]["airdata"].endswith("air_data.csv")
        # the elevation model, for the ground under the recording
        assert made.dems == [((15.9695, 47.0021, 15.9785, 47.0084),
                              made.folder, "EPSG:32633")]
        assert prepared["epsg"] == 32633
        assert prepared["dem"] == os.path.join(made.folder, "gem15_dem.glb")
        assert prepared["dem_metadata"] == os.path.join(made.folder,
                                                        "gem15_dem.json")
        # the route, the correction, the labels
        assert prepared["route"] == {
            "poses": os.path.join(made.target, "poses_t.json"),
            "frames": 12, "replaced": False}
        assert prepared["correction"]["correction"]["translation"]["z"] == \
            46.65
        assert prepared["labels"]["boxes"] == BOXES
        assert prepared["labels"]["on"] == "published"
        assert prepared["labels_wanted"] is True
        assert made.routes == [prepared]
        assert made.seen == sorted(made.seen) and made.seen[-1] == 100

    def test_what_was_taken_is_noted_in_the_flight(self, made):
        made.prepare()

        state = bd.read_state(made.target)
        assert state["flight"] == 7
        assert state["doi"] == "10.5281/zenodo.222"
        assert state["route"] == {"frames": 12}
        assert state["labels"]["wanted"] is True
        assert state["correction"]["ranges"] == 0

    def test_only_the_recordings(self, made):
        prepared = made.prepare(route=False, labels=False, correction=False)

        assert (prepared["route"], prepared["labels"],
                prepared["correction"]) == (None, None, None)
        assert prepared["notes"] == [] and made.routes == []
        assert made.session.ranges.asked == []
        assert not os.path.exists(os.path.join(made.target, "poses_t.json"))
        assert bd.read_state(made.target) == {"flight": 7,
                                              "elevation": "recording"}
        assert prepared["dem"]                  # the model all the same
        assert prepared["videos"] is True

    def test_only_the_correction(self, made):
        prepared = made.prepare(route=False, labels=False)

        assert prepared["correction"]["convergence"] == pytest.approx(
            CONVERGENCE, abs=5e-3)
        assert (prepared["route"], prepared["labels"]) == (None, None)
        # the annotations are fetched whole, as the dataset's script does;
        # only the correction is put into the flight
        assert sorted(os.listdir(os.path.join(
            made.target, "bambi_dataset"))) == [
            "7_correction.json", "7_correction_qgis.json", "7_gt.txt",
            "7_matched_poses.json", "7_metadata.json",
            "7_track_mapping.json", "dataset.json"]
        assert not os.path.exists(os.path.join(made.target, "poses_t.json"))
        assert not os.path.exists(os.path.join(made.target, "labels_t"))

    def test_labels_without_the_route_wait_for_the_frames(self, made):
        prepared = made.prepare(route=False)

        assert prepared["labels"] is None and prepared["labels_wanted"]
        assert prepared["route"] is None and made.routes == []
        assert bd.read_state(made.target)["labels"] == {"wanted": True}
        assert made.log[-1] == ("The labels are put on the frames once they "
                                "are extracted ('Extract Frames')")
        # ... which the extraction then does
        published = samples.published_poses()
        samples.put_extraction(made.target, samples.own_poses(published))
        assert bd.after_extraction(made.target)["tracks"] == 2

    def test_a_flight_that_was_not_labelled(self, tmp_path):
        made = Made(tmp_path, labelled=False)

        prepared = made.prepare()

        assert prepared["notes"] == [
            "The flight is not in the labelled dataset: neither a route "
            "nor labels nor a correction were published for it."]
        assert made.log[-1] == "Flight 7: " + prepared["notes"][0]
        assert prepared["files"]["thermal_videos"] and prepared["dem"]

    def test_a_model_that_cannot_be_fetched_does_not_cost_the_flight(
            self, made):
        made.dem_fails = environment.EnvironmentFailure("no connection")

        prepared = made.prepare()

        assert prepared["dem"] == "" and prepared["route"] is None
        assert prepared["notes"][0] == (
            "The elevation model could not be fetched (no connection). "
            "Fetch it with 'Download DEM (worldwide)' on the Input tab.")
        assert prepared["notes"][1].startswith(
            "Without an elevation model the published route cannot be "
            "placed.")
        # what needs no model is there: the correction, and the labels
        # wait for the frames
        assert prepared["correction"] is not None
        assert prepared["labels"] is None and prepared["labels_wanted"]
        assert prepared["epsg"] == 32633

    def test_no_model_is_made_where_none_is_asked_for(self, made):
        prepared = made.prepare(elevation=None, route=False, labels=False)

        assert made.dems == [] and prepared["dem"] == ""
        assert prepared["notes"] == []

    def test_a_model_that_is_there_is_used(self, made):
        metadata = samples.dem_metadata(made.folder, "bev_dem")

        prepared = made.prepare()

        assert made.dems == []
        assert prepared["dem_metadata"] == metadata
        assert prepared["route"]["poses"]

    def test_the_labelled_flight_out_of_reach(self, made):
        made.session.ranges.files.clear()

        prepared = made.prepare()

        assert prepared["notes"] == [
            "The labelled flight 7 is not where it was published "
            f"({made.flight.labelled.url})."]
        assert prepared["files"]["thermal_videos"]
        assert bd.read_state(made.target) == {"flight": 7,
                                              "elevation": "recording"}

    def test_a_labelled_flight_without_labels(self, tmp_path):
        made = Made(tmp_path, leave_out=("_gt.txt",))

        prepared = made.prepare()

        assert prepared["notes"] == [
            "No labels were published for the flight."]
        assert prepared["route"]["poses"] and prepared["labels"] is None

    def test_a_route_whose_files_cannot_be_written(self, made):
        def broken(prepared):
            raise RuntimeError("no flight log")

        prepared = bd.prepare_flight(
            made.flight, made.folder, made.session,
            elevation=made.elevation, make_route=broken,
            sleep=lambda seconds: None)

        assert prepared["notes"] == [
            "The route's files could not be written (no flight log): run "
            "'Generate Flight Route'."]
        assert prepared["route"]["poses"]

    def test_recordings_that_cannot_be_fetched_are_no_flight(self, made):
        made.session.raw_fails = ConnectionError("reset")

        with pytest.raises(bd.DatasetError, match="could not be downloaded"):
            made.prepare(attempts=2)

        assert made.dems == [] and made.session.ranges.asked == []

    def test_stopped_while_the_recordings_arrive(self, made):
        with pytest.raises(checked_download.Cancelled):
            made.prepare(cancelled=lambda: True)

    def test_stopped_while_the_model_is_made(self, made):
        made.dem_fails = environment.Cancelled()

        with pytest.raises(environment.Cancelled):
            made.prepare()

    def test_stopped_while_the_published_files_arrive(self, made):
        calls = []

        def cancelled():                # once the recordings are there
            calls.append(1)
            return os.path.isfile(os.path.join(made.folder, "air_data.csv"))

        with pytest.raises(remote_zip.Cancelled):
            made.prepare(cancelled=cancelled)

        assert made.dems == []

    def test_asked_for_again_nothing_is_fetched_twice(self, made):
        made.prepare()
        streamed, asked = (len(made.session.streamed),
                           len(made.session.ranges.asked))

        prepared = made.prepare()

        assert len(made.session.streamed) == streamed
        assert len(made.session.ranges.asked) == asked
        assert len(made.dems) == 1
        assert prepared["labels"]["boxes"] == BOXES and prepared["notes"] == []


class TestWithoutTheVideos:
    """A fast overview: the poses and labels of a flight, and nothing of
    its recordings."""

    def test_only_the_annotations_are_fetched(self, made):
        prepared = made.prepare(videos=False)

        assert made.session.streamed == []
        assert sorted(os.listdir(made.folder)) == [
            "gem15_dem.glb", "gem15_dem.json", "qgis"]
        assert prepared["videos"] is False
        assert prepared["files"]["thermal_videos"] == []
        assert prepared["notes"] == []

    def test_the_route_and_the_labels_are_there_all_the_same(self, made):
        prepared = made.prepare(videos=False)

        assert prepared["route"]["frames"] == 12
        assert prepared["labels"]["boxes"] == BOXES
        assert prepared["correction"]["correction"]["translation"]["z"] == \
            46.65
        assert made.routes == [prepared]

    def test_the_elevation_model_is_for_the_ground_under_the_poses(self,
                                                                   made):
        made.prepare(videos=False)

        ((west, south, east, north), folder, crs), = made.dems
        published = samples.published_poses()["images"]
        assert (west, east) == (published[0]["lng"], published[-1]["lng"])
        assert (south, north) == (published[-1]["lat"], published[0]["lat"])
        assert (folder, crs) == (made.folder, "EPSG:32633")
        assert bd.read_state(made.target)["elevation"] == "poses"

    def test_the_videos_can_be_fetched_later(self, made):
        made.prepare(videos=False)

        prepared = made.prepare()

        assert prepared["videos"] is True
        assert prepared["files"]["airdata"].endswith("air_data.csv")
        # the model is made anew, for the whole recording
        assert len(made.dems) == 2
        assert made.dems[1][0] == (15.9695, 47.0021, 15.9785, 47.0084)
        assert bd.read_state(made.target)["elevation"] == "recording"
        assert prepared["labels"]["boxes"] == BOXES
        # ... once
        made.prepare()
        assert len(made.dems) == 2

    def test_its_settings_name_no_recordings(self, made):
        prepared = made.prepare(videos=False)

        values = bd.flight_configuration(prepared, {})

        assert values["Input/ThermalVideoPaths"] == ""
        assert values["Input/AirdataPath"] == ""
        assert values["Input/DemPath"].endswith("gem15_dem.glb")
        assert values["Input/TargetCrs"] == "EPSG:32633"
        assert values["Correction/TransZ"] == 46.65

    def test_a_flight_that_was_not_labelled_has_nothing_to_fetch(self,
                                                                 tmp_path):
        made = Made(tmp_path, labelled=False)

        with pytest.raises(bd.DatasetError, match="without its videos "
                                                  "there is nothing"):
            made.prepare(videos=False)

        assert not os.path.exists(made.folder)

    def test_annotations_out_of_reach_are_no_flight(self, made):
        made.session.ranges.files.clear()

        with pytest.raises(bd.DatasetError, match="not where it was "
                                                  "published"):
            made.prepare(videos=False)

    def test_where_the_published_frames_were_taken(self):
        poses = samples.published_poses(3)
        poses["images"].append({"lat": "north"})

        west, south, east, north = bd.poses_bounds(poses)

        assert (west, east) == (15.9770, pytest.approx(15.97702))
        assert (south, north) == (pytest.approx(47.00496), 47.0050)
        assert bd.poses_bounds({"images": [{}]}) is None
        assert bd.poses_bounds(None) is None


# ---------------------------------------------------------------------------
# The flight's settings
# ---------------------------------------------------------------------------

class TestConfiguration:
    @pytest.fixture
    def prepared(self, made):
        return made.prepare()

    def test_every_setting_of_a_flight_is_there(self, prepared):
        values = bd.flight_configuration(prepared, {})

        assert set(values) == set(CONFIG_KEYS)
        stored = {}
        save_config_entries(values, *flights_core.config_writers(stored))
        assert set(stored) == set(CONFIG_KEYS)

    def test_the_recordings_and_the_ground_are_filled_in(self, prepared):
        values = bd.flight_configuration(prepared, {})
        folder = prepared["folder"]

        def there(name):
            return os.path.join(folder, name)

        assert values["Input/VideoMode"] is True
        assert values["Input/EmbeddedSrt"] is False
        assert values["Input/ThermalVideoPaths"] == there(
            "DJI_20241119101935_0003_T.MP4")
        assert values["Input/ThermalSrtPaths"] == there(
            "DJI_20241119101935_0003_T.SRT")
        assert values["Input/RgbVideoPaths"] == there(
            "DJI_20241119101935_0003_W.MP4")
        assert values["Input/RgbSrtPaths"] == there(
            "DJI_20241119101935_0003_W.SRT")
        assert values["Input/AirdataPath"] == there("air_data.csv")
        assert values["Input/DemPath"] == there("gem15_dem.glb")
        assert values["Input/DemMetadataPath"] == there("gem15_dem.json")
        assert values["Input/TargetFolder"] == there("qgis")
        assert values["Input/TargetCrs"] == "EPSG:32633"

    def test_the_calibrations_are_the_ones_the_flight_came_with(self,
                                                                prepared):
        values = bd.flight_configuration(
            prepared, {"Input/ThermalCalibrationPreset": "DJI M30T"},
            custom_calibration="Custom file...")

        assert values["Input/ThermalCalibrationPreset"] == "Custom file..."
        assert values["Input/ThermalCalibrationPath"].endswith("T_calib.json")
        assert values["Input/RgbCalibrationPreset"] == "Custom file..."
        assert values["Input/RgbCalibrationPath"].endswith("W_calib.json")

    def test_a_flight_without_a_calibration_keeps_the_preset(self, prepared):
        prepared["files"]["thermal_calibration"] = ""

        values = bd.flight_configuration(
            prepared, {"Input/ThermalCalibrationPreset": "DJI M30T"})

        assert values["Input/ThermalCalibrationPreset"] == "DJI M30T"

    def test_the_published_correction_is_the_flights(self, prepared):
        values = bd.flight_configuration(prepared, {})

        assert values["Correction/RotationUnit"] == 0          # radians
        assert values["Correction/TransZ"] == 46.65
        assert values["Correction/RotZ"] == pytest.approx(
            0.033 - math.radians(prepared["correction"]["convergence"]))
        assert (values["Correction/TransX"], values["Correction/RotX"]) == (
            0.0, 0.0)
        assert values["Correction/AdditionalCorrections"] == "[]"
        assert values["Input/CorrectionPath"] == prepared["correction"][
            "path"]

    def test_ranges_of_frames_are_stored_as_the_form_stores_them(self):
        correction = bd.flight_correction(RANGES, 0.7)

        settings = bd.correction_settings({"correction": correction,
                                           "path": "/there/c.json"})

        assert json.loads(settings["Correction/AdditionalCorrections"]) == \
            correction["additional"]

    def test_everything_else_is_what_a_new_flight_starts_with(self,
                                                              prepared):
        defaults = {"Detection/Confidence": 0.42,
                    "Correction/TransZ": 99.0}

        values = bd.flight_configuration(prepared, defaults)

        assert values["Detection/Confidence"] == 0.42
        assert values["Detection/SampleRate"] == 1         # the schema's
        assert values["Correction/TransZ"] == 46.65

    def test_a_flight_without_a_correction_keeps_noughts(self, made):
        prepared = made.prepare(correction=False)

        values = bd.flight_configuration(prepared, {})

        assert bd.correction_settings(None) == {}
        assert values["Correction/TransZ"] == 0.0
        assert values["Input/CorrectionPath"] == ""

    @pytest.mark.parametrize("stored, has", [
        ({}, False),
        ({"Correction/TransZ": "0.0", "Correction/RotZ": "0"}, False),
        ({"Correction/AdditionalCorrections": "[]"}, False),
        ({"Correction/AdditionalCorrections": "{}"}, False),
        ({"Correction/AdditionalCorrections": "not json"}, False),
        ({"Correction/TransZ": "abc"}, False),
        ({"Correction/TransZ": "1.5"}, True),
        ({"Correction/RotX": "-0.01"}, True),
        ({"Correction/AdditionalCorrections": '[{"start": 1}]'}, True),
    ])
    def test_whether_a_flight_has_a_correction_of_its_own(self, stored, has):
        assert bd.has_correction(stored) is has
