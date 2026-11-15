# -*- coding: utf-8 -*-
"""Colours, indicators and layer attributes of the animals on the map
(core/map_symbols.py)."""
import pytest

from bambi_wildlife_detection.core import (
    detection_store, label_store, map_symbols, store, track_store)
from bambi_wildlife_detection.core.schema_editor import (
    SchemaEditor, SchemaError)


class TestColours:
    def test_seeded_species_have_fixed_distinct_colours(self):
        names = [name for _id, name in store.BASE_SPECIES + store.SEEDED_SPECIES]
        colours = [map_symbols.default_colour(name) for name in names]
        assert len(set(colours)) == len(names)
        assert map_symbols.default_colour("Roe Deer ") == \
            map_symbols.SEEDED_COLOURS["roe deer"]

    def test_a_new_species_always_gets_the_same_colour(self):
        first = map_symbols.default_colour("lynx")
        assert first in map_symbols.PALETTE
        assert map_symbols.default_colour("lynx") == first
        assert map_symbols.default_colour("") == map_symbols.FALLBACK_COLOUR

    @pytest.mark.parametrize("value,expected", [
        ("#FF7F00", "#ff7f00"), ("ff7f00", "#ff7f00"), (" #a50f15 ", "#a50f15")])
    def test_normalise(self, value, expected):
        assert map_symbols.normalise_colour(value) == expected

    @pytest.mark.parametrize("value", ["red", "#fff", "#12345g", "", None])
    def test_what_is_not_a_colour_is_refused(self, value):
        with pytest.raises(ValueError):
            map_symbols.normalise_colour(value)

    def test_a_flight_without_a_store_has_no_colours(self, tmp_path):
        assert map_symbols.species_colours(str(tmp_path)) == {}
        assert map_symbols.species_colours("") == {}
        assert map_symbols.colour_of({}, "fox") == \
            map_symbols.SEEDED_COLOURS["fox"]
        assert map_symbols.colour_of({"fox": "#000000"}, "fox") == "#000000"
        assert map_symbols.colour_of({}, None) == map_symbols.FALLBACK_COLOUR


class TestProjectColours:
    def test_an_override_is_kept_with_the_project(self, tmp_path):
        root = str(tmp_path)
        with SchemaEditor(root) as editor:
            fox = editor.species_by_name("fox")["species_id"]
            assert not editor.has_own_colour(fox)
            editor.set_species_colour(fox, "#123ABC")
            assert editor.has_own_colour(fox)
            assert editor.species_colours()[fox] == "#123abc"
            editor.commit()
        colours = map_symbols.species_colours(root)
        assert colours["fox"] == "#123abc"
        assert colours["hare"] == map_symbols.SEEDED_COLOURS["hare"]
        assert set(colours) >= {"animal", "unknown", "not-an-animal"}

    def test_the_colour_follows_a_rename(self, tmp_path):
        root = str(tmp_path)
        with SchemaEditor(root) as editor:
            fox = editor.species_by_name("fox")["species_id"]
            editor.set_species_colour(fox, "#123abc")
            editor.rename_species(fox, "red fox")
            editor.commit()
        assert map_symbols.species_colours(root)["red fox"] == "#123abc"

    def test_back_to_the_default(self, tmp_path):
        root = str(tmp_path)
        with SchemaEditor(root) as editor:
            fox = editor.species_by_name("fox")["species_id"]
            editor.set_species_colour(fox, "#123abc")
            editor.set_species_colour(fox, None)
            assert not editor.has_own_colour(fox)
            editor.commit()
        assert map_symbols.species_colours(root)["fox"] == \
            map_symbols.SEEDED_COLOURS["fox"]

    def test_the_base_classes_can_have_a_colour(self, tmp_path):
        root = str(tmp_path)
        with SchemaEditor(root) as editor:
            editor.set_species_colour(0, "#000000")
            editor.commit()
        assert map_symbols.species_colours(root)["animal"] == "#000000"

    def test_an_uncommitted_colour_is_discarded(self, tmp_path):
        root = str(tmp_path)
        with SchemaEditor(root) as editor:
            editor.set_species_colour(1, "#123abc")
        assert map_symbols.species_colours(root)["roe deer"] == \
            map_symbols.SEEDED_COLOURS["roe deer"]

    def test_bad_input_is_a_schema_error(self, tmp_path):
        with SchemaEditor(str(tmp_path)) as editor:
            with pytest.raises(SchemaError, match="Not a colour"):
                editor.set_species_colour(1, "reddish")
            with pytest.raises(SchemaError, match="No species"):
                editor.set_species_colour(999, "#000000")

    def test_deleting_a_species_drops_its_colour(self, tmp_path):
        root = str(tmp_path)
        with SchemaEditor(root) as editor:
            lynx = editor.add_species("lynx")
            editor.set_species_colour(lynx, "#123abc")
            editor.delete_species(lynx)
            assert map_symbols.stored_colours(editor.conn) == {}
            editor.commit()

    def test_a_damaged_entry_is_ignored(self, tmp_path):
        root = str(tmp_path)
        conn = store.open_store(store.project_path(root), store.PROJECT)
        store.set_meta(conn, map_symbols.META_KEY,
                       '{"1": "#123abc", "x": "#000000", "2": "nope"}')
        assert map_symbols.stored_colours(conn) == {1: "#123abc"}
        store.set_meta(conn, map_symbols.META_KEY, "not json")
        assert map_symbols.stored_colours(conn) == {}
        conn.close()


class TestMarks:
    @pytest.mark.parametrize("sex,expected", [
        ("female", "♀"), ("Male", "♂"), ("unknown", ""), ("", ""),
        (None, ""), ("hermaphrodite", "H")])       # a project's own label
    def test_sex_mark(self, sex, expected):
        assert map_symbols.sex_mark(sex) == expected

    @pytest.mark.parametrize("age,expected", [
        ("adult", "a"), ("Juvenile", "j"), ("subadult", "s"),
        ("unknown", ""), (None, ""), ("yearling", "y")])
    def test_age_mark(self, age, expected):
        assert map_symbols.age_mark(age) == expected

    def test_an_individual_keeps_its_colour(self):
        first = map_symbols.individual_colour(7)
        assert first in map_symbols.INDIVIDUAL_PALETTE
        assert map_symbols.individual_colour(7) == first
        assert map_symbols.individual_colour(8) != first
        assert map_symbols.individual_colour(None) == \
            map_symbols.UNTRACKED_COLOUR

    def test_attributes_keep_what_is_known(self):
        assert map_symbols.attributes("roe deer", "female", "unknown", 3) == {
            "species": "roe deer", "sex": "female", "age": "",
            "sex_mark": "♀", "age_mark": "",
            "box_colour": map_symbols.individual_colour(3)}
        assert map_symbols.attributes() == {
            "species": "", "sex": "", "age": "", "sex_mark": "",
            "age_mark": "", "box_colour": map_symbols.UNTRACKED_COLOUR}
        assert tuple(map_symbols.attributes()) == map_symbols.FIELDS

    def test_species_present(self):
        assert map_symbols.species_present(
            ["fox", None, "", "hare", "fox"]) == ["fox", "hare"]


@pytest.fixture
def flight(tmp_path):
    """Track 1: a red deer; track 2: a hare; one detection in no track,
    mapped to roe deer."""
    root = str(tmp_path)
    label_store.vocabulary(root, create=True)
    names = label_store.vocabulary(root)["species_by_name"]
    project = store.open_store(store.project_path(root), store.PROJECT)
    source_id = detection_store.ensure_source(project, detection_store.DETECTOR)
    project.execute(
        "INSERT INTO class_mapping (source_id, source_class, species_id) "
        "VALUES (?, ?, ?)", (source_id, "0", names["roe deer"]))
    project.commit()
    project.close()
    detection_store.record_detections(root, "t", [
        {"frame": frame, "x1": 1.0, "y1": 1.0, "x2": 5.0, "y2": 5.0,
         "confidence": 0.9, "source_class": "0"} for frame in range(4)])
    ids = [d["detection_id"] for d in track_store.load_detections(root, "t")]
    species = {1: names["red deer"], 2: names["hare"]}
    track_store.record_tracks(root, "t", [
        {"track_id": 1, "detection_id": ids[0]},
        {"track_id": 1, "detection_id": ids[1]},
        {"track_id": 2, "detection_id": ids[2]}],
        species_of=lambda legacy: species[legacy])
    return root, ids


class TestFromTheStores:
    def test_tracks_are_what_the_inventory_says(self, flight):
        root, _ids = flight
        tracks = map_symbols.track_attributes(root, "t")
        assert set(tracks) == {1, 2}
        assert tracks[1]["species"] == "red deer"
        assert tracks[2] == {"species": "hare", "sex": "", "age": "",
                             "sex_mark": "", "age_mark": "",
                             "box_colour": map_symbols.individual_colour(2)}
        assert set(map_symbols.track_attributes(root, "t", [2])) == {2}

    def test_a_tracked_detection_is_what_its_track_is(self, flight):
        root, ids = flight
        detections = map_symbols.detection_attributes(root, "t")
        assert set(detections) == set(ids)
        assert detections[ids[0]]["species"] == "red deer"
        assert detections[ids[2]]["species"] == "hare"
        # ... and has its track's box colour
        assert detections[ids[0]]["box_colour"] == \
            map_symbols.individual_colour(1)
        # in no track: the species its class maps to, the neutral box
        assert detections[ids[3]] == {
            "species": "roe deer", "sex": "", "age": "", "sex_mark": "",
            "age_mark": "", "box_colour": map_symbols.UNTRACKED_COLOUR}

    def test_nothing_stored_is_nothing_to_draw(self, tmp_path):
        assert map_symbols.track_attributes(str(tmp_path), "t") == {}
        assert map_symbols.detection_attributes(str(tmp_path), "t") == {}
