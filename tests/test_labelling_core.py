# -*- coding: utf-8 -*-
"""Unit tests for core.labelling (labelling tool data model & geometry).

The GUI dialog re-exports everything tested here; these tests import from
the new core location. ``_GeoPropagator`` is exercised with fake
alfspy/trimesh/bambi modules and the identity camera from tests.fakes.
"""
import json
import sys

import pytest

from bambi_wildlife_detection.core.labelling import (
    CustomField,
    LabelStore,
    LabelTrack,
    SPECIES_CLASSES,
    TRACK_COLORS_RGB,
    _FrameMatcher,
    _GeoPropagator,
    box_in_valid_area,
    coerce_attributes,
    custom_fields_from_dicts,
    find_overlapping_tracks,
    group_track_ids,
    keyframe_window,
    load_valid_mask,
    _load_detections_by_frame,
    _load_pixel_tracks,
    _pose_epochs,
    merge_tracks,
    propagation_frames,
    read_custom_fields,
    split_track,
    track_color_rgb,
    track_world_positions,
    validate_custom_fields,
    write_custom_fields,
)
from tests.fakes import install_fake_render_stack, make_module


class TestTrackColor:
    def test_cycles_through_palette(self):
        assert track_color_rgb(0) == TRACK_COLORS_RGB[0]
        assert track_color_rgb(len(TRACK_COLORS_RGB)) == TRACK_COLORS_RGB[0]
        assert track_color_rgb(3) == TRACK_COLORS_RGB[3]


class TestLabelTrack:
    def test_interpolation_between_keyframes(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (0, 0, 10, 10))
        track.set_keyframe(10, (100, 50, 110, 60))
        box, is_kf, occ = track.box_at(5)
        assert box == (50.0, 25.0, 60.0, 35.0)
        assert is_kf is False
        assert occ == "none"

    def test_keyframe_returned_exactly(self):
        track = LabelTrack(1)
        track.set_keyframe(3, (1, 2, 3, 4), occlusion="partially")
        box, is_kf, occ = track.box_at(3)
        assert box == (1.0, 2.0, 3.0, 4.0)
        assert is_kf is True
        assert occ == "partially"

    def test_outside_range_returns_none(self):
        track = LabelTrack(1)
        track.set_keyframe(5, (0, 0, 1, 1))
        track.set_keyframe(8, (0, 0, 1, 1))
        assert track.box_at(4) is None
        assert track.box_at(9) is None
        assert LabelTrack(2).box_at(0) is None

    def test_stop_frame_creates_gap(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (0, 0, 1, 1), stop=True)
        track.set_keyframe(10, (5, 5, 6, 6))
        assert track.box_at(0) is not None      # the stop frame itself has a box
        assert track.box_at(5) is None          # gap after the stop frame
        assert track.box_at(10) is not None

    def test_visible_segments_split_at_stops(self):
        track = LabelTrack(1)
        for f in (0, 5, 10, 20):
            track.set_keyframe(f, (0, 0, 1, 1))
        assert track.visible_segments() == [(0, 20)]
        track.set_keyframe(5, (0, 0, 1, 1), stop=True)
        assert track.visible_segments() == [(0, 5), (10, 20)]

    def test_occlusion_inherited_from_interpolation_state(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (0, 0, 1, 1), occlusion="partially")
        track.set_keyframe(10, (0, 0, 1, 1), occlusion="fully")
        # a new keyframe in between inherits the interpolated occlusion
        track.set_keyframe(5, (2, 2, 3, 3))
        assert track.keyframes[5]["occlusion"] == "partially"

    def test_box_update_preserves_stop_and_occlusion(self):
        track = LabelTrack(1)
        track.set_keyframe(4, (0, 0, 1, 1), occlusion="fully", stop=True)
        track.set_keyframe(4, (9, 9, 10, 10))  # box-only update
        assert track.is_stop(4) is True
        assert track.keyframes[4]["occlusion"] == "fully"

    def test_remove_keyframe(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (0, 0, 1, 1))
        assert track.remove_keyframe(0) is True
        assert track.remove_keyframe(0) is False
        assert track.frame_range() is None

    def test_transform_keyframes_translates_all(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (0, 0, 10, 10))
        track.set_keyframe(10, (100, 50, 110, 60))
        # move the frame-0 box by (+5, +3), same size
        track.transform_keyframes((0, 0, 10, 10), (5, 3, 15, 13))
        assert track.box_at(0)[0] == (5.0, 3.0, 15.0, 13.0)
        assert track.box_at(10)[0] == (105.0, 53.0, 115.0, 63.0)

    def test_transform_keyframes_scales_about_each_centre(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (0, 0, 10, 10))
        track.set_keyframe(10, (100, 100, 120, 140))
        # double the width, keep height and centre of the frame-0 box
        track.transform_keyframes((0, 0, 10, 10), (-5, 0, 15, 10))
        assert track.box_at(0)[0] == (-5.0, 0.0, 15.0, 10.0)
        assert track.box_at(10)[0] == (90.0, 100.0, 130.0, 140.0)

    def test_transform_keyframes_commutes_with_interpolation(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (0, 0, 10, 10))
        track.set_keyframe(10, (100, 50, 130, 90))
        old = track.box_at(5)[0]
        # drag the interpolated frame-5 box: shift and scale it
        new = (old[0] + 8, old[1] - 2, old[2] + 20, old[3] + 4)
        track.transform_keyframes(old, new)
        assert track.box_at(5)[0] == pytest.approx(new)

    def test_transform_keyframes_clamps_to_bounds(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (2, 2, 12, 12))
        track.set_keyframe(10, (90, 90, 98, 98))
        track.transform_keyframes(
            (2, 2, 12, 12), (7, 7, 17, 17), bounds=(100, 100))
        assert track.box_at(0)[0] == (7.0, 7.0, 17.0, 17.0)
        assert track.box_at(10)[0] == (95.0, 95.0, 100.0, 100.0)

    def test_transform_keyframes_degenerate_old_box_keeps_sizes(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (10, 10, 20, 20))
        # zero-sized reference box: only the translation is applied
        track.transform_keyframes((5, 5, 5, 5), (8, 9, 8, 9))
        assert track.box_at(0)[0] == (13.0, 14.0, 23.0, 24.0)

    def test_dict_round_trip(self):
        track = LabelTrack(7, species="red deer", sex="female", age="adult")
        track.set_keyframe(2, (1, 2, 3, 4), occlusion="partially", stop=True)
        track.set_keyframe(9, (5, 6, 7, 8))
        clone = LabelTrack.from_dict(json.loads(json.dumps(track.to_dict())))
        assert clone.track_id == 7
        assert clone.species == "red deer"
        assert clone.keyframes == track.keyframes


class TestLabelStore:
    @pytest.fixture
    def store(self, tmp_path):
        return LabelStore(str(tmp_path), "t")

    def _add_track(self, store, track_id=1, species="red deer"):
        track = LabelTrack(track_id, species=species)
        track.set_keyframe(0, (0, 0, 10, 10))
        track.set_keyframe(2, (20, 20, 30, 30))
        store.tracks[track_id] = track
        return track

    def test_save_load_round_trip(self, store, tmp_path):
        self._add_track(store)
        store.save()
        reloaded = LabelStore(str(tmp_path), "t")
        reloaded.load()
        assert reloaded.tracks[1].keyframes == store.tracks[1].keyframes
        assert reloaded.tracks[1].species == "red deer"

    def test_next_track_id(self, store):
        assert store.next_track_id() == 1
        self._add_track(store, 5)
        assert store.next_track_id() == 6

    def test_species_class_ids_appends_custom_species(self, store):
        self._add_track(store, 1, species="red deer")
        self._add_track(store, 2, species="Moose")   # custom, mixed case
        self._add_track(store, 3, species="ibex")    # custom
        mapping = store.species_class_ids()
        assert mapping["red deer"] == SPECIES_CLASSES.index("red deer")
        # customs appended alphabetically after the default taxonomy
        assert mapping["ibex"] == len(SPECIES_CLASSES)
        assert mapping["moose"] == len(SPECIES_CLASSES) + 1

    def test_csv_export_contract(self, store):
        self._add_track(store)
        store.save()
        with open(store.csv_path, encoding="utf-8") as fh:
            lines = fh.readlines()
        assert lines[0].startswith("# frame,track_id,x1")
        # 3 interpolated frames (0, 1, 2)
        assert len(lines) == 4
        frame1 = lines[2].strip().split(",")
        assert frame1[0] == "1"
        assert frame1[2] == "10.00"     # interpolated x1
        assert frame1[-1] == "0"        # not a keyframe

    def test_export_to_detections_and_round_trip(self, store):
        self._add_track(store)
        _det_file, count = store.export_to_detections()
        assert count == 3
        by_frame = _load_detections_by_frame(store.target_folder, "t")
        assert sorted(by_frame.keys()) == [0, 1, 2]
        x1, y1, x2, y2, conf, cls = by_frame[1][0]
        assert (x1, y1) == (10.0, 10.0)
        assert conf == 1.0
        # The store resolves the species by name, so the class is its
        # species_id rather than the positional index detections.txt used.
        from bambi_wildlife_detection.core import label_store

        by_name = {entry["name"]: entry["species_id"]
                   for entry in label_store.vocabulary(
                       store.target_folder)["species"]}
        assert cls == by_name["red deer"]

    def test_export_preserves_detector_output_and_replaces_block(self, store, tmp_path):
        det_dir = tmp_path / "detections_t"
        det_dir.mkdir()
        det_file = det_dir / "detections.txt"
        det_file.write_text(
            "# frame x1 y1 x2 y2 confidence class_id\n"
            "0 1.00 1.00 2.00 2.00 0.8000 1\n")
        self._add_track(store)
        store.export_to_detections()
        store.export_to_detections()  # re-export must replace, not append
        content = det_file.read_text()
        assert content.count(LabelStore.DETECTIONS_MARKER) == 1
        assert "0.8000 1" in content            # detector line preserved
        assert "# class_id mapping:" in content

    def test_replace_detections_discards_detector_output_and_tracks(
            self, store, tmp_path):
        det_dir = tmp_path / "detections_t"
        det_dir.mkdir()
        (det_dir / "detections.txt").write_text(
            "# frame x1 y1 x2 y2 confidence class_id\n"
            "0 1.00 1.00 2.00 2.00 0.8000 1\n")
        tracks_dir = tmp_path / "tracks_t"
        tracks_dir.mkdir()
        (tracks_dir / "tracks_pixel.csv").write_text("0,1,0,0,1,1,0.9,1,0\n")
        px_dir = tmp_path / "tracks_pixel_t"
        px_dir.mkdir()
        (px_dir / "tracks_pixel.csv").write_text("0,1,0,0,1,1,0.9,1,0\n")
        other_dir = tmp_path / "tracks_w"       # other modality stays
        other_dir.mkdir()

        self._add_track(store)
        det_file, count, removed = store.replace_detections()

        assert count == 3
        content = (det_dir / "detections.txt").read_text()
        assert "0.8000 1" not in content        # detector line discarded
        assert content.count(LabelStore.DETECTIONS_MARKER) == 1
        by_frame = _load_detections_by_frame(store.target_folder, "t")
        assert sorted(by_frame.keys()) == [0, 1, 2]
        assert not tracks_dir.exists()
        assert not px_dir.exists()
        assert other_dir.exists()
        assert sorted(removed) == sorted([str(tracks_dir), str(px_dir)])

        # A later merge export keeps a single marker block.
        store.export_to_detections()
        assert (det_dir / "detections.txt").read_text().count(
            LabelStore.DETECTIONS_MARKER) == 1

    def test_replace_detections_without_track_folders(self, store, tmp_path):
        self._add_track(store)
        det_file, count, removed = store.replace_detections()
        assert count == 3
        assert removed == []
        assert (tmp_path / "detections_t" / "detections.txt").is_file()


class TestCustomField:
    def test_rejects_empty_reserved_and_unknown(self):
        with pytest.raises(ValueError, match="needs a name"):
            CustomField("  ")
        with pytest.raises(ValueError, match="reserved"):
            CustomField("Species")          # reserved, case-insensitive
        with pytest.raises(ValueError, match="type"):
            CustomField("weight", "decimal")
        with pytest.raises(ValueError, match="scope"):
            CustomField("weight", "float", "video")

    def test_name_is_stripped(self):
        assert CustomField(" weight ", "float").name == "weight"

    @pytest.mark.parametrize("type_, raw, expected", [
        ("int", "42", 42),
        ("int", "3.0", 3),
        ("int", True, 1),
        ("float", "1.5", 1.5),
        ("string", 12, "12"),
        ("bool", "yes", True),
        ("bool", "0", False),
        ("bool", "", False),
        ("datetime", "2023-09-20T10:00:00", "2023-09-20T10:00:00"),
        ("datetime", "", ""),
    ])
    def test_coerce(self, type_, raw, expected):
        assert CustomField("f", type_).coerce(raw) == expected

    @pytest.mark.parametrize("type_, raw", [
        ("int", "abc"), ("float", "abc"), ("bool", "maybe"),
        ("datetime", "not-a-date"), ("int", None),
    ])
    def test_coerce_rejects_bad_values(self, type_, raw):
        with pytest.raises(ValueError):
            CustomField("f", type_).coerce(raw)

    def test_coerce_accepts_datetime_objects(self):
        from datetime import datetime
        value = CustomField("f", "datetime").coerce(datetime(2023, 9, 20))
        assert value == "2023-09-20T00:00:00"

    def test_defaults_per_type(self):
        assert [CustomField("f", t).default
                for t in ("int", "float", "string", "bool", "datetime")] == \
            [0, 0.0, "", False, ""]

    def test_dict_round_trip_and_equality(self):
        field = CustomField("weight", "float", "keyframe")
        assert CustomField.from_dict(field.to_dict()) == field
        assert field != CustomField("weight", "float", "track")

    def test_from_dicts_skips_invalid_and_duplicates(self):
        fields = custom_fields_from_dicts([
            {"name": "weight", "type": "float", "scope": "track"},
            {"name": "weight", "type": "int", "scope": "track"},   # duplicate
            {"name": "species", "type": "string"},                 # reserved
            {"name": "", "type": "string"},                        # unnamed
            {"name": "note", "type": "bogus"},                     # bad type
        ])
        assert [f.name for f in fields] == ["weight"]

    def test_validate_rejects_case_insensitive_duplicates(self):
        with pytest.raises(ValueError, match="Duplicate"):
            validate_custom_fields([CustomField("Note"), CustomField("note")])

    def test_coerce_attributes_drops_unknown_and_unfitting(self):
        fields = [CustomField("count", "int"), CustomField("note", "string")]
        result = coerce_attributes(
            {"count": "7", "note": "hi", "gone": 1, }, fields)
        assert result == {"count": 7, "note": "hi"}
        # a value that does not fit the (re)typed field is dropped
        assert coerce_attributes({"count": "abc"}, fields) == {}


class TestCustomFieldSharing:
    """Export / import of the field schema as a standalone JSON file."""

    SCHEMA = [CustomField("weight", "float", "track"),
              CustomField("blurry", "bool", "keyframe")]

    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "fields.json")
        write_custom_fields(path, self.SCHEMA)
        assert read_custom_fields(path) == self.SCHEMA

    def test_exported_file_is_self_describing(self, tmp_path):
        path = tmp_path / "fields.json"
        write_custom_fields(str(path), self.SCHEMA)
        data = json.loads(path.read_text())
        assert data["format"] == "bambi-labelling-fields"
        assert data["version"] == 1
        assert [f["name"] for f in data["custom_fields"]] == ["weight", "blurry"]

    def test_empty_schema_round_trips(self, tmp_path):
        path = str(tmp_path / "fields.json")
        write_custom_fields(path, [])
        assert read_custom_fields(path) == []

    def test_reads_a_flight_labels_json(self, tmp_path):
        """A colleague's labels.json is a valid import source."""
        store = LabelStore(str(tmp_path), "t")
        store.set_custom_fields(self.SCHEMA)
        store.save()
        assert read_custom_fields(store.json_path) == self.SCHEMA

    def test_reads_a_bare_list(self, tmp_path):
        path = tmp_path / "fields.json"
        path.write_text(json.dumps([{"name": "weight", "type": "float",
                                     "scope": "track"}]))
        assert [f.name for f in read_custom_fields(str(path))] == ["weight"]

    @pytest.mark.parametrize("content, message", [
        ("{ not json", "valid JSON"),
        ('{"format": "something-else"}', "Unknown file format"),
        ('{"format": "bambi-labelling-fields", "version": 99, '
         '"custom_fields": []}', "newer version"),
        ('{"format": "bambi-labelling-fields", "version": "x", '
         '"custom_fields": []}', "Invalid file version"),
        ('{"modality": "t", "tracks": []}', "no 'custom_fields'"),
        ('"a string"', "no custom field definitions"),
        ('{"custom_fields": {"weight": "float"}}', "must be a list"),
        ('{"custom_fields": ["weight"]}', "Field 1 is not a field definition"),
        ('{"custom_fields": [{"name": "species"}]}', "Field 1: .*reserved"),
        ('{"custom_fields": [{"name": "a"}, {"name": "A"}]}', "Duplicate"),
    ])
    def test_rejects_bad_files(self, tmp_path, content, message):
        path = tmp_path / "fields.json"
        path.write_text(content)
        with pytest.raises(ValueError, match=message):
            read_custom_fields(str(path))

    def test_missing_file_raises_oserror(self, tmp_path):
        with pytest.raises(OSError):
            read_custom_fields(str(tmp_path / "nope.json"))

    def test_import_is_strict_where_settings_seeding_is_lenient(self, tmp_path):
        """A bad field fails the import instead of being silently dropped."""
        entries = [{"name": "weight", "type": "float"},
                   {"name": "note", "type": "bogus"}]
        path = tmp_path / "fields.json"
        path.write_text(json.dumps({"custom_fields": entries}))
        with pytest.raises(ValueError, match="Field 2"):
            read_custom_fields(str(path))
        # the lenient path (QSettings seeding) keeps the usable field
        assert [f.name for f in custom_fields_from_dicts(entries)] == ["weight"]


class TestCustomFieldValues:
    def test_track_attributes_round_trip(self):
        track = LabelTrack(1, attributes={"weight": 12.5})
        clone = LabelTrack.from_dict(json.loads(json.dumps(track.to_dict())))
        assert clone.attributes == {"weight": 12.5}

    def test_empty_attributes_are_not_serialised(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (0, 0, 1, 1))
        assert "attributes" not in track.to_dict()
        assert "attributes" not in track.keyframes[0]

    def test_keyframe_attributes_round_trip(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (0, 0, 1, 1), attributes={"blurry": True})
        clone = LabelTrack.from_dict(json.loads(json.dumps(track.to_dict())))
        assert clone.keyframes[0]["attributes"] == {"blurry": True}

    def test_interpolated_frames_inherit_previous_keyframe(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (0, 0, 1, 1), attributes={"blurry": True})
        track.set_keyframe(10, (0, 0, 1, 1), attributes={"blurry": False})
        assert track.attributes_at(5) == {"blurry": True}
        assert track.attributes_at(10) == {"blurry": False}

    def test_attributes_at_returns_none_without_a_box(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (0, 0, 1, 1), stop=True)
        track.set_keyframe(10, (0, 0, 1, 1))
        assert track.attributes_at(5) is None     # gap after the stop frame
        assert track.attributes_at(20) is None    # outside the range

    def test_new_keyframe_inherits_interpolated_attributes(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (0, 0, 1, 1), attributes={"blurry": True})
        track.set_keyframe(10, (0, 0, 1, 1), attributes={"blurry": False})
        track.set_keyframe(5, (2, 2, 3, 3))      # box-only
        assert track.keyframes[5]["attributes"] == {"blurry": True}

    def test_box_update_preserves_attributes(self):
        track = LabelTrack(1)
        track.set_keyframe(4, (0, 0, 1, 1), attributes={"blurry": True})
        track.set_keyframe(4, (9, 9, 10, 10))
        assert track.keyframes[4]["attributes"] == {"blurry": True}


class TestLabelStoreCustomFields:
    @pytest.fixture
    def store(self, tmp_path):
        return LabelStore(str(tmp_path), "t")

    def _track_with_values(self, store):
        track = LabelTrack(1, attributes={"weight": 12.5})
        track.set_keyframe(0, (0, 0, 1, 1), attributes={"blurry": True})
        track.set_keyframe(2, (0, 0, 1, 1), attributes={"blurry": False})
        store.tracks[1] = track
        return track

    def _schema(self):
        return [CustomField("weight", "float", "track"),
                CustomField("blurry", "bool", "keyframe")]

    def test_schema_and_values_round_trip(self, store, tmp_path):
        store.set_custom_fields(self._schema())
        self._track_with_values(store)
        store.save()

        reloaded = LabelStore(str(tmp_path), "t")
        reloaded.load()
        assert reloaded.schema_defined
        assert reloaded.custom_fields == self._schema()
        assert reloaded.tracks[1].attributes == {"weight": 12.5}
        assert reloaded.tracks[1].keyframes[0]["attributes"] == {"blurry": True}

    def test_schema_defined_false_for_legacy_file(self, store, tmp_path):
        (tmp_path / "labels_t").mkdir()
        (tmp_path / "labels_t" / "labels.json").write_text(
            json.dumps({"modality": "t", "tracks": []}))
        store.load()
        assert store.schema_defined is False
        assert store.custom_fields == []

    def test_load_drops_values_without_a_matching_field(self, store, tmp_path):
        store.set_custom_fields(self._schema())
        self._track_with_values(store)
        store.save()
        # hand-edit the file: schema shrinks, values stay behind
        data = json.loads((tmp_path / "labels_t" / "labels.json").read_text())
        data["custom_fields"] = [{"name": "weight", "type": "float",
                                  "scope": "track"}]
        (tmp_path / "labels_t" / "labels.json").write_text(json.dumps(data))

        reloaded = LabelStore(str(tmp_path), "t")
        reloaded.load()
        assert reloaded.tracks[1].attributes == {"weight": 12.5}
        assert "attributes" not in reloaded.tracks[1].keyframes[0]

    def test_set_custom_fields_removes_dropped_values(self, store):
        store.set_custom_fields(self._schema())
        track = self._track_with_values(store)
        store.set_custom_fields([CustomField("weight", "float", "track")])
        assert track.attributes == {"weight": 12.5}
        assert "attributes" not in track.keyframes[0]

    def test_set_custom_fields_converts_retyped_values(self, store):
        store.set_custom_fields([CustomField("weight", "float", "track")])
        store.tracks[1] = LabelTrack(1, attributes={"weight": 12.5})
        store.set_custom_fields([CustomField("weight", "int", "track")])
        assert store.tracks[1].attributes == {"weight": 12}

    def test_set_custom_fields_drops_values_on_scope_change(self, store):
        store.set_custom_fields(self._schema())
        track = self._track_with_values(store)
        store.set_custom_fields([CustomField("weight", "float", "keyframe")])
        assert track.attributes == {}

    def test_set_custom_fields_rejects_duplicates(self, store):
        with pytest.raises(ValueError, match="Duplicate"):
            store.set_custom_fields([CustomField("a"), CustomField("a")])

    def test_count_values(self, store):
        store.set_custom_fields(self._schema())
        self._track_with_values(store)
        weight, blurry = self._schema()
        assert store.count_values(weight) == 1
        assert store.count_values(blurry) == 2       # one per key frame
        assert store.count_values(CustomField("nope", "int")) == 0

    def test_fields_for_scope(self, store):
        store.set_custom_fields(self._schema())
        assert [f.name for f in store.fields_for("track")] == ["weight"]
        assert [f.name for f in store.fields_for("keyframe")] == ["blurry"]

    def test_csv_and_detections_exports_ignore_custom_fields(self, store):
        store.set_custom_fields(self._schema())
        track = self._track_with_values(store)
        track.species = "red deer"
        store.save()
        header, *rows = open(store.csv_path, encoding="utf-8").readlines()
        assert header.strip().endswith("occlusion,keyframe")
        assert all(len(r.strip().split(",")) == 11 for r in rows)
        _det_file, count = store.export_to_detections()
        assert count == 3


class TestPoseEpochs:
    def test_parses_iso_timestamps(self):
        epochs = _pose_epochs([
            {"timestamp": "2023-09-20T10:00:00+02:00"},
            {"timestamp": ""},
            {"timestamp": "not-a-date"},
            {},
        ])
        assert epochs[0] is not None
        assert epochs[1:] == [None, None, None]


class TestFrameMatcher:
    def _images(self, *offsets):
        return [{"timestamp": f"2023-09-20T10:00:{o:02d}+02:00"} for o in offsets]

    def test_matches_nearest_by_time(self):
        src = self._images(0, 10, 20)
        dst = self._images(1, 9, 21)
        matcher = _FrameMatcher(src, dst)
        assert matcher.usable
        assert matcher.match(0) == (0, pytest.approx(1.0))
        assert matcher.match(1) == (1, pytest.approx(1.0))
        assert matcher.match(2) == (2, pytest.approx(1.0))

    def test_out_of_range_and_missing_timestamps(self):
        matcher = _FrameMatcher(self._images(0), self._images(5))
        assert matcher.match(-1) is None
        assert matcher.match(1) is None
        no_ts = _FrameMatcher([{}], [{}])
        assert not no_ts.usable
        assert no_ts.match(0) is None


class TestLoaders:
    """The read-only overlays come from the store, like everything else.

    They used to parse detections.txt and tracks_pixel.csv, so the overlay
    could disagree with the store the tool writes into.
    """

    @staticmethod
    def _project(root):
        from bambi_wildlife_detection.core import detection_store, track_store

        detection_store.record_detections(root, "t", [
            {"frame": 0, "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0,
             "confidence": 0.9, "source_class": "0"},
            {"frame": 0, "x1": 5.0, "y1": 6.0, "x2": 7.0, "y2": 8.0,
             "confidence": 0.8, "source_class": "0"},
            {"frame": 1, "x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0,
             "confidence": 0.7, "source_class": "0"},
        ])
        ids = [d["detection_id"]
               for d in track_store.load_detections(root, "t")]
        track_store.record_tracks(root, "t", [
            {"track_id": 1, "detection_id": ids[2]},
            {"track_id": 1, "detection_id": ids[0]},
        ])
        return ids

    def test_load_detections_by_frame(self, tmp_path):
        root = str(tmp_path)
        self._project(root)
        result = _load_detections_by_frame(root, "t")
        assert len(result[0]) == 2
        assert result[0][0][:4] == (1.0, 2.0, 3.0, 4.0)
        assert len(result[1]) == 1

    def test_load_pixel_tracks_sorted_by_frame(self, tmp_path):
        root = str(tmp_path)
        self._project(root)
        result = _load_pixel_tracks(root, "t")
        entries = list(result.values())[0]
        assert [e["frame"] for e in entries] == [0, 1]

    def test_the_text_files_are_not_read(self, tmp_path):
        det = tmp_path / "detections_t"
        det.mkdir()
        (det / "detections.txt").write_text("0 1 2 3 4 0.9 2")
        trk = tmp_path / "tracks_t"
        trk.mkdir()
        (trk / "tracks_pixel.csv").write_text("0,1,0,0,1,1,0.9,2")

        assert _load_detections_by_frame(str(tmp_path), "t") == {}
        assert _load_pixel_tracks(str(tmp_path), "t") == {}

    def test_missing_files_return_empty(self, tmp_path):
        assert _load_detections_by_frame(str(tmp_path), "t") == {}
        assert _load_pixel_tracks(str(tmp_path), "t") == {}

    def test_an_unknown_modality_is_empty(self, tmp_path):
        assert _load_detections_by_frame(str(tmp_path), "x") == {}
        assert _load_pixel_tracks(str(tmp_path), "x") == {}


# ---------------------------------------------------------------------------
# _GeoPropagator (fake alfspy / trimesh / bambi stack)
# ---------------------------------------------------------------------------

WORLD_QUAD = [(-0.5, -0.25, 0.0), (0.5, -0.25, 0.0),
              (0.5, 0.25, 0.0), (-0.5, 0.25, 0.0)]


@pytest.fixture
def fake_geo_stack(monkeypatch):
    """alfspy/trimesh/bambi fakes: ray-casting always returns WORLD_QUAD."""
    alfspy = install_fake_render_stack(monkeypatch)
    rendering = sys.modules["alfspy.core.rendering"]
    rendering.Resolution = lambda w, h: (w, h)

    render_mod = make_module("alfspy.render")
    render_mod.__path__ = []
    render_render = make_module("alfspy.render.render")

    class _MeshData:
        vertices = [(0, 0, 0)]
        indices = [(0, 0, 0)]

    render_render.read_gltf = lambda path: (_MeshData(), None)
    alfspy.render = render_mod
    render_mod.render = render_render
    monkeypatch.setitem(sys.modules, "alfspy.render", render_mod)
    monkeypatch.setitem(sys.modules, "alfspy.render.render", render_render)

    class Trimesh:
        def __init__(self, vertices=None, faces=None):
            self.vertices = vertices
            self.faces = faces

    monkeypatch.setitem(sys.modules, "trimesh", make_module("trimesh", Trimesh=Trimesh))

    projection_util = make_module("bambi.util.projection_util")
    projection_util.label_to_world_coordinates = (
        lambda label_coords, resolution, mesh, cam: list(WORLD_QUAD))
    util_pkg = make_module("bambi.util")
    util_pkg.__path__ = []
    util_pkg.projection_util = projection_util
    bambi_pkg = make_module("bambi")
    bambi_pkg.__path__ = []
    bambi_pkg.util = util_pkg
    monkeypatch.setitem(sys.modules, "bambi", bambi_pkg)
    monkeypatch.setitem(sys.modules, "bambi.util", util_pkg)
    monkeypatch.setitem(sys.modules, "bambi.util.projection_util", projection_util)


def _images(n=3):
    return [{"location": [0.0, 0.0, 10.0], "rotation": [0.0, 0.0, 0.0],
             "fovy": [50.0]} for _ in range(n)]


class TestPropagationFrames:
    def test_no_sampling_returns_target_only(self):
        assert propagation_frames(0, 25, 0) == [25]

    def test_samples_forward_and_ends_on_target(self):
        assert propagation_frames(0, 25, 10) == [10, 20, 25]

    def test_samples_backward(self):
        assert propagation_frames(25, 0, 10) == [15, 5, 0]

    def test_target_on_the_grid_is_not_duplicated(self):
        assert propagation_frames(0, 20, 10) == [10, 20]

    def test_step_larger_than_offset_yields_target_only(self):
        assert propagation_frames(0, 5, 10) == [5]

    def test_same_frame_yields_nothing(self):
        assert propagation_frames(7, 7, 10) == []


class TestKeyframeWindow:
    def test_short_lists_are_shown_in_full(self):
        frames = [0, 5, 10]
        assert keyframe_window(frames, 5, max_shown=12) == frames

    def test_a_full_list_is_not_elided(self):
        frames = list(range(12))
        assert keyframe_window(frames, 0, max_shown=12) == frames

    def test_the_window_follows_the_current_frame(self):
        frames = list(range(0, 200, 10))  # 20 key frames
        # first + last, and the 10 remaining slots centred on frame 100
        assert keyframe_window(frames, 100, max_shown=12) == [
            0, None, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, None, 190]

    def test_first_and_last_are_always_kept(self):
        frames = list(range(0, 200, 10))
        for current in (0, 55, 100, 190, 500):
            window = keyframe_window(frames, current, max_shown=12)
            assert window[0] == frames[0]
            assert window[-1] == frames[-1]

    def test_no_leading_gap_when_the_window_starts_at_the_front(self):
        frames = list(range(0, 200, 10))
        window = keyframe_window(frames, 0, max_shown=12)
        assert window[:3] == [0, 10, 20]
        assert window[-2:] == [None, 190]

    def test_no_trailing_gap_when_the_window_reaches_the_end(self):
        frames = list(range(0, 200, 10))
        window = keyframe_window(frames, 190, max_shown=12)
        assert window[:2] == [0, None]
        assert window[-3:] == [170, 180, 190]

    def test_no_key_frame_is_listed_twice(self):
        frames = list(range(0, 200, 10))
        for current in (0, 30, 100, 190):
            shown = [f for f in keyframe_window(frames, current, max_shown=12)
                     if f is not None]
            assert len(shown) == len(set(shown))


class TestValidMask:
    @staticmethod
    def _half_masked(height, width):
        import numpy as np
        mask = np.full((height, width), 255, dtype=np.uint8)
        mask[:, width // 2:] = 0  # right half masked off (black)
        return mask

    def test_no_mask_treats_every_box_as_valid(self):
        assert box_in_valid_area((0, 0, 10, 10), None, 640, 512)

    def test_box_inside_white_area_is_valid(self):
        mask = self._half_masked(512, 640)
        assert box_in_valid_area((10, 10, 100, 100), mask, 640, 512)

    def test_box_reaching_into_black_area_is_invalid(self):
        mask = self._half_masked(512, 640)
        assert not box_in_valid_area((300, 10, 340, 50), mask, 640, 512)

    def test_box_fully_in_black_area_is_invalid(self):
        mask = self._half_masked(512, 640)
        assert not box_in_valid_area((400, 10, 500, 50), mask, 640, 512)

    def test_mask_with_other_resolution_is_scaled(self):
        # half-resolution mask for a 640x512 frame
        mask = self._half_masked(256, 320)
        assert box_in_valid_area((10, 10, 100, 100), mask, 640, 512)
        assert not box_in_valid_area((400, 10, 500, 50), mask, 640, 512)

    def test_load_valid_mask_missing_file_returns_none(self, tmp_path):
        assert load_valid_mask(str(tmp_path), "t") is None

    def test_load_valid_mask_reads_camera_specific_file(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        import numpy as np
        cv2.imwrite(str(tmp_path / "mask_W.png"),
                    np.full((8, 16), 255, dtype=np.uint8))
        loaded = load_valid_mask(str(tmp_path), "w")
        assert loaded is not None
        assert loaded.shape == (8, 16)
        # the thermal modality must not pick up the RGB mask
        assert load_valid_mask(str(tmp_path), "t") is None


class TestGeoPropagator:
    def _propagator(self, tmp_path, with_dem=True):
        dem = tmp_path / "dem.glb"
        if with_dem:
            dem.write_bytes(b"glb")
        return _GeoPropagator(str(tmp_path), str(dem), "", "t")

    def test_missing_dem_raises(self, fake_geo_stack, tmp_path):
        prop = self._propagator(tmp_path, with_dem=False)
        with pytest.raises(RuntimeError, match="DEM mesh file not found"):
            prop.load()

    def test_frame_range_validation(self, fake_geo_stack, tmp_path):
        prop = self._propagator(tmp_path)
        with pytest.raises(RuntimeError, match="poses file range"):
            prop.propagate((0, 0, 1, 1), 0, 9, _images(3), 640, 512)

    def test_propagate_projects_through_dem(self, fake_geo_stack, tmp_path):
        prop = self._propagator(tmp_path)
        result = prop.propagate((10, 10, 20, 20), 0, 1, _images(3), 640, 512)
        # WORLD_QUAD through the identity camera on a 640x512 frame
        assert result == pytest.approx((160.0, 192.0, 480.0, 320.0))
        assert prop.is_loaded

    def test_propagate_between_modalities_uses_target_resolution(
            self, fake_geo_stack, tmp_path):
        prop = self._propagator(tmp_path)
        result = prop.propagate_between(
            (10, 10, 20, 20), 0, _images(2), 1280, 1024,
            1, _images(4), 640, 512)
        assert result == pytest.approx((160.0, 192.0, 480.0, 320.0))

    def test_propagate_series_samples_intermediate_frames(
            self, fake_geo_stack, tmp_path):
        prop = self._propagator(tmp_path)
        boxes, failures = prop.propagate_series(
            (10, 10, 20, 20), 0, 25, _images(30), 640, 512, step=10)
        assert [f for f, _ in boxes] == [10, 20, 25]
        assert failures == []
        for _, box in boxes:
            assert box == pytest.approx((160.0, 192.0, 480.0, 320.0))

    def test_propagate_series_without_sampling_hits_target_only(
            self, fake_geo_stack, tmp_path):
        prop = self._propagator(tmp_path)
        boxes, _ = prop.propagate_series(
            (10, 10, 20, 20), 0, 25, _images(30), 640, 512, step=0)
        assert [f for f, _ in boxes] == [25]

    def test_propagate_series_reports_unprojectable_frames(
            self, fake_geo_stack, tmp_path, monkeypatch):
        prop = self._propagator(tmp_path)
        real = prop._world_to_box

        def fail_on_frame_10(world, dst_frame, *args, **kwargs):
            if dst_frame == 10:
                raise RuntimeError("outside frame 10")
            return real(world, dst_frame, *args, **kwargs)

        monkeypatch.setattr(prop, "_world_to_box", fail_on_frame_10)
        boxes, failures = prop.propagate_series(
            (10, 10, 20, 20), 0, 25, _images(30), 640, 512, step=10)
        assert [f for f, _ in boxes] == [20, 25]
        assert [f for f, _ in failures] == [10]

    def test_propagate_series_raises_when_source_misses_dem(
            self, fake_geo_stack, tmp_path, monkeypatch):
        import sys as _sys
        monkeypatch.setattr(
            _sys.modules["bambi.util.projection_util"],
            "label_to_world_coordinates",
            lambda *a, **k: [])
        prop = self._propagator(tmp_path)
        with pytest.raises(RuntimeError, match="no mesh intersection"):
            prop.propagate_series(
                (10, 10, 20, 20), 0, 25, _images(30), 640, 512, step=10)

    def test_projection_into_masked_off_area_fails(
            self, fake_geo_stack, tmp_path):
        cv2 = pytest.importorskip("cv2")
        import numpy as np
        # WORLD_QUAD projects to (160, 192, 480, 320) on 640x512 — mask off
        # exactly that region, so the propagation must be rejected.
        mask = np.full((512, 640), 255, dtype=np.uint8)
        mask[192:321, 160:481] = 0
        cv2.imwrite(str(tmp_path / "mask_T.png"), mask)
        prop = self._propagator(tmp_path)
        with pytest.raises(RuntimeError, match="valid .white. mask area"):
            prop.propagate((10, 10, 20, 20), 0, 1, _images(3), 640, 512)

    def test_projection_into_white_mask_area_passes(
            self, fake_geo_stack, tmp_path):
        cv2 = pytest.importorskip("cv2")
        import numpy as np
        cv2.imwrite(str(tmp_path / "mask_T.png"),
                    np.full((512, 640), 255, dtype=np.uint8))
        prop = self._propagator(tmp_path)
        result = prop.propagate((10, 10, 20, 20), 0, 1, _images(3), 640, 512)
        assert result == pytest.approx((160.0, 192.0, 480.0, 320.0))

    def test_correction_json_is_loaded(self, fake_geo_stack, tmp_path):
        (tmp_path / "correction.json").write_text(json.dumps({
            "translation": {"x": 1.0, "y": 2.0, "z": 3.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        }))
        prop = self._propagator(tmp_path)
        prop.load()
        assert prop._correction["translation"]["z"] == 3.0


# ---------------------------------------------------------------------------
# Merging & splitting tracks
# ---------------------------------------------------------------------------

def _track(track_id, frames, **kwargs):
    track = LabelTrack(track_id, **kwargs)
    for f in frames:
        track.set_keyframe(f, (f, f, f + 10, f + 10))
    return track


class TestMergeTracks:
    def test_keyframes_are_combined(self):
        merged, conflicts = merge_tracks(
            [_track(4, [20, 30]), _track(2, [0, 10])])
        assert merged.track_id == 2          # the lowest id survives
        assert merged.frames() == [0, 10, 20, 30]
        assert conflicts == 0

    def test_lower_track_id_wins_on_a_shared_frame(self):
        a, b = _track(1, [0]), _track(3, [0])
        b.set_keyframe(0, (99, 99, 100, 100))
        merged, conflicts = merge_tracks([b, a])
        assert conflicts == 1
        assert merged.keyframes[0]["x1"] == 0.0

    def test_gap_between_the_tracks_becomes_a_stop_frame(self):
        merged, _ = merge_tracks([_track(1, [0, 10]), _track(2, [50, 60])])
        assert merged.is_stop(10) is True     # gap 10 -> 50 is not bridged
        assert merged.box_at(30) is None
        assert merged.is_stop(60) is False    # nothing follows the last one

    def test_gaps_can_be_interpolated_instead(self):
        merged, _ = merge_tracks(
            [_track(1, [0, 10]), _track(2, [50, 60])], mark_gaps=False)
        assert merged.is_stop(10) is False
        assert merged.box_at(30) is not None

    def test_adjacent_frames_are_not_marked_as_a_gap(self):
        merged, _ = merge_tracks([_track(1, [0, 10]), _track(2, [11, 20])])
        assert merged.is_stop(10) is False

    def test_existing_stop_frames_survive(self):
        a = _track(1, [0, 10])
        a.set_keyframe(10, (0, 0, 1, 1), stop=True)
        merged, _ = merge_tracks([a, _track(2, [20, 30])], mark_gaps=False)
        assert merged.is_stop(10) is True

    def test_unknown_classes_are_filled_in_from_the_other_tracks(self):
        a = _track(1, [0], species="unknown", sex="unknown")
        b = _track(2, [10], species="red deer", sex="female", age="adult")
        merged, _ = merge_tracks([a, b])
        assert (merged.species, merged.sex, merged.age) == \
            ("red deer", "female", "adult")

    def test_known_classes_of_the_lowest_id_are_kept(self):
        a = _track(1, [0], species="roe deer")
        b = _track(2, [10], species="red deer")
        merged, _ = merge_tracks([a, b])
        assert merged.species == "roe deer"

    def test_custom_attributes_are_merged_lowest_id_first(self):
        a = _track(1, [0], attributes={"observer": "ann"})
        b = _track(2, [10], attributes={"observer": "bob", "note": "far"})
        merged, _ = merge_tracks([a, b])
        assert merged.attributes == {"observer": "ann", "note": "far"}

    def test_keyframe_attributes_are_copied_not_shared(self):
        a = _track(1, [0])
        a.set_keyframe(0, (0, 0, 1, 1), attributes={"note": "x"})
        merged, _ = merge_tracks([a, _track(2, [10])])
        merged.keyframes[0]["attributes"]["note"] = "changed"
        assert a.keyframes[0]["attributes"]["note"] == "x"

    def test_a_single_track_cannot_be_merged(self):
        with pytest.raises(ValueError, match="at least two"):
            merge_tracks([_track(1, [0])])


class TestSplitTrack:
    def test_split_on_a_keyframe_keeps_it_on_both_sides(self):
        head, tail = split_track(_track(1, [0, 10, 20]), 10, 7)
        assert head.track_id == 1 and head.frames() == [0, 10]
        assert tail.track_id == 7 and tail.frames() == [10, 20]
        assert head.keyframes[10] == tail.keyframes[10]

    def test_split_on_an_interpolated_frame_freezes_the_box(self):
        track = LabelTrack(1)
        track.set_keyframe(0, (0, 0, 10, 10))
        track.set_keyframe(10, (100, 50, 110, 60))
        head, tail = split_track(track, 5, 2)
        assert head.frames() == [0, 5] and tail.frames() == [5, 10]
        assert head.box_at(5)[0] == (50.0, 25.0, 60.0, 35.0)
        assert tail.box_at(5)[0] == (50.0, 25.0, 60.0, 35.0)

    def test_classes_and_attributes_are_carried_over(self):
        track = _track(1, [0, 10], species="fox", sex="male",
                       attributes={"note": "near the road"})
        head, tail = split_track(track, 5, 9)
        assert tail.species == "fox" and tail.sex == "male"
        assert tail.attributes == {"note": "near the road"}
        tail.attributes["note"] = "changed"
        assert head.attributes["note"] == "near the road"

    def test_the_two_parts_are_independent_of_the_original(self):
        track = _track(1, [0, 10, 20])
        head, _tail = split_track(track, 10, 2)
        head.keyframes[0]["x1"] = 999.0
        assert track.keyframes[0]["x1"] == 0.0

    def test_split_outside_the_range_is_rejected(self):
        with pytest.raises(ValueError, match="no bounding box"):
            split_track(_track(1, [10, 20]), 30, 2)

    def test_split_inside_a_gap_is_rejected(self):
        track = _track(1, [0, 30])
        track.set_keyframe(0, (0, 0, 1, 1), stop=True)
        with pytest.raises(ValueError, match="no bounding box"):
            split_track(track, 15, 2)

    def test_split_on_the_first_or_last_keyframe_is_rejected(self):
        track = _track(1, [0, 10, 20])
        for frame in (0, 20):
            with pytest.raises(ValueError, match="must lie between"):
                split_track(track, frame, 2)


class _StubPropagator:
    """Returns a world quad centred on a per-frame position."""

    def __init__(self, centres):
        self.centres = centres  # frame -> (x, y) or None to fail

    def _box_to_world(self, box, frame, images, width, height):
        centre = self.centres.get(frame)
        if centre is None:
            raise RuntimeError("no mesh intersection")
        x, y = centre
        return [(x - 1, y - 1, 0.0), (x + 1, y - 1, 0.0),
                (x + 1, y + 1, 0.0), (x - 1, y + 1, 0.0)]


class TestTrackWorldPositions:
    def test_box_centres_are_projected_per_keyframe(self):
        prop = _StubPropagator({0: (10.0, 20.0), 10: (12.0, 20.0)})
        positions = track_world_positions(
            prop, _track(1, [0, 10]), _images(20), 640, 512)
        assert positions == {0: (10.0, 20.0), 10: (12.0, 20.0)}

    def test_unprojectable_keyframes_are_skipped(self):
        prop = _StubPropagator({0: (1.0, 1.0), 10: None})
        positions = track_world_positions(
            prop, _track(1, [0, 10]), _images(20), 640, 512)
        assert list(positions) == [0]

    def test_real_propagator_centres_the_world_quad(
            self, fake_geo_stack, tmp_path):
        dem = tmp_path / "dem.glb"
        dem.write_bytes(b"glb")
        prop = _GeoPropagator(str(tmp_path), str(dem), "", "t")
        positions = track_world_positions(
            prop, _track(1, [0, 1]), _images(3), 640, 512)
        # WORLD_QUAD is centred on the origin
        assert positions == {0: (0.0, 0.0), 1: (0.0, 0.0)}


class TestFindOverlappingTracks:
    def test_close_tracks_are_proposed_with_their_distance(self):
        pairs = find_overlapping_tracks(
            {1: {0: (0.0, 0.0)}, 2: {50: (2.0, 0.0)}}, max_distance=3.0)
        assert pairs == [(1, 2, 2.0, 50)]  # 2 m apart, 50 frames apart

    def test_distant_tracks_are_not_proposed(self):
        pairs = find_overlapping_tracks(
            {1: {0: (0.0, 0.0)}, 2: {50: (20.0, 0.0)}}, max_distance=3.0)
        assert pairs == []

    def test_the_closest_pair_of_positions_counts(self):
        pairs = find_overlapping_tracks(
            {1: {0: (0.0, 0.0), 10: (100.0, 0.0)},
             2: {20: (101.0, 0.0), 30: (200.0, 0.0)}}, max_distance=2.0)
        assert pairs[0][:3] == (1, 2, 1.0)

    def test_overlapping_frame_ranges_report_a_zero_gap(self):
        pairs = find_overlapping_tracks(
            {1: {0: (0.0, 0.0), 100: (0.0, 0.0)}, 2: {50: (0.5, 0.0)}},
            max_distance=1.0)
        assert pairs[0][3] == 0

    def test_a_large_frame_gap_can_be_excluded(self):
        positions = {1: {0: (0.0, 0.0)}, 2: {500: (0.5, 0.0)}}
        assert find_overlapping_tracks(positions, 1.0, max_frame_gap=100) == []
        assert find_overlapping_tracks(positions, 1.0, max_frame_gap=0)
        assert find_overlapping_tracks(positions, 1.0, max_frame_gap=600)

    def test_results_are_sorted_by_distance(self):
        pairs = find_overlapping_tracks(
            {1: {0: (0.0, 0.0)}, 2: {10: (2.0, 0.0)}, 3: {20: (0.5, 0.0)}},
            max_distance=5.0)
        assert [p[2] for p in pairs] == sorted(p[2] for p in pairs)
        assert pairs[0][:2] == (1, 3)

    def test_tracks_without_positions_are_ignored(self):
        pairs = find_overlapping_tracks(
            {1: {0: (0.0, 0.0)}, 2: {}}, max_distance=100.0)
        assert pairs == []


class TestGroupTrackIds:
    def test_chained_pairs_become_one_group(self):
        assert group_track_ids([(1, 2), (2, 5)]) == [[1, 2, 5]]

    def test_independent_pairs_stay_separate(self):
        assert group_track_ids([(1, 2), (4, 7)]) == [[1, 2], [4, 7]]

    def test_no_pairs_yield_no_groups(self):
        assert group_track_ids([]) == []
