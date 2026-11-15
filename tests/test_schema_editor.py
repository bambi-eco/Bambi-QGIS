# -*- coding: utf-8 -*-
"""Tests for the project vocabulary editor (EXCHANGE_FORMAT_PLAN.md §3.1, §5.3).

The rules under test all exist for one reason: after an edit, stored data must
still mean what it meant before. Species and enum values are referenced by id
so renames are free; fields are referenced by name so renames rewrite rows; and
nothing that is still referenced may be deleted.
"""
import json
import os

import pytest

from bambi_wildlife_detection.core import store
from bambi_wildlife_detection.core.schema_editor import (
    SchemaEditor, SchemaError)


@pytest.fixture
def editor(tmp_path):
    with SchemaEditor(str(tmp_path)) as ed:
        yield ed


@pytest.fixture
def editor_with_data(tmp_path):
    """A project with one detection, one track and one label track."""
    root = str(tmp_path)
    with SchemaEditor(root) as ed:
        wolf = ed.add_species("wolf")
        ed.add_field("collar_id", "string", "track")
        ed.commit()

    det = store.open_store(
        store.stage_path(root, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    det.execute(
        "INSERT INTO detections (frame, source_id, species_id, source_class, "
        "attributes) VALUES (1, 1, ?, '7', ?)",
        (wolf, json.dumps({"occlusion": 1})))
    det.commit()
    det.close()

    labels = store.open_store(
        store.stage_path(root, store.LABELS, "t"), store.LABELS, "t")
    labels.execute(
        "INSERT INTO label_tracks (label_track_id, species_id, attributes) "
        "VALUES (1, ?, ?)", (wolf, json.dumps({"collar_id": "R-114",
                                               "sex": 1})))
    labels.commit()
    labels.close()

    with SchemaEditor(root) as ed:
        yield ed


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------

def test_new_project_has_the_base_classes(editor):
    names = {s["name"]: s["species_id"] for s in editor.species()}
    assert names["animal"] == 0
    assert names["unknown"] == -1
    assert names["not-an-animal"] == -2


def test_add_species_appends_a_new_id(editor):
    assert editor.add_species("wolf") == 10
    assert editor.add_species("lynx") == 11


def test_add_species_rejects_a_blank_name(editor):
    with pytest.raises(SchemaError, match="needs a name"):
        editor.add_species("   ")


def test_add_species_rejects_a_duplicate(editor):
    editor.add_species("wolf")
    with pytest.raises(SchemaError, match="already exists"):
        editor.add_species("wolf")


def test_add_species_strips_whitespace(editor):
    editor.add_species("  wolf  ")
    assert editor.species_by_name("wolf") is not None


def test_deleted_species_ids_are_not_reissued(editor):
    first = editor.add_species("wolf")
    editor.delete_species(first)
    assert editor.add_species("lynx") > first


def test_rename_species_keeps_the_id(editor_with_data):
    wolf = editor_with_data.species_by_name("wolf")["species_id"]
    editor_with_data.rename_species(wolf, "grey wolf")
    assert editor_with_data.species_by_name("grey wolf")["species_id"] == wolf


def test_rename_species_leaves_stored_rows_untouched(editor_with_data, tmp_path):
    wolf = editor_with_data.species_by_name("wolf")["species_id"]
    editor_with_data.rename_species(wolf, "grey wolf")
    editor_with_data.commit()

    conn = store.open_store(
        store.stage_path(str(tmp_path), store.DETECTIONS, "t"),
        store.DETECTIONS, "t")
    assert conn.execute(
        "SELECT species_id FROM detections").fetchone()["species_id"] == wolf
    conn.close()


@pytest.mark.parametrize("name", ["animal", "unknown", "not-an-animal"])
def test_base_species_cannot_be_renamed(editor, name):
    species_id = editor.species_by_name(name)["species_id"]
    with pytest.raises(SchemaError, match="cannot be renamed"):
        editor.rename_species(species_id, "something else")


@pytest.mark.parametrize("name", ["animal", "unknown", "not-an-animal"])
def test_base_species_cannot_be_deleted(editor, name):
    species_id = editor.species_by_name(name)["species_id"]
    with pytest.raises(SchemaError, match="cannot be deleted"):
        editor.delete_species(species_id)


def test_rename_species_rejects_a_clash(editor):
    editor.add_species("wolf")
    lynx = editor.add_species("lynx")
    with pytest.raises(SchemaError, match="already exists"):
        editor.rename_species(lynx, "wolf")


def test_rename_species_to_its_own_name_is_allowed(editor):
    wolf = editor.add_species("wolf")
    editor.rename_species(wolf, "wolf")


def test_unused_species_can_be_deleted(editor):
    wolf = editor.add_species("wolf")
    editor.delete_species(wolf)
    assert editor.species_by_name("wolf") is None


def test_species_in_use_cannot_be_deleted(editor_with_data):
    wolf = editor_with_data.species_by_name("wolf")["species_id"]
    with pytest.raises(SchemaError, match="still in use"):
        editor_with_data.delete_species(wolf)


def test_species_usage_reports_where(editor_with_data):
    wolf = editor_with_data.species_by_name("wolf")["species_id"]
    usage = editor_with_data.species_usage(wolf)
    assert usage["detections"] == 1
    assert usage["label_tracks"] == 1


def test_unknown_species_id_is_rejected(editor):
    with pytest.raises(SchemaError, match="No species"):
        editor.rename_species(4242, "x")
    with pytest.raises(SchemaError, match="No species"):
        editor.delete_species(4242)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

def test_seeded_enums_are_present(editor):
    assert {e["name"] for e in editor.enums()} == {"sex", "age", "occlusion"}


def test_enum_values_come_back_in_display_order(editor):
    sex = editor.enum_by_name("sex")["enum_id"]
    assert [v["label"] for v in editor.enum_values(sex)] == \
        ["unknown", "female", "male"]


def test_add_enum_value_appends(editor):
    age = editor.enum_by_name("age")["enum_id"]
    assert editor.add_enum_value(age, "subadult") == 3
    assert [v["label"] for v in editor.enum_values(age)][-1] == "subadult"


def test_add_enum_value_rejects_a_duplicate(editor):
    sex = editor.enum_by_name("sex")["enum_id"]
    with pytest.raises(SchemaError, match="already a value"):
        editor.add_enum_value(sex, "female")


def test_add_enum_value_rejects_a_blank_label(editor):
    sex = editor.enum_by_name("sex")["enum_id"]
    with pytest.raises(SchemaError, match="needs a label"):
        editor.add_enum_value(sex, "  ")


def test_deleted_enum_value_ids_are_not_reissued(editor):
    age = editor.enum_by_name("age")["enum_id"]
    first = editor.add_enum_value(age, "subadult")
    editor.delete_enum_value(age, first)
    assert editor.add_enum_value(age, "yearling") > first


def test_rename_enum_value_keeps_its_id(editor):
    sex = editor.enum_by_name("sex")["enum_id"]
    editor.rename_enum_value(sex, 1, "Female")
    values = {v["value_id"]: v["label"] for v in editor.enum_values(sex)}
    assert values[1] == "Female"


def test_rename_enum_value_leaves_stored_rows_untouched(editor_with_data, tmp_path):
    """The point of storing value_id rather than the label (§5.1)."""
    sex = editor_with_data.enum_by_name("sex")["enum_id"]
    editor_with_data.rename_enum_value(sex, 1, "Female")
    editor_with_data.commit()

    conn = store.open_store(
        store.stage_path(str(tmp_path), store.LABELS, "t"), store.LABELS, "t")
    attributes = json.loads(conn.execute(
        "SELECT attributes FROM label_tracks").fetchone()["attributes"])
    assert attributes["sex"] == 1
    conn.close()


def test_rename_enum_value_rejects_a_clash(editor):
    sex = editor.enum_by_name("sex")["enum_id"]
    with pytest.raises(SchemaError, match="already a value"):
        editor.rename_enum_value(sex, 1, "male")


def test_reorder_changes_display_order_only(editor):
    sex = editor.enum_by_name("sex")["enum_id"]
    editor.reorder_enum_values(sex, [2, 1, 0])
    values = editor.enum_values(sex)
    assert [v["label"] for v in values] == ["male", "female", "unknown"]
    assert [v["value_id"] for v in values] == [2, 1, 0]


def test_reorder_must_list_every_value(editor):
    sex = editor.enum_by_name("sex")["enum_id"]
    with pytest.raises(SchemaError, match="exactly the enum's current values"):
        editor.reorder_enum_values(sex, [0, 1])


def test_enum_value_in_use_cannot_be_deleted(editor_with_data):
    sex = editor_with_data.enum_by_name("sex")["enum_id"]
    with pytest.raises(SchemaError, match="still in use"):
        editor_with_data.delete_enum_value(sex, 1)


def test_unused_enum_value_can_be_deleted(editor_with_data):
    sex = editor_with_data.enum_by_name("sex")["enum_id"]
    editor_with_data.delete_enum_value(sex, 2)   # 'male', unused
    assert 2 not in {v["value_id"] for v in editor_with_data.enum_values(sex)}


def test_enum_value_usage_finds_keyframe_scope_values(editor_with_data):
    occlusion = editor_with_data.enum_by_name("occlusion")["enum_id"]
    assert editor_with_data.enum_value_usage(occlusion, 1) == {"detections": 1}


def test_new_enum_can_be_added(editor):
    behaviour = editor.add_enum("behaviour")
    editor.add_enum_value(behaviour, "grazing")
    assert [v["label"] for v in editor.enum_values(behaviour)] == ["grazing"]


def test_add_enum_rejects_a_duplicate(editor):
    with pytest.raises(SchemaError, match="already exists"):
        editor.add_enum("sex")


# ---------------------------------------------------------------------------
# Custom fields
# ---------------------------------------------------------------------------

def test_seeded_fields_are_protected(editor):
    fields = {f["name"]: f for f in editor.fields()}
    assert fields["sex"]["protected"] == 1
    assert fields["occlusion"]["scope"] == "detection"


def test_add_string_field(editor):
    """Free text stays available - it is just not in the default setup."""
    editor.add_field("collar_id", "string", "track")
    assert any(f["name"] == "collar_id" for f in editor.fields("track"))


def test_add_enum_field_requires_an_enum(editor):
    with pytest.raises(SchemaError, match="needs an enum"):
        editor.add_field("mood", "enum", "track")


def test_add_enum_field_with_an_enum(editor):
    behaviour = editor.add_enum("behaviour")
    editor.add_field("behaviour", "enum", "detection", behaviour)
    field = [f for f in editor.fields("detection") if f["name"] == "behaviour"][0]
    assert field["enum_id"] == behaviour


def test_non_enum_field_drops_a_stray_enum_id(editor):
    sex = editor.enum_by_name("sex")["enum_id"]
    editor.add_field("note", "string", "track", sex)
    field = [f for f in editor.fields("track") if f["name"] == "note"][0]
    assert field["enum_id"] is None


@pytest.mark.parametrize("name", ["frame", "x1", "species_id", "attributes"])
def test_reserved_field_names_are_rejected(editor, name):
    with pytest.raises(SchemaError, match="reserved"):
        editor.add_field(name, "string", "track")


def test_unknown_type_and_scope_are_rejected(editor):
    with pytest.raises(SchemaError, match="field type"):
        editor.add_field("x", "quaternion", "track")
    with pytest.raises(SchemaError, match="field scope"):
        editor.add_field("x", "string", "galaxy")


def test_same_name_in_two_scopes_is_allowed(editor):
    editor.add_field("note", "string", "track")
    editor.add_field("note", "string", "detection")
    assert len([f for f in editor.fields() if f["name"] == "note"]) == 2


def test_duplicate_field_in_one_scope_is_rejected(editor):
    editor.add_field("note", "string", "track")
    with pytest.raises(SchemaError, match="already exists"):
        editor.add_field("note", "string", "track")


def test_field_rename_rewrites_stored_attributes(editor_with_data, tmp_path):
    """Fields are keyed by name, so a rename must migrate the rows (§5.1)."""
    rewritten = editor_with_data.rename_field("collar_id", "track", "collar")
    editor_with_data.commit()
    assert rewritten == 1

    conn = store.open_store(
        store.stage_path(str(tmp_path), store.LABELS, "t"), store.LABELS, "t")
    attributes = json.loads(conn.execute(
        "SELECT attributes FROM label_tracks").fetchone()["attributes"])
    assert attributes["collar"] == "R-114"
    assert "collar_id" not in attributes
    assert attributes["sex"] == 1          # untouched
    conn.close()


def test_field_rename_updates_the_schema(editor_with_data):
    editor_with_data.rename_field("collar_id", "track", "collar")
    names = {f["name"] for f in editor_with_data.fields("track")}
    assert "collar" in names and "collar_id" not in names


def test_protected_field_cannot_be_renamed_or_deleted(editor):
    with pytest.raises(SchemaError, match="built-in"):
        editor.rename_field("sex", "track", "gender")
    with pytest.raises(SchemaError, match="built-in"):
        editor.delete_field("sex", "track")


def test_field_rename_rejects_a_clash(editor):
    editor.add_field("a", "string", "track")
    editor.add_field("b", "string", "track")
    with pytest.raises(SchemaError, match="already"):
        editor.rename_field("a", "track", "b")


def test_field_with_values_needs_force_to_delete(editor_with_data):
    with pytest.raises(SchemaError, match="still holds values"):
        editor_with_data.delete_field("collar_id", "track")


def test_forced_field_delete_strips_the_values(editor_with_data, tmp_path):
    rewritten = editor_with_data.delete_field("collar_id", "track", force=True)
    editor_with_data.commit()
    assert rewritten == 1

    conn = store.open_store(
        store.stage_path(str(tmp_path), store.LABELS, "t"), store.LABELS, "t")
    attributes = json.loads(conn.execute(
        "SELECT attributes FROM label_tracks").fetchone()["attributes"])
    assert "collar_id" not in attributes
    assert attributes["sex"] == 1
    conn.close()


def test_unused_field_deletes_cleanly(editor):
    editor.add_field("note", "string", "track")
    assert editor.delete_field("note", "track") == 0


def test_field_usage_reports_where(editor_with_data):
    assert editor_with_data.field_usage("collar_id") == {"label_tracks": 1}


# ---------------------------------------------------------------------------
# Nothing is changed, and nothing is locked, until the edits are saved
# ---------------------------------------------------------------------------

def _detection_source(root):
    project = store.open_store(store.project_path(root), store.PROJECT)
    project.execute(
        "INSERT INTO detection_sources (source_id, kind, created_at) "
        "VALUES (1, 'detector', '')")
    project.commit()
    project.close()


def _attributes(root, kind, table):
    conn = store.open_store(store.stage_path(root, kind, "t"), kind, "t")
    try:
        return [json.loads(row["attributes"]) for row in conn.execute(
            f"SELECT attributes FROM {table}")]
    finally:
        conn.close()


def _species_names(root):
    project = store.open_store(store.project_path(root), store.PROJECT)
    try:
        return {row["name"] for row in project.execute(
            "SELECT name FROM species")}
    finally:
        project.close()


def test_an_open_editor_does_not_lock_the_project(editor, tmp_path):
    """The editor kept its edits in an open transaction until the dialog was
    closed. A step that finished meanwhile could not record its results:
    "database is locked"."""
    import sqlite3

    editor.add_species("wolf")                       # an edit is pending

    other = sqlite3.connect(store.project_path(str(tmp_path)), timeout=0.2)
    try:
        other.execute("INSERT OR REPLACE INTO bambi_meta (key, value) "
                      "VALUES ('a step finished', 'yes')")
        other.commit()
    finally:
        other.close()
    editor.commit()                                  # ... and is still saved

    project = store.open_store(store.project_path(str(tmp_path)),
                               store.PROJECT)
    assert store.get_meta(project, "a step finished") == "yes"
    project.close()
    assert "wolf" in _species_names(str(tmp_path))


def test_closing_without_saving_changes_nothing(editor_with_data, tmp_path):
    """Cancelling rolled the field's name back - but its stored rows had
    been renamed already, and those of a field deleted with its values were
    gone."""
    root = str(tmp_path)
    editor_with_data.add_species("lynx")
    editor_with_data.rename_field("collar_id", "track", "collar")
    editor_with_data.delete_field("collar", "track", force=True)
    editor_with_data.close()

    assert "lynx" not in _species_names(root)
    assert _attributes(root, store.LABELS, "label_tracks") == [
        {"collar_id": "R-114", "sex": 1}]
    with SchemaEditor(root) as again:
        assert [f["name"] for f in again.fields("track")
                if not f["protected"]] == ["collar_id"]


def test_a_rename_reaches_the_stored_rows_when_it_is_saved(editor_with_data,
                                                           tmp_path):
    root = str(tmp_path)
    assert editor_with_data.rename_field("collar_id", "track", "collar") == 1
    assert _attributes(root, store.LABELS, "label_tracks")[0].get(
        "collar_id") == "R-114"                       # not yet
    editor_with_data.commit()
    assert _attributes(root, store.LABELS, "label_tracks") == [
        {"collar": "R-114", "sex": 1}]


def test_a_field_renamed_twice_is_rewritten_once(editor_with_data, tmp_path):
    editor_with_data.rename_field("collar_id", "track", "collar")
    assert editor_with_data.rename_field("collar", "track", "tag") == 1
    assert editor_with_data.field_usage("tag", "track") == {"label_tracks": 1}
    editor_with_data.commit()
    assert _attributes(str(tmp_path), store.LABELS, "label_tracks") == [
        {"tag": "R-114", "sex": 1}]


def test_a_deleted_fields_values_go_when_it_is_saved(editor_with_data,
                                                     tmp_path):
    root = str(tmp_path)
    assert editor_with_data.delete_field("collar_id", "track", force=True) == 1
    assert "collar_id" in _attributes(root, store.LABELS, "label_tracks")[0]
    editor_with_data.commit()
    assert _attributes(root, store.LABELS, "label_tracks") == [{"sex": 1}]


def test_a_field_of_another_scope_keeps_its_values(editor_with_data,
                                                   tmp_path):
    """A field is its name *and* its scope. Renaming the track field also
    renamed the values of the detection field of the same name."""
    root = str(tmp_path)
    det = store.open_store(
        store.stage_path(root, store.DETECTIONS, "t"), store.DETECTIONS, "t")
    det.execute("UPDATE detections SET attributes = ?",
                (json.dumps({"collar_id": "seen", "occlusion": 1}),))
    det.commit()
    det.close()
    editor_with_data.add_field("collar_id", "string", "detection")

    assert editor_with_data.field_usage("collar_id", "track") == {
        "label_tracks": 1}
    assert editor_with_data.field_usage("collar_id", "detection") == {
        "detections": 1}
    editor_with_data.rename_field("collar_id", "track", "collar")
    editor_with_data.commit()

    assert _attributes(root, store.LABELS, "label_tracks") == [
        {"collar": "R-114", "sex": 1}]
    assert _attributes(root, store.DETECTIONS, "detections") == [
        {"collar_id": "seen", "occlusion": 1}]


def test_two_fields_can_swap_names(tmp_path):
    root = str(tmp_path)
    with SchemaEditor(root) as editor:
        editor.add_field("a", "string", "track")
        editor.add_field("b", "string", "track")
        editor.commit()
    labels = store.open_store(
        store.stage_path(root, store.LABELS, "t"), store.LABELS, "t")
    labels.execute(
        "INSERT INTO label_tracks (label_track_id, species_id, attributes) "
        "VALUES (1, 0, ?)", (json.dumps({"a": "was a", "b": "was b"}),))
    labels.commit()
    labels.close()

    with SchemaEditor(root) as editor:
        editor.rename_field("a", "track", "tmp")
        editor.rename_field("b", "track", "a")
        editor.rename_field("tmp", "track", "b")
        editor.commit()

    assert _attributes(root, store.LABELS, "label_tracks") == [
        {"a": "was b", "b": "was a"}]


def test_a_change_made_elsewhere_is_not_overwritten(editor, tmp_path):
    root = str(tmp_path)
    editor.add_species("wolf")
    with SchemaEditor(root) as other:
        other.add_species("lynx")
        other.commit()

    with pytest.raises(SchemaError, match="changed by something else"):
        editor.commit()
    assert _species_names(root) >= {"lynx"}
    assert "wolf" not in _species_names(root)


def test_a_busy_store_refuses_the_save_and_takes_the_renames_back(
        editor_with_data, tmp_path, monkeypatch):
    import sqlite3

    def busy(self, real):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(SchemaEditor, "_write_back", busy)
    editor_with_data.rename_field("collar_id", "track", "collar")
    with pytest.raises(SchemaError, match="in use"):
        editor_with_data.commit()
    assert _attributes(str(tmp_path), store.LABELS, "label_tracks") == [
        {"collar_id": "R-114", "sex": 1}]


def test_saving_twice_is_saving_once(editor, tmp_path):
    editor.add_species("wolf")
    editor.commit()
    editor.add_species("lynx")
    editor.commit()
    assert _species_names(str(tmp_path)) >= {"wolf", "lynx"}


def test_the_working_copy_is_removed(tmp_path):
    with SchemaEditor(str(tmp_path)) as editor:
        folder = editor._folder
        assert os.path.isdir(folder)
    assert not os.path.exists(folder)


def test_the_base_classes_stay_guarded_through_a_save(editor, tmp_path):
    """The store's own triggers apply to what was edited - and a save that
    edits nothing about them touches none of them."""
    editor.set_species_colour(0, "#123456")
    editor.add_species("wolf")
    editor.commit()
    assert {"animal", "unknown", "not-an-animal"} <= _species_names(
        str(tmp_path))


# ---------------------------------------------------------------------------
# Detector class mapping
# ---------------------------------------------------------------------------

def test_set_and_read_a_class_mapping(editor):
    editor.conn.execute(
        "INSERT INTO detection_sources (kind, created_at) VALUES ('detector', '')")
    wolf = editor.add_species("wolf")
    editor.set_class_mapping(1, "7", wolf)
    assert editor.class_mapping(1) == [{"source_class": "7", "species_id": wolf}]


def test_mapping_to_a_missing_species_is_rejected(editor):
    editor.conn.execute(
        "INSERT INTO detection_sources (kind, created_at) VALUES ('detector', '')")
    with pytest.raises(SchemaError, match="No species"):
        editor.set_class_mapping(1, "7", 4242)


def test_reapply_mapping_updates_existing_detections(editor_with_data, tmp_path):
    """A corrected mapping is applied to stored rows, not re-detected (§3.1)."""
    # The detector is on record before its classes are mapped, as it is
    # once a detection has run.
    _detection_source(str(tmp_path))
    with SchemaEditor(str(tmp_path)) as editor:
        lynx = editor.add_species("lynx")
        editor.set_class_mapping(1, "7", lynx)
        editor.commit()
        assert editor.reapply_class_mapping(1) == 1

    conn = store.open_store(
        store.stage_path(str(tmp_path), store.DETECTIONS, "t"),
        store.DETECTIONS, "t")
    assert conn.execute(
        "SELECT species_id FROM detections").fetchone()["species_id"] == lynx
    conn.close()


def test_reapply_mapping_falls_back_to_animal_when_unmapped(
        editor_with_data, tmp_path):
    editor_with_data.conn.execute(
        "INSERT INTO detection_sources (source_id, kind, created_at) "
        "VALUES (1, 'agnostic', '')")
    editor_with_data.commit()

    assert editor_with_data.reapply_class_mapping(1) == 1
    conn = store.open_store(
        store.stage_path(str(tmp_path), store.DETECTIONS, "t"),
        store.DETECTIONS, "t")
    assert conn.execute(
        "SELECT species_id FROM detections").fetchone()["species_id"] == 0
    conn.close()


def test_editor_works_on_a_project_without_stage_files(tmp_path):
    """Vocabulary editing must not require any stage to have run yet."""
    with SchemaEditor(str(tmp_path)) as ed:
        wolf = ed.add_species("wolf")
        assert ed.species_usage(wolf) == {}
        ed.delete_species(wolf)


# ---------------------------------------------------------------------------
# Publishing taxonomy (§8.1)
# ---------------------------------------------------------------------------

def test_seeded_species_have_editable_scientific_names(editor):
    roe = editor.species_by_name("roe deer")
    assert roe["scientific_name"] == "Capreolus capreolus"
    assert roe["taxon_rank"] == "species"
    assert roe["gbif_taxon_key"] is None


def test_set_taxonomy_round_trips(editor):
    wolf = editor.add_species("wolf")
    editor.set_taxonomy(wolf, "Canis lupus", "species", 5219173)
    stored = editor.species_by_name("wolf")
    assert stored["scientific_name"] == "Canis lupus"
    assert stored["gbif_taxon_key"] == 5219173


def test_taxonomy_can_be_cleared(editor):
    roe = editor.species_by_name("roe deer")["species_id"]
    editor.set_taxonomy(roe, "")
    assert editor.species_by_name("roe deer")["scientific_name"] is None


@pytest.mark.parametrize("name", ["animal", "unknown", "not-an-animal"])
def test_base_classes_cannot_carry_a_taxon(editor, name):
    """'animal' is deliberately not a taxon; publishing it as one is nonsense."""
    species_id = editor.species_by_name(name)["species_id"]
    with pytest.raises(SchemaError, match="not taxa"):
        editor.set_taxonomy(species_id, "Animalia")


def test_a_non_numeric_gbif_key_is_rejected(editor):
    wolf = editor.add_species("wolf")
    with pytest.raises(SchemaError, match="gbif.org"):
        editor.set_taxonomy(wolf, "Canis lupus", "species", "not-a-key")


def test_a_negative_gbif_key_is_rejected(editor):
    wolf = editor.add_species("wolf")
    with pytest.raises(SchemaError, match="positive"):
        editor.set_taxonomy(wolf, "Canis lupus", "species", -1)


def test_renaming_a_species_keeps_its_taxonomy(editor):
    wolf = editor.add_species("wolf")
    editor.set_taxonomy(wolf, "Canis lupus", "species", 5219173)
    editor.rename_species(wolf, "grey wolf")
    assert editor.species_by_name("grey wolf")["gbif_taxon_key"] == 5219173


# ---------------------------------------------------------------------------
# What the detector actually emitted (§3.1)
# ---------------------------------------------------------------------------

def test_observed_classes_come_from_the_detections(editor_with_data):
    """A table built only from class_mapping shows nothing until someone guesses."""
    observed = editor_with_data.observed_source_classes(1)
    assert [row["source_class"] for row in observed] == ["7"]
    assert observed[0]["detections"] == 1
    assert observed[0]["mapped"] is False
    assert observed[0]["species_id"] == store.FALLBACK_SPECIES_ID


def test_observed_classes_include_mapped_ones_never_seen(editor_with_data):
    editor_with_data.conn.execute(
        "INSERT INTO detection_sources (source_id, kind, created_at) "
        "VALUES (1, 'detector', '')")
    wolf = editor_with_data.species_by_name("wolf")["species_id"]
    editor_with_data.set_class_mapping(1, "9", wolf)
    observed = {row["source_class"]: row for row in
                editor_with_data.observed_source_classes(1)}
    assert observed["9"]["mapped"] is True
    assert observed["9"]["detections"] == 0


def test_observed_classes_report_the_mapped_species(editor_with_data):
    editor_with_data.conn.execute(
        "INSERT INTO detection_sources (source_id, kind, created_at) "
        "VALUES (1, 'detector', '')")
    wolf = editor_with_data.species_by_name("wolf")["species_id"]
    editor_with_data.set_class_mapping(1, "7", wolf)
    observed = editor_with_data.observed_source_classes(1)
    assert observed[0]["species_id"] == wolf
    assert observed[0]["mapped"] is True


def test_observed_classes_sort_numerically(editor):
    editor.conn.execute(
        "INSERT INTO detection_sources (source_id, kind, created_at) "
        "VALUES (1, 'detector', '')")
    for name in ("10", "2", "1", "bird"):
        editor.set_class_mapping(1, name, 0)
    names = [row["source_class"] for row in editor.observed_source_classes(1)]
    assert names == ["1", "2", "10", "bird"]


def test_observed_classes_of_a_project_without_detections(editor):
    assert editor.observed_source_classes(1) == []
