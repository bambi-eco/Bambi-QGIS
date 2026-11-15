# -*- coding: utf-8 -*-
"""Where the labelling tool reads its label tracks from.

The label tracks exist twice: in the project's store (``labels.gpkg``), which
is what the rest of the plugin reads and changes, and in ``labels.json``, the
tool's own copy. The tool used to read the file alone, so

* a labelled track deleted in the inspector came back with the next save;
* a species renamed in the Project Schema fell back to "animal";
* a missing or unreadable ``labels.json`` made the next save empty the store.

The store now wins while the file is still the copy written beside it; the
file wins when it is the newer of the two; and whichever side is not loaded
although it differs is kept as a file.
"""
import json
import os

import pytest

from bambi_wildlife_detection.core import label_store, labelling, store
from bambi_wildlife_detection.core.labelling import (
    FROM_JSON, FROM_STORE, CustomField, LabelStore, LabelTrack)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(root):
    store.open_store(store.project_path(root), store.PROJECT).close()
    return root


def _tracks():
    doe = LabelTrack(1, species="roe deer", sex="female", age="adult",
                     attributes={"collar": True, "note": "left ear tag"},
                     origin_track_id=17)
    doe.set_keyframe(1, (10.0, 20.0, 30.0, 40.0), occlusion="occluded",
                     attributes={"posture": "lying"})
    doe.set_keyframe(4, (30.5, 40.25, 50.0, 60.0), occlusion="clear",
                     stop=True)
    doe.set_keyframe(9, (31.0, 41.0, 51.0, 61.0), occlusion="clear")
    doe.keyframes[1]["origin_detection_id"] = 5
    stag = LabelTrack(2, species="red deer", sex="male", age="unknown")
    stag.set_keyframe(2, (100.0, 100.0, 140.0, 150.0))
    return {1: doe, 2: stag}


FIELDS = [CustomField("collar", "bool", "track"),
          CustomField("note", "string", "track"),
          CustomField("posture", "string", "keyframe")]


def _saved(root, tracks=None):
    ls = LabelStore(root, "t")
    ls.tracks = _tracks() if tracks is None else tracks
    ls.set_custom_fields(list(FIELDS))
    ls.save()
    return ls


def _loaded(root):
    ls = LabelStore(root, "t")
    ls.load()
    return ls


def _documents(ls):
    return [ls.tracks[i].to_dict() for i in sorted(ls.tracks)]


def _stored_ids(root):
    return [t["label_track_id"] for t in label_store.load_tracks(root, "t")]


def _forget_fingerprint(root):
    """A store as versions before 9.1 left it: no record of the file."""
    conn = store.open_store(
        store.stage_path(root, store.LABELS, "t"), store.LABELS, "t")
    try:
        with store.transaction(conn):
            conn.execute("DELETE FROM bambi_meta WHERE key = ?",
                         (label_store.JSON_FINGERPRINT_KEY,))
    finally:
        conn.close()


def _delete_in_store(root, label_track_id):
    """What another tool does: change the store, not the file."""
    kept = [t for t in label_store.load_tracks(root, "t")
            if t["label_track_id"] != label_track_id]
    label_store.save_tracks(root, "t", kept)


def _rewrite_json(ls, change):
    with open(ls.json_path, encoding="utf-8") as fh:
        document = json.load(fh)
    change(document)
    with open(ls.json_path, "w", encoding="utf-8") as fh:
        json.dump(document, fh)


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

class TestChoice:
    def choose(self, **changes):
        state = dict(json_readable=True, json_fingerprint="abc",
                     json_mtime=100.0, store_present=True,
                     recorded_fingerprint="abc", store_mtime=100.5,
                     stored_tracks=3)
        state.update(changes)
        return labelling.choose_label_source(**state)

    def test_the_store_wins_while_the_file_is_its_copy(self):
        assert self.choose() == FROM_STORE
        assert self.choose(stored_tracks=0) == FROM_STORE     # all deleted

    def test_a_file_changed_since_wins(self):
        assert self.choose(json_fingerprint="other") == FROM_JSON

    def test_without_a_store_there_is_only_the_file(self):
        assert self.choose(store_present=False) == FROM_JSON
        assert self.choose(store_present=False,
                           json_readable=False) == FROM_JSON

    def test_without_a_readable_file_there_is_only_the_store(self):
        assert self.choose(json_readable=False,
                           json_fingerprint=None) == FROM_STORE
        assert self.choose(json_readable=False, json_fingerprint=None,
                           recorded_fingerprint=None) == FROM_STORE

    def test_a_store_without_a_record_wins_when_it_holds_tracks(self):
        assert self.choose(recorded_fingerprint=None) == FROM_STORE

    def test_an_empty_store_without_a_record_was_never_written(self):
        assert self.choose(recorded_fingerprint=None,
                           stored_tracks=0) == FROM_JSON

    def test_a_clearly_newer_file_was_saved_without_the_store(self):
        assert self.choose(recorded_fingerprint=None, json_mtime=110.0,
                           store_mtime=100.0) == FROM_JSON
        # the order of one save: file, then store
        assert self.choose(recorded_fingerprint=None, json_mtime=100.0,
                           store_mtime=100.0) == FROM_STORE
        assert self.choose(recorded_fingerprint=None, json_mtime=101.0,
                           store_mtime=100.0) == FROM_STORE


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

def test_what_was_saved_is_what_the_store_gives_back(tmp_path):
    root = _project(str(tmp_path))
    saved = _saved(root)

    loaded = _loaded(root)

    assert loaded.loaded_from == FROM_STORE
    assert loaded.load_notes == []
    assert _documents(loaded) == _documents(saved)
    assert loaded.custom_fields == FIELDS
    assert loaded.schema_defined


def test_a_folder_without_a_project_store_reads_the_file(tmp_path):
    """Un-migrated 5.x folders have no store; nothing changes for them."""
    root = str(tmp_path / "legacy")
    saved = _saved(root)

    loaded = _loaded(root)

    assert loaded.loaded_from == FROM_JSON
    assert _documents(loaded) == _documents(saved)
    assert not label_store.has_labels_store(root, "t")


def test_no_labels_at_all(tmp_path):
    root = _project(str(tmp_path))
    loaded = _loaded(root)
    assert loaded.tracks == {} and loaded.load_notes == []
    assert not loaded.schema_defined


def test_saving_leaves_no_temporary_file(tmp_path):
    root = _project(str(tmp_path))
    ls = _saved(root)
    assert sorted(os.listdir(ls.folder)) == ["labels.csv", "labels.json"]


# ---------------------------------------------------------------------------
# The store was changed by another tool
# ---------------------------------------------------------------------------

def test_a_track_deleted_in_the_store_does_not_come_back(tmp_path):
    root = _project(str(tmp_path))
    _saved(root)
    _delete_in_store(root, 2)

    loaded = _loaded(root)

    assert sorted(loaded.tracks) == [1]
    assert loaded.load_notes == []          # nothing unusual happened
    loaded.save()
    assert _stored_ids(root) == [1]
    with open(loaded.json_path, encoding="utf-8") as fh:
        assert [t["track_id"] for t in json.load(fh)["tracks"]] == [1]


def test_a_track_deleted_in_the_inspector_does_not_come_back(tmp_path):
    """The route it really takes: the labelled track is materialised, then
    deleted as a track - which removes its label track from the store."""
    from bambi_wildlife_detection.core import review

    root = _project(str(tmp_path))
    ls = _saved(root)
    ls.export_to_detections()
    manual = {row["label_track_id"]: row["track_id"]
              for row in label_store.load_tracks(root, "t")}
    assert manual[2] is not None

    review.delete_tracks(root, "t", [manual[2]])

    loaded = _loaded(root)
    assert sorted(loaded.tracks) == [1]
    loaded.save()
    loaded.export_to_detections()
    assert _stored_ids(root) == [1]


def test_a_renamed_species_keeps_its_labels(tmp_path):
    """Ids make a rename free in the store; the file still had the old name,
    which no longer resolved - the label became "animal"."""
    root = _project(str(tmp_path))
    _saved(root)
    conn = store.open_store(store.project_path(root), store.PROJECT)
    try:
        with store.transaction(conn):
            conn.execute("UPDATE species SET name = 'Capreolus' "
                         "WHERE name = 'roe deer'")
    finally:
        conn.close()

    loaded = _loaded(root)

    assert loaded.tracks[1].species == "Capreolus"
    loaded.save()
    stored = {t["label_track_id"]: t for t in label_store.load_tracks(root, "t")}
    assert stored[1]["species_id"] == 1 != store.FALLBACK_SPECIES_ID


# ---------------------------------------------------------------------------
# The file is missing or cannot be read
# ---------------------------------------------------------------------------

def test_a_missing_file_does_not_empty_the_store(tmp_path):
    root = _project(str(tmp_path))
    saved = _saved(root)
    os.remove(saved.json_path)

    loaded = _loaded(root)

    assert loaded.loaded_from == FROM_STORE
    assert sorted(loaded.tracks) == [1, 2]
    loaded.save()
    assert _stored_ids(root) == [1, 2]
    assert os.path.isfile(loaded.json_path)


def test_a_missing_file_keeps_the_custom_values(tmp_path):
    """The field schema is written to the file only. Without it the values
    have no field, and values without a field are dropped - so the fields
    are worked out from the values."""
    root = _project(str(tmp_path))
    saved = _saved(root)
    os.remove(saved.json_path)

    loaded = _loaded(root)

    assert loaded.schema_defined
    assert {(f.name, f.type, f.scope) for f in loaded.custom_fields} == {
        ("collar", "bool", "track"), ("note", "string", "track"),
        ("posture", "string", "keyframe")}
    assert _documents(loaded) == _documents(saved)
    loaded.save()
    assert _documents(_loaded(root)) == _documents(saved)


@pytest.mark.parametrize("content", [
    '{"modality": "t", "tracks": [{"track_id": 1, "keyfr',      # cut short
    "",                                                         # empty
    "[1, 2, 3]",                                                # not labels
    '{"tracks": [{"track_id": 1, "keyframes": {"3": {"x1": 1}}}]}',
])
def test_an_unreadable_file_does_not_empty_the_store(tmp_path, content):
    root = _project(str(tmp_path))
    saved = _saved(root)
    with open(saved.json_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    loaded = _loaded(root)

    assert loaded.loaded_from == FROM_STORE
    assert sorted(loaded.tracks) == [1, 2]
    assert len(loaded.load_notes) == 1
    assert "could not be read" in loaded.load_notes[0]
    loaded.save()
    assert _stored_ids(root) == [1, 2]
    # the file as it was is kept, not written over
    with open(loaded.json_path + ".unreadable", encoding="utf-8") as fh:
        assert fh.read() == content
    assert _loaded(root).load_notes == []


def test_an_unreadable_file_without_a_store_is_kept_too(tmp_path):
    root = str(tmp_path / "legacy")
    saved = _saved(root)
    with open(saved.json_path, "w", encoding="utf-8") as fh:
        fh.write("{ not json")

    ls = LabelStore(root, "t")
    with pytest.raises(ValueError, match="could not be read"):
        ls.load()
    assert ls.tracks == {}

    track = LabelTrack(1, species="fox")
    track.set_keyframe(1, (0.0, 0.0, 1.0, 1.0))
    ls.tracks[1] = track
    ls.save()

    with open(ls.json_path + ".unreadable", encoding="utf-8") as fh:
        assert fh.read() == "{ not json"
    assert sorted(_loaded(root).tracks) == [1]


def test_a_store_that_cannot_be_read_is_not_written_over(tmp_path, monkeypatch):
    root = _project(str(tmp_path))
    saved = _saved(root)
    with open(saved.json_path, encoding="utf-8") as fh:
        before = fh.read()

    def locked(*_args, **_kwargs):
        raise RuntimeError("database is locked")

    ls = LabelStore(root, "t")
    with monkeypatch.context() as patched:
        patched.setattr(label_store, "load_tracks", locked)
        with pytest.raises(RuntimeError, match="locked"):
            ls.load()

    assert ls.tracks == {}
    with pytest.raises(RuntimeError, match="open it again"):
        ls.save()
    assert _stored_ids(root) == [1, 2]
    with open(saved.json_path, encoding="utf-8") as fh:
        assert fh.read() == before
    # opening again works
    assert sorted(_loaded(root).tracks) == [1, 2]


# ---------------------------------------------------------------------------
# The file was changed from outside
# ---------------------------------------------------------------------------

def test_a_file_put_there_from_outside_is_loaded(tmp_path):
    root = _project(str(tmp_path))
    saved = _saved(root)

    def edit(document):
        document["tracks"] = [t for t in document["tracks"]
                              if t["track_id"] == 1]
        document["tracks"][0]["sex"] = "male"
    _rewrite_json(saved, edit)

    loaded = _loaded(root)

    assert loaded.loaded_from == FROM_JSON
    assert sorted(loaded.tracks) == [1]
    assert loaded.tracks[1].sex == "male"
    assert len(loaded.load_notes) == 1
    assert "changed outside" in loaded.load_notes[0]
    # what the store held is not lost to it
    replaced = os.path.join(loaded.folder, labelling.REPLACED_NAME)
    with open(replaced, encoding="utf-8") as fh:
        assert [t["track_id"] for t in json.load(fh)["tracks"]] == [1, 2]

    loaded.save()
    again = _loaded(root)
    assert again.loaded_from == FROM_STORE and again.load_notes == []
    assert sorted(again.tracks) == [1] and again.tracks[1].sex == "male"


def test_a_file_rewritten_with_the_same_labels_leaves_nothing_behind(tmp_path):
    """Reformatted, say: the fingerprint differs, the labels do not."""
    root = _project(str(tmp_path))
    saved = _saved(root)
    _rewrite_json(saved, lambda document: None)

    loaded = _loaded(root)

    assert loaded.loaded_from == FROM_JSON
    assert _documents(loaded) == _documents(saved)
    assert not os.path.exists(
        os.path.join(loaded.folder, labelling.REPLACED_NAME))


# ---------------------------------------------------------------------------
# A store written before fingerprints were recorded
# ---------------------------------------------------------------------------

def test_an_older_store_in_step_with_its_file_loads_quietly(tmp_path):
    root = _project(str(tmp_path))
    saved = _saved(root)
    _forget_fingerprint(root)

    loaded = _loaded(root)

    assert loaded.loaded_from == FROM_STORE
    assert loaded.load_notes == []
    assert _documents(loaded) == _documents(saved)
    assert not os.path.exists(loaded.json_path + ".bak")


def test_an_older_store_that_was_changed_wins_and_the_file_is_kept(tmp_path):
    root = _project(str(tmp_path))
    saved = _saved(root)
    with open(saved.json_path, encoding="utf-8") as fh:
        before = fh.read()
    _delete_in_store(root, 2)
    _forget_fingerprint(root)

    loaded = _loaded(root)

    assert loaded.loaded_from == FROM_STORE
    assert sorted(loaded.tracks) == [1]
    assert len(loaded.load_notes) == 1
    assert "labels.json.bak" in loaded.load_notes[0]
    with open(loaded.json_path + ".bak", encoding="utf-8") as fh:
        assert fh.read() == before


def test_an_older_empty_store_does_not_hide_the_file(tmp_path):
    """A labels store nobody saved labels into - the worst mistake would be
    to show no labels and let the next save write that over the file."""
    root = _project(str(tmp_path))
    saved = _saved(root)
    label_store.save_tracks(root, "t", [])
    _forget_fingerprint(root)

    loaded = _loaded(root)

    assert loaded.loaded_from == FROM_JSON
    assert _documents(loaded) == _documents(saved)
    assert loaded.load_notes == []


def test_an_older_store_with_a_newer_file_loads_the_file(tmp_path):
    """The file is written first; when the store then failed, the file is
    the newer state."""
    root = _project(str(tmp_path))
    saved = _saved(root)
    _delete_in_store(root, 2)
    _forget_fingerprint(root)
    stamp = os.path.getmtime(
        store.stage_path(root, store.LABELS, "t")) + 60.0
    os.utime(saved.json_path, (stamp, stamp))

    loaded = _loaded(root)

    assert loaded.loaded_from == FROM_JSON
    assert sorted(loaded.tracks) == [1, 2]
    assert "newer" in loaded.load_notes[0]


# ---------------------------------------------------------------------------
# Species the store could not name
# ---------------------------------------------------------------------------

def _add_species(root, name):
    conn = store.open_store(store.project_path(root), store.PROJECT)
    try:
        with store.transaction(conn):
            cur = conn.execute(
                "INSERT INTO species (name, protected) VALUES (?, 0)", (name,))
            return int(cur.lastrowid)
    finally:
        conn.close()


def _as_fallback(root, label_track_id):
    """What versions before 9.1 stored for a name with a capital letter."""
    conn = store.open_store(
        store.stage_path(root, store.LABELS, "t"), store.LABELS, "t")
    try:
        with store.transaction(conn):
            conn.execute(
                "UPDATE label_tracks SET species_id = ? "
                "WHERE label_track_id = ?",
                (store.FALLBACK_SPECIES_ID, label_track_id))
    finally:
        conn.close()


def test_a_species_stored_as_animal_gets_its_name_back(tmp_path):
    """Labels saved before 9.1 for "Alpine ibex" are "animal" in the store
    and "Alpine ibex" in the file. Reading the store must not make the
    damage final."""
    root = _project(str(tmp_path))
    ibex = _add_species(root, "Alpine ibex")
    tracks = _tracks()
    tracks[2].species = "Alpine ibex"
    _saved(root, tracks)
    _as_fallback(root, 2)
    _forget_fingerprint(root)

    loaded = _loaded(root)

    assert loaded.loaded_from == FROM_STORE
    assert loaded.tracks[2].species == "Alpine ibex"
    assert loaded.load_notes == []          # nothing differs from the file
    loaded.save()
    stored = {t["label_track_id"]: t for t in label_store.load_tracks(root, "t")}
    assert stored[2]["species_id"] == ibex


def test_a_name_the_vocabulary_lacks_stays_in_the_file(tmp_path):
    """Typed before the project had a vocabulary. It cannot be stored, but
    it is not thrown away: once the species exists, the label finds it."""
    root = _project(str(tmp_path))
    tracks = _tracks()
    tracks[2].species = "chupacabra"
    _saved(root, tracks)

    loaded = _loaded(root)
    assert loaded.tracks[2].species == "chupacabra"
    loaded.save()

    added = _add_species(root, "Chupacabra")
    later = _loaded(root)
    later.save()
    stored = {t["label_track_id"]: t for t in label_store.load_tracks(root, "t")}
    assert stored[2]["species_id"] == added


def test_a_label_that_is_animal_in_both_stays_animal(tmp_path):
    root = _project(str(tmp_path))
    tracks = _tracks()
    tracks[2].species = "animal"
    _saved(root, tracks)
    assert _loaded(root).tracks[2].species == "animal"


# ---------------------------------------------------------------------------
# Reading the store's rows
# ---------------------------------------------------------------------------

class TestFromStore:
    VOCABULARY = {
        "species": [{"species_id": 0, "name": "animal"},
                    {"species_id": 3, "name": "Alpine ibex"}],
        "enums": {"sex": [{"value_id": 1, "label": "female"}],
                  "occlusion": [{"value_id": 0, "label": "clear"}]},
    }

    def row(self, **changes):
        row = {"label_track_id": 4, "species_id": 3, "origin_track_id": None,
               "attributes": {"sex": 1, "tag": "A7"},
               "keyframes": [{"frame": 2, "x1": 1, "y1": 2, "x2": 3, "y2": 4,
                              "stop": 0, "origin_detection_id": None,
                              "attributes": {"occlusion": 0}}]}
        row.update(changes)
        return row

    def test_ids_become_names(self):
        track = labelling.tracks_from_store([self.row()], self.VOCABULARY)[4]
        assert (track.species, track.sex, track.age) == (
            "Alpine ibex", "female", "unknown")
        assert track.attributes == {"tag": "A7"}
        assert track.keyframes == {2: {
            "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0, "occlusion": "clear"}}

    def test_what_is_not_stored_is_not_made_up(self):
        row = self.row(species_id=99, attributes={"sex": 7, "age": [1]})
        row["keyframes"][0]["attributes"] = {}
        track = labelling.tracks_from_store([row], self.VOCABULARY)[4]
        assert (track.species, track.sex, track.age) == (
            "unknown", "unknown", "unknown")
        # no occlusion annotated - which is not the same as "clear"
        assert track.keyframes[2]["occlusion"] == "none"


class TestInferredFields:
    def track(self, attributes=None, keyframe_attributes=None):
        track = LabelTrack(1, attributes=attributes or {})
        track.set_keyframe(1, (0.0, 0.0, 1.0, 1.0),
                           attributes=keyframe_attributes or {})
        return track

    def test_types_follow_the_values(self):
        fields = labelling.infer_custom_fields([self.track(
            {"n": 3, "w": 2.5, "ok": True, "s": "x"}, {"p": "lying"})])
        assert {(f.name, f.type, f.scope) for f in fields} == {
            ("n", "int", "track"), ("w", "float", "track"),
            ("ok", "bool", "track"), ("s", "string", "track"),
            ("p", "string", "keyframe")}

    def test_known_fields_stay_as_they_are(self):
        known = [CustomField("n", "float", "track")]
        fields = labelling.infer_custom_fields(
            [self.track({"n": 3, "extra": 1})], known)
        assert fields[0] is known[0]
        assert [(f.name, f.type) for f in fields] == [
            ("n", "float"), ("extra", "int")]

    def test_mixed_values_get_a_type_that_holds_both(self):
        fields = labelling.infer_custom_fields([
            self.track({"a": 1, "b": 1}), self.track({"a": 2.5, "b": "x"})])
        assert {f.name: f.type for f in fields} == {
            "a": "float", "b": "string"}

    def test_reserved_names_are_not_fields(self):
        assert labelling.infer_custom_fields(
            [self.track({"species": "fox", "x1": 3})]) == []
