# -*- coding: utf-8 -*-
"""The id-stability invariants of EXCHANGE_FORMAT_PLAN.md §4.1.

These are the contract of the 6.0 rework: if renaming, reordering or deleting
part of the vocabulary could change what stored data means, the format has
merely reinvented ``species_class_ids()``. The table in §4.1 is only worth
anything if it is executable, so it is asserted here row by row.
"""
import sqlite3

import pytest

from bambi_wildlife_detection.core import store


@pytest.fixture
def project(tmp_path):
    conn = store.open_store(store.project_path(str(tmp_path)), store.PROJECT)
    yield conn
    conn.close()


def _species(conn):
    return {row["name"]: row["species_id"]
            for row in conn.execute("SELECT species_id, name FROM species")}


def _add_source(conn, kind="detector"):
    cur = conn.execute(
        "INSERT INTO detection_sources (kind, created_at) VALUES (?, '')",
        (kind,))
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

def test_three_base_classes_exist(project):
    names = _species(project)
    assert names["animal"] == 0
    assert names["unknown"] == -1
    assert names["not-an-animal"] == -2


def test_base_classes_are_protected(project):
    rows = project.execute(
        "SELECT name FROM species WHERE protected = 1").fetchall()
    assert {row["name"] for row in rows} == {"animal", "unknown", "not-an-animal"}


def test_sign_separates_base_from_concrete_species(project):
    """`species_id <= 0` is a base class, `>= 1` a concrete species (§3.1)."""
    for row in project.execute("SELECT species_id, protected FROM species"):
        assert (row["species_id"] <= 0) == bool(row["protected"])


def test_legacy_taxonomy_keeps_its_indices(project):
    """5.x class ids must keep their meaning after migration (§9)."""
    names = _species(project)
    assert names["roe deer"] == 1
    assert names["red deer"] == 2
    assert names["other"] == 9


def test_base_species_cannot_be_deleted(project):
    for name in ("animal", "unknown", "not-an-animal"):
        with pytest.raises(sqlite3.IntegrityError, match="protected species"):
            project.execute("DELETE FROM species WHERE name = ?", (name,))


def test_base_species_cannot_be_renamed(project):
    with pytest.raises(sqlite3.IntegrityError, match="protected species"):
        project.execute(
            "UPDATE species SET name = 'critter' WHERE species_id = 0")


def test_concrete_species_can_be_renamed_keeping_its_id(project):
    project.execute("UPDATE species SET name = 'Roe Deer' WHERE species_id = 1")
    assert _species(project)["Roe Deer"] == 1


def test_concrete_species_can_be_deleted(project):
    project.execute("DELETE FROM species WHERE species_id = 9")
    assert "other" not in _species(project)


# ---------------------------------------------------------------------------
# Append-only ids
# ---------------------------------------------------------------------------

def test_species_ids_are_never_renumbered(project):
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        project.execute("UPDATE species SET species_id = 42 WHERE species_id = 1")


def test_enum_value_ids_are_never_renumbered(project):
    enum_id = project.execute(
        "SELECT enum_id FROM enums WHERE name = 'sex'").fetchone()["enum_id"]
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        project.execute(
            "UPDATE enum_values SET value_id = 9 "
            "WHERE enum_id = ? AND value_id = 1", (enum_id,))


def test_next_species_id_is_above_every_existing_one(project):
    assert store.next_species_id(project) == 10
    project.execute("INSERT INTO species (species_id, name) VALUES (10, 'wolf')")
    assert store.next_species_id(project) == 11


def test_deleted_species_ids_are_not_reissued(project):
    """Adding a species after deleting one must not reuse the freed id.

    Only :func:`store.reserve_species_id` guarantees this — ``next_species_id``
    is a non-reserving peek, and a bare ``MAX(species_id) + 1`` would hand the
    id of a deleted top-most species straight back.
    """
    ids = []
    for name in ("wolf", "lynx", "badger"):
        species_id = store.reserve_species_id(project)
        project.execute(
            "INSERT INTO species (species_id, name) VALUES (?, ?)",
            (species_id, name))
        ids.append(species_id)

    project.execute("DELETE FROM species WHERE species_id IN (?, ?)",
                    (ids[-1], ids[-2]))
    assert store.reserve_species_id(project) > max(ids)


def test_adding_a_species_never_renumbers_existing_ones(project):
    """The 5.x bug: a new custom species renumbered the ones after it."""
    before = _species(project)
    for name in ("badger", "aardvark", "zebra"):
        project.execute(
            "INSERT INTO species (species_id, name) VALUES (?, ?)",
            (store.next_species_id(project), name))
    after = _species(project)
    for name, species_id in before.items():
        assert after[name] == species_id


def test_next_enum_value_id_appends(project):
    enum_id = project.execute(
        "SELECT enum_id FROM enums WHERE name = 'age'").fetchone()["enum_id"]
    assert store.next_enum_value_id(project, enum_id) == 3
    project.execute(
        "INSERT INTO enum_values (enum_id, value_id, label, ordinal) "
        "VALUES (?, 3, 'subadult', 3)", (enum_id,))
    assert store.next_enum_value_id(project, enum_id) == 4


def test_next_enum_value_id_of_empty_enum(project):
    cur = project.execute("INSERT INTO enums (name) VALUES ('behaviour')")
    assert store.next_enum_value_id(project, cur.lastrowid) == 0


# ---------------------------------------------------------------------------
# Seeded enums and fields (§5.2)
# ---------------------------------------------------------------------------

def test_seeded_enums_exist_with_their_values(project):
    labels = {}
    for row in project.execute(
            "SELECT e.name AS enum, v.label AS label, v.value_id AS value_id "
            "FROM enums e JOIN enum_values v USING (enum_id)"):
        labels.setdefault(row["enum"], {})[row["label"]] = row["value_id"]

    assert labels["sex"] == {"unknown": 0, "female": 1, "male": 2}
    assert labels["age"] == {"unknown": 0, "adult": 1, "juvenile": 2}
    # Two-valued, and named after what the occlusion classifier reports, so
    # predictions need no translation into a different vocabulary.
    assert labels["occlusion"] == {"clear": 0, "occluded": 1}


def test_seeded_fields_are_enum_typed_and_protected(project):
    rows = {row["name"]: row for row in project.execute(
        "SELECT name, type, scope, enum_id, protected FROM field_schema")}
    assert rows["sex"]["type"] == "enum" and rows["sex"]["scope"] == "track"
    assert rows["age"]["scope"] == "track"
    assert rows["occlusion"]["scope"] == "detection"
    assert all(row["protected"] == 1 for row in rows.values())
    assert all(row["enum_id"] is not None for row in rows.values())


def test_enum_labels_are_unique_within_an_enum(project):
    enum_id = project.execute(
        "SELECT enum_id FROM enums WHERE name = 'sex'").fetchone()["enum_id"]
    with pytest.raises(sqlite3.IntegrityError):
        project.execute(
            "INSERT INTO enum_values (enum_id, value_id, label, ordinal) "
            "VALUES (?, 7, 'female', 7)", (enum_id,))


def test_renaming_an_enum_value_keeps_its_id(project):
    """Stored rows hold value_id, so a rename must not touch them (§5.1)."""
    enum_id = project.execute(
        "SELECT enum_id FROM enums WHERE name = 'sex'").fetchone()["enum_id"]
    project.execute(
        "UPDATE enum_values SET label = 'Female' "
        "WHERE enum_id = ? AND value_id = 1", (enum_id,))
    row = project.execute(
        "SELECT label FROM enum_values WHERE enum_id = ? AND value_id = 1",
        (enum_id,)).fetchone()
    assert row["label"] == "Female"


def test_ordinal_can_be_reordered_freely(project):
    enum_id = project.execute(
        "SELECT enum_id FROM enums WHERE name = 'age'").fetchone()["enum_id"]
    project.execute(
        "UPDATE enum_values SET ordinal = 99 WHERE enum_id = ? AND value_id = 0",
        (enum_id,))
    row = project.execute(
        "SELECT ordinal FROM enum_values WHERE enum_id = ? AND value_id = 0",
        (enum_id,)).fetchone()
    assert row["ordinal"] == 99


def test_enum_typed_field_requires_an_enum(project):
    with pytest.raises(sqlite3.IntegrityError):
        project.execute(
            "INSERT INTO field_schema (name, type, scope) "
            "VALUES ('mood', 'enum', 'track')")


def test_string_fields_remain_available(project):
    """Free text stays a supported type — it is just not in the default setup."""
    project.execute(
        "INSERT INTO field_schema (name, type, scope) "
        "VALUES ('collar_id', 'string', 'track')")
    row = project.execute(
        "SELECT type FROM field_schema WHERE name = 'collar_id'").fetchone()
    assert row["type"] == "string"


# ---------------------------------------------------------------------------
# Class mapping (§3.1)
# ---------------------------------------------------------------------------

def test_unmapped_class_falls_back_to_animal(project):
    source_id = _add_source(project)
    assert store.resolve_species(project, source_id, "elk") == 0


def test_species_agnostic_source_needs_no_mapping_rows(project):
    """A detector with no vocabulary at all resolves everything to animal."""
    source_id = _add_source(project, "agnostic-detector")
    for raw in ("0", "1", "blob", None):
        assert store.resolve_species(project, source_id, raw) == 0


def test_mapped_class_resolves_to_its_species(project):
    source_id = _add_source(project)
    project.execute(
        "INSERT INTO class_mapping (source_id, source_class, species_id) "
        "VALUES (?, '3', 1)", (source_id,))
    assert store.resolve_species(project, source_id, "3") == 1
    assert store.resolve_species(project, source_id, 3) == 1


def test_mapping_is_scoped_to_its_source(project):
    """Two detectors may use the same raw class id for different species."""
    a = _add_source(project, "detector-a")
    b = _add_source(project, "detector-b")
    project.execute(
        "INSERT INTO class_mapping (source_id, source_class, species_id) "
        "VALUES (?, '1', 1)", (a,))
    project.execute(
        "INSERT INTO class_mapping (source_id, source_class, species_id) "
        "VALUES (?, '1', 2)", (b,))
    assert store.resolve_species(project, a, "1") == 1
    assert store.resolve_species(project, b, "1") == 2


def test_mapping_cannot_point_at_a_missing_species(project):
    source_id = _add_source(project)
    with pytest.raises(sqlite3.IntegrityError):
        project.execute(
            "INSERT INTO class_mapping (source_id, source_class, species_id) "
            "VALUES (?, '1', 4242)", (source_id,))


def test_changing_a_mapping_does_not_touch_species_ids(project):
    """Re-resolving is an UPDATE over rows; the vocabulary is unaffected."""
    source_id = _add_source(project)
    project.execute(
        "INSERT INTO class_mapping (source_id, source_class, species_id) "
        "VALUES (?, '1', 1)", (source_id,))
    before = _species(project)
    project.execute(
        "UPDATE class_mapping SET species_id = 2 "
        "WHERE source_id = ? AND source_class = '1'", (source_id,))
    assert store.resolve_species(project, source_id, "1") == 2
    assert _species(project) == before


# ---------------------------------------------------------------------------
# Seeding is idempotent
# ---------------------------------------------------------------------------

def test_seeding_twice_changes_nothing(project):
    before = _species(project)
    store.seed_vocabulary(project)
    assert _species(project) == before
    assert project.execute(
        "SELECT COUNT(*) AS n FROM enum_values").fetchone()["n"] == 8


def test_seeding_does_not_revert_user_edits(tmp_path):
    """Reopening a project must not disturb a renamed or extended taxonomy."""
    path = str(tmp_path / "project.gpkg")
    conn = store.open_store(path, store.PROJECT)
    conn.execute("UPDATE species SET name = 'Roe Deer' WHERE species_id = 1")
    conn.execute("INSERT INTO species (species_id, name) VALUES (10, 'wolf')")
    conn.commit()
    conn.close()

    conn = store.open_store(path, store.PROJECT)
    store.seed_vocabulary(conn)
    names = _species(conn)
    assert names["Roe Deer"] == 1
    assert names["wolf"] == 10
    assert "roe deer" not in names
    conn.close()


def test_an_existing_three_value_occlusion_enum_is_left_alone(tmp_path):
    """A project created before the two-value seed keeps its own vocabulary.

    Enum value ids are append-only, so 0/1/2 in an old project mean
    none/partially/fully and must go on meaning that — re-seeding must not
    relabel them into clear/occluded and silently reinterpret every stored
    occlusion value. The classifier's label mapping is what bridges the two.
    """
    path = str(tmp_path / "project.gpkg")
    conn = store.open_store(path, store.PROJECT)
    enum_id = conn.execute(
        "SELECT enum_id FROM enums WHERE name = 'occlusion'").fetchone()[
            "enum_id"]
    # Rewrite the seeded values into what a pre-6.1 project holds.
    conn.execute("DELETE FROM enum_values WHERE enum_id = ?", (enum_id,))
    conn.executemany(
        "INSERT INTO enum_values (enum_id, value_id, label, ordinal) "
        "VALUES (?, ?, ?, ?)",
        [(enum_id, 0, "none", 0), (enum_id, 1, "partially", 1),
         (enum_id, 2, "fully", 2)])
    conn.commit()
    conn.close()

    conn = store.open_store(path, store.PROJECT)   # re-seeds on open
    store.seed_vocabulary(conn)                    # and again, explicitly
    labels = {row["label"]: row["value_id"] for row in conn.execute(
        "SELECT label, value_id FROM enum_values WHERE enum_id = ?",
        (enum_id,))}
    assert labels == {"none": 0, "partially": 1, "fully": 2}
    conn.close()


def test_species_id_is_not_reissued_after_deleting_the_highest(project):
    """A plain MAX()+1 would hand the id straight back (§4.1)."""
    first = store.reserve_species_id(project)
    project.execute("INSERT INTO species (species_id, name) VALUES (?, 'wolf')",
                    (first,))
    project.execute("DELETE FROM species WHERE species_id = ?", (first,))
    assert store.reserve_species_id(project) > first


def test_enum_value_id_is_not_reissued_after_deleting_the_highest(project):
    enum_id = project.execute(
        "SELECT enum_id FROM enums WHERE name = 'age'").fetchone()["enum_id"]
    first = store.reserve_enum_value_id(project, enum_id)
    project.execute(
        "INSERT INTO enum_values (enum_id, value_id, label, ordinal) "
        "VALUES (?, ?, 'subadult', ?)", (enum_id, first, first))
    project.execute(
        "DELETE FROM enum_values WHERE enum_id = ? AND value_id = ?",
        (enum_id, first))
    assert store.reserve_enum_value_id(project, enum_id) > first


def test_note_species_id_advances_the_high_water_mark(project):
    """Migration assigns ids itself; they must not be handed out again."""
    store.note_species_id(project, 42)
    assert store.next_species_id(project) == 43


def test_note_species_id_never_moves_backwards(project):
    store.note_species_id(project, 42)
    store.note_species_id(project, 11)
    assert store.next_species_id(project) == 43


def test_reserve_is_stable_across_reopen(tmp_path):
    path = str(tmp_path / "project.gpkg")
    conn = store.open_store(path, store.PROJECT)
    first = store.reserve_species_id(conn)
    conn.commit()
    conn.close()

    conn = store.open_store(path, store.PROJECT)
    assert store.reserve_species_id(conn) > first
    conn.close()
