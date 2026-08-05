# -*- coding: utf-8 -*-
"""Tests for store creation, paths, versioning and transactions.

Covers EXCHANGE_FORMAT_PLAN.md §3 (schema), §10 Phase 0 and the version-guard
ground rule of §11. The id/vocabulary invariants of §4.1 live in
``test_store_ids.py``.
"""
import os
import sqlite3

import pytest

from bambi_wildlife_detection.core import gpkg, store


def _tables(conn):
    return {row["name"] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'")}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def test_project_path(tmp_path):
    assert store.project_path(str(tmp_path)) == \
        os.path.join(str(tmp_path), "project.gpkg")


@pytest.mark.parametrize("modality", store.MODALITIES)
def test_stage_path_per_modality(tmp_path, modality):
    path = store.stage_path(str(tmp_path), store.TRACKS, modality)
    assert path.endswith(os.path.join(f"bambi_{modality}", "tracks.gpkg"))


def test_stage_path_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError, match="stage kind"):
        store.stage_path(str(tmp_path), "nonsense", "t")


def test_stage_path_rejects_unknown_modality(tmp_path):
    with pytest.raises(ValueError, match="modality"):
        store.stage_path(str(tmp_path), store.TRACKS, "x")


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

def test_open_store_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError, match="store kind"):
        store.open_store(str(tmp_path / "x.gpkg"), "nonsense")


@pytest.mark.parametrize("kind,expected", [
    (store.PROJECT, {"species", "enums", "enum_values", "field_schema",
                     "detection_sources", "class_mapping", "stages"}),
    (store.DETECTIONS, {"detections"}),
    (store.GEOREFERENCED, {"detections_geo", "georef_failures"}),
    (store.TRACKS, {"track_runs", "tracks", "track_members"}),
    (store.FOV, {"fov_polygons", "fov_vertices"}),
    (store.LABELS, {"label_tracks", "label_keyframes",
                    "label_track_origin_xmodal"}),
    (store.SEGMENTATION, {"segments"}),
])
def test_each_kind_creates_its_tables(tmp_path, kind, expected):
    conn = store.open_store(str(tmp_path / f"{kind}.gpkg"), kind, "t")
    assert expected <= _tables(conn)
    assert "bambi_meta" in _tables(conn)
    conn.close()


@pytest.mark.parametrize("kind", [store.PROJECT] + list(store.STAGE_KINDS))
def test_created_file_is_a_geopackage(tmp_path, kind):
    path = str(tmp_path / f"{kind}.gpkg")
    store.open_store(path, kind, "t").close()
    assert gpkg.is_geopackage(path)


@pytest.mark.parametrize("kind", [store.PROJECT] + list(store.STAGE_KINDS))
def test_payload_tables_are_registered_for_qgis(tmp_path, kind):
    """Without a gpkg_contents row a table is invisible to the GPKG driver."""
    conn = store.open_store(str(tmp_path / f"{kind}.gpkg"), kind, "t")
    registered = set(gpkg.registered_tables(conn))
    payload = _tables(conn) - {"sqlite_sequence"}
    payload = {t for t in payload if not t.startswith("gpkg_")}
    assert payload == registered
    conn.close()


def test_geopackage_internals_are_not_registered_as_payload(tmp_path):
    conn = store.open_store(str(tmp_path / "p.gpkg"), store.PROJECT)
    assert not any(t.startswith("gpkg_") for t in gpkg.registered_tables(conn))
    conn.close()


def test_meta_is_written_on_creation(tmp_path):
    conn = store.open_store(str(tmp_path / "d.gpkg"), store.DETECTIONS, "w")
    assert int(store.get_meta(conn, "schema_version")) == store.SCHEMA_VERSION
    assert store.get_meta(conn, "kind") == store.DETECTIONS
    assert store.get_meta(conn, "modality") == "w"
    assert store.get_meta(conn, "created_at")
    conn.close()


def test_get_meta_returns_default_for_missing_key(tmp_path):
    conn = store.open_store(str(tmp_path / "d.gpkg"), store.DETECTIONS)
    assert store.get_meta(conn, "nope", "fallback") == "fallback"
    conn.close()


def test_set_meta_overwrites(tmp_path):
    conn = store.open_store(str(tmp_path / "d.gpkg"), store.DETECTIONS)
    store.set_meta(conn, "generation", 3)
    store.set_meta(conn, "generation", 4)
    assert store.get_meta(conn, "generation") == "4"
    conn.close()


def test_reopening_does_not_recreate_or_clear(tmp_path):
    path = str(tmp_path / "d.gpkg")
    conn = store.open_store(path, store.DETECTIONS, "t")
    conn.execute(
        "INSERT INTO detections (frame, source_id) VALUES (1, 1)")
    conn.commit()
    conn.close()

    conn = store.open_store(path, store.DETECTIONS, "t")
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM detections").fetchone()["n"] == 1
    conn.close()


def test_open_store_without_create_rejects_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        store.open_store(str(tmp_path / "absent.gpkg"),
                         store.DETECTIONS, create=False)


# ---------------------------------------------------------------------------
# Version and kind guards
# ---------------------------------------------------------------------------

def test_newer_schema_version_is_refused(tmp_path):
    path = str(tmp_path / "d.gpkg")
    conn = store.open_store(path, store.DETECTIONS)
    store.set_meta(conn, "schema_version", store.SCHEMA_VERSION + 1)
    conn.commit()
    conn.close()

    with pytest.raises(store.StoreVersionError, match="newer version"):
        store.open_store(path, store.DETECTIONS)


def test_same_schema_version_is_accepted(tmp_path):
    path = str(tmp_path / "d.gpkg")
    store.open_store(path, store.DETECTIONS).close()
    store.open_store(path, store.DETECTIONS).close()


def test_unreadable_schema_version_is_refused(tmp_path):
    path = str(tmp_path / "d.gpkg")
    conn = store.open_store(path, store.DETECTIONS)
    store.set_meta(conn, "schema_version", "not-a-number")
    conn.commit()
    conn.close()

    with pytest.raises(store.StoreVersionError, match="not a BAMBI store"):
        store.open_store(path, store.DETECTIONS)


def test_foreign_file_is_refused(tmp_path):
    path = str(tmp_path / "foreign.gpkg")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE whatever (a INTEGER)")
    conn.commit()
    conn.close()

    with pytest.raises(store.StoreVersionError):
        store.open_store(path, store.DETECTIONS)


def test_wrong_kind_is_refused(tmp_path):
    """Opening tracks.gpkg as a detections store is a bug, not an empty result."""
    path = str(tmp_path / "tracks.gpkg")
    store.open_store(path, store.TRACKS, "t").close()

    with pytest.raises(store.StoreVersionError, match="not 'detections'"):
        store.open_store(path, store.DETECTIONS, "t")


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

def test_transaction_commits_on_success(tmp_path):
    path = str(tmp_path / "d.gpkg")
    conn = store.open_store(path, store.DETECTIONS)
    with store.transaction(conn):
        conn.execute("INSERT INTO detections (frame, source_id) VALUES (5, 1)")
    conn.close()

    conn = store.open_store(path, store.DETECTIONS)
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM detections").fetchone()["n"] == 1
    conn.close()


def test_transaction_rolls_back_on_error(tmp_path):
    """A cancelled stage must leave no half-written table (§11)."""
    path = str(tmp_path / "d.gpkg")
    conn = store.open_store(path, store.DETECTIONS)
    with pytest.raises(RuntimeError):
        with store.transaction(conn):
            conn.execute("INSERT INTO detections (frame, source_id) VALUES (5, 1)")
            raise RuntimeError("stage cancelled")

    assert conn.execute(
        "SELECT COUNT(*) AS n FROM detections").fetchone()["n"] == 0
    conn.close()


# ---------------------------------------------------------------------------
# Schema details that downstream phases depend on
# ---------------------------------------------------------------------------

def test_label_materialisation_key_is_unique(tmp_path):
    """The (label_track_id, frame) upsert key of §6.2."""
    conn = store.open_store(str(tmp_path / "d.gpkg"), store.DETECTIONS)
    conn.execute("INSERT INTO detections (frame, source_id, label_track_id) "
                 "VALUES (10, 1, 4)")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO detections (frame, source_id, label_track_id) "
                     "VALUES (10, 1, 4)")
    conn.close()


def test_detector_rows_are_not_constrained_by_the_label_key(tmp_path):
    """The partial index must not restrict rows without a label track."""
    conn = store.open_store(str(tmp_path / "d.gpkg"), store.DETECTIONS)
    conn.executemany(
        "INSERT INTO detections (frame, source_id) VALUES (?, ?)",
        [(10, 1), (10, 1), (10, 1)])
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM detections").fetchone()["n"] == 3
    conn.close()


def test_detection_ids_are_never_reused(tmp_path):
    """AUTOINCREMENT: a deleted id must not come back meaning another animal."""
    conn = store.open_store(str(tmp_path / "d.gpkg"), store.DETECTIONS)
    conn.execute("INSERT INTO detections (frame, source_id) VALUES (1, 1)")
    first = conn.execute("SELECT MAX(detection_id) AS m FROM detections"
                         ).fetchone()["m"]
    conn.execute("DELETE FROM detections")
    conn.execute("INSERT INTO detections (frame, source_id) VALUES (2, 1)")
    second = conn.execute("SELECT MAX(detection_id) AS m FROM detections"
                          ).fetchone()["m"]
    assert second > first
    conn.close()


def test_track_ids_are_never_reused(tmp_path):
    conn = store.open_store(str(tmp_path / "t.gpkg"), store.TRACKS)
    conn.execute("INSERT INTO track_runs (kind) VALUES ('builtin')")
    conn.execute("INSERT INTO tracks (run_id) VALUES (1)")
    first = conn.execute("SELECT MAX(track_id) AS m FROM tracks").fetchone()["m"]
    conn.execute("DELETE FROM tracks")
    conn.execute("INSERT INTO tracks (run_id) VALUES (1)")
    second = conn.execute("SELECT MAX(track_id) AS m FROM tracks").fetchone()["m"]
    assert second > first
    conn.close()


def test_species_id_defaults_to_animal(tmp_path):
    """'No species' must never be representable (§3.1)."""
    conn = store.open_store(str(tmp_path / "d.gpkg"), store.DETECTIONS)
    conn.execute("INSERT INTO detections (frame, source_id) VALUES (1, 1)")
    row = conn.execute("SELECT species_id FROM detections").fetchone()
    assert row["species_id"] == store.FALLBACK_SPECIES_ID
    conn.close()


def test_species_id_cannot_be_null(tmp_path):
    conn = store.open_store(str(tmp_path / "d.gpkg"), store.DETECTIONS)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO detections (frame, source_id, species_id) "
                     "VALUES (1, 1, NULL)")
    conn.close()


# ---------------------------------------------------------------------------
# In-place schema upgrades (§11)
# ---------------------------------------------------------------------------

def _downgrade_to_v1(path):
    """Rebuild the species table in its v1 shape, without the taxonomy columns.

    Done with CREATE/INSERT/DROP/RENAME rather than ALTER TABLE DROP COLUMN,
    which needs SQLite 3.35+.
    """
    conn = store.open_store(path, store.PROJECT)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("CREATE TABLE species_v1 (species_id INTEGER PRIMARY KEY, "
                 "name TEXT UNIQUE NOT NULL, protected INTEGER NOT NULL "
                 "DEFAULT 0)")
    conn.execute("INSERT INTO species_v1 SELECT species_id, name, protected "
                 "FROM species")
    conn.execute("DROP TABLE species")
    conn.execute("ALTER TABLE species_v1 RENAME TO species")
    store.set_meta(conn, "schema_version", 1)
    conn.commit()
    conn.close()


def test_an_older_store_is_upgraded_in_place(tmp_path):
    """Additive upgrades only, so an older plugin can still read the file."""
    path = str(tmp_path / "project.gpkg")
    store.open_store(path, store.PROJECT).close()
    _downgrade_to_v1(path)

    conn = store.open_store(path, store.PROJECT)
    try:
        assert int(store.get_meta(conn, "schema_version")) == store.SCHEMA_VERSION
        row = conn.execute(
            "SELECT scientific_name FROM species WHERE name = 'roe deer'"
        ).fetchone()
        assert row["scientific_name"] == "Capreolus capreolus"
    finally:
        conn.close()


def test_upgrading_does_not_disturb_existing_rows(tmp_path):
    path = str(tmp_path / "project.gpkg")
    conn = store.open_store(path, store.PROJECT)
    conn.execute("INSERT INTO species (species_id, name) VALUES (10, 'wolf')")
    conn.commit()
    conn.close()
    _downgrade_to_v1(path)

    conn = store.open_store(path, store.PROJECT)
    try:
        names = {r["name"] for r in conn.execute("SELECT name FROM species")}
        assert "wolf" in names
    finally:
        conn.close()


def test_a_current_store_is_not_upgraded(tmp_path):
    path = str(tmp_path / "project.gpkg")
    store.open_store(path, store.PROJECT).close()
    conn = store.open_store(path, store.PROJECT)
    try:
        assert int(store.get_meta(conn, "schema_version")) == store.SCHEMA_VERSION
    finally:
        conn.close()
