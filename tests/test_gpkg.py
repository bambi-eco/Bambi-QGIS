# -*- coding: utf-8 -*-
"""Tests for the aspatial GeoPackage container (``core/gpkg.py``).

The whole per-stage-file layout of EXCHANGE_FORMAT_PLAN.md §2 rests on these
files being valid GeoPackages, so the boilerplate is checked directly rather
than only through the higher-level store.
"""
import sqlite3

import pytest

from bambi_wildlife_detection.core import gpkg


@pytest.fixture
def db(tmp_path):
    conn = gpkg.connect(str(tmp_path / "sample.gpkg"))
    yield conn
    conn.close()


def test_connect_creates_file_with_geopackage_magic(tmp_path):
    path = str(tmp_path / "new.gpkg")
    conn = gpkg.connect(path)
    conn.close()
    assert gpkg.is_geopackage(path)


def test_connect_creates_missing_parent_directories(tmp_path):
    path = str(tmp_path / "bambi_t" / "detections.gpkg")
    conn = gpkg.connect(path)
    conn.close()
    assert gpkg.is_geopackage(path)


def test_connect_without_create_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        gpkg.connect(str(tmp_path / "absent.gpkg"), create=False)


def test_application_id_and_user_version(db):
    assert db.execute("PRAGMA application_id").fetchone()[0] == gpkg.APPLICATION_ID
    assert db.execute("PRAGMA user_version").fetchone()[0] == gpkg.USER_VERSION


def test_mandatory_spatial_ref_sys_rows(db):
    ids = {row["srs_id"] for row in
           db.execute("SELECT srs_id FROM gpkg_spatial_ref_sys")}
    assert {-1, 0, 4326} <= ids


def test_foreign_keys_enabled(db):
    assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_reopening_preserves_container(tmp_path):
    path = str(tmp_path / "again.gpkg")
    conn = gpkg.connect(path)
    conn.execute("CREATE TABLE payload (id INTEGER PRIMARY KEY)")
    gpkg.register_table(conn, "payload")
    conn.commit()
    conn.close()

    conn = gpkg.connect(path)
    assert gpkg.registered_tables(conn) == ["payload"]
    conn.close()


def test_is_geopackage_rejects_plain_sqlite(tmp_path):
    path = str(tmp_path / "plain.sqlite")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (a INTEGER)")
    conn.commit()
    conn.close()
    assert not gpkg.is_geopackage(path)


def test_is_geopackage_rejects_non_sqlite_and_missing(tmp_path):
    junk = tmp_path / "junk.gpkg"
    junk.write_bytes(b"not a database at all, not even close")
    assert not gpkg.is_geopackage(str(junk))
    assert not gpkg.is_geopackage(str(tmp_path / "nothing-here.gpkg"))


def test_register_table_announces_attributes_content(db):
    db.execute("CREATE TABLE detections (detection_id INTEGER PRIMARY KEY)")
    gpkg.register_table(db, "detections", description="pixel-space boxes")

    row = db.execute(
        "SELECT data_type, identifier, description, srs_id "
        "FROM gpkg_contents WHERE table_name = 'detections'").fetchone()
    assert row["data_type"] == gpkg.DATA_TYPE_ATTRIBUTES
    assert row["identifier"] == "detections"
    assert row["description"] == "pixel-space boxes"
    assert row["srs_id"] == 0


def test_register_table_sets_last_change(db):
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
    gpkg.register_table(db, "t")
    row = db.execute(
        "SELECT last_change FROM gpkg_contents WHERE table_name = 't'").fetchone()
    assert row["last_change"]


def test_register_table_rejects_unknown_table(db):
    with pytest.raises(ValueError, match="No such table"):
        gpkg.register_table(db, "definitely_not_there")


def test_register_table_is_idempotent(db):
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
    gpkg.register_table(db, "t")
    gpkg.register_table(db, "t")
    assert gpkg.registered_tables(db) == ["t"]


def test_register_tables_registers_each(db):
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY)")
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY)")
    gpkg.register_tables(db, ["a", "b"])
    assert gpkg.registered_tables(db) == ["a", "b"]


def test_attach_enables_cross_file_joins(tmp_path):
    left = str(tmp_path / "left.gpkg")
    right = str(tmp_path / "right.gpkg")

    conn = gpkg.connect(right)
    conn.execute("CREATE TABLE members (detection_id INTEGER, track_id INTEGER)")
    conn.execute("INSERT INTO members VALUES (7, 3)")
    conn.commit()
    conn.close()

    conn = gpkg.connect(left)
    conn.execute("CREATE TABLE detections (detection_id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO detections VALUES (7)")
    gpkg.attach(conn, right, "trk")

    row = conn.execute(
        "SELECT d.detection_id, m.track_id FROM detections d "
        "JOIN trk.members m USING (detection_id)").fetchone()
    assert (row["detection_id"], row["track_id"]) == (7, 3)

    gpkg.detach(conn, "trk")
    conn.close()


def test_attach_rejects_invalid_alias(db, tmp_path):
    other = str(tmp_path / "other.gpkg")
    gpkg.connect(other).close()
    with pytest.raises(ValueError, match="alias"):
        gpkg.attach(db, other, "not a valid alias; DROP TABLE x")


def test_detach_rejects_invalid_alias(db):
    with pytest.raises(ValueError, match="alias"):
        gpkg.detach(db, "bad alias")


def test_vacuum_runs_after_deletes(db):
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, blob TEXT)")
    db.executemany("INSERT INTO t (blob) VALUES (?)",
                   [("x" * 500,) for _ in range(200)])
    db.commit()
    db.execute("DELETE FROM t")
    gpkg.vacuum(db)
    assert db.execute("SELECT COUNT(*) AS n FROM t").fetchone()["n"] == 0


def test_geometry_columns_table_exists_and_is_empty(db):
    """GDAL refuses to open a GeoPackage without it, even with no geometry.

    Established by the Phase 0 spike in tests_qgis/test_store_layers.py; this
    guards the container against someone "tidying away" an unused table.
    """
    tables = {row["name"] for row in db.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "gpkg_geometry_columns" in tables
    assert db.execute(
        "SELECT COUNT(*) AS n FROM gpkg_geometry_columns").fetchone()["n"] == 0
