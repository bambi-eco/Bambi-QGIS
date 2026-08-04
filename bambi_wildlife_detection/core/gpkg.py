# -*- coding: utf-8 -*-
"""Minimal GeoPackage container support, stdlib ``sqlite3`` only.

The 6.0 exchange format stores every pipeline stage in its own SQLite file
(see EXCHANGE_FORMAT_PLAN.md §2). Those files hold plain typed columns and no
geometry, but they are still written as valid *GeoPackages* using the
**aspatial** profile so that QGIS's Browser panel opens them directly and the
attribute-table / "toggle editing → delete rows" workflow works without any
plugin code.

That only needs a small, well-specified amount of boilerplate:

* ``PRAGMA application_id`` = ``0x47504B47`` (``"GPKG"``) and
  ``PRAGMA user_version`` = ``10300`` (GeoPackage 1.3.0),
* the ``gpkg_spatial_ref_sys`` table with its three mandatory rows,
* the ``gpkg_contents`` table, with one ``data_type='attributes'`` row per
  payload table,
* an **empty** ``gpkg_geometry_columns`` table.

That last one is not optional in practice. The specification requires it only
for files containing feature tables, but GDAL's GPKG driver (verified against
3.4.1, the version in the QGIS test image) refuses to open a file without it —
``ogr.Open()`` returns ``None`` and every layer is invalid. Creating it empty
costs nothing and is what makes the aspatial layout work at all; see
``tests_qgis/test_store_layers.py``, which is the spike that established this.

No GeoPackage geometry is ever encoded — QGIS map layers keep being built in
code from the tables, as they were from the text files before.

References: OGC GeoPackage 1.3.0, clauses 1.1.1.1 (core tables) and 2.4
(attributes data type).
"""

import os
import sqlite3
from typing import Iterable, Optional

#: ``"GPKG"`` interpreted as a big-endian 4-byte integer.
APPLICATION_ID = 0x47504B47

#: GeoPackage 1.3.0, encoded as ``10000 * major + 100 * minor + patch``.
USER_VERSION = 10300

#: ``data_type`` of a non-spatial payload table in ``gpkg_contents``.
DATA_TYPE_ATTRIBUTES = "attributes"

_SRS_TABLE = """
CREATE TABLE IF NOT EXISTS gpkg_spatial_ref_sys (
    srs_name                 TEXT NOT NULL,
    srs_id                   INTEGER NOT NULL PRIMARY KEY,
    organization             TEXT NOT NULL,
    organization_coordsys_id INTEGER NOT NULL,
    definition               TEXT NOT NULL,
    description              TEXT
)
"""

_CONTENTS_TABLE = """
CREATE TABLE IF NOT EXISTS gpkg_contents (
    table_name  TEXT NOT NULL PRIMARY KEY,
    data_type   TEXT NOT NULL,
    identifier  TEXT UNIQUE,
    description TEXT DEFAULT '',
    last_change DATETIME NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    min_x       DOUBLE,
    min_y       DOUBLE,
    max_x       DOUBLE,
    max_y       DOUBLE,
    srs_id      INTEGER,
    CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id)
        REFERENCES gpkg_spatial_ref_sys(srs_id)
)
"""

# The three rows every GeoPackage must carry. -1/0 are the "undefined" SRSes
# used by aspatial content; 4326 is required to be present even when unused.
_WGS84_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
    'AUTHORITY["EPSG","6326"]],'
    'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AUTHORITY["EPSG","4326"]]'
)

# Required by GDAL even with no geometry anywhere in the file — without it the
# GPKG driver declines to open the database at all. Always stays empty here.
_GEOMETRY_COLUMNS_TABLE = """
CREATE TABLE IF NOT EXISTS gpkg_geometry_columns (
    table_name         TEXT NOT NULL,
    column_name        TEXT NOT NULL,
    geometry_type_name TEXT NOT NULL,
    srs_id             INTEGER NOT NULL,
    z                  TINYINT NOT NULL,
    m                  TINYINT NOT NULL,
    CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),
    CONSTRAINT uk_gc_table_name UNIQUE (table_name),
    CONSTRAINT fk_gc_tn FOREIGN KEY (table_name)
        REFERENCES gpkg_contents(table_name),
    CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id)
        REFERENCES gpkg_spatial_ref_sys(srs_id)
)
"""

_MANDATORY_SRS = (
    ("WGS 84 geodetic", 4326, "EPSG", 4326, _WGS84_WKT,
     "longitude/latitude coordinates in decimal degrees on the WGS 84 spheroid"),
    ("Undefined cartesian SRS", -1, "NONE", -1, "undefined",
     "undefined cartesian coordinate reference system"),
    ("Undefined geographic SRS", 0, "NONE", 0, "undefined",
     "undefined geographic coordinate reference system"),
)


def is_geopackage(path: str) -> bool:
    """True when *path* is an SQLite file carrying the GeoPackage magic.

    Reads the ``application_id`` from the SQLite header (bytes 68..71, big
    endian) without opening a connection, so a corrupt or foreign file is
    rejected cheaply rather than raising somewhere deeper.
    """
    try:
        with open(path, "rb") as fh:
            header = fh.read(72)
    except OSError:
        return False
    if len(header) < 72 or not header.startswith(b"SQLite format 3\x00"):
        return False
    return int.from_bytes(header[68:72], "big") == APPLICATION_ID


def connect(path: str, create: bool = True) -> sqlite3.Connection:
    """Open *path* as a GeoPackage, creating the container when missing.

    The connection has foreign keys enabled and rows returned as
    :class:`sqlite3.Row`.

    Journal mode is left at SQLite's default (``DELETE``) rather than WAL. WAL
    would allow a reader to run alongside the writer, but the plugin never lets
    QGIS hold a stage file open (EXCHANGE_FORMAT_PLAN.md §11), so there is no
    concurrency to buy — and WAL costs two things that matter here: it needs
    shared memory, so it fails outright on network shares where survey projects
    plausibly live, and it leaves ``-wal``/``-shm`` sidecars that a user copying
    "the .gpkg" would silently leave behind.

    :param create: when False, a missing file raises :class:`FileNotFoundError`
        instead of being created.
    """
    exists = os.path.isfile(path)
    if not exists and not create:
        raise FileNotFoundError(path)

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if not exists:
        _init_container(conn)
    return conn


def _init_container(conn: sqlite3.Connection) -> None:
    """Write the GeoPackage boilerplate into a freshly created database."""
    # Set once, at creation: `journal_mode` briefly needs an exclusive lock, so
    # issuing it on every open would fail whenever anything else holds the file.
    conn.execute("PRAGMA journal_mode = DELETE")
    # PRAGMA values are integers, not user input; sqlite3 does not accept
    # placeholders in a PRAGMA statement.
    conn.execute("PRAGMA application_id = {:d}".format(APPLICATION_ID))
    conn.execute("PRAGMA user_version = {:d}".format(USER_VERSION))
    conn.execute(_SRS_TABLE)
    conn.execute(_CONTENTS_TABLE)
    conn.execute(_GEOMETRY_COLUMNS_TABLE)
    conn.executemany(
        "INSERT OR IGNORE INTO gpkg_spatial_ref_sys "
        "(srs_name, srs_id, organization, organization_coordsys_id, "
        " definition, description) VALUES (?, ?, ?, ?, ?, ?)",
        _MANDATORY_SRS)
    conn.commit()


def register_table(conn: sqlite3.Connection, table: str,
                   identifier: Optional[str] = None,
                   description: str = "") -> None:
    """Announce *table* to QGIS as non-spatial GeoPackage content.

    Without a ``gpkg_contents`` row the table exists but is invisible to the
    GeoPackage driver, which is the whole reason for using this container.

    *table* is a schema-controlled identifier, never user input — it is
    validated against the database's own table list rather than interpolated
    blindly.
    """
    known = {row["name"] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'")}
    if table not in known:
        raise ValueError(f"No such table to register: {table!r}")

    conn.execute(
        "INSERT OR REPLACE INTO gpkg_contents "
        "(table_name, data_type, identifier, description, srs_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (table, DATA_TYPE_ATTRIBUTES, identifier or table, description, 0))


def register_tables(conn: sqlite3.Connection, tables: Iterable[str]) -> None:
    """Register every table in *tables* (see :func:`register_table`)."""
    for table in tables:
        register_table(conn, table)


def registered_tables(conn: sqlite3.Connection) -> list:
    """The payload tables currently announced in ``gpkg_contents``."""
    return [row["table_name"] for row in conn.execute(
        "SELECT table_name FROM gpkg_contents ORDER BY table_name")]


def attach(conn: sqlite3.Connection, path: str, alias: str) -> None:
    """``ATTACH`` another stage file under *alias* for cross-stage queries.

    Foreign keys are not enforced across attached databases; what the joins
    rely on is stable ids, not enforcement (EXCHANGE_FORMAT_PLAN.md §2).
    """
    if not alias.isidentifier():
        raise ValueError(f"Invalid attach alias: {alias!r}")
    # sqlite3 accepts a placeholder for the path but not for the alias, which
    # is why the alias is validated above instead of being bound.
    conn.execute(f"ATTACH DATABASE ? AS {alias}", (path,))


def detach(conn: sqlite3.Connection, alias: str) -> None:
    """Undo :func:`attach`.

    Commits first: SQLite refuses to detach a database while a transaction is
    open, and any preceding read leaves one open under the default isolation
    level. Callers that need the write to stay pending should detach before
    starting it rather than the other way round.
    """
    if not alias.isidentifier():
        raise ValueError(f"Invalid attach alias: {alias!r}")
    conn.commit()
    conn.execute(f"DETACH DATABASE {alias}")


def vacuum(conn: sqlite3.Connection) -> None:
    """Reclaim space after a stage reset.

    ``DELETE`` alone does not shrink the file, and a project folder that only
    ever grows is a confusing thing to debug (EXCHANGE_FORMAT_PLAN.md §11).
    """
    conn.commit()
    conn.execute("VACUUM")
