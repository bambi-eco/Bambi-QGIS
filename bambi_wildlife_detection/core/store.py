# -*- coding: utf-8 -*-
"""The 6.0 pipeline store: schema, creation and invariant enforcement.

Implements EXCHANGE_FORMAT_PLAN.md §3–§5. Every pipeline stage owns one
GeoPackage file (see :mod:`core.gpkg` for the container), plus one
project-level file holding the vocabulary shared by all of them::

    <target_folder>/
        project.gpkg              # species, enums, custom fields, stage state
        bambi_t/                  # thermal — one file per stage
            detections.gpkg
            georeferenced.gpkg
            tracks.gpkg
            fov.gpkg
            labels.gpkg
            segmentation.gpkg
        bambi_w/                  # RGB — same set

The vocabulary lives in ``project.gpkg`` rather than in the detections stage
because it must outlive any single stage: ``rm bambi_t/detections.gpkg`` is a
supported reset (§7), and it must not take the project's species list with it.
It is also modality-independent — thermal and RGB share one species list, which
is what the cross-modality label copy assumes.

Nothing in this module knows about QGIS, and ``sqlite3`` is stdlib, so the whole
store layer is reachable from the unit suite.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

from . import gpkg

#: Bumped on every schema change. Readers refuse anything newer; older files
#: are upgraded in place by :func:`_upgrade`, which may only *add* things.
SCHEMA_VERSION = 3

#: Store kinds; each maps to one file and one DDL set.
PROJECT = "project"
DETECTIONS = "detections"
GEOREFERENCED = "georeferenced"
TRACKS = "tracks"
FOV = "fov"
LABELS = "labels"
SEGMENTATION = "segmentation"

STAGE_KINDS = (DETECTIONS, GEOREFERENCED, TRACKS, FOV, LABELS, SEGMENTATION)

#: Modality suffixes, matching the ``_t`` / ``_w`` convention of 5.x.
MODALITIES = ("t", "w")

# ---------------------------------------------------------------------------
# Base vocabulary
# ---------------------------------------------------------------------------

#: The three protected base classes present in every project (§3.1).
#: ``animal`` keeps id 0 because that is what the detector emits and what every
#: unresolved detection falls back to; the other two are negative so they can
#: never collide with a legacy class id (0–9 built-ins, 10+ custom species).
BASE_SPECIES = (
    (0, "animal"),
    (-1, "unknown"),
    (-2, "not-an-animal"),
)

#: Species id every unresolved detection falls back to.
FALLBACK_SPECIES_ID = 0

#: ``species_id`` of the "this is not an animal" base class. Excluded from every
#: survey analytic (§8.2) — a labelled false positive biases a density estimate.
NOT_AN_ANIMAL_SPECIES_ID = -2

#: Concrete species seeded from the 5.x taxonomy at their existing indices, so
#: class ids written by 5.x keep their meaning. Index 0 ("unknown") is *not*
#: here: it becomes the base class ``animal``, see §9.
SEEDED_SPECIES = (
    (1, "roe deer"), (2, "red deer"), (3, "fallow deer"), (4, "wild boar"),
    (5, "chamois"), (6, "fox"), (7, "hare"), (8, "bird"), (9, "other"),
)

#: Scientific names for the seeded taxonomy, as ``(scientific_name, rank)``.
#: These are *editable defaults*, not assertions: "fox" and "hare" assume the
#: Central-European species the plugin is used for, and "bird" is a class
#: rather than a species. GBIF taxon keys are deliberately **not** seeded —
#: an invented identifier would publish confidently wrong data, so they are
#: left for the user to fill in from gbif.org.
SEEDED_TAXONOMY = {
    "roe deer": ("Capreolus capreolus", "species"),
    "red deer": ("Cervus elaphus", "species"),
    "fallow deer": ("Dama dama", "species"),
    "wild boar": ("Sus scrofa", "species"),
    "chamois": ("Rupicapra rupicapra", "species"),
    "fox": ("Vulpes vulpes", "species"),
    "hare": ("Lepus europaeus", "species"),
    "bird": ("Aves", "class"),
}

#: Enums seeded from the 5.x hardcoded taxonomies, with the fields that use
#: them: ``(enum_name, values, field_name, field_scope)``. These are ordinary
#: rows — they are the worked examples a user copies when defining their own.
SEEDED_ENUMS = (
    ("sex", ("unknown", "female", "male"), "sex", "track"),
    ("age", ("unknown", "adult", "juvenile"), "age", "track"),
    ("occlusion", ("none", "partially", "fully"), "occlusion", "detection"),
)

FIELD_TYPES = ("int", "float", "string", "bool", "datetime", "enum")
FIELD_SCOPES = ("detection", "track", "frame", "project")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_META_DDL = """
CREATE TABLE IF NOT EXISTS bambi_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

_PROJECT_DDL = """
CREATE TABLE species (
    species_id      INTEGER PRIMARY KEY,
    name            TEXT UNIQUE NOT NULL,
    protected       INTEGER NOT NULL DEFAULT 0,
    -- Taxonomy, for publishing (§8.1). `name` is the label the user works with
    -- and is a *vernacular* name — Darwin Core needs the scientific one, and a
    -- GBIF taxon key lets the backbone resolve it exactly instead of matching
    -- a string. Both optional: a project that never publishes needs neither.
    scientific_name TEXT,
    taxon_rank      TEXT,
    gbif_taxon_key  INTEGER
);

CREATE TABLE enums (
    enum_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT UNIQUE NOT NULL,
    protected  INTEGER NOT NULL DEFAULT 0,
    -- Highest value_id ever handed out, so deleting the last value does not
    -- let its id come back meaning something else (§4.1).
    high_water INTEGER NOT NULL DEFAULT -1
);

CREATE TABLE enum_values (
    enum_id  INTEGER NOT NULL REFERENCES enums(enum_id),
    value_id INTEGER NOT NULL,
    label    TEXT NOT NULL,
    ordinal  INTEGER NOT NULL,
    PRIMARY KEY (enum_id, value_id)
);
CREATE UNIQUE INDEX ux_enum_label ON enum_values(enum_id, label);

CREATE TABLE field_schema (
    name      TEXT NOT NULL,
    type      TEXT NOT NULL,
    scope     TEXT NOT NULL,
    enum_id   INTEGER REFERENCES enums(enum_id),
    protected INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (name, scope),
    CHECK (type <> 'enum' OR enum_id IS NOT NULL)
);

CREATE TABLE detection_sources (
    source_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    model       TEXT,
    version     TEXT,
    -- 0 = never run; the first recorded run makes it 1. Bumped per source, so
    -- one producer re-running leaves the others' generations alone (§4.1).
    generation  INTEGER NOT NULL DEFAULT 0,
    config_hash TEXT,
    created_at  TEXT
);

CREATE TABLE class_mapping (
    source_id    INTEGER NOT NULL REFERENCES detection_sources(source_id),
    source_class TEXT NOT NULL,
    species_id   INTEGER NOT NULL DEFAULT 0 REFERENCES species(species_id),
    PRIMARY KEY (source_id, source_class)
);

-- The form values that produced this flight's outputs (§10.2). Keeping them
-- beside the outputs makes the folder self-describing: hand someone the folder
-- and the settings that produced it come with it.
CREATE TABLE config (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE stages (
    stage             TEXT NOT NULL,
    modality          TEXT NOT NULL,
    state             TEXT NOT NULL,
    row_count         INTEGER,
    started_at        TEXT,
    finished_at       TEXT,
    generation        INTEGER,
    input_fingerprint TEXT,
    PRIMARY KEY (stage, modality)
);

-- Invariant (§4.1): the base classes are permanent. Enforced in the database
-- so it holds for every caller, including a user poking at the file by hand.
CREATE TRIGGER trg_species_no_delete_protected
BEFORE DELETE ON species FOR EACH ROW WHEN OLD.protected = 1
BEGIN
    SELECT RAISE(ABORT, 'protected species cannot be deleted');
END;

CREATE TRIGGER trg_species_no_rename_protected
BEFORE UPDATE OF name, species_id ON species FOR EACH ROW
WHEN OLD.protected = 1
BEGIN
    SELECT RAISE(ABORT, 'protected species cannot be renamed');
END;

-- Invariant (§4.1): species and enum ids are append-only; nothing renumbers.
CREATE TRIGGER trg_species_no_renumber
BEFORE UPDATE OF species_id ON species FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'species_id is append-only and never renumbered');
END;

CREATE TRIGGER trg_enum_value_no_renumber
BEFORE UPDATE OF value_id ON enum_values FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'value_id is append-only and never renumbered');
END;
"""

_DETECTIONS_DDL = """
CREATE TABLE detections (
    detection_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    frame          INTEGER NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    confidence     REAL,
    species_id     INTEGER NOT NULL DEFAULT 0,
    source_id      INTEGER NOT NULL,
    source_class   TEXT,
    label_track_id INTEGER,
    attributes     TEXT
);
CREATE INDEX ix_det_frame   ON detections(frame);
CREATE INDEX ix_det_source  ON detections(source_id);
CREATE INDEX ix_det_species ON detections(species_id);
-- The materialisation key of the labelling export (§6.2): re-exporting a label
-- track keeps the detection ids of unchanged (track, frame) pairs.
CREATE UNIQUE INDEX ux_det_label ON detections(label_track_id, frame)
    WHERE label_track_id IS NOT NULL;
"""

_GEOREFERENCED_DDL = """
CREATE TABLE detections_geo (
    detection_id INTEGER PRIMARY KEY,
    gx1 REAL, gy1 REAL, gz1 REAL,
    gx2 REAL, gy2 REAL, gz2 REAL
);

CREATE TABLE georef_failures (
    detection_id INTEGER PRIMARY KEY,
    reason       TEXT NOT NULL
);
"""

_TRACKS_DDL = """
CREATE TABLE track_runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    tracker     TEXT,
    version     TEXT,
    generation  INTEGER NOT NULL DEFAULT 1,
    config_hash TEXT,
    created_at  TEXT,
    is_active   INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE tracks (
    track_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER NOT NULL REFERENCES track_runs(run_id),
    species_id INTEGER NOT NULL DEFAULT 0,
    attributes TEXT
);
CREATE INDEX ix_tracks_run ON tracks(run_id);

CREATE TABLE track_members (
    track_id     INTEGER NOT NULL REFERENCES tracks(track_id),
    detection_id INTEGER NOT NULL,
    interpolated INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (track_id, detection_id)
);
CREATE INDEX ix_tm_det ON track_members(detection_id);
"""

_FOV_DDL = """
CREATE TABLE fov_polygons (
    frame    INTEGER PRIMARY KEY,
    n_points INTEGER NOT NULL
);

CREATE TABLE fov_vertices (
    frame INTEGER NOT NULL,
    seq   INTEGER NOT NULL,
    x REAL, y REAL, z REAL,
    PRIMARY KEY (frame, seq)
);
"""

_LABELS_DDL = """
CREATE TABLE label_tracks (
    label_track_id  INTEGER PRIMARY KEY,
    species_id      INTEGER NOT NULL DEFAULT 0,
    origin_track_id INTEGER,
    track_id        INTEGER,
    attributes      TEXT
);

CREATE TABLE label_keyframes (
    label_track_id      INTEGER NOT NULL REFERENCES label_tracks(label_track_id),
    frame               INTEGER NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    stop                INTEGER NOT NULL DEFAULT 0,
    origin_detection_id INTEGER,
    attributes          TEXT,
    PRIMARY KEY (label_track_id, frame)
);

CREATE TABLE label_track_origin_xmodal (
    label_track_id     INTEGER PRIMARY KEY,
    source_modality    TEXT NOT NULL,
    source_label_track INTEGER NOT NULL
);
"""

_SEGMENTATION_DDL = """
CREATE TABLE segments (
    segment_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER,
    frame        INTEGER NOT NULL,
    polygon_px   TEXT,
    polygon_geo  TEXT,
    attributes   TEXT
);
CREATE INDEX ix_seg_frame ON segments(frame);
"""

_DDL: Dict[str, str] = {
    PROJECT: _PROJECT_DDL,
    DETECTIONS: _DETECTIONS_DDL,
    GEOREFERENCED: _GEOREFERENCED_DDL,
    TRACKS: _TRACKS_DDL,
    FOV: _FOV_DDL,
    LABELS: _LABELS_DDL,
    SEGMENTATION: _SEGMENTATION_DDL,
}


class StoreVersionError(RuntimeError):
    """Raised when a file was written by a newer plugin than this one.

    Mirrors the behaviour of ``core.labelling.read_custom_fields``: refuse
    with an actionable message rather than misreading the data.
    """


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def project_path(target_folder: str) -> str:
    """Path of the project-level store."""
    return os.path.join(target_folder, "project.gpkg")


def stage_path(target_folder: str, kind: str, modality: str) -> str:
    """Path of one stage file, e.g. ``<target>/bambi_t/tracks.gpkg``."""
    if kind not in STAGE_KINDS:
        raise ValueError(f"Unknown stage kind: {kind!r}")
    if modality not in MODALITIES:
        raise ValueError(f"Unknown modality: {modality!r}")
    return os.path.join(target_folder, f"bambi_{modality}", f"{kind}.gpkg")


# ---------------------------------------------------------------------------
# Open / create
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute(
        "SELECT value FROM bambi_meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else row["value"]


def set_meta(conn: sqlite3.Connection, key: str, value) -> None:
    """Write one ``bambi_meta`` entry."""
    conn.execute(
        "INSERT OR REPLACE INTO bambi_meta (key, value) VALUES (?, ?)",
        (key, None if value is None else str(value)))


def get_meta(conn: sqlite3.Connection, key: str,
             default: Optional[str] = None) -> Optional[str]:
    """Read one ``bambi_meta`` entry."""
    value = _read_meta(conn, key)
    return default if value is None else value


def open_store(path: str, kind: str, modality: str = "",
               create: bool = True) -> sqlite3.Connection:
    """Open (or create) the store of *kind* at *path*.

    Refuses files written by a newer schema version, and files whose recorded
    kind does not match — opening ``tracks.gpkg`` as a detections store is a
    programming error worth catching immediately, not a silent empty result.
    """
    if kind not in _DDL:
        raise ValueError(f"Unknown store kind: {kind!r}")

    existed = os.path.isfile(path)
    conn = gpkg.connect(path, create=create)
    conn.executescript(_META_DDL)

    if not existed:
        conn.executescript(_DDL[kind])
        set_meta(conn, "schema_version", SCHEMA_VERSION)
        set_meta(conn, "kind", kind)
        set_meta(conn, "modality", modality)
        set_meta(conn, "created_at", _utc_now())
        gpkg.register_tables(conn, _payload_tables(conn))
        if kind == PROJECT:
            seed_vocabulary(conn)
        conn.commit()
        return conn

    _check_compatible(conn, path, kind)
    _upgrade(conn, kind)
    return conn


#: Additive-only upgrades, keyed by the version they produce. Each entry is a
#: list of statements applied to a store of the previous version. They may add
#: tables, columns or indexes; they may never drop or rewrite anything, because
#: an older plugin must still be able to read the file afterwards.
_UPGRADES = {
    3: {
        PROJECT: [
            "CREATE TABLE IF NOT EXISTS config "
            "(key TEXT PRIMARY KEY, value TEXT)",
        ],
    },
    2: {
        PROJECT: [
            "ALTER TABLE species ADD COLUMN scientific_name TEXT",
            "ALTER TABLE species ADD COLUMN taxon_rank TEXT",
            "ALTER TABLE species ADD COLUMN gbif_taxon_key INTEGER",
        ],
    },
}


def _upgrade(conn: sqlite3.Connection, kind: str) -> None:
    """Bring an older store up to :data:`SCHEMA_VERSION` in place."""
    raw = _read_meta(conn, "schema_version")
    try:
        version = int(raw)
    except (TypeError, ValueError):
        return
    if version >= SCHEMA_VERSION:
        return

    with transaction(conn):
        for target in range(version + 1, SCHEMA_VERSION + 1):
            for statement in _UPGRADES.get(target, {}).get(kind, []):
                conn.execute(statement)
        set_meta(conn, "schema_version", SCHEMA_VERSION)
        if kind == PROJECT:
            seed_vocabulary(conn)


def _check_compatible(conn: sqlite3.Connection, path: str, kind: str) -> None:
    raw = _read_meta(conn, "schema_version")
    try:
        version = int(raw)
    except (TypeError, ValueError):
        conn.close()
        raise StoreVersionError(
            f"{path} is not a BAMBI store (no readable schema version).")
    if version > SCHEMA_VERSION:
        conn.close()
        raise StoreVersionError(
            f"{path} was written by a newer version of the plugin "
            f"(schema {version}, this one reads up to {SCHEMA_VERSION}). "
            "Please update the BAMBI plugin.")

    stored_kind = _read_meta(conn, "kind")
    if stored_kind != kind:
        conn.close()
        raise StoreVersionError(
            f"{path} holds a {stored_kind!r} store, not {kind!r}.")


def _payload_tables(conn: sqlite3.Connection) -> List[str]:
    """Payload tables of the connection — everything but GeoPackage internals."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name NOT LIKE 'gpkg_%' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name")
    return [row["name"] for row in rows]


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Run a block as one transaction; a cancelled stage leaves no half state."""
    try:
        yield conn
    except BaseException:
        conn.rollback()
        raise
    conn.commit()


# ---------------------------------------------------------------------------
# Vocabulary seeding
# ---------------------------------------------------------------------------

def seed_vocabulary(conn: sqlite3.Connection) -> None:
    """Install the base species, the seeded enums and their fields (§3.1, §5.2).

    Idempotent: re-running leaves an existing vocabulary untouched, so opening
    a project store never disturbs a user's taxonomy.
    """
    conn.executemany(
        "INSERT OR IGNORE INTO species (species_id, name, protected) "
        "VALUES (?, ?, 1)", BASE_SPECIES)
    conn.executemany(
        "INSERT OR IGNORE INTO species (species_id, name, protected) "
        "VALUES (?, ?, 0)", SEEDED_SPECIES)
    # Only fill taxonomy that is still blank, so a user's corrections survive.
    for name, (scientific_name, rank) in SEEDED_TAXONOMY.items():
        conn.execute(
            "UPDATE species SET scientific_name = ?, taxon_rank = ? "
            "WHERE name = ? AND scientific_name IS NULL",
            (scientific_name, rank, name))

    for enum_name, values, field_name, field_scope in SEEDED_ENUMS:
        conn.execute(
            "INSERT OR IGNORE INTO enums (name, protected) VALUES (?, 1)",
            (enum_name,))
        row = conn.execute(
            "SELECT enum_id FROM enums WHERE name = ?", (enum_name,)).fetchone()
        enum_id = row["enum_id"]
        conn.executemany(
            "INSERT OR IGNORE INTO enum_values (enum_id, value_id, label, ordinal) "
            "VALUES (?, ?, ?, ?)",
            [(enum_id, i, label, i) for i, label in enumerate(values)])
        conn.execute(
            "INSERT OR IGNORE INTO field_schema "
            "(name, type, scope, enum_id, protected) VALUES (?, 'enum', ?, ?, 1)",
            (field_name, field_scope, enum_id))


#: Highest species id the seeded taxonomy occupies; custom species start above.
_SPECIES_BASE_HIGH_WATER = max(species_id for species_id, _ in SEEDED_SPECIES)


def _species_high_water(conn: sqlite3.Connection) -> int:
    stored = get_meta(conn, "species_high_water")
    high = _SPECIES_BASE_HIGH_WATER if stored is None else int(stored)
    row = conn.execute(
        "SELECT MAX(species_id) AS m FROM species WHERE species_id >= 0"
    ).fetchone()
    if row["m"] is not None:
        high = max(high, int(row["m"]))
    return high


def next_species_id(conn: sqlite3.Connection) -> int:
    """The id a newly added species *would* get. Does not reserve it."""
    return _species_high_water(conn) + 1


def reserve_species_id(conn: sqlite3.Connection) -> int:
    """Take the next species id, recording it so it is never handed out twice.

    A plain ``MAX(species_id) + 1`` would reissue the id of a deleted species
    whenever it happened to be the highest — and a stale reference to it would
    then silently mean a different animal. The high-water mark is what makes
    the ids append-only in the sense §4.1 claims.
    """
    species_id = next_species_id(conn)
    set_meta(conn, "species_high_water", species_id)
    return species_id


def note_species_id(conn: sqlite3.Connection, species_id: int) -> None:
    """Record an externally chosen species id against the high-water mark.

    Used by the migration, which assigns the ids 5.x would have derived.
    """
    if species_id > _species_high_water(conn):
        set_meta(conn, "species_high_water", species_id)


def _enum_high_water(conn: sqlite3.Connection, enum_id: int) -> int:
    row = conn.execute(
        "SELECT high_water FROM enums WHERE enum_id = ?", (enum_id,)).fetchone()
    high = -1 if row is None else int(row["high_water"])
    values = conn.execute(
        "SELECT MAX(value_id) AS m FROM enum_values WHERE enum_id = ?",
        (enum_id,)).fetchone()
    if values["m"] is not None:
        high = max(high, int(values["m"]))
    return high


def next_enum_value_id(conn: sqlite3.Connection, enum_id: int) -> int:
    """The id a newly added enum value *would* get. Does not reserve it."""
    return _enum_high_water(conn, enum_id) + 1


def reserve_enum_value_id(conn: sqlite3.Connection, enum_id: int) -> int:
    """Take the next enum value id (append-only, like :func:`reserve_species_id`)."""
    value_id = next_enum_value_id(conn, enum_id)
    conn.execute("UPDATE enums SET high_water = ? WHERE enum_id = ?",
                 (value_id, enum_id))
    return value_id


def note_enum_value_id(conn: sqlite3.Connection, enum_id: int,
                       value_id: int) -> None:
    """Record an externally chosen enum value id against the high-water mark."""
    if value_id > _enum_high_water(conn, enum_id):
        conn.execute("UPDATE enums SET high_water = ? WHERE enum_id = ?",
                     (value_id, enum_id))


def resolve_species(conn: sqlite3.Connection, source_id: int,
                    source_class: Optional[str]) -> int:
    """Map a producer's raw class onto a species id.

    Anything unmapped — including a species-agnostic detector, which simply has
    no mapping rows — resolves to :data:`FALLBACK_SPECIES_ID` (``animal``).
    """
    if source_class is None:
        return FALLBACK_SPECIES_ID
    row = conn.execute(
        "SELECT species_id FROM class_mapping "
        "WHERE source_id = ? AND source_class = ?",
        (source_id, str(source_class))).fetchone()
    return FALLBACK_SPECIES_ID if row is None else int(row["species_id"])
