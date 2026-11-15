# -*- coding: utf-8 -*-
"""Editing the project vocabulary: species, enums and custom fields.

Implements EXCHANGE_FORMAT_PLAN.md §3.1 and §5.3. This is the *only* place the
vocabulary changes - confining creation to one editor is what actually fixes the
unstable-class-id bug (§1.3), because ``species_class_ids()`` can no longer mint
an id from an arbitrary string typed into a combo box.

The rules it enforces, all of which exist so that stored data cannot change
meaning behind the user's back:

* base species (``species_id <= 0``) are permanent - no rename, no delete;
* species and enum-value ids are append-only, so a deleted id is never reissued;
* nothing referenced may be deleted;
* renaming a species or an enum value touches no stored row, because rows hold
  ids rather than labels;
* renaming a *field* does rewrite stored rows, because ``attributes`` is keyed
  by field name - a deliberate trade for readable attribute blobs (§5.1).

**Edits are made on a private copy and written back in one go.** The editor
used to open the project store and keep its edits in an open transaction
until the dialog was closed - which holds a write lock on ``project.gpkg``
for as long as the dialog is open. A step that finished meanwhile could not
record its results ("database is locked"). And a field rename rewrote the
stored rows at once, in the other stores, before the vocabulary was saved:
cancelling the dialog rolled the name back but left the rows renamed - or,
for a field deleted with its values, gone.

:class:`SchemaEditor` therefore copies the store when it is opened and edits
the copy. :meth:`~SchemaEditor.commit` writes the vocabulary back in one
short transaction and only then touches stored rows; closing without it
changes nothing anywhere. If something else changed the vocabulary in the
meantime, the commit is refused rather than overwriting it.

Headless: the Qt dialog in ``bambi_schema_dialog.py`` is a thin shell over this.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from typing import Dict, List, Optional, Tuple

from . import gpkg, store

#: Names a custom field may not take - they are already columns or reserved
#: keys in the label model. Mirrors ``core.labelling.RESERVED_FIELD_NAMES``.
RESERVED_FIELD_NAMES = frozenset({
    "track_id", "species", "species_id", "keyframes", "attributes",
    "frame", "x1", "y1", "x2", "y2", "stop", "detection_id", "source_id",
    "source_class", "confidence", "label_track_id", "run_id", "interpolated",
})

#: Tables whose ``attributes`` column may hold custom-field values, per store
#: kind. Used for the "is this still referenced?" checks.
_ATTRIBUTE_TABLES = {
    store.DETECTIONS: ("detections",),
    store.TRACKS: ("tracks",),
    store.LABELS: ("label_tracks", "label_keyframes"),
    store.SEGMENTATION: ("segments",),
}

#: Where the values of a field of a given scope are stored. A field is
#: identified by name *and* scope - a track field and a detection field may
#: share a name - so renaming or deleting one must leave the other's values
#: alone. Scopes not listed here have no table of their own; they are looked
#: for everywhere, as every field used to be.
_SCOPE_TABLES = {
    "track": {store.TRACKS: ("tracks",), store.LABELS: ("label_tracks",)},
    "detection": {store.DETECTIONS: ("detections",),
                  store.LABELS: ("label_keyframes",),
                  store.SEGMENTATION: ("segments",)},
}

#: The tables and settings that make up the vocabulary - what the editor
#: changes, and so what a commit writes back. Children first: that is the
#: order they are emptied in.
_VOCABULARY_TABLES = ("class_mapping", "field_schema", "enum_values",
                      "enums", "species")
_VOCABULARY_KEYS = {
    "species": ("species_id",),
    "enums": ("enum_id",),
    "enum_values": ("enum_id", "value_id"),
    "field_schema": ("name", "scope"),
    "class_mapping": ("source_id", "source_class"),
}
_VOCABULARY_META = ("species_high_water", "species_colours")

#: Tables carrying a ``species_id``, per store kind.
_SPECIES_TABLES = {
    store.DETECTIONS: ("detections",),
    store.TRACKS: ("tracks",),
    store.LABELS: ("label_tracks",),
}


class SchemaError(ValueError):
    """A vocabulary edit that would break one of the rules above."""


_IN_USE = ("The project's files are in use - a step is probably writing its "
           "results - so the changes could not be saved ({}). Nothing was "
           "changed; try again in a moment.")


def _fingerprint(conn) -> str:
    """A digest of the vocabulary in *conn*: equal exactly when the species,
    enums, fields, class mappings and their settings are."""
    digest = hashlib.sha256()
    for table in sorted(_VOCABULARY_TABLES):
        columns = [row[1] for row in conn.execute(
            f"PRAGMA table_info({table})")]  # nosec B608 - fixed names
        rows = conn.execute(
            f"SELECT * FROM {table} "  # nosec B608 - fixed names
            f"ORDER BY {', '.join(columns)}").fetchall()
        digest.update(repr((table, [tuple(row) for row in rows])).encode())
    for key in _VOCABULARY_META:
        digest.update(repr((key, store.get_meta(conn, key))).encode())
    return digest.hexdigest()


class SchemaEditor:
    """Vocabulary of one project, with the stage files available for checks.

    Works on a private copy of the project store (see the module docstring):
    nothing is changed, and nothing is locked, until :meth:`commit`.
    Usable as a context manager; the copy is discarded on exit.
    """

    def __init__(self, target_folder: str):
        self.target_folder = target_folder
        self._path = store.project_path(target_folder)
        # Creates and seeds the store of a folder that has none yet.
        store.open_store(self._path, store.PROJECT).close()

        self._folder = tempfile.mkdtemp(prefix="bambi_schema_")
        self._work = os.path.join(self._folder, "project.gpkg")
        source = gpkg.connect(self._path, create=False)
        target = sqlite3.connect(self._work)
        try:
            source.backup(target)       # a consistent copy, even mid-run
        finally:
            target.close()
            source.close()
        self.conn = store.open_store(self._work, store.PROJECT)
        self._base = _fingerprint(self.conn)
        #: ``(scope, name now) -> name the stored rows carry``, for fields
        #: renamed since the copy was taken.
        self._stored_as: Dict[Tuple[str, str], str] = {}
        #: ``(scope, stored name)`` of fields deleted together with their
        #: values.
        self._dropped: List[Tuple[str, str]] = []

    def __enter__(self) -> "SchemaEditor":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        """Discard the working copy. Edits not committed are gone."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        shutil.rmtree(self._folder, ignore_errors=True)

    def commit(self) -> None:
        """Write the edits into the project - the vocabulary in one short
        transaction, then the stored rows of renamed and deleted fields.

        :raises SchemaError: when the project's vocabulary was changed by
            something else since this editor was opened, or a store is in
            use by a running step. Nothing has been changed then.
        """
        self.conn.commit()
        renames = [(scope, stored, name)
                   for (scope, name), stored in self._stored_as.items()
                   if stored != name]
        try:
            real = store.open_store(self._path, store.PROJECT)
        except sqlite3.OperationalError as exc:
            raise SchemaError(_IN_USE.format(exc)) from exc
        applied: List[Tuple[str, str, Optional[str]]] = []
        try:
            if _fingerprint(real) != self._base:
                raise SchemaError(
                    "The species, enums, fields or class mappings of this "
                    "project were changed by something else while they were "
                    "being edited here. Nothing was saved - open the editor "
                    "again to work on the current state.")
            try:
                for step in self._rename_steps(renames):
                    self._rewrite_attribute_key(*step)
                    applied.append(step)
                self._write_back(real)
            except sqlite3.OperationalError as exc:
                # Take the renames back, so that names and rows still agree.
                for scope, old, new in reversed(applied):
                    try:
                        self._rewrite_attribute_key(scope, new, old)
                    except sqlite3.OperationalError:  # nosec B110
                        pass
                raise SchemaError(_IN_USE.format(exc)) from exc
        finally:
            real.close()

        # Values of deleted fields go last: should a store be busy now, they
        # stay behind under a name no field has any more - which is harmless,
        # where dropping them before the field was gone would not be.
        for scope, stored in self._dropped:
            try:
                self._rewrite_attribute_key(scope, stored, None)
            except sqlite3.OperationalError:  # nosec B110
                pass
        self._stored_as.clear()
        self._dropped = []
        self._base = _fingerprint(self.conn)

    def _write_back(self, real) -> None:
        """Bring the vocabulary of *real* in line with the working copy's,
        row by row: what the copy no longer has is deleted, what it has new
        is inserted, and of a row both have only the columns that differ
        are written - the store's own guards (a base species cannot be
        renamed or deleted, ids are never renumbered) apply to exactly what
        was edited, and to nothing that was not."""
        with store.transaction(real):
            # A parent and its children change in one go; the references
            # only have to hold at the end.
            real.execute("PRAGMA defer_foreign_keys = ON")
            for table in _VOCABULARY_TABLES:
                self._sync_rows(real, table, deletions=True)
            for table in reversed(_VOCABULARY_TABLES):
                self._sync_rows(real, table, deletions=False)
            for key in _VOCABULARY_META:
                value = store.get_meta(self.conn, key)
                if value is None:
                    real.execute(
                        "DELETE FROM bambi_meta WHERE key = ?", (key,))
                elif value != store.get_meta(real, key):
                    store.set_meta(real, key, value)

    def _sync_rows(self, real, table: str, deletions: bool) -> None:
        keys = _VOCABULARY_KEYS[table]
        where = " AND ".join(f"{column} = ?" for column in keys)

        def rows(conn):
            found = conn.execute(
                f"SELECT * FROM {table}").fetchall()  # nosec B608 - fixed
            return {tuple(row[column] for column in keys): row
                    for row in found}

        have, want = rows(real), rows(self.conn)
        if deletions:
            for key in have.keys() - want.keys():
                real.execute(
                    f"DELETE FROM {table} WHERE {where}",  # nosec B608 - fixed
                    key)
            return
        for key, row in want.items():
            columns = row.keys()
            if key not in have:
                marks = ", ".join("?" for _ in columns)
                real.execute(
                    f"INSERT INTO {table} ({', '.join(columns)}) "  # nosec B608
                    f"VALUES ({marks})", tuple(row))
                continue
            changed = [column for column in columns
                       if column not in keys and row[column] != have[key][column]]
            if changed:
                assignments = ", ".join(f"{column} = ?" for column in changed)
                real.execute(
                    f"UPDATE {table} SET {assignments} "  # nosec B608 - fixed
                    f"WHERE {where}",
                    tuple(row[column] for column in changed) + key)

    # -- stage files -----------------------------------------------------

    def _stage_files(self, kind: str) -> List[str]:
        paths = []
        for modality in store.MODALITIES:
            path = store.stage_path(self.target_folder, kind, modality)
            if os.path.isfile(path):
                paths.append(path)
        return paths

    # ------------------------------------------------------------------
    # Species
    # ------------------------------------------------------------------

    def species(self) -> List[dict]:
        """Every species, base classes first, then concrete ones by id."""
        return [dict(row) for row in self.conn.execute(
            "SELECT species_id, name, protected, scientific_name, taxon_rank, "
            "gbif_taxon_key FROM species ORDER BY species_id")]

    def species_colours(self) -> Dict[int, str]:
        """``species_id -> colour`` of every species: the project's own
        choice where it made one, the default otherwise
        (:mod:`core.map_symbols`)."""
        from . import map_symbols

        overrides = map_symbols.stored_colours(self.conn)
        colours = {}
        for row in self.species():
            own = overrides.get(row["species_id"])
            colours[row["species_id"]] = (
                own or map_symbols.default_colour(row["name"]))
        return colours

    def has_own_colour(self, species_id: int) -> bool:
        """True when the project chose this species' colour itself."""
        from . import map_symbols
        return int(species_id) in map_symbols.stored_colours(self.conn)

    def set_species_colour(self, species_id: int,
                           colour: Optional[str]) -> None:
        """Give a species its map colour (``#rrggbb``); ``None`` goes back
        to the default. Any species, the base classes included - a colour
        changes how it looks, not what it means."""
        from . import map_symbols

        if self.conn.execute(
                "SELECT 1 FROM species WHERE species_id = ?",
                (species_id,)).fetchone() is None:
            raise SchemaError(f"No species with id {species_id}.")
        try:
            map_symbols.store_colour(self.conn, species_id, colour)
        except ValueError as exc:
            raise SchemaError(str(exc)) from exc

    def species_by_name(self, name: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT species_id, name, protected, scientific_name, taxon_rank, "
            "gbif_taxon_key FROM species WHERE name = ?",
            (name.strip(),)).fetchone()
        return None if row is None else dict(row)

    def set_taxonomy(self, species_id: int, scientific_name: str = "",
                     taxon_rank: str = "",
                     gbif_taxon_key: Optional[int] = None) -> None:
        """Attach publishing taxonomy to a species.

        Only these two formats care (§8.1): Darwin Core needs
        ``scientificName``, and a GBIF taxon key lets the backbone resolve the
        record exactly instead of matching a name string. A project that never
        publishes can leave both empty.

        Base classes are rejected - ``animal`` and ``unknown`` are deliberately
        *not* taxa, and giving them one would publish nonsense.
        """
        row = self.conn.execute(
            "SELECT protected FROM species WHERE species_id = ?",
            (species_id,)).fetchone()
        if row is None:
            raise SchemaError(f"No species with id {species_id}.")
        if row["protected"]:
            raise SchemaError(
                "The base classes are not taxa, so they cannot carry a "
                "scientific name or a GBIF key.")
        if gbif_taxon_key is not None:
            try:
                gbif_taxon_key = int(gbif_taxon_key)
            except (TypeError, ValueError):
                raise SchemaError(
                    "A GBIF taxon key is the number from the species' page on "
                    "gbif.org, e.g. 2440947.")
            if gbif_taxon_key <= 0:
                raise SchemaError("A GBIF taxon key must be positive.")

        self.conn.execute(
            "UPDATE species SET scientific_name = ?, taxon_rank = ?, "
            "gbif_taxon_key = ? WHERE species_id = ?",
            ((scientific_name or "").strip() or None,
             (taxon_rank or "").strip() or None,
             gbif_taxon_key, species_id))

    def add_species(self, name: str) -> int:
        """Add a concrete species, returning its new (never reused) id."""
        name = (name or "").strip()
        if not name:
            raise SchemaError("A species needs a name.")
        if self.species_by_name(name) is not None:
            raise SchemaError(f"A species called '{name}' already exists.")
        species_id = store.reserve_species_id(self.conn)
        self.conn.execute(
            "INSERT INTO species (species_id, name, protected) VALUES (?, ?, 0)",
            (species_id, name))
        return species_id

    def rename_species(self, species_id: int, name: str) -> None:
        """Rename a concrete species; its id - and every stored row - is kept."""
        name = (name or "").strip()
        if not name:
            raise SchemaError("A species needs a name.")
        row = self.conn.execute(
            "SELECT protected FROM species WHERE species_id = ?",
            (species_id,)).fetchone()
        if row is None:
            raise SchemaError(f"No species with id {species_id}.")
        if row["protected"]:
            raise SchemaError(
                "The base classes (animal, unknown, not-an-animal) cannot be "
                "renamed - other data and exports rely on them.")
        clash = self.species_by_name(name)
        if clash is not None and clash["species_id"] != species_id:
            raise SchemaError(f"A species called '{name}' already exists.")
        self.conn.execute(
            "UPDATE species SET name = ? WHERE species_id = ?",
            (name, species_id))

    def species_usage(self, species_id: int) -> Dict[str, int]:
        """Where *species_id* is still referenced, as ``label -> count``."""
        usage: Dict[str, int] = {}
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM class_mapping WHERE species_id = ?",
            (species_id,)).fetchone()
        if row["n"]:
            usage["class mappings"] = int(row["n"])

        for kind, tables in _SPECIES_TABLES.items():
            for path in self._stage_files(kind):
                conn = store.open_store(path, kind)
                try:
                    for table in tables:
                        found = conn.execute(
                            f"SELECT COUNT(*) AS n FROM {table} "  # nosec B608
                            "WHERE species_id = ?", (species_id,)).fetchone()
                        if found["n"]:
                            usage[table] = usage.get(table, 0) + int(found["n"])
                finally:
                    conn.close()
        return usage

    def delete_species(self, species_id: int) -> None:
        """Delete a concrete, unreferenced species."""
        row = self.conn.execute(
            "SELECT name, protected FROM species WHERE species_id = ?",
            (species_id,)).fetchone()
        if row is None:
            raise SchemaError(f"No species with id {species_id}.")
        if row["protected"]:
            raise SchemaError(
                "The base classes (animal, unknown, not-an-animal) are part of "
                "every project and cannot be deleted.")
        usage = self.species_usage(species_id)
        if usage:
            where = ", ".join(f"{n} in {table}" for table, n in sorted(usage.items()))
            raise SchemaError(
                f"'{row['name']}' is still in use ({where}). Reassign those "
                "first, or rename the species instead of deleting it.")
        self.conn.execute("DELETE FROM species WHERE species_id = ?", (species_id,))
        # Its colour goes with it: ids are never reused, but a stale entry
        # would still be exported with the project.
        self._drop_colour(species_id)

    def _drop_colour(self, species_id: int) -> None:
        from . import map_symbols
        if int(species_id) in map_symbols.stored_colours(self.conn):
            map_symbols.store_colour(self.conn, species_id, None)

    # ------------------------------------------------------------------
    # Enums
    # ------------------------------------------------------------------

    def enums(self) -> List[dict]:
        return [dict(row) for row in self.conn.execute(
            "SELECT enum_id, name, protected FROM enums ORDER BY name")]

    def enum_by_name(self, name: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT enum_id, name, protected FROM enums WHERE name = ?",
            (name.strip(),)).fetchone()
        return None if row is None else dict(row)

    def add_enum(self, name: str) -> int:
        name = (name or "").strip()
        if not name:
            raise SchemaError("An enum needs a name.")
        if self.enum_by_name(name) is not None:
            raise SchemaError(f"An enum called '{name}' already exists.")
        cur = self.conn.execute("INSERT INTO enums (name) VALUES (?)", (name,))
        return int(cur.lastrowid)

    def enum_values(self, enum_id: int) -> List[dict]:
        """Values in display order - ``ordinal``, not id."""
        return [dict(row) for row in self.conn.execute(
            "SELECT value_id, label, ordinal FROM enum_values "
            "WHERE enum_id = ? ORDER BY ordinal, value_id", (enum_id,))]

    def add_enum_value(self, enum_id: int, label: str) -> int:
        """Append a value. Its id is new even if an old one was deleted."""
        label = (label or "").strip()
        if not label:
            raise SchemaError("An enum value needs a label.")
        exists = self.conn.execute(
            "SELECT 1 FROM enum_values WHERE enum_id = ? AND label = ?",
            (enum_id, label)).fetchone()
        if exists is not None:
            raise SchemaError(f"'{label}' is already a value of this enum.")
        value_id = store.reserve_enum_value_id(self.conn, enum_id)
        ordinal = self.conn.execute(
            "SELECT IFNULL(MAX(ordinal), -1) + 1 AS o FROM enum_values "
            "WHERE enum_id = ?", (enum_id,)).fetchone()["o"]
        self.conn.execute(
            "INSERT INTO enum_values (enum_id, value_id, label, ordinal) "
            "VALUES (?, ?, ?, ?)", (enum_id, value_id, label, ordinal))
        return value_id

    def rename_enum_value(self, enum_id: int, value_id: int, label: str) -> None:
        """Rename a value. Stored rows hold the id, so none are touched."""
        label = (label or "").strip()
        if not label:
            raise SchemaError("An enum value needs a label.")
        clash = self.conn.execute(
            "SELECT value_id FROM enum_values WHERE enum_id = ? AND label = ?",
            (enum_id, label)).fetchone()
        if clash is not None and clash["value_id"] != value_id:
            raise SchemaError(f"'{label}' is already a value of this enum.")
        self.conn.execute(
            "UPDATE enum_values SET label = ? WHERE enum_id = ? AND value_id = ?",
            (label, enum_id, value_id))

    def reorder_enum_values(self, enum_id: int, value_ids: List[int]) -> None:
        """Set the display order. Ids are untouched - only ``ordinal`` moves."""
        known = {row["value_id"] for row in self.enum_values(enum_id)}
        if set(value_ids) != known:
            raise SchemaError(
                "Reordering must list exactly the enum's current values.")
        for ordinal, value_id in enumerate(value_ids):
            self.conn.execute(
                "UPDATE enum_values SET ordinal = ? "
                "WHERE enum_id = ? AND value_id = ?",
                (ordinal, enum_id, value_id))

    def enum_value_usage(self, enum_id: int, value_id: int) -> Dict[str, int]:
        """Rows still holding this value, per table."""
        usage: Dict[str, int] = {}
        for row in self.conn.execute(
                "SELECT name, scope FROM field_schema WHERE enum_id = ?",
                (enum_id,)).fetchall():
            # under the name the rows carry, in the tables of the field's
            # scope - a field of another scope and the same name is another
            # field, with an enum of its own
            stored = self._stored_as.get((row["scope"], row["name"]),
                                         row["name"])
            found = self._attribute_usage(
                lambda attributes, key=stored:
                    attributes.get(key) == value_id,
                [stored], row["scope"])
            for table, count in found.items():
                usage[table] = usage.get(table, 0) + count
        return usage

    def delete_enum_value(self, enum_id: int, value_id: int) -> None:
        row = self.conn.execute(
            "SELECT label FROM enum_values WHERE enum_id = ? AND value_id = ?",
            (enum_id, value_id)).fetchone()
        if row is None:
            raise SchemaError("No such enum value.")
        usage = self.enum_value_usage(enum_id, value_id)
        if usage:
            where = ", ".join(f"{n} in {table}" for table, n in sorted(usage.items()))
            raise SchemaError(
                f"'{row['label']}' is still in use ({where}). Reassign those "
                "rows first, or rename the value instead of deleting it.")
        self.conn.execute(
            "DELETE FROM enum_values WHERE enum_id = ? AND value_id = ?",
            (enum_id, value_id))

    # ------------------------------------------------------------------
    # Custom fields
    # ------------------------------------------------------------------

    def fields(self, scope: str = "") -> List[dict]:
        if scope:
            rows = self.conn.execute(
                "SELECT name, type, scope, enum_id, protected FROM field_schema "
                "WHERE scope = ? ORDER BY name", (scope,))
        else:
            rows = self.conn.execute(
                "SELECT name, type, scope, enum_id, protected FROM field_schema "
                "ORDER BY scope, name")
        return [dict(row) for row in rows]

    def add_field(self, name: str, type: str = "string", scope: str = "track",
                  enum_id: Optional[int] = None) -> None:
        name = (name or "").strip()
        if not name:
            raise SchemaError("A field needs a name.")
        if name.lower() in RESERVED_FIELD_NAMES:
            raise SchemaError(f"'{name}' is a reserved name.")
        if type not in store.FIELD_TYPES:
            raise SchemaError(f"Unknown field type '{type}'.")
        if scope not in store.FIELD_SCOPES:
            raise SchemaError(f"Unknown field scope '{scope}'.")
        if type == "enum" and enum_id is None:
            raise SchemaError("An enum field needs an enum to draw its values from.")
        if type != "enum":
            enum_id = None
        exists = self.conn.execute(
            "SELECT 1 FROM field_schema WHERE name = ? AND scope = ?",
            (name, scope)).fetchone()
        if exists is not None:
            raise SchemaError(f"A '{scope}' field called '{name}' already exists.")
        self.conn.execute(
            "INSERT INTO field_schema (name, type, scope, enum_id) "
            "VALUES (?, ?, ?, ?)", (name, type, scope, enum_id))

    def rename_field(self, name: str, scope: str, new_name: str) -> int:
        """Rename a field **and the stored rows that use it**.

        Unlike every other rename here this touches data, because ``attributes``
        is a JSON object keyed by field name (§5.1). The rows are rewritten
        when the edits are committed; the number that will be is returned.
        Only the tables of the field's own scope are touched - a detection
        field of the same name keeps its values.
        """
        new_name = (new_name or "").strip()
        if not new_name:
            raise SchemaError("A field needs a name.")
        if new_name.lower() in RESERVED_FIELD_NAMES:
            raise SchemaError(f"'{new_name}' is a reserved name.")
        row = self.conn.execute(
            "SELECT protected FROM field_schema WHERE name = ? AND scope = ?",
            (name, scope)).fetchone()
        if row is None:
            raise SchemaError(f"No '{scope}' field called '{name}'.")
        if row["protected"]:
            raise SchemaError(
                f"'{name}' is a built-in field and cannot be renamed.")
        clash = self.conn.execute(
            "SELECT 1 FROM field_schema WHERE name = ? AND scope = ?",
            (new_name, scope)).fetchone()
        if clash is not None:
            raise SchemaError(
                f"A '{scope}' field called '{new_name}' already exists.")

        stored = self._stored_as.pop((scope, name), name)
        self._stored_as[(scope, new_name)] = stored
        self.conn.execute(
            "UPDATE field_schema SET name = ? WHERE name = ? AND scope = ?",
            (new_name, name, scope))
        return sum(self._key_usage(scope, stored).values())

    def field_usage(self, name: str, scope: str = "") -> Dict[str, int]:
        """Rows holding a value of the field, per table. With *scope*, in
        the tables of that scope only - and under the name the rows still
        carry, for a field renamed in this session."""
        stored = self._stored_as.get((scope, name), name)
        return self._key_usage(scope, stored)

    def _key_usage(self, scope: str, stored: str) -> Dict[str, int]:
        return self._attribute_usage(
            lambda attributes: stored in attributes, [stored], scope)

    def delete_field(self, name: str, scope: str, force: bool = False) -> int:
        """Remove a field; with *force*, its stored values go too - when the
        edits are committed.

        Returns the number of rows whose attributes will be rewritten.
        """
        row = self.conn.execute(
            "SELECT protected FROM field_schema WHERE name = ? AND scope = ?",
            (name, scope)).fetchone()
        if row is None:
            raise SchemaError(f"No '{scope}' field called '{name}'.")
        if row["protected"]:
            raise SchemaError(
                f"'{name}' is a built-in field and cannot be deleted.")
        usage = self.field_usage(name, scope)
        if usage and not force:
            where = ", ".join(f"{n} in {table}" for table, n in sorted(usage.items()))
            raise SchemaError(
                f"'{name}' still holds values ({where}). Deleting it discards "
                "them; pass force=True to confirm.")
        stored = self._stored_as.pop((scope, name), name)
        if usage:
            self._dropped.append((scope, stored))
        self.conn.execute(
            "DELETE FROM field_schema WHERE name = ? AND scope = ?",
            (name, scope))
        return sum(usage.values())

    # ------------------------------------------------------------------
    # Attribute-blob helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tables_of(scope: str) -> Dict[str, tuple]:
        """``store kind -> tables`` holding the values of *scope* fields."""
        return _SCOPE_TABLES.get(scope, _ATTRIBUTE_TABLES)

    def _attribute_usage(self, predicate, names: List[str],
                         scope: str = "") -> Dict[str, int]:
        """Count rows whose ``attributes`` satisfy *predicate*.

        Rows are pre-filtered in SQL with a cheap ``LIKE`` on the field names and
        confirmed in Python, so this works on SQLite builds without JSON1.
        """
        usage: Dict[str, int] = {}
        for kind, tables in self._tables_of(scope).items():
            for path in self._stage_files(kind):
                conn = store.open_store(path, kind)
                try:
                    for table in tables:
                        count = 0
                        for row in self._candidate_rows(conn, table, names):
                            attributes = _load_attributes(row["attributes"])
                            if attributes and predicate(attributes):
                                count += 1
                        if count:
                            usage[table] = usage.get(table, 0) + count
                finally:
                    conn.close()
        return usage

    @staticmethod
    def _candidate_rows(conn, table: str, names: List[str]):
        clause = " OR ".join("attributes LIKE ?" for _ in names)
        patterns = [f'%"{name}"%' for name in names]
        return conn.execute(
            f"SELECT rowid AS _rid, attributes FROM {table} "  # nosec B608
            f"WHERE attributes IS NOT NULL AND ({clause})", patterns).fetchall()

    @staticmethod
    def _rename_steps(renames: List[Tuple[str, str, str]]
                      ) -> List[Tuple[str, str, str]]:
        """The key rewrites that carry out *renames* ``(scope, from, to)``.

        One rename is one rewrite. Several go through names nothing else
        uses, so that swapping two fields - A to B and B to A - does not pour
        both into one.
        """
        if len(renames) < 2:
            return list(renames)
        # plain characters only: the rows are searched as JSON text, where
        # anything else is written as an escape
        parked = [(scope, old, f"__bambi_renaming_{index}__")
                  for index, (scope, old, _new) in enumerate(renames)]
        return parked + [(scope, park, new) for (scope, _old, park),
                         (_s, _o, new) in zip(parked, renames)]

    def _rewrite_attribute_key(self, scope: str, name: str,
                               new_name: Optional[str]) -> int:
        """Rename (or drop, when *new_name* is None) a key in the attributes
        of every row of the tables *scope* fields are stored in."""
        rewritten = 0
        for kind, tables in self._tables_of(scope).items():
            for path in self._stage_files(kind):
                conn = store.open_store(path, kind)
                try:
                    with store.transaction(conn):
                        for table in tables:
                            for row in self._candidate_rows(conn, table, [name]):
                                attributes = _load_attributes(row["attributes"])
                                if not attributes or name not in attributes:
                                    continue
                                value = attributes.pop(name)
                                if new_name is not None:
                                    attributes[new_name] = value
                                conn.execute(
                                    f"UPDATE {table} SET attributes = ? "  # nosec B608
                                    "WHERE rowid = ?",
                                    (json.dumps(attributes), row["_rid"]))
                                rewritten += 1
                finally:
                    conn.close()
        return rewritten

    # ------------------------------------------------------------------
    # Detector class mapping
    # ------------------------------------------------------------------

    def sources(self) -> List[dict]:
        return [dict(row) for row in self.conn.execute(
            "SELECT source_id, kind, model, version, generation "
            "FROM detection_sources ORDER BY source_id")]

    def class_mapping(self, source_id: int) -> List[dict]:
        return [dict(row) for row in self.conn.execute(
            "SELECT source_class, species_id FROM class_mapping "
            "WHERE source_id = ? ORDER BY source_class", (source_id,))]

    def observed_source_classes(self, source_id: int) -> List[dict]:
        """Raw classes this producer actually emitted, with their counts.

        A mapping table built only from ``class_mapping`` shows nothing until
        someone has already guessed the vocabulary. Reading the classes back
        off the detections means the table lists what the model really
        produced, including the ones still resolving to ``animal``.
        """
        counts: Dict[str, int] = {}
        for path in self._stage_files(store.DETECTIONS):
            conn = store.open_store(path, store.DETECTIONS)
            try:
                for row in conn.execute(
                        "SELECT source_class, COUNT(*) AS n FROM detections "
                        "WHERE source_id = ? AND source_class IS NOT NULL "
                        "GROUP BY source_class", (source_id,)):
                    counts[row["source_class"]] = \
                        counts.get(row["source_class"], 0) + int(row["n"])
            finally:
                conn.close()

        mapped = {row["source_class"]: row["species_id"]
                  for row in self.class_mapping(source_id)}
        classes = sorted(set(counts) | set(mapped), key=_class_sort_key)
        return [{"source_class": name,
                 "species_id": mapped.get(name, store.FALLBACK_SPECIES_ID),
                 "mapped": name in mapped,
                 "detections": counts.get(name, 0)}
                for name in classes]

    def set_class_mapping(self, source_id: int, source_class: str,
                          species_id: int) -> None:
        """Point one of a producer's raw classes at a species.

        Changing this moves no boxes, so nothing upstream is invalidated; the
        detections' ``species_id`` is re-resolved from the raw ``source_class``
        they already carry (§3.1).
        """
        exists = self.conn.execute(
            "SELECT 1 FROM species WHERE species_id = ?", (species_id,)).fetchone()
        if exists is None:
            raise SchemaError(f"No species with id {species_id}.")
        self.conn.execute(
            "INSERT OR REPLACE INTO class_mapping "
            "(source_id, source_class, species_id) VALUES (?, ?, ?)",
            (source_id, str(source_class), species_id))

    def reapply_class_mapping(self, source_id: int) -> int:
        """Re-resolve stored detections of *source_id* from their raw class.

        This is what makes keeping ``source_class`` worthwhile: a corrected
        mapping is applied to existing rows instead of re-running the detector.
        Returns the number of detections whose species changed.
        """
        changed = 0
        for path in self._stage_files(store.DETECTIONS):
            conn = store.open_store(path, store.DETECTIONS)
            try:
                with store.transaction(conn):
                    rows = conn.execute(
                        "SELECT detection_id, source_class, species_id "
                        "FROM detections WHERE source_id = ?",
                        (source_id,)).fetchall()
                    for row in rows:
                        resolved = store.resolve_species(
                            self.conn, source_id, row["source_class"])
                        if resolved != row["species_id"]:
                            conn.execute(
                                "UPDATE detections SET species_id = ? "
                                "WHERE detection_id = ?",
                                (resolved, row["detection_id"]))
                            changed += 1
            finally:
                conn.close()
        return changed


def _class_sort_key(name: str):
    """Sort raw classes numerically where they are numbers, else by name.

    Detector classes are usually "0", "1", "2" …, which sort wrongly as text
    once there are ten of them.
    """
    try:
        return (0, int(name), "")
    except (TypeError, ValueError):
        return (1, 0, str(name))


def _load_attributes(raw) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}
