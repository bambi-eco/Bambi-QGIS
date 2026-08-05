# -*- coding: utf-8 -*-
"""Editing the project vocabulary: species, enums and custom fields.

Implements EXCHANGE_FORMAT_PLAN.md §3.1 and §5.3. This is the *only* place the
vocabulary changes — confining creation to one editor is what actually fixes the
unstable-class-id bug (§1.3), because ``species_class_ids()`` can no longer mint
an id from an arbitrary string typed into a combo box.

The rules it enforces, all of which exist so that stored data cannot change
meaning behind the user's back:

* base species (``species_id <= 0``) are permanent — no rename, no delete;
* species and enum-value ids are append-only, so a deleted id is never reissued;
* nothing referenced may be deleted;
* renaming a species or an enum value touches no stored row, because rows hold
  ids rather than labels;
* renaming a *field* does rewrite stored rows, because ``attributes`` is keyed
  by field name — a deliberate trade for readable attribute blobs (§5.1).

Headless: the Qt dialog in ``bambi_schema_dialog.py`` is a thin shell over this.
"""

import json
import os
from typing import Dict, List, Optional

from . import store

#: Names a custom field may not take — they are already columns or reserved
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

#: Tables carrying a ``species_id``, per store kind.
_SPECIES_TABLES = {
    store.DETECTIONS: ("detections",),
    store.TRACKS: ("tracks",),
    store.LABELS: ("label_tracks",),
}


class SchemaError(ValueError):
    """A vocabulary edit that would break one of the rules above."""


class SchemaEditor:
    """Vocabulary of one project, with the stage files available for checks.

    Usable as a context manager; the underlying connections are closed on exit.
    """

    def __init__(self, target_folder: str):
        self.target_folder = target_folder
        self.conn = store.open_store(
            store.project_path(target_folder), store.PROJECT)

    def __enter__(self) -> "SchemaEditor":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

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

        Base classes are rejected — ``animal`` and ``unknown`` are deliberately
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
        """Rename a concrete species; its id — and every stored row — is kept."""
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
                "renamed — other data and exports rely on them.")
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
        """Values in display order — ``ordinal``, not id."""
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
        """Set the display order. Ids are untouched — only ``ordinal`` moves."""
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
        fields = [row["name"] for row in self.conn.execute(
            "SELECT name FROM field_schema WHERE enum_id = ?", (enum_id,))]
        if not fields:
            return {}
        return self._attribute_usage(
            lambda attributes: any(
                attributes.get(name) == value_id for name in fields),
            fields)

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
        """Rename a field **and rewrite the stored rows that use it**.

        Unlike every other rename here this touches data, because ``attributes``
        is a JSON object keyed by field name (§5.1). Returns the number of rows
        rewritten.
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

        rewritten = self._rewrite_attribute_key(name, new_name)
        self.conn.execute(
            "UPDATE field_schema SET name = ? WHERE name = ? AND scope = ?",
            (new_name, name, scope))
        return rewritten

    def field_usage(self, name: str) -> Dict[str, int]:
        return self._attribute_usage(lambda attributes: name in attributes, [name])

    def delete_field(self, name: str, scope: str, force: bool = False) -> int:
        """Remove a field; with *force*, its stored values go too.

        Returns the number of rows whose attributes were rewritten.
        """
        row = self.conn.execute(
            "SELECT protected FROM field_schema WHERE name = ? AND scope = ?",
            (name, scope)).fetchone()
        if row is None:
            raise SchemaError(f"No '{scope}' field called '{name}'.")
        if row["protected"]:
            raise SchemaError(
                f"'{name}' is a built-in field and cannot be deleted.")
        usage = self.field_usage(name)
        if usage and not force:
            where = ", ".join(f"{n} in {table}" for table, n in sorted(usage.items()))
            raise SchemaError(
                f"'{name}' still holds values ({where}). Deleting it discards "
                "them; pass force=True to confirm.")
        rewritten = self._rewrite_attribute_key(name, None) if usage else 0
        self.conn.execute(
            "DELETE FROM field_schema WHERE name = ? AND scope = ?",
            (name, scope))
        return rewritten

    # ------------------------------------------------------------------
    # Attribute-blob helpers
    # ------------------------------------------------------------------

    def _attribute_usage(self, predicate, names: List[str]) -> Dict[str, int]:
        """Count rows whose ``attributes`` satisfy *predicate*.

        Rows are pre-filtered in SQL with a cheap ``LIKE`` on the field names and
        confirmed in Python, so this works on SQLite builds without JSON1.
        """
        usage: Dict[str, int] = {}
        for kind, tables in _ATTRIBUTE_TABLES.items():
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

    def _rewrite_attribute_key(self, name: str, new_name: Optional[str]) -> int:
        """Rename (or drop, when *new_name* is None) a key in every attributes blob."""
        rewritten = 0
        for kind, tables in _ATTRIBUTE_TABLES.items():
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
