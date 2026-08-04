# -*- coding: utf-8 -*-
"""One-shot import of a 5.x target folder into the 6.0 store.

Implements EXCHANGE_FORMAT_PLAN.md §9. Reads the legacy text/JSON outputs and
writes the equivalent rows into ``project.gpkg`` and the per-stage files. The
legacy files are never modified — migration is additive, so a failed or
half-finished run can simply be deleted and retried.

Two things happen here for the last time, and nowhere else in the codebase:

* **The detection↔track linkage is recovered by matching coordinates.** That is
  the very defect 6.0 removes (§1.2a), but a 5.x project holds no other record
  of it. Unlike ``core/track_export.py``, a failed match costs one detection its
  track membership instead of discarding the whole frame.
* **The species→class-id mapping is derived.** ``species_class_ids()`` computes
  it alphabetically on every export; migration reproduces that once, freezes it
  into the ``species`` table and never derives it again (§1.3).

The ambiguity of legacy ``class_id 0`` — "animal" to the detector, "unknown" to
the labelling tool — is resolved by which side of ``DETECTIONS_MARKER`` a row
sits on, the only place that distinction survives. It is recorded as ordinary
``class_mapping`` rows so the resolution stays data-driven afterwards.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from . import store

LogFn = Optional[callable]

#: Marker written by the labelling tool into ``detections.txt``. Duplicated
#: from ``core.labelling.LabelStore`` so migration does not import the labelling
#: model (and the Qt-free import graph stays shallow).
DETECTIONS_MARKER = "# --- labelled detections (BAMBI labelling tool) ---"

#: Rounding used to match a legacy box back to its detection. Matches the
#: precision ``core/track_export.py`` used, so migration recovers exactly the
#: links the 5.x pipeline itself would have found.
_MATCH_PRECISION = 3


def _key(*vals) -> tuple:
    return tuple(round(float(v), _MATCH_PRECISION) for v in vals)


class MigrationReport:
    """Counts and warnings from one migration run."""

    def __init__(self):
        self.counts: Dict[str, int] = {}
        self.warnings: List[str] = []

    def add(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return f"MigrationReport(counts={self.counts}, warnings={self.warnings})"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def legacy_folder(target_folder: str, base: str, modality: str) -> str:
    """Path of a 5.x output folder, e.g. ``<target>/detections_t``."""
    return os.path.join(target_folder, f"{base}_{modality}")


def has_legacy_outputs(target_folder: str) -> bool:
    """True when *target_folder* holds anything a migration would read."""
    for modality in store.MODALITIES:
        for base in ("detections", "georeferenced", "tracks", "fov",
                     "labels", "segmentation"):
            if os.path.isdir(legacy_folder(target_folder, base, modality)):
                return True
    return False


def is_migrated(target_folder: str) -> bool:
    """True when a 6.0 store already exists for *target_folder*."""
    return os.path.isfile(store.project_path(target_folder))


def legacy_modalities(target_folder: str) -> List[str]:
    """Modalities with at least one legacy output folder."""
    found = []
    for modality in store.MODALITIES:
        for base in ("detections", "georeferenced", "tracks", "fov",
                     "labels", "segmentation"):
            if os.path.isdir(legacy_folder(target_folder, base, modality)):
                found.append(modality)
                break
    return found


# ---------------------------------------------------------------------------
# Legacy parsers
# ---------------------------------------------------------------------------

def read_legacy_detections(path: str) -> Tuple[List[dict], List[dict]]:
    """Split ``detections.txt`` into the detector block and the label block.

    Returns ``(detector_rows, label_rows)``; each row is a dict with
    ``frame``/``x1``/``y1``/``x2``/``y2``/``confidence``/``class_id``. Rows
    below :data:`DETECTIONS_MARKER` are the labelling tool's export, and their
    ``class_id`` uses the label taxonomy rather than the detector's.
    """
    detector: List[dict] = []
    labels: List[dict] = []
    target = detector

    if not os.path.isfile(path):
        return detector, labels

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            if stripped == DETECTIONS_MARKER:
                target = labels
                continue
            stripped = stripped.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 5:
                continue
            try:
                target.append({
                    "frame": int(parts[0]),
                    "x1": float(parts[1]), "y1": float(parts[2]),
                    "x2": float(parts[3]), "y2": float(parts[4]),
                    "confidence": float(parts[5]) if len(parts) > 5 else 1.0,
                    "class_id": int(parts[6]) if len(parts) > 6 else 0,
                })
            except (ValueError, IndexError):
                continue
    return detector, labels


def read_legacy_georeferenced(path: str) -> List[dict]:
    """Parse ``georeferenced.txt`` in file order, keeping invalid rows.

    ``core/pipeline_outputs.load_georef_detections_by_frame`` drops rows with a
    negative corner; migration keeps them so they can become ``georef_failures``
    instead of vanishing (§9).
    """
    rows: List[dict] = []
    if not os.path.isfile(path):
        return rows

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                rows.append({
                    "idx": int(parts[0]), "frame": int(parts[1]),
                    "gx1": float(parts[2]), "gy1": float(parts[3]),
                    "gz1": float(parts[4]), "gx2": float(parts[5]),
                    "gy2": float(parts[6]), "gz2": float(parts[7]),
                    "confidence": float(parts[8]), "class_id": int(parts[9]),
                })
            except (ValueError, IndexError):
                continue
    return rows


def read_legacy_tracks(path: str) -> List[dict]:
    """Parse ``tracks.csv`` (geo-space track points)."""
    rows: List[dict] = []
    if not os.path.isfile(path):
        return rows

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 10:
                continue
            try:
                rows.append({
                    "frame": int(float(parts[0])),
                    "track_id": int(float(parts[1])),
                    "gx1": float(parts[2]), "gy1": float(parts[3]),
                    "gz1": float(parts[4]), "gx2": float(parts[5]),
                    "gy2": float(parts[6]), "gz2": float(parts[7]),
                    "confidence": float(parts[8]),
                    "class_id": int(float(parts[9])),
                    "interpolated": int(float(parts[10])) if len(parts) > 10 else 0,
                })
            except (ValueError, IndexError):
                continue
    return rows


def read_legacy_pixel_tracks(path: str) -> List[dict]:
    """Parse ``tracks_pixel.csv`` — the preferred linkage source (§9)."""
    rows: List[dict] = []
    if not os.path.isfile(path):
        return rows

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                rows.append({
                    "frame": int(float(parts[0])),
                    "track_id": int(float(parts[1])),
                    "x1": float(parts[2]), "y1": float(parts[3]),
                    "x2": float(parts[4]), "y2": float(parts[5]),
                    "confidence": float(parts[6]) if len(parts) > 6 else 1.0,
                    "class_id": int(float(parts[7])) if len(parts) > 7 else 0,
                    "interpolated": int(float(parts[8])) if len(parts) > 8 else 0,
                })
            except (ValueError, IndexError):
                continue
    return rows


def read_legacy_fov(path: str) -> Dict[int, List[Tuple[float, float, float]]]:
    """Parse ``fov_polygons.txt`` into ``frame -> [(x, y, z), …]``."""
    polygons: Dict[int, List[Tuple[float, float, float]]] = {}
    if not os.path.isfile(path):
        return polygons

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                frame = int(parts[0])
                n_points = int(parts[1])
            except ValueError:
                continue
            points = []
            for i in range(n_points):
                base = 2 + i * 3
                if base + 2 < len(parts):
                    try:
                        points.append((float(parts[base]),
                                       float(parts[base + 1]),
                                       float(parts[base + 2])))
                    except ValueError:
                        break
            if points:
                polygons[frame] = points
    return polygons


def read_legacy_labels(path: str) -> dict:
    """Load ``labels.json`` as-is (``{}`` when absent or unreadable)."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: The 5.x species taxonomy, in list order. Index 0 ("unknown") becomes the
#: ``animal`` base class; 1–9 keep their ids (see ``store.SEEDED_SPECIES``).
LEGACY_SPECIES_CLASSES = [
    "unknown", "roe deer", "red deer", "fallow deer", "wild boar",
    "chamois", "fox", "hare", "bird", "other",
]


def legacy_species_class_ids(species_names) -> Dict[str, int]:
    """Reproduce ``LabelStore.species_class_ids()`` exactly, once.

    Built-ins keep their list index; anything else is appended in alphabetical
    order. This is the last time the mapping is derived rather than stored.
    """
    mapping = {name: i for i, name in enumerate(LEGACY_SPECIES_CLASSES)}
    custom = sorted({
        (name or "unknown").strip().lower() for name in species_names
    } - set(LEGACY_SPECIES_CLASSES))
    for i, name in enumerate(custom):
        mapping[name] = len(LEGACY_SPECIES_CLASSES) + i
    return mapping


def migrate_vocabulary(project, target_folder: str,
                       report: MigrationReport) -> Dict[str, int]:
    """Freeze the species mapping and import the custom field schema.

    Returns the legacy ``class_id -> species_id`` mapping for label rows, which
    differs from the detector's only in class 0 (``unknown`` vs ``animal``).
    """
    label_species: set = set()
    custom_fields: List[dict] = []
    for modality in store.MODALITIES:
        data = read_legacy_labels(os.path.join(
            legacy_folder(target_folder, "labels", modality), "labels.json"))
        for track in data.get("tracks", []) or []:
            label_species.add((track.get("species") or "unknown").strip().lower())
        for field in data.get("custom_fields", []) or []:
            if field not in custom_fields:
                custom_fields.append(field)

    legacy_map = legacy_species_class_ids(label_species)

    # Custom species keep the ids the 5.x export would have given them.
    for name, class_id in sorted(legacy_map.items(), key=lambda kv: kv[1]):
        if class_id < len(LEGACY_SPECIES_CLASSES):
            continue  # built-in, already seeded
        project.execute(
            "INSERT OR IGNORE INTO species (species_id, name) VALUES (?, ?)",
            (class_id, name))
        report.add("species_custom")

    for field in custom_fields:
        name = (field.get("name") or "").strip()
        if not name:
            continue
        # 5.x scopes: "track" stays, "keyframe" becomes per-detection (§5).
        scope = "detection" if field.get("scope") == "keyframe" else "track"
        project.execute(
            "INSERT OR IGNORE INTO field_schema (name, type, scope) "
            "VALUES (?, ?, ?)", (name, field.get("type", "string"), scope))
        report.add("custom_fields")

    return legacy_map


def _ensure_source(project, kind: str, model: str = "") -> int:
    """Fetch or create a ``detection_sources`` row."""
    row = project.execute(
        "SELECT source_id FROM detection_sources WHERE kind = ? AND "
        "IFNULL(model, '') = ?", (kind, model)).fetchone()
    if row is not None:
        return int(row["source_id"])
    cur = project.execute(
        "INSERT INTO detection_sources (kind, model, created_at) "
        "VALUES (?, ?, ?)", (kind, model, ""))
    return int(cur.lastrowid)


def _seed_class_mapping(project, source_id: int, class_ids, zero_species: int,
                        ) -> None:
    """Record identity mappings, with class 0 pointing at *zero_species*.

    Making the ``class_id 0`` decision a mapping row rather than a hardcoded
    branch means it stays visible and editable after migration.
    """
    for class_id in sorted(set(class_ids)):
        species_id = zero_species if class_id == 0 else class_id
        exists = project.execute(
            "SELECT 1 FROM species WHERE species_id = ?", (species_id,)).fetchone()
        if exists is None:
            species_id = store.FALLBACK_SPECIES_ID
        project.execute(
            "INSERT OR IGNORE INTO class_mapping "
            "(source_id, source_class, species_id) VALUES (?, ?, ?)",
            (source_id, str(class_id), species_id))


# ---------------------------------------------------------------------------
# Stage migration
# ---------------------------------------------------------------------------

def migrate_detections(target_folder: str, modality: str, project,
                       report: MigrationReport) -> Dict[tuple, int]:
    """Import ``detections.txt``; returns ``(frame, box) -> detection_id``.

    The returned index is what the tracks migration uses to recover membership
    without re-deriving anything from geometry a second time.
    """
    path = os.path.join(
        legacy_folder(target_folder, "detections", modality), "detections.txt")
    detector_rows, label_rows = read_legacy_detections(path)
    if not detector_rows and not label_rows:
        return {}

    detector_source = _ensure_source(project, "detector")
    _seed_class_mapping(project, detector_source,
                        [r["class_id"] for r in detector_rows],
                        zero_species=store.FALLBACK_SPECIES_ID)
    if label_rows:
        manual_source = _ensure_source(project, "manual")
        # Below the marker, class 0 is the labelling tool's "unknown", not the
        # detector's "animal" — the one place that distinction survives (§1.3).
        _seed_class_mapping(project, manual_source,
                            [r["class_id"] for r in label_rows],
                            zero_species=-1)
    else:
        manual_source = None

    conn = store.open_store(
        store.stage_path(target_folder, store.DETECTIONS, modality),
        store.DETECTIONS, modality)
    index: Dict[tuple, int] = {}
    try:
        with store.transaction(conn):
            for rows, source_id, tag in (
                    (detector_rows, detector_source, "detections_detector"),
                    (label_rows, manual_source, "detections_manual")):
                for row in rows:
                    if source_id is None:
                        continue
                    species_id = store.resolve_species(
                        project, source_id, str(row["class_id"]))
                    cur = conn.execute(
                        "INSERT INTO detections (frame, x1, y1, x2, y2, "
                        "confidence, species_id, source_id, source_class) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (row["frame"], row["x1"], row["y1"], row["x2"],
                         row["y2"], row["confidence"], species_id, source_id,
                         str(row["class_id"])))
                    detection_id = int(cur.lastrowid)
                    index.setdefault(
                        (row["frame"], _key(row["x1"], row["y1"],
                                            row["x2"], row["y2"])),
                        detection_id)
                    report.add(tag)
    finally:
        conn.close()
    return index


def migrate_georeferenced(target_folder: str, modality: str,
                          report: MigrationReport) -> Dict[tuple, int]:
    """Import ``georeferenced.txt``, keyed to detections by per-frame order.

    Returns ``(frame, geo_box) -> detection_id`` so tracks recorded in geo space
    can still be linked when no pixel tracks file exists.
    """
    path = os.path.join(
        legacy_folder(target_folder, "georeferenced", modality),
        "georeferenced.txt")
    geo_rows = read_legacy_georeferenced(path)
    if not geo_rows:
        return {}

    det_path = store.stage_path(target_folder, store.DETECTIONS, modality)
    if not os.path.isfile(det_path):
        report.warn(f"[{modality}] georeferenced.txt without detections.txt — "
                    "skipped")
        return {}

    det_conn = store.open_store(det_path, store.DETECTIONS, modality)
    try:
        by_frame: Dict[int, List[int]] = {}
        for row in det_conn.execute(
                "SELECT detection_id, frame FROM detections "
                "ORDER BY detection_id"):
            by_frame.setdefault(row["frame"], []).append(row["detection_id"])
    finally:
        det_conn.close()

    geo_by_frame: Dict[int, List[dict]] = {}
    for row in geo_rows:
        geo_by_frame.setdefault(row["frame"], []).append(row)

    conn = store.open_store(
        store.stage_path(target_folder, store.GEOREFERENCED, modality),
        store.GEOREFERENCED, modality)
    geo_index: Dict[tuple, int] = {}
    try:
        with store.transaction(conn):
            for frame, rows in geo_by_frame.items():
                detection_ids = by_frame.get(frame, [])
                if len(detection_ids) != len(rows):
                    # 5.x aligned these positionally; when the counts disagree
                    # the alignment is unrecoverable. Note it rather than guess.
                    report.warn(
                        f"[{modality}] frame {frame}: {len(rows)} geo rows vs "
                        f"{len(detection_ids)} detections — left unlinked")
                    report.add("georef_unlinked", len(rows))
                    continue
                for detection_id, row in zip(detection_ids, rows):
                    if row["gx1"] < 0 or row["gy1"] < 0:
                        conn.execute(
                            "INSERT OR REPLACE INTO georef_failures "
                            "(detection_id, reason) VALUES (?, 'legacy_invalid')",
                            (detection_id,))
                        report.add("georef_failures")
                        continue
                    conn.execute(
                        "INSERT OR REPLACE INTO detections_geo "
                        "(detection_id, gx1, gy1, gz1, gx2, gy2, gz2) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (detection_id, row["gx1"], row["gy1"], row["gz1"],
                         row["gx2"], row["gy2"], row["gz2"]))
                    geo_index.setdefault(
                        (frame, _key(row["gx1"], row["gy1"],
                                     row["gx2"], row["gy2"])), detection_id)
                    report.add("detections_geo")
    finally:
        conn.close()
    return geo_index


def migrate_tracks(target_folder: str, modality: str,
                   pixel_index: Dict[tuple, int],
                   geo_index: Dict[tuple, int],
                   report: MigrationReport) -> None:
    """Import tracks, preferring ``tracks_pixel.csv`` for the linkage (§9)."""
    folder = legacy_folder(target_folder, "tracks", modality)
    pixel_rows = read_legacy_pixel_tracks(
        os.path.join(folder, "tracks_pixel.csv"))
    geo_rows = read_legacy_tracks(os.path.join(folder, "tracks.csv"))
    if not pixel_rows and not geo_rows:
        return

    if pixel_rows:
        rows, index, box_keys = pixel_rows, pixel_index, ("x1", "y1", "x2", "y2")
        source = "tracks_pixel.csv"
    else:
        rows, index, box_keys = geo_rows, geo_index, ("gx1", "gy1", "gx2", "gy2")
        source = "tracks.csv"

    conn = store.open_store(
        store.stage_path(target_folder, store.TRACKS, modality),
        store.TRACKS, modality)
    try:
        with store.transaction(conn):
            cur = conn.execute(
                "INSERT INTO track_runs (kind, tracker, created_at, is_active) "
                "VALUES ('builtin', ?, '', 1)", (f"migrated from {source}",))
            run_id = int(cur.lastrowid)

            track_ids: Dict[int, int] = {}
            unmatched = 0
            for row in rows:
                legacy_id = row["track_id"]
                if legacy_id not in track_ids:
                    species_id = row.get("class_id", 0)
                    new = conn.execute(
                        "INSERT INTO tracks (run_id, species_id) VALUES (?, ?)",
                        (run_id, species_id if species_id is not None else 0))
                    track_ids[legacy_id] = int(new.lastrowid)
                    report.add("tracks")

                key = (row["frame"], _key(*[row[k] for k in box_keys]))
                detection_id = index.get(key)
                if detection_id is None:
                    # One detection loses its membership — not, as in 5.x, the
                    # whole frame (§9).
                    unmatched += 1
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO track_members "
                    "(track_id, detection_id, interpolated) VALUES (?, ?, ?)",
                    (track_ids[legacy_id], detection_id,
                     row.get("interpolated", 0)))
                report.add("track_members")

            if unmatched:
                report.add("track_members_unmatched", unmatched)
                report.warn(
                    f"[{modality}] {unmatched} track point(s) could not be "
                    f"matched to a detection and were left out")
    finally:
        conn.close()


def migrate_fov(target_folder: str, modality: str,
                report: MigrationReport) -> None:
    """Import ``fov_polygons.txt``."""
    polygons = read_legacy_fov(os.path.join(
        legacy_folder(target_folder, "fov", modality), "fov_polygons.txt"))
    if not polygons:
        return

    conn = store.open_store(
        store.stage_path(target_folder, store.FOV, modality),
        store.FOV, modality)
    try:
        with store.transaction(conn):
            for frame, points in polygons.items():
                conn.execute(
                    "INSERT OR REPLACE INTO fov_polygons (frame, n_points) "
                    "VALUES (?, ?)", (frame, len(points)))
                conn.executemany(
                    "INSERT OR REPLACE INTO fov_vertices "
                    "(frame, seq, x, y, z) VALUES (?, ?, ?, ?, ?)",
                    [(frame, seq, x, y, z)
                     for seq, (x, y, z) in enumerate(points)])
                report.add("fov_polygons")
    finally:
        conn.close()


def _resolve_enum_value(project, enum_name: str, label: str,
                        report: MigrationReport) -> Optional[int]:
    """Map a legacy label string onto a ``value_id``, appending if unknown.

    Unmatched values are added rather than dropped, so hand-edited or older
    files keep their data (§9).
    """
    row = project.execute(
        "SELECT enum_id FROM enums WHERE name = ?", (enum_name,)).fetchone()
    if row is None:
        return None
    enum_id = int(row["enum_id"])

    label = (label or "").strip()
    if not label:
        return None
    match = project.execute(
        "SELECT value_id FROM enum_values WHERE enum_id = ? AND label = ?",
        (enum_id, label)).fetchone()
    if match is not None:
        return int(match["value_id"])

    value_id = store.next_enum_value_id(project, enum_id)
    project.execute(
        "INSERT INTO enum_values (enum_id, value_id, label, ordinal) "
        "VALUES (?, ?, ?, ?)", (enum_id, value_id, label, value_id))
    report.add("enum_values_appended")
    report.warn(f"enum '{enum_name}': appended unseen value '{label}'")
    return value_id


def migrate_labels(target_folder: str, modality: str, project,
                   legacy_map: Dict[str, int],
                   report: MigrationReport) -> None:
    """Import ``labels.json`` into the label tables."""
    data = read_legacy_labels(os.path.join(
        legacy_folder(target_folder, "labels", modality), "labels.json"))
    tracks = data.get("tracks") or []
    if not tracks:
        return

    conn = store.open_store(
        store.stage_path(target_folder, store.LABELS, modality),
        store.LABELS, modality)
    try:
        with store.transaction(conn):
            for track in tracks:
                species = (track.get("species") or "unknown").strip().lower()
                species_id = legacy_map.get(species, store.FALLBACK_SPECIES_ID)
                if species == "unknown":
                    species_id = -1  # the base class, not 'animal'

                attributes = dict(track.get("attributes") or {})
                for field, enum_name in (("sex", "sex"), ("age", "age")):
                    value_id = _resolve_enum_value(
                        project, enum_name, track.get(field, ""), report)
                    if value_id is not None:
                        attributes[field] = value_id

                label_track_id = int(track.get("track_id", 0))
                conn.execute(
                    "INSERT OR REPLACE INTO label_tracks "
                    "(label_track_id, species_id, attributes) VALUES (?, ?, ?)",
                    (label_track_id, species_id, json.dumps(attributes)))
                report.add("label_tracks")

                for frame, entry in (track.get("keyframes") or {}).items():
                    kf_attributes = dict(entry.get("attributes") or {})
                    value_id = _resolve_enum_value(
                        project, "occlusion", entry.get("occlusion", ""), report)
                    if value_id is not None:
                        kf_attributes["occlusion"] = value_id
                    conn.execute(
                        "INSERT OR REPLACE INTO label_keyframes "
                        "(label_track_id, frame, x1, y1, x2, y2, stop, "
                        " attributes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (label_track_id, int(frame),
                         float(entry["x1"]), float(entry["y1"]),
                         float(entry["x2"]), float(entry["y2"]),
                         1 if entry.get("stop") else 0,
                         json.dumps(kf_attributes)))
                    report.add("label_keyframes")
    finally:
        conn.close()


def migrate_segmentation(target_folder: str, modality: str,
                         report: MigrationReport) -> None:
    """Import the SAM3 pixel/geo segmentation JSON."""
    folder = legacy_folder(target_folder, "segmentation", modality)
    pixel = _load_segmentation(os.path.join(folder, "segmentation_pixel.json"))
    georef = _load_segmentation(os.path.join(folder, "segmentation_georef.json"))
    if not pixel and not georef:
        return

    geo_by_frame: Dict[int, List[list]] = {}
    for frame_result in georef:
        for prompt in frame_result.get("prompts", []) or []:
            for prediction in prompt.get("predictions", []) or []:
                geo_by_frame.setdefault(
                    int(frame_result.get("frame_idx", -1)), []).append(
                        prediction.get("world_polygons") or [])

    conn = store.open_store(
        store.stage_path(target_folder, store.SEGMENTATION, modality),
        store.SEGMENTATION, modality)
    try:
        with store.transaction(conn):
            for frame_result in pixel:
                frame = int(frame_result.get("frame_idx", -1))
                geo_queue = list(geo_by_frame.get(frame, []))
                for prompt in frame_result.get("prompts", []) or []:
                    for prediction in prompt.get("predictions", []) or []:
                        polygons = prediction.get("polygons")
                        world = geo_queue.pop(0) if geo_queue else None
                        conn.execute(
                            "INSERT INTO segments "
                            "(frame, polygon_px, polygon_geo, attributes) "
                            "VALUES (?, ?, ?, ?)",
                            (frame,
                             json.dumps(polygons) if polygons else None,
                             json.dumps(world) if world else None,
                             json.dumps({
                                 "prompt": prompt.get("prompt", ""),
                                 "confidence": prediction.get("confidence", 0.0),
                             })))
                        report.add("segments")
    finally:
        conn.close()


def _load_segmentation(path: str) -> list:
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (ValueError, OSError):
        return []
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def migrate_project(target_folder: str, log_fn: LogFn = None) -> MigrationReport:
    """Migrate every legacy output in *target_folder* into the 6.0 store.

    Additive and repeatable: the legacy files are only read. Existing 6.0
    stores are *not* rebuilt — delete them first to re-run a migration.
    """
    report = MigrationReport()

    def log(message: str) -> None:
        if log_fn:
            log_fn(message)

    modalities = legacy_modalities(target_folder)
    if not modalities:
        report.warn("No 5.x outputs found — nothing to migrate.")
        return report

    project = store.open_store(store.project_path(target_folder), store.PROJECT)
    try:
        with store.transaction(project):
            legacy_map = migrate_vocabulary(project, target_folder, report)
        log(f"Vocabulary: {report.counts.get('species_custom', 0)} custom "
            f"species, {report.counts.get('custom_fields', 0)} custom field(s)")

        if os.path.isdir(legacy_folder(target_folder, "tracks_pixel", "t")) or \
                os.path.isdir(legacy_folder(target_folder, "tracks_pixel", "w")):
            report.warn(
                "A TRex import folder is present: 5.x recorded TRex detections "
                "in the same file and format as the detector's, so they are "
                "imported as 'detector' and cannot be told apart afterwards.")

        for modality in modalities:
            log(f"Migrating modality '{modality}'…")
            with store.transaction(project):
                pixel_index = migrate_detections(
                    target_folder, modality, project, report)
            geo_index = migrate_georeferenced(target_folder, modality, report)
            migrate_tracks(target_folder, modality, pixel_index, geo_index,
                           report)
            migrate_fov(target_folder, modality, report)
            with store.transaction(project):
                migrate_labels(target_folder, modality, project, legacy_map,
                               report)
            migrate_segmentation(target_folder, modality, report)
    finally:
        project.close()

    for key in sorted(report.counts):
        log(f"  {key}: {report.counts[key]}")
    for warning in report.warnings:
        log(f"  warning: {warning}")
    return report
