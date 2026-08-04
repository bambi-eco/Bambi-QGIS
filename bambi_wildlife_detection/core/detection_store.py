# -*- coding: utf-8 -*-
"""Writing detections into the 6.0 store (EXCHANGE_FORMAT_PLAN.md §3.1, §4).

Every producer of detections — the detector, the TRex importer, the labelling
tool — goes through :func:`record_detections`, which is what makes the
per-source generation model of §4.1 hold:

* a producer owns a ``detection_sources`` row and may only delete or re-mint
  rows carrying its ``source_id``;
* re-running it bumps *that source's* generation and leaves every other
  producer's detections untouched;
* ``detection_id`` comes from ``AUTOINCREMENT`` and is never reused, so a stale
  reference cannot come to mean a different animal;
* the raw class is stored next to the resolved species, so a corrected
  ``class_mapping`` can be re-applied without re-running the producer.

Keeping this out of ``bambi_processing.py`` means the write path is reachable
from the unit suite, unlike the stage that calls it.
"""

import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from . import store

#: Producer kinds that own detections.
DETECTOR = "detector"
TREX = "trex"
MANUAL = "manual"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_source(project, kind: str, model: str = "", version: str = "",
                  config_hash: str = "") -> int:
    """Fetch or create the ``detection_sources`` row for a producer."""
    row = project.execute(
        "SELECT source_id FROM detection_sources "
        "WHERE kind = ? AND IFNULL(model, '') = ?", (kind, model)).fetchone()
    if row is not None:
        source_id = int(row["source_id"])
        if config_hash:
            project.execute(
                "UPDATE detection_sources SET config_hash = ?, version = ? "
                "WHERE source_id = ?", (config_hash, version, source_id))
        return source_id
    cur = project.execute(
        "INSERT INTO detection_sources "
        "(kind, model, version, config_hash, created_at) VALUES (?, ?, ?, ?, ?)",
        (kind, model, version, config_hash, _utc_now()))
    return int(cur.lastrowid)


def source_generation(project, source_id: int) -> int:
    row = project.execute(
        "SELECT generation FROM detection_sources WHERE source_id = ?",
        (source_id,)).fetchone()
    return 0 if row is None else int(row["generation"])


def record_detections(target_folder: str, modality: str,
                      rows: Iterable[dict], kind: str = DETECTOR,
                      model: str = "", version: str = "",
                      config_hash: str = "", log_fn=None) -> Dict[str, int]:
    """Replace *kind*'s detections for one modality with *rows*.

    Each row needs ``frame``, ``x1``, ``y1``, ``x2``, ``y2`` and may carry
    ``confidence``, ``source_class`` and ``attributes`` (a JSON string).

    Only this producer's rows are removed — a re-run of the detector never
    disturbs manually labelled or TRex-imported detections sharing the table.

    Returns ``{"written": n, "replaced": m, "generation": g}``.
    """
    rows = list(rows)

    project = store.open_store(
        store.project_path(target_folder), store.PROJECT)
    try:
        with store.transaction(project):
            source_id = ensure_source(project, kind, model, version, config_hash)
        species = {
            str(row["source_class"]): store.resolve_species(
                project, source_id, str(row["source_class"]))
            for row in rows if row.get("source_class") is not None
        }
        fallback = store.FALLBACK_SPECIES_ID

        conn = store.open_store(
            store.stage_path(target_folder, store.DETECTIONS, modality),
            store.DETECTIONS, modality)
        try:
            with store.transaction(conn):
                cur = conn.execute(
                    "DELETE FROM detections WHERE source_id = ?", (source_id,))
                replaced = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
                conn.executemany(
                    "INSERT INTO detections (frame, x1, y1, x2, y2, confidence, "
                    "species_id, source_id, source_class, attributes) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(row["frame"], row["x1"], row["y1"], row["x2"], row["y2"],
                      row.get("confidence", 1.0),
                      species.get(str(row.get("source_class")), fallback),
                      source_id,
                      None if row.get("source_class") is None
                      else str(row["source_class"]),
                      row.get("attributes"))
                     for row in rows])
        finally:
            conn.close()

        with store.transaction(project):
            project.execute(
                "UPDATE detection_sources SET generation = generation + 1 "
                "WHERE source_id = ?", (source_id,))
            generation = source_generation(project, source_id)
    finally:
        project.close()

    if log_fn:
        replaced_note = f", replaced {replaced}" if replaced else ""
        log_fn(f"Store: wrote {len(rows)} detection(s) for source '{kind}' "
               f"(generation {generation}{replaced_note})")
    return {"written": len(rows), "replaced": replaced, "generation": generation}


def detection_counts(target_folder: str, modality: str) -> Dict[str, int]:
    """Detections per producer kind, for logging and the stage status."""
    path = store.stage_path(target_folder, store.DETECTIONS, modality)
    if not os.path.isfile(path):
        return {}

    project = store.open_store(
        store.project_path(target_folder), store.PROJECT, create=False) \
        if os.path.isfile(store.project_path(target_folder)) else None
    if project is None:
        return {}
    try:
        kinds = {int(row["source_id"]): row["kind"] for row in project.execute(
            "SELECT source_id, kind FROM detection_sources")}
    finally:
        project.close()

    conn = store.open_store(path, store.DETECTIONS, modality)
    try:
        counts: Dict[str, int] = {}
        for row in conn.execute(
                "SELECT source_id, COUNT(*) AS n FROM detections "
                "GROUP BY source_id"):
            kind = kinds.get(int(row["source_id"]), "unknown")
            counts[kind] = counts.get(kind, 0) + int(row["n"])
        return counts
    finally:
        conn.close()


def rows_from_legacy_text(path: str) -> List[dict]:
    """Read a ``detections.txt`` into :func:`record_detections` rows.

    Used by the dual-write parity check: the text file the stage wrote is
    re-read and compared against what the store holds.
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
            if len(parts) < 5:
                continue
            try:
                rows.append({
                    "frame": int(parts[0]),
                    "x1": float(parts[1]), "y1": float(parts[2]),
                    "x2": float(parts[3]), "y2": float(parts[4]),
                    "confidence": float(parts[5]) if len(parts) > 5 else 1.0,
                    "source_class": parts[6] if len(parts) > 6 else "0",
                })
            except (ValueError, IndexError):
                continue
    return rows


def compare_with_legacy_text(target_folder: str, modality: str,
                             path: str, kind: str = DETECTOR
                             ) -> Optional[str]:
    """Check the store against the legacy text file for one producer.

    Returns ``None`` when they agree, otherwise a description of the first
    disagreement. This is the dual-write parity gate of §10 Phase 2 — the
    strongest safety net available while both paths coexist, and it costs
    almost nothing to run.
    """
    expected = rows_from_legacy_text(path)

    project_path = store.project_path(target_folder)
    if not os.path.isfile(project_path):
        return "no project store" if expected else None
    project = store.open_store(project_path, store.PROJECT)
    try:
        row = project.execute(
            "SELECT source_id FROM detection_sources WHERE kind = ?",
            (kind,)).fetchone()
    finally:
        project.close()
    if row is None:
        return f"no '{kind}' source in the store" if expected else None

    conn = store.open_store(
        store.stage_path(target_folder, store.DETECTIONS, modality),
        store.DETECTIONS, modality)
    try:
        actual = [dict(r) for r in conn.execute(
            "SELECT frame, x1, y1, x2, y2, confidence, source_class "
            "FROM detections WHERE source_id = ? ORDER BY detection_id",
            (row["source_id"],))]
    finally:
        conn.close()

    if len(actual) != len(expected):
        return (f"{len(expected)} row(s) in {os.path.basename(path)} but "
                f"{len(actual)} in the store")
    for i, (want, got) in enumerate(zip(expected, actual)):
        for key in ("frame", "x1", "y1", "x2", "y2"):
            if abs(float(want[key]) - float(got[key])) > 1e-6:
                return (f"row {i}: {key} is {got[key]} in the store, "
                        f"{want[key]} in the text file")
        if str(want["source_class"]) != str(got["source_class"]):
            return (f"row {i}: class is {got['source_class']} in the store, "
                    f"{want['source_class']} in the text file")
    return None
