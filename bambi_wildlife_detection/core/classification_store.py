# -*- coding: utf-8 -*-
"""Embedding bookkeeping and classification results against the 6.1 store.

One file per modality, ``bambi_{m}/classification.gpkg``, holding three things:

* **which detections have been embedded**, and by which run. That is the only
  fact about an embedding that is not derivable — where its vector lives
  follows from the frame it came from and its own id (see
  :mod:`core.embedding_files`) — and it is what makes a re-run incremental
  rather than another hour of GPU time;
* **per-frame head output**, keeping the head's own class label verbatim so
  nothing is lost in mapping it onto the project vocabulary;
* **the per-track call**, with the vote that produced it.

The vote counts are stored rather than just the winning label because a
demographic call is only as good as its margin: "male, 106 of 115 clear
frames" is what tells an ecologist whether an individual is borderline, and it
lets the quorum be changed afterwards without re-running anything.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set

from . import store

#: Tasks that produce a per-track call. ``occlusion`` is deliberately absent:
#: it stays per frame, because it selects which frames the others may vote
#: over rather than describing the animal.
VOTED_TASKS = ("species", "sex", "life_stage")

#: Life stage is the one task with two possible authors: a classifier where
#: someone has one, and the box-area estimate everywhere else. Both write here,
#: and ``model`` says which produced a given row.
LIFE_STAGE = "life_stage"

#: ``track_predictions.model`` of a row the size estimate produced.
SIZE_MODEL = "box-area"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _conn(target_folder: str, modality: str):
    return store.open_store(
        store.stage_path(target_folder, store.CLASSIFICATION, modality),
        store.CLASSIFICATION, modality)


def has_store(target_folder: str, modality: str) -> bool:
    """True when this modality has a classification store on disk."""
    return os.path.isfile(
        store.stage_path(target_folder, store.CLASSIFICATION, modality))


# ---------------------------------------------------------------------------
# Embedding runs
# ---------------------------------------------------------------------------

def start_embedding_run(target_folder: str, modality: str, backbone: str,
                        dim: int, crop_size: int, padding: float,
                        projection: str, folder: str,
                        thermal_anchored: bool = False,
                        plugin_version: str = "", log_fn=None) -> int:
    """Begin a run, deactivating any previous one, and return its id.

    Changing the projection or the backbone starts a *new* run rather than
    overwriting the old one, so switching between perspective and
    orthorectified crops — or trying a different backbone — does not discard
    work that is expensive to reproduce. :func:`reuse_embedding_run` is what
    resumes one instead.
    """
    conn = _conn(target_folder, modality)
    try:
        with store.transaction(conn):
            conn.execute("UPDATE embedding_runs SET is_active = 0")
            cur = conn.execute(
                "INSERT INTO embedding_runs (backbone, dim, crop_size, "
                "padding, projection, thermal_anchored, folder, "
                "plugin_version, created_at, is_active) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                (backbone, dim, crop_size, padding, projection,
                 1 if thermal_anchored else 0, folder, plugin_version,
                 _utc_now()))
            run_id = int(cur.lastrowid)
    finally:
        conn.close()

    if log_fn:
        log_fn(f"Store: started embedding run {run_id} ({projection}, "
               f"{backbone})")
    return run_id


def active_embedding_run(target_folder: str,
                         modality: str) -> Optional[dict]:
    """The embedding run the classifiers should read."""
    if not has_store(target_folder, modality):
        return None
    conn = _conn(target_folder, modality)
    try:
        row = conn.execute(
            "SELECT run_id, backbone, dim, crop_size, padding, projection, "
            "thermal_anchored, folder, plugin_version, created_at "
            "FROM embedding_runs WHERE is_active = 1 "
            "ORDER BY run_id DESC LIMIT 1").fetchone()
        return None if row is None else dict(row)
    finally:
        conn.close()


def reuse_embedding_run(target_folder: str, modality: str, backbone: str,
                        dim: int, crop_size: int, padding: float,
                        projection: str,
                        thermal_anchored: bool = False) -> Optional[dict]:
    """An existing run these settings could continue, if there is one.

    Matching on the settings that change the vectors — and only those — is what
    lets a flight be embedded in several sessions: a run interrupted halfway
    resumes instead of starting again. ``folder`` and ``plugin_version`` are
    deliberately not compared; neither changes what a vector contains.
    """
    if not has_store(target_folder, modality):
        return None
    conn = _conn(target_folder, modality)
    try:
        row = conn.execute(
            "SELECT run_id, backbone, dim, crop_size, padding, projection, "
            "thermal_anchored, folder, plugin_version, created_at "
            "FROM embedding_runs WHERE backbone = ? AND dim = ? AND "
            "crop_size = ? AND padding = ? AND projection = ? AND "
            "thermal_anchored = ? ORDER BY run_id DESC LIMIT 1",
            (backbone, dim, crop_size, padding, projection,
             1 if thermal_anchored else 0)).fetchone()
        return None if row is None else dict(row)
    finally:
        conn.close()


def activate_embedding_run(target_folder: str, modality: str,
                           run_id: int) -> None:
    """Make *run_id* the run everything downstream reads."""
    conn = _conn(target_folder, modality)
    try:
        with store.transaction(conn):
            conn.execute("UPDATE embedding_runs SET is_active = 0")
            conn.execute(
                "UPDATE embedding_runs SET is_active = 1 WHERE run_id = ?",
                (run_id,))
    finally:
        conn.close()


def record_embedded(target_folder: str, modality: str, run_id: int,
                    detection_ids: Iterable[int]) -> int:
    """Record that these detections now have a vector on disk.

    Written per batch rather than once at the end, so an interrupted run leaves
    behind exactly what it finished and resumes from there.
    """
    ids = [(int(detection_id), run_id) for detection_id in detection_ids]
    if not ids:
        return 0
    conn = _conn(target_folder, modality)
    try:
        with store.transaction(conn):
            conn.executemany(
                "INSERT OR IGNORE INTO embeddings (detection_id, run_id) "
                "VALUES (?, ?)", ids)
    finally:
        conn.close()
    return len(ids)


def embedded_ids(target_folder: str, modality: str,
                 run_id: int) -> Set[int]:
    """Detections already embedded by *run_id*."""
    if not has_store(target_folder, modality):
        return set()
    conn = _conn(target_folder, modality)
    try:
        return {int(row["detection_id"]) for row in conn.execute(
            "SELECT detection_id FROM embeddings WHERE run_id = ?", (run_id,))}
    finally:
        conn.close()


def pending_ids(target_folder: str, modality: str, run_id: int,
                detection_ids: Iterable[int]) -> List[int]:
    """Which of *detection_ids* this run has not embedded yet, in order."""
    done = embedded_ids(target_folder, modality, run_id)
    return [int(i) for i in detection_ids if int(i) not in done]


def forget_embedded(target_folder: str, modality: str, run_id: int,
                    detection_ids: Iterable[int]) -> int:
    """Drop membership rows whose vectors are gone.

    Reconciliation for the files-win rule: a ``.npz`` deleted by hand makes the
    row a lie, and the honest response is to re-embed rather than to fail on
    read.
    """
    ids = [(int(detection_id), run_id) for detection_id in detection_ids]
    if not ids:
        return 0
    conn = _conn(target_folder, modality)
    try:
        with store.transaction(conn):
            conn.executemany(
                "DELETE FROM embeddings WHERE detection_id = ? AND run_id = ?",
                ids)
    finally:
        conn.close()
    return len(ids)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def record_frame_predictions(target_folder: str, modality: str, task: str,
                             predictions: Iterable[dict],
                             log_fn=None) -> int:
    """Replace this modality's per-frame output for *task*.

    Each prediction is ``{"detection_id", "label", "class_index", "prob",
    "modality_in", "model"}``. Replacing rather than appending keeps one answer
    per detection per task: re-running a head with different weights must not
    leave the previous verdict beside the new one.
    """
    predictions = list(predictions)
    conn = _conn(target_folder, modality)
    try:
        with store.transaction(conn):
            conn.execute("DELETE FROM frame_predictions WHERE task = ?",
                         (task,))
            conn.executemany(
                "INSERT INTO frame_predictions (detection_id, task, label, "
                "class_index, prob, modality_in, model) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(row["detection_id"], task, row["label"],
                  int(row.get("class_index", -1)), float(row["prob"]),
                  row.get("modality_in", ""), row.get("model", ""))
                 for row in predictions])
    finally:
        conn.close()

    if log_fn:
        log_fn(f"Store: {len(predictions)} frame prediction(s) for '{task}'")
    return len(predictions)


def record_track_predictions(target_folder: str, modality: str, task: str,
                             predictions: Iterable[dict],
                             log_fn=None) -> int:
    """Replace this modality's per-track calls for *task*.

    Each prediction is ``{"track_id", "label", "votes", "n", "fraction",
    "modality_in", "model", "evidence"}``.
    """
    predictions = list(predictions)
    conn = _conn(target_folder, modality)
    try:
        with store.transaction(conn):
            conn.execute("DELETE FROM track_predictions WHERE task = ?",
                         (task,))
            conn.executemany(
                "INSERT INTO track_predictions (track_id, task, label, votes, "
                "n, fraction, modality_in, model, evidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(row["track_id"], task, row["label"], int(row["votes"]),
                  int(row["n"]), float(row["fraction"]),
                  row.get("modality_in", ""), row.get("model", ""),
                  json.dumps(row.get("evidence") or {}))
                 for row in predictions])
    finally:
        conn.close()

    if log_fn:
        log_fn(f"Store: {len(predictions)} track call(s) for '{task}'")
    return len(predictions)


def frame_predictions(target_folder: str, modality: str,
                      task: str) -> List[dict]:
    """Per-frame output for *task*, ordered by detection."""
    if not has_store(target_folder, modality):
        return []
    conn = _conn(target_folder, modality)
    try:
        return [dict(row) for row in conn.execute(
            "SELECT detection_id, label, class_index, prob, modality_in, "
            "model FROM frame_predictions WHERE task = ? "
            "ORDER BY detection_id", (task,))]
    finally:
        conn.close()


def track_predictions(target_folder: str, modality: str,
                      task: str) -> List[dict]:
    """Per-track calls for *task*, ordered by track."""
    if not has_store(target_folder, modality):
        return []
    conn = _conn(target_folder, modality)
    try:
        rows = []
        for row in conn.execute(
                "SELECT track_id, label, votes, n, fraction, modality_in, "
                "model, evidence FROM track_predictions WHERE task = ? "
                "ORDER BY track_id", (task,)):
            entry = dict(row)
            entry["evidence"] = _load_json(row["evidence"])
            rows.append(entry)
        return rows
    finally:
        conn.close()


def _load_json(raw) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def clear_task(target_folder: str, modality: str, task: str) -> Dict[str, int]:
    """Remove everything *task* produced, leaving the embeddings alone.

    Resetting one head must not throw away the vectors every other head reads —
    which is why the ``classification`` stage does not own the store file (see
    ``core.stages.STAGE_STORE_KIND``) and clears its rows instead.
    """
    if not has_store(target_folder, modality):
        return {"frames": 0, "tracks": 0}
    conn = _conn(target_folder, modality)
    try:
        with store.transaction(conn):
            frames = conn.execute(
                "DELETE FROM frame_predictions WHERE task = ?", (task,))
            frame_count = frames.rowcount if frames.rowcount > 0 else 0
            tracks = conn.execute(
                "DELETE FROM track_predictions WHERE task = ?", (task,))
            track_count = tracks.rowcount if tracks.rowcount > 0 else 0
    finally:
        conn.close()
    return {"frames": frame_count, "tracks": track_count}


def tasks_present(target_folder: str, modality: str) -> List[str]:
    """Which tasks have produced a per-track call in this modality."""
    if not has_store(target_folder, modality):
        return []
    conn = _conn(target_folder, modality)
    try:
        return [row["task"] for row in conn.execute(
            "SELECT DISTINCT task FROM track_predictions ORDER BY task")]
    finally:
        conn.close()
