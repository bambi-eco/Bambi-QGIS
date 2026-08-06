# -*- coding: utf-8 -*-
"""Cross-modal track matching against the 6.1 store.

Records which thermal track and which RGB track are the same animal, and which
of their detections correspond frame by frame.

Two decisions shape this module:

* **A match belongs to neither modality**, so it lives in ``matches.gpkg``
  beside ``project.gpkg`` rather than under ``bambi_t/`` or ``bambi_w/``.
  Filing it under one side would quietly make that side authoritative, and
  resetting that modality would take the other's matches with it.
* **The per-frame pairing is stored, not recomputed.** Building a ``matched``
  feature vector means "the RGB detection that goes with this thermal one",
  and answering that from :func:`detection_pairs` is a join. Re-deriving it
  from the affine at read time would let the answer drift as soon as anything
  upstream changed, which is the class of bug the 6.0 store exists to remove.

Runs are versioned like tracking runs: a new one deactivates the previous,
which is kept, so re-running with a different gate stays comparable.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from . import store


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _conn(target_folder: str):
    return store.open_store(store.matches_path(target_folder), store.MATCHES)


def has_store(target_folder: str) -> bool:
    """True when this flight has been matched at least once."""
    return os.path.isfile(store.matches_path(target_folder))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def record_matches(target_folder: str, matches: Iterable[dict],
                   affine: Optional[Tuple[List[List[float]], List[float]]] = None,
                   affine_rmse: Optional[float] = None,
                   config_hash: str = "", log_fn=None) -> Dict[str, int]:
    """Record one matching run.

    Each entry of *matches* is ``{"track_id_t", "track_id_w", "shared",
    "median_dist", "conf_t", "conf_w", "pairs": [{"frame_t", "frame_w",
    "detection_id_t", "detection_id_w", "dist"}]}``.

    The one-to-one property is enforced by the schema rather than trusted here:
    a caller that assigned a track twice gets an ``IntegrityError`` instead of
    silently double-counting an animal.
    """
    matches = list(matches)

    conn = _conn(target_folder)
    try:
        with store.transaction(conn):
            conn.execute("UPDATE match_runs SET is_active = 0")
            cur = conn.execute(
                "INSERT INTO match_runs (affine, affine_rmse, n_pairs, "
                "config_hash, created_at, is_active) VALUES (?, ?, ?, ?, ?, 1)",
                (json.dumps(affine) if affine is not None else None,
                 affine_rmse, len(matches), config_hash, _utc_now()))
            run_id = int(cur.lastrowid)

            pair_count = 0
            for match in matches:
                row = conn.execute(
                    "INSERT INTO track_matches (run_id, track_id_t, "
                    "track_id_w, shared, median_dist, conf_t, conf_w) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (run_id, match["track_id_t"], match["track_id_w"],
                     match["shared"], match["median_dist"],
                     match.get("conf_t"), match.get("conf_w")))
                match_id = int(row.lastrowid)
                pairs = match.get("pairs") or []
                conn.executemany(
                    "INSERT INTO detection_matches (match_id, frame_t, "
                    "frame_w, detection_id_t, detection_id_w, dist) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [(match_id, pair["frame_t"], pair["frame_w"],
                      pair["detection_id_t"], pair["detection_id_w"],
                      pair.get("dist")) for pair in pairs])
                pair_count += len(pairs)
    finally:
        conn.close()

    if log_fn:
        log_fn(f"Store: recorded {len(matches)} matched track pair(s) with "
               f"{pair_count} frame correspondence(s) as run {run_id}")
    return {"run_id": run_id, "matches": len(matches), "pairs": pair_count}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def active_run(target_folder: str) -> Optional[dict]:
    """The matching run everything downstream should read."""
    if not has_store(target_folder):
        return None
    conn = _conn(target_folder)
    try:
        row = conn.execute(
            "SELECT run_id, affine, affine_rmse, n_pairs, config_hash, "
            "created_at FROM match_runs WHERE is_active = 1 "
            "ORDER BY run_id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        result = dict(row)
        result["affine"] = (json.loads(row["affine"])
                            if row["affine"] else None)
        return result
    finally:
        conn.close()


def _resolve_run(conn, run_id: Optional[int]) -> Optional[int]:
    if run_id is not None:
        return run_id
    row = conn.execute(
        "SELECT run_id FROM match_runs WHERE is_active = 1 "
        "ORDER BY run_id DESC LIMIT 1").fetchone()
    return None if row is None else int(row["run_id"])


def track_matches(target_folder: str,
                  run_id: Optional[int] = None) -> List[dict]:
    """Matched track pairs of a run, closest first."""
    if not has_store(target_folder):
        return []
    conn = _conn(target_folder)
    try:
        resolved = _resolve_run(conn, run_id)
        if resolved is None:
            return []
        return [dict(row) for row in conn.execute(
            "SELECT match_id, track_id_t, track_id_w, shared, median_dist, "
            "conf_t, conf_w FROM track_matches WHERE run_id = ? "
            "ORDER BY median_dist", (resolved,))]
    finally:
        conn.close()


def detection_pairs(target_folder: str,
                    run_id: Optional[int] = None) -> List[dict]:
    """Every frame-level correspondence of a run."""
    if not has_store(target_folder):
        return []
    conn = _conn(target_folder)
    try:
        resolved = _resolve_run(conn, run_id)
        if resolved is None:
            return []
        return [dict(row) for row in conn.execute(
            "SELECT d.match_id, d.frame_t, d.frame_w, d.detection_id_t, "
            "d.detection_id_w, d.dist FROM detection_matches d "
            "JOIN track_matches m ON m.match_id = d.match_id "
            "WHERE m.run_id = ? ORDER BY d.frame_t", (resolved,))]
    finally:
        conn.close()


def partner_detections(target_folder: str, modality: str,
                       run_id: Optional[int] = None) -> Dict[int, int]:
    """Map each detection of *modality* onto its partner in the other one.

    This is what turns two 1280-d vectors into one 2560-d ``matched`` feature:
    the caller walks its own modality's detections and looks the other side up
    here, with a missing key meaning "this frame has no counterpart", which the
    matched heads cannot accept a zero-filled substitute for.
    """
    if modality not in store.MODALITIES:
        raise ValueError(f"Unknown modality: {modality!r}")
    forward = modality == "t"
    return {
        (pair["detection_id_t"] if forward else pair["detection_id_w"]):
        (pair["detection_id_w"] if forward else pair["detection_id_t"])
        for pair in detection_pairs(target_folder, run_id)
    }


def partner_tracks(target_folder: str, modality: str,
                   run_id: Optional[int] = None) -> Dict[int, int]:
    """Map each track of *modality* onto the track it was matched with."""
    if modality not in store.MODALITIES:
        raise ValueError(f"Unknown modality: {modality!r}")
    forward = modality == "t"
    return {
        (match["track_id_t"] if forward else match["track_id_w"]):
        (match["track_id_w"] if forward else match["track_id_t"])
        for match in track_matches(target_folder, run_id)
    }


def summary(target_folder: str, tracks_t: int, tracks_w: int,
            run_id: Optional[int] = None) -> Dict[str, int]:
    """Counts in the shape of the paper's Table 2.

    "30 of 94 raw tracks confirmed" is the number that tells a user whether the
    gate is doing its job, so it is computed here rather than left to the UI.
    """
    matches = track_matches(target_folder, run_id)
    confirmed = len(matches)
    return {
        "rgb": tracks_w,
        "thermal": tracks_t,
        "raw": tracks_t + tracks_w,
        "confirmed": confirmed,
        # A confirmed pair accounts for one track on each side.
        "unmatched": tracks_t + tracks_w - 2 * confirmed,
    }
