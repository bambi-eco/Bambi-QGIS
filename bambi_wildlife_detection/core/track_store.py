# -*- coding: utf-8 -*-
"""Geo-referencing and tracking against the 6.0 store.

Implements EXCHANGE_FORMAT_PLAN.md §3.2, §3.3 and §12.2. This is where the
identity model pays for itself:

* geo-referencing keys its results on ``detection_id``, so a detection that
  cannot be placed on the DEM costs *itself* and is recorded in
  ``georef_failures`` with a reason — rather than vanishing and taking the
  positional alignment of its frame with it;
* tracking records membership as ``(track_id, detection_id)`` pairs taken
  straight from the tracker's own output, so ``tracks_pixel`` becomes a join.

That last point is what allows ``core/track_export.py`` to be deleted: it
existed only to reconstruct this linkage by rounding geo coordinates to three
decimals and hashing them, discarding whole frames when the counts disagreed.

Every detection ends up either geo-referenced or explicitly failed —
:func:`accounting` asserts exactly that, which the text format could not express.
"""

import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from . import store

#: Reasons a detection could not be placed on the DEM. These mirror the
#: classifications ``BambiProcessor._classify_projection_miss`` already makes;
#: they were log lines, and are now queryable rows.
GEOREF_REASONS = (
    "above_horizon",     # camera near-horizontal — these pixels are sky
    "beyond_mesh",       # the view reached ground the DEM does not cover
    "no_pose",           # no camera pose for that frame
    "projection_error",  # ray casting raised
    "legacy_invalid",    # negative corner imported from 5.x
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def has_store(target_folder: str, modality: str) -> bool:
    """True when a detections store exists to geo-reference against."""
    return os.path.isfile(
        store.stage_path(target_folder, store.DETECTIONS, modality))


# ---------------------------------------------------------------------------
# Geo-referencing
# ---------------------------------------------------------------------------

def load_detections(target_folder: str, modality: str) -> List[dict]:
    """Detections to geo-reference, each carrying its ``detection_id``.

    Reading the boxes from the store rather than re-parsing ``detections.txt``
    is what removes the ordering assumption: nothing downstream has to line two
    files up row by row.
    """
    path = store.stage_path(target_folder, store.DETECTIONS, modality)
    if not os.path.isfile(path):
        return []
    conn = store.open_store(path, store.DETECTIONS, modality)
    try:
        return [dict(row) for row in conn.execute(
            "SELECT detection_id, frame, x1, y1, x2, y2, confidence, "
            "species_id, source_class FROM detections ORDER BY detection_id")]
    finally:
        conn.close()


def record_georeference(target_folder: str, modality: str,
                        resolved: Iterable[dict],
                        failures: Iterable[dict] = (),
                        log_fn=None) -> Dict[str, int]:
    """Replace this modality's geo-referencing results.

    *resolved* rows need ``detection_id`` and ``gx1``…``gz2``; *failures* need
    ``detection_id`` and ``reason``.
    """
    resolved = list(resolved)
    failures = list(failures)

    conn = store.open_store(
        store.stage_path(target_folder, store.GEOREFERENCED, modality),
        store.GEOREFERENCED, modality)
    try:
        with store.transaction(conn):
            conn.execute("DELETE FROM detections_geo")
            conn.execute("DELETE FROM georef_failures")
            conn.executemany(
                "INSERT INTO detections_geo (detection_id, gx1, gy1, gz1, "
                "gx2, gy2, gz2) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(row["detection_id"], row["gx1"], row["gy1"], row["gz1"],
                  row["gx2"], row["gy2"], row["gz2"]) for row in resolved])
            conn.executemany(
                "INSERT INTO georef_failures (detection_id, reason) "
                "VALUES (?, ?)",
                [(row["detection_id"], row["reason"]) for row in failures])
            store.set_meta(conn, "updated_at", _utc_now())
    finally:
        conn.close()

    if log_fn:
        log_fn(f"Store: {len(resolved)} detection(s) geo-referenced, "
               f"{len(failures)} recorded as failures")
    return {"resolved": len(resolved), "failed": len(failures)}


def load_georeferenced(target_folder: str, modality: str) -> List[dict]:
    """Geo-referenced boxes joined to their detections, ordered by id."""
    geo_path = store.stage_path(target_folder, store.GEOREFERENCED, modality)
    det_path = store.stage_path(target_folder, store.DETECTIONS, modality)
    if not (os.path.isfile(geo_path) and os.path.isfile(det_path)):
        return []

    conn = store.open_store(det_path, store.DETECTIONS, modality)
    try:
        from . import gpkg
        gpkg.attach(conn, geo_path, "geo")
        rows = [dict(row) for row in conn.execute(
            "SELECT d.detection_id, d.frame, d.confidence, d.species_id, "
            "g.gx1, g.gy1, g.gz1, g.gx2, g.gy2, g.gz2 "
            "FROM detections d JOIN geo.detections_geo g USING (detection_id) "
            "ORDER BY d.frame, d.detection_id")]
        gpkg.detach(conn, "geo")
        return rows
    finally:
        conn.close()


def failure_summary(target_folder: str, modality: str) -> Dict[str, int]:
    """Geo-referencing failures per reason."""
    path = store.stage_path(target_folder, store.GEOREFERENCED, modality)
    if not os.path.isfile(path):
        return {}
    conn = store.open_store(path, store.GEOREFERENCED, modality)
    try:
        return {row["reason"]: int(row["n"]) for row in conn.execute(
            "SELECT reason, COUNT(*) AS n FROM georef_failures GROUP BY reason")}
    finally:
        conn.close()


def accounting(target_folder: str, modality: str) -> Dict[str, object]:
    """Check every detection is either geo-referenced or explicitly failed.

    Returns ``{"detections": n, "resolved": n, "failed": n, "unaccounted":
    [ids], "both": [ids]}``. ``unaccounted`` being non-empty means detections
    went missing silently — the failure mode 6.0 exists to remove — and ``both``
    means a detection was recorded as resolved *and* failed.
    """
    det_path = store.stage_path(target_folder, store.DETECTIONS, modality)
    geo_path = store.stage_path(target_folder, store.GEOREFERENCED, modality)
    if not os.path.isfile(det_path):
        return {"detections": 0, "resolved": 0, "failed": 0,
                "unaccounted": [], "both": []}

    conn = store.open_store(det_path, store.DETECTIONS, modality)
    try:
        all_ids = {row["detection_id"] for row in conn.execute(
            "SELECT detection_id FROM detections")}
    finally:
        conn.close()

    resolved: set = set()
    failed: set = set()
    if os.path.isfile(geo_path):
        conn = store.open_store(geo_path, store.GEOREFERENCED, modality)
        try:
            resolved = {row["detection_id"] for row in conn.execute(
                "SELECT detection_id FROM detections_geo")}
            failed = {row["detection_id"] for row in conn.execute(
                "SELECT detection_id FROM georef_failures")}
        finally:
            conn.close()

    return {
        "detections": len(all_ids),
        "resolved": len(resolved),
        "failed": len(failed),
        "unaccounted": sorted(all_ids - resolved - failed),
        "both": sorted(resolved & failed),
    }


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def record_tracks(target_folder: str, modality: str,
                  members: Iterable[dict], kind: str = "builtin",
                  tracker: str = "", version: str = "",
                  config_hash: str = "", species_of=None,
                  log_fn=None) -> Dict[str, int]:
    """Record one tracking run.

    *members* are ``{"track_id": legacy id, "detection_id": id,
    "interpolated": 0|1}`` — the tracker's own output, with no re-derivation of
    which detection belongs to which track.

    A new ``track_runs`` row is created and made active; earlier runs are kept
    but deactivated, so comparing trackers stays possible and re-running one
    never disturbs another (§3.3).
    """
    members = list(members)

    conn = store.open_store(
        store.stage_path(target_folder, store.TRACKS, modality),
        store.TRACKS, modality)
    try:
        with store.transaction(conn):
            conn.execute("UPDATE track_runs SET is_active = 0")
            generation = conn.execute(
                "SELECT IFNULL(MAX(generation), 0) + 1 AS g FROM track_runs "
                "WHERE kind = ?", (kind,)).fetchone()["g"]
            cur = conn.execute(
                "INSERT INTO track_runs (kind, tracker, version, generation, "
                "config_hash, created_at, is_active) VALUES (?, ?, ?, ?, ?, ?, 1)",
                (kind, tracker, version, generation, config_hash, _utc_now()))
            run_id = int(cur.lastrowid)

            track_ids: Dict[int, int] = {}
            for member in members:
                legacy = member["track_id"]
                if legacy not in track_ids:
                    species_id = 0 if species_of is None else species_of(legacy)
                    new = conn.execute(
                        "INSERT INTO tracks (run_id, species_id) VALUES (?, ?)",
                        (run_id, species_id))
                    track_ids[legacy] = int(new.lastrowid)
                conn.execute(
                    "INSERT OR IGNORE INTO track_members "
                    "(track_id, detection_id, interpolated) VALUES (?, ?, ?)",
                    (track_ids[legacy], member["detection_id"],
                     member.get("interpolated", 0)))
    finally:
        conn.close()

    if log_fn:
        log_fn(f"Store: recorded {len(track_ids)} track(s) with "
               f"{len(members)} member(s) as run {run_id} ({kind})")
    return {"run_id": run_id, "tracks": len(track_ids),
            "members": len(members), "generation": generation}


def superseded_track_ids(target_folder: str, modality: str) -> set:
    """Tracker tracks that a label track was imported from, and now replaces.

    "Import as label track" copies a pipeline track into an editable label
    track; once that label track is materialised, both describe the same
    animal. Counting them together would double it, so the original is
    superseded — but *only* that one. A label track drawn from scratch has no
    ``origin_track_id`` and supersedes nothing (§8.2).
    """
    path = store.stage_path(target_folder, store.LABELS, modality)
    if not os.path.isfile(path):
        return set()

    conn = store.open_store(path, store.LABELS, modality)
    try:
        return {int(row["origin_track_id"]) for row in conn.execute(
            "SELECT origin_track_id FROM label_tracks "
            "WHERE origin_track_id IS NOT NULL AND track_id IS NOT NULL")}
    finally:
        conn.close()


def manual_run(target_folder: str, modality: str) -> Optional[dict]:
    """The labelling tool's run, if it has materialised anything."""
    path = store.stage_path(target_folder, store.TRACKS, modality)
    if not os.path.isfile(path):
        return None
    conn = store.open_store(path, store.TRACKS, modality)
    try:
        row = conn.execute(
            "SELECT run_id, kind, tracker, generation, created_at "
            "FROM track_runs WHERE kind = 'manual' "
            "ORDER BY run_id DESC LIMIT 1").fetchone()
        return None if row is None else dict(row)
    finally:
        conn.close()


def analysis_runs(target_folder: str, modality: str,
                  include_manual: bool = True) -> List[int]:
    """Run ids the layers and analytics should read together.

    Tracker runs are *alternatives* — builtin, boxmot and TRex describe the same
    animals differently — so exactly one is active at a time. The labelling
    tool's run is *additive*: its tracks are usually animals the detector
    missed, so it is pooled alongside rather than chosen between. Tracker tracks
    that a label track superseded are excluded by
    :func:`superseded_track_ids`, which is what keeps the pooling honest.
    """
    runs = []
    active = active_run(target_folder, modality)
    if active is not None:
        runs.append(active["run_id"])
    if include_manual:
        manual = manual_run(target_folder, modality)
        if manual is not None and manual["run_id"] not in runs:
            runs.append(manual["run_id"])
    return runs


def active_run(target_folder: str, modality: str) -> Optional[dict]:
    """The tracking run the layers and analytics should use."""
    path = store.stage_path(target_folder, store.TRACKS, modality)
    if not os.path.isfile(path):
        return None
    conn = store.open_store(path, store.TRACKS, modality)
    try:
        row = conn.execute(
            "SELECT run_id, kind, tracker, generation, created_at "
            "FROM track_runs WHERE is_active = 1 AND kind <> 'manual' "
            "ORDER BY run_id DESC LIMIT 1").fetchone()
        return None if row is None else dict(row)
    finally:
        conn.close()


def load_pixel_tracks(target_folder: str, modality: str,
                      run_id: Optional[int] = None) -> List[dict]:
    """Pixel-space tracks — the join that replaces ``core/track_export.py``.

    No coordinate rounding, no per-frame ordering assumption, and a detection
    that failed to geo-reference no longer costs its frame-mates their tracks.
    """
    det_path = store.stage_path(target_folder, store.DETECTIONS, modality)
    trk_path = store.stage_path(target_folder, store.TRACKS, modality)
    if not (os.path.isfile(det_path) and os.path.isfile(trk_path)):
        return []

    if run_id is None:
        run_ids = analysis_runs(target_folder, modality)
    else:
        run_ids = [run_id]
    if not run_ids:
        return []

    superseded = superseded_track_ids(target_folder, modality)
    placeholders = ", ".join("?" for _ in run_ids)

    conn = store.open_store(det_path, store.DETECTIONS, modality)
    try:
        from . import gpkg
        gpkg.attach(conn, trk_path, "trk")
        rows = [dict(row) for row in conn.execute(
            "SELECT m.track_id, d.detection_id, d.frame, d.x1, d.y1, d.x2, "
            "d.y2, d.confidence, d.species_id, m.interpolated "
            "FROM trk.track_members m "
            "JOIN trk.tracks t ON t.track_id = m.track_id "
            "JOIN detections d ON d.detection_id = m.detection_id "
            f"WHERE t.run_id IN ({placeholders}) "  # nosec B608 — ints only
            "ORDER BY d.frame, m.track_id", run_ids)]
        gpkg.detach(conn, "trk")
        return [row for row in rows if row["track_id"] not in superseded]
    finally:
        conn.close()


def track_orphans(target_folder: str, modality: str) -> List[int]:
    """Track members pointing at a detection that no longer exists.

    Should always be empty: a producer may only delete rows it owns, and
    deleting detections invalidates the tracking stage (§7).
    """
    det_path = store.stage_path(target_folder, store.DETECTIONS, modality)
    trk_path = store.stage_path(target_folder, store.TRACKS, modality)
    if not (os.path.isfile(det_path) and os.path.isfile(trk_path)):
        return []

    conn = store.open_store(det_path, store.DETECTIONS, modality)
    try:
        from . import gpkg
        gpkg.attach(conn, trk_path, "trk")
        rows = [int(row["detection_id"]) for row in conn.execute(
            "SELECT m.detection_id FROM trk.track_members m "
            "LEFT JOIN detections d ON d.detection_id = m.detection_id "
            "WHERE d.detection_id IS NULL")]
        gpkg.detach(conn, "trk")
        return sorted(set(rows))
    finally:
        conn.close()
