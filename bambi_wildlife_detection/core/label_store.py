# -*- coding: utf-8 -*-
"""Label persistence and materialisation against the 6.0 store.

Implements EXCHANGE_FORMAT_PLAN.md §6. Two jobs:

* **persistence** — the sparse key frames go to ``labels.gpkg`` instead of
  ``labels.json``, with species and enum values stored as ids;
* **materialisation** — the interpolated boxes are upserted into ``detections``
  under the labelling tool's own producer, and their grouping into ``tracks``
  under its own run.

The upsert is the delicate part (§6.2). Labels are *generated*: every export
re-derives every box from the key frames. If that re-minted ``detection_id``s,
editing one key frame would invalidate geo-referencing and tracking for the
whole flight — and labelling is inherently iterative, so that would make the
tool unusable. Keying the upsert on ``(label_track_id, frame)`` means:

===========================  ==================================================
unchanged ``(track, frame)`` keeps its ``detection_id``
box moved                    keeps its id, and **loses its geo row** so
                             geo-referencing redoes just that detection
frame newly covered          mints a new id
frame no longer covered      the row is deleted
===========================  ==================================================

Re-running geo-referencing therefore becomes a left join rather than a rebuild.
Tracking cannot be incremental — it is sequential and global — so it is fully
invalidated instead, and that asymmetry is deliberate.
"""

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

from . import store

#: The labelling tool's producer and run names. It owns only rows carrying
#: these, so a detector or tracker re-run never touches its work, and vice
#: versa (§4.1).
MANUAL_KIND = "manual"

#: Confidence recorded for a hand-drawn box.
MANUAL_CONFIDENCE = 1.0


#: Base classes in the order they should be offered, rather than by id — the
#: ids are ``0, -1, -2`` and ascending order would read oddly in a combo box.
_BASE_SPECIES_ORDER = ("animal", "unknown", "not-an-animal")


def vocabulary(target_folder: str) -> dict:
    """The project's species and enum values, ordered for display.

    Returns ``{}`` when the project has no 6.0 store, which is what lets the
    labelling tool fall back to its built-in lists on an un-migrated folder.
    """
    path = store.project_path(target_folder)
    if not os.path.isfile(path):
        return {}

    conn = store.open_store(path, store.PROJECT)
    try:
        rows = [dict(row) for row in conn.execute(
            "SELECT species_id, name, protected FROM species")]
        base = [r for name in _BASE_SPECIES_ORDER
                for r in rows if r["name"] == name]
        concrete = sorted((r for r in rows if not r["protected"]),
                          key=lambda r: r["species_id"])
        species = base + concrete

        enums: Dict[str, List[dict]] = {}
        for row in conn.execute(
                "SELECT e.name AS enum, v.label AS label, "
                "v.value_id AS value_id FROM enums e "
                "JOIN enum_values v USING (enum_id) "
                "ORDER BY e.name, v.ordinal, v.value_id"):
            enums.setdefault(row["enum"], []).append(
                {"value_id": row["value_id"], "label": row["label"]})

        return {
            "species": species,
            "species_by_name": {r["name"]: r["species_id"] for r in species},
            "enums": enums,
            "enum_ids": {name: {v["label"]: v["value_id"] for v in values}
                         for name, values in enums.items()},
        }
    finally:
        conn.close()


def _labels_conn(target_folder: str, modality: str):
    return store.open_store(
        store.stage_path(target_folder, store.LABELS, modality),
        store.LABELS, modality)


def _detections_conn(target_folder: str, modality: str):
    return store.open_store(
        store.stage_path(target_folder, store.DETECTIONS, modality),
        store.DETECTIONS, modality)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_tracks(target_folder: str, modality: str, tracks: Iterable[dict],
                log_fn=None) -> int:
    """Replace this modality's label tracks.

    Each track is ``{"label_track_id", "species_id", "attributes",
    "origin_track_id"?, "keyframes": [{"frame", "x1".."y2", "stop"?,
    "attributes"?, "origin_detection_id"?}]}``.
    """
    tracks = list(tracks)
    conn = _labels_conn(target_folder, modality)
    try:
        with store.transaction(conn):
            keep = [t["label_track_id"] for t in tracks]
            existing = [row["label_track_id"] for row in conn.execute(
                "SELECT label_track_id FROM label_tracks")]
            for label_track_id in existing:
                if label_track_id not in keep:
                    conn.execute(
                        "DELETE FROM label_keyframes WHERE label_track_id = ?",
                        (label_track_id,))
                    conn.execute(
                        "DELETE FROM label_tracks WHERE label_track_id = ?",
                        (label_track_id,))

            for track in tracks:
                conn.execute(
                    "INSERT OR REPLACE INTO label_tracks (label_track_id, "
                    "species_id, origin_track_id, track_id, attributes) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (track["label_track_id"],
                     track.get("species_id", store.FALLBACK_SPECIES_ID),
                     track.get("origin_track_id"),
                     track.get("track_id"),
                     json.dumps(track.get("attributes") or {})))
                conn.execute(
                    "DELETE FROM label_keyframes WHERE label_track_id = ?",
                    (track["label_track_id"],))
                conn.executemany(
                    "INSERT INTO label_keyframes (label_track_id, frame, "
                    "x1, y1, x2, y2, stop, origin_detection_id, attributes) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(track["label_track_id"], kf["frame"],
                      kf["x1"], kf["y1"], kf["x2"], kf["y2"],
                      1 if kf.get("stop") else 0,
                      kf.get("origin_detection_id"),
                      json.dumps(kf.get("attributes") or {}))
                     for kf in track.get("keyframes", [])])
    finally:
        conn.close()

    if log_fn:
        log_fn(f"Store: saved {len(tracks)} label track(s)")
    return len(tracks)


def load_tracks(target_folder: str, modality: str) -> List[dict]:
    """Load this modality's label tracks, key frames ordered by frame."""
    path = store.stage_path(target_folder, store.LABELS, modality)
    if not os.path.isfile(path):
        return []

    conn = _labels_conn(target_folder, modality)
    try:
        tracks = []
        for row in conn.execute(
                "SELECT label_track_id, species_id, origin_track_id, track_id, "
                "attributes FROM label_tracks ORDER BY label_track_id"):
            track = dict(row)
            track["attributes"] = _load_json(row["attributes"])
            track["keyframes"] = [
                dict(kf, attributes=_load_json(kf["attributes"]))
                for kf in conn.execute(
                    "SELECT frame, x1, y1, x2, y2, stop, origin_detection_id, "
                    "attributes FROM label_keyframes WHERE label_track_id = ? "
                    "ORDER BY frame", (row["label_track_id"],))]
            tracks.append(track)
        return tracks
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


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interpolate(keyframes: List[dict]) -> List[dict]:
    """Expand sparse key frames into a box per covered frame.

    Mirrors ``LabelTrack.box_at``: linear between key frames, and no box at all
    between a frame flagged ``stop`` and the next key frame.
    """
    if not keyframes:
        return []
    by_frame = {kf["frame"]: kf for kf in keyframes}
    frames = sorted(by_frame)

    rows = []
    for frame in range(frames[0], frames[-1] + 1):
        if frame in by_frame:
            kf = by_frame[frame]
            rows.append({"frame": frame, "x1": kf["x1"], "y1": kf["y1"],
                         "x2": kf["x2"], "y2": kf["y2"], "keyframe": True,
                         "attributes": dict(kf.get("attributes") or {})})
            continue
        previous = max(f for f in frames if f < frame)
        following = min(f for f in frames if f > frame)
        before = by_frame[previous]
        if before.get("stop"):
            continue  # the animal disappeared here
        after = by_frame[following]
        alpha = (frame - previous) / (following - previous)
        rows.append({
            "frame": frame,
            "x1": before["x1"] + alpha * (after["x1"] - before["x1"]),
            "y1": before["y1"] + alpha * (after["y1"] - before["y1"]),
            "x2": before["x2"] + alpha * (after["x2"] - before["x2"]),
            "y2": before["y2"] + alpha * (after["y2"] - before["y2"]),
            "keyframe": False,
            "attributes": dict(before.get("attributes") or {}),
        })
    return rows


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------

def _manual_source_id(target_folder: str) -> int:
    project = store.open_store(
        store.project_path(target_folder), store.PROJECT)
    try:
        with store.transaction(project):
            from .detection_store import ensure_source
            return ensure_source(project, MANUAL_KIND)
    finally:
        project.close()


def _box_changed(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
    return any(abs(float(x) - float(y)) > 1e-6 for x, y in zip(a, b))


def materialise(target_folder: str, modality: str,
                tracks: Optional[List[dict]] = None,
                write_tracks: bool = True, log_fn=None) -> Dict[str, int]:
    """Upsert the label boxes into ``detections`` and ``tracks`` (§6.2, §6.5).

    Returns counts of ``kept`` / ``moved`` / ``added`` / ``removed`` detections
    plus the number of tracks written. ``moved`` rows have had their geo row
    dropped, so geo-referencing will redo exactly those.
    """
    if tracks is None:
        tracks = load_tracks(target_folder, modality)

    source_id = _manual_source_id(target_folder)

    desired: Dict[Tuple[int, int], dict] = {}
    for track in tracks:
        for row in interpolate(track.get("keyframes", [])):
            desired[(track["label_track_id"], row["frame"])] = {
                "x1": row["x1"], "y1": row["y1"],
                "x2": row["x2"], "y2": row["y2"],
                "species_id": track.get("species_id",
                                        store.FALLBACK_SPECIES_ID),
                "attributes": row["attributes"],
            }

    kept = moved = added = removed = 0
    invalidated: List[int] = []
    detection_of: Dict[Tuple[int, int], int] = {}

    conn = _detections_conn(target_folder, modality)
    try:
        with store.transaction(conn):
            existing = {}
            for row in conn.execute(
                    "SELECT detection_id, label_track_id, frame, x1, y1, x2, "
                    "y2, species_id, attributes FROM detections "
                    "WHERE label_track_id IS NOT NULL"):
                existing[(row["label_track_id"], row["frame"])] = dict(row)

            for key, want in desired.items():
                have = existing.get(key)
                attributes = json.dumps(want["attributes"])
                if have is None:
                    cur = conn.execute(
                        "INSERT INTO detections (frame, x1, y1, x2, y2, "
                        "confidence, species_id, source_id, source_class, "
                        "label_track_id, attributes) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                        (key[1], want["x1"], want["y1"], want["x2"],
                         want["y2"], MANUAL_CONFIDENCE, want["species_id"],
                         source_id, key[0], attributes))
                    detection_of[key] = int(cur.lastrowid)
                    added += 1
                    continue

                detection_of[key] = have["detection_id"]
                box_moved = _box_changed(
                    (have["x1"], have["y1"], have["x2"], have["y2"]),
                    (want["x1"], want["y1"], want["x2"], want["y2"]))
                species_changed = have["species_id"] != want["species_id"]
                attributes_changed = (
                    _load_json(have["attributes"]) != want["attributes"])
                metadata_changed = species_changed or attributes_changed

                if box_moved or metadata_changed:
                    conn.execute(
                        "UPDATE detections SET x1 = ?, y1 = ?, x2 = ?, y2 = ?, "
                        "species_id = ?, attributes = ? WHERE detection_id = ?",
                        (want["x1"], want["y1"], want["x2"], want["y2"],
                         want["species_id"], attributes, have["detection_id"]))
                if box_moved:
                    # The geometry changed, so its world position is stale.
                    invalidated.append(have["detection_id"])
                    moved += 1
                else:
                    kept += 1

            for key, have in existing.items():
                if key in desired:
                    continue
                conn.execute("DELETE FROM detections WHERE detection_id = ?",
                             (have["detection_id"],))
                invalidated.append(have["detection_id"])
                removed += 1
    finally:
        conn.close()

    _invalidate_geo(target_folder, modality, invalidated)

    written_tracks = 0
    if write_tracks:
        written_tracks = _write_manual_tracks(
            target_folder, modality, tracks, detection_of, log_fn)

    if log_fn:
        log_fn(f"Store: {added} added, {kept} unchanged, {moved} moved, "
               f"{removed} removed; {len(invalidated)} detection(s) need "
               "re-geo-referencing")
    return {"kept": kept, "moved": moved, "added": added, "removed": removed,
            "invalidated": len(invalidated), "tracks": written_tracks}


def _invalidate_geo(target_folder: str, modality: str,
                    detection_ids: List[int]) -> None:
    """Drop the geo rows of detections whose box moved or vanished.

    Geo-referencing selects detections with neither a geo row nor a recorded
    failure, so removing these is what makes the re-run incremental (§6.2).
    """
    if not detection_ids:
        return
    path = store.stage_path(target_folder, store.GEOREFERENCED, modality)
    if not os.path.isfile(path):
        return

    conn = store.open_store(path, store.GEOREFERENCED, modality)
    try:
        with store.transaction(conn):
            conn.executemany(
                "DELETE FROM detections_geo WHERE detection_id = ?",
                [(i,) for i in detection_ids])
            conn.executemany(
                "DELETE FROM georef_failures WHERE detection_id = ?",
                [(i,) for i in detection_ids])
    finally:
        conn.close()


def _write_manual_tracks(target_folder: str, modality: str,
                         tracks: List[dict],
                         detection_of: Dict[Tuple[int, int], int],
                         log_fn=None) -> int:
    """Record the label tracks as tracks under the labelling tool's own run.

    A label track *is* a track — its membership is given by the annotator
    rather than computed — so downstream consumers stop special-casing manual
    data (§6.5). No association logic happens here (§6.6).
    """
    conn = store.open_store(
        store.stage_path(target_folder, store.TRACKS, modality),
        store.TRACKS, modality)
    try:
        with store.transaction(conn):
            previous = [int(row["run_id"]) for row in conn.execute(
                "SELECT run_id FROM track_runs WHERE kind = ?", (MANUAL_KIND,))]
            for run_id in previous:
                conn.execute(
                    "DELETE FROM track_members WHERE track_id IN "
                    "(SELECT track_id FROM tracks WHERE run_id = ?)", (run_id,))
                conn.execute("DELETE FROM tracks WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM track_runs WHERE run_id = ?", (run_id,))

            if not tracks:
                return 0

            generation = conn.execute(
                "SELECT IFNULL(MAX(generation), 0) + 1 AS g FROM track_runs "
                "WHERE kind = ?", (MANUAL_KIND,)).fetchone()["g"]
            cur = conn.execute(
                "INSERT INTO track_runs (kind, tracker, generation, "
                "created_at, is_active) VALUES (?, 'labelling tool', ?, '', 0)",
                (MANUAL_KIND, generation))
            run_id = int(cur.lastrowid)

            written = 0
            for track in tracks:
                members = [detection_id
                           for (label_track_id, _frame), detection_id
                           in detection_of.items()
                           if label_track_id == track["label_track_id"]]
                if not members:
                    continue
                new = conn.execute(
                    "INSERT INTO tracks (run_id, species_id, attributes) "
                    "VALUES (?, ?, ?)",
                    (run_id, track.get("species_id", store.FALLBACK_SPECIES_ID),
                     json.dumps(track.get("attributes") or {})))
                track_id = int(new.lastrowid)
                conn.executemany(
                    "INSERT OR IGNORE INTO track_members "
                    "(track_id, detection_id, interpolated) VALUES (?, ?, 0)",
                    [(track_id, detection_id) for detection_id in members])
                written += 1
    finally:
        conn.close()

    _record_manual_track_ids(target_folder, modality)
    if log_fn and written:
        log_fn(f"Store: recorded {written} manual track(s)")
    return written


def _record_manual_track_ids(target_folder: str, modality: str) -> None:
    """Link each label track to the manual track it produced."""
    from . import gpkg

    labels_path = store.stage_path(target_folder, store.LABELS, modality)
    tracks_path = store.stage_path(target_folder, store.TRACKS, modality)
    det_path = store.stage_path(target_folder, store.DETECTIONS, modality)
    if not all(os.path.isfile(p) for p in (labels_path, tracks_path, det_path)):
        return

    conn = _labels_conn(target_folder, modality)
    try:
        gpkg.attach(conn, tracks_path, "trk")
        gpkg.attach(conn, det_path, "det")
        pairs = conn.execute(
            "SELECT DISTINCT d.label_track_id AS label_track_id, "
            "m.track_id AS track_id FROM det.detections d "
            "JOIN trk.track_members m ON m.detection_id = d.detection_id "
            "JOIN trk.tracks t ON t.track_id = m.track_id "
            "JOIN trk.track_runs r ON r.run_id = t.run_id "
            "WHERE r.kind = ? AND d.label_track_id IS NOT NULL",
            (MANUAL_KIND,)).fetchall()
        with store.transaction(conn):
            for row in pairs:
                conn.execute(
                    "UPDATE label_tracks SET track_id = ? "
                    "WHERE label_track_id = ?",
                    (row["track_id"], row["label_track_id"]))
        gpkg.detach(conn, "trk")
        gpkg.detach(conn, "det")
    finally:
        conn.close()


def clear_materialised(target_folder: str, modality: str,
                       log_fn=None) -> int:
    """Remove every detection the labelling tool produced.

    The destructive sibling of :func:`materialise`, and the store equivalent of
    "Replace detections in project" — except that it removes only the tool's own
    rows, so what happens to the detector's output is a separate, explicit
    decision rather than a side effect.
    """
    return materialise(target_folder, modality, tracks=[],
                       log_fn=log_fn)["removed"]


def clear_detector_detections(target_folder: str, modality: str) -> int:
    """Delete every detection *not* produced by the labelling tool.

    Together with :func:`materialise` this is "Replace detections in project":
    the marker-block rewrite of 5.x becomes one delete plus the normal upsert.
    """
    conn = _detections_conn(target_folder, modality)
    try:
        with store.transaction(conn):
            cur = conn.execute(
                "DELETE FROM detections WHERE label_track_id IS NULL")
            return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    finally:
        conn.close()
