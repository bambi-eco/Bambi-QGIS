# -*- coding: utf-8 -*-
"""Removing wrong detections and tracks from a project.

The inspector shows a frame with its boxes; a reviewer looking at one can see
that it is a rock, a shadow or a duplicate. This module is what "delete" does
then: it removes the detection (or the whole track) from every store that
knows about it, so no stage is left pointing at something that no longer
exists - a geo row without its detection, a track member without its box, a
classifier vote for an animal that is gone. ``track_store.track_orphans``
stays empty afterwards, which is the invariant this module protects.

What a deletion touches:

* ``detections``, ``detections_geo``, ``georef_failures`` - the box itself
  and where it was placed (or why it was not);
* ``track_members`` of *every* run, active or not, and tracks left without a
  member; a track's own row, its classifier votes (``track_predictions``)
  and its cross-modal match (``track_matches`` + ``detection_matches``);
* ``embeddings`` and ``frame_predictions`` - the vector on disk is left in
  the ``.npz`` (it is harmless there) but the store no longer claims it;
* ``segments`` cut from the detection;
* a label track that was materialised into the deleted track: the track
  and the annotation describe the same animal, so deleting the animal
  deletes both. Anything less would re-materialise the track on the next
  save in the labelling tool.

One detection *inside* a label track is different: its box is defined by
the key frames, and removing the row would only have it come back on the
next save. Those are edited in the labelling tool, and
:func:`delete_detections` refuses them with a message saying so.

Deletions are permanent - the store keeps no history of removed rows. The
callers confirm with the user first.

Nothing here imports QGIS; the layer clean-up lives in
``bambi_layer_sync`` on the plugin side.
"""
import os
from typing import Dict, Iterable, List, Optional, Set

from . import store


class ReviewError(RuntimeError):
    """A deletion the store cannot honour as asked."""


def _placeholders(values) -> str:
    return ", ".join("?" for _ in values)


def _rowcount(cursor) -> int:
    return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0


def _open(target_folder: str, kind: str, modality: str):
    """The stage store, or ``None`` when the stage never ran."""
    path = store.stage_path(target_folder, kind, modality)
    if not os.path.isfile(path):
        return None
    return store.open_store(path, kind, modality)


# ---------------------------------------------------------------------------
# Looking before deleting
# ---------------------------------------------------------------------------

def label_tracks_of(target_folder: str, modality: str,
                    detection_ids: Iterable[int]) -> Dict[int, int]:
    """``{detection_id: label_track_id}`` for the detections the labelling
    tool materialised; detector detections are absent."""
    ids = [int(i) for i in detection_ids]
    conn = _open(target_folder, store.DETECTIONS, modality)
    if conn is None or not ids:
        return {}
    try:
        return {int(row["detection_id"]): int(row["label_track_id"])
                for row in conn.execute(
                    "SELECT detection_id, label_track_id FROM detections "
                    f"WHERE detection_id IN ({_placeholders(ids)}) "  # nosec B608
                    "AND label_track_id IS NOT NULL", ids)}
    finally:
        conn.close()


def track_members(target_folder: str, modality: str,
                  track_id: int) -> List[int]:
    """The detection ids of one track, in store order."""
    conn = _open(target_folder, store.TRACKS, modality)
    if conn is None:
        return []
    try:
        return [int(row["detection_id"]) for row in conn.execute(
            "SELECT detection_id FROM track_members WHERE track_id = ? "
            "ORDER BY detection_id", (int(track_id),))]
    finally:
        conn.close()


def describe_track_deletion(target_folder: str, modality: str,
                            track_id: int) -> dict:
    """What deleting *track_id* would take with it, for a confirmation.

    ``{"detections": n, "matched": bool, "label_track_ids": [...],
    "exists": bool}``.
    """
    members = track_members(target_folder, modality, track_id)
    exists = False
    conn = _open(target_folder, store.TRACKS, modality)
    if conn is not None:
        try:
            exists = conn.execute(
                "SELECT 1 FROM tracks WHERE track_id = ?",
                (int(track_id),)).fetchone() is not None
        finally:
            conn.close()
    return {
        "exists": exists,
        "detections": len(members),
        "matched": _is_matched(target_folder, modality, track_id),
        "label_track_ids": _materialised_label_tracks(
            target_folder, modality, [track_id]),
    }


def _is_matched(target_folder: str, modality: str, track_id: int) -> bool:
    path = store.matches_path(target_folder)
    if not os.path.isfile(path):
        return False
    conn = store.open_store(path, store.MATCHES)
    try:
        column = "track_id_t" if modality == "t" else "track_id_w"
        return conn.execute(
            f"SELECT 1 FROM track_matches WHERE {column} = ? LIMIT 1",  # nosec B608
            (int(track_id),)).fetchone() is not None
    finally:
        conn.close()


def _materialised_label_tracks(target_folder: str, modality: str,
                               track_ids: Iterable[int]) -> List[int]:
    ids = [int(i) for i in track_ids]
    conn = _open(target_folder, store.LABELS, modality)
    if conn is None or not ids:
        return []
    try:
        return [int(row["label_track_id"]) for row in conn.execute(
            "SELECT label_track_id FROM label_tracks "
            f"WHERE track_id IN ({_placeholders(ids)}) "  # nosec B608
            "ORDER BY label_track_id", ids)]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Deleting
# ---------------------------------------------------------------------------

def delete_detections(target_folder: str, modality: str,
                      detection_ids: Iterable[int],
                      log_fn=None) -> Dict[str, object]:
    """Remove detections and everything recorded about them.

    A track that loses its last member is removed too (with its votes and
    its match), and reported under ``tracks_removed``. Detections that a
    label track materialised are refused with :class:`ReviewError` before
    anything is touched - see the module docstring.

    Returns ``{"detections": n, "tracks_removed": [ids], "tracks_kept":
    [ids]}``; ``tracks_kept`` are tracks that lost a member but survive.
    """
    if modality not in store.MODALITIES:
        raise ValueError(f"Unknown modality: {modality!r}")
    ids = sorted({int(i) for i in detection_ids})
    if not ids:
        return {"detections": 0, "tracks_removed": [], "tracks_kept": []}

    labelled = label_tracks_of(target_folder, modality, ids)
    if labelled:
        detection_id, label_track_id = next(iter(sorted(labelled.items())))
        raise ReviewError(
            f"Detection {detection_id} is a box of label track "
            f"{label_track_id}, drawn in the labelling tool. Remove or edit "
            "it there: deleting it here would only have it come back on the "
            "next save.")

    result = _delete_detection_rows(target_folder, modality, ids)
    mark_reports_stale(target_folder, modality)
    if log_fn:
        removed = result["tracks_removed"]
        note = (f", {len(removed)} track(s) left empty and removed"
                if removed else "")
        log_fn(f"Store: deleted {result['detections']} detection(s) "
               f"({modality}){note}")
    return result


def delete_tracks(target_folder: str, modality: str,
                  track_ids: Iterable[int], log_fn=None) -> Dict[str, object]:
    """Remove tracks with their detections and everything derived from them.

    Every member detection goes (through :func:`delete_detections`' cascade,
    without its label-track guard: a materialised label track is deleted
    along with the track, which is the one case where removing its boxes
    is right). Returns ``{"tracks": n, "detections": n,
    "label_tracks": [ids]}``.
    """
    if modality not in store.MODALITIES:
        raise ValueError(f"Unknown modality: {modality!r}")
    ids = sorted({int(i) for i in track_ids})
    if not ids:
        return {"tracks": 0, "detections": 0, "label_tracks": []}

    members: Set[int] = set()
    for track_id in ids:
        members.update(track_members(target_folder, modality, track_id))

    label_ids = _materialised_label_tracks(target_folder, modality, ids)
    _delete_label_tracks(target_folder, modality, label_ids)

    detections = 0
    if members:
        detections = _delete_detection_rows(
            target_folder, modality, sorted(members))["detections"]
    # Tracks without a member (nothing to cascade from) and the ones the
    # detection cascade already took: removing again is a no-op.
    removed = _delete_track_rows(target_folder, modality, ids)
    mark_reports_stale(target_folder, modality)

    if log_fn:
        note = (f", {len(label_ids)} label track(s) with them"
                if label_ids else "")
        log_fn(f"Store: deleted {len(ids)} track(s) ({modality}) with "
               f"{detections} detection(s){note}")
    return {"tracks": len(ids), "detections": detections,
            "label_tracks": label_ids, "removed": removed}


#: Stages whose written output is a count or a report over the tracks, and
#: so is wrong the moment one is deleted. The per-animal results
#: (classification, matching) stay valid for every other animal and are
#: cleaned row-wise by the cascade instead.
REPORT_STAGES = ("track_inventory", "density", "population",
                 "perpendicular", "track_perpendicular")


def mark_reports_stale(target_folder: str, modality: str) -> List[str]:
    """Flag the aggregate stages as stale after a deletion.

    Only stages recorded as complete are touched; the files stay on disk,
    as the stage machinery's "stale is flagged, not deleted" rule has it.
    Returns the stages that changed.
    """
    from . import stages

    if not os.path.isfile(store.project_path(target_folder)):
        return []
    current = stages.states(target_folder, modality)
    changed = []
    for stage in REPORT_STAGES:
        row = current.get(stage)
        if row is None or row.get("state") != stages.COMPLETE:
            continue
        stages.set_state(target_folder, stage, modality, stages.STALE)
        changed.append(stage)
    return changed


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------

def _delete_detection_rows(target_folder: str, modality: str,
                           ids: List[int]) -> Dict[str, object]:
    marks = _placeholders(ids)

    conn = _open(target_folder, store.DETECTIONS, modality)
    deleted = 0
    if conn is not None:
        try:
            with store.transaction(conn):
                deleted = _rowcount(conn.execute(
                    f"DELETE FROM detections WHERE detection_id IN ({marks})",  # nosec B608
                    ids))
        finally:
            conn.close()

    conn = _open(target_folder, store.GEOREFERENCED, modality)
    if conn is not None:
        try:
            with store.transaction(conn):
                conn.execute(
                    f"DELETE FROM detections_geo WHERE detection_id IN ({marks})",  # nosec B608
                    ids)
                conn.execute(
                    f"DELETE FROM georef_failures WHERE detection_id IN ({marks})",  # nosec B608
                    ids)
        finally:
            conn.close()

    # Track membership, in every run: an inactive run pointing at a deleted
    # detection would be an orphan the moment it is activated again.
    touched: List[int] = []
    emptied: List[int] = []
    conn = _open(target_folder, store.TRACKS, modality)
    if conn is not None:
        try:
            with store.transaction(conn):
                touched = sorted({int(row["track_id"]) for row in conn.execute(
                    "SELECT DISTINCT track_id FROM track_members "
                    f"WHERE detection_id IN ({marks})", ids)})  # nosec B608
                conn.execute(
                    f"DELETE FROM track_members WHERE detection_id IN ({marks})",  # nosec B608
                    ids)
                if touched:
                    emptied = sorted({int(row["track_id"]) for row in conn.execute(
                        "SELECT t.track_id FROM tracks t "
                        "LEFT JOIN track_members m ON m.track_id = t.track_id "
                        f"WHERE t.track_id IN ({_placeholders(touched)}) "  # nosec B608
                        "AND m.track_id IS NULL", touched)})
        finally:
            conn.close()

    conn = _open(target_folder, store.CLASSIFICATION, modality)
    if conn is not None:
        try:
            with store.transaction(conn):
                conn.execute(
                    f"DELETE FROM embeddings WHERE detection_id IN ({marks})",  # nosec B608
                    ids)
                conn.execute(
                    f"DELETE FROM frame_predictions WHERE detection_id IN ({marks})",  # nosec B608
                    ids)
        finally:
            conn.close()

    conn = _open(target_folder, store.SEGMENTATION, modality)
    if conn is not None:
        try:
            with store.transaction(conn):
                conn.execute(
                    f"DELETE FROM segments WHERE detection_id IN ({marks})",  # nosec B608
                    ids)
        finally:
            conn.close()

    matches_path = store.matches_path(target_folder)
    if os.path.isfile(matches_path):
        conn = store.open_store(matches_path, store.MATCHES)
        try:
            column = "detection_id_t" if modality == "t" else "detection_id_w"
            with store.transaction(conn):
                conn.execute(
                    f"DELETE FROM detection_matches WHERE {column} IN ({marks})",  # nosec B608
                    ids)
        finally:
            conn.close()

    if emptied:
        _delete_track_rows(target_folder, modality, emptied)

    return {
        "detections": deleted,
        "tracks_removed": emptied,
        "tracks_kept": [t for t in touched if t not in emptied],
    }


def _delete_track_rows(target_folder: str, modality: str,
                       ids: List[int]) -> int:
    """The track's own rows: membership, the track, its votes, its match."""
    marks = _placeholders(ids)
    removed = 0

    conn = _open(target_folder, store.TRACKS, modality)
    if conn is not None:
        try:
            with store.transaction(conn):
                conn.execute(
                    f"DELETE FROM track_members WHERE track_id IN ({marks})",  # nosec B608
                    ids)
                removed = _rowcount(conn.execute(
                    f"DELETE FROM tracks WHERE track_id IN ({marks})",  # nosec B608
                    ids))
        finally:
            conn.close()

    conn = _open(target_folder, store.CLASSIFICATION, modality)
    if conn is not None:
        try:
            with store.transaction(conn):
                conn.execute(
                    f"DELETE FROM track_predictions WHERE track_id IN ({marks})",  # nosec B608
                    ids)
        finally:
            conn.close()

    matches_path = store.matches_path(target_folder)
    if os.path.isfile(matches_path):
        conn = store.open_store(matches_path, store.MATCHES)
        try:
            column = "track_id_t" if modality == "t" else "track_id_w"
            with store.transaction(conn):
                conn.execute(
                    "DELETE FROM detection_matches WHERE match_id IN "
                    f"(SELECT match_id FROM track_matches WHERE {column} IN ({marks}))",  # nosec B608
                    ids)
                conn.execute(
                    f"DELETE FROM track_matches WHERE {column} IN ({marks})",  # nosec B608
                    ids)
        finally:
            conn.close()

    # A label track materialised into one of these tracks now points at
    # nothing. ``delete_tracks`` has removed such annotations beforehand;
    # this is for a track emptied member by member, whose annotation
    # stays (the labelling tool would refuse it anyway, see the guard).
    conn = _open(target_folder, store.LABELS, modality)
    if conn is not None:
        try:
            with store.transaction(conn):
                conn.execute(
                    "UPDATE label_tracks SET track_id = NULL "
                    f"WHERE track_id IN ({marks})", ids)  # nosec B608
        finally:
            conn.close()
    return removed


def _delete_label_tracks(target_folder: str, modality: str,
                         label_track_ids: List[int]) -> int:
    """Remove label tracks and their key frames from the labels store."""
    if not label_track_ids:
        return 0
    conn = _open(target_folder, store.LABELS, modality)
    if conn is None:
        return 0
    marks = _placeholders(label_track_ids)
    try:
        with store.transaction(conn):
            conn.execute(
                f"DELETE FROM label_keyframes WHERE label_track_id IN ({marks})",  # nosec B608
                label_track_ids)
            conn.execute(
                "DELETE FROM label_track_origin_xmodal "
                f"WHERE label_track_id IN ({marks})", label_track_ids)  # nosec B608
            return _rowcount(conn.execute(
                f"DELETE FROM label_tracks WHERE label_track_id IN ({marks})",  # nosec B608
                label_track_ids))
    finally:
        conn.close()


def track_of_detection(target_folder: str, modality: str,
                       detection_id: int) -> Optional[int]:
    """The track (of the analysed runs) a detection belongs to, if any."""
    from . import track_store

    run_ids = track_store.analysis_runs(target_folder, modality)
    conn = _open(target_folder, store.TRACKS, modality)
    if conn is None or not run_ids:
        return None
    try:
        row = conn.execute(
            "SELECT m.track_id FROM track_members m "
            "JOIN tracks t ON t.track_id = m.track_id "
            "WHERE m.detection_id = ? "
            f"AND t.run_id IN ({_placeholders(run_ids)}) "  # nosec B608
            "ORDER BY t.run_id DESC LIMIT 1",
            [int(detection_id)] + list(run_ids)).fetchone()
        return None if row is None else int(row["track_id"])
    finally:
        conn.close()
