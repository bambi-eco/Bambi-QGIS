# -*- coding: utf-8 -*-
"""Write classification results where the rest of the plugin reads them.

The classifiers record their answers in ``classification.gpkg`` with full
provenance — the model, the per-frame probability, the vote margin. That file
is the authority, but nothing else in the plugin reads it. This module projects
those answers onto the fields the 6.0 store already has, and from there the
exporters, the QGIS layers, the survey analytics and the labelling tool all see
them with no changes to any of them:

==================  ===============================================
occlusion (frame)   ``detections.attributes["occlusion"]``
species (track)     ``tracks.species_id`` **and** its detections'
sex (track)         ``tracks.attributes["sex"]``
life stage (track)  ``tracks.attributes["age"]``
==================  ===============================================

**Species is written at both levels on purpose.** Exports and analytics filter
on the *detection's* species (``core/exporters/common.py`` selects
``d.species_id``), so writing only the track would leave every consumer blind
to the result. The track carries the individual-level answer; the detections
carry it too so the existing queries find it.

Three rules keep the write safe:

* **A detection that already has a species keeps it.** Only rows still on the
  fallback ``animal`` are filled in, so a detector class mapping someone
  configured is never silently overwritten — unless they ask for that.
* **Manual tracks are never touched.** A hand annotation outranks a model.
* **The projection is idempotent and re-runnable.** ``track_predictions``
  remains the source of truth, so if something else rewrites the detection
  species later — re-applying a detector class mapping does exactly that —
  the fix is to run this again, not to re-run the classifiers.
"""

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

from . import store

#: Task name → the field it lands in, and at which scope.
TRACK_FIELDS = {"sex": "sex", "life_stage": "age"}

#: The species id a detection has when nothing has identified it.
FALLBACK = store.FALLBACK_SPECIES_ID


def label_values(spec: dict, vocabulary: dict, task: str) -> Dict[str, int]:
    """``{model label: project value id}`` for one task.

    Prefers the mapping the user configured — keyed on class index, which is
    the only thing a model guarantees — and falls back to matching names, so a
    model that already speaks the project's language needs no configuration.
    """
    labels = list(spec.get("class_labels") or [])
    configured = spec.get("labels") or {}

    resolved: Dict[str, int] = {}
    for index, label in enumerate(labels):
        value = configured.get(str(index))
        if value is not None:
            resolved[label] = int(value)

    known = _vocabulary_values(vocabulary, task)
    for label in labels:
        if label not in resolved and label.lower() in known:
            resolved[label] = known[label.lower()]
    # A head whose classes were never listed can still be resolved by name.
    for label, value in known.items():
        resolved.setdefault(label, value)
    return resolved


def _vocabulary_values(vocabulary: dict, task: str) -> Dict[str, int]:
    """Project values a task can map onto, keyed by lower-case label."""
    if not vocabulary:
        return {}
    if task == "species":
        return {row["name"].lower(): row["species_id"]
                for row in vocabulary.get("species", [])}
    enum_name = {"life_stage": "age"}.get(task, task)
    return {row["label"].lower(): row["value_id"]
            for row in vocabulary.get("enums", {}).get(enum_name, [])}


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------

def _load_json(raw) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def apply_occlusion(target_folder: str, modality: str,
                    predictions: Iterable[dict], values: Dict[str, int],
                    log_fn=None) -> int:
    """Write per-frame occlusion onto the detections that carry it."""
    path = store.stage_path(target_folder, store.DETECTIONS, modality)
    if not os.path.isfile(path):
        return 0

    wanted = {}
    for row in predictions:
        value = values.get(row["label"])
        if value is not None:
            wanted[int(row["detection_id"])] = int(value)
    if not wanted:
        return 0

    written = 0
    conn = store.open_store(path, store.DETECTIONS, modality)
    try:
        with store.transaction(conn):
            existing = {
                int(r["detection_id"]): _load_json(r["attributes"])
                for r in conn.execute(
                    "SELECT detection_id, attributes FROM detections")}
            for detection_id, value in wanted.items():
                attributes = existing.get(detection_id)
                if attributes is None:
                    continue
                if attributes.get("occlusion") == value:
                    continue
                attributes["occlusion"] = value
                conn.execute(
                    "UPDATE detections SET attributes = ? "
                    "WHERE detection_id = ?",
                    (json.dumps(attributes), detection_id))
                written += 1
    finally:
        conn.close()

    if log_fn and written:
        log_fn(f"Applied occlusion to {written} detection(s)")
    return written


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------

def _manual_run_id(target_folder: str, modality: str) -> Optional[int]:
    from . import track_store

    manual = track_store.manual_run(target_folder, modality)
    return None if manual is None else int(manual["run_id"])


def apply_track_species(target_folder: str, modality: str,
                        predictions: Iterable[dict], values: Dict[str, int],
                        overwrite_detections: bool = False,
                        log_fn=None) -> Dict[str, int]:
    """Write the species vote onto tracks and their detections.

    Both levels, because the exporters and the analytics filter on the
    detection's species while the track is where the individual's answer
    belongs.
    """
    tracks_path = store.stage_path(target_folder, store.TRACKS, modality)
    det_path = store.stage_path(target_folder, store.DETECTIONS, modality)
    if not (os.path.isfile(tracks_path) and os.path.isfile(det_path)):
        return {"tracks": 0, "detections": 0}

    wanted = {}
    for row in predictions:
        value = values.get(row["label"])
        if value is not None:
            wanted[int(row["track_id"])] = int(value)
    if not wanted:
        return {"tracks": 0, "detections": 0}

    manual_run = _manual_run_id(target_folder, modality)
    tracks_written = 0
    members: Dict[int, List[int]] = {}

    conn = store.open_store(tracks_path, store.TRACKS, modality)
    try:
        with store.transaction(conn):
            for row in conn.execute(
                    "SELECT track_id, run_id, species_id FROM tracks"):
                track_id = int(row["track_id"])
                if track_id not in wanted:
                    continue
                if manual_run is not None and row["run_id"] == manual_run:
                    continue      # a hand annotation outranks a model
                if row["species_id"] != wanted[track_id]:
                    conn.execute(
                        "UPDATE tracks SET species_id = ? WHERE track_id = ?",
                        (wanted[track_id], track_id))
                    tracks_written += 1
                members[track_id] = []

            for row in conn.execute(
                    "SELECT track_id, detection_id FROM track_members"):
                track_id = int(row["track_id"])
                if track_id in members:
                    members[track_id].append(int(row["detection_id"]))
    finally:
        conn.close()

    species_of_detection = {
        detection_id: wanted[track_id]
        for track_id, detection_ids in members.items()
        for detection_id in detection_ids}

    detections_written = _write_detection_species(
        det_path, modality, species_of_detection, overwrite_detections)

    if log_fn:
        log_fn(f"Applied species to {tracks_written} track(s) and "
               f"{detections_written} detection(s)")
    return {"tracks": tracks_written, "detections": detections_written}


def _write_detection_species(path: str, modality: str,
                             species_of_detection: Dict[int, int],
                             overwrite: bool) -> int:
    """Fill in detection species, leaving identified ones alone by default."""
    if not species_of_detection:
        return 0

    written = 0
    conn = store.open_store(path, store.DETECTIONS, modality)
    try:
        with store.transaction(conn):
            current = {int(r["detection_id"]): int(r["species_id"])
                       for r in conn.execute(
                           "SELECT detection_id, species_id FROM detections")}
            for detection_id, species_id in species_of_detection.items():
                have = current.get(detection_id)
                if have is None or have == species_id:
                    continue
                # A detector that identified the animal itself keeps its
                # answer, so a configured class mapping is not undone.
                if have != FALLBACK and not overwrite:
                    continue
                conn.execute(
                    "UPDATE detections SET species_id = ? "
                    "WHERE detection_id = ?", (species_id, detection_id))
                written += 1
    finally:
        conn.close()
    return written


def apply_track_attribute(target_folder: str, modality: str, field: str,
                          predictions: Iterable[dict],
                          values: Dict[str, int], log_fn=None) -> int:
    """Write a per-track enum attribute (``sex``, ``age``)."""
    path = store.stage_path(target_folder, store.TRACKS, modality)
    if not os.path.isfile(path):
        return 0

    wanted = {}
    for row in predictions:
        value = values.get(row["label"])
        if value is not None:
            wanted[int(row["track_id"])] = int(value)
    if not wanted:
        return 0

    manual_run = _manual_run_id(target_folder, modality)
    written = 0
    conn = store.open_store(path, store.TRACKS, modality)
    try:
        with store.transaction(conn):
            for row in conn.execute(
                    "SELECT track_id, run_id, attributes FROM tracks"):
                track_id = int(row["track_id"])
                if track_id not in wanted:
                    continue
                if manual_run is not None and row["run_id"] == manual_run:
                    continue
                attributes = _load_json(row["attributes"])
                if attributes.get(field) == wanted[track_id]:
                    continue
                attributes[field] = wanted[track_id]
                conn.execute(
                    "UPDATE tracks SET attributes = ? WHERE track_id = ?",
                    (json.dumps(attributes), track_id))
                written += 1
    finally:
        conn.close()

    if log_fn and written:
        log_fn(f"Applied {field} to {written} track(s)")
    return written


# ---------------------------------------------------------------------------
# Everything at once
# ---------------------------------------------------------------------------

def apply_all(target_folder: str, modality: str, models: dict,
              tasks: Sequence[str] = (), overwrite_detections: bool = False,
              log_fn=None) -> Dict[str, int]:
    """Project every stored classification result onto the canonical fields.

    Idempotent: running it twice changes nothing the second time, and running
    it after something else rewrote the detection species puts the answer back.
    """
    from . import classification_store, label_store

    vocabulary = label_store.vocabulary(target_folder)
    if not vocabulary:
        if log_fn:
            log_fn("This project has no vocabulary to write results into.")
        return {}

    wanted = set(tasks) if tasks else {"occlusion", "species", "sex",
                                       classification_store.LIFE_STAGE}
    counts: Dict[str, int] = {}

    if "occlusion" in wanted:
        values = label_values(models.get("occlusion") or {}, vocabulary,
                              "occlusion")
        counts["occlusion"] = apply_occlusion(
            target_folder, modality,
            classification_store.frame_predictions(
                target_folder, modality, "occlusion"),
            values, log_fn=log_fn)

    if "species" in wanted:
        values = label_values(models.get("species") or {}, vocabulary,
                              "species")
        applied = apply_track_species(
            target_folder, modality,
            classification_store.track_predictions(
                target_folder, modality, "species"),
            values, overwrite_detections=overwrite_detections, log_fn=log_fn)
        counts["species_tracks"] = applied["tracks"]
        counts["species_detections"] = applied["detections"]

    for task, field in TRACK_FIELDS.items():
        if task not in wanted:
            continue
        values = label_values(models.get(task) or {}, vocabulary, task)
        counts[field] = apply_track_attribute(
            target_folder, modality, field,
            classification_store.track_predictions(
                target_folder, modality, task),
            values, log_fn=log_fn)

    return counts
