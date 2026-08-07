# -*- coding: utf-8 -*-
"""Carry labels from one camera's animals onto their matched partners.

A stage classifies the camera it reads, so a flight can easily end up with the
species decided on RGB and nothing on the thermal side — and then the sex head,
which picks its model per species, has nothing to go on there. This is the way
across: for every pair A3 confirmed, copy what one side knows onto the other.

Three rules, and they are the whole of it:

* **Canonical fields only.** ``classification.gpkg`` records what a model saw
  and how sure it was, per camera. A label that arrived over the match was not
  observed on this camera at all, so writing it there would invent a run that
  never happened. The synced answer belongs on the animal, which is exactly
  where the exporters, the layers, the analytics and the labelling tool read.
* **Fill, never replace.** A track that already carries an answer keeps it,
  whoever put it there. Sync is for the gaps.
* **Hand annotations are untouchable**, on both sides: a manual run is never
  written to, and never read past.

Only pairs of the active matching run are considered; an animal one camera
never saw has no partner and is left alone.
"""

import json
import os
from typing import Dict, Optional

from . import store

#: Per-track attributes that travel, alongside the species column.
TRACK_ATTRIBUTES = ("sex", "age")


def _load_json(raw) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _manual_run_id(target_folder: str, modality: str) -> Optional[int]:
    from . import track_store

    manual = track_store.manual_run(target_folder, modality)
    return None if manual is None else int(manual["run_id"])


def _identified(species_id) -> bool:
    """True when a species id is an answer rather than the fallback."""
    if species_id is None:
        return False
    return int(species_id) != store.FALLBACK_SPECIES_ID


def _read_tracks(target_folder: str, modality: str) -> Dict[int, dict]:
    """``{track_id: {"species_id", "attributes", "manual"}}``."""
    path = store.stage_path(target_folder, store.TRACKS, modality)
    if not os.path.isfile(path):
        return {}
    manual_run = _manual_run_id(target_folder, modality)
    conn = store.open_store(path, store.TRACKS, modality)
    try:
        return {
            int(row["track_id"]): {
                "species_id": row["species_id"],
                "attributes": _load_json(row["attributes"]),
                "manual": manual_run is not None and row["run_id"] == manual_run,
            }
            for row in conn.execute(
                "SELECT track_id, run_id, species_id, attributes FROM tracks")}
    finally:
        conn.close()


def sync_tracks(target_folder: str, source: str, destination: str,
                log_fn=None) -> Dict[str, int]:
    """Copy species, sex and age onto the destination camera's animals."""
    from . import match_store

    partners = match_store.partner_tracks(target_folder, source)
    if not partners:
        return {"species": 0, "sex": 0, "age": 0}

    from_tracks = _read_tracks(target_folder, source)
    to_tracks = _read_tracks(target_folder, destination)
    if not (from_tracks and to_tracks):
        return {"species": 0, "sex": 0, "age": 0}

    counts = {"species": 0, "sex": 0, "age": 0}
    path = store.stage_path(target_folder, store.TRACKS, destination)
    conn = store.open_store(path, store.TRACKS, destination)
    try:
        with store.transaction(conn):
            for source_id, destination_id in partners.items():
                have = from_tracks.get(source_id)
                want = to_tracks.get(destination_id)
                if have is None or want is None or want["manual"]:
                    continue

                species = have["species_id"]
                if _identified(species) and not _identified(want["species_id"]):
                    conn.execute(
                        "UPDATE tracks SET species_id = ? WHERE track_id = ?",
                        (int(species), destination_id))
                    counts["species"] += 1

                attributes = dict(want["attributes"])
                changed = False
                for field in TRACK_ATTRIBUTES:
                    value = have["attributes"].get(field)
                    if value is None or attributes.get(field) is not None:
                        continue
                    attributes[field] = value
                    counts[field] += 1
                    changed = True
                if changed:
                    conn.execute(
                        "UPDATE tracks SET attributes = ? WHERE track_id = ?",
                        (json.dumps(attributes), destination_id))
    finally:
        conn.close()

    if log_fn:
        log_fn(f"Synced species onto {counts['species']} track(s), sex onto "
               f"{counts['sex']}, age onto {counts['age']}")
    return counts


def sync_detections(target_folder: str, source: str, destination: str,
                    log_fn=None) -> Dict[str, int]:
    """Copy per-frame occlusion and species onto the paired detections.

    Species rides along at detection level for the same reason the classifiers
    write it there: the exporters and the analytics filter on the detection's
    species, so a track-only answer would be invisible to them.
    """
    from . import match_store

    partners = match_store.partner_detections(target_folder, source)
    if not partners:
        return {"occlusion": 0, "species": 0}

    source_path = store.stage_path(target_folder, store.DETECTIONS, source)
    dest_path = store.stage_path(target_folder, store.DETECTIONS, destination)
    if not (os.path.isfile(source_path) and os.path.isfile(dest_path)):
        return {"occlusion": 0, "species": 0}

    conn = store.open_store(source_path, store.DETECTIONS, source)
    try:
        have = {int(row["detection_id"]): (row["species_id"],
                                           _load_json(row["attributes"]))
                for row in conn.execute(
                    "SELECT detection_id, species_id, attributes "
                    "FROM detections")}
    finally:
        conn.close()

    counts = {"occlusion": 0, "species": 0}
    conn = store.open_store(dest_path, store.DETECTIONS, destination)
    try:
        with store.transaction(conn):
            for row in conn.execute(
                    "SELECT detection_id, species_id, attributes "
                    "FROM detections"):
                detection_id = int(row["detection_id"])
                partner = partners.get(detection_id)
                source_row = have.get(partner) if partner else None
                if source_row is None:
                    continue
                species, attributes_in = source_row

                if _identified(species) and not _identified(row["species_id"]):
                    conn.execute(
                        "UPDATE detections SET species_id = ? "
                        "WHERE detection_id = ?", (int(species), detection_id))
                    counts["species"] += 1

                occlusion = attributes_in.get("occlusion")
                attributes = _load_json(row["attributes"])
                if occlusion is not None and attributes.get(
                        "occlusion") is None:
                    attributes["occlusion"] = occlusion
                    conn.execute(
                        "UPDATE detections SET attributes = ? "
                        "WHERE detection_id = ?",
                        (json.dumps(attributes), detection_id))
                    counts["occlusion"] += 1
    finally:
        conn.close()

    if log_fn:
        log_fn(f"Synced occlusion onto {counts['occlusion']} detection(s) and "
               f"species onto {counts['species']}")
    return counts


def sync_all(target_folder: str, source: str, destination: str,
             log_fn=None) -> Dict[str, int]:
    """Everything that travels, in one call."""
    if source not in store.MODALITIES or destination not in store.MODALITIES:
        raise ValueError(f"Unknown modality: {source!r} -> {destination!r}")
    if source == destination:
        raise ValueError("Labels are synced between the two cameras, not onto "
                         "the camera they came from.")

    counts = dict(sync_tracks(target_folder, source, destination,
                              log_fn=log_fn))
    detections = sync_detections(target_folder, source, destination,
                                 log_fn=log_fn)
    counts["occlusion"] = detections["occlusion"]
    counts["species_detections"] = detections["species"]
    return counts
