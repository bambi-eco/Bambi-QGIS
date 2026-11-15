# -*- coding: utf-8 -*-
"""A person's word on a track the pipeline made.

The labelling tool is where an animal is drawn by hand. A reviewer going
through the pipeline's tracks needs something lighter: this track is a red
deer, not a roe deer; this one is a rock. That is what is written here - the
species, sex or age of a tracker's track, set by a person.

**A person's word stands.** The classifiers write their calls onto the same
fields (:mod:`core.apply_results`), every time they run. A field a person set
is therefore marked - :data:`BY_HAND` in the track's attributes lists them -
and the projection of the classifier results leaves a marked field alone. A
label synced across from the other camera fills gaps only, so it never
reaches a field that has a value either.

**Species is written on the boxes too**, as the classifiers do: the exports
read a box's species, the survey tools the animal's.

Tracks of the labelling tool are not changed here. They are made from their
labels on every save, so a change has to be made where the labels are.

Free of QGIS and Qt.
"""

import json
import os
from typing import Iterable, List, Optional

from . import store

#: Track attribute listing the fields a person set: a JSON list.
BY_HAND = "by_hand"
#: The fields a person can set.
SPECIES, SEX, AGE = "species", "sex", "age"
FIELDS = (SPECIES, SEX, AGE)


class TrackReviewError(RuntimeError):
    """A change the store cannot take as asked."""


def by_hand(attributes) -> List[str]:
    """The fields of a track a person set, from its attributes (a dict, or
    the JSON it is stored as)."""
    if isinstance(attributes, str):
        try:
            attributes = json.loads(attributes)
        except ValueError:
            return []
    marked = (attributes or {}).get(BY_HAND) if isinstance(
        attributes, dict) else None
    if not isinstance(marked, (list, tuple)):
        return []
    return [field for field in FIELDS if field in marked]


def _marked(attributes: dict, field: str) -> dict:
    fields = set(by_hand(attributes)) | {field}
    attributes[BY_HAND] = [name for name in FIELDS if name in fields]
    return attributes


def _track(conn, target_folder: str, modality: str, track_id: int) -> dict:
    """The track's row; refuses what must not be changed here."""
    row = conn.execute(
        "SELECT t.track_id, t.species_id, t.attributes, r.kind "
        "FROM tracks t JOIN track_runs r ON r.run_id = t.run_id "
        "WHERE t.track_id = ?", (int(track_id),)).fetchone()
    if row is None:
        raise TrackReviewError(f"There is no track {track_id}.")
    if row["kind"] == "manual":
        raise TrackReviewError(
            f"Track {track_id} was labelled by hand: it is made from its "
            "labels on every save, so change it in the Labelling Tool.")
    return row


def _attributes(raw) -> dict:
    try:
        data = json.loads(raw) if raw else {}
    except ValueError:
        data = {}
    return data if isinstance(data, dict) else {}


def set_species(target_folder: str, modality: str, track_id: int,
                species_id: int) -> bool:
    """Say what species a tracked animal is - on the track and on its boxes
    - and mark it as a person's word.

    :return: whether anything changed
    :raises TrackReviewError: for a track that is not there, a hand-labelled
        one, or a species the project does not have
    """
    tracks_path = store.stage_path(target_folder, store.TRACKS, modality)
    det_path = store.stage_path(target_folder, store.DETECTIONS, modality)
    if not os.path.isfile(tracks_path):
        raise TrackReviewError("This camera has no tracks.")
    project = store.open_store(store.project_path(target_folder),
                               store.PROJECT)
    try:
        known = project.execute(
            "SELECT 1 FROM species WHERE species_id = ?",
            (int(species_id),)).fetchone() is not None
    finally:
        project.close()
    if not known:
        raise TrackReviewError(f"There is no species with id {species_id}.")

    conn = store.open_store(tracks_path, store.TRACKS, modality)
    try:
        with store.transaction(conn):
            row = _track(conn, target_folder, modality, track_id)
            attributes = _attributes(row["attributes"])
            changed = int(row["species_id"]) != int(species_id) or \
                SPECIES not in by_hand(attributes)
            conn.execute(
                "UPDATE tracks SET species_id = ?, attributes = ? "
                "WHERE track_id = ?",
                (int(species_id), json.dumps(_marked(attributes, SPECIES)),
                 int(track_id)))
            members = [int(member["detection_id"]) for member in conn.execute(
                "SELECT detection_id FROM track_members WHERE track_id = ?",
                (int(track_id),))]
    finally:
        conn.close()

    if members and os.path.isfile(det_path):
        conn = store.open_store(det_path, store.DETECTIONS, modality)
        try:
            with store.transaction(conn):
                marks = ", ".join("?" for _ in members)
                # a hand-drawn box belongs to its label, whatever track it
                # is in
                cursor = conn.execute(
                    "UPDATE detections SET species_id = ? "  # nosec B608
                    f"WHERE detection_id IN ({marks}) "
                    "AND label_track_id IS NULL AND species_id <> ?",
                    [int(species_id)] + members + [int(species_id)])
                changed = changed or (cursor.rowcount or 0) > 0
        finally:
            conn.close()
    return changed


def set_attribute(target_folder: str, modality: str, track_id: int,
                  field: str, value: Optional[int]) -> bool:
    """Say what sex or age a tracked animal is, and mark it as a person's
    word. *value* is the id of the enum value; ``None`` says "not known" -
    which is a person's word too, and stands.

    :return: whether anything changed
    """
    if field not in (SEX, AGE):
        raise TrackReviewError(
            f"'{field}' is not something a reviewer sets on a track.")
    path = store.stage_path(target_folder, store.TRACKS, modality)
    if not os.path.isfile(path):
        raise TrackReviewError("This camera has no tracks.")
    conn = store.open_store(path, store.TRACKS, modality)
    try:
        with store.transaction(conn):
            row = _track(conn, target_folder, modality, track_id)
            attributes = _attributes(row["attributes"])
            before = json.dumps(attributes, sort_keys=True)
            if value is None:
                attributes.pop(field, None)
            else:
                attributes[field] = int(value)
            _marked(attributes, field)
            after = json.dumps(attributes, sort_keys=True)
            if after == before:
                return False
            conn.execute(
                "UPDATE tracks SET attributes = ? WHERE track_id = ?",
                (json.dumps(attributes), int(track_id)))
            return True
    finally:
        conn.close()


def release(target_folder: str, modality: str, track_id: int,
            fields: Iterable[str] = FIELDS) -> bool:
    """Take back the mark on *fields*: the values stay as they are, and the
    next classifier run may write its own again.

    :return: whether a mark was removed
    """
    path = store.stage_path(target_folder, store.TRACKS, modality)
    if not os.path.isfile(path):
        return False
    conn = store.open_store(path, store.TRACKS, modality)
    try:
        with store.transaction(conn):
            row = conn.execute(
                "SELECT attributes FROM tracks WHERE track_id = ?",
                (int(track_id),)).fetchone()
            if row is None:
                return False
            attributes = _attributes(row["attributes"])
            kept = [name for name in by_hand(attributes)
                    if name not in set(fields)]
            if kept == by_hand(attributes):
                return False
            if kept:
                attributes[BY_HAND] = kept
            else:
                attributes.pop(BY_HAND, None)
            conn.execute(
                "UPDATE tracks SET attributes = ? WHERE track_id = ?",
                (json.dumps(attributes), int(track_id)))
            return True
    finally:
        conn.close()
