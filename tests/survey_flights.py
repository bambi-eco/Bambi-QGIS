# -*- coding: utf-8 -*-
"""Flights for the tests of the survey analytics: animals where the test
says they are, in the stores the plugin reads them from.

A flight is described by its tracked animals::

    make_flight(folder, [
        {"id": 1, "species": "red deer", "at": (10, 20), "frames": (30, 45)},
        {"id": 2, "species": "red deer", "at": (40, 20), "frames": (32, 50)},
    ])

``at`` is the animal's place in metres east and north of :data:`ORIGIN` (its
boxes are spread a little around it, so that the middle of them is exactly
there), ``frames`` the first and the last frame it is in. Frame *n* was taken
*n* seconds after :data:`START` unless ``seconds_per_frame`` says otherwise.
"""
import json
import os
from datetime import datetime, timedelta, timezone

from bambi_wildlife_detection.core import (
    detection_store, label_store, store, track_store)

#: South-west corner the animals' places are measured from (EPSG:32633).
ORIGIN = (500000.0, 5300000.0)
START = datetime(2024, 5, 1, 5, 0, 0, tzinfo=timezone.utc)
EPSG = 32633


def species_id(folder: str, name: str) -> int:
    """The id of species *name* in a flight, added to its schema if new."""
    names = label_store.vocabulary(folder, create=True)["species_by_name"]
    if name in names:
        return names[name]
    conn = store.open_store(store.project_path(folder), store.PROJECT)
    try:
        with store.transaction(conn):
            conn.execute(
                "INSERT INTO species (species_id, name, protected) VALUES "
                "((SELECT MAX(species_id) + 1 FROM species), ?, 0)", (name,))
    finally:
        conn.close()
    return label_store.vocabulary(folder)["species_by_name"][name]


def write_poses(folder: str, n_frames: int, modality: str = "t",
                seconds_per_frame: float = 1.0, timed: bool = True,
                locations=None, start=None) -> str:
    """``poses_{m}.json`` for *n_frames* frames. *locations* gives the
    drone's place per frame (mesh-local metres); a line due east at 5 m per
    frame otherwise. *start* is when the first frame was taken
    (:data:`START` unless given)."""
    start = start or START
    images = []
    for frame in range(n_frames):
        if locations is not None:
            x, y = locations[frame]
        else:
            x, y = 5.0 * frame, 0.0
        image = {"imagefile": f"frame_{frame:06d}.jpg",
                 "location": [x, y, 80.0]}
        if timed:
            image["epoch"] = (start + timedelta(
                seconds=frame * seconds_per_frame)).isoformat()
        images.append(image)
    path = os.path.join(folder, f"poses_{modality}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"images": images}, handle)
    return path


def write_footprints(folder: str, boxes, modality: str = "t") -> None:
    """The ground the camera saw: one footprint per *box* ``(min x, min y,
    max x, max y)`` in metres from :data:`ORIGIN`, as the frames 0, 1, ..."""
    from bambi_wildlife_detection.core import fov_store

    polygons = {}
    for frame, (x0, y0, x1, y1) in enumerate(boxes):
        polygons[frame] = [
            (ORIGIN[0] + x, ORIGIN[1] + y, 0.0)
            for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    fov_store.record_fov(folder, modality, polygons)


def make_flight(folder: str, tracks, modality: str = "t",
                seconds_per_frame: float = 1.0, timed: bool = True,
                loose_boxes=(), n_frames=None, locations=None,
                seen=None, start=None) -> str:
    """A processed flight: detections, their places, the tracks.

    :param tracks: ``{"id", "species", "at": (x, y) | None, "frames":
        (first, last)}``; ``at`` ``None`` = never geo-referenced. Optional
        ``"sex"`` / ``"age"`` labels and ``"approved"``.
    :param loose_boxes: ``{"species", "at", "frame"}`` - boxes of no track
    :param seen: the ground the camera saw (:func:`write_footprints`);
        ``None`` = no field-of-view footprints at all
    :param start: when the flight was flown (:data:`START` unless given)
    :return: *folder*
    """
    os.makedirs(folder, exist_ok=True)
    label_store.vocabulary(folder, create=True)
    ids = {track["species"]: species_id(folder, track["species"])
           for track in list(tracks) + list(loose_boxes)}

    boxes = []          # (frame, track id or None, place or None, species)
    for track in tracks:
        first, last = track["frames"]
        at = track.get("at")
        # two boxes, either side of the place: their middle is the place
        for frame, shift in ((first, -1.0), (last, 1.0)):
            place = None if at is None else (at[0] + shift, at[1])
            boxes.append((frame, track["id"], place, track["species"]))
    for box in loose_boxes:
        boxes.append((box["frame"], None, box["at"], box["species"]))

    detection_store.record_detections(folder, modality, [
        {"frame": frame, "x1": 10.0 + 7 * n, "y1": 5.0,
         "x2": 14.0 + 7 * n, "y2": 9.0, "confidence": 0.9,
         "source_class": species}
        for n, (frame, _track, _place, species) in enumerate(boxes)])
    stored = sorted(track_store.load_detections(folder, modality),
                    key=lambda row: row["x1"])
    detection_ids = [row["detection_id"] for row in stored]

    # what the detector called a box is its species, by the class mapping
    project = store.open_store(store.project_path(folder), store.PROJECT)
    try:
        source = detection_store.ensure_source(
            project, detection_store.DETECTOR)
        with store.transaction(project):
            for name, ident in ids.items():
                project.execute(
                    "INSERT OR REPLACE INTO class_mapping "
                    "(source_id, source_class, species_id) VALUES (?, ?, ?)",
                    (source, name, ident))
    finally:
        project.close()
    conn = store.open_store(
        store.stage_path(folder, store.DETECTIONS, modality),
        store.DETECTIONS, modality)
    try:
        with store.transaction(conn):
            for detection_id, (_f, _t, _p, species) in zip(detection_ids,
                                                           boxes):
                conn.execute(
                    "UPDATE detections SET species_id = ? "
                    "WHERE detection_id = ?", (ids[species], detection_id))
    finally:
        conn.close()

    track_store.record_georeference(folder, modality, [
        {"detection_id": detection_id,
         "gx1": ORIGIN[0] + place[0] - 0.5, "gy1": ORIGIN[1] + place[1] - 0.5,
         "gz1": 0.0,
         "gx2": ORIGIN[0] + place[0] + 0.5, "gy2": ORIGIN[1] + place[1] + 0.5,
         "gz2": 0.0}
        for detection_id, (_f, _t, place, _s) in zip(detection_ids, boxes)
        if place is not None])
    species_of_track = {track["id"]: ids[track["species"]]
                        for track in tracks}
    if tracks:
        track_store.record_tracks(
            folder, modality,
            [{"track_id": track, "detection_id": detection_id}
             for detection_id, (_f, track, _p, _s) in zip(detection_ids, boxes)
             if track is not None],
            species_of=lambda track: species_of_track[track])
    for track in tracks:
        for key in ("sex", "age", "approved"):
            if track.get(key) is not None:
                track_store.set_track_attribute(
                    folder, modality, track["id"], key, track[key])

    last = max([frame for frame, _t, _p, _s in boxes] or [0])
    write_poses(folder, n_frames or last + 1, modality, seconds_per_frame,
                timed, locations, start)
    if seen is not None:
        write_footprints(folder, seen, modality)
    return folder


def make_labelled_flight(folder, materialise=True):
    """Two animals and a rock.

    A roe deer the tracker lost half-way (two tracks), a red deer the
    detector calls roe deer and loses after four frames, and a warm rock it
    takes for an animal on two frames. All three were labelled: the red deer
    by importing its track and correcting it, the rock as not-an-animal.
    """
    from bambi_wildlife_detection.core import classification_store

    os.makedirs(folder, exist_ok=True)
    roe, red = species_id(folder, "roe deer"), species_id(folder, "red deer")
    project = store.open_store(store.project_path(folder), store.PROJECT)
    source = detection_store.ensure_source(
        project, detection_store.DETECTOR, "yolo_wildlife.pt")
    with store.transaction(project):
        project.execute(
            "INSERT OR REPLACE INTO class_mapping (source_id, source_class, "
            "species_id) VALUES (?, 'roe', ?)", (source, roe))
    project.close()

    def box(frame, at, confidence):
        return {"frame": frame, "x1": at[0], "y1": at[1], "x2": at[0] + 10.0,
                "y2": at[1] + 10.0, "confidence": confidence,
                "source_class": "roe"}

    rows = [box(f, (100.0, 100.0), 0.9) for f in range(6)]          # 1-6
    rows += [box(f, (300.0, 300.0), 0.5) for f in range(4)]         # 7-10
    rows += [box(f, (500.0, 500.0), 0.2) for f in (2, 3)]           # 11-12
    detection_store.record_detections(folder, "t", rows,
                                      model="yolo_wildlife.pt")
    ids = [row["detection_id"] for row in sorted(
        track_store.load_detections(folder, "t"),
        key=lambda row: row["detection_id"])]
    membership = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4]
    track_store.record_tracks(folder, "t", [
        {"track_id": track, "detection_id": detection}
        for track, detection in zip(membership, ids)], tracker="builtin")
    track_store.set_track_attribute(folder, "t", 1, "sex", "female")
    classification_store.record_track_predictions(folder, "t", "species", [
        {"track_id": 1, "label": "roe deer", "votes": 3, "n": 3,
         "fraction": 1.0, "model": "/models/species_v3.pt"},
        {"track_id": 3, "label": "roe deer", "votes": 3, "n": 4,
         "fraction": 0.75, "model": "/models/species_v3.pt"}])

    vocabulary = label_store.vocabulary(folder)
    female = next(value["value_id"] for value in vocabulary["enums"]["sex"]
                  if value["label"] == "female") \
        if "enums" in vocabulary else None

    def keyframes(at, frames):
        return [{"frame": frame, "x1": at[0], "y1": at[1],
                 "x2": at[0] + 10.0, "y2": at[1] + 10.0}
                for frame in frames]

    label_store.save_tracks(folder, "t", [
        {"label_track_id": 1, "species_id": roe,
         "attributes": {} if female is None else {"sex": female},
         "keyframes": keyframes((100.0, 100.0), (0, 5))},
        {"label_track_id": 2, "species_id": red, "origin_track_id": 3,
         "keyframes": keyframes((300.0, 300.0), (0, 5))},
        {"label_track_id": 3,
         "species_id": store.NOT_AN_ANIMAL_SPECIES_ID,
         "keyframes": keyframes((500.0, 500.0), (2, 3))},
    ])
    if materialise:
        label_store.materialise(folder, "t")
    return folder
