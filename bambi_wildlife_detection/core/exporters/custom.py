# -*- coding: utf-8 -*-
"""An export format of the user's own (core.custom_models, kind
"exporter"): the plugin reads the store, the user's class writes the
files.

The class is handed everything the built-in formats are made from, as
plain Python - no id that only the store understands, species and enum
values as their names::

    data = {
      "flight":   {"name", "folder", "camera", "crs"},
      "frames":   [{"index", "file", "path", "timestamp", "duplicated"}],
      "frame_size": (width, height) or None,
      "species":  ["roe deer", ...]            the names that occur
      "detections": [{"detection_id", "frame", "box": (x1, y1, x2, y2),
                      "confidence", "species", "track_id", "attributes",
                      "world": (x1, y1, z1, x2, y2, z2) or None,
                      "lonlat": (lon, lat) or None}],
      "tracks":   [{"track_id", "species", "attributes", "detection_ids",
                    "first_frame", "last_frame"}],
      "segmentations": [{"segment_id", "detection_id", "frame",
                         "polygon": [ring, ...] or None      pixels
                         "world":   [ring, ...] or None      project CRS
                         "lonlat":  [ring, ...] or None      WGS84
                         "attributes"}],
      "options":  {"include_images", "include_not_an_animal"},
    }

What is in it follows the export it is part of, as for every format:
``not-an-animal`` is left out unless asked for, and inside
``common.only_confirmed`` only what both cameras saw is there.
"""

import json
import os
from typing import List, Optional

from . import common

CAMERAS = {"t": "thermal", "w": "rgb"}


def _rings(raw) -> Optional[List[List[List[float]]]]:
    """A stored polygon as a list of rings of ``[x, y]``, or ``None``."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    first = parsed[0]
    nested = isinstance(first, list) and first and isinstance(first[0], list)
    rings = []
    for ring in (parsed if nested else [parsed]):
        points = [[float(point[0]), float(point[1])] for point in ring
                  if isinstance(point, (list, tuple)) and len(point) >= 2]
        if len(points) >= 3:
            rings.append(points)
    return rings or None


def _attributes(raw) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_segmentations(target_folder: str, modality: str,
                       epsg: Optional[int] = None,
                       detection_ids=None) -> List[dict]:
    """The stored segmentation masks, as an exporter is handed them.

    :param detection_ids: the detections that are exported; a mask of
        another detection is left out with it (``None``: every mask). A
        mask of no detection is one of the flight and stays.
    """
    from .. import store

    path = store.stage_path(target_folder, store.SEGMENTATION, modality)
    if not os.path.isfile(path):
        return []
    conn = store.open_store(path, store.SEGMENTATION, modality)
    try:
        rows = [dict(row) for row in conn.execute(
            "SELECT segment_id, detection_id, frame, polygon_px, "
            "polygon_geo, attributes FROM segments "
            "ORDER BY frame, segment_id")]
    finally:
        conn.close()

    masks = []
    for row in rows:
        detection = row.get("detection_id")
        if detection_ids is not None and detection is not None \
                and int(detection) not in detection_ids:
            continue
        world = _rings(row.get("polygon_geo"))
        lonlat = None
        if world and epsg:
            lonlat = [[list(pair) for pair in common.to_wgs84(ring, epsg)]
                      for ring in world]
        masks.append({
            "segment_id": int(row["segment_id"]),
            "detection_id": None if detection is None else int(detection),
            "frame": int(row["frame"]),
            "polygon": _rings(row.get("polygon_px")),
            "world": world, "lonlat": lonlat,
            "attributes": _attributes(row.get("attributes")),
        })
    return masks


def build_data(target_folder: str, modality: str,
               epsg: Optional[int] = None,
               include_not_an_animal: bool = True,
               include_images: bool = True, name: str = "") -> dict:
    """Everything a custom exporter is handed for one flight and camera
    (see the module's description)."""
    vocabulary = common.load_vocabulary(target_folder)
    rows, animals = common.load_export_rows(
        target_folder, modality, vocabulary, include_not_an_animal)
    species = vocabulary["species"]

    placed = [row for row in rows if row.get("gx1") is not None]
    lonlat = {}
    if epsg and placed:
        centres = common.to_wgs84(
            [((row["gx1"] + row["gx2"]) / 2.0,
              (row["gy1"] + row["gy2"]) / 2.0) for row in placed], epsg)
        lonlat = {row["detection_id"]: (float(lon), float(lat))
                  for row, (lon, lat) in zip(placed, centres)}

    detections = []
    for row in rows:
        world = None
        if row.get("gx1") is not None:
            world = tuple(row[key] for key in
                          ("gx1", "gy1", "gz1", "gx2", "gy2", "gz2"))
        detections.append({
            "detection_id": int(row["detection_id"]),
            "frame": int(row["frame"]),
            "box": (row["x1"], row["y1"], row["x2"], row["y2"]),
            "confidence": row["confidence"],
            "species": species.get(row["species_id"], ""),
            "track_id": row.get("track_id"),
            "attributes": dict(row["properties"]),
            "world": world,
            "lonlat": lonlat.get(row["detection_id"]),
        })

    tracks = []
    for track_id, members in sorted(common.tracks_of(rows).items()):
        animal = animals.get(track_id) or {}
        frames_of = [member["frame"] for member in members]
        tracks.append({
            "track_id": track_id,
            "species": animal.get("species", ""),
            "attributes": dict(animal.get("attributes") or {}),
            "detection_ids": [int(member["detection_id"]) for member in
                              sorted(members, key=lambda r: r["frame"])],
            "first_frame": min(frames_of), "last_frame": max(frames_of),
        })

    frames_folder = os.path.join(target_folder, f"frames_{modality}")
    frames = [{
        "index": frame["frame"], "file": frame["imagefile"],
        "path": os.path.join(frames_folder, frame["imagefile"]),
        "timestamp": frame.get("epoch") or "",
        "duplicated": frame["duplicated"],
    } for frame in common.load_frames(target_folder, modality)]

    exported = {detection["detection_id"] for detection in detections}
    name = name or os.path.basename(os.path.normpath(target_folder))
    return {
        "flight": {"name": name,
                   "folder": target_folder,
                   "camera": CAMERAS.get(modality, modality),
                   "crs": f"EPSG:{int(epsg)}" if epsg else None},
        "frames": frames,
        "frame_size": common.frame_size(target_folder, modality),
        "species": sorted({detection["species"] for detection in detections
                           if detection["species"]}),
        "detections": detections,
        "tracks": tracks,
        "segmentations": load_segmentations(
            target_folder, modality, epsg, exported),
        "options": {"include_images": bool(include_images),
                    "include_not_an_animal": bool(include_not_an_animal)},
    }


def export_custom(target_folder: str, modality: str, output_folder: str,
                  custom: Optional[dict] = None, epsg: Optional[int] = None,
                  include_not_an_animal: bool = True,
                  include_images: bool = True, model_dir: str = "",
                  log_fn=None) -> str:
    """Hand the flight to the user's exporter and let it write into
    *output_folder*.

    :param custom: the implementation's spec (``path``, ``class``,
        ``options``), as the Custom exporter box stores it
    :raises ExportError: without an exporter, or when the user's code
        fails - with what it said
    """
    from .. import custom_models

    if not custom or not (custom.get("path") or "").strip():
        raise common.ExportError(
            "No custom exporter is chosen. Select the .py file with your "
            "exporter under Processing - Export - Custom exporter.")
    log = log_fn or (lambda message: None)
    data = build_data(target_folder, modality, epsg, include_not_an_animal,
                      include_images)
    common.ensure_folder(output_folder)
    context = custom_models.Context(
        kind=custom_models.EXPORTER, camera=CAMERAS.get(modality, modality),
        model_dir=model_dir, target_folder=target_folder,
        log=lambda message: log(f"  {message}"))
    try:
        exporter = custom_models.open_implementation(
            dict(custom, source=custom_models.SOURCE),
            custom_models.EXPORTER, context)
        try:
            exporter.export(data, output_folder)
        finally:
            exporter.close()
    except custom_models.CustomModelError as exc:
        raise common.ExportError(str(exc)) from exc
    log(f"{exporter.label}: {len(data['detections'])} detection(s), "
        f"{len(data['tracks'])} track(s), {len(data['segmentations'])} "
        f"segmentation(s) → {output_folder}")
    return output_folder
