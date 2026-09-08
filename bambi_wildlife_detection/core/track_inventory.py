# -*- coding: utf-8 -*-
"""One row per tracked individual, with everything the project knows about it.

The pipeline's answers about an animal are spread over five stores - its
boxes and their confidences, the geo-referencing, the tracking run, the
classification votes and the cross-modal match - plus the labelling tool's
annotations and the flight-route distances. A census needs them side by side:
which animals were seen, from when to when, where they entered and left the
frame, what they were called and how sure the caller was. This module joins
them into one table and writes it as CSV (for a spreadsheet) and JSON (for
anything else).

Pure Python on top of the store readers, no QGIS, so it is unit-tested.
"""
import csv
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

from . import store

#: Column order of the CSV. Everything the project knows, most-asked first.
COLUMNS = [
    # The user's own verdict on the row, set in the report; first, because
    # it is the one column the user writes.
    "approved",
    "track_id", "camera",
    "species", "species_vote_fraction", "species_votes", "species_model",
    "sex", "sex_vote_fraction", "sex_model",
    "age", "age_source",
    "n_boxes", "n_occluded", "n_clear", "n_occlusion_unknown",
    "n_interpolated", "n_georeferenced",
    "mean_confidence", "min_confidence", "max_confidence",
    "start_frame", "end_frame", "n_frames", "frame_span",
    "start_time", "end_time", "duration_s",
    "start_x", "start_y", "start_z", "end_x", "end_y", "end_z", "crs",
    "start_lat", "start_lon", "end_lat", "end_lon",
    "path_length_m", "displacement_m",
    "mean_box_width_px", "mean_box_height_px",
    "matched_track", "matched_camera",
    "label_track_ids", "label_species", "label_sex", "label_age",
    "perpendicular_distance_m",
    "attributes",
]

_OTHER_CAMERA = {"t": "w", "w": "t"}
_CAMERA_NAMES = {"t": "thermal", "w": "rgb"}


def _mean(values: List[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _centre(row: dict) -> Optional[Tuple[float, float, float]]:
    """Ground centre of a geo-referenced box, or ``None``."""
    if row.get("gx1") is None or row.get("gx2") is None:
        return None
    return ((row["gx1"] + row["gx2"]) / 2.0,
            (row["gy1"] + row["gy2"]) / 2.0,
            ((row.get("gz1") or 0.0) + (row.get("gz2") or 0.0)) / 2.0)


def _to_wgs84(points: Iterable[Optional[Tuple[float, float]]],
              epsg: Optional[int]) -> List[Optional[Tuple[float, float]]]:
    """``(lat, lon)`` per point, ``None`` where absent or unconvertible."""
    points = list(points)
    if not epsg or not any(points):
        return [None for _ in points]
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs(
            f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    except Exception:  # nosec B110 - no pyproj, or an unknown CRS
        return [None for _ in points]
    result = []
    for point in points:
        if point is None:
            result.append(None)
            continue
        try:
            lon, lat = transformer.transform(point[0], point[1])
            result.append((lat, lon) if math.isfinite(lat) else None)
        except Exception:  # nosec B110
            result.append(None)
    return result


def _frame_times(target_folder: str, modality: str) -> Dict[int, str]:
    """``frame -> capture time`` as stored in the poses (ISO or epoch)."""
    from .exporters import common

    times = {}
    for frame in common.load_frames(target_folder, modality):
        if frame.get("epoch") not in (None, ""):
            times[frame["frame"]] = frame["epoch"]
    return times


def _seconds(value) -> Optional[float]:
    """Epoch seconds of a stored capture time, or ``None``."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(value)).timestamp()
    except Exception:  # nosec B110
        return None


def _perpendicular_distances(target_folder: str,
                             modality: str) -> Dict[int, float]:
    """``track_id -> metres`` from the flight-route step, whichever camera's
    route it was computed against."""
    result: Dict[int, float] = {}
    for route in ("t", "w"):
        path = os.path.join(target_folder, f"flight_route_{route}",
                            f"perpendicular_tracks_{modality}.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                entries = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(entries, dict):
            entries = list(entries.values())
        for entry in entries or []:
            if not isinstance(entry, dict) or entry.get("track_id") is None:
                continue
            for key in ("distance_m", "perpendicular_distance", "distance"):
                if entry.get(key) is not None:
                    result.setdefault(int(entry["track_id"]),
                                      float(entry[key]))
                    break
        if result:
            break
    return result


def build_inventory(target_folder: str, modality: str,
                    epsg: Optional[int] = None,
                    species_ids: Optional[Iterable[int]] = None) -> List[dict]:
    """One dict per track of the active tracking run(s), keys as ``COLUMNS``.

    :param target_folder: the pipeline output root folder
    :param modality: ``"t"`` or ``"w"``
    :param epsg: the project CRS the geo-referenced boxes are in, for the
        WGS84 columns; ``None`` leaves them empty
    :param species_ids: the Survey Analytics species filter - only tracks
        whose species is one of these are listed; ``None`` lists every track
    """
    wanted_species = None if species_ids is None else {int(s) for s in species_ids}
    from . import (classification_store, label_store, match_store,
                   track_store)
    from .exporters import common

    if modality not in store.MODALITIES:
        raise ValueError(f"Unknown modality: {modality!r}")

    vocabulary = common.load_vocabulary(target_folder)
    species_names = vocabulary["species"]
    enum_labels = vocabulary["enum_labels"]

    rows = common.load_detections(target_folder, modality,
                                  include_not_an_animal=True, include_geo=True)
    by_track: Dict[int, List[dict]] = {}
    for row in rows:
        if row.get("track_id") is None:
            continue
        by_track.setdefault(int(row["track_id"]), []).append(row)

    labels = track_store.track_labels(target_folder, modality)
    times = _frame_times(target_folder, modality)
    votes = {task: {int(p["track_id"]): p for p in
                    classification_store.track_predictions(
                        target_folder, modality, task)}
             for task in ("species", "sex", classification_store.LIFE_STAGE)}
    try:
        partners = match_store.partner_tracks(target_folder, modality)
    except Exception:  # nosec B110 - no match store is no partner
        partners = {}
    # Annotations reach a row two ways: a label track imported from a
    # pipeline track (origin) that was not materialised, or one that was -
    # then the materialised track is the animal of record and the original
    # is superseded (track_store.superseded_track_ids).
    manual: Dict[int, List[dict]] = {}
    for label_track in label_store.load_tracks(target_folder, modality):
        materialised = label_track.get("track_id")
        origin = label_track.get("origin_track_id")
        key = materialised if materialised is not None else origin
        if key is not None:
            manual.setdefault(int(key), []).append(label_track)
    perpendicular = _perpendicular_distances(target_folder, modality)

    interpolated = {}
    for member in track_store.load_pixel_tracks(target_folder, modality):
        interpolated[int(member["detection_id"])] = int(
            member.get("interpolated") or 0)

    inventory = []
    for track_id in sorted(by_track):
        members = sorted(by_track[track_id],
                         key=lambda r: (r["frame"], r["detection_id"]))
        known = labels.get(track_id, {})
        attributes = common.resolve_attributes(
            json.dumps(known.get("attributes") or {}), enum_labels)

        # Species: the track's own, then what the classifier voted.
        species_id = known.get("species_id")
        if species_id is None:
            species_id = members[0].get("species_id")
        if wanted_species is not None and species_id not in wanted_species:
            continue
        species_vote = votes["species"].get(track_id)
        sex_vote = votes["sex"].get(track_id)
        age_vote = votes[classification_store.LIFE_STAGE].get(track_id)

        occlusion_labels = enum_labels.get("occlusion", {})
        n_occluded = n_clear = n_unknown = 0
        for row in members:
            level = common.resolve_attributes(
                row.get("attributes"), enum_labels).get("occlusion")
            if level is None:
                n_unknown += 1
            elif str(level).lower() == "occluded" or (
                    level in occlusion_labels.values()
                    and str(level).lower() != "clear"):
                n_occluded += 1
            else:
                n_clear += 1

        confidences = [r["confidence"] for r in members
                       if r.get("confidence") is not None]
        centres = [_centre(r) for r in members]
        placed = [c for c in centres if c is not None]
        start = placed[0] if placed else None
        end = placed[-1] if placed else None
        path_length = sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(placed, placed[1:]))
        displacement = (math.hypot(end[0] - start[0], end[1] - start[1])
                        if start and end else None)
        wgs = _to_wgs84([start and start[:2], end and end[:2]], epsg)

        frames = [r["frame"] for r in members]
        start_time = times.get(frames[0])
        end_time = times.get(frames[-1])
        t0, t1 = _seconds(start_time), _seconds(end_time)

        manual_tracks = manual.get(track_id, [])
        manual_species = sorted({
            species_names.get(lt.get("species_id"), "")
            for lt in manual_tracks if lt.get("species_id") is not None} - {""})
        manual_attributes = [common.resolve_attributes(
            json.dumps(lt.get("attributes") or {}), enum_labels)
            for lt in manual_tracks]

        def _manual(field: str) -> str:
            found = sorted({str(a[field]) for a in manual_attributes
                            if a.get(field) not in (None, "")})
            return "; ".join(found)

        extra = {k: v for k, v in attributes.items()
                 if k not in ("sex", "age", "approved")}

        inventory.append({
            "approved": bool(attributes.get("approved", False)),
            "track_id": track_id,
            "camera": _CAMERA_NAMES[modality],
            "species": species_names.get(species_id, ""),
            "species_vote_fraction": (species_vote or {}).get("fraction"),
            "species_votes": (f"{species_vote['votes']}/{species_vote['n']}"
                              if species_vote else ""),
            "species_model": (species_vote or {}).get("model", ""),
            "sex": attributes.get("sex", ""),
            "sex_vote_fraction": (sex_vote or {}).get("fraction"),
            "sex_model": (sex_vote or {}).get("model", ""),
            "age": attributes.get("age", ""),
            "age_source": ((age_vote or {}).get("evidence") or {}).get(
                "source", (age_vote or {}).get("model", "")),
            "n_boxes": len(members),
            "n_occluded": n_occluded,
            "n_clear": n_clear,
            "n_occlusion_unknown": n_unknown,
            "n_interpolated": sum(interpolated.get(int(r["detection_id"]), 0)
                                  for r in members),
            "n_georeferenced": len(placed),
            "mean_confidence": _mean(confidences),
            "min_confidence": min(confidences) if confidences else None,
            "max_confidence": max(confidences) if confidences else None,
            "start_frame": frames[0],
            "end_frame": frames[-1],
            "n_frames": len(set(frames)),
            "frame_span": frames[-1] - frames[0] + 1,
            "start_time": start_time if start_time is not None else "",
            "end_time": end_time if end_time is not None else "",
            "duration_s": (t1 - t0) if t0 is not None and t1 is not None
            else None,
            "start_x": start[0] if start else None,
            "start_y": start[1] if start else None,
            "start_z": start[2] if start else None,
            "end_x": end[0] if end else None,
            "end_y": end[1] if end else None,
            "end_z": end[2] if end else None,
            "crs": f"EPSG:{epsg}" if epsg and placed else "",
            "start_lat": wgs[0][0] if wgs[0] else None,
            "start_lon": wgs[0][1] if wgs[0] else None,
            "end_lat": wgs[1][0] if wgs[1] else None,
            "end_lon": wgs[1][1] if wgs[1] else None,
            "path_length_m": path_length if len(placed) > 1 else None,
            "displacement_m": displacement,
            "mean_box_width_px": _mean([abs(r["x2"] - r["x1"]) for r in members]),
            "mean_box_height_px": _mean([abs(r["y2"] - r["y1"]) for r in members]),
            "matched_track": partners.get(track_id),
            "matched_camera": (_CAMERA_NAMES[_OTHER_CAMERA[modality]]
                               if track_id in partners else ""),
            "label_track_ids": "; ".join(
                str(lt["label_track_id"]) for lt in manual_tracks),
            "label_species": "; ".join(manual_species),
            "label_sex": _manual("sex"),
            "label_age": _manual("age"),
            "perpendicular_distance_m": perpendicular.get(track_id),
            "attributes": json.dumps(extra, sort_keys=True) if extra else "",
        })
    return inventory


def summarise(inventory: List[dict]) -> List[str]:
    """Log lines: how many animals, of what, and how many were called."""
    if not inventory:
        return ["No tracks to list."]
    by_species: Dict[str, int] = {}
    for row in inventory:
        name = row["species"] or "unidentified"
        by_species[name] = by_species.get(name, 0) + 1
    parts = ", ".join(f"{n} {name}" for name, n in
                      sorted(by_species.items(), key=lambda x: -x[1]))
    lines = [f"{len(inventory)} tracked individual(s): {parts}"]
    sexed = sum(1 for row in inventory if row["sex"] not in ("", "unknown"))
    aged = sum(1 for row in inventory if row["age"] not in ("", "unknown"))
    matched = sum(1 for row in inventory if row["matched_track"] is not None)
    placed = sum(1 for row in inventory if row["start_x"] is not None)
    approved = sum(1 for row in inventory if row.get("approved"))
    lines.append(f"{sexed} sexed, {aged} aged, {matched} matched with the "
                 f"other camera, {placed} with a ground position")
    lines.append(f"{approved} of {len(inventory)} approved")
    return lines


def cell_text(value) -> str:
    """One value as it appears in the CSV and the report: ``None`` empty,
    floats trimmed of trailing zeros."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".") if math.isfinite(value) else ""
    return str(value)


_cell = cell_text


def write_csv(inventory: List[dict], path: str) -> str:
    """Write the inventory as CSV with the ``COLUMNS`` header."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for row in inventory:
            writer.writerow([_cell(row.get(column)) for column in COLUMNS])
    return path


def write_json(inventory: List[dict], path: str) -> str:
    """Write the inventory as a JSON list, ``None`` kept as ``null``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(inventory, fh, indent=2)
    return path


def read_json(path: str) -> List[dict]:
    """Read a written inventory back, for the QGIS layer."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else []
