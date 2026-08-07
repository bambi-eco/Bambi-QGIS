# -*- coding: utf-8 -*-
"""GeoJSON and TRex ``.npz`` exporters (EXCHANGE_FORMAT_PLAN.md §8.1).

GeoJSON keeps ``not-an-animal`` by default: it is a record of what was seen and
decided, not a training set, so a labelled false positive is information rather
than noise.

The TRex writer closes the loop with the importer that has existed since 5.x —
BAMBI could read TRex tracklets but never write them.
"""

import json
import os
from typing import List, Optional

from . import common


def export_geojson(target_folder: str, modality: str, output_path: str,
                   epsg: Optional[int] = None,
                   include_not_an_animal: bool = True,
                   tracks_only: bool = False, log_fn=None) -> str:
    """Write geo-referenced detections (or tracks) as GeoJSON.

    Detections become point features at the centre of their world box; tracks
    become LineStrings through their detections in frame order. Custom fields
    land in ``properties`` with enum values resolved to labels.
    """
    vocabulary = common.load_vocabulary(target_folder)
    all_rows = common.load_detections(
        target_folder, modality, include_not_an_animal)
    rows = [row for row in all_rows if row.get("gx1") is not None]

    features = []
    if tracks_only:
        for track_id, points in sorted(common.tracks_of(rows).items()):
            ordered = sorted(points, key=lambda r: r["frame"])
            coordinates = [[(p["gx1"] + p["gx2"]) / 2.0,
                            (p["gy1"] + p["gy2"]) / 2.0] for p in ordered]
            if len(coordinates) < 2:
                continue
            first = ordered[0]
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "properties": {
                    "track_id": track_id,
                    "species": vocabulary["species"].get(first["species_id"], ""),
                    "species_id": first["species_id"],
                    "n_detections": len(ordered),
                    "first_frame": ordered[0]["frame"],
                    "last_frame": ordered[-1]["frame"],
                    **common.resolve_attributes(
                        first["attributes"], vocabulary["enum_labels"]),
                },
            })
    else:
        for row in rows:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [(row["gx1"] + row["gx2"]) / 2.0,
                                    (row["gy1"] + row["gy2"]) / 2.0],
                },
                "properties": {
                    "detection_id": row["detection_id"],
                    "frame": row["frame"],
                    "track_id": row.get("track_id"),
                    "species": vocabulary["species"].get(row["species_id"], ""),
                    "species_id": row["species_id"],
                    "confidence": row["confidence"],
                    **common.resolve_attributes(
                        row["attributes"], vocabulary["enum_labels"]),
                },
            })

    document = {"type": "FeatureCollection", "features": features}
    if epsg:
        document["crs"] = {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"},
        }

    common.ensure_folder(os.path.dirname(os.path.abspath(output_path)))
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2)

    if log_fn:
        message = f"GeoJSON: {len(features)} feature(s) → {output_path}"
        if not features and all_rows:
            if not rows:
                # Everything was dropped before tracks were even considered.
                message += (f" — none of the {len(all_rows)} detection(s) are "
                            "geo-referenced. Run 'Geo-reference detections'.")
            elif tracks_only:
                message += common.no_tracks_hint(target_folder, modality)
        log_fn(message)
    return output_path


def export_segmentation_geojson(target_folder: str, modality: str,
                                output_path: str, epsg: Optional[int] = None,
                                include_not_an_animal: bool = True,
                                log_fn=None) -> str:
    """Write the geo-referenced segmentation masks as GeoJSON polygons.

    The animal export describes *where* an animal was, as a point or a track
    line; this one describes the shape SAM3 outlined. They are different
    geometries answering different questions, so they are separate files rather
    than one mixed collection that no GIS tool could style sensibly.

    Masks that were never geo-referenced are skipped and reported — a
    pixel-space polygon has no place in a world-coordinate document.
    """
    from .. import store

    path = store.stage_path(target_folder, store.SEGMENTATION, modality)
    if not os.path.isfile(path):
        raise common.ExportError(
            f"No segmentation stored for modality '{modality}'. Run SAM3 "
            "segmentation first.")

    conn = store.open_store(path, store.SEGMENTATION, modality)
    try:
        rows = [dict(row) for row in conn.execute(
            "SELECT segment_id, detection_id, frame, polygon_geo, attributes "
            "FROM segments ORDER BY frame, segment_id")]
    finally:
        conn.close()

    features, without_world = [], 0
    for row in rows:
        rings = _polygon_rings(row.get("polygon_geo"))
        if not rings:
            without_world += 1
            continue
        properties = {
            "segment_id": row["segment_id"],
            "detection_id": row.get("detection_id"),
            "frame": row["frame"],
        }
        properties.update(_segment_attributes(row.get("attributes")))
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": rings},
            "properties": properties,
        })

    document = {"type": "FeatureCollection", "features": features}
    if epsg:
        document["crs"] = {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"},
        }

    common.ensure_folder(os.path.dirname(os.path.abspath(output_path)))
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2)

    if log_fn:
        message = f"GeoJSON: {len(features)} segmentation(s) → {output_path}"
        if without_world:
            message += (f" — {without_world} mask(s) have no world "
                        "coordinates. Run 'Geo-Reference Segmentation'.")
        log_fn(message)
    return output_path


def _polygon_rings(raw):
    """GeoJSON ring list for a stored polygon, or ``None`` if unusable.

    The store keeps whatever the segmenter produced, so this accepts both a
    bare ring and an already-nested ring list, and refuses anything too short
    to be a polygon.
    """
    if not raw:
        return None
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None

    # [[x, y], …] is one ring; [[[x, y], …], …] is already a ring list.
    first = parsed[0]
    nested = isinstance(first, list) and first and isinstance(first[0], list)
    rings = parsed if nested else [parsed]

    cleaned = []
    for ring in rings:
        points = [list(point[:2]) for point in ring
                  if isinstance(point, (list, tuple)) and len(point) >= 2]
        if len(points) < 3:
            continue
        # GeoJSON polygons are closed; the store need not be.
        if points[0] != points[-1]:
            points.append(list(points[0]))
        cleaned.append(points)
    return cleaned or None


def _segment_attributes(raw) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def export_trex_npz(target_folder: str, modality: str, output_folder: str,
                    include_not_an_animal: bool = False, log_fn=None) -> List[str]:
    """Write one ``.npz`` per track, in the shape the TRex importer reads.

    Closes the loop with ``run_trex_import``: BAMBI has been able to read TRex
    tracklets since 5.x and never to write them, so tracks corrected here could
    not go back the way they came.
    """
    import numpy as np

    rows = common.load_detections(
        target_folder, modality, include_not_an_animal, include_geo=False)
    grouped = common.tracks_of(rows)
    if not grouped:
        hint = common.no_tracks_hint(target_folder, modality)
        raise common.ExportError(f"No tracks to export.{hint}")

    common.ensure_folder(output_folder)
    written = []
    for track_id, points in sorted(grouped.items()):
        ordered = sorted(points, key=lambda r: r["frame"])
        path = os.path.join(output_folder, f"track_{track_id}.npz")
        np.savez(
            path,
            frame=np.array([p["frame"] for p in ordered], dtype=np.int64),
            X=np.array([[p["x1"], p["x2"]] for p in ordered], dtype=np.float32),
            Y=np.array([[p["y1"], p["y2"]] for p in ordered], dtype=np.float32),
            conf=np.array([p["confidence"] for p in ordered], dtype=np.float32),
            cls=np.array([p["species_id"] for p in ordered], dtype=np.int64),
        )
        written.append(path)

    if log_fn:
        log_fn(f"TRex: {len(written)} tracklet file(s) → {output_folder}")
    return written
