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
    rows = [row for row in common.load_detections(
        target_folder, modality, include_not_an_animal)
        if row.get("gx1") is not None]

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
        log_fn(f"GeoJSON: {len(features)} feature(s) → {output_path}")
    return output_path


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
        raise common.ExportError(
            "No tracks to export — run tracking (or import tracklets) first.")

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
