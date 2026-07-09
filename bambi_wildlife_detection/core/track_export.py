# -*- coding: utf-8 -*-
"""Derive a pixel-space MOT tracks file from geo-referenced tracks.

Moved from ``BambiProcessor._write_pixel_tracks_from_geo`` (which
delegates here). The built-in tracker runs in geo space and only writes
``tracks.csv``; this reconstructs the pixel-space ``tracks_pixel.csv`` the
advanced / TRex pipelines emit, so downstream tools see a uniform output.

The reconstruction relies on ``detections.txt`` and ``georeferenced.txt``
sharing the same per-frame ordering (georeferenced is built from the
detections in order): the k-th pixel box maps to the k-th geo box, whose
coordinates then look up the assigned track id in ``tracks.csv``. This is
independent of the (frame, track_id) re-sorting applied to tracks.csv.
"""

import os
from collections import defaultdict


def _key(vals):
    return tuple(round(v, 3) for v in vals)


def write_pixel_tracks_from_geo(target_folder: str, camera_suffix: str,
                                log_fn=None) -> str:
    """Create ``tracks_{cam}/tracks_pixel.csv`` from geo-referenced tracks.

    Returns the written file path, or ``""`` when nothing was written (a
    pixel tracks file already exists, inputs are missing, or no track id
    could be matched).
    """
    tracks_folder = os.path.join(target_folder, f"tracks_{camera_suffix}")
    pixel_file = os.path.join(tracks_folder, "tracks_pixel.csv")
    if os.path.exists(pixel_file):
        return ""  # already produced by the tracker backend

    det_file = os.path.join(target_folder, f"detections_{camera_suffix}", "detections.txt")
    georef_file = os.path.join(target_folder, f"georeferenced_{camera_suffix}", "georeferenced.txt")
    tracks_file = os.path.join(tracks_folder, "tracks.csv")
    if not (os.path.exists(det_file) and os.path.exists(georef_file) and os.path.exists(tracks_file)):
        return ""

    # Pixel detections grouped by frame, in file order.
    det_by_frame = defaultdict(list)
    with open(det_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 5:
                continue
            try:
                det_by_frame[int(p[0])].append((
                    float(p[1]), float(p[2]), float(p[3]), float(p[4]),
                    float(p[5]) if len(p) > 5 else 1.0,
                    int(p[6]) if len(p) > 6 else 0))
            except ValueError:
                continue

    # Geo detections grouped by frame, in file order (same order as detections).
    geo_by_frame = defaultdict(list)
    with open(georef_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            try:
                geo_by_frame[int(p[1])].append(
                    (float(p[2]), float(p[3]), float(p[5]), float(p[6])))
            except (ValueError, IndexError):
                continue

    # Track id keyed by (frame, rounded geo box); skip interpolated rows.
    trk_lookup = {}
    with open(tracks_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split(",")
            if len(p) < 7:
                continue
            try:
                if len(p) > 10 and int(float(p[10])):
                    continue
                frame = int(float(p[0]))
                tid = int(float(p[1]))
                key = (frame, _key((float(p[2]), float(p[3]),
                                    float(p[5]), float(p[6]))))
            except (ValueError, IndexError):
                continue
            trk_lookup.setdefault(key, tid)

    if not trk_lookup:
        return ""

    rows = []
    skipped_frames = 0
    for frame, boxes in det_by_frame.items():
        geos = geo_by_frame.get(frame, [])
        if len(geos) != len(boxes):
            skipped_frames += 1  # dropped detections break positional alignment
            continue
        for box, geo in zip(boxes, geos):
            tid = trk_lookup.get((frame, _key(geo)))
            if tid is None:
                continue
            x1, y1, x2, y2, conf, cls = box
            rows.append((frame, tid, x1, y1, x2, y2, conf, cls))

    if not rows:
        return ""

    rows.sort(key=lambda r: (r[0], r[1]))
    os.makedirs(tracks_folder, exist_ok=True)
    with open(pixel_file, "w", encoding="utf-8") as f:
        f.write("# frame,track_id,x1,y1,x2,y2,conf,cls,interpolated\n")
        for (frame, tid, x1, y1, x2, y2, conf, cls) in rows:
            f.write(f"{frame},{tid},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},"
                    f"{conf:.4f},{cls},0\n")

    if log_fn:
        msg = f"Wrote pixel-space tracks: {pixel_file} ({len(rows)} rows)"
        if skipped_frames:
            msg += (f"; {skipped_frames} frame(s) skipped "
                    "(detection/geo count mismatch)")
        log_fn(msg)

    return pixel_file
