# -*- coding: utf-8 -*-
"""Headless data collection and layout maths of the Video Creator.

Moved from ``bambi_video_creator.py`` (whose ``VideoCreatorDialog`` methods
now delegate here): track colours, availability checks against the target
folder, all pipeline-output loaders including the geo-join track pairing,
and the map-canvas coordinate/formatting helpers. The cv2 rendering loop
stays in the dialog.

Functions that reported through the dialog log take an optional
``log_fn(message)`` callable.
"""

import os
import json
from typing import Callable, Optional

TRACK_COLORS = [
    (229, 57, 53), (142, 36, 170), (30, 136, 229), (0, 172, 193),
    (67, 160, 71), (251, 140, 0), (109, 76, 65), (3, 155, 229),
    (216, 27, 96), (94, 53, 177), (0, 137, 123), (192, 202, 51),
]


def track_color_bgr(track_id):
    r, g, b = TRACK_COLORS[int(track_id) % len(TRACK_COLORS)]
    return (b, g, r)


def id_to_color(identifier, saturation=0.65, lightness=0.5):
    """Deterministic BGR colour for a track/detection id (mirrors the TRex tool)."""
    import colorsys
    import hashlib
    h = hashlib.sha256(str(identifier).encode("utf-8")).digest()
    hue = int.from_bytes(h[:4], "big") / 2 ** 32
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return (int(b * 255), int(g * 255), int(r * 255))


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def pixel_tracks_available(target, suffix):
    """True if any source :func:`load_pixel_tracks` could use is present:
    a direct pixel-track CSV, or the detections + tracks.csv needed for the
    geo-join / line-index fallbacks."""
    for folder in (f"tracks_{suffix}", f"tracks_pixel_{suffix}"):
        for fname in ("tracks_pixel_undistorted.csv", "tracks_pixel.csv"):
            if os.path.exists(os.path.join(target, folder, fname)):
                return True
    return (os.path.exists(os.path.join(target, f"detections_{suffix}", "detections.txt"))
            and os.path.exists(os.path.join(target, f"tracks_{suffix}", "tracks.csv")))  # noqa: W503, W504


def availability_warnings(params):
    """Return a human-readable warning for every selected option whose
    backing data is missing on disk. Loaders degrade gracefully (draw
    nothing) when data is absent, so these are warnings, not errors."""
    target = params["target"]
    labels = {"t": "Thermal", "w": "RGB"}
    warns = []

    def exists(*parts):
        return os.path.exists(os.path.join(target, *parts))

    # ---- Video panels + their per-frame image source ------------------
    for s in params["video_suffixes"]:
        lbl = labels.get(s, s)
        if params["source"] == "extracted":
            if not exists(f"poses_{s}.json") or not exists(f"frames_{s}"):
                warns.append(f"{lbl} video: extracted frames missing "
                             f"(poses_{s}.json / frames_{s}/).")
        elif params["ortho_kind"] == "alfs":
            if not exists(f"alfs_{s}", "alfs.tif"):
                warns.append(f"{lbl} video: ALFS orthophoto missing "
                             f"(alfs_{s}/alfs.tif).")
        else:
            if not exists(f"geotiffs_{s}"):
                warns.append(f"{lbl} video: per-frame geotiffs missing "
                             f"(geotiffs_{s}/).")

    # ---- Overlays (drawn only on extracted frames) --------------------
    if params["source"] == "extracted":
        for s in params["video_suffixes"]:
            lbl = labels.get(s, s)
            if (params["overlay"] == "detections"
                    and not exists(f"detections_{s}", "detections.txt")):  # noqa: W503, W504
                warns.append(f"{lbl} detections overlay: "
                             f"detections_{s}/detections.txt missing.")
            elif (params["overlay"] == "tracks"
                  and not pixel_tracks_available(target, s)):  # noqa: W503, W504
                warns.append(f"{lbl} tracks overlay: no track data found.")

    # ---- Map panel ----------------------------------------------------
    if params["map"]:
        cam = params["map_camera"]
        if params["map_flight"] and not exists(f"flight_route_{cam}", "camera_positions.geojson"):
            warns.append("Map flight path: "
                         f"flight_route_{cam}/camera_positions.geojson missing.")
        if params["map_fov"] and not exists(f"fov_{cam}", "fov_polygons.txt"):
            warns.append(f"Map field of view: fov_{cam}/fov_polygons.txt missing.")
        if params["map_det"] and not exists(f"georeferenced_{cam}", "georeferenced.txt"):
            warns.append("Map detections: "
                         f"georeferenced_{cam}/georeferenced.txt missing.")
        if params["map_trk"] and not exists(f"tracks_{cam}", "tracks.csv"):
            warns.append(f"Map tracks: tracks_{cam}/tracks.csv missing.")
        if params["map_perp"] and not (
                exists(f"flight_route_{cam}", f"perpendicular_{cam}.json")
                or exists(f"flight_route_{cam}", "perpendicular.json")):  # noqa: W503, W504
            warns.append("Map perpendicular distances: "
                         f"flight_route_{cam}/perpendicular_{cam}.json missing.")

    # ---- Info panel ---------------------------------------------------
    if params["info"]:
        cam = params["info_camera"]
        lbl = labels.get(cam, cam)
        if params["info_dets"] and not exists(f"detections_{cam}", "detections.txt"):
            warns.append(f"Info panel detections ({lbl}): "
                         f"detections_{cam}/detections.txt missing.")
        if params["info_tracks"] and not pixel_tracks_available(target, cam):
            warns.append(f"Info panel tracks ({lbl}): no track data found.")
        if params["info_area"] and not exists(f"fov_{cam}", "fov_polygons.txt"):
            warns.append(f"Info panel monitored area ({lbl}): "
                         f"fov_{cam}/fov_polygons.txt missing.")

    return warns


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_poses(target, suffix, log_fn: Optional[Callable[[str], None]] = None):
    """Return the ordered imagefile list for a camera, or [] if absent."""
    path = os.path.join(target, f"poses_{suffix}.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [img.get("imagefile", "") for img in data.get("images", [])]
    except Exception as exc:  # noqa: BLE001
        if log_fn:
            log_fn(f"Warning: could not read {path}: {exc}")
        return []


def load_pixel_detections(target, suffix):
    """frame -> list of (x1, y1, x2, y2, class_id)."""
    path = os.path.join(target, f"detections_{suffix}", "detections.txt")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 5:
                continue
            frame = int(float(p[0]))
            out.setdefault(frame, []).append(
                (float(p[1]), float(p[2]), float(p[3]), float(p[4]),
                 p[6] if len(p) > 6 else "0"))
    return out


def load_detection_rows(target, suffix):
    """Ordered list of (frame, x1, y1, x2, y2) pixel detections, in file order."""
    path = os.path.join(target, f"detections_{suffix}", "detections.txt")
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 5:
                continue
            try:
                rows.append((int(float(p[0])), float(p[1]), float(p[2]),
                             float(p[3]), float(p[4])))
            except ValueError:
                continue
    return rows


def load_track_id_rows(target, suffix):
    """Ordered list of (frame, track_id) from the track output, in file order,
    excluding interpolated rows so it stays aligned with the detection file.

    Prefers the pixel track CSV when present, else the geo-referenced
    ``tracks.csv`` (our tracks are typically only geo-referenced)."""
    for fname, interp_col in (("tracks_pixel.csv", 8), ("tracks.csv", 10)):
        path = os.path.join(target, f"tracks_{suffix}", fname)
        if not os.path.exists(path):
            continue
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = line.split(",")
                if len(p) < 2:
                    continue
                try:
                    frame = int(float(p[0]))
                    tid = int(float(p[1]))
                    interp = int(float(p[interp_col])) if len(p) > interp_col else 0
                except (ValueError, IndexError):
                    continue
                if interp:
                    continue
                rows.append((frame, tid))
        return rows
    return []


def load_georef_rows(target, suffix):
    """Ordered list of (frame, x1, y1, x2, y2) geo detections, in file order.

    Row format: idx frame x1 y1 z1 x2 y2 z2 conf cls."""
    path = os.path.join(target, f"georeferenced_{suffix}", "georeferenced.txt")
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            try:
                rows.append((int(p[1]), float(p[2]), float(p[3]),
                             float(p[5]), float(p[6])))
            except (ValueError, IndexError):
                continue
    return rows


def parse_pixel_tracks_csv(path, interp_col):
    """frame -> list of (track_id, x1, y1, x2, y2) from a pixel MOT CSV."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split(",")
            if len(p) < 6:
                continue
            try:
                if len(p) > interp_col and int(float(p[interp_col])):
                    continue
                out.setdefault(int(float(p[0])), []).append(
                    (int(float(p[1])), float(p[2]), float(p[3]),
                     float(p[4]), float(p[5])))
            except (ValueError, IndexError):
                continue
    return out


def coord_key(coords):
    return tuple(round(v, 3) for v in coords)


def pixel_tracks_from_store(target, suffix):
    """frame -> [(track_id, x1, y1, x2, y2)] from the 6.0 store, or ``{}``.

    Replaces the reconstruction the overlay used to rely on. The old path also
    paired ``tracks_pixel.csv`` rows with ``detections.txt`` rows by position
    (``load_track_id_rows``), which broke in exactly the way
    ``core/track_export.py`` did whenever a detection was dropped.
    """
    from . import store, track_store

    if suffix not in store.MODALITIES:
        return {}
    rows = track_store.load_pixel_tracks(target, suffix)
    out = {}
    for row in rows:
        out.setdefault(row["frame"], []).append(
            (row["track_id"], row["x1"], row["y1"], row["x2"], row["y2"]))
    return out


def load_pixel_tracks(target, suffix, log_fn: Optional[Callable[[str], None]] = None):
    """frame -> list of (track_id, x1, y1, x2, y2) in extracted-frame pixel space.

    Strategy, in order of reliability:
    0. The 6.0 store, where membership is recorded as (track_id, detection_id)
       and the answer is a join — no matching of any kind.
    1. A pixel-space track CSV if one exists (exact boxes + ids).
    2. Otherwise pair the geo-referenced tracks to the pixel detections by
       matching geo coordinates through georeferenced.txt. This is order-
       independent, so it is correct even when tracks.csv is re-sorted by
       (frame, track_id) or contains interpolated rows.
    3. As a last resort, pair strictly by line index (valid only when the
       detection and track files share the same ordering, e.g. TRex import).

    Steps 1–3 all reconstruct something the pipeline knew and threw away; they
    remain only for projects with no store (EXCHANGE_FORMAT_PLAN.md §8.2).
    """
    # 0. The store. Exact, and unaffected by dropped detections or re-sorting.
    stored = pixel_tracks_from_store(target, suffix)
    if stored:
        return stored

    # 1. Direct pixel tracks. Prefer the undistorted file, which matches the
    # extracted (undistorted) frames the overlay is drawn on; the plain
    # tracks_pixel.csv is in raw video space for the TRex pipelines.
    for folder in (f"tracks_{suffix}", f"tracks_pixel_{suffix}"):
        for fname in ("tracks_pixel_undistorted.csv", "tracks_pixel.csv"):
            path = os.path.join(target, folder, fname)
            if os.path.exists(path):
                return parse_pixel_tracks_csv(path, 8)

    # 2. Geo-coordinate join.
    out = pair_tracks_via_geo(target, suffix)
    if out:
        return out

    # 3. Line-index fallback.
    return pair_tracks_by_line_index(target, suffix, log_fn=log_fn)


def pair_tracks_via_geo(target, suffix):
    """Assign track ids to pixel detections by matching geo coordinates.

    detections.txt and georeferenced.txt share the same per-frame ordering
    (georeferenced is built from the detections in order), so the k-th pixel
    box in a frame corresponds to the k-th geo box; that geo box's coordinates
    then look up the track id in tracks.csv regardless of its row order."""
    from collections import defaultdict

    det_rows = load_detection_rows(target, suffix)
    geo_rows = load_georef_rows(target, suffix)
    if not det_rows or not geo_rows:
        return {}

    # Track id keyed by (frame, rounded geo box) from the geo tracks csv.
    trk_lookup = {}
    tpath = os.path.join(target, f"tracks_{suffix}", "tracks.csv")
    if not os.path.exists(tpath):
        return {}
    with open(tpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split(",")
            if len(p) < 7:
                continue
            try:
                if len(p) > 10 and int(float(p[10])):
                    continue  # interpolated row, no detection
                frame = int(float(p[0]))
                tid = int(float(p[1]))
                key = (frame, coord_key(
                    (float(p[2]), float(p[3]), float(p[5]), float(p[6]))))
            except (ValueError, IndexError):
                continue
            trk_lookup.setdefault(key, tid)
    if not trk_lookup:
        return {}

    det_by_frame = defaultdict(list)
    for (frame, x1, y1, x2, y2) in det_rows:
        det_by_frame[frame].append((x1, y1, x2, y2))
    geo_by_frame = defaultdict(list)
    for (frame, x1, y1, x2, y2) in geo_rows:
        geo_by_frame[frame].append((x1, y1, x2, y2))

    out = {}
    matched = 0
    for frame, boxes in det_by_frame.items():
        geos = geo_by_frame.get(frame, [])
        if len(geos) != len(boxes):
            continue  # cannot align this frame safely (dropped detections)
        for box, geo in zip(boxes, geos):
            tid = trk_lookup.get((frame, coord_key(geo)))
            if tid is not None:
                out.setdefault(frame, []).append((tid,) + box)
                matched += 1
    return out if matched else {}


def pair_tracks_by_line_index(target, suffix,
                              log_fn: Optional[Callable[[str], None]] = None):
    """Pair pixel detections with track ids by shared line index (last resort)."""
    det_rows = load_detection_rows(target, suffix)
    id_rows = load_track_id_rows(target, suffix)
    out = {}
    if not det_rows or not id_rows:
        return out
    n = min(len(det_rows), len(id_rows))
    skipped = 0
    for i in range(n):
        dframe, x1, y1, x2, y2 = det_rows[i]
        tframe, tid = id_rows[i]
        if tframe != dframe:
            skipped += 1
            continue
        out.setdefault(dframe, []).append((tid, x1, y1, x2, y2))
    if skipped and log_fn:
        log_fn(
            f"Tracks overlay ({suffix}): {skipped}/{n} rows skipped "
            "(detection/track lines out of sync).")
    return out


def load_fov_polygons(target, suffix):
    """frame -> list of (x, y). Geo coordinates in the data CRS."""
    path = os.path.join(target, f"fov_{suffix}", "fov_polygons.txt")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 2:
                continue
            frame = int(p[0])
            n = int(p[1])
            pts = []
            for i in range(n):
                idx = 2 + i * 3
                if idx + 2 < len(p):
                    pts.append((float(p[idx]), float(p[idx + 1])))
            if len(pts) >= 3:
                out[frame] = pts
    return out


def load_perpendicular(target, suffix):
    """frame -> list of ((cx, cy), (fx, fy), distance)."""
    path = os.path.join(target, f"flight_route_{suffix}", f"perpendicular_{suffix}.json")
    if not os.path.exists(path):
        # Legacy name from before the perpendicular outputs were suffixed
        # with the detection camera.
        path = os.path.join(target, f"flight_route_{suffix}", "perpendicular.json")
    out = {}
    if not os.path.exists(path):
        return out
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001
        return out
    for r in data.get("perpendiculas", []):
        frame = int(r.get("frame", -1))
        c = r.get("detection_center") or []
        foot = r.get("foot_point") or []
        if len(c) >= 2 and len(foot) >= 2:
            out.setdefault(frame, []).append(
                ((c[0], c[1]), (foot[0], foot[1]), r.get("distance", 0.0)))
    return out


def load_camera_positions(target, suffix):
    """frame_idx -> (x, y) per-frame camera positions in the data CRS, from
    flight_route_{suffix}/camera_positions.geojson (one Point per pose). Used
    to reveal the flight path progressively, frame by frame."""
    path = os.path.join(target, f"flight_route_{suffix}", "camera_positions.geojson")
    out = {}
    if not os.path.exists(path):
        return out
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001
        return out
    for i, feat in enumerate(data.get("features", [])):
        coords = feat.get("geometry", {}).get("coordinates", [])
        if len(coords) < 2:
            continue
        frame_idx = feat.get("properties", {}).get("frame_idx", i)
        out[int(frame_idx)] = (coords[0], coords[1])
    return out


def load_geo_tracks(target, suffix):
    """frame -> list of {tid, x1, y1, x2, y2} from tracks_{suffix}/tracks.csv."""
    path = os.path.join(target, f"tracks_{suffix}", "tracks.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split(",")
            if len(p) < 7:
                continue
            # frame, tid, x1, y1, z1, x2, y2, z2, ...
            frame = int(float(p[0]))
            out.setdefault(frame, []).append({
                "tid": int(float(p[1])),
                "x1": float(p[2]), "y1": float(p[3]),
                "x2": float(p[5]), "y2": float(p[6]),
            })
    return out


def load_geo_detections(target, suffix):
    """frame -> list of {tid, x1, y1, x2, y2} from georeferenced_{suffix}.

    Row format (space-separated): idx frame x1 y1 z1 x2 y2 z2 conf cls.
    A per-frame running index is used as a pseudo track id for colouring."""
    path = os.path.join(target, f"georeferenced_{suffix}", "georeferenced.txt")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            try:
                frame = int(p[1])
                x1, y1 = float(p[2]), float(p[3])
                x2, y2 = float(p[5]), float(p[6])
            except (ValueError, IndexError):
                continue
            if x1 < 0 or y1 < 0:
                continue
            lst = out.setdefault(frame, [])
            lst.append({"tid": len(lst), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out


# ---------------------------------------------------------------------------
# Map canvas geometry / formatting
# ---------------------------------------------------------------------------

def pad_extent_to_aspect(extent, width, height, margin):
    min_x, max_x, min_y, max_y = extent
    draw_w = width - 2 * margin
    draw_h = height - 2 * margin
    target_ar = draw_w / draw_h
    data_w = max(max_x - min_x, 1e-6)
    data_h = max(max_y - min_y, 1e-6)
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    if data_w / data_h > target_ar:
        new_h = data_w / target_ar
        min_y, max_y = cy - new_h / 2, cy + new_h / 2
    else:
        new_w = data_h * target_ar
        min_x, max_x = cx - new_w / 2, cx + new_w / 2
    return (min_x, max_x, min_y, max_y)


def make_canvas_cfg(extent, width, height, margin):
    min_x, max_x, min_y, max_y = extent
    span_x = max(max_x - min_x, 1e-6)
    return {
        "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y,
        "scale": (width - 2 * margin) / span_x,
        "margin": margin, "width": width, "height": height,
    }


def world_to_canvas(x, y, cfg):
    px = int(cfg["margin"] + (x - cfg["min_x"]) * cfg["scale"])
    py = int(cfg["height"] - (cfg["margin"] + (y - cfg["min_y"]) * cfg["scale"]))
    return px, py


def fmt_area(area_m2):
    if area_m2 >= 1e6:
        return f"{area_m2 / 1e6:.2f} km2"
    if area_m2 >= 1e4:
        return f"{area_m2 / 1e4:.2f} ha"
    return f"{area_m2:.0f} m2"


def fmt_area_ratio(area_m2, total_m2):
    """Format 'observed / total unit (pct)' with a shared unit picked from
    the total, mirroring the 'x / n' style used for detections/tracks."""
    if total_m2 >= 1e6:
        div, unit, dec = 1e6, "km2", 2
    elif total_m2 >= 1e4:
        div, unit, dec = 1e4, "ha", 2
    else:
        div, unit, dec = 1.0, "m2", 0
    pct = (area_m2 / total_m2 * 100.0) if total_m2 > 0 else 0.0
    return (f"{area_m2 / div:.{dec}f} / {total_m2 / div:.{dec}f} {unit} "
            f"({pct:.0f}%)")
