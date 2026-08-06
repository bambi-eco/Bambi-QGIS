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

def _has_stage(target, kind, suffix):
    """True when a stage's store exists for this modality."""
    from . import store

    if suffix not in store.MODALITIES:
        return False
    return os.path.isfile(store.stage_path(target, kind, suffix))


def pixel_tracks_available(target, suffix):
    """True when there are tracks to draw.

    The overlay needs each box tied to a track id, which is what the store
    records; the 5.x files could only be paired back together by guesswork
    (§8.2), so they are not consulted.
    """
    from . import store

    return _has_stage(target, store.TRACKS, suffix)


def availability_warnings(params):
    """A human-readable warning for every selected option with no data behind
    it. The loaders draw nothing when data is absent, so these are warnings.

    Overlay data comes from the store, so the warnings name the step to run
    rather than a file to look for — a 5.x project has the files and still
    cannot be drawn from them.
    """
    from . import store

    target = params["target"]
    labels = {"t": "Thermal", "w": "RGB"}
    warns = []

    def exists(*parts):
        return os.path.exists(os.path.join(target, *parts))

    # ---- Video panels + their per-frame image source ------------------
    for s_ in params["video_suffixes"]:
        lbl = labels.get(s_, s_)
        if params["source"] == "extracted":
            if not exists(f"poses_{s_}.json") or not exists(f"frames_{s_}"):
                warns.append(f"{lbl} video: extracted frames missing "
                             f"(poses_{s_}.json / frames_{s_}/).")
        elif params["ortho_kind"] == "alfs":
            if not exists(f"alfs_{s_}", "alfs.tif"):
                warns.append(f"{lbl} video: ALFS orthophoto missing "
                             f"(alfs_{s_}/alfs.tif).")
        else:
            if not exists(f"geotiffs_{s_}"):
                warns.append(f"{lbl} video: per-frame geotiffs missing "
                             f"(geotiffs_{s_}/).")

    # ---- Overlays (drawn only on extracted frames) --------------------
    if params["source"] == "extracted":
        for s_ in params["video_suffixes"]:
            lbl = labels.get(s_, s_)
            if (params["overlay"] == "detections"
                    and not _has_stage(target, store.DETECTIONS, s_)):  # noqa: W503, W504
                warns.append(f"{lbl} detections overlay: no detections "
                             "stored — run 'Detect Animals'.")
            elif (params["overlay"] == "tracks"
                  and not pixel_tracks_available(target, s_)):  # noqa: W503, W504
                warns.append(f"{lbl} tracks overlay: no tracks stored — run "
                             "'Track Animals'.")

    # ---- Map panel ----------------------------------------------------
    if params["map"]:
        cam = params["map_camera"]
        if params["map_flight"] and not exists(f"flight_route_{cam}", "camera_positions.geojson"):
            warns.append("Map flight path: "
                         f"flight_route_{cam}/camera_positions.geojson missing.")
        if params["map_fov"] and not _has_stage(target, store.FOV, cam):
            warns.append("Map field of view: none stored — run 'Calculate "
                         "Field of View'.")
        if params["map_det"] and not _has_stage(target, store.GEOREFERENCED, cam):
            warns.append("Map detections: none geo-referenced — run "
                         "'Geo-Reference Detections'.")
        if params["map_trk"] and not _has_stage(target, store.TRACKS, cam):
            warns.append("Map tracks: no tracks stored — run 'Track Animals'.")
        if params["map_perp"] and not (
                exists(f"flight_route_{cam}", f"perpendicular_{cam}.json")
                or exists(f"flight_route_{cam}", "perpendicular.json")):  # noqa: W503, W504
            warns.append("Map perpendicular distances: "
                         f"flight_route_{cam}/perpendicular_{cam}.json missing.")

    # ---- Info panel ---------------------------------------------------
    if params["info"]:
        cam = params["info_camera"]
        lbl = labels.get(cam, cam)
        if params["info_dets"] and not _has_stage(target, store.DETECTIONS, cam):
            warns.append(f"Info panel detections ({lbl}): none stored — run "
                         "'Detect Animals'.")
        if params["info_tracks"] and not pixel_tracks_available(target, cam):
            warns.append(f"Info panel tracks ({lbl}): no tracks stored — run "
                         "'Track Animals'.")
        if params["info_area"] and not _has_stage(target, store.FOV, cam):
            warns.append(f"Info panel monitored area ({lbl}): none stored — "
                         "run 'Calculate Field of View'.")

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
    """frame -> list of (x1, y1, x2, y2, class_id) in extracted-frame pixels."""
    from . import store, track_store

    if not _has_stage(target, store.DETECTIONS, suffix):
        return {}
    out = {}
    for row in track_store.load_detections(target, suffix):
        out.setdefault(row["frame"], []).append(
            (row["x1"], row["y1"], row["x2"], row["y2"],
             str(row["species_id"])))
    return out


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
    """frame -> list of (track_id, x1, y1, x2, y2) in extracted-frame pixels.

    A join: membership is recorded as (track_id, detection_id), so no pairing
    of any kind is needed. This used to try three reconstructions below the
    store — a pixel CSV, a geo-coordinate match through georeferenced.txt, and
    finally a row-index match — each of which could silently mis-assign a box
    to the wrong animal when the files were re-sorted or a detection was
    dropped (§8.2).
    """
    stored = pixel_tracks_from_store(target, suffix)
    if not stored and log_fn:
        log_fn("No tracks in the store for this camera. Run tracking, or use "
               "'Migrate 5.x…' on the Input tab for an older project.")
    return stored


def load_fov_polygons(target, suffix):
    """frame -> list of (x, y). Geo coordinates in the data CRS."""
    from . import fov_store

    return {frame: [(x, y) for x, y, _z in points]
            for frame, points in fov_store.load_fov(target, suffix).items()
            if len(points) >= 3}


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
    """frame -> list of {tid, x1, y1, x2, y2} in world coordinates."""
    from . import pipeline_outputs

    out = {}
    tracks = pipeline_outputs.load_geo_tracks_by_id(
        os.path.join(target, f"tracks_{suffix}", "tracks.csv"))
    for track_id, points in tracks.items():
        for point in points:
            out.setdefault(point["frame"], []).append({
                "tid": track_id,
                "x1": point["x1"], "y1": point["y1"],
                "x2": point["x2"], "y2": point["y2"],
            })
    return out


def load_geo_detections(target, suffix):
    """frame -> list of {tid, x1, y1, x2, y2} in world coordinates.

    A per-frame running index stands in for a track id, purely to colour the
    boxes apart; these are detections and have no tracks.
    """
    from . import pipeline_outputs

    by_frame = pipeline_outputs.load_georef_detections_by_frame(
        os.path.join(target, f"georeferenced_{suffix}", "georeferenced.txt"))
    out = {}
    for frame, rows in by_frame.items():
        lst = out.setdefault(frame, [])
        for row in rows:
            lst.append({"tid": len(lst),
                        "x1": row["x1"], "y1": row["y1"],
                        "x2": row["x2"], "y2": row["y2"]})
    return out


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
