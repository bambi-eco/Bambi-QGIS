# -*- coding: utf-8 -*-
"""Headless logic of the BAMBI click/inspection tool.

Moved from ``bambi_click_tool.py`` (whose ``BambiClickTool`` methods now
delegate here): pipeline-output loaders, viewer frame assembly with box
interpolation, DEM ray-casting and the map-point → image-pixel projection.

QGIS-specific pieces are injected by the caller:

* ``log_fn(message, level)`` replaces ``QgsMessageLog`` (level is
  ``"info"`` or ``"warning"``),
* ``sample_elevation(xy, origin, dem_json_path)`` replaces the
  QgsRasterLayer-based elevation sampling fallback,
* ``mesh_cache`` is a caller-owned dict so repeated clicks on the same DEM
  reuse the loaded mesh/BVH.
"""

import os
import json
from typing import Callable, Dict, List, Optional, Tuple


def _noop_log(message: str, level: str = "info") -> None:
    pass


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_georef_track_dets(target_folder: str, track_id: int,
                           camera_suffix: str = "t") -> List[dict]:
    """Return frame/confidence/class_id entries for *track_id* from the
    geo-referenced track CSVs (``tracks_t/*.csv`` or ``tracks_w/*.csv``,
    excluding ``_pixel.csv``).

    Format: ``frame,track_id,x1,y1,z1,x2,y2,z2,confidence,class_id[,interpolated]``
    """
    tracks_folder = os.path.join(target_folder, f"tracks_{camera_suffix}")
    if not os.path.isdir(tracks_folder):
        return []

    result: List[dict] = []
    try:
        for fname in os.listdir(tracks_folder):
            if not fname.endswith(".csv") or fname.endswith("_pixel.csv"):
                continue
            csv_path = os.path.join(tracks_folder, fname)
            with open(csv_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    if len(parts) >= 10 and int(parts[1]) == track_id:
                        result.append({
                            "frame": int(parts[0]),
                            "confidence": float(parts[8]),
                            "class_id": int(parts[9]),
                            "interpolated": int(parts[10]) if len(parts) > 10 else 0,
                        })
            if result:
                break  # found in this file, no need to search further
    except Exception:  # nosec B110
        pass

    return sorted(result, key=lambda d: d["frame"])


def load_pixel_detections(det_file: str) -> List[dict]:
    """Parse a ``detections_{t,w}/detections.txt`` file.

    Format (space-separated, comment header with #):
    ``frame x1 y1 x2 y2 confidence class_id``
    """
    result: List[dict] = []
    if not os.path.isfile(det_file):
        return result
    try:
        with open(det_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    result.append({
                        "frame": int(parts[0]),
                        "x1": float(parts[1]),
                        "y1": float(parts[2]),
                        "x2": float(parts[3]),
                        "y2": float(parts[4]),
                        "confidence": float(parts[5]),
                        "class_id": int(parts[6]) if len(parts) > 6 else 0,
                    })
    except Exception:  # nosec B110
        pass
    return result


def load_pixel_tracks(tracks_file: str) -> Dict[int, List[dict]]:
    """Parse ``tracks/tracks_pixel.csv``.

    Format (comma-separated, comment header with #):
    ``frame,track_id,x1,y1,x2,y2,conf,cls[,interpolated]``
    """
    result: Dict[int, List[dict]] = {}
    if not os.path.isfile(tracks_file):
        return result
    try:
        with open(tracks_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 8:
                    tid = int(parts[1])
                    entry = {
                        "frame": int(parts[0]),
                        "x1": float(parts[2]),
                        "y1": float(parts[3]),
                        "x2": float(parts[4]),
                        "y2": float(parts[5]),
                        "conf": float(parts[6]),
                        "cls": int(parts[7]),
                    }
                    result.setdefault(tid, []).append(entry)
    except Exception:  # nosec B110
        pass
    return result


def resolve_image_paths(target_folder: str, frame_idx: int) -> tuple:
    """Return ``(path_t, path_w)`` for frame *frame_idx*.

    Each element is the filesystem path to the thermal / RGB frame image,
    or an empty string when that frame type has not been extracted.
    """
    paths = []
    for poses_name, frames_dir in [
        ("poses_t.json", "frames_t"),
        ("poses_w.json", "frames_w"),
    ]:
        poses_path = os.path.join(target_folder, poses_name)
        found = ""
        if os.path.isfile(poses_path):
            try:
                with open(poses_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                images = data.get("images", [])
                if 0 <= frame_idx < len(images):
                    imagefile = images[frame_idx].get("imagefile", "")
                    candidate = os.path.join(target_folder, frames_dir, imagefile)
                    if imagefile and os.path.isfile(candidate):
                        found = candidate
            except Exception:  # nosec B110
                pass
        paths.append(found)
    return paths[0], paths[1]


def find_dem_mesh_path(dem_path: str,
                       dem_json_path: Optional[str]) -> Optional[str]:
    """Return the path to the DEM mesh file (.glb / .gltf), or None."""
    if dem_path and dem_path.lower().endswith((".glb", ".gltf")):
        if os.path.isfile(dem_path):
            return dem_path
    if dem_json_path:
        base = os.path.splitext(dem_json_path)[0]
        for ext in (".glb", ".gltf"):
            candidate = base + ext
            if os.path.isfile(candidate):
                return candidate
    return None


# ---------------------------------------------------------------------------
# Viewer frame assembly
# ---------------------------------------------------------------------------

def build_frames_from_pixel_tracks(
    track_dets, all_tracks, track_id, target_folder, boxes_modality: str
) -> List[dict]:
    """Build viewer frame list from pixel-space track data.

    The ``interpolated`` flag from tracks_pixel.csv is forwarded as the
    7th element of each green box tuple so the viewer can draw dashed lines.
    """
    frames = []
    for det in track_dets:
        fi = det["frame"]
        is_interp = det.get("interpolated", 0)
        other_on_frame = [
            (d["x1"], d["y1"], d["x2"], d["y2"], d["conf"], d["cls"])
            for tid, dets in all_tracks.items()
            if tid != track_id
            for d in dets
            if d["frame"] == fi
        ]
        path_t, path_w = resolve_image_paths(target_folder, fi)
        frames.append({
            "frame_idx": fi,
            "image_path_t": path_t,
            "image_path_w": path_w,
            "boxes_modality": boxes_modality,
            "boxes_green": [
                (det["x1"], det["y1"], det["x2"], det["y2"],
                 det["conf"], det["cls"], is_interp)
            ],
            "boxes_blue": other_on_frame,
        })
    return frames


def build_frames_from_georef(
    georef_dets, all_pixel_dets, target_folder, boxes_modality: str
) -> List[dict]:
    """Build viewer frame list by matching geo-referenced track detections
    back to pixel detections via (frame, confidence, class_id).

    Frames where no pixel match is found (interpolated frames whose bbox
    was never written to detections.txt) get their bounding box
    interpolated linearly from neighbouring actual detections and are
    flagged with is_interpolated=1 so the viewer draws a dashed border.
    """
    frames = []
    for gd in georef_dets:
        fi = gd["frame"]
        conf = gd["confidence"]
        cls = gd["class_id"]
        is_interp = gd.get("interpolated", 0)

        same_frame = [d for d in all_pixel_dets if d["frame"] == fi]

        # Match by confidence ± tolerance and class
        matched = [
            d for d in same_frame
            if d["class_id"] == cls and abs(d["confidence"] - conf) < 0.0015
        ]
        others = [d for d in same_frame if d not in matched]

        path_t, path_w = resolve_image_paths(target_folder, fi)
        frames.append({
            "frame_idx": fi,
            "image_path_t": path_t,
            "image_path_w": path_w,
            "boxes_modality": boxes_modality,
            "boxes_green": [
                (d["x1"], d["y1"], d["x2"], d["y2"],
                 d["confidence"], d["class_id"], is_interp)
                for d in matched
            ],
            "boxes_blue": [
                (d["x1"], d["y1"], d["x2"], d["y2"],
                 d["confidence"], d["class_id"])
                for d in others
            ],
        })

    # For frames where the pixel match failed (interpolated frames that
    # were never written to detections.txt), interpolate from neighbours.
    fill_interpolated_boxes(frames)
    return frames


def fill_interpolated_boxes(frames: List[dict]) -> None:
    """Fill empty ``boxes_green`` entries by linear interpolation.

    Operates in-place.  Interpolated boxes are flagged with
    ``is_interpolated=1`` so the viewer draws them with a dashed border.
    """
    n = len(frames)
    for i, frame in enumerate(frames):
        if frame["boxes_green"]:
            continue  # already has a real or previously-interpolated box

        # Find nearest frames before and after that have actual boxes
        j_before = next(
            (j for j in range(i - 1, -1, -1) if frames[j]["boxes_green"]),
            None,
        )
        j_after = next(
            (j for j in range(i + 1, n) if frames[j]["boxes_green"]),
            None,
        )

        if j_before is None and j_after is None:
            continue  # no reference boxes anywhere — leave empty

        if j_before is not None and j_after is not None:
            b1 = frames[j_before]["boxes_green"][0]
            b2 = frames[j_after]["boxes_green"][0]
            fi1, fi2, fi = (
                frames[j_before]["frame_idx"],
                frames[j_after]["frame_idx"],
                frame["frame_idx"],
            )
            alpha = (fi - fi1) / (fi2 - fi1) if fi2 != fi1 else 0.5
        else:
            # Only one side available — copy the nearest box unchanged
            src = frames[j_before if j_before is not None else j_after]
            b1 = b2 = src["boxes_green"][0]
            alpha = 0.0

        x1 = b1[0] + alpha * (b2[0] - b1[0])
        y1 = b1[1] + alpha * (b2[1] - b1[1])
        x2 = b1[2] + alpha * (b2[2] - b1[2])
        y2 = b1[3] + alpha * (b2[3] - b1[3])
        conf = float(b1[4]) if len(b1) > 4 else 0.0
        cls = int(b1[5]) if len(b1) > 5 else 0

        frame["boxes_green"] = [(x1, y1, x2, y2, conf, cls, 1)]


# ---------------------------------------------------------------------------
# DEM ray-casting and map-point projection
# ---------------------------------------------------------------------------

def ray_cast_dem_z(
    local_xy: Tuple[float, float],
    mesh_path: Optional[str],
    mesh_cache: Dict[str, object],
) -> Optional[float]:
    """Return local z of the DEM mesh at *(local_x, local_y)* via a
    downward vertical ray-cast.

    The mesh and its BVH are cached in *mesh_cache* by path so subsequent
    calls for the same DEM are cheap.  Returns ``None`` on failure or if no
    mesh is available.
    """
    if not mesh_path:
        return None
    try:
        import numpy as np
        from alfspy.render.render import read_gltf
        from trimesh import Trimesh

        if mesh_path not in mesh_cache:
            mesh_data, _ = read_gltf(mesh_path)
            tri_mesh = Trimesh(
                vertices=mesh_data.vertices,
                faces=mesh_data.indices,
            )
            try:
                _ = tri_mesh.triangles_tree   # pre-build BVH
            except Exception:  # nosec B110
                pass
            mesh_cache[mesh_path] = tri_mesh

        tri_mesh = mesh_cache[mesh_path]

        ray_origins = np.array([[local_xy[0], local_xy[1], 10_000.0]])
        ray_directions = np.array([[0.0, 0.0, -1.0]])
        locations, _, _ = tri_mesh.ray.intersects_location(
            ray_origins, ray_directions
        )
        if len(locations) > 0:
            return float(np.max(locations[:, 2]))
    except Exception:  # nosec B110
        pass
    return None


def project_map_point(
    xy: Tuple[float, float],
    frame_idx: int,
    target_folder: str,
    dem_path: str,
    correction_path: str,
    modality: str,
    mesh_cache: Dict[str, object],
    sample_elevation: Optional[Callable] = None,
    log_fn: Optional[Callable[[str, str], None]] = None,
) -> Optional[Tuple[float, float]]:
    """Project a geographic map coordinate into image pixel space.

    Replicates the exact camera construction and projection maths of
    :class:`~.bambi_box_projector.BoxProjectionWorker` so the result is
    pixel-identical to transferring a geo-referenced bounding box between
    modalities — with a single point instead of eight box corners.

    *sample_elevation(xy, origin, dem_json_path)* is an optional fallback
    used when the mesh ray-cast fails (the click tool injects a
    QgsRasterLayer-based sampler).

    Returns ``(pixel_x, pixel_y)`` or ``None`` on failure / out-of-frame.
    """
    log = log_fn or _noop_log
    import numpy as np
    from .camera_pose import build_camera, world_to_pixel
    from .corrections import correction_for_frame, read_correction

    try:
        from pyrr import Vector3, Quaternion  # noqa: F401 — availability probe
        from alfspy.core.rendering import Camera  # noqa: F401
    except ImportError:
        return None

    try:
        # ---- Poses ---------------------------------------------------
        poses_path = os.path.join(target_folder, f"poses_{modality}.json")
        if not os.path.isfile(poses_path):
            log(f"[FoV click | {modality}]  poses file not found: {poses_path}",
                "warning")
            return None
        with open(poses_path, "r", encoding="utf-8") as fh:
            poses_data = json.load(fh)
        images = poses_data.get("images", [])
        if frame_idx >= len(images):
            log(f"[FoV click | {modality}]  frame_idx {frame_idx} >= "
                f"len(images) {len(images)}", "warning")
            return None

        # ---- DEM origin (identical to BoxProjectionWorker) -----------
        origin = (0.0, 0.0, 0.0)

        def _try_load_origin(json_path):
            if not json_path or not os.path.isfile(json_path):
                return None
            try:
                with open(json_path, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                o = d.get("origin")
                if o and len(o) >= 3:
                    return (float(o[0]), float(o[1]), float(o[2]))
            except Exception:  # nosec B110
                pass
            return None

        dem_json_found = None
        if dem_path:
            dem_json_path = (
                dem_path
                .replace(".gltf", ".json")
                .replace(".glb", ".json")
            )
            o = _try_load_origin(dem_json_path)
            if o:
                origin = o
                dem_json_found = dem_json_path

        if origin == (0.0, 0.0, 0.0):
            for search_dir in [target_folder, os.path.dirname(target_folder)]:
                if not search_dir or not os.path.isdir(search_dir):
                    continue
                for fname in os.listdir(search_dir):
                    if not fname.lower().endswith(".json"):
                        continue
                    json_path = os.path.join(search_dir, fname)
                    o = _try_load_origin(json_path)
                    if o and (o[0] != 0.0 or o[1] != 0.0):
                        origin = o
                        dem_json_found = json_path
                        break
                if origin != (0.0, 0.0, 0.0):
                    break

        # ---- Terrain elevation at click position ---------------------
        local_xy = (xy[0] - origin[0], xy[1] - origin[1])
        mesh_path = find_dem_mesh_path(dem_path, dem_json_found)
        local_z = ray_cast_dem_z(local_xy, mesh_path, mesh_cache)
        elev_src = "mesh"
        if local_z is None and sample_elevation is not None:
            local_z = sample_elevation(xy, origin, dem_json_found)
            elev_src = "raster" if local_z is not None else "NONE"
        if local_z is None:
            local_z = 0.0
            elev_src = "fallback(0)"

        # ---- Correction (identical to BoxProjectionWorker) -----------
        corr = read_correction(target_folder, correction_path)
        t_corr, r_corr = correction_for_frame(frame_idx, corr)

        # ---- Camera (single source: core.camera_pose, 1× correction) -
        meta = images[frame_idx]
        fovy = meta.get("fovy", [50])
        if isinstance(fovy, list):
            fovy = fovy[0]
        rot_vals = meta["rotation"]

        camera = build_camera(meta, t_corr, r_corr, aspect_ratio=1.0)

        # ---- Image dimensions (same approach as BoxProjectionWorker) -
        # Read from the first frame in the poses file, not from the
        # specific image_path — matches BoxProjectionWorker exactly.
        frames_dir = os.path.join(target_folder, f"frames_{modality}")
        img_width, img_height = 640, 512  # fallback
        if images:
            first_file = images[0].get("imagefile", "")
            candidate = os.path.join(frames_dir, first_file)
            if first_file and os.path.isfile(candidate):
                try:
                    import cv2
                    img = cv2.imread(candidate)
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                except Exception:  # nosec B110
                    pass

        # ---- Project (single point, same as _project_georef_box_to_pixels)
        ox, oy, oz = origin
        point = np.array(
            [[xy[0] - ox, xy[1] - oy, local_z]],
            dtype=np.float64,
        )

        pxs, pys = world_to_pixel(point, img_width, img_height, camera)
        px, py = float(pxs[0]), float(pys[0])

        in_bounds = 0 <= px <= img_width and 0 <= py <= img_height

        # ---- Debug: find nearest georef entry for comparison ---------
        georef_info = ""
        try:
            georef_path = os.path.join(target_folder, f"georeferenced_{modality}", "georeferenced.txt")
            if os.path.isfile(georef_path):
                with open(georef_path, "r", encoding="utf-8") as gf:
                    best_dist = float("inf")
                    best_line = ""
                    for line in gf:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) >= 10 and int(parts[1]) == frame_idx:
                            gx = (float(parts[2]) + float(parts[5])) / 2
                            gy = (float(parts[3]) + float(parts[6])) / 2
                            d = (gx - xy[0])**2 + (gy - xy[1])**2
                            if d < best_dist:
                                best_dist = d
                                best_line = line
                    if best_line:
                        georef_info = f"\n  Nearest georef : {best_line}"
        except Exception:  # nosec B110
            pass

        log(
            f"[FoV click | {modality}]  frame={frame_idx}\n"
            f"  Click map      : x={xy[0]:.3f}  y={xy[1]:.3f}\n"
            f"  Origin         : x={origin[0]:.3f}  y={origin[1]:.3f}  z={origin[2]:.3f}\n"
            f"  Local point    : x={point[0, 0]:.3f}  y={point[0, 1]:.3f}  z={local_z:.3f} ({elev_src})\n"
            f"  Cam location   : {list(meta['location'])}\n"
            f"  Cam rotation   : {list(rot_vals)}\n"
            f"  fovy={fovy}  img={img_width}x{img_height}\n"
            f"  Correction t   : {dict(t_corr)}  r: {dict(r_corr)}\n"
            f"  Projected pixel: px={px:.1f}  py={py:.1f}  in_bounds={in_bounds}"
            f"{georef_info}",
            "info",
        )

        if in_bounds:
            return (px, py)
        return None

    except Exception as exc:
        import traceback
        log(
            f"[FoV click | {modality}]  Exception: {exc}\n"
            f"{traceback.format_exc()}",
            "warning",
        )
        return None
