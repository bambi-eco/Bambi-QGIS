# -*- coding: utf-8 -*-
"""
BAMBI Box Projector
===================

Re-projects geo-referenced bounding boxes (world space, UTM) into the pixel
space of a *different* camera modality (thermal ↔ RGB).

The forward geo-referencing step (pixel → world) requires DEM ray-casting.
The *reverse* step implemented here (world → pixel) is pure camera-projection
math: it only needs the DEM JSON file (to read the mesh origin offset) plus
the target modality's poses file.  No DEM mesh is loaded.

Public API
----------
``BoxProjectionWorker(QThread)``
    Run box projection in a background thread.  Signals:

    ``progress(int)``   — 0‒100
    ``finished(dict)``  — ``{frame_idx: {"green": [boxes], "blue": [boxes]}}``
    ``error(str)``      — human-readable message on failure

Box tuple format: ``(x1, y1, x2, y2, confidence, class_id)``
World coords in the georef file have the DEM origin offset already applied;
this worker subtracts it before calling ``world_to_pixel_coord``.
"""

import os
import json
from typing import Dict, List, Optional, Tuple

from qgis.PyQt.QtCore import QThread, pyqtSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_correction(target_folder: str, correction_path: str) -> dict:
    """Load correction.json, return dict with translation/rotation/additional."""
    path = ""
    if correction_path and os.path.isfile(correction_path):
        path = correction_path
    else:
        fallback = os.path.join(target_folder, "correction.json")
        if os.path.isfile(fallback):
            path = fallback

    if path:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def _correction_for_frame(frame_idx: int, corr: dict) -> Tuple[dict, dict]:
    """Return (translation_dict, rotation_dict) for *frame_idx*."""
    default_t = corr.get("translation", {"x": 0.0, "y": 0.0, "z": 0.0})
    default_r = corr.get("rotation",    {"x": 0.0, "y": 0.0, "z": 0.0})
    for entry in corr.get("additional", []):
        s = entry.get("start", 0)
        e = entry.get("end", float("inf"))
        if s <= frame_idx <= e:
            return (
                entry.get("translation", default_t),
                entry.get("rotation",    default_r),
            )
    return default_t, default_r


def _load_georef(target_folder: str) -> List[dict]:
    """Parse georeferenced/georeferenced.txt.

    Format: ``idx frame x1 y1 z1 x2 y2 z2 confidence class_id``
    Coordinates have the DEM origin offset already added.
    """
    path = os.path.join(target_folder, "georeferenced", "georeferenced.txt")
    result: List[dict] = []
    if not os.path.isfile(path):
        return result
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 10:
                    result.append({
                        "frame":      int(parts[1]),
                        "x1":         float(parts[2]),
                        "y1":         float(parts[3]),
                        "z1":         float(parts[4]),
                        "x2":         float(parts[5]),
                        "y2":         float(parts[6]),
                        "z2":         float(parts[7]),
                        "confidence": float(parts[8]),
                        "class_id":   int(parts[9]),
                    })
    except Exception:
        pass
    return result


def _match_boxes_to_georef(
    viewer_boxes: List[tuple],
    georef_on_frame: List[dict],
    tol: float = 0.0015,
) -> List[dict]:
    """Return the georef entries that match the given viewer box tuples.

    Matching is by (class_id, confidence ± tol).
    """
    matched: List[dict] = []
    for box in viewer_boxes:
        if len(box) < 6:
            continue
        conf = float(box[4])
        cls  = int(box[5])
        for g in georef_on_frame:
            if g["class_id"] == cls and abs(g["confidence"] - conf) < tol:
                matched.append(g)
                break
    return matched


def _world_to_pixel(
    corners: "np.ndarray",
    img_width: int,
    img_height: int,
    camera,
) -> "Tuple[np.ndarray, np.ndarray]":
    """Project world-space coordinates to pixel space.

    Reimplements the math of ``alfspy.core.convert.world_to_pixel_coord``
    directly to avoid a broadcasting bug in that function when more than one
    point is passed at a time (``ndc[:, 3]`` produces shape ``(N,)`` which
    cannot broadcast against ``ndc`` of shape ``(N, 4)``).

    Returns ``(pixel_xs, pixel_ys)`` arrays of the same length as *corners*.
    """
    import numpy as np

    coords = np.reshape(corners, (-1, 3)).astype(np.float64)
    n = len(coords)

    # Homogeneous coordinates (N, 4)
    homo = np.ones((n, 4), dtype=np.float64)
    homo[:, :3] = coords

    # View and projection (use float64 throughout)
    view = np.array(camera.get_view(), dtype=np.float64)
    proj = np.array(camera.get_proj(), dtype=np.float64)

    cam_coords = homo @ view          # (N, 4)
    ndc        = cam_coords @ proj    # (N, 4)

    # Perspective divide — use column slice [:,3:4] so shape stays (N,1)
    w          = ndc[:, 3:4]
    ndc_norm   = ndc / w             # (N, 4)

    pixel_xs = (ndc_norm[:, 0] + 1.0) * img_width  / 2.0
    pixel_ys = img_height - (ndc_norm[:, 1] + 1.0) * img_height / 2.0

    return pixel_xs, pixel_ys


def _project_georef_box_to_pixels(
    georef: dict,
    origin: Tuple[float, float, float],
    camera,
    img_width: int,
    img_height: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Project a world-space georef box onto *camera* and return pixel bbox.

    The eight corners of the 3-D axis-aligned bounding box are projected; the
    bounding rectangle of all in-frustum projected corners is returned as
    ``(x1, y1, x2, y2)`` in pixel space, or ``None`` if all corners are
    outside the image.
    """
    import numpy as np

    ox, oy, oz = origin
    lx1 = georef["x1"] - ox
    ly1 = georef["y1"] - oy
    lz1 = georef["z1"] - oz
    lx2 = georef["x2"] - ox
    ly2 = georef["y2"] - oy
    lz2 = georef["z2"] - oz

    corners = np.array([
        (lx1, ly1, lz1), (lx2, ly1, lz1), (lx2, ly2, lz1), (lx1, ly2, lz1),
        (lx1, ly1, lz2), (lx2, ly1, lz2), (lx2, ly2, lz2), (lx1, ly2, lz2),
    ], dtype=np.float64)

    try:
        pxs, pys = _world_to_pixel(corners, img_width, img_height, camera)
    except Exception:
        return None

    # Keep only corners that project within the image
    valid_mask = (
        (pxs >= -0.5) & (pxs <= img_width  + 0.5) &
        (pys >= -0.5) & (pys <= img_height + 0.5)
    )
    if not np.any(valid_mask):
        return None

    return (
        float(max(0.0,              pxs[valid_mask].min())),
        float(max(0.0,              pys[valid_mask].min())),
        float(min(img_width  - 1.0, pxs[valid_mask].max())),
        float(min(img_height - 1.0, pys[valid_mask].max())),
    )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class BoxProjectionWorker(QThread):
    """Background worker that re-projects geo-referenced boxes to the other
    camera modality's pixel space.

    Parameters
    ----------
    target_folder : str
        Root output folder (contains ``georeferenced/``, ``poses_*.json``).
    dem_path : str
        Path to the DEM GLTF/GLB file (the matching ``.json`` is derived from
        it to read the mesh origin offset).
    correction_path : str
        Explicit path to ``correction.json``; falls back to
        ``target_folder/correction.json``.
    src_modality : str
        ``"t"`` or ``"w"`` — the modality in which the existing boxes live.
    frames : list of dict
        Viewer frame dicts (keys: ``frame_idx``, ``boxes_green``,
        ``boxes_blue``).
    """

    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)   # {frame_idx: {"green": [...], "blue": [...]}}
    error    = pyqtSignal(str)

    def __init__(
        self,
        target_folder: str,
        dem_path: str,
        correction_path: str,
        src_modality: str,
        frames: List[dict],
        parent=None,
    ):
        super().__init__(parent)
        self._target_folder   = target_folder
        self._dem_path        = dem_path
        self._correction_path = correction_path
        self._src_modality    = src_modality   # "t" or "w"
        self._frames          = frames

    def run(self):
        try:
            result = self._project()
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))

    # ------------------------------------------------------------------

    def _project(self) -> dict:
        import numpy as np
        from pyrr import Vector3, Quaternion

        try:
            from alfspy.core.rendering import Camera
        except ImportError as exc:
            raise RuntimeError(
                "alfspy is not available — cannot project bounding boxes.\n"
                f"({exc})"
            )

        self.progress.emit(5)

        # ---- DEM origin offset -------------------------------------------
        origin = (0.0, 0.0, 0.0)

        def _try_load_origin(json_path: str):
            if not json_path or not os.path.isfile(json_path):
                return None
            try:
                with open(json_path, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                o = d.get("origin")
                if o and len(o) >= 3:
                    return (float(o[0]), float(o[1]), float(o[2]))
            except Exception:
                pass
            return None

        # Primary: derive JSON path from the stored DEM path
        if self._dem_path:
            dem_json_path = (
                self._dem_path
                .replace(".gltf", ".json")
                .replace(".glb",  ".json")
            )
            o = _try_load_origin(dem_json_path)
            if o:
                origin = o

        # Fallback: scan target_folder and its parent for *.json with "origin"
        if origin == (0.0, 0.0, 0.0):
            search_dirs = [
                self._target_folder,
                os.path.dirname(self._target_folder),
            ]
            for search_dir in search_dirs:
                if not search_dir or not os.path.isdir(search_dir):
                    continue
                for fname in os.listdir(search_dir):
                    if not fname.lower().endswith(".json"):
                        continue
                    o = _try_load_origin(os.path.join(search_dir, fname))
                    if o and (o[0] != 0.0 or o[1] != 0.0):
                        origin = o
                        break
                if origin != (0.0, 0.0, 0.0):
                    break

        self.progress.emit(15)

        # ---- Correction --------------------------------------------------
        corr = _read_correction(self._target_folder, self._correction_path)

        # ---- Geo-referenced detections -----------------------------------
        georef_all = _load_georef(self._target_folder)
        if not georef_all:
            raise RuntimeError(
                "No geo-referenced detections found.\n"
                "Please run the geo-referencing step first."
            )

        # Index by frame for fast lookup
        georef_by_frame: Dict[int, List[dict]] = {}
        for g in georef_all:
            georef_by_frame.setdefault(g["frame"], []).append(g)

        self.progress.emit(25)

        # ---- Target (other) modality poses -------------------------------
        dst_modality = "w" if self._src_modality == "t" else "t"
        poses_path = os.path.join(
            self._target_folder, f"poses_{dst_modality}.json"
        )
        if not os.path.isfile(poses_path):
            raise RuntimeError(
                f"Poses file not found for the {'RGB' if dst_modality == 'w' else 'thermal'} "
                f"modality: {poses_path}"
            )
        with open(poses_path, "r", encoding="utf-8") as fh:
            poses_data = json.load(fh)
        images = poses_data.get("images", [])

        self.progress.emit(35)

        # ---- Image resolution of target modality -------------------------
        frames_dir = os.path.join(
            self._target_folder, f"frames_{dst_modality}"
        )
        img_width, img_height = 640, 512  # fallback
        if images:
            first_file = images[0].get("imagefile", "")
            candidate  = os.path.join(frames_dir, first_file)
            if first_file and os.path.isfile(candidate):
                try:
                    import cv2
                    img = cv2.imread(candidate)
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                except Exception:
                    pass

        self.progress.emit(45)

        # ---- Project each viewer frame -----------------------------------
        results: dict = {}
        n = len(self._frames)

        for i, frame_data in enumerate(self._frames):
            frame_idx   = frame_data.get("frame_idx")
            boxes_green = frame_data.get("boxes_green", [])
            boxes_blue  = frame_data.get("boxes_blue",  [])

            if frame_idx is None or frame_idx >= len(images):
                results[i] = {"green": [], "blue": []}
                continue

            # Camera pose for this frame in the destination modality
            meta = images[frame_idx]
            fovy = meta.get("fovy", [50])
            if isinstance(fovy, list):
                fovy = fovy[0]

            position = Vector3(meta["location"])
            rot_vals = meta["rotation"]

            # Apply correction (same correction as used for geo-referencing)
            t_corr, r_corr = _correction_for_frame(frame_idx, corr)
            cor_t = Vector3([t_corr.get("x", 0), t_corr.get("y", 0), t_corr.get("z", 0)], dtype="f4")
            cor_r = Vector3([r_corr.get("x", 0), r_corr.get("y", 0), r_corr.get("z", 0)], dtype="f4")

            rotation_eulers = (
                Vector3([np.deg2rad(v % 360.0) for v in rot_vals]) - cor_r
            ) * -1
            position = position + cor_t
            rotation_quat = Quaternion.from_eulers(rotation_eulers)

            camera = Camera(
                fovy=fovy,
                aspect_ratio=1.0,
                position=position,
                rotation=rotation_quat,
            )

            # Match viewer boxes to geo-referenced entries
            on_frame = georef_by_frame.get(frame_idx, [])

            def _project_list(viewer_boxes):
                matched = _match_boxes_to_georef(viewer_boxes, on_frame)
                proj_boxes = []
                for g in matched:
                    bbox = _project_georef_box_to_pixels(
                        g, origin, camera, img_width, img_height
                    )
                    if bbox is not None:
                        proj_boxes.append((
                            bbox[0], bbox[1], bbox[2], bbox[3],
                            g["confidence"], g["class_id"],
                        ))
                return proj_boxes

            results[i] = {
                "green": _project_list(boxes_green),
                "blue":  _project_list(boxes_blue),
            }

            progress = 45 + int(((i + 1) / max(1, n)) * 50)
            self.progress.emit(min(progress, 95))

        self.progress.emit(100)
        return results
