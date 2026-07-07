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

# Correction handling and world→pixel projection moved to core (shared with
# the click tool, labelling tool and correction wizard); re-exported under
# their old names for compatibility.
from .core.camera_pose import build_camera
from .core.camera_pose import world_to_pixel as _world_to_pixel  # noqa: F401
from .core.corrections import (  # noqa: F401 — re-exported API
    correction_for_frame as _correction_for_frame,
    read_correction as _read_correction,
)


def _load_georef(target_folder: str, src_modality: str) -> List[dict]:
    """Parse ``georeferenced_{src_modality}/georeferenced.txt``.

    The geo-referenced detections live in the camera-specific folder matching
    the modality the boxes were detected in (``_t`` thermal, ``_w`` RGB).

    Format: ``idx frame x1 y1 z1 x2 y2 z2 confidence class_id``
    Coordinates have the DEM origin offset already added.
    """
    path = os.path.join(
        target_folder, f"georeferenced_{src_modality}", "georeferenced.txt")
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
                        "frame": int(parts[1]),
                        "x1": float(parts[2]),
                        "y1": float(parts[3]),
                        "z1": float(parts[4]),
                        "x2": float(parts[5]),
                        "y2": float(parts[6]),
                        "z2": float(parts[7]),
                        "confidence": float(parts[8]),
                        "class_id": int(parts[9]),
                    })
    except Exception:  # nosec B110
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
        cls = int(box[5])
        for g in georef_on_frame:
            if g["class_id"] == cls and abs(g["confidence"] - conf) < tol:
                matched.append(g)
                break
    return matched


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
    valid_mask = (pxs >= -0.5) & (pxs <= img_width + 0.5) & (pys >= -0.5) & (pys <= img_height + 0.5)

    if not np.any(valid_mask):
        return None

    return (
        float(max(0.0, pxs[valid_mask].min())),
        float(max(0.0, pys[valid_mask].min())),
        float(min(img_width - 1.0, pxs[valid_mask].max())),
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
        Root output folder (contains ``georeferenced_{t,w}/``, ``poses_*.json``).
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
    finished = pyqtSignal(dict)  # {frame_idx: {"green": [...], "blue": [...]}}
    error = pyqtSignal(str)

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
        self._target_folder = target_folder
        self._dem_path = dem_path
        self._correction_path = correction_path
        self._src_modality = src_modality  # "t" or "w"
        self._frames = frames

    def run(self):
        try:
            result = self._project()
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))

    # ------------------------------------------------------------------

    def _project(self) -> dict:
        try:
            from alfspy.core.rendering import Camera  # noqa: F401 — probe
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
            except Exception:  # nosec B110
                pass
            return None

        # Primary: derive JSON path from the stored DEM path
        if self._dem_path:
            dem_json_path = (
                self._dem_path
                .replace(".gltf", ".json")
                .replace(".glb", ".json")
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
        # Boxes were detected in the source modality, so the geo-referenced
        # file lives in that modality's camera-specific folder.
        georef_all = _load_georef(self._target_folder, self._src_modality)
        if not georef_all:
            src_label = "RGB" if self._src_modality == "w" else "thermal"
            raise RuntimeError(
                f"No geo-referenced detections found for the {src_label} modality "
                f"(expected georeferenced_{self._src_modality}/georeferenced.txt).\n"
                "Please run the 'Geo-Reference Detections' step first."
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
            candidate = os.path.join(frames_dir, first_file)
            if first_file and os.path.isfile(candidate):
                try:
                    import cv2
                    img = cv2.imread(candidate)
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                except Exception:  # nosec B110
                    pass

        self.progress.emit(45)

        # ---- Project each viewer frame -----------------------------------
        results: dict = {}
        n = len(self._frames)

        for i, frame_data in enumerate(self._frames):
            frame_idx = frame_data.get("frame_idx")
            boxes_green = frame_data.get("boxes_green", [])
            boxes_blue = frame_data.get("boxes_blue", [])

            if frame_idx is None or frame_idx >= len(images):
                results[i] = {"green": [], "blue": []}
                continue

            # Camera pose for this frame in the destination modality, with
            # the same correction as used for geo-referencing (1× rule).
            t_corr, r_corr = _correction_for_frame(frame_idx, corr)
            camera = build_camera(
                images[frame_idx], t_corr, r_corr, aspect_ratio=1.0)

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
                "blue": _project_list(boxes_blue),
            }

            progress = 45 + int(((i + 1) / max(1, n)) * 50)
            self.progress.emit(min(progress, 95))

        self.progress.emit(100)
        return results
