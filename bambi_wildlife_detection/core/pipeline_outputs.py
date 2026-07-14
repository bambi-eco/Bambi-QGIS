# -*- coding: utf-8 -*-
"""Readers for pipeline output files used by the QGIS layer builders.

Moved from ``bambi_dock_widget.py`` (whose methods delegate here). Further
format-specific readers live in :mod:`core.inspection`,
:mod:`core.video_export` and :mod:`core.labelling`; they return different
shapes for their specific consumers and are deliberately not unified.

File formats
------------
``tracks_{m}/tracks.csv``
    ``frame,track_id,x1,y1,z1,x2,y2,z2,confidence,class_id[,interpolated]``
``fov_{m}/fov_polygons.txt``
    ``frame_idx num_points x1 y1 z1 x2 y2 z2 …``
``georeferenced_{m}/georeferenced.txt``
    ``idx frame x1 y1 z1 x2 y2 z2 confidence class_id``
"""

import json
import os
from typing import Callable, Dict, Optional, Tuple

LogFn = Optional[Callable[[str], None]]


def read_dem_origin_xy(dem_path: str = "",
                       dem_metadata_path: str = "") -> Tuple[float, float]:
    """The DEM's ``origin`` (x, y) — the shift from mesh-local to world CRS.

    Poses store camera positions mesh-locally (world CRS minus this origin),
    while every geo-referenced product is in the world CRS, so anything that
    mixes the two has to add it back. Falls back to ``(0.0, 0.0)`` when no
    metadata is found, which is correct for a DEM built at the world origin.

    The metadata sits next to the mesh as a ``.json`` of the same name unless
    *dem_metadata_path* names it explicitly.
    """
    candidates = []
    if dem_metadata_path:
        candidates.append(dem_metadata_path)
    if dem_path:
        candidates.append(
            dem_path.replace(".gltf", ".json").replace(".glb", ".json"))

    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                origin = json.load(fh).get("origin", [0, 0, 0])
            return float(origin[0]), float(origin[1])
        except (ValueError, TypeError, IndexError, OSError):
            continue
    return 0.0, 0.0


def load_geo_tracks_by_id(csv_path: str, log_fn: LogFn = None) -> Dict[int, list]:
    """Load geo-referenced tracks from a CSV file, keyed by track id."""
    tracks = {}

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split(',')
                if len(parts) >= 10:
                    try:
                        frame = int(parts[0])
                        track_id = int(parts[1])
                        x1 = float(parts[2])
                        y1 = float(parts[3])
                        z1 = float(parts[4])
                        x2 = float(parts[5])
                        y2 = float(parts[6])
                        z2 = float(parts[7])
                        conf = float(parts[8])
                        cls = int(parts[9])
                        interpolated = int(parts[10]) if len(parts) > 10 else 0

                        if track_id not in tracks:
                            tracks[track_id] = []

                        tracks[track_id].append({
                            'frame': frame,
                            'x1': x1, 'y1': y1, 'z1': z1,
                            'x2': x2, 'y2': y2, 'z2': z2,
                            'confidence': conf,
                            'class_id': cls,
                            'interpolated': interpolated
                        })
                    except (ValueError, IndexError):
                        continue

    except Exception as e:
        if log_fn:
            log_fn(f"Error reading {csv_path}: {str(e)}")

    return tracks


def load_fov_polygons_3d(fov_file: str, log_fn: LogFn = None) -> Dict[int, list]:
    """Load FoV polygons from file.

    :param fov_file: Path to FoV polygons file
    :return: Dictionary mapping frame index to list of (x, y, z) points
    """
    polygons = {}

    try:
        with open(fov_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                frame_idx = int(parts[0])
                num_points = int(parts[1])

                if num_points == 0:
                    continue

                # Parse points (x y z triplets)
                points = []
                for i in range(num_points):
                    idx = 2 + i * 3
                    if idx + 2 < len(parts):
                        x = float(parts[idx])
                        y = float(parts[idx + 1])
                        z = float(parts[idx + 2])
                        points.append((x, y, z))

                if points:
                    polygons[frame_idx] = points

    except Exception as e:
        if log_fn:
            log_fn(f"Error reading FoV file: {str(e)}")

    return polygons


def load_georef_detections_by_frame(georef_file: str,
                                    log_fn: LogFn = None) -> Dict[int, list]:
    """Load geo-referenced detections grouped by frame.

    :param georef_file: Path to georeferenced detections file
    :return: Dictionary mapping frame index to list of detections
    """
    from collections import defaultdict

    frame_detections = defaultdict(list)

    try:
        with open(georef_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) >= 10:
                    try:
                        idx = int(parts[0])
                        frame = int(parts[1])
                        x1 = float(parts[2])
                        y1 = float(parts[3])
                        z1 = float(parts[4])
                        x2 = float(parts[5])
                        y2 = float(parts[6])
                        z2 = float(parts[7])
                        conf = float(parts[8])
                        cls = int(parts[9])

                        # Skip invalid detections
                        if x1 < 0 or y1 < 0:
                            continue

                        frame_detections[frame].append({
                            'idx': idx,
                            'frame': frame,
                            'x1': x1, 'y1': y1, 'z1': z1,
                            'x2': x2, 'y2': y2, 'z2': z2,
                            'confidence': conf,
                            'class_id': cls
                        })
                    except (ValueError, IndexError):
                        continue

    except Exception as e:
        if log_fn:
            log_fn(f"Error reading georeferenced file: {str(e)}")

    return dict(frame_detections)
