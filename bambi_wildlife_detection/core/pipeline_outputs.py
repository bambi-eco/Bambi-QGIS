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


def _store_source(path: str, kind: str):
    """``(target_folder, modality)`` for a legacy output path, if it has a store.

    The layer builders are handed a file path like
    ``<target>/tracks_t/tracks.csv``; the store equivalent lives at
    ``<target>/bambi_t/tracks.gpkg``. Deriving one from the other keeps the
    change to the builders down to "prefer the store", without threading a new
    argument through every call site.

    Returns ``None`` when there is no store file to read.
    """
    import os

    from . import store

    folder = os.path.basename(os.path.dirname(path))
    target_folder = os.path.dirname(os.path.dirname(path))
    if "_" not in folder:
        return None
    modality = folder.rsplit("_", 1)[1]
    if modality not in store.MODALITIES:
        return None
    if not os.path.isfile(store.stage_path(target_folder, kind, modality)):
        return None
    return target_folder, modality


def load_geo_tracks_by_id(csv_path: str, log_fn: LogFn = None) -> Dict[int, list]:
    """Load geo-referenced tracks, keyed by track id.

    Prefers the 6.0 store — where membership is a join rather than a
    coordinate match — and falls back to ``tracks.csv`` for projects that have
    no store yet. The returned shape is identical either way, so the QGIS layer
    builders are unaffected.
    """
    source = _store_source(csv_path, "tracks")
    if source is not None:
        rows = _geo_tracks_from_store(*source)
        if rows:
            return rows

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


def _geo_tracks_from_store(target_folder: str, modality: str) -> Dict[int, list]:
    """Geo-referenced track points of the active run, keyed by track id."""
    import os

    from . import gpkg, store, track_store

    run = track_store.active_run(target_folder, modality)
    if run is None:
        return {}

    det_path = store.stage_path(target_folder, store.DETECTIONS, modality)
    geo_path = store.stage_path(target_folder, store.GEOREFERENCED, modality)
    trk_path = store.stage_path(target_folder, store.TRACKS, modality)
    if not all(os.path.isfile(p) for p in (det_path, geo_path, trk_path)):
        return {}

    conn = store.open_store(det_path, store.DETECTIONS, modality)
    try:
        gpkg.attach(conn, geo_path, "geo")
        gpkg.attach(conn, trk_path, "trk")
        rows = conn.execute(
            "SELECT m.track_id, d.frame, d.confidence, d.species_id, "
            "g.gx1, g.gy1, g.gz1, g.gx2, g.gy2, g.gz2, m.interpolated "
            "FROM trk.track_members m "
            "JOIN trk.tracks t ON t.track_id = m.track_id "
            "JOIN detections d ON d.detection_id = m.detection_id "
            "JOIN geo.detections_geo g ON g.detection_id = d.detection_id "
            "WHERE t.run_id = ? ORDER BY m.track_id, d.frame",
            (run["run_id"],)).fetchall()
        gpkg.detach(conn, "geo")
        gpkg.detach(conn, "trk")
    finally:
        conn.close()

    tracks: Dict[int, list] = {}
    for row in rows:
        tracks.setdefault(row["track_id"], []).append({
            "frame": row["frame"],
            "x1": row["gx1"], "y1": row["gy1"], "z1": row["gz1"],
            "x2": row["gx2"], "y2": row["gy2"], "z2": row["gz2"],
            "confidence": row["confidence"] if row["confidence"] is not None else 1.0,
            "class_id": row["species_id"],
            "interpolated": row["interpolated"],
        })
    return tracks


def load_fov_polygons_3d(fov_file: str, log_fn: LogFn = None) -> Dict[int, list]:
    """Load FoV polygons from file.

    :param fov_file: Path to FoV polygons file
    :return: Dictionary mapping frame index to list of (x, y, z) points
    """
    source = _store_source(fov_file, "fov")
    if source is not None:
        stored = _fov_from_store(*source)
        if stored:
            return stored

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

    source = _store_source(georef_file, "georeferenced")
    if source is not None:
        rows = _georef_from_store(*source)
        if rows:
            return rows

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


def _georef_from_store(target_folder: str, modality: str) -> Dict[int, list]:
    """Geo-referenced detections from the store, grouped by frame.

    Unlike the text reader this does not have to skip rows with a negative
    corner: a detection that failed to project has no ``detections_geo`` row at
    all, and is accounted for in ``georef_failures`` instead (§3.2).
    """
    from . import track_store

    rows = track_store.load_georeferenced(target_folder, modality)
    grouped: Dict[int, list] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row["frame"], []).append({
            "idx": index,
            "detection_id": row["detection_id"],
            "frame": row["frame"],
            "x1": row["gx1"], "y1": row["gy1"], "z1": row["gz1"],
            "x2": row["gx2"], "y2": row["gy2"], "z2": row["gz2"],
            "confidence": row["confidence"] if row["confidence"] is not None else 1.0,
            "class_id": row["species_id"],
        })
    return grouped


def _fov_from_store(target_folder: str, modality: str) -> Dict[int, list]:
    """FoV polygons from the store, keyed by frame."""
    import os

    from . import store

    path = store.stage_path(target_folder, store.FOV, modality)
    if not os.path.isfile(path):
        return {}

    conn = store.open_store(path, store.FOV, modality)
    try:
        polygons: Dict[int, list] = {}
        for row in conn.execute(
                "SELECT frame, seq, x, y, z FROM fov_vertices "
                "ORDER BY frame, seq"):
            polygons.setdefault(row["frame"], []).append(
                (row["x"], row["y"], row["z"]))
        return polygons
    finally:
        conn.close()
