# -*- coding: utf-8 -*-
"""Readers for pipeline output files used by the QGIS layer builders.

Moved from ``bambi_dock_widget.py`` (whose methods delegate here). Further
format-specific readers live in :mod:`core.inspection`,
:mod:`core.video_export` and :mod:`core.labelling`; they return different
shapes for their specific consumers and are deliberately not unified.

Everything here reads the 6.0 store. The functions still take a legacy file
path, but only to locate the project — ``<target>/tracks_t/tracks.csv`` says
where ``<target>/bambi_t/tracks.gpkg`` is. The text files themselves are no
longer parsed by anything except :mod:`core.migration`, which exists to import
them once.
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
    """Geo-referenced tracks from the store, keyed by track id.

    Takes the legacy *csv_path* only to locate the project: the store lives at
    ``<target>/bambi_{m}/tracks.gpkg`` beside it, and the file itself is never
    read. A project with no store returns nothing rather than being
    reconstructed from text, because a reconstruction is a guess about which
    detection a track point came from and the store knows (§1.2a).
    """
    source = _store_source(csv_path, "tracks")
    if source is None:
        if log_fn:
            log_fn(_no_store_message(csv_path, "tracks"))
        return {}
    return _geo_tracks_from_store(*source)


def _no_store_message(path: str, kind: str) -> str:
    """Why nothing was loaded, and what to do about it."""
    return (f"No {kind} store for {os.path.dirname(path)}. The 5.x text files "
            "are no longer read — use 'Migrate 5.x…' on the Input tab, or "
            "re-run the step.")


def _geo_tracks_from_store(target_folder: str, modality: str) -> Dict[int, list]:
    """Geo-referenced track points of the active run, keyed by track id."""
    import os

    from . import gpkg, store, track_store

    run_ids = track_store.analysis_runs(target_folder, modality)
    if not run_ids:
        return {}

    det_path = store.stage_path(target_folder, store.DETECTIONS, modality)
    geo_path = store.stage_path(target_folder, store.GEOREFERENCED, modality)
    trk_path = store.stage_path(target_folder, store.TRACKS, modality)
    if not all(os.path.isfile(p) for p in (det_path, geo_path, trk_path)):
        return {}

    superseded = track_store.superseded_track_ids(target_folder, modality)
    placeholders = ", ".join("?" for _ in run_ids)

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
            f"WHERE t.run_id IN ({placeholders}) "  # nosec B608 — ints only
            "ORDER BY m.track_id, d.frame", run_ids).fetchall()
        gpkg.detach(conn, "geo")
        gpkg.detach(conn, "trk")
    finally:
        conn.close()

    tracks: Dict[int, list] = {}
    for row in rows:
        if row["track_id"] in superseded:
            continue
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
    """Field-of-view polygons from the store, keyed by frame.

    *fov_file* locates the project; the file itself is not read (see
    :func:`load_geo_tracks_by_id`).
    """
    source = _store_source(fov_file, "fov")
    if source is None:
        if log_fn:
            log_fn(_no_store_message(fov_file, "field of view"))
        return {}
    return _fov_from_store(*source)


def load_georef_detections_by_frame(georef_file: str,
                                    log_fn: LogFn = None) -> Dict[int, list]:
    """Geo-referenced detections from the store, grouped by frame.

    *georef_file* locates the project; the file itself is not read (see
    :func:`load_geo_tracks_by_id`).
    """
    source = _store_source(georef_file, "georeferenced")
    if source is None:
        if log_fn:
            log_fn(_no_store_message(georef_file, "geo-referencing"))
        return {}
    return _georef_from_store(*source)


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
