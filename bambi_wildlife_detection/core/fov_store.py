# -*- coding: utf-8 -*-
"""Field-of-view footprints in the store (EXCHANGE_FORMAT_PLAN.md §11).

The FoV step long wrote only ``fov_{m}/fov_polygons.txt``, so ``fov.gpkg``
existed solely where a 5.x project had been migrated. Everything that reads a
footprint — the coverage map, the transect areas a population estimate divides
by, the QGIS layers — now reads the store, so the step has to write it.

One polygon per frame, its vertices ordered: ``seq`` is what makes the ring
reconstructable, since a set of points has no edges.
"""

import os
from typing import Dict, Iterable, Sequence, Tuple

from . import store

Point = Tuple[float, float, float]


def record_fov(target_folder: str, modality: str,
               polygons: Dict[int, Sequence[Point]], log_fn=None) -> int:
    """Replace this modality's footprints with *polygons*.

    Keyed by frame, so re-running the step for a subset of frames updates those
    and leaves the rest — the same shape the text file had, where a frame's
    line was rewritten wholesale.
    """
    conn = store.open_store(
        store.stage_path(target_folder, store.FOV, modality),
        store.FOV, modality)
    try:
        with store.transaction(conn):
            for frame, points in polygons.items():
                points = [p for p in points if p is not None]
                conn.execute("DELETE FROM fov_vertices WHERE frame = ?",
                             (int(frame),))
                if not points:
                    conn.execute("DELETE FROM fov_polygons WHERE frame = ?",
                                 (int(frame),))
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO fov_polygons (frame, n_points) "
                    "VALUES (?, ?)", (int(frame), len(points)))
                conn.executemany(
                    "INSERT INTO fov_vertices (frame, seq, x, y, z) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [(int(frame), seq, float(x), float(y), float(z))
                     for seq, (x, y, z) in enumerate(points)])
    finally:
        conn.close()

    if log_fn:
        log_fn(f"Store: recorded {len(polygons)} field-of-view polygon(s)")
    return len(polygons)


def load_fov(target_folder: str, modality: str) -> Dict[int, list]:
    """``frame -> [(x, y, z), …]`` in vertex order, or ``{}``."""
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


def frames(target_folder: str, modality: str) -> Iterable[int]:
    """Frames that have a footprint, in order."""
    return sorted(load_fov(target_folder, modality))
