# -*- coding: utf-8 -*-
"""Topographic wetness index from the flight's own DEM.

Where water collects is largely decided by the shape of the ground: a cell
that drains a large area and is nearly flat stays wet, a ridge or a steep
slope does not. The topographic wetness index (TWI) puts that in one number,

    TWI = ln(a / tan(beta))

with *a* the upslope area draining through a cell per unit contour width and
*beta* its slope. It is static, needs no download and - unlike a satellite
index - works under closed forest, which makes it the complement to NDMI
(:mod:`core.env_sentinel2`).

The DEM the pipeline already uses is a mesh, so it is first sampled onto the
flight's grid (:mod:`core.environment`), with a margin around it: a cell's
wetness depends on the slope above it, and without the margin every cell
near the edge would look drier than it is.

Steps: sample the mesh, fill the depressions so every cell drains somewhere
(priority flood), route the flow to all lower neighbours in proportion to
the slope towards them (multiple flow direction, which avoids the stripes a
single direction leaves), accumulate, and combine with the local slope.
"""

import heapq
import math
from typing import Callable, Optional, Tuple

from .environment import EnvironmentFailure, Grid, check_cancelled

#: Extra ground around the flight's grid used for the flow routing, metres.
MARGIN_M = 500.0

#: Smallest slope (as a tangent) used in the index, so flat cells stay finite.
MIN_SLOPE = 0.001

#: A DEM with less relief than this over the whole extent is a flat surface
#: (the aquatic-survey mesh), on which a wetness index means nothing.
MIN_RELIEF_M = 0.5

#: (row offset, column offset, distance in cells) of the eight neighbours.
NEIGHBOURS = (
    (-1, -1, math.sqrt(2.0)), (-1, 0, 1.0), (-1, 1, math.sqrt(2.0)),
    (0, -1, 1.0), (0, 1, 1.0),
    (1, -1, math.sqrt(2.0)), (1, 0, 1.0), (1, 1, math.sqrt(2.0)),
)


# --------------------------------------------------------------------------- #
# The DEM as a grid
# --------------------------------------------------------------------------- #

def load_mesh_vertices(mesh_path: str):
    """``(N, 3)`` vertex array of a GLB/GLTF DEM mesh, in mesh-local metres."""
    import numpy as np

    try:
        from alfspy.render.render import read_gltf
        mesh_data, _ = read_gltf(mesh_path)
        return np.asarray(mesh_data.vertices, dtype=np.float64).reshape(-1, 3)
    except ImportError:
        import trimesh
        mesh = trimesh.load(mesh_path, force="mesh")
        return np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)


def grid_from_vertices(vertices, origin, grid: Grid):
    """Heights of *grid*'s cell centres, linearly interpolated between the
    mesh *vertices* (mesh-local, shifted by *origin*); NaN outside the mesh."""
    import numpy as np
    from scipy.interpolate import LinearNDInterpolator

    vertices = np.asarray(vertices, dtype=np.float64)
    x = vertices[:, 0] + origin[0]
    y = vertices[:, 1] + origin[1]
    z = vertices[:, 2] + origin[2]
    min_x, min_y, max_x, max_y = grid.bounds
    # Keep a ring of vertices outside the grid so its edge cells are still
    # interpolated rather than extrapolated.
    pad = 20.0 * grid.resolution
    in_x = (x >= min_x - pad) & (x <= max_x + pad)
    in_y = (y >= min_y - pad) & (y <= max_y + pad)
    keep = in_x & in_y & np.isfinite(z)
    if np.count_nonzero(keep) < 3:
        return np.full(grid.shape, np.nan, dtype=np.float64)
    points = np.column_stack([x[keep], y[keep]])
    points, unique = np.unique(points, axis=0, return_index=True)
    if len(points) < 3:
        return np.full(grid.shape, np.nan, dtype=np.float64)
    interpolator = LinearNDInterpolator(points, z[keep][unique])
    xs, ys = grid.cell_centres()
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.asarray(interpolator(grid_x, grid_y), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Hydrology
# --------------------------------------------------------------------------- #

def fill_depressions(dem, epsilon: float = 1e-3,
                     cancelled: Optional[Callable[[], bool]] = None):
    """*dem* with every depression raised until it drains to the edge.

    Priority flood: starting from the border, cells are visited lowest
    first and none is left lower than the cell it was reached from, plus a
    tiny *epsilon* so that a filled flat still has a direction to drain in.
    NaN cells are outside the DEM and act as border.
    """
    import numpy as np

    dem = np.asarray(dem, dtype=np.float64)
    height, width = dem.shape
    valid = np.isfinite(dem)
    filled = dem.copy()
    closed = ~valid

    heap = []
    for row in range(height):
        for col in range(width):
            if not valid[row, col]:
                continue
            on_edge = row in (0, height - 1) or col in (0, width - 1)
            if not on_edge:
                on_edge = not all((
                    valid[row - 1, col], valid[row + 1, col],
                    valid[row, col - 1], valid[row, col + 1]))
            if on_edge:
                heap.append((filled[row, col], row, col))
                closed[row, col] = True
    heapq.heapify(heap)

    visited = 0
    while heap:
        level, row, col = heapq.heappop(heap)
        visited += 1
        if visited % 50000 == 0:
            check_cancelled(cancelled)
        for d_row, d_col, _distance in NEIGHBOURS:
            n_row, n_col = row + d_row, col + d_col
            if not (0 <= n_row < height and 0 <= n_col < width):
                continue
            if closed[n_row, n_col]:
                continue
            closed[n_row, n_col] = True
            raised = max(filled[n_row, n_col], level + epsilon)
            filled[n_row, n_col] = raised
            heapq.heappush(heap, (raised, n_row, n_col))
    return filled


def flow_accumulation(filled, exponent: float = 1.1,
                      cancelled: Optional[Callable[[], bool]] = None):
    """Number of cells draining through each cell, itself included.

    Multiple flow direction: a cell passes its water to every lower
    neighbour, weighted by the slope towards it raised to *exponent*.
    """
    import numpy as np

    filled = np.asarray(filled, dtype=np.float64)
    height, width = filled.shape
    valid = np.isfinite(filled)
    padded = np.full((height + 2, width + 2), np.inf)
    padded[1:-1, 1:-1] = np.where(valid, filled, np.inf)
    padded_width = width + 2

    centre = padded[1:-1, 1:-1]
    weights = np.zeros((height, width, 8), dtype=np.float64)
    offsets = np.zeros(8, dtype=np.int64)
    for index, (d_row, d_col, distance) in enumerate(NEIGHBOURS):
        neighbour = padded[1 + d_row:height + 1 + d_row,
                           1 + d_col:width + 1 + d_col]
        with np.errstate(invalid="ignore"):
            drop = (centre - neighbour) / distance
        drop = np.where(np.isfinite(drop) & (drop > 0), drop, 0.0)
        weights[:, :, index] = drop ** exponent
        offsets[index] = d_row * padded_width + d_col
    totals = weights.sum(axis=2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(totals > 0, weights / totals, 0.0)

    accumulation = np.zeros(padded.shape, dtype=np.float64)
    accumulation[1:-1, 1:-1] = valid.astype(np.float64)
    flat = accumulation.ravel()
    weights = weights.reshape(-1, 8)

    rows, cols = np.nonzero(valid)
    order = np.argsort(-filled[rows, cols], kind="stable")
    padded_index = (rows + 1) * padded_width + (cols + 1)
    weight_index = rows * width + cols
    for step, position in enumerate(order):
        if step % 50000 == 0:
            check_cancelled(cancelled)
        share = weights[weight_index[position]]
        if not share.any():
            continue
        source = padded_index[position]
        flat[source + offsets] += flat[source] * share
    result = accumulation[1:-1, 1:-1]
    return np.where(valid, result, np.nan)


def slope_tangent(dem, resolution: float):
    """Local slope of *dem* as a tangent (rise over run)."""
    import numpy as np

    dem = np.asarray(dem, dtype=np.float64)
    # Gradients need neighbours; a NaN beside a cell would blank it, so the
    # outside is filled with the nearest inside value along each axis first.
    filled = dem.copy()
    if np.isnan(filled).any():
        mean = np.nanmean(filled) if np.isfinite(filled).any() else 0.0
        filled = np.where(np.isfinite(filled), filled, mean)
    d_row, d_col = np.gradient(filled, resolution)
    slope = np.hypot(d_row, d_col)
    return np.where(np.isfinite(dem), slope, np.nan)


def wetness_index(dem, resolution: float,
                  cancelled: Optional[Callable[[], bool]] = None):
    """Topographic wetness index of a DEM grid; NaN where the DEM is."""
    import numpy as np

    dem = np.asarray(dem, dtype=np.float64)
    filled = fill_depressions(dem, cancelled=cancelled)
    accumulation = flow_accumulation(filled, cancelled=cancelled)
    slope = np.maximum(slope_tangent(dem, resolution), MIN_SLOPE)
    with np.errstate(divide="ignore", invalid="ignore"):
        index = np.log(accumulation * resolution / slope)
    return np.where(np.isfinite(dem), index, np.nan).astype(np.float32)


# --------------------------------------------------------------------------- #
# The source
# --------------------------------------------------------------------------- #

def compute(grid: Grid, mesh_path: str, origin,
            load_vertices: Callable = load_mesh_vertices,
            log: Optional[Callable[[str], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None,
            progress: Optional[Callable[[int], None]] = None,
            ) -> Tuple[dict, dict]:
    """``({"twi": index}, metadata)`` for *grid* from the DEM mesh."""
    import numpy as np

    if not mesh_path:
        raise EnvironmentFailure(
            "No DEM mesh is set for this flight. Select one on the Input "
            "tab.")
    if origin is None:
        raise EnvironmentFailure(
            "The DEM's metadata JSON names no origin, so the mesh cannot be "
            "placed in the project CRS.")
    margin = int(round(MARGIN_M / grid.resolution))
    work = grid.expanded(margin)
    if log:
        log(f"Sampling the DEM onto a {work.width} x {work.height} grid "
            f"({grid.resolution:g} m cells, {MARGIN_M:g} m margin)")
    vertices = load_vertices(mesh_path)
    check_cancelled(cancelled)
    dem = grid_from_vertices(vertices, origin, work)
    if progress:
        progress(30)

    core = dem[margin:margin + grid.height, margin:margin + grid.width]
    covered = float(np.count_nonzero(np.isfinite(core))) / core.size
    if covered < 0.5:
        raise EnvironmentFailure(
            f"The DEM covers only {covered * 100:.0f}% of the flight's "
            "extent. Use a DEM that covers the whole flight.")
    relief = float(np.nanmax(dem) - np.nanmin(dem))
    if relief < MIN_RELIEF_M:
        raise EnvironmentFailure(
            "The DEM is a flat surface, so there is no terrain to derive a "
            "wetness index from.")

    check_cancelled(cancelled)
    if log:
        log("Routing the flow and computing the wetness index")
    index = wetness_index(dem, work.resolution, cancelled=cancelled)
    if progress:
        progress(95)
    index = index[margin:margin + grid.height, margin:margin + grid.width]
    valid = index[np.isfinite(index)]
    metadata = {
        "dem": mesh_path,
        "margin_m": MARGIN_M,
        "dem_coverage": round(covered, 4),
        "relief_m": round(relief, 2),
        "method": "priority-flood fill, multiple flow direction, "
                  "ln(a / tan(slope))",
        "min": round(float(valid.min()), 3) if valid.size else None,
        "max": round(float(valid.max()), 3) if valid.size else None,
        "mean": round(float(valid.mean()), 3) if valid.size else None,
    }
    return {"twi": index}, metadata
