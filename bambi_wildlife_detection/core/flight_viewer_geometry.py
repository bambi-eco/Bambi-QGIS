# -*- coding: utf-8 -*-
"""Line / point geometry and overlay images of the 3D Flight Viewer.

Plain NumPy, so it is testable without a GL context. Colours are RGBA floats
in 0..1. Segment arrays are ``(N, 2, 3)``, their colours ``(N, 2, 4)``.
"""

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

ROUTE_PAST = (1.0, 0.62, 0.15, 1.0)
ROUTE_FUTURE = (0.75, 0.78, 0.82, 0.55)
POSE_COLOR = (0.55, 0.75, 1.0, 0.85)
CURRENT_COLOR = (1.0, 0.2, 0.2, 1.0)
FRUSTUM_COLOR = (1.0, 0.85, 0.1, 1.0)
FOOTPRINT_COLOR = (1.0, 0.85, 0.1, 0.9)

#: Colours cycled through track ids (RGB, 0..255) - readable on imagery.
TRACK_PALETTE = (
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
)
DETECTION_COLOR = (255, 230, 0)


def track_color(track_id: int) -> Tuple[int, int, int]:
    return TRACK_PALETTE[int(track_id) % len(TRACK_PALETTE)]


def _empty():
    return np.zeros((0, 2, 3)), np.zeros((0, 2, 4))


def polyline_segments(points, color) -> Tuple[np.ndarray, np.ndarray]:
    """Consecutive points as segments, skipping non-finite points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return _empty()
    seg = np.stack([pts[:-1], pts[1:]], axis=1)
    ok = np.all(np.isfinite(seg.reshape(len(seg), -1)), axis=1)
    seg = seg[ok]
    col = np.broadcast_to(np.asarray(color, dtype=np.float64),
                          (len(seg), 2, 4)).copy()
    return seg, col


def route_segments(positions, current: int, past=ROUTE_PAST,
                   future=ROUTE_FUTURE):
    """The flight route, flown part in *past*, rest in *future* colour."""
    pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    current = min(max(int(current), 0), max(len(pts) - 1, 0))
    a, ca = polyline_segments(pts[:current + 1], past)
    b, cb = polyline_segments(pts[current:], future)
    return np.concatenate([a, b]), np.concatenate([ca, cb])


def pose_markers(positions, current: int, max_markers: int = 300,
                 color=POSE_COLOR):
    """Every k-th camera position (k so at most *max_markers*)."""
    pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if not len(pts):
        return np.zeros((0, 3)), np.zeros((0, 4))
    step = max(1, int(np.ceil(len(pts) / max_markers)))
    idx = [i for i in range(0, len(pts), step) if i != current]
    out = pts[idx]
    return out, np.broadcast_to(np.asarray(color, float), (len(out), 4)).copy()


def frustum_segments(apex, corners, outline=None, color=FRUSTUM_COLOR,
                     outline_color=FOOTPRINT_COLOR):
    """Apex-to-corner edges, the corner rectangle and the ground outline."""
    apex = np.asarray(apex, dtype=np.float64).reshape(3)
    c = np.asarray(corners, dtype=np.float64).reshape(4, 3)
    segs = [np.stack([apex, c[i]]) for i in range(4)]
    segs += [np.stack([c[i], c[(i + 1) % 4]]) for i in range(4)]
    seg = np.array(segs)
    col = np.broadcast_to(np.asarray(color, float), (len(seg), 2, 4)).copy()
    if outline is not None and len(outline) > 1:
        ring = np.vstack([outline, outline[:1]])
        s2, c2 = polyline_segments(ring, outline_color)
        seg, col = np.concatenate([seg, s2]), np.concatenate([col, c2])
    return seg, col


def lift(points, metres: float = 0.3):
    """Raise ground points a little so lines are not buried in the DEM."""
    pts = np.array(points, dtype=np.float64, copy=True).reshape(-1, 3)
    pts[:, 2] += metres
    return pts


def track_segments(ground_tracks: Dict[int, np.ndarray],
                   alpha: Dict[int, float] = None):
    """History polylines of tracks: ``track_id -> (N, 3)`` ground points."""
    segs, cols = [_empty()[0]], [_empty()[1]]
    for track_id, pts in ground_tracks.items():
        rgb = np.asarray(track_color(track_id), float) / 255.0
        a = (alpha or {}).get(track_id, 1.0)
        s, c = polyline_segments(lift(pts), (*rgb, a))
        segs.append(s)
        cols.append(c)
    return np.concatenate(segs), np.concatenate(cols)


# ---------------------------------------------------------------------------
# Overlay image with boxes (draped like the frame)
# ---------------------------------------------------------------------------

def _rect(img, x1, y1, x2, y2, rgba, thickness):
    h, w = img.shape[:2]
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w - 1), min(y2, h - 1)
    if x2 < x1 or y2 < y1:
        return
    t = max(1, int(thickness))
    img[y1:min(y1 + t, y2 + 1), x1:x2 + 1] = rgba
    img[max(y2 - t + 1, y1):y2 + 1, x1:x2 + 1] = rgba
    img[y1:y2 + 1, x1:min(x1 + t, x2 + 1)] = rgba
    img[y1:y2 + 1, max(x2 - t + 1, x1):x2 + 1] = rgba


def box_overlay(size: Tuple[int, int], boxes: Iterable[dict],
                scale: float = 1.0, thickness: int = 3,
                color_of=None, fill_alpha: int = 40) -> np.ndarray:
    """RGBA image of box outlines, transparent elsewhere.

    :param size: ``(width, height)`` of the overlay (the frame texture size)
    :param boxes: dicts with ``x1, y1, x2, y2`` in original frame pixels and
        optionally ``track_id``
    :param scale: overlay size / original frame size
    :param color_of: ``box -> (r, g, b)``; default: track colour, or yellow
    """
    w, h = int(size[0]), int(size[1])
    img = np.zeros((h, w, 4), dtype=np.uint8)
    for b in boxes:
        if color_of is not None:
            rgb = color_of(b)
        elif b.get("track_id") is not None:
            rgb = track_color(b["track_id"])
        else:
            rgb = DETECTION_COLOR
        x1, y1, x2, y2 = (float(b[k]) * scale for k in ("x1", "y1", "x2", "y2"))
        if fill_alpha:
            xa, xb = sorted((int(round(x1)), int(round(x2))))
            ya, yb = sorted((int(round(y1)), int(round(y2))))
            region = img[max(ya, 0):max(yb, 0), max(xa, 0):max(xb, 0)]
            region[region[..., 3] == 0] = (*rgb, fill_alpha)
        _rect(img, x1, y1, x2, y2, (*rgb, 255), thickness)
    return img


def scaled_size(width: int, height: int, max_edge: int) -> Tuple[int, int]:
    """Size after shrinking so the longer edge is at most *max_edge*."""
    longest = max(width, height)
    if longest <= max_edge or longest <= 0:
        return int(width), int(height)
    f = max_edge / float(longest)
    return max(1, int(round(width * f))), max(1, int(round(height * f)))


def ortho_matrix_for_bounds(lower: Sequence[float], upper: Sequence[float],
                            z_low: float = -1.0e4, z_high: float = 1.0e4):
    """World -> clip matrix that drapes an image over an XY extent.

    Image row 0 is the north (max y) edge, as for a GeoTIFF.
    """
    from .flight_viewer import orthographic
    view = np.eye(4)
    return orthographic(lower[0], upper[0], lower[1], upper[1],
                        -z_high, -z_low) @ view


# ---------------------------------------------------------------------------
# Annotations on the ground (3D lines)
# ---------------------------------------------------------------------------

def box_edge_pixels(boxes, samples_per_edge: int = 8) -> np.ndarray:
    """Pixel points around each box, clockwise from the top-left corner.

    :return: ``(N, 4 * samples_per_edge, 2)``
    """
    out = []
    t = np.arange(samples_per_edge) / float(samples_per_edge)
    for b in boxes:
        x1, y1, x2, y2 = (float(b[k]) for k in ("x1", "y1", "x2", "y2"))
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        ring = []
        for i in range(4):
            a, c = corners[i], corners[(i + 1) % 4]
            ring.append(a + (c - a) * t[:, None])
        out.append(np.vstack(ring))
    return np.array(out).reshape(-1, 4 * samples_per_edge, 2)


def ring_segments(ring, color) -> Tuple[np.ndarray, np.ndarray]:
    """A closed outline as solid segments (non-finite points leave gaps)."""
    pts = np.asarray(ring, dtype=np.float64).reshape(-1, 3)
    return polyline_segments(np.vstack([pts, pts[:1]]), color)


def dashed_segments(ring, color, dash: float, gap: Optional[float] = None,
                    closed: bool = True):
    """A polyline as dashes of *dash* metres with *gap* metres between them
    (measured along the line)."""
    pts = np.asarray(ring, dtype=np.float64).reshape(-1, 3)
    if closed and len(pts):
        pts = np.vstack([pts, pts[:1]])
    gap = dash if gap is None else gap
    period = dash + gap
    segs = []
    for a, b in zip(pts[:-1], pts[1:]):
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        segs.append((a, b))
    if not segs or period <= 0:
        return _empty()
    out = []
    s0 = 0.0
    for a, b in segs:
        length = float(np.linalg.norm(b - a))
        if length <= 0:
            continue
        s = 0.0
        while s < length:
            phase = (s0 + s) % period
            if phase < dash:                        # inside a dash
                run = min(dash - phase, length - s)
                p = a + (b - a) * (s / length)
                q = a + (b - a) * ((s + run) / length)
                out.append((p, q))
            else:                                   # inside a gap
                run = min(period - phase, length - s)
            s += run
        s0 += length
    if not out:
        return _empty()
    seg = np.array(out)
    return seg, np.broadcast_to(np.asarray(color, float), (len(seg), 2, 4)).copy()


def nearest_screen_point(points, mvp, width: int, height: int, x: float,
                         y: float, max_px: float):
    """Index of the point drawn nearest to pixel ``(x, y)`` (y down) within
    *max_px*, or None. Points behind the viewer are ignored."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(pts):
        return None
    clip = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(mvp).T
    front = clip[:, 3] > 1e-9
    with np.errstate(invalid="ignore", divide="ignore"):
        sx = (clip[:, 0] / clip[:, 3] + 1.0) * width / 2.0
        sy = (1.0 - clip[:, 1] / clip[:, 3]) * height / 2.0
    d = np.hypot(sx - x, sy - y)
    d[~front | ~np.isfinite(d)] = np.inf
    i = int(np.argmin(d))
    return i if d[i] <= max_px else None
