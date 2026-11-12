# -*- coding: utf-8 -*-
"""3D reconstruction of a flight from its RGB frames, with the DEM as prior.

The poses are known (GPS/IMU plus correction factors, optionally refined
from the images), and the DEM is a bare-earth model - so what is
reconstructed is the height of whatever stands on the ground: vegetation,
buildings.

Sparse
    SIFT features between nearby keyframes, triangulated with the cameras'
    projection matrices and kept when they reproject within a few pixels
    and lie in the height band above the DEM (:func:`sparse_points`).

Dense (pairwise stereo)
    Every keyframe is matched against its next keyframes as a rectified
    stereo pair in image space; a pair's orientation comes from its images
    (or the refined poses) and its scale from the ground on the DEM. The
    pairs' depth maps vote for a height per ground cell and the median wins
    (:func:`stereo_surface`) - a 2.5D surface, the right shape for nadir
    drone data.

Pose refinement (optional)
    A small bundle adjustment of the keyframes from SIFT tracks, with the
    telemetry as soft prior, put back on the DEM (:func:`refine_poses`).

Texturing
    A high-resolution orthophoto of the surface from the RGB frames or -
    linked by capture time - the thermal frames with their own poses
    (:func:`texture_ortho`).

All coordinates are DEM-local metres; cameras are
:class:`core.flight_viewer.FrameCamera`.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .flight_viewer import FrameCamera, cast_rays, vertex_normals

ProgressFn = Optional[Callable[[str, float], None]]
CancelFn = Optional[Callable[[], bool]]


class Cancelled(Exception):
    pass


def _check(cancel: CancelFn):
    if cancel is not None and cancel():
        raise Cancelled()


# ---------------------------------------------------------------------------
# Keyframes and cameras
# ---------------------------------------------------------------------------

def select_keyframes(positions, indices: Sequence[int],
                     spacing: float) -> List[int]:
    """Frames at least *spacing* metres (horizontally) apart along the route."""
    indices = list(indices)
    if not indices:
        return []
    pos = np.asarray(positions, dtype=np.float64)
    keep = [indices[0]]
    for i in indices[1:]:
        if np.linalg.norm(pos[i, :2] - pos[keep[-1], :2]) >= spacing:
            keep.append(i)
    if keep[-1] != indices[-1] and len(indices) > 1 \
            and np.linalg.norm(pos[indices[-1], :2] - pos[keep[-1], :2]) >= spacing / 2:
        keep.append(indices[-1])
    return keep


def opencv_projection(camera: FrameCamera, width: int, height: int):
    """``(K, R, t)`` in OpenCV / COLMAP convention (x right, y down, z
    forward) for an image of ``width`` x ``height`` pixels."""
    tan = math.tan(math.radians(camera.fovy) / 2.0)
    f = (height / 2.0) / tan
    fx = (width / 2.0) / (tan * camera.aspect)   # = f when the aspects agree
    K = np.array([[fx, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
    flip = np.diag([1.0, -1.0, -1.0])
    R = flip @ camera.view[:3, :3]
    t = flip @ camera.view[:3, 3]
    return K, R, t


def project_points(camera: FrameCamera, width: int, height: int, points):
    """Pixel coordinates (x right, y down) of world points and a mask of
    those in front of the camera and inside the image."""
    K, R, t = opencv_projection(camera, width, height)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cam = pts @ R.T + t
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    ok = (z > 1e-6) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    return u, v, ok


def sample_image(image: np.ndarray, u, v) -> np.ndarray:
    """Bilinear samples of *image* at pixel coordinates ``u, v``."""
    import cv2
    shape = np.shape(u)
    flat_u = np.asarray(u, dtype=np.float32).ravel()
    flat_v = np.asarray(v, dtype=np.float32).ravel()
    n = flat_u.size
    # remap wants maps below 32767 pixels per side: fold into rows.
    cols = 4096
    rows = max(1, -(-n // cols))
    pad = rows * cols - n
    mu = np.pad(flat_u, (0, pad)).reshape(rows, cols)
    mv = np.pad(flat_v, (0, pad)).reshape(rows, cols)
    img = np.asarray(image, dtype=np.float32)
    channels = img.shape[2] if img.ndim == 3 else 1
    out = cv2.remap(img, mu, mv, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    out = out.reshape(rows * cols, -1)[:n]
    return out.reshape(shape + ((channels,) if channels > 1 else ()))


# ---------------------------------------------------------------------------
# Ground grid
# ---------------------------------------------------------------------------

@dataclass
class GroundGrid:
    """A north-up grid of cell centres over ``lower``-``upper`` (XY)."""
    lower: np.ndarray
    upper: np.ndarray
    gsd: float
    width: int = 0
    height: int = 0

    def __post_init__(self):
        self.lower = np.asarray(self.lower, dtype=np.float64)[:2]
        self.upper = np.asarray(self.upper, dtype=np.float64)[:2]
        size = self.upper - self.lower
        # An explicit size wins: recomputing it from an extent that is an
        # exact multiple of gsd can round up by one cell (float error).
        if self.width <= 0:
            self.width = max(2, int(math.ceil(size[0] / self.gsd - 1e-6)))
        if self.height <= 0:
            self.height = max(2, int(math.ceil(size[1] / self.gsd - 1e-6)))
        self.upper = self.lower + np.array([self.width, self.height]) * self.gsd

    @classmethod
    def fitting(cls, lower, upper, gsd: float, max_cells: int) -> "GroundGrid":
        """A grid over the extent, coarsened until it has at most *max_cells*."""
        size = np.asarray(upper, float)[:2] - np.asarray(lower, float)[:2]
        cells = (size[0] / gsd) * (size[1] / gsd)
        if cells > max_cells:
            gsd *= math.sqrt(cells / max_cells)
        return cls(lower, upper, gsd)

    def xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(X, Y)`` cell-centre coordinates, row 0 = north."""
        xs = self.lower[0] + (np.arange(self.width) + 0.5) * self.gsd
        ys = self.upper[1] - (np.arange(self.height) + 0.5) * self.gsd
        return np.meshgrid(xs, ys)


def dem_heights(grid: GroundGrid, caster, top: float = 1.0e4) -> np.ndarray:
    """DEM height under every cell (NaN where the DEM has no surface)."""
    X, Y = grid.xy()
    origins = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, top)])
    dirs = np.tile([0.0, 0.0, -1.0], (X.size, 1))
    hits = cast_rays(caster, origins, dirs)
    return hits[:, 2].reshape(X.shape)


# ---------------------------------------------------------------------------
# Views and results
# ---------------------------------------------------------------------------

@dataclass
class View:
    """A frame used by the reconstruction: camera plus (scaled) image."""
    index: int
    camera: FrameCamera
    image: np.ndarray               # (H, W, 3) float32 or uint8, RGB
    mask: Optional[np.ndarray] = None   # (H, W) bool, False = no image there

    @property
    def size(self) -> Tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]

    def valid(self, u, v, ok) -> np.ndarray:
        """*ok* restricted to pixels the undistortion mask keeps."""
        if self.mask is None:
            return ok
        ok = np.array(ok, copy=True)
        ui = np.clip(np.rint(np.where(ok, u, 0)), 0, self.mask.shape[1] - 1).astype(int)
        vi = np.clip(np.rint(np.where(ok, v, 0)), 0, self.mask.shape[0] - 1).astype(int)
        return ok & self.mask[vi, ui]


@dataclass
class DenseResult:
    offset: np.ndarray              # (H, W) height above the DEM, NaN = unknown
    confidence: np.ndarray          # (H, W) 0..1
    views: np.ndarray               # (H, W) number of stereo pairs voting
    color: np.ndarray               # (H, W, 3) unused, kept for old callers
    heights: np.ndarray             # the height bins
    extras: Dict[str, np.ndarray] = field(default_factory=dict)


def fill_holes(values: np.ndarray, valid: Optional[np.ndarray] = None,
               iterations: int = 200) -> np.ndarray:
    """Fill NaN cells from their known neighbours (repeated 3x3 averaging
    of the known cells), so gaps get the height around them instead of
    dropping to the ground. Cells outside *valid* stay NaN."""
    import cv2
    out = np.array(values, dtype=np.float32, copy=True)
    known = np.isfinite(out)
    if valid is None:
        valid = np.ones(out.shape, bool)
    todo = valid & ~known
    if not todo.any() or not known.any():
        return out
    val = np.where(known, out, 0.0).astype(np.float32)
    w = known.astype(np.float32)
    for _ in range(iterations):
        s = cv2.blur(val * w, (3, 3), borderType=cv2.BORDER_CONSTANT)
        n = cv2.blur(w, (3, 3), borderType=cv2.BORDER_CONSTANT)
        grow = todo & (n > 0) & (w == 0)
        if not grow.any():
            break
        val[grow] = s[grow] / n[grow]
        w[grow] = 1.0
        todo &= ~grow
        if not todo.any():
            break
    out = np.where(w > 0, val, np.nan)
    out[~valid] = np.nan
    return out


def fill_offsets(offset: np.ndarray, median_window: int = 3,
                 valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Surface heights for meshing: unknown cells take the heights around
    them (:func:`fill_holes`; the DEM, 0, only where nothing is known), then
    a median removes isolated spikes."""
    import cv2
    filled = fill_holes(offset, valid)
    filled = np.where(np.isfinite(filled), filled, 0.0).astype(np.float32)
    if median_window >= 3:
        filled = cv2.medianBlur(filled, 3 if median_window < 5 else 5)
    return filled


def covered_area(measured: np.ndarray, margin: int = 2) -> np.ndarray:
    """The area to mesh: measured cells, a small margin, and every gap
    enclosed by them (ground hidden beside a crown gets the height around
    it rather than a hole)."""
    import cv2
    m = cv2.dilate(np.asarray(measured, np.uint8),
                   np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    # Flood the outside from the border; what is not reached is inside.
    padded = np.pad(m, 1)
    outside = padded.copy()
    cv2.floodFill(outside, None, (0, 0), 2)
    return (outside != 2)[1:-1, 1:-1]


def upsample_heights(grid: "GroundGrid", z: np.ndarray,
                     fine: "GroundGrid") -> np.ndarray:
    """Bilinear heights of a coarse grid at the cell centres of a finer one
    (NaN where the coarse grid has none)."""
    import cv2
    X, Y = fine.xy()
    # Pixel coordinates of the fine cell centres in the coarse grid.
    u = (X - grid.lower[0]) / grid.gsd - 0.5
    v = (grid.upper[1] - Y) / grid.gsd - 0.5
    src = np.where(np.isfinite(z), z, np.nan).astype(np.float32)
    known = np.isfinite(src)
    filled = np.where(known, src, 0.0).astype(np.float32)
    val = cv2.remap(filled, u.astype(np.float32), v.astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wgt = cv2.remap(known.astype(np.float32), u.astype(np.float32),
                    v.astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(wgt > 0.5, val / np.maximum(wgt, 1e-6), np.nan)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Surface mesh and texturing
# ---------------------------------------------------------------------------

def heightfield_mesh(grid: GroundGrid, z: np.ndarray):
    """``(vertices, faces, normals)`` of a heightfield on the grid's cell
    centres; cells with a NaN height are left out."""
    X, Y = grid.xy()
    h, w = X.shape
    verts = np.column_stack([X.ravel(), Y.ravel(), z.ravel()]).astype(np.float32)
    idx = np.arange(h * w).reshape(h, w)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([np.column_stack([a, c, b]), np.column_stack([b, c, d])])
    good = np.isfinite(verts[:, 2])
    faces = faces[good[faces].all(axis=1)]
    verts = np.where(good[:, None], verts, 0.0).astype(np.float32)
    return verts, faces.astype(np.uint32), vertex_normals(verts, faces)


def texture_from_frames(grid: GroundGrid, surface_z: np.ndarray,
                        views: Sequence[View], caster=None,
                        progress: ProgressFn = None, cancel: CancelFn = None,
                        visible: Optional[List[np.ndarray]] = None):
    """Colour every cell from the frames that see it.

    Each frame contributes with weight ``cos(angle)^2`` between the view ray
    and the vertical, so nadir views dominate; with a *caster* over the
    surface, a cell hidden from a camera by the surface itself (a tree in
    front of the ground) is skipped for that camera.

    :return: ``(colour (H, W, 3) float32, views (H, W))``
    """
    X, Y = grid.xy()
    shape = X.shape
    pts = np.column_stack([X.ravel(), Y.ravel(), surface_z.ravel()])
    acc = np.zeros((X.size, 3))
    wsum = np.zeros(X.size)
    seen = np.zeros(X.size, np.int32)
    finite = np.isfinite(pts[:, 2])
    for k, view in enumerate(views):
        _check(cancel)
        if progress:
            progress(f"Texturing from frame {view.index}", k / max(len(views), 1))
        w, h = view.size
        u, v, ok = project_points(view.camera, w, h, pts)
        ok = view.valid(u, v, ok) & finite
        if visible is not None:
            ok &= np.asarray(visible[k]).reshape(-1)
        if not ok.any():
            continue
        cam_pos = view.camera.position
        rays = cam_pos - pts[ok]
        dist = np.linalg.norm(rays, axis=1)
        dirs = rays / dist[:, None]
        if caster is not None:
            starts = pts[ok] + dirs * max(grid.gsd * 0.75, 0.05)
            hits = cast_rays(caster, starts, dirs)
            blocked = np.isfinite(hits[:, 0]) & (
                np.linalg.norm(hits - starts, axis=1) < dist - grid.gsd)
            sel = np.flatnonzero(ok)
            ok[sel[blocked]] = False
            dirs = dirs[~blocked]
        if not ok.any():
            continue
        col = sample_image(view.image, u[ok], v[ok])
        if col.ndim == 1:
            col = np.repeat(col[:, None], 3, axis=1)
        weight = np.clip(dirs[:, 2], 0.0, 1.0) ** 2
        acc[ok] += col[:, :3] * weight[:, None]
        wsum[ok] += weight
        seen[ok] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        color = np.where(wsum[:, None] > 0, acc / wsum[:, None], np.nan)
    return color.reshape(shape + (3,)).astype(np.float32), seen.reshape(shape)


def visibility_masks(grid: GroundGrid, surface_z: np.ndarray,
                     views: Sequence[View], caster,
                     cancel: CancelFn = None) -> List[np.ndarray]:
    """Per view: which cells of *grid* the camera sees unoccluded by the
    surface itself (ray cast towards the camera)."""
    X, Y = grid.xy()
    pts = np.column_stack([X.ravel(), Y.ravel(), surface_z.ravel()])
    finite = np.isfinite(pts[:, 2])
    out = []
    for view in views:
        _check(cancel)
        seen = np.zeros(len(pts), bool)
        w, h = view.size
        u, v, ok = project_points(view.camera, w, h, pts)
        ok &= finite
        if ok.any() and caster is not None:
            rays = view.camera.position - pts[ok]
            dist = np.linalg.norm(rays, axis=1)
            dirs = rays / dist[:, None]
            starts = pts[ok] + dirs * max(grid.gsd * 0.75, 0.05)
            hits = cast_rays(caster, starts, dirs)
            blocked = np.isfinite(hits[:, 0]) & (
                np.linalg.norm(hits - starts, axis=1) < dist - grid.gsd)
            sel = np.flatnonzero(ok)
            seen[sel[~blocked]] = True
        else:
            seen = ok
        out.append(seen.reshape(X.shape))
    return out


def texture_ortho(grid: GroundGrid, surface_z: np.ndarray, views: Sequence[View],
                  texture_gsd: float, max_edge: int = 4096,
                  visibility: Optional[List[np.ndarray]] = None,
                  rows_per_block: int = 256, progress: ProgressFn = None,
                  cancel: CancelFn = None):
    """A high-resolution orthophoto of the reconstructed surface.

    The surface is coarse (its ground cells); the texture is
    sampled from the frames on a much finer grid (*texture_gsd*, at most
    *max_edge* pixels along the longer side), with the surface heights
    interpolated - a sharp image draped on a smooth shape. *visibility*
    (from :func:`visibility_masks`, on the coarse grid) skips cameras that
    do not see a spot. Gaps are filled from their surroundings.

    :return: ``(texture grid, RGB uint8 image (H, W, 3))``
    """
    size = grid.upper - grid.lower
    gsd = max(float(texture_gsd), float(max(size)) / max_edge)
    fine = GroundGrid(grid.lower, grid.upper, gsd)
    img = np.full((fine.height, fine.width, 3), np.nan, np.float32)
    for r0 in range(0, fine.height, rows_per_block):
        _check(cancel)
        if progress:
            progress("Texturing", r0 / max(fine.height, 1))
        r1 = min(fine.height, r0 + rows_per_block)
        top = fine.upper[1] - r0 * gsd
        block = GroundGrid([fine.lower[0], top - (r1 - r0) * gsd],
                           [fine.upper[0], top], gsd,
                           width=fine.width, height=r1 - r0)
        z = upsample_heights(grid, surface_z, block)
        vis = None
        if visibility is not None:
            BX, BY = block.xy()
            cu = np.clip(((BX - grid.lower[0]) / grid.gsd).astype(int), 0, grid.width - 1)
            cv = np.clip(((grid.upper[1] - BY) / grid.gsd).astype(int), 0, grid.height - 1)
            vis = [m[cv, cu] for m in visibility]
        color, _ = texture_from_frames(block, z, views, None, visible=vis)
        img[r0:r1] = color
    for c in range(3):
        img[..., c] = fill_holes(img[..., c], iterations=60)
    img = np.nan_to_num(img, nan=0.0)
    return fine, np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sparse
# ---------------------------------------------------------------------------

def sparse_points(views: Sequence[View], dem_caster=None,
                  band: Tuple[float, float] = (-2.0, 40.0),
                  neighbours: int = 2, ratio: float = 0.75,
                  max_error: float = 2.0, max_features: int = 4000,
                  progress: ProgressFn = None, cancel: CancelFn = None):
    """Triangulate SIFT matches between each keyframe and its next
    *neighbours* keyframes, with the cameras fixed.

    :return: ``(points (N, 3), colours (N, 3) uint8, errors (N,))``
    """
    import cv2
    sift = cv2.SIFT_create(nfeatures=max_features)
    feats = []
    for k, view in enumerate(views):
        _check(cancel)
        if progress:
            progress(f"Features in frame {view.index}", 0.5 * k / max(len(views), 1))
        img = np.asarray(view.image)
        grey = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                            cv2.COLOR_RGB2GRAY)
        feature_mask = None
        if view.mask is not None:
            # Stay clear of the mask edge: the border itself is a strong edge.
            feature_mask = cv2.erode(view.mask.astype(np.uint8) * 255,
                                     np.ones((9, 9), np.uint8))
        kp, desc = sift.detectAndCompute(grey, feature_mask)
        pts = np.array([p.pt for p in kp], np.float64).reshape(-1, 2)
        feats.append((pts, desc))
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    out_p, out_c, out_e = [], [], []
    pairs = [(i, j) for i in range(len(views))
             for j in range(i + 1, min(len(views), i + 1 + neighbours))]
    for n, (i, j) in enumerate(pairs):
        _check(cancel)
        if progress:
            progress(f"Triangulating frames {views[i].index}/{views[j].index}",
                     0.5 + 0.5 * n / max(len(pairs), 1))
        (p1, d1), (p2, d2) = feats[i], feats[j]
        if d1 is None or d2 is None or len(p1) < 8 or len(p2) < 8:
            continue
        knn = matcher.knnMatch(d1, d2, k=2)
        good = [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
        if len(good) < 8:
            continue
        a = p1[[m.queryIdx for m in good]]
        b = p2[[m.trainIdx for m in good]]
        (w1, h1), (w2, h2) = views[i].size, views[j].size
        K1, R1, t1 = opencv_projection(views[i].camera, w1, h1)
        K2, R2, t2 = opencv_projection(views[j].camera, w2, h2)
        P1 = K1 @ np.hstack([R1, t1[:, None]])
        P2 = K2 @ np.hstack([R2, t2[:, None]])
        X = cv2.triangulatePoints(P1, P2, a.T, b.T)
        with np.errstate(invalid="ignore", divide="ignore"):
            X = (X[:3] / X[3]).T
        errs = []
        ok = np.isfinite(X).all(axis=1)
        for P, uv in ((P1, a), (P2, b)):
            proj = np.hstack([X, np.ones((len(X), 1))]) @ P.T
            with np.errstate(invalid="ignore", divide="ignore"):
                pix = proj[:, :2] / proj[:, 2:3]
            ok &= proj[:, 2] > 0
            errs.append(np.linalg.norm(pix - uv, axis=1))
        err = np.maximum(errs[0], errs[1])
        ok &= err < max_error
        if dem_caster is not None and ok.any():
            ground = cast_rays(dem_caster,
                               np.column_stack([X[:, :2], np.full(len(X), 1e4)]),
                               np.tile([0.0, 0.0, -1.0], (len(X), 1)))[:, 2]
            above = X[:, 2] - ground
            ok &= np.isfinite(above) & (above >= band[0]) & (above <= band[1])
        if not ok.any():
            continue
        col = sample_image(views[i].image, a[ok, 0], a[ok, 1])
        out_p.append(X[ok])
        out_c.append(np.clip(col[:, :3], 0, 255).astype(np.uint8))
        out_e.append(err[ok])
    if not out_p:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros(0)
    return np.vstack(out_p), np.vstack(out_c), np.concatenate(out_e)


# ---------------------------------------------------------------------------
# Pose and field-of-view refinement from the images (optional)
# ---------------------------------------------------------------------------

@dataclass
class RefineReport:
    points: int = 0
    observations: int = 0
    rms_before: float = float("nan")      # reprojection error, pixels
    rms_after: float = float("nan")
    fovy_before: float = float("nan")     # degrees (first keyframe)
    fovy_after: float = float("nan")
    median_shift: float = 0.0             # metres
    median_turn: float = 0.0              # degrees


def _features(views, max_features, cancel, progress=None, share=1.0):
    import cv2
    sift = cv2.SIFT_create(nfeatures=max_features)
    feats = []
    for k, view in enumerate(views):
        _check(cancel)
        if progress:
            progress(f"Features in frame {view.index}", share * k / max(len(views), 1))
        img = np.asarray(view.image)
        grey = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                            cv2.COLOR_RGB2GRAY)
        fmask = None
        if view.mask is not None:
            fmask = cv2.erode(view.mask.astype(np.uint8) * 255,
                              np.ones((9, 9), np.uint8))
        kp, desc = sift.detectAndCompute(grey, fmask)
        feats.append((np.array([p.pt for p in kp], np.float64).reshape(-1, 2), desc))
    return feats


def _match_tracks(feats, neighbours, ratio, cancel):
    """Tracks (lists of (view, point)) from ratio-tested pairwise matches."""
    import cv2
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(feats)):
        for j in range(i + 1, min(len(feats), i + 1 + neighbours)):
            _check(cancel)
            d1, d2 = feats[i][1], feats[j][1]
            if d1 is None or d2 is None or len(d1) < 8 or len(d2) < 8:
                continue
            back = {m[0].queryIdx: m[0].trainIdx
                    for m in matcher.knnMatch(d2, d1, k=1) if m}
            # Ratio test and mutual best match.
            good = [m[0] for m in matcher.knnMatch(d1, d2, k=2)
                    if len(m) == 2 and m[0].distance < ratio * m[1].distance
                    and back.get(m[0].trainIdx) == m[0].queryIdx]  # noqa: W503
            if len(good) < 12:
                continue
            # Geometric check that does not depend on the (uncertain) poses:
            # the matches must agree on one epipolar geometry.
            p1 = feats[i][0][[m.queryIdx for m in good]]
            p2 = feats[j][0][[m.trainIdx for m in good]]
            _, inl = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.5, 0.999)
            if inl is None:
                continue
            for m, keep in zip(good, inl.ravel()):
                if keep:
                    a, b = find((i, m.queryIdx)), find((j, m.trainIdx))
                    if a != b:
                        parent[a] = b
    groups = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)
    tracks = []
    for members in groups.values():
        views_in = [v for v, _ in members]
        if len(members) >= 2 and len(set(views_in)) == len(views_in):
            tracks.append(sorted(members))
    return tracks


def _rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def camera_from_opencv(K, R, t, width: int, height: int,
                       near: float = 0.1, far: float = 10000.0) -> FrameCamera:
    """A :class:`FrameCamera` from OpenCV intrinsics / extrinsics."""
    from .flight_viewer import perspective
    flip = np.diag([1.0, -1.0, -1.0])
    view = np.eye(4)
    view[:3, :3] = flip @ R
    view[:3, 3] = flip @ t
    fovy = math.degrees(2.0 * math.atan((height / 2.0) / K[1, 1]))
    aspect = (width / 2.0) / K[0, 0] / ((height / 2.0) / K[1, 1])
    return FrameCamera(view, perspective(fovy, aspect, near, far), fovy, aspect)


def lower_mode(values, width: float) -> float:
    """The densest value in the lower half of *values* (bins of *width*).

    For heights above the DEM this is the ground: ground points form one
    dense cluster, vegetation spreads over many heights above it.
    """
    v = np.sort(np.asarray(values, dtype=np.float64))
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan")
    low = v[: max(1, len(v) // 2 + 1)]
    if low[-1] - low[0] < width:
        return float(np.median(low))
    edges = np.arange(low[0], low[-1] + width, width)
    counts, edges = np.histogram(low, bins=edges)
    k = int(np.argmax(counts))
    near = low[(low >= edges[k] - width / 2) & (low <= edges[k + 1] + width / 2)]
    return float(np.median(near)) if len(near) else float(np.median(low))


def ground_sampler(grid: "GroundGrid", dem_z: np.ndarray):
    """``f(x, y) -> DEM height`` by bilinear interpolation of a DEM height
    grid (NaN outside it) - fast enough to call inside an optimiser."""
    z = np.asarray(dem_z, dtype=np.float64)

    def f(x, y):
        u = (np.asarray(x) - grid.lower[0]) / grid.gsd - 0.5
        v = (grid.upper[1] - np.asarray(y)) / grid.gsd - 0.5
        inside = (u >= 0) & (v >= 0) & (u <= z.shape[1] - 1) & (v <= z.shape[0] - 1)
        u = np.clip(u, 0, z.shape[1] - 1.000001)
        v = np.clip(v, 0, z.shape[0] - 1.000001)
        i, j = v.astype(int), u.astype(int)
        fv_, fu = v - i, u - j
        out = (z[i, j] * (1 - fv_) * (1 - fu) + z[i, j + 1] * (1 - fv_) * fu
               + z[i + 1, j] * fv_ * (1 - fu) + z[i + 1, j + 1] * fv_ * fu)  # noqa: W503
        return np.where(inside, out, np.nan)
    return f


def refine_poses(views: Sequence[View], refine_fov: bool = True,
                 ground=None, sigma_ground: float = 0.5,
                 sigma_position: float = 2.0, sigma_rotation_deg: float = 2.0,
                 neighbours: int = 3, ratio: float = 0.75,
                 max_features: int = 3000, max_points: int = 4000,
                 progress: ProgressFn = None, cancel: CancelFn = None):
    """Refine the keyframes' poses - and optionally one shared field of view -
    from the images (a small bundle adjustment).

    SIFT tracks between neighbouring keyframes are triangulated and every
    camera's position and rotation (plus a common focal-length scale) are
    adjusted to minimise the reprojection error, with the telemetry poses
    as soft priors (*sigma_position* metres, *sigma_rotation_deg* degrees),
    so the solution stays where the drone was but becomes consistent with
    the images. Depth from nearby frames is very sensitive to relative pose
    and focal-length errors, which is what this fixes.

    With straight-down frames from one altitude the focal length cannot be
    told apart from the height of the scene, so the field of view needs
    *ground* (``f(x, y) -> DEM height``, see :func:`ground_sampler`): points
    are softly held on the DEM (*sigma_ground* metres, robustly, so points on
    trees do not count).

    :return: ``(refined views, RefineReport)``
    """
    import cv2
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    views = list(views)
    report = RefineReport()
    if len(views) < 2:
        return views, report
    feats = _features(views, max_features, cancel, progress, 0.4)
    if progress:
        progress("Matching features between keyframes", 0.4)
    tracks = _match_tracks(feats, neighbours, ratio, cancel)
    tracks.sort(key=len, reverse=True)
    tracks = tracks[:max_points]

    Ks, Rs, Cs = [], [], []
    for v in views:
        w, h = v.size
        K, R, t = opencv_projection(v.camera, w, h)
        Ks.append(K)
        Rs.append(R)
        Cs.append(-R.T @ t)

    # Initial points: triangulate each track from its two furthest views.
    pts, obs_view, obs_point, obs_uv = [], [], [], []
    for members in tracks:
        (i, a), (j, b) = members[0], members[-1]
        P1 = Ks[i] @ np.hstack([Rs[i], (-Rs[i] @ Cs[i])[:, None]])
        P2 = Ks[j] @ np.hstack([Rs[j], (-Rs[j] @ Cs[j])[:, None]])
        Xh = cv2.triangulatePoints(P1, P2, feats[i][0][a][:, None],
                                   feats[j][0][b][:, None]).ravel()
        if abs(Xh[3]) < 1e-12:
            continue
        X = Xh[:3] / Xh[3]
        if not np.all(np.isfinite(X)):
            continue
        if any((Rs[v] @ (X - Cs[v]))[2] <= 0 for v, _ in members):
            continue
        pid = len(pts)
        pts.append(X)
        for v, f in members:
            obs_view.append(v)
            obs_point.append(pid)
            obs_uv.append(feats[v][0][f])
    if len(pts) < 20:
        return views, report
    pts = np.array(pts)
    obs_view = np.array(obs_view)
    obs_point = np.array(obs_point)
    obs_uv = np.array(obs_uv)
    nv, npnt = len(views), len(pts)
    report.points, report.observations = npnt, len(obs_uv)

    sr = math.radians(sigma_rotation_deg)
    n_cam = 6 * nv
    n_glob = 1 if refine_fov else 0

    def unpack(x):
        cam = x[:n_cam].reshape(nv, 6)
        scale = math.exp(x[n_cam]) if refine_fov else 1.0
        X = x[n_cam + n_glob:].reshape(npnt, 3)
        return cam, scale, X

    def residuals(x):
        cam, scale, X = unpack(x)
        out = np.empty((len(obs_uv), 2))
        for v in range(nv):
            sel = obs_view == v
            if not sel.any():
                continue
            R = _rodrigues(cam[v, :3]) @ Rs[v]
            C = Cs[v] + cam[v, 3:]
            pc = (X[obs_point[sel]] - C) @ R.T
            z = np.where(np.abs(pc[:, 2]) < 1e-6, 1e-6, pc[:, 2])
            K = Ks[v]
            out[sel, 0] = scale * K[0, 0] * pc[:, 0] / z + K[0, 2] - obs_uv[sel, 0]
            out[sel, 1] = scale * K[1, 1] * pc[:, 1] / z + K[1, 2] - obs_uv[sel, 1]
        prior = np.concatenate([(cam[:, :3] / sr).ravel(),
                                (cam[:, 3:] / sigma_position).ravel()])
        parts = [out.ravel(), prior]
        if ground is not None:
            g = ground(X[:, 0], X[:, 1])
            parts.append(np.where(np.isfinite(g),
                                  (X[:, 2] - np.nan_to_num(g)) / sigma_ground, 0.0))
        return np.concatenate(parts)

    first = {}
    for v, p in zip(obs_view, obs_point):
        first.setdefault(p, v)
    cam_of = np.array([first[p] for p in range(npnt)])

    def depth_scale(X, centres):
        """Focal scale from the DEM: a wrong focal length scales every
        triangulated depth by one factor, and the DEM gives the true distance
        to the ground. The median ratio (most points are on the ground) is
        robust to vegetation."""
        g = ground(X[:, 0], X[:, 1])
        tri = centres[:, 2] - X[:, 2]
        ok = np.isfinite(g) & (tri > 1e-3)
        if ok.sum() < 20:
            return 1.0
        ratio = (centres[ok, 2] - g[ok]) / tri[ok]
        # Points on vegetation give larger ratios than ground points; the
        # ground is the dense cluster at the low end (see lower_mode).
        s0 = lower_mode(ratio, 0.04)
        return s0 if np.isfinite(s0) and 0.6 < s0 < 1.4 else 1.0

    x0 = np.concatenate([np.zeros(n_cam + n_glob), pts.ravel()])
    n_obs_all = len(obs_uv)

    def rms(x, keep):
        r = residuals(x)[:2 * n_obs_all].reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum(r[keep] ** 2, axis=1))))

    def solve(x_start, w_obs, w_ground, nfev, fix_fov=False):
        """Weighted least squares over poses, points (and the focal scale
        unless *fix_fov*); robustness comes from the weights."""
        sq_obs = np.repeat(np.sqrt(w_obs), 2)
        sq_g = np.sqrt(w_ground)
        free = np.ones(len(x_start), bool)
        if refine_fov and fix_fov:
            free[n_cam] = False
        full = x_start.copy()

        def fun(y):
            x = full.copy()
            x[free] = y
            r = residuals(x)
            r[:2 * n_obs_all] *= sq_obs
            if ground is not None:
                r[2 * n_obs_all + n_cam:] *= sq_g
            return r

        m = 2 * n_obs_all + n_cam + (npnt if ground is not None else 0)
        sparsity = lil_matrix((m, len(x_start)), dtype=int)
        if ground is not None:
            r0 = 2 * n_obs_all + n_cam
            for k in range(3):
                sparsity[r0 + np.arange(npnt), n_cam + n_glob + 3 * np.arange(npnt) + k] = 1
        rows = np.arange(n_obs_all)
        for k in range(6):
            sparsity[2 * rows, 6 * obs_view + k] = 1
            sparsity[2 * rows + 1, 6 * obs_view + k] = 1
        if refine_fov:
            sparsity[2 * rows, n_cam] = 1
            sparsity[2 * rows + 1, n_cam] = 1
        for k in range(3):
            sparsity[2 * rows, n_cam + n_glob + 3 * obs_point + k] = 1
            sparsity[2 * rows + 1, n_cam + n_glob + 3 * obs_point + k] = 1
        for k in range(n_cam):
            sparsity[2 * n_obs_all + k, k] = 1
        sparsity = sparsity.tocsc()[:, np.flatnonzero(free)]
        y = least_squares(fun, x_start[free], jac_sparsity=sparsity,
                          method="trf", max_nfev=nfev, x_scale=x_scale[free],
                          verbose=0).x
        out = full.copy()
        out[free] = y
        return out

    def reweight(x, c_px, c_ground):
        """Cauchy weights: wrong matches and points standing on trees or
        buildings (far from the DEM) lose their pull."""
        e = np.linalg.norm(residuals(x)[:2 * n_obs_all].reshape(-1, 2), axis=1)
        w_obs = 1.0 / (1.0 + (e / c_px) ** 2)
        w_g = np.ones(npnt)
        if ground is not None:
            _, _, X = unpack(x)
            g = ground(X[:, 0], X[:, 1])
            d = np.where(np.isfinite(g), X[:, 2] - np.nan_to_num(g), 0.0)
            # Nothing lies below the ground: only points above the DEM
            # (vegetation, buildings) may lose the pull.
            w_g = np.where(d > 0, 1.0 / (1.0 + (d / c_ground) ** 2), 1.0)
        return w_obs, w_g, e

    # Typical step sizes: rotations ~0.01 rad, positions and points ~0.5 m,
    # the focal scale ~1 %. Balances the steps of very different units.
    x_scale = np.concatenate([np.tile([0.01] * 3 + [0.5] * 3, nv),
                              [0.01] * n_glob, np.full(3 * npnt, 0.5)])
    report.rms_before = rms(x0, np.ones(n_obs_all, bool))
    x = x0.copy()
    fov_rounds = 1
    for rnd in range(fov_rounds):
        if refine_fov and ground is not None:
            cam, scale, X = unpack(x)
            centres = np.array([Cs[v] + cam[v, 3:] for v in range(nv)])[cam_of]
            ratio = depth_scale(X, centres)
            x[n_cam] = math.log(scale * ratio)
            # Only the depth was off, not the ground position.
            X = X.copy()
            X[:, 2] = centres[:, 2] - ratio * (centres[:, 2] - X[:, 2])
            x[n_cam + n_glob:] = X.ravel()
        w_obs, w_g, _ = reweight(x, 10.0, 3.0)
        for it, c_px in enumerate((10.0, 4.0, 2.0)):
            _check(cancel)
            if progress:
                progress(f"Adjusting {nv} poses and {npnt} points",
                         0.5 + 0.45 * (rnd * 3 + it) / (fov_rounds * 3))
            x = solve(x, w_obs, np.zeros(npnt), 60, fix_fov=True)
            w_obs, _, err = reweight(x, c_px, 1.0)
    keep = err < max(3.0, 3.0 * float(np.median(err)))
    if ground is not None:
        # The adjusted scene can float up or down as a whole within the
        # position priors; put it back on the DEM. Vegetation stands above
        # the ground, so the ground is the dense cluster at the low end of
        # the points (lower_mode) - that goes onto the DEM. A pure vertical
        # shift - the relative geometry stays.
        cam, scale, X = unpack(x)
        g = ground(X[:, 0], X[:, 1])
        d = X[:, 2] - g
        d = d[np.isfinite(d)]
        if len(d) >= 20:
            dz = lower_mode(d, 0.25)
            cam = cam.copy()
            cam[:, 5] -= dz
            X = X.copy()
            X[:, 2] -= dz
            x = np.concatenate([cam.ravel(), x[n_cam:n_cam + n_glob], X.ravel()])
    sol_x = x
    report.observations = int(keep.sum())
    report.rms_after = rms(sol_x, keep)

    class _Sol:
        x = sol_x
    sol = _Sol()
    cam, scale, _ = unpack(sol.x)

    refined = []
    shifts, turns = [], []
    for v, view in enumerate(views):
        R = _rodrigues(cam[v, :3]) @ Rs[v]
        C = Cs[v] + cam[v, 3:]
        K = Ks[v].copy()
        K[0, 0] *= scale
        K[1, 1] *= scale
        w, h = view.size
        new_cam = camera_from_opencv(K, R, -R @ C, w, h)
        refined.append(View(view.index, new_cam, view.image, view.mask))
        shifts.append(np.linalg.norm(cam[v, 3:]))
        turns.append(math.degrees(np.linalg.norm(cam[v, :3])))
    report.fovy_before = views[0].camera.fovy
    report.fovy_after = refined[0].camera.fovy
    report.median_shift = float(np.median(shifts))
    report.median_turn = float(np.median(turns))
    if progress:
        progress("Poses refined", 1.0)
    return refined, report


def height_above(points, caster) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ground = cast_rays(caster, np.column_stack([pts[:, :2], np.full(len(pts), 1e4)]),
                       np.tile([0.0, 0.0, -1.0], (len(pts), 1)))[:, 2]
    return pts[:, 2] - ground


# ---------------------------------------------------------------------------
# Colour maps and exports
# ---------------------------------------------------------------------------

def height_colors(values, vmin: float = 0.0, vmax: float = 25.0) -> np.ndarray:
    """RGB floats 0..1 for heights above ground (blue - green - yellow - red)."""
    t = np.clip(np.nan_to_num((np.asarray(values, dtype=np.float64) - vmin) / max(vmax - vmin, 1e-6)), 0, 1)
    stops = np.array([[0.20, 0.30, 0.75], [0.15, 0.70, 0.35],
                      [0.95, 0.85, 0.20], [0.85, 0.20, 0.15]])
    pos = t * (len(stops) - 1)
    i = np.minimum(pos.astype(int), len(stops) - 2)
    f = (pos - i)[..., None]
    out = stops[i] * (1 - f) + stops[i + 1] * f
    return np.where(np.isfinite(np.asarray(values))[..., None], out, 0.5)


def write_colmap_model(folder: str, views: Sequence[View], names: Sequence[str]):
    """COLMAP text model with the fixed poses and no points, ready for
    ``colmap point_triangulator`` and dense reconstruction elsewhere."""
    import os
    os.makedirs(folder, exist_ok=True)
    cams, imgs = [], []
    for k, (view, name) in enumerate(zip(views, names), start=1):
        w, h = view.size
        K, R, t = opencv_projection(view.camera, w, h)
        cams.append(f"{k} PINHOLE {w} {h} {K[0, 0]:.6f} {K[1, 1]:.6f} "
                    f"{K[0, 2]:.6f} {K[1, 2]:.6f}")
        q = _quat_from_matrix(R)
        imgs.append(f"{k} {q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {k} {name}\n")
    with open(os.path.join(folder, "cameras.txt"), "w", encoding="utf-8") as fh:
        fh.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n" + "\n".join(cams) + "\n")
    with open(os.path.join(folder, "images.txt"), "w", encoding="utf-8") as fh:
        fh.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
                 "# (second line per image: no 2D points)\n")
        for line in imgs:
            fh.write(line + "\n")
    with open(os.path.join(folder, "points3D.txt"), "w", encoding="utf-8") as fh:
        fh.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")


def _quat_from_matrix(R) -> Tuple[float, float, float, float]:
    """(w, x, y, z) of a rotation matrix."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return w, x, y, z


def write_geotiff(path: str, array: np.ndarray, grid: GroundGrid,
                  origin=None, epsg: Optional[int] = None,
                  nodata: float = -9999.0) -> bool:
    """Single-band float32 (or 3-band uint8) GeoTIFF of a grid; False when
    GDAL is not available (QGIS always ships it)."""
    try:
        from osgeo import gdal, osr
    except ImportError:
        return False
    arr = np.asarray(array)
    bands = 1 if arr.ndim == 2 else arr.shape[2]
    dtype = gdal.GDT_Float32 if bands == 1 else gdal.GDT_Byte
    ox, oy = (float(origin[0]), float(origin[1])) if origin is not None else (0.0, 0.0)
    ds = gdal.GetDriverByName("GTiff").Create(
        path, grid.width, grid.height, bands, dtype, ["COMPRESS=DEFLATE"])
    ds.SetGeoTransform((ox + grid.lower[0], grid.gsd, 0.0,
                        oy + grid.upper[1], 0.0, -grid.gsd))
    if epsg:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(epsg))
        ds.SetProjection(srs.ExportToWkt())
    if bands == 1:
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(nodata)
        band.WriteArray(np.where(np.isfinite(arr), arr, nodata).astype(np.float32))
    else:
        for b in range(bands):
            ds.GetRasterBand(b + 1).WriteArray(
                np.clip(np.nan_to_num(arr[..., b]), 0, 255).astype(np.uint8))
    ds.FlushCache()
    ds = None
    return True


# ---------------------------------------------------------------------------
# Keeping a dense result for the next session
# ---------------------------------------------------------------------------

SURFACE_FILE = "surface.npz"


def _grid_record(prefix, grid):
    return {f"{prefix}_lower": grid.lower, f"{prefix}_upper": grid.upper,
            f"{prefix}_gsd": np.float64(grid.gsd),
            f"{prefix}_size": np.array([grid.width, grid.height])}


def _grid_from(data, prefix):
    w, h = (int(v) for v in data[f"{prefix}_size"])
    return GroundGrid(data[f"{prefix}_lower"], data[f"{prefix}_upper"],
                      float(data[f"{prefix}_gsd"]), width=w, height=h)


def save_surface(path: str, grid: GroundGrid, surface_z, dem_z, confidence,
                 views, textures=None, origin=None) -> None:
    """Store a dense reconstruction so the viewer can show it again later.

    :param textures: ``{"rgb": (grid, image), "thermal": (grid, image)}``
    """
    rec = dict(_grid_record("grid", grid),
               surface_z=np.asarray(surface_z, np.float32),
               dem_z=np.asarray(dem_z, np.float32),
               confidence=np.asarray(confidence, np.float32),
               views=np.asarray(views, np.uint16),
               origin=np.asarray(origin if origin is not None else [np.nan] * 3,
                                 np.float64))
    for name, (tex_grid, image) in (textures or {}).items():
        rec.update(_grid_record(f"{name}_grid", tex_grid))
        rec[name] = np.asarray(image, np.uint8)
    np.savez_compressed(path, **rec)


def load_surface(path: str) -> dict:
    """Read what :func:`save_surface` wrote.

    :return: dict with ``grid``, ``surface_z``, ``dem_z``, ``confidence``,
        ``views``, ``origin`` and, when stored, ``rgb`` / ``tex_grid`` and
        ``thermal`` / ``thermal_grid``
    """
    with np.load(path) as data:
        out = dict(grid=_grid_from(data, "grid"),
                   surface_z=data["surface_z"], dem_z=data["dem_z"],
                   confidence=data["confidence"], views=data["views"],
                   origin=data["origin"])
        if "rgb" in data:
            out.update(rgb=data["rgb"], tex_grid=_grid_from(data, "rgb_grid"))
        if "thermal" in data:
            out.update(thermal=data["thermal"],
                       thermal_grid=_grid_from(data, "thermal_grid"))
    return out


def nadir_distances(positions, surface_height, ground_height) -> List[dict]:
    """Vertical (nadir) distance of camera positions to the reconstructed
    surface and to the DEM.

    :param positions: ``(N, 3)`` camera positions, DEM-local
    :param surface_height: ``f(x, y) -> z`` of the surface (NaN outside)
    :param ground_height: ``f(x, y) -> z`` of the DEM (NaN outside)
    :return: per position ``x, y, z, surface, dem, above_surface,
        above_dem`` (``None`` where unknown)
    """
    pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    s = np.asarray(surface_height(pts[:, 0], pts[:, 1]), dtype=np.float64)
    g = np.asarray(ground_height(pts[:, 0], pts[:, 1]), dtype=np.float64)
    rows = []
    for (x, y, z), sv, gv in zip(pts, s, g):
        rows.append(dict(
            x=float(x), y=float(y), z=float(z),
            surface=float(sv) if np.isfinite(sv) else None,
            dem=float(gv) if np.isfinite(gv) else None,
            above_surface=float(z - sv) if np.isfinite(sv) else None,
            above_dem=float(z - gv) if np.isfinite(gv) else None))
    return rows


def write_nadir_csv(path: str, rows: List[dict], names: Sequence[str],
                    origin=None) -> None:
    """CSV of :func:`nadir_distances` with project-CRS coordinates."""
    import csv
    ox, oy, oz = (float(v) for v in origin) if origin is not None else (0, 0, 0)

    def f(v, off=0.0):
        return "" if v is None else f"{v + off:.3f}"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "imagefile", "east", "north", "altitude",
                    "surface_height", "dem_height", "above_surface",
                    "above_dem"])
        for i, (row, name) in enumerate(zip(rows, names)):
            w.writerow([i, name, f(row["x"], ox), f(row["y"], oy),
                        f(row["z"], oz), f(row["surface"], oz),
                        f(row["dem"], oz), f(row["above_surface"]),
                        f(row["above_dem"])])


# ---------------------------------------------------------------------------
# Dense surface: pairwise stereo, merged by votes per ground cell
# ---------------------------------------------------------------------------

class HeightVotes:
    """Per ground cell, a histogram of heights above the DEM voted for by
    point sets (one vote per set and cell: the set's median height there)."""

    def __init__(self, grid: GroundGrid, dem_z: np.ndarray,
                 heights: Sequence[float]):
        self.grid = grid
        self.dem_z = np.asarray(dem_z, dtype=np.float64)
        self.heights = np.asarray(heights, dtype=np.float64)
        if len(self.heights) < 2:
            raise ValueError("at least two heights are needed")
        self.step = float(self.heights[1] - self.heights[0])
        self.votes = np.zeros((len(self.heights), grid.height, grid.width),
                              np.uint16)

    def cells(self, points):
        """(row, col, height above DEM, ok) of world points."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        g = self.grid
        col = np.floor((pts[:, 0] - g.lower[0]) / g.gsd).astype(np.int64)
        row = np.floor((g.upper[1] - pts[:, 1]) / g.gsd).astype(np.int64)
        ok = (col >= 0) & (col < g.width) & (row >= 0) & (row < g.height) \
            & np.isfinite(pts[:, 2])
        above = np.full(len(pts), np.nan)
        above[ok] = pts[ok, 2] - self.dem_z[row[ok], col[ok]]
        lo = self.heights[0] - self.step / 2
        hi = self.heights[-1] + self.step / 2
        ok &= np.isfinite(above) & (above >= lo) & (above < hi)
        return row, col, above, ok

    def add(self, points, top: bool = False) -> int:
        """One vote per cell the point set covers; returns the cells voted.

        The set's height in a cell is the mean of its points there, or with
        *top* the highest (for clouds whose points sample a canopy)."""
        row, col, above, ok = self.cells(points)
        if not ok.any():
            return 0
        g = self.grid
        flat = row[ok] * g.width + col[ok]
        h = above[ok]
        n_cells = g.width * g.height
        if top:
            best = np.full(n_cells, -np.inf)
            np.maximum.at(best, flat, h)
            hit = np.isfinite(best)
            value = best[hit]
        else:
            count = np.bincount(flat, minlength=n_cells)
            total = np.bincount(flat, weights=h, minlength=n_cells)
            hit = count > 0
            value = total[hit] / count[hit]
        k = np.clip(np.rint((value - self.heights[0]) / self.step).astype(int),
                    0, len(self.heights) - 1)
        cells = np.flatnonzero(hit)
        view = self.votes.reshape(len(self.heights), -1)
        view[k, cells] = np.minimum(view[k, cells].astype(np.int64) + 1, 65535)
        return int(hit.sum())

    def result(self, min_votes: int = 1) -> "DenseResult":
        """Median height per cell; confidence = share of the votes within
        one step of it."""
        v = self.votes.astype(np.float32)
        total = v.sum(axis=0)
        cum = np.cumsum(v, axis=0)
        k = np.argmax(cum >= np.maximum(total, 1e-9) / 2.0, axis=0)
        rows, cols = np.indices(total.shape)
        near = v[k, rows, cols]
        for d in (-1, 1):
            kk = np.clip(k + d, 0, len(self.heights) - 1)
            near = near + np.where(kk != k, v[kk, rows, cols], 0.0)
        ok = total >= max(1, min_votes)
        offset = np.where(ok, self.heights[k], np.nan).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            conf = np.where(ok, near / total, 0.0).astype(np.float32)
        return DenseResult(offset, conf, total.astype(np.uint16),
                           np.zeros(total.shape + (3,), np.float32),
                           self.heights)


# -- pairwise stereo ---------------------------------------------------------

def _grey8(view: "View") -> np.ndarray:
    import cv2
    return cv2.cvtColor(np.clip(np.asarray(view.image), 0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY)


def relative_pose_from_images(v1: "View", v2: "View", R0, T0,
                              ratio: float = 0.75, min_inliers: int = 40):
    """Relative pose ``x2 = R x1 + T`` of a pair from its images (essential
    matrix from SIFT matches), with the baseline length of the telemetry.

    Falls back to the telemetry pose ``(R0, T0)`` when the images do not
    give a trustworthy answer (few inliers, or a direction far from the
    telemetry's).

    :return: ``(R, T, from_images: bool)``
    """
    import cv2
    sift = cv2.SIFT_create(nfeatures=3000)
    feats = []
    for v in (v1, v2):
        fmask = None
        if v.mask is not None:
            fmask = cv2.erode(v.mask.astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
        kp, desc = sift.detectAndCompute(_grey8(v), fmask)
        if desc is None or len(kp) < min_inliers:
            return R0, T0, False
        feats.append((np.array([p.pt for p in kp], np.float64), desc))
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    back = {m[0].queryIdx: m[0].trainIdx
            for m in matcher.knnMatch(feats[1][1], feats[0][1], k=1) if m}
    good = [m[0] for m in matcher.knnMatch(feats[0][1], feats[1][1], k=2)
            if len(m) == 2 and m[0].distance < ratio * m[1].distance and back.get(m[0].trainIdx) == m[0].queryIdx]
    if len(good) < min_inliers:
        return R0, T0, False
    norm = []
    for v, pts, idx in ((v1, feats[0][0], [m.queryIdx for m in good]),
                        (v2, feats[1][0], [m.trainIdx for m in good])):
        K, _, _ = opencv_projection(v.camera, *v.size)
        p = pts[idx]
        norm.append(np.column_stack([(p[:, 0] - K[0, 2]) / K[0, 0],
                                     (p[:, 1] - K[1, 2]) / K[1, 1]]))
    K1, _, _ = opencv_projection(v1.camera, *v1.size)
    E, inl = cv2.findEssentialMat(norm[0], norm[1], np.eye(3), cv2.RANSAC,
                                  0.999, 1.5 / K1[0, 0])
    if E is None or E.shape != (3, 3) or inl is None or inl.sum() < min_inliers:
        return R0, T0, False
    n, R, t, _ = cv2.recoverPose(E, norm[0], norm[1], np.eye(3), mask=inl)
    if n < min_inliers:
        return R0, T0, False
    t = t.ravel()
    base = float(np.linalg.norm(T0))
    if base < 1e-6 or np.dot(t, T0) / base < math.cos(math.radians(30)):
        return R0, T0, False
    return R, t * base, True


def stereo_pair_points(v1: "View", v2: "View", depth_range: Tuple[float, float],
                       from_images: bool = True, block: int = 5,
                       stride: int = 2, max_disparities: int = 512,
                       _swapped: bool = False, _pose=None):
    """World points from one pair of frames by rectified stereo matching
    (OpenCV semi-global block matching with a left-right check).

    :param depth_range: ``(nearest, farthest)`` distance of the scene along
        the first camera's optical axis, metres
    :return: ``(points (N, 3), camera centre of the reference view,
        geometry from images?)`` or ``None`` when the pair cannot be matched
    """
    import cv2
    (w, h) = v1.size
    if v2.size != (w, h):
        return None
    K1, R1w, t1 = opencv_projection(v1.camera, w, h)
    K2, R2w, t2 = opencv_projection(v2.camera, w, h)
    if _pose is None:
        R = R2w @ R1w.T
        T = t2 - R @ t1
        used_images = False
        if from_images:
            R, T, used_images = relative_pose_from_images(v1, v2, R, T)
    else:
        R, T, used_images = _pose
    B = float(np.linalg.norm(T))
    if B < 1e-3:
        return None
    zero = np.zeros(5)
    R1r, R2r, P1, P2, _, _, _ = cv2.stereoRectify(
        K1, zero, K2, zero, (w, h), R, T.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
    vertical = abs(P2[1, 3]) > abs(P2[0, 3])
    shift = P2[1, 3] if vertical else P2[0, 3]
    if shift > 0:
        # The second camera is on the "left": swap the roles once.
        if _swapped:
            return None
        return stereo_pair_points(v2, v1, depth_range, from_images, block,
                                  stride, max_disparities, True,
                                  (R.T, -R.T @ T, used_images))
    m1 = cv2.initUndistortRectifyMap(K1, zero, R1r, P1, (w, h), cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(K2, zero, R2r, P2, (w, h), cv2.CV_32FC1)
    left = cv2.remap(_grey8(v1), *m1, cv2.INTER_LINEAR)
    right = cv2.remap(_grey8(v2), *m2, cv2.INTER_LINEAR)
    ones = np.full((h, w), 255, np.uint8)
    valid = cv2.remap(ones if v1.mask is None else v1.mask.astype(np.uint8) * 255,
                      *m1, cv2.INTER_NEAREST) > 0
    f = float(P1[1, 1] if vertical else P1[0, 0])
    near, far = max(0.5, depth_range[0]), max(depth_range[1], depth_range[0] + 1)
    d_min = int(math.floor(f * B / far)) - 2
    d_max = int(math.ceil(f * B / near)) + 2
    num = min(max_disparities, max(16, int(math.ceil((d_max - d_min) / 16.0)) * 16))
    if vertical:
        left, right = left.T.copy(), right.T.copy()
    sgbm = cv2.StereoSGBM_create(
        minDisparity=d_min, numDisparities=num, blockSize=block,
        P1=8 * block * block, P2=32 * block * block, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(left, right).astype(np.float32) / 16.0
    if vertical:
        disp = disp.T
    ok = valid & (disp > d_min - 0.5) & (disp > 0.1)
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    sel = ok[vs, us]
    if sel.sum() < 100:
        return None
    u = us[sel].astype(np.float64)
    v = vs[sel].astype(np.float64)
    Z = f * B / disp[vs, us][sel].astype(np.float64)
    rect = np.column_stack([(u - P1[0, 2]) * Z / P1[0, 0],
                            (v - P1[1, 2]) * Z / P1[1, 1], Z])
    cam = rect @ R1r                       # = (R1r.T @ rect.T).T
    centre = -R1w.T @ t1
    world = cam @ R1w + centre             # = (R1w.T @ cam.T).T + C
    return world, centre, used_images


def stereo_surface(grid: GroundGrid, dem_z: np.ndarray, views: Sequence["View"],
                   heights: Sequence[float], neighbours: int = 2,
                   from_images: bool = True, dem_scale: bool = True,
                   block: int = 5, stride: int = 2,
                   progress: ProgressFn = None,
                   cancel: CancelFn = None) -> "DenseResult":
    """Dense surface from pairwise stereo.

    Every keyframe is matched against its next *neighbours* keyframes in a
    rectified pair; each pair's depth map becomes world points that vote for
    a height per ground cell, and the median vote wins.

    With *from_images* the pair's relative orientation comes from the images
    (essential matrix), so telemetry errors between the two frames do not
    tilt the depth map. The baseline length still comes from the telemetry;
    *dem_scale* corrects it per pair by scaling the depths so the pair's
    ground (the dense low cluster of its points) lies on the DEM.
    """
    heights = np.asarray(heights, dtype=np.float64)
    votes = HeightVotes(grid, dem_z, heights)
    ground = ground_sampler(grid, dem_z)
    pairs = [(i, j) for i in range(len(views))
             for j in range(i + 1, min(len(views), i + 1 + max(1, neighbours)))]
    used = img_pairs = 0
    for n, (i, j) in enumerate(pairs):
        _check(cancel)
        if progress:
            progress(f"Stereo pair {views[i].index} / {views[j].index}",
                     n / max(len(pairs), 1))
        c = views[i].camera.position
        g0 = ground(np.array([c[0]]), np.array([c[1]]))[0]
        if not np.isfinite(g0):
            continue
        above = c[2] - g0
        depth = (above - heights[-1] - 2.0, (above - heights[0]) * 1.15 + 2.0)
        out = stereo_pair_points(views[i], views[j], depth, from_images,
                                 block, stride)
        if out is None:
            continue
        pts, centre, from_img = out
        if dem_scale:
            g = ground(pts[:, 0], pts[:, 1])
            tri = centre[2] - pts[:, 2]
            ok = np.isfinite(g) & (tri > 1e-3)
            if ok.sum() >= 200:
                s = lower_mode(((centre[2] - g[ok]) / tri[ok])[::7], 0.01)
                if np.isfinite(s) and 0.85 < s < 1.15:
                    pts = centre + (pts - centre) * s
        if votes.add(pts):
            used += 1
            img_pairs += int(from_img)
    if progress:
        progress("Stereo done", 1.0)
    res = votes.result()
    res.extras["pairs"] = np.array([len(pairs), used, img_pairs])
    return res
