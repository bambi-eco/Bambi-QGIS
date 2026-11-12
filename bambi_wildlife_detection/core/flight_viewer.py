# -*- coding: utf-8 -*-
"""Headless logic of the 3D Flight Viewer.

Everything the viewer draws is derived here, without Qt or OpenGL, so it can
be unit-tested: the flight's poses with their correction factors applied, one
camera per frame (view/projection matrices, position, frustum), the DEM
arrays, the ALFS frame window, frame matching between the two cameras, and
the annotation geometry for a frame.

Matrix convention
-----------------
alfspy and pyrr use row vectors (``clip = p @ V @ P``). The viewer's OpenGL
code uses the usual column-vector convention (``clip = P @ V @ p``). Every
matrix returned by this module is **column-vector**; :func:`to_row` converts
back for code that wants alfspy's form.

Cameras
-------
Every camera comes from :func:`core.camera_pose.render_camera` - the plugin's
one camera model, shared by geo-referencing and rendering - so what the
viewer drapes and projects is where the plugin's results are.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Camera keys of the plugin (``poses_{key}.json``, ``frames_{key}/``).
MODALITIES = ("w", "t")
MODALITY_LABELS = {"w": "RGB", "t": "Thermal"}


DEFAULT_FOVY = 50.0
_ZERO = {"x": 0.0, "y": 0.0, "z": 0.0}


# ---------------------------------------------------------------------------
# Small matrix helpers (column-vector convention)
# ---------------------------------------------------------------------------

def to_row(matrix) -> np.ndarray:
    """Column-vector matrix -> row-vector (alfspy/pyrr) matrix and back."""
    return np.asarray(matrix, dtype=np.float64).T.copy()


def perspective(fovy_deg: float, aspect: float, near: float,
                far: float) -> np.ndarray:
    """OpenGL perspective projection (column-vector)."""
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / max(aspect, 1e-9)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(left: float, right: float, bottom: float, top: float,
                 near: float, far: float) -> np.ndarray:
    """OpenGL orthographic projection (column-vector)."""
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """View matrix looking from *eye* at *target* (column-vector)."""
    eye = np.asarray(eye, dtype=np.float64)
    f = np.asarray(target, dtype=np.float64) - eye
    f /= max(np.linalg.norm(f), 1e-12)
    up = np.asarray(up, dtype=np.float64)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-9:          # looking straight along up
        s = np.cross(f, np.array([0.0, 1.0, 0.0]))
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float64)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def transform_points(matrix, points) -> np.ndarray:
    """Apply a 4x4 column-vector matrix to (N, 3) points (with divide)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homo = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(matrix).T
    return homo[:, :3] / homo[:, 3:4]


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

@dataclass
class FrameCamera:
    """One frame's corrected camera. Matrices are column-vector."""
    view: np.ndarray
    proj: np.ndarray
    fovy: float
    aspect: float

    @property
    def to_world(self) -> np.ndarray:
        return np.linalg.inv(self.view)

    @property
    def position(self) -> np.ndarray:
        return self.to_world[:3, 3].copy()

    @property
    def rotation(self) -> np.ndarray:
        """Camera-to-world rotation; columns are right, up and backward."""
        return self.to_world[:3, :3].copy()

    @property
    def forward(self) -> np.ndarray:
        return -self.rotation[:, 2]

    @property
    def up(self) -> np.ndarray:
        return self.rotation[:, 1]

    @property
    def shot_matrix(self) -> np.ndarray:
        """World -> clip space of this camera, i.e. what textures the DEM."""
        return self.proj @ self.view

    def with_aspect(self, aspect: float, near: float = 0.5,
                    far: float = 20000.0) -> "FrameCamera":
        """Same pose, a projection for a viewport of another aspect ratio."""
        return FrameCamera(self.view.copy(),
                           perspective(self.fovy, aspect, near, far),
                           self.fovy, aspect)


def pose_fovy(meta: dict, default: float = DEFAULT_FOVY) -> float:
    fovy = meta.get("fovy", default)
    if isinstance(fovy, (list, tuple)):
        fovy = fovy[0] if fovy else default
    try:
        return float(fovy)
    except (TypeError, ValueError):
        return float(default)


def _vec(d: Optional[dict]) -> List[float]:
    d = d or _ZERO
    return [float(d.get("x", 0.0)), float(d.get("y", 0.0)),
            float(d.get("z", 0.0))]


def camera_for_pose(meta: dict, translation: Optional[dict],
                    rotation: Optional[dict], aspect: float = 1.0) -> FrameCamera:
    """The corrected camera of one poses-file entry.

    :param meta: one entry of ``poses_{cam}.json`` ``images``
    :param translation: correction translation ``{x, y, z}`` in metres
    :param rotation: correction rotation ``{x, y, z}`` in radians
    :param aspect: width / height of the frame
    """
    from .camera_pose import render_camera
    camera = render_camera(meta, translation or _ZERO, rotation or _ZERO, aspect)
    view = to_row(np.asarray(camera.get_view(dtype="f8"), dtype=np.float64))
    proj = to_row(np.asarray(camera.get_proj(dtype="f8"), dtype=np.float64))
    return FrameCamera(view, proj, pose_fovy(meta), float(aspect))


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------

def correction_from_config(config: dict) -> dict:
    """The dock config's correction in ``correction.json`` form."""
    return {
        "translation": dict(config.get("translation") or _ZERO),
        "rotation": dict(config.get("rotation") or _ZERO),
        "additional": [dict(a) for a in
                       config.get("additional_corrections") or []],
    }


def correction_for(frame_idx: int, correction: dict) -> Tuple[dict, dict]:
    """(translation, rotation) that applies to *frame_idx*."""
    from .corrections import correction_for_frame
    return correction_for_frame(frame_idx, correction or {})


# ---------------------------------------------------------------------------
# Frustum and rays
# ---------------------------------------------------------------------------

def camera_rays(camera: FrameCamera, ndc_xy) -> np.ndarray:
    """World-space unit directions through NDC points ``(x, y)`` in [-1, 1]."""
    ndc = np.asarray(ndc_xy, dtype=np.float64).reshape(-1, 2)
    t = math.tan(math.radians(camera.fovy) / 2.0)
    local = np.column_stack([ndc[:, 0] * t * camera.aspect,
                             ndc[:, 1] * t, -np.ones(len(ndc))])
    world = local @ camera.rotation.T
    return world / np.linalg.norm(world, axis=1, keepdims=True)


def pixel_rays(camera: FrameCamera, width: int, height: int,
               pixels) -> np.ndarray:
    """World-space directions through image pixels (x right, y down)."""
    px = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    ndc = np.column_stack([2.0 * px[:, 0] / width - 1.0,
                           1.0 - 2.0 * px[:, 1] / height])
    return camera_rays(camera, ndc)


#: Image corners in NDC, clockwise from top-left.
CORNERS_NDC = ((-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0))


def frustum_corners(camera: FrameCamera, distance: float) -> np.ndarray:
    """(4, 3) points along the corner rays at *distance* metres."""
    return camera.position + camera_rays(camera, CORNERS_NDC) * distance


def cast_rays(caster, origins, directions) -> np.ndarray:
    """First hits of rays on the DEM; NaN rows for misses.

    *caster* is an alfspy ray caster (``intersects_first``) or a trimesh mesh,
    as :func:`core.render_context.make_ray_caster` returns either.
    """
    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    if len(origins) == 1 and len(directions) > 1:
        origins = np.repeat(origins, len(directions), axis=0)
    out = np.full((len(directions), 3), np.nan)
    if caster is None or not len(directions):
        return out
    if hasattr(caster, "intersects_first"):
        hits, index = caster.intersects_first(origins, directions)
    else:
        hits, index, _ = caster.ray.intersects_location(
            origins, directions, multiple_hits=False)
    hits = np.asarray(hits, dtype=np.float64).reshape(-1, 3)
    index = np.asarray(index, dtype=int).reshape(-1)
    if len(index):
        out[index] = hits
    return out


def pick_nearest(origin, direction, casters):
    """First hit of one ray among several surfaces (e.g. DEM and the
    reconstructed surface).

    :param casters: ``[(name, caster), ...]``; ``None`` casters are skipped
    :return: ``(point, name)`` of the nearest hit, or ``(None, None)``
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(1, 3)
    direction = np.asarray(direction, dtype=np.float64).reshape(1, 3)
    best, best_name, best_d = None, None, np.inf
    for name, caster in casters:
        if caster is None:
            continue
        hit = cast_rays(caster, origin, direction)[0]
        if not np.all(np.isfinite(hit)):
            continue
        d = float(np.linalg.norm(hit - origin[0]))
        if d < best_d:
            best, best_name, best_d = hit, name, d
    return best, best_name


def frustum_ground(camera: FrameCamera, caster, samples_per_edge: int = 8,
                   far: float = 1000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Corner hits and the footprint outline of a camera on the DEM.

    :return: ``(corners (4,3), outline (4*samples_per_edge, 3))``; a ray that
        misses the DEM ends *far* metres along its direction instead.
    """
    ndc = []
    for i in range(4):
        a, b = np.array(CORNERS_NDC[i]), np.array(CORNERS_NDC[(i + 1) % 4])
        for s in range(samples_per_edge):
            ndc.append(a + (b - a) * s / samples_per_edge)
    dirs = camera_rays(camera, ndc)
    hits = cast_rays(caster, camera.position[None, :], dirs)
    miss = ~np.isfinite(hits[:, 0])
    hits[miss] = camera.position + dirs[miss] * far
    corners = hits[::samples_per_edge].copy()
    return corners, hits


# ---------------------------------------------------------------------------
# Frame windows and stepping
# ---------------------------------------------------------------------------

def alfs_window(center: int, count: int, n: int, stride: int = 1,
                exclude=None) -> List[int]:
    """The ALFS window: the frames from ``n/2`` before to ``n/2`` after
    *center*, every *stride*-th one (always including *center*), clipped to
    ``[0, count)``. Frames in *exclude* - the duplicated ones, whose image
    was not taken where their pose is - are left out, *center* included."""
    stride = max(1, int(stride))
    if count <= 0:
        return []
    center = min(max(int(center), 0), count - 1)
    k = (max(0, int(n)) // 2) // stride
    exclude = exclude or ()
    return [center + j * stride for j in range(-k, k + 1)
            if 0 <= center + j * stride < count and center + j * stride not in exclude]


def step_index(index: int, delta: int, count: int, loop: bool = False) -> int:
    """Move *index* by *delta* frames, clamped (or wrapped with *loop*)."""
    if count <= 0:
        return 0
    target = int(index) + int(delta)
    if loop:
        return target % count
    return min(max(target, 0), count - 1)


# ---------------------------------------------------------------------------
# Time line and frame matching
# ---------------------------------------------------------------------------

def timeline(images: List[dict]) -> List[Optional[float]]:
    """Seconds since the first timed frame, ``None`` where untimed."""
    from .frame_matching import pose_epochs
    epochs = pose_epochs(images)
    first = next((e for e in epochs if e is not None), None)
    return [None if e is None or first is None else e - first for e in epochs]


def playback_delays(times: Sequence[Optional[float]], fallback_fps: float,
                    speed: float = 1.0, minimum: float = 0.005) -> List[float]:
    """Seconds to wait after each frame for real-time playback at *speed*."""
    speed = max(speed, 1e-6)
    default = 1.0 / max(fallback_fps, 1e-6)
    out = []
    for a, b in zip(times, list(times[1:]) + [None]):
        if a is None or b is None or b <= a:
            out.append(default / speed)
        else:
            out.append(max(minimum, (b - a) / speed))
    return out


def matching_frame(src_images: List[dict], dst_images: List[dict],
                   src_index: int) -> int:
    """Index on the other camera taken at the same moment (or the same
    index, clamped, when the poses carry no timestamps)."""
    if not dst_images:
        return 0
    from .frame_matching import FrameMatcher
    matcher = FrameMatcher(src_images, dst_images)
    if matcher.usable:
        found = matcher.match(src_index)
        if found is not None:
            return int(found[0])
    return min(max(int(src_index), 0), len(dst_images) - 1)


# ---------------------------------------------------------------------------
# Undistortion masks
# ---------------------------------------------------------------------------

def find_mask(target_folder: str, modality: str) -> str:
    """The undistortion mask of a camera (white = valid pixel), or ``""``.

    Same lookup as the ALFS step: ``mask_T.png`` / ``mask_W.png`` in the
    target folder, else the file the poses name under ``mask``.
    """
    import json
    specific = os.path.join(target_folder, f"mask_{modality.upper()}.png")
    if os.path.isfile(specific):
        return specific
    try:
        with open(os.path.join(target_folder, f"poses_{modality}.json"),
                  "r", encoding="utf-8") as fh:
            name = json.load(fh).get("mask")
    except (OSError, ValueError, AttributeError):
        name = None
    if name:
        path = os.path.join(target_folder, name)
        if os.path.isfile(path):
            return path
    return ""


def resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour resize (keeps a binary mask binary)."""
    img = np.asarray(image)
    rows = (np.arange(height) * img.shape[0] / float(height)).astype(int)
    cols = (np.arange(width) * img.shape[1] / float(width)).astype(int)
    return img[rows][:, cols]


def mask_values(mask: np.ndarray) -> np.ndarray:
    """A mask image as weights 0..1 (first channel; alpha if it has one and
    the colour is uniform)."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = m.astype(np.float32)
    return m / 255.0 if m.max() > 1.0 else m


def erode_mask(mask, px: int) -> np.ndarray:
    """Shrink the valid (True) area of a boolean mask by *px* pixels
    (square neighbourhood). Pixels next to the undistortion border are a
    blend of image and black border once sampled; eroding drops that rim -
    the same idea as the GeoTIFF export's edge erosion."""
    m = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(px))):
        p = np.pad(m, 1, constant_values=False)
        m = (p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2]
             & p[1:-1, 2:] & p[:-2, :-2] & p[:-2, 2:] & p[2:, :-2] & p[2:, 2:])  # noqa: W503
    return m


def mask_for(mask_image, width: int, height: int, erode_px: int = 0) -> np.ndarray:
    """A mask image as a boolean mask of ``width`` x ``height``, eroded by
    *erode_px* pixels at that size."""
    valid = mask_values(resize_nearest(mask_image, width, height)) > 0.5
    return erode_mask(valid, erode_px) if erode_px > 0 else valid


def scaled_erosion_mask(mask_image, erode_px: int, image_max_edge: int) -> np.ndarray:
    """The mask at its own resolution as uint8 0/255, eroded by *erode_px*
    pixels of an image whose longest edge is *image_max_edge* (frames are
    shrunk before use, so the erosion is scaled up to the mask's size)."""
    m = np.asarray(mask_image)
    h, w = m.shape[:2]
    scale = max(1.0, max(w, h) / float(max(image_max_edge, 1)))
    px = int(math.ceil(erode_px * scale)) if erode_px > 0 else 0
    return (mask_for(m, w, h, px) * 255).astype(np.uint8)


def apply_mask(rgba: np.ndarray, mask: np.ndarray, erode_px: int = 0) -> np.ndarray:
    """*rgba* with masked-out pixels made transparent (mask resized to fit,
    then eroded by *erode_px* pixels of *rgba*)."""
    out = np.array(rgba, dtype=np.uint8, copy=True)
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)
    if out.shape[2] == 3:
        out = np.concatenate([out, np.full(out.shape[:2] + (1,), 255, np.uint8)], 2)
    valid = mask_for(mask, out.shape[1], out.shape[0], erode_px)
    out[..., 3] = np.where(valid, out[..., 3], 0).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def boxes_on_frame(detections: Iterable[dict], frame: int) -> List[dict]:
    """The detections (pixel boxes) of one frame."""
    return [d for d in detections if int(d.get("frame", -1)) == int(frame)]


@dataclass
class TrackState:
    track_id: int
    history: List[dict]            # boxes up to and including the frame
    current: Optional[dict]        # the box on the frame itself, if any


def tracks_at(tracks: Dict[int, List[dict]], frame: int,
              history_frames: Optional[int] = None,
              linger_frames: int = 0) -> List[TrackState]:
    """Track histories as of *frame*.

    :param tracks: ``track_id -> [box with "frame", …]``
    :param history_frames: keep only the last N frames of history
        (``None`` = all)
    :param linger_frames: keep a track whose last box is at most this many
        frames old (shown faded); ``0`` = only tracks seen on *frame*
    """
    states = []
    for track_id, boxes in tracks.items():
        past = sorted((b for b in boxes if int(b["frame"]) <= frame),
                      key=lambda b: int(b["frame"]))
        if not past:
            continue
        last = int(past[-1]["frame"])
        if frame - last > linger_frames:
            continue
        if history_frames is not None:
            past = [b for b in past if int(b["frame"]) > frame - history_frames]
        current = past[-1] if past and int(past[-1]["frame"]) == frame else None
        states.append(TrackState(int(track_id), past, current))
    return states


def box_centers(boxes: Iterable[dict]) -> np.ndarray:
    return np.array([[(b["x1"] + b["x2"]) / 2.0, (b["y1"] + b["y2"]) / 2.0]
                     for b in boxes], dtype=np.float64).reshape(-1, 2)


# ---------------------------------------------------------------------------
# DEM
# ---------------------------------------------------------------------------

@dataclass
class DemData:
    vertices: np.ndarray                  # (N, 3) float32, DEM-local metres
    faces: np.ndarray                     # (M, 3) uint32
    normals: np.ndarray                   # (N, 3) float32
    uvs: Optional[np.ndarray] = None      # (N, 2) float32
    texture: Optional[np.ndarray] = None  # (H, W, 3|4) uint8, RGB(A)
    origin: Optional[Tuple[float, float, float]] = None

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def center(self) -> np.ndarray:
        lo, hi = self.bounds
        return (lo + hi) / 2.0


def vertex_normals(vertices, faces) -> np.ndarray:
    """Area-weighted vertex normals."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(length > 0, n / np.where(length > 0, length, 1.0),
                 np.array([0.0, 0.0, 1.0]))
    # A heightfield faces up; flip meshes wound the other way.
    if n[:, 2].mean() < 0:
        n = -n
    return n.astype(np.float32)


def load_dem(mesh_path: str, origin=None) -> DemData:
    """Read a DEM ``.glb`` / ``.gltf`` into arrays the viewer can upload."""
    try:
        from alfspy.render.render import read_gltf
    except ImportError as exc:
        raise RuntimeError(f"alfspy not available: {exc}") from exc
    mesh_data, texture_data = read_gltf(mesh_path)
    vertices = np.asarray(mesh_data.vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(mesh_data.indices, dtype=np.uint32).reshape(-1, 3)
    uvs = None
    if getattr(mesh_data, "uvs", None) is not None and len(mesh_data.uvs):
        uvs = np.asarray(mesh_data.uvs, dtype=np.float32).reshape(-1, 2)
    texture = None
    tex = getattr(texture_data, "texture", None) if texture_data else None
    if tex is not None and np.size(tex):
        tex = np.asarray(tex)
        if tex.dtype != np.uint8:
            tex = np.clip(tex * (255.0 if tex.max() <= 1.0 else 1.0),
                          0, 255).astype(np.uint8)
        if tex.ndim == 3 and tex.shape[2] >= 3:   # alfspy textures are BGR(A)
            tex = tex[..., [2, 1, 0] + ([3] if tex.shape[2] == 4 else [])]
        texture = tex
    return DemData(vertices, faces, vertex_normals(vertices, faces), uvs,
                   texture, origin)


def make_caster(dem: DemData):
    """A ray caster over the DEM (alfspy's, or a trimesh fallback)."""
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(f"trimesh not available: {exc}") from exc
    from .render_context import make_ray_caster
    mesh = trimesh.Trimesh(vertices=dem.vertices, faces=dem.faces,
                           process=False)
    return make_ray_caster(mesh)


#: DEM detail levels of the 3D viewer: label -> share of the triangles kept.
DEM_DETAIL = (("High (full mesh)", 1.0), ("Medium (66 %)", 0.66),
              ("Low (33 %)", 0.33))


@dataclass
class DemSimplification:
    """A reduced DEM for display and its height error against the full one."""
    dem: DemData
    fraction: float                 # triangles kept / triangles of the full DEM
    max_error: float                # metres, at the full DEM's vertices
    rms_error: float


def simplify_dem(dem: DemData, fraction: float, caster=None,
                 max_samples: int = 200_000) -> DemSimplification:
    """Resample the DEM on a regular grid with about *fraction* of its
    triangles - for drawing only; ray casting keeps the full mesh.

    Heights come from the full DEM (straight down onto it), so the reduced
    surface lies on the original; nodes off the DEM are left out. Texture
    coordinates are carried over when the original maps them linearly from
    x / y (as BAMBI's DEMs do). The error is measured vertically at (up to
    *max_samples* of) the full DEM's vertices.
    """
    from .flight_reconstruction import heightfield_mesh

    if fraction >= 0.999:
        return DemSimplification(dem, 1.0, 0.0, 0.0)
    caster = caster if caster is not None else make_caster(dem)
    lo, hi = dem.bounds
    size = np.maximum(hi[:2] - lo[:2], 1e-6)
    cells = max(1.0, len(dem.faces) * float(fraction) / 2.0)
    step = math.sqrt(size[0] * size[1] / cells)
    nx = max(2, int(round(size[0] / step)) + 1)
    ny = max(2, int(round(size[1] / step)) + 1)
    # Nodes on the very edge of the DEM can miss it in the ray cast; keep
    # them a hair inside so the reduced DEM covers the same extent.
    eps = 1e-4 * step
    xs = np.linspace(lo[0] + eps, hi[0] - eps, nx)
    ys = np.linspace(hi[1] - eps, lo[1] + eps, ny)   # row 0 = north
    X, Y = np.meshgrid(xs, ys)
    top = float(hi[2]) + 100.0
    hits = cast_rays(caster, np.column_stack([X.ravel(), Y.ravel(),
                                              np.full(X.size, top)]),
                     np.tile([0.0, 0.0, -1.0], (X.size, 1)))
    Z = hits[:, 2].reshape(X.shape)

    class _Grid:                                     # heightfield_mesh's view
        def xy(self):
            return X, Y
    verts, faces, normals = heightfield_mesh(_Grid(), Z)

    uvs = None
    if dem.uvs is not None and len(dem.uvs) == len(dem.vertices):
        A = np.column_stack([dem.vertices[:, 0], dem.vertices[:, 1],
                             np.ones(len(dem.vertices))]).astype(np.float64)
        coef, *_ = np.linalg.lstsq(A, dem.uvs.astype(np.float64), rcond=None)
        if np.abs(A @ coef - dem.uvs).max() < 1e-3:
            uvs = (np.column_stack([verts[:, 0], verts[:, 1],
                                    np.ones(len(verts))]) @ coef).astype(np.float32)
    texture = dem.texture if uvs is not None else None
    reduced = DemData(verts, faces, normals, uvs, texture, dem.origin)

    # Vertical error at the full DEM's vertices.
    src = dem.vertices
    if len(src) > max_samples:
        src = src[np.random.default_rng(0).choice(len(src), max_samples,
                                                  replace=False)]
    import trimesh
    small = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    from .render_context import make_ray_caster
    got = cast_rays(make_ray_caster(small),
                    np.column_stack([src[:, 0], src[:, 1],
                                     np.full(len(src), top)]),
                    np.tile([0.0, 0.0, -1.0], (len(src), 1)))[:, 2]
    diff = np.abs(got - src[:, 2])
    diff = diff[np.isfinite(diff)]
    max_err = float(diff.max()) if len(diff) else 0.0
    rms = float(np.sqrt(np.mean(diff ** 2))) if len(diff) else 0.0
    kept = len(faces) / max(len(dem.faces), 1)
    return DemSimplification(reduced, kept, max_err, rms)


# ---------------------------------------------------------------------------
# The scene
# ---------------------------------------------------------------------------

@dataclass
class FlightScene:
    """One camera of one flight, with corrected cameras on demand."""
    target_folder: str
    modality: str
    images: List[dict]
    correction: dict = field(default_factory=dict)
    aspect: float = 1.0
    image_size: Optional[Tuple[int, int]] = None   # (width, height)
    _cameras: Dict[int, FrameCamera] = field(default_factory=dict, repr=False)
    _times: Optional[List[Optional[float]]] = field(default=None, repr=False)

    @classmethod
    def load(cls, target_folder: str, modality: str,
             correction: Optional[dict] = None,
             image_size: Optional[Tuple[int, int]] = None) -> "FlightScene":
        from .inspection import _pose_images
        images = list(_pose_images(target_folder, modality))
        aspect = 1.0
        if image_size and image_size[1]:
            aspect = float(image_size[0]) / float(image_size[1])
        return cls(target_folder, modality, images, correction or {},
                   aspect, image_size)

    def __len__(self) -> int:
        return len(self.images)

    # -- duplicated (frozen thermal) frames, core.poses ----------------------

    def duplicated(self) -> set:
        """Indices of the frames whose image repeats the previous one.
        Their pose is valid, so they stay on the route; whatever uses the
        image (trail, ALFS, compare, textures) leaves them out."""
        cached = getattr(self, "_duplicated", None)
        if cached is None:
            from . import poses
            cached = poses.duplicated_frames(self.images)
            self._duplicated = cached
        return cached

    def is_duplicated(self, index: int) -> bool:
        return index in self.duplicated()

    def usable_frame(self, index: int) -> int:
        """*index*, or the nearest frame that is not duplicated."""
        from . import poses
        found = poses.nearest_usable(self.images, index)
        return index if found is None else found

    # -- frames ------------------------------------------------------------

    def frame_path(self, index: int) -> str:
        if not 0 <= index < len(self.images):
            return ""
        name = self.images[index].get("imagefile", "")
        return os.path.join(self.target_folder, f"frames_{self.modality}",
                            name) if name else ""

    def times(self) -> List[Optional[float]]:
        if self._times is None:
            self._times = timeline(self.images)
        return self._times

    # -- cameras -----------------------------------------------------------

    def set_correction(self, correction: dict) -> None:
        self.correction = correction or {}
        self._cameras.clear()

    def camera(self, index: int) -> FrameCamera:
        cam = self._cameras.get(index)
        if cam is None:
            t, r = correction_for(index, self.correction)
            cam = camera_for_pose(self.images[index], t, r, self.aspect)
            self._cameras[index] = cam
        return cam

    def positions(self) -> np.ndarray:
        """(N, 3) corrected camera positions of every frame."""
        return np.array([self.camera(i).position
                         for i in range(len(self.images))]).reshape(-1, 3)

    def raw_positions(self) -> np.ndarray:
        """(N, 3) uncorrected pose locations."""
        return np.array([m.get("location", [0, 0, 0])[:3]
                         for m in self.images], dtype=np.float64).reshape(-1, 3)


# ---------------------------------------------------------------------------
# alfspy interop (ALFS rendering of viewer cameras)
# ---------------------------------------------------------------------------

def alfs_pose(camera: FrameCamera):
    """``(Vector3 position, Quaternion rotation)`` for an alfspy ``Camera`` or
    ``CtxShot`` that reproduces *camera*'s view without any correction - the
    correction is already part of *camera*."""
    try:
        from pyrr import Matrix44, Quaternion, Vector3
    except ImportError as exc:
        raise RuntimeError(f"pyrr not available: {exc}") from exc
    to_world_row = to_row(camera.to_world)       # pyrr row-vector layout
    rot = np.eye(4)
    rot[:3, :3] = to_world_row[:3, :3]
    quat = Quaternion.from_matrix(Matrix44(rot.astype("f4")))
    return Vector3(camera.position.astype("f4")), quat


def alfs_camera(camera: FrameCamera, near: float = 0.5, far: float = 20000.0):
    """An alfspy perspective ``Camera`` equal to *camera*."""
    from alfspy.core.rendering import Camera
    position, rotation = alfs_pose(camera)
    return Camera(camera.fovy, camera.aspect, position=position,
                  rotation=rotation, near=near, far=far)


def alfs_ortho_camera(lower, upper, top: float, bottom: float):
    """An alfspy orthographic camera looking straight down on an XY extent
    (image row 0 = north edge)."""
    from pyrr import Quaternion, Vector3
    from alfspy.core.rendering import Camera
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    size = upper[:2] - lower[:2]
    center = (lower[:2] + upper[:2]) / 2.0
    height = float(top) + 10.0
    return Camera(orthogonal=True, orthogonal_size=(float(size[0]), float(size[1])),
                  position=Vector3([center[0], center[1], height]),
                  rotation=Quaternion(), near=1.0,
                  far=max(10.0, height - float(bottom) + 10.0))


def window_extent(cameras: Iterable[FrameCamera], caster,
                  pad_fraction: float = 0.05, max_distance: float = 2000.0):
    """XY bounds covering the ground footprints of *cameras*.

    :return: ``(lower (2,), upper (2,))`` or ``None`` when nothing was hit
    """
    pts = []
    for cam in cameras:
        _, outline = frustum_ground(cam, caster, samples_per_edge=4,
                                    far=max_distance)
        hit = outline[np.linalg.norm(outline - cam.position, axis=1) < max_distance - 1e-6]
        if len(hit):
            pts.append(hit[:, :2])
    if not pts:
        return None
    pts = np.vstack(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = (hi - lo) * pad_fraction + 1.0
    return lo - pad, hi + pad


def integral_to_rgba(accum, coverage, min_coverage: float = 1.0) -> np.ndarray:
    """Normalise an ALFS ``(accum, coverage)`` pair to RGBA ``uint8``.

    Pixels seen by fewer than *min_coverage* frames are transparent
    (minimum number of frames per pixel).
    """
    accum = np.asarray(accum, dtype=np.float32)
    cov = np.asarray(coverage, dtype=np.float32)
    rgb = accum[..., :3] / np.maximum(cov[..., None], 1e-6)
    if rgb.size and np.nanmax(rgb) <= 1.01:
        rgb = rgb * 255.0
    out = np.zeros(cov.shape + (4,), dtype=np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[..., 3] = np.where(cov >= max(min_coverage, 1e-3) - 1e-6, 255, 0)
    return out
