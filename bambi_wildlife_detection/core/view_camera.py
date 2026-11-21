# -*- coding: utf-8 -*-
"""Orbit camera of the 3D Flight Viewer (pure NumPy, column-vector matrices).

Z is up, like the DEM. The camera orbits a target point: azimuth is measured
counter-clockwise from +x, elevation upwards from the horizontal plane.
"""

import math
from typing import Optional, Sequence

import numpy as np

from .flight_viewer import FrameCamera, look_at, perspective

MIN_ELEVATION = -89.0
MAX_ELEVATION = 89.0


class OrbitCamera:
    def __init__(self, target: Sequence[float] = (0.0, 0.0, 0.0),
                 distance: float = 500.0, azimuth: float = -90.0,
                 elevation: float = 45.0, fovy: float = 45.0):
        self.target = np.asarray(target, dtype=np.float64).copy()
        self.distance = float(distance)
        self.azimuth = float(azimuth)
        self.elevation = float(elevation)
        self.fovy = float(fovy)
        self.min_distance = 1.0
        self.max_distance = 1.0e6

    # -- geometry ----------------------------------------------------------

    def direction(self) -> np.ndarray:
        """Unit vector from the target to the eye."""
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        return np.array([math.cos(el) * math.cos(az),
                         math.cos(el) * math.sin(az),
                         math.sin(el)])

    def eye(self) -> np.ndarray:
        return self.target + self.direction() * self.distance

    def view(self) -> np.ndarray:
        return look_at(self.eye(), self.target, (0.0, 0.0, 1.0))

    def clip_planes(self, scene_radius: float = 0.0):
        near = max(0.05, self.distance * 1e-3)
        far = max(self.distance * 4.0, self.distance + 4.0 * scene_radius,
                  near * 10.0)
        return near, far

    def proj(self, aspect: float, scene_radius: float = 0.0) -> np.ndarray:
        near, far = self.clip_planes(scene_radius)
        return perspective(self.fovy, aspect, near, far)

    def basis(self):
        """(right, up) unit vectors of the screen in world space."""
        view = self.view()
        return view[0, :3].copy(), view[1, :3].copy()

    # -- interaction -------------------------------------------------------

    def orbit(self, d_azimuth: float, d_elevation: float) -> None:
        self.azimuth = (self.azimuth + d_azimuth) % 360.0
        self.elevation = min(MAX_ELEVATION, max(MIN_ELEVATION,
                                                self.elevation + d_elevation))

    def pan(self, dx_px: float, dy_px: float, viewport_height: int) -> None:
        """Move the target with the mouse: *dx/dy* in pixels."""
        right, up = self.basis()
        scale = (2.0 * self.distance * math.tan(math.radians(self.fovy) / 2.0) / max(viewport_height, 1))
        self.target = self.target - right * dx_px * scale + up * dy_px * scale

    def zoom(self, factor: float, towards: Optional[Sequence[float]] = None) -> None:
        """Scale the distance by *factor*; with *towards*, zoom to that point."""
        new = min(self.max_distance, max(self.min_distance,
                                         self.distance * factor))
        if towards is not None:
            towards = np.asarray(towards, dtype=np.float64)
            ratio = new / self.distance if self.distance else 1.0
            self.target = towards + (self.target - towards) * ratio
        self.distance = new

    def fit(self, lower: Sequence[float], upper: Sequence[float]) -> None:
        """Frame an axis-aligned box from the current direction."""
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        self.target = (lower + upper) / 2.0
        radius = float(np.linalg.norm(upper - lower)) / 2.0
        self.distance = max(self.min_distance,
                            radius / math.sin(math.radians(self.fovy) / 2.0))

    def adopt(self, camera: FrameCamera, target_distance: float) -> None:
        """Continue orbiting from where *camera* (a drone camera) looks."""
        forward = camera.forward / max(np.linalg.norm(camera.forward), 1e-12)
        self.distance = max(self.min_distance, float(target_distance))
        self.target = camera.position + forward * self.distance
        back = -forward
        self.elevation = min(MAX_ELEVATION, max(
            MIN_ELEVATION, math.degrees(math.asin(max(-1.0, min(1.0, back[2]))))))
        self.azimuth = math.degrees(math.atan2(back[1], back[0])) % 360.0

    def ray(self, x_px: float, y_px: float, width: int, height: int):
        """World ray (origin, unit direction) through a viewport pixel."""
        aspect = width / max(height, 1)
        t = math.tan(math.radians(self.fovy) / 2.0)
        nx = 2.0 * x_px / max(width, 1) - 1.0
        ny = 1.0 - 2.0 * y_px / max(height, 1)
        inv = np.linalg.inv(self.view())
        d = inv[:3, :3] @ np.array([nx * t * aspect, ny * t, -1.0])
        return inv[:3, 3].copy(), d / np.linalg.norm(d)


# ---------------------------------------------------------------------------
# Which way the view looks
# ---------------------------------------------------------------------------

COMPASS_POINTS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def view_bearing(view: np.ndarray) -> float:
    """The direction a view looks in, as a bearing on the map: degrees
    clockwise from north (+y), 0 to 360.

    A view that looks down at a slant looks towards where its horizontal
    part points; one that looks straight down has no such part, and "the
    direction it looks in" is what is up on the screen. The two agree
    wherever both exist, so they are added: the bearing is that of the
    horizontal part of *forward + up*.

    :param view: a 4x4 view matrix as :func:`core.flight_viewer.look_at`
        makes it (rows: right, up, backward)
    """
    view = np.asarray(view, dtype=np.float64)
    ahead = -view[2, :2] + view[1, :2]
    if not np.all(np.isfinite(ahead)) or float(np.hypot(*ahead)) < 1e-9:
        return 0.0
    bearing = math.degrees(math.atan2(ahead[0], ahead[1])) % 360.0
    # a hair west of north is north, not 360
    return 0.0 if bearing > 360.0 - 1e-9 else bearing


def compass_point(bearing: float) -> str:
    """``"NE"`` for 47 degrees: the nearest of the eight points."""
    return COMPASS_POINTS[int(((float(bearing) % 360.0) + 22.5) // 45.0) % 8]


# ---------------------------------------------------------------------------
# How large what is seen is
# ---------------------------------------------------------------------------

def metres_per_pixel(distance: float, fovy: float, height_px: float) -> float:
    """How much of the scene one pixel of the screen covers, at *distance*
    in front of a camera with a vertical field of view of *fovy* degrees
    in a view *height_px* high - metres, as the terrain model's own
    coordinates are. ``0.0`` where there is no saying."""
    values = (float(distance), float(fovy), float(height_px))
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        return 0.0
    if values[1] >= 180.0:
        return 0.0
    return 2.0 * values[0] * math.tan(math.radians(values[1]) / 2.0) / values[2]


def scale_length(per_pixel: float, widest_px: float) -> float:
    """The length a scale bar stands for: the largest round one - 1, 2 or
    5 times a power of ten - that is no wider than *widest_px* on the
    screen. ``0.0`` where there is no scale."""
    per_pixel, widest_px = float(per_pixel), float(widest_px)
    if not (math.isfinite(per_pixel) and per_pixel > 0.0 and widest_px > 0.0):
        return 0.0
    most = per_pixel * widest_px
    power = math.floor(math.log10(most))
    for exponent in (power, power - 1):
        for step in (5.0, 2.0, 1.0):
            length = step * 10.0 ** exponent
            if length <= most * (1.0 + 1e-9):
                return length
    return 0.0


def scale_text(length: float) -> str:
    """``"50 m"``, ``"2 km"``, ``"20 cm"``: a scale bar's length, said in
    the unit that needs no decimals."""
    length = float(length)
    if length >= 1000.0:
        return f"{length / 1000.0:g} km"
    if length >= 1.0:
        return f"{length:g} m"
    if length >= 0.01:
        return f"{length * 100.0:g} cm"
    return f"{length * 1000.0:g} mm"
