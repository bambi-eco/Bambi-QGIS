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
