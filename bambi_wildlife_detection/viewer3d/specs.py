# -*- coding: utf-8 -*-
"""Plain-data scene description of the 3D viewer (no OpenGL import)."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Layer:
    """A texture draped onto the DEM through *matrix* (world -> clip)."""
    texture_key: object
    matrix: np.ndarray
    alpha: float = 1.0
    z_offset: float = 0.0
    min_alpha: float = 0.004


@dataclass
class LineSpec:
    segments: np.ndarray      # (N, 2, 3)
    colors: np.ndarray        # (N, 2, 4) floats 0..1
    width: float = 2.0
    on_top: bool = False      # ignore depth (e.g. the current frustum)


@dataclass
class PointSpec:
    points: np.ndarray        # (N, 3)
    colors: np.ndarray        # (N, 4)
    size: float = 4.0
    round: bool = True


@dataclass
class SurfaceSpec:
    vertices: np.ndarray      # (N, 3)
    faces: np.ndarray         # (M, 3)
    normals: np.ndarray       # (N, 3)
    colors: np.ndarray        # (N, 4)
    shade: float = 0.5
    texture: Optional[np.ndarray] = None   # RGB(A) image draped from above
    texture_matrix: Optional[np.ndarray] = None   # world -> clip of the image
