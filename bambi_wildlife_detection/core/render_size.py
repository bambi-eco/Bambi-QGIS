# -*- coding: utf-8 -*-
"""Raster size for the ALFS / GeoTIFF renders.

Two ways to say how big a render should be, selected by the "Fixed render
size" checkbox in the ALFS/GeoTiff configuration:

``ground resolution`` (the historic mode)
    The user gives metres per pixel and the pixel count follows from the
    extent being covered. Scale is fixed, output size varies with the area.

``fixed render size``
    The user gives the pixel count directly and the ground resolution
    follows from the extent. Output size is fixed, scale varies with the
    area — useful when every render has to land on the same canvas (fixed
    per-frame GeoTIFFs, sampling-mode tiles of comparable weight) and when
    the DEM extent would otherwise make a fine ground resolution explode
    into a raster of unusable size.

Nothing here touches Qt or QGIS, so the sizing rules are unit-testable on
their own.
"""

import math

DEFAULT_RENDER_WIDTH = 2048
DEFAULT_RENDER_HEIGHT = 2048


def uses_fixed_size(config):
    """True when the render size is given in pixels rather than in m/px."""
    return bool(config.get("alfs_fixed_render_size", False))


def resolve_render_size(width_meters, height_meters, ground_resolution,
                        config, max_dim=None):
    """Return the ``(width_px, height_px)`` for an extent of *width_meters*
    by *height_meters*.

    In ground-resolution mode the extent is divided by *ground_resolution*
    (rounded up, as the callers have always done). In fixed-size mode the
    extent is ignored and the configured pixel counts are returned — the two
    axes are independent, so a render whose extent is not the same shape as
    the requested canvas simply gets non-square pixels. That is fine for
    every consumer here: the GeoTIFF transform carries a separate x and y
    pixel size, and the orthographic render camera is sized from the extent,
    not from the pixel grid.

    *max_dim* caps the longer side (preserving the aspect of the resolved
    size), matching the memory limits the individual callers already had.
    """
    if uses_fixed_size(config):
        width_px = int(config.get("alfs_render_width") or DEFAULT_RENDER_WIDTH)
        height_px = int(config.get("alfs_render_height") or DEFAULT_RENDER_HEIGHT)
    else:
        if not ground_resolution or ground_resolution <= 0:
            raise ValueError(
                f"Ground resolution must be positive, got {ground_resolution!r}")
        width_px = int(math.ceil(width_meters / ground_resolution))
        height_px = int(math.ceil(height_meters / ground_resolution))

    width_px = max(1, width_px)
    height_px = max(1, height_px)

    if max_dim and (width_px > max_dim or height_px > max_dim):
        scale = max_dim / float(max(width_px, height_px))
        width_px = max(1, int(width_px * scale))
        height_px = max(1, int(height_px * scale))

    return width_px, height_px


def describe(config, ground_resolution):
    """One log line stating which mode is in force, for the run log."""
    if uses_fixed_size(config):
        width_px = int(config.get("alfs_render_width") or DEFAULT_RENDER_WIDTH)
        height_px = int(config.get("alfs_render_height") or DEFAULT_RENDER_HEIGHT)
        return (f"Render size: fixed {width_px} x {height_px} px "
                f"(ground resolution derived from the extent)")
    return f"Ground resolution: {ground_resolution} m/px"
