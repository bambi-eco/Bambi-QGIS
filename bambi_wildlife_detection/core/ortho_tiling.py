# -*- coding: utf-8 -*-
"""Pure geometry helpers of the ALFS / orthomosaic stages.

Moved from ``BambiProcessor`` (which delegates here). These sub-steps were
buried inside the big ``run_alfs`` / ``run_orthomosaic`` methods; extracting
them makes the tile arithmetic, shot culling, content cropping and
averaging merge independently testable. Heavy dependencies (the alfspy
``Camera``/``Vector3`` classes, ``rasterio.merge.merge``) are passed in by
the caller, so this module imports only numpy.
"""

import math

import numpy as np


def create_tile_camera(global_camera, global_bounds, global_res,
                       tx, ty, tw, th, Vector3, Camera):
    """Create an orthographic sub-camera covering one tile of the mosaic.

    *tx*/*ty* are the tile's top-left pixel offset in the global image and
    *tw*/*th* its pixel size; *global_bounds* is ``(min_x, min_y, max_x,
    max_y)`` in world coordinates. Returns a new ``Camera`` positioned over
    the tile centre with the tile's world extent as its orthographic size.
    """
    min_x, min_y, max_x, max_y = global_bounds

    pixel_to_world_x = (max_x - min_x) / global_res.width
    pixel_to_world_y = (max_y - min_y) / global_res.height

    # Tile bounds
    tile_min_x = min_x + tx * pixel_to_world_x
    tile_max_x = min_x + (tx + tw) * pixel_to_world_x
    tile_max_y = max_y - ty * pixel_to_world_y
    tile_min_y = max_y - (ty + th) * pixel_to_world_y

    tile_center_x = (tile_min_x + tile_max_x) / 2.0
    tile_center_y = (tile_min_y + tile_max_y) / 2.0
    tile_ortho_width = tile_max_x - tile_min_x
    tile_ortho_height = tile_max_y - tile_min_y

    return Camera(
        orthogonal=True,
        orthogonal_size=(tile_ortho_width, tile_ortho_height),
        position=Vector3([
            tile_center_x, tile_center_y, global_camera.transform.position.z
        ], dtype='f4'),
        rotation=global_camera.transform.rotation,
        near=global_camera.near,
        far=global_camera.far
    )


def filter_shots_for_tile(shots, tile_geo_bounds, log_fn=None, fov_default=50.0):
    """Return the subset of shots whose ground footprint overlaps a tile.

    Uses circle-AABB intersection: the camera footprint is approximated as a
    circle centred on the ground projection of the shot, with radius derived
    from altitude and vertical FoV.  A generous 2.5× multiplier accounts for
    oblique cameras and imprecise altitude values.

    Falls back to the full shot list when no shot passes the filter (so tiles
    on the boundary of the flight area still receive at least one shot).
    """
    min_x, min_y, max_x, max_y = tile_geo_bounds
    relevant = []
    for shot in shots:
        pos = shot.camera.transform.position
        sx, sy, sz = float(pos.x), float(pos.y), float(pos.z)
        altitude = abs(sz)
        if altitude < 1.0:
            # No meaningful altitude — include unconditionally
            relevant.append(shot)
            continue
        try:
            fov_y = shot.camera.fovy or fov_default
        except Exception:
            fov_y = fov_default
        radius = altitude * math.tan(math.radians(fov_y / 2)) * 2.5
        # Nearest point on the AABB to the shot position
        near_x = max(min_x, min(sx, max_x))
        near_y = max(min_y, min(sy, max_y))
        dist_sq = (sx - near_x) ** 2 + (sy - near_y) ** 2
        if dist_sq <= radius * radius:
            relevant.append(shot)
    if not relevant:
        return shots
    if log_fn and len(relevant) < len(shots):
        log_fn(f"  Shot filter: {len(relevant)}/{len(shots)} shots relevant for this tile")
    return relevant


def crop_to_content(image, bounds):
    """Crop an RGBA image to the minimal box containing non-empty pixels.

    Returns ``(cropped_image, new_bounds)``; when the image is entirely
    transparent the input is returned unchanged.
    """
    min_x, min_y, max_x, max_y = bounds
    height, width = image.shape[:2]

    alpha = image[:, :, 3]
    non_empty = alpha > 0

    if not np.any(non_empty):
        return image, bounds

    rows = np.any(non_empty, axis=1)
    cols = np.any(non_empty, axis=0)

    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    cropped = image[row_min:row_max + 1, col_min:col_max + 1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    new_min_x = min_x + col_min * pixel_size_x
    new_max_x = min_x + (col_max + 1) * pixel_size_x
    new_max_y = max_y - row_min * pixel_size_y
    new_min_y = max_y - (row_max + 1) * pixel_size_y

    return cropped, (new_min_x, new_min_y, new_max_x, new_max_y)


def erode_valid_mask(valid_mask, erosion_px):
    """Shrink a boolean validity mask inward by *erosion_px* pixels.

    Each frame rendered as an orthophoto has an antialiased/interpolated
    fringe one or two pixels wide at the footprint border, where the alpha
    channel is non-zero but the RGB has bled towards black. Left in place
    that dark ring shows up as a seam wherever frames overlap in the
    orthomosaic. Eroding the mask before it becomes the GeoTIFF nodata mask
    drops the ring to nodata so neighbouring frames cover it instead.

    Implemented as *erosion_px* successive 4-neighbour erosions, i.e. a
    diamond (city-block) structuring element of radius *erosion_px*; for the
    1-2 px rim this targets it is effectively identical to a box kernel.
    ``erosion_px <= 0`` or an all-empty mask returns the input unchanged.

    :param valid_mask: 2-D boolean ndarray (True = valid pixel).
    :param erosion_px: number of pixels to erode from the border.
    :returns: eroded boolean ndarray of the same shape.
    """
    if erosion_px <= 0 or not np.any(valid_mask):
        return valid_mask
    m = valid_mask.astype(bool)
    # Border pixels (outside the array) count as invalid, so shifts that fall
    # off the edge erode inward there too. Iterate 4-neighbour erosions
    # erosion_px times → equivalent to a (2r+1) square structuring element.
    for _ in range(int(erosion_px)):
        eroded = m.copy()
        eroded[1:, :] &= m[:-1, :]
        eroded[:-1, :] &= m[1:, :]
        eroded[:, 1:] &= m[:, :-1]
        eroded[:, :-1] &= m[:, 1:]
        # Also erode along the array boundary itself.
        eroded[0, :] = False
        eroded[-1, :] = False
        eroded[:, 0] = False
        eroded[:, -1] = False
        m = eroded
    return m


def merge_orthomosaic_average(datasets, nodata, rio_merge):
    """Merge datasets by averaging overlapping valid pixels.

    ``rasterio.merge`` has no built-in "average" method, so we accumulate a
    per-pixel sum (in a float64 buffer to avoid uint8 overflow) together with
    a per-pixel count of valid contributors, then divide.

    A cheap first pass with the built-in ``first`` method establishes the
    exact output grid (shape + transform); the averaging pass then reuses the
    same inputs/parameters, so rasterio produces an identically aligned grid
    and the per-source ``roff``/``coff`` offsets index our count array
    correctly.

    :returns: (mosaic ndarray of the sources' dtype, output affine transform)
    """
    src_dtype = datasets[0].dtypes[0]

    # First pass: determine the exact output grid rasterio will use.
    base, out_transform = rio_merge(datasets, method="first", nodata=nodata)
    out_height, out_width = base.shape[1], base.shape[2]
    del base

    count = np.zeros((out_height, out_width), dtype=np.float64)

    def _sum_valid(merged_data, new_data, merged_mask, new_mask,
                   index=None, roff=0, coff=0, **kwargs):
        # Masks are True where data is *invalid* (nodata); ~mask == valid.
        valid = ~new_mask
        np.add(merged_data, new_data, out=merged_data, where=valid,
               casting="unsafe")
        band0 = valid[0] if valid.ndim == 3 else valid
        h, w = band0.shape
        count[roff:roff + h, coff:coff + w] += band0
        if merged_mask.shape == valid.shape:
            merged_mask[valid] = False

    # Second pass: accumulate the sum in a float64 buffer.
    summed, _ = rio_merge(
        datasets, method=_sum_valid, nodata=nodata, dtype="float64"
    )

    count_b = count[np.newaxis, :, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(count_b > 0, summed / count_b, nodata)
    avg = np.rint(avg).astype(src_dtype)
    return avg, out_transform
