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


#: For a camera whose matrices are not known, its reach is estimated from
#: its height and vertical field of view, as if it looked straight down;
#: this factor covers the frame's corners and some tilt.
_NADIR_MARGIN = 2.5
#: A corner ray this close to the horizon (cosine of its angle from straight
#: down) reaches "everywhere".
_HORIZON = 0.02


def _corner_rays(shot):
    """The directions of a shot's four frame corners in the world, from the
    matrices the renderer projects it with - ``None`` when it has none.

    The renderer maps a world point *p* into a shot's image with
    ``proj * correction * view * p``; taking that back for the corners of
    the image gives the edges of what the shot can light. No assumption
    about angles or their order is made here: whatever the renderer draws
    is inside these rays.
    """
    try:
        view = np.asarray(shot.get_view(), dtype=np.float64).reshape(4, 4)
        correction = np.asarray(shot.get_correction(),
                                dtype=np.float64).reshape(4, 4)
        proj = np.asarray(shot.get_proj(), dtype=np.float64).reshape(4, 4)
        # pyrr matrices act on row vectors: clip = p @ view @ correction @ proj
        inverse = np.linalg.inv(view @ correction @ proj)
    except Exception:  # noqa: BLE001 - a stand-in, or a degenerate camera
        return None
    rays = []
    for x, y in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
        near = np.array([x, y, -1.0, 1.0]) @ inverse
        far = np.array([x, y, 1.0, 1.0]) @ inverse
        if abs(near[3]) < 1e-12 or abs(far[3]) < 1e-12:
            return None
        direction = far[:3] / far[3] - near[:3] / near[3]
        length = float(np.linalg.norm(direction))
        if not np.isfinite(length) or length <= 0.0:
            return None
        rays.append(direction / length)
    return np.array(rays)


def shot_reach(shot, ground_z=0.0, fov_default=50.0):
    """``(x, y, radius)``: the circle on the map outside of which a shot
    sees no ground.

    Its centre is the point below the camera. Its radius is the camera's
    height above the lowest ground times the tangent of the angle, from
    straight down, of the frame corner that looks out furthest - which is
    as far as that corner's ray gets before it meets ground at any height.
    ``inf`` when that cannot be bounded: a corner at or above the horizon,
    or a camera not above the lowest ground.

    :param ground_z: height of the lowest ground, in the shots' coordinates
        (0 for a DEM mesh, whose heights count from its lowest point)
    """
    position = shot.camera.transform.position
    x, y, z = float(position.x), float(position.y), float(position.z)
    height = z - float(ground_z)
    if height < 1.0:
        return x, y, math.inf

    rays = _corner_rays(shot)
    if rays is None:
        try:
            fov_y = shot.camera.fovy or fov_default
        except Exception:  # noqa: BLE001
            fov_y = fov_default
        half_height = height * math.tan(math.radians(fov_y / 2))
        return x, y, half_height * _NADIR_MARGIN
    downward = float(np.min(-rays[:, 2]))     # cosine of the widest angle
    if downward <= _HORIZON:
        return x, y, math.inf
    # 2 % on top: the frame's edge is a pixel, not a line
    return x, y, height * math.sqrt(1.0 - downward ** 2) / downward * 1.02


class ShotReach:
    """Which shots can see a tile - worked out once for a set of shots.

    A tile used to get *all* shots when none of them reached it, "so tiles
    on the boundary still receive at least one shot". The mosaic spans the
    whole DEM, so that was the case for every tile away from the flight:
    each had every frame of the flight read, uploaded and rendered into it,
    to be found empty and thrown away. A tile no shot reaches gets no shot
    now, and is not rendered.

    That is only safe when a shot's reach is never underestimated, which
    the estimate it replaces did for a camera that does not look straight
    down (its frames were then missing from tiles they covered). See
    :func:`shot_reach`.
    """

    def __init__(self, shots, ground_z=0.0, fov_default=50.0):
        self.shots = list(shots)
        reach = [shot_reach(shot, ground_z, fov_default)
                 for shot in self.shots]
        self._reach = np.array(reach, dtype=np.float64).reshape(-1, 3)

    def for_tile(self, tile_geo_bounds, log_fn=None):
        """The shots that can see ``(min_x, min_y, max_x, max_y)``; ``[]``
        when none does."""
        if not self.shots:
            return []
        min_x, min_y, max_x, max_y = tile_geo_bounds
        x, y, radius = self._reach[:, 0], self._reach[:, 1], self._reach[:, 2]
        # the tile's point nearest to the one below the camera
        distance = np.hypot(x - np.clip(x, min_x, max_x),
                            y - np.clip(y, min_y, max_y))
        relevant = [shot for shot, sees in zip(self.shots, distance <= radius)
                    if sees]
        if log_fn and relevant and len(relevant) < len(self.shots):
            log_fn(f"  Shot filter: {len(relevant)}/{len(self.shots)} shots "
                   "relevant for this tile")
        return relevant


def filter_shots_for_tile(shots, tile_geo_bounds, log_fn=None,
                          fov_default=50.0, ground_z=0.0):
    """The shots that can see a tile; ``[]`` when none does, and the tile
    need not be rendered. See :class:`ShotReach`, which does the same for
    many tiles without working out each shot's reach again."""
    return ShotReach(shots, ground_z, fov_default).for_tile(
        tile_geo_bounds, log_fn)


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
