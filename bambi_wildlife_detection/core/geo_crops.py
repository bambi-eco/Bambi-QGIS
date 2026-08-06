# -*- coding: utf-8 -*-
"""Crops taken from the orthorectified frames rather than the camera frames.

The classification heads ship in three projection variants — ``non_geo`` for
the native camera view, ``geo_1k`` and ``geo_2k`` for orthophotos — because a
crop from a warped image does not look like a crop from a perspective one. A
head has to see the kind of image it was trained on, so choosing the variant
also chooses where the crops come from.

The detections themselves never move: they are stored in the camera's pixel
space and geo-referenced into world coordinates once. A geo crop is therefore
the world box mapped through the *raster's* own affine transform into that
GeoTIFF's pixel space — which keeps one geo-referencing result serving both
projections, rather than re-detecting on the orthophotos.

``rasterio`` is imported lazily: a project that only ever uses perspective
crops does not need it loaded.
"""

import os
from typing import Dict, Iterable, Optional, Sequence, Tuple

#: Nominal source width of each orthorectified head variant, from the
#: repositories' own naming. Used only to warn when the exported rasters are a
#: long way off — the head was fitted to features at roughly that scale.
NOMINAL_WIDTH = {"geo_1k": 1024, "geo_2k": 2048}

#: How far from nominal a raster may be before it is worth mentioning.
_TOLERANCE = 0.5


def is_geo(projection: str) -> bool:
    """True for the projections whose crops come from GeoTIFFs."""
    return projection in NOMINAL_WIDTH


def geotiff_path(target_folder: str, modality: str, frame: int) -> str:
    """Where the GeoTIFF export writes a frame.

    Mirrors ``BambiProcessor.run_export_geotiffs``: ``{frame:08d}.tiff`` in
    ``geotiffs_{m}/``, keyed on the frame index rather than the image name.
    """
    return os.path.join(target_folder, f"geotiffs_{modality}",
                        f"{int(frame):08d}.tiff")


def available_frames(target_folder: str, modality: str) -> Dict[int, str]:
    """Frame index → GeoTIFF path, for whatever has been exported."""
    folder = os.path.join(target_folder, f"geotiffs_{modality}")
    if not os.path.isdir(folder):
        return {}
    found = {}
    for name in os.listdir(folder):
        stem, extension = os.path.splitext(name)
        if extension.lower() not in (".tif", ".tiff"):
            continue
        try:
            found[int(stem)] = os.path.join(folder, name)
        except ValueError:
            continue   # not a frame-numbered export
    return found


# ---------------------------------------------------------------------------
# World -> raster pixels
# ---------------------------------------------------------------------------

def world_to_pixel(transform, x: float, y: float) -> Tuple[float, float]:
    """Map a world coordinate into raster pixel coordinates.

    *transform* is a rasterio/affine ``Affine``; its inverse is the mapping
    from world space back into the raster.
    """
    column, row = ~transform * (float(x), float(y))
    return (float(column), float(row))


def geo_box_to_pixel(transform, box: Sequence[float]
                     ) -> Tuple[float, float, float, float]:
    """Map a world-space box ``(gx1, gy1, gx2, gy2)`` into raster pixels.

    The corners are re-ordered afterwards because a north-up raster has a
    negative y scale: the world's lower-left corner is the raster's *bottom*,
    so mapping the corners straight through yields ``y1 > y2``. Sorting keeps
    the result a well-formed box whichever way the raster is oriented.
    """
    gx1, gy1, gx2, gy2 = (float(v) for v in box)
    first = world_to_pixel(transform, gx1, gy1)
    second = world_to_pixel(transform, gx2, gy2)
    return (min(first[0], second[0]), min(first[1], second[1]),
            max(first[0], second[0]), max(first[1], second[1]))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_geotiff(path: str):
    """Return ``(image, transform)`` for a GeoTIFF, or ``(None, None)``.

    The image comes back as ``HxWx3`` uint8, whatever the raster holds: the
    export writes RGB (sometimes with an alpha band marking the parts of the
    footprint the frame does not cover), and thermal exports may be a single
    band.
    """
    import numpy as np

    if not os.path.isfile(path):
        return None, None

    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover — environment dependent
        raise RuntimeError(
            "rasterio is required to crop from orthorectified frames. "
            "Install it from the Dependency Manager, or set the "
            "classification projection to Perspective.") from exc

    try:
        with rasterio.open(path) as raster:
            data = raster.read()
            transform = raster.transform
    except Exception:
        # A half-written export is treated like an absent one: the caller
        # skips the frame and says so, rather than failing the whole run.
        return None, None

    # rasterio gives (bands, rows, cols); images are (rows, cols, bands).
    image = np.transpose(data, (1, 2, 0))
    if image.dtype != np.uint8:
        image = _to_uint8(image)

    bands = image.shape[2]
    if bands == 1:
        image = np.repeat(image, 3, axis=2)
    elif bands >= 4:
        image = image[:, :, :3]
    elif bands == 2:  # pragma: no cover — unusual, but not worth crashing on
        image = np.repeat(image[:, :, :1], 3, axis=2)

    return np.ascontiguousarray(image), transform


def _to_uint8(image):
    """Scale a non-8-bit raster into 8 bits, per raster rather than per crop.

    Doing it per crop would make the same animal look different depending on
    what else was in its window.
    """
    import numpy as np

    finite = image[np.isfinite(image)] if image.dtype.kind == "f" else image
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (np.clip(image, low, high) - low) * (255.0 / (high - low))
    return scaled.astype(np.uint8)


# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------

def resolution_warning(projection: str, width: int) -> Optional[str]:
    """Flag a raster whose size is far from the chosen variant's nominal one.

    Not an error: the export resolution follows the flight and the DEM, and
    the heads still produce an answer at another scale. But a ``geo_2k`` head
    reading 400 px orthophotos is being used outside what it was fitted to,
    and that is worth saying once rather than leaving it to show up as poor
    accuracy.
    """
    nominal = NOMINAL_WIDTH.get(projection)
    if not nominal or not width:
        return None
    ratio = float(width) / float(nominal)
    if _TOLERANCE <= ratio <= 1.0 / _TOLERANCE:
        return None
    return (f"The exported GeoTIFFs are {width} px wide, but the "
            f"'{projection}' classifiers were fitted around {nominal} px. "
            "The results are still produced, but treat them with care — or "
            f"pick the variant closer to your export.")


def missing_frames(target_folder: str, modality: str,
                   frames: Iterable[int]) -> list:
    """Which of *frames* have no GeoTIFF exported."""
    available = available_frames(target_folder, modality)
    return sorted({int(frame) for frame in frames} - set(available))
