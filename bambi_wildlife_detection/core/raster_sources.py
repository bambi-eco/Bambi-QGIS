# -*- coding: utf-8 -*-
"""The imagery a detector or classifier looks at (9.0.0).

Three kinds, selected as "Projection" on the Detection and Classification
tabs:

* **perspective** - the extracted camera frames;
* **geotiff** - the orthorectified per-frame GeoTIFFs (``geotiffs_{m}/``);
* **alfs** - the light-field integrals of ALFS sampling mode
  (``alfs_{m}/sample_{frame}_tile_RR_CC.tif``): one integral per *central
  frame*, focused on the DEM, in which occluding vegetation is blurred
  away. Prepared for ALFS detectors and classifiers.

The two raster kinds are georeferenced, so a box found in one has its world
position for free and reaches the camera frame through the DEM
(:mod:`core.ortho_detection`), and a detection's crop is found through its
world coordinates. What differs is which frames have a raster of their own:
every exported frame for GeoTIFFs, only the central frames for ALFS - a
frame in between uses the nearest central frame's integral.
"""

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

PERSPECTIVE = "perspective"
GEOTIFF = "geotiff"
ALFS = "alfs"
KINDS = (PERSPECTIVE, GEOTIFF, ALFS)

#: What a custom implementation's ``context.projection`` says.
CONTEXT_NAMES = {PERSPECTIVE: "perspective", GEOTIFF: "orthographic",
                 ALFS: "alfs"}

LABELS = {PERSPECTIVE: "camera frames",
          GEOTIFF: "GeoTIFFs (orthorectified frames)",
          ALFS: "ALFS light-field integrals"}

#: What has to have run for a raster kind, for messages.
PREREQUISITE = {
    GEOTIFF: "'Export Frames as GeoTIFF'",
    ALFS: "'Generate ALFS' with sampling mode enabled (one integral per "
          "central frame)",
}

_SAMPLE = re.compile(r"^sample_(\d+)_tile_\d+_\d+\.tiff?$", re.IGNORECASE)


def kind_of(projection: Optional[str]) -> str:
    """The kind behind a projection setting: the detection values
    (``perspective`` / ``geotiff`` / ``alfs``) and the classification ones
    (``non_geo`` / ``geo_1k`` / ``geo_2k`` / ``alfs``)."""
    value = (projection or "").lower()
    if value == ALFS:
        return ALFS
    if value == GEOTIFF or value.startswith("geo_"):
        return GEOTIFF
    return PERSPECTIVE


def is_raster(projection: Optional[str]) -> bool:
    return kind_of(projection) != PERSPECTIVE


def context_name(projection: Optional[str]) -> str:
    return CONTEXT_NAMES[kind_of(projection)]


# ---------------------------------------------------------------------------
# ALFS samples
# ---------------------------------------------------------------------------

def alfs_samples(target_folder: str, modality: str) -> Dict[int, List[str]]:
    """``central frame -> [tile paths]`` of the sampling-mode integrals."""
    folder = os.path.join(target_folder, f"alfs_{modality}")
    found: Dict[int, List[str]] = {}
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return found
    for name in names:
        match = _SAMPLE.match(name)
        if match:
            found.setdefault(int(match.group(1)), []).append(
                os.path.join(folder, name))
    return found


def nearest_frame(frames: Sequence[int], frame: int,
                  max_gap: Optional[int] = None) -> Optional[int]:
    """The entry of *frames* closest to *frame* (the earlier one on a tie),
    or None when there is none within *max_gap* frames."""
    best = None
    for candidate in frames:
        distance = abs(candidate - frame)
        if max_gap is not None and distance > max_gap:
            continue
        if best is None or (distance, candidate) < (abs(best - frame), best):
            best = candidate
    return best


# ---------------------------------------------------------------------------
# A source of rasters
# ---------------------------------------------------------------------------

class RasterSource:
    """The georeferenced rasters of one camera, by frame.

    :param max_gap: ALFS only - how many frames away a central frame's
        integral may be to stand in for a frame without one of its own
        (None = the nearest, however far)
    :param reader: ``path -> (image, transform)``; ``geo_crops.read_geotiff``
        by default
    """

    def __init__(self, target_folder: str, modality: str, kind: str,
                 max_gap: Optional[int] = None, reader=None,
                 cache_rasters: int = 4):
        if kind not in (GEOTIFF, ALFS):
            raise ValueError(f"Not a raster projection: {kind!r}")
        self.kind = kind
        self.max_gap = max_gap
        self._reader = reader
        if kind == GEOTIFF:
            from . import geo_crops
            self._paths = {frame: [path] for frame, path in
                           geo_crops.available_frames(
                               target_folder, modality).items()}
        else:
            self._paths = alfs_samples(target_folder, modality)
        self._frames = sorted(self._paths)
        self._cache: Dict[str, tuple] = {}
        self._order: List[str] = []
        self._cache_rasters = max(1, int(cache_rasters))

    def frames(self) -> List[int]:
        """Frames that have a raster of their own."""
        return list(self._frames)

    def paths(self, frame: int) -> List[str]:
        """The rasters of *frame* itself (several tiles for a large ALFS
        integral), or ``[]``."""
        return list(self._paths.get(int(frame), []))

    def frame_for(self, frame: int) -> Optional[int]:
        """The frame whose raster stands for *frame*: itself, or for ALFS
        the nearest central frame within ``max_gap``."""
        frame = int(frame)
        if frame in self._paths:
            return frame
        if self.kind != ALFS:
            return None
        return nearest_frame(self._frames, frame, self.max_gap)

    def read(self, path: str):
        """``(image, transform)`` of one raster, or ``(None, None)``."""
        if path in self._cache:
            return self._cache[path]
        reader = self._reader
        if reader is None:
            from . import geo_crops
            reader = geo_crops.read_geotiff
        loaded = reader(path)
        self._cache[path] = loaded
        self._order.append(path)
        while len(self._order) > self._cache_rasters:
            self._cache.pop(self._order.pop(0), None)
        return loaded

    def read_for(self, frame: int, world_box: Optional[Sequence[float]] = None):
        """The raster showing *world_box* ``(gx1, gy1, gx2, gy2)`` at
        *frame*: ``(image, transform)``, or ``(None, None)``.

        With several tiles the one containing the box centre is taken;
        without a box, the first that can be read.
        """
        source = self.frame_for(frame)
        if source is None:
            return None, None
        for path in self._paths[source]:
            image, transform = self.read(path)
            if image is None:
                continue
            if world_box is None or _contains(image, transform, world_box):
                return image, transform
        return None, None


def _contains(image, transform, world_box: Sequence[float]) -> bool:
    """True when the centre of *world_box* lies inside the raster."""
    x = (float(world_box[0]) + float(world_box[2])) / 2.0
    y = (float(world_box[1]) + float(world_box[3])) / 2.0
    column, row = ~transform * (x, y)
    return 0.0 <= column < image.shape[1] and 0.0 <= row < image.shape[0]


def missing_message(kind: str, camera_name: str, what: str) -> str:
    """Why *what* cannot run on a raster kind that has not been produced."""
    return (f"No {camera_name} {LABELS[kind]} found. {what} is set to use "
            f"them, so run {PREREQUISITE[kind]} for this camera first - or "
            "set the projection to Perspective.")
