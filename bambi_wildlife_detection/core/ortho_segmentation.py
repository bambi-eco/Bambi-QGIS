# -*- coding: utf-8 -*-
"""Prompted segmentation on the orthomosaic instead of the frames.

A frame sees a tree canopy once per pass, so segmenting frames yields one
mask per sighting and the same canopy several times over. The orthomosaic
(``orthomosaic_{t|w}/orthomosaic.tif`` from step P6) shows every canopy
once, so a mask found there is one object on the map - and, the file being
a GeoTIFF, its pixel coordinates map to world coordinates by an affine
transform, so no ray casting is needed afterwards.

SAM3 works at 1008 px and an orthomosaic is tens of thousands of pixels a
side, so the mosaic is cut into overlapping **tiles** which are segmented
one by one with the same backends the frames use. Text-prompt masks from
neighbouring tiles are then **merged**: all of a prompt's masks are painted
into one raster at (reduced) mosaic resolution and its connected regions
become the objects, so a canopy that straddled a tile border is one polygon.
Point prompts define one object each and are answered on a tile centred on
the clicks, so they need no merge.

Results carry the same structure as the frame results - one "frame" entry
with ``frame_idx`` :data:`ORTHO_FRAME` - and are written beside them, so the
QGIS layers and the GeoJSON export are shared.

Everything heavy (``rasterio``, ``numpy``, ``cv2``, ``PIL``) is imported
lazily; this module imports in the headless test stub.
"""

import math
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import sam3_local

ORTHO_FILE = "orthomosaic.tif"

#: ``frame_idx`` of the single orthomosaic entry - never a real frame.
ORTHO_FRAME = -1

#: Default tile edge and overlap in mosaic pixels. A tile of SAM3's working
#: size loses nothing; the overlap keeps an object cut by one tile border
#: whole in the neighbouring tile, so the merge sees it in one piece.
DEFAULT_TILE = 1024
DEFAULT_OVERLAP = 128

#: Longer side of the raster the masks are merged in. 8192 px is 64 MB of
#: uint8 per prompt - fine - and a 3 cm mosaic still merges at ~10 cm.
MERGE_MAX_SIDE = 8192

#: Longer side of the preview the tool shows.
PREVIEW_MAX_SIDE = 4096


class OrthoError(RuntimeError):
    """The orthomosaic could not be read or segmented."""


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------

def orthomosaic_path(target_folder: str, modality: str) -> str:
    return os.path.join(target_folder, f"orthomosaic_{modality}", ORTHO_FILE)


@dataclass
class OrthoInfo:
    """What the tool needs to know before segmenting."""
    path: str
    width: int
    height: int
    transform: List[float]          # a, b, c, d, e, f  (x = c + a·col + b·row)
    epsg: Optional[int]
    bands: int
    nodata: Optional[float]
    dtype: str

    @property
    def gsd(self) -> float:
        """Ground sampling distance in CRS units per pixel (metres for UTM)."""
        return math.hypot(self.transform[0], self.transform[3])


def read_info(path: str) -> OrthoInfo:
    if not os.path.isfile(path):
        raise OrthoError(
            f"No orthomosaic at {path} - run P6 Generate Orthomosaic first.")
    try:
        import rasterio
    except ImportError as exc:
        raise OrthoError(
            "rasterio is required to read the orthomosaic. Install the core "
            "dependencies from the Dependency Manager.") from exc
    with rasterio.open(path) as dataset:
        transform = dataset.transform
        epsg = None
        try:
            epsg = dataset.crs.to_epsg() if dataset.crs else None
        except Exception:  # nosec B110 - an exotic CRS; the tool falls back
            epsg = None
        return OrthoInfo(
            path=path, width=int(dataset.width), height=int(dataset.height),
            transform=[float(transform.a), float(transform.b), float(transform.c),
                       float(transform.d), float(transform.e), float(transform.f)],
            epsg=epsg, bands=int(dataset.count),
            nodata=(None if dataset.nodata is None else float(dataset.nodata)),
            dtype=str(dataset.dtypes[0]))


def pixel_to_world(transform: Sequence[float], x: float, y: float
                   ) -> Tuple[float, float]:
    """The world position of mosaic pixel (*x*, *y*) - column, row."""
    a, b, c, d, e, f = transform
    return (c + a * x + b * y, f + d * x + e * y)


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def tile_windows(width: int, height: int, tile: int = DEFAULT_TILE,
                 overlap: int = DEFAULT_OVERLAP) -> List[Tuple[int, int, int, int]]:
    """``(x0, y0, w, h)`` windows covering the mosaic, row by row.

    Tiles step by ``tile - overlap``; the last tile of a row or column is
    pulled back to end at the mosaic edge, so no sliver tile is produced and
    every tile is *tile* wide unless the mosaic itself is smaller.
    """
    tile = max(64, int(tile))
    overlap = max(0, min(int(overlap), tile - 32))
    step = tile - overlap

    def starts(extent: int) -> List[int]:
        if extent <= tile:
            return [0]
        positions = list(range(0, extent - tile, step))
        positions.append(extent - tile)
        return positions

    windows = []
    for y0 in starts(height):
        for x0 in starts(width):
            windows.append((x0, y0, min(tile, width - x0), min(tile, height - y0)))
    return windows


def tile_count(width: int, height: int, tile: int, overlap: int) -> int:
    return len(tile_windows(width, height, tile, overlap))


def window_around(cx: float, cy: float, width: int, height: int,
                  tile: int) -> Tuple[int, int, int, int]:
    """A *tile*-sized window centred on (*cx*, *cy*), clamped to the mosaic."""
    tile = max(64, int(tile))
    w = min(tile, width)
    h = min(tile, height)
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    x0 = max(0, min(x0, width - w))
    y0 = max(0, min(y0, height - h))
    return (x0, y0, w, h)


# ---------------------------------------------------------------------------
# Reading pixels
# ---------------------------------------------------------------------------

class OrthoReader:
    """Reads the mosaic as 8-bit RGB, tile by tile, never whole.

    A thermal mosaic has one band and may be 16-bit or float; it is
    stretched between its 2nd and 98th percentile (measured on the preview)
    and repeated into three channels, which is what a thermal frame looks
    like to SAM3 as well. An RGB mosaic's first three bands are used as
    they are; an alpha band marks nodata.
    """

    def __init__(self, path: str):
        try:
            import rasterio
        except ImportError as exc:
            raise OrthoError(
                "rasterio is required to read the orthomosaic. Install the "
                "core dependencies from the Dependency Manager.") from exc
        self._rasterio = rasterio
        self.info = read_info(path)
        self._dataset = rasterio.open(path)
        self._stretch: Optional[Tuple[float, float]] = None

    def close(self) -> None:
        try:
            self._dataset.close()
        except Exception:  # nosec B110
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- bands ----------------------------------------------------------------

    def _indexes(self) -> List[int]:
        bands = self.info.bands
        return [1, 2, 3] if bands >= 3 else [1]

    def _alpha_index(self) -> Optional[int]:
        return 4 if self.info.bands == 4 else (2 if self.info.bands == 2 else None)

    def _to_rgb8(self, data, alpha=None):
        """``(bands, h, w)`` of any dtype -> ``(h, w, 3)`` uint8 + valid mask."""
        import numpy as np

        data = np.asarray(data)
        valid = np.ones(data.shape[1:], dtype=bool)
        nodata = self.info.nodata
        if nodata is not None:
            valid &= ~np.all(data == nodata, axis=0)
        if alpha is not None:
            valid &= np.asarray(alpha) > 0
        if data.dtype != np.uint8:
            lo, hi = self._stretch or self._measure_stretch()
            scaled = (data.astype(np.float32) - lo) / max(1e-6, hi - lo)
            data = (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
        if data.shape[0] == 1:
            data = np.repeat(data, 3, axis=0)
        rgb = np.transpose(data[:3], (1, 2, 0)).copy()
        rgb[~valid] = 0
        return rgb, valid

    def _measure_stretch(self) -> Tuple[float, float]:
        import numpy as np

        factor = max(1, int(math.ceil(max(self.info.width, self.info.height) / 1024)))
        shape = (len(self._indexes()), max(1, self.info.height // factor),
                 max(1, self.info.width // factor))
        sample = self._dataset.read(self._indexes(), out_shape=shape)
        values = sample.astype(np.float32)
        if self.info.nodata is not None:
            values = values[values != self.info.nodata]
        values = values[np.isfinite(values)]
        if values.size == 0:
            self._stretch = (0.0, 1.0)
        else:
            lo, hi = np.percentile(values, [2, 98])
            self._stretch = (float(lo), float(hi if hi > lo else lo + 1.0))
        return self._stretch

    # -- reads ----------------------------------------------------------------

    def preview(self, max_side: int = PREVIEW_MAX_SIDE):
        """``(rgb uint8 (h, w, 3), scale)`` with *scale* = mosaic px per preview px."""
        factor = max(1, int(math.ceil(max(self.info.width, self.info.height)
                                      / float(max_side))))
        h = max(1, int(math.ceil(self.info.height / factor)))
        w = max(1, int(math.ceil(self.info.width / factor)))
        data = self._dataset.read(self._indexes(), out_shape=(len(self._indexes()), h, w))
        alpha = None
        if self._alpha_index() is not None:
            alpha = self._dataset.read(self._alpha_index(), out_shape=(h, w))
        rgb, _valid = self._to_rgb8(data, alpha)
        return rgb, self.info.width / float(w)

    def tile(self, window: Tuple[int, int, int, int]):
        """``rgb uint8 (h, w, 3)`` of *window*, or ``None`` when it is all nodata."""
        from rasterio.windows import Window

        x0, y0, w, h = window
        rio_window = Window(x0, y0, w, h)
        data = self._dataset.read(self._indexes(), window=rio_window)
        alpha = None
        if self._alpha_index() is not None:
            alpha = self._dataset.read(self._alpha_index(), window=rio_window)
        rgb, valid = self._to_rgb8(data, alpha)
        if not valid.any():
            return None
        return rgb


# ---------------------------------------------------------------------------
# Merging tile results
# ---------------------------------------------------------------------------

def _offset_polygons(polygons, dx: float, dy: float):
    return [[[float(x) + dx, float(y) + dy] for x, y in ring] for ring in polygons]


def _ring_bbox(ring) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def _bboxes_touch(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def merge_tiles(width: int, height: int,
                tiles: Iterable[Tuple[Tuple[int, int, int, int], List[dict]]],
                merge_max_side: int = MERGE_MAX_SIDE) -> List[dict]:
    """One prompt entry per prompt, its masks unified across tiles.

    *tiles* pairs each window with the entries the backend returned for it
    (in tile pixel coordinates). Text entries of one prompt are painted into
    a shared raster - the mosaic at ``1/scale`` - and its connected regions
    are the merged objects, numbered from 1 as ``object_id``; each takes the
    highest confidence of the tile masks it overlaps. Point entries are one
    object each already and are only moved into mosaic coordinates.
    """
    import numpy as np

    scale = max(1, int(math.ceil(max(width, height) / float(merge_max_side))))
    mw = max(1, int(math.ceil(width / scale)))
    mh = max(1, int(math.ceil(height / scale)))

    rasters: Dict[str, object] = {}
    scored: Dict[str, List[Tuple[Tuple[float, float, float, float], float]]] = {}
    order: List[str] = []
    point_entries: List[dict] = []

    for (x0, y0, _w, _h), entries in tiles:
        for entry in entries or []:
            if entry.get("prompt_type") == "point":
                moved = dict(entry)
                moved["predictions"] = [
                    dict(pred, polygons=_offset_polygons(
                        pred.get("polygons", []), x0, y0))
                    for pred in entry.get("predictions", [])]
                point_entries.append(moved)
                continue
            name = entry.get("prompt", "")
            if name not in rasters:
                rasters[name] = np.zeros((mh, mw), dtype=np.uint8)
                scored[name] = []
                order.append(name)
            raster = rasters[name]
            for pred in entry.get("predictions", []):
                confidence = float(pred.get("confidence", 0.0))
                for ring in pred.get("polygons", []):
                    if len(ring) < 3:
                        continue
                    pts = np.array([[(x + x0) / scale, (y + y0) / scale]
                                    for x, y in ring], dtype=np.float64)
                    _fill(raster, pts)
                    scored[name].append((_ring_bbox(pts.tolist()), confidence))

    merged: List[dict] = []
    for name in order:
        predictions = []
        rings = sam3_local.mask_to_polygons(rasters[name])
        for index, ring in enumerate(rings, start=1):
            bbox = _ring_bbox(ring)
            confidence = max(
                [c for b, c in scored[name] if _bboxes_touch(b, bbox)],
                default=0.0)
            predictions.append({
                "confidence": confidence,
                "polygons": [[[(x + 0.5) * scale, (y + 0.5) * scale]
                              for x, y in ring]],
                "object_id": index,
            })
        merged.append({"prompt": name, "prompt_type": "text",
                       "predictions": predictions})
    return merged + point_entries


def _fill(raster, pts) -> None:
    import numpy as np

    try:
        import cv2
    except ImportError as exc:
        raise OrthoError(
            "OpenCV (cv2) is required to merge the tile masks. Install the "
            "core dependencies from the Dependency Manager.") from exc
    cv2.fillPoly(raster, [np.round(pts).astype(np.int32)], 1)


# ---------------------------------------------------------------------------
# Geo-referencing
# ---------------------------------------------------------------------------

def georeference(pixel_results: List[dict]) -> List[dict]:
    """The geo twin of the orthomosaic pixel results: an affine, no DEM.

    Each entry carries the transform it was segmented with, so the geo file
    is a pure function of the pixel file and is rewritten with it. World
    points get a ``z`` of 0.0: the mosaic is a map, not a surface, and the
    consumers read x/y only.
    """
    georef = []
    for entry in pixel_results:
        transform = entry.get("transform")
        if not transform:
            continue
        geo_entry = {k: v for k, v in entry.items() if k != "prompts"}
        geo_entry["prompts"] = []
        for prompt in entry.get("prompts", []):
            geo_prompt = {k: v for k, v in prompt.items() if k != "predictions"}
            geo_prompt["predictions"] = []
            for pred in prompt.get("predictions", []):
                geo_pred = {k: v for k, v in pred.items()
                            if k not in ("polygons", "box")}
                geo_pred["world_polygons"] = [
                    [[*pixel_to_world(transform, x, y), 0.0] for x, y in ring]
                    for ring in pred.get("polygons", []) if len(ring) >= 3]
                if geo_pred["world_polygons"]:
                    geo_prompt["predictions"].append(geo_pred)
            geo_entry["prompts"].append(geo_prompt)
        georef.append(geo_entry)
    return georef


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run_orthomosaic(request, backend, progress_fn=None, log_fn=None,
                    cancel_check=None) -> List[dict]:
    """Segment the mosaic tile by tile; one pixel-space entry in a list.

    *request* is a :class:`segmentation.SegmentationRequest` with
    ``source == SOURCE_ORTHO``; *backend* is loaded. *progress_fn* receives a
    fraction 0..1. The entry is shaped like a frame entry plus ``source``,
    ``width``, ``height``, ``transform`` and ``epsg``, so :func:`georeference`
    can rebuild the geo file from it at any time.
    """
    from . import segmentation as seg

    path = orthomosaic_path(request.target_folder, request.modality)
    camera = "Thermal" if request.modality == "t" else "RGB"
    tile = int(request.tile_size or DEFAULT_TILE)
    overlap = int(request.tile_overlap if request.tile_overlap is not None
                  else DEFAULT_OVERLAP)

    def _progress(done, total):
        if progress_fn:
            progress_fn(done / max(1, total))

    with OrthoReader(path) as reader:
        info = reader.info
        if log_fn:
            log_fn(f"{camera} orthomosaic: {info.width} x {info.height} px, "
                   f"{info.gsd * 100:.1f} cm/px, {info.bands} band(s), "
                   f"EPSG:{info.epsg or '?'}")
        with tempfile.TemporaryDirectory(prefix="bambi_ortho_") as tmp:
            if request.texts:
                tiles = _segment_text_tiles(
                    reader, request, backend, tile, overlap, tmp, _progress,
                    log_fn, cancel_check)
                if log_fn:
                    log_fn("Merging the tile masks into one map…")
                prompts = merge_tiles(info.width, info.height, tiles)
            else:
                prompts = _segment_point_objects(
                    reader, request, backend, tile, tmp, _progress, log_fn,
                    cancel_check)

    entry = {
        "frame_idx": ORTHO_FRAME,
        "imagefile": ORTHO_FILE,
        "source": seg.SOURCE_ORTHO,
        "width": info.width,
        "height": info.height,
        "transform": list(info.transform),
        "epsg": info.epsg,
        "prompts": prompts,
    }
    if log_fn:
        found = sum(len(p.get("predictions", [])) for p in prompts)
        log_fn(f"Orthomosaic segmentation complete: {found} object(s)")
    return [entry]


def _write_tile(tmp: str, name: str, rgb) -> str:
    from PIL import Image

    path = os.path.join(tmp, f"{name}.png")
    Image.fromarray(rgb).save(path)
    return path


def _segment_text_tiles(reader, request, backend, tile, overlap, tmp,
                        progress, log_fn, cancel_check):
    from . import segmentation as seg

    info = reader.info
    windows = tile_windows(info.width, info.height, tile, overlap)
    if log_fn:
        log_fn(f"{len(windows)} tile(s) of {tile} px with {overlap} px overlap")
    results = []
    skipped = 0
    for index, window in enumerate(windows):
        if cancel_check and cancel_check():
            raise seg.CancelledError("Segmentation cancelled")
        rgb = reader.tile(window)
        if rgb is None:
            skipped += 1
        else:
            path = _write_tile(tmp, f"tile_{window[1]}_{window[0]}", rgb)
            try:
                entries = backend.segment_image(
                    path, request.texts, [], request.object_names,
                    request.confidence)
                results.append((window, entries))
            except seg.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad tile is logged
                if log_fn:
                    log_fn(f"Warning: tile at {window[:2]} failed: {exc}")
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        progress(index + 1, len(windows))
        if log_fn and (index + 1) % 10 == 0:
            log_fn(f"Segmented {index + 1}/{len(windows)} tiles")
    if log_fn and skipped:
        log_fn(f"{skipped} empty tile(s) skipped")
    return results


def _segment_point_objects(reader, request, backend, tile, tmp, progress,
                           log_fn, cancel_check):
    from . import segmentation as seg

    info = reader.info
    grouped: Dict[int, List] = {}
    for point in request.points:
        grouped.setdefault(int(point.object_id), []).append(point)
    entries: List[dict] = []
    for index, (object_id, points) in enumerate(sorted(grouped.items())):
        if cancel_check and cancel_check():
            raise seg.CancelledError("Segmentation cancelled")
        cx = sum(p.x for p in points) / len(points)
        cy = sum(p.y for p in points) / len(points)
        window = window_around(cx, cy, info.width, info.height, tile)
        x0, y0, _w, _h = window
        rgb = reader.tile(window)
        if rgb is None:
            if log_fn:
                log_fn(f"Warning: object {object_id} lies on empty mosaic")
            progress(index + 1, len(grouped))
            continue
        local = [seg.PointPrompt(object_id, ORTHO_FRAME, p.x - x0, p.y - y0,
                                 p.positive) for p in points]
        path = _write_tile(tmp, f"object_{object_id}", rgb)
        try:
            found = backend.segment_image(path, [], local, request.object_names,
                                          request.confidence)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        for entry in found:
            moved = dict(entry)
            moved["predictions"] = [
                dict(pred, polygons=_offset_polygons(pred.get("polygons", []), x0, y0))
                for pred in entry.get("predictions", [])]
            entries.append(moved)
        progress(index + 1, len(grouped))
    return entries
