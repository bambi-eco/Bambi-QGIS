# -*- coding: utf-8 -*-
"""Small writers of the 3D Flight Viewer: PNG + world file, PLY.

Deliberately dependency-free (zlib / struct), so exports work whichever
optional packages are installed.
"""

import os
import struct
import zlib
from typing import Optional, Sequence

import numpy as np


def write_png(path: str, image) -> None:
    """Write an ``(H, W, 3|4)`` uint8 image as PNG."""
    img = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if img.ndim == 2:
        img = img[:, :, None]
    h, w, c = img.shape
    color_type = {1: 0, 3: 2, 4: 6}[c]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        body = tag + data
        return (struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type,
                                            0, 0, 0)))
        fh.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        fh.write(chunk(b"IEND", b""))


def write_world_file(image_path: str, lower: Sequence[float],
                     upper: Sequence[float], width: int, height: int,
                     origin: Optional[Sequence[float]] = None,
                     epsg: Optional[int] = None) -> str:
    """World file (+ ``.aux.xml`` with the CRS) for a north-up image covering
    the DEM-local extent ``lower``-``upper``.

    :return: the world file's path
    """
    ox, oy = (float(origin[0]), float(origin[1])) if origin is not None else (0.0, 0.0)
    px = (float(upper[0]) - float(lower[0])) / width
    py = (float(upper[1]) - float(lower[1])) / height
    ext = os.path.splitext(image_path)[1].lower()
    world_ext = {".png": ".pgw", ".jpg": ".jgw", ".tif": ".tfw"}.get(ext, ".wld")
    world = os.path.splitext(image_path)[0] + world_ext
    with open(world, "w", encoding="utf-8") as fh:
        fh.write("\n".join(f"{v:.10f}" for v in (
            px, 0.0, 0.0, -py,
            ox + float(lower[0]) + px / 2.0,
            oy + float(upper[1]) - py / 2.0)) + "\n")
    if epsg:
        with open(image_path + ".aux.xml", "w", encoding="utf-8") as fh:
            fh.write(f"<PAMDataset>\n  <SRS>EPSG:{int(epsg)}</SRS>\n</PAMDataset>\n")
    return world


def write_ply(path: str, points, colors=None) -> None:
    """Binary little-endian PLY point cloud (colours as uint8 RGB)."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    header = ["ply", "format binary_little_endian 1.0",
              f"element vertex {len(pts)}",
              "property float x", "property float y", "property float z"]
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if colors is not None:
        header += ["property uchar red", "property uchar green",
                   "property uchar blue"]
        dtype += [("r", "u1"), ("g", "u1"), ("b", "u1")]
    header.append("end_header")
    rec = np.zeros(len(pts), dtype=dtype)
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if colors is not None:
        col = np.asarray(colors)
        if col.dtype != np.uint8:
            col = np.clip(col * (255.0 if col.max() <= 1.0 else 1.0), 0, 255)
        col = col.astype(np.uint8).reshape(-1, col.shape[-1])
        rec["r"], rec["g"], rec["b"] = col[:, 0], col[:, 1], col[:, 2]
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(rec.tobytes())


def read_ply(path: str):
    """Read what :func:`write_ply` wrote: ``(points, colors or None)``."""
    with open(path, "rb") as fh:
        props, count = [], 0
        while True:
            line = fh.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property"):
                _, kind, name = line.split()
                props.append((name, {"float": "<f4", "uchar": "u1"}[kind]))
            elif line == "end_header":
                break
        rec = np.frombuffer(fh.read(), dtype=props, count=count)
    pts = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64)
    names = [n for n, _ in props]
    cols = np.column_stack([rec["red" if "red" in names else "r"],
                            rec["green" if "green" in names else "g"],
                            rec["blue" if "blue" in names else "b"]]) \
        if ("r" in names or "red" in names) else None
    return pts, cols


# ---------------------------------------------------------------------------
# Orthomosaic of a flight (for draping in the 3D viewer)
# ---------------------------------------------------------------------------

def find_orthomosaic(target_folder: str, modality: str) -> str:
    """The orthomosaic the pipeline wrote for a camera
    (``orthomosaic_{t|w}/orthomosaic.tif``), or ``""``."""
    path = os.path.join(target_folder, f"orthomosaic_{modality}",
                        "orthomosaic.tif")
    return path if os.path.isfile(path) else ""


def _to_rgba(bands: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``(C, H, W)`` raster bands + validity mask as RGBA ``uint8``."""
    bands = np.asarray(bands)
    if bands.shape[0] >= 3:
        rgb = bands[:3]
    else:
        rgb = np.repeat(bands[:1], 3, axis=0)
    if rgb.dtype != np.uint8:
        # e.g. float temperatures: stretch the valid values to 0..255
        data = rgb.astype(np.float64)
        ok = valid & np.all(np.isfinite(data), axis=0)
        if ok.any():
            lo, hi = np.percentile(data[:, ok], [1, 99])
            data = (data - lo) / max(hi - lo, 1e-9) * 255.0
        rgb = np.clip(np.nan_to_num(data), 0, 255).astype(np.uint8)
    out = np.empty(rgb.shape[1:] + (4,), np.uint8)
    out[..., :3] = np.moveaxis(rgb, 0, -1)
    out[..., 3] = np.where(valid, 255, 0)
    return out


def read_raster_rgba(path: str, max_edge: int = 4096):
    """Read a north-up georeferenced raster, shrunk so its longer edge is at
    most *max_edge* (overviews are used when the file has them).

    :return: ``(rgba (H, W, 4) uint8, (x_min, y_min), (x_max, y_max))`` in
        the raster's CRS; pixels without data are transparent
    :raises RuntimeError: when neither rasterio nor GDAL is available, or
        the raster is rotated
    """
    try:
        import rasterio
    except ImportError:
        rasterio = None
    if rasterio is not None:
        with rasterio.open(path) as ds:
            t = ds.transform
            if abs(t.b) > 1e-9 or abs(t.d) > 1e-9:
                raise RuntimeError("rotated rasters are not supported")
            scale = min(1.0, max_edge / float(max(ds.width, ds.height)))
            shape = (max(1, int(round(ds.height * scale))),
                     max(1, int(round(ds.width * scale))))
            count = min(ds.count, 3) if ds.count >= 3 else 1
            bands = ds.read(list(range(1, count + 1)), out_shape=(count,) + shape)
            valid = ds.dataset_mask(out_shape=shape) > 0
            b = ds.bounds
        return _to_rgba(bands, valid), (b.left, b.bottom), (b.right, b.top)
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise RuntimeError("reading the orthomosaic needs rasterio or GDAL") from exc
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"cannot open {path}")
    gt = ds.GetGeoTransform()
    if abs(gt[2]) > 1e-9 or abs(gt[4]) > 1e-9:
        raise RuntimeError("rotated rasters are not supported")
    w, h = ds.RasterXSize, ds.RasterYSize
    scale = min(1.0, max_edge / float(max(w, h)))
    bw, bh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    count = min(ds.RasterCount, 3) if ds.RasterCount >= 3 else 1
    bands = np.stack([ds.GetRasterBand(i + 1).ReadAsArray(buf_xsize=bw, buf_ysize=bh)
                      for i in range(count)])
    valid = ds.GetRasterBand(1).GetMaskBand().ReadAsArray(buf_xsize=bw,
                                                          buf_ysize=bh) > 0
    x0, x1 = gt[0], gt[0] + gt[1] * w
    y1, y0 = gt[3], gt[3] + gt[5] * h
    return _to_rgba(bands, valid), (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))
