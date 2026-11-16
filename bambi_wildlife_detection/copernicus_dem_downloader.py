# -*- coding: utf-8 -*-
"""An elevation model for a flight anywhere on earth, from the Copernicus
DEM (see :mod:`core.copernicus_dem`).

:func:`build_dem` reads the heights under a flight, writes them as a
GeoTIFF and makes of that the mesh and the metadata the pipeline works
with - as the Austrian download does from its own tiles.
:class:`CopernicusDEMWorker` runs it off the main thread.
"""

import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

from qgis.PyQt.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

#: The files a flight's worldwide elevation model is written as.
STEM = "copernicus_dem"
#: The mesh generator takes a height of exactly nought for "no data" and
#: leaves a hole there - which is what the sea would be. A centimetre up it
#: is ground like any other.
SEA_LEVEL_M = 0.01


def build_dem(bounds_wgs84: Tuple[float, float, float, float],
              output_folder: str, output_crs: str,
              padding_m: float = 150.0,
              log: Optional[Callable[[str], None]] = None,
              progress: Optional[Callable[[int], None]] = None,
              cancelled: Optional[Callable[[], bool]] = None,
              fetch: Optional[Callable] = None) -> dict:
    """Build the elevation model of a flight from the Copernicus DEM.

    :param bounds_wgs84: ``(west, south, east, north)`` of the flight's
        positions, degrees
    :param output_folder: where ``copernicus_dem.tif``, ``.glb`` and
        ``.json`` are written
    :param output_crs: the UTM CRS of the mesh, e.g. ``"EPSG:32633"``
    :param padding_m: ground kept around the positions (never less than
        five cells of the model)
    :param fetch: what reads the heights -
        :func:`core.copernicus_dem.fetch` unless given
    :return: ``{"mesh", "metadata", "geotiff", "source", "grid"}`` - the
        three files, what the heights were read from, and the grid they
        are on
    :raises core.environment.EnvironmentFailure: no heights there, or the
        service did not answer
    :raises core.environment.Cancelled: stopped by the user
    """
    import numpy as np

    from .austria_dem_downloader import BoundingBox, GLTFMeshGenerator
    from .core import copernicus_dem, environment
    from .core.crs_utils import is_valid_utm_crs, parse_epsg_from_text

    if not is_valid_utm_crs(output_crs):
        raise environment.EnvironmentFailure(
            f"'{output_crs}' is not a UTM CRS: the elevation model needs "
            "one (EPSG:32601-32660 north, EPSG:32701-32760 south).")
    epsg = parse_epsg_from_text(output_crs)
    west, south, east, north = (float(value) for value in bounds_wgs84)
    bounds = BoundingBox(south, west, north, east).to_crs(f"EPSG:{epsg}")
    grid = copernicus_dem.grid_for(bounds, epsg, padding_m)
    if log:
        log(f"Elevation for {grid.width * grid.resolution / 1000:.1f} x "
            f"{grid.height * grid.resolution / 1000:.1f} km around the "
            f"flight, in EPSG:{epsg}")

    def scaled(value: int) -> None:
        if progress:
            progress(int(5 + value * 0.6))

    heights, source = (fetch or copernicus_dem.fetch)(
        grid, log=log, cancelled=cancelled, progress=scaled)
    environment.check_cancelled(cancelled)
    heights = np.where(heights == 0, np.float32(SEA_LEVEL_M),
                       heights).astype(np.float32)

    os.makedirs(output_folder, exist_ok=True)
    base = Path(output_folder) / STEM
    geotiff = base.with_suffix(".tif")
    mesh = base.with_suffix(".glb")
    metadata = base.with_suffix(".json")
    environment.write_raster(str(geotiff), heights, grid,
                             description=source.get("product", ""))
    if progress:
        progress(70)
    generator = GLTFMeshGenerator(
        simplify_factor=1, log_callback=log,
        progress_callback=lambda p: progress(70 + int(p * 0.28))
        if progress else None)
    if not generator.generate_mesh(geotiff, mesh, metadata,
                                   source_crs=f"EPSG:{epsg}"):
        raise environment.EnvironmentFailure(
            "The mesh could not be made from the heights (see the log).")

    # Where the heights came from goes with the model: what uses it can
    # tell a surface model at 30 m from a terrain model at 1 m.
    with open(metadata, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    document["source"] = {
        key: source.get(key) for key in (
            "product", "editions", "tiles", "attribution", "vertical_datum",
            "surface_model", "sea_share")}
    document["source"]["resolution_m"] = grid.resolution
    with open(metadata, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=4)
    if progress:
        progress(100)
    return {"mesh": str(mesh), "metadata": str(metadata),
            "geotiff": str(geotiff), "source": source, "grid": grid}


class CopernicusDEMWorker(QObject):
    """Builds a flight's elevation model from the Copernicus DEM in a
    background thread."""

    finished = pyqtSignal(bool, str)    # success, mesh path or message
    progress = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(self, bounds_wgs84, output_folder: str, output_crs: str,
                 padding: float = 150.0):
        super().__init__()
        self.bounds_wgs84 = tuple(bounds_wgs84)
        self.output_folder = output_folder
        self.output_crs = output_crs
        self.padding = padding
        self._cancelled = False

    def cancel(self):
        """Stop at the next tile."""
        self._cancelled = True

    def run(self):
        from .core import environment

        try:
            self.log.emit("Fetching elevation from the Copernicus DEM...")
            built = build_dem(
                self.bounds_wgs84, self.output_folder, self.output_crs,
                self.padding, log=self.log.emit, progress=self.progress.emit,
                cancelled=lambda: self._cancelled)
        except environment.Cancelled:
            self.finished.emit(False, "Cancelled")
            return
        except environment.EnvironmentFailure as exc:
            self.log.emit(str(exc))
            self.finished.emit(False, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - told to the user, whole
            import traceback

            logger.error("Copernicus DEM: %s", exc)
            traceback.print_exc()
            self.finished.emit(
                False, f"The elevation model could not be made: {exc}")
            return
        self.log.emit(f"Elevation model written: {built['mesh']} "
                      f"({built['source'].get('product', '')})")
        self.finished.emit(True, built["mesh"])
