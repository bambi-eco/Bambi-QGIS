# -*- coding: utf-8 -*-
"""An elevation model for a flight anywhere on earth.

Two open models can be read without an account, and only under the flight:

* Google's **GEM-15** (:mod:`core.gem15_dem`) - the bare ground at 15 m,
  for most of the world. Levelled to the Copernicus DEM on open ground and
  with its gaps filled from it;
* the **Copernicus DEM** (:mod:`core.copernicus_dem`) - the surface, with
  the canopy of a forest in it, at 30 m, for all of it.

:func:`build_dem` reads the heights under a flight - the terrain model
where there is one, unless told otherwise - writes them as a GeoTIFF and
makes of that the mesh and the metadata the pipeline works with, as the
Austrian download does from its own tiles. :class:`CopernicusDEMWorker`
runs it off the main thread.
"""

import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

from qgis.PyQt.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

#: What the heights can be taken from: the terrain model where it has the
#: flight and the surface model where it has not; or one of the two.
AUTO, GEM15, COPERNICUS = "auto", "gem15", "copernicus"
SOURCES = (
    (AUTO, "Best available: terrain (Google GEM-15, 15 m), "
           "else surface (Copernicus, 30 m)"),
    (GEM15, "Terrain model: Google GEM-15 (15 m)"),
    (COPERNICUS, "Surface model: Copernicus DEM (30 m)"),
)
#: The files a flight's elevation model is written as, by what it is from.
STEMS = {GEM15: "gem15_dem", COPERNICUS: "copernicus_dem"}
#: The files of the Copernicus DEM (kept for what was built before 9.1).
STEM = STEMS[COPERNICUS]
#: What of a source's description is kept with the model.
_SOURCE_KEYS = (
    "product", "editions", "tiles", "attribution", "vertical_datum",
    "surface_model", "sea_share", "levelled_by_m", "open_ground_cells",
    "filled_share", "filled_from")
#: The mesh generator takes a height of exactly nought for "no data" and
#: leaves a hole there - which is what the sea would be. A centimetre up it
#: is ground like any other.
SEA_LEVEL_M = 0.01


def _heights(bounds, epsg: int, padding_m: float, source: str, log,
             progress, cancelled, fetch, fetch_terrain):
    """``(kind, grid, heights, description)`` of the model *source* asks
    for: the terrain model's where it has the flight, else - unless only
    the terrain model will do - the surface model's."""
    from .core import copernicus_dem, gem15_dem

    surface = fetch or copernicus_dem.fetch
    if source in (AUTO, GEM15):
        grid = gem15_dem.grid_for(bounds, epsg, padding_m)
        try:
            heights, description = (fetch_terrain or gem15_dem.fetch)(
                grid, reference=surface, log=log, cancelled=cancelled,
                progress=progress)
            return GEM15, grid, heights, description
        except gem15_dem.NoTerrain as exc:
            if source == GEM15:
                raise
            if log:
                log(f"{exc} The Copernicus DEM is used instead: a surface "
                    "model at 30 m, with the canopy of a forest in it.")
    grid = copernicus_dem.grid_for(bounds, epsg, padding_m)
    heights, description = surface(
        grid, log=log, cancelled=cancelled, progress=progress)
    return COPERNICUS, grid, heights, description


def build_dem(bounds_wgs84: Tuple[float, float, float, float],
              output_folder: str, output_crs: str,
              padding_m: float = 150.0,
              log: Optional[Callable[[str], None]] = None,
              progress: Optional[Callable[[int], None]] = None,
              cancelled: Optional[Callable[[], bool]] = None,
              fetch: Optional[Callable] = None,
              source: str = AUTO,
              fetch_terrain: Optional[Callable] = None) -> dict:
    """Build the elevation model of a flight from an open worldwide model.

    :param bounds_wgs84: ``(west, south, east, north)`` of the flight's
        positions, degrees
    :param output_folder: where ``gem15_dem.tif``, ``.glb`` and ``.json``
        are written - ``copernicus_dem.*`` where the heights are the
        Copernicus DEM's
    :param output_crs: the UTM CRS of the mesh, e.g. ``"EPSG:32633"``
    :param padding_m: ground kept around the positions (never less than
        150 m)
    :param source: :data:`AUTO` - the terrain model (GEM-15) where it has
        the flight, else the surface model (Copernicus); :data:`GEM15` or
        :data:`COPERNICUS` for one of them and nothing else
    :param fetch: what reads the Copernicus DEM -
        :func:`core.copernicus_dem.fetch` unless given
    :param fetch_terrain: what reads GEM-15 - :func:`core.gem15_dem.fetch`
        unless given
    :return: ``{"mesh", "metadata", "geotiff", "source", "grid", "kind"}``
        - the three files, what the heights were read from, the grid they
        are on, and which of the two models it is
    :raises core.environment.EnvironmentFailure: no heights there, or the
        service did not answer
    :raises core.environment.Cancelled: stopped by the user
    """
    import numpy as np

    from .austria_dem_downloader import BoundingBox, GLTFMeshGenerator
    from .core import environment
    from .core.crs_utils import is_valid_utm_crs, parse_epsg_from_text

    if source not in (AUTO, GEM15, COPERNICUS):
        raise environment.EnvironmentFailure(
            f"'{source}' is no elevation source: {AUTO}, {GEM15} or "
            f"{COPERNICUS}.")
    if not is_valid_utm_crs(output_crs):
        raise environment.EnvironmentFailure(
            f"'{output_crs}' is not a UTM CRS: the elevation model needs "
            "one (EPSG:32601-32660 north, EPSG:32701-32760 south).")
    epsg = parse_epsg_from_text(output_crs)
    west, south, east, north = (float(value) for value in bounds_wgs84)
    bounds = BoundingBox(south, west, north, east).to_crs(f"EPSG:{epsg}")

    def scaled(value: int) -> None:
        if progress:
            progress(int(5 + value * 0.6))

    kind, grid, heights, source_description = _heights(
        bounds, epsg, padding_m, source, log, scaled, cancelled, fetch,
        fetch_terrain)
    if log:
        log(f"Elevation for {grid.width * grid.resolution / 1000:.1f} x "
            f"{grid.height * grid.resolution / 1000:.1f} km around the "
            f"flight, in cells of {grid.resolution:g} m, EPSG:{epsg}")
    source = source_description
    environment.check_cancelled(cancelled)
    heights = np.where(heights == 0, np.float32(SEA_LEVEL_M),
                       heights).astype(np.float32)

    os.makedirs(output_folder, exist_ok=True)
    base = Path(output_folder) / STEMS[kind]
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
        key: source[key] for key in _SOURCE_KEYS if key in source}
    document["source"]["resolution_m"] = grid.resolution
    with open(metadata, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=4)
    if progress:
        progress(100)
    return {"mesh": str(mesh), "metadata": str(metadata),
            "geotiff": str(geotiff), "source": source, "grid": grid,
            "kind": kind}


def describe(metadata_path: str) -> str:
    """What a model that :func:`build_dem` made is, in a few sentences for
    whoever just had it built - read from the metadata written with it.
    ``""`` where that cannot be read."""
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            source = json.load(handle)["source"]
        product = str(source["product"])
    except (OSError, ValueError, KeyError, TypeError):
        return ""
    size = source.get("resolution_m")
    cells = f" at {size:g} m" if isinstance(size, (int, float)) else ""
    if source.get("surface_model"):
        return (
            f"It was made from the {product}: a surface model{cells}. Over "
            "forest it lies in the canopy - allow for that under 'Position "
            "Uncertainty' (Terrain model).")
    lines = [f"It was made from the {product}: the bare ground{cells}."]
    if "levelled_by_m" in source and source["levelled_by_m"] is None:
        lines.append(
            "It could not be levelled - too little open ground under the "
            "flight - so its heights may lie a few metres low or high.")
    filled = source.get("filled_share") or 0.0
    if filled >= 0.005:
        lines.append(
            f"{filled * 100:.0f} % of it had no value (water, snow) and is "
            f"taken from {source.get('filled_from') or 'the surface model'}.")
    lines.append(
        "Its heights are good to a few metres, under trees less so: set "
        "what you take them to be under 'Position Uncertainty' (Terrain "
        "model).")
    return " ".join(lines)


class CopernicusDEMWorker(QObject):
    """Builds a flight's elevation model from GEM-15 or the Copernicus DEM
    in a background thread."""

    finished = pyqtSignal(bool, str)    # success, mesh path or message
    progress = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(self, bounds_wgs84, output_folder: str, output_crs: str,
                 padding: float = 150.0, source: str = AUTO):
        super().__init__()
        self.bounds_wgs84 = tuple(bounds_wgs84)
        self.output_folder = output_folder
        self.output_crs = output_crs
        self.padding = padding
        self.source = source
        self._cancelled = False

    def cancel(self):
        """Stop at the next tile."""
        self._cancelled = True

    def run(self):
        from .core import environment

        try:
            self.log.emit("Fetching the elevation under the flight...")
            built = build_dem(
                self.bounds_wgs84, self.output_folder, self.output_crs,
                self.padding, log=self.log.emit, progress=self.progress.emit,
                cancelled=lambda: self._cancelled, source=self.source)
        except environment.Cancelled:
            self.finished.emit(False, "Cancelled")
            return
        except environment.EnvironmentFailure as exc:
            self.log.emit(str(exc))
            self.finished.emit(False, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - told to the user, whole
            import traceback

            logger.error("Worldwide elevation: %s", exc)
            traceback.print_exc()
            self.finished.emit(
                False, f"The elevation model could not be made: {exc}")
            return
        self.log.emit(f"Elevation model written: {built['mesh']} "
                      f"({built['source'].get('product', '')})")
        self.finished.emit(True, built["mesh"])
