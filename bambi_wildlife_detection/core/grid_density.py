# -*- coding: utf-8 -*-
"""Animal density cell by cell.

The survey's ground is cut into equal cells (:mod:`core.survey_grid`), and
for every cell the question is the census's, asked small: how many animals,
on how much ground.

**Density is animals over ground that was seen.** A cell the drone crossed
only at one corner holds few animals because little of it was looked at, not
because few are there. So the density of a cell is its animals divided by
the ground the cameras saw *in the cell* - the field-of-view footprints,
inside the survey area - not by the size of the cell.

**A cell that was hardly seen says nothing.** One animal on the sliver of a
cell that was seen makes a density no survey would stand behind. Cells of
which less than a share (:data:`DEFAULT_MIN_SEEN`, half) was seen are kept on
the map as *too little seen*, with what was found there, and left out of
every statistic. The share is of the whole cell, so a cell that the survey
area cuts down to a corner is treated the same way.

**Ground flown more than once.** Two ways to read a cell several flights saw
(:data:`REPEAT_MODES`):

* ``average`` - all sightings over all ground looked at: ``sum(animals) /
  sum(ground seen)``, the mean density over the visits. Right for flights
  over neighbouring ground and for repeats that are to be averaged.
* ``highest`` - the flight with the highest density: what was there at
  least, on the visit that found most. Only flights that saw enough of the
  cell themselves are compared.

**What comes out** (:func:`analyse`): one record per cell - counts and
density, per species, by sex and age, the groups (:mod:`core.groups`) among
its animals, whether it stands out with its neighbours (Getis-Ord Gi*) - one
per cell and flight, and the statistics over the cells that count
(:mod:`core.spatial_stats`): the mean density with an interval from
resampling the cells, the total for the survey area, how clumped the animals
are, whether neighbouring cells are alike, and the share of cells that held
any animal.

shapely for the geometry, numpy for the statistics; no QGIS.
"""

import csv
import json
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence

from . import spatial_stats, survey_grid

AVERAGE = "average"
HIGHEST = "highest"
REPEAT_MODES = (AVERAGE, HIGHEST)
REPEAT_LABELS = {AVERAGE: "Average over the flights",
                 HIGHEST: "Highest flight"}

#: A cell counts when at least this share of it was seen.
DEFAULT_MIN_SEEN = 0.5

COUNTED = "counted"
TOO_LITTLE = "too little seen"
NOT_SEEN = "not seen"

ALL = "all"
UNKNOWN = "unknown"


class GridDensityError(RuntimeError):
    """The grid cannot be analysed as it was asked for."""


def slug(name: str) -> str:
    """A species name as part of a column name: ``'roe deer'`` ->
    ``'roe_deer'``."""
    keep = [c.lower() if c.isalnum() else "_" for c in (name or "")]
    text = "".join(keep).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return text or "species"


def _label(value) -> str:
    text = "" if value is None else str(value).strip()
    return text.lower() if text else UNKNOWN


def _per_km2(animals: float, ground_m2: float) -> Optional[float]:
    return animals / (ground_m2 / 1e6) if ground_m2 > 0 else None


# ---------------------------------------------------------------------------
# One flight in one cell
# ---------------------------------------------------------------------------

def _empty_visit() -> dict:
    return {"animals": 0, "species": {}, "sex": {}, "age": {},
            "groups": set(), "ungrouped": 0}


def _add(visit: dict, record: dict) -> None:
    visit["animals"] += 1
    species = record.get("species") or UNKNOWN
    visit["species"][species] = visit["species"].get(species, 0) + 1
    for key in ("sex", "age"):
        label = _label(record.get(key))
        visit[key][label] = visit[key].get(label, 0) + 1
    if record.get("group") is None:
        visit["ungrouped"] += 1
    else:
        visit["groups"].add(record["group"])


def _groups_of(visit: dict) -> Optional[int]:
    """How many groups the animals of a visit formed; ``None`` where an
    animal came without one."""
    return None if visit["ungrouped"] else len(visit["groups"])


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------

def analyse(spec: survey_grid.GridSpec, flights: Sequence[dict], region,
            min_seen: float = DEFAULT_MIN_SEEN, repeat: str = AVERAGE,
            plot_m2: Optional[float] = None,
            species_order: Optional[Sequence[str]] = None,
            progress: Optional[Callable[[int], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None) -> dict:
    """Count a survey cell by cell.

    :param flights: ``{"name", "records", "seen"}`` per flight - the animals
        it counts (``{"species", "sex", "age", "group", "x", "y"}``; one
        without a place is in no cell) and the ground its camera saw, as a
        shapely geometry
    :param region: the ground the grid is laid over, as a shapely geometry:
        the survey area, or all the ground the flights saw
    :param min_seen: the share of a cell that has to have been seen for it
        to count (0 to 1)
    :param repeat: how a cell several flights saw is read
        (:data:`REPEAT_MODES`)
    :param plot_m2: the size of the survey area, for the total; ``None``
        without one
    :param species_order: the order the species are listed in (others
        follow)
    :return: the result document - see :func:`flat_cells`,
        :func:`flat_flights` and the ``"statistics"`` it holds
    :raises GridDensityError: when there is no ground to lay a grid over
    """
    import shapely
    from shapely.geometry import Polygon

    spec.validate()
    if repeat not in REPEAT_MODES:
        raise GridDensityError(
            f"Unknown way to read repeated flights {repeat!r}; expected "
            f"{' or '.join(REPEAT_MODES)}.")
    min_seen = min(1.0, max(0.0, float(min_seen)))
    if region is None or region.is_empty or region.area <= 0:
        raise GridDensityError(
            "There is no ground to lay a grid over: no survey area is set "
            "and the flights have no field-of-view footprints.")
    if not flights:
        raise GridDensityError("No flight to analyse.")

    # ---- the cells that touch the region --------------------------------- #
    try:
        candidates = survey_grid.cells(spec, region.bounds)
    except survey_grid.GridError as exc:
        raise GridDensityError(str(exc)) from exc
    polygons = [Polygon(cell.ring) for cell in candidates]
    full_m2 = spec.cell_area_m2
    # A cell belongs to the grid when some of it lies on the ground - not
    # when it only touches the border from outside.
    shared = _shared_area(polygons, region)
    keep = [area > 1e-9 * full_m2 for area in shared]
    grid = [cell for cell, kept in zip(candidates, keep) if kept]
    polygons = [polygon for polygon, kept in zip(polygons, keep) if kept]
    in_region = [area for area, kept in zip(shared, keep) if kept]
    if not grid:
        raise GridDensityError("The grid has no cell on the survey's ground.")

    def step(done: int, of: int, low: int, high: int) -> None:
        if cancelled and cancelled():
            raise InterruptedError("Grid analysis cancelled")
        if progress:
            progress(low + int((high - low) * done / max(1, of)))

    # ---- how much of each cell was seen ----------------------------------- #
    seen_all = shapely.unary_union(
        [flight["seen"] for flight in flights
         if flight.get("seen") is not None and not flight["seen"].is_empty])
    seen_all = seen_all.intersection(region)
    seen_union = _shared_area(polygons, seen_all)
    step(1, 4, 5, 40)

    tree = shapely.STRtree(polygons)
    visits: List[Dict[str, dict]] = [OrderedDict() for _ in grid]
    seen_by: List[Dict[str, float]] = [OrderedDict() for _ in grid]
    outside = unplaced = placed = 0
    for number, flight in enumerate(flights):
        name = flight["name"]
        ground = flight.get("seen")
        if ground is not None and not ground.is_empty:
            ground = ground.intersection(region)
            for index, area in enumerate(_shared_area(polygons, ground)):
                if area > 0:
                    seen_by[index][name] = area
                    visits[index][name] = _empty_visit()
        records = [r for r in flight.get("records") or []]
        located = [r for r in records
                   if r.get("x") is not None and r.get("y") is not None]
        unplaced += len(records) - len(located)
        if located:
            points = shapely.points(
                [(float(r["x"]), float(r["y"])) for r in located])
            pairs = tree.query(points, predicate="intersects")
            cell_of: Dict[int, int] = {}
            for point_index, cell_index in zip(pairs[0].tolist(),
                                               pairs[1].tolist()):
                # on an edge two cells have it: the first one keeps it
                if point_index not in cell_of or cell_index < cell_of[point_index]:
                    cell_of[point_index] = cell_index
            inside = shapely.intersects(region, points)
            for point_index, record in enumerate(located):
                cell_index = cell_of.get(point_index)
                if cell_index is None or not inside[point_index]:
                    outside += 1
                    continue
                placed += 1
                visit = visits[cell_index].setdefault(name, _empty_visit())
                _add(visit, record)
        step(number + 1, len(flights), 40, 70)

    # ---- one record per cell --------------------------------------------- #
    seen_species: List[str] = [name for name in (species_order or [])]
    cells: List[dict] = []
    by_flight: List[dict] = []
    for index, cell in enumerate(grid):
        of_cell = visits[index]
        for name, visit in of_cell.items():
            for species in visit["species"]:
                if species not in seen_species:
                    seen_species.append(species)
            ground = seen_by[index].get(name, 0.0)
            by_flight.append({
                "cell": cell.name, "flight": name,
                "seen_m2": ground, "seen_share": ground / full_m2,
                "animals": visit["animals"],
                "density_per_km2": _per_km2(visit["animals"], ground),
                "species": dict(visit["species"]),
                "groups": _groups_of(visit),
            })
        record = _combine(of_cell, seen_by[index], full_m2, min_seen, repeat,
                          seen_union[index])
        share = seen_union[index] / full_m2
        record.update({
            "cell": cell.name, "col": cell.col, "row": cell.row,
            "x": cell.centre[0], "y": cell.centre[1],
            "ring": [[x, y] for x, y in cell.ring],
            "area_m2": full_m2,
            "in_area_m2": in_region[index],
            "seen_m2": seen_union[index],
            "seen_share": share,
            "flights": len(seen_by[index]),
        })
        cells.append(record)
    used = {species for cell in cells for species in cell["species"]}
    species_names = [name for name in seen_species if name in used]
    step(1, 1, 70, 80)

    # ---- the statistics over the cells that count ------------------------ #
    counted = [cell for cell in cells if cell["status"] == COUNTED]
    near = survey_grid.neighbours(
        [c for c in grid if c.name in {cell["cell"] for cell in counted}])
    links = spatial_stats.index_neighbours(
        [cell["cell"] for cell in counted], near)
    for cell in cells:
        cell["gi_z"], cell["spot"] = None, ""
        for species in species_names:
            entry = cell["species"].setdefault(species, {
                "animals": 0, "effort_m2": cell["effort_m2"],
                "density_per_km2": 0.0 if cell["effort_m2"] > 0 else None})
            entry["gi_z"], entry["spot"] = None, ""
    statistics = OrderedDict()
    statistics[ALL], scores = _statistics(
        counted, links, plot_m2,
        lambda cell: (cell["animals"], cell["effort_m2"]))
    for cell, z in zip(counted, scores):
        cell["gi_z"], cell["spot"] = z, spatial_stats.spot(z)
    for species in species_names:
        statistics[species], scores = _statistics(
            counted, links, plot_m2,
            lambda cell, s=species: (cell["species"][s]["animals"],
                                     cell["species"][s]["effort_m2"]))
        for cell, z in zip(counted, scores):
            cell["species"][species]["gi_z"] = z
            cell["species"][species]["spot"] = spatial_stats.spot(z)
    step(1, 1, 80, 95)

    region_m2 = float(region.area)
    return {
        "grid": spec.to_dict(),
        "grid_text": spec.describe(),
        "min_seen": min_seen,
        "repeat": repeat,
        "flights": [flight["name"] for flight in flights],
        "species": species_names,
        "sex_labels": _labels(cells, "sex"),
        "age_labels": _labels(cells, "age"),
        "summary": {
            "cells": len(cells),
            "counted": len(counted),
            "too_little_seen": sum(1 for c in cells
                                   if c["status"] == TOO_LITTLE),
            "not_seen": sum(1 for c in cells if c["status"] == NOT_SEEN),
            "region_ha": region_m2 / 1e4,
            "seen_ha": float(seen_all.area) / 1e4,
            "seen_share": (float(seen_all.area) / region_m2
                           if region_m2 > 0 else None),
            "plot_ha": None if plot_m2 is None else plot_m2 / 1e4,
            "animals_in_cells": placed,
            "animals_in_counted_cells": sum(c["animals"] for c in counted),
            "animals_outside": outside,
            "animals_unplaced": unplaced,
        },
        "statistics": statistics,
        "cells": cells,
        "by_flight": by_flight,
    }


def _shared_area(polygons, geometry) -> List[float]:
    """The area each cell shares with *geometry*."""
    import shapely

    if geometry is None or geometry.is_empty:
        return [0.0] * len(polygons)
    shapely.prepare(geometry)
    areas = [0.0] * len(polygons)
    touching = shapely.intersects(geometry, polygons)
    within = shapely.contains_properly(geometry, polygons)
    for index, polygon in enumerate(polygons):
        if within[index]:
            areas[index] = float(polygon.area)
        elif touching[index]:
            areas[index] = float(polygon.intersection(geometry).area)
    return areas


def _combine(visits: Dict[str, dict], seen: Dict[str, float], full_m2: float,
             min_seen: float, repeat: str, union_m2: float) -> dict:
    """What a cell holds, from what each flight found in it.

    *seen* is the ground each flight saw in the cell, *union_m2* the ground
    any of them saw.
    """
    record = {"status": NOT_SEEN, "animals": 0, "effort_m2": 0.0,
              "density_per_km2": None, "species": {}, "sex": {}, "age": {},
              "groups": None, "mean_group_size": None, "best_flight": None}
    names = [name for name in visits if seen.get(name, 0.0) > 0]

    def everything(of: Sequence[str]) -> None:
        """All sightings of the flights *of*, added up."""
        groups, ungrouped = 0, False
        for name in of:
            visit = visits[name]
            record["animals"] += visit["animals"]
            for species, n in visit["species"].items():
                entry = record["species"].setdefault(species, {"animals": 0})
                entry["animals"] += n
            for key in ("sex", "age"):
                for label, n in visit[key].items():
                    record[key][label] = record[key].get(label, 0) + n
            found = _groups_of(visit)
            if found is None:
                ungrouped = True
            else:
                groups += found
        record["groups"] = None if ungrouped else groups

    if not names:
        # Animals on ground nobody is known to have seen: counted where
        # they are, never part of a density.
        everything(list(visits))
        for entry in record["species"].values():
            entry.update({"effort_m2": 0.0, "density_per_km2": None})
    elif repeat == HIGHEST:
        enough = [name for name in names
                  if seen[name] / full_m2 >= min_seen - 1e-12]
        pool = enough or names

        def density(name, species=None):
            visit = visits[name]
            n = visit["animals"] if species is None else \
                visit["species"].get(species, 0)
            return n / seen[name]

        # the highest density; of equals, the flight that comes first
        best = max(pool, key=lambda name: (density(name), -pool.index(name)))
        chosen = visits[best]
        record.update({
            "animals": chosen["animals"], "effort_m2": seen[best],
            "sex": dict(chosen["sex"]), "age": dict(chosen["age"]),
            "groups": _groups_of(chosen), "best_flight": best})
        for species in {s for name in pool for s in visits[name]["species"]}:
            top = max(pool, key=lambda name, s=species: (
                density(name, s), -pool.index(name)))
            n = visits[top]["species"].get(species, 0)
            record["species"][species] = {
                "animals": n, "effort_m2": seen[top],
                "density_per_km2": _per_km2(n, seen[top])}
        record["status"] = COUNTED if enough else TOO_LITTLE
    else:
        effort = sum(seen[name] for name in names)
        everything(list(visits))
        record["effort_m2"] = effort
        for entry in record["species"].values():
            entry.update({"effort_m2": effort, "density_per_km2": _per_km2(
                entry["animals"], effort)})
        record["status"] = COUNTED \
            if union_m2 / full_m2 >= min_seen - 1e-12 else TOO_LITTLE

    record["density_per_km2"] = _per_km2(record["animals"],
                                         record["effort_m2"])
    if record["groups"]:
        record["mean_group_size"] = record["animals"] / record["groups"]
    return record


def _statistics(counted: Sequence[dict], links, plot_m2, pick):
    """The statistics of one kind of animal over the counted cells, and the
    Gi* score of each of them; *pick* gives a cell's ``(animals, ground in
    m²)``."""
    animals = [pick(cell)[0] for cell in counted]
    ground_km2 = [pick(cell)[1] / 1e6 for cell in counted]
    density = spatial_stats.ratio_interval(animals, ground_km2)
    values = [n / g if g > 0 else 0.0 for n, g in zip(animals, ground_km2)]
    scores = spatial_stats.gi_star(values, links)
    occupied = sum(1 for n in animals if n > 0)
    low, high = spatial_stats.wilson(occupied, len(counted))
    total = None
    if plot_m2 is not None and density["estimate"] is not None:
        scale = plot_m2 / 1e6

        def scaled(value):
            return None if value is None else value * scale

        total = {"estimate": scaled(density["estimate"]),
                 "low": scaled(density["low"]),
                 "high": scaled(density["high"])}
    return {
        "animals": int(sum(animals)),
        "cells": len(counted),
        "density_per_km2": density,
        "total_in_survey_area": total,
        "dispersion": spatial_stats.dispersion(animals, ground_km2),
        "morans_i": spatial_stats.morans_i(values, links),
        "occupancy": {
            "occupied": occupied, "cells": len(counted),
            "share": occupied / len(counted) if counted else None,
            "low": low, "high": high},
        "hot_spots": sum(1 for z in scores
                         if spatial_stats.spot(z).startswith("hot")),
        "cold_spots": sum(1 for z in scores
                          if spatial_stats.spot(z).startswith("cold")),
    }, scores


def _labels(cells: Sequence[dict], key: str) -> List[str]:
    found = {label for cell in cells for label in cell[key]}
    labels = sorted(found - {UNKNOWN})
    if UNKNOWN in found:
        labels.append(UNKNOWN)
    return labels


# ---------------------------------------------------------------------------
# Environment values per cell
# ---------------------------------------------------------------------------

def zonal_by_cells(array, grid, rings: Sequence[Sequence], categorical: bool,
                   nodata=0) -> List[Optional[tuple]]:
    """The value of one environment raster in every cell.

    The cells are drawn into the raster once and the pixels of each counted
    out - looking at the whole raster for every cell would take minutes on a
    grid of a few thousand.

    :param grid: the raster's :class:`core.environment.Grid`
    :param rings: the corners of each cell
    :return: per cell ``(mean,)`` for a continuous layer, ``(class, share)``
        - the most common class and its share of the cell - for a
        categorical one; ``None`` where the cell holds no pixel with a value
    """
    import numpy as np
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    array = np.asarray(array)
    shapes = [({"type": "Polygon",
                "coordinates": [[list(p) for p in ring] + [list(ring[0])]]},
               index + 1) for index, ring in enumerate(rings)]
    if not shapes:
        return []
    labels = rasterize(shapes, out_shape=grid.shape, fill=0, dtype="int32",
                       transform=Affine(*grid.geotransform))
    found: List[Optional[tuple]] = [None] * len(rings)
    if categorical:
        valid = (labels > 0) & (array != nodata)
        if not valid.any():
            return found
        cells = labels[valid].astype(np.int64)
        classes = array[valid].astype(np.int64)
        span = int(classes.max()) + 1
        pairs, counts = np.unique(cells * span + classes, return_counts=True)
        totals = np.bincount(cells, minlength=len(rings) + 1)
        best: Dict[int, tuple] = {}
        for pair, count in zip(pairs.tolist(), counts.tolist()):
            cell, value = divmod(pair, span)
            if cell not in best or count > best[cell][1]:
                best[cell] = (value, count)
        for cell, (value, count) in best.items():
            found[cell - 1] = (value, count / float(totals[cell]))
        return found
    values = array.astype(np.float64)
    valid = (labels > 0) & np.isfinite(values)
    if not valid.any():
        return found
    cells = labels[valid].astype(np.int64)
    sums = np.bincount(cells, weights=values[valid], minlength=len(rings) + 1)
    counts = np.bincount(cells, minlength=len(rings) + 1)
    for cell in np.nonzero(counts)[0].tolist():
        if cell:
            found[cell - 1] = (float(sums[cell] / counts[cell]),)
    return found


def add_environment(document: dict, layer_sets: Sequence[dict],
                    class_name=None) -> List[str]:
    """Give every cell the environment it lies in.

    :param layer_sets: per flight ``{layer key: (array, grid)}``
        (:func:`core.habitat.load_layers`), in the order of the flights. A
        cell takes each layer from the first flight whose raster has a value
        there.
    :param class_name: turns a class value of a categorical layer into its
        name
    :return: the keys of the layers that were found, in the order of
        :data:`core.environment.LAYERS`
    """
    from . import environment

    cells = document["cells"]
    rings = [cell["ring"] for cell in cells]
    for cell in cells:
        cell["environment"] = {}
    keys = []
    for layer in environment.LAYERS:
        categorical = layer.kind == environment.CATEGORICAL
        for layers in layer_sets:
            if layer.key not in layers:
                continue
            array, grid = layers[layer.key]
            values = zonal_by_cells(array, grid, rings, categorical)
            for cell, value in zip(cells, values):
                if value is None or layer.key in cell["environment"]:
                    continue
                if categorical:
                    name = class_name(value[0]) if class_name else value[0]
                    cell["environment"][layer.key] = name
                    cell["environment"][f"{layer.key}_share"] = round(
                        value[1], 3)
                else:
                    cell["environment"][layer.key] = round(value[0], 4)
            if layer.key not in keys:
                keys.append(layer.key)
    document["environment_layers"] = keys
    return keys


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _ha(value_m2) -> Optional[float]:
    return None if value_m2 is None else value_m2 / 1e4


def flat_cells(document: dict) -> List[OrderedDict]:
    """One row per cell, every value in a column of its own - the cells
    table, and the attributes of the layer on the map."""
    species = document.get("species", [])
    layers = document.get("environment_layers", [])
    rows = []
    for cell in document["cells"]:
        row = OrderedDict([
            ("cell", cell["cell"]), ("status", cell["status"]),
            ("animals", cell["animals"]),
            ("density_per_km2", cell["density_per_km2"]),
            ("groups", cell["groups"]),
            ("mean_group_size", cell["mean_group_size"]),
            ("seen_share", cell["seen_share"]),
            ("seen_ha", _ha(cell["seen_m2"])),
            ("effort_ha", _ha(cell["effort_m2"])),
            ("flights", cell["flights"]),
            ("best_flight", cell.get("best_flight")),
            ("area_ha", _ha(cell["area_m2"])),
            ("in_area_ha", _ha(cell["in_area_m2"])),
            ("gi_z", cell.get("gi_z")), ("spot", cell.get("spot") or ""),
        ])
        for name in species:
            entry = cell["species"].get(name) or {}
            key = slug(name)
            row[f"n_{key}"] = entry.get("animals", 0)
            row[f"per_km2_{key}"] = entry.get("density_per_km2")
            row[f"spot_{key}"] = entry.get("spot") or ""
        for label in document.get("sex_labels", []):
            row[f"sex_{slug(label)}"] = cell["sex"].get(label, 0)
        for label in document.get("age_labels", []):
            row[f"age_{slug(label)}"] = cell["age"].get(label, 0)
        environment = cell.get("environment") or {}
        for key in layers:
            row[key] = environment.get(key)
            if f"{key}_share" in environment:
                row[f"{key}_share"] = environment[f"{key}_share"]
        row["x"], row["y"] = cell["x"], cell["y"]
        rows.append(row)
    # every row the same columns, also where a value is missing
    columns: List[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    return [OrderedDict((column, row.get(column)) for column in columns)
            for row in rows]


def flat_flights(document: dict) -> List[OrderedDict]:
    """One row per cell and flight that saw it or found an animal in it."""
    species = document.get("species", [])
    rows = []
    for visit in document["by_flight"]:
        row = OrderedDict([
            ("cell", visit["cell"]), ("flight", visit["flight"]),
            ("animals", visit["animals"]),
            ("density_per_km2", visit["density_per_km2"]),
            ("groups", visit["groups"]),
            ("seen_ha", _ha(visit["seen_m2"])),
            ("seen_share", visit["seen_share"]),
        ])
        for name in species:
            row[f"n_{slug(name)}"] = visit["species"].get(name, 0)
        rows.append(row)
    return rows


def _cell_text(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 4)
    return value


def write_table(rows: Sequence[dict], path: str) -> str:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if rows:
            columns = list(rows[0].keys())
            writer.writerow(columns)
            for row in rows:
                writer.writerow([_cell_text(row[column])
                                 for column in columns])
    return path


def write_json(document: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle)
    return path


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=OrderedDict)


# ---------------------------------------------------------------------------
# In words
# ---------------------------------------------------------------------------

PATTERN_TEXT = {
    spatial_stats.CLUSTERED: "clumped",
    spatial_stats.RANDOM: "scattered at random",
    spatial_stats.REGULAR: "evenly spread",
}


def _interval(entry: Optional[dict], digits: int = 1) -> str:
    if not entry or entry.get("estimate") is None:
        return "-"
    text = f"{entry['estimate']:.{digits}f}"
    if entry.get("low") is not None:
        text += f" ({entry['low']:.{digits}f}-{entry['high']:.{digits}f})"
    return text


def describe(document: dict) -> List[str]:
    """The result in a few lines, for the log."""
    summary = document["summary"]
    lines = [
        f"Grid: {document['grid_text']} - {summary['cells']} cell(s), "
        f"{summary['counted']} counted, {summary['too_little_seen']} too "
        f"little seen, {summary['not_seen']} not seen"]
    if summary.get("seen_share") is not None:
        lines.append(
            f"{summary['seen_ha']:.1f} of {summary['region_ha']:.1f} ha "
            f"were seen ({summary['seen_share'] * 100:.0f} %)")
    left_out = summary["animals_outside"] + summary["animals_unplaced"]
    if left_out:
        lines.append(
            f"{summary['animals_outside']} animal(s) outside the grid's "
            f"ground and {summary['animals_unplaced']} without a position "
            "are in no cell")
    for name, stats in document["statistics"].items():
        if not stats["cells"]:
            continue
        what = "All animals" if name == ALL else name
        text = (f"{what}: {stats['animals']} in the counted cells, "
                f"{_interval(stats['density_per_km2'])} /km²")
        if stats.get("total_in_survey_area"):
            text += (f", {_interval(stats['total_in_survey_area'], 0)} in "
                     "the survey area")
        pattern = stats["dispersion"].get("pattern")
        if pattern:
            text += (f"; {PATTERN_TEXT[pattern]} (variance-to-mean "
                     f"{stats['dispersion']['index']:.2f})")
        occupancy = stats["occupancy"]
        if occupancy["share"] is not None:
            text += (f"; in {occupancy['occupied']} of {occupancy['cells']} "
                     "cells")
        if stats["hot_spots"]:
            text += f"; {stats['hot_spots']} hot-spot cell(s)"
        lines.append(text)
    return lines
