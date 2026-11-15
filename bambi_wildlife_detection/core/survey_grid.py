# -*- coding: utf-8 -*-
"""A grid over the survey: equal cells to count animals in.

Transects describe a survey along the way it was flown. A grid describes it
by the ground: the monitored area is cut into cells of one size and shape,
and what was found is looked at cell by cell - how dense, how clumped, where
the hot spots are (:mod:`core.grid_density`).

**Shapes.** Hexagons, squares, equilateral triangles - the three regular
shapes that cover a plane without gaps - and rectangles with a width and a
height of their own. Hexagons are the default: every neighbour of a hexagon
shares a whole edge with it and is equally far away, which is what the
neighbourhood statistics assume.

**Size.** Given as the area of a cell or as the length of its edge,
whichever is easier to think in (:func:`edge_for_area`); a rectangle is
given by its sides.

**The same ground is the same cell, in every survey.** The grid is not laid
from a corner of whatever is analysed today. It is laid from the origin of
the map's coordinate system: a square cell of 100 m always runs from a
multiple of 100 to the next. Two surveys of the same area, or one survey
analysed again after a flight was added, have the same cells with the same
names - which is what lets them be compared.

Pure geometry: no QGIS, no shapely.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

HEXAGON = "hexagon"
SQUARE = "square"
TRIANGLE = "triangle"
RECTANGLE = "rectangle"
SHAPES = (HEXAGON, SQUARE, TRIANGLE, RECTANGLE)
SHAPE_LABELS = {HEXAGON: "Hexagons", SQUARE: "Squares",
                TRIANGLE: "Triangles", RECTANGLE: "Rectangles"}

#: A cell's size is given by its area, or by the length of its edge.
BY_AREA = "area"
BY_EDGE = "edge"
SIZE_MODES = (BY_AREA, BY_EDGE)

#: One hectare.
DEFAULT_AREA_M2 = 10000.0
#: The smallest edge a cell may have: below a metre the grid is finer than
#: the positions it sorts.
MIN_EDGE_M = 1.0
#: More cells than this is a typing error, not a grid.
MAX_CELLS = 250000

_SQRT3 = math.sqrt(3.0)

Point = Tuple[float, float]


class GridError(ValueError):
    """The grid cannot be made as it was asked for."""


def edge_for_area(shape: str, area_m2: float) -> float:
    """The edge of a regular cell of *shape* that has *area_m2*."""
    area_m2 = float(area_m2)
    if shape == HEXAGON:
        return math.sqrt(2.0 * area_m2 / (3.0 * _SQRT3))
    if shape == TRIANGLE:
        return math.sqrt(4.0 * area_m2 / _SQRT3)
    if shape in (SQUARE, RECTANGLE):
        return math.sqrt(area_m2)
    raise GridError(f"Unknown cell shape {shape!r}.")


def area_for_edge(shape: str, edge_m: float) -> float:
    """The area of a regular cell of *shape* with edges of *edge_m*."""
    edge_m = float(edge_m)
    if shape == HEXAGON:
        return 1.5 * _SQRT3 * edge_m * edge_m
    if shape == TRIANGLE:
        return _SQRT3 / 4.0 * edge_m * edge_m
    if shape in (SQUARE, RECTANGLE):
        return edge_m * edge_m
    raise GridError(f"Unknown cell shape {shape!r}.")


@dataclass
class GridSpec:
    """How the grid is cut: the shape of a cell and its size."""

    shape: str = HEXAGON
    size_mode: str = BY_AREA
    area_m2: float = DEFAULT_AREA_M2
    edge_m: float = 100.0
    #: The sides of a rectangle (east-west, north-south).
    width_m: float = 100.0
    height_m: float = 100.0

    def validate(self) -> "GridSpec":
        if self.shape not in SHAPES:
            raise GridError(
                f"Unknown cell shape {self.shape!r}; expected one of "
                f"{', '.join(SHAPES)}.")
        if self.shape == RECTANGLE:
            sides = (self.width_m, self.height_m)
        else:
            if self.size_mode not in SIZE_MODES:
                raise GridError(
                    f"A cell's size is given by its {BY_AREA} or its "
                    f"{BY_EDGE}, not {self.size_mode!r}.")
            sides = (self.edge,)
        for side in sides:
            if not (isinstance(side, (int, float)) and math.isfinite(side)):
                raise GridError("The size of a cell has to be a number.")
            if side < MIN_EDGE_M:
                raise GridError(
                    f"A cell with an edge of {side:.2f} m is finer than the "
                    f"positions of the animals; the smallest is "
                    f"{MIN_EDGE_M:g} m.")
        return self

    @property
    def edge(self) -> float:
        """The edge of a cell (not meaningful for a rectangle)."""
        if self.size_mode == BY_EDGE:
            return float(self.edge_m)
        return edge_for_area(self.shape, self.area_m2)

    @property
    def cell_area_m2(self) -> float:
        if self.shape == RECTANGLE:
            return float(self.width_m) * float(self.height_m)
        return area_for_edge(self.shape, self.edge)

    def describe(self) -> str:
        """The grid in a few words, for a log or a legend."""
        area = self.cell_area_m2
        size = (f"{area / 1e4:.4g} ha" if area >= 1000.0
                else f"{area:.4g} m²")
        if self.shape == RECTANGLE:
            return (f"rectangles of {self.width_m:g} m x {self.height_m:g} m "
                    f"({size})")
        return (f"{SHAPE_LABELS[self.shape].lower()} of {size} "
                f"(edge {self.edge:.1f} m)")

    def to_dict(self) -> dict:
        return {"shape": self.shape, "size_mode": self.size_mode,
                "area_m2": float(self.area_m2), "edge_m": float(self.edge_m),
                "width_m": float(self.width_m),
                "height_m": float(self.height_m),
                "cell_area_m2": self.cell_area_m2}

    @classmethod
    def from_config(cls, config: dict) -> "GridSpec":
        """The grid the Survey Analytics settings describe
        (``grid_shape``, ``grid_size_mode``, ``grid_cell_area_ha``,
        ``grid_cell_edge_m``, ``grid_cell_width_m``, ``grid_cell_height_m``).
        """
        config = config or {}
        return cls(
            shape=str(config.get("grid_shape") or HEXAGON).lower(),
            size_mode=str(config.get("grid_size_mode") or BY_AREA).lower(),
            area_m2=float(config.get("grid_cell_area_ha") or 1.0) * 1e4,
            edge_m=float(config.get("grid_cell_edge_m") or 100.0),
            width_m=float(config.get("grid_cell_width_m") or 100.0),
            height_m=float(config.get("grid_cell_height_m") or 100.0),
        ).validate()


@dataclass
class Cell:
    """One cell: its name, its corners and its middle.

    ``name`` is made of the cell's column and row counted from the origin of
    the coordinate system, so it names the same ground in every grid of the
    same shape and size.
    """

    name: str
    col: int
    row: int
    ring: List[Point] = field(default_factory=list)
    centre: Point = (0.0, 0.0)


def _name(col: int, row: int) -> str:
    return f"c{col}r{row}"


def _rectangles(width: float, height: float, bounds) -> List[Cell]:
    minx, miny, maxx, maxy = bounds
    cells = []
    for row in range(math.floor(miny / height), math.floor(maxy / height) + 1):
        for col in range(math.floor(minx / width),
                         math.floor(maxx / width) + 1):
            # Corners are whole multiples of the sides, so that the corner
            # two cells share is the very same number in both.
            x0, x1 = col * width, (col + 1) * width
            y0, y1 = row * height, (row + 1) * height
            cells.append(Cell(
                _name(col, row), col, row,
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                ((x0 + x1) / 2.0, (y0 + y1) / 2.0)))
    return cells


def _hexagons(edge: float, bounds) -> List[Cell]:
    """Flat-topped hexagons: columns 1.5 edges apart, every other column
    half a cell higher."""
    minx, miny, maxx, maxy = bounds
    step_x = 1.5 * edge
    height = _SQRT3 * edge
    cells = []
    for col in range(math.floor((minx - edge) / step_x),
                     math.ceil((maxx + edge) / step_x) + 1):
        shift = height / 2.0 if col % 2 else 0.0
        for row in range(math.floor((miny - height - shift) / height),
                         math.ceil((maxy + height - shift) / height) + 1):
            cx, cy = col * step_x, row * height + shift
            if cx + edge < minx or cx - edge > maxx:
                continue
            if cy + height / 2.0 < miny or cy - height / 2.0 > maxy:
                continue
            # The corners lie on a lattice of half edges and half heights:
            # counted in whole steps of it, the corner two cells share is
            # the very same number in both.
            ix, iy = 3 * col, 2 * row + (1 if col % 2 else 0)
            half_edge, half_height = edge / 2.0, height / 2.0
            ring = [((ix + dx) * half_edge, (iy + dy) * half_height)
                    for dx, dy in ((2, 0), (1, 1), (-1, 1), (-2, 0),
                                   (-1, -1), (1, -1))]
            cells.append(Cell(_name(col, row), col, row, ring,
                              (ix * half_edge, iy * half_height)))
    return cells


def _triangles(edge: float, bounds) -> List[Cell]:
    """Equilateral triangles in rows, pointing up and down by turns; the
    column counts half edges."""
    minx, miny, maxx, maxy = bounds
    half = edge / 2.0
    height = _SQRT3 / 2.0 * edge
    cells = []
    for row in range(math.floor(miny / height), math.floor(maxy / height) + 1):
        y0, y1 = row * height, (row + 1) * height
        for col in range(math.floor(minx / half) - 2,
                         math.floor(maxx / half) + 1):
            x0, x1, x2 = col * half, (col + 1) * half, (col + 2) * half
            if x2 < minx or x0 > maxx:
                continue
            if (col + row) % 2 == 0:            # pointing up
                ring = [(x0, y0), (x2, y0), (x1, y1)]
            else:                               # pointing down
                ring = [(x0, y1), (x1, y0), (x2, y1)]
            centre = (sum(p[0] for p in ring) / 3.0,
                      sum(p[1] for p in ring) / 3.0)
            cells.append(Cell(_name(col, row), col, row, ring, centre))
    return cells


def cells(spec: GridSpec, bounds: Sequence[float],
          max_cells: int = MAX_CELLS) -> List[Cell]:
    """The cells of the grid around *bounds* ``(min x, min y, max x,
    max y)``, row by row from the south: every cell that reaches into it,
    and at its corners a few whose surrounding box does - whoever lays the
    grid over an area keeps the cells that touch the area.

    :raises GridError: when that would be more than *max_cells* cells
    """
    spec.validate()
    minx, miny, maxx, maxy = (float(value) for value in bounds)
    if not all(math.isfinite(v) for v in (minx, miny, maxx, maxy)) or \
            maxx < minx or maxy < miny:
        raise GridError(f"Not an extent to lay a grid over: {bounds!r}.")
    expected = (maxx - minx + 1.0) * (maxy - miny + 1.0) / spec.cell_area_m2
    if expected > max_cells:
        raise GridError(
            f"A grid of {spec.describe()} over this area would have about "
            f"{expected:,.0f} cells. Choose larger cells.")
    bounds = (minx, miny, maxx, maxy)
    if spec.shape == HEXAGON:
        found = _hexagons(spec.edge, bounds)
    elif spec.shape == TRIANGLE:
        found = _triangles(spec.edge, bounds)
    elif spec.shape == SQUARE:
        found = _rectangles(spec.edge, spec.edge, bounds)
    else:
        found = _rectangles(float(spec.width_m), float(spec.height_m), bounds)
    found.sort(key=lambda cell: (cell.row, cell.col))
    return found


def neighbours(grid: Sequence[Cell]) -> Dict[str, List[str]]:
    """``cell name -> names of the cells that share an edge with it``,
    among the cells of *grid*.

    Cells that only touch at a corner are not neighbours: for squares and
    triangles that is the difference between four (three) neighbours and
    eight (twelve); for hexagons there is no such difference, which is one
    reason to prefer them.
    """
    def key(a: Point, b: Point):
        a = (round(a[0], 3), round(a[1], 3))
        b = (round(b[0], 3), round(b[1], 3))
        return (a, b) if a <= b else (b, a)

    edges: Dict[tuple, List[str]] = {}
    for cell in grid:
        ring = cell.ring
        for index, start in enumerate(ring):
            edges.setdefault(
                key(start, ring[(index + 1) % len(ring)]), []).append(
                cell.name)
    found: Dict[str, List[str]] = {cell.name: [] for cell in grid}
    for names in edges.values():
        if len(names) == 2:
            found[names[0]].append(names[1])
            found[names[1]].append(names[0])
    return {name: sorted(set(others)) for name, others in found.items()}


def ring_area(ring: Sequence[Point]) -> float:
    """The area a ring of corners encloses (shoelace formula)."""
    total = 0.0
    for index, (x0, y0) in enumerate(ring):
        x1, y1 = ring[(index + 1) % len(ring)]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def cell_at(spec: GridSpec, x: float, y: float) -> Optional[str]:
    """The name of the cell that holds a point - for squares and
    rectangles, where it can be read off the coordinates. ``None`` for the
    other shapes (their cells are found by looking)."""
    if spec.shape == SQUARE:
        width = height = spec.edge
    elif spec.shape == RECTANGLE:
        width, height = float(spec.width_m), float(spec.height_m)
    else:
        return None
    return _name(math.floor(x / width), math.floor(y / height))
