# -*- coding: utf-8 -*-
"""Where the heights of an elevation mesh are.

An elevation model is a raster: every cell has one height, and that height
is the height at the cell's **centre**. The mesh made from it put each
height at its cell's south-west corner instead - ``col * size`` east and
``(rows - row - 1) * size`` north of the raster's corner - so the whole
terrain lay half a cell too far to the south-west: half a metre with the
Austrian one-metre model, five metres with a ten-metre one. On a slope that
is a height error at every position (2 m on a 20 degree slope with ten-metre
cells), and with it an error in where a camera ray meets the ground.

Meshes made since carry :data:`HEIGHTS_AT` in their metadata. One without it
that was made from a raster is recognised as shifted, by how much, and the
steps that place things on the terrain say so.

No dependencies; free of QGIS and Qt.
"""

from typing import Optional, Tuple

#: Metadata entry of a mesh whose heights are where they belong.
HEIGHTS_AT = "heights_at"
CELL_CENTRES = "cell centres"


def cell_position(row: int, col: int, rows: int,
                  cell_x: float, cell_y: float) -> Tuple[float, float]:
    """``(east, north)`` of the height of raster cell (*row*, *col*), in
    metres from the south-west corner of a raster of *rows* rows - the
    cell's centre. Row 0 is the northernmost."""
    return (col + 0.5) * cell_x, (rows - row - 0.5) * cell_y


def cell_size(metadata: dict) -> Optional[Tuple[float, float]]:
    """``(east-west, north-south)`` size of the cells a mesh was made from,
    from its metadata - ``None`` for a mesh not made from a raster (a flat
    surface) or metadata that does not say."""
    transform = (metadata or {}).get("transform")
    try:
        size = abs(float(transform[0])), abs(float(transform[4]))
    except (TypeError, ValueError, IndexError):
        return None
    return size if size[0] > 0 and size[1] > 0 else None


def stale_shift(metadata: dict) -> Optional[Tuple[float, float]]:
    """How far the terrain of a mesh lies to the ``(west, south)`` of where
    it is, in metres - ``None`` for a mesh whose heights are at the cell
    centres, or which was not made from a raster."""
    if (metadata or {}).get(HEIGHTS_AT) == CELL_CENTRES:
        return None
    size = cell_size(metadata)
    if size is None:
        return None
    return size[0] / 2.0, size[1] / 2.0


def stale_note(metadata: dict) -> Optional[str]:
    """A sentence for the log about a mesh that is shifted, or ``None``."""
    shift = stale_shift(metadata)
    if shift is None:
        return None
    west, south = shift
    return (
        "Note: this elevation mesh was made with version 9.0 or earlier, "
        f"which put its terrain {west:g} m too far west and {south:g} m too "
        "far south (half a cell of the elevation model). Positions on "
        "sloping ground are off accordingly. Create the mesh again "
        "(Download DEM, or Convert a GeoTIFF) to correct it; results "
        "computed with the old mesh change slightly when computed again.")
