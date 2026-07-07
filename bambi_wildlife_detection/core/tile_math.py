# -*- coding: utf-8 -*-
"""Slippy-map (Web-Mercator XYZ) tile arithmetic.

Moved from ``bambi_video_creator._MapTileProvider``; standard OSM tile
formulas (https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames).
"""

import math


def latlon_to_tile(lat, lon, zoom):
    """Return the (x, y) tile index containing *lat*/*lon* at *zoom*."""
    n = 2.0 ** zoom
    xt = int((lon + 180.0) / 360.0 * n)
    yt = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return xt, yt


def tile_to_latlon(x, y, zoom):
    """Return the (lat, lon) of the north-west corner of tile (x, y)."""
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon
