# -*- coding: utf-8 -*-
"""Unit tests for core.tile_math (slippy-map tile arithmetic)."""
import pytest

from bambi_wildlife_detection.core.tile_math import latlon_to_tile, tile_to_latlon


class TestTileMath:
    def test_zoom_zero_is_single_tile(self):
        assert latlon_to_tile(47.5, 13.2, 0) == (0, 0)
        assert latlon_to_tile(-33.9, 151.2, 0) == (0, 0)

    def test_quadrants_at_zoom_one(self):
        # Northern/western hemisphere -> tile (0, 0); south-east -> (1, 1)
        assert latlon_to_tile(45.0, -90.0, 1) == (0, 0)
        assert latlon_to_tile(-45.0, 90.0, 1) == (1, 1)

    def test_tile_origin_latlon(self):
        # Tile (0, 0) NW corner is 180 deg west at the Mercator limit
        lat, lon = tile_to_latlon(0, 0, 1)
        assert lon == pytest.approx(-180.0)
        assert lat == pytest.approx(85.0511, abs=1e-3)
        # Centre of the world map
        lat, lon = tile_to_latlon(1, 1, 1)
        assert (lat, lon) == (pytest.approx(0.0), pytest.approx(0.0))

    def test_roundtrip_nw_corner(self):
        for zoom in (5, 10, 15):
            xt, yt = latlon_to_tile(47.5, 13.2, zoom)
            lat, lon = tile_to_latlon(xt, yt, zoom)
            # NW corner maps back into the same tile
            assert latlon_to_tile(lat - 1e-9, lon + 1e-9, zoom) == (xt, yt)

    def test_higher_zoom_doubles_indices(self):
        x10, y10 = latlon_to_tile(47.5, 13.2, 10)
        x11, y11 = latlon_to_tile(47.5, 13.2, 11)
        assert x11 in (2 * x10, 2 * x10 + 1)
        assert y11 in (2 * y10, 2 * y10 + 1)
