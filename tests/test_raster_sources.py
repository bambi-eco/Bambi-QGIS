# -*- coding: utf-8 -*-
"""Perspective / GeoTIFF / ALFS as the imagery of a stage
(core.raster_sources, 9.0.0)."""

import os

import numpy as np
import pytest

from bambi_wildlife_detection.core import raster_sources as rs


class Affine:
    """1 m pixels; upper-left corner at (x0, y0)."""

    def __init__(self, x0, y0):
        self.x0, self.y0 = x0, y0

    def __invert__(self):
        outer = self

        class Inverse:
            def __mul__(self, point):
                return (point[0] - outer.x0, outer.y0 - point[1])
        return Inverse()


def _touch(folder, name):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    open(path, "wb").close()
    return path


def test_projection_settings_map_to_three_kinds():
    assert rs.kind_of("non_geo") == rs.PERSPECTIVE
    assert rs.kind_of("perspective") == rs.PERSPECTIVE
    assert rs.kind_of(None) == rs.PERSPECTIVE
    assert rs.kind_of("geo_1k") == rs.kind_of("geo_2k") == rs.GEOTIFF
    assert rs.kind_of("geotiff") == rs.GEOTIFF
    assert rs.kind_of("alfs") == rs.ALFS
    assert not rs.is_raster("non_geo") and rs.is_raster("alfs")
    assert [rs.context_name(p) for p in ("non_geo", "geo_2k", "alfs")] == \
        ["perspective", "orthographic", "alfs"]


def test_alfs_samples_are_found_by_central_frame(tmp_path):
    folder = str(tmp_path / "alfs_t")
    _touch(folder, "sample_000010_tile_00_00.tif")
    _touch(folder, "sample_000010_tile_00_01.tif")
    _touch(folder, "sample_000040_tile_00_00.tif")
    _touch(folder, "alfs.tif")                       # the whole-flight mosaic
    samples = rs.alfs_samples(str(tmp_path), "t")
    assert sorted(samples) == [10, 40]
    assert len(samples[10]) == 2
    assert rs.alfs_samples(str(tmp_path), "w") == {}


def test_the_nearest_central_frame_stands_in():
    frames = [10, 40, 70]
    assert rs.nearest_frame(frames, 12) == 10
    assert rs.nearest_frame(frames, 25) == 10        # tie: the earlier one
    assert rs.nearest_frame(frames, 26) == 40
    assert rs.nearest_frame(frames, 200) == 70
    assert rs.nearest_frame(frames, 200, max_gap=15) is None
    assert rs.nearest_frame([], 3) is None


def test_an_alfs_source_reads_the_tile_that_shows_the_box(tmp_path):
    folder = str(tmp_path / "alfs_t")
    west = _touch(folder, "sample_000010_tile_00_00.tif")
    east = _touch(folder, "sample_000010_tile_00_01.tif")
    rasters = {west: (np.zeros((50, 50, 3), np.uint8), Affine(0, 100)),
               east: (np.ones((50, 50, 3), np.uint8), Affine(50, 100))}
    reads = []

    def reader(path):
        reads.append(path)
        return rasters[path]

    source = rs.RasterSource(str(tmp_path), "t", rs.ALFS, max_gap=5,
                             reader=reader)
    assert source.frames() == [10]
    assert source.paths(10) == [west, east] and source.paths(11) == []
    assert source.frame_for(13) == 10 and source.frame_for(30) is None

    image, _t = source.read_for(12, (60.0, 70.0, 64.0, 74.0))
    assert image.max() == 1                          # the east tile
    image, _t = source.read_for(12, (10.0, 70.0, 14.0, 74.0))
    assert image.max() == 0
    assert source.read_for(12, (500.0, 70.0, 504.0, 74.0)) == (None, None)
    assert source.read_for(30, (60.0, 70.0, 64.0, 74.0)) == (None, None)
    assert len(reads) == 2                           # each tile read once


def test_a_geotiff_source_has_one_raster_per_exported_frame(tmp_path):
    folder = str(tmp_path / "geotiffs_w")
    _touch(folder, "00000003.tiff")
    _touch(folder, "00000007.tiff")
    source = rs.RasterSource(
        str(tmp_path), "w", rs.GEOTIFF,
        reader=lambda path: (np.zeros((4, 4, 3), np.uint8), Affine(0, 4)))
    assert source.frames() == [3, 7]
    assert source.frame_for(5) is None               # no stand-in
    assert source.read_for(3)[0] is not None
    assert source.read_for(5) == (None, None)


def test_perspective_is_not_a_raster_source(tmp_path):
    with pytest.raises(ValueError):
        rs.RasterSource(str(tmp_path), "t", rs.PERSPECTIVE)


def test_the_message_names_what_has_to_run():
    text = rs.missing_message(rs.ALFS, "Thermal", "Detection")
    assert "sampling mode" in text and "Thermal" in text
