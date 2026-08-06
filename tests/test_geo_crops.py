# -*- coding: utf-8 -*-
"""Crops taken from the orthorectified frames.

Written against real GeoTIFFs rather than a fake transform: the whole point of
this module is agreeing with what rasterio actually does with a raster's
affine, and a stand-in would only prove it agrees with itself.
"""
import os

import numpy as np
import pytest

from bambi_wildlife_detection.core import geo_crops

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402


# A north-up raster: 200x100 px at 0.5 m, origin at (1000, 2000). North-up
# means the y scale is negative, which is what makes corner ordering matter.
ORIGIN_X, ORIGIN_Y, PIXEL = 1000.0, 2000.0, 0.5
WIDTH, HEIGHT = 200, 100


def _write_geotiff(path, bands=3, dtype="uint8", width=WIDTH, height=HEIGHT):
    transform = from_origin(ORIGIN_X, ORIGIN_Y, PIXEL, PIXEL)
    data = np.zeros((bands, height, width), dtype=dtype)
    # A bright square whose world position is known exactly.
    data[:, 10:20, 30:40] = 255 if dtype == "uint8" else 1000
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(
            path, "w", driver="GTiff", height=height, width=width,
            count=bands, dtype=dtype, transform=transform,
            crs="EPSG:32633") as raster:
        raster.write(data)
    return transform


@pytest.fixture
def flight(tmp_path):
    root = str(tmp_path)
    _write_geotiff(geo_crops.geotiff_path(root, "t", 0))
    _write_geotiff(geo_crops.geotiff_path(root, "t", 5))
    return root


# ---------------------------------------------------------------------------
# Locating exports
# ---------------------------------------------------------------------------

class TestLocating:

    def test_the_path_matches_the_export_naming(self, tmp_path):
        path = geo_crops.geotiff_path(str(tmp_path), "w", 42)
        assert path.endswith(os.path.join("geotiffs_w", "00000042.tiff"))

    def test_available_frames_are_found(self, flight):
        assert sorted(geo_crops.available_frames(flight, "t")) == [0, 5]

    def test_an_absent_folder_yields_nothing(self, tmp_path):
        assert geo_crops.available_frames(str(tmp_path), "w") == {}

    def test_non_frame_files_are_ignored(self, flight):
        folder = os.path.join(flight, "geotiffs_t")
        open(os.path.join(folder, "orthomosaic.tif"), "wb").close()
        open(os.path.join(folder, "notes.txt"), "w").close()
        assert sorted(geo_crops.available_frames(flight, "t")) == [0, 5]

    def test_missing_frames_are_reported(self, flight):
        assert geo_crops.missing_frames(flight, "t", [0, 3, 5, 9]) == [3, 9]

    def test_only_geo_projections_use_rasters(self):
        assert geo_crops.is_geo("geo_1k") and geo_crops.is_geo("geo_2k")
        assert not geo_crops.is_geo("non_geo")


# ---------------------------------------------------------------------------
# World -> pixel
# ---------------------------------------------------------------------------

class TestMapping:

    def test_the_origin_maps_to_the_top_left(self, flight):
        _image, transform = geo_crops.read_geotiff(
            geo_crops.geotiff_path(flight, "t", 0))
        assert geo_crops.world_to_pixel(transform, ORIGIN_X, ORIGIN_Y) == \
            pytest.approx((0.0, 0.0))

    def test_a_known_world_point_maps_to_its_pixel(self, flight):
        _image, transform = geo_crops.read_geotiff(
            geo_crops.geotiff_path(flight, "t", 0))
        # 30 px right and 10 px down from the origin.
        world_x = ORIGIN_X + 30 * PIXEL
        world_y = ORIGIN_Y - 10 * PIXEL
        assert geo_crops.world_to_pixel(transform, world_x, world_y) == \
            pytest.approx((30.0, 10.0))

    def test_a_box_comes_back_well_formed_despite_the_flipped_axis(self, flight):
        """North-up rasters have a negative y scale, so mapping the corners
        straight through gives y1 > y2."""
        _image, transform = geo_crops.read_geotiff(
            geo_crops.geotiff_path(flight, "t", 0))
        box = geo_crops.geo_box_to_pixel(transform, (
            ORIGIN_X + 30 * PIXEL, ORIGIN_Y - 20 * PIXEL,
            ORIGIN_X + 40 * PIXEL, ORIGIN_Y - 10 * PIXEL))

        assert box[0] < box[2] and box[1] < box[3]
        assert box == pytest.approx((30.0, 10.0, 40.0, 20.0))

    def test_corner_order_does_not_matter(self, flight):
        _image, transform = geo_crops.read_geotiff(
            geo_crops.geotiff_path(flight, "t", 0))
        corners = (ORIGIN_X + 30 * PIXEL, ORIGIN_Y - 20 * PIXEL,
                   ORIGIN_X + 40 * PIXEL, ORIGIN_Y - 10 * PIXEL)
        swapped = (corners[2], corners[3], corners[0], corners[1])
        assert geo_crops.geo_box_to_pixel(transform, corners) == \
            pytest.approx(geo_crops.geo_box_to_pixel(transform, swapped))

    def test_the_mapped_box_lands_on_the_content(self, flight):
        """End to end: the bright square's world box maps back onto it."""
        image, transform = geo_crops.read_geotiff(
            geo_crops.geotiff_path(flight, "t", 0))
        box = geo_crops.geo_box_to_pixel(transform, (
            ORIGIN_X + 30 * PIXEL, ORIGIN_Y - 20 * PIXEL,
            ORIGIN_X + 40 * PIXEL, ORIGIN_Y - 10 * PIXEL))
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        assert (image[y1:y2, x1:x2] == 255).all()


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

class TestReading:

    def test_a_three_band_raster_reads_as_rgb(self, flight):
        image, _t = geo_crops.read_geotiff(
            geo_crops.geotiff_path(flight, "t", 0))
        assert image.shape == (HEIGHT, WIDTH, 3)
        assert image.dtype == np.uint8

    def test_a_single_band_raster_is_widened(self, tmp_path):
        path = geo_crops.geotiff_path(str(tmp_path), "t", 1)
        _write_geotiff(path, bands=1)
        image, _t = geo_crops.read_geotiff(path)
        assert image.shape == (HEIGHT, WIDTH, 3)

    def test_an_alpha_band_is_dropped(self, tmp_path):
        path = geo_crops.geotiff_path(str(tmp_path), "t", 1)
        _write_geotiff(path, bands=4)
        image, _t = geo_crops.read_geotiff(path)
        assert image.shape[2] == 3

    def test_a_sixteen_bit_raster_is_scaled_once_per_raster(self, tmp_path):
        """Scaling per crop would make the same animal look different
        depending on what else was in its window."""
        path = geo_crops.geotiff_path(str(tmp_path), "t", 1)
        _write_geotiff(path, dtype="uint16")
        image, _t = geo_crops.read_geotiff(path)
        assert image.dtype == np.uint8
        assert image.max() == 255 and image.min() == 0

    def test_the_result_is_contiguous(self, flight):
        """cv2.resize refuses a negative or non-contiguous stride."""
        image, _t = geo_crops.read_geotiff(
            geo_crops.geotiff_path(flight, "t", 0))
        assert image.flags["C_CONTIGUOUS"]

    def test_an_absent_raster_reads_as_none(self, tmp_path):
        image, transform = geo_crops.read_geotiff(
            geo_crops.geotiff_path(str(tmp_path), "t", 99))
        assert image is None and transform is None

    def test_a_corrupt_raster_reads_as_none(self, tmp_path):
        """A half-written export is skipped with a warning, not a dead run."""
        path = geo_crops.geotiff_path(str(tmp_path), "t", 1)
        os.makedirs(os.path.dirname(path))
        with open(path, "wb") as handle:
            handle.write(b"not a geotiff")
        assert geo_crops.read_geotiff(path) == (None, None)


# ---------------------------------------------------------------------------
# Resolution sanity
# ---------------------------------------------------------------------------

class TestResolutionWarning:

    def test_a_matching_export_is_silent(self):
        assert geo_crops.resolution_warning("geo_2k", 2048) is None
        assert geo_crops.resolution_warning("geo_1k", 1024) is None

    def test_moderate_differences_are_tolerated(self):
        # The export resolution follows the flight and the DEM; it is never
        # going to land exactly on the nominal figure.
        assert geo_crops.resolution_warning("geo_2k", 1400) is None
        assert geo_crops.resolution_warning("geo_1k", 1500) is None

    def test_a_far_smaller_export_is_flagged(self):
        message = geo_crops.resolution_warning("geo_2k", 400)
        assert message and "400 px" in message and "2048" in message

    def test_a_far_larger_export_is_flagged(self):
        assert geo_crops.resolution_warning("geo_1k", 4096) is not None

    def test_perspective_is_never_flagged(self):
        assert geo_crops.resolution_warning("non_geo", 640) is None

    def test_an_unknown_width_is_not_flagged(self):
        assert geo_crops.resolution_warning("geo_1k", 0) is None
