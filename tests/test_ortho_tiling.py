# -*- coding: utf-8 -*-
"""Unit tests for core.ortho_tiling (ALFS/orthomosaic geometry helpers)."""
import numpy as np

from bambi_wildlife_detection.core.ortho_tiling import (
    create_tile_camera,
    crop_to_content,
    erode_valid_mask,
    filter_shots_for_tile,
    merge_orthomosaic_average,
)


# ---------------------------------------------------------------------------
# create_tile_camera (fake Camera / Vector3)
# ---------------------------------------------------------------------------

class _Res:
    def __init__(self, w, h):
        self.width, self.height = w, h


class _Pos:
    def __init__(self, z):
        self.z = z


class _Transform:
    def __init__(self, z, rotation):
        self.position = _Pos(z)
        self.rotation = rotation


class _GlobalCamera:
    def __init__(self, z=100.0, rotation="ROT", near=0.1, far=1000.0):
        self.transform = _Transform(z, rotation)
        self.near, self.far = near, far


class _FakeCamera:
    def __init__(self, orthogonal, orthogonal_size, position, rotation, near, far):
        self.orthogonal = orthogonal
        self.orthogonal_size = orthogonal_size
        self.position = position
        self.rotation = rotation
        self.near, self.far = near, far


def _vec3(values, dtype=None):
    return np.asarray(values, dtype=np.float64)


class TestCreateTileCamera:
    def test_full_tile_covers_whole_extent(self):
        cam = create_tile_camera(
            _GlobalCamera(), (0.0, 0.0, 100.0, 80.0), _Res(1000, 800),
            0, 0, 1000, 800, _vec3, _FakeCamera)
        assert cam.orthogonal is True
        assert cam.orthogonal_size == (100.0, 80.0)
        # centre of the full extent
        assert list(cam.position[:2]) == [50.0, 40.0]
        assert cam.position[2] == 100.0
        assert cam.rotation == "ROT"

    def test_quadrant_tile_bounds_and_centre(self):
        # top-left 500x400 tile of a 1000x800 / 100x80m mosaic
        cam = create_tile_camera(
            _GlobalCamera(), (0.0, 0.0, 100.0, 80.0), _Res(1000, 800),
            0, 0, 500, 400, _vec3, _FakeCamera)
        assert cam.orthogonal_size == (50.0, 40.0)
        # world y is flipped (row 0 is the north/max_y edge)
        assert list(cam.position[:2]) == [25.0, 60.0]

    def test_camera_near_far_inherited(self):
        cam = create_tile_camera(
            _GlobalCamera(near=0.5, far=2000.0),
            (0.0, 0.0, 10.0, 10.0), _Res(10, 10),
            0, 0, 10, 10, _vec3, _FakeCamera)
        assert (cam.near, cam.far) == (0.5, 2000.0)


# ---------------------------------------------------------------------------
# filter_shots_for_tile (duck-typed shots)
# ---------------------------------------------------------------------------

class _ShotPos:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _ShotCam:
    def __init__(self, x, y, z, fovy=50.0):
        self.transform = type("T", (), {"position": _ShotPos(x, y, z)})()
        self.fovy = fovy


class _Shot:
    def __init__(self, x, y, z, fovy=50.0):
        self.camera = _ShotCam(x, y, z, fovy)


class TestFilterShotsForTile:
    TILE = (0.0, 0.0, 100.0, 100.0)

    def test_shot_over_tile_kept(self):
        over = _Shot(50.0, 50.0, 100.0)
        assert filter_shots_for_tile([over], self.TILE) == [over]

    def test_distant_shot_culled(self):
        over = _Shot(50.0, 50.0, 100.0)          # footprint radius ~116 m
        far = _Shot(5000.0, 5000.0, 100.0)       # far outside
        result = filter_shots_for_tile([over, far], self.TILE)
        assert result == [over]

    def test_low_altitude_shot_always_included(self):
        low = _Shot(9999.0, 9999.0, 0.5)         # altitude < 1 m
        assert low in filter_shots_for_tile([low], self.TILE)

    def test_empty_result_falls_back_to_all(self):
        far1 = _Shot(9000.0, 9000.0, 50.0)
        far2 = _Shot(-9000.0, -9000.0, 50.0)
        # none overlap → fall back to the full list
        assert filter_shots_for_tile([far1, far2], self.TILE) == [far1, far2]

    def test_logs_when_filtering(self):
        logs = []
        over = _Shot(50.0, 50.0, 100.0)
        far = _Shot(5000.0, 5000.0, 100.0)
        filter_shots_for_tile([over, far], self.TILE, log_fn=logs.append)
        assert any("1/2 shots relevant" in m for m in logs)


# ---------------------------------------------------------------------------
# crop_to_content
# ---------------------------------------------------------------------------

class TestCropToContent:
    def _rgba(self, h, w):
        return np.zeros((h, w, 4), dtype=np.uint8)

    def test_crops_to_opaque_region_and_rescales_bounds(self):
        img = self._rgba(100, 100)
        img[20:60, 30:70, 3] = 255      # opaque block
        cropped, bounds = crop_to_content(img, (0.0, 0.0, 100.0, 100.0))
        assert cropped.shape[:2] == (40, 40)
        min_x, min_y, max_x, max_y = bounds
        assert (min_x, max_x) == (30.0, 70.0)
        # rows are measured from the top (max_y) downward
        assert max_y == 80.0 and min_y == 40.0

    def test_fully_transparent_returns_input(self):
        img = self._rgba(10, 10)
        bounds = (0.0, 0.0, 5.0, 5.0)
        out, out_bounds = crop_to_content(img, bounds)
        assert out is img
        assert out_bounds == bounds

    def test_single_opaque_pixel(self):
        img = self._rgba(50, 50)
        img[25, 40, 3] = 255
        cropped, bounds = crop_to_content(img, (0.0, 0.0, 50.0, 50.0))
        assert cropped.shape[:2] == (1, 1)
        assert bounds == (40.0, 24.0, 41.0, 25.0)


# ---------------------------------------------------------------------------
# merge_orthomosaic_average (fake rasterio.merge)
# ---------------------------------------------------------------------------

class _FakeDataset:
    def __init__(self, dtype="uint8"):
        self.dtypes = [dtype]


class TestMergeOrthomosaicAverage:
    def test_averages_overlapping_valid_pixels(self):
        nodata = 0
        # Two 1x2x2 sources; rasterio.merge is faked to feed both into the
        # accumulator, so we can check the averaging arithmetic directly.
        src_a = np.array([[[10, 0], [40, 100]]], dtype=np.float64)
        src_b = np.array([[[20, 60], [0, 200]]], dtype=np.float64)
        mask_a = src_a == nodata
        mask_b = src_b == nodata

        def fake_merge(datasets, method=None, nodata=None, dtype=None):
            transform = "AFFINE"
            if method == "first":
                return np.zeros((1, 2, 2), dtype="uint8"), transform
            # averaging pass: apply the accumulator method to each source
            merged = np.zeros((1, 2, 2), dtype="float64")
            merged_mask = np.ones((1, 2, 2), dtype=bool)
            method(merged, src_a, merged_mask, mask_a, roff=0, coff=0)
            method(merged, src_b, merged_mask, mask_b, roff=0, coff=0)
            return merged, transform

        avg, transform = merge_orthomosaic_average(
            [_FakeDataset("uint8")], nodata, fake_merge)

        assert transform == "AFFINE"
        # pixel (0,0): (10+20)/2 = 15 ; (0,1): only b valid -> 60 ;
        # (1,0): only a valid -> 40 ; (1,1): (100+200)/2 = 150
        assert avg.tolist() == [[[15, 60], [40, 150]]]
        assert avg.dtype == np.uint8

    def test_all_nodata_pixel_stays_nodata(self):
        nodata = 255
        src = np.array([[[255, 30]]], dtype=np.float64)   # first px is nodata
        mask = src == nodata

        def fake_merge(datasets, method=None, nodata=None, dtype=None):
            if method == "first":
                return np.zeros((1, 1, 2), dtype="uint8"), "T"
            merged = np.zeros((1, 1, 2), dtype="float64")
            merged_mask = np.ones((1, 1, 2), dtype=bool)
            method(merged, src, merged_mask, mask, roff=0, coff=0)
            return merged, "T"

        avg, _ = merge_orthomosaic_average(
            [_FakeDataset("uint8")], nodata, fake_merge)
        assert avg[0, 0, 0] == 255      # untouched nodata
        assert avg[0, 0, 1] == 30


# ---------------------------------------------------------------------------
# erode_valid_mask
# ---------------------------------------------------------------------------

class TestErodeValidMask:
    def test_zero_erosion_returns_input(self):
        mask = np.ones((5, 5), dtype=bool)
        out = erode_valid_mask(mask, 0)
        assert out is mask

    def test_negative_erosion_returns_input(self):
        mask = np.ones((5, 5), dtype=bool)
        assert erode_valid_mask(mask, -3) is mask

    def test_empty_mask_returned_unchanged(self):
        mask = np.zeros((5, 5), dtype=bool)
        assert erode_valid_mask(mask, 2) is mask

    def test_one_px_removes_full_border(self):
        mask = np.ones((5, 5), dtype=bool)
        out = erode_valid_mask(mask, 1)
        # The outer ring is dropped; a 3x3 interior remains.
        expected = np.zeros((5, 5), dtype=bool)
        expected[1:4, 1:4] = True
        assert np.array_equal(out, expected)

    def test_two_px_erosion(self):
        mask = np.ones((7, 7), dtype=bool)
        out = erode_valid_mask(mask, 2)
        expected = np.zeros((7, 7), dtype=bool)
        expected[2:5, 2:5] = True
        assert np.array_equal(out, expected)

    def test_erosion_larger_than_mask_clears_it(self):
        mask = np.ones((3, 3), dtype=bool)
        out = erode_valid_mask(mask, 5)
        assert not out.any()

    def test_diamond_shape_on_interior_blob(self):
        # A single interior True pixel is removed by a 1 px erosion because
        # its 4-neighbours must all be valid.
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        assert not erode_valid_mask(mask, 1).any()

    def test_does_not_mutate_input(self):
        mask = np.ones((4, 4), dtype=bool)
        original = mask.copy()
        erode_valid_mask(mask, 1)
        assert np.array_equal(mask, original)
