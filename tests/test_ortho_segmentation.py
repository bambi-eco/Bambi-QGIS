# -*- coding: utf-8 -*-
"""Segmenting the orthomosaic: tiling, merging across tiles, geo-referencing.

A tree canopy visible in ten frames is ten masks on the frames and one on
the mosaic - that is the point of this route, and the merge is what makes
a canopy cut by a tile border one polygon rather than two.
"""
import json
import os

import pytest

from bambi_wildlife_detection.core import ortho_segmentation as ortho
from bambi_wildlife_detection.core import segmentation as seg


def _square(x, y, size):
    return [[x, y], [x + size, y], [x + size, y + size], [x, y + size]]


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

class TestTiling:
    def test_tiles_cover_the_mosaic_without_slivers(self):
        windows = ortho.tile_windows(2500, 1300, tile=1000, overlap=100)
        assert all(w == 1000 and h == 1000 for _x, _y, w, h in windows)
        xs = sorted({x for x, _y, _w, _h in windows})
        ys = sorted({y for _x, y, _w, _h in windows})
        assert xs == [0, 900, 1500]        # the last tile ends at the edge
        assert ys == [0, 300]
        assert max(x + w for x, _y, w, _h in windows) == 2500
        assert max(y + h for _x, y, _w, h in windows) == 1300

    def test_a_small_mosaic_is_one_tile(self):
        assert ortho.tile_windows(500, 400, tile=1024) == [(0, 0, 500, 400)]
        assert ortho.tile_count(500, 400, 1024, 128) == 1

    def test_window_around_a_click_is_clamped(self):
        assert ortho.window_around(10, 10, 5000, 5000, 1000) == (0, 0, 1000, 1000)
        assert ortho.window_around(4990, 2500, 5000, 5000, 1000) == (4000, 2000, 1000, 1000)
        assert ortho.window_around(100, 100, 600, 600, 1000) == (0, 0, 600, 600)


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

class TestMerge:
    def test_a_canopy_across_a_tile_border_becomes_one_object(self):
        pytest.importorskip("cv2")
        # Two 1000-px tiles overlapping by 200; the same 300-px canopy is seen
        # in both, at tile x 700..1000 in the first and -100..200 in the second
        # (clipped to 0..200 there).
        left = ((0, 0, 1000, 1000), [seg.text_entry("tree", [
            {"confidence": 0.7, "polygons": [_square(700, 300, 300)]}])])
        right = ((800, 0, 1000, 1000), [seg.text_entry("tree", [
            {"confidence": 0.9, "polygons": [_square(0, 300, 200)]},
            {"confidence": 0.6, "polygons": [_square(500, 500, 100)]}])])
        merged = ortho.merge_tiles(1800, 1000, [left, right], merge_max_side=1800)
        assert [m["prompt"] for m in merged] == ["tree"]
        predictions = merged[0]["predictions"]
        assert len(predictions) == 2

        def width(prediction):
            xs = [x for x, _y in prediction["polygons"][0]]
            return max(xs) - min(xs)
        big = max(predictions, key=width)
        xs = [x for x, _y in big["polygons"][0]]
        ys = [y for _x, y in big["polygons"][0]]
        # One polygon spanning the border, with the higher of the two scores.
        assert min(xs) == pytest.approx(700, abs=2) and max(xs) == pytest.approx(1000, abs=2)
        assert min(ys) == pytest.approx(300, abs=2) and max(ys) == pytest.approx(600, abs=2)
        assert big["confidence"] == 0.9
        assert sorted(p["object_id"] for p in predictions) == [1, 2]

    def test_merge_raster_is_downscaled_for_huge_mosaics(self):
        pytest.importorskip("cv2")
        tiles = [((0, 0, 1000, 1000), [seg.text_entry("tree", [
            {"confidence": 0.5, "polygons": [_square(100, 100, 400)]}])])]
        merged = ortho.merge_tiles(40000, 40000, tiles, merge_max_side=4000)
        ring = merged[0]["predictions"][0]["polygons"][0]
        xs = [x for x, _y in ring]
        # Coordinates come back in full mosaic pixels, within one merge cell.
        assert min(xs) == pytest.approx(100, abs=10) and max(xs) == pytest.approx(500, abs=10)

    def test_point_objects_are_moved_not_merged(self):
        tiles = [((300, 400, 1000, 1000), [seg.point_entry(2, {2: "calf"}, [
            {"confidence": 1.0, "polygons": [_square(10, 20, 5)]}])])]
        merged = ortho.merge_tiles(5000, 5000, tiles)
        assert merged[0]["prompt"] == "calf"
        assert merged[0]["predictions"][0]["polygons"][0][0] == [310.0, 420.0]


# ---------------------------------------------------------------------------
# Geo-referencing
# ---------------------------------------------------------------------------

class TestGeoreference:
    TRANSFORM = [0.05, 0.0, 500000.0, 0.0, -0.05, 5200000.0]

    def test_pixel_to_world_uses_the_affine(self):
        assert ortho.pixel_to_world(self.TRANSFORM, 0, 0) == (500000.0, 5200000.0)
        x, y = ortho.pixel_to_world(self.TRANSFORM, 100, 200)
        assert (x, y) == (500005.0, 5199990.0)

    def test_georeference_mirrors_the_pixel_file(self):
        pixel = [{"frame_idx": ortho.ORTHO_FRAME, "imagefile": "orthomosaic.tif",
                  "source": seg.SOURCE_ORTHO, "width": 10, "height": 10,
                  "transform": self.TRANSFORM, "epsg": 32633,
                  "prompts": [seg.text_entry("tree", [
                      {"confidence": 0.8, "object_id": 1,
                       "polygons": [_square(0, 0, 100)]},
                      {"confidence": 0.8, "object_id": 2, "polygons": [[[1, 1]]]}])]}]
        geo = ortho.georeference(pixel)
        assert geo[0]["source"] == seg.SOURCE_ORTHO and geo[0]["epsg"] == 32633
        preds = geo[0]["prompts"][0]["predictions"]
        assert len(preds) == 1                     # the degenerate ring is dropped
        assert preds[0]["object_id"] == 1
        assert preds[0]["world_polygons"][0][0] == [500000.0, 5200000.0, 0.0]
        assert preds[0]["world_polygons"][0][2] == [500005.0, 5199995.0, 0.0]
        assert "polygons" not in preds[0]

    def test_the_merge_keeps_the_transform_so_the_geo_file_is_not_empty(self):
        """Regression: 'Add to QGIS' said 'No geo-referenced masks' because
        the per-prompt merge rebuilt the entry without its transform."""
        run = [{"frame_idx": ortho.ORTHO_FRAME, "imagefile": "orthomosaic.tif",
                "source": seg.SOURCE_ORTHO, "width": 10, "height": 10,
                "transform": self.TRANSFORM, "epsg": 32633,
                "prompts": [seg.text_entry("tree", [
                    {"confidence": 0.8, "object_id": 1,
                     "polygons": [_square(0, 0, 100)]}])]}]
        merged = seg.merge_results([], run, ["tree"])
        assert merged[0]["transform"] == self.TRANSFORM
        rerun = seg.merge_results(merged, run, ["tree"])
        assert rerun[0]["epsg"] == 32633
        geo = ortho.georeference(rerun)
        assert geo and geo[0]["prompts"][0]["predictions"]

    def test_geojson_names_the_source(self, tmp_path):
        pytest.importorskip("pyproj")
        geo = ortho.georeference([{
            "frame_idx": ortho.ORTHO_FRAME, "imagefile": "orthomosaic.tif",
            "source": seg.SOURCE_ORTHO, "transform": self.TRANSFORM,
            "prompts": [seg.text_entry("tree", [
                {"confidence": 0.8, "object_id": 3, "polygons": [_square(0, 0, 100)]}])]}])
        out = str(tmp_path / "o.geojson")
        seg.export_geojson(geo, out, 32633, "w")
        with open(out, encoding="utf-8") as fh:
            feature = json.load(fh)["features"][0]
        assert feature["properties"]["source"] == seg.SOURCE_ORTHO
        assert feature["properties"]["frame"] == ortho.ORTHO_FRAME
        assert feature["properties"]["object_id"] == 3


# ---------------------------------------------------------------------------
# Request and store
# ---------------------------------------------------------------------------

class TestRequestAndStore:
    def test_the_mosaic_has_no_sequence_and_needs_no_frames(self):
        request = seg.SegmentationRequest("x", "t", source=seg.SOURCE_ORTHO,
                                          texts=["tree"])
        request.validate()                         # no frames needed
        with pytest.raises(seg.SegmentationError, match="no sequence"):
            seg.SegmentationRequest("x", "t", source=seg.SOURCE_ORTHO,
                                    mode=seg.MODE_SEQUENCE, texts=["tree"]).validate()
        with pytest.raises(seg.SegmentationError, match="overlap"):
            seg.SegmentationRequest("x", "t", source=seg.SOURCE_ORTHO, texts=["t"],
                                    tile_size=512, tile_overlap=600).validate()
        # Clicks on the mosaic carry the mosaic frame id, not a frame.
        seg.SegmentationRequest("x", "t", source=seg.SOURCE_ORTHO,
                                points=[seg.PointPrompt(1, ortho.ORTHO_FRAME, 5, 5)]
                                ).validate()

    def test_store_keeps_mosaic_results_beside_the_frame_results(self, tmp_path):
        frames = seg.SegmentationStore(str(tmp_path), "w")
        mosaic = seg.SegmentationStore(str(tmp_path), "w", seg.SOURCE_ORTHO)
        assert os.path.dirname(frames.pixel_path) == os.path.dirname(mosaic.pixel_path)
        assert frames.pixel_path != mosaic.pixel_path
        mosaic.save_pixel([{"frame_idx": -1, "prompts": []}])
        mosaic.save_georef([{"frame_idx": -1, "prompts": []}])
        assert mosaic.has_pixel() and mosaic.has_georef()
        assert not frames.has_pixel()


# ---------------------------------------------------------------------------
# The run, on a real (tiny) GeoTIFF
# ---------------------------------------------------------------------------

class _TileBackend(seg.SegmentationBackend):
    """Returns one square per text prompt in every tile it is shown, and
    one square around the first click of every object."""

    capabilities = frozenset({seg.CAP_TEXT, seg.CAP_POINTS})

    def __init__(self):
        super().__init__()
        self.tiles = []

    def segment_image(self, image_path, texts, points, names, confidence):
        from PIL import Image
        with Image.open(image_path) as handle:
            size = handle.size
        self.tiles.append((os.path.basename(image_path), size))
        entries = [seg.text_entry(t, [{"confidence": 0.9,
                                       "polygons": [_square(10, 10, 40)]}])
                   for t in texts]
        for object_id in sorted({p.object_id for p in points}):
            first = next(p for p in points if p.object_id == object_id)
            entries.append(seg.point_entry(object_id, names, [
                {"confidence": 1.0,
                 "polygons": [_square(first.x - 5, first.y - 5, 10)]}]))
        return entries


def _write_mosaic(tmp_path, modality="w", size=(300, 200), bands=3, dtype="uint8"):
    rasterio = pytest.importorskip("rasterio")
    np = pytest.importorskip("numpy")
    from rasterio.transform import from_origin
    folder = tmp_path / "out" / f"orthomosaic_{modality}"
    folder.mkdir(parents=True)
    path = folder / "orthomosaic.tif"
    width, height = size
    data = np.random.default_rng(0).integers(
        1, 200 if dtype == "uint8" else 4000, (bands, height, width)).astype(dtype)
    data[:, :, :50] = 0                       # a nodata stripe on the left
    with rasterio.open(
            str(path), "w", driver="GTiff", width=width, height=height,
            count=bands, dtype=dtype, crs="EPSG:32633",
            transform=from_origin(500000.0, 5200000.0, 0.1, 0.1), nodata=0) as dst:
        dst.write(data)
    return str(tmp_path / "out")


class TestRun:
    def test_text_prompts_tile_the_mosaic_and_merge(self, tmp_path):
        pytest.importorskip("cv2")
        pytest.importorskip("PIL")
        root = _write_mosaic(tmp_path)
        backend = _TileBackend()
        logs = []
        progress = []
        results = seg.run_segmentation(
            seg.SegmentationRequest(root, "w", source=seg.SOURCE_ORTHO,
                                    texts=["tree"], tile_size=256, tile_overlap=56),
            progress_fn=progress.append, log_fn=logs.append, backend=backend)
        assert len(results) == 1
        entry = results[0]
        assert entry["frame_idx"] == ortho.ORTHO_FRAME
        assert entry["source"] == seg.SOURCE_ORTHO
        assert (entry["width"], entry["height"]) == (300, 200)
        assert entry["epsg"] == 32633
        assert entry["transform"][0] == pytest.approx(0.1)
        # 300 wide with 256-px tiles stepping 200: two columns, one row.
        assert len(backend.tiles) == 2
        assert all(size == (256, 200) for _n, size in backend.tiles)
        assert [p["prompt"] for p in entry["prompts"]] == ["tree"]
        assert entry["prompts"][0]["predictions"]
        assert progress[0] == 5 and progress[-1] == 100
        assert any("tile(s)" in line for line in logs)

    def test_point_prompts_are_answered_on_a_tile_around_the_click(self, tmp_path):
        pytest.importorskip("cv2")
        pytest.importorskip("PIL")
        root = _write_mosaic(tmp_path)
        backend = _TileBackend()
        results = seg.run_segmentation(
            seg.SegmentationRequest(
                root, "w", source=seg.SOURCE_ORTHO, tile_size=256,
                points=[seg.PointPrompt(1, ortho.ORTHO_FRAME, 250, 150)],
                object_names={1: "oak"}),
            backend=backend)
        prompt = results[0]["prompts"][0]
        assert prompt["prompt"] == "oak" and prompt["prompt_type"] == seg.PROMPT_POINT
        ring = prompt["predictions"][0]["polygons"][0]
        # Back in mosaic pixels: the square sits around the click.
        assert ring[0] == [245.0, 145.0]

    def test_a_thermal_mosaic_is_stretched_to_rgb(self, tmp_path):
        pytest.importorskip("PIL")
        root = _write_mosaic(tmp_path, modality="t", bands=1, dtype="uint16")
        with ortho.OrthoReader(ortho.orthomosaic_path(root, "t")) as reader:
            rgb, scale = reader.preview(max_side=100)
            assert rgb.shape[2] == 3 and rgb.dtype.name == "uint8"
            assert scale == pytest.approx(3.0)
            assert reader.tile((0, 0, 40, 40)) is None       # nodata stripe
            tile = reader.tile((100, 0, 40, 40))
            assert tile is not None and tile.max() > 0

    def test_a_missing_mosaic_is_named(self, tmp_path):
        with pytest.raises(ortho.OrthoError, match="P6"):
            ortho.read_info(str(tmp_path / "nope.tif"))
