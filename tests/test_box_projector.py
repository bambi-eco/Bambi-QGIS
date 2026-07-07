# -*- coding: utf-8 -*-
"""Unit tests for the cross-modality box projection helpers."""
import json

import pytest

from bambi_wildlife_detection.bambi_box_projector import (
    _correction_for_frame,
    _load_georef,
    _match_boxes_to_georef,
    _project_georef_box_to_pixels,
    _read_correction,
    _world_to_pixel,
)


class TestReadCorrection:
    def test_explicit_path_wins_over_fallback(self, tmp_path):
        explicit = tmp_path / "explicit.json"
        explicit.write_text(json.dumps({"translation": {"x": 1}}))
        fallback = tmp_path / "correction.json"
        fallback.write_text(json.dumps({"translation": {"x": 2}}))

        corr = _read_correction(str(tmp_path), str(explicit))
        assert corr["translation"]["x"] == 1

    def test_falls_back_to_target_folder(self, tmp_path):
        (tmp_path / "correction.json").write_text(json.dumps({"translation": {"x": 2}}))
        corr = _read_correction(str(tmp_path), "")
        assert corr["translation"]["x"] == 2

    def test_invalid_json_returns_empty(self, tmp_path):
        (tmp_path / "correction.json").write_text("{not valid json")
        assert _read_correction(str(tmp_path), "") == {}

    def test_missing_files_return_empty(self, tmp_path):
        assert _read_correction(str(tmp_path), str(tmp_path / "nope.json")) == {}


class TestCorrectionForFrame:
    CORR = {
        "translation": {"x": 1.0, "y": 0.0, "z": 0.0},
        "rotation": {"x": 0.0, "y": 0.0, "z": 5.0},
        "additional": [
            {"start": 10, "end": 20, "translation": {"x": 9.0, "y": 0.0, "z": 0.0}},
        ],
    }

    def test_default_outside_ranges(self):
        t, r = _correction_for_frame(5, self.CORR)
        assert t["x"] == 1.0
        assert r["z"] == 5.0

    def test_range_entry_applies_inclusively(self):
        for frame in (10, 15, 20):
            t, _ = _correction_for_frame(frame, self.CORR)
            assert t["x"] == 9.0
        t, _ = _correction_for_frame(21, self.CORR)
        assert t["x"] == 1.0

    def test_range_entry_inherits_missing_rotation_from_default(self):
        _, r = _correction_for_frame(15, self.CORR)
        assert r["z"] == 5.0

    def test_empty_correction_gives_zeros(self):
        t, r = _correction_for_frame(0, {})
        assert t == {"x": 0.0, "y": 0.0, "z": 0.0}
        assert r == {"x": 0.0, "y": 0.0, "z": 0.0}


class TestLoadGeoref:
    def test_parses_valid_lines_and_skips_junk(self, tmp_path):
        folder = tmp_path / "georeferenced_t"
        folder.mkdir()
        (folder / "georeferenced.txt").write_text(
            "# idx frame x1 y1 z1 x2 y2 z2 conf cls\n"
            "\n"
            "0 12 100.0 200.0 300.0 101.0 201.0 301.0 0.87 3\n"
            "1 13 too short\n"
            "2 14 110.0 210.0 310.0 111.0 211.0 311.0 0.55 1\n"
        )
        result = _load_georef(str(tmp_path), "t")
        assert len(result) == 2
        first = result[0]
        assert first["frame"] == 12
        assert first["x1"] == 100.0
        assert first["z2"] == 301.0
        assert first["confidence"] == 0.87
        assert first["class_id"] == 3

    def test_missing_file_returns_empty_list(self, tmp_path):
        assert _load_georef(str(tmp_path), "w") == []


class TestMatchBoxesToGeoref:
    GEOREF = [
        {"class_id": 1, "confidence": 0.90, "frame": 0},
        {"class_id": 2, "confidence": 0.90, "frame": 0},
        {"class_id": 1, "confidence": 0.50, "frame": 0},
    ]

    def test_matches_by_class_and_confidence(self):
        boxes = [(0, 0, 10, 10, 0.90, 2)]
        matched = _match_boxes_to_georef(boxes, self.GEOREF)
        assert len(matched) == 1
        assert matched[0]["class_id"] == 2

    def test_confidence_tolerance(self):
        boxes = [(0, 0, 10, 10, 0.5011, 1)]
        assert len(_match_boxes_to_georef(boxes, self.GEOREF, tol=0.0015)) == 1
        assert _match_boxes_to_georef(boxes, self.GEOREF, tol=0.0005) == []

    def test_short_tuples_are_skipped(self):
        assert _match_boxes_to_georef([(0, 0, 10, 10, 0.9)], self.GEOREF) == []


class _FakeCamera:
    """Row-vector pinhole camera: identity view, w = z projection.

    With this projection ndc = (x/z, y/z, ...), so a point at (0, 0, z>0)
    lands exactly in the image centre.
    """

    def get_view(self):
        return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    def get_proj(self):
        return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]]


class TestWorldToPixel:
    WIDTH, HEIGHT = 800, 600

    @pytest.fixture(autouse=True)
    def _numpy(self):
        pytest.importorskip("numpy")

    def test_optical_axis_maps_to_image_centre(self):
        pxs, pys = _world_to_pixel([(0.0, 0.0, 1.0)], self.WIDTH, self.HEIGHT, _FakeCamera())
        assert pxs[0] == pytest.approx(400.0)
        assert pys[0] == pytest.approx(300.0)

    def test_axis_orientation(self):
        # ndc x = +1 → right image edge; ndc y = +1 → top row (y flipped).
        pxs, pys = _world_to_pixel(
            [(1.0, 0.0, 1.0), (0.0, 1.0, 1.0)], self.WIDTH, self.HEIGHT, _FakeCamera())
        assert pxs[0] == pytest.approx(800.0)
        assert pys[0] == pytest.approx(300.0)
        assert pxs[1] == pytest.approx(400.0)
        assert pys[1] == pytest.approx(0.0)

    def test_multiple_points_no_broadcasting_bug(self):
        # Regression guard: the alfspy original had a broadcasting bug for
        # more than one point; the reimplementation must handle batches.
        points = [(0.1 * i, 0.05 * i, 1.0 + i) for i in range(8)]
        pxs, pys = _world_to_pixel(points, self.WIDTH, self.HEIGHT, _FakeCamera())
        assert len(pxs) == 8
        assert len(pys) == 8
        # Each batch result must equal the single-point projection.
        for i, point in enumerate(points):
            px, py = _world_to_pixel([point], self.WIDTH, self.HEIGHT, _FakeCamera())
            assert pxs[i] == pytest.approx(px[0])
            assert pys[i] == pytest.approx(py[0])


class TestProjectGeorefBoxToPixels:
    WIDTH, HEIGHT = 800, 600

    @pytest.fixture(autouse=True)
    def _numpy(self):
        pytest.importorskip("numpy")

    @staticmethod
    def _georef(x1, y1, z1, x2, y2, z2):
        return {"x1": x1, "y1": y1, "z1": z1, "x2": x2, "y2": y2, "z2": z2}

    def test_centred_box_projects_around_image_centre(self):
        georef = self._georef(-0.1, -0.1, 1.0, 0.1, 0.1, 1.2)
        result = _project_georef_box_to_pixels(
            georef, (0.0, 0.0, 0.0), _FakeCamera(), self.WIDTH, self.HEIGHT)
        assert result is not None
        x1, y1, x2, y2 = result
        assert x1 < 400.0 < x2
        assert y1 < 300.0 < y2

    def test_origin_offset_is_subtracted(self):
        # Same box, but expressed with a DEM origin offset of (50, 60, 70).
        georef = self._georef(49.9, 59.9, 71.0, 50.1, 60.1, 71.2)
        result = _project_georef_box_to_pixels(
            georef, (50.0, 60.0, 70.0), _FakeCamera(), self.WIDTH, self.HEIGHT)
        assert result is not None
        x1, y1, x2, y2 = result
        assert x1 < 400.0 < x2
        assert y1 < 300.0 < y2

    def test_box_outside_frustum_returns_none(self):
        georef = self._georef(100.0, 100.0, 1.0, 100.2, 100.2, 1.2)
        assert _project_georef_box_to_pixels(
            georef, (0.0, 0.0, 0.0), _FakeCamera(), self.WIDTH, self.HEIGHT) is None

    def test_result_is_clamped_to_image_bounds(self):
        # Corners at ndc ±1.001 project a fraction of a pixel outside the
        # image (still within the ±0.5 px validity band) and must be clamped
        # to [0, width-1] x [0, height-1].
        georef = self._georef(-1.001, -1.001, 1.0, 1.001, 1.001, 1.0)
        result = _project_georef_box_to_pixels(
            georef, (0.0, 0.0, 0.0), _FakeCamera(), self.WIDTH, self.HEIGHT)
        assert result == (0.0, 0.0, self.WIDTH - 1.0, self.HEIGHT - 1.0)
