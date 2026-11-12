# -*- coding: utf-8 -*-
"""Correction editing of the 3D Flight Viewer (core/flight_corrections.py)."""
import json
import math

import numpy as np
import pytest

from bambi_wildlife_detection.core import flight_corrections as fc
from bambi_wildlife_detection.core import flight_viewer_io as fio
from bambi_wildlife_detection.core.corrections import correction_for_frame

CORR = {"translation": {"x": 1, "y": 2, "z": 3},
        "rotation": {"x": 0, "y": 0, "z": 0.1},
        "additional": [{"start": 10, "end": 20,
                        "translation": {"x": 5}, "rotation": {"z": 0.2}}]}


def test_normalized_fills_missing_axes():
    n = fc.normalized(CORR)
    assert n["additional"][0]["translation"] == {"x": 5.0, "y": 0.0, "z": 0.0}
    assert fc.normalized(None)["rotation"] == {"x": 0.0, "y": 0.0, "z": 0.0}


def test_entry_for_frame_matches_the_lookup_the_pipeline_uses():
    for frame in (0, 10, 15, 20, 21):
        entry = fc.entry_for_frame(CORR, frame)
        t, _ = correction_for_frame(frame, CORR)
        assert (entry is not None) == (t.get("x") == 5)


def test_set_entry_and_ranges_do_not_mutate_the_input():
    out = fc.set_entry(CORR, 0, {"x": 9}, {"z": 0.3})
    assert out["additional"][0]["translation"]["x"] == 9
    assert CORR["additional"][0]["translation"] == {"x": 5}
    out, idx = fc.add_range(out, 40, 30)
    assert idx == 1 and out["additional"][1]["start"] == 30
    assert out["additional"][1]["translation"] == fc._xyz(CORR["translation"])
    assert len(fc.remove_range(out, 0)["additional"]) == 1


def test_vector_round_trip():
    t, r = fc.from_vector(fc.to_vector({"x": 1, "y": 2, "z": 3},
                                       {"x": 4, "y": 5, "z": 6}))
    assert t == {"x": 1, "y": 2, "z": 3} and r == {"x": 4, "y": 5, "z": 6}


def test_write_correction_keeps_unknown_keys_and_backs_up(tmp_path):
    path = tmp_path / "correction.json"
    path.write_text(json.dumps({"translation": {"x": 0}, "note": "keep me"}))
    backup = fc.write_correction(str(path), CORR)
    data = json.loads(path.read_text())
    assert data["note"] == "keep me"
    assert data["translation"] == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert data["additional"][0]["start"] == 10
    assert json.loads(open(backup).read())["note"] == "keep me"
    assert fc.write_correction(str(tmp_path / "new.json"), CORR) == ""


def test_correction_path_prefers_the_configured_file(tmp_path):
    explicit = tmp_path / "c.json"
    explicit.write_text("{}")
    assert fc.correction_path({"correction_path": str(explicit)}) == str(explicit)
    assert fc.correction_path({"target_folder": str(tmp_path)}).endswith(
        "correction.json")


def test_sharpness_prefers_detail_and_ignores_coverage_edges():
    rng = np.random.default_rng(0)
    sharp = np.zeros((60, 60, 4))
    sharp[..., :3] = rng.uniform(0, 255, (60, 60, 1))
    sharp[..., 3] = 255
    blurred = sharp.copy()
    k = np.ones(5) / 5
    for c in range(3):
        for axis in (0, 1):
            blurred[..., c] = np.apply_along_axis(
                lambda v: np.convolve(v, k, mode="same"), axis, blurred[..., c])
    assert fc.sharpness(sharp) > 5 * fc.sharpness(blurred)
    flat = np.zeros((60, 60, 4))
    flat[:, :30, :3] = 200
    flat[..., 3] = 255
    flat[:, 30:, 3] = 0                    # the step is the coverage edge
    assert fc.sharpness(flat) == pytest.approx(0.0)
    assert math.isnan(fc.sharpness(np.zeros((60, 60, 4))))


def test_pattern_search_finds_a_quadratic_minimum():
    target = np.array([0.7, -1.3, 0.2, 0.01, 0.0, -0.02])
    f = lambda x: float(np.sum(((x - target) / fc.default_steps()) ** 2))
    x, best, evals = fc.pattern_search(f, np.zeros(6), fc.default_steps(),
                                       min_scale=1 / 256, max_evals=2000)
    np.testing.assert_allclose(x, target, atol=0.02)
    assert best < 1e-2


def test_pattern_search_only_moves_active_parameters():
    f = lambda x: float(np.sum((x - 1.0) ** 2))
    x, _, _ = fc.pattern_search(f, np.zeros(6), np.ones(6),
                                active=[False, False, True, False, False, True])
    np.testing.assert_allclose(x, [0, 0, 1, 0, 0, 1], atol=1e-6)


def test_pattern_search_stops_when_asked():
    calls = []
    fc.pattern_search(lambda x: calls.append(1) or 0.0, np.zeros(6), np.ones(6),
                      should_stop=lambda: len(calls) >= 3)
    assert len(calls) == 3


def test_refine_windows_include_the_current_frame():
    assert fc.refine_windows(100, 42, 3, 5) == [5, 42, 94]
    assert fc.refine_windows(100, 42, 1, 5) == [42]
    assert fc.refine_windows(0, 0, 3, 5) == []


def test_mark_pose_results_stale(tmp_path, monkeypatch):
    from bambi_wildlife_detection.core import stages
    calls = []
    monkeypatch.setattr(stages, "states", lambda folder, m: {
        "georeference": {"state": stages.COMPLETE, "row_count": 3,
                         "fingerprint": "f"},
        "alfs": {"state": stages.STALE}} if m == "t" else {})
    monkeypatch.setattr(stages, "set_state",
                        lambda *a, **k: calls.append(a))
    monkeypatch.setattr(stages, "mark_dependents_stale",
                        lambda folder, stage, m: ["tracking"])
    flagged = fc.mark_pose_results_stale(str(tmp_path))
    assert flagged == ["georeference (t)", "tracking (t)"]
    assert calls == [(str(tmp_path), "georeference", "t", stages.STALE)]


# -- io --------------------------------------------------------------------------

def test_png_writer_produces_a_readable_file(tmp_path):
    img = np.zeros((7, 5, 4), np.uint8)
    img[2, 3] = (10, 20, 30, 255)
    path = tmp_path / "x.png"
    fio.write_png(str(path), img)
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG") and b"IEND" in data
    cv2 = pytest.importorskip("cv2")
    back = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(back[2, 3], (30, 20, 10, 255))   # BGRA


def test_world_file_places_the_pixel_centres(tmp_path):
    png = tmp_path / "a.png"
    world = fio.write_world_file(str(png), [0, 0], [100, 50], 200, 100,
                                 origin=(1000, 2000, 0), epsg=32633)
    vals = [float(v) for v in open(world).read().split()]
    assert vals == pytest.approx([0.5, 0, 0, -0.5, 1000.25, 2049.75])
    assert "EPSG:32633" in open(str(png) + ".aux.xml").read()


def test_ply_round_trip(tmp_path):
    pts = np.random.default_rng(1).normal(size=(10, 3))
    cols = np.arange(30, dtype=np.uint8).reshape(10, 3)
    fio.write_ply(str(tmp_path / "p.ply"), pts, cols)
    p, c = fio.read_ply(str(tmp_path / "p.ply"))
    np.testing.assert_allclose(p, pts, atol=1e-6)
    np.testing.assert_array_equal(c, cols)


# -- orthomosaic -------------------------------------------------------------------

def test_find_orthomosaic(tmp_path):
    assert fio.find_orthomosaic(str(tmp_path), "w") == ""
    folder = tmp_path / "orthomosaic_w"
    folder.mkdir()
    (folder / "orthomosaic.tif").write_bytes(b"x")
    assert fio.find_orthomosaic(str(tmp_path), "w").endswith("orthomosaic.tif")
    assert fio.find_orthomosaic(str(tmp_path), "t") == ""


def test_read_raster_rgba_downsamples_and_keeps_the_extent(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    data = np.zeros((3, 200, 400), np.uint8)
    data[0, :, :200] = 250                       # west half red
    data[1, :, 200:] = 250                       # east half green
    data[:, :20, :] = 0                          # north strip: nodata
    path = str(tmp_path / "o.tif")
    with rasterio.open(path, "w", driver="GTiff", width=400, height=200, count=3,
                       dtype="uint8", nodata=0, crs="EPSG:32633",
                       transform=from_origin(1000.0, 5200.0, 0.5, 0.5)) as dst:
        dst.write(data)
    rgba, lower, upper = fio.read_raster_rgba(path, max_edge=100)
    assert rgba.shape == (50, 100, 4)
    assert lower == pytest.approx((1000.0, 5100.0)) and upper == pytest.approx((1200.0, 5200.0))
    assert rgba[2, 50, 3] == 0 and rgba[30, 20, 3] == 255     # nodata transparent
    assert rgba[30, 20, 0] > 200 and rgba[30, 80, 1] > 200


def test_single_band_float_raster_is_stretched():
    bands = np.linspace(10.0, 30.0, 12, dtype=np.float32).reshape(1, 3, 4)
    out = fio._to_rgba(bands, np.ones((3, 4), bool))
    assert out.dtype == np.uint8 and out[0, 0, 0] < 20 and out[2, 3, 0] > 235
    assert (out[..., 0] == out[..., 1]).all()
