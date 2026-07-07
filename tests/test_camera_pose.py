# -*- coding: utf-8 -*-
"""Unit tests for core.camera_pose — the single source of pose→camera
construction shared by geo-referencing, box projection, click inspection,
labelling propagation and the correction wizard.

The numeric assertions here pin the two conventions that have caused real
bugs before and MUST NOT change silently:

* the 1× rotation-correction rule (corrections subtracted exactly once),
* the euler negation (``* -1``) converting pose rotation to camera rotation.
"""
import sys

import numpy as np
import pytest

from bambi_wildlife_detection.core.camera_pose import (
    build_camera,
    frame_camera,
    world_to_pixel,
)
from tests.fakes import IdentityCamera, install_fake_render_stack

META = {
    "location": [10.0, 20.0, 100.0],
    "rotation": [10.0, 20.0, -10.0],   # degrees; negative exercises % 360
    "fovy": [50.0],
}
T_CORR = {"x": 1.0, "y": 2.0, "z": -2.825}
R_CORR = {"x": 0.0, "y": 0.0, "z": 0.011}


def expected_eulers():
    wrapped = np.deg2rad(np.array([10.0, 20.0, 350.0]))       # -10 % 360
    corr = np.array([0.0, 0.0, 0.011])
    return -1 * (wrapped - corr)                              # 1× correction


class TestBuildCamera:
    @pytest.fixture(autouse=True)
    def render_stack(self, monkeypatch):
        install_fake_render_stack(monkeypatch)

    def test_translation_applied_once(self):
        camera = build_camera(META, T_CORR, R_CORR)
        assert np.allclose(camera.position, [11.0, 22.0, 97.175])

    def test_rotation_correction_applied_once_and_negated(self):
        camera = build_camera(META, T_CORR, R_CORR)
        # fakes.Quaternion.from_eulers returns its input as a tuple, so the
        # camera's rotation exposes the euler maths directly.
        assert np.allclose(camera.rotation, expected_eulers())
        # Regression pin: a 2x correction (the old GeoTIFF-export bug class)
        # must NOT match.
        wrong_2x = -1 * (np.deg2rad([10.0, 20.0, 350.0]) - 2 * 0.011 * np.array([0, 0, 1]))
        assert not np.allclose(camera.rotation, wrong_2x)

    def test_zero_correction_keeps_pose(self):
        camera = build_camera(META, {}, {})
        assert np.allclose(camera.position, META["location"])
        assert np.allclose(camera.rotation,
                           -np.deg2rad([10.0, 20.0, 350.0]))

    @pytest.mark.parametrize("fovy_raw,expected", [
        ([50.0], 50.0),
        ((42.0, 1.0), 42.0),
        (35, 35.0),
        ([], 50.0),
        (None, None),   # sentinel: key removed entirely
    ])
    def test_fovy_variants(self, fovy_raw, expected):
        meta = dict(META)
        if fovy_raw is None:
            del meta["fovy"]
            expected = 50.0
        else:
            meta["fovy"] = fovy_raw
        assert build_camera(meta, {}, {}).fovy == expected

    def test_aspect_ratio_forwarded(self):
        camera = build_camera(META, {}, {}, aspect_ratio=640 / 512)
        assert camera.aspect_ratio == pytest.approx(1.25)

    def test_missing_render_stack_raises_runtime_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyrr", None)
        with pytest.raises(RuntimeError, match="not available"):
            build_camera(META, {}, {})


class TestFrameCamera:
    @pytest.fixture(autouse=True)
    def render_stack(self, monkeypatch):
        install_fake_render_stack(monkeypatch)

    def test_uses_per_frame_correction_ranges(self):
        correction = {
            "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "additional": [{
                "start": 10, "end": 20,
                "translation": {"x": 5.0, "y": 0.0, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            }],
        }
        images = [dict(META) for _ in range(30)]
        default_cam = frame_camera(images, 5, correction)
        ranged_cam = frame_camera(images, 15, correction)
        assert np.allclose(default_cam.position, META["location"])
        assert np.allclose(
            ranged_cam.position,
            np.array(META["location"]) + [5.0, 0.0, 0.0])

    def test_none_correction_equals_empty(self):
        images = [dict(META)]
        assert np.allclose(
            frame_camera(images, 0, None).position,
            frame_camera(images, 0, {}).position)


class TestWorldToPixel:
    def test_identity_camera_ndc_mapping(self):
        camera = IdentityCamera()
        pxs, pys = world_to_pixel(
            [(-0.5, -0.25, 0.0), (0.5, 0.25, 0.0)], 640, 512, camera)
        assert pxs.tolist() == [160.0, 480.0]
        assert pys.tolist() == [320.0, 192.0]
