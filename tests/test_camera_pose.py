# -*- coding: utf-8 -*-
"""Unit tests for core.camera_pose - the single source of pose→camera
construction shared by geo-referencing, box projection, click inspection,
labelling propagation and the correction wizard.

The numeric assertions here pin the conventions that have caused real bugs
before and MUST NOT change silently:

* the 1× rotation-correction rule (corrections subtracted exactly once),
* the corrected pose reaching ``quaternion_from_drone_pose`` in degrees, so
  the heading is applied about world up rather than the optical axis,
* the conjugation that adapts the result to a legacy alfspy's ray path.
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


def expected_pose_degrees():
    """The corrected pose, in degrees, as handed to quaternion_from_drone_pose."""
    wrapped = np.deg2rad(np.array([10.0, 20.0, 350.0]))       # -10 % 360
    corr = np.array([0.0, 0.0, 0.011])
    return np.degrees(wrapped - corr)                         # 1× correction


class TestBuildCamera:
    @pytest.fixture(autouse=True)
    def render_stack(self, monkeypatch):
        install_fake_render_stack(monkeypatch)

    def test_translation_applied_once(self):
        camera = build_camera(META, T_CORR, R_CORR)
        assert np.allclose(camera.position, [11.0, 22.0, 97.175])

    def test_rotation_correction_applied_once(self):
        camera = build_camera(META, T_CORR, R_CORR)
        # fakes.quaternion_from_drone_pose records the degrees it received.
        assert np.allclose(camera.rotation.degrees, expected_pose_degrees())
        # Regression pin: a 2x correction (the old GeoTIFF-export bug class)
        # must NOT match.
        wrong_2x = np.degrees(
            np.deg2rad([10.0, 20.0, 350.0]) - 2 * 0.011 * np.array([0, 0, 1]))
        assert not np.allclose(camera.rotation.degrees, wrong_2x)

    def test_zero_correction_keeps_pose(self):
        camera = build_camera(META, {}, {})
        assert np.allclose(camera.position, META["location"])
        assert np.allclose(camera.rotation.degrees, [10.0, 20.0, 350.0])

    def test_pose_reaches_the_drone_pose_helper_in_degrees(self):
        """Not radians - the helper's contract is degrees."""
        camera = build_camera(META, {}, {})
        assert max(abs(v) for v in camera.rotation.degrees) > 7.0

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


class TestRayConvention:
    """The camera construction must follow the installed alfspy.

    Older alfspy built world rays with ``dirs @ R33.T`` and callers negated the
    eulers to compensate; alfs_py 86e0d92 changed it to ``dirs @ R33``, at which
    point the negation silently corrupts every projection. ``build_camera``
    picks the matching construction - these tests pin the selection, and
    ``tests_qgis/test_camera_pose_conventions.py`` proves the two branches
    produce the same world rays against the real pyrr.
    """

    def test_fixed_uses_the_helper_result_directly(self, monkeypatch):
        install_fake_render_stack(monkeypatch, ray_convention="fixed")
        rotation = build_camera(META, T_CORR, R_CORR).rotation
        assert not rotation.conjugated
        assert np.allclose(rotation.degrees, expected_pose_degrees())

    def test_legacy_conjugates_the_helper_result(self, monkeypatch):
        install_fake_render_stack(monkeypatch, ray_convention="legacy")
        rotation = build_camera(META, T_CORR, R_CORR).rotation
        assert rotation.conjugated
        assert np.allclose(rotation.degrees, expected_pose_degrees())

    def test_both_branches_start_from_the_same_pose(self, monkeypatch):
        install_fake_render_stack(monkeypatch, ray_convention="legacy")
        legacy = build_camera(META, T_CORR, R_CORR).rotation
        install_fake_render_stack(monkeypatch, ray_convention="fixed")
        fixed = build_camera(META, T_CORR, R_CORR).rotation
        assert np.allclose(legacy.degrees, fixed.degrees)
        assert legacy.conjugated != fixed.conjugated

    def test_position_is_unaffected_by_the_convention(self, monkeypatch):
        install_fake_render_stack(monkeypatch, ray_convention="legacy")
        legacy = np.asarray(build_camera(META, T_CORR, R_CORR).position)
        install_fake_render_stack(monkeypatch, ray_convention="fixed")
        fixed = np.asarray(build_camera(META, T_CORR, R_CORR).position)
        assert np.allclose(legacy, fixed)


class TestWorldToPixel:
    def test_identity_camera_ndc_mapping(self):
        camera = IdentityCamera()
        pxs, pys = world_to_pixel(
            [(-0.5, -0.25, 0.0), (0.5, 0.25, 0.0)], 640, 512, camera)
        assert pxs.tolist() == [160.0, 480.0]
        assert pys.tolist() == [320.0, 192.0]
