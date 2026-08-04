# -*- coding: utf-8 -*-
"""Unit tests for the geo-referencing miss diagnostics.

When a detection's rays never intersect the DEM mesh it is silently dropped;
these helpers explain the shortfall in the step log.
"""
import numpy as np
import pytest

from bambi_wildlife_detection.bambi_processing import BambiProcessor


class _Rotation:
    def __init__(self, matrix33):
        self.matrix33 = np.asarray(matrix33, dtype=float)


class _Transform:
    def __init__(self, position, matrix33):
        self.position = np.asarray(position, dtype=float)
        self.rotation = _Rotation(matrix33)


class _Camera:
    """Minimal stand-in for ``alfspy.core.rendering.Camera``."""

    def __init__(self, position, matrix33, fovy=60.0, aspect_ratio=1.0):
        self.transform = _Transform(position, matrix33)
        self.fovy = fovy
        self.aspect_ratio = aspect_ratio


class _Resolution:
    def __init__(self, width, height):
        self.width = width
        self.height = height


RES = _Resolution(1024, 1024)
# Mesh spanning 0..500 m in both axes at height 0.
BOUNDS = (0.0, 500.0, 0.0, 500.0, 0.0)
# Centre pixel box.
CENTRE_BOX = [500.0, 500.0, 524.0, 500.0, 524.0, 524.0, 500.0, 524.0]
# Box well inside the upper half of the frame.
UPPER_BOX = [500.0, 100.0, 524.0, 100.0, 524.0, 124.0, 500.0, 124.0]

NADIR = np.eye(3)  # camera -Z axis points straight down
# 90° pitch up about X: the camera's -Z now points along +Y (horizontal).
LEVEL = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]])


class TestClassifyProjectionMiss:
    def test_level_camera_upper_frame_reports_above_horizon(self):
        camera = _Camera(position=[250.0, 250.0, 50.0], matrix33=LEVEL)

        reason, ground_range = BambiProcessor._classify_projection_miss(
            camera, UPPER_BOX, RES, BOUNDS)

        assert reason == "above_horizon"
        assert ground_range is None

    def test_level_camera_straddling_horizon_prefers_beyond_mesh(self):
        """A box on the horizon has corners both rising and hitting far ground.

        The distance is the more actionable of the two, so it wins.
        """
        camera = _Camera(position=[250.0, 250.0, 50.0], matrix33=LEVEL)

        reason, ground_range = BambiProcessor._classify_projection_miss(
            camera, CENTRE_BOX, RES, BOUNDS)

        assert reason == "beyond_mesh"
        assert ground_range > 500.0

    def test_nadir_camera_outside_mesh_reports_beyond_mesh(self):
        # Straight down, but positioned well past the mesh's +X edge.
        camera = _Camera(position=[900.0, 250.0, 50.0], matrix33=NADIR)

        reason, ground_range = BambiProcessor._classify_projection_miss(
            camera, CENTRE_BOX, RES, BOUNDS)

        assert reason == "beyond_mesh"
        # Looking straight down, the ground point is directly beneath.
        assert ground_range == pytest.approx(0.0, abs=2.0)

    def test_ground_range_grows_with_obliquity(self):
        """A shallower look angle must report a longer ground range."""
        tilt = np.radians(70.0)  # tip the camera towards the horizon
        rot = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(tilt), -np.sin(tilt)],
                        [0.0, np.sin(tilt), np.cos(tilt)]])
        camera = _Camera(position=[250.0, 900.0, 50.0], matrix33=rot)

        reason, ground_range = BambiProcessor._classify_projection_miss(
            camera, CENTRE_BOX, RES, BOUNDS)

        assert reason == "beyond_mesh"
        # 50 m up at a 20° depression lands roughly 137 m away.
        assert ground_range == pytest.approx(50.0 / np.tan(np.radians(20.0)), rel=0.15)

    def test_ray_landing_inside_footprint_reports_other(self):
        camera = _Camera(position=[250.0, 250.0, 50.0], matrix33=NADIR)

        reason, ground_range = BambiProcessor._classify_projection_miss(
            camera, CENTRE_BOX, RES, BOUNDS)

        assert reason == "other"
        assert ground_range is not None


class TestMissSummaryLog:
    @staticmethod
    def _capture(miss_stats, miss_ranges=(), total=100):
        lines = []
        BambiProcessor._log_projection_miss_summary(
            miss_stats, list(miss_ranges), total, lines.append)
        return "\n".join(lines)

    def test_silent_when_nothing_dropped(self):
        text = self._capture({"above_horizon": 0, "beyond_mesh": 0,
                              "other": 0, "no_pose": 0})
        assert text == ""

    def test_reports_each_category_and_padding_hint(self):
        text = self._capture(
            {"above_horizon": 20, "beyond_mesh": 30, "other": 5, "no_pose": 5},
            miss_ranges=[100.0, 200.0, 300.0],
            total=100,
        )

        assert "60 of 100 detections (60.0%)" in text
        assert "20 above the horizon" in text
        assert "30 beyond the DEM edge" in text
        assert "median 200 m" in text
        assert "Padding" in text
        assert "5 at grazing angles" in text
        assert "5 reference frames that have no pose" in text

    def test_omits_categories_with_no_hits(self):
        text = self._capture({"above_horizon": 7, "beyond_mesh": 0,
                              "other": 0, "no_pose": 0})

        assert "above the horizon" in text
        assert "beyond the DEM edge" not in text
        assert "Padding" not in text
