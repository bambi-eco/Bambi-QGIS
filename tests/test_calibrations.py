# -*- coding: utf-8 -*-
"""Unit tests for the built-in camera calibration presets."""
import numpy as np
import pytest

from bambi_wildlife_detection.bambi_calibrations import (
    RGB_CALIBRATIONS,
    THERMAL_CALIBRATIONS,
)


class TestPresetContract:
    """Every preset must be a usable OpenCV calibration dict."""

    @pytest.mark.parametrize("name,calib", sorted(
        list(THERMAL_CALIBRATIONS.items()) + list(RGB_CALIBRATIONS.items())))
    def test_preset_structure(self, name, calib):
        mtx = np.asarray(calib["mtx"], dtype=float)
        assert mtx.shape == (3, 3)
        # Camera matrix invariants: positive focal lengths, affine last row
        assert mtx[0, 0] > 0 and mtx[1, 1] > 0
        assert mtx[2].tolist() == [0.0, 0.0, 1.0]
        # Distortion coefficients: k1..k4/k5, possibly nested one level
        dist = np.asarray(calib["dist"], dtype=float).ravel()
        assert dist.size >= 4

    def test_expected_default_presets_exist(self):
        # These names are hard-referenced by the integration suite and the
        # dock widget's preset combo defaults.
        assert "DJI M30T (T;Video)" in THERMAL_CALIBRATIONS
        assert "DJI M30T (W;Video)" in RGB_CALIBRATIONS

    def test_thermal_and_rgb_preset_names_are_disjoint(self):
        assert not set(THERMAL_CALIBRATIONS) & set(RGB_CALIBRATIONS)
