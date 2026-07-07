# -*- coding: utf-8 -*-
"""Unit tests for the distance-sampling statistics in BambiProcessor.

Covers the Gaussian blur fallback (density heatmap), the lognormal
confidence interval helper, and maximum-likelihood fitting of the
half-normal / hazard-rate detection functions.
"""
import math

import pytest

np = pytest.importorskip("numpy")

from bambi_wildlife_detection.bambi_processing import BambiProcessor  # noqa: E402


class TestGaussianBlurNumpy:
    def test_zero_sigma_returns_unblurred_float64(self):
        grid = np.arange(9, dtype=np.float32).reshape(3, 3)
        out = BambiProcessor._gaussian_blur_numpy(grid, 0)
        assert out.dtype == np.float64
        assert np.array_equal(out, grid.astype(np.float64))

    def test_interior_impulse_mass_is_conserved(self):
        grid = np.zeros((41, 41))
        grid[20, 20] = 1.0
        out = BambiProcessor._gaussian_blur_numpy(grid, sigma=2.0)
        # The kernel support (3 sigma) is far from the borders, so no mass is
        # lost to edge truncation.
        assert out.sum() == pytest.approx(1.0, abs=1e-9)
        assert out[20, 20] == out.max()

    def test_blur_is_symmetric_around_impulse(self):
        grid = np.zeros((41, 41))
        grid[20, 20] = 1.0
        out = BambiProcessor._gaussian_blur_numpy(grid, sigma=3.0)
        assert np.allclose(out, out.T)
        assert np.allclose(out, out[::-1, :])
        assert np.allclose(out, out[:, ::-1])


class TestLognormalCi:
    def test_degenerate_inputs_collapse_to_estimate(self):
        assert BambiProcessor._lognormal_ci(100.0, 0.0) == [100.0, 100.0]
        assert BambiProcessor._lognormal_ci(0.0, 0.5) == [0.0, 0.0]
        assert BambiProcessor._lognormal_ci(100.0, float("nan")) == [100.0, 100.0]

    def test_interval_brackets_estimate(self):
        lo, hi = BambiProcessor._lognormal_ci(100.0, 0.3)
        assert lo < 100.0 < hi

    def test_interval_is_symmetric_in_log_space(self):
        estimate = 42.0
        lo, hi = BambiProcessor._lognormal_ci(estimate, 0.25)
        assert math.sqrt(lo * hi) == pytest.approx(estimate, rel=1e-12)

    def test_wider_cv_gives_wider_interval(self):
        lo1, hi1 = BambiProcessor._lognormal_ci(100.0, 0.1)
        lo2, hi2 = BambiProcessor._lognormal_ci(100.0, 0.5)
        assert hi2 - lo2 > hi1 - lo1


class TestFitDetectionFunction:
    W = 40.0
    SIGMA_TRUE = 10.0

    @pytest.fixture(scope="class")
    def half_normal_distances(self):
        rng = np.random.default_rng(42)
        x = np.abs(rng.normal(0.0, self.SIGMA_TRUE, size=800))
        return x[x <= self.W]

    def test_half_normal_recovers_sigma(self, processor, half_normal_distances):
        fit = processor._fit_detection_function(
            "half-normal", half_normal_distances, self.W)
        assert fit is not None
        assert fit["name"] == "half-normal"
        sigma = fit["params"]["sigma"]
        assert sigma == pytest.approx(self.SIGMA_TRUE, rel=0.15)

        # With w = 4 sigma, truncation is negligible and the analytic ESW of
        # the half-normal is sigma * sqrt(pi / 2).
        esw_analytic = sigma * math.sqrt(math.pi / 2.0)
        assert fit["esw"] == pytest.approx(esw_analytic, rel=0.02)
        assert math.isfinite(fit["aic"])
        assert fit["cv_esw"] >= 0.0
        # The fitted detection function g must satisfy g(0) == 1.
        assert float(fit["g"](0.0)) == pytest.approx(1.0)

    def test_unknown_model_returns_none(self, processor, half_normal_distances):
        assert processor._fit_detection_function(
            "uniform", half_normal_distances, self.W) is None

    def test_hazard_rate_fit_structure(self, processor, half_normal_distances):
        pytest.importorskip("scipy")
        fit = processor._fit_detection_function(
            "hazard-rate", half_normal_distances, self.W)
        assert fit is not None
        assert fit["params"]["sigma"] > 0.0
        assert fit["params"]["b"] > 1.0
        assert 0.0 < fit["esw"] <= self.W
        assert math.isfinite(fit["aic"])

    def test_log_fn_receives_summary_line(self, processor, half_normal_distances):
        messages = []
        processor._fit_detection_function(
            "half-normal", half_normal_distances, self.W, log_fn=messages.append)
        assert any("half-normal" in m and "ESW" in m for m in messages)
