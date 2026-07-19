# -*- coding: utf-8 -*-
"""Unit tests for core.thermal_curve (tone curve + flight temperature scan)."""
import numpy as np
import pytest

from bambi_wildlife_detection.core.thermal_curve import (
    DEFAULT_POINTS,
    TemperatureScan,
    ThermalCurve,
    normalize_points,
    scan_temperatures,
)


class TestNormalizePoints:
    def test_default_points_pass_through(self):
        assert normalize_points(DEFAULT_POINTS) == [(0.0, 0.0), (1.0, 1.0)]

    def test_sorted_by_x(self):
        pts = normalize_points([(1.0, 1.0), (0.5, 0.2), (0.0, 0.0)])
        assert [p[0] for p in pts] == sorted(p[0] for p in pts)

    def test_clipped_to_unit_square(self):
        pts = normalize_points([(-0.5, -2.0), (0.5, 3.0), (1.5, 0.5)])
        for x, y in pts:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_endpoints_inserted_when_missing(self):
        pts = normalize_points([(0.3, 0.1), (0.7, 0.9)])
        assert pts[0] == (0.0, 0.1)
        assert pts[-1] == (1.0, 0.9)

    def test_near_duplicate_x_dropped(self):
        pts = normalize_points([(0.0, 0.0), (0.5, 0.2), (0.501, 0.9), (1.0, 1.0)])
        assert len(pts) == 3

    def test_garbage_falls_back_to_default(self):
        assert normalize_points([("a", None)]) == list(DEFAULT_POINTS)
        assert normalize_points([]) == list(DEFAULT_POINTS)


class TestThermalCurveLut:
    def test_identity_curve_is_linear(self):
        curve = ThermalCurve(0.0, 40.0)
        lut = curve.lut(101)
        assert np.allclose(lut, np.linspace(0.0, 1.0, 101), atol=1e-9)

    def test_lut_hits_control_points(self):
        pts = [(0.0, 0.0), (0.25, 0.6), (1.0, 1.0)]
        curve = ThermalCurve(0.0, 40.0, pts)
        lut = curve.lut(1001)
        xs = np.linspace(0.0, 1.0, 1001)
        for x, y in pts:
            idx = int(round(x * 1000))
            assert lut[idx] == pytest.approx(y, abs=1e-6)
            assert xs[idx] == pytest.approx(x, abs=1e-6)

    def test_monotone_points_give_monotone_lut(self):
        curve = ThermalCurve(0.0, 40.0,
                             [(0.0, 0.0), (0.2, 0.05), (0.4, 0.7),
                              (0.8, 0.9), (1.0, 1.0)])
        lut = curve.lut(512)
        assert np.all(np.diff(lut) >= -1e-12)

    def test_lut_stays_in_unit_interval(self):
        curve = ThermalCurve(0.0, 40.0,
                             [(0.0, 1.0), (0.5, 0.0), (1.0, 1.0)])
        lut = curve.lut(512)
        assert lut.min() >= 0.0
        assert lut.max() <= 1.0

    def test_two_point_flat_curve(self):
        curve = ThermalCurve(0.0, 40.0, [(0.0, 0.5), (1.0, 0.5)])
        assert np.allclose(curve.lut(64), 0.5)


class TestThermalCurveApply:
    def test_identity_maps_domain_linearly(self):
        curve = ThermalCurve(10.0, 30.0)
        arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        out = curve.apply(arr)
        assert out == pytest.approx([0.0, 0.5, 1.0], abs=1e-3)

    def test_out_of_domain_clamps_to_endpoint_values(self):
        # Inverted endpoints: cold → bright, hot → dark
        curve = ThermalCurve(10.0, 30.0, [(0.0, 0.9), (1.0, 0.1)])
        out = curve.apply(np.array([-50.0, 100.0], dtype=np.float32))
        assert out[0] == pytest.approx(0.9, abs=1e-3)
        assert out[1] == pytest.approx(0.1, abs=1e-3)

    def test_degenerate_domain_is_widened(self):
        curve = ThermalCurve(20.0, 20.0)
        assert curve.domain_hi > curve.domain_lo
        out = curve.apply(np.array([20.0], dtype=np.float32))
        assert np.all(np.isfinite(out))

    def test_preserves_shape(self):
        curve = ThermalCurve(0.0, 40.0)
        out = curve.apply(np.zeros((7, 5), dtype=np.float32))
        assert out.shape == (7, 5)


class TestSerialisation:
    def test_round_trip(self):
        curve = ThermalCurve(5.5, 38.0, [(0.0, 0.0), (0.4, 0.7), (1.0, 1.0)])
        restored = ThermalCurve.from_dict(curve.to_dict())
        assert restored.domain_lo == curve.domain_lo
        assert restored.domain_hi == curve.domain_hi
        assert restored.points == curve.points

    def test_from_dict_defaults(self):
        curve = ThermalCurve.from_dict({})
        assert curve.domain_lo < curve.domain_hi
        assert curve.points == list(DEFAULT_POINTS)

    def test_describe_and_identity(self):
        curve = ThermalCurve(0.0, 40.0)
        assert curve.is_identity()
        assert "linear" in curve.describe()
        bent = ThermalCurve(0.0, 40.0, [(0.0, 0.0), (0.5, 0.8), (1.0, 1.0)])
        assert not bent.is_identity()
        assert "3 points" in bent.describe()


class TestTemperatureScan:
    def _fake_parse(self, path):
        # path encodes the image content: "lo:hi"
        lo, hi = (float(v) for v in path.split(":"))
        return np.linspace(lo, hi, 1000, dtype=np.float32).reshape(20, 50)

    def test_exact_min_max(self):
        scan = scan_temperatures(["5:20", "8:31.5", "-2:10"], self._fake_parse)
        assert scan.n_images == 3
        assert scan.minimum == pytest.approx(-2.0)
        assert scan.maximum == pytest.approx(31.5)

    def test_histogram_counts_all_pixels(self):
        scan = scan_temperatures(["0:10", "0:10"], self._fake_parse)
        assert int(scan.counts.sum()) == 2000

    def test_percentiles_are_ordered_and_within_range(self):
        scan = scan_temperatures(["0:100"], self._fake_parse)
        p1, p50, p99 = scan.percentile(1), scan.percentile(50), scan.percentile(99)
        assert p1 < p50 < p99
        assert p1 == pytest.approx(1.0, abs=1.0)
        assert p50 == pytest.approx(50.0, abs=1.0)
        assert p99 == pytest.approx(99.0, abs=1.0)

    def test_suggested_domain_exact_vs_robust(self):
        scan = scan_temperatures(["0:100"], self._fake_parse)
        exact = scan.suggested_domain(robust=False)
        robust = scan.suggested_domain(robust=True)
        assert exact == pytest.approx((0.0, 100.0))
        assert robust[0] > exact[0]
        assert robust[1] < exact[1]

    def test_parse_errors_are_skipped(self):
        def parse(path):
            if path == "bad":
                raise RuntimeError("boom")
            return self._fake_parse(path)

        scan = scan_temperatures(["0:10", "bad", "5:15"], parse)
        assert scan.n_images == 2
        assert scan.n_errors == 1

    def test_cancel_stops_early(self):
        seen = []

        def cancel():
            return len(seen) >= 2

        def progress(i, total, path):
            seen.append(path)

        scan = scan_temperatures(["0:1", "0:1", "0:1", "0:1"],
                                 self._fake_parse,
                                 progress_cb=progress, cancel_cb=cancel)
        assert scan.cancelled
        assert scan.n_images == 2

    def test_empty_scan_has_no_domain(self):
        scan = TemperatureScan()
        assert scan.suggested_domain() is None
        assert scan.percentile(50) is None

    def test_nonfinite_pixels_ignored(self):
        def parse(path):
            arr = np.full((4, 4), np.nan, dtype=np.float32)
            if path == "some":
                arr[0, 0] = 12.0
                arr[0, 1] = 18.0
            return arr

        scan = scan_temperatures(["some", "allnan"], parse)
        assert scan.n_images == 1
        assert scan.minimum == pytest.approx(12.0)
        assert scan.maximum == pytest.approx(18.0)
