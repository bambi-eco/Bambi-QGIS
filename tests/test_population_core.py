# -*- coding: utf-8 -*-
"""Unit tests for core.population (transect-based population estimation).

The estimators are ports of the R analysis in ``scripts/run_analysis.R``
(Praschl et al. 2026), so the naive/bootstrap tests pin the exact formulas
and the ZINB test checks the fit recovers a known simulated density.
"""
import math

import numpy as np
import pytest

from bambi_wildlife_detection.core.population import (
    HA_PER_KM2,
    assign_tracks,
    estimate_bootstrap,
    estimate_naive,
    estimate_population,
    estimate_zinb,
    geometry_to_rings,
    merged_fov_area,
    point_to_polyline_distance,
    transect_centerline,
)


def _images_along_x(n, spacing=10.0):
    return [{"location": [i * spacing, 0.0, 100.0]} for i in range(n)]


def _square(cx, cy, half=5.0):
    """A square FoV footprint as (x, y, z) triples."""
    return [(cx - half, cy - half, 0.0), (cx + half, cy - half, 0.0),
            (cx + half, cy + half, 0.0), (cx - half, cy + half, 0.0)]


class TestTransectCenterline:
    def test_adds_dem_origin_offset(self):
        images = _images_along_x(3)
        line = transect_centerline(images, 0, 2, x_offset=1000.0, y_offset=2000.0)
        assert line == [(1000.0, 2000.0), (1010.0, 2000.0), (1020.0, 2000.0)]

    def test_reversed_range_and_clamping(self):
        images = _images_along_x(3)
        assert transect_centerline(images, 2, 0) == transect_centerline(images, 0, 2)
        assert len(transect_centerline(images, 0, 99)) == 3

    def test_skips_frames_without_location(self):
        images = [{"location": [0, 0, 1]}, {}, {"location": [10, 0, 1]}]
        assert transect_centerline(images, 0, 2) == [(0.0, 0.0), (10.0, 0.0)]


class TestPointToPolylineDistance:
    def test_perpendicular_to_a_segment(self):
        line = [(0.0, 0.0), (100.0, 0.0)]
        dist, foot = point_to_polyline_distance(50.0, 30.0, line)
        assert dist == pytest.approx(30.0)
        assert foot == pytest.approx((50.0, 0.0))

    def test_projection_is_clamped_to_the_segment(self):
        # Beyond the end of the line the distance is to the end point, not to
        # the infinite line through it (which would give 5.0).
        line = [(0.0, 0.0), (100.0, 0.0)]
        dist, foot = point_to_polyline_distance(130.0, 5.0, line)
        assert dist == pytest.approx(math.hypot(30.0, 5.0))
        assert foot == pytest.approx((100.0, 0.0))

    def test_empty_and_single_point_lines(self):
        assert point_to_polyline_distance(0, 0, [])[0] == float("inf")
        dist, foot = point_to_polyline_distance(3.0, 4.0, [(0.0, 0.0)])
        assert dist == pytest.approx(5.0)
        assert foot == (0.0, 0.0)


class TestAssignTracks:
    def _lines(self):
        # Two parallel transects 100 m apart, both running east
        return {
            1: [(0.0, 0.0), (100.0, 0.0)],
            2: [(0.0, 100.0), (100.0, 100.0)],
        }

    def _track(self, tid, x, y, frame=0):
        return {"track_id": tid, "last_frame": frame, "class_id": 0,
                "detection_center": [x, y, 0.0]}

    def test_assigns_to_the_nearest_transect(self):
        tracks = [self._track(1, 50.0, 10.0), self._track(2, 50.0, 90.0)]
        out = assign_tracks(tracks, self._lines())
        assert [a["transect_id"] for a in out] == [1, 2]
        assert out[0]["distance_m"] == pytest.approx(10.0)
        assert out[1]["distance_m"] == pytest.approx(10.0)

    def test_records_distances_to_every_transect(self):
        out = assign_tracks([self._track(1, 50.0, 10.0)], self._lines())
        assert out[0]["distances"] == {1: pytest.approx(10.0),
                                       2: pytest.approx(90.0)}

    def test_truncation_leaves_far_tracks_unassigned(self):
        tracks = [self._track(1, 50.0, 10.0), self._track(2, 50.0, 50.0)]
        out = assign_tracks(tracks, self._lines(), truncation=20.0)
        assert out[0]["transect_id"] == 1
        assert out[1]["transect_id"] is None
        assert out[1]["truncated"] is True
        assert out[1]["distance_m"] == pytest.approx(50.0)

    def test_zero_truncation_keeps_everything(self):
        out = assign_tracks([self._track(1, 50.0, 5000.0)], self._lines(),
                            truncation=0.0)
        assert out[0]["transect_id"] == 2
        assert out[0]["truncated"] is False

    def test_tracks_without_a_position_are_skipped(self):
        tracks = [{"track_id": 9, "detection_center": []},
                  self._track(1, 50.0, 10.0)]
        out = assign_tracks(tracks, self._lines())
        assert [a["track_id"] for a in out] == [1]

    def test_no_transects_yields_unassigned(self):
        out = assign_tracks([self._track(1, 0.0, 0.0)], {})
        assert out[0]["transect_id"] is None
        assert out[0]["distance_m"] is None


class TestMergedFovArea:
    def test_overlapping_footprints_are_unioned_not_summed(self):
        # Two 10x10 squares offset by 5 m: union = 150 m², sum would be 200 m²
        polygons = {0: _square(0.0, 0.0), 1: _square(5.0, 0.0)}
        area, geom = merged_fov_area(polygons, [0, 1])
        assert area == pytest.approx(150.0)
        assert geom is not None

    def test_frames_without_a_footprint_are_skipped(self):
        polygons = {0: _square(0.0, 0.0)}
        area, _ = merged_fov_area(polygons, [0, 1, 2])
        assert area == pytest.approx(100.0)

    def test_no_usable_footprint(self):
        area, geom = merged_fov_area({}, [0, 1])
        assert area == 0.0
        assert geom is None
        # a degenerate polygon (< 3 points) is not usable either
        area, geom = merged_fov_area({0: [(0, 0, 0), (1, 1, 0)]}, [0])
        assert (area, geom) == (0.0, None)

    def test_geometry_to_rings(self):
        _area, geom = merged_fov_area({0: _square(0.0, 0.0)}, [0])
        rings = geometry_to_rings(geom)
        assert len(rings) == 1
        assert rings[0][0] == rings[0][-1]      # closed ring
        assert geometry_to_rings(None) == []


class TestEstimateNaive:
    def test_matches_the_r_formula(self):
        # sum(count)/sum(ha)*100 = 6/4*100 = 150 animals per 100 ha
        r = estimate_naive([1, 2, 3], [1.0, 1.0, 2.0])
        assert r["density_per_100ha"] == pytest.approx(150.0)
        assert r["total_count"] == 6
        assert r["total_ha"] == 4.0
        assert r["error"] is None

    def test_zero_area_is_an_error_not_a_crash(self):
        r = estimate_naive([1], [0.0])
        assert r["density_per_100ha"] is None
        assert "zero" in r["error"]


class TestEstimateBootstrap:
    def test_centres_on_the_naive_density(self):
        counts = [2, 3, 1, 4, 2, 3]
        areas = [1.0] * 6
        naive = estimate_naive(counts, areas)["density_per_100ha"]
        r = estimate_bootstrap(counts, areas, n_boot=500, seed=1)
        assert r["density_per_100ha"] == pytest.approx(naive, rel=0.1)
        assert r["ci95"][0] < naive < r["ci95"][1]
        assert r["n_valid"] == 500

    def test_is_reproducible_for_a_seed(self):
        counts, areas = [0, 5, 2, 1], [1.0, 2.0, 1.5, 0.5]
        a = estimate_bootstrap(counts, areas, n_boot=200, seed=7)
        b = estimate_bootstrap(counts, areas, n_boot=200, seed=7)
        c = estimate_bootstrap(counts, areas, n_boot=200, seed=8)
        assert a["density_per_100ha"] == b["density_per_100ha"]
        assert a["density_per_100ha"] != c["density_per_100ha"]

    def test_needs_at_least_two_transects(self):
        r = estimate_bootstrap([3], [1.0])
        assert r["density_per_100ha"] is None
        assert "at least 2" in r["error"]


class TestEstimateZinb:
    def test_recovers_a_simulated_density(self):
        # Simulate ZINB counts: 30 % structural zeros, mean count ~ 4 per ha
        rng = np.random.default_rng(12345)
        n = 60
        areas = rng.uniform(0.5, 3.0, size=n)
        mu = np.exp(0.2 + 1.0 * areas)
        counts = rng.negative_binomial(5, 5 / (5 + mu))
        counts = np.where(rng.random(n) < 0.3, 0, counts)

        r = estimate_zinb(counts.tolist(), areas.tolist())
        assert r["error"] is None
        assert r["converged"] is True

        # The ZINB density must land near the naive one on data this dense,
        # and the CI must be a finite interval around it.
        naive = estimate_naive(counts.tolist(), areas.tolist())["density_per_100ha"]
        assert r["density_per_100ha"] == pytest.approx(naive, rel=0.25)
        assert r["se"] > 0
        assert r["ci95"][0] < r["density_per_100ha"] < r["ci95"][1]
        assert 0.0 <= r["params"]["zero_inflation_prob"] <= 1.0
        assert math.isfinite(r["aic"])

    def test_needs_enough_transects(self):
        r = estimate_zinb([1, 0, 2], [1.0, 1.0, 1.0])
        assert r["density_per_100ha"] is None
        assert "at least 4" in r["error"]

    def test_constant_area_has_no_variance_to_regress_on(self):
        r = estimate_zinb([1, 0, 2, 3, 0], [1.0] * 5)
        assert r["density_per_100ha"] is None
        assert "same area" in r["error"]


class TestZinbBoundaries:
    """The two ZINB parameters whose MLE can run to infinity must be flagged."""

    def test_no_excess_zeros_flags_the_zero_inflation_boundary(self):
        # Counts with no zeros at all: p_zero -> 0, the fit degenerates to NB
        rng = np.random.default_rng(4)
        areas = rng.uniform(1.0, 3.0, size=40)
        counts = rng.integers(3, 12, size=40)

        r = estimate_zinb(counts.tolist(), areas.tolist())
        assert r["zero_inflation_at_boundary"] is True
        assert r["params"]["zero_inflation_prob"] < 1e-6
        # The density survives — only the CI is affected by the degeneracy
        assert r["density_per_100ha"] is not None
        assert "no excess zeros" in r["error"]

    def test_poisson_limit_flags_the_dispersion_boundary(self):
        # Zero-inflated *Poisson* counts: the NB dispersion runs to infinity
        rng = np.random.default_rng(5)
        areas = rng.uniform(1.0, 3.0, size=50)
        counts = rng.poisson(np.exp(0.3 + 0.5 * areas))
        counts = np.where(rng.random(50) < 0.4, 0, counts)

        r = estimate_zinb(counts.tolist(), areas.tolist())
        assert r["dispersion_at_boundary"] is True
        assert r["density_per_100ha"] is not None
        assert "Poisson limit" in r["error"]
        # A boundary parameter is held fixed, so the SE stays computable —
        # a Hessian taken across the bound would be meaningless.
        assert r["se"] is not None and r["se"] > 0


class TestZinbAgainstGlmmTMB:
    """Golden values from the R analysis this module ports.

    Both cells are from Praschl et al. 2026 (``data/Flug1XDaim``, An-/Rückflug
    transects excluded); the references are the ``dens_zinb``/``se_zinb`` that
    ``glmmTMB(count ~ ha, ziformula = ~1, family = nbinom2)`` produced.
    """

    # lembach x Daim2 — the cell whose likelihood is multimodal: a single-start
    # optimiser settles in the degenerate p_zero -> 0 mode and lands on 56.41.
    LEMBACH_COUNTS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 5, 5]
    LEMBACH_AREAS = [
        1.475133, 1.521986, 1.575163, 1.577327, 1.593400, 1.679427, 1.696479,
        1.698978, 1.763420, 1.827896, 1.844881, 1.845307, 1.848589, 1.861824,
        1.887924, 1.923417, 1.992447, 2.017391, 2.038991, 2.071418, 2.129862,
        2.166071, 2.747946, 1.768878, 1.946804, 2.157280, 2.245080, 2.087712,
        1.820423, 1.907402, 2.200424, 2.181985, 2.221030]

    # ststefan x Daim2 — a well-identified fit (no parameter on a bound)
    STSTEFAN_COUNTS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3,
                       4, 5]
    STSTEFAN_AREAS = [
        1.488923, 1.599765, 1.690057, 1.714709, 1.726957, 1.757419, 1.791569,
        1.808079, 1.890679, 1.948752, 2.021423, 2.021726, 2.059094, 2.065921,
        2.072363, 2.088126, 2.101806, 2.123316, 2.129358, 2.158377, 2.203595,
        2.264995, 2.276287, 2.448357, 2.708561, 2.731780, 1.834966, 1.949574,
        1.996594, 2.132053, 2.241751, 2.628096, 2.875178, 1.778265, 1.988550,
        2.199333, 2.257071, 2.149337, 2.208404, 1.845604]

    def test_multimodal_cell_reaches_the_global_optimum(self):
        r = estimate_zinb(self.LEMBACH_COUNTS, self.LEMBACH_AREAS)
        # glmmTMB: dens_zinb = 57.313, se_zinb = 29.847
        assert r["density_per_100ha"] == pytest.approx(57.313, rel=1e-3)
        assert r["se"] == pytest.approx(29.847, rel=1e-2)
        # The degenerate local optimum a single start finds sits at 56.41 —
        # guard the multistart that gets us off it.
        assert r["density_per_100ha"] > 57.0
        assert r["error"] is None

    def test_identified_cell_matches_density_and_se(self):
        r = estimate_zinb(self.STSTEFAN_COUNTS, self.STSTEFAN_AREAS)
        # glmmTMB: dens_zinb = 32.672, se_zinb = 2.495
        assert r["density_per_100ha"] == pytest.approx(32.672, rel=1e-3)
        assert r["se"] == pytest.approx(2.495, rel=1e-2)
        assert r["ci95"][0] == pytest.approx(27.78, abs=0.05)
        assert r["ci95"][1] == pytest.approx(37.56, abs=0.05)
        assert r["error"] is None

    def test_naive_matches_r_exactly(self):
        # R: sum(count)/sum(ha)*100 = 39.481 (lembach x Daim2)
        r = estimate_naive(self.LEMBACH_COUNTS, self.LEMBACH_AREAS)
        assert r["density_per_100ha"] == pytest.approx(39.481, rel=1e-4)


class TestEstimatePopulation:
    def _transects(self):
        return [
            {"count": 2, "area_ha": 1.0},
            {"count": 0, "area_ha": 1.5},
            {"count": 4, "area_ha": 2.0},
            {"count": 1, "area_ha": 0.5},
            {"count": 0, "area_ha": 1.2},
            {"count": 3, "area_ha": 1.8},
        ]

    def test_runs_the_requested_methods_only(self):
        r = estimate_population(self._transects(), methods=("naive",))
        assert set(r["estimates"]) == {"naive"}
        assert r["n_transects"] == 6
        assert r["total_count"] == 10
        assert r["n_zero_transects"] == 2

    def test_all_three_methods(self):
        r = estimate_population(self._transects(), n_boot=200, seed=3)
        assert set(r["estimates"]) == {"naive", "bootstrap", "zinb"}
        assert r["estimates"]["naive"]["density_per_100ha"] == pytest.approx(
            10 / 8.0 * HA_PER_KM2)

    def test_study_area_extrapolates_to_an_abundance(self):
        r = estimate_population(self._transects(), methods=("naive",),
                                study_area_ha=500.0)
        naive = r["estimates"]["naive"]
        # density is per 100 ha, so 500 ha holds density * 5 animals
        assert naive["abundance_study_area"] == pytest.approx(
            naive["density_per_100ha"] * 5.0)

    def test_without_study_area_no_abundance_is_reported(self):
        r = estimate_population(self._transects(), methods=("naive",))
        assert "abundance_study_area" not in r["estimates"]["naive"]

    def test_a_failing_method_does_not_sink_the_others(self):
        # 2 transects: ZINB cannot fit, naive still can
        few = [{"count": 1, "area_ha": 1.0}, {"count": 0, "area_ha": 2.0}]
        r = estimate_population(few)
        assert r["estimates"]["naive"]["density_per_100ha"] is not None
        assert r["estimates"]["zinb"]["density_per_100ha"] is None
        assert r["estimates"]["zinb"]["error"]
