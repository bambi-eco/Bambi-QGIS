# -*- coding: utf-8 -*-
"""Unit tests for core.correction_solver (analytic tz/rz solver).

The solver is exercised against an exact synthetic model: a flat ground
plane at z = 0 and two cameras that both observed the same ground feature.
The injected ``geo_ref`` simulates the DEM ray-cast analytically: a height
error stretches the ray's horizontal distance proportionally, a yaw error
rotates it around the camera. At the true correction both geo-referenced
points coincide, so the solver must drive the pair error to ~zero.
"""
import math

import pytest

from bambi_wildlife_detection.core.correction_solver import (
    CorrectionSolver,
    circles_intersect,
    wrap_deg,
    wrap_rad,
)

TZ_TRUE = -2.5
RZ_TRUE = 0.5

CAMS = [(0.0, 0.0), (100.0, 0.0)]     # camera XY positions
Z_CAM = [100.0, 100.0]                # pose altitude (before tz correction)
Q_TRUE = (50.0, 40.0)                 # the ground feature both sides clicked


class FlatGroundModel:
    """Analytic stand-in for the DEM ray-casting callbacks."""

    def __init__(self):
        self.azimuth = []
        self.tan_elev = []
        for i, c in enumerate(CAMS):
            dx, dy = Q_TRUE[0] - c[0], Q_TRUE[1] - c[1]
            r_true = math.hypot(dx, dy)
            self.azimuth.append(math.atan2(dy, dx))
            self.tan_elev.append(r_true / (Z_CAM[i] + TZ_TRUE))

    def geo_ref(self, side, corr):
        tz = corr["translation"]["z"]
        rz = corr["rotation"]["z"]
        h = Z_CAM[side] + tz          # camera height above the ground plane
        if h <= 0:
            return None               # camera below ground: ray misses
        dist = h * self.tan_elev[side]
        az = self.azimuth[side] + (rz - RZ_TRUE)
        c = CAMS[side]
        return (c[0] + dist * math.cos(az), c[1] + dist * math.sin(az), 0.0)

    def camera_xy(self, side, corr):
        t = corr["translation"]
        return (CAMS[side][0] + t.get("x", 0.0), CAMS[side][1] + t.get("y", 0.0))

    def camera_z(self, side, corr):
        return Z_CAM[side] + corr["translation"]["z"]


def _corr(tz, rz):
    return {
        "translation": {"x": 0.0, "y": 0.0, "z": tz},
        "rotation": {"x": 0.0, "y": 0.0, "z": rz},
    }


@pytest.fixture
def model():
    return FlatGroundModel()


@pytest.fixture
def solver(model):
    return CorrectionSolver(model.geo_ref, model.camera_xy, model.camera_z)


def _pair_error(model, tz, rz):
    p1 = model.geo_ref(0, _corr(tz, rz))
    p2 = model.geo_ref(1, _corr(tz, rz))
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class TestWrapHelpers:
    def test_wrap_deg(self):
        assert wrap_deg(190.0) == pytest.approx(-170.0)
        assert wrap_deg(-190.0) == pytest.approx(170.0)
        # boundary maps to -180 (implementation wraps to [-180, +180))
        assert wrap_deg(180.0) == pytest.approx(-180.0)

    def test_wrap_rad(self):
        assert wrap_rad(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
        assert wrap_rad(-3 * math.pi / 2) == pytest.approx(math.pi / 2)

    def test_circles_intersect(self):
        assert circles_intersect((0, 0), 60.0, (100, 0), 60.0) is True
        assert circles_intersect((0, 0), 40.0, (100, 0), 40.0) is False
        assert circles_intersect((0, 0), 10.0, (0, 5), 2.0) is False  # nested


class TestSolveTzRz:
    def test_recovers_overlap_from_perturbed_start(self, solver, model):
        result = solver.solve_tz_rz(_corr(TZ_TRUE + 7.0, RZ_TRUE + 0.15))
        assert result is not None
        tz, rz = result
        # The solution must bring both geo-referenced points into overlap
        # (the solver's own GN tolerance is 2 cm).
        assert _pair_error(model, tz, rz) < 0.05

    def test_keeps_exact_start_solution(self, solver):
        result = solver.solve_tz_rz(_corr(TZ_TRUE, RZ_TRUE))
        assert result == (pytest.approx(TZ_TRUE), pytest.approx(RZ_TRUE))

    def test_recovers_truth_for_small_perturbation(self, solver, model):
        # Close to the truth the equal-yaw-delta root is unique, so the
        # solver must find the true parameters, not just any overlap.
        result = solver.solve_tz_rz(_corr(TZ_TRUE + 3.0, RZ_TRUE - 0.05))
        assert result is not None
        tz, rz = result
        assert _pair_error(model, tz, rz) < 0.05
        assert tz == pytest.approx(TZ_TRUE, abs=0.5)
        assert wrap_rad(rz - RZ_TRUE) == pytest.approx(0.0, abs=0.01)

    def test_coincident_cameras_return_none(self, model):
        def same_xy(side, corr):
            return (0.0, 0.0)

        solver = CorrectionSolver(model.geo_ref, same_xy, model.camera_z)
        assert solver.solve_tz_rz(_corr(0.0, 0.0)) is None

    def test_ray_miss_returns_none(self, model):
        solver = CorrectionSolver(
            lambda side, corr: None, model.camera_xy, model.camera_z)
        assert solver.solve_tz_rz(_corr(0.0, 0.0)) is None

    def test_status_messages_emitted(self, model):
        messages = []
        solver = CorrectionSolver(
            model.geo_ref, model.camera_xy, model.camera_z,
            status_fn=messages.append)
        solver.solve_tz_rz(_corr(TZ_TRUE + 7.0, RZ_TRUE + 0.15))
        assert any("Evaluating candidate" in m for m in messages)


class TestProbeZ:
    def test_steps_up_until_circles_intersect(self, solver):
        # At tz = -60 the view circles are far too small to intersect.
        result = solver.probe_z(_corr(-60.0, RZ_TRUE))
        assert result > -60.0
        circles = solver.compute_circles(_corr(result, RZ_TRUE))
        (c1, r1, _), (c2, r2, _) = circles
        assert circles_intersect(c1, r1, c2, r2)

    def test_steps_down_to_intersection_boundary(self, solver):
        # At the true tz the circles intersect; probing walks down to the
        # boundary and steps back inside it.
        result = solver.probe_z(_corr(TZ_TRUE, RZ_TRUE))
        circles = solver.compute_circles(_corr(result, RZ_TRUE))
        (c1, r1, _), (c2, r2, _) = circles
        assert circles_intersect(c1, r1, c2, r2)
        below = solver.compute_circles(_corr(result - 1.0, RZ_TRUE))
        (c1, r1, _), (c2, r2, _) = below
        assert not circles_intersect(c1, r1, c2, r2)


class TestFindBestRz:
    def test_scan_finds_true_yaw(self, solver):
        best = solver.find_best_rz(_corr(TZ_TRUE, 0.0))
        # 360-step scan over 2π: resolution ~0.0175 rad
        assert wrap_rad(best - RZ_TRUE) == pytest.approx(0.0, abs=0.02)

    def test_all_misses_keep_start_value(self, model):
        solver = CorrectionSolver(
            lambda side, corr: None, model.camera_xy, model.camera_z)
        assert solver.find_best_rz(_corr(0.0, 1.25)) == pytest.approx(
            wrap_rad(1.25))


class TestComputeCircles:
    def test_radii_match_camera_to_point_distance(self, solver):
        circles = solver.compute_circles(_corr(TZ_TRUE, RZ_TRUE))
        (c1, r1, p1), (c2, r2, p2) = circles
        assert c1 == CAMS[0] and c2 == CAMS[1]
        assert p1 == pytest.approx(Q_TRUE)
        assert p2 == pytest.approx(Q_TRUE)
        assert r1 == pytest.approx(math.hypot(50.0, 40.0))

    def test_missing_data_returns_none(self, model):
        solver = CorrectionSolver(
            lambda side, corr: None, model.camera_xy, model.camera_z)
        assert solver.compute_circles(_corr(0.0, 0.0)) is None
