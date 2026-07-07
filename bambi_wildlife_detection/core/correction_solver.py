# -*- coding: utf-8 -*-
"""Analytic tz/rz correction solver of the Correction Wizard.

Moved from ``bambi_correction_wizard._ProbeWorker`` (which now delegates
here). The wizard's two reference clicks are geo-referenced through the DEM;
perfect overlap means both geo-referenced points coincide at a single world
point q. q must then be an intersection point of the two circles (centre =
camera XY, radius = horizontal camera-to-point distance), and the yaw
correction that moves point i onto q is the angle from c_i→p_i to c_i→q.
Because one global yaw applies to both cameras, the correct tz is the one
whose circle intersection yields the SAME yaw delta on both sides.

For flat terrain the radius responds linearly to camera height:
``r_i(Δtz) = r_i + (r_i / h_i)·Δtz`` with h_i the camera height above the
ground point (exact for a horizontal ground plane). This turns the problem
into 1-D root finding in Δtz that needs no ray-casting; the roots seed a
Gauss-Newton refinement that uses true DEM ray-casts to absorb terrain
relief.

The DEM/camera specifics are injected as callables so the solver is testable
with synthetic geometry:

* ``geo_ref(side_idx, correction) -> (x, y, z) | None`` — ray-cast the
  side's reference click onto the DEM under *correction*
* ``camera_xy(side_idx, correction) -> (x, y) | None``
* ``camera_z(side_idx, correction) -> float | None``
"""

import math

import numpy as np


def wrap_deg(v: float) -> float:
    """Wrap *v* (degrees) to (−180, +180]."""
    return ((v + 180.0) % 360.0) - 180.0


def wrap_rad(v: float) -> float:
    """Wrap *v* (radians) to (−π, +π]."""
    return ((v + math.pi) % (2.0 * math.pi)) - math.pi


def circles_intersect(c1, r1: float, c2, r2: float) -> bool:
    """Return True when the two circles share at least one point."""
    d = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
    return abs(r1 - r2) <= d <= r1 + r2


class CorrectionSolver:
    """Solves the z-offset + yaw correction from two reference clicks."""

    _GN_TOL_M = 0.02      # stop refining when points overlap within 2 cm
    _GN_MAX_ITER = 12
    _GN_EPS_TZ = 0.25     # finite-difference step for tz (m)
    _GN_EPS_RZ = 0.002    # finite-difference step for rz (rad)

    def __init__(self, geo_ref, camera_xy, camera_z,
                 max_steps: int = 200, status_fn=None):
        self._geo_ref = geo_ref
        self._camera_xy = camera_xy
        self._camera_z = camera_z
        self._max_steps = max_steps
        self._status_fn = status_fn

    def _status(self, message: str) -> None:
        if self._status_fn:
            self._status_fn(message)

    # ------------------------------------------------------------------

    def compute_circles(self, correction: dict):
        c1 = self._camera_xy(0, correction)
        c2 = self._camera_xy(1, correction)
        p1 = self._geo_ref(0, correction)
        p2 = self._geo_ref(1, correction)
        if any(v is None for v in (c1, c2, p1, p2)):
            return None
        r1 = math.sqrt((p1[0] - c1[0]) ** 2 + (p1[1] - c1[1]) ** 2)
        r2 = math.sqrt((p2[0] - c2[0]) ** 2 + (p2[1] - c2[1]) ** 2)
        return (c1, r1, (p1[0], p1[1])), (c2, r2, (p2[0], p2[1]))

    def _with_tz_rz(self, correction: dict, tz: float, rz: float) -> dict:
        return {
            'translation': {**correction['translation'], 'z': tz},
            'rotation': {**correction['rotation'], 'z': rz},
        }

    def _pair_error(self, correction: dict):
        """Return (horizontal |p1−p2|, p1, p2) or (None, None, None)."""
        p1 = self._geo_ref(0, correction)
        p2 = self._geo_ref(1, correction)
        if p1 is None or p2 is None:
            return None, None, None
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1]), p1, p2

    @staticmethod
    def _branch_g(geom: dict, branch: float, delta: float):
        """Yaw mismatch g(Δ) = δ1 − δ2 at the circle-intersection point of
        the given *branch* (+1 / −1), or None when the circles do not
        intersect for this Δ.  Returns (g, δ1)."""
        r1 = geom['r1'] + geom['k1'] * delta
        r2 = geom['r2'] + geom['k2'] * delta
        d = geom['d']
        if r1 <= 0.0 or r2 <= 0.0:
            return None
        if not (abs(r1 - r2) <= d <= r1 + r2):
            return None
        a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
        hh = r1 * r1 - a * a
        h = math.sqrt(hh) if hh > 0.0 else 0.0
        c1, c2 = geom['c1'], geom['c2']
        ex, ey = geom['ex'], geom['ey']
        qx = c1[0] + a * ex[0] + branch * h * ey[0]
        qy = c1[1] + a * ex[1] + branch * h * ey[1]
        d1 = wrap_rad(math.atan2(qy - c1[1], qx - c1[0]) - geom['a1'])
        d2 = wrap_rad(math.atan2(qy - c2[1], qx - c2[0]) - geom['a2'])
        return wrap_rad(d1 - d2), d1

    def _branch_roots(self, geom: dict, branch: float) -> list:
        """Find Δtz values where both sides need the same yaw delta.

        Scans a generous Δ range (the feasible sub-range where the circles
        intersect is detected on the fly), then bisects each sign change of
        g(Δ).  Returns a list of ``(Δtz, yaw_delta)`` tuples.
        """
        # r/k = h, so the scan spans up to ±2× the flight height.
        span = 2.0 * max(geom['r1'] / geom['k1'], geom['r2'] / geom['k2'])
        span = min(max(span, 20.0), 500.0)
        n = 400
        roots = []
        prev = None   # (delta, g)
        for i in range(n + 1):
            delta = -span + (2.0 * span) * i / n
            res = self._branch_g(geom, branch, delta)
            if res is None:
                prev = None
                continue
            g, d1 = res
            if abs(g) < 1e-9:
                roots.append((delta, d1))
                prev = None
                continue
            if prev is not None and prev[1] * g < 0 and abs(g - prev[1]) < math.pi:
                lo, glo = prev
                hi = delta
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    rm = self._branch_g(geom, branch, mid)
                    if rm is None:
                        break
                    if glo * rm[0] <= 0:
                        hi = mid
                    else:
                        lo, glo = mid, rm[0]
                rm = self._branch_g(geom, branch, 0.5 * (lo + hi))
                if rm is not None:
                    roots.append((0.5 * (lo + hi), rm[1]))
            prev = (delta, g)
        return roots

    def solve_tz_rz(self, corr: dict):
        """Solve for (tz, rz) giving perfect point overlap.

        Returns ``(tz, rz)`` or None when the geometry cannot be set up
        (ray misses, degenerate camera placement, …).
        """
        tz0 = corr['translation']['z']
        rz0 = corr['rotation']['z']

        c1 = self._camera_xy(0, corr)
        c2 = self._camera_xy(1, corr)
        z1 = self._camera_z(0, corr)
        z2 = self._camera_z(1, corr)
        p1 = self._geo_ref(0, corr)
        p2 = self._geo_ref(1, corr)
        if any(v is None for v in (c1, c2, z1, z2, p1, p2)):
            return None

        r1 = math.hypot(p1[0] - c1[0], p1[1] - c1[1])
        r2 = math.hypot(p2[0] - c2[0], p2[1] - c2[1])
        h1 = z1 - p1[2]
        h2 = z2 - p2[2]
        if min(r1, r2) < 1e-6 or min(h1, h2) < 1e-3:
            return None

        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return None   # coincident cameras — yaw is unconstrained

        geom = {
            'c1': c1, 'c2': c2, 'd': d,
            'ex': (dx / d, dy / d), 'ey': (-dy / d, dx / d),
            'r1': r1, 'r2': r2,
            'k1': r1 / h1, 'k2': r2 / h2,
            'a1': math.atan2(p1[1] - c1[1], p1[0] - c1[0]),
            'a2': math.atan2(p2[1] - c2[1], p2[0] - c2[0]),
        }

        # Candidate seeds from both intersection branches; the yaw delta is
        # also tried sign-flipped so the seed survives either camera-yaw
        # convention.  Each candidate costs two ray-casts to evaluate.
        seeds = [(0.0, 0.0)]
        for branch in (1.0, -1.0):
            for droot, yaw in self._branch_roots(geom, branch):
                seeds.append((droot, yaw))
                seeds.append((droot, -yaw))

        best_err = None
        best = None
        for i, (dtz, drz) in enumerate(seeds):
            self._status(
                f"Evaluating candidate {i + 1} / {len(seeds)}…"
            )
            cand_tz = tz0 + dtz
            cand_rz = wrap_rad(rz0 + drz)
            err, _, _ = self._pair_error(
                self._with_tz_rz(corr, cand_tz, cand_rz))
            if err is not None and (best_err is None or err < best_err):
                best_err = err
                best = (cand_tz, cand_rz)
        if best is None:
            return None

        # Gauss-Newton refinement on F(tz, rz) = p1 − p2 with true DEM
        # ray-casting and a numeric Jacobian.
        tz, rz = best
        for it in range(self._GN_MAX_ITER):
            err, bp1, bp2 = self._pair_error(self._with_tz_rz(corr, tz, rz))
            if err is None:
                break
            self._status(
                f"Refining…  iteration {it + 1} / {self._GN_MAX_ITER},  "
                f"point distance = {err:.3f} m"
            )
            if err < self._GN_TOL_M:
                break
            fx = bp1[0] - bp2[0]
            fy = bp1[1] - bp2[1]

            et, tp1, tp2 = self._pair_error(
                self._with_tz_rz(corr, tz + self._GN_EPS_TZ, rz))
            er, rp1, rp2 = self._pair_error(
                self._with_tz_rz(corr, tz, rz + self._GN_EPS_RZ))
            if et is None or er is None:
                break
            j11 = ((tp1[0] - tp2[0]) - fx) / self._GN_EPS_TZ
            j21 = ((tp1[1] - tp2[1]) - fy) / self._GN_EPS_TZ
            j12 = ((rp1[0] - rp2[0]) - fx) / self._GN_EPS_RZ
            j22 = ((rp1[1] - rp2[1]) - fy) / self._GN_EPS_RZ
            det = j11 * j22 - j12 * j21
            if abs(det) < 1e-12:
                break
            dtz = -(j22 * fx - j12 * fy) / det
            drz = -(-j21 * fx + j11 * fy) / det
            dtz = max(-25.0, min(25.0, dtz))
            drz = max(-0.5, min(0.5, drz))

            # Backtracking line search so a bad full step never diverges
            lam = 1.0
            improved = False
            for _ in range(5):
                ne, _, _ = self._pair_error(
                    self._with_tz_rz(corr, tz + lam * dtz, rz + lam * drz))
                if ne is not None and ne < err:
                    tz += lam * dtz
                    rz += lam * drz
                    improved = True
                    break
                lam *= 0.5
            if not improved:
                break

        return (tz, wrap_rad(rz))

    # ------------------------------------------------------------------

    def probe_z(self, correction: dict) -> float:
        tz = correction['translation']['z']

        def _ok(z: float) -> bool:
            c = {
                'translation': {**correction['translation'], 'z': z},
                'rotation': dict(correction['rotation']),
            }
            circles = self.compute_circles(c)
            if circles is None:
                return False
            (c1, r1, _), (c2, r2, _) = circles
            return circles_intersect(c1, r1, c2, r2)

        if not _ok(tz):
            for step in range(self._max_steps):
                tz += 1.0
                self._status(
                    f"Fallback  —  Probing z-offset…  "
                    f"(step {step + 1} / {self._max_steps},  tz = {tz:.1f})"
                )
                if _ok(tz):
                    break
        else:
            for step in range(self._max_steps):
                tz -= 1.0
                self._status(
                    f"Fallback  —  Finding z boundary…  "
                    f"(step {step + 1} / {self._max_steps},  tz = {tz:.1f})"
                )
                if not _ok(tz):
                    tz += 1.0
                    break
        return tz

    def find_best_rz(self, correction: dict) -> float:
        best_rz = correction['rotation']['z']
        best_dist = float('inf')
        angles = np.linspace(0.0, 2 * math.pi, 360)

        for i, rz in enumerate(angles):
            if i % 36 == 0:
                self._status(
                    f"Fallback  —  Scanning yaw…  ({i} / 360)"
                )
            c = {
                'translation': dict(correction['translation']),
                'rotation': {**correction['rotation'], 'z': float(rz)},
            }
            p1 = self._geo_ref(0, c)
            p2 = self._geo_ref(1, c)
            if p1 is None or p2 is None:
                continue
            d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            if d < best_dist:
                best_dist = d
                best_rz = float(rz)

        return wrap_rad(best_rz)
