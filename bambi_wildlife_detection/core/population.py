# -*- coding: utf-8 -*-
"""Transect-based population estimation (naive / bootstrap / ZINB).

Port of the R analysis used in Praschl et al. 2026 (``scripts/run_analysis.R``,
``Praschl_Auswertung.qmd``): every transect contributes one row with a *count*
(animals assigned to it) and an *area* in hectares (the monitored ground area),
and three estimators turn that table into a density:

``naive``
    ``sum(count) / sum(ha) * 100`` — animals per 100 ha (= per km²).
``bootstrap``
    Transects are resampled with replacement (``n_boot`` times); the naive
    density is recomputed on every resample and the mean, SE and percentile
    95 % CI of that distribution are reported.
``zinb``
    Zero-inflated negative binomial regression ``count ~ ha`` with a constant
    zero-inflation term, NB2 parameterisation (``var = mu + mu²/theta``) —
    the ``glmmTMB(count ~ ha, ziformula = ~1, family = nbinom2)`` model of the
    R script, fitted here by maximum likelihood with scipy.  The density is
    ``sum(fitted) / sum(ha) * 100`` with the fitted values on the response
    scale (``(1 - p_zero) * mu``, like ``predict(type = "response")``) and the
    SE from the delta method, combined as ``sqrt(sum(se²)) / sum(ha) * 100``.

The two inputs the plugin has to build first are:

* **counts** — tracks are assigned to the transect whose centre line (the
  flight path between its start and end frame) is nearest in *perpendicular
  distance*, optionally truncated at a maximum distance.
* **areas** — the union of the per-frame field-of-view footprints of the
  frames inside the transect's frame range.

Like every ``core`` module this file must stay importable without QGIS;
shapely and scipy are imported lazily inside the functions that need them.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

#: Densities are reported per 100 ha, which is exactly 1 km² — the unit the R
#: script predicts on ("posteriores predicten auf 1km2").
HA_PER_KM2 = 100.0


# ---------------------------------------------------------------------------
# Geometry: transect centre lines and perpendicular assignment
# ---------------------------------------------------------------------------

def transect_centerline(images: List[dict], first_frame: int, last_frame: int,
                        x_offset: float = 0.0,
                        y_offset: float = 0.0) -> List[Tuple[float, float]]:
    """Flight path of one transect as a world-CRS polyline.

    The poses file stores camera positions mesh-locally (world CRS minus the
    DEM origin), so the origin offset is added back here — the tracks'
    perpendicular positions live in the world CRS.
    """
    line: List[Tuple[float, float]] = []
    lo = max(0, min(first_frame, last_frame))
    hi = min(len(images) - 1, max(first_frame, last_frame))
    for idx in range(lo, hi + 1):
        loc = images[idx].get("location")
        try:
            line.append((float(loc[0]) + x_offset, float(loc[1]) + y_offset))
        except (TypeError, ValueError, IndexError):
            continue
    return line


def point_to_polyline_distance(px: float, py: float,
                               line: Sequence[Tuple[float, float]],
                               ) -> Tuple[float, Optional[Tuple[float, float]]]:
    """Shortest distance from a point to a polyline, and the foot point.

    The projection onto each segment is clamped to the segment, so the foot
    never leaves the transect — a track beyond the transect's end measures to
    its end point rather than to the infinite line through it.
    """
    if not line:
        return float("inf"), None
    if len(line) == 1:
        ax, ay = line[0]
        return math.hypot(px - ax, py - ay), (ax, ay)

    best_dist = float("inf")
    best_foot: Optional[Tuple[float, float]] = None
    for i in range(len(line) - 1):
        ax, ay = line[i]
        bx, by = line[i + 1]
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            fx, fy = ax, ay
        else:
            t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            fx, fy = ax + t * dx, ay + t * dy
        dist = math.hypot(px - fx, py - fy)
        if dist < best_dist:
            best_dist, best_foot = dist, (fx, fy)
    return best_dist, best_foot


def assign_tracks(tracks: List[dict],
                  centerlines: Dict[int, List[Tuple[float, float]]],
                  truncation: float = 0.0) -> List[dict]:
    """Assign every track to the transect nearest in perpendicular distance.

    :param tracks: entries of ``perpendicular_tracks_{m}.json`` — each needs a
        ``detection_center`` ``[x, y, z]`` (world CRS) and ``last_frame``.
    :param centerlines: ``transect_id -> polyline`` (world CRS)
    :param truncation: maximum perpendicular distance in metres; tracks farther
        from every transect stay unassigned (``transect_id`` is ``None``).
        ``0`` (or negative) keeps every track.
    :return: one dict per track with the assignment, its distance and the
        distances to all transects (``distances``), so callers can report the
        runner-up / diagnose ambiguous cases.
    """
    assignments: List[dict] = []
    for track in tracks:
        center = track.get("detection_center") or []
        try:
            px, py = float(center[0]), float(center[1])
        except (TypeError, ValueError, IndexError):
            continue

        distances: Dict[int, float] = {}
        for transect_id, line in centerlines.items():
            dist, _foot = point_to_polyline_distance(px, py, line)
            if math.isfinite(dist):
                distances[transect_id] = dist

        best_id: Optional[int] = None
        best_dist = float("inf")
        for transect_id, dist in distances.items():
            if dist < best_dist:
                best_id, best_dist = transect_id, dist

        truncated = bool(
            truncation and truncation > 0 and best_dist > truncation)
        if best_id is None or truncated:
            assignments.append({
                "track_id": track.get("track_id"),
                "last_frame": track.get("last_frame"),
                "x": px, "y": py,
                "class_id": track.get("class_id"),
                "transect_id": None,
                "distance_m": None if best_id is None else best_dist,
                "truncated": truncated,
                "distances": distances,
            })
            continue

        assignments.append({
            "track_id": track.get("track_id"),
            "last_frame": track.get("last_frame"),
            "x": px, "y": py,
            "class_id": track.get("class_id"),
            "transect_id": best_id,
            "distance_m": best_dist,
            "truncated": False,
            "distances": distances,
        })
    return assignments


# ---------------------------------------------------------------------------
# Monitored area: union of the per-frame FoV footprints
# ---------------------------------------------------------------------------

def merged_fov_area(fov_polygons: Dict[int, list], frames: Sequence[int]):
    """Union the FoV footprints of *frames*; return ``(area_m2, geometry)``.

    *fov_polygons* maps a frame index to its ground footprint as a list of
    ``(x, y, z)`` world coordinates (``fov_{m}/fov_polygons.txt``, read by
    :func:`core.pipeline_outputs.load_fov_polygons_3d`).  Frames without a
    footprint — e.g. because the FoV step sampled every N-th frame — are
    skipped; the geometry is a shapely (Multi)Polygon or ``None`` when the
    transect has no usable footprint at all.

    Self-intersecting footprints are repaired with a zero-width buffer, which
    is what QGIS' ``makeValid`` does for the merged-FoV layer.
    """
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError as exc:
        raise RuntimeError(
            "shapely is required to merge the field-of-view footprints into a "
            f"transect area, but could not be imported: {exc}"
        ) from exc

    geoms = []
    for frame in frames:
        points = fov_polygons.get(frame)
        if not points or len(points) < 3:
            continue
        poly = Polygon([(float(p[0]), float(p[1])) for p in points])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_valid and not poly.is_empty:
            geoms.append(poly)

    if not geoms:
        return 0.0, None

    merged = unary_union(geoms)
    if merged.is_empty:
        return 0.0, None
    return float(merged.area), merged


def geometry_to_rings(geometry) -> List[List[List[float]]]:
    """Exterior rings of a shapely (Multi)Polygon as GeoJSON-style coordinates.

    Holes are dropped: the FoV union of a transect can contain gaps where the
    footprint sampling was too coarse, and those must not be counted as
    monitored area — but they are already excluded from ``merged_fov_area``'s
    area, so the rings are for display only.
    """
    if geometry is None or geometry.is_empty:
        return []
    polys = getattr(geometry, "geoms", None) or [geometry]
    rings = []
    for poly in polys:
        exterior = getattr(poly, "exterior", None)
        if exterior is None:
            continue
        rings.append([[float(x), float(y)] for x, y in exterior.coords])
    return rings


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def estimate_naive(counts: Sequence[float],
                   areas_ha: Sequence[float]) -> Dict[str, object]:
    """Naive density: ``sum(count) / sum(ha) * 100`` (animals per 100 ha)."""
    total_count = float(sum(counts))
    total_ha = float(sum(areas_ha))
    if total_ha <= 0:
        return {"method": "naive", "density_per_100ha": None,
                "error": "total transect area is zero"}
    return {
        "method": "naive",
        "density_per_100ha": total_count / total_ha * HA_PER_KM2,
        "total_count": total_count,
        "total_ha": total_ha,
        "error": None,
    }


def estimate_bootstrap(counts: Sequence[float], areas_ha: Sequence[float],
                       n_boot: int = 999,
                       seed: int = 42) -> Dict[str, object]:
    """Bootstrap the naive density by resampling transects with replacement."""
    import numpy as np

    y = np.asarray(counts, dtype=np.float64)
    ha = np.asarray(areas_ha, dtype=np.float64)
    if y.size < 2 or float(ha.sum()) <= 0:
        return {"method": "bootstrap", "density_per_100ha": None,
                "error": "need at least 2 transects with a positive area"}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, y.size, size=(int(n_boot), y.size))
    boot_counts = y[idx].sum(axis=1)
    boot_ha = ha[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot_dens = np.where(
            boot_ha > 0, boot_counts / boot_ha * HA_PER_KM2, np.nan)
    valid = boot_dens[np.isfinite(boot_dens)]
    if valid.size == 0:
        return {"method": "bootstrap", "density_per_100ha": None,
                "error": "all bootstrap resamples had zero area"}

    return {
        "method": "bootstrap",
        "density_per_100ha": float(valid.mean()),
        "se": float(valid.std(ddof=1)) if valid.size > 1 else 0.0,
        "ci95": [float(np.percentile(valid, 2.5)),
                 float(np.percentile(valid, 97.5))],
        "n_boot": int(n_boot),
        "n_valid": int(valid.size),
        "seed": int(seed),
        "error": None,
    }


#: Bounds of the two parameters whose MLE can run off to infinity. They are
#: not tuning knobs but the edge of the model: at ``log(theta) = 20`` the
#: negative binomial is numerically a Poisson (no overdispersion left to
#: estimate), and at ``|gamma| = 20`` the zero-inflation probability is 0 or 1
#: to within 2e-9. The likelihood is flat beyond them, so a fit that lands on
#: one means that parameter is *unidentified* — see :func:`_boundary_params`.
LOG_THETA_BOUND = 20.0
GAMMA_BOUND = 20.0


def _zinb_nll(params, y, x):
    """Negative log-likelihood of the ZINB model (NB2, constant inflation)."""
    import numpy as np
    from scipy.special import gammaln

    b0, b1, log_theta, gamma = params
    # Keep exp() in range: a diverging linear predictor otherwise overflows
    # before the optimiser can back off.
    eta = np.clip(b0 + b1 * x, -30.0, 30.0)
    mu = np.exp(eta)
    theta = np.exp(np.clip(log_theta, -LOG_THETA_BOUND, LOG_THETA_BOUND))
    gamma = float(np.clip(gamma, -GAMMA_BOUND, GAMMA_BOUND))

    # log p(zero inflation) and log(1 - p), from the logit intercept
    log_p = -np.logaddexp(0.0, -gamma)        # log(sigmoid(gamma))
    log_1mp = -np.logaddexp(0.0, gamma)       # log(1 - sigmoid(gamma))

    # NB2 log pmf
    log_nb = (gammaln(y + theta) - gammaln(theta) - gammaln(y + 1.0)
              + theta * (np.log(theta) - np.log(theta + mu))
              + y * (np.log(mu) - np.log(theta + mu)))

    is_zero = y <= 0
    ll = np.empty_like(mu)
    # zeros: structural zero OR a sampled zero from the NB
    ll[is_zero] = np.logaddexp(log_p, log_1mp + log_nb[is_zero])
    ll[~is_zero] = log_1mp + log_nb[~is_zero]

    total = float(np.sum(ll))
    return np.inf if not np.isfinite(total) else -total


def _numeric_hessian(fn, params, eps: float = 1e-5):
    """Central-difference Hessian of *fn* at *params*."""
    import numpy as np

    p = np.asarray(params, dtype=np.float64)
    n = p.size
    hess = np.zeros((n, n), dtype=np.float64)
    steps = eps * np.maximum(np.abs(p), 1.0)
    for i in range(n):
        for j in range(i, n):
            pi, pj = np.zeros(n), np.zeros(n)
            pi[i] = steps[i]
            pj[j] = steps[j]
            f_pp = fn(p + pi + pj)
            f_pm = fn(p + pi - pj)
            f_mp = fn(p - pi + pj)
            f_mm = fn(p - pi - pj)
            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j])
            hess[i, j] = hess[j, i] = value
    return hess


def _zinb_starts(y, x) -> List["object"]:
    """Deterministic grid of starting points for the ZINB fit.

    The zero-inflation logit is what separates the likelihood's modes: a start
    that assumes no inflation tends to slide into the degenerate
    ``p_zero -> 0`` optimum even when a better fit with genuine inflation
    exists.  The grid therefore spans "no inflation" through "most of the
    observed zeros are structural", crossed with a flat and a data-driven
    slope for ``ha``.
    """
    import numpy as np

    mean_count = float(y.mean())
    positive = y[y > 0]
    conditional_mean = float(positive.mean()) if positive.size else mean_count
    zero_frac = float(np.mean(y <= 0))

    # Rough slope of the log counts on the area, as a non-zero guess for b1.
    slope = 0.0
    if float(np.ptp(x)) > 0:
        slope = float(np.polyfit(x, np.log1p(y), 1)[0])

    b0_options = [
        math.log(max(mean_count, 0.1)),
        math.log(max(conditional_mean, 0.1)),
    ]
    b1_options = [0.0, slope] if abs(slope) > 1e-9 else [0.0]
    log_theta_options = [0.0, 1.5]

    gamma_options = [-4.0]                       # essentially no inflation
    for share in (0.5, 0.9):                     # part / most zeros structural
        p = min(max(share * zero_frac, 1e-3), 0.95)
        gamma_options.append(math.log(p / (1.0 - p)))

    starts = []
    for b0 in b0_options:
        for b1 in b1_options:
            for log_theta in log_theta_options:
                for gamma in gamma_options:
                    starts.append(np.array([b0, b1, log_theta, gamma]))
    return starts


def estimate_zinb(counts: Sequence[float],
                  areas_ha: Sequence[float]) -> Dict[str, object]:
    """Zero-inflated negative binomial regression of ``count ~ ha``.

    Mirrors ``glmmTMB(count ~ ha, ziformula = ~1, family = nbinom2)`` followed
    by ``augment(type.predict = "response")``: the fitted value of a transect
    is ``(1 - p_zero) * exp(b0 + b1 * ha)`` and the reported density is
    ``sum(fitted) / sum(ha) * 100``.  Standard errors come from the delta
    method on the MLE covariance (the inverse observed information) and are
    combined across transects as ``sqrt(sum(se²))``, exactly as in the R code.
    """
    import numpy as np

    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover — scipy is a hard dependency
        return {"method": "zinb", "density_per_100ha": None,
                "error": f"scipy not available: {exc}"}

    y = np.asarray(counts, dtype=np.float64)
    x = np.asarray(areas_ha, dtype=np.float64)
    total_ha = float(x.sum())
    # 4 parameters (b0, b1, log_theta, zi intercept) need at least as many rows.
    if y.size < 4 or total_ha <= 0:
        return {"method": "zinb", "density_per_100ha": None,
                "error": "need at least 4 transects with a positive area"}
    if np.allclose(x, x[0]):
        return {"method": "zinb", "density_per_100ha": None,
                "error": "all transects have the same area — 'ha' has no variance"}

    def nll(params):
        return _zinb_nll(params, y, x)

    # The ZINB likelihood is multimodal: a single start regularly settles in
    # the local optimum where the zero-inflation collapses (p_zero -> 0) while
    # a better fit with a real zero-inflation component exists. Every start of
    # the grid is optimised and the best likelihood wins — this is what makes
    # the estimates reproduce glmmTMB's.
    best = None
    for start in _zinb_starts(y, x):
        result = minimize(nll, start, method="Nelder-Mead",
                          options={"maxiter": 10000, "xatol": 1e-9,
                                   "fatol": 1e-9})
        polished = minimize(nll, result.x, method="BFGS",
                            options={"maxiter": 5000})
        for candidate in (result, polished):
            if not np.all(np.isfinite(candidate.x)) or not np.isfinite(candidate.fun):
                continue
            if best is None or candidate.fun < best.fun:
                best = candidate

    if best is None:
        return {"method": "zinb", "density_per_100ha": None,
                "error": "model did not converge"}

    params = np.clip(
        best.x,
        [-np.inf, -np.inf, -LOG_THETA_BOUND, -GAMMA_BOUND],
        [np.inf, np.inf, LOG_THETA_BOUND, GAMMA_BOUND],
    )
    b0, b1, log_theta, gamma = (float(v) for v in params)
    eta = np.clip(b0 + b1 * x, -30.0, 30.0)
    mu = np.exp(eta)
    p_zero = float(1.0 / (1.0 + np.exp(-gamma)))
    fitted = (1.0 - p_zero) * mu          # predict(type = "response")
    density = float(fitted.sum()) / total_ha * HA_PER_KM2

    # ---- Delta-method SE of the fitted values --------------------------
    # d fitted / d(b0, b1, log_theta, gamma):
    #   b0    -> fitted
    #   b1    -> fitted * ha
    #   theta -> 0 (the dispersion does not shift the mean)
    #   gamma -> -fitted * p_zero
    grad = np.column_stack([
        fitted,
        fitted * x,
        np.zeros_like(fitted),
        -fitted * p_zero,
    ])
    fixed, warnings = _boundary_params(log_theta, gamma)
    se_density, ci, se_error = _delta_method_se(
        nll, params, grad, total_ha, density, fixed=fixed)
    if se_error:
        warnings.append(se_error)

    return {
        "method": "zinb",
        "density_per_100ha": density,
        "se": se_density,
        "ci95": ci,
        "params": {
            "intercept": b0,
            "ha": b1,
            "theta": float(np.exp(log_theta)),
            "zero_inflation_prob": p_zero,
        },
        "log_likelihood": float(-best.fun),
        "aic": float(2 * 4 + 2 * best.fun),
        "converged": bool(best.success),
        "dispersion_at_boundary": 2 in fixed,
        "zero_inflation_at_boundary": 3 in fixed,
        "error": "; ".join(warnings) if warnings else None,
    }


def _boundary_params(log_theta: float, gamma: float):
    """Parameter indices sitting on a bound, plus the warnings to report.

    Two of the four ZINB parameters have an MLE that can legitimately run to
    infinity, and the likelihood is flat once it gets there:

    * ``log(theta) -> +inf`` — the counts carry no overdispersion beyond what
      the zero-inflation already explains, so the conditional model is a
      Poisson and the NB dispersion is unidentified.
    * ``|gamma| -> inf`` — there are no excess zeros (``p_zero -> 0``, the fit
      reducing to a plain negative binomial) or nothing but zeros.

    A parameter that has run into its bound carries no curvature, so including
    it in the observed information yields a singular (or step-size dependent,
    hence meaningless) matrix. It is therefore held fixed for the delta method
    — sound because the fitted values do not depend on it there: the mean is
    free of ``theta`` altogether, and ``d fitted / d gamma = -fitted * p_zero``
    vanishes as ``p_zero -> 0``.

    R/glmmTMB stops its optimiser at some large-but-finite value instead and
    propagates that point's curvature, so its CI for such a fit is an artefact
    of where it happened to stop. The density itself is unaffected — that is
    why it still reproduces to within 0.02 % on the reference data.
    """
    fixed = []
    warnings = []
    if abs(log_theta) >= LOG_THETA_BOUND - 1e-6:
        fixed.append(2)
        warnings.append(
            "the negative-binomial dispersion is unidentified (the counts "
            "reach the Poisson limit) — it was held fixed, so the confidence "
            "interval understates the true uncertainty")
    if abs(gamma) >= GAMMA_BOUND - 1e-6:
        fixed.append(3)
        warnings.append(
            "zero-inflation collapsed to the boundary — the counts show no "
            "excess zeros, so the fit reduces to a plain negative binomial")
    return fixed, warnings


def _delta_method_se(nll, params, grad, total_ha: float, density: float,
                     fixed: Optional[Sequence[int]] = None):
    """SE and 95 % CI of the density from the MLE covariance (delta method).

    Returns ``(se, ci95, error)``. Parameters listed in *fixed* are held at
    their value (their row/column is dropped from the observed information);
    :func:`_boundary_params` supplies the ones sitting on a bound, where the
    likelihood is flat and a numeric Hessian would be meaningless. Dropping a
    parameter is only safe when the fitted values do not depend on it — i.e.
    its column in *grad* is zero — which is asserted here rather than assumed.

    R combines the per-transect SEs as ``sqrt(sum(se_fit²)) / sum(ha) * 100``,
    which is reproduced here.
    """
    import numpy as np

    n_params = len(params)
    fixed = set(fixed or ())
    free = [i for i in range(n_params) if i not in fixed]
    for i in fixed:
        if np.any(np.abs(grad[:, i]) > 1e-8):
            return None, None, ("a boundary parameter still moves the fitted "
                                "values — no standard errors")
    if not free:
        return None, None, "no free parameters left — no standard errors"

    # Profile out the fixed parameters: differentiate only in the free ones.
    free_params = np.asarray(params, dtype=np.float64)[free]

    def profile_nll(sub):
        full = np.asarray(params, dtype=np.float64).copy()
        full[free] = sub
        return nll(full)

    hess = _numeric_hessian(profile_nll, free_params)
    if not np.all(np.isfinite(hess)):
        return None, None, "information matrix is not finite — no standard errors"

    singular_msg = "information matrix is singular — no standard errors"
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        return None, None, singular_msg
    if not np.all(np.isfinite(cov)):
        return None, None, singular_msg

    sub_grad = grad[:, free]
    var_fitted = np.einsum("ij,jk,ik->i", sub_grad, cov, sub_grad)
    if not np.all(np.isfinite(var_fitted)) or np.any(var_fitted < 0):
        return None, None, "standard errors are not finite (singular information)"

    se = float(math.sqrt(float(np.sum(var_fitted))) / total_ha * HA_PER_KM2)
    return se, [density - 1.96 * se, density + 1.96 * se], None


def estimate_population(transects: List[dict], methods: Sequence[str] = (
        "naive", "bootstrap", "zinb"), n_boot: int = 999, seed: int = 42,
        study_area_ha: float = 0.0) -> Dict[str, object]:
    """Run the requested estimators over the per-transect count/area table.

    :param transects: one dict per transect with ``count`` and ``area_ha``
    :param study_area_ha: when > 0, every density is extrapolated to an
        abundance for a study area of this size (``density / 100 * ha``).
    """
    counts = [float(t.get("count", 0)) for t in transects]
    areas = [float(t.get("area_ha", 0.0)) for t in transects]

    estimates: Dict[str, object] = {}
    if "naive" in methods:
        estimates["naive"] = estimate_naive(counts, areas)
    if "bootstrap" in methods:
        estimates["bootstrap"] = estimate_bootstrap(counts, areas, n_boot, seed)
    if "zinb" in methods:
        estimates["zinb"] = estimate_zinb(counts, areas)

    if study_area_ha and study_area_ha > 0:
        for est in estimates.values():
            density = est.get("density_per_100ha")
            if density is None:
                continue
            scale = study_area_ha / HA_PER_KM2
            est["abundance_study_area"] = density * scale
            ci = est.get("ci95")
            if ci:
                est["abundance_ci95"] = [ci[0] * scale, ci[1] * scale]

    return {
        "n_transects": len(transects),
        "total_count": float(sum(counts)),
        "total_ha": float(sum(areas)),
        "n_zero_transects": int(sum(1 for c in counts if c == 0)),
        "study_area_ha": float(study_area_ha or 0.0),
        "estimates": estimates,
    }
