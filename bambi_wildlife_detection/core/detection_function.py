# -*- coding: utf-8 -*-
"""Detection functions whose width depends on where an animal was seen.

Conventional distance sampling fits one detection function to all
observations: the chance of seeing an animal falls off with its distance
from the line, the same everywhere. From a drone it does not. Under closed
canopy an animal a few metres off the line is already hidden; from twice
the height it is half as many pixels. One curve for everything is the
average of all that. It still gives a fair density for a survey as a whole,
but it cannot say what the canopy or the height did - which is what the
next flight is planned by - and a curve that fits each observation's
conditions describes the distances better and usually narrows the
interval.

**The model** (Marques and Buckland 2003). The *scale* of the detection
function - how fast detectability falls off - depends on covariates of the
observation::

    sigma_i = exp(b0 + b1 * z_i1 + b2 * z_i2 + ...)

with the half-normal ``g(x) = exp(-x^2 / (2 sigma^2))`` or the hazard-rate
``g(x) = 1 - exp(-(x / sigma)^-b)`` as the key. The covariates are
standardised before fitting; the coefficients are reported per unit of the
covariate. A positive coefficient means: more of this, and animals are
seen from further away.

**The density.** Each observation stands for more animals the narrower the
strip it could be seen in, so with ``mu_i`` the effective strip half-width
for observation *i* and ``s_i`` its group size::

    D = sum_i  s_i / (2 * L * mu_i)

**Its uncertainty** has two parts: how the line totals of that sum vary
between the lines (the same estimator as for the encounter rate, see
:mod:`core.distance_effort` - it carries the variation of group size with
it), and what not knowing the coefficients exactly adds (delta method,
from the curvature of the likelihood).

**Which model.** Every candidate - each key with every subset of the
covariates - is fitted and the one with the lowest AIC is used; the plain
models without covariates are among them, so a covariate that explains
nothing costs nothing.

numpy and scipy are imported inside the functions that need them. Free of
QGIS and Qt.
"""

import math
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

HALF_NORMAL = "half-normal"
HAZARD_RATE = "hazard-rate"
KEYS = (HALF_NORMAL, HAZARD_RATE)

#: The covariates a survey can offer: ``key -> (what it is called, unit)``.
CANOPY = "canopy_cover_pct"
HEIGHT = "flight_height_m"
COVARIATES = {
    CANOPY: ("canopy cover", "%"),
    HEIGHT: ("flight height", "m"),
}

#: With fewer observations a covariate is not asked to explain anything.
MIN_OBSERVATIONS = 20
#: Points on which a hazard-rate curve is integrated.
GRID = 512


def label(key: str) -> str:
    """What a covariate is called."""
    return COVARIATES.get(key, (key.replace("_", " "), ""))[0]


def unit(key: str) -> str:
    return COVARIATES.get(key, ("", ""))[1]


def model_name(key: str, covariates: Sequence[str] = ()) -> str:
    """``half-normal`` - or ``half-normal + canopy cover + flight height``."""
    return " + ".join([key] + [label(name) for name in covariates])


def subsets(names: Sequence[str]) -> List[Tuple[str, ...]]:
    """Every way of using some of the covariates, the fewest first."""
    found: List[Tuple[str, ...]] = []
    for size in range(1, len(names) + 1):
        found.extend(combinations(names, size))
    return found


def usable(values: Dict[str, Sequence[Optional[float]]], n: int
           ) -> Tuple[Dict[str, list], Dict[str, str]]:
    """Which covariates a model can use.

    A covariate has to be known for every observation - a model cannot be
    fitted to some of them - and has to differ between them.

    :param values: ``name -> value per observation`` (``None`` = not known)
    :param n: the number of observations
    :return: ``(usable, why not)`` - the values of those that can be used,
        and for the others the reason in words
    """
    kept: Dict[str, list] = {}
    refused: Dict[str, str] = {}
    for name, of_name in values.items():
        known = [float(v) for v in of_name
                 if v is not None and math.isfinite(float(v))]
        if len(of_name) != n or len(known) < n:
            refused[name] = (
                f"known for {len(known)} of {n} observations only")
            continue
        if max(known) - min(known) <= 1e-9:
            refused[name] = (
                f"the same for every observation ({known[0]:g} {unit(name)})"
                .rstrip())
            continue
        kept[name] = known
    if kept and n < MIN_OBSERVATIONS:
        for name in list(kept):
            refused[name] = (
                f"{n} observations are too few to tell what it does (at "
                f"least {MIN_OBSERVATIONS})")
        kept = {}
    return kept, refused


# ---------------------------------------------------------------------------
# The curves
# ---------------------------------------------------------------------------

def _half_normal_width(sigma, w: float):
    """Effective strip half-width of a half-normal curve, truncated at w."""
    import numpy as np
    from scipy.special import erf

    sigma = np.asarray(sigma, dtype=np.float64)
    return sigma * math.sqrt(math.pi / 2.0) * erf(w / (sigma * math.sqrt(2.0)))


def _hazard(x, sigma, shape: float):
    """The hazard-rate curve at *x* (broadcast against *sigma*)."""
    import numpy as np

    x, sigma = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                   np.asarray(sigma, dtype=np.float64))
    out = np.ones(x.shape, dtype=np.float64)
    positive = x > 0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        out[positive] = 1.0 - np.exp(
            -np.power(x[positive] / sigma[positive], -shape))
    return out


def _hazard_width(sigma, shape: float, w: float):
    """Effective strip half-width of a hazard-rate curve, per sigma."""
    import numpy as np

    sigma = np.asarray(sigma, dtype=np.float64)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    distinct, where = np.unique(sigma, return_inverse=True)
    xs = np.linspace(0.0, w, GRID)
    widths = trapezoid(_hazard(xs[None, :], distinct[:, None], shape), xs,
                       axis=1)
    return widths[where]


class Fit:
    """A fitted detection function.

    ``sigma`` and ``mu`` hold, per observation, the scale of its curve and
    its effective strip half-width; ``theta`` the fitted parameters -
    intercept, one coefficient per covariate (on the standardised scale),
    and for the hazard-rate the logarithm of its shape less one - with
    ``covariance`` their covariance matrix.
    """

    def __init__(self, key: str, covariates: Sequence[str], w: float,
                 means, sds):
        self.key = key
        self.covariates = tuple(covariates)
        self.w = float(w)
        self.means = means
        self.sds = sds
        self.theta = None
        self.covariance = None
        self.log_likelihood = None
        self.sigma = None
        self.mu = None

    # -- what the parameters mean ------------------------------------------ #

    @property
    def name(self) -> str:
        return model_name(self.key, self.covariates)

    @property
    def k(self) -> int:
        return len(self.theta)

    @property
    def aic(self) -> float:
        return 2.0 * self.k - 2.0 * self.log_likelihood

    @property
    def shape(self) -> Optional[float]:
        """The hazard-rate's *b*; ``None`` for the half-normal."""
        if self.key != HAZARD_RATE:
            return None
        return 1.0 + math.exp(float(self.theta[-1]))

    def standardise(self, values):
        """Raw covariate values (``n x covariates``) on the scale the
        coefficients are for."""
        import numpy as np

        values = np.asarray(values, dtype=np.float64)
        if not self.covariates:
            return values           # n rows, no columns
        values = values.reshape(-1, len(self.covariates))
        return (values - self.means) / self.sds

    def scales(self, z, theta=None):
        """sigma for standardised covariates *z* (``n x covariates``)."""
        import numpy as np

        theta = self.theta if theta is None else theta
        theta = np.asarray(theta, dtype=np.float64)
        eta = theta[0] + np.asarray(z, dtype=np.float64) @ theta[
            1:1 + len(self.covariates)]
        return np.exp(np.clip(eta, -20.0, 20.0))

    def widths(self, sigma, theta=None):
        """Effective strip half-width for each *sigma*."""
        theta = self.theta if theta is None else theta
        if self.key == HALF_NORMAL:
            return _half_normal_width(sigma, self.w)
        return _hazard_width(sigma, 1.0 + math.exp(float(theta[-1])), self.w)

    def curve(self, x, sigma, theta=None):
        """g(x) for each pair of *x* and *sigma* (broadcast)."""
        import numpy as np

        theta = self.theta if theta is None else theta
        if self.key == HALF_NORMAL:
            x, sigma = np.broadcast_arrays(
                np.asarray(x, dtype=np.float64),
                np.asarray(sigma, dtype=np.float64))
            return np.exp(-(x * x) / (2.0 * sigma * sigma))
        return _hazard(x, sigma, 1.0 + math.exp(float(theta[-1])))

    def mean_curve(self, xs):
        """The detection function averaged over the observations, each
        counted for the animals it stands for - what a histogram of the
        distances is compared with."""
        import numpy as np

        xs = np.asarray(xs, dtype=np.float64)
        weights = 1.0 / self.mu
        curves = self.curve(xs[None, :], self.sigma[:, None])
        return (curves * weights[:, None]).sum(axis=0) / weights.sum()

    def coefficients(self) -> List[dict]:
        """Per covariate: the coefficient per unit of it, its standard
        error, and whether it differs from nought::

            {"name", "label", "unit", "coefficient", "se", "z", "p_value"}
        """
        from scipy import stats

        rows = []
        for index, name in enumerate(self.covariates, start=1):
            scale = float(self.sds[index - 1])
            value = float(self.theta[index]) / scale
            error = math.sqrt(max(0.0, float(
                self.covariance[index][index]))) / scale
            score = value / error if error > 0 else None
            rows.append({
                "name": name, "label": label(name), "unit": unit(name),
                "coefficient": value, "se": error, "z": score,
                "p_value": None if score is None
                else float(2.0 * stats.norm.sf(abs(score))),
            })
        return rows


def _negative_log_likelihood(fit: Fit, theta, x, z) -> float:
    import numpy as np

    sigma = fit.scales(z, theta)
    mu = fit.widths(sigma, theta)
    if not np.all(np.isfinite(mu)) or np.any(mu <= 0):
        return 1e12
    if fit.key == HALF_NORMAL:
        value = np.sum(x * x / (2.0 * sigma * sigma) + np.log(mu))
    else:
        g = np.clip(fit.curve(x, sigma, theta), 1e-12, 1.0)
        value = np.sum(np.log(mu) - np.log(g))
    return float(value) if math.isfinite(float(value)) else 1e12


def _hessian(function, theta):
    """Second derivatives of *function* at *theta*, by central differences."""
    import numpy as np

    theta = np.asarray(theta, dtype=np.float64)
    m = theta.size
    h = 1e-4 * (np.abs(theta) + 1.0)
    hessian = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            total = 0.0
            for sign_i, sign_j in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                moved = theta.copy()
                moved[i] += sign_i * h[i]
                moved[j] += sign_j * h[j]
                total += sign_i * sign_j * function(moved)
            hessian[i, j] = hessian[j, i] = total / (4.0 * h[i] * h[j])
    return hessian


def fit(distances, w: float, key: str = HALF_NORMAL,
        covariates: Optional[Dict[str, Sequence[float]]] = None
        ) -> Optional[Fit]:
    """Fit a detection function by maximum likelihood.

    :param distances: perpendicular distances, none beyond *w*
    :param w: the truncation distance
    :param key: :data:`HALF_NORMAL` or :data:`HAZARD_RATE`
    :param covariates: ``name -> value per observation``; none = the
        conventional model
    :return: the fit, or ``None`` where no maximum with a usable curvature
        was found
    """
    import numpy as np
    from scipy.optimize import minimize

    if key not in KEYS:
        raise ValueError(f"Unknown detection function '{key}'.")
    x = np.asarray(distances, dtype=np.float64)
    n = int(x.size)
    names = list(covariates or {})
    if n < 2:
        return None
    if names:
        raw = np.column_stack([np.asarray(covariates[name], dtype=np.float64)
                               for name in names])
        means, sds = raw.mean(axis=0), raw.std(axis=0)
        if np.any(sds <= 0) or not np.all(np.isfinite(raw)):
            return None
    else:
        raw = np.zeros((n, 0))
        means, sds = np.zeros(0), np.ones(0)
    model = Fit(key, names, w, means, sds)
    z = model.standardise(raw)

    def nll(theta):
        return _negative_log_likelihood(model, theta, x, z)

    start = [math.log(max(float(np.std(x)), w / 1000.0, 1e-6))]
    start += [0.0] * len(names)
    if key == HAZARD_RATE:
        start.append(0.0)
    options = {"xatol": 1e-7, "fatol": 1e-9, "maxiter": 2000 * len(start),
               "maxfev": 4000 * len(start)}
    found = minimize(nll, start, method="Nelder-Mead", options=options)
    # the simplex shrinks around where it is; a second run from there finds
    # what the first one stopped short of
    found = minimize(nll, found.x, method="Nelder-Mead", options=options)
    theta = np.asarray(found.x, dtype=np.float64)
    best = float(found.fun)
    if not math.isfinite(best) or best >= 1e11:
        return None
    try:
        covariance = np.linalg.inv(_hessian(nll, theta))
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(covariance)) or np.any(
            np.diag(covariance) <= 0):
        return None

    model.theta = theta
    model.covariance = covariance
    model.log_likelihood = -best
    model.sigma = model.scales(z)
    model.mu = model.widths(model.sigma)
    return model


# ---------------------------------------------------------------------------
# The density that goes with it
# ---------------------------------------------------------------------------

def estimate(model: Fit, sizes: Sequence[float], line_index: Sequence[int],
             line_lengths: Sequence[float],
             covariates: Optional[Dict[str, Sequence[float]]] = None) -> dict:
    """The density of animals under a fitted model, and how sure it is.

    :param sizes: the group size of each observation
    :param line_index: the line each observation was seen from
    :param line_lengths: the length of every line, metres
    :param covariates: the values the model was fitted with (needed for the
        uncertainty of a model with covariates)
    :return: ``{"density_m2", "group_density_m2", "detection_probability",
        "esw_m", "cv_encounter", "encounter_variance", "cv_detection",
        "cv_density"}``
    """
    import numpy as np

    from . import distance_effort

    sizes = np.asarray(sizes, dtype=np.float64)
    lines = np.asarray(line_index, dtype=np.int64)
    lengths = [float(length) for length in line_lengths]
    total_length = sum(lengths)
    n = int(sizes.size)
    names = list(model.covariates)
    if names:
        raw = np.column_stack([np.asarray(covariates[name], dtype=np.float64)
                               for name in names])
    else:
        raw = np.zeros((n, 0))
    z = model.standardise(raw)

    def density(theta) -> float:
        mu = model.widths(model.scales(z, theta), theta)
        return float(np.sum(sizes / mu) / (2.0 * total_length))

    stands_for = sizes / model.mu
    value = float(stands_for.sum() / (2.0 * total_length))
    groups = float((1.0 / model.mu).sum() / (2.0 * total_length))

    per_line = np.bincount(lines, weights=stands_for, minlength=len(lengths))
    cv_encounter, how = distance_effort.encounter_rate_cv(
        per_line.tolist(), lengths)
    if cv_encounter is None or how == distance_effort.POISSON:
        # one line: the sightings as a Poisson sample, each with the
        # animals it stands for
        cv_encounter = float(
            math.sqrt(float(np.sum(stands_for ** 2))) / stands_for.sum())
        how = distance_effort.POISSON

    theta = np.asarray(model.theta, dtype=np.float64)
    step = 1e-4 * (np.abs(theta) + 1.0)
    gradient = np.zeros(theta.size)
    for index in range(theta.size):
        up, down = theta.copy(), theta.copy()
        up[index] += step[index]
        down[index] -= step[index]
        gradient[index] = (density(up) - density(down)) / (2.0 * step[index])
    variance = float(gradient @ np.asarray(model.covariance) @ gradient)
    cv_detection = math.sqrt(variance) / value if variance > 0 and value > 0 \
        else 0.0

    probability = n / float((model.w / model.mu).sum())
    return {
        "density_m2": value,
        "group_density_m2": groups,
        "detection_probability": probability,
        "esw_m": probability * model.w,
        "cv_encounter": cv_encounter,
        "encounter_variance": how,
        "cv_detection": cv_detection,
        "cv_density": math.sqrt(cv_encounter ** 2 + cv_detection ** 2),
    }


def effects(model: Fit, covariates: Dict[str, Sequence[float]]) -> List[dict]:
    """What each covariate of a model does, in the units of the survey.

    Per covariate the row of :meth:`Fit.coefficients`, and the effective
    strip half-width at a low and a high value of it - the 10th and the
    90th percentile of what was observed - with the other covariates at
    their mean: ``low``, ``high``, ``esw_low_m``, ``esw_high_m``.
    """
    import numpy as np

    rows = model.coefficients()
    for index, row in enumerate(rows):
        values = np.asarray(covariates[row["name"]], dtype=np.float64)
        low, high = (float(v) for v in np.percentile(values, [10, 90]))
        widths = []
        for value in (low, high):
            z = np.zeros((1, len(model.covariates)))
            z[0, index] = (value - model.means[index]) / model.sds[index]
            widths.append(float(model.widths(model.scales(z))[0]))
        row.update({"low": low, "high": high, "esw_low_m": widths[0],
                    "esw_high_m": widths[1],
                    "mean": float(values.mean()),
                    "min": float(values.min()), "max": float(values.max())})
    return rows


#: Below this p a covariate's effect is said to be one.
EFFECT_ALPHA = 0.05


def effect_sentence(row: dict) -> str:
    """One covariate's effect as it is said."""
    chance = row.get("p_value")
    widths = (f"the effective strip half-width is {row['esw_low_m']:.1f} m "
              f"at {row['low']:.0f} {row['unit']} and "
              f"{row['esw_high_m']:.1f} m at {row['high']:.0f} {row['unit']}")
    odds = ""
    if chance is not None:
        odds = " (p < 0.001)" if chance < 0.001 else f" (p = {chance:.3f})"
    if chance is None or chance >= EFFECT_ALPHA:
        return (f"{row['label'].capitalize()}: no clear effect - "
                f"{widths}{odds}.")
    how = "further away" if row["coefficient"] > 0 else "less far"
    return (f"The more {row['label']}, the {how} animals are seen: "
            f"{widths}{odds}.")
