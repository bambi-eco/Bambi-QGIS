# -*- coding: utf-8 -*-
"""What a pattern of counts over cells says: how dense, how clumped, where.

The statistics behind the grid analysis (:mod:`core.grid_density`), kept
apart from it because they know nothing of drones or animals - they take
counts, the ground each count was made on, and which cells lie next to which.

* :func:`ratio_interval` - the density over all cells, ``sum(counts) /
  sum(ground)``, with an interval from **resampling the cells**: the cells
  are drawn again and again with replacement, and the middle 95 % of the
  densities that gives is the interval. It asks nothing of how the counts
  are distributed. It does take the cells as independent, so where animals
  are clumped across cells (:func:`morans_i` above its expectation) it is
  somewhat too narrow.
* :func:`dispersion` - the **variance-to-mean ratio**, generalised to cells
  that were not all seen equally: 1 for animals scattered at random, above
  1 for clumped ones, below for evenly spaced ones, with the chi-square
  test that goes with it.
* :func:`gi_star` - **Getis-Ord Gi***: for every cell, whether it and its
  neighbours together hold more (a hot spot) or less (a cold spot) than
  cells picked at random would, as a z-score.
* :func:`morans_i` - **Moran's I**: whether neighbouring cells are more alike
  than cells in general, with a permutation test.
* :func:`wilson` - the interval of a share, for the share of cells that
  held any animal at all.

numpy; scipy for the chi-square distribution where it is installed, a close
approximation where it is not. Random draws use a fixed seed: the same data
give the same result every time.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

#: Seed of every resampling: results are reproducible.
SEED = 20240501
#: z-scores at which a hot or cold spot is called, with their confidence.
LEVELS = ((2.5758, 99), (1.9600, 95), (1.6449, 90))

CLUSTERED = "clustered"
RANDOM = "random"
REGULAR = "regular"


def _normal_sf(z: float) -> float:
    """Upper tail of the standard normal distribution."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _chi2_sf(x: float, df: int) -> float:
    """Upper tail of the chi-square distribution."""
    if df <= 0:
        return float("nan")
    try:
        from scipy import stats
        return float(stats.chi2.sf(x, df))
    except ImportError:
        # Wilson and Hilferty: the cube root of a chi-square over its
        # degrees of freedom is close to normal.
        if x <= 0:
            return 1.0
        spread = 2.0 / (9.0 * df)
        z = ((x / df) ** (1.0 / 3.0) - (1.0 - spread)) / math.sqrt(spread)
        return _normal_sf(z)


# ---------------------------------------------------------------------------
# Density with an interval
# ---------------------------------------------------------------------------

def ratio_interval(counts: Sequence[float], ground: Sequence[float],
                   n_boot: int = 2000, level: float = 0.95,
                   seed: int = SEED) -> dict:
    """``sum(counts) / sum(ground)`` with an interval from resampling.

    :param counts: what was counted in each cell
    :param ground: the ground each count was made on (any unit; the ratio
        is per that unit)
    :return: ``{"estimate", "low", "high", "se", "n", "n_boot"}`` -
        ``low`` / ``high`` / ``se`` are ``None`` with fewer than two cells,
        the estimate with no ground at all
    """
    import numpy as np

    counts = np.asarray(list(counts), dtype=np.float64)
    ground = np.asarray(list(ground), dtype=np.float64)
    n = int(counts.size)
    total = float(ground.sum())
    result = {"estimate": float(counts.sum()) / total if total > 0 else None,
              "low": None, "high": None, "se": None, "n": n,
              "n_boot": int(n_boot)}
    if n < 2 or total <= 0:
        return result
    rng = np.random.default_rng(seed)
    ratios = np.empty(n_boot, dtype=np.float64)
    # in batches: a thousand cells drawn two thousand times fit in memory,
    # a hundred thousand do not
    batch = max(1, min(n_boot, 4_000_000 // n))
    done = 0
    while done < n_boot:
        size = min(batch, n_boot - done)
        drawn = rng.integers(0, n, size=(size, n))
        on = ground[drawn].sum(axis=1)
        seen = counts[drawn].sum(axis=1)
        ratios[done:done + size] = np.where(on > 0, seen / np.where(
            on > 0, on, 1.0), np.nan)
        done += size
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size < 2:
        return result
    tail = (1.0 - level) / 2.0
    result["low"] = float(np.quantile(ratios, tail))
    result["high"] = float(np.quantile(ratios, 1.0 - tail))
    result["se"] = float(ratios.std(ddof=1))
    return result


# ---------------------------------------------------------------------------
# Clumped, random or regular
# ---------------------------------------------------------------------------

def dispersion(counts: Sequence[float],
               ground: Optional[Sequence[float]] = None,
               alpha: float = 0.05) -> dict:
    """The variance-to-mean ratio of counts, and what it says.

    Animals scattered at random give counts whose variance equals their
    mean. With cells seen to different extents, "the mean" of a cell is what
    its ground would hold at the overall density, and the ratio becomes
    ``sum((count - expected)^2 / expected) / (cells - 1)`` - which is the
    plain variance-to-mean ratio when every cell has the same ground.

    :param ground: the ground each count was made on; ``None`` = all equal
    :return: ``{"index", "chi2", "df", "p_clustered", "p_regular",
        "pattern", "n"}`` - *pattern* is ``"clustered"``, ``"regular"`` or
        ``"random"`` at *alpha* (two-sided). ``index`` is ``None`` with fewer
        than two cells or nothing counted.
    """
    counts = [float(value) for value in counts]
    ground = [1.0] * len(counts) if ground is None else [
        float(value) for value in ground]
    pairs = [(c, g) for c, g in zip(counts, ground) if g > 0]
    n = len(pairs)
    total = sum(c for c, _g in pairs)
    result = {"index": None, "chi2": None, "df": max(0, n - 1),
              "p_clustered": None, "p_regular": None, "pattern": None,
              "n": n}
    if n < 2 or total <= 0:
        return result
    density = total / sum(g for _c, g in pairs)
    chi2 = sum((c - density * g) ** 2 / (density * g) for c, g in pairs)
    df = n - 1
    upper = _chi2_sf(chi2, df)
    result.update({
        "index": chi2 / df, "chi2": chi2,
        "p_clustered": upper, "p_regular": 1.0 - upper})
    if upper < alpha / 2.0:
        result["pattern"] = CLUSTERED
    elif 1.0 - upper < alpha / 2.0:
        result["pattern"] = REGULAR
    else:
        result["pattern"] = RANDOM
    return result


# ---------------------------------------------------------------------------
# Hot spots
# ---------------------------------------------------------------------------

def gi_star(values: Sequence[float],
            neighbours: Sequence[Sequence[int]]) -> List[Optional[float]]:
    """Getis-Ord Gi* of every cell, as a z-score.

    A cell's neighbourhood is the cell itself and the cells next to it
    (*neighbours*: for each cell the indices of those). Its value is
    compared with what the same number of cells picked at random would
    hold: well above is a hot spot, well below a cold spot (Ord and Getis
    1995, with weights of one).

    :return: one z-score per cell; ``None`` where there is none - fewer
        than two cells, all values equal, or a neighbourhood that is the
        whole grid
    """
    values = [float(value) for value in values]
    n = len(values)
    if n < 2:
        return [None] * n
    mean = sum(values) / n
    spread_2 = sum(value * value for value in values) / n - mean * mean
    if spread_2 <= 1e-15 * max(1.0, mean * mean):
        return [None] * n
    spread = math.sqrt(spread_2)
    scores: List[Optional[float]] = []
    for index in range(n):
        around = set(neighbours[index]) | {index}
        weight = len(around)
        under = (n * weight - weight * weight) / (n - 1.0)
        if under <= 0:
            scores.append(None)
            continue
        local = sum(values[other] for other in around)
        scores.append((local - mean * weight) / (spread * math.sqrt(under)))
    return scores


def spot(z: Optional[float]) -> str:
    """What a Gi* z-score is called: ``"hot spot (99 %)"`` ...
    ``"cold spot (90 %)"``, or ``""`` where nothing stands out."""
    if z is None:
        return ""
    for threshold, confidence in LEVELS:
        if z >= threshold:
            return f"hot spot ({confidence} %)"
        if z <= -threshold:
            return f"cold spot ({confidence} %)"
    return ""


# ---------------------------------------------------------------------------
# Are neighbours alike
# ---------------------------------------------------------------------------

def morans_i(values: Sequence[float], neighbours: Sequence[Sequence[int]],
             permutations: int = 999, seed: int = SEED) -> dict:
    """Moran's I: whether cells next to each other have similar values.

    Above its expectation (``-1 / (n - 1)``, close to zero) neighbours are
    alike - high next to high, low next to low; below it they differ. Its
    significance is found by shuffling the values over the cells: how often
    does chance alone give an I as far from the expectation as this one.

    :return: ``{"i", "expected", "z", "p_value", "permutations",
        "pattern", "n", "links"}`` - *pattern* is ``"clustered"`` (alike),
        ``"regular"`` (unlike) or ``"random"`` at 5 %. ``i`` is ``None``
        where it cannot be worked out: fewer than three cells, no cell with
        a neighbour, or all values equal.
    """
    import numpy as np

    values = np.asarray(list(values), dtype=np.float64)
    n = int(values.size)
    first, second = [], []
    for index, others in enumerate(neighbours):
        for other in others:
            if other != index:
                first.append(index)
                second.append(other)
    result = {"i": None, "expected": None, "z": None, "p_value": None,
              "permutations": int(permutations), "pattern": None, "n": n,
              "links": len(first) // 2}
    if n < 3 or not first:
        return result
    centred = values - values.mean()
    scatter = float((centred * centred).sum())
    if scatter <= 1e-15 * max(1.0, float((values * values).sum())):
        return result
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    scale = n / float(first.size) / scatter

    def statistic(z):
        return float((z[first] * z[second]).sum()) * scale

    observed = statistic(centred)
    expected = -1.0 / (n - 1.0)
    result.update({"i": observed, "expected": expected})
    if permutations <= 0:
        return result
    rng = np.random.default_rng(seed)
    shuffled = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        shuffled[index] = statistic(rng.permutation(centred))
    off = abs(observed - expected) - 1e-12
    as_far = int((np.abs(shuffled - expected) >= off).sum())
    p_value = (as_far + 1.0) / (permutations + 1.0)
    spread = float(shuffled.std(ddof=1))
    result["z"] = (observed - float(shuffled.mean())) / spread \
        if spread > 0 else None
    result["p_value"] = p_value
    if p_value < 0.05:
        result["pattern"] = CLUSTERED if observed > expected else REGULAR
    else:
        result["pattern"] = RANDOM
    return result


# ---------------------------------------------------------------------------
# A share and how sure it is
# ---------------------------------------------------------------------------

def wilson(successes: int, n: int, z: float = 1.96
           ) -> Tuple[Optional[float], Optional[float]]:
    """The Wilson interval of a share ``successes / n`` - well behaved near
    0 and 1, where the textbook interval is not. ``(None, None)`` for
    ``n = 0``."""
    if n <= 0:
        return None, None
    share = successes / n
    under = 1.0 + z * z / n
    centre = (share + z * z / (2.0 * n)) / under
    inner = share * (1.0 - share) / n + z * z / (4.0 * n * n)
    half = z * math.sqrt(inner) / under
    return max(0.0, centre - half), min(1.0, centre + half)


def index_neighbours(names: Sequence[str],
                     near: Dict[str, Sequence[str]]) -> List[List[int]]:
    """Neighbours by name as neighbours by position in *names*; names that
    are not in the list are dropped."""
    position = {name: index for index, name in enumerate(names)}
    return [[position[other] for other in near.get(name, ())
             if other in position] for name in names]
