# -*- coding: utf-8 -*-
"""Tone curve for thermal imagery.

A :class:`ThermalCurve` maps per-pixel temperatures (°C) to normalized
display intensities in ``[0, 1]``. The mapping is defined by

* a temperature *domain* ``[domain_lo, domain_hi]`` — temperatures are
  normalized into this window first (values outside are clamped), and
* a list of *control points* in normalized ``(x, y)`` coordinates through
  which a monotone cubic curve (PCHIP, Fritsch–Carlson) is interpolated,
  exactly like the "Curves" tool in image editors.

Unlike the plain lower/upper thresholds (which render out-of-range pixels
black), the curve clamps: temperatures below the domain map to the curve
value at ``x = 0`` and temperatures above to the value at ``x = 1``.

Because a flight rarely has well-known temperature bounds up front, the
module also provides :func:`scan_temperatures`, which parses a collection
of radiometric images and accumulates the exact global minimum/maximum, a
fixed-bin histogram and (approximate) percentiles — the backend of the
"Auto Detect" button in the UI.

This module is pure numpy (no Qt/QGIS imports) so it is unit-testable in
the stubbed test tier.
"""

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# Default control points: identity mapping (linear stretch over the domain).
DEFAULT_POINTS: List[Tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]

# Minimum horizontal separation between two control points (normalized x).
MIN_POINT_SEPARATION = 0.01

# Fixed histogram binning used by scan_temperatures. 0.25 °C bins covering
# every temperature a DJI radiometric camera can plausibly report.
_HIST_LO = -100.0
_HIST_HI = 600.0
_HIST_STEP = 0.25


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch–Carlson slopes for a monotone cubic Hermite interpolant."""
    h = np.diff(x)
    d = np.diff(y) / h
    n = len(x)
    m = np.zeros(n, dtype=np.float64)
    if n == 2:
        m[:] = d[0]
        return m
    m[0] = d[0]
    m[-1] = d[-1]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    return m


def _pchip_eval(x: np.ndarray, y: np.ndarray, m: np.ndarray,
                xs: np.ndarray) -> np.ndarray:
    """Evaluate the cubic Hermite interpolant at *xs* (clamped to range)."""
    xs = np.clip(xs, x[0], x[-1])
    idx = np.clip(np.searchsorted(x, xs, side="right") - 1, 0, len(x) - 2)
    h = x[idx + 1] - x[idx]
    t = (xs - x[idx]) / h
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    start = h00 * y[idx] + h10 * h * m[idx]
    end = h01 * y[idx + 1] + h11 * h * m[idx + 1]
    return start + end


def normalize_points(
    points: Sequence[Sequence[float]],
) -> List[Tuple[float, float]]:
    """Return a sanitized copy of *points* usable as curve control points.

    Coordinates are clipped to ``[0, 1]``, points are sorted by x, near-
    duplicate x values are dropped, and endpoint columns at ``x = 0`` and
    ``x = 1`` are guaranteed (inserted by extending the outermost y when
    missing). Falls back to :data:`DEFAULT_POINTS` when fewer than two
    usable points remain.
    """
    cleaned = []
    for p in points:
        try:
            x = min(max(float(p[0]), 0.0), 1.0)
            y = min(max(float(p[1]), 0.0), 1.0)
        except (TypeError, ValueError, IndexError):
            continue
        cleaned.append((x, y))
    cleaned.sort(key=lambda p: p[0])

    deduped: List[Tuple[float, float]] = []
    for x, y in cleaned:
        if deduped and x - deduped[-1][0] < MIN_POINT_SEPARATION:
            continue
        deduped.append((x, y))
    if len(deduped) < 2:
        return list(DEFAULT_POINTS)

    if deduped[0][0] > 0.0:
        deduped.insert(0, (0.0, deduped[0][1]))
    if deduped[-1][0] < 1.0:
        deduped.append((1.0, deduped[-1][1]))
    return deduped


class ThermalCurve:
    """A tone curve over a temperature domain.

    :param domain_lo: Temperature (°C) mapped to normalized position 0.
    :param domain_hi: Temperature (°C) mapped to normalized position 1.
    :param points: Control points in normalized ``(x, y)`` coordinates.
    """

    def __init__(self, domain_lo: float, domain_hi: float,
                 points: Optional[Sequence[Sequence[float]]] = None) -> None:
        self.domain_lo = float(domain_lo)
        self.domain_hi = float(domain_hi)
        if self.domain_hi <= self.domain_lo:
            self.domain_hi = self.domain_lo + 1.0
        self.points = normalize_points(points if points is not None
                                       else DEFAULT_POINTS)
        self._lut_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "domain_lo": self.domain_lo,
            "domain_hi": self.domain_hi,
            "points": [[x, y] for x, y in self.points],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ThermalCurve":
        return cls(
            domain_lo=float(data.get("domain_lo", 0.0)),
            domain_hi=float(data.get("domain_hi", 40.0)),
            points=data.get("points", DEFAULT_POINTS),
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def lut(self, size: int = 256) -> np.ndarray:
        """Return the curve sampled at *size* equidistant x positions."""
        if self._lut_cache is None or len(self._lut_cache) != size:
            px = np.asarray([p[0] for p in self.points], dtype=np.float64)
            py = np.asarray([p[1] for p in self.points], dtype=np.float64)
            slopes = _pchip_slopes(px, py)
            xs = np.linspace(0.0, 1.0, size)
            self._lut_cache = np.clip(
                _pchip_eval(px, py, slopes, xs), 0.0, 1.0)
        return self._lut_cache

    def apply(self, temperatures: np.ndarray, lut_size: int = 1024) -> np.ndarray:
        """Map a temperature array (°C) to normalized intensities [0, 1]."""
        arr = np.asarray(temperatures, dtype=np.float32)
        span = self.domain_hi - self.domain_lo
        t = np.clip((arr - self.domain_lo) / span, 0.0, 1.0)
        lut = self.lut(lut_size)
        xs = np.linspace(0.0, 1.0, lut_size)
        return np.interp(t, xs, lut).astype(np.float32)

    # ------------------------------------------------------------------
    # Introspection helpers (UI labels)
    # ------------------------------------------------------------------

    def is_identity(self) -> bool:
        """True when the control points form the unmodified diagonal."""
        if len(self.points) != 2:
            return False
        (x0, y0), (x1, y1) = self.points
        starts_at_origin = abs(x0) < 1e-9 and abs(y0) < 1e-9
        ends_at_one = abs(x1 - 1.0) < 1e-9 and abs(y1 - 1.0) < 1e-9
        return starts_at_origin and ends_at_one

    def describe(self) -> str:
        text = (f"{self.domain_lo:.1f} – {self.domain_hi:.1f} °C, "
                f"{len(self.points)} points")
        return f"{text} (linear)" if self.is_identity() else text


class TemperatureScan:
    """Aggregated result of scanning a set of radiometric images.

    Attributes: ``minimum`` / ``maximum`` (exact, °C), ``counts`` /
    ``bin_edges`` (fixed-bin histogram over all scanned pixels),
    ``n_images`` (successfully parsed), ``n_errors`` (skipped) and
    ``cancelled``.
    """

    def __init__(self) -> None:
        self.minimum: Optional[float] = None
        self.maximum: Optional[float] = None
        self.bin_edges = np.arange(
            _HIST_LO, _HIST_HI + _HIST_STEP, _HIST_STEP)
        self.counts = np.zeros(len(self.bin_edges) - 1, dtype=np.int64)
        self.n_images = 0
        self.n_errors = 0
        self.cancelled = False

    def add(self, temperatures: np.ndarray) -> None:
        arr = np.asarray(temperatures, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            self.n_errors += 1
            return
        lo = float(finite.min())
        hi = float(finite.max())
        self.minimum = lo if self.minimum is None else min(self.minimum, lo)
        self.maximum = hi if self.maximum is None else max(self.maximum, hi)
        clipped = np.clip(finite, _HIST_LO, _HIST_HI - _HIST_STEP)
        hist, _ = np.histogram(clipped, bins=self.bin_edges)
        self.counts += hist
        self.n_images += 1

    def percentile(self, q: float) -> Optional[float]:
        """Approximate percentile (0–100) from the accumulated histogram."""
        total = int(self.counts.sum())
        if total == 0:
            return None
        target = total * (min(max(q, 0.0), 100.0) / 100.0)
        cum = np.cumsum(self.counts)
        idx = int(np.searchsorted(cum, target))
        idx = min(idx, len(self.counts) - 1)
        # Centre of the bin that contains the target rank
        return float(self.bin_edges[idx] + _HIST_STEP / 2.0)

    def suggested_domain(self, robust: bool = False) -> Optional[Tuple[float, float]]:
        """Return (lo, hi) for the curve domain, or None when nothing scanned.

        With ``robust=True`` the 1st/99th percentiles are used instead of the
        exact extremes, which suppresses dead/hot pixels.
        """
        if self.n_images == 0 or self.minimum is None:
            return None
        if robust:
            lo = self.percentile(1.0)
            hi = self.percentile(99.0)
            if lo is not None and hi is not None and hi > lo:
                return (lo, hi)
        return (self.minimum, self.maximum)


def scan_temperatures(
    paths: Sequence[str],
    parse_fn: Callable[[str], np.ndarray],
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> TemperatureScan:
    """Parse every image in *paths* and accumulate temperature statistics.

    :param parse_fn: Callable mapping a file path to a 2-D °C array
        (typically ``Thermal.parse``). Images whose parse raises are
        counted in ``n_errors`` and skipped.
    :param progress_cb: Optional ``(index, total, path)`` callback invoked
        before each image is parsed.
    :param cancel_cb: Optional callable checked before each image; when it
        returns True the scan stops and the partial result is returned with
        ``cancelled`` set.
    """
    scan = TemperatureScan()
    total = len(paths)
    for i, path in enumerate(paths):
        if cancel_cb is not None and cancel_cb():
            scan.cancelled = True
            break
        if progress_cb is not None:
            progress_cb(i, total, path)
        try:
            arr = parse_fn(path)
        except Exception:
            scan.n_errors += 1
            continue
        scan.add(arr)
    return scan
