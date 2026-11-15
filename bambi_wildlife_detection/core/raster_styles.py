# -*- coding: utf-8 -*-
"""How the survey rasters are coloured - as data, so it can be tested
without QGIS. The dock turns these into renderers."""

from typing import List, Tuple

Colour = Tuple[int, int, int, int]

#: Ground inside the survey area that no frame saw: a grey that is neither
#: end of the ramp.
GAP_COLOUR: Colour = (90, 90, 90, 200)

#: From "seen once" to "seen most often": light to dark, one direction.
_COVERAGE_RAMP = ((255, 255, 204), (161, 218, 180), (65, 182, 196),
                  (44, 127, 184), (37, 52, 148))


def _ramp(fraction: float) -> Colour:
    """The ramp's colour at *fraction* (0 = lightest, 1 = darkest)."""
    fraction = min(max(fraction, 0.0), 1.0)
    position = fraction * (len(_COVERAGE_RAMP) - 1)
    lower = min(int(position), len(_COVERAGE_RAMP) - 2)
    share = position - lower
    low, high = _COVERAGE_RAMP[lower], _COVERAGE_RAMP[lower + 1]
    return tuple(int(round(low[i] + (high[i] - low[i]) * share))
                 for i in range(3)) + (255,)


def coverage_classes(minimum: float, maximum: float
                     ) -> List[Tuple[float, Colour, str]]:
    """The classes of a coverage map with counts from *minimum* to
    *maximum*, as ``(upper bound, rgba, label)`` in rising order - the
    items of a discrete colour ramp, where a value takes the first class
    whose bound it does not exceed.

    **0 is a class of its own**: ground of the survey area that no frame
    saw. On a ramp it would be the coolest colour, "seen a little"; it is
    the one thing the map is looked at for. The counts above it are one
    class each up to eight frames, and double from there (9-16, 17-32, ...),
    so that a few places seen hundreds of times do not flatten the rest.
    """
    top = max(1, int(round(maximum)))
    classes: List[Tuple[float, Colour, str]] = []
    if minimum <= 0:
        classes.append((0.0, GAP_COLOUR, "not seen"))

    bounds: List[Tuple[int, int]] = []
    lower = 1
    while lower <= top:
        upper = lower if lower <= 8 else min(top, (lower - 1) * 2)
        bounds.append((lower, upper))
        lower = upper + 1
    for index, (lower, upper) in enumerate(bounds):
        fraction = index / (len(bounds) - 1) if len(bounds) > 1 else 0.0
        label = str(lower) if lower == upper else f"{lower}-{upper}"
        classes.append((float(upper), _ramp(fraction), label))
    return classes
