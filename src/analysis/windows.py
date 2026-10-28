"""Analysis time windows, in seconds relative to cue onset (t = 0 s).

Single source of truth for the project: scripts and notebooks import these
constants and helpers instead of restating the numbers, so a window can never
mean one thing in the pipeline and another in a figure.

Each window is ``(lo, hi, hi_inclusive)``. ``pre`` is the baseline window
before the cue; ``early`` and ``late`` split the post-cue response.

``early`` and ``late`` share the boundary at 2.0 s and both include it. The
primary measure over these windows is a trapezoidal integral, which needs
both endpoints, so the single sample at t = 2.0 s contributes to both.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Window = tuple[float, float, bool]

PRE_WINDOW: Window = (-1.0, 0.0, False)
EARLY_WINDOW: Window = (0.2, 2.0, True)
LATE_WINDOW: Window = (2.0, 4.0, True)

WINDOWS: dict[str, Window] = {
    "pre": PRE_WINDOW,
    "early": EARLY_WINDOW,
    "late": LATE_WINDOW,
}


def window_mask(t: ArrayLike, window: Window) -> NDArray[np.bool_]:
    """Boolean mask selecting the samples of ``t`` that fall inside ``window``."""
    lo, hi, hi_inclusive = window
    t = np.asarray(t, dtype=float)
    return (t >= lo) & ((t <= hi) if hi_inclusive else (t < hi))


def window_auc(t: ArrayLike, y: ArrayLike, window: Window) -> np.floating | NDArray:
    """Trapezoidal integral of ``y`` over ``window``.

    ``t`` indexes the first axis of ``y``, so a 2-D ``y`` of shape
    ``(n_time, n_series)`` yields one value per series.

    Raises
    ------
    ValueError
        If fewer than two samples fall inside the window. An integral over
        zero or one sample is zero, which would silently pass as data.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = window_mask(t, window)
    if mask.sum() < 2:
        raise ValueError(
            f"window {window} contains {mask.sum()} sample(s) of the given time "
            "axis; at least 2 are needed for a trapezoidal integral"
        )
    return np.trapezoid(y[mask, ...], t[mask], axis=0)


def window_mean(t: ArrayLike, y: ArrayLike, window: Window) -> np.floating | NDArray:
    """Mean of ``y`` over ``window``, ignoring NaNs.

    ``t`` indexes the first axis of ``y``, so a 2-D ``y`` of shape
    ``(n_time, n_series)`` yields one value per series. An empty window gives
    NaN, shaped like a single sample of ``y``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = window_mask(t, window)
    if not mask.any():
        return np.float64(np.nan) if y.ndim == 1 else np.full(y.shape[1:], np.nan)
    return np.nanmean(y[mask, ...], axis=0)
