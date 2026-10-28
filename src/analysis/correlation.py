"""Per-mouse rank correlation, tested across animals.

A session-level correlation computed over the pooled sessions of every mouse
treats repeated measurements of one animal as independent, which they are
not. The two-stage alternative here keeps the mouse as the unit: a Spearman
rho inside each animal, then a test of those rho values against zero.

:func:`per_mouse_rho` is the first stage and returns one row per mouse, so a
figure can draw the distribution it tested. :func:`rho_against_zero` is the
second.

:func:`per_mouse_slope` is the same first stage in the units of the measure
rather than of ranks, for the questions that compare one animal's rate of
change with another's — how fast its separation grew against how fast it
learned the task.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

#: A mouse needs at least this many sessions for its rho to be worth having.
#: Below three, a rank correlation can only land on a couple of values.
MIN_SESSIONS_PER_MOUSE = 3


def per_mouse_rho(
    sessions: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    mouse_col: str = "mouse",
    min_sessions: int = MIN_SESSIONS_PER_MOUSE,
) -> pd.DataFrame:
    """Spearman rho between ``x_col`` and ``y_col`` within each mouse.

    ``sessions`` is one row per session. Mice with fewer than
    ``min_sessions`` rows are left out, as are any whose rho is undefined
    because one of the two columns does not vary within that animal.

    Returns ``mouse``, ``n_sessions`` and ``rho``, one row per mouse.
    """
    rows = []
    for mouse, g in sessions.groupby(mouse_col, sort=True):
        if len(g) < min_sessions:
            continue
        with warnings.catch_warnings():
            # A constant column is the case the next line handles; scipy
            # warning about it as well is noise. Only that one is silenced.
            warnings.simplefilter("ignore", stats.ConstantInputWarning)
            rho = stats.spearmanr(g[x_col], g[y_col]).statistic
        if not np.isfinite(rho):
            continue  # one of the columns does not vary for this mouse
        rows.append({mouse_col: mouse, "n_sessions": len(g), "rho": float(rho)})
    return pd.DataFrame(rows, columns=[mouse_col, "n_sessions", "rho"])


def rho_against_zero(rho: Sequence[float]) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank test of the per-mouse rho against zero.

    The question is whether the correlation inside an animal points the same
    way across animals, not whether any one animal's rho is large. Returns
    ``n_mice``, ``median_rho`` and ``p``.

    The signed-rank test assumes the values are symmetric about their centre
    under the null; rho is bounded on [-1, 1], which keeps that close enough
    to true. A sign test is the assumption-free fallback.
    """
    rho = np.asarray(rho, dtype=float)
    assert rho.size > 0, "no mouse had a defined rho"
    assert np.isfinite(rho).all(), "rho values must all be finite"
    return {
        "n_mice": int(rho.size),
        "median_rho": float(np.median(rho)),
        "p": float(stats.wilcoxon(rho).pvalue),
    }


def per_mouse_slope(
    sessions: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    mouse_col: str = "mouse",
    min_sessions: int = MIN_SESSIONS_PER_MOUSE,
) -> pd.DataFrame:
    """Least-squares slope of ``y_col`` on ``x_col`` within each mouse.

    The first stage of the two-stage analysis in the units of the measures
    themselves: how much the distance grew per training day in this animal,
    which can then be set against how much its behaviour improved per
    training day. A rank correlation cannot be used for that comparison —
    two rank correlations of 1.0 say nothing about which animal changed more.

    Mice with fewer than ``min_sessions`` rows are left out, as is any whose
    ``x_col`` does not vary. Returns ``mouse``, ``n_sessions`` and ``slope``,
    one row per mouse.
    """
    rows = []
    for mouse, g in sessions.groupby(mouse_col, sort=True):
        if len(g) < min_sessions:
            continue
        x = g[x_col].to_numpy(dtype=float)
        y = g[y_col].to_numpy(dtype=float)
        if np.ptp(x) == 0:
            continue  # no spread in x, so no slope to speak of
        slope = stats.linregress(x, y).slope
        rows.append({mouse_col: mouse, "n_sessions": len(g), "slope": float(slope)})
    return pd.DataFrame(rows, columns=[mouse_col, "n_sessions", "slope"])
