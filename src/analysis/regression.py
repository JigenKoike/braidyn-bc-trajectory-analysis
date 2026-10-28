"""Session-level summaries of a distance timecourse, and their trend over days.

Two steps, kept apart because the first is reused and the second is not:

1. :func:`auc_per_session` / :func:`window_mean_per_session` collapse the long
   distance timecourse — one row per session × pair × timepoint — into one
   value per session × pair, over a window from :mod:`analysis.windows`.
2. :func:`day_regression` fits one ordinary least-squares line of that value
   on training day, per outcome pair.

The regressions are descriptive, one per pair, and their slopes are **not**
compared with each other; the figures say so. The p values are corrected
over the three pairs within one window, which is the family a co-author asked
for (2026-10-05). The figures use the Benjamini-Hochberg q, the correction
the mixed models' tables already use (co-author, 2026-10-23); the Holm p
beside it is what the figures used before, kept for the analyses that were
read with it. Sessions are treated as independent here — the repeated
measurement of each mouse is handled by the mixed models in
:mod:`analysis.lmm`, not by these lines.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .multiple_comparison import add_corrected_column
from .pairs import PAIR_ORDER
from .windows import Window, window_auc, window_mean

#: Columns that identify one session's value for one pair.
SESSION_KEYS: tuple[str, ...] = ("pair", "mouse", "day", "session_id")


def _per_session(
    distance_long: pd.DataFrame,
    window: Window,
    reduce,
    *,
    time_col: str,
    value_col: str,
    group_cols: Sequence[str],
    out_col: str,
) -> pd.DataFrame:
    group_cols = list(group_cols)
    ordered = distance_long.sort_values(list(group_cols) + [time_col])
    return (
        ordered.groupby(group_cols)
        .apply(
            lambda g: reduce(g[time_col].to_numpy(), g[value_col].to_numpy(), window),
            include_groups=False,
        )
        .rename(out_col)
        .reset_index()
    )


def auc_per_session(
    distance_long: pd.DataFrame,
    window: Window,
    *,
    time_col: str = "t",
    value_col: str = "dist",
    group_cols: Sequence[str] = SESSION_KEYS,
    out_col: str = "auc",
) -> pd.DataFrame:
    """Trapezoidal area under the timecourse over ``window``, per group.

    ``distance_long`` is the long distance table: ``group_cols`` plus a time
    column and a value column. Returns ``group_cols`` plus ``out_col``.
    """
    return _per_session(
        distance_long, window, window_auc,
        time_col=time_col, value_col=value_col, group_cols=group_cols, out_col=out_col,
    )


def window_mean_per_session(
    distance_long: pd.DataFrame,
    window: Window,
    *,
    time_col: str = "t",
    value_col: str = "dist",
    group_cols: Sequence[str] = SESSION_KEYS,
    out_col: str = "mean",
) -> pd.DataFrame:
    """Mean of the timecourse over ``window``, per group.

    Unlike the AUC this does not scale with the window's width, so the two
    response windows can be compared directly despite their different lengths.
    """
    return _per_session(
        distance_long, window, window_mean,
        time_col=time_col, value_col=value_col, group_cols=group_cols, out_col=out_col,
    )


def day_regression(
    sessions: pd.DataFrame,
    *,
    value_col: str = "auc",
    day_col: str = "day",
    pair_col: str = "pair",
    mouse_col: str = "mouse",
    pair_order: Sequence[str] = PAIR_ORDER,
    predict_days: tuple[int, int] = (1, 15),
    label: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """One least-squares line of ``value_col`` on training day, per outcome pair.

    Returns one row per pair with the slope, its standard error and 95%
    confidence interval, the uncorrected p value, the Holm-corrected p and the
    Benjamini-Hochberg q (family for both: the pairs in this table), Pearson
    r, the intercept, the fitted
    values at ``predict_days`` and the relative change between them.

    ``label`` (e.g. ``{"window": "late"}``) becomes the leading columns, so
    tables for several windows can be concatenated.
    """
    label = dict(label) if label is not None else {}
    first, last = predict_days
    rows = []
    for pair in pair_order:
        sub = sessions[sessions[pair_col] == pair]
        x = sub[day_col].to_numpy(dtype=float)
        y = sub[value_col].to_numpy(dtype=float)
        res = stats.linregress(x, y)
        n = len(sub)
        tcrit = stats.t.ppf(0.975, n - 2)
        pred_first = res.intercept + res.slope * first
        pred_last = res.intercept + res.slope * last
        rows.append({
            **label,
            "pair": pair,
            "n_sessions": n,
            "n_mice": sub[mouse_col].nunique(),
            "slope_per_day": res.slope,
            "slope_se": res.stderr,
            "ci95_lo": res.slope - tcrit * res.stderr,
            "ci95_hi": res.slope + tcrit * res.stderr,
            "p_value": res.pvalue,
            "r": res.rvalue,
            "intercept": res.intercept,
            f"pred_day{first}": pred_first,
            f"pred_day{last}": pred_last,
            f"rel_change_day{first}_to_day{last}": pred_last / pred_first - 1,
        })
    out = pd.DataFrame(rows)
    # Over the three pairs: one family per window.
    add_corrected_column(out, "p_value", "p_holm", method="holm")
    add_corrected_column(out, "p_value", "q_value", method="fdr_bh")
    cols = [c for c in out.columns if c != "q_value"]
    cols.insert(cols.index("p_holm") + 1, "q_value")
    return out[cols]


def ci_excludes_zero(row: Mapping[str, float]) -> bool:
    """Whether a regression row's 95% interval lies entirely on one side of 0.

    The slope panels used this to fill a marker until 2026-10-23; they now
    fill it where the corrected q is below 0.05, as their asterisks do, and
    no figure calls this. The two can disagree — the interval is not
    corrected and the q is — which is why the figures moved to one test.
    """
    return bool(row["ci95_lo"] > 0 or row["ci95_hi"] < 0)


def fitted_line(row: Mapping[str, float], days: np.ndarray) -> np.ndarray:
    """The fitted values of a :func:`day_regression` row, for drawing its line."""
    return row["intercept"] + row["slope_per_day"] * np.asarray(days, dtype=float)
