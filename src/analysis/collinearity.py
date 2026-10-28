"""How much the predictors of a model overlap.

Two views of the same question. The **correlation matrix** shows which pairs
move together; the **variance inflation factor** shows, for each predictor,
how much of it the others already carry together — the square of its
standard error is inflated by exactly this factor compared with a predictor
the others could not predict at all:

    VIF_j = 1 / (1 - R_j^2)

where R_j^2 is from regressing predictor j on all the others. 1 is no
overlap; 5 is the common point to look closer, 10 the common point of
concern.

With repeated sessions per mouse there are two kinds of overlap to tell
apart. Across all sessions, two predictors can correlate because some mice
sit high on both — a difference between animals. Within mice, they correlate
because on the days one is high in a given animal, so is the other.
:func:`center_within` removes each mouse's mean, which leaves the second.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def center_within(data: pd.DataFrame, columns: Sequence[str], group: pd.Series) -> pd.DataFrame:
    """Each column minus its group's mean: only the variation inside a group is left."""
    columns = list(columns)
    return data[columns] - data[columns].groupby(group).transform("mean")


def variance_inflation(data: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.Series:
    """The VIF of each column, given all the other columns.

    Each column is regressed on the rest with an intercept, by least squares.
    Rows with a missing value in any column are dropped first, so every VIF
    comes from the same rows. A column the others predict perfectly gets
    ``inf``.
    """
    columns = list(data.columns if columns is None else columns)
    assert len(columns) >= 2, "a VIF needs at least two predictors"
    x = data[columns].dropna().to_numpy(dtype=float)
    assert len(x) > len(columns), "fewer rows than predictors"
    out = {}
    for j, name in enumerate(columns):
        y = x[:, j]
        others = np.column_stack([np.ones(len(x)), np.delete(x, j, axis=1)])
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        resid = y - others @ beta
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan
        out[name] = np.inf if r2 >= 1 - 1e-12 else 1 / (1 - r2)
    return pd.Series(out, name="VIF")
