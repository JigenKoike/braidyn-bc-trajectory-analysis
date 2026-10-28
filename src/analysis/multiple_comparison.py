"""Multiple-comparison corrections, with the families this project uses.

Thin wrappers over :mod:`statsmodels.stats.multitest`, kept here so that
every figure and table applies the same correction the same way, and so that
the choice of family is stated once rather than in each notebook.

Families used in the paper (agreed with a co-author, 2026-10-05):

- Descriptive regressions of window AUC on training day: **Benjamini-Hochberg**,
  over the three outcome pairs within one window, reported as q. Holm until
  2026-10-23, when a co-author asked for the correction the mixed models use;
  the Holm p is still computed beside it.
- Linear mixed-effects models: **Benjamini-Hochberg**, over the predictors of
  one pair's model. The three pairs are separate models, so they are separate
  families, and the likelihood-ratio and Wald tests are corrected separately.

Benjamini-Hochberg thresholds depend on the whole family's p-value
distribution, not just its size, so the family is chosen by what is reported
as one claim — never by which grouping gives the most significant results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from statsmodels.stats.multitest import multipletests


def holm(p_values: ArrayLike) -> NDArray[np.float64]:
    """Holm-corrected p values, controlling the family-wise error rate."""
    return multipletests(np.asarray(p_values, dtype=float), method="holm")[1]


def benjamini_hochberg(p_values: ArrayLike) -> NDArray[np.float64]:
    """Benjamini-Hochberg q values, controlling the false discovery rate."""
    return multipletests(np.asarray(p_values, dtype=float), method="fdr_bh")[1]


def add_corrected_column(
    table: pd.DataFrame,
    p_column: str,
    corrected_column: str,
    *,
    within: str | list[str] | None = None,
    method: str = "fdr_bh",
) -> pd.DataFrame:
    """Insert a corrected-p column just after ``p_column``, modifying ``table``.

    ``within`` names the column(s) that delimit each family; leaving it out
    treats the whole table as one family. ``method`` is ``"fdr_bh"`` for q
    values or ``"holm"`` for family-wise corrected p values.
    """
    correct = {"fdr_bh": benjamini_hochberg, "holm": holm}[method]
    values = (
        table.groupby(within)[p_column].transform(lambda s: correct(s.to_numpy()))
        if within is not None
        else pd.Series(correct(table[p_column].to_numpy()), index=table.index)
    )
    table.insert(table.columns.get_loc(p_column) + 1, corrected_column, values)
    return table
