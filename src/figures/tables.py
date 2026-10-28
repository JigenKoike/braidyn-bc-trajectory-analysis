"""Reshape the tidy LMM tables into the shapes the manuscript prints.

:func:`analysis.lmm.lmm_tables` returns one row per pair × predictor, which
is what later analysis steps read. The manuscript instead prints one row per
pair (model fit) or per predictor (the other two), with the three pairs side
by side. Both are written out: the tidy ones as ``*_full.csv``.

Column order follows the manuscript's model formula, not the fitted order.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from analysis.lmm import PREDICTORS
from analysis.pairs import PAIR_ORDER

from .formatting import fmt_p, p_paren
from .palette import PAIR_LABEL, PRED_DISPLAY

#: Column order of the one-sheet table, as the manuscript prints it. Each q
#: value follows the p value it corrects, so the pair is read together.
WINDOW_LMM_COLUMNS: tuple[str, ...] = (
    "Predictor", "β", "95% CI", "Wald p", "Wald q", "ΔAIC", "LRT p", "LRT q",
)


def model_fit_table(
    model_fit: pd.DataFrame, pair_order: Sequence[str] = PAIR_ORDER
) -> pd.DataFrame:
    """One row per pair: n, AIC, both R², and which predictors stood out."""
    out = (
        model_fit.set_index("pair")
        .loc[list(pair_order), [
            "n_sessions", "AIC", "marginal_R2", "conditional_R2",
            "predictors_CI_excludes_0", "predictors_wald_q_lt_0.05",
        ]]
        .round({"AIC": 2, "marginal_R2": 3, "conditional_R2": 3})
    )
    out.index = [PAIR_LABEL[p] for p in out.index]
    out.index.name = "pair"
    return out


def leave_one_out_table(
    leave_one_out: pd.DataFrame,
    pair_order: Sequence[str] = PAIR_ORDER,
    predictors: Sequence[str] = PREDICTORS,
) -> pd.DataFrame:
    """One row per predictor, with ΔAIC and the LRT p and q of each pair."""
    i = leave_one_out.set_index(["pair", "predictor"])
    cols = {}
    for pair in pair_order:
        lab = PAIR_LABEL[pair]
        cols[f"{lab} ΔAIC"] = [f"{i.loc[(pair, p), 'delta_AIC']:+.2f}" for p in predictors]
        cols[f"{lab} LRT p"] = [fmt_p(i.loc[(pair, p), "lrt_p"]) for p in predictors]
        cols[f"{lab} LRT q"] = [fmt_p(i.loc[(pair, p), "lrt_q"]) for p in predictors]
    out = pd.DataFrame(cols, index=list(predictors))
    out.index.name = "predictor"
    return out


def coefficient_table(
    coefficients: pd.DataFrame,
    pair_order: Sequence[str] = PAIR_ORDER,
    predictors: Sequence[str] = PREDICTORS,
) -> pd.DataFrame:
    """Two rows per predictor: β with its Wald p and q, then the 95% CI.

    The manuscript's §2.6 asks whether a coefficient is positive with a CI
    that excludes zero, so the interval is printed next to the estimate. The
    intervals themselves are uncorrected; only the p values carry q values.
    """
    i = coefficients.set_index(["pair", "predictor"])
    rows = []
    for p in predictors:
        top = {"predictor": p, "value": "β (Wald p)"}
        bottom = {"predictor": "", "value": "95% CI"}
        for pair in pair_order:
            r = i.loc[(pair, p)]
            top[PAIR_LABEL[pair]] = (
                f"{r['beta']:+.5f} {p_paren(r['wald_p'])[:-1]}, q = {fmt_p(r['wald_q'])})"
            )
            bottom[PAIR_LABEL[pair]] = f"[{r['ci95_lo']:+.5f}, {r['ci95_hi']:+.5f}]"
        rows += [top, bottom]
    return pd.DataFrame(rows)


def window_lmm_table(
    model_fit: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    coefficients: pd.DataFrame,
    pair_order: Sequence[str] = PAIR_ORDER,
    predictors: Sequence[str] = PREDICTORS,
) -> pd.DataFrame:
    """One window's whole LMM result on a single sheet, as the manuscript prints it.

    Each pair is introduced by a row naming it and carrying its n and both
    R², and is followed by one row per predictor. The introducing row is
    blank in the other columns, so the sheet pastes straight into a document
    where that row is merged across the width.

    This is the shape new Tables 2 (early window) and 3 (late window) take.
    It is not :func:`compact_tables`, which builds the three separate tables
    the earlier draft printed.
    """
    fit = model_fit.set_index("pair")
    coef = coefficients.set_index(["pair", "predictor"])
    loo = leave_one_out.set_index(["pair", "predictor"])
    rows: list[dict[str, str]] = []
    for pair in pair_order:
        f = fit.loc[pair]
        heading = (
            f"{PAIR_LABEL[pair]}   n = {int(f['n_sessions'])} sessions, "
            f"{int(f['n_mice'])} mice   marginal R² = {f['marginal_R2']:.3f}, "
            f"conditional R² = {f['conditional_R2']:.3f}"
        )
        rows.append({**{c: "" for c in WINDOW_LMM_COLUMNS}, "Predictor": heading})
        for p in predictors:
            c, lo = coef.loc[(pair, p)], loo.loc[(pair, p)]
            rows.append({
                "Predictor": PRED_DISPLAY[p],
                "β": f"{c['beta']:+.5f}",
                "95% CI": f"[{c['ci95_lo']:+.5f}, {c['ci95_hi']:+.5f}]",
                "Wald p": fmt_p(c["wald_p"]),
                "Wald q": fmt_p(c["wald_q"]),
                "ΔAIC": f"{lo['delta_AIC']:+.2f}",
                "LRT p": fmt_p(lo["lrt_p"]),
                "LRT q": fmt_p(lo["lrt_q"]),
            })
    return pd.DataFrame(rows, columns=list(WINDOW_LMM_COLUMNS))


def window_lmm_table_full(
    model_fit: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    coefficients: pd.DataFrame,
    pair_order: Sequence[str] = PAIR_ORDER,
    predictors: Sequence[str] = PREDICTORS,
) -> pd.DataFrame:
    """The same numbers unformatted: one row per pair × predictor.

    The per-pair model fit is repeated down that pair's rows, so this one
    file answers everything :func:`window_lmm_table` shows without having to
    be read alongside another. Any label columns the fits carried — a
    ``window`` column, say — are kept and joined on.
    """
    on = [c for c in coefficients.columns if c in set(leave_one_out.columns)]
    assert "pair" in on and "predictor" in on, "the two tidy tables share no key"
    out = coefficients.merge(leave_one_out, on=on, validate="one_to_one")

    pair_keys = [c for c in on if c != "predictor"]
    fit_cols = ["n_sessions", "n_mice", "AIC", "marginal_R2", "conditional_R2"]
    out = out.merge(model_fit[pair_keys + fit_cols], on=pair_keys, validate="many_to_one")

    order = {(pair, p): i for i, (pair, p) in enumerate(
        (pair, p) for pair in pair_order for p in predictors)}
    out = (out.assign(_order=[order[(a, b)] for a, b in zip(out["pair"], out["predictor"])])
              .sort_values("_order")
              .drop(columns="_order")
              .reset_index(drop=True))
    return out


def compact_tables(
    model_fit: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    coefficients: pd.DataFrame,
    pair_order: Sequence[str] = PAIR_ORDER,
    predictors: Sequence[str] = PREDICTORS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """All three manuscript-shaped tables, in the order they are numbered."""
    return (
        model_fit_table(model_fit, pair_order),
        leave_one_out_table(leave_one_out, pair_order, predictors),
        coefficient_table(coefficients, pair_order, predictors),
    )
