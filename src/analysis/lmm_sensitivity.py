"""Does the training-day effect hold when the mice are allowed to differ more?

The models in :mod:`analysis.lmm` give every mouse its own intercept and one
slope to share. That says the animals start at different levels but learn at
the same rate, and the standard error of the shared slope is computed as
though every session were an independent word on it. If the rates really
differ between mice, that variation has nowhere to go but the residual, and
the slope looks better determined than it is.

Two ways of checking, with different things to trust:

- :func:`fit_random_slope` widens the model — a slope per mouse as well as an
  intercept — and asks the same question of it. This is the direct answer,
  at the cost of two more variance parameters estimated from 25 animals.
  :func:`random_slope_lrt` asks it by likelihood ratio, the test the
  paper's figures and tables use for the random-intercept models.
- :func:`cluster_bootstrap` leaves the model alone and re-draws the animals:
  25 mice sampled with replacement from the 25 there are, each bringing all
  of its sessions, refitted a few thousand times. The spread of the
  coefficient over those draws is what "another 25 mice" would have done,
  and it reflects differences in slope without anything being said about
  them in the model.

They can disagree, and if they do that is the finding.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from .lmm import OPTIMIZERS

#: Two optimizers' log-likelihoods count as the same answer within this.
#: Optimizers stop at slightly different points on a flat summit — 0.015
#: apart on this data's widest case — so the tolerance has to clear that
#: while staying far below what it is there to catch: the degenerate
#: solution it refuses sits more than six hundred above its neighbours.
AGREEMENT_TOLERANCE = 0.1

#: How many optimizers have to reach a solution before it is believed.
MIN_AGREEING = 2

#: Bootstrap draws. Enough that the 2.5th and 97.5th percentiles are steady
#: to the digits reported.
DEFAULT_DRAWS = 2000


def choose_agreed_fit(
    fits: dict[str, Any],
    tol: float = AGREEMENT_TOLERANCE,
    min_agreeing: int = MIN_AGREEING,
) -> tuple[Any | None, list[str]]:
    """The best log-likelihood that at least ``min_agreeing`` optimizers reached.

    ``fits`` maps an optimizer's name to its fitted result. Taking the
    highest log-likelihood outright — what :func:`analysis.lmm.fit_best_mixedlm`
    does, safely, for a single random intercept — stops being safe once the
    random-effect covariance has three parameters: on this data Nelder-Mead
    reports convergence at a log-likelihood more than six hundred above every
    other optimizer, with a p value of exactly 1, which is a degenerate
    solution rather than a better one. Requiring a second optimizer to find
    the same place rules that out without a search over three dimensions.

    **Every result is considered, converged or not.** The convergence flag is
    not trustworthy for these models in either direction: it is set on fits
    that stopped short (the reason :mod:`analysis.lmm` grid-checks its own),
    and cleared on fits that two other optimizers agree with to five decimal
    places. Two independent optimizers reaching the same summit is the better
    evidence. Among the ones that agree, a fit whose flag is set is preferred,
    so the caller gets the better-behaved copy of the same solution.

    Returns ``(result, agreeing_methods)``, or ``(None, [])`` if no solution
    was found twice.
    """
    if not fits:
        return None, []
    ordered = sorted(fits.items(), key=lambda kv: -kv[1].llf)
    for _, res in ordered:
        agreeing = {m: r for m, r in fits.items() if abs(r.llf - res.llf) <= tol}
        if len(agreeing) < min_agreeing:
            continue
        converged = [r for r in agreeing.values() if getattr(r, "converged", True)]
        return (converged[0] if converged else res), sorted(agreeing)
    return None, []


def fit_random_slope(
    formula: str,
    data: pd.DataFrame,
    group_col: str = "mouse",
    slope_col: str = "day",
    optimizers: Iterable[str] = OPTIMIZERS,
) -> tuple[Any | None, list[str], dict[str, float]]:
    """Fit ``formula`` with a random intercept *and* a random slope on ``slope_col``.

    The fixed effects, the standardization and the estimation method are
    whatever the caller already uses; the only change is the random part,
    which gains the variance of the per-mouse slope and its covariance with
    the intercept. The fit is chosen by :func:`choose_agreed_fit`.

    Returns ``(result, agreeing_methods, llf_by_method)``.
    """
    model = smf.mixedlm(formula, data, groups=data[group_col],
                        re_formula=f"~{slope_col}")
    fits: dict[str, Any] = {}
    llf_by_method: dict[str, float] = {}
    for method in optimizers:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = model.fit(reml=False, method=method)
            except Exception:  # noqa: BLE001 — a failed optimizer is just one fewer
                continue
        llf_by_method[method] = float(res.llf)
        fits[method] = res
    result, agreeing = choose_agreed_fit(fits)
    return result, agreeing, llf_by_method


#: How far below zero a likelihood-ratio χ² may fall before it is taken as a
#: fit gone wrong rather than the optimizers' resolution. A nested model
#: cannot fit better than the full one; the agreeing optimizers differ by up
#: to 0.015 in log-likelihood on this data, so a χ² of −0.05 is already
#: beyond what two honest fits produce.
LRT_NEGATIVE_SLACK = 0.05


def _agreed_random_slope(
    formula: str, data: pd.DataFrame, group_col: str, slope_col: str,
    optimizers: Iterable[str], what: str,
) -> tuple[Any, float]:
    """:func:`fit_random_slope`, raising if no two optimizers agree; with their spread."""
    res, agreeing, llfs = fit_random_slope(formula, data, group_col, slope_col, optimizers)
    if res is None:
        raise RuntimeError(f"{what}: no two optimizers agreed ({llfs})")
    agreed = [llfs[m] for m in agreeing]
    return res, float(max(agreed) - min(agreed))


def random_slope_lrt(
    data: pd.DataFrame,
    predictors: Sequence[str],
    *,
    outcome: str = "auc",
    group_col: str = "mouse",
    slope_col: str = "day",
    optimizers: Iterable[str] = OPTIMIZERS,
) -> tuple[Any, pd.DataFrame]:
    """The random-slope model, and a likelihood-ratio test of each predictor in it.

    The full model is ``outcome ~ predictors`` with an intercept and a
    ``slope_col`` slope per group, as :func:`fit_random_slope` fits it. Each
    predictor is then dropped from the fixed part in turn, and **the random
    part stays** in every reduced model. Dropping ``slope_col`` itself
    therefore asks whether the *mean* of the groups' slopes is zero, each
    group still free to have its own: one parameter, away from any boundary,
    so χ² with one degree of freedom applies, as it does for the
    random-intercept models in :func:`analysis.lmm.lmm_tables`. Dropping the
    random slope as well would test a variance on the edge of its range, a
    different question with a different null distribution.

    Every fit is chosen by :func:`choose_agreed_fit`, so a log-likelihood is
    known only as well as the optimizers that agreed on it agree with one
    another. That range is reported for each fit (``llf_spread_full``,
    ``llf_spread_reduced``); the χ² is good to about twice their sum. There is
    no grid check here as there is for the random intercept: the random part
    has three parameters, not one.

    Rows missing a predictor or the outcome are dropped first, as
    :func:`analysis.lmm.fit_leave_one_out` drops them.

    Returns ``(full_result, table)``: the table has one row per predictor,
    with ``llf_full``, ``llf_reduced``, ``lrt_chi2``, ``lrt_p`` and the two
    spreads. It is not corrected for multiple comparisons; the caller chooses
    the family. Raises if a fit has no two optimizers in agreement, or if a
    reduced model fits better than the full one by more than
    :data:`LRT_NEGATIVE_SLACK`.
    """
    predictors = list(predictors)
    d = data.dropna(subset=predictors + [outcome]).reset_index(drop=True)
    full, spread_full = _agreed_random_slope(
        f"{outcome} ~ " + " + ".join(predictors), d, group_col, slope_col, optimizers, "full model")
    rows = []
    for p in predictors:
        reduced, spread_reduced = _agreed_random_slope(
            f"{outcome} ~ " + " + ".join(q for q in predictors if q != p), d, group_col,
            slope_col, optimizers, f"without {p}")
        chi2 = 2 * (full.llf - reduced.llf)
        if chi2 < -LRT_NEGATIVE_SLACK:
            raise RuntimeError(f"without {p}: the reduced model fits better (χ² = {chi2:.3f}); "
                               "one of the two fits stopped short")
        rows.append({"predictor": p, "llf_full": float(full.llf),
                     "llf_reduced": float(reduced.llf), "lrt_chi2": float(chi2),
                     "lrt_p": float(stats.chi2.sf(max(chi2, 0.0), df=1)),
                     "llf_spread_full": spread_full, "llf_spread_reduced": spread_reduced})
    return full, pd.DataFrame(rows)


def variance_explained(res: Any) -> dict[str, float]:
    """Marginal and conditional R² of a mixed model, with or without a random slope.

    The marginal R² is the share of the total variance the fixed effects
    carry; the conditional R² adds the random effects. With a random
    intercept alone this is Nakagawa & Schielzeth (2013), exactly as
    :func:`analysis.lmm.lmm_tables` computes it. A random slope makes the
    random effects' variance depend on the predictor it multiplies, so it is
    averaged over the observations — Johnson (2014):

        var_random = mean over rows i of  z_i' Σ z_i

    where z_i is the row's random-effects design — (1, day) here — and Σ
    their covariance. With an intercept alone z_i = (1) and this is Σ itself,
    so the two cases are one formula.

    Returns the three variances and the two R², in the fit's outcome units.
    """
    var_fixed = float(np.var(res.model.exog @ np.asarray(res.fe_params), ddof=1))
    z = np.asarray(res.model.exog_re, dtype=float)
    cov = np.asarray(res.cov_re, dtype=float)
    var_random = float(np.mean(np.einsum("ij,jk,ik->i", z, cov, z)))
    var_resid = float(res.scale)
    total = var_fixed + var_random + var_resid
    return {"var_fixed": var_fixed, "var_random": var_random, "var_residual": var_resid,
            "marginal_R2": var_fixed / total,
            "conditional_R2": (var_fixed + var_random) / total}


def resample_groups(
    data: pd.DataFrame, group_col: str, rng: np.random.Generator
) -> pd.DataFrame:
    """Draw as many groups as there are, with replacement, and relabel them.

    A group drawn twice has to become two groups: left under one label the
    model would see a single mouse with twice the sessions, which is the
    opposite of what the draw is meant to represent.
    """
    groups = data[group_col].unique()
    drawn = rng.choice(groups, size=len(groups), replace=True)
    parts = []
    for i, group in enumerate(drawn):
        part = data[data[group_col] == group].copy()
        part[group_col] = f"{group}#{i}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _fit_once(formula: str, data: pd.DataFrame, group_col: str) -> Any | None:
    """One fit for a bootstrap draw: the quick optimizer, then the others.

    The draws are thousands, so trying every optimizer on each is wasted
    work; lbfgs answers almost always, and the rest are there for when it
    does not converge.
    """
    model = smf.mixedlm(formula, data, groups=data[group_col])
    for method in ("lbfgs", "bfgs", "powell"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = model.fit(reml=False, method=method)
            except Exception:  # noqa: BLE001
                continue
        if res.converged:
            return res
    return None


def cluster_bootstrap(
    formula: str,
    data: pd.DataFrame,
    params: Sequence[str],
    *,
    group_col: str = "mouse",
    n_draws: int = DEFAULT_DRAWS,
    rng: np.random.Generator | int | None = None,
    fit: Callable[[str, pd.DataFrame, str], Any | None] = _fit_once,
) -> tuple[pd.DataFrame, int]:
    """Refit ``formula`` on mice resampled with replacement, ``n_draws`` times.

    Returns ``(draws, n_failed)``: one row per successful draw holding each
    name in ``params``, and how many draws no optimizer converged on. A draw
    that fails is dropped rather than retried — it is one resampled set of
    animals the model could not be fitted to, and replacing it with another
    would quietly select the agreeable ones.
    """
    rng = np.random.default_rng(rng)
    rows, failed = [], 0
    for _ in range(n_draws):
        res = fit(formula, resample_groups(data, group_col, rng), group_col)
        if res is None:
            failed += 1
            continue
        rows.append({name: float(res.params[name]) for name in params})
    return pd.DataFrame(rows, columns=list(params)), failed


def bootstrap_interval(draws: Sequence[float], alpha: float = 0.05) -> dict[str, float]:
    """Percentile interval and a two-sided p value from bootstrap draws.

    The p value is the usual bootstrap one: twice the smaller of the two
    tails on either side of zero, floored at what the number of draws can
    resolve, since no finite set of draws can say a p is zero.
    """
    draws = np.asarray(draws, dtype=float)
    assert draws.size > 0, "no draws to summarize"
    below = float(np.mean(draws < 0))
    p = 2 * min(below, 1 - below)
    return {
        "n_draws": int(draws.size),
        "estimate_median": float(np.median(draws)),
        "ci95_lo": float(np.quantile(draws, alpha / 2)),
        "ci95_hi": float(np.quantile(draws, 1 - alpha / 2)),
        "p": max(p, 1.0 / draws.size),
    }
