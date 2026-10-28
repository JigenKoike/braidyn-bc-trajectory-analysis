"""Linear mixed-effects models of window AUC on session-level predictors.

One model per outcome pair: the window AUC of the PC1-PC3 distance explained
by the session predictors, with a random intercept per mouse, fitted by
maximum likelihood. Each predictor is also dropped in turn, so the tables can
report how much each one contributes (leave-one-out ΔAIC and a
likelihood-ratio test).

Two things here are deliberate and easy to get wrong if reimplemented:

**Fitting.** L-BFGS can report convergence while stopping short of the
maximum — on this data it sometimes halts after three iterations partway up
the likelihood surface. So every optimizer is tried and the converged fit
with the highest log-likelihood wins, and :func:`profile_max_llf` then checks
that fit against a direct grid search over the one parameter the optimizers
search. A fit below the grid maximum raises rather than passing quietly.

**Scaling.** The AUC values are small enough that the random-effect variance
sits near the optimizer's boundary. :data:`OUTCOME_SCALES` gives a fallback:
if any model fails to converge on the raw values, every model for that pair
is refitted on a rescaled outcome and the reported quantities are converted
back, so the scale never leaks into the results. (In practice the raw scale
has always worked; the fallback exists so a failure is handled rather than
silently reported.)
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from .multiple_comparison import add_corrected_column
from .pairs import PAIR_ORDER

#: Session-level predictors, in the order of the manuscript's model formula.
PREDICTORS: tuple[str, ...] = (
    "day",
    "pull_dur_first_median",
    "response_rate",
    "success_rate_v1",
    "F_blue_pre",
    "F_UV_pre",
    "T_pull_final",
)

#: Every optimizer is tried; see the module docstring.
OPTIMIZERS: tuple[str, ...] = ("lbfgs", "bfgs", "cg", "powell", "nm")

#: Outcome scales to try, in order. The raw AUC first.
OUTCOME_SCALES: tuple[float, ...] = (1.0, 100.0)

#: Grid points for the profile-likelihood check of each fit.
PROFILE_GRID_N = 1000

#: A fit may fall below the grid maximum by at most this much log-likelihood.
PROFILE_TOL = 1e-3


def fit_best_mixedlm(
    formula: str,
    data: pd.DataFrame,
    group_col: str = "mouse",
    optimizers: Iterable[str] = OPTIMIZERS,
) -> tuple[Any, str | None, list[str], list[str], dict[str, float]]:
    """Fit with every optimizer and keep the converged fit with the highest log-likelihood.

    Returns ``(result, method, failed_attempts, warning_messages, llf_by_method)``,
    with ``result`` and ``method`` ``None`` if nothing converged.
    """
    model = smf.mixedlm(formula, data, groups=data[group_col])
    fits: dict[str, tuple[Any, list[str]]] = {}
    failed: list[str] = []
    for method in optimizers:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                res = model.fit(reml=False, method=method)
            except Exception as e:  # noqa: BLE001 — any optimizer failure is just a failed attempt
                failed.append(f"{method}: {type(e).__name__}")
                continue
        if not res.converged:
            failed.append(f"{method}: not converged")
            continue
        fits[method] = (res, sorted({str(w.message).split(",")[0].rstrip(".") for w in caught}))
    llf_by_method = {m: r.llf for m, (r, _) in fits.items()}
    if not fits:
        return None, None, failed, [], llf_by_method
    best = max(fits, key=lambda m: fits[m][0].llf)
    return fits[best][0], best, failed, fits[best][1], llf_by_method


def profile_max_llf(res: Any, grid_n: int = PROFILE_GRID_N) -> tuple[float, float, float]:
    """Grid-search the log-likelihood over the one parameter the optimizers search.

    That parameter is sqrt(var_mouse / var_resid); the fixed effects and the
    residual variance are profiled out. Returns ``(grid_max_llf,
    grid_argmax, sqrt_ratio_of_the_fit)``.
    """
    model = res.model
    assert model.k_re2 == 1 and model.k_vc == 0, "profile check assumes a single random intercept"
    model.reml = False
    s_fit = float(np.sqrt(res.cov_re.iloc[0, 0] / res.scale))
    grid = np.linspace(0.0, max(2.0, 3 * s_fit), grid_n + 1)[1:]  # s = 0 is excluded (not finite)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ll = np.array([model.loglike(np.array([s]), profile_fe=True) for s in grid])
    ok = np.isfinite(ll)
    k = int(np.argmax(np.where(ok, ll, -np.inf)))
    return float(ll[k]), float(grid[k]), s_fit


def fit_leave_one_out(
    data: pd.DataFrame,
    predictors: Sequence[str] = PREDICTORS,
    *,
    outcome: str = "auc",
    group_col: str = "mouse",
    scales: Sequence[float] = OUTCOME_SCALES,
    convergence_log: list[dict] | None = None,
    label: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, float, dict[str, Any]]:
    """Fit the full model and each model with one predictor left out.

    ``data`` holds one row per session with the predictors, the ``outcome``
    column and ``group_col``; rows missing any of them are dropped. ``label``
    identifies this fit (e.g. ``{"window": "early", "pair": "success-miss"}``)
    in the convergence log and in error messages.

    Returns ``(data_used, outcome_scale_used, fits)``, where ``fits`` maps
    ``"full"`` and each dropped predictor's name to its fitted result.
    """
    predictors = list(predictors)
    label = dict(label) if label is not None else {}
    where = " / ".join(str(v) for v in label.values())
    d = data.dropna(subset=predictors + [outcome]).reset_index(drop=True)
    specs = [("full", predictors)] + [(p, [q for q in predictors if q != p]) for p in predictors]

    for scale in scales:
        dd = d.copy()
        dd["y"] = dd[outcome] * scale
        fits: dict[str, Any] = {}
        all_ok = True
        for name, preds in specs:
            res, method, failed, msgs, llfs = fit_best_mixedlm(
                "y ~ " + " + ".join(preds), dd, group_col
            )
            row = {
                **label,
                "outcome_scale": scale,
                "model": "full" if name == "full" else f"without {name}",
                "converged": res is not None,
                "optimizer_used": method,
                "llf_by_optimizer": "; ".join(f"{m}={v:.4f}" for m, v in llfs.items()),
                "profile_grid_max_llf": np.nan,
                "profile_grid_argmax": np.nan,
                "fit_sqrt_ratio": np.nan,
                "failed_attempts": "; ".join(failed),
                "warnings": "; ".join(msgs),
            }
            if res is None:
                all_ok = False
            else:
                gmax, garg, s_fit = profile_max_llf(res)
                row.update(
                    profile_grid_max_llf=gmax, profile_grid_argmax=garg, fit_sqrt_ratio=s_fit
                )
                assert res.llf >= gmax - PROFILE_TOL, (
                    f"{where} / {row['model']}: best fit llf {res.llf:.4f} "
                    f"is below the grid maximum {gmax:.4f}"
                )
                fits[name] = res
            if convergence_log is not None:
                convergence_log.append(row)
        if all_ok:
            return d, scale, fits
    raise RuntimeError(f"{where}: some models did not converge at any outcome scale")


def lmm_tables(
    fitted: Mapping[str, Mapping[str, Any]],
    *,
    predictors: Sequence[str] = PREDICTORS,
    pair_order: Sequence[str] = PAIR_ORDER,
    predictor_mean: Mapping[str, float] | None = None,
    predictor_sd: Mapping[str, float] | None = None,
    label: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the three tidy result tables from the fits of every pair.

    ``fitted`` maps each pair to ``{"data": ..., "scale": ..., "fits": ...}``,
    as returned by :func:`fit_leave_one_out`. ``predictor_mean`` and
    ``predictor_sd`` are the raw-unit mean and SD each predictor was
    standardized by, recorded alongside the coefficients.

    Returns ``(model_fit, leave_one_out, coefficients)``:

    - **model_fit** — one row per pair: n, AIC, variance components and the
      Nakagawa & Schielzeth marginal / conditional R². The marginal variance
      is that of the fixed-effect prediction Xβ; statsmodels' ``fittedvalues``
      includes the mouse intercepts and is not used.
    - **leave_one_out** — one row per pair × predictor: ΔAIC and the
      likelihood-ratio test of dropping it, with BH q values.
    - **coefficients** — one row per pair × predictor: β, SE, 95% CI and the
      Wald test, with BH q values.

    Everything is reported in the original AUC units, undoing the outcome
    scale the fit used.
    """
    predictors = list(predictors)
    label = dict(label) if label is not None else {}
    # Not `or {}`: these arrive as pandas Series, whose truth value raises.
    predictor_mean = {} if predictor_mean is None else dict(predictor_mean)
    predictor_sd = {} if predictor_sd is None else dict(predictor_sd)
    t2_rows, t3_rows, t4_rows = [], [], []

    for pair in pair_order:
        d, c, fits = fitted[pair]["data"], fitted[pair]["scale"], fitted[pair]["fits"]
        full = fits["full"]
        n = len(d)
        llf = full.llf + n * np.log(c)  # log-likelihood in original AUC units
        aic = full.aic - 2 * n * np.log(c)
        var_fixed = float(np.var(full.model.exog @ full.fe_params, ddof=1)) / c**2
        var_mouse = float(full.cov_re.iloc[0, 0]) / c**2
        var_resid = float(full.scale) / c**2
        total = var_fixed + var_mouse + var_resid

        ci = full.conf_int()
        sig_preds = []
        for p in predictors:
            lo, hi = ci.loc[p, 0] / c, ci.loc[p, 1] / c
            if lo > 0 or hi < 0:
                sig_preds.append(p)
            t4_rows.append({
                **label, "pair": pair, "predictor": p,
                "beta": full.params[p] / c, "se": full.bse[p] / c,
                "ci95_lo": lo, "ci95_hi": hi,
                "z": full.tvalues[p], "wald_p": full.pvalues[p],
                "predictor_mean_raw": predictor_mean.get(p, np.nan),
                "predictor_sd_raw": predictor_sd.get(p, np.nan),
            })

        t2_rows.append({
            **label, "pair": pair, "n_sessions": n, "n_mice": d["mouse"].nunique(),
            "AIC": aic, "logLik": llf,
            "var_fixed": var_fixed, "var_mouse": var_mouse, "var_residual": var_resid,
            "marginal_R2": var_fixed / total,
            "conditional_R2": (var_fixed + var_mouse) / total,
            "outcome_scale_used": c,
            "predictors_CI_excludes_0": ", ".join(sig_preds),
        })

        for p in predictors:
            red = fits[p]
            chi2 = 2 * (full.llf - red.llf)
            t3_rows.append({
                **label, "pair": pair, "predictor": p,
                "delta_AIC": red.aic - full.aic,
                "lrt_chi2": chi2,
                "lrt_p": float(stats.chi2.sf(max(chi2, 0.0), df=1)),
            })

    t2, t3, t4 = pd.DataFrame(t2_rows), pd.DataFrame(t3_rows), pd.DataFrame(t4_rows)
    assert (t3["lrt_chi2"] > -1e-6).all(), "a reduced model has higher likelihood than the full model"

    # One BH family per pair: that model's predictors. The three pairs are
    # separate models, and the LRT and Wald tests are corrected separately.
    add_corrected_column(t3, "lrt_p", "lrt_q", within="pair", method="fdr_bh")
    add_corrected_column(t4, "wald_p", "wald_q", within="pair", method="fdr_bh")
    q_by_pair = t4[t4["wald_q"] < 0.05].groupby("pair")["predictor"].apply(lambda s: ", ".join(s))
    t2["predictors_wald_q_lt_0.05"] = t2["pair"].map(q_by_pair).fillna("")
    return t2, t3, t4
