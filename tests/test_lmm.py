"""Tests for `analysis.lmm`, on small synthetic data with known structure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.lmm import (
    PREDICTORS,
    PROFILE_TOL,
    fit_best_mixedlm,
    fit_leave_one_out,
    lmm_tables,
    profile_max_llf,
)

BETA_X, BETA_Z, MOUSE_SD, NOISE_SD = 2.0, 0.0, 1.5, 1.0


def synthetic(n_mice: int = 12, n_sessions: int = 15, seed: int = 0) -> pd.DataFrame:
    """y driven by x, unrelated to z, with a per-mouse intercept."""
    rng = np.random.default_rng(seed)
    mice = [f"m{i:02d}" for i in range(n_mice)]
    intercept = dict(zip(mice, rng.normal(0, MOUSE_SD, n_mice)))
    rows = []
    for m in mice:
        for _ in range(n_sessions):
            x, z = rng.normal(), rng.normal()
            rows.append({
                "mouse": m, "x": x, "z": z,
                "y": intercept[m] + BETA_X * x + BETA_Z * z + rng.normal(0, NOISE_SD),
            })
    return pd.DataFrame(rows)


def test_fit_recovers_the_fixed_effects() -> None:
    res, method, failed, _, llfs = fit_best_mixedlm("y ~ x + z", synthetic())
    assert res is not None and method is not None
    assert res.params["x"] == pytest.approx(BETA_X, abs=0.15)
    assert res.params["z"] == pytest.approx(BETA_Z, abs=0.15)
    assert llfs, "at least one optimizer should have converged"


def test_the_chosen_fit_has_the_highest_log_likelihood_of_those_that_converged() -> None:
    res, method, _, _, llfs = fit_best_mixedlm("y ~ x + z", synthetic())
    assert llfs[method] == max(llfs.values())
    assert res.llf == pytest.approx(llfs[method])


def test_an_unfittable_formula_is_reported_rather_than_raising() -> None:
    d = synthetic(n_mice=3, n_sessions=2)
    d["constant"] = 1.0  # collinear with the intercept
    res, method, failed, _, _ = fit_best_mixedlm("y ~ constant", d)
    if res is None:
        assert failed, "a failure should say which optimizers were tried"
    else:
        assert method is not None


def test_the_fit_sits_at_the_profile_likelihood_maximum() -> None:
    # The point of the grid check: a fit below the grid maximum means the
    # optimizer stopped short, which L-BFGS has done on this project's data.
    res, _, _, _, _ = fit_best_mixedlm("y ~ x + z", synthetic())
    grid_max, grid_argmax, s_fit = profile_max_llf(res, grid_n=200)
    assert res.llf >= grid_max - PROFILE_TOL
    assert s_fit == pytest.approx(grid_argmax, rel=0.1)


def test_leave_one_out_fits_the_full_model_and_one_model_per_predictor() -> None:
    log: list[dict] = []
    d, scale, fits = fit_leave_one_out(
        synthetic(), ["x", "z"], outcome="y", convergence_log=log, label={"pair": "demo"})
    assert set(fits) == {"full", "x", "z"}
    assert scale == 1.0
    assert len(d) == len(synthetic())
    assert [r["model"] for r in log] == ["full", "without x", "without z"]
    assert all(r["pair"] == "demo" and r["converged"] for r in log)


def test_leave_one_out_drops_rows_missing_a_predictor() -> None:
    d = synthetic()
    d.loc[:4, "z"] = np.nan
    used, _, _ = fit_leave_one_out(d, ["x", "z"], outcome="y")
    assert len(used) == len(d) - 5


def test_dropping_the_predictor_that_matters_costs_likelihood() -> None:
    _, _, fits = fit_leave_one_out(synthetic(), ["x", "z"], outcome="y")
    assert fits["full"].llf - fits["x"].llf > fits["full"].llf - fits["z"].llf
    assert fits["x"].aic > fits["full"].aic  # dropping x makes the model worse


def _fitted_three_pairs() -> dict:
    out = {}
    for i, pair in enumerate(["success-miss", "failure-miss", "success-failure"]):
        d, scale, fits = fit_leave_one_out(
            synthetic(seed=i), ["x", "z"], outcome="y")
        out[pair] = dict(data=d, scale=scale, fits=fits)
    return out


def test_tables_report_each_pair_and_each_predictor() -> None:
    fitted = _fitted_three_pairs()
    t2, t3, t4 = lmm_tables(fitted, predictors=["x", "z"])
    assert len(t2) == 3
    assert len(t3) == len(t4) == 6
    assert set(t3.columns) >= {"pair", "predictor", "delta_AIC", "lrt_p", "lrt_q"}
    assert set(t4.columns) >= {"beta", "se", "ci95_lo", "ci95_hi", "wald_p", "wald_q"}


def test_the_corrected_column_follows_its_p_column() -> None:
    t2, t3, t4 = lmm_tables(_fitted_three_pairs(), predictors=["x", "z"])
    assert list(t3.columns).index("lrt_q") == list(t3.columns).index("lrt_p") + 1
    assert list(t4.columns).index("wald_q") == list(t4.columns).index("wald_p") + 1
    assert (t3["lrt_q"] >= t3["lrt_p"] - 1e-12).all()


def test_r_squared_is_a_proportion_and_marginal_never_exceeds_conditional() -> None:
    t2, _, _ = lmm_tables(_fitted_three_pairs(), predictors=["x", "z"])
    assert ((t2["marginal_R2"] > 0) & (t2["marginal_R2"] < 1)).all()
    assert (t2["marginal_R2"] <= t2["conditional_R2"]).all()
    # Nakagawa & Schielzeth: the three variances are the whole of it.
    total = t2["var_fixed"] + t2["var_mouse"] + t2["var_residual"]
    np.testing.assert_allclose(t2["conditional_R2"],
                               (t2["var_fixed"] + t2["var_mouse"]) / total)


def test_a_label_becomes_the_leading_columns() -> None:
    t2, t3, t4 = lmm_tables(_fitted_three_pairs(), predictors=["x", "z"],
                            label={"window": "late"})
    for t in (t2, t3, t4):
        assert list(t.columns)[:2] == ["window", "pair"]
        assert (t["window"] == "late").all()


def test_the_raw_scale_of_each_predictor_is_carried_into_the_coefficients() -> None:
    _, _, t4 = lmm_tables(_fitted_three_pairs(), predictors=["x", "z"],
                          predictor_mean={"x": 1.0, "z": 2.0},
                          predictor_sd={"x": 3.0, "z": 4.0})
    assert set(t4.loc[t4["predictor"] == "x", "predictor_mean_raw"]) == {1.0}
    assert set(t4.loc[t4["predictor"] == "z", "predictor_sd_raw"]) == {4.0}


def test_the_manuscript_predictor_order_is_the_formula_order() -> None:
    assert PREDICTORS[0] == "day"
    assert PREDICTORS[-1] == "T_pull_final"
    assert len(set(PREDICTORS)) == len(PREDICTORS) == 7
