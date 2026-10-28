"""Tests for the two ways of letting the mice differ more than the model does."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats

from analysis.correlation import per_mouse_slope
from analysis.lmm_sensitivity import (
    AGREEMENT_TOLERANCE,
    LRT_NEGATIVE_SLACK,
    bootstrap_interval,
    choose_agreed_fit,
    cluster_bootstrap,
    fit_random_slope,
    random_slope_lrt,
    resample_groups,
    variance_explained,
)


class _Fit:
    """Stands in for a fitted result; the choice rule reads llf and converged."""

    def __init__(self, llf: float, converged: bool = True):
        self.llf = llf
        self.converged = converged


def _panel(n_mice=20, n_sessions=12, slope=0.5, slope_sd=0.0, seed=0) -> pd.DataFrame:
    """Sessions of several mice, each with its own intercept and maybe its own slope."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_mice):
        intercept = rng.normal(0, 1.0)
        own_slope = slope + rng.normal(0, slope_sd)
        for day in range(1, n_sessions + 1):
            rows.append({"mouse": f"m{m}", "day": float(day),
                         "y": intercept + own_slope * day + rng.normal(0, 1.0)})
    return pd.DataFrame(rows)


def test_the_agreed_fit_is_the_best_one_two_optimizers_found() -> None:
    fits = {"lbfgs": _Fit(100.0), "bfgs": _Fit(100.0), "nm": _Fit(700.0)}
    res, agreeing = choose_agreed_fit(fits)
    # 700 is the highest and would win outright; only one optimizer saw it.
    assert res.llf == 100.0
    assert agreeing == ["bfgs", "lbfgs"]


def test_a_solution_found_twice_wins_over_a_lower_one_found_three_times() -> None:
    fits = {"a": _Fit(10.0), "b": _Fit(10.0), "c": _Fit(10.0),
            "d": _Fit(50.0), "e": _Fit(50.0)}
    assert choose_agreed_fit(fits)[0].llf == 50.0


def test_nothing_is_returned_when_no_two_optimizers_agree() -> None:
    assert choose_agreed_fit({"a": _Fit(1.0), "b": _Fit(2.0)}) == (None, [])
    assert choose_agreed_fit({}) == (None, [])


def test_an_unflagged_fit_still_counts_towards_agreement() -> None:
    # statsmodels clears the convergence flag on fits that two other
    # optimizers match to five decimals; the agreement is the better evidence.
    fits = {"lbfgs": _Fit(605.3303), "bfgs": _Fit(584.0178),
            "cg": _Fit(605.3374, converged=False), "nm": _Fit(605.3374, converged=False)}
    res, agreeing = choose_agreed_fit(fits)
    assert agreeing == ["cg", "lbfgs", "nm"]
    assert res.converged, "the flagged copy of the agreed solution should be preferred"


def test_the_agreed_solution_is_returned_even_if_none_of_it_is_flagged() -> None:
    fits = {"a": _Fit(10.0, converged=False), "b": _Fit(10.0, converged=False)}
    res, agreeing = choose_agreed_fit(fits)
    assert res is not None and agreeing == ["a", "b"]


def test_agreement_is_within_a_tolerance_not_exact() -> None:
    fits = {"a": _Fit(10.0), "b": _Fit(10.0 + AGREEMENT_TOLERANCE / 2)}
    assert choose_agreed_fit(fits)[0] is not None
    fits = {"a": _Fit(10.0), "b": _Fit(10.0 + AGREEMENT_TOLERANCE * 10)}
    assert choose_agreed_fit(fits)[0] is None


def test_the_random_slope_model_recovers_a_known_slope() -> None:
    data = _panel(slope=0.5, slope_sd=0.2, seed=1)
    res, agreeing, _ = fit_random_slope("y ~ day", data)
    assert len(agreeing) >= 2
    assert res.params["day"] == pytest.approx(0.5, abs=0.1)
    # Three parameters in the random part now: two variances and a covariance.
    assert res.cov_re.shape == (2, 2)


def test_the_random_slope_widens_the_interval_when_the_slopes_differ() -> None:
    # The whole point of the exercise: with slopes varying between mice, the
    # shared slope is less well determined than a common-slope model thinks.
    import statsmodels.formula.api as smf

    data = _panel(slope=0.5, slope_sd=0.35, n_mice=25, seed=2)
    intercept_only = smf.mixedlm("y ~ day", data, groups=data["mouse"]).fit(reml=False)
    with_slope, _, _ = fit_random_slope("y ~ day", data)
    assert with_slope.bse["day"] > intercept_only.bse["day"] * 2


def _panel_with_noise(seed=3, **kw) -> pd.DataFrame:
    """The panel plus a predictor that has nothing to do with the outcome."""
    data = _panel(**kw, seed=seed)
    data["noise"] = np.random.default_rng(seed + 100).normal(size=len(data))
    return data


def test_the_lrt_finds_a_real_predictor_and_not_a_null_one() -> None:
    data = _panel_with_noise(slope=0.5, slope_sd=0.2)
    _, table = random_slope_lrt(data, ["day", "noise"], outcome="y")
    t = table.set_index("predictor")
    assert t.loc["day", "lrt_p"] < 1e-6
    assert t.loc["noise", "lrt_p"] > 0.05
    assert (t["lrt_chi2"] > -LRT_NEGATIVE_SLACK).all()


def test_the_lrt_is_twice_the_log_likelihood_difference_of_the_two_fits() -> None:
    data = _panel_with_noise(slope=0.3, slope_sd=0.2)
    full, table = random_slope_lrt(data, ["day", "noise"], outcome="y")
    reduced, _, _ = fit_random_slope("y ~ noise", data)
    row = table.set_index("predictor").loc["day"]
    assert row["llf_full"] == pytest.approx(full.llf)
    assert row["lrt_chi2"] == pytest.approx(2 * (full.llf - reduced.llf), abs=1e-6)
    assert row["lrt_p"] == pytest.approx(stats.chi2.sf(row["lrt_chi2"], df=1))


def test_every_reduced_model_keeps_the_random_slope(monkeypatch) -> None:
    # Dropping day from the fixed part must leave each mouse its own slope:
    # the test is of the mean slope, not of the slopes' variance.
    import analysis.lmm_sensitivity as mod

    calls = []
    real = mod.fit_random_slope

    def spy(formula, data, group_col="mouse", slope_col="day", optimizers=mod.OPTIMIZERS):
        calls.append((formula, slope_col))
        return real(formula, data, group_col, slope_col, optimizers)

    monkeypatch.setattr(mod, "fit_random_slope", spy)
    data = _panel_with_noise(slope=0.3, slope_sd=0.2)
    random_slope_lrt(data, ["day", "noise"], outcome="y")
    assert calls == [("y ~ day + noise", "day"), ("y ~ noise", "day"), ("y ~ day", "day")]


def test_the_lrt_refuses_a_fit_no_two_optimizers_agree_on(monkeypatch) -> None:
    import analysis.lmm_sensitivity as mod

    monkeypatch.setattr(mod, "fit_random_slope",
                        lambda *a, **k: (None, [], {"lbfgs": 1.0, "nm": 700.0}))
    with pytest.raises(RuntimeError, match="no two optimizers agreed"):
        random_slope_lrt(_panel_with_noise(), ["day", "noise"], outcome="y")


def test_the_lrt_refuses_a_reduced_model_that_fits_better(monkeypatch) -> None:
    # Nested models cannot do that; if the numbers say so, a fit stopped short.
    import analysis.lmm_sensitivity as mod

    fits = iter([_Fit(10.0), _Fit(11.0)])
    monkeypatch.setattr(mod, "fit_random_slope",
                        lambda *a, **k: (next(fits), ["bfgs", "lbfgs"], {"bfgs": 0.0, "lbfgs": 0.0}))
    with pytest.raises(RuntimeError, match="fits better"):
        random_slope_lrt(_panel_with_noise(), ["day"], outcome="y")


def test_a_mouse_drawn_twice_becomes_two_mice() -> None:
    data = _panel(n_mice=4, n_sessions=3)
    rng = np.random.default_rng(0)
    drawn = resample_groups(data, "mouse", rng)
    assert len(drawn) == len(data)  # as many sessions as there were
    assert drawn["mouse"].nunique() == 4  # as many labels as mice drawn
    # Labels are new, so a mouse drawn twice cannot be read as one with more
    # sessions; every label carries exactly one mouse's worth.
    assert set(drawn.groupby("mouse").size()) == {3}
    assert not set(drawn["mouse"]) & set(data["mouse"])


def test_resampling_is_with_replacement() -> None:
    # Over many draws some mice come up more than once and others not at all;
    # a draw that never repeats would be a permutation, which is a different
    # thing entirely.
    data = _panel(n_mice=5, n_sessions=2)
    rng = np.random.default_rng(1)
    origins = [sorted(m.split("#")[0] for m in
                      resample_groups(data, "mouse", rng)["mouse"].unique())
               for _ in range(50)]
    assert any(len(set(o)) < 5 for o in origins), "no draw repeated a mouse"


def test_the_bootstrap_interval_lands_near_the_models_own() -> None:
    # With mice that really do share a slope, the two ways of measuring the
    # uncertainty should agree; the bootstrap earns its keep when they do not.
    import statsmodels.formula.api as smf

    data = _panel(n_mice=25, n_sessions=10, slope=0.5, slope_sd=0.0, seed=3)
    fitted = smf.mixedlm("y ~ day", data, groups=data["mouse"]).fit(reml=False)
    draws, failed = cluster_bootstrap("y ~ day", data, ["day"], n_draws=200, rng=0)
    assert failed == 0
    summary = bootstrap_interval(draws["day"])
    width = summary["ci95_hi"] - summary["ci95_lo"]
    assert width == pytest.approx(2 * 1.96 * fitted.bse["day"], rel=0.4)
    assert summary["ci95_lo"] < 0.5 < summary["ci95_hi"]


def test_the_bootstrap_sees_slope_differences_the_model_does_not() -> None:
    # Mice with different slopes: the common-slope model's standard error
    # says the shared slope is well determined, the bootstrap disagrees.
    import statsmodels.formula.api as smf

    data = _panel(n_mice=25, n_sessions=10, slope=0.5, slope_sd=0.4, seed=4)
    fitted = smf.mixedlm("y ~ day", data, groups=data["mouse"]).fit(reml=False)
    draws, _ = cluster_bootstrap("y ~ day", data, ["day"], n_draws=200, rng=0)
    summary = bootstrap_interval(draws["day"])
    width = summary["ci95_hi"] - summary["ci95_lo"]
    assert width > 2 * 1.96 * fitted.bse["day"] * 2


def test_a_seed_makes_the_draws_repeat() -> None:
    data = _panel(n_mice=8, n_sessions=5)
    first, _ = cluster_bootstrap("y ~ day", data, ["day"], n_draws=20, rng=5)
    again, _ = cluster_bootstrap("y ~ day", data, ["day"], n_draws=20, rng=5)
    pd.testing.assert_frame_equal(first, again)


def test_a_draw_that_cannot_be_fitted_is_counted_rather_than_replaced() -> None:
    # Every other draw refuses to be fitted; the ones that failed are counted
    # and the asked-for number of draws is not topped back up with easier ones.
    attempts = []

    def flaky(formula, d, g):
        attempts.append(None)
        return None if len(attempts) % 2 else _fit_stub()

    data = _panel(n_mice=6, n_sessions=4)
    draws, failed = cluster_bootstrap(
        "y ~ day", data, ["day"], n_draws=10, rng=0, fit=flaky)
    assert len(attempts) == 10
    assert failed == 5
    assert len(draws) == 5


def _fit_stub():
    class _R:
        params = {"day": 1.0}
    return _R()


def test_the_bootstrap_p_value_cannot_be_smaller_than_the_draws_allow() -> None:
    # Every draw on one side of zero means the p value is at the floor the
    # number of draws sets, not zero.
    summary = bootstrap_interval(np.full(500, 2.0))
    assert summary["p"] == pytest.approx(1 / 500)
    summary = bootstrap_interval(np.concatenate([np.full(400, 2.0), np.full(100, -2.0)]))
    assert summary["p"] == pytest.approx(0.4)


def test_per_mouse_slope_is_the_slope_inside_each_animal() -> None:
    sessions = pd.DataFrame({
        "mouse": ["a"] * 4 + ["b"] * 4 + ["c"] * 2,
        "day": [1, 2, 3, 4] * 2 + [1, 2],
        "value": [1, 2, 3, 4, 10, 8, 6, 4, 0, 1],
    })
    out = per_mouse_slope(sessions, "day", "value")
    assert list(out["mouse"]) == ["a", "b"]  # c is one session short of the floor
    assert out.loc[0, "slope"] == pytest.approx(1.0)
    assert out.loc[1, "slope"] == pytest.approx(-2.0)


def test_per_mouse_slope_drops_a_mouse_whose_x_does_not_move() -> None:
    sessions = pd.DataFrame({
        "mouse": ["a"] * 3 + ["b"] * 3,
        "day": [1, 2, 3, 5, 5, 5],
        "value": [1, 2, 3, 1, 2, 3],
    })
    assert list(per_mouse_slope(sessions, "day", "value")["mouse"]) == ["a"]


# --- variance explained -------------------------------------------------------

def _stand_in(x: np.ndarray, fe: list[float], cov: list[list[float]], scale: float,
              slope: bool) -> SimpleNamespace:
    """A fitted result reduced to what the R² reads: designs, estimates, variances."""
    exog = np.column_stack([np.ones_like(x), x])
    exog_re = exog if slope else exog[:, :1]
    return SimpleNamespace(model=SimpleNamespace(exog=exog, exog_re=exog_re),
                           fe_params=np.array(fe), cov_re=np.array(cov), scale=scale)


def test_with_an_intercept_alone_the_r2_is_nakagawa_and_schielzeth() -> None:
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    out = variance_explained(_stand_in(x, [0.3, 2.0], [[1.5]], 0.5, slope=False))
    var_fixed = np.var(2.0 * x, ddof=1)
    assert out["var_random"] == pytest.approx(1.5)
    assert out["marginal_R2"] == pytest.approx(var_fixed / (var_fixed + 1.5 + 0.5))
    assert out["conditional_R2"] == pytest.approx((var_fixed + 1.5) / (var_fixed + 1.5 + 0.5))


def test_a_random_slope_adds_its_variance_where_the_predictor_is() -> None:
    # Johnson (2014): the mean over rows of (1, x) Σ (1, x)'.
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    cov = [[1.0, 0.5], [0.5, 2.0]]
    out = variance_explained(_stand_in(x, [0.0, 1.0], cov, 1.0, slope=True))
    expected = 1.0 + 2 * 0.5 * x.mean() + 2.0 * (x**2).mean()
    assert out["var_random"] == pytest.approx(expected)
    # Where x is centred and of unit mean square, that is the two variances added.
    xc = np.array([-1.0, 1.0, -1.0, 1.0])
    centred = variance_explained(_stand_in(xc, [0.0, 1.0], cov, 1.0, slope=True))
    assert centred["var_random"] == pytest.approx(1.0 + 2.0)


def test_on_a_fitted_random_intercept_it_is_the_tables_own_formula() -> None:
    d = _panel(n_mice=15, n_sessions=8, seed=7)
    res = smf.mixedlm("y ~ day", d, groups=d["mouse"]).fit(reml=False)
    out = variance_explained(res)
    var_fixed = float(np.var(res.model.exog @ res.fe_params, ddof=1))
    var_mouse, var_resid = float(res.cov_re.iloc[0, 0]), float(res.scale)
    assert out["marginal_R2"] == pytest.approx(var_fixed / (var_fixed + var_mouse + var_resid))
    assert out["conditional_R2"] == pytest.approx(
        (var_fixed + var_mouse) / (var_fixed + var_mouse + var_resid))
