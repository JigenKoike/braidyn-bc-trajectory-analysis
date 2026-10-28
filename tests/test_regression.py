"""Tests for `analysis.regression`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.multiple_comparison import benjamini_hochberg, holm
from analysis.pairs import PAIR_ORDER
from analysis.regression import (
    auc_per_session,
    ci_excludes_zero,
    day_regression,
    fitted_line,
    window_mean_per_session,
)
from analysis.windows import EARLY_WINDOW, LATE_WINDOW

TRUE_SLOPE = 0.0025


def distance_long(level_by_session: dict[str, float]) -> pd.DataFrame:
    """A flat timecourse per session, so its AUC and mean are known exactly."""
    t = np.arange(-30, 121) / 30.0
    rows = []
    for i, (sid, level) in enumerate(level_by_session.items()):
        for tv in t:
            rows.append({"pair": "success-miss", "mouse": f"m{i % 3}", "day": i + 1,
                         "session_id": sid, "t": tv, "dist": level})
    return pd.DataFrame(rows)


def test_auc_of_a_flat_timecourse_is_level_times_window_width() -> None:
    d = auc_per_session(distance_long({"s1": 0.1, "s2": 0.2}), EARLY_WINDOW)
    assert list(d.columns) == ["pair", "mouse", "day", "session_id", "auc"]
    np.testing.assert_allclose(sorted(d["auc"]), [0.1 * 1.8, 0.2 * 1.8])


def test_the_late_window_is_wider_so_the_same_level_gives_a_larger_auc() -> None:
    long = distance_long({"s1": 0.1})
    early = float(auc_per_session(long, EARLY_WINDOW)["auc"].iloc[0])
    late = float(auc_per_session(long, LATE_WINDOW)["auc"].iloc[0])
    assert late == pytest.approx(early * 2.0 / 1.8)


def test_window_mean_does_not_scale_with_the_window_width() -> None:
    long = distance_long({"s1": 0.1})
    early = float(window_mean_per_session(long, EARLY_WINDOW)["mean"].iloc[0])
    late = float(window_mean_per_session(long, LATE_WINDOW)["mean"].iloc[0])
    assert early == pytest.approx(0.1) and late == pytest.approx(0.1)


def test_auc_matches_a_direct_trapezoid_on_a_varying_timecourse() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(-30, 121) / 30.0
    y = rng.normal(size=t.size)
    long = pd.DataFrame({"pair": "success-miss", "mouse": "m0", "day": 1,
                         "session_id": "s1", "t": t, "dist": y})
    m = (t >= 0.2) & (t <= 2.0)
    got = float(auc_per_session(long, EARLY_WINDOW)["auc"].iloc[0])
    assert got == pytest.approx(np.trapezoid(y[m], t[m]))


def test_auc_is_computed_in_time_order_whatever_the_input_order() -> None:
    long = distance_long({"s1": 0.1})
    shuffled = long.sample(frac=1.0, random_state=0)
    a = float(auc_per_session(long, EARLY_WINDOW)["auc"].iloc[0])
    b = float(auc_per_session(shuffled, EARLY_WINDOW)["auc"].iloc[0])
    assert a == pytest.approx(b)


def sessions_on_a_line(noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pair in PAIR_ORDER:
        for mouse in range(5):
            for day in range(1, 16):
                rows.append({
                    "pair": pair, "mouse": f"m{mouse}", "day": day,
                    "session_id": f"{pair}-{mouse}-{day}",
                    "auc": 0.1 + TRUE_SLOPE * day + rng.normal(0, noise),
                })
    return pd.DataFrame(rows)


def test_regression_recovers_an_exact_line() -> None:
    reg = day_regression(sessions_on_a_line())
    assert list(reg["pair"]) == list(PAIR_ORDER)
    np.testing.assert_allclose(reg["slope_per_day"], TRUE_SLOPE)
    np.testing.assert_allclose(reg["intercept"], 0.1)
    np.testing.assert_allclose(reg["r"], 1.0)


def test_regression_reports_the_counts_behind_each_line() -> None:
    reg = day_regression(sessions_on_a_line())
    assert set(reg["n_sessions"]) == {5 * 15}
    assert set(reg["n_mice"]) == {5}


def test_the_confidence_interval_brackets_the_true_slope() -> None:
    reg = day_regression(sessions_on_a_line(noise=0.02, seed=1))
    assert ((reg["ci95_lo"] < TRUE_SLOPE) & (TRUE_SLOPE < reg["ci95_hi"])).all()
    assert (reg["ci95_lo"] < reg["slope_per_day"]).all()
    assert (reg["slope_per_day"] < reg["ci95_hi"]).all()


def test_predictions_and_relative_change_follow_the_fitted_line() -> None:
    reg = day_regression(sessions_on_a_line()).iloc[0]
    assert reg["pred_day1"] == pytest.approx(reg["intercept"] + reg["slope_per_day"])
    assert reg["pred_day15"] == pytest.approx(reg["intercept"] + 15 * reg["slope_per_day"])
    assert reg["rel_change_day1_to_day15"] == pytest.approx(
        reg["pred_day15"] / reg["pred_day1"] - 1)


def test_the_prediction_days_are_named_in_the_columns() -> None:
    reg = day_regression(sessions_on_a_line(), predict_days=(1, 10))
    assert "pred_day10" in reg.columns
    assert "rel_change_day1_to_day10" in reg.columns


def test_holm_sits_next_to_the_raw_p_and_corrects_over_the_pairs() -> None:
    reg = day_regression(sessions_on_a_line(noise=0.05, seed=2))
    cols = list(reg.columns)
    assert cols[cols.index("p_value") + 1] == "p_holm"
    np.testing.assert_allclose(reg["p_holm"], holm(reg["p_value"].to_numpy()))
    assert (reg["p_holm"] >= reg["p_value"] - 1e-12).all()


def test_a_label_becomes_the_leading_columns() -> None:
    reg = day_regression(sessions_on_a_line(), label={"window": "late"})
    assert list(reg.columns)[:2] == ["window", "pair"]
    assert (reg["window"] == "late").all()


def test_tables_for_two_windows_concatenate() -> None:
    s = sessions_on_a_line()
    both = pd.concat([day_regression(s, label={"window": w}) for w in ("early", "late")],
                     ignore_index=True)
    assert len(both) == 2 * len(PAIR_ORDER)
    assert list(both["window"]) == ["early"] * 3 + ["late"] * 3


def test_ci_excludes_zero_only_when_the_interval_stays_on_one_side() -> None:
    assert ci_excludes_zero({"ci95_lo": 0.001, "ci95_hi": 0.004})
    assert ci_excludes_zero({"ci95_lo": -0.004, "ci95_hi": -0.001})
    assert not ci_excludes_zero({"ci95_lo": -0.001, "ci95_hi": 0.004})
    assert not ci_excludes_zero({"ci95_lo": 0.0, "ci95_hi": 0.004})


def test_fitted_line_draws_the_row_back() -> None:
    row = day_regression(sessions_on_a_line()).iloc[0]
    days = np.array([1.0, 15.0])
    np.testing.assert_allclose(fitted_line(row, days), [row["pred_day1"], row["pred_day15"]])


def test_the_q_value_follows_the_holm_p_and_is_benjamini_hochberg_over_the_pairs() -> None:
    reg = day_regression(sessions_on_a_line(noise=0.05, seed=2))
    cols = list(reg.columns)
    assert cols[cols.index("p_holm") + 1] == "q_value"
    np.testing.assert_allclose(reg["q_value"], benjamini_hochberg(reg["p_value"].to_numpy()))
    # Never stricter than Holm, never below the raw p.
    assert (reg["q_value"] <= reg["p_holm"] + 1e-12).all()
    assert (reg["q_value"] >= reg["p_value"] - 1e-12).all()
