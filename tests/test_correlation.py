"""Tests for the two-stage per-mouse rank correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.correlation import MIN_SESSIONS_PER_MOUSE, per_mouse_rho, rho_against_zero


def _sessions(per_mouse: dict[str, tuple[list[float], list[float]]]) -> pd.DataFrame:
    rows = []
    for mouse, (x, y) in per_mouse.items():
        rows += [{"mouse": mouse, "x": a, "y": b} for a, b in zip(x, y)]
    return pd.DataFrame(rows)


def test_per_mouse_rho_is_one_row_per_mouse() -> None:
    out = per_mouse_rho(_sessions({
        "A": ([1, 2, 3, 4], [10, 20, 30, 40]),   # perfectly increasing
        "B": ([1, 2, 3], [30, 20, 10]),          # perfectly decreasing
    }), "x", "y")
    assert list(out["mouse"]) == ["A", "B"]
    assert list(out["n_sessions"]) == [4, 3]
    assert out.loc[0, "rho"] == pytest.approx(1.0)
    assert out.loc[1, "rho"] == pytest.approx(-1.0)


def test_per_mouse_rho_drops_a_mouse_with_too_few_sessions() -> None:
    sessions = _sessions({
        "A": ([1, 2, 3], [1, 2, 3]),
        "B": ([1, 2], [1, 2]),  # one short of the floor
    })
    assert MIN_SESSIONS_PER_MOUSE == 3
    assert list(per_mouse_rho(sessions, "x", "y")["mouse"]) == ["A"]
    # The floor is a parameter, not a law of nature.
    assert list(per_mouse_rho(sessions, "x", "y", min_sessions=2)["mouse"]) == ["A", "B"]


def test_per_mouse_rho_drops_a_mouse_whose_measure_does_not_vary() -> None:
    # A constant column makes rho undefined rather than zero, and scipy says nan.
    out = per_mouse_rho(_sessions({
        "A": ([1, 2, 3], [1, 2, 3]),
        "B": ([5, 5, 5], [1, 2, 3]),
    }), "x", "y")
    assert list(out["mouse"]) == ["A"]


def test_rho_against_zero_reports_the_median_and_a_two_sided_p() -> None:
    out = rho_against_zero([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert out["n_mice"] == 6
    assert out["median_rho"] == pytest.approx(0.35)
    assert 0 < out["p"] < 0.05  # every mouse on the same side of zero


def test_rho_against_zero_sees_nothing_in_a_balanced_sample() -> None:
    out = rho_against_zero([-0.3, -0.1, 0.1, 0.3])
    assert out["median_rho"] == pytest.approx(0.0)
    assert out["p"] > 0.5


def test_rho_against_zero_refuses_an_empty_or_unusable_sample() -> None:
    with pytest.raises(AssertionError):
        rho_against_zero([])
    with pytest.raises(AssertionError):
        rho_against_zero([0.1, np.nan])
