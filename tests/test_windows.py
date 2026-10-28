"""Tests for `analysis.windows`."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.windows import (
    EARLY_WINDOW,
    LATE_WINDOW,
    PRE_WINDOW,
    WINDOWS,
    window_auc,
    window_mask,
    window_mean,
)


def _peth_axis() -> np.ndarray:
    """The project's cue-aligned axis: 1 s pre + 4 s post at 30 Hz."""
    return np.arange(-30, 121) / 30.0


def test_window_constants_match_the_registry() -> None:
    assert WINDOWS == {"pre": PRE_WINDOW, "early": EARLY_WINDOW, "late": LATE_WINDOW}
    assert PRE_WINDOW == (-1.0, 0.0, False)
    assert EARLY_WINDOW == (0.2, 2.0, True)
    assert LATE_WINDOW == (2.0, 4.0, True)


def test_pre_window_excludes_the_cue_sample() -> None:
    t = _peth_axis()
    mask = window_mask(t, PRE_WINDOW)
    assert mask.sum() == 30
    assert t[mask].min() == pytest.approx(-1.0)
    assert t[mask].max() == pytest.approx(-1 / 30.0)


def test_response_window_sample_counts() -> None:
    # The counts the figure notebooks rely on: 55 early samples, 61 late ones.
    t = _peth_axis()
    assert window_mask(t, EARLY_WINDOW).sum() == 55
    assert window_mask(t, LATE_WINDOW).sum() == 61


def test_early_and_late_share_the_boundary_sample() -> None:
    # Documented overlap: t = 2.0 s belongs to both, because the AUC over each
    # window is a trapezoidal integral that needs both endpoints.
    t = _peth_axis()
    shared = window_mask(t, EARLY_WINDOW) & window_mask(t, LATE_WINDOW)
    assert shared.sum() == 1
    assert t[shared][0] == pytest.approx(2.0)


def test_auc_of_a_constant_is_the_window_width() -> None:
    t = _peth_axis()
    y = np.ones_like(t)
    assert window_auc(t, y, EARLY_WINDOW) == pytest.approx(1.8)
    assert window_auc(t, y, LATE_WINDOW) == pytest.approx(2.0)


def test_auc_matches_a_direct_trapezoid_over_the_same_mask() -> None:
    rng = np.random.default_rng(0)
    t = _peth_axis()
    y = rng.normal(size=t.size)
    m = window_mask(t, EARLY_WINDOW)
    assert window_auc(t, y, EARLY_WINDOW) == pytest.approx(np.trapezoid(y[m], t[m]))


def test_auc_handles_one_series_per_column() -> None:
    t = _peth_axis()
    y = np.column_stack([np.ones_like(t), 2 * np.ones_like(t)])
    np.testing.assert_allclose(window_auc(t, y, EARLY_WINDOW), [1.8, 3.6])


def test_auc_refuses_a_window_too_narrow_to_integrate() -> None:
    t = np.array([0.0, 5.0])  # no sample inside the early window
    with pytest.raises(ValueError, match="trapezoidal"):
        window_auc(t, np.ones_like(t), EARLY_WINDOW)


def test_mean_uses_only_the_samples_inside_the_window() -> None:
    t = _peth_axis()
    y = np.where(window_mask(t, PRE_WINDOW), 5000.0, 8000.0)
    assert window_mean(t, y, PRE_WINDOW) == pytest.approx(5000.0)
    assert window_mean(t, y, EARLY_WINDOW) == pytest.approx(8000.0)


def test_mean_handles_one_series_per_column_and_ignores_nan() -> None:
    t = _peth_axis()
    y = np.column_stack([np.full(t.size, 3.0), np.full(t.size, 7.0)])
    y[window_mask(t, PRE_WINDOW).argmax(), 0] = np.nan
    np.testing.assert_allclose(window_mean(t, y, PRE_WINDOW), [3.0, 7.0])


def test_mean_of_an_empty_window_is_nan() -> None:
    t = np.array([10.0, 11.0])
    assert np.isnan(window_mean(t, np.ones_like(t), PRE_WINDOW))
    np.testing.assert_array_equal(
        window_mean(t, np.ones((2, 3)), PRE_WINDOW), np.full(3, np.nan)
    )
