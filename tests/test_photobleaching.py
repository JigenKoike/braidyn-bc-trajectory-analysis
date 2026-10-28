"""Tests for `analysis.photobleaching`."""

from __future__ import annotations

import numpy as np

from analysis.photobleaching import (
    session_F_peth,
    union_F_from_per_roi,
)


def test_session_F_peth_shape_and_no_baseline_subtract() -> None:
    # signal where F_per_roi[t, r] = 1000 * (r + 1) + t
    # (constant + small linear ramp so we can predict means)
    timestamps = np.arange(100, dtype=float)
    F_per_roi = np.outer(np.arange(100, dtype=float), np.ones(3)) + (np.arange(3) + 1) * 1000.0
    # F_per_roi shape (100, 3); ROI 0 ranges 1000..1099, ROI 1 ranges 2000..2099, etc.

    cue_times = np.array([50.0])
    pre, post = 3, 3
    peth, n = session_F_peth(F_per_roi, timestamps, cue_times, pre, post)
    assert peth is not None
    assert peth.shape == (pre + post + 1, 3)
    assert n == 1

    # Window around t=50: indices 47..53 → values for ROI 0 are 1047..1053
    # mean = 1050; for ROI 1 it's 2050; for ROI 2 it's 3050
    np.testing.assert_array_almost_equal(peth.mean(axis=0), [1050, 2050, 3050])


def test_session_F_peth_no_valid_returns_none() -> None:
    timestamps = np.arange(100, dtype=float)
    F_per_roi = np.zeros((100, 4))
    # cue at t=0 with pre=3 → window starts at -3 → out of bounds
    peth, n = session_F_peth(F_per_roi, timestamps, np.array([0.0]), pre_frames=3, post_frames=3)
    assert peth is None
    assert n == 0


def test_union_F_matches_pixel_weighted_average() -> None:
    # 2 ROIs, ROI 0 has 100 pixels, ROI 1 has 300 pixels.
    # F[t, 0] = 10, F[t, 1] = 20 (constants).
    # Union F should be (10*100 + 20*300) / 400 = 7000 / 400 = 17.5.
    F_per_roi = np.zeros((50, 2))
    F_per_roi[:, 0] = 10.0
    F_per_roi[:, 1] = 20.0
    n_px = np.array([100, 300])
    F_union = union_F_from_per_roi(F_per_roi, n_px)
    np.testing.assert_array_almost_equal(F_union, np.full(50, 17.5))


def test_union_F_equal_pixels_is_simple_mean() -> None:
    F_per_roi = np.array([[1.0, 3.0], [5.0, 7.0]])
    n_px = np.array([10, 10])  # equal weights
    F_union = union_F_from_per_roi(F_per_roi, n_px)
    np.testing.assert_array_almost_equal(F_union, [2.0, 6.0])  # row means
