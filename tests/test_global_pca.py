"""Tests for `analysis.global_pca`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.global_pca import (
    condition_peth_matrices,
    pc_distance,
    session_peth_matrix,
    success_miss_distance,
    union_peth_matrix,
)


def _make_signal(n_frames=200, n_roi=4, fs=30.0):
    ts = np.arange(n_frames) / fs
    # per-ROI signal: constant baseline + small time ramp so PCA has structure
    signal = np.zeros((n_frames, n_roi), dtype=float)
    for r in range(n_roi):
        signal[:, r] = 100 * (r + 1) + 0.1 * np.arange(n_frames)
    return signal, ts


def test_condition_peth_matrices_all_present() -> None:
    signal, ts = _make_signal()
    trials = pd.DataFrame({
        "trial_outcome_paper": ["success", "success", "failure", "miss", "miss"],
        "start_time": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    out = condition_peth_matrices(signal, ts, trials, pre_frames=5, post_frames=10)
    assert set(out.keys()) == {"success", "failure", "miss"}
    for oc, expected_n in (("success", 2), ("failure", 1), ("miss", 2)):
        M, n = out[oc]
        assert n == expected_n
        assert M.shape == (4, 16)  # (n_roi, pre+post+1)


def test_condition_peth_matrices_missing_condition() -> None:
    # No failure trials — dict should still contain the key with (None, 0).
    signal, ts = _make_signal()
    trials = pd.DataFrame({
        "trial_outcome_paper": ["success", "miss"],
        "start_time": [1.0, 4.0],
    })
    out = condition_peth_matrices(signal, ts, trials, pre_frames=5, post_frames=10)
    assert set(out.keys()) == {"success", "failure", "miss"}
    assert out["failure"] == (None, 0)
    assert out["success"][1] == 1
    assert out["miss"][1] == 1


def test_union_peth_matrix_trial_weighted() -> None:
    # Two conditions with known PETHs and different trial counts. Union
    # should equal trial-count-weighted average, matching a direct
    # session_peth_matrix call on the pooled cue times.
    signal, ts = _make_signal()
    trials = pd.DataFrame({
        "trial_outcome_paper": ["success"] * 3 + ["miss"] * 2,
        "start_time":          [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    condition_peths = condition_peth_matrices(
        signal, ts, trials,
        outcomes=("success", "miss"),
        pre_frames=5, post_frames=10,
    )
    union = union_peth_matrix(condition_peths)
    # Direct computation on pooled cue times:
    M_direct, _ = session_peth_matrix(
        signal, ts, trials["start_time"].to_numpy(),
        pre_frames=5, post_frames=10,
    )
    np.testing.assert_allclose(union, M_direct, atol=1e-10)


def test_union_peth_matrix_all_empty_returns_none() -> None:
    signal, ts = _make_signal()
    empty_dict = {"success": (None, 0), "miss": (None, 0)}
    assert union_peth_matrix(empty_dict) is None


def test_pc_distance_symmetric() -> None:
    Z_a = np.random.RandomState(0).randn(10, 5)
    Z_b = np.random.RandomState(1).randn(10, 5)
    d_ab = pc_distance(Z_a, Z_b, n_pc=3)
    d_ba = pc_distance(Z_b, Z_a, n_pc=3)
    np.testing.assert_allclose(d_ab, d_ba)
    assert d_ab.shape == (10,)


def test_pc_distance_ignores_higher_pcs() -> None:
    # Distance should only use the first n_pc columns; changing later
    # columns must not affect the result.
    Z_a = np.zeros((5, 10))
    Z_b = np.zeros((5, 10))
    Z_b[:, 0] = 1.0  # difference on PC0 only
    Z_b[:, 5:] = 999.0  # noise beyond n_pc=3
    d = pc_distance(Z_a, Z_b, n_pc=3)
    np.testing.assert_allclose(d, np.ones(5))


def test_success_miss_distance_is_alias() -> None:
    # Backward-compatible alias should behave identically.
    assert success_miss_distance is pc_distance
