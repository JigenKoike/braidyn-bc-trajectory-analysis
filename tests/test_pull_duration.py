"""Tests for `analysis.pull_duration`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.pull_duration import compute_pull_durations


def _make_signal(n_frames=300, fs=30.0):
    """Helper: create a state_lever signal and its timestamps."""
    ts = np.arange(n_frames) / fs
    sl = np.zeros(n_frames, dtype=np.float64)
    return sl, ts


def test_single_press_within_window() -> None:
    sl, ts = _make_signal()
    # Press from t=1.0 to t=1.5 (15 frames at 30 Hz)
    sl[30:45] = 1
    trials = pd.DataFrame({"start_time": [0.5], "stop_time": [2.0]})
    out = compute_pull_durations(sl, ts, trials)
    assert out["n_presses"].iloc[0] == 1
    assert not out["already_down"].iloc[0]
    np.testing.assert_allclose(out["pull_dur_first"].iloc[0], 0.5, atol=1 / 30)
    np.testing.assert_allclose(out["pull_dur_max"].iloc[0], 0.5, atol=1 / 30)


def test_no_press_in_window() -> None:
    sl, ts = _make_signal()
    sl[200:210] = 1  # press well outside the trial window
    trials = pd.DataFrame({"start_time": [0.5], "stop_time": [2.0]})
    out = compute_pull_durations(sl, ts, trials)
    assert out["n_presses"].iloc[0] == 0
    assert np.isnan(out["pull_dur_first"].iloc[0])
    assert np.isnan(out["pull_dur_max"].iloc[0])


def test_already_down_at_cue() -> None:
    # Lever pressed before the cue and released after — counts as already_down,
    # and the in-window press counter stays at 0 (no fresh 0→1 transition).
    sl, ts = _make_signal()
    sl[15:60] = 1  # press from t=0.5 to t=2.0
    trials = pd.DataFrame({"start_time": [1.0], "stop_time": [3.0]})
    out = compute_pull_durations(sl, ts, trials)
    assert out["already_down"].iloc[0]
    assert out["n_presses"].iloc[0] == 0
    assert np.isnan(out["pull_dur_first"].iloc[0])
    assert np.isnan(out["pull_dur_max"].iloc[0])


def test_multi_press_first_vs_max() -> None:
    sl, ts = _make_signal()
    # Press 1: short, 30..35 (5 frames = ~0.167s)
    # Press 2: long,  50..80 (30 frames = ~1.0s)   ← max
    # Press 3: med,   90..105 (15 frames = 0.5s)
    sl[30:35] = 1
    sl[50:80] = 1
    sl[90:105] = 1
    trials = pd.DataFrame({"start_time": [0.5], "stop_time": [4.0]})
    out = compute_pull_durations(sl, ts, trials)
    assert out["n_presses"].iloc[0] == 3
    np.testing.assert_allclose(out["pull_dur_first"].iloc[0], 5 / 30, atol=1 / 30)
    np.testing.assert_allclose(out["pull_dur_max"].iloc[0], 30 / 30, atol=1 / 30)


def test_press_extending_past_stop_time() -> None:
    # Press starts in window but mouse holds way past stop_time — duration
    # should reflect actual hold time (until next release), not be truncated.
    sl, ts = _make_signal()
    sl[40:200] = 1  # press for ~5.3s starting at t=1.33
    trials = pd.DataFrame({"start_time": [1.0], "stop_time": [2.0]})
    out = compute_pull_durations(sl, ts, trials)
    assert out["n_presses"].iloc[0] == 1
    np.testing.assert_allclose(out["pull_dur_first"].iloc[0], 160 / 30, atol=1 / 30)


def test_index_preserved() -> None:
    sl, ts = _make_signal()
    sl[30:45] = 1
    trials = pd.DataFrame(
        {"start_time": [0.5, 5.0], "stop_time": [2.0, 7.0]}, index=[7, 19]
    )
    out = compute_pull_durations(sl, ts, trials)
    assert list(out.index) == [7, 19]


def test_shape_mismatch_raises() -> None:
    sl = np.zeros(100)
    ts = np.arange(200)
    trials = pd.DataFrame({"start_time": [0.0], "stop_time": [1.0]})
    try:
        compute_pull_durations(sl, ts, trials)
    except ValueError as e:
        assert "shape" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_empty_trials_returns_empty() -> None:
    sl, ts = _make_signal()
    trials = pd.DataFrame({"start_time": [], "stop_time": []})
    out = compute_pull_durations(sl, ts, trials)
    assert len(out) == 0
    assert set(out.columns) == {"already_down", "n_presses", "pull_dur_first", "pull_dur_max"}
