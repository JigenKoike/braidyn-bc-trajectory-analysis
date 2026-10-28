"""Tests for `analysis.cue_align`."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.cue_align import cue_align, cue_aligned_time_axis


def _ramp(n: int) -> np.ndarray:
    """Identity signal: signal[t] = t."""
    return np.arange(n, dtype=float)


def test_basic_alignment_window_contents() -> None:
    timestamps = np.arange(100, dtype=float)  # 1 Hz, t = 0..99
    signal = _ramp(100).reshape(100, 1)
    cue_times = np.array([10.0, 50.0])
    slices, valid = cue_align(signal, timestamps, cue_times, pre_frames=3, post_frames=3)

    assert slices.shape == (2, 7, 1)
    assert valid.tolist() == [0, 1]
    # Window around t=10 with searchsorted(side="left") picks idx=10:
    # frames 7..13 inclusive → values 7..13
    np.testing.assert_array_equal(slices[0, :, 0], np.arange(7, 14, dtype=float))
    np.testing.assert_array_equal(slices[1, :, 0], np.arange(47, 54, dtype=float))


def test_cue_frame_at_index_pre_frames() -> None:
    timestamps = np.arange(100, dtype=float)
    signal = _ramp(100)
    pre, post = 5, 10
    cue_time = 30.0
    slices, valid = cue_align(signal, timestamps, np.array([cue_time]), pre, post)
    assert slices.shape == (1, pre + post + 1)
    # The cue frame is at index pre_frames; signal[t=30] == 30.
    assert slices[0, pre] == cue_time


def test_drop_out_of_bounds_left() -> None:
    timestamps = np.arange(100, dtype=float)
    signal = _ramp(100).reshape(100, 1)
    # cue at t=2 → start = 2 - 3 = -1, out of range
    cue_times = np.array([2.0, 50.0])
    slices, valid = cue_align(signal, timestamps, cue_times, pre_frames=3, post_frames=3)
    assert valid.tolist() == [1]
    assert slices.shape == (1, 7, 1)


def test_drop_out_of_bounds_right() -> None:
    timestamps = np.arange(100, dtype=float)
    signal = _ramp(100).reshape(100, 1)
    # cue at t=98 → end = 98 + 3 + 1 = 102, out of range (n_frames = 100)
    cue_times = np.array([98.0, 50.0])
    slices, valid = cue_align(signal, timestamps, cue_times, pre_frames=3, post_frames=3)
    assert valid.tolist() == [1]


def test_baseline_subtract_uses_first_pre_frames() -> None:
    timestamps = np.arange(20, dtype=float)
    signal = _ramp(20).reshape(20, 1)  # F_t = t
    cue_times = np.array([10.0])
    slices, _ = cue_align(
        signal, timestamps, cue_times,
        pre_frames=3, post_frames=3,
        baseline_subtract=True,
    )
    # Window around t=10: indices 7..13 → values [7,8,9,10,11,12,13]
    # Baseline = mean([7,8,9]) = 8.
    # After subtraction: [-1, 0, 1, 2, 3, 4, 5]
    np.testing.assert_array_almost_equal(
        slices[0, :, 0],
        np.array([-1, 0, 1, 2, 3, 4, 5], dtype=float),
    )


def test_baseline_subtract_promotes_int_to_float() -> None:
    timestamps = np.arange(20, dtype=float)
    signal = _ramp(20).astype(np.uint16).reshape(20, 1)
    slices, _ = cue_align(
        signal, timestamps, np.array([10.0]),
        pre_frames=3, post_frames=3,
        baseline_subtract=True,
    )
    # Promoted so we can hold negative values
    assert np.issubdtype(slices.dtype, np.floating)


def test_multi_feature_signal() -> None:
    timestamps = np.arange(100, dtype=float)
    # signal shape (100, 5): each column is the same ramp
    signal = np.tile(_ramp(100)[:, None], (1, 5))
    cue_times = np.array([50.0])
    slices, _ = cue_align(signal, timestamps, cue_times, pre_frames=3, post_frames=3)
    assert slices.shape == (1, 7, 5)
    for feature_idx in range(5):
        np.testing.assert_array_equal(slices[0, :, feature_idx], np.arange(47, 54, dtype=float))


def test_empty_cue_list_returns_empty_with_correct_shape() -> None:
    timestamps = np.arange(100, dtype=float)
    signal = np.tile(_ramp(100)[:, None], (1, 4))  # (100, 4)
    slices, valid = cue_align(
        signal, timestamps, np.array([], dtype=float),
        pre_frames=3, post_frames=3,
    )
    assert slices.shape == (0, 7, 4)
    assert valid.shape == (0,)


def test_all_cues_out_of_bounds_returns_empty() -> None:
    timestamps = np.arange(100, dtype=float)
    signal = _ramp(100).reshape(100, 1)
    cue_times = np.array([0.0, 1.0])  # both too early for pre_frames=3
    slices, valid = cue_align(signal, timestamps, cue_times, pre_frames=3, post_frames=3)
    assert slices.shape == (0, 7, 1)
    assert valid.shape == (0,)


def test_non_uniform_timestamps() -> None:
    """`searchsorted` uses timestamp values, not assumed uniform sampling."""
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    signal = np.arange(10, dtype=float).reshape(10, 1)  # signal[t] = frame index
    # cue at t=1.7 → searchsorted(side="left") returns idx where 1.7 would be inserted
    # timestamps[3] = 1.5, timestamps[4] = 2.0 → idx = 4 (1.7 > 1.5)
    cue_times = np.array([1.7])
    slices, valid = cue_align(signal, timestamps, cue_times, pre_frames=2, post_frames=2)
    assert valid.tolist() == [0]
    # window = [4-2 : 4+3] = [2:7] → frames 2,3,4,5,6
    np.testing.assert_array_equal(slices[0, :, 0], np.array([2, 3, 4, 5, 6], dtype=float))


def test_time_axis_basic() -> None:
    ta = cue_aligned_time_axis(pre_frames=30, post_frames=120, frame_rate=30.0)
    assert ta.shape == (151,)
    assert ta[0] == pytest.approx(-1.0)
    assert ta[30] == pytest.approx(0.0)  # cue frame is at index pre_frames
    assert ta[-1] == pytest.approx(4.0)


def test_time_axis_symmetric_window() -> None:
    ta = cue_aligned_time_axis(pre_frames=10, post_frames=10, frame_rate=10.0)
    assert ta.shape == (21,)
    np.testing.assert_array_almost_equal(ta, np.arange(-10, 11) / 10.0)
