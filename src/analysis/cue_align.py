"""Cue-aligned slicing for BraiDyn-BC time series.

A pure-numpy helper used to build trial-by-trial slices of any per-frame
signal (ΔF/F traces, raw F traces, behavioral signals, keypoints, ...)
around event times such as cue onsets or reward times. The same helper
serves PCA (which averages slices across trials per condition) and
photobleaching analysis (which averages across trials and ROIs).

The window length is `pre_frames + post_frames + 1` — the +1 frame is the
cue/event frame itself, located at index `pre_frames` of each slice.
"""

from __future__ import annotations

import numpy as np


def cue_align(
    signal: np.ndarray,
    timestamps: np.ndarray,
    cue_times: np.ndarray,
    pre_frames: int,
    post_frames: int,
    *,
    baseline_subtract: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build cue-aligned trial slices from a per-frame signal.

    For each cue, take the first frame at or after it — ``searchsorted(
    side="left")`` on ``timestamps``, not the nearest frame — and extract a
    window of length ``pre_frames + post_frames + 1`` with that frame at
    index ``pre_frames``. There is no interpolation between frames. Trials
    whose window falls outside the frame range are silently dropped.

    Parameters
    ----------
    signal : array, shape (n_frames, ...) or (n_frames,)
        Per-frame time series. The first axis must be the frame axis;
        any remaining axes (features / ROIs / spatial dims) are preserved.
    timestamps : array, shape (n_frames,)
        Frame times in seconds, monotonically increasing.
    cue_times : array, shape (n_cues,)
        Event times (cue onsets, reward, etc.) in seconds.
    pre_frames, post_frames : int
        Number of frames before and after the cue. Window length is
        ``pre_frames + post_frames + 1``.
    baseline_subtract : bool, keyword-only
        If True, subtract the per-trial mean of the pre-cue window
        (frames ``[0:pre_frames]``) from every frame of that trial.

    Returns
    -------
    slices : array, shape (n_valid_trials, window, ...)
        Cue-aligned slices. Dtype matches ``signal`` unless
        ``baseline_subtract`` forces a float promotion.
    valid_idx : array, shape (n_valid_trials,) int
        Indices into the original ``cue_times`` of trials kept (those
        whose window fit inside the imaging range).
    """
    signal = np.asarray(signal)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    cue_times = np.asarray(cue_times, dtype=np.float64)

    window = pre_frames + post_frames + 1
    n_frames_total = len(timestamps)

    trig_idx = np.searchsorted(timestamps, cue_times, side="left")
    starts = trig_idx - pre_frames
    ends = trig_idx + post_frames + 1
    valid_mask = (starts >= 0) & (ends <= n_frames_total)
    valid_idx = np.where(valid_mask)[0]

    out_shape = (len(valid_idx), window) + tuple(signal.shape[1:])
    if len(valid_idx) == 0:
        return np.empty(out_shape, dtype=signal.dtype), valid_idx

    slices = np.empty(out_shape, dtype=signal.dtype)
    for i, t_idx in enumerate(valid_idx):
        slices[i] = signal[int(starts[t_idx]) : int(ends[t_idx])]

    if baseline_subtract:
        # Promote to float for safe subtraction (e.g. uint16 → float64)
        if not np.issubdtype(slices.dtype, np.floating):
            slices = slices.astype(np.float64)
        baseline = slices[:, :pre_frames, ...].mean(axis=1, keepdims=True)
        slices = slices - baseline

    return slices, valid_idx


def cue_aligned_time_axis(
    pre_frames: int, post_frames: int, frame_rate: float
) -> np.ndarray:
    """Time axis relative to cue (in seconds).

    Returns an array of length ``pre_frames + post_frames + 1`` whose
    values span ``[-pre_frames / frame_rate, +post_frames / frame_rate]``,
    with 0 at index ``pre_frames`` (the cue frame).
    """
    return np.arange(-pre_frames, post_frames + 1) / float(frame_rate)
