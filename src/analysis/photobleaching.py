"""Photobleaching analysis helpers — per-ROI raw F over learning.

Computes cue-aligned, trial-averaged raw F PETHs per ROI. Summarizing a PETH
over a time window is :mod:`analysis.windows`' job; the baseline (pre-cue)
mean is what the downstream analyses use as a session-level F value.

Per-ROI raw F is richer than a single brain-wide mean: the brain-wide value
can be derived from the per-ROI array via ``union_F_from_per_roi`` (a
pixel-count-weighted average), but region-specific photobleaching ("motor
cortex decays but visual cortex does not") is only visible per ROI.
"""

from __future__ import annotations

import numpy as np

from .cue_align import cue_align


def session_F_peth(
    F_per_roi: np.ndarray,
    timestamps: np.ndarray,
    cue_times: np.ndarray,
    pre_frames: int,
    post_frames: int,
) -> tuple[np.ndarray | None, int]:
    """Trial-averaged raw F PETH per ROI for one condition.

    Parameters
    ----------
    F_per_roi : array, shape (n_frames, n_roi)
        Per-ROI raw fluorescence over time (one channel).
    timestamps : array, shape (n_frames,)
    cue_times : array, shape (n_cues,)
    pre_frames, post_frames : int

    Returns
    -------
    F_peth : array of shape (window, n_roi) or None
        Trial-averaged raw F PETH. NOT baseline-subtracted — raw absolute
        F is what photobleaching analysis needs.
    n_valid_trials : int
    """
    slices, valid = cue_align(
        F_per_roi, timestamps, cue_times,
        pre_frames=pre_frames, post_frames=post_frames,
        baseline_subtract=False,
    )
    if len(valid) == 0:
        return None, 0
    return slices.mean(axis=0), len(valid)


def union_F_from_per_roi(
    F_per_roi: np.ndarray,
    n_pixels_per_roi: np.ndarray,
) -> np.ndarray:
    """Derive brain-wide union F (matching sandbox phaseB) from per-ROI F.

    Each ROI's F is already the mean over its mask pixels. The union mean
    is the pixel-count-weighted average across ROIs:
        F_union[t] = sum_r (F_per_roi[t, r] * n_px[r]) / sum(n_px)
    """
    n_pixels_per_roi = np.asarray(n_pixels_per_roi, dtype=np.float64)
    F_per_roi = np.asarray(F_per_roi, dtype=np.float64)
    return (F_per_roi @ n_pixels_per_roi) / n_pixels_per_roi.sum()
