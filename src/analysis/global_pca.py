"""Fitting one PC space to the grand mean response, and projecting into it.

The steps, which ``scripts/run_pairwise_pca.py`` runs end to end:

1. For each task-day session, compute the trial-averaged, baseline-
   subtracted cue-aligned PETH of each outcome — success, failure and miss
   (:func:`condition_peth_matrices`) — and combine them weighted by their
   numbers of valid trials (:func:`union_peth_matrix`), which is the mean
   over all of the session's valid trials → ``M_session`` of shape
   ``(n_roi, n_time)``.
2. Take the grand mean of ``M_session`` across sessions, each session
   counting equally → ``M_global`` ``(n_roi, n_time)``.
3. Fit ``PCA(n_components=20)`` on ``M_global`` — ROIs are samples, time
   points are features, each time point centred across ROIs. The "ROI
   weights" ``W = pca.transform(M_global)`` give each ROI's score on each
   PC; a column's length is that PC's singular value, not 1.
4. For each session × outcome, project the trial-averaged PETH ``M_c`` into
   PC space: ``Z_c = M_c.T @ W`` has shape ``(n_time, n_pc)``, with no
   centring of ``M_c`` first.
5. The distance between two outcomes in PC space (first ``n_pc=3``) over
   time: ``||Z_a[:, :3] - Z_b[:, :3]||`` per time point
   (:func:`pc_distance`), for success–miss, failure–miss and
   success–failure.

Steps 1–3 are this module's responsibility. Steps 4–5 are also here as
small projection helpers. Driver scripts/notebooks compose these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .cue_align import cue_align


# Defaults from the sandbox notebook: 1 s pre + 4 s post @ 30 Hz.
DEFAULT_PRE_FRAMES = 30
DEFAULT_POST_FRAMES = 120
DEFAULT_FRAME_RATE = 30.0


def session_peth_matrix(
    signal: np.ndarray,
    timestamps: np.ndarray,
    cue_times: np.ndarray,
    pre_frames: int = DEFAULT_PRE_FRAMES,
    post_frames: int = DEFAULT_POST_FRAMES,
) -> tuple[np.ndarray | None, int]:
    """Trial-averaged, baseline-subtracted cue-aligned PETH.

    ``signal`` is typically per-ROI ΔF/F ``(n_frames, n_roi)``. Returns
    ``(M, n_valid_trials)`` where ``M`` has shape ``(n_roi, n_time)``, or
    ``(None, 0)`` if no trials' windows fit inside the imaging range.
    """
    slices, valid = cue_align(
        signal, timestamps, cue_times,
        pre_frames=pre_frames, post_frames=post_frames,
        baseline_subtract=True,
    )
    if len(valid) == 0:
        return None, 0
    # slices: (n_valid_trials, window, n_roi); take trial mean then transpose.
    return slices.mean(axis=0).T, len(valid)


@dataclass(frozen=True)
class GlobalPCAResult:
    """Result of a Global PCA fit across sessions."""

    pca: PCA
    M_global: np.ndarray   # (n_roi, n_time) — grand mean across sessions
    W: np.ndarray          # (n_roi, n_pc) — ROI weights (pca.transform(M_global))
    n_sessions_used: int


def fit_global_pca(
    M_per_session: Iterable[np.ndarray],
    n_components: int = 20,
    svd_solver: str = "full",
    random_state: int = 42,
) -> GlobalPCAResult:
    """Average per-session PETH matrices and fit PCA on the grand mean.

    All input matrices must have the same shape ``(n_roi, n_time)``.
    """
    Ms = list(M_per_session)
    if not Ms:
        raise ValueError("fit_global_pca: empty session list")
    stacked = np.stack(Ms, axis=0)  # (n_sessions, n_roi, n_time)
    M_global = stacked.mean(axis=0)  # (n_roi, n_time)
    pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)
    pca.fit(M_global)
    W = pca.transform(M_global)  # (n_roi, n_pc)
    return GlobalPCAResult(pca=pca, M_global=M_global, W=W, n_sessions_used=len(Ms))


def project_into_pc_space(M: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Project a ``(n_roi, n_time)`` PETH into PC space.

    Returns a ``(n_time, n_pc)`` trajectory.
    """
    return M.T @ W


def pc_distance(
    Z_a: np.ndarray,
    Z_b: np.ndarray,
    n_pc: int = 3,
) -> np.ndarray:
    """Euclidean distance in PC space (first ``n_pc`` components) per time.

    ``Z_a`` and ``Z_b`` are ``(n_time, n_pc_total)`` trajectories from
    :func:`project_into_pc_space`. Returns a ``(n_time,)`` array. Which
    outcome each argument represents is up to the caller — the function
    is symmetric in ``Z_a`` and ``Z_b``.
    """
    return np.linalg.norm(Z_a[:, :n_pc] - Z_b[:, :n_pc], axis=1)


# Backward-compatible alias for the success–miss-only pipeline.
success_miss_distance = pc_distance


def condition_peth_matrices(
    signal: np.ndarray,
    timestamps: np.ndarray,
    trials: pd.DataFrame,
    outcomes: Iterable[str] = ("success", "failure", "miss"),
    outcome_col: str = "trial_outcome_paper",
    cue_col: str = "start_time",
    pre_frames: int = DEFAULT_PRE_FRAMES,
    post_frames: int = DEFAULT_POST_FRAMES,
) -> dict[str, tuple[np.ndarray | None, int]]:
    """Per-outcome trial-averaged cue-aligned PETHs for one session.

    Calls :func:`session_peth_matrix` once per outcome, using the cue
    times of trials whose ``outcome_col`` equals that outcome value.

    Parameters
    ----------
    signal : array, shape (n_frames, n_roi) or (n_frames,)
        Per-frame signal (typically ROI ΔF/F).
    timestamps : array, shape (n_frames,)
    trials : DataFrame
        Must contain ``outcome_col`` and ``cue_col``.
    outcomes : iterable of str
        Outcome labels to compute PETHs for. Missing labels get
        ``(None, 0)`` in the result.

    Returns
    -------
    dict mapping outcome name → ``(M, n_valid_trials)``, where ``M`` has
    shape ``(n_roi, n_time)`` (or ``None`` if no trials of that outcome
    had windows fitting inside the imaging range).
    """
    out: dict[str, tuple[np.ndarray | None, int]] = {}
    for oc in outcomes:
        cue_times = trials.loc[trials[outcome_col] == oc, cue_col].to_numpy(dtype=float)
        M, n = session_peth_matrix(
            signal, timestamps, cue_times,
            pre_frames=pre_frames, post_frames=post_frames,
        )
        out[oc] = (M, n)
    return out


def union_peth_matrix(
    condition_peths: Mapping[str, tuple[np.ndarray | None, int]],
) -> np.ndarray | None:
    """Trial-count-weighted average of per-condition PETHs.

    Given the dict returned by :func:`condition_peth_matrices`, compute
    the trial-count-weighted mean of the per-condition PETH matrices —
    i.e. what a single call to :func:`session_peth_matrix` would return
    if the outcomes' cues were pooled. This is the PETH used to fit the
    global PC space when all outcomes are combined.

    Returns ``None`` if every condition has 0 valid trials.
    """
    Ms, ns = [], []
    for M, n in condition_peths.values():
        if M is not None and n > 0:
            Ms.append(M)
            ns.append(n)
    if not Ms:
        return None
    stacked = np.stack(Ms, axis=0)  # (n_cond, n_roi, n_time)
    weights = np.asarray(ns, dtype=np.float64)
    return np.tensordot(weights, stacked, axes=1) / weights.sum()
