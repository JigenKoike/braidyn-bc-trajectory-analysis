"""How much each PC counts in the distance between two outcomes.

The PC coordinates are each outcome's mean activity multiplied by the ROI
weights ``W`` (ROIs × PCs). ``W``'s columns are not of unit length: the PCA
scores keep the scale of the data, so each column's squared length is
proportional to the variance its PC explains. The distance over PC1–3 is
therefore stretched along PC1 — the **variance weight** the paper uses —
and is decided almost entirely by it.

**Equal weight** divides each coordinate by its column's length, which is
the same as projecting with unit-length weights: every PC counts the same,
whatever share of the variance it explains. On this data that raises PC2
about threefold and PC3 about elevenfold relative to PC1.

Both are a rescaling of the coordinates the pipeline already saved, so no
activity has to be projected again: :func:`weighted_pc_distance` takes the
saved trajectories and a per-PC scale.
"""

from __future__ import annotations

import numpy as np

from .cv_distance import N_PC_DISTANCE


def equal_weight_scale(weights: np.ndarray, n_pc: int = N_PC_DISTANCE) -> np.ndarray:
    """The per-PC factor that gives each of the first ``n_pc`` PCs unit-length weights.

    ``weights`` is the ROI × PC weight matrix (``data/paper/pca/roi_weights.npy``).
    """
    w = np.asarray(weights, dtype=np.float64)
    assert w.ndim == 2 and w.shape[1] >= n_pc, f"weights should be ROIs × PCs, got {w.shape}"
    norms = np.linalg.norm(w[:, :n_pc], axis=0)
    assert (norms > 0).all(), "a PC with all-zero weights cannot be rescaled"
    return 1.0 / norms


def weighted_pc_distance(z_a: np.ndarray, z_b: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Distance between two trajectories over the first ``len(scale)`` PCs, each rescaled.

    ``z_a`` and ``z_b`` are ``(n_time, n_pc_total)`` trajectories. With
    ``scale`` all ones this is :func:`analysis.global_pca.pc_distance`, the
    variance-weighted distance; with :func:`equal_weight_scale` it is the
    equally weighted one.
    """
    scale = np.asarray(scale, dtype=np.float64)
    n_pc = len(scale)
    a = np.asarray(z_a, dtype=np.float64)[:, :n_pc]
    b = np.asarray(z_b, dtype=np.float64)[:, :n_pc]
    assert a.shape == b.shape and a.shape[1] == n_pc, (
        f"trajectories {a.shape} and {b.shape} do not cover {n_pc} PCs")
    return np.linalg.norm((a - b) * scale, axis=1)
