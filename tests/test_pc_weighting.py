"""Tests for the variance- and equally weighted PC distances."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.global_pca import pc_distance
from analysis.pc_weighting import equal_weight_scale, weighted_pc_distance


def _trajectories(seed: int = 0, n_time: int = 40, n_pc_total: int = 6):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_time, n_pc_total)), rng.normal(size=(n_time, n_pc_total))


def test_unit_scale_is_the_papers_distance() -> None:
    a, b = _trajectories()
    np.testing.assert_allclose(weighted_pc_distance(a, b, np.ones(3)), pc_distance(a, b, n_pc=3))


def test_equal_weight_is_projecting_with_unit_length_weights() -> None:
    # The coordinates are activity × W. Dividing each by its column's length
    # has to give exactly what activity × (W with unit columns) gives.
    rng = np.random.default_rng(1)
    W = rng.normal(size=(44, 20)) * np.array([3.0, 1.0, 0.3] + [0.1] * 17)
    x_a, x_b = rng.normal(size=(30, 44)), rng.normal(size=(30, 44))
    scale = equal_weight_scale(W)
    W_unit = W[:, :3] / np.linalg.norm(W[:, :3], axis=0)
    expected = np.linalg.norm(x_a @ W_unit - x_b @ W_unit, axis=1)
    np.testing.assert_allclose(weighted_pc_distance(x_a @ W, x_b @ W, scale), expected)


def test_the_scale_undoes_each_columns_length() -> None:
    W = np.zeros((4, 3))
    W[0, 0], W[1, 1], W[2, 2] = 2.0, 0.5, 0.1
    np.testing.assert_allclose(equal_weight_scale(W), [0.5, 2.0, 10.0])


def test_only_the_first_pcs_enter_the_distance() -> None:
    a, b = _trajectories()
    b2 = b.copy()
    b2[:, 3:] += 100.0   # components past the third must not matter
    np.testing.assert_allclose(weighted_pc_distance(a, b, np.ones(3)),
                               weighted_pc_distance(a, b2, np.ones(3)))


def test_a_pc_without_weights_is_refused() -> None:
    W = np.ones((5, 3))
    W[:, 2] = 0.0
    with pytest.raises(AssertionError, match="all-zero"):
        equal_weight_scale(W)
