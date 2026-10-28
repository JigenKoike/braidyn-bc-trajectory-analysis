"""Tests for `analysis.multiple_comparison`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.multiple_comparison import (
    add_corrected_column,
    benjamini_hochberg,
    holm,
)


def test_holm_multiplies_the_smallest_p_by_the_family_size() -> None:
    # Holm's first step is Bonferroni on the smallest p; later steps use
    # shrinking multipliers and the result is kept non-decreasing.
    p = np.array([0.01, 0.02, 0.03, 0.04])
    np.testing.assert_allclose(holm(p), [0.04, 0.06, 0.06, 0.06])


def test_holm_never_lowers_a_p_value() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20)
    assert (holm(p) >= p - 1e-12).all()


def test_benjamini_hochberg_leaves_the_largest_p_alone() -> None:
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.900])
    q = benjamini_hochberg(p)
    assert q[-1] == pytest.approx(0.900)
    assert (q >= p - 1e-12).all()


def test_benjamini_hochberg_is_never_stricter_than_holm() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(size=30)
    assert (benjamini_hochberg(p) <= holm(p) + 1e-12).all()


def test_corrections_are_monotone_in_the_original_p_values() -> None:
    # A smaller p never ends up with a larger corrected value. Equality is
    # expected rather than a strict ranking: both methods clamp at 1, and
    # Holm clamps most of a null family there.
    rng = np.random.default_rng(2)
    p = rng.uniform(size=15)
    for correct in (holm, benjamini_hochberg):
        order = np.argsort(p)
        corrected = correct(p)[order]
        assert (np.diff(corrected) >= -1e-12).all()


def _table() -> pd.DataFrame:
    return pd.DataFrame({
        "pair": ["a"] * 3 + ["b"] * 3,
        "predictor": ["x", "y", "z"] * 2,
        "p": [0.01, 0.02, 0.03, 0.40, 0.50, 0.60],
        "tail": [1, 2, 3, 4, 5, 6],
    })


def test_add_corrected_column_sits_right_after_the_p_column() -> None:
    t = add_corrected_column(_table(), "p", "q", within="pair")
    assert list(t.columns) == ["pair", "predictor", "p", "q", "tail"]


def test_add_corrected_column_corrects_within_each_family_separately() -> None:
    t = add_corrected_column(_table(), "p", "q", within="pair")
    for pair in ("a", "b"):
        sub = t[t["pair"] == pair]
        np.testing.assert_allclose(sub["q"].to_numpy(), benjamini_hochberg(sub["p"].to_numpy()))


def test_a_family_per_pair_differs_from_one_family_over_everything() -> None:
    per_pair = add_corrected_column(_table(), "p", "q", within="pair")["q"].to_numpy()
    pooled = add_corrected_column(_table(), "p", "q")["q"].to_numpy()
    assert not np.allclose(per_pair, pooled)


def test_holm_is_available_through_the_same_entry_point() -> None:
    t = add_corrected_column(_table(), "p", "p_holm", within="pair", method="holm")
    sub = t[t["pair"] == "a"]
    np.testing.assert_allclose(sub["p_holm"].to_numpy(), holm(sub["p"].to_numpy()))
