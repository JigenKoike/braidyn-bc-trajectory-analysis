"""Tests for the variance inflation factor and within-group centring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.collinearity import center_within, variance_inflation


def test_unrelated_predictors_have_a_vif_near_one() -> None:
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.normal(size=(5000, 4)), columns=list("abcd"))
    assert variance_inflation(data).between(0.99, 1.01).all()


def test_the_vif_is_one_over_one_minus_r_squared() -> None:
    # b is a plus noise of the same size: R² of b on a is 0.5, so VIF = 2,
    # and the same for a on b.
    rng = np.random.default_rng(1)
    a = rng.normal(size=200_000)
    data = pd.DataFrame({"a": a, "b": a + rng.normal(size=a.size)})
    assert variance_inflation(data).to_numpy() == pytest.approx([2.0, 2.0], rel=0.02)


def test_a_predictor_the_others_make_exactly_is_infinite() -> None:
    # The response rate is the success rate plus the failure rate; given both,
    # it says nothing new.
    rng = np.random.default_rng(2)
    success, failure = rng.uniform(0, 0.5, 100), rng.uniform(0, 0.4, 100)
    data = pd.DataFrame({"success": success, "failure": failure, "response": success + failure})
    assert np.isinf(variance_inflation(data)["response"])


def test_the_vif_does_not_care_about_scale() -> None:
    rng = np.random.default_rng(3)
    data = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    data["c"] += data["a"]
    scaled = data * [1.0, 1000.0, 0.001]
    pd.testing.assert_series_equal(variance_inflation(data), variance_inflation(scaled))


def test_rows_with_a_missing_value_are_left_out_of_every_vif() -> None:
    rng = np.random.default_rng(4)
    data = pd.DataFrame(rng.normal(size=(100, 3)), columns=list("abc"))
    with_gap = data.copy()
    with_gap.loc[5, "b"] = np.nan
    pd.testing.assert_series_equal(variance_inflation(with_gap), variance_inflation(data.drop(index=5)))


def test_centring_leaves_only_the_variation_inside_each_group() -> None:
    data = pd.DataFrame({"x": [1.0, 3.0, 10.0, 14.0], "y": [5.0, 5.0, 0.0, 2.0]})
    group = pd.Series(["a", "a", "b", "b"])
    out = center_within(data, ["x", "y"], group)
    np.testing.assert_allclose(out["x"], [-1, 1, -2, 2])
    np.testing.assert_allclose(out["y"], [0, 0, -1, 1])


def test_a_correlation_carried_by_group_means_disappears_within_groups() -> None:
    # Two mice: one high on both x and y, one low on both. Across sessions x
    # and y correlate strongly; inside each mouse they do not at all.
    rng = np.random.default_rng(5)
    rows = []
    for mouse, level in (("hi", 10.0), ("lo", -10.0)):
        for _ in range(500):
            rows.append({"mouse": mouse, "x": level + rng.normal(), "y": level + rng.normal()})
    data = pd.DataFrame(rows)
    assert data[["x", "y"]].corr().iloc[0, 1] > 0.95
    within = center_within(data, ["x", "y"], data["mouse"])
    assert abs(within.corr().iloc[0, 1]) < 0.1
