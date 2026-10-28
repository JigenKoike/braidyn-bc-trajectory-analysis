"""Tests for the trial-count-free distance estimators.

The claim being tested is one about expectations, so most of these draw many
synthetic sessions from a known truth and ask where the estimates centre.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.cv_distance import (
    bias_corrected_squared_distance,
    cross_validated_squared_distance,
    naive_squared_distance,
    subsampled_distance,
    subsampled_squared_distance,
    trial_variance_bias,
)

# Each time point of a session is an independent draw, so the number of them
# sets how finely these tests can locate an expectation; twelve keeps the
# Monte Carlo error under the tolerances below without slowing the suite.
N_TIME, N_PC, SIGMA = 12, 3, 1.0


def _draw(n_a: int, n_b: int, true_distance: float, rng: np.random.Generator):
    """Trials of two outcomes separated by ``true_distance`` along PC1."""
    a = rng.normal(0, SIGMA, (n_a, N_TIME, N_PC))
    b = rng.normal(0, SIGMA, (n_b, N_TIME, N_PC))
    a[:, :, 0] += true_distance
    return a, b


def _mean_over_sessions(estimator, n_a, n_b, true_distance, n_sessions=4000, seed=0):
    """Where ``estimator(trials_a, trials_b, rng)`` centres over many sessions."""
    rng = np.random.default_rng(seed)
    values = [estimator(*_draw(n_a, n_b, true_distance, rng), rng)
              for _ in range(n_sessions)]
    return float(np.mean(values))


def _no_rng(estimator):
    """An estimator that takes no ``rng``, called the same way as one that does."""
    return lambda a, b, _rng: estimator(a, b)


def _with_rng(estimator):
    return lambda a, b, rng: estimator(a, b, rng=rng)


def test_the_naive_distance_is_biased_by_the_predicted_amount() -> None:
    # The bias the module exists for: tr(Sigma_A)/n_A + tr(Sigma_B)/n_B.
    n_a, n_b, true_d = 58, 19, 1.0
    predicted = N_PC * SIGMA**2 * (1 / n_a + 1 / n_b)
    found = _mean_over_sessions(
        _no_rng(naive_squared_distance), n_a, n_b, true_d) - true_d**2
    assert found == pytest.approx(predicted, rel=0.05)


@pytest.mark.parametrize("n_b", [10, 19, 100])
def test_the_cross_validated_distance_is_unbiased_whatever_the_trial_count(n_b: int) -> None:
    found = _mean_over_sessions(
        _with_rng(cross_validated_squared_distance), 58, n_b, 1.0)
    assert found == pytest.approx(1.0, abs=0.03)


@pytest.mark.parametrize("n_b", [10, 19, 100])
def test_the_bias_corrected_distance_is_unbiased_whatever_the_trial_count(n_b: int) -> None:
    found = _mean_over_sessions(
        _no_rng(bias_corrected_squared_distance), 58, n_b, 1.0)
    assert found == pytest.approx(1.0, abs=0.03)


def test_both_estimators_centre_on_zero_when_the_outcomes_do_not_differ() -> None:
    # The property the main figures' distance does not have: no separation
    # means no distance, rather than a distance that grows as trials get few.
    for estimator in (_with_rng(cross_validated_squared_distance),
                      _no_rng(bias_corrected_squared_distance)):
        assert _mean_over_sessions(estimator, 58, 10, 0.0) == pytest.approx(0.0, abs=0.02)
    assert _mean_over_sessions(_no_rng(naive_squared_distance), 58, 10, 0.0) > 0.3


def test_the_correction_is_the_naive_distance_less_the_estimated_bias() -> None:
    rng = np.random.default_rng(1)
    a, b = _draw(30, 12, 0.5, rng)
    np.testing.assert_allclose(
        bias_corrected_squared_distance(a, b),
        naive_squared_distance(a, b) - trial_variance_bias(a, b),
    )


def test_a_seed_makes_the_splits_repeat() -> None:
    rng = np.random.default_rng(2)
    a, b = _draw(20, 9, 0.5, rng)
    first = cross_validated_squared_distance(a, b, rng=7)
    np.testing.assert_allclose(first, cross_validated_squared_distance(a, b, rng=7))
    assert not np.allclose(first, cross_validated_squared_distance(a, b, rng=8))


def test_subsampling_to_every_trial_is_the_naive_distance() -> None:
    # With nothing left to choose, the subsample is the whole sample.
    rng = np.random.default_rng(3)
    a, b = _draw(8, 8, 0.7, rng)
    np.testing.assert_allclose(
        subsampled_squared_distance(a, b, n_trials=8, n_repeats=3, rng=0),
        naive_squared_distance(a, b),
    )


def test_subsampling_gives_every_session_the_same_bias() -> None:
    # The point of it: two sessions with very different trial counts come out
    # carrying the same bias, so the difference between them cannot be one of
    # sample size. Both are measured at n = 10.
    few = _mean_over_sessions(
        lambda a, b, rng: subsampled_squared_distance(a, b, 10, n_repeats=4, rng=rng),
        58, 12, 1.0, n_sessions=2000)
    many = _mean_over_sessions(
        lambda a, b, rng: subsampled_squared_distance(a, b, 10, n_repeats=4, rng=rng),
        58, 100, 1.0, n_sessions=2000)
    assert few == pytest.approx(many, abs=0.05)
    # And that shared bias is the one for ten trials of each, not for all of them.
    assert few == pytest.approx(1.0 + N_PC * SIGMA**2 * (1 / 10 + 1 / 10), rel=0.05)


def test_too_few_trials_to_split_is_an_error_rather_than_a_number() -> None:
    rng = np.random.default_rng(4)
    a, b = _draw(5, 1, 1.0, rng)
    with pytest.raises(AssertionError, match="a split needs"):
        cross_validated_squared_distance(a, b)
    with pytest.raises(AssertionError, match="trial variance needs"):
        trial_variance_bias(a, b)
    with pytest.raises(AssertionError, match="fewer than the 10 asked for"):
        subsampled_squared_distance(a, b, n_trials=10)


def test_the_components_beyond_the_distance_are_left_out() -> None:
    # The arrays carry whatever components they were saved with; the distance
    # is over the first three, as everywhere else in the analysis.
    rng = np.random.default_rng(5)
    a = rng.normal(0, 1.0, (40, N_TIME, 6))
    b = rng.normal(0, 1.0, (40, N_TIME, 6))
    a[:, :, 4] += 50.0  # a huge separation on a component nothing reads
    assert naive_squared_distance(a, b) == pytest.approx(
        naive_squared_distance(a[:, :, :3], b[:, :, :3]))


# --- the plain (not squared) subsampled distance ------------------------------

def test_the_plain_subsampled_distance_to_every_trial_is_the_distance_of_the_means() -> None:
    # With nothing left to choose, every draw is the whole sample: the plain
    # distance the main figures plot.
    rng = np.random.default_rng(4)
    a, b = _draw(9, 9, 0.6, rng)
    np.testing.assert_allclose(
        subsampled_distance(a, b, n_trials=9, n_repeats=3, rng=0),
        np.linalg.norm(a.mean(axis=0) - b.mean(axis=0), axis=1),
    )


def test_the_plain_and_squared_subsampled_distances_draw_the_same_trials() -> None:
    # Same seed, same draws. Averaging distances rather than their squares
    # can only come out lower (Jensen), and equal when every draw is alike.
    rng = np.random.default_rng(5)
    a, b = _draw(30, 14, 0.8, rng)
    plain = subsampled_distance(a, b, 10, n_repeats=6, rng=11)
    squared = subsampled_squared_distance(a, b, 10, n_repeats=6, rng=11)
    assert (plain <= np.sqrt(squared) + 1e-12).all()
    one_plain = subsampled_distance(a, b, 10, n_repeats=1, rng=12)
    one_squared = subsampled_squared_distance(a, b, 10, n_repeats=1, rng=12)
    np.testing.assert_allclose(one_plain, np.sqrt(one_squared))


def test_the_plain_subsampled_distance_repeats_with_a_seed_and_refuses_too_few_trials() -> None:
    rng = np.random.default_rng(6)
    a, b = _draw(20, 12, 0.5, rng)
    np.testing.assert_array_equal(subsampled_distance(a, b, 10, rng=3),
                                  subsampled_distance(a, b, 10, rng=3))
    with pytest.raises(AssertionError, match="fewer than"):
        subsampled_distance(a, b[:9], 10, rng=0)
