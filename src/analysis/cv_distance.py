"""Distances between two outcomes' trial means that do not reward small samples.

The distance the figures use is the length of the difference between two
trial means. A trial mean carries an error, and the length of a difference
grows with that error whichever way it points:

    E[|m_A - m_B|^2] = |mu_A - mu_B|^2 + tr(Sigma_A)/n_A + tr(Sigma_B)/n_B

where ``mu`` is the mean the trials are drawn around, ``Sigma`` the
trial-to-trial covariance in the PC space, and ``n`` the number of trials.
The last two terms are an upward bias, not noise: averaging sessions does
not remove them. They matter here because the trial counts are not a
constant of the experiment — misses fall and failures rise over training
(Figure S1) — so a distance measured this way drifts with training even if
nothing about the brain does.

Three estimators that do not, each with a different cost:

- :func:`cross_validated_squared_distance` splits the trials in two and
  takes the inner product of two independently estimated difference
  vectors. The errors of the two halves are independent, so their product
  has expectation zero and the bias never forms. This is the cross-validated
  (``crossnobis``-style) distance of Walther et al. (2016), without the
  multivariate noise normalization — the cross-validated *Euclidean*
  distance that Guggenmos et al. (2018) recommend as a default.
- :func:`bias_corrected_squared_distance` instead estimates the two bias
  terms from the trial variance and subtracts them. Same target, different
  route, so the two agreeing is a check on both.
- :func:`subsampled_squared_distance` draws the same number of trials from
  every session, which leaves the bias in place but makes it equal
  everywhere, so it cannot masquerade as an effect of training.

All three return a **squared** distance, which the first two can report as
negative when the outcomes do not really differ. That is the estimators
working: an unbiased estimate of a quantity near zero has to be able to
fall below it. They are not interchangeable with the distance the main
figures plot, and belong beside it as a check rather than in place of it.

:func:`subsampled_distance` is the matched-count idea applied to the
distance itself, not its square: the quantity the main figures plot,
measured on the same number of trials in every session. It is what the
paper's trial-count control uses (Figure 5), because it needs no new
quantity explained. Its bias is no longer additive — a square root sits
over the sum — but it is the same in every session, so it still cannot
pass for an effect of training.
"""

from __future__ import annotations

import numpy as np

#: Components the distance is taken over, as in :func:`analysis.global_pca.pc_distance`.
N_PC_DISTANCE = 3

#: Random splits averaged over by default. Beyond about ten the estimate
#: stops steadying: what is left is the trial noise itself, not the luck of
#: the split.
DEFAULT_SPLITS = 10

#: Trials each outcome needs for a split to leave something on both sides.
MIN_TRIALS_FOR_SPLIT = 2


def _as_trials(Z: np.ndarray, n_pc: int, name: str) -> np.ndarray:
    """Check one outcome's trials and cut them down to ``n_pc`` components."""
    Z = np.asarray(Z, dtype=np.float64)
    assert Z.ndim == 3, f"{name} should be (n_trials, n_time, n_pc), got {Z.shape}"
    assert Z.shape[0] > 0, f"{name} holds no trials"
    assert Z.shape[2] >= n_pc, f"{name} holds {Z.shape[2]} components, fewer than {n_pc}"
    return Z[:, :, :n_pc]


def naive_squared_distance(
    Z_a: np.ndarray, Z_b: np.ndarray, n_pc: int = N_PC_DISTANCE
) -> np.ndarray:
    """The squared distance between the two trial means, per time point.

    The square of what the figures plot, and the quantity the other
    estimators here are the bias-free versions of.
    """
    a = _as_trials(Z_a, n_pc, "Z_a").mean(axis=0)
    b = _as_trials(Z_b, n_pc, "Z_b").mean(axis=0)
    diff = a - b
    return np.einsum("tp,tp->t", diff, diff)


def trial_variance_bias(
    Z_a: np.ndarray, Z_b: np.ndarray, n_pc: int = N_PC_DISTANCE
) -> np.ndarray:
    """``tr(Sigma_A)/n_A + tr(Sigma_B)/n_B``, estimated from the trials.

    The bias :func:`naive_squared_distance` carries, per time point. The
    trial variance is the usual unbiased one (``ddof=1``), summed over the
    components, so a single trial of either outcome leaves it undefined.
    """
    total = np.zeros(_as_trials(Z_a, n_pc, "Z_a").shape[1])
    for Z, name in ((Z_a, "Z_a"), (Z_b, "Z_b")):
        trials = _as_trials(Z, n_pc, name)
        n = trials.shape[0]
        assert n >= 2, f"{name} holds {n} trial(s); the trial variance needs at least two"
        total = total + trials.var(axis=0, ddof=1).sum(axis=1) / n
    return total


def bias_corrected_squared_distance(
    Z_a: np.ndarray, Z_b: np.ndarray, n_pc: int = N_PC_DISTANCE
) -> np.ndarray:
    """The squared distance with its estimated bias subtracted, per time point."""
    return naive_squared_distance(Z_a, Z_b, n_pc) - trial_variance_bias(Z_a, Z_b, n_pc)


def _halves(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(n)
    return order[: n // 2], order[n // 2:]


def cross_validated_squared_distance(
    Z_a: np.ndarray,
    Z_b: np.ndarray,
    n_pc: int = N_PC_DISTANCE,
    n_splits: int = DEFAULT_SPLITS,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """The cross-validated squared distance, per time point.

    Each outcome's trials are split in two at random; the difference vector
    of one half is dotted with the difference vector of the other. Written
    out, with ``e`` for the error each half's means carry:

        (mu + e1) . (mu + e2) = |mu|^2 + mu.e2 + mu.e1 + e1.e2

    and every term after the first has expectation zero, because ``e1`` and
    ``e2`` come from different trials. ``n_splits`` of these are averaged;
    ``rng`` fixes which splits, so a run repeats.

    An odd number of trials puts the extra one in the second half. Both
    outcomes need at least :data:`MIN_TRIALS_FOR_SPLIT` trials.
    """
    rng = np.random.default_rng(rng)
    a = _as_trials(Z_a, n_pc, "Z_a")
    b = _as_trials(Z_b, n_pc, "Z_b")
    for trials, name in ((a, "Z_a"), (b, "Z_b")):
        assert trials.shape[0] >= MIN_TRIALS_FOR_SPLIT, (
            f"{name} holds {trials.shape[0]} trial(s); a split needs "
            f"at least {MIN_TRIALS_FOR_SPLIT}")
    assert n_splits >= 1, "n_splits must be at least one"

    total = np.zeros(a.shape[1])
    for _ in range(n_splits):
        a1, a2 = _halves(a.shape[0], rng)
        b1, b2 = _halves(b.shape[0], rng)
        u = a[a1].mean(axis=0) - b[b1].mean(axis=0)
        v = a[a2].mean(axis=0) - b[b2].mean(axis=0)
        total += np.einsum("tp,tp->t", u, v)
    return total / n_splits


def subsampled_squared_distance(
    Z_a: np.ndarray,
    Z_b: np.ndarray,
    n_trials: int,
    n_pc: int = N_PC_DISTANCE,
    n_repeats: int = DEFAULT_SPLITS,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """The squared distance from ``n_trials`` of each outcome, per time point.

    Drawn without replacement and averaged over ``n_repeats`` draws. The
    bias is still there — it is ``tr(Sigma_A)/n + tr(Sigma_B)/n`` now — but
    it no longer varies with how many trials a session happened to collect,
    which is what lets the training-day slope be read without it. Sessions
    holding fewer than ``n_trials`` of either outcome cannot be measured
    this way and are the price of the method.
    """
    total = sum(np.einsum("tp,tp->t", u, u)
                for u in _subsample_differences(Z_a, Z_b, n_trials, n_pc, n_repeats, rng))
    return total / n_repeats


def subsampled_distance(
    Z_a: np.ndarray,
    Z_b: np.ndarray,
    n_trials: int,
    n_pc: int = N_PC_DISTANCE,
    n_repeats: int = DEFAULT_SPLITS,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """The distance — not squared — from ``n_trials`` of each outcome, per time point.

    Each draw takes ``n_trials`` of each outcome without replacement and
    measures the distance between their means, as
    :func:`analysis.global_pca.pc_distance` does for all of them; the
    ``n_repeats`` draws' distances are then averaged. Averaging distances,
    not the trials, is what keeps every draw at the bias of ``n_trials``
    trials, and so every session at the same bias. Sessions holding fewer
    than ``n_trials`` of either outcome cannot be measured this way.
    """
    total = sum(np.sqrt(np.einsum("tp,tp->t", u, u))
                for u in _subsample_differences(Z_a, Z_b, n_trials, n_pc, n_repeats, rng))
    return total / n_repeats


def _subsample_differences(
    Z_a: np.ndarray,
    Z_b: np.ndarray,
    n_trials: int,
    n_pc: int,
    n_repeats: int,
    rng: np.random.Generator | int | None,
):
    """The difference of the two subsample means, once per draw.

    Shared by the squared and the plain subsampled distance, so the two draw
    the same trials from the same seed.
    """
    rng = np.random.default_rng(rng)
    a = _as_trials(Z_a, n_pc, "Z_a")
    b = _as_trials(Z_b, n_pc, "Z_b")
    for trials, name in ((a, "Z_a"), (b, "Z_b")):
        assert trials.shape[0] >= n_trials, (
            f"{name} holds {trials.shape[0]} trials, fewer than the {n_trials} asked for")
    for _ in range(n_repeats):
        yield (a[rng.choice(a.shape[0], n_trials, replace=False)].mean(axis=0)
               - b[rng.choice(b.shape[0], n_trials, replace=False)].mean(axis=0))
