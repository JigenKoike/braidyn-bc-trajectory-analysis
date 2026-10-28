"""Tests for `analysis.outcomes`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.outcomes import relabel_outcomes_to_paper


def test_basic_relabel() -> None:
    trials = pd.DataFrame({
        "trial_outcome": ["success", "miss", "miss", "success", "miss"],
        "pull_onset": [1.0, np.nan, 5.0, 7.0, np.nan],
    })
    out = relabel_outcomes_to_paper(trials)
    assert out["trial_outcome_paper"].tolist() == [
        "success",  # original success unchanged
        "miss",     # no pull → true miss
        "failure",  # pull but not success → failure
        "success",  # original success unchanged
        "miss",     # no pull → true miss
    ]


def test_does_not_mutate_input() -> None:
    trials = pd.DataFrame({
        "trial_outcome": ["success", "miss"],
        "pull_onset": [1.0, np.nan],
    })
    _ = relabel_outcomes_to_paper(trials)
    assert "trial_outcome_paper" not in trials.columns


def test_missing_columns_no_crash() -> None:
    # Resting-state sessions have no trials at all, but if someone passes
    # a partial DataFrame, the function should not crash.
    trials = pd.DataFrame({"x": [1, 2, 3]})
    out = relabel_outcomes_to_paper(trials)
    assert "trial_outcome_paper" in out.columns


def test_only_outcome_column() -> None:
    # If pull_onset is missing but trial_outcome exists, just copy it.
    trials = pd.DataFrame({"trial_outcome": ["success", "miss"]})
    out = relabel_outcomes_to_paper(trials)
    assert out["trial_outcome_paper"].tolist() == ["success", "miss"]


def test_all_success() -> None:
    trials = pd.DataFrame({
        "trial_outcome": ["success"] * 5,
        "pull_onset": [0.1, 0.2, np.nan, 0.4, np.nan],  # pull_onset irrelevant for success
    })
    out = relabel_outcomes_to_paper(trials)
    assert (out["trial_outcome_paper"] == "success").all()
