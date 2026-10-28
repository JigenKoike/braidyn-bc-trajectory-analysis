"""Tests for the session-level predictor table the mixed models are fitted on."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis.lmm import PREDICTORS
from analysis.predictors import session_predictors, standardize

MOUSE = "VG1-GC#105"


def _session_id(day: int) -> str:
    return f"sub-{MOUSE}_ses-2024-01-{23 + day}-task-day{day}_widefield+behavior"


@pytest.fixture
def paper_data(tmp_path: Path) -> Path:
    """A miniature of what build_paper_dataset.py writes: two sessions, stacked."""
    # Day 1: 4 trials, 2 success, 1 miss.  Day 2: 3 trials, 1 success, 2 miss.
    sessions = {
        1: (["success", "miss", "failure", "success"], [0.1, 0.3, 0.5, 0.7],
            [0.20, 0.20, 0.25, 0.30]),
        2: (["miss", "miss", "success"], [0.2, 0.4, 0.6], [0.4, 0.4, 0.4]),
    }
    trials, pulls = [], []
    for day, (outcomes, first_pull, t_pull) in sessions.items():
        n = len(outcomes)
        trials.append(pd.DataFrame({
            "session_id": _session_id(day), "mouse": MOUSE, "day": day,
            "start_time": range(n),
            "trial_outcome": outcomes,
            "pull_duration_for_success": t_pull,
        }))
        pulls.append(pd.DataFrame({
            "session_id": _session_id(day), "mouse": MOUSE, "day": day,
            "n_presses": [1] * n,
            "pull_dur_first": first_pull,
        }))
    pd.concat(trials, ignore_index=True).to_parquet(tmp_path / "trials.parquet", index=False)
    pd.concat(pulls, ignore_index=True).to_parquet(tmp_path / "pull_duration.parquet", index=False)
    pd.DataFrame({
        "session_id": [_session_id(1), _session_id(2)],
        "F_blue_pre_mean": [10000.0, 12000.0],
        "F_UV_pre_mean": [5000.0, 6000.0],
    }).to_csv(tmp_path / "baseline_F_per_session.csv", index=False)
    return tmp_path


def test_session_predictors_is_one_row_per_session(paper_data: Path) -> None:
    out = session_predictors(paper_data)
    assert len(out) == 2
    assert list(out["day"]) == [1, 2]
    assert set(out["mouse"]) == {MOUSE}
    assert set(PREDICTORS) <= set(out.columns)


def test_session_predictors_follow_the_manuscript_definitions(paper_data: Path) -> None:
    out = session_predictors(paper_data).set_index("day")
    # Day 1: 4 trials, 2 success, 1 miss.
    assert out.loc[1, "success_rate_v1"] == pytest.approx(0.5)
    assert out.loc[1, "response_rate"] == pytest.approx(0.75)  # one minus the miss rate
    assert out.loc[1, "pull_dur_first_median"] == pytest.approx(0.4)
    assert out.loc[1, "T_pull_final"] == pytest.approx(0.30)  # the last trial's requirement
    # Day 2: 3 trials, 1 success, 2 miss.
    assert out.loc[2, "response_rate"] == pytest.approx(1 / 3)
    assert out.loc[2, "F_blue_pre"] == pytest.approx(12000.0)


def test_session_predictors_keep_the_sessions_apart(paper_data: Path) -> None:
    # The two sessions arrive in one table; a measure must not be pooled
    # across them. Day 2's median first pull is its own three trials', 0.4.
    out = session_predictors(paper_data).set_index("day")
    assert out.loc[2, "pull_dur_first_median"] == pytest.approx(0.4)
    assert out.loc[1, "n_trials"] == 4
    assert out.loc[2, "n_trials"] == 3


def test_session_predictors_refuse_a_session_out_of_time_order(paper_data: Path) -> None:
    # T_pull_final is read off each session's last row.
    trials = pd.read_parquet(paper_data / "trials.parquet")
    first = trials["session_id"] == _session_id(1)
    trials.loc[first, "start_time"] = trials.loc[first, "start_time"][::-1].to_numpy()
    trials.to_parquet(paper_data / "trials.parquet", index=False)
    with pytest.raises(AssertionError, match="time order"):
        session_predictors(paper_data)


def test_session_predictors_refuse_a_session_with_no_baseline(paper_data: Path) -> None:
    one_row = pd.read_csv(paper_data / "baseline_F_per_session.csv").iloc[:1]
    one_row.to_csv(paper_data / "baseline_F_per_session.csv", index=False)
    with pytest.raises(AssertionError, match="missing"):
        session_predictors(paper_data)


def test_standardize_centres_and_scales_every_predictor() -> None:
    raw = pd.DataFrame({p: [float(i), 2.0 * i + 1, 3.0 - i] for i, p in enumerate(PREDICTORS, 1)}
                       | {"mouse": ["a", "b", "c"]})
    z, mean, sd = standardize(raw)
    for p in PREDICTORS:
        assert z[p].mean() == pytest.approx(0.0, abs=1e-12)
        assert z[p].std(ddof=1) == pytest.approx(1.0)
        assert mean[p] == pytest.approx(raw[p].mean())
        assert sd[p] == pytest.approx(raw[p].std(ddof=1))
    assert list(z["mouse"]) == ["a", "b", "c"]  # the other columns are left alone


def test_standardize_refuses_a_predictor_that_does_not_vary() -> None:
    raw = pd.DataFrame({p: [1.0, 2.0, 3.0] for p in PREDICTORS})
    raw[PREDICTORS[2]] = 1.0
    with pytest.raises(AssertionError, match=PREDICTORS[2]):
        standardize(raw)
