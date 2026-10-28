"""The session-level predictors the mixed models are fitted on.

One row per task session: three behavioural measures read from the trial
table, one from the pull-duration table, and the two pre-cue fluorescence
baselines. :data:`analysis.lmm.PREDICTORS` names them and fixes their order.

This lives here rather than in a notebook because the early-window and
late-window figures both need it, and the one thing that must not differ
between those two is how the predictors were built.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .lmm import PREDICTORS
from .outcomes import relabel_outcomes_to_paper


def session_predictors(paper_data: Path) -> pd.DataFrame:
    """One row per task session, in raw units.

    ``paper_data`` is the folder ``scripts/build_paper_dataset.py`` writes —
    ``data/paper`` — holding ``trials.parquet``, ``pull_duration.parquet``
    and ``baseline_F_per_session.csv``. The first two are trial by trial,
    every session stacked into one table and keyed by ``session_id``; this
    reduces them to one row each.

    The behavioural definitions follow the manuscript's §2.3:
    ``response_rate`` is one minus the miss rate, and ``T_pull_final`` is the
    pull duration required at the session's last trial.
    """
    paper_data = Path(paper_data)
    trials = relabel_outcomes_to_paper(pd.read_parquet(paper_data / "trials.parquet"))
    pull = pd.read_parquet(paper_data / "pull_duration.parquet")
    assert trials["trial_outcome_paper"].isin(["success", "miss", "failure"]).all()
    # T_pull_final is read off each session's last row, which is its last
    # trial only if the table is in time order.
    assert trials.groupby("session_id")["start_time"].apply(
        lambda s: s.is_monotonic_increasing).all(), "a session's trials are out of time order"

    first_pull = (pull[pull["n_presses"] >= 1]
                  .groupby("session_id")["pull_dur_first"].median()
                  .rename("pull_dur_first_median"))
    behaviour = (trials.groupby("session_id", sort=True)
                 .apply(lambda g: pd.Series({
                     "mouse": g["mouse"].iloc[0],
                     "day": int(g["day"].iloc[0]),
                     "n_trials": len(g),
                     "success_rate_v1": (g["trial_outcome_paper"] == "success").sum() / len(g),
                     "response_rate": 1 - (g["trial_outcome_paper"] == "miss").sum() / len(g),
                     "T_pull_final": float(g["pull_duration_for_success"].iloc[-1]),
                 }), include_groups=False)
                 .reset_index())

    baseline = (pd.read_csv(paper_data / "baseline_F_per_session.csv")
                [["session_id", "F_blue_pre_mean", "F_UV_pre_mean"]]
                .rename(columns={"F_blue_pre_mean": "F_blue_pre", "F_UV_pre_mean": "F_UV_pre"}))
    out = (behaviour.merge(first_pull, on="session_id", how="left")
                    .merge(baseline, on="session_id", how="left"))
    missing = out[list(PREDICTORS)].isna().sum()
    assert not missing.any(), f"predictors missing in some sessions: {missing[missing > 0].to_dict()}"
    return out


def standardize(
    predictors_raw: pd.DataFrame, predictors: Sequence[str] = PREDICTORS
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """z-standardize the predictors and return them with the mean and SD used.

    Over **every** session in ``predictors_raw``, not over the subset a given
    outcome pair happens to have: a coefficient then means the same thing in
    all three models, which is what the manuscript's §2.3 describes and what
    lets the three be printed in one table.

    Returns ``(standardized, mean, sd)``, the two in raw units so the tables
    can record what a standard deviation was.
    """
    predictors = list(predictors)
    mean = predictors_raw[predictors].mean()
    sd = predictors_raw[predictors].std(ddof=1)
    assert (sd > 0).all(), f"these predictors do not vary: {list(sd[sd <= 0].index)}"
    out = predictors_raw.copy()
    out[predictors] = (predictors_raw[predictors] - mean) / sd
    return out, mean, sd
