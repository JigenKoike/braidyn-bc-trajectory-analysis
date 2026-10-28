"""Trial outcome relabeling following the BraiDyn-BC paper convention.

The paper's convention separates "no attempt" (miss) from "attempt but
failed" (failure) by whether the animal produced a pull. The rules below
apply it to the NWB ``trial_outcome`` column. The released BraiDyn-BC files
already carry ``success`` / ``failure`` / ``miss`` labels that follow it:
over all 48,549 task trials the rules change no label (checked 2026-10-23),
so they act as a check that the convention holds rather than as a fix.

This mirrors the sandbox notebooks' ``fix_outcome_labels_to_paper`` helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def relabel_outcomes_to_paper(
    trials: pd.DataFrame,
    out_col: str = "trial_outcome_paper",
) -> pd.DataFrame:
    """Return a copy of ``trials`` with a paper-style outcome column added.

    Rules (applied to the existing ``trial_outcome`` column):

      - ``"success"`` → ``"success"`` (unchanged)
      - not success AND ``pull_onset`` is NaN → ``"miss"`` (no attempt)
      - not success AND ``pull_onset`` is set → ``"failure"`` (attempt that failed)

    If the required columns aren't present (e.g. resting-state sessions),
    the new column falls back to the existing ``trial_outcome`` value or NaN.
    """
    trials = trials.copy()
    if "pull_onset" not in trials.columns or "trial_outcome" not in trials.columns:
        trials[out_col] = trials.get("trial_outcome", np.nan)
        return trials

    out = trials["trial_outcome"].astype(str).copy()
    has_pull = ~trials["pull_onset"].isna()
    is_success = (out == "success")
    out.loc[~has_pull & ~is_success] = "miss"
    out.loc[has_pull & ~is_success] = "failure"
    trials[out_col] = out
    return trials
