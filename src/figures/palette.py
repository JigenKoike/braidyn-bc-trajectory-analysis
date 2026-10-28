"""Colours and display names, so every figure names and colours things alike.

The three palettes are deliberately kept apart, because a reader who sees the
same hue in two panels should be seeing the same thing:

- :data:`PAIR_COLOR` — the outcome *pairs*, blue / green / orange, matching
  the ordering in the manuscript's appendix note.
- :data:`OUTCOME_COLOR` — the individual *outcomes*. Red for success, purple
  for failure, neutral grey for a miss (the trial with no response). Red and
  purple are both matplotlib tab10 hues, familiar from countless figures, and
  neither hue family is used by the pair palette. Every pair separates by
  ΔE ≥ 11 under protanopia and deuteranopia simulation (Machado 2009, in
  OKLab) and by ΔE ≥ 21 in normal vision, and all three clear 3:1 contrast
  against the page.
- :data:`PC_COLORS` — the principal components, black / purple / brown, again
  picked to clash with neither of the above.

Two known limitations, both checked and accepted:

- Under protanopia simulation the pair colours green and orange come out
  nearly the same (ΔE 0.7). The colours stay as they are, so figures that
  rely on telling those two apart should also differ in line style or carry
  a direct label.
- The pair blue and the outcome grey sit at ΔE 13 in normal vision, just
  under the usual floor of 15. They never share a plot — the outcomes appear
  in one panel and the pairs in another — so no reader has to tell them
  apart.
"""

from __future__ import annotations

from analysis.pairs import PAIR_ORDER

#: Outcome pairs. Blue, green, orange.
PAIR_COLOR: dict[str, str] = {
    "success-miss": "#1f77b4",
    "failure-miss": "#2ca02c",
    "success-failure": "#ff7f0e",
}

#: Pair names as written in figures, with an en dash.
PAIR_LABEL: dict[str, str] = {
    "success-miss": "success–miss",
    "failure-miss": "failure–miss",
    "success-failure": "success–failure",
}

#: The same, broken over two lines for narrow tick labels.
PAIR_LABEL_2LINE: dict[str, str] = {
    "success-miss": "success–\nmiss",
    "failure-miss": "failure–\nmiss",
    "success-failure": "success–\nfailure",
}

#: Individual trial outcomes. Red, purple, neutral grey.
OUTCOME_COLOR: dict[str, str] = {
    "success": "#d62728",
    "failure": "#9467bd",
    "miss": "#7f7f7f",
}

#: Imaging channels: the Ca-sensitive blue and the Ca-insensitive UV.
CHANNEL_COLOR: dict[str, str] = {"blue": "#1f77b4", "UV": "#9467bd"}

#: PC1, PC2, PC3 as lines or bars.
PC_COLORS: tuple[str, ...] = ("black", "tab:purple", "tab:brown")

#: Sequential colour map, for quantities that only run one way.
SEQUENTIAL_CMAP = "viridis"

#: Diverging colour map for signed quantities, with zero at the pale middle.
#: PC weights are signed and their sign is the point, so a sequential map
#: would hide it: under viridis zero lands on a mid green that reads as a
#: value rather than as nothing.
DIVERGING_CMAP = "RdBu_r"

#: Markers distinguishing the analysis windows where both are plotted.
WINDOW_MARKER: dict[str, str] = {"early": "o", "late": "s"}

#: Window names as written in figures.
WINDOW_LABEL: dict[str, str] = {
    "pre": "Pre (−1–0 s)",
    "early": "Early (0.2–2.0 s)",
    "late": "Late (2.0–4.0 s)",
}

#: Axis label for the outcome measure. Spelled out because "PC1-PC3
#: distance" reads as a difference between two components; the quantity is
#: a Euclidean distance in the space the first three components span.
DISTANCE_LABEL = "Distance in PC1\u20133 space (a.u.)"

#: Model predictors as written in figures and tables.
PRED_DISPLAY: dict[str, str] = {
    "day": "Training day",   # the manuscript's word for it (co-authors, 2026-10-26)
    "pull_dur_first_median": "First-pull duration",
    "response_rate": "Response rate",
    "success_rate_v1": "Success rate",
    "F_blue_pre": "Baseline F (blue)",
    "F_UV_pre": "Baseline F (UV)",
    "T_pull_final": "T_pull final",
}

#: The two behavioural predictors added before submission
#: (``analysis.behavior_covariates.BEHAVIOR_PREDICTORS``), kept apart from
#: :data:`PRED_DISPLAY`, which names exactly the paper's seven.
BEHAVIOR_DISPLAY: dict[str, str] = {
    "motion": "Body motion",
    "pupil": "Pupil size",
}

assert set(PAIR_COLOR) == set(PAIR_LABEL) == set(PAIR_LABEL_2LINE) == set(PAIR_ORDER), (
    "the pair palettes and analysis.pairs.PAIR_ORDER have drifted apart"
)
