# %% [markdown]
# # Figure 5 — Does the training-day effect hold up?
#
# Figures 2–4 rest on three choices a reader could question: how much each
# PC counts in the distance, how many trials each outcome happens to have,
# and whether the mice may differ in how fast they change. This figure makes
# each choice the other way and asks whether the training-day effect stays.
#
# | Panel | Content |
# |---|---|
# | **A** | PC weighting: the day slope of the window area, with the PCs weighted by the variance they explain (as in the paper) and weighted equally; early and late window side by side |
# | **B** | Trial-count control: the same slope with the distance from all trials (as in the paper) and from ten trials of each outcome; early and late window |
# | **C** | Individual variability: the training-day coefficient of the paper's mixed model with a random intercept per mouse (as in Figures 3 and 4) and with a random training-day slope as well; one plot per pair |
#
# **What is computed.**
#
# - *PC weighting.* The PC coordinates keep the scale of the PCA scores, so
#   each PC's weight is the square root of the variance it explains and the
#   distance is decided almost entirely by PC1. Equal weight divides each
#   coordinate by its weight vector's length, so the three PCs count the
#   same (`analysis.pc_weighting`).
# - *Trial count.* A mean of fewer trials sits further from another by chance,
#   and the trial counts change with training. Ten trials of each outcome are
#   drawn without replacement, the distance taken between their means, and
#   the draw repeated ten times and averaged (`analysis.cv_distance`,
#   `scripts/run_trial_subsampling.py`). Every session then carries the
#   chance part of ten trials, and sessions with fewer than ten trials of
#   either outcome drop out.
# - In **A** and **B** each window area is z-scored across the sessions of its pair
#   and measure before the regression on training day — equal weighting and
#   subsampling both change the distance's scale — so the slope reads in SD
#   per day. Rescaling leaves the p value as it was.
# - *Individual variability.* The paper's model (seven predictors, maximum
#   likelihood), with the random part widened to an intercept and a
#   training-day slope per mouse and their covariance
#   (`analysis.lmm_sensitivity`).
#
# **Statistics.** In **A** and **B**, least squares of the z-scored area on training day,
# as in Figure 2, Benjamini-Hochberg over the three pairs of one window and
# one measure; a filled marker is q < 0.05. In **C**, the likelihood-ratio
# test of the day coefficient, Benjamini-Hochberg over the seven predictors of
# that one model, as in Figures 3 and 4; red is q < 0.05. Under the random
# slope the reduced model keeps each mouse's slope, so the test asks whether
# their mean is zero. The intervals are Wald 95% intervals, not corrected.
# The q values are printed at the right of each row.
#
# **Inputs**
#
# All of it from `data/paper/`, which `scripts/build_paper_dataset.py` fills
# from the pipeline's output.
#
# | File | Originally made by |
# |---|---|
# | `trajectories.npz`, `pca/roi_weights.npy`, `pca/peth_time_axis_sec.npy` | `scripts/run_pairwise_pca.py` |
# | `distance_timecourse.csv` | `scripts/run_pairwise_pca.py` |
# | `distance_timecourse_10trials.csv` | `scripts/run_trial_subsampling.py`, from `scripts/extract_trial_projections.py` |
# | `trials.parquet`, `pull_duration.parquet`, `baseline_F_per_session.csv` | the extraction scripts — the predictors |

# %%
import sys
from pathlib import Path

_p = Path.cwd().resolve()
while _p != _p.parent and not (_p / "pyproject.toml").exists():
    _p = _p.parent
REPO_ROOT = _p
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from analysis.cv_distance import N_PC_DISTANCE
from analysis.lmm import PREDICTORS, fit_leave_one_out, lmm_tables
from analysis.lmm_sensitivity import random_slope_lrt
from analysis.multiple_comparison import add_corrected_column
from analysis.pairs import PAIR_ORDER, PAIRS
from analysis.pc_weighting import equal_weight_scale, weighted_pc_distance
from analysis.predictors import session_predictors, standardize
from analysis.regression import auc_per_session, day_regression
from analysis.windows import EARLY_WINDOW, LATE_WINDOW, window_auc
from figures.formatting import decimal_formatter, with_star
from figures.palette import PAIR_COLOR, PAIR_LABEL
from figures.style import (
    LABEL_SIZE,
    TICK_SIZE,
    WIDTH_2COL,
    clean_axes,
    display_path,
    equal_panel_widths,
    PANEL_LABEL_SIZE,
    group_xlabel,
    section_heading,
    save_fig,
    use_paper_style,
)

use_paper_style()

DATA_DIR = REPO_ROOT / "data/paper"
OUT_DIR = REPO_ROOT / "output/paper_figures"
PRESUB_DIR = REPO_ROOT / "output/presubmission"   # earlier runs, to check against

WINDOWS = {"early": EARLY_WINDOW, "late": LATE_WINDOW}
Q_CUTOFF = 0.05
SIG_COLOR = "tab:red"
NULL_COLOR = "0.45"
FOCUS = "day"

print(f"output: {display_path(OUT_DIR)}")

# %% [markdown]
# ## 1. PC weighting
#
# Both distances from the trajectories the pipeline saved. The
# variance-weighted one is the paper's distance, rebuilt here and checked
# against the packaged timecourse.

# %%
W = np.load(DATA_DIR / "pca/roi_weights.npy")
t_axis = np.load(DATA_DIR / "pca/peth_time_axis_sec.npy").astype(np.float64)
traj = np.load(DATA_DIR / "trajectories.npz")
distance_long = pd.read_csv(DATA_DIR / "distance_timecourse.csv")
distance_long = distance_long[distance_long["pair"].isin(PAIR_ORDER)]
measured = distance_long[["mouse", "day", "session_id", "pair"]].drop_duplicates()

SCALES = {"variance weight": np.ones(N_PC_DISTANCE), "equal weight": equal_weight_scale(W)}
print("weight of each PC relative to PC1 under equal weighting: "
      + ", ".join(f"PC{k + 1} ×{SCALES['equal weight'][k] / SCALES['equal weight'][0]:.1f}"
                  for k in range(N_PC_DISTANCE)))

pair_members = dict(zip(PAIR_ORDER, PAIRS))
packaged = {k: g.sort_values("t")["dist"].to_numpy()
            for k, g in distance_long.groupby(["session_id", "pair"])}
rows, worst = [], 0.0
for r in measured.itertuples():
    a, b = pair_members[r.pair]
    za, zb = traj[f"{r.session_id}/{a}"], traj[f"{r.session_id}/{b}"]
    for measure, scale in SCALES.items():
        d = weighted_pc_distance(za, zb, scale)
        if measure == "variance weight":
            ref = packaged[(r.session_id, r.pair)]
            worst = max(worst, float(np.max(np.abs(d - ref) / np.maximum(ref, 1e-12))))
        for window, span in WINDOWS.items():
            rows.append({"mouse": r.mouse, "day": r.day, "session_id": r.session_id,
                         "pair": r.pair, "measure": measure, "window": window,
                         "auc": window_auc(t_axis, d, span)})
pc_areas = pd.DataFrame(rows)
assert worst < 1e-5, f"the paper's distance was not reproduced (relative {worst:.1e})"
print(f"{len(measured)} session-pairs; the variance-weighted distance matches the packaged "
      f"one to {worst:.1e} (relative)")


def zscored_regressions(areas: pd.DataFrame) -> pd.DataFrame:
    """Day regression of the window area, z-scored within pair, per window and measure."""
    fits = []
    for (window, measure), g in areas.groupby(["window", "measure"], sort=False):
        g = g.copy()
        g["z"] = g.groupby("pair")["auc"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
        fits.append(day_regression(g, value_col="z", label={"window": window, "measure": measure}))
    return pd.concat(fits, ignore_index=True)


pc_fits = zscored_regressions(pc_areas)
print(pc_fits[["window", "measure", "pair", "n_sessions", "slope_per_day", "ci95_lo",
               "ci95_hi", "p_value", "q_value"]].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

# The same regressions as the analysis that proposed this panel.
earlier = PRESUB_DIR / "pc_weight_normalization_day_regressions.csv"
if earlier.exists():
    check = pc_fits.merge(pd.read_csv(earlier), on=["window", "measure", "pair"], suffixes=("", "_old"))
    assert len(check) == len(pc_fits)
    for col in ("slope_per_day", "p_value"):
        assert np.allclose(check[col], check[f"{col}_old"], rtol=1e-9, atol=0), col
    print("matches the earlier PC-weighting analysis (slopes and p values)")
else:
    print("the earlier PC-weighting run is not here; skipping the comparison")

# %% [markdown]
# ## 2. Trial count
#
# The paper's distance against the one measured on ten trials of each
# outcome. The ten-trial sessions are a subset of the paper's: a pair is
# measured only where both outcomes have at least ten trials.

# %%
ten = pd.read_csv(DATA_DIR / "distance_timecourse_10trials.csv")
ten = ten[ten["pair"].isin(PAIR_ORDER)]
keys = ["session_id", "pair"]
assert ten[keys].drop_duplicates().merge(measured[keys]).shape[0] == ten[keys].drop_duplicates().shape[0], \
    "a ten-trial session-pair is missing from the paper's distances"

rows = []
for measure, long in (("all trials", distance_long), ("10 trials each", ten)):
    for window, span in WINDOWS.items():
        rows.append(auc_per_session(long, span).assign(measure=measure, window=window))
tc_areas = pd.concat(rows, ignore_index=True)
tc_fits = zscored_regressions(tc_areas)
print(tc_fits[["window", "measure", "pair", "n_sessions", "slope_per_day", "ci95_lo",
               "ci95_hi", "p_value", "q_value"]].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

# "All trials" is the paper's distance: the same rows as the variance weight above.
same = tc_fits[tc_fits["measure"] == "all trials"].merge(
    pc_fits[pc_fits["measure"] == "variance weight"], on=["window", "pair"], suffixes=("", "_pc"))
assert np.allclose(same["slope_per_day"], same["slope_per_day_pc"], rtol=1e-9)
assert np.allclose(same["q_value"], same["q_value_pc"], rtol=1e-9)

# %% [markdown]
# ## 3. Individual variability
#
# The paper's model and its random-slope version on the same sessions, each
# predictor tested by likelihood ratio as in Tables 2 and 3. The random
# intercept goes through the functions that make those tables, so its numbers
# have to be theirs. The random slope keeps each mouse's slope in every
# reduced model (`analysis.lmm_sensitivity.random_slope_lrt`).

# %%
predictors_z, Z_MEAN, Z_SD = standardize(session_predictors(DATA_DIR))
COLUMNS = ["window", "pair", "structure", "predictor", "n_sessions", "beta", "se",
           "ci95_lo", "ci95_hi", "wald_p", "lrt_chi2", "lrt_p",
           "llf_spread_full", "llf_spread_reduced"]
parts = []
for window, span in WINDOWS.items():
    auc = auc_per_session(distance_long, span)
    fitted = {}
    for pair in PAIR_ORDER:
        pair_data = predictors_z.merge(auc.loc[auc["pair"] == pair, ["session_id", "auc"]],
                                       on="session_id")
        data, scale, fits = fit_leave_one_out(pair_data, label={"window": window, "pair": pair})
        fitted[pair] = {"data": data, "scale": scale, "fits": fits}
        rs, rs_lrt = random_slope_lrt(data, PREDICTORS, slope_col=FOCUS)
        ci = rs.conf_int()
        parts.append(rs_lrt.assign(
            window=window, pair=pair, structure="random slope", n_sessions=len(data),
            beta=[float(rs.params[p]) for p in PREDICTORS],
            se=[float(rs.bse[p]) for p in PREDICTORS],
            ci95_lo=[float(ci.loc[p, 0]) for p in PREDICTORS],
            ci95_hi=[float(ci.loc[p, 1]) for p in PREDICTORS],
            wald_p=[float(rs.pvalues[p]) for p in PREDICTORS]))
    model_fit, loo, coefs = lmm_tables(fitted, predictor_mean=Z_MEAN, predictor_sd=Z_SD,
                                       label={"window": window})
    ri = coefs.merge(loo, on=["window", "pair", "predictor"]).merge(
        model_fit[["pair", "n_sessions"]], on="pair")
    parts.append(ri.assign(structure="random intercept"))
RANK = {"window": list(WINDOWS), "pair": list(PAIR_ORDER),
        "structure": ["random intercept", "random slope"], "predictor": list(PREDICTORS)}
coefficients = (pd.concat(parts, ignore_index=True)
                .reindex(columns=COLUMNS)
                .sort_values(list(RANK), key=lambda s: s.map(RANK[s.name].index))
                .reset_index(drop=True))
# One family per model, as in Tables 2 and 3: its seven predictors, the two
# tests corrected apart from each other.
for test in ("wald", "lrt"):
    add_corrected_column(coefficients, f"{test}_p", f"{test}_q",
                         within=["window", "pair", "structure"], method="fdr_bh")
re_day = coefficients[coefficients["predictor"] == FOCUS].reset_index(drop=True)

tables = pd.concat([pd.read_csv(OUT_DIR / "table2_early_window_lmm_full.csv"),
                    pd.read_csv(OUT_DIR / "table3_late_window_lmm_full.csv")], ignore_index=True)
check = coefficients[coefficients["structure"] == "random intercept"].merge(
    tables, on=["window", "pair", "predictor"], suffixes=("", "_table"))
assert len(check) == len(PREDICTORS) * len(WINDOWS) * len(PAIR_ORDER)
for col in ("beta", "wald_p", "wald_q", "lrt_p", "lrt_q"):
    assert np.allclose(check[col], check[f"{col}_table"], rtol=1e-9, atol=0), col
print("random intercept: every coefficient and test matches Tables 2 and 3")

rs_all = coefficients[coefficients["structure"] == "random slope"]
earlier = PRESUB_DIR / "individual_differences_coefficients.csv"
if earlier.exists():
    old = pd.read_csv(earlier)
    old = old[old["treatment"] == "random slope"]
    check = rs_all.merge(old, on=["window", "pair", "predictor"], suffixes=("", "_old"))
    assert len(check) == len(PREDICTORS) * len(WINDOWS) * len(PAIR_ORDER)
    for new_col, old_col in (("beta", "beta_old"), ("wald_p", "p"), ("wald_q", "q")):
        assert np.allclose(check[new_col], check[old_col], rtol=1e-9, atol=0), new_col
    print("random slope, Wald: matches the earlier individual-differences analysis")
else:
    print("the earlier individual-differences run is not here; skipping the comparison")
earlier = PRESUB_DIR / "individual_differences_random_slope_lrt.csv"
if earlier.exists():
    check = rs_all.merge(pd.read_csv(earlier), on=["window", "pair", "predictor"],
                         suffixes=("", "_old"))
    assert len(check) == len(PREDICTORS) * len(WINDOWS) * len(PAIR_ORDER)
    for col in ("lrt_chi2", "lrt_p", "lrt_q"):
        assert np.allclose(check[col], check[f"{col}_old"], rtol=1e-9, atol=0), col
    print("random slope, likelihood ratio: matches the earlier individual-differences analysis")
else:
    print("the earlier random-slope likelihood-ratio run is not here; skipping the comparison")
spread = rs_all[["llf_spread_full", "llf_spread_reduced"]].to_numpy().max()
print(f"random slope: the agreeing optimizers differ by at most {spread:.4f} in log-likelihood, "
      f"so each χ² is good to about {4 * spread:.3f}")

print(re_day[["window", "pair", "structure", "n_sessions", "beta", "ci95_lo", "ci95_hi",
              "wald_q", "lrt_chi2", "lrt_p", "lrt_q"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

# %% [markdown]
# ## 4. The figure
#
# Laid out as the co-authors' sketch: the two slope comparisons side by side
# on top, the random-effects comparison below. In **A** and **B** the four plots
# share one axis, since all four are slopes in SD per day; each row's q sits
# at its right. **C** reads like Figures 3 and 4, each pair on its own scale.
#
# Each of the three is one panel made of several plots, so its letter and a
# heading share one line above it (`figures.style.section_heading`).

# %%
PAD = 0.75      # room right of the intervals for the q values, as a share of the range
HEAD_GAP = 0.4  # extra space above each heading row
Q_HEADER = "q (FDR)"   # names the column of numbers at the right of every plot
Q_HEADER_Y = 1.0       # a row of its own, above the first heading


def sentence_case(label: str) -> str:
    """A row label as Figures 3 and 4 write theirs: capital first letter only."""
    return label[:1].upper() + label[1:]


def q_column_header(ax) -> None:
    """Name the q column, in a row above the plot's first heading."""
    ax.text(0.99, Q_HEADER_Y, Q_HEADER, transform=ax.get_yaxis_transform(), ha="right",
            va="center", fontsize=TICK_SIZE, fontstyle="italic", color="0.25")


def slope_panel(ax, fits: pd.DataFrame, window: str, measures: list[str], labels: bool) -> None:
    """One window of a slope comparison: a heading per pair, a row per measure."""
    ticks, names, heads, y = [], [], [], 0.0
    for pair in PAIR_ORDER:
        ticks.append(y)
        names.append(PAIR_LABEL[pair])
        heads.append((len(ticks) - 1, pair))
        y -= 1.0
        for measure in measures:
            r = fits[(fits["window"] == window) & (fits["pair"] == pair)
                     & (fits["measure"] == measure)].iloc[0]
            sig = r["q_value"] < Q_CUTOFF
            ax.errorbar(r["slope_per_day"], y,
                        xerr=[[r["slope_per_day"] - r["ci95_lo"]], [r["ci95_hi"] - r["slope_per_day"]]],
                        fmt="o", ms=5, lw=1.4, capsize=2.5, color=PAIR_COLOR[pair],
                        markerfacecolor=PAIR_COLOR[pair] if sig else "white", markeredgewidth=1.2)
            ax.text(0.99, y, with_star(r["q_value"], Q_CUTOFF),
                    transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=TICK_SIZE, color=PAIR_COLOR[pair])
            ticks.append(y)
            names.append(sentence_case(measure))
            y -= 1.0
        y -= HEAD_GAP
    ax.axvline(0, ls="--", lw=0.8, color="black")
    ax.set_yticks(ticks)
    ax.set_yticklabels(names if labels else [], fontsize=TICK_SIZE)
    if labels:
        for i, pair in heads:
            lab = ax.get_yticklabels()[i]
            lab.set_fontweight("bold")
            lab.set_color(PAIR_COLOR[pair])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y + HEAD_GAP - 0.4, Q_HEADER_Y + 0.6)
    q_column_header(ax)
    ax.set_title(f"{window.capitalize()} window", fontsize=LABEL_SIZE)
    ax.set_xlabel(" ")   # the room for the label the pair of plots shares (below)
    clean_axes(ax)


def random_effects_panel(ax, pair: str, labels: bool) -> None:
    """The day coefficient of one pair: both windows, both random structures."""
    ticks, names, heads, y = [], [], [], 0.0
    for window in WINDOWS:
        ticks.append(y)
        names.append(f"{window.capitalize()} window")
        heads.append(len(ticks) - 1)
        y -= 1.0
        for structure in ("random intercept", "random slope"):
            r = re_day[(re_day["window"] == window) & (re_day["pair"] == pair)
                       & (re_day["structure"] == structure)].iloc[0]
            colour = SIG_COLOR if r["lrt_q"] < Q_CUTOFF else NULL_COLOR
            ax.errorbar(r["beta"], y, xerr=[[r["beta"] - r["ci95_lo"]], [r["ci95_hi"] - r["beta"]]],
                        fmt="o", ms=5, lw=1.4, capsize=2.5, color=colour)
            ax.text(0.99, y, with_star(r["lrt_q"], Q_CUTOFF),
                    transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=TICK_SIZE, color=colour)
            ticks.append(y)
            names.append(sentence_case(structure))
            y -= 1.0
        y -= HEAD_GAP
    ax.axvline(0, ls="--", lw=0.8, color="black")
    ax.set_yticks(ticks)
    ax.set_yticklabels(names if labels else [], fontsize=TICK_SIZE)
    if labels:
        for i in heads:
            ax.get_yticklabels()[i].set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y + HEAD_GAP - 0.4, Q_HEADER_Y + 0.6)
    q_column_header(ax)
    panel = re_day[re_day["pair"] == pair]
    lo, hi = panel["ci95_lo"].min(), panel["ci95_hi"].max()
    ax.set_xlim(lo - 0.06 * (hi - lo), hi + PAD * (hi - lo))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.xaxis.set_major_formatter(decimal_formatter(3))
    ax.set_title(PAIR_LABEL[pair], fontsize=LABEL_SIZE)
    ax.set_xlabel("Training-day β (AUC per SD)")
    clean_axes(ax)


# Not square: four panels share the top row and three the bottom, so square
# panels would make the top ones small and leave the bottom ones' few rows
# spread over too much height. The top row gets the height instead.
fig = plt.figure(figsize=(WIDTH_2COL, 5.4), layout="constrained")
top, bottom = fig.subfigures(2, 1, height_ratios=[1.3, 1.0])
top.suptitle(" ", fontsize=PANEL_LABEL_SIZE)   # the room the letters and headings go in
axes_top = top.subplots(1, 4)
slope_panel(axes_top[0], pc_fits, "early", list(SCALES), labels=True)
slope_panel(axes_top[1], pc_fits, "late", list(SCALES), labels=False)
slope_panel(axes_top[2], tc_fits, "early", ["all trials", "10 trials each"], labels=True)
slope_panel(axes_top[3], tc_fits, "late", ["all trials", "10 trials each"], labels=False)
both = pd.concat([pc_fits, tc_fits])
lo, hi = min(0.0, both["ci95_lo"].min()), both["ci95_hi"].max()
for ax in axes_top:
    ax.set_xlim(lo - 0.06 * (hi - lo), hi + PAD * (hi - lo))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))

bottom.suptitle(" ", fontsize=PANEL_LABEL_SIZE)
axes_bottom = bottom.subplots(1, 3)
for k, (ax, pair) in enumerate(zip(axes_bottom, PAIR_ORDER)):
    random_effects_panel(ax, pair, labels=k == 0)

fig.draw_without_rendering()
renderer = fig.canvas.get_renderer()
HEADING_GAP = 7.0   # points between a heading and the plot titles below it
section_heading(top, axes_top[:2], "A", "PC weighting", level_with=axes_top, gap_points=HEADING_GAP)
section_heading(top, axes_top[2:], "B", "Trial-count control", level_with=axes_top,
                gap_points=HEADING_GAP)
section_heading(bottom, axes_bottom, "C", "Individual variability", gap_points=HEADING_GAP)
# Each pair of plots in A and B shares one axis, so it gets one label: a
# label per plot is too wide for these narrow plots and runs into the next.
SLOPE_LABEL = "Slope against training day (SD/day)"   # the co-authors' wording, 2026-10-26
group_xlabel(top, axes_top[:2], SLOPE_LABEL)
group_xlabel(top, axes_top[2:], SLOPE_LABEL)
print(f"  top panels {equal_panel_widths(axes_top):.4f} wide, "
      f"bottom panels {equal_panel_widths(axes_bottom):.4f} wide (figure fraction)")
# In display pixels: an axes' position is relative to its subfigure, not to
# the figure, so it cannot be turned into inches with the figure's size.
for row, axes in (("top", axes_top), ("bottom", axes_bottom)):
    box = axes[0].get_window_extent(renderer)
    print(f"  {row} panels: {box.width / fig.dpi:.2f} x {box.height / fig.dpi:.2f} inch")

save_fig(fig, "fig5_robustness", OUT_DIR)
plt.show()

# %% [markdown]
# ## 5. The numbers

# %%
pc_fits.to_csv(OUT_DIR / "fig5_pc_weighting_regressions.csv", index=False)
tc_fits.to_csv(OUT_DIR / "fig5_trial_count_regressions.csv", index=False)
coefficients.to_csv(OUT_DIR / "fig5_random_effects_coefficients.csv", index=False)
for name in ("pc_weighting_regressions", "trial_count_regressions", "random_effects_coefficients"):
    print(f"  saved: {display_path(OUT_DIR)}/fig5_{name}.csv")

# %% [markdown]
# ## 6. Numbers quoted in the text

# %%
for title, fits in (("PC weighting", pc_fits), ("trial count", tc_fits)):
    print(f"\n{title}: slope (SD / day) [95% CI], q (BH over the three pairs)")
    for _, r in fits.iterrows():
        print(f"  {r['window']:<5} {r['measure']:<16} {PAIR_LABEL[r['pair']]:<17} "
              f"n = {int(r['n_sessions']):>3}  {r['slope_per_day']:+.4f} "
              f"[{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}]  p = {r['p_value']:.4g}, "
              f"q = {r['q_value']:.4g}{' *' if r['q_value'] < Q_CUTOFF else ''}")

print("\nindividual variability: day β [95% CI], likelihood-ratio χ², p and q "
      "(BH over the seven predictors)")
for _, r in re_day.iterrows():
    print(f"  {r['window']:<5} {PAIR_LABEL[r['pair']]:<17} {r['structure']:<17} "
          f"{r['beta']:.5f} [{r['ci95_lo']:.5f}, {r['ci95_hi']:.5f}]  "
          f"χ² = {r['lrt_chi2']:.3f}, p = {r['lrt_p']:.4g}, q = {r['lrt_q']:.4g}"
          f"{' *' if r['lrt_q'] < Q_CUTOFF else ''}")
