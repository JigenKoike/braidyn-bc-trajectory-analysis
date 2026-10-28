# %% [markdown]
# # Figure 3 and Table 2 — What the early-window separation goes with
#
# Figure 2 asked whether the separation grows with training, and treated the
# sessions of a mouse as independent. This one asks what else it moves with,
# and gives each mouse its own intercept so that repeated sessions of one
# animal are no longer counted as independent.
#
# **What is computed.** One linear mixed-effects model per outcome pair: the
# early-window AUC explained by seven session-level predictors, with a random
# intercept per mouse, fitted by maximum likelihood. Each predictor is then
# dropped in turn, to ask what the model loses without it.
#
# | Panel | Content |
# |---|---|
# | **A** | success–miss: β and its 95% confidence interval, one row per predictor |
# | **B** | failure–miss: the same |
# | **C** | success–failure: the same |
# | **D** | Leave-one-out ΔAIC, every predictor against every pair |
#
# **Table 2** comes from the same fits: one sheet, a heading row per pair, then
# its seven predictors with β, the interval and both tests.
#
# **Statistics.** Red in **A**–**C** and an asterisk in **D** mark the same
# thing: the likelihood-ratio test of dropping that predictor from the model,
# q < 0.05, Benjamini-Hochberg over the seven predictors of that one pair. The
# three pairs are separate models. The confidence intervals are Wald
# intervals and are not corrected, so a red point and an interval clear of
# zero need not agree. Table 2 also gives the Wald test of each coefficient,
# corrected separately from the likelihood-ratio test.
#
# **Inputs**
#
# All of it from `data/paper/`, which `scripts/build_paper_dataset.py` fills
# from the pipeline's output.
#
# | File | Originally made by |
# |---|---|
# | `distance_timecourse.csv` | `scripts/run_pairwise_pca.py` |
# | `baseline_F_per_session.csv` | `scripts/run_photobleaching.py` |
# | `trials.parquet` | `scripts/extract_session_data.py` |
# | `pull_duration.parquet` | `scripts/extract_pull_duration.py` |

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

from analysis.lmm import PREDICTORS, fit_leave_one_out, lmm_tables
from analysis.pairs import PAIR_ORDER
from analysis.predictors import session_predictors, standardize
from analysis.regression import auc_per_session
from analysis.windows import EARLY_WINDOW
from figures.formatting import decimal_formatter, signed
from figures.palette import (
    DIVERGING_CMAP,
    PAIR_LABEL,
    PAIR_LABEL_2LINE,
    PRED_DISPLAY,
)
from figures.style import (
    display_path,
    LABEL_SIZE,
    TICK_SIZE,
    WIDTH_2COL,
    clean_axes,
    panel_label,
    save_fig,
    size_colorbar,
    use_paper_style,
)
from figures.tables import window_lmm_table, window_lmm_table_full

use_paper_style()

DATA_DIR = REPO_ROOT / "data/paper"
DIST_CSV = DATA_DIR / "distance_timecourse.csv"
OUT_DIR = REPO_ROOT / "output/paper_figures"

WINDOW = {"window": "early"}  # recorded in the tidy tables, so late joins cleanly
Q_CUTOFF = 0.05
SIG_COLOR = "tab:red"
NULL_COLOR = "0.45"

print(f"output: {display_path(OUT_DIR)}")

# %% [markdown]
# ## 1. The session-level predictors
#
# One row per task session. The behavioural three are read from the trial
# table, the baseline fluorescence from the photobleaching output. Every
# predictor is z-standardized over **all** task sessions, not over the
# subset a given pair happens to have, so one β means the same thing in all
# three models.

# %%
pred_raw = session_predictors(DATA_DIR)
predictors_z, Z_MEAN, Z_SD = standardize(pred_raw)

print(f"task sessions: {len(pred_raw)}, mice: {pred_raw['mouse'].nunique()}")
print("\nraw-scale mean and SD used for z-scoring (all task sessions):")
print(pd.DataFrame({"mean": Z_MEAN, "sd": Z_SD}).to_string(float_format=lambda v: f"{v:.4g}"))

# %% [markdown]
# ## 2. The outcome: one early-window AUC per session and pair
#
# The same number Figure 2 regressed on training day, from the same function.

# %%
distance_long = pd.read_csv(DIST_CSV)
distance_long = distance_long[distance_long["pair"].isin(PAIR_ORDER)]
sessions = auc_per_session(distance_long, EARLY_WINDOW)

print(sessions.groupby("pair")["auc"].describe().loc[list(PAIR_ORDER)]
      .to_string(float_format=lambda v: f"{v:.4g}"))

# %% [markdown]
# ## 3. The models
#
# For each pair: the full model, then the seven models with one predictor
# dropped. Every optimizer is tried and the best converged fit is checked
# against a grid search of the likelihood — `analysis.lmm` explains why, and
# records what happened for each fit in the convergence log.

# %%
convergence_log: list[dict] = []
fitted = {}
for pair in PAIR_ORDER:
    pair_data = predictors_z.merge(
        sessions.loc[sessions["pair"] == pair, ["session_id", "auc"]], on="session_id"
    )
    data, scale, fits = fit_leave_one_out(
        pair_data, convergence_log=convergence_log, label={**WINDOW, "pair": pair}
    )
    fitted[pair] = {"data": data, "scale": scale, "fits": fits}
    print(f"{PAIR_LABEL[pair]:<17} n = {len(data):>3} sessions, "
          f"{data['mouse'].nunique()} mice, outcome scale {scale:g}")

model_fit, leave_one_out, coefficients = lmm_tables(
    fitted, predictor_mean=Z_MEAN, predictor_sd=Z_SD, label=WINDOW
)

# %% [markdown]
# ## 4. The figure

# %%
# The height makes A-C square at the width the two columns give them. Panel
# D is a different kind of plot and keeps its own proportions; its colour
# bar stays inside the layout, where it belongs to D alone.
fig = plt.figure(figsize=(WIDTH_2COL, 6.3), layout="constrained")
subfigs = fig.subfigures(2, 2)
forest_panels = {
    "success-miss": (subfigs[0, 0], "A"),
    "failure-miss": (subfigs[0, 1], "B"),
    "success-failure": (subfigs[1, 0], "C"),
}
ypos = np.arange(len(PREDICTORS))[::-1]  # the formula's first predictor on top
forest_axes = []
coef_i = coefficients.set_index(["pair", "predictor"])
loo_i = leave_one_out.set_index(["pair", "predictor"])  # the test that colours A-C and stars D

# --- A-C: the coefficients of each pair's full model ---
# Each panel keeps its own x range, as in the layout the co-author sent: the
# question each one answers is which of its own predictors clear zero, and a
# scale shared with the widest pair would squeeze the other two.
for pair, (subfig, letter) in forest_panels.items():
    ax = subfig.subplots()
    for y, p in zip(ypos, PREDICTORS):
        r = coef_i.loc[(pair, p)]
        ax.errorbar(r["beta"], y,
                    xerr=[[r["beta"] - r["ci95_lo"]], [r["ci95_hi"] - r["beta"]]],
                    fmt="o", ms=5, lw=1.4, capsize=2.5,
                    color=SIG_COLOR if loo_i.loc[(pair, p), "lrt_q"] < Q_CUTOFF else NULL_COLOR)
    ax.axvline(0, ls="--", lw=0.8, color="black")
    ax.set_yticks(ypos)
    ax.set_yticklabels([PRED_DISPLAY[p] for p in PREDICTORS])
    ax.set_ylim(-0.6, len(PREDICTORS) - 0.4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(decimal_formatter(3))
    ax.set_xlabel("β (AUC per SD)")
    ax.set_title(PAIR_LABEL[pair], fontsize=LABEL_SIZE)
    # Square. Left to itself the panel comes out a tenth wider than tall,
    # which reads as a mistake rather than as a choice.
    ax.set_box_aspect(1)
    clean_axes(ax)
    panel_label(subfig, letter)
    forest_axes.append(ax)

# --- D: what each predictor is worth to each model ---
ax_d = subfigs[1, 1].subplots()
delta =np.array([[loo_i.loc[(pair, p), "delta_AIC"] for pair in PAIR_ORDER] for p in PREDICTORS])
q = np.array([[loo_i.loc[(pair, p), "lrt_q"] for pair in PAIR_ORDER] for p in PREDICTORS])
vmax = float(np.ceil(np.abs(delta).max()))  # symmetric, so the diverging map puts zero at its middle
im = ax_d.imshow(delta, cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
for i in range(delta.shape[0]):
    for j in range(delta.shape[1]):
        ax_d.text(j, i, signed(delta[i, j], 2, plus=True) + ("*" if q[i, j] < Q_CUTOFF else ""),
                  ha="center", va="center", fontsize=TICK_SIZE,
                  color="white" if abs(delta[i, j]) > 0.6 * vmax else "black")
ax_d.set_xticks(range(len(PAIR_ORDER)))
ax_d.set_xticklabels([PAIR_LABEL_2LINE[p] for p in PAIR_ORDER])
ax_d.set_yticks(range(len(PREDICTORS)))
ax_d.set_yticklabels([PRED_DISPLAY[p] for p in PREDICTORS])
ax_d.set_title("Leave-one-out ΔAIC", fontsize=LABEL_SIZE)
size_colorbar(subfigs[1, 1].colorbar(im, ax=ax_d, fraction=0.05, pad=0.03), "ΔAIC")
panel_label(subfigs[1, 1], "D")

fig.draw_without_rendering()
width_in, height_in = fig.get_size_inches()
for letter, ax in zip("ABC", forest_axes):
    box = ax.get_position()
    w, h = box.width * width_in / 2, box.height * height_in / 2  # each is half the figure
    print(f"  panel {letter}: {w:.3f} x {h:.3f} inch, width/height {w / h:.3f}")
    assert abs(w / h - 1) < 0.01, f"panel {letter} is not square: {w / h:.3f}"

save_fig(fig, "fig3_lmm_early_window", OUT_DIR)
plt.show()

# %% [markdown]
# ## 5. Table 2
#
# Two files: the sheet the manuscript prints, and the same numbers tidy and
# unrounded for anything downstream.

# %%
table2 = window_lmm_table(model_fit, leave_one_out, coefficients)
table2_full = window_lmm_table_full(model_fit, leave_one_out, coefficients)

table2.to_csv(OUT_DIR / "table2_early_window_lmm.csv", index=False)
table2_full.to_csv(OUT_DIR / "table2_early_window_lmm_full.csv", index=False)
pd.DataFrame(convergence_log).to_csv(OUT_DIR / "fig3_lmm_convergence.csv", index=False)
print(f"  saved: {display_path(OUT_DIR)}/table2_early_window_lmm{{,_full}}.csv")
print(f"  saved: {display_path(OUT_DIR)}/fig3_lmm_convergence.csv")
print()
print(table2.to_string(index=False))

# %% [markdown]
# ## 6. Against the previous run
#
# The numbers must be the ones the earlier notebook produced; only their
# arrangement has changed. `table3_loo_delta_aic_full.csv` and
# `table4_coefficients_full.csv` are that notebook's output, kept for this
# comparison. It is skipped if they are no longer around.

# %%
previous = {"leave_one_out": OUT_DIR / "table3_loo_delta_aic_full.csv",
            "coefficients": OUT_DIR / "table4_coefficients_full.csv"}
if all(p.exists() for p in previous.values()):
    for name, path, new in [("leave_one_out", previous["leave_one_out"], leave_one_out),
                            ("coefficients", previous["coefficients"], coefficients)]:
        old = pd.read_csv(path)
        keys = ["pair", "predictor"]
        merged = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
        assert len(merged) == len(old) == len(new), f"{name}: the two runs cover different rows"
        worst = {}
        for col in [c for c in old.columns if c not in keys and old[c].dtype.kind == "f"]:
            a, b = merged[f"{col}_old"].to_numpy(), merged[f"{col}_new"].to_numpy()
            worst[col] = float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300)))
        largest = max(worst, key=worst.get)
        print(f"{name}: {len(merged)} rows, largest relative difference "
              f"{worst[largest]:.2e} (in {largest})")
        assert worst[largest] < 1e-8, f"{name}: {largest} moved by {worst[largest]:.2e}"
else:
    print("the previous run's CSVs are gone; skipping the comparison")

# %% [markdown]
# ## 7. Numbers quoted in the text

# %%
print(f"early window: {EARLY_WINDOW[0]} to {EARLY_WINDOW[1]} s\n")
for pair in PAIR_ORDER:
    f = model_fit.set_index("pair").loc[pair]
    print(f"{PAIR_LABEL[pair]:<17} n = {int(f['n_sessions'])} sessions, {int(f['n_mice'])} mice, "
          f"marginal R² = {f['marginal_R2']:.3f}, conditional R² = {f['conditional_R2']:.3f}")
    wald = coefficients[(coefficients["pair"] == pair) & (coefficients["wald_q"] < Q_CUTOFF)]
    lrt = leave_one_out[(leave_one_out["pair"] == pair) & (leave_one_out["lrt_q"] < Q_CUTOFF)]
    print(f"{'':<17} Wald q < {Q_CUTOFF}: "
          f"{', '.join(PRED_DISPLAY[p] for p in wald['predictor']) or 'none'}")
    print(f"{'':<17} LRT  q < {Q_CUTOFF}: "
          f"{', '.join(PRED_DISPLAY[p] for p in lrt['predictor']) or 'none'}")

n_wald = int((coefficients["wald_q"] < Q_CUTOFF).sum())
n_lrt = int((leave_one_out["lrt_q"] < Q_CUTOFF).sum())
print(f"\nacross the three models: {n_wald} of {len(coefficients)} Wald tests and "
      f"{n_lrt} of {len(leave_one_out)} likelihood-ratio tests reach q < {Q_CUTOFF}")
