# %% [markdown]
# # Figure 4 and Table 3 — The late window
#
# Figures 2 and 3 are about the early window, 0.2 to 2.0 s after the cue. This
# one puts both of their questions to the late window, 2.0 to 4.0 s, with
# everything else held the same: does the separation grow across training
# days, and what does it go with session by session.
#
# **What is computed.** What Figures 2 and 3 compute, over the other window —
# the same predictors, built by the same function, and the same models. The
# two windows are 1.8 and 2.0 s wide, so an AUC from one is not comparable
# with an AUC from the other; the standardized coefficients are, since β is in
# AUC per standard deviation of a predictor.
#
# | Panel | Content |
# |---|---|
# | **A** | Late-window AUC against training day, one column per outcome pair |
# | **B** | β and its 95% confidence interval, one column per pair |
#
# **Table 3** is Table 2 for this window, built by the same two functions. The
# leave-one-out ΔAIC is in it rather than in a panel of its own.
#
# **Statistics.** Red in **B** marks q < 0.05 for the likelihood-ratio test of
# dropping that predictor, Benjamini-Hochberg over the seven predictors of
# that one model, exactly as in Figure 3. The intervals are Wald intervals,
# not corrected. Table 3 also gives the Wald tests, corrected separately. The day
# regressions drawn in **A** are Benjamini-Hochberg corrected over the three
# pairs of this window — a family of its own, separate from Figure 2's, which
# covers the early window.
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
from analysis.regression import auc_per_session, day_regression, fitted_line
from analysis.windows import LATE_WINDOW
from figures.formatting import decimal_formatter
from figures.palette import PAIR_COLOR, PAIR_LABEL, PRED_DISPLAY
from figures.style import (
    display_path,
    LABEL_SIZE,
    WIDTH_2COL,
    uniform_columns,
    clean_axes,
    equal_panel_widths,
    panel_label,
    save_fig,
    use_paper_style,
)
from figures.tables import window_lmm_table, window_lmm_table_full

use_paper_style()

DATA_DIR = REPO_ROOT / "data/paper"
DIST_CSV = DATA_DIR / "distance_timecourse.csv"
OUT_DIR = REPO_ROOT / "output/paper_figures"

WINDOW = {"window": "late"}
DAY_TICKS = [1, 5, 10, 15]
REGRESSION_DAYS = (1, 15)
Y_HEADROOM = 1.04  # as in Figure 2: one scale per row, from zero, with the top clear of the frame
Y_TICK_DECIMALS = 2
Q_CUTOFF = 0.05
SIG_COLOR = "tab:red"
NULL_COLOR = "0.45"

print(f"output: {display_path(OUT_DIR)}")
print(f"late window: {LATE_WINDOW[0]} to {LATE_WINDOW[1]} s")

# %% [markdown]
# ## 1. Predictors and outcome
#
# The predictors are built and standardized exactly as in Figure 3 — the
# same function, so the two windows cannot drift apart. Only the window the
# AUC is taken over differs.

# %%
pred_raw = session_predictors(DATA_DIR)
predictors_z, Z_MEAN, Z_SD = standardize(pred_raw)

distance_long = pd.read_csv(DIST_CSV)
distance_long = distance_long[distance_long["pair"].isin(PAIR_ORDER)]
sessions = auc_per_session(distance_long, LATE_WINDOW)

print(f"task sessions: {len(pred_raw)}, mice: {pred_raw['mouse'].nunique()}\n")
print(sessions.groupby("pair")["auc"].describe().loc[list(PAIR_ORDER)]
      .to_string(float_format=lambda v: f"{v:.4g}"))

# %% [markdown]
# ## 2. The trend over days
#
# One least-squares line per pair, with the three p values corrected by
# Benjamini-Hochberg within this window — the same family and correction
# Figure 2 uses for the early one.

# %%
regression = day_regression(sessions, predict_days=REGRESSION_DAYS, label=WINDOW)
print(regression.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

# %% [markdown]
# ## 3. The models
#
# Identical to Figure 3's, on the late-window AUC.

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
# Both rows square, once uniform_columns has set the column width: the
# height and the equal row shares are what get them there, and the assertion
# below holds them. Seven predictor rows fit a square panel comfortably --
# Figure 3's are square with the same seven -- and the figure reads as one
# set with that one.
fig = plt.figure(figsize=(WIDTH_2COL, 5.05), layout="constrained")
sf_a, sf_b = fig.subfigures(2, 1, height_ratios=[1.0, 1.0])

# --- A: every session's late-window AUC against the day it was recorded ---
# One scale across the three pairs, from zero, as in Figure 2: they are areas
# under the same measure in the same units, and a per-panel scale draws the
# smaller pairs at the size of the largest.
axes_a = sf_a.subplots(1, len(PAIR_ORDER), sharey=True)
jitter_rng = np.random.default_rng(0)  # only so overlapping sessions stay visible
by_pair = regression.set_index("pair")
for ax, pair in zip(axes_a, PAIR_ORDER):
    sub = sessions[sessions["pair"] == pair]
    jitter = jitter_rng.uniform(-0.15, 0.15, size=len(sub))
    ax.scatter(sub["day"] + jitter, sub["auc"], s=7, alpha=0.35,
               color=PAIR_COLOR[pair], linewidths=0)
    daily = sub.groupby("day")["auc"].agg(["mean", "sem"])
    ax.errorbar(daily.index, daily["mean"], yerr=daily["sem"], fmt="o", ms=3,
                color="black", lw=1.0, capsize=2)
    days = np.array(REGRESSION_DAYS, dtype=float)
    ax.plot(days, fitted_line(by_pair.loc[pair], days), ls="--", lw=1.3, color="red")
    ax.set_xticks(DAY_TICKS)
    ax.set_xlim(0, 16)
    ax.set_xlabel("Training day")
    ax.set_title(PAIR_LABEL[pair], fontsize=LABEL_SIZE)
    clean_axes(ax)
axes_a[0].set_ylim(0, sessions["auc"].max() * Y_HEADROOM)
axes_a[0].yaxis.set_major_formatter(decimal_formatter(Y_TICK_DECIMALS))
axes_a[0].set_ylabel("Late-window AUC (a.u.·s)")  # the unit as in Figure 2B
panel_label(sf_a, "A")

# --- B: the coefficients of each pair's full model ---
# Each panel keeps its own x range, as in Figure 3. The predictor names go
# down the left once rather than three times: repeating them costs each
# panel enough width to leave B's plots two thirds the size of A's, and the
# three rows of names are the same names in the same order.
axes_b = sf_b.subplots(1, len(PAIR_ORDER), sharey=True)
ypos = np.arange(len(PREDICTORS))[::-1]  # the formula's first predictor on top
coef_i = coefficients.set_index(["pair", "predictor"])
loo_i = leave_one_out.set_index(["pair", "predictor"])  # the test that colours B
for ax, pair in zip(axes_b, PAIR_ORDER):
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
    clean_axes(ax)
panel_label(sf_b, "B")

# A and B are the same three pairs in the same order, so their columns line
# up; the layout engine does not do this across two rows on its own.
fig.draw_without_rendering()
for i, (x0, x1) in enumerate(uniform_columns(fig, axes_a, axes_b), start=1):
    print(f"  column {i}: [{x0:.4f}, {x1:.4f}]")
print(f"  panel width in A: {equal_panel_widths(axes_a):.4f}, "
      f"in B: {equal_panel_widths(axes_b):.4f}")
for name, axes, sub in (("A", axes_a, sf_a), ("B", axes_b, sf_b)):
    box = axes[0].get_position()
    w = box.width * sub.bbox.width / fig.dpi
    h = box.height * sub.bbox.height / fig.dpi
    print(f"  panel {name}: {w:.3f} x {h:.3f} inch, width/height {w / h:.3f}")
    assert abs(w / h - 1) < 0.01, f"panel {name} is not square: {w / h:.3f}"

save_fig(fig, "fig4_late_window", OUT_DIR)
plt.show()

# %% [markdown]
# ## 5. Table 3
#
# The same two shapes as Table 2, from the same two functions.

# %%
table3 = window_lmm_table(model_fit, leave_one_out, coefficients)
table3_full = window_lmm_table_full(model_fit, leave_one_out, coefficients)

table3.to_csv(OUT_DIR / "table3_late_window_lmm.csv", index=False)
table3_full.to_csv(OUT_DIR / "table3_late_window_lmm_full.csv", index=False)
# Not "fig4_regression_stats.csv": the earlier notebook set wrote that name
# for the *early* window, and Figure 2 is checked against it.
regression.to_csv(OUT_DIR / "fig4_late_regression_stats.csv", index=False)
pd.DataFrame(convergence_log).to_csv(OUT_DIR / "fig4_lmm_convergence.csv", index=False)
print(f"  saved: {display_path(OUT_DIR)}/table3_late_window_lmm{{,_full}}.csv")
print(f"  saved: {display_path(OUT_DIR)}/fig4_late_regression_stats.csv")
print(f"  saved: {display_path(OUT_DIR)}/fig4_lmm_convergence.csv")
print()
print(table3.to_string(index=False))

# %% [markdown]
# ## 6. Against the previous run
#
# The earlier late-window notebook wrote `late_window/late_table*_full.csv`
# and `late_regression_stats.csv`. The numbers here must be those, only
# rearranged. Skipped if the files are gone.

# %%
LATE_DIR = OUT_DIR / "late_window"
previous = {
    "leave_one_out": (LATE_DIR / "late_table3_loo_delta_aic_full.csv", leave_one_out),
    "coefficients": (LATE_DIR / "late_table4_coefficients_full.csv", coefficients),
    "regression": (LATE_DIR / "late_regression_stats.csv", regression),
}
for name, (path, new) in previous.items():
    if not path.exists():
        print(f"{name}: {path.name} is gone; skipping")
        continue
    old = pd.read_csv(path)
    keys = ["pair", "predictor"] if "predictor" in old.columns else ["pair"]
    merged = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    assert len(merged) == len(old) == len(new), f"{name}: the two runs cover different rows"
    worst = {}
    for col in [c for c in old.columns if c not in keys and old[c].dtype.kind == "f"]:
        if f"{col}_new" not in merged.columns:
            continue  # a column the earlier run had and this one does not
        a, b = merged[f"{col}_old"].to_numpy(), merged[f"{col}_new"].to_numpy()
        worst[col] = float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300)))
    largest = max(worst, key=worst.get)
    print(f"{name}: {len(merged)} rows, {len(worst)} numeric columns, "
          f"largest relative difference {worst[largest]:.2e} (in {largest})")
    assert worst[largest] < 1e-8, f"{name}: {largest} moved by {worst[largest]:.2e}"

# %% [markdown]
# ## 7. Numbers quoted in the text

# %%
print("late-window AUC against training day, per pair")
for _, r in regression.iterrows():
    change = r[f"rel_change_day{REGRESSION_DAYS[0]}_to_day{REGRESSION_DAYS[1]}"] * 100
    print(f"  {PAIR_LABEL[r['pair']]:<17} slope {r['slope_per_day']:+.5f} "
          f"[{r['ci95_lo']:+.5f}, {r['ci95_hi']:+.5f}]  "
          f"p = {r['p_value']:.4g}, q = {r['q_value']:.4g}  "
          f"day {REGRESSION_DAYS[0]} to {REGRESSION_DAYS[1]}: {change:+.1f}%")

print("\nthe mixed models")
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
