# %% [markdown]
# # Figure 2 — How the separation changes with training
#
# Figure 1 showed the three outcomes lying apart in the shared PC space. This
# one asks whether that separation grows as an animal learns, across the
# fifteen training days.
#
# **What is computed.** Each session's distance curve is reduced to one number
# — the area under it over the early window, 0.2 to 2.0 s after the cue — and
# that number is regressed on training day, once per outcome pair.
#
# | Panel | Content |
# |---|---|
# | **A** | Distance timecourse per training day, one column per outcome pair; the early window shaded |
# | **B** | Early-window AUC against training day, one column per pair |
# | **C** | Slope per day of each pair, with its 95% confidence interval |
#
# **Statistics.** The three regressions are separate and descriptive: each asks
# whether that one pair's separation grows, and their slopes are not compared
# with one another. Their p values are corrected over the three pairs of this
# window by Benjamini-Hochberg — the correction the mixed models of Figures 3
# and 4 use — and reported as q. In **C** a filled circle and an asterisk
# both mark q < 0.05, as in Figure 5. Sessions of a mouse are treated as
# independent here; the repeated measurement is what the mixed models of
# Figure 3 are for.
#
# **Inputs**
#
# | File | Made by |
# |---|---|
# | `data/paper/distance_timecourse.csv` | `scripts/run_pairwise_pca.py`, via `build_paper_dataset.py` |

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
from matplotlib import colormaps, colors
from matplotlib.cm import ScalarMappable

from analysis.pairs import PAIR_ORDER
from analysis.regression import auc_per_session, day_regression, fitted_line
from analysis.windows import EARLY_WINDOW
from figures.formatting import SIG_CUTOFF, decimal_formatter, fmt_p, signed, with_star
from figures.palette import DISTANCE_LABEL, PAIR_COLOR, PAIR_LABEL, SEQUENTIAL_CMAP
from figures.style import (
    display_path,
    CBAR_OVERHANG,
    LABEL_SIZE,
    TICK_SIZE,
    WIDTH_2COL,
    uniform_columns,
    assert_page_width,
    clean_axes,
    equal_panel_widths,
    outside_colorbar,
    panel_label,
    place_outside_colorbar,
    save_fig,
    use_paper_style,
)

use_paper_style()

DIST_CSV = REPO_ROOT / "data/paper/distance_timecourse.csv"
OUT_DIR = REPO_ROOT / "output/paper_figures"

DAY_TICKS = [1, 5, 10, 15]
TIME_TICKS = [-1, 0, 1, 2, 3, 4]
REGRESSION_DAYS = (1, 15)  # the fitted line is drawn and quoted between these

# Panels A and B each run from zero to one common top, so that the three pairs
# can be read against one another. A little headroom keeps the tallest curve
# off the frame.
Y_HEADROOM = 1.04
# Both rows get the same number of characters down the left, which is what
# lines A's panels up with B's: the layout engine sizes each panel around its
# own decorations, so equal tick labels mean equal columns.
Y_TICK_DECIMALS = 2

print(f"output: {display_path(OUT_DIR)}")

# %% [markdown]
# ## 1. The distance timecourses
#
# Averaging within a mouse before averaging across mice, so a mouse with more
# sessions on a given day does not count for more than one with fewer.

# %%
distance_long = pd.read_csv(DIST_CSV)
distance_long = distance_long[distance_long["pair"].isin(PAIR_ORDER)]

per_mouse_day = (distance_long
                 .groupby(["pair", "mouse", "day", "t"], as_index=False)
                 .agg(dist=("dist", "mean")))
per_day = (per_mouse_day
           .groupby(["pair", "day", "t"], as_index=False)
           .agg(dist=("dist", "mean"), n_mice=("mouse", "nunique")))
all_days = sorted(per_day["day"].dropna().unique())

print(distance_long.groupby("pair")
      .agg(sessions=("session_id", "nunique"), mice=("mouse", "nunique"))
      .reindex(list(PAIR_ORDER)).to_string())
print(f"\ntraining days: {min(all_days):.0f} to {max(all_days):.0f}")

# %% [markdown]
# ## 2. One number per session, and its trend over days
#
# The area under each session's distance curve across the early window, then
# one ordinary least-squares line per pair.

# %%
sessions = auc_per_session(distance_long, EARLY_WINDOW)
regression = day_regression(sessions, predict_days=REGRESSION_DAYS)

print(regression.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

# %% [markdown]
# ## 3. The figure
#
# Panels A and B share their columns and, within a row, one vertical scale:
# the three pairs are the same measure in the same units.

# %%
fig = plt.figure(figsize=(WIDTH_2COL - CBAR_OVERHANG, 7.2), layout="constrained")
sf_a, sf_b, sf_c = fig.subfigures(3, 1, height_ratios=[1.0, 1.0, 0.82])

day_colours = colormaps[SEQUENTIAL_CMAP]
day_norm = colors.Normalize(vmin=min(all_days), vmax=max(all_days))

# --- A: one timecourse per training day ---
# One y scale across the three pairs, from zero. They are distances in the
# same space and the same units, so a per-panel scale was drawing the two
# smaller pairs at the size of the largest one.
axes_a = sf_a.subplots(1, len(PAIR_ORDER), sharey=True)
for ax, pair in zip(axes_a, PAIR_ORDER):
    sub = per_day[per_day["pair"] == pair]
    for day, g in sub.groupby("day"):
        g = g.sort_values("t")
        ax.plot(g["t"], g["dist"], color=day_colours(day_norm(day)), lw=1.1)
    ax.axvline(0, ls="--", lw=0.7, color="black")
    # Only the early window, which is the one this figure summarizes. Figure 1E
    # shows both; Figure 4 shades the late one.
    ax.axvspan(EARLY_WINDOW[0], EARLY_WINDOW[1], color="0.87", zorder=0)
    ax.set_xticks(TIME_TICKS)
    ax.set_xlim(-1, 4)
    ax.set_xlabel("Time from cue (s)")
    ax.set_title(PAIR_LABEL[pair], fontsize=LABEL_SIZE)
    clean_axes(ax)
axes_a[0].set_ylim(0, per_day["dist"].max() * Y_HEADROOM)
axes_a[0].yaxis.set_major_formatter(decimal_formatter(Y_TICK_DECIMALS))
axes_a[0].set_ylabel(DISTANCE_LABEL)
# The colour bar goes in by hand, past the right edge of the subfigure. Handing
# it to the layout engine instead takes its width out of the three panels, and
# then A's columns no longer sit over B's — which is the one thing this figure
# has to get right, since A and B are the same three pairs. The saved bbox is
# tight, so what hangs past the edge still lands in the file.
bar, cax = outside_colorbar(sf_a, ScalarMappable(norm=day_norm, cmap=day_colours), "Training day")
bar.set_ticks([d for d in DAY_TICKS if min(all_days) <= d <= max(all_days)])
panel_label(sf_a, "A")

# --- B: the early-window AUC of every session, against day ---
axes_b = sf_b.subplots(1, len(PAIR_ORDER), sharey=True)  # one scale, as in A
jitter_rng = np.random.default_rng(0)  # only so overlapping sessions stay visible
by_pair = regression.set_index("pair")
for ax, pair in zip(axes_b, PAIR_ORDER):
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
axes_b[0].set_ylim(0, sessions["auc"].max() * Y_HEADROOM)
axes_b[0].yaxis.set_major_formatter(decimal_formatter(Y_TICK_DECIMALS))
# A distance in arbitrary units integrated over seconds. The unit was left to
# the caption until the co-authors asked for it on the axis (2026-10-26).
axes_b[0].set_ylabel("Early-window AUC (a.u.·s)")
panel_label(sf_b, "B")

# --- C: the slope of each line, with its interval ---
ax_c = sf_c.subplots()
ypos = np.arange(len(PAIR_ORDER))[::-1]  # success-miss on top, as everywhere else
for y, pair in zip(ypos, PAIR_ORDER):
    r = by_pair.loc[pair]
    ax_c.errorbar(r["slope_per_day"], y,
                  xerr=[[r["slope_per_day"] - r["ci95_lo"]], [r["ci95_hi"] - r["slope_per_day"]]],
                  fmt="o", ms=7, lw=1.5, capsize=3, color=PAIR_COLOR[pair],
                  markerfacecolor=PAIR_COLOR[pair] if r["q_value"] < SIG_CUTOFF else "white",
                  markeredgewidth=1.5)
    ax_c.annotate(f"{signed(r['slope_per_day'], 4, plus=True)}  (n = {int(r['n_sessions'])})\n"
                  f"p = {fmt_p(r['p_value'])}, q = {with_star(r['q_value'])}",
                  xy=(r["ci95_hi"], y), xytext=(8, 0), textcoords="offset points",
                  va="center", ha="left", fontsize=TICK_SIZE)
ax_c.axvline(0, ls="--", lw=0.8, color="black")
ax_c.set_yticks(ypos)
ax_c.set_yticklabels([PAIR_LABEL[p] for p in PAIR_ORDER])
ax_c.set_ylim(-0.6, len(PAIR_ORDER) - 0.4)
low = min(0.0, regression["ci95_lo"].min())
high = regression["ci95_hi"].max()
span = high - low
ax_c.set_xlim(low - 0.08 * span, high + 0.78 * span)  # room for the values at the right
ax_c.xaxis.set_major_formatter(decimal_formatter(3))
# "against training day", not "per training day": a slope already carries the
# "per", so the second one reads as though the AUC itself were a per-day
# quantity. The unit is B's per day, on the axis as the co-authors asked.
ax_c.set_xlabel("Slope of early-window AUC against training day (a.u.·s/day)")
ax_c.set_title("Early-window trend per pair", fontsize=LABEL_SIZE)
# What a filled circle and an asterisk mean, and that the three regressions
# are separate and their slopes uncompared, all belong in the caption. Prose
# set inside a panel costs space and ends up at a size nobody reads.
clean_axes(ax_c)
panel_label(sf_c, "C")

# Let the layout engine settle the panels, then put A's columns over B's. The
# two rows do not reserve the same margins on their own: A's last x tick sits
# on the frame at t = 4 s and hangs over the edge, which costs that row some
# width, while B's last tick at day 15 sits inside its limit of 16.
fig.draw_without_rendering()
assert sf_a.bbox.x0 == sf_b.bbox.x0 and sf_a.bbox.x1 == sf_b.bbox.x1, \
    "the two subfigures do not span the same width, so their axes cannot be compared"
for i, (x0, x1) in enumerate(uniform_columns(fig, axes_a, axes_b), start=1):
    print(f"  column {i}: [{x0:.4f}, {x1:.4f}]")
place_outside_colorbar(cax, axes_a[-1], gap=0.010, width=0.009)
print(f"  panel width {equal_panel_widths(axes_a):.4f} across the row")
print(f"  width with the colour bar: {assert_page_width(fig):.3f} inch (page is {WIDTH_2COL})")

save_fig(fig, "fig2_learning_related_change", OUT_DIR)
plt.show()

# %% [markdown]
# ## 4. Numbers quoted in the text

# %%
print("early-window AUC against training day, per pair")
for _, r in regression.iterrows():
    change = r[f"rel_change_day{REGRESSION_DAYS[0]}_to_day{REGRESSION_DAYS[1]}"] * 100
    print(f"  {PAIR_LABEL[r['pair']]:<17} slope {r['slope_per_day']:+.5f} "
          f"[{r['ci95_lo']:+.5f}, {r['ci95_hi']:+.5f}]  "
          f"p = {r['p_value']:.4g}, q = {r['q_value']:.4g}  "
          f"n = {int(r['n_sessions'])} sessions, {int(r['n_mice'])} mice")
    print(f"  {'':<17} day {REGRESSION_DAYS[0]} to {REGRESSION_DAYS[1]}: "
          f"{r[f'pred_day{REGRESSION_DAYS[0]}']:.4f} to "
          f"{r[f'pred_day{REGRESSION_DAYS[1]}']:.4f} ({change:+.1f}%)")

regression.to_csv(OUT_DIR / "fig2_regression_stats.csv", index=False)
print(f"\nsaved: {display_path(OUT_DIR)}/fig2_regression_stats.csv")
