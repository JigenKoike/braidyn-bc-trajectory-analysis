# %% [markdown]
# # Figure 1 — The PC space, and how the three outcomes move apart inside it
#
# Every later figure measures a distance in one shared coordinate system.
# This one shows what that system is, what it was built from, and what the
# three trial outcomes look like inside it.
#
# **What is computed.** One PCA, fitted to the grand mean response pooled over
# all task sessions and all three outcomes. Each outcome's mean trajectory is
# projected into the first three components, and the Euclidean distance
# between the two outcomes of each pair is taken over time. Nothing is tested
# here; the figures that follow do the testing.
#
# | Panel | Content |
# |---|---|
# | **A** | Grand mean PETH across sessions (44 ROIs × time) — the input to the PCA |
# | **B** | Time profiles of PC1–PC3 |
# | **C** | PC1–PC3 ROI weights on the Allen CCF dorsal map |
# | **D** | Mean trajectory of each outcome in the PC1–PC2 plane |
# | **E** | Distance between each outcome pair in PC1–3 space, over time |
#
# **Inputs**
#
# | File | Made by |
# |---|---|
# | `data/paper/pca/` | `scripts/run_pairwise_pca.py`, via `build_paper_dataset.py` |
# | `data/paper/trajectories.npz` | the same |
# | `data/paper/projected_sessions.csv` | the same |
# | `data/paper/distance_timecourse.csv` | the same |
# | `data/paper/AllenCCF_masks.h5` | downloaded — Allen CCF ROI masks (MIT, © Keisuke Sehara) |

# %%
import sys
from pathlib import Path

_p = Path.cwd().resolve()
while _p != _p.parent and not (_p / "pyproject.toml").exists():
    _p = _p.parent
REPO_ROOT = _p
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis.pairs import OUTCOMES, PAIR_ORDER
from analysis.windows import EARLY_WINDOW, LATE_WINDOW
from figures.palette import (
    DISTANCE_LABEL,
    DIVERGING_CMAP,
    OUTCOME_COLOR,
    PAIR_COLOR,
    PAIR_LABEL,
    SEQUENTIAL_CMAP,
)
from figures.style import (
    display_path,
    DENSE_TICK_SIZE,
    LABEL_SIZE,
    TICK_SIZE,
    WIDTH_2COL,
    clean_axes,
    panel_label,
    save_fig,
    size_colorbar,
    staggered_tick_labels,
    use_paper_style,
)

use_paper_style()

DATA_DIR = REPO_ROOT / "data/paper"
ART_DIR = DATA_DIR / "pca"
TRAJ_FILE = DATA_DIR / "trajectories.npz"
DIST_CSV = DATA_DIR / "distance_timecourse.csv"
MASK_FILE = DATA_DIR / "AllenCCF_masks.h5"
OUT_DIR = REPO_ROOT / "output/paper_figures"

N_PC_SHOWN = 3  # panels B, C and D show the leading three components
MIN_TRIALS_PANEL_D = 2  # a session contributes to panel D only if every outcome has this many

print(f"output: {display_path(OUT_DIR)}")

# %% [markdown]
# ## 1. The fitted PC space
#
# The PCA is fitted once, on the grand mean PETH pooled over all task
# sessions and all three outcomes. `roi_weights` are the ROI scores scaled by
# the singular values, which is the basis every trajectory is projected onto.

# %%
grand_mean = np.load(ART_DIR / "global_grand_mean_roi_time.npy")  # (n_roi, n_time)
roi_weights = np.load(ART_DIR / "roi_weights.npy")  # (n_roi, n_pc)
components = np.load(ART_DIR / "pca_components.npy")  # (n_pc, n_time)
explained = np.load(ART_DIR / "pca_explained_variance_ratio.npy")  # (n_pc,)
peth_t = np.load(ART_DIR / "peth_time_axis_sec.npy")  # (n_time,)
roi_names = (ART_DIR / "roi_names.txt").read_text().splitlines()

print(f"grand mean PETH : {grand_mean.shape}  (ROI x time)")
print(f"ROI weights     : {roi_weights.shape}")
print(f"PC time profiles: {components.shape}")
print(f"time axis       : [{peth_t.min():.2f}, {peth_t.max():.2f}] s, {peth_t.size} points")
print("explained variance: " + ", ".join(
    f"PC{i + 1} {explained[i] * 100:.1f}%" for i in range(N_PC_SHOWN)))

# %% [markdown]
# ### Why panels B and C are on such different scales
#
# The PCA is fitted on 44 ROIs by 151 time points, and scikit-learn reads rows
# as samples — so the samples here are **ROIs**, a component is a time-course
# shape, and each ROI gets a score saying how much of each shape it holds:
#
#     ROI i time course = mean + W[i,1]·profile_1 + W[i,2]·profile_2 + ...
#
# Panel **B** draws the profiles, panel **C** the scores. The profiles are
# given unit length, so all of the amplitude sits in the scores: ‖W_k‖ comes
# to 0.741 / 0.246 / 0.067, the square roots of the explained variances. That
# is why B's three rows look comparable while C's three maps differ tenfold in
# span, and why C carries a colour bar per component rather than one shared.
#
# Panel **D** projects each outcome's PETH onto the *scores*, so for D the
# scores are the basis and C is the ruler, not B. For the grand mean the two
# views collapse into one (checked: 2e-7 relative), so D would say nothing new
# if it drew the grand mean. What it draws is the three outcomes departing
# from that shared curve — by 53 to 97 per cent of its own magnitude,
# depending on the component — and panel **E** measures that departure.
#
# %% [markdown]
# ## 2. Panel D — mean trajectory per outcome
#
# `run_pairwise_pca.py` already stored, for each session and outcome, the
# trial-averaged PETH projected into the PC space. Projection is linear, so
# averaging trials before projecting gives the same trajectory as projecting
# each trial and averaging after; this was checked against the per-trial path
# over 287 sessions and agreed to float32 storage precision.
#
# A session contributes only if all three outcomes have at least
# `MIN_TRIALS_PANEL_D` trials, so each outcome's mean rests on the same
# sessions.

# %%
sessions = pd.read_csv(DATA_DIR / "projected_sessions.csv")
enough = sessions[
    (sessions[[f"n_{oc}" for oc in OUTCOMES]] >= MIN_TRIALS_PANEL_D).all(axis=1)
]

# One archive holds every session's trajectory, keyed "<session>/<outcome>".
per_outcome: dict[str, list[np.ndarray]] = {oc: [] for oc in OUTCOMES}
with np.load(TRAJ_FILE) as arrays:
    for session_id in enough["session_id"]:
        for oc in OUTCOMES:
            per_outcome[oc].append(arrays[f"{session_id}/{oc}"][:, :N_PC_SHOWN])
mean_trajectory = {oc: np.mean(per_outcome[oc], axis=0) for oc in OUTCOMES}

print(f"sessions with >= {MIN_TRIALS_PANEL_D} trials in every outcome: "
      f"{len(enough)} / {len(sessions)}, {enough['mouse'].nunique()} mice")

# %% [markdown]
# ## 3. Panel E — distance between outcome pairs
#
# The grand mean over sessions of the PC1–PC3 Euclidean distance between the
# two outcomes of each pair. The shaded band is the early response window the
# later figures summarize.

# %%
distance_long = pd.read_csv(DIST_CSV)
distance_long = distance_long[distance_long["pair"].isin(PAIR_ORDER)]
grand_distance = distance_long.groupby(["pair", "t"], as_index=False)["dist"].mean()
sessions_per_pair = distance_long.groupby("pair")["session_id"].nunique()

print(distance_long.groupby("pair")
      .agg(sessions=("session_id", "nunique"), mice=("mouse", "nunique"))
      .reindex(list(PAIR_ORDER)).to_string())

# %% [markdown]
# ## 4. Panel C — ROI masks on the Allen CCF dorsal map
#
# Loaded the way `bdbc_nwb_explorer` loads them: key is `<name>_<side initial>`,
# a mask is where the stored array is positive, and the outline entries are
# skipped. Filling each mask with its ROI's weight leaves the gaps between
# masks unpainted, and those gaps read as the boundary lines.

# %%
ccf_masks = {}
with h5py.File(MASK_FILE, "r") as src:
    for key in sorted(src["masks"].keys()):
        entry = src["masks"][key]
        if "outline" in entry.attrs["name"]:
            continue
        ccf_masks[f"{entry.attrs['name']}_{entry.attrs['side'][0].lower()}"] = np.array(entry) > 0

missing = sorted(set(roi_names) - set(ccf_masks))
assert not missing, f"ROIs without a CCF mask: {missing}"

atlas_masks = np.stack([ccf_masks[name] for name in roi_names])  # same order as roi_weights
overlapping = int((atlas_masks.sum(axis=0) > 1).sum())
assert overlapping == 0, f"{overlapping} pixels are claimed by more than one ROI"
print(f"masks: {atlas_masks.shape}, no overlapping pixels")


def pc_weight_image(pc: int) -> np.ndarray:
    """An atlas-shaped image with each ROI's area filled by its PC weight."""
    img = np.full(atlas_masks.shape[1:], np.nan, dtype=np.float32)
    for i in range(len(roi_names)):
        img[atlas_masks[i]] = roi_weights[i, pc]
    return img


# %% [markdown]
# ## 5. The composite figure
#
# Three rows, each its own subfigure: A|B, then C, then D|E. Nesting them this
# way lets constrained layout size each row independently, instead of letting
# panel A's 44 tick labels squeeze whatever sits beside it.
#
# The panels have very different natural shapes — A is 44 rows deep, B is 3,
# and C is three squares — so the arrangement follows from that. A needs
# height and gets a tall cell; B needs width and none of the height, so it is
# stretched to fill beside A rather than left floating above white space; C
# needs width most of all, because three square maps in half a page come out
# too small to read beside D and E, so it takes a row to itself.
#
# Each cortical map in C carries its own colour bar. A shared one would make
# PC3 — whose weights are an order of magnitude smaller than PC1's — read as
# a single flat colour.

# %%
fig = plt.figure(figsize=(WIDTH_2COL, 7.9), layout="constrained")
sf_top, sf_c, sf_bottom = fig.subfigures(3, 1, height_ratios=[1.10, 0.76, 1.0])
sf_a, sf_b = sf_top.subfigures(1, 2, width_ratios=[1.18, 1.0])

half_sample = (peth_t[1] - peth_t[0]) / 2  # so each column is centred on its time point
time_extent = [peth_t.min() - half_sample, peth_t.max() + half_sample]
TIME_TICKS = [-1, 0, 1, 2, 3, 4]  # the same on every panel with a time axis

# --- A: grand mean PETH, the input to the PCA ---
ax_a = sf_a.subplots()
im_a = ax_a.imshow(grand_mean, aspect="auto", cmap=SEQUENTIAL_CMAP,
                   extent=[*time_extent, len(roi_names), 0], interpolation="nearest")
ax_a.axvline(0, color="white", ls="--", lw=0.8)
ax_a.set_xticks(TIME_TICKS)
ax_a.set_xlabel("Time from cue (s)")
ax_a.set_yticks(np.arange(len(roi_names)) + 0.5)
ax_a.set_yticklabels(roi_names, fontsize=DENSE_TICK_SIZE)
ax_a.tick_params(axis="y", length=1.5, pad=1)
# 44 ROIs down one side: two staggered columns keep the names readable.
staggered_tick_labels(ax_a, "y", offset_points=34)
ax_a.set_title("Grand mean PETH", fontsize=LABEL_SIZE)
panel_label(sf_a, "A")
size_colorbar(sf_a.colorbar(im_a, ax=ax_a, fraction=0.045, pad=0.02),
              "ΔF/F (baseline-subtracted)")

# --- B: PC time profiles ---
ax_b = sf_b.subplots()
profiles = components[:N_PC_SHOWN]
im_b = ax_b.imshow(profiles, aspect="auto", cmap=SEQUENTIAL_CMAP,
                   extent=[*time_extent, N_PC_SHOWN, 0], interpolation="nearest")
ax_b.axvline(0, color="white", ls="--", lw=0.8)
ax_b.set_yticks(np.arange(N_PC_SHOWN) + 0.5)
ax_b.set_yticklabels([f"PC{i + 1} ({explained[i] * 100:.1f}%)" for i in range(N_PC_SHOWN)],
                     fontsize=TICK_SIZE)
ax_b.set_xticks(TIME_TICKS)
ax_b.set_xlabel("Time from cue (s)")
ax_b.set_title("PC time profiles", fontsize=LABEL_SIZE)
panel_label(sf_b, "B")
size_colorbar(sf_b.colorbar(im_b, ax=ax_b, fraction=0.045, pad=0.02),
              "Component value")

# --- C: PC weights on the cortical map ---
axes_c = sf_c.subplots(1, N_PC_SHOWN)
for pc, ax in enumerate(axes_c):
    vmax = np.abs(roi_weights[:, pc]).max()
    im_c = ax.imshow(pc_weight_image(pc), cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax,
                     interpolation="none")
    ax.set_axis_off()
    ax.set_title(f"PC{pc + 1}", fontsize=LABEL_SIZE)
    # Upright bars beside the maps: the maps are square in a wide cell, so the
    # space a bar would take underneath is going spare at their sides anyway.
    # Round steps rather than the two extremes, so the bars can be read off
    # at a glance; the span differs tenfold between PCs, which is the point.
    cb_c = sf_c.colorbar(im_c, ax=ax, fraction=0.055, pad=0.0)
    size_colorbar(cb_c)
# All three bars carry the same quantity, so it is named once underneath
# rather than three times wedged between each map and its bar.
sf_c.supxlabel("PC weight (a.u.)", fontsize=LABEL_SIZE)
panel_label(sf_c, "C")

# --- D, E ---
sf_d, sf_e = sf_bottom.subfigures(1, 2, width_ratios=[1.0, 1.3])
ax_d, ax_e = sf_d.subplots(), sf_e.subplots()

# Marked times: the two ends, the cue, and the early/late boundary the later
# figures split on.
MARKED_TIMES = [
    (peth_t.min(), "o", 38, f"\u2212{abs(peth_t.min()):.0f} s"),
    (0.0, "^", 34, "cue (0 s)"),
    (EARLY_WINDOW[1], "D", 20, f"+{EARLY_WINDOW[1]:.0f} s"),
    (peth_t.max(), "s", 38, f"+{peth_t.max():.0f} s"),
]
for oc in OUTCOMES:
    traj = mean_trajectory[oc]
    ax_d.plot(traj[:, 0], traj[:, 1], color=OUTCOME_COLOR[oc], lw=1.8, label=oc)
    for t_mark, marker, size, _ in MARKED_TIMES:
        i = int(np.argmin(np.abs(peth_t - t_mark)))
        ax_d.scatter(*traj[i, :2], color=OUTCOME_COLOR[oc], marker=marker, s=size,
                     edgecolor="black", linewidth=0.9, zorder=5)
# The two axes are not drawn to a common scale: PC2 spans an order of magnitude
# less than PC1, and forcing equal scaling flattens the trajectories into lines.
ax_d.set_xlabel("PC1 (a.u.)")
ax_d.set_ylabel("PC2 (a.u.)")
# A little headroom below the paths, so the marker key has somewhere to sit.
ax_d.set_ylim(bottom=ax_d.get_ylim()[0] - 0.0018)
ax_d.set_box_aspect(1)
clean_axes(ax_d)
# Outcomes at the empty upper left; the times along each path at the lower
# right, in the headroom opened above.
outcome_legend = ax_d.legend(frameon=False, fontsize=TICK_SIZE, loc="upper left")
ax_d.add_artist(outcome_legend)
marker_handles = [
    Line2D([], [], ls="none", marker=marker, ms=4.5, color="0.4",
           markeredgecolor="black", markeredgewidth=0.7, label=label)
    for _, marker, _, label in MARKED_TIMES
]
ax_d.legend(handles=marker_handles, ncol=2, frameon=False, fontsize=TICK_SIZE,
            loc="lower right", handletextpad=0.3, labelspacing=0.25,
            columnspacing=1.0, borderpad=0.1)
ax_d.set_title("Mean trajectories", fontsize=LABEL_SIZE)
panel_label(sf_d, "D")

for pair in PAIR_ORDER:
    g = grand_distance[grand_distance["pair"] == pair]
    ax_e.plot(g["t"], g["dist"], lw=1.8, color=PAIR_COLOR[pair],
              label=f"{PAIR_LABEL[pair]} (n = {sessions_per_pair[pair]})")
ax_e.axvline(0, ls="--", lw=0.8, color="black")
for window, name, shade in ((EARLY_WINDOW, "early", "0.87"), (LATE_WINDOW, "late", "0.94")):
    ax_e.axvspan(window[0], window[1], color=shade, zorder=0)
    ax_e.annotate(f"{name} window", xy=(np.mean(window[:2]), 0.015),
                  xycoords=("data", "axes fraction"), ha="center", va="bottom",
                  fontsize=TICK_SIZE, color="0.35")
ax_e.set_ylim(top=None)
ax_e.set_xlabel("Time from cue (s)")
ax_e.set_ylabel(DISTANCE_LABEL)
ax_e.set_xticks(TIME_TICKS)
ax_e.set_xlim(-1, 4)
ax_e.set_ylim(bottom=0)
clean_axes(ax_e)
ax_e.legend(frameon=False, fontsize=TICK_SIZE, loc="upper right")
ax_e.set_title("Separation between outcome pairs", fontsize=LABEL_SIZE)
panel_label(sf_e, "E")

save_fig(fig, "fig1_pc_space_and_trajectories", OUT_DIR)
plt.show()

# %% [markdown]
# ## 6. Numbers quoted in the text
#
# Marker key for panel D: circle = window start (−1 s), triangle = cue (0 s),
# square = window end (+4 s).

# %%
print("explained variance")
for i in range(N_PC_SHOWN):
    print(f"  PC{i + 1}: {explained[i] * 100:.2f}%")
print(f"  PC1-{N_PC_SHOWN} cumulative: {explained[:N_PC_SHOWN].sum() * 100:.2f}%")

print(f"\npanel D: {len(enough)} sessions, {enough['mouse'].nunique()} mice")
print("\npanel E: peak of the grand mean distance")
for pair in PAIR_ORDER:
    g = grand_distance[grand_distance["pair"] == pair]
    peak = g.loc[g["dist"].idxmax()]
    print(f"  {PAIR_LABEL[pair]:<17} {peak['dist']:.4f} at t = {peak['t']:+.2f} s "
          f"(n = {sessions_per_pair[pair]} sessions)")

print("\ntop-5 ROIs by |weight| per PC")
for pc in range(N_PC_SHOWN):
    order = np.argsort(-np.abs(roi_weights[:, pc]))[:5]
    named = ", ".join(f"{roi_names[i]} ({roi_weights[i, pc]:+.3f})" for i in order)
    print(f"  PC{pc + 1}: {named}")
