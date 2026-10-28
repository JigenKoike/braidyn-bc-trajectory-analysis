# braidyn-bc-trajectory-analysis

Code and data for the figures and tables of *<paper title>* (<authors>,
<journal>, <year>).

The question is whether the cortical response to a trial separates by its
outcome — success, failure, or no response at all — and whether that
separation grows as a mouse learns. Widefield calcium imaging from 25 mice
over 364 training sessions is reduced to one shared PC space; the distance
between two outcomes inside that space is the measure everything else is
built on.

The data is the public [BraiDyn-BC](https://github.com/BraiDyn-BC) release,
about 6.4 TB of NWB files across 495 sessions; the analysis uses the 364
task sessions among them. This repository
carries the 43 MB those figures actually need, so that redrawing them is a
clone and a minute rather than a download and two days.

## Redrawing the figures

```bash
git clone https://github.com/JigenKoike/braidyn-bc-trajectory-analysis.git
cd braidyn-bc-trajectory-analysis

curl -LsSf https://astral.sh/uv/install.sh | sh    # if you do not have uv
uv sync --extra notebook --extra dev

.venv/bin/nbstripout --install --attributes .gitattributes   # once per clone

for nb in notebooks/paper/fig*.ipynb; do
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

`uv` fetches Python 3.11 and every dependency; nothing else is needed. The
five notebooks take **about a minute** together on a laptop and write to
`output/paper_figures/`. They read `data/paper/`, which is in this
repository, and nothing else.

Opening them in Jupyter works the same way — `.venv/bin/jupyter lab` — and
is the better way to read them, since each one explains what it computes
before computing it.

Without `uv`, `requirements.txt` holds the same 131 packages at the same
versions, exported from `uv.lock` and hash-pinned:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps
```

### What comes out

| Notebook | Figure | Tables and statistics |
|---|---|---|
| `fig1_pc_space_and_trajectories` | Figure 1 — the PC space and the outcome trajectories | |
| `fig2_learning_related_change` | Figure 2 — separation against training day | `fig2_regression_stats.csv` |
| `fig3_lmm_early_window` | Figure 3 — what the early-window separation goes with | **Table 2**, convergence log |
| `fig4_late_window` | Figure 4 — the same over the late window | **Table 3**, regression stats, convergence log |
| `fig5_robustness` | Figure 5 — whether the training-day effect holds up under other PC weights, matched trial counts and a random slope per mouse | the regressions and coefficients behind each panel |

Every figure is written as PNG, PDF and SVG at 300 dpi. The SVG keeps its
text as text, so a label can be moved in a vector editor without going back
to the notebook.

Each notebook also prints the numbers quoted in the manuscript, and checks
itself: Figures 3 and 4 compare their fits against the earlier run's CSVs
and stop if anything has moved.

## Rebuilding the data from the raw files

Not needed to redraw the figures. This is the path for checking that
`data/paper/` is what the raw data gives.

| Step | Command | Produces | Cost |
|---|---|---|---|
| 1 | `scripts/sync_s3_to_server.py` | the NWB files, about 6.4 TB | days, network-bound; the bucket is public, no credentials |
| 2 | `scripts/extract_session_data.py` | per-session ΔF/F, behaviour and trials — 42.5 GB | about two days |
| 3 | `scripts/extract_pull_duration.py` | per-trial pull durations | minutes |
| 4 | `scripts/run_pairwise_pca.py` | the PC space, the trajectories, the distances | reads 19.8 GB |
| 5 | `scripts/run_photobleaching.py` | pre-cue fluorescence per session | 35 seconds |
| 6 | `scripts/extract_trial_projections.py` | every trial projected into the PC space, 80 MB | reads 19.8 GB, about a minute |
| 7 | `scripts/run_trial_subsampling.py` | the distance on ten trials of each outcome (Figure 5) | seconds |
| 8 | `scripts/build_paper_dataset.py` | `data/paper/` — 16 files, 43 MB | seconds |

Steps 1 and 2 want a machine with room for the raw files and a way to leave
a job running, such as `tmux`. Each script takes `--help`.

Then:

```bash
python3 scripts/build_paper_dataset.py --verify
```

rebuilds into a temporary folder, compares it against the copy in this
repository, and names anything that differs. It compares values rather than
bytes: an npz records the modification time of everything inside it and
parquet carries writer metadata, so two runs of the same script differ byte
for byte while meaning the same thing. A rebuild on another machine also
moves the last bits of a float around, since BLAS sums in a different order,
so the comparison allows a relative 1e-9 and reports anything larger.

## Layout

| Path | What is in it |
|---|---|
| `data/paper/` | the 43 MB the figures read — the only part of `data/` in git |
| `notebooks/paper/` | one notebook per figure, paired with a `py:percent` script |
| `src/analysis/` | the analysis: windows, mixed models, corrections, correlations |
| `src/figures/` | how every figure is laid out, coloured, sized and saved |
| `src/braidyn_io/` | reading the NWB files and the S3 bucket |
| `scripts/` | the pipeline above, from the bucket to `data/paper/` |
| `tests/` | the unit tests — run them with `.venv/bin/pytest` |
| `output/` | what the notebooks write (gitignored) |

A few conventions, in case you edit:

- **The notebooks define no analysis.** Windows, models, corrections,
  colours and type sizes all come from `src/`, so that two figures cannot
  disagree about what an "early window" is.
- **Each notebook is paired with a `.py`** of the same name, by the
  `jupytext.toml` in that directory. The `.ipynb` is what you run; the `.py`
  is what git can show a readable diff of. Edit either, run
  `jupytext --sync <file>`, commit both.
- **Notebook outputs are stripped on commit** by
  [nbstripout](https://github.com/kynan/nbstripout), declared in
  `.gitattributes` and installed by the command above.

## Statistics

Three analyses answer overlapping questions, and each corrects for multiple
comparisons over a family of its own. The captions say which; in short:

- **Figures 2 and 4**, the regression of separation on training day:
  Benjamini-Hochberg over the three outcome pairs, within one window,
  reported as q. The early and the late window are separate families.
- **Figures 3 and 4**, the mixed models: the likelihood-ratio test of
  dropping each predictor, Benjamini-Hochberg over the seven predictors of
  one model. The three pairs are separate models. Tables 2 and 3 also give
  the Wald test of each coefficient, corrected apart from the
  likelihood-ratio test; the intervals drawn are Wald intervals.
- **Figure 5**: in A and B, the same regressions as Figures 2 and 4, on the
  window area z-scored within each pair, Benjamini-Hochberg over the three
  pairs of one window and one measure; in C, the likelihood-ratio test of the
  mixed models of Figures 3 and 4, Benjamini-Hochberg over the seven
  predictors of one model. With the random slope, the reduced model keeps
  each mouse's slope, so the test of training day is of their mean.

Uncorrected p values are printed beside the corrected ones wherever both fit.

## Licence and attribution

- **Code** — MIT, Copyright (c) 2026 University of Fukui, written by Jigen
  Koike and Takahiro Yamauchi. See `LICENSE`.
- **Data in `data/paper/`** — CC-BY-4.0, except the file below.
- **`data/paper/AllenCCF_masks.h5`** — Allen CCF ROI masks, from
  [BraiDyn-BC/bdbc-nwb-explorer](https://github.com/BraiDyn-BC/bdbc-nwb-explorer)
  (MIT, © Keisuke Sehara). See `NOTICE`.
- **The raw data** is the BraiDyn-BC release; cite it as that project asks.

If you use this code or the packaged data, please cite the paper above.
