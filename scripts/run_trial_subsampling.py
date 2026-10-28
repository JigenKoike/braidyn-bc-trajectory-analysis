#!/usr/bin/env python3
"""The distance between two outcomes, measured on ten trials of each, per session.

The distance the paper plots is between two outcomes' trial means, and a
mean of fewer trials sits further from the other by chance alone. Misses
fall and failures rise over training, so that chance part drifts with
training too. Measuring every session on the same number of trials gives
every session the same chance part, which then cannot pass for an effect of
training: Figure 5's trial-count control.

For each session and outcome pair with at least ``--n-trials`` trials of
both outcomes, ten (by default) are drawn from each without replacement, the
distance between their means is taken over PC1–3 exactly as the pipeline
takes it over all of them, and this is repeated ``--n-draws`` times and
averaged (:func:`analysis.cv_distance.subsampled_distance`). Each session
and pair has its own random stream, seeded from ``--seed`` and its name, so
the result does not depend on the order sessions are visited in or on which
others are present.

Reads the per-trial projections ``extract_trial_projections.py`` wrote;
writes one long table in the same layout as the pipeline's
``pairwise_distance_timecourse.csv`` (mouse, day, session_id, pair, t,
dist), and a JSON file with the settings. Under a minute.

Usage:
    python3 scripts/run_trial_subsampling.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.cv_distance import N_PC_DISTANCE, subsampled_distance  # noqa: E402
from analysis.pairs import PAIRS  # noqa: E402
from run_pairwise_pca import _log  # noqa: E402

#: Trials of each outcome per draw. Ten keeps most sessions measurable while
#: being enough trials for a mean to mean something.
N_TRIALS = 10

#: Draws averaged per session and pair, as in the analysis that chose the
#: method (2026-10-14).
N_DRAWS = 10

#: The root of every session's random stream.
SEED = 20261023


def session_rng(seed: int, session_id: str, pair: str) -> np.random.Generator:
    """A random stream of its own for one session and pair."""
    return np.random.default_rng([seed, zlib.crc32(f"{session_id}/{pair}".encode())])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projections", type=Path,
                    default=Path("output/pca_pairwise/trial_projections"))
    ap.add_argument("--pca", type=Path,
                    default=Path("output/pca_pairwise/global_pca_artifacts"),
                    help="for the time axis, the same one the pipeline's distances carry")
    ap.add_argument("--out", type=Path,
                    default=Path("output/pca_pairwise/distances/distance_timecourse_10trials.csv"))
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    peth_t = np.load(args.pca / "peth_time_axis_sec.npy").astype(np.float64)
    sessions = pd.read_csv(args.projections / "trial_projection_sessions.csv")
    _log(f"{len(sessions)} sessions of trial projections in {args.projections}; "
         f"{args.n_trials} trials per outcome, {args.n_draws} draws, seed {args.seed}")

    rows, n_written, n_short = [], {}, {}
    t0 = time.time()
    for s in sessions.sort_values("session_id").itertuples():
        with np.load(args.projections / f"trials_{s.session_id}.npz") as z:
            # The saved axis is float32; the pipeline's is not. Same frames.
            assert np.allclose(z["time_vector"], peth_t, atol=1e-6), "time axes disagree"
            trials = {o: z[o] for o in ("success", "failure", "miss") if o in z}
        for a, b in PAIRS:
            pair = f"{a}-{b}"
            if a not in trials or b not in trials:
                continue
            if len(trials[a]) < args.n_trials or len(trials[b]) < args.n_trials:
                n_short[pair] = n_short.get(pair, 0) + 1
                continue
            dist = subsampled_distance(trials[a], trials[b], args.n_trials,
                                       n_pc=N_PC_DISTANCE, n_repeats=args.n_draws,
                                       rng=session_rng(args.seed, s.session_id, pair))
            rows.extend({"mouse": s.mouse, "day": s.day, "session_id": s.session_id,
                         "pair": pair, "t": float(t), "dist": float(d)}
                        for t, d in zip(peth_t, dist))
            n_written[pair] = n_written.get(pair, 0) + 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    # No timestamp: the same inputs and settings give the same file, so a
    # rebuild can be checked against the copy in data/paper byte for byte.
    meta = {
        "script": "scripts/run_trial_subsampling.py",
        "projections": str(args.projections),
        "n_trials_per_outcome": args.n_trials,
        "n_draws": args.n_draws,
        "seed": args.seed,
        "n_pc": N_PC_DISTANCE,
        "sessions_per_pair": n_written,
        "sessions_too_few_trials_per_pair": n_short,
    }
    args.out.with_name(args.out.stem + "_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    for a, b in PAIRS:
        pair = f"{a}-{b}"
        _log(f"  {pair:<16} {n_written.get(pair, 0)} sessions measured, "
             f"{n_short.get(pair, 0)} with fewer than {args.n_trials} trials of an outcome")
    _log(f"  written to {args.out} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
