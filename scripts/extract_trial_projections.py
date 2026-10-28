#!/usr/bin/env python3
"""Project every trial into the fitted PC space, one file per session.

``run_pairwise_pca.py`` keeps each outcome's trial *mean*, which is all the
distance between two outcomes needs — until the question is what the number
of trials does to that distance. The distance between two means carries the
error of both, and that error shrinks as trials are added, so a session with
few misses measures a longer distance than one with many even when nothing
about the brain differs. Separating that from a real effect needs the
trials, not their mean; see :mod:`analysis.cv_distance`.

So this makes a second pass over the same ΔF/F, aligning and baselining each
trial exactly as the pipeline does, and projecting it into the PC space the
pipeline fitted. Only the components the distance uses are kept, which is
what keeps the result small: 44 ROIs over a 30-minute recording become 3
components over a 5-second window, and 19.8 GB becomes about 90 MB.

The arithmetic is checked rather than assumed: a projection is linear, so
the mean of the trial projections has to equal the projection of the trial
mean — the trajectory the pipeline already saved. Every session is compared
against it and the worst disagreement is reported.

Reads ``data/derived`` and the PC space in ``--pca``; writes one npz per
session. About a minute.

Usage:
    python3 scripts/extract_trial_projections.py
    python3 scripts/extract_trial_projections.py --mice VG1-GC#105
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.cue_align import cue_align  # noqa: E402
from analysis.outcomes import relabel_outcomes_to_paper  # noqa: E402
from analysis.pairs import OUTCOMES  # noqa: E402
# One definition of "which folders are task sessions", shared with the
# pipeline rather than written out a second time and left to drift.
from run_pairwise_pca import _log, _short_mouse, discover_sessions  # noqa: E402

#: How far the mean of the trial projections may sit from the trajectory the
#: pipeline saved. The two are the same quantity by algebra; what separates
#: them is that the saved one was rounded to float32.
TRAJECTORY_TOLERANCE = 1e-5


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derived", type=Path, default=Path("data/derived"))
    ap.add_argument("--pca", type=Path,
                    default=Path("output/pca_pairwise/global_pca_artifacts"))
    ap.add_argument("--trajectories", type=Path,
                    default=Path("output/pca_pairwise/trajectory_arrays"),
                    help="the pipeline's trial-mean trajectories, to check against")
    ap.add_argument("--out", type=Path, default=Path("output/pca_pairwise/trial_projections"))
    ap.add_argument("--n-pc", type=int, default=3,
                    help="components to keep; the distance is taken over three")
    ap.add_argument("--mice", type=str, default="")
    args = ap.parse_args()

    mice_filter = {m.strip() for m in args.mice.split(",") if m.strip()} or None
    args.out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.pca / "meta.json").read_text())
    W = np.load(args.pca / "roi_weights.npy").astype(np.float64)
    peth_t = np.load(args.pca / "peth_time_axis_sec.npy")
    pre_frames = int(meta["pretrigger_frames"])
    post_frames = int(meta["posttrigger_frames"])
    assert len(peth_t) == pre_frames + post_frames + 1, "the time axis and the window disagree"
    assert W.shape[1] >= args.n_pc, f"the PC space holds {W.shape[1]} components"
    _log(f"PC space from {args.pca} ({meta['generated_at']}), "
         f"keeping {args.n_pc} of {W.shape[1]} components")

    sessions = discover_sessions(args.derived, mice_filter)
    _log(f"Found {len(sessions)} task-day session(s) under {args.derived}")
    if not sessions:
        _log("ERROR: no sessions to process")
        return 2

    rows, worst = [], (0.0, "")
    n_checked = 0
    t0 = time.time()
    for sd, parsed in sessions:
        sid = sd.name
        dff_arr = np.load(sd / "dff.npz")
        dff, timestamps = dff_arr["dff"], dff_arr["timestamps"]
        dff_arr.close()
        trials = relabel_outcomes_to_paper(pd.read_parquet(sd / "trials.parquet"))

        saved: dict[str, np.ndarray] = {"time_vector": peth_t.astype(np.float32)}
        counts: dict[str, int] = {}
        for outcome in OUTCOMES:
            cue_times = trials.loc[
                trials["trial_outcome_paper"] == outcome, "start_time"].to_numpy(float)
            slices, valid = cue_align(dff, timestamps, cue_times,
                                      pre_frames=pre_frames, post_frames=post_frames,
                                      baseline_subtract=True)
            counts[outcome] = len(valid)
            if len(valid) == 0:
                continue
            # (n_trials, n_time, n_roi) @ (n_roi, n_pc) -> (n_trials, n_time, n_pc),
            # the same projection project_into_pc_space applies to a PETH.
            saved[outcome] = (slices @ W[:, : args.n_pc]).astype(np.float32)

        np.savez_compressed(args.out / f"trials_{sid}.npz", **saved)
        rows.append({"mouse": _short_mouse(parsed.mouse), "day": parsed.day,
                     "session_id": sid, **{f"n_{o}": counts[o] for o in OUTCOMES}})

        # The check: mean over trials == the saved trajectory.
        traj_path = args.trajectories / f"trajs_{sid}.npz"
        if not traj_path.exists():
            continue
        with np.load(traj_path) as traj:
            for outcome in OUTCOMES:
                if outcome not in traj or outcome not in saved:
                    continue
                mine = saved[outcome].astype(np.float64).mean(axis=0)
                theirs = traj[outcome][:, : args.n_pc].astype(np.float64)
                scale = np.abs(theirs).max()
                rel = float(np.abs(mine - theirs).max() / scale) if scale else 0.0
                n_checked += 1
                worst = max(worst, (rel, f"{sid}/{outcome}"))

    written = pd.DataFrame(rows)
    written.to_csv(args.out / "trial_projection_sessions.csv", index=False)
    (args.out / "meta.json").write_text(json.dumps({
        "script": "scripts/extract_trial_projections.py",
        "generated_at": time.strftime("%F %T"),
        "pc_space": str(args.pca),
        "pc_space_generated_at": meta["generated_at"],
        "n_pc_kept": args.n_pc,
        "pretrigger_frames": pre_frames,
        "posttrigger_frames": post_frames,
        "n_sessions": len(rows),
        "n_trials": int(written[[f"n_{o}" for o in OUTCOMES]].to_numpy().sum()),
        "max_relative_difference_from_trajectories": worst[0],
    }, indent=2) + "\n")

    _log(f"  {len(rows)} sessions, "
         f"{int(written[[f'n_{o}' for o in OUTCOMES]].to_numpy().sum()):,} trials "
         f"({time.time() - t0:.1f}s)")
    size = sum(f.stat().st_size for f in args.out.glob("trials_*.npz"))
    _log(f"  {size / 1e6:.1f} MB written to {args.out}")
    _log(f"  checked against {n_checked} saved trajectories; "
         f"worst relative difference {worst[0]:.2e} ({worst[1]})")
    if worst[0] >= TRAJECTORY_TOLERANCE:
        _log(f"ERROR: that is beyond {TRAJECTORY_TOLERANCE:.0e}; the projection "
             "does not reproduce the pipeline's trajectories")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
