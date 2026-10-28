#!/usr/bin/env python3
"""Pairwise PCA distance pipeline: fit on success + failure + miss.

One PC space is fitted to all three outcomes together. Each outcome's PETH
is then projected into it per session, and the Euclidean distance between
the two members of each pair is taken over time, for **all three pairs**:

    success–miss   (matches the paper's original comparison)
    success–failure
    failure–miss

The PC space is fit on the trial-count-weighted union PETH per session
(what :func:`union_peth_matrix` returns), which — under (a) trial-mean
weighting — is exactly what a single ``session_peth_matrix`` call on
the pooled cue times would return.

Usage:
    python3 scripts/run_pairwise_pca.py
    python3 scripts/run_pairwise_pca.py --min-trials 5

Output layout (under ``--out``):
    global_pca_artifacts/
        meta.json
        global_grand_mean_roi_time.npy    # M_global (n_roi, n_time) float32
        roi_weights.npy                    # W (n_roi, n_pc)
        pca_components.npy                 # PCA components (n_pc, n_time)
        pca_explained_variance_ratio.npy
        peth_time_axis_sec.npy
        roi_names.txt
        pca_model.joblib
    trajectory_arrays/
        trajs_<session_id>.npz             # time_vector, success, failure, miss
    projected_sessions.csv                 # one row per session, n_ per outcome
    distances/
        pairwise_distance_timecourse.csv   # long: session × pair × t

The distance timecourse is the single source of truth for every downstream
window measure; summarizing it over a window is the caller's job, using the
shared definitions in :mod:`analysis.windows`.
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

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.cue_align import cue_aligned_time_axis  # noqa: E402
from analysis.global_pca import (  # noqa: E402
    DEFAULT_FRAME_RATE,
    DEFAULT_POST_FRAMES,
    DEFAULT_PRE_FRAMES,
    condition_peth_matrices,
    fit_global_pca,
    pc_distance,
    project_into_pc_space,
    union_peth_matrix,
)
from analysis.outcomes import relabel_outcomes_to_paper  # noqa: E402
from analysis.windows import EARLY_WINDOW, window_auc  # noqa: E402
from braidyn_io.paths import SessionId, parse_session_filename  # noqa: E402


OUTCOMES = ("success", "failure", "miss")
PAIRS: tuple[tuple[str, str], ...] = (
    ("success", "miss"),
    ("success", "failure"),
    ("failure", "miss"),
)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def _short_mouse(mouse: str) -> str:
    return mouse.removeprefix("sub-")


def discover_sessions(
    derived_root: Path,
    mice_filter: set[str] | None = None,
) -> list[tuple[Path, SessionId]]:
    sessions: list[tuple[Path, SessionId]] = []
    for sd in sorted((derived_root / "per_session").iterdir()):
        if not sd.is_dir():
            continue
        try:
            parsed = parse_session_filename(sd.name + ".nwb")
        except ValueError:
            continue
        if parsed.task != "task":
            continue
        if mice_filter and parsed.mouse not in mice_filter:
            continue
        if not (sd / "dff.npz").exists() or not (sd / "trials.parquet").exists():
            continue
        sessions.append((sd, parsed))
    sessions.sort(key=lambda x: (x[1].mouse, x[1].date, x[1].day))
    return sessions


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--derived", type=Path, default=Path("data/derived"))
    ap.add_argument("--out", type=Path, default=Path("output/pca_pairwise"))
    ap.add_argument("--mice", type=str, default="")
    ap.add_argument("--pre-frames", type=int, default=DEFAULT_PRE_FRAMES)
    ap.add_argument("--post-frames", type=int, default=DEFAULT_POST_FRAMES)
    ap.add_argument("--frame-rate", type=float, default=DEFAULT_FRAME_RATE)
    ap.add_argument("--n-components", type=int, default=20)
    ap.add_argument("--n-pc-distance", type=int, default=3)
    ap.add_argument("--min-trials", type=int, default=3,
                    help="min trials per condition for a session-pair to contribute a distance (default 3)")
    args = ap.parse_args()

    mice_filter = {m.strip() for m in args.mice.split(",") if m.strip()} or None

    artifact_dir = args.out / "global_pca_artifacts"
    traj_dir = args.out / "trajectory_arrays"
    dist_dir = args.out / "distances"
    for d in (artifact_dir, traj_dir, dist_dir):
        d.mkdir(parents=True, exist_ok=True)

    sessions = discover_sessions(args.derived, mice_filter)
    _log(f"Found {len(sessions)} task-day session(s) under {args.derived}")
    if not sessions:
        _log("ERROR: no sessions to process")
        return 2

    first_sd, _ = sessions[0]
    roi_meta = json.loads((first_sd / "roi_metadata.json").read_text())
    roi_names = list(roi_meta["names"])
    n_roi = len(roi_names)

    # ------------------------------------------------------------------
    # Step 1: per-session condition PETHs → union → grand mean → PCA fit
    # ------------------------------------------------------------------
    _log("=== Step 1: per-session PETH (success + failure + miss union) ===")
    M_for_fit: list[np.ndarray] = []
    sessions_in_fit: list[tuple[Path, SessionId]] = []
    condition_peths_per_session: list[dict[str, tuple[np.ndarray | None, int]]] = []
    n_skip_step1 = 0
    t0 = time.time()

    for sd, parsed in sessions:
        dff_arr = np.load(sd / "dff.npz")
        dff = dff_arr["dff"]
        timestamps = dff_arr["timestamps"]
        dff_arr.close()
        trials = relabel_outcomes_to_paper(pd.read_parquet(sd / "trials.parquet"))

        cond_peths = condition_peth_matrices(
            dff, timestamps, trials,
            outcomes=OUTCOMES,
            pre_frames=args.pre_frames, post_frames=args.post_frames,
        )
        M_union = union_peth_matrix(cond_peths)
        if M_union is None:
            n_skip_step1 += 1
            continue

        M_for_fit.append(M_union)
        sessions_in_fit.append((sd, parsed))
        condition_peths_per_session.append(cond_peths)

    _log(f"  {len(M_for_fit)} sessions contributed to PCA fit "
         f"(skipped {n_skip_step1}; {time.time() - t0:.1f}s)")

    _log("=== fitting Global PCA (S + F + M union) ===")
    result = fit_global_pca(M_for_fit, n_components=args.n_components, random_state=42)
    _log(f"  M_global shape: {result.M_global.shape}")
    _log(f"  W shape:        {result.W.shape}")
    _log(f"  top-5 explained variance ratio: {result.pca.explained_variance_ratio_[:5]}")

    peth_t = cue_aligned_time_axis(args.pre_frames, args.post_frames, args.frame_rate)

    # Save PCA artifacts
    np.save(artifact_dir / "global_grand_mean_roi_time.npy",
            result.M_global.astype(np.float32))
    np.save(artifact_dir / "roi_weights.npy", result.W.astype(np.float32))
    np.save(artifact_dir / "pca_components.npy", result.pca.components_)
    np.save(artifact_dir / "pca_explained_variance_ratio.npy",
            result.pca.explained_variance_ratio_)
    np.save(artifact_dir / "peth_time_axis_sec.npy", peth_t.astype(np.float32))
    (artifact_dir / "roi_names.txt").write_text("\n".join(roi_names), encoding="utf-8")
    joblib.dump(result.pca, artifact_dir / "pca_model.joblib")

    mice_in_fit = sorted({_short_mouse(p.mouse) for _, p in sessions_in_fit})
    meta = {
        "n_sessions_in_fit": int(len(M_for_fit)),
        "n_mice_in_fit": int(len(mice_in_fit)),
        "mice": mice_in_fit,
        "outcomes_used_in_fit": list(OUTCOMES),
        "peth_weighting": "trial-mean (option a) — union across outcomes, weighted by trial count",
        "n_roi": int(n_roi),
        "pretrigger_frames": int(args.pre_frames),
        "posttrigger_frames": int(args.post_frames),
        "frame_rate": float(args.frame_rate),
        "n_pc_distance": int(args.n_pc_distance),
        "min_trials_per_condition_for_pair": int(args.min_trials),
        "pairs_computed": [f"{a}-{b}" for a, b in PAIRS],
        "derived_root": str(args.derived),
        "script": "scripts/run_pairwise_pca.py",
        "generated_at": time.strftime("%F %T"),
    }
    (artifact_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Step 2: per-session projection + pairwise distances
    # ------------------------------------------------------------------
    _log("=== Step 2: per-session projection + pairwise distances ===")
    rows_session: list[dict] = []
    rows_long: list[dict] = []
    n_pair_written: dict[str, int] = {f"{a}-{b}": 0 for a, b in PAIRS}
    early_auc: dict[str, list[float]] = {f"{a}-{b}": [] for a, b in PAIRS}  # for the summary log
    t0 = time.time()

    for (sd, parsed), cond_peths in zip(sessions_in_fit, condition_peths_per_session):
        sid = sd.name
        mouse_short = _short_mouse(parsed.mouse)

        # Project each outcome that has a valid PETH into PC space
        Z_per_outcome: dict[str, np.ndarray | None] = {}
        n_per_outcome: dict[str, int] = {}
        for oc in OUTCOMES:
            M, n = cond_peths[oc]
            n_per_outcome[oc] = n
            Z_per_outcome[oc] = (
                project_into_pc_space(M, result.W).astype(np.float32)
                if M is not None and n > 0 else None
            )

        # Save trajectories (skip outcome if projection missing)
        traj_data: dict[str, np.ndarray] = {"time_vector": peth_t.astype(np.float32)}
        for oc, Z in Z_per_outcome.items():
            if Z is not None:
                traj_data[oc] = Z
        np.savez_compressed(traj_dir / f"trajs_{sid}.npz", **traj_data)

        rows_session.append({
            "mouse": mouse_short, "day": parsed.day, "session_id": sid,
            "n_time": int(len(peth_t)),
            "n_success": n_per_outcome["success"],
            "n_failure": n_per_outcome["failure"],
            "n_miss":    n_per_outcome["miss"],
        })

        # Compute per-pair distance if both conditions meet the trial threshold
        for a, b in PAIRS:
            pair_name = f"{a}-{b}"
            if (n_per_outcome[a] < args.min_trials
                    or n_per_outcome[b] < args.min_trials
                    or Z_per_outcome[a] is None
                    or Z_per_outcome[b] is None):
                continue

            dist = pc_distance(
                Z_per_outcome[a].astype(np.float64),
                Z_per_outcome[b].astype(np.float64),
                n_pc=args.n_pc_distance,
            )

            for t_val, d_val in zip(peth_t, dist):
                rows_long.append({
                    "mouse": mouse_short, "day": parsed.day, "session_id": sid,
                    "pair": pair_name,
                    "t": float(t_val), "dist": float(d_val),
                })
            n_pair_written[pair_name] += 1
            early_auc[pair_name].append(float(window_auc(peth_t, dist, EARLY_WINDOW)))

    _log(f"  {len(rows_session)} sessions saved ({time.time() - t0:.1f}s)")
    for pair_name, n in n_pair_written.items():
        _log(f"  pair {pair_name:18s} distances written for {n} sessions")

    pd.DataFrame(rows_session).to_csv(args.out / "projected_sessions.csv", index=False)
    pd.DataFrame(rows_long).to_csv(
        dist_dir / "pairwise_distance_timecourse.csv", index=False
    )

    _log("=" * 60)
    _log(f"Step 1 (fit):  {len(M_for_fit)} sessions used, {n_skip_step1} skipped")
    _log(f"Step 2 (proj): {len(rows_session)} sessions, "
         f"{sum(n_pair_written.values())} pair-distance timecourses")
    _log(f"Mice in fit:   {len(mice_in_fit)}")
    for pair_name, aucs in early_auc.items():
        if aucs:
            a = np.asarray(aucs)
            _log(f"  {pair_name:18s} early-window AUC: [{a.min():.4f}, {a.max():.4f}] "
                 f"mean={a.mean():.4f}")
    _log(f"outputs in: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
