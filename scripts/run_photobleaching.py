#!/usr/bin/env python3
"""Run the photobleaching analysis on derived data.

For every task-day session under ``--derived``, compute per-ROI cue-aligned
raw F PETHs for the success and miss conditions, derive a session-level
baseline F value from the pre-cue window (:data:`analysis.windows.PRE_WINDOW`),
and write a summary CSV plus per-session F-PETH NPZ files. Optionally merges
with the Global PCA distance summary to produce a single
``f_vs_distance_merged.csv`` for correlation plotting.

Usage:
    python3 scripts/run_photobleaching.py
    python3 scripts/run_photobleaching.py --derived data/derived --out output/photobleaching

Output layout (under ``--out``):
    meta.json
    f_peth_per_roi/<session_id>.npz
        time_vector, F_blue_success, F_blue_miss, F_UV_success, F_UV_miss,
        n_pixels_per_roi
    baseline_F_per_session.csv           # one row per session
    f_vs_distance_merged.csv             # merged with PCA distance summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make our packages importable when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.cue_align import cue_aligned_time_axis  # noqa: E402
from analysis.global_pca import (  # noqa: E402
    DEFAULT_FRAME_RATE,
    DEFAULT_POST_FRAMES,
    DEFAULT_PRE_FRAMES,
)
from analysis.outcomes import relabel_outcomes_to_paper  # noqa: E402
from analysis.photobleaching import session_F_peth  # noqa: E402
from analysis.windows import PRE_WINDOW, window_mean  # noqa: E402
from braidyn_io.paths import parse_session_filename  # noqa: E402


def _log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def _short_mouse(mouse: str) -> str:
    return mouse.removeprefix("sub-")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--derived", type=Path, default=Path("data/derived"))
    ap.add_argument("--out", type=Path, default=Path("output/photobleaching"))
    ap.add_argument("--mice", type=str, default="",
                    help="comma-separated mouse IDs (default: all)")
    ap.add_argument("--pre-frames", type=int, default=DEFAULT_PRE_FRAMES)
    ap.add_argument("--post-frames", type=int, default=DEFAULT_POST_FRAMES)
    ap.add_argument("--frame-rate", type=float, default=DEFAULT_FRAME_RATE)
    ap.add_argument(
        "--pca-summary",
        type=Path,
        default=Path("output/pca_global/success_miss_distance/success_miss_distance_summary_per_session.csv"),
        help="if present, merge the F values with this CSV on session_id",
    )
    args = ap.parse_args()

    mice_filter = {m.strip() for m in args.mice.split(",") if m.strip()} or None

    peth_dir = args.out / "f_peth_per_roi"
    peth_dir.mkdir(parents=True, exist_ok=True)

    # Discover task-day sessions with widefield_F_per_roi available.
    sessions = []
    for sd in sorted((args.derived / "per_session").iterdir()):
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
        if not (sd / "widefield_F_per_roi.npz").exists():
            continue
        if not (sd / "trials.parquet").exists():
            continue
        sessions.append((sd, parsed))

    _log(f"Found {len(sessions)} task-day session(s)")
    if not sessions:
        _log("ERROR: no sessions to process")
        return 2

    peth_t = cue_aligned_time_axis(args.pre_frames, args.post_frames, args.frame_rate)
    rows: list[dict] = []
    t0 = time.time()

    for i, (sd, parsed) in enumerate(sessions, 1):
        sid = sd.name
        # Load per-ROI raw F for this session
        wf = np.load(sd / "widefield_F_per_roi.npz")
        F_blue = wf["F_blue"] if "F_blue" in wf.files else None
        F_UV = wf["F_UV"] if "F_UV" in wf.files else None
        timestamps = wf["timestamps"]
        n_pixels_per_roi = wf["n_pixels_per_roi"]
        wf.close()

        # Load + relabel trials
        trials = relabel_outcomes_to_paper(pd.read_parquet(sd / "trials.parquet"))
        cue_s = trials.loc[trials["trial_outcome_paper"] == "success", "start_time"].to_numpy()
        cue_m = trials.loc[trials["trial_outcome_paper"] == "miss", "start_time"].to_numpy()

        peth_dict: dict[str, np.ndarray] = {
            "time_vector": peth_t.astype(np.float32),
            "n_pixels_per_roi": n_pixels_per_roi,
        }
        row: dict = {
            "mouse": _short_mouse(parsed.mouse),
            "day": parsed.day,
            "session_id": sid,
            "n_success": int(len(cue_s)),
            "n_miss": int(len(cue_m)),
        }

        for ch_short, F in (("blue", F_blue), ("UV", F_UV)):
            if F is None:
                # Channel absent in this session (e.g. modality=widefield only)
                row[f"F_{ch_short}_pre_mean"] = float("nan")
                continue

            peth_s, n_valid_s = session_F_peth(F, timestamps, cue_s,
                                               args.pre_frames, args.post_frames)
            peth_m, n_valid_m = session_F_peth(F, timestamps, cue_m,
                                               args.pre_frames, args.post_frames)

            valid_peths = [p for p in (peth_s, peth_m) if p is not None]
            if not valid_peths:
                row[f"F_{ch_short}_pre_mean"] = float("nan")
                continue

            # Average the success and miss PETHs into one session-typical PETH.
            avg_peth = np.mean(np.stack(valid_peths, axis=0), axis=0)  # (window, n_roi)

            # Collapse to a brain-wide (union) F PETH, then take the pre-cue baseline.
            avg_union = (avg_peth * n_pixels_per_roi[np.newaxis, :]).sum(axis=1) / n_pixels_per_roi.sum()
            row[f"F_{ch_short}_pre_mean"] = float(window_mean(peth_t, avg_union, PRE_WINDOW))

            # Save per-ROI PETHs (success/miss separately) for later region-level analyses.
            if peth_s is not None:
                peth_dict[f"F_{ch_short}_success"] = peth_s.astype(np.float32)
            if peth_m is not None:
                peth_dict[f"F_{ch_short}_miss"] = peth_m.astype(np.float32)

        np.savez_compressed(peth_dir / f"{sid}.npz", **peth_dict)
        rows.append(row)

    elapsed = time.time() - t0
    _log(f"  processed {len(rows)} sessions in {elapsed:.1f}s "
         f"({elapsed / max(len(rows), 1):.2f}s/session)")

    # Write summary CSV
    df = pd.DataFrame(rows)
    csv_columns = [
        "mouse", "day", "session_id", "n_success", "n_miss",
        "F_blue_pre_mean", "F_UV_pre_mean",
    ]
    df = df[csv_columns]
    df.to_csv(args.out / "baseline_F_per_session.csv", index=False)

    # Merge with PCA distance if available
    if args.pca_summary.exists():
        df_dist = pd.read_csv(args.pca_summary)
        df_merged = pd.merge(df, df_dist, on=["mouse", "day", "session_id"], how="inner")
        df_merged.to_csv(args.out / "f_vs_distance_merged.csv", index=False)
        _log(f"  merged with PCA distance: {len(df)} F rows × {len(df_dist)} dist rows → {len(df_merged)} merged rows")
    else:
        _log(f"  PCA summary {args.pca_summary} not found — skipping merge")

    # Metadata
    mice_in_run = sorted(df["mouse"].unique())
    meta = {
        "n_sessions": int(len(df)),
        "n_mice": int(len(mice_in_run)),
        "mice": mice_in_run,
        "pretrigger_frames": int(args.pre_frames),
        "posttrigger_frames": int(args.post_frames),
        "frame_rate": float(args.frame_rate),
        "derived_root": str(args.derived),
        "script": "scripts/run_photobleaching.py",
        "generated_at": time.strftime("%F %T"),
    }
    (args.out / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Final summary
    _log("=" * 60)
    _log(f"sessions:  {len(df)} across {len(mice_in_run)} mice")
    if len(df) > 0:
        _log(f"F_blue pre_mean range: [{df['F_blue_pre_mean'].min():.0f}, "
             f"{df['F_blue_pre_mean'].max():.0f}]  mean={df['F_blue_pre_mean'].mean():.0f}")
        _log(f"F_UV   pre_mean range: [{df['F_UV_pre_mean'].min():.0f}, "
             f"{df['F_UV_pre_mean'].max():.0f}]  mean={df['F_UV_pre_mean'].mean():.0f}")
    _log(f"outputs in: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
