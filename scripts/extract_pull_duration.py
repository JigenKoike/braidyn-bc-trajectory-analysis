#!/usr/bin/env python3
"""Compute per-trial pull-duration features from state_lever.

For every task-day session under ``--derived``, compute ``pull_dur_first``
and ``pull_dur_max`` per trial using the 30 Hz binary ``state_lever``
signal in ``behavior.npz``, and write one parquet per session next to the
existing ``trials.parquet``:

    data/derived/per_session/<session_id>/pull_duration.parquet

This is a supplementary extraction; the original ``trials.parquet`` is
NOT modified.

Usage:
    python3 scripts/extract_pull_duration.py
    python3 scripts/extract_pull_duration.py --derived data/derived
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make our packages importable when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.pull_duration import compute_pull_durations  # noqa: E402
from braidyn_io.paths import parse_session_filename  # noqa: E402


def _log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--derived", type=Path, default=Path("data/derived"))
    ap.add_argument("--mice", type=str, default="",
                    help="comma-separated mouse IDs (default: all)")
    args = ap.parse_args()

    mice_filter = {m.strip() for m in args.mice.split(",") if m.strip()} or None

    session_dirs = sorted(d for d in (args.derived / "per_session").iterdir() if d.is_dir())
    n_written = 0
    n_skip = 0
    t0 = time.time()

    for sd in session_dirs:
        try:
            parsed = parse_session_filename(sd.name + ".nwb")
        except ValueError:
            n_skip += 1
            continue
        if parsed.task != "task":
            n_skip += 1
            continue
        if mice_filter and parsed.mouse not in mice_filter:
            continue
        beh_path = sd / "behavior.npz"
        trials_path = sd / "trials.parquet"
        if not beh_path.exists() or not trials_path.exists():
            n_skip += 1
            continue

        with np.load(beh_path) as beh:
            if "state_lever" not in beh.files:
                n_skip += 1
                continue
            sl = beh["state_lever"]
            ts = beh["timestamps"]

        trials = pd.read_parquet(trials_path)
        out = compute_pull_durations(sl, ts, trials)
        out.to_parquet(sd / "pull_duration.parquet")
        n_written += 1

    elapsed = time.time() - t0
    _log(f"processed {n_written} sessions in {elapsed:.1f}s  ({n_skip} skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
