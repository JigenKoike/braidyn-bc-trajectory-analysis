#!/usr/bin/env python3
"""Collect the data the paper's figures need into one folder, ``data/paper/``.

The analysis pipeline leaves its results spread over two trees and about
1,100 files, most of them one per session. Almost none of that is needed to
redraw the figures: 34 MB of it is, and this gathers that 34 MB into
``data/paper/`` as a dozen files, small enough to live in the repository
beside the notebooks that read it.

Nothing is recomputed. Everything written here is a copy, or a concatenation
of copies, of what ``run_pairwise_pca.py``, ``run_photobleaching.py``,
``run_trial_subsampling.py`` and the per-session extraction already produced.

Usage::

    python3 scripts/build_paper_dataset.py
    python3 scripts/build_paper_dataset.py --verify     # compare, write nothing

``--verify`` is for whoever rebuilds the pipeline from the raw NWB files: it
says whether what they produced matches the copy in the repository. It
compares values rather than bytes, because an npz stores the modification
time of everything inside it and a parquet carries writer metadata, so two
runs of this script differ byte for byte while meaning the same thing.

Output layout (under ``--out``)::

    pca/                            the fitted PC space, as run_pairwise_pca left it
    trajectories.npz                364 per-session files in one
    distance_timecourse.csv         the distance of every pair, over time
    distance_timecourse_10trials.csv  the same, on ten trials of each outcome (Figure 5)
    distance_timecourse_10trials_meta.json  how those were drawn
    projected_sessions.csv          how many trials of each outcome a session gave
    trials.parquet                  364 per-session files in one, trial by trial
    pull_duration.parquet           the same
    baseline_F_per_session.csv      pre-cue fluorescence, one row per session
    AllenCCF_masks.h5               the ROI masks the cortical maps are drawn on
    MANIFEST.json                   what each file is, how big, and its checksum
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from braidyn_io.paths import parse_session_filename  # noqa: E402

#: Files of the fitted PC space to copy as they are. ``pca_model.joblib`` is
#: deliberately absent: nothing reads it, it is a pickle of a scikit-learn
#: object and so breaks on a different version of the library, and
#: unpickling a file out of a downloaded repository is not a thing to ask of
#: a reader. Everything the notebooks use is in the .npy files beside it.
PCA_ARTIFACTS = (
    "global_grand_mean_roi_time.npy",
    "roi_weights.npy",
    "pca_components.npy",
    "pca_explained_variance_ratio.npy",
    "peth_time_axis_sec.npy",
    "roi_names.txt",
    "meta.json",
)

OUTCOMES = ("success", "failure", "miss")

#: A value may differ from the copy in the repository by this much, relative,
#: and still count as the same number. Rebuilding on another machine moves
#: the last bits around: BLAS sums in a different order.
VERIFY_TOLERANCE = 1e-9


def _log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def _task_sessions(derived: Path) -> list[Path]:
    """Task-day session directories that have both of the tables we need."""
    out = []
    for session_dir in sorted(derived.iterdir()):
        if not session_dir.is_dir():
            continue
        try:
            parsed = parse_session_filename(session_dir.name + ".nwb")
        except ValueError:
            continue
        if parsed.task != "task":
            continue
        if all((session_dir / f).exists() for f in ("trials.parquet", "pull_duration.parquet")):
            out.append(session_dir)
    return out


def _stack_per_session(session_dirs: list[Path], filename: str) -> pd.DataFrame:
    """One table from one per-session table each, keyed by session.

    ``session_id``, ``mouse`` and ``day`` are added so that the stacked
    file stands on its own: nothing downstream has to take a session apart
    by its name to find out whose it is.
    """
    frames = []
    for session_dir in session_dirs:
        parsed = parse_session_filename(session_dir.name + ".nwb")
        frame = pd.read_parquet(session_dir / filename)
        frame.insert(0, "day", int(parsed.day))
        frame.insert(0, "mouse", parsed.mouse.removeprefix("sub-"))
        frame.insert(0, "session_id", session_dir.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _stack_trajectories(traj_dir: Path, session_ids: list[str]) -> dict[str, np.ndarray]:
    """One archive from one file per session.

    Keys are ``<session_id>/<outcome>``. Not every session has all three
    outcomes — a session where one of them never happened has no trajectory
    for it — so the keys present say which ones do. ``time_vector`` is
    stored once; this raises if any session disagrees about it.
    """
    stacked: dict[str, np.ndarray] = {}
    time_vector = None
    for session_id in session_ids:
        path = traj_dir / f"trajs_{session_id}.npz"
        if not path.exists():
            continue
        with np.load(path) as arrays:
            if time_vector is None:
                time_vector = arrays["time_vector"]
            elif not np.array_equal(time_vector, arrays["time_vector"]):
                raise ValueError(f"{session_id}: a different time vector from the others")
            for outcome in OUTCOMES:
                if outcome in arrays.files:
                    stacked[f"{session_id}/{outcome}"] = arrays[outcome]
    if time_vector is None:
        raise ValueError(f"no trajectory files found under {traj_dir}")
    stacked["time_vector"] = time_vector
    return stacked


def _checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _describe(out: Path) -> dict:
    """What ended up in the folder, for MANIFEST.json."""
    files = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files[str(path.relative_to(out))] = {
                "bytes": path.stat().st_size,
                "sha256": _checksum(path),
            }
    return files


def _compare(built: Path, existing: Path) -> list[str]:
    """Differences between two builds, as sentences. Empty means they agree."""
    problems = []
    new, old = _describe(built), _describe(existing)
    for name in sorted(set(old) - set(new)):
        problems.append(f"{name}: in the existing copy, not in this build")
    for name in sorted(set(new) - set(old)):
        problems.append(f"{name}: in this build, not in the existing copy")

    for name in sorted(set(new) & set(old)):
        if new[name]["sha256"] == old[name]["sha256"]:
            continue  # identical; nothing to compare
        a, b = built / name, existing / name
        if name.endswith(".csv"):
            problems += _compare_frames(pd.read_csv(a), pd.read_csv(b), name)
        elif name.endswith(".parquet"):
            problems += _compare_frames(pd.read_parquet(a), pd.read_parquet(b), name)
        elif name.endswith(".npy"):
            problems += _compare_arrays(np.load(a), np.load(b), name)
        elif name.endswith(".npz"):
            with np.load(a) as x, np.load(b) as y:
                if set(x.files) != set(y.files):
                    problems.append(f"{name}: holds different sessions")
                else:
                    for key in sorted(x.files):
                        problems += _compare_arrays(x[key], y[key], f"{name}[{key}]")
        else:
            problems.append(f"{name}: differs, and is not a format this can compare")
    return problems


def _compare_arrays(new: np.ndarray, old: np.ndarray, name: str) -> list[str]:
    if new.shape != old.shape:
        return [f"{name}: shape {new.shape} against {old.shape}"]
    if new.dtype.kind not in "fc":
        return [] if np.array_equal(new, old) else [f"{name}: values differ"]
    scale = np.maximum(np.abs(old), 1e-300)
    worst = float(np.max(np.abs(new - old) / scale))
    return [] if worst < VERIFY_TOLERANCE else [f"{name}: relative difference {worst:.2e}"]


def _compare_frames(new: pd.DataFrame, old: pd.DataFrame, name: str) -> list[str]:
    if list(new.columns) != list(old.columns):
        return [f"{name}: different columns"]
    if len(new) != len(old):
        return [f"{name}: {len(new)} rows against {len(old)}"]
    problems = []
    for column in new.columns:
        if new[column].dtype.kind in "fc":
            problems += _compare_arrays(new[column].to_numpy(), old[column].to_numpy(),
                                        f"{name}.{column}")
        elif not new[column].equals(old[column]):
            problems.append(f"{name}.{column}: values differ")
    return problems


def build(source: Path, out: Path) -> dict:
    """Write the whole folder. Returns its manifest."""
    pca_src = source / "output/pca_pairwise/global_pca_artifacts"
    traj_src = source / "output/pca_pairwise/trajectory_arrays"
    derived = source / "data/derived/per_session"

    out.mkdir(parents=True, exist_ok=True)
    (out / "pca").mkdir(exist_ok=True)
    for name in PCA_ARTIFACTS:
        shutil.copy2(pca_src / name, out / "pca" / name)
    _log(f"pca/: {len(PCA_ARTIFACTS)} files")

    sessions = _task_sessions(derived)
    _log(f"{len(sessions)} task sessions with both tables")

    trajectories = _stack_trajectories(traj_src, [d.name for d in sessions])
    np.savez_compressed(out / "trajectories.npz", **trajectories)
    _log(f"trajectories.npz: {len(trajectories) - 1} trajectories from "
         f"{len({k.split('/')[0] for k in trajectories if '/' in k})} sessions")

    for filename, label in (("trials.parquet", "trials"),
                            ("pull_duration.parquet", "pull durations")):
        table = _stack_per_session(sessions, filename)
        table.to_parquet(out / filename, index=False)
        _log(f"{filename}: {len(table):,} rows of {label} from {len(sessions)} sessions")

    for src, dst in (
        (source / "output/pca_pairwise/distances/pairwise_distance_timecourse.csv",
         out / "distance_timecourse.csv"),
        (source / "output/pca_pairwise/distances/distance_timecourse_10trials.csv",
         out / "distance_timecourse_10trials.csv"),
        (source / "output/pca_pairwise/distances/distance_timecourse_10trials_meta.json",
         out / "distance_timecourse_10trials_meta.json"),
        (source / "output/pca_pairwise/projected_sessions.csv",
         out / "projected_sessions.csv"),
        (source / "output/photobleaching/baseline_F_per_session.csv",
         out / "baseline_F_per_session.csv"),
        (source / "data/external/AllenCCF_masks.h5", out / "AllenCCF_masks.h5"),
    ):
        shutil.copy2(src, dst)
        _log(f"{dst.name}: copied")

    manifest = {
        "built_by": "scripts/build_paper_dataset.py",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_task_sessions": len(sessions),
        "files": _describe(out),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(f["bytes"] for f in manifest["files"].values())
    _log(f"MANIFEST.json: {len(manifest['files'])} files, {total / 1e6:.1f} MB")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=_REPO_ROOT,
                    help="repository holding output/ and data/derived/ (default: this one)")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "data/paper")
    ap.add_argument("--verify", action="store_true",
                    help="build into a temporary folder and compare with --out; write nothing")
    args = ap.parse_args()

    if not args.verify:
        build(args.source, args.out)
        _log(f"written to {args.out}")
        return 0

    if not args.out.exists():
        _log(f"nothing to compare against: {args.out} does not exist")
        return 1
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        build(args.source, Path(tmp) / "paper")
        problems = _compare(Path(tmp) / "paper", args.out)
    if problems:
        _log(f"{len(problems)} difference(s) from the existing copy:")
        for p in problems:
            print(f"  {p}")
        return 1
    _log(f"this build agrees with {args.out}, to a relative {VERIFY_TOLERANCE:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
