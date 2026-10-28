#!/usr/bin/env python3
"""
Download BraiDyn-BC NWB files from the public S3 bucket to a local directory.

Listing uses Python stdlib (no AWS credentials needed — bucket is public-read).
Downloading shells out to `curl -C -` so that transfers can resume from the
local byte position — required because an intermediate network hop between
this host and S3 caps single HTTPS connections at ~10.72 GB, which otherwise
truncates any task-session file (~15 GB each).

Skips files already present with matching size, and whitelists .nwb only.

Usage:
    # dry run: list folders and sizes, no download
    python3 scripts/sync_s3_to_server.py --dest /tmp/dummy --dry-run

    # single-mouse test
    python3 scripts/sync_s3_to_server.py --dest /path/to/dest --mice 'sub-VG1-GC#99'

    # full run (typical: on server under tmux)
    python3 scripts/sync_s3_to_server.py --dest /path/to/dest
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

# Make `braidyn_io` importable when running this script directly with system
# python (e.g. `python3 scripts/sync_s3_to_server.py ...` on the server),
# without requiring `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from braidyn_io.s3_fetch import S3_BUCKET_URL, list_files, list_folders  # noqa: E402


def _log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def _safe_destination(dest: Path, key: str) -> Path:
    """Where ``key`` is to be written under ``dest``, refusing to leave it.

    The key comes from the bucket listing, which is someone else's data. A
    key holding ``..`` would otherwise send the write outside ``dest`` —
    ``dest / "a/../../etc/x"`` resolves above it — and this script is meant
    to be run with enough room for terabytes, which usually means run
    somewhere that can write a lot of places.
    """
    resolved_dest = dest.resolve()
    target = (resolved_dest / key).resolve()
    if not target.is_relative_to(resolved_dest):
        raise ValueError(f"object key would be written outside {dest}: {key!r}")
    return target


def download_with_retries(url: str, path: Path, max_retries: int = 10) -> None:
    """Download URL to path via `curl -C -` (resumable).

    curl reads the local file size on each invocation and issues a
    `Range: bytes=N-` request starting from that byte. Transient drops
    (including the ~10.72 GB single-connection cap observed against this S3
    route) are recovered by curl's own --retry plus this outer retry loop.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err = ""
    for attempt in range(max_retries):
        cmd = [
            "curl",
            "-L",
            "-o", str(path),
            "-C", "-",
            "-f",
            "--silent",
            "--show-error",
            "--connect-timeout", "30",
            "--retry", "5",
            "--retry-delay", "5",
            "--speed-time", "60",
            "--speed-limit", "10000",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return
        last_err = (result.stderr or "").strip().splitlines()[-1] if result.stderr else ""
        if attempt == max_retries - 1:
            break
        wait = min(60, 5 * (2**attempt))
        _log(f"    curl attempt {attempt + 1}/{max_retries} rc={result.returncode}: {last_err[:200]}; retry in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"curl failed after {max_retries} attempts: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dest", required=True, type=Path, help="destination directory")
    ap.add_argument(
        "--mice",
        type=str,
        default="",
        help="comma-separated mouse IDs to restrict to (default: all)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="list folders with NWB counts and sizes, do not download",
    )
    args = ap.parse_args()

    mice_filter = {m.strip() for m in args.mice.split(",") if m.strip()}

    _log("Listing S3 folders...")
    folders = sorted(list_folders())
    if mice_filter:
        folders = [f for f in folders if f.rstrip("/") in mice_filter]
        if not folders:
            _log(f"ERROR: no matching folders for --mice {sorted(mice_filter)}")
            return 2
    _log(f"{len(folders)} folder(s) selected")

    if args.dry_run:
        _log("--- DRY RUN: folder sizes (no download) ---")
        grand_total_gb = 0.0
        for folder_prefix in folders:
            mouse = folder_prefix.rstrip("/")
            files = list_files(folder_prefix)
            nwb = [(k, s) for k, s in files if k.endswith(".nwb")]
            total_gb = sum(s for _, s in nwb) / (1024**3)
            grand_total_gb += total_gb
            print(f"  {mouse:30s}  {len(nwb):4d} NWB  {total_gb:8.2f} GB", flush=True)
        _log(f"TOTAL: {grand_total_gb:.2f} GB across {len(folders)} folders")
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    total_dl = total_skip = total_err = 0
    grand_start = time.time()

    for fi, folder_prefix in enumerate(folders, 1):
        mouse = folder_prefix.rstrip("/")
        _log(f"[{fi}/{len(folders)}] {mouse}")
        files = list_files(folder_prefix)
        nwb = [(k, s) for k, s in files if k.endswith(".nwb")]
        total_gb = sum(s for _, s in nwb) / (1024**3)
        _log(f"  {len(nwb)} NWB ({total_gb:.1f} GB), skipping {len(files) - len(nwb)} non-NWB")

        m_start = time.time()
        for i, (key, size) in enumerate(nwb, 1):
            local = _safe_destination(args.dest, key)
            local.parent.mkdir(parents=True, exist_ok=True)
            if local.exists() and local.stat().st_size == size:
                total_skip += 1
                continue
            url = f"{S3_BUCKET_URL}/{urllib.parse.quote(key)}"
            name = key.split("/")[-1]
            gb = size / (1024**3)
            print(f"  [{i}/{len(nwb)}] {name} ({gb:.2f} GB)...", end="", flush=True)
            try:
                t0 = time.time()
                download_with_retries(url, local)
                dt = time.time() - t0
                speed = size / (1024**2) / dt if dt > 0 else 0
                print(f" done ({dt:.0f}s, {speed:.0f} MB/s)", flush=True)
                total_dl += 1
            except Exception as e:
                print(f" ERROR: {e}", flush=True)
                total_err += 1

        _log(f"  {mouse} done in {(time.time() - m_start) / 60:.1f} min")

    elapsed = time.time() - grand_start
    _log("=" * 60)
    _log(f"Downloaded: {total_dl}  Skipped: {total_skip}  Errors: {total_err}")
    _log(f"Total time: {elapsed / 3600:.1f} h   Destination: {args.dest}")
    return 0 if total_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
