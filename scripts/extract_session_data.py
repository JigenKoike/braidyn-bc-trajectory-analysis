#!/usr/bin/env python3
"""Extract per-session derived data from BraiDyn-BC NWB files.

For each NWB file under ``--nwb-root``, writes a structured set of files
into ``--out``. Per-mouse files (subject metadata, ROI metadata, ROI masks)
are written once per mouse from the first session encountered.

All time series are saved at their native sampling rate (no cue-alignment).
Cue-aligned slices are generated locally from these files as needed.

Usage:
    # One session, to check the setup
    python3 scripts/extract_session_data.py \\
        --nwb-root /path/to/nwb \\
        --out /path/to/derived \\
        --mice 'sub-VG1-GC#48' \\
        --limit 1

    # All of them (hours to days; run it under tmux or screen)
    python3 scripts/extract_session_data.py \\
        --nwb-root /path/to/nwb \\
        --out /path/to/derived

Output layout (under --out):
    meta_global.json                              # extraction provenance
    per_mouse/<mouse_id>/
        subject.json                              # subject/animal metadata (mouse-level)
    per_session/<session_id>/
        session_meta.json                         # date, day, type, imaging params, ...
        trials.parquet                            # trial dataframe (skip if no trials)
        dff.npz                                   # dff, dff_B, dff_V + timestamps
        behavior.npz                              # downsampled 1D signals + timestamps
        keypoints.npz                             # downsampled DLC keypoints + timestamps
        roi_metadata.json                         # ROI names + descriptions for THIS session
        roi_masks.npz                             # masks, masks_V (if any), mask_union for THIS session
        widefield_F_per_roi.npz                   # per-ROI mean F (n_frames, n_roi) for blue + UV, using THIS session's masks

Note: ROI shapes vary slightly across sessions of the same mouse (the upstream
NWB pipeline runs segmentation per session). Storing per-session masks lets
widefield_F_per_roi use the correct mask, matching sandbox phaseB's behavior.

widefield_F_per_roi has the same (frames × ROIs) shape as dFF arrays, so it
preserves region-level information (the brain-wide union mean is derivable
locally as `F_blue.mean(axis=1)` etc. if needed).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

# Make braidyn_io importable when run directly without `pip install -e .`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pynwb import NWBHDF5IO  # noqa: E402

from braidyn_io.paths import SessionId, parse_session_filename  # noqa: E402


# 1D scalar signals to pull from processing/downsampled/. Each is read as
# {signal_name: data}; their shared timestamps go into a single timestamps key.
BEHAVIOR_SIGNALS = [
    "CO2_level", "air_pressure", "humidity", "room_temp",
    "lever", "lick", "lick_rate", "reward", "tone",
    "motion", "state_lever", "state_task",
]

# Eye / pupil features. (h5_path_under_downsampled, output_key_in_npz)
EYE_FEATURES = [
    ("eye_position/center_x", "eye_x"),
    ("eye_position/center_y", "eye_y"),
    ("pupil_tracking/diameter", "pupil_diameter"),
]

# Keypoint groups under processing/downsampled/. Each has child groups per keypoint
# with a `data` (N, 2) array. We drop confidence to keep size manageable.
KEYPOINT_GROUPS = [
    ("body_video_keypoints", "body"),
    ("eye_video_keypoints", "eye"),
    ("face_video_keypoints", "face"),
]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def _decode(val: Any) -> Any:
    """Decode bytes / object arrays to plain Python str / list."""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.ndarray):
        if val.dtype.kind in ("S", "O"):
            return [_decode(x) for x in val.tolist()]
        return val.tolist()
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    return val


def _read_scalar(h5_obj: Any, key: str, default: Any = None) -> Any:
    """Read a (possibly-bytes) scalar dataset and decode."""
    if key not in h5_obj:
        return default
    return _decode(h5_obj[key][()])


# ---------------------------------------------------------------------------
# Per-mouse extractors
# ---------------------------------------------------------------------------

def extract_subject(h5: h5py.File) -> dict:
    sub = h5.get("general/subject")
    if sub is None:
        return {}
    keys = [
        "subject_id", "species", "strain", "genotype",
        "sex", "age", "date_of_birth", "description", "weight",
    ]
    return {k: _read_scalar(sub, k) for k in keys}


def extract_roi_metadata_and_masks(h5: h5py.File) -> tuple[dict, dict]:
    """Return (roi_metadata_json_dict, roi_masks_npz_arrays_dict)."""
    rois_path = "processing/ophys/ImageSegmentation/ROIs"
    if rois_path not in h5:
        return {}, {}
    rois = h5[rois_path]
    names = _decode(rois["roi_name"][:])
    descriptions = _decode(rois["roi_description"][:])
    ids = rois["id"][:].tolist()
    masks = np.asarray(rois["image_mask"][:])  # (n_roi, H, W) bool

    rois_v_path = "processing/ophys/ImageSegmentation/ROIs_V"
    masks_v = None
    if rois_v_path in h5 and "image_mask" in h5[rois_v_path]:
        masks_v = np.asarray(h5[f"{rois_v_path}/image_mask"][:])

    meta = {
        "n_rois": int(len(names)),
        "ids": ids,
        "names": list(names),
        "descriptions": list(descriptions),
        "mask_shape": list(masks.shape[1:]),
    }
    arrays: dict[str, np.ndarray] = {
        "masks": masks,
        "mask_union": masks.any(axis=0),
    }
    if masks_v is not None:
        arrays["masks_V"] = masks_v
    return meta, arrays


# ---------------------------------------------------------------------------
# Per-session extractors
# ---------------------------------------------------------------------------

def extract_session_meta(h5: h5py.File, sid: SessionId, session_id: str) -> dict:
    out: dict[str, Any] = {
        "session_id": session_id,
        "mouse_id": sid.mouse,
        "date": sid.date.isoformat(),
        "session_type": sid.task,
        "day": sid.day,
        "recording_modality": sid.modality,
        "experimenter": _read_scalar(h5, "general/experimenter"),
        "institution": _read_scalar(h5, "general/institution"),
        "lab": _read_scalar(h5, "general/lab"),
        "notes": _read_scalar(h5, "general/notes"),
        "session_description": _read_scalar(h5, "session_description"),
        "session_start_time": _read_scalar(h5, "session_start_time"),
        "nwb_identifier": _read_scalar(h5, "identifier"),
        "timestamps_reference_time": _read_scalar(h5, "timestamps_reference_time"),
    }
    # Imaging planes (blue and UV)
    iplanes: dict[str, dict] = {}
    for plane_name in ("ImagingPlane_blue", "ImagingPlane_UV"):
        plane_path = f"general/optophysiology/{plane_name}"
        if plane_path not in h5:
            continue
        plane = h5[plane_path]
        iplanes[plane_name] = {
            "description": _read_scalar(plane, "description"),
            "excitation_lambda": _read_scalar(plane, "excitation_lambda"),
            "imaging_rate": _read_scalar(plane, "imaging_rate"),
            "indicator": _read_scalar(plane, "indicator"),
            "location": _read_scalar(plane, "location"),
            "grid_spacing": _read_scalar(plane, "grid_spacing"),
            "origin_coords": _read_scalar(plane, "origin_coords"),
        }
    out["imaging_planes"] = iplanes
    return out


def extract_trials(nwb_path: Path) -> pd.DataFrame | None:
    """Use pynwb (gives a clean DataFrame). Returns None if no trials present."""
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        if nwbfile.trials is None:
            return None
        return nwbfile.trials.to_dataframe()


def extract_dff(h5: h5py.File) -> dict[str, np.ndarray]:
    """Read dFF / dFF_B / dFF_V arrays + shared timestamps."""
    out: dict[str, np.ndarray] = {}
    base = "processing/ophys/DfOverF"
    if base not in h5:
        return out
    timestamps: np.ndarray | None = None
    for key in ("dFF", "dFF_B", "dFF_V"):
        ds_path = f"{base}/{key}/data"
        ts_path = f"{base}/{key}/timestamps"
        if ds_path in h5:
            out[key.lower()] = h5[ds_path][:]
            if timestamps is None and ts_path in h5:
                timestamps = h5[ts_path][:]
    if timestamps is not None:
        out["timestamps"] = timestamps
    return out


def extract_behavior(h5: h5py.File) -> dict[str, np.ndarray]:
    """Read 1D downsampled signals + eye/pupil features."""
    out: dict[str, np.ndarray] = {}
    base = "processing/downsampled"
    timestamps: np.ndarray | None = None
    for name in BEHAVIOR_SIGNALS:
        data_path = f"{base}/{name}/data"
        ts_path = f"{base}/{name}/timestamps"
        if data_path in h5:
            out[name] = h5[data_path][:]
            if timestamps is None and ts_path in h5:
                timestamps = h5[ts_path][:]
    for h5_subpath, out_name in EYE_FEATURES:
        data_path = f"{base}/{h5_subpath}/data"
        ts_path = f"{base}/{h5_subpath}/timestamps"
        if data_path in h5:
            out[out_name] = h5[data_path][:]
            if timestamps is None and ts_path in h5:
                timestamps = h5[ts_path][:]
    if timestamps is not None:
        out["timestamps"] = timestamps
    return out


def extract_keypoints(h5: h5py.File) -> dict[str, np.ndarray]:
    """Read all DLC keypoints' data arrays (drop confidence to limit size)."""
    out: dict[str, np.ndarray] = {}
    base = "processing/downsampled"
    timestamps: np.ndarray | None = None
    for grp_name, prefix in KEYPOINT_GROUPS:
        grp_path = f"{base}/{grp_name}"
        if grp_path not in h5:
            continue
        grp = h5[grp_path]
        for child_name in grp.keys():
            child = grp[child_name]
            if not isinstance(child, h5py.Group):
                continue
            if "data" not in child:
                continue
            ds = child["data"]
            if ds.ndim != 2:  # we only want (n_time, 2) position arrays
                continue
            out[f"{prefix}/{child_name}"] = ds[:]
            if timestamps is None and "timestamps" in child:
                timestamps = child["timestamps"][:]
    if timestamps is not None:
        out["timestamps"] = timestamps
    return out


def compute_widefield_F_per_roi(
    h5: h5py.File,
    masks: np.ndarray,
    chunk_frames: int = 1000,
) -> dict[str, Any]:
    """Stream-read widefield raw and compute per-frame, per-ROI mean F.

    For each ROI, the mean is taken over that ROI's mask pixels — so the
    output has the same (frames × ROIs) shape as the upstream dFF arrays,
    just expressing absolute fluorescence instead of ΔF/F. This lets
    photobleaching analysis be done per region (not only brain-wide), and
    the brain-wide union mean can be derived locally if needed.

    Memory budget: chunk_frames × H × W × 2 bytes (uint16) + small per-ROI
    extracts. 1000 × 288 × 288 × 2 ≈ 165 MB per chunk.

    Output keys:
        F_blue:  (n_frames, n_roi) float64
        F_UV:    (n_frames, n_roi) float64    (omitted if UV channel absent)
        timestamps: (n_frames,) float64       (shared between blue and UV)
        n_pixels_per_roi: (n_roi,) int64
    """
    n_roi, H, W = masks.shape
    masks_flat = masks.reshape(n_roi, -1)  # (n_roi, H*W) bool — view
    n_px_per_roi = masks_flat.sum(axis=1).astype(np.int64)  # (n_roi,)

    out: dict[str, Any] = {}
    timestamps_saved = False
    for ch in ("widefield_blue", "widefield_UV"):
        data_path = f"acquisition/{ch}/data"
        ts_path = f"acquisition/{ch}/timestamps"
        if data_path not in h5:
            continue
        data = h5[data_path]  # (n_frames, H, W) uint16
        n_frames = data.shape[0]
        f_per_roi = np.empty((n_frames, n_roi), dtype=np.float64)
        for start in range(0, n_frames, chunk_frames):
            end = min(start + chunk_frames, n_frames)
            chunk = data[start:end, :, :]  # (chunk, H, W) uint16
            chunk_flat = chunk.reshape(end - start, -1)  # (chunk, H*W) view of same buffer
            for roi_idx in range(n_roi):
                # Boolean indexing returns a (chunk, n_px_roi) copy; mean accumulates in float64.
                f_per_roi[start:end, roi_idx] = chunk_flat[:, masks_flat[roi_idx]].mean(
                    axis=1, dtype=np.float64
                )
        ch_short = "blue" if "blue" in ch else "UV"
        out[f"F_{ch_short}"] = f_per_roi
        if not timestamps_saved and ts_path in h5:
            out["timestamps"] = h5[ts_path][:]
            timestamps_saved = True
    out["n_pixels_per_roi"] = n_px_per_roi
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def process_session(nwb_path: Path, out_root: Path, skip_existing: bool = True) -> str:
    """Extract one session, writing outputs at per-file granularity.

    Each output is checked individually against `skip_existing`. ROI masks
    are extracted *per session* — not per mouse — because the upstream NWB
    pipeline's ImageSegmentation/ROIs varies slightly across sessions of
    the same mouse, and widefield_F_mean must use that session's own mask
    for the mean to be physically meaningful (matching sandbox phaseB).
    """
    sid = parse_session_filename(nwb_path)
    session_id = nwb_path.stem
    session_dir = out_root / "per_session" / session_id
    mouse_dir = out_root / "per_mouse" / sid.mouse
    session_dir.mkdir(parents=True, exist_ok=True)
    mouse_dir.mkdir(parents=True, exist_ok=True)

    # Per-file existence check (replaces the old all-or-nothing skip).
    need = {
        "subject":      not (mouse_dir / "subject.json").exists(),
        "session_meta": not (session_dir / "session_meta.json").exists(),
        "trials":       not (session_dir / "trials.parquet").exists(),
        "dff":          not (session_dir / "dff.npz").exists(),
        "behavior":     not (session_dir / "behavior.npz").exists(),
        "keypoints":    not (session_dir / "keypoints.npz").exists(),
        "roi":          not ((session_dir / "roi_metadata.json").exists()
                             and (session_dir / "roi_masks.npz").exists()),
        "widefield":    not (session_dir / "widefield_F_per_roi.npz").exists(),
    }
    if not skip_existing:
        need = {k: True for k in need}
    if not any(need.values()):
        return "skip (all outputs present)"

    # Trials via pynwb (opens NWB separately for clean DataFrame access).
    if need["trials"]:
        try:
            trials_df = extract_trials(nwb_path)
            if trials_df is not None:
                trials_df.to_parquet(session_dir / "trials.parquet", index=True)
        except Exception as e:
            _log(f"  WARN trials read failed: {type(e).__name__}: {e}")

    # Everything else via h5py. Open once if any of those outputs need work.
    h5_keys = ("subject", "session_meta", "dff", "behavior", "keypoints", "roi", "widefield")
    if not any(need[k] for k in h5_keys):
        return "done (trials only)"

    with h5py.File(nwb_path, "r") as h5:
        if need["subject"]:
            write_json(mouse_dir / "subject.json", extract_subject(h5))

        if need["session_meta"]:
            write_json(session_dir / "session_meta.json",
                       extract_session_meta(h5, sid, session_id))

        if need["dff"]:
            dff = extract_dff(h5)
            if dff:
                np.savez_compressed(session_dir / "dff.npz", **dff)

        if need["behavior"]:
            beh = extract_behavior(h5)
            if beh:
                np.savez_compressed(session_dir / "behavior.npz", **beh)

        if need["keypoints"]:
            kp = extract_keypoints(h5)
            if kp:
                np.savez_compressed(session_dir / "keypoints.npz", **kp)

        # ROI + widefield: get masks if either output is missing.
        masks: np.ndarray | None = None
        if need["roi"] or need["widefield"]:
            roi_meta, roi_arrays = extract_roi_metadata_and_masks(h5)
            if need["roi"]:
                if roi_meta:
                    write_json(session_dir / "roi_metadata.json", roi_meta)
                if roi_arrays:
                    np.savez_compressed(session_dir / "roi_masks.npz", **roi_arrays)
            if roi_arrays:
                masks = roi_arrays.get("masks")

        if need["widefield"]:
            # If ROI was already on disk (we didn't extract just now), load masks from there.
            if masks is None and (session_dir / "roi_masks.npz").exists():
                with np.load(session_dir / "roi_masks.npz") as arrs:
                    if "masks" in arrs.files:
                        masks = arrs["masks"]
            if masks is not None:
                wf = compute_widefield_F_per_roi(h5, masks)
                if wf:
                    np.savez_compressed(session_dir / "widefield_F_per_roi.npz", **wf)
            else:
                _log("  WARN no ROI masks available; skipping widefield_F_per_roi")

    return "done"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--nwb-root", required=True, type=Path,
                    help="root directory containing sub-VG1-GC#<id>/ subdirs")
    ap.add_argument("--out", required=True, type=Path,
                    help="output directory")
    ap.add_argument("--mice", type=str, default="",
                    help="comma-separated mouse IDs to restrict to (default: all)")
    ap.add_argument("--session-types", type=str, default="task,resting-state,sensory-stim",
                    help="comma-separated session types to include (default: all 3)")
    ap.add_argument("--limit", type=int, default=0,
                    help="if > 0, process only first N matched files (for testing)")
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="reprocess sessions even if outputs already exist")
    args = ap.parse_args()

    mice_filter = {m.strip() for m in args.mice.split(",") if m.strip()}
    session_types = {t.strip() for t in args.session_types.split(",") if t.strip()}

    _log(f"Scanning {args.nwb_root}...")
    candidates = sorted(args.nwb_root.rglob("*.nwb"))

    todo: list[Path] = []
    for p in candidates:
        try:
            sid = parse_session_filename(p)
        except ValueError:
            continue
        if sid.task not in session_types:
            continue
        if mice_filter and sid.mouse not in mice_filter:
            continue
        todo.append(p)
    if args.limit > 0:
        todo = todo[: args.limit]

    _log(f"To process: {len(todo)} NWB files")
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "meta_global.json", {
        "extracted_at": time.strftime("%F %T"),
        "nwb_root": str(args.nwb_root),
        "session_types": sorted(session_types),
        "mice_filter": sorted(mice_filter) if mice_filter else None,
        "n_files": len(todo),
        "script": "extract_session_data.py",
    })

    n_done = n_skip = n_err = 0
    t_global = time.time()
    for i, nwb_path in enumerate(todo, 1):
        _log(f"[{i}/{len(todo)}] {nwb_path.name}")
        t0 = time.time()
        try:
            status = process_session(
                nwb_path, args.out, skip_existing=not args.no_skip_existing
            )
            dt = time.time() - t0
            _log(f"  {status} ({dt:.1f}s)")
            if "skip" in status:
                n_skip += 1
            else:
                n_done += 1
        except Exception as e:
            n_err += 1
            _log(f"  ERROR: {type(e).__name__}: {e}")
        finally:
            gc.collect()

    elapsed = time.time() - t_global
    _log("=" * 60)
    _log(f"Done: {n_done}  Skipped: {n_skip}  Errors: {n_err}")
    _log(f"Total time: {elapsed / 60:.1f} min")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
