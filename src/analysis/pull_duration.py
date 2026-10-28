"""Trial-level pull-duration features from the ``state_lever`` signal.

NWB-derived ``trials.parquet`` stores ``pull_duration_for_success`` — the
threshold the task imposed (Tpull), NOT the time the animal actually held
the lever. To get the actual hold time we threshold the binary 30 Hz
``state_lever`` signal in ``behavior.npz`` and pair down/up transitions
within each trial's (cue, stop] window.

The signal is quantized at ~33 ms (30 Hz), so very short presses are
under-resolved. About 0.17% of success trials end up with a measured
first-press duration shorter than Tpull due to this quantization — a
tolerable error rate.

Two scalar features per trial are returned:
  - ``pull_dur_first`` — duration of the first 0→1 press after cue (P1)
  - ``pull_dur_max``   — duration of the longest press in (cue, stop] (P3)

Plus diagnostic columns:
  - ``already_down``   — bool, lever was already pressed at the cue frame
  - ``n_presses``      — int, number of 0→1 transitions in (cue, stop]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_pull_durations(
    state_lever: np.ndarray,
    timestamps: np.ndarray,
    trials: pd.DataFrame,
) -> pd.DataFrame:
    """Per-trial pull-duration features from the binary lever-state signal.

    Parameters
    ----------
    state_lever : array, shape (n_frames,)
        Binary lever state at 30 Hz: 1 = pressed, 0 = released.
    timestamps : array, shape (n_frames,)
        Frame times in seconds, monotonically increasing.
    trials : DataFrame
        Must contain ``start_time`` (cue) and ``stop_time`` columns.

    Returns
    -------
    DataFrame indexed like ``trials`` with columns:
        ``already_down``, ``n_presses``,
        ``pull_dur_first``, ``pull_dur_max``.

    Notes
    -----
    Presses are restricted to the ``(start_time, stop_time]`` window —
    presses already in progress at cue (already_down=True) are NOT
    counted toward ``pull_dur_first`` or ``pull_dur_max``. This avoids
    contaminating the trial's pull features with prior-trial activity.
    A press that begins inside the window is paired with the next
    release whether or not that release falls inside the window; this
    captures hold times that continue past ``stop_time``.
    """
    sl = np.asarray(state_lever).astype(np.int8)
    ts = np.asarray(timestamps, dtype=np.float64)
    if sl.shape != ts.shape:
        raise ValueError(
            f"state_lever shape {sl.shape} != timestamps shape {ts.shape}"
        )

    d = np.diff(sl)
    down_t = ts[np.where(d == 1)[0] + 1]
    up_t = ts[np.where(d == -1)[0] + 1]

    # If the recording ends with the lever pressed, there's a trailing
    # down with no matching up; fall back to the last frame so every
    # press has a release.
    if len(down_t) > len(up_t):
        up_t = np.append(up_t, ts[-1])

    starts = trials["start_time"].to_numpy(dtype=np.float64)
    stops = trials["stop_time"].to_numpy(dtype=np.float64)

    already_down = np.empty(len(trials), dtype=bool)
    n_presses = np.zeros(len(trials), dtype=np.int32)
    pull_first = np.full(len(trials), np.nan, dtype=np.float64)
    pull_max = np.full(len(trials), np.nan, dtype=np.float64)

    for i, (cue, stop) in enumerate(zip(starts, stops)):
        cue_idx = max(0, int(np.searchsorted(ts, cue, side="right")) - 1)
        already_down[i] = bool(sl[cue_idx] == 1)

        in_win = (down_t > cue) & (down_t <= stop)
        presses = down_t[in_win]
        n_presses[i] = len(presses)
        if len(presses) == 0:
            continue

        # Each press paired with the next release after it
        durs = np.empty(len(presses), dtype=np.float64)
        for k, pt in enumerate(presses):
            rel = up_t[up_t > pt]
            durs[k] = (rel[0] - pt) if len(rel) else np.nan
        pull_first[i] = durs[0]
        pull_max[i] = np.nanmax(durs) if not np.all(np.isnan(durs)) else np.nan

    return pd.DataFrame(
        {
            "already_down": already_down,
            "n_presses": n_presses,
            "pull_dur_first": pull_first,
            "pull_dur_max": pull_max,
        },
        index=trials.index,
    )
