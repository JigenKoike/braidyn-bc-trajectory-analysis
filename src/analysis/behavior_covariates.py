"""Body motion and pupil size as one number per session, for the mixed models.

The models' predictors are one value per session, shared by the early- and
late-window models alike, and these two are built to match: each summarizes
the frames of every trial's window, cue −1 s to +4 s — the window the neural
PETHs are cut on — so the behaviour and arousal of the trial period are in
it, and the value does not depend on which distance window is being
modelled.

**Motion** is the output of a load cell under the body chamber, low-pass
filtered at 10 Hz with its time average subtracted (Kondo et al. 2025,
*Scientific Data*). Being zero-mean, its average says nothing; its standard
deviation over the trial windows is the amplitude of the force the body puts
on the chamber, which is how much the animal moved.

**Pupil** diameter is the major axis of an ellipse fitted to thirty
DeepLabCut points on the pupil edge, frame by frame (same source). When the
fit fails the axis can come out absurd — up to 10³⁴ pixels in this data —
or not at all, so the series is cleaned first: anything non-finite or
outside a band around the session's median is treated as missing. The
session's value is the mean of what remains in the trial windows, and a
session missing too much of its trial-window pupil gets no value at all
rather than one built from a remnant.

**Neither is comparable across mice as it stands.** The load cell was never
calibrated, so how many volts a given movement makes depends on the rig and
on the animal's weight; the pupil is measured in camera pixels, so its size
depends on the magnification. Both act as a factor that is fixed for a mouse
and different between mice. :func:`normalize_within_mouse` divides each
session's value by the mean over that mouse's sessions, which cancels any
such factor exactly and leaves how this day compared with the mouse's usual.
What is given up is the difference between mice — which, uncalibrated, could
not be told apart from the instruments anyway.

A ratio, not its log. The log would cancel the factor just as well, and is
often taken so that twice and half the usual sit equally far from it; on
this data it skews the values instead (low-motion sessions stretch out to the
left), and the plain ratio reads directly as "this much above the usual".
The change was made on 2026-10-19, at the user's request.

The values are stored raw and normalized when read, so the normalization can
change without the session data being extracted again.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .cue_align import cue_align
from .global_pca import DEFAULT_POST_FRAMES, DEFAULT_PRE_FRAMES

#: Pupil values outside this band, as multiples of the session's median, are
#: taken for failed ellipse fits. Real dilation and constriction stay well
#: inside it: over a trial the pupil moves by about a fifth.
PUPIL_BAND = (0.5, 2.0)

#: A session missing more than this share of its trial-window pupil frames
#: gets no pupil value.
MAX_PUPIL_MISSING = 0.20

#: The two predictors this module adds to the seven, under the names the
#: models use for them once normalized.
BEHAVIOR_PREDICTORS: tuple[str, ...] = ("motion", "pupil")


def classify_pupil_frames(
    pupil: np.ndarray, band: tuple[float, float] = PUPIL_BAND
) -> dict[str, np.ndarray]:
    """Which frames of a session's pupil failed, and how.

    Returns boolean masks ``missing`` (not finite), ``too_large`` and
    ``too_small`` (outside ``band`` times the median of the finite frames),
    and ``failed``, their union. The median is used because it is not moved
    by the failures themselves, however large they are.

    This is the one definition of a failed frame: :func:`clean_pupil` applies
    it, and the quality checks count with it, so the two cannot disagree.
    """
    p = np.asarray(pupil, dtype=np.float64)
    missing = ~np.isfinite(p)
    if missing.all():
        none = np.zeros_like(missing)
        return {"missing": missing, "too_large": none, "too_small": none, "failed": missing}
    median = float(np.median(p[~missing]))
    assert median > 0, f"the session's median pupil is {median}, not a size"
    lo, hi = band
    with np.errstate(invalid="ignore"):
        too_large = ~missing & (p > hi * median)
        too_small = ~missing & (p < lo * median)
    return {"missing": missing, "too_large": too_large, "too_small": too_small,
            "failed": missing | too_large | too_small}


def clean_pupil(pupil: np.ndarray, band: tuple[float, float] = PUPIL_BAND) -> np.ndarray:
    """A float copy of ``pupil`` with failed frames set to NaN.

    Failed as :func:`classify_pupil_frames` defines it.
    """
    p = np.array(pupil, dtype=np.float64)
    p[classify_pupil_frames(p, band)["failed"]] = np.nan
    return p


def _trial_frames(
    signal: np.ndarray,
    timestamps: np.ndarray,
    cue_times: np.ndarray,
    pre_frames: int,
    post_frames: int,
) -> tuple[np.ndarray, int]:
    """Every frame of every trial's window, and how many trials fitted."""
    slices, valid = cue_align(signal, timestamps, cue_times,
                              pre_frames=pre_frames, post_frames=post_frames)
    return slices.reshape(-1), len(valid)


def pupil_session_value(
    pupil: np.ndarray,
    timestamps: np.ndarray,
    cue_times: np.ndarray,
    *,
    band: tuple[float, float] = PUPIL_BAND,
    max_missing: float = MAX_PUPIL_MISSING,
    pre_frames: int = DEFAULT_PRE_FRAMES,
    post_frames: int = DEFAULT_POST_FRAMES,
) -> dict[str, Any]:
    """The session's pupil size: the mean over every trial window, cleaned.

    Returns ``pupil_px`` (NaN if more than ``max_missing`` of the window
    frames failed), ``pupil_missing_fraction`` over those frames, and
    ``pupil_n_trials``. Frames outside the trial windows count for nothing,
    either way.
    """
    frames, n_trials = _trial_frames(clean_pupil(pupil, band), timestamps, cue_times,
                                     pre_frames, post_frames)
    if frames.size == 0:
        return {"pupil_px": np.nan, "pupil_missing_fraction": np.nan, "pupil_n_trials": 0}
    missing = float(np.isnan(frames).mean())
    value = float(np.nanmean(frames)) if missing <= max_missing else np.nan
    return {"pupil_px": value, "pupil_missing_fraction": missing, "pupil_n_trials": n_trials}


def motion_session_value(
    motion: np.ndarray,
    timestamps: np.ndarray,
    cue_times: np.ndarray,
    *,
    pre_frames: int = DEFAULT_PRE_FRAMES,
    post_frames: int = DEFAULT_POST_FRAMES,
) -> dict[str, Any]:
    """The session's motion: the standard deviation over every trial window.

    Returns ``motion_sd`` and ``motion_n_trials``. The load-cell signal has
    no missing frames in this data; should one turn up, it is left out of
    the SD rather than allowed to make it NaN.
    """
    frames, n_trials = _trial_frames(np.asarray(motion, dtype=np.float64), timestamps,
                                     cue_times, pre_frames, post_frames)
    if frames.size == 0 or np.isnan(frames).all():
        return {"motion_sd": np.nan, "motion_n_trials": n_trials}
    return {"motion_sd": float(np.nanstd(frames)), "motion_n_trials": n_trials}


def session_covariates(
    timestamps: np.ndarray,
    cue_times: np.ndarray,
    motion: np.ndarray,
    pupil: np.ndarray | None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Both values for one session. ``pupil`` is None where none was recorded."""
    out = motion_session_value(motion, timestamps, cue_times,
                               **{k: v for k, v in kwargs.items() if k.endswith("_frames")})
    if pupil is None:
        out.update({"pupil_px": np.nan, "pupil_missing_fraction": np.nan, "pupil_n_trials": 0})
    else:
        out.update(pupil_session_value(pupil, timestamps, cue_times, **kwargs))
    return out


def normalize_within_mouse(values: pd.Series, mouse: pd.Series) -> pd.Series:
    """The value over the mean of that mouse's sessions, minus one.

    Any factor fixed within a mouse — a camera's magnification, a load
    cell's sensitivity — multiplies the value and its mouse's mean alike,
    and cancels. What remains is how this session compared with the mouse's
    usual: 0 is the usual, +0.1 is 10% above it, −0.1 is 10% below.

    Pass **every** session of a mouse, not the subset one model happens to
    use, so that the same session gets the same value in every model. Missing
    values stay missing and do not enter the mean; a mouse with a single
    session gets 0.
    """
    values = pd.Series(values, dtype=float)
    present = values.dropna()
    assert (present > 0).all(), "a size or an amplitude has to be positive to be taken as a ratio"
    usual = values.groupby(pd.Series(mouse, index=values.index)).transform("mean")
    return values / usual - 1


def add_behavior_covariates(sessions: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    """``sessions`` with ``motion`` and ``pupil`` added, each normalized within mouse.

    ``sessions`` is one row per task session with ``session_id`` and
    ``mouse`` — what :func:`analysis.predictors.session_predictors` returns.
    ``covariates`` is the raw table ``scripts/extract_behavior_covariates.py``
    writes. Every session has to be in it: motion exists for all of them, so a
    session that is missing was not extracted, which is an error rather than
    a gap. A session without a pupil value keeps NaN.

    The mouse means come from the sessions passed, so pass every task
    session, as :func:`normalize_within_mouse` asks.
    """
    clash = set(BEHAVIOR_PREDICTORS) & set(sessions.columns)
    assert not clash, f"sessions already has {sorted(clash)}"
    out = sessions.merge(covariates[["session_id", "motion_sd", "pupil_px"]],
                         on="session_id", how="left", validate="1:1")
    absent = out.loc[out["motion_sd"].isna(), "session_id"].tolist()
    assert not absent, f"sessions missing from the covariate table: {absent[:3]}"
    out["motion"] = normalize_within_mouse(out["motion_sd"], out["mouse"])
    out["pupil"] = normalize_within_mouse(out["pupil_px"], out["mouse"])
    return out
