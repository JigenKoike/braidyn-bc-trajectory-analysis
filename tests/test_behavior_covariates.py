"""Tests for the session-level motion and pupil values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.behavior_covariates import (
    MAX_PUPIL_MISSING,
    add_behavior_covariates,
    classify_pupil_frames,
    clean_pupil,
    motion_session_value,
    normalize_within_mouse,
    pupil_session_value,
    session_covariates,
)
from analysis.global_pca import DEFAULT_POST_FRAMES, DEFAULT_PRE_FRAMES

FPS = 30.0
WINDOW = DEFAULT_PRE_FRAMES + DEFAULT_POST_FRAMES + 1


def _session(n_frames: int = 3000, cues_s=(10.0, 30.0, 50.0, 70.0)):
    """A time axis and cue times whose windows all fit inside it."""
    t = np.arange(n_frames) / FPS
    return t, np.array(cues_s)


def _window_mask(t: np.ndarray, cues: np.ndarray) -> np.ndarray:
    """Which frames fall inside some trial's window."""
    mask = np.zeros(len(t), dtype=bool)
    for c in cues:
        i = int(np.searchsorted(t, c, side="left"))
        mask[i - DEFAULT_PRE_FRAMES:i + DEFAULT_POST_FRAMES + 1] = True
    return mask


# --- cleaning ---------------------------------------------------------------

def test_failed_fits_become_missing_and_the_rest_is_untouched() -> None:
    p = np.full(100, 50.0)
    p[[3, 4]] = [np.inf, np.nan]
    p[10] = 4e34          # the kind of value a failed ellipse fit gives
    p[11] = 101.0         # just over twice the median
    p[12] = 24.0          # just under half
    p[13] = 99.0          # inside the band
    out = clean_pupil(p)
    assert np.isnan(out[[3, 4, 10, 11, 12]]).all()
    assert out[13] == 99.0
    assert (out[20:] == 50.0).all()


def test_cleaning_does_not_touch_its_input() -> None:
    p = np.array([50.0, np.inf, 1e30, 50.0])
    clean_pupil(p)
    assert np.isinf(p[1]) and p[2] == 1e30


def test_the_band_is_set_by_the_median_so_the_failures_cannot_move_it() -> None:
    # A third of the frames absurdly large: a mean would be dragged off by
    # orders of magnitude, the median is not, so the real frames survive.
    p = np.r_[np.full(200, 60.0), np.full(100, 1e20)]
    out = clean_pupil(p)
    assert np.isnan(out[200:]).all()
    assert (out[:200] == 60.0).all()


def test_a_session_with_no_finite_pupil_comes_back_all_missing() -> None:
    assert np.isnan(clean_pupil(np.full(10, np.nan))).all()


def test_each_failed_frame_is_classified_once_and_the_union_is_what_cleaning_removes() -> None:
    p = np.full(50, 60.0)
    p[[1, 2]] = [np.nan, np.inf]
    p[5] = 1e30
    p[8] = 20.0
    flags = classify_pupil_frames(p)
    assert list(np.flatnonzero(flags["missing"])) == [1, 2]
    assert list(np.flatnonzero(flags["too_large"])) == [5]
    assert list(np.flatnonzero(flags["too_small"])) == [8]
    kinds = flags["missing"].astype(int) + flags["too_large"] + flags["too_small"]
    assert kinds.max() == 1, "a frame was counted as two kinds of failure"
    np.testing.assert_array_equal(flags["failed"], np.isnan(clean_pupil(p)))


# --- the trial windows ------------------------------------------------------

def test_the_window_is_the_neural_one() -> None:
    # cue -1 s to +4 s, the frame of the cue included: 151 frames at 30 Hz.
    assert (DEFAULT_PRE_FRAMES, DEFAULT_POST_FRAMES, WINDOW) == (30, 120, 151)


def test_the_pupil_value_is_the_mean_of_the_trial_windows_only() -> None:
    t, cues = _session()
    inside = _window_mask(t, cues)
    p = np.where(inside, 60.0, 50.0)   # outside the windows does not count
    out = pupil_session_value(p, t, cues)
    assert out["pupil_px"] == pytest.approx(60.0)
    assert out["pupil_n_trials"] == len(cues)
    assert out["pupil_missing_fraction"] == 0.0


def test_missing_frames_outside_the_windows_do_not_count_against_a_session() -> None:
    t, cues = _session()
    p = np.full(len(t), 60.0)
    p[~_window_mask(t, cues)] = np.nan
    out = pupil_session_value(p, t, cues)
    assert out["pupil_missing_fraction"] == 0.0
    assert out["pupil_px"] == pytest.approx(60.0)


def test_a_session_missing_too_much_gets_no_value_but_says_how_much() -> None:
    t, cues = _session()
    inside = np.flatnonzero(_window_mask(t, cues))
    p = np.full(len(t), 60.0)
    p[inside[: int(len(inside) * (MAX_PUPIL_MISSING + 0.05))]] = np.nan
    out = pupil_session_value(p, t, cues)
    assert np.isnan(out["pupil_px"])
    assert out["pupil_missing_fraction"] > MAX_PUPIL_MISSING
    # Just under the limit, the value stands.
    p = np.full(len(t), 60.0)
    p[inside[: int(len(inside) * (MAX_PUPIL_MISSING - 0.05))]] = np.nan
    assert pupil_session_value(p, t, cues)["pupil_px"] == pytest.approx(60.0)


def test_motion_is_the_amplitude_over_the_trial_windows() -> None:
    t, cues = _session()
    inside = _window_mask(t, cues)
    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.03, len(t))
    m[~inside] = rng.normal(0, 0.5, (~inside).sum())   # big movement between trials
    out = motion_session_value(m, t, cues)
    assert out["motion_sd"] == pytest.approx(0.03, rel=0.05)
    assert out["motion_n_trials"] == len(cues)


def test_a_trial_whose_window_runs_off_the_recording_is_left_out() -> None:
    t, cues = _session(n_frames=1000)          # 33 s of recording
    cues = np.r_[cues[:1], 32.0]               # the second window needs 4 s more
    assert motion_session_value(np.zeros(len(t)) + 1e-3 * np.arange(len(t)), t, cues)[
        "motion_n_trials"] == 1


def test_a_session_without_a_pupil_recording_still_has_its_motion() -> None:
    t, cues = _session()
    rng = np.random.default_rng(1)
    out = session_covariates(t, cues, rng.normal(0, 0.02, len(t)), None)
    assert out["motion_sd"] > 0
    assert np.isnan(out["pupil_px"]) and out["pupil_n_trials"] == 0


# --- normalization within a mouse ---------------------------------------------

def _two_mice():
    values = pd.Series([50.0, 60.0, 55.0, 100.0, 120.0, 110.0])
    mouse = pd.Series(["a", "a", "a", "b", "b", "b"])
    return values, mouse


def test_a_factor_fixed_within_a_mouse_cancels_exactly() -> None:
    # The point of the normalization: a camera that magnifies mouse b's eye
    # 1.7 times, or a load cell that reads it 1.7 times as strongly, leaves
    # its normalized values exactly where they were.
    values, mouse = _two_mice()
    before = normalize_within_mouse(values, mouse)
    scaled = values.copy()
    scaled[mouse == "b"] *= 1.7
    after = normalize_within_mouse(scaled, mouse)
    pd.testing.assert_series_equal(before, after)


def test_each_mouse_is_centred_on_its_usual() -> None:
    values, mouse = _two_mice()
    out = normalize_within_mouse(values, mouse)
    assert out.groupby(mouse).mean().abs().max() < 1e-12
    # b's sessions are a's times two, so the two mice come out identical.
    np.testing.assert_allclose(out[mouse == "a"].to_numpy(), out[mouse == "b"].to_numpy())


def test_a_session_10_percent_above_the_usual_reads_plus_0_1() -> None:
    # A plain ratio to the mouse's mean, minus one: no log anywhere.
    out = normalize_within_mouse(pd.Series([90.0, 110.0]), pd.Series(["a"] * 2))
    np.testing.assert_allclose(out.to_numpy(), [-0.1, 0.1], rtol=1e-12)
    # How two sessions of one mouse compare survives intact: 110 is 110/90 of 90.
    assert (out.iloc[1] + 1) / (out.iloc[0] + 1) == pytest.approx(110 / 90, rel=1e-12)


def test_missing_values_stay_missing_and_do_not_enter_the_mean() -> None:
    values = pd.Series([50.0, np.nan, 200.0, 80.0])
    mouse = pd.Series(["a", "a", "a", "b"])
    out = normalize_within_mouse(values, mouse)
    assert np.isnan(out.iloc[1])
    assert out.iloc[0] == pytest.approx(50 / np.mean([50, 200]) - 1)
    assert out.iloc[3] == 0.0   # a mouse with one session is its own usual


def test_a_value_that_is_not_positive_is_refused() -> None:
    with pytest.raises(AssertionError, match="positive"):
        normalize_within_mouse(pd.Series([1.0, 0.0]), pd.Series(["a", "a"]))


# --- joining them to the session predictors -----------------------------------

def _sessions_and_covariates():
    sessions = pd.DataFrame({"session_id": ["s1", "s2", "s3", "s4"],
                             "mouse": ["a", "a", "b", "b"], "day": [1, 2, 1, 2]})
    # Not in the sessions' order, and with a session no model uses.
    covariates = pd.DataFrame({"session_id": ["s4", "s2", "s1", "s3", "s9"],
                               "motion_sd": [0.4, 0.2, 0.1, 0.2, 9.0],
                               "pupil_px": [80.0, np.nan, 50.0, 60.0, 1.0]})
    return sessions, covariates


def test_the_covariates_join_their_sessions_normalized_within_mouse() -> None:
    sessions, covariates = _sessions_and_covariates()
    out = add_behavior_covariates(sessions, covariates)
    assert list(out["session_id"]) == list(sessions["session_id"])
    raw = covariates.set_index("session_id").loc[out["session_id"]].reset_index()
    pd.testing.assert_series_equal(out["motion"], normalize_within_mouse(raw["motion_sd"], out["mouse"]),
                                   check_names=False)
    pd.testing.assert_series_equal(out["pupil"], normalize_within_mouse(raw["pupil_px"], out["mouse"]),
                                   check_names=False)
    # s2 has no pupil value: it stays missing, and mouse a's usual is s1 alone.
    assert np.isnan(out.loc[1, "pupil"]) and out.loc[0, "pupil"] == 0.0


def test_a_session_absent_from_the_covariates_is_an_error() -> None:
    sessions, covariates = _sessions_and_covariates()
    with pytest.raises(AssertionError, match="missing from the covariate table"):
        add_behavior_covariates(sessions, covariates[covariates["session_id"] != "s3"])
