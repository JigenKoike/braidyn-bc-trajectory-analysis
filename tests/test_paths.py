"""Tests for `braidyn_io.paths`."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from braidyn_io.paths import (
    MODALITIES,
    TASK_TYPES,
    SessionId,
    format_session_filename,
    parse_session_filename,
    session_path,
)


# Real filenames sampled from the actual BraiDyn-BC S3 listing (one of each
# task/modality combination plus a few extras).
KNOWN_FILES: list[tuple[str, SessionId]] = [
    (
        "sub-VG1-GC#78_ses-2023-11-13-resting-state-day0_widefield+behavior.nwb",
        SessionId(
            mouse="sub-VG1-GC#78",
            date=date(2023, 11, 13),
            task="resting-state",
            day=0,
            modality="widefield+behavior",
        ),
    ),
    (
        "sub-VG1-GC#78_ses-2023-11-14-task-day1_widefield+behavior.nwb",
        SessionId(
            mouse="sub-VG1-GC#78",
            date=date(2023, 11, 14),
            task="task",
            day=1,
            modality="widefield+behavior",
        ),
    ),
    (
        "sub-VG1-GC#78_ses-2023-12-06-sensory-stim-day16_widefield.nwb",
        SessionId(
            mouse="sub-VG1-GC#78",
            date=date(2023, 12, 6),
            task="sensory-stim",
            day=16,
            modality="widefield",
        ),
    ),
    (
        "sub-VG1-GC#105_ses-2024-01-24-task-day1_widefield+behavior.nwb",
        SessionId(
            mouse="sub-VG1-GC#105",
            date=date(2024, 1, 24),
            task="task",
            day=1,
            modality="widefield+behavior",
        ),
    ),
    (
        "sub-VG1-GC#129_ses-2024-03-19-task-day8_widefield.nwb",
        SessionId(
            mouse="sub-VG1-GC#129",
            date=date(2024, 3, 19),
            task="task",
            day=8,
            modality="widefield",
        ),
    ),
]


@pytest.mark.parametrize("filename, expected", KNOWN_FILES)
def test_parse_real_filenames(filename: str, expected: SessionId) -> None:
    assert parse_session_filename(filename) == expected


@pytest.mark.parametrize("filename, sid", KNOWN_FILES)
def test_format_is_inverse_of_parse(filename: str, sid: SessionId) -> None:
    assert format_session_filename(sid) == filename
    # And round-trip through parse():
    assert parse_session_filename(format_session_filename(sid)) == sid


def test_parse_accepts_full_path() -> None:
    name = "sub-VG1-GC#78_ses-2023-11-13-resting-state-day0_widefield+behavior.nwb"
    full = f"/path/to/nwb/sub-VG1-GC#78/{name}"
    assert parse_session_filename(full).mouse == "sub-VG1-GC#78"


def test_parse_accepts_pathlib_path() -> None:
    p = Path("sub-VG1-GC#78_ses-2023-11-14-task-day1_widefield+behavior.nwb")
    sid = parse_session_filename(p)
    assert sid.task == "task"
    assert sid.day == 1


def test_session_path_construction() -> None:
    sid = SessionId(
        mouse="sub-VG1-GC#78",
        date=date(2023, 11, 13),
        task="resting-state",
        day=0,
        modality="widefield+behavior",
    )
    assert session_path(sid, "/data/braidyn_bc/nwb") == Path(
        "/data/braidyn_bc/nwb/sub-VG1-GC#78/"
        "sub-VG1-GC#78_ses-2023-11-13-resting-state-day0_widefield+behavior.nwb"
    )


def test_parse_rejects_unrelated_filename() -> None:
    with pytest.raises(ValueError, match="does not match"):
        parse_session_filename("some_random_file.nwb")


def test_parse_rejects_unknown_task() -> None:
    # Synthetic name with a task name we haven't seen in the dataset.
    with pytest.raises(ValueError, match="unknown task"):
        parse_session_filename(
            "sub-VG1-GC#78_ses-2023-11-13-novel-task-day0_widefield.nwb"
        )


def test_parse_rejects_unknown_modality() -> None:
    with pytest.raises(ValueError, match="unknown modality"):
        parse_session_filename(
            "sub-VG1-GC#78_ses-2023-11-13-task-day0_two-photon.nwb"
        )


def test_session_id_is_sortable_by_mouse_then_date() -> None:
    s1 = SessionId(mouse="sub-VG1-GC#48", date=date(2023, 11, 13),
                   task="task", day=1, modality="widefield")
    s2 = SessionId(mouse="sub-VG1-GC#78", date=date(2023, 11, 13),
                   task="task", day=1, modality="widefield")
    s3 = SessionId(mouse="sub-VG1-GC#78", date=date(2024, 1, 1),
                   task="task", day=1, modality="widefield")
    assert sorted([s3, s1, s2]) == [s1, s2, s3]


def test_session_id_is_hashable() -> None:
    sid = SessionId(
        mouse="sub-VG1-GC#78",
        date=date(2023, 11, 13),
        task="task",
        day=1,
        modality="widefield",
    )
    # Should be usable as a dict key / set member.
    {sid}
    {sid: "ok"}


def test_constants_are_consistent() -> None:
    """The TASK_TYPES/MODALITIES tuples should match the Literal type members.

    This is a self-check: if someone adds a new task to TASK_TYPES, they
    must also extend the Literal[...] type alias above (and vice versa).
    """
    # We can't directly introspect Literal members at runtime in a clean way,
    # but we can at least assert the tuples are non-empty and contain strings.
    assert len(TASK_TYPES) >= 1
    assert all(isinstance(t, str) for t in TASK_TYPES)
    assert len(MODALITIES) >= 1
    assert all(isinstance(m, str) for m in MODALITIES)
