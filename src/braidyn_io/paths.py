"""Path and filename conventions for BraiDyn-BC NWB sessions.

A session NWB filename encodes mouse, date, task, day, and recording modality::

    sub-VG1-GC#78_ses-2023-11-13-resting-state-day0_widefield+behavior.nwb
    └─── mouse ──┘    └── date ──┘└──── task ───┘└day┘└──── modality ───┘

`parse_session_filename` turns such a name into a `SessionId` dataclass.
`format_session_filename` is its inverse. `session_path` gives the canonical
on-disk location under a data root (one subdirectory per mouse).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal


TASK_TYPES: tuple[str, ...] = ("resting-state", "task", "sensory-stim")
MODALITIES: tuple[str, ...] = ("widefield+behavior", "widefield")

TaskType = Literal["resting-state", "task", "sensory-stim"]
Modality = Literal["widefield+behavior", "widefield"]


@dataclass(frozen=True, order=True)
class SessionId:
    """Structured identifier for one BraiDyn-BC NWB session.

    Frozen + ordered, so instances are hashable (usable as dict keys / in sets)
    and sortable (by mouse, then date, then task, then day, then modality).
    """

    mouse: str           # e.g. "sub-VG1-GC#78"
    date: date           # session calendar date
    task: TaskType       # one of TASK_TYPES
    day: int             # day index from the start of the experiment (0, 1, ...)
    modality: Modality   # one of MODALITIES


# `-day` and `_<modality>` act as right-anchors so the non-greedy `.+?` for
# task captures the full task name even when it contains hyphens
# (e.g. "resting-state", "sensory-stim").
_FILENAME_PATTERN = re.compile(
    r"^(?P<mouse>sub-VG1-GC#\d+)"
    r"_ses-(?P<date>\d{4}-\d{2}-\d{2})"
    r"-(?P<task>.+?)"
    r"-day(?P<day>\d+)"
    r"_(?P<modality>[^.]+)"
    r"\.nwb$"
)


def parse_session_filename(filename: str | Path) -> SessionId:
    """Parse a session NWB filename into a `SessionId`.

    Accepts either a bare filename or a path — only the basename is parsed.
    Raises `ValueError` on unrecognized format, unknown task, or unknown modality.
    """
    name = Path(filename).name
    m = _FILENAME_PATTERN.match(name)
    if m is None:
        raise ValueError(f"filename does not match BraiDyn-BC session pattern: {name!r}")

    task = m["task"]
    if task not in TASK_TYPES:
        raise ValueError(
            f"unknown task {task!r} in {name!r}; known tasks: {TASK_TYPES}"
        )
    modality = m["modality"]
    if modality not in MODALITIES:
        raise ValueError(
            f"unknown modality {modality!r} in {name!r}; known modalities: {MODALITIES}"
        )

    y, mo, d = (int(x) for x in m["date"].split("-"))
    return SessionId(
        mouse=m["mouse"],
        date=date(y, mo, d),
        task=task,  # type: ignore[arg-type]  # validated above
        day=int(m["day"]),
        modality=modality,  # type: ignore[arg-type]  # validated above
    )


def format_session_filename(sid: SessionId) -> str:
    """Round-trip inverse of `parse_session_filename`."""
    return (
        f"{sid.mouse}"
        f"_ses-{sid.date:%Y-%m-%d}"
        f"-{sid.task}"
        f"-day{sid.day}"
        f"_{sid.modality}"
        f".nwb"
    )


def session_path(sid: SessionId, root: str | Path) -> Path:
    """Canonical on-disk path: ``<root>/<mouse>/<filename>.nwb``.

    This matches both the S3 key layout and the layout written by
    ``scripts/sync_s3_to_server.py``.
    """
    return Path(root) / sid.mouse / format_session_filename(sid)
