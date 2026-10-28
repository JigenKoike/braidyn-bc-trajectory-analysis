"""Check a local copy of the NWB files against the bucket it came from.

Typical use is a pre-flight before heavy analysis — confirm that every
expected session file is present locally with the correct byte size before
spending hours on a notebook that loads them.

The S3 ETag would normally be useful for content checksums, but BraiDyn-BC
files are uploaded multipart so the ETag is not a plain MD5 of the file
contents — size match is the strongest cheap check we have. (TCP guarantees
byte fidelity, and sequential writes preserve order, so for our streaming
downloads "size match" is effectively "content match".)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .s3_fetch import list_files


@dataclass(frozen=True)
class FileListDiff:
    """What a local copy has, and what the bucket has, set side by side.

    Keys throughout are S3-style paths like ``"sub-VG1-GC#78/<file>.nwb"`` —
    relative to whatever root the local/s3 listings used.
    """

    matching: list[str] = field(default_factory=list)
    """Keys present in both with identical byte size."""

    missing_local: list[str] = field(default_factory=list)
    """Keys in S3 but absent locally — need downloading."""

    extra_local: list[str] = field(default_factory=list)
    """Keys present locally but absent in S3 — unexpected files."""

    size_mismatch: list[tuple[str, int, int]] = field(default_factory=list)
    """Keys present in both with differing sizes: ``(key, local_size, s3_size)``."""

    @property
    def is_clean(self) -> bool:
        """True iff every expected file is present locally with the correct size."""
        return not (self.missing_local or self.extra_local or self.size_mismatch)

    def summary(self) -> str:
        """One-line human summary, e.g. ``"495 OK | 0 missing | 0 mismatched | 0 extra"``."""
        return (
            f"{len(self.matching)} OK | "
            f"{len(self.missing_local)} missing | "
            f"{len(self.size_mismatch)} mismatched | "
            f"{len(self.extra_local)} extra"
        )


def scan_local_files(directory: Path | str, ext: str = ".nwb") -> dict[str, int]:
    """Walk ``directory`` and return ``{relative_path: size_bytes}`` for files with ``ext``.

    Keys are POSIX-style relative paths from ``directory`` (e.g.
    ``"sub-VG1-GC#78/foo.nwb"``), matching how S3 object keys are returned by
    :func:`fetch_remote_file_list`. ``ext`` is matched case-sensitively against the
    file suffix.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(root)
    return {
        p.relative_to(root).as_posix(): p.stat().st_size
        for p in root.rglob(f"*{ext}")
        if p.is_file()
    }


def fetch_remote_file_list(prefix: str = "", ext: str = ".nwb") -> dict[str, int]:
    """List the ``ext`` files the bucket holds under ``prefix``, with their sizes.

    ``prefix=""`` lists the whole bucket (slow — many requests). To list a
    single mouse, pass ``prefix="sub-VG1-GC#78/"``.
    """
    return {key: size for key, size in list_files(prefix) if key.endswith(ext)}


def compare_file_lists(
    local: dict[str, int],
    s3: dict[str, int],
) -> FileListDiff:
    """Compare two ``{key: size}`` listings, returning what differs.

    Pure function — no I/O. Combine with :func:`scan_local_files` and
    :func:`fetch_remote_file_list` to produce a full check.
    """
    matching: list[str] = []
    size_mismatch: list[tuple[str, int, int]] = []
    s3_keys = set(s3)
    local_keys = set(local)

    for key in s3_keys & local_keys:
        if local[key] == s3[key]:
            matching.append(key)
        else:
            size_mismatch.append((key, local[key], s3[key]))

    missing_local = sorted(s3_keys - local_keys)
    extra_local = sorted(local_keys - s3_keys)

    return FileListDiff(
        matching=sorted(matching),
        missing_local=missing_local,
        extra_local=extra_local,
        size_mismatch=sorted(size_mismatch),
    )
