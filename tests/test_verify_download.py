"""Tests for `braidyn_io.verify_download`."""

from __future__ import annotations

from pathlib import Path

from braidyn_io.verify_download import (
    FileListDiff,
    scan_local_files,
    compare_file_lists,
)


def test_scan_local_files_reports_name_and_size(tmp_path: Path) -> None:
    """Files with the requested extension are listed with their byte sizes."""
    (tmp_path / "mouse_a").mkdir()
    (tmp_path / "mouse_a" / "s1.nwb").write_bytes(b"x" * 100)
    (tmp_path / "mouse_a" / "s2.nwb").write_bytes(b"x" * 250)
    (tmp_path / "mouse_b").mkdir()
    (tmp_path / "mouse_b" / "s1.nwb").write_bytes(b"x" * 7)

    listing = scan_local_files(tmp_path)
    assert listing == {
        "mouse_a/s1.nwb": 100,
        "mouse_a/s2.nwb": 250,
        "mouse_b/s1.nwb": 7,
    }


def test_scan_local_files_filters_by_extension(tmp_path: Path) -> None:
    """Files not matching `ext` are excluded."""
    (tmp_path / "a.nwb").write_bytes(b"x" * 10)
    (tmp_path / "b.mp4").write_bytes(b"x" * 20)
    (tmp_path / "c.txt").write_bytes(b"x" * 5)

    nwb_only = scan_local_files(tmp_path, ext=".nwb")
    assert set(nwb_only) == {"a.nwb"}

    mp4_only = scan_local_files(tmp_path, ext=".mp4")
    assert set(mp4_only) == {"b.mp4"}


def test_scan_local_files_rejects_a_missing_directory(tmp_path: Path) -> None:
    """A missing directory raises (better than silently returning empty)."""
    import pytest

    with pytest.raises(NotADirectoryError):
        scan_local_files(tmp_path / "nope")


def test_compare_file_lists_when_everything_matches() -> None:
    local = {"a.nwb": 10, "b.nwb": 20}
    s3 = {"a.nwb": 10, "b.nwb": 20}
    diff = compare_file_lists(local, s3)
    assert diff.matching == ["a.nwb", "b.nwb"]
    assert diff.missing_local == []
    assert diff.size_mismatch == []
    assert diff.extra_local == []
    assert diff.is_clean


def test_compare_file_lists_detects_a_missing_file() -> None:
    local = {"a.nwb": 10}
    s3 = {"a.nwb": 10, "b.nwb": 20}
    diff = compare_file_lists(local, s3)
    assert diff.matching == ["a.nwb"]
    assert diff.missing_local == ["b.nwb"]
    assert not diff.is_clean


def test_compare_file_lists_detects_a_wrong_size() -> None:
    local = {"a.nwb": 10, "b.nwb": 99}     # b is partial / corrupted
    s3 = {"a.nwb": 10, "b.nwb": 20}
    diff = compare_file_lists(local, s3)
    assert diff.matching == ["a.nwb"]
    assert diff.size_mismatch == [("b.nwb", 99, 20)]
    assert not diff.is_clean


def test_compare_file_lists_detects_an_extra_local_file() -> None:
    local = {"a.nwb": 10, "b.nwb": 20, "stray.nwb": 7}
    s3 = {"a.nwb": 10, "b.nwb": 20}
    diff = compare_file_lists(local, s3)
    assert diff.extra_local == ["stray.nwb"]
    assert not diff.is_clean


def test_diff_summary_format() -> None:
    diff = FileListDiff(
        matching=["a", "b", "c"],
        missing_local=["d"],
        size_mismatch=[("e", 1, 2)],
        extra_local=[],
    )
    assert diff.summary() == "3 OK | 1 missing | 1 mismatched | 0 extra"


def test_local_vs_s3_via_compute(tmp_path: Path) -> None:
    """End to end: write some local files, stand in for the bucket, compare."""
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "a.nwb").write_bytes(b"x" * 100)
    (tmp_path / "m" / "b.nwb").write_bytes(b"x" * 50)  # will be flagged partial

    local = scan_local_files(tmp_path)
    s3 = {"m/a.nwb": 100, "m/b.nwb": 200, "m/c.nwb": 300}

    diff = compare_file_lists(local, s3)
    assert diff.matching == ["m/a.nwb"]
    assert diff.missing_local == ["m/c.nwb"]
    assert diff.size_mismatch == [("m/b.nwb", 50, 200)]
    assert not diff.is_clean
