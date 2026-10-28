"""Tests for the S3 mirror script's handling of object keys."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "sync_s3_to_server", Path(__file__).resolve().parent.parent / "scripts/sync_s3_to_server.py"
)
sync = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sync)


def test_a_key_lands_under_the_destination(tmp_path: Path) -> None:
    got = sync._safe_destination(tmp_path, "sub-VG1-GC#48/session.nwb")
    assert got == (tmp_path / "sub-VG1-GC#48/session.nwb").resolve()


@pytest.mark.parametrize("key", [
    "../escaped.nwb",
    "sub-VG1/../../escaped.nwb",
    "a/b/../../../escaped.nwb",
])
def test_a_key_that_climbs_out_is_refused(tmp_path: Path, key: str) -> None:
    # The keys come from someone else's bucket listing, and this script is
    # run where there is room for terabytes.
    with pytest.raises(ValueError, match="outside"):
        sync._safe_destination(tmp_path, key)


def test_dots_inside_a_name_are_fine(tmp_path: Path) -> None:
    got = sync._safe_destination(tmp_path, "sub-VG1/ses-2024-01-24..task.nwb")
    assert got.parent == (tmp_path / "sub-VG1").resolve()
