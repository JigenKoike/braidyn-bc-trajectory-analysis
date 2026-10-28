"""BraiDyn-BC data access layer.

This package isolates the BraiDyn-BC dataset's I/O conventions from the
analysis code, so the same path/listing/integrity utilities can be reused
across notebooks, scripts, and (eventually) other projects that consume the
same dataset.

Modules:

- :mod:`braidyn_io.paths` — parse and format session NWB filenames.
- :mod:`braidyn_io.s3_fetch` — list folders/files on the public S3 bucket.
- :mod:`braidyn_io.verify_download` — compare local files against the S3 source.
"""

from .verify_download import (
    FileListDiff,
    scan_local_files,
    compare_file_lists,
    fetch_remote_file_list,
)
from .paths import (
    MODALITIES,
    TASK_TYPES,
    SessionId,
    format_session_filename,
    parse_session_filename,
    session_path,
)
from .s3_fetch import S3_BUCKET_URL, list_files, list_folders

__all__ = [
    # paths
    "SessionId",
    "TASK_TYPES",
    "MODALITIES",
    "parse_session_filename",
    "format_session_filename",
    "session_path",
    # s3_fetch
    "S3_BUCKET_URL",
    "list_folders",
    "list_files",
    # verify_download
    "FileListDiff",
    "scan_local_files",
    "fetch_remote_file_list",
    "compare_file_lists",
]
