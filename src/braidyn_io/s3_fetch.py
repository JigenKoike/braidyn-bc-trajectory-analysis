"""Read-only access to the BraiDyn-BC public S3 bucket via Python stdlib.

The dataset lives at ``https://braidyn-bc-buckets.s3.amazonaws.com/`` as a
publicly-readable bucket. We use unauthenticated HTTP ``ListObjectsV2`` calls
through ``urllib`` — no AWS credentials, no ``boto3``.

For *downloading* files in bulk, see ``scripts/sync_s3_to_server.py``: bulk
transfers shell out to ``curl -C -`` to handle a ~10.72 GB single-connection
cap observed on some network routes to S3.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


S3_BUCKET_URL = "https://braidyn-bc-buckets.s3.amazonaws.com"


def _log(msg: str) -> None:
    """Timestamped log line to stdout (matches the format used by sync scripts)."""
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def _s3_list(
    params: dict[str, str],
    max_retries: int = 5,
) -> tuple[ET.Element, str]:
    """Execute one ListObjectsV2 call with exponential-backoff retries.

    Returns ``(root_xml_element, namespace_prefix_with_braces)``. The namespace
    prefix is the leading ``{http://...}`` part that S3's XML responses use, or
    an empty string if the XML has no default namespace.
    """
    query = urllib.parse.urlencode(params)
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(f"{S3_BUCKET_URL}?{query}", timeout=60) as resp:
                root = ET.fromstring(resp.read().decode())
            ns = root.tag.split("}")[0] + "}" if "}" in root.tag else ""
            return root, ns
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 5 * (2**attempt)
            _log(
                f"  S3 list retry {attempt + 1}/{max_retries} after {wait}s: "
                f"{type(e).__name__}: {e}"
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover


def list_folders() -> list[str]:
    """Return every top-level prefix (one per mouse) in the bucket.

    Each returned string ends with ``/`` — e.g. ``"sub-VG1-GC#78/"`` —
    matching S3's prefix convention.
    """
    root, ns = _s3_list({"list-type": "2", "delimiter": "/", "max-keys": "1000"})
    return [cp.find(f"{ns}Prefix").text for cp in root.findall(f"{ns}CommonPrefixes")]


def list_files(prefix: str) -> list[tuple[str, int]]:
    """Return ``(key, size_in_bytes)`` for every object whose key starts with ``prefix``.

    Pagination is handled transparently. Filtering by extension (e.g. ``.nwb``)
    is the caller's responsibility.
    """
    all_files: list[tuple[str, int]] = []
    continuation_token: str | None = None
    while True:
        params: dict[str, str] = {
            "list-type": "2",
            "prefix": prefix,
            "max-keys": "1000",
        }
        if continuation_token:
            params["continuation-token"] = continuation_token
        root, ns = _s3_list(params)
        for item in root.findall(f"{ns}Contents"):
            key = item.find(f"{ns}Key").text
            size = int(item.find(f"{ns}Size").text)
            all_files.append((key, size))
        truncated = root.find(f"{ns}IsTruncated")
        if truncated is not None and truncated.text == "true":
            token = root.find(f"{ns}NextContinuationToken")
            continuation_token = token.text if token is not None else None
            if not continuation_token:
                break
        else:
            break
    return all_files
