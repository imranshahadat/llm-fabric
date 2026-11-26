"""Upload staging: validate and stage files into the ingest upload directory.

Staging is the first ingestion stage. Whatever the caller hands over — a
local path, a directory, raw bytes, or a URL — it ends up as a
:class:`~project6_1.domains.interfaces.ingestion.SourceFile` inside the
configured upload directory, ready for a loader.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ....core.config.settings import AppSettings, get_settings
from ....core.exceptions.ingestion import (
    FileTooLargeError,
    UnsupportedFileTypeError,
    UploadStagingError,
)
from ....core.logging.setup import get_logger
from ...interfaces.ingestion.base import SourceFile
from .factory import LOADER_REGISTRY
from .shared import sanitize_filename, suffix_of

_log = get_logger().bind(component="upload")


def stage_upload(
    source: str | Path | bytes,
    *,
    filename: str | None = None,
    settings: AppSettings | None = None,
) -> list[SourceFile]:
    """Validate and stage one or more files, returning what was staged.

    Accepted inputs:

    - ``str``/``Path`` pointing to a file or a directory (directories are
      walked recursively and every supported file inside is staged).
    - ``str`` starting with ``http://`` or ``https://`` — the file is
      downloaded into the upload directory.
    - ``bytes`` — raw file content; ``filename`` must name its extension
      (e.g. ``"report.pdf"``).

    Files are copied into ``settings.ingest_upload_dir`` (created on
    demand), so the caller's originals are never touched. Size, emptiness
    and extension are validated here — loaders can trust staged files.
    """
    s = settings or get_settings()
    upload_dir = Path(s.ingest_upload_dir).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = max(s.ingest_max_file_mb, 0) * 1024 * 1024

    if isinstance(source, bytes):
        return _stage_bytes(source, filename, upload_dir, max_bytes)
    return _stage_path(Path(source), filename, upload_dir, max_bytes)


def _stage_path(
    source: Path,
    filename: str | None,
    upload_dir: Path,
    max_bytes: int,
) -> list[SourceFile]:
    """Stage a file or every supported file inside a directory."""
    text = str(source)

    if text.startswith(("http://", "https://")):
        return [_stage_url(text, filename, upload_dir, max_bytes)]

    if not source.exists():
        raise UploadStagingError(f"Upload source does not exist: {source}")

    if source.is_dir():
        staged: list[SourceFile] = []
        for child in sorted(source.rglob("*")):
            if child.is_file() and suffix_of(child.name) in LOADER_REGISTRY:
                staged.extend(_stage_file(child, None, upload_dir, max_bytes))
        if not staged:
            raise UnsupportedFileTypeError(
                f"Directory {source} contains no files with a supported "
                f"extension; supported: {sorted(LOADER_REGISTRY)}"
            )
        return staged

    return _stage_file(source, filename, upload_dir, max_bytes)


def _stage_file(
    path: Path,
    filename: str | None,
    upload_dir: Path,
    max_bytes: int,
) -> list[SourceFile]:
    """Stage a single on-disk file."""
    original = filename or path.name
    file_type = suffix_of(original)

    if file_type not in LOADER_REGISTRY:
        raise UnsupportedFileTypeError(
            f"No loader handles file type {file_type!r} ({original}); "
            f"supported: {sorted(LOADER_REGISTRY)}"
        )
    if not path.is_file():
        raise UploadStagingError(f"Upload source is not a regular file: {path}")

    size = path.stat().st_size
    if size == 0:
        raise UploadStagingError(f"Upload source is empty: {path}")
    _check_size(size, original, max_bytes)

    target = upload_dir / sanitize_filename(original, upload_dir)
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)

    staged: SourceFile = {
        "path": str(target),
        "filename": target.name,
        "file_type": file_type,
        "size_bytes": size,
    }
    _log.info("file_staged", filename=target.name, size_bytes=size)
    return [staged]


def _stage_bytes(
    data: bytes,
    filename: str | None,
    upload_dir: Path,
    max_bytes: int,
) -> list[SourceFile]:
    """Stage raw bytes under the given (or a generated) filename."""
    if not data:
        raise UploadStagingError("Uploaded bytes are empty")

    original = filename or f"upload-{uuid.uuid4().hex[:8]}.bin"
    file_type = suffix_of(original)
    if file_type not in LOADER_REGISTRY:
        raise UnsupportedFileTypeError(
            f"No loader handles file type {file_type!r} ({original}); "
            f"supported: {sorted(LOADER_REGISTRY)}"
        )
    _check_size(len(data), original, max_bytes)

    target = upload_dir / sanitize_filename(original, upload_dir)
    target.write_bytes(data)

    staged: SourceFile = {
        "path": str(target),
        "filename": target.name,
        "file_type": file_type,
        "size_bytes": len(data),
    }
    _log.info("file_staged", filename=target.name, size_bytes=len(data))
    return [staged]


def _stage_url(
    url: str,
    filename: str | None,
    upload_dir: Path,
    max_bytes: int,
) -> SourceFile:
    """Download a URL into the upload directory and stage it."""
    name = filename or Path(urlparse(url).path).name or "download.bin"
    file_type = suffix_of(name)
    if file_type not in LOADER_REGISTRY:
        raise UnsupportedFileTypeError(
            f"No loader handles file type {file_type!r} ({name}); "
            f"supported: {sorted(LOADER_REGISTRY)}"
        )

    request = Request(url, headers={"User-Agent": "project6-1-ingest/0.1"})  # noqa: S310
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310
            data = response.read(max_bytes + 1 if max_bytes else None)
    except OSError as error:
        raise UploadStagingError(f"Could not download {url}: {error}") from error

    return _stage_bytes(data, name, upload_dir, max_bytes)[0]


def _check_size(size: int, name: str, max_bytes: int) -> None:
    """Raise when ``size`` exceeds the configured limit."""
    if max_bytes and size > max_bytes:
        raise FileTooLargeError(
            f"{name} is {size} bytes; the limit is {max_bytes} bytes (INGEST_MAX_FILE_MB)"
        )
