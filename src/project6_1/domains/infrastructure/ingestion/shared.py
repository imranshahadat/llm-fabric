"""Small helpers shared across ingestion adapters."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_SLUG = re.compile(r"-{2,}")


def document_id(source: str, content: Any, *extra: Any) -> str:
    """Deterministic document/chunk ID: stable across re-ingestion of the same file.

    A pure hash (not a random UUID) means re-uploading an unchanged file
    produces the same IDs, so vector-store upserts overwrite cleanly
    instead of duplicating.
    """
    digest = hashlib.sha1(  # noqa: S324 - identity, not security
        "|".join(str(part) for part in (source, *extra, content)).encode("utf-8")
    ).hexdigest()
    return digest[:32]


def suffix_of(filename: str) -> str:
    """Return the lowercased extension of ``filename`` without the dot.

    Compound extensions used by real-world documents (``.tar.gz`` is not
    one of them, but ``report.PDF`` is) are normalized to lowercase.
    Files with no extension map to an empty string.
    """
    suffix = Path(filename).suffix.lower()
    return suffix[1:] if suffix.startswith(".") else suffix


def rows_to_markdown(rows: list[list[str]]) -> str:
    """Render table rows as a GitHub-flavored markdown table.

    The first row is treated as the header. Empty tables return "".
    Cell values have newlines flattened so each row is one line.
    """
    cleaned = [[str(cell).replace("\n", " ").strip() for cell in row] for row in rows if row]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    def _line(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    header = _line(cleaned[0])
    divider = "| " + " | ".join(["---"] * width) + " |"
    body = [_line(row) for row in cleaned[1:]]
    return "\n".join([header, divider, *body])


def sanitize_filename(filename: str, target_dir: Path) -> str:
    """Return a safe, unique filename inside ``target_dir``.

    Strips directory components, unicode-normalizes, replaces unsafe
    characters, and adds a short numeric suffix if the name already
    exists (``report.pdf`` -> ``report-1.pdf``) so staged uploads never
    overwrite each other.
    """
    stem = Path(filename.replace("\\", "/")).name
    stem = unicodedata.normalize("NFKC", stem).strip()
    cleaned = _SLUG.sub("-", _SAFE_NAME.sub("-", stem)) or "upload"

    # Keep the extension; collapse runs of dashes in the stem only.
    path = Path(cleaned)
    ext = path.suffix.lower()
    stem_part = path.stem.strip("-") or "upload"

    candidate = f"{stem_part}{ext}"
    counter = 1
    while (target_dir / candidate).exists():
        candidate = f"{stem_part}-{counter}{ext}"
        counter += 1
    return candidate
