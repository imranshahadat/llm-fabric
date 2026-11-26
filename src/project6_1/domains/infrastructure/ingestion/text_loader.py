"""Loaders for plain-text families: TXT, MD, CSV, JSON, HTML.

One module, one loader class per family. They all share the same
document shape; CSV and HTML tables become markdown ``kind="table"``
documents so the downstream splitter treats them like PDF tables.
"""

from __future__ import annotations

import csv
import io
import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ....core.config.settings import AppSettings
from ....core.exceptions.ingestion import DocumentLoadError
from ....core.logging.setup import get_logger
from ...interfaces.ingestion.base import BaseLoader, Document, SourceFile
from .shared import document_id, rows_to_markdown

# CSV rows per table document (header repeated in each).
_CSV_ROWS_PER_DOC = 200


class _TextLoaderBase(BaseLoader):
    """Shared plumbing for the single-file text loaders."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._log = get_logger().bind(loader=type(self).__name__)

    def load(self, source: SourceFile) -> list[Document]:
        """Read the staged file and parse it according to the family."""
        path = Path(source["path"])
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise DocumentLoadError(f"Could not read {source['filename']}: {error}") from error

        try:
            documents = self._parse(raw, source)
        except DocumentLoadError:
            raise
        except Exception as error:
            raise DocumentLoadError(f"Could not parse {source['filename']}: {error}") from error

        self._log.info("loaded", filename=source["filename"], documents=len(documents))
        return documents

    def _parse(self, raw: bytes, source: SourceFile) -> list[Document]:
        raise NotImplementedError

    def _document(self, content: str, source: SourceFile, kind: str, **extra: Any) -> Document:
        metadata: dict[str, Any] = {
            "source": source["filename"],
            "file_type": source["file_type"],
            "kind": kind,
            **extra,
        }
        return Document(
            id=document_id(source["filename"], content),
            content=content,
            metadata=metadata,
        )


class TextLoader(_TextLoaderBase):
    """Loads TXT and Markdown files as one document."""

    supported_extensions = ("txt", "md", "markdown")

    def _parse(self, raw: bytes, source: SourceFile) -> list[Document]:
        text = _decode(raw).strip()
        return [self._document(text, source, "text")] if text else []


class JsonLoader(_TextLoaderBase):
    """Loads JSON files as one pretty-printed document."""

    supported_extensions = ("json",)

    def _parse(self, raw: bytes, source: SourceFile) -> list[Document]:
        payload = json.loads(_decode(raw))
        text = json.dumps(payload, ensure_ascii=False, indent=2).strip()
        return [self._document(text, source, "text")] if text else []


class CsvLoader(_TextLoaderBase):
    """Loads CSV files as markdown table documents (header repeated per group)."""

    supported_extensions = ("csv", "tsv")

    def _parse(self, raw: bytes, source: SourceFile) -> list[Document]:
        dialect = "excel-tab" if source["file_type"] == "tsv" else "excel"
        reader = csv.reader(io.StringIO(_decode(raw)), dialect=dialect)
        rows = [list(row) for row in reader if any(cell.strip() for cell in row)]
        if not rows:
            return []

        header, data_rows = rows[0], rows[1:]
        documents: list[Document] = []
        for start in range(0, len(data_rows), _CSV_ROWS_PER_DOC):
            group = data_rows[start : start + _CSV_ROWS_PER_DOC]
            markdown = rows_to_markdown([header, *group])
            if markdown:
                documents.append(
                    self._document(
                        markdown,
                        source,
                        "table",
                        rows=len(group) + 1,
                        cols=len(header),
                        row_offset=start,
                    )
                )
        return documents


class HtmlLoader(_TextLoaderBase):
    """Loads HTML files: readable text and <table> elements as markdown."""

    supported_extensions = ("html", "htm")

    def _parse(self, raw: bytes, source: SourceFile) -> list[Document]:
        parser = _HtmlContentParser()
        parser.feed(_decode(raw))
        parser.close()

        documents: list[Document] = []
        text = "\n\n".join(parser.blocks).strip()
        if text:
            documents.append(self._document(text, source, "text"))
        for index, rows in enumerate(parser.tables):
            markdown = rows_to_markdown(rows)
            if markdown:
                documents.append(
                    self._document(
                        markdown,
                        source,
                        "table",
                        element_index=index,
                        rows=len(rows),
                        cols=len(rows[0]) if rows else 0,
                    )
                )
        return documents


class _HtmlContentParser(HTMLParser):
    """Collects visible text blocks and table cell grids from HTML."""

    _SKIP = frozenset({"script", "style", "head", "title", "meta", "noscript"})
    _BLOCK_END = frozenset(
        {
            "p",
            "div",
            "section",
            "article",
            "li",
            "tr",
            "br",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "pre",
            "blockquote",
            "td",
            "th",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.tables: list[list[list[str]]] = []
        self._buffer: list[str] = []
        self._skip_depth = 0
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return

        text = " ".join("".join(self._buffer).split())
        self._buffer = []

        if self._cell is not None and self._row is not None and tag in ("td", "th"):
            # Cell data was collected in _cell, not the shared buffer.
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif self._row is not None and self._table is not None and tag == "tr":
            if any(self._row):
                self._table.append(self._row)
            self._row = None
        elif self._table is not None and tag == "table":
            if self._table:
                self.tables.append(self._table)
            self._table = None
        elif tag in self._BLOCK_END and text:
            self.blocks.append(text)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            if self._cell is not None:
                self._cell.append(data)
            else:
                self._buffer.append(data)


def _decode(raw: bytes) -> str:
    """Decode bytes as UTF-8 with a latin-1 fallback for legacy files."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")
