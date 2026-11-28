"""DOCX loader: sections and tables only — no images, no OCR.

The document body is walked in order. Paragraphs are accumulated into
section-level documents (a heading starts a new section); tables are
emitted as markdown documents with ``kind="table"``. Embedded pictures
are skipped on purpose: this pipeline is text-only.
"""

from __future__ import annotations

from typing import Any

from ....core.exceptions.ingestion import DocumentLoadError
from ....core.logging.setup import get_logger
from ...interfaces.ingestion.base import BaseLoader, Document, SourceFile
from .shared import document_id, rows_to_markdown

# Accumulated paragraph characters before the section is flushed even
# without a new heading (keeps documents embeddable-sized).
_FLUSH_CHARS = 1000

# Paragraph styles that start a new section ("Heading 1", "标题 1", ...).
_HEADING_STYLE_PREFIX = "heading"


class DocxLoader(BaseLoader):
    """Loads .docx files with section grouping and table extraction."""

    supported_extensions = ("docx",)

    def __init__(self) -> None:
        self._log = get_logger().bind(loader="docx")

    def load(self, source: SourceFile) -> list[Document]:
        """Parse a staged .docx into ordered text/table documents."""
        from docx import Document as DocxDocument
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        path = source["path"]
        # python-docx ships no type stubs; the object is Any at this boundary.
        docx_doc: Any
        try:
            docx_doc = DocxDocument(path)
        except Exception as error:
            raise DocumentLoadError(
                f"Could not parse DOCX {source['filename']}: {error}"
            ) from error

        documents: list[Document] = []
        section_title = ""
        buffer: list[str] = []
        buffer_chars = 0
        table_index = 0

        def flush() -> None:
            """Emit the buffered paragraphs as one section document."""
            nonlocal buffer, buffer_chars
            text = "\n".join(buffer).strip()
            if text:
                metadata = self._meta(source, kind="text")
                if section_title:
                    metadata["section"] = section_title
                documents.append(self._document(text, metadata))
            buffer, buffer_chars = [], 0

        for child in docx_doc.element.body.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]

            if tag == "p":
                paragraph: Any = Paragraph(child, docx_doc)
                text = paragraph.text.strip()
                style: Any = paragraph.style
                style_name = str(style.name if style else "").lower()
                if not text:
                    continue
                if style_name.startswith(_HEADING_STYLE_PREFIX):
                    flush()
                    # The heading becomes the section title (metadata), not
                    # body text, so it is not duplicated in the content.
                    section_title = text
                    continue
                buffer.append(text)
                buffer_chars += len(text)
                if buffer_chars >= _FLUSH_CHARS:
                    flush()

            elif tag == "tbl":
                flush()
                table = Table(child, docx_doc)
                rows = [[cell.text for cell in row.cells] for row in table.rows]
                markdown = rows_to_markdown(rows)
                if markdown:
                    metadata = self._meta(source, kind="table")
                    metadata["rows"] = len(rows)
                    metadata["cols"] = len(rows[0]) if rows else 0
                    metadata["element_index"] = table_index
                    if section_title:
                        # Keep the surrounding heading for retrieval context.
                        metadata["section"] = section_title
                    documents.append(self._document(markdown, metadata))
                table_index += 1

        flush()

        self._log.info("docx_loaded", filename=source["filename"], documents=len(documents))
        return documents

    def _meta(self, source: SourceFile, *, kind: str) -> dict[str, Any]:
        """Standard metadata keys every document from this file carries."""
        return {"source": source["filename"], "file_type": source["file_type"], "kind": kind}

    def _document(self, content: str, metadata: dict[str, Any]) -> Document:
        """Build one Document with a deterministic, re-ingest-safe ID."""
        return Document(
            id=document_id(metadata["source"], content, metadata.get("page")),
            content=content,
            metadata=metadata,
        )
