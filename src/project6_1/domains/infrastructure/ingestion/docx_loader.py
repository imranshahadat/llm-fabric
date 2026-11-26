"""DOCX loader: extracts sections, tables and embedded images.

The document body is walked in order. Paragraphs are accumulated into
section-level documents (a heading starts a new section); tables are
emitted as markdown documents with ``kind="table"``. Embedded pictures
are OCR'd when OCR is enabled (``kind="image"``).
"""

from __future__ import annotations

from typing import Any

from ....core.config.settings import AppSettings
from ....core.exceptions.ingestion import DocumentLoadError
from ....core.logging.setup import get_logger
from ...interfaces.ingestion.base import BaseLoader, Document, SourceFile
from .ocr import OcrEngine
from .shared import document_id, rows_to_markdown

# Accumulated paragraph characters before the section is flushed even
# without a new heading (keeps documents embeddable-sized).
_FLUSH_CHARS = 1000

# Paragraph styles that start a new section ("Heading 1", "标题 1", ...).
_HEADING_STYLE_PREFIX = "heading"

# OCR text shorter than this from a picture is noise (logos) — dropped.
_OCR_MIN_CHARS = 20


class DocxLoader(BaseLoader):
    """Loads .docx files with section grouping and table extraction."""

    supported_extensions = ("docx",)

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._log = get_logger().bind(loader="docx")
        self._ocr = OcrEngine(settings)

    def load(self, source: SourceFile) -> list[Document]:
        """Parse a staged .docx into ordered documents."""
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
            nonlocal buffer, buffer_chars, section_title
            text = "\n".join(buffer).strip()
            if text:
                content = f"{section_title}\n{text}" if section_title else text
                metadata = self._meta(source, kind="text")
                if section_title:
                    metadata["section"] = section_title
                documents.append(self._document(content, metadata))
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
                    section_title = text
                    buffer_chars += len(text)
                    buffer.append(text)
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
                    documents.append(self._document(markdown, metadata))
                table_index += 1

        flush()

        documents.extend(self._extract_images(docx_doc, source))
        self._log.info("docx_loaded", filename=source["filename"], documents=len(documents))
        return documents

    def _extract_images(self, docx_doc: Any, source: SourceFile) -> list[Document]:
        """OCR meaningful pictures embedded in the document."""
        if not self._settings.ingest_enable_ocr or not self._ocr.available:
            return []

        documents: list[Document] = []
        for index, shape in enumerate(docx_doc.inline_shapes):
            try:
                blob = shape.image.blob
            except Exception as error:  # linked / non-picture shapes have no blob
                self._log.warning("image_blob_unreadable", error=str(error))
                continue
            ocr_text = self._ocr.read_bytes(blob)
            if ocr_text and len(ocr_text) >= _OCR_MIN_CHARS:
                metadata = self._meta(source, kind="image")
                metadata["element_index"] = index
                documents.append(self._document(ocr_text, metadata))
        return documents

    def _meta(self, source: SourceFile, *, kind: str) -> dict[str, Any]:
        return {"source": source["filename"], "file_type": source["file_type"], "kind": kind}

    def _document(self, content: str, metadata: dict[str, Any]) -> Document:
        return Document(
            id=document_id(metadata["source"], content, metadata.get("page")),
            content=content,
            metadata=metadata,
        )
