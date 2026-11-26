"""PDF loader: extracts text, tables and images from any PDF.

Strategy per page (order preserved by document order of elements):

1. **Tables** — ``page.find_tables()`` detects ruled and borderless tables;
   each is emitted as a markdown :class:`Document` with ``kind="table"``.
2. **Text** — text blocks overlapping a table bbox are skipped, the rest
   are joined in reading order (``kind="text"``).
3. **Images** — embedded raster images are extracted and OCR'd
   (``kind="image"``) when OCR is enabled and produces meaningful text.
4. **Scanned pages** — a page with no text layer and no tables is
   rasterized and OCR'd whole (``kind="page_ocr"``).

OCR is optional: it degrades gracefully when ``pytesseract``/the
``tesseract`` binary are unavailable (a warning is recorded instead).
"""

from __future__ import annotations

from typing import Any

import pymupdf

from ....core.config.settings import AppSettings
from ....core.exceptions.ingestion import DocumentLoadError
from ....core.logging.setup import get_logger
from ...interfaces.ingestion.base import BaseLoader, Document, SourceFile
from .ocr import OcrEngine
from .shared import document_id

# Text shorter than this from an image OCR is treated as noise (logos,
# page furniture) and dropped.
_OCR_MIN_CHARS = 20

# A text block overlapping a table bbox by this share of its area counts
# as table content (bboxes can undershoot the drawn grid slightly).
_TABLE_OVERLAP_SHARE = 0.5

# Table bboxes are grown by this many points per side before overlap tests.
_TABLE_BBOX_PAD = 6.0


class PdfLoader(BaseLoader):
    """Loads PDFs with table, image and scanned-page handling."""

    supported_extensions = ("pdf",)

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._log = get_logger().bind(loader="pdf")
        self._ocr = OcrEngine(settings)

    def load(self, source: SourceFile) -> list[Document]:
        """Parse a staged PDF into page-ordered documents."""
        documents: list[Document] = []

        # pymupdf 2.x ships only partial type stubs; the objects are kept
        # as Any at this boundary so the strict checker sees our types.
        pdf_doc: Any = None
        try:
            pdf_doc = pymupdf.open(source["path"])
            for page_number in range(1, int(pdf_doc.page_count) + 1):
                page: Any = pdf_doc[page_number - 1]
                documents.extend(self._load_page(pdf_doc, page, page_number, source))
        except DocumentLoadError:
            raise
        except Exception as error:  # pymupdf raises RuntimeError/ValueError on bad files
            raise DocumentLoadError(f"Could not parse PDF {source['filename']}: {error}") from error
        finally:
            if pdf_doc is not None:
                pdf_doc.close()

        self._log.info(
            "pdf_loaded",
            filename=source["filename"],
            documents=len(documents),
        )
        return documents

    # ---- per-page extraction ------------------------------------------------

    def _load_page(
        self,
        doc: Any,
        page: Any,
        page_number: int,
        source: SourceFile,
    ) -> list[Document]:
        """Extract tables, text, images (and OCR fallbacks) from one page."""
        meta = self._page_meta(source, page_number)
        documents: list[Document] = []

        tables: Any = page.find_tables()
        table_bboxes: list[Any] = [pymupdf.Rect(table.bbox) for table in tables.tables]

        for index, table in enumerate(tables.tables):
            markdown = str(table.to_markdown(clean=False))
            if markdown.strip():
                documents.append(
                    self._document(
                        markdown.strip(),
                        {
                            **meta,
                            "kind": "table",
                            "element_index": index,
                            "rows": int(table.row_count),
                            "cols": int(table.col_count),
                        },
                    )
                )

        text_documents = self._extract_text(page, table_bboxes, meta)
        documents.extend(text_documents)

        did_page_ocr = False
        if not documents and self._settings.ingest_enable_ocr:
            # No text layer and no tables -> scanned page: OCR the raster.
            ocr_text = self._ocr.rasterize_and_read(page)
            if ocr_text:
                documents.append(self._document(ocr_text, {**meta, "kind": "page_ocr"}))
                did_page_ocr = True

        if self._settings.ingest_pdf_extract_images and not did_page_ocr:
            # Whole-page OCR already covered embedded images; per-image OCR
            # would only duplicate the text.
            documents.extend(self._extract_images(doc, page, meta))

        return documents

    def _extract_text(
        self,
        page: Any,
        table_bboxes: list[Any],
        meta: dict[str, Any],
    ) -> list[Document]:
        """Join non-table text blocks into reading-order documents."""
        blocks: Any = page.get_text("blocks")
        kept: list[tuple[float, float, str]] = []
        for block in blocks:
            x0, y0, x1, y1, text = (
                float(block[0]),
                float(block[1]),
                float(block[2]),
                float(block[3]),
                str(block[4]),
            )
            if not text.strip():
                continue
            if _inside_any_table((x0, y0, x1, y1), table_bboxes):
                continue
            kept.append((y0, x0, text.strip()))

        if not kept:
            return []

        kept.sort(key=lambda item: (round(item[0], 1), item[1]))
        content = "\n".join(text for _y, _x, text in kept)
        return [self._document(content, {**meta, "kind": "text"})]

    def _extract_images(
        self,
        doc: Any,
        page: Any,
        meta: dict[str, Any],
    ) -> list[Document]:
        """OCR meaningful raster images embedded in the page."""
        if not self._settings.ingest_enable_ocr or not self._ocr.available:
            return []

        documents: list[Document] = []
        for index, image in enumerate(page.get_images(full=True)):
            xref = image[0]
            try:
                extracted: Any = doc.extract_image(xref)
            except Exception as error:  # malformed image stream — skip, keep loading
                self._log.warning("image_extract_failed", xref=xref, error=str(error))
                continue

            ocr_text = self._ocr.read_bytes(extracted["image"])
            if ocr_text and len(ocr_text) >= _OCR_MIN_CHARS:
                documents.append(
                    self._document(
                        ocr_text,
                        {
                            **meta,
                            "kind": "image",
                            "element_index": index,
                            "image_ext": str(extracted["ext"]),
                        },
                    )
                )
        return documents

    # ---- helpers ------------------------------------------------------------

    def _page_meta(self, source: SourceFile, page_number: int) -> dict[str, Any]:
        return {
            "source": source["filename"],
            "file_type": source["file_type"],
            "page": page_number,
        }

    def _document(self, content: str, metadata: dict[str, Any]) -> Document:
        return Document(
            id=document_id(metadata["source"], content, metadata.get("page")),
            content=content,
            metadata=metadata,
        )


def _inside_any_table(
    block: tuple[float, float, float, float],
    table_bboxes: list[Any],
) -> bool:
    """True when a text block lies mostly inside a table bbox."""
    block_rect: Any = pymupdf.Rect(*block)
    block_area = float(block_rect.width) * float(block_rect.height)
    if block_area <= 0:
        return False
    for bbox in table_bboxes:
        padded = _pad(bbox, _TABLE_BBOX_PAD)
        intersection: Any = block_rect & padded  # empty rect when disjoint
        if intersection.is_empty:
            continue
        overlap = float(intersection.width) * float(intersection.height)
        if overlap / block_area >= _TABLE_OVERLAP_SHARE:
            return True
    return False


def _pad(rect: Any, margin: float) -> Any:
    """Grow a rect by ``margin`` points on every side."""
    return pymupdf.Rect(
        rect.x0 - margin,
        rect.y0 - margin,
        rect.x1 + margin,
        rect.y1 + margin,
    )
