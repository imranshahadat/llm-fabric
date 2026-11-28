"""PDF loader: text and table extraction with pymupdf — no images, no OCR.

Strategy per page (document order is preserved):

1. **Tables** — ``page.find_tables()`` detects ruled and borderless
   tables; each becomes a markdown :class:`Document` (``kind="table"``).
2. **Text** — text blocks lying inside a table bbox are skipped (they
   are table cell text); the rest are joined in reading order
   (``kind="text"``).
3. **Image-only pages** — a page with no text layer and no tables (a
   scan) yields nothing and is logged. We deliberately do not OCR.

Scanned PDFs are therefore out of scope by design: this pipeline works
on PDFs with a real text layer (digitally generated PDFs).
"""

from __future__ import annotations

from typing import Any

import pymupdf

from ....core.exceptions.ingestion import DocumentLoadError
from ....core.logging.setup import get_logger
from ...interfaces.ingestion.base import BaseLoader, Document, SourceFile
from .shared import document_id

# A text block overlapping a table bbox by this share of its area counts
# as table content (bboxes can undershoot the drawn grid slightly).
_TABLE_OVERLAP_SHARE = 0.5

# Table bboxes are grown by this many points per side before overlap tests.
_TABLE_BBOX_PAD = 6.0


class PdfLoader(BaseLoader):
    """Loads text and tables from PDFs that have a text layer."""

    supported_extensions = ("pdf",)

    def __init__(self) -> None:
        self._log = get_logger().bind(loader="pdf")

    def load(self, source: SourceFile) -> list[Document]:
        """Parse a staged PDF into page-ordered text/table documents."""
        documents: list[Document] = []

        # pymupdf ships only partial type stubs; objects are kept as Any
        # at this boundary so the strict checker sees our types.
        pdf_doc: Any = None
        try:
            pdf_doc = pymupdf.open(source["path"])
            for page_number in range(1, int(pdf_doc.page_count) + 1):
                page: Any = pdf_doc[page_number - 1]
                documents.extend(self._load_page(page, page_number, source))
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

    def _load_page(self, page: Any, page_number: int, source: SourceFile) -> list[Document]:
        """Extract tables and text from one page, in document order."""
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

        documents.extend(self._extract_text(page, table_bboxes, meta))

        if not documents:
            # No text layer and no tables -> a scan or purely visual page.
            self._log.info("page_skipped_no_text", filename=source["filename"], page=page_number)
        return documents

    def _extract_text(
        self,
        page: Any,
        table_bboxes: list[Any],
        meta: dict[str, Any],
    ) -> list[Document]:
        """Join non-table text blocks into one reading-order document."""
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

    # ---- helpers ------------------------------------------------------------

    def _page_meta(self, source: SourceFile, page_number: int) -> dict[str, Any]:
        """Standard metadata keys every document on a page carries."""
        return {
            "source": source["filename"],
            "file_type": source["file_type"],
            "page": page_number,
        }

    def _document(self, content: str, metadata: dict[str, Any]) -> Document:
        """Build one Document with a deterministic, re-ingest-safe ID."""
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
