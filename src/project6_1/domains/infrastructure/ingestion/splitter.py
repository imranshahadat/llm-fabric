"""Table-aware recursive character splitter.

Plain text is split on a hierarchy of natural boundaries (paragraphs,
lines, sentences, words) down to character level, with character overlap
between consecutive chunks. Markdown tables are kept semantically
intact: small tables stay whole, large ones are split by rows with the
header block repeated on every chunk so no chunk loses its columns.
"""

from __future__ import annotations

from ....core.config.settings import AppSettings
from ....core.logging.setup import get_logger
from ...interfaces.ingestion.base import BaseSplitter, Chunk, Document
from .shared import document_id

_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


class RecursiveSplitter(BaseSplitter):
    """Recursive character splitter with table header repetition."""

    def __init__(self, settings: AppSettings) -> None:
        self._size = max(settings.ingest_chunk_size, 50)
        self._overlap = min(settings.ingest_chunk_overlap, self._size // 2)
        self._log = get_logger().bind(splitter="recursive")

    def split(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into chunks, preserving and enriching metadata."""
        chunks: list[Chunk] = []
        for document in documents:
            pieces = self._pieces_for(document)
            for index, text in enumerate(pieces):
                metadata = {
                    **document["metadata"],
                    "parent_document_id": document["id"],
                    "chunk_index": index,
                    "chunk_count": len(pieces),
                }
                chunks.append(
                    Chunk(
                        id=document_id(document["id"], index, text),
                        text=text,
                        metadata=metadata,
                    )
                )
        self._log.info("split_documents", documents=len(documents), chunks=len(chunks))
        return chunks

    # ---- dispatch -----------------------------------------------------------

    def _pieces_for(self, document: Document) -> list[str]:
        """Choose the splitting strategy for one document."""
        content = document["content"].strip()
        if not content:
            return []
        if document["metadata"].get("kind") == "table":
            return self._split_table(content)
        return self._split_text(content)

    # ---- plain text ---------------------------------------------------------

    def _split_text(self, text: str, separators: tuple[str, ...] = _SEPARATORS) -> list[str]:
        """Recursively split ``text`` on natural boundaries, then merge with overlap."""
        separator, remaining = self._pick_separator(text, separators)
        splits = text.split(separator) if separator else list(text)

        pieces: list[str] = []
        small: list[str] = []
        for part in splits:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self._size:
                small.append(part)
                continue
            if small:
                pieces.extend(self._merge(small, separator))
                small = []
            pieces.extend(self._split_text(part, remaining or separators[1:]))

        if small:
            pieces.extend(self._merge(small, separator))
        return pieces

    def _pick_separator(
        self, text: str, separators: tuple[str, ...]
    ) -> tuple[str, tuple[str, ...]]:
        """Return the first separator that occurs in ``text`` plus the rest."""
        for index, separator in enumerate(separators):
            if separator == "":
                return "", ()
            if separator in text:
                return separator, separators[index + 1 :]
        return "", ()

    def _merge(self, pieces: list[str], separator: str) -> list[str]:
        """Join small pieces into chunks up to the size limit, with overlap.

        Every piece fits ``self._size`` already (larger ones were split
        recursively before reaching here), so a chunk always assembles
        complete pieces — text is never cut mid-word at this stage.
        """
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for piece in pieces:
            extra = len(piece) + (len(separator) if current else 0)
            if current and current_len + extra > self._size:
                chunks.append(separator.join(current).strip())
                current, current_len = self._overlap_tail(current, separator)
            current.append(piece)
            current_len += len(piece) + (len(separator) if len(current) > 1 else 0)

        if current:
            chunks.append(separator.join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _overlap_tail(self, pieces: list[str], separator: str) -> tuple[list[str], int]:
        """Take trailing pieces worth up to ``self._overlap`` chars as the next
        chunk's opening context."""
        tail: list[str] = []
        total = 0
        for piece in reversed(pieces):
            candidate = len(piece) + (len(separator) if tail else 0)
            if total + candidate > self._overlap and tail:
                break
            tail.insert(0, piece)
            total += candidate
        return tail, total

    # ---- markdown tables ----------------------------------------------------

    def _split_table(self, content: str) -> list[str]:
        """Split a markdown table by rows, repeating the header on each chunk."""
        lines = content.splitlines()
        if len(lines) < 2:
            return [content] if content.strip() else []

        if len(content) <= self._size:
            return [content]

        header_block = "\n".join(lines[:2])  # header row + |---| divider
        if len(header_block) + 2 > self._size:
            # Degenerate table (huge header): fall back to text splitting.
            return self._split_text(content)

        chunks: list[str] = []
        rows: list[str] = []
        rows_len = 0
        for line in lines[2:]:
            if rows and rows_len + len(line) + 1 + len(header_block) > self._size:
                chunks.append("\n".join([header_block, *rows]))
                rows, rows_len = [], 0
            rows.append(line)
            rows_len += len(line) + 1
        if rows:
            chunks.append("\n".join([header_block, *rows]))
        return [chunk for chunk in chunks if chunk.strip()]
