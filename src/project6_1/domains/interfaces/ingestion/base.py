"""Base interfaces and data structures for document ingestion.

The ingestion port models the three stages of getting raw files into the
RAG knowledge base:

1. **Upload / staging** — a file (path, bytes, or URL) is validated and
   copied into the configured upload directory (:class:`SourceFile`).
2. **Loading** — a loader adapter parses a staged file into structured
   :class:`Document` items (text blocks, tables, images, OCR output).
3. **Splitting** — a splitter adapter turns documents into retrieval-ready
   :class:`Chunk` items sized for embedding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict


class SourceFile(TypedDict):
    """A validated, staged file ready to be loaded."""

    path: str
    filename: str
    file_type: str
    size_bytes: int


class Document(TypedDict):
    """One extracted content element from a source file.

    ``content`` is normalized plain text or markdown (tables are markdown,
    images carry their OCR transcript). ``metadata`` always contains
    ``source``, ``file_type``, ``kind`` (text | table | image | page_ocr)
    and, for paged formats, the 1-based ``page`` number.
    """

    id: str
    content: str
    metadata: dict[str, Any]


class Chunk(TypedDict):
    """A retrieval-ready piece of a :class:`Document` produced by a splitter."""

    id: str
    text: str
    metadata: dict[str, Any]


class IngestionResult(TypedDict):
    """Summary of a full ingestion run (upload -> load -> split)."""

    files: list[SourceFile]
    documents: list[Document]
    chunks: list[Chunk]
    warnings: list[str]


class BaseLoader(ABC):
    """Base interface that all document loaders must implement.

    A loader parses one staged file into :class:`Document` items. It must
    never raise for "empty" content — it returns an empty list instead —
    but it must raise :class:`~project6_1.core.exceptions.ingestion.DocumentLoadError`
    for unreadable or corrupt files.
    """

    #: Lowercase file extensions this loader handles (without the dot).
    supported_extensions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, source: SourceFile) -> list[Document]:
        """Parse the staged file and return its extracted documents."""
        raise NotImplementedError


class BaseSplitter(ABC):
    """Base interface that all splitters must implement.

    A splitter consumes :class:`Document` items and produces
    :class:`Chunk` items. It must preserve document metadata on every
    chunk and add its own bookkeeping keys (``chunk_index``,
    ``parent_document_id``, ...).
    """

    @abstractmethod
    def split(self, documents: list[Document]) -> list[Chunk]:
        """Split extracted documents into retrieval-ready chunks."""
        raise NotImplementedError
