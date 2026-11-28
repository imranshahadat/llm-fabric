"""Factory for building ingestion loaders and splitters.

``LOADER_REGISTRY`` maps file extensions to loader classes. Upload
staging consults the same registry, so a file type can be staged if and
only if a loader exists for it.
"""

from __future__ import annotations

from ....core.config.settings import AppSettings, get_settings
from ....core.exceptions.ingestion import UnsupportedFileTypeError
from ...interfaces.ingestion.base import BaseLoader, BaseSplitter
from .docx_loader import DocxLoader
from .pdf_loader import PdfLoader
from .splitter import RecursiveSplitter
from .text_loader import CsvLoader, HtmlLoader, JsonLoader, TextLoader

# File extension -> loader class. Extensions are lowercase, no dot.
# Text-only loaders need no settings, so classes are registered directly.
LOADER_REGISTRY: dict[str, type[BaseLoader]] = {
    "pdf": PdfLoader,
    "docx": DocxLoader,
    "txt": TextLoader,
    "md": TextLoader,
    "markdown": TextLoader,
    "csv": CsvLoader,
    "tsv": CsvLoader,
    "json": JsonLoader,
    "html": HtmlLoader,
    "htm": HtmlLoader,
}


def build_loader(file_type: str, settings: AppSettings | None = None) -> BaseLoader:
    """Create the loader registered for ``file_type`` (e.g. ``"pdf"``).

    ``settings`` is accepted for API compatibility but unused: the
    loaders are pure text extractors with nothing to configure.
    """
    loader_class = LOADER_REGISTRY.get(file_type.lower())
    if loader_class is None:
        raise UnsupportedFileTypeError(
            f"No loader handles file type {file_type!r}; supported: {sorted(LOADER_REGISTRY)}"
        )
    return loader_class()


def available_loaders() -> tuple[str, ...]:
    """Return every file type the ingestion pipeline can load."""
    return tuple(sorted(LOADER_REGISTRY.keys()))


def build_splitter(settings: AppSettings | None = None) -> BaseSplitter:
    """Create the configured chunking splitter."""
    return RecursiveSplitter(settings or get_settings())
