"""Factory for building ingestion loaders and splitters from settings."""

from __future__ import annotations

from collections.abc import Callable

from ....core.config.settings import AppSettings, get_settings
from ....core.exceptions.ingestion import UnsupportedFileTypeError
from ...interfaces.ingestion.base import BaseLoader, BaseSplitter
from .docx_loader import DocxLoader
from .pdf_loader import PdfLoader
from .splitter import RecursiveSplitter
from .text_loader import CsvLoader, HtmlLoader, JsonLoader, TextLoader

# File extension -> loader factory. Extensions are lowercase, no dot.
LOADER_REGISTRY: dict[str, Callable[[AppSettings], BaseLoader]] = {
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
    """Create the loader adapter registered for ``file_type`` (e.g. ``"pdf"``)."""
    loader_class = LOADER_REGISTRY.get(file_type.lower())
    if loader_class is None:
        raise UnsupportedFileTypeError(
            f"No loader handles file type {file_type!r}; supported: {sorted(LOADER_REGISTRY)}"
        )
    return loader_class(settings or get_settings())


def available_loaders() -> tuple[str, ...]:
    """Return every file type the ingestion pipeline can load."""
    return tuple(sorted(LOADER_REGISTRY.keys()))


def build_splitter(settings: AppSettings | None = None) -> BaseSplitter:
    """Create the configured chunking splitter."""
    return RecursiveSplitter(settings or get_settings())
