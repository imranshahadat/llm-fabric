"""Ingestion infrastructure adapters: upload staging, loaders, splitters, service."""

from .factory import available_loaders, build_loader, build_splitter
from .pdf_loader import PdfLoader
from .service import IngestionService
from .splitter import RecursiveSplitter
from .upload import stage_upload

__all__ = [
    "IngestionService",
    "PdfLoader",
    "RecursiveSplitter",
    "available_loaders",
    "build_loader",
    "build_splitter",
    "stage_upload",
]
