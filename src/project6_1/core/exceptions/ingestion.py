"""Ingestion error hierarchy: upload staging, loading and splitting failures."""

from __future__ import annotations

from .base import AppError


class IngestionError(AppError):
    """Base error for the document ingestion pipeline."""


class UnsupportedFileTypeError(IngestionError):
    """The uploaded file has an extension no registered loader handles."""


class FileTooLargeError(IngestionError):
    """The uploaded file exceeds the configured size limit."""


class UploadStagingError(IngestionError):
    """A file could not be staged (missing, unreadable, download failed)."""


class DocumentLoadError(IngestionError):
    """A staged file exists but could not be parsed by its loader."""
