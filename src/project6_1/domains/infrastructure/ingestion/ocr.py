"""Tesseract-backed OCR with graceful degradation.

OCR is a *best-effort* enrichment: if tesseract or pytesseract is not
installed, the engine reports itself unavailable and callers skip OCR
(a scanned PDF simply yields no page text rather than an error).
"""

from __future__ import annotations

import io
import shutil
from typing import Any

from ....core.config.settings import AppSettings
from ....core.logging.setup import get_logger


class OcrEngine:
    """OCR wrapper around pytesseract for raster images and PDF pages."""

    def __init__(self, settings: AppSettings) -> None:
        self._language = settings.ingest_ocr_language
        self._dpi = settings.ingest_ocr_dpi
        self._log = get_logger().bind(component="ocr")
        self._pytesseract: Any | None = None
        if settings.ingest_enable_ocr and shutil.which("tesseract"):
            try:  # pytesseract is an optional runtime dependency
                import pytesseract

                self._pytesseract = pytesseract
            except ImportError:
                self._log.warning("ocr_unavailable", reason="pytesseract package not installed")
        elif settings.ingest_enable_ocr:
            self._log.warning("ocr_unavailable", reason="tesseract binary not found on PATH")

    @property
    def available(self) -> bool:
        """True when tesseract and pytesseract are both usable."""
        return self._pytesseract is not None

    def read_bytes(self, image_bytes: bytes) -> str:
        """OCR a raster image given as bytes; empty string on failure."""
        if not self._pytesseract:
            return ""
        try:
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))
            return str(self._pytesseract.image_to_string(image, lang=self._language)).strip()
        except Exception as error:
            self._log.warning("ocr_failed", error=str(error))
            return ""

    def rasterize_and_read(self, page: Any) -> str:
        """Rasterize a PDF page at the configured DPI and OCR the result."""
        if not self._pytesseract:
            return ""
        try:
            pixmap: Any = page.get_pixmap(dpi=self._dpi)
            return self.read_bytes(pixmap.tobytes("png"))
        except Exception as error:
            self._log.warning("ocr_rasterize_failed", error=str(error))
            return ""
