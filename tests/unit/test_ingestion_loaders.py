"""Unit tests for ingestion loaders: PDF, DOCX, CSV, HTML, JSON, TXT.

PDF/DOCX fixtures are generated on the fly with the same libraries the
loaders use (pymupdf, python-docx), covering text pages, ruled tables
and an image-only (scanned) page — which the text-only pipeline must
skip without failing.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from project6_1.core.exceptions.ingestion import DocumentLoadError
from project6_1.domains.infrastructure.ingestion.docx_loader import DocxLoader
from project6_1.domains.infrastructure.ingestion.factory import build_loader
from project6_1.domains.infrastructure.ingestion.pdf_loader import PdfLoader
from project6_1.domains.infrastructure.ingestion.text_loader import (
    CsvLoader,
    HtmlLoader,
    JsonLoader,
    TextLoader,
)
from project6_1.domains.interfaces.ingestion.base import SourceFile


def make_source(path: Path, file_type: str | None = None) -> SourceFile:
    """Build a SourceFile dict for an on-disk fixture."""
    return SourceFile(
        path=str(path),
        filename=path.name,
        file_type=file_type or path.suffix.lstrip("."),
        size_bytes=path.stat().st_size,
    )


# ---- fixtures ----------------------------------------------------------------


@pytest.fixture()
def pdf_with_all_elements(tmp_path: Path) -> Path:
    """PDF: text page + ruled table page + image-only (scanned) page."""
    path = tmp_path / "sample.pdf"
    # pymupdf objects are kept as Any: its stubs are only partial.
    doc: Any = pymupdf.open()

    page = doc.new_page()
    page.insert_text((72, 90), "Simple text page", fontsize=14)
    page.insert_text((72, 120), "It mentions a budget of $3,000.", fontsize=11)

    page2 = doc.new_page()
    cols = [72, 200, 330]
    rows = [100, 130, 160, 190]
    for x in cols:
        page2.draw_line((x, rows[0]), (x, rows[-1]))
    for y in rows:
        page2.draw_line((cols[0], y), (cols[-1], y))
    for r, row in enumerate([("City", "Sales"), ("Berlin", "120"), ("Oslo", "90")]):
        for c, value in enumerate(row):
            page2.insert_text((cols[c] + 4, rows[r] + 20), value, fontsize=10)

    # Image-only page: text exists only as pixels, so a text-only loader
    # must skip it (no OCR in this pipeline).
    page3 = doc.new_page()
    scan: Any = pymupdf.open()
    scan_page = scan.new_page(width=500, height=200)
    scan_page.insert_text((20, 50), "Scanned delivery note 42", fontsize=18)
    scan_page.insert_text((20, 90), "Left warehouse on March 3.", fontsize=12)
    scan_pixels = scan_page.get_pixmap(dpi=150)
    scan.close()
    page3.insert_image(pymupdf.Rect(50, 50, 350, 150), pixmap=scan_pixels)

    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture()
def docx_with_table(tmp_path: Path) -> Path:
    """DOCX: two headed sections plus one 2x2 table."""
    from docx import Document

    path = tmp_path / "policy.docx"
    doc: Any = Document()  # python-docx ships no type stubs
    doc.add_heading("Expense Rules", level=1)
    doc.add_paragraph("Meals above $75 need a receipt.")
    doc.add_heading("Limits", level=2)
    doc.add_paragraph("Hotels are capped at $200 per night.")
    table = doc.add_table(rows=2, cols=2)
    table.style = "Table Grid"
    table.cell(0, 0).text = "Category"
    table.cell(0, 1).text = "Limit"
    table.cell(1, 0).text = "Hotel"
    table.cell(1, 1).text = "$200"
    doc.save(str(path))
    return path


# ---- PDF loader --------------------------------------------------------------


def test_pdf_extracts_text_and_table(pdf_with_all_elements: Path) -> None:
    docs = PdfLoader().load(make_source(pdf_with_all_elements))

    kinds = [d["metadata"]["kind"] for d in docs]
    assert "text" in kinds
    assert "table" in kinds

    text_doc = next(d for d in docs if d["metadata"]["kind"] == "text")
    assert "budget of $3,000" in text_doc["content"]
    assert text_doc["metadata"]["page"] == 1

    table_doc = next(d for d in docs if d["metadata"]["kind"] == "table")
    assert table_doc["content"].startswith("|")
    assert "Berlin" in table_doc["content"]
    assert table_doc["metadata"]["page"] == 2
    assert table_doc["metadata"]["rows"] == 3


def test_pdf_skips_image_only_page(pdf_with_all_elements: Path) -> None:
    """The scanned page has no text layer -> skipped, no error, no OCR."""
    docs = PdfLoader().load(make_source(pdf_with_all_elements))

    kinds = [d["metadata"]["kind"] for d in docs]
    assert "image" not in kinds and "page_ocr" not in kinds
    # Only the two pages with a text layer produce documents.
    assert {d["metadata"]["page"] for d in docs} == {1, 2}
    assert all("delivery note" not in d["content"] for d in docs)


def test_corrupt_pdf_raises_load_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4 this is not really a pdf\x00\x01\x02")
    with pytest.raises(DocumentLoadError):
        PdfLoader().load(make_source(path))


# ---- DOCX loader -------------------------------------------------------------


def test_docx_sections_and_table(docx_with_table: Path) -> None:
    docs = DocxLoader().load(make_source(docx_with_table))

    kinds = [d["metadata"]["kind"] for d in docs]
    assert "text" in kinds and "table" in kinds

    table_doc = next(d for d in docs if d["metadata"]["kind"] == "table")
    assert "| Hotel | $200 |" in table_doc["content"]
    assert table_doc["metadata"]["rows"] == 2

    text_docs = [d for d in docs if d["metadata"]["kind"] == "text"]
    joined = "\n".join(d["content"] for d in text_docs)
    assert "receipt" in joined
    assert "$200 per night" in joined
    # Heading text is kept as section metadata on section documents.
    assert any(d["metadata"].get("section") == "Limits" for d in text_docs)
    # The heading must NOT be duplicated inside the section content.
    assert all(d["content"].count("Expense Rules") == 0 for d in text_docs)
    assert all(d["content"].count("Limits") == 0 for d in text_docs)
    # A table below a heading carries that heading as section context.
    assert table_doc["metadata"].get("section") == "Limits"


def test_docx_with_inline_picture_is_text_only(tmp_path: Path) -> None:
    """A picture inside a .docx is ignored; text and tables still load."""
    from docx import Document

    path = tmp_path / "with_image.docx"
    doc: Any = Document()
    doc.add_heading("Report", level=1)
    doc.add_paragraph("Picture follows.")

    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    doc.add_picture(io.BytesIO(png), width=914400)
    doc.save(str(path))

    docs = DocxLoader().load(make_source(path))
    kinds = [d["metadata"]["kind"] for d in docs]
    assert "image" not in kinds
    assert "Picture follows." in "\n".join(d["content"] for d in docs)


# ---- text family loaders -----------------------------------------------------


def test_text_loader_txt_and_md(tmp_path: Path) -> None:
    loader = TextLoader()

    txt = tmp_path / "notes.txt"
    txt.write_text("first line\nsecond line", encoding="utf-8")
    docs = loader.load(make_source(txt))
    assert len(docs) == 1
    assert docs[0]["metadata"]["file_type"] == "txt"

    md = tmp_path / "readme.md"
    md.write_text("# Title\n\nBody text.", encoding="utf-8")
    docs_md = loader.load(make_source(md))
    assert "# Title" in docs_md[0]["content"]


def test_text_loader_latin1_fallback(tmp_path: Path) -> None:
    loader = TextLoader()
    path = tmp_path / "legacy.txt"
    path.write_bytes("caf\xe9 cr\xe8me".encode("latin-1"))
    docs = loader.load(make_source(path))
    assert "caf\xe9" in docs[0]["content"]


def test_json_loader(tmp_path: Path) -> None:
    loader = JsonLoader()
    path = tmp_path / "data.json"
    path.write_text('{"a": 1, "b": ["x", "y"]}', encoding="utf-8")
    docs = loader.load(make_source(path))
    assert '"a": 1' in docs[0]["content"]


def test_csv_loader_becomes_markdown_table(tmp_path: Path) -> None:
    loader = CsvLoader()
    path = tmp_path / "cities.csv"
    path.write_text("city,country\nLisbon,Portugal\nOslo,Norway\n", encoding="utf-8")
    docs = loader.load(make_source(path))

    assert len(docs) == 1
    assert docs[0]["metadata"]["kind"] == "table"
    assert "| Lisbon | Portugal |" in docs[0]["content"]


def test_tsv_loader(tmp_path: Path) -> None:
    loader = CsvLoader()
    path = tmp_path / "matrix.tsv"
    path.write_text("a\tb\n1\t2\n", encoding="utf-8")
    docs = loader.load(make_source(path))
    assert "| 1 | 2 |" in docs[0]["content"]


def test_html_loader_text_and_table(tmp_path: Path) -> None:
    loader = HtmlLoader()
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><style>b{color:red}</style></head><body>"
        "<p>Total revenue rose.</p>"
        "<table><tr><th>Q</th><th>Rev</th></tr><tr><td>Q1</td><td>10</td></tr></table>"
        "</body></html>",
        encoding="utf-8",
    )
    docs = loader.load(make_source(path))

    assert "Total revenue rose" in "\n".join(d["content"] for d in docs)
    table_doc = next(d for d in docs if d["metadata"]["kind"] == "table")
    assert "Q1" in table_doc["content"] and "10" in table_doc["content"]


# ---- factory routing ---------------------------------------------------------


def test_factory_builds_correct_loader() -> None:
    assert isinstance(build_loader("pdf"), PdfLoader)
    assert isinstance(build_loader("docx"), DocxLoader)
    assert isinstance(build_loader("csv"), CsvLoader)
    assert isinstance(build_loader("html"), HtmlLoader)
    assert isinstance(build_loader("json"), JsonLoader)
    assert isinstance(build_loader("md"), TextLoader)


def test_factory_rejects_unknown_type() -> None:
    from project6_1.core.exceptions.ingestion import UnsupportedFileTypeError

    with pytest.raises(UnsupportedFileTypeError):
        build_loader("exe")
