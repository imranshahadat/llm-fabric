"""End-to-end ingestion demo for Project6.

Builds sample files (PDF with text/table/scanned page, DOCX with table,
CSV, HTML), runs the full upload -> load -> split pipeline on each, and
prints what came out. Run with:  uv run python scripts/ingest_demo.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pymupdf

SRC = str(Path(__file__).resolve().parent.parent / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from project6_1.core.config.settings import AppSettings  # noqa: E402
from project6_1.domains.infrastructure.ingestion import IngestionService  # noqa: E402

WORK = Path(tempfile.mkdtemp(prefix="ingest_demo_"))


def build_pdf(path: Path) -> None:
    """Text page + ruled table page + scanned page (text rendered as image)."""
    doc = pymupdf.open()

    page = doc.new_page()
    page.insert_text((72, 90), "ACME Annual Report 2026", fontsize=18)
    page.insert_text((72, 120), "Revenue grew 14 percent year over year.", fontsize=11)
    page.insert_text((72, 140), "Operating costs stayed flat at $9.2M.", fontsize=11)

    page2 = doc.new_page()
    cols = [72, 200, 330, 450]
    rows = [100, 130, 160, 190, 220]
    for x in cols:
        page2.draw_line((x, rows[0]), (x, rows[-1]))
    for y in rows:
        page2.draw_line((cols[0], y), (cols[-1], y))
    cells = [
        ("Region", "Revenue", "Growth", "Headcount"),
        ("EMEA", "$2.4M", "+8%", "142"),
        ("APAC", "$1.8M", "+21%", "97"),
        ("Americas", "$5.1M", "+14%", "310"),
    ]
    for r, row in enumerate(cells):
        for c, value in enumerate(row):
            page2.insert_text((cols[c] + 4, rows[r] + 20), value, fontsize=10)

    # Scanned page: text exists only as pixels, so only OCR can read it.
    # Built purely with pymupdf (render -> rasterize -> insert pixmap) to
    # avoid encoder quirks of other imaging libraries.
    page3 = doc.new_page()
    scan = pymupdf.open()
    scan_page = scan.new_page(width=600, height=300)
    scan_page.insert_text((30, 60), "Scanned memo: server migration", fontsize=20)
    scan_page.insert_text((30, 100), "The datacenter move finishes on December 15.", fontsize=14)
    scan_page.insert_text((30, 140), "Downtime is expected to be under one hour.", fontsize=14)
    scan_pixels = scan_page.get_pixmap(dpi=150)
    scan.close()
    page3.insert_image(pymupdf.Rect(50, 50, 450, 200), pixmap=scan_pixels)

    doc.save(path)
    doc.close()


def build_docx(path: Path) -> None:
    from docx import Document

    doc = Document()
    doc.add_heading("Procurement Policy", level=1)
    doc.add_paragraph("All purchases above $5,000 require two approvals.")
    doc.add_paragraph("Approved vendors are listed in the registry.")
    doc.add_heading("Vendor Fees", level=2)
    table = doc.add_table(rows=3, cols=2)
    table.style = "Table Grid"
    table.cell(0, 0).text = "Vendor"
    table.cell(0, 1).text = "Annual Fee"
    table.cell(1, 0).text = "Northwind"
    table.cell(1, 1).text = "$12,000"
    table.cell(2, 0).text = "Contoso"
    table.cell(2, 1).text = "$8,500"
    doc.save(path)


def build_csv(path: Path) -> None:
    path.write_text(
        "city,country,population\nLisbon,Portugal,545k\nOslo,Norway,700k\nKyoto,Japan,1460k\n",
        encoding="utf-8",
    )


def build_html(path: Path) -> None:
    path.write_text(
        "<html><head><style>body{color:red}</style></head><body>"
        "<h1>Pricing</h1><p>The starter plan costs $9 per month.</p>"
        "<table><tr><th>Plan</th><th>Price</th></tr>"
        "<tr><td>Starter</td><td>$9</td></tr><tr><td>Pro</td><td>$29</td></tr></table>"
        "</body></html>",
        encoding="utf-8",
    )


def run(name: str, service: IngestionService, path: Path) -> None:
    result = service.ingest(str(path), filename=path.name)
    print(f"\n=== {name} ===")
    for document in result["documents"]:
        page = document["metadata"].get("page", "-")
        kind = document["metadata"].get("kind")
        preview = document["content"].replace("\n", " ")[:90]
        print(f"  doc page={page} kind={kind}: {preview}")
    print(f"  -> {len(result['documents'])} documents, {len(result['chunks'])} chunks")


def main() -> int:
    WORK.mkdir(exist_ok=True)
    build_pdf(WORK / "report.pdf")
    build_docx(WORK / "policy.docx")
    build_csv(WORK / "cities.csv")
    build_html(WORK / "pricing.html")

    settings = AppSettings(
        ingest_upload_dir=str(WORK / "uploads"),
        ingest_chunk_size=400,
        ingest_chunk_overlap=80,
        ingest_enable_ocr=True,
    )
    service = IngestionService(settings)

    run("PDF (text + table + scanned page)", service, WORK / "report.pdf")
    run("DOCX (sections + table)", service, WORK / "policy.docx")
    run("CSV (as markdown table)", service, WORK / "cities.csv")
    run("HTML (text + table)", service, WORK / "pricing.html")

    # Content assertions — the pipeline must have found everything.
    pdf = service.ingest(str(WORK / "report.pdf"))
    contents = [d["content"] for d in pdf["documents"]]
    kinds = [d["metadata"].get("kind") for d in pdf["documents"]]

    checks = {
        "text page found": any("14 percent" in c for c in contents),
        "table as markdown found": any(
            "EMEA" in c and d["metadata"].get("kind") == "table"
            for d, c in zip(pdf["documents"], contents, strict=True)
        ),
        "scanned page OCR found": any("datacenter move" in c for c in contents),
        "table kind present": "table" in kinds,
        "page_ocr kind present": "page_ocr" in kinds,
        "chunks produced": len(pdf["chunks"]) > 0,
        "chunk metadata carries source": all(
            "source" in ch["metadata"] and "parent_document_id" in ch["metadata"]
            for ch in pdf["chunks"]
        ),
    }
    print("\n=== CHECKS ===")
    failed = False
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        failed |= not ok

    print("\nDEMO " + ("OK" if not failed else "FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
