"""Unit tests for the table-aware recursive splitter."""

from __future__ import annotations

import tempfile
from itertools import pairwise
from typing import Any

import pytest

from project6_1.core.config.settings import AppSettings
from project6_1.domains.infrastructure.ingestion.splitter import RecursiveSplitter
from project6_1.domains.interfaces.ingestion.base import Document


def make_settings(**overrides: Any) -> AppSettings:
    return AppSettings(ingest_upload_dir=tempfile.mkdtemp(prefix="ingest-test-"), **overrides)


def make_document(content: str, kind: str = "text", **extra: Any) -> Document:
    metadata: dict[str, Any] = {
        "source": "sample.pdf",
        "file_type": "pdf",
        "page": 3,
        "kind": kind,
        **extra,
    }
    return Document(id="doc-1", content=content, metadata=metadata)


def test_small_text_stays_single_chunk() -> None:
    splitter = RecursiveSplitter(make_settings())
    chunks = splitter.split([make_document("A short paragraph.")])

    assert len(chunks) == 1
    assert chunks[0]["text"] == "A short paragraph."
    assert chunks[0]["metadata"]["parent_document_id"] == "doc-1"
    assert chunks[0]["metadata"]["chunk_index"] == 0
    assert chunks[0]["metadata"]["chunk_count"] == 1
    # Document metadata is preserved on the chunk.
    assert chunks[0]["metadata"]["page"] == 3
    assert chunks[0]["metadata"]["source"] == "sample.pdf"
    assert chunks[0]["metadata"]["kind"] == "text"


def test_long_text_respects_chunk_size() -> None:
    splitter = RecursiveSplitter(make_settings(ingest_chunk_size=200, ingest_chunk_overlap=0))
    text = "\n\n".join(f"Paragraph {i} " + "word " * 12 for i in range(30))
    chunks = splitter.split([make_document(text)])

    assert len(chunks) > 1
    assert all(len(c["text"]) <= 200 for c in chunks)
    assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c["metadata"]["chunk_count"] == len(chunks) for c in chunks)


def test_consecutive_chunks_overlap() -> None:
    splitter = RecursiveSplitter(make_settings(ingest_chunk_size=150, ingest_chunk_overlap=40))
    text = "\n".join(f"Sentence number {i} is here." for i in range(40))
    chunks = splitter.split([make_document(text)])

    assert len(chunks) > 1
    for first, second in pairwise(chunks):
        head = second["text"].split("\n")[0]
        assert head in first["text"] or head in text


def test_never_breaks_mid_word_on_natural_boundaries() -> None:
    splitter = RecursiveSplitter(make_settings(ingest_chunk_size=80, ingest_chunk_overlap=0))
    text = " ".join("elephant giraffe dolphin" for _ in range(30))
    chunks = splitter.split([make_document(text)])
    words = {w for chunk in chunks for w in chunk["text"].split() if w.isalpha()}
    assert words <= {"elephant", "giraffe", "dolphin"}


def test_small_table_kept_whole() -> None:
    table = "\n".join(["| Region | Sales |", "| --- | --- |", "| EMEA | 100 |", "| APAC | 200 |"])
    splitter = RecursiveSplitter(make_settings(ingest_chunk_size=500))
    chunks = splitter.split([make_document(table, kind="table")])

    assert len(chunks) == 1
    assert "| APAC | 200 |" in chunks[0]["text"]
    assert chunks[0]["metadata"]["kind"] == "table"


def test_large_table_repeats_header_per_chunk() -> None:
    rows = [f"| item-{i} | {i * 7} kg |" for i in range(60)]
    table = "\n".join(["| Item | Weight |", "| --- | --- |", *rows])
    splitter = RecursiveSplitter(make_settings(ingest_chunk_size=300, ingest_chunk_overlap=0))
    chunks = splitter.split([make_document(table, kind="table")])

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["text"].startswith("| Item | Weight |")
        assert "| --- | --- |" in chunk["text"]
        assert len(chunk["text"]) <= 300 + 40  # small tolerance


def test_empty_document_produces_no_chunks() -> None:
    splitter = RecursiveSplitter(make_settings())
    assert splitter.split([make_document("   ")]) == []


def test_ids_are_stable_and_unique() -> None:
    splitter = RecursiveSplitter(make_settings())
    text_a = "\n".join(f"line {i}" for i in range(50))
    text_b = "\n".join(f"other {i}" for i in range(50))
    docs = [make_document(text_a), make_document(text_b)]
    chunks = splitter.split(docs)

    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    # Deterministic: re-splitting the same input reproduces the same ids.
    assert ids == [c["id"] for c in RecursiveSplitter(make_settings()).split(docs)]


@pytest.mark.parametrize("size", [50, 100])
def test_chunk_size_below_minimum_clamped(size: int) -> None:
    splitter = RecursiveSplitter(make_settings(ingest_chunk_size=size, ingest_chunk_overlap=200))
    text = "word " * 100
    chunks = splitter.split([make_document(text)])
    assert all(len(c["text"]) <= size + 40 for c in chunks)
