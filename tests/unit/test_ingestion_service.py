"""Unit tests for upload staging, the factory registry and IngestionService."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from project6_1.core.config.settings import AppSettings
from project6_1.core.exceptions.ingestion import (
    FileTooLargeError,
    UnsupportedFileTypeError,
    UploadStagingError,
)
from project6_1.domains.infrastructure.ingestion.factory import (
    available_loaders,
    build_loader,
    build_splitter,
)
from project6_1.domains.infrastructure.ingestion.service import IngestionService
from project6_1.domains.infrastructure.ingestion.upload import stage_upload
from project6_1.domains.interfaces.embed.base import BaseEmbed, EmbedRequest
from project6_1.domains.interfaces.vector_store.base import (
    BaseVectorStore,
    SearchQuery,
    SearchResult,
    VectorDocument,
)


def make_settings(tmp_path: Path, **overrides: Any) -> AppSettings:
    defaults: dict[str, Any] = {"ingest_upload_dir": str(tmp_path / "uploads")}
    defaults.update(overrides)
    return AppSettings(**defaults)


# ---- fakes -------------------------------------------------------------------


class FakeEmbed(BaseEmbed):
    """Deterministic embedding for tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate_embedding(self, request: EmbedRequest) -> list[float]:
        self.calls.append(request["text"])
        return [0.1, 0.2, 0.3]


class FakeVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.upserted: list[VectorDocument] = []

    async def upsert(self, documents: list[VectorDocument]) -> None:
        self.upserted.extend(documents)

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        return []

    async def delete(self, ids: list[str]) -> None:
        pass


# ---- staging -----------------------------------------------------------------


def test_stage_bytes_writes_into_upload_dir(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    staged = stage_upload(b"hello world", filename="notes.txt", settings=settings)

    assert len(staged) == 1
    source = staged[0]
    assert source["file_type"] == "txt"
    assert source["size_bytes"] == 11
    assert Path(source["path"]).read_text() == "hello world"
    assert Path(source["path"]).parent == Path(settings.ingest_upload_dir)


def test_stage_path_copies_into_upload_dir(tmp_path: Path) -> None:
    original = tmp_path / "originals"
    original.mkdir()
    (original / "doc.txt").write_text("copy me")

    settings = make_settings(tmp_path)
    staged = stage_upload(original / "doc.txt", settings=settings)

    staged_path = Path(staged[0]["path"])
    assert staged_path.parent == Path(settings.ingest_upload_dir)
    assert staged_path.read_text() == "copy me"
    assert (original / "doc.txt").exists()  # original untouched


def test_stage_directory_stages_supported_files_only(tmp_path: Path) -> None:
    folder = tmp_path / "inbox"
    folder.mkdir()
    (folder / "a.txt").write_text("a")
    (folder / "b.exe").write_text("b")
    (folder / "sub").mkdir()
    (folder / "sub" / "c.md").write_text("c")

    staged = stage_upload(folder, settings=make_settings(tmp_path))
    names = [s["filename"] for s in staged]
    assert names == ["a.txt", "c.md"]


def test_stage_same_name_twice_does_not_overwrite(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    stage_upload(b"first", filename="doc.txt", settings=settings)
    staged = stage_upload(b"second", filename="doc.txt", settings=settings)

    assert Path(staged[0]["path"]).read_bytes() == b"second"
    assert staged[0]["filename"] == "doc-1.txt"


def test_stage_rejects_unsupported_extension(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        stage_upload(b"MZ", filename="virus.exe", settings=make_settings(tmp_path))


def test_stage_rejects_empty_bytes(tmp_path: Path) -> None:
    with pytest.raises(UploadStagingError):
        stage_upload(b"", filename="empty.txt", settings=make_settings(tmp_path))


def test_stage_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(UploadStagingError):
        stage_upload(tmp_path / "nope.txt", settings=make_settings(tmp_path))


def test_stage_rejects_oversized_file(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, ingest_max_file_mb=1)
    with pytest.raises(FileTooLargeError):
        stage_upload(b"x" * (1024 * 1024 + 1), filename="big.txt", settings=settings)


def test_stage_allows_file_at_limit(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, ingest_max_file_mb=1)
    staged = stage_upload(b"x" * 1024, filename="ok.txt", settings=settings)
    assert staged[0]["size_bytes"] == 1024


# ---- factory -----------------------------------------------------------------


def test_available_loaders_cover_document_families() -> None:
    loaders = available_loaders()
    for expected in ("pdf", "docx", "txt", "md", "csv", "json", "html"):
        assert expected in loaders


def test_build_splitter_returns_configured_splitter(tmp_path: Path) -> None:
    splitter = build_splitter(make_settings(tmp_path, ingest_chunk_size=333))
    assert splitter._size == 333  # type: ignore[attr-defined]


def test_build_loader_rejects_unknown(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        build_loader("rar", make_settings(tmp_path))


# ---- service -----------------------------------------------------------------


def test_service_ingests_bytes_end_to_end(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = IngestionService(settings)

    result = service.ingest(b"alpha\nbeta\ngamma", filename="notes.txt")

    assert len(result["files"]) == 1
    assert len(result["documents"]) == 1
    assert len(result["chunks"]) == 1
    chunk = result["chunks"][0]
    assert chunk["metadata"]["source"] == "notes.txt"
    assert "gamma" in chunk["text"]


def test_service_ingests_directory(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "a.txt").write_text("alpha document")
    (inbox / "b.md").write_text("# markdown document")

    result = IngestionService(make_settings(tmp_path)).ingest(inbox)
    assert {f["file_type"] for f in result["files"]} == {"txt", "md"}
    assert len(result["chunks"]) == 2


async def test_ingest_and_index_embeds_and_upserts(tmp_path: Path) -> None:
    embed = FakeEmbed()
    store = FakeVectorStore()
    service = IngestionService(make_settings(tmp_path), embed=embed, vector_store=store)

    text = "\n\n".join(f"Paragraph {i} about invoicing." for i in range(20))
    result = await service.ingest_and_index(text.encode(), filename="invoice.txt")

    assert embed.calls, "embedding model was never called"
    assert len(store.upserted) == len(result["chunks"])

    first = store.upserted[0]
    assert first["embedding"] == [0.1, 0.2, 0.3]
    assert first["text"] in embed.calls
    # Qdrant point ids must be valid UUIDs.
    uuid.UUID(first["id"])  # raises if not UUID-shaped
    # Metadata flows through to the payload.
    assert first["metadata"]["source"] == "invoice.txt"
    assert "parent_document_id" in first["metadata"]


async def test_ingest_and_index_without_chunks_is_noop(tmp_path: Path) -> None:
    store = FakeVectorStore()
    service = IngestionService(make_settings(tmp_path), embed=FakeEmbed(), vector_store=store)
    result = await service.ingest_and_index(b"   ", filename="blank.txt")
    assert store.upserted == []
    assert result["chunks"] == []
