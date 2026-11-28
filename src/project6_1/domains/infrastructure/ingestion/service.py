"""Ingestion orchestrator: upload -> load -> split (-> embed -> index).

``ingest`` is the deterministic local pipeline: it stages files, parses
them into documents and splits those into chunks. ``ingest_and_index``
extends it with the remote half — chunk embeddings via the configured
embed provider and an upsert into the configured vector store — so a
single call takes a file from raw upload to searchable.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from ....core.config.settings import AppSettings, get_settings
from ....core.logging.setup import get_logger
from ...interfaces.embed.base import BaseEmbed
from ...interfaces.ingestion.base import (
    BaseLoader,
    BaseSplitter,
    Chunk,
    Document,
    IngestionResult,
    SourceFile,
)
from ...interfaces.vector_store.base import BaseVectorStore, VectorDocument
from .factory import build_loader, build_splitter
from .upload import stage_upload

# Chunks embedded per vector-store upsert round.
_INDEX_BATCH_SIZE = 16


class IngestionService:
    """Facade over the ingestion pipeline; embed/index parts are optional."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        embed: BaseEmbed | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embed = embed
        self._vector_store = vector_store
        self._log = get_logger().bind(service="ingestion")

    # ---- pipeline stages -----------------------------------------------------

    def upload(
        self,
        source: str | Path | bytes,
        *,
        filename: str | None = None,
    ) -> list[SourceFile]:
        """Stage files for ingestion; see :func:`stage_upload` for inputs."""
        return stage_upload(source, filename=filename, settings=self._settings)

    def load(self, files: list[SourceFile]) -> list[Document]:
        """Parse staged files into documents using the type-matched loader."""
        documents: list[Document] = []
        for file in files:
            loader = self._loader_for(file)
            documents.extend(loader.load(file))
        self._log.info("load_stage_done", files=len(files), documents=len(documents))
        return documents

    def split(self, documents: list[Document]) -> list[Chunk]:
        """Turn documents into retrieval-ready chunks."""
        chunks = self._splitter().split(documents)
        self._log.info("split_stage_done", documents=len(documents), chunks=len(chunks))
        return chunks

    def ingest(
        self,
        source: str | Path | bytes,
        *,
        filename: str | None = None,
    ) -> IngestionResult:
        """Run upload -> load -> split and return everything produced."""
        files = self.upload(source, filename=filename)
        documents = self.load(files)
        chunks = self.split(documents)
        self._log.info(
            "ingestion_complete",
            files=len(files),
            documents=len(documents),
            chunks=len(chunks),
        )
        return IngestionResult(files=files, documents=documents, chunks=chunks, warnings=[])

    # ---- indexing ------------------------------------------------------------

    async def ingest_and_index(
        self,
        source: str | Path | bytes,
        *,
        filename: str | None = None,
    ) -> IngestionResult:
        """Run the full pipeline and index the chunks into the vector store.

        Chunk IDs are converted to dashed UUIDs (Qdrant point IDs must be
        UUIDs or integers); metadata stays in the payload unchanged.
        """
        result = self.ingest(source, filename=filename)
        if not result["chunks"]:
            self._log.warning("index_skipped", reason="no chunks produced")
            return result

        vector_store = self._vector_store or self._build_vector_store()
        embed = self._embed or self._build_embed()

        total = len(result["chunks"])
        for start in range(0, total, _INDEX_BATCH_SIZE):
            batch = result["chunks"][start : start + _INDEX_BATCH_SIZE]
            vector_documents = [await self._embed_chunk(embed, chunk) for chunk in batch]
            await vector_store.upsert(vector_documents)
            self._log.info(
                "index_batch_upserted",
                upserted=len(batch),
                total=total,
            )

        return result

    async def _embed_chunk(self, embed: BaseEmbed, chunk: Chunk) -> VectorDocument:
        """Embed one chunk and shape it as a vector-store document."""
        embedding = await embed.generate_embedding({"text": chunk["text"]})
        return VectorDocument(
            id=str(uuid.UUID(hex=chunk["id"])),
            text=chunk["text"],
            embedding=embedding,
            metadata=chunk["metadata"],
        )

    def _build_embed(self) -> BaseEmbed:
        # Imported lazily: provider adapters pull heavy SDKs; the pure
        # load/split pipeline must not require any API keys.
        from ..embed.factory import build_embed

        return build_embed(self._settings)

    def _build_vector_store(self) -> BaseVectorStore:
        from ..vector_store.factory import build_vector_store

        return build_vector_store(self._settings)

    # ---- adapters ------------------------------------------------------------

    def _loader_for(self, file: SourceFile) -> BaseLoader:
        """Build the loader registered for this file's type."""
        return build_loader(file["file_type"], self._settings)

    def _splitter(self) -> BaseSplitter:
        """Build the configured chunk splitter."""
        return build_splitter(self._settings)
