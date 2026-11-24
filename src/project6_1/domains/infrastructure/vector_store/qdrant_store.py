"""Qdrant-backed vector store.

Supports dense similarity search and hybrid search (dense + sparse) via the
Query API. Requires qdrant-client >= 1.10 and a Qdrant server >= 1.10.
"""

from __future__ import annotations

from typing import Any, cast

from qdrant_client import QdrantClient, models

from ....core.config.settings import AppSettings
from ....core.logging.setup import get_logger
from ...interfaces.vector_store.base import (
    BaseVectorStore,
    SearchQuery,
    SearchResult,
    VectorDocument,
)


class QdrantVectorStore(BaseVectorStore):
    """Vector store backed by Qdrant. Owns collection setup, upserts and search."""

    PROVIDER = "qdrant"

    # Named sparse vector used for the sparse half of hybrid search.
    SPARSE_VECTOR_NAME = "text_sparse"

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._log = get_logger().bind(provider=self.PROVIDER)

        api_key: str | None = (
            settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        )

        self._client = QdrantClient(url=settings.qdrant_url, api_key=api_key)
        self._collection_name = settings.qdrant_collection_name
        self._vector_size = settings.qdrant_vector_size

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection (one unnamed dense vector + one sparse vector) if missing."""
        if self._client.collection_exists(self._collection_name):
            return

        self._log.info(
            "creating_qdrant_collection",
            collection=self._collection_name,
            vector_size=self._vector_size,
        )
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=self._vector_size,
                distance=models.Distance.COSINE,
            ),
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                ),
            },
        )

    async def upsert(self, documents: list[VectorDocument]) -> None:
        """Insert or overwrite documents.

        Optional hybrid search: pass metadata["sparse_embedding"] as
        {"indices": [...], "values": [...]}. It's stored as a named sparse
        vector, not in the payload.
        """
        points: list[models.PointStruct] = []
        for doc in documents:
            metadata = doc["metadata"]

            # "" is the key Qdrant uses for the unnamed (default) dense vector.
            vector: dict[str, Any] = {"": doc["embedding"]}
            payload: dict[str, Any] = {
                "text": doc["text"],
                **{k: v for k, v in metadata.items() if k != "sparse_embedding"},
            }

            sparse = cast("dict[str, Any] | None", metadata.get("sparse_embedding"))
            if sparse is not None:
                vector[self.SPARSE_VECTOR_NAME] = models.SparseVector(
                    indices=sparse["indices"],
                    values=sparse["values"],
                )

            points.append(models.PointStruct(id=doc["id"], vector=vector, payload=payload))

        self._client.upsert(collection_name=self._collection_name, points=points)
        self._log.info(
            "upserted_documents",
            count=len(documents),
            collection=self._collection_name,
        )

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Return the best matches for a query embedding.

        Pass query["sparse_query_embedding"] as {"indices": [...], "values": [...]}
        to run hybrid search. Fused scores are RRF ranks, not cosine similarity.
        """
        search_filter = self._build_filter(query.get("filter"))
        sparse_query = cast("dict[str, Any] | None", query.get("sparse_query_embedding"))

        if sparse_query is not None:
            # Hybrid: fetch candidates from both vectors, then fuse ranks.
            # Filters must be applied inside each prefetch leg.
            response = self._client.query_points(
                collection_name=self._collection_name,
                prefetch=[
                    # Omitting "using" targets the unnamed dense vector.
                    models.Prefetch(
                        query=query["query_embedding"],
                        limit=query["top_k"],
                        filter=search_filter,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_query["indices"],
                            values=sparse_query["values"],
                        ),
                        using=self.SPARSE_VECTOR_NAME,
                        limit=query["top_k"],
                        filter=search_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=query["top_k"],
                with_payload=True,
            )
        else:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=query["query_embedding"],
                limit=query["top_k"],
                query_filter=search_filter,
                with_payload=True,
            )

        search_results: list[SearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            search_results.append(
                SearchResult(
                    id=str(point.id),
                    score=float(point.score),
                    text=str(payload.get("text") or ""),
                    metadata={k: v for k, v in payload.items() if k != "text"},
                )
            )

        self._log.info(
            "vector_search",
            results_count=len(search_results),
            collection=self._collection_name,
        )
        return search_results

    def _build_filter(self, raw: dict[str, Any] | None) -> models.Filter | None:
        """Convert a simple {field: value} mapping into a Qdrant filter."""
        if not raw:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
                for key, value in raw.items()
            ]
        )

    async def delete(self, ids: list[str]) -> None:
        """Remove documents by ID."""
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(
                points=cast("list[models.ExtendedPointId]", ids),
            ),
        )
        self._log.info(
            "deleted_documents",
            count=len(ids),
            collection=self._collection_name,
        )
