"""Cohere reranker adapter built on LangChain's CohereRerank integration."""

from __future__ import annotations

from ....core.config.settings import AppSettings
from ....core.exceptions.llm import LLMError
from ...interfaces.reranker.base import RerankRequest, RerankResult
from ._base import BaseLangChainReranker


class CohereReranker(BaseLangChainReranker):
    """Adapter that reranks documents using Cohere's rerank API."""

    PROVIDER = "cohere"
    DEFAULT_RERANK_MODEL = "rerank-v5.0-fast"

    def __init__(self, settings: AppSettings) -> None:
        super().__init__(settings)

        try:
            from langchain_cohere import CohereRerank
        except ImportError as import_error:  # pragma: no cover
            raise LLMError(
                "langchain-cohere is not installed; run `uv sync --extra cohere`",
                provider=self.PROVIDER,
            ) from import_error

        # Pass the SecretStr straight through — CohereRerank.cohere_api_key wants SecretStr.
        self._client = CohereRerank(
            model=self._rerank_model_name,
            cohere_api_key=self._settings.cohere_api_key,
            top_n=20,  # over-fetch in _rerank_sync, then slice down to top_k
        )

    def _get_configured_model_name(self) -> str:
        """Return the Cohere rerank model name from settings, or use the default."""
        return self._settings.cohere_rerank_models or self.DEFAULT_RERANK_MODEL

    def _get_provider_api_key(self) -> str | None:
        """Return the Cohere API key as plain text, or None if unset."""
        secret = self._settings.cohere_api_key
        return secret.get_secret_value() if secret else None

    def _rerank_sync(self, request: RerankRequest, top_k: int) -> list[RerankResult]:
        """Send the query + documents to Cohere and return the top_k hits by relevance."""
        from langchain_core.documents import Document

        docs = [Document(page_content=text) for text in request["documents"]]
        compressed = self._client.compress_documents(docs, request["query"])

        results: list[RerankResult] = []
        for rank, doc in enumerate(compressed[:top_k]):
            score = doc.metadata.get("relevance_score", 0.0)
            results.append(
                {
                    "document": doc.page_content,
                    "score": float(score),
                    "index": rank,  # new rank position; 0 = most relevant
                }
            )
        return results
