"""Redis-backed semantic cache.

Stores query/answer pairs in Redis (self-hosted or Redis Cloud — point
REDIS_URL at your Cloud endpoint, e.g.
``rediss://default:<password>@<host>:<port>``).

Design notes, kept simple on purpose:

* One Redis key per entry: ``{namespace}:{scope}:{uuid}``, value is a
  JSON blob (query, answer, embedding), expiry = CACHE_TTL_SECONDS.
* Lookup scans the scope's keys and compares embeddings in-process with
  a numpy cosine similarity. Avoids the RediSearch module so it works on
  every Redis plan, including the free Redis Cloud tier.
* Redis is treated as an optimization, never a dependency: any Redis
  error is logged and turned into a miss so the request still succeeds.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

import numpy as np
import redis.asyncio as aioredis

from ....core.config.settings import AppSettings
from ....core.logging.setup import get_logger
from ...interfaces.semantic_cache.base import BaseSemanticCache, CacheHit


class RedisSemanticCache(BaseSemanticCache):
    """Semantic cache backed by Redis strings, scoped and TTL-limited."""

    PROVIDER = "redis"

    def __init__(self, settings: AppSettings, client: aioredis.Redis | None = None) -> None:
        self._settings = settings
        self._log = get_logger().bind(component="semantic_cache", provider=self.PROVIDER)
        # An injected client (tests, custom pools) wins over building one.
        self._client = client or aioredis.from_url(
            settings.redis_url,
            decode_responses=False,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )

    def _scope_pattern(self, scope: str) -> str:
        """Return the SCAN pattern that covers every key in one scope."""
        return f"{self._settings.cache_namespace}:{scope}:*"

    async def _scan_keys(self, scope: str) -> list[bytes]:
        """Collect every cache key in one scope as plain bytes."""
        # redis-py ships partial type stubs: cast the iterator once so the
        # rest of the file works with concrete `bytes` keys.
        iterator = cast(
            "AsyncIterator[bytes]",
            self._client.scan_iter(match=self._scope_pattern(scope)),  # pyright: ignore[reportUnknownMemberType]
        )
        return [key async for key in iterator]

    async def lookup(self, embedding: list[float], *, scope: str = "default") -> CacheHit | None:
        """Return the best cached answer above the similarity threshold, else None."""
        try:
            return await self._lookup_inner(embedding, scope=scope)
        except Exception as exc:
            self._log.warning("cache_lookup_failed", error=f"{type(exc).__name__}: {exc}")
            return None

    async def _lookup_inner(self, embedding: list[float], *, scope: str) -> CacheHit | None:
        """Scan the scope, score every stored embedding, keep the best match."""
        keys = await self._scan_keys(scope)
        if not keys:
            return None

        raw_entries = cast(
            "list[bytes | None]",
            await self._client.mget(keys),
        )
        query_vector = np.asarray(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:  # a zero vector can't be scored
            return None

        best_similarity = -1.0
        best: CacheHit | None = None
        for key, raw in zip(keys, raw_entries, strict=True):
            if raw is None:
                continue
            entry = self._decode(raw, key=key)
            if entry is None or len(entry["embedding"]) != len(embedding):
                continue

            stored_vector = np.asarray(entry["embedding"], dtype=np.float32)
            stored_norm = np.linalg.norm(stored_vector)
            if stored_norm == 0:
                continue

            similarity = float(np.dot(query_vector, stored_vector) / (query_norm * stored_norm))
            if similarity > best_similarity:
                best_similarity = similarity
                best = CacheHit(
                    answer=entry["answer"],
                    similarity=similarity,
                    cache_key=key.decode("utf-8", errors="replace"),
                )

        if best is None or best_similarity < self._settings.cache_similarity_threshold:
            return None

        self._log.info(
            "cache_hit",
            scope=scope,
            similarity=round(best_similarity, 4),
        )
        return best

    async def store(
        self,
        *,
        query: str,
        answer: str,
        embedding: list[float],
        scope: str = "default",
    ) -> None:
        """Cache one query/answer pair with the configured TTL."""
        payload = json.dumps(
            {"query": query, "answer": answer, "embedding": embedding},
            separators=(",", ":"),
        )
        key = f"{self._settings.cache_namespace}:{scope}:{uuid.uuid4().hex}"

        try:
            await self._client.set(key, payload, ex=self._settings.cache_ttl_seconds)
            await self._enforce_scope_cap(scope)
        except Exception as exc:
            self._log.warning("cache_store_failed", error=f"{type(exc).__name__}: {exc}")

    async def _enforce_scope_cap(self, scope: str) -> None:
        """Evict closest-to-expiry entries when a scope exceeds its cap."""
        keys = await self._scan_keys(scope)
        excess = len(keys) - self._settings.cache_max_entries_per_scope
        if excess <= 0:
            return

        # Lowest TTL expires first anyway — evict those, they'd vanish soonest.
        ttl_map: list[tuple[bytes, int]] = []
        for key in keys:
            ttl = await self._client.ttl(key)
            ttl_map.append((key, int(ttl)))
        ttl_map.sort(key=lambda item: item[1])

        evict = [key for key, _ in ttl_map[:excess]]
        if evict:
            await self._client.delete(*evict)
            self._log.info("cache_evicted", scope=scope, count=len(evict))

    async def clear(self, scope: str = "default") -> int:
        """Delete every entry in a scope. Returns the number removed."""
        try:
            keys = await self._scan_keys(scope)
            if not keys:
                return 0
            removed = int(await self._client.delete(*keys))
            self._log.info("cache_cleared", scope=scope, count=removed)
            return removed
        except Exception as exc:
            self._log.warning("cache_clear_failed", error=f"{type(exc).__name__}: {exc}")
            return 0

    async def ping(self) -> bool:
        """Return True if Redis answers PING."""
        try:
            pong = await self._client.ping()  # pyright: ignore[reportUnknownMemberType]
            return bool(cast("object", pong))
        except Exception as exc:
            self._log.warning("cache_ping_failed", error=f"{type(exc).__name__}: {exc}")
            return False

    def _decode(self, raw: bytes, *, key: bytes) -> dict[str, Any] | None:
        """Parse one stored JSON blob; a corrupt entry is skipped, not raised."""
        try:
            entry = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._log.warning("cache_entry_corrupt", key=key.decode("utf-8", errors="replace"))
            return None

        if not isinstance(entry, dict):
            self._log.warning("cache_entry_corrupt", key=key.decode("utf-8", errors="replace"))
            return None

        typed = cast("dict[str, Any]", entry)
        answer = typed.get("answer")
        embedding = typed.get("embedding")
        if isinstance(answer, str) and isinstance(embedding, list):
            return {"answer": answer, "embedding": embedding}

        self._log.warning("cache_entry_corrupt", key=key.decode("utf-8", errors="replace"))
        return None
