"""Unit tests for the Redis semantic cache (runs against fakeredis)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fakeredis import aioredis as fake_aioredis

from project6_1.core.config.settings import AppSettings
from project6_1.domains.infrastructure.cache.redis_cache import RedisSemanticCache
from project6_1.domains.interfaces.semantic_cache.base import BaseSemanticCache


def make_settings(**overrides: object) -> AppSettings:
    """Build settings with fast test defaults."""
    defaults: dict[str, object] = {
        "cache_enabled": True,
        "cache_namespace": "testcache",
        "cache_ttl_seconds": 60,
        "cache_similarity_threshold": 0.92,
        "cache_max_entries_per_scope": 4,
    }
    defaults.update(overrides)
    return AppSettings(_env_file=None, **defaults)  # type: ignore[call-arg]


@pytest.fixture
async def cache() -> AsyncIterator[RedisSemanticCache]:
    """A semantic cache wired to a fresh in-memory Redis."""
    client = fake_aioredis.FakeRedis()
    yield RedisSemanticCache(make_settings(), client)
    await client.aclose()


def unit(vector: list[float]) -> list[float]:
    """Normalize a vector so cosine similarity becomes a dot product."""
    norm = sum(v * v for v in vector) ** 0.5
    return [v / norm for v in vector]


E1 = unit([1.0, 0.0, 0.0])
E2 = unit([0.0, 1.0, 0.0])
E_SIMILAR = unit([0.999, 0.044, 0.0])  # ~0.999 cosine to E1
E_DIFFERENT = unit([0.0, 0.0, 1.0])


class TestLookup:
    async def test_empty_cache_is_miss(self, cache: RedisSemanticCache) -> None:
        assert await cache.lookup(E1) is None

    async def test_exact_match_is_hit(self, cache: RedisSemanticCache) -> None:
        await cache.store(query="q1", answer="a1", embedding=E1)
        hit = await cache.lookup(E1)
        assert hit is not None
        assert hit["answer"] == "a1"
        assert hit["similarity"] == pytest.approx(1.0, abs=1e-6)

    async def test_near_match_within_threshold_is_hit(self, cache: RedisSemanticCache) -> None:
        await cache.store(query="q1", answer="a1", embedding=E1)
        hit = await cache.lookup(E_SIMILAR)
        assert hit is not None
        assert hit["answer"] == "a1"

    async def test_below_threshold_is_miss(self, cache: RedisSemanticCache) -> None:
        await cache.store(query="q1", answer="a1", embedding=E1)
        assert await cache.lookup(E2) is None

    async def test_best_of_several_entries_wins(self, cache: RedisSemanticCache) -> None:
        await cache.store(query="q1", answer="answer-one", embedding=E1)
        await cache.store(query="q2", answer="answer-two", embedding=E2)
        hit = await cache.lookup(E_SIMILAR)
        assert hit is not None
        assert hit["answer"] == "answer-one"

    async def test_scopes_are_isolated(self, cache: RedisSemanticCache) -> None:
        await cache.store(query="q1", answer="tenant-a", embedding=E1, scope="tenant-a")
        assert await cache.lookup(E1, scope="tenant-b") is None
        assert (await cache.lookup(E1, scope="tenant-a")) is not None

    async def test_dimension_mismatch_is_skipped(self, cache: RedisSemanticCache) -> None:
        await cache.store(query="q1", answer="a1", embedding=[1.0, 0.0])
        assert await cache.lookup(E1) is None

    async def test_zero_query_vector_is_miss(self, cache: RedisSemanticCache) -> None:
        await cache.store(query="q1", answer="a1", embedding=E1)
        assert await cache.lookup([0.0, 0.0, 0.0]) is None


class TestStore:
    async def test_entries_expire(self, cache: RedisSemanticCache) -> None:
        await cache.store(query="q1", answer="a1", embedding=E1)
        client = cache._client  # pyright: ignore[reportPrivateUsage]
        keys = await cache._scan_keys("default")  # pyright: ignore[reportPrivateUsage]
        assert len(keys) == 1
        assert await client.ttl(keys[0]) > 0

    async def test_cap_evicts_oldest_ttl_entries(self) -> None:
        client = fake_aioredis.FakeRedis()
        capped = RedisSemanticCache(make_settings(cache_max_entries_per_scope=2), client)
        for i in range(4):
            await capped.store(query=f"q{i}", answer=f"a{i}", embedding=E1)
        keys = await capped._scan_keys("default")  # pyright: ignore[reportPrivateUsage]
        assert len(keys) == 2
        await client.aclose()


class TestFailureHandling:
    async def test_broken_backend_lookup_is_miss_not_crash(self) -> None:
        class ExplodingClient:
            async def scan_iter(self, *args: object, **kwargs: object) -> object:
                raise ConnectionError("redis down")
                yield  # pragma: no cover

        broken = RedisSemanticCache(
            make_settings(),
            ExplodingClient(),  # type: ignore[arg-type]
        )
        assert await broken.lookup(E1) is None

    async def test_broken_backend_store_does_not_raise(self) -> None:
        class ExplodingClient:
            async def set(self, *args: object, **kwargs: object) -> None:
                raise ConnectionError("redis down")

            async def scan_iter(self, *args: object, **kwargs: object) -> object:
                raise ConnectionError("redis down")
                yield  # pragma: no cover

        broken = RedisSemanticCache(
            make_settings(),
            ExplodingClient(),  # type: ignore[arg-type]
        )
        await broken.store(query="q", answer="a", embedding=E1)  # must not raise

    async def test_corrupt_entry_is_skipped(self, cache: RedisSemanticCache) -> None:
        client = cache._client  # pyright: ignore[reportPrivateUsage]
        await client.set(b"testcache:default:bad", b"not-json{")
        await cache.store(query="q1", answer="a1", embedding=E1)
        hit = await cache.lookup(E1)
        assert hit is not None
        assert hit["answer"] == "a1"


class TestClearAndPing:
    async def test_clear_removes_only_that_scope(self, cache: RedisSemanticCache) -> None:
        await cache.store(query="q1", answer="a1", embedding=E1, scope="a")
        await cache.store(query="q2", answer="a2", embedding=E2, scope="b")
        removed = await cache.clear(scope="a")
        assert removed == 1
        assert await cache.lookup(E1, scope="a") is None
        assert (await cache.lookup(E2, scope="b")) is not None

    async def test_ping_true_on_live_backend(self, cache: RedisSemanticCache) -> None:
        assert await cache.ping() is True

    async def test_implements_port(self, cache: RedisSemanticCache) -> None:
        assert isinstance(cache, BaseSemanticCache)
