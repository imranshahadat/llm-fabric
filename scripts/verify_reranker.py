"""End-to-end smoke test for the reranker adapter."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_SRC: Path = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from project6_1.domains.infrastructure.reranker.factory import build_reranker  # noqa: E402
from project6_1.domains.interfaces.reranker.base import RerankResult  # noqa: E402

_QUERY: str = "What is the capital of France?"
_DOCUMENTS: list[str] = [
    "Paris is the capital of France.",
    "The Eiffel Tower is in Tokyo.",
    "France borders Germany and Spain.",
    "Croissants are a popular French pastry.",
    "Python is a programming language.",
]


async def _verify() -> list[RerankResult]:
    """Call the configured reranker on a small fixed query and return the top-3 hits."""
    reranker = build_reranker()
    return await reranker.rerank(
        {"query": _QUERY, "documents": _DOCUMENTS},
        top_k=3,
    )


def main() -> int:
    """Run the smoke test and print results; return 0 on success, 1 on failure."""
    print("[1/1] Calling Cohere RERANK API...")
    try:
        results: list[RerankResult] = asyncio.run(_verify())
        print("  SUCCESS")
        for r in results:
            print(f"  #{r['index']}  score={r['score']:.4f}  text={r['document']!r}")
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
