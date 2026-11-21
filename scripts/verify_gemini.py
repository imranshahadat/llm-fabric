"""End-to-end smoke test for the Gemini embed + chat adapters.

Usage:
    python scripts/verify_gemini.py
    python scripts/verify_gemini.py "your text here"

Requires a real GEMINI_API_KEY in the project .env file.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make src/ importable when the script is run from the project root.
_SRC: Path = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_TEXT: str = "The quick brown fox jumps over the lazy dog."


async def _verify_embed(text: str) -> list[float]:
    from project6_1.domains.infrastructure.embed.factory import build_embed

    adapter = build_embed()
    return await adapter.generate_embedding({"text": text})


async def _verify_chat(prompt: str) -> str:
    from project6_1.domains.infrastructure.llm.factory import build_llm

    llm = build_llm()
    messages = [{"role": "user", "content": prompt}]
    return await llm.generate(messages)


def main() -> int:
    text: str = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT

    print(f"Text:     {text!r}")
    print("-" * 60)

    # ---- Embed check ----
    print("[1/2] Calling Gemini EMBEDDING API...")
    try:
        vector = asyncio.run(_verify_embed(text))
        print("  SUCCESS")
        print(f"  Dimension:  {len(vector)}")
        print(f"  First 5:    {vector[:5]}")
        print(f"  All floats: {all(isinstance(v, float) for v in vector)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return 1

    print("-" * 60)

    # ---- Chat check ----
    print("[2/2] Calling Gemini CHAT API...")
    try:
        response = asyncio.run(_verify_chat(
            f"Reply with exactly one short sentence about: {text!r}"
        ))
        print("  SUCCESS")
        print(f"  Response:   {response!r}")
        print(f"  Length:     {len(response)} chars")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return 1

    print("-" * 60)
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())