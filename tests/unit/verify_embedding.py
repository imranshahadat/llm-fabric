from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Make src/ importable when the script is run from the project root.
_SRC: Path = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# The dummy text used when no command-line argument is supplied.
DEFAULT_TEXT: str = "The quick brown fox jumps over the lazy dog."


async def _get_embedding(text: str) -> list[float]:
    """Build the adapter and request an embedding for ``text``."""
    from project6_1.domains.infrastructure.embed.factory import build_embed

    adapter = build_embed()
    return await adapter.generate_embedding({"text": text})


def main() -> int:
    """Entry point — returns 0 on success, 1 on failure."""
    text: str = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT

    print(f"Text:     {text!r}")
    print("Calling Gemini embedding API...")

    try:
        vector: list[float] = asyncio.run(_get_embedding(text))
    except Exception as exc:  # noqa: BLE001 — we want to print any failure
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        return 1

    print("\nSUCCESS")
    print(f"  Dimension:  {len(vector)}")
    print(f"  First 5:    {vector[:5]}")
    print(f"  All floats: {all(isinstance(v, float) for v in vector)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())