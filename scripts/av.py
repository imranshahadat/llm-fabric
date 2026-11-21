"""Verify Gemini chat API works correctly."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from project6_1.core.config.settings import get_settings
from project6_1.domains.infrastructure.llm.factory import build_llm


async def test_gemini_chat() -> None:
    settings = get_settings()
    
    print("-" * 60)
    print("[2/2] Calling Gemini CHAT API...")
    print(f"  Provider:   {settings.llm_provider.value}")
    print(f"  Model:      {settings.gemini_model}")
    
    llm = build_llm(settings)
    
    # MUST use await here
    response = await llm.generate([
        {"role": "user", "content": "The quick brown fox jumps over the lazy dog."}
    ])
    
    print(f"  SUCCESS")
    print(f"  Response:   {response[:200]}")


def main() -> int:
    """Run verification tests."""
    try:
        asyncio.run(test_gemini_chat())
        return 0
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())