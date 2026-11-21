"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add src to path for imports
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Reset settings cache before each test to avoid cross-test contamination."""
    from project6_1.core.config.settings import reset_settings_cache as reset

    reset()
    yield
    reset()
