"""Structured logging setup using structlog (JSON for production, colored console for dev)."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_LOGGER_NAME = "project6_1.llm"
_CONFIGURED = False


def configure_logging(
    *,
    level: str = "INFO",
    json_renderer: bool = True,
    stream: Any = None,
) -> None:
    """Configure structlog once per process with JSON or console formatting."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True  # pyright: ignore[reportConstantRedefinition]

    logging.basicConfig(
        stream=stream or sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        force=True,
    )

    renderer: Any
    if json_renderer:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=stream or sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = _LOGGER_NAME) -> Any:
    """Return a bound structlog logger, auto-configuring logging if not yet set up."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name).bind()


def reset_logging_for_tests() -> None:
    """Reset the setup flag so tests can reconfigure logging cleanly."""
    global _CONFIGURED
    _CONFIGURED = False  # pyright: ignore[reportConstantRedefinition]
