"""Base interface for metadata filtering on vector searches.

Metadata filtering lets callers narrow a vector search to documents whose
payload matches simple conditions (e.g. only PDF chunks, only a given
source file, only tables). Filters are plain dicts so they stay
provider-agnostic; each vector store adapter translates them into its
native query language (Qdrant: `models.Filter`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast


class MetadataFilterError(Exception):
    """A metadata filter was invalid (unknown field, bad operator, bad value)."""

    def __init__(self, message: str, *, field: str = "") -> None:
        super().__init__(message)
        self.field = field


# Operators a caller may use, mapped to Qdrant match/range conditions.
SUPPORTED_OPERATORS: tuple[str, ...] = ("eq", "neq", "in", "nin", "gt", "gte", "lt", "lte")


class BaseMetadataFilterBuilder(ABC):
    """Base interface that all metadata filter builders must implement."""

    @abstractmethod
    def build(
        self,
        conditions: dict[str, Any],
    ) -> Any:
        """Turn `{field: {"op": ..., "value": ...}}` conditions into a native filter."""
        raise NotImplementedError


def parse_filter_conditions(
    raw: dict[str, Any] | None,
    *,
    allowed_fields: frozenset[str],
) -> dict[str, dict[str, Any]]:
    """Validate a raw filter dict and return it in normalized form.

    Accepts two shapes per field:

    * shorthand: ``{"source": "report.pdf"}`` — becomes ``eq``.
    * explicit:  ``{"page": {"op": "gte", "value": 3}}``.

    Unknown fields or operators raise :class:`MetadataFilterError` so a
    typo fails loudly at the edge instead of silently returning nothing.
    """
    if not raw:
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for field, spec in raw.items():
        if field not in allowed_fields:
            raise MetadataFilterError(
                f"filter field {field!r} is not allowed; allowed fields: {sorted(allowed_fields)}",
                field=field,
            )

        if isinstance(spec, dict):
            condition = cast("dict[str, Any]", spec)
            op = str(condition.get("op", "")).lower()
            value = condition.get("value")
            if op not in SUPPORTED_OPERATORS:
                raise MetadataFilterError(
                    f"unsupported operator {op!r} on field {field!r}; "
                    f"supported: {list(SUPPORTED_OPERATORS)}",
                    field=field,
                )
            if op in ("in", "nin") and not isinstance(value, list):
                raise MetadataFilterError(
                    f"operator {op!r} on field {field!r} requires a list value",
                    field=field,
                )
            normalized[field] = {"op": op, "value": value}
        else:
            # Shorthand {"source": "x"} means equality.
            normalized[field] = {"op": "eq", "value": spec}

    return normalized
