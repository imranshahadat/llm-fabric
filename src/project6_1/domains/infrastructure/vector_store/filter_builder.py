"""Qdrant implementation of the metadata filter builder.

Translates normalized conditions (from `parse_filter_conditions`) into
`qdrant_client.models.Filter` objects that the vector store passes to
every search call.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from ....core.config.settings import AppSettings
from ...interfaces.metadata_filter.base import BaseMetadataFilterBuilder


class QdrantFilterBuilder(BaseMetadataFilterBuilder):
    """Builds Qdrant `models.Filter` trees from validated metadata conditions."""

    def __init__(self, settings: AppSettings) -> None:
        self._allowed_fields = settings.allowed_metadata_filter_fields

    def build(self, conditions: dict[str, Any] | None) -> models.Filter | None:
        """Return a Qdrant filter for the given conditions, or None if empty."""
        # Import lazily to avoid a circular import with the vector store package.
        from ...interfaces.metadata_filter.base import parse_filter_conditions

        normalized = parse_filter_conditions(conditions, allowed_fields=self._allowed_fields)
        if not normalized:
            return None

        must: list[models.Condition] = []
        must_not: list[models.Condition] = []
        for field, spec in normalized.items():
            op = spec["op"]
            value = spec["value"]

            if op == "eq":
                must.append(self._match(field, value))
            elif op == "neq":
                must_not.append(self._match(field, value))
            elif op == "in":
                # Qdrant matches any of the values (OR inside one field).
                must.append(models.Filter(should=[self._match(field, v) for v in value]))
            elif op == "nin":
                must_not.append(models.Filter(should=[self._match(field, v) for v in value]))
            else:
                must.append(self._range(field, op, value))

        filter_kwargs: dict[str, Any] = {}
        if must:
            filter_kwargs["must"] = must
        if must_not:
            filter_kwargs["must_not"] = must_not
        return models.Filter(**filter_kwargs)

    @staticmethod
    def _match(field: str, value: Any) -> models.FieldCondition:
        """Build an exact-match condition for one field/value pair."""
        return models.FieldCondition(key=field, match=models.MatchValue(value=value))

    @staticmethod
    def _range(field: str, op: str, value: Any) -> models.FieldCondition:
        """Build a numeric range condition (gt/gte/lt/lte) for one field."""
        range_kwargs: dict[str, float | int] = {}
        if op == "gt":
            range_kwargs["gt"] = value
        elif op == "gte":
            range_kwargs["gte"] = value
        elif op == "lt":
            range_kwargs["lt"] = value
        else:
            range_kwargs["lte"] = value
        return models.FieldCondition(key=field, range=models.Range(**range_kwargs))
