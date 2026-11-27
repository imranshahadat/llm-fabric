"""Unit tests for metadata filter parsing and the Qdrant filter builder."""

from __future__ import annotations

import pytest
from qdrant_client import models

from project6_1.core.config.settings import AppSettings
from project6_1.domains.infrastructure.vector_store.filter_builder import QdrantFilterBuilder
from project6_1.domains.interfaces.metadata_filter.base import (
    MetadataFilterError,
    parse_filter_conditions,
)

ALLOWED = frozenset({"source", "file_type", "kind", "page"})


def make_settings(**overrides: object) -> AppSettings:
    """Build settings with test-friendly defaults."""
    return AppSettings(
        _env_file=None,  # type: ignore[call-arg]
        metadata_filter_fields="source,file_type,kind,page",
        **overrides,  # type: ignore[arg-type]
    )


class TestParseFilterConditions:
    def test_empty_filter_returns_empty(self) -> None:
        assert parse_filter_conditions(None, allowed_fields=ALLOWED) == {}
        assert parse_filter_conditions({}, allowed_fields=ALLOWED) == {}

    def test_shorthand_becomes_eq(self) -> None:
        result = parse_filter_conditions({"source": "report.pdf"}, allowed_fields=ALLOWED)
        assert result == {"source": {"op": "eq", "value": "report.pdf"}}

    def test_explicit_operator_is_kept(self) -> None:
        result = parse_filter_conditions(
            {"page": {"op": "gte", "value": 3}}, allowed_fields=ALLOWED
        )
        assert result == {"page": {"op": "gte", "value": 3}}

    def test_unknown_field_is_rejected(self) -> None:
        with pytest.raises(MetadataFilterError, match="not allowed"):
            parse_filter_conditions({"secret": "x"}, allowed_fields=ALLOWED)

    def test_unknown_operator_is_rejected(self) -> None:
        with pytest.raises(MetadataFilterError, match="unsupported operator"):
            parse_filter_conditions({"page": {"op": "regex", "value": "3"}}, allowed_fields=ALLOWED)

    def test_in_requires_list(self) -> None:
        with pytest.raises(MetadataFilterError, match="requires a list"):
            parse_filter_conditions(
                {"kind": {"op": "in", "value": "table"}}, allowed_fields=ALLOWED
            )

    def test_error_carries_field_name(self) -> None:
        with pytest.raises(MetadataFilterError) as exc_info:
            parse_filter_conditions({"nope": 1}, allowed_fields=ALLOWED)
        assert exc_info.value.field == "nope"


class TestQdrantFilterBuilder:
    def build(self, conditions: dict[str, object] | None) -> models.Filter | None:
        builder = QdrantFilterBuilder(make_settings())
        return builder.build(conditions)  # type: ignore[arg-type]

    @staticmethod
    def conditions_of(result: models.Filter, *, negate: bool = False) -> list[models.Condition]:
        """Pull the condition list out of a Qdrant Filter (must or must_not)."""
        raw = result.must_not if negate else result.must
        assert isinstance(raw, list), f"expected a condition list, got {raw!r}"
        return raw

    def test_no_conditions_yields_none(self) -> None:
        assert self.build(None) is None
        assert self.build({}) is None

    def test_eq_builds_must_match(self) -> None:
        result = self.build({"source": "report.pdf"})
        assert result is not None
        conditions = self.conditions_of(result)
        assert len(conditions) == 1
        condition = conditions[0]
        assert isinstance(condition, models.FieldCondition)
        assert condition.key == "source"
        assert isinstance(condition.match, models.MatchValue)
        assert condition.match.value == "report.pdf"

    def test_neq_builds_must_not(self) -> None:
        result = self.build({"kind": {"op": "neq", "value": "image"}})
        assert result is not None
        assert len(self.conditions_of(result, negate=True)) == 1

    def test_in_builds_should_filter(self) -> None:
        result = self.build({"kind": {"op": "in", "value": ["table", "text"]}})
        assert result is not None
        inner = self.conditions_of(result)[0]
        assert isinstance(inner, models.Filter)
        should = inner.should
        assert isinstance(should, list) and len(should) == 2

    def test_range_builds_numeric_range(self) -> None:
        result = self.build({"page": {"op": "gte", "value": 3}})
        assert result is not None
        condition = self.conditions_of(result)[0]
        assert isinstance(condition, models.FieldCondition)
        assert condition.range is not None
        assert condition.range.gte == 3

    def test_disallowed_field_raises(self) -> None:
        with pytest.raises(MetadataFilterError):
            self.build({"tenant": "acme"})

    def test_settings_allowlist_off_disallows_everything(self) -> None:
        builder = QdrantFilterBuilder(
            AppSettings(_env_file=None, metadata_filter_fields="")  # type: ignore[call-arg]
        )
        with pytest.raises(MetadataFilterError):
            builder.build({"source": "report.pdf"})

    def test_backward_compatible_shorthand_via_store_path(self) -> None:
        """The old simple {field: value} shape must keep working."""
        result = self.build({"file_type": "pdf"})
        assert result is not None
        condition = self.conditions_of(result)[0]
        assert isinstance(condition, models.FieldCondition)
        assert condition.key == "file_type"
