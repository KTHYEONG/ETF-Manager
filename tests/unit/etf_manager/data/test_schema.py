"""Unit tests for the dataset specification registry."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from src.etf_manager.data.schema import (
    DATASET_SPECS,
    AvailabilityKind,
    Dataset,
    TotalReturnSource,
    spec_for,
)


def test_spec_a01_registry_completeness() -> None:
    """SPEC-A01-registry-completeness"""
    assert set(DATASET_SPECS.keys()) == set(Dataset)
    for member in Dataset:
        spec = spec_for(member)
        assert len(spec.key) > 0
        assert all(column in spec.columns for column in spec.key)
        assert spec.observation_column in spec.columns
        rule = spec.availability
        if rule.kind is AvailabilityKind.RELEASE_COLUMN:
            assert rule.release_column is not None
            assert rule.release_column in spec.columns
            assert spec.revisable is True
        elif rule.kind is AvailabilityKind.FIXED_LAG:
            assert rule.lag is not None
            assert rule.lag > timedelta(0)
        elif rule.kind is AvailabilityKind.SESSION_CLOSE:
            assert rule.calendar_name is not None
        else:
            raise AssertionError(f"unhandled availability kind: {rule.kind}")
    prices = spec_for(Dataset.PRICES)
    assert prices.total_return_source is not TotalReturnSource.NOT_APPLICABLE
    assert all(isinstance(dtype, pl.DataType) for dtype in prices.columns.values())
