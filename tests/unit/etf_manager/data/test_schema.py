"""Unit tests for the dataset specification registry."""

from __future__ import annotations

from datetime import timedelta

import polars as pl
import pytest

from src.etf_manager.data.schema import (
    DATASET_SPECS,
    AvailabilityKind,
    Dataset,
    MissingPolicy,
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


@pytest.mark.parametrize("scenario_id", ["SPEC-C09-registry-cpi-macro-key"])
def test_spec_c09_registry_cpi_macro_key(scenario_id: str) -> None:
    """SPEC-C09-registry-cpi-macro-key"""
    assert set(DATASET_SPECS) == set(Dataset)
    assert Dataset.CPI in DATASET_SPECS

    macro = spec_for(Dataset.MACRO)
    assert macro.key == ("series_id", "observation_date")
    assert "series_id" in macro.columns
    assert macro.revisable is True

    cpi = spec_for(Dataset.CPI)
    assert cpi.availability.kind is AvailabilityKind.FIXED_LAG
    assert cpi.availability.lag == timedelta(days=45)
    assert cpi.key == ("period_end",)
    assert cpi.missing_policy is MissingPolicy.EXPLICIT_GAP
    assert cpi.revisable is False
    assert cpi.nullable_columns == frozenset({"value"})
    assert set(cpi.columns) == {"period_end", "value", "source", "retrieved_at"}

    fx = spec_for(Dataset.FX)
    assert fx.missing_policy is MissingPolicy.EXPLICIT_GAP
    assert "usdkrw" in fx.nullable_columns


@pytest.mark.parametrize("scenario_id", ["SPEC-M01-etf-metadata-schema"])
def test_spec_m01_etf_metadata_schema(scenario_id: str) -> None:
    """SPEC-M01-etf-metadata-schema"""
    spec = spec_for(Dataset.ETF_METADATA)
    assert spec.key == ("ticker", "effective_date")
    assert spec.schema_version == "2"
    assert {"expense_ratio", "aum_usd", "avg_dollar_volume", "sleeve", "is_leveraged", "is_inverse", "inception_date"} <= set(spec.columns)
    assert spec.availability.kind is AvailabilityKind.RELEASE_COLUMN
    assert spec.availability.release_column == "filing_date"
    assert spec.missing_policy is MissingPolicy.FAIL
    assert spec.revisable is True
    assert spec.observation_column == "effective_date"
    assert spec.columns["is_leveraged"] == pl.Int64()
    assert spec.columns["is_inverse"] == pl.Int64()
