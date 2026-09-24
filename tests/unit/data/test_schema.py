"""Unit tests for the dataset specification registry."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from src.data.schema import (
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
    assert macro.key == ("series_id", "observation_date", "release_date")
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


@pytest.mark.parametrize("scenario_id", ["SPEC-C-research-returns-schema"])
def test_spec_c_research_returns_schema(scenario_id: str) -> None:
    """SPEC-C-research-returns-schema"""
    assert set(DATASET_SPECS.keys()) == set(Dataset)

    spec = spec_for(Dataset.RESEARCH_RETURNS)
    assert spec.key == ("series_id", "date")
    assert set(spec.columns) == {"series_id", "date", "simple_return", "label", "source", "retrieved_at"}
    assert spec.columns["date"] == pl.Date()
    assert spec.columns["simple_return"] == pl.Float64()
    assert spec.observation_column == "date"
    assert spec.availability.kind is AvailabilityKind.SESSION_CLOSE
    assert spec.availability.calendar_name == "XNYS"
    assert spec.missing_policy is MissingPolicy.FAIL
    assert spec.revisable is False
    assert spec.total_return_source is TotalReturnSource.NOT_APPLICABLE
    assert spec.schema_version == "1"


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


def test_schema_after_tax_registry_covers_new_members() -> None:
    """New FX_KRW_BASE and RATES members carry the declared keys, lags, and nullables."""
    assert set(DATASET_SPECS) == set(Dataset)

    base = spec_for(Dataset.FX_KRW_BASE)
    assert base.key == ("date",)
    assert base.observation_column == "date"
    assert set(base.columns) == {"date", "usdkrw", "source", "retrieved_at"}
    assert base.availability.kind is AvailabilityKind.FIXED_LAG
    assert base.availability.lag == timedelta(hours=12)
    assert base.missing_policy is MissingPolicy.EXPLICIT_GAP
    assert base.nullable_columns == frozenset({"usdkrw"})
    assert base.revisable is False
    assert base.schema_version == "1"

    rates = spec_for(Dataset.RATES)
    assert rates.key == ("series_id", "observation_date")
    assert rates.observation_column == "observation_date"
    assert set(rates.columns) == {"series_id", "observation_date", "value", "source", "retrieved_at"}
    assert rates.availability.kind is AvailabilityKind.FIXED_LAG
    assert rates.availability.lag == timedelta(days=4)
    assert rates.missing_policy is MissingPolicy.EXPLICIT_GAP
    assert rates.nullable_columns == frozenset({"value"})
    assert rates.revisable is False
    assert rates.schema_version == "1"


def test_schema_base_rate_visible_before_same_day_us_close() -> None:
    """FX_KRW_BASE availability precedes the same-day XNYS close."""
    import polars as pl

    from src.data.calendar import load_calendar
    from src.data.pit import stamp_availability

    spec = spec_for(Dataset.FX_KRW_BASE)
    frame = pl.DataFrame(
        {
            "date": [date(2024, 7, 3)],
            "usdkrw": [1300.0],
            "source": ["ecos"],
            "retrieved_at": [datetime(2024, 7, 3, 5, 0, tzinfo=UTC)],
        },
        schema=dict(spec.columns),
    )
    stamped = stamp_availability(frame, spec, None)
    assert stamped.get_column("available_at").to_list()[0] == datetime(2024, 7, 3, 12, 0, tzinfo=UTC)
    assert stamped.get_column("available_at").to_list()[0] < load_calendar().close_ts(date(2024, 7, 3))


def test_schema_korean_only_business_day_stamps_without_calendar() -> None:
    """A Korean-only business day stamps without touching the XNYS calendar."""
    import polars as pl

    from src.data.pit import stamp_availability

    spec = spec_for(Dataset.FX_KRW_BASE)
    frame = pl.DataFrame(
        {
            "date": [date(2024, 7, 4)],
            "usdkrw": [1300.0],
            "source": ["ecos"],
            "retrieved_at": [datetime(2024, 7, 4, 5, 0, tzinfo=UTC)],
        },
        schema=dict(spec.columns),
    )
    stamped = stamp_availability(frame, spec, None)
    assert stamped.get_column("available_at").to_list()[0] == datetime(2024, 7, 4, 12, 0, tzinfo=UTC)
