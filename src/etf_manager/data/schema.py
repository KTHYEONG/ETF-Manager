"""Dataset specification registry: schema, availability rule, and policy declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import Final

import polars as pl

TS_DTYPE: Final[pl.Datetime] = pl.Datetime("us", "UTC")


class Dataset(StrEnum):
    """Logical dataset identities; tickers and vendor names never appear here."""

    PRICES = "prices"
    FX = "fx"
    MACRO = "macro"
    CPI = "cpi"
    FACTORS = "factors"
    ETF_METADATA = "etf_metadata"


class AvailabilityKind(StrEnum):
    """How the first moment a row may be consumed is determined."""

    SESSION_CLOSE = "session_close"
    RELEASE_COLUMN = "release_column"
    FIXED_LAG = "fixed_lag"


class MissingPolicy(StrEnum):
    """Fail-closed default; gaps are declared, never silently repaired."""

    FAIL = "fail"
    DROP = "drop"
    EXPLICIT_GAP = "explicit_gap"


class TotalReturnSource(StrEnum):
    """Which representation carries total return for a price dataset."""

    ADJUSTED_PRICE = "adjusted_price"
    RAW_PLUS_DIVIDEND = "raw_plus_dividend"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class AvailabilityRule:
    """Declarative publication-lag rule resolved into ``available_at`` at ingest."""

    kind: AvailabilityKind
    calendar_name: str | None = None
    release_column: str | None = None
    lag: timedelta | None = None


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Immutable contract of one dataset: columns, key, PIT semantics."""

    dataset: Dataset
    columns: Mapping[str, pl.DataType]
    key: tuple[str, ...]
    observation_column: str
    availability: AvailabilityRule
    missing_policy: MissingPolicy
    revisable: bool
    total_return_source: TotalReturnSource
    schema_version: str
    # Only value fields of EXPLICIT_GAP datasets may be declared nullable.
    nullable_columns: frozenset[str] = frozenset()


def _build_specs() -> dict[Dataset, DatasetSpec]:
    prices = DatasetSpec(
        dataset=Dataset.PRICES,
        columns={
            "ticker": pl.String(),
            "date": pl.Date(),
            "open": pl.Float64(),
            "high": pl.Float64(),
            "low": pl.Float64(),
            "close": pl.Float64(),
            "volume": pl.Int64(),
            "adjusted_close": pl.Float64(),
            "dividend": pl.Float64(),
            "split_factor": pl.Float64(),
            "source": pl.String(),
            "retrieved_at": TS_DTYPE,
        },
        key=("ticker", "date"),
        observation_column="date",
        availability=AvailabilityRule(kind=AvailabilityKind.SESSION_CLOSE, calendar_name="XNYS"),
        missing_policy=MissingPolicy.FAIL,
        revisable=False,
        total_return_source=TotalReturnSource.ADJUSTED_PRICE,
        schema_version="1",
    )
    fx = DatasetSpec(
        dataset=Dataset.FX,
        columns={
            "date": pl.Date(),
            "usdkrw": pl.Float64(),
            "source": pl.String(),
            "retrieved_at": TS_DTYPE,
        },
        key=("date",),
        observation_column="date",
        availability=AvailabilityRule(kind=AvailabilityKind.SESSION_CLOSE, calendar_name="XNYS"),
        # Vendor FX calendars differ from XNYS; published gap days stay as null rows.
        missing_policy=MissingPolicy.EXPLICIT_GAP,
        revisable=False,
        total_return_source=TotalReturnSource.NOT_APPLICABLE,
        schema_version="1",
        nullable_columns=frozenset({"usdkrw"}),
    )
    macro = DatasetSpec(
        dataset=Dataset.MACRO,
        columns={
            "series_id": pl.String(),
            "observation_date": pl.Date(),
            "release_date": TS_DTYPE,
            "value": pl.Float64(),
        },
        key=("series_id", "observation_date"),
        observation_column="observation_date",
        availability=AvailabilityRule(kind=AvailabilityKind.RELEASE_COLUMN, release_column="release_date"),
        missing_policy=MissingPolicy.EXPLICIT_GAP,
        revisable=True,
        total_return_source=TotalReturnSource.NOT_APPLICABLE,
        schema_version="1",
        nullable_columns=frozenset({"value"}),
    )
    cpi = DatasetSpec(
        dataset=Dataset.CPI,
        columns={
            "period_end": pl.Date(),
            "value": pl.Float64(),
            "source": pl.String(),
            "retrieved_at": TS_DTYPE,
        },
        key=("period_end",),
        observation_column="period_end",
        availability=AvailabilityRule(kind=AvailabilityKind.FIXED_LAG, lag=timedelta(days=45)),
        missing_policy=MissingPolicy.EXPLICIT_GAP,
        revisable=False,
        total_return_source=TotalReturnSource.NOT_APPLICABLE,
        schema_version="1",
        nullable_columns=frozenset({"value"}),
    )
    factors = DatasetSpec(
        dataset=Dataset.FACTORS,
        columns={
            "period_end": pl.Date(),
            "mkt_rf": pl.Float64(),
            "smb": pl.Float64(),
            "hml": pl.Float64(),
            "rmw": pl.Float64(),
            "cma": pl.Float64(),
            "mom": pl.Float64(),
            "rf": pl.Float64(),
            "source": pl.String(),
            "retrieved_at": TS_DTYPE,
        },
        key=("period_end",),
        observation_column="period_end",
        availability=AvailabilityRule(
            kind=AvailabilityKind.FIXED_LAG, lag=timedelta(days=60)
        ),
        missing_policy=MissingPolicy.EXPLICIT_GAP,
        revisable=False,
        total_return_source=TotalReturnSource.NOT_APPLICABLE,
        schema_version="1",
        nullable_columns=frozenset({"mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"}),
    )
    etf_metadata = DatasetSpec(
        dataset=Dataset.ETF_METADATA,
        columns={
            "ticker": pl.String(),
            "effective_date": pl.Date(),
            "filing_date": TS_DTYPE,
            "sleeve": pl.String(),
            "expense_ratio": pl.Float64(),
            "aum_usd": pl.Float64(),
            "avg_dollar_volume": pl.Float64(),
            "is_leveraged": pl.Int64(),
            "is_inverse": pl.Int64(),
            "inception_date": pl.Date(),
            "source": pl.String(),
            "retrieved_at": TS_DTYPE,
        },
        key=("ticker", "effective_date"),
        observation_column="effective_date",
        availability=AvailabilityRule(kind=AvailabilityKind.RELEASE_COLUMN, release_column="filing_date"),
        missing_policy=MissingPolicy.FAIL,
        revisable=True,
        total_return_source=TotalReturnSource.NOT_APPLICABLE,
        schema_version="2",
    )
    return {
        Dataset.PRICES: prices,
        Dataset.FX: fx,
        Dataset.MACRO: macro,
        Dataset.CPI: cpi,
        Dataset.FACTORS: factors,
        Dataset.ETF_METADATA: etf_metadata,
    }


DATASET_SPECS: Final[Mapping[Dataset, DatasetSpec]] = _build_specs()
del _build_specs


def spec_for(dataset: Dataset) -> DatasetSpec:
    """Resolve the immutable spec of a dataset; unknown members fail closed."""
    try:
        return DATASET_SPECS[dataset]
    except KeyError as exc:
        raise ValueError(f"no dataset spec registered for {dataset!r}") from exc
