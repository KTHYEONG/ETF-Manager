"""Unit tests for hysteresis ETF mapping."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Final

import polars as pl
import pytest

from src.etf_manager.data.calendar import load_calendar
from src.etf_manager.data.pipeline import ingest
from src.etf_manager.data.schema import Dataset, spec_for
from src.etf_manager.etf.mapping import MappingConfig, apply_etf_mapping

_CALENDAR = load_calendar("XNYS")
_RETRIEVED_AT = datetime(2023, 11, 1, 5, 0, tzinfo=UTC)
_AUM_USD: Final[float] = 2e11
_AVG_DOLLAR_VOLUME: Final[float] = 4e10


def _constant_prices(tickers: tuple[str, ...], days: tuple[date, ...]) -> pl.DataFrame:
    """Constant 100.0 adjusted closes per (ticker, day) pair."""
    spec = spec_for(Dataset.PRICES)
    flat: dict[str, list[object]] = {
        "ticker": [],
        "date": [],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [],
        "adjusted_close": [],
        "dividend": [],
        "split_factor": [],
        "source": [],
        "retrieved_at": [],
    }
    for ticker in tickers:
        for day in days:
            flat["ticker"].append(ticker)
            flat["date"].append(day)
            flat["open"].append(98.0)
            flat["high"].append(102.0)
            flat["low"].append(97.0)
            flat["close"].append(100.0)
            flat["volume"].append(10_000)
            flat["adjusted_close"].append(100.0)
            flat["dividend"].append(0.0)
            flat["split_factor"].append(1.0)
            flat["source"].append("synthetic")
            flat["retrieved_at"].append(_RETRIEVED_AT)
    return ingest(pl.DataFrame(flat, schema=dict(spec.columns)), Dataset.PRICES)


def _etf_metadata(rows: list[dict[str, object]]) -> pl.DataFrame:
    """Availability-stamped ETF_METADATA frame from raw filing rows."""
    spec = spec_for(Dataset.ETF_METADATA)
    columns = {name: [row[name] for row in rows] for name in spec.columns}
    return ingest(pl.DataFrame(columns, schema=dict(spec.columns)), Dataset.ETF_METADATA)


def _candidate_row(*, ticker: str, sleeve: str, expense_ratio: float, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "ticker": ticker,
        "effective_date": date(2023, 12, 1),
        "filing_date": datetime(2023, 12, 5, tzinfo=UTC),
        "sleeve": sleeve,
        "expense_ratio": expense_ratio,
        "aum_usd": _AUM_USD,
        "avg_dollar_volume": _AVG_DOLLAR_VOLUME,
        "is_leveraged": 0,
        "is_inverse": 0,
        "inception_date": date(2010, 1, 4),
        "source": "synthetic",
        "retrieved_at": _RETRIEVED_AT,
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize("scenario_id", ["ETF-M03-hysteresis"])
def test_etf_m03_hysteresis(scenario_id: str) -> None:
    """ETF-M03-hysteresis"""
    days = tuple(_CALENDAR.sessions(date(2024, 1, 2), date(2024, 1, 31)))
    prices = _constant_prices(("VTI", "ITOT"), days)
    signal_at = _CALENDAR.close_ts(days[-1])
    mapping = MappingConfig(min_improvement=0.02, fit_window=5, td_window=5)
    weights = {"VTI": 0.5}

    sticky_metadata = _etf_metadata(
        [
            _candidate_row(ticker="VTI", sleeve="VTI", expense_ratio=0.0011),
            _candidate_row(ticker="ITOT", sleeve="VTI", expense_ratio=0.0001),
        ]
    )
    mapped, incumbents = apply_etf_mapping(
        weights, prices, sticky_metadata, signal_at, mapping, {"VTI": "VTI"}
    )
    # Challenger is cheaper by 0.001 < min_improvement, so the incumbent stays.
    assert mapped == {"VTI": pytest.approx(0.5)}
    assert incumbents == {"VTI": "VTI"}

    switch_metadata = _etf_metadata(
        [
            _candidate_row(ticker="VTI", sleeve="VTI", expense_ratio=0.0501),
            _candidate_row(ticker="ITOT", sleeve="VTI", expense_ratio=0.0001),
        ]
    )
    mapped, incumbents = apply_etf_mapping(
        weights, prices, switch_metadata, signal_at, mapping, {"VTI": "VTI"}
    )
    # Challenger is cheaper by 0.05 >= min_improvement, so the incumbent switches.
    assert mapped == {"ITOT": pytest.approx(0.5)}
    assert incumbents == {"VTI": "ITOT"}
