"""Unit tests for PIT ETF hard filters and scores."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Final

import polars as pl
import pytest

from src.etf_manager.data.calendar import load_calendar
from src.etf_manager.data.pipeline import ingest
from src.etf_manager.data.schema import Dataset, spec_for
from src.etf_manager.etf.mapping import MappingConfig
from src.etf_manager.etf.score import etf_score, passes_hard_filters

_CALENDAR = load_calendar("XNYS")
_RETRIEVED_AT = datetime(2023, 11, 1, 5, 0, tzinfo=UTC)
_SIGNAL_AT = _CALENDAR.close_ts(date(2024, 1, 30))
_AVG_DOLLAR_VOLUME: Final[float] = 4e12


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


def _vt_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "ticker": "VT",
        "effective_date": date(2023, 11, 15),
        "filing_date": datetime(2023, 11, 20, tzinfo=UTC),
        "sleeve": "VT",
        "expense_ratio": 0.001,
        "aum_usd": 4e10,
        "avg_dollar_volume": _AVG_DOLLAR_VOLUME,
        "is_leveraged": 0,
        "is_inverse": 0,
        "inception_date": date(2008, 6, 12),
        "source": "synthetic",
        "retrieved_at": _RETRIEVED_AT,
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize("scenario_id", ["ETF-M02-score-pit-and-expense"])
def test_etf_m02_score_pit_and_expense(scenario_id: str) -> None:
    """ETF-M02-score-pit-and-expense"""
    days = tuple(_CALENDAR.sessions(date(2023, 12, 1), date(2024, 1, 31)))
    prices = _constant_prices(("VT",), days)
    # The cheaper filing (expense 0.0002) is published after the signal instant.
    metadata = _etf_metadata(
        [
            _vt_row(expense_ratio=0.001),
            _vt_row(
                expense_ratio=0.0002,
                effective_date=date(2023, 12, 1),
                filing_date=datetime(2024, 6, 3, tzinfo=UTC),
            ),
        ]
    )
    mapping = MappingConfig()

    score = etf_score(prices, metadata, ticker="VT", sleeve="VT", signal_at=_SIGNAL_AT, mapping=mapping)

    spread = 1.0 / (_AVG_DOLLAR_VOLUME**0.5)
    assert score == pytest.approx(1 - 0.001 - spread, rel=1e-9)

    leveraged_row = _vt_row(is_leveraged=1)
    assert not passes_hard_filters(leveraged_row, sleeve="VT", signal_at=_SIGNAL_AT, mapping=mapping)

    with pytest.raises(ValueError, match="timezone-aware"):
        etf_score(
            prices,
            metadata,
            ticker="VT",
            sleeve="VT",
            signal_at=datetime(2024, 1, 30),
            mapping=mapping,
        )
