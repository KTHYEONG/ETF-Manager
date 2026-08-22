"""Integration tests for the read-path seam."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl

from src.etf_manager.data.calendar import load_calendar
from src.etf_manager.data.pipeline import ingest
from src.etf_manager.data.query import load_as_of
from src.etf_manager.data.schema import Dataset, spec_for


def _prices_frame(dates: list[date], closes: list[float], ticker: str = "AAA") -> pl.DataFrame:
    spec = spec_for(Dataset.PRICES)
    n = len(dates)
    return pl.DataFrame(
        {
            "ticker": [ticker] * n,
            "date": list(dates),
            "open": [value * 0.98 for value in closes],
            "high": [value * 1.02 for value in closes],
            "low": [value * 0.97 for value in closes],
            "close": list(closes),
            "volume": [10_000] * n,
            "adjusted_close": list(closes),
            "dividend": [0.0] * n,
            "split_factor": [1.0] * n,
            "source": ["synthetic"] * n,
            "retrieved_at": [datetime(2024, 2, 1, 5, 0, tzinfo=UTC)] * n,
        },
        schema=dict(spec.columns),
    )


def test_qry_a12_load_as_of_round_trip() -> None:
    """QRY-A12-load-as-of-round-trip"""
    calendar = load_calendar("XNYS")
    raw = _prices_frame([date(2024, 1, 29), date(2024, 1, 30), date(2024, 1, 31)], [100.0, 101.0, 102.0])
    frame = ingest(raw, Dataset.PRICES)
    visible = load_as_of(frame, Dataset.PRICES, calendar.close_ts(date(2024, 1, 30)))
    assert visible.height == 2
    assert visible.get_column("date").to_list() == [date(2024, 1, 29), date(2024, 1, 30)]

    before_any = load_as_of(
        frame,
        Dataset.PRICES,
        calendar.close_ts(date(2024, 1, 29)) - timedelta(microseconds=1),
    )
    assert before_any.height == 0
