"""Integration tests for the read-path seam."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from src.etf_manager.data.calendar import load_calendar
from src.etf_manager.data.pipeline import ingest
from src.etf_manager.data.query import load_as_of
from src.etf_manager.data.schema import Dataset


def test_qry_a12_load_as_of_round_trip() -> None:
    """QRY-A12-load-as-of-round-trip"""
    calendar = load_calendar("XNYS")
    raw = pl.DataFrame(
        {
            "date": [date(2024, 1, 29), date(2024, 1, 30), date(2024, 1, 31)],
            "close": [100.0, 101.0, 102.0],
        },
        schema={"date": pl.Date, "close": pl.Float64},
    )
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
