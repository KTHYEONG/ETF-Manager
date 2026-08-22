"""Integration tests for the ingest write-path seam."""

from __future__ import annotations

from datetime import date

import polars as pl

from src.etf_manager.data.pipeline import ingest
from src.etf_manager.data.pit import AVAILABLE_AT
from src.etf_manager.data.schema import Dataset

TS_DTYPE = pl.Datetime("us", "UTC")


def test_pipe_a11_ingest_stamps_availability() -> None:
    """PIPE-A11-ingest-stamps-availability"""
    raw = pl.DataFrame(
        {"date": [date(2024, 1, 31), date(2024, 2, 1)], "close": [100.0, 101.0]},
        schema={"date": pl.Date, "close": pl.Float64},
    )
    raw_snapshot = raw.clone()
    stamped = ingest(raw, Dataset.PRICES)

    assert stamped.height == raw.height
    assert set(stamped.columns) == {*raw.columns, AVAILABLE_AT}
    assert stamped.schema[AVAILABLE_AT] == TS_DTYPE
    assert raw.equals(raw_snapshot)

    broken = raw.drop("date")
    try:
        ingest(broken, Dataset.PRICES)
        raised = False
    except ValueError:
        raised = True
    assert raised is True
