"""Supplementary unit tests for trailing-vol input guards."""

from __future__ import annotations

import math
import statistics
from datetime import date, datetime

import polars as pl
import pytest

from src.data.calendar import load_calendar
from src.features.risk import trailing_simple_vol

_CALENDAR = load_calendar("XNYS")


def _returns_frame(values: list[float]) -> pl.DataFrame:
    days = _CALENDAR.sessions(date(2024, 1, 2), date(2024, 1, 31))[: len(values) + 1][1:]
    return pl.DataFrame(
        {
            "date": list(days),
            "ticker": ["TEST"] * len(values),
            "simple_return": values,
            "available_at": [_CALENDAR.close_ts(day) for day in days],
        },
        schema={
            "date": pl.Date,
            "ticker": pl.String,
            "simple_return": pl.Float64,
            "available_at": pl.Datetime("us", "UTC"),
        },
    )


def test_trailing_vol_rejects_naive_as_of() -> None:
    """A naive as_of_ts must fail closed instead of silently comparing wall clocks."""
    with pytest.raises(ValueError, match="timezone-aware"):
        trailing_simple_vol(_returns_frame([0.01, 0.02]), as_of_ts=datetime(2024, 1, 10), window=2)


def test_trailing_vol_rejects_degenerate_window() -> None:
    """Sample stdev needs at least two observations."""
    aware = _CALENDAR.close_ts(date(2024, 1, 10))
    with pytest.raises(ValueError, match="window"):
        trailing_simple_vol(_returns_frame([0.01, 0.02]), as_of_ts=aware, window=1)


def test_trailing_vol_rejects_non_finite_returns() -> None:
    """NaN or inf inside the visible window must raise, never poison the stdev."""
    aware = _CALENDAR.close_ts(date(2024, 1, 31))
    with pytest.raises(ValueError, match="finite"):
        trailing_simple_vol(_returns_frame([0.01, math.nan, 0.03]), as_of_ts=aware, window=3)
    with pytest.raises(ValueError, match="finite"):
        trailing_simple_vol(_returns_frame([0.01, math.inf, 0.03]), as_of_ts=aware, window=3)


def test_trailing_vol_zero_stdev_fails_closed() -> None:
    """Constant returns give zero dispersion; inverse-vol weighting would divide by it."""
    aware = _CALENDAR.close_ts(date(2024, 1, 31))
    with pytest.raises(ValueError, match="positive"):
        trailing_simple_vol(_returns_frame([0.02, 0.02, 0.02]), as_of_ts=aware, window=3)


def test_trailing_vol_two_point_window() -> None:
    """window=2 is the minimal valid sample-stdev window."""
    values = [0.04, -0.02]
    sigma = trailing_simple_vol(
        _returns_frame(values),
        as_of_ts=_CALENDAR.close_ts(date(2024, 1, 31)),
        window=2,
    )
    assert sigma == pytest.approx(statistics.stdev(values))
    assert sigma > 0.0
