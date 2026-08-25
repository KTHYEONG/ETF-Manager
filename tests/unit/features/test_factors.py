"""Unit tests for PIT trailing OLS factor loadings."""

from __future__ import annotations

import calendar as _calendar
import math
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from src.etf_manager.data.pit import AVAILABLE_AT
from src.etf_manager.features.factors import FACTOR_COLUMNS, estimate_factor_loadings

_TICKER = "VTI"
_WINDOW = 36


def _month_ends(count: int, start_year: int = 2019, start_month: int = 1) -> list[date]:
    ends: list[date] = []
    year, month = start_year, start_month
    for _ in range(count):
        ends.append(date(year, month, _calendar.monthrange(year, month)[1]))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return ends


def _close_ts(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 21, tzinfo=UTC)


def _price_frame(months: list[date]) -> pl.DataFrame:
    count = len(months)
    return pl.DataFrame(
        {
            "ticker": [_TICKER] * count,
            "date": months,
            "adjusted_close": [100.0 + 0.5 * index for index in range(count)],
            AVAILABLE_AT: [_close_ts(month) for month in months],
        }
    )


def _factor_value(name: str, index: int) -> float:
    table = {
        "mkt_rf": 0.010 + 0.001 * math.sin(index),
        "smb": 0.002 * math.cos(index / 2.0),
        "hml": 0.003 * math.sin(index / 3.0),
        "rmw": 0.0015 * math.cos(index / 5.0),
        "cma": 0.001 * math.sin(index / 7.0),
        "mom": 0.0005 * ((index % 7) - 3) / 3.0,
    }
    return table[name]


def _factor_frame(months: list[date], *, hidden_from: int | None = None) -> pl.DataFrame:
    """Factor rows per month; rows at ``hidden_from`` onward become invisible."""
    count = len(months)
    available_at = []
    for index, month in enumerate(months):
        stamp = _close_ts(month) + timedelta(days=60)
        if hidden_from is not None and index >= hidden_from:
            stamp = _close_ts(month) + timedelta(days=60 * 365)
        available_at.append(stamp)
    data: dict[str, list[object]] = {"period_end": months}
    for name in FACTOR_COLUMNS:
        data[name] = [_factor_value(name, index) for index in range(count)]
    data["rf"] = [0.0005] * count
    data[AVAILABLE_AT] = available_at
    return pl.DataFrame(data)


@pytest.mark.parametrize("scenario_id", ["FEAT-H02-loadings-pit"])
def test_feat_h02_loadings_pit(scenario_id: str) -> None:
    """FEAT-H02-loadings-pit"""
    months = _month_ends(38)
    prices = _price_frame(months)
    factors = _factor_frame(months, hidden_from=37)
    signal_at = factors.get_column(AVAILABLE_AT)[36] + timedelta(hours=1)

    loadings = estimate_factor_loadings(prices, factors, ticker=_TICKER, signal_at=signal_at, window=_WINDOW)

    assert set(loadings) == {"alpha", *FACTOR_COLUMNS}
    assert all(math.isfinite(value) for value in loadings.values())
    assert "hml" in loadings

    # A 37th month whose factor bar is visible only after signal_at must not change the fit.
    truncated = estimate_factor_loadings(
        _price_frame(months[:37]), _factor_frame(months[:37]), ticker=_TICKER,
        signal_at=signal_at, window=_WINDOW,
    )
    assert truncated == loadings


@pytest.mark.parametrize("scenario_id", ["FEAT-H02-loadings-pit"])
def test_feat_h02_loadings_insufficient_months_fail_closed(scenario_id: str) -> None:
    """FEAT-H02-loadings-pit"""
    months = _month_ends(38)
    prices = _price_frame(months)
    factors = _factor_frame(months, hidden_from=36)
    signal_at = factors.get_column(AVAILABLE_AT)[35] + timedelta(hours=1)

    with pytest.raises(ValueError, match="36"):
        estimate_factor_loadings(prices, factors, ticker=_TICKER, signal_at=signal_at, window=_WINDOW)


@pytest.mark.parametrize("scenario_id", ["FEAT-H02-loadings-pit"])
def test_feat_h02_loadings_reject_naive_signal(scenario_id: str) -> None:
    """FEAT-H02-loadings-pit"""
    months = _month_ends(40)
    with pytest.raises(ValueError, match="timezone-aware"):
        estimate_factor_loadings(
            _price_frame(months), _factor_frame(months), ticker=_TICKER,
            signal_at=datetime(2024, 1, 1), window=_WINDOW,
        )
