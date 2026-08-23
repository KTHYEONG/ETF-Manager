"""Unit tests for PIT trailing price-path drawdown guards."""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from src.etf_manager.data.calendar import load_calendar
from src.etf_manager.features.drawdown import trailing_price_drawdown

_CALENDAR = load_calendar("XNYS")
_DAYS = _CALENDAR.sessions(date(2024, 1, 2), date(2024, 12, 31))


def _prices_frame(closes: list[float], *, hidden_last: float | None = None) -> pl.DataFrame:
    """Consecutive-session closes; the optional extra bar is published after the window."""
    all_closes = [*closes, *([] if hidden_last is None else [hidden_last])]
    days = _DAYS[: len(all_closes)]
    return pl.DataFrame(
        {
            "ticker": ["TEST"] * len(all_closes),
            "date": list(days),
            "adjusted_close": all_closes,
            "available_at": [_CALENDAR.close_ts(day) for day in days],
        },
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "adjusted_close": pl.Float64,
            "available_at": pl.Datetime("us", "UTC"),
        },
    )


def test_trailing_drawdown_peak_to_trough() -> None:
    """A 100 -> 80 dip after a 120 peak yields the same fraction as analytics.max_drawdown."""
    frame = _prices_frame([100.0, 120.0, 80.0, 110.0])
    as_of = _CALENDAR.close_ts(_DAYS[3])
    assert trailing_price_drawdown(frame, ticker="TEST", as_of_ts=as_of, window=4) == pytest.approx(
        80.0 / 120.0 - 1.0
    )


def test_trailing_drawdown_pit_visibility_and_guards() -> None:
    """Hidden bars never enter the window; naive timestamps and short windows fail closed."""
    frame = _prices_frame([100.0, 120.0, 80.0, 110.0], hidden_last=10.0)
    as_of = _CALENDAR.close_ts(_DAYS[3])
    assert trailing_price_drawdown(frame, ticker="TEST", as_of_ts=as_of, window=4) == pytest.approx(
        80.0 / 120.0 - 1.0
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        trailing_price_drawdown(frame, ticker="TEST", as_of_ts=datetime(2024, 1, 10), window=4)
    with pytest.raises(ValueError, match="visible sessions"):
        trailing_price_drawdown(frame, ticker="TEST", as_of_ts=as_of, window=5)
    with pytest.raises(ValueError, match="window"):
        trailing_price_drawdown(frame, ticker="TEST", as_of_ts=as_of, window=0)
