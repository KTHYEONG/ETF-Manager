"""Unit tests for the point-in-time market view over PRICES and RATES."""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest

from src.data.calendar import load_calendar
from src.data.pipeline import ingest
from src.data.schema import Dataset, spec_for
from src.features.pit_market import PitMarket, PitMarketError

_CALENDAR = load_calendar("XNYS")
_RETRIEVED_AT = datetime(2024, 4, 1, 5, 0, tzinfo=UTC)


def _prices_frame(days: tuple[date, ...], closes: tuple[float, ...], ticker: str = "QQQ") -> pl.DataFrame:
    spec = spec_for(Dataset.PRICES)
    return ingest(
        pl.DataFrame(
            {
                "ticker": [ticker] * len(days),
                "date": list(days),
                "open": list(closes),
                "high": list(closes),
                "low": list(closes),
                "close": list(closes),
                "volume": [10_000] * len(days),
                "adjusted_close": list(closes),
                "dividend": [0.0] * len(days),
                "split_factor": [1.0] * len(days),
                "source": ["synthetic"] * len(days),
                "retrieved_at": [_RETRIEVED_AT] * len(days),
            },
            schema=dict(spec.columns),
        ),
        Dataset.PRICES,
    )


def _rates_frame(observations: tuple[tuple[date, float | None], ...]) -> pl.DataFrame:
    spec = spec_for(Dataset.RATES)
    return ingest(
        pl.DataFrame(
            {
                "series_id": ["DTB3"] * len(observations),
                "observation_date": [day for day, _ in observations],
                "value": [value for _, value in observations],
                "source": ["synthetic"] * len(observations),
                "retrieved_at": [_RETRIEVED_AT] * len(observations),
            },
            schema=dict(spec.columns),
        ),
        Dataset.RATES,
    )


def test_future_rows_invisible() -> None:
    """March perturbations never leak into February-visible month ends."""
    days = _CALENDAR.sessions(date(2024, 1, 2), date(2024, 3, 28))
    closes = tuple(100.0 + float(index) for index in range(len(days)))
    market = PitMarket(_prices_frame(days, closes), None)
    as_of = _CALENDAR.close_ts(date(2024, 2, 29))
    jan_close = closes[days.index(date(2024, 1, 31))]
    feb_close = closes[days.index(date(2024, 2, 29))]
    assert market.month_end_adjusted_closes("QQQ", as_of, 2) == (jan_close, feb_close)
    perturbed = tuple(value * 999.0 if day >= date(2024, 3, 1) else value for day, value in zip(days, closes, strict=True))
    disturbed = PitMarket(_prices_frame(days, perturbed), None)
    assert disturbed.month_end_adjusted_closes("QQQ", as_of, 2) == (jan_close, feb_close)
    assert disturbed.daily_adjusted_closes("QQQ", as_of, 5) == market.daily_adjusted_closes("QQQ", as_of, 5)


def test_short_history_fails_closed() -> None:
    """Fewer visible month ends than asked, unknown tickers, and bad counts fail."""
    days = _CALENDAR.sessions(date(2024, 1, 2), date(2024, 5, 31))
    market = PitMarket(_prices_frame(days, tuple(100.0 for _ in days)), None)
    as_of = _CALENDAR.close_ts(date(2024, 5, 31))
    assert len(market.month_end_adjusted_closes("QQQ", as_of, 5)) == 5
    with pytest.raises(PitMarketError):
        market.month_end_adjusted_closes("QQQ", as_of, 10)
    with pytest.raises(PitMarketError):
        market.daily_adjusted_closes("QQQ", as_of, len(days) + 1)
    with pytest.raises(PitMarketError):
        market.month_end_adjusted_closes("UNKNOWN", as_of, 1)
    with pytest.raises(PitMarketError):
        market.month_end_adjusted_closes("QQQ", as_of, 0)
    with pytest.raises(PitMarketError):
        market.daily_adjusted_closes("QQQ", as_of, -3)


def test_rate_as_of_respects_lag() -> None:
    """A rate stamped after the instant is invisible; the prior observation wins."""
    rates = _rates_frame(((date(2024, 1, 1), 4.5), (date(2024, 1, 5), 4.75)))
    days = _CALENDAR.sessions(date(2024, 1, 2), date(2024, 1, 10))
    market = PitMarket(_prices_frame(days, tuple(100.0 for _ in days)), rates)
    assert market.rate_percent("DTB3", datetime(2024, 1, 8, tzinfo=UTC)) == pytest.approx(4.5)
    assert market.rate_percent("DTB3", datetime(2024, 1, 10, tzinfo=UTC)) == pytest.approx(4.75)
    with pytest.raises(PitMarketError):
        market.rate_percent("DTB3", datetime(2024, 1, 4, tzinfo=UTC))
    with pytest.raises(PitMarketError):
        market.rate_percent("UNKNOWN", datetime(2024, 1, 10, tzinfo=UTC))


def test_rates_without_frame_fails_closed() -> None:
    """Rate access without a loaded RATES frame fails instead of inventing a rate."""
    days = _CALENDAR.sessions(date(2024, 1, 2), date(2024, 1, 10))
    market = PitMarket(_prices_frame(days, tuple(100.0 for _ in days)), None)
    with pytest.raises(PitMarketError, match="not loaded"):
        market.rate_percent("DTB3", datetime(2024, 1, 10, tzinfo=UTC))


def test_naive_instant_rejected() -> None:
    """Naive decision instants fail on every accessor."""
    days = _CALENDAR.sessions(date(2024, 1, 2), date(2024, 3, 29))
    market = PitMarket(
        _prices_frame(days, tuple(100.0 for _ in days)),
        _rates_frame(((date(2024, 1, 1), 4.5),)),
    )
    naive = datetime(2024, 2, 29, 12, 0)
    with pytest.raises(PitMarketError):
        market.month_end_adjusted_closes("QQQ", naive, 1)
    with pytest.raises(PitMarketError):
        market.daily_adjusted_closes("QQQ", naive, 1)
    with pytest.raises(PitMarketError):
        market.rate_percent("DTB3", naive)
