"""Unit tests for PIT session-return and trailing-vol feature scenarios."""

from __future__ import annotations

import statistics
from datetime import UTC, date, datetime

import polars as pl
import pytest

from src.data.calendar import load_calendar
from src.data.pipeline import ingest
from src.data.pit import TS_DTYPE
from src.data.schema import Dataset, spec_for
from src.features.risk import trailing_simple_vol
from src.features.returns import session_returns
from src.policy.targets import PolicyError

_CALENDAR = load_calendar("XNYS")
_RETRIEVED_AT = datetime(2024, 4, 1, 5, 0, tzinfo=UTC)


def _vt_panel(closes: list[float]) -> pl.DataFrame:
    """Synthetic VT panel with SESSION_CLOSE availability stamping via ingest."""
    days = _CALENDAR.sessions(date(2024, 1, 2), date(2024, 1, 10))[: len(closes)]
    spec = spec_for(Dataset.PRICES)
    n = len(closes)
    raw = pl.DataFrame(
        {
            "ticker": ["VT"] * n,
            "date": list(days),
            "open": [value * 0.98 for value in closes],
            "high": [value * 1.02 for value in closes],
            "low": [value * 0.97 for value in closes],
            "close": closes,
            "volume": [10_000] * n,
            "adjusted_close": closes,
            "dividend": [0.0] * n,
            "split_factor": [1.0] * n,
            "source": ["synthetic"] * n,
            "retrieved_at": [_RETRIEVED_AT] * n,
        },
        schema=dict(spec.columns),
    )
    return ingest(raw, Dataset.PRICES)


@pytest.mark.parametrize("scenario_id", ["FEAT-G01-session-returns-pit"])
def test_feat_g01_session_returns_pit(scenario_id: str) -> None:
    """FEAT-G01-session-returns-pit"""
    prices = _vt_panel([100.0, 110.0, 99.0])

    out = session_returns(prices, ticker="VT")

    assert out.height == 2
    assert out.get_column("simple_return").to_list() == pytest.approx([0.10, -0.10])
    assert out.item(0, "available_at") == _CALENDAR.close_ts(date(2024, 1, 3))
    assert out.item(0, "available_at") != _CALENDAR.close_ts(date(2024, 1, 2))

    with pytest.raises(ValueError, match="session_returns"):
        session_returns(prices.filter(pl.col("date") == date(2024, 1, 2)), ticker="VT")
    with pytest.raises(ValueError, match="session_returns"):
        session_returns(prices, ticker="MISSING")


def _returns_frame(values: list[float], *, last_available_after: datetime | None = None) -> pl.DataFrame:
    """Minimal returns frame; optionally stamps the last row beyond a later close."""
    days = _CALENDAR.sessions(date(2024, 1, 2), date(2024, 1, 31))[: len(values) + 1][1:]
    stamps = [_CALENDAR.close_ts(day) for day in days]
    if last_available_after is not None:
        stamps[-1] = last_available_after
    return pl.DataFrame(
        {
            "date": list(days),
            "ticker": ["TEST"] * len(values),
            "simple_return": values,
            "available_at": stamps,
        },
        schema={"date": pl.Date, "ticker": pl.String, "simple_return": pl.Float64, "available_at": TS_DTYPE},
    )


@pytest.mark.parametrize("scenario_id", ["FEAT-G02-trailing-vol-window"])
def test_feat_g02_trailing_vol_window(scenario_id: str) -> None:
    """FEAT-G02-trailing-vol-window"""
    values = [0.01, 0.03, 0.05]
    as_of = _CALENDAR.close_ts(date(2024, 1, 10))

    sigma = trailing_simple_vol(_returns_frame(values), as_of_ts=as_of, window=3)

    assert sigma == pytest.approx(statistics.stdev(values))

    with pytest.raises((ValueError, PolicyError)):
        trailing_simple_vol(_returns_frame(values[:2]), as_of_ts=as_of, window=3)

    hidden_last = _returns_frame(values, last_available_after=_CALENDAR.close_ts(date(2024, 1, 31)))
    with pytest.raises((ValueError, PolicyError)):
        trailing_simple_vol(hidden_last, as_of_ts=as_of, window=3)
