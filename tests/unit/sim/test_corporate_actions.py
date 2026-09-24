"""Unit tests for corporate-action normalization from vendor price fields."""

from __future__ import annotations

from datetime import date
from fractions import Fraction

import polars as pl
import pytest

from src.sim.corporate_actions import (
    CorporateActionError,
    CorporateActionKind,
    corporate_actions_from_prices,
    normalize_split_factor,
)


def _prices(rows: list[tuple[str, date, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": [ticker for ticker, _day, _div, _split in rows],
            "date": [day for _ticker, day, _div, _split in rows],
            "dividend": [dividend for _ticker, _day, dividend, _split in rows],
            "split_factor": [split for _ticker, _day, _div, split in rows],
        },
        schema={"ticker": pl.String, "date": pl.Date, "dividend": pl.Float64, "split_factor": pl.Float64},
    )


def test_noisy_forward_split_normalizes() -> None:
    """A noisy vendor factor maps onto the exact forward ratio."""
    assert normalize_split_factor(3.000003) == Fraction(3)
    assert normalize_split_factor(2.0) == Fraction(2)


def test_reverse_split_normalizes() -> None:
    """Fractional factors map onto exact reverse ratios."""
    assert normalize_split_factor(0.5) == Fraction(1, 2)
    assert normalize_split_factor(0.25) == Fraction(1, 4)


def test_unmappable_factor_fails_closed() -> None:
    """Factors far from any exact ratio never synthesize shares."""
    with pytest.raises(CorporateActionError):
        normalize_split_factor(1.37)


def test_split_factor_boundaries() -> None:
    """Non-finite, non-positive, unit, and non-numeric factors fail closed."""
    for raw in (float("nan"), float("inf"), 0.0, -2.0, 1.0, 1.5):
        with pytest.raises(CorporateActionError):
            normalize_split_factor(raw)
    with pytest.raises(CorporateActionError):
        normalize_split_factor("2")  # type: ignore[arg-type]


def test_same_day_split_precedes_dividend() -> None:
    """A same-day dividend is attributed on the post-split share count."""
    frame = _prices([("SOXX", date(2024, 6, 10), 0.1, 2.0)])
    events = corporate_actions_from_prices(frame, tickers=("SOXX",), start=date(2024, 1, 1), end=date(2024, 12, 31))
    assert [event.kind for event in events] == [CorporateActionKind.SPLIT, CorporateActionKind.DIVIDEND]
    assert events[0].split_ratio == Fraction(2)
    assert events[0].dividend_usd is None
    assert events[1].dividend_usd == pytest.approx(0.1)
    assert events[1].split_ratio is None


def test_window_filter_excludes_early_events() -> None:
    """Events with ex-dates before the window never surface."""
    frame = _prices(
        [
            ("QQQ", date(2023, 12, 29), 0.5, 1.0),
            ("QQQ", date(2024, 3, 18), 0.5, 1.0),
        ]
    )
    events = corporate_actions_from_prices(frame, tickers=("QQQ",), start=date(2024, 1, 1), end=date(2024, 12, 31))
    assert [event.ex_date for event in events] == [date(2024, 3, 18)]


def test_quiet_rows_produce_no_event() -> None:
    """Rows with unit split factor and zero dividend are not events."""
    frame = _prices([("SPY", date(2024, 5, 10), 0.0, 1.0)])
    assert corporate_actions_from_prices(frame, tickers=("SPY",), start=date(2024, 1, 1), end=date(2024, 12, 31)) == ()


def test_negative_dividend_fails_closed() -> None:
    """A negative cash amount is a data error, never a payout."""
    frame = _prices([("SPY", date(2024, 5, 10), -0.5, 1.0)])
    with pytest.raises(CorporateActionError, match="dividend"):
        corporate_actions_from_prices(frame, tickers=("SPY",), start=date(2024, 1, 1), end=date(2024, 12, 31))


def test_unmappable_row_split_fails_closed() -> None:
    """An unmappable row factor aborts extraction instead of skewing lots."""
    frame = _prices([("EFA", date(2005, 6, 9), 0.0, 1.37)])
    with pytest.raises(CorporateActionError):
        corporate_actions_from_prices(frame, tickers=("EFA",), start=date(2005, 1, 1), end=date(2005, 12, 31))


def test_ticker_filter_and_ordering() -> None:
    """Only requested tickers surface, ordered by date then ticker."""
    frame = _prices(
        [
            ("SPY", date(2024, 6, 10), 0.2, 1.0),
            ("QQQ", date(2024, 6, 10), 0.3, 1.0),
            ("DIA", date(2024, 6, 10), 0.4, 1.0),
        ]
    )
    events = corporate_actions_from_prices(frame, tickers=("QQQ", "SPY"), start=date(2024, 1, 1), end=date(2024, 12, 31))
    assert [event.ticker for event in events] == ["QQQ", "SPY"]
