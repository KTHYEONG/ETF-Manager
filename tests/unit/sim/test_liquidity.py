"""Invariant tests for point-in-time simulated fill liquidity."""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest

from src.sim.liquidity import LiquidityBreachError, LiquidityConfig, check_fill_liquidity

_TICKER = "TEST"
_SESSION = date(2024, 1, 10)
_CLOSE_TS = datetime(2024, 1, 10, 21, 0, tzinfo=UTC)
_VISIBLE_AT = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)


def _prices(rows: tuple[tuple[date, float, int, datetime], ...]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": [_TICKER] * len(rows),
            "date": [row[0] for row in rows],
            "close": [row[1] for row in rows],
            "volume": [row[2] for row in rows],
            "available_at": [row[3] for row in rows],
        },
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "close": pl.Float64,
            "volume": pl.Int64,
            "available_at": pl.Datetime("us", "UTC"),
        },
    )


def _config(*, window: int = 2, participation: float = 0.01) -> LiquidityConfig:
    return LiquidityConfig(
        adv_window_sessions=window,
        max_adv_participation=participation,
        reject_zero_volume=True,
    )


def test_zero_volume_session_rejects_fill() -> None:
    prices = _prices(
        (
            (date(2024, 1, 8), 100.0, 5_000, _VISIBLE_AT),
            (date(2024, 1, 9), 100.0, 5_000, _VISIBLE_AT),
            (_SESSION, 100.0, 0, _CLOSE_TS),
        )
    )

    with pytest.raises(LiquidityBreachError) as exc_info:
        check_fill_liquidity(prices, _TICKER, _SESSION, _CLOSE_TS, 10, _config(participation=0.5))

    message = str(exc_info.value)
    assert _TICKER in message
    assert _SESSION.isoformat() in message
    assert "notional=" in message
    assert "ADV=" in message
    assert "cap=" in message


def test_participation_cap_enforced_at_boundary() -> None:
    prices = _prices(
        (
            (date(2024, 1, 8), 100.0, 10_000, _VISIBLE_AT),
            (date(2024, 1, 9), 100.0, 10_000, _VISIBLE_AT),
            (_SESSION, 100.0, 1_000_000, _CLOSE_TS),
        )
    )

    check_fill_liquidity(prices, _TICKER, _SESSION, _CLOSE_TS, 100, _config())
    with pytest.raises(LiquidityBreachError, match="participation cap exceeded"):
        check_fill_liquidity(prices, _TICKER, _SESSION, _CLOSE_TS, 101, _config())


def test_execution_session_excluded_from_adv() -> None:
    prices = _prices(
        (
            (date(2024, 1, 9), 100.0, 10, _VISIBLE_AT),
            (_SESSION, 100.0, 1_000_000, _CLOSE_TS),
        )
    )

    with pytest.raises(LiquidityBreachError, match="participation cap exceeded"):
        check_fill_liquidity(prices, _TICKER, _SESSION, _CLOSE_TS, 10, _config(window=1))


def test_future_rows_are_invisible() -> None:
    visible_rows = (
        (date(2024, 1, 6), 100.0, 1_000, _VISIBLE_AT),
        (date(2024, 1, 7), 100.0, 1_000, _VISIBLE_AT),
        (date(2024, 1, 8), 100.0, 1_000, _VISIBLE_AT),
    )
    future_at = datetime(2024, 1, 11, 0, 0, tzinfo=UTC)
    base = _prices((*visible_rows, (_SESSION, 100.0, 1_000, _CLOSE_TS)))
    corrupted = _prices(
        (
            *visible_rows,
            (date(2024, 1, 9), 1_000_000.0, 1_000_000, future_at),
            (_SESSION, 100.0, 1_000, _CLOSE_TS),
        )
    )

    messages: list[str] = []
    for prices in (base, corrupted):
        with pytest.raises(LiquidityBreachError) as exc_info:
            check_fill_liquidity(prices, _TICKER, _SESSION, _CLOSE_TS, 100, _config())
        messages.append(str(exc_info.value))

    assert messages[0] == messages[1]
    assert "ADV=100000.00" in messages[0]


def test_short_history_fails_closed() -> None:
    prices = _prices(
        (
            (date(2024, 1, 9), 100.0, 10_000, _VISIBLE_AT),
            (_SESSION, 100.0, 10_000, _CLOSE_TS),
        )
    )

    with pytest.raises(LiquidityBreachError) as exc_info:
        check_fill_liquidity(prices, _TICKER, _SESSION, _CLOSE_TS, 10, _config())

    message = str(exc_info.value)
    assert "trailing history 1/2 sessions" in message
    assert "ADV=unavailable" in message
    assert "cap=unavailable" in message


def test_zero_shares_is_no_op() -> None:
    prices = _prices(((_SESSION, 100.0, 0, _CLOSE_TS),))

    check_fill_liquidity(prices, _TICKER, _SESSION, _CLOSE_TS, 0, _config(window=1))


@pytest.mark.parametrize(
    ("window", "participation"),
    [
        (0, 0.1),
        (2, 0.0),
        (2, 1.01),
    ],
)
def test_invalid_config_rejected(window: int, participation: float) -> None:
    expected = "adv_window_sessions" if window < 1 else "max_adv_participation"
    with pytest.raises(ValueError, match=expected):
        LiquidityConfig(window, participation, True)


def test_missing_execution_price_fails_closed() -> None:
    prices = _prices(((date(2024, 1, 9), 100.0, 10_000, _VISIBLE_AT),))

    with pytest.raises(LiquidityBreachError, match="missing or ambiguous execution price"):
        check_fill_liquidity(prices, _TICKER, _SESSION, _CLOSE_TS, 10, _config(window=1))


def test_zero_volume_can_be_explicitly_allowed() -> None:
    prices = _prices(
        (
            (date(2024, 1, 9), 100.0, 10_000, _VISIBLE_AT),
            (_SESSION, 100.0, 0, _CLOSE_TS),
        )
    )
    config = LiquidityConfig(
        adv_window_sessions=1,
        max_adv_participation=1.0,
        reject_zero_volume=False,
    )

    check_fill_liquidity(prices, _TICKER, _SESSION, _CLOSE_TS, 10, config)
