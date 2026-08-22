"""Unit tests for the fast-mode single-sleeve KRW DCA baseline."""

from __future__ import annotations

from datetime import UTC, date, datetime
from collections.abc import Mapping

import polars as pl
import pytest

from src.etf_manager.data.calendar import load_calendar
from src.etf_manager.data.pipeline import ingest
from src.etf_manager.data.schema import Dataset, spec_for
from src.etf_manager.sim.baseline import (
    BaselineConfig,
    BaselineDataError,
    BaselineId,
    run_baseline,
)

_CALENDAR = load_calendar("XNYS")
_RETRIEVED_AT = datetime(2024, 4, 1, 5, 0, tzinfo=UTC)
_CONTRIBUTION_KRW = 1_300_000.0
_PANEL_START = date(2024, 1, 10)
_PANEL_END = date(2024, 2, 28)
_CONFIG_START = date(2024, 1, 15)
_CONFIG_END = date(2024, 2, 26)


def _sessions(start: date, end: date) -> tuple[date, ...]:
    return _CALENDAR.sessions(start, end)


def _panel_window() -> tuple[date, ...]:
    return _sessions(_PANEL_START, _PANEL_END)


def _prices_panel(
    days: tuple[date, ...],
    closes_by_ticker: Mapping[str, Mapping[date, float]],
    *,
    default_close: float = 100.0,
) -> pl.DataFrame:
    spec = spec_for(Dataset.PRICES)
    tickers: list[str] = []
    dates: list[date] = []
    closes: list[float] = []
    for ticker in sorted(closes_by_ticker):
        overrides = closes_by_ticker[ticker]
        for day in days:
            tickers.append(ticker)
            dates.append(day)
            closes.append(float(overrides.get(day, default_close)))
    n = len(dates)
    return pl.DataFrame(
        {
            "ticker": tickers,
            "date": dates,
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


def _fx_panel(
    days: tuple[date, ...],
    rates: Mapping[date, float],
    *,
    default_rate: float = 1300.0,
) -> pl.DataFrame:
    spec = spec_for(Dataset.FX)
    ordered = list(days)
    values = [float(rates.get(day, default_rate)) for day in ordered]
    return pl.DataFrame(
        {
            "date": ordered,
            "usdkrw": values,
            "source": ["synthetic"] * len(ordered),
            "retrieved_at": [_RETRIEVED_AT] * len(ordered),
        },
        schema=dict(spec.columns),
    )


def _config(ticker: str, start: date, end: date, **overrides: object) -> BaselineConfig:
    values: dict[str, object] = {
        "baseline": BaselineId.B0_GLOBAL,
        "ticker": ticker,
        "start": start,
        "end": end,
        "monthly_contribution_krw": _CONTRIBUTION_KRW,
    }
    values.update(overrides)
    return BaselineConfig(**values)  # type: ignore[arg-type]


def test_sim_d02_delayed_fill_not_signal_close() -> None:
    """SIM-D02-delayed-fill-not-signal-close"""
    window = _panel_window()
    signal_day = date(2024, 1, 31)
    execution_day = date(2024, 2, 1)

    prices = ingest(_prices_panel(window, {"AAA": {signal_day: 100.0, execution_day: 110.0}}), Dataset.PRICES)
    fx = ingest(_fx_panel(window, {}), Dataset.FX)
    result = run_baseline(_config("AAA", _CONFIG_START, _CONFIG_END), prices, fx)

    first = result.snapshots[0]
    assert first.session == execution_day
    assert first.shares == 9.0

    budget_usd = _CONTRIBUTION_KRW / 1300.0
    bought_price = (budget_usd - first.cash_usd) / first.shares
    assert bought_price == pytest.approx(110.0, abs=1e-9), "fill must use execution-session close P_e"
    assert abs(bought_price - 100.0) > 1.0, "signal-session close P_s must never be the fill price"


def test_sim_d03_cash_conservation() -> None:
    """SIM-D03-cash-conservation"""
    window = _panel_window()

    prices = ingest(_prices_panel(window, {"AAA": {}}), Dataset.PRICES)
    fx = ingest(_fx_panel(window, {}), Dataset.FX)
    result = run_baseline(_config("AAA", _CONFIG_START, _CONFIG_END), prices, fx)

    last = result.snapshots[-1]
    last_price = 100.0
    last_fx = 1300.0
    identity = last.cash_krw + last.cash_usd * last_fx + last.shares * last_price * last_fx
    assert identity == pytest.approx(result.terminal_wealth_krw, abs=1e-6)

    total_contributions = sum(snapshot.contribution_krw for snapshot in result.snapshots)
    assert total_contributions == pytest.approx(_CONTRIBUTION_KRW * len(result.snapshots), abs=1e-9)

    assert all(snapshot.cash_krw >= 0.0 for snapshot in result.snapshots)
    assert all(snapshot.cash_usd >= 0.0 for snapshot in result.snapshots)
    assert all(snapshot.shares >= 0.0 for snapshot in result.snapshots)
    assert last.cash_usd < last_price


def test_sim_d04_fail_closed_missing_px() -> None:
    """SIM-D04-fail-closed-missing-px"""
    window = _panel_window()
    january = [day for day in window if day < date(2024, 2, 1)]

    absent_on_execution = ingest(
        _panel_with_dates({"AAA": window, "BBB": tuple(january)}),
        Dataset.PRICES,
    )
    fx = ingest(_fx_panel(window, {}), Dataset.FX)
    with pytest.raises(BaselineDataError):
        run_baseline(_config("BBB", _CONFIG_START, _CONFIG_END), absent_on_execution, fx)

    null_fx = _fx_frame_with_nulls(window, date(2024, 2, 1))
    prices = ingest(_prices_panel(window, {"AAA": {}}), Dataset.PRICES)
    stamped_null_fx = ingest(null_fx, Dataset.FX)
    with pytest.raises(BaselineDataError):
        run_baseline(_config("AAA", _CONFIG_START, _CONFIG_END), prices, stamped_null_fx)

    zero_contribution = _config("AAA", _CONFIG_START, _CONFIG_END, monthly_contribution_krw=0.0)
    with pytest.raises(ValueError, match="monthly_contribution_krw"):
        run_baseline(zero_contribution, prices, fx)


def _panel_with_dates(series: Mapping[str, tuple[date, ...]]) -> pl.DataFrame:
    spec = spec_for(Dataset.PRICES)
    tickers: list[str] = []
    dates: list[date] = []
    for ticker in sorted(series):
        for day in series[ticker]:
            tickers.append(ticker)
            dates.append(day)
    n = len(dates)
    return pl.DataFrame(
        {
            "ticker": tickers,
            "date": dates,
            "open": [98.0] * n,
            "high": [102.0] * n,
            "low": [97.0] * n,
            "close": [100.0] * n,
            "volume": [10_000] * n,
            "adjusted_close": [100.0] * n,
            "dividend": [0.0] * n,
            "split_factor": [1.0] * n,
            "source": ["synthetic"] * n,
            "retrieved_at": [_RETRIEVED_AT] * n,
        },
        schema=dict(spec.columns),
    )


def _fx_frame_with_nulls(days: tuple[date, ...], null_day: date) -> pl.DataFrame:
    spec = spec_for(Dataset.FX)
    ordered = list(days)
    values: list[float | None] = [None if day == null_day else 1300.0 for day in ordered]
    return pl.DataFrame(
        {
            "date": ordered,
            "usdkrw": values,
            "source": ["synthetic"] * len(ordered),
            "retrieved_at": [_RETRIEVED_AT] * len(ordered),
        },
        schema=dict(spec.columns),
    )


def test_sim_d05_b0_b1_same_cashflow() -> None:
    """SIM-D05-b0-b1-same-cashflow"""
    window = _panel_window()

    prices = ingest(
        _prices_panel(
            window,
            {
                "AAA": {date(2024, 1, 31): 100.0, date(2024, 2, 1): 110.0},
                "BBB": {date(2024, 1, 31): 105.0, date(2024, 2, 1): 95.0},
            },
        ),
        Dataset.PRICES,
    )
    fx = ingest(_fx_panel(window, {}), Dataset.FX)

    global_result = run_baseline(_config("AAA", _CONFIG_START, _CONFIG_END), prices, fx)
    us_config = _config("BBB", _CONFIG_START, _CONFIG_END, baseline=BaselineId.B1_US)
    us_result = run_baseline(us_config, prices, fx)

    contributions_global = tuple(snapshot.contribution_krw for snapshot in global_result.snapshots)
    contributions_us = tuple(snapshot.contribution_krw for snapshot in us_result.snapshots)
    assert contributions_global == contributions_us

    shares_global = tuple(snapshot.shares for snapshot in global_result.snapshots)
    shares_us = tuple(snapshot.shares for snapshot in us_result.snapshots)
    assert shares_global != shares_us
    assert global_result.terminal_wealth_krw != us_result.terminal_wealth_krw
