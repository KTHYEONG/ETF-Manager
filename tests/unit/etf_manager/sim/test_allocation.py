"""Unit tests for the buy-only multi-sleeve contribution allocation."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Final

import polars as pl
import pytest

from src.etf_manager.data.calendar import load_calendar
from src.etf_manager.data.pipeline import ingest
from src.etf_manager.data.schema import Dataset, spec_for
from src.etf_manager.policy.targets import PolicyId
from src.etf_manager.sim.allocation import AllocationConfig, AllocationDataError, run_allocation
from src.etf_manager.sim.baseline import BaselineConfig, BaselineId, run_baseline

_CALENDAR = load_calendar("XNYS")
_RETRIEVED_AT = datetime(2024, 4, 1, 5, 0, tzinfo=UTC)
_CONTRIBUTION_KRW = 1_300_000.0
_PANEL_START: Final[date] = date(2024, 1, 10)
_PANEL_END: Final[date] = date(2024, 2, 28)
_CONFIG_START: Final[date] = date(2024, 1, 15)
_CONFIG_END: Final[date] = date(2024, 2, 26)


def _panel_window() -> tuple[date, ...]:
    return _CALENDAR.sessions(_PANEL_START, _PANEL_END)


def _prices_panel(days: tuple[date, ...], tickers: tuple[str, ...]) -> pl.DataFrame:
    """Constant 100.0 adjusted closes per (ticker, day) pair."""
    spec = spec_for(Dataset.PRICES)
    rows_ticker: list[str] = []
    rows_date: list[date] = []
    for ticker in tickers:
        for day in days:
            rows_ticker.append(ticker)
            rows_date.append(day)
    n = len(rows_date)
    return pl.DataFrame(
        {
            "ticker": rows_ticker,
            "date": rows_date,
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


def _fx_panel(days: tuple[date, ...]) -> pl.DataFrame:
    spec = spec_for(Dataset.FX)
    ordered = list(days)
    return pl.DataFrame(
        {
            "date": ordered,
            "usdkrw": [1300.0] * len(ordered),
            "source": ["synthetic"] * len(ordered),
            "retrieved_at": [_RETRIEVED_AT] * len(ordered),
        },
        schema=dict(spec.columns),
    )


def _constant_cpi() -> pl.DataFrame:
    """FIXED_LAG 45d stamping makes the level visible at every 2024 execution close."""
    spec = spec_for(Dataset.CPI)
    return ingest(
        pl.DataFrame(
            {
                "period_end": [date(2023, 12, 1)],
                "value": [100.0],
                "source": ["synthetic"],
                "retrieved_at": [_RETRIEVED_AT],
            },
            schema=dict(spec.columns),
        ),
        Dataset.CPI,
    )


def _allocation_config(policy: PolicyId) -> AllocationConfig:
    return AllocationConfig(
        policy=policy,
        start=_CONFIG_START,
        end=_CONFIG_END,
        monthly_contribution_krw=_CONTRIBUTION_KRW,
    )


@pytest.mark.parametrize("scenario_id", ["SIM-G05-s0-matches-b0"])
def test_sim_g05_s0_matches_b0(scenario_id: str) -> None:
    """SIM-G05-s0-matches-b0"""
    window = _panel_window()
    prices = ingest(_prices_panel(window, ("VT",)), Dataset.PRICES)
    fx = ingest(_fx_panel(window), Dataset.FX)
    cpi = _constant_cpi()

    baseline = run_baseline(
        BaselineConfig(
            baseline=BaselineId.B0_GLOBAL,
            ticker="VT",
            start=_CONFIG_START,
            end=_CONFIG_END,
            monthly_contribution_krw=_CONTRIBUTION_KRW,
        ),
        prices,
        fx,
        cpi,
    )
    result = run_allocation(_allocation_config(PolicyId.S0_GLOBAL), prices, fx, cpi)

    assert result.terminal_wealth_krw == pytest.approx(baseline.terminal_wealth_krw, rel=1e-6)
    contributions_alloc = tuple(snapshot.contribution_krw for snapshot in result.snapshots)
    contributions_base = tuple(snapshot.contribution_krw for snapshot in baseline.snapshots)
    assert contributions_alloc == contributions_base


@pytest.mark.parametrize("scenario_id", ["SIM-G06-buy-only-split"])
def test_sim_g06_buy_only_split(scenario_id: str) -> None:
    """SIM-G06-buy-only-split"""
    window = _panel_window()
    prices = ingest(_prices_panel(window, ("VT", "VTI", "VEA", "VWO")), Dataset.PRICES)
    fx = ingest(_fx_panel(window), Dataset.FX)
    cpi = _constant_cpi()

    result = run_allocation(_allocation_config(PolicyId.S2_REGIONAL), prices, fx, cpi)

    first_shares = result.snapshots[0].shares
    assert set(first_shares) == {"VTI", "VEA", "VWO"}
    assert all(first_shares[ticker] > 0.0 for ticker in ("VTI", "VEA", "VWO"))
    for previous, current in zip(result.snapshots, result.snapshots[1:], strict=False):
        for ticker in ("VTI", "VEA", "VWO"):
            assert current.shares[ticker] >= previous.shares[ticker]

    global_result = run_allocation(_allocation_config(PolicyId.S0_GLOBAL), prices, fx, cpi)
    assert tuple(s.contribution_krw for s in result.snapshots) == tuple(
        s.contribution_krw for s in global_result.snapshots
    )

    january = tuple(day for day in window if day < date(2024, 2, 1))
    missing_vwo = ingest(
        pl.concat([_prices_panel(window, ("VTI", "VEA")), _prices_panel(january, ("VWO",))]),
        Dataset.PRICES,
    )
    with pytest.raises(AllocationDataError):
        run_allocation(_allocation_config(PolicyId.S2_REGIONAL), missing_vwo, fx, cpi)
