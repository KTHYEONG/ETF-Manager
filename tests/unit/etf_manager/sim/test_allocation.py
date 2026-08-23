"""Unit tests for the buy-only multi-sleeve contribution allocation."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from typing import Final

import polars as pl
import pytest

import src.etf_manager.sim.allocation as allocation_module
from src.etf_manager.data.calendar import load_calendar
from src.etf_manager.data.pipeline import ingest
from src.etf_manager.data.schema import Dataset, spec_for
from src.etf_manager.policy.currency import CurrencyConfig
from src.etf_manager.policy.overlay import OverlayConfig
from src.etf_manager.policy.targets import PolicyId
from src.etf_manager.policy.tilt import FactorTilt
from src.etf_manager.sim.allocation import AllocationConfig, AllocationDataError, AllocationResult, run_allocation
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


@pytest.mark.parametrize("scenario_id", ["SIM-H04-tilt-none-identity"])
def test_sim_h04_tilt_none_identity(scenario_id: str) -> None:
    """SIM-H04-tilt-none-identity"""
    window = _panel_window()
    prices = ingest(_prices_panel(window, ("VT", "VTI", "VEA", "VWO")), Dataset.PRICES)
    fx = ingest(_fx_panel(window), Dataset.FX)
    cpi = _constant_cpi()

    reference = run_allocation(_allocation_config(PolicyId.S0_GLOBAL), prices, fx, cpi)
    with_factors_frame = run_allocation(
        _allocation_config(PolicyId.S0_GLOBAL), prices, fx, cpi, factors=pl.DataFrame()
    )
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

    # A None tilt must reproduce the Phase 3 path exactly, factors frame or not.
    assert with_factors_frame.terminal_wealth_krw == pytest.approx(reference.terminal_wealth_krw, rel=1e-6)
    assert reference.terminal_wealth_krw == pytest.approx(baseline.terminal_wealth_krw, rel=1e-6)

    tilted_config = replace(
        _allocation_config(PolicyId.S2_REGIONAL),
        tilt=FactorTilt(factor="hml", intensity=0.1),
    )
    with pytest.raises(ValueError, match="factors"):
        run_allocation(tilted_config, prices, fx, cpi)


def test_sim_h04_store_loads_factors_only_for_tilt(monkeypatch: pytest.MonkeyPatch) -> None:
    """SIM-H04-tilt-none-identity"""
    requested: list[Dataset] = []
    loaded: list[Dataset] = []
    captured: dict[str, object] = {}

    def fake_latest(settings: object, dataset: Dataset) -> object:
        requested.append(dataset)
        return object()

    def fake_visible(settings: object, dataset: Dataset, decision_ts: object) -> pl.DataFrame:
        loaded.append(dataset)
        return pl.DataFrame()

    def fake_run(
        config: AllocationConfig,
        prices: pl.DataFrame,
        fx: pl.DataFrame,
        cpi: pl.DataFrame,
        factors: pl.DataFrame | None = None,
        macro: pl.DataFrame | None = None,
    ) -> AllocationResult:
        captured["config"] = config
        captured["factors"] = factors
        captured["macro"] = macro
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=1.0,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=0.8,
            xirr_real=-0.1,
        )

    monkeypatch.setattr(allocation_module, "latest_artifact", fake_latest)
    monkeypatch.setattr(allocation_module, "load_visible", fake_visible)
    monkeypatch.setattr(allocation_module, "run_allocation", fake_run)

    plain = allocation_module.run_allocation_from_store(_allocation_config(PolicyId.S0_GLOBAL), settings=object())  # type: ignore[arg-type]
    assert plain.terminal_wealth_krw == 1.0
    assert Dataset.FACTORS not in requested
    assert Dataset.FACTORS not in loaded
    assert captured["factors"] is None

    requested.clear()
    loaded.clear()
    captured.clear()
    tilted = replace(_allocation_config(PolicyId.S2_REGIONAL), tilt=FactorTilt(factor="hml", intensity=0.1))
    allocation_module.run_allocation_from_store(tilted, settings=object())  # type: ignore[arg-type]
    assert Dataset.FACTORS in requested
    assert Dataset.FACTORS in loaded
    assert isinstance(captured["factors"], pl.DataFrame)


def _vti_rally_panel(days: tuple[date, ...]) -> pl.DataFrame:
    """Constant 100 closes except VTI jumps to 400 from 2024-02-27 onward."""
    spec = spec_for(Dataset.PRICES)
    tickers: list[str] = []
    dates: list[date] = []
    closes: list[float] = []
    for ticker in ("VTI", "VEA", "VWO"):
        for day in days:
            tickers.append(ticker)
            dates.append(day)
            closes.append(400.0 if ticker == "VTI" and day >= date(2024, 2, 27) else 100.0)
    n = len(dates)
    return pl.DataFrame(
        {
            "ticker": tickers,
            "date": dates,
            "open": [close * 0.98 for close in closes],
            "high": [close * 1.02 for close in closes],
            "low": [close * 0.97 for close in closes],
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


@pytest.mark.parametrize("scenario_id", ["SIM-I04-allocation-band-path"])
def test_sim_i04_allocation_band_path(scenario_id: str) -> None:
    """SIM-I04-allocation-band-path"""
    window = _panel_window()
    prices = ingest(_vti_rally_panel(window), Dataset.PRICES)
    fx = ingest(_fx_panel(window), Dataset.FX)
    cpi = _constant_cpi()
    sleeves = ("VTI", "VEA", "VWO")

    banded = run_allocation(
        replace(_allocation_config(PolicyId.S2_REGIONAL), rebalance_band=0.0), prices, fx, cpi
    )
    identity = run_allocation(
        replace(_allocation_config(PolicyId.S2_REGIONAL), rebalance_band=None), prices, fx, cpi
    )
    default = run_allocation(_allocation_config(PolicyId.S2_REGIONAL), prices, fx, cpi)

    # Explicit None must reproduce the Phase 3 spend path exactly.
    assert identity.terminal_wealth_krw == pytest.approx(default.terminal_wealth_krw, rel=1e-6)

    for result in (banded, identity):
        for previous, current in zip(result.snapshots, result.snapshots[1:], strict=False):
            for ticker in sleeves:
                assert current.shares[ticker] >= previous.shares[ticker]

    # By month 2 VTI is overweight beyond any band, so deficit-proportional mixing
    # routes strictly more cash to each underweight sleeve than the target mix.
    for ticker in ("VEA", "VWO"):
        banded_buys = banded.snapshots[-1].shares[ticker] - banded.snapshots[0].shares[ticker]
        identity_buys = identity.snapshots[-1].shares[ticker] - identity.snapshots[0].shares[ticker]
        assert banded_buys > identity_buys > 0.0


_LONG_PANEL_START: Final[date] = date(2022, 12, 1)


def _long_rising_panel(days: tuple[date, ...]) -> pl.DataFrame:
    """Identical monotonic 0.1%/session closes per sleeve over a multi-year window."""
    spec = spec_for(Dataset.PRICES)
    tickers: list[str] = []
    dates: list[date] = []
    closes: list[float] = []
    for ticker in ("VTI", "VEA", "VWO"):
        for index, day in enumerate(days):
            tickers.append(ticker)
            dates.append(day)
            closes.append(100.0 * 1.001**index)
    n = len(dates)
    return pl.DataFrame(
        {
            "ticker": tickers,
            "date": dates,
            "open": [close * 0.98 for close in closes],
            "high": [close * 1.02 for close in closes],
            "low": [close * 0.97 for close in closes],
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


@pytest.mark.parametrize("scenario_id", ["SIM-J04-overlay-none-identity"])
def test_sim_j04_overlay_none_identity(scenario_id: str) -> None:
    """SIM-J04-overlay-none-identity"""
    window = _panel_window()
    prices = ingest(_prices_panel(window, ("VT",)), Dataset.PRICES)
    fx = ingest(_fx_panel(window), Dataset.FX)
    cpi = _constant_cpi()

    result = run_allocation(_allocation_config(PolicyId.S0_GLOBAL), prices, fx, cpi)
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

    assert result.terminal_wealth_krw == pytest.approx(baseline.terminal_wealth_krw, rel=1e-6)


@pytest.mark.parametrize("scenario_id", ["SIM-J04-overlay-none-identity"])
def test_sim_j04_overlay_buy_only_long_panel(scenario_id: str) -> None:
    """SIM-J04-overlay-none-identity"""
    days = _CALENDAR.sessions(_LONG_PANEL_START, date(2024, 2, 29))
    prices = ingest(_long_rising_panel(days), Dataset.PRICES)
    fx = ingest(_fx_panel(days), Dataset.FX)
    cpi = _constant_cpi()
    config = replace(
        _allocation_config(PolicyId.S2_REGIONAL), overlay=OverlayConfig(max_shift=0.10)
    )

    result = run_allocation(config, prices, fx, cpi)

    assert all(snapshot.shares for snapshot in result.snapshots)
    for previous, current in zip(result.snapshots, result.snapshots[1:], strict=False):
        for ticker in ("VTI", "VEA", "VWO"):
            assert current.shares[ticker] >= previous.shares[ticker]


def _rising_fx_panel(days: tuple[date, ...]) -> pl.DataFrame:
    """Strictly increasing usdkrw prints so every trailing window flags expensive USD."""
    spec = spec_for(Dataset.FX)
    ordered = list(days)
    return pl.DataFrame(
        {
            "date": ordered,
            "usdkrw": [1300.0 + 10.0 * index for index in range(len(ordered))],
            "source": ["synthetic"] * len(ordered),
            "retrieved_at": [_RETRIEVED_AT] * len(ordered),
        },
        schema=dict(spec.columns),
    )


@pytest.mark.parametrize("scenario_id", ["SIM-K04-currency-none-identity"])
def test_sim_k04_currency_none_identity(scenario_id: str) -> None:
    """SIM-K04-currency-none-identity"""
    window = _panel_window()
    prices = ingest(_prices_panel(window, ("VT",)), Dataset.PRICES)
    fx = ingest(_fx_panel(window), Dataset.FX)
    cpi = _constant_cpi()

    reference = run_allocation(_allocation_config(PolicyId.S0_GLOBAL), prices, fx, cpi)
    none_currency = run_allocation(
        replace(_allocation_config(PolicyId.S0_GLOBAL), currency=None), prices, fx, cpi
    )
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

    assert none_currency.terminal_wealth_krw == pytest.approx(reference.terminal_wealth_krw, rel=1e-6)
    assert reference.terminal_wealth_krw == pytest.approx(baseline.terminal_wealth_krw, rel=1e-6)


@pytest.mark.parametrize("scenario_id", ["SIM-K04-currency-none-identity"])
def test_sim_k04_defer_keeps_cash_krw(scenario_id: str) -> None:
    """SIM-K04-currency-none-identity"""
    window = _panel_window()
    prices = ingest(_prices_panel(window, ("VT",)), Dataset.PRICES)
    rising_fx = ingest(_rising_fx_panel(window), Dataset.FX)
    cpi = _constant_cpi()
    deferred_config = replace(
        _allocation_config(PolicyId.S0_GLOBAL),
        currency=CurrencyConfig(max_defer=1.0, expensive_percentile=0.1, percentile_window=2),
    )

    deferred = run_allocation(deferred_config, prices, rising_fx, cpi)
    plain = run_allocation(_allocation_config(PolicyId.S0_GLOBAL), prices, rising_fx, cpi)

    assert deferred.snapshots[-1].cash_krw > plain.snapshots[-1].cash_krw
