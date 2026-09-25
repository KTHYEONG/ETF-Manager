"""Unit tests for the after-tax accumulation engine (spec 3/4 invariant scenarios)."""

from __future__ import annotations

from dataclasses import replace as _replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from src.data.calendar import load_calendar
from src.data.pipeline import ingest
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.policy.targets import PolicyId
from src.policy.weight_rule import CASH_SLEEVE
from src.sim.after_tax_engine import (
    AfterTaxConfig,
    AfterTaxDataError,
    ExecutionMode,
    _check_scheduled_cash_before_final_execution,
    run_after_tax,
    run_after_tax_from_store,
)
from src.sim.allocation import AllocationConfig, run_allocation
from src.sim.tax import BasisMethod, load_tax_regime

_CALENDAR = load_calendar("XNYS")
_RETRIEVED_AT = datetime(2024, 4, 1, 5, 0, tzinfo=UTC)
_REGIME = load_tax_regime("configs/tax/kr_overseas_equity.json")
_MA_REGIME = _replace(_REGIME, basis_method=BasisMethod.MOVING_AVERAGE)


class _ReadingRule:
    tickers = frozenset({"QQQ"})
    requires_cash_rate = False

    def __call__(self, signal_at: datetime, market: Any) -> dict[str, float]:
        market.month_end_adjusted_closes("QQQ", signal_at, 1)
        return {"QQQ": 1.0}


def _sessions(start: date, end: date) -> tuple[date, ...]:
    return _CALENDAR.sessions(start, end)


def _prices_frame(
    days: tuple[date, ...],
    closes_by_ticker: dict[str, list[float]],
    *,
    dividends_by_ticker: dict[str, list[float]] | None = None,
    splits_by_ticker: dict[str, list[float]] | None = None,
) -> pl.DataFrame:
    spec = spec_for(Dataset.PRICES)
    tickers: list[str] = []
    dates: list[date] = []
    closes: list[float] = []
    dividends: list[float] = []
    splits: list[float] = []
    for ticker in sorted(closes_by_ticker):
        series = closes_by_ticker[ticker]
        ticker_dividends = (dividends_by_ticker or {}).get(ticker, [0.0] * len(days))
        ticker_splits = (splits_by_ticker or {}).get(ticker, [1.0] * len(days))
        for day, close, dividend, split in zip(days, series, ticker_dividends, ticker_splits, strict=True):
            tickers.append(ticker)
            dates.append(day)
            closes.append(close)
            dividends.append(dividend)
            splits.append(split)
    n = len(dates)
    return ingest(
        pl.DataFrame(
            {
                "ticker": tickers,
                "date": dates,
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [10_000] * n,
                "adjusted_close": closes,
                "dividend": dividends,
                "split_factor": splits,
                "source": ["synthetic"] * n,
                "retrieved_at": [_RETRIEVED_AT] * n,
            },
            schema=dict(spec.columns),
        ),
        Dataset.PRICES,
    )


def _fx_frame(days: tuple[date, ...], rate: float = 1300.0, *, legacy: bool = False) -> pl.DataFrame:
    dataset = Dataset.FX if legacy else Dataset.FX_KRW_BASE
    spec = spec_for(dataset)
    return ingest(
        pl.DataFrame(
            {
                "date": list(days),
                "usdkrw": [rate] * len(days),
                "source": ["synthetic"] * len(days),
                "retrieved_at": [_RETRIEVED_AT] * len(days),
            },
            schema=dict(spec.columns),
        ),
        dataset,
    )


def _cpi_frame() -> pl.DataFrame:
    spec = spec_for(Dataset.CPI)
    return ingest(
        pl.DataFrame(
            {
                "period_end": [date(2023, 12, 1), date(2024, 6, 1)],
                "value": [100.0, 100.0],
                "source": ["synthetic", "synthetic"],
                "retrieved_at": [_RETRIEVED_AT, _RETRIEVED_AT],
            },
            schema=dict(spec.columns),
        ),
        Dataset.CPI,
    )


def _rates_frame(observations: list[tuple[date, float | None]], series: str = "DTB3") -> pl.DataFrame:
    spec = spec_for(Dataset.RATES)
    return ingest(
        pl.DataFrame(
            {
                "series_id": [series] * len(observations),
                "observation_date": [day for day, _ in observations],
                "value": [value for _, value in observations],
                "source": ["synthetic"] * len(observations),
                "retrieved_at": [_RETRIEVED_AT] * len(observations),
            },
            schema=dict(spec.columns),
        ),
        Dataset.RATES,
    )


def _config(**overrides: Any) -> AfterTaxConfig:
    values: dict[str, Any] = {
        "start": date(2024, 1, 15),
        "end": date(2024, 12, 31),
        "monthly_contribution_krw": 1_300_000.0,
        "tax_regime": _REGIME,
        "targets": {"QQQ": 1.0},
    }
    values.update(overrides)
    return AfterTaxConfig(**values)  # type: ignore[arg-type]


def _book_basis_krw(result: Any, total_contributed: float, fx: float = 1300.0) -> float:
    """Terminal lot basis from ledger conservation (constant FX, zero friction).

    Contributions plus realized gains (market inflows through sales) either sit in
    earmarks or remain as lot basis; taxes and fees are zero in the calling runs.
    """
    gains = sum(disposal.gain_krw for disposal in result.disposals)
    return total_contributed + gains - result.snapshots[-1].cash_usd * fx


def test_legacy_parity_single_sleeve_adjusted_mode() -> None:
    """Adjusted mode without tax or harvest reproduces the legacy engine within 1e-4."""
    window = _sessions(date(2024, 1, 2), date(2025, 12, 31))
    closes = [100.0 + 0.05 * index for index in range(len(window))]
    prices = _prices_frame(window, {"QQQ": closes})
    fx_base = _fx_frame(window)
    fx_legacy = _fx_frame(window, legacy=True)
    cpi = _cpi_frame()
    result = run_after_tax(
        _config(price_mode="adjusted", tax_enabled=False, harvest_gains=False, label="parity"),
        prices,
        fx_base,
        cpi,
    )
    legacy = run_allocation(
        AllocationConfig(
            policy=PolicyId.QQQ,
            start=date(2024, 1, 15),
            end=date(2024, 12, 31),
            monthly_contribution_krw=1_300_000.0,
            targets_override={"QQQ": 1.0},
        ),
        prices,
        fx_legacy,
        cpi,
    )
    assert result.sell_count == 0
    assert result.terminal_after_tax_krw == pytest.approx(legacy.terminal_wealth_krw, rel=1e-4)


def test_earmark_accumulates_unaffordable_sleeve() -> None:
    """A 10% sleeve above one month's budget buys only once its earmark covers a share."""
    window = _sessions(date(2024, 1, 2), date(2024, 12, 31))
    prices = _prices_frame(
        window, {"QQQ": [100.0] * len(window), "SOXX": [700.0] * len(window), "SPY": [50.0] * len(window)}
    )
    result = run_after_tax(
        _config(
            end=date(2024, 9, 30),
            targets={"QQQ": 0.9, "SOXX": 0.1, "SPY": 0.0},
            tax_enabled=False,
            harvest_gains=False,
        ),
        prices,
        _fx_frame(window),
        _cpi_frame(),
    )
    assert len(result.snapshots) == 9
    for snapshot in result.snapshots[:6]:
        assert snapshot.shares.get("SOXX", 0.0) == 0.0
    assert result.snapshots[-1].shares.get("SOXX", 0.0) == 1.0
    for index, snapshot in enumerate(result.snapshots):
        funding = (index + 1) * 1_300_000.0 * 0.1
        assert snapshot.shares.get("SOXX", 0.0) * 700.0 * 1300.0 <= funding + 1.0


def test_buy_only_never_sells_without_harvest() -> None:
    """Buy-only with harvesting off never disposes; share counts never decrease."""
    window = _sessions(date(2024, 1, 2), date(2026, 6, 30))
    prices = _prices_frame(window, {"QQQ": [100.0] * len(window), "SPY": [50.0] * len(window)})
    result = run_after_tax(
        _config(end=date(2025, 12, 31), targets={"QQQ": 0.6, "SPY": 0.4}, harvest_gains=False),
        prices,
        _fx_frame(window),
        _cpi_frame(),
    )
    assert result.sell_count == 0
    assert result.disposals == ()
    previous = {"QQQ": 0.0, "SPY": 0.0}
    for snapshot in result.snapshots:
        for ticker in ("QQQ", "SPY"):
            assert snapshot.shares.get(ticker, 0.0) >= previous[ticker]
            previous[ticker] = snapshot.shares.get(ticker, 0.0)


def test_harvest_stays_tax_free() -> None:
    """December harvesting keeps every year's net gain within the deduction."""

    class _TwoSleeve:
        tickers = frozenset({"QQQ", "SPY"})
        requires_cash_rate = False

        def __call__(self, signal_at: datetime, market: Any) -> dict[str, float]:
            return {"QQQ": 0.5, "SPY": 0.5}

    window = _sessions(date(2024, 1, 2), date(2026, 6, 30))
    qqq = [100.0 * (1.008 ** (index / 21)) for index in range(len(window))]
    spy = [50.0 * (1.006 ** (index / 21)) for index in range(len(window))]
    prices = _prices_frame(window, {"QQQ": qqq, "SPY": spy})
    fx = _fx_frame(window)
    cpi = _cpi_frame()
    plain = _config(
        end=date(2025, 12, 31),
        monthly_contribution_krw=2_000_000.0,
        targets=None,
        rule=_TwoSleeve(),
        harvest_gains=False,
    )
    harvested = _config(
        end=date(2025, 12, 31),
        monthly_contribution_krw=2_000_000.0,
        targets=None,
        rule=_TwoSleeve(),
        harvest_gains=True,
    )
    without = run_after_tax(plain, prices, fx, cpi)
    with_harvest = run_after_tax(harvested, prices, fx, cpi)
    net_by_year: dict[int, float] = {}
    for disposal in with_harvest.disposals:
        net_by_year[disposal.tax_year] = net_by_year.get(disposal.tax_year, 0.0) + disposal.gain_krw
    assert net_by_year
    for net in net_by_year.values():
        assert net <= 2_500_000.0 + 1e-6
    assert with_harvest.taxes_paid_krw == 0.0
    assert with_harvest.sell_count > 0
    assert with_harvest.terminal_after_tax_krw > without.terminal_after_tax_krw


def test_harvest_resets_basis_upward() -> None:
    """The harvest run carries a higher terminal lot basis than the no-harvest run."""
    window = _sessions(date(2024, 1, 2), date(2026, 6, 30))
    qqq = [100.0 * (1.008 ** (index / 21)) for index in range(len(window))]
    prices = _prices_frame(window, {"QQQ": qqq})
    fx = _fx_frame(window)
    cpi = _cpi_frame()
    kwargs: dict[str, Any] = {
        "end": date(2025, 12, 31),
        "monthly_contribution_krw": 2_000_000.0,
        "targets": {"QQQ": 1.0},
    }
    without = run_after_tax(_config(harvest_gains=False, **kwargs), prices, fx, cpi)
    with_harvest = run_after_tax(_config(harvest_gains=True, **kwargs), prices, fx, cpi)
    total = 24 * 2_000_000.0
    assert _book_basis_krw(with_harvest, total) > _book_basis_krw(without, total)


def test_december_settlement_cross_year_skips_harvest() -> None:
    """A December execution settling in January harvests nothing that month."""
    window = _sessions(date(2024, 1, 2), date(2025, 6, 30))
    closes = [100.0 + 0.5 * index for index in range(len(window))]
    prices = _prices_frame(window, {"QQQ": closes})
    result = run_after_tax(
        _config(
            start=date(2024, 11, 1),
            end=date(2024, 12, 31),
            fill_delay_sessions=21,
            harvest_gains=True,
        ),
        prices,
        _fx_frame(window),
        _cpi_frame(),
    )
    executions = [snapshot.session for snapshot in result.snapshots]
    assert date(2024, 12, 31) in executions
    assert result.disposals == ()


def test_tax_assessed_and_paid_in_may() -> None:
    """A 2025 switch sale is assessed for 2025 and paid from May 2026 contributions."""

    class _SwitchRule:
        tickers = frozenset({"QQQ", "SPY"})
        requires_cash_rate = False

        def __call__(self, signal_at: datetime, market: Any) -> dict[str, float]:
            return {"SPY": 1.0} if signal_at.year >= 2025 else {"QQQ": 1.0}

    window = _sessions(date(2024, 1, 2), date(2026, 12, 31))
    months_2024 = len([day for day in window if day.year == 2024])
    qqq = [100.0 + (300.0 / months_2024) * min(index, months_2024) for index in range(len(window))]
    spy = [50.0] * len(window)
    prices = _prices_frame(window, {"QQQ": qqq, "SPY": spy})
    result = run_after_tax(
        _config(
            end=date(2026, 8, 31),
            monthly_contribution_krw=2_000_000.0,
            targets=None,
            rule=_SwitchRule(),
            mode=ExecutionMode.REBALANCE_BAND,
            rebalance_band=0.05,
            harvest_gains=True,
        ),
        prices,
        _fx_frame(window),
        _cpi_frame(),
    )
    gain_2025 = sum(disposal.gain_krw for disposal in result.disposals if disposal.tax_year == 2025)
    assert gain_2025 > 2_500_000.0
    expected = 0.22 * (gain_2025 - 2_500_000.0)
    assert result.taxes_paid_krw == pytest.approx(expected, rel=1e-9)
    assert result.snapshots[-1].tax_payable_krw == 0.0
    paid_sessions = [
        snapshot.session
        for previous, snapshot in zip(result.snapshots[:-1], result.snapshots[1:], strict=True)
        if snapshot.taxes_paid_krw > previous.taxes_paid_krw
    ]
    assert paid_sessions
    assert all(session >= date(2026, 5, 1) for session in paid_sessions)
    assert len(paid_sessions) > 1


def test_dividend_withholding() -> None:
    """A $1 dividend on 10 shares credits $8.50 and books 13,000 KRW income."""
    window = _sessions(date(2024, 1, 2), date(2025, 6, 30))
    dividends = [1.0 if day == date(2024, 2, 15) else 0.0 for day in window]
    spy_dividends = [0.5 if day == date(2024, 2, 15) else 0.0 for day in window]
    prices = _prices_frame(
        window,
        {"QQQ": [100.0] * len(window), "SPY": [50.0] * len(window)},
        dividends_by_ticker={"QQQ": dividends, "SPY": spy_dividends},
    )
    result = run_after_tax(
        _config(targets={"QQQ": 1.0}, fractional_shares=True, tax_enabled=True, harvest_gains=False),
        prices,
        _fx_frame(window),
        _cpi_frame(),
    )
    assert result.snapshots[1].shares.get("QQQ", 0.0) == pytest.approx(20.085)
    assert "SPY" not in result.snapshots[1].shares
    assert result.annual_financial_income_krw.get(2024, 0.0) == pytest.approx(13_000.0)


def test_split_preserves_wealth() -> None:
    """A 3:1 split triples shares with unchanged basis and no NAV jump."""
    window = _sessions(date(2024, 1, 2), date(2024, 12, 31))
    ex = date(2024, 2, 15)
    raws = [300.0 if day < ex else 100.0 for day in window]
    splits = [3.0 if day == ex else 1.0 for day in window]
    spec = spec_for(Dataset.PRICES)
    frame = pl.DataFrame(
        {
            "ticker": ["QQQ"] * len(window),
            "date": list(window),
            "open": raws,
            "high": raws,
            "low": raws,
            "close": raws,
            "volume": [10_000] * len(window),
            "adjusted_close": [300.0] * len(window),
            "dividend": [0.0] * len(window),
            "split_factor": splits,
            "source": ["synthetic"] * len(window),
            "retrieved_at": [_RETRIEVED_AT] * len(window),
        },
        schema=dict(spec.columns),
    )
    prices = ingest(frame, Dataset.PRICES)
    result = run_after_tax(
        _config(end=date(2024, 3, 29), fractional_shares=True, tax_enabled=False, harvest_gains=False),
        prices,
        _fx_frame(window),
        _cpi_frame(),
    )
    assert result.snapshots[0].shares.get("QQQ", 0.0) == pytest.approx(1300000.0 / 1300.0 / 300.0)
    assert result.snapshots[1].shares.get("QQQ", 0.0) == pytest.approx(20.0)
    assert result.snapshots[1].nav_krw - result.snapshots[0].nav_krw == pytest.approx(1_300_000.0)
    assert result.snapshots[2].shares.get("QQQ", 0.0) == pytest.approx(30.0)
    assert _book_basis_krw(result, 3 * 1_300_000.0) == pytest.approx(3 * 1_300_000.0)


def test_reverse_split_cash_in_lieu() -> None:
    """A 1:2 split on 7 integer shares keeps 3 and disposes 0.5 for cash."""
    window = _sessions(date(2024, 1, 2), date(2024, 12, 31))
    ex = date(2024, 2, 15)
    raws = [100.0 if day < ex else 200.0 for day in window]
    splits = [0.5 if day == ex else 1.0 for day in window]
    prices = _prices_frame(window, {"QQQ": raws}, splits_by_ticker={"QQQ": splits})
    result = run_after_tax(
        _config(
            end=date(2024, 3, 29),
            monthly_contribution_krw=1_000_000.0,
            tax_enabled=False,
            harvest_gains=False,
        ),
        prices,
        _fx_frame(window),
        _cpi_frame(),
    )
    assert len(result.disposals) == 1
    assert result.disposals[0].quantity == pytest.approx(0.5)
    assert result.snapshots[0].shares.get("QQQ", 0.0) == 7.0
    assert result.snapshots[1].shares.get("QQQ", 0.0) == 7.0


def test_after_tax_nav_bounds() -> None:
    """After-tax NAV never exceeds NAV; without tax or costs they coincide."""
    window = _sessions(date(2024, 1, 2), date(2026, 6, 30))
    closes = [100.0 * (1.005 ** (index / 21)) for index in range(len(window))]
    prices = _prices_frame(window, {"QQQ": closes})
    fx = _fx_frame(window)
    cpi = _cpi_frame()
    taxed = run_after_tax(_config(end=date(2025, 12, 31), harvest_gains=True), prices, fx, cpi)
    for snapshot in taxed.snapshots:
        assert snapshot.after_tax_nav_krw <= snapshot.nav_krw
    clean = run_after_tax(
        _config(end=date(2025, 12, 31), tax_enabled=False, harvest_gains=False),
        prices,
        fx,
        cpi,
    )
    for snapshot in clean.snapshots:
        assert snapshot.after_tax_nav_krw == snapshot.nav_krw


def test_rebalance_band_triggers_sells() -> None:
    """An 8pp drift past a 0.05 band disposes and restores drift within band."""
    window = _sessions(date(2024, 1, 2), date(2025, 6, 30))
    qqq = [100.0 * (2.0 ** (index / (len(window) - 1))) for index in range(len(window))]
    prices = _prices_frame(window, {"QQQ": qqq, "SPY": [50.0] * len(window)})
    result = run_after_tax(
        _config(
            monthly_contribution_krw=5_000_000.0,
            targets={"QQQ": 0.5, "SPY": 0.5},
            mode=ExecutionMode.REBALANCE_BAND,
            rebalance_band=0.05,
            tax_enabled=False,
            harvest_gains=False,
        ),
        prices,
        _fx_frame(window),
        _cpi_frame(),
    )
    assert result.sell_count > 0
    last = result.snapshots[-1]
    day = last.session
    total = last.nav_krw
    drift = max(
        abs(last.shares.get(ticker, 0.0) * (qqq[window.index(day)] if ticker == "QQQ" else 50.0) * 1300.0 / total - 0.5)
        for ticker in ("QQQ", "SPY")
    )
    assert drift <= 0.05 + 1e-6


def test_rebalance_exit_of_sleeve_drains_earmark_and_caps_sale_at_holdings() -> None:
    """Dropping a sleeve to 0% recovers its unspent earmark first and never sells beyond held shares."""

    class _ExitRule:
        tickers = frozenset({"QQQ", "SPY"})
        requires_cash_rate = False

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, signal_at: datetime, market: Any) -> dict[str, float]:
            self.calls += 1
            return {"QQQ": 0.5, "SPY": 0.5} if self.calls <= 3 else {"QQQ": 1.0}

    window = _sessions(date(2024, 1, 2), date(2024, 12, 31))
    # SPY gaps down right before the exit so last month's unspent earmark buys more than a share.
    spy = [400.0 if day < date(2024, 5, 1) else 150.0 for day in window]
    prices = _prices_frame(window, {"QQQ": [100.0] * len(window), "SPY": spy})
    result = run_after_tax(
        _config(
            end=date(2024, 11, 29),
            monthly_contribution_krw=1_300_000.0,
            targets=None,
            rule=_ExitRule(),
            mode=ExecutionMode.REBALANCE_BAND,
            rebalance_band=0.05,
            tax_enabled=False,
            harvest_gains=False,
        ),
        prices,
        _fx_frame(window),
        _cpi_frame(),
    )
    last = result.snapshots[-1]
    assert last.shares.get("SPY", 0.0) == 0.0
    assert result.sell_count > 0
    assert last.after_tax_nav_krw <= last.nav_krw


def test_cash_sleeve_is_redeployed_when_rule_returns_to_risk_assets() -> None:
    """After a risk-off spell the parked cash must flow back into the risk sleeve, not idle forever."""

    class _RoundTripRule:
        tickers = frozenset({"QQQ"})
        requires_cash_rate = True

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, signal_at: datetime, market: Any) -> dict[str, float]:
            self.calls += 1
            return {CASH_SLEEVE: 1.0} if self.calls <= 3 else {"QQQ": 1.0}

    window = _sessions(date(2024, 1, 2), date(2024, 11, 29))
    prices = _prices_frame(window, {"QQQ": [100.0] * len(window)})
    result = run_after_tax(
        _config(
            end=date(2024, 10, 31),
            targets=None,
            rule=_RoundTripRule(),
            mode=ExecutionMode.REBALANCE_BAND,
            rebalance_band=0.05,
            tax_enabled=False,
            harvest_gains=False,
            fractional_shares=True,
        ),
        prices,
        _fx_frame(window),
        _cpi_frame(),
        _rates_frame([(date(2024, 1, 2), 0.0)]),
    )
    parked = result.snapshots[2]
    last = result.snapshots[-1]
    assert parked.cash_usd == pytest.approx(3_900_000.0 / 1300.0)
    assert parked.shares.get("QQQ", 0.0) == 0.0
    assert last.cash_usd == pytest.approx(0.0, abs=1e-6)
    assert last.shares["QQQ"] * 100.0 * 1300.0 == pytest.approx(last.nav_krw)


def test_cash_sleeve_accrues_after_tax_interest() -> None:
    """A full-cash rule earns DTB3 5% over the exact day count net of interest tax."""

    class _CashRule:
        tickers = frozenset()
        requires_cash_rate = True

        def __call__(self, signal_at: datetime, market: Any) -> dict[str, float]:
            return {CASH_SLEEVE: 1.0}

    window = _sessions(date(2024, 1, 2), date(2024, 12, 31))
    prices = _prices_frame(window, {"QQQ": [100.0] * len(window)})
    rates = _rates_frame([(date(2024, 1, 2), 5.0), (date(2024, 1, 3), None), (date(2024, 3, 1), 6.0)])
    result = run_after_tax(
        _config(end=date(2024, 3, 29), targets=None, rule=_CashRule()),
        prices,
        _fx_frame(window),
        _cpi_frame(),
        rates,
    )
    assert len(result.snapshots) == 3
    first, second = result.snapshots[0], result.snapshots[1]
    monthly_usd = 1_300_000.0 / 1300.0
    assert first.cash_usd == pytest.approx(monthly_usd)
    days = (second.session - first.session).days
    expected = monthly_usd * (1.0 + 0.05 * days / 360.0 * (1.0 - 0.154)) + monthly_usd
    assert second.cash_usd == pytest.approx(expected, rel=1e-9)


def test_cash_without_rates_fails_closed() -> None:
    """A cash allocation without a RATES frame cannot accrue interest."""

    class _CashRule:
        tickers = frozenset()
        requires_cash_rate = True

        def __call__(self, signal_at: datetime, market: Any) -> dict[str, float]:
            return {CASH_SLEEVE: 1.0}

    window = _sessions(date(2024, 1, 2), date(2024, 12, 31))
    prices = _prices_frame(window, {"QQQ": [100.0] * len(window)})
    with pytest.raises(AfterTaxDataError):
        run_after_tax(
            _config(end=date(2024, 3, 29), targets=None, rule=_CashRule()),
            prices,
            _fx_frame(window),
            _cpi_frame(),
        )


def test_stale_fx_fails_closed() -> None:
    """No base rate within the staleness bound aborts the run."""
    window = _sessions(date(2024, 1, 2), date(2024, 12, 31))
    prices = _prices_frame(window, {"QQQ": [100.0] * len(window)})
    stale_fx = _fx_frame(_sessions(date(2024, 1, 2), date(2024, 1, 31)))
    with pytest.raises(AfterTaxDataError):
        run_after_tax(_config(start=date(2024, 6, 1)), prices, stale_fx, _cpi_frame())


def test_financial_income_disclosure() -> None:
    """Dividend income above the threshold flags the breach year."""
    window = _sessions(date(2024, 1, 2), date(2025, 6, 30))
    dividends = [2000.0 if day == date(2024, 2, 15) else 0.0 for day in window]
    prices = _prices_frame(window, {"QQQ": [100.0] * len(window)}, dividends_by_ticker={"QQQ": dividends})
    result = run_after_tax(
        _config(fractional_shares=True, harvest_gains=False),
        prices,
        _fx_frame(window),
        _cpi_frame(),
    )
    assert result.annual_financial_income_krw.get(2024, 0.0) == pytest.approx(10.0 * 2000.0 * 1300.0)
    assert 2024 in result.financial_income_breach_years


def test_lookahead_invariance() -> None:
    """Perturbing rows after the 12th execution leaves the first 12 snapshots identical."""
    window = _sessions(date(2024, 1, 2), date(2026, 6, 30))
    closes = [100.0 + 0.1 * index for index in range(len(window))]
    prices = _prices_frame(window, {"QQQ": closes})
    fx = _fx_frame(window)
    cpi = _cpi_frame()
    config = _config(targets=None, rule=_ReadingRule(), tax_enabled=False, harvest_gains=False)
    first_run = run_after_tax(config, prices, fx, cpi)
    pivot = first_run.snapshots[11].session
    perturbed = [close * 2.0 if day > pivot else close for day, close in zip(window, closes, strict=True)]
    disturbed = run_after_tax(config, _prices_frame(window, {"QQQ": perturbed}), fx, cpi)
    assert first_run.snapshots[:12] == disturbed.snapshots[:12]


def test_fifo_vs_moving_average_without_disposals() -> None:
    """Without disposals both basis methods trace identical after-tax paths."""
    window = _sessions(date(2024, 1, 2), date(2026, 6, 30))
    closes = [100.0 * (1.005 ** (index / 21)) for index in range(len(window))]
    prices = _prices_frame(window, {"QQQ": closes, "SPY": [50.0] * len(window)})
    fx = _fx_frame(window)
    cpi = _cpi_frame()
    kwargs: dict[str, Any] = {
        "end": date(2025, 12, 31),
        "targets": {"QQQ": 0.6, "SPY": 0.4},
        "harvest_gains": False,
    }
    fifo = run_after_tax(_config(**kwargs), prices, fx, cpi)
    average = run_after_tax(_config(tax_regime=_MA_REGIME, **kwargs), prices, fx, cpi)
    assert [snapshot.after_tax_nav_krw for snapshot in fifo.snapshots] == [
        snapshot.after_tax_nav_krw for snapshot in average.snapshots
    ]


def test_config_validation() -> None:
    """Contradictory targets, costs, and execution parameters fail fast."""
    window = _sessions(date(2024, 1, 2), date(2024, 6, 28))
    prices = _prices_frame(window, {"QQQ": [100.0] * len(window)})
    fx = _fx_frame(window)
    cpi = _cpi_frame()
    bad_kwargs: list[tuple[dict[str, Any], str]] = [
        ({"monthly_contribution_krw": 0.0}, "positive"),
        ({"monthly_contribution_krw": -5.0}, "positive"),
        ({"monthly_contribution_krw": float("nan")}, "positive"),
        ({"monthly_contribution_krw": "big"}, "number"),
        ({"targets": {"QQQ": 1.0}, "rule": _ReadingRule()}, "exactly one"),
        ({"targets": None}, "exactly one"),
        ({"targets": {"QQQ": 0.5}}, "sum to 1.0"),
        ({"targets": {"QQQ": 1.2, "SPY": -0.2}}, "nonnegative"),
        ({"mode": ExecutionMode.REBALANCE_BAND}, "numeric"),
        ({"mode": ExecutionMode.REBALANCE_BAND, "rebalance_band": 1.5}, "lie in"),
        ({"fill_delay_sessions": 0}, ">= 1"),
        ({"fx_max_staleness_days": -1}, ">= 0"),
        ({"price_mode": "nominal"}, "raw"),
        ({"commission_bps": -1.0}, "nonnegative"),
        ({"fx_spread_bps": float("inf")}, "nonnegative"),
    ]
    for kwargs, match in bad_kwargs:
        with pytest.raises(ValueError, match=match):
            run_after_tax(_config(end=date(2024, 6, 28), **kwargs), prices, fx, cpi)


def test_rule_non_simplex_rejected() -> None:
    """A rule returning a non-simplex fails with ValueError."""

    class _BadRule:
        tickers = frozenset({"QQQ"})
        requires_cash_rate = False

        def __call__(self, signal_at: datetime, market: Any) -> dict[str, float]:
            return {"QQQ": 0.5}

    window = _sessions(date(2024, 1, 2), date(2024, 6, 28))
    with pytest.raises(ValueError, match="weights"):
        run_after_tax(
            _config(end=date(2024, 6, 28), targets=None, rule=_BadRule()),
            _prices_frame(window, {"QQQ": [100.0] * len(window)}),
            _fx_frame(window),
            _cpi_frame(),
        )


def test_empty_schedule_fails_closed() -> None:
    """An inverted date range yields AfterTaxDataError, never an empty path."""
    window = _sessions(date(2024, 1, 2), date(2024, 6, 28))
    with pytest.raises(AfterTaxDataError, match="empty decision schedule"):
        run_after_tax(
            _config(start=date(2024, 6, 1), end=date(2024, 1, 1)),
            _prices_frame(window, {"QQQ": [100.0] * len(window)}),
            _fx_frame(window),
            _cpi_frame(),
        )


def test_missing_market_data_fails_closed() -> None:
    """Missing price rows, null FX, flat CPI, and ragged frames abort the run."""
    window = _sessions(date(2024, 1, 2), date(2024, 6, 28))
    prices = _prices_frame(window, {"QQQ": [100.0] * len(window)})
    fx = _fx_frame(window)
    cpi = _cpi_frame()
    thin = _prices_frame(window, {"SPY": [50.0] * len(window)})
    with pytest.raises(AfterTaxDataError, match="price row"):
        run_after_tax(_config(end=date(2024, 6, 28)), thin, fx, cpi)
    null_fx = _fx_frame(window).with_columns(pl.lit(None, dtype=pl.Float64()).alias("usdkrw"))
    with pytest.raises(AfterTaxDataError, match="usdkrw"):
        run_after_tax(_config(end=date(2024, 6, 28)), prices, null_fx, cpi)
    spec = spec_for(Dataset.CPI)
    future_cpi = ingest(
        pl.DataFrame(
            {
                "period_end": [date(2025, 1, 1)],
                "value": [100.0],
                "source": ["synthetic"],
                "retrieved_at": [_RETRIEVED_AT],
            },
            schema=dict(spec.columns),
        ),
        Dataset.CPI,
    )
    with pytest.raises(AfterTaxDataError, match="CPI"):
        run_after_tax(_config(end=date(2024, 6, 28)), prices, fx, future_cpi)
    ragged = prices.drop("close")
    with pytest.raises(AfterTaxDataError, match="columns"):
        run_after_tax(_config(end=date(2024, 6, 28)), ragged, fx, cpi)
    ragged_fx = fx.drop("usdkrw")
    with pytest.raises(AfterTaxDataError, match="columns"):
        run_after_tax(_config(end=date(2024, 6, 28)), prices, ragged_fx, cpi)
    ragged_cpi = cpi.drop("value")
    with pytest.raises(AfterTaxDataError, match="columns"):
        run_after_tax(_config(end=date(2024, 6, 28)), prices, fx, ragged_cpi)


def test_harvest_rebuy_limited_by_proceeds() -> None:
    """High commissions can strand harvest proceeds in the sleeve earmark."""
    window = _sessions(date(2024, 1, 2), date(2026, 6, 30))
    closes = [500.0 + 0.5 * index for index in range(len(window))]
    prices = _prices_frame(window, {"QQQ": closes})
    result = run_after_tax(
        _config(
            end=date(2025, 12, 31),
            monthly_contribution_krw=1_300_000.0,
            commission_bps=500.0,
            harvest_gains=True,
        ),
        prices,
        _fx_frame(window),
        _cpi_frame(),
    )
    assert result.sell_count > 0
    assert result.snapshots[-1].cash_usd > 0.0


def test_cash_wrong_series_fails_closed() -> None:
    """A cash sleeve naming an unloaded series cannot accrue."""

    class _CashRule:
        tickers = frozenset()
        requires_cash_rate = True

        def __call__(self, signal_at: datetime, market: Any) -> dict[str, float]:
            return {CASH_SLEEVE: 1.0}

    window = _sessions(date(2024, 1, 2), date(2024, 6, 28))
    prices = _prices_frame(window, {"QQQ": [100.0] * len(window)})
    rates = _rates_frame([(date(2024, 1, 2), 5.0)], series="OTHER")
    with pytest.raises(AfterTaxDataError, match="rate"):
        run_after_tax(
            _config(end=date(2024, 6, 28), targets=None, rule=_CashRule()),
            prices,
            _fx_frame(window),
            _cpi_frame(),
            rates,
        )


def test_run_after_tax_from_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Latest trusted partitions load and simulate; cash runs pull RATES too."""
    from src.data.catalog import latest_artifact
    from src.data.pipeline import persist_ingest
    from src.data.storage import RawPayload

    monkeypatch.chdir(tmp_path)
    settings = DataSettings(data_root="data")

    def _payload() -> RawPayload:
        return RawPayload(
            provider="synthetic",
            endpoint="probe",
            request_params={},
            retrieved_at=_RETRIEVED_AT,
            extension="json",
            content=b"{}",
        )

    window = _sessions(date(2024, 1, 2), date(2025, 6, 30))
    closes = [100.0 + 0.05 * index for index in range(len(window))]
    spec_prices = spec_for(Dataset.PRICES)
    raw_prices = pl.DataFrame(
        {
            "ticker": ["QQQ"] * len(window),
            "date": list(window),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [10_000] * len(window),
            "adjusted_close": closes,
            "dividend": [0.0] * len(window),
            "split_factor": [1.0] * len(window),
            "source": ["synthetic"] * len(window),
            "retrieved_at": [_RETRIEVED_AT] * len(window),
        },
        schema=dict(spec_prices.columns),
    )
    spec_fx = spec_for(Dataset.FX_KRW_BASE)
    raw_fx = pl.DataFrame(
        {
            "date": list(window),
            "usdkrw": [1300.0] * len(window),
            "source": ["synthetic"] * len(window),
            "retrieved_at": [_RETRIEVED_AT] * len(window),
        },
        schema=dict(spec_fx.columns),
    )
    spec_cpi = spec_for(Dataset.CPI)
    raw_cpi = pl.DataFrame(
        {
            "period_end": [date(2023, 12, 1)],
            "value": [100.0],
            "source": ["synthetic"],
            "retrieved_at": [_RETRIEVED_AT],
        },
        schema=dict(spec_cpi.columns),
    )
    for frame, dataset in ((raw_prices, Dataset.PRICES), (raw_fx, Dataset.FX_KRW_BASE), (raw_cpi, Dataset.CPI)):
        persist_ingest(frame, dataset, _payload(), settings)
    assert latest_artifact(settings, Dataset.PRICES).manifest.row_count > 0

    plain = run_after_tax_from_store(_config(label="store"), settings)
    assert len(plain.snapshots) == 12
    assert plain.terminal_after_tax_krw > 0

    class _CashRule:
        tickers = frozenset()
        requires_cash_rate = True

        def __call__(self, signal_at: datetime, market: Any) -> dict[str, float]:
            return {CASH_SLEEVE: 1.0}

    spec_rates = spec_for(Dataset.RATES)
    raw_rates = pl.DataFrame(
        {
            "series_id": ["DTB3"],
            "observation_date": [date(2024, 1, 2)],
            "value": [5.0],
            "source": ["synthetic"],
            "retrieved_at": [_RETRIEVED_AT],
        },
        schema=dict(spec_rates.columns),
    )
    persist_ingest(raw_rates, Dataset.RATES, _payload(), settings)
    cash = run_after_tax_from_store(_config(rule=_CashRule(), targets=None, label="cash"), settings)
    assert cash.snapshots[-1].cash_usd > 0

    with pytest.raises(AfterTaxDataError, match="empty decision schedule"):
        run_after_tax_from_store(_config(start=date(2024, 6, 1), end=date(2024, 1, 1)), settings)


def test_zero_price_fails_closed() -> None:
    """A non-positive mark aborts instead of minting infinite shares."""
    window = _sessions(date(2024, 1, 2), date(2024, 6, 28))
    prices = _prices_frame(window, {"QQQ": [100.0] * len(window)}).with_columns(
        pl.when(pl.col("date") >= date(2024, 3, 1))
        .then(0.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    with pytest.raises(AfterTaxDataError, match="non-positive close"):
        run_after_tax(
            _config(end=date(2024, 6, 28)),
            prices,
            _fx_frame(window),
            _cpi_frame(),
        )


def test_future_availability_fx_skipped() -> None:
    """An FX row stamped after the execution close is invisible, then stale."""
    spec = spec_for(Dataset.FX_KRW_BASE)
    fx = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 6, 3)],
            "usdkrw": [1300.0, 1300.0],
            "source": ["synthetic", "synthetic"],
            "retrieved_at": [_RETRIEVED_AT, _RETRIEVED_AT],
            "available_at": [
                datetime(2024, 1, 2, 12, 0, tzinfo=UTC),
                datetime(2025, 12, 31, 12, 0, tzinfo=UTC),
            ],
        },
        schema=dict(spec.columns, available_at=pl.Datetime("us", "UTC")),
    )
    window = _sessions(date(2024, 1, 2), date(2024, 12, 31))
    with pytest.raises(AfterTaxDataError, match="usdkrw"):
        run_after_tax(
            _config(start=date(2024, 6, 1), end=date(2024, 6, 28)),
            _prices_frame(window, {"QQQ": [100.0] * len(window)}),
            fx,
            _cpi_frame(),
        )


def test_rates_columns_validated() -> None:
    """A RATES frame without its value column fails before any accrual."""

    class _CashRule:
        tickers = frozenset()
        requires_cash_rate = True

        def __call__(self, signal_at: datetime, market: Any) -> dict[str, float]:
            return {CASH_SLEEVE: 1.0}

    window = _sessions(date(2024, 1, 2), date(2024, 6, 28))
    prices = _prices_frame(window, {"QQQ": [100.0] * len(window)})
    rates = _rates_frame([(date(2024, 1, 2), 5.0)]).drop("value")
    with pytest.raises(AfterTaxDataError, match="columns"):
        run_after_tax(_config(end=date(2024, 6, 28), targets=None, rule=_CashRule()), prices, _fx_frame(window), _cpi_frame(), rates)


def test_usd_flow_verification() -> None:
    """Balanced earmark flows pass; a KRW-scale drift raises."""
    from src.sim.after_tax_engine import _verify_usd_flows

    _verify_usd_flows(context="probe", before_usd=100.0, after_usd=110.0, inflows_usd=20.0, outflows_usd=10.0, fx=1300.0)
    with pytest.raises(AfterTaxDataError, match="conservation"):
        _verify_usd_flows(context="probe", before_usd=100.0, after_usd=120.0, inflows_usd=20.0, outflows_usd=10.0, fx=1300.0)


def test_band_with_cash_sleeve_rebalances_both_ways() -> None:
    """A 50/50 QQQ/CASH band sells rallies into cash and spends cash on drawdowns."""
    window = _sessions(date(2024, 1, 2), date(2025, 6, 30))
    closes = [100.0 if day < date(2024, 4, 1) else (300.0 if day < date(2024, 7, 1) else 150.0) for day in window]
    prices = _prices_frame(window, {"QQQ": closes})
    result = run_after_tax(
        _config(
            targets={"QQQ": 0.5, CASH_SLEEVE: 0.5},
            mode=ExecutionMode.REBALANCE_BAND,
            rebalance_band=0.0,
            tax_enabled=False,
            harvest_gains=False,
        ),
        prices,
        _fx_frame(window),
        _cpi_frame(),
        _rates_frame([(date(2024, 1, 2), 5.0)]),
    )
    assert result.sell_count > 0
    assert result.snapshots[-1].shares.get("QQQ", 0.0) > 0.0


def test_harvest_skips_dust_lot() -> None:
    """A crumb budget leaves a pricey share untouched while cheaper lots harvest."""

    class _StaticTrio:
        tickers = frozenset({"BIG", "HI", "QQQ"})
        requires_cash_rate = False

        def __call__(self, signal_at: datetime, market: Any) -> dict[str, float]:
            return {"BIG": 0.5, "HI": 0.475, "QQQ": 0.025}

    window = _sessions(date(2024, 9, 2), date(2025, 6, 30))
    big = [100.0 if day < date(2024, 12, 2) else 119.20 for day in window]
    hi = [9500.0 if day < date(2024, 12, 2) else 10000.0 for day in window]
    qqq = [100.0 if day < date(2024, 12, 2) else 100.50 for day in window]
    prices = _prices_frame(window, {"BIG": big, "HI": hi, "QQQ": qqq})
    result = run_after_tax(
        _config(
            start=date(2024, 10, 1),
            end=date(2024, 12, 31),
            monthly_contribution_krw=26_000_000.0,
            targets=None,
            rule=_StaticTrio(),
            harvest_gains=True,
        ),
        prices,
        _fx_frame(window),
        _cpi_frame(),
    )
    december_trades = [d for d in result.disposals if d.trade_session == date(2024, 12, 2)]
    assert december_trades
    assert all(d.ticker != "HI" for d in december_trades)
    assert result.snapshots[-1].shares.get("HI", 0.0) == 2.0


def test_harvest_exhaustion_stops_tickers() -> None:
    """A capped fractional sale ends harvesting without touching later sleeves."""

    class _StaticDuo:
        tickers = frozenset({"BIG", "HI"})
        requires_cash_rate = False

        def __call__(self, signal_at: datetime, market: Any) -> dict[str, float]:
            return {"BIG": 0.525, "HI": 0.475}

    window = _sessions(date(2024, 9, 2), date(2025, 6, 30))
    big = [100.0 if day < date(2024, 12, 2) else 200.0 for day in window]
    hi = [9500.0 if day < date(2024, 12, 2) else 10000.0 for day in window]
    prices = _prices_frame(window, {"BIG": big, "HI": hi})
    result = run_after_tax(
        _config(
            start=date(2024, 10, 1),
            end=date(2024, 12, 31),
            monthly_contribution_krw=26_000_000.0,
            targets=None,
            rule=_StaticDuo(),
            fractional_shares=True,
            harvest_gains=True,
        ),
        prices,
        _fx_frame(window),
        _cpi_frame(),
    )
    december_trades = [d for d in result.disposals if d.trade_session == date(2024, 12, 2)]
    assert len(december_trades) == 1
    assert december_trades[0].ticker == "BIG"
    assert result.sell_count == 1


def _decision_executions(start: date, end: date) -> tuple[date, ...]:
    from src.data.schedule import build_decision_schedule

    return tuple(
        point.execution_session
        for point in build_decision_schedule(start, end, frequency="monthly", fill_delay_sessions=1)
    )


def test_schedule_matching_monthly_reproduces_monthly_result() -> None:
    """A schedule paying the monthly amount on each signal month traces the monthly path."""
    from src.data.schedule import build_decision_schedule

    window = _sessions(date(2024, 1, 2), date(2025, 12, 31))
    closes = [100.0 + 0.05 * index for index in range(len(window))]
    prices = _prices_frame(window, {"QQQ": closes})
    fx = _fx_frame(window)
    cpi = _cpi_frame()
    monthly = run_after_tax(_config(), prices, fx, cpi)
    points = build_decision_schedule(date(2024, 1, 15), date(2024, 12, 31), frequency="monthly", fill_delay_sessions=1)
    scheduled = run_after_tax(
        _config(
            monthly_contribution_krw=0.0,
            contribution_schedule_krw={point.signal_session: 1_300_000.0 for point in points},
        ),
        prices,
        fx,
        cpi,
    )
    assert [snapshot.session for snapshot in scheduled.snapshots] == [
        snapshot.session for snapshot in monthly.snapshots
    ]
    assert scheduled.terminal_after_tax_krw == pytest.approx(monthly.terminal_after_tax_krw, rel=1e-9)
    assert scheduled.xirr_after_tax_real == pytest.approx(monthly.xirr_after_tax_real, rel=1e-9)


def test_scheduled_cash_waits_for_its_date() -> None:
    """A single mid-run deposit leaves earlier steps empty and lands on its execution."""
    window = _sessions(date(2024, 1, 2), date(2024, 12, 31))
    prices = _prices_frame(window, {"QQQ": [100.0] * len(window)})
    result = run_after_tax(
        _config(
            end=date(2024, 6, 28),
            monthly_contribution_krw=0.0,
            contribution_schedule_krw={date(2024, 3, 15): 6_000_000.0},
        ),
        prices,
        _fx_frame(window),
        _cpi_frame(),
    )
    executions = _decision_executions(date(2024, 1, 15), date(2024, 6, 28))
    expected_session = min(session for session in executions if session >= date(2024, 3, 15))
    funded = [snapshot for snapshot in result.snapshots if snapshot.contribution_krw > 0]
    assert len(funded) == 1
    assert funded[0].session == expected_session
    assert funded[0].contribution_krw == 6_000_000.0
    for snapshot in result.snapshots:
        if snapshot.session < expected_session:
            assert snapshot.contribution_krw == 0.0
            assert all(shares == 0.0 for shares in snapshot.shares.values())


def test_snapshot_contributions_sum_to_schedule_total() -> None:
    """Irregular January/May deposits over two years reconcile exactly with snapshots."""
    window = _sessions(date(2024, 1, 2), date(2026, 6, 30))
    closes = [100.0 + 0.05 * index for index in range(len(window))]
    prices = _prices_frame(window, {"QQQ": closes})
    funding = {
        date(2024, 1, 20): 6_000_000.0,
        date(2024, 5, 31): 990_000.0,
        date(2025, 1, 15): 6_000_000.0,
        date(2025, 5, 31): 990_000.0,
    }
    result = run_after_tax(
        _config(
            end=date(2025, 12, 31),
            monthly_contribution_krw=0.0,
            contribution_schedule_krw=funding,
        ),
        prices,
        _fx_frame(window),
        _cpi_frame(),
    )
    assert sum(snapshot.contribution_krw for snapshot in result.snapshots) == sum(funding.values())


def test_schedule_and_monthly_are_mutually_exclusive() -> None:
    """A schedule requires monthly zero; monthly zero requires a schedule."""
    window = _sessions(date(2024, 1, 2), date(2024, 6, 28))
    prices = _prices_frame(window, {"QQQ": [100.0] * len(window)})
    fx = _fx_frame(window)
    cpi = _cpi_frame()
    with pytest.raises(ValueError, match="when contribution_schedule_krw is set"):
        run_after_tax(
            _config(
                end=date(2024, 6, 28),
                monthly_contribution_krw=1_000_000.0,
                contribution_schedule_krw={date(2024, 2, 1): 1_000_000.0},
            ),
            prices,
            fx,
            cpi,
        )
    with pytest.raises(ValueError, match="positive"):
        run_after_tax(_config(end=date(2024, 6, 28), monthly_contribution_krw=0.0), prices, fx, cpi)


def test_invalid_schedule_entries_rejected() -> None:
    """Empty, non-positive, or misdated schedules fail closed with the offender named."""
    window = _sessions(date(2024, 1, 2), date(2024, 6, 28))
    prices = _prices_frame(window, {"QQQ": [100.0] * len(window)})
    fx = _fx_frame(window)
    cpi = _cpi_frame()
    bad_schedules: list[tuple[dict[Any, Any], str]] = [
        ({}, "non-empty"),
        ({date(2024, 2, 1): 0.0}, "positive"),
        ({date(2024, 2, 1): -100.0}, "positive"),
        ({date(2024, 2, 1): float("nan")}, "positive"),
        ({date(2024, 2, 1): float("inf")}, "positive"),
        ({date(2024, 2, 1): True}, "positive"),
        ({"2024-02-01": 1_000_000.0}, "must be a date"),
        ({date(2024, 1, 1): 1_000_000.0}, "outside"),
        ({date(2024, 7, 1): 1_000_000.0}, "outside"),
    ]
    for funding, match in bad_schedules:
        with pytest.raises(ValueError, match=match):
            run_after_tax(
                _config(end=date(2024, 6, 28), monthly_contribution_krw=0.0, contribution_schedule_krw=funding),
                prices,
                fx,
                cpi,
            )


def test_cash_after_final_execution_fails_closed() -> None:
    """Cash dated after the final execution session can never be invested.

    The monthly schedule always executes past ``end``, so a post-execution date
    within ``[start, end]`` cannot arise end-to-end; the guard is exercised
    directly, while cash dated on ``end`` itself still invests at the final fill.
    """
    with pytest.raises(AfterTaxDataError, match="falls after the final execution session"):
        _check_scheduled_cash_before_final_execution({date(2024, 7, 10): 1_000_000.0}, date(2024, 7, 1))
    _check_scheduled_cash_before_final_execution({date(2024, 7, 1): 1_000_000.0}, date(2024, 7, 1))
    window = _sessions(date(2024, 1, 2), date(2024, 7, 31))
    result = run_after_tax(
        _config(
            end=date(2024, 7, 15),
            monthly_contribution_krw=0.0,
            contribution_schedule_krw={date(2024, 7, 15): 1_000_000.0},
        ),
        _prices_frame(window, {"QQQ": [100.0] * len(window)}),
        _fx_frame(window),
        _cpi_frame(),
    )
    assert sum(snapshot.contribution_krw for snapshot in result.snapshots) == 1_000_000.0


def test_run_after_tax_from_store_settlement_fx_remains_pinned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A concurrent FX ingest after resolve leaves settlement-cutoff reads pinned."""
    import src.sim.after_tax_engine as _engine
    from src.data.catalog import resolve_snapshot as _real_resolve
    from src.data.pipeline import persist_ingest
    from src.data.schema import Dataset as _Dataset
    from src.data.storage import RawPayload as _RawPayload

    monkeypatch.chdir(tmp_path)
    settings = DataSettings(data_root="data")

    def _payload() -> _RawPayload:
        return _RawPayload(
            provider="synthetic",
            endpoint="probe",
            request_params={},
            retrieved_at=_RETRIEVED_AT,
            extension="json",
            content=b"{}",
        )

    window = _sessions(date(2024, 1, 2), date(2024, 7, 31))
    closes = [100.0] * len(window)
    spec_prices = spec_for(Dataset.PRICES)
    persist_ingest(
        pl.DataFrame(
            {
                "ticker": ["QQQ"] * len(window),
                "date": list(window),
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [10_000] * len(window),
                "adjusted_close": closes,
                "dividend": [0.0] * len(window),
                "split_factor": [1.0] * len(window),
                "source": ["synthetic"] * len(window),
                "retrieved_at": [_RETRIEVED_AT] * len(window),
            },
            schema=dict(spec_prices.columns),
        ),
        Dataset.PRICES,
        _payload(),
        settings,
    )
    spec_fx = spec_for(Dataset.FX_KRW_BASE)
    raw_fx = pl.DataFrame(
        {
            "date": list(window),
            "usdkrw": [1300.0] * len(window),
            "source": ["synthetic"] * len(window),
            "retrieved_at": [_RETRIEVED_AT] * len(window),
        },
        schema=dict(spec_fx.columns),
    )
    persist_ingest(raw_fx, Dataset.FX_KRW_BASE, _payload(), settings)
    spec_cpi = spec_for(Dataset.CPI)
    persist_ingest(
        pl.DataFrame(
            {
                "period_end": [date(2023, 12, 1)],
                "value": [100.0],
                "source": ["synthetic"],
                "retrieved_at": [_RETRIEVED_AT],
            },
            schema=dict(spec_cpi.columns),
        ),
        Dataset.CPI,
        _payload(),
        settings,
    )
    config = _config(end=date(2024, 6, 28))
    expected = _engine.run_after_tax_from_store(config, settings)

    def _resolving_then_publishing(inner_settings: object, datasets: tuple[_Dataset, ...]) -> object:
        snapshot = _real_resolve(inner_settings, datasets)  # type: ignore[arg-type]
        raced = raw_fx.with_columns(pl.lit(9999.0).alias("usdkrw"))
        persist_ingest(
            raced,
            Dataset.FX_KRW_BASE,
            _RawPayload(
                provider="synthetic",
                endpoint="probe",
                request_params={},
                retrieved_at=datetime(2024, 8, 1, 5, 0, tzinfo=UTC),
                extension="json",
                content=b"{}",
            ),
            inner_settings,  # type: ignore[arg-type]
        )
        return snapshot

    monkeypatch.setattr(_engine, "resolve_snapshot", _resolving_then_publishing)
    actual = _engine.run_after_tax_from_store(config, settings)
    assert actual.terminal_after_tax_krw == pytest.approx(expected.terminal_after_tax_krw, rel=1e-9)


def test_market_indexes_match_engine_references() -> None:
    """Extracted indexes are the engine's single market-read path."""
    import src.sim.after_tax_engine as engine_module
    from src.sim.after_tax_market import _CpiIndex, _FxIndex, _PriceIndex, _RateIndex

    assert engine_module._PriceIndex is _PriceIndex
    assert engine_module._FxIndex is _FxIndex
    assert engine_module._CpiIndex is _CpiIndex
    assert engine_module._RateIndex is _RateIndex
    assert engine_module.AfterTaxDataError is not None


def test_market_indexes_resolve_pinned_execution_marks() -> None:
    """Index lookups at an execution instant return the pinned marks."""
    from src.sim.after_tax_market import _CpiIndex, _FxIndex, _PriceIndex, _RateIndex

    days = _sessions(date(2024, 1, 2), date(2024, 3, 29))
    prices = _prices_frame(days, {"QQQ": [100.0] * len(days)})
    fx = _fx_frame(days)
    cpi = _cpi_frame()
    rates = _rates_frame([(days[0], 0.04)])
    instant = _CALENDAR.close_ts(days[-1])

    price_index = _PriceIndex(prices)
    assert price_index.price("QQQ", days[-1], instant, adjusted=False) == pytest.approx(100.0)
    assert price_index.price("QQQ", days[-1], instant, adjusted=True) == pytest.approx(100.0)
    assert _FxIndex(fx).resolve(days[-1], instant, 7) == pytest.approx(1300.0)
    assert _CpiIndex(cpi).resolve(instant) == pytest.approx(100.0)
    junk_row = pl.DataFrame(
        {
            "period_end": [date(2024, 1, 15)],
            "value": [-5.0],
            "source": ["synthetic"],
            "retrieved_at": [_RETRIEVED_AT],
            "available_at": [datetime(2024, 2, 29, tzinfo=UTC)],
        },
        schema=cpi.schema,
    )
    junk_cpi = pl.concat([cpi, junk_row])
    assert _CpiIndex(junk_cpi).resolve(instant) == pytest.approx(100.0)
    assert _RateIndex(rates).resolve("DTB3", instant) == pytest.approx(0.04)


def test_market_indexes_reject_late_and_missing_marks() -> None:
    """Marks unavailable at the settlement instant fail closed."""
    from src.sim.after_tax_market import _CpiIndex, _FxIndex, _PriceIndex, _RateIndex

    days = _sessions(date(2024, 1, 2), date(2024, 3, 29))
    prices = _prices_frame(days, {"QQQ": [100.0] * len(days)})
    fx = _fx_frame(days)
    instant = _CALENDAR.close_ts(days[-1])

    early_instant = _CALENDAR.close_ts(days[0])
    with pytest.raises(AfterTaxDataError):
        _PriceIndex(prices).price("QQQ", days[-1], early_instant, adjusted=False)
    with pytest.raises(AfterTaxDataError):
        _PriceIndex(prices).price("MISSING", days[-1], instant, adjusted=False)

    late_fx = fx.with_columns(pl.col("available_at") + pl.duration(days=60))
    with pytest.raises(AfterTaxDataError):
        _FxIndex(late_fx).resolve(days[-1], instant, 7)
    with pytest.raises(AfterTaxDataError):
        _FxIndex(fx).resolve(date(2024, 6, 28), instant, 7)

    emptied_cpi = _cpi_frame().filter(pl.col("value") > 1e18)
    with pytest.raises(AfterTaxDataError):
        _CpiIndex(emptied_cpi).resolve(instant)

    with pytest.raises(AfterTaxDataError):
        _RateIndex(None).resolve("DTB3", instant)
    with pytest.raises(AfterTaxDataError):
        _RateIndex(_rates_frame([(days[0], 0.04)])).resolve("MISSING", instant)


def test_market_index_constructors_reject_ragged_frames() -> None:
    """Frames lacking required columns cannot back an index."""
    from src.sim.after_tax_market import _CpiIndex, _FxIndex, _PriceIndex, _RateIndex

    days = _sessions(date(2024, 1, 2), date(2024, 1, 31))
    with pytest.raises(AfterTaxDataError):
        _PriceIndex(_prices_frame(days, {"QQQ": [100.0] * len(days)}).drop("close"))
    with pytest.raises(AfterTaxDataError):
        _FxIndex(_fx_frame(days).drop("usdkrw"))
    with pytest.raises(AfterTaxDataError):
        _CpiIndex(_cpi_frame().drop("value"))
    with pytest.raises(AfterTaxDataError):
        _RateIndex(_rates_frame([(days[0], 0.04)]).drop("value"))
