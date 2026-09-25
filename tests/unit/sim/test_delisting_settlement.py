"""Invariant tests for delisting settlement in both accumulation engines."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, date, datetime

import polars as pl
import pytest

from src.data.calendar import load_calendar
from src.data.pipeline import ingest
from src.data.schema import Dataset, spec_for
from src.data.universe import MembershipEntry, UniverseMembership
from src.policy.targets import PolicyId
from src.sim.after_tax_engine import AfterTaxConfig, AfterTaxDataError, run_after_tax
from src.sim.allocation import AllocationConfig, AllocationDataError, run_allocation
from src.sim.tax import load_tax_regime

_CALENDAR = load_calendar("XNYS")
_RETRIEVED_AT = datetime(2026, 1, 1, tzinfo=UTC)
_START = date(2024, 1, 15)
_LAST_TRADING_DATE = date(2024, 2, 16)
_REGIME = load_tax_regime("configs/tax/kr_overseas_equity.json")


def _sessions(start: date, end: date) -> tuple[date, ...]:
    return _CALENDAR.sessions(start, end)


def _prices(
    days: tuple[date, ...],
    closes: Mapping[str, list[float]],
    *,
    adjusted: Mapping[str, list[float]] | None = None,
) -> pl.DataFrame:
    spec = spec_for(Dataset.PRICES)
    adjusted = adjusted or closes
    tickers: list[str] = []
    dates: list[date] = []
    raw_closes: list[float] = []
    adjusted_closes: list[float] = []
    for ticker in sorted(closes):
        for day, close, adjusted_close in zip(
            days, closes[ticker], adjusted[ticker], strict=True
        ):
            tickers.append(ticker)
            dates.append(day)
            raw_closes.append(close)
            adjusted_closes.append(adjusted_close)
    count = len(dates)
    return ingest(
        pl.DataFrame(
            {
                "ticker": tickers,
                "date": dates,
                "open": raw_closes,
                "high": raw_closes,
                "low": raw_closes,
                "close": raw_closes,
                "volume": [10_000] * count,
                "adjusted_close": adjusted_closes,
                "dividend": [0.0] * count,
                "split_factor": [1.0] * count,
                "source": ["synthetic"] * count,
                "retrieved_at": [_RETRIEVED_AT] * count,
            },
            schema=dict(spec.columns),
        ),
        Dataset.PRICES,
    )


def _old_closes(days: tuple[date, ...], *, post_delisting: float = 180.0) -> list[float]:
    return [
        120.0
        if day == _LAST_TRADING_DATE
        else post_delisting
        if day > _LAST_TRADING_DATE
        else 80.0
        for day in days
    ]


def _fx(days: tuple[date, ...]) -> pl.DataFrame:
    spec = spec_for(Dataset.FX)
    return ingest(
        pl.DataFrame(
            {
                "date": list(days),
                "usdkrw": [1_300.0] * len(days),
                "source": ["synthetic"] * len(days),
                "retrieved_at": [_RETRIEVED_AT] * len(days),
            },
            schema=dict(spec.columns),
        ),
        Dataset.FX,
    )


def _cpi() -> pl.DataFrame:
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


def _rates() -> pl.DataFrame:
    spec = spec_for(Dataset.RATES)
    return ingest(
        pl.DataFrame(
            {
                "series_id": ["DTB3"],
                "observation_date": [date(2023, 12, 1)],
                "value": [0.0],
                "source": ["synthetic"],
                "retrieved_at": [_RETRIEVED_AT],
            },
            schema=dict(spec.columns),
        ),
        Dataset.RATES,
    )


def _membership(*tickers: str) -> UniverseMembership:
    entries = {
        ticker: MembershipEntry(
            ticker=ticker,
            listing_date=date(2020, 1, 2),
            last_trading_date=_LAST_TRADING_DATE if ticker == "OLD" else None,
            evidence_url=f"https://example.com/{ticker}",
        )
        for ticker in tickers
    }
    return UniverseMembership(entries=entries, sha256="test")


def _allocation(
    targets: Mapping[str, float],
    membership: UniverseMembership,
    *,
    commission_bps: float,
    end: date = date(2024, 3, 31),
) -> AllocationConfig:
    return AllocationConfig(
        policy=PolicyId.VT,
        start=_START,
        end=end,
        monthly_contribution_krw=1_300_000.0,
        commission_bps=commission_bps,
        targets_override=dict(targets),
        membership=membership,
    )


def _after_tax(
    targets: Mapping[str, float],
    membership: UniverseMembership,
    *,
    end: date,
    monthly_contribution_krw: float = 13_130_000.0,
    commission_bps: float = 100.0,
    tax_enabled: bool = True,
) -> AfterTaxConfig:
    return AfterTaxConfig(
        start=_START,
        end=end,
        monthly_contribution_krw=monthly_contribution_krw,
        tax_regime=replace(_REGIME, annual_deduction_krw=0.0),
        targets=dict(targets),
        harvest_gains=False,
        tax_enabled=tax_enabled,
        commission_bps=commission_bps,
        fractional_shares=True,
        membership=membership,
    )


def test_buy_only_position_liquidates_at_unadjusted_last_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    days = _sessions(date(2024, 1, 2), date(2024, 4, 1))
    prices = _prices(days, {"OLD": _old_closes(days)})
    config = _allocation({"OLD": 1.0}, _membership("OLD"), commission_bps=100.0)

    with caplog.at_level(logging.WARNING, logger="src.sim.allocation"):
        result = run_allocation(config, prices, _fx(days), _cpi())

    first = result.snapshots[0]
    settlement_index = next(
        index for index, snapshot in enumerate(result.snapshots) if snapshot.session > _LAST_TRADING_DATE
    )
    settled = result.snapshots[settlement_index]
    quantity = first.shares["OLD"]
    expected_cash = quantity * 120.0 * (1.0 - 100.0 / 10_000.0)
    expected_fee = quantity * 120.0 * (100.0 / 10_000.0) * 1_300.0

    assert quantity > 0
    assert "OLD" not in settled.shares
    assert settled.cash_usd - first.cash_usd == pytest.approx(expected_cash)
    assert settled.fees_krw == pytest.approx(expected_fee)
    assert all("OLD" not in snapshot.shares for snapshot in result.snapshots[settlement_index:])
    assert "[PORTFOLIO] event=delisted_no_eligible_target" in caplog.text


def test_delisted_weight_redistributes_to_survivor() -> None:
    """Delisting proceeds (6 x 120 USD) plus dust are redeployed into the survivor."""
    days = _sessions(date(2024, 1, 2), date(2024, 4, 1))
    prices = _prices(
        days,
        {"OLD": _old_closes(days), "KEEP": [100.0] * len(days)},
    )
    config = _allocation(
        {"OLD": 0.5, "KEEP": 0.5},
        _membership("OLD", "KEEP"),
        commission_bps=0.0,
    )

    result = run_allocation(config, prices, _fx(days), _cpi())

    first = result.snapshots[0]
    settled = result.snapshots[1]
    assert first.shares["OLD"] == 6
    assert first.shares["KEEP"] == 5
    assert "OLD" not in settled.shares
    # 740 USD(청산 720 + 잔여 20)가 KEEP 7주로 재투자되고 40 USD만 남는다.
    assert settled.shares["KEEP"] == 22
    assert settled.cash_usd == pytest.approx(40.0)


def test_allocation_ignores_post_delisting_prices() -> None:
    days = _sessions(date(2024, 1, 2), date(2024, 4, 1))
    base = _prices(days, {"OLD": _old_closes(days)})
    corrupted = _prices(days, {"OLD": _old_closes(days, post_delisting=1_000_000.0)})
    config = _allocation({"OLD": 1.0}, _membership("OLD"), commission_bps=0.0)

    baseline = run_allocation(config, base, _fx(days), _cpi())
    perturbed = run_allocation(config, corrupted, _fx(days), _cpi())

    assert perturbed.snapshots == baseline.snapshots
    assert perturbed.terminal_wealth_krw == pytest.approx(baseline.terminal_wealth_krw)
    assert perturbed.terminal_wealth_real_krw == pytest.approx(baseline.terminal_wealth_real_krw)


def test_allocation_missing_last_close_fails_closed() -> None:
    days = _sessions(date(2024, 1, 2), date(2024, 4, 1))
    prices = _prices(days, {"OLD": _old_closes(days)}).filter(
        ~((pl.col("ticker") == "OLD") & (pl.col("date") == _LAST_TRADING_DATE))
    )

    with pytest.raises(AllocationDataError) as exc_info:
        run_allocation(
            _allocation({"OLD": 1.0}, _membership("OLD"), commission_bps=0.0),
            prices,
            _fx(days),
            _cpi(),
        )

    assert "OLD" in str(exc_info.value)
    assert _LAST_TRADING_DATE.isoformat() in str(exc_info.value)


def test_after_tax_delisting_realizes_gain_tax_and_conserves_ledger() -> None:
    days = _sessions(date(2024, 1, 2), date(2025, 2, 3))
    prices = _prices(days, {"OLD": _old_closes(days)})
    config = _after_tax(
        {"OLD": 1.0},
        _membership("OLD"),
        end=date(2025, 1, 31),
    )

    result = run_after_tax(config, prices, _fx(days), _cpi(), _rates())

    assert len(result.disposals) == 1
    disposal = result.disposals[0]
    settlement_index = next(
        index for index, snapshot in enumerate(result.snapshots) if snapshot.session > _LAST_TRADING_DATE
    )
    settlement = result.snapshots[settlement_index]
    assert disposal.quantity == pytest.approx(125.0)
    assert disposal.trade_session == _LAST_TRADING_DATE
    assert disposal.settle_session == date(2024, 2, 20)
    assert disposal.proceeds_krw == pytest.approx(19_305_000.0)
    assert disposal.basis_krw == pytest.approx(13_130_000.0)
    assert disposal.gain_krw == pytest.approx(6_175_000.0)
    assert settlement.fees_krw == pytest.approx(195_000.0)
    assert "OLD" not in settlement.shares
    assert result.sell_count == 1

    assessed = result.snapshots[-1]
    assert assessed.tax_payable_krw == pytest.approx(1_358_500.0)
    contributions = sum(snapshot.contribution_krw for snapshot in result.snapshots)
    expected_nav = contributions + disposal.gain_krw - assessed.tax_payable_krw
    assert assessed.nav_krw == pytest.approx(expected_nav)


def test_after_tax_delisting_redistributes_and_ignores_post_prices() -> None:
    days = _sessions(date(2024, 1, 2), date(2024, 4, 1))
    base = _prices(
        days,
        {"OLD": _old_closes(days), "KEEP": [100.0] * len(days)},
    )
    corrupted = _prices(
        days,
        {
            "OLD": _old_closes(days, post_delisting=1_000_000.0),
            "KEEP": [100.0] * len(days),
        },
    )
    config = _after_tax(
        {"OLD": 0.5, "KEEP": 0.5},
        _membership("OLD", "KEEP"),
        end=date(2024, 3, 31),
        commission_bps=0.0,
        tax_enabled=False,
    )

    baseline = run_after_tax(config, base, _fx(days), _cpi(), _rates())
    perturbed = run_after_tax(config, corrupted, _fx(days), _cpi(), _rates())

    assert "OLD" in baseline.snapshots[0].shares
    assert "OLD" not in baseline.snapshots[1].shares
    assert baseline.snapshots[1].shares["KEEP"] > baseline.snapshots[0].shares["KEEP"]
    assert perturbed.snapshots == baseline.snapshots
    assert perturbed.disposals == baseline.disposals


def test_after_tax_missing_last_close_fails_closed() -> None:
    days = _sessions(date(2024, 1, 2), date(2024, 4, 1))
    prices = _prices(days, {"OLD": _old_closes(days)}).filter(
        ~((pl.col("ticker") == "OLD") & (pl.col("date") == _LAST_TRADING_DATE))
    )

    with pytest.raises(AfterTaxDataError) as exc_info:
        run_after_tax(
            _after_tax(
                {"OLD": 1.0},
                _membership("OLD"),
                end=date(2024, 3, 31),
                tax_enabled=False,
            ),
            prices,
            _fx(days),
            _cpi(),
            _rates(),
        )

    assert "OLD" in str(exc_info.value)
    assert _LAST_TRADING_DATE.isoformat() in str(exc_info.value)
