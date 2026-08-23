"""Buy-only multi-sleeve contribution allocation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final

import polars as pl

from src.etf_manager.analytics.metrics import max_drawdown, real_krw, xirr
from src.etf_manager.data.calendar import DEFAULT_CALENDAR_NAME, load_calendar
from src.etf_manager.data.catalog import latest_artifact, load_visible
from src.etf_manager.data.query import load_as_of
from src.etf_manager.data.schedule import build_decision_schedule
from src.etf_manager.data.schema import Dataset
from src.etf_manager.policy.currency import conversion_fraction
from src.etf_manager.policy.overlay import apply_bounded_overlay
from src.etf_manager.policy.targets import PolicyId, resolve_targets
from src.etf_manager.policy.tilt import resolve_tilted_targets
from src.etf_manager.sim.contribution import allocate_contribution

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from src.etf_manager.data.settings import DataSettings
    from src.etf_manager.policy.currency import CurrencyConfig
    from src.etf_manager.policy.overlay import OverlayConfig
    from src.etf_manager.policy.tilt import FactorTilt

logger = logging.getLogger(__name__)

_BPS: Final[float] = 10_000.0

__all__ = [
    "AllocationConfig",
    "AllocationDataError",
    "AllocationResult",
    "AllocationSnapshot",
    "run_allocation",
    "run_allocation_from_store",
]


class AllocationDataError(RuntimeError):
    """Missing PIT price, FX, or CPI at an execution close; never skipped silently."""


@dataclass(frozen=True, slots=True)
class AllocationConfig:
    """Policy identity plus external cashflow and implementation parameters."""

    policy: PolicyId
    start: date
    end: date
    monthly_contribution_krw: float
    fill_delay_sessions: int = 1
    fx_spread_bps: float = 0.0
    commission_bps: float = 0.0
    tilt: FactorTilt | None = None
    rebalance_band: float | None = None
    overlay: OverlayConfig | None = None
    currency: CurrencyConfig | None = None


@dataclass(frozen=True, slots=True)
class AllocationSnapshot:
    """Portfolio state at one execution close; ledger is the SSOT for the path."""

    session: date
    cash_krw: float
    cash_usd: float
    shares: Mapping[str, float]
    mark_krw: float
    contribution_krw: float
    fees_krw: float


@dataclass(frozen=True, slots=True)
class AllocationResult:
    """Full path plus summary metrics."""

    config: AllocationConfig
    snapshots: tuple[AllocationSnapshot, ...]
    terminal_wealth_krw: float
    xirr: float
    max_drawdown: float
    terminal_wealth_real_krw: float
    xirr_real: float


def run_allocation(
    config: AllocationConfig,
    prices: pl.DataFrame,
    fx: pl.DataFrame,
    cpi: pl.DataFrame,
    factors: pl.DataFrame | None = None,
    macro: pl.DataFrame | None = None,
) -> AllocationResult:
    """Simulate buy-only multi-sleeve monthly DCA with delayed fills on in-memory PIT frames.

    Target weights resolve at each decision point's ``signal_at``; fills use
    execution-session close prices and FX. New money follows the targets and no
    position is ever sold, so integer-lot rounding dust stays in unmarked cash.
    Nominal marks drive the equity path; CPI levels only deflate terminal wealth
    and the money-weighted rate into first-snapshot purchasing power.

    Raises:
        ValueError: When ``monthly_contribution_krw`` is not positive.
        PolicyError: When weight resolution fails closed at a signal instant.
        AllocationDataError: When the schedule is empty or a required price, FX,
            or CPI observation is missing, non-positive, or null at an execution close.
        XirrError: When the money-weighted rate cannot be identified.
    """
    if config.monthly_contribution_krw <= 0:
        raise ValueError("monthly_contribution_krw must be positive")
    schedule = build_decision_schedule(config.start, config.end, fill_delay_sessions=config.fill_delay_sessions)
    if not schedule:
        raise AllocationDataError(f"empty decision schedule over [{config.start.isoformat()}, {config.end.isoformat()}]")
    calendar = load_calendar(DEFAULT_CALENDAR_NAME)

    cash_krw = 0.0
    cash_usd = 0.0
    shares_by_ticker: dict[str, int] = {}
    snapshots: list[AllocationSnapshot] = []
    cpi_levels: list[float] = []
    for point in schedule:
        close_ts = calendar.close_ts(point.execution_session)
        usdkrw = _visible_fx(fx, point.execution_session, close_ts)
        cpi_level = _visible_cpi(cpi, point.execution_session, close_ts)
        if config.tilt is None:
            targets = resolve_targets(config.policy, prices, point.signal_at)
        else:
            if factors is None:
                raise ValueError("factor tilt requires a factors frame")
            targets = resolve_tilted_targets(config.policy, prices, factors, point.signal_at, config.tilt)
        if config.overlay is not None:
            targets = apply_bounded_overlay(
                targets, prices, point.signal_at, config.overlay, macro=macro
            )
        fraction = 1.0 if config.currency is None else conversion_fraction(fx, point.signal_at, config.currency)

        contribution = config.monthly_contribution_krw
        cash_krw += contribution
        investable_krw = min(cash_krw, contribution)
        fx_gross = usdkrw * (1.0 + config.fx_spread_bps / _BPS)
        marks_krw = {
            ticker: float(shares_by_ticker.get(ticker, 0)) * _visible_close(prices, ticker, point.execution_session, close_ts) * usdkrw
            for ticker in targets
        }
        nav_krw = sum(marks_krw.values()) + cash_usd * usdkrw + cash_krw
        spend_weights = (
            targets
            if config.rebalance_band is None
            else allocate_contribution(
                targets=targets,
                marks_krw=marks_krw,
                nav_krw=nav_krw,
                commission_bps=config.commission_bps,
                rebalance_band=config.rebalance_band,
            )
        )
        fees_krw = 0.0
        position_value_usd = 0.0
        # Overlay residual and FX defer both stay in cash: spend only the converted budget.
        convert_krw = investable_krw * fraction
        sleeve_budget_krw = convert_krw * sum(spend_weights.values())
        for ticker, weight in spend_weights.items():
            price = _visible_close(prices, ticker, point.execution_session, close_ts)
            spend_usd = sleeve_budget_krw * weight / fx_gross
            fee_usd = spend_usd * config.commission_bps / _BPS
            bought = math.floor((spend_usd - fee_usd) / price)
            shares_by_ticker[ticker] = shares_by_ticker.get(ticker, 0) + bought
            cash_usd += spend_usd - fee_usd - bought * price
            fees_krw += spend_usd * (fx_gross - usdkrw) + fee_usd * fx_gross
            position_value_usd += shares_by_ticker[ticker] * price
        cash_krw -= sleeve_budget_krw
        snapshots.append(
            AllocationSnapshot(
                session=point.execution_session,
                cash_krw=cash_krw,
                cash_usd=cash_usd,
                shares=dict(shares_by_ticker),
                mark_krw=position_value_usd * usdkrw + cash_usd * usdkrw + cash_krw,
                contribution_krw=contribution,
                fees_krw=fees_krw,
            )
        )
        cpi_levels.append(cpi_level)

    terminal_wealth_krw = snapshots[-1].mark_krw
    cashflows = [(calendar.close_ts(snapshot.session), -snapshot.contribution_krw) for snapshot in snapshots]
    cashflows.append((cashflows[-1][0], terminal_wealth_krw))
    money_weighted_rate = xirr(cashflows)

    # Real KRW deflates to first-snapshot purchasing power; nominal legs stay untouched.
    base_cpi = cpi_levels[0]
    terminal_wealth_real_krw = real_krw(terminal_wealth_krw, cpi_index=cpi_levels[-1], cpi_base=base_cpi)
    real_cashflows = [
        (calendar.close_ts(snapshot.session), -snapshot.contribution_krw * base_cpi / level)
        for snapshot, level in zip(snapshots, cpi_levels, strict=True)
    ]
    real_cashflows.append((real_cashflows[-1][0], terminal_wealth_real_krw))
    result = AllocationResult(
        config=config,
        snapshots=tuple(snapshots),
        terminal_wealth_krw=terminal_wealth_krw,
        xirr=money_weighted_rate,
        max_drawdown=max_drawdown([snapshot.mark_krw for snapshot in snapshots]),
        terminal_wealth_real_krw=terminal_wealth_real_krw,
        xirr_real=xirr(real_cashflows),
    )
    logger.info(
        "[DATA] event=allocation_done policy=%s steps=%d terminal_krw=%.2f xirr=%.6f mdd=%.4f",
        str(config.policy),
        len(snapshots),
        terminal_wealth_krw,
        money_weighted_rate,
        result.max_drawdown,
    )
    return result


def run_allocation_from_store(config: AllocationConfig, settings: DataSettings) -> AllocationResult:
    """Load latest PRICES, FX, and CPI partitions (plus FACTORS/MACRO when required), then simulate.

    FACTORS is required only when ``config.tilt`` is set; a plain policy must not
    depend on the factors dataset at all. MACRO is loaded only for an overlay
    with a VIX threshold.

    Raises:
        UntrustedDatasetError: When any required dataset lacks a manifest-verified partition.
        AllocationDataError: When the schedule is empty or fills lack data.
    """
    need_macro = config.overlay is not None and config.overlay.vix_threshold is not None
    datasets = (
        (Dataset.PRICES, Dataset.FX, Dataset.CPI, Dataset.FACTORS)
        if config.tilt is not None
        else (Dataset.PRICES, Dataset.FX, Dataset.CPI)
    )
    if need_macro:
        datasets = (*datasets, Dataset.MACRO)
    # Pre-flight trust gate: fail closed before simulating on untrusted partitions.
    for dataset in datasets:
        latest_artifact(settings, dataset)
    schedule = build_decision_schedule(config.start, config.end, fill_delay_sessions=config.fill_delay_sessions)
    if not schedule:
        raise AllocationDataError(f"empty decision schedule over [{config.start.isoformat()}, {config.end.isoformat()}]")
    cutoff = load_calendar(DEFAULT_CALENDAR_NAME).close_ts(schedule[-1].execution_session)
    prices = load_visible(settings, Dataset.PRICES, cutoff)
    fx = load_visible(settings, Dataset.FX, cutoff)
    cpi = load_visible(settings, Dataset.CPI, cutoff)
    factors = load_visible(settings, Dataset.FACTORS, cutoff) if config.tilt is not None else None
    macro = load_visible(settings, Dataset.MACRO, cutoff) if need_macro else None
    return run_allocation(config, prices, fx, cpi, factors=factors, macro=macro)


def _visible_close(prices: pl.DataFrame, ticker: str, session: date, close_ts: datetime) -> float:
    """Adjusted close of ``ticker`` visible at the execution close; fail-closed."""
    visible = load_as_of(prices, Dataset.PRICES, close_ts)
    rows = visible.filter((pl.col("ticker") == ticker) & (pl.col("date") == session))
    if rows.is_empty():
        raise AllocationDataError(f"missing {ticker!r} price row on {session.isoformat()} at its execution close")
    value = rows.item(0, "adjusted_close")
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
        raise AllocationDataError(f"non-positive adjusted_close for {ticker!r} on {session.isoformat()}")
    return float(value)


def _visible_fx(fx: pl.DataFrame, session: date, close_ts: datetime) -> float:
    """USD/KRW mid rate visible at the execution close; fail-closed."""
    visible = load_as_of(fx, Dataset.FX, close_ts)
    rows = visible.filter(pl.col("date") == session)
    if rows.is_empty():
        raise AllocationDataError(f"missing usdkrw row on {session.isoformat()} at its execution close")
    value = rows.item(0, "usdkrw")
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
        raise AllocationDataError(f"null or non-positive usdkrw on {session.isoformat()}")
    return float(value)


def _visible_cpi(cpi: pl.DataFrame, session: date, close_ts: datetime) -> float:
    """Latest positive CPI level by period_end visible at the execution close; fail-closed."""
    visible = load_as_of(cpi, Dataset.CPI, close_ts)
    rows = visible.filter(pl.col("value").is_finite() & (pl.col("value") > 0.0)).sort("period_end")
    if rows.is_empty():
        raise AllocationDataError(f"missing positive CPI row on {session.isoformat()} at its execution close")
    return float(rows.item(rows.height - 1, "value"))
