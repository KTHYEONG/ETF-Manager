"""Research-proxy index DCA: compounds Ken French daily market returns, never PRICES."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Final

import polars as pl

from src.analytics.metrics import max_drawdown, real_krw, xirr
from src.data.calendar import DEFAULT_CALENDAR_NAME, load_calendar
from src.data.catalog import latest_artifact, load_visible
from src.data.query import load_as_of
from src.data.schedule import build_decision_schedule
from src.data.schema import Dataset
from src.policy.currency import conversion_fraction
from src.policy.targets import PolicyId, all_policy_tickers
from src.sim.allocation import (
    AllocationDataError,
    AllocationResult,
    AllocationSnapshot,
    _visible_cpi,
    _visible_fx,
)

if TYPE_CHECKING:
    from datetime import date, datetime

    from src.data.settings import DataSettings
    from src.sim.allocation import AllocationConfig

logger = logging.getLogger(__name__)

_BPS: Final[float] = 10_000.0
_BASE_INDEX_LEVEL: Final[float] = 1.0

__all__ = [
    "run_research_proxy",
    "run_research_proxy_from_store",
]


def run_research_proxy(
    config: AllocationConfig,
    returns: pl.DataFrame,
    fx: pl.DataFrame,
    cpi: pl.DataFrame,
) -> AllocationResult:
    """Simulate buy-only monthly DCA into a fractional US-market index unit.

    The wealth path holds fractional index units of ``I_t = I_{t-1} * (1 + R_t)``
    with ``I_0 = 1``; each contribution converts at the FX mid on the delayed
    execution session and buys at the post-return close index level. Cashflow,
    fill delay, and CPI-real metrics mirror the ETF allocation exactly, but no
    commission, spread grid, or TER applies: Wave C is identity isolation.

    Args:
        config: Allocation config whose policy must be the research proxy identity.
        returns: Availability-stamped RESEARCH_RETURNS frame (one series).
        fx: Availability-stamped FX frame for KRW/USD conversion marks.
        cpi: Availability-stamped CPI frame for real-wealth deflation.

    Returns:
        AllocationResult shaped like the ETF engine's output.

    Raises:
        ValueError: When the policy is not FF_PROXY, costs are nonzero, or the
            frame violates the research_proxy identity contract (I9).
        AllocationDataError: When the schedule is empty or an execution session lacks
            a finite return greater than -1 at its close, or FX/CPI marks are missing.
        XirrError: When the money-weighted rate cannot be identified.
    """
    if config.policy is not PolicyId.FF_PROXY:
        raise ValueError(f"run_research_proxy requires FF_PROXY, got {config.policy!s}")
    if config.commission_bps != 0.0 or config.fx_spread_bps != 0.0:
        raise ValueError(
            f"research_proxy is cost-free identity isolation; commission_bps and fx_spread_bps must be 0, "
            f"got commission_bps={config.commission_bps!r}, fx_spread_bps={config.fx_spread_bps!r}"
        )
    series_id = _resolve_series_identity(returns)
    schedule = build_decision_schedule(config.start, config.end, fill_delay_sessions=config.fill_delay_sessions)
    if not schedule:
        raise AllocationDataError(f"empty decision schedule over [{config.start.isoformat()}, {config.end.isoformat()}]")
    calendar = load_calendar(DEFAULT_CALENDAR_NAME)

    cash_krw = 0.0
    cash_usd = 0.0
    units = 0.0
    index_level = _BASE_INDEX_LEVEL
    snapshots: list[AllocationSnapshot] = []
    cpi_levels: list[float] = []
    for point in schedule:
        close_ts = calendar.close_ts(point.execution_session)
        usdkrw = _visible_fx(fx, point.execution_session, close_ts)
        cpi_level = _visible_cpi(cpi, point.execution_session, close_ts)
        session_return = _visible_return(returns, point.execution_session, close_ts)
        # Buy at the post-return close index level of the execution session.
        index_level *= 1.0 + session_return

        contribution = config.monthly_contribution_krw
        cash_krw += contribution
        investable_krw = min(cash_krw, contribution)
        fraction = 1.0 if config.currency is None else conversion_fraction(fx, point.signal_at, config.currency)
        fx_gross = usdkrw * (1.0 + config.fx_spread_bps / _BPS)
        spend_krw = investable_krw * fraction
        spend_usd = spend_krw / fx_gross
        # No commission applies to an index unit; only the (enforced-zero) FX spread leaks.
        fees_krw = spend_usd * (fx_gross - usdkrw)
        units += spend_usd / index_level
        cash_krw -= spend_krw
        mark_krw = units * index_level * usdkrw + cash_usd * usdkrw + cash_krw
        snapshots.append(
            AllocationSnapshot(
                session=point.execution_session,
                cash_krw=cash_krw,
                cash_usd=cash_usd,
                shares={series_id: units},
                mark_krw=mark_krw,
                contribution_krw=contribution,
                fees_krw=fees_krw,
            )
        )
        cpi_levels.append(cpi_level)

    terminal_wealth_krw = snapshots[-1].mark_krw
    cashflows = [(calendar.close_ts(snapshot.session), -snapshot.contribution_krw) for snapshot in snapshots]
    cashflows.append((cashflows[-1][0], terminal_wealth_krw))
    money_weighted_rate = xirr(cashflows)

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
        "[DATA] event=research_proxy_done series=%s steps=%d terminal_krw=%.2f xirr=%.6f mdd=%.4f",
        series_id,
        len(snapshots),
        terminal_wealth_krw,
        money_weighted_rate,
        result.max_drawdown,
    )
    return result


def run_research_proxy_from_store(config: AllocationConfig, settings: DataSettings) -> AllocationResult:
    """Load latest RESEARCH_RETURNS, FX, and CPI partitions (trust-gated), then simulate.

    Raises:
        UntrustedDatasetError: When any required dataset lacks a manifest-verified partition.
        AllocationDataError: When the schedule is empty or fills lack data.
        ValueError: When the config or frames violate the research_proxy contract.
    """
    for dataset in (Dataset.RESEARCH_RETURNS, Dataset.FX, Dataset.CPI):
        latest_artifact(settings, dataset)
    schedule = build_decision_schedule(config.start, config.end, fill_delay_sessions=config.fill_delay_sessions)
    if not schedule:
        raise AllocationDataError(f"empty decision schedule over [{config.start.isoformat()}, {config.end.isoformat()}]")
    cutoff = load_calendar(DEFAULT_CALENDAR_NAME).close_ts(schedule[-1].execution_session)
    returns = load_visible(settings, Dataset.RESEARCH_RETURNS, cutoff)
    fx = load_visible(settings, Dataset.FX, cutoff)
    cpi = load_visible(settings, Dataset.CPI, cutoff)
    return run_research_proxy(config, returns, fx, cpi)


def _resolve_series_identity(returns: pl.DataFrame) -> str:
    """Single non-ticker series id; any policy-ticker collision fails closed (I9)."""
    banned_tickers = set(all_policy_tickers())
    series_ids = returns.get_column("series_id").unique().to_list() if returns.height > 0 else []
    for series_id in series_ids:
        if series_id in banned_tickers:
            raise ValueError(f"I9 violation: research_proxy series_id {series_id!r} collides with a policy ticker")
    if len(series_ids) != 1:
        raise ValueError(f"research_proxy requires exactly one series_id, got {len(series_ids)}")
    resolved = series_ids[0]
    return str(resolved)


def _visible_return(returns: pl.DataFrame, session: date, close_ts: datetime) -> float:
    """Simple total return visible at the execution close; fail-closed on gaps."""
    visible = load_as_of(returns, Dataset.RESEARCH_RETURNS, close_ts)
    rows = visible.filter(pl.col("date") == session)
    if rows.is_empty():
        raise AllocationDataError(f"missing research_proxy simple_return on {session.isoformat()} at its execution close")
    value = rows.item(0, "simple_return")
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AllocationDataError(f"non-finite research_proxy simple_return on {session.isoformat()}")
    session_return = float(value)
    if session_return <= -1.0:
        raise AllocationDataError(f"research_proxy simple_return {session_return!r} wipes out wealth on {session.isoformat()}")
    return session_return
