"""Fast-mode single-sleeve KRW DCA baseline."""

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
from src.etf_manager.policy.targets import BASELINE_ALIASES, BaselineId
from src.etf_manager.sim.lots import fill_integer_buys

if TYPE_CHECKING:
    from datetime import datetime

    from src.etf_manager.data.settings import DataSettings

logger = logging.getLogger(__name__)

_BPS: Final[float] = 10_000.0

__all__ = [
    "BASELINE_ALIASES",
    "BaselineConfig",
    "BaselineDataError",
    "BaselineId",
    "BaselineResult",
    "LedgerSnapshot",
    "run_baseline",
    "run_baseline_from_store",
]


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """External cashflow and implementation parameters shared across candidates."""

    baseline: BaselineId
    ticker: str
    start: date
    end: date
    monthly_contribution_krw: float
    fx_spread_bps: float = 0.0
    commission_bps: float = 0.0
    fill_delay_sessions: int = 1


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    """Portfolio state at one execution close; ledger is the SSOT for the path."""

    session: date
    cash_krw: float
    cash_usd: float
    shares: float
    mark_krw: float
    contribution_krw: float
    fees_krw: float


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """Full path plus summary metrics."""

    config: BaselineConfig
    snapshots: tuple[LedgerSnapshot, ...]
    terminal_wealth_krw: float
    xirr: float
    max_drawdown: float
    terminal_wealth_real_krw: float
    xirr_real: float


class BaselineDataError(RuntimeError):
    """Missing PIT price or FX at an execution close; never skipped silently."""


def run_baseline(
    config: BaselineConfig,
    prices: pl.DataFrame,
    fx: pl.DataFrame,
    cpi: pl.DataFrame,
) -> BaselineResult:
    """Simulate buy-only monthly DCA with delayed fills on in-memory PIT frames.

    Contributions are credited and fills occur only on the execution session of
    each decision point; the signal-session close is never used as a fill price.
    Nominal marks drive the equity path; CPI levels only deflate terminal wealth
    and the money-weighted rate into first-snapshot purchasing power.

    Raises:
        ValueError: When ``monthly_contribution_krw`` is not positive.
        BaselineDataError: When the schedule is empty or a required price, FX,
            or CPI observation is missing, non-positive, or null at an execution close.
        XirrError: When the money-weighted rate cannot be identified.
    """
    if config.monthly_contribution_krw <= 0:
        raise ValueError("monthly_contribution_krw must be positive")
    schedule = build_decision_schedule(config.start, config.end, fill_delay_sessions=config.fill_delay_sessions)
    if not schedule:
        raise BaselineDataError(f"empty decision schedule over [{config.start.isoformat()}, {config.end.isoformat()}]")
    calendar = load_calendar(DEFAULT_CALENDAR_NAME)

    cash_krw = 0.0
    cash_usd = 0.0
    shares = 0
    snapshots: list[LedgerSnapshot] = []
    cpi_levels: list[float] = []
    for point in schedule:
        close_ts = calendar.close_ts(point.execution_session)
        price = _visible_close(prices, config.ticker, point.execution_session, close_ts)
        usdkrw = _visible_fx(fx, point.execution_session, close_ts)
        cpi_level = _visible_cpi(cpi, point.execution_session, close_ts)

        contribution = config.monthly_contribution_krw
        cash_krw += contribution
        investable_krw = min(cash_krw, contribution)
        fx_gross = usdkrw * (1.0 + config.fx_spread_bps / _BPS)
        # Recycled dust joins the fresh conversion as one ticket; commission bills the whole trade.
        lots, cash_usd, commission_krw = fill_integer_buys(
            cash_usd=cash_usd,
            sleeve_budget_krw=investable_krw,
            fx_gross=fx_gross,
            weights={config.ticker: 1.0},
            prices={config.ticker: price},
            commission_bps=config.commission_bps,
        )
        shares += lots[config.ticker]
        cash_krw -= investable_krw
        fees_krw = investable_krw * (fx_gross - usdkrw) + commission_krw
        snapshots.append(
            LedgerSnapshot(
                session=point.execution_session,
                cash_krw=cash_krw,
                cash_usd=cash_usd,
                shares=float(shares),
                mark_krw=shares * price * usdkrw + cash_usd * usdkrw + cash_krw,
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
    result = BaselineResult(
        config=config,
        snapshots=tuple(snapshots),
        terminal_wealth_krw=terminal_wealth_krw,
        xirr=money_weighted_rate,
        max_drawdown=max_drawdown([snapshot.mark_krw for snapshot in snapshots]),
        terminal_wealth_real_krw=terminal_wealth_real_krw,
        xirr_real=xirr(real_cashflows),
    )
    logger.info(
        "[DATA] event=baseline_done baseline=%s ticker=%s steps=%d terminal_krw=%.2f xirr=%.6f mdd=%.4f",
        str(config.baseline),
        config.ticker,
        len(snapshots),
        terminal_wealth_krw,
        money_weighted_rate,
        result.max_drawdown,
    )
    return result


def run_baseline_from_store(config: BaselineConfig, settings: DataSettings) -> BaselineResult:
    """Load latest PRICES, FX, and CPI partitions, then run :func:`run_baseline`.

    Raises:
        UntrustedDatasetError: When any dataset lacks a manifest-verified partition.
        BaselineDataError: When the schedule is empty or fills lack data.
    """
    # Pre-flight trust gate: fail closed before simulating on untrusted partitions.
    for dataset in (Dataset.PRICES, Dataset.FX, Dataset.CPI):
        latest_artifact(settings, dataset)
    schedule = build_decision_schedule(config.start, config.end, fill_delay_sessions=config.fill_delay_sessions)
    if not schedule:
        raise BaselineDataError(f"empty decision schedule over [{config.start.isoformat()}, {config.end.isoformat()}]")
    cutoff = load_calendar(DEFAULT_CALENDAR_NAME).close_ts(schedule[-1].execution_session)
    prices = load_visible(settings, Dataset.PRICES, cutoff)
    fx = load_visible(settings, Dataset.FX, cutoff)
    cpi = load_visible(settings, Dataset.CPI, cutoff)
    return run_baseline(config, prices, fx, cpi)


def _visible_close(prices: pl.DataFrame, ticker: str, session: date, close_ts: datetime) -> float:
    """Adjusted close of ``ticker`` visible at the execution close; fail-closed."""
    visible = load_as_of(prices, Dataset.PRICES, close_ts)
    rows = visible.filter((pl.col("ticker") == ticker) & (pl.col("date") == session))
    if rows.is_empty():
        raise BaselineDataError(f"missing {ticker!r} price row on {session.isoformat()} at its execution close")
    value = rows.item(0, "adjusted_close")
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
        raise BaselineDataError(f"non-positive adjusted_close for {ticker!r} on {session.isoformat()}")
    return float(value)


def _visible_fx(fx: pl.DataFrame, session: date, close_ts: datetime) -> float:
    """USD/KRW mid rate visible at the execution close; fail-closed."""
    visible = load_as_of(fx, Dataset.FX, close_ts)
    rows = visible.filter(pl.col("date") == session)
    if rows.is_empty():
        raise BaselineDataError(f"missing usdkrw row on {session.isoformat()} at its execution close")
    value = rows.item(0, "usdkrw")
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
        raise BaselineDataError(f"null or non-positive usdkrw on {session.isoformat()}")
    return float(value)


def _visible_cpi(cpi: pl.DataFrame, session: date, close_ts: datetime) -> float:
    """Latest positive CPI level by period_end visible at the execution close; fail-closed."""
    visible = load_as_of(cpi, Dataset.CPI, close_ts)
    rows = visible.filter(pl.col("value").is_finite() & (pl.col("value") > 0.0)).sort("period_end")
    if rows.is_empty():
        raise BaselineDataError(f"missing positive CPI row on {session.isoformat()} at its execution close")
    return float(rows.item(rows.height - 1, "value"))
