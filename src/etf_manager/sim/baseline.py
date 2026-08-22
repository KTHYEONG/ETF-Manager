"""Fast-mode single-sleeve KRW DCA baseline."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import polars as pl

from src.etf_manager.analytics.metrics import max_drawdown, xirr
from src.etf_manager.data.calendar import DEFAULT_CALENDAR_NAME, load_calendar
from src.etf_manager.data.catalog import latest_artifact, load_visible
from src.etf_manager.data.query import load_as_of
from src.etf_manager.data.schedule import build_decision_schedule
from src.etf_manager.data.schema import Dataset

if TYPE_CHECKING:
    from datetime import datetime

    from src.etf_manager.data.settings import DataSettings

logger = logging.getLogger(__name__)

_BPS: Final[float] = 10_000.0

__all__ = [
    "BaselineConfig",
    "BaselineDataError",
    "BaselineId",
    "BaselineResult",
    "LedgerSnapshot",
    "run_baseline",
    "run_baseline_from_store",
]


class BaselineId(StrEnum):
    """Named one-ticker accumulation policies; weights live only in config."""

    B0_GLOBAL = "b0_global"
    B1_US = "b1_us"


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


class BaselineDataError(RuntimeError):
    """Missing PIT price or FX at an execution close; never skipped silently."""


def run_baseline(
    config: BaselineConfig,
    prices: pl.DataFrame,
    fx: pl.DataFrame,
) -> BaselineResult:
    """Simulate buy-only monthly DCA with delayed fills on in-memory PIT frames.

    Contributions are credited and fills occur only on the execution session of
    each decision point; the signal-session close is never used as a fill price.

    Raises:
        ValueError: When ``monthly_contribution_krw`` is not positive.
        BaselineDataError: When the schedule is empty or a required price/FX
            observation is missing, non-positive, or null at an execution close.
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
    for point in schedule:
        close_ts = calendar.close_ts(point.execution_session)
        price = _visible_close(prices, config.ticker, point.execution_session, close_ts)
        usdkrw = _visible_fx(fx, point.execution_session, close_ts)

        contribution = config.monthly_contribution_krw
        cash_krw += contribution
        investable_krw = min(cash_krw, contribution)
        fx_gross = usdkrw * (1.0 + config.fx_spread_bps / _BPS)
        spend_usd = investable_krw / fx_gross
        fee_usd = spend_usd * config.commission_bps / _BPS
        bought = math.floor((spend_usd - fee_usd) / price)
        shares += bought
        cash_usd += spend_usd - fee_usd - bought * price
        cash_krw -= investable_krw
        fees_krw = spend_usd * (fx_gross - usdkrw) + fee_usd * fx_gross
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

    terminal_wealth_krw = snapshots[-1].mark_krw
    cashflows = [(calendar.close_ts(snapshot.session), -snapshot.contribution_krw) for snapshot in snapshots]
    cashflows.append((cashflows[-1][0], terminal_wealth_krw))
    money_weighted_rate = xirr(cashflows)
    result = BaselineResult(
        config=config,
        snapshots=tuple(snapshots),
        terminal_wealth_krw=terminal_wealth_krw,
        xirr=money_weighted_rate,
        max_drawdown=max_drawdown([snapshot.mark_krw for snapshot in snapshots]),
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
    """Load latest PRICES and FX partitions, then run :func:`run_baseline`.

    Raises:
        UntrustedDatasetError: When either dataset lacks a manifest-verified partition.
        BaselineDataError: When the schedule is empty or fills lack data.
    """
    # Pre-flight trust gate: fail closed before simulating on untrusted partitions.
    for dataset in (Dataset.PRICES, Dataset.FX):
        latest_artifact(settings, dataset)
    schedule = build_decision_schedule(config.start, config.end, fill_delay_sessions=config.fill_delay_sessions)
    if not schedule:
        raise BaselineDataError(f"empty decision schedule over [{config.start.isoformat()}, {config.end.isoformat()}]")
    cutoff = load_calendar(DEFAULT_CALENDAR_NAME).close_ts(schedule[-1].execution_session)
    prices = load_visible(settings, Dataset.PRICES, cutoff)
    fx = load_visible(settings, Dataset.FX, cutoff)
    return run_baseline(config, prices, fx)


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
