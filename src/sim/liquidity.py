"""Point-in-time execution-capacity controls for simulated fills."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import polars as pl

from src.data.query import load_as_of
from src.data.schema import Dataset

__all__ = [
    "LiquidityBreachError",
    "LiquidityConfig",
    "check_fill_liquidity",
]


@dataclass(frozen=True, slots=True)
class LiquidityConfig:
    """Execution capacity bounds for backtest fills.

    Attributes:
        adv_window_sessions: Trailing sessions, strictly before the execution session,
            used for average daily dollar volume.
        max_adv_participation: Maximum order notional as a fraction of that average (0, 1].
        reject_zero_volume: Refuse fills on sessions with zero reported volume.
    """

    adv_window_sessions: int
    max_adv_participation: float
    reject_zero_volume: bool

    def __post_init__(self) -> None:
        if self.adv_window_sessions < 1:
            raise ValueError("adv_window_sessions must be at least 1")
        if not 0.0 < self.max_adv_participation <= 1.0:
            raise ValueError("max_adv_participation must be in (0, 1]")


class LiquidityBreachError(RuntimeError):
    """Raised when a fill would exceed declared market capacity."""


def _metric(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.2f}"


def _breach_message(
    reason: str,
    ticker: str,
    session: date,
    notional: float | None,
    adv: float | None,
    cap: float | None,
) -> str:
    return (
        f"liquidity breach: ticker={ticker} session={session.isoformat()} reason={reason}; "
        f"notional={_metric(notional)}; ADV={_metric(adv)}; cap={_metric(cap)}"
    )


def check_fill_liquidity(
    prices: pl.DataFrame,
    ticker: str,
    session: date,
    close_ts: datetime,
    shares: int | float,
    config: LiquidityConfig,
) -> None:
    """Fail closed when a fill trades on a dead session or beyond ADV participation.

    Args:
        prices: PIT-visible PRICES rows.
        ticker: Execution instrument.
        session: Execution session.
        close_ts: Execution close instant; only rows visible at it are used.
        shares: Absolute share quantity of the order.
        config: Declared bounds.

    Raises:
        LiquidityBreachError: If the session volume is zero while rejected, trailing
            history is shorter than the window, or notional exceeds the participation cap.
    """
    if shares == 0:
        return

    visible = load_as_of(prices, Dataset.PRICES, close_ts).filter(pl.col("ticker") == ticker)
    execution_rows = visible.filter(pl.col("date") == session)
    if execution_rows.height != 1:
        raise LiquidityBreachError(
            _breach_message(
                "missing or ambiguous execution price",
                ticker,
                session,
                None,
                None,
                None,
            )
        )

    execution_close = float(execution_rows.get_column("close").item())
    execution_volume = int(execution_rows.get_column("volume").item())
    notional = execution_close * abs(float(shares))
    history = visible.filter(pl.col("date") < session).sort("date")
    if history.height < config.adv_window_sessions:
        raise LiquidityBreachError(
            _breach_message(
                (
                    f"trailing history {history.height}/{config.adv_window_sessions} sessions; "
                    f"reject_zero_volume={config.reject_zero_volume}"
                ),
                ticker,
                session,
                notional,
                None,
                None,
            )
        )

    trailing = history.tail(config.adv_window_sessions)
    adv = float(trailing.select((pl.col("close") * pl.col("volume")).mean()).item())
    cap = adv * config.max_adv_participation
    if config.reject_zero_volume and execution_volume == 0:
        raise LiquidityBreachError(_breach_message("zero reported volume", ticker, session, notional, adv, cap))
    if notional > cap:
        raise LiquidityBreachError(_breach_message("participation cap exceeded", ticker, session, notional, adv, cap))
