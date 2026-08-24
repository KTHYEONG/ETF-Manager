"""Explicit contribution reserve ledger driven by mutually exclusive PIT rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import polars as pl

from src.etf_manager.features.drawdown import trailing_price_drawdown
from src.etf_manager.features.momentum import trailing_compound_return
from src.etf_manager.features.returns import session_returns
from src.etf_manager.policy.targets import PolicyError

if TYPE_CHECKING:
    from datetime import datetime

__all__ = ["ReserveConfig", "ReserveDecision", "apply_reserve_schedule"]

_MAX_WITHHOLD_CEILING: Final[float] = 0.10


@dataclass(frozen=True, slots=True)
class ReserveConfig:
    """Reserve ledger parameters; at most ``max_withhold`` of one contribution moves."""

    max_withhold: float
    trend_window: int = 126
    drawdown_window: int = 252
    drawdown_trigger: float = -0.15

    def __post_init__(self) -> None:
        if not 0.0 < self.max_withhold <= _MAX_WITHHOLD_CEILING:
            raise ValueError(
                f"max_withhold must lie in (0, {_MAX_WITHHOLD_CEILING}], got {self.max_withhold!r}"
            )


@dataclass(frozen=True, slots=True)
class ReserveDecision:
    """One month's split of the credited contribution into investable KRW and reserve."""

    investable_krw: float
    reserve_krw: float


def apply_reserve_schedule(
    *,
    contribution_krw: float,
    reserve_krw: float,
    prices: pl.DataFrame,
    ticker: str,
    signal_at: datetime,
    config: ReserveConfig,
) -> ReserveDecision:
    """Split one credited contribution using only signals visible at ``signal_at``.

    Mutually exclusive rules on a single PIT feature pair per call: deploy
    ``min(reserve_krw, cap)`` back into investable KRW when the trailing drawdown is at
    or below ``config.drawdown_trigger``; otherwise withhold ``cap`` when the trailing
    compound trend is positive; otherwise pass the contribution through untouched.
    The returned ``reserve_krw`` never goes negative. Feature failures fail closed
    with ``PolicyError``.

    Raises:
        PolicyError: When ``signal_at`` is naive or either feature window is unusable.
    """
    try:
        drawdown = trailing_price_drawdown(
            prices, ticker=ticker, as_of_ts=signal_at, window=config.drawdown_window
        )
        trend = trailing_compound_return(
            session_returns(prices, ticker=ticker), as_of_ts=signal_at, window=config.trend_window
        )
    except ValueError as exc:
        raise PolicyError(f"reserve feature failed closed: {exc}") from exc

    cap = config.max_withhold * contribution_krw
    if drawdown <= config.drawdown_trigger:
        deploy = min(reserve_krw, cap)
        return ReserveDecision(investable_krw=contribution_krw + deploy, reserve_krw=reserve_krw - deploy)
    if trend > 0.0:
        return ReserveDecision(investable_krw=contribution_krw - cap, reserve_krw=reserve_krw + cap)
    return ReserveDecision(investable_krw=contribution_krw, reserve_krw=reserve_krw)
