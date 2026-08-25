"""Explicit contribution reserve ledger driven by mutually exclusive PIT rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import polars as pl

from src.features.drawdown import trailing_price_drawdown
from src.features.momentum import trailing_compound_return
from src.features.returns import session_returns
from src.policy.targets import PolicyError

if TYPE_CHECKING:
    from datetime import datetime

__all__ = ["ReserveConfig", "ReserveDecision", "apply_reserve_schedule"]

_MAX_WITHHOLD_CEILING: Final[float] = 0.10
_MIN_INVEST_CEILING: Final[float] = 1.00
_MAX_INVEST_CEILING: Final[float] = 2.00
_RESERVE_MONTHS_CEILING: Final[float] = 6.00
_SHALLOW_DEPTH: Final[float] = 0.10
_MODERATE_DEPTH: Final[float] = 0.20
_DEEP_DEPTH: Final[float] = 0.30


@dataclass(frozen=True, slots=True)
class ReserveConfig:
    """Reserve ledger parameters for the binary ``v1`` or piecewise ``v2`` schedule."""

    max_withhold: float
    trend_window: int = 126
    drawdown_window: int = 252
    drawdown_trigger: float = -0.15
    schedule: Literal["v1", "v2"] = "v1"
    min_invest_multiplier: float = 0.80
    max_invest_multiplier: float = 2.00
    reserve_max_months: float = 6.00

    def __post_init__(self) -> None:
        if not 0.0 < self.max_withhold <= _MAX_WITHHOLD_CEILING:
            raise ValueError(
                f"max_withhold must lie in (0, {_MAX_WITHHOLD_CEILING}], got {self.max_withhold!r}"
            )
        if self.schedule not in ("v1", "v2"):
            raise ValueError(f"schedule must be 'v1' or 'v2', got {self.schedule!r}")
        if not 0.0 < self.min_invest_multiplier < _MIN_INVEST_CEILING:
            raise ValueError(
                f"min_invest_multiplier must lie in (0, {_MIN_INVEST_CEILING}), "
                f"got {self.min_invest_multiplier!r}"
            )
        if not _MIN_INVEST_CEILING < self.max_invest_multiplier <= _MAX_INVEST_CEILING:
            raise ValueError(
                f"max_invest_multiplier must lie in ({_MIN_INVEST_CEILING}, {_MAX_INVEST_CEILING}], "
                f"got {self.max_invest_multiplier!r}"
            )
        if not 0.0 < self.reserve_max_months <= _RESERVE_MONTHS_CEILING:
            raise ValueError(
                f"reserve_max_months must lie in (0, {_RESERVE_MONTHS_CEILING}], "
                f"got {self.reserve_max_months!r}"
            )


def _piecewise_multiplier(
    *, depth: float, trend: float, min_invest_multiplier: float, max_invest_multiplier: float
) -> float:
    """Map trailing drawdown depth onto the deterministic v2 invest multiplier."""
    if depth < _SHALLOW_DEPTH:
        return min_invest_multiplier if trend > 0.0 else 1.00
    if depth < _MODERATE_DEPTH:
        return 1.25 + 0.25 * (depth - _SHALLOW_DEPTH) / _SHALLOW_DEPTH
    if depth < _DEEP_DEPTH:
        return 1.50 + 0.50 * (depth - _MODERATE_DEPTH) / _SHALLOW_DEPTH
    return max_invest_multiplier


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

    Schedule ``v1`` keeps the mutually exclusive rules on a single PIT feature pair per
    call: deploy ``min(reserve_krw, cap)`` back into investable KRW when the trailing
    drawdown is at or below ``config.drawdown_trigger``; otherwise withhold ``cap`` when
    the trailing compound trend is positive; otherwise pass the contribution through
    untouched. Schedule ``v2`` scales the contribution by a deterministic piecewise
    multiplier of depth and trend funded only from the explicit reserve stock; overflow
    above the ``reserve_max_months`` stock cap is invested instead of withheld. The
    returned ``reserve_krw`` never goes negative and both schedules close the ledger
    identity ``investable + reserve == contribution + old reserve``. Feature failures
    fail closed with ``PolicyError``.

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

    if config.schedule == "v2":
        stock = contribution_krw + reserve_krw
        desired = (
            _piecewise_multiplier(
                depth=-drawdown,
                trend=trend,
                min_invest_multiplier=config.min_invest_multiplier,
                max_invest_multiplier=config.max_invest_multiplier,
            )
            * contribution_krw
        )
        investable = min(desired, stock)
        new_reserve = stock - investable
        stock_cap = config.reserve_max_months * contribution_krw
        if new_reserve > stock_cap:
            new_reserve = stock_cap
            investable = stock - new_reserve
        return ReserveDecision(investable_krw=investable, reserve_krw=new_reserve)

    cap = config.max_withhold * contribution_krw
    if drawdown <= config.drawdown_trigger:
        deploy = min(reserve_krw, cap)
        return ReserveDecision(investable_krw=contribution_krw + deploy, reserve_krw=reserve_krw - deploy)
    if trend > 0.0:
        return ReserveDecision(investable_krw=contribution_krw - cap, reserve_krw=reserve_krw + cap)
    return ReserveDecision(investable_krw=contribution_krw, reserve_krw=reserve_krw)
