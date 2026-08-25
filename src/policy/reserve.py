"""Explicit contribution reserve ledger driven by mutually exclusive PIT rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import polars as pl

from src.features.drawdown import trailing_price_drawdown
from src.features.momentum import trailing_compound_return
from src.features.returns import session_returns
from src.policy.overlay import visible_macro_level
from src.policy.targets import PolicyError

if TYPE_CHECKING:
    from datetime import datetime

__all__ = ["ReserveConfig", "ReserveDecision", "apply_reserve_schedule"]

_MAX_WITHHOLD_CEILING: Final[float] = 0.10
_MIN_INVEST_CEILING: Final[float] = 1.00
_MAX_INVEST_CEILING: Final[float] = 2.00
_V3_MAX_INVEST_CEILING: Final[float] = 3.00
_RESERVE_MONTHS_CEILING: Final[float] = 6.00
_SHALLOW_DEPTH: Final[float] = 0.10
_MODERATE_DEPTH: Final[float] = 0.20
_DEEP_DEPTH: Final[float] = 0.30
_LEGACY_MIN_INVEST: Final[float] = 0.80
_LEGACY_MAX_INVEST: Final[float] = 2.00
_V3_MIN_INVEST: Final[float] = 0.70
_DEFAULT_VIX_THRESHOLD: Final[float] = 20.0


@dataclass(frozen=True, slots=True)
class ReserveConfig:
    """Reserve ledger parameters for the binary ``v1``, piecewise ``v2``, or ERP ``v3`` schedule."""

    max_withhold: float
    trend_window: int = 126
    drawdown_window: int = 252
    drawdown_trigger: float = -0.15
    schedule: Literal["v1", "v2", "v3"] = "v1"
    min_invest_multiplier: float = _LEGACY_MIN_INVEST
    max_invest_multiplier: float = _LEGACY_MAX_INVEST
    reserve_max_months: float = 6.00
    vix_series_id: str = "VIXCLS"
    vix_threshold: float = _DEFAULT_VIX_THRESHOLD

    def __post_init__(self) -> None:
        if not 0.0 < self.max_withhold <= _MAX_WITHHOLD_CEILING:
            raise ValueError(
                f"max_withhold must lie in (0, {_MAX_WITHHOLD_CEILING}], got {self.max_withhold!r}"
            )
        if self.schedule not in ("v1", "v2", "v3"):
            raise ValueError(f"schedule must be 'v1', 'v2', or 'v3', got {self.schedule!r}")
        if self.schedule == "v3":
            # Legacy v1/v2 baselines count as unset and rebase onto the wider v3 band.
            if self.min_invest_multiplier == _LEGACY_MIN_INVEST:
                object.__setattr__(self, "min_invest_multiplier", _V3_MIN_INVEST)
            if self.max_invest_multiplier == _LEGACY_MAX_INVEST:
                object.__setattr__(self, "max_invest_multiplier", _V3_MAX_INVEST_CEILING)
        if not 0.0 < self.min_invest_multiplier < _MIN_INVEST_CEILING:
            raise ValueError(
                f"min_invest_multiplier must lie in (0, {_MIN_INVEST_CEILING}), "
                f"got {self.min_invest_multiplier!r}"
            )
        max_invest_ceiling = _V3_MAX_INVEST_CEILING if self.schedule == "v3" else _MAX_INVEST_CEILING
        if not _MIN_INVEST_CEILING < self.max_invest_multiplier <= max_invest_ceiling:
            raise ValueError(
                f"max_invest_multiplier must lie in ({_MIN_INVEST_CEILING}, {max_invest_ceiling}], "
                f"got {self.max_invest_multiplier!r}"
            )
        if not 0.0 < self.reserve_max_months <= _RESERVE_MONTHS_CEILING:
            raise ValueError(
                f"reserve_max_months must lie in (0, {_RESERVE_MONTHS_CEILING}], "
                f"got {self.reserve_max_months!r}"
            )
        if not self.vix_threshold > 0.0:
            raise ValueError(f"vix_threshold must be positive, got {self.vix_threshold!r}")


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


def _erp_multiplier(
    *,
    depth: float,
    vix: float,
    vix_threshold: float,
    min_invest_multiplier: float,
    max_invest_multiplier: float,
) -> float:
    """Map drawdown depth or VIX stress (whichever is cheaper) onto the v3 invest multiplier."""
    cheap = max(
        min(1.0, max(0.0, depth) / _DEEP_DEPTH),
        min(1.0, max(0.0, vix / vix_threshold - 1.0)),
    )
    if cheap == 0.0:
        return min_invest_multiplier
    return 1.0 + cheap * (max_invest_multiplier - 1.0)


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
    macro: pl.DataFrame | None = None,
) -> ReserveDecision:
    """Split one credited contribution using only signals visible at ``signal_at``.

    Schedule ``v1`` keeps the mutually exclusive rules on a single PIT feature pair per
    call: deploy ``min(reserve_krw, cap)`` back into investable KRW when the trailing
    drawdown is at or below ``config.drawdown_trigger``; otherwise withhold ``cap`` when
    the trailing compound trend is positive; otherwise pass the contribution through
    untouched. Schedules ``v2`` and ``v3`` scale the contribution by a deterministic
    invest multiplier funded only from the explicit reserve stock; overflow above the
    ``reserve_max_months`` stock cap is invested instead of withheld. ``v2`` keys the
    multiplier on depth and trend alone; ``v3`` keys it on depth or VIX stress via the
    latest macro row visible at the signal instant, so it fails closed without a usable
    macro frame while ``v1`` and ``v2`` ignore ``macro`` entirely. The returned
    ``reserve_krw`` never goes negative and every schedule closes the ledger identity
    ``investable + reserve == contribution + old reserve``. Feature failures fail closed
    with ``PolicyError``.

    Raises:
        PolicyError: When ``signal_at`` is naive, either feature window is unusable, or
            a ``v3`` decision lacks a visible finite macro row for its series id.
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

    if config.schedule != "v1":
        if config.schedule == "v3":
            multiplier = _erp_multiplier(
                depth=-drawdown,
                vix=visible_macro_level(macro, config.vix_series_id, signal_at),
                vix_threshold=config.vix_threshold,
                min_invest_multiplier=config.min_invest_multiplier,
                max_invest_multiplier=config.max_invest_multiplier,
            )
        else:
            multiplier = _piecewise_multiplier(
                depth=-drawdown,
                trend=trend,
                min_invest_multiplier=config.min_invest_multiplier,
                max_invest_multiplier=config.max_invest_multiplier,
            )
        stock = contribution_krw + reserve_krw
        investable = min(multiplier * contribution_krw, stock)
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
