"""Causal KAFI deployment: fixed external credits with an explicit KRW reserve ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import polars as pl

from src.features.kafi import DEFAULT_CREDIT_SERIES_ID, kafi_opportunity_score
from src.policy.targets import PolicyError

if TYPE_CHECKING:
    from datetime import datetime

__all__ = ["KafiDeploymentConfig", "KafiDeploymentDecision", "apply_kafi_deployment"]

_MIN_MULTIPLIER_CEILING: Final[float] = 1.00
_MAX_MULTIPLIER_CEILING: Final[float] = 1.50
_MIN_RANK_WINDOW: Final[int] = 63
_LEDGER_TOLERANCE: Final[float] = 1e-9


@dataclass(frozen=True, slots=True)
class KafiDeploymentConfig:
    """Opportunity-oriented KAFI sizing with symmetric deploy band around a neutral score of 50."""

    equity_ticker: str = "QQQ"
    bond_ticker: str = "IEF"
    credit_series_id: str = DEFAULT_CREDIT_SERIES_ID
    min_multiplier: float = 0.70
    max_multiplier: float = 1.30
    rank_window: int = 252

    def __post_init__(self) -> None:
        if not self.equity_ticker or not self.bond_ticker or not self.credit_series_id:
            raise ValueError("equity_ticker, bond_ticker, and credit_series_id must be non-empty")
        if not 0.0 < self.min_multiplier < _MIN_MULTIPLIER_CEILING:
            raise ValueError(
                f"min_multiplier must lie in (0, {_MIN_MULTIPLIER_CEILING}), got {self.min_multiplier!r}"
            )
        if not _MIN_MULTIPLIER_CEILING < self.max_multiplier <= _MAX_MULTIPLIER_CEILING:
            raise ValueError(
                f"max_multiplier must lie in ({_MIN_MULTIPLIER_CEILING}, {_MAX_MULTIPLIER_CEILING}], "
                f"got {self.max_multiplier!r}"
            )
        if self.rank_window < _MIN_RANK_WINDOW:
            raise ValueError(f"rank_window must be at least {_MIN_RANK_WINDOW}, got {self.rank_window!r}")


@dataclass(frozen=True, slots=True)
class KafiDeploymentDecision:
    """One month's split of the credited contribution into investable KRW and reserve."""

    investable_krw: float
    reserve_krw: float


def apply_kafi_deployment(
    *,
    contribution_krw: float,
    reserve_krw: float,
    signal_at: datetime,
    prices: pl.DataFrame,
    fx: pl.DataFrame,
    macro: pl.DataFrame,
    config: KafiDeploymentConfig,
) -> KafiDeploymentDecision:
    """Deploy only from the explicit stock ``contribution + reserve`` using the opportunity score.

    The raw multiplier rises with the opportunity score and is capped by the available stock,
    so cumulative deploy never exceeds cumulative external inflow. The ledger identity
    ``investable + reserve == contribution + old_reserve`` closes every month without
    borrowing future credits.

    Raises:
        ValueError: When ``contribution_krw`` is not positive.
        PolicyError: When the opportunity lookup fails closed on insufficient PIT history.
    """
    if contribution_krw <= 0:
        raise ValueError(f"contribution_krw must be positive, got {contribution_krw!r}")
    try:
        score = kafi_opportunity_score(
            prices=prices,
            fx=fx,
            macro=macro,
            equity_ticker=config.equity_ticker,
            bond_ticker=config.bond_ticker,
            signal_at=signal_at,
            rank_window=config.rank_window,
            credit_series_id=config.credit_series_id,
        )
    except ValueError as exc:
        raise PolicyError(f"kafi deployment failed closed: {exc}") from exc

    min_m = config.min_multiplier
    max_m = config.max_multiplier
    multiplier = min_m + (max_m - min_m) * score / 100.0
    stock = float(contribution_krw) + float(reserve_krw)
    raw_investable = float(contribution_krw) * multiplier
    investable = min(raw_investable, stock)
    new_reserve = stock - investable
    if new_reserve < -_LEDGER_TOLERANCE:
        raise PolicyError("kafi deployment reserve became negative")
    if new_reserve < 0.0:
        new_reserve = 0.0
    return KafiDeploymentDecision(investable_krw=investable, reserve_krw=new_reserve)
