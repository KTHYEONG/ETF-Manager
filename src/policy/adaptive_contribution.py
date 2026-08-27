"""Stateless adaptive monthly contribution sized by the KAFI opportunity score."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import polars as pl

from src.features.kafi import DEFAULT_CREDIT_SERIES_ID, kafi_opportunity_score
from src.policy.targets import PolicyError

if TYPE_CHECKING:
    from datetime import datetime

__all__ = [
    "OPERATIONAL_ADAPTIVE_CONTRIBUTION",
    "AdaptiveContributionConfig",
    "size_adaptive_contribution",
]

_MIN_MULTIPLIER_FLOOR: Final[float] = 0.00
_MIN_MULTIPLIER_CEILING: Final[float] = 1.00
_MAX_MULTIPLIER_FLOOR: Final[float] = 1.00
_MAX_MULTIPLIER_CEILING: Final[float] = 2.00
_NEUTRAL_SCORE: Final[float] = 50.0
_MIN_RANK_WINDOW: Final[int] = 63


@dataclass(frozen=True, slots=True)
class AdaptiveContributionConfig:
    """Piecewise KAFI-opportunity band around the neutral base credit; no ledger state.

    The multiplier is ``1 - (1 - min_multiplier) * ((50 - score) / 50) ** downside_power``
    below the neutral score of 50 and ``1 + (max_multiplier - 1) *
    ((score - 50) / 50) ** upside_power`` at or above it, so score 0 emits nothing,
    score 50 emits the base credit, and score 100 emits twice the base.
    """

    equity_ticker: str = "QQQ"
    bond_ticker: str = "IEF"
    credit_series_id: str = DEFAULT_CREDIT_SERIES_ID
    min_multiplier: float = 0.0
    max_multiplier: float = 2.0
    downside_power: float = 3.5
    upside_power: float = 0.35
    rank_window: int = 126
    include_vol_dampener: bool = False
    dispersion: float = 1.15
    neutral_deadband: float = 4.0

    def __post_init__(self) -> None:
        if not self.equity_ticker or not self.bond_ticker or not self.credit_series_id:
            raise ValueError("equity_ticker, bond_ticker, and credit_series_id must be non-empty")
        if (
            not math.isfinite(self.min_multiplier)
            or not _MIN_MULTIPLIER_FLOOR <= self.min_multiplier < _MIN_MULTIPLIER_CEILING
        ):
            raise ValueError(
                f"min_multiplier must lie in [{_MIN_MULTIPLIER_FLOOR}, {_MIN_MULTIPLIER_CEILING}), "
                f"got {self.min_multiplier!r}"
            )
        if (
            not math.isfinite(self.max_multiplier)
            or not _MAX_MULTIPLIER_FLOOR < self.max_multiplier <= _MAX_MULTIPLIER_CEILING
        ):
            raise ValueError(
                f"max_multiplier must lie in ({_MAX_MULTIPLIER_FLOOR}, {_MAX_MULTIPLIER_CEILING}], "
                f"got {self.max_multiplier!r}"
            )
        for name in ("downside_power", "upside_power"):
            power = float(getattr(self, name))
            if not math.isfinite(power) or power <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {power!r}")
        if self.rank_window < _MIN_RANK_WINDOW:
            raise ValueError(f"rank_window must be at least {_MIN_RANK_WINDOW}, got {self.rank_window!r}")
        if not math.isfinite(self.dispersion) or self.dispersion <= 0.0:
            raise ValueError(f"dispersion must be finite and positive, got {self.dispersion!r}")
        if not math.isfinite(self.neutral_deadband) or self.neutral_deadband < 0.0:
            raise ValueError(f"neutral_deadband must be finite and non-negative, got {self.neutral_deadband!r}")


def size_adaptive_contribution(
    *,
    base_contribution_krw: float,
    signal_at: datetime,
    prices: pl.DataFrame,
    fx: pl.DataFrame,
    macro: pl.DataFrame,
    config: AdaptiveContributionConfig,
) -> float:
    """Size one month's external credit from the PIT KAFI opportunity score.

    Every call is independent: no horizon sum conservation, debt, terminal settlement,
    or reserve state is accepted or maintained. The emitted credit lies inside
    ``[min_multiplier, max_multiplier] * base`` (defaults ``[0, 2 * base]``), so a zero
    month is legal while the caller owns the all-zero-path verdict.

    Raises:
        ValueError: When ``base_contribution_krw`` is not finite and positive.
        PolicyError: When the opportunity lookup fails closed on insufficient PIT history.
    """
    base = float(base_contribution_krw)
    if not math.isfinite(base) or base <= 0.0:
        raise ValueError(f"base_contribution_krw must be positive, got {base_contribution_krw!r}")
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
            include_vol_dampener=config.include_vol_dampener,
            dispersion=config.dispersion,
        )
    except ValueError as exc:
        raise PolicyError(f"adaptive contribution failed closed: {exc}") from exc

    if abs(score - _NEUTRAL_SCORE) <= config.neutral_deadband:
        score = _NEUTRAL_SCORE

    if score < _NEUTRAL_SCORE:
        t = min(max((_NEUTRAL_SCORE - score) / _NEUTRAL_SCORE, 0.0), 1.0)
        multiplier = 1.0 - (1.0 - config.min_multiplier) * math.pow(t, config.downside_power)
    else:
        t = min(max((score - _NEUTRAL_SCORE) / _NEUTRAL_SCORE, 0.0), 1.0)
        multiplier = 1.0 + (config.max_multiplier - 1.0) * math.pow(t, config.upside_power)
    credit = min(max(multiplier * base, config.min_multiplier * base), config.max_multiplier * base)
    if not math.isfinite(credit):
        raise PolicyError("adaptive contribution collapsed to a non-finite credit")
    return credit


# WF-adopted QQQ sizing (rank 126, no_vol, deadband 4); locked on the operational path.
OPERATIONAL_ADAPTIVE_CONTRIBUTION: Final[AdaptiveContributionConfig] = AdaptiveContributionConfig(
    rank_window=126,
    downside_power=3.5,
    upside_power=0.35,
    include_vol_dampener=False,
    dispersion=1.15,
    neutral_deadband=4.0,
    min_multiplier=0.0,
    max_multiplier=2.0,
)
