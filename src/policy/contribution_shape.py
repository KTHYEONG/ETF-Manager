"""Horizon-conserved monthly contribution shaping driven by the KAFI score."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

import polars as pl

from src.features.kafi import DEFAULT_CREDIT_SERIES_ID, kafi_score
from src.policy.targets import PolicyError

if TYPE_CHECKING:
    from datetime import datetime

__all__ = ["ContributionBudgetState", "ContributionShapeConfig", "shape_monthly_contribution"]

_MIN_MULTIPLIER_CEILING: Final[float] = 1.00
_MAX_MULTIPLIER_CEILING: Final[float] = 2.00
_MIN_BUDGET_WINDOW_MONTHS: Final[int] = 3
_MAX_BUDGET_WINDOW_MONTHS: Final[int] = 24
_MIN_RANK_WINDOW: Final[int] = 63
_FEASIBILITY_TOLERANCE: Final[float] = 1e-9


@dataclass(frozen=True, slots=True)
class ContributionShapeConfig:
    """KAFI-shaped contribution parameters; the band encodes realistic +/-30-50% savings swing."""

    equity_ticker: str = "QQQ"
    bond_ticker: str = "IEF"
    credit_series_id: str = DEFAULT_CREDIT_SERIES_ID
    min_multiplier: float = 0.70
    max_multiplier: float = 1.50
    budget_window_months: int = 12
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
        if not _MIN_BUDGET_WINDOW_MONTHS <= self.budget_window_months <= _MAX_BUDGET_WINDOW_MONTHS:
            raise ValueError(
                f"budget_window_months must lie in [{_MIN_BUDGET_WINDOW_MONTHS}, "
                f"{_MAX_BUDGET_WINDOW_MONTHS}], got {self.budget_window_months!r}"
            )
        if self.rank_window < _MIN_RANK_WINDOW:
            raise ValueError(f"rank_window must be at least {_MIN_RANK_WINDOW}, got {self.rank_window!r}")


@dataclass(frozen=True, slots=True)
class ContributionBudgetState:
    """Immutable I5h ledger position; callers thread replacements between months.

    Attributes:
        horizon_months: Planned credit count for the evaluation window; required.
        months_elapsed: Credits already shaped inside the horizon.
        cumulative_dev_krw: Signed drift of deployed credits versus the flat baseline;
            its final-month forced offset is what makes conservation exact.
    """

    horizon_months: int | None = None
    months_elapsed: int = 0
    cumulative_dev_krw: float = 0.0


def shape_monthly_contribution(
    *,
    base_contribution_krw: float,
    signal_at: datetime,
    prices: pl.DataFrame,
    fx: pl.DataFrame,
    macro: pl.DataFrame,
    config: ContributionShapeConfig,
    budget_state: ContributionBudgetState,
) -> tuple[float, ContributionBudgetState]:
    """Map the KAFI score onto one month's credit and project it onto I5h solvency.

    The raw multiplier ``m(s)`` is clipped into a feasibility corridor that (a)
    keeps every credit inside ``[min_multiplier, max_multiplier] * base``, (b) caps
    borrowing by the repayment capacity of the next ``budget_window_months`` at the
    maximum save-down speed, and (c) forces the final planned month to absorb the
    whole residual drift, so any completed horizon sums to exactly ``N * base``.
    Surplus months repay outstanding drift before creating slack below the base.

    Raises:
        ValueError: When the base contribution is not positive.
        PolicyError: When no horizon was planned, the horizon is exhausted, or the
            KAFI lookup fails closed on insufficient PIT history.
    """
    if base_contribution_krw <= 0:
        raise ValueError(f"base_contribution_krw must be positive, got {base_contribution_krw!r}")
    if budget_state.horizon_months is None:
        raise PolicyError("contribution shaping requires a planned horizon_months")
    remaining = budget_state.horizon_months - budget_state.months_elapsed
    if remaining <= 0:
        raise PolicyError(
            f"contribution shaping consumed all {budget_state.horizon_months} planned months"
        )
    try:
        score = kafi_score(
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
        raise PolicyError(f"contribution shaping failed closed: {exc}") from exc

    min_m = config.min_multiplier
    max_m = config.max_multiplier
    base = float(base_contribution_krw)
    raw_credit = base * (min_m + (max_m - min_m) * (100.0 - score) / 100.0)
    cum = budget_state.cumulative_dev_krw
    future = remaining - 1
    # Corridor: future months must always retain enough opposite-side room to cancel the drift.
    dev_lo = -future * (max_m - 1.0) * base - cum
    dev_hi = future * (1.0 - min_m) * base - cum
    # Rolling solvency: total borrowed drift may never exceed what the remaining
    # window could repay at the fastest legal save-down.
    window_capacity = (min(config.budget_window_months, remaining) - 1) * (1.0 - min_m) * base
    dev_hi = min(dev_hi, window_capacity - cum)
    lower = max(min_m * base, base + dev_lo)
    upper = min(max_m * base, base + dev_hi)
    if lower > upper + base * _FEASIBILITY_TOLERANCE:
        raise PolicyError("I5h projection became infeasible; refusing to emit an off-band credit")
    credit = min(max(raw_credit, lower), upper)
    # Corridor edges may disagree by float dust; snapping keeps the band exact while
    # the induced conservation drift stays orders below the 1e-6 relative identity.
    if credit < min_m * base:
        credit = min_m * base
    elif credit > max_m * base:
        credit = max_m * base
    deviation = credit - base
    next_state = replace(
        budget_state,
        months_elapsed=budget_state.months_elapsed + 1,
        cumulative_dev_krw=cum + deviation,
    )
    return credit, next_state
