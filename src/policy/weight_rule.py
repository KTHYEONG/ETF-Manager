"""Target-weight rule contract for the after-tax accumulation engine (L3 policy)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

from src.features.pit_market import PitMarket

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

CASH_SLEEVE: Final[str] = "CASH"

__all__ = ["CASH_SLEEVE", "WeightRule"]


class WeightRule(Protocol):
    """PIT target-weight provider evaluated at each month-end signal instant.

    Returns a nonnegative simplex over ETF tickers plus optional ``CASH_SLEEVE`` (USD
    held at the T-bill rate). ``tickers`` lists every ETF the rule may return (data
    loading and warmup checks); ``requires_cash_rate`` declares whether RATES is needed.
    Lives in the policy layer so rule implementations never import the simulator.
    """

    @property
    def tickers(self) -> frozenset[str]:
        """Every ETF ticker this rule may weight."""
        ...

    @property
    def requires_cash_rate(self) -> bool:
        """Whether the rule reads RATES through the market view."""
        ...

    def __call__(self, signal_at: datetime, market: PitMarket) -> Mapping[str, float]:
        """Target weights at ``signal_at``; a nonnegative simplex over tickers plus cash."""
        ...
