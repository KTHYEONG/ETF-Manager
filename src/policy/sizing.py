"""ERP-preserving invest-multiplier sizing from PIT depth-trend regimes."""

from __future__ import annotations

from typing import Final

__all__ = ["erp_preserving_multiplier"]

# Same regime cutoffs as the reserve ledger; duplicated to avoid a sizing -> reserve import cycle.
_SHALLOW_DEPTH: Final[float] = 0.10
_MODERATE_DEPTH: Final[float] = 0.20


def erp_preserving_multiplier(
    *, depth: float, trend: float, min_invest_multiplier: float, max_invest_multiplier: float
) -> float:
    """Map trailing drawdown depth onto the v4 ERP-preserving invest multiplier.

    A positive trailing trend keeps the contribution whole until the drawdown is
    at least moderate, where it deploys at ``max_invest_multiplier``; a non-positive
    trend passes through in shallow drawdowns, withholds to ``min_invest_multiplier``
    in moderate ones, and deploys at maximum only once the drawdown is deep enough.
    """
    if trend > 0.0:
        return max_invest_multiplier if depth >= _MODERATE_DEPTH else 1.0
    if depth < _SHALLOW_DEPTH:
        return 1.0
    if depth < _MODERATE_DEPTH:
        return min_invest_multiplier
    return max_invest_multiplier
