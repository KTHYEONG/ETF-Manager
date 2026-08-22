"""Accumulation performance metrics from a marked KRW equity curve."""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

_XIRR_TOLERANCE: Final[float] = 1e-8
_XIRR_MAX_ITERATIONS: Final[int] = 50
_DAYS_PER_YEAR: Final[float] = 365.25
_SECONDS_PER_DAY: Final[float] = 86400.0


class XirrError(ValueError):
    """Raised when the money-weighted rate cannot be identified."""


def xirr(cashflows: Sequence[tuple[datetime, float]]) -> float:
    """Money-weighted annual rate; fails closed if Newton does not converge.

    Year fractions measure ``(t - t0)`` on a 365.25-day year from the earliest
    timestamp. Raises XirrError when amounts lack a sign change or |NPV| stays
    above tolerance after 50 iterations.
    """
    if not cashflows:
        raise XirrError("xirr requires at least one cashflow")
    ordered = sorted(cashflows, key=lambda item: item[0])
    origin = ordered[0][0]
    times = tuple((when - origin).total_seconds() / (_DAYS_PER_YEAR * _SECONDS_PER_DAY) for when, _ in ordered)
    amounts = tuple(amount for _, amount in ordered)
    if not (any(amount > 0.0 for amount in amounts) and any(amount < 0.0 for amount in amounts)):
        raise XirrError("cashflows must contain both negative and positive legs")

    rate = 0.1
    npv = _net_present_value(amounts, times, rate)
    for _ in range(_XIRR_MAX_ITERATIONS):
        if abs(npv) <= _XIRR_TOLERANCE:
            return rate
        derivative = sum(
            -step * amount * (1.0 + rate) ** (-step - 1.0) for step, amount in zip(times, amounts, strict=True)
        )
        candidate = rate - npv / derivative if derivative != 0.0 and math.isfinite(derivative) else rate - max(npv, 1.0)
        while candidate <= -1.0:
            candidate = (rate + candidate) / 2.0
        if not math.isfinite(candidate):
            break
        rate = candidate
        npv = _net_present_value(amounts, times, rate)
    if abs(npv) > _XIRR_TOLERANCE:
        raise XirrError(f"xirr did not converge: |NPV|={abs(npv):.3e} after {_XIRR_MAX_ITERATIONS} iterations")
    return rate


def max_drawdown(equity_krw: Sequence[float]) -> float:
    """Peak-to-trough decline as a non-positive fraction of the running peak."""
    if not equity_krw:
        raise ValueError("max_drawdown requires a non-empty equity series")
    peak = equity_krw[0]
    drawdown = 0.0
    for value in equity_krw:
        if value > peak:
            peak = value
        elif peak > 0.0:
            drawdown = min(drawdown, value / peak - 1.0)
    return max(drawdown, -1.0)


def _net_present_value(amounts: tuple[float, ...], times: tuple[float, ...], rate: float) -> float:
    total = 0.0
    for step, amount in zip(times, amounts, strict=True):
        total += amount * (1.0 + rate) ** -step
    return total
