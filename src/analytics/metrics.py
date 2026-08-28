"""Accumulation performance metrics from a marked KRW equity curve."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence as _Seq  # noqa: F401

_XIRR_TOLERANCE: Final[float] = 1e-6
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


def real_krw(nominal_krw: float, *, cpi_index: float, cpi_base: float) -> float:
    """Deflate nominal KRW into base-period purchasing power via ``cpi_base / cpi_index``.

    Raises:
        ValueError: When any argument is non-finite or either CPI level is non-positive.
    """
    if any(not math.isfinite(value) for value in (nominal_krw, cpi_index, cpi_base)):
        raise ValueError("real_krw arguments must be finite")
    if cpi_index <= 0.0 or cpi_base <= 0.0:
        raise ValueError("cpi_index and cpi_base must be positive")
    return nominal_krw * cpi_base / cpi_index


def recovery_months(sessions: Sequence[date], marks: Sequence[float]) -> int | None:
    """Calendar months from MDD trough to first recovery at or above prior peak.

    Identifies the deepest drawdown's peak and trough, then scans forward for
    the first mark >= peak. Returns 0 when no drawdown exists and None when
    the cohort never recovers.

    Raises:
        ValueError: When lengths mismatch, either sequence is empty, or marks
            contain non-finite values.
    """
    if len(sessions) != len(marks):
        raise ValueError("sessions and marks must have equal length")
    if len(sessions) < 1:
        raise ValueError("sessions and marks must be non-empty")
    for value in marks:
        if not math.isfinite(float(value)):
            raise ValueError(f"marks must be finite, got {value!r}")

    peak_val = float(marks[0])
    max_dd = 0.0
    trough_idx = 0
    peak_for_trough_val = peak_val

    for idx, raw in enumerate(marks):
        val = float(raw)
        if val > peak_val:
            peak_val = val
        elif peak_val > 0.0:
            dd = val / peak_val - 1.0
            if dd < max_dd:
                max_dd = dd
                trough_idx = idx
                peak_for_trough_val = peak_val

    if max_dd == 0.0:
        return 0

    trough_date = sessions[trough_idx]
    for idx in range(trough_idx + 1, len(marks)):
        if float(marks[idx]) >= peak_for_trough_val:
            rec_date = sessions[idx]
            return (rec_date.year - trough_date.year) * 12 + (rec_date.month - trough_date.month)
    return None


def _net_present_value(amounts: tuple[float, ...], times: tuple[float, ...], rate: float) -> float:
    total = 0.0
    for step, amount in zip(times, amounts, strict=True):
        total += amount * (1.0 + rate) ** -step
    return total
