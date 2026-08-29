"""Reporting-only QQQ decision-cadence diagnostics on the locked policy path."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Literal

from src.analytics.regimes import QQQ_REGIME_WINDOWS
from src.policy.targets import PolicyError, PolicyId
from src.sim.allocation import AllocationConfig, AllocationDataError

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.sim.allocation import AllocationResult

__all__ = [
    "CadenceComparison",
    "compare_qqq_cadence",
]


@dataclass(frozen=True, slots=True)
class CadenceComparison:
    """Reporting-only decision-cadence outcome on one regime window; never an adoption input."""

    name: str
    start: date
    end: date
    monthly: AllocationResult
    month_open: AllocationResult
    twice_monthly: AllocationResult


def compare_qqq_cadence(
    *,
    runner: Callable[[AllocationConfig], AllocationResult],
    contribution_krw: float,
    windows: tuple[tuple[str, date, date], ...] = QQQ_REGIME_WINDOWS,
) -> tuple[CadenceComparison, ...]:
    """Run locked QQQ thrice per window on identical cashflows: monthly versus month-open and twice-monthly.

    Reporting-only diagnostics (up to 3 runner calls per window): no ablation, walk-forward
    gate, or adoption decision may run here. A window whose any arm fails closed with
    ``PolicyError`` or ``AllocationDataError`` is omitted. ``twice_monthly`` may produce
    more snapshots than monthly/month_open; only those two arms must match.

    Raises:
        ValueError: On non-positive ``contribution_krw``, when monthly and month_open arms
            of a compared window produce diverging snapshot counts, or when every window was
            omitted and no usable comparison remains.
    """
    if contribution_krw <= 0.0:
        raise ValueError(f"contribution_krw must be positive, got {contribution_krw!r}")
    comparisons: list[CadenceComparison] = []
    cadences: tuple[Literal["monthly", "month_open", "twice_monthly"], ...] = (
        "monthly",
        "month_open",
        "twice_monthly",
    )
    for name, start, end in windows:
        try:
            results = [
                runner(
                    AllocationConfig(
                        policy=PolicyId.QQQ,
                        start=start,
                        end=end,
                        monthly_contribution_krw=float(contribution_krw),
                        cadence=cadence,
                    )
                )
                for cadence in cadences
            ]
        except PolicyError:
            continue
        except AllocationDataError:
            continue
        lengths = [len(result.snapshots) for result in results]
        if lengths[0] != lengths[1]:
            raise ValueError(f"window {name!r} monthly/month_open snapshot lengths diverge: {lengths[:2]}")
        comparisons.append(
            CadenceComparison(
                name=name,
                start=start,
                end=end,
                monthly=results[0],
                month_open=results[1],
                twice_monthly=results[2],
            )
        )
    if not comparisons:
        raise ValueError(
            f"no usable cadence comparisons over {len(windows)} windows; every window failed closed"
        )
    return tuple(comparisons)
