"""Reporting-only S8 decision-cadence diagnostics on the locked policy path."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from src.etf_manager.analytics.regimes import S8_REGIME_WINDOWS
from src.etf_manager.policy.targets import PolicyError, PolicyId
from src.etf_manager.sim.allocation import AllocationConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.etf_manager.sim.allocation import AllocationResult

__all__ = [
    "CadenceComparison",
    "compare_s8_cadence",
]


@dataclass(frozen=True, slots=True)
class CadenceComparison:
    """Reporting-only month-open-versus-monthly outcome on one regime window; never an adoption input."""

    name: str
    start: date
    end: date
    monthly: AllocationResult
    month_open: AllocationResult


def compare_s8_cadence(
    *,
    runner: Callable[[AllocationConfig], AllocationResult],
    contribution_krw: float,
    windows: tuple[tuple[str, date, date], ...] = S8_REGIME_WINDOWS,
) -> tuple[CadenceComparison, ...]:
    """Run locked S8 twice per window on identical cashflows: default monthly versus month-open cadence.

    Reporting-only diagnostics (up to 2 runner calls per window): no ablation, walk-forward
    gate, or adoption decision may run here. A window whose either arm fails closed with
    ``PolicyError`` is omitted; ``ValueError`` failures such as diverging snapshot counts
    still propagate.

    Raises:
        ValueError: On non-positive ``contribution_krw``, when the two arms of a compared
            window produce diverging snapshot counts, or when every window was omitted and
            no usable comparison remains.
    """
    if contribution_krw <= 0.0:
        raise ValueError(f"contribution_krw must be positive, got {contribution_krw!r}")
    comparisons: list[CadenceComparison] = []
    for name, start, end in windows:
        try:
            monthly = runner(
                AllocationConfig(
                    policy=PolicyId.S8_US_NASDAQ,
                    start=start,
                    end=end,
                    monthly_contribution_krw=float(contribution_krw),
                )
            )
            month_open = runner(
                AllocationConfig(
                    policy=PolicyId.S8_US_NASDAQ,
                    start=start,
                    end=end,
                    monthly_contribution_krw=float(contribution_krw),
                    cadence="month_open",
                )
            )
        except PolicyError:
            continue
        if len(monthly.snapshots) != len(month_open.snapshots):
            raise ValueError(
                f"window {name!r} snapshot lengths diverge: "
                f"{len(monthly.snapshots)} vs {len(month_open.snapshots)}"
            )
        comparisons.append(
            CadenceComparison(
                name=name,
                start=start,
                end=end,
                monthly=monthly,
                month_open=month_open,
            )
        )
    if not comparisons:
        raise ValueError(
            f"no usable cadence comparisons over {len(windows)} windows; every window failed closed"
        )
    return tuple(comparisons)
