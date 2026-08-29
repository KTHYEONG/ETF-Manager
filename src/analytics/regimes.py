"""Reporting-only QQQ regime windows versus the locked VTI baseline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final

from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig, AllocationDataError

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.sim.allocation import AllocationResult

logger = logging.getLogger(__name__)

__all__ = [
    "QQQ_REGIME_WINDOWS",
    "RegimeComparison",
    "compare_policy_regimes",
]

QQQ_REGIME_WINDOWS: Final[tuple[tuple[str, date, date], ...]] = (
    ("calendar_max", date(2012, 8, 31), date(2024, 8, 31)),
    ("gfc_crisis", date(2007, 10, 1), date(2009, 3, 31)),
    ("pre_ai", date(2010, 1, 4), date(2019, 12, 31)),
    ("shipped_old", date(2014, 1, 3), date(2024, 8, 31)),
    ("bear_2022", date(2022, 1, 3), date(2022, 12, 30)),
    ("recent_2023_2024", date(2023, 1, 3), date(2024, 8, 31)),
)


@dataclass(frozen=True, slots=True)
class RegimeComparison:
    """Reporting-only QQQ-versus-VTI outcome on one regime window; never an adoption input."""

    name: str
    start: date
    end: date
    baseline: AllocationResult
    candidate: AllocationResult


def compare_policy_regimes(
    *,
    runner: Callable[[AllocationConfig], AllocationResult],
    contribution_krw: float,
    windows: tuple[tuple[str, date, date], ...] = QQQ_REGIME_WINDOWS,
) -> tuple[RegimeComparison, ...]:
    """Run VTI then QQQ per window on identical external cashflows.

    Both arms share the window's start/end and the monthly contribution, so any
    wealth gap is policy-driven. Reporting-only diagnostics: no gate and no
    adoption decision may run here.

    Raises:
        ValueError: On non-positive ``contribution_krw``, or when the two arms of
            a window produce diverging snapshot counts.
    """
    if contribution_krw <= 0.0:
        raise ValueError(f"contribution_krw must be positive, got {contribution_krw!r}")
    comparisons: list[RegimeComparison] = []
    for name, start, end in windows:
        try:
            baseline = runner(
                AllocationConfig(
                    policy=PolicyId.VTI,
                    start=start,
                    end=end,
                    monthly_contribution_krw=float(contribution_krw),
                )
            )
            candidate = runner(
                AllocationConfig(
                    policy=PolicyId.QQQ,
                    start=start,
                    end=end,
                    monthly_contribution_krw=float(contribution_krw),
                )
            )
        except AllocationDataError as exc:
            logger.warning("[DATA] event=regime_window_skipped name=%s reason=%s", name, exc)
            continue
        if len(baseline.snapshots) != len(candidate.snapshots):
            raise ValueError(
                f"window {name!r} snapshot lengths diverge: "
                f"{len(baseline.snapshots)} vs {len(candidate.snapshots)}"
            )
        comparisons.append(
            RegimeComparison(name=name, start=start, end=end, baseline=baseline, candidate=candidate)
        )
    if not comparisons:
        raise ValueError(f"no usable regime comparisons over {len(windows)} windows; every window lacked catalog coverage")
    return tuple(comparisons)
