"""Reporting-only reserve usage reconstruction for the locked S8 path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final

from src.etf_manager.analytics.regimes import S8_REGIME_WINDOWS
from src.etf_manager.policy.reserve import ReserveConfig
from src.etf_manager.policy.targets import PolicyError, PolicyId
from src.etf_manager.sim.allocation import AllocationConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from src.etf_manager.sim.allocation import AllocationResult, AllocationSnapshot

__all__ = [
    "ReserveComparison",
    "ReserveUsage",
    "compare_s8_reserve",
    "summarize_reserve_usage",
]

_LEDGER_TOLERANCE: Final[float] = 1e-9
_RESERVE_WITHHOLD_CAP: Final[float] = 0.10


@dataclass(frozen=True, slots=True)
class ReserveUsage:
    """Reconstructed reserve-ledger totals over one allocation path."""

    withheld_total: float
    redeployed_total: float
    extra_investment_ratio: float
    cash_drag_ratio: float
    reserve_idle_months: int
    reserve_deployment_events: int


def summarize_reserve_usage(snapshots: Sequence[AllocationSnapshot]) -> ReserveUsage:
    """Reconstruct withhold/redeploy/idle/cash-drag totals from reserve first differences.

    PIT-free single pass O(T) over ``reserve_krw`` first differences only; delayed KRW
    redeployed onto a contribution is never counted as a second external inflow. The
    ledger identity ``withheld_total - redeployed_total == final reserve_krw`` must
    close within 1e-9.

    Raises:
        ValueError: On empty input, negative or non-finite ``reserve_krw``,
            non-finite ``contribution_krw`` / ``mark_krw``, a non-positive
            contribution sum, or a violated ledger identity.
    """
    if not snapshots:
        raise ValueError("snapshots must not be empty")
    contributions: list[float] = []
    drags: list[float] = []
    positive_deltas: list[float] = []
    negative_deltas: list[float] = []
    idle_months = 0
    deployment_events = 0
    previous_reserve = 0.0
    for snapshot in snapshots:
        for name, value in (
            ("reserve_krw", snapshot.reserve_krw),
            ("contribution_krw", snapshot.contribution_krw),
            ("mark_krw", snapshot.mark_krw),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if snapshot.reserve_krw < 0.0:
            raise ValueError(f"reserve_krw must not be negative, got {snapshot.reserve_krw!r}")
        delta = snapshot.reserve_krw - previous_reserve
        if delta >= 0.0:
            positive_deltas.append(delta)
            if snapshot.reserve_krw > 0.0:
                idle_months += 1
        else:
            negative_deltas.append(-delta)
            deployment_events += 1
        previous_reserve = snapshot.reserve_krw
        contributions.append(snapshot.contribution_krw)
        if snapshot.mark_krw > 0.0:
            drags.append(snapshot.reserve_krw / snapshot.mark_krw)
    contribution_sum = math.fsum(contributions)
    if not contribution_sum > 0.0:
        raise ValueError(f"contribution_krw sum must be positive, got {contribution_sum!r}")
    withheld_total = math.fsum(positive_deltas)
    redeployed_total = math.fsum(negative_deltas)
    if abs((withheld_total - redeployed_total) - previous_reserve) > _LEDGER_TOLERANCE:
        raise ValueError(
            "reserve ledger identity violated: withheld-redeployed="
            f"{withheld_total - redeployed_total!r} vs final reserve {previous_reserve!r}"
        )
    return ReserveUsage(
        withheld_total=withheld_total,
        redeployed_total=redeployed_total,
        extra_investment_ratio=redeployed_total / contribution_sum,
        cash_drag_ratio=math.fsum(drags) / len(drags) if drags else 0.0,
        reserve_idle_months=idle_months,
        reserve_deployment_events=deployment_events,
    )


@dataclass(frozen=True, slots=True)
class ReserveComparison:
    """Reporting-only reserve-versus-plain outcome on one regime window; never an adoption input."""

    name: str
    start: date
    end: date
    plain: AllocationResult
    reserved: AllocationResult
    plain_usage: ReserveUsage
    reserved_usage: ReserveUsage


def compare_s8_reserve(
    *,
    runner: Callable[[AllocationConfig], AllocationResult],
    contribution_krw: float,
    windows: tuple[tuple[str, date, date], ...] = S8_REGIME_WINDOWS,
    reserve: ReserveConfig | None = None,
) -> tuple[ReserveComparison, ...]:
    """Run locked S8 twice per window on identical cashflows: no reserve versus capped withhold.

    Reporting-only diagnostics (up to 2 runner calls per window): no ablation, walk-forward
    gate, or adoption decision may run here. A window whose either arm fails closed with
    ``PolicyError`` (e.g. QQQ warmup below 252 sessions) is omitted; ``ValueError``
    failures such as diverging snapshot counts still propagate.

    Raises:
        ValueError: On non-positive ``contribution_krw``, when the two arms of a
            compared window produce diverging snapshot counts, or when every
            window was omitted and no usable comparison remains.
    """
    if contribution_krw <= 0.0:
        raise ValueError(f"contribution_krw must be positive, got {contribution_krw!r}")
    reserved_config = (
        reserve if reserve is not None else ReserveConfig(max_withhold=_RESERVE_WITHHOLD_CAP)
    )
    comparisons: list[ReserveComparison] = []
    for name, start, end in windows:
        try:
            plain = runner(
                AllocationConfig(
                    policy=PolicyId.S8_US_NASDAQ,
                    start=start,
                    end=end,
                    monthly_contribution_krw=float(contribution_krw),
                )
            )
            reserved = runner(
                AllocationConfig(
                    policy=PolicyId.S8_US_NASDAQ,
                    start=start,
                    end=end,
                    monthly_contribution_krw=float(contribution_krw),
                    reserve=reserved_config,
                )
            )
        except PolicyError:
            continue
        if len(plain.snapshots) != len(reserved.snapshots):
            raise ValueError(
                f"window {name!r} snapshot lengths diverge: "
                f"{len(plain.snapshots)} vs {len(reserved.snapshots)}"
            )
        comparisons.append(
            ReserveComparison(
                name=name,
                start=start,
                end=end,
                plain=plain,
                reserved=reserved,
                plain_usage=summarize_reserve_usage(plain.snapshots),
                reserved_usage=summarize_reserve_usage(reserved.snapshots),
            )
        )
    if not comparisons:
        raise ValueError(
            f"no usable reserve comparisons over {len(windows)} windows; every window failed closed"
        )
    return tuple(comparisons)
