"""Reporting-only QQQ drawdown-blend recipe ratios versus the locked QQQ and VTI paths."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final

from src.analytics.regimes import QQQ_REGIME_WINDOWS
from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.sim.allocation import AllocationResult

__all__ = [
    "QQQ_BLEND_RECIPES",
    "BlendComparison",
    "compare_qqq_blends",
]

QQQ_BLEND_RECIPES: Final[tuple[tuple[str, dict[str, float]], ...]] = (
    ("qqq", {"QQQ": 1.0}),
    ("qqq90_vti10", {"QQQ": 0.90, "VTI": 0.10}),
    ("qqq80_vti20", {"QQQ": 0.80, "VTI": 0.20}),
    ("qqq70_vti30", {"QQQ": 0.70, "VTI": 0.30}),
    ("qqq60_vti40", {"QQQ": 0.60, "VTI": 0.40}),
    ("qqq80_ief20", {"QQQ": 0.80, "IEF": 0.20}),
    ("qqq70_ief30", {"QQQ": 0.70, "IEF": 0.30}),
    ("vti", {"VTI": 1.0}),
)


@dataclass(frozen=True, slots=True)
class BlendComparison:
    """Reporting-only blend outcome on one regime window; never an adoption input."""

    name: str
    start: date
    end: date
    recipe: str
    qqq_baseline: AllocationResult
    vti_baseline: AllocationResult
    candidate: AllocationResult


def compare_qqq_blends(
    *,
    runner: Callable[[AllocationConfig], AllocationResult],
    contribution_krw: float,
    windows: tuple[tuple[str, date, date], ...] = QQQ_REGIME_WINDOWS,
) -> tuple[BlendComparison, ...]:
    """Run every blend recipe per window on identical external cashflows.

    ``qqq`` reuses the locked QQQ target path and ``vti`` the locked VTI baseline,
    both without an override; every other recipe pins explicit weights via
    ``targets_override``. Reporting-only diagnostics: no ablation, walk-forward
    gate, or adoption decision may run here.

    Raises:
        ValueError: On non-positive ``contribution_krw``, or when recipes of a
            window produce diverging snapshot counts.
    """
    if contribution_krw <= 0.0:
        raise ValueError(f"contribution_krw must be positive, got {contribution_krw!r}")
    comparisons: list[BlendComparison] = []
    for name, start, end in windows:
        results: dict[str, AllocationResult] = {}
        for recipe_id, weights in QQQ_BLEND_RECIPES:
            if recipe_id == "vti":
                config = AllocationConfig(
                    policy=PolicyId.VTI,
                    start=start,
                    end=end,
                    monthly_contribution_krw=float(contribution_krw),
                )
            elif recipe_id == "qqq":
                config = AllocationConfig(
                    policy=PolicyId.QQQ,
                    start=start,
                    end=end,
                    monthly_contribution_krw=float(contribution_krw),
                )
            else:
                config = AllocationConfig(
                    policy=PolicyId.QQQ,
                    start=start,
                    end=end,
                    monthly_contribution_krw=float(contribution_krw),
                    targets_override=dict(weights),
                )
            results[recipe_id] = runner(config)
        counts = {len(result.snapshots) for result in results.values()}
        if len(counts) != 1:
            detail = ", ".join(f"{recipe}={len(results[recipe].snapshots)}" for recipe in results)
            raise ValueError(f"window {name!r} snapshot counts diverge across recipes: {detail}")
        qqq_baseline = results["qqq"]
        vti_baseline = results["vti"]
        for recipe_id, _ in QQQ_BLEND_RECIPES:
            comparisons.append(
                BlendComparison(
                    name=name,
                    start=start,
                    end=end,
                    recipe=recipe_id,
                    qqq_baseline=qqq_baseline,
                    vti_baseline=vti_baseline,
                    candidate=results[recipe_id],
                )
            )
    return tuple(comparisons)
