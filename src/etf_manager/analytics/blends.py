"""Reporting-only S8 drawdown-blend recipe ratios versus the locked S8 and S1 paths."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final

from src.etf_manager.analytics.regimes import S8_REGIME_WINDOWS
from src.etf_manager.policy.targets import PolicyId
from src.etf_manager.sim.allocation import AllocationConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.etf_manager.sim.allocation import AllocationResult

__all__ = [
    "S8_BLEND_RECIPES",
    "BlendComparison",
    "compare_s8_blends",
]

S8_BLEND_RECIPES: Final[tuple[tuple[str, dict[str, float]], ...]] = (
    ("s8_qqq", {"QQQ": 1.0}),
    ("qqq90_vti10", {"QQQ": 0.90, "VTI": 0.10}),
    ("qqq80_vti20", {"QQQ": 0.80, "VTI": 0.20}),
    ("qqq70_vti30", {"QQQ": 0.70, "VTI": 0.30}),
    ("qqq60_vti40", {"QQQ": 0.60, "VTI": 0.40}),
    ("qqq80_ief20", {"QQQ": 0.80, "IEF": 0.20}),
    ("qqq70_ief30", {"QQQ": 0.70, "IEF": 0.30}),
    ("s1_vti", {"VTI": 1.0}),
)


@dataclass(frozen=True, slots=True)
class BlendComparison:
    """Reporting-only blend outcome on one regime window; never an adoption input."""

    name: str
    start: date
    end: date
    recipe: str
    s8_baseline: AllocationResult
    s1_baseline: AllocationResult
    candidate: AllocationResult


def compare_s8_blends(
    *,
    runner: Callable[[AllocationConfig], AllocationResult],
    contribution_krw: float,
    windows: tuple[tuple[str, date, date], ...] = S8_REGIME_WINDOWS,
) -> tuple[BlendComparison, ...]:
    """Run every blend recipe per window on identical external cashflows.

    ``s8_qqq`` reuses the locked S8_US_NASDAQ target path and ``s1_vti`` the locked
    S1_US baseline, both without an override; every other recipe pins explicit
    weights via ``targets_override``. Reporting-only diagnostics: no ablation,
    walk-forward gate, or adoption decision may run here.

    Raises:
        ValueError: On non-positive ``contribution_krw``, or when recipes of a
            window produce diverging snapshot counts.
    """
    if contribution_krw <= 0.0:
        raise ValueError(f"contribution_krw must be positive, got {contribution_krw!r}")
    comparisons: list[BlendComparison] = []
    for name, start, end in windows:
        results: dict[str, AllocationResult] = {}
        for recipe_id, weights in S8_BLEND_RECIPES:
            if recipe_id == "s1_vti":
                config = AllocationConfig(
                    policy=PolicyId.S1_US,
                    start=start,
                    end=end,
                    monthly_contribution_krw=float(contribution_krw),
                )
            elif recipe_id == "s8_qqq":
                config = AllocationConfig(
                    policy=PolicyId.S8_US_NASDAQ,
                    start=start,
                    end=end,
                    monthly_contribution_krw=float(contribution_krw),
                )
            else:
                config = AllocationConfig(
                    policy=PolicyId.S8_US_NASDAQ,
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
        s8_baseline = results["s8_qqq"]
        s1_baseline = results["s1_vti"]
        for recipe_id, _ in S8_BLEND_RECIPES:
            comparisons.append(
                BlendComparison(
                    name=name,
                    start=start,
                    end=end,
                    recipe=recipe_id,
                    s8_baseline=s8_baseline,
                    s1_baseline=s1_baseline,
                    candidate=results[recipe_id],
                )
            )
    return tuple(comparisons)
