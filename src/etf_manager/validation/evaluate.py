"""Cohort evaluation over an injected simulation runner."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import date

from src.etf_manager.sim.allocation import AllocationConfig, AllocationResult

__all__ = ["evaluate_cohort_wealths"]


def evaluate_cohort_wealths(
    template: AllocationConfig,
    cohorts: Sequence[tuple[date, date]],
    runner: Callable[[AllocationConfig], AllocationResult],
) -> tuple[float, ...]:
    """Real terminal wealth per cohort; ``template`` is never mutated.

    Each cohort re-runs the simulation on a copy of ``template`` with only
    start/end swapped, preserving policy, tilt, overlay, currency, mapping,
    and cost parameters exactly.

    Raises:
        ValueError: When ``cohorts`` is empty.
    """
    if len(cohorts) < 1:
        raise ValueError("cohorts must contain at least one (start, end) pair")
    return tuple(
        runner(replace(template, start=c_start, end=c_end)).terminal_wealth_real_krw
        for c_start, c_end in cohorts
    )
