"""Unit tests for cohort evaluation helpers."""

from __future__ import annotations

from datetime import date

import pytest

from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig, AllocationResult
from src.validation.evaluate import evaluate_cohort_results, evaluate_cohort_wealths


def _result(config: AllocationConfig, wealth: float = 100.0) -> AllocationResult:
    return AllocationResult(
        config=config,
        snapshots=(),
        terminal_wealth_krw=wealth,
        xirr=0.0,
        max_drawdown=0.0,
        terminal_wealth_real_krw=wealth,
        xirr_real=0.0,
    )


def _template() -> AllocationConfig:
    return AllocationConfig(
        policy=PolicyId.VT,
        start=date(2020, 1, 1),
        end=date(2020, 12, 31),
        monthly_contribution_krw=1_000_000.0,
    )


@pytest.mark.parametrize("scenario_id", ["VAL-EVA-cohort-results"])
def test_val_eva_cohort_results(scenario_id: str) -> None:
    """VAL-EVA-cohort-results"""
    template = _template()
    cohorts = ((date(2020, 1, 1), date(2020, 6, 30)), (date(2020, 7, 1), date(2020, 12, 31)))
    seen: list[tuple[date, date]] = []

    def runner(config: AllocationConfig) -> AllocationResult:
        seen.append((config.start, config.end))
        return _result(config, wealth=10.0 * len(seen))

    results = evaluate_cohort_results(template, cohorts, runner)

    assert len(results) == 2
    assert results[0].terminal_wealth_real_krw == pytest.approx(10.0)
    assert results[1].terminal_wealth_real_krw == pytest.approx(20.0)
    assert seen == list(cohorts)
    assert template.start == date(2020, 1, 1)
    assert template.end == date(2020, 12, 31)

    wealth_runner_calls = 0

    def wealth_runner(config: AllocationConfig) -> AllocationResult:
        nonlocal wealth_runner_calls
        wealth_runner_calls += 1
        return _result(config, wealth=10.0 * wealth_runner_calls)

    wealths = evaluate_cohort_wealths(template, cohorts, wealth_runner)
    assert wealths == (10.0, 20.0)

    with pytest.raises(ValueError, match="cohorts"):
        evaluate_cohort_results(template, (), runner)
