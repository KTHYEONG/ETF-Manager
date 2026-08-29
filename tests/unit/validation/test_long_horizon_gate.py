"""Long horizon gate tests."""
from __future__ import annotations

import pytest

from src.validation.accumulation_cohort import AccumulationCohortReport, AccumulationCohortRow, CohortOverlapMetadata
from src.validation.gate import long_horizon_passes


def _make_report(cohort_count: int, median_ratio: float, step_months: int = 12, horizon_months: int = 120) -> AccumulationCohortReport:
    overlap = CohortOverlapMetadata(horizon_months=horizon_months, step_months=step_months)
    rows = tuple(
        AccumulationCohortRow(candidate_wealth=110.0, baseline_wealth=100.0, ratio=1.1, candidate_recovery_months=0)
        for _ in range(cohort_count)
    )
    # Use median_ratio param
    return AccumulationCohortReport(
        name="test",
        overlap=overlap,
        rows=rows,
        median_ratio=median_ratio,
        p10_ratio=median_ratio - 0.05,
        worst_ratio=median_ratio - 0.1,
        win_rate=1.0,
        bootstrap_p05_ratio_mean=median_ratio - 0.02,
        unrecovered_cohort_count=0,
    )


@pytest.mark.parametrize("scenario_id", ["LH-A-fail-below-min-cohorts"])
def test_lh_a_fail_below_min_cohorts(scenario_id: str) -> None:
    report = _make_report(cohort_count=9, median_ratio=1.05)
    verdict = long_horizon_passes(report, min_cohorts=10, min_median_ratio=1.0)
    assert verdict.passes is False
    assert "9" in verdict.reason
    assert verdict.cohort_count == 9


@pytest.mark.parametrize("scenario_id", ["LH-B-pass-at-threshold"])
def test_lh_b_pass_at_threshold(scenario_id: str) -> None:
    report = _make_report(cohort_count=10, median_ratio=1.03, step_months=12, horizon_months=120)
    verdict = long_horizon_passes(report, min_cohorts=10, min_median_ratio=1.0)
    assert verdict.passes is True
    assert verdict.cohort_count == 10
    assert verdict.median_ratio == pytest.approx(1.03)
    # overlap dependence disclosed when step < horizon
    assert verdict.overlap_dependence_disclosed is True
    # also test non-overlapping case
    report2 = _make_report(cohort_count=10, median_ratio=1.03, step_months=120, horizon_months=120)
    verdict2 = long_horizon_passes(report2, min_cohorts=10, min_median_ratio=1.0)
    assert verdict2.overlap_dependence_disclosed is False
