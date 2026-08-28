"""Unit tests for rolling accumulation cohort reporting."""

from __future__ import annotations

import math
from datetime import date

import pytest

from src.analytics.metrics import recovery_months
from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot
from src.validation.accumulation_cohort import (
    AccumulationCohortReport,
    CohortOverlapMetadata,
    run_accumulation_cohort_report,
    summarize_accumulation_cohorts,
)
from src.validation.experiment import CandidateSpec, ExperimentSpec


def _allocation_result(
    config: AllocationConfig,
    *,
    wealth: float,
    sessions: tuple[date, ...] = (),
    marks: tuple[float, ...] = (),
) -> AllocationResult:
    snapshots = tuple(
        AllocationSnapshot(
            session=session,
            cash_krw=0.0,
            cash_usd=0.0,
            shares={},
            mark_krw=mark,
            contribution_krw=0.0,
            fees_krw=0.0,
        )
        for session, mark in zip(sessions, marks, strict=True)
    )
    return AllocationResult(
        config=config,
        snapshots=snapshots,
        terminal_wealth_krw=wealth,
        xirr=0.0,
        max_drawdown=0.0,
        terminal_wealth_real_krw=wealth,
        xirr_real=0.0,
    )


@pytest.mark.parametrize("scenario_id", ["ACC-COH-recovery-months"])
def test_acc_coh_recovery_months(scenario_id: str) -> None:
    """ACC-COH-recovery-months"""
    v_sessions = (date(2020, 1, 15), date(2020, 2, 15), date(2020, 3, 15), date(2020, 4, 15))
    assert recovery_months(v_sessions, (100.0, 80.0, 90.0, 100.0)) == 2

    flat_sessions = (date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1))
    assert recovery_months(flat_sessions, (100.0, 100.0, 100.0)) == 0

    decline_sessions = (date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1))
    assert recovery_months(decline_sessions, (100.0, 90.0, 80.0)) is None


@pytest.mark.parametrize("scenario_id", ["ACC-COH-overlap-metadata"])
def test_acc_coh_overlap_metadata(scenario_id: str) -> None:
    """ACC-COH-overlap-metadata"""
    overlapping = CohortOverlapMetadata(horizon_months=120, step_months=1)
    assert overlapping.overlap_months == 119
    assert overlapping.independent_sample_warning is True

    non_overlapping = CohortOverlapMetadata(horizon_months=120, step_months=120)
    assert non_overlapping.overlap_months == 0
    assert non_overlapping.independent_sample_warning is False


@pytest.mark.parametrize("scenario_id", ["ACC-COH-summarize-stats"])
def test_acc_coh_summarize_stats(scenario_id: str) -> None:
    """ACC-COH-summarize-stats"""
    overlap = CohortOverlapMetadata(horizon_months=12, step_months=12)
    report = summarize_accumulation_cohorts(
        candidate_wealths=(110.0, 115.0, 120.0),
        baseline_wealths=(100.0, 100.0, 100.0),
        candidate_recovery_months=(0, 0, 0),
        overlap=overlap,
        bootstrap_paths=2000,
        seed=7,
    )

    assert isinstance(report, AccumulationCohortReport)
    assert report.median_ratio == pytest.approx(1.15)
    assert report.worst_ratio == pytest.approx(1.10)
    assert report.win_rate == pytest.approx(1.0)
    assert report.bootstrap_p05_ratio_mean > 1.0


@pytest.mark.parametrize("scenario_id", ["ACC-COH-run-report"])
def test_acc_coh_run_report(scenario_id: str) -> None:
    """ACC-COH-run-report"""
    spec = ExperimentSpec(
        name="acc_coh_smoke",
        start=date(2020, 1, 1),
        end=date(2021, 12, 31),
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        baseline=CandidateSpec(id="qqq", policy="qqq", modules=0),
        candidates=[CandidateSpec(id="vti", policy="s1_us", modules=1)],
    )

    def runner(config: AllocationConfig) -> AllocationResult:
        wealth = 120.0 if config.policy is PolicyId.VTI else 100.0
        return _allocation_result(config, wealth=wealth)

    report = run_accumulation_cohort_report(
        spec,
        runner,
        horizon_months=12,
        step_months=12,
        bootstrap_paths=100,
        seed=7,
    )

    assert len(report.rows) == 2
    assert all(row.ratio == pytest.approx(1.2) for row in report.rows)
    assert math.isfinite(report.median_ratio)
    assert math.isfinite(report.bootstrap_p05_ratio_mean)
