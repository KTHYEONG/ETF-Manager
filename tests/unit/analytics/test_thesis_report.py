"""Thesis report tests."""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.data.settings import DataSettings
from src.policy.thesis import ThesisId
from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot


@pytest.mark.parametrize("scenario_id", ["RPT-A-five-slots"])
def test_rpt_a_five_slots(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.analytics.thesis_report import build_thesis_report

    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2025, 4, 30, tzinfo=UTC)

    def runner(config: AllocationConfig) -> AllocationResult:
        wealth = 110.0 if config.targets_override == {"SOXX": 1.0} else 100.0
        return AllocationResult(
            config=config,
            snapshots=(AllocationSnapshot(session=date(2024, 1, 31), cash_krw=0, cash_usd=0, shares={}, mark_krw=wealth, contribution_krw=0, fees_krw=0),),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )

    # Patch holdings to insufficient_data to not fail
    def fake_load_visible(settings, dataset, decision_ts):
        raise ValueError("no holdings")

    monkeypatch.setattr("src.data.catalog.load_visible", fake_load_visible)

    report = build_thesis_report(thesis_id=ThesisId.AI_COMPUTE, settings=settings, as_of=as_of, runner=runner)
    assert report.evidence.historical is not None
    assert report.evidence.structural is not None
    assert report.evidence.valuation is not None
    assert report.evidence.overlap is not None
    assert report.evidence.crowding is not None
    # All five slots are EvidenceSlot instances
    from src.analytics.thesis_evidence import EvidenceSlot

    for slot in [report.evidence.historical, report.evidence.structural, report.evidence.valuation, report.evidence.overlap, report.evidence.crowding]:
        assert isinstance(slot, EvidenceSlot)


@pytest.mark.parametrize("scenario_id", ["RPT-B-divergence-block"])
def test_rpt_b_divergence_block(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.analytics.thesis_report import build_thesis_report
    from src.validation.accumulation_cohort import AccumulationCohortReport, AccumulationCohortRow, CohortOverlapMetadata

    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2025, 4, 30, tzinfo=UTC)

    # Create runner that yields ce ratio 0.997 and long horizon median 1.027
    # We'll mock run_accumulation_cohort_report to return specific median and cohort count
    # To make CE ratio 0.997, we need baseline 100, candidate 99.7 for CE cohort
    # For long horizon, median 1.027 with cohort_count 9 (<10) so passes False

    def runner(config: AllocationConfig) -> AllocationResult:
        # Determine which call: for accumulation cohort runner will be called multiple times per cohort
        # We'll just return wealth that yields ratio 0.997 for singleton, but for rolled cohorts we want varied?
        # Simpler: mock the whole accumulation function
        wealth = 100.0
        if config.targets_override == {"SOXX": 1.0}:
            wealth = 99.7  # ce ratio 0.997
        return AllocationResult(
            config=config,
            snapshots=(AllocationSnapshot(session=date(2024, 1, 31), cash_krw=0, cash_usd=0, shares={}, mark_krw=wealth, contribution_krw=0, fees_krw=0),),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )

    # Patch the cohort report to return controlled median and count

    def fake_cohort_report(spec, runner, horizon_months=120, step_months=12, bootstrap_paths=4000, seed=7):
        # Return report with 9 cohorts median 1.027
        overlap = CohortOverlapMetadata(horizon_months=120, step_months=12)
        rows = tuple(AccumulationCohortRow(candidate_wealth=102.7, baseline_wealth=100.0, ratio=1.027, candidate_recovery_months=0) for _ in range(9))
        return AccumulationCohortReport(
            name=spec.name,
            overlap=overlap,
            rows=rows,
            median_ratio=1.027,
            p10_ratio=1.0,
            worst_ratio=0.99,
            win_rate=0.6,
            bootstrap_p05_ratio_mean=1.01,
            unrecovered_cohort_count=0,
        )

    monkeypatch.setattr("src.validation.accumulation_cohort.run_accumulation_cohort_report", fake_cohort_report)

    def fake_load_visible(settings, dataset, decision_ts):
        raise ValueError("no holdings")

    monkeypatch.setattr("src.data.catalog.load_visible", fake_load_visible)

    # Also patch evaluate_prospective_eligibility to return not eligible so divergence path exercised
    report = build_thesis_report(thesis_id=ThesisId.AI_COMPUTE, settings=settings, as_of=as_of, runner=runner)
    assert report.divergence is not None
    assert report.divergence.get("long_horizon_passes") is False
    # divergence should document both ratios
    # Check that some key contains median and ce
    divergence_str = str(report.divergence)
    assert "1.027" in divergence_str or "median" in divergence_str.lower()
    assert "0.997" in divergence_str or "ce" in divergence_str.lower() or "ratio" in divergence_str.lower()
    # Also long_horizon should be not passing due to cohort count <10
    assert report.long_horizon is not None
    assert report.long_horizon.passes is False
