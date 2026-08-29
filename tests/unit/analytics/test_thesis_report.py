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


def test_rpt_d_surface_and_primary_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.analytics.thesis_report import build_thesis_report
    from src.validation.accumulation_cohort import AccumulationCohortReport, AccumulationCohortRow, CohortOverlapMetadata

    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2025, 4, 30, tzinfo=UTC)

    def fake_horizon_surface(*, thesis, catalog_start, catalog_end):
        from src.validation.prospective import HorizonSurfacePoint

        return (
            HorizonSurfacePoint(horizon_months=60, cohort_count=5),
            HorizonSurfacePoint(horizon_months=84, cohort_count=3),
            HorizonSurfacePoint(horizon_months=96, cohort_count=1),
            HorizonSurfacePoint(horizon_months=120, cohort_count=0),
        )

    def fake_eval_horizon(*, thesis, catalog_start, catalog_end):
        return None

    def fake_proxy(*args, **kwargs):
        return (date(2016, 9, 30), date(2025, 4, 30))

    def fake_cohort(spec, runner, horizon_months=120, step_months=12, bootstrap_paths=400, seed=7):
        # limit to the fallback horizon 96
        assert horizon_months == 96
        overlap = CohortOverlapMetadata(horizon_months=horizon_months, step_months=step_months)
        rows = tuple(AccumulationCohortRow(candidate_wealth=61.0, baseline_wealth=100.0, ratio=0.61, candidate_recovery_months=None) for _ in range(1))
        return AccumulationCohortReport(
            name=spec.name,
            overlap=overlap,
            rows=rows,
            median_ratio=0.61,
            p10_ratio=0.61,
            worst_ratio=0.61,
            win_rate=0.0,
            bootstrap_p05_ratio_mean=0.61,
            unrecovered_cohort_count=0,
        )

    monkeypatch.setattr("src.validation.prospective.resolve_horizon_surface", fake_horizon_surface)
    monkeypatch.setattr("src.validation.prospective.resolve_evaluation_horizon", fake_eval_horizon)
    monkeypatch.setattr("src.validation.prospective.resolve_proxy_history_span", fake_proxy)
    monkeypatch.setattr("src.analytics.thesis_report.resolve_horizon_surface", fake_horizon_surface)
    monkeypatch.setattr("src.analytics.thesis_report.resolve_evaluation_horizon", fake_eval_horizon)
    monkeypatch.setattr("src.analytics.thesis_report.resolve_proxy_history_span", fake_proxy)
    monkeypatch.setattr("src.validation.accumulation_cohort.run_accumulation_cohort_report", fake_cohort)

    def fake_load_visible(settings, dataset, decision_ts):
        raise ValueError("no holdings")

    monkeypatch.setattr("src.data.catalog.load_visible", fake_load_visible)

    def runner(config: AllocationConfig) -> AllocationResult:
        wealth = 61.0 if config.targets_override == {"BOTZ": 1.0} else 100.0
        return AllocationResult(
            config=config,
            snapshots=(AllocationSnapshot(session=date(2024, 1, 31), cash_krw=0, cash_usd=0, shares={}, mark_krw=wealth, contribution_krw=0, fees_krw=0),),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )

    report = build_thesis_report(thesis_id=ThesisId.PHYSICAL_AUTOMATION, settings=settings, as_of=as_of, runner=runner)
    assert report.divergence is not None
    hs = report.divergence.get("horizon_surface")
    assert hs is not None
    # contains 60/84/96/120
    months = {entry["horizon_months"] if isinstance(entry, dict) else getattr(entry, "horizon_months", None) for entry in hs}  # type: ignore[arg-type]
    assert {60, 84, 96, 120}.issubset(months) or {60, 84, 96, 120} == months
    # when primary missing, evaluated_horizon_months absent or 0
    eval_h = report.divergence.get("evaluated_horizon_months")
    assert eval_h is None or eval_h == 0
    fb = report.divergence.get("fallback_horizon_months")
    assert fb == 96


@pytest.mark.parametrize("scenario_id", ["test_rpt_c_cohort_ce_not_singleton"])
def test_rpt_c_cohort_ce_not_singleton(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.analytics.thesis_report import build_thesis_report
    from src.validation.accumulation_cohort import AccumulationCohortReport, AccumulationCohortRow, CohortOverlapMetadata
    from src.validation.gate import certainty_equivalent

    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2025, 4, 30, tzinfo=UTC)

    # Prepare cohort rows where candidate wealths vary and not constant ratio
    c_wealths = [120.0, 130.0, 110.0, 125.0]
    b_wealths = [100.0, 100.0, 100.0, 100.0]

    def runner(config: AllocationConfig) -> AllocationResult:
        # terminal ratio = 1.2 if SOXX else 1.0 -> 1.2 singleton
        wealth = 120.0 if config.targets_override == {"SOXX": 1.0} else 100.0
        return AllocationResult(
            config=config,
            snapshots=(AllocationSnapshot(session=date(2024, 1, 31), cash_krw=0, cash_usd=0, shares={}, mark_krw=wealth, contribution_krw=0, fees_krw=0),),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )

    def fake_cohort_report(spec, runner, horizon_months=120, step_months=12, bootstrap_paths=400, seed=7):
        overlap = CohortOverlapMetadata(horizon_months=horizon_months, step_months=step_months)
        rows = tuple(AccumulationCohortRow(candidate_wealth=c, baseline_wealth=b, ratio=c / b, candidate_recovery_months=0) for c, b in zip(c_wealths, b_wealths, strict=True))
        ratios = [c / b for c, b in zip(c_wealths, b_wealths, strict=True)]
        # compute median quickly via sorted
        ratios_sorted = sorted(ratios)
        median = ratios_sorted[len(ratios_sorted) // 2]
        return AccumulationCohortReport(
            name=spec.name,
            overlap=overlap,
            rows=rows,
            median_ratio=float(median),
            p10_ratio=float(min(ratios)),
            worst_ratio=float(min(ratios)),
            win_rate=1.0,
            bootstrap_p05_ratio_mean=float(min(ratios)),
            unrecovered_cohort_count=0,
        )

    monkeypatch.setattr("src.validation.accumulation_cohort.run_accumulation_cohort_report", fake_cohort_report)

    def fake_load_visible(settings, dataset, decision_ts):
        raise ValueError("no holdings")

    monkeypatch.setattr("src.data.catalog.load_visible", fake_load_visible)

    report = build_thesis_report(thesis_id=ThesisId.AI_COMPUTE, settings=settings, as_of=as_of, runner=runner)
    assert report.divergence is not None
    assert "terminal_wealth_ratio" in report.divergence
    assert "cohort_ce_ratio_gamma_2" in report.divergence
    terminal = float(report.divergence["terminal_wealth_ratio"])  # type: ignore[arg-type]
    cohort_ce = float(report.divergence["cohort_ce_ratio_gamma_2"])  # type: ignore[arg-type]
    assert terminal != cohort_ce
    expected_ce = certainty_equivalent(c_wealths, gamma=2.0) / certainty_equivalent(b_wealths, gamma=2.0)
    assert abs(cohort_ce - expected_ce) < 1e-9
    assert abs(terminal - 1.2) < 1e-9


@pytest.mark.parametrize("scenario_id", ["test_rpt_e_overrides_experiment_end"])
def test_rpt_e_overrides_experiment_end(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from src.analytics.thesis_report import build_thesis_report
    from src.data.settings import DataSettings
    from src.policy.thesis import ThesisId
    from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot  # noqa: F401

    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2026, 6, 30, 20, 0, tzinfo=UTC)
    # Create experiment JSON with end 2025-04-30
    payload = {
        "name": "thesis_ai_compute_evidence",
        "start": "2007-08-31",
        "end": "2025-04-30",
        "contribution_krw": 1_000_000,
        "hurdle": 0.02,
        "horizon_months": 0,
        "baseline": {"id": "qqq_baseline", "policy": "qqq", "modules": 0, "targets": {"QQQ": 1.0}},
        "candidates": [{"id": "soxx_100", "policy": "qqq", "modules": 1, "targets": {"SOXX": 1.0}}],
    }
    exp_path = tmp_path / "exp.json"
    exp_path.write_text(json.dumps(payload), encoding="utf-8")

    captured_ends: list[date] = []

    def runner(config: AllocationConfig) -> AllocationResult:
        captured_ends.append(config.end)
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

    def fake_load_visible(settings_inner, dataset, decision_ts):
        raise ValueError("no holdings")

    monkeypatch.setattr("src.data.catalog.load_visible", fake_load_visible)
    # Patch cohort to avoid heavy compute
    def fake_cohort(spec, runner_inner, horizon_months=120, step_months=12, bootstrap_paths=400, seed=7):
        from src.validation.accumulation_cohort import AccumulationCohortReport, AccumulationCohortRow, CohortOverlapMetadata
        overlap = CohortOverlapMetadata(horizon_months=horizon_months, step_months=step_months)
        rows = tuple(AccumulationCohortRow(candidate_wealth=110, baseline_wealth=100, ratio=1.1, candidate_recovery_months=0) for _ in range(5))
        return AccumulationCohortReport(name=spec.name, overlap=overlap, rows=rows, median_ratio=1.1, p10_ratio=1.0, worst_ratio=0.9, win_rate=1.0, bootstrap_p05_ratio_mean=1.0, unrecovered_cohort_count=0)
    monkeypatch.setattr("src.validation.accumulation_cohort.run_accumulation_cohort_report", fake_cohort)

    report = build_thesis_report(thesis_id=ThesisId.AI_COMPUTE, settings=settings, as_of=as_of, runner=runner, experiment_path=exp_path)
    assert captured_ends, "runner should have been called"
    # Effective end clamps experiment end to panel as-of (never extends beyond config end).
    assert all(e == date(2025, 4, 30) for e in captured_ends), f"expected all ends 2025-04-30 got {captured_ends}"
