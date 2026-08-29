"""Prospective eligibility and paper forward tests."""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.policy.thesis import ThesisId, ThesisSpec, Horizon
from src.data.settings import DataSettings
from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot
from src.validation.experiment import CandidateSpec, ExperimentSpec
from src.validation.registry import freeze_baseline_config_hash
from src.validation.prospective import evaluate_prospective_eligibility, run_prospective_paper_forward
from src.policy.targets import PolicyId


def _thesis_with_min_years(min_years: int) -> ThesisSpec:
    return ThesisSpec(
        id=ThesisId.PHYSICAL_AUTOMATION,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=min_years, target_years=min_years + 5),
        causal_chain=["a"],
        falsifiers=["f1"],
        candidate_sleeves=["physical_automation"],
        historical_proxies=["BOTZ"],
    )


@pytest.mark.parametrize("scenario_id", ["PROSP-A-botz-eligible"])
def test_prosp_a_botz_eligible(scenario_id: str) -> None:
    # catalog span approx 8.7 years: from 2016-09-30 to 2025-04-30
    catalog_start = date(2016, 9, 30)
    catalog_end = date(2025, 4, 30)
    span_days = (catalog_end - catalog_start).days
    span_years = span_days / 365.25
    assert 8.0 < span_years < 9.0

    thesis5 = ThesisSpec(
        id=ThesisId.PHYSICAL_AUTOMATION,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=5, target_years=5),
        causal_chain=["a"],
        falsifiers=["f1"],
        candidate_sleeves=["physical_automation"],
        historical_proxies=["BOTZ"],
    )
    el5 = evaluate_prospective_eligibility(thesis=thesis5, catalog_start=catalog_start, catalog_end=catalog_end)
    assert el5.eligible is False
    assert el5.min_years_required == 5
    assert "min_years" in el5.reason

    thesis10 = ThesisSpec(
        id=ThesisId.PHYSICAL_AUTOMATION,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=10, target_years=10),
        causal_chain=["a"],
        falsifiers=["f1"],
        candidate_sleeves=["physical_automation"],
        historical_proxies=["BOTZ"],
    )
    el10 = evaluate_prospective_eligibility(thesis=thesis10, catalog_start=catalog_start, catalog_end=catalog_end)
    assert el10.eligible is True
    assert el10.min_years_required == 10


def test_prosp_c_min_years_gate_not_target() -> None:
    catalog_start = date(2016, 9, 30)
    catalog_end = date(2025, 4, 30)
    span_years = (catalog_end - catalog_start).days / 365.25
    assert 8.0 < span_years < 9.0
    # Horizon(min=5,target=10): span >=5 so not eligible
    thesis5_10 = ThesisSpec(
        id=ThesisId.PHYSICAL_AUTOMATION,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["f1"],
        candidate_sleeves=["physical_automation"],
        historical_proxies=["BOTZ"],
    )
    el = evaluate_prospective_eligibility(thesis=thesis5_10, catalog_start=catalog_start, catalog_end=catalog_end)
    assert el.eligible is False
    assert el.min_years_required == 5
    assert "min_years" in el.reason
    # Horizon(min=10,target=10): span <10 so eligible
    thesis10_10 = ThesisSpec(
        id=ThesisId.PHYSICAL_AUTOMATION,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=10, target_years=10),
        causal_chain=["a"],
        falsifiers=["f1"],
        candidate_sleeves=["physical_automation"],
        historical_proxies=["BOTZ"],
    )
    el2 = evaluate_prospective_eligibility(thesis=thesis10_10, catalog_start=catalog_start, catalog_end=catalog_end)
    assert el2.eligible is True
    assert el2.min_years_required == 10


def test_prosp_d_adaptive_horizon_botz_panel() -> None:
    from datetime import timedelta

    from src.validation.prospective import resolve_evaluation_horizon

    thesis = ThesisSpec(
        id=ThesisId.PHYSICAL_AUTOMATION,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["f1"],
        candidate_sleeves=["physical_automation"],
        historical_proxies=["BOTZ"],
    )
    start = date(2016, 9, 30)
    end = date(2025, 4, 30)
    horizon = resolve_evaluation_horizon(thesis=thesis, catalog_start=start, catalog_end=end)
    assert horizon is None
    # short span <5y -> None
    short_end = start + timedelta(days=int(4.5 * 365.25))
    short_h = resolve_evaluation_horizon(thesis=thesis, catalog_start=start, catalog_end=short_end)
    assert short_h is None


def test_prosp_e_horizon_surface_preregistered() -> None:
    from src.validation.prospective import resolve_horizon_surface

    thesis = ThesisSpec(
        id=ThesisId.PHYSICAL_AUTOMATION,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["f1"],
        candidate_sleeves=["physical_automation"],
        historical_proxies=["BOTZ"],
    )
    start = date(2016, 9, 30)
    end = date(2025, 4, 30)
    surface = resolve_horizon_surface(thesis=thesis, catalog_start=start, catalog_end=end)
    assert len(surface) == 4
    by_month = {p.horizon_months: p.cohort_count for p in surface}
    assert set(by_month.keys()) == {60, 84, 96, 120}
    assert by_month[120] == 0
    assert by_month[96] >= 1
    assert by_month[60] >= by_month[96]


def test_prosp_f_primary_target_when_feasible() -> None:
    from src.validation.prospective import resolve_evaluation_horizon

    thesis = ThesisSpec(
        id=ThesisId.PHYSICAL_AUTOMATION,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["f1"],
        candidate_sleeves=["physical_automation"],
        historical_proxies=["BOTZ"],
    )
    start = date(2007, 8, 31)
    end = date(2025, 4, 30)
    horizon = resolve_evaluation_horizon(thesis=thesis, catalog_start=start, catalog_end=end)
    assert horizon is not None
    assert horizon.horizon_months == 120
    assert horizon.span_capped is False


@pytest.mark.parametrize("scenario_id", ["PROSP-B-paper-reconcile"])
def test_prosp_b_paper_reconcile(scenario_id: str, tmp_path: Path) -> None:
    spec = ExperimentSpec(
        name="prospective_test",
        start=date(2016, 9, 30),
        end=date(2025, 4, 30),
        contribution_krw=1_000_000,
        hurdle=0.02,
        horizon_months=0,
        baseline=CandidateSpec(id="qqq_baseline", policy=PolicyId.QQQ, modules=0, targets={"QQQ": 1.0}),
        candidates=[CandidateSpec(id="botz_100", policy=PolicyId.QQQ, modules=1, targets={"BOTZ": 1.0})],
    )
    frozen_at = datetime(2025, 1, 1, tzinfo=UTC)
    targets_hash = freeze_baseline_config_hash(spec)
    from src.validation.prospective import ProspectiveFreezeRecord

    freeze = ProspectiveFreezeRecord(
        thesis_id="physical_automation",
        frozen_at=frozen_at,
        targets_hash=targets_hash,
        experiment_name=spec.name,
    )
    settings = DataSettings(data_root=tmp_path / "data")

    def runner(config: AllocationConfig) -> AllocationResult:
        # Verify allocation window [frozen_at.date(), spec.end]
        assert config.start == frozen_at.date()
        assert config.end == spec.end
        # Create snapshots with integer lots
        snap1 = AllocationSnapshot(session=date(2025, 1, 31), cash_krw=0, cash_usd=0, shares={"BOTZ": 10}, mark_krw=1000000, contribution_krw=1000000, fees_krw=0)
        snap2 = AllocationSnapshot(session=date(2025, 2, 28), cash_krw=0, cash_usd=0, shares={"BOTZ": 20}, mark_krw=2000000, contribution_krw=1000000, fees_krw=0)
        return AllocationResult(
            config=config,
            snapshots=(snap1, snap2),
            terminal_wealth_krw=2000000,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=2000000,
            xirr_real=0.0,
        )

    broker = run_prospective_paper_forward(spec=spec, freeze=freeze, settings=settings, runner=runner)
    # Check integer lots match final snapshot shares
    assert broker.position("BOTZ") == 20
    # also test hash mismatch fails
    bad_freeze = ProspectiveFreezeRecord(
        thesis_id="physical_automation",
        frozen_at=frozen_at,
        targets_hash="badhash",
        experiment_name=spec.name,
    )
    with pytest.raises(ValueError, match="targets_hash"):
        run_prospective_paper_forward(spec=spec, freeze=bad_freeze, settings=settings, runner=runner)
