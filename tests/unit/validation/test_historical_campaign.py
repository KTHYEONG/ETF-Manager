# ruff: noqa: PT011,S101,RUF043,C408,B905
"""Historical campaign tests - generated from contract."""
from __future__ import annotations

def test_final_historical_arms_registry() -> None:
    import pytest

    from src.validation.historical_campaign import FINAL_HISTORICAL_ARMS, FinalHistoricalArmId

    ids = {arm.arm_id for arm in FINAL_HISTORICAL_ARMS}
    assert ids == {
        FinalHistoricalArmId.B0_QQQ100,
        FinalHistoricalArmId.C1_QQQ95_SOXX5,
        FinalHistoricalArmId.C2_QQQ90_SOXX10,
        FinalHistoricalArmId.C3_QQQ85_SOXX15,
    }
    for arm in FINAL_HISTORICAL_ARMS:
        assert arm.targets is not None
        assert arm.adaptive is False
        assert sum(arm.targets.values()) == pytest.approx(1.0)
    b0 = next(a for a in FINAL_HISTORICAL_ARMS if a.arm_id == FinalHistoricalArmId.B0_QQQ100)
    assert b0.targets == {"QQQ": 1.0}
    c2 = next(a for a in FINAL_HISTORICAL_ARMS if a.arm_id == FinalHistoricalArmId.C2_QQQ90_SOXX10)
    assert c2.targets == {"QQQ": 0.9, "SOXX": 0.1}


def test_assert_final_campaign_spec_rejects_adaptive() -> None:
    from datetime import date

    import pytest

    from src.validation.experiment import AdaptiveContributionSpec, CandidateSpec, ExperimentSpec
    from src.validation.historical_campaign import assert_final_campaign_spec
    from src.validation.research_posture import ObjectiveFamily

    good = ExperimentSpec(
        name="final_historical_campaign_v1",
        start=date(2016, 7, 1),
        end=date(2026, 6, 30),
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=120,
        objective="long_horizon",
        objective_family=ObjectiveFamily.CAPITAL_ALLOCATION,
        baseline=CandidateSpec(id="b0_qqq100", policy="qqq", modules=0, targets={"QQQ": 1.0}),
        candidates=[
            CandidateSpec(id="c1", policy="qqq", modules=1, targets={"QQQ": 0.95, "SOXX": 0.05}),
            CandidateSpec(id="c2", policy="qqq", modules=1, targets={"QQQ": 0.9, "SOXX": 0.1}),
            CandidateSpec(id="c3", policy="qqq", modules=1, targets={"QQQ": 0.85, "SOXX": 0.15}),
        ],
    )
    assert_final_campaign_spec(good)
    bad = good.model_copy(update={"adaptive_contribution": AdaptiveContributionSpec()})
    with pytest.raises(ValueError, match="adaptive_contribution|capital_allocation"):
        assert_final_campaign_spec(bad)


def test_assert_final_campaign_spec_requires_objective_family() -> None:
    from datetime import date

    import pytest

    from src.validation.experiment import CandidateSpec, ExperimentSpec, KafiDeploymentSpec
    from src.validation.historical_campaign import assert_final_campaign_spec
    from src.validation.research_posture import ObjectiveFamily

    base_kwargs = dict(
        name="final_historical_campaign_v1",
        start=date(2016, 7, 1),
        end=date(2026, 6, 30),
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=120,
        objective="long_horizon",
        baseline=CandidateSpec(id="b0", policy="qqq", modules=0, targets={"QQQ": 1.0}),
        candidates=[
            CandidateSpec(id="c2", policy="qqq", modules=1, targets={"QQQ": 0.9, "SOXX": 0.1}),
        ],
    )
    missing = ExperimentSpec(**base_kwargs)
    with pytest.raises(ValueError, match="objective_family"):
        assert_final_campaign_spec(missing)
    wrong_family = ExperimentSpec(
        **base_kwargs,
        objective_family=ObjectiveFamily.DEPLOYMENT_TIMING,
        kafi_deployment=KafiDeploymentSpec(),
    )
    with pytest.raises(ValueError, match="capital_allocation|objective_family"):
        assert_final_campaign_spec(wrong_family)


def test_audit_regime_coverage_flags_gaps() -> None:
    from datetime import date

    from src.validation.historical_campaign import audit_regime_coverage

    cohorts = ((date(2016, 7, 1), date(2026, 6, 30)),)
    report = audit_regime_coverage(cohorts=cohorts)
    by_name = {row.regime_name: row for row in report.rows}
    assert "dot_com" in by_name
    assert by_name["dot_com"].covered is False
    assert by_name["ai_boom_2023"].covered is True
    assert by_name["ai_boom_2023"].overlap_months >= 1
    assert report.independent_sample_warning is True


def test_build_trial_lineage_census_from_index() -> None:
    from pathlib import Path

    from src.validation.historical_campaign import build_trial_lineage_census

    census = build_trial_lineage_census(
        index_path=Path("configs/experiments/INDEX.json"),
        experiments_dir=Path("configs/experiments"),
    )
    families = {row.family_id for row in census.families}
    assert "soxx" in families
    assert "adaptive" in families
    assert census.total_experiments >= 10
    soxx = next(r for r in census.families if r.family_id == "soxx")
    assert soxx.experiment_count >= 1
    assert soxx.active_count + soxx.archived_count == soxx.experiment_count


def test_run_final_historical_campaign_synthetic() -> None:
    import pytest

    from datetime import date

    from src.policy.targets import PolicyId
    from src.sim.allocation import AllocationConfig, AllocationResult
    from src.validation.experiment import CandidateSpec, ExperimentSpec
    from src.validation.historical_campaign import run_final_historical_campaign
    from src.validation.research_posture import ObjectiveFamily

    class _Runner:
        def __call__(self, config: AllocationConfig) -> AllocationResult:
            bonus = 0.0
            if config.targets_override and config.targets_override.get("SOXX", 0.0) > 0.0:
                bonus = float(config.targets_override["SOXX"]) * 10.0
            wealth = 100.0 + bonus
            return AllocationResult(
                config=config,
                snapshots=(),
                terminal_wealth_krw=wealth,
                xirr=0.0,
                max_drawdown=-0.1,
                terminal_wealth_real_krw=wealth,
                xirr_real=0.05,
                total_contribution_real_krw=90.0,
            )

    spec = ExperimentSpec(
        name="final_historical_campaign_v1",
        start=date(2016, 7, 1),
        end=date(2026, 6, 30),
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=120,
        objective="long_horizon",
        objective_family=ObjectiveFamily.CAPITAL_ALLOCATION,
        baseline=CandidateSpec(id="b0_qqq100", policy=PolicyId.QQQ, modules=0, targets={"QQQ": 1.0}),
        candidates=[
            CandidateSpec(id="c1", policy=PolicyId.QQQ, modules=1, targets={"QQQ": 0.95, "SOXX": 0.05}),
            CandidateSpec(id="c2", policy=PolicyId.QQQ, modules=1, targets={"QQQ": 0.9, "SOXX": 0.1}),
            CandidateSpec(id="c3", policy=PolicyId.QQQ, modules=1, targets={"QQQ": 0.85, "SOXX": 0.15}),
        ],
    )
    report = run_final_historical_campaign(
        spec,
        _Runner(),
        seed=7,
        bootstrap_paths=50,
        cohort_horizon_months=24,
        cohort_step_months=12,
    )
    assert report.campaign_id == "FINAL_HISTORICAL_CAMPAIGN_V1"
    assert len(report.arm_rows) == 4
    assert report.arm_rows[0].arm_id == "b0_qqq100"
    assert report.arm_rows[0].median_ratio == pytest.approx(1.0)
    assert report.operational_unlock is False
    assert report.tax_sensitivity.status == "not_modelled"
    assert report.pre_history_proxy.status == "unavailable"
    assert report.lineage_census.total_experiments >= 1
    assert len(report.regime_coverage.rows) >= 5
    assert all(row.cohort_count >= 1 for row in report.arm_rows)
    assert report.arm_rows[1].median_ratio >= report.arm_rows[0].median_ratio


def test_write_final_historical_campaign_report(tmp_path) -> None:
    import json
    from datetime import date

    from src.data.settings import DataSettings
    from src.validation.historical_campaign import (
        FinalHistoricalArmMetrics,
        FinalHistoricalCampaignReport,
        PreHistoryProxyStressReport,
        RegimeCoverageReport,
        RegimeCoverageRow,
        TaxSensitivityMilestone,
        TrialLineageCensusReport,
        TrialLineageFamilyRow,
        write_final_historical_campaign_report,
    )

    report = FinalHistoricalCampaignReport(
        campaign_id="FINAL_HISTORICAL_CAMPAIGN_V1",
        window_start=date(2016, 7, 1),
        window_end=date(2026, 6, 30),
        arm_rows=(
            FinalHistoricalArmMetrics(
                arm_id="c2_qqq90_soxx10",
                targets={"QQQ": 0.9, "SOXX": 0.1},
                cohort_count=10,
                median_ratio=1.02,
                p10_ratio=1.0,
                worst_ratio=0.99,
                win_rate=0.9,
                ce_gamma_10=1.002,
                bootstrap_win_rate=0.8,
                bootstrap_p05=0.97,
                xirr_real=0.05,
                cost_stress_worst_ratio=0.98,
                fx_stress_worst_ratio=0.99,
                cohort_starts=(date(2016, 7, 1),),
                cohort_ends=(date(2026, 6, 30),),
            ),
        ),
        regime_coverage=RegimeCoverageReport(
            rows=(RegimeCoverageRow(regime_name="ai_boom_2023", covered=True, overlap_months=30),),
            independent_sample_warning=True,
        ),
        lineage_census=TrialLineageCensusReport(
            total_experiments=20,
            families=(TrialLineageFamilyRow(family_id="soxx", experiment_count=5, active_count=3, archived_count=2),),
        ),
        tax_sensitivity=TaxSensitivityMilestone(
            status="not_modelled",
            rationale="buy_only_accumulation_defers_realization_tax_until_sale; no PIT tax ledger model",
        ),
        pre_history_proxy=PreHistoryProxyStressReport(status="unavailable", reason="test"),
        operational_unlock=False,
    )
    settings = DataSettings(data_root=str(tmp_path))
    path = write_final_historical_campaign_report(report, settings, experiment_id="fhc_v1")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["campaign_id"] == "FINAL_HISTORICAL_CAMPAIGN_V1"
    assert payload["operational_unlock"] is False
    assert payload["lineage_census"]["total_experiments"] == 20
    assert payload["arm_rows"][0]["arm_id"] == "c2_qqq90_soxx10"


def test_resolve_final_campaign_window_requires_multiple_cohorts() -> None:
    from datetime import date

    import pytest

    from src.policy.targets import PolicyId
    from src.validation.experiment import BaselineSpec, CandidateSpec, ExperimentSpec
    from src.validation.historical_campaign import resolve_final_campaign_window
    from src.validation.research_posture import ObjectiveFamily

    wide_spec = ExperimentSpec(
        name="final_historical_campaign_v1",
        start=date(2006, 1, 1),
        end=date(2026, 6, 30),
        contribution_krw=1_000_000.0,
        objective_family=ObjectiveFamily.CAPITAL_ALLOCATION,
        baseline=BaselineSpec(id="b0", policy=PolicyId.QQQ, modules=0, targets={"QQQ": 1.0}),
        candidates=(CandidateSpec(id="c2", policy=PolicyId.QQQ, modules=1, targets={"QQQ": 0.9, "SOXX": 0.1}),),
    )
    start, end, cohorts = resolve_final_campaign_window(wide_spec, settings=None)
    assert len(cohorts) >= 10
    assert start <= cohorts[0][0]
    narrow_spec = ExperimentSpec(
        name="final_historical_campaign_v1",
        start=date(2016, 7, 1),
        end=date(2026, 6, 30),
        contribution_krw=1_000_000.0,
        objective_family=ObjectiveFamily.CAPITAL_ALLOCATION,
        baseline=BaselineSpec(id="b0", policy=PolicyId.QQQ, modules=0, targets={"QQQ": 1.0}),
        candidates=(),
    )
    with pytest.raises(ValueError, match="cohort"):
        resolve_final_campaign_window(narrow_spec, settings=None)

def test_final_historical_campaign_uses_unitized_bootstrap(monkeypatch) -> None:
    import ast
    from pathlib import Path

    source = Path("src/validation/historical_campaign.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "monthly_unitized_returns" in names
    assert "monthly_simple_returns" not in source.split("_compute_arm_metrics")[1].split("def run_final_historical_campaign")[0]



def test_compute_paired_cost_stress_ratios_candidate_beats_baseline() -> None:
    from datetime import date

    from src.policy.targets import PolicyId
    from src.sim.allocation import AllocationConfig, AllocationResult
    from src.validation.cost_grid import COST_SCENARIOS
    from src.validation.experiment import BaselineSpec, CandidateSpec, ExperimentSpec
    from src.validation.historical_campaign import compute_paired_cost_stress_ratios
    from src.validation.research_posture import ObjectiveFamily

    spec = ExperimentSpec(
        name="final_historical_campaign_v1",
        start=date(2016, 1, 1),
        end=date(2026, 6, 30),
        contribution_krw=1_000_000.0,
        objective_family=ObjectiveFamily.CAPITAL_ALLOCATION,
        baseline=BaselineSpec(id="b0", policy=PolicyId.QQQ, modules=0, targets={"QQQ": 1.0}),
        candidates=(CandidateSpec(id="c2", policy=PolicyId.QQQ, modules=1, targets={"QQQ": 0.9, "SOXX": 0.1}),),
    )

    def _runner(cfg: AllocationConfig) -> AllocationResult:
        is_candidate = cfg.targets_override is not None and float(cfg.targets_override.get("SOXX", 0.0)) > 0.0
        wealth = 110.0 if is_candidate else 100.0
        return AllocationResult(
            config=cfg,
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )

    rows = compute_paired_cost_stress_ratios(
        _runner,
        spec=spec,
        baseline_targets={"QQQ": 1.0},
        candidate_targets={"QQQ": 0.9, "SOXX": 0.1},
        scenarios=COST_SCENARIOS,
    )
    assert len(rows) == len(COST_SCENARIOS)
    ideal = next(r for r in rows if r.scenario_id == "ideal")
    assert ideal.candidate_over_baseline_ratio == 1.1



def test_classify_regime_coverage_tier_thresholds() -> None:
    from src.validation.historical_campaign import classify_regime_coverage_tier

    assert classify_regime_coverage_tier(overlap_months=45, regime_duration_months=50) == "full"
    assert classify_regime_coverage_tier(overlap_months=30, regime_duration_months=50) == "substantial"
    assert classify_regime_coverage_tier(overlap_months=5, regime_duration_months=50) == "partial"
    assert classify_regime_coverage_tier(overlap_months=0, regime_duration_months=50) == "none"



def test_audit_pre_history_mix_proxy_stress_reports_ratio(monkeypatch, tmp_path) -> None:
    from datetime import UTC, date, datetime
    from types import SimpleNamespace

    import polars as pl

    from src.data.settings import DataSettings
    from src.sim.allocation import AllocationConfig, AllocationResult
    from src.validation.historical_campaign import audit_pre_history_mix_proxy_stress

    session = date(1998, 3, 31)
    returns = pl.DataFrame(
        {
            "date": [session, session],
            "series_id": ["NDX100", "SOX"],
            "simple_return": [0.0, 0.0],
            "available_at": [session, session],
        }
    )

    monkeypatch.setattr("src.data.catalog.latest_artifact", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(
        "src.data.catalog.load_visible",
        lambda _settings, _dataset, _cutoff: returns,
    )
    monkeypatch.setattr(
        "src.data.schedule.build_decision_schedule",
        lambda *a, **k: [SimpleNamespace(execution_session=session, signal_session=session)],
    )
    monkeypatch.setattr(
        "src.data.calendar.load_calendar",
        lambda *a, **k: SimpleNamespace(
            close_ts=lambda _s: datetime(1998, 3, 31, 20, 0, tzinfo=UTC),
        ),
    )

    seen_series: list[list[str]] = []

    def _fake_proxy(cfg: AllocationConfig, _settings: DataSettings, frame: pl.DataFrame) -> AllocationResult:
        series = frame.get_column("series_id").unique().to_list()
        seen_series.append(series)
        wealth = 120.0 if series == ["PROXY_MIX"] else 100.0
        return AllocationResult(
            config=cfg,
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )

    monkeypatch.setattr(
        "src.validation.historical_campaign.run_research_proxy_from_store_with_returns",
        _fake_proxy,
    )
    report = audit_pre_history_mix_proxy_stress(
        DataSettings(data_root=str(tmp_path / "data")),
        window_start=date(1998, 3, 1),
        window_end=date(2002, 10, 31),
        contribution_krw=1_000_000.0,
        baseline_series="NDX100",
        candidate_weights={"NDX100": 0.9, "SOX": 0.1},
        regime_name="dot_com",
    )
    assert seen_series[0] == ["NDX100"]
    assert seen_series[1] == ["PROXY_MIX"]
    assert report.evidence_tier == "proxy_stress_only"
    assert report.status == "available"
    assert report.regime_name == "dot_com"
    assert report.candidate_over_baseline_ratio == 1.2



def test_write_final_historical_report_includes_paired_cost_stress(tmp_path) -> None:
    import json
    from datetime import date

    from src.data.settings import DataSettings
    from src.validation.historical_campaign import (
        FinalHistoricalArmMetrics,
        FinalHistoricalCampaignReport,
        PairedCostStressRow,
        PreHistoryMixProxyStressReport,
        RegimeCoverageReport,
        RegimeCoverageRow,
        PreHistoryProxyStressReport,
        TrialLineageCensusReport,
        TaxSensitivityMilestone,
        write_final_historical_campaign_report,
    )
    from src.validation.registry import TrialLineageHashCensus

    report = FinalHistoricalCampaignReport(
        campaign_id="FINAL_HISTORICAL_CAMPAIGN_V1",
        window_start=date(2006, 1, 1),
        window_end=date(2026, 6, 30),
        arm_rows=(
            FinalHistoricalArmMetrics(
                arm_id="c2_qqq90_soxx10",
                targets={"QQQ": 0.9, "SOXX": 0.1},
                cohort_count=10,
                median_ratio=1.02,
                p10_ratio=1.01,
                worst_ratio=0.99,
                win_rate=0.8,
                ce_gamma_10=1.002,
                bootstrap_win_rate=0.8,
                bootstrap_p05=0.98,
                xirr_real=0.05,
                cost_stress_worst_ratio=0.97,
                fx_stress_worst_ratio=0.96,
                cohort_starts=(date(2006, 1, 1),),
                cohort_ends=(date(2016, 1, 1),),
                paired_cost_stress=(PairedCostStressRow(scenario_id="ideal", candidate_over_baseline_ratio=1.05),),
            ),
        ),
        regime_coverage=RegimeCoverageReport(
            rows=(RegimeCoverageRow(regime_name="dot_com", covered=True, overlap_months=12, coverage_tier="partial", coverage_fraction=0.25),),
            independent_sample_warning=False,
        ),
        lineage_census=TrialLineageCensusReport(total_experiments=0, families=()),
        tax_sensitivity=TaxSensitivityMilestone(status="not_modelled", rationale="test"),
        pre_history_proxy=PreHistoryProxyStressReport(status="unavailable", reason="test"),
        pre_history_mix_proxy=(
            PreHistoryMixProxyStressReport(
                evidence_tier="proxy_stress_only",
                status="available",
                regime_name="dot_com",
                candidate_over_baseline_ratio=1.05,
            ),
        ),
        lineage_hash_census=TrialLineageHashCensus(unique_config_hashes=2, total_run_records=3),
        operational_unlock=False,
    )
    path = write_final_historical_campaign_report(report, DataSettings(data_root=str(tmp_path)), experiment_id="test")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["arm_rows"][0]["paired_cost_stress"][0]["scenario_id"] == "ideal"
    assert payload["regime_coverage"]["rows"][0]["coverage_tier"] == "partial"
    assert payload["lineage_hash_census"]["unique_config_hashes"] == 2
    assert payload["pre_history_mix_proxy"][0]["regime_name"] == "dot_com"

