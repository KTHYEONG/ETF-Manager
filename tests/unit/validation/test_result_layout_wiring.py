"""Smoke guards that each mechanical writer delegates to the result store layout."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

from src.data.settings import DataSettings
from src.policy.targets import PolicyId
from src.validation.experiment import CandidateSpec, ExperimentSpec


def _settings(tmp_path: Path) -> DataSettings:
    return DataSettings(data_root=tmp_path / "data")


def _campaign(name: str, *, adopted: bool = True):  # type: ignore[no-untyped-def]
    from src.validation.walk_forward import CampaignReport

    return CampaignReport(
        name=name,
        candidate_id="cand",
        modules=1,
        folds=(),
        baseline_test_ce={},
        candidate_test_ce={},
        chosen_test_ce={},
        process_adopted_vs_baseline=adopted,
    )


def _spec(name: str) -> ExperimentSpec:
    return ExperimentSpec(
        name=name,
        start=date(2012, 1, 3),
        end=date(2024, 12, 31),
        contribution_krw=1_000_000.0,
        hurdle=0.02,
        horizon_months=0,
        baseline=CandidateSpec(id="qqq_baseline", policy=PolicyId.QQQ, modules=0),
        candidates=[CandidateSpec(id="cand", policy=PolicyId.QQQ, modules=1)],
    )


def test_cost_grid_writer_uses_result_store(tmp_path: Path) -> None:
    from src.validation.cost_grid import CostGridReport, CostScenario, ScenarioOutcome, write_cost_grid_report

    settings = _settings(tmp_path)
    report = CostGridReport(
        name="grid_smoke",
        outcomes=(
            ScenarioOutcome(scenario=CostScenario(id="base", commission_bps=10.0, fx_spread_bps=20.0), campaign=_campaign("grid_smoke")),
        ),
    )
    out = write_cost_grid_report(report, settings, experiment_id="abc123")
    assert out == tmp_path / "data" / "runs" / "grid_smoke" / "costs_abc123.json"
    assert json.loads(out.read_text(encoding="utf-8"))["name"] == "grid_smoke"


def test_cadence_robustness_writer_uses_result_store(tmp_path: Path) -> None:
    from src.validation.cadence_robustness import CadenceRobustnessReport, write_cadence_robustness_report
    from src.validation.cost_grid import CostGridReport

    settings = _settings(tmp_path)
    report = CadenceRobustnessReport(
        name="cadence_smoke",
        cost_grid=CostGridReport(name="cadence_smoke", outcomes=()),
        baseline_wealths=(100.0,),
        candidate_wealths=(101.0,),
        worst_cohort_ok=True,
        bootstrap_tail_ok=True,
        robust_adopted=True,
    )
    out = write_cadence_robustness_report(report, settings, experiment_id="abc123")
    assert out == tmp_path / "data" / "runs" / "cadence_smoke" / "robustness_abc123.json"


def test_strategy_selection_writer_uses_result_store(tmp_path: Path) -> None:
    from src.validation.strategy_selection import StrategyArmRow, StrategySelectionReport, StrategyVerdict, write_strategy_selection_report

    settings = _settings(tmp_path)
    report = StrategySelectionReport(
        name="select_smoke",
        baseline_arm_id="b1",
        objective="growth_first",
        rows=(
            StrategyArmRow(arm_id="b1", process_adopted_vs_baseline=False, pooled_oos_real_gain=1.0, pooled_oos_tw_ratio=1.0, in_sample_real_gain=1.0, verdict=StrategyVerdict.RESEARCH_ONLY, fold_count=1),
        ),
        in_sample_champion_arm_id="b1",
        oos_eligible_arm_ids=(),
        recommended_arm_id="b1",
        operational_unlock=False,
        selection_reason="test",
    )
    out = write_strategy_selection_report(report, settings, experiment_id="abc123")
    assert out == tmp_path / "data" / "runs" / "select_smoke" / "selection_abc123.json"


def test_accumulation_writer_uses_result_store(tmp_path: Path) -> None:
    from src.validation.accumulation_cohort import AccumulationCohortReport, CohortOverlapMetadata, write_accumulation_cohort_report

    settings = _settings(tmp_path)
    report = AccumulationCohortReport(
        name="accum_smoke",
        overlap=CohortOverlapMetadata(horizon_months=120, step_months=12),
        rows=(),
        median_ratio=1.0,
        p10_ratio=1.0,
        worst_ratio=1.0,
        win_rate=1.0,
        bootstrap_p05_ratio_mean=1.0,
        unrecovered_cohort_count=0,
    )
    out = write_accumulation_cohort_report(report, settings, experiment_id="abc123")
    assert out == tmp_path / "data" / "runs" / "accum_smoke" / "accumulation_abc123.json"


def test_feasibility_writer_uses_result_store(tmp_path: Path) -> None:
    from src.validation.feasibility_audit import FeasibilityDependencyProfile, StaticDcaWindowReport, write_feasibility_audit_report

    settings = _settings(tmp_path)
    report = StaticDcaWindowReport(
        name="feas_smoke",
        requested_start=date(2020, 1, 1),
        requested_end=date(2024, 12, 31),
        dependency=FeasibilityDependencyProfile(profile="p", required_datasets=(), requires_macro=False),
        ticker_coverage=(),
        dataset_coverage=(),
        limiting_factors=(),
        earliest_feasible_start=None,
        latest_feasible_end=None,
        cohort_count_120m_step12=0,
        resolve_violations=(),
    )
    out = write_feasibility_audit_report(report, settings, audit_id="audit1")
    assert out == tmp_path / "data" / "runs" / "feas_smoke" / "feasibility_audit1.json"


def test_prospective_freeze_writer_uses_result_store(tmp_path: Path) -> None:
    from src.validation.prospective import ProspectiveFreezeRecord
    from src.validation.registry import write_prospective_freeze_record

    settings = _settings(tmp_path)
    spec = _spec("freeze_smoke")
    freeze = ProspectiveFreezeRecord(
        thesis_id="ai_compute",
        frozen_at=datetime(2026, 1, 1, tzinfo=UTC),
        targets_hash="abc",
        experiment_name="freeze_smoke",
    )
    out = write_prospective_freeze_record(spec=spec, freeze=freeze, settings=settings)
    assert out.parent == tmp_path / "data" / "runs" / "freeze_smoke"
    assert out.name.startswith("prospective_freeze_")


def test_historical_census_scans_results_root(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import json as _json

    from src.data.paths import results_root
    from src.policy.targets import PolicyId as _PolicyId
    from src.sim.allocation import AllocationResult as _Result
    from src.validation.experiment import CandidateSpec as _Cand, ExperimentSpec as _Spec
    from src.validation.historical_campaign import run_final_historical_campaign
    from src.validation.research_posture import ObjectiveFamily as _Family

    monkeypatch.chdir(tmp_path)
    settings = DataSettings(data_root=tmp_path / "data")

    class _Runner:
        def __call__(self, config):  # type: ignore[no-untyped-def]
            wealth = 100.0 + float((config.targets_override or {}).get("SOXX", 0.0)) * 10.0
            return _Result(
                config=config,
                snapshots=(),
                terminal_wealth_krw=wealth,
                xirr=0.0,
                max_drawdown=-0.1,
                terminal_wealth_real_krw=wealth,
                xirr_real=0.05,
                total_contribution_real_krw=90.0,
            )

    spec = _Spec(
        name="final_historical_campaign_v1",
        start=date(2016, 7, 1),
        end=date(2026, 6, 30),
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=120,
        objective="long_horizon",
        objective_family=_Family.CAPITAL_ALLOCATION,
        baseline=_Cand(id="b0_qqq100", policy=_PolicyId.QQQ, modules=0, targets={"QQQ": 1.0}),
        candidates=[
            _Cand(id="c1", policy=_PolicyId.QQQ, modules=1, targets={"QQQ": 0.95, "SOXX": 0.05}),
            _Cand(id="c2", policy=_PolicyId.QQQ, modules=1, targets={"QQQ": 0.9, "SOXX": 0.1}),
            _Cand(id="c3", policy=_PolicyId.QQQ, modules=1, targets={"QQQ": 0.85, "SOXX": 0.15}),
        ],
    )
    seed_dir = results_root(settings) / "seed_exp"
    seed_dir.mkdir(parents=True)
    (seed_dir / "seed.json").write_text(_json.dumps({"config_hash": "abc"}), encoding="utf-8")

    import src.validation.historical_campaign as _hc

    monkeypatch.setattr(
        _hc,
        "audit_pre_history_proxy_stress",
        lambda *args, **kwargs: _hc.PreHistoryProxyStressReport(status="unavailable", reason="test"),
    )
    monkeypatch.setattr(
        _hc,
        "audit_pre_history_mix_proxy_stress",
        lambda *args, **kwargs: _hc.PreHistoryMixProxyStressReport(evidence_tier="t", status="unavailable"),
    )
    report = run_final_historical_campaign(
        spec, _Runner(), seed=7, bootstrap_paths=10, cohort_horizon_months=24, cohort_step_months=12, settings=settings
    )
    assert report.lineage_hash_census is not None
    assert report.lineage_hash_census.total_run_records == 1
