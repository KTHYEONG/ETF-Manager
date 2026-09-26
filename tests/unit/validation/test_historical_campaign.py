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
    from src.validation.historical_campaign import build_trial_lineage_census

    from src.data.paths import EXPERIMENT_INDEX_PATH, EXPERIMENTS_DIR

    census = build_trial_lineage_census(
        index_path=EXPERIMENT_INDEX_PATH,
        experiments_dir=EXPERIMENTS_DIR,
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
        candidates=(CandidateSpec(id="c2", policy=PolicyId.QQQ, modules=1, targets={"QQQ": 0.9, "SOXX": 0.1}),),
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

    monkeypatch.setattr("src.data.catalog.resolve_snapshot", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(
        "src.data.catalog.load_snapshot_visible",
        lambda _snapshot, _dataset, _cutoff: returns,
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
        "src.validation.historical_campaign_audit.run_research_proxy_from_store_with_returns",
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



def test_resolve_final_campaign_window_short_lake_fails_closed(monkeypatch, tmp_path) -> None:
    """A lake shorter than the 120M grid fails closed after reading the pinned window."""
    from datetime import UTC, date, datetime

    import polars as pl
    import pytest

    from src.data.calendar import load_calendar
    from src.data.pipeline import persist_ingest
    from src.data.schema import Dataset
    from src.data.settings import DataSettings
    from src.data.storage import RawPayload
    from src.policy.targets import PolicyId
    from src.validation.experiment import BaselineSpec, CandidateSpec, ExperimentSpec
    from src.validation.historical_campaign import resolve_final_campaign_window
    from src.validation.research_posture import ObjectiveFamily

    monkeypatch.chdir(tmp_path)
    settings = DataSettings(data_root="data")
    retrieved_at = datetime(2024, 1, 5, 5, 0, tzinfo=UTC)
    sessions = list(load_calendar("XNYS").sessions(date(2024, 1, 2), date(2024, 1, 31)))
    rows = [
        {
            "ticker": ticker, "date": day, "open": 100.0, "high": 101.0,
            "low": 99.0, "close": 100.0, "volume": 10_000,
            "adjusted_close": 100.0, "dividend": 0.0, "split_factor": 1.0,
            "source": "synthetic", "retrieved_at": retrieved_at,
        }
        for ticker in ("QQQ", "SOXX")
        for day in sessions
    ]
    persist_ingest(
        pl.DataFrame(rows, schema={
            "ticker": pl.String, "date": pl.Date, "open": pl.Float64, "high": pl.Float64,
            "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64,
            "adjusted_close": pl.Float64, "dividend": pl.Float64, "split_factor": pl.Float64,
            "source": pl.String, "retrieved_at": pl.Datetime("us", "UTC"),
        }),
        Dataset.PRICES,
        RawPayload(provider="synthetic", endpoint="probe", request_params={},
                   retrieved_at=retrieved_at, extension="json", content=b"{}"),
        settings,
    )
    wide_spec = ExperimentSpec(
        name="final_historical_campaign_v1",
        start=date(2006, 1, 1),
        end=date(2026, 6, 30),
        contribution_krw=1_000_000.0,
        objective_family=ObjectiveFamily.CAPITAL_ALLOCATION,
        baseline=BaselineSpec(id="b0", policy=PolicyId.QQQ, modules=0, targets={"QQQ": 1.0}),
        candidates=(CandidateSpec(id="c2", policy=PolicyId.QQQ, modules=1, targets={"QQQ": 0.9, "SOXX": 0.1}),),
    )
    with pytest.raises(ValueError, match="cohort"):
        resolve_final_campaign_window(wide_spec, settings)



def test_run_final_historical_campaign_records_snapshot_hashes(monkeypatch, tmp_path) -> None:
    """The report carries pinned manifest identities for consumed and absent datasets."""
    from datetime import UTC, date, datetime

    import polars as pl

    from src.data.calendar import load_calendar
    from src.data.pipeline import persist_ingest
    from src.data.schema import Dataset, spec_for
    from src.data.settings import DataSettings
    from src.data.storage import RawPayload
    from src.policy.targets import PolicyId
    from src.sim.allocation import AllocationConfig, AllocationResult
    from src.validation.experiment import CandidateSpec, ExperimentSpec
    from src.validation.historical_campaign import run_final_historical_campaign
    from src.validation.research_posture import ObjectiveFamily

    monkeypatch.chdir(tmp_path)
    settings = DataSettings(data_root="data")
    retrieved_at = datetime(2024, 1, 5, 5, 0, tzinfo=UTC)
    sessions = list(load_calendar("XNYS").sessions(date(2024, 1, 2), date(2024, 1, 31)))
    persist_ingest(
        pl.DataFrame(
            {
                "series_id": ["NDX100"] * len(sessions),
                "date": sessions,
                "simple_return": [0.001] * len(sessions),
                "label": ["research"] * len(sessions),
                "source": ["synthetic"] * len(sessions),
                "retrieved_at": [retrieved_at] * len(sessions),
            },
            schema=dict(spec_for(Dataset.RESEARCH_RETURNS).columns),
        ),
        Dataset.RESEARCH_RETURNS,
        RawPayload(provider="synthetic", endpoint="probe", request_params={},
                   retrieved_at=retrieved_at, extension="json", content=b"{}"),
        settings,
    )

    class _Runner:
        def __call__(self, config: AllocationConfig) -> AllocationResult:
            return AllocationResult(
                config=config,
                snapshots=(),
                terminal_wealth_krw=100.0,
                xirr=0.0,
                max_drawdown=-0.1,
                terminal_wealth_real_krw=100.0,
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
        spec, _Runner(), seed=7, bootstrap_paths=5,
        cohort_horizon_months=24, cohort_step_months=12, settings=settings,
    )
    assert report.manifest_hashes["prices"] is None
    assert report.manifest_hashes["research_returns"] is not None



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



def test_research_proxy_remains_labeled(monkeypatch, tmp_path) -> None:
    """Pinned reads keep research returns and tradable prices in distinct identities."""
    from datetime import UTC, date, datetime

    import polars as pl

    from src.data.calendar import load_calendar
    from src.data.catalog import load_snapshot_visible, resolve_snapshot
    from src.data.pipeline import persist_ingest
    from src.data.schema import Dataset, spec_for
    from src.data.settings import DataSettings
    from src.data.storage import RawPayload

    monkeypatch.chdir(tmp_path)
    settings = DataSettings(data_root="data")
    retrieved_at = datetime(2024, 1, 5, 5, 0, tzinfo=UTC)

    def _payload() -> RawPayload:
        return RawPayload(
            provider="synthetic",
            endpoint="probe",
            request_params={},
            retrieved_at=retrieved_at,
            extension="json",
            content=b"{}",
        )

    sessions = list(load_calendar("XNYS").sessions(date(2024, 1, 2), date(2024, 1, 31)))
    returns_spec = spec_for(Dataset.RESEARCH_RETURNS)
    persist_ingest(
        pl.DataFrame(
            {
                "series_id": ["NDX100"] * len(sessions),
                "date": sessions,
                "simple_return": [0.001] * len(sessions),
                "label": ["research"] * len(sessions),
                "source": ["synthetic"] * len(sessions),
                "retrieved_at": [retrieved_at] * len(sessions),
            },
            schema=dict(returns_spec.columns),
        ),
        Dataset.RESEARCH_RETURNS,
        _payload(),
        settings,
    )
    prices_spec = spec_for(Dataset.PRICES)
    persist_ingest(
        pl.DataFrame(
            {
                "ticker": ["QQQ"] * len(sessions),
                "date": sessions,
                "open": [100.0] * len(sessions),
                "high": [101.0] * len(sessions),
                "low": [99.0] * len(sessions),
                "close": [100.0] * len(sessions),
                "volume": [10_000] * len(sessions),
                "adjusted_close": [100.0] * len(sessions),
                "dividend": [0.0] * len(sessions),
                "split_factor": [1.0] * len(sessions),
                "source": ["synthetic"] * len(sessions),
                "retrieved_at": [retrieved_at] * len(sessions),
            },
            schema=dict(prices_spec.columns),
        ),
        Dataset.PRICES,
        _payload(),
        settings,
    )

    snapshot = resolve_snapshot(settings, (Dataset.PRICES, Dataset.RESEARCH_RETURNS))
    assert snapshot.artifacts[Dataset.PRICES].manifest.dataset is Dataset.PRICES
    assert snapshot.artifacts[Dataset.RESEARCH_RETURNS].manifest.dataset is Dataset.RESEARCH_RETURNS
    cutoff = load_calendar("XNYS").close_ts(sessions[-1])
    prices = load_snapshot_visible(snapshot, Dataset.PRICES, cutoff)
    returns = load_snapshot_visible(snapshot, Dataset.RESEARCH_RETURNS, cutoff)
    assert "ticker" in prices.columns
    assert "series_id" not in prices.columns
    assert "series_id" in returns.columns
    assert "ticker" not in returns.columns
    assert set(returns.get_column("series_id").unique().to_list()).isdisjoint(
        set(prices.get_column("ticker").unique().to_list())
    )

    from src.validation.historical_campaign import _catalog_research_returns_min_date

    assert _catalog_research_returns_min_date(settings) == sessions[0]



def test_audit_evidence_parity_across_extraction() -> None:
    from datetime import date
    from pathlib import Path
    from types import SimpleNamespace

    from src.data.paths import EXPERIMENT_INDEX_PATH, EXPERIMENTS_DIR
    from src.data.settings import DataSettings
    from src.policy.targets import PolicyId
    from src.validation import historical_campaign as legacy
    from src.validation import historical_campaign_audit as extracted

    old_census = legacy.build_trial_lineage_census(
        index_path=Path(EXPERIMENT_INDEX_PATH), experiments_dir=Path(EXPERIMENTS_DIR)
    )
    new_census = extracted.build_trial_lineage_census(
        index_path=Path(EXPERIMENT_INDEX_PATH), experiments_dir=Path(EXPERIMENTS_DIR)
    )
    assert old_census == new_census
    assert new_census.total_experiments == sum(row.experiment_count for row in new_census.families)

    cohorts = ((date(2016, 7, 1), date(2026, 6, 30)),)
    assert legacy.audit_regime_coverage(cohorts=cohorts) == extracted.audit_regime_coverage(cohorts=cohorts)
    assert legacy.classify_regime_coverage_tier(overlap_months=0, regime_duration_months=10) == "none"

    settings = DataSettings(data_root="data")
    old_stress = legacy.audit_pre_history_mix_proxy_stress(
        settings,
        window_start=date(1998, 3, 1),
        window_end=date(2002, 10, 31),
        contribution_krw=1_000_000.0,
        baseline_series="",
        candidate_weights={},
        regime_name="dot_com",
    )
    new_stress = extracted.audit_pre_history_mix_proxy_stress(
        settings,
        window_start=date(1998, 3, 1),
        window_end=date(2002, 10, 31),
        contribution_krw=1_000_000.0,
        baseline_series="",
        candidate_weights={},
        regime_name="dot_com",
    )
    assert old_stress == new_stress
    assert new_stress.evidence_tier == "proxy_stress_only"
    assert new_stress.status == "unavailable"

    spec = SimpleNamespace(
        baseline=SimpleNamespace(policy=PolicyId.QQQ, targets={"QQQ": 1.0}),
        candidates=[SimpleNamespace(policy=PolicyId.QQQ, targets={"QQQ": 0.9, "SOXX": 0.1})],
        start=date(2016, 7, 1),
        end=date(2026, 6, 30),
        contribution_krw=1_000_000.0,
    )

    def runner(cfg):
        targets = dict(cfg.targets_override or {})
        wealth = 2_000_000.0 if targets == {"QQQ": 0.9, "SOXX": 0.1} else 1_000_000.0
        return SimpleNamespace(terminal_wealth_real_krw=wealth)

    scenarios = [SimpleNamespace(id="flat", commission_bps=0.0, fx_spread_bps=0.0)]
    rows = legacy.compute_paired_cost_stress_ratios(
        runner,
        spec=spec,
        baseline_targets={"QQQ": 1.0},
        candidate_targets={"QQQ": 0.9, "SOXX": 0.1},
        scenarios=scenarios,
    )
    assert [row.scenario_id for row in rows] == ["flat"]
    assert rows[0].candidate_over_baseline_ratio == 2.0


def _parity_historical_report():
    from datetime import date

    from src.validation.historical_campaign import (
        FinalHistoricalArmMetrics,
        FinalHistoricalCampaignReport,
        PreHistoryProxyStressReport,
        RegimeCoverageReport,
        RegimeCoverageRow,
        TaxSensitivityMilestone,
        TrialLineageCensusReport,
    )

    arm = FinalHistoricalArmMetrics(
        arm_id="c2_qqq90_soxx10",
        targets={"QQQ": 0.9, "SOXX": 0.1},
        cohort_count=4,
        median_ratio=1.05,
        p10_ratio=0.98,
        worst_ratio=0.95,
        win_rate=0.75,
        ce_gamma_10=1.02,
        bootstrap_win_rate=0.7,
        bootstrap_p05=0.99,
        xirr_real=0.08,
        cost_stress_worst_ratio=1.01,
        fx_stress_worst_ratio=1.0,
        cohort_starts=(date(2016, 7, 1),),
        cohort_ends=(date(2026, 6, 30),),
        paired_cost_stress=(),
    )
    return FinalHistoricalCampaignReport(
        campaign_id="FINAL_HISTORICAL_CAMPAIGN_V1",
        window_start=date(2016, 7, 1),
        window_end=date(2026, 6, 30),
        arm_rows=(arm,),
        regime_coverage=RegimeCoverageReport(
            rows=(RegimeCoverageRow(regime_name="gfc", covered=True, overlap_months=18, coverage_tier="full", coverage_fraction=1.0),),
            independent_sample_warning=True,
        ),
        lineage_census=TrialLineageCensusReport(total_experiments=0, families=()),
        tax_sensitivity=TaxSensitivityMilestone(status="not_modelled", rationale="probe"),
        pre_history_proxy=PreHistoryProxyStressReport(status="unavailable", reason="probe"),
        operational_unlock=False,
        manifest_hashes={"prices": "abc123"},
    )


def test_write_final_historical_campaign_report_parity(tmp_path) -> None:
    import json

    from src.data.settings import DataSettings
    from src.validation import historical_campaign as legacy
    from src.validation import historical_campaign_report as extracted

    report = _parity_historical_report()
    old_path = legacy.write_final_historical_campaign_report(
        report, DataSettings(data_root=tmp_path / "old" / "data"), experiment_id="parity"
    )
    new_path = extracted.write_final_historical_campaign_report(
        report, DataSettings(data_root=tmp_path / "new" / "data"), experiment_id="parity"
    )
    old_payload = json.loads(old_path.read_text(encoding="utf-8"))
    new_payload = json.loads(new_path.read_text(encoding="utf-8"))
    assert old_payload == new_payload
    assert new_payload["campaign_id"] == "FINAL_HISTORICAL_CAMPAIGN_V1"
    assert new_payload["operational_unlock"] is False
    assert new_payload["manifest_hashes"] == {"prices": "abc123"}
    assert new_payload["arm_rows"][0]["median_ratio"] == 1.05


def test_audit_boundary_branches_pure() -> None:
    from datetime import date

    from src.validation import historical_campaign_audit as audit

    assert audit._months_between_inclusive(date(2020, 2, 1), date(2020, 1, 1)) == 0
    assert audit.classify_regime_coverage_tier(overlap_months=0, regime_duration_months=0) == "none"
    assert audit.audit_regime_coverage(cohorts=[]).independent_sample_warning is False


def test_audit_pre_history_mix_proxy_stress_failure_branches(monkeypatch, tmp_path) -> None:
    from datetime import UTC, date, datetime
    from types import SimpleNamespace

    import polars as pl

    from src.data.settings import DataSettings
    from src.data.storage import UntrustedDatasetError
    from src.validation import historical_campaign_audit as audit

    settings = DataSettings(data_root=str(tmp_path / "data"))
    session = date(2000, 1, 3)
    frame = pl.DataFrame({"date": [session], "series_id": ["NDX100"]})

    def _kwargs():
        return dict(
            window_start=date(1998, 3, 1),
            window_end=date(2002, 10, 31),
            contribution_krw=1_000_000.0,
            baseline_series="NDX100",
            candidate_weights={"NDX100": 0.9, "SOX": 0.1},
            regime_name="dot_com",
        )

    def _boom(*args, **kwargs):
        raise RuntimeError("probe")

    def _missing_catalog(*args, **kwargs):
        raise UntrustedDatasetError("probe")

    monkeypatch.setattr("src.validation.prospective_registry._allocation_end_within_as_of", _boom)
    monkeypatch.setattr("src.data.catalog.resolve_snapshot", _missing_catalog)
    fallen_back = audit.audit_pre_history_mix_proxy_stress(settings, **_kwargs())
    assert fallen_back.status == "unavailable"
    assert fallen_back.window_end == date(2002, 10, 31)

    monkeypatch.setattr(
        "src.validation.prospective_registry._allocation_end_within_as_of",
        lambda *args, **kwargs: date(2002, 10, 31),
    )
    monkeypatch.setattr("src.data.catalog.resolve_snapshot", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr("src.data.catalog.load_snapshot_visible", lambda *args, **kwargs: frame)
    monkeypatch.setattr("src.data.schedule.build_decision_schedule", lambda *args, **kwargs: [])
    empty_schedule = audit.audit_pre_history_mix_proxy_stress(settings, **_kwargs())
    assert empty_schedule.status == "unavailable"
    assert "empty proxy schedule" in empty_schedule.reason

    monkeypatch.setattr(
        "src.data.schedule.build_decision_schedule",
        lambda *args, **kwargs: [SimpleNamespace(execution_session=session, signal_session=session)],
    )
    monkeypatch.setattr(
        "src.data.calendar.load_calendar",
        lambda *args, **kwargs: SimpleNamespace(close_ts=lambda _s: datetime(2000, 1, 3, 20, 0, tzinfo=UTC)),
    )
    missing_series = audit.audit_pre_history_mix_proxy_stress(settings, **_kwargs())
    assert missing_series.status == "unavailable"
    assert "missing research_returns series" in missing_series.reason

    full_frame = pl.DataFrame(
        {
            "date": [session, session],
            "series_id": ["NDX100", "SOX"],
            "simple_return": [0.0, 0.0],
            "available_at": [session, session],
        }
    )
    monkeypatch.setattr("src.data.catalog.load_snapshot_visible", lambda *args, **kwargs: full_frame)
    monkeypatch.setattr(
        "src.validation.historical_campaign_audit.run_research_proxy_from_store_with_returns",
        lambda *args, **kwargs: SimpleNamespace(terminal_wealth_real_krw=0.0, xirr_real=0.0),
    )
    non_positive = audit.audit_pre_history_mix_proxy_stress(settings, **_kwargs())
    assert non_positive.status == "unavailable"
    assert "non-positive baseline proxy wealth" in non_positive.reason


def test_catalog_research_returns_min_date_branches(monkeypatch, tmp_path) -> None:
    from datetime import date
    from types import SimpleNamespace

    import polars as pl

    from src.data.schema import Dataset
    from src.data.settings import DataSettings
    from src.validation import historical_campaign_audit as audit

    settings = DataSettings(data_root=str(tmp_path / "data"))
    snapshot = SimpleNamespace(artifacts={Dataset.RESEARCH_RETURNS: SimpleNamespace(manifest_path="probe")})
    monkeypatch.setattr("src.data.catalog.resolve_snapshot", lambda *args, **kwargs: snapshot)

    monkeypatch.setattr(
        "src.data.storage.DataStore.read_normalized",
        lambda *args, **kwargs: pl.DataFrame({"date": []}, schema={"date": pl.Date}),
    )
    assert audit._catalog_research_returns_min_date(settings) is None
    started_late = audit.audit_pre_history_proxy_stress(
        settings,
        proxy_start=date(2002, 10, 31),
        proxy_end=date(1998, 3, 1),
        contribution_krw=1_000_000.0,
    )
    assert started_late.status == "unavailable"
    assert started_late.reason == "proxy_start_after_end"

    monkeypatch.setattr(
        "src.data.storage.DataStore.read_normalized",
        lambda *args, **kwargs: pl.DataFrame({"date": [None]}, schema={"date": pl.Date}),
    )
    assert audit._catalog_research_returns_min_date(settings) is None
    assert (
        audit.audit_pre_history_proxy_stress(
            settings,
            proxy_start=date(1998, 3, 1),
            proxy_end=date(2002, 10, 31),
            contribution_krw=1_000_000.0,
        ).status
        == "unavailable"
    )


def test_audit_pre_history_proxy_stress_available(monkeypatch, tmp_path) -> None:
    from datetime import date
    from types import SimpleNamespace

    import polars as pl

    from src.data.schema import Dataset
    from src.data.settings import DataSettings
    from src.validation import historical_campaign_audit as audit

    settings = DataSettings(data_root=str(tmp_path / "data"))
    snapshot = SimpleNamespace(artifacts={Dataset.RESEARCH_RETURNS: SimpleNamespace(manifest_path="probe")})
    monkeypatch.setattr("src.data.catalog.resolve_snapshot", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(
        "src.data.storage.DataStore.read_normalized",
        lambda *args, **kwargs: pl.DataFrame({"date": [date(1990, 1, 1)]}),
    )
    monkeypatch.setattr(
        "src.sim.research_proxy.run_research_proxy_from_store",
        lambda *args, **kwargs: SimpleNamespace(terminal_wealth_real_krw=100.0, xirr_real=0.05),
    )
    report = audit.audit_pre_history_proxy_stress(
        settings,
        proxy_start=date(1998, 3, 1),
        proxy_end=date(2002, 10, 31),
        contribution_krw=1_000_000.0,
    )
    assert report.status == "available"
    assert report.proxy_window_start == date(1998, 3, 1)
    assert report.terminal_wealth_real_krw == 100.0
    assert report.xirr_real == 0.05
    monkeypatch.setattr(
        "src.sim.research_proxy.run_research_proxy_from_store",
        lambda *args, **kwargs: SimpleNamespace(terminal_wealth_real_krw=0.0, xirr_real=0.0),
    )
    monkeypatch.setattr(
        "src.data.storage.DataStore.read_normalized",
        lambda *args, **kwargs: pl.DataFrame({"date": [date(1998, 3, 1)]}),
    )
    flat = audit.audit_pre_history_proxy_stress(
        settings,
        proxy_start=date(1998, 3, 1),
        proxy_end=date(2002, 10, 31),
        contribution_krw=1_000_000.0,
    )
    assert flat.status == "unavailable"
    assert "non-positive proxy terminal wealth" in flat.reason
