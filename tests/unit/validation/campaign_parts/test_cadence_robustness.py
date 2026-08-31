"""Unit tests for the walk-forward adoption campaign."""

from __future__ import annotations

import json
from datetime import date

import pytest


from src.data.settings import DataSettings
from src.policy.adaptive_contribution import AdaptiveContributionConfig
from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig, AllocationResult
from src.validation.campaign import (
    run_cadence_robustness,
    run_walk_forward_adoption,
    write_campaign_report,
)
from src.validation.experiment import (
    AdaptiveContributionSpec,
    CadenceSpec,
    CandidateSpec,
    ExperimentSpec,
    ReserveSpec,
    load_experiment_config,
    resolve_adaptive_contribution,
    resolve_baseline_adaptive_contribution,
)



def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="wf_s0_s1",
        start=date(2012, 4, 1),
        end=date(2024, 11, 30),
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        train_months=60,
        test_months=36,
        baseline=CandidateSpec(id="s0_global", policy="s0_global", modules=0),
        candidates=[CandidateSpec(id="s1_us", policy="s1_us", modules=1)],
    )


class _RecordingRunner:
    """Callable runner that records every injected AllocationConfig."""

    def __init__(self, wealth_by_policy: dict[PolicyId, float]) -> None:
        self._wealth_by_policy = wealth_by_policy
        self.configs: list[AllocationConfig] = []

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        self.configs.append(config)
        wealth = float(self._wealth_by_policy[config.policy])
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )



class _GrowthRunner:
    """Candidate real TW runs 1.5% above baseline each window; MDD never worsens."""

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        wealth, mdd = (101.5, -0.24) if config.cadence != "monthly" else (100.0, -0.25)
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=mdd,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )


class _ReserveGrowthRunner:
    """Reserve arms run 1.5% above baseline each window; MDD never worsens."""

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        wealth, mdd = (101.5, -0.24) if config.reserve is not None else (100.0, -0.25)
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=mdd,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )


@pytest.mark.parametrize("scenario_id", ["GF-C-wf-objective"])
def test_gf_c_wf_objective(scenario_id: str) -> None:
    """GF-C-wf-objective"""
    common: dict[str, object] = {
        "name": "gf_c_wf",
        "start": date(2012, 4, 1),
        "end": date(2024, 11, 30),
        "contribution_krw": 1_000_000.0,
        "hurdle": 0.02,
        "horizon_months": 0,
        "train_months": 60,
        "test_months": 36,
        "baseline": CandidateSpec(id="base", policy="vt", modules=0),
        "candidates": [CandidateSpec(id="cand", policy="vti", modules=1)],
    }
    ce_spec = ExperimentSpec(**common)

    ce_report = run_walk_forward_adoption(ce_spec, _GrowthRunner())

    # 1.5% edge loses to the 2% complexity hurdle on both train and process gates.
    assert ce_spec.objective == "ce"
    assert all(fold.train_adopted is False for fold in ce_report.folds)
    assert ce_report.process_adopted_vs_baseline is False

    gf_spec = ExperimentSpec(
        **common,
        objective="growth_first",
        cadence=CadenceSpec(anchor="month_open"),
    )

    gf_report = run_walk_forward_adoption(gf_spec, _GrowthRunner())

    assert all(fold.train_adopted is True for fold in gf_report.folds)
    assert gf_report.process_adopted_vs_baseline is True

    gf_reserve_spec = ExperimentSpec(
        **common,
        objective="growth_first",
        reserve=ReserveSpec(max_withhold=0.10, schedule="v3"),
    )
    gf_reserve_report = run_walk_forward_adoption(gf_reserve_spec, _ReserveGrowthRunner())

    assert all(fold.train_adopted is True for fold in gf_reserve_report.folds)
    assert gf_reserve_report.process_adopted_vs_baseline is True

    # model_copy bypasses model validation, so the campaign must fail closed itself.
    with pytest.raises(ValueError, match="cadence"):
        run_walk_forward_adoption(gf_spec.model_copy(update={"cadence": None}), _GrowthRunner())
    with pytest.raises(ValueError, match="reserve"):
        run_walk_forward_adoption(
            gf_reserve_spec.model_copy(update={"reserve": None}),
            _ReserveGrowthRunner(),
        )


class _CadenceEdgeRunner:
    """Month-open real TW runs 5% above monthly; stress commissions can erase the edge."""

    def __init__(self, *, stress_collapses: bool) -> None:
        self._stress_collapses = stress_collapses

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        edge_active = config.cadence != "monthly" and not (
            self._stress_collapses and config.commission_bps >= 50.0
        )
        wealth, mdd = (105.0, -0.24) if edge_active else (100.0, -0.25)
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=mdd,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )


def _cadence_robustness_spec() -> ExperimentSpec:
    """Growth-first month-open candidate on a window that fits cohorts and folds."""
    return ExperimentSpec(
        name="gf_r_qqq_month_open",
        start=date(2012, 4, 1),
        end=date(2024, 11, 30),
        contribution_krw=1_000_000.0,
        hurdle=0.02,
        horizon_months=0,
        objective="growth_first",
        train_months=60,
        test_months=36,
        cadence=CadenceSpec(anchor="month_open"),
        baseline=CandidateSpec(id="base_monthly", policy="vt", modules=0),
        candidates=[CandidateSpec(id="cand_month_open", policy="vti", modules=1)],
    )


@pytest.mark.parametrize("scenario_id", ["GF-R-cadence-and"])
def test_gf_r_cadence_and(scenario_id: str) -> None:
    """GF-R-cadence-and"""
    spec = _cadence_robustness_spec()
    report = run_cadence_robustness(spec, _CadenceEdgeRunner(stress_collapses=False), n_paths=40, seed=7)

    assert report.robust_adopted is True
    assert report.worst_cohort_ok is True
    assert report.bootstrap_tail_ok is True
    assert report.cost_grid.all_scenarios_adopted is True
    assert report.baseline_wealths
    assert len(report.candidate_wealths) == len(report.baseline_wealths)
    assert all(
        candidate / baseline == pytest.approx(1.05)
        for candidate, baseline in zip(report.candidate_wealths, report.baseline_wealths, strict=True)
    )
    # The spec is never mutated.
    assert spec.contribution_krw == pytest.approx(1_000_000.0)

    stressed = run_cadence_robustness(spec, _CadenceEdgeRunner(stress_collapses=True), n_paths=40, seed=7)

    assert stressed.robust_adopted is False
    assert stressed.cost_grid.all_scenarios_adopted is False
    assert stressed.worst_cohort_ok is True
    assert stressed.bootstrap_tail_ok is True
    # Cohort arms stay populated even when the cost arm vetoes adoption.
    assert len(stressed.candidate_wealths) == len(report.candidate_wealths)

    ce_objective = spec.model_copy(update={"objective": "ce"})
    with pytest.raises(ValueError, match="growth_first"):
        run_cadence_robustness(ce_objective, _CadenceEdgeRunner(stress_collapses=False), n_paths=40, seed=7)

    with pytest.raises(ValueError, match="cadence"):
        run_cadence_robustness(
            spec.model_copy(update={"cadence": None}),
            _CadenceEdgeRunner(stress_collapses=False),
            n_paths=40,
            seed=7,
        )


class _AdaptiveWealthRunner:
    """Adaptive arms gain real wealth, profit, and XIRR over baseline arms."""

    def __init__(self) -> None:
        self.configs: list[AllocationConfig] = []

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        self.configs.append(config)
        if config.adaptive_contribution is not None:
            wealth, contribution, xirr_real, mdd = 120.0, 95.0, 0.20, -0.24
        else:
            wealth, contribution, xirr_real, mdd = 100.0, 90.0, 0.10, -0.25
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=mdd,
            terminal_wealth_real_krw=wealth,
            xirr_real=xirr_real,
            total_contribution_real_krw=contribution,
        )


class _LosingAdaptiveRunner:
    """Adaptive arms lose real TW and profit on train, so adoption never happens."""

    def __init__(self) -> None:
        self.configs: list[AllocationConfig] = []

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        self.configs.append(config)
        if config.adaptive_contribution is not None:
            wealth, contribution, xirr_real = 90.0, 90.0, 0.05
        else:
            wealth, contribution, xirr_real = 100.0, 90.0, 0.10
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=-0.25,
            terminal_wealth_real_krw=wealth,
            xirr_real=xirr_real,
            total_contribution_real_krw=contribution,
        )


def _acg_spec() -> ExperimentSpec:
    """Adaptive-growth candidate on the shared walk-forward window."""
    return ExperimentSpec(
        name="wf_acg",
        start=date(2012, 4, 1),
        end=date(2024, 11, 30),
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        objective="adaptive_growth",
        train_months=60,
        test_months=36,
        adaptive_contribution={},
        baseline=CandidateSpec(id="s0_global", policy="s0_global", modules=0),
        candidates=[CandidateSpec(id="cand_acg", policy="vti", modules=1)],
    )


@pytest.mark.parametrize("scenario_id", ["WF-ACG-adoption"])
def test_wf_acg_adoption(scenario_id: str, tmp_path) -> None:
    """WF-ACG-adoption"""
    runner = _AdaptiveWealthRunner()
    report = run_walk_forward_adoption(_acg_spec(), runner)

    assert len(report.folds) > 0
    assert all(fold.train_adopted is True for fold in report.folds)
    assert report.process_adopted_vs_baseline is True
    for fold_index in range(len(report.folds)):
        chunk = runner.configs[fold_index * 4 : (fold_index + 1) * 4]
        assert len(chunk) == 4
        adaptive_states = [config.adaptive_contribution is not None for config in chunk]
        # Baseline train/test stay flat; candidate and adopted chosen arms carry the module.
        assert adaptive_states == [False, True, False, True]
        assert isinstance(chunk[1].adaptive_contribution, AdaptiveContributionConfig)
        fold = report.folds[fold_index]
        # Every arm records TW, total real contribution, real gain, and real XIRR.
        assert fold.baseline_test_wealth == pytest.approx(100.0)
        assert fold.candidate_test_wealth == pytest.approx(120.0)
        assert fold.chosen_test_wealth == pytest.approx(120.0)
        assert fold.baseline_total_contribution_real_krw == pytest.approx(90.0)
        assert fold.candidate_total_contribution_real_krw == pytest.approx(95.0)
        assert fold.chosen_total_contribution_real_krw == pytest.approx(95.0)
        assert fold.baseline_real_gain == pytest.approx(10.0)
        assert fold.candidate_real_gain == pytest.approx(25.0)
        assert fold.chosen_xirr_real == pytest.approx(0.20)

    settings = DataSettings(data_root=str(tmp_path))
    out_path = write_campaign_report(report, settings, experiment_id="acg")
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    record = payload["folds"][0]
    for arm in ("baseline", "candidate", "chosen"):
        assert f"{arm}_test_wealth" in record
        assert f"{arm}_total_contribution_real_krw" in record
        assert f"{arm}_real_gain" in record
        assert f"{arm}_xirr_real" in record


@pytest.mark.parametrize("scenario_id", ["WF-ACG-adoption"])
def test_wf_acg_rejection_and_delegation(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """WF-ACG-adoption"""
    losing = _LosingAdaptiveRunner()
    rejected = run_walk_forward_adoption(_acg_spec(), losing)

    assert len(rejected.folds) > 0
    assert all(fold.train_adopted is False for fold in rejected.folds)
    assert rejected.process_adopted_vs_baseline is False
    for fold_index in range(len(rejected.folds)):
        chunk = losing.configs[fold_index * 4 : (fold_index + 1) * 4]
        adaptive_states = [config.adaptive_contribution is not None for config in chunk]
        # Without train adoption the chosen arm falls back to the flat baseline.
        assert adaptive_states == [False, True, False, True]

    # Process adoption delegates to contribution_growth_process_passes.
    monkeypatch.setattr(
        "src.validation.campaign.contribution_growth_process_passes",
        lambda **_kwargs: False,
    )
    vetoed = run_walk_forward_adoption(_acg_spec(), _AdaptiveWealthRunner())
    assert vetoed.process_adopted_vs_baseline is False

    # model_copy bypasses model validation, so the campaign must fail closed itself.
    with pytest.raises(ValueError, match="adaptive_contribution"):
        run_walk_forward_adoption(
            _acg_spec().model_copy(update={"adaptive_contribution": None}), _AdaptiveWealthRunner()
        )


@pytest.mark.parametrize("scenario_id", ["WF-AG-baseline-arm"])
def test_wf_ag_baseline_arm(scenario_id: str) -> None:
    """WF-AG-baseline-arm"""
    baseline_adaptive_spec = _acg_spec().model_copy(
        update={"name": "wf_acg_baseline_arm", "baseline_adaptive_contribution": AdaptiveContributionSpec()}
    )
    runner = _AdaptiveWealthRunner()
    report = run_walk_forward_adoption(baseline_adaptive_spec, runner)

    resolved_baseline = resolve_baseline_adaptive_contribution(baseline_adaptive_spec)
    assert isinstance(resolved_baseline, AdaptiveContributionConfig)
    assert len(report.folds) > 0
    for fold_index in range(len(report.folds)):
        chunk = runner.configs[fold_index * 4 : (fold_index + 1) * 4]
        # Baseline train/test carry the locked adaptive config; candidate keeps its own.
        assert chunk[0].adaptive_contribution == resolved_baseline
        assert chunk[2].adaptive_contribution == resolved_baseline
        assert chunk[1].adaptive_contribution == resolve_adaptive_contribution(baseline_adaptive_spec)

    flat_runner = _AdaptiveWealthRunner()
    run_walk_forward_adoption(_acg_spec(), flat_runner)
    for fold_index in range(len(flat_runner.configs) // 4):
        flat_chunk = flat_runner.configs[fold_index * 4 : (fold_index + 1) * 4]
        assert flat_chunk[0].adaptive_contribution is None
        assert flat_chunk[2].adaptive_contribution is None


@pytest.mark.parametrize("scenario_id", ["WF-AG-v4-process"])
def test_wf_ag_v4_process(scenario_id: str) -> None:
    """WF-AG-v4-process"""

    class _V4AdaptiveRunner:
        def __call__(self, config: AllocationConfig) -> AllocationResult:
            if config.adaptive_contribution is None:
                wealth, contribution, xirr_real, mdd = 100.0, 90.0, 0.10, -0.25
            elif config.adaptive_contribution.neutral_deadband >= 4.0:
                wealth, contribution, xirr_real, mdd = 130.0, 100.0, 0.22, -0.24
            else:
                wealth, contribution, xirr_real, mdd = 110.0, 95.0, 0.15, -0.25
            return AllocationResult(
                config=config,
                snapshots=(),
                terminal_wealth_krw=wealth,
                xirr=0.0,
                max_drawdown=mdd,
                terminal_wealth_real_krw=wealth,
                xirr_real=xirr_real,
                total_contribution_real_krw=contribution,
            )

    spec = load_experiment_config("configs/experiments/wf_qqq_adaptive_v4.json")
    report = run_walk_forward_adoption(spec, _V4AdaptiveRunner())

    assert len(report.folds) >= 2
    assert all(fold.train_adopted is True for fold in report.folds)
    assert report.process_adopted_vs_baseline is True

    baseline_tw = sum(fold.baseline_test_wealth for fold in report.folds)
    chosen_tw = sum(fold.chosen_test_wealth for fold in report.folds)
    assert chosen_tw / baseline_tw >= 1.08

    for fold in report.folds:
        assert fold.chosen_xirr_real >= fold.baseline_xirr_real


class _MixConfigRunner:
    """Records configs for static-mix walk-forward wiring checks."""

    def __init__(self) -> None:
        self.configs: list[AllocationConfig] = []

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        self.configs.append(config)
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=100.0,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=100.0,
            xirr_real=0.0,
        )


@pytest.mark.parametrize("scenario_id", ["CAM-MIX-override-wired"])
def test_cam_mix_override_wired(scenario_id: str) -> None:
    """CAM-MIX-override-wired"""
    spec = ExperimentSpec(
        name="wf_mix",
        start=date(2012, 4, 1),
        end=date(2024, 11, 30),
        contribution_krw=1_000_000.0,
        hurdle=0.02,
        horizon_months=0,
        train_months=60,
        test_months=36,
        baseline=CandidateSpec(id="qqq_baseline", policy=PolicyId.QQQ, modules=0),
        candidates=[
            CandidateSpec(
                id="qqq_vti_mix",
                policy=PolicyId.QQQ,
                modules=1,
                targets={"QQQ": 0.8, "VTI": 0.2},
            )
        ],
    )
    runner = _MixConfigRunner()
    report = run_walk_forward_adoption(spec, runner)

    assert len(report.folds) > 0
    baseline_configs = [config for config in runner.configs if config.targets_override is None]
    candidate_configs = [
        config for config in runner.configs if config.targets_override == {"QQQ": 0.8, "VTI": 0.2}
    ]
    assert baseline_configs
    assert candidate_configs
    assert all(config.adaptive_contribution is None for config in runner.configs)


def test_wf_rejected_adaptive_baseline_keeps_full_identity() -> None:
    from datetime import date

    from src.policy.targets import PolicyId
    from src.sim.allocation import AllocationResult
    from src.validation.experiment import AdaptiveContributionSpec, CandidateSpec, ExperimentSpec
    from src.validation.walk_forward import run_walk_forward_adoption

    class _LosingAdaptiveRunner:
        def __init__(self) -> None:
            self.configs: list[AllocationConfig] = []

        def __call__(self, config: AllocationConfig) -> AllocationResult:
            self.configs.append(config)
            if config.adaptive_contribution is not None:
                wealth, contribution, xirr_real = 90.0, 90.0, 0.05
            else:
                wealth, contribution, xirr_real = 100.0, 90.0, 0.10
            return AllocationResult(
                config=config,
                snapshots=(),
                terminal_wealth_krw=wealth,
                xirr=0.0,
                max_drawdown=-0.25,
                terminal_wealth_real_krw=wealth,
                xirr_real=xirr_real,
                total_contribution_real_krw=contribution,
            )

    spec = ExperimentSpec(
        name="wf_adaptive_baseline_reject",
        start=date(2012, 4, 1),
        end=date(2024, 11, 30),
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        objective="adaptive_growth",
        train_months=60,
        test_months=36,
        adaptive_contribution=AdaptiveContributionSpec(),
        baseline_adaptive_contribution=AdaptiveContributionSpec(),
        baseline=CandidateSpec(
            id="qqq90_soxx10_adaptive_v5",
            policy=PolicyId.QQQ,
            modules=0,
            targets={"QQQ": 0.9, "SOXX": 0.1},
        ),
        candidates=[
            CandidateSpec(
                id="qqq90_soxx10_adaptive_challenger",
                policy=PolicyId.QQQ,
                modules=1,
                targets={"QQQ": 0.9, "SOXX": 0.1},
            )
        ],
    )
    runner = _LosingAdaptiveRunner()
    report = run_walk_forward_adoption(spec, runner)
    assert len(report.folds) > 0
    assert all(fold.train_adopted is False for fold in report.folds)
    assert len(runner.configs) == 4 * len(report.folds)
    for fold in report.folds:
        assert fold.chosen_policy is PolicyId.QQQ
        assert fold.chosen_test_wealth == fold.baseline_test_wealth
        assert fold.chosen_total_contribution_real_krw == fold.baseline_total_contribution_real_krw
        assert fold.chosen_xirr_real == fold.baseline_xirr_real
        assert fold.chosen_test_wealth == pytest.approx(90.0)
        assert fold.chosen_xirr_real == pytest.approx(0.05)
        assert fold.chosen_total_contribution_real_krw == pytest.approx(90.0)
    for fold_index in range(len(report.folds)):
        chunk = runner.configs[fold_index * 4 : (fold_index + 1) * 4]
        assert len(chunk) == 4
        assert all(cfg.adaptive_contribution is not None for cfg in chunk)


