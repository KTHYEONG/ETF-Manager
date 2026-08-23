"""Unit tests for the walk-forward adoption campaign."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from src.etf_manager.policy.targets import PolicyId
from src.etf_manager.sim.allocation import AllocationConfig, AllocationResult
from src.etf_manager.validation.campaign import (
    COST_SCENARIOS,
    run_walk_forward_adoption,
    run_walk_forward_cost_grid,
)
from src.etf_manager.validation.experiment import CandidateSpec, ExperimentSpec


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


@pytest.mark.parametrize("scenario_id", ["WF-A-select-on-train"])
def test_wf_a_select_on_train(scenario_id: str) -> None:
    """WF-A-select-on-train"""
    runner = _RecordingRunner({PolicyId.S0_GLOBAL: 100.0, PolicyId.S1_US: 110.0})

    report = run_walk_forward_adoption(_spec(), runner)

    assert len(report.folds) > 0
    assert all(fold.train_adopted is True for fold in report.folds)
    assert all(fold.chosen_policy is PolicyId.S1_US for fold in report.folds)
    assert report.process_adopted_vs_baseline is True

    first_train_pair = runner.configs[:2]
    for config in first_train_pair:
        assert config.start == date(2012, 4, 1)
        assert config.monthly_contribution_krw == pytest.approx(1_000_000.0)
        assert config.fill_delay_sessions == 1
        assert config.commission_bps == pytest.approx(0.0)
        assert config.tilt is None
    assert [config.policy for config in first_train_pair] == [PolicyId.S0_GLOBAL, PolicyId.S1_US]
    # Baseline, candidate, and chosen all run on test even when chosen repeats an arm.
    assert len(runner.configs) == 5 * len(report.folds)


@pytest.mark.parametrize("scenario_id", ["WF-A-keep-baseline"])
def test_wf_a_keep_baseline(scenario_id: str) -> None:
    """WF-A-keep-baseline"""
    runner = _RecordingRunner({PolicyId.S0_GLOBAL: 100.0, PolicyId.S1_US: 100.0})

    report = run_walk_forward_adoption(_spec(), runner)

    assert len(report.folds) > 0
    assert all(fold.train_adopted is False for fold in report.folds)
    assert all(fold.chosen_policy is PolicyId.S0_GLOBAL for fold in report.folds)
    # ratio == 1.0 fails the strict > 1 + delta0 * modules hurdle.
    assert report.process_adopted_vs_baseline is False


@pytest.mark.parametrize("scenario_id", ["WF-A-select-on-train"])
def test_wf_a_rejects_invalid_spec(scenario_id: str) -> None:
    """WF-A-select-on-train"""
    with pytest.raises(ValueError, match="train_months"):
        run_walk_forward_adoption(
            ExperimentSpec(
                name="wf_no_months",
                start=date(2012, 4, 1),
                end=date(2024, 11, 30),
                contribution_krw=1_000_000.0,
                delta0=0.02,
                horizon_months=0,
                baseline=CandidateSpec(id="s0_global", policy="s0_global", modules=0),
                candidates=[CandidateSpec(id="s1_us", policy="s1_us", modules=1)],
            ),
            _RecordingRunner(dict.fromkeys(PolicyId, 100.0)),
        )

    multi_candidate = _spec().model_copy(
        update={
            "candidates": [
                CandidateSpec(id="s1_us", policy="s1_us", modules=1),
                CandidateSpec(id="s2_regional", policy="s2_regional", modules=1),
            ]
        }
    )
    with pytest.raises(ValueError, match="exactly one candidate"):
        run_walk_forward_adoption(multi_candidate, _RecordingRunner(dict.fromkeys(PolicyId, 100.0)))

    tiny_window = _spec().model_copy(update={"end": date(2012, 5, 1)})
    with pytest.raises(ValueError, match="no walk-forward folds"):
        run_walk_forward_adoption(tiny_window, _RecordingRunner(dict.fromkeys(PolicyId, 100.0)))


class _StressCostRunner:
    """Candidate arm collapses only once commissions reach the stress level."""

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        wealth = 120.0 if config.policy is PolicyId.S1_US else 100.0
        if config.policy is PolicyId.S1_US and config.commission_bps >= 50.0:
            wealth = 90.0
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )


@pytest.mark.parametrize("scenario_id", ["WF-B-spec-costs-injected"])
def test_wf_b_spec_costs_injected(scenario_id: str) -> None:
    """WF-B-spec-costs-injected"""
    runner = _RecordingRunner({PolicyId.S0_GLOBAL: 100.0, PolicyId.S1_US: 110.0})
    spec = _spec().model_copy(update={"commission_bps": 10.0, "fx_spread_bps": 20.0})

    report = run_walk_forward_adoption(spec, runner)

    assert len(report.folds) > 0
    for config in runner.configs:
        assert config.commission_bps == pytest.approx(10.0)
        assert config.fx_spread_bps == pytest.approx(20.0)
        assert config.fill_delay_sessions == 1
        assert config.tilt is None
        assert config.overlay is None
        assert config.currency is None
        assert config.mapping is None
    assert report.process_adopted_vs_baseline is True


@pytest.mark.parametrize("scenario_id", ["WF-B-grid-four-scenarios"])
def test_wf_b_grid_four_scenarios(scenario_id: str) -> None:
    """WF-B-grid-four-scenarios"""
    runner = _RecordingRunner({PolicyId.S0_GLOBAL: 100.0, PolicyId.S1_US: 120.0})
    spec = _spec()

    grid = run_walk_forward_cost_grid(spec, runner)

    assert [outcome.scenario.id for outcome in grid.outcomes] == ["ideal", "low", "base", "stress"]
    expected_bps = ((0.0, 0.0), (5.0, 10.0), (10.0, 20.0), (50.0, 50.0))
    for outcome, (commission_bps, fx_spread_bps) in zip(grid.outcomes, expected_bps, strict=True):
        assert outcome.scenario.commission_bps == pytest.approx(commission_bps)
        assert outcome.scenario.fx_spread_bps == pytest.approx(fx_spread_bps)
    with pytest.raises(FrozenInstanceError):
        COST_SCENARIOS[0].id = "mutated"  # type: ignore[misc]

    fold_count = len(grid.outcomes[0].campaign.folds)
    configs_per_scenario = 5 * fold_count
    for index, outcome in enumerate(grid.outcomes):
        chunk = runner.configs[index * configs_per_scenario : (index + 1) * configs_per_scenario]
        assert chunk
        for config in chunk:
            assert config.commission_bps == pytest.approx(outcome.scenario.commission_bps)
            assert config.fx_spread_bps == pytest.approx(outcome.scenario.fx_spread_bps)
    assert grid.all_scenarios_adopted is True
    assert spec.commission_bps == pytest.approx(0.0)
    assert spec.fx_spread_bps == pytest.approx(0.0)


@pytest.mark.parametrize("scenario_id", ["WF-B-grid-stress-can-flip"])
def test_wf_b_grid_stress_can_flip(scenario_id: str) -> None:
    """WF-B-grid-stress-can-flip"""
    grid = run_walk_forward_cost_grid(_spec(), _StressCostRunner())

    assert grid.all_scenarios_adopted is False
    ideal = grid.outcomes[0]
    assert ideal.scenario.id == "ideal"
    assert ideal.campaign.process_adopted_vs_baseline is True
    stress = grid.outcomes[-1]
    assert stress.scenario.id == "stress"
    assert stress.campaign.process_adopted_vs_baseline is False
