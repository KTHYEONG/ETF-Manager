"""Unit tests for the walk-forward adoption campaign."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from src.data.settings import DataSettings
from src.policy.adaptive_contribution import AdaptiveContributionConfig
from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig, AllocationResult
from src.validation.campaign import (
    COST_SCENARIOS,
    run_cadence_robustness,
    run_walk_forward_adoption,
    run_walk_forward_cost_grid,
    run_walk_forward_proxy_adoption,
    write_campaign_report,
)
from src.validation.experiment import (
    AdaptiveContributionSpec,
    CadenceSpec,
    CandidateSpec,
    CurrencySpec,
    ExperimentSpec,
    MappingSpec,
    OverlaySpec,
    ReserveSpec,
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


@pytest.mark.parametrize("scenario_id", ["WF-A-select-on-train"])
def test_wf_a_select_on_train(scenario_id: str) -> None:
    """WF-A-select-on-train"""
    runner = _RecordingRunner({PolicyId.VT: 100.0, PolicyId.VTI: 110.0})

    report = run_walk_forward_adoption(_spec(), runner)

    assert len(report.folds) > 0
    assert all(fold.train_adopted is True for fold in report.folds)
    assert all(fold.chosen_policy is PolicyId.VTI for fold in report.folds)
    assert report.process_adopted_vs_baseline is True

    first_train_pair = runner.configs[:2]
    for config in first_train_pair:
        assert config.start == date(2012, 4, 1)
        assert config.monthly_contribution_krw == pytest.approx(1_000_000.0)
        assert config.fill_delay_sessions == 1
        assert config.commission_bps == pytest.approx(0.0)
        assert config.tilt is None
    assert [config.policy for config in first_train_pair] == [PolicyId.VT, PolicyId.VTI]
    # Baseline, candidate, and chosen all run on test even when chosen repeats an arm.
    assert len(runner.configs) == 5 * len(report.folds)


@pytest.mark.parametrize("scenario_id", ["WF-A-keep-baseline"])
def test_wf_a_keep_baseline(scenario_id: str) -> None:
    """WF-A-keep-baseline"""
    runner = _RecordingRunner({PolicyId.VT: 100.0, PolicyId.VTI: 100.0})

    report = run_walk_forward_adoption(_spec(), runner)

    assert len(report.folds) > 0
    assert all(fold.train_adopted is False for fold in report.folds)
    assert all(fold.chosen_policy is PolicyId.VT for fold in report.folds)
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
        wealth = 120.0 if config.policy is PolicyId.VTI else 100.0
        if config.policy is PolicyId.VTI and config.commission_bps >= 50.0:
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
    runner = _RecordingRunner({PolicyId.VT: 100.0, PolicyId.VTI: 110.0})
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


def _proxy_spec() -> ExperimentSpec:
    """Wave C contract: S0 ETF baseline versus the R1 research-proxy candidate."""
    return ExperimentSpec(
        name="wf_s0_r1",
        start=date(2012, 4, 1),
        end=date(2024, 11, 30),
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        train_months=60,
        test_months=36,
        baseline=CandidateSpec(id="s0_global", policy="s0_global", modules=0),
        candidates=[CandidateSpec(id="r1_us_mkt_ff", policy="r1_us_mkt_ff", modules=1)],
    )


@pytest.mark.parametrize("scenario_id", ["WF-C-proxy-vs-baseline-gate"])
def test_wf_c_proxy_vs_baseline_gate(scenario_id: str) -> None:
    """WF-C-proxy-vs-baseline-gate"""
    etf_runner = _RecordingRunner({PolicyId.VT: 100.0})
    proxy_runner = _RecordingRunner({PolicyId.FF_PROXY: 110.0})

    report = run_walk_forward_proxy_adoption(_proxy_spec(), etf_runner, proxy_runner)

    assert len(report.folds) > 0
    assert report.process_adopted_vs_baseline is True
    assert etf_runner.configs
    assert all(config.policy is PolicyId.VT for config in etf_runner.configs)
    assert proxy_runner.configs
    assert all(config.policy is PolicyId.FF_PROXY for config in proxy_runner.configs)

    equal = run_walk_forward_proxy_adoption(
        _proxy_spec(),
        _RecordingRunner({PolicyId.VT: 100.0}),
        _RecordingRunner({PolicyId.FF_PROXY: 100.0}),
    )
    # ratio == 1.0 fails the strict > 1 + delta0 * modules hurdle.
    assert equal.process_adopted_vs_baseline is False


@pytest.mark.parametrize("scenario_id", ["WF-C-reject-costs-and-etf-candidate"])
def test_wf_c_reject_costs_and_etf_candidate(scenario_id: str) -> None:
    """WF-C-reject-costs-and-etf-candidate"""
    etf_runner = _RecordingRunner({PolicyId.VT: 100.0})
    proxy_runner = _RecordingRunner({PolicyId.FF_PROXY: 110.0})

    commission = _proxy_spec().model_copy(update={"commission_bps": 10.0})
    with pytest.raises(ValueError, match="commission_bps"):
        run_walk_forward_proxy_adoption(commission, etf_runner, proxy_runner)

    spread = _proxy_spec().model_copy(update={"fx_spread_bps": 10.0})
    with pytest.raises(ValueError, match="fx_spread_bps"):
        run_walk_forward_proxy_adoption(spread, etf_runner, proxy_runner)

    etf_candidate = _proxy_spec().model_copy(
        update={"candidates": [CandidateSpec(id="s1_us", policy="s1_us", modules=1)]}
    )
    with pytest.raises(ValueError, match=r"FF_PROXY|candidate"):
        run_walk_forward_proxy_adoption(etf_candidate, etf_runner, proxy_runner)

    proxy_baseline = _proxy_spec().model_copy(
        update={"baseline": CandidateSpec(id="r1_us_mkt_ff", policy="r1_us_mkt_ff", modules=0)}
    )
    with pytest.raises(ValueError, match="baseline"):
        run_walk_forward_proxy_adoption(proxy_baseline, etf_runner, proxy_runner)


@pytest.mark.parametrize("scenario_id", ["WF-B-grid-four-scenarios"])
def test_wf_b_grid_four_scenarios(scenario_id: str) -> None:
    """WF-B-grid-four-scenarios"""
    runner = _RecordingRunner({PolicyId.VT: 100.0, PolicyId.VTI: 120.0})
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


class _OverlayWealthRunner:
    """Records configs; overlay arms outperform to force train adoption."""

    def __init__(self) -> None:
        self.configs: list[AllocationConfig] = []

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        self.configs.append(config)
        wealth = 120.0 if config.overlay is not None else 100.0
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )


@pytest.mark.parametrize("scenario_id", ["WF-G-overlay-same-policy"])
def test_wf_g_overlay_same_policy(scenario_id: str) -> None:
    """WF-G-overlay-same-policy"""
    spec = ExperimentSpec(
        name="wf_g_overlay",
        start=date(2012, 4, 1),
        end=date(2024, 11, 30),
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        train_months=60,
        test_months=36,
        baseline=CandidateSpec(id="s1_us_base", policy="s1_us", modules=0),
        candidates=[CandidateSpec(id="s1_us_overlay", policy="s1_us", modules=1)],
        overlay=OverlaySpec(max_shift=0.10),
    )
    runner = _OverlayWealthRunner()

    report = run_walk_forward_adoption(spec, runner)

    assert len(report.folds) > 0
    assert all(fold.train_adopted is True for fold in report.folds)
    assert report.process_adopted_vs_baseline is True
    for fold_index in range(len(report.folds)):
        chunk = runner.configs[fold_index * 5 : (fold_index + 1) * 5]
        assert len(chunk) == 5
        overlay_states = [config.overlay is not None for config in chunk]
        # Baseline train/test stay un-overlayed; candidate and chosen arms carry it.
        assert overlay_states == [False, True, False, True, True]
        for config in chunk:
            if config.overlay is not None:
                assert config.overlay.max_shift == pytest.approx(0.10)
            assert config.policy is PolicyId.VTI
            assert config.fill_delay_sessions == 1

    etf_runner = _RecordingRunner(dict.fromkeys(PolicyId, 100.0))
    proxy_runner = _RecordingRunner(dict.fromkeys(PolicyId, 100.0))
    with pytest.raises(ValueError, match="overlay"):
        run_walk_forward_proxy_adoption(spec, etf_runner, proxy_runner)


class _ReserveWealthRunner:
    """Records configs; reserve arms outperform to force train adoption."""

    def __init__(self) -> None:
        self.configs: list[AllocationConfig] = []

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        self.configs.append(config)
        wealth = 120.0 if config.reserve is not None else 100.0
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )


@pytest.mark.parametrize("scenario_id", ["WF-H-reserve-same-policy"])
def test_wf_h_reserve_same_policy(scenario_id: str) -> None:
    """WF-H-reserve-same-policy"""
    spec = ExperimentSpec(
        name="wf_h_reserve",
        start=date(2012, 4, 1),
        end=date(2024, 11, 30),
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        train_months=60,
        test_months=36,
        baseline=CandidateSpec(id="s1_us_base", policy="s1_us", modules=0),
        candidates=[CandidateSpec(id="s1_us_reserve", policy="s1_us", modules=1)],
        reserve=ReserveSpec(max_withhold=0.10),
    )
    runner = _ReserveWealthRunner()

    report = run_walk_forward_adoption(spec, runner)

    assert len(report.folds) > 0
    assert all(fold.train_adopted is True for fold in report.folds)
    assert report.process_adopted_vs_baseline is True
    for fold_index in range(len(report.folds)):
        chunk = runner.configs[fold_index * 5 : (fold_index + 1) * 5]
        assert len(chunk) == 5
        reserve_states = [config.reserve is not None for config in chunk]
        # Baseline train/test stay un-reserved; candidate and chosen arms carry it.
        assert reserve_states == [False, True, False, True, True]
        for config in chunk:
            if config.reserve is not None:
                assert config.reserve.max_withhold == pytest.approx(0.10)
            assert config.policy is PolicyId.VTI
            assert config.fill_delay_sessions == 1

    proxy_reject = spec.model_copy(update={"reserve": ReserveSpec(max_withhold=0.05)})
    etf_runner = _RecordingRunner(dict.fromkeys(PolicyId, 100.0))
    proxy_runner = _RecordingRunner(dict.fromkeys(PolicyId, 100.0))
    with pytest.raises(ValueError, match="reserve"):
        run_walk_forward_proxy_adoption(proxy_reject, etf_runner, proxy_runner)


class _MappingWealthRunner:
    """Records configs; mapping arms outperform to force train adoption."""

    def __init__(self) -> None:
        self.configs: list[AllocationConfig] = []

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        self.configs.append(config)
        wealth = 120.0 if config.mapping is not None else 100.0
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )


@pytest.mark.parametrize("scenario_id", ["WF-J-mapping-same-policy"])
def test_wf_j_mapping_same_policy(scenario_id: str) -> None:
    """WF-J-mapping-same-policy"""
    spec = ExperimentSpec(
        name="wf_j_mapping",
        start=date(2012, 4, 1),
        end=date(2024, 11, 30),
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        train_months=60,
        test_months=36,
        baseline=CandidateSpec(id="s1_us_base", policy="s1_us", modules=0),
        candidates=[CandidateSpec(id="s1_us_mapping", policy="s1_us", modules=1)],
        mapping=MappingSpec(min_improvement=0.02),
    )
    runner = _MappingWealthRunner()

    report = run_walk_forward_adoption(spec, runner)

    assert len(report.folds) > 0
    assert all(fold.train_adopted is True for fold in report.folds)
    assert report.process_adopted_vs_baseline is True
    for fold_index in range(len(report.folds)):
        chunk = runner.configs[fold_index * 5 : (fold_index + 1) * 5]
        assert len(chunk) == 5
        mapping_states = [config.mapping is not None for config in chunk]
        # Baseline train/test stay unmapped; candidate and chosen arms carry it.
        assert mapping_states == [False, True, False, True, True]
        for config in chunk:
            if config.mapping is not None:
                assert config.mapping.min_improvement == pytest.approx(0.02)
            assert config.policy is PolicyId.VTI
            assert config.fill_delay_sessions == 1

    etf_runner = _RecordingRunner(dict.fromkeys(PolicyId, 100.0))
    proxy_runner = _RecordingRunner(dict.fromkeys(PolicyId, 100.0))
    with pytest.raises(ValueError, match="mapping"):
        run_walk_forward_proxy_adoption(spec, etf_runner, proxy_runner)


def _currency_spec() -> ExperimentSpec:
    """S1 versus S1+currency on the shared walk-forward window."""
    return ExperimentSpec(
        name="wf_k_currency",
        start=date(2012, 4, 1),
        end=date(2024, 11, 30),
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        train_months=60,
        test_months=36,
        baseline=CandidateSpec(id="s1_us_base", policy="s1_us", modules=0),
        candidates=[CandidateSpec(id="s1_us_currency", policy="s1_us", modules=1)],
        currency=CurrencySpec(max_defer=0.10),
    )


class _CurrencyWealthRunner:
    """Records configs; currency arms outperform to force train adoption."""

    def __init__(self) -> None:
        self.configs: list[AllocationConfig] = []

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        self.configs.append(config)
        wealth = 120.0 if config.currency is not None else 100.0
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )


@pytest.mark.parametrize("scenario_id", ["WF-K-currency-same-policy"])
def test_wf_k_currency_same_policy(scenario_id: str) -> None:
    """WF-K-currency-same-policy"""
    runner = _CurrencyWealthRunner()

    report = run_walk_forward_adoption(_currency_spec(), runner)

    assert len(report.folds) > 0
    assert all(fold.train_adopted is True for fold in report.folds)
    assert report.process_adopted_vs_baseline is True
    for fold_index in range(len(report.folds)):
        chunk = runner.configs[fold_index * 5 : (fold_index + 1) * 5]
        assert len(chunk) == 5
        currency_states = [config.currency is not None for config in chunk]
        # Baseline train/test stay undeferred; candidate and chosen arms carry it.
        assert currency_states == [False, True, False, True, True]
        for config in chunk:
            if config.currency is not None:
                assert config.currency.max_defer == pytest.approx(0.10)
            assert config.policy is PolicyId.VTI
            assert config.fill_delay_sessions == 1


@pytest.mark.parametrize("scenario_id", ["WF-K-currency-cost-grid"])
def test_wf_k_currency_cost_grid(scenario_id: str) -> None:
    """WF-K-currency-cost-grid"""
    runner = _CurrencyWealthRunner()

    grid = run_walk_forward_cost_grid(_currency_spec(), runner)

    assert grid.all_scenarios_adopted is True
    fold_count = len(grid.outcomes[0].campaign.folds)
    cursor = 0
    for outcome in grid.outcomes:
        block = runner.configs[cursor : cursor + 5 * fold_count]
        cursor += len(block)
        assert len(block) == 5 * fold_count
        for index, config in enumerate(block):
            assert config.commission_bps == pytest.approx(outcome.scenario.commission_bps)
            assert config.fx_spread_bps == pytest.approx(outcome.scenario.fx_spread_bps)
            carries_currency = index % 5 in (1, 3, 4)
            assert (config.currency is not None) is carries_currency
            if config.currency is not None:
                assert config.currency.max_defer == pytest.approx(0.10)
    assert cursor == len(runner.configs)


@pytest.mark.parametrize("scenario_id", ["WF-K-proxy-rejects-currency"])
def test_wf_k_proxy_rejects_currency(scenario_id: str) -> None:
    """WF-K-proxy-rejects-currency"""
    etf_runner = _RecordingRunner(dict.fromkeys(PolicyId, 100.0))
    proxy_runner = _RecordingRunner(dict.fromkeys(PolicyId, 100.0))

    with pytest.raises(ValueError, match="currency"):
        run_walk_forward_proxy_adoption(_currency_spec(), etf_runner, proxy_runner)


class _CadenceWealthRunner:
    """Records configs; month-open arms outperform to force train adoption."""

    def __init__(self) -> None:
        self.configs: list[AllocationConfig] = []

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        self.configs.append(config)
        wealth = 120.0 if config.cadence == "month_open" else 100.0
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )


def _cadence_spec() -> ExperimentSpec:
    """S1 month-end baseline versus S1 month-open candidate on the shared window."""
    return ExperimentSpec(
        name="wf_l_cadence",
        start=date(2012, 4, 1),
        end=date(2024, 11, 30),
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        train_months=60,
        test_months=36,
        cadence=CadenceSpec(anchor="month_open"),
        baseline=CandidateSpec(id="s1_us_base", policy="s1_us", modules=0),
        candidates=[CandidateSpec(id="s1_us_month_open", policy="s1_us", modules=1)],
    )


@pytest.mark.parametrize("scenario_id", ["WF-L-cadence-candidate"])
def test_wf_l_cadence_candidate(scenario_id: str) -> None:
    """WF-L-cadence-candidate"""
    runner = _CadenceWealthRunner()

    report = run_walk_forward_adoption(_cadence_spec(), runner)

    assert len(report.folds) > 0
    assert all(fold.train_adopted is True for fold in report.folds)
    assert report.process_adopted_vs_baseline is True
    for fold_index in range(len(report.folds)):
        chunk = runner.configs[fold_index * 5 : (fold_index + 1) * 5]
        assert len(chunk) == 5
        # Baseline train/test stay monthly; the adopted chosen arm opens the month.
        assert [config.cadence for config in chunk] == [
            "monthly",
            "month_open",
            "monthly",
            "month_open",
            "month_open",
        ]

    flat_runner = _RecordingRunner({PolicyId.VT: 100.0, PolicyId.VTI: 100.0})
    flat_report = run_walk_forward_adoption(_cadence_spec(), flat_runner)

    assert all(fold.train_adopted is False for fold in flat_report.folds)
    for fold_index in range(len(flat_report.folds)):
        chunk = flat_runner.configs[fold_index * 5 : (fold_index + 1) * 5]
        assert len(chunk) == 5
        # Without train adoption the chosen arm falls back to the monthly cadence.
        assert [config.cadence for config in chunk] == [
            "monthly",
            "month_open",
            "monthly",
            "month_open",
            "monthly",
        ]

    etf_runner = _RecordingRunner(dict.fromkeys(PolicyId, 100.0))
    proxy_runner = _RecordingRunner(dict.fromkeys(PolicyId, 100.0))
    with pytest.raises(ValueError, match="cadence"):
        run_walk_forward_proxy_adoption(_cadence_spec(), etf_runner, proxy_runner)


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
        chunk = runner.configs[fold_index * 5 : (fold_index + 1) * 5]
        assert len(chunk) == 5
        adaptive_states = [config.adaptive_contribution is not None for config in chunk]
        # Baseline train/test stay flat; candidate and adopted chosen arms carry the module.
        assert adaptive_states == [False, True, False, True, True]
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
        chunk = losing.configs[fold_index * 5 : (fold_index + 1) * 5]
        adaptive_states = [config.adaptive_contribution is not None for config in chunk]
        # Without train adoption the chosen arm falls back to the flat baseline.
        assert adaptive_states == [False, True, False, True, False]

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
        chunk = runner.configs[fold_index * 5 : (fold_index + 1) * 5]
        # Baseline train/test carry the locked adaptive config; candidate keeps its own.
        assert chunk[0].adaptive_contribution == resolved_baseline
        assert chunk[2].adaptive_contribution == resolved_baseline
        assert chunk[1].adaptive_contribution == resolve_adaptive_contribution(baseline_adaptive_spec)

    flat_runner = _AdaptiveWealthRunner()
    run_walk_forward_adoption(_acg_spec(), flat_runner)
    for fold_index in range(len(flat_runner.configs) // 5):
        flat_chunk = flat_runner.configs[fold_index * 5 : (fold_index + 1) * 5]
        assert flat_chunk[0].adaptive_contribution is None
        assert flat_chunk[2].adaptive_contribution is None

