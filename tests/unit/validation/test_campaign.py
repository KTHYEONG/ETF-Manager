"""Unit tests for the walk-forward adoption campaign."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig, AllocationResult
from src.validation.campaign import (
    COST_SCENARIOS,
    run_walk_forward_adoption,
    run_walk_forward_cost_grid,
    run_walk_forward_proxy_adoption,
)
from src.validation.experiment import (
    CadenceSpec,
    CandidateSpec,
    CurrencySpec,
    ExperimentSpec,
    MappingSpec,
    OverlaySpec,
    ReserveSpec,
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
