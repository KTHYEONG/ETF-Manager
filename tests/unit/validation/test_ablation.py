"""Unit tests for the identical-cashflow M0/M1 ablation gate."""

from __future__ import annotations

from datetime import date

import pytest

from src.etf_manager.policy.targets import PolicyId
from src.etf_manager.sim.allocation import AllocationConfig, AllocationResult
from src.etf_manager.validation.ablation import run_ablation
from src.etf_manager.validation.experiment import (
    CadenceSpec,
    CandidateSpec,
    CurrencySpec,
    ExperimentSpec,
    MappingSpec,
    OverlaySpec,
    ReserveSpec,
)

_WINDOW = (date(2012, 1, 3), date(2024, 12, 31))


def _spec(modules: int) -> ExperimentSpec:
    return ExperimentSpec(
        name="abl_w1",
        start=_WINDOW[0],
        end=_WINDOW[1],
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        baseline=CandidateSpec(id="m0_global", policy="s0_global", modules=0),
        candidates=[CandidateSpec(id="m1_us", policy="s1_us", modules=modules)],
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


def _runner() -> _RecordingRunner:
    return _RecordingRunner({PolicyId.S0_GLOBAL: 100.0, PolicyId.S1_US: 110.0})


@pytest.mark.parametrize("scenario_id", ["ABL-W1-identical-cashflow-gate"])
def test_abl_w1_identical_cashflow_gate(scenario_id: str) -> None:
    """ABL-W1-identical-cashflow-gate"""
    runner = _runner()
    report = run_ablation(_spec(modules=1), runner)

    assert [config.policy for config in runner.configs] == [PolicyId.S0_GLOBAL, PolicyId.S1_US]
    for config in runner.configs:
        assert (config.start, config.end) == _WINDOW
        assert config.monthly_contribution_krw == pytest.approx(1_000_000.0)
        assert config.fill_delay_sessions == 1
        assert config.commission_bps == pytest.approx(0.0)
        assert config.fx_spread_bps == pytest.approx(0.0)
        assert config.tilt is None
        assert config.overlay is None
        assert config.currency is None
        assert config.mapping is None

    row = report.rows[0]
    assert row.candidate_id == "m1_us"
    assert row.wealths == (pytest.approx(110.0),)
    for gamma in (2.0, 5.0, 10.0):
        assert report.baseline_ce[gamma] == pytest.approx(100.0)
        assert row.ce[gamma] == pytest.approx(110.0)
        assert row.ce_ratio[gamma] == pytest.approx(1.10)
    assert row.adopted is True


@pytest.mark.parametrize("scenario_id", ["ABL-W1-identical-cashflow-gate"])
def test_abl_w1_complexity_hurdle_rejects(scenario_id: str) -> None:
    """ABL-W1-identical-cashflow-gate"""
    report = run_ablation(_spec(modules=10), _runner())

    row = report.rows[0]
    assert row.modules == 10
    assert row.ce_ratio[2.0] == pytest.approx(1.10)
    assert row.adopted is False


@pytest.mark.parametrize("scenario_id", ["ABL-W1-identical-cashflow-gate"])
def test_abl_w1_candidate_order(scenario_id: str) -> None:
    """ABL-W1-identical-cashflow-gate"""
    spec = ExperimentSpec(
        name="abl_order",
        start=_WINDOW[0],
        end=_WINDOW[1],
        contribution_krw=500_000.0,
        delta0=0.02,
        horizon_months=0,
        baseline=CandidateSpec(id="m0_global", policy="s0_global", modules=0),
        candidates=[
            CandidateSpec(id="c2", policy="s2_regional", modules=1),
            CandidateSpec(id="c4", policy="s4_defensive", modules=1),
            CandidateSpec(id="c1", policy="s1_us", modules=1),
        ],
    )
    runner = _RecordingRunner(dict.fromkeys(PolicyId, 100.0))

    report = run_ablation(spec, runner)

    assert [row.candidate_id for row in report.rows] == ["c2", "c4", "c1"]
    assert [config.policy for config in runner.configs] == [
        PolicyId.S0_GLOBAL,
        PolicyId.S2_REGIONAL,
        PolicyId.S4_DEFENSIVE,
        PolicyId.S1_US,
    ]


@pytest.mark.parametrize("scenario_id", ["ABL-G-overlay-candidate-only"])
def test_abl_g_overlay_candidate_only(scenario_id: str) -> None:
    """ABL-G-overlay-candidate-only"""
    spec = ExperimentSpec(
        name="abl_g_overlay",
        start=_WINDOW[0],
        end=_WINDOW[1],
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        overlay=OverlaySpec(max_shift=0.10),
        baseline=CandidateSpec(id="s1_us_base", policy="s1_us", modules=0),
        candidates=[CandidateSpec(id="s1_us_overlay", policy="s1_us", modules=1)],
    )
    runner = _RecordingRunner({PolicyId.S1_US: 120.0})

    run_ablation(spec, runner)

    assert len(runner.configs) == 2
    baseline_config, candidate_config = runner.configs
    assert baseline_config.overlay is None
    assert candidate_config.overlay is not None
    assert candidate_config.overlay.max_shift == pytest.approx(0.10)
    for config in (baseline_config, candidate_config):
        assert config.policy is PolicyId.S1_US
        assert (config.start, config.end) == _WINDOW
        assert config.monthly_contribution_krw == pytest.approx(1_000_000.0)
        assert config.fill_delay_sessions == 1

    plain_runner = _runner()
    run_ablation(_spec(modules=1), plain_runner)
    assert all(config.overlay is None for config in plain_runner.configs)


@pytest.mark.parametrize("scenario_id", ["ABL-H-reserve-candidate-only"])
def test_abl_h_reserve_candidate_only(scenario_id: str) -> None:
    """ABL-H-reserve-candidate-only"""
    spec = ExperimentSpec(
        name="abl_h_reserve",
        start=_WINDOW[0],
        end=_WINDOW[1],
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        reserve=ReserveSpec(max_withhold=0.10),
        baseline=CandidateSpec(id="s1_us_base", policy="s1_us", modules=0),
        candidates=[CandidateSpec(id="s1_us_reserve", policy="s1_us", modules=1)],
    )
    runner = _RecordingRunner({PolicyId.S1_US: 120.0})

    run_ablation(spec, runner)

    assert len(runner.configs) == 2
    baseline_config, candidate_config = runner.configs
    assert baseline_config.reserve is None
    assert candidate_config.reserve is not None
    assert candidate_config.reserve.max_withhold == pytest.approx(0.10)
    for config in (baseline_config, candidate_config):
        assert config.policy is PolicyId.S1_US
        assert (config.start, config.end) == _WINDOW
        assert config.monthly_contribution_krw == pytest.approx(1_000_000.0)
        assert config.fill_delay_sessions == 1

    plain_runner = _runner()
    run_ablation(_spec(modules=1), plain_runner)
    assert all(config.reserve is None for config in plain_runner.configs)


@pytest.mark.parametrize("scenario_id", ["ABL-J-mapping-candidate-only"])
def test_abl_j_mapping_candidate_only(scenario_id: str) -> None:
    """ABL-J-mapping-candidate-only"""
    spec = ExperimentSpec(
        name="abl_j_mapping",
        start=_WINDOW[0],
        end=_WINDOW[1],
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        mapping=MappingSpec(min_improvement=0.02),
        baseline=CandidateSpec(id="s1_us_base", policy="s1_us", modules=0),
        candidates=[CandidateSpec(id="s1_us_mapping", policy="s1_us", modules=1)],
    )
    runner = _RecordingRunner({PolicyId.S1_US: 120.0})

    run_ablation(spec, runner)

    assert len(runner.configs) == 2
    baseline_config, candidate_config = runner.configs
    assert baseline_config.mapping is None
    assert candidate_config.mapping is not None
    assert candidate_config.mapping.min_improvement == pytest.approx(0.02)
    for config in (baseline_config, candidate_config):
        assert config.policy is PolicyId.S1_US
        assert (config.start, config.end) == _WINDOW
        assert config.monthly_contribution_krw == pytest.approx(1_000_000.0)
        assert config.fill_delay_sessions == 1

    plain_runner = _runner()
    run_ablation(_spec(modules=1), plain_runner)
    assert all(config.mapping is None for config in plain_runner.configs)


@pytest.mark.parametrize("scenario_id", ["ABL-K-currency-candidate-only"])
def test_abl_k_currency_candidate_only(scenario_id: str) -> None:
    """ABL-K-currency-candidate-only"""
    spec = ExperimentSpec(
        name="abl_k_currency",
        start=_WINDOW[0],
        end=_WINDOW[1],
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        currency=CurrencySpec(max_defer=0.10),
        baseline=CandidateSpec(id="s1_us_base", policy="s1_us", modules=0),
        candidates=[CandidateSpec(id="s1_us_currency", policy="s1_us", modules=1)],
    )
    runner = _RecordingRunner({PolicyId.S1_US: 120.0})

    run_ablation(spec, runner)

    assert len(runner.configs) == 2
    baseline_config, candidate_config = runner.configs
    assert baseline_config.currency is None
    assert candidate_config.currency is not None
    assert candidate_config.currency.max_defer == pytest.approx(0.10)
    for config in (baseline_config, candidate_config):
        assert config.policy is PolicyId.S1_US
        assert (config.start, config.end) == _WINDOW
        assert config.monthly_contribution_krw == pytest.approx(1_000_000.0)
        assert config.fill_delay_sessions == 1
        assert config.commission_bps == pytest.approx(0.0)
        assert config.fx_spread_bps == pytest.approx(0.0)

    plain_runner = _runner()
    run_ablation(_spec(modules=1), plain_runner)
    assert all(config.currency is None for config in plain_runner.configs)


@pytest.mark.parametrize("scenario_id", ["ABL-L-cadence-candidate"])
def test_abl_l_cadence_candidate(scenario_id: str) -> None:
    """ABL-L-cadence-candidate"""
    spec = ExperimentSpec(
        name="abl_l_cadence",
        start=_WINDOW[0],
        end=_WINDOW[1],
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        cadence=CadenceSpec(anchor="month_open"),
        baseline=CandidateSpec(id="s1_us_base", policy="s1_us", modules=0),
        candidates=[CandidateSpec(id="s1_us_month_open", policy="s1_us", modules=1)],
    )
    runner = _RecordingRunner({PolicyId.S1_US: 120.0})

    run_ablation(spec, runner)

    assert len(runner.configs) == 2
    baseline_config, candidate_config = runner.configs
    assert baseline_config.cadence == "monthly"
    assert candidate_config.cadence == "month_open"
    for config in (baseline_config, candidate_config):
        # Identical cashflow once per calendar month; no other module rides along.
        assert config.overlay is None
        assert config.reserve is None
        assert config.mapping is None
        assert config.currency is None
        assert (config.start, config.end) == _WINDOW
        assert config.monthly_contribution_krw == pytest.approx(1_000_000.0)
        assert config.fill_delay_sessions == 1
