"""Unit tests for the identical-cashflow M0/M1 ablation gate."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig, AllocationResult
from src.validation.ablation import run_ablation
from src.validation.experiment import (
    CadenceSpec,
    CandidateSpec,
    CurrencySpec,
    ExperimentSpec,
    MappingSpec,
    OverlaySpec,
    ReserveSpec,
    assert_experiment_preregistration,
    load_experiment_config,
)
from src.policy.thesis import load_thesis_registry

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
    return _RecordingRunner({PolicyId.VT: 100.0, PolicyId.VTI: 110.0})


@pytest.mark.parametrize("scenario_id", ["ABL-W1-identical-cashflow-gate"])
def test_abl_w1_identical_cashflow_gate(scenario_id: str) -> None:
    """ABL-W1-identical-cashflow-gate"""
    runner = _runner()
    report = run_ablation(_spec(modules=1), runner)

    assert [config.policy for config in runner.configs] == [PolicyId.VT, PolicyId.VTI]
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
        PolicyId.VT,
        PolicyId.WORLD_SPLIT,
        PolicyId.VT_TREAS,
        PolicyId.VTI,
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
    runner = _RecordingRunner({PolicyId.VTI: 120.0})

    run_ablation(spec, runner)

    assert len(runner.configs) == 2
    baseline_config, candidate_config = runner.configs
    assert baseline_config.overlay is None
    assert candidate_config.overlay is not None
    assert candidate_config.overlay.max_shift == pytest.approx(0.10)
    for config in (baseline_config, candidate_config):
        assert config.policy is PolicyId.VTI
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
    runner = _RecordingRunner({PolicyId.VTI: 120.0})

    run_ablation(spec, runner)

    assert len(runner.configs) == 2
    baseline_config, candidate_config = runner.configs
    assert baseline_config.reserve is None
    assert candidate_config.reserve is not None
    assert candidate_config.reserve.max_withhold == pytest.approx(0.10)
    for config in (baseline_config, candidate_config):
        assert config.policy is PolicyId.VTI
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
    runner = _RecordingRunner({PolicyId.VTI: 120.0})

    run_ablation(spec, runner)

    assert len(runner.configs) == 2
    baseline_config, candidate_config = runner.configs
    assert baseline_config.mapping is None
    assert candidate_config.mapping is not None
    assert candidate_config.mapping.min_improvement == pytest.approx(0.02)
    for config in (baseline_config, candidate_config):
        assert config.policy is PolicyId.VTI
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
    runner = _RecordingRunner({PolicyId.VTI: 120.0})

    run_ablation(spec, runner)

    assert len(runner.configs) == 2
    baseline_config, candidate_config = runner.configs
    assert baseline_config.currency is None
    assert candidate_config.currency is not None
    assert candidate_config.currency.max_defer == pytest.approx(0.10)
    for config in (baseline_config, candidate_config):
        assert config.policy is PolicyId.VTI
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
    runner = _RecordingRunner({PolicyId.VTI: 120.0})

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


@pytest.mark.parametrize("scenario_id", ["ABL-ACG-identical-cashflow-reject"])
def test_abl_acg_identical_cashflow_reject(scenario_id: str) -> None:
    """ABL-ACG-identical-cashflow-reject"""
    spec = ExperimentSpec(
        name="abl_acg",
        start=_WINDOW[0],
        end=_WINDOW[1],
        contribution_krw=1_000_000.0,
        delta0=0.02,
        horizon_months=0,
        objective="adaptive_growth",
        adaptive_contribution={},
        baseline=CandidateSpec(id="m0_global", policy="s0_global", modules=0),
        candidates=[CandidateSpec(id="m1_us", policy="s1_us", modules=1)],
    )
    runner = _runner()

    with pytest.raises(ValueError, match="identical cashflow"):
        run_ablation(spec, runner)

    # The rejection happens before the runner is ever invoked.
    assert runner.configs == []


class _MixWealthRunner:
    """Wealth keyed by whether an arm carries a static mix override."""

    def __init__(self) -> None:
        self.configs: list[AllocationConfig] = []

    def __call__(self, config: AllocationConfig) -> AllocationResult:
        self.configs.append(config)
        wealth = 110.0 if config.targets_override == {"QQQ": 0.9, "VTI": 0.1} else 100.0
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )


@pytest.mark.parametrize("scenario_id", ["ABL-MIX-override-wired"])
def test_abl_mix_override_wired(scenario_id: str) -> None:
    """ABL-MIX-override-wired"""
    spec = ExperimentSpec(
        name="abl_mix",
        start=_WINDOW[0],
        end=_WINDOW[1],
        contribution_krw=1_000_000.0,
        hurdle=0.02,
        horizon_months=0,
        baseline=CandidateSpec(id="qqq_baseline", policy=PolicyId.QQQ, modules=0),
        candidates=[
            CandidateSpec(
                id="qqq_vti_mix",
                policy=PolicyId.QQQ,
                modules=1,
                targets={"QQQ": 0.9, "VTI": 0.1},
            )
        ],
    )
    runner = _MixWealthRunner()
    report = run_ablation(spec, runner)

    assert len(runner.configs) == 2
    assert runner.configs[0].targets_override is None
    assert runner.configs[1].targets_override == {"QQQ": 0.9, "VTI": 0.1}
    for config in runner.configs:
        assert config.monthly_contribution_krw == pytest.approx(1_000_000.0)
        assert config.adaptive_contribution is None
    assert report.rows[0].adopted is True


@pytest.mark.parametrize("scenario_id", ["ABL-THESIS-prereg-wired"])
def test_abl_thesis_prereg_wired(scenario_id: str) -> None:
    """ABL-THESIS-prereg-wired"""
    spec = load_experiment_config("configs/experiments/m_thesis_ai_compute_soxx.json")
    registry = load_thesis_registry(Path("configs/theses"))
    assert_experiment_preregistration(spec, registry)

    runner = _MixWealthRunner()
    report = run_ablation(spec, runner)
    assert len(report.rows) == 1
    assert report.rows[0].candidate_id == "soxx_100"

@pytest.mark.parametrize("scenario_id", ["ABL-LH-adopted-from-cohort"])
def test_abl_lh_adopted_from_cohort(scenario_id: str) -> None:
    from src.validation.accumulation_cohort import AccumulationCohortReport, AccumulationCohortRow, CohortOverlapMetadata
    # Mock runners for ablation with long_horizon objective
    # Need experiment spec with objective long_horizon
    spec = ExperimentSpec(
        name="abl_lh",
        start=_WINDOW[0],
        end=_WINDOW[1],
        contribution_krw=1_000_000.0,
        hurdle=0.02,
        horizon_months=120,
        objective="long_horizon",
        baseline=CandidateSpec(id="qqq_baseline", policy="s0_global", modules=0, targets={"QQQ": 1.0}),
        candidates=[CandidateSpec(id="soxx_100", policy="qqq", modules=1, targets={"SOXX": 1.0})],
    )
    # Patch cohort report to control median and count
    from unittest.mock import patch

    # Case 1: should adopt when passes True (cohort 10 median 1.03) regardless of CE ratio below 1.02
    def fake_pass_report(spec, runner, horizon_months=120, step_months=12, bootstrap_paths=4000, seed=7):
        overlap = CohortOverlapMetadata(horizon_months=120, step_months=12)
        rows = tuple(AccumulationCohortRow(candidate_wealth=103, baseline_wealth=100, ratio=1.03, candidate_recovery_months=0) for _ in range(10))
        return AccumulationCohortReport(name=spec.name, overlap=overlap, rows=rows, median_ratio=1.03, p10_ratio=1.0, worst_ratio=0.99, win_rate=1.0, bootstrap_p05_ratio_mean=1.0, unrecovered_cohort_count=0)

    def runner_low_ce(config):
        # CE ratio will be ~0.99 (below 1.02) but long horizon passes
        # ablation still computes CE ratios; we set wealths to give low ce
        # baseline wealth 100, candidate 99.5 => ce ratio 0.995
        wealth = 99.5 if config.targets_override == {"SOXX": 1.0} else 100.0
        return AllocationResult(config=config, snapshots=(), terminal_wealth_krw=wealth, xirr=0.0, max_drawdown=0.0, terminal_wealth_real_krw=wealth, xirr_real=0.0)

    with patch("src.validation.accumulation_cohort.run_accumulation_cohort_report", side_effect=fake_pass_report):
        report = run_ablation(spec, runner_low_ce)
        assert report.rows[0].adopted is True
        # CE ratios still stored
        assert report.rows[0].ce_ratio[2.0] == pytest.approx(0.995)

    # Case 2: fail when cohort count below threshold
    def fake_fail_report(spec, runner, horizon_months=120, step_months=12, bootstrap_paths=4000, seed=7):
        overlap = CohortOverlapMetadata(horizon_months=120, step_months=12)
        rows = tuple(AccumulationCohortRow(candidate_wealth=103, baseline_wealth=100, ratio=1.03, candidate_recovery_months=0) for _ in range(9))
        return AccumulationCohortReport(name=spec.name, overlap=overlap, rows=rows, median_ratio=1.03, p10_ratio=1.0, worst_ratio=0.99, win_rate=1.0, bootstrap_p05_ratio_mean=1.0, unrecovered_cohort_count=0)

    with patch("src.validation.accumulation_cohort.run_accumulation_cohort_report", side_effect=fake_fail_report):
        report2 = run_ablation(spec, runner_low_ce)
        assert report2.rows[0].adopted is False
