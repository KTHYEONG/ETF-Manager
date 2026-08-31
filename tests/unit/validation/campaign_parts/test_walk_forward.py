# ruff: noqa: F401, F811
"""Unit tests for the walk-forward adoption campaign."""

from __future__ import annotations

import json  # noqa: F401
from dataclasses import FrozenInstanceError  # noqa: F401
from datetime import date

import pytest

import src.validation.walk_forward  # noqa: F401  # co-mod anchor for lean_check AST linkage

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
    # Baseline and candidate run on train and test; chosen reuses arm (4 per fold).
    assert len(runner.configs) == 4 * len(report.folds)


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


def test_walk_forward_compound_growth_objective_wires() -> None:
    from datetime import date

    from src.sim.allocation import AllocationConfig, AllocationResult, Snapshot
    from src.validation.experiment import load_experiment_config
    from src.validation.walk_forward import run_walk_forward_adoption

    spec = load_experiment_config("configs/experiments/wf_soxx100_compound_growth.json")

    def runner(cfg: AllocationConfig) -> AllocationResult:
        snap = Snapshot(session=date(2020, 1, 31), contribution_krw=1.0, nav_krw=1.0)
        targets = cfg.targets_override or {}
        if targets.get("SOXX") == 1.0:
            tw, contrib = 200.0, 100.0
        else:
            tw, contrib = 150.0, 100.0
        return AllocationResult(
            config=cfg,
            snapshots=(snap,),
            terminal_wealth_krw=tw,
            xirr=0.0,
            max_drawdown=-0.3,
            terminal_wealth_real_krw=tw,
            xirr_real=0.1,
            total_contribution_real_krw=contrib,
        )

    report = run_walk_forward_adoption(spec, runner)
    assert report.process_adopted_vs_baseline is True

def test_walk_forward_adoption_still_single_candidate_only() -> None:
    scenario_id = "test_run_walk_forward_adoption_still_single_candidate_only"
    import pytest

    from src.policy.targets import PolicyId
    from src.validation.walk_forward import run_walk_forward_adoption
    from tests.unit.validation.campaign_parts.test_walk_forward import _RecordingRunner, _spec

    multi_candidate = _spec().model_copy(
        update={'candidates': [_spec().candidates[0], _spec().candidates[0].model_copy(update={'id': 'dup'})]}
    )
    with pytest.raises(ValueError, match='exactly one candidate'):
        run_walk_forward_adoption(multi_candidate, _RecordingRunner(dict.fromkeys(PolicyId, 100.0)))


def test_wf_reject_preserves_adaptive_baseline_arm_identity() -> None:
  """I14: train_adopted=False => chosen fold metrics match baseline test arm."""
  from src.policy.adaptive_contribution import FROZEN_ADAPTIVE_V5
  from src.policy.targets import PolicyId
  from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot
  from src.validation.experiment import AdaptiveContributionSpec, CandidateSpec, ExperimentSpec
  from src.validation.walk_forward import run_walk_forward_adoption

  frozen = AdaptiveContributionSpec(
      rank_window=126,
      downside_power=4.0,
      upside_power=0.25,
      min_multiplier=0.0,
      max_multiplier=2.0,
      include_vol_dampener=False,
      dispersion=1.35,
      neutral_deadband=5.0,
  )
  spec = ExperimentSpec(
      name="wf_adaptive_baseline_reject",
      start=date(2016, 7, 1),
      end=date(2022, 6, 30),
      contribution_krw=1_000_000.0,
      delta0=0.02,
      horizon_months=0,
      train_months=24,
      test_months=12,
      objective="compound_growth",
      baseline=CandidateSpec(
          id="qqq90_soxx10_adaptive_v5",
          policy="qqq",
          modules=1,
          targets={"QQQ": 0.9, "SOXX": 0.1},
      ),
      candidates=[
          CandidateSpec(
              id="qqq95_soxx5_adaptive_v5",
              policy="qqq",
              modules=2,
              targets={"QQQ": 0.95, "SOXX": 0.05},
          )
      ],
      adaptive_contribution=frozen,
      baseline_adaptive_contribution=frozen,
  )

  class _AdaptiveAwareRunner:
      def __call__(self, config: AllocationConfig) -> AllocationResult:
          adaptive = config.adaptive_contribution is FROZEN_ADAPTIVE_V5
          tw = 120.0 if adaptive else 100.0
          contrib = 95.0 if adaptive else 90.0
          snap = AllocationSnapshot(
              session=config.start,
              cash_krw=0.0,
              cash_usd=0.0,
              shares={},
              mark_krw=tw,
              contribution_krw=1_000_000.0,
              fees_krw=0.0,
          )
          return AllocationResult(
              config=config,
              snapshots=(snap,),
              terminal_wealth_krw=tw,
              xirr=0.0,
              max_drawdown=-0.1,
              terminal_wealth_real_krw=tw,
              xirr_real=0.10 if adaptive else 0.05,
              total_contribution_real_krw=contrib,
          )

  report = run_walk_forward_adoption(spec, _AdaptiveAwareRunner())
  assert len(report.folds) > 0
  for fold in report.folds:
      if not fold.train_adopted:
          assert fold.chosen_test_wealth == pytest.approx(fold.baseline_test_wealth)
          assert fold.chosen_total_contribution_real_krw == pytest.approx(fold.baseline_total_contribution_real_krw)
          assert fold.chosen_xirr_real == pytest.approx(fold.baseline_xirr_real)
          assert fold.chosen_real_gain == pytest.approx(fold.baseline_real_gain)

