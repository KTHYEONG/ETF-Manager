"""Walk-forward adoption campaign over an injected allocation runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final

from src.etf_manager.policy.targets import PolicyId
from src.etf_manager.sim.allocation import AllocationConfig
from src.etf_manager.validation.experiment import ExperimentSpec, resolve_overlay, resolve_reserve
from src.etf_manager.validation.gate import adoption_passes, certainty_equivalent
from src.etf_manager.validation.windows import walk_forward_windows

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from src.etf_manager.data.settings import DataSettings
    from src.etf_manager.policy.overlay import OverlayConfig
    from src.etf_manager.policy.reserve import ReserveConfig
    from src.etf_manager.sim.allocation import AllocationResult

__all__ = [
    "COST_SCENARIOS",
    "CampaignReport",
    "CostGridReport",
    "CostScenario",
    "FoldOutcome",
    "run_walk_forward_adoption",
    "run_walk_forward_cost_grid",
    "run_walk_forward_proxy_adoption",
    "write_campaign_report",
    "write_cost_grid_report",
]

_CE_GAMMAS: Final[tuple[float, ...]] = (2.0, 5.0, 10.0)


@dataclass(frozen=True, slots=True)
class FoldOutcome:
    """Train-phase adoption decision plus realized test-phase wealths."""

    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_adopted: bool
    chosen_policy: PolicyId
    baseline_test_wealth: float
    candidate_test_wealth: float
    chosen_test_wealth: float


@dataclass(frozen=True, slots=True)
class CampaignReport:
    """Every fold outcome plus the pooled adopted-vs-baseline CE verdict.

    ``candidate_test_ce`` is the always-candidate diagnostic; the verdict comes
    only from chosen versus baseline.
    """

    name: str
    candidate_id: str
    modules: int
    folds: tuple[FoldOutcome, ...]
    baseline_test_ce: Mapping[float, float]
    candidate_test_ce: Mapping[float, float]
    chosen_test_ce: Mapping[float, float]
    process_adopted_vs_baseline: bool


@dataclass(frozen=True, slots=True)
class CostScenario:
    """Fixed transaction-cost pair applied to every arm of one grid pass."""

    id: str
    commission_bps: float
    fx_spread_bps: float


COST_SCENARIOS: Final[tuple[CostScenario, ...]] = (
    CostScenario(id="ideal", commission_bps=0.0, fx_spread_bps=0.0),
    CostScenario(id="low", commission_bps=5.0, fx_spread_bps=10.0),
    CostScenario(id="base", commission_bps=10.0, fx_spread_bps=20.0),
    CostScenario(id="stress", commission_bps=50.0, fx_spread_bps=50.0),
)


@dataclass(frozen=True, slots=True)
class ScenarioOutcome:
    """One cost scenario paired with its walk-forward campaign verdict."""

    scenario: CostScenario
    campaign: CampaignReport


@dataclass(frozen=True, slots=True)
class CostGridReport:
    """Walk-forward adoption verdicts across the fixed cost grid."""

    name: str
    outcomes: tuple[ScenarioOutcome, ...]

    @property
    def all_scenarios_adopted(self) -> bool:
        """True only when every scenario's campaign beats the baseline."""
        return all(outcome.campaign.process_adopted_vs_baseline for outcome in self.outcomes)


def _arm_config(
    spec: ExperimentSpec,
    policy: PolicyId,
    start: date,
    end: date,
    overlay: OverlayConfig | None,
    reserve: ReserveConfig | None,
) -> AllocationConfig:
    """Identical cashflow/costs for every arm on one sliced window."""
    return AllocationConfig(
        policy=policy,
        start=start,
        end=end,
        monthly_contribution_krw=spec.contribution_krw,
        fill_delay_sessions=1,
        fx_spread_bps=spec.fx_spread_bps,
        commission_bps=spec.commission_bps,
        tilt=None,
        rebalance_band=None,
        overlay=overlay,
        reserve=reserve,
        currency=None,
        mapping=None,
    )


def _singleton_ce(wealth: float) -> Mapping[float, float]:
    """CE gammas of a one-observation wealth vector."""
    return {gamma: certainty_equivalent((wealth,), gamma=gamma) for gamma in _CE_GAMMAS}


def run_walk_forward_adoption(
    spec: ExperimentSpec,
    runner: Callable[[AllocationConfig], AllocationResult],
) -> CampaignReport:
    """Select per fold on train CE, then realize chosen-versus-baseline wealth on test.

    The runner is called once per arm per phase (baseline then candidate on train;
    baseline, candidate, and chosen on test — chosen re-runs even when it repeats
    another arm); ``spec`` is never mutated. Baseline arms stay un-overlayed and
    un-reserved; candidate arms carry ``resolve_overlay(spec)`` and
    ``resolve_reserve(spec)``, and the chosen test arm keeps them only when the fold
    adopted on train.

    Raises:
        ValueError: When train/test months are absent, the candidate count is not
            one, or no walk-forward fold fits the window.
    """
    if spec.train_months is None or spec.test_months is None:
        raise ValueError("walk-forward adoption requires both train_months and test_months")
    if len(spec.candidates) != 1:
        raise ValueError(f"expected exactly one candidate, got {len(spec.candidates)}")
    candidate = spec.candidates[0]
    windows = walk_forward_windows(
        spec.start,
        spec.end,
        train_months=spec.train_months,
        test_months=spec.test_months,
    )
    if not windows:
        raise ValueError("no walk-forward folds fit the experiment window")
    candidate_overlay = resolve_overlay(spec)
    candidate_reserve = resolve_reserve(spec)

    def real_wealth(
        policy: PolicyId,
        start: date,
        end: date,
        arm_overlay: OverlayConfig | None,
        arm_reserve: ReserveConfig | None,
    ) -> float:
        return runner(
            _arm_config(spec, policy, start, end, arm_overlay, arm_reserve)
        ).terminal_wealth_real_krw

    folds: list[FoldOutcome] = []
    baseline_wealths: list[float] = []
    candidate_wealths: list[float] = []
    chosen_wealths: list[float] = []
    for train_start, train_end, test_start, test_end in windows:
        baseline_train = real_wealth(spec.baseline.policy, train_start, train_end, None, None)
        candidate_train = real_wealth(
            candidate.policy, train_start, train_end, candidate_overlay, candidate_reserve
        )
        train_adopted = adoption_passes(
            _singleton_ce(candidate_train),
            _singleton_ce(baseline_train),
            delta0=spec.delta0,
            modules=candidate.modules,
        )
        chosen_policy = candidate.policy if train_adopted else spec.baseline.policy
        keep_modules = (candidate_overlay, candidate_reserve) if train_adopted else (None, None)
        baseline_test = real_wealth(spec.baseline.policy, test_start, test_end, None, None)
        candidate_test = real_wealth(
            candidate.policy, test_start, test_end, candidate_overlay, candidate_reserve
        )
        chosen_test = real_wealth(chosen_policy, test_start, test_end, *keep_modules)
        folds.append(
            FoldOutcome(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_adopted=train_adopted,
                chosen_policy=chosen_policy,
                baseline_test_wealth=baseline_test,
                candidate_test_wealth=candidate_test,
                chosen_test_wealth=chosen_test,
            )
        )
        baseline_wealths.append(baseline_test)
        candidate_wealths.append(candidate_test)
        chosen_wealths.append(chosen_test)

    baseline_ce = {gamma: certainty_equivalent(baseline_wealths, gamma=gamma) for gamma in _CE_GAMMAS}
    candidate_ce = {gamma: certainty_equivalent(candidate_wealths, gamma=gamma) for gamma in _CE_GAMMAS}
    chosen_ce = {gamma: certainty_equivalent(chosen_wealths, gamma=gamma) for gamma in _CE_GAMMAS}
    process_adopted_vs_baseline = adoption_passes(
        chosen_ce, baseline_ce, delta0=spec.delta0, modules=candidate.modules
    )
    return CampaignReport(
        name=spec.name,
        candidate_id=candidate.id,
        modules=candidate.modules,
        folds=tuple(folds),
        baseline_test_ce=baseline_ce,
        candidate_test_ce=candidate_ce,
        chosen_test_ce=chosen_ce,
        process_adopted_vs_baseline=process_adopted_vs_baseline,
    )


def run_walk_forward_proxy_adoption(
    spec: ExperimentSpec,
    etf_runner: Callable[[AllocationConfig], AllocationResult],
    proxy_runner: Callable[[AllocationConfig], AllocationResult],
) -> CampaignReport:
    """Run the Wave C identity-isolation campaign: ETF baseline versus research proxy.

    The ETF runner serves only the non-proxy arms and the proxy runner only the
    R1 arm, so a dispatching runner feeds the shared walk-forward CE gate;
    ``spec`` is never mutated.

    Raises:
        ValueError: When train/test months are absent, an overlay or reserve spec is
            set, the candidate is not exactly one R1_US_MKT_FF policy, the baseline
            itself is the proxy identity, or any transaction cost is nonzero.
    """
    if spec.train_months is None or spec.test_months is None:
        raise ValueError("walk-forward proxy adoption requires both train_months and test_months")
    if spec.overlay is not None:
        raise ValueError("walk-forward proxy adoption does not support overlay specs")
    if spec.reserve is not None:
        raise ValueError("walk-forward proxy adoption does not support reserve specs")
    if len(spec.candidates) != 1:
        raise ValueError(f"expected exactly one candidate, got {len(spec.candidates)}")
    candidate = spec.candidates[0]
    if candidate.policy is not PolicyId.R1_US_MKT_FF:
        raise ValueError(f"proxy campaign candidate must be R1_US_MKT_FF (research_proxy), got {candidate.policy!s}")
    if spec.baseline.policy is PolicyId.R1_US_MKT_FF:
        raise ValueError("proxy campaign baseline must be an ETF policy, not the research_proxy identity")
    if spec.commission_bps != 0.0 or spec.fx_spread_bps != 0.0:
        raise ValueError(
            "Wave C identity isolation requires commission_bps == 0 and fx_spread_bps == 0, "
            f"got commission_bps={spec.commission_bps!r}, fx_spread_bps={spec.fx_spread_bps!r}"
        )

    def _dispatching_runner(config: AllocationConfig) -> AllocationResult:
        runner = proxy_runner if config.policy is PolicyId.R1_US_MKT_FF else etf_runner
        return runner(config)

    return run_walk_forward_adoption(spec, _dispatching_runner)


def _fold_records(folds: tuple[FoldOutcome, ...]) -> list[dict[str, object]]:
    """JSON-ready records for one campaign's folds."""
    return [
        {
            "train_start": fold.train_start.isoformat(),
            "train_end": fold.train_end.isoformat(),
            "test_start": fold.test_start.isoformat(),
            "test_end": fold.test_end.isoformat(),
            "train_adopted": fold.train_adopted,
            "chosen_policy": str(fold.chosen_policy),
            "baseline_test_wealth": fold.baseline_test_wealth,
            "candidate_test_wealth": fold.candidate_test_wealth,
            "chosen_test_wealth": fold.chosen_test_wealth,
        }
        for fold in folds
    ]


def write_campaign_report(report: CampaignReport, settings: DataSettings, experiment_id: str) -> Path:
    """Persist the campaign verdict under ``experiments/{name}_{experiment_id}.json``.

    Returns:
        Path: The written UTF-8 JSON artifact.
    """
    payload = {
        "name": report.name,
        "experiment_id": experiment_id,
        "process_adopted_vs_baseline": report.process_adopted_vs_baseline,
        "fold_count": len(report.folds),
        "folds": _fold_records(report.folds),
    }
    experiments_dir = settings.resolved_data_root() / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    out_path = experiments_dir / f"{report.name}_{experiment_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def run_walk_forward_cost_grid(
    spec: ExperimentSpec,
    runner: Callable[[AllocationConfig], AllocationResult],
    scenarios: tuple[CostScenario, ...] = COST_SCENARIOS,
) -> CostGridReport:
    """Re-run the identical walk-forward campaign once per fixed cost scenario.

    ``spec`` is never mutated: each scenario applies its own bps via
    ``model_copy`` and then delegates to the single-campaign runner.

    Raises:
        ValueError: Propagated from the campaign when the spec violates the
            walk-forward contract.
    """
    outcomes = [
        ScenarioOutcome(
            scenario=scenario,
            campaign=run_walk_forward_adoption(
                spec.model_copy(
                    update={
                        "commission_bps": scenario.commission_bps,
                        "fx_spread_bps": scenario.fx_spread_bps,
                    }
                ),
                runner,
            ),
        )
        for scenario in scenarios
    ]
    return CostGridReport(name=spec.name, outcomes=tuple(outcomes))


def write_cost_grid_report(report: CostGridReport, settings: DataSettings, experiment_id: str) -> Path:
    """Persist the grid verdict under ``experiments/{name}_costs_{experiment_id}.json``.

    Returns:
        Path: The written UTF-8 JSON artifact.
    """
    payload = {
        "name": report.name,
        "experiment_id": experiment_id,
        "all_scenarios_adopted": report.all_scenarios_adopted,
        "scenarios": [
            {
                "id": outcome.scenario.id,
                "commission_bps": outcome.scenario.commission_bps,
                "fx_spread_bps": outcome.scenario.fx_spread_bps,
                "process_adopted_vs_baseline": outcome.campaign.process_adopted_vs_baseline,
                "fold_count": len(outcome.campaign.folds),
                "folds": _fold_records(outcome.campaign.folds),
            }
            for outcome in report.outcomes
        ],
    }
    experiments_dir = settings.resolved_data_root() / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    out_path = experiments_dir / f"{report.name}_costs_{experiment_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
