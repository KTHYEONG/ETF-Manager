"""Walk-forward adoption campaign over an injected allocation runner."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final, Literal

from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig
from src.validation.evaluate import evaluate_cohort_wealths
from src.validation.experiment import (
    ExperimentSpec,
    resolve_adaptive_contribution,
    resolve_baseline_adaptive_contribution,
    resolve_cadence,
    resolve_contribution_shape,
    resolve_currency,
    resolve_kafi_deployment,
    resolve_mapping,
    resolve_overlay,
    resolve_reserve,
)
from src.validation.gate import (
    adoption_passes,
    bootstrap_tail_passes,
    certainty_equivalent,
    contribution_growth_process_passes,
    contribution_growth_train_passes,
    growth_first_process_passes,
    growth_first_train_passes,
    worst_cohort_passes,
)
from src.validation.windows import rolling_cohorts, walk_forward_windows

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from src.data.settings import DataSettings
    from src.etf.mapping import MappingConfig
    from src.policy.adaptive_contribution import AdaptiveContributionConfig
    from src.policy.contribution_shape import ContributionShapeConfig
    from src.policy.currency import CurrencyConfig
    from src.policy.kafi_deployment import KafiDeploymentConfig
    from src.policy.overlay import OverlayConfig
    from src.policy.reserve import ReserveConfig
    from src.sim.allocation import AllocationResult

__all__ = [
    "COST_SCENARIOS",
    "CadenceRobustnessReport",
    "CampaignReport",
    "CostGridReport",
    "CostScenario",
    "FoldOutcome",
    "run_cadence_robustness",
    "run_walk_forward_adoption",
    "run_walk_forward_cost_grid",
    "run_walk_forward_proxy_adoption",
    "warm_baseline_arm_cache",
    "write_cadence_robustness_report",
    "write_campaign_report",
    "write_cost_grid_report",
]

_CE_GAMMAS: Final[tuple[float, ...]] = (2.0, 5.0, 10.0)


@dataclass(frozen=True, slots=True)
class FoldOutcome:
    """Train-phase adoption decision plus realized test-phase wealths.

    ``*_test_wealth`` is real terminal wealth (TW); the contribution/gain/XIRR
    triplets carry each arm's total real contributed capital, real profit, and
    money-weighted real rate so variable-cashflow arms stay capital-aware.
    """

    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_adopted: bool
    chosen_policy: PolicyId
    baseline_test_wealth: float
    candidate_test_wealth: float
    chosen_test_wealth: float
    baseline_total_contribution_real_krw: float = 0.0
    candidate_total_contribution_real_krw: float = 0.0
    chosen_total_contribution_real_krw: float = 0.0
    baseline_real_gain: float = 0.0
    candidate_real_gain: float = 0.0
    chosen_real_gain: float = 0.0
    baseline_xirr_real: float = 0.0
    candidate_xirr_real: float = 0.0
    chosen_xirr_real: float = 0.0


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


@dataclass(frozen=True, slots=True)
class CadenceRobustnessReport:
    """Cadence-arm verdicts: cost grid, worst cohort, and bootstrap tail gates."""

    name: str
    cost_grid: CostGridReport
    baseline_wealths: tuple[float, ...]
    candidate_wealths: tuple[float, ...]
    worst_cohort_ok: bool
    bootstrap_tail_ok: bool
    robust_adopted: bool


def _arm_config(
    spec: ExperimentSpec,
    policy: PolicyId,
    start: date,
    end: date,
    overlay: OverlayConfig | None,
    reserve: ReserveConfig | None,
    mapping: MappingConfig | None,
    currency: CurrencyConfig | None,
    contribution_shape: ContributionShapeConfig | None = None,
    kafi_deployment: KafiDeploymentConfig | None = None,
    adaptive_contribution: AdaptiveContributionConfig | None = None,
    cadence: Literal["monthly", "month_open", "twice_monthly"] = "monthly",
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
        currency=currency,
        mapping=mapping,
        contribution_shape=contribution_shape,
        kafi_deployment=kafi_deployment,
        adaptive_contribution=adaptive_contribution,
        cadence=cadence,
    )


def _singleton_ce(wealth: float) -> Mapping[float, float]:
    """CE gammas of a one-observation wealth vector."""
    return {gamma: certainty_equivalent((wealth,), gamma=gamma) for gamma in _CE_GAMMAS}


def _real_profit(result: AllocationResult) -> float:
    """Real terminal wealth minus real contributed capital."""
    return result.terminal_wealth_real_krw - result.total_contribution_real_krw


def warm_baseline_arm_cache(
    spec: ExperimentSpec,
    runner: Callable[[AllocationConfig], AllocationResult],
) -> dict[tuple[date, date], AllocationResult]:
    """Pre-compute baseline-arm allocations for every walk-forward train/test slice."""
    if spec.train_months is None or spec.test_months is None:
        raise ValueError("walk-forward adoption requires both train_months and test_months")
    baseline_adaptive = resolve_baseline_adaptive_contribution(spec)
    windows = walk_forward_windows(
        spec.start, spec.end, train_months=spec.train_months, test_months=spec.test_months
    )
    if not windows:
        raise ValueError("no walk-forward folds fit the experiment window")
    cache: dict[tuple[date, date], AllocationResult] = {}
    for train_start, train_end, test_start, test_end in windows:
        for start, end in ((train_start, train_end), (test_start, test_end)):
            key = (start, end)
            if key in cache:
                continue
            cache[key] = runner(
                _arm_config(
                    spec,
                    spec.baseline.policy,
                    start,
                    end,
                    None,
                    None,
                    None,
                    None,
                    adaptive_contribution=baseline_adaptive,
                )
            )
    return cache


def run_walk_forward_adoption(
    spec: ExperimentSpec,
    runner: Callable[[AllocationConfig], AllocationResult],
    *,
    baseline_arm_cache: Mapping[tuple[date, date], AllocationResult] | None = None,
) -> CampaignReport:
    """Select per fold on train CE, then realize chosen-versus-baseline wealth on test.

    The runner is called once per arm per phase (baseline then candidate on train;
    baseline, candidate, and chosen on test — chosen re-runs even when it repeats
    another arm); ``spec`` is never mutated. Baseline arms stay un-overlayed,
    un-reserved, unmapped, and undeferred;
    they carry only ``resolve_baseline_adaptive_contribution(spec)`` when present; candidate arms carry
    ``resolve_overlay(spec)``, ``resolve_reserve(spec)``, ``resolve_mapping(spec)``,
    and ``resolve_currency(spec)``, and the chosen test arm keeps them only when
    the fold adopted on train. Baseline arms signal month-end; candidate arms
    signal on ``resolve_cadence(spec)`` and the chosen test arm only after train adoption.

    With ``objective == 'growth_first'`` the train gate adopts on any strict
    real-TW gain whose MDD (``AllocationResult.max_drawdown``) deepens at most
    0.02, and the process verdict requires pooled chosen-over-baseline gain with
    every fold ratio at least 0.97; the complexity-penalized CE hurdle applies
    only to ``ce``.

    With ``objective == 'adaptive_growth'`` the train gate is capital-aware
    (strict TW and real-profit gains, non-inferior real XIRR, MDD slack) via
    ``contribution_growth_train_passes``, and the process verdict delegates to
    ``contribution_growth_process_passes`` over per-fold TW, real-gain, and
    real-XIR sequences; every fold records each arm's metrics.

    Raises:
        ValueError: When train/test months are absent, the candidate count is not
            one, no walk-forward fold fits the window, a growth_first objective
            lacks a cadence module, or an adaptive_growth objective lacks its module.
    """
    if spec.train_months is None or spec.test_months is None:
        raise ValueError("walk-forward adoption requires both train_months and test_months")
    if len(spec.candidates) != 1:
        raise ValueError(f"expected exactly one candidate, got {len(spec.candidates)}")
    growth_first_modules = (spec.cadence, spec.reserve, spec.contribution_shape, spec.kafi_deployment)
    if (
        spec.objective == "growth_first"
        and sum(module is not None for module in growth_first_modules) != 1
    ):
        raise ValueError(
            "objective 'growth_first' requires exactly one of a cadence, reserve, "
            "contribution_shape, or kafi_deployment module"
        )
    candidate = spec.candidates[0]
    windows = walk_forward_windows(spec.start, spec.end, train_months=spec.train_months, test_months=spec.test_months)
    if not windows:
        raise ValueError("no walk-forward folds fit the experiment window")
    candidate_overlay = resolve_overlay(spec)
    candidate_reserve = resolve_reserve(spec)
    candidate_mapping = resolve_mapping(spec)
    candidate_currency = resolve_currency(spec)
    candidate_contribution_shape = resolve_contribution_shape(spec)
    candidate_kafi_deployment = resolve_kafi_deployment(spec)
    candidate_adaptive_contribution = resolve_adaptive_contribution(spec)
    baseline_adaptive_contribution = resolve_baseline_adaptive_contribution(spec)
    candidate_cadence = resolve_cadence(spec) or "monthly"
    if spec.objective == "adaptive_growth" and candidate_adaptive_contribution is None:
        # model_copy bypasses model validation, so the campaign must fail closed itself.
        raise ValueError("objective 'adaptive_growth' requires exactly one adaptive_contribution module")

    def arm_result(
        policy: PolicyId,
        start: date,
        end: date,
        arm_overlay: OverlayConfig | None,
        arm_reserve: ReserveConfig | None,
        arm_mapping: MappingConfig | None,
        arm_currency: CurrencyConfig | None,
        arm_contribution_shape: ContributionShapeConfig | None = None,
        arm_kafi_deployment: KafiDeploymentConfig | None = None,
        arm_adaptive_contribution: AdaptiveContributionConfig | None = None,
        arm_cadence: Literal["monthly", "month_open", "twice_monthly"] = "monthly",
    ) -> AllocationResult:
        return runner(
            _arm_config(
                spec,
                policy,
                start,
                end,
                arm_overlay,
                arm_reserve,
                arm_mapping,
                arm_currency,
                arm_contribution_shape,
                arm_kafi_deployment,
                arm_adaptive_contribution,
                arm_cadence,
            )
        )

    folds: list[FoldOutcome] = []
    baseline_wealths: list[float] = []
    candidate_wealths: list[float] = []
    chosen_wealths: list[float] = []
    baseline_gains: list[float] = []
    chosen_gains: list[float] = []
    baseline_xirrs: list[float] = []
    chosen_xirrs: list[float] = []
    def _baseline_arm(start: date, end: date) -> AllocationResult:
        if baseline_arm_cache is not None:
            cached = baseline_arm_cache.get((start, end))
            if cached is not None:
                return cached
        return arm_result(
            spec.baseline.policy,
            start,
            end,
            None,
            None,
            None,
            None,
            arm_adaptive_contribution=baseline_adaptive_contribution,
        )

    for train_start, train_end, test_start, test_end in windows:
        baseline_train_arm = _baseline_arm(train_start, train_end)
        candidate_train_arm = arm_result(
            candidate.policy,
            train_start,
            train_end,
            candidate_overlay,
            candidate_reserve,
            candidate_mapping,
            candidate_currency,
            candidate_contribution_shape,
            candidate_kafi_deployment,
            candidate_adaptive_contribution,
            candidate_cadence,
        )
        if spec.objective == "adaptive_growth":
            train_adopted = contribution_growth_train_passes(
                candidate_tw=candidate_train_arm.terminal_wealth_real_krw,
                baseline_tw=baseline_train_arm.terminal_wealth_real_krw,
                candidate_real_gain=_real_profit(candidate_train_arm),
                baseline_real_gain=_real_profit(baseline_train_arm),
                candidate_xirr_real=candidate_train_arm.xirr_real,
                baseline_xirr_real=baseline_train_arm.xirr_real,
                candidate_mdd=candidate_train_arm.max_drawdown,
                baseline_mdd=baseline_train_arm.max_drawdown,
            )
        elif spec.objective == "growth_first":
            train_adopted = growth_first_train_passes(
                candidate_tw=candidate_train_arm.terminal_wealth_real_krw,
                baseline_tw=baseline_train_arm.terminal_wealth_real_krw,
                candidate_mdd=candidate_train_arm.max_drawdown,
                baseline_mdd=baseline_train_arm.max_drawdown,
            )
        else:
            train_adopted = adoption_passes(
                _singleton_ce(candidate_train_arm.terminal_wealth_real_krw),
                _singleton_ce(baseline_train_arm.terminal_wealth_real_krw),
                delta0=spec.hurdle,
                modules=candidate.modules,
            )
        chosen_policy = candidate.policy if train_adopted else spec.baseline.policy
        keep_extras = (
            (candidate_overlay, candidate_reserve, candidate_mapping, candidate_currency)
            if train_adopted
            else (None, None, None, None)
        )
        chosen_contribution_shape = candidate_contribution_shape if train_adopted else None
        chosen_kafi_deployment = candidate_kafi_deployment if train_adopted else None
        chosen_adaptive_contribution = candidate_adaptive_contribution if train_adopted else None
        chosen_cadence = candidate_cadence if train_adopted else "monthly"
        baseline_test_arm = _baseline_arm(test_start, test_end)
        candidate_test_arm = arm_result(
            candidate.policy,
            test_start,
            test_end,
            candidate_overlay,
            candidate_reserve,
            candidate_mapping,
            candidate_currency,
            candidate_contribution_shape,
            candidate_kafi_deployment,
            candidate_adaptive_contribution,
            candidate_cadence,
        )
        chosen_test_arm = arm_result(
            chosen_policy,
            test_start,
            test_end,
            *keep_extras,
            chosen_contribution_shape,
            chosen_kafi_deployment,
            chosen_adaptive_contribution,
            chosen_cadence,
        )
        baseline_test = baseline_test_arm.terminal_wealth_real_krw
        candidate_test = candidate_test_arm.terminal_wealth_real_krw
        chosen_test = chosen_test_arm.terminal_wealth_real_krw
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
                baseline_total_contribution_real_krw=baseline_test_arm.total_contribution_real_krw,
                candidate_total_contribution_real_krw=candidate_test_arm.total_contribution_real_krw,
                chosen_total_contribution_real_krw=chosen_test_arm.total_contribution_real_krw,
                baseline_real_gain=_real_profit(baseline_test_arm),
                candidate_real_gain=_real_profit(candidate_test_arm),
                chosen_real_gain=_real_profit(chosen_test_arm),
                baseline_xirr_real=baseline_test_arm.xirr_real,
                candidate_xirr_real=candidate_test_arm.xirr_real,
                chosen_xirr_real=chosen_test_arm.xirr_real,
            )
        )
        baseline_wealths.append(baseline_test)
        candidate_wealths.append(candidate_test)
        chosen_wealths.append(chosen_test)
        baseline_gains.append(_real_profit(baseline_test_arm))
        chosen_gains.append(_real_profit(chosen_test_arm))
        baseline_xirrs.append(baseline_test_arm.xirr_real)
        chosen_xirrs.append(chosen_test_arm.xirr_real)

    baseline_ce = {gamma: certainty_equivalent(baseline_wealths, gamma=gamma) for gamma in _CE_GAMMAS}
    candidate_ce = {gamma: certainty_equivalent(candidate_wealths, gamma=gamma) for gamma in _CE_GAMMAS}
    chosen_ce = {gamma: certainty_equivalent(chosen_wealths, gamma=gamma) for gamma in _CE_GAMMAS}
    if spec.objective == "adaptive_growth":
        process_adopted_vs_baseline = contribution_growth_process_passes(
            chosen_test_tw=tuple(chosen_wealths),
            baseline_test_tw=tuple(baseline_wealths),
            chosen_test_real_gain=tuple(chosen_gains),
            baseline_test_real_gain=tuple(baseline_gains),
            chosen_test_xirr_real=tuple(chosen_xirrs),
            baseline_test_xirr_real=tuple(baseline_xirrs),
        )
    elif spec.objective == "growth_first":
        process_adopted_vs_baseline = growth_first_process_passes(
            chosen_test=tuple(chosen_wealths),
            baseline_test=tuple(baseline_wealths),
        )
    else:
        process_adopted_vs_baseline = adoption_passes(
            chosen_ce, baseline_ce, delta0=spec.hurdle, modules=candidate.modules
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
        ValueError: When train/test months are absent, an overlay, reserve,
            mapping, currency, or cadence spec is set, the candidate is not exactly
            one FF_PROXY policy, the baseline itself is the proxy identity, or
            any transaction cost is nonzero.
    """
    if spec.train_months is None or spec.test_months is None:
        raise ValueError("walk-forward proxy adoption requires both train_months and test_months")
    if spec.overlay is not None:
        raise ValueError("walk-forward proxy adoption does not support overlay specs")
    if spec.reserve is not None:
        raise ValueError("walk-forward proxy adoption does not support reserve specs")
    if spec.mapping is not None:
        raise ValueError("walk-forward proxy adoption does not support mapping specs")
    if spec.currency is not None:
        raise ValueError("walk-forward proxy adoption does not support currency specs")
    if spec.cadence is not None:
        raise ValueError("walk-forward proxy adoption does not support cadence specs")
    if len(spec.candidates) != 1:
        raise ValueError(f"expected exactly one candidate, got {len(spec.candidates)}")
    candidate = spec.candidates[0]
    if candidate.policy is not PolicyId.FF_PROXY:
        raise ValueError(f"proxy campaign candidate must be FF_PROXY (research_proxy), got {candidate.policy!s}")
    if spec.baseline.policy is PolicyId.FF_PROXY:
        raise ValueError("proxy campaign baseline must be an ETF policy, not the research_proxy identity")
    if spec.commission_bps != 0.0 or spec.fx_spread_bps != 0.0:
        raise ValueError(
            "Wave C identity isolation requires commission_bps == 0 and fx_spread_bps == 0, "
            f"got commission_bps={spec.commission_bps!r}, fx_spread_bps={spec.fx_spread_bps!r}"
        )

    def _dispatching_runner(config: AllocationConfig) -> AllocationResult:
        runner = proxy_runner if config.policy is PolicyId.FF_PROXY else etf_runner
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
            "baseline_total_contribution_real_krw": fold.baseline_total_contribution_real_krw,
            "candidate_total_contribution_real_krw": fold.candidate_total_contribution_real_krw,
            "chosen_total_contribution_real_krw": fold.chosen_total_contribution_real_krw,
            "baseline_real_gain": fold.baseline_real_gain,
            "candidate_real_gain": fold.candidate_real_gain,
            "chosen_real_gain": fold.chosen_real_gain,
            "baseline_xirr_real": fold.baseline_xirr_real,
            "candidate_xirr_real": fold.candidate_xirr_real,
            "chosen_xirr_real": fold.chosen_xirr_real,
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


def run_cadence_robustness(
    spec: ExperimentSpec,
    runner: Callable[[AllocationConfig], AllocationResult],
    *,
    n_paths: int,
    seed: int,
    horizon_months: int = 36,
    step_months: int = 12,
) -> CadenceRobustnessReport:
    """Adopt a cadence arm only when every robustness sub-gate holds.

    The cost arm re-runs the walk-forward grid; the cohort arms roll fixed
    ``horizon_months``/``step_months`` cohorts over the full window with the
    baseline on the monthly cadence and the candidate on ``resolve_cadence(spec)``
    (both un-overlayed, un-reserved, unmapped, undeferred). All three sub-gates
    are always evaluated — no short-circuit — and ``spec`` is never mutated.

    Raises:
        ValueError: When the objective is not growth_first, no cadence resolves,
            or no rolling cohort fits the experiment window.
    """
    candidate_cadence = resolve_cadence(spec)
    if spec.objective != "growth_first" or candidate_cadence is None:
        raise ValueError("cadence robustness requires objective 'growth_first' and a resolvable cadence")
    cohorts = rolling_cohorts(spec.start, spec.end, horizon_months=horizon_months, step_months=step_months)
    if not cohorts:
        raise ValueError(
            f"no rolling cohorts fit [{spec.start.isoformat()}, {spec.end.isoformat()}] "
            f"with horizon_months={horizon_months}, step_months={step_months}"
        )
    cost_grid = run_walk_forward_cost_grid(spec, runner)
    baseline_template = _arm_config(spec, spec.baseline.policy, spec.start, spec.end, None, None, None, None)
    candidate_template = _arm_config(
        spec,
        spec.candidates[0].policy,
        spec.start,
        spec.end,
        None,
        None,
        None,
        None,
        cadence=candidate_cadence,
    )
    baseline_wealths = evaluate_cohort_wealths(baseline_template, cohorts, runner)
    candidate_wealths = evaluate_cohort_wealths(candidate_template, cohorts, runner)
    worst_cohort_ok = worst_cohort_passes(candidate_wealths, baseline_wealths)
    bootstrap_tail_ok = bootstrap_tail_passes(candidate_wealths, baseline_wealths, n_paths=n_paths, seed=seed)
    return CadenceRobustnessReport(
        name=spec.name,
        cost_grid=cost_grid,
        baseline_wealths=baseline_wealths,
        candidate_wealths=candidate_wealths,
        worst_cohort_ok=worst_cohort_ok,
        bootstrap_tail_ok=bootstrap_tail_ok,
        robust_adopted=cost_grid.all_scenarios_adopted and worst_cohort_ok and bootstrap_tail_ok,
    )


def write_cadence_robustness_report(
    report: CadenceRobustnessReport, settings: DataSettings, experiment_id: str
) -> Path:
    """Persist the verdict under ``experiments/{name}_robustness_{experiment_id}.json``.

    Returns:
        Path: The written UTF-8 JSON artifact.
    """
    payload = {
        "name": report.name,
        "experiment_id": experiment_id,
        "robust_adopted": report.robust_adopted,
        "all_scenarios_adopted": report.cost_grid.all_scenarios_adopted,
        "worst_cohort_ok": report.worst_cohort_ok,
        "bootstrap_tail_ok": report.bootstrap_tail_ok,
        "cohort_count": len(report.candidate_wealths),
    }
    experiments_dir = settings.resolved_data_root() / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    out_path = experiments_dir / f"{report.name}_robustness_{experiment_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
