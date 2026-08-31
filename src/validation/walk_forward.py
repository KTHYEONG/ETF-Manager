"""Walk-forward adoption campaign over an injected allocation runner."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final, Literal

from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig
from src.validation.experiment import (
    ExperimentSpec,
    resolve_adaptive_contribution,
    resolve_arm_targets,
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
    certainty_equivalent,
    compound_growth_process_passes,
    compound_growth_train_passes,
    contribution_growth_train_passes,
    growth_first_process_passes,
    growth_first_train_passes,
)
from src.validation.gate import (
    contribution_growth_process_passes as _orig_contribution_growth_process_passes,
)
from src.validation.windows import walk_forward_windows


def contribution_growth_process_passes(*args, **kwargs):  # type: ignore[no-untyped-def]
    import sys

    cam = sys.modules.get("src.validation.campaign")
    if cam is not None:
        patched = getattr(cam, "contribution_growth_process_passes", None)
        if patched is not None and patched is not _orig_contribution_growth_process_passes and patched is not contribution_growth_process_passes:
            return patched(*args, **kwargs)
    return _orig_contribution_growth_process_passes(*args, **kwargs)

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
    "CampaignReport",
    "FoldOutcome",
    "run_walk_forward_adoption",
    "run_walk_forward_proxy_adoption",
    "run_walk_forward_tournament",
    "warm_baseline_arm_cache",
    "write_campaign_report",
]

_CE_GAMMAS: Final[tuple[float, ...]] = (2.0, 5.0, 10.0)


@dataclass(frozen=True, slots=True)
class FoldOutcome:
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
    name: str
    candidate_id: str
    modules: int
    folds: tuple[FoldOutcome, ...]
    baseline_test_ce: Mapping[float, float]
    candidate_test_ce: Mapping[float, float]
    chosen_test_ce: Mapping[float, float]
    process_adopted_vs_baseline: bool


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
    targets_override: Mapping[str, float] | None = None,
) -> AllocationConfig:
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
        targets_override=targets_override,
    )


def _singleton_ce(wealth: float) -> Mapping[float, float]:
    return {gamma: certainty_equivalent((wealth,), gamma=gamma) for gamma in _CE_GAMMAS}


def _real_profit(result: AllocationResult) -> float:
    return result.terminal_wealth_real_krw - result.total_contribution_real_krw


def warm_baseline_arm_cache(
    spec: ExperimentSpec,
    runner: Callable[[AllocationConfig], AllocationResult],
) -> dict[tuple[date, date], AllocationResult]:
    if spec.train_months is None or spec.test_months is None:
        raise ValueError("walk-forward adoption requires both train_months and test_months")
    baseline_adaptive = resolve_baseline_adaptive_contribution(spec)
    windows = walk_forward_windows(spec.start, spec.end, train_months=spec.train_months, test_months=spec.test_months)
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
                    targets_override=resolve_arm_targets(spec.baseline),
                )
            )
    return cache


def run_walk_forward_adoption(
    spec: ExperimentSpec,
    runner: Callable[[AllocationConfig], AllocationResult],
    *,
    baseline_arm_cache: Mapping[tuple[date, date], AllocationResult] | None = None,
) -> CampaignReport:
    if spec.train_months is None or spec.test_months is None:
        raise ValueError("walk-forward adoption requires both train_months and test_months")
    if len(spec.candidates) != 1:
        raise ValueError(f"expected exactly one candidate, got {len(spec.candidates)}")
    growth_first_modules = (spec.cadence, spec.reserve, spec.contribution_shape, spec.kafi_deployment)
    if spec.objective == "growth_first" and sum(module is not None for module in growth_first_modules) != 1:
        raise ValueError("objective 'growth_first' requires exactly one of a cadence, reserve, contribution_shape, or kafi_deployment module")
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
        raise ValueError("objective 'adaptive_growth' requires exactly one adaptive_contribution module")
    baseline_targets_override = resolve_arm_targets(spec.baseline)
    candidate_targets_override = resolve_arm_targets(candidate)

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
        targets_override: Mapping[str, float] | None = None,
    ) -> AllocationResult:
        return runner(
            _arm_config(
                spec, policy, start, end, arm_overlay, arm_reserve, arm_mapping, arm_currency,
                arm_contribution_shape, arm_kafi_deployment, arm_adaptive_contribution, arm_cadence,
                targets_override=targets_override,
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
            spec.baseline.policy, start, end, None, None, None, None,
            arm_adaptive_contribution=baseline_adaptive_contribution,
            targets_override=baseline_targets_override,
        )

    for train_start, train_end, test_start, test_end in windows:
        baseline_train_arm = _baseline_arm(train_start, train_end)
        candidate_train_arm = arm_result(
            candidate.policy, train_start, train_end, candidate_overlay, candidate_reserve,
            candidate_mapping, candidate_currency, candidate_contribution_shape,
            candidate_kafi_deployment, candidate_adaptive_contribution, candidate_cadence,
            targets_override=candidate_targets_override,
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
        elif spec.objective == "compound_growth":
            train_adopted = compound_growth_train_passes(
                candidate_tw=candidate_train_arm.terminal_wealth_real_krw,
                baseline_tw=baseline_train_arm.terminal_wealth_real_krw,
                candidate_real_gain=_real_profit(candidate_train_arm),
                baseline_real_gain=_real_profit(baseline_train_arm),
                candidate_xirr_real=candidate_train_arm.xirr_real,
                baseline_xirr_real=baseline_train_arm.xirr_real,
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
                delta0=spec.hurdle, modules=candidate.modules,
            )
        chosen_policy = candidate.policy if train_adopted else spec.baseline.policy
        keep_extras = (candidate_overlay, candidate_reserve, candidate_mapping, candidate_currency) if train_adopted else (None, None, None, None)
        chosen_contribution_shape = candidate_contribution_shape if train_adopted else None
        chosen_kafi_deployment = candidate_kafi_deployment if train_adopted else None
        chosen_adaptive_contribution = candidate_adaptive_contribution if train_adopted else None
        chosen_cadence = candidate_cadence if train_adopted else "monthly"
        baseline_test_arm = _baseline_arm(test_start, test_end)
        candidate_test_arm = arm_result(
            candidate.policy, test_start, test_end, candidate_overlay, candidate_reserve,
            candidate_mapping, candidate_currency, candidate_contribution_shape,
            candidate_kafi_deployment, candidate_adaptive_contribution, candidate_cadence,
            targets_override=candidate_targets_override,
        )
        chosen_targets_override = candidate_targets_override if train_adopted else baseline_targets_override
        chosen_test_arm = arm_result(
            chosen_policy, test_start, test_end, *keep_extras,
            chosen_contribution_shape, chosen_kafi_deployment, chosen_adaptive_contribution,
            chosen_cadence, targets_override=chosen_targets_override,
        )
        baseline_test = baseline_test_arm.terminal_wealth_real_krw
        candidate_test = candidate_test_arm.terminal_wealth_real_krw
        chosen_test = chosen_test_arm.terminal_wealth_real_krw
        folds.append(
            FoldOutcome(
                train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end,
                train_adopted=train_adopted, chosen_policy=chosen_policy,
                baseline_test_wealth=baseline_test, candidate_test_wealth=candidate_test, chosen_test_wealth=chosen_test,
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
        process_adopted_vs_baseline = contribution_growth_process_passes(  # type: ignore[no-untyped-call]
            chosen_test_tw=tuple(chosen_wealths), baseline_test_tw=tuple(baseline_wealths),
            chosen_test_real_gain=tuple(chosen_gains), baseline_test_real_gain=tuple(baseline_gains),
            chosen_test_xirr_real=tuple(chosen_xirrs), baseline_test_xirr_real=tuple(baseline_xirrs),
        )
    elif spec.objective == "compound_growth":
        process_adopted_vs_baseline = compound_growth_process_passes(
            chosen_test_tw=tuple(chosen_wealths), baseline_test_tw=tuple(baseline_wealths),
            chosen_test_real_gain=tuple(chosen_gains), baseline_test_real_gain=tuple(baseline_gains),
            chosen_test_xirr_real=tuple(chosen_xirrs), baseline_test_xirr_real=tuple(baseline_xirrs),
        )
    elif spec.objective == "growth_first":
        process_adopted_vs_baseline = growth_first_process_passes(chosen_test=tuple(chosen_wealths), baseline_test=tuple(baseline_wealths))
    else:
        process_adopted_vs_baseline = adoption_passes(chosen_ce, baseline_ce, delta0=spec.hurdle, modules=candidate.modules)
    return CampaignReport(
        name=spec.name, candidate_id=candidate.id, modules=candidate.modules, folds=tuple(folds),
        baseline_test_ce=baseline_ce, candidate_test_ce=candidate_ce, chosen_test_ce=chosen_ce,
        process_adopted_vs_baseline=process_adopted_vs_baseline,
    )


def run_walk_forward_proxy_adoption(
    spec: ExperimentSpec,
    etf_runner: Callable[[AllocationConfig], AllocationResult],
    proxy_runner: Callable[[AllocationConfig], AllocationResult],
) -> CampaignReport:
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
        raise ValueError(f"Wave C identity isolation requires commission_bps == 0 and fx_spread_bps == 0, got commission_bps={spec.commission_bps!r}, fx_spread_bps={spec.fx_spread_bps!r}")

    def _dispatching_runner(config: AllocationConfig) -> AllocationResult:
        runner = proxy_runner if config.policy is PolicyId.FF_PROXY else etf_runner
        return runner(config)

    return run_walk_forward_adoption(spec, _dispatching_runner)


def run_walk_forward_tournament(
    spec: ExperimentSpec,
    runner: Callable[[AllocationConfig], AllocationResult],
    *,
    baseline_arm_cache: Mapping[tuple[date, date], AllocationResult] | None = None,
) -> dict[str, CampaignReport]:
    if len(spec.candidates) < 1:
        raise ValueError(f"expected at least one candidate, got {len(spec.candidates)}")
    # shared cache ensures N candidates do not duplicate baseline runs
    cache = baseline_arm_cache
    if cache is None:
        cache = warm_baseline_arm_cache(spec, runner)
    reports: dict[str, CampaignReport] = {}
    for candidate in spec.candidates:
        clone = spec.model_copy(update={"candidates": [candidate]})
        reports[candidate.id] = run_walk_forward_adoption(clone, runner, baseline_arm_cache=cache)
    return reports


def _fold_records(folds: tuple[FoldOutcome, ...]) -> list[dict[str, object]]:
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
    payload = {
        "name": report.name,
        "experiment_id": experiment_id,
        "process_adopted_vs_baseline": report.process_adopted_vs_baseline,
        "fold_count": len(report.folds),
        "folds": _fold_records(report.folds),
    }
    from src.data.paths import experiments_dir

    out_dir = experiments_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.name}_{experiment_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
