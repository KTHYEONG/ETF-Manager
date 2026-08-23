"""Identical-cashflow ablation over an injected allocation runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from src.etf_manager.policy.targets import PolicyId
from src.etf_manager.sim.allocation import AllocationConfig
from src.etf_manager.validation.evaluate import evaluate_cohort_wealths
from src.etf_manager.validation.experiment import CandidateSpec, ExperimentSpec
from src.etf_manager.validation.gate import adoption_passes, certainty_equivalent
from src.etf_manager.validation.windows import rolling_cohorts

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from src.etf_manager.sim.allocation import AllocationResult

__all__ = ["AblationReport", "AblationRow", "run_ablation"]

_CE_GAMMAS: Final[tuple[float, ...]] = (2.0, 5.0, 10.0)


@dataclass(frozen=True, slots=True)
class AblationRow:
    """Complexity-penalized gate outcome for one candidate arm."""

    candidate_id: str
    policy: PolicyId
    modules: int
    wealths: tuple[float, ...]
    ce: Mapping[float, float]
    ce_ratio: Mapping[float, float]
    adopted: bool


@dataclass(frozen=True, slots=True)
class AblationReport:
    """Baseline arm plus every gated candidate row."""

    name: str
    baseline_policy: PolicyId
    baseline_modules: int
    baseline_wealths: tuple[float, ...]
    baseline_ce: Mapping[float, float]
    rows: tuple[AblationRow, ...]


def _arm_config(spec: ExperimentSpec, policy: PolicyId) -> AllocationConfig:
    """Identical cashflow/window/costs for every arm; only ``policy`` differs."""
    return AllocationConfig(
        policy=policy,
        start=spec.start,
        end=spec.end,
        monthly_contribution_krw=spec.contribution_krw,
        fill_delay_sessions=1,
        fx_spread_bps=0.0,
        commission_bps=0.0,
        tilt=None,
        rebalance_band=None,
        overlay=None,
        currency=None,
        mapping=None,
    )


def _wealth_vector(
    spec: ExperimentSpec,
    config: AllocationConfig,
    runner: Callable[[AllocationConfig], AllocationResult],
) -> tuple[float, ...]:
    """Singleton terminal wealth at horizon 0; otherwise non-overlapping cohorts."""
    if spec.horizon_months == 0:
        return (runner(config).terminal_wealth_real_krw,)
    cohorts = rolling_cohorts(
        spec.start,
        spec.end,
        horizon_months=spec.horizon_months,
        step_months=spec.horizon_months,
    )
    return evaluate_cohort_wealths(config, cohorts, runner)


def _gated_row(
    spec: ExperimentSpec,
    candidate: CandidateSpec,
    baseline_ce: Mapping[float, float],
    runner: Callable[[AllocationConfig], AllocationResult],
) -> AblationRow:
    """Simulate one candidate and apply the delta0*modules adoption gate."""
    config = _arm_config(spec, candidate.policy)
    wealths = _wealth_vector(spec, config, runner)
    ce = {gamma: certainty_equivalent(wealths, gamma=gamma) for gamma in _CE_GAMMAS}
    ce_ratio = {gamma: ce[gamma] / baseline_ce[gamma] for gamma in _CE_GAMMAS}
    adopted = adoption_passes(ce, baseline_ce, delta0=spec.delta0, modules=candidate.modules)
    return AblationRow(
        candidate_id=candidate.id,
        policy=candidate.policy,
        modules=candidate.modules,
        wealths=wealths,
        ce=ce,
        ce_ratio=ce_ratio,
        adopted=adopted,
    )


def run_ablation(
    spec: ExperimentSpec,
    runner: Callable[[AllocationConfig], AllocationResult],
) -> AblationReport:
    """Run the baseline then each candidate in JSON order on identical cashflows.

    The runner is injected once per arm (per cohort when ``horizon_months > 0``),
    keeping the ablation linear in arms times sessions; ``spec`` is never mutated.

    Raises:
        ValueError: When any wealth vector is empty or non-positive.
    """
    baseline_config = _arm_config(spec, spec.baseline.policy)
    baseline_wealths = _wealth_vector(spec, baseline_config, runner)
    baseline_ce = {
        gamma: certainty_equivalent(baseline_wealths, gamma=gamma) for gamma in _CE_GAMMAS
    }
    rows = tuple(_gated_row(spec, candidate, baseline_ce, runner) for candidate in spec.candidates)
    return AblationReport(
        name=spec.name,
        baseline_policy=spec.baseline.policy,
        baseline_modules=spec.baseline.modules,
        baseline_wealths=baseline_wealths,
        baseline_ce=baseline_ce,
        rows=rows,
    )
