"""Cadence robustness adoption gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.validation.cost_grid import CostGridReport, run_walk_forward_cost_grid
from src.validation.evaluate import evaluate_cohort_wealths
from src.validation.experiment import ExperimentSpec, resolve_arm_targets, resolve_cadence
from src.validation.gate import bootstrap_tail_passes, worst_cohort_passes
from src.validation.walk_forward import _arm_config
from src.validation.windows import rolling_cohorts

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.data.settings import DataSettings
    from src.sim.allocation import AllocationConfig, AllocationResult

__all__ = [
    "CadenceRobustnessReport",
    "run_cadence_robustness",
    "write_cadence_robustness_report",
]


@dataclass(frozen=True, slots=True)
class CadenceRobustnessReport:
    name: str
    cost_grid: CostGridReport
    baseline_wealths: tuple[float, ...]
    candidate_wealths: tuple[float, ...]
    worst_cohort_ok: bool
    bootstrap_tail_ok: bool
    robust_adopted: bool


def run_cadence_robustness(
    spec: ExperimentSpec,
    runner: Callable[[AllocationConfig], AllocationResult],
    *,
    n_paths: int,
    seed: int,
    horizon_months: int = 36,
    step_months: int = 12,
) -> CadenceRobustnessReport:
    candidate_cadence = resolve_cadence(spec)
    if spec.objective != "growth_first" or candidate_cadence is None:
        raise ValueError("cadence robustness requires objective 'growth_first' and a resolvable cadence")
    cohorts = rolling_cohorts(spec.start, spec.end, horizon_months=horizon_months, step_months=step_months)
    if not cohorts:
        raise ValueError(f"no rolling cohorts fit [{spec.start.isoformat()}, {spec.end.isoformat()}] with horizon_months={horizon_months}, step_months={step_months}")
    cost_grid = run_walk_forward_cost_grid(spec, runner)
    baseline_template = _arm_config(
        spec, spec.baseline.policy, spec.start, spec.end, None, None, None, None,
        targets_override=resolve_arm_targets(spec.baseline),
    )
    candidate_template = _arm_config(
        spec, spec.candidates[0].policy, spec.start, spec.end, None, None, None, None,
        cadence=candidate_cadence, targets_override=resolve_arm_targets(spec.candidates[0]),
    )
    baseline_wealths = evaluate_cohort_wealths(baseline_template, cohorts, runner)
    candidate_wealths = evaluate_cohort_wealths(candidate_template, cohorts, runner)
    worst_cohort_ok = worst_cohort_passes(candidate_wealths, baseline_wealths)
    bootstrap_tail_ok = bootstrap_tail_passes(candidate_wealths, baseline_wealths, n_paths=n_paths, seed=seed)
    return CadenceRobustnessReport(
        name=spec.name, cost_grid=cost_grid, baseline_wealths=baseline_wealths,
        candidate_wealths=candidate_wealths, worst_cohort_ok=worst_cohort_ok,
        bootstrap_tail_ok=bootstrap_tail_ok,
        robust_adopted=cost_grid.all_scenarios_adopted and worst_cohort_ok and bootstrap_tail_ok,
    )


def write_cadence_robustness_report(report: CadenceRobustnessReport, settings: DataSettings, experiment_id: str) -> Path:
    payload = {
        "name": report.name,
        "experiment_id": experiment_id,
        "robust_adopted": report.robust_adopted,
        "all_scenarios_adopted": report.cost_grid.all_scenarios_adopted,
        "worst_cohort_ok": report.worst_cohort_ok,
        "bootstrap_tail_ok": report.bootstrap_tail_ok,
        "cohort_count": len(report.candidate_wealths),
    }
    from src.data.paths import experiments_dir

    out_dir = experiments_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.name}_robustness_{experiment_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
