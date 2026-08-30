"""Cost-grid walk-forward campaign."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from src.validation.experiment import ExperimentSpec
from src.validation.walk_forward import _fold_records, run_walk_forward_adoption

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.data.settings import DataSettings
    from src.sim.allocation import AllocationConfig, AllocationResult
    from src.validation.walk_forward import CampaignReport

__all__ = [
    "COST_SCENARIOS",
    "CostGridReport",
    "CostScenario",
    "ScenarioOutcome",
    "run_walk_forward_cost_grid",
    "write_cost_grid_report",
]


@dataclass(frozen=True, slots=True)
class CostScenario:
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
    scenario: CostScenario
    campaign: CampaignReport


@dataclass(frozen=True, slots=True)
class CostGridReport:
    name: str
    outcomes: tuple[ScenarioOutcome, ...]

    @property
    def all_scenarios_adopted(self) -> bool:
        return all(outcome.campaign.process_adopted_vs_baseline for outcome in self.outcomes)


def run_walk_forward_cost_grid(
    spec: ExperimentSpec,
    runner: Callable[[AllocationConfig], AllocationResult],
    scenarios: tuple[CostScenario, ...] = COST_SCENARIOS,
) -> CostGridReport:
    outcomes = [
        ScenarioOutcome(
            scenario=scenario,
            campaign=run_walk_forward_adoption(
                spec.model_copy(update={"commission_bps": scenario.commission_bps, "fx_spread_bps": scenario.fx_spread_bps}),
                runner,
            ),
        )
        for scenario in scenarios
    ]
    return CostGridReport(name=spec.name, outcomes=tuple(outcomes))


def write_cost_grid_report(report: CostGridReport, settings: DataSettings, experiment_id: str) -> Path:
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
    from src.data.paths import experiments_dir

    out_dir = experiments_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.name}_costs_{experiment_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
