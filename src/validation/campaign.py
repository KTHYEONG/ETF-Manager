"""Facade re-exporting walk-forward campaign splits."""

from __future__ import annotations

from src.validation.cadence_robustness import (
    CadenceRobustnessReport,
    run_cadence_robustness,
    write_cadence_robustness_report,
)
from src.validation.cost_grid import (
    COST_SCENARIOS,
    CostGridReport,
    CostScenario,
    ScenarioOutcome,
    run_walk_forward_cost_grid,
    write_cost_grid_report,
)
from src.validation.gate import contribution_growth_process_passes
from src.validation.walk_forward import (
    _CE_GAMMAS,
    CampaignReport,
    FoldOutcome,
    _arm_config,
    _fold_records,
    _real_profit,
    _singleton_ce,
    run_walk_forward_adoption,
    run_walk_forward_proxy_adoption,
    warm_baseline_arm_cache,
    write_campaign_report,
)

__all__ = [
    "COST_SCENARIOS",
    "_CE_GAMMAS",
    "CadenceRobustnessReport",
    "CampaignReport",
    "CostGridReport",
    "CostScenario",
    "FoldOutcome",
    "ScenarioOutcome",
    "_arm_config",
    "_fold_records",
    "_real_profit",
    "_singleton_ce",
    "contribution_growth_process_passes",
    "run_cadence_robustness",
    "run_walk_forward_adoption",
    "run_walk_forward_cost_grid",
    "run_walk_forward_proxy_adoption",
    "warm_baseline_arm_cache",
    "write_cadence_robustness_report",
    "write_campaign_report",
    "write_cost_grid_report",
]
