"""Research-convergence kernel (Phase A)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from enum import StrEnum
from typing import Final, Literal

if False:  # type checking only
    from src.sim.allocation import AllocationResult

__all__ = [
    "ECONOMIC_CE_GAMMA10_FLOOR",
    "ECONOMIC_MEDIAN_RATIO_FLOOR",
    "SEEN_HISTORY_CUTOFF",
    "ObjectiveFamily",
    "StrategyRole",
    "assert_objective_family_invariants",
    "assert_prospective_observation",
    "classify_strategy_role",
    "economic_effect_passes",
    "is_seen_history",
    "observation_epoch",
    "select_chosen_test_arm",
]

SEEN_HISTORY_CUTOFF: Final[date] = date(2026, 8, 28)
ECONOMIC_MEDIAN_RATIO_FLOOR: Final[float] = 1.01
ECONOMIC_CE_GAMMA10_FLOOR: Final[float] = 1.0


class StrategyRole(StrEnum):
    IMMUTABLE_BENCHMARK = "immutable_benchmark"
    PROVISIONAL_INCUMBENT = "provisional_incumbent"
    CONSERVATIVE_CHALLENGER = "conservative_challenger"
    AGGRESSIVE_CHALLENGER = "aggressive_challenger"
    FROZEN_RESEARCH = "frozen_research"
    REJECTED_VEHICLE = "rejected_vehicle"
    PROSPECTIVE_WATCH = "prospective_watch"


class ObjectiveFamily(StrEnum):
    CAPITAL_ALLOCATION = "capital_allocation"
    DEPLOYMENT_TIMING = "deployment_timing"


def select_chosen_test_arm(
    *,
    train_adopted: bool,
    candidate_test_arm: AllocationResult,
    baseline_test_arm: AllocationResult,
) -> AllocationResult:
    return candidate_test_arm if train_adopted else baseline_test_arm


def economic_effect_passes(
    *,
    median_ratio: float,
    ce_gamma_10: float,
    bootstrap_ok: bool,
) -> bool:
    if not math.isfinite(float(median_ratio)) or not math.isfinite(float(ce_gamma_10)):
        return False
    if float(median_ratio) < ECONOMIC_MEDIAN_RATIO_FLOOR:
        return False
    if float(ce_gamma_10) < ECONOMIC_CE_GAMMA10_FLOOR:
        return False
    return bool(bootstrap_ok)


def is_seen_history(as_of: date) -> bool:
    return as_of <= SEEN_HISTORY_CUTOFF


def observation_epoch(as_of: date) -> Literal["seen_history", "prospective_oos"]:
    return "seen_history" if is_seen_history(as_of) else "prospective_oos"


def assert_prospective_observation(as_of: date) -> None:
    if is_seen_history(as_of):
        raise ValueError(f"seen_history: {as_of.isoformat()} <= {SEEN_HISTORY_CUTOFF.isoformat()}")


def classify_strategy_role(
    *,
    targets: Mapping[str, float],
    adaptive: bool,
) -> StrategyRole:
    if adaptive:
        return StrategyRole.FROZEN_RESEARCH
    normalized: dict[str, float] = {}
    for k, v in targets.items():
        key = str(k).strip().upper()
        if not key:
            continue
        normalized[key] = float(v)
    # positive check: weight > 0 (allow tolerance)
    def _has_positive(ticker: str) -> bool:
        w = normalized.get(ticker)
        return w is not None and float(w) > 0.0

    if _has_positive("ROBO") or _has_positive("BOTZ"):
        return StrategyRole.REJECTED_VEHICLE
    if _has_positive("PAVE") or _has_positive("GRID"):
        return StrategyRole.PROSPECTIVE_WATCH
    # Check QQQ mixes
    # exact matches with tolerance 1e-9
    def _is_close(a: float, b: float) -> bool:
        return abs(float(a) - float(b)) <= 1e-9

    if len(normalized) == 1 and "QQQ" in normalized and _is_close(normalized["QQQ"], 1.0):
        return StrategyRole.IMMUTABLE_BENCHMARK
    if len(normalized) == 2 and "QQQ" in normalized and "SOXX" in normalized:
        q = float(normalized["QQQ"])
        s = float(normalized["SOXX"])
        if _is_close(q, 0.9) and _is_close(s, 0.1):
            return StrategyRole.PROVISIONAL_INCUMBENT
        if _is_close(q, 0.85) and _is_close(s, 0.15):
            return StrategyRole.AGGRESSIVE_CHALLENGER
        if _is_close(q, 0.95) and _is_close(s, 0.05):
            return StrategyRole.CONSERVATIVE_CHALLENGER
    raise ValueError(f"unregistered mix {dict(targets)!r}")


def assert_objective_family_invariants(
    *,
    family: ObjectiveFamily,
    adaptive_contribution_set: bool,
    baseline_adaptive_set: bool,
    kafi_deployment_set: bool,
    reserve_set: bool,
    contribution_shape_set: bool = False,
) -> None:
    if family is ObjectiveFamily.CAPITAL_ALLOCATION:
        if kafi_deployment_set:
            raise ValueError("capital_allocation: kafi_deployment not allowed for capital_allocation")
        if reserve_set:
            raise ValueError("capital_allocation: reserve not allowed for capital_allocation")
        if contribution_shape_set:
            raise ValueError("capital_allocation: contribution_shape not allowed for capital_allocation")
    if adaptive_contribution_set or baseline_adaptive_set:
        raise ValueError("adaptive_contribution not allowed for objective_family")
    if family is ObjectiveFamily.DEPLOYMENT_TIMING and not (
        kafi_deployment_set or reserve_set
    ):
        raise ValueError("deployment_timing requires kafi_deployment or reserve")
