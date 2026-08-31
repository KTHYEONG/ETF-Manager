
"""Unified walk-forward tournament strategy selection."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Final

import polars as pl

from src.data.calendar import DEFAULT_CALENDAR_NAME, load_calendar
from src.data.catalog import latest_artifact, load_visible
from src.data.schedule import build_decision_schedule
from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.sim.allocation import AllocationConfig, AllocationResult, run_allocation
from src.validation.campaign import warm_baseline_arm_cache
from src.validation.experiment import (
    ExperimentSpec,
    resolve_adaptive_contribution,
    resolve_arm_targets,
    resolve_cadence,
    resolve_contribution_shape,
    resolve_currency,
    resolve_kafi_deployment,
    resolve_mapping,
    resolve_overlay,
    resolve_reserve,
)
from src.validation.walk_forward import CampaignReport, run_walk_forward_adoption

__all__ = [
    "SelectionAllocationContext",
    "StrategyArmRow",
    "StrategySelectionReport",
    "StrategyVerdict",
    "make_selection_runner",
    "preload_selection_context",
    "run_strategy_selection",
    "select_recommended_arm",
    "write_strategy_selection_report",
]

_DEFAULT_MAX_WF_EVALUATIONS: Final[int] = 12


class StrategyVerdict(StrEnum):
    OOS_ELIGIBLE = "oos_eligible"
    RESEARCH_ONLY = "research_only"


@dataclass(frozen=True, slots=True)
class StrategyArmRow:
    arm_id: str
    process_adopted_vs_baseline: bool
    pooled_oos_real_gain: float
    pooled_oos_tw_ratio: float
    in_sample_real_gain: float | None
    verdict: StrategyVerdict
    fold_count: int


@dataclass(frozen=True, slots=True)
class StrategySelectionReport:
    name: str
    baseline_arm_id: str
    objective: str
    rows: tuple[StrategyArmRow, ...]
    in_sample_champion_arm_id: str | None
    oos_eligible_arm_ids: tuple[str, ...]
    recommended_arm_id: str
    operational_unlock: bool
    selection_reason: str


@dataclass(frozen=True, slots=True)
class SelectionAllocationContext:
    prices: pl.DataFrame
    fx: pl.DataFrame
    cpi: pl.DataFrame
    macro: pl.DataFrame


def select_recommended_arm(rows: Sequence[StrategyArmRow], *, baseline_arm_id: str) -> tuple[str, str]:
    eligible = [r for r in rows if r.process_adopted_vs_baseline]
    if not eligible:
        return baseline_arm_id, f"no_oos_eligible fallback to baseline {baseline_arm_id}"
    best = eligible[0]
    for cand in eligible[1:]:
        if float(cand.pooled_oos_real_gain) > float(best.pooled_oos_real_gain):
            best = cand
    return best.arm_id, f"oos_eligible max_gain {best.arm_id} pooled_gain={float(best.pooled_oos_real_gain):.6f}"


def preload_selection_context(settings: DataSettings, *, start: date, end: date) -> SelectionAllocationContext:
    for dataset in (Dataset.PRICES, Dataset.FX, Dataset.CPI, Dataset.MACRO):
        latest_artifact(settings, dataset)
    schedule = build_decision_schedule(start, end, frequency="monthly", fill_delay_sessions=1)
    if not schedule:
        raise ValueError(f"empty decision schedule over [{start.isoformat()}, {end.isoformat()}]")
    cutoff = load_calendar(DEFAULT_CALENDAR_NAME).close_ts(schedule[-1].execution_session)
    return SelectionAllocationContext(
        prices=load_visible(settings, Dataset.PRICES, cutoff),
        fx=load_visible(settings, Dataset.FX, cutoff),
        cpi=load_visible(settings, Dataset.CPI, cutoff),
        macro=load_visible(settings, Dataset.MACRO, cutoff),
    )


def make_selection_runner(settings: DataSettings, template: ExperimentSpec) -> Callable[[ExperimentSpec], CampaignReport]:
    context = preload_selection_context(settings, start=template.start, end=template.end)

    def _allocation_runner(config: AllocationConfig) -> AllocationResult:
        return run_allocation(config, context.prices, context.fx, context.cpi, macro=context.macro)

    baseline_cache = warm_baseline_arm_cache(template, _allocation_runner)

    def _wf_runner(spec: ExperimentSpec) -> CampaignReport:
        return run_walk_forward_adoption(spec, _allocation_runner, baseline_arm_cache=baseline_cache)

    return _wf_runner


def _build_in_sample_config(spec: ExperimentSpec, candidate_id: str) -> AllocationConfig:
    cand = next((c for c in spec.candidates if c.id == candidate_id), None)
    if cand is None:
        raise ValueError(f"candidate {candidate_id!r} not found")
    overlay = resolve_overlay(spec)
    reserve = resolve_reserve(spec)
    mapping = resolve_mapping(spec)
    currency = resolve_currency(spec)
    contribution_shape = resolve_contribution_shape(spec)
    kafi_deployment = resolve_kafi_deployment(spec)
    adaptive = resolve_adaptive_contribution(spec)
    cadence = resolve_cadence(spec) or "monthly"
    targets = resolve_arm_targets(cand)
    return AllocationConfig(
        policy=cand.policy,
        start=spec.start,
        end=spec.end,
        monthly_contribution_krw=float(spec.contribution_krw),
        fill_delay_sessions=1,
        fx_spread_bps=float(spec.fx_spread_bps),
        commission_bps=float(spec.commission_bps),
        overlay=overlay,
        reserve=reserve,
        currency=currency,
        mapping=mapping,
        contribution_shape=contribution_shape,
        kafi_deployment=kafi_deployment,
        adaptive_contribution=adaptive,
        cadence=cadence,
        targets_override=targets,
    )


def _pooled_metrics(report: CampaignReport) -> tuple[float, float]:
    if not report.folds:
        return 0.0, 1.0
    chosen_gain = sum(float(f.chosen_real_gain) for f in report.folds)
    tw_ratio = 1.0
    try:
        chosen_tw = sum(float(f.chosen_test_wealth) for f in report.folds)
        baseline_tw = sum(float(f.baseline_test_wealth) for f in report.folds)
        if baseline_tw > 0 and math.isfinite(chosen_tw) and math.isfinite(baseline_tw):
            tw_ratio = chosen_tw / baseline_tw
    except Exception:
        tw_ratio = 1.0
    return float(chosen_gain), float(tw_ratio)


def run_strategy_selection(
    spec: ExperimentSpec,
    runner: Callable[[ExperimentSpec], CampaignReport],
    *,
    in_sample_runner: Callable[[AllocationConfig], AllocationResult] | None = None,
    max_wf_evaluations: int | None = None,
    parallel_workers: int | None = None,
) -> StrategySelectionReport:
    max_evals = _DEFAULT_MAX_WF_EVALUATIONS if max_wf_evaluations is None else int(max_wf_evaluations)
    if max_evals < 1:
        raise ValueError(f"max_wf_evaluations must be >=1, got {max_wf_evaluations!r}")
    if parallel_workers is not None and int(parallel_workers) < 1:
        raise ValueError(f"parallel_workers must be >=1, got {parallel_workers!r}")
    candidates = list(spec.candidates)
    if not candidates:
        raise ValueError("spec must have at least one candidate")
    # in-sample gains disclosure
    in_sample_gains: dict[str, float | None] = {c.id: None for c in candidates}
    if in_sample_runner is not None:
        for cand in candidates:
            cfg = _build_in_sample_config(spec, cand.id)
            res = in_sample_runner(cfg)
            gain = float(res.terminal_wealth_real_krw) - float(res.total_contribution_real_krw)
            in_sample_gains[cand.id] = float(gain)
    # Phase-A screening
    kept_ids: set[str]
    dropped_ids: set[str] = set()
    if len(candidates) > max_evals:
        if in_sample_runner is not None:
            # sort by gain descending, keep top N, preserve ties by prereg order? Use stable sort.
            # Build list sorted by gain; candidates with None gain go last
            def _gain_key(c) -> float:  # type: ignore[no-untyped-def]
                g = in_sample_gains.get(c.id)
                return float(g) if g is not None and math.isfinite(float(g)) else float("-inf")
            sorted_cands = sorted(candidates, key=_gain_key, reverse=True)
            kept = sorted_cands[:max_evals]
            kept_ids = {c.id for c in kept}
            dropped_ids = {c.id for c in candidates if c.id not in kept_ids}
        else:
            kept_ids = {c.id for c in candidates[:max_evals]}
            dropped_ids = {c.id for c in candidates[max_evals:]}
    else:
        kept_ids = {c.id for c in candidates}
    # Run WF for kept
    reports: dict[str, CampaignReport] = {}
    kept_candidates = [c for c in candidates if c.id in kept_ids]
    # preserve prereg order for kept_candidates (already)
    def _run_one(cand) -> tuple[str, CampaignReport]:  # type: ignore[no-untyped-def]
        clone = spec.model_copy(update={"candidates": [cand]})
        rep = runner(clone)
        return cand.id, rep

    if parallel_workers is not None and int(parallel_workers) > 1 and len(kept_candidates) > 1:
        workers = int(parallel_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_run_one, cand): cand.id for cand in kept_candidates}
            for fut in as_completed(future_map):
                cid, rep = fut.result()
                reports[cid] = rep
    else:
        for cand in kept_candidates:
            cid, rep = _run_one(cand)
            reports[cid] = rep

    rows: list[StrategyArmRow] = []
    for cand in candidates:
        cid = cand.id
        if cid in dropped_ids:
            gain_is = in_sample_gains.get(cid)
            rows.append(
                StrategyArmRow(
                    arm_id=cid,
                    process_adopted_vs_baseline=False,
                    pooled_oos_real_gain=0.0,
                    pooled_oos_tw_ratio=1.0,
                    in_sample_real_gain=gain_is,
                    verdict=StrategyVerdict.RESEARCH_ONLY,
                    fold_count=0,
                )
            )
        else:
            rep_opt = reports.get(cid)
            if rep_opt is None:
                raise ValueError(f"missing WF report for {cid!r}")
            rep = rep_opt
            pooled_gain, tw_ratio = _pooled_metrics(rep)
            adopted = bool(rep.process_adopted_vs_baseline)
            verdict = StrategyVerdict.OOS_ELIGIBLE if adopted else StrategyVerdict.RESEARCH_ONLY
            gain_is = in_sample_gains.get(cid)
            rows.append(
                StrategyArmRow(
                    arm_id=cid,
                    process_adopted_vs_baseline=adopted,
                    pooled_oos_real_gain=float(pooled_gain),
                    pooled_oos_tw_ratio=float(tw_ratio),
                    in_sample_real_gain=gain_is,
                    verdict=verdict,
                    fold_count=len(rep.folds),
                )
            )
    # disclosure champion
    in_sample_champion: str | None = None
    if in_sample_runner is not None:
        # max in_sample_real_gain
        best_id = None
        best_gain = float("-inf")
        for cand in candidates:
            g = in_sample_gains.get(cand.id)
            if g is not None and math.isfinite(float(g)) and float(g) > best_gain:
                best_gain = float(g)
                best_id = cand.id
        in_sample_champion = best_id
    oos_eligible_ids = tuple(r.arm_id for r in rows if r.process_adopted_vs_baseline)
    recommended_id, reason = select_recommended_arm(tuple(rows), baseline_arm_id=spec.baseline.id)
    return StrategySelectionReport(
        name=spec.name,
        baseline_arm_id=spec.baseline.id,
        objective=spec.objective,
        rows=tuple(rows),
        in_sample_champion_arm_id=in_sample_champion,
        oos_eligible_arm_ids=oos_eligible_ids,
        recommended_arm_id=recommended_id,
        operational_unlock=False,
        selection_reason=reason,
    )


def write_strategy_selection_report(report: StrategySelectionReport, settings: DataSettings, experiment_id: str) -> Path:
    payload = {
        "name": report.name,
        "experiment_id": experiment_id,
        "baseline_arm_id": report.baseline_arm_id,
        "objective": report.objective,
        "operational_unlock": report.operational_unlock,
        "recommended_arm_id": report.recommended_arm_id,
        "selection_reason": report.selection_reason,
        "in_sample_champion_arm_id": report.in_sample_champion_arm_id,
        "oos_eligible_arm_ids": list(report.oos_eligible_arm_ids),
        "rows": [
            {
                "arm_id": r.arm_id,
                "process_adopted_vs_baseline": r.process_adopted_vs_baseline,
                "pooled_oos_real_gain": r.pooled_oos_real_gain,
                "pooled_oos_tw_ratio": r.pooled_oos_tw_ratio,
                "in_sample_real_gain": r.in_sample_real_gain,
                "verdict": str(r.verdict),
                "fold_count": r.fold_count,
            }
            for r in report.rows
        ],
    }
    from src.data.paths import experiments_dir

    out_dir = experiments_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.name}_{experiment_id}_selection.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
