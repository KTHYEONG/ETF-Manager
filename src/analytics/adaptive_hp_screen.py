"""Deterministic adaptive-contribution HP neighbourhood screen; reporting only."""

from __future__ import annotations

import itertools
import math
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import polars as pl

from src.data.calendar import DEFAULT_CALENDAR_NAME, load_calendar
from src.data.catalog import latest_artifact, load_visible
from src.data.schedule import build_decision_schedule
from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.policy.adaptive_contribution import (
    OPERATIONAL_ADAPTIVE_CONTRIBUTION,
    AdaptiveContributionConfig,
)
from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig, AllocationResult, run_allocation
from src.validation.campaign import run_walk_forward_adoption, warm_baseline_arm_cache
from src.validation.experiment import (
    AdaptiveContributionSpec,
    CandidateSpec,
    ExperimentSpec,
)

if TYPE_CHECKING:
    from src.validation.campaign import CampaignReport

__all__ = [
    "AdaptiveHpScreenReport",
    "HpAllocationContext",
    "HpArmRow",
    "HpArmVerdict",
    "build_adaptive_hp_experiment",
    "make_hp_wf_runner",
    "preload_hp_allocation_context",
    "screen_adaptive_contribution_hp",
]

_DEFAULT_START: Final[date] = date(2015, 6, 1)
_DEFAULT_END: Final[date] = date(2024, 8, 31)
_DEFAULT_HURDLE: Final[float] = 0.02
_DEFAULT_TRAIN_MONTHS: Final[int] = 60
_DEFAULT_TEST_MONTHS: Final[int] = 36

_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "downside_power": (2.0, 5.0),
    "upside_power": (0.15, 0.70),
    "dispersion": (0.90, 1.60),
    "neutral_deadband": (0.0, 8.0),
}
_DELTAS: Final[dict[str, float]] = {
    "downside_power": 0.25,
    "upside_power": 0.05,
    "dispersion": 0.075,
    "neutral_deadband": 1.0,
}
_LOCK_RANK_WINDOW: Final[int] = 126
_LOCK_INCLUDE_VOL: Final[bool] = False
_LOCK_MIN_MULT: Final[float] = 0.0
_LOCK_MAX_MULT: Final[float] = 2.0
_BASELINE_POLICY: Final[PolicyId] = PolicyId.QQQ


class HpArmVerdict(StrEnum):
    """Reporting-only verdict for one screened arm."""

    ADOPT_CANDIDATE = "adopt_candidate"
    RESEARCH_ONLY = "research_only"


@dataclass(frozen=True, slots=True)
class HpArmRow:
    """Outcome of one candidate arm versus the locked v5 baseline."""

    candidate: AdaptiveContributionConfig
    baseline: AdaptiveContributionConfig
    process_adopted_vs_baseline: bool
    pooled_tw_ratio: float
    verdict: HpArmVerdict

    @property
    def downside_power(self) -> float:
        return self.candidate.downside_power

    @property
    def upside_power(self) -> float:
        return self.candidate.upside_power

    @property
    def dispersion(self) -> float:
        return self.candidate.dispersion

    @property
    def neutral_deadband(self) -> float:
        return self.candidate.neutral_deadband

    @property
    def rank_window(self) -> int:
        return self.candidate.rank_window

    @property
    def include_vol_dampener(self) -> bool:
        return self.candidate.include_vol_dampener

    @property
    def min_multiplier(self) -> float:
        return self.candidate.min_multiplier

    @property
    def max_multiplier(self) -> float:
        return self.candidate.max_multiplier


@dataclass(frozen=True, slots=True)
class AdaptiveHpScreenReport:
    """Full HP screen output; operational_unlock is always False."""

    operational_unlock: bool
    champion: HpArmRow | None
    rows: tuple[HpArmRow, ...]
    evaluations: int


@dataclass(frozen=True, slots=True)
class HpAllocationContext:
    """Preloaded PIT datasets reused across HP walk-forward evaluations."""

    prices: pl.DataFrame
    fx: pl.DataFrame
    cpi: pl.DataFrame
    macro: pl.DataFrame


def _clip(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def _validate_contribution_krw(value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"contribution_krw must be finite and positive, got {value!r}")


def _candidate_key(cfg: AdaptiveContributionConfig) -> tuple[float, float, float, float]:
    return (
        round(float(cfg.downside_power), 4),
        round(float(cfg.upside_power), 4),
        round(float(cfg.dispersion), 4),
        round(float(cfg.neutral_deadband), 4),
    )


def preload_hp_allocation_context(
    settings: DataSettings,
    *,
    start: date | None = None,
    end: date | None = None,
) -> HpAllocationContext:
    """Load catalog partitions once for the HP screen window."""
    eff_start = start if start is not None else _DEFAULT_START
    eff_end = end if end is not None else _DEFAULT_END
    for dataset in (Dataset.PRICES, Dataset.FX, Dataset.CPI, Dataset.MACRO):
        latest_artifact(settings, dataset)
    schedule = build_decision_schedule(eff_start, eff_end, frequency="monthly", fill_delay_sessions=1)
    if not schedule:
        raise ValueError(f"empty decision schedule over [{eff_start.isoformat()}, {eff_end.isoformat()}]")
    cutoff = load_calendar(DEFAULT_CALENDAR_NAME).close_ts(schedule[-1].execution_session)
    return HpAllocationContext(
        prices=load_visible(settings, Dataset.PRICES, cutoff),
        fx=load_visible(settings, Dataset.FX, cutoff),
        cpi=load_visible(settings, Dataset.CPI, cutoff),
        macro=load_visible(settings, Dataset.MACRO, cutoff),
    )


def make_hp_allocation_runner(context: HpAllocationContext) -> Callable[[AllocationConfig], AllocationResult]:
    """Return an allocation runner that reuses preloaded datasets."""

    def _run(config: AllocationConfig) -> AllocationResult:
        return run_allocation(
            config,
            context.prices,
            context.fx,
            context.cpi,
            macro=context.macro,
        )

    return _run


def make_hp_wf_runner(
    settings: DataSettings,
    *,
    contribution_krw: float,
    start: date | None = None,
    end: date | None = None,
) -> Callable[[ExperimentSpec], CampaignReport]:
    """Build a cached walk-forward runner for HP screening."""
    _validate_contribution_krw(contribution_krw)
    context = preload_hp_allocation_context(settings, start=start, end=end)
    allocation_runner = make_hp_allocation_runner(context)
    template = build_adaptive_hp_experiment(
        name="hp_template",
        candidate=OPERATIONAL_ADAPTIVE_CONTRIBUTION,
        baseline=OPERATIONAL_ADAPTIVE_CONTRIBUTION,
        contribution_krw=float(contribution_krw),
        start=start,
        end=end,
    )
    baseline_cache = warm_baseline_arm_cache(template, allocation_runner)

    def _wf_runner(spec: ExperimentSpec) -> CampaignReport:
        return run_walk_forward_adoption(
            spec,
            allocation_runner,
            baseline_arm_cache=baseline_cache,
        )

    return _wf_runner


def _to_spec_module(cfg: AdaptiveContributionConfig) -> AdaptiveContributionSpec:
    return AdaptiveContributionSpec(
        equity_ticker=cfg.equity_ticker,
        bond_ticker=cfg.bond_ticker,
        credit_series_id=cfg.credit_series_id,
        min_multiplier=cfg.min_multiplier,
        max_multiplier=cfg.max_multiplier,
        downside_power=cfg.downside_power,
        upside_power=cfg.upside_power,
        rank_window=cfg.rank_window,
        include_vol_dampener=cfg.include_vol_dampener,
        dispersion=cfg.dispersion,
        neutral_deadband=cfg.neutral_deadband,
    )


def build_adaptive_hp_experiment(
    *,
    name: str,
    candidate: AdaptiveContributionConfig,
    baseline: AdaptiveContributionConfig,
    contribution_krw: float,
    start: date | None = None,
    end: date | None = None,
) -> ExperimentSpec:
    """Build a walk-forward adaptive_growth spec for one candidate vs baseline."""
    _validate_contribution_krw(contribution_krw)
    if not name:
        raise ValueError("name must be non-empty")
    for cfg, label in ((candidate, "candidate"), (baseline, "baseline")):
        if cfg.rank_window != _LOCK_RANK_WINDOW:
            raise ValueError(f"{label} rank_window must be {_LOCK_RANK_WINDOW}, got {cfg.rank_window!r}")
        if cfg.include_vol_dampener is not _LOCK_INCLUDE_VOL:
            raise ValueError(f"{label} include_vol_dampener must be {_LOCK_INCLUDE_VOL}, got {cfg.include_vol_dampener!r}")
        if cfg.min_multiplier != _LOCK_MIN_MULT or cfg.max_multiplier != _LOCK_MAX_MULT:
            raise ValueError(f"{label} min/max multiplier must be {_LOCK_MIN_MULT}/{_LOCK_MAX_MULT}")
    eff_start = start if start is not None else _DEFAULT_START
    eff_end = end if end is not None else _DEFAULT_END
    return ExperimentSpec(
        name=name,
        start=eff_start,
        end=eff_end,
        contribution_krw=float(contribution_krw),
        hurdle=_DEFAULT_HURDLE,
        objective="adaptive_growth",
        horizon_months=0,
        train_months=_DEFAULT_TRAIN_MONTHS,
        test_months=_DEFAULT_TEST_MONTHS,
        baseline=CandidateSpec(id="s8_us_nasdaq_adaptive_ops", policy=_BASELINE_POLICY, modules=1),
        candidates=[CandidateSpec(id="s8_us_nasdaq_adaptive_hp_candidate", policy=_BASELINE_POLICY, modules=1)],
        adaptive_contribution=_to_spec_module(candidate),
        baseline_adaptive_contribution=_to_spec_module(baseline),
    )


def _make_candidate(
    *,
    downside_power: float,
    upside_power: float,
    dispersion: float,
    neutral_deadband: float,
) -> AdaptiveContributionConfig:
    dp = _clip(downside_power, *_BOUNDS["downside_power"])
    up = _clip(upside_power, *_BOUNDS["upside_power"])
    disp = _clip(dispersion, *_BOUNDS["dispersion"])
    dead = _clip(neutral_deadband, *_BOUNDS["neutral_deadband"])
    return AdaptiveContributionConfig(
        rank_window=_LOCK_RANK_WINDOW,
        include_vol_dampener=_LOCK_INCLUDE_VOL,
        min_multiplier=_LOCK_MIN_MULT,
        max_multiplier=_LOCK_MAX_MULT,
        downside_power=float(dp),
        upside_power=float(up),
        dispersion=float(disp),
        neutral_deadband=float(dead),
    )


def _phase_a_candidates() -> list[AdaptiveContributionConfig]:
    anchor = OPERATIONAL_ADAPTIVE_CONTRIBUTION
    per_axis: dict[str, list[float]] = {}
    for axis_key in ("downside_power", "upside_power", "dispersion", "neutral_deadband"):
        a = float(getattr(anchor, axis_key))
        d = _DELTAS[axis_key]
        low, high = _BOUNDS[axis_key]
        vals = [_clip(a - d, low, high), _clip(a, low, high), _clip(a + d, low, high)]
        seen: set[float] = set()
        uniq: list[float] = []
        for v in vals:
            r = round(float(v), 4)
            if r not in seen:
                seen.add(r)
                uniq.append(float(v))
        per_axis[axis_key] = uniq
    combos: list[AdaptiveContributionConfig] = []
    seen_keys: set[tuple[float, float, float, float]] = set()
    for dp, up, disp, dead in itertools.product(
        per_axis["downside_power"],
        per_axis["upside_power"],
        per_axis["dispersion"],
        per_axis["neutral_deadband"],
    ):
        combo_key = (round(float(dp), 4), round(float(up), 4), round(float(disp), 4), round(float(dead), 4))
        if combo_key in seen_keys:
            continue
        seen_keys.add(combo_key)
        combos.append(_make_candidate(downside_power=dp, upside_power=up, dispersion=disp, neutral_deadband=dead))
    anchor_key = _candidate_key(anchor)
    for idx, cfg in enumerate(combos):
        if _candidate_key(cfg) == anchor_key and idx != 0:
            combos.insert(0, combos.pop(idx))
            break
    return combos


def _row_from_report(
    *,
    candidate: AdaptiveContributionConfig,
    baseline: AdaptiveContributionConfig,
    report: CampaignReport,
) -> HpArmRow:
    chosen_sum = sum(float(f.chosen_test_wealth) for f in report.folds)
    baseline_sum = sum(float(f.baseline_test_wealth) for f in report.folds)
    if baseline_sum <= 0.0 or not math.isfinite(chosen_sum) or not math.isfinite(baseline_sum):
        ratio = float("nan") if baseline_sum == 0 else chosen_sum / baseline_sum
    else:
        ratio = chosen_sum / baseline_sum
    process_adopted = bool(report.process_adopted_vs_baseline)
    if not math.isfinite(ratio):
        verdict = HpArmVerdict.RESEARCH_ONLY
    elif process_adopted and ratio > 1.0:
        verdict = HpArmVerdict.ADOPT_CANDIDATE
    else:
        verdict = HpArmVerdict.RESEARCH_ONLY
    return HpArmRow(
        candidate=candidate,
        baseline=baseline,
        process_adopted_vs_baseline=process_adopted,
        pooled_tw_ratio=float(ratio),
        verdict=verdict,
    )


def _evaluate_one(
    candidate: AdaptiveContributionConfig,
    baseline: AdaptiveContributionConfig,
    contribution_krw: float,
    wf_runner: Callable[[ExperimentSpec], CampaignReport],
    seq: int,
) -> HpArmRow:
    spec = build_adaptive_hp_experiment(
        name=f"hp_{seq}",
        candidate=candidate,
        baseline=baseline,
        contribution_krw=contribution_krw,
    )
    report = wf_runner(spec)
    return _row_from_report(candidate=candidate, baseline=baseline, report=report)


def _default_parallel_workers() -> int:
    cpus = os.cpu_count() or 1
    return max(1, min(cpus, 8))


def _evaluate_many(
    *,
    candidates: list[AdaptiveContributionConfig],
    baseline: AdaptiveContributionConfig,
    contribution_krw: float,
    wf_runner: Callable[[ExperimentSpec], CampaignReport],
    parallel_workers: int,
    seq_start: int,
) -> list[HpArmRow]:
    if parallel_workers <= 1 or len(candidates) <= 1:
        return [
            _evaluate_one(candidate, baseline, contribution_krw, wf_runner, seq_start + idx)
            for idx, candidate in enumerate(candidates)
        ]
    rows: list[HpArmRow | None] = [None] * len(candidates)
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        future_map = {
            executor.submit(
                _evaluate_one,
                candidate,
                baseline,
                contribution_krw,
                wf_runner,
                seq_start + idx,
            ): idx
            for idx, candidate in enumerate(candidates)
        }
        for future in as_completed(future_map):
            rows[future_map[future]] = future.result()
    return [row for row in rows if row is not None]


def screen_adaptive_contribution_hp(
    *,
    contribution_krw: float,
    wf_runner: Callable[[ExperimentSpec], CampaignReport],
    max_evaluations: int = 128,
    parallel_workers: int | None = None,
) -> AdaptiveHpScreenReport:
    """Deterministic two-phase HP screen; reporting only, never unlocks."""
    _validate_contribution_krw(contribution_krw)
    if not isinstance(max_evaluations, int) or isinstance(max_evaluations, bool) or max_evaluations < 1:
        raise ValueError(f"max_evaluations must be integer >=1, got {max_evaluations!r}")
    workers = _default_parallel_workers() if parallel_workers is None else parallel_workers
    if workers < 1:
        raise ValueError(f"parallel_workers must be >=1, got {workers!r}")

    baseline = OPERATIONAL_ADAPTIVE_CONTRIBUTION
    rows: list[HpArmRow] = []
    seen_keys: set[tuple[float, float, float, float]] = set()
    evaluations = 0

    def try_evaluate(cfg: AdaptiveContributionConfig) -> bool:
        nonlocal evaluations
        if evaluations >= max_evaluations:
            return False
        key = _candidate_key(cfg)
        if key in seen_keys:
            return False
        seen_keys.add(key)
        row = _evaluate_one(cfg, baseline, float(contribution_krw), wf_runner, evaluations)
        rows.append(row)
        evaluations += 1
        return True

    phase_a = _phase_a_candidates()
    phase_a_budget = min(len(phase_a), max_evaluations)
    phase_a_batch = phase_a[:phase_a_budget]
    unseen_phase_a = [cfg for cfg in phase_a_batch if _candidate_key(cfg) not in seen_keys]
    if unseen_phase_a:
        if workers > 1 and len(unseen_phase_a) > 1:
            batch_rows = _evaluate_many(
                candidates=unseen_phase_a,
                baseline=baseline,
                contribution_krw=float(contribution_krw),
                wf_runner=wf_runner,
                parallel_workers=workers,
                seq_start=evaluations,
            )
            for cfg in unseen_phase_a:
                seen_keys.add(_candidate_key(cfg))
            rows.extend(batch_rows)
            evaluations += len(batch_rows)
        else:
            for cfg in phase_a_batch:
                if evaluations >= max_evaluations:
                    break
                try_evaluate(cfg)

    adopt_rows = [
        r
        for r in rows
        if r.process_adopted_vs_baseline and math.isfinite(r.pooled_tw_ratio) and r.pooled_tw_ratio > 1.0
    ]
    if adopt_rows:
        max_ratio = max(r.pooled_tw_ratio for r in adopt_rows)
        tied = [r for r in adopt_rows if r.pooled_tw_ratio == max_ratio]
        tied.sort(key=lambda r: (r.downside_power, r.upside_power, r.dispersion, r.neutral_deadband))
        best = tied[0]
        current_best_cfg = best.candidate
        current_best_ratio = best.pooled_tw_ratio
    else:
        current_best_cfg = baseline
        v5_rows = [r for r in rows if r.candidate == baseline]
        if v5_rows:
            current_best_ratio = v5_rows[0].pooled_tw_ratio if math.isfinite(v5_rows[0].pooled_tw_ratio) else 1.0
        else:
            current_best_ratio = 1.0

    if not adopt_rows:
        return AdaptiveHpScreenReport(
            operational_unlock=False,
            champion=None,
            rows=tuple(rows),
            evaluations=evaluations,
        )

    steps = dict(_DELTAS)
    cur_down = float(current_best_cfg.downside_power)
    cur_up = float(current_best_cfg.upside_power)
    cur_disp = float(current_best_cfg.dispersion)
    cur_dead = float(current_best_cfg.neutral_deadband)

    for rnd in range(3):
        if evaluations >= max_evaluations:
            break
        improved = False
        factor = 0.5**rnd
        axes = [
            ("downside_power", cur_down, _BOUNDS["downside_power"][0], _BOUNDS["downside_power"][1], steps["downside_power"] * factor),
            ("upside_power", cur_up, _BOUNDS["upside_power"][0], _BOUNDS["upside_power"][1], steps["upside_power"] * factor),
            ("dispersion", cur_disp, _BOUNDS["dispersion"][0], _BOUNDS["dispersion"][1], steps["dispersion"] * factor),
            ("neutral_deadband", cur_dead, _BOUNDS["neutral_deadband"][0], _BOUNDS["neutral_deadband"][1], steps["neutral_deadband"] * factor),
        ]
        for axis_name, cur_val, low, high, step in axes:
            if evaluations >= max_evaluations:
                break
            candidates_to_try: list[AdaptiveContributionConfig] = []
            for direction in (1, -1):
                new_val = _clip(cur_val + direction * step, low, high)
                if round(float(new_val), 4) == round(float(cur_val), 4):
                    continue
                if axis_name == "downside_power":
                    cfg = _make_candidate(downside_power=new_val, upside_power=cur_up, dispersion=cur_disp, neutral_deadband=cur_dead)
                elif axis_name == "upside_power":
                    cfg = _make_candidate(downside_power=cur_down, upside_power=new_val, dispersion=cur_disp, neutral_deadband=cur_dead)
                elif axis_name == "dispersion":
                    cfg = _make_candidate(downside_power=cur_down, upside_power=cur_up, dispersion=new_val, neutral_deadband=cur_dead)
                else:
                    cfg = _make_candidate(downside_power=cur_down, upside_power=cur_up, dispersion=cur_disp, neutral_deadband=new_val)
                if _candidate_key(cfg) in seen_keys:
                    continue
                candidates_to_try.append(cfg)
            best_axis_cfg: AdaptiveContributionConfig | None = None
            best_axis_ratio = current_best_ratio
            for cfg in candidates_to_try:
                if evaluations >= max_evaluations:
                    break
                key = _candidate_key(cfg)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                row = _evaluate_one(cfg, baseline, float(contribution_krw), wf_runner, evaluations)
                rows.append(row)
                evaluations += 1
                if row.process_adopted_vs_baseline and math.isfinite(row.pooled_tw_ratio) and row.pooled_tw_ratio > best_axis_ratio:
                    best_axis_cfg = cfg
                    best_axis_ratio = row.pooled_tw_ratio
            if best_axis_cfg is not None:
                if axis_name == "downside_power":
                    cur_down = float(best_axis_cfg.downside_power)
                elif axis_name == "upside_power":
                    cur_up = float(best_axis_cfg.upside_power)
                elif axis_name == "dispersion":
                    cur_disp = float(best_axis_cfg.dispersion)
                else:
                    cur_dead = float(best_axis_cfg.neutral_deadband)
                current_best_ratio = best_axis_ratio
                improved = True
        if not improved:
            break

    eligible = [
        r
        for r in rows
        if r.process_adopted_vs_baseline and math.isfinite(r.pooled_tw_ratio) and r.pooled_tw_ratio > 1.0
    ]
    champion: HpArmRow | None = None
    if eligible:
        eligible.sort(key=lambda r: (-r.pooled_tw_ratio, r.downside_power, r.upside_power, r.dispersion, r.neutral_deadband))
        champion = eligible[0]

    return AdaptiveHpScreenReport(
        operational_unlock=False,
        champion=champion,
        rows=tuple(rows),
        evaluations=evaluations,
    )
