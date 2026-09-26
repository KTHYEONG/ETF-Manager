"""Robust pension decision evaluation across independent evidence tiers and horizons."""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from src.sim.pension_monthly import (
    MonthlyReturnPanel,
    PensionPathResult,
    WeightSchedule,
    block_bootstrap_panels,
    simulate_cohorts,
    simulate_pension_dca,
)
from src.validation.pension_decision_config import PensionDecisionSpec

logger = logging.getLogger(__name__)

__all__ = [
    "CandidateScore",
    "PensionDecisionReport",
    "PensionDecisionStatus",
    "assert_tax_rank_neutrality",
    "ce_ratio",
    "evaluate_pension_decision",
]

PensionDecisionStatus = Literal["ADOPT_CANDIDATE", "KEEP_BENCHMARK", "KEEP_INCUMBENT", "NO_DECISION"]

_MONTHS_PER_YEAR: Final[int] = 12


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Primary-gamma evidence for one candidate in one tier-horizon cell."""

    candidate_id: str
    tier: str
    horizon_years: int
    gamma: float
    ce_ratio: float
    median_ratio: float
    worst_ratio: float
    cohort_count: int
    median_pre_retirement_drawdown: float


@dataclass(frozen=True, slots=True)
class PensionDecisionReport:
    """Scores for every candidate, the selection, and the verdict with reasons."""

    name: str
    status: PensionDecisionStatus
    selected_id: str | None
    equivalent_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    scores: tuple[CandidateScore, ...]
    robust_scores: Mapping[str, float]
    sensitivity_robust_scores: Mapping[float, Mapping[str, float]]
    bootstrap_win_share: Mapping[str, float]
    tax_rank_agreement: bool
    manifest_hashes: Mapping[str, str]
    trial_count: int
    dominance_min_ratio: Mapping[str, float]
    dominance_min_ratio_by_tier: Mapping[str, Mapping[str, float]]
    guard_excluded_ids: tuple[str, ...]
    control_scores: Mapping[str, float]
    control_vs_reference: Mapping[str, float]


def ce_ratio(ratios: Sequence[float], gamma: float) -> float:
    """CRRA certainty equivalent of paired wealth ratios (gamma 1 = geometric mean)."""
    if gamma == 1.0:
        return math.exp(math.fsum(math.log(ratio) for ratio in ratios) / len(ratios))
    power = 1.0 - gamma
    return float((math.fsum(ratio**power for ratio in ratios) / len(ratios)) ** (1.0 / power))


_ce_ratio = ce_ratio


def _paired_ratios(own: Sequence[float], base: Sequence[float]) -> tuple[float, ...]:
    """Paired terminal-wealth ratios sharing one cohort index."""
    return tuple(own_value / base_value for own_value, base_value in zip(own, base, strict=True))


def _cell_scores(
    candidate_ids: Sequence[str],
    tier: str,
    horizon_years: int,
    gamma: float,
    terminals: Mapping[str, tuple[float, ...]],
    drawdowns: Mapping[str, tuple[float, ...]],
    benchmark_id: str,
) -> list[CandidateScore]:
    """Score every candidate against paired benchmark terminals sharing one cohort index."""
    benchmark = terminals[benchmark_id]
    cell: list[CandidateScore] = []
    for candidate_id in candidate_ids:
        ratios = tuple(
            own / base for own, base in zip(terminals[candidate_id], benchmark, strict=True)
        )
        cell.append(
            CandidateScore(
                candidate_id=candidate_id,
                tier=tier,
                horizon_years=horizon_years,
                gamma=gamma,
                ce_ratio=ce_ratio(ratios, gamma),
                median_ratio=statistics.median(ratios),
                worst_ratio=min(ratios),
                cohort_count=len(ratios),
                median_pre_retirement_drawdown=statistics.median(drawdowns[candidate_id]),
            )
        )
    return cell


def _schedule_drag(annual_drag: Mapping[str, float], schedule: WeightSchedule) -> dict[str, float]:
    """Restrict a universe-wide drag map to the sleeves one schedule actually holds."""
    return {sleeve: drag for sleeve, drag in annual_drag.items() if sleeve in schedule.start_weights}


def evaluate_pension_decision(
    spec: PensionDecisionSpec,
    modern: MonthlyReturnPanel,
    century: MonthlyReturnPanel,
    *,
    seed: int,
    incumbent_id: str | None = None,
    realized: MonthlyReturnPanel | None = None,
) -> PensionDecisionReport:
    """Choose the pension holding that maximizes robust long-run growth versus the benchmark.

    Each candidate's robust score is the minimum, over the two independent tiers
    (tradable ETF era and century proxy) and all horizons, of the primary-gamma
    certainty equivalent of its paired terminal-wealth ratio to the benchmark.
    Candidates within the equivalence band of the best score are treated as equal
    growth; among them the one with the mildest pre-retirement drawdown is chosen,
    so equal growth is never bought with extra late-stage risk. The century block
    bootstrap measures uncertainty and is not a third vote in the minimum because
    it resamples the same history. The bootstrap probes the longest accumulation,
    where compounding differences are largest.

    When a dominance reference is declared, a candidate is eligible for selection only if its
    paired certainty equivalent against the reference is at least one minus the equivalence band
    in every tier-horizon cell. A challenger must therefore not lose the tradable era to buy a
    century-tier edge, nor the reverse. The reference is always eligible. The best score and the
    equivalence set are computed over eligible candidates only. Controls are scored on the modern
    tier alone, never enter the robust minimum, the bootstrap, or the selection, and exist to
    record why an allocation without a century proxy was not adopted.

    When ``spec.realized_horizons_years`` is non-empty a third tier, built only from actual ETF
    prices, is scored for candidates over those horizons. Its cells enter the robust minimum and
    the dominance guard exactly like the other tiers, so months filled by a research proxy can
    score a candidate but can never be the only evidence that lets it replace the reference. Among
    candidates within the equivalence band of the best eligible score, selection prefers the mildest
    pre-retirement drawdown, then the lowest start-weighted fee drag, then the reference, then the
    lexicographically smallest id, so exactly one holding is returned.

    Args:
        spec: Pre-registered decision config.
        modern: Tradable ETF tier panel.
        century: Century research tier panel.
        seed: Bootstrap seed.
        incumbent_id: Currently held candidate from a frozen record, if any.
        realized: Realized-ETF tier panel (tier ``realized``), required exactly when
            ``spec.realized_horizons_years`` is non-empty.

    Returns:
        Scores for every candidate, the selection, and the verdict with reasons.

    Raises:
        ValueError: If a panel lacks a sleeve required by any candidate, the incumbent id is unknown,
            the drag map names a sleeve no candidate or control uses, ``realized`` is missing while
            realized horizons are declared, ``realized`` is provided while no realized horizon is
            declared, the realized tier label is not ``realized``, the realized panel lacks a
            candidate sleeve, or a realized horizon does not fit the realized panel.
    """
    candidate_ids = tuple(spec.candidates)
    if incumbent_id is not None and incumbent_id not in spec.candidates:
        raise ValueError(f"pension incumbent_id {incumbent_id!r} is not among candidates")
    reference_id = spec.dominance_reference_id
    if reference_id is not None and reference_id not in spec.candidates:
        raise ValueError(f"pension dominance_reference_id {reference_id!r} is not among candidates")
    need_guard = reference_id is not None
    have_controls = bool(spec.controls)
    candidate_sleeves = {sleeve for schedule in spec.candidates.values() for sleeve in schedule.start_weights}
    control_sleeves = {sleeve for schedule in spec.controls.values() for sleeve in schedule.start_weights}
    modern_missing = sorted((candidate_sleeves | control_sleeves) - set(modern.returns))
    if modern_missing:
        raise ValueError(f"pension modern panel lacks sleeves required by candidates and controls: {modern_missing}")
    century_missing = sorted(candidate_sleeves - set(century.returns))
    if century_missing:
        raise ValueError(f"pension century panel lacks sleeves required by candidates: {century_missing}")
    unknown_drags = sorted(set(spec.annual_drag_by_sleeve) - candidate_sleeves - control_sleeves)
    if unknown_drags:
        raise ValueError(f"pension annual_drag_by_sleeve names unknown sleeves: {unknown_drags}")
    realized_horizons = tuple(spec.realized_horizons_years)
    if realized_horizons:
        if realized is None:
            raise ValueError("pension realized panel is required while realized_horizons_years is non-empty")
        if realized.tier != "realized":
            raise ValueError(f"pension realized panel tier must be 'realized', got {realized.tier!r}")
        realized_missing = sorted(candidate_sleeves - set(realized.returns))
        if realized_missing:
            raise ValueError(f"pension realized panel lacks sleeves required by candidates: {realized_missing}")
    elif realized is not None:
        raise ValueError("pension realized panel is provided while realized_horizons_years is empty")
    schedules = dict(spec.candidates)

    scores: list[CandidateScore] = []
    robust: dict[str, list[float]] = {candidate_id: [] for candidate_id in candidate_ids}
    guard_lists: dict[str, list[float]] = {candidate_id: [] for candidate_id in candidate_ids}
    control_lists: dict[str, list[float]] = {control_id: [] for control_id in spec.controls}
    control_ref_lists: dict[str, list[float]] = {control_id: [] for control_id in spec.controls}
    sensitivity: dict[float, dict[str, list[float]]] = {
        gamma: {candidate_id: [] for candidate_id in candidate_ids} for gamma in spec.sensitivity_gammas
    }
    tier_drawdown: dict[str, dict[str, list[float]]] = {
        candidate_id: {} for candidate_id in candidate_ids
    }
    guard_by_tier: dict[str, dict[str, list[float]]] = {}
    tiers: tuple[tuple[str, MonthlyReturnPanel, tuple[int, ...]], ...] = (
        ("modern", modern, spec.horizons_years),
        ("century", century, spec.horizons_years),
    )
    if realized is not None:
        tiers += (("realized", realized, realized_horizons),)
    for tier, panel, horizons in tiers:
        for horizon in horizons:
            if horizon * _MONTHS_PER_YEAR > len(panel.months):
                if tier == "realized":
                    raise ValueError(
                        f"pension realized horizon {horizon}y exceeds {len(panel.months)} realized months"
                    )
                logger.info(
                    "[PORTFOLIO] event=pension_decision_skip tier=%s horizon_years=%d panel_months=%d",
                    tier,
                    horizon,
                    len(panel.months),
                )
                continue
            active = schedules if tier != "modern" or not have_controls else {**schedules, **spec.controls}
            results: dict[str, tuple[PensionPathResult, ...]] = {}
            for schedule_id, schedule in active.items():
                results.update(
                    simulate_cohorts(
                        panel,
                        {schedule_id: schedule},
                        years=horizon,
                        step_months=spec.step_months,
                        pre_retirement_months=spec.pre_retirement_months,
                        annual_drag=_schedule_drag(spec.annual_drag_by_sleeve, schedule),
                    )
                )
            terminals = {
                candidate_id: tuple(result.terminal_value for result in results[candidate_id])
                for candidate_id in candidate_ids
            }
            drawdowns = {
                candidate_id: tuple(result.pre_retirement_drawdown for result in results[candidate_id])
                for candidate_id in candidate_ids
            }
            scores.extend(
                _cell_scores(
                    candidate_ids, panel.tier, horizon, spec.primary_gamma,
                    terminals, drawdowns, spec.benchmark_id,
                )
            )
            for score in scores[-len(candidate_ids):]:
                robust[score.candidate_id].append(score.ce_ratio)
                tier_drawdown[score.candidate_id].setdefault(tier, []).append(score.median_pre_retirement_drawdown)
            for gamma in spec.sensitivity_gammas:
                for scored in _cell_scores(
                    candidate_ids, panel.tier, horizon, gamma, terminals, drawdowns, spec.benchmark_id
                ):
                    sensitivity[gamma][scored.candidate_id].append(scored.ce_ratio)
            ref_terms = terminals[reference_id] if reference_id is not None else None
            if ref_terms is not None:
                for candidate_id in candidate_ids:
                    guard_ratio = ce_ratio(
                        _paired_ratios(terminals[candidate_id], ref_terms), spec.primary_gamma
                    )
                    guard_lists[candidate_id].append(guard_ratio)
                    guard_by_tier.setdefault(tier, {}).setdefault(candidate_id, []).append(guard_ratio)
            if have_controls and tier == "modern":
                base_terms = terminals[spec.benchmark_id]
                for control_id in spec.controls:
                    control_terms = tuple(result.terminal_value for result in results[control_id])
                    control_lists[control_id].append(
                        ce_ratio(_paired_ratios(control_terms, base_terms), spec.primary_gamma)
                    )
                    if ref_terms is not None:
                        control_ref_lists[control_id].append(
                            ce_ratio(_paired_ratios(control_terms, ref_terms), spec.primary_gamma)
                        )
    for candidate_id in candidate_ids:
        if not robust[candidate_id]:
            raise ValueError(f"pension candidate {candidate_id!r} has no feasible tier-horizon cell")
    robust_scores = {candidate_id: min(values) for candidate_id, values in robust.items()}
    sensitivity_scores = {
        gamma: {candidate_id: min(values) for candidate_id, values in per_gamma.items()}
        for gamma, per_gamma in sensitivity.items()
    }

    bootstrap_horizon = max(spec.horizons_years)
    if bootstrap_horizon * _MONTHS_PER_YEAR > len(century.months):
        raise ValueError(
            f"pension bootstrap horizon {bootstrap_horizon}y exceeds {len(century.months)} century months"
        )
    boot_panels = block_bootstrap_panels(
        century,
        n_paths=spec.bootstrap_paths,
        horizon_months=bootstrap_horizon * _MONTHS_PER_YEAR,
        block_months=spec.bootstrap_block_months,
        seed=seed,
    )
    wins = dict.fromkeys(candidate_ids, 0)
    benchmark_schedule = schedules[spec.benchmark_id]
    benchmark_drag = _schedule_drag(spec.annual_drag_by_sleeve, benchmark_schedule)
    candidate_drags = {
        candidate_id: _schedule_drag(spec.annual_drag_by_sleeve, schedules[candidate_id])
        for candidate_id in candidate_ids
    }
    for boot in boot_panels:
        base = simulate_pension_dca(
            boot,
            benchmark_schedule,
            start_month_index=0,
            years=bootstrap_horizon,
            pre_retirement_months=0,
            annual_drag=benchmark_drag,
        ).terminal_value
        for candidate_id in candidate_ids:
            terminal = simulate_pension_dca(
                boot,
                schedules[candidate_id],
                start_month_index=0,
                years=bootstrap_horizon,
                pre_retirement_months=0,
                annual_drag=candidate_drags[candidate_id],
            ).terminal_value
            if terminal / base > 1.0:
                wins[candidate_id] += 1
    win_share = {candidate_id: wins[candidate_id] / len(boot_panels) for candidate_id in candidate_ids}

    dominance_min_ratio: dict[str, float] = {}
    dominance_min_ratio_by_tier: dict[str, dict[str, float]] = {}
    guard_excluded: tuple[str, ...] = ()
    if need_guard:
        assert reference_id is not None
        dominance_min_ratio = {
            candidate_id: min(values) for candidate_id, values in guard_lists.items()
        }
        dominance_min_ratio_by_tier = {
            tier: {candidate_id: min(values) for candidate_id, values in per_tier.items()}
            for tier, per_tier in guard_by_tier.items()
        }
        threshold = 1.0 - spec.equivalence_band
        guard_excluded = tuple(
            sorted(candidate_id for candidate_id in candidate_ids if dominance_min_ratio[candidate_id] < threshold)
        )
        logger.info(
            "[PORTFOLIO] event=pension_decision_guard reference=%s excluded=%d",
            reference_id,
            len(guard_excluded),
        )
    control_scores = {control_id: min(values) for control_id, values in control_lists.items()}
    control_vs_reference = (
        {control_id: min(values) for control_id, values in control_ref_lists.items()} if need_guard else {}
    )
    excluded_set = set(guard_excluded)
    pool = [candidate_id for candidate_id in candidate_ids if candidate_id not in excluded_set]
    best = max(robust_scores[candidate_id] for candidate_id in pool)
    equivalent = sorted(candidate for candidate in pool if robust_scores[candidate] >= best - spec.equivalence_band)
    worst_tier_drawdown = {
        candidate: min(statistics.median(drawdowns) for drawdowns in tier_drawdown[candidate].values())
        for candidate in equivalent
    }
    fee_drag = {
        candidate: math.fsum(
            schedules[candidate].start_weights[sleeve] * spec.annual_drag_by_sleeve.get(sleeve, 0.0)
            for sleeve in schedules[candidate].start_weights
        )
        for candidate in equivalent
    }
    selection = min(
        equivalent,
        key=lambda candidate: (
            -worst_tier_drawdown[candidate],
            fee_drag[candidate],
            0 if candidate == reference_id else 1,
            candidate,
        ),
    )

    reasons: list[str] = []
    status: PensionDecisionStatus
    selected_id: str | None
    if selection == spec.benchmark_id:
        status, selected_id = "KEEP_BENCHMARK", spec.benchmark_id
    elif any(robust_scores[neighbor] <= 1.0 for neighbor in spec.neighbors.get(selection, ())):
        status, selected_id = "NO_DECISION", None
        reasons.append("KNIFE_EDGE")
    elif win_share[selection] < spec.min_bootstrap_win_share:
        status, selected_id = "NO_DECISION", None
        reasons.append("BOOTSTRAP_WEAK")
    elif incumbent_id is not None and incumbent_id in equivalent:
        status, selected_id = "KEEP_INCUMBENT", incumbent_id
    else:
        status, selected_id = "ADOPT_CANDIDATE", selection
    if need_guard:
        unguarded_best = max(robust_scores.values())
        best_excluded = [
            candidate_id
            for candidate_id, value in robust_scores.items()
            if candidate_id in excluded_set and value == unguarded_best
        ]
        if best_excluded:
            reasons.append("DOMINANCE_GUARD")
        if best_excluded and realized is not None:
            realized_mins = dominance_min_ratio_by_tier.get("realized", {})
            threshold = 1.0 - spec.equivalence_band
            if any(realized_mins.get(candidate_id, threshold) < threshold for candidate_id in best_excluded):
                reasons.append("REALIZED_VETO")
    trial_count_raw = spec.lineage.get("related_trial_count", 0)
    if isinstance(trial_count_raw, bool) or not isinstance(trial_count_raw, int):
        raise ValueError("pension lineage.related_trial_count must be a non-negative integer")
    trial_count = trial_count_raw + len(candidate_ids) + len(spec.controls)
    logger.info(
        "[PORTFOLIO] event=pension_decision_done name=%s status=%s selected=%s trial_count=%d",
        spec.name,
        status,
        selected_id or "NONE",
        trial_count,
    )
    return PensionDecisionReport(
        name=spec.name,
        status=status,
        selected_id=selected_id,
        equivalent_ids=tuple(equivalent),
        reasons=tuple(reasons),
        scores=tuple(scores),
        robust_scores=robust_scores,
        sensitivity_robust_scores=sensitivity_scores,
        bootstrap_win_share=win_share,
        tax_rank_agreement=True,
        manifest_hashes={},
        trial_count=trial_count,
        dominance_min_ratio=dominance_min_ratio,
        dominance_min_ratio_by_tier=dominance_min_ratio_by_tier,
        guard_excluded_ids=guard_excluded,
        control_scores=control_scores,
        control_vs_reference=control_vs_reference,
    )


def assert_tax_rank_neutrality(
    report: PensionDecisionReport,
    campaign_summaries: Sequence[Mapping[str, object]],
    arm_map: Mapping[str, str],
) -> bool:
    """Check that the after-tax pension engine orders mapped arms like the modern tier.

    Returns:
        True when the median after-tax ratio ordering of mapped campaign arms equals
        the modern-tier median ratio ordering for the same horizon set.
    """
    reverse = {candidate: arm for arm, candidate in arm_map.items()}
    modern_medians: dict[tuple[str, int], float] = {}
    for score in report.scores:
        if score.tier == "modern" and score.candidate_id in reverse:
            modern_medians[(score.candidate_id, score.horizon_years)] = score.median_ratio
    campaign_medians: dict[tuple[str, int], float] = {}
    for summary in campaign_summaries:
        arm = summary.get("arm_id")
        horizon_months = summary.get("horizon_months")
        median = summary.get("median_wealth_ratio")
        candidate = arm_map.get(arm) if isinstance(arm, str) else None
        if (
            candidate is None
            or not isinstance(horizon_months, int)
            or isinstance(median, bool)
            or not isinstance(median, float | int)
        ):
            continue
        campaign_medians[(candidate, horizon_months // _MONTHS_PER_YEAR)] = float(median)
    horizons = sorted(
        {horizon for _, horizon in modern_medians} & {horizon for _, horizon in campaign_medians}
    )
    if not horizons:
        return False
    for horizon in horizons:
        if any(
            (candidate, horizon) not in campaign_medians or (candidate, horizon) not in modern_medians
            for candidate in reverse
        ):
            return False
        campaign_order = sorted(reverse, key=lambda candidate: (-campaign_medians[(candidate, horizon)], candidate))
        modern_order = sorted(reverse, key=lambda candidate: (-modern_medians[(candidate, horizon)], candidate))
        if campaign_order != modern_order:
            return False
    return True
