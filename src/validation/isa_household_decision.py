"""ISA household operating-policy decision across budgets, profiles, and evidence tiers."""

from __future__ import annotations

import json
import logging
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np

from src.sim.isa_household import (
    HouseholdOutcome,
    HouseholdPlan,
    expand_pension_profile,
    revalue_household,
    scenario_growth,
    simulate_household,
)
from src.sim.isa_tax import IsaTaxRegime
from src.sim.pension_monthly import MonthlyReturnPanel, WeightSchedule, block_bootstrap_panels
from src.sim.pension_tax import PensionTaxRegime
from src.sim.tax import KrOverseasTaxRegime
from src.validation.isa_household_config import IsaHouseholdSpec
from src.validation.pension_decision import ce_ratio
from src.validation.pension_decision_config import PensionDecisionSpec
from src.validation.pension_decision_record import PensionDecisionRecord

logger = logging.getLogger(__name__)

__all__ = [
    "IsaArmCell",
    "IsaBudgetDecision",
    "IsaHouseholdReport",
    "evaluate_isa_household",
    "freeze_isa_household_decision",
    "isa_household_markdown",
]

_MONTHS_PER_YEAR: Final[int] = 12
_FREEZABLE_STATUSES: Final[frozenset[str]] = frozenset({"ADOPT_ARM", "KEEP_BASELINE"})


@dataclass(frozen=True, slots=True)
class IsaArmCell:
    """Evidence for one arm in one budget x profile x tier x horizon cell, paired with the baseline arm."""

    budget_krw: int
    arm_id: str
    profile_id: str
    tier: str
    horizon_years: int
    ce_ratio: float
    median_ratio: float
    worst_ratio: float
    scenario_count: int
    median_household_net_krw: float
    median_pension_locked_share: float


@dataclass(frozen=True, slots=True)
class IsaBudgetDecision:
    """Operating-policy verdict for one ISA annual budget."""

    budget_krw: int
    status: str
    selected_arm_id: str
    equivalent_arm_ids: tuple[str, ...]
    robust_scores: Mapping[str, float]
    per_profile_best: Mapping[str, str]
    bootstrap_win_share: float | None
    sensitivity_selected: Mapping[int, str]
    holding_scores: Mapping[str, float]
    holding_consistent: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IsaHouseholdReport:
    """Full evidence and verdicts of one ISA household decision run."""

    name: str
    incumbent_id: str
    trial_count: int
    cells: tuple[IsaArmCell, ...]
    decisions: tuple[IsaBudgetDecision, ...]


def _schedule_drag(annual_drag: Mapping[str, float], schedule: WeightSchedule) -> dict[str, float]:
    return {sleeve: drag for sleeve, drag in annual_drag.items() if sleeve in schedule.start_weights}


def _locked_shares(outcome: HouseholdOutcome) -> list[float]:
    shares: list[float] = []
    for index in range(int(outcome.household_net_krw.shape[0])):
        household = float(outcome.household_net_krw[index])
        locked = float(outcome.pension_balance_krw[index]) - float(outcome.pension_uncredited_krw[index])
        shares.append(locked / household if household != 0.0 else 0.0)
    return shares


def _select_best(robust: Mapping[str, float], order: Sequence[str], band: float) -> tuple[str, tuple[str, ...]]:
    best = max(robust[arm_id] for arm_id in order)
    equivalents = tuple(arm_id for arm_id in order if robust[arm_id] >= best - band)
    return equivalents[0], equivalents


def evaluate_isa_household(
    spec: IsaHouseholdSpec,
    decision_spec: PensionDecisionSpec,
    record: PensionDecisionRecord,
    modern: MonthlyReturnPanel,
    century: MonthlyReturnPanel,
    *,
    isa_regime: IsaTaxRegime,
    pension_regime: PensionTaxRegime,
    overseas_regime: KrOverseasTaxRegime,
    seed: int,
) -> IsaHouseholdReport:
    """Choose, per ISA budget, the operating mode with the best worst-case household growth.

    The holding is the frozen pension incumbent so arms differ only in routing and tax. Each arm's
    robust score is the minimum, over declared income profiles and feasible tier-horizon cells, of the
    primary-gamma certainty equivalent of its household-net ratio to the baseline arm on identical
    scenarios; future income is unknown, so the policy must hold up under every declared life path.
    Among arms within the equivalence band of the best score, the earliest in the registered order wins,
    so extra lock-up or complexity is never bought for a statistically equal result. The selection must
    also beat the baseline on most century block-bootstrap paths for every profile and must not change
    when the pension is re-valued over the sensitivity drawing horizons. Finally every pension-decision
    candidate is scored as the household holding under the selected arm; if some candidate beats the
    frozen incumbent by more than the band, the verdict carries a holding-review reason but the frozen
    pension decision is never replaced here.

    Raises:
        ValueError: If the record's incumbent is not a candidate of ``decision_spec``, a panel lacks an
            incumbent or candidate sleeve, no tier-horizon cell is feasible, or the bootstrap horizon
            (max of ``spec.horizons_years``) exceeds the century panel.
    """
    if record.incumbent_id not in decision_spec.candidates:
        raise ValueError(f"ISA household incumbent {record.incumbent_id!r} is not among candidates")
    holding = record.incumbent_schedule
    candidate_sleeves = {sleeve for schedule in decision_spec.candidates.values() for sleeve in schedule.start_weights}
    holding_sleeves = set(holding.start_weights)
    for panel, label in ((modern, "modern"), (century, "century")):
        missing = sorted((candidate_sleeves | holding_sleeves) - set(panel.returns))
        if missing:
            raise ValueError(f"ISA household {label} panel lacks sleeves: {missing}")
    holding_drag = _schedule_drag(decision_spec.annual_drag_by_sleeve, holding)

    arm_order = tuple(spec.arms)
    baseline_id = spec.baseline_arm_id
    band = spec.equivalence_band
    gamma = decision_spec.primary_gamma

    feasible: list[tuple[str, MonthlyReturnPanel, int, np.ndarray]] = []
    for tier, panel in (("modern", modern), ("century", century)):
        for horizon in spec.horizons_years:
            if horizon * _MONTHS_PER_YEAR > len(panel.months):
                logger.info(
                    "[PORTFOLIO] event=isa_household_skip tier=%s horizon_years=%d panel_months=%d",
                    tier,
                    horizon,
                    len(panel.months),
                )
                continue
            growth = scenario_growth(
                [panel],
                holding,
                horizon_years=horizon,
                step_months=spec.step_months,
                annual_drag=holding_drag,
            )
            feasible.append((tier, panel, horizon, growth))
    if not feasible:
        raise ValueError("ISA household has no feasible tier-horizon cell")
    max_horizon = max(spec.horizons_years)
    if max_horizon * _MONTHS_PER_YEAR > len(century.months):
        raise ValueError(f"ISA household bootstrap horizon {max_horizon}y exceeds {len(century.months)} century months")

    cells: list[IsaArmCell] = []
    decisions: list[IsaBudgetDecision] = []
    for budget in spec.isa_budgets_krw:
        outcomes: dict[tuple[str, str, int, str], HouseholdOutcome] = {}
        for profile in spec.profiles:
            for tier, _panel, horizon, growth in feasible:
                plan = HouseholdPlan(
                    plan_start_year=spec.plan_start_year,
                    horizon_years=horizon,
                    pension_annual_krw=spec.pension_annual_krw,
                    isa_annual_budget_krw=budget,
                    annuity_drawing_years=spec.annuity_drawing_years,
                )
                for arm_id in arm_order:
                    outcomes[(profile.profile_id, tier, horizon, arm_id)] = simulate_household(
                        growth,
                        holding,
                        spec.arms[arm_id],
                        plan,
                        profile,
                        isa_regime=isa_regime,
                        pension_regime=pension_regime,
                        overseas_regime=overseas_regime,
                    )
        ce_values: dict[str, list[float]] = {arm_id: [] for arm_id in arm_order}
        per_profile_values: dict[str, dict[str, list[float]]] = {
            profile.profile_id: {arm_id: [] for arm_id in arm_order} for profile in spec.profiles
        }
        for profile in spec.profiles:
            for tier, _panel, horizon, _growth in feasible:
                base = outcomes[(profile.profile_id, tier, horizon, baseline_id)].household_net_krw
                for arm_id in arm_order:
                    own = outcomes[(profile.profile_id, tier, horizon, arm_id)].household_net_krw
                    ratios = tuple(float(o) / float(b) for o, b in zip(own, base, strict=True))
                    score = ce_ratio(ratios, gamma)
                    ce_values[arm_id].append(score)
                    per_profile_values[profile.profile_id][arm_id].append(score)
                    outcome = outcomes[(profile.profile_id, tier, horizon, arm_id)]
                    cells.append(
                        IsaArmCell(
                            budget_krw=budget,
                            arm_id=arm_id,
                            profile_id=profile.profile_id,
                            tier=tier,
                            horizon_years=horizon,
                            ce_ratio=score,
                            median_ratio=statistics.median(ratios),
                            worst_ratio=min(ratios),
                            scenario_count=len(ratios),
                            median_household_net_krw=statistics.median(float(v) for v in own),
                            median_pension_locked_share=statistics.median(_locked_shares(outcome)),
                        )
                    )
        robust = {arm_id: min(values) for arm_id, values in ce_values.items()}
        per_profile_best = {
            profile_id: _select_best({arm_id: min(values) for arm_id, values in per_values.items()}, arm_order, band)[0]
            for profile_id, per_values in per_profile_values.items()
        }
        selected, equivalents = _select_best(robust, arm_order, band)

        bootstrap_win: float | None = None
        if selected != baseline_id:
            boot_panels = block_bootstrap_panels(
                century,
                n_paths=spec.bootstrap_paths,
                horizon_months=max_horizon * _MONTHS_PER_YEAR,
                block_months=spec.bootstrap_block_months,
                seed=seed,
            )
            boot_growth = scenario_growth(
                list(boot_panels),
                holding,
                horizon_years=max_horizon,
                step_months=max_horizon * _MONTHS_PER_YEAR,
                annual_drag=holding_drag,
            )
            wins: list[float] = []
            for profile in spec.profiles:
                boot_plan = HouseholdPlan(
                    plan_start_year=spec.plan_start_year,
                    horizon_years=max_horizon,
                    pension_annual_krw=spec.pension_annual_krw,
                    isa_annual_budget_krw=budget,
                    annuity_drawing_years=spec.annuity_drawing_years,
                )
                own = simulate_household(
                    boot_growth,
                    holding,
                    spec.arms[selected],
                    boot_plan,
                    profile,
                    isa_regime=isa_regime,
                    pension_regime=pension_regime,
                    overseas_regime=overseas_regime,
                ).household_net_krw
                base = simulate_household(
                    boot_growth,
                    holding,
                    spec.arms[baseline_id],
                    boot_plan,
                    profile,
                    isa_regime=isa_regime,
                    pension_regime=pension_regime,
                    overseas_regime=overseas_regime,
                ).household_net_krw
                wins.append(sum(1 for o, b in zip(own, base, strict=True) if float(o) > float(b)) / len(own))
            bootstrap_win = min(wins)

        sensitivity_selected: dict[int, str] = {}
        for drawing in spec.sensitivity_drawing_years:
            sens_values: dict[str, list[float]] = {arm_id: [] for arm_id in arm_order}
            for profile in spec.profiles:
                for tier, _panel, horizon, _growth in feasible:
                    expanded = expand_pension_profile(profile, plan_start_year=spec.plan_start_year, plan_years=horizon)
                    revalued = {
                        arm_id: revalue_household(
                            outcomes[(profile.profile_id, tier, horizon, arm_id)],
                            drawing_years=drawing,
                            profile=expanded,
                            regime=pension_regime,
                        )
                        for arm_id in arm_order
                    }
                    base = revalued[baseline_id]
                    for arm_id in arm_order:
                        ratios = tuple(float(o) / float(b) for o, b in zip(revalued[arm_id], base, strict=True))
                        sens_values[arm_id].append(ce_ratio(ratios, gamma))
            sens_robust = {arm_id: min(values) for arm_id, values in sens_values.items()}
            sensitivity_selected[drawing], _ = _select_best(sens_robust, arm_order, band)

        # 상태는 우선순위로 하나만 정하되, 사유는 실패한 검사를 모두 남긴다.
        bootstrap_weak = bootstrap_win is not None and bootstrap_win < spec.min_bootstrap_win_share
        exit_sensitive = any(choice != selected for choice in sensitivity_selected.values())
        reasons: list[str] = []
        if bootstrap_weak:
            reasons.append("BOOTSTRAP_WEAK")
        if exit_sensitive:
            reasons.append("EXIT_SENSITIVE")
        if selected == baseline_id:
            status = "KEEP_BASELINE"
        elif bootstrap_weak:
            status = "BOOTSTRAP_WEAK"
        elif exit_sensitive:
            status = "EXIT_SENSITIVE"
        else:
            status = "ADOPT_ARM"

        holding_scores: dict[str, float] = {}
        for candidate_id, schedule in decision_spec.candidates.items():
            candidate_drag = _schedule_drag(decision_spec.annual_drag_by_sleeve, schedule)
            candidate_values: list[float] = []
            for profile in spec.profiles:
                for tier, panel, horizon, _growth in feasible:
                    candidate_growth = scenario_growth(
                        [panel],
                        schedule,
                        horizon_years=horizon,
                        step_months=spec.step_months,
                        annual_drag=candidate_drag,
                    )
                    plan = HouseholdPlan(
                        plan_start_year=spec.plan_start_year,
                        horizon_years=horizon,
                        pension_annual_krw=spec.pension_annual_krw,
                        isa_annual_budget_krw=budget,
                        annuity_drawing_years=spec.annuity_drawing_years,
                    )
                    candidate_outcome = simulate_household(
                        candidate_growth,
                        schedule,
                        spec.arms[selected],
                        plan,
                        profile,
                        isa_regime=isa_regime,
                        pension_regime=pension_regime,
                        overseas_regime=overseas_regime,
                    )
                    incumbent_outcome = outcomes[(profile.profile_id, tier, horizon, selected)]
                    ratios = tuple(
                        float(o) / float(b)
                        for o, b in zip(
                            candidate_outcome.household_net_krw,
                            incumbent_outcome.household_net_krw,
                            strict=True,
                        )
                    )
                    candidate_values.append(ce_ratio(ratios, gamma))
            holding_scores[candidate_id] = min(candidate_values)
        holding_consistent = max(holding_scores.values()) < 1.0 + band
        if not holding_consistent:
            reasons.append("HOLDING_REVIEW")

        decisions.append(
            IsaBudgetDecision(
                budget_krw=budget,
                status=status,
                selected_arm_id=selected,
                equivalent_arm_ids=equivalents,
                robust_scores=dict(robust),
                per_profile_best=dict(per_profile_best),
                bootstrap_win_share=bootstrap_win,
                sensitivity_selected=dict(sensitivity_selected),
                holding_scores=dict(holding_scores),
                holding_consistent=holding_consistent,
                reasons=tuple(reasons),
            )
        )
        logger.info(
            "[PORTFOLIO] event=isa_household_budget_done budget_krw=%d status=%s selected=%s bootstrap_win=%s",
            budget,
            status,
            selected,
            f"{bootstrap_win:.3f}" if bootstrap_win is not None else "NONE",
        )
    trial_count = spec.lineage.related_trial_count + len(spec.arms) * len(spec.isa_budgets_krw)
    return IsaHouseholdReport(
        name=spec.name,
        incumbent_id=record.incumbent_id,
        trial_count=trial_count,
        cells=tuple(cells),
        decisions=tuple(decisions),
    )


def isa_household_markdown(report: IsaHouseholdReport) -> str:
    """Human summary: one table per budget (arm, robust score, per-profile winners, status, reasons)."""
    lines = [
        f"# ISA 하우스홀드 결정 {report.name}",
        "",
        f"- 현행 홀딩: {report.incumbent_id}",
        f"- 시행 횟수: {report.trial_count}",
        "",
    ]
    for decision in report.decisions:
        lines += [
            f"## 예산 {decision.budget_krw}원",
            "",
            f"- 상태: {decision.status}",
            f"- 선택: {decision.selected_arm_id}",
            f"- 동등 후보: {', '.join(decision.equivalent_arm_ids) if decision.equivalent_arm_ids else '없음'}",
            f"- 사유: {', '.join(decision.reasons) if decision.reasons else '없음'}",
            (
                f"- 부트스트랩 승률: {decision.bootstrap_win_share:.3f}"
                if decision.bootstrap_win_share is not None
                else "- 부트스트랩 승률: 없음"
            ),
            "",
            "| 운용 방식 | 강건 점수 |",
            "| --- | --- |",
        ]
        for arm_id, score in sorted(decision.robust_scores.items()):
            lines.append(f"| {arm_id} | {score:.6f} |")
        lines += [
            "",
            "| 프로파일 | 최적 운용 방식 |",
            "| --- | --- |",
        ]
        for profile_id, best in sorted(decision.per_profile_best.items()):
            lines.append(f"| {profile_id} | {best} |")
        lines += [
            "",
            "| 민감도(연금 수령 연수) | 선택 |",
            "| --- | --- |",
        ]
        for drawing, choice in sorted(decision.sensitivity_selected.items()):
            lines.append(f"| {drawing} | {choice} |")
        lines += [
            "",
            "| 후보 홀딩 | 강건 점수 |",
            "| --- | --- |",
        ]
        for candidate_id, score in sorted(decision.holding_scores.items()):
            lines.append(f"| {candidate_id} | {score:.6f} |")
        lines += ["", f"- 홀딩 일관성: {'예' if decision.holding_consistent else '아니오'}", ""]
    return "\n".join(lines) + "\n"


def freeze_isa_household_decision(
    report: IsaHouseholdReport,
    *,
    output_dir: Path,
    frozen_at: datetime,
    git_commit: str,
    config_sha256: str,
    pension_record_id: str,
    run_id: str,
) -> Path:
    """Write an immutable operating-policy record; refuses to overwrite an existing file.

    Raises: ValueError if any budget status is not ADOPT_ARM or KEEP_BASELINE, ``frozen_at`` is naive,
        or the target file already exists.
    """
    for decision in report.decisions:
        if decision.status not in _FREEZABLE_STATUSES:
            raise ValueError(f"ISA household budget {decision.budget_krw} status {decision.status!r} is not freezable")
    if frozen_at.tzinfo is None or frozen_at.utcoffset() is None:
        raise ValueError(f"ISA household frozen_at must be timezone-aware, got {frozen_at!r}")
    if not git_commit.strip():
        raise ValueError("ISA household git_commit must be a non-blank string")
    if not config_sha256.strip():
        raise ValueError("ISA household config_sha256 must be a non-blank string")
    record_id = f"{report.name}__{config_sha256.strip()[:16]}"
    path = Path(output_dir) / f"{record_id}.json"
    if path.exists():
        raise ValueError(f"ISA household decision record already exists: {path.as_posix()}")
    document: dict[str, object] = {
        "record_id": record_id,
        "frozen_at": frozen_at.astimezone(UTC).isoformat(),
        "git_commit": git_commit.strip(),
        "config_sha256": config_sha256.strip(),
        "pension_record_id": pension_record_id,
        "run_id": run_id,
        "incumbent_id": report.incumbent_id,
        "trial_count": report.trial_count,
        "decisions": [
            {
                "budget_krw": decision.budget_krw,
                "status": decision.status,
                "selected_arm_id": decision.selected_arm_id,
                "equivalent_arm_ids": list(decision.equivalent_arm_ids),
                "robust_scores": dict(decision.robust_scores),
                "per_profile_best": dict(decision.per_profile_best),
                "bootstrap_win_share": decision.bootstrap_win_share,
                "sensitivity_selected": {str(year): arm for year, arm in decision.sensitivity_selected.items()},
                "holding_scores": dict(decision.holding_scores),
                "holding_consistent": decision.holding_consistent,
                "reasons": list(decision.reasons),
            }
            for decision in report.decisions
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("[PORTFOLIO] event=isa_household_record_frozen record=%s", record_id)
    return path
