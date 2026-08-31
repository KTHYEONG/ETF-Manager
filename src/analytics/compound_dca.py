"""Compound DCA tournament — reporting-only with QQQ/SOXX mixes and risk-budget arms."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final

from src.analytics.thesis.incremental import INCREMENTAL_SATELLITE_WEIGHTS
from src.policy.adaptive_contribution import OPERATIONAL_ADAPTIVE_CONTRIBUTION
from src.policy.mix_risk_budget import OPERATIONAL_MIX_RISK_BUDGET
from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig

if TYPE_CHECKING:
    from src.sim.allocation import AllocationResult

COMPOUND_MDD_SLACK: Final[float] = 0.02

OPERATIONAL_COMPOUND_BASELINE_ARM_ID: Final[str] = "qqq90_soxx10_adaptive_v5"

COMPOUND_DCA_ARM_IDS: Final[tuple[str, ...]] = (
    "qqq_flat",
    "qqq_adaptive_v5",
    "qqq90_soxx10_flat",
    "qqq90_soxx10_adaptive_v5",
    "qqq95_soxx5_adaptive_v5",
    "qqq85_soxx15_adaptive_v5",
    "soxx90_qqq10_flat",
    "soxx90_qqq10_adaptive_v5",
    "soxx100_flat",
    "soxx100_adaptive_v5",
    "qqq_soxx_riskbudget_flat",
    "qqq_soxx_riskbudget_adaptive_v5",
)

COMPOUND_DCA_MIX_TARGETS: Final[dict[str, float]] = {"QQQ": 0.9, "SOXX": 0.1}

COMPOUND_DCA_ROLE_SWAP_TARGETS: Final[dict[str, float]] = {"SOXX": 0.9, "QQQ": 0.1}

COMPOUND_DCA_SOXX100_TARGETS: Final[dict[str, float]] = {"SOXX": 1.0}

COMPOUND_DCA_WINDOW: Final[tuple[date, date]] = (date(2016, 7, 1), date(2026, 6, 30))


def qqq_soxx_intensity_targets(satellite_weight: float) -> dict[str, float]:
    """Return QQQ/SOXX simplex for preregistered satellite weights.

    Raises:
        ValueError: When ``satellite_weight`` is not in ``INCREMENTAL_SATELLITE_WEIGHTS``.
    """
    w = float(satellite_weight)
    if w not in INCREMENTAL_SATELLITE_WEIGHTS:
        raise ValueError(f"satellite_weight {satellite_weight!r} not in INCREMENTAL_SATELLITE_WEIGHTS {INCREMENTAL_SATELLITE_WEIGHTS!r}")
    qqq = 1.0 - w
    soxx = w
    total = qqq + soxx
    if not math.isfinite(qqq) or not math.isfinite(soxx) or abs(total - 1.0) > 1e-12:
        raise ValueError(f"weights must sum to 1.0 within 1e-12, got {total!r}")
    return {"QQQ": qqq, "SOXX": soxx}


def select_mdd_feasible_champion(
    rows: Sequence[CompoundDcaArmRow],
    *,
    baseline_arm_id: str = OPERATIONAL_COMPOUND_BASELINE_ARM_ID,
    mdd_slack: float = COMPOUND_MDD_SLACK,
) -> str | None:
    """Select max real_gain arm feasible within MDD slack of baseline.

    An arm is feasible iff ``candidate.max_drawdown >= baseline.max_drawdown - mdd_slack``.

    Raises:
        ValueError: On non-finite or negative ``mdd_slack``, empty ``rows``, or missing baseline.
    """
    if not math.isfinite(float(mdd_slack)) or float(mdd_slack) < 0.0:
        raise ValueError(f"mdd_slack must be finite and non-negative, got {mdd_slack!r}")
    if len(rows) == 0:
        raise ValueError("rows must be non-empty")
    baseline_row: CompoundDcaArmRow | None = None
    for r in rows:
        if r.arm_id == baseline_arm_id:
            baseline_row = r
            break
    if baseline_row is None:
        raise ValueError(f"baseline_arm_id {baseline_arm_id!r} not found in rows")
    baseline_mdd = float(baseline_row.max_drawdown)
    if not math.isfinite(baseline_mdd):
        raise ValueError(f"baseline max_drawdown must be finite, got {baseline_mdd!r}")
    feasible: list[CompoundDcaArmRow] = []
    for r in rows:
        mdd = float(r.max_drawdown)
        if not math.isfinite(mdd):
            raise ValueError(f"max_drawdown must be finite, got {mdd!r} for {r.arm_id!r}")
        if mdd >= baseline_mdd - float(mdd_slack):
            feasible.append(r)
    if not feasible:
        # Baseline itself is always feasible (mdd >= baseline_mdd - slack), so this branch
        # is unreachable unless slack is NaN (already rejected) or baseline mdd non-finite.
        return None
    best = feasible[0]
    best_gain = float(best.real_gain)
    for cand in feasible[1:]:
        gain = float(cand.real_gain)
        if gain > best_gain:
            best = cand
            best_gain = gain
    return best.arm_id


@dataclass(frozen=True, slots=True)
class CompoundDcaArmRow:
    """One tournament arm outcome."""

    arm_id: str
    config: AllocationConfig
    result: AllocationResult
    terminal_wealth_krw: float
    terminal_wealth_real_krw: float
    total_contribution_real_krw: float
    real_gain: float
    xirr: float
    xirr_real: float
    max_drawdown: float


@dataclass(frozen=True, slots=True)
class CompoundDcaReport:
    """Reporting-only tournament outcome across arms."""

    rows: tuple[CompoundDcaArmRow, ...]
    champion_arm_id: str
    operational_unlock: bool
    start: date
    end: date
    contribution_krw: float
    mdd_feasible_champion_arm_id: str | None
    mdd_baseline_arm_id: str
    mdd_slack: float
    growth_champion_arm_id: str = ""
    recommended_arm_id: str = ""


def compare_compound_dca(
    *,
    runner: Callable[[AllocationConfig], AllocationResult],
    contribution_krw: float,
    start: date | None = None,
    end: date | None = None,
) -> CompoundDcaReport:
    """Run arms on identical windows with I5 sizing-family checks.

    Raises:
        ValueError: On non-finite or non-positive ``contribution_krw``, diverging
            snapshot counts, or I5 credit mismatch within a sizing family.
    """
    if not math.isfinite(float(contribution_krw)) or float(contribution_krw) <= 0.0:
        raise ValueError(f"contribution_krw must be finite and > 0, got {contribution_krw!r}")
    window_start = start if start is not None else COMPOUND_DCA_WINDOW[0]
    window_end = end if end is not None else COMPOUND_DCA_WINDOW[1]

    configs: list[tuple[str, AllocationConfig]] = []
    for arm_id in COMPOUND_DCA_ARM_IDS:
        is_adaptive = arm_id.endswith("adaptive_v5")
        adaptive = OPERATIONAL_ADAPTIVE_CONTRIBUTION if is_adaptive else None
        targets: dict[str, float] | None = None
        mrb = None
        if arm_id.startswith("qqq90_soxx10"):
            targets = dict(qqq_soxx_intensity_targets(0.10))
        elif arm_id.startswith("qqq95_soxx5"):
            targets = dict(qqq_soxx_intensity_targets(0.05))
        elif arm_id.startswith("qqq85_soxx15"):
            targets = dict(qqq_soxx_intensity_targets(0.15))
        elif arm_id.startswith("soxx90_qqq10"):
            targets = dict(COMPOUND_DCA_ROLE_SWAP_TARGETS)
        elif arm_id.startswith("soxx100"):
            targets = dict(COMPOUND_DCA_SOXX100_TARGETS)
        elif arm_id.startswith("qqq_soxx_riskbudget"):
            targets = None
            mrb = OPERATIONAL_MIX_RISK_BUDGET
        else:
            targets = None
        cfg = AllocationConfig(
            policy=PolicyId.QQQ,
            start=window_start,
            end=window_end,
            monthly_contribution_krw=float(contribution_krw),
            adaptive_contribution=adaptive,
            targets_override=targets,
            mix_risk_budget=mrb,
            rebalance_band=None,
        )
        configs.append((arm_id, cfg))

    results: dict[str, AllocationResult] = {}
    for arm_id, cfg in configs:
        results[arm_id] = runner(cfg)

    counts = {len(results[arm_id].snapshots) for arm_id in COMPOUND_DCA_ARM_IDS}
    if len(counts) != 1:
        detail = ", ".join(f"{arm_id}={len(results[arm_id].snapshots)}" for arm_id in COMPOUND_DCA_ARM_IDS)
        raise ValueError(f"snapshot counts diverge across arms: {detail}")

    def credits(arm_id: str) -> tuple[float, ...]:
        return tuple(s.contribution_krw for s in results[arm_id].snapshots)

    # I5 family checks: flat vs adaptive (SOXX100 diagnostic-only excluded from identity check)
    flat_arms = [aid for aid in COMPOUND_DCA_ARM_IDS if not aid.endswith("adaptive_v5") and not aid.startswith("soxx100")]
    adaptive_arms = [aid for aid in COMPOUND_DCA_ARM_IDS if aid.endswith("adaptive_v5") and not aid.startswith("soxx100")]
    # Keep diagnostic arms in counts check but not in credit identity (they share window but SOXX100 is vehicle diagnostic)
    flat_ref = credits(flat_arms[0]) if flat_arms else ()
    for aid in flat_arms[1:]:
        if credits(aid) != flat_ref:
            raise ValueError(f"I5 flat credits must be identical: {flat_arms[0]!r} vs {aid!r} mismatch {flat_ref!r} vs {credits(aid)!r}")
    adaptive_ref = credits(adaptive_arms[0]) if adaptive_arms else ()
    for aid in adaptive_arms[1:]:
        if credits(aid) != adaptive_ref:
            raise ValueError(f"I5 adaptive credits must be identical: {adaptive_arms[0]!r} vs {aid!r} mismatch {adaptive_ref!r} vs {credits(aid)!r}")

    rows: list[CompoundDcaArmRow] = []
    for arm_id in COMPOUND_DCA_ARM_IDS:
        res = results[arm_id]
        cfg = next(c for aid, c in configs if aid == arm_id)
        real_gain = float(res.terminal_wealth_real_krw) - float(res.total_contribution_real_krw)
        rows.append(
            CompoundDcaArmRow(
                arm_id=arm_id,
                config=cfg,
                result=res,
                terminal_wealth_krw=float(res.terminal_wealth_krw),
                terminal_wealth_real_krw=float(res.terminal_wealth_real_krw),
                total_contribution_real_krw=float(res.total_contribution_real_krw),
                real_gain=real_gain,
                xirr=float(res.xirr),
                xirr_real=float(res.xirr_real),
                max_drawdown=float(res.max_drawdown),
            )
        )

    champion = rows[0].arm_id
    best = rows[0].real_gain
    for row in rows[1:]:
        if row.real_gain >= best:
            best = row.real_gain
            champion = row.arm_id

    mdd_feasible = select_mdd_feasible_champion(
        rows,
        baseline_arm_id=OPERATIONAL_COMPOUND_BASELINE_ARM_ID,
        mdd_slack=COMPOUND_MDD_SLACK,
    )

    return CompoundDcaReport(
        rows=tuple(rows),
        champion_arm_id=champion,
        operational_unlock=False,
        start=window_start,
        end=window_end,
        contribution_krw=float(contribution_krw),
        mdd_feasible_champion_arm_id=mdd_feasible,
        mdd_baseline_arm_id=OPERATIONAL_COMPOUND_BASELINE_ARM_ID,
        mdd_slack=float(COMPOUND_MDD_SLACK),
        growth_champion_arm_id=champion,
        recommended_arm_id=champion,
    )
