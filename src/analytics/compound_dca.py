"""Compound DCA tournament — reporting-only QQQ vs QQQ90/SOXX10 with flat vs adaptive sizing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final

from src.policy.adaptive_contribution import OPERATIONAL_ADAPTIVE_CONTRIBUTION
from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.sim.allocation import AllocationResult

COMPOUND_DCA_ARM_IDS: Final[tuple[str, ...]] = (
    "qqq_flat",
    "qqq_adaptive_v5",
    "qqq90_soxx10_flat",
    "qqq90_soxx10_adaptive_v5",
)

COMPOUND_DCA_MIX_TARGETS: Final[dict[str, float]] = {"QQQ": 0.9, "SOXX": 0.1}

COMPOUND_DCA_WINDOW: Final[tuple[date, date]] = (date(2016, 7, 1), date(2026, 6, 30))


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
    """Reporting-only tournament outcome across four arms."""

    rows: tuple[CompoundDcaArmRow, ...]
    champion_arm_id: str
    operational_unlock: bool
    start: date
    end: date
    contribution_krw: float


def compare_compound_dca(
    *,
    runner: Callable[[AllocationConfig], AllocationResult],
    contribution_krw: float,
    start: date | None = None,
    end: date | None = None,
) -> CompoundDcaReport:
    """Run four QQQ arms on identical windows with I5 sizing-family checks.

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
        is_mix = arm_id.startswith("qqq90_soxx10")
        is_adaptive = arm_id.endswith("adaptive_v5")
        targets = dict(COMPOUND_DCA_MIX_TARGETS) if is_mix else None
        adaptive = OPERATIONAL_ADAPTIVE_CONTRIBUTION if is_adaptive else None
        cfg = AllocationConfig(
            policy=PolicyId.QQQ,
            start=window_start,
            end=window_end,
            monthly_contribution_krw=float(contribution_krw),
            adaptive_contribution=adaptive,
            targets_override=targets,
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

    flat_pair = (credits("qqq_flat"), credits("qqq90_soxx10_flat"))
    if flat_pair[0] != flat_pair[1]:
        raise ValueError(f"I5 flat credits must be identical: qqq_flat vs qqq90_soxx10_flat mismatch {flat_pair[0]!r} vs {flat_pair[1]!r}")

    adaptive_pair = (credits("qqq_adaptive_v5"), credits("qqq90_soxx10_adaptive_v5"))
    if adaptive_pair[0] != adaptive_pair[1]:
        raise ValueError(f"I5 adaptive credits must be identical: qqq_adaptive_v5 vs qqq90_soxx10_adaptive_v5 mismatch {adaptive_pair[0]!r} vs {adaptive_pair[1]!r}")

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
        if row.real_gain > best:
            best = row.real_gain
            champion = row.arm_id

    return CompoundDcaReport(
        rows=tuple(rows),
        champion_arm_id=champion,
        operational_unlock=False,
        start=window_start,
        end=window_end,
        contribution_krw=float(contribution_krw),
    )
