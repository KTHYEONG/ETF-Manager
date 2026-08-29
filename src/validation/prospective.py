"""Prospective OOS eligibility and paper-forward execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime

from src.data.settings import DataSettings
from src.execution.broker import PaperBroker
from src.policy.thesis import ThesisSpec
from src.sim.allocation import AllocationConfig, AllocationResult
from src.validation.experiment import ExperimentSpec

__all__ = [
    "ProspectiveEligibility",
    "ProspectiveFreezeRecord",
    "evaluate_prospective_eligibility",
    "run_prospective_paper_forward",
]


@dataclass(frozen=True, slots=True)
class ProspectiveEligibility:
    eligible: bool
    catalog_span_years: float
    min_years_required: int
    reason: str


@dataclass(frozen=True, slots=True)
class ProspectiveFreezeRecord:
    thesis_id: str
    frozen_at: datetime
    targets_hash: str
    experiment_name: str


def evaluate_prospective_eligibility(
    *, thesis: ThesisSpec, catalog_start: date, catalog_end: date
) -> ProspectiveEligibility:
    """Eligible when catalog span is shorter than thesis horizon."""
    days = (catalog_end - catalog_start).days
    span_years = days / 365.25
    min_years = int(thesis.horizon.min_years)
    eligible = span_years < float(min_years)
    reason = f"catalog_span {span_years:.2f}y {'<' if eligible else '>='} min_years {min_years}"
    return ProspectiveEligibility(
        eligible=eligible,
        catalog_span_years=float(span_years),
        min_years_required=int(min_years),
        reason=reason,
    )


def run_prospective_paper_forward(
    *,
    spec: ExperimentSpec,
    freeze: ProspectiveFreezeRecord,
    settings: DataSettings,
    runner: Callable[[AllocationConfig], AllocationResult],
) -> PaperBroker:
    """Run allocation from freeze date to spec end and reconcile paper lots."""
    from src.execution.broker import replay_paper
    from src.sim.allocation import AllocationConfig
    from src.validation.experiment import resolve_arm_targets
    from src.validation.registry import freeze_baseline_config_hash

    expected = freeze_baseline_config_hash(spec)
    if freeze.targets_hash != expected:
        raise ValueError(f"targets_hash mismatch {freeze.targets_hash!r} != {expected!r}")
    start = freeze.frozen_at.date()
    end = spec.end
    if start > end:
        raise ValueError(f"freeze start {start.isoformat()} is after spec end {end.isoformat()}")
    arm = spec.candidates[0] if spec.candidates else spec.baseline
    targets = resolve_arm_targets(arm)
    config = AllocationConfig(
        policy=arm.policy,
        start=start,
        end=end,
        monthly_contribution_krw=float(spec.contribution_krw),
        fill_delay_sessions=1,
        commission_bps=float(spec.commission_bps),
        fx_spread_bps=float(spec.fx_spread_bps),
        targets_override=targets,
    )
    result = runner(config)
    broker = replay_paper(result)
    return broker
