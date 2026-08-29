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
    "EvaluationHorizon",
    "HorizonSurfacePoint",
    "ProspectiveEligibility",
    "ProspectiveFreezeRecord",
    "evaluate_prospective_eligibility",
    "resolve_evaluation_horizon",
    "resolve_horizon_surface",
    "resolve_proxy_history_span",
    "run_prospective_paper_forward",
]


@dataclass(frozen=True, slots=True)
class ProspectiveEligibility:
    eligible: bool
    catalog_span_years: float
    min_years_required: int
    reason: str


@dataclass(frozen=True, slots=True)
class EvaluationHorizon:
    horizon_months: int
    target_months: int
    min_months: int
    span_years: float
    span_capped: bool
    reason: str


@dataclass(frozen=True, slots=True)
class HorizonSurfacePoint:
    horizon_months: int
    cohort_count: int


@dataclass(frozen=True, slots=True)
class ProspectiveFreezeRecord:
    thesis_id: str
    frozen_at: datetime
    targets_hash: str
    experiment_name: str


def evaluate_prospective_eligibility(
    *, thesis: ThesisSpec, catalog_start: date, catalog_end: date
) -> ProspectiveEligibility:
    """Eligible when history span is shorter than thesis horizon min_years."""
    days = (catalog_end - catalog_start).days
    span_years = days / 365.25
    required_years = int(thesis.horizon.min_years)
    eligible = span_years < float(required_years)
    reason = f"proxy_span {span_years:.2f}y {'<' if eligible else '>='} min_years {required_years}"
    return ProspectiveEligibility(
        eligible=eligible,
        catalog_span_years=float(span_years),
        min_years_required=required_years,
        reason=reason,
    )


def resolve_evaluation_horizon(
    *, thesis: ThesisSpec, catalog_start: date, catalog_end: date
) -> EvaluationHorizon | None:
    """Return target horizon only when rolling_cohorts(..., horizon_months=target) has length >=1.

    Returns ``None`` when ``span_years < min_years`` or target horizon yields no cohort.
    """

    from src.validation.windows import rolling_cohorts

    days = (catalog_end - catalog_start).days
    span_years = days / 365.25
    min_years = int(thesis.horizon.min_years)
    target_years = int(thesis.horizon.target_years)
    min_months = min_years * 12
    target_months = target_years * 12
    if span_years < float(min_years):
        return None
    cohorts = rolling_cohorts(
        catalog_start, catalog_end, horizon_months=target_months, step_months=12
    )
    if len(cohorts) >= 1:
        reason = (
            f"evaluated_horizon {target_months} target {target_months} "
            f"span_capped False span {span_years:.2f}y"
        )
        return EvaluationHorizon(
            horizon_months=int(target_months),
            target_months=int(target_months),
            min_months=int(min_months),
            span_years=float(span_years),
            span_capped=False,
            reason=reason,
        )
    return None


def resolve_horizon_surface(
    *, thesis: ThesisSpec, catalog_start: date, catalog_end: date
) -> tuple[HorizonSurfacePoint, ...]:
    """Emit points for each of 60,84,96,120 within [min_months,target_months] with cohort counts."""
    from src.validation.windows import rolling_cohorts

    min_months = int(thesis.horizon.min_years) * 12
    target_months = int(thesis.horizon.target_years) * 12
    points: list[HorizonSurfacePoint] = []
    for m in (60, 84, 96, 120):
        if min_months <= m <= target_months:
            cohorts = rolling_cohorts(catalog_start, catalog_end, horizon_months=m, step_months=12)
            points.append(HorizonSurfacePoint(horizon_months=int(m), cohort_count=len(cohorts)))
    return tuple(points)


def resolve_proxy_history_span(
    *,
    settings: DataSettings,
    thesis: ThesisSpec,
    as_of: datetime,
) -> tuple[date, date]:
    """Return first/last price session for the thesis primary proxy at ``as_of``."""
    import polars as pl

    from src.data.catalog import load_visible
    from src.data.schema import Dataset

    if not thesis.historical_proxies:
        raise ValueError("thesis has no historical_proxies")
    proxy = thesis.historical_proxies[0].value
    prices = load_visible(settings, Dataset.PRICES, as_of)
    ticker_prices = prices.filter(pl.col("ticker") == proxy)
    if ticker_prices.is_empty():
        raise ValueError(f"no price history for proxy {proxy!r}")
    start_raw = ticker_prices.get_column("date").min()
    end_raw = ticker_prices.get_column("date").max()
    if not isinstance(start_raw, date) or not isinstance(end_raw, date):
        raise ValueError(f"invalid price dates for proxy {proxy!r}")
    return start_raw, end_raw


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
