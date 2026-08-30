"""Thesis decision synthesis (Wave 7)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from src.analytics.thesis.meaning import (
    HistoricalQuality,
    ThesisMeaningSnapshot,
    VehicleEvidenceStatus,
    classify_thesis_meaning,
)
from src.analytics.thesis.report import ThesisReport

__all__ = ["ThesisDecision", "ThesisDecisionRecord", "synthesize_thesis_decision"]


class ThesisDecision(StrEnum):
    REJECT = "reject"
    WATCH = "watch"
    PROSPECTIVE = "prospective"
    CONTINUE_RESEARCH = "continue_research"


@dataclass(frozen=True, slots=True)
class ThesisDecisionRecord:
    decision: ThesisDecision
    rationale: str
    metrics: Mapping[str, float | int | str | bool]
    meaning: ThesisMeaningSnapshot | None = None


def _build_meaning(report: ThesisReport, median: float | None, cohort_ce: float | None) -> ThesisMeaningSnapshot:
    # span and horizon bounds
    span_years = float(report.prospective.catalog_span_years)
    min_years = int(report.prospective.min_years_required)
    target_years = 10
    try:
        if report.divergence is not None and report.divergence.get("target_years") is not None:
            target_years = int(str(report.divergence["target_years"]))
        else:
            from src.policy.thesis import get_thesis, load_thesis_registry

            reg = load_thesis_registry(Path("configs/theses"))
            thesis = get_thesis(reg, report.thesis_id)
            min_years = int(thesis.horizon.min_years)
            target_years = int(thesis.horizon.target_years)
    except Exception:  # noqa: BLE001,S110
        pass

    # primary cohort count: present only when evaluated_horizon exists
    primary_cohort_count: int | None = None
    try:
        evaluated = None
        if report.divergence is not None:
            evaluated = report.divergence.get("evaluated_horizon_months")
        if evaluated is not None and int(str(evaluated)) > 0:
            if report.divergence is not None and report.divergence.get("cohort_count") is not None:
                primary_cohort_count = int(str(report.divergence["cohort_count"]))
            elif report.long_horizon is not None:
                primary_cohort_count = int(report.long_horizon.cohort_count)
            else:
                primary_cohort_count = None
        else:
            primary_cohort_count = None
    except Exception:  # noqa: BLE001
        primary_cohort_count = None

    overlap_disclosed = False
    try:
        if report.long_horizon is not None:
            overlap_disclosed = bool(report.long_horizon.overlap_dependence_disclosed)
        elif report.divergence is not None and report.divergence.get("overlap_dependence_disclosed") is not None:
            overlap_disclosed = bool(report.divergence.get("overlap_dependence_disclosed"))
    except Exception:  # noqa: BLE001
        overlap_disclosed = False

    meaning = classify_thesis_meaning(
        span_years=span_years,
        min_years=min_years,
        target_years=target_years,
        primary_cohort_count=primary_cohort_count,
        median_ratio=median,
        cohort_ce_ratio=cohort_ce,
        overlap_dependence_disclosed=overlap_disclosed,
    )
    return meaning


def synthesize_thesis_decision(report: ThesisReport) -> ThesisDecisionRecord:
    """Synthesize decision from report without calling adoption gate."""
    median: float | None = None
    cohort_ce: float | None = None
    lh_passes: bool | None = None

    if report.divergence is not None:
        try:
            val = report.divergence.get("median_ratio", report.divergence.get("historical_median_ratio", None))
            median = float(val) if val is not None else None  # type: ignore[arg-type]
        except Exception:
            median = None
        try:
            raw = report.divergence.get("cohort_ce_ratio_gamma_2")
            cohort_ce = float(raw) if raw is not None else None  # type: ignore[arg-type]
        except Exception:
            cohort_ce = None
        try:
            v = report.divergence.get("long_horizon_passes")
            lh_passes = bool(v) if v is not None else None
        except Exception:
            lh_passes = None

    if median is None:
        try:
            if report.evidence.historical.status == "computed":
                raw_m = report.evidence.historical.metrics.get("median_ratio")
                if raw_m is not None:
                    median = float(raw_m)
            elif report.long_horizon is not None:
                median = float(report.long_horizon.median_ratio)
        except Exception:
            median = None

    if lh_passes is None and report.long_horizon is not None:
        lh_passes = bool(report.long_horizon.passes)

    meaning = _build_meaning(report, median, cohort_ce)

    if report.prospective.eligible or meaning.historical_quality == HistoricalQuality.PROSPECTIVE_ONLY:
        return ThesisDecisionRecord(
            decision=ThesisDecision.PROSPECTIVE,
            rationale=f"prospective eligible: {report.prospective.reason}",
            metrics={
                "prospective_eligible": True,
                "catalog_span_years": float(report.prospective.catalog_span_years),
                "min_years_required": int(report.prospective.min_years_required),
            },
            meaning=meaning,
        )

    # Vehicle reject takes precedence over watch
    if meaning.vehicle_status == VehicleEvidenceStatus.REJECTED_PROXY:
        return ThesisDecisionRecord(
            decision=ThesisDecision.REJECT,
            rationale=f"reject weak median {median:.4f} cohort_ce {cohort_ce:.4f}" if median is not None and cohort_ce is not None else f"reject vehicle {meaning.vehicle_status.value}",
            metrics={
                "median_ratio": float(median) if median is not None else 0.0,
                "cohort_ce_ratio_gamma_2": float(cohort_ce) if cohort_ce is not None else 0.0,
                "vehicle_status": meaning.vehicle_status.value,
            } if median is not None else {"vehicle_status": meaning.vehicle_status.value},
            meaning=meaning,
        )

    # Watch ignores long_horizon_passes (M-11)
    if median is not None and cohort_ce is not None and median >= 1.0 and cohort_ce < 1.02:
        return ThesisDecisionRecord(
            decision=ThesisDecision.WATCH,
            rationale=f"watch median {median:.4f} cohort_ce {cohort_ce:.4f} lh_passes {lh_passes}",
            metrics={
                "median_ratio": float(median),
                "cohort_ce_ratio_gamma_2": float(cohort_ce),
                "long_horizon_passes": bool(lh_passes) if lh_passes is not None else False,
            },
            meaning=meaning,
        )

    # Fallback median-only reject when cohort_ce missing
    if median is not None and cohort_ce is None and median < 0.98:  # noqa: SIM102
        return ThesisDecisionRecord(
            decision=ThesisDecision.REJECT,
            rationale=f"reject weak median {median:.4f} median-only",
            metrics={
                "median_ratio": float(median),
            },
            meaning=meaning,
        )

    if cohort_ce is not None and cohort_ce >= 1.02:
        rationale = f"continue_research median {median:.4f} cohort_ce {cohort_ce:.4f}" if median is not None else "continue_research"
    else:
        if median is not None and cohort_ce is not None:
            rationale = f"continue_research median {median:.4f} cohort_ce {cohort_ce:.4f}"
        elif median is not None:
            rationale = f"continue_research median {median:.4f}"
        else:
            rationale = "continue_research"
    metrics: dict[str, float | int | str | bool] = {}
    if median is not None:
        metrics["median_ratio"] = float(median)
    if cohort_ce is not None:
        metrics["cohort_ce_ratio_gamma_2"] = float(cohort_ce)
    if lh_passes is not None:
        metrics["long_horizon_passes"] = bool(lh_passes)
    metrics["prospective_eligible"] = False
    return ThesisDecisionRecord(
        decision=ThesisDecision.CONTINUE_RESEARCH,
        rationale=rationale,
        metrics=metrics,
        meaning=meaning,
    )
