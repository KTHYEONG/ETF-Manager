"""Thesis decision synthesis (Wave 7)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from src.analytics.thesis_report import ThesisReport

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


def synthesize_thesis_decision(report: ThesisReport) -> ThesisDecisionRecord:
    """Synthesize decision from report without calling adoption gate."""
    if report.prospective.eligible:
        return ThesisDecisionRecord(
            decision=ThesisDecision.PROSPECTIVE,
            rationale=f"prospective eligible: {report.prospective.reason}",
            metrics={
                "prospective_eligible": True,
                "catalog_span_years": float(report.prospective.catalog_span_years),
                "min_years_required": int(report.prospective.min_years_required),
            },
        )

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

    if median is not None and cohort_ce is not None and lh_passes is not None and median >= 1.0 and cohort_ce < 1.02 and not lh_passes:
        return ThesisDecisionRecord(
            decision=ThesisDecision.WATCH,
            rationale=f"watch median {median:.4f} cohort_ce {cohort_ce:.4f} lh_passes {lh_passes}",
            metrics={
                "median_ratio": float(median),
                "cohort_ce_ratio_gamma_2": float(cohort_ce),
                "long_horizon_passes": bool(lh_passes),
            },
        )

    if median is not None:
        if cohort_ce is not None:
            if cohort_ce < 0.98 and median < 1.0:
                return ThesisDecisionRecord(
                    decision=ThesisDecision.REJECT,
                    rationale=f"reject weak median {median:.4f} cohort_ce {cohort_ce:.4f}",
                    metrics={
                        "median_ratio": float(median),
                        "cohort_ce_ratio_gamma_2": float(cohort_ce),
                    },
                )
        else:
            if median < 0.98:
                return ThesisDecisionRecord(
                    decision=ThesisDecision.REJECT,
                    rationale=f"reject weak median {median:.4f} median-only",
                    metrics={
                        "median_ratio": float(median),
                    },
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
    )
