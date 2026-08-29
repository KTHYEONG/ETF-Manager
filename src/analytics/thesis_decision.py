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
    # Prospective eligibility has top priority
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

    # Extract median and CE from divergence or evidence
    median: float | None = None
    ce_ratio: float | None = None
    lh_passes: bool | None = None

    if report.divergence is not None:
        try:
            median = float(report.divergence.get("median_ratio", report.divergence.get("historical_median_ratio", 0)))  # type: ignore[arg-type]
        except Exception:
            median = None
        try:
            ce_ratio = float(report.divergence.get("ce_ratio_gamma_2"))  # type: ignore[arg-type]
        except Exception:
            ce_ratio = None
        try:
            lh_passes = bool(report.divergence.get("long_horizon_passes"))
        except Exception:
            lh_passes = None

    # Fallback median from historical slot
    if median is None:
        try:
            if report.evidence.historical.status == "computed":
                median = float(report.evidence.historical.metrics.get("median_ratio", 0))
            elif report.long_horizon is not None:
                median = float(report.long_horizon.median_ratio)
        except Exception:
            median = None

    # Fallback ce from divergence only; if missing we cannot evaluate watch/reject gates that need ce
    if lh_passes is None and report.long_horizon is not None:
        lh_passes = bool(report.long_horizon.passes)

    # Watch: median>=1.0 and ce<1.02 and not lh pass
    if median is not None and ce_ratio is not None and lh_passes is not None and median >= 1.0 and ce_ratio < 1.02 and not lh_passes:
        return ThesisDecisionRecord(
            decision=ThesisDecision.WATCH,
            rationale=f"watch divergence median {median:.4f} ce {ce_ratio:.4f} lh_passes {lh_passes}",
            metrics={
                "median_ratio": float(median),
                "ce_ratio_gamma_2": float(ce_ratio),
                "long_horizon_passes": bool(lh_passes),
            },
        )

    # Reject: ce<0.98 and median<1.0
    if median is not None and ce_ratio is not None and ce_ratio < 0.98 and median < 1.0:
        return ThesisDecisionRecord(
            decision=ThesisDecision.REJECT,
            rationale=f"reject weak ce {ce_ratio:.4f} median {median:.4f}",
            metrics={
                "median_ratio": float(median),
                "ce_ratio_gamma_2": float(ce_ratio),
            },
        )

    # Default continue_research
    rationale = "continue_research"
    metrics: dict[str, float | int | str | bool] = {}
    if median is not None:
        metrics["median_ratio"] = float(median)
    if ce_ratio is not None:
        metrics["ce_ratio_gamma_2"] = float(ce_ratio)
    if lh_passes is not None:
        metrics["long_horizon_passes"] = bool(lh_passes)
    metrics["prospective_eligible"] = False
    return ThesisDecisionRecord(
        decision=ThesisDecision.CONTINUE_RESEARCH,
        rationale=rationale,
        metrics=metrics,
    )
