"""Thesis meaning vector (Wave B)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "HistoricalQuality",
    "PortfolioEvidenceStatus",
    "ThesisEvidenceStatus",
    "ThesisMeaningSnapshot",
    "VehicleEvidenceStatus",
    "classify_thesis_meaning",
]


class HistoricalQuality(StrEnum):
    PROSPECTIVE_ONLY = "prospective_only"
    PARTIAL_HISTORY = "partial_history"
    TARGET_THIN = "target_thin"
    TARGET_ROBUST = "target_robust"


class VehicleEvidenceStatus(StrEnum):
    ACTIVE_PROXY = "active_proxy"
    REJECTED_PROXY = "rejected_proxy"
    PENDING = "pending"


class ThesisEvidenceStatus(StrEnum):
    UNRESOLVED = "unresolved"


class PortfolioEvidenceStatus(StrEnum):
    UNVERIFIED = "unverified"


@dataclass(frozen=True, slots=True)
class ThesisMeaningSnapshot:
    thesis_status: ThesisEvidenceStatus
    vehicle_status: VehicleEvidenceStatus
    portfolio_status: PortfolioEvidenceStatus
    historical_quality: HistoricalQuality
    history_available: bool
    evidence_sufficient: bool
    thin_sample_warning: bool


def classify_thesis_meaning(
    *,
    span_years: float,
    min_years: int,
    target_years: int,
    primary_cohort_count: int | None,
    median_ratio: float | None,
    cohort_ce_ratio: float | None,
    overlap_dependence_disclosed: bool,
    path_bootstrap_ok: bool = False,
) -> ThesisMeaningSnapshot:
    """Pure classifier implementing M-2..M-6 and vehicle rules; TARGET_ROBUST only if path_bootstrap_ok."""
    history_available = float(span_years) >= float(min_years)
    evidence_sufficient = primary_cohort_count is not None and int(primary_cohort_count) >= 1

    if not history_available:
        historical_quality = HistoricalQuality.PROSPECTIVE_ONLY
    elif not evidence_sufficient:
        historical_quality = HistoricalQuality.PARTIAL_HISTORY
    elif bool(path_bootstrap_ok):
        historical_quality = HistoricalQuality.TARGET_ROBUST
    else:
        historical_quality = HistoricalQuality.TARGET_THIN

    thin_sample_warning = primary_cohort_count is not None and int(primary_cohort_count) < 10

    if median_ratio is not None and float(median_ratio) >= 1.0:
        vehicle_status = VehicleEvidenceStatus.ACTIVE_PROXY
    elif median_ratio is not None and float(median_ratio) < 1.0:
        if cohort_ce_ratio is not None:
            if float(cohort_ce_ratio) < 0.98:
                vehicle_status = VehicleEvidenceStatus.REJECTED_PROXY
            else:
                vehicle_status = VehicleEvidenceStatus.PENDING
        else:
            if float(median_ratio) < 0.98:
                vehicle_status = VehicleEvidenceStatus.REJECTED_PROXY
            else:
                vehicle_status = VehicleEvidenceStatus.PENDING
    else:
        vehicle_status = VehicleEvidenceStatus.PENDING

    # Silence unused param warning while keeping it for quality taxonomy
    _ = target_years
    _ = overlap_dependence_disclosed

    thesis_status = ThesisEvidenceStatus.UNRESOLVED
    portfolio_status = PortfolioEvidenceStatus.UNVERIFIED

    return ThesisMeaningSnapshot(
        thesis_status=thesis_status,
        vehicle_status=vehicle_status,
        portfolio_status=portfolio_status,
        historical_quality=historical_quality,
        history_available=bool(history_available),
        evidence_sufficient=bool(evidence_sufficient),
        thin_sample_warning=bool(thin_sample_warning),
    )
