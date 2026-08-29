"""Thesis meaning classification tests (Wave B)."""
from __future__ import annotations

from src.analytics.thesis_meaning import (
    HistoricalQuality,
    PortfolioEvidenceStatus,
    ThesisEvidenceStatus,
    VehicleEvidenceStatus,
    classify_thesis_meaning,
)


def test_mean_a_soxx_like_active_thin_unresolved() -> None:
    snap = classify_thesis_meaning(
        span_years=18.67,
        min_years=5,
        target_years=10,
        primary_cohort_count=8,
        median_ratio=1.278,
        cohort_ce_ratio=1.20,
        overlap_dependence_disclosed=True,
    )
    assert snap.thesis_status == ThesisEvidenceStatus.UNRESOLVED
    assert snap.vehicle_status == VehicleEvidenceStatus.ACTIVE_PROXY
    assert snap.portfolio_status == PortfolioEvidenceStatus.UNVERIFIED
    assert snap.historical_quality == HistoricalQuality.TARGET_THIN
    assert snap.history_available is True
    assert snap.evidence_sufficient is True
    assert snap.thin_sample_warning is True


def test_mean_b_grid_like_rejected_proxy_thesis_unresolved() -> None:
    snap = classify_thesis_meaning(
        span_years=18.67,
        min_years=5,
        target_years=10,
        primary_cohort_count=6,
        median_ratio=0.788,
        cohort_ce_ratio=0.65,
        overlap_dependence_disclosed=True,
    )
    assert snap.vehicle_status == VehicleEvidenceStatus.REJECTED_PROXY
    assert snap.thesis_status == ThesisEvidenceStatus.UNRESOLVED
    assert snap.historical_quality == HistoricalQuality.TARGET_THIN


def test_mean_c_botz_partial_history() -> None:
    snap = classify_thesis_meaning(
        span_years=8.62,
        min_years=5,
        target_years=10,
        primary_cohort_count=None,
        median_ratio=0.613,
        cohort_ce_ratio=0.56,
        overlap_dependence_disclosed=True,
    )
    assert snap.historical_quality == HistoricalQuality.PARTIAL_HISTORY
    assert snap.history_available is True
    assert snap.evidence_sufficient is False
    assert snap.vehicle_status == VehicleEvidenceStatus.REJECTED_PROXY
    assert snap.thesis_status == ThesisEvidenceStatus.UNRESOLVED


def test_mean_d_never_target_robust_without_bootstrap() -> None:
    snap_thin = classify_thesis_meaning(
        span_years=18.67,
        min_years=5,
        target_years=10,
        primary_cohort_count=12,
        median_ratio=1.1,
        cohort_ce_ratio=1.05,
        overlap_dependence_disclosed=True,
        path_bootstrap_ok=False,
    )
    assert snap_thin.historical_quality == HistoricalQuality.TARGET_THIN
    assert snap_thin.historical_quality != HistoricalQuality.TARGET_ROBUST
    snap_robust = classify_thesis_meaning(
        span_years=18.67,
        min_years=5,
        target_years=10,
        primary_cohort_count=12,
        median_ratio=1.1,
        cohort_ce_ratio=1.05,
        overlap_dependence_disclosed=True,
        path_bootstrap_ok=True,
    )
    assert snap_robust.historical_quality == HistoricalQuality.TARGET_ROBUST


def test_inc_h7_meaning_default_unverified() -> None:
    snap = classify_thesis_meaning(
        span_years=18.67,
        min_years=5,
        target_years=10,
        primary_cohort_count=8,
        median_ratio=1.278,
        cohort_ce_ratio=1.20,
        overlap_dependence_disclosed=True,
    )
    assert snap.portfolio_status == PortfolioEvidenceStatus.UNVERIFIED
    # explicit portfolio_status should be honored
    snap2 = classify_thesis_meaning(
        span_years=18.67,
        min_years=5,
        target_years=10,
        primary_cohort_count=8,
        median_ratio=1.278,
        cohort_ce_ratio=1.20,
        overlap_dependence_disclosed=True,
        portfolio_status=PortfolioEvidenceStatus.HISTORICALLY_PROMISING,
    )
    assert snap2.portfolio_status == PortfolioEvidenceStatus.HISTORICALLY_PROMISING
