"""Thesis decision tests (Wave 7)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.analytics.thesis_evidence import EvidenceSlot, EvidenceSnapshot
from src.analytics.thesis_report import ThesisReport
from src.policy.thesis import ThesisId, ThesisStatus
from src.validation.prospective import ProspectiveEligibility
from src.analytics.thesis_decision import ThesisDecision, synthesize_thesis_decision
from src.validation.gate import LongHorizonVerdict


def _make_report(
    *,
    prospective_eligible: bool = False,
    divergence: dict | None = None,
    long_horizon_passes: bool | None = None,
    long_horizon_median: float = 1.0,
    historical_median: float = 1.0,
) -> ThesisReport:
    slot_hist = EvidenceSlot(status="computed", summary="ok", metrics={"median_ratio": historical_median})
    slot = EvidenceSlot(status="computed", summary="ok", metrics={})
    snapshot = EvidenceSnapshot(
        thesis_id=ThesisId.AI_COMPUTE,
        as_of=datetime(2025, 4, 30, tzinfo=UTC),
        historical=slot_hist,
        structural=slot,
        valuation=slot,
        overlap=slot,
        crowding=slot,
    )
    prospective = ProspectiveEligibility(
        eligible=prospective_eligible,
        catalog_span_years=8.0 if not prospective_eligible else 3.0,
        min_years_required=5,
        reason="test",
    )
    long_horizon = None
    if long_horizon_passes is not None:
        long_horizon = LongHorizonVerdict(
            passes=long_horizon_passes,
            cohort_count=9,
            median_ratio=long_horizon_median,
            overlap_dependence_disclosed=True,
            reason="test",
        )
    # Build divergence if provided else None or synthesize
    div = divergence
    return ThesisReport(
        thesis_id=ThesisId.AI_COMPUTE,
        evidence=snapshot,
        long_horizon=long_horizon,
        prospective=prospective,
        suggested_status=ThesisStatus.RESEARCH,
        next_falsifier="f1",
        divergence=div,
    )


@pytest.mark.parametrize("scenario_id", ["DEC-A-prospective-priority"])
def test_dec_a_prospective_priority(scenario_id: str) -> None:
    """DEC-A-prospective-priority"""
    # prospective eligible True should return prospective regardless of CE ratio
    report = _make_report(
        prospective_eligible=True,
        divergence={"median_ratio": 0.9, "ce_ratio_gamma_2": 0.9, "long_horizon_passes": False},
        long_horizon_passes=False,
        long_horizon_median=0.9,
    )
    rec = synthesize_thesis_decision(report)
    assert rec.decision == ThesisDecision.PROSPECTIVE


@pytest.mark.parametrize("scenario_id", ["DEC-B-watch-divergence"])
def test_dec_b_watch_divergence(scenario_id: str) -> None:
    """DEC-B-watch-divergence"""
    report = _make_report(
        prospective_eligible=False,
        divergence={"median_ratio": 1.027, "ce_ratio_gamma_2": 0.997, "long_horizon_passes": False},
        long_horizon_passes=False,
        long_horizon_median=1.027,
    )
    rec = synthesize_thesis_decision(report)
    assert rec.decision == ThesisDecision.WATCH


@pytest.mark.parametrize("scenario_id", ["DEC-C-reject-weak"])
def test_dec_c_reject_weak(scenario_id: str) -> None:
    """DEC-C-reject-weak"""
    report = _make_report(
        prospective_eligible=False,
        divergence={"median_ratio": 0.98, "ce_ratio_gamma_2": 0.975, "long_horizon_passes": False},
        long_horizon_passes=False,
        long_horizon_median=0.98,
        historical_median=0.98,
    )
    rec = synthesize_thesis_decision(report)
    assert rec.decision == ThesisDecision.REJECT


@pytest.mark.parametrize("scenario_id", ["DEC-D-no-adoption"])
def test_dec_d_no_adoption(scenario_id: str) -> None:
    """DEC-D-no-adoption"""
    text_decision = Path("src/analytics/thesis_decision.py").read_text(encoding="utf-8")
    assert "adoption_passes" not in text_decision
    text_wave = Path("src/analytics/thesis_wave.py").read_text(encoding="utf-8")
    assert "adoption_passes" not in text_wave
