"""Wave D exit assessment tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.analytics.incremental_portfolio import (
    BuyOnlyAttribution,
    IncrementalArmId,
    IncrementalArmReport,
    IncrementalPortfolioReport,
    PathBootstrapVerdict,
)
from src.analytics.thesis_evidence import EvidenceSlot, EvidenceSnapshot
from src.analytics.thesis_report import ThesisReport
from src.analytics.thesis_wave import ThesisWaveEntry, ThesisWaveReport
from src.analytics.thesis_meaning import PortfolioEvidenceStatus
from src.analytics.wave_d_exit import assess_wave_d_exit, write_wave_d_exit_markdown
from src.policy.thesis import ThesisId, ThesisStatus
from src.validation.prospective import ProspectiveEligibility


def _slot(status: str) -> EvidenceSlot:
    return EvidenceSlot(status=status, summary=f"{status}", metrics={"median_ratio": 1.1} if status == "computed" else {})  # type: ignore[arg-type]


def _make_wave(
    *,
    structural: str = "computed",
    valuation: str = "computed",
    crowding: str = "computed",
    freshness_status: str = "FRESH",
) -> ThesisWaveReport:
    as_of = datetime(2026, 6, 30, 20, 0, tzinfo=UTC)
    panel_as_of = datetime(2026, 6, 30, 20, 0, tzinfo=UTC)
    snap = EvidenceSnapshot(
        thesis_id=ThesisId.AI_COMPUTE,
        as_of=as_of,
        historical=_slot("computed"),
        structural=_slot(structural),
        valuation=_slot(valuation),
        overlap=_slot("computed"),
        crowding=_slot(crowding),
    )
    report = ThesisReport(
        thesis_id=ThesisId.AI_COMPUTE,
        evidence=snap,
        long_horizon=None,
        prospective=ProspectiveEligibility(eligible=False, catalog_span_years=10.0, min_years_required=5, reason="test"),
        suggested_status=ThesisStatus.RESEARCH,
        next_falsifier="f1",
        divergence=None,
    )
    from src.analytics.thesis_decision import ThesisDecision, ThesisDecisionRecord

    decision = ThesisDecisionRecord(decision=ThesisDecision.CONTINUE_RESEARCH, rationale="test", metrics={})
    entry = ThesisWaveEntry(
        thesis_id=ThesisId.AI_COMPUTE,
        report=report,
        decision=decision,
        experiment_path=Path("configs/experiments/m_thesis_ai_compute_soxx_120m.json"),
    )
    return ThesisWaveReport(
        as_of=as_of,
        entries=(entry,),
        failures=(),
        panel_as_of=panel_as_of,
        lag_days=10,
        freshness_status=freshness_status,
    )


def _make_arm(cohort_count: int, win_ok: bool = True) -> IncrementalArmReport:
    verdict = PathBootstrapVerdict(n_paths=400, win_rate=0.6 if win_ok else 0.4, p05_terminal_ratio=1.0, ok=win_ok)
    attr = BuyOnlyAttribution(
        target_soxx_weight=0.05,
        mean_realized_soxx_weight=0.05,
        terminal_realized_soxx_weight=0.05,
        mean_abs_weight_drift=0.0,
        terminal_weight_drift=0.0,
        incremental_wealth_ratio=1.0,
    )
    return IncrementalArmReport(
        arm_id=IncrementalArmId.QQQ95_SOXX5,
        soxx_weight=0.05,
        median_ratio=1.05,
        p10_ratio=1.0,
        worst_ratio=0.99,
        win_rate=0.6,
        cohort_count=cohort_count,
        ce_gamma_2=1.05,
        ce_gamma_5=1.05,
        ce_gamma_10=1.05,
        attribution=attr,
        path_bootstrap=verdict,
    )


def _make_incremental(
    *,
    portfolio_status: PortfolioEvidenceStatus = PortfolioEvidenceStatus.HISTORICALLY_PROMISING,
    cohort_count: int = 10,
    freshness_status: str = "FRESH",
) -> IncrementalPortfolioReport:
    as_of = datetime(2026, 6, 30, 20, 0, tzinfo=UTC)
    panel_as_of = datetime(2026, 6, 30, 20, 0, tzinfo=UTC)
    # need 3 arms for realism; all same cohort_count
    arms = tuple(_make_arm(cohort_count) for _ in range(3))
    # adjust arm ids
    arms = (
        IncrementalArmReport(
            arm_id=IncrementalArmId.QQQ95_SOXX5,
            soxx_weight=0.05,
            median_ratio=1.05,
            p10_ratio=1.0,
            worst_ratio=0.99,
            win_rate=0.6,
            cohort_count=cohort_count,
            ce_gamma_2=1.05,
            ce_gamma_5=1.05,
            ce_gamma_10=1.05,
            attribution=arms[0].attribution,
            path_bootstrap=arms[0].path_bootstrap,
        ),
        IncrementalArmReport(
            arm_id=IncrementalArmId.QQQ90_SOXX10,
            soxx_weight=0.10,
            median_ratio=1.05,
            p10_ratio=1.0,
            worst_ratio=0.99,
            win_rate=0.6,
            cohort_count=cohort_count,
            ce_gamma_2=1.05,
            ce_gamma_5=1.05,
            ce_gamma_10=1.05,
            attribution=arms[1].attribution,
            path_bootstrap=arms[1].path_bootstrap,
        ),
        IncrementalArmReport(
            arm_id=IncrementalArmId.QQQ85_SOXX15,
            soxx_weight=0.15,
            median_ratio=1.05,
            p10_ratio=1.0,
            worst_ratio=0.99,
            win_rate=0.6,
            cohort_count=cohort_count,
            ce_gamma_2=1.05,
            ce_gamma_5=1.05,
            ce_gamma_10=1.05,
            attribution=arms[2].attribution,
            path_bootstrap=arms[2].path_bootstrap,
        ),
    )
    return IncrementalPortfolioReport(
        thesis_id="ai_compute",
        as_of=as_of,
        panel_as_of=panel_as_of,
        lag_days=10,
        freshness_status=freshness_status,
        arms=arms,
        portfolio_status=portfolio_status,
    )


@pytest.mark.parametrize("scenario_id", ["test_assess_wave_d_exit_reference_ready"])
def test_assess_wave_d_exit_reference_ready(scenario_id: str) -> None:
    wave = _make_wave(structural="computed", valuation="computed", crowding="computed", freshness_status="FRESH")
    inc = _make_incremental(portfolio_status=PortfolioEvidenceStatus.HISTORICALLY_PROMISING, cohort_count=10, freshness_status="FRESH")
    assessment = assess_wave_d_exit(thesis_id=ThesisId.AI_COMPUTE, wave=wave, incremental=inc)
    assert assessment.reference_slice_ready is True
    assert assessment.operational_challenger_ready is True
    assert assessment.track_f_complete is True


@pytest.mark.parametrize("scenario_id", ["test_assess_wave_d_exit_track_f_incomplete"])
def test_assess_wave_d_exit_track_f_incomplete(scenario_id: str) -> None:
    wave = _make_wave(structural="computed", valuation="unknown", crowding="computed")
    inc = _make_incremental(portfolio_status=PortfolioEvidenceStatus.HISTORICALLY_PROMISING, cohort_count=10)
    assessment = assess_wave_d_exit(thesis_id=ThesisId.AI_COMPUTE, wave=wave, incremental=inc)
    assert assessment.track_f_complete is False
    assert assessment.reference_slice_ready is False
    assert any("valuation" in b for b in assessment.blockers)


@pytest.mark.parametrize("scenario_id", ["test_assess_wave_d_exit_missing_thesis"])
def test_assess_wave_d_exit_missing_thesis(scenario_id: str) -> None:
    wave = _make_wave()
    inc = _make_incremental()
    with pytest.raises(ValueError, match="absent"):  # noqa: PT011
        assess_wave_d_exit(thesis_id=ThesisId.AI_POWER_BOTTLENECK, wave=wave, incremental=inc)


@pytest.mark.parametrize("scenario_id", ["test_write_wave_d_exit_markdown_evidence_table"])
def test_write_wave_d_exit_markdown_evidence_table(scenario_id: str, tmp_path: Path) -> None:
    wave = _make_wave()
    inc = _make_incremental()
    assessment = assess_wave_d_exit(thesis_id=ThesisId.AI_COMPUTE, wave=wave, incremental=inc)
    out = tmp_path / "wave_d.md"
    write_wave_d_exit_markdown(assessment, wave, inc, out)
    text = out.read_text(encoding="utf-8")
    for slot in ("structural", "valuation", "crowding", "overlap", "historical"):
        assert slot in text
    assert "reference_slice_ready" in text
