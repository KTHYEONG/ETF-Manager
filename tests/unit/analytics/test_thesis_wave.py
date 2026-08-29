"""Thesis wave tests (Wave 7)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.data.settings import DataSettings
from src.policy.thesis import ThesisId
from src.sim.allocation import AllocationConfig, AllocationResult, AllocationSnapshot


@pytest.mark.parametrize("scenario_id", ["WAVE-A-three-entries"])
def test_wave_a_three_entries(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """WAVE-A-three-entries"""
    from src.analytics.thesis_wave import run_thesis_wave

    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2025, 4, 30, tzinfo=UTC)

    # Mock build_thesis_report to return distinct reports
    from src.analytics.thesis_evidence import EvidenceSlot, EvidenceSnapshot
    from src.analytics.thesis_report import ThesisReport
    from src.policy.thesis import ThesisStatus
    from src.validation.prospective import ProspectiveEligibility

    def fake_build(*, thesis_id, settings, as_of, runner, experiment_path=None, include_regime=False):
        slot = EvidenceSlot(status="computed", summary="ok", metrics={"median_ratio": 1.0, "overlap_pct": 10.0})
        snap = EvidenceSnapshot(thesis_id=thesis_id, as_of=as_of, historical=slot, structural=slot, valuation=slot, overlap=slot, crowding=slot)
        return ThesisReport(
            thesis_id=thesis_id,
            evidence=snap,
            long_horizon=None,
            prospective=ProspectiveEligibility(eligible=False, catalog_span_years=8.0, min_years_required=5, reason="test"),
            suggested_status=ThesisStatus.RESEARCH,
            next_falsifier="f1",
            divergence=None,
        )

    monkeypatch.setattr("src.analytics.thesis_wave.build_thesis_report", fake_build)

    # Need experiment_map handling: ensure default map points to existing files (use tmp map)
    # Create a temporary experiment_map file at configs/theses/experiment_map.json via monkeypatching loader
    import src.analytics.thesis_wave as tw

    def fake_load_map(path=Path("configs/theses/experiment_map.json")):
        return {
            ThesisId.AI_COMPUTE: Path("configs/experiments/m_thesis_ai_compute_soxx_120m.json"),
            ThesisId.AI_POWER_BOTTLENECK: Path("configs/experiments/m_thesis_ai_power_bottleneck_grid.json"),
            ThesisId.PHYSICAL_AUTOMATION: Path("configs/experiments/m_thesis_physical_automation_botz_prospective.json"),
        }

    monkeypatch.setattr(tw, "load_thesis_experiment_map", fake_load_map)

    def runner(config: AllocationConfig) -> AllocationResult:
        return AllocationResult(
            config=config,
            snapshots=(AllocationSnapshot(session=date(2024, 1, 31), cash_krw=0, cash_usd=0, shares={}, mark_krw=100.0, contribution_krw=0, fees_krw=0),),
            terminal_wealth_krw=100.0,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=100.0,
            xirr_real=0.0,
        )

    wave = run_thesis_wave(settings=settings, as_of=as_of, runner=runner)
    assert len(wave.entries) == 3
    assert [e.thesis_id for e in wave.entries] == [ThesisId.AI_COMPUTE, ThesisId.AI_POWER_BOTTLENECK, ThesisId.PHYSICAL_AUTOMATION]


@pytest.mark.parametrize("scenario_id", ["WAVE-B-experiment-map-fail"])
def test_wave_b_experiment_map_fail(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """WAVE-B-experiment-map-fail"""
    from src.analytics.thesis_wave import load_thesis_experiment_map, run_thesis_wave

    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2025, 4, 30, tzinfo=UTC)

    # Create map file missing ai_power_bottleneck
    bad_path = tmp_path / "bad_map.json"
    import json

    bad_path.write_text(json.dumps({"ai_compute": "configs/experiments/m_thesis_ai_compute_soxx_120m.json", "physical_automation": "configs/experiments/m_thesis_physical_automation_botz_prospective.json"}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing"):  # noqa: PT011
        load_thesis_experiment_map(bad_path)

    # Also test run_thesis_wave fails closed when loader would miss key
    import src.analytics.thesis_wave as tw

    def fake_bad_load(path=Path("configs/theses/experiment_map.json")):
        return {
            ThesisId.AI_COMPUTE: Path("configs/experiments/m_thesis_ai_compute_soxx_120m.json"),
            # missing ai_power_bottleneck
            ThesisId.PHYSICAL_AUTOMATION: Path("configs/experiments/m_thesis_physical_automation_botz_prospective.json"),
        }

    monkeypatch.setattr(tw, "load_thesis_experiment_map", fake_bad_load)

    def runner(config: AllocationConfig) -> AllocationResult:
        return AllocationResult(
            config=config,
            snapshots=(AllocationSnapshot(session=date(2024, 1, 31), cash_krw=0, cash_usd=0, shares={}, mark_krw=100.0, contribution_krw=0, fees_krw=0),),
            terminal_wealth_krw=100.0,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=100.0,
            xirr_real=0.0,
        )

    with pytest.raises(ValueError, match="missing"):  # noqa: PT011
        run_thesis_wave(settings=settings, as_of=as_of, runner=runner)


@pytest.mark.parametrize("scenario_id", ["WAVE-C-markdown-panel-freshness"])
def test_wave_c_markdown_panel_freshness(scenario_id: str, tmp_path: Path) -> None:
    """WAVE-C-markdown-panel-freshness"""
    from src.analytics.thesis_wave import ThesisWaveReport, write_thesis_wave_markdown

    as_of = datetime(2026, 6, 30, 20, 0, tzinfo=UTC)
    panel_as_of = datetime(2026, 6, 30, 20, 0, tzinfo=UTC)
    wave = ThesisWaveReport(
        as_of=as_of,
        entries=(),
        failures=(),
        panel_as_of=panel_as_of,
        lag_days=29,
        freshness_status="FRESH",
    )
    md_path = tmp_path / "wave.md"
    write_thesis_wave_markdown(wave, md_path)
    text = md_path.read_text(encoding="utf-8")
    assert "lag_days: 29" in text
    assert "freshness_status: FRESH" in text
    assert f"panel_as_of: {panel_as_of.isoformat()}" in text
