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
        experiment_path=Path("configs/research/m_thesis_ai_compute_soxx_120m.json"),
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


@pytest.mark.parametrize("scenario_id", ["test_run_thesis_pipeline_writes_result_store"])
def test_run_thesis_pipeline_writes_result_store(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline artifacts land under results/thesis_<id>/ without touching docs/."""
    import src.analytics.thesis.incremental as incremental_mod
    import src.analytics.thesis.wave as wave_mod
    import src.analytics.wave_d_exit as wde_mod
    import src.data.panel_freshness as panel_mod
    from src.data.panel_freshness import CatalogPanelReport, PanelFreshnessStatus
    from src.data.settings import DataSettings

    monkeypatch.chdir(tmp_path)
    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2026, 6, 30, 20, 0, tzinfo=UTC)
    wave = _make_wave()
    inc = _make_incremental()

    def fake_panel(settings, reference_now=None):  # type: ignore[no-untyped-def]
        return CatalogPanelReport(
            panel_as_of=as_of,
            lag_days=1,
            status=PanelFreshnessStatus.FRESH,
            ticker_last_session={},
            cpi_last_observation=None,
            fx_last_observation=None,
            holdings_last_filing=None,
        )

    monkeypatch.setattr(panel_mod, "resolve_catalog_panel_as_of", fake_panel)
    monkeypatch.setattr(wave_mod, "run_thesis_wave", lambda **kwargs: wave)  # type: ignore[arg-type]
    monkeypatch.setattr(incremental_mod, "run_incremental_portfolio", lambda **kwargs: inc)  # type: ignore[arg-type]

    code = wde_mod.run_thesis_pipeline_command(thesis_id="ai_compute", as_of=None, settings=settings)
    assert code == 0
    assert not (tmp_path / "docs").exists()
    artifacts = list((tmp_path / "data" / "runs" / "thesis_ai_compute").glob("wave_d_exit_*.json"))
    assert len(artifacts) == 1
    assert artifacts[0].with_suffix(".md").is_file()


def test_catalog_max_price_session_reads_pinned_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The as-of guard reads the last pinned PRICES session from one snapshot."""
    from datetime import date as _date

    import polars as _pl

    from src.analytics.thesis.wave_d_exit import _catalog_max_price_session
    from src.data.calendar import load_calendar as _load_calendar
    from src.data.pipeline import persist_ingest as _persist
    from src.data.schema import Dataset as _Dataset
    from src.data.schema import spec_for as _spec_for
    from src.data.settings import DataSettings as _Settings
    from src.data.storage import RawPayload as _Payload

    monkeypatch.chdir(tmp_path)
    settings = _Settings(data_root="data")
    retrieved_at = datetime(2024, 1, 5, 5, 0, tzinfo=UTC)
    sessions = list(_load_calendar("XNYS").sessions(_date(2024, 1, 2), _date(2024, 1, 31)))
    _persist(
        _pl.DataFrame(
            {
                "ticker": ["QQQ"] * len(sessions),
                "date": sessions,
                "open": [100.0] * len(sessions),
                "high": [101.0] * len(sessions),
                "low": [99.0] * len(sessions),
                "close": [100.0] * len(sessions),
                "volume": [10_000] * len(sessions),
                "adjusted_close": [100.0] * len(sessions),
                "dividend": [0.0] * len(sessions),
                "split_factor": [1.0] * len(sessions),
                "source": ["synthetic"] * len(sessions),
                "retrieved_at": [retrieved_at] * len(sessions),
            },
            schema=dict(_spec_for(_Dataset.PRICES).columns),
        ),
        _Dataset.PRICES,
        _Payload(provider="synthetic", endpoint="probe", request_params={},
                 retrieved_at=retrieved_at, extension="json", content=b"{}"),
        settings,
    )
    assert _catalog_max_price_session(settings) == sessions[-1]


def test_run_thesis_pipeline_explicit_as_of_checks_pinned_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit as-of at the last pinned session passes the snapshot guard."""
    from datetime import date as _date

    import polars as _pl

    import src.analytics.thesis.incremental as incremental_mod
    import src.analytics.thesis.wave as wave_mod
    import src.analytics.wave_d_exit as wde_mod
    import src.data.panel_freshness as panel_mod
    from src.data.calendar import load_calendar as _load_calendar
    from src.data.panel_freshness import CatalogPanelReport, PanelFreshnessStatus
    from src.data.pipeline import persist_ingest as _persist
    from src.data.schema import Dataset as _Dataset
    from src.data.schema import spec_for as _spec_for
    from src.data.settings import DataSettings
    from src.data.storage import RawPayload as _Payload

    monkeypatch.chdir(tmp_path)
    settings = DataSettings(data_root=tmp_path / "data")
    sessions = list(_load_calendar("XNYS").sessions(_date(2024, 1, 2), _date(2024, 1, 31)))
    retrieved_at = datetime(2024, 1, 5, 5, 0, tzinfo=UTC)
    _persist(
        _pl.DataFrame(
            {
                "ticker": ["QQQ"] * len(sessions),
                "date": sessions,
                "open": [100.0] * len(sessions),
                "high": [101.0] * len(sessions),
                "low": [99.0] * len(sessions),
                "close": [100.0] * len(sessions),
                "volume": [10_000] * len(sessions),
                "adjusted_close": [100.0] * len(sessions),
                "dividend": [0.0] * len(sessions),
                "split_factor": [1.0] * len(sessions),
                "source": ["synthetic"] * len(sessions),
                "retrieved_at": [retrieved_at] * len(sessions),
            },
            schema=dict(_spec_for(_Dataset.PRICES).columns),
        ),
        _Dataset.PRICES,
        _Payload(provider="synthetic", endpoint="probe", request_params={},
                 retrieved_at=retrieved_at, extension="json", content=b"{}"),
        settings,
    )
    wave = _make_wave()
    inc = _make_incremental()
    as_of = datetime(2024, 1, 31, 20, 0, tzinfo=UTC)

    def fake_panel(settings, reference_now=None):  # type: ignore[no-untyped-def]
        return CatalogPanelReport(
            panel_as_of=as_of,
            lag_days=1,
            status=PanelFreshnessStatus.FRESH,
            ticker_last_session={},
            cpi_last_observation=None,
            fx_last_observation=None,
            holdings_last_filing=None,
        )

    monkeypatch.setattr(panel_mod, "resolve_catalog_panel_as_of", fake_panel)
    monkeypatch.setattr(wave_mod, "run_thesis_wave", lambda **kwargs: wave)  # type: ignore[arg-type]
    monkeypatch.setattr(incremental_mod, "run_incremental_portfolio", lambda **kwargs: inc)  # type: ignore[arg-type]

    code = wde_mod.run_thesis_pipeline_command(
        thesis_id="ai_compute", as_of="2024-01-31", settings=settings
    )
    assert code == 0
    artifacts = list((tmp_path / "data" / "runs" / "thesis_ai_compute").glob("wave_d_exit_*.json"))
    assert len(artifacts) == 1
