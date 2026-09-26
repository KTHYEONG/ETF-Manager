"""Command-runner tests for thesis wave/incremental result-store wiring."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.cli_commands.thesis as thesis_mod
from src.data.panel_freshness import CatalogPanelReport, PanelFreshnessStatus
from src.data.settings import DataSettings


def test_thesis_incremental_writes_result_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Incremental artifacts land under results/thesis_<id>/ without touching docs/."""
    from src.analytics.incremental_portfolio import IncrementalPortfolioReport
    from src.analytics.thesis_meaning import PortfolioEvidenceStatus
    from src.policy.thesis import ThesisId

    monkeypatch.chdir(tmp_path)
    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2026, 6, 30, 20, 0, tzinfo=UTC)
    tid = ThesisId.AI_COMPUTE
    report = IncrementalPortfolioReport(
        thesis_id=tid.value,
        as_of=as_of,
        panel_as_of=as_of,
        lag_days=1,
        freshness_status="FRESH",
        arms=(),
        portfolio_status=PortfolioEvidenceStatus.HISTORICALLY_PROMISING,
    )

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

    monkeypatch.setattr(thesis_mod, "resolve_catalog_panel_as_of", fake_panel)
    monkeypatch.setattr(thesis_mod, "run_incremental_portfolio", lambda **kwargs: report)  # type: ignore[arg-type]

    import src.policy.thesis as thesis_policy

    monkeypatch.setattr(
        thesis_policy,
        "load_thesis_registry",
        lambda path: {tid: SimpleNamespace(historical_proxies=[])},
    )

    code = thesis_mod.run_thesis_incremental_command(
        thesis_id=tid.value, as_of=None, settings=settings, seed=7, bootstrap_paths=10
    )
    assert code == 0
    assert not (tmp_path / "docs").exists()
    artifacts = list((tmp_path / "data" / "runs" / "thesis_ai_compute").glob("thesis_incremental_*.json"))
    assert len(artifacts) == 1
