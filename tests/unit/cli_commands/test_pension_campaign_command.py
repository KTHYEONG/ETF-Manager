"""End-to-end tests of ``run_pension_campaign_command`` against a tiny synthetic lake."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import src.cli_commands.campaign as campaign_mod
from src import cli
from src.cli import main
from src.data.calendar import load_calendar
from src.data.catalog import latest_artifact
from src.data.paths import results_root
from src.data.pipeline import persist_ingest
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import RawPayload

_REPO = Path(__file__).resolve().parents[3]
_RETRIEVED_AT = datetime(2024, 1, 1, 5, 0, tzinfo=UTC)
_GIT_COMMIT = "0" * 40


def _payload() -> RawPayload:
    return RawPayload(
        provider="synthetic",
        endpoint="probe",
        request_params={},
        retrieved_at=_RETRIEVED_AT,
        extension="json",
        content=b"{}",
    )


def _persist_proxy_lake(settings: DataSettings) -> None:
    sessions = list(load_calendar("XNYS").sessions(date(2023, 1, 1), date(2024, 12, 31)))
    rows = []
    for ticker, base in (("SPY", 400.0), ("QQQ", 300.0)):
        for index, day in enumerate(sessions):
            price = base + 0.1 * index
            rows.append(
                {
                    "ticker": ticker, "date": day, "open": price, "high": price, "low": price,
                    "close": price, "volume": 10_000, "adjusted_close": price, "dividend": 0.0,
                    "split_factor": 1.0, "source": "synthetic", "retrieved_at": _RETRIEVED_AT,
                }
            )
    prices = pl.DataFrame(
        rows,
        schema={
            "ticker": pl.String, "date": pl.Date, "open": pl.Float64, "high": pl.Float64,
            "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64, "adjusted_close": pl.Float64,
            "dividend": pl.Float64, "split_factor": pl.Float64, "source": pl.String,
            "retrieved_at": pl.Datetime("us", "UTC"),
        },
    ).select(list(spec_for(Dataset.PRICES).columns))
    persist_ingest(prices, Dataset.PRICES, _payload(), settings)
    fx = pl.DataFrame(
        {"date": sessions, "usdkrw": [1300.0] * len(sessions), "source": ["synthetic"] * len(sessions),
         "retrieved_at": [_RETRIEVED_AT] * len(sessions)},
        schema=dict(spec_for(Dataset.FX_KRW_BASE).columns),
    )
    persist_ingest(fx, Dataset.FX_KRW_BASE, _payload(), settings)


def _profile_entry(profile_id: str, years: list[int], **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "profile_id": profile_id,
        "birth_date": "1985-01-01",
        "account_open_date": "2015-01-01",
        "pension_start_date": None,
        "income_kind": "wage",
        "annual_income_krw": {str(y): 50_000_000 for y in years},
        "remaining_national_tax_krw": {str(y): 5_000_000 for y in years},
        "remaining_local_tax_krw": {str(y): 5_000_000 for y in years},
        "other_private_pension_income_krw": {str(y): 0 for y in years},
    }
    entry.update(overrides)
    return entry


def _campaign_config(mode: str, tax_path: str, identity_path: str) -> dict[str, Any]:
    years = [2023, 2024]
    targets = {"379800": 1.0} if mode == "kr_live" else {"SPY": 1.0}
    arms = [{"arm_id": "base", "role": "baseline", "targets": targets}]
    if mode == "us_proxy":
        arms.append({"arm_id": "cand", "role": "candidate", "targets": {"QQQ": 1.0}})
    return {
        "name": "pension_cli_smoke",
        "start": "2023-01-01",
        "end": "2024-12-31",
        "market_mode": mode,
        "horizons_months": [12],
        "step_months": 12,
        "available_cash_events_krw": {f"{y}-01-03": 6_000_000 for y in years},
        "contribution_dates": {str(y): [f"{y}-01-15"] for y in years},
        "tax_credit_settlement_dates": {str(y): f"{y + 1}-05-31" for y in years},
        "retirement_start_year": 2030,
        "withdrawal_amounts_krw": {},
        "profiles": [_profile_entry("accum", years)],
        "tax_regime_path": tax_path,
        "etf_identity_path": identity_path,
        "baseline_arm_id": "base",
        "arms": arms,
        "execution_spread_bps": 0.0,
        "commission_bps": 0.0,
        "max_fx_age_days": 7,
        "max_cpi_age_days": 75,
        "extra_annual_drag_by_ticker": {},
    }


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[DataSettings, Path, Path]:
    for name in ("kr_pension_2026.json",):
        target = tmp_path / "configs" / "tax" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((_REPO / "configs" / "tax" / name).read_text(encoding="utf-8"), encoding="utf-8")
    identity_target = tmp_path / "configs" / "data" / "pension_etfs_2026.json"
    identity_target.parent.mkdir(parents=True, exist_ok=True)
    identity_target.write_text(
        (_REPO / "configs" / "data" / "pension_etfs_2026.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(campaign_mod, "_resolve_git_commit", lambda: _GIT_COMMIT)
    proxy_path = tmp_path / "proxy.json"
    proxy_path.write_text(
        json.dumps(_campaign_config("us_proxy", "configs/tax/kr_pension_2026.json", "configs/data/pension_etfs_2026.json")),
        encoding="utf-8",
    )
    live_path = tmp_path / "live.json"
    live_path.write_text(
        json.dumps(_campaign_config("kr_live", "configs/tax/kr_pension_2026.json", "configs/data/pension_etfs_2026.json")),
        encoding="utf-8",
    )
    return DataSettings(data_root="data"), proxy_path, live_path


def test_pension_campaign_command_is_reporting_only(
    workspace: tuple[DataSettings, Path, Path],
) -> None:
    """Given a trusted lake, the command writes JSON plus markdown and leaves prices untouched."""
    settings, proxy_path, _ = workspace
    _persist_proxy_lake(settings)
    before = latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256

    code = campaign_mod.run_pension_campaign_command(config_path=str(proxy_path), settings=settings, seed=7)

    assert code == 0
    reports = sorted((results_root(settings) / "pension_cli_smoke").glob("pension_*.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["evidence_status"] == "INSUFFICIENT_INDEPENDENT_20Y_EVIDENCE"
    assert payload["market_mode"] == "us_proxy"
    assert payload["market_coverage_start"] <= payload["market_coverage_end"]
    assert payload["provenance"]["git_commit"] == _GIT_COMMIT
    assert payload["provenance"]["seed"] == "7"
    assert len(payload["provenance"]["config_sha256"]) == 64
    assert len(payload["provenance"]["tax_regime_sha256"]) == 64
    assert {summary["arm_id"] for summary in payload["summaries"]} == {"base", "cand"}
    assert "adopt" not in json.dumps(payload)
    assert reports[0].with_suffix(".md").exists()
    assert latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256 == before


def test_pension_campaign_command_experiment_id_is_deterministic(
    workspace: tuple[DataSettings, Path, Path],
) -> None:
    """Config bytes, commit, manifests, regime, and seed pin the report filename."""
    settings, proxy_path, _ = workspace
    _persist_proxy_lake(settings)

    assert campaign_mod.run_pension_campaign_command(config_path=str(proxy_path), settings=settings, seed=1) == 0
    first = sorted(p.name for p in (results_root(settings) / "pension_cli_smoke").glob("pension_*.json"))
    assert campaign_mod.run_pension_campaign_command(config_path=str(proxy_path), settings=settings, seed=1) == 0
    second = sorted(p.name for p in (results_root(settings) / "pension_cli_smoke").glob("pension_*.json"))

    assert first == second
    assert len(first) == 1


def test_pension_campaign_command_missing_source_returns_one(
    workspace: tuple[DataSettings, Path, Path], caplog: pytest.LogCaptureFixture
) -> None:
    """KR_LIVE without a trusted Korean ETF partition exits nonzero with no fabricated history."""
    settings, _, live_path = workspace
    _persist_proxy_lake(settings)

    with caplog.at_level("ERROR"):
        code = campaign_mod.run_pension_campaign_command(config_path=str(live_path), settings=settings, seed=7)

    assert code == 1
    assert "pension_campaign_cli_failed" in caplog.text
    assert not list((results_root(settings) / "pension_cli_smoke").glob("pension_*.json"))


def test_pension_campaign_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pension-campaign target calls the command once with config and seed; seed is required."""
    captured: dict[str, object] = {}

    def fake_command(*, config_path: str, settings: object, seed: int) -> int:
        captured["config_path"] = config_path
        captured["seed"] = seed
        return 0

    monkeypatch.setattr(campaign_mod, "run_pension_campaign_command", fake_command)
    monkeypatch.setattr(cli, "run_pension_campaign_command", fake_command, raising=False)
    exit_code = main(["run", "pension-campaign", "--config", "c.json", "--seed", "7"])
    assert exit_code == 0
    assert captured == {"config_path": "c.json", "seed": 7}
    assert main(["run", "pension-campaign", "--config", "c.json"]) == 2
