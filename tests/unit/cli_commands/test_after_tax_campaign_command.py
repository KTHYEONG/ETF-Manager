"""End-to-end tests of ``run_after_tax_campaign_command`` against a tiny synthetic lake."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

import src.cli_commands.campaign as campaign_mod
from src.data.calendar import load_calendar
from src.data.paths import results_root
from src.data.pipeline import persist_ingest
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import RawPayload

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


def _campaign_config(*, start: str = "2024-01-01", end: str = "2025-06-30") -> dict[str, object]:
    return {
        "name": "after_tax_cli_smoke",
        "start": start,
        "end": end,
        "contribution_krw": 1_000_000,
        "commission_bps": 10.0,
        "fx_spread_bps": 20.0,
        "tax_regime_path": "configs/tax/kr_overseas_equity.json",
        "horizons_months": [12],
        "primary_horizon_months": 12,
        "step_months": 6,
        "crash_regimes": ["covid"],
        "crash_window_months": 36,
        "fractional_shares": False,
        "bootstrap_paths": 20,
        "baseline_arm_id": "b1_qqq",
        "arms": [
            {
                "arm_id": "b1_qqq",
                "role": "baseline",
                "rule": {"rule_id": "static", "core_targets": {"QQQ": 1.0}},
                "mode": "buy_only",
                "rebalance_band": None,
                "harvest_gains": False,
            },
            {
                "arm_id": "c1_qqq_tgh",
                "role": "operational_candidate",
                "rule": {"rule_id": "static", "core_targets": {"QQQ": 1.0}},
                "mode": "buy_only",
                "rebalance_band": None,
                "harvest_gains": True,
            },
        ],
        "gate": {"median_ratio_floor": 1.0, "worst_ratio_floor": 0.97, "bootstrap_p05_floor": 1.0},
    }


def _persist_lake(settings: DataSettings, *, with_rates: bool = True, with_fx: bool = True) -> None:
    window = load_calendar("XNYS").sessions(date(2023, 1, 2), date(2025, 12, 31))
    closes = [100.0 + 0.2 * index for index in range(len(window))]
    n = len(window)
    prices = pl.DataFrame(
        {
            "ticker": ["QQQ"] * n,
            "date": list(window),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [10_000] * n,
            "adjusted_close": closes,
            "dividend": [0.0] * n,
            "split_factor": [1.0] * n,
            "source": ["synthetic"] * n,
            "retrieved_at": [_RETRIEVED_AT] * n,
        },
        schema=dict(spec_for(Dataset.PRICES).columns),
    )
    persist_ingest(prices, Dataset.PRICES, _payload(), settings)
    if with_fx:
        fx = pl.DataFrame(
            {
                "date": list(window),
                "usdkrw": [1300.0] * n,
                "source": ["synthetic"] * n,
                "retrieved_at": [_RETRIEVED_AT] * n,
            },
            schema=dict(spec_for(Dataset.FX_KRW_BASE).columns),
        )
        persist_ingest(fx, Dataset.FX_KRW_BASE, _payload(), settings)
    months = [date(year, month, 1) for year in (2023, 2024, 2025) for month in range(1, 13)]
    cpi = pl.DataFrame(
        {
            "period_end": months,
            "value": [100.0 + 0.1 * index for index in range(len(months))],
            "source": ["synthetic"] * len(months),
            "retrieved_at": [_RETRIEVED_AT] * len(months),
        },
        schema=dict(spec_for(Dataset.CPI).columns),
    )
    persist_ingest(cpi, Dataset.CPI, _payload(), settings)
    if with_rates:
        rates = pl.DataFrame(
            {
                "series_id": ["DTB3"],
                "observation_date": [date(2023, 1, 3)],
                "value": [4.0],
                "source": ["synthetic"],
                "retrieved_at": [_RETRIEVED_AT],
            },
            schema=dict(spec_for(Dataset.RATES).columns),
        )
        persist_ingest(rates, Dataset.RATES, _payload(), settings)


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[DataSettings, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    tax_dir = tmp_path / "configs" / "tax"
    tax_dir.mkdir(parents=True)
    (tax_dir / "kr_overseas_equity.json").write_text(
        (repo_root / "configs" / "tax" / "kr_overseas_equity.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(campaign_mod, "_resolve_git_commit", lambda: _GIT_COMMIT)
    config_path = tmp_path / "campaign.json"
    config_path.write_text(json.dumps(_campaign_config()), encoding="utf-8")
    return DataSettings(data_root="data"), config_path


def test_after_tax_campaign_command_writes_reports(workspace: tuple[DataSettings, Path]) -> None:
    """Given a trusted lake, the command runs every arm and persists JSON plus markdown."""
    settings, config_path = workspace
    _persist_lake(settings)

    code = campaign_mod.run_after_tax_campaign_command(config_path=str(config_path), settings=settings, seed=7)

    assert code == 0
    reports = sorted((results_root(settings) / "after_tax_cli_smoke").glob("after_tax_*.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["operational_unlock"] is False
    arms = {summary["arm_id"] for summary in payload["summaries"]}
    assert arms == {"b1_qqq", "c1_qqq_tgh"}
    baseline = next(summary for summary in payload["summaries"] if summary["arm_id"] == "b1_qqq")
    assert baseline["median_ratio"] == pytest.approx(1.0)
    assert reports[0].with_suffix(".md").exists()


def test_after_tax_campaign_command_experiment_id_is_deterministic(workspace: tuple[DataSettings, Path]) -> None:
    """The report filename derives from config bytes, commit, and data manifest only."""
    settings, config_path = workspace
    _persist_lake(settings)

    assert campaign_mod.run_after_tax_campaign_command(config_path=str(config_path), settings=settings, seed=1) == 0
    first = sorted(p.name for p in (results_root(settings) / "after_tax_cli_smoke").glob("after_tax_*.json"))
    assert campaign_mod.run_after_tax_campaign_command(config_path=str(config_path), settings=settings, seed=2) == 0
    second = sorted(p.name for p in (results_root(settings) / "after_tax_cli_smoke").glob("after_tax_*.json"))

    assert first == second
    assert len(first) == 1


def test_after_tax_campaign_command_missing_dataset_returns_one(
    workspace: tuple[DataSettings, Path], caplog: pytest.LogCaptureFixture
) -> None:
    """A lake without RATES fails closed and logs the failure class instead of raising."""
    settings, config_path = workspace
    _persist_lake(settings, with_rates=False)

    with caplog.at_level("ERROR"):
        code = campaign_mod.run_after_tax_campaign_command(config_path=str(config_path), settings=settings, seed=7)

    assert code == 1
    assert "after_tax_campaign_cli_failed" in caplog.text
    assert not list((results_root(settings) / "after_tax_cli_smoke").glob("after_tax_*.json"))


def test_after_tax_campaign_command_empty_schedule_returns_one(workspace: tuple[DataSettings, Path]) -> None:
    """A weekend-only window has no month-end session, which is an AfterTaxDataError, mapped to exit code 1."""
    settings, config_path = workspace
    _persist_lake(settings)
    config_path.write_text(
        json.dumps(_campaign_config(start="2024-07-06", end="2024-07-07")), encoding="utf-8"
    )

    assert campaign_mod.run_after_tax_campaign_command(config_path=str(config_path), settings=settings, seed=7) == 1


def test_after_tax_campaign_command_invalid_spec_returns_one(workspace: tuple[DataSettings, Path]) -> None:
    """Spec validation errors surface as exit code 1 without touching the lake."""
    settings, config_path = workspace
    bad = _campaign_config()
    bad["baseline_arm_id"] = "missing_arm"
    config_path.write_text(json.dumps(bad), encoding="utf-8")

    assert campaign_mod.run_after_tax_campaign_command(config_path=str(config_path), settings=settings, seed=7) == 1
