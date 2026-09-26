"""CLI contract tests for the ISA household operating-policy command."""

from __future__ import annotations

from pathlib import Path

import pytest

import src.cli_commands.campaign as campaign_mod
from src import cli
from src.cli import main
from src.cli_commands.parser import _build_parser
from src.data.paths import results_root
from src.data.settings import DataSettings


def test_parser_registers_isa_household_target() -> None:
    """The isa-household target parses config, seed, and freeze."""
    args = _build_parser().parse_args(["run", "isa-household", "--config", "x.json", "--seed", "7", "--freeze"])
    assert args.target == "isa-household"
    assert args.config == "x.json"
    assert args.seed == 7
    assert args.freeze is True


def test_invalid_config_fails_closed(tmp_path: Path) -> None:
    """A nonexistent config returns 1 and writes nothing under the results root."""
    settings = DataSettings(data_root=str(tmp_path / "data"))
    code = campaign_mod.run_isa_household_command(
        config_path=str(tmp_path / "missing.json"),
        settings=settings,
        seed=7,
    )
    assert code == 1
    assert not list(results_root(settings).rglob("*.json"))


def test_dispatch_reaches_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """The isa-household target calls the command once with config, seed, and freeze."""
    captured: dict[str, object] = {}

    def fake_command(*, config_path: str, settings: object, seed: int, freeze: object) -> int:
        captured["config_path"] = config_path
        captured["settings"] = settings
        captured["seed"] = seed
        captured["freeze"] = freeze
        return 0

    monkeypatch.setattr(cli, "run_isa_household_command", fake_command)
    assert main(["run", "isa-household", "--config", "c.json", "--seed", "7", "--freeze"]) == 0
    assert captured["config_path"] == "c.json"
    assert captured["seed"] == 7
    assert captured["freeze"] is True
    assert isinstance(captured["settings"], DataSettings)
    assert main(["run", "isa-household", "--config", "c.json"]) == 2


"""Success-path harness below: synthetic snapshots wired through the real evaluators."""

import calendar as _calendar
import json as _json
from datetime import UTC as _UTC
from datetime import date as _date
from datetime import datetime as _datetime
from types import SimpleNamespace as _SimpleNamespace

import polars as _pl

import src.cli_commands.campaign as _campaign_mod
import src.data.catalog as _catalog_mod
import src.validation.isa_household_decision as _decision_mod
from src.data.schema import Dataset as _Dataset

_GIT_COMMIT = "0" * 40


def _month_ends(first: _date, count: int) -> list[_date]:
    out: list[_date] = []
    year, month = first.year, first.month
    for _ in range(count):
        out.append(_date(year, month, _calendar.monthrange(year, month)[1]))
        month += 1
        if month > 12:
            year += 1
            month = 1
    return out


def _prices_frame(months: list[_date]) -> _pl.DataFrame:
    tickers, dates, closes, stamps = [], [], [], []
    for ticker, rate in (("QQQ", 0.008), ("SCHD", 0.004)):
        for index, day in enumerate(months):
            tickers.append(ticker)
            dates.append(day)
            closes.append(100.0 * (1.0 + rate) ** index)
            stamps.append(_datetime(2000, 1, 1, tzinfo=_UTC))
    return _pl.DataFrame(
        {
            "ticker": tickers,
            "date": dates,
            "adjusted_close": closes,
            "available_at": stamps,
        },
        schema={
            "ticker": _pl.String,
            "date": _pl.Date,
            "adjusted_close": _pl.Float64,
            "available_at": _pl.Datetime("us", "UTC"),
        },
    )


def _research_frame(months: list[_date], *, losses: bool = False) -> _pl.DataFrame:
    series, ends, returns, stamps = [], [], [], []
    for series_id, rate in (("s_qqq", 0.008), ("s_schd", 0.004)):
        for index, day in enumerate(months):
            series.append(series_id)
            ends.append(day)
            if losses and index < len(months) // 2:
                returns.append(-0.02)
            elif losses:
                returns.append(0.025)
            else:
                returns.append(rate)
            stamps.append(_datetime(2000, 1, 1, tzinfo=_UTC))
    return _pl.DataFrame(
        {
            "series_id": series,
            "period_end": ends,
            "simple_return": returns,
            "available_at": stamps,
        },
        schema={
            "series_id": _pl.String,
            "period_end": _pl.Date,
            "simple_return": _pl.Float64,
            "available_at": _pl.Datetime("us", "UTC"),
        },
    )


def _write_pension_config(tmp_path: Path) -> str:
    document = {
        "name": "cli_probe",
        "benchmark_id": "inc",
        "candidates": {
            "inc": {
                "start_weights": {"QQQ": 0.9, "SCHD": 0.1},
                "end_weights": {"QQQ": 0.9, "SCHD": 0.1},
                "glide_years": 0,
            },
            "alt": {
                "start_weights": {"QQQ": 1.0},
                "end_weights": {"QQQ": 1.0},
                "glide_years": 0,
            },
        },
        "neighbors": {},
        "century_series": {"QQQ": "s_qqq", "SCHD": "s_schd"},
        "modern_start": "2000-01-31",
        "modern_end": "2007-12-31",
        "century_start": "1990-01-31",
        "century_end": "1997-12-31",
        "horizons_years": [6],
        "step_months": 12,
        "pre_retirement_months": 0,
        "primary_gamma": 1.0,
        "sensitivity_gammas": [],
        "equivalence_band": 0.005,
        "min_bootstrap_win_share": 0.6,
        "bootstrap_paths": 10,
        "bootstrap_block_months": 12,
        "annual_drag_by_sleeve": {},
        "tax_crosscheck_campaign_path": "configs/research/pension_campaign_v1.json",
        "tax_crosscheck_arm_map": {"x": "inc"},
        "review_every_months": 12,
        "lineage": {"related_trial_count": 0},
    }
    path = tmp_path / "pension_decision.json"
    path.write_text(_json.dumps(document), encoding="utf-8")
    return str(path)


def _write_record(tmp_path: Path) -> str:
    schedule = {
        "start_weights": {"QQQ": 0.9, "SCHD": 0.1},
        "end_weights": {"QQQ": 0.9, "SCHD": 0.1},
        "glide_years": 0,
    }
    document = {
        "record_id": "cli_rec",
        "frozen_at": "2026-08-31T00:00:00+00:00",
        "git_commit": _GIT_COMMIT,
        "config_sha256": "b" * 64,
        "manifest_hashes": {},
        "seen_history_cutoff": "2007-12-31",
        "status": "ADOPT_CANDIDATE",
        "incumbent_id": "inc",
        "incumbent_schedule": schedule,
        "equivalent_ids": [],
        "benchmark_id": "inc",
        "benchmark_schedule": schedule,
        "review_every_months": 12,
        "previous_record_id": None,
    }
    path = tmp_path / "record.json"
    path.write_text(_json.dumps(document), encoding="utf-8")
    return str(path)


def _write_household_config(
    tmp_path: Path, pension_config: str, record: str, *, weak: bool = False
) -> str:
    profile = {
        "profile_id": "cli",
        "birth_date": "1990-01-01",
        "pension_account_open_date": "2020-01-01",
        "initial_isa_tax_class": "general",
        "phases": [
            {
                "first_plan_year_offset": 0,
                "income_kind": "wage",
                "annual_income_krw": 30_000_000 if weak else 45_000_000,
                "remaining_national_tax_krw": 0 if weak else 10_000_000,
                "remaining_local_tax_krw": 0 if weak else 1_000_000,
            }
        ],
    }
    document = {
        "name": "cli_smoke",
        "pension_decision_config_path": pension_config,
        "pension_record_path": record,
        "isa_tax_regime_path": "configs/tax/kr_isa_2026.json",
        "pension_tax_regime_path": "configs/tax/kr_pension_2026.json",
        "overseas_tax_regime_path": "configs/tax/kr_overseas_equity.json",
        "plan_start_year": 2027,
        "horizons_years": [6],
        "step_months": 12,
        "pension_annual_krw": 6_000_000,
        "isa_budgets_krw": [6_000_000],
        "annuity_drawing_years": 10,
        "sensitivity_drawing_years": [20],
        "baseline_arm_id": "hold",
        "arms": {
            "hold": {"arm_id": "hold", "mode": "hold", "cycle_years": None},
            "all": {"arm_id": "all", "mode": "roll_to_pension_all", "cycle_years": 3},
        },
        "profiles": [profile],
        "equivalence_band": 0.0001 if weak else 0.005,
        "min_bootstrap_win_share": 0.99 if weak else 0.6,
        "bootstrap_paths": 50,
        "bootstrap_block_months": 12,
        "lineage": {
            "related_trial_count": 0,
            "related_trials": [],
            "first_test_date": "2026-09-26",
            "post_hoc_disclosure": "",
        },
        "notes": "",
    }
    path = tmp_path / ("household_weak.json" if weak else "household.json")
    path.write_text(_json.dumps(document), encoding="utf-8")
    return str(path)


def _install_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, losses: bool = False
) -> None:
    price_months = _month_ends(_date(1999, 12, 31), 97)
    modern_months = _month_ends(_date(1990, 1, 31), 96)
    frames = {
        "prices": _prices_frame(price_months),
        "research_monthly": _research_frame(modern_months, losses=losses),
    }
    snapshot = _SimpleNamespace(
        artifacts={
            _Dataset.PRICES: _SimpleNamespace(manifest_path="/m/prices.json"),
            _Dataset.RESEARCH_MONTHLY: _SimpleNamespace(manifest_path="/m/research.json"),
        }
    )
    monkeypatch.setattr(_catalog_mod, "resolve_snapshot", lambda *_a, **_k: snapshot)
    monkeypatch.setattr(
        _catalog_mod, "load_snapshot_visible", lambda _snap, dataset, _ts: frames[str(dataset)]
    )
    monkeypatch.setattr(_campaign_mod, "_resolve_git_commit", lambda: _GIT_COMMIT)


def test_command_writes_reproducible_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command exits zero and persists a payload with every required key."""
    _install_frames(tmp_path, monkeypatch)
    settings = DataSettings(data_root=str(tmp_path / "data"))
    config = _write_household_config(tmp_path, _write_pension_config(tmp_path), _write_record(tmp_path))
    code = _campaign_mod.run_isa_household_command(config_path=config, settings=settings, seed=7)
    assert code == 0
    artifacts = list((results_root(settings) / "cli_smoke").glob("isa_household_*.json"))
    assert len(artifacts) == 1
    payload = _json.loads(artifacts[0].read_text(encoding="utf-8"))
    for key in (
        "name", "incumbent_id", "pension_record_id", "trial_count", "lineage",
        "manifest_hashes", "config_sha256", "git_commit", "seed", "decisions", "cells", "notes",
    ):
        assert key in payload
    assert payload["decisions"][0]["budget_krw"] == 6_000_000
    assert payload["decisions"][0]["selected_arm_id"] in ("hold", "all")
    assert payload["manifest_hashes"] == {"prices": "prices", "research_monthly": "research"}
    assert (artifacts[0].with_suffix(".md")).exists()


def test_freeze_delegates_to_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--freeze forwards the decisive report to the record writer."""
    _install_frames(tmp_path, monkeypatch)
    captured: dict[str, object] = {}

    def fake_freeze(report: object, **kwargs: object) -> Path:
        captured["kwargs"] = kwargs
        captured["report"] = report
        return tmp_path / "frozen.json"

    monkeypatch.setattr(_decision_mod, "freeze_isa_household_decision", fake_freeze)
    settings = DataSettings(data_root=str(tmp_path / "data"))
    config = _write_household_config(tmp_path, _write_pension_config(tmp_path), _write_record(tmp_path))
    code = _campaign_mod.run_isa_household_command(config_path=config, settings=settings, seed=7, freeze=True)
    assert code == 0
    assert isinstance(captured.get("kwargs"), dict)


def test_freeze_skipped_without_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """--freeze with a weak verdict warns, writes no record, and still exits zero."""
    _install_frames(tmp_path, monkeypatch, losses=True)
    settings = DataSettings(data_root=str(tmp_path / "data"))
    config = _write_household_config(
        tmp_path, _write_pension_config(tmp_path), _write_record(tmp_path), weak=True
    )
    with caplog.at_level("WARNING"):
        code = _campaign_mod.run_isa_household_command(config_path=config, settings=settings, seed=7, freeze=True)
    assert code == 0
    assert "isa_household_freeze_skipped" in caplog.text


def test_command_rejects_century_span_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A century panel not spanning the registered window fails closed with exit 1."""
    price_months = _month_ends(_date(1999, 12, 31), 97)
    modern_months = _month_ends(_date(1990, 1, 31), 95)
    frames = {
        "prices": _prices_frame(price_months),
        "research_monthly": _research_frame(modern_months),
    }
    snapshot = _SimpleNamespace(
        artifacts={
            _Dataset.PRICES: _SimpleNamespace(manifest_path="/m/prices.json"),
            _Dataset.RESEARCH_MONTHLY: _SimpleNamespace(manifest_path="/m/research.json"),
        }
    )
    monkeypatch.setattr(_catalog_mod, "resolve_snapshot", lambda *_a, **_k: snapshot)
    monkeypatch.setattr(
        _catalog_mod, "load_snapshot_visible", lambda _snap, dataset, _ts: frames[str(dataset)]
    )
    monkeypatch.setattr(_campaign_mod, "_resolve_git_commit", lambda: _GIT_COMMIT)
    settings = DataSettings(data_root=str(tmp_path / "data"))
    config = _write_household_config(tmp_path, _write_pension_config(tmp_path), _write_record(tmp_path))
    code = _campaign_mod.run_isa_household_command(config_path=config, settings=settings, seed=7)
    assert code == 1


def test_freeze_writes_record_under_frozen_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--freeze persists the record under <data>/frozen/isa and never under records/."""
    _install_frames(tmp_path, monkeypatch)
    settings = DataSettings(data_root=str(tmp_path / "data"))
    config = _write_household_config(tmp_path, _write_pension_config(tmp_path), _write_record(tmp_path))
    code = _campaign_mod.run_isa_household_command(config_path=config, settings=settings, seed=7, freeze=True)
    assert code == 0
    frozen = list((tmp_path / "data" / "frozen" / "isa").glob("*.json"))
    assert len(frozen) == 1
    assert not Path("records").exists()
