"""Unit tests for the ingest CLI dispatch."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from src.etf_manager import cli
from src.etf_manager.cli import main
from src.etf_manager.data.providers.base import ProviderError
from src.etf_manager.data.schema import Dataset
from src.etf_manager.data.settings import DataSettings
from src.etf_manager.etf.mapping import MappingConfig
from src.etf_manager.policy.targets import PolicyId, all_policy_tickers
from src.etf_manager.sim.allocation import (
    AllocationConfig,
    AllocationDataError,
    AllocationResult,
    AllocationSnapshot,
)
from src.etf_manager.sim.baseline import BaselineConfig, BaselineId, BaselineResult


@pytest.fixture(autouse=True)
def _provider_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve secrets from the process env so no sops or network call happens."""
    for name in ("TIINGO_API", "FRED_API", "ECOS_API"):
        monkeypatch.setenv(name, f"unit-test-{name}")


def test_cli_d07_ingest_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-D07-ingest-dispatch"""
    captured: dict[str, object] = {}

    def fake_prices(tickers: tuple[str, ...], start: date, end: date, **kwargs: object) -> None:
        captured["tickers"] = tickers
        captured["start"] = start
        captured["end"] = end

    monkeypatch.setattr(cli, "fetch_and_persist_prices", fake_prices)

    exit_code = main(["ingest", "prices", "--tickers", "VT", "--start", "2024-01-01", "--end", "2024-02-01"])
    assert exit_code == 0
    assert captured["tickers"] == ("VT",)
    assert captured["start"] == date(2024, 1, 1)
    assert captured["end"] == date(2024, 2, 1)

    fx_calls: list[int] = []

    def fake_fx(**kwargs: object) -> None:
        fx_calls.append(1)

    monkeypatch.setattr(cli, "fetch_and_persist_fx", fake_fx)

    invalid_exit = main(["ingest", "fx", "--start", "2024-01-01", "--end", "2024-02-01"])
    assert invalid_exit == 2
    assert fx_calls == []


class _FakeManifest:
    def __init__(self, row_count: int) -> None:
        self.row_count = row_count
        self.normalized_sha256 = "f" * 64


class _FakeArtifact:
    def __init__(self, row_count: int) -> None:
        self.manifest = _FakeManifest(row_count)


@pytest.mark.parametrize("scenario_id", ["CLI-E02-ingest-smoke-required-ok"])
def test_cli_e02_ingest_smoke_required_ok(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-E02-ingest-smoke-required-ok"""
    calls = {"fx": 0, "prices": 0, "cpi": 0}
    seen_datasets: list[Dataset] = []

    def fake_fx(**kwargs: object) -> _FakeArtifact:
        calls["fx"] += 1
        return _FakeArtifact(4)

    def fake_prices(tickers: tuple[str, ...], start: date, end: date, **kwargs: object) -> _FakeArtifact:
        calls["prices"] += 1
        return _FakeArtifact(4)

    def fake_cpi(start: date, end: date, **kwargs: object) -> _FakeArtifact:
        calls["cpi"] += 1
        raise ProviderError("ecos cpi rejected")

    def fake_latest(settings: object, dataset: Dataset) -> _FakeArtifact:
        seen_datasets.append(dataset)
        return _FakeArtifact(4)

    monkeypatch.setattr(cli, "fetch_and_persist_fx", fake_fx)
    monkeypatch.setattr(cli, "fetch_and_persist_prices", fake_prices)
    monkeypatch.setattr(cli, "fetch_and_persist_cpi", fake_cpi)
    monkeypatch.setattr(cli, "latest_artifact", fake_latest)

    exit_code = main(["ingest", "smoke"])

    assert exit_code == 0
    assert calls == {"fx": 1, "prices": 1, "cpi": 1}
    assert set(seen_datasets) == {Dataset.PRICES, Dataset.FX}


@pytest.mark.parametrize("scenario_id", ["CLI-E03-ingest-smoke-prices-fail"])
def test_cli_e03_ingest_smoke_prices_fail(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-E03-ingest-smoke-prices-fail"""
    fx_calls: list[int] = []
    prices_calls: list[int] = []

    def fake_fx(**kwargs: object) -> _FakeArtifact:
        fx_calls.append(1)
        return _FakeArtifact(4)

    def fake_prices(tickers: tuple[str, ...], start: date, end: date, **kwargs: object) -> _FakeArtifact:
        prices_calls.append(1)
        raise ProviderError("tiingo request failed")

    monkeypatch.setattr(cli, "fetch_and_persist_fx", fake_fx)
    monkeypatch.setattr(cli, "fetch_and_persist_prices", fake_prices)
    monkeypatch.setattr(
        cli,
        "latest_artifact",
        lambda settings, dataset: pytest.fail("catalog must not be consulted after required fetch failure"),
    )

    exit_code = main(["ingest", "smoke"])

    assert exit_code == 1
    assert len(fx_calls) == 1
    assert len(prices_calls) == 1


@pytest.mark.parametrize("scenario_id", ["CLI-E04-run-baseline"])
def test_cli_e04_run_baseline(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-E04-run-baseline"""
    captured: list[BaselineConfig] = []

    def fake_run(config: BaselineConfig, settings: object) -> BaselineResult:
        captured.append(config)
        return BaselineResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=1.0,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=0.8,
            xirr_real=-0.1,
        )

    monkeypatch.setattr(cli, "run_baseline_from_store", fake_run)

    argv = [
        "run",
        "baseline",
        "--id",
        "b0_global",
        "--ticker",
        "VT",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-31",
        "--contribution-krw",
        "1000000",
    ]
    exit_code = main(argv)

    assert exit_code == 0
    assert len(captured) == 1
    assert captured[0] == BaselineConfig(
        baseline=BaselineId.B0_GLOBAL,
        ticker="VT",
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        monthly_contribution_krw=1000000.0,
    )

    missing_ticker_argv = ["run", "baseline", "--id", "b0_global", "--start", "2024-01-01",
                           "--end", "2024-01-31", "--contribution-krw", "1000000"]
    assert main(missing_ticker_argv) == 2


@pytest.mark.parametrize("scenario_id", ["CLI-F03-ingest-history"])
def test_cli_f03_ingest_history(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-F03-ingest-history"""
    calls = {"fx": 0, "prices": 0, "cpi": 0, "factors": 0, "macro": 0}
    seen_tickers: tuple[str, ...] = ()
    seen_series_ids: list[str] = []
    seen_datasets: list[Dataset] = []

    def fake_fx(**kwargs: object) -> _FakeArtifact:
        calls["fx"] += 1
        return _FakeArtifact(8)

    def fake_prices(tickers: tuple[str, ...], start: date, end: date, **kwargs: object) -> _FakeArtifact:
        nonlocal seen_tickers
        calls["prices"] += 1
        seen_tickers = tickers
        return _FakeArtifact(8)

    def fake_cpi(start: date, end: date, **kwargs: object) -> _FakeArtifact:
        calls["cpi"] += 1
        return _FakeArtifact(8)

    def fake_factors(start: date, end: date, **kwargs: object) -> _FakeArtifact:
        calls["factors"] += 1
        return _FakeArtifact(8)

    def fake_macro(series_id: str, start: date, end: date, **kwargs: object) -> _FakeArtifact:
        calls["macro"] += 1
        seen_series_ids.append(series_id)
        return _FakeArtifact(8)

    def fake_latest(settings: object, dataset: Dataset) -> _FakeArtifact:
        seen_datasets.append(dataset)
        return _FakeArtifact(8)

    monkeypatch.setattr(cli, "fetch_and_persist_fx", fake_fx)
    monkeypatch.setattr(cli, "fetch_and_persist_prices", fake_prices)
    monkeypatch.setattr(cli, "fetch_and_persist_cpi", fake_cpi)
    monkeypatch.setattr(cli, "fetch_and_persist_factors", fake_factors)
    monkeypatch.setattr(cli, "fetch_and_persist_macro", fake_macro)
    monkeypatch.setattr(cli, "latest_artifact", fake_latest)

    exit_code = main(["ingest", "history", "--start", "2020-01-01", "--end", "2020-12-31"])

    assert exit_code == 0
    assert calls == {"fx": 1, "prices": 1, "cpi": 1, "factors": 1, "macro": 1}
    assert seen_tickers == all_policy_tickers()
    assert seen_series_ids == ["VIXCLS"]
    assert set(seen_datasets) == {Dataset.PRICES, Dataset.FX, Dataset.CPI, Dataset.FACTORS, Dataset.MACRO}

    assert main(["ingest", "history"]) == 2


def test_cli_f03_ingest_history_cpi_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-F03-ingest-history"""

    def fake_fx(**kwargs: object) -> _FakeArtifact:
        return _FakeArtifact(8)

    def fake_prices(tickers: tuple[str, ...], start: date, end: date, **kwargs: object) -> _FakeArtifact:
        return _FakeArtifact(8)

    def fake_cpi(start: date, end: date, **kwargs: object) -> _FakeArtifact:
        raise ProviderError("ecos cpi rejected")

    monkeypatch.setattr(cli, "fetch_and_persist_fx", fake_fx)
    monkeypatch.setattr(cli, "fetch_and_persist_prices", fake_prices)
    monkeypatch.setattr(cli, "fetch_and_persist_cpi", fake_cpi)

    exit_code = main(["ingest", "history", "--start", "2020-01-01", "--end", "2020-12-31"])

    assert exit_code == 1


@pytest.mark.parametrize("scenario_id", ["CLI-F04-baseline-real-log"])
def test_cli_f04_baseline_real_log(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-F04-baseline-real-log"""

    def fake_run(config: BaselineConfig, settings: object) -> BaselineResult:
        return BaselineResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=1.0,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=0.8,
            xirr_real=-0.1,
        )

    monkeypatch.setattr(cli, "run_baseline_from_store", fake_run)

    argv = [
        "run",
        "baseline",
        "--id",
        "b0_global",
        "--ticker",
        "VT",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-31",
        "--contribution-krw",
        "1000000",
    ]
    assert main(argv) == 0


@pytest.mark.parametrize("scenario_id", ["CLI-G07-run-policy"])
def test_cli_g07_run_policy(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-G07-run-policy"""
    captured: list[AllocationConfig] = []

    def fake_run(config: AllocationConfig, settings: object) -> AllocationResult:
        captured.append(config)
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=2.5,
            xirr=0.12,
            max_drawdown=-0.05,
            terminal_wealth_real_krw=2.0,
            xirr_real=0.09,
        )

    monkeypatch.setattr(cli, "run_allocation_from_store", fake_run)

    argv = [
        "run",
        "policy",
        "--id",
        "s2_regional",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-31",
        "--contribution-krw",
        "1000000",
    ]
    exit_code = main(argv)

    assert exit_code == 0
    assert len(captured) == 1
    assert captured[0] == AllocationConfig(
        policy=PolicyId.S2_REGIONAL,
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        monthly_contribution_krw=1000000.0,
    )
    assert main(
        ["run", "policy", "--start", "2024-01-01", "--end", "2024-01-31", "--contribution-krw", "1"]
    ) == 2

    def failing_run(config: AllocationConfig, settings: object) -> AllocationResult:
        raise AllocationDataError("missing sleeve price")

    monkeypatch.setattr(cli, "run_allocation_from_store", failing_run)
    assert main(argv) == 1


@pytest.mark.parametrize("scenario_id", ["CLI-H06-ingest-factors-and-tilt-flags"])
def test_cli_h06_ingest_factors_and_tilt_flags(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-H06-ingest-factors-and-tilt-flags"""
    factor_calls: list[tuple[date, date]] = []

    def fake_factors(start: date, end: date, **kwargs: object) -> _FakeArtifact:
        factor_calls.append((start, end))
        return _FakeArtifact(12)

    monkeypatch.setattr(cli, "fetch_and_persist_factors", fake_factors)

    exit_code = main(["ingest", "factors", "--start", "2010-01-01", "--end", "2010-12-31"])

    assert exit_code == 0
    assert len(factor_calls) == 1
    assert factor_calls[0] == (date(2010, 1, 1), date(2010, 12, 31))

    assert main(["ingest", "factors"]) == 2
    assert len(factor_calls) == 1

    captured: list[AllocationConfig] = []

    def fake_run(config: AllocationConfig, settings: object) -> AllocationResult:
        captured.append(config)
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=2.5,
            xirr=0.12,
            max_drawdown=-0.05,
            terminal_wealth_real_krw=2.0,
            xirr_real=0.09,
        )

    monkeypatch.setattr(cli, "run_allocation_from_store", fake_run)

    base_argv = [
        "run",
        "policy",
        "--id",
        "s2_regional",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-31",
        "--contribution-krw",
        "1",
    ]
    assert main([*base_argv, "--tilt-factor", "hml"]) == 2
    assert main([*base_argv, "--tilt-intensity", "0.1"]) == 2
    assert captured == []

    exit_code = main([*base_argv, "--tilt-factor", "hml", "--tilt-intensity", "0.1"])

    assert exit_code == 0
    assert len(captured) == 1
    tilt = captured[0].tilt
    assert tilt is not None
    assert tilt.factor == "hml"
    assert tilt.intensity == pytest.approx(0.1)

    exit_code = main(base_argv)
    assert exit_code == 0
    assert captured[1].tilt is None


@pytest.mark.parametrize("scenario_id", ["CLI-I05-rebalance-band-flag"])
def test_cli_i05_rebalance_band_flag(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-I05-rebalance-band-flag"""
    captured: list[AllocationConfig] = []

    def fake_run(config: AllocationConfig, settings: object) -> AllocationResult:
        captured.append(config)
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=2.5,
            xirr=0.12,
            max_drawdown=-0.05,
            terminal_wealth_real_krw=2.0,
            xirr_real=0.09,
        )

    monkeypatch.setattr(cli, "run_allocation_from_store", fake_run)

    base_argv = [
        "run",
        "policy",
        "--id",
        "s2_regional",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-31",
        "--contribution-krw",
        "1000000",
    ]
    assert main([*base_argv, "--rebalance-band", "0.05"]) == 0
    assert len(captured) == 1
    assert captured[0].rebalance_band == pytest.approx(0.05)

    assert main(base_argv) == 0
    assert len(captured) == 2
    assert captured[1].rebalance_band is None

    for invalid in ("-0.1", "1.0"):
        assert main([*base_argv, "--rebalance-band", invalid]) == 1
    assert len(captured) == 2


@pytest.mark.parametrize("scenario_id", ["CLI-J05-overlay-flags"])
def test_cli_j05_overlay_flags(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-J05-overlay-flags"""
    captured: list[AllocationConfig] = []

    def fake_run(config: AllocationConfig, settings: object) -> AllocationResult:
        captured.append(config)
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=2.5,
            xirr=0.12,
            max_drawdown=-0.05,
            terminal_wealth_real_krw=2.0,
            xirr_real=0.09,
        )

    monkeypatch.setattr(cli, "run_allocation_from_store", fake_run)

    base_argv = [
        "run",
        "policy",
        "--id",
        "s2_regional",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-31",
        "--contribution-krw",
        "1000000",
    ]

    assert main([*base_argv, "--overlay-max-shift", "0.1"]) == 0
    assert len(captured) == 1
    overlay = captured[0].overlay
    assert overlay is not None
    assert overlay.max_shift == pytest.approx(0.1)
    assert overlay.vix_threshold is None

    assert main([*base_argv, "--vix-threshold", "20"]) == 2
    assert len(captured) == 1

    assert main([*base_argv, "--overlay-max-shift", "0.1", "--vix-threshold", "20"]) == 0
    assert len(captured) == 2
    overlay_gated = captured[1].overlay
    assert overlay_gated is not None
    assert overlay_gated.vix_threshold == pytest.approx(20.0)


@pytest.mark.parametrize("scenario_id", ["CLI-K05-fx-flags"])
def test_cli_k05_fx_flags(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-K05-fx-flags"""
    captured: list[AllocationConfig] = []

    def fake_run(config: AllocationConfig, settings: object) -> AllocationResult:
        captured.append(config)
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=2.5,
            xirr=0.12,
            max_drawdown=-0.05,
            terminal_wealth_real_krw=2.0,
            xirr_real=0.09,
        )

    monkeypatch.setattr(cli, "run_allocation_from_store", fake_run)

    base_argv = [
        "run",
        "policy",
        "--id",
        "s2_regional",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-31",
        "--contribution-krw",
        "1000000",
    ]

    assert main([*base_argv, "--fx-max-defer", "0.2"]) == 0
    assert len(captured) == 1
    currency = captured[0].currency
    assert currency is not None
    assert currency.max_defer == pytest.approx(0.2)
    assert currency.expensive_percentile == pytest.approx(0.80)

    assert main([*base_argv, "--fx-expensive-percentile", "0.9"]) == 2
    assert len(captured) == 1

    assert main([*base_argv, "--fx-max-defer", "0.2", "--fx-expensive-percentile", "0.9"]) == 0
    assert len(captured) == 2
    gated = captured[1].currency
    assert gated is not None
    assert gated.expensive_percentile == pytest.approx(0.9)

    assert main(base_argv) == 0
    assert len(captured) == 3
    assert captured[2].currency is None


@pytest.mark.parametrize("scenario_id", ["CLI-M05-map-etf-flags"])
def test_cli_m05_map_etf_flags(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-M05-map-etf-flags"""
    captured: list[AllocationConfig] = []

    def fake_run(config: AllocationConfig, settings: object) -> AllocationResult:
        captured.append(config)
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=2.5,
            xirr=0.12,
            max_drawdown=-0.05,
            terminal_wealth_real_krw=2.0,
            xirr_real=0.09,
        )

    monkeypatch.setattr(cli, "run_allocation_from_store", fake_run)

    base_argv = [
        "run",
        "policy",
        "--id",
        "s2_regional",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-31",
        "--contribution-krw",
        "1000000",
    ]

    assert main([*base_argv, "--map-etf"]) == 0
    mapping = captured[-1].mapping
    assert isinstance(mapping, MappingConfig)
    assert mapping.min_improvement == pytest.approx(0.02)

    assert main([*base_argv, "--map-min-improvement", "0.05"]) == 2

    assert main([*base_argv, "--map-etf", "--map-min-improvement", "0.05"]) == 0
    tuned = captured[-1].mapping
    assert tuned is not None
    assert tuned.min_improvement == pytest.approx(0.05)

    assert main(base_argv) == 0
    assert captured[-1].mapping is None


@pytest.mark.parametrize("scenario_id", ["VAL-V05-cli-validate"])
def test_val_v05_cli_validate(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """VAL-V05-cli-validate"""
    policy_configs: list[AllocationConfig] = []
    baseline_configs: list[BaselineConfig] = []

    def fake_policy_run(config: AllocationConfig, settings: object) -> AllocationResult:
        policy_configs.append(config)
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=2.5,
            xirr=0.12,
            max_drawdown=-0.05,
            terminal_wealth_real_krw=2.0,
            xirr_real=0.09,
        )

    def fake_baseline_run(config: BaselineConfig, settings: object) -> BaselineResult:
        baseline_configs.append(config)
        return BaselineResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=1.25,
            xirr=0.06,
            max_drawdown=-0.03,
            terminal_wealth_real_krw=1.0,
            xirr_real=0.04,
        )

    monkeypatch.setattr(cli, "run_allocation_from_store", fake_policy_run)
    monkeypatch.setattr(cli, "run_baseline_from_store", fake_baseline_run)
    monkeypatch.setattr(cli, "latest_artifact", lambda settings, dataset: _FakeArtifact(8))

    base_argv = [
        "run",
        "validate",
        "--id",
        "s0_global",
        "--start",
        "2024-01-01",
        "--end",
        "2024-12-31",
        "--contribution-krw",
        "1000000",
        "--horizon-months",
        "12",
    ]

    assert main(base_argv) == 0
    assert len(policy_configs) == 1
    assert len(baseline_configs) == 1
    assert (policy_configs[0].start, policy_configs[0].end) == (date(2024, 1, 1), date(2024, 12, 31))
    assert (baseline_configs[0].start, baseline_configs[0].end) == (date(2024, 1, 1), date(2024, 12, 31))
    assert baseline_configs[0].baseline == BaselineId.B0_GLOBAL

    assert main([*base_argv, "--bootstrap-paths", "2"]) == 2
    assert main([*base_argv, "--bootstrap-paths", "2", "--seed", "1"]) == 0


@pytest.mark.parametrize("scenario_id", ["CLI-X04-paper-flags"])
def test_cli_x04_paper_flags(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-X04-paper-flags"""
    captured: list[AllocationConfig] = []

    def fake_run(config: AllocationConfig, settings: object) -> AllocationResult:
        captured.append(config)
        empty = AllocationSnapshot(
            session=date(2024, 1, 5),
            cash_krw=1_000_000.0,
            cash_usd=0.0,
            shares={},
            mark_krw=1_000_000.0,
            contribution_krw=1_000_000.0,
            fees_krw=0.0,
        )
        filled = replace(empty, session=date(2024, 1, 31), shares={"VT": 1})
        return AllocationResult(
            config=config,
            snapshots=(empty, filled),
            terminal_wealth_krw=1_300_000.0,
            xirr=0.12,
            max_drawdown=-0.05,
            terminal_wealth_real_krw=1_200_000.0,
            xirr_real=0.09,
        )

    monkeypatch.setattr(cli, "run_allocation_from_store", fake_run)

    argv = [
        "run",
        "paper",
        "--id",
        "s0_global",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-31",
        "--contribution-krw",
        "1000000",
    ]
    assert main(argv) == 0
    assert len(captured) == 1
    assert captured[0] == AllocationConfig(
        policy=PolicyId.S0_GLOBAL,
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        monthly_contribution_krw=1000000.0,
    )

    missing_id_argv = ["run", "paper", "--start", "2024-01-01", "--end", "2024-01-31",
                       "--contribution-krw", "1000000"]
    assert main(missing_id_argv) == 2
    assert len(captured) == 1


@pytest.mark.parametrize("scenario_id", ["CLI-W1-ablation-dispatch"])
def test_cli_w1_ablation_dispatch(
    scenario_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI-W1-ablation-dispatch"""
    wealth_by_policy = {PolicyId.S0_GLOBAL: 100.0, PolicyId.S1_US: 110.0, PolicyId.S4_DEFENSIVE: 120.0}
    captured: list[AllocationConfig] = []

    def fake_run(config: AllocationConfig, settings: object) -> AllocationResult:
        captured.append(config)
        wealth = float(wealth_by_policy[config.policy])
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.0,
        )

    monkeypatch.setattr(cli, "run_allocation_from_store", fake_run)
    monkeypatch.setattr(cli, "latest_artifact", lambda settings, dataset: _FakeArtifact(8))

    payload = {
        "name": "m0_m1_strategic",
        "start": "2012-01-03",
        "end": "2024-12-31",
        "contribution_krw": 1_000_000,
        "delta0": 0.02,
        "horizon_months": 0,
        "baseline": {"id": "m0_global", "policy": "s0_global", "modules": 0},
        "candidates": [
            {"id": "s1_us", "policy": "s1_us", "modules": 1},
            {"id": "s4_defensive", "policy": "s4_defensive", "modules": 1},
        ],
    }
    config_path = tmp_path / "m0_m1.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(["run", "ablation", "--config", str(config_path)])

    assert exit_code == 0
    assert len(captured) == 1 + len(payload["candidates"])
    assert [config.policy for config in captured] == [
        PolicyId.S0_GLOBAL,
        PolicyId.S1_US,
        PolicyId.S4_DEFENSIVE,
    ]
    assert len({config.monthly_contribution_krw for config in captured}) == 1
    assert captured[0].monthly_contribution_krw == pytest.approx(1_000_000.0)

    assert main(["run", "ablation"]) == 2


@pytest.mark.parametrize("scenario_id", ["CLI-WF-dispatch"])
def test_cli_wf_dispatch(
    scenario_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI-WF-dispatch"""

    def fake_run(config: AllocationConfig, settings: object) -> AllocationResult:
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=100.0,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=100.0,
            xirr_real=0.0,
        )

    monkeypatch.setattr(cli, "run_allocation_from_store", fake_run)
    monkeypatch.setattr(cli, "latest_artifact", lambda settings, dataset: _FakeArtifact(8))

    assert main(["run", "walk-forward"]) == 2

    payload = {
        "name": "wf_s0_s1",
        "start": "2012-04-01",
        "end": "2024-11-30",
        "contribution_krw": 1_000_000,
        "delta0": 0.02,
        "horizon_months": 0,
        "train_months": 60,
        "test_months": 36,
        "baseline": {"id": "s0_global", "policy": "s0_global", "modules": 0},
        "candidates": [{"id": "s1_us", "policy": "s1_us", "modules": 1}],
    }
    wf_path = tmp_path / "wf_s0_s1.json"
    wf_path.write_text(json.dumps(payload), encoding="utf-8")

    data_root = tmp_path / "data"
    monkeypatch.setattr(cli, "DataSettings", lambda: DataSettings(data_root=data_root))

    exit_code = main(["run", "walk-forward", "--config", str(wf_path)])

    assert exit_code == 0
    reports = list((data_root / "experiments").glob("*.json"))
    assert len(reports) == 1
    written = json.loads(reports[0].read_text(encoding="utf-8"))
    assert "process_adopted_vs_baseline" in written

    assert main(["run", "walk-forward", "--config", "configs/experiments/m0_m1.json"]) == 1


@pytest.mark.parametrize("scenario_id", ["CLI-horizon-default-36"])
def test_cli_horizon_default_36(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI-horizon-default-36"""
    captured: list[int] = []

    def fake_validate(**kwargs: object) -> int:
        captured.append(int(kwargs["horizon_months"]))
        return 0

    monkeypatch.setattr(cli, "run_validate_command", fake_validate)

    argv = [
        "run",
        "validate",
        "--id",
        "s0_global",
        "--start",
        "2024-01-01",
        "--end",
        "2024-12-31",
        "--contribution-krw",
        "1000000",
    ]
    assert main(argv) == 0
    assert captured[-1] == 36

    assert main([*argv, "--horizon-months", "12"]) == 0
    assert captured[-1] == 12
