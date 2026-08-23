"""Unit tests for the ingest CLI dispatch."""

from __future__ import annotations

from datetime import date

import pytest

from src.etf_manager import cli
from src.etf_manager.cli import main
from src.etf_manager.data.providers.base import ProviderError
from src.etf_manager.data.schema import Dataset
from src.etf_manager.policy.targets import PolicyId
from src.etf_manager.sim.allocation import AllocationConfig, AllocationDataError, AllocationResult
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
    calls = {"fx": 0, "prices": 0, "cpi": 0}
    seen_tickers: tuple[str, ...] = ()

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

    def fake_latest(settings: object, dataset: Dataset) -> _FakeArtifact:
        return _FakeArtifact(8)

    monkeypatch.setattr(cli, "fetch_and_persist_fx", fake_fx)
    monkeypatch.setattr(cli, "fetch_and_persist_prices", fake_prices)
    monkeypatch.setattr(cli, "fetch_and_persist_cpi", fake_cpi)
    monkeypatch.setattr(cli, "latest_artifact", fake_latest)

    exit_code = main(["ingest", "history", "--start", "2020-01-01", "--end", "2020-12-31"])

    assert exit_code == 0
    assert calls == {"fx": 1, "prices": 1, "cpi": 1}
    assert seen_tickers == ("VT", "VTI")

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
