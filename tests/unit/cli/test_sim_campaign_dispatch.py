"""Sim and campaign dispatch tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import cli
from src.cli import main
from src.data.settings import DataSettings
from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig, AllocationResult


class _FakeManifest:
    def __init__(self, row_count: int) -> None:
        self.row_count = row_count
        self.normalized_sha256 = "f" * 64


class _FakeArtifact:
    def __init__(self, row_count: int) -> None:
        self.manifest = _FakeManifest(row_count)


@pytest.mark.parametrize("scenario_id", ["CLI-G07-run-policy"])
def test_cli_g07_run_policy(scenario_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """test_cli_g07_run_policy"""
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

    import src.cli_commands.sim_run as sim_mod

    monkeypatch.setattr(sim_mod, "run_allocation_from_store", fake_run)
    monkeypatch.setattr(sim_mod, "require_feasibility", lambda **kwargs: None)
    monkeypatch.setattr(cli, "run_allocation_from_store", fake_run, raising=False)
    monkeypatch.setattr(cli, "require_feasibility", lambda **kwargs: None, raising=False)

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
        policy=PolicyId.WORLD_SPLIT,
        start=captured[0].start,
        end=captured[0].end,
        monthly_contribution_krw=1000000.0,
    )
    assert main(
        ["run", "policy", "--start", "2024-01-01", "--end", "2024-01-31", "--contribution-krw", "1"]
    ) == 2

    def failing_run(config: AllocationConfig, settings: object) -> AllocationResult:
        from src.sim.allocation import AllocationDataError

        raise AllocationDataError("missing sleeve price")

    monkeypatch.setattr(sim_mod, "run_allocation_from_store", failing_run)
    monkeypatch.setattr(cli, "run_allocation_from_store", failing_run, raising=False)
    assert main(argv) == 1


@pytest.mark.parametrize("scenario_id", ["CLI-W1-ablation-dispatch"])
def test_cli_w1_ablation_dispatch(
    scenario_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """test_cli_w1_ablation_dispatch"""
    wealth_by_policy = {PolicyId.VT: 100.0, PolicyId.VTI: 110.0, PolicyId.VT_TREAS: 120.0}
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

    import src.cli_commands.campaign as camp_mod

    monkeypatch.setattr(camp_mod, "assert_experiment_feasible", lambda spec, settings: None)
    monkeypatch.setattr(camp_mod, "run_allocation_from_store", fake_run)
    monkeypatch.setattr(camp_mod, "latest_artifact", lambda settings, dataset: _FakeArtifact(8))
    monkeypatch.setattr(cli, "assert_experiment_feasible", lambda spec, settings: None, raising=False)
    monkeypatch.setattr(cli, "run_allocation_from_store", fake_run, raising=False)
    monkeypatch.setattr(cli, "latest_artifact", lambda settings, dataset: _FakeArtifact(8), raising=False)

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
        PolicyId.VT,
        PolicyId.VTI,
        PolicyId.VT_TREAS,
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
    """test_cli_wf_dispatch"""

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

    import src.cli_commands.campaign as camp_mod

    monkeypatch.setattr(camp_mod, "assert_experiment_feasible", lambda spec, settings: None)
    monkeypatch.setattr(camp_mod, "run_allocation_from_store", fake_run)
    monkeypatch.setattr(camp_mod, "latest_artifact", lambda settings, dataset: _FakeArtifact(8))
    monkeypatch.setattr(cli, "assert_experiment_feasible", lambda spec, settings: None, raising=False)
    monkeypatch.setattr(cli, "run_allocation_from_store", fake_run, raising=False)
    monkeypatch.setattr(cli, "latest_artifact", lambda settings, dataset: _FakeArtifact(8), raising=False)

    assert main(["run", "walk-forward"]) == 2

    payload = {
        "name": "wf_s0_s1",
        "start": "2014-01-03",
        "end": "2024-09-30",
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
    monkeypatch.setattr(camp_mod, "DataSettings", lambda: DataSettings(data_root=data_root))
    monkeypatch.setattr(cli, "DataSettings", lambda: DataSettings(data_root=data_root), raising=False)

    exit_code = main(["run", "walk-forward", "--config", str(wf_path)])

    assert exit_code == 0
    reports = list((data_root / "results" / "experiments").glob("*.json"))
    assert len(reports) == 1
    written = json.loads(reports[0].read_text(encoding="utf-8"))
    assert "process_adopted_vs_baseline" in written

    assert main(["run", "walk-forward", "--config", "configs/experiments/m0_m1.json"]) == 1
