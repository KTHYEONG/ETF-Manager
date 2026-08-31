from __future__ import annotations

def test_prospective_bundle_v1_arms() -> None:
    from pathlib import Path

    from src.validation.prospective_registry import ProspectiveArmRole, load_prospective_bundle

    bundle = load_prospective_bundle(Path("configs/prospective/prospective_2026_v1.json"))
    assert bundle.bundle_id == "PROSPECTIVE_2026_V1"
    roles = {arm.role for arm in bundle.arms}
    assert ProspectiveArmRole.IMMUTABLE_BENCHMARK in roles
    assert ProspectiveArmRole.PROVISIONAL_INCUMBENT in roles
    assert ProspectiveArmRole.DEPLOYMENT_TIMING in roles
    benchmark = next(a for a in bundle.arms if a.role is ProspectiveArmRole.IMMUTABLE_BENCHMARK)
    assert benchmark.targets == {"QQQ": 1.0}
    incumbent = next(a for a in bundle.arms if a.role is ProspectiveArmRole.PROVISIONAL_INCUMBENT)
    assert incumbent.targets == {"QQQ": 0.9, "SOXX": 0.1}
    assert incumbent.adaptive_contribution is None
    timing = next(a for a in bundle.arms if a.role is ProspectiveArmRole.DEPLOYMENT_TIMING)
    assert timing.kafi_deployment is not None
    assert timing.adaptive_contribution is None
    assert bundle.seen_history_cutoff.isoformat() == "2026-08-28"

def test_freeze_prospective_bundle_writes_registry(tmp_path) -> None:
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    from src.data.settings import DataSettings
    from src.validation.prospective_registry import freeze_prospective_bundle, load_prospective_bundle

    source = Path("configs/prospective/prospective_2026_v1.json")
    out = tmp_path / "registry"
    record = freeze_prospective_bundle(
        bundle_path=source,
        output_dir=out,
        frozen_at=datetime(2026, 8, 31, 12, 0, tzinfo=UTC),
        git_commit="abc123",
        settings=DataSettings(data_root=str(tmp_path / "data")),
    )
    assert record.bundle_id == "PROSPECTIVE_2026_V1"
    assert len(record.arm_hashes) == 3
    written = out / "prospective_2026_v1_frozen.json"
    assert written.is_file()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["bundle_id"] == "PROSPECTIVE_2026_V1"
    reloaded = load_prospective_bundle(written)
    assert reloaded.frozen_at == record.frozen_at

def test_assert_strategy_identity_unchanged_rejects_edit() -> None:
    from datetime import date

    import pytest

    from src.policy.targets import PolicyId
    from src.sim.allocation import AllocationConfig
    from src.validation.prospective_registry import (
        FrozenStrategyArm,
        assert_strategy_identity_unchanged,
        strategy_arm_identity_hash,
    )

    ok = AllocationConfig(
        policy=PolicyId.QQQ,
        start=date(2026, 9, 1),
        end=date(2026, 9, 30),
        monthly_contribution_krw=1_000_000.0,
        targets_override={"QQQ": 0.9, "SOXX": 0.1},
    )
    frozen = FrozenStrategyArm(
        arm_id="incumbent",
        policy=PolicyId.QQQ,
        targets={"QQQ": 0.9, "SOXX": 0.1},
        identity_hash=strategy_arm_identity_hash(ok),
    )
    assert_strategy_identity_unchanged(frozen, ok)
    bad = AllocationConfig(
        policy=PolicyId.QQQ,
        start=date(2026, 9, 1),
        end=date(2026, 9, 30),
        monthly_contribution_krw=1_000_000.0,
        targets_override={"QQQ": 0.85, "SOXX": 0.15},
    )
    with pytest.raises(ValueError, match=r"identity|targets"):
        assert_strategy_identity_unchanged(frozen, bad)

def test_run_prospective_monitor_rejects_seen_history(tmp_path) -> None:
    from datetime import date
    from pathlib import Path

    import pytest

    from src.data.settings import DataSettings
    from src.validation.prospective_registry import load_prospective_bundle, run_prospective_monitor

    bundle = load_prospective_bundle(Path("configs/prospective/prospective_2026_v1.json"))
    with pytest.raises(ValueError, match="seen_history"):
        run_prospective_monitor(
            bundle=bundle,
            as_of=date(2026, 8, 28),
            runner=lambda _cfg: (_ for _ in ()).throw(AssertionError("should not run")),
            settings=DataSettings(data_root=str(tmp_path)),
        )

def test_run_prospective_monitor_records_observations(tmp_path) -> None:
    from datetime import date
    from pathlib import Path

    from src.data.settings import DataSettings
    from src.policy.targets import PolicyId
    from src.sim.allocation import AllocationConfig, AllocationResult
    from src.validation.prospective_registry import load_prospective_bundle, run_prospective_monitor

    bundle = load_prospective_bundle(Path("configs/prospective/prospective_2026_v1.json"))
    settings = DataSettings(data_root=str(tmp_path))

    def _runner(config: AllocationConfig) -> AllocationResult:
        wealth = 110.0 if config.targets_override and config.targets_override.get("SOXX") else 100.0
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=-0.1,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.06,
            total_contribution_real_krw=90.0,
        )

    report = run_prospective_monitor(
        bundle=bundle,
        as_of=date(2026, 9, 30),
        runner=_runner,
        settings=settings,
        registry_dir=tmp_path / "prospective_registry",
    )
    assert report.as_of == date(2026, 9, 30)
    assert len(report.observations) == len(bundle.arms)
    assert all(obs.terminal_wealth_real_krw > 0.0 for obs in report.observations)
    assert report.registry_path.is_file()
    assert PolicyId.QQQ in {obs.policy for obs in report.observations}

def test_prospective_bundle_forbids_adaptive_contribution(tmp_path) -> None:
    import json
    from pathlib import Path

    import pytest

    from src.validation.prospective_registry import load_prospective_bundle

    payload = json.loads(Path("configs/prospective/prospective_2026_v1.json").read_text(encoding="utf-8"))
    payload["arms"][1]["adaptive_contribution"] = {}
    bad_path = tmp_path / "bad_bundle.json"
    bad_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="adaptive_contribution"):
        load_prospective_bundle(bad_path)

def test_prospective_monitor_uses_cumulative_inception_window(tmp_path) -> None:  # noqa: C408,F401
    from datetime import date
    from pathlib import Path  # noqa: F401

    from src.data.settings import DataSettings
    from src.sim.allocation import AllocationConfig, AllocationResult, Snapshot
    from src.validation.prospective_registry import ProspectiveArmRole, ProspectiveBundleSpec, FrozenStrategyArm, run_prospective_monitor
    from src.validation.research_posture import ObjectiveFamily, SEEN_HISTORY_CUTOFF

    calls: list[tuple[date, date]] = []

    def _runner(cfg: AllocationConfig) -> AllocationResult:
        calls.append((cfg.start, cfg.end))
        snaps = (
            Snapshot(session=cfg.start, mark_krw=1_000_000.0, contribution_krw=1_000_000.0, reserve_krw=0.0),
            Snapshot(session=cfg.end, mark_krw=2_000_000.0, contribution_krw=1_000_000.0, reserve_krw=250_000.0),
        )
        return AllocationResult(
            config=cfg,
            snapshots=snaps,
            terminal_wealth_krw=2_000_000.0,
            xirr=0.1,
            max_drawdown=0.0,
            terminal_wealth_real_krw=2_000_000.0,
            xirr_real=0.1,
            total_contribution_real_krw=2_000_000.0,
        )

    bundle = ProspectiveBundleSpec(
        bundle_id="PROSPECTIVE_2026_V1",
        seen_history_cutoff=SEEN_HISTORY_CUTOFF,
        prospective_start=date(2026, 9, 1),
        contribution_krw=1_000_000.0,
        arms=(
            FrozenStrategyArm(
                arm_id="incumbent_kafi_timing",
                policy="qqq",
                targets={"QQQ": 0.9, "SOXX": 0.1},
                role=ProspectiveArmRole.DEPLOYMENT_TIMING,
                kafi_deployment={"equity_ticker": "QQQ", "bond_ticker": "IEF", "credit_series_id": "BAA10Y", "min_multiplier": 0.7, "max_multiplier": 1.3, "rank_window": 252},
                objective_family=ObjectiveFamily.DEPLOYMENT_TIMING,
            ),
        ),
    )
    as_of = date(2026, 10, 31)
    report = run_prospective_monitor(
        bundle=bundle,
        as_of=as_of,
        runner=_runner,
        settings=DataSettings(data_root=str(tmp_path / "data")),
        registry_dir=tmp_path,
        runtime_git_commit="abc123def456",
    )
    assert calls, "runner must be invoked"
    assert calls[0][0] == date(2026, 9, 1)
    assert calls[0][1] <= as_of
    kafi_obs = next(o for o in report.observations if o.arm_id == "incumbent_kafi_timing")
    assert kafi_obs.reserve_krw == 250_000.0
    assert kafi_obs.cumulative_terminal_real_krw == 2_000_000.0

def test_strategy_identity_hash_distinguishes_kafi_params() -> None:  # noqa: C408
    from datetime import date

    from src.policy.kafi_deployment import KafiDeploymentConfig
    from src.policy.targets import PolicyId
    from src.sim.allocation import AllocationConfig
    from src.validation.prospective_registry import strategy_arm_identity_hash

    base = dict(  # noqa: C408
        policy=PolicyId.QQQ,
        start=date(2026, 9, 1),
        end=date(2026, 9, 30),
        monthly_contribution_krw=1_000_000.0,
        fill_delay_sessions=1,
        commission_bps=0.0,
        fx_spread_bps=0.0,
        targets_override={"QQQ": 0.9, "SOXX": 0.1},
    )
    cfg_a = AllocationConfig(**base, kafi_deployment=KafiDeploymentConfig(min_multiplier=0.7, max_multiplier=1.3, rank_window=252))
    cfg_b = AllocationConfig(**base, kafi_deployment=KafiDeploymentConfig(min_multiplier=0.1, max_multiplier=1.5, rank_window=63))
    hash_a = strategy_arm_identity_hash(cfg_a)
    hash_b = strategy_arm_identity_hash(cfg_b)
    assert hash_a != hash_b

def test_assert_runtime_engine_commit_fail_closed() -> None:
    import pytest

    from src.validation.prospective_registry import assert_runtime_engine_commit

    with pytest.raises(ValueError, match="runtime_engine_commit"):
        assert_runtime_engine_commit(
            frozen_git_commit="aaa111",
            runtime_git_commit="bbb222",
            behavior_preserving_migration=False,
        )
    assert_runtime_engine_commit(
        frozen_git_commit="aaa111",
        runtime_git_commit="bbb222",
        behavior_preserving_migration=True,
        approved_runtime_commits=("bbb222",),
    )


def test_run_prospective_monitor_requires_runtime_commit_for_frozen_bundle(tmp_path) -> None:
    from datetime import date

    import pytest

    from src.data.settings import DataSettings
    from src.sim.allocation import AllocationConfig, AllocationResult
    from src.validation.prospective_registry import (
        FrozenStrategyArm,
        ProspectiveArmRole,
        ProspectiveBundleSpec,
        run_prospective_monitor,
    )
    from src.validation.research_posture import ObjectiveFamily, SEEN_HISTORY_CUTOFF

    bundle = ProspectiveBundleSpec(
        bundle_id="PROSPECTIVE_2026_V1",
        seen_history_cutoff=SEEN_HISTORY_CUTOFF,
        prospective_start=date(2026, 9, 1),
        git_commit="frozen123",
        arms=(
            FrozenStrategyArm(
                arm_id="benchmark_qqq100",
                policy="qqq",
                targets={"QQQ": 1.0},
                role=ProspectiveArmRole.IMMUTABLE_BENCHMARK,
                objective_family=ObjectiveFamily.CAPITAL_ALLOCATION,
            ),
        ),
    )

    def _runner(_cfg: AllocationConfig) -> AllocationResult:
        return AllocationResult(
            config=_cfg,
            snapshots=(),
            terminal_wealth_krw=100.0,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=100.0,
            xirr_real=0.0,
        )

    with pytest.raises(ValueError, match="runtime_git_commit required"):
        run_prospective_monitor(
            bundle=bundle,
            as_of=date(2026, 9, 30),
            runner=_runner,
            settings=DataSettings(data_root=str(tmp_path / "data")),
            registry_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="runtime_engine_commit"):
        run_prospective_monitor(
            bundle=bundle,
            as_of=date(2026, 9, 30),
            runner=_runner,
            settings=DataSettings(data_root=str(tmp_path / "data")),
            registry_dir=tmp_path,
            runtime_git_commit="different456",
        )

def test_load_prospective_bundle_self_contained_targets(tmp_path) -> None:  # noqa: F401
    import json
    from pathlib import Path  # noqa: F401

    from src.validation.prospective_registry import load_prospective_bundle

    payload = {
        "bundle_id": "PROSPECTIVE_2026_V1",
        "seen_history_cutoff": "2026-08-28",
        "prospective_start": "2026-09-01",
        "contribution_krw": 1000000,
        "arms": [
            {
                "arm_id": "benchmark_qqq100",
                "policy": "qqq",
                "targets": {"QQQ": 1.0},
                "role": "immutable_benchmark",
                "objective_family": "capital_allocation",
            },
            {
                "arm_id": "incumbent_qqq90_soxx10",
                "policy": "qqq",
                "targets": {"QQQ": 0.9, "SOXX": 0.1},
                "role": "provisional_incumbent",
                "objective_family": "capital_allocation",
            },
            {
                "arm_id": "incumbent_kafi_timing",
                "policy": "qqq",
                "targets": {"QQQ": 0.9, "SOXX": 0.1},
                "role": "deployment_timing",
                "kafi_deployment": {
                    "equity_ticker": "QQQ",
                    "bond_ticker": "IEF",
                    "credit_series_id": "BAA10Y",
                    "min_multiplier": 0.7,
                    "max_multiplier": 1.3,
                    "rank_window": 252,
                },
                "objective_family": "deployment_timing",
            },
        ],
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    bundle = load_prospective_bundle(path)
    incumbent = next(a for a in bundle.arms if a.arm_id == "incumbent_qqq90_soxx10")
    assert incumbent.targets == {"QQQ": 0.9, "SOXX": 0.1}

