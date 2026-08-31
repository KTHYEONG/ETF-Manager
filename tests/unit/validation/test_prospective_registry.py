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
