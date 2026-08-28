"""Unit tests for experiment identity hashing."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.data.settings import DataSettings
from src.policy.targets import PolicyId
from src.sim.allocation import AllocationConfig
from src.validation.ablation import AblationReport, AblationRow
from src.validation.experiment import CandidateSpec, ExperimentSpec
from src.validation.registry import (
    build_ablation_arm_outcomes,
    freeze_baseline_config_hash,
    make_experiment,
    write_ablation_run_record,
)


@pytest.mark.parametrize("scenario_id", ["REG-MIX-identity-hash"])
def test_reg_mix_identity_hash(scenario_id: str) -> None:
    """REG-MIX-identity-hash"""
    bare = make_experiment(
        config=AllocationConfig(
            policy=PolicyId.QQQ,
            start=date(2012, 1, 3),
            end=date(2024, 12, 31),
            monthly_contribution_krw=1_000_000.0,
            targets_override={"QQQ": 1.0},
        ),
        manifest_hash="manifest",
        git_commit="deadbeef",
        seed=None,
        metrics={},
    )
    mixed = make_experiment(
        config=AllocationConfig(
            policy=PolicyId.QQQ,
            start=date(2012, 1, 3),
            end=date(2024, 12, 31),
            monthly_contribution_krw=1_000_000.0,
            targets_override={"QQQ": 0.8, "GRID": 0.2},
        ),
        manifest_hash="manifest",
        git_commit="deadbeef",
        seed=None,
        metrics={},
    )
    assert bare.config_hash != mixed.config_hash
    assert bare.experiment_id != mixed.experiment_id


def _synthetic_ablation_report() -> AblationReport:
    return AblationReport(
        name="reg_arm",
        baseline_policy=PolicyId.QQQ,
        baseline_modules=0,
        baseline_wealths=(100.0,),
        baseline_ce={2.0: 100.0, 5.0: 100.0, 10.0: 100.0},
        rows=(
            AblationRow(
                candidate_id="reject_arm",
                policy=PolicyId.QQQ,
                modules=1,
                wealths=(99.0,),
                ce={2.0: 99.0, 5.0: 99.0, 10.0: 99.0},
                ce_ratio={2.0: 0.99, 5.0: 0.99, 10.0: 0.99},
                adopted=False,
            ),
            AblationRow(
                candidate_id="adopt_arm",
                policy=PolicyId.QQQ,
                modules=1,
                wealths=(103.0,),
                ce={2.0: 103.0, 5.0: 103.0, 10.0: 103.0},
                ce_ratio={2.0: 1.03, 5.0: 1.03, 10.0: 1.03},
                adopted=True,
            ),
        ),
    )


@pytest.mark.parametrize("scenario_id", ["REG-ARM-all-logged"])
def test_reg_arm_all_logged(scenario_id: str) -> None:
    """REG-ARM-all-logged"""
    outcomes = build_ablation_arm_outcomes(_synthetic_ablation_report())
    assert len(outcomes) == 2
    assert [outcome.adopted for outcome in outcomes] == [False, True]


@pytest.mark.parametrize("scenario_id", ["REG-ARM-write-json"])
def test_reg_arm_write_json(scenario_id: str, tmp_path: Path) -> None:
    """REG-ARM-write-json"""
    report = _synthetic_ablation_report()
    spec = ExperimentSpec(
        name="reg_arm",
        start=date(2012, 1, 3),
        end=date(2024, 12, 31),
        contribution_krw=1_000_000.0,
        hurdle=0.02,
        horizon_months=0,
        baseline=CandidateSpec(id="qqq_baseline", policy=PolicyId.QQQ, modules=0),
        candidates=[CandidateSpec(id="reject_arm", policy=PolicyId.QQQ, modules=1)],
    )
    record = make_experiment(
        config=AllocationConfig(
            policy=PolicyId.QQQ,
            start=spec.start,
            end=spec.end,
            monthly_contribution_krw=spec.contribution_krw,
        ),
        manifest_hash="manifest",
        git_commit="deadbeef",
        seed=None,
        metrics={},
    )
    settings = DataSettings(data_root=str(tmp_path))
    out_path = write_ablation_run_record(spec=spec, report=report, record=record, settings=settings)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(payload["arms"]) == len(report.rows)
    assert any(arm["adopted"] is False for arm in payload["arms"])


@pytest.mark.parametrize("scenario_id", ["REG-ARM-baseline-frozen"])
def test_reg_arm_baseline_frozen(scenario_id: str) -> None:
    """REG-ARM-baseline-frozen"""
    base_spec = ExperimentSpec(
        name="freeze",
        start=date(2012, 1, 3),
        end=date(2024, 12, 31),
        contribution_krw=1_000_000.0,
        hurdle=0.02,
        horizon_months=0,
        baseline=CandidateSpec(
            id="qqq_baseline",
            policy=PolicyId.QQQ,
            modules=0,
            targets={"QQQ": 1.0},
        ),
        candidates=[CandidateSpec(id="cand", policy=PolicyId.QQQ, modules=1)],
    )
    changed_spec = ExperimentSpec(
        name="freeze",
        start=date(2012, 1, 3),
        end=date(2024, 12, 31),
        contribution_krw=1_000_000.0,
        hurdle=0.02,
        horizon_months=0,
        baseline=CandidateSpec(
            id="qqq_baseline",
            policy=PolicyId.QQQ,
            modules=0,
            targets={"QQQ": 0.9, "SOXX": 0.1},
        ),
        candidates=[CandidateSpec(id="cand", policy=PolicyId.QQQ, modules=1)],
    )
    base_hash = freeze_baseline_config_hash(base_spec)
    changed_hash = freeze_baseline_config_hash(changed_spec)
    assert len(base_hash) == 64
    assert base_hash != changed_hash
