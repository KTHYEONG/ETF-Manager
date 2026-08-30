"""Unit tests for experiment JSON spec loading."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.etf.mapping import DEFAULT_CANDIDATES, MappingConfig
from src.policy.adaptive_contribution import AdaptiveContributionConfig
from src.policy.contribution_shape import ContributionShapeConfig
from src.policy.kafi_deployment import KafiDeploymentConfig
from src.policy.currency import CurrencyConfig
from src.policy.overlay import OverlayConfig
from src.policy.reserve import ReserveConfig
from src.policy.targets import PolicyId
from src.policy.thesis import ThesisId, load_thesis_registry
from src.sim.allocation import AllocationConfig
from src.validation.experiment import (
    AdaptiveContributionSpec,
    CadenceSpec,
    CandidateSpec,
    ExperimentSpec,
    PreregistrationSpec,
    assert_experiment_preregistration,
    experiment_target_tickers,
    load_experiment_config,
    resolve_adaptive_contribution,
    resolve_arm_targets,
    resolve_baseline_adaptive_contribution,
    resolve_cadence,
    resolve_contribution_shape,
    resolve_currency,
    resolve_kafi_deployment,
    resolve_mapping,
    resolve_overlay,
    resolve_reserve,
)
from src.validation.registry import make_experiment


@pytest.mark.parametrize("scenario_id", ["EXP-MIX-targets-schema"])
def test_exp_mix_targets_schema(scenario_id: str) -> None:
    """EXP-MIX-targets-schema"""
    arm = CandidateSpec(id="mix", policy=PolicyId.QQQ, modules=1, targets={"QQQ": 0.9, "GRID": 0.1})
    assert arm.targets == {"QQQ": 0.9, "GRID": 0.1}
    assert sum(arm.targets.values()) == pytest.approx(1.0, abs=1e-12)
    assert resolve_arm_targets(arm) == {"QQQ": 0.9, "GRID": 0.1}
    assert resolve_arm_targets(CandidateSpec(id="bare", policy=PolicyId.QQQ, modules=0)) is None
    spec = ExperimentSpec(
        name="mix_schema",
        start=date(2012, 1, 3),
        end=date(2024, 12, 31),
        contribution_krw=1_000_000.0,
        hurdle=0.02,
        horizon_months=0,
        baseline=CandidateSpec(id="qqq_baseline", policy=PolicyId.QQQ, modules=0),
        candidates=[arm],
    )
    assert experiment_target_tickers(spec) == ("QQQ", "GRID")


@pytest.mark.parametrize("scenario_id", ["EXP-MIX-targets-fail-closed"])
def test_exp_mix_targets_fail_closed(scenario_id: str) -> None:
    """EXP-MIX-targets-fail-closed"""
    with pytest.raises(ValidationError):
        CandidateSpec(id="empty", policy=PolicyId.QQQ, modules=1, targets={})
    with pytest.raises(ValidationError):
        CandidateSpec(id="half", policy=PolicyId.QQQ, modules=1, targets={"QQQ": 0.5})
    with pytest.raises(ValidationError):
        CandidateSpec(id="neg", policy=PolicyId.QQQ, modules=1, targets={"QQQ": 1.2, "GRID": -0.2})
    with pytest.raises(ValidationError):
        CandidateSpec(id="nan", policy=PolicyId.QQQ, modules=1, targets={"QQQ": float("nan")})
    with pytest.raises(ValidationError):
        CandidateSpec(id="blank", policy=PolicyId.QQQ, modules=1, targets={"": 1.0})
    with pytest.raises(ValidationError):
        CandidateSpec.model_validate(
            {
                "id": "dup",
                "policy": "qqq",
                "modules": 1,
                "targets": {"qqq": 0.5, "QQQ": 0.5},
            }
        )


@pytest.mark.parametrize("scenario_id", ["EXP-MIX-json-iwf"])
def test_exp_mix_json_iwf(scenario_id: str) -> None:
    """EXP-MIX-json-iwf"""
    spec = load_experiment_config("configs/experiments/m_qqq_iwf.json")
    assert spec.contribution_krw == pytest.approx(1_000_000.0)
    assert spec.hurdle == pytest.approx(0.02)
    assert spec.horizon_months == 36
    assert spec.adaptive_contribution is None
    assert spec.baseline.policy is PolicyId.QQQ
    assert spec.baseline.targets is None
    assert len(spec.candidates) == 1
    candidate = spec.candidates[0]
    assert candidate.id == "iwf_100"
    assert candidate.policy is PolicyId.QQQ
    assert candidate.modules == 1
    assert candidate.targets == {"IWF": 1.0}


@pytest.mark.parametrize("scenario_id", ["EXP-MIX-json-grid"])
def test_exp_mix_json_grid(scenario_id: str) -> None:
    """EXP-MIX-json-grid"""
    spec = load_experiment_config("configs/experiments/m_qqq_grid.json")
    assert spec.adaptive_contribution is None
    assert spec.baseline.policy is PolicyId.QQQ
    assert spec.baseline.targets is None
    assert len(spec.candidates) == 3
    grid_weights = [candidate.targets["GRID"] for candidate in spec.candidates if candidate.targets]
    qqq_residuals = [candidate.targets["QQQ"] for candidate in spec.candidates if candidate.targets]
    assert grid_weights == [pytest.approx(0.05), pytest.approx(0.10), pytest.approx(0.15)]
    assert qqq_residuals == [pytest.approx(0.95), pytest.approx(0.90), pytest.approx(0.85)]
    assert all(candidate.modules == 1 for candidate in spec.candidates)


@pytest.mark.parametrize("scenario_id", ["EXP-MIX-json-future-core-wf"])
def test_exp_mix_json_future_core_wf(scenario_id: str) -> None:
    """EXP-MIX-json-future-core-wf"""
    spec = load_experiment_config("configs/experiments/wf_qqq_future_core.json")
    assert spec.train_months == 60
    assert spec.test_months == 36
    assert spec.horizon_months == 0
    assert spec.adaptive_contribution is None
    assert len(spec.candidates) == 1
    assert spec.candidates[0].targets == {
        "QQQ": pytest.approx(0.80),
        "GRID": pytest.approx(0.10),
        "XLI": pytest.approx(0.10),
    }


@pytest.mark.parametrize("scenario_id", ["REG-MIX-identity-hash"])
def test_reg_mix_identity_hash(scenario_id: str) -> None:
    """REG-MIX-identity-hash"""
    base = AllocationConfig(
        policy=PolicyId.QQQ,
        start=date(2012, 1, 3),
        end=date(2024, 12, 31),
        monthly_contribution_krw=1_000_000.0,
    )
    bare = make_experiment(
        config=base,
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


@pytest.mark.parametrize("scenario_id", ["EXP-THESIS-schema-load"])
def test_exp_thesis_schema_load(scenario_id: str) -> None:
    """EXP-THESIS-schema-load"""
    spec = load_experiment_config("configs/experiments/m_thesis_ai_compute_soxx.json")
    assert spec.thesis_id == ThesisId.AI_COMPUTE
    assert spec.preregistration is not None
    assert spec.preregistration.weights_locked is True


@pytest.mark.parametrize("scenario_id", ["EXP-THESIS-fail-unknown"])
def test_exp_thesis_fail_unknown(scenario_id: str) -> None:
    """EXP-THESIS-fail-unknown"""
    spec = ExperimentSpec.model_construct(
        name="unknown_thesis",
        start=date(2012, 1, 3),
        end=date(2024, 12, 31),
        contribution_krw=1_000_000.0,
        hurdle=0.02,
        objective="ce",
        horizon_months=0,
        commission_bps=0.0,
        fx_spread_bps=0.0,
        train_months=None,
        test_months=None,
        overlay=None,
        reserve=None,
        mapping=None,
        currency=None,
        cadence=None,
        contribution_shape=None,
        kafi_deployment=None,
        adaptive_contribution=None,
        baseline_adaptive_contribution=None,
        baseline=CandidateSpec(id="qqq_baseline", policy=PolicyId.QQQ, modules=0),
        candidates=[CandidateSpec(id="cand", policy=PolicyId.QQQ, modules=1)],
        thesis_id="nonexistent",
        preregistration=None,
    )
    with pytest.raises(ValueError, match="unknown thesis"):
        assert_experiment_preregistration(spec, {})


@pytest.mark.parametrize("scenario_id", ["EXP-PREREG-universe-lock"])
def test_exp_prereg_universe_lock(scenario_id: str) -> None:
    """EXP-PREREG-universe-lock"""
    spec = ExperimentSpec(
        name="prereg_iwf_reject",
        start=date(2012, 1, 3),
        end=date(2024, 12, 31),
        contribution_krw=1_000_000.0,
        hurdle=0.02,
        horizon_months=0,
        thesis_id=ThesisId.AI_COMPUTE,
        preregistration=PreregistrationSpec(universe_locked=True),
        baseline=CandidateSpec(
            id="qqq_baseline",
            policy=PolicyId.QQQ,
            modules=0,
            targets={"QQQ": 1.0},
        ),
        candidates=[
            CandidateSpec(
                id="iwf_only",
                policy=PolicyId.QQQ,
                modules=1,
                targets={"IWF": 1.0},
            )
        ],
    )
    registry = load_thesis_registry(Path("configs/theses"))
    with pytest.raises(ValueError, match="IWF"):
        assert_experiment_preregistration(spec, registry)


@pytest.mark.parametrize("scenario_id", ["EXP-PREREG-legacy-unchanged"])
def test_exp_prereg_legacy_unchanged(scenario_id: str) -> None:
    """EXP-PREREG-legacy-unchanged"""
    spec = load_experiment_config("configs/experiments/m_qqq_iwf.json")
    assert spec.thesis_id is None
    assert spec.preregistration is None

@pytest.mark.parametrize("scenario_id", ["EXP-LH-objective-load"])
def test_exp_lh_objective_load(scenario_id: str) -> None:
    spec = load_experiment_config("configs/experiments/m_thesis_ai_compute_soxx_120m.json")
    assert spec.objective == "long_horizon"
    assert spec.horizon_months == 120


@pytest.mark.parametrize("scenario_id", ["EXP-GRID-load"])
def test_exp_grid_load(scenario_id: str) -> None:
    """EXP-GRID-load"""
    spec = load_experiment_config("configs/experiments/m_thesis_ai_power_bottleneck_grid.json")
    assert spec.thesis_id == ThesisId.AI_POWER_BOTTLENECK
    assert spec.candidates[0].targets == {"GRID": 1.0}
    assert spec.preregistration is not None
    assert spec.preregistration.weights_locked is True
