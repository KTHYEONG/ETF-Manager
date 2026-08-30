# ruff: noqa
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


def _payload() -> dict[str, object]:
    return {
        "name": "m0_m1_strategic",
        "start": "2012-01-03",
        "end": "2024-12-31",
        "contribution_krw": 1_000_000,
        "hurdle": 0.02,
        "horizon_months": 0,
        "baseline": {"id": "m0_global", "policy": "vt", "modules": 0},
        "candidates": [
            {"id": "s1_us", "policy": "vti", "modules": 1},
            {"id": "s2_regional", "policy": "world_split", "modules": 1},
            {"id": "s3_global_bond", "policy": "vt_bnd", "modules": 1},
            {"id": "s4_defensive", "policy": "vt_treas", "modules": 1},
        ],
    }


def _write(tmp_path: Path, payload: dict[str, object]) -> str:
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return str(config_path)


def _adaptive_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "wf_qqq_adaptive_contribution",
        "start": "2015-06-01",
        "end": "2026-06-30",
        "contribution_krw": 1_000_000,
        "hurdle": 0.02,
        "objective": "adaptive_growth",
        "horizon_months": 0,
        "train_months": 60,
        "test_months": 36,
        "baseline": {"id": "s8_us_nasdaq", "policy": "qqq", "modules": 0},
        "candidates": [{"id": "s8_us_nasdaq_adaptive_contribution", "policy": "qqq", "modules": 1}],
        "adaptive_contribution": {},
    }
    payload.update(overrides)  # type: ignore[arg-type]
    return payload


@pytest.mark.parametrize("scenario_id", ["EXP-ACG-schema-wiring"])
def test_exp_acg_schema_wiring(scenario_id: str) -> None:
    """EXP-ACG-schema-wiring"""
    spec = load_experiment_config("configs/experiments/wf_qqq_adaptive_contribution.json")

    assert spec.objective == "adaptive_growth"
    assert spec.contribution_krw == pytest.approx(1_000_000.0)
    assert spec.train_months == 60
    assert spec.test_months == 36
    assert spec.baseline.policy is PolicyId.QQQ
    assert spec.baseline.modules == 0
    assert [candidate.modules for candidate in spec.candidates] == [1]
    module = spec.adaptive_contribution
    assert module is not None
    assert module.min_multiplier == pytest.approx(0.0)
    assert module.max_multiplier == pytest.approx(2.0)
    assert module.downside_power == pytest.approx(2.5)
    assert module.upside_power == pytest.approx(0.7)
    assert module.rank_window == 126

    resolved = resolve_adaptive_contribution(spec)
    assert isinstance(resolved, AdaptiveContributionConfig)
    assert resolved.min_multiplier == pytest.approx(0.0)
    assert resolved.max_multiplier == pytest.approx(2.0)
    assert resolved.downside_power == pytest.approx(2.5)
    assert resolved.upside_power == pytest.approx(0.7)
    assert resolved.rank_window == 126

    omitted = ExperimentSpec.model_validate(_payload())
    assert resolve_adaptive_contribution(omitted) is None

    for conflict_key, conflict_value in (
        ("overlay", {"max_shift": 0.05}),
        ("reserve", {"max_withhold": 0.05}),
        ("mapping", {"min_improvement": 0.02}),
        ("currency", {"max_defer": 0.10}),
        ("cadence", {"anchor": "month_open"}),
        ("contribution_shape", {}),
        ("kafi_deployment", {}),
    ):
        payload = _adaptive_payload()
        payload[conflict_key] = conflict_value
        with pytest.raises(ValidationError):
            ExperimentSpec.model_validate(payload)

    missing_module = _adaptive_payload()
    missing_module["adaptive_contribution"] = None
    with pytest.raises(ValueError, match="adaptive_contribution"):
        ExperimentSpec.model_validate(missing_module)

    growth_first_conflict = _adaptive_payload()
    growth_first_conflict["objective"] = "growth_first"
    with pytest.raises(ValidationError, match="growth_first"):
        ExperimentSpec.model_validate(growth_first_conflict)

    weak_modules = _adaptive_payload()
    weak_modules["candidates"] = [{"id": "c", "policy": "qqq", "modules": 0}]
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(weak_modules)

    unknown_key = _adaptive_payload()
    unknown_key["adaptive_contribution"] = {"bogus": True}
    with pytest.raises(ValueError, match="bogus"):
        ExperimentSpec.model_validate(unknown_key)


@pytest.mark.parametrize("scenario_id", ["ACR-EXP-schema-defaults"])
def test_acr_exp_schema_defaults(scenario_id: str) -> None:
    """ACR-EXP-schema-defaults"""
    spec = load_experiment_config("configs/experiments/wf_qqq_adaptive_contribution.json")

    module = spec.adaptive_contribution
    assert module is not None
    assert module.rank_window == 126
    assert module.downside_power == pytest.approx(2.5)
    assert module.upside_power == pytest.approx(0.7)
    assert module.min_multiplier == pytest.approx(0.0)
    assert module.max_multiplier == pytest.approx(2.0)

    resolved = resolve_adaptive_contribution(spec)
    assert isinstance(resolved, AdaptiveContributionConfig)
    assert resolved.rank_window == 126
    assert resolved.downside_power == pytest.approx(2.5)
    assert resolved.upside_power == pytest.approx(0.7)
    assert resolved.min_multiplier == pytest.approx(0.0)
    assert resolved.max_multiplier == pytest.approx(2.0)

    spec_defaults = AdaptiveContributionSpec()
    config_defaults = AdaptiveContributionConfig()
    assert spec_defaults.neutral_deadband == pytest.approx(0.0)
    assert config_defaults.neutral_deadband == pytest.approx(5.0)
    for field_name in (
        "equity_ticker",
        "bond_ticker",
        "credit_series_id",
        "min_multiplier",
        "max_multiplier",
        "rank_window",
    ):
        assert getattr(spec_defaults, field_name) == getattr(config_defaults, field_name)

    conflict = _adaptive_payload()
    conflict["reserve"] = {"max_withhold": 0.05}
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(conflict)


@pytest.mark.parametrize("scenario_id", ["EXP-AG-baseline-adaptive"])
def test_exp_ag_baseline_adaptive(scenario_id: str) -> None:
    """EXP-AG-baseline-adaptive"""
    spec = load_experiment_config("configs/experiments/wf_qqq_adaptive_v2.json")

    assert spec.objective == "adaptive_growth"
    assert spec.train_months == 60
    assert spec.test_months == 36
    baseline_spec = spec.baseline_adaptive_contribution
    candidate_spec = spec.adaptive_contribution
    assert baseline_spec is not None
    assert candidate_spec is not None
    assert baseline_spec.include_vol_dampener is True
    assert baseline_spec.upside_power == pytest.approx(0.7)
    assert candidate_spec.include_vol_dampener is False
    assert candidate_spec.upside_power == pytest.approx(0.5)

    resolved_baseline = resolve_baseline_adaptive_contribution(spec)
    resolved_candidate = resolve_adaptive_contribution(spec)
    assert isinstance(resolved_baseline, AdaptiveContributionConfig)
    assert isinstance(resolved_candidate, AdaptiveContributionConfig)
    assert resolved_baseline.include_vol_dampener is True
    assert resolved_baseline.upside_power == pytest.approx(0.7)
    assert resolved_candidate.include_vol_dampener is False
    assert resolved_candidate.upside_power == pytest.approx(0.5)
    for field_name in (
        "equity_ticker",
        "bond_ticker",
        "credit_series_id",
        "min_multiplier",
        "max_multiplier",
        "downside_power",
        "upside_power",
        "rank_window",
        "include_vol_dampener",
    ):
        assert getattr(resolved_baseline, field_name) == getattr(baseline_spec, field_name)
        assert getattr(resolved_candidate, field_name) == getattr(candidate_spec, field_name)

    missing_candidate = _adaptive_payload()
    missing_candidate["baseline_adaptive_contribution"] = {}
    missing_candidate["adaptive_contribution"] = None
    with pytest.raises(ValueError, match="adaptive_contribution"):
        ExperimentSpec.model_validate(missing_candidate)

    wrong_objective = _adaptive_payload()
    wrong_objective["baseline_adaptive_contribution"] = {}
    wrong_objective["objective"] = "ce"
    with pytest.raises(ValueError, match="adaptive_growth"):
        ExperimentSpec.model_validate(wrong_objective)

    xor_conflict = _adaptive_payload()
    xor_conflict["baseline_adaptive_contribution"] = {}
    xor_conflict["reserve"] = {"max_withhold": 0.05}
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(xor_conflict)


@pytest.mark.parametrize("scenario_id", ["EXP-AG-v3-config"])
def test_exp_ag_v3_config(scenario_id: str) -> None:
    """EXP-AG-v3-config"""
    spec = load_experiment_config("configs/experiments/wf_qqq_adaptive_v3.json")

    assert spec.objective == "adaptive_growth"
    baseline_spec = spec.baseline_adaptive_contribution
    candidate_spec = spec.adaptive_contribution
    assert baseline_spec is not None
    assert candidate_spec is not None
    assert baseline_spec.include_vol_dampener is True
    assert baseline_spec.upside_power == pytest.approx(0.7)
    assert baseline_spec.downside_power == pytest.approx(2.5)
    assert baseline_spec.dispersion == pytest.approx(1.0)
    assert candidate_spec.include_vol_dampener is False
    assert candidate_spec.upside_power == pytest.approx(0.5)
    assert candidate_spec.downside_power == pytest.approx(3.0)
    assert candidate_spec.dispersion == pytest.approx(1.0)

    resolved_baseline = resolve_baseline_adaptive_contribution(spec)
    resolved_candidate = resolve_adaptive_contribution(spec)
    assert isinstance(resolved_baseline, AdaptiveContributionConfig)
    assert isinstance(resolved_candidate, AdaptiveContributionConfig)
    for field_name in (
        "equity_ticker",
        "bond_ticker",
        "credit_series_id",
        "min_multiplier",
        "max_multiplier",
        "downside_power",
        "upside_power",
        "rank_window",
        "include_vol_dampener",
        "dispersion",
    ):
        assert getattr(resolved_baseline, field_name) == getattr(baseline_spec, field_name)
        assert getattr(resolved_candidate, field_name) == getattr(candidate_spec, field_name)


@pytest.mark.parametrize("scenario_id", ["EXP-AG-v4-json"])
def test_exp_ag_v4_json(scenario_id: str) -> None:
    """EXP-AG-v4-json"""
    spec = load_experiment_config("configs/experiments/wf_qqq_adaptive_v4.json")

    assert spec.objective == "adaptive_growth"
    baseline_spec = spec.baseline_adaptive_contribution
    candidate_spec = spec.adaptive_contribution
    assert baseline_spec is not None
    assert candidate_spec is not None
    assert candidate_spec.downside_power == pytest.approx(3.5)
    assert candidate_spec.upside_power == pytest.approx(0.35)
    assert candidate_spec.include_vol_dampener is False
    assert candidate_spec.dispersion == pytest.approx(1.15)
    assert candidate_spec.neutral_deadband == pytest.approx(4.0)
    assert baseline_spec.include_vol_dampener is True
    assert baseline_spec.upside_power == pytest.approx(0.7)
    assert baseline_spec.neutral_deadband == pytest.approx(0.0)

    resolved_baseline = resolve_baseline_adaptive_contribution(spec)
    resolved_candidate = resolve_adaptive_contribution(spec)
    assert resolved_candidate.neutral_deadband == pytest.approx(4.0)
    assert resolved_baseline.neutral_deadband == pytest.approx(0.0)


@pytest.mark.parametrize("scenario_id", ["EXP-AG-v5-resolve"])
def test_exp_ag_v5_resolve(scenario_id: str) -> None:
    """EXP-AG-v5-resolve"""
    spec = load_experiment_config("configs/experiments/wf_qqq_adaptive_v5.json")

    assert spec.objective == "adaptive_growth"
    baseline_spec = spec.baseline_adaptive_contribution
    candidate_spec = spec.adaptive_contribution
    assert baseline_spec is not None
    assert candidate_spec is not None
    assert candidate_spec.dispersion == pytest.approx(1.35)
    assert candidate_spec.upside_power == pytest.approx(0.25)
    assert candidate_spec.downside_power == pytest.approx(4.0)
    assert candidate_spec.neutral_deadband == pytest.approx(5.0)
    assert candidate_spec.include_vol_dampener is False
    assert candidate_spec.rank_window == 126
    assert baseline_spec.dispersion == pytest.approx(1.15)
    assert baseline_spec.upside_power == pytest.approx(0.35)
    assert baseline_spec.downside_power == pytest.approx(3.5)
    assert baseline_spec.neutral_deadband == pytest.approx(4.0)
    assert baseline_spec.include_vol_dampener is False
    assert baseline_spec.rank_window == 126

    resolved_candidate = resolve_adaptive_contribution(spec)
    resolved_baseline = resolve_baseline_adaptive_contribution(spec)
    assert resolved_candidate is not None
    assert resolved_baseline is not None
    assert resolved_candidate.dispersion == pytest.approx(1.35)
    assert resolved_candidate.upside_power == pytest.approx(0.25)
    assert resolved_candidate.downside_power == pytest.approx(4.0)
    assert resolved_candidate.neutral_deadband == pytest.approx(5.0)
    assert resolved_baseline.dispersion == pytest.approx(1.15)
    assert resolved_baseline.upside_power == pytest.approx(0.35)
    assert resolved_baseline.downside_power == pytest.approx(3.5)
    assert resolved_baseline.neutral_deadband == pytest.approx(4.0)


def test_exp_ag_soxx10_adaptive_v5_resolve() -> None:
    import pytest
    from src.validation.experiment import load_experiment_config, resolve_adaptive_contribution, resolve_baseline_adaptive_contribution
    spec = load_experiment_config('configs/experiments/wf_qqq_soxx10_adaptive_v5.json')
    assert spec.objective == 'adaptive_growth'
    assert spec.thesis_id is not None and spec.thesis_id.value == 'ai_compute'
    assert spec.baseline.modules == 1
    assert spec.candidates[0].modules == 2
    assert spec.baseline.targets == {'QQQ': 1.0}
    assert spec.candidates[0].targets == {'QQQ': 0.9, 'SOXX': 0.1}
    assert spec.cadence is None and spec.overlay is None and spec.reserve is None
    cand = resolve_adaptive_contribution(spec)
    base = resolve_baseline_adaptive_contribution(spec)
    assert cand is not None and base is not None
    assert cand.dispersion == pytest.approx(1.35)
    assert cand.upside_power == pytest.approx(0.25)
    assert cand.downside_power == pytest.approx(4.0)
    assert cand.neutral_deadband == pytest.approx(5.0)
    assert cand.include_vol_dampener is False
    assert cand.rank_window == 126
    assert base.dispersion == pytest.approx(cand.dispersion)
    assert base.upside_power == pytest.approx(cand.upside_power)
    assert base.downside_power == pytest.approx(cand.downside_power)
    assert base.neutral_deadband == pytest.approx(cand.neutral_deadband)


