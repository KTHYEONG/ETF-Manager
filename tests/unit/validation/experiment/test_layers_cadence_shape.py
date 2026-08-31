"""Unit tests for experiment JSON spec loading."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.policy.contribution_shape import ContributionShapeConfig
from src.policy.kafi_deployment import KafiDeploymentConfig
from src.policy.targets import PolicyId
from src.validation.experiment import (
    ExperimentSpec,
    load_experiment_config,
    resolve_contribution_shape,
    resolve_kafi_deployment,
    resolve_reserve,
)


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


@pytest.mark.parametrize("scenario_id", ["NAM-GF-json-objective"])
def test_nam_gf_json_objective(scenario_id: str) -> None:
    """NAM-GF-json-objective"""
    spec = load_experiment_config("configs/experiments/wf_qqq_cadence.json")

    assert spec.objective == "growth_first"

    omitted = ExperimentSpec.model_validate(_payload())
    assert omitted.objective == "ce"

    unknown = _payload()
    unknown["objective"] = "median_tw"
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(unknown)

    no_cadence = _payload()
    no_cadence["objective"] = "growth_first"
    with pytest.raises(ValueError, match="cadence"):
        ExperimentSpec.model_validate(no_cadence)

    reserve_only = _payload()
    reserve_only["objective"] = "growth_first"
    reserve_only["reserve"] = {
        "schedule": "v3",
        "max_withhold": 0.10,
        "min_invest_multiplier": 0.70,
        "max_invest_multiplier": 3.0,
    }
    parsed = ExperimentSpec.model_validate(reserve_only)
    assert parsed.cadence is None
    assert parsed.reserve is not None
    assert parsed.reserve.schedule == "v3"

    both_modules = _payload()
    both_modules["objective"] = "growth_first"
    both_modules["cadence"] = {"anchor": "month_open"}
    both_modules["reserve"] = {"schedule": "v3", "max_withhold": 0.10}
    with pytest.raises(ValueError, match="cadence"):
        ExperimentSpec.model_validate(both_modules)


@pytest.mark.parametrize("scenario_id", ["EXP-L-qqq-reserve-v3"])
def test_exp_l_qqq_reserve_v3_json(scenario_id: str) -> None:
    """EXP-L-qqq-reserve-v3"""
    spec = load_experiment_config("configs/experiments/wf_qqq_reserve_v3.json")

    assert spec.name == "wf_qqq_reserve_v3"
    assert spec.objective == "growth_first"
    assert spec.start == date(2007, 10, 31)
    assert spec.end == date(2026, 6, 30)
    assert spec.contribution_krw == pytest.approx(1_000_000.0)
    assert spec.hurdle == pytest.approx(0.02)
    assert spec.train_months == 60
    assert spec.test_months == 36
    assert spec.baseline.policy is PolicyId.QQQ
    assert spec.baseline.modules == 0
    assert [candidate.policy for candidate in spec.candidates] == [PolicyId.QQQ]
    assert [candidate.modules for candidate in spec.candidates] == [1]
    assert spec.cadence is None
    assert spec.reserve is not None
    assert spec.reserve.schedule == "v3"
    assert spec.reserve.min_invest_multiplier == pytest.approx(0.70)
    assert spec.reserve.max_invest_multiplier == pytest.approx(3.0)
    assert spec.reserve.vix_threshold == pytest.approx(25.0)
    assert spec.reserve.reserve_max_months == pytest.approx(2.0)

    resolved = resolve_reserve(spec)
    assert resolved is not None
    assert resolved.schedule == "v3"
    assert resolved.vix_threshold == pytest.approx(25.0)
    assert resolved.reserve_max_months == pytest.approx(2.0)
    assert resolved.max_invest_multiplier == pytest.approx(3.0)


@pytest.mark.parametrize("scenario_id", ["EXP-L-qqq-reserve-v4"])
def test_exp_l_qqq_reserve_v4_json(scenario_id: str) -> None:
    """EXP-L-qqq-reserve-v4"""
    spec = load_experiment_config("configs/experiments/wf_qqq_reserve_v4.json")

    assert spec.name == "wf_qqq_reserve_v4"
    assert spec.objective == "growth_first"
    assert spec.start == date(2007, 8, 31)
    assert spec.end == date(2026, 6, 30)
    assert spec.contribution_krw == pytest.approx(1_000_000.0)
    assert spec.train_months == 60
    assert spec.test_months == 36
    assert spec.baseline.policy is PolicyId.QQQ
    assert spec.baseline.modules == 0
    assert [candidate.policy for candidate in spec.candidates] == [PolicyId.QQQ]
    assert [candidate.modules for candidate in spec.candidates] == [1]
    assert spec.cadence is None
    assert spec.reserve is not None
    assert spec.reserve.schedule == "v4"
    assert spec.reserve.min_invest_multiplier == pytest.approx(0.70)
    assert spec.reserve.max_invest_multiplier == pytest.approx(3.0)
    assert spec.reserve.vix_threshold == pytest.approx(20.0)
    assert spec.reserve.reserve_max_months == pytest.approx(2.0)

    resolved = resolve_reserve(spec)
    assert resolved is not None
    assert resolved.schedule == "v4"
    assert resolved.vix_threshold == pytest.approx(20.0)


def _shape_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "wf_qqq_kafi_shape",
        "start": "2007-08-31",
        "end": "2026-06-30",
        "contribution_krw": 1_000_000,
        "hurdle": 0.02,
        "objective": "growth_first",
        "horizon_months": 0,
        "train_months": 60,
        "test_months": 36,
        "baseline": {"id": "s8_us_nasdaq", "policy": "qqq", "modules": 0},
        "candidates": [{"id": "s8_us_nasdaq_kafi_shape", "policy": "qqq", "modules": 1}],
        "contribution_shape": {},
    }
    payload.update(overrides)  # type: ignore[arg-type]
    return payload


@pytest.mark.parametrize("scenario_id", ["EXP-K-shape-xor"])
def test_exp_k_shape_xor(scenario_id: str, tmp_path: Path) -> None:
    """EXP-K-shape-xor"""
    spec = ExperimentSpec.model_validate(_shape_payload())
    assert spec.contribution_shape is not None
    assert spec.reserve is None
    assert spec.overlay is None
    assert spec.cadence is None

    resolved = resolve_contribution_shape(spec)
    assert isinstance(resolved, ContributionShapeConfig)
    assert resolved.min_multiplier == pytest.approx(0.70)
    assert resolved.max_multiplier == pytest.approx(1.50)

    for conflict in ("reserve", "overlay", "cadence", "currency", "mapping"):
        payload = _shape_payload()
        payload[conflict] = {"anchor": "month_open"} if conflict == "cadence" else {"max_shift": 0.05}
        if conflict == "reserve":
            payload[conflict] = {"max_withhold": 0.05}
        if conflict == "mapping":
            payload[conflict] = {"min_improvement": 0.02}
        if conflict == "currency":
            payload[conflict] = {"max_defer": 0.5}
        with pytest.raises(ValidationError):
            ExperimentSpec.model_validate(payload)

    weak_modules = _shape_payload()
    weak_modules["candidates"] = [{"id": "c", "policy": "qqq", "modules": 0}]
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(weak_modules)

    ce_objective = _shape_payload(objective="ce")
    ce_objective["objective"] = "ce"
    spec_ce = ExperimentSpec.model_validate(ce_objective)
    assert resolve_contribution_shape(spec_ce) is not None


@pytest.mark.parametrize("scenario_id", ["EXP-K-shape-wf-json"])
def test_exp_k_shape_wf_json(scenario_id: str) -> None:
    """EXP-K-shape-wf-json"""
    spec = load_experiment_config("configs/experiments/wf_qqq_kafi_shape.json")

    assert spec.baseline.policy is PolicyId.QQQ
    assert spec.baseline.modules == 0
    assert [candidate.policy for candidate in spec.candidates] == [PolicyId.QQQ]
    assert [candidate.modules for candidate in spec.candidates] == [1]
    assert spec.objective == "growth_first"
    assert spec.contribution_krw == pytest.approx(1_000_000.0)
    assert spec.contribution_shape is not None


def _deployment_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "wf_qqq_kafi_deployment",
        "start": "2015-06-01",
        "end": "2026-06-30",
        "contribution_krw": 1_000_000,
        "hurdle": 0.02,
        "objective": "growth_first",
        "horizon_months": 0,
        "train_months": 60,
        "test_months": 36,
        "baseline": {"id": "s8_us_nasdaq", "policy": "qqq", "modules": 0},
        "candidates": [{"id": "s8_us_nasdaq_kafi_deployment", "policy": "qqq", "modules": 1}],
        "kafi_deployment": {},
    }
    payload.update(overrides)  # type: ignore[arg-type]
    return payload


@pytest.mark.parametrize("scenario_id", ["EXP-M-deployment-wf-json"])
def test_exp_m_deployment_wf_json(scenario_id: str) -> None:
    """EXP-M-deployment-wf-json"""
    spec = load_experiment_config("configs/experiments/wf_qqq_kafi_deployment.json")

    assert spec.baseline.policy is PolicyId.QQQ
    assert spec.baseline.modules == 0
    assert [candidate.policy for candidate in spec.candidates] == [PolicyId.QQQ]
    assert [candidate.modules for candidate in spec.candidates] == [1]
    assert spec.objective == "growth_first"
    assert spec.kafi_deployment is not None
    assert spec.contribution_shape is None

    resolved = resolve_kafi_deployment(spec)
    assert isinstance(resolved, KafiDeploymentConfig)
    assert resolved.max_multiplier == pytest.approx(1.30)


@pytest.mark.parametrize("scenario_id", ["EXP-M-deployment-xor"])
def test_exp_m_deployment_xor(scenario_id: str) -> None:
    """EXP-M-deployment-xor"""
    ExperimentSpec.model_validate(_deployment_payload())

    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(_deployment_payload(contribution_shape={}))

    for conflict in ("reserve", "cadence"):
        payload = _deployment_payload()
        if conflict == "reserve":
            payload[conflict] = {"max_withhold": 0.05}
        else:
            payload[conflict] = {"anchor": "month_open"}
        with pytest.raises(ValidationError):
            ExperimentSpec.model_validate(payload)


