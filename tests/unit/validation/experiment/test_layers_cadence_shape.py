"""Unit tests for experiment JSON spec loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.policy.contribution_shape import ContributionShapeConfig
from src.validation.experiment import (
    ExperimentSpec,
    resolve_contribution_shape,
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


