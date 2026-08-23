"""Unit tests for experiment JSON spec loading."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.etf_manager.policy.targets import PolicyId
from src.etf_manager.validation.experiment import ExperimentSpec, load_experiment_config


def _payload() -> dict[str, object]:
    return {
        "name": "m0_m1_strategic",
        "start": "2012-01-03",
        "end": "2024-12-31",
        "contribution_krw": 1_000_000,
        "delta0": 0.02,
        "horizon_months": 0,
        "baseline": {"id": "m0_global", "policy": "s0_global", "modules": 0},
        "candidates": [
            {"id": "s1_us", "policy": "s1_us", "modules": 1},
            {"id": "s2_regional", "policy": "s2_regional", "modules": 1},
            {"id": "s3_global_bond", "policy": "s3_global_bond", "modules": 1},
            {"id": "s4_defensive", "policy": "s4_defensive", "modules": 1},
        ],
    }


def _write(tmp_path: Path, payload: dict[str, object]) -> str:
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return str(config_path)


@pytest.mark.parametrize("scenario_id", ["EXP-W1-load-fail-closed"])
def test_exp_w1_load_round_trip(scenario_id: str, tmp_path: Path) -> None:
    """EXP-W1-load-fail-closed"""
    spec = load_experiment_config(_write(tmp_path, _payload()))

    assert isinstance(spec, ExperimentSpec)
    assert spec.name == "m0_m1_strategic"
    assert spec.start == date(2012, 1, 3)
    assert spec.end == date(2024, 12, 31)
    assert spec.contribution_krw == pytest.approx(1_000_000.0)
    assert spec.delta0 == pytest.approx(0.02)
    assert spec.horizon_months == 0
    assert spec.baseline.id == "m0_global"
    assert spec.baseline.policy is PolicyId.S0_GLOBAL
    assert spec.baseline.modules == 0
    assert [candidate.id for candidate in spec.candidates] == [
        "s1_us",
        "s2_regional",
        "s3_global_bond",
        "s4_defensive",
    ]
    assert [candidate.policy for candidate in spec.candidates] == [
        PolicyId.S1_US,
        PolicyId.S2_REGIONAL,
        PolicyId.S3_GLOBAL_BOND,
        PolicyId.S4_DEFENSIVE,
    ]
    assert all(candidate.modules == 1 for candidate in spec.candidates)


@pytest.mark.parametrize("scenario_id", ["EXP-M2-json-load"])
def test_exp_m2_json_load(scenario_id: str) -> None:
    """EXP-M2-json-load"""
    spec = load_experiment_config("configs/experiments/m1_m2.json")

    assert spec.name == "m1_m2_us_core_value"
    assert spec.start == date(2012, 4, 1)
    assert spec.end == date(2024, 11, 30)
    assert spec.contribution_krw == pytest.approx(1_000_000.0)
    assert spec.delta0 == pytest.approx(0.02)
    assert spec.horizon_months == 36
    assert spec.baseline.id == "s1_us"
    assert spec.baseline.policy is PolicyId.S1_US
    assert spec.baseline.modules == 0
    assert [candidate.id for candidate in spec.candidates] == ["s6_us_core_value"]
    assert [candidate.policy for candidate in spec.candidates] == [PolicyId.S6_US_CORE_VALUE]
    assert [candidate.modules for candidate in spec.candidates] == [1]


@pytest.mark.parametrize("scenario_id", ["EXP-WF-optional-months"])
def test_exp_wf_optional_months(scenario_id: str) -> None:
    """EXP-WF-optional-months"""
    m0_m1 = load_experiment_config("configs/experiments/m0_m1.json")
    assert m0_m1.train_months is None
    assert m0_m1.test_months is None

    only_train = dict(_payload())
    only_train["train_months"] = 60
    with pytest.raises(ValueError, match="both train_months and test_months"):
        ExperimentSpec.model_validate(only_train)

    wf = load_experiment_config("configs/experiments/wf_s0_s1.json")
    assert wf.train_months == 60
    assert wf.test_months == 36
    assert wf.baseline.policy is PolicyId.S0_GLOBAL
    assert [candidate.policy for candidate in wf.candidates] == [PolicyId.S1_US]


_FAIL_CASES = [
    ("empty candidates", lambda payload: payload.update(candidates=[]), "candidates"),
    (
        "duplicate candidate id",
        lambda payload: payload["candidates"].append(dict(payload["candidates"][0])),  # type: ignore[arg-type]
        "duplicate candidate id",
    ),
    (
        "unknown policy",
        lambda payload: payload["candidates"][0].update(policy="not_a_policy"),  # type: ignore[union-attr]
        "unknown policy",
    ),
    ("non-positive contribution", lambda payload: payload.update(contribution_krw=0), "contribution_krw"),
    ("start after end", lambda payload: payload.update(start="2025-01-01"), "is after end"),
    ("negative delta0", lambda payload: payload.update(delta0=-0.01), "delta0"),
    ("negative modules", lambda payload: payload["baseline"].update(modules=-1), "modules"),  # type: ignore[union-attr]
    ("negative horizon_months", lambda payload: payload.update(horizon_months=-3), "horizon_months"),
]


@pytest.mark.parametrize(
    ("scenario_id", "label"),
    [("EXP-W1-load-fail-closed", label) for label, _, _ in _FAIL_CASES],
)
def test_exp_w1_fail_closed(
    scenario_id: str,
    label: str,
    tmp_path: Path,
) -> None:
    """EXP-W1-load-fail-closed"""
    mutate = next(mutator for name, mutator, _ in _FAIL_CASES if name == label)
    match = next(pattern for name, _, pattern in _FAIL_CASES if name == label)

    payload = _payload()
    mutate(payload)

    with pytest.raises(ValueError, match=match):
        load_experiment_config(_write(tmp_path, payload))
