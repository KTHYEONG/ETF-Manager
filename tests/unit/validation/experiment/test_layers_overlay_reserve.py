"""Unit tests for experiment JSON spec loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.policy.targets import PolicyId
from src.validation.experiment import (
    ExperimentSpec,
    load_experiment_config,
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
    ("negative hurdle", lambda payload: payload.update(hurdle=-0.01), "hurdle"),
    ("negative modules", lambda payload: payload["baseline"].update(modules=-1), "modules"),  # type: ignore[union-attr]
    ("negative horizon_months", lambda payload: payload.update(horizon_months=-3), "horizon_months"),
]

@pytest.mark.parametrize("scenario_id", ["EXP-G-overlay-fail-closed"])
def test_exp_g_overlay_fail_closed(scenario_id: str) -> None:
    """EXP-G-overlay-fail-closed"""
    too_large = _payload()
    too_large["overlay"] = {"max_shift": 0.11}
    with pytest.raises(ValueError, match="max_shift"):
        ExperimentSpec.model_validate(too_large)

    zero_shift = _payload()
    zero_shift["overlay"] = {"max_shift": 0}
    with pytest.raises(ValueError, match="max_shift"):
        ExperimentSpec.model_validate(zero_shift)

    zero_modules = _payload()
    zero_modules["overlay"] = {"max_shift": 0.10}
    for candidate in zero_modules["candidates"]:  # type: ignore[union-attr]
        candidate["modules"] = 0
    with pytest.raises(ValueError, match="modules"):
        ExperimentSpec.model_validate(zero_modules)

    unknown_key = _payload()
    unknown_key["overlay"] = {"max_shift": 0.10, "bogus": True}
    with pytest.raises(ValueError, match="bogus"):
        ExperimentSpec.model_validate(unknown_key)


def _alias_payload(canonical: bool) -> dict[str, object]:
    """Same experiment expressed with canonical or legacy JSON keys and policy ids."""
    policy = "vti" if canonical else "s1_us"
    module_key = "modules" if canonical else "extra_rules"
    return {
        "name": "naming_alias",
        "start": "2012-01-03",
        "end": "2024-12-31",
        "contribution_krw": 1_000_000,
        "hurdle" if canonical else "delta0": 0.02,
        "horizon_months": 0,
        "baseline": {"id": "base", "policy": policy, module_key: 0},
        "candidates": [{"id": "cand", "policy": policy, module_key: 1}],
        "overlay": {"max_shift" if canonical else "max_tilt": 0.10},
    }


@pytest.mark.parametrize("scenario_id", ["NAM-A02-experiment-json-aliases"])
def test_nam_a02_experiment_json_aliases(scenario_id: str, tmp_path: Path) -> None:
    """NAM-A02-experiment-json-aliases"""
    legacy = load_experiment_config(_write(tmp_path, _alias_payload(canonical=False)))
    assert legacy.hurdle == pytest.approx(0.02)
    assert legacy.candidates[0].modules == 1
    assert legacy.overlay is not None
    assert legacy.overlay.max_shift == pytest.approx(0.10)
    assert legacy.baseline.policy is PolicyId.VTI
    assert legacy.candidates[0].policy is PolicyId.VTI

    canonical = load_experiment_config(_write(tmp_path, _alias_payload(canonical=True)))
    assert canonical.hurdle == pytest.approx(0.02)
    assert canonical.candidates[0].modules == 1
    assert canonical.overlay is not None
    assert canonical.overlay.max_shift == pytest.approx(0.10)
    assert canonical.baseline.policy is PolicyId.VTI
    assert canonical.candidates[0].policy is PolicyId.VTI

    dumped = json.loads(canonical.model_dump_json(by_alias=True))
    assert dumped["hurdle"] == pytest.approx(0.02)
    assert dumped["baseline"]["modules"] == 0
    assert dumped["candidates"][0]["modules"] == 1
    assert dumped["overlay"]["max_shift"] == pytest.approx(0.10)
    assert dumped["candidates"][0]["policy"] == "vti"

    conflict = _payload()
    conflict["hurdle"] = 0.02
    conflict["delta0"] = 0.03
    with pytest.raises(ValueError, match="hurdle"):
        load_experiment_config(_write(tmp_path, conflict))

