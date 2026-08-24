"""Unit tests for experiment JSON spec loading."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.etf_manager.policy.overlay import OverlayConfig
from src.etf_manager.policy.targets import PolicyId
from src.etf_manager.validation.experiment import (
    ExperimentSpec,
    load_experiment_config,
    resolve_overlay,
)


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


@pytest.mark.parametrize("scenario_id", ["EXP-D-universe-json"])
def test_exp_d_universe_json(scenario_id: str) -> None:
    """EXP-D-universe-json"""
    m1_d = load_experiment_config("configs/experiments/m1_d_universe.json")

    assert m1_d.name == "m1_d_universe"
    assert m1_d.start == date(2012, 4, 1)
    assert m1_d.end == date(2024, 11, 30)
    assert m1_d.contribution_krw == pytest.approx(1_000_000.0)
    assert m1_d.delta0 == pytest.approx(0.02)
    assert m1_d.horizon_months == 36
    assert m1_d.baseline.id == "s1_us"
    assert m1_d.baseline.policy is PolicyId.S1_US
    assert m1_d.baseline.modules == 0
    assert [candidate.id for candidate in m1_d.candidates] == ["s7_us_large_cap"]
    assert [candidate.policy for candidate in m1_d.candidates] == [PolicyId.S7_US_LARGE_CAP]
    assert [candidate.modules for candidate in m1_d.candidates] == [1]

    wf = load_experiment_config("configs/experiments/wf_s1_s7.json")

    assert wf.name == "wf_s1_s7"
    assert wf.train_months == 60
    assert wf.test_months == 36
    assert wf.horizon_months == 0
    assert wf.baseline.id == "s1_us"
    assert wf.baseline.policy is PolicyId.S1_US
    assert wf.baseline.modules == 0
    assert [candidate.id for candidate in wf.candidates] == ["s7_us_large_cap"]
    assert [candidate.policy for candidate in wf.candidates] == [PolicyId.S7_US_LARGE_CAP]
    assert [candidate.modules for candidate in wf.candidates] == [1]


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


@pytest.mark.parametrize("scenario_id", ["EXP-WF-B-json-costs"])
def test_exp_wf_b_json_costs(scenario_id: str, tmp_path: Path) -> None:
    """EXP-WF-B-json-costs"""
    defaults = load_experiment_config(_write(tmp_path, _payload()))
    assert defaults.commission_bps == pytest.approx(0.0)
    assert defaults.fx_spread_bps == pytest.approx(0.0)

    payload = _payload()
    payload.update(commission_bps=10, fx_spread_bps=20)
    loaded = load_experiment_config(_write(tmp_path, payload))
    assert loaded.commission_bps == pytest.approx(10.0)
    assert loaded.fx_spread_bps == pytest.approx(20.0)

    negative = _payload()
    negative.update(commission_bps=-0.1)
    with pytest.raises(ValueError, match="commission_bps"):
        load_experiment_config(_write(tmp_path, negative))


@pytest.mark.parametrize("scenario_id", ["EXP-G-overlay-json"])
def test_exp_g_overlay_json(scenario_id: str) -> None:
    """EXP-G-overlay-json"""
    spec = load_experiment_config("configs/experiments/wf_s1_overlay.json")

    assert spec.overlay is not None
    assert spec.overlay.max_shift == pytest.approx(0.10)
    assert spec.overlay.vix_threshold is None
    assert spec.baseline.policy is PolicyId.S1_US
    assert spec.baseline.modules == 0
    assert len(spec.candidates) == 1
    assert spec.candidates[0].policy is PolicyId.S1_US
    assert spec.candidates[0].modules == 1
    assert spec.train_months == 60
    assert spec.test_months == 36
    assert spec.start == date(2014, 1, 3)
    assert spec.end == date(2024, 9, 30)

    omitted = ExperimentSpec.model_validate(_payload())
    assert omitted.overlay is None

    resolved = resolve_overlay(spec)
    assert isinstance(resolved, OverlayConfig)
    assert resolved.max_shift == pytest.approx(0.10)
    assert resolve_overlay(omitted) is None


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
    policy = "us" if canonical else "s1_us"
    module_key = "extra_rules" if canonical else "modules"
    return {
        "name": "naming_alias",
        "start": "2012-01-03",
        "end": "2024-12-31",
        "contribution_krw": 1_000_000,
        "hurdle" if canonical else "delta0": 0.02,
        "horizon_months": 0,
        "baseline": {"id": "base", "policy": policy, module_key: 0},
        "candidates": [{"id": "cand", "policy": policy, module_key: 1}],
        "overlay": {"max_tilt" if canonical else "max_shift": 0.10},
    }


@pytest.mark.parametrize("scenario_id", ["NAM-A02-experiment-json-aliases"])
def test_nam_a02_experiment_json_aliases(scenario_id: str, tmp_path: Path) -> None:
    """NAM-A02-experiment-json-aliases"""
    legacy = load_experiment_config(_write(tmp_path, _alias_payload(canonical=False)))
    assert legacy.delta0 == pytest.approx(0.02)
    assert legacy.candidates[0].modules == 1
    assert legacy.overlay is not None
    assert legacy.overlay.max_shift == pytest.approx(0.10)
    assert legacy.baseline.policy is PolicyId.S1_US
    assert legacy.candidates[0].policy is PolicyId.S1_US

    canonical = load_experiment_config(_write(tmp_path, _alias_payload(canonical=True)))
    assert canonical.delta0 == pytest.approx(0.02)
    assert canonical.candidates[0].modules == 1
    assert canonical.overlay is not None
    assert canonical.overlay.max_shift == pytest.approx(0.10)
    assert canonical.baseline.policy is PolicyId.S1_US
    assert canonical.candidates[0].policy is PolicyId.S1_US

    dumped = json.loads(canonical.model_dump_json(by_alias=True))
    assert dumped["hurdle"] == pytest.approx(0.02)
    assert dumped["baseline"]["extra_rules"] == 0
    assert dumped["candidates"][0]["extra_rules"] == 1
    assert dumped["overlay"]["max_tilt"] == pytest.approx(0.10)
    assert dumped["candidates"][0]["policy"] == "us"

    conflict = _payload()
    conflict["hurdle"] = 0.02
    conflict["delta0"] = 0.03
    with pytest.raises(ValueError, match="hurdle"):
        load_experiment_config(_write(tmp_path, conflict))


@pytest.mark.parametrize("scenario_id", ["NAM-A03-shipped-wf-canonical"])
def test_nam_a03_shipped_wf_canonical(scenario_id: str) -> None:
    """NAM-A03-shipped-wf-canonical"""
    spec = load_experiment_config("configs/experiments/wf_s1_overlay.json")

    assert spec.baseline.policy is PolicyId.S1_US
    assert spec.overlay is not None
    assert spec.overlay.max_shift == pytest.approx(0.10)
    assert spec.start == date(2014, 1, 3)

    text = Path("configs/experiments/wf_s1_overlay.json").read_text(encoding="utf-8")
    for key in ("hurdle", "extra_rules", "max_tilt"):
        assert key in text
    assert "delta0" not in text
    assert "max_shift" not in text
    payload = json.loads(text)
    assert payload["overlay"]["max_tilt"] == pytest.approx(0.10)
    policies = [arm["policy"] for arm in (payload["baseline"], *payload["candidates"])]
    assert set(policies) == {"us"}
