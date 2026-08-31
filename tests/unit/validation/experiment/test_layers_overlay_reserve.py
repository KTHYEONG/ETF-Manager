"""Unit tests for experiment JSON spec loading."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.policy.overlay import OverlayConfig
from src.policy.reserve import ReserveConfig
from src.policy.targets import PolicyId
from src.validation.experiment import (
    ExperimentSpec,
    load_experiment_config,
    resolve_overlay,
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

@pytest.mark.parametrize("scenario_id", ["EXP-G-overlay-json"])
def test_exp_g_overlay_json(scenario_id: str) -> None:
    """EXP-G-overlay-json"""
    spec = load_experiment_config("configs/experiments/wf_vti_overlay.json")

    assert spec.overlay is not None
    assert spec.overlay.max_shift == pytest.approx(0.10)
    assert spec.overlay.vix_threshold is None
    assert spec.baseline.policy is PolicyId.VTI
    assert spec.baseline.modules == 0
    assert len(spec.candidates) == 1
    assert spec.candidates[0].policy is PolicyId.VTI
    assert spec.candidates[0].modules == 1
    assert spec.train_months == 60
    assert spec.test_months == 36
    assert spec.start == date(2014, 1, 3)
    assert spec.end == date(2024, 8, 31)

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


@pytest.mark.parametrize("scenario_id", ["EXP-H-reserve-json"])
def test_exp_h_reserve_json(scenario_id: str) -> None:
    """EXP-H-reserve-json"""
    spec = load_experiment_config("configs/experiments/wf_vti_reserve.json")

    assert spec.reserve is not None
    assert spec.reserve.max_withhold == pytest.approx(0.10)
    assert spec.overlay is None
    assert spec.baseline.modules == 0
    assert len(spec.candidates) == 1
    assert spec.candidates[0].modules == 1
    assert spec.baseline.policy is PolicyId.VTI
    assert spec.candidates[0].policy is PolicyId.VTI
    assert spec.train_months == 60
    assert spec.test_months == 36

    resolved = resolve_reserve(spec)
    assert isinstance(resolved, ReserveConfig)
    assert resolved.max_withhold == pytest.approx(0.10)

    omitted = ExperimentSpec.model_validate(_payload())
    assert omitted.reserve is None
    assert resolve_reserve(omitted) is None

    overlay_and_reserve = _payload()
    overlay_and_reserve["overlay"] = {"max_shift": 0.10}
    overlay_and_reserve["reserve"] = {"max_withhold": 0.05}
    with pytest.raises(ValueError, match="overlay"):
        ExperimentSpec.model_validate(overlay_and_reserve)

    zero_modules = _payload()
    zero_modules["reserve"] = {"max_withhold": 0.05}
    for candidate in zero_modules["candidates"]:  # type: ignore[union-attr]
        candidate["modules"] = 0
    with pytest.raises(ValueError, match="modules"):
        ExperimentSpec.model_validate(zero_modules)

    too_large = _payload()
    too_large["reserve"] = {"max_withhold": 0.11}
    with pytest.raises(ValueError, match="max_withhold"):
        ExperimentSpec.model_validate(too_large)

    canonical = _payload()
    canonical["reserve"] = {"withhold_cap": 0.10}
    parsed_alias = ExperimentSpec.model_validate(canonical)
    assert parsed_alias.reserve is not None
    assert parsed_alias.reserve.max_withhold == pytest.approx(0.10)

    unknown_key = _payload()
    unknown_key["reserve"] = {"max_withhold": 0.05, "bogus": True}
    with pytest.raises(ValueError, match="bogus"):
        ExperimentSpec.model_validate(unknown_key)


@pytest.mark.parametrize("scenario_id", ["EXP-H-qqq-reserve-json"])
def test_exp_h_qqq_reserve_json(scenario_id: str) -> None:
    """EXP-H-qqq-reserve-json"""
    spec = load_experiment_config("configs/experiments/wf_qqq_reserve.json")

    assert spec.name == "wf_qqq_reserve"
    assert spec.start == date(2007, 8, 31)
    assert spec.end == date(2026, 6, 30)
    assert spec.contribution_krw == pytest.approx(1_000_000.0)
    assert spec.train_months == 60
    assert spec.test_months == 36
    assert spec.baseline.modules == 0
    assert spec.baseline.policy is PolicyId.QQQ
    assert len(spec.candidates) == 1
    candidate = spec.candidates[0]
    assert candidate.id == "s8_us_nasdaq_reserve"
    assert candidate.policy is PolicyId.QQQ
    assert candidate.modules == 1
    assert spec.reserve is not None
    assert spec.reserve.max_withhold == pytest.approx(0.10)
    assert spec.overlay is None

    resolved = resolve_reserve(spec)
    assert isinstance(resolved, ReserveConfig)
    assert resolved.max_withhold == pytest.approx(0.10)



@pytest.mark.parametrize("scenario_id", ["EXP-H-qqq-reserve-v2-json"])
def test_exp_h_qqq_reserve_v2_json(scenario_id: str) -> None:
    """EXP-H-qqq-reserve-v2-json"""
    spec = load_experiment_config("configs/experiments/wf_qqq_reserve_v2.json")

    assert spec.name == "wf_qqq_reserve_v2"
    assert spec.start == date(2007, 8, 31)
    assert spec.end == date(2026, 6, 30)
    assert spec.contribution_krw == pytest.approx(1_000_000.0)
    assert spec.train_months == 60
    assert spec.test_months == 36
    assert spec.baseline.modules == 0
    assert spec.baseline.policy is PolicyId.QQQ
    assert len(spec.candidates) == 1
    candidate = spec.candidates[0]
    assert candidate.id == "s8_us_nasdaq_reserve_v2"
    assert candidate.policy is PolicyId.QQQ
    assert candidate.modules == 1
    assert spec.overlay is None
    assert spec.reserve is not None
    assert spec.reserve.schedule == "v2"

    resolved = resolve_reserve(spec)
    assert isinstance(resolved, ReserveConfig)
    assert resolved.schedule == "v2"
    assert resolved.min_invest_multiplier == pytest.approx(0.80)
    assert resolved.max_invest_multiplier == pytest.approx(2.00)
    assert resolved.reserve_max_months == pytest.approx(6.0)

    legacy_resolved = resolve_reserve(load_experiment_config("configs/experiments/wf_qqq_reserve.json"))
    assert isinstance(legacy_resolved, ReserveConfig)
    assert legacy_resolved.schedule == "v1"
    assert legacy_resolved.max_withhold == pytest.approx(0.10)


@pytest.mark.parametrize("scenario_id", ["EXP-G-qqq-overlay-json"])
def test_exp_g_qqq_overlay_json(scenario_id: str) -> None:
    """EXP-G-qqq-overlay-json"""
    spec = load_experiment_config("configs/experiments/wf_qqq_overlay.json")

    assert spec.name == "wf_qqq_overlay"
    assert spec.start == date(2007, 8, 31)
    assert spec.end == date(2026, 6, 30)
    assert spec.contribution_krw == pytest.approx(1_000_000.0)
    assert spec.train_months == 60
    assert spec.test_months == 36
    assert spec.baseline.modules == 0
    assert spec.baseline.policy is PolicyId.QQQ
    assert len(spec.candidates) == 1
    candidate = spec.candidates[0]
    assert candidate.id == "s8_us_nasdaq_overlay"
    assert candidate.policy is PolicyId.QQQ
    assert candidate.modules == 1
    assert spec.overlay is not None
    assert spec.overlay.max_shift == pytest.approx(0.10)
    assert spec.overlay.vix_threshold is None
    assert spec.reserve is None

    resolved = resolve_overlay(spec)
    assert isinstance(resolved, OverlayConfig)
    assert resolved.max_shift == pytest.approx(0.10)


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


@pytest.mark.parametrize("scenario_id", ["NAM-A-json-keys"])
def test_nam_a_json_keys(scenario_id: str) -> None:
    """NAM-A-json-keys"""
    spec = load_experiment_config("configs/experiments/wf_qqq_cadence.json")

    assert spec.baseline.policy is PolicyId.QQQ
    assert spec.cadence is not None
    assert spec.cadence.anchor == "month_open"
    assert spec.hurdle == pytest.approx(0.02)
    assert spec.candidates[0].modules == 1

    dumped = json.loads(spec.model_dump_json(by_alias=True))
    text = json.dumps(dumped)
    assert "hurdle" in text
    assert "modules" in text
    assert "extra_rules" not in text
    assert "delta0" not in text

    conflict = _payload()
    conflict["hurdle"] = 0.02
    conflict["delta0"] = 0.03
    with pytest.raises(ValueError, match="hurdle"):
        ExperimentSpec.model_validate(conflict)


@pytest.mark.parametrize("scenario_id", ["NAM-A03-shipped-wf-canonical"])
def test_nam_a03_shipped_wf_canonical(scenario_id: str) -> None:
    """NAM-A03-shipped-wf-canonical"""
    spec = load_experiment_config("configs/experiments/wf_vti_overlay.json")

    assert spec.baseline.policy is PolicyId.VTI
    assert spec.overlay is not None
    assert spec.overlay.max_shift == pytest.approx(0.10)
    assert spec.start == date(2014, 1, 3)

    text = Path("configs/experiments/wf_vti_overlay.json").read_text(encoding="utf-8")
    for key in ("hurdle", "modules", "max_shift"):
        assert key in text
    for legacy in ("delta0", "extra_rules", "max_tilt", "withhold_cap"):
        assert legacy not in text
    payload = json.loads(text)
    assert payload["overlay"]["max_shift"] == pytest.approx(0.10)
    policies = [arm["policy"] for arm in (payload["baseline"], *payload["candidates"])]
    assert set(policies) == {"vti"}


