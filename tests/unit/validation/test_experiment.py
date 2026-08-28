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
from src.sim.allocation import AllocationConfig
from src.validation.experiment import (
    AdaptiveContributionSpec,
    CadenceSpec,
    CandidateSpec,
    ExperimentSpec,
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


@pytest.mark.parametrize("scenario_id", ["EXP-W1-load-fail-closed"])
def test_exp_w1_load_round_trip(scenario_id: str, tmp_path: Path) -> None:
    """EXP-W1-load-fail-closed"""
    spec = load_experiment_config(_write(tmp_path, _payload()))

    assert isinstance(spec, ExperimentSpec)
    assert spec.name == "m0_m1_strategic"
    assert spec.start == date(2012, 1, 3)
    assert spec.end == date(2024, 12, 31)
    assert spec.contribution_krw == pytest.approx(1_000_000.0)
    assert spec.hurdle == pytest.approx(0.02)
    assert spec.horizon_months == 0
    assert spec.baseline.id == "m0_global"
    assert spec.baseline.policy is PolicyId.VT
    assert spec.baseline.modules == 0
    assert [candidate.id for candidate in spec.candidates] == [
        "s1_us",
        "s2_regional",
        "s3_global_bond",
        "s4_defensive",
    ]
    assert [candidate.policy for candidate in spec.candidates] == [
        PolicyId.VTI,
        PolicyId.WORLD_SPLIT,
        PolicyId.VT_BND,
        PolicyId.VT_TREAS,
    ]
    assert all(candidate.modules == 1 for candidate in spec.candidates)


@pytest.mark.parametrize("scenario_id", ["EXP-M2-json-load"])
def test_exp_m2_json_load(scenario_id: str) -> None:
    """EXP-M2-json-load"""
    spec = load_experiment_config("configs/experiments/m1_m2.json")

    assert spec.name == "m1_m2_us_core_value"
    assert spec.start == date(2014, 1, 3)
    assert spec.end == date(2024, 9, 30)
    assert spec.contribution_krw == pytest.approx(1_000_000.0)
    assert spec.hurdle == pytest.approx(0.02)
    assert spec.horizon_months == 36
    assert spec.baseline.id == "s1_us"
    assert spec.baseline.policy is PolicyId.VTI
    assert spec.baseline.modules == 0
    assert [candidate.id for candidate in spec.candidates] == ["s6_us_core_value"]
    assert [candidate.policy for candidate in spec.candidates] == [PolicyId.VTI_VTV]
    assert [candidate.modules for candidate in spec.candidates] == [1]


@pytest.mark.parametrize("scenario_id", ["EXP-D-universe-json"])
def test_exp_d_universe_json(scenario_id: str) -> None:
    """EXP-D-universe-json"""
    m1_d = load_experiment_config("configs/experiments/m1_d_universe.json")

    assert m1_d.name == "m1_d_universe"
    assert m1_d.start == date(2014, 1, 3)
    assert m1_d.end == date(2024, 9, 30)
    assert m1_d.contribution_krw == pytest.approx(1_000_000.0)
    assert m1_d.hurdle == pytest.approx(0.02)
    assert m1_d.horizon_months == 36
    assert m1_d.baseline.id == "s1_us"
    assert m1_d.baseline.policy is PolicyId.VTI
    assert m1_d.baseline.modules == 0
    assert [candidate.id for candidate in m1_d.candidates] == ["s7_us_large_cap"]
    assert [candidate.policy for candidate in m1_d.candidates] == [PolicyId.IVV]
    assert [candidate.modules for candidate in m1_d.candidates] == [1]

    wf = load_experiment_config("configs/experiments/wf_vti_ivv.json")

    assert wf.name == "wf_vti_ivv"
    assert wf.train_months == 60
    assert wf.test_months == 36
    assert wf.horizon_months == 0
    assert wf.baseline.id == "s1_us"
    assert wf.baseline.policy is PolicyId.VTI
    assert wf.baseline.modules == 0
    assert [candidate.id for candidate in wf.candidates] == ["s7_us_large_cap"]
    assert [candidate.policy for candidate in wf.candidates] == [PolicyId.IVV]
    assert [candidate.modules for candidate in wf.candidates] == [1]


@pytest.mark.parametrize("scenario_id", ["EXP-N-nasdaq-json"])
def test_exp_n_nasdaq_json(scenario_id: str) -> None:
    """EXP-N-nasdaq-json"""
    m1_n = load_experiment_config("configs/experiments/m1_n_nasdaq.json")

    assert m1_n.name == "m1_n_nasdaq"
    assert m1_n.start == date(2006, 10, 31)
    assert m1_n.end == date(2026, 6, 30)
    assert m1_n.contribution_krw == pytest.approx(1_000_000.0)
    assert m1_n.hurdle == pytest.approx(0.02)
    assert m1_n.horizon_months == 36
    assert m1_n.baseline.id == "s1_us"
    assert m1_n.baseline.policy is PolicyId.VTI
    assert m1_n.baseline.modules == 0
    assert [candidate.id for candidate in m1_n.candidates] == ["s8_us_nasdaq"]
    assert [candidate.policy for candidate in m1_n.candidates] == [PolicyId.QQQ]
    assert [candidate.modules for candidate in m1_n.candidates] == [1]

    wf = load_experiment_config("configs/experiments/wf_vti_qqq.json")

    assert wf.name == "wf_vti_qqq"
    assert wf.train_months == 60
    assert wf.test_months == 36
    assert wf.horizon_months == 0
    assert wf.contribution_krw == pytest.approx(1_000_000.0)
    assert wf.hurdle == pytest.approx(0.02)
    assert wf.start == date(2006, 10, 31)
    assert wf.end == date(2026, 6, 30)
    assert wf.baseline.id == "s1_us"
    assert wf.baseline.policy is PolicyId.VTI
    assert wf.baseline.modules == 0
    assert [candidate.id for candidate in wf.candidates] == ["s8_us_nasdaq"]
    assert [candidate.policy for candidate in wf.candidates] == [PolicyId.QQQ]
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

    wf = load_experiment_config("configs/experiments/wf_vt_vti.json")
    assert wf.train_months == 60
    assert wf.test_months == 36
    assert wf.baseline.policy is PolicyId.VT
    assert [candidate.policy for candidate in wf.candidates] == [PolicyId.VTI]


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


@pytest.mark.parametrize("scenario_id", ["EXP-J-mapping-json"])
def test_exp_j_mapping_json(scenario_id: str) -> None:
    """EXP-J-mapping-json"""
    spec = load_experiment_config("configs/experiments/wf_vti_mapping.json")

    assert spec.mapping is not None
    assert spec.mapping.min_improvement == pytest.approx(0.02)
    assert spec.baseline.modules == 0
    assert spec.candidates[0].modules == 1
    assert spec.overlay is None
    assert spec.reserve is None
    assert spec.baseline.policy is PolicyId.VTI
    assert spec.candidates[0].policy is PolicyId.VTI
    assert spec.train_months == 60
    assert spec.test_months == 36
    assert spec.start == date(2014, 1, 3)
    assert spec.end == date(2024, 9, 30)

    resolved = resolve_mapping(spec)
    assert isinstance(resolved, MappingConfig)
    assert resolved.min_improvement == pytest.approx(0.02)
    assert resolved.candidates == DEFAULT_CANDIDATES

    omitted = ExperimentSpec.model_validate(_payload())
    assert omitted.mapping is None
    assert resolve_mapping(omitted) is None

    defaulted = _payload()
    defaulted["mapping"] = {}
    parsed_default = ExperimentSpec.model_validate(defaulted)
    assert parsed_default.mapping is not None
    assert parsed_default.mapping.min_improvement == pytest.approx(0.02)

    overlay_and_mapping = _payload()
    overlay_and_mapping["overlay"] = {"max_shift": 0.10}
    overlay_and_mapping["mapping"] = {"min_improvement": 0.02}
    with pytest.raises(ValueError, match="overlay"):
        ExperimentSpec.model_validate(overlay_and_mapping)

    reserve_and_mapping = _payload()
    reserve_and_mapping["reserve"] = {"max_withhold": 0.05}
    reserve_and_mapping["mapping"] = {"min_improvement": 0.02}
    with pytest.raises(ValueError, match="reserve"):
        ExperimentSpec.model_validate(reserve_and_mapping)

    zero_modules = _payload()
    zero_modules["mapping"] = {"min_improvement": 0.02}
    for candidate in zero_modules["candidates"]:  # type: ignore[union-attr]
        candidate["modules"] = 0
    with pytest.raises(ValueError, match="modules"):
        ExperimentSpec.model_validate(zero_modules)

    unknown_key = _payload()
    unknown_key["mapping"] = {"min_improvement": 0.02, "bogus": True}
    with pytest.raises(ValueError, match="bogus"):
        ExperimentSpec.model_validate(unknown_key)

    zero_min = _payload()
    zero_min["mapping"] = {"min_improvement": 0}
    with pytest.raises(ValueError, match="min_improvement"):
        ExperimentSpec.model_validate(zero_min)

    above_one = _payload()
    above_one["mapping"] = {"min_improvement": 1.01}
    with pytest.raises(ValueError, match="min_improvement"):
        ExperimentSpec.model_validate(above_one)


@pytest.mark.parametrize("scenario_id", ["EXP-K-currency-json"])
def test_exp_k_currency_json(scenario_id: str) -> None:
    """EXP-K-currency-json"""
    spec = load_experiment_config("configs/experiments/wf_vti_currency.json")

    assert spec.contribution_krw == pytest.approx(1_000_000.0)
    assert spec.train_months == 60
    assert spec.test_months == 36
    assert spec.start == date(2014, 1, 3)
    assert spec.end == date(2024, 9, 30)
    assert spec.overlay is None
    assert spec.reserve is None
    assert spec.mapping is None
    assert spec.baseline.modules == 0
    assert spec.candidates[0].modules == 1
    assert spec.currency is not None
    assert spec.currency.max_defer == pytest.approx(0.10)
    assert spec.currency.expensive_percentile == pytest.approx(0.80)

    resolved = resolve_currency(spec)
    assert isinstance(resolved, CurrencyConfig)
    assert resolved.max_defer == pytest.approx(0.10)
    assert resolved.expensive_percentile == pytest.approx(0.80)
    assert resolved.percentile_window == 252

    omitted = ExperimentSpec.model_validate(_payload())
    assert omitted.currency is None
    assert resolve_currency(omitted) is None

    overlay_and_currency = _payload()
    overlay_and_currency["overlay"] = {"max_shift": 0.10}
    overlay_and_currency["currency"] = {"max_defer": 0.10}
    with pytest.raises(ValueError, match="currency"):
        ExperimentSpec.model_validate(overlay_and_currency)

    reserve_and_currency = _payload()
    reserve_and_currency["reserve"] = {"max_withhold": 0.05}
    reserve_and_currency["currency"] = {"max_defer": 0.10}
    with pytest.raises(ValueError, match="currency"):
        ExperimentSpec.model_validate(reserve_and_currency)

    mapping_and_currency = _payload()
    mapping_and_currency["mapping"] = {"min_improvement": 0.02}
    mapping_and_currency["currency"] = {"max_defer": 0.10}
    with pytest.raises(ValueError, match="currency"):
        ExperimentSpec.model_validate(mapping_and_currency)

    zero_modules = _payload()
    zero_modules["currency"] = {"max_defer": 0.10}
    for candidate in zero_modules["candidates"]:  # type: ignore[union-attr]
        candidate["modules"] = 0
    with pytest.raises(ValueError, match="modules"):
        ExperimentSpec.model_validate(zero_modules)

    zero_defer = _payload()
    zero_defer["currency"] = {"max_defer": 0}
    with pytest.raises(ValueError, match="max_defer"):
        ExperimentSpec.model_validate(zero_defer)

    above_one = _payload()
    above_one["currency"] = {"max_defer": 1.01}
    with pytest.raises(ValueError, match="max_defer"):
        ExperimentSpec.model_validate(above_one)

    unknown_key = _payload()
    unknown_key["currency"] = {"max_defer": 0.10, "bogus": True}
    with pytest.raises(ValueError, match="bogus"):
        ExperimentSpec.model_validate(unknown_key)


@pytest.mark.parametrize("scenario_id", ["EXP-L-cadence-json"])
def test_exp_l_cadence_json(scenario_id: str) -> None:
    """EXP-L-cadence-json"""
    spec = load_experiment_config("configs/experiments/wf_vti_cadence.json")

    assert spec.cadence is not None
    assert spec.cadence.anchor == "month_open"
    assert resolve_cadence(spec) == "month_open"
    assert spec.contribution_krw == pytest.approx(1_000_000.0)
    assert spec.hurdle == pytest.approx(0.02)
    assert spec.train_months == 60
    assert spec.test_months == 36
    assert spec.start == date(2014, 1, 3)
    assert spec.end == date(2024, 9, 30)
    assert spec.baseline.id == "s1_us"
    assert spec.baseline.policy is PolicyId.VTI
    assert spec.baseline.modules == 0
    assert [candidate.id for candidate in spec.candidates] == ["s1_us_month_open"]
    assert [candidate.modules for candidate in spec.candidates] == [1]
    assert spec.overlay is None
    assert spec.reserve is None
    assert spec.mapping is None
    assert spec.currency is None

    omitted = ExperimentSpec.model_validate(_payload())
    assert omitted.cadence is None
    assert resolve_cadence(omitted) is None

    overlay_and_cadence = _payload()
    overlay_and_cadence["overlay"] = {"max_shift": 0.10}
    overlay_and_cadence["cadence"] = {"anchor": "month_open"}
    with pytest.raises(ValueError, match="cadence"):
        ExperimentSpec.model_validate(overlay_and_cadence)

    zero_modules = _payload()
    zero_modules["cadence"] = {"anchor": "month_open"}
    for candidate in zero_modules["candidates"]:  # type: ignore[union-attr]
        candidate["modules"] = 0
    with pytest.raises(ValueError, match="modules"):
        ExperimentSpec.model_validate(zero_modules)

    with pytest.raises(ValueError, match="anchor"):
        CadenceSpec(anchor="month_end")


@pytest.mark.parametrize("scenario_id", ["EXP-L-qqq-cadence-json"])
def test_exp_l_qqq_cadence_json(scenario_id: str) -> None:
    """EXP-L-qqq-cadence-json"""
    spec = load_experiment_config("configs/experiments/wf_qqq_cadence.json")

    assert spec.name == "wf_qqq_cadence"
    assert spec.start == date(2007, 8, 31)
    assert spec.end == date(2026, 6, 30)
    assert spec.contribution_krw == pytest.approx(1_000_000.0)
    assert spec.hurdle == pytest.approx(0.02)
    assert spec.horizon_months == 0
    assert spec.train_months == 60
    assert spec.test_months == 36
    assert spec.cadence is not None
    assert spec.cadence.anchor == "month_open"
    assert resolve_cadence(spec) == "month_open"
    assert spec.baseline.id == "s8_us_nasdaq"
    assert spec.baseline.policy is PolicyId.QQQ
    assert spec.baseline.modules == 0
    assert [candidate.id for candidate in spec.candidates] == ["s8_us_nasdaq_month_open"]
    assert [candidate.policy for candidate in spec.candidates] == [PolicyId.QQQ]
    assert [candidate.modules for candidate in spec.candidates] == [1]
    assert spec.overlay is None
    assert spec.reserve is None
    assert spec.mapping is None
    assert spec.currency is None


@pytest.mark.parametrize("scenario_id", ["EXP-L-qqq-cadence-twice"])
def test_exp_l_qqq_cadence_twice(scenario_id: str) -> None:
    """EXP-L-qqq-cadence-twice"""
    spec = load_experiment_config("configs/experiments/wf_qqq_cadence_twice.json")

    assert spec.name == "wf_qqq_cadence_twice"
    assert spec.start == date(2007, 8, 31)
    assert spec.end == date(2026, 6, 30)
    assert spec.contribution_krw == pytest.approx(1_000_000.0)
    assert spec.hurdle == pytest.approx(0.02)
    assert spec.train_months == 60
    assert spec.test_months == 36
    assert spec.cadence is not None
    assert spec.cadence.anchor == "twice_monthly"
    assert resolve_cadence(spec) == "twice_monthly"
    assert spec.baseline.id == "s8_us_nasdaq"
    assert spec.baseline.policy is PolicyId.QQQ
    assert spec.baseline.modules == 0
    assert len(spec.candidates) == 1
    candidate = spec.candidates[0]
    assert candidate.id == "s8_us_nasdaq_twice"
    assert candidate.policy is PolicyId.QQQ
    assert candidate.modules == 1
    assert spec.overlay is None
    assert spec.reserve is None
    assert spec.mapping is None
    assert spec.currency is None

    with pytest.raises(ValueError, match="anchor"):
        CadenceSpec(anchor="month_end")

    overlay_and_cadence = _payload()
    overlay_and_cadence["overlay"] = {"max_shift": 0.10}
    overlay_and_cadence["cadence"] = {"anchor": "twice_monthly"}
    with pytest.raises(ValueError, match="cadence"):
        ExperimentSpec.model_validate(overlay_and_cadence)


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
