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
    assert spec.end == date(2024, 8, 31)

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
    assert spec.end == date(2024, 8, 31)
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
    assert spec.end == date(2024, 8, 31)
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


