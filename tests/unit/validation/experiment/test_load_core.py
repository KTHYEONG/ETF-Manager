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
    assert spec.end == date(2024, 8, 31)
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
    assert m1_d.end == date(2024, 8, 31)
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


