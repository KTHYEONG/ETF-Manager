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

