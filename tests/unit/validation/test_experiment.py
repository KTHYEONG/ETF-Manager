"""Shim: experiment tests moved to tests/unit/validation/experiment/."""

from __future__ import annotations

import pytest

from src.validation.experiment import load_experiment_config


@pytest.mark.parametrize("scenario_id", ["EXP-H-qqq-reserve-v2-json"])
def test_exp_h_qqq_reserve_v2_json(scenario_id: str) -> None:
    """EXP-H-qqq-reserve-v2-json — legacy path fallback via archive."""
    spec = load_experiment_config("configs/experiments/wf_qqq_reserve_v2.json")
    assert spec.name == "wf_qqq_reserve_v2"
