"""Unit tests for CE, adoption gate, and plateau selection."""

from __future__ import annotations

import pytest

from src.validation.gate import (
    adoption_passes,
    certainty_equivalent,
    growth_first_process_passes,
    growth_first_train_passes,
    select_plateau,
)


@pytest.mark.parametrize("scenario_id", ["VAL-V02-ce-and-gate"])
def test_val_v02_ce_and_gate(scenario_id: str) -> None:
    """VAL-V02-ce-and-gate"""
    assert certainty_equivalent((2.0, 2.0, 2.0), gamma=2) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="gamma"):
        certainty_equivalent((1.0,), gamma=1.0)
    with pytest.raises(ValueError, match="wealth"):
        certainty_equivalent((2.0, -1.0), gamma=5.0)

    candidate = {2.0: 1.1, 5.0: 1.1, 10.0: 1.1}
    baseline = {2.0: 1.0, 5.0: 1.0, 10.0: 1.0}
    # Hurdle is 1 + delta0 * modules; boundary equality must fail the strict inequality.
    assert adoption_passes(candidate, baseline, delta0=0.05, modules=1) is True
    assert adoption_passes(candidate, baseline, delta0=0.05, modules=2) is False
    assert adoption_passes(candidate, baseline, delta0=0.0, modules=0) is True
    with pytest.raises(ValueError, match="gamma"):
        adoption_passes({2.0: 1.1}, {5.0: 1.0}, delta0=0.0, modules=0)

    assert select_plateau((0.08, 0.09, 0.10, 0.11), (1.0, 1.02, 1.03, 0.50), rel_tol=0.05) == pytest.approx(0.09)
    with pytest.raises(ValueError, match="disconnected"):
        select_plateau((0.08, 0.09, 0.10, 0.11), (1.03, 0.50, 1.02, 1.00), rel_tol=0.05)


@pytest.mark.parametrize("scenario_id", ["GF-A-train-and-process"])
def test_gf_a_train_and_process(scenario_id: str) -> None:
    """GF-A-train-and-process"""
    assert (
        growth_first_train_passes(candidate_tw=1.03, baseline_tw=1.0, candidate_mdd=-0.27, baseline_mdd=-0.28)
        is True
    )
    # Exact TW tie fails the strict-increase requirement.
    assert (
        growth_first_train_passes(candidate_tw=1.0, baseline_tw=1.0, candidate_mdd=-0.27, baseline_mdd=-0.28)
        is False
    )
    # MDD deeper than baseline by more than the default 0.02 slack fails.
    assert (
        growth_first_train_passes(candidate_tw=1.03, baseline_tw=1.0, candidate_mdd=-0.31, baseline_mdd=-0.28)
        is False
    )
    # Exactly at the slack boundary (-0.30 >= -0.28 - 0.02) passes.
    assert (
        growth_first_train_passes(candidate_tw=1.03, baseline_tw=1.0, candidate_mdd=-0.30, baseline_mdd=-0.28)
        is True
    )
    with pytest.raises(ValueError, match="finite"):
        growth_first_train_passes(
            candidate_tw=float("nan"), baseline_tw=1.0, candidate_mdd=-0.27, baseline_mdd=-0.28
        )

    assert growth_first_process_passes(chosen_test=(1.02, 1.01), baseline_test=(1.0, 1.0)) is True
    # A single fold below the default 0.97 floor vetoes adoption despite the pooled gain.
    assert growth_first_process_passes(chosen_test=(1.10, 0.96), baseline_test=(1.0, 1.0)) is False
    with pytest.raises(ValueError, match="finite"):
        growth_first_process_passes(chosen_test=(float("inf"), 1.0), baseline_test=(1.0, 1.0))
    with pytest.raises(ValueError, match="length"):
        growth_first_process_passes(chosen_test=(1.01,), baseline_test=(1.0, 1.0))
