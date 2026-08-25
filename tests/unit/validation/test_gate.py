"""Unit tests for CE, adoption gate, and plateau selection."""

from __future__ import annotations

import pytest

from src.etf_manager.validation.gate import adoption_passes, certainty_equivalent, select_plateau


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
