"""Unit tests for the buy-only contribution mixer."""

from __future__ import annotations

import pytest

from src.sim.contribution import allocate_contribution


@pytest.mark.parametrize("scenario_id", ["SIM-I01-band-none-identity"])
def test_sim_i01_band_none_identity(scenario_id: str) -> None:
    """SIM-I01-band-none-identity"""
    fractions = allocate_contribution(
        targets={"A": 0.7, "B": 0.3},
        marks_krw={"A": 90.0, "B": 10.0},
        nav_krw=100.0,
        commission_bps=0.0,
        rebalance_band=None,
    )
    assert fractions == {"A": pytest.approx(0.7, abs=1e-12), "B": pytest.approx(0.3, abs=1e-12)}


@pytest.mark.parametrize("scenario_id", ["SIM-I02-underweight-gets-cash"])
def test_sim_i02_underweight_gets_cash(scenario_id: str) -> None:
    """SIM-I02-underweight-gets-cash"""
    common = {
        "targets": {"A": 0.5, "B": 0.5},
        "marks_krw": {"A": 80.0, "B": 20.0},
        "nav_krw": 100.0,
        "commission_bps": 0.0,
    }
    tight = allocate_contribution(rebalance_band=0.0, **common)
    loose = allocate_contribution(rebalance_band=0.4, **common)

    assert tight["B"] > tight["A"]
    assert tight["B"] == pytest.approx(1.0)
    # band=0.4 misses the strict band rule (0.2 < 0.5-0.4 is false) but the
    # fallback underweight rule still routes all cash to B.
    assert loose["B"] == pytest.approx(1.0)


@pytest.mark.parametrize("scenario_id", ["SIM-I03-cost-aware-score"])
def test_sim_i03_cost_aware_score(scenario_id: str) -> None:
    """SIM-I03-cost-aware-score"""
    common = {
        "targets": {"A": 0.5, "B": 0.5},
        "marks_krw": {"A": 0.0, "B": 0.0},
        "nav_krw": 100.0,
        "rebalance_band": 0.0,
    }
    free = allocate_contribution(commission_bps=0.0, **common)
    costly = allocate_contribution(commission_bps=100.0, **common)

    for fractions in (free, costly):
        assert fractions["A"] == pytest.approx(0.5)
        assert fractions["B"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="rebalance_band"):
        allocate_contribution(
            targets={"A": 0.5, "B": 0.5},
            marks_krw={"A": 0.0, "B": 0.0},
            nav_krw=100.0,
            commission_bps=0.0,
            rebalance_band=1.0,
        )
    with pytest.raises(ValueError, match="nav_krw"):
        allocate_contribution(
            targets={"A": 0.5, "B": 0.5},
            marks_krw={"A": 0.0, "B": 0.0},
            nav_krw=0.0,
            commission_bps=0.0,
            rebalance_band=0.0,
        )


def test_r1_validation_and_missing_marks() -> None:
    """Remaining R1 clauses: simplex, negative marks/commission/band, mark key handling."""
    valid = {"nav_krw": 10.0, "commission_bps": 0.0}
    missing_mark_mix = allocate_contribution(
        rebalance_band=0.0, marks_krw={"C": 3.0}, targets={"A": 0.6, "B": 0.4}, **valid
    )
    assert sum(missing_mark_mix.values()) == pytest.approx(1.0)
    assert set(missing_mark_mix) == {"A", "B"}

    invalid_cases: list[tuple[dict[str, object], str]] = [
        ({**valid, "targets": {"A": 0.9, "B": 0.4}, "marks_krw": {}, "rebalance_band": None}, "simplex"),
        ({**valid, "targets": {"A": 1.2, "B": -0.2}, "marks_krw": {}, "rebalance_band": None}, "simplex"),
        ({**valid, "targets": {"A": 0.6, "B": 0.4}, "marks_krw": {"A": -1.0}, "rebalance_band": None}, "marks_krw"),
        (
            {**valid, "commission_bps": -0.1, "targets": {"A": 0.6, "B": 0.4}, "marks_krw": {}, "rebalance_band": None},
            "commission_bps",
        ),
        ({**valid, "targets": {"A": 0.6, "B": 0.4}, "marks_krw": {}, "rebalance_band": -0.01}, "rebalance_band"),
        ({**valid, "targets": {"A": 0.6, "B": 0.4}, "marks_krw": {}, "rebalance_band": 1.0}, "rebalance_band"),
        ({**valid, "nav_krw": -5.0, "targets": {"A": 0.6, "B": 0.4}, "marks_krw": {}, "rebalance_band": None}, "nav_krw"),
    ]
    for case, match in invalid_cases:
        with pytest.raises(ValueError, match=match):
            allocate_contribution(**case)  # type: ignore[arg-type]
