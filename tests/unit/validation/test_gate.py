"""Unit tests for CE, adoption gate, and plateau selection."""

from __future__ import annotations

import math

import pytest

from src.validation.gate import (
    adoption_passes,
    bootstrap_tail_passes,
    certainty_equivalent,
    cohort_win_rate,
    contiguous_adopted_plateau,
    contribution_growth_process_passes,
    contribution_growth_train_passes,
    growth_first_process_passes,
    growth_first_train_passes,
    select_plateau,
    wealth_quantile,
    worst_cohort_passes,
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


@pytest.mark.parametrize("scenario_id", ["GF-R-worst-and-tail"])
def test_gf_r_worst_and_tail(scenario_id: str) -> None:
    """GF-R-worst-and-tail"""
    low = wealth_quantile((1.0, 2.0, 3.0, 4.0), 0.05)
    assert math.isfinite(low)
    assert low <= wealth_quantile((1.0, 2.0, 3.0, 4.0), 0.5)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        wealth_quantile((1.0,), 1.0)
    with pytest.raises(ValueError, match="observation"):
        wealth_quantile((), 0.5)
    with pytest.raises(ValueError, match="finite"):
        wealth_quantile((float("inf"), 1.0), 0.5)

    # min ratio 0.98 clears the default 0.97 floor; one 0.96 fold vetoes.
    assert worst_cohort_passes((1.0, 0.98, 1.05), (1.0, 1.0, 1.0)) is True
    assert worst_cohort_passes((1.10, 0.96), (1.0, 1.0)) is False
    with pytest.raises(ValueError, match="length"):
        worst_cohort_passes((1.0,), (1.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        worst_cohort_passes((1.0, -1.0), (1.0, 1.0))

    flat = (1.02,) * 8
    degraded = (1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 0.5)
    assert bootstrap_tail_passes(flat, (1.0,) * 8, n_paths=40, seed=7) is True
    assert bootstrap_tail_passes(degraded, (1.0,) * 8, n_paths=40, seed=7) is False
    with pytest.raises(ValueError, match="n_paths"):
        bootstrap_tail_passes(flat, (1.0,) * 8, n_paths=0, seed=7)


@pytest.mark.parametrize("scenario_id", ["GATE-ACG-capital-aware"])
def test_gate_acg_capital_aware(scenario_id: str) -> None:
    """GATE-ACG-capital-aware"""
    common = {
        "candidate_tw": 110.0,
        "baseline_tw": 100.0,
        "candidate_real_gain": 11.0,
        "baseline_real_gain": 10.0,
        "candidate_xirr_real": 0.09,
        "baseline_xirr_real": 0.08,
        "candidate_mdd": -0.25,
        "baseline_mdd": -0.25,
    }
    # Higher TW with a weaker real gain fails: extra invested inflows cannot win.
    assert contribution_growth_train_passes(**{**common, "candidate_real_gain": 9.0}) is False
    # Higher TW and gain but inferior real XIRR fails.
    assert (
        contribution_growth_train_passes(**{**common, "candidate_xirr_real": 0.07}) is False
    )
    # MDD deeper than baseline by more than the default slack fails; at it passes.
    deep = common | {"candidate_mdd": -0.31, "baseline_mdd": -0.28}
    assert contribution_growth_train_passes(**deep) is False
    edge = common | {"candidate_xirr_real": 0.08, "candidate_mdd": -0.30, "baseline_mdd": -0.28}
    assert contribution_growth_train_passes(**edge) is True
    # Exact TW tie fails the strict inequality.
    assert contribution_growth_train_passes(**{**common, "candidate_tw": 100.0}) is False
    with pytest.raises(ValueError, match="finite"):
        contribution_growth_train_passes(**{**common, "candidate_tw": float("nan")})

    passing = {
        "chosen_test_tw": (102.0, 101.0),
        "baseline_test_tw": (100.0, 100.0),
        "chosen_test_real_gain": (12.0, 11.0),
        "baseline_test_real_gain": (10.0, 10.0),
        "chosen_test_xirr_real": (0.09, 0.08),
        "baseline_test_xirr_real": (0.08, 0.08),
    }
    assert contribution_growth_process_passes(**passing) is True
    # One fold below the 0.97 TW floor vetoes despite pooled TW and gain gains.
    weak_fold = passing | {"chosen_test_tw": (120.0, 96.0)}
    assert contribution_growth_process_passes(**weak_fold) is False
    # Pooled real gain failing vetoes even when pooled TW gains.
    weak_gain = {
        "chosen_test_tw": (105.0,),
        "baseline_test_tw": (100.0,),
        "chosen_test_real_gain": (4.0,),
        "baseline_test_real_gain": (5.0,),
        "chosen_test_xirr_real": (0.09,),
        "baseline_test_xirr_real": (0.08,),
    }
    assert contribution_growth_process_passes(**weak_gain) is False
    # An inferior fold XIRR vetoes.
    weak_xirr = {
        "chosen_test_tw": (102.0,),
        "baseline_test_tw": (100.0,),
        "chosen_test_real_gain": (11.0,),
        "baseline_test_real_gain": (10.0,),
        "chosen_test_xirr_real": (0.06,),
        "baseline_test_xirr_real": (0.08,),
    }
    assert contribution_growth_process_passes(**weak_xirr) is False
    # The fold-floor boundary is inclusive at exactly 0.97.
    boundary = passing | {"chosen_test_tw": (97.0, 110.0)}
    assert contribution_growth_process_passes(**boundary) is True

    with pytest.raises(ValueError, match="length"):
        contribution_growth_process_passes(**{**passing, "chosen_test_tw": (102.0,)})
    with pytest.raises(ValueError, match="finite"):
        contribution_growth_process_passes(**{**passing, "chosen_test_tw": (float("inf"), 101.0)})


@pytest.mark.parametrize("scenario_id", ["GAT-MIX-plateau-contiguous"])
def test_gat_mix_plateau_contiguous(scenario_id: str) -> None:
    """GAT-MIX-plateau-contiguous"""
    assert contiguous_adopted_plateau((0.05, 0.10, 0.15), (False, True, True)) is True
    assert contiguous_adopted_plateau((0.05, 0.10, 0.15), (True, False, True)) is False
    assert contiguous_adopted_plateau((0.05, 0.10, 0.15), (False, True, False)) is False
    with pytest.raises(ValueError, match="nonempty"):
        contiguous_adopted_plateau((), ())
    with pytest.raises(ValueError, match="length"):
        contiguous_adopted_plateau((0.05, 0.10), (True,))
    with pytest.raises(ValueError, match="strictly increasing"):
        contiguous_adopted_plateau((0.10, 0.05), (True, True))


@pytest.mark.parametrize("scenario_id", ["GAT-MIX-cohort-win-rate"])
def test_gat_mix_cohort_win_rate(scenario_id: str) -> None:
    """GAT-MIX-cohort-win-rate"""
    assert cohort_win_rate((2.0, 1.0, 3.0), (1.0, 1.0, 2.0)) == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="length"):
        cohort_win_rate((2.0,), (1.0, 1.0))
    with pytest.raises(ValueError, match="strictly positive"):
        cohort_win_rate((0.0, 1.0), (1.0, 1.0))


def test_compound_growth_train_passes_without_mdd_veto() -> None:
    from src.validation.gate import compound_growth_train_passes

    # SOXX100-like: +5.98억 real_gain scale but MDD -29.4% vs baseline -20.7%
    assert compound_growth_train_passes(
        candidate_tw=598.0,
        baseline_tw=129.0,
        candidate_real_gain=529.0,
        baseline_real_gain=2.0,
        candidate_xirr_real=0.12,
        baseline_xirr_real=0.08,
    ) is True
    # MDD deeper than contribution_growth would allow — still passes compound_growth
    deep_mdd = compound_growth_train_passes(
        candidate_tw=110.0,
        baseline_tw=100.0,
        candidate_real_gain=11.0,
        baseline_real_gain=10.0,
        candidate_xirr_real=0.09,
        baseline_xirr_real=0.08,
    )
    assert deep_mdd is True
    # Inferior real_gain fails
    assert compound_growth_train_passes(
        candidate_tw=110.0,
        baseline_tw=100.0,
        candidate_real_gain=9.0,
        baseline_real_gain=10.0,
        candidate_xirr_real=0.09,
        baseline_xirr_real=0.08,
    ) is False


def test_compound_growth_process_pooled_gain_no_mdd() -> None:
    from src.validation.gate import compound_growth_process_passes

    passing = {
        "chosen_test_tw": (102.0, 101.0),
        "baseline_test_tw": (100.0, 100.0),
        "chosen_test_real_gain": (12.0, 11.0),
        "baseline_test_real_gain": (10.0, 10.0),
        "chosen_test_xirr_real": (0.09, 0.08),
        "baseline_test_xirr_real": (0.08, 0.08),
    }
    assert compound_growth_process_passes(**passing) is True
    # Fold at exactly 0.95 TW ratio passes (relaxed vs 0.97)
    boundary = passing | {"chosen_test_tw": (95.0, 110.0)}
    assert compound_growth_process_passes(**boundary) is True
    # Fold below 0.95 fails
    weak = passing | {"chosen_test_tw": (94.0, 110.0)}
    assert compound_growth_process_passes(**weak) is False
