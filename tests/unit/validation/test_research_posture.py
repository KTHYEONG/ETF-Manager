"""Generated research posture tests."""

from __future__ import annotations

def test_select_chosen_test_arm_reuses_baseline_on_reject() -> None:
    from datetime import date

    from src.policy.adaptive_contribution import FROZEN_ADAPTIVE_V5
    from src.policy.targets import PolicyId
    from src.sim.allocation import AllocationConfig, AllocationResult
    from src.validation.research_posture import select_chosen_test_arm

    def _arm(*, adaptive: bool, wealth: float) -> AllocationResult:
        return AllocationResult(
            config=AllocationConfig(
                policy=PolicyId.QQQ,
                start=date(2020, 1, 2),
                end=date(2020, 6, 30),
                monthly_contribution_krw=1_000_000.0,
                adaptive_contribution=FROZEN_ADAPTIVE_V5 if adaptive else None,
                targets_override={"QQQ": 0.9, "SOXX": 0.1},
            ),
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.10 if adaptive else 0.05,
            total_contribution_real_krw=95.0 if adaptive else 90.0,
        )

    baseline = _arm(adaptive=True, wealth=90.0)
    candidate = _arm(adaptive=True, wealth=80.0)
    chosen = select_chosen_test_arm(
        train_adopted=False,
        candidate_test_arm=candidate,
        baseline_test_arm=baseline,
    )
    assert chosen is baseline
    assert chosen is not candidate
    assert chosen.config.adaptive_contribution is FROZEN_ADAPTIVE_V5
    assert chosen.terminal_wealth_real_krw == 90.0
    assert chosen.xirr_real == 0.10
    assert chosen.total_contribution_real_krw == 95.0


def test_select_chosen_test_arm_reuses_candidate_on_adopt() -> None:
    from datetime import date

    from src.policy.targets import PolicyId
    from src.sim.allocation import AllocationConfig, AllocationResult
    from src.validation.research_posture import select_chosen_test_arm

    def _arm(wealth: float) -> AllocationResult:
        return AllocationResult(
            config=AllocationConfig(
                policy=PolicyId.QQQ,
                start=date(2020, 1, 2),
                end=date(2020, 6, 30),
                monthly_contribution_krw=1_000_000.0,
            ),
            snapshots=(),
            terminal_wealth_krw=wealth,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=wealth,
            xirr_real=0.12,
            total_contribution_real_krw=90.0,
        )

    baseline = _arm(100.0)
    candidate = _arm(120.0)
    chosen = select_chosen_test_arm(
        train_adopted=True,
        candidate_test_arm=candidate,
        baseline_test_arm=baseline,
    )
    assert chosen is candidate
    assert chosen is not baseline
    assert chosen.terminal_wealth_real_krw == 120.0


def test_economic_effect_passes_rejects_sub_hurdle_median() -> None:
    import pytest

    from src.validation.research_posture import (
        ECONOMIC_CE_GAMMA10_FLOOR,
        ECONOMIC_MEDIAN_RATIO_FLOOR,
        economic_effect_passes,
    )

    assert pytest.approx(1.01) == ECONOMIC_MEDIAN_RATIO_FLOOR
    assert pytest.approx(1.0) == ECONOMIC_CE_GAMMA10_FLOOR
    assert (
        economic_effect_passes(median_ratio=1.00074, ce_gamma_10=0.99913, bootstrap_ok=True)
        is False
    )
    assert (
        economic_effect_passes(median_ratio=1.00074, ce_gamma_10=1.0, bootstrap_ok=True)
        is False
    )
    assert (
        economic_effect_passes(median_ratio=1.0199, ce_gamma_10=1.0024, bootstrap_ok=False)
        is False
    )


def test_economic_effect_passes_accepts_soxx10_scale() -> None:
    from src.validation.research_posture import economic_effect_passes

    assert (
        economic_effect_passes(median_ratio=1.0199, ce_gamma_10=1.0024, bootstrap_ok=True)
        is True
    )
    assert (
        economic_effect_passes(median_ratio=1.01, ce_gamma_10=1.0, bootstrap_ok=True)
        is True
    )
    assert (
        economic_effect_passes(median_ratio=1.0070, ce_gamma_10=1.0007, bootstrap_ok=True)
        is False
    )


def test_seen_history_cutoff_splits_prospective_oos() -> None:
    from datetime import date

    import pytest

    from src.validation.research_posture import (
        SEEN_HISTORY_CUTOFF,
        assert_prospective_observation,
        is_seen_history,
        observation_epoch,
    )

    assert date(2026, 8, 28) == SEEN_HISTORY_CUTOFF
    assert is_seen_history(date(2026, 8, 28)) is True
    assert is_seen_history(date(2026, 8, 27)) is True
    assert is_seen_history(date(2026, 8, 29)) is False
    assert observation_epoch(date(2026, 8, 28)) == "seen_history"
    assert observation_epoch(date(2026, 9, 1)) == "prospective_oos"
    with pytest.raises(ValueError, match="seen_history"):
        assert_prospective_observation(date(2026, 8, 28))
    assert_prospective_observation(date(2026, 9, 1))


def test_classify_strategy_role_catalog_and_forbid_new_weights() -> None:
    import pytest

    from src.validation.research_posture import StrategyRole, classify_strategy_role

    assert (
        classify_strategy_role(targets={"QQQ": 1.0}, adaptive=False)
        is StrategyRole.IMMUTABLE_BENCHMARK
    )
    assert (
        classify_strategy_role(targets={"QQQ": 0.9, "SOXX": 0.1}, adaptive=False)
        is StrategyRole.PROVISIONAL_INCUMBENT
    )
    assert (
        classify_strategy_role(targets={"QQQ": 0.85, "SOXX": 0.15}, adaptive=False)
        is StrategyRole.AGGRESSIVE_CHALLENGER
    )
    assert (
        classify_strategy_role(targets={"QQQ": 0.95, "SOXX": 0.05}, adaptive=False)
        is StrategyRole.CONSERVATIVE_CHALLENGER
    )
    assert (
        classify_strategy_role(targets={"QQQ": 0.9, "SOXX": 0.1}, adaptive=True)
        is StrategyRole.FROZEN_RESEARCH
    )
    assert (
        classify_strategy_role(targets={"QQQ": 0.9, "ROBO": 0.1}, adaptive=False)
        is StrategyRole.REJECTED_VEHICLE
    )
    assert (
        classify_strategy_role(targets={"QQQ": 0.9, "PAVE": 0.1}, adaptive=False)
        is StrategyRole.PROSPECTIVE_WATCH
    )
    with pytest.raises(ValueError, match="unregistered mix"):
        classify_strategy_role(targets={"QQQ": 0.8, "SOXX": 0.2}, adaptive=False)


def test_objective_family_capital_allocation_forbids_adaptive() -> None:
    import pytest

    from src.validation.research_posture import ObjectiveFamily, assert_objective_family_invariants

    assert_objective_family_invariants(
        family=ObjectiveFamily.CAPITAL_ALLOCATION,
        adaptive_contribution_set=False,
        baseline_adaptive_set=False,
        kafi_deployment_set=False,
        reserve_set=False,
    )
    with pytest.raises(ValueError, match="adaptive_contribution"):
        assert_objective_family_invariants(
            family=ObjectiveFamily.CAPITAL_ALLOCATION,
            adaptive_contribution_set=True,
            baseline_adaptive_set=False,
            kafi_deployment_set=False,
            reserve_set=False,
        )
    with pytest.raises(ValueError, match="adaptive_contribution"):
        assert_objective_family_invariants(
            family=ObjectiveFamily.DEPLOYMENT_TIMING,
            adaptive_contribution_set=True,
            baseline_adaptive_set=False,
            kafi_deployment_set=True,
            reserve_set=False,
        )
    with pytest.raises(ValueError, match="kafi_deployment|reserve"):  # noqa: RUF043
        assert_objective_family_invariants(
            family=ObjectiveFamily.DEPLOYMENT_TIMING,
            adaptive_contribution_set=False,
            baseline_adaptive_set=False,
            kafi_deployment_set=False,
            reserve_set=False,
        )
    assert_objective_family_invariants(
        family=ObjectiveFamily.DEPLOYMENT_TIMING,
        adaptive_contribution_set=False,
        baseline_adaptive_set=False,
        kafi_deployment_set=True,
        reserve_set=False,
    )




def test_assert_objective_family_capital_allocation_forbids_timing_modules() -> None:
    import pytest

    from src.validation.research_posture import ObjectiveFamily, assert_objective_family_invariants

    with pytest.raises(ValueError, match="capital_allocation"):
        assert_objective_family_invariants(
            family=ObjectiveFamily.CAPITAL_ALLOCATION,
            adaptive_contribution_set=False,
            baseline_adaptive_set=False,
            kafi_deployment_set=True,
            reserve_set=False,
            contribution_shape_set=False,
        )
    with pytest.raises(ValueError, match="capital_allocation"):
        assert_objective_family_invariants(
            family=ObjectiveFamily.CAPITAL_ALLOCATION,
            adaptive_contribution_set=False,
            baseline_adaptive_set=False,
            kafi_deployment_set=False,
            reserve_set=True,
            contribution_shape_set=False,
        )
    with pytest.raises(ValueError, match="capital_allocation"):
        assert_objective_family_invariants(
            family=ObjectiveFamily.CAPITAL_ALLOCATION,
            adaptive_contribution_set=False,
            baseline_adaptive_set=False,
            kafi_deployment_set=False,
            reserve_set=False,
            contribution_shape_set=True,
        )

