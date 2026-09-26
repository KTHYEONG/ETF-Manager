"""Invariant guards for the ISA household simulator."""

from __future__ import annotations

import calendar
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from src.sim.isa_household import (
    HouseholdPlan,
    HouseholdProfile,
    IncomePhase,
    IsaArm,
    IsaOperatingMode,
    expand_pension_profile,
    quote_threshold_annuity_exit,
    revalue_household,
    scenario_growth,
    simulate_household,
)
from src.sim.isa_tax import IsaTaxClass, quote_isa_closure
from src.sim.pension_monthly import MonthlyReturnPanel, WeightSchedule
from src.sim.pension_tax import load_pension_tax_regime
from src.sim.isa_tax import load_isa_tax_regime
from src.sim.tax import load_tax_regime

_ISA_PATH = Path("configs/tax/kr_isa_2026.json")
_PENSION_PATH = Path("configs/tax/kr_pension_2026.json")
_OVERSEAS_PATH = Path("configs/tax/kr_overseas_equity.json")


def _regimes():
    return (
        load_isa_tax_regime(_ISA_PATH),
        load_pension_tax_regime(_PENSION_PATH),
        load_tax_regime(_OVERSEAS_PATH),
    )


def _schedule() -> WeightSchedule:
    weights = {"QQQ": 0.9, "SCHD": 0.1}
    return WeightSchedule(start_weights=dict(weights), end_weights=dict(weights), glide_years=0)


def _phase(
    *, offset: int = 0, income: int = 45_000_000, national: int = 10_000_000, local: int = 10_000_000
) -> IncomePhase:
    return IncomePhase(
        first_plan_year_offset=offset,
        income_kind="wage",
        annual_income_krw=income,
        remaining_national_tax_krw=national,
        remaining_local_tax_krw=local,
    )


def _profile_zero(*, initial: IsaTaxClass = IsaTaxClass.GENERAL) -> HouseholdProfile:
    return HouseholdProfile(
        profile_id="zero",
        birth_date=date(1990, 1, 1),
        pension_account_open_date=date(2020, 1, 1),
        initial_isa_tax_class=initial,
        phases=(_phase(income=30_000_000, national=0, local=0),),
    )


def _profile_ample(*, income: int = 45_000_000, initial: IsaTaxClass = IsaTaxClass.GENERAL) -> HouseholdProfile:
    return HouseholdProfile(
        profile_id="ample",
        birth_date=date(1990, 1, 1),
        pension_account_open_date=date(2020, 1, 1),
        initial_isa_tax_class=initial,
        phases=(_phase(income=income),),
    )


def _plan(*, horizon: int = 6, budget: int = 20_000_000, pension: int = 6_000_000, drawing: int = 10) -> HouseholdPlan:
    return HouseholdPlan(
        plan_start_year=2027,
        horizon_years=horizon,
        pension_annual_krw=pension,
        isa_annual_budget_krw=budget,
        annuity_drawing_years=drawing,
    )


def _ones(scenarios: int, horizon: int) -> np.ndarray:
    return np.ones((scenarios, horizon * 12, 2), dtype=np.float64)


def _arms(cycle: int | None = 3) -> list[IsaArm]:
    return [
        IsaArm(arm_id="no_isa", mode=IsaOperatingMode.NO_ISA, cycle_years=None),
        IsaArm(arm_id="hold", mode=IsaOperatingMode.HOLD, cycle_years=None),
        IsaArm(arm_id="side", mode=IsaOperatingMode.ROLL_TO_SIDE, cycle_years=cycle),
        IsaArm(arm_id="cap", mode=IsaOperatingMode.ROLL_TO_PENSION_CREDIT_CAP, cycle_years=cycle),
        IsaArm(arm_id="all", mode=IsaOperatingMode.ROLL_TO_PENSION_ALL, cycle_years=cycle),
    ]


def _run(arm: IsaArm, plan: HouseholdPlan, profile: HouseholdProfile, growth: np.ndarray):
    isa_regime, pension_regime, overseas_regime = _regimes()
    return simulate_household(
        growth,
        _schedule(),
        arm,
        plan,
        profile,
        isa_regime=isa_regime,
        pension_regime=pension_regime,
        overseas_regime=overseas_regime,
    )


def test_zero_return_conserves_cash() -> None:
    """Growth of one with no tax capacity preserves every won of external cash."""
    plan = _plan(horizon=6)
    growth = _ones(2, 6)
    for arm in _arms():
        outcome = _run(arm, plan, _profile_zero(), growth)
        assert outcome.contributed_krw == 6 * (20_000_000 + 6_000_000)
        np.testing.assert_allclose(outcome.isa_tax_krw, 0.0)
        np.testing.assert_allclose(outcome.side_tax_krw, 0.0)
        np.testing.assert_allclose(outcome.credits_krw, 0.0)
        np.testing.assert_allclose(outcome.pension_exit_tax_krw, 0.0)
        np.testing.assert_allclose(outcome.household_net_krw, float(outcome.contributed_krw))


def test_zero_return_credits_flow_to_household() -> None:
    """Ample capacity turns pension credits into household wealth with no side tax."""
    plan = _plan(horizon=6)
    outcome = _run(
        IsaArm(arm_id="hold", mode=IsaOperatingMode.HOLD, cycle_years=None), plan, _profile_ample(), _ones(2, 6)
    )
    np.testing.assert_allclose(outcome.isa_tax_krw, 0.0)
    np.testing.assert_allclose(outcome.side_tax_krw, 0.0)
    expected = float(outcome.contributed_krw) + outcome.credits_krw - outcome.pension_exit_tax_krw
    np.testing.assert_allclose(outcome.household_net_krw, expected)


def test_cash_identity_across_arms() -> None:
    """Every arm reports identical external cash equal to horizon times annual budgets."""
    rng = np.random.default_rng(7)
    growth = 1.0 + (rng.random((3, 4 * 12, 2)) - 0.45) * 0.02
    plan = _plan(horizon=4, budget=10_000_000, pension=5_000_000)
    contributeds = set()
    for arm in _arms():
        outcome = _run(arm, plan, _profile_ample(), growth)
        contributeds.add(outcome.contributed_krw)
        assert outcome.contributed_krw == 4 * (10_000_000 + 5_000_000)
    assert len(contributeds) == 1


def test_accounting_identity() -> None:
    """Household net splits exactly across ISA, side, pension, and pending refund."""
    rng = np.random.default_rng(11)
    growth = 1.0 + (rng.random((4, 5 * 12, 2)) - 0.45) * 0.03
    plan = _plan(horizon=5)
    outcome = _run(
        IsaArm(arm_id="all", mode=IsaOperatingMode.ROLL_TO_PENSION_ALL, cycle_years=3),
        plan,
        _profile_ample(),
        growth,
    )
    parts = outcome.isa_net_krw + outcome.side_net_krw + outcome.pension_net_krw + outcome.pending_refund_krw
    np.testing.assert_allclose(outcome.household_net_krw, parts, atol=1e-6 * outcome.contributed_krw)


def test_no_isa_never_touches_isa() -> None:
    """The no-ISA arm keeps every ISA field at zero."""
    rng = np.random.default_rng(3)
    growth = 1.0 + (rng.random((2, 4 * 12, 2)) - 0.45) * 0.02
    outcome = _run(
        IsaArm(arm_id="no_isa", mode=IsaOperatingMode.NO_ISA, cycle_years=None),
        _plan(horizon=4),
        _profile_ample(),
        growth,
    )
    np.testing.assert_allclose(outcome.isa_net_krw, 0.0)
    np.testing.assert_allclose(outcome.transfers_krw, 0.0)
    np.testing.assert_allclose(outcome.isa_tax_krw, 0.0)


def test_hold_respects_lifetime_cap() -> None:
    """An eight-year full budget fills the 100M lifetime cap with 60M overflowing."""
    plan = _plan(horizon=8)
    outcome = _run(
        IsaArm(arm_id="hold", mode=IsaOperatingMode.HOLD, cycle_years=None), plan, _profile_zero(), _ones(1, 8)
    )
    np.testing.assert_allclose(outcome.isa_net_krw, [100_000_000.0])
    np.testing.assert_allclose(outcome.side_net_krw, [60_000_000.0])


def test_roll_transfers_every_cycle() -> None:
    """Two maturities of 60M move to pension with 60M left in the final account."""
    plan = _plan(horizon=9)
    outcome = _run(
        IsaArm(arm_id="all", mode=IsaOperatingMode.ROLL_TO_PENSION_ALL, cycle_years=3),
        plan,
        _profile_zero(),
        _ones(1, 9),
    )
    np.testing.assert_allclose(outcome.transfers_krw, [120_000_000.0])
    np.testing.assert_allclose(outcome.isa_net_krw, [60_000_000.0])


def test_credit_cap_routing() -> None:
    """The credit-cap arm moves 30M to pension and leaves the other 30M in side."""
    plan = _plan(horizon=6)
    outcome = _run(
        IsaArm(arm_id="cap", mode=IsaOperatingMode.ROLL_TO_PENSION_CREDIT_CAP, cycle_years=3),
        plan,
        _profile_zero(),
        _ones(1, 6),
    )
    np.testing.assert_allclose(outcome.transfers_krw, [30_000_000.0])
    np.testing.assert_allclose(outcome.side_net_krw, [30_000_000.0])


def test_reopen_reclassifies_from_prior_year_income() -> None:
    """The reopened account uses the general allowance while HOLD keeps preferential."""
    isa_regime, _, _ = _regimes()
    growth = np.ones((1, 6 * 12, 2), dtype=np.float64)
    growth[:, 36:, :] = 1.02
    profile = HouseholdProfile(
        profile_id="reclass",
        birth_date=date(1990, 1, 1),
        pension_account_open_date=date(2020, 1, 1),
        initial_isa_tax_class=IsaTaxClass.PREFERENTIAL,
        phases=(_phase(income=60_000_000),),
    )
    plan = _plan(horizon=6)
    rolled = _run(IsaArm(arm_id="side", mode=IsaOperatingMode.ROLL_TO_SIDE, cycle_years=3), plan, profile, growth)
    held = _run(IsaArm(arm_id="hold", mode=IsaOperatingMode.HOLD, cycle_years=None), plan, profile, growth)
    roll_balance = int(rolled.isa_net_krw[0] + rolled.isa_tax_krw[0])
    expected_general = quote_isa_closure(
        isa_regime, tax_class=IsaTaxClass.GENERAL, balance_krw=roll_balance, principal_krw=60_000_000, account_years=3
    )
    assert rolled.isa_tax_krw[0] == pytest.approx(expected_general.national_tax_krw + expected_general.local_tax_krw)
    expected_preferential = quote_isa_closure(
        isa_regime,
        tax_class=IsaTaxClass.PREFERENTIAL,
        balance_krw=roll_balance,
        principal_krw=60_000_000,
        account_years=3,
    )
    assert rolled.isa_tax_krw[0] != pytest.approx(
        expected_preferential.national_tax_krw + expected_preferential.local_tax_krw
    )
    hold_balance = int(held.isa_net_krw[0] + held.isa_tax_krw[0])
    expected_hold = quote_isa_closure(
        isa_regime,
        tax_class=IsaTaxClass.PREFERENTIAL,
        balance_krw=hold_balance,
        principal_krw=100_000_000,
        account_years=6,
    )
    assert held.isa_tax_krw[0] == pytest.approx(expected_hold.national_tax_krw + expected_hold.local_tax_krw)


def test_roll_cycle_below_minimum_rejected() -> None:
    """A two-year roll cycle violates the three-year minimum contract term."""
    with pytest.raises(ValueError, match="minimum contract"):
        _run(
            IsaArm(arm_id="short", mode=IsaOperatingMode.ROLL_TO_SIDE, cycle_years=2),
            _plan(horizon=4),
            _profile_ample(),
            _ones(1, 4),
        )


def test_refund_timing() -> None:
    """Refunds for the first two tax years land in side; the last year stays pending."""
    plan = _plan(horizon=3)
    outcome = _run(
        IsaArm(arm_id="hold", mode=IsaOperatingMode.HOLD, cycle_years=None), plan, _profile_ample(), _ones(1, 3)
    )
    np.testing.assert_allclose(outcome.pending_refund_krw, [990_000.0])
    np.testing.assert_allclose(outcome.side_net_krw, [1_980_000.0])
    np.testing.assert_allclose(outcome.credits_krw, [2_970_000.0])


def test_threshold_annuity_splits_rates() -> None:
    """Taxable receipts split at the threshold with the age-band rate below it."""
    _, pension_regime, _ = _regimes()
    profile = expand_pension_profile(_profile_ample(), plan_start_year=2027, plan_years=3)
    quote = quote_threshold_annuity_exit(
        valuation_date=date(2030, 12, 31),
        balance_krw=60_000_000,
        uncredited_principal_krw=10_000_000,
        drawing_years=2,
        profile=profile,
        regime=pension_regime,
    )
    assert quote.tax_free_krw == 10_000_000
    assert quote.low_rate_taxable_krw == 30_000_000
    assert quote.high_rate_taxable_krw == 20_000_000
    assert quote.national_tax_krw == 1_500_000 + 3_000_000
    assert quote.net_krw == 60_000_000 - quote.national_tax_krw - quote.local_tax_krw


def test_longer_drawing_never_lowers_net() -> None:
    """Spreading taxable receipts over more threshold-sized years cannot hurt."""
    rng = np.random.default_rng(5)
    growth = 1.0 + (rng.random((2, 6 * 12, 2)) - 0.42) * 0.03
    plan = _plan(horizon=6)
    outcome = _run(IsaArm(arm_id="hold", mode=IsaOperatingMode.HOLD, cycle_years=None), plan, _profile_ample(), growth)
    _, pension_regime, _ = _regimes()
    expanded = expand_pension_profile(_profile_ample(), plan_start_year=2027, plan_years=6)
    nets = [
        revalue_household(outcome, drawing_years=years, profile=expanded, regime=pension_regime)
        for years in (10, 20, 30)
    ]
    assert bool((nets[1] >= nets[0] - 1e-6).all())
    assert bool((nets[2] >= nets[1] - 1e-6).all())


def test_growth_shape_validated() -> None:
    """A mismatched month axis or a non-positive factor fails closed."""
    plan = _plan(horizon=4)
    profile = _profile_ample()
    arm = IsaArm(arm_id="hold", mode=IsaOperatingMode.HOLD, cycle_years=None)
    with pytest.raises(ValueError, match="month axis"):
        _run(arm, plan, profile, _ones(1, 5))
    bad = _ones(1, 4)
    bad[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        _run(arm, plan, profile, bad)


def test_scenario_windows() -> None:
    """Rolling windows stack every 12-month step without extending past the panel."""
    months = tuple(
        date(
            2020 + (m - 1) // 12,
            (m - 1) % 12 + 1,
            calendar.monthrange(2020 + (m - 1) // 12, (m - 1) % 12 + 1)[1],
        )
        for m in range(1, 37)
    )
    panel = MonthlyReturnPanel(
        tier="modern",
        months=months,
        returns={"QQQ": tuple([0.01] * 36), "SCHD": tuple([0.02] * 36)},
    )
    growth = scenario_growth([panel], _schedule(), horizon_years=2, step_months=12, annual_drag={})
    assert growth.shape == (2, 24, 2)
    np.testing.assert_allclose(growth[1, 0, :], [1.01, 1.02])


def test_profile_expansion_covers_prior_year() -> None:
    """Expansion spans one year before the plan through one year after its end."""
    profile = HouseholdProfile(
        profile_id="phased",
        birth_date=date(1990, 1, 1),
        pension_account_open_date=date(2020, 1, 1),
        initial_isa_tax_class=IsaTaxClass.GENERAL,
        phases=(_phase(offset=0, income=30_000_000), _phase(offset=2, income=60_000_000)),
    )
    expanded = expand_pension_profile(profile, plan_start_year=2027, plan_years=5)
    assert sorted(expanded.annual_income_krw) == list(range(2026, 2033))
    for year in (2026, 2027, 2028):
        assert expanded.annual_income_krw[year] == 30_000_000
    for year in (2029, 2030, 2031, 2032):
        assert expanded.annual_income_krw[year] == 60_000_000


def test_deterministic() -> None:
    """Identical inputs produce bit-identical arrays."""
    rng = np.random.default_rng(9)
    growth = 1.0 + (rng.random((2, 4 * 12, 2)) - 0.45) * 0.02
    plan = _plan(horizon=4)
    profile = _profile_ample()
    arm = IsaArm(arm_id="all", mode=IsaOperatingMode.ROLL_TO_PENSION_ALL, cycle_years=3)
    first = _run(arm, plan, profile, growth)
    second = _run(arm, plan, profile, growth)
    for field in (
        "household_net_krw",
        "isa_net_krw",
        "side_net_krw",
        "pension_net_krw",
        "pending_refund_krw",
        "pension_balance_krw",
        "pension_uncredited_krw",
        "pension_lump_sum_net_krw",
        "isa_tax_krw",
        "side_tax_krw",
        "pension_exit_tax_krw",
        "credits_krw",
        "transfers_krw",
    ):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))


def test_inputs_fail_closed() -> None:
    """Blank ids, bad cycles, phases, plans, drags, and growth all fail closed."""
    with pytest.raises(ValueError, match="arm_id"):
        IsaArm(arm_id="", mode=IsaOperatingMode.HOLD, cycle_years=None)
    with pytest.raises(ValueError, match="operating mode"):
        IsaArm(arm_id="x", mode="hold", cycle_years=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cycle_years"):
        IsaArm(arm_id="x", mode=IsaOperatingMode.ROLL_TO_SIDE, cycle_years=None)
    with pytest.raises(ValueError, match="cycle_years"):
        IsaArm(arm_id="x", mode=IsaOperatingMode.ROLL_TO_SIDE, cycle_years=0)
    with pytest.raises(ValueError, match="cycle_years"):
        IsaArm(arm_id="x", mode=IsaOperatingMode.HOLD, cycle_years=3)
    with pytest.raises(ValueError, match="offset"):
        IncomePhase(
            first_plan_year_offset=-1,
            income_kind="wage",
            annual_income_krw=1,
            remaining_national_tax_krw=0,
            remaining_local_tax_krw=0,
        )
    with pytest.raises(ValueError, match="integer"):
        IncomePhase(
            first_plan_year_offset="0",  # type: ignore[arg-type]
            income_kind="wage",
            annual_income_krw=1,
            remaining_national_tax_krw=0,
            remaining_local_tax_krw=0,
        )
    with pytest.raises(ValueError, match="income_kind"):
        IncomePhase(
            first_plan_year_offset=0,
            income_kind="business",  # type: ignore[arg-type]
            annual_income_krw=1,
            remaining_national_tax_krw=0,
            remaining_local_tax_krw=0,
        )
    with pytest.raises(ValueError, match="integer amount"):
        IncomePhase(
            first_plan_year_offset=0,
            income_kind="wage",
            annual_income_krw=1.5,  # type: ignore[arg-type]
            remaining_national_tax_krw=0,
            remaining_local_tax_krw=0,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        IncomePhase(
            first_plan_year_offset=0,
            income_kind="wage",
            annual_income_krw=-1,
            remaining_national_tax_krw=0,
            remaining_local_tax_krw=0,
        )
    with pytest.raises(ValueError, match="profile_id"):
        HouseholdProfile(
            profile_id="",
            birth_date=date(1990, 1, 1),
            pension_account_open_date=date(2020, 1, 1),
            initial_isa_tax_class=IsaTaxClass.GENERAL,
            phases=(_phase(),),
        )
    with pytest.raises(ValueError, match="dates"):
        HouseholdProfile(
            profile_id="p",
            birth_date="1990-01-01",  # type: ignore[arg-type]
            pension_account_open_date=date(2020, 1, 1),
            initial_isa_tax_class=IsaTaxClass.GENERAL,
            phases=(_phase(),),
        )
    with pytest.raises(ValueError, match="tax class"):
        HouseholdProfile(
            profile_id="p",
            birth_date=date(1990, 1, 1),
            pension_account_open_date=date(2020, 1, 1),
            initial_isa_tax_class="general",  # type: ignore[arg-type]
            phases=(_phase(),),
        )
    with pytest.raises(ValueError, match="non-empty"):
        HouseholdProfile(
            profile_id="p",
            birth_date=date(1990, 1, 1),
            pension_account_open_date=date(2020, 1, 1),
            initial_isa_tax_class=IsaTaxClass.GENERAL,
            phases=(),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        HouseholdProfile(
            profile_id="p",
            birth_date=date(1990, 1, 1),
            pension_account_open_date=date(2020, 1, 1),
            initial_isa_tax_class=IsaTaxClass.GENERAL,
            phases=(_phase(offset=0), _phase(offset=0)),
        )
    with pytest.raises(ValueError, match="offset must be 0"):
        HouseholdProfile(
            profile_id="p",
            birth_date=date(1990, 1, 1),
            pension_account_open_date=date(2020, 1, 1),
            initial_isa_tax_class=IsaTaxClass.GENERAL,
            phases=(_phase(offset=1),),
        )
    with pytest.raises(ValueError, match="positive integer"):
        HouseholdPlan(
            plan_start_year=2027,
            horizon_years=0,
            pension_annual_krw=1,
            isa_annual_budget_krw=1,
            annuity_drawing_years=1,
        )
    with pytest.raises(ValueError, match="positive integer"):
        HouseholdPlan(
            plan_start_year=True,  # type: ignore[arg-type]
            horizon_years=1,
            pension_annual_krw=1,
            isa_annual_budget_krw=1,
            annuity_drawing_years=1,
        )
    with pytest.raises(ValueError, match="plan_years"):
        expand_pension_profile(_profile_ample(), plan_start_year=2027, plan_years=0)
    with pytest.raises(ValueError, match="plan_years"):
        expand_pension_profile(_profile_ample(), plan_start_year=2027, plan_years="5")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="plan_start_year"):
        expand_pension_profile(_profile_ample(), plan_start_year="2027", plan_years=5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="horizon_years"):
        scenario_growth(
            [MonthlyReturnPanel(tier="m", months=(date(2020, 1, 31),), returns={"QQQ": (0.01,), "SCHD": (0.01,)})],
            _schedule(),
            horizon_years=0,
            step_months=1,
            annual_drag={},
        )
    with pytest.raises(ValueError, match="step_months"):
        scenario_growth(
            [MonthlyReturnPanel(tier="m", months=(date(2020, 1, 31),), returns={"QQQ": (0.01,), "SCHD": (0.01,)})],
            _schedule(),
            horizon_years=1,
            step_months=0,
            annual_drag={},
        )
    with pytest.raises(ValueError, match="numeric"):
        scenario_growth(
            [MonthlyReturnPanel(tier="m", months=(date(2020, 1, 31),), returns={"QQQ": (0.01,), "SCHD": (0.01,)})],
            _schedule(),
            horizon_years=1,
            step_months=1,
            annual_drag={"QQQ": "0.01"},  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="lacks schedule sleeves"):
        scenario_growth(
            [MonthlyReturnPanel(tier="m", months=(date(2020, 1, 31),), returns={"QQQ": (0.01,)})],
            _schedule(),
            horizon_years=1,
            step_months=1,
            annual_drag={},
        )
    with pytest.raises(ValueError, match="no rolling window"):
        scenario_growth(
            [MonthlyReturnPanel(tier="m", months=(date(2020, 1, 31),), returns={"QQQ": (0.01,), "SCHD": (0.01,)})],
            _schedule(),
            horizon_years=2,
            step_months=1,
            annual_drag={},
        )
    with pytest.raises(ValueError, match="outside the schedule"):
        scenario_growth(
            MonthlyReturnPanel(
                tier="modern",
                months=(date(2020, 1, 31),),
                returns={"QQQ": (0.01,), "SCHD": (0.01,)},
            ),
            _schedule(),
            horizon_years=1,
            step_months=1,
            annual_drag={"UNKNOWN": 0.01},
        )
    with pytest.raises(ValueError, match="non-negative"):
        scenario_growth(
            MonthlyReturnPanel(
                tier="modern",
                months=(date(2020, 1, 31),),
                returns={"QQQ": (0.01,), "SCHD": (0.01,)},
            ),
            _schedule(),
            horizon_years=1,
            step_months=1,
            annual_drag={"QQQ": -0.01},
        )
    with pytest.raises(ValueError, match="drawing_years"):
        quote_threshold_annuity_exit(
            valuation_date=date(2030, 12, 31),
            balance_krw=1,
            uncredited_principal_krw=0,
            drawing_years=0,
            profile=expand_pension_profile(_profile_ample(), plan_start_year=2027, plan_years=1),
            regime=load_pension_tax_regime(_PENSION_PATH),
        )
    with pytest.raises(ValueError, match="integer amount"):
        quote_threshold_annuity_exit(
            valuation_date=date(2030, 12, 31),
            balance_krw=1.5,  # type: ignore[arg-type]
            uncredited_principal_krw=0,
            drawing_years=1,
            profile=expand_pension_profile(_profile_ample(), plan_start_year=2027, plan_years=1),
            regime=load_pension_tax_regime(_PENSION_PATH),
        )
    with pytest.raises(ValueError, match="integer amount"):
        quote_threshold_annuity_exit(
            valuation_date=date(2030, 12, 31),
            balance_krw=1,
            uncredited_principal_krw="0",  # type: ignore[arg-type]
            drawing_years=1,
            profile=expand_pension_profile(_profile_ample(), plan_start_year=2027, plan_years=1),
            regime=load_pension_tax_regime(_PENSION_PATH),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        quote_threshold_annuity_exit(
            valuation_date=date(2030, 12, 31),
            balance_krw=-1,
            uncredited_principal_krw=0,
            drawing_years=1,
            profile=expand_pension_profile(_profile_ample(), plan_start_year=2027, plan_years=1),
            regime=load_pension_tax_regime(_PENSION_PATH),
        )
    with pytest.raises(ValueError, match="sleeve axis"):
        _run(
            IsaArm(arm_id="hold", mode=IsaOperatingMode.HOLD, cycle_years=None),
            _plan(horizon=2),
            _profile_ample(),
            np.ones((1, 2 * 12, 3)),
        )
    with pytest.raises(ValueError, match="3-D"):
        _run(
            IsaArm(arm_id="hold", mode=IsaOperatingMode.HOLD, cycle_years=None),
            _plan(horizon=2),
            _profile_ample(),
            np.ones((2 * 12, 2)),
        )
    with pytest.raises(ValueError, match="ISA budget"):
        _run(
            IsaArm(arm_id="hold", mode=IsaOperatingMode.HOLD, cycle_years=None),
            _plan(horizon=2, budget=20_000_001),
            _profile_ample(),
            _ones(1, 2),
        )
    with pytest.raises(ValueError, match="pension contribution"):
        _run(
            IsaArm(arm_id="hold", mode=IsaOperatingMode.HOLD, cycle_years=None),
            _plan(horizon=2, pension=18_000_001),
            _profile_ample(),
            _ones(1, 2),
        )


def test_threshold_age_counts_birthday() -> None:
    """A valuation before the birthday in that year uses the younger age band."""
    _, pension_regime, _ = _regimes()
    late_birth = HouseholdProfile(
        profile_id="late",
        birth_date=date(1960, 12, 15),
        pension_account_open_date=date(2020, 1, 1),
        initial_isa_tax_class=IsaTaxClass.GENERAL,
        phases=(_phase(),),
    )
    expanded = expand_pension_profile(late_birth, plan_start_year=2027, plan_years=3)
    quote = quote_threshold_annuity_exit(
        valuation_date=date(2030, 6, 30),
        balance_krw=20_000_000,
        uncredited_principal_krw=0,
        drawing_years=10,
        profile=expanded,
        regime=pension_regime,
    )
    assert quote.low_rate_taxable_krw == 20_000_000
    assert quote.national_tax_krw == 1_000_000


def test_short_panel_skipped_when_other_fits() -> None:
    """A panel shorter than the horizon is skipped while a fitting panel supplies windows."""
    short = MonthlyReturnPanel(tier="short", months=(date(2020, 1, 31),), returns={"QQQ": (0.01,), "SCHD": (0.02,)})
    months = tuple(
        date(2020 + (m - 1) // 12, (m - 1) % 12 + 1, calendar.monthrange(2020 + (m - 1) // 12, (m - 1) % 12 + 1)[1])
        for m in range(1, 25)
    )
    long = MonthlyReturnPanel(
        tier="long", months=months, returns={"QQQ": tuple([0.01] * 24), "SCHD": tuple([0.02] * 24)}
    )
    growth = scenario_growth([short, long], _schedule(), horizon_years=2, step_months=24, annual_drag={})
    assert growth.shape == (1, 24, 2)
