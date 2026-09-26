"""ISA + pension-savings household simulator: pre-registered ISA operating modes, statutory cash routing, and after-tax valuation."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_FLOOR, Decimal
from enum import StrEnum
from itertools import pairwise
from typing import Final, Literal

import numpy as np

from src.sim.isa_tax import (
    IsaTaxClass,
    IsaTaxRegime,
    classify_isa_tax_class,
    credit_maximizing_transfer_krw,
    isa_contribution_room_krw,
    quote_isa_closure,
    quote_isa_transfer_credit,
)
from src.sim.pension_monthly import MonthlyReturnPanel, WeightSchedule
from src.sim.pension_tax import (
    PensionExitKind,
    PensionTaxProfile,
    PensionTaxRegime,
    quote_annual_pension_credit,
    quote_pension_exit,
)
from src.sim.tax import KrOverseasTaxRegime, annual_capital_gains_tax_krw

logger = logging.getLogger(__name__)

__all__ = [
    "HouseholdOutcome",
    "HouseholdPlan",
    "HouseholdProfile",
    "IncomePhase",
    "IsaArm",
    "IsaOperatingMode",
    "ThresholdAnnuityExitQuote",
    "expand_pension_profile",
    "quote_threshold_annuity_exit",
    "revalue_household",
    "scenario_growth",
    "simulate_household",
]

_MONTHS_PER_YEAR: Final[int] = 12


class IsaOperatingMode(StrEnum):
    """How the ISA channel's annual budget and matured balances are routed."""

    NO_ISA = "no_isa"
    HOLD = "hold"
    ROLL_TO_SIDE = "roll_to_side"
    ROLL_TO_PENSION_CREDIT_CAP = "roll_to_pension_credit_cap"
    ROLL_TO_PENSION_ALL = "roll_to_pension_all"


_ROLL_MODES: Final[frozenset[IsaOperatingMode]] = frozenset(
    {
        IsaOperatingMode.ROLL_TO_SIDE,
        IsaOperatingMode.ROLL_TO_PENSION_CREDIT_CAP,
        IsaOperatingMode.ROLL_TO_PENSION_ALL,
    }
)


@dataclass(frozen=True, slots=True)
class IsaArm:
    """One pre-registered operating policy.

    HOLD keeps the first account open to the valuation date, so its allowance class stays the one
    fixed at opening. Roll modes terminate the account every ``cycle_years`` and immediately open a
    new one whose class is re-determined from the prior tax year's income.

    Raises: ValueError if ``arm_id`` is blank, a roll mode lacks a positive integer ``cycle_years``,
        or a non-roll mode carries one.
    """

    arm_id: str
    mode: IsaOperatingMode
    cycle_years: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.arm_id, str) or not self.arm_id:
            raise ValueError("isa arm_id must be a nonempty string")
        if not isinstance(self.mode, IsaOperatingMode):
            raise ValueError(f"unknown ISA operating mode {self.mode!r}")
        if self.mode in _ROLL_MODES:
            if isinstance(self.cycle_years, bool) or not isinstance(self.cycle_years, int):
                raise ValueError("roll arms require a positive integer cycle_years")
            if self.cycle_years <= 0:
                raise ValueError("roll arms require a positive integer cycle_years")
        elif self.cycle_years is not None:
            raise ValueError("non-roll arms must not carry cycle_years")


@dataclass(frozen=True, slots=True)
class IncomePhase:
    """Income and residual tax capacity that hold from ``first_plan_year_offset`` until the next phase."""

    first_plan_year_offset: int
    income_kind: Literal["wage", "comprehensive"]
    annual_income_krw: int
    remaining_national_tax_krw: int
    remaining_local_tax_krw: int

    def __post_init__(self) -> None:
        if isinstance(self.first_plan_year_offset, bool) or not isinstance(self.first_plan_year_offset, int):
            raise ValueError("first_plan_year_offset must be an integer")
        if self.first_plan_year_offset < 0:
            raise ValueError("first_plan_year_offset must be nonnegative")
        if self.income_kind not in ("wage", "comprehensive"):
            raise ValueError(f"income_kind {self.income_kind!r} is unsupported")
        for name in ("annual_income_krw", "remaining_national_tax_krw", "remaining_local_tax_krw"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer amount of KRW")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True, slots=True)
class HouseholdProfile:
    """Declared life-stage income path; nothing about income or tax capacity is inferred.

    Raises: ValueError if ``profile_id`` is blank, phases are empty, the first phase offset is not 0,
        offsets are not strictly increasing, or any amount is negative or non-integer.
    """

    profile_id: str
    birth_date: date
    pension_account_open_date: date
    initial_isa_tax_class: IsaTaxClass
    phases: tuple[IncomePhase, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("profile_id must be a nonempty string")
        if not isinstance(self.birth_date, date) or not isinstance(self.pension_account_open_date, date):
            raise ValueError("birth_date and pension_account_open_date must be dates")
        if not isinstance(self.initial_isa_tax_class, IsaTaxClass):
            raise ValueError(f"unknown ISA tax class {self.initial_isa_tax_class!r}")
        phases = tuple(self.phases)
        if not phases:
            raise ValueError("phases must be non-empty")
        if phases[0].first_plan_year_offset != 0:
            raise ValueError("the first phase offset must be 0")
        offsets = [phase.first_plan_year_offset for phase in phases]
        if any(later <= earlier for earlier, later in pairwise(offsets)):
            raise ValueError("phase offsets must be strictly increasing")
        object.__setattr__(self, "phases", phases)


@dataclass(frozen=True, slots=True)
class HouseholdPlan:
    """Cash plan shared by every arm so outcomes are compared under identical external cash.

    Raises: ValueError if any field is not a positive integer.
    """

    plan_start_year: int
    horizon_years: int
    pension_annual_krw: int
    isa_annual_budget_krw: int
    annuity_drawing_years: int

    def __post_init__(self) -> None:
        for name in (
            "plan_start_year",
            "horizon_years",
            "pension_annual_krw",
            "isa_annual_budget_krw",
            "annuity_drawing_years",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be a positive integer")
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ThresholdAnnuityExitQuote:
    """Pension balance valued as annuity receipts that respect the private-pension threshold."""

    balance_krw: int
    tax_free_krw: int
    low_rate_taxable_krw: int
    high_rate_taxable_krw: int
    national_tax_krw: int
    local_tax_krw: int
    net_krw: int


@dataclass(frozen=True, slots=True)
class HouseholdOutcome:
    """Per-scenario after-tax household result at the valuation date (arrays share one scenario axis)."""

    valuation_date: date
    contributed_krw: int
    household_net_krw: np.ndarray
    isa_net_krw: np.ndarray
    side_net_krw: np.ndarray
    pension_net_krw: np.ndarray
    pending_refund_krw: np.ndarray
    pension_balance_krw: np.ndarray
    pension_uncredited_krw: np.ndarray
    pension_lump_sum_net_krw: np.ndarray
    isa_tax_krw: np.ndarray
    side_tax_krw: np.ndarray
    pension_exit_tax_krw: np.ndarray
    credits_krw: np.ndarray
    transfers_krw: np.ndarray


def _phase_for_offset(profile: HouseholdProfile, offset: int) -> IncomePhase:
    active = profile.phases[0]
    for phase in profile.phases:
        if phase.first_plan_year_offset <= offset:
            active = phase
        else:
            break
    return active


def _full_years_since(start: date, end: date) -> int:
    years = end.year - start.year
    if (end.month, end.day) < (start.month, start.day):
        years -= 1
    return years


def _age_rate(regime: PensionTaxRegime, age: int) -> float:
    rate = regime.age_withholding_bands[0][1]
    for min_age, band_rate in regime.age_withholding_bands:
        if age >= min_age:
            rate = band_rate
    return rate


def _won_floor(amount: int, rate: float) -> int:
    quantized = Decimal(amount) * Decimal(str(rate))
    return int(quantized.to_integral_value(rounding=ROUND_FLOOR))


def expand_pension_profile(profile: HouseholdProfile, *, plan_start_year: int, plan_years: int) -> PensionTaxProfile:
    """Materialize explicit per-year income and capacity maps for the pension tax functions.

    Covers tax years ``plan_start_year - 1`` through ``plan_start_year + plan_years`` inclusive so the
    prior-year income needed for ISA re-classification and the last refund year always exist; the
    year before the plan takes the first phase. Other private-pension income is 0 in every year and
    no pension commencement date is set.

    Raises: ValueError if ``plan_years`` is below 1.
    """
    if isinstance(plan_years, bool) or not isinstance(plan_years, int):
        raise ValueError("plan_years must be a positive integer")
    if plan_years < 1:
        raise ValueError("plan_years must be at least 1")
    if isinstance(plan_start_year, bool) or not isinstance(plan_start_year, int):
        raise ValueError("plan_start_year must be an integer year")
    income: dict[int, int] = {}
    national: dict[int, int] = {}
    local: dict[int, int] = {}
    other: dict[int, int] = {}
    for year in range(plan_start_year - 1, plan_start_year + plan_years + 1):
        offset = year - plan_start_year
        phase = profile.phases[0] if offset < 0 else _phase_for_offset(profile, offset)
        income[year] = phase.annual_income_krw
        national[year] = phase.remaining_national_tax_krw
        local[year] = phase.remaining_local_tax_krw
        other[year] = 0
    return PensionTaxProfile(
        profile_id=profile.profile_id,
        birth_date=profile.birth_date,
        account_open_date=profile.pension_account_open_date,
        income_kind=profile.phases[0].income_kind,
        annual_income_krw=income,
        remaining_national_tax_krw=national,
        remaining_local_tax_krw=local,
        other_private_pension_income_krw=other,
    )


def scenario_growth(
    panels: Sequence[MonthlyReturnPanel],
    schedule: WeightSchedule,
    *,
    horizon_years: int,
    step_months: int,
    annual_drag: Mapping[str, float],
) -> np.ndarray:
    """Stack monthly net growth factors for every rolling window of every panel.

    Returns: Array of shape (scenarios, horizon_years * 12, sleeves) with sleeves in sorted order of
        ``schedule.start_weights``; windows start every ``step_months`` from each panel's first month
        and never extend past its last. A bootstrap path whose length equals the horizon contributes
        exactly one scenario.
    Raises: ValueError if a panel lacks a schedule sleeve, a drag names a sleeve outside the schedule,
        no window fits, or a drag is negative or non-finite.
    """
    if isinstance(horizon_years, bool) or not isinstance(horizon_years, int) or horizon_years < 1:
        raise ValueError("horizon_years must be a positive integer")
    if isinstance(step_months, bool) or not isinstance(step_months, int) or step_months < 1:
        raise ValueError("step_months must be a positive integer")
    sleeves = tuple(sorted(schedule.start_weights))
    for sleeve, drag in annual_drag.items():
        if sleeve not in sleeves:
            raise ValueError(f"annual_drag names a sleeve outside the schedule: {sleeve!r}")
        if isinstance(drag, bool) or not isinstance(drag, float | int):
            raise ValueError(f"annual_drag for {sleeve!r} must be numeric")
        value = float(drag)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"annual_drag for {sleeve!r} must be finite and non-negative")
    monthly_factor = np.array(
        [(1.0 - float(annual_drag.get(sleeve, 0.0))) ** (1.0 / _MONTHS_PER_YEAR) for sleeve in sleeves],
        dtype=np.float64,
    )
    horizon_months = horizon_years * _MONTHS_PER_YEAR
    windows: list[np.ndarray] = []
    for panel in panels:
        missing = sorted(set(sleeves) - set(panel.returns))
        if missing:
            raise ValueError(f"panel lacks schedule sleeves: {missing}")
        gross = np.array(
            [[float(panel.returns[sleeve][index]) for sleeve in sleeves] for index in range(len(panel.months))],
            dtype=np.float64,
        )
        net = (1.0 + gross) * monthly_factor.reshape(1, len(sleeves))
        start = 0
        fitted = False
        while start + horizon_months <= len(panel.months):
            windows.append(net[start : start + horizon_months])
            fitted = True
            start += step_months
        if not fitted and len(panel.months) < horizon_months:
            continue
    if not windows:
        raise ValueError("no rolling window fits the horizon")
    return np.stack(windows, axis=0)


def quote_threshold_annuity_exit(
    *,
    valuation_date: date,
    balance_krw: int,
    uncredited_principal_krw: int,
    drawing_years: int,
    profile: PensionTaxProfile,
    regime: PensionTaxRegime,
) -> ThresholdAnnuityExitQuote:
    """Value the whole pension as annuity receipts spread over ``drawing_years``.

    Uncredited principal leaves tax-free. Taxable receipts up to the private-pension threshold per
    drawing year bear the age-band rate; the remainder bears the above-threshold separate rate.
    Post-valuation returns are ignored, matching ANNUITY_LOW's convention.

    Raises: ValueError if ``drawing_years`` is below 1 or an amount is negative.
    """
    if isinstance(drawing_years, bool) or not isinstance(drawing_years, int) or drawing_years < 1:
        raise ValueError("drawing_years must be a positive integer")
    if isinstance(balance_krw, bool) or not isinstance(balance_krw, int):
        raise ValueError("balance_krw must be an integer amount of KRW")
    if isinstance(uncredited_principal_krw, bool) or not isinstance(uncredited_principal_krw, int):
        raise ValueError("uncredited_principal_krw must be an integer amount of KRW")
    if balance_krw < 0 or uncredited_principal_krw < 0:
        raise ValueError("pension exit amounts must be nonnegative")
    tax_free = min(balance_krw, uncredited_principal_krw)
    taxable = balance_krw - tax_free
    low = min(taxable, regime.private_pension_threshold_krw * drawing_years)
    high = taxable - low
    age = _full_years_since(profile.birth_date, valuation_date)
    rate = _age_rate(regime, age)
    national = _won_floor(low, rate) + _won_floor(high, regime.above_threshold_separate_rate)
    local = _won_floor(national, regime.local_surcharge_rate)
    return ThresholdAnnuityExitQuote(
        balance_krw=balance_krw,
        tax_free_krw=tax_free,
        low_rate_taxable_krw=low,
        high_rate_taxable_krw=high,
        national_tax_krw=national,
        local_tax_krw=local,
        net_krw=balance_krw - national - local,
    )


def revalue_household(
    outcome: HouseholdOutcome,
    *,
    drawing_years: int,
    profile: PensionTaxProfile,
    regime: PensionTaxRegime,
) -> np.ndarray:
    """Household net wealth with the pension re-valued under another drawing horizon; nothing else changes."""
    scenarios = int(outcome.household_net_krw.shape[0])
    revised = np.empty(scenarios, dtype=np.float64)
    for index in range(scenarios):
        balance = int(outcome.pension_balance_krw[index])
        uncredited = int(outcome.pension_uncredited_krw[index])
        quote = quote_threshold_annuity_exit(
            valuation_date=outcome.valuation_date,
            balance_krw=balance,
            uncredited_principal_krw=uncredited,
            drawing_years=drawing_years,
            profile=profile,
            regime=regime,
        )
        revised[index] = (
            float(outcome.isa_net_krw[index])
            + float(outcome.side_net_krw[index])
            + float(quote.net_krw)
            + float(outcome.pending_refund_krw[index])
        )
    return revised


def _weight_rows(schedule: WeightSchedule, horizon_years: int, sleeves: tuple[str, ...]) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    for year in range(horizon_years):
        weights = schedule.weights_for_year(year, horizon_years)
        rows.append(np.array([float(weights[sleeve]) for sleeve in sleeves], dtype=np.float64))
    return rows


def simulate_household(
    growth: np.ndarray,
    schedule: WeightSchedule,
    arm: IsaArm,
    plan: HouseholdPlan,
    profile: HouseholdProfile,
    *,
    isa_regime: IsaTaxRegime,
    pension_regime: PensionTaxRegime,
    overseas_regime: KrOverseasTaxRegime,
) -> HouseholdOutcome:
    """Run one arm for one household over every scenario and value it after tax.

    External cash each plan year is exactly ``pension_annual_krw + isa_annual_budget_krw`` for every
    arm; cash an arm cannot shelter goes to the side account, so arms differ only in where money sits
    and which taxes and credits it triggers. Credits refund in February of the following tax year into
    the side account; a refund not yet paid at valuation is counted as a receivable. At valuation every
    open ISA is closed under its allowance class (an account younger than the minimum term is valued as
    held to maturity with zero further return), the side account is liquidated in one tax year, and the
    pension is valued with ``quote_threshold_annuity_exit`` (primary) and LUMP_SUM (reported only).

    Raises: ValueError if ``growth`` is not 3-D, its month axis differs from ``horizon_years * 12``, its
        sleeve axis differs from the schedule, any factor is non-finite or not positive, a roll arm's
        ``cycle_years`` is below the ISA minimum contract term, the ISA budget exceeds the ISA annual
        limit, or ``pension_annual_krw`` exceeds the pension annual contribution limit.
    """
    scenarios_raw = int(growth.shape[0]) if isinstance(growth, np.ndarray) and growth.ndim == 3 else 0
    logger.debug(
        "[PORTFOLIO] event=isa_household_sim arm=%s profile=%s scenarios=%d",
        arm.arm_id,
        profile.profile_id,
        scenarios_raw,
    )
    if not isinstance(growth, np.ndarray) or growth.ndim != 3:
        raise ValueError("growth must be a 3-D array of monthly growth factors")
    horizon = plan.horizon_years
    expected_months = horizon * _MONTHS_PER_YEAR
    scenarios, months, sleeve_count = growth.shape
    if months != expected_months:
        raise ValueError(f"growth month axis {months} differs from horizon {expected_months}")
    sleeves = tuple(sorted(schedule.start_weights))
    if sleeve_count != len(sleeves):
        raise ValueError(f"growth sleeve axis {sleeve_count} differs from the schedule {len(sleeves)}")
    if not np.all(np.isfinite(growth)) or bool((growth <= 0).any()):
        raise ValueError("growth factors must be finite and positive")
    if arm.mode in _ROLL_MODES:
        assert arm.cycle_years is not None
        if arm.cycle_years < isa_regime.minimum_contract_years:
            raise ValueError("roll cycle_years is below the ISA minimum contract term")
    if plan.isa_annual_budget_krw > isa_regime.annual_contribution_limit_krw:
        raise ValueError("ISA budget exceeds the ISA annual limit")
    if plan.pension_annual_krw > pension_regime.annual_contribution_limit_krw:
        raise ValueError("pension contribution exceeds the pension annual contribution limit")

    pension_profile = expand_pension_profile(profile, plan_start_year=plan.plan_start_year, plan_years=horizon)
    ordinary = [
        quote_annual_pension_credit(plan.plan_start_year + k, plan.pension_annual_krw, pension_profile, pension_regime)
        for k in range(horizon)
    ]
    credit_cap_amount = credit_maximizing_transfer_krw(isa_regime)
    weight_rows = _weight_rows(schedule, horizon, sleeves)

    isa_hold = np.zeros((scenarios, len(sleeves)), dtype=np.float64)
    pension_hold = np.zeros((scenarios, len(sleeves)), dtype=np.float64)
    side_hold = np.zeros((scenarios, len(sleeves)), dtype=np.float64)
    side_basis = np.zeros(scenarios, dtype=np.float64)
    pension_uncredited = np.zeros(scenarios, dtype=np.int64)
    pension_credited_total = np.zeros(scenarios, dtype=np.int64)
    isa_tax_acc = np.zeros(scenarios, dtype=np.float64)
    side_tax_acc = np.zeros(scenarios, dtype=np.float64)
    exit_tax_acc = np.zeros(scenarios, dtype=np.float64)
    credits_acc = np.zeros(scenarios, dtype=np.float64)
    transfers_acc = np.zeros(scenarios, dtype=np.float64)
    pending = np.zeros(scenarios, dtype=np.float64)
    refund_schedule: dict[int, np.ndarray] = {}

    isa_open_k = 0
    isa_class = profile.initial_isa_tax_class
    isa_cum_principal = 0

    for month in range(months):
        year_index = month // _MONTHS_PER_YEAR
        if month % _MONTHS_PER_YEAR == 0:
            k = year_index
            tax_year = plan.plan_start_year + k
            weights = weight_rows[k]
            to_pension = np.zeros(scenarios, dtype=np.float64)
            to_side = np.zeros(scenarios, dtype=np.float64)
            transfer_national = np.zeros(scenarios, dtype=np.float64)
            transfer_local = np.zeros(scenarios, dtype=np.float64)
            transfer_credited = np.zeros(scenarios, dtype=np.int64)
            if arm.mode in _ROLL_MODES and k > 0 and (k - isa_open_k) == arm.cycle_years:
                assert arm.cycle_years is not None
                for index in range(scenarios):
                    balance = math.floor(float(isa_hold[index].sum()))
                    quote = quote_isa_closure(
                        isa_regime,
                        tax_class=isa_class,
                        balance_krw=balance,
                        principal_krw=isa_cum_principal,
                        account_years=arm.cycle_years,
                    )
                    isa_tax_acc[index] += float(quote.national_tax_krw + quote.local_tax_krw)
                    net = float(quote.net_proceeds_krw)
                    if arm.mode is IsaOperatingMode.ROLL_TO_SIDE:
                        to_side[index] = net
                    elif arm.mode is IsaOperatingMode.ROLL_TO_PENSION_CREDIT_CAP:
                        routed = min(net, float(credit_cap_amount))
                        to_pension[index] = routed
                        to_side[index] = net - routed
                    else:
                        to_pension[index] = net
                    transfers_acc[index] += float(to_pension[index])
                    if to_pension[index] > 0:
                        transfer_quote = quote_isa_transfer_credit(
                            tax_year,
                            int(to_pension[index]),
                            regular=ordinary[k],
                            profile=pension_profile,
                            pension_regime=pension_regime,
                            isa_regime=isa_regime,
                        )
                        transfer_national[index] = float(transfer_quote.national_credit_krw)
                        transfer_local[index] = float(transfer_quote.local_credit_krw)
                        transfer_credited[index] = int(transfer_quote.credited_principal_krw)
                isa_hold[:] = 0.0
                prior_year = tax_year - 1
                prior_offset = prior_year - plan.plan_start_year
                prior_phase = profile.phases[0] if prior_offset < 0 else _phase_for_offset(profile, prior_offset)
                isa_class = classify_isa_tax_class(
                    isa_regime,
                    income_kind=prior_phase.income_kind,
                    prior_year_income_krw=prior_phase.annual_income_krw,
                )
                isa_open_k = k
                isa_cum_principal = 0
            ordinary_quote = ordinary[k]
            ordinary_refund = float(ordinary_quote.national_credit_krw + ordinary_quote.local_credit_krw)
            credits_acc += ordinary_refund + transfer_national + transfer_local
            year_refund = np.full(scenarios, ordinary_refund, dtype=np.float64) + transfer_national + transfer_local
            refund_month = _MONTHS_PER_YEAR * (k + 1) + 1
            if refund_month >= months:
                pending += year_refund
            elif float(year_refund.sum()) != 0.0:
                prior = refund_schedule.get(refund_month)
                refund_schedule[refund_month] = year_refund if prior is None else prior + year_refund
            pension_credited_total += int(ordinary_quote.credited_principal_krw)
            pension_uncredited += np.full(
                scenarios, int(plan.pension_annual_krw - ordinary_quote.credited_principal_krw), dtype=np.int64
            )
            for index in range(scenarios):
                if transfer_credited[index] != 0 or to_pension[index] != 0:
                    pension_credited_total[index] += int(transfer_credited[index])
                    pension_uncredited[index] += int(int(to_pension[index]) - int(transfer_credited[index]))
            if arm.mode is IsaOperatingMode.NO_ISA:
                isa_contrib = 0
            else:
                room = isa_contribution_room_krw(
                    isa_regime, elapsed_full_years=k - isa_open_k, cumulative_contributed_krw=isa_cum_principal
                )
                isa_contrib = min(plan.isa_annual_budget_krw, room)
            side_remainder = float(plan.isa_annual_budget_krw - isa_contrib)
            isa_cum_principal += isa_contrib
            row = weights.reshape(1, -1)
            isa_hold = (isa_hold.sum(axis=1, keepdims=True) + float(isa_contrib)) * row
            pension_total = pension_hold.sum(axis=1) + float(plan.pension_annual_krw) + to_pension
            pension_hold = pension_total.reshape(-1, 1) * row
            deposit = side_remainder + to_side
            side_hold += deposit.reshape(-1, 1) * row
            side_basis += deposit
        due = refund_schedule.pop(month, None)
        if due is not None:
            current_weights = weight_rows[month // _MONTHS_PER_YEAR].reshape(1, -1)
            side_hold += due.reshape(-1, 1) * current_weights
            side_basis += due
        isa_hold *= growth[:, month, :]
        pension_hold *= growth[:, month, :]
        side_hold *= growth[:, month, :]

    valuation_date = date(plan.plan_start_year + horizon - 1, _MONTHS_PER_YEAR, 31)
    isa_net = np.zeros(scenarios, dtype=np.float64)
    side_net = np.zeros(scenarios, dtype=np.float64)
    pension_net = np.zeros(scenarios, dtype=np.float64)
    pension_balance = np.zeros(scenarios, dtype=np.float64)
    pension_lump_net = np.zeros(scenarios, dtype=np.float64)
    for index in range(scenarios):
        if arm.mode is IsaOperatingMode.NO_ISA:
            isa_net[index] = 0.0
        else:
            balance = math.floor(float(isa_hold[index].sum()))
            account_years = max(horizon - isa_open_k, isa_regime.minimum_contract_years)
            quote = quote_isa_closure(
                isa_regime,
                tax_class=isa_class,
                balance_krw=balance,
                principal_krw=isa_cum_principal,
                account_years=account_years,
            )
            isa_tax_acc[index] += float(quote.national_tax_krw + quote.local_tax_krw)
            isa_net[index] = float(quote.net_proceeds_krw)
        side_value = math.floor(float(side_hold[index].sum()))
        basis = float(side_basis[index])
        tax = math.floor(float(annual_capital_gains_tax_krw(float(side_value) - basis, overseas_regime)))
        side_tax_acc[index] = float(tax)
        side_net[index] = float(side_value - tax)
        pension_value = math.floor(float(pension_hold[index].sum()))
        pension_balance[index] = float(pension_value)
        threshold_quote = quote_threshold_annuity_exit(
            valuation_date=valuation_date,
            balance_krw=pension_value,
            uncredited_principal_krw=int(pension_uncredited[index]),
            drawing_years=plan.annuity_drawing_years,
            profile=pension_profile,
            regime=pension_regime,
        )
        exit_tax_acc[index] = float(threshold_quote.national_tax_krw + threshold_quote.local_tax_krw)
        pension_net[index] = float(threshold_quote.net_krw)
        lump = quote_pension_exit(
            PensionExitKind.LUMP_SUM,
            valuation_date=valuation_date,
            balance_krw=pension_value,
            uncredited_principal_krw=int(pension_uncredited[index]),
            foreign_tax_withheld_krw=0,
            profile=pension_profile,
            regime=pension_regime,
        )
        pension_lump_net[index] = float(lump.net_krw)
    household_net = isa_net + side_net + pension_net + pending
    contributed = int(horizon * (plan.pension_annual_krw + plan.isa_annual_budget_krw))
    return HouseholdOutcome(
        valuation_date=valuation_date,
        contributed_krw=contributed,
        household_net_krw=household_net,
        isa_net_krw=isa_net,
        side_net_krw=side_net,
        pension_net_krw=pension_net,
        pending_refund_krw=pending,
        pension_balance_krw=pension_balance,
        pension_uncredited_krw=pension_uncredited,
        pension_lump_sum_net_krw=pension_lump_net,
        isa_tax_krw=isa_tax_acc,
        side_tax_krw=side_tax_acc,
        pension_exit_tax_krw=exit_tax_acc,
        credits_krw=credits_acc,
        transfers_krw=transfers_acc,
    )
