"""Pension household view: same cash through pension plus a side account vs general only."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.sim.after_tax_engine import AfterTaxConfig, ExecutionMode
from src.sim.pension_engine import PensionBacktestResult, PensionDataError
from src.sim.tax import KrOverseasTaxRegime
from src.validation.gate import wealth_quantile

__all__ = [
    "GeneralRunner",
    "HouseholdCashPlan",
    "HouseholdRow",
    "HouseholdSummary",
    "PensionHouseholdReport",
    "PensionHouseholdSpec",
    "build_household_cash_plan",
    "evaluate_household_arm_cohort",
    "evaluate_household_row",
    "parse_household_spec",
    "summarize_household",
]

_HOUSEHOLD_KEYS: frozenset[str] = frozenset(
    {
        "general_tax_regime_path",
        "commission_bps",
        "fx_spread_bps",
        "harvest_gains",
        "fractional_shares",
    }
)


@dataclass(frozen=True, slots=True)
class PensionHouseholdSpec:
    """General-account assumptions for the same-cash comparison, loaded from the campaign config."""

    general_tax_regime_path: str
    commission_bps: float
    fx_spread_bps: float
    harvest_gains: bool
    fractional_shares: bool


@dataclass(frozen=True, slots=True)
class HouseholdCashPlan:
    """Same-cash split of one cohort's household cash between the pension and a side general account.

    ``household_deposits_krw`` is the counterfactual schedule (all available cash);
    ``side_deposits_krw`` holds leftover cash plus settled credit refunds.
    """

    household_deposits_krw: Mapping[date, float]
    side_deposits_krw: Mapping[date, float]
    pension_contributions_krw: int
    leftover_krw: int
    settled_refunds_krw: int
    pending_refund_krw: int


@dataclass(frozen=True, slots=True)
class HouseholdRow:
    """One (arm, profile, cohort) same-cash outcome; all values nominal KRW at the cohort end."""

    arm_id: str
    profile_id: str
    horizon_months: int
    cohort_start: date
    cohort_end: date
    plan: HouseholdCashPlan
    general_only_krw: float
    side_account_krw: float
    pension_lump_sum_net_krw: int
    pension_annuity_low_net_krw: int
    pension_annuity_high_net_krw: int
    household_liquidation_krw: float
    household_annuity_low_krw: float
    household_annuity_high_krw: float
    account_advantage_liquidation: float
    account_advantage_annuity_low: float
    account_advantage_annuity_high: float


@dataclass(frozen=True, slots=True)
class HouseholdSummary:
    """Account effect and asset effect for one arm, horizon, and profile over included cohorts."""

    arm_id: str
    horizon_months: int
    profile_id: str
    cohort_count: int
    excluded_payout_cohorts: int
    median_account_advantage_liquidation: float | None
    worst_account_advantage_liquidation: float | None
    median_account_advantage_annuity_low: float | None
    median_account_advantage_annuity_high: float | None
    median_asset_effect_household: float | None
    median_asset_effect_general_only: float | None


@dataclass(frozen=True, slots=True)
class PensionHouseholdReport:
    """Same-cash household rows and per-profile summaries for one campaign."""

    spec: PensionHouseholdSpec
    rows: tuple[HouseholdRow, ...]
    summaries: tuple[HouseholdSummary, ...]


GeneralRunner = Callable[[AfterTaxConfig], float]
"""Returns the terminal after-tax liquidation value (KRW) of one general-account run."""


def parse_household_spec(document: object) -> PensionHouseholdSpec:
    """Validate the campaign config's ``household_view`` object.

    Raises:
        ValueError: Missing/extra keys, non-string or missing regime path, negative or
            non-finite bps, or non-bool flags.
    """
    if not isinstance(document, dict):
        raise ValueError("household_view must be an object")
    keys = set(document.keys())
    if keys != _HOUSEHOLD_KEYS:
        raise ValueError(
            f"household_view has unexpected keys: missing={sorted(_HOUSEHOLD_KEYS - keys)} "
            f"extra={sorted(keys - _HOUSEHOLD_KEYS)}"
        )
    regime_path = document["general_tax_regime_path"]
    if not isinstance(regime_path, str) or not regime_path.strip():
        raise ValueError(f"household_view general_tax_regime_path must be a non-blank string, got {regime_path!r}")
    if not Path(regime_path).is_file():
        raise ValueError(f"household_view general_tax_regime_path not found: {regime_path!r}")
    bps: dict[str, float] = {}
    for name in ("commission_bps", "fx_spread_bps"):
        value = document[name]
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"household_view {name} must be finite and nonnegative, got {value!r}")
        bps[name] = float(value)
    flags: dict[str, bool] = {}
    for name in ("harvest_gains", "fractional_shares"):
        value = document[name]
        if not isinstance(value, bool):
            raise ValueError(f"household_view {name} must be a bool, got {value!r}")
        flags[name] = value
    return PensionHouseholdSpec(
        general_tax_regime_path=regime_path,
        commission_bps=bps["commission_bps"],
        fx_spread_bps=bps["fx_spread_bps"],
        harvest_gains=flags["harvest_gains"],
        fractional_shares=flags["fractional_shares"],
    )


def build_household_cash_plan(
    result: PensionBacktestResult,
    *,
    available_cash_events_krw: Mapping[date, int],
    tax_credit_settlement_dates: Mapping[int, date],
    cohort_start: date,
    cohort_end: date,
) -> HouseholdCashPlan:
    """Split the cohort's available cash into pension contributions, leftover, and refunds.

    Leftover for a year is that year's available cash minus that year's pension
    contributions, deposited on the year's first available-cash date; refunds are the
    national+local credits of each tax year deposited on their settlement date when it
    falls within the cohort, otherwise counted as a pending receivable.

    Raises:
        ValueError: When a year's contributions exceed that year's available cash
            (cross-year carry is not modelled) or the result contains payouts.
    """
    if result.withdrawals or result.payout_shortfalls_krw:
        raise ValueError("household cash plan excludes cohorts with in-window pension payouts")
    available = {day: amount for day, amount in available_cash_events_krw.items() if cohort_start <= day <= cohort_end}
    contributed_by_year: dict[int, int] = {}
    for day, amount in result.contribution_cashflows_krw:
        if not cohort_start <= day <= cohort_end:
            continue
        contributed_by_year[day.year] = contributed_by_year.get(day.year, 0) + amount
    available_by_year: dict[int, int] = {}
    first_available_by_year: dict[int, date] = {}
    for day in sorted(available):
        available_by_year[day.year] = available_by_year.get(day.year, 0) + available[day]
        first_available_by_year.setdefault(day.year, day)
    side: dict[date, float] = {}
    leftover_total = 0
    for year in sorted(set(available_by_year) | set(contributed_by_year)):
        contributed = contributed_by_year.get(year, 0)
        cash = available_by_year.get(year, 0)
        if contributed > cash:
            raise ValueError(
                f"household cash plan cannot fund {contributed} KRW of {year} pension contributions "
                f"from {cash} KRW of available cash (cross-year carry is not modelled)"
            )
        leftover = cash - contributed
        leftover_total += leftover
        if leftover > 0:
            deposit_day = first_available_by_year[year]
            side[deposit_day] = side.get(deposit_day, 0.0) + float(leftover)
    settled_total = 0
    pending_total = 0
    for credit in result.tax_credits:
        refund = credit.national_credit_krw + credit.local_credit_krw
        settle_day = tax_credit_settlement_dates[credit.tax_year]
        if settle_day <= cohort_end:
            settled_total += refund
            if refund > 0:
                side[settle_day] = side.get(settle_day, 0.0) + float(refund)
        else:
            pending_total += refund
    contributed_total = sum(contributed_by_year.values())
    return HouseholdCashPlan(
        household_deposits_krw={day: float(amount) for day, amount in sorted(available.items())},
        side_deposits_krw=dict(sorted(side.items())),
        pension_contributions_krw=contributed_total,
        leftover_krw=leftover_total,
        settled_refunds_krw=settled_total,
        pending_refund_krw=pending_total,
    )


def evaluate_household_row(
    *,
    arm_id: str,
    profile_id: str,
    horizon_months: int,
    cohort_start: date,
    cohort_end: date,
    plan: HouseholdCashPlan,
    pension_lump_sum_net_krw: int,
    pension_annuity_low_net_krw: int,
    pension_annuity_high_net_krw: int,
    general_only_krw: float,
    side_account_krw: float,
) -> HouseholdRow:
    """Combine pension exit nets with the side general account against the counterfactual.

    Raises:
        PensionDataError: When the counterfactual value is not positive.
    """
    if not math.isfinite(general_only_krw) or general_only_krw <= 0:
        raise PensionDataError(f"household counterfactual for {arm_id!r} is not positive: {general_only_krw!r}")
    household_liquidation = pension_lump_sum_net_krw + side_account_krw + plan.pending_refund_krw
    household_low = pension_annuity_low_net_krw + side_account_krw + plan.pending_refund_krw
    household_high = pension_annuity_high_net_krw + side_account_krw + plan.pending_refund_krw
    return HouseholdRow(
        arm_id=arm_id,
        profile_id=profile_id,
        horizon_months=horizon_months,
        cohort_start=cohort_start,
        cohort_end=cohort_end,
        plan=plan,
        general_only_krw=general_only_krw,
        side_account_krw=side_account_krw,
        pension_lump_sum_net_krw=pension_lump_sum_net_krw,
        pension_annuity_low_net_krw=pension_annuity_low_net_krw,
        pension_annuity_high_net_krw=pension_annuity_high_net_krw,
        household_liquidation_krw=household_liquidation,
        household_annuity_low_krw=household_low,
        household_annuity_high_krw=household_high,
        account_advantage_liquidation=household_liquidation / general_only_krw,
        account_advantage_annuity_low=household_low / general_only_krw,
        account_advantage_annuity_high=household_high / general_only_krw,
    )


def evaluate_household_arm_cohort(
    *,
    arm_id: str,
    profile_id: str,
    horizon_months: int,
    cohort_start: date,
    cohort_end: date,
    result: PensionBacktestResult,
    available_cash_events_krw: Mapping[date, int],
    tax_credit_settlement_dates: Mapping[int, date],
    pension_nets_krw: tuple[int, int, int],
    arm_targets: Mapping[str, float],
    household_spec: PensionHouseholdSpec,
    general_regime: KrOverseasTaxRegime,
    runner: GeneralRunner,
    counterfactual_cache: dict[tuple[int, str, date, date], float],
    excluded: dict[tuple[str, int, str], int],
) -> HouseholdRow | None:
    """Evaluate one (arm, profile, cohort) same-cash outcome, reusing cached counterfactuals.

    Rows with in-window pension payouts are skipped (counted in ``excluded``) because
    payout cash would need a consumption model. The counterfactual general-only value
    is computed once per (horizon, arm, cohort); an empty side schedule yields 0.0
    without an engine call, and a side schedule identical to the counterfactual schedule
    reuses the cached value so a zero-credit household measures exactly 1.0.
    """
    key = (horizon_months, arm_id, cohort_start, cohort_end)
    if result.withdrawals or result.payout_shortfalls_krw:
        excluded[(arm_id, horizon_months, profile_id)] = excluded.get((arm_id, horizon_months, profile_id), 0) + 1
        return None
    plan = build_household_cash_plan(
        result,
        available_cash_events_krw=available_cash_events_krw,
        tax_credit_settlement_dates=tax_credit_settlement_dates,
        cohort_start=cohort_start,
        cohort_end=cohort_end,
    )
    cached = counterfactual_cache.get(key)
    if cached is None:
        general_only = runner(
            AfterTaxConfig(
                start=cohort_start,
                end=cohort_end,
                monthly_contribution_krw=0.0,
                contribution_schedule_krw=dict(plan.household_deposits_krw),
                targets=dict(arm_targets),
                tax_regime=general_regime,
                mode=ExecutionMode.BUY_ONLY,
                harvest_gains=household_spec.harvest_gains,
                commission_bps=household_spec.commission_bps,
                fx_spread_bps=household_spec.fx_spread_bps,
                fractional_shares=household_spec.fractional_shares,
                tax_enabled=True,
            )
        )
        counterfactual_cache[key] = general_only
    else:
        general_only = cached
    if not plan.side_deposits_krw:
        side_value = 0.0
    elif plan.side_deposits_krw == plan.household_deposits_krw:
        side_value = general_only
    else:
        side_value = runner(
            AfterTaxConfig(
                start=cohort_start,
                end=cohort_end,
                monthly_contribution_krw=0.0,
                contribution_schedule_krw=dict(plan.side_deposits_krw),
                targets=dict(arm_targets),
                tax_regime=general_regime,
                mode=ExecutionMode.BUY_ONLY,
                harvest_gains=household_spec.harvest_gains,
                commission_bps=household_spec.commission_bps,
                fx_spread_bps=household_spec.fx_spread_bps,
                fractional_shares=household_spec.fractional_shares,
                tax_enabled=True,
            )
        )
    lump_net, low_net, high_net = pension_nets_krw
    return evaluate_household_row(
        arm_id=arm_id,
        profile_id=profile_id,
        horizon_months=horizon_months,
        cohort_start=cohort_start,
        cohort_end=cohort_end,
        plan=plan,
        pension_lump_sum_net_krw=lump_net,
        pension_annuity_low_net_krw=low_net,
        pension_annuity_high_net_krw=high_net,
        general_only_krw=general_only,
        side_account_krw=side_value,
    )


def summarize_household(
    rows: Sequence[HouseholdRow],
    *,
    baseline_arm_id: str,
    excluded: Mapping[tuple[str, int, str], int],
) -> tuple[HouseholdSummary, ...]:
    """Summarize account and asset effects per (arm, horizon, profile) over included cohorts."""
    groups: dict[tuple[str, int, str], list[HouseholdRow]] = {}
    for row in rows:
        groups.setdefault((row.arm_id, row.horizon_months, row.profile_id), []).append(row)
    baseline_rows: dict[tuple[str, int, date, date], HouseholdRow] = {}
    for row in rows:
        if row.arm_id == baseline_arm_id:
            baseline_rows[(row.profile_id, row.horizon_months, row.cohort_start, row.cohort_end)] = row
    summaries: list[HouseholdSummary] = []
    for (arm_id, horizon_months, profile_id) in sorted(set(groups) | set(excluded)):
        group = groups.get((arm_id, horizon_months, profile_id), [])
        liquidations = [row.account_advantage_liquidation for row in group]
        lows = [row.account_advantage_annuity_low for row in group]
        highs = [row.account_advantage_annuity_high for row in group]
        household_effects: list[float] = []
        general_effects: list[float] = []
        for row in group:
            base = baseline_rows.get((row.profile_id, row.horizon_months, row.cohort_start, row.cohort_end))
            if base is None or base.household_liquidation_krw <= 0 or base.general_only_krw <= 0:
                continue
            household_effects.append(row.household_liquidation_krw / base.household_liquidation_krw)
            general_effects.append(row.general_only_krw / base.general_only_krw)
        summaries.append(
            HouseholdSummary(
                arm_id=arm_id,
                horizon_months=horizon_months,
                profile_id=profile_id,
                cohort_count=len(group),
                excluded_payout_cohorts=excluded.get((arm_id, horizon_months, profile_id), 0),
                median_account_advantage_liquidation=wealth_quantile(liquidations, 0.5) if liquidations else None,
                worst_account_advantage_liquidation=min(liquidations) if liquidations else None,
                median_account_advantage_annuity_low=wealth_quantile(lows, 0.5) if lows else None,
                median_account_advantage_annuity_high=wealth_quantile(highs, 0.5) if highs else None,
                median_asset_effect_household=wealth_quantile(household_effects, 0.5) if household_effects else None,
                median_asset_effect_general_only=wealth_quantile(general_effects, 0.5) if general_effects else None,
            )
        )
    return tuple(summaries)
