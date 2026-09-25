"""Invariant guards for the pension household same-cash view."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from src.sim.after_tax_engine import AfterTaxConfig, ExecutionMode
from src.sim.pension_engine import PensionBacktestResult, PensionDataError, PensionMarketMode
from src.sim.pension_tax import PensionCreditQuote, PensionWithdrawalQuote
from src.sim.tax import load_tax_regime
from src.validation.gate import wealth_quantile
from src.validation.pension_household import (
    HouseholdCashPlan,
    HouseholdRow,
    PensionHouseholdSpec,
    build_household_cash_plan,
    evaluate_household_arm_cohort,
    evaluate_household_row,
    parse_household_spec,
    summarize_household,
)

_REGIME = load_tax_regime("configs/tax/kr_overseas_equity.json")
_TARGETS = {"SPY": 1.0}
_HOUSEHOLD_SPEC = PensionHouseholdSpec(
    general_tax_regime_path="configs/tax/kr_overseas_equity.json",
    commission_bps=10.0,
    fx_spread_bps=20.0,
    harvest_gains=True,
    fractional_shares=False,
)


def _credit(year: int, refund_krw: int) -> PensionCreditQuote:
    return PensionCreditQuote(
        tax_year=year,
        contributed_krw=0,
        theoretical_national_credit_krw=refund_krw,
        theoretical_local_credit_krw=0,
        credited_principal_krw=0,
        uncredited_principal_krw=0,
        national_credit_krw=refund_krw,
        local_credit_krw=0,
    )


def _result(
    *,
    contributions: dict[date, int] | None = None,
    refunds: dict[int, int] | None = None,
    withdrawals: tuple[PensionWithdrawalQuote, ...] = (),
    shortfalls: tuple[tuple[date, int], ...] = (),
) -> PensionBacktestResult:
    return PensionBacktestResult(
        snapshots=(),
        contribution_cashflows_krw=tuple(sorted((contributions or {}).items())),
        tax_credits=tuple(_credit(year, refund) for year, refund in sorted((refunds or {}).items())),
        withdrawals=withdrawals,
        payout_shortfalls_krw=shortfalls,
        terminal_nav_krw=0,
        after_tax_external_cashflows_krw=(),
        is_retirement_terminal=False,
        market_mode=PensionMarketMode.US_PROXY,
        terminal_credited_principal_krw=0,
        terminal_uncredited_principal_krw=0,
        foreign_tax_withheld_krw=0,
    )


def _withdrawal() -> PensionWithdrawalQuote:
    return PensionWithdrawalQuote(
        withdrawal_date=date(2024, 12, 1),
        gross_withdrawal_krw=1_000_000,
        tax_free_krw=0,
        taxable_krw=1_000_000,
        national_tax_krw=150_000,
        local_tax_krw=15_000,
        remaining_uncredited_principal_krw=0,
        remaining_credited_principal_krw=0,
    )


def _available(*years: int) -> dict[date, int]:
    return {date(year, 1, 3): 6_000_000 for year in years}


def _settlements(*years: int) -> dict[int, date]:
    return {year: date(year + 1, 5, 31) for year in years}


def _plan(
    *,
    household: dict[date, float] | None = None,
    side: dict[date, float] | None = None,
    pending: int = 0,
) -> HouseholdCashPlan:
    return HouseholdCashPlan(
        household_deposits_krw=household or {},
        side_deposits_krw=side or {},
        pension_contributions_krw=0,
        leftover_krw=0,
        settled_refunds_krw=0,
        pending_refund_krw=pending,
    )


def test_build_household_cash_plan_same_cash_identity() -> None:
    """Half of each year's cash funds the pension; the rest lands on the available date."""
    result = _result(
        contributions={date(2023, 1, 15): 3_000_000, date(2024, 1, 15): 3_000_000},
        refunds={2023: 495_000, 2024: 495_000},
    )
    plan = build_household_cash_plan(
        result,
        available_cash_events_krw=_available(2023, 2024),
        tax_credit_settlement_dates=_settlements(2023, 2024),
        cohort_start=date(2023, 1, 1),
        cohort_end=date(2024, 12, 31),
    )
    assert plan.pension_contributions_krw == 6_000_000
    assert plan.leftover_krw == 6_000_000
    assert plan.settled_refunds_krw == 495_000
    assert plan.pending_refund_krw == 495_000
    assert plan.side_deposits_krw == {
        date(2023, 1, 3): 3_000_000.0,
        date(2024, 1, 3): 3_000_000.0,
        date(2024, 5, 31): 495_000.0,
    }
    assert sum(plan.household_deposits_krw.values()) == plan.pension_contributions_krw + plan.leftover_krw


def test_build_household_cash_plan_refunds_settled_or_pending() -> None:
    """A refund settling inside the cohort is deposited; a later one stays receivable."""
    result = _result(refunds={2023: 495_000, 2024: 495_000})
    plan = build_household_cash_plan(
        result,
        available_cash_events_krw=_available(2023, 2024),
        tax_credit_settlement_dates=_settlements(2023, 2024),
        cohort_start=date(2023, 1, 1),
        cohort_end=date(2024, 12, 31),
    )
    assert plan.side_deposits_krw[date(2024, 5, 31)] == 495_000.0
    assert plan.pending_refund_krw == 495_000
    assert plan.settled_refunds_krw == 495_000


def test_build_household_cash_plan_cross_year_carry_rejected() -> None:
    """Contributions above the year's available cash fail closed instead of borrowing."""
    result = _result(contributions={date(2024, 1, 15): 7_000_000})
    with pytest.raises(ValueError, match="cross-year carry"):
        build_household_cash_plan(
            result,
            available_cash_events_krw=_available(2024),
            tax_credit_settlement_dates=_settlements(2024),
            cohort_start=date(2024, 1, 1),
            cohort_end=date(2024, 12, 31),
        )


def test_build_household_cash_plan_payout_rejected() -> None:
    """Results with withdrawals or shortfalls cannot be split without a consumption model."""
    payout = _result(withdrawals=(_withdrawal(),))
    short = _result(shortfalls=((date(2024, 12, 1), 100),))
    for result in (payout, short):
        with pytest.raises(ValueError, match="payout"):
            build_household_cash_plan(
                result,
                available_cash_events_krw=_available(2024),
                tax_credit_settlement_dates=_settlements(2024),
                cohort_start=date(2024, 1, 1),
                cohort_end=date(2024, 12, 31),
            )


def test_zero_credit_household_equals_general_only() -> None:
    """No contributions and no credits reuse the counterfactual value for exactly 1.0."""
    calls: list[dict[date, float]] = []

    def _runner(config: AfterTaxConfig) -> float:
        calls.append(dict(config.contribution_schedule_krw or {}))
        return sum((config.contribution_schedule_krw or {}).values()) * 1.5

    cache: dict[tuple[int, str, date, date], float] = {}
    excluded: dict[tuple[str, int, str], int] = {}
    row = evaluate_household_arm_cohort(
        arm_id="sp500",
        profile_id="zero",
        horizon_months=24,
        cohort_start=date(2023, 1, 1),
        cohort_end=date(2024, 12, 31),
        result=_result(),
        available_cash_events_krw=_available(2023, 2024),
        tax_credit_settlement_dates=_settlements(2023, 2024),
        pension_nets_krw=(0, 0, 0),
        arm_targets=_TARGETS,
        household_spec=_HOUSEHOLD_SPEC,
        general_regime=_REGIME,
        runner=_runner,
        counterfactual_cache=cache,
        excluded=excluded,
    )
    assert row is not None
    assert row.plan.side_deposits_krw == row.plan.household_deposits_krw
    assert row.account_advantage_liquidation == 1.0
    assert row.account_advantage_annuity_low == 1.0
    assert row.account_advantage_annuity_high == 1.0
    assert len(calls) == 1
    assert calls[0] == {date(2023, 1, 3): 6_000_000.0, date(2024, 1, 3): 6_000_000.0}
    assert excluded == {}


def test_evaluate_household_row_composition() -> None:
    """Household value adds each pension net to the side account plus pending refunds."""
    plan = _plan(pending=1_000_000)
    row = evaluate_household_row(
        arm_id="sp500",
        profile_id="full",
        horizon_months=24,
        cohort_start=date(2023, 1, 1),
        cohort_end=date(2024, 12, 31),
        plan=plan,
        pension_lump_sum_net_krw=8_350_000,
        pension_annuity_low_net_krw=9_500_000,
        pension_annuity_high_net_krw=8_500_000,
        general_only_krw=10_000_000.0,
        side_account_krw=2_000_000.0,
    )
    assert row.household_liquidation_krw == 8_350_000 + 2_000_000 + 1_000_000
    assert row.household_annuity_low_krw == 9_500_000 + 2_000_000 + 1_000_000
    assert row.household_annuity_high_krw == 8_500_000 + 2_000_000 + 1_000_000
    assert row.household_annuity_low_krw >= row.household_annuity_high_krw >= row.household_liquidation_krw
    assert row.account_advantage_liquidation == pytest.approx(row.household_liquidation_krw / 10_000_000.0)


def test_counterfactual_cached_per_arm_cohort() -> None:
    """Three profiles sharing one arm and cohort trigger one counterfactual run."""
    calls: list[dict[date, float]] = []

    def _runner(config: AfterTaxConfig) -> float:
        calls.append(dict(config.contribution_schedule_krw or {}))
        assert config.monthly_contribution_krw == 0.0
        assert config.targets == _TARGETS
        assert config.mode is ExecutionMode.BUY_ONLY
        assert config.tax_enabled is True
        return sum((config.contribution_schedule_krw or {}).values()) * 1.5 + 1.0

    cache: dict[tuple[int, str, date, date], float] = {}
    excluded: dict[tuple[str, int, str], int] = {}
    counterfactual = {date(2023, 1, 3): 6_000_000.0, date(2024, 1, 3): 6_000_000.0}
    for index, contributed in enumerate((6_000_000, 3_000_000, 0)):
        result = _result(
            contributions={date(2023, 1, 15): contributed // 2, date(2024, 1, 15): contributed - contributed // 2},
            refunds={2023: 495_000, 2024: 495_000} if index else {},
        )
        row = evaluate_household_arm_cohort(
            arm_id="sp500",
            profile_id=f"profile_{index}",
            horizon_months=24,
            cohort_start=date(2023, 1, 1),
            cohort_end=date(2024, 12, 31),
            result=result,
            available_cash_events_krw=_available(2023, 2024),
            tax_credit_settlement_dates=_settlements(2023, 2024),
            pension_nets_krw=(1_000_000, 2_000_000, 1_500_000),
            arm_targets=_TARGETS,
            household_spec=_HOUSEHOLD_SPEC,
            general_regime=_REGIME,
            runner=_runner,
            counterfactual_cache=cache,
            excluded=excluded,
        )
        assert row is not None
    assert calls.count(counterfactual) == 1
    assert len(cache) == 1
    assert excluded == {}


def _household_row(
    *,
    arm: str,
    profile: str,
    horizon: int = 24,
    start: date = date(2023, 1, 1),
    end: date = date(2024, 12, 31),
    general: float,
    household: float,
) -> HouseholdRow:
    return evaluate_household_row(
        arm_id=arm,
        profile_id=profile,
        horizon_months=horizon,
        cohort_start=start,
        cohort_end=end,
        plan=_plan(),
        pension_lump_sum_net_krw=0,
        pension_annuity_low_net_krw=0,
        pension_annuity_high_net_krw=0,
        general_only_krw=general,
        side_account_krw=household,
    )


def test_summarize_household_per_profile() -> None:
    """Summaries split by profile; asset effects pair the same profile and cohort."""
    rows = [
        _household_row(arm="sp500", profile="full", general=10_000_000.0, household=11_000_000.0),
        _household_row(arm="sp500", profile="full", general=20_000_000.0, household=22_000_000.0,
                       start=date(2024, 1, 1), end=date(2025, 12, 31)),
        _household_row(arm="nasdaq", profile="full", general=10_000_000.0, household=12_000_000.0),
        _household_row(arm="sp500", profile="zero", general=10_000_000.0, household=10_000_000.0),
        _household_row(arm="nasdaq", profile="zero", general=10_000_000.0, household=10_000_000.0),
    ]
    summaries = summarize_household(rows, baseline_arm_id="sp500", excluded={})
    assert {(s.arm_id, s.horizon_months, s.profile_id) for s in summaries} == {
        ("sp500", 24, "full"), ("nasdaq", 24, "full"), ("sp500", 24, "zero"), ("nasdaq", 24, "zero"),
    }
    nasdaq_full = next(s for s in summaries if (s.arm_id, s.profile_id) == ("nasdaq", "full"))
    assert nasdaq_full.cohort_count == 1
    assert nasdaq_full.median_account_advantage_liquidation == pytest.approx(1.2)
    assert nasdaq_full.median_asset_effect_household == pytest.approx(12_000_000.0 / 11_000_000.0)
    assert nasdaq_full.median_asset_effect_general_only == pytest.approx(1.0)
    sp500_full = next(s for s in summaries if (s.arm_id, s.profile_id) == ("sp500", "full"))
    assert sp500_full.median_account_advantage_liquidation == pytest.approx(
        wealth_quantile([1.1, 1.1], 0.5)
    )
    assert sp500_full.median_asset_effect_household == pytest.approx(1.0)
    zero = next(s for s in summaries if (s.arm_id, s.profile_id) == ("nasdaq", "zero"))
    assert zero.median_account_advantage_liquidation == pytest.approx(1.0)


def test_build_household_cash_plan_ignores_out_of_cohort_contributions() -> None:
    """Contributions dated outside the cohort window do not enter the plan."""
    result = _result(contributions={date(2022, 1, 15): 1_000_000, date(2023, 1, 15): 3_000_000})
    plan = build_household_cash_plan(
        result,
        available_cash_events_krw=_available(2023),
        tax_credit_settlement_dates=_settlements(2023),
        cohort_start=date(2023, 1, 1),
        cohort_end=date(2023, 12, 31),
    )
    assert plan.pension_contributions_krw == 3_000_000
    assert plan.leftover_krw == 3_000_000


def test_summarize_household_skips_unpaired_baseline_cohorts() -> None:
    """Asset effects cover only cohorts with a baseline row; otherwise None."""
    rows = [
        _household_row(arm="nasdaq", profile="full", general=10_000_000.0, household=12_000_000.0),
        _household_row(
            arm="sp500", profile="full", general=20_000_000.0, household=22_000_000.0,
            start=date(2024, 1, 1), end=date(2025, 12, 31),
        ),
    ]
    summaries = summarize_household(rows, baseline_arm_id="sp500", excluded={})
    nasdaq = next(s for s in summaries if s.arm_id == "nasdaq")
    assert nasdaq.median_account_advantage_liquidation == pytest.approx(1.2)
    assert nasdaq.median_asset_effect_household is None
    assert nasdaq.median_asset_effect_general_only is None
    sp500 = next(s for s in summaries if s.arm_id == "sp500")
    assert sp500.median_asset_effect_household == pytest.approx(1.0)
    assert sp500.median_asset_effect_general_only == pytest.approx(1.0)


def test_summarize_household_excluded_cohorts_counted() -> None:
    """Skipped payout cohorts are counted; fully excluded keys report None statistics."""
    rows = [_household_row(arm="sp500", profile="full", general=10_000_000.0, household=11_000_000.0)]
    summaries = summarize_household(
        rows, baseline_arm_id="sp500", excluded={("sp500", 24, "full"): 1, ("nasdaq", 24, "full"): 2}
    )
    included = next(s for s in summaries if s.arm_id == "sp500")
    assert included.cohort_count == 1
    assert included.excluded_payout_cohorts == 1
    assert included.median_account_advantage_liquidation == pytest.approx(1.1)
    missing = next(s for s in summaries if s.arm_id == "nasdaq")
    assert missing.cohort_count == 0
    assert missing.excluded_payout_cohorts == 2
    assert missing.median_account_advantage_liquidation is None
    assert missing.worst_account_advantage_liquidation is None
    assert missing.median_asset_effect_household is None
    assert missing.median_asset_effect_general_only is None


def test_evaluate_nonpositive_counterfactual_fails_closed() -> None:
    """A zero or negative counterfactual value never divides."""
    plan = _plan()
    for bad in (0.0, -100.0):
        with pytest.raises(PensionDataError, match="not positive"):
            evaluate_household_row(
                arm_id="sp500",
                profile_id="full",
                horizon_months=24,
                cohort_start=date(2023, 1, 1),
                cohort_end=date(2024, 12, 31),
                plan=plan,
                pension_lump_sum_net_krw=1_000_000,
                pension_annuity_low_net_krw=2_000_000,
                pension_annuity_high_net_krw=1_500_000,
                general_only_krw=bad,
                side_account_krw=0.0,
            )


def test_parse_household_spec() -> None:
    """Valid blocks parse; malformed blocks fail closed naming the field."""
    valid = {
        "general_tax_regime_path": "configs/tax/kr_overseas_equity.json",
        "commission_bps": 10.0,
        "fx_spread_bps": 20.0,
        "harvest_gains": True,
        "fractional_shares": False,
    }
    parsed = parse_household_spec(valid)
    assert parsed.general_tax_regime_path == "configs/tax/kr_overseas_equity.json"
    assert parsed.commission_bps == pytest.approx(10.0)
    assert parsed.harvest_gains is True
    assert parsed.fractional_shares is False
    invalid: list[tuple[Any, str]] = [
        ([], "must be an object"),
        ({**valid, "extra": 1}, "extra"),
        ({k: v for k, v in valid.items() if k != "harvest_gains"}, "missing"),
        ({**valid, "general_tax_regime_path": ""}, "non-blank string"),
        ({**valid, "general_tax_regime_path": 7}, "non-blank string"),
        ({**valid, "general_tax_regime_path": "missing.json"}, "not found"),
        ({**valid, "commission_bps": -1.0}, "nonnegative"),
        ({**valid, "fx_spread_bps": float("nan")}, "nonnegative"),
        ({**valid, "commission_bps": float("inf")}, "nonnegative"),
        ({**valid, "commission_bps": True}, "nonnegative"),
        ({**valid, "harvest_gains": 1}, "must be a bool"),
        ({**valid, "fractional_shares": "no"}, "must be a bool"),
    ]
    for document, match in invalid:
        with pytest.raises(ValueError, match=match):
            parse_household_spec(document)


def test_payout_cohorts_excluded_and_counted() -> None:
    """The orchestrator skips payout cohorts and counts them per key."""
    cache: dict[tuple[int, str, date, date], float] = {}
    excluded: dict[tuple[str, int, str], int] = {}
    kwargs: dict[str, Any] = {
        "arm_id": "sp500",
        "profile_id": "retiree",
        "horizon_months": 24,
        "cohort_start": date(2023, 1, 1),
        "cohort_end": date(2024, 12, 31),
        "available_cash_events_krw": _available(2023, 2024),
        "tax_credit_settlement_dates": _settlements(2023, 2024),
        "pension_nets_krw": (1_000_000, 2_000_000, 1_500_000),
        "arm_targets": _TARGETS,
        "household_spec": _HOUSEHOLD_SPEC,
        "general_regime": _REGIME,
        "runner": lambda config: 1.0,
        "counterfactual_cache": cache,
        "excluded": excluded,
    }
    assert evaluate_household_arm_cohort(result=_result(withdrawals=(_withdrawal(),)), **kwargs) is None
    assert evaluate_household_arm_cohort(result=_result(shortfalls=((date(2024, 12, 1), 5),)), **kwargs) is None
    assert excluded == {("sp500", 24, "retiree"): 2}
    assert cache == {}
