"""Invariant guards for the buy-only pension backtest engine."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from src.sim.pension_engine import (
    PensionBacktestConfig,
    PensionDataError,
    PensionMarketMode,
    proxy_krw_marks,
    run_pension_backtest,
)
from src.sim.pension_tax import (
    PensionTaxProfile,
    PensionTaxRegime,
    load_pension_tax_regime,
    quote_annual_pension_withdrawal,
)

_REGIME_PATH = Path(__file__).resolve().parents[3] / "configs" / "tax" / "kr_pension_2026.json"


def _regime() -> PensionTaxRegime:
    return load_pension_tax_regime(_REGIME_PATH)


def _profile(
    *,
    birth: date = date(1985, 1, 1),
    opened: date = date(2015, 1, 1),
    years: tuple[int, ...] = (2023, 2024),
    income: int = 50_000_000,
    national: int = 2_000_000,
    local: int = 2_000_000,
    other: int = 0,
    pension_start: date | None = None,
) -> PensionTaxProfile:
    return PensionTaxProfile(
        profile_id="probe",
        birth_date=birth,
        account_open_date=opened,
        income_kind="wage",
        annual_income_krw=dict.fromkeys(years, income),
        remaining_national_tax_krw=dict.fromkeys(years, national),
        remaining_local_tax_krw=dict.fromkeys(years, local),
        other_private_pension_income_krw=dict.fromkeys(years, other),
        pension_start_date=pension_start,
    )


def _us_prices(sessions: list[date], tickers: tuple[str, ...] = ("SPY",)) -> pl.DataFrame:
    rows = []
    for ticker in tickers:
        base = 400.0 if ticker == "SPY" else 300.0
        for index, day in enumerate(sessions):
            price = base + 0.1 * index
            rows.append({"ticker": ticker, "date": day, "close": price, "adjusted_close": price, "dividend": 0.0})
    return pl.DataFrame(rows)


def _fx(sessions: list[date], *, rate: float = 1300.0) -> pl.DataFrame:
    return pl.DataFrame({"date": list(sessions), "usdkrw": [rate] * len(sessions)})


def _xnys_sessions(start: date, end: date) -> list[date]:
    from src.data.calendar import load_calendar

    return list(load_calendar("XNYS").sessions(start, end))


def _xkrx_sessions(start: date, end: date) -> list[date]:
    from src.data.calendar import load_calendar

    return list(load_calendar("XKRX").sessions(start, end))


def _kr_prices(
    sessions: list[date],
    *,
    ticker: str = "379800",
    dist_on: date | None = None,
    dist_pay: date | None = None,
    split_on: date | None = None,
) -> pl.DataFrame:
    data: dict[str, list[object]] = {
        "ticker": [],
        "date": [],
        "close_krw": [],
        "nav_krw": [],
        "distribution_krw": [],
        "distribution_pay_date": [],
        "split_factor": [],
        "volume": [],
    }
    for index, day in enumerate(sessions):
        close = 10000.0 + 2.0 * index
        factor = 2.0 if split_on is not None and day == split_on else 1.0
        if split_on is not None and day >= split_on:
            close /= 2.0
        data["ticker"].append(ticker)
        data["date"].append(day)
        data["close_krw"].append(close)
        data["nav_krw"].append(close)
        data["distribution_krw"].append(100.0 if dist_on is not None and day == dist_on else 0.0)
        data["distribution_pay_date"].append(dist_pay if dist_on is not None and day == dist_on else None)
        data["split_factor"].append(factor)
        data["volume"].append(1000)
    return pl.DataFrame(
        data,
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "close_krw": pl.Float64,
            "nav_krw": pl.Float64,
            "distribution_krw": pl.Float64,
            "distribution_pay_date": pl.Date,
            "split_factor": pl.Float64,
            "volume": pl.Int64,
        },
    )


def _config(
    *,
    start: date = date(2023, 1, 1),
    end: date = date(2024, 12, 31),
    targets: dict[str, float] | None = None,
    cash: dict[int, int] | None = None,
    cash_events: dict[date, int] | None = None,
    dates: dict[int, tuple[date, ...]] | None = None,
    retirement: int = 2030,
    withdrawals: dict[int, int] | None = None,
    mode: PensionMarketMode = PensionMarketMode.US_PROXY,
    max_fx_age_days: int = 7,
    spread: float = 0.0,
    commission: float = 0.0,
    drag: dict[str, float] | None = None,
) -> PensionBacktestConfig:
    return PensionBacktestConfig(
        start=start,
        end=end,
        targets=dict(targets) if targets is not None else {"SPY": 1.0},
        available_cash_events_krw=dict(cash_events) if cash_events is not None else {
            max(start, date(year, 1, 3)): amount
            for year, amount in (dict(cash) if cash is not None else {2023: 6_000_000, 2024: 6_000_000}).items()
        },
        contribution_dates=dict(dates) if dates is not None else {},
        tax_credit_settlement_dates={year: date(year + 1, 5, 31) for year in range(start.year, end.year + 1)},
        retirement_start_year=retirement,
        withdrawal_amounts_krw=dict(withdrawals) if withdrawals is not None else {},
        market_mode=mode,
        max_fx_age_days=max_fx_age_days,
        execution_spread_bps=spread,
        commission_bps=commission,
        extra_annual_drag_by_ticker=dict(drag) if drag is not None else {},
    )


def _monthly_dates(year: int) -> tuple[date, ...]:
    return tuple(date(year, month, 15) for month in range(1, 13))


def test_same_cashflow_schedules_share_credit_and_never_overspend() -> None:
    """Lump-sum and monthly schedules earn the same credit without spending early cash."""
    sessions = _xnys_sessions(date(2023, 1, 1), date(2024, 12, 31))
    prices = _us_prices(sessions)
    fx = _fx(sessions)
    cash = {2023: 6_000_000, 2024: 6_000_000}
    lump = _config(dates={2023: (date(2023, 1, 3),), 2024: (date(2024, 1, 3),)}, cash=cash)
    monthly = _config(dates={2023: _monthly_dates(2023), 2024: _monthly_dates(2024)}, cash=cash)
    first = run_pension_backtest(lump, prices, fx, _profile(), _regime())
    second = run_pension_backtest(monthly, prices, fx, _profile(), _regime())
    assert [c.national_credit_krw for c in first.tax_credits] == [c.national_credit_krw for c in second.tax_credits]
    assert [c.local_credit_krw for c in first.tax_credits] == [c.local_credit_krw for c in second.tax_credits]
    for result in (first, second):
        assert all(snap.cash_krw >= 0 for snap in result.snapshots)
        assert result.snapshots[-1].cumulative_contributions_krw == 12_000_000
        assert result.tax_credits[0].contributed_krw == 6_000_000


def test_monthly_income_cannot_fund_early_lump_and_refund_waits_for_settlement() -> None:
    """Only received cash can fund contributions; tax credits arrive after tax-year close."""
    sessions = _xnys_sessions(date(2023, 1, 1), date(2024, 12, 31))
    cash_events = {day: 500_000 for year in (2023, 2024) for day in _monthly_dates(year)}
    early = _config(
        cash_events=cash_events,
        dates={2023: (date(2023, 1, 15),), 2024: (date(2024, 1, 15),)},
    )
    with pytest.raises(ValueError, match="exceeds cash available by that date"):
        run_pension_backtest(early, _us_prices(sessions), _fx(sessions), _profile(), _regime())

    monthly = _config(cash_events=cash_events, dates={year: _monthly_dates(year) for year in (2023, 2024)})
    year_end = _config(
        cash_events=cash_events,
        dates={year: (date(year, 12, 15),) for year in (2023, 2024)},
    )
    monthly_result = run_pension_backtest(monthly, _us_prices(sessions), _fx(sessions), _profile(), _regime())
    year_end_result = run_pension_backtest(year_end, _us_prices(sessions), _fx(sessions), _profile(), _regime())
    assert [quote.contributed_krw for quote in monthly_result.tax_credits] == [6_000_000, 6_000_000]
    assert [quote.national_credit_krw for quote in monthly_result.tax_credits] == [
        quote.national_credit_krw for quote in year_end_result.tax_credits
    ]
    assert monthly_result.after_tax_external_cashflows_krw == ((date(2024, 5, 31), 990_000),)


def test_contributions_stop_at_the_smallest_amount_earning_all_usable_credit() -> None:
    """Zero, partial, and sufficient tax capacity produce distinct funded account balances."""
    sessions = _xnys_sessions(date(2023, 1, 1), date(2023, 12, 31))
    config = _config(
        start=date(2023, 1, 1), end=date(2023, 12, 31),
        cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 15),)},
    )
    outcomes = []
    for national, local in ((0, 0), (450_000, 45_000), (2_000_000, 2_000_000)):
        profile = _profile(years=(2023,), national=national, local=local)
        outcomes.append(run_pension_backtest(config, _us_prices(sessions), _fx(sessions), profile, _regime()))
    assert [result.tax_credits[0].contributed_krw for result in outcomes] == [0, 3_000_000, 6_000_000]
    assert [result.tax_credits[0].national_credit_krw for result in outcomes] == [0, 450_000, 900_000]
    assert [result.snapshots[-1].cumulative_contributions_krw for result in outcomes] == [0, 3_000_000, 6_000_000]
    assert outcomes[0].terminal_nav_krw == 0


def test_first_year_payout_uses_commencement_value() -> None:
    """The first pension-year limit uses the value when commencement takes effect."""
    sessions = _xnys_sessions(date(2023, 1, 1), date(2023, 12, 31))
    config = _config(
        start=date(2023, 1, 1), end=date(2023, 12, 31),
        cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 15),)},
        retirement=2023, withdrawals={2023: 1_000_000},
    )
    profile = _profile(birth=date(1960, 1, 1), opened=date(2018, 1, 1), years=(2023,), pension_start=date(2023, 12, 1))
    with pytest.raises(ValueError, match="pension-year-1 annual limit"):
        run_pension_backtest(config, _us_prices(sessions), _fx(sessions), profile, _regime())
    within = replace(config, withdrawal_amounts_krw={2023: 500_000})
    result = run_pension_backtest(within, _us_prices(sessions), _fx(sessions), profile, _regime())
    assert result.withdrawals[0].gross_withdrawal_krw == 500_000


def test_eligible_payout_does_not_hide_missing_tax_profile_data() -> None:
    """A missing pension-income input is an error even when age and account tenure qualify."""
    sessions = _xnys_sessions(date(2023, 1, 1), date(2023, 12, 31))
    config = _config(
        start=date(2023, 1, 1), end=date(2023, 12, 31),
        cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 15),)},
        retirement=2023, withdrawals={2023: 1_000_000},
    )
    profile = _profile(birth=date(1960, 1, 1), opened=date(2000, 1, 1), years=(2023,), pension_start=date(2023, 12, 1))
    object.__setattr__(profile, "other_private_pension_income_krw", {})
    with pytest.raises(ValueError, match="other_private_pension_income_krw has no entry"):
        run_pension_backtest(config, _us_prices(sessions), _fx(sessions), profile, _regime())


def test_payout_eligibility_waits_for_birth_and_account_anniversaries() -> None:
    """A December payout before either anniversary remains an intermediate account value."""
    sessions = _xnys_sessions(date(2024, 1, 1), date(2024, 12, 31))
    config = _config(
        start=date(2024, 1, 1), end=date(2024, 12, 31),
        cash={2024: 6_000_000}, dates={2024: (date(2024, 1, 15),)},
        retirement=2024, withdrawals={2024: 1_000_000},
    )
    for birth, opened in ((date(1969, 12, 31), date(2000, 1, 1)),
                          (date(1960, 1, 1), date(2019, 12, 31))):
        profile = _profile(birth=birth, opened=opened, years=(2024,))
        result = run_pension_backtest(config, _us_prices(sessions), _fx(sessions), profile, _regime())
        assert result.withdrawals == ()
        assert result.is_retirement_terminal is False


def test_payout_waits_for_pension_commencement_date() -> None:
    """A payout day before the commencement application takes effect stays an intermediate value."""
    sessions = _xnys_sessions(date(2024, 1, 1), date(2024, 6, 30))
    config = _config(
        start=date(2024, 1, 1), end=date(2024, 6, 30),
        cash={2024: 6_000_000}, dates={2024: (date(2024, 1, 15),)},
        retirement=2024, withdrawals={2024: 1_000_000},
    )
    profile = _profile(birth=date(1960, 1, 1), opened=date(2000, 1, 1), years=(2024,), pension_start=date(2024, 12, 1))
    result = run_pension_backtest(config, _us_prices(sessions), _fx(sessions), profile, _regime())
    assert result.withdrawals == ()
    assert result.is_retirement_terminal is False


def test_contribution_requires_an_open_pension_account() -> None:
    """A funded contribution cannot precede the declared pension account opening date."""
    sessions = _xnys_sessions(date(2023, 1, 1), date(2023, 12, 31))
    config = _config(
        start=date(2023, 1, 1), end=date(2023, 12, 31),
        cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 15),)},
    )
    closed = _profile(opened=date(2023, 2, 1), years=(2023,))
    with pytest.raises(ValueError, match="precedes account opening"):
        run_pension_backtest(config, _us_prices(sessions), _fx(sessions), closed, _regime())
    opened = _profile(opened=date(2023, 1, 15), years=(2023,))
    result = run_pension_backtest(config, _us_prices(sessions), _fx(sessions), opened, _regime())
    assert result.contribution_cashflows_krw == ((date(2023, 1, 15), 6_000_000),)


def test_contribution_after_pension_commencement_is_rejected() -> None:
    """Contributions after commencement violate the ordinary pension-account conditions."""
    sessions = _xnys_sessions(date(2023, 1, 1), date(2024, 12, 31))
    config = _config(
        dates={2023: (date(2023, 1, 15),), 2024: (date(2024, 12, 15),)},
        retirement=2024, withdrawals={2024: 1_000_000},
    )
    profile = _profile(birth=date(1960, 1, 1), opened=date(2000, 1, 1), pension_start=date(2024, 12, 1))
    with pytest.raises(ValueError, match="follows pension commencement"):
        run_pension_backtest(config, _us_prices(sessions), _fx(sessions), profile, _regime())


def test_internal_flows_create_no_general_account_tax() -> None:
    """Distributions and payout sales settle only through the pension tax quotes."""
    sessions = _xkrx_sessions(date(2023, 1, 1), date(2023, 12, 31))
    ex = sessions[len(sessions) // 2]
    pay = sessions[len(sessions) // 2 + 5]
    prices = _kr_prices(sessions, dist_on=ex, dist_pay=pay, split_on=sessions[len(sessions) // 4])
    profile = _profile(birth=date(1960, 1, 1), opened=date(2000, 1, 1), years=(2023,), pension_start=date(2023, 12, 1))
    config = PensionBacktestConfig(
        start=date(2023, 1, 1),
        end=date(2023, 12, 31),
        targets={"379800": 1.0},
        available_cash_events_krw={date(2023, 1, 3): 6_000_000},
        contribution_dates={2023: (date(2023, 1, 10),)},
        tax_credit_settlement_dates={2023: date(2024, 5, 31)},
        retirement_start_year=2023,
        withdrawal_amounts_krw={2023: 1_000_000},
        market_mode=PensionMarketMode.KR_LIVE,
        max_fx_age_days=7,
        execution_spread_bps=0.0,
        commission_bps=0.0,
        extra_annual_drag_by_ticker={},
    )
    result = run_pension_backtest(config, prices, None, profile, _regime())
    assert len(result.withdrawals) == 1
    assert result.is_retirement_terminal is False
    quote = result.withdrawals[0]
    prior_navs = [snap.nav_krw for snap in result.snapshots if snap.session < quote.withdrawal_date]
    expected = quote_annual_pension_withdrawal(
        quote.withdrawal_date,
        1_000_000,
        prior_navs[-1],
        0,
        0,
        6_000_000,
        profile,
        _regime(),
    )
    assert (quote.national_tax_krw, quote.local_tax_krw) == (expected.national_tax_krw, expected.local_tax_krw)
    assert not hasattr(result, "taxes_paid_krw")
    assert not hasattr(result, "realized_gain_krw")


def test_live_run_stops_before_listing_instead_of_splicing() -> None:
    """A KR_LIVE window starting before the first listed month aborts the arm."""
    sessions = _xkrx_sessions(date(2023, 1, 1), date(2023, 12, 31))
    prices = _kr_prices(sessions)
    config = _config(start=date(2020, 1, 1), end=date(2023, 12, 31), mode=PensionMarketMode.KR_LIVE,
                     targets={"379800": 1.0}, cash=dict.fromkeys((2020, 2021, 2022, 2023), 6_000_000))
    with pytest.raises(PensionDataError, match="refusing to splice"):
        run_pension_backtest(config, prices, None, _profile(years=(2020, 2021, 2022, 2023)), _regime())


def test_distribution_cash_enters_only_on_pay_date() -> None:
    """Live distribution cash appears on the pay date and NAV reconciles throughout."""
    sessions = _xkrx_sessions(date(2023, 1, 1), date(2023, 6, 30))
    ex = sessions[len(sessions) // 2]
    pay = sessions[len(sessions) // 2 + 4]
    prices = _kr_prices(sessions, dist_on=ex, dist_pay=pay)
    by_session = {row["date"]: row for row in prices.to_dicts()}
    config = _config(start=date(2023, 1, 1), end=date(2023, 6, 30), mode=PensionMarketMode.KR_LIVE,
                     targets={"379800": 1.0}, cash={2023: 6_000_000},
                     dates={2023: (date(2023, 1, 10),)})
    result = run_pension_backtest(config, prices, None, _profile(years=(2023,)), _regime())
    before = next(snap for snap in result.snapshots if snap.session == pay)
    prior = result.snapshots[[snap.session for snap in result.snapshots].index(pay) - 1]
    units = before.units_by_ticker["379800"]
    assert before.cash_krw - prior.cash_krw == pytest.approx(100.0 * units, abs=2.0)
    for snap in result.snapshots:
        marked = snap.units_by_ticker["379800"] * by_session[snap.session]["close_krw"]
        accrued = 100.0 * units if ex <= snap.session < pay else 0.0
        assert snap.distribution_receivable_krw == pytest.approx(accrued, abs=1.0)
        assert abs(snap.nav_krw - (snap.cash_krw + marked + accrued)) <= 1.0


def test_fully_paid_account_is_the_only_retirement_terminal() -> None:
    """A complete legal payout leaves no deferred pension balance at the terminal mark."""
    sessions = _xnys_sessions(date(2023, 1, 1), date(2023, 12, 31))
    prices = _us_prices(sessions).with_columns(pl.lit(400.0).alias("adjusted_close"))
    config = _config(
        start=date(2023, 1, 1), end=date(2023, 12, 31),
        cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 15),)},
        retirement=2023, withdrawals={2023: 6_000_000},
    )
    profile = _profile(birth=date(1960, 1, 1), opened=date(2000, 1, 1), years=(2023,), pension_start=date(2023, 12, 1))
    result = run_pension_backtest(config, prices, _fx(sessions), profile, _regime())
    assert result.terminal_nav_krw == 0
    assert result.payout_shortfalls_krw == ()
    assert result.is_retirement_terminal is True


def test_path_ending_before_55_is_intermediate() -> None:
    """A 20-year path ending before age 55 carries no pension payout tax."""
    sessions = _xnys_sessions(date(2005, 1, 1), date(2024, 12, 31))
    prices = _us_prices(sessions)
    fx = _fx(sessions)
    years = tuple(range(2005, 2025))
    profile = _profile(birth=date(1980, 6, 1), opened=date(2000, 1, 1), years=years)
    config = _config(start=date(2005, 1, 1), end=date(2024, 12, 31),
                     cash=dict.fromkeys(years, 6_000_000),
                     dates={year: (date(year, 1, 15),) for year in years},
                     retirement=2020, withdrawals=dict.fromkeys(range(2020, 2025), 1_000_000))
    result = run_pension_backtest(config, prices, fx, profile, _regime())
    assert result.is_retirement_terminal is False
    assert result.withdrawals == ()


def test_missing_fx_zero_quote_and_double_fee() -> None:
    """Missing FX or a zero quote fails; live NAV never bears operating costs twice."""
    sessions = _xnys_sessions(date(2023, 1, 1), date(2023, 12, 31))
    prices = _us_prices(sessions)
    fx = _fx(sessions)
    config = _config(start=date(2023, 1, 1), end=date(2023, 12, 31), dates={2023: (date(2023, 1, 10),)}, cash={2023: 6_000_000})
    with pytest.raises(PensionDataError, match="USD/KRW"):
        run_pension_backtest(config, prices, None, _profile(years=(2023,)), _regime())
    broken = prices.with_columns(
        pl.when(pl.col("date") == sessions[10]).then(0.0).otherwise(pl.col("adjusted_close")).alias("adjusted_close")
    )
    with pytest.raises(PensionDataError, match="missing or zero"):
        run_pension_backtest(config, broken, fx, _profile(years=(2023,)), _regime())

    live_sessions = _xkrx_sessions(date(2023, 1, 1), date(2023, 12, 31))
    live = _kr_prices(live_sessions)
    live_config = _config(start=date(2023, 1, 1), end=date(2023, 12, 31), mode=PensionMarketMode.KR_LIVE,
                          targets={"379800": 1.0}, cash={2023: 6_000_000},
                          dates={2023: (date(2023, 1, 10),)}, spread=5.0, commission=10.0)
    live_result = run_pension_backtest(live_config, live, None, _profile(years=(2023,)), _regime())
    fees = live_result.snapshots[-1].cumulative_fees_krw
    assert fees > 0
    assert fees < 6_000_000 * 0.002
    by_session = {row["date"]: row["close_krw"] for row in live.to_dicts()}
    for snap in live_result.snapshots:
        marked = snap.units_by_ticker["379800"] * by_session[snap.session]
        assert abs(snap.nav_krw - (snap.cash_krw + marked)) <= 1.0


def test_missing_bar_and_drag_and_splits() -> None:
    """A gapped ticker aborts; drag marks a sensitivity; splits preserve units."""
    sessions = _xnys_sessions(date(2023, 1, 1), date(2023, 6, 30))
    prices = _us_prices(sessions, tickers=("SPY", "QQQ"))
    gapped = prices.filter(~((pl.col("ticker") == "QQQ") & (pl.col("date") == sessions[20])))
    fx = _fx(sessions)
    config = _config(start=date(2023, 1, 1), end=date(2023, 6, 30), targets={"SPY": 0.5, "QQQ": 0.5},
                     cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 10),)})
    with pytest.raises(PensionDataError, match="no forward fill"):
        run_pension_backtest(config, gapped, fx, _profile(years=(2023,)), _regime())

    live_sessions = _xkrx_sessions(date(2023, 1, 1), date(2023, 6, 30))
    live = _kr_prices(live_sessions, split_on=live_sessions[50])
    plain_config = _config(start=date(2023, 1, 1), end=date(2023, 6, 30), mode=PensionMarketMode.KR_LIVE,
                           targets={"379800": 1.0}, cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 10),)})
    drag_config = _config(start=date(2023, 1, 1), end=date(2023, 6, 30), mode=PensionMarketMode.KR_LIVE,
                          targets={"379800": 1.0}, cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 10),)},
                          drag={"379800": 0.05})
    plain = run_pension_backtest(plain_config, live, None, _profile(years=(2023,)), _regime())
    dragged = run_pension_backtest(drag_config, live, None, _profile(years=(2023,)), _regime())
    assert dragged.terminal_nav_krw < plain.terminal_nav_krw
    split_day = live_sessions[50]
    pre_split = next(snap for snap in reversed(plain.snapshots) if snap.session < split_day)
    post_split = next(snap for snap in plain.snapshots if snap.session >= split_day)
    assert pre_split.units_by_ticker["379800"] > 0
    assert post_split.units_by_ticker["379800"] == pytest.approx(pre_split.units_by_ticker["379800"] * 2.0)
    next_session = plain.snapshots[[snap.session for snap in plain.snapshots].index(split_day) + 1]
    assert next_session.units_by_ticker["379800"] == pytest.approx(post_split.units_by_ticker["379800"])


def test_split_on_purchase_session_changes_only_preexisting_units() -> None:
    """New shares bought at a post-split close are not split a second time."""
    sessions = _xkrx_sessions(date(2023, 1, 1), date(2023, 3, 31))
    split_day = next(day for day in sessions if day.month == 2)
    prices = _kr_prices(sessions, split_on=split_day)
    config = _config(
        start=date(2023, 1, 1), end=date(2023, 3, 31), mode=PensionMarketMode.KR_LIVE,
        targets={"379800": 1.0}, cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 10),)},
    )
    result = run_pension_backtest(config, prices, None, _profile(years=(2023,)), _regime())
    split_close = next(row["close_krw"] for row in prices.to_dicts() if row["date"] == split_day)
    on_split = next(snap for snap in result.snapshots if snap.session == split_day)
    assert on_split.units_by_ticker["379800"] == math.floor(6_000_000 / split_close)
    assert result.snapshots[-1].units_by_ticker["379800"] == on_split.units_by_ticker["379800"]


def _us_dividend_prices(
    sessions: list[date], *, div_on: date | None = None, div_rate: float = 0.01
) -> pl.DataFrame:
    rows = []
    base = 400.0
    for index, day in enumerate(sessions):
        price = base + 0.1 * index
        rows.append({
            "ticker": "SPY",
            "date": day,
            "close": price,
            "adjusted_close": price,
            "dividend": div_rate * price if div_on is not None and day == div_on else 0.0,
        })
    return pl.DataFrame(rows)


def test_zero_withholding_reproduces_adjusted_marks() -> None:
    """With rate 0.0 and dividends present, marks equal adjusted_close x fx and nothing is withheld."""
    from src.sim.pension_engine import _proxy_marks

    sessions = _xnys_sessions(date(2023, 1, 1), date(2023, 12, 31))
    div_on = sessions[len(sessions) // 2]
    prices = _us_dividend_prices(sessions, div_on=div_on)
    fx = _fx(sessions)
    zero = replace(_regime(), foreign_dividend_withholding_rate=0.0)
    marks, withheld = _proxy_marks(prices, fx, 7, zero.foreign_dividend_withholding_rate)
    assert proxy_krw_marks(
        prices, fx, max_fx_age_days=7, withholding_rate=zero.foreign_dividend_withholding_rate
    ) == marks
    by_date = {day: 400.0 + 0.1 * index for index, day in enumerate(sessions)}
    for day in sessions:
        assert marks[("SPY", day)] == pytest.approx(by_date[day] * 1300.0, rel=1e-12)
    assert all(value == 0.0 for value in withheld.values())
    config = _config(start=date(2023, 1, 1), end=date(2023, 12, 31),
                     cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 15),)})
    result = run_pension_backtest(config, prices, fx, _profile(years=(2023,)), zero)
    assert result.foreign_tax_withheld_krw == 0


def test_withholding_lowers_nav_and_accrues_foreign_tax() -> None:
    """A 1% dividend held at rate 0.15 lowers NAV and accrues 15% of the dividend value."""
    from src.sim.pension_engine import _proxy_marks

    sessions = _xnys_sessions(date(2023, 1, 1), date(2023, 12, 31))
    div_on = sessions[len(sessions) // 2]
    div_index = sessions.index(div_on)
    prices = _us_dividend_prices(sessions, div_on=div_on)
    fx = _fx(sessions)
    config = _config(start=date(2023, 1, 1), end=date(2023, 12, 31),
                     cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 15),)})
    zero = replace(_regime(), foreign_dividend_withholding_rate=0.0)
    plain = run_pension_backtest(config, prices, fx, _profile(years=(2023,)), zero)
    gross = run_pension_backtest(config, prices, fx, _profile(years=(2023,)), _regime())
    assert _regime().foreign_dividend_withholding_rate == pytest.approx(0.15)
    assert gross.terminal_nav_krw < plain.terminal_nav_krw
    assert gross.foreign_tax_withheld_krw > 0
    _, withheld_per_unit = _proxy_marks(prices, fx, 7, 0.15)
    div_price = 400.0 + 0.1 * div_index
    assert withheld_per_unit[("SPY", div_on)] == pytest.approx(0.15 * (0.01 * div_price) * 1300.0, rel=1e-9)
    units_before = plain.snapshots[div_index - 1].units_by_ticker["SPY"]
    expected = units_before * 0.15 * (0.01 * div_price) * 1300.0
    assert gross.foreign_tax_withheld_krw == pytest.approx(expected, abs=2.0)


def test_dividend_before_first_purchase_accrues_nothing() -> None:
    """A dividend paid before any units are held leaves no withheld tax."""
    sessions = _xnys_sessions(date(2023, 1, 1), date(2023, 12, 31))
    prices = _us_dividend_prices(sessions, div_on=sessions[0])
    config = _config(start=date(2023, 1, 1), end=date(2023, 12, 31),
                     cash={2023: 6_000_000}, dates={2023: (date(2023, 6, 15),)})
    result = run_pension_backtest(config, prices, _fx(sessions), _profile(years=(2023,)), _regime())
    assert result.foreign_tax_withheld_krw == 0


def test_proxy_dividend_columns_fail_closed() -> None:
    """Missing close/dividend columns or a null dividend aborts the arm."""
    sessions = _xnys_sessions(date(2023, 1, 1), date(2023, 12, 31))
    prices = _us_dividend_prices(sessions)
    fx = _fx(sessions)
    config = _config(start=date(2023, 1, 1), end=date(2023, 12, 31),
                     cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 15),)})
    profile = _profile(years=(2023,))
    with pytest.raises(PensionDataError, match="required column"):
        run_pension_backtest(config, prices.drop("dividend"), fx, profile, _regime())
    with pytest.raises(PensionDataError, match="required column"):
        run_pension_backtest(config, prices.drop("close"), fx, profile, _regime())
    null_div = prices.with_columns(
        pl.when(pl.col("date") == sessions[10])
        .then(None)
        .otherwise(pl.col("dividend"))
        .alias("dividend")
    )
    with pytest.raises(PensionDataError, match="dividend"):
        run_pension_backtest(config, null_div, fx, profile, _regime())
    null_close = prices.with_columns(
        pl.when(pl.col("date") == sessions[10])
        .then(None)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    with pytest.raises(PensionDataError, match="missing or zero"):
        run_pension_backtest(config, null_close, fx, profile, _regime())
    zero_close = prices.with_columns(
        pl.when(pl.col("date") == sessions[10])
        .then(0.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    with pytest.raises(PensionDataError, match="missing or zero"):
        run_pension_backtest(config, zero_close, fx, profile, _regime())
    text_quote = prices.with_columns(pl.lit("x").alias("adjusted_close"))
    with pytest.raises(PensionDataError, match="missing or zero"):
        run_pension_backtest(config, text_quote, fx, profile, _regime())
    negative_div = prices.with_columns(
        pl.when(pl.col("date") == sessions[10])
        .then(-5.0)
        .otherwise(pl.col("dividend"))
        .alias("dividend")
    )
    with pytest.raises(PensionDataError, match="dividend"):
        run_pension_backtest(config, negative_div, fx, profile, _regime())


def test_future_dividend_perturbation_invariance() -> None:
    """Dividends after T leave every snapshot dated on or before T identical."""
    sessions = _xnys_sessions(date(2023, 1, 1), date(2023, 12, 31))
    cutoff = sessions[len(sessions) // 2]
    base = _us_dividend_prices(sessions)
    shocked = base.with_columns(
        pl.when(pl.col("date") > cutoff)
        .then(50.0)
        .otherwise(pl.col("dividend"))
        .alias("dividend")
    )
    config = _config(start=date(2023, 1, 1), end=date(2023, 12, 31),
                     cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 15),)})
    profile = _profile(years=(2023,))
    first = run_pension_backtest(config, base, _fx(sessions), profile, _regime())
    second = run_pension_backtest(config, shocked, _fx(sessions), profile, _regime())
    assert [snap for snap in first.snapshots if snap.session <= cutoff] == [
        snap for snap in second.snapshots if snap.session <= cutoff
    ]


def test_principal_bases_reported() -> None:
    """Without withdrawals the terminal bases reconcile with cumulative contributions."""
    sessions = _xnys_sessions(date(2023, 1, 1), date(2023, 12, 31))
    config = _config(start=date(2023, 1, 1), end=date(2023, 12, 31),
                     cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 15),)})
    result = run_pension_backtest(config, _us_prices(sessions), _fx(sessions), _profile(years=(2023,)), _regime())
    assert (
        result.terminal_credited_principal_krw + result.terminal_uncredited_principal_krw
        == result.snapshots[-1].cumulative_contributions_krw == 6_000_000
    )


def test_kr_live_reports_zero_withheld() -> None:
    """The live fixture accrues no foreign withholding."""
    live_sessions = _xkrx_sessions(date(2023, 1, 1), date(2023, 12, 31))
    config = _config(start=date(2023, 1, 1), end=date(2023, 12, 31), mode=PensionMarketMode.KR_LIVE,
                     targets={"379800": 1.0}, cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 10),)})
    result = run_pension_backtest(config, _kr_prices(live_sessions), None, _profile(years=(2023,)), _regime())
    assert result.foreign_tax_withheld_krw == 0


def test_engine_inputs_fail_closed() -> None:
    """Malformed configs, gaps, and illegal payouts never produce a result."""
    sessions = _xnys_sessions(date(2023, 1, 1), date(2023, 12, 31))
    prices = _us_prices(sessions)
    fx = _fx(sessions)
    profile = _profile(years=(2023,))
    regime = _regime()
    base_cash = {2023: 6_000_000}
    base_dates = {2023: (date(2023, 1, 10),)}
    bad_configs = [
        (_config(start=date(2024, 1, 1), end=date(2023, 1, 1)), "after end"),
        (_config(targets={}), "non-empty"),
        (_config(targets={"SPY": "x"}), "numeric weight"),
        (_config(targets={"SPY": 0.0}), "lie in"),
        (_config(targets={"SPY": 0.5, "QQQ": 0.4}), "sum to 1"),
        (_config(spread=-1.0), "must lie in"),
        (_config(max_fx_age_days=-1), "max_fx_age_days"),
        (_config(spread="x"), "finite nonnegative"),
        (_config(drag={"SPY": 2.0}), "must lie in"),
        (_config(retirement="x"), "integer year"),
        (_config(withdrawals={2023: -1}), "nonnegative integer"),
        (_config(withdrawals={2025: 100}), "beyond the backtest end"),
        (_config(retirement=2024, withdrawals={2023: 100}), "precedes the retirement"),
        (_config(dates={2023: (date(2022, 1, 1),)}), "outside the backtest window"),
    ]
    for bad, match in bad_configs:
        with pytest.raises(ValueError, match=match):
            run_pension_backtest(bad, prices, fx, profile, regime)
    bad_mode = _config()
    object.__setattr__(bad_mode, "market_mode", "proxy")
    with pytest.raises(ValueError, match="unsupported"):
        run_pension_backtest(bad_mode, prices, fx, profile, regime)
    with pytest.raises(ValueError, match="available cash date"):
        run_pension_backtest(_config(cash_events={date(2022, 1, 3): 1}), prices, fx, profile, regime)
    missing_settlement = _config()
    object.__setattr__(missing_settlement, "tax_credit_settlement_dates", {})
    with pytest.raises(ValueError, match="explicit later settlement date"):
        run_pension_backtest(missing_settlement, prices, fx, profile, regime)
    with pytest.raises(ValueError, match="nonnegative integer"):
        run_pension_backtest(_config(cash={2023: -1}), prices, fx, profile, regime)
    with pytest.raises(PensionDataError, match="empty"):
        run_pension_backtest(_config(), prices.head(0), fx, profile, regime)
    with pytest.raises(PensionDataError, match="ticker/date"):
        run_pension_backtest(_config(), prices.drop("ticker"), fx, profile, regime)
    with pytest.raises(PensionDataError, match="no session inside"):
        run_pension_backtest(_config(start=date(2020, 1, 1), end=date(2020, 12, 31), cash={2020: 6_000_000}),
                             prices, fx, _profile(years=(2020,)), regime)
    with pytest.raises(PensionDataError, match="no row for target"):
        run_pension_backtest(_config(targets={"VTI": 1.0}), prices, fx, profile, regime)
    with pytest.raises(PensionDataError, match="required column"):
        run_pension_backtest(_config(), prices.drop("adjusted_close"), fx, profile, regime)
    with pytest.raises(PensionDataError, match="fx misses"):
        run_pension_backtest(_config(), prices, fx.drop("usdkrw"), profile, regime)
    with pytest.raises(PensionDataError, match="fx frame is empty"):
        run_pension_backtest(_config(), prices, fx.head(0), profile, regime)
    with pytest.raises(PensionDataError, match="nonpositive quote"):
        run_pension_backtest(_config(), prices, _fx(sessions, rate=0.0), profile, regime)
    with pytest.raises(PensionDataError, match="missing on or before"):
        run_pension_backtest(_config(), prices, _fx(sessions[100:]), profile, regime)
    with pytest.raises(PensionDataError, match="stale"):
        run_pension_backtest(_config(max_fx_age_days=7), prices, _fx(sessions[:1]), profile, regime)
    with pytest.raises(ValueError, match="takes no fx"):
        run_pension_backtest(
            _config(start=date(2023, 1, 1), end=date(2023, 12, 31), mode=PensionMarketMode.KR_LIVE,
                    targets={"379800": 1.0}, cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 10),)}),
            _kr_prices(_xkrx_sessions(date(2023, 1, 1), date(2023, 12, 31))),
            fx, profile, regime,
        )
    live_only = _kr_prices(_xkrx_sessions(date(2023, 1, 1), date(2023, 12, 31)))
    live_config = _config(start=date(2023, 1, 1), end=date(2023, 12, 31), mode=PensionMarketMode.KR_LIVE,
                    targets={"379800": 1.0}, cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 10),)})
    with pytest.raises(PensionDataError, match="required column"):
        run_pension_backtest(live_config, live_only.drop("close_krw"), None, profile, regime)
    zero_close = live_only.with_columns(
        pl.when(pl.col("date") == live_only.get_column("date").to_list()[10])
        .then(0.0).otherwise(pl.col("close_krw")).alias("close_krw")
    )
    with pytest.raises(PensionDataError, match="missing or zero"):
        run_pension_backtest(live_config, zero_close, None, profile, regime)
    zero_split = live_only.with_columns(
        pl.when(pl.col("date") == live_only.get_column("date").to_list()[10])
        .then(0.0).otherwise(pl.col("split_factor")).alias("split_factor")
    )
    with pytest.raises(PensionDataError, match="split factor"):
        run_pension_backtest(live_config, zero_split, None, profile, regime)
    negative_dist = live_only.with_columns(
        pl.when(pl.col("date") == live_only.get_column("date").to_list()[10])
        .then(-5.0).otherwise(pl.col("distribution_krw")).alias("distribution_krw")
    )
    with pytest.raises(PensionDataError, match="distribution for"):
        run_pension_backtest(live_config, negative_dist, None, profile, regime)
    bad_pay = live_only.with_columns(
        pl.when(pl.col("date") == live_only.get_column("date").to_list()[50])
        .then(50.0).otherwise(pl.col("distribution_krw")).alias("distribution_krw")
    )
    with pytest.raises(PensionDataError, match="no valid pay date"):
        run_pension_backtest(live_config, bad_pay, None, profile, regime)
    retiree = _profile(birth=date(1960, 1, 1), opened=date(2005, 1, 1), years=(2023,), pension_start=date(2023, 12, 1))
    shortage = run_pension_backtest(
        _config(start=date(2023, 1, 1), end=date(2023, 12, 31), retirement=2023,
                withdrawals={2023: 500_000_000}, dates={2023: (date(2023, 1, 10),)},
                cash={2023: 6_000_000}),
        prices, fx, retiree, regime,
    )
    assert shortage.withdrawals[0].gross_withdrawal_krw + shortage.payout_shortfalls_krw[0][1] == 500_000_000
    assert shortage.is_retirement_terminal is False
    with pytest.raises(PensionDataError, match="no executable session"):
        run_pension_backtest(
            _config(start=date(2023, 12, 1), end=date(2023, 12, 31), retirement=2023,
                    withdrawals={2023: 100}, cash={2023: 0}, dates={}),
            _us_prices(_xnys_sessions(date(2023, 12, 1), date(2023, 12, 31))),
            _fx(_xnys_sessions(date(2023, 12, 1), date(2023, 12, 31))),
            _profile(birth=date(1960, 1, 1), opened=date(2010, 1, 1), years=(2023,)), regime,
        )


def test_proxy_krw_marks_matches_engine_reference() -> None:
    """Extracted mark builders are the engine's single mark source."""
    import src.sim.pension_engine as engine_module
    import src.sim.pension_marks as marks_module

    assert engine_module.proxy_krw_marks is marks_module.proxy_krw_marks
    assert engine_module.PensionDataError is marks_module.PensionDataError

    sessions = _xnys_sessions(date(2023, 1, 2), date(2023, 3, 31))
    prices = _us_prices(sessions)
    fx = _fx(sessions)
    marks = proxy_krw_marks(prices, fx, max_fx_age_days=7, withholding_rate=0.15)
    assert marks[("SPY", sessions[0])] == pytest.approx(400.0 * 1300.0)
    assert marks_module.proxy_krw_marks(prices, fx, max_fx_age_days=7, withholding_rate=0.15) == marks


def test_mark_modes_carry_distinct_source_labels() -> None:
    """Proxy and live runs keep their evidence labels on identical economics."""
    xnys = _xnys_sessions(date(2023, 1, 1), date(2023, 12, 31))
    xkrx = _xkrx_sessions(date(2023, 1, 1), date(2023, 12, 31))
    profile = _profile(years=(2023,))
    regime = _regime()

    proxy_result = run_pension_backtest(
        _config(start=date(2023, 1, 1), end=date(2023, 12, 31),
               cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 15),)}),
        _us_prices(xnys),
        _fx(xnys),
        profile,
        regime,
    )
    live_result = run_pension_backtest(
        _config(start=date(2023, 1, 1), end=date(2023, 12, 31), mode=PensionMarketMode.KR_LIVE,
               targets={"379800": 1.0}, cash={2023: 6_000_000}, dates={2023: (date(2023, 1, 10),)}),
        _kr_prices(xkrx),
        None,
        profile,
        regime,
    )
    assert proxy_result.market_mode is PensionMarketMode.US_PROXY
    assert live_result.market_mode is PensionMarketMode.KR_LIVE
    assert proxy_result.terminal_nav_krw > 0
    assert live_result.terminal_nav_krw > 0
    assert live_result.foreign_tax_withheld_krw == 0


def test_proxy_marks_rejects_missing_fx_frame() -> None:
    """Proxy marks without an FX frame fail closed before any fill."""
    sessions = _xnys_sessions(date(2023, 1, 2), date(2023, 3, 31))
    with pytest.raises(PensionDataError):
        proxy_krw_marks(_us_prices(sessions), None, max_fx_age_days=7, withholding_rate=0.15)  # type: ignore[arg-type]
