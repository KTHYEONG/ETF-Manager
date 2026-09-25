"""Buy-only personal-pension backtest engine over certified market-mode rows."""

from __future__ import annotations

import bisect
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final, cast

import polars as pl

from src.sim.pension_tax import (
    PensionCreditQuote,
    PensionTaxProfile,
    PensionTaxRegime,
    PensionWithdrawalQuote,
    quote_annual_pension_credit,
    quote_annual_pension_withdrawal,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PensionBacktestConfig",
    "PensionBacktestResult",
    "PensionDataError",
    "PensionMarketMode",
    "PensionSnapshot",
    "run_pension_backtest",
]

_BPS: Final[float] = 10_000.0
_WEIGHT_SUM_TOLERANCE: Final[float] = 1e-9
_DAYS_PER_YEAR: Final[float] = 365.0


class PensionDataError(RuntimeError):
    """Missing sessions, prices, FX, or payout rows; the affected arm aborts, never fills forward."""


class PensionMarketMode(StrEnum):
    """Evidence source of the price panel; proxy exposure is never live fund history."""

    US_PROXY = "us_proxy"
    KR_LIVE = "kr_live"


@dataclass(frozen=True, slots=True)
class PensionBacktestConfig:
    """Dated cash availability, buy-only targets, market source, and optional legal payout schedule."""

    start: date
    end: date
    targets: Mapping[str, float]
    available_cash_events_krw: Mapping[date, int]
    contribution_dates: Mapping[int, tuple[date, ...]]
    tax_credit_settlement_dates: Mapping[int, date]
    retirement_start_year: int
    withdrawal_amounts_krw: Mapping[int, int]
    market_mode: PensionMarketMode
    max_fx_age_days: int
    execution_spread_bps: float
    commission_bps: float
    extra_annual_drag_by_ticker: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class PensionSnapshot:
    """One dated account valuation whose cash and asset units reconcile."""

    session: date
    cash_krw: int
    distribution_receivable_krw: int
    units_by_ticker: Mapping[str, float]
    nav_krw: int
    cumulative_contributions_krw: int
    cumulative_fees_krw: int


@dataclass(frozen=True, slots=True)
class PensionBacktestResult:
    """Reconciled pension path and dated tax flows with an explicit evidence label."""

    snapshots: tuple[PensionSnapshot, ...]
    contribution_cashflows_krw: tuple[tuple[date, int], ...]
    tax_credits: tuple[PensionCreditQuote, ...]
    withdrawals: tuple[PensionWithdrawalQuote, ...]
    payout_shortfalls_krw: tuple[tuple[date, int], ...]
    terminal_nav_krw: int
    after_tax_external_cashflows_krw: tuple[tuple[date, int], ...]
    is_retirement_terminal: bool
    market_mode: PensionMarketMode
    terminal_credited_principal_krw: int
    terminal_uncredited_principal_krw: int
    foreign_tax_withheld_krw: int


def _validate_config(config: PensionBacktestConfig) -> tuple[str, ...]:
    if config.start > config.end:
        raise ValueError(f"pension start {config.start.isoformat()} is after end {config.end.isoformat()}")
    if not isinstance(config.market_mode, PensionMarketMode):
        raise ValueError(f"pension market_mode {config.market_mode!r} is unsupported")
    if isinstance(config.max_fx_age_days, bool) or not isinstance(config.max_fx_age_days, int) or config.max_fx_age_days < 0:
        raise ValueError("pension max_fx_age_days must be a nonnegative integer")
    if not config.targets:
        raise ValueError("pension targets must be non-empty")
    total_weight = 0.0
    for ticker, weight in config.targets.items():
        if not ticker or isinstance(weight, bool) or not isinstance(weight, float | int):
            raise ValueError(f"pension target {ticker!r} must carry a numeric weight")
        if not 0.0 < float(weight) <= 1.0:
            raise ValueError(f"pension target {ticker!r} weight must lie in (0, 1]")
        total_weight += float(weight)
    if abs(total_weight - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"pension targets must sum to 1, got {total_weight!r}")
    for name, value in (("execution_spread_bps", config.execution_spread_bps), ("commission_bps", config.commission_bps)):
        if isinstance(value, bool) or not isinstance(value, float | int):
            raise ValueError(f"pension {name} must be a finite nonnegative number")
        if not 0.0 <= float(value) < _BPS:
            raise ValueError(f"pension {name} must lie in [0, 10000)")
    for ticker, drag in config.extra_annual_drag_by_ticker.items():
        if isinstance(drag, bool) or not isinstance(drag, float | int) or not 0.0 <= float(drag) < 1.0:
            raise ValueError(f"pension drag for {ticker!r} must lie in [0, 1)")
    if isinstance(config.retirement_start_year, bool) or not isinstance(config.retirement_start_year, int):
        raise ValueError("pension retirement_start_year must be an integer year")
    for year, amount in config.withdrawal_amounts_krw.items():
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError(f"pension withdrawal for {year} must be a nonnegative integer")
        if year > config.end.year:
            raise ValueError(f"pension withdrawal for {year} lies beyond the backtest end")
        if amount > 0 and year < config.retirement_start_year:
            raise ValueError(f"pension withdrawal for {year} precedes the retirement start year")
    for day, amount in config.available_cash_events_krw.items():
        if not isinstance(day, date) or day < config.start or day > config.end:
            raise ValueError(f"pension available cash date {day!r} lies outside the backtest window")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError(f"pension available cash on {day.isoformat()} must be a nonnegative integer")
    for year, dates in config.contribution_dates.items():
        for day in dates:
            if day.year != year or day < config.start or day > config.end:
                raise ValueError(f"pension contribution date {day.isoformat()} lies outside the backtest window")
    for year in range(config.start.year, config.end.year + 1):
        settlement = config.tax_credit_settlement_dates.get(year)
        if settlement is None or not isinstance(settlement, date) or settlement <= date(year, 12, 31):
            raise ValueError(f"pension tax credit for {year} requires an explicit later settlement date")
    return tuple(config.targets)


def _proxy_marks(
    prices: pl.DataFrame, fx: pl.DataFrame | None, max_fx_age_days: int, withholding_rate: float
) -> tuple[dict[tuple[str, date], float], dict[tuple[str, date], float]]:
    """Build net-of-withholding KRW marks and per-unit withheld tax for US proxies."""
    if fx is None:
        raise PensionDataError("US_PROXY mode requires an as-of USD/KRW frame")
    for column in ("ticker", "date", "close", "adjusted_close", "dividend"):
        if column not in prices.columns:
            raise PensionDataError(f"US_PROXY prices miss required column {column!r}")
    for column in ("date", "usdkrw"):
        if column not in fx.columns:
            raise PensionDataError(f"US_PROXY fx misses required column {column!r}")
    fx_rows = sorted(fx.select("date", "usdkrw").to_dicts(), key=lambda row: row["date"])
    if not fx_rows:
        raise PensionDataError("US_PROXY fx frame is empty")
    if any(row["usdkrw"] is None or row["usdkrw"] <= 0 for row in fx_rows):
        raise PensionDataError("US_PROXY fx carries a missing or nonpositive quote")
    fx_dates = [row["date"] for row in fx_rows]
    fx_values = [float(row["usdkrw"]) for row in fx_rows]

    def _fx_at(day: date) -> float:
        position = bisect.bisect_right(fx_dates, day) - 1
        if position < 0:
            raise PensionDataError(f"US_PROXY fx is missing on or before {day.isoformat()}")
        if (day - fx_dates[position]).days > max_fx_age_days:
            raise PensionDataError(
                f"US_PROXY fx quote on {fx_dates[position].isoformat()} is stale for {day.isoformat()}"
            )
        return fx_values[position]

    marks: dict[tuple[str, date], float] = {}
    withheld: dict[tuple[str, date], float] = {}
    rows = prices.select("ticker", "date", "close", "adjusted_close", "dividend").to_dicts()
    by_ticker: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_ticker.setdefault(str(row["ticker"]), []).append(row)
    for ticker, ticker_rows in by_ticker.items():
        ordered = sorted(ticker_rows, key=lambda row: row["date"])  # type: ignore[arg-type,return-value]
        index: float | None = None
        prev_close = 0.0
        prev_index = 0.0
        prev_quote = 0.0
        for row in ordered:
            day = cast("date", row["date"])
            close = row["close"]
            quote = row["adjusted_close"]
            dividend = row["dividend"]
            if isinstance(close, bool) or not isinstance(close, float | int):
                raise PensionDataError(f"US_PROXY close for {ticker!r} on {day!r} is missing or zero")
            if not math.isfinite(close) or close <= 0:
                raise PensionDataError(f"US_PROXY close for {ticker!r} on {day!r} is missing or zero")
            if isinstance(quote, bool) or not isinstance(quote, float | int):
                raise PensionDataError(f"US_PROXY price for {ticker!r} on {day!r} is missing or zero")
            if not math.isfinite(quote) or quote <= 0:
                raise PensionDataError(f"US_PROXY price for {ticker!r} on {day!r} is missing or zero")
            if isinstance(dividend, bool) or not isinstance(dividend, float | int):
                raise PensionDataError(f"US_PROXY dividend for {ticker!r} on {day!r} is missing")
            if not math.isfinite(dividend) or dividend < 0:
                raise PensionDataError(f"US_PROXY dividend for {ticker!r} on {day!r} is missing")
            fx_rate = _fx_at(day)
            if index is None:
                index = float(quote)
            else:
                ratio = float(quote) / prev_quote
                drag = withholding_rate * float(dividend) / prev_close
                index = prev_index * (ratio - drag)
                if float(dividend) > 0:
                    withheld[(ticker, day)] = (
                        withholding_rate * float(dividend) / prev_close * prev_index * fx_rate
                    )
            marks[(ticker, day)] = index * fx_rate
            prev_quote = float(quote)
            prev_close = float(close)
            prev_index = index
    return marks, withheld


def _live_marks(
    prices: pl.DataFrame,
) -> tuple[dict[tuple[str, date], float], dict[tuple[str, date], float], dict[tuple[str, date], date | None], dict[tuple[str, date], float]]:
    for column in ("ticker", "date", "close_krw", "distribution_krw", "distribution_pay_date", "split_factor"):
        if column not in prices.columns:
            raise PensionDataError(f"KR_LIVE prices miss required column {column!r}")
    closes: dict[tuple[str, date], float] = {}
    distributions: dict[tuple[str, date], float] = {}
    pay_dates: dict[tuple[str, date], date | None] = {}
    splits: dict[tuple[str, date], float] = {}
    rows = prices.select("ticker", "date", "close_krw", "distribution_krw", "distribution_pay_date", "split_factor").to_dicts()
    for row in rows:
        close = row["close_krw"]
        if close is None or close <= 0:
            raise PensionDataError(f"KR_LIVE close for {row['ticker']!r} on {row['date']!r} is missing or zero")
        factor = row["split_factor"]
        if factor is None or factor <= 0:
            raise PensionDataError(f"KR_LIVE split factor for {row['ticker']!r} on {row['date']!r} is missing or zero")
        dist = row["distribution_krw"]
        if dist is None or dist < 0:
            raise PensionDataError(f"KR_LIVE distribution for {row['ticker']!r} on {row['date']!r} is missing")
        key = (str(row["ticker"]), row["date"])
        closes[key] = float(close)
        distributions[key] = float(dist)
        pay_dates[key] = row["distribution_pay_date"]
        splits[key] = float(factor)
    return closes, distributions, pay_dates, splits


def _is_eligible(withdrawal_day: date, profile: PensionTaxProfile, regime: PensionTaxRegime) -> bool:
    age = withdrawal_day.year - profile.birth_date.year
    if (withdrawal_day.month, withdrawal_day.day) < (profile.birth_date.month, profile.birth_date.day):
        age -= 1
    account_years = withdrawal_day.year - profile.account_open_date.year
    if (withdrawal_day.month, withdrawal_day.day) < (profile.account_open_date.month, profile.account_open_date.day):
        account_years -= 1
    return age >= regime.minimum_pension_age and account_years >= regime.minimum_account_years


def run_pension_backtest(
    config: PensionBacktestConfig,
    prices: pl.DataFrame,
    fx: pl.DataFrame | None,
    profile: PensionTaxProfile,
    regime: PensionTaxRegime,
) -> PensionBacktestResult:
    """Value a buy-only pension account and its dated tax credits through payout.

    Args: prices are certified market-mode rows; fx is required only for US proxies;
        profile supplies tax capacity and age; regime is a frozen policy scenario.
    Returns: Dated account/cash snapshots, tax-credit receipts, legal pension payouts,
        after-tax external cashflows, and a source-labeled terminal state.
    Raises: PensionDataError or ValueError on missing sessions, unsupported taxes,
        illegal withdrawals, noncausal prices, or unreconciled cash and units.
    """
    tickers = _validate_config(config)
    if prices.is_empty():
        raise PensionDataError("pension prices frame is empty")
    if "ticker" not in prices.columns or "date" not in prices.columns:
        raise PensionDataError("pension prices miss ticker/date columns")
    window = prices.filter((pl.col("date") >= config.start) & (pl.col("date") <= config.end))
    if window.is_empty():
        raise PensionDataError("pension prices carry no session inside the backtest window")
    for ticker in tickers:
        if window.filter(pl.col("ticker") == ticker).is_empty():
            raise PensionDataError(f"pension prices carry no row for target {ticker!r}; refusing to splice")

    live_distributions: dict[tuple[str, date], float] = {}
    live_pay_dates: dict[tuple[str, date], date | None] = {}
    live_splits: dict[tuple[str, date], float] = {}
    withheld_per_unit_krw: dict[tuple[str, date], float] = {}
    if config.market_mode is PensionMarketMode.US_PROXY:
        base_marks, withheld_per_unit_krw = _proxy_marks(
            window, fx, config.max_fx_age_days, regime.foreign_dividend_withholding_rate
        )
    else:
        if fx is not None:
            raise ValueError("KR_LIVE mode takes no fx frame; Korean closes are already KRW")
        base_marks, live_distributions, live_pay_dates, live_splits = _live_marks(window)

    sessions: list[date] = sorted({day for day in window.get_column("date").to_list() if config.start <= day <= config.end})
    required_months = set()
    cursor = date(config.start.year, config.start.month, 1)
    while cursor <= config.end:
        required_months.add((cursor.year, cursor.month))
        cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
    for ticker in tickers:
        ticker_months = {
            (day.year, day.month)
            for day in window.filter(pl.col("ticker") == ticker).get_column("date").to_list()
        }
        missing = sorted(required_months - ticker_months)
        if missing:
            preview = ", ".join(f"{year}-{month:02d}" for year, month in missing[:3])
            raise PensionDataError(f"pension prices for {ticker!r} miss months {preview}; refusing to splice")

    def mark(ticker: str, day: date) -> float:
        raw = base_marks.get((ticker, day))
        if raw is None:
            raise PensionDataError(f"pension price for {ticker!r} on {day.isoformat()} is missing; no forward fill")
        drag = float(config.extra_annual_drag_by_ticker.get(ticker, 0.0))
        if drag == 0.0:
            return raw
        decayed: float = raw * (1.0 - drag) ** ((day - config.start).days / _DAYS_PER_YEAR)
        return decayed

    month_decisions: dict[tuple[int, int], date] = {}
    for day in sessions:
        month_decisions[(day.year, day.month)] = day
    executions: list[date] = []
    for decision in sorted(month_decisions.values()):
        following = [day for day in sessions if day > decision]
        if following:
            executions.append(following[0])
    executions = sorted(set(executions))
    execution_set = set(executions)

    contribution_events: list[tuple[date, int]] = []
    available_by_year: dict[int, int] = {}
    for day, amount in config.available_cash_events_krw.items():
        available_by_year[day.year] = available_by_year.get(day.year, 0) + amount
    unspent_available = 0
    for year in range(config.start.year, config.end.year + 1):
        budget = min(unspent_available + available_by_year.get(year, 0), regime.annual_credit_limit_krw)
        max_quote = quote_annual_pension_credit(year, budget, profile, regime)
        target_credit = (max_quote.national_credit_krw, max_quote.local_credit_krw)
        lower, upper = 0, budget
        while lower < upper:
            midpoint = (lower + upper) // 2
            quote = quote_annual_pension_credit(year, midpoint, profile, regime)
            if (quote.national_credit_krw, quote.local_credit_krw) == target_credit:
                upper = midpoint
            else:
                lower = midpoint + 1
        yearly_total = lower
        dates = tuple(sorted(config.contribution_dates.get(year, ())))
        if yearly_total > 0 and dates:
            per_date, remainder = divmod(yearly_total, len(dates))
            for index, day in enumerate(dates):
                contribution_events.append((day, per_date + (1 if index < remainder else 0)))
            unspent_available += available_by_year.get(year, 0) - yearly_total
        else:
            unspent_available += available_by_year.get(year, 0)
    contribution_events.sort()
    for day, amount in contribution_events:
        if amount > 0 and day < profile.account_open_date:
            raise ValueError(
                f"pension contribution on {day.isoformat()} precedes account opening "
                f"{profile.account_open_date.isoformat()}"
            )
        if amount > 0 and profile.pension_start_date is not None and day >= profile.pension_start_date:
            raise ValueError(f"pension contribution on {day.isoformat()} follows pension commencement")
    available_cash = 0
    available_events = sorted(config.available_cash_events_krw.items())
    available_index = 0
    for day, amount in contribution_events:
        while available_index < len(available_events) and available_events[available_index][0] <= day:
            available_cash += available_events[available_index][1]
            available_index += 1
        if amount > available_cash:
            raise ValueError(f"pension contribution on {day.isoformat()} exceeds cash available by that date")
        available_cash -= amount
    event_index = 0

    credits: list[PensionCreditQuote] = []
    for year in range(config.start.year, config.end.year + 1):
        yearly_total = sum(amount for day, amount in contribution_events if day.year == year)
        quote = quote_annual_pension_credit(year, yearly_total, profile, regime)
        credits.append(quote)
    credited_basis_by_event: list[int] = []
    contributed_by_year: dict[int, int] = {}
    for day, amount in contribution_events:
        quote = credits[day.year - config.start.year]
        prior = contributed_by_year.get(day.year, 0)
        current = prior + amount
        credited_basis_by_event.append(
            (current * quote.credited_principal_krw) // quote.contributed_krw
            - (prior * quote.credited_principal_krw) // quote.contributed_krw
        )
        contributed_by_year[day.year] = current

    payout_days: dict[date, int] = {}
    payout_years = sorted(year for year, amount in config.withdrawal_amounts_krw.items() if amount > 0)
    for year in payout_years:
        year_executions = [day for day in executions if day.year == year]
        if not year_executions:
            raise PensionDataError(f"pension payout for {year} has no executable session; needs market rows")
        payout_days[year_executions[-1]] = year

    cost_rate = (float(config.execution_spread_bps) + float(config.commission_bps)) / _BPS
    cash = 0.0
    units = dict.fromkeys(tickers, 0.0)
    receivables: list[tuple[date, float]] = []
    cumulative_contributions = 0
    cumulative_fees = 0.0
    snapshots: list[PensionSnapshot] = []
    withdrawals: list[PensionWithdrawalQuote] = []
    payout_shortfalls: list[tuple[date, int]] = []
    external: list[tuple[date, int]] = [
        (config.tax_credit_settlement_dates[credit.tax_year], credit.national_credit_krw + credit.local_credit_krw)
        for credit in credits
        if config.tax_credit_settlement_dates[credit.tax_year] <= config.end
    ]
    opening_nav_by_year: dict[int, int] = {}
    uncredited_basis = 0
    credited_basis = 0
    foreign_tax_withheld = 0.0
    attempted = 0

    for day in sessions:
        for ticker in tickers:
            per_unit = withheld_per_unit_krw.get((ticker, day))
            if per_unit is not None:
                foreign_tax_withheld += units[ticker] * per_unit
        if config.market_mode is PensionMarketMode.KR_LIVE:
            for ticker in tickers:
                factor = live_splits.get((ticker, day))
                if factor is not None and factor != 1.0:
                    units[ticker] *= factor
            for ticker in tickers:
                key = (ticker, day)
                if key in live_distributions and live_distributions[key] > 0:
                    pay = live_pay_dates[key]
                    if pay is None or pay < day:
                        raise PensionDataError(f"KR_LIVE distribution for {ticker!r} has no valid pay date")
                    receivables.append((pay, live_distributions[key] * units[ticker]))
            due = 0.0
            remaining: list[tuple[date, float]] = []
            for pay_day, receivable_amount in receivables:
                if pay_day <= day:
                    due += receivable_amount
                else:
                    remaining.append((pay_day, receivable_amount))
            cash += due
            receivables = remaining
        while event_index < len(contribution_events) and contribution_events[event_index][0] <= day:
            amount = contribution_events[event_index][1]
            cash += amount
            cumulative_contributions += amount
            credited_basis += credited_basis_by_event[event_index]
            uncredited_basis += amount - credited_basis_by_event[event_index]
            event_index += 1
        if day in execution_set and cash > 0:
            spendable = cash
            cash = 0.0
            for ticker in tickers:
                alloc = spendable * float(config.targets[ticker])
                price = mark(ticker, day)
                if config.market_mode is PensionMarketMode.KR_LIVE:
                    bought = float(math.floor(alloc / (price * (1.0 + cost_rate))))
                else:
                    bought = alloc / (price * (1.0 + cost_rate))
                cost = bought * price
                fee = cost * cost_rate
                units[ticker] += bought
                cash += alloc - cost - fee
                cumulative_fees += fee
        if cash < -1e-6:  # pragma: no cover - defended by construction; fail-closed tripwire
            raise PensionDataError("pension cash went negative; a schedule spent cash before it existed")
        if day.year not in opening_nav_by_year:
            if profile.pension_start_date is not None and day.year == profile.pension_start_date.year:
                if day >= profile.pension_start_date:
                    opening_nav_by_year[day.year] = (
                        int(
                            cash + sum(units[ticker] * mark(ticker, day) for ticker in tickers)
                            + sum(amount for _, amount in receivables)
                        )
                        if day == profile.pension_start_date
                        else (snapshots[-1].nav_krw if snapshots else 0)
                    )
            else:
                opening_nav_by_year[day.year] = snapshots[-1].nav_krw if snapshots else 0
        payout_year = payout_days.get(day)
        if payout_year is not None and _is_eligible(day, profile, regime):
            attempted += 1
            opening_nav = opening_nav_by_year.get(payout_year, 0)
            portfolio_value = sum(units[ticker] * mark(ticker, day) for ticker in tickers)
            requested = config.withdrawal_amounts_krw[payout_year]
            payable = min(requested, max(0, math.floor(cash + portfolio_value)))
            if payable < requested:
                payout_shortfalls.append((day, requested - payable))
            if payable > 0:
                payout = quote_annual_pension_withdrawal(
                    day, payable, opening_nav, 0, uncredited_basis, credited_basis, profile, regime,
                )
                withdrawals.append(payout)
                uncredited_basis = payout.remaining_uncredited_principal_krw
                credited_basis = payout.remaining_credited_principal_krw
                sale_needed = payable - cash
                if sale_needed > 0:
                    for ticker in tickers:
                        ticker_value = units[ticker] * mark(ticker, day)
                        weight = ticker_value / portfolio_value if portfolio_value > 0 else 0.0
                        units[ticker] -= sale_needed * weight / mark(ticker, day)
                    cash += sale_needed
                cash -= payable
                external.append((day, payable - payout.national_tax_krw - payout.local_tax_krw))
        nav = int(cash + sum(units[ticker] * mark(ticker, day) for ticker in tickers) + sum(amount for _, amount in receivables))
        snapshots.append(
            PensionSnapshot(
                session=day,
                cash_krw=int(cash),
                distribution_receivable_krw=int(sum(amount for _, amount in receivables)),
                units_by_ticker=dict(units),
                nav_krw=nav,
                cumulative_contributions_krw=cumulative_contributions,
                cumulative_fees_krw=int(cumulative_fees),
            )
        )

    external.sort()
    result_nav = snapshots[-1].nav_krw
    logger.info(
        "[SIM] event=pension_backtest_done mode=%s sessions=%d terminal_nav=%d withdrawals=%d",
        str(config.market_mode),
        len(snapshots),
        result_nav,
        len(withdrawals),
    )
    return PensionBacktestResult(
        snapshots=tuple(snapshots),
        contribution_cashflows_krw=tuple((day, amount) for day, amount in contribution_events if amount > 0),
        tax_credits=tuple(credits),
        withdrawals=tuple(withdrawals),
        payout_shortfalls_krw=tuple(payout_shortfalls),
        terminal_nav_krw=result_nav,
        after_tax_external_cashflows_krw=tuple(external),
        is_retirement_terminal=attempted > 0 and attempted == len(payout_years) and not payout_shortfalls and result_nav == 0,
        market_mode=config.market_mode,
        terminal_credited_principal_krw=credited_basis,
        terminal_uncredited_principal_krw=uncredited_basis,
        foreign_tax_withheld_krw=int(foreign_tax_withheld),
    )
