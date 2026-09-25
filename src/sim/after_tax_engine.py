"""Monthly accumulation engine with lot-level Korean overseas-equity taxation."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal, cast

import polars as pl

from src.analytics.metrics import max_drawdown, real_krw, xirr
from src.data.calendar import DEFAULT_CALENDAR_NAME, TradingCalendar, load_calendar
from src.data.catalog import load_snapshot_visible, resolve_snapshot
from src.data.schedule import DecisionPoint, build_decision_schedule
from src.data.schema import Dataset
from src.features.pit_market import PitMarket
from src.policy.weight_rule import CASH_SLEEVE, WeightRule
from src.sim.after_tax_market import AfterTaxDataError, _CpiIndex, _FxIndex, _PriceIndex, _RateIndex
from src.sim.corporate_actions import CorporateAction, CorporateActionKind, corporate_actions_from_prices
from src.sim.tax import KrOverseasTaxRegime, annual_capital_gains_tax_krw
from src.sim.tax_lots import RealizedDisposal, TaxLotBook

if TYPE_CHECKING:
    from src.data.settings import DataSettings

logger = logging.getLogger(__name__)

_BPS: Final[float] = 10_000.0
_WEIGHT_SUM_TOLERANCE: Final[float] = 1e-6
_CASH_TOLERANCE_KRW: Final[float] = 1e-6
_HARVEST_EXHAUSTED_KRW: Final[float] = 1e-6
_DAYS_PER_YEAR_360: Final[float] = 360.0

__all__ = [
    "AfterTaxConfig",
    "AfterTaxDataError",
    "AfterTaxResult",
    "AfterTaxSnapshot",
    "ExecutionMode",
    "run_after_tax",
    "run_after_tax_from_store",
]


class ExecutionMode(StrEnum):
    """How holdings move toward target weights."""

    BUY_ONLY = "buy_only"
    REBALANCE_BAND = "rebalance_band"


@dataclass(frozen=True, slots=True)
class AfterTaxConfig:
    """External cashflow, target source, execution, friction, and tax parameters.

    Exactly one of ``targets`` (static simplex) or ``rule`` must be set.
    ``tax_enabled=False`` disables capital-gains tax, dividend withholding, and interest
    tax for frictionless decomposition (callers also pass zero commission/spread).
    ``price_mode="adjusted"`` exists only for parity with the legacy engine: fills and
    marks use adjusted closes and corporate actions are not applied.
    Exactly one funding mode applies: a positive ``monthly_contribution_krw`` with no
    schedule, or ``monthly_contribution_krw == 0`` with a non-empty
    ``contribution_schedule_krw`` of positive KRW amounts dated within ``[start, end]``;
    scheduled cash is deposited at the first execution session on or after its date.
    """

    start: date
    end: date
    monthly_contribution_krw: float
    contribution_schedule_krw: Mapping[date, float] | None = field(default=None, kw_only=True)
    tax_regime: KrOverseasTaxRegime
    targets: Mapping[str, float] | None = None
    rule: WeightRule | None = None
    mode: ExecutionMode = ExecutionMode.BUY_ONLY
    rebalance_band: float | None = None
    harvest_gains: bool = True
    tax_enabled: bool = True
    commission_bps: float = 0.0
    fx_spread_bps: float = 0.0
    price_mode: Literal["raw", "adjusted"] = "raw"
    fractional_shares: bool = False
    fill_delay_sessions: int = 1
    fx_max_staleness_days: int = 7
    cash_rate_series: str = "DTB3"
    label: str = ""


@dataclass(frozen=True, slots=True)
class AfterTaxSnapshot:
    """Ledger state at one execution close; the snapshot path is the SSOT for metrics."""

    session: date
    contribution_krw: float
    cash_krw: float
    cash_usd: float
    shares: Mapping[str, float]
    nav_krw: float
    after_tax_nav_krw: float
    realized_gain_ytd_krw: float
    tax_payable_krw: float
    taxes_paid_krw: float
    fees_krw: float


@dataclass(frozen=True, slots=True)
class AfterTaxResult:
    """Full path plus after-tax summary metrics (nominal and first-snapshot real KRW)."""

    config: AfterTaxConfig
    snapshots: tuple[AfterTaxSnapshot, ...]
    disposals: tuple[RealizedDisposal, ...]
    terminal_after_tax_krw: float
    terminal_after_tax_real_krw: float
    terminal_pre_liquidation_krw: float
    liquidation_tax_krw: float
    taxes_paid_krw: float
    xirr_after_tax_real: float
    total_contribution_real_krw: float
    max_drawdown_after_tax: float
    annual_financial_income_krw: Mapping[int, float]
    financial_income_breach_years: tuple[int, ...]
    sell_count: int


def _check_scheduled_cash_before_final_execution(
    funding: Mapping[date, float], last_execution: date
) -> None:
    """Fail closed on scheduled cash that the final execution session would never invest."""
    late = sorted(day for day in funding if day > last_execution)
    if late:
        raise AfterTaxDataError(
            f"scheduled cash on {late[0].isoformat()} falls after the final execution session {last_execution.isoformat()}"
        )


def run_after_tax(
    config: AfterTaxConfig,
    prices: pl.DataFrame,
    fx: pl.DataFrame,
    cpi: pl.DataFrame,
    rates: pl.DataFrame | None = None,
) -> AfterTaxResult:
    """Simulate accumulation with lot-level Korean overseas-equity taxation.

    Month-end signals, fills at the ``fill_delay_sessions``-later close (raw prices in
    raw mode), per-sleeve USD earmarks so unaffordable sleeves accumulate until a whole
    share fits, dividends credited net of withholding to their own sleeve, splits applied
    to lots, optional band rebalancing with sells, December tax-gain harvesting up to the
    remaining annual deduction with immediate rebuy, annual tax assessed by settlement
    year and paid from KRW contributions in the regime's payment month, and after-tax
    liquidation net worth marked at every execution close. ``fx`` is a USD/KRW frame with
    ``date``, ``usdkrw``, ``available_at`` (FX_KRW_BASE in production; the legacy FX frame
    only for parity) resolved as-of each instant within ``fx_max_staleness_days``.
    Funding is either a positive ``monthly_contribution_krw`` deposited every step or,
    with ``monthly_contribution_krw == 0``, a dated ``contribution_schedule_krw`` whose
    amounts land at the first execution session on or after their date.

    Raises:
        ValueError: On a non-positive contribution, both/neither of ``targets``/``rule``,
            a non-simplex static target, a missing/invalid band for REBALANCE_BAND,
            ``fill_delay_sessions < 1``, or ``fx_max_staleness_days < 0``.
        AfterTaxDataError: On an empty schedule, missing/stale price, FX, CPI, or rate,
            or scheduled cash dated after the final execution session.
        PitMarketError: When a rule lacks visible history at a signal instant.
        CorporateActionError: On unmappable vendor split factors.
        TaxLotError: On an internally inconsistent disposal (never expected; fail closed).
        XirrError: When the after-tax money-weighted rate cannot be identified.
    """
    _validate_config(config)
    calendar = load_calendar(DEFAULT_CALENDAR_NAME)
    schedule = build_decision_schedule(
        config.start, config.end, frequency="monthly", fill_delay_sessions=config.fill_delay_sessions
    )
    if not schedule:
        raise AfterTaxDataError(f"empty decision schedule over [{config.start.isoformat()}, {config.end.isoformat()}]")
    if config.contribution_schedule_krw is not None:
        _check_scheduled_cash_before_final_execution(
            config.contribution_schedule_krw, schedule[-1].execution_session
        )
    price_index = _PriceIndex(prices)
    fx_index = _FxIndex(fx)
    cpi_index = _CpiIndex(cpi)
    rate_index = _RateIndex(rates)
    market = PitMarket(prices, rates)
    actions = _extract_actions(prices, config)
    book = TaxLotBook(config.tax_regime.basis_method)
    state = _EngineState(config, calendar)
    previous_execution: date | None = None
    for point in schedule:
        state.step(
            point,
            price_index=price_index,
            fx_index=fx_index,
            cpi_index=cpi_index,
            rate_index=rate_index,
            market=market,
            actions=actions,
            book=book,
            previous_execution=previous_execution,
        )
        previous_execution = point.execution_session
    return state.finish()


def run_after_tax_from_store(config: AfterTaxConfig, settings: DataSettings) -> AfterTaxResult:
    """Load pinned PRICES, FX_KRW_BASE, CPI (and RATES when the cash sleeve is used) then simulate.

    Raises:
        UntrustedDatasetError: When a required pinned dataset is absent or changes.
        AfterTaxDataError: As in :func:`run_after_tax`.
    """
    needs_rates = (config.rule is not None and config.rule.requires_cash_rate) or (
        config.targets is not None and CASH_SLEEVE in config.targets
    )
    datasets: tuple[Dataset, ...] = (Dataset.PRICES, Dataset.FX_KRW_BASE, Dataset.CPI)
    if needs_rates:
        datasets = (*datasets, Dataset.RATES)
    snapshot = resolve_snapshot(settings, datasets)
    calendar = load_calendar(DEFAULT_CALENDAR_NAME)
    schedule = build_decision_schedule(
        config.start, config.end, frequency="monthly", fill_delay_sessions=config.fill_delay_sessions
    )
    if not schedule:
        raise AfterTaxDataError(f"empty decision schedule over [{config.start.isoformat()}, {config.end.isoformat()}]")
    cutoff = calendar.close_ts(schedule[-1].execution_session)
    prices = load_snapshot_visible(snapshot, Dataset.PRICES, cutoff)
    cpi = load_snapshot_visible(snapshot, Dataset.CPI, cutoff)
    last_settle = schedule[-1].execution_session
    for _ in range(config.tax_regime.settlement_sessions):
        last_settle = calendar.next_session(last_settle)
    fx = load_snapshot_visible(snapshot, Dataset.FX_KRW_BASE, calendar.close_ts(last_settle))
    rate_frame = load_snapshot_visible(snapshot, Dataset.RATES, cutoff) if needs_rates else None
    return run_after_tax(config, prices, fx, cpi, rate_frame)


def _validate_config(config: AfterTaxConfig) -> None:
    """Fail closed on contradictory targets, costs, or execution parameters."""
    if not isinstance(config.monthly_contribution_krw, int | float) or isinstance(
        config.monthly_contribution_krw, bool
    ):
        raise ValueError("monthly_contribution_krw must be a number")
    funding = config.contribution_schedule_krw
    if funding is None:
        if not math.isfinite(config.monthly_contribution_krw) or config.monthly_contribution_krw <= 0:
            raise ValueError("monthly_contribution_krw must be positive")
    else:
        if config.monthly_contribution_krw != 0.0:
            raise ValueError(
                "monthly_contribution_krw must be 0.0 when contribution_schedule_krw is set, "
                f"got {config.monthly_contribution_krw!r}"
            )
        if not isinstance(funding, Mapping) or not funding:
            raise ValueError("contribution_schedule_krw must be a non-empty mapping of dates to KRW amounts")
        for day, amount in funding.items():
            if not isinstance(day, date) or isinstance(day, datetime):
                raise ValueError(f"contribution_schedule_krw key {day!r} must be a date")
            if day < config.start or day > config.end:
                raise ValueError(
                    f"contribution_schedule_krw date {day.isoformat()} lies outside "
                    f"[{config.start.isoformat()}, {config.end.isoformat()}]"
                )
            if (
                isinstance(amount, bool)
                or not isinstance(amount, int | float)
                or not math.isfinite(amount)
                or amount <= 0
            ):
                raise ValueError(
                    f"contribution_schedule_krw[{day.isoformat()}] must be a finite positive amount, got {amount!r}"
                )
    if (config.targets is None) == (config.rule is None):
        raise ValueError("exactly one of targets or rule must be set")
    if config.targets is not None:
        _check_simplex(config.targets, "targets")
    if config.mode is ExecutionMode.REBALANCE_BAND:
        band = config.rebalance_band
        if band is None or isinstance(band, bool) or not isinstance(band, int | float):
            raise ValueError("REBALANCE_BAND requires a numeric rebalance_band")
        if not math.isfinite(band) or band < 0.0 or band > 1.0:
            raise ValueError(f"rebalance_band must lie in [0, 1], got {band!r}")
    if config.fill_delay_sessions < 1:
        raise ValueError(f"fill_delay_sessions must be >= 1, got {config.fill_delay_sessions!r}")
    if config.fx_max_staleness_days < 0:
        raise ValueError(f"fx_max_staleness_days must be >= 0, got {config.fx_max_staleness_days!r}")
    if config.price_mode not in ("raw", "adjusted"):
        raise ValueError(f"price_mode must be 'raw' or 'adjusted', got {config.price_mode!r}")
    for name in ("commission_bps", "fx_spread_bps"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")


def _check_simplex(weights: Mapping[str, float], name: str) -> None:
    """Fail closed on negative, non-finite, or non-simplex weights within 1e-6."""
    total = 0.0
    for ticker, weight in weights.items():
        value = float(weight)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name}[{ticker!r}] must be finite nonnegative, got {weight!r}")
        total += value
    if not math.isfinite(total) or abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"{name} weights must sum to 1.0 within 1e-6, got {total!r}")


def _extract_actions(prices: pl.DataFrame, config: AfterTaxConfig) -> tuple[CorporateAction, ...]:
    """Extract the run's corporate actions once; per-step windows filter by ex-date."""
    if config.price_mode != "raw":
        return ()
    universe = set(config.targets) if config.targets is not None else set(config.rule.tickers)  # type: ignore[union-attr]
    universe.discard(CASH_SLEEVE)
    return corporate_actions_from_prices(prices, tickers=universe, start=config.start, end=config.end)


def _day_end(day: date) -> datetime:
    """End-of-day UTC instant used for ex-date visibility (no calendar lookup)."""
    return datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=1)


class _Payable:
    """Outstanding assessed tax with its earliest payable date; mutated as paid."""

    __slots__ = ("amount", "due")

    def __init__(self, due: date, amount: float) -> None:
        self.due = due
        self.amount = amount


def _verify_usd_flows(
    *,
    context: str,
    before_usd: float,
    after_usd: float,
    inflows_usd: float,
    outflows_usd: float,
    fx: float,
) -> None:
    """Ledger conservation: earmark deltas reconcile flows within 1e-6 KRW."""
    drift_krw = abs(after_usd - (before_usd + inflows_usd - outflows_usd)) * fx
    if drift_krw > _CASH_TOLERANCE_KRW:
        raise AfterTaxDataError(f"cash conservation breach at {context}: drift {drift_krw:.6f} KRW")


class _EngineState:
    """Mutable per-run ledger advanced one execution session at a time."""

    def __init__(self, config: AfterTaxConfig, calendar: TradingCalendar) -> None:
        self._config = config
        self._calendar = calendar
        self._regime = config.tax_regime
        self._commission = config.commission_bps / _BPS
        self._spread = config.fx_spread_bps / _BPS
        self._adjusted = config.price_mode == "adjusted"
        self._earmarks: dict[str, float] = {}
        self._cash_krw = 0.0
        self._net_by_year: dict[int, float] = {}
        self._assessed: set[int] = set()
        self._pending: list[_Payable] = []
        self._tax_payable = 0.0
        self._taxes_paid = 0.0
        self._income: dict[int, float] = {}
        self._disposals: list[RealizedDisposal] = []
        self._snapshots: list[AfterTaxSnapshot] = []
        self._cpi_levels: list[float] = []
        self._sell_count = 0
        self._last_liquidation_tax = 0.0

    def step(
        self,
        point: DecisionPoint,
        *,
        price_index: _PriceIndex,
        fx_index: _FxIndex,
        cpi_index: _CpiIndex,
        rate_index: _RateIndex,
        market: PitMarket,
        actions: tuple[CorporateAction, ...],
        book: TaxLotBook,
        previous_execution: date | None,
    ) -> None:
        """Advance the ledger through one signal/execution pair in spec sequence."""
        config = self._config
        day = point.execution_session
        close = self._calendar.close_ts(day)
        settle = day
        for _ in range(self._regime.settlement_sessions):
            settle = self._calendar.next_session(settle)
        settle_close = self._calendar.close_ts(settle)
        fx_day = fx_index.resolve(day, close, config.fx_max_staleness_days)
        settle_fx = fx_index.resolve(settle, settle_close, config.fx_max_staleness_days)
        earmark_before = sum(self._earmarks.values())
        inflows_usd = 0.0
        outflows_usd = 0.0
        fees_krw = 0.0

        # Corporate actions with ex-dates in (previous_execution, day].
        for action in actions:
            if action.ex_date > day:
                continue
            if previous_execution is not None and action.ex_date <= previous_execution:
                continue
            if action.kind is CorporateActionKind.SPLIT:
                pre_basis = book.basis_krw(action.ticker)
                remainder = book.apply_split(action.ticker, action.split_ratio)  # type: ignore[arg-type]
                if remainder > 0:
                    price_ex = price_index.price(action.ticker, action.ex_date, _day_end(action.ex_date), adjusted=False)
                    fx_ex = fx_index.resolve(action.ex_date, _day_end(action.ex_date), config.fx_max_staleness_days)
                    proceeds_usd = remainder * price_ex
                    proceeds_krw = proceeds_usd * fx_ex
                    removed_basis = pre_basis - book.basis_krw(action.ticker)
                    gain = proceeds_krw - removed_basis
                    self._disposals.append(
                        RealizedDisposal(
                            ticker=action.ticker,
                            trade_session=action.ex_date,
                            settle_session=action.ex_date,
                            quantity=remainder,
                            proceeds_krw=proceeds_krw,
                            basis_krw=removed_basis,
                            gain_krw=gain,
                        )
                    )
                    self._net_by_year[action.ex_date.year] = self._net_by_year.get(action.ex_date.year, 0.0) + gain
                    self._earmarks[action.ticker] = self._earmarks.get(action.ticker, 0.0) + proceeds_usd
                    inflows_usd += proceeds_usd
            else:
                held = book.quantity(action.ticker)
                if held > 0 and action.dividend_usd is not None:
                    gross_usd = held * action.dividend_usd
                    credit_usd = gross_usd if not config.tax_enabled else gross_usd * (1.0 - self._regime.dividend_withholding_rate)
                    self._earmarks[action.ticker] = self._earmarks.get(action.ticker, 0.0) + credit_usd
                    inflows_usd += credit_usd
                    fx_ex = fx_index.resolve(action.ex_date, _day_end(action.ex_date), config.fx_max_staleness_days)
                    self._income[action.ex_date.year] = self._income.get(action.ex_date.year, 0.0) + gross_usd * fx_ex

        # Cash-sleeve accrual from the previous execution close at the visible rate.
        cash_balance = self._earmarks.get(CASH_SLEEVE, 0.0)
        if cash_balance > 0 and previous_execution is not None:
            previous_close = self._calendar.close_ts(previous_execution)
            accrual_days = (day - previous_execution).days
            rate = rate_index.resolve(config.cash_rate_series, previous_close)
            gross_usd = cash_balance * rate / 100.0 * accrual_days / _DAYS_PER_YEAR_360
            credit_usd = gross_usd if not config.tax_enabled else gross_usd * (1.0 - self._regime.interest_tax_rate)
            self._earmarks[CASH_SLEEVE] = cash_balance + credit_usd
            inflows_usd += credit_usd
            self._income[day.year] = self._income.get(day.year, 0.0) + gross_usd * fx_day

        # Assess every tax year before this execution's year; pay what is due.
        for assessed_year in sorted(self._net_by_year):
            if assessed_year >= day.year or assessed_year in self._assessed:
                continue
            self._assessed.add(assessed_year)
            due_tax = 0.0 if not config.tax_enabled else annual_capital_gains_tax_krw(
                self._net_by_year[assessed_year], self._regime
            )
            if due_tax > 0:
                self._pending.append(_Payable(date(assessed_year + 1, self._regime.payment_month, 1), due_tax))
                self._tax_payable += due_tax
        funding = config.contribution_schedule_krw
        if funding is None:
            deposit = config.monthly_contribution_krw
        elif previous_execution is None:
            deposit = float(sum(amount for funded_day, amount in funding.items() if funded_day <= day))
        else:
            deposit = float(
                sum(amount for funded_day, amount in funding.items() if previous_execution < funded_day <= day)
            )
        self._cash_krw += deposit
        due_now = sum(entry.amount for entry in self._pending if entry.due <= day)
        if due_now > 0:
            paid = min(due_now, self._cash_krw)
            self._cash_krw -= paid
            self._taxes_paid += paid
            self._tax_payable -= paid
            remaining = paid
            for entry in self._pending:
                if entry.due <= day and remaining > 0 and entry.amount > 0:
                    take = min(entry.amount, remaining)
                    entry.amount -= take
                    remaining -= take
            self._pending = [entry for entry in self._pending if entry.amount > 0]
        converted_krw = self._cash_krw
        self._cash_krw = 0.0

        # Convert the rest to USD; the spread is a fee, never a tax expense.
        fx_gross = fx_day * (1.0 + self._spread)
        converted_usd = converted_krw / fx_gross if converted_krw > 0 else 0.0
        fees_krw += converted_krw - converted_usd * fx_day
        inflows_usd += converted_usd

        # Target weights from the static simplex or the PIT rule.
        weights = (
            dict(config.targets)
            if config.targets is not None
            else dict(cast(WeightRule, config.rule)(point.signal_at, market))
        )
        _check_simplex(weights, "weights")
        weight_total = sum(weights.values())
        weights = {ticker: weight / weight_total for ticker, weight in weights.items()}
        for ticker, weight in weights.items():
            if weight > 0 and converted_usd > 0:
                self._earmarks[ticker] = self._earmarks.get(ticker, 0.0) + converted_usd * weight

        # Optional band rebalancing with sells; buy-only never sells here.
        if config.mode is ExecutionMode.REBALANCE_BAND:
            pool_usd, cash_take_usd, sell_fees = self._rebalance(
                weights, day, fx_day, settle, settle_fx, price_index, book, config
            )
            fees_krw += sell_fees
            inflows_usd += pool_usd - cash_take_usd

        # December tax-gain harvesting settled within the same year, then rebuy.
        if (
            config.harvest_gains
            and config.tax_enabled
            and day.month == 12
            and settle.year == day.year
        ):
            harvested_fees, harvest_in, harvest_out = self._harvest(
                weights, day, fx_day, settle, settle_fx, price_index, book, config
            )
            fees_krw += harvested_fees
            inflows_usd += harvest_in
            outflows_usd += harvest_out

        # Fill each sleeve earmark; record lots at the settlement base rate.
        fill_tickers = sorted(ticker for ticker in self._earmarks if ticker != CASH_SLEEVE and self._earmarks[ticker] > 0)
        for ticker in fill_tickers:
            earmark = self._earmarks[ticker]
            price = price_index.price(ticker, day, close, adjusted=self._adjusted)
            cost_per_share = price * (1.0 + self._commission)
            lots = earmark / cost_per_share if config.fractional_shares else math.floor(earmark / cost_per_share)
            if lots <= 0:
                continue
            gross_usd = lots * cost_per_share
            if gross_usd > earmark:
                gross_usd = earmark
            book.buy(ticker, lots, gross_usd, settle_fx, trade_session=day, settle_session=settle)
            self._earmarks[ticker] = max(0.0, earmark - gross_usd)
            outflows_usd += gross_usd
            fees_krw += lots * price * self._commission * fx_day

        _verify_usd_flows(
            context=f"{day.isoformat()} fills",
            before_usd=earmark_before,
            after_usd=sum(self._earmarks.values()),
            inflows_usd=inflows_usd,
            outflows_usd=outflows_usd,
            fx=fx_day,
        )

        # Snapshot marks and the after-tax liquidation value.
        cpi_level = cpi_index.resolve(close)
        self._cpi_levels.append(cpi_level)
        positions: dict[str, float] = {}
        positions_usd = 0.0
        for ticker in book.tickers():
            price = price_index.price(ticker, day, close, adjusted=self._adjusted)
            positions[ticker] = book.quantity(ticker)
            positions_usd += positions[ticker] * price
        usd_total = sum(self._earmarks.values())
        nav_krw = positions_usd * fx_day + usd_total * fx_day + self._cash_krw - self._tax_payable
        year_gain = self._net_by_year.get(day.year, 0.0)
        if config.tax_enabled:
            unrealized = sum(
                book.unrealized_gain_krw(ticker, price_index.price(ticker, day, close, adjusted=self._adjusted) * (1.0 - self._commission), fx_day)
                for ticker in book.tickers()
            )
            liquidation_tax = self._regime.capital_gains_rate * max(0.0, year_gain + unrealized - self._regime.annual_deduction_krw)
        else:
            liquidation_tax = 0.0
        sell_friction = sum(
            positions[ticker] * price_index.price(ticker, day, close, adjusted=self._adjusted) for ticker in positions
        ) * self._commission * fx_day
        spread_cost = (positions_usd + usd_total) * fx_day * self._spread
        after_tax_nav = nav_krw - liquidation_tax - sell_friction - spread_cost
        self._last_liquidation_tax = liquidation_tax
        self._snapshots.append(
            AfterTaxSnapshot(
                session=day,
                contribution_krw=deposit,
                cash_krw=self._cash_krw,
                cash_usd=usd_total,
                shares=dict(positions),
                nav_krw=nav_krw,
                after_tax_nav_krw=after_tax_nav,
                realized_gain_ytd_krw=year_gain,
                tax_payable_krw=self._tax_payable,
                taxes_paid_krw=self._taxes_paid,
                fees_krw=fees_krw,
            )
        )
        logger.debug(
            "[PORTFOLIO] event=after_tax_step session=%s nav=%.2f after_tax=%.2f",
            day.isoformat(),
            nav_krw,
            after_tax_nav,
        )

    def _rebalance(
        self,
        weights: Mapping[str, float],
        day: date,
        fx_day: float,
        settle: date,
        settle_fx: float,
        price_index: _PriceIndex,
        book: TaxLotBook,
        config: AfterTaxConfig,
    ) -> tuple[float, float, float]:
        """Sell overweight sleeves past the band, then route pool USD to deficits.

        Returns pool USD moved into underweight earmarks, the internal cash-take
        portion (excluded from inflow accounting), and fee KRW. Post-trade
        values price positions at the execution close with earmarks counting toward
        their sleeve, so routing follows the same drift definition as the trigger.
        """
        close = self._calendar.close_ts(day)
        sleeve_values = self._sleeve_values(weights, day, fx_day, price_index, book, close)
        total = sum(sleeve_values.values()) + self._cash_krw
        if total <= 0:  # pragma: no cover - unreachable with positive contributions
            return 0.0, 0.0, 0.0
        if self._max_drift(sleeve_values, weights, total) <= config.rebalance_band:  # type: ignore[operator]
            return 0.0, 0.0, 0.0
        pool_usd = 0.0
        cash_take_usd = 0.0
        fees_krw = 0.0
        for ticker in sorted(sleeve_values):
            target_value = weights.get(ticker, 0.0) * total
            excess = sleeve_values[ticker] - target_value
            if excess <= 0:
                continue
            # 슬리브 평가액에는 미체결 earmark(USD)가 포함되므로 매도 전에 earmark부터 회수한다.
            take_usd = min(excess / fx_day, self._earmarks.get(ticker, 0.0))
            self._earmarks[ticker] = self._earmarks.get(ticker, 0.0) - take_usd
            pool_usd += take_usd
            cash_take_usd += take_usd
            excess -= take_usd * fx_day
            if ticker == CASH_SLEEVE or excess <= 0:
                continue
            price = price_index.price(ticker, day, close, adjusted=self._adjusted)
            sellable = excess / (price * fx_day)
            # 목표 0% 슬리브는 보유 전량까지만 매도 가능(공매도 금지)
            lots = min(sellable, book.quantity(ticker)) if config.fractional_shares else min(math.floor(sellable), math.floor(book.quantity(ticker)))
            if lots <= 0:
                continue
            proceeds_usd = lots * price * (1.0 - self._commission)
            disposal = book.sell(ticker, lots, proceeds_usd, settle_fx, trade_session=day, settle_session=settle)
            self._disposals.append(disposal)
            self._net_by_year[disposal.tax_year] = self._net_by_year.get(disposal.tax_year, 0.0) + disposal.gain_krw
            pool_usd += proceeds_usd
            fees_krw += lots * price * self._commission * fx_day
            self._sell_count += 1
        post_values = self._sleeve_values(weights, day, fx_day, price_index, book, close)
        route_total = sum(post_values.values()) + self._cash_krw + pool_usd * fx_day
        deficits = {
            ticker: weights[ticker] * route_total - post_values.get(ticker, 0.0)
            for ticker in weights
            if weights[ticker] * route_total - post_values.get(ticker, 0.0) > 0
        }
        deficit_total = sum(deficits.values())
        for ticker, deficit in deficits.items():
            self._earmarks[ticker] = self._earmarks.get(ticker, 0.0) + pool_usd * deficit / deficit_total
        return pool_usd, cash_take_usd, fees_krw

    def _sleeve_values(
        self,
        weights: Mapping[str, float],
        day: date,
        fx_day: float,
        price_index: _PriceIndex,
        book: TaxLotBook,
        close: datetime,
    ) -> dict[str, float]:
        """Sleeve values at the execution close; earmarks count toward their sleeve."""
        values: dict[str, float] = {}
        # 목표에서 빠진 슬리브(예: 위험회피 뒤 복귀한 CASH)도 잔여 earmark가 있으면 평가·회수 대상이다.
        held_earmarks = {ticker for ticker, usd in self._earmarks.items() if usd > 0}
        for ticker in set(book.tickers()) | set(weights) | held_earmarks:
            if ticker == CASH_SLEEVE:
                values[ticker] = self._earmarks.get(ticker, 0.0) * fx_day
                continue
            price = price_index.price(ticker, day, close, adjusted=self._adjusted)
            values[ticker] = book.quantity(ticker) * price * fx_day + self._earmarks.get(ticker, 0.0) * fx_day
        return values

    @staticmethod
    def _max_drift(values: Mapping[str, float], weights: Mapping[str, float], total: float) -> float:
        """Largest absolute weight drift of sleeve values against targets."""
        drift = 0.0
        for ticker, value in values.items():
            candidate = abs(value / total - weights.get(ticker, 0.0))
            if candidate > drift:
                drift = candidate
        return drift

    def _harvest(
        self,
        weights: Mapping[str, float],
        day: date,
        fx_day: float,
        settle: date,
        settle_fx: float,
        price_index: _PriceIndex,
        book: TaxLotBook,
        config: AfterTaxConfig,
    ) -> tuple[float, float, float]:
        """Sell deduction-bounded gains per sleeve and immediately rebuy; return fees/in/out USD."""
        budget = self._regime.annual_deduction_krw - self._net_by_year.get(day.year, 0.0)
        if budget <= 0:
            return 0.0, 0.0, 0.0
        candidates = sorted(
            (
                ticker
                for ticker in weights
                if ticker != CASH_SLEEVE and weights[ticker] > 0 and book.quantity(ticker) > 0
            ),
            key=lambda ticker: book.quantity(ticker)
            * price_index.price(ticker, day, self._calendar.close_ts(day), adjusted=self._adjusted),
            reverse=True,
        )
        fees_krw = 0.0
        inflows_usd = 0.0
        outflows_usd = 0.0
        realized = 0.0
        for ticker in candidates:
            remaining = budget - realized
            price = price_index.price(ticker, day, self._calendar.close_ts(day), adjusted=self._adjusted)
            net_price = price * (1.0 - self._commission)
            harvestable = book.harvestable_quantity(ticker, net_price, fx_day, remaining)
            lots = harvestable if config.fractional_shares else math.floor(harvestable)
            if lots <= 0:
                continue
            proceeds_usd = lots * net_price
            disposal = book.sell(ticker, lots, proceeds_usd, settle_fx, trade_session=day, settle_session=settle)
            self._disposals.append(disposal)
            self._net_by_year[disposal.tax_year] = self._net_by_year.get(disposal.tax_year, 0.0) + disposal.gain_krw
            realized += disposal.gain_krw
            fees_krw += lots * price * self._commission * fx_day
            self._sell_count += 1
            rebuy_cost = price * (1.0 + self._commission)
            rebuy = proceeds_usd / rebuy_cost if config.fractional_shares else math.floor(proceeds_usd / rebuy_cost)
            rebuy = min(rebuy, lots)
            if rebuy > 0:
                rebuy_gross = rebuy * rebuy_cost
                book.buy(ticker, rebuy, rebuy_gross, settle_fx, trade_session=day, settle_session=settle)
                outflows_usd += rebuy_gross
                fees_krw += rebuy * price * self._commission * fx_day
            leftover = proceeds_usd - (rebuy * rebuy_cost if rebuy > 0 else 0.0)
            self._earmarks[ticker] = self._earmarks.get(ticker, 0.0) + leftover
            inflows_usd += proceeds_usd
            if realized >= budget - _HARVEST_EXHAUSTED_KRW:
                break
        return fees_krw, inflows_usd, outflows_usd

    def finish(self) -> AfterTaxResult:
        """Assemble the result path with nominal and first-snapshot real metrics."""
        config = self._config
        snapshots = tuple(self._snapshots)
        calendar = self._calendar
        terminal_after_tax = snapshots[-1].after_tax_nav_krw
        terminal_pre = snapshots[-1].nav_krw
        base_cpi = self._cpi_levels[0]
        terminal_real = real_krw(terminal_after_tax, cpi_index=self._cpi_levels[-1], cpi_base=base_cpi)
        cashflows = [(calendar.close_ts(snapshot.session), -snapshot.contribution_krw) for snapshot in snapshots]
        cashflows.append((cashflows[-1][0], terminal_after_tax))
        real_cashflows = [
            (calendar.close_ts(snapshot.session), -snapshot.contribution_krw * base_cpi / level)
            for snapshot, level in zip(snapshots, self._cpi_levels, strict=True)
        ]
        real_cashflows.append((real_cashflows[-1][0], terminal_real))
        total_contribution_real = sum(
            snapshot.contribution_krw * base_cpi / level
            for snapshot, level in zip(snapshots, self._cpi_levels, strict=True)
        )
        breach = tuple(
            sorted(
                year
                for year, income in self._income.items()
                if income > self._regime.financial_income_threshold_krw
            )
        )
        result = AfterTaxResult(
            config=config,
            snapshots=snapshots,
            disposals=tuple(self._disposals),
            terminal_after_tax_krw=terminal_after_tax,
            terminal_after_tax_real_krw=terminal_real,
            terminal_pre_liquidation_krw=terminal_pre,
            liquidation_tax_krw=self._last_liquidation_tax,
            taxes_paid_krw=self._taxes_paid,
            xirr_after_tax_real=xirr(real_cashflows),
            total_contribution_real_krw=total_contribution_real,
            max_drawdown_after_tax=max_drawdown([snapshot.after_tax_nav_krw for snapshot in snapshots]),
            annual_financial_income_krw=dict(self._income),
            financial_income_breach_years=breach,
            sell_count=self._sell_count,
        )
        logger.info(
            "[PORTFOLIO] event=after_tax_run_done label=%s steps=%d terminal_after_tax_krw=%.2f taxes_paid_krw=%.2f sells=%d",
            config.label,
            len(snapshots),
            terminal_after_tax,
            self._taxes_paid,
            self._sell_count,
        )
        return result
