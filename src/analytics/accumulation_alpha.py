"""Reporting-only QQQ buy-cadence accumulation screen; never an adoption gate."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import polars as pl

from src.data.calendar import (
    DEFAULT_CALENDAR_NAME,
    clamp_inclusive_session_range,
    load_calendar,
    next_execution_session,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "AccumulationScreenReport",
    "ArmScreenRow",
    "ArmVerdict",
    "screen_qqq_accumulation",
]

_BASELINE_ARM: Final[str] = "month_end"
_MIN_MONTH_SESSIONS: Final[int] = 8

_ARM_ORDER: Final[tuple[str, ...]] = (
    "month_end",
    "month_open",
    "twice_monthly",
    "session_k1",
    "session_k2",
    "session_k4",
    "session_k9",
    "weekly4",
    "dip2_wait5",
    "dip3_wait10",
    "dip5_wait15",
)

_SECONDARY_ARMS: Final[frozenset[str]] = frozenset(
    {
        "twice_monthly",
        "weekly4",
        "session_k1",
        "session_k9",
        "dip2_wait5",
        "dip3_wait10",
        "dip5_wait15",
    }
)

_SPLIT_ARMS: Final[frozenset[str]] = frozenset({"twice_monthly", "weekly4"})

# 이름 -> (월초 대비 낙폭 theta, 미발동 시 fallback 세션 인덱스 max_wait)
_DIP_SPECS: Final[dict[str, tuple[float, int]]] = {
    "dip2_wait5": (0.02, 5),
    "dip3_wait10": (0.03, 10),
    "dip5_wait15": (0.05, 15),
}


class ArmVerdict(StrEnum):
    """Outcome class of one screened arm; only ``adopt`` may unlock operations."""

    REJECT = "reject"
    ADOPT = "adopt"
    RESEARCH_ONLY = "research_only"


@dataclass(frozen=True, slots=True)
class ArmScreenRow:
    """Reporting-only outcome of one frozen arm versus the month-end baseline."""

    name: str
    verdict: ArmVerdict
    terminal_wealth: float
    ratio_vs_month_end: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    mean_log_fill_gap_vs_end: float | None
    log_fill_p_value: float | None


@dataclass(frozen=True, slots=True)
class AccumulationScreenReport:
    """Full screen output on identical cashflows; unlock stays False unless some arm adopts."""

    ticker: str
    start: date
    end: date
    usable_months: int
    rows: tuple[ArmScreenRow, ...]
    operational_unlock: bool
    recommended_research_arm: str | None


def _signal_entries(
    name: str,
    days: Sequence[date],
    closes: Sequence[float],
) -> tuple[tuple[date, float], ...]:
    """Resolve one frozen arm's in-month signal sessions and contribution weights."""
    count = len(days)
    if name == "month_end":
        return ((days[count - 1], 1.0),)
    if name == "month_open":
        return ((days[0], 1.0),)
    if name == "twice_monthly":
        return ((days[0], 0.5), (days[count - 1], 0.5))
    if name == "weekly4":
        indices = sorted({0, count // 4, count // 2, (3 * count) // 4})
        return tuple((days[index], 0.25) for index in indices)
    if name in _DIP_SPECS:
        theta, max_wait = _DIP_SPECS[name]
        threshold = closes[0] * (1.0 - theta)
        search_window = min(max_wait + 1, count)
        triggered = next(
            (index for index in range(search_window) if closes[index] <= threshold),
            None,
        )
        index = min(max_wait, count - 1) if triggered is None else triggered
        return ((days[index], 1.0),)
    offset = int(name.removeprefix("session_k"))
    return ((days[min(offset, count - 1)], 1.0),)


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    """Linear-interpolated percentile of an ascending sequence."""
    position = (len(ordered) - 1) * quantile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _two_sided_p_value(gaps: Sequence[float]) -> float | None:
    """Normal-approximation two-sided p-value of the paired mean; None when undefined."""
    count = len(gaps)
    if count < 2:
        return None
    mean = sum(gaps) / count
    deviation = math.sqrt(sum((gap - mean) ** 2 for gap in gaps) / (count - 1))
    if deviation == 0.0:
        return 1.0 if mean == 0.0 else 0.0
    statistic = mean / (deviation / math.sqrt(count))
    return math.erfc(abs(statistic) / math.sqrt(2.0))


def screen_qqq_accumulation(
    *,
    prices: pl.DataFrame,
    ticker: str = "QQQ",
    start: date,
    end: date,
    monthly_contribution: float,
    hurdle: float = 0.02,
    bootstrap_draws: int = 4000,
    seed: int = 7,
    calendar_name: str = DEFAULT_CALENDAR_NAME,
) -> AccumulationScreenReport:
    """Screen frozen buy-cadence arms over identical monthly cashflows.

    Every calendar month spends ``monthly_contribution`` exactly once per arm
    (split arms divide it across their fills); fills always execute one full
    exchange session after the signal. A month is usable only when it holds at
    least eight in-range sessions and every arm's fills resolve inside the
    price frame; all arms are then compared on that same month set.

    Args:
        prices: Price frame with ``ticker``, ``date``, ``close``, ``adjusted_close``.
        ticker: Vehicle to screen; defaults to QQQ.
        start: Inclusive range start.
        end: Inclusive range end.
        monthly_contribution: Cash credited once per calendar month per arm.
        hurdle: Adoption margin required above a ratio of 1.0.
        bootstrap_draws: Number of month-resample draws behind each CI.
        seed: Deterministic RNG seed for the resampling.
        calendar_name: Exchange calendar backing session arithmetic.

    Returns:
        Reporting-only report; no adoption gate or policy lock is consulted.

    Raises:
        ValueError: On non-positive contribution, negative hurdle, fewer than
            one bootstrap draw, a missing ticker, or zero usable months.
    """
    if monthly_contribution <= 0.0:
        raise ValueError(f"monthly_contribution must be positive, got {monthly_contribution!r}")
    if hurdle < 0.0:
        raise ValueError(f"hurdle must be non-negative, got {hurdle!r}")
    if bootstrap_draws < 1:
        raise ValueError(f"bootstrap_draws must be at least 1, got {bootstrap_draws!r}")

    ticker_frame = prices.filter(pl.col("ticker") == ticker)
    if ticker_frame.is_empty():
        raise ValueError(f"ticker {ticker!r} missing from prices")
    adjusted: dict[date, float] = {}
    closes: dict[date, float] = {}
    for record in ticker_frame.iter_rows(named=True):
        day: date = record["date"]
        adjusted[day] = float(record["adjusted_close"])
        closes[day] = float(record["close"])

    calendar = load_calendar(calendar_name)
    start, end = clamp_inclusive_session_range(calendar, start, end)
    in_range_days = [day for day in adjusted if start <= day <= end]
    if not in_range_days:
        raise ValueError(f"no {ticker!r} prices within [{start.isoformat()}, {end.isoformat()}]")
    last_price_date = max(in_range_days)
    last_adjusted_close = adjusted[last_price_date]

    month_map: dict[tuple[int, int], list[date]] = {}
    for day in calendar.sessions(start, end):
        month_map.setdefault((day.year, day.month), []).append(day)

    month_shares: dict[str, list[float]] = {name: [] for name in _ARM_ORDER}
    single_fills: dict[str, list[float]] = {name: [] for name in _ARM_ORDER}
    usable_months = 0
    for days in month_map.values():
        if len(days) < _MIN_MONTH_SESSIONS or any(day not in closes for day in days):
            continue
        month_closes = [closes[day] for day in days]
        resolved: dict[str, list[tuple[float, float]]] = {}
        failed = False
        for name in _ARM_ORDER:
            entries: list[tuple[float, float]] = []
            for signal_day, weight in _signal_entries(name, days, month_closes):
                fill_day = next_execution_session(calendar, signal_day, 1)
                fill_price = adjusted.get(fill_day)
                if fill_price is None or fill_day > last_price_date:
                    failed = True
                    break
                entries.append((weight, fill_price))
            if not entries:
                failed = True
                break
            resolved[name] = entries
        if failed:
            continue
        usable_months += 1
        for name in _ARM_ORDER:
            fills = resolved[name]
            month_shares[name].append(sum(weight / price for weight, price in fills))
            if len(fills) == 1:
                single_fills[name].append(fills[0][1])

    if usable_months == 0:
        raise ValueError(
            f"zero usable months for {ticker!r}; every month lacked sessions or resolvable fills"
        )

    totals = {name: sum(month_shares[name]) * last_adjusted_close for name in _ARM_ORDER}
    baseline_total = totals[_BASELINE_ARM]
    ratios = {name: totals[name] / baseline_total for name in _ARM_ORDER}

    rng = random.Random(seed)  # noqa: S311 -- deterministic seeded resampling, not cryptographic
    baseline_series = month_shares[_BASELINE_ARM]
    draw_ratios: dict[str, list[float]] = {name: [] for name in _ARM_ORDER}
    for _ in range(bootstrap_draws):
        picks = rng.choices(range(usable_months), k=usable_months)
        denominator = sum(baseline_series[index] for index in picks)
        for name in _ARM_ORDER:
            numerator = sum(month_shares[name][index] for index in picks)
            draw_ratios[name].append(numerator / denominator)

    confidence: dict[str, tuple[float, float]] = {}
    for name in _ARM_ORDER:
        ordered = sorted(draw_ratios[name])
        confidence[name] = (_percentile(ordered, 2.5), _percentile(ordered, 97.5))

    baseline_fills = single_fills[_BASELINE_ARM]
    stats: dict[str, tuple[float | None, float | None]] = {}
    for name in _ARM_ORDER:
        if name in _SPLIT_ARMS:
            stats[name] = (None, None)
            continue
        gaps = [
            math.log(end_fill / arm_fill)
            for end_fill, arm_fill in zip(baseline_fills, single_fills[name], strict=True)
        ]
        stats[name] = (sum(gaps) / len(gaps), _two_sided_p_value(gaps))

    open_ratio = ratios["month_open"]
    rows: list[ArmScreenRow] = []
    for name in _ARM_ORDER:
        if name == _BASELINE_ARM or ratios[name] <= 1.0 or (name in _SECONDARY_ARMS and ratios[name] < open_ratio):
            verdict = ArmVerdict.REJECT
        elif ratios[name] > 1.0 + hurdle and confidence[name][0] > 1.0:
            verdict = ArmVerdict.ADOPT
        else:
            verdict = ArmVerdict.RESEARCH_ONLY
        rows.append(
            ArmScreenRow(
                name=name,
                verdict=verdict,
                terminal_wealth=totals[name],
                ratio_vs_month_end=ratios[name],
                bootstrap_ci_low=confidence[name][0],
                bootstrap_ci_high=confidence[name][1],
                mean_log_fill_gap_vs_end=stats[name][0],
                log_fill_p_value=stats[name][1],
            )
        )

    challengers = [row for row in rows if row.verdict is not ArmVerdict.REJECT]
    recommended = max(challengers, key=lambda row: row.terminal_wealth).name if challengers else None
    return AccumulationScreenReport(
        ticker=ticker,
        start=start,
        end=end,
        usable_months=usable_months,
        rows=tuple(rows),
        operational_unlock=any(row.verdict is ArmVerdict.ADOPT for row in rows),
        recommended_research_arm=recommended,
    )
