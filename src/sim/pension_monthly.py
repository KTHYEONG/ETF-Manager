"""Monthly pension DCA simulator over aligned sleeve return panels."""

from __future__ import annotations

import calendar as _calendar
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from itertools import pairwise
from typing import Final

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

__all__ = [
    "MonthlyReturnPanel",
    "PensionPathResult",
    "WeightSchedule",
    "block_bootstrap_panels",
    "panel_from_prices",
    "panel_from_research",
    "simulate_cohorts",
    "simulate_pension_dca",
]

_WEIGHT_SUM_TOLERANCE: Final[float] = 1e-9
_MONTHS_PER_YEAR: Final[int] = 12


def _require_simplex(weights: Mapping[str, float], label: str) -> dict[str, float]:
    """Copy a sleeve-weight mapping after simplex validation; fail closed."""
    if not weights:
        raise ValueError(f"pension {label} must be non-empty")
    total = 0.0
    cleaned: dict[str, float] = {}
    for sleeve, weight in weights.items():
        if not sleeve or not isinstance(sleeve, str):
            raise ValueError(f"pension {label} carries an invalid sleeve id {sleeve!r}")
        if isinstance(weight, bool) or not isinstance(weight, float | int):
            raise ValueError(f"pension {label} weight for {sleeve!r} must be numeric")
        value = float(weight)
        if not math.isfinite(value):
            raise ValueError(f"pension {label} weight for {sleeve!r} must be finite")
        if value < 0.0:
            raise ValueError(f"pension {label} weight for {sleeve!r} must be non-negative")
        total += value
        cleaned[sleeve] = value
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"pension {label} must sum to 1, got {total!r}")
    return cleaned


def _require_month_end(day: date, label: str) -> None:
    if not isinstance(day, date) or isinstance(day, datetime):
        raise ValueError(f"pension {label} month {day!r} must be a date")
    if day.day != _calendar.monthrange(day.year, day.month)[1]:
        raise ValueError(f"pension {label} month {day.isoformat()} is not a month-end")


def _require_as_of(as_of: datetime) -> datetime:
    if not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError(f"pension as_of must be timezone-aware, got {as_of!r}")
    return as_of.astimezone(UTC)


def _month_index(day: date) -> int:
    return day.year * _MONTHS_PER_YEAR + day.month


def _shift_month(base: date, offset: int) -> date:
    total = base.year * _MONTHS_PER_YEAR + (base.month - 1) + offset
    year, month0 = divmod(total, _MONTHS_PER_YEAR)
    return date(year, month0 + 1, _calendar.monthrange(year, month0 + 1)[1])


@dataclass(frozen=True, slots=True)
class WeightSchedule:
    """Target sleeve weights over the accumulation years of one pension cohort.

    Rebalancing inside a Korean pension account is not a taxable event, so the
    schedule may change weights every contribution year. A fixed mix has
    `glide_years == 0`; a glide path holds `start_weights` through year index
    `total_years - glide_years - 1`, then moves linearly so the final contribution
    year (index `total_years - 1`) uses exactly `end_weights`.

    Attributes:
        start_weights: Simplex over sleeve ids.
        end_weights: Simplex over the same sleeve ids.
        glide_years: Final contribution years spent gliding (0 for a fixed mix).

    Raises:
        ValueError: If weights are not finite, negative, not summing to 1 within
            tolerance, sleeve sets differ, or glide_years is negative.
    """

    start_weights: Mapping[str, float]
    end_weights: Mapping[str, float]
    glide_years: int

    def __post_init__(self) -> None:
        start = _require_simplex(self.start_weights, "start_weights")
        end = _require_simplex(self.end_weights, "end_weights")
        if set(start) != set(end):
            raise ValueError(f"pension schedule sleeve sets differ: {sorted(start)} vs {sorted(end)}")
        if isinstance(self.glide_years, bool) or not isinstance(self.glide_years, int):
            raise ValueError(f"pension glide_years must be an integer, got {self.glide_years!r}")
        if self.glide_years < 0:
            raise ValueError(f"pension glide_years must be non-negative, got {self.glide_years!r}")
        object.__setattr__(self, "start_weights", start)
        object.__setattr__(self, "end_weights", end)

    def weights_for_year(self, year_index: int, total_years: int) -> Mapping[str, float]:
        """Weights applied at the contribution of year `year_index` (0-based) in a `total_years` cohort."""
        if isinstance(year_index, bool) or not isinstance(year_index, int):
            raise ValueError(f"pension year_index must be an integer, got {year_index!r}")
        if isinstance(total_years, bool) or not isinstance(total_years, int) or total_years < 1:
            raise ValueError(f"pension total_years must be a positive integer, got {total_years!r}")
        if not 0 <= year_index < total_years:
            raise ValueError(f"pension year_index {year_index!r} lies outside a {total_years}-year cohort")
        if self.glide_years == 0 or total_years == 1:
            return dict(self.start_weights)
        if self.glide_years > total_years:
            raise ValueError(f"pension glide_years {self.glide_years!r} exceeds total_years {total_years!r}")
        hold_last = total_years - self.glide_years - 1
        if year_index <= hold_last:
            return dict(self.start_weights)
        fraction = (year_index - hold_last) / self.glide_years
        return {
            sleeve: (1.0 - fraction) * self.start_weights[sleeve] + fraction * self.end_weights[sleeve]
            for sleeve in self.start_weights
        }


@dataclass(frozen=True, slots=True)
class MonthlyReturnPanel:
    """Aligned monthly simple returns for every sleeve over contiguous month-ends.

    Attributes:
        tier: Evidence tier label (e.g. `modern`, `century`, `bootstrap`).
        months: Ascending contiguous month-end dates.
        returns: Sleeve id to returns aligned with `months`.

    Raises:
        ValueError: If months are not contiguous month-ends, a sleeve series length
            differs, or any return is non-finite or at or below -1.
    """

    tier: str
    months: tuple[date, ...]
    returns: Mapping[str, tuple[float, ...]]

    def __post_init__(self) -> None:
        if not self.tier or not isinstance(self.tier, str):
            raise ValueError(f"pension panel tier must be a non-empty string, got {self.tier!r}")
        months = tuple(self.months)
        if not months:
            raise ValueError("pension panel months must be non-empty")
        for day in months:
            _require_month_end(day, "panel")
        for earlier, later in pairwise(months):
            if _month_index(later) - _month_index(earlier) != 1:
                raise ValueError(f"pension panel months are not contiguous: {earlier.isoformat()} then {later.isoformat()}")
        if not self.returns:
            raise ValueError("pension panel returns must be non-empty")
        cleaned: dict[str, tuple[float, ...]] = {}
        for sleeve, series in self.returns.items():
            if not sleeve or not isinstance(sleeve, str):
                raise ValueError(f"pension panel carries an invalid sleeve id {sleeve!r}")
            values = tuple(series)
            if len(values) != len(months):
                raise ValueError(
                    f"pension panel series for {sleeve!r} has length {len(values)}, expected {len(months)}"
                )
            for value in values:
                if isinstance(value, bool) or not isinstance(value, float | int):
                    raise ValueError(f"pension panel return for {sleeve!r} must be numeric")
                numeric = float(value)
                if not math.isfinite(numeric) or numeric <= -1.0:
                    raise ValueError(f"pension panel return for {sleeve!r} must be finite and above -1")
            cleaned[sleeve] = tuple(float(value) for value in values)
        object.__setattr__(self, "months", months)
        object.__setattr__(self, "returns", cleaned)


@dataclass(frozen=True, slots=True)
class PensionPathResult:
    """Terminal wealth, funding, and drawdown of one simulated pension cohort."""

    terminal_value: float
    contributed: float
    max_drawdown: float
    pre_retirement_drawdown: float


def _visible_at(frame: pl.DataFrame, as_of: datetime, label: str) -> pl.DataFrame:
    if "available_at" not in frame.columns:
        raise ValueError(f"pension {label} frame misses the availability stamp column")
    return frame.filter(pl.col("available_at") <= pl.lit(as_of))


def panel_from_research(
    frame: pl.DataFrame, sleeve_series: Mapping[str, str], as_of: datetime
) -> MonthlyReturnPanel:
    """Build the century tier from RESEARCH_MONTHLY rows visible at `as_of`.

    Args:
        frame: RESEARCH_MONTHLY rows.
        sleeve_series: Sleeve id to research series id (e.g. SPY to ff_mkt_monthly).
        as_of: Timezone-aware evaluation instant; later availability is invisible.

    Raises:
        ValueError: If a mapped series is absent, months differ between series, or a gap exists.
    """
    cutoff = _require_as_of(as_of)
    if not sleeve_series:
        raise ValueError("pension sleeve_series must be non-empty")
    for column in ("series_id", "period_end", "simple_return"):
        if column not in frame.columns:
            raise ValueError(f"pension research frame misses required column {column!r}")
    visible = _visible_at(frame, cutoff, "research")
    reference: list[date] | None = None
    aligned: dict[str, tuple[float, ...]] = {}
    for sleeve, series_id in sleeve_series.items():
        if not sleeve or not series_id:
            raise ValueError(f"pension sleeve_series carries an invalid mapping {sleeve!r} to {series_id!r}")
        rows = visible.filter(pl.col("series_id") == series_id).sort("period_end")
        if rows.is_empty():
            raise ValueError(f"pension research series {series_id!r} has no visible row at {cutoff.isoformat()}")
        months = rows.get_column("period_end").to_list()
        if len(set(months)) != len(months):
            raise ValueError(f"pension research series {series_id!r} carries duplicate month-ends")
        if reference is None:
            reference = list(months)
        elif list(months) != reference:
            raise ValueError(f"pension research series {series_id!r} covers different months; refusing to splice")
        aligned[sleeve] = tuple(float(value) for value in rows.get_column("simple_return").to_list())
    assert reference is not None
    panel = MonthlyReturnPanel(tier="century", months=tuple(reference), returns=aligned)
    logger.info(
        "[DATA] event=pension_century_panel months=%d sleeves=%d",
        len(panel.months),
        len(panel.returns),
    )
    return panel


def _targets_in_range(start: date, end: date) -> tuple[date, ...]:
    if start > end:
        raise ValueError(f"pension window start {start.isoformat()} is after end {end.isoformat()}")
    targets: list[date] = []
    cursor = date(start.year, start.month, 1)
    while True:
        month_end = date(cursor.year, cursor.month, _calendar.monthrange(cursor.year, cursor.month)[1])
        if month_end > end:
            break
        if month_end >= start:
            targets.append(month_end)
        cursor = _shift_month(cursor, 1)
    if not targets:
        raise ValueError(f"pension window [{start.isoformat()}, {end.isoformat()}] covers no month-end")
    return tuple(targets)


def panel_from_prices(
    prices: pl.DataFrame, sleeves: Sequence[str], as_of: datetime, start: date, end: date
) -> MonthlyReturnPanel:
    """Build the modern tier from month-end total returns of tradable ETFs.

    Month returns use the adjusted close of the last available session of each
    calendar month. Values stay in USD: paired ratios of USD sleeves converted at
    the same instants are invariant to the KRW rate.

    Raises:
        ValueError: If any sleeve lacks a month-end close inside [start, end] or rows are not visible at `as_of`.
    """
    cutoff = _require_as_of(as_of)
    names = tuple(sleeves)
    if not names or len(set(names)) != len(names):
        raise ValueError("pension sleeves must be non-empty and unique")
    for column in ("ticker", "date", "adjusted_close"):
        if column not in prices.columns:
            raise ValueError(f"pension prices frame misses required column {column!r}")
    targets = _targets_in_range(start, end)
    visible = _visible_at(prices, cutoff, "prices")
    aligned: dict[str, tuple[float, ...]] = {}
    for sleeve in names:
        sub = visible.filter(pl.col("ticker") == sleeve).sort("date")
        if sub.is_empty():
            raise ValueError(f"pension prices carry no visible row for {sleeve!r} at {cutoff.isoformat()}")
        last_close: dict[tuple[int, int], tuple[date, float]] = {}
        for day, close in zip(
            sub.get_column("date").to_list(), sub.get_column("adjusted_close").to_list(), strict=True
        ):
            if (
                close is None
                or isinstance(close, bool)
                or not isinstance(close, float | int)
                or not math.isfinite(float(close))
                or float(close) <= 0.0
            ):
                raise ValueError(f"pension adjusted_close for {sleeve!r} on {day!r} must be finite and positive")
            last_close[(day.year, day.month)] = (day, float(close))
        closes: list[float] = []
        for month_end in targets:
            entry = last_close.get((month_end.year, month_end.month))
            if entry is None:
                raise ValueError(f"pension prices for {sleeve!r} lack a month-end close for {month_end.isoformat()}")
            closes.append(entry[1])
        first = targets[0]
        prev_year = first.year if first.month > 1 else first.year - 1
        prev_month = first.month - 1 if first.month > 1 else _MONTHS_PER_YEAR
        prior = last_close.get((prev_year, prev_month))
        if prior is None:
            raise ValueError(f"pension prices for {sleeve!r} lack the prior month-end close before {targets[0].isoformat()}")
        full = [prior[1], *closes]
        aligned[sleeve] = tuple(
            next_close / prev_close - 1.0 for prev_close, next_close in pairwise(full)
        )
    panel = MonthlyReturnPanel(tier="modern", months=targets, returns=aligned)
    logger.info(
        "[DATA] event=pension_modern_panel months=%d sleeves=%d",
        len(panel.months),
        len(panel.returns),
    )
    return panel


def _validate_drag(annual_drag: Mapping[str, float], sleeves: tuple[str, ...]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for sleeve, drag in annual_drag.items():
        if sleeve not in sleeves:
            raise ValueError(f"pension annual_drag names an unknown sleeve {sleeve!r}")
        if isinstance(drag, bool) or not isinstance(drag, float | int):
            raise ValueError(f"pension annual_drag for {sleeve!r} must be numeric")
        value = float(drag)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"pension annual_drag for {sleeve!r} must be finite and non-negative")
        if value >= 1.0:
            raise ValueError(f"pension annual_drag for {sleeve!r} must lie below 1")
        cleaned[sleeve] = value
    return cleaned


def _panel_matrix(panel: MonthlyReturnPanel, sleeves: tuple[str, ...]) -> np.ndarray:
    return np.array([[panel.returns[sleeve][index] for sleeve in sleeves] for index in range(len(panel.months))])


def _run_cohorts(
    net: np.ndarray,
    weight_rows: list[np.ndarray],
    starts: np.ndarray,
    years: int,
    pre_retirement_months: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compound every cohort over shared monthly growth factors.

    Args:
        net: (months, sleeves) monthly growth factors net of drag.
        weight_rows: Per-contribution-year (sleeves,) simplex vectors.
        starts: (cohorts,) panel start indices sharing one return path each.
        years: Contribution years per cohort.

    Returns:
        Terminal values, full-cohort drawdowns, and pre-retirement drawdowns per cohort.
    """
    cohorts = int(starts.shape[0])
    sleeve_count = net.shape[1]
    horizon = years * _MONTHS_PER_YEAR
    holdings = np.zeros((cohorts, sleeve_count), dtype=np.float64)
    values = np.empty((cohorts, horizon), dtype=np.float64)
    for year in range(years):
        weights = weight_rows[year].reshape(1, sleeve_count)
        holdings = (holdings.sum(axis=1, keepdims=True) + 1.0) * weights
        for month in range(_MONTHS_PER_YEAR):
            index = year * _MONTHS_PER_YEAR + month
            holdings = holdings * net[starts + index]
            values[:, index] = holdings.sum(axis=1)
    peaks = np.maximum.accumulate(values, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(peaks > 0.0, values / peaks - 1.0, 0.0)
    max_drawdown = np.minimum(0.0, relative.min(axis=1))
    if pre_retirement_months == 0:
        pre_drawdown = np.zeros(cohorts, dtype=np.float64)
    else:
        tail = relative[:, horizon - pre_retirement_months :]
        pre_drawdown = np.minimum(0.0, tail.min(axis=1))
    return values[:, -1], max_drawdown, pre_drawdown


def _cohort_inputs(
    panel: MonthlyReturnPanel,
    schedule: WeightSchedule,
    years: int,
    pre_retirement_months: int,
    annual_drag: Mapping[str, float],
) -> tuple[tuple[str, ...], np.ndarray, list[np.ndarray]]:
    if isinstance(years, bool) or not isinstance(years, int) or years < 1:
        raise ValueError(f"pension years must be a positive integer, got {years!r}")
    horizon = years * _MONTHS_PER_YEAR
    if (
        isinstance(pre_retirement_months, bool)
        or not isinstance(pre_retirement_months, int)
        or not 0 <= pre_retirement_months <= horizon
    ):
        raise ValueError(f"pension pre_retirement_months must lie in [0, {horizon}], got {pre_retirement_months!r}")
    sleeves = tuple(sorted(schedule.start_weights))
    missing = sorted(set(sleeves) - set(panel.returns))
    if missing:
        raise ValueError(f"pension schedule requires sleeves missing from panel: {missing}")
    drags = _validate_drag(annual_drag, sleeves)
    gross = _panel_matrix(panel, sleeves)
    monthly_factor = np.array(
        [(1.0 - drags.get(sleeve, 0.0)) ** (1.0 / _MONTHS_PER_YEAR) for sleeve in sleeves], dtype=np.float64
    )
    net = (1.0 + gross) * monthly_factor.reshape(1, len(sleeves))
    weight_rows = [
        np.array([schedule.weights_for_year(year, years)[sleeve] for sleeve in sleeves], dtype=np.float64)
        for year in range(years)
    ]
    return sleeves, net, weight_rows


def simulate_pension_dca(
    panel: MonthlyReturnPanel,
    schedule: WeightSchedule,
    *,
    start_month_index: int,
    years: int,
    pre_retirement_months: int,
    annual_drag: Mapping[str, float],
) -> PensionPathResult:
    """Simulate one annual-contribution cohort with yearly tax-free rebalancing.

    One unit is contributed at the first month of each year window, the whole
    account is rebalanced to that year's schedule weights, and holdings compound
    monthly net of the per-sleeve annual drag.

    Raises:
        ValueError: If the cohort extends past the panel, years < 1, or a drag is negative or non-finite.
    """
    if isinstance(start_month_index, bool) or not isinstance(start_month_index, int) or start_month_index < 0:
        raise ValueError(f"pension start_month_index must be a non-negative integer, got {start_month_index!r}")
    _, net, weight_rows = _cohort_inputs(panel, schedule, years, pre_retirement_months, annual_drag)
    horizon = years * _MONTHS_PER_YEAR
    if start_month_index + horizon > len(panel.months):
        raise ValueError(
            f"pension cohort [{start_month_index}, {start_month_index + horizon}) extends past {len(panel.months)} months"
        )
    starts = np.array([start_month_index], dtype=np.int64)
    terminal, max_drawdown, pre_drawdown = _run_cohorts(net, weight_rows, starts, years, pre_retirement_months)
    return PensionPathResult(
        terminal_value=float(terminal[0]),
        contributed=float(years),
        max_drawdown=float(max_drawdown[0]),
        pre_retirement_drawdown=float(pre_drawdown[0]),
    )


def simulate_cohorts(
    panel: MonthlyReturnPanel,
    schedules: Mapping[str, WeightSchedule],
    *,
    years: int,
    step_months: int,
    pre_retirement_months: int,
    annual_drag: Mapping[str, float],
) -> Mapping[str, tuple[PensionPathResult, ...]]:
    """Run every schedule over the same rolling cohort starts so results are paired by index."""
    if not schedules:
        raise ValueError("pension schedules must be non-empty")
    if isinstance(step_months, bool) or not isinstance(step_months, int) or step_months < 1:
        raise ValueError(f"pension step_months must be a positive integer, got {step_months!r}")
    if isinstance(years, bool) or not isinstance(years, int) or years < 1:
        raise ValueError(f"pension years must be a positive integer, got {years!r}")
    horizon = years * _MONTHS_PER_YEAR
    if horizon > len(panel.months):
        raise ValueError(f"pension {years}-year cohort exceeds {len(panel.months)} panel months")
    starts = np.arange(0, len(panel.months) - horizon + 1, step_months, dtype=np.int64)
    results: dict[str, tuple[PensionPathResult, ...]] = {}
    for name, schedule in schedules.items():
        _, net, weight_rows = _cohort_inputs(panel, schedule, years, pre_retirement_months, annual_drag)
        terminal, max_drawdown, pre_drawdown = _run_cohorts(net, weight_rows, starts, years, pre_retirement_months)
        results[name] = tuple(
            PensionPathResult(
                terminal_value=float(terminal_value),
                contributed=float(years),
                max_drawdown=float(drawdown),
                pre_retirement_drawdown=float(pre),
            )
            for terminal_value, drawdown, pre in zip(terminal, max_drawdown, pre_drawdown, strict=True)
        )
    logger.info(
        "[PORTFOLIO] event=pension_cohorts_done schedules=%d cohorts=%d years=%d",
        len(results),
        int(starts.shape[0]),
        years,
    )
    return results


def block_bootstrap_panels(
    panel: MonthlyReturnPanel,
    *,
    n_paths: int,
    horizon_months: int,
    block_months: int,
    seed: int,
) -> tuple[MonthlyReturnPanel, ...]:
    """Resample whole cross-sectional month blocks so sleeve co-movement is preserved.

    Raises:
        ValueError: If block or horizon exceed the panel length or counts are below 1.
    """
    for label, value in (("n_paths", n_paths), ("horizon_months", horizon_months), ("block_months", block_months)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"pension {label} must be a positive integer, got {value!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"pension seed must be an integer, got {seed!r}")
    panel_length = len(panel.months)
    if block_months > panel_length:
        raise ValueError(f"pension block_months {block_months} exceeds panel length {panel_length}")
    if horizon_months > panel_length:
        raise ValueError(f"pension horizon_months {horizon_months} exceeds panel length {panel_length}")
    try:
        rng = np.random.default_rng(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"pension seed {seed!r} is not usable") from exc
    sleeves = tuple(sorted(panel.returns))
    source = _panel_matrix(panel, sleeves)
    months = tuple(_shift_month(panel.months[0], offset) for offset in range(horizon_months))
    latest_block_start = panel_length - block_months
    paths: list[MonthlyReturnPanel] = []
    for _ in range(n_paths):
        draws: list[int] = []
        while len(draws) < horizon_months:
            block_start = int(rng.integers(0, latest_block_start + 1))
            draws.extend(range(block_start, min(block_start + block_months, panel_length)))
        index = np.array(draws[:horizon_months], dtype=np.int64)
        block = source[index]
        returns = {sleeve: tuple(float(value) for value in block[:, position]) for position, sleeve in enumerate(sleeves)}
        paths.append(MonthlyReturnPanel(tier="bootstrap", months=months, returns=returns))
    logger.info(
        "[DATA] event=pension_bootstrap_done paths=%d horizon=%d block=%d",
        n_paths,
        horizon_months,
        block_months,
    )
    return tuple(paths)
