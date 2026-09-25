"""Standalone pension-account arm ranking under uncertain tech premium."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final

from src.validation.bootstrap import moving_block_bootstrap
from src.validation.gate import wealth_quantile

logger = logging.getLogger(__name__)

__all__ = [
    "ArmGrowthRow",
    "CapmScenarioModel",
    "DcaTailStats",
    "DeltaEstimate",
    "GrowthRegretTable",
    "MonthlyReturnPanel",
    "PensionSelectionDataError",
    "bootstrap_dca_tail",
    "build_monthly_krw_panel",
    "delta_grid",
    "estimate_tilt_delta",
    "expected_returns",
    "fit_capm_scenario_model",
    "growth_regret_table",
]

_MONTHS_PER_YEAR: Final[float] = 12.0
_MIN_RETURN_MONTHS: Final[int] = 24
_WEIGHT_SUM_TOLERANCE: Final[float] = 1e-9
_PARALLEL_GROWTH_TOLERANCE: Final[float] = 1e-12
_ZERO_VARIANCE_TOLERANCE: Final[float] = 1e-12


class PensionSelectionDataError(RuntimeError):
    """Raised when the return panel cannot be built without splicing or imputing data."""


@dataclass(frozen=True, slots=True)
class MonthlyReturnPanel:
    """Month-end simple KRW returns net of unrecoverable foreign withholding.

    ``returns[m][k]`` is the return of ``tickers[k]`` from month-end ``m-1`` to
    month-end ``m``. The base calendar month carries no return.
    """

    tickers: tuple[str, ...]
    month_ends: tuple[date, ...]
    returns: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class CapmScenarioModel:
    """Annual covariance, CAPM betas, and tech-residual loadings defining scenario returns."""

    tickers: tuple[str, ...]
    covariance_annual: tuple[tuple[float, ...], ...]
    betas: Mapping[str, float]
    tilt_loadings: Mapping[str, float]
    market_weights: Mapping[str, float]
    risk_free_annual: float
    equity_risk_premium_annual: float
    anchor_ticker: str


@dataclass(frozen=True, slots=True)
class DeltaEstimate:
    """Sample and shrunk estimates of the annual tech premium delta."""

    anchor_ticker: str
    n_months: int
    sample_alpha_annual: float
    sample_alpha_se_annual: float
    prior_mean_annual: float
    prior_sd_annual: float
    posterior_mean_annual: float
    posterior_sd_annual: float


@dataclass(frozen=True, slots=True)
class ArmGrowthRow:
    """Annual log growth of one fixed-weight arm across the delta grid."""

    arm_id: str
    volatility_annual: float
    growth_by_delta: tuple[float, ...]
    max_regret: float
    mean_regret: float


@dataclass(frozen=True, slots=True)
class GrowthRegretTable:
    """Growth and regret by arm and delta, with exact baseline breakevens."""

    deltas: tuple[float, ...]
    baseline_arm_id: str
    rows: tuple[ArmGrowthRow, ...]
    breakeven_vs_baseline: Mapping[str, float | None]


@dataclass(frozen=True, slots=True)
class DcaTailStats:
    """Bootstrap tail outcomes of unit monthly contributions into one arm."""

    arm_id: str
    delta: float
    horizon_months: int
    n_paths: int
    quantile: float
    low_quantile_terminal_multiple: float
    median_terminal_multiple: float
    high_quantile_pre_retirement_drawdown: float
    prob_below_principal: float


def _calendar_months(start: date, end: date) -> tuple[tuple[int, int], ...]:
    year, month = start.year, start.month
    months: list[tuple[int, int]] = []
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return tuple(months)


def build_monthly_krw_panel(
    marks: Mapping[tuple[str, date], float],
    *,
    tickers: Sequence[str],
    start: date,
    end: date,
) -> MonthlyReturnPanel:
    """Build synchronized month-end KRW simple returns from engine marks.

    Args:
        marks: Net-of-withholding KRW marks keyed by ticker and session date.
        tickers: Requested tickers in output column order.
        start: First day of the estimation window; its calendar month is the base.
        end: Last day of the estimation window.

    Returns:
        A panel containing every return month after the base through the end month.

    Raises:
        PensionSelectionDataError: If a ticker lacks a mark in any calendar month.
        ValueError: If tickers are empty or duplicated, or the window is reversed.
    """
    selected_tickers = tuple(tickers)
    if not selected_tickers or any(not ticker.strip() for ticker in selected_tickers):
        raise ValueError("tickers must be non-empty and contain no blank values")
    if len(set(selected_tickers)) != len(selected_tickers):
        raise ValueError("tickers must be unique")
    if start > end:
        raise ValueError("start must not be after end")

    requested = set(selected_tickers)
    latest: dict[tuple[str, int, int], tuple[date, float]] = {}
    for (ticker, day), mark in marks.items():
        if ticker not in requested or not start <= day <= end:
            continue
        key = (ticker, day.year, day.month)
        current = latest.get(key)
        if current is None or day > current[0]:
            latest[key] = (day, float(mark))

    observations: list[tuple[date, dict[str, float]]] = []
    for year, month in _calendar_months(start, end):
        month_marks: dict[str, float] = {}
        month_dates: list[date] = []
        for ticker in selected_tickers:
            observation = latest.get((ticker, year, month))
            if observation is None:
                raise PensionSelectionDataError(f"{ticker} has no mark in {year:04d}-{month:02d}")
            day, mark = observation
            month_dates.append(day)
            month_marks[ticker] = mark
        if not month_dates:  # pragma: no cover - _calendar_months always yields at least one month
            raise PensionSelectionDataError(f"panel window has no month in {year:04d}-{month:02d}")
        observations.append((max(month_dates), month_marks))

    if len(observations) - 1 < _MIN_RETURN_MONTHS:
        raise PensionSelectionDataError(
            f"monthly return panel requires at least {_MIN_RETURN_MONTHS} returns, got {len(observations) - 1}"
        )
    month_ends = tuple(day for day, _ in observations[1:])
    returns = tuple(
        tuple(
            observations[position + 1][1][ticker] / observations[position][1][ticker] - 1.0
            for ticker in selected_tickers
        )
        for position in range(len(observations) - 1)
    )
    return MonthlyReturnPanel(tickers=selected_tickers, month_ends=month_ends, returns=returns)


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_variance(values: Sequence[float], *, ddof: int) -> float:
    if len(values) <= ddof:
        raise ValueError(f"return panel must contain more than {ddof} observations")
    average = _mean(values)
    return math.fsum((value - average) ** 2 for value in values) / (len(values) - ddof)


def _sample_covariance(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean, right_mean = _mean(left), _mean(right)
    return math.fsum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    ) / (len(left) - 1)


def _normalized_weights(weights: Mapping[str, float], *, label: str) -> dict[str, float]:
    if not weights:
        raise ValueError(f"{label} must be non-empty")
    normalized: dict[str, float] = {}
    for ticker, raw_weight in weights.items():
        if not ticker or isinstance(raw_weight, bool) or not isinstance(raw_weight, float | int):
            raise ValueError(f"{label} weight for {ticker!r} must be a finite number in (0, 1]")
        weight = float(raw_weight)
        if not math.isfinite(weight) or not 0.0 < weight <= 1.0:
            raise ValueError(f"{label} weight for {ticker!r} must be a finite number in (0, 1]")
        normalized[ticker] = weight
    total = math.fsum(normalized.values())
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"{label} weights must sum to 1, got {total!r}")
    return normalized


def _market_returns(
    panel: MonthlyReturnPanel,
    market_weights: Mapping[str, float],
) -> tuple[float, ...]:
    indices = {ticker: index for index, ticker in enumerate(panel.tickers)}
    selected = tuple((ticker, market_weights[ticker]) for ticker in sorted(market_weights))
    return tuple(
        math.fsum(weight * row[indices[ticker]] for ticker, weight in selected)
        for row in panel.returns
    )


def fit_capm_scenario_model(
    panel: MonthlyReturnPanel,
    *,
    market_weights: Mapping[str, float],
    anchor_ticker: str,
    risk_free_annual: float,
    equity_risk_premium_annual: float,
) -> CapmScenarioModel:
    """Estimate annual covariance, CAPM betas, and tech-residual loadings.

    Raises:
        ValueError: If weights, rates, references, or anchor residual variation are invalid.
    """
    weights = _normalized_weights(market_weights, label="market_weights")
    panel_tickers = set(panel.tickers)
    unknown_market = sorted(set(weights) - panel_tickers)
    if unknown_market:
        raise ValueError(f"market weights reference tickers outside the panel: {unknown_market}")
    if anchor_ticker not in panel_tickers:
        raise ValueError(f"anchor ticker {anchor_ticker!r} is outside the panel")
    if not math.isfinite(risk_free_annual) or not math.isfinite(equity_risk_premium_annual):
        raise ValueError("CAPM rates must be finite")

    n_months = len(panel.returns)
    if n_months < 3:
        raise ValueError("CAPM scenario estimation requires at least three monthly returns")
    monthly_risk_free = risk_free_annual / _MONTHS_PER_YEAR
    market = _market_returns(panel, weights)
    market_excess = tuple(value - monthly_risk_free for value in market)
    market_variance = _sample_variance(market_excess, ddof=1)
    if market_variance <= 0.0:
        raise ValueError("market proxy has zero excess-return variance")

    index = {ticker: position for position, ticker in enumerate(panel.tickers)}
    asset_excess = {
        ticker: tuple(row[index[ticker]] - monthly_risk_free for row in panel.returns)
        for ticker in panel.tickers
    }
    covariance = tuple(
        tuple(
            _sample_covariance(asset_excess[left], asset_excess[right]) * _MONTHS_PER_YEAR
            for right in panel.tickers
        )
        for left in panel.tickers
    )
    betas = {
        ticker: _sample_covariance(asset_excess[ticker], market_excess) / market_variance
        for ticker in panel.tickers
    }
    residuals = {
        ticker: tuple(value - betas[ticker] * market_value for value, market_value in zip(
            asset_excess[ticker], market_excess, strict=True
        ))
        for ticker in panel.tickers
    }
    anchor_variance = _sample_variance(residuals[anchor_ticker], ddof=1)
    if anchor_variance <= _ZERO_VARIANCE_TOLERANCE:
        raise ValueError("anchor CAPM residual variance must exceed 1e-12")
    tilt_loadings = {
        ticker: (
            1.0
            if ticker == anchor_ticker
            else _sample_covariance(residuals[ticker], residuals[anchor_ticker]) / anchor_variance
        )
        for ticker in panel.tickers
    }
    return CapmScenarioModel(
        tickers=panel.tickers,
        covariance_annual=covariance,
        betas=betas,
        tilt_loadings=tilt_loadings,
        market_weights=weights,
        risk_free_annual=risk_free_annual,
        equity_risk_premium_annual=equity_risk_premium_annual,
        anchor_ticker=anchor_ticker,
    )


def _validate_delta(delta: float) -> float:
    if isinstance(delta, bool) or not math.isfinite(delta):
        raise ValueError("delta must be finite")
    return float(delta)


def expected_returns(model: CapmScenarioModel, delta: float) -> dict[str, float]:
    """Return annual arithmetic expected returns under one tech-premium scenario."""
    scenario_delta = _validate_delta(delta)
    return {
        ticker: (
            model.risk_free_annual
            + model.betas[ticker] * model.equity_risk_premium_annual
            + scenario_delta * model.tilt_loadings[ticker]
        )
        for ticker in model.tickers
    }


def estimate_tilt_delta(
    panel: MonthlyReturnPanel,
    model: CapmScenarioModel,
    *,
    prior_mean_annual: float,
    prior_sd_annual: float,
) -> DeltaEstimate:
    """Shrink the anchor's sample CAPM alpha toward a long-history normal prior.

    Raises:
        ValueError: If the prior is invalid or the anchor residual has no estimable variation.
    """
    if not math.isfinite(prior_mean_annual):
        raise ValueError("prior_mean_annual must be finite")
    if isinstance(prior_sd_annual, bool) or not math.isfinite(prior_sd_annual) or prior_sd_annual <= 0.0:
        raise ValueError("prior_sd_annual must be finite and positive")

    n_months = len(panel.returns)
    monthly_risk_free = model.risk_free_annual / _MONTHS_PER_YEAR
    market_excess = tuple(
        value - monthly_risk_free
        for value in _market_returns(panel, model.market_weights)
    )
    index = {ticker: position for position, ticker in enumerate(panel.tickers)}
    asset_excess = tuple(
        row[index[model.anchor_ticker]] - monthly_risk_free for row in panel.returns
    )
    beta = model.betas[model.anchor_ticker]
    residuals = tuple(value - beta * market_value for value, market_value in zip(
        asset_excess, market_excess, strict=True
    ))
    sample_alpha = (
        _mean(residuals) * _MONTHS_PER_YEAR / model.tilt_loadings[model.anchor_ticker]
    )
    sample_se = (
        math.sqrt(_sample_variance(residuals, ddof=2)) / math.sqrt(n_months) * _MONTHS_PER_YEAR
    )
    if sample_se <= 0.0:
        raise ValueError("anchor sample alpha standard error must be positive")

    prior_precision = 1.0 / prior_sd_annual**2
    sample_precision = 1.0 / sample_se**2
    posterior_precision = prior_precision + sample_precision
    posterior_mean = (
        prior_precision * prior_mean_annual + sample_precision * sample_alpha
    ) / posterior_precision
    posterior_sd = math.sqrt(1.0 / posterior_precision)
    return DeltaEstimate(
        anchor_ticker=model.anchor_ticker,
        n_months=n_months,
        sample_alpha_annual=sample_alpha,
        sample_alpha_se_annual=sample_se,
        prior_mean_annual=prior_mean_annual,
        prior_sd_annual=prior_sd_annual,
        posterior_mean_annual=posterior_mean,
        posterior_sd_annual=posterior_sd,
    )


def delta_grid(estimate: DeltaEstimate, *, z: float, n_points: int) -> tuple[float, ...]:
    """Build an inclusive symmetric grid over posterior mean plus or minus z standard deviations.

    Raises:
        ValueError: If z is not finite and positive or n_points is not odd and at least three.
    """
    if isinstance(z, bool) or not math.isfinite(z) or z <= 0.0:
        raise ValueError("z must be finite and positive")
    if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 3 or n_points % 2 == 0:
        raise ValueError("n_points must be an odd integer >= 3")
    low = estimate.posterior_mean_annual - z * estimate.posterior_sd_annual
    high = estimate.posterior_mean_annual + z * estimate.posterior_sd_annual
    step = (high - low) / (n_points - 1)
    grid = [low + index * step for index in range(n_points)]
    grid[n_points // 2] = estimate.posterior_mean_annual
    return tuple(grid)


def _validated_arms(
    arms: Mapping[str, Mapping[str, float]],
    model_tickers: Sequence[str],
) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    if not arms:
        raise ValueError("arms must be non-empty")
    known = set(model_tickers)
    validated: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    for arm_id, weights in arms.items():
        if not arm_id.strip():
            raise ValueError("arm_id must be non-blank")
        normalized = _normalized_weights(weights, label=f"arm {arm_id!r}")
        unknown = sorted(set(normalized) - known)
        if unknown:
            raise ValueError(f"arm {arm_id!r} references tickers outside the model: {unknown}")
        validated.append((arm_id, tuple(normalized.items())))
    return tuple(validated)


def _arm_components(
    model: CapmScenarioModel,
    weights: Mapping[str, float],
) -> tuple[float, float, float]:
    indices = {ticker: index for index, ticker in enumerate(model.tickers)}
    intercept = math.fsum(
        weight
        * (model.risk_free_annual + model.betas[ticker] * model.equity_risk_premium_annual)
        for ticker, weight in weights.items()
    )
    slope = math.fsum(weight * model.tilt_loadings[ticker] for ticker, weight in weights.items())
    variance = math.fsum(
        weight_left
        * model.covariance_annual[indices[left]][indices[right]]
        * weight_right
        for left, weight_left in weights.items()
        for right, weight_right in weights.items()
    )
    return intercept - 0.5 * variance, slope, math.sqrt(variance)


def growth_regret_table(
    model: CapmScenarioModel,
    arms: Mapping[str, Mapping[str, float]],
    deltas: Sequence[float],
    *,
    baseline_arm_id: str,
) -> GrowthRegretTable:
    """Evaluate fixed-weight log growth and regret for every arm across delta scenarios.

    Raises:
        ValueError: If arms, deltas, weights, baseline, or ticker references are invalid.
    """
    validated_arms = _validated_arms(arms, model.tickers)
    delta_values = tuple(_validate_delta(delta) for delta in deltas)
    if not delta_values:
        raise ValueError("deltas must be non-empty")
    if baseline_arm_id not in arms:
        raise ValueError(f"baseline arm {baseline_arm_id!r} is not present")

    components = {
        arm_id: _arm_components(model, dict(weights)) for arm_id, weights in validated_arms
    }
    growth = {
        arm_id: tuple(intercept + slope * delta for delta in delta_values)
        for arm_id, (intercept, slope, _) in components.items()
    }
    best_by_delta = tuple(max(values[index] for values in growth.values()) for index in range(len(delta_values)))
    rows = tuple(
        ArmGrowthRow(
            arm_id=arm_id,
            volatility_annual=components[arm_id][2],
            growth_by_delta=growth[arm_id],
            max_regret=max(best - value for best, value in zip(best_by_delta, growth[arm_id], strict=True)),
            mean_regret=math.fsum(
                best - value for best, value in zip(best_by_delta, growth[arm_id], strict=True)
            ) / len(delta_values),
        )
        for arm_id, _ in validated_arms
    )
    base_intercept, base_slope, _ = components[baseline_arm_id]
    breakevens = {
        arm_id: (
            None
            if arm_id == baseline_arm_id or abs(slope - base_slope) <= _PARALLEL_GROWTH_TOLERANCE
            else (base_intercept - intercept) / (slope - base_slope)
        )
        for arm_id, (intercept, slope, _) in components.items()
    }
    return GrowthRegretTable(
        deltas=delta_values,
        baseline_arm_id=baseline_arm_id,
        rows=rows,
        breakeven_vs_baseline=breakevens,
    )


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def bootstrap_dca_tail(
    panel: MonthlyReturnPanel,
    model: CapmScenarioModel,
    arms: Mapping[str, Mapping[str, float]],
    *,
    delta: float,
    horizon_months: int,
    n_paths: int,
    block_months: int,
    pre_retirement_months: int,
    quantile: float,
    seed: int,
) -> tuple[DcaTailStats, ...]:
    """Measure paired DCA tail risk on mean-adjusted block-resampled monthly returns.

    Returns:
        One tail-statistics row per arm in input mapping order.

    Raises:
        ValueError: If scenario, sample sizes, probability, weights, or tickers are invalid.
    """
    scenario_delta = _validate_delta(delta)
    validated_arms = _validated_arms(arms, model.tickers)
    panel_months = len(panel.returns)
    horizon = _positive_int(horizon_months, name="horizon_months")
    if horizon > panel_months:
        raise ValueError(f"horizon_months must not exceed panel length {panel_months}")
    paths = _positive_int(n_paths, name="n_paths")
    block = _positive_int(block_months, name="block_months")
    if block > panel_months:
        raise ValueError(f"block_months must not exceed panel length {panel_months}")
    pre_retirement = _positive_int(pre_retirement_months, name="pre_retirement_months")
    if pre_retirement > horizon:
        raise ValueError("pre_retirement_months must not exceed horizon_months")
    if not math.isfinite(quantile) or not 0.0 < quantile < 0.5:
        raise ValueError("quantile must lie in (0, 0.5)")

    expected = expected_returns(model, scenario_delta)
    index = {ticker: position for position, ticker in enumerate(panel.tickers)}
    adjusted_logs: dict[str, tuple[float, ...]] = {}
    for ticker in model.tickers:
        sample_logs = tuple(math.log1p(value) for value in (row[index[ticker]] for row in panel.returns))
        sample_mean = _mean(sample_logs)
        annual_variance = model.covariance_annual[index[ticker]][index[ticker]]
        target_mean = (expected[ticker] - 0.5 * annual_variance) / _MONTHS_PER_YEAR
        shift = target_mean - sample_mean
        adjusted_logs[ticker] = tuple(value + shift for value in sample_logs)

    sampled_paths = moving_block_bootstrap(
        tuple(float(sample_index) for sample_index in range(panel_months)),
        block_size=block,
        n_paths=paths,
        seed=seed,
    )
    plans = tuple(
        (arm_id, tuple((ticker, weight) for ticker, weight in weights))
        for arm_id, weights in validated_arms
    )
    terminal_multiples: dict[str, list[float]] = {arm_id: [] for arm_id, _ in plans}
    drawdowns: dict[str, list[float]] = {arm_id: [] for arm_id, _ in plans}
    trailing_start = horizon - pre_retirement
    for path_index, path in enumerate(sampled_paths, start=1):
        for arm_id, targets in plans:
            units = [0.0] * len(targets)
            peak = 0.0
            worst_drawdown = 0.0
            for step, sampled_month in enumerate(path[:horizon]):
                sampled_index = int(sampled_month)
                for unit_index, target in enumerate(targets):
                    units[unit_index] += target[1]
                for unit_index, (ticker, _weight) in enumerate(targets):
                    units[unit_index] *= math.exp(adjusted_logs[ticker][sampled_index])
                value = math.fsum(units)
                peak = max(peak, value)
                if step >= trailing_start:
                    worst_drawdown = max(worst_drawdown, 1.0 - value / peak)
            terminal_multiples[arm_id].append(math.fsum(units) / horizon)
            drawdowns[arm_id].append(worst_drawdown)
        if path_index % 500 == 0 or path_index == paths:
            logger.info(
                "[ALGO] event=pension_tail_progress delta=%s paths_done=%d n_paths=%d",
                scenario_delta,
                path_index,
                paths,
            )

    return tuple(
        DcaTailStats(
            arm_id=arm_id,
            delta=scenario_delta,
            horizon_months=horizon,
            n_paths=paths,
            quantile=quantile,
            low_quantile_terminal_multiple=wealth_quantile(terminal_multiples[arm_id], quantile),
            median_terminal_multiple=wealth_quantile(terminal_multiples[arm_id], 0.5),
            high_quantile_pre_retirement_drawdown=wealth_quantile(drawdowns[arm_id], 1.0 - quantile),
            prob_below_principal=sum(value < 1.0 for value in terminal_multiples[arm_id]) / paths,
        )
        for arm_id, _ in plans
    )
