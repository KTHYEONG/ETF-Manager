"""PIT trailing OLS factor loadings (research; never spliced onto prices)."""

from __future__ import annotations

import math
from datetime import UTC
from typing import TYPE_CHECKING, Final

import polars as pl

from src.etf_manager.data.pit import AVAILABLE_AT, TS_DTYPE

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

FACTOR_COLUMNS: Final[tuple[str, ...]] = ("mkt_rf", "smb", "hml", "rmw", "cma", "mom")
_RF_COLUMN: Final[str] = "rf"
_OUTPUT_KEYS: Final[tuple[str, ...]] = ("alpha", *FACTOR_COLUMNS)
_SINGULAR_TOLERANCE: Final[float] = 1e-12

__all__ = ["FACTOR_COLUMNS", "estimate_factor_loadings", "ols_with_intercept"]


def estimate_factor_loadings(
    prices: pl.DataFrame,
    factors: pl.DataFrame,
    *,
    ticker: str,
    signal_at: datetime,
    window: int = 36,
) -> dict[str, float]:
    """Regress trailing monthly excess returns of ``ticker`` on the six factors.

    Monthly returns come from month-end ``adjusted_close`` sessions whose bar is
    visible at ``signal_at``; factor rows must also be visible at ``signal_at``.
    Only months with complete factors enter, and the last ``window`` such months
    are used. The input frames are never mutated.

    Returns:
        Loadings dict keyed ``alpha, mkt_rf, smb, hml, rmw, cma, mom`` (decimals).

    Raises:
        ValueError: On a naive ``signal_at``, too few usable months, or singular
            normal equations.
    """
    if signal_at.tzinfo is None:
        raise ValueError(f"signal_at must be timezone-aware, got naive datetime {signal_at!r}")
    if window < 1:
        raise ValueError(f"window must be positive, got {window}")
    cutoff = signal_at.astimezone(UTC)
    visible_prices = prices.filter(
        (pl.col("ticker") == ticker) & (pl.col(AVAILABLE_AT) <= pl.lit(cutoff, dtype=TS_DTYPE))
    ).sort("date")
    if visible_prices.height < 2:
        raise ValueError(f"estimate_factor_loadings requires at least 2 visible rows for {ticker!r}")
    monthly_returns = _monthly_returns(visible_prices)
    panel = (
        monthly_returns.join(_visible_factor_panel(factors, cutoff), on="period_end", how="inner")
        .sort("period_end")
        .tail(window)
    )
    if panel.height < window:
        raise ValueError(
            f"estimate_factor_loadings requires {window} visible months for {ticker!r}, got {panel.height}"
        )
    excess = panel.get_column("monthly_return") - panel.get_column(_RF_COLUMN)
    return ols_with_intercept(
        excess.to_list(),
        {name: panel.get_column(name).to_list() for name in FACTOR_COLUMNS},
    )


def ols_with_intercept(y: Sequence[float], regressors: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Closed-form OLS via normal equations; intercept first, no linear-algebra deps.

    Structurally zero regressor columns contribute a coefficient of ``0.0``
    instead of making the system singular; collinearity among nonzero columns
    fails closed.

    Returns:
        Dict with key ``alpha`` plus one entry per named regressor.

    Raises:
        ValueError: On length mismatches, too few observations, or singular X'X.
    """
    observations = len(y)
    lengths = {name: len(values) for name, values in regressors.items()}
    if any(length != observations for length in lengths.values()):
        raise ValueError(f"regressor lengths {lengths} mismatch y length {observations}")
    active = {
        name: [float(value) for value in values]
        for name, values in regressors.items()
        if max((abs(float(value)) for value in values), default=0.0) > 0.0
    }
    names = ("alpha", *active)
    if observations < len(names):
        raise ValueError(f"ols needs at least {len(names)} observations, got {observations}")
    design: list[list[float]] = [[1.0, *[row[i] for row in active.values()]] for i in range(observations)]
    coefficients = _solve_normal_equations(design, [float(value) for value in y])
    resolved = dict(zip(names, coefficients, strict=True))
    return {key: resolved.get(key, 0.0) for key in _OUTPUT_KEYS if key == "alpha" or key in regressors}


def _monthly_returns(visible_prices: pl.DataFrame) -> pl.DataFrame:
    """Month-end simple returns stamped with their own (later) bar availability."""
    return (
        visible_prices.with_columns(pl.col("date").dt.month_end().alias("period_end"))
        .group_by("period_end")
        .agg(pl.col("adjusted_close").last(), pl.col(AVAILABLE_AT).last())
        .sort("period_end")
        .with_columns(
            (pl.col("adjusted_close") / pl.col("adjusted_close").shift(1) - 1.0).alias("monthly_return")
        )
        .slice(1)
        .select("period_end", "monthly_return", AVAILABLE_AT)
    )


def _visible_factor_panel(factors: pl.DataFrame, cutoff: datetime) -> pl.DataFrame:
    """Factor rows visible at ``cutoff``; incomplete gap months are unusable."""
    return (
        factors.filter(pl.col(AVAILABLE_AT) <= pl.lit(cutoff, dtype=TS_DTYPE))
        .sort(["period_end", AVAILABLE_AT])
        .unique(subset=["period_end"], keep="last")
        .select("period_end", *FACTOR_COLUMNS, _RF_COLUMN)
        .drop_nulls()
    )


def _solve_normal_equations(design: list[list[float]], y: list[float]) -> list[float]:
    """Solve (X'X) b = X'y by Gaussian elimination with partial pivoting.

    Raises:
        ValueError: When X'X is numerically singular.
    """
    width = len(design[0])
    gram = [[sum(row[i] * row[j] for row in design) for j in range(width)] for i in range(width)]
    rhs = [sum(row[i] * value for row, value in zip(design, y, strict=True)) for i in range(width)]
    scale = max((abs(value) for row in gram for value in row), default=0.0)
    tolerance = _SINGULAR_TOLERANCE * max(1.0, scale)
    augmented = [[*gram_row[:], rhs_value] for gram_row, rhs_value in zip(gram, rhs, strict=True)]
    for column in range(width):
        pivot_row = max(range(column, width), key=lambda index: abs(augmented[index][column]))
        pivot = augmented[pivot_row][column]
        if abs(pivot) <= tolerance:
            raise ValueError("singular normal equations: factor columns are collinear")
        if pivot_row != column:
            augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]
        for row_index in range(column + 1, width):
            factor = augmented[row_index][column] / pivot
            if factor == 0.0:
                continue
            for column_index in range(column, width + 1):
                augmented[row_index][column_index] -= factor * augmented[column][column_index]
    solution = [0.0] * width
    for row_index in range(width - 1, -1, -1):
        residual = augmented[row_index][width]
        residual -= sum(
            augmented[row_index][index] * solution[index] for index in range(row_index + 1, width)
        )
        solution[row_index] = residual / augmented[row_index][row_index]
    if not all(math.isfinite(value) for value in solution):
        raise ValueError("normal equations produced non-finite coefficients")
    return solution
