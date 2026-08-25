"""Ex-post factor attribution of a realized excess path (reporting only; I9)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import polars as pl

from src.features.factors import FACTOR_COLUMNS, ols_with_intercept

if TYPE_CHECKING:
    from collections.abc import Mapping

_MIN_OBSERVATIONS: Final[int] = 36
_DEGENERATE_TOLERANCE: Final[float] = 1e-12

__all__ = ["AttributionResult", "attribute_factor_returns"]


@dataclass(frozen=True, slots=True)
class AttributionResult:
    """OLS attribution summary over the six factors (decimals)."""

    alpha: float
    betas: Mapping[str, float]
    r_squared: float


def attribute_factor_returns(excess_returns: pl.Series, factors: pl.DataFrame) -> AttributionResult:
    """Attribute realized monthly excess simple returns to the six factors.

    Reporting-only diagnostics: the series is aligned with the most recent
    visible factor rows and never feeds back into targets or prices. A
    near-zero-variance excess path is degenerate and fails closed for R^2.

    Returns:
        Alpha, per-factor betas, and R^2 of the intercept OLS fit.

    Raises:
        ValueError: On fewer than 36 observations, missing factor months,
            singular regressions, or a degenerate excess path.
    """
    n = excess_returns.len()
    if n < _MIN_OBSERVATIONS:
        raise ValueError(f"attribution requires at least {_MIN_OBSERVATIONS} observations, got {n}")
    y = [float(value) for value in excess_returns.to_list()]
    panel = (
        factors.select("period_end", *FACTOR_COLUMNS)
        .drop_nulls()
        .sort("period_end")
        .tail(n)
    )
    if panel.height < n:
        raise ValueError(f"attribution requires {n} factor rows, found {panel.height}")
    coefficients = ols_with_intercept(
        y, {name: panel.get_column(name).to_list() for name in FACTOR_COLUMNS}
    )
    fitted = [
        coefficients["alpha"]
        + sum(coefficients[name] * float(panel.item(row, name)) for name in FACTOR_COLUMNS)
        for row in range(n)
    ]
    mean_y = sum(y) / n
    sse = sum((actual - estimate) ** 2 for actual, estimate in zip(y, fitted, strict=True))
    total = sum((value - mean_y) ** 2 for value in y)
    if total > _DEGENERATE_TOLERANCE:
        r_squared = 1.0 - sse / total
    elif abs(sse) <= _DEGENERATE_TOLERANCE:
        r_squared = 1.0
    else:
        raise ValueError("degenerate excess-return path: zero variance with unexplained residual")
    return AttributionResult(
        alpha=coefficients["alpha"],
        betas={name: coefficients.get(name, 0.0) for name in FACTOR_COLUMNS},
        r_squared=r_squared,
    )
