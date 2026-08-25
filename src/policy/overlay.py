"""Bounded dynamic overlay on strategic sleeve weights."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING, Final

import polars as pl

from src.data.pit import AVAILABLE_AT, TS_DTYPE
from src.features.drawdown import trailing_price_drawdown
from src.features.momentum import trailing_compound_return
from src.features.returns import session_returns
from src.features.risk import trailing_simple_vol
from src.policy.targets import PolicyError

if TYPE_CHECKING:
    from datetime import datetime

__all__ = ["OverlayConfig", "apply_bounded_overlay", "visible_macro_level"]

_MAX_SHIFT_CEILING: Final[float] = 0.10


@dataclass(frozen=True, slots=True)
class OverlayConfig:
    """Bounded overlay parameters; every sleeve scale stays in [1-max_shift, 1+max_shift]."""

    max_shift: float = 0.10
    trend_window: int = 126
    vol_window: int = 63
    drawdown_window: int = 252
    drawdown_trigger: float = -0.15
    vix_threshold: float | None = None
    vix_series_id: str = "VIXCLS"

    def __post_init__(self) -> None:
        if not 0.0 < self.max_shift <= _MAX_SHIFT_CEILING:
            raise ValueError(
                f"max_shift must lie in (0, {_MAX_SHIFT_CEILING}], got {self.max_shift!r}"
            )


def apply_bounded_overlay(
    weights: dict[str, float],
    prices: pl.DataFrame,
    signal_at: datetime,
    overlay: OverlayConfig,
    macro: pl.DataFrame | None = None,
) -> dict[str, float]:
    """Rescale each sleeve weight by ``1 + max_shift * u_i`` using only PIT signals.

    ``u_i`` combines trend (+1/0/-1 at weight 0.5), cross-sleeve relative vol and
    drawdown-trigger signs (each -1/0 at weight 0.25), clipped to [-1, 1]; an
    optional VIX gate subtracts a full unit before clipping again. A boosted sum
    above one is renormalized; a sum below one leaves the residual as cash.
    Feature failures fail closed with ``PolicyError``.

    Raises:
        ValueError: When ``signal_at`` is naive.
        PolicyError: When any required feature window or the optional VIX row is unusable.
    """
    if signal_at.tzinfo is None:
        raise ValueError(f"signal_at must be timezone-aware, got naive datetime {signal_at!r}")
    try:
        trend_signs: dict[str, float] = {}
        sleeve_vols: dict[str, float] = {}
        for sleeve in weights:
            returns = session_returns(prices, ticker=sleeve)
            compound = trailing_compound_return(returns, as_of_ts=signal_at, window=overlay.trend_window)
            trend_signs[sleeve] = (compound > 0.0) - (compound < 0.0)
            sleeve_vols[sleeve] = trailing_simple_vol(returns, as_of_ts=signal_at, window=overlay.vol_window)
        drawdowns = {
            sleeve: trailing_price_drawdown(
                prices, ticker=sleeve, as_of_ts=signal_at, window=overlay.drawdown_window
            )
            for sleeve in weights
        }
    except ValueError as exc:
        raise PolicyError(f"overlay feature failed closed: {exc}") from exc

    mean_vol = sum(sleeve_vols.values()) / len(sleeve_vols)
    scores: dict[str, float] = {}
    for sleeve in weights:
        vol_sign = -1.0 if sleeve_vols[sleeve] > mean_vol else 0.0
        dd_sign = -1.0 if drawdowns[sleeve] < overlay.drawdown_trigger else 0.0
        score = 0.5 * trend_signs[sleeve] + 0.25 * vol_sign + 0.25 * dd_sign
        scores[sleeve] = min(1.0, max(-1.0, score))
    if overlay.vix_threshold is not None:
        vix_level = visible_macro_level(macro, overlay.vix_series_id, signal_at)
        if vix_level > overlay.vix_threshold:
            scores = {sleeve: min(1.0, max(-1.0, score - 1.0)) for sleeve, score in scores.items()}

    shifted = {
        sleeve: max(0.0, weights[sleeve] * (1.0 + overlay.max_shift * score))
        for sleeve, score in scores.items()
    }
    total = sum(shifted.values())
    if total > 1.0:
        shifted = {sleeve: weight / total for sleeve, weight in shifted.items()}
    return shifted


def visible_macro_level(macro: pl.DataFrame | None, series_id: str, signal_at: datetime) -> float:
    """Latest finite MACRO level for ``series_id`` visible at ``signal_at``; fail-closed."""
    if macro is None:
        raise PolicyError(f"VIX gate requires a macro frame for {series_id!r}")
    cutoff = signal_at.astimezone(UTC)
    visible = macro.filter(
        (pl.col("series_id") == series_id)
        & (pl.col(AVAILABLE_AT) <= pl.lit(cutoff, dtype=TS_DTYPE))
        & pl.col("value").is_finite()
    ).sort(["observation_date", AVAILABLE_AT])
    if visible.is_empty():
        raise PolicyError(f"no visible macro row for {series_id!r} at the signal instant")
    return float(visible.item(visible.height - 1, "value"))
