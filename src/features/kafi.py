"""KAFI composite: PIT fear/greed components mapped onto trailing percentile ranks."""

from __future__ import annotations

import math
from datetime import UTC, date
from typing import TYPE_CHECKING, Final

import polars as pl

from src.data.pit import AVAILABLE_AT, TS_DTYPE

if TYPE_CHECKING:
    from datetime import datetime

__all__ = [
    "DEFAULT_CREDIT_SERIES_ID",
    "KAFI_COMPONENT_IDS",
    "KAFI_OPPORTUNITY_COMPONENT_IDS",
    "earliest_kafi_signal_session",
    "kafi_components",
    "kafi_opportunity_components",
    "kafi_opportunity_score",
    "kafi_score",
]

KAFI_COMPONENT_IDS: Final[tuple[str, ...]] = (
    "momentum",
    "drawdown_depth",
    "equity_bond_rel",
    "credit_oas",
    "fx_stress",
    "vol_dampener",
)

KAFI_OPPORTUNITY_COMPONENT_IDS: Final[tuple[str, ...]] = KAFI_COMPONENT_IDS

_SMA_WINDOW: Final[int] = 125
_DRAWDOWN_WINDOW: Final[int] = 252
_RELATIVE_WINDOW: Final[int] = 21
_VOL_WINDOW: Final[int] = 21
_FX_Z_WINDOW: Final[int] = 252
# HY OAS (BAMLH0A0HYM2) is FRED-only; BAA10Y is the ALFRED PIT credit-stress proxy.
DEFAULT_CREDIT_SERIES_ID: Final[str] = "BAA10Y"
# Realized vol is empirically anti-predictive for this accumulator, so its rank is
# compressed toward neutral instead of ever encoding high vol as fear.
_VOL_DAMPENER_ALPHA: Final[float] = 0.25


def kafi_components(
    *,
    prices: pl.DataFrame,
    fx: pl.DataFrame,
    macro: pl.DataFrame,
    equity_ticker: str,
    bond_ticker: str,
    signal_at: datetime,
    rank_window: int = 252,
    credit_series_id: str = DEFAULT_CREDIT_SERIES_ID,
) -> dict[str, float]:
    """Equal-weight component scores in [0, 100]; higher means greed.

    Every component reads only rows whose ``available_at`` stamp is at or before
    ``signal_at`` and maps its latest value onto a mid-percentile rank over the
    last ``rank_window`` visible observations of its own series. The credit
    component selects the pinned HY OAS series id; any missing window fails
    closed rather than imputing a score.

    Raises:
        ValueError: When ``signal_at`` is naive, ``rank_window`` cannot form a
            rank, any input series carries unusable rows, or fewer than
            ``rank_window`` observations remain visible for a component.
    """
    if rank_window < 2:
        raise ValueError(f"rank_window must admit a percentile rank (>= 2), got {rank_window}")
    cutoff = _utc_cutoff(signal_at)
    eq_closes = _ticker_closes(prices, equity_ticker, cutoff)
    scores: dict[str, float] = {
        "momentum": _series_rank(_momentum_values(eq_closes), rank_window, "momentum"),
        "drawdown_depth": _series_rank(_drawdown_values(eq_closes), rank_window, "drawdown_depth"),
        "equity_bond_rel": _relative_rank(prices, equity_ticker, bond_ticker, cutoff, rank_window),
        "credit_oas": _macro_rank(macro, cutoff, rank_window, credit_series_id),
        "fx_stress": _fx_rank(fx, cutoff, rank_window),
    }
    vol_rank = _series_rank(_vol_values(eq_closes), rank_window, "vol_dampener")
    scores["vol_dampener"] = 50.0 + _VOL_DAMPENER_ALPHA * (vol_rank - 50.0)
    ordered = {component: float(scores[component]) for component in KAFI_COMPONENT_IDS}
    for component, value in ordered.items():
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise ValueError(f"kafi component {component!r} produced an out-of-range score {value!r}")
    return ordered


def kafi_opportunity_components(
    *,
    prices: pl.DataFrame,
    fx: pl.DataFrame,
    macro: pl.DataFrame,
    equity_ticker: str,
    bond_ticker: str,
    signal_at: datetime,
    rank_window: int = 252,
    credit_series_id: str = DEFAULT_CREDIT_SERIES_ID,
) -> dict[str, float]:
    """Equal-weight opportunity scores in [0, 100]; higher means a better QQQ entry."""
    greed = kafi_components(
        prices=prices,
        fx=fx,
        macro=macro,
        equity_ticker=equity_ticker,
        bond_ticker=bond_ticker,
        signal_at=signal_at,
        rank_window=rank_window,
        credit_series_id=credit_series_id,
    )
    opportunity = {
        "momentum": 100.0 - greed["momentum"],
        "drawdown_depth": greed["drawdown_depth"],
        "equity_bond_rel": 100.0 - greed["equity_bond_rel"],
        "credit_oas": greed["credit_oas"],
        "fx_stress": 100.0 - greed["fx_stress"],
        "vol_dampener": 50.0,
    }
    ordered = {component: float(opportunity[component]) for component in KAFI_OPPORTUNITY_COMPONENT_IDS}
    for component, value in ordered.items():
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise ValueError(f"kafi opportunity component {component!r} produced an out-of-range score {value!r}")
    return ordered


def kafi_opportunity_score(
    *,
    prices: pl.DataFrame,
    fx: pl.DataFrame,
    macro: pl.DataFrame,
    equity_ticker: str,
    bond_ticker: str,
    signal_at: datetime,
    rank_window: int = 252,
    credit_series_id: str = DEFAULT_CREDIT_SERIES_ID,
) -> float:
    """Equal-weight mean of the six opportunity component scores in [0, 100]."""
    components = kafi_opportunity_components(
        prices=prices,
        fx=fx,
        macro=macro,
        equity_ticker=equity_ticker,
        bond_ticker=bond_ticker,
        signal_at=signal_at,
        rank_window=rank_window,
        credit_series_id=credit_series_id,
    )
    score = sum(components.values()) / len(components)
    if not math.isfinite(score):
        raise ValueError(f"kafi opportunity score collapsed to a non-finite value {score!r}")
    return score


def kafi_score(
    *,
    prices: pl.DataFrame,
    fx: pl.DataFrame,
    macro: pl.DataFrame,
    equity_ticker: str,
    bond_ticker: str,
    signal_at: datetime,
    rank_window: int = 252,
    credit_series_id: str = DEFAULT_CREDIT_SERIES_ID,
) -> float:
    """Equal-weight mean of the six KAFI component scores in [0, 100].

    Raises:
        ValueError: Under the same fail-closed conditions as ``kafi_components``.
    """
    components = kafi_components(
        prices=prices,
        fx=fx,
        macro=macro,
        equity_ticker=equity_ticker,
        bond_ticker=bond_ticker,
        signal_at=signal_at,
        rank_window=rank_window,
        credit_series_id=credit_series_id,
    )
    score = sum(components.values()) / len(components)
    if not math.isfinite(score):
        raise ValueError(f"kafi score collapsed to a non-finite value {score!r}")
    return score


def earliest_kafi_signal_session(
    *,
    prices: pl.DataFrame,
    fx: pl.DataFrame,
    macro: pl.DataFrame,
    equity_ticker: str,
    bond_ticker: str,
    start: date,
    end: date,
    rank_window: int = 252,
    credit_series_id: str = DEFAULT_CREDIT_SERIES_ID,
) -> date | None:
    """First month-end session whose KAFI score is computable on PIT inputs."""
    from src.data.schedule import build_decision_schedule

    for point in build_decision_schedule(start, end):
        try:
            kafi_score(
                prices=prices,
                fx=fx,
                macro=macro,
                equity_ticker=equity_ticker,
                bond_ticker=bond_ticker,
                signal_at=point.signal_at,
                rank_window=rank_window,
                credit_series_id=credit_series_id,
            )
        except ValueError:
            continue
        return point.signal_session
    return None


def _utc_cutoff(signal_at: datetime) -> datetime:
    if signal_at.tzinfo is None:
        raise ValueError(f"signal_at must be timezone-aware, got naive datetime {signal_at!r}")
    return signal_at.astimezone(UTC)


def _visible(frame: pl.DataFrame, cutoff: datetime) -> pl.DataFrame:
    return frame.filter(pl.col(AVAILABLE_AT) <= pl.lit(cutoff, dtype=TS_DTYPE))


def _ticker_closes(prices: pl.DataFrame, ticker: str, cutoff: datetime) -> pl.Series:
    visible = (
        _visible(prices, cutoff)
        .filter(pl.col("ticker") == ticker)
        .sort("date")
        .get_column("adjusted_close")
    )
    if visible.null_count() > 0 or not bool(visible.is_finite().all()) or bool((visible <= 0.0).any()):
        raise ValueError(f"kafi requires finite positive adjusted_close rows for {ticker!r}")
    return visible


def _momentum_values(closes: pl.Series) -> list[float]:
    sma = closes.rolling_mean(_SMA_WINDOW)
    return (closes / sma - 1.0).drop_nulls().to_list()


def _drawdown_values(closes: pl.Series) -> list[float]:
    peak = closes.rolling_max(_DRAWDOWN_WINDOW)
    return (1.0 - closes / peak).drop_nulls().to_list()


def _vol_values(closes: pl.Series) -> list[float]:
    rets = closes / closes.shift(1) - 1.0
    return rets.rolling_std(_VOL_WINDOW, ddof=1).drop_nulls().to_list()


def _compound_expr(close: pl.Expr, window: int) -> pl.Expr:
    ret = close / close.shift(1) - 1.0
    return ret.log1p().rolling_sum(window).exp() - 1.0


def _relative_rank(
    prices: pl.DataFrame, equity_ticker: str, bond_ticker: str, cutoff: datetime, rank_window: int
) -> float:
    visible = _visible(prices, cutoff)
    equity = visible.filter(pl.col("ticker") == equity_ticker).sort("date")
    bond = visible.filter(pl.col("ticker") == bond_ticker).sort("date")
    eq_comp = equity.select(
        pl.col("date"),
        _compound_expr(pl.col("adjusted_close"), _RELATIVE_WINDOW).alias("equity_comp"),
    )
    bond_comp = bond.select(
        pl.col("date"),
        _compound_expr(pl.col("adjusted_close"), _RELATIVE_WINDOW).alias("bond_comp"),
    )
    joined = eq_comp.join(bond_comp, on="date", how="inner").drop_nulls()
    values = (joined.get_column("equity_comp") - joined.get_column("bond_comp")).to_list()
    return _series_rank([float(value) for value in values], rank_window, "equity_bond_rel")


def _macro_rank(
    macro: pl.DataFrame, cutoff: datetime, rank_window: int, credit_series_id: str
) -> float:
    values = _pit_macro_values(macro, credit_series_id, cutoff)
    return _series_rank(values, rank_window, "credit_oas")


def _pit_macro_values(macro: pl.DataFrame, series_id: str, cutoff: datetime) -> list[float]:
    """Latest vintage per observation date visible at ``cutoff``."""
    visible = (
        _visible(macro, cutoff)
        .filter((pl.col("series_id") == series_id) & pl.col("value").is_finite())
        .sort(["observation_date", AVAILABLE_AT])
    )
    if visible.is_empty():
        return []
    deduped = (
        visible.group_by("observation_date", maintain_order=True)
        .agg(pl.col("value").last())
        .sort("observation_date")
    )
    return [float(value) for value in deduped.get_column("value").to_list()]


def _fx_rank(fx: pl.DataFrame, cutoff: datetime, rank_window: int) -> float:
    levels = (
        _visible(fx, cutoff)
        .filter(pl.col("usdkrw").is_finite())
        .sort("date")
        .get_column("usdkrw")
        .cast(pl.Float64)
    )
    z = (
        pl.DataFrame({"level": levels})
        .with_columns(
            pl.col("level").rolling_mean(_FX_Z_WINDOW).alias("mean"),
            pl.col("level").rolling_std(_FX_Z_WINDOW, ddof=1).alias("std"),
        )
        .select(
            pl.when(pl.col("std") > 0.0)
            .then((pl.col("level") - pl.col("mean")) / pl.col("std"))
            .otherwise(0.0)
            .alias("z")
        )
        .get_column("z")
        .drop_nulls()
        .to_list()
    )
    return _series_rank([float(value) for value in z], rank_window, "fx_stress")


def _series_rank(values: list[float], rank_window: int, component: str) -> float:
    """Mid-percentile rank (ties share mass) of the latest value over the trailing window."""
    if len(values) < rank_window:
        raise ValueError(
            f"kafi component {component!r} requires {rank_window} visible observations, "
            f"found {len(values)}"
        )
    window = values[-rank_window:]
    current = window[-1]
    below = sum(1 for value in window if value < current)
    ties = sum(1 for value in window if value == current)
    score = 100.0 * (below + 0.5 * ties) / rank_window
    if not math.isfinite(score):
        raise ValueError(f"kafi component {component!r} produced a non-finite rank")
    return score
