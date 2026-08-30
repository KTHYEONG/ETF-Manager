"""Valuation evidence from relative richness and pricing collapse."""
from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from src.analytics.thesis_evidence import EvidenceSlot
from src.data.catalog import load_visible
from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.policy.thesis import ThesisSpec

__all__ = [
    "compute_valuation_slot",
    "pit_price_series",
    "relative_richness_percentile",
    "trailing_total_return_pct",
]


def _price_col(frame: pl.DataFrame) -> str:
    if "adjusted_close" in frame.columns:
        return "adjusted_close"
    if "close" in frame.columns:
        return "close"
    if "value" in frame.columns:
        return "value"
    raise ValueError("price frame missing adjusted_close/close/value")


def _ensure_as_of_aware(as_of: datetime) -> datetime:
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    return as_of.astimezone(UTC)


def pit_price_series(*, prices: pl.DataFrame, ticker: str, as_of: datetime) -> pl.DataFrame:
    if prices.is_empty():
        return prices.clear()
    cutoff = _ensure_as_of_aware(as_of)
    frame = prices
    # filter by ticker if column exists
    if "ticker" in frame.columns:
        frame = frame.filter(pl.col("ticker") == ticker)
    if frame.is_empty():
        return frame.sort("date") if "date" in frame.columns else frame
    # PIT filter
    if "available_at" in frame.columns:
        frame = frame.filter(pl.col("available_at") <= pl.lit(cutoff, dtype=pl.Datetime("us", "UTC")))
    else:
        # fallback to date cutoff
        if "date" in frame.columns:
            frame = frame.filter(pl.col("date") <= pl.lit(cutoff.date(), dtype=pl.Date))
    if frame.is_empty():
        return frame
    # sort by date
    if "date" in frame.columns:
        frame = frame.sort("date", maintain_order=True)
        # dedup per date keeping last
        if frame.height > 1:
            frame = frame.filter(pl.struct("date").is_last_distinct())
            frame = frame.sort("date")
    return frame


def relative_richness_percentile(
    *, vehicle: pl.DataFrame, benchmark: pl.DataFrame, trailing_sessions: int
) -> tuple[float, float]:
    if vehicle.is_empty() or benchmark.is_empty():
        raise ValueError("empty series for richness")
    v_col = _price_col(vehicle)
    b_col = _price_col(benchmark)
    # Prepare renamed frames for join
    v_sel = vehicle.select([pl.col("date"), pl.col(v_col).alias("v_price")]).sort("date")
    b_sel = benchmark.select([pl.col("date"), pl.col(b_col).alias("b_price")]).sort("date")
    joined = v_sel.join(b_sel, on="date", how="inner").sort("date")
    if joined.is_empty():
        raise ValueError("no aligned dates for richness")
    # compute ratios
    ratios = (joined.get_column("v_price") / joined.get_column("b_price")).to_list()
    # numeric filter finite
    ratios_clean: list[float] = []
    for r in ratios:
        try:
            fv = float(r)
        except Exception:  # noqa: BLE001,S112
            continue
        if fv != fv:  # NaN
            continue
        if fv == float("inf") or fv == float("-inf"):
            continue
        ratios_clean.append(fv)
    if not ratios_clean:
        raise ValueError("no valid ratios")
    # trailing window
    if trailing_sessions > 0 and len(ratios_clean) > trailing_sessions:
        window = ratios_clean[-trailing_sessions:]
    else:
        window = ratios_clean
    latest = float(window[-1])
    count_le = sum(1 for v in window if float(v) <= latest)
    pctile = float(count_le) / float(len(window)) * 100.0
    return (latest, pctile)


def trailing_total_return_pct(*, series: pl.DataFrame, lookback_sessions: int) -> float:
    if series.is_empty():
        raise ValueError("empty series for return")
    price_col = _price_col(series)
    sorted_frame = series.sort("date")
    if sorted_frame.height < 2:
        raise ValueError("insufficient rows for return")
    # dedup date
    sorted_frame = sorted_frame.filter(pl.col(price_col).is_not_null() & pl.col(price_col).is_finite())
    if sorted_frame.height < 2:
        raise ValueError("no valid prices")
    n = sorted_frame.height
    if lookback_sessions <= 0:
        raise ValueError("lookback must be positive")
    start_idx = 0 if n <= lookback_sessions else n - 1 - lookback_sessions
    prices_list = sorted_frame.get_column(price_col).to_list()
    latest_price = float(prices_list[-1])
    past_price = float(prices_list[start_idx])
    if past_price == 0:
        raise ValueError("past price zero")
    return (latest_price / past_price - 1.0) * 100.0


def compute_valuation_slot(*, thesis: ThesisSpec, settings: DataSettings, as_of: datetime) -> EvidenceSlot:
    from src.data.thesis_fundamentals import load_valuation_spec

    spec = load_valuation_spec(thesis_id=thesis.id)
    if spec is None:
        return EvidenceSlot(status="unknown", summary="valuation not configured", metrics={})

    try:
        prices = load_visible(settings, Dataset.PRICES, as_of)
    except Exception as exc:  # noqa: BLE001
        return EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})

    try:
        vehicle_series = pit_price_series(prices=prices, ticker=spec.vehicle_ticker, as_of=as_of)
        benchmark_series = pit_price_series(prices=prices, ticker=spec.benchmark_ticker, as_of=as_of)
    except Exception as exc:  # noqa: BLE001
        return EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})

    if vehicle_series.is_empty() or benchmark_series.is_empty():
        return EvidenceSlot(
            status="insufficient_data",
            summary="insufficient price data for valuation",
            metrics={"error": "empty vehicle or benchmark series"},
        )

    # Align via inner join to check min_sessions
    try:
        v_col = _price_col(vehicle_series)
        b_col = _price_col(benchmark_series)
        v_sel = vehicle_series.select([pl.col("date"), pl.col(v_col).alias("v_price")]).sort("date")
        b_sel = benchmark_series.select([pl.col("date"), pl.col(b_col).alias("b_price")]).sort("date")
        joined = v_sel.join(b_sel, on="date", how="inner").sort("date")
    except Exception as exc:  # noqa: BLE001
        return EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})

    if joined.height < spec.min_sessions:
        return EvidenceSlot(
            status="insufficient_data",
            summary=f"insufficient aligned sessions {joined.height} < {spec.min_sessions}",
            metrics={"error": "insufficient_aligned_sessions", "observed": int(joined.height), "required": int(spec.min_sessions)},
        )

    # compute richness
    try:
        latest_ratio, pctile = relative_richness_percentile(
            vehicle=vehicle_series, benchmark=benchmark_series, trailing_sessions=spec.trailing_sessions
        )
    except Exception as exc:  # noqa: BLE001
        return EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})

    if pctile >= float(spec.rich_percentile):
        label = "rich"
    elif pctile <= float(spec.cheap_percentile):
        label = "cheap"
    else:
        label = "fair"

    # trailing return
    try:
        trailing_ret = trailing_total_return_pct(series=vehicle_series, lookback_sessions=spec.return_lookback_sessions)
    except Exception as exc:  # noqa: BLE001
        return EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})

    falsifier_active = bool(trailing_ret < float(spec.collapse_return_pct))

    metrics: dict[str, float | int | str] = {
        "vehicle_ticker": spec.vehicle_ticker,
        "benchmark_ticker": spec.benchmark_ticker,
        "relative_ratio": float(latest_ratio),
        "richness_percentile": float(pctile),
        "richness_label": str(label),
        "trailing_return_pct": float(trailing_ret),
        "falsifier_semiconductor_pricing_collapse_active": bool(falsifier_active),
    }
    summary = f"valuation: {label} richness {pctile:.1f}% ratio {latest_ratio:.4f} return {trailing_ret:.2f}% falsifier {falsifier_active}"
    return EvidenceSlot(status="computed", summary=summary, metrics=metrics)
