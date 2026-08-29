"""Holdings overlap diagnostics; never calls adoption gate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl

from src.data.query import load_as_of
from src.data.schema import Dataset

__all__ = ["HoldingsOverlapReport", "overlap_time_series", "pairwise_overlap", "thesis_overlap_vs_incumbent"]

_WEIGHT_SUM_MIN = 95.0
_WEIGHT_SUM_RENORM_MAX = 110.0
_PLACEHOLDER_CUSIPS = frozenset({"000000000", "00000000", "999999999"})


@dataclass(frozen=True, slots=True)
class HoldingsOverlapReport:
    """Overlap between two vehicles at a PIT instant."""

    vehicle_a: str
    vehicle_b: str
    as_of: datetime
    overlap_pct: float
    shared_holdings_count: int
    a_only_weight_pct: float
    b_only_weight_pct: float


def _ensure_tz(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    return dt.astimezone(UTC)


def _resolve_pit_holdings(holdings: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    as_of_utc = _ensure_tz(as_of)
    if "available_at" in holdings.columns:
        return load_as_of(holdings, Dataset.ETF_HOLDINGS, as_of_utc)
    cutoff = as_of_utc
    filtered = holdings.filter(pl.col("filing_date") <= pl.lit(cutoff, dtype=pl.Datetime("us", "UTC")))
    return filtered


def _latest_report_snapshot(frame: pl.DataFrame, *, etf_ticker: str) -> pl.DataFrame:
    ticker_frame = frame.filter(pl.col("etf_ticker") == etf_ticker)
    if ticker_frame.is_empty():
        return ticker_frame
    max_report = ticker_frame.select(pl.col("report_date").max()).item()
    snap = ticker_frame.filter(pl.col("report_date") == max_report)
    if snap.is_empty():
        return snap
    # keep max filing_date per holding_id within this snapshot
    # group by holding_id, pick row with max filing_date
    snap_sorted = snap.sort(["holding_id", "filing_date"])
    deduped = snap_sorted.filter(pl.struct("holding_id").is_last_distinct())
    total = float(deduped.select(pl.col("weight_pct").sum()).item() or 0.0)
    if total < _WEIGHT_SUM_MIN or total > _WEIGHT_SUM_RENORM_MAX:
        raise ValueError(f"weight sum {total:.2f} outside band [{_WEIGHT_SUM_MIN:.0f}, {_WEIGHT_SUM_RENORM_MAX:.0f}] for {etf_ticker} report_date {max_report}")
    if total > 100.0:
        scale = 100.0 / total
        deduped = deduped.with_columns((pl.col("weight_pct") * scale).alias("weight_pct"))
    return deduped


def _identifier_key(frame: pl.DataFrame) -> pl.DataFrame:
    # Create matching identifier: cusip preferred else isin else holding_id
    return frame.with_columns(
        pl.when(
            pl.col("cusip").is_not_null()
            & (pl.col("cusip") != "")
            & (~pl.col("cusip").is_in(list(_PLACEHOLDER_CUSIPS)))
        )
        .then(pl.col("cusip"))
        .when(pl.col("isin").is_not_null() & (pl.col("isin") != ""))
        .then(pl.col("isin"))
        .otherwise(pl.col("holding_id"))
        .alias("_match_id")
    )


def pairwise_overlap(
    holdings: pl.DataFrame,
    *,
    vehicle_a: str,
    vehicle_b: str,
    as_of: datetime,
) -> HoldingsOverlapReport:
    """Compute overlap as sum(min(w_a,w_b))/100 over matched identifiers."""
    as_of_utc = _ensure_tz(as_of)
    pit = _resolve_pit_holdings(holdings, as_of_utc)
    if pit.is_empty():
        raise ValueError(f"no PIT row exists for as_of {as_of_utc.isoformat()}")
    if pit.filter(pl.col("etf_ticker") == vehicle_a).is_empty() or pit.filter(pl.col("etf_ticker") == vehicle_b).is_empty():
        missing = vehicle_a if pit.filter(pl.col("etf_ticker") == vehicle_a).is_empty() else vehicle_b
        raise ValueError(f"no PIT row exists for as_of {as_of_utc.isoformat()} for vehicle {missing}")

    a_rows = _latest_report_snapshot(pit, etf_ticker=vehicle_a)
    b_rows = _latest_report_snapshot(pit, etf_ticker=vehicle_b)
    if a_rows.is_empty() or b_rows.is_empty():
        raise ValueError(f"no PIT row exists for as_of {as_of_utc.isoformat()} for vehicle {vehicle_a if a_rows.is_empty() else vehicle_b}")

    # Validate weight_pct in [0,100] - already enforced at ingest but check here
    for frame, label in ((a_rows, vehicle_a), (b_rows, vehicle_b)):
        invalid = frame.filter((pl.col("weight_pct") < 0) | (pl.col("weight_pct") > 100))
        if invalid.height > 0:
            raise ValueError(f"weight_pct out of [0,100] for {label}")

    a_keyed = _identifier_key(a_rows).select(["_match_id", "weight_pct"])
    b_keyed = _identifier_key(b_rows).select(["_match_id", "weight_pct"])

    # Aggregate if duplicate match ids within vehicle (sum?)
    # Use group by _match_id sum weight_pct
    a_agg = a_keyed.group_by("_match_id").agg(pl.col("weight_pct").sum().alias("w_a"))
    b_agg = b_keyed.group_by("_match_id").agg(pl.col("weight_pct").sum().alias("w_b"))

    joined = a_agg.join(b_agg, on="_match_id", how="full", coalesce=True)
    # Fill nulls 0
    joined = joined.with_columns(pl.col("w_a").fill_null(0.0), pl.col("w_b").fill_null(0.0))

    shared = joined.filter((pl.col("w_a") > 0) & (pl.col("w_b") > 0))
    overlap = float(shared.select((pl.min_horizontal("w_a", "w_b")).sum()).item() or 0.0)
    # overlap_pct is sum min /100 already? Requirement: min(w_a,w_b)/100 summed. So if w in pct (0-100), min/100 sums to fraction*100? Actually requirement says min(w_a,w_b)/100 summed => result in 0-100 scale? We'll return overlap as sum(min)/? The spec OVL-A expects A={X:60,Y:40}, B={X:50,Z:50} => overlap 50. That matches sum min =50. So return sum(min) directly (since division by 100? But 50 already is?). To match, sum(min) =50.
    # If we did /100 would get 0.5. So we keep sum(min).
    overlap_pct = float(overlap)

    shared_count = int(shared.height)

    total_a = float(a_agg.select(pl.col("w_a").sum()).item() or 0.0)
    total_b = float(b_agg.select(pl.col("w_b").sum()).item() or 0.0)
    a_only = total_a - overlap_pct
    b_only = total_b - overlap_pct
    if not (0.0 <= a_only <= 100.0):
        raise ValueError(f"a_only_weight_pct {a_only:.2f} outside [0,100]")
    if not (0.0 <= b_only <= 100.0):
        raise ValueError(f"b_only_weight_pct {b_only:.2f} outside [0,100]")

    return HoldingsOverlapReport(
        vehicle_a=vehicle_a,
        vehicle_b=vehicle_b,
        as_of=as_of_utc,
        overlap_pct=overlap_pct,
        shared_holdings_count=shared_count,
        a_only_weight_pct=float(a_only),
        b_only_weight_pct=float(b_only),
    )


def thesis_overlap_vs_incumbent(
    holdings: pl.DataFrame,
    *,
    proxy_ticker: str,
    incumbent: str = "QQQ",
    as_of: datetime,
) -> HoldingsOverlapReport:
    """Convenience for thesis proxy vs incumbent."""
    return pairwise_overlap(holdings, vehicle_a=proxy_ticker, vehicle_b=incumbent, as_of=as_of)


def overlap_time_series(
    holdings: pl.DataFrame,
    *,
    vehicle_a: str,
    vehicle_b: str,
    as_ofs: Sequence[datetime],
) -> tuple[HoldingsOverlapReport, ...]:
    """Compute overlap per PIT instant, skipping missing dates without raising."""
    from collections.abc import Sequence as Seq  # noqa: F401

    reports: list[HoldingsOverlapReport] = []
    for as_of in as_ofs:
        try:
            rep = pairwise_overlap(holdings, vehicle_a=vehicle_a, vehicle_b=vehicle_b, as_of=as_of)
        except ValueError as exc:
            if "no PIT row exists" in str(exc):
                continue
            raise
        reports.append(rep)
    return tuple(reports)
