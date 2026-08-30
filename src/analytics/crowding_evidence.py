"""Crowding evidence from holdings concentration."""
from __future__ import annotations

from datetime import datetime

import polars as pl

from src.analytics.thesis_evidence import EvidenceSlot
from src.data.catalog import load_visible
from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.policy.thesis import ThesisSpec

__all__ = ["compute_crowding_slot", "holdings_concentration_metrics"]


def holdings_concentration_metrics(*, snapshot: pl.DataFrame, top_n: int) -> dict[str, float]:
    if snapshot.is_empty():
        raise ValueError("empty snapshot")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if "weight_pct" not in snapshot.columns:
        raise ValueError("snapshot missing weight_pct")
    # ensure finite weights
    frame = snapshot.filter(pl.col("weight_pct").is_not_null() & pl.col("weight_pct").is_finite())
    if frame.is_empty():
        raise ValueError("no valid weights")
    # sort descending by weight_pct
    sorted_frame = frame.sort("weight_pct", descending=True)
    weights = sorted_frame.get_column("weight_pct").to_list()
    # hhi on fractions
    fractions = [float(w) / 100.0 for w in weights]
    hhi = float(sum(f * f for f in fractions))
    top_n_weights = weights[:top_n] if len(weights) >= top_n else weights
    top_sum = float(sum(float(w) for w in top_n_weights))
    # For consistency with spec top5, if top_n !=5 but spec asks 5, caller passes 5, so top5 sum equals top_n sum.
    # But metrics key is top5_weight_pct even if top_n differs? Requirement says top5_weight_pct.
    # Use top5_weight_pct key always.
    effective_n = float(1.0 / hhi) if hhi > 0 else float("inf")
    holdings_count = float(len(weights))
    return {
        "hhi": float(hhi),
        "top5_weight_pct": float(top_sum),
        "effective_n": float(effective_n),
        "holdings_count": float(holdings_count),
    }


def compute_crowding_slot(*, thesis: ThesisSpec, settings: DataSettings, as_of: datetime) -> EvidenceSlot:
    from src.data.thesis_fundamentals import load_crowding_spec

    spec = load_crowding_spec(thesis_id=thesis.id)
    if spec is None:
        return EvidenceSlot(status="unknown", summary="crowding not configured", metrics={})

    try:
        holdings = load_visible(settings, Dataset.ETF_HOLDINGS, as_of)
    except Exception as exc:  # noqa: BLE001
        return EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})

    if holdings.is_empty():
        return EvidenceSlot(
            status="insufficient_data",
            summary="no holdings visible at as_of",
            metrics={"error": "empty holdings"},
        )

    # use overlap._latest_report_snapshot
    try:
        from src.analytics.overlap import _latest_report_snapshot

        snapshot = _latest_report_snapshot(holdings, etf_ticker=spec.vehicle_ticker)
    except Exception as exc:  # noqa: BLE001
        return EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})

    if snapshot.is_empty():
        return EvidenceSlot(
            status="insufficient_data",
            summary=f"no snapshot for {spec.vehicle_ticker}",
            metrics={"error": "empty snapshot"},
        )

    # _latest_report_snapshot already validates weight band and may raise; empty handled above.
    # Additional guard if caller bypassed validation: check weight sum band explicit
    try:
        total = float(snapshot.select(pl.col("weight_pct").sum()).item() or 0.0)
        # If snapshot was not via _latest_report_snapshot path due to mock, still enforce band
        if total < 95.0 or total > 110.0:
            return EvidenceSlot(
                status="insufficient_data",
                summary=f"weight sum {total:.2f} outside band [95,110]",
                metrics={"error": "weight_sum_outside_band", "total": float(total)},
            )
    except Exception as exc:  # noqa: BLE001
        return EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})

    try:
        conc = holdings_concentration_metrics(snapshot=snapshot, top_n=spec.top_n)
    except Exception as exc:  # noqa: BLE001
        return EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})

    hhi = float(conc["hhi"])
    top5 = float(conc["top5_weight_pct"])
    if hhi > float(spec.concentrated_hhi_threshold) or top5 > float(spec.concentrated_top5_pct):
        label = "concentrated"
    else:
        label = "dispersed"

    metrics: dict[str, float | int | str] = {
        "vehicle_ticker": spec.vehicle_ticker,
        "hhi": float(hhi),
        "top5_weight_pct": float(top5),
        "effective_n": float(conc["effective_n"]),
        "holdings_count": int(conc["holdings_count"]),
        "concentration_label": str(label),
        "top_n": int(spec.top_n),
    }
    summary = f"crowding: {label} hhi {hhi:.4f} top5 {top5:.1f}% n {int(conc['holdings_count'])}"
    return EvidenceSlot(status="computed", summary=summary, metrics=metrics)
