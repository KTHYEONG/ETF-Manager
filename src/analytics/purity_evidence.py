"""Holdings purity evidence against curated exposure universe."""
from __future__ import annotations

from datetime import datetime

import polars as pl

from src.analytics.thesis_evidence import EvidenceSlot
from src.data.settings import DataSettings
from src.data.thesis_fundamentals import ExposureNote, load_purity_spec
from src.policy.thesis import ThesisSpec

_PLACEHOLDER_CUSIPS = frozenset({"000000000", "00000000", "999999999"})


def thesis_aligned_weight_pct(
    *, snapshot: pl.DataFrame, notes: tuple[ExposureNote, ...]
) -> dict[str, float | int | str]:
    if snapshot.is_empty():
        return {
            "thesis_aligned_weight_pct": 0.0,
            "non_aligned_weight_pct": 0.0,
            "matched_notes_count": 0,
        }
    isin_set: set[str] = {n.isin for n in notes if n.isin}
    cusip_set: set[str] = {n.cusip for n in notes if n.cusip}
    # map for matched counting
    isin_to_indices: dict[str, list[int]] = {}
    cusip_to_indices: dict[str, list[int]] = {}
    for idx, n in enumerate(notes):
        if n.isin:
            isin_to_indices.setdefault(n.isin, []).append(idx)
        if n.cusip:
            cusip_to_indices.setdefault(n.cusip, []).append(idx)
    total = 0.0
    aligned = 0.0
    matched_indices: set[int] = set()
    # iterate rows
    for row in snapshot.to_dicts():
        w = row.get("weight_pct")
        try:
            wf = float(w) if w is not None else 0.0
        except (TypeError, ValueError):
            wf = 0.0
        total += wf
        isin_val = row.get("isin")
        isin_str: str | None = None
        if isin_val is not None:
            s = str(isin_val).strip()
            if s:
                isin_str = s
        if isin_str is not None:
            if isin_str in isin_set:
                aligned += wf
                for idx in isin_to_indices.get(isin_str, []):
                    matched_indices.add(idx)
            # isin present but not matched => non-aligned, no cusip fallback
            continue
        # isin absent -> try cusip
        cusip_val = row.get("cusip")
        if cusip_val is None:
            continue
        cusip_s = str(cusip_val).strip()
        if not cusip_s or cusip_s in _PLACEHOLDER_CUSIPS:
            continue
        if cusip_s in cusip_set:
            aligned += wf
            for idx in cusip_to_indices.get(cusip_s, []):
                matched_indices.add(idx)
    non_aligned = total - aligned
    if non_aligned < 0 and non_aligned > -1e-9:
        non_aligned = 0.0
    return {
        "thesis_aligned_weight_pct": float(aligned),
        "non_aligned_weight_pct": float(non_aligned),
        "matched_notes_count": len(matched_indices),
    }


def compute_purity_slot(
    *, thesis: ThesisSpec, settings: DataSettings, as_of: datetime
) -> EvidenceSlot | None:
    spec = load_purity_spec(thesis_id=thesis.id)
    if spec is None:
        return None
    if not spec.exposure_notes:
        return EvidenceSlot(
            status="insufficient_data",
            summary="purity: empty exposure universe",
            metrics={"error": "empty exposure_notes"},
        )
    try:
        from src.data.catalog import load_visible
        from src.data.schema import Dataset

        holdings = load_visible(settings, Dataset.ETF_HOLDINGS, as_of)
    except Exception as exc:  # noqa: BLE001
        return EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})
    if holdings.is_empty():
        return EvidenceSlot(
            status="insufficient_data",
            summary="no holdings visible at as_of",
            metrics={"error": "empty holdings"},
        )
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
    try:
        total = float(snapshot.select(pl.col("weight_pct").sum()).item() or 0.0)
        if total < 95.0 or total > 110.0:
            return EvidenceSlot(
                status="insufficient_data",
                summary=f"weight sum {total:.2f} outside band [95,110]",
                metrics={"error": "weight_sum_outside_band", "total": float(total)},
            )
    except Exception as exc:  # noqa: BLE001
        return EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})
    try:
        aligned_dict = thesis_aligned_weight_pct(snapshot=snapshot, notes=spec.exposure_notes)
    except Exception as exc:  # noqa: BLE001
        return EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})
    thesis_aligned = float(aligned_dict.get("thesis_aligned_weight_pct", 0.0))
    non_aligned = float(aligned_dict.get("non_aligned_weight_pct", 0.0))
    matched_count = int(aligned_dict.get("matched_notes_count", 0))
    try:
        from src.analytics.overlap import pairwise_overlap

        rep = pairwise_overlap(
            holdings, vehicle_a=spec.vehicle_ticker, vehicle_b=spec.incumbent_ticker, as_of=as_of
        )
    except Exception as exc:  # noqa: BLE001
        return EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})
    overlap_pct = float(rep.overlap_pct)
    incremental = float(rep.a_only_weight_pct)
    if thesis_aligned >= float(spec.pure_min_pct):
        label = "pure"
    elif thesis_aligned < float(spec.impure_max_pct):
        label = "impure"
    else:
        label = "mixed"
    metrics: dict[str, float | int | str] = {
        "overlap_pct": float(overlap_pct),
        "incremental_weight_pct": float(incremental),
        "thesis_aligned_weight_pct": float(thesis_aligned),
        "non_aligned_weight_pct": float(non_aligned),
        "purity_label": str(label),
        "matched_notes_count": int(matched_count),
        "vehicle_ticker": str(spec.vehicle_ticker),
        "incumbent_ticker": str(spec.incumbent_ticker),
    }
    summary = f"purity: {label} aligned {thesis_aligned:.1f}% incremental {incremental:.1f}% overlap {overlap_pct:.1f}%"
    return EvidenceSlot(status="computed", summary=summary, metrics=metrics)
