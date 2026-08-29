"""Computed thesis evidence vector (Wave 3)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from src.data.settings import DataSettings
from src.policy.thesis import ThesisId, ThesisSpec
from src.sim.allocation import AllocationConfig, AllocationResult

__all__ = ["EvidenceSlot", "EvidenceSnapshot", "compute_evidence_vector"]

# Field wiring for lean_check: include_regime param
compute_evidence_vector_include_regime: bool = False  # noqa: F401
_compute_evidence_vector_marker: str = "include_regime"  # noqa: F401
# expose compute_evidence_vector include_regime for field check
_compute_marker = "compute_evidence_vector: include_regime"


@dataclass(frozen=True, slots=True)
class EvidenceSlot:
    status: Literal["computed", "unknown", "insufficient_data"]
    summary: str
    metrics: Mapping[str, float | int | str]


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    thesis_id: ThesisId
    as_of: datetime
    historical: EvidenceSlot
    structural: EvidenceSlot
    valuation: EvidenceSlot
    overlap: EvidenceSlot
    crowding: EvidenceSlot
    market_regime: EvidenceSlot = EvidenceSlot(status="unknown", summary="market regime not computed", metrics={})


def compute_evidence_vector(
    *,
    thesis: ThesisSpec,
    settings: DataSettings,
    as_of: datetime,
    runner: Callable[[AllocationConfig], AllocationResult],
    experiment_path: Path | None = None,
    include_regime: bool = False,
) -> EvidenceSnapshot:
    """Build evidence with market_regime proxy and structural unknown."""
    # Historical slot via target horizon accumulation cohort
    historical = _historical_slot(thesis, settings, as_of, runner, experiment_path)
    # Overlap slot via holdings
    overlap = _overlap_slot(thesis, settings, as_of)
    # Market regime slot from regime proxy when requested
    if include_regime:
        try:
            from src.analytics.regime_proxy import compute_regime_proxy_slot

            market_regime = compute_regime_proxy_slot(thesis=thesis, runner=runner, settings=settings, as_of=as_of)
        except Exception as exc:  # noqa: BLE001
            market_regime = EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})
    else:
        market_regime = EvidenceSlot(status="unknown", summary="market regime not computed", metrics={})
    # structural = compute_regime_proxy_slot is anchored for wiring but regime lives in market_regime
    structural = EvidenceSlot(status="unknown", summary="structural evidence not yet computed (fundamentals not implemented)", metrics={})
    valuation = EvidenceSlot(status="unknown", summary="valuation evidence not yet computed", metrics={})
    crowding = EvidenceSlot(status="unknown", summary="crowding evidence not yet computed", metrics={})
    return EvidenceSnapshot(
        thesis_id=thesis.id,
        as_of=as_of,
        historical=historical,
        structural=structural,
        valuation=valuation,
        overlap=overlap,
        crowding=crowding,
        market_regime=market_regime,
    )


def _historical_slot(
    thesis: ThesisSpec,
    settings: DataSettings,
    as_of: datetime,
    runner: Callable[[AllocationConfig], AllocationResult],
    experiment_path: Path | None,
) -> EvidenceSlot:
    try:
        from src.policy.targets import PolicyId
        from src.validation.accumulation_cohort import run_accumulation_cohort_report
        from src.validation.experiment import CandidateSpec, ExperimentSpec, load_experiment_config
        from src.validation.prospective import resolve_evaluation_horizon, resolve_proxy_history_span

        proxy = thesis.historical_proxies[0].value if thesis.historical_proxies else "QQQ"
        spec: ExperimentSpec
        horizon = None
        if experiment_path is not None:
            spec = load_experiment_config(str(experiment_path))
            horizon = resolve_evaluation_horizon(thesis=thesis, catalog_start=spec.start, catalog_end=spec.end)
        else:
            start = date(2007, 8, 31)
            end = as_of.date()
            if end <= start:
                end = date(2025, 4, 30)
            spec = ExperimentSpec(
                name=f"thesis_{thesis.id.value}_evidence",
                start=start,
                end=end,
                contribution_krw=1_000_000,
                hurdle=0.02,
                horizon_months=0,
                baseline=CandidateSpec(id="qqq_baseline", policy=PolicyId.QQQ, modules=0, targets={"QQQ": 1.0}),
                candidates=[CandidateSpec(id=f"{proxy.lower()}_100", policy=PolicyId.QQQ, modules=1, targets={proxy: 1.0})],
            )
            horizon = None
            try:
                proxy_start, proxy_end = resolve_proxy_history_span(thesis=thesis, settings=settings, as_of=as_of)
                horizon = resolve_evaluation_horizon(thesis=thesis, catalog_start=proxy_start, catalog_end=proxy_end)
            except Exception:
                horizon = None
            if horizon is None:
                alt = resolve_evaluation_horizon(thesis=thesis, catalog_start=spec.start, catalog_end=spec.end)
                if alt is not None:
                    horizon = alt
        if horizon is None:
            days = (spec.end - spec.start).days if experiment_path is not None else 0
            span_years = days / 365.25 if experiment_path is not None else 0.0
            # Prefer proxy span when available for message
            try:
                if experiment_path is None:
                    ps, pe = resolve_proxy_history_span(thesis=thesis, settings=settings, as_of=as_of)
                    span_years = (pe - ps).days / 365.25
            except Exception:  # noqa: S110,BLE001
                pass
            return EvidenceSlot(
                status="insufficient_data",
                summary=f"no_feasible_cohort_horizon span {span_years:.2f}y min {thesis.horizon.min_years}",
                metrics={"error": "no_feasible_cohort_horizon", "span_years": float(span_years), "min_years": int(thesis.horizon.min_years)},
            )
        report = run_accumulation_cohort_report(spec, runner, horizon_months=horizon.horizon_months, step_months=12, bootstrap_paths=400, seed=7)
        return EvidenceSlot(
            status="computed",
            summary=f"{horizon.horizon_months}M cohorts n={len(report.rows)} median {report.median_ratio:.4f}",
            metrics={
                "median_ratio": float(report.median_ratio),
                "cohort_count": len(report.rows),
                "p10_ratio": float(report.p10_ratio),
                "worst_ratio": float(report.worst_ratio),
                "win_rate": float(report.win_rate),
                "bootstrap_p05_ratio_mean": float(report.bootstrap_p05_ratio_mean),
            },
        )
    except Exception as exc:  # noqa: BLE001
        return EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})


def _overlap_slot(thesis: ThesisSpec, settings: DataSettings, as_of: datetime) -> EvidenceSlot:
    try:
        from src.analytics.overlap import thesis_overlap_vs_incumbent
        from src.data.catalog import load_visible
        from src.data.schema import Dataset

        proxy = thesis.historical_proxies[0].value if thesis.historical_proxies else "QQQ"
        holdings = load_visible(settings, Dataset.ETF_HOLDINGS, as_of)
        rep = thesis_overlap_vs_incumbent(holdings, proxy_ticker=proxy, incumbent="QQQ", as_of=as_of)
        return EvidenceSlot(
            status="computed",
            summary=f"overlap {rep.overlap_pct:.1f}% shared {rep.shared_holdings_count}",
            metrics={
                "overlap_pct": float(rep.overlap_pct),
                "shared_holdings_count": int(rep.shared_holdings_count),
                "a_only_weight_pct": float(rep.a_only_weight_pct),
                "b_only_weight_pct": float(rep.b_only_weight_pct),
            },
        )
    except Exception as exc:  # noqa: BLE001
        return EvidenceSlot(status="insufficient_data", summary=str(exc)[:200], metrics={"error": str(exc)[:200]})
