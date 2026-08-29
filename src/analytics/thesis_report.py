"""Thesis report composition."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from src.analytics.thesis_evidence import EvidenceSnapshot, compute_evidence_vector
from src.data.settings import DataSettings
from src.policy.thesis import ThesisId, ThesisSpec, ThesisStatus, get_thesis, load_thesis_registry
from src.sim.allocation import AllocationConfig, AllocationResult
from src.validation.experiment import CandidateSpec, ExperimentSpec, load_experiment_config, resolve_arm_targets
from src.validation.gate import LongHorizonVerdict, certainty_equivalent, long_horizon_passes
from src.validation.prospective import ProspectiveEligibility, evaluate_prospective_eligibility

__all__ = ["ThesisReport", "build_thesis_report", "write_thesis_report"]


@dataclass(frozen=True, slots=True)
class ThesisReport:
    thesis_id: ThesisId
    evidence: EvidenceSnapshot
    long_horizon: LongHorizonVerdict | None
    prospective: ProspectiveEligibility
    suggested_status: ThesisStatus
    next_falsifier: str
    divergence: Mapping[str, object] | None


def _resolve_evidence_spec(
    thesis: ThesisSpec,
    as_of: datetime,
    experiment_path: Path | None,
) -> ExperimentSpec:
    from src.policy.targets import PolicyId

    proxy = thesis.historical_proxies[0].value if thesis.historical_proxies else "QQQ"
    if experiment_path is not None:
        return load_experiment_config(str(experiment_path))
    end = as_of.date() if as_of.date() > date(2007, 8, 31) else date(2025, 4, 30)
    return ExperimentSpec(
        name=f"thesis_{thesis.id.value}_evidence",
        start=date(2007, 8, 31),
        end=end,
        contribution_krw=1_000_000,
        hurdle=0.02,
        horizon_months=0,
        baseline=CandidateSpec(id="qqq_baseline", policy=PolicyId.QQQ, modules=0, targets={"QQQ": 1.0}),
        candidates=[CandidateSpec(id=f"{proxy.lower()}_100", policy=PolicyId.QQQ, modules=1, targets={proxy: 1.0})],
    )


def _ce_ratio_gamma_2(
    spec: ExperimentSpec,
    runner: Callable[[AllocationConfig], AllocationResult],
) -> float | None:
    if not spec.candidates:
        return None
    candidate = spec.candidates[0]
    baseline_config = AllocationConfig(
        policy=spec.baseline.policy,
        start=spec.start,
        end=spec.end,
        monthly_contribution_krw=float(spec.contribution_krw),
        fill_delay_sessions=1,
        commission_bps=float(spec.commission_bps),
        fx_spread_bps=float(spec.fx_spread_bps),
        targets_override=resolve_arm_targets(spec.baseline),
    )
    candidate_config = AllocationConfig(
        policy=candidate.policy,
        start=spec.start,
        end=spec.end,
        monthly_contribution_krw=float(spec.contribution_krw),
        fill_delay_sessions=1,
        commission_bps=float(spec.commission_bps),
        fx_spread_bps=float(spec.fx_spread_bps),
        targets_override=resolve_arm_targets(candidate),
    )
    try:
        baseline_wealth = float(runner(baseline_config).terminal_wealth_real_krw)
        candidate_wealth = float(runner(candidate_config).terminal_wealth_real_krw)
        baseline_ce = certainty_equivalent((baseline_wealth,), gamma=2.0)
        candidate_ce = certainty_equivalent((candidate_wealth,), gamma=2.0)
        return candidate_ce / baseline_ce
    except ValueError:
        return None


def build_thesis_report(
    *,
    thesis_id: ThesisId,
    settings: DataSettings,
    as_of: datetime,
    runner: Callable[[AllocationConfig], AllocationResult],
    experiment_path: Path | None = None,
) -> ThesisReport:
    """Compose evidence, long-horizon gate, prospective, and divergence."""
    registry = load_thesis_registry(Path("configs/theses"))
    thesis = get_thesis(registry, thesis_id)

    evidence = compute_evidence_vector(
        thesis=thesis, settings=settings, as_of=as_of, runner=runner, experiment_path=experiment_path
    )

    evidence_spec = _resolve_evidence_spec(thesis, as_of, experiment_path)
    ce_ratio = _ce_ratio_gamma_2(evidence_spec, runner)

    # Long horizon verdict from accumulation cohort
    long_horizon: LongHorizonVerdict | None = None
    try:
        from src.validation.accumulation_cohort import run_accumulation_cohort_report

        report = run_accumulation_cohort_report(
            evidence_spec, runner, horizon_months=120, step_months=12, bootstrap_paths=400, seed=7
        )
        long_horizon = long_horizon_passes(report)
    except Exception:  # noqa: BLE001
        long_horizon = None

    # Prospective eligibility from catalog span
    prospective: ProspectiveEligibility
    try:
        from src.data.catalog import load_visible
        from src.data.schema import Dataset

        # Use PRICES dataset span as proxy for catalog span
        try:
            prices = load_visible(settings, Dataset.PRICES, as_of)
            if not prices.is_empty():
                start_raw = prices.get_column("date").min()
                end_raw = prices.get_column("date").max()
                catalog_start = start_raw if isinstance(start_raw, date) else date(2006, 8, 31)
                catalog_end = end_raw if isinstance(end_raw, date) else as_of.date()
            else:
                raise ValueError("empty prices")
        except Exception:
            # fallback to thesis horizon window
            catalog_start = date(2016, 9, 30)
            catalog_end = as_of.date()
        prospective = evaluate_prospective_eligibility(thesis=thesis, catalog_start=catalog_start, catalog_end=catalog_end)
    except Exception as exc:  # noqa: BLE001
        prospective = ProspectiveEligibility(eligible=False, catalog_span_years=0.0, min_years_required=int(thesis.horizon.min_years), reason=str(exc)[:200])

    suggested_status = ThesisStatus.PROSPECTIVE_CHALLENGER if prospective.eligible else thesis.status
    next_falsifier = thesis.falsifiers[0] if thesis.falsifiers else ""

    # Divergence when CE and long horizon metrics coexist
    divergence: Mapping[str, object] | None = None
    if long_horizon is not None and ce_ratio is not None:
        median = (
            float(evidence.historical.metrics.get("median_ratio", long_horizon.median_ratio))
            if evidence.historical.status == "computed"
            else float(long_horizon.median_ratio)
        )
        div: dict[str, object] = {
            "long_horizon_passes": bool(long_horizon.passes),
            "median_ratio": float(median),
            "ce_ratio_gamma_2": float(ce_ratio),
            "long_horizon_median": float(long_horizon.median_ratio),
            "cohort_count": int(long_horizon.cohort_count),
            "overlap_dependence_disclosed": bool(long_horizon.overlap_dependence_disclosed),
        }
        if evidence.historical.status == "computed":
            div["historical_median_ratio"] = float(evidence.historical.metrics.get("median_ratio", median))
        divergence = div

    return ThesisReport(
        thesis_id=thesis_id,
        evidence=evidence,
        long_horizon=long_horizon,
        prospective=prospective,
        suggested_status=suggested_status,
        next_falsifier=next_falsifier,
        divergence=divergence,
    )


def write_thesis_report(report: ThesisReport, settings: DataSettings) -> Path:
    """Persist thesis report JSON under thesis_reports."""
    payload = {
        "thesis_id": report.thesis_id.value,
        "as_of": report.evidence.as_of.isoformat(),
        "evidence": {
            "historical": {"status": report.evidence.historical.status, "summary": report.evidence.historical.summary, "metrics": dict(report.evidence.historical.metrics)},
            "structural": {"status": report.evidence.structural.status, "summary": report.evidence.structural.summary, "metrics": dict(report.evidence.structural.metrics)},
            "valuation": {"status": report.evidence.valuation.status, "summary": report.evidence.valuation.summary, "metrics": dict(report.evidence.valuation.metrics)},
            "overlap": {"status": report.evidence.overlap.status, "summary": report.evidence.overlap.summary, "metrics": dict(report.evidence.overlap.metrics)},
            "crowding": {"status": report.evidence.crowding.status, "summary": report.evidence.crowding.summary, "metrics": dict(report.evidence.crowding.metrics)},
        },
        "long_horizon": None if report.long_horizon is None else {
            "passes": report.long_horizon.passes,
            "cohort_count": report.long_horizon.cohort_count,
            "median_ratio": report.long_horizon.median_ratio,
            "overlap_dependence_disclosed": report.long_horizon.overlap_dependence_disclosed,
            "reason": report.long_horizon.reason,
        },
        "prospective": {
            "eligible": report.prospective.eligible,
            "catalog_span_years": report.prospective.catalog_span_years,
            "min_years_required": report.prospective.min_years_required,
            "reason": report.prospective.reason,
        },
        "suggested_status": report.suggested_status.value,
        "next_falsifier": report.next_falsifier,
        "divergence": dict(report.divergence) if report.divergence is not None else None,
    }
    out_dir = settings.resolved_data_root() / "thesis_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_as_of = report.evidence.as_of.isoformat().replace(":", "-")
    out_path = out_dir / f"{report.thesis_id.value}_{safe_as_of}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
