"""Thesis report composition."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from src.analytics.thesis.evidence import EvidenceSnapshot, compute_evidence_vector
from src.data.settings import DataSettings
from src.policy.thesis import ThesisId, ThesisSpec, ThesisStatus, get_thesis, load_thesis_registry
from src.sim.allocation import AllocationConfig, AllocationResult
from src.validation.experiment import CandidateSpec, ExperimentSpec, load_experiment_config, resolve_arm_targets
from src.validation.gate import LongHorizonVerdict, certainty_equivalent, long_horizon_passes
from src.validation.prospective import (
    ProspectiveEligibility,
    evaluate_prospective_eligibility,
    resolve_evaluation_horizon,
    resolve_horizon_surface,
    resolve_proxy_history_span,
)

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
    from src.data.panel_freshness import effective_thesis_end
    from src.policy.targets import PolicyId

    proxy = thesis.historical_proxies[0].value if thesis.historical_proxies else "QQQ"
    if experiment_path is not None:
        spec = load_experiment_config(str(experiment_path))
        # Override experiment end with effective_thesis_end(as_of) when thesis path
        try:
            eff_end = min(spec.end, effective_thesis_end(as_of))
            if spec.end != eff_end:
                spec = spec.model_copy(update={"end": eff_end})
        except Exception:  # noqa: S110
            pass  # noqa: S110
        _ = effective_thesis_end
        return spec
    end = as_of.date() if as_of.date() > date(2007, 8, 31) else date(2025, 4, 30)
    _ = effective_thesis_end(as_of) if as_of.tzinfo is not None else None
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
        if baseline_wealth == 0:
            return None
        return candidate_wealth / baseline_wealth
    except ValueError:
        return None


def _cohort_ce_ratio_gamma_2(report) -> float | None:  # type: ignore[no-untyped-def]  # noqa: F821
    try:
        if not report.rows:
            return None
        cands = [float(r.candidate_wealth) for r in report.rows]
        bases = [float(r.baseline_wealth) for r in report.rows]
        c_ce = certainty_equivalent(cands, gamma=2.0)
        b_ce = certainty_equivalent(bases, gamma=2.0)
        if b_ce == 0:
            return None
        return float(c_ce / b_ce)
    except Exception:
        return None


def build_thesis_report(
    *,
    thesis_id: ThesisId,
    settings: DataSettings,
    as_of: datetime,
    runner: Callable[[AllocationConfig], AllocationResult],
    experiment_path: Path | None = None,
    include_regime: bool = False,
) -> ThesisReport:
    """Compose evidence, long-horizon gate, prospective, and divergence."""
    registry = load_thesis_registry(Path("configs/theses"))
    thesis = get_thesis(registry, thesis_id)

    evidence = compute_evidence_vector(
        thesis=thesis, settings=settings, as_of=as_of, runner=runner, experiment_path=experiment_path, include_regime=include_regime
    )

    evidence_spec = _resolve_evidence_spec(thesis, as_of, experiment_path)
    ce_ratio = _ce_ratio_gamma_2(evidence_spec, runner)
    terminal_wealth_ratio = ce_ratio

    # Adaptive evaluation horizon: from experiment window or proxy span
    evaluation_horizon = None
    try:
        if experiment_path is not None:
            evaluation_horizon = resolve_evaluation_horizon(thesis=thesis, catalog_start=evidence_spec.start, catalog_end=evidence_spec.end)
        else:
            try:
                ps, pe = resolve_proxy_history_span(thesis=thesis, settings=settings, as_of=as_of)
                evaluation_horizon = resolve_evaluation_horizon(thesis=thesis, catalog_start=ps, catalog_end=pe)
            except Exception:
                evaluation_horizon = None
            if evaluation_horizon is None:
                try:
                    evaluation_horizon = resolve_evaluation_horizon(
                        thesis=thesis, catalog_start=evidence_spec.start, catalog_end=evidence_spec.end
                    )
                except Exception:
                    evaluation_horizon = None
    except Exception:
        evaluation_horizon = None

    # Horizon surface for preregistered months
    horizon_surface: tuple = ()  # type: ignore[type-arg]
    surface_start: date | None = None
    surface_end: date | None = None
    try:
        if experiment_path is not None:
            surface_start, surface_end = evidence_spec.start, evidence_spec.end
        else:
            try:
                ps2, pe2 = resolve_proxy_history_span(thesis=thesis, settings=settings, as_of=as_of)
                surface_start, surface_end = ps2, pe2
            except Exception:
                surface_start, surface_end = evidence_spec.start, evidence_spec.end
        horizon_surface = resolve_horizon_surface(thesis=thesis, catalog_start=surface_start, catalog_end=surface_end)
    except Exception:
        horizon_surface = ()

    # Long horizon verdict from accumulation cohort (primary target only)
    long_horizon: LongHorizonVerdict | None = None
    cohort_ce_ratio: float | None = None
    try:
        from src.validation.accumulation_cohort import run_accumulation_cohort_report

        if evaluation_horizon is not None:
            report = run_accumulation_cohort_report(
                evidence_spec, runner, horizon_months=evaluation_horizon.horizon_months, step_months=12, bootstrap_paths=400, seed=7
            )
            long_horizon = long_horizon_passes(report)
            cohort_ce_ratio = _cohort_ce_ratio_gamma_2(report)
        else:
            long_horizon = None
    except Exception:  # noqa: BLE001
        long_horizon = None
        cohort_ce_ratio = None

    # Fallback when primary absent but history_available: longest feasible surface month
    fallback_horizon_months: int | None = None
    if evaluation_horizon is None and horizon_surface:
        try:
            span_days_fb = (surface_end - surface_start).days if surface_start is not None and surface_end is not None else 0
            span_years_fb = span_days_fb / 365.25
            history_available_fb = span_years_fb >= float(thesis.horizon.min_years)
        except Exception:  # noqa: BLE001
            history_available_fb = False
        if history_available_fb:
            feasible = [p for p in horizon_surface if p.cohort_count >= 1]
            if feasible:
                fallback_point = max(feasible, key=lambda p: p.horizon_months)
                fallback_horizon_months = int(fallback_point.horizon_months)
                # At most one fallback accumulation
                try:
                    from src.validation.accumulation_cohort import run_accumulation_cohort_report

                    fb_report = run_accumulation_cohort_report(
                        evidence_spec, runner, horizon_months=fallback_horizon_months, step_months=12, bootstrap_paths=400, seed=7
                    )
                    if long_horizon is None:
                        long_horizon = long_horizon_passes(fb_report)
                        if cohort_ce_ratio is None:
                            cohort_ce_ratio = _cohort_ce_ratio_gamma_2(fb_report)
                except Exception:  # noqa: BLE001,S110
                    pass

    # Prospective eligibility from primary proxy listing span (not full catalog)
    prospective: ProspectiveEligibility
    try:
        proxy_start, proxy_end = resolve_proxy_history_span(thesis=thesis, settings=settings, as_of=as_of)
        prospective = evaluate_prospective_eligibility(
            thesis=thesis, catalog_start=proxy_start, catalog_end=proxy_end
        )
        # When span >= min_years but no feasible cohort horizon, keep prospective False
        if evaluation_horizon is None:
            try:
                span_years_check = (proxy_end - proxy_start).days / 365.25
                if span_years_check >= float(thesis.horizon.min_years) and prospective.eligible:
                    prospective = ProspectiveEligibility(
                        eligible=False,
                        catalog_span_years=prospective.catalog_span_years,
                        min_years_required=prospective.min_years_required,
                        reason="no_feasible_cohort_horizon " + prospective.reason,
                    )
            except Exception:  # noqa: S110,BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        prospective = ProspectiveEligibility(
            eligible=False,
            catalog_span_years=0.0,
            min_years_required=int(thesis.horizon.min_years),
            reason=str(exc)[:200],
        )

    suggested_status = ThesisStatus.PROSPECTIVE_CHALLENGER if prospective.eligible else thesis.status
    next_falsifier = thesis.falsifiers[0] if thesis.falsifiers else ""

    # Divergence with meaning inputs and horizon surface
    divergence: Mapping[str, object] | None = None
    # Build divergence whenever we have any metrics or surface to emit
    should_emit = (long_horizon is not None and terminal_wealth_ratio is not None) or bool(horizon_surface) or fallback_horizon_months is not None
    if should_emit:
        # Determine median for divergence
        median: float | None = None
        if long_horizon is not None:
            median = float(long_horizon.median_ratio)
        if evidence.historical.status == "computed":
            with contextlib.suppress(Exception):
                median = float(evidence.historical.metrics.get("median_ratio", median if median is not None else 0))
        # fallback median when evidence not computed but terminal exists
        if median is None and terminal_wealth_ratio is not None:
            median = float(terminal_wealth_ratio)

        div: dict[str, object] = {}
        if long_horizon is not None:
            div["long_horizon_passes"] = bool(long_horizon.passes)
            div["median_ratio"] = float(median) if median is not None else float(long_horizon.median_ratio)
            div["long_horizon_median"] = float(long_horizon.median_ratio)
            div["cohort_count"] = int(long_horizon.cohort_count)
            div["overlap_dependence_disclosed"] = bool(long_horizon.overlap_dependence_disclosed)
        elif median is not None:
            div["median_ratio"] = float(median)
        if terminal_wealth_ratio is not None:
            div["terminal_wealth_ratio"] = float(terminal_wealth_ratio)
        if cohort_ce_ratio is not None:
            div["cohort_ce_ratio_gamma_2"] = float(cohort_ce_ratio)
            div["ce_ratio_gamma_2"] = float(cohort_ce_ratio)
        if evidence.historical.status == "computed":
            with contextlib.suppress(Exception):
                div["historical_median_ratio"] = float(evidence.historical.metrics.get("median_ratio", median if median is not None else 0))
        # horizon surface emission
        div["horizon_surface"] = tuple({"horizon_months": int(p.horizon_months), "cohort_count": int(p.cohort_count)} for p in horizon_surface)
        if evaluation_horizon is not None:
            div["evaluated_horizon_months"] = int(evaluation_horizon.horizon_months)
            div["target_years"] = int(thesis.horizon.target_years)
            div["span_capped"] = bool(evaluation_horizon.span_capped)
        else:
            div["evaluated_horizon_months"] = 0
            div["target_years"] = int(thesis.horizon.target_years)
            div["span_capped"] = False
            if fallback_horizon_months is not None:
                div["fallback_horizon_months"] = int(fallback_horizon_months)
        # meaning inputs via classify
        try:
            from src.analytics.thesis.meaning import classify_thesis_meaning

            # Derive primary count for meaning: use evaluated horizon cohort_count if present else None
            primary_count: int | None = None
            if evaluation_horizon is not None and long_horizon is not None:
                # need to check if long_horizon came from fallback; fallback case already has fallback_horizon_months
                if fallback_horizon_months is None:
                    primary_count = int(long_horizon.cohort_count) if long_horizon is not None else None
                else:
                    primary_count = None
            # fallback for BOTZ-like where evidence.historical may have no count should stay None
            # median and ce already derived
            span_years_mean = float(prospective.catalog_span_years)
            overlap_disclosed = bool(long_horizon.overlap_dependence_disclosed) if long_horizon is not None else False
            meaning = classify_thesis_meaning(
                span_years=span_years_mean,
                min_years=int(thesis.horizon.min_years),
                target_years=int(thesis.horizon.target_years),
                primary_cohort_count=primary_count,
                median_ratio=median,
                cohort_ce_ratio=cohort_ce_ratio,
                overlap_dependence_disclosed=overlap_disclosed,
            )
            div["historical_quality"] = meaning.historical_quality.value
            div["history_available"] = bool(meaning.history_available)
            div["evidence_sufficient"] = bool(meaning.evidence_sufficient)
            div["vehicle_status"] = meaning.vehicle_status.value
            div["thesis_status"] = meaning.thesis_status.value
            div["portfolio_status"] = meaning.portfolio_status.value
            div["thin_sample_warning"] = bool(meaning.thin_sample_warning)
        except Exception:  # noqa: BLE001,S110
            pass
        divergence = div
    elif long_horizon is not None and terminal_wealth_ratio is not None:
        median = (
            float(evidence.historical.metrics.get("median_ratio", long_horizon.median_ratio))
            if evidence.historical.status == "computed"
            else float(long_horizon.median_ratio)
        )
        div2: dict[str, object] = {
            "long_horizon_passes": bool(long_horizon.passes),
            "median_ratio": float(median),
            "terminal_wealth_ratio": float(terminal_wealth_ratio),
            "long_horizon_median": float(long_horizon.median_ratio),
            "cohort_count": int(long_horizon.cohort_count),
            "overlap_dependence_disclosed": bool(long_horizon.overlap_dependence_disclosed),
        }
        if cohort_ce_ratio is not None:
            div2["cohort_ce_ratio_gamma_2"] = float(cohort_ce_ratio)
            div2["ce_ratio_gamma_2"] = float(cohort_ce_ratio)
        if evidence.historical.status == "computed":
            div2["historical_median_ratio"] = float(evidence.historical.metrics.get("median_ratio", median))
        if evaluation_horizon is not None:
            div2["evaluated_horizon_months"] = int(evaluation_horizon.horizon_months)
            div2["target_years"] = int(thesis.horizon.target_years)
            div2["span_capped"] = bool(evaluation_horizon.span_capped)
        else:
            div2["evaluated_horizon_months"] = 0
            div2["target_years"] = int(thesis.horizon.target_years)
            div2["span_capped"] = False
        divergence = div2

    # Panel freshness enrichment for divergence
    try:
        from src.data.panel_freshness import resolve_catalog_panel_as_of

        panel_report = resolve_catalog_panel_as_of(settings, reference_now=datetime.now(UTC))
        eff_end = evidence_spec.end
        if divergence is not None:
            div_update = dict(divergence)
            div_update["catalog_lag_days"] = int(panel_report.lag_days)
            div_update["panel_freshness"] = str(panel_report.status.value)
            div_update["panel_as_of"] = panel_report.panel_as_of.isoformat()
            div_update["effective_end"] = eff_end.isoformat() if isinstance(eff_end, date) else str(eff_end)
            divergence = div_update
        else:
            divergence = {
                "catalog_lag_days": int(panel_report.lag_days),
                "panel_freshness": str(panel_report.status.value),
                "panel_as_of": panel_report.panel_as_of.isoformat(),
                "effective_end": eff_end.isoformat() if isinstance(eff_end, date) else str(eff_end),
            }
            if terminal_wealth_ratio is not None:
                divergence["terminal_wealth_ratio"] = float(terminal_wealth_ratio)
    except Exception:  # noqa: S110
        pass  # noqa: S110

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
            "market_regime": {"status": report.evidence.market_regime.status, "summary": report.evidence.market_regime.summary, "metrics": dict(report.evidence.market_regime.metrics)},
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
    from src.data.paths import thesis_reports_dir

    out_dir = thesis_reports_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_as_of = report.evidence.as_of.isoformat().replace(":", "-")
    out_path = out_dir / f"{report.thesis_id.value}_{safe_as_of}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
