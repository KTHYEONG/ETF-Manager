"""Historical campaign evidence writer: deterministic JSON decision records."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.data.settings import DataSettings

if TYPE_CHECKING:
    from src.validation.historical_campaign import FinalHistoricalCampaignReport

__all__ = [
    "write_final_historical_campaign_report",
]


def write_final_historical_campaign_report(
    report: FinalHistoricalCampaignReport,
    settings: DataSettings,
    *,
    experiment_id: str,
) -> Path:
    """Persist the existing historical campaign evidence without changing its decision schema.

    Args:
        report: Completed campaign and audits.
        settings: Result output root.
        experiment_id: Existing run identity.

    Returns:
        JSON report path.

    Raises:
        OSError: If publication fails.
    """
    from src.data.result_store import ResultKind, write_result

    payload = {
        "campaign_id": report.campaign_id,
        "window_start": report.window_start.isoformat(),
        "window_end": report.window_end.isoformat(),
        "manifest_hashes": dict(report.manifest_hashes),
        "window": {
            "start": report.window_start.isoformat(),
            "end": report.window_end.isoformat(),
        },
        "arm_rows": [
            {
                "arm_id": row.arm_id,
                "targets": dict(row.targets),
                "cohort_count": int(row.cohort_count),
                "median_ratio": float(row.median_ratio),
                "p10_ratio": float(row.p10_ratio),
                "worst_ratio": float(row.worst_ratio),
                "win_rate": float(row.win_rate),
                "ce_gamma_10": float(row.ce_gamma_10),
                "bootstrap_win_rate": float(row.bootstrap_win_rate),
                "bootstrap_p05": float(row.bootstrap_p05),
                "xirr_real": float(row.xirr_real),
                "cost_stress_worst_ratio": float(row.cost_stress_worst_ratio),
                "fx_stress_worst_ratio": float(row.fx_stress_worst_ratio),
                "cohort_starts": [d.isoformat() for d in row.cohort_starts],
                "cohort_ends": [d.isoformat() for d in row.cohort_ends],
                "paired_cost_stress": [
                    {"scenario_id": p.scenario_id, "candidate_over_baseline_ratio": float(p.candidate_over_baseline_ratio)}
                    for p in getattr(row, "paired_cost_stress", ())
                ],
            }
            for row in report.arm_rows
        ],
        "regime_coverage": {
            "rows": [
                {
                    "regime_name": r.regime_name,
                    "covered": bool(r.covered),
                    "overlap_months": int(r.overlap_months),
                    "coverage_tier": str(r.coverage_tier),
                    "coverage_fraction": float(r.coverage_fraction),
                }
                for r in report.regime_coverage.rows
            ],
            "independent_sample_warning": bool(report.regime_coverage.independent_sample_warning),
        },
        "lineage_census": {
            "total_experiments": int(report.lineage_census.total_experiments),
            "families": [
                {
                    "family_id": f.family_id,
                    "experiment_count": int(f.experiment_count),
                    "active_count": int(f.active_count),
                    "archived_count": int(f.archived_count),
                }
                for f in report.lineage_census.families
            ],
        },
        "lineage_hash_census": (
            {
                "unique_config_hashes": int(report.lineage_hash_census.unique_config_hashes),
                "total_run_records": int(report.lineage_hash_census.total_run_records),
            }
            if report.lineage_hash_census is not None
            else None
        ),
        "pre_history_mix_proxy": [
            {
                "evidence_tier": row.evidence_tier,
                "status": row.status,
                "regime_name": row.regime_name,
                "baseline_terminal_real_krw": row.baseline_terminal_real_krw,
                "candidate_terminal_real_krw": row.candidate_terminal_real_krw,
                "candidate_over_baseline_ratio": row.candidate_over_baseline_ratio,
                "window_start": row.window_start.isoformat() if row.window_start is not None else None,
                "window_end": row.window_end.isoformat() if row.window_end is not None else None,
                "reason": row.reason,
            }
            for row in report.pre_history_mix_proxy
        ],
        "operational_unlock": bool(report.operational_unlock),
        "tax_sensitivity": {
            "status": report.tax_sensitivity.status,
            "rationale": report.tax_sensitivity.rationale,
        },
        "pre_history_proxy": {
            "status": report.pre_history_proxy.status,
            "reason": report.pre_history_proxy.reason,
            "proxy_window_start": (
                report.pre_history_proxy.proxy_window_start.isoformat()
                if report.pre_history_proxy.proxy_window_start is not None
                else None
            ),
            "proxy_window_end": (
                report.pre_history_proxy.proxy_window_end.isoformat()
                if report.pre_history_proxy.proxy_window_end is not None
                else None
            ),
            "terminal_wealth_real_krw": report.pre_history_proxy.terminal_wealth_real_krw,
            "xirr_real": report.pre_history_proxy.xirr_real,
        },
    }
    ref = write_result(
        settings,
        experiment=report.campaign_id,
        kind=ResultKind.FINAL_HISTORICAL,
        run_id=experiment_id,
        payload=payload,
    )
    return ref.json_path
