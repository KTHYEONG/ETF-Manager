"""After-tax campaign evidence writer: deterministic JSON and Markdown records."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.data.settings import DataSettings

if TYPE_CHECKING:
    from src.validation.after_tax_campaign import AfterTaxCampaignReport

__all__ = [
    "write_after_tax_campaign_report",
]


def write_after_tax_campaign_report(
    report: AfterTaxCampaignReport, settings: DataSettings, experiment_id: str
) -> Path:
    """Publish the existing cohort and gate evidence from a completed after-tax campaign.

    Args:
        report: Completed cohort calculation.
        settings: Result output root.
        experiment_id: Existing run identity.

    Returns:
        JSON report path.

    Raises:
        OSError: If publication fails.
    """
    from src.data.result_store import ResultKind, write_result

    payload = {
        "name": report.name,
        "experiment_id": experiment_id,
        "operational_unlock": bool(report.operational_unlock),
        "summaries": [
            {
                "arm_id": summary.arm_id,
                "role": str(summary.role),
                "horizon_months": summary.horizon_months,
                "cohort_count": summary.cohort_count,
                "median_ratio": summary.median_ratio,
                "worst_ratio": summary.worst_ratio,
                "best_ratio": summary.best_ratio,
                "win_rate": summary.win_rate,
                "frictionless_median_ratio": summary.frictionless_median_ratio,
                "bootstrap_p05_ratio": summary.bootstrap_p05_ratio,
                "crash_first_median_ratio": summary.crash_first_median_ratio,
                "other_median_ratio": summary.other_median_ratio,
                "median_max_drawdown_after_tax": summary.median_max_drawdown_after_tax,
                "median_taxes_paid_krw": summary.median_taxes_paid_krw,
                "median_sell_count": summary.median_sell_count,
                "gate_passes": summary.gate_passes,
            }
            for summary in report.summaries
        ],
        "rows": [
            {
                "arm_id": row.arm_id,
                "horizon_months": row.horizon_months,
                "cohort_start": row.cohort_start.isoformat(),
                "cohort_end": row.cohort_end.isoformat(),
                "after_tax_real_krw": row.after_tax_real_krw,
                "ratio": row.ratio,
                "frictionless_ratio": row.frictionless_ratio,
                "max_drawdown_after_tax": row.max_drawdown_after_tax,
                "taxes_paid_krw": row.taxes_paid_krw,
                "sell_count": row.sell_count,
                "crash_first": row.crash_first,
                "financial_income_breach_years": row.financial_income_breach_years,
            }
            for row in report.rows
        ],
    }
    lines = [
        f"# After-tax campaign {report.name}",
        "",
        f"experiment_id: {experiment_id}",
        f"operational_unlock: {report.operational_unlock}",
        "",
        "| arm | horizon | cohorts | median | worst | p05 | gate |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {summary.arm_id} | {summary.horizon_months} | {summary.cohort_count} "
        f"| {summary.median_ratio:.4f} | {summary.worst_ratio:.4f} "
        f"| {summary.bootstrap_p05_ratio:.4f} | {summary.gate_passes} |"
        for summary in report.summaries
    )
    ref = write_result(
        settings,
        experiment=report.name,
        kind=ResultKind.AFTER_TAX,
        run_id=experiment_id,
        payload=payload,
        markdown="\n".join(lines) + "\n",
    )
    return ref.json_path
