"""Pension campaign evidence writer: deterministic JSON and Markdown decision records."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from src.data.catalog import resolve_snapshot
from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.data.storage import UntrustedDatasetError
from src.sim.pension_engine import PensionMarketMode

if TYPE_CHECKING:
    from src.data.catalog import CatalogSnapshot
    from src.validation.pension_campaign import PensionCampaignReport
    from src.validation.pension_household import PensionHouseholdReport

__all__ = [
    "write_pension_campaign_report",
]

_FX_FALLBACK_NOTE: Final[str] = (
    "USD/KRW on Korean-holiday sessions comes from FRED DEXKOUS (NY-noon buying rate) "
    "instead of the ECOS base rate; see fx_provenance for count and same-day basis."
)
_SOXX_BREAK_NOTE: Final[str] = "SOXX observations before 2021-06-21 carry the disclosed index-break label."
_SHORT_LIVE_NOTE: Final[str] = "Current live Korean ETF history is too short for an observed 20-year test."
_UNDEFINED_RATIO_NOTE: Final[str] = (
    "Cohorts whose baseline after-tax wealth is zero (no contribution was made, for example a profile "
    "with no usable tax credit) have no defined paired ratio and are excluded from ratio, rate, "
    "and drawdown statistics; see the undefined column."
)
_EXIT_CONVENTION_NOTE: Final[str] = (
    "Ratios value each account as if closed at the cohort end with the non-pension tax "
    "(credit clawed back); pension-receipt values are a bracket that assumes deferral to "
    "eligibility and no further return; see years_to_draw_at_threshold."
)
_HOUSEHOLD_NOTE: Final[str] = (
    "Household view compares the same available cash: pension (credit-optimal contributions, "
    "valued at exit) plus a general account holding leftover cash and credit refunds, versus "
    "all cash in a general account holding the same US-listed ETFs under Korean overseas-equity "
    "tax with year-end gain harvesting."
)

def _snapshot_stem(snapshot: CatalogSnapshot, dataset: Dataset) -> str:
    """Manifest identity pinned by a snapshot, never a post-hoc latest lookup."""
    return Path(snapshot.artifacts[dataset].manifest_path).stem

def _manifest_hashes(settings: DataSettings, mode: PensionMarketMode) -> dict[str, str | None]:
    datasets = (
        (Dataset.KR_ETF_PRICES, Dataset.CPI)
        if mode is PensionMarketMode.KR_LIVE else (Dataset.PRICES, Dataset.FX_KRW_BASE, Dataset.FX, Dataset.CPI)
    )
    hashes: dict[str, str | None] = {}
    for dataset in datasets:
        try:
            snapshot = resolve_snapshot(settings, (dataset,))
        except UntrustedDatasetError:
            hashes[str(dataset)] = None
        else:
            hashes[str(dataset)] = _snapshot_stem(snapshot, dataset)
    return hashes

def _household_view_payload(report: PensionHouseholdReport | None) -> dict[str, object] | None:
    """JSON-safe household view with ISO dates and plan totals (not per-date schedules)."""
    if report is None:
        return None
    return {
        "config": {
            "general_tax_regime_path": report.spec.general_tax_regime_path,
            "commission_bps": report.spec.commission_bps,
            "fx_spread_bps": report.spec.fx_spread_bps,
            "harvest_gains": report.spec.harvest_gains,
            "fractional_shares": report.spec.fractional_shares,
        },
        "rows": [
            {
                "arm_id": row.arm_id,
                "profile_id": row.profile_id,
                "horizon_months": row.horizon_months,
                "cohort_start": row.cohort_start.isoformat(),
                "cohort_end": row.cohort_end.isoformat(),
                "pension_contributions_krw": row.plan.pension_contributions_krw,
                "leftover_krw": row.plan.leftover_krw,
                "settled_refunds_krw": row.plan.settled_refunds_krw,
                "pending_refund_krw": row.plan.pending_refund_krw,
                "general_only_krw": row.general_only_krw,
                "side_account_krw": row.side_account_krw,
                "pension_lump_sum_net_krw": row.pension_lump_sum_net_krw,
                "pension_annuity_low_net_krw": row.pension_annuity_low_net_krw,
                "pension_annuity_high_net_krw": row.pension_annuity_high_net_krw,
                "household_liquidation_krw": row.household_liquidation_krw,
                "household_annuity_low_krw": row.household_annuity_low_krw,
                "household_annuity_high_krw": row.household_annuity_high_krw,
                "account_advantage_liquidation": row.account_advantage_liquidation,
                "account_advantage_annuity_low": row.account_advantage_annuity_low,
                "account_advantage_annuity_high": row.account_advantage_annuity_high,
            }
            for row in report.rows
        ],
        "summaries": [
            {
                "arm_id": summary.arm_id,
                "horizon_months": summary.horizon_months,
                "profile_id": summary.profile_id,
                "cohort_count": summary.cohort_count,
                "excluded_payout_cohorts": summary.excluded_payout_cohorts,
                "median_account_advantage_liquidation": summary.median_account_advantage_liquidation,
                "worst_account_advantage_liquidation": summary.worst_account_advantage_liquidation,
                "median_account_advantage_annuity_low": summary.median_account_advantage_annuity_low,
                "median_account_advantage_annuity_high": summary.median_account_advantage_annuity_high,
                "median_asset_effect_household": summary.median_asset_effect_household,
                "median_asset_effect_general_only": summary.median_asset_effect_general_only,
            }
            for summary in report.summaries
        ],
    }

def write_pension_campaign_report(
    report: PensionCampaignReport,
    settings: DataSettings,
    *,
    experiment_id: str,
    provenance: Mapping[str, str] | None = None,
) -> Path:
    """Write the existing pension evidence schema from a completed campaign result.

    Args:
        report: Finished campaign calculation.
        settings: Result output root.
        experiment_id: Existing run identity.
        provenance: Exact pinned input manifest identities.

    Returns:
        The JSON report path.

    Raises:
        OSError: If an output cannot be written safely.
    """
    from src.data.result_store import ResultKind, write_result

    fx_provenance: dict[str, object] = dict(report.fx_provenance)
    evidence_notes: list[str] = [_SOXX_BREAK_NOTE, _SHORT_LIVE_NOTE, _EXIT_CONVENTION_NOTE]
    fallback_session_count = int(cast("int", fx_provenance.get("fallback_session_count", 0) or 0))
    if fallback_session_count > 0:
        evidence_notes.append(_FX_FALLBACK_NOTE)
    if any(summary.undefined_ratio_cohorts > 0 for summary in report.summaries):
        evidence_notes.append(_UNDEFINED_RATIO_NOTE)
    if report.household is not None:
        evidence_notes.append(_HOUSEHOLD_NOTE)
    payload = {
        "name": report.name,
        "experiment_id": experiment_id,
        "provenance": dict(provenance or {}),
        "market_mode": str(report.market_mode),
        "market_coverage_start": report.market_coverage_start.isoformat(),
        "market_coverage_end": report.market_coverage_end.isoformat(),
        "real_data_status": report.real_data_status,
        "fx_provenance": fx_provenance,
        "evidence_status": report.evidence_status,
        "evidence_notes": evidence_notes,
        "exit_convention": {
            "primary": "lump_sum",
            "brackets": ["annuity_low", "annuity_high"],
            "foreign_tax_credit_rate": report.foreign_tax_credit_rate,
            "foreign_dividend_withholding_rate": report.foreign_dividend_withholding_rate,
        },
        "manifest_hashes": dict(report.manifest_hashes),
        "household_view": _household_view_payload(report.household),
        "summaries": [
            {
                "arm_id": summary.arm_id,
                "horizon_months": summary.horizon_months,
                "cohort_count": summary.cohort_count,
                "undefined_ratio_cohorts": summary.undefined_ratio_cohorts,
                "fully_undefined_profiles": list(summary.fully_undefined_profiles),
                "independent_window_count": summary.independent_window_count,
                "underperforming_cohorts": summary.underperforming_cohorts,
                "median_wealth_ratio": summary.median_wealth_ratio,
                "worst_wealth_ratio": summary.worst_wealth_ratio,
                "median_annuity_low_ratio": summary.median_annuity_low_ratio,
                "median_annuity_high_ratio": summary.median_annuity_high_ratio,
                "median_cashflow_normalized_rate": summary.median_cashflow_normalized_rate,
                "median_max_drawdown": summary.median_max_drawdown,
                "evidence_status": summary.evidence_status,
            }
            for summary in report.summaries
        ],
        "rows": [
            {
                "arm_id": row.arm_id,
                "profile_id": row.profile_id,
                "cohort_start": row.cohort_start.isoformat(),
                "cohort_end": row.cohort_end.isoformat(),
                "horizon_months": row.horizon_months,
                "market_mode": str(row.market_mode),
                "terminal_nav_krw": row.terminal_nav_krw,
                "after_tax_wealth_krw": row.after_tax_wealth_krw,
                "terminal_nav_real_krw": row.terminal_nav_real_krw,
                "after_tax_wealth_real_krw": row.after_tax_wealth_real_krw,
                "liquidation_wealth_krw": row.liquidation_wealth_krw,
                "liquidation_wealth_real_krw": row.liquidation_wealth_real_krw,
                "annuity_low_wealth_krw": row.annuity_low_wealth_krw,
                "annuity_high_wealth_krw": row.annuity_high_wealth_krw,
                "years_to_draw_at_threshold": row.years_to_draw_at_threshold,
                "foreign_tax_withheld_krw": row.foreign_tax_withheld_krw,
                "after_tax_payout_krw": row.after_tax_payout_krw,
                "contributed_krw": row.contributed_krw,
                "gross_credit_krw": row.gross_credit_krw,
                "usable_credit_krw": row.usable_credit_krw,
                "credit_received_krw": row.credit_received_krw,
                "withdrawal_tax_krw": row.withdrawal_tax_krw,
                "payout_shortfall_krw": row.payout_shortfall_krw,
                "is_retirement_terminal": row.is_retirement_terminal,
                "cashflow_normalized_rate": row.cashflow_normalized_rate,
                "max_drawdown": row.max_drawdown,
                "paired_wealth_ratio": row.paired_wealth_ratio,
                "historical_overlap_group": row.historical_overlap_group,
            }
            for row in report.cohort_rows
        ],
    }
    lines = [
        f"# Pension campaign {report.name}",
        "",
        f"experiment_id: {experiment_id}",
        f"market_mode: {report.market_mode}",
        f"market_coverage: {report.market_coverage_start}..{report.market_coverage_end}",
        f"real_data_status: {report.real_data_status}",
        f"evidence_status: {report.evidence_status}",
        f"note: {_SHORT_LIVE_NOTE}",
        f"note: {_SOXX_BREAK_NOTE}",
        f"note: {_EXIT_CONVENTION_NOTE}",
    ]
    if fx_provenance:
        share_value = float(cast("float", fx_provenance.get("fallback_session_share") or 0.0))
        lines.append(
            f"fx_fallback: status={fx_provenance.get('fallback_status')} "
            f"sessions={fx_provenance.get('fallback_session_count')} "
            f"share={share_value:.4f}"
        )
        if fallback_session_count > 0:
            lines.append(f"note: {_FX_FALLBACK_NOTE}")

    def _fmt(value: float | None) -> str:
        return f"{value:.4f}" if value is not None else "N/A"

    if report.household is None:
        lines.append("household_view: not provided")
    else:
        lines.append(f"note: {_HOUSEHOLD_NOTE}")
        lines.extend(
            [
                "",
                "## Household same-cash view",
                "",
                "| arm | horizon | profile | cohorts | excluded | acct adv (lump) | worst | "
                "acct adv (annuity low) | acct adv (annuity high) | asset effect household | asset effect general |",
                "|---|---|---|---|---|---|---|---|---|---|---|",
            ]
        )
        lines.extend(
            f"| {summary.arm_id} | {summary.horizon_months} | {summary.profile_id} | {summary.cohort_count} "
            f"| {summary.excluded_payout_cohorts} | {_fmt(summary.median_account_advantage_liquidation)} "
            f"| {_fmt(summary.worst_account_advantage_liquidation)} "
            f"| {_fmt(summary.median_account_advantage_annuity_low)} "
            f"| {_fmt(summary.median_account_advantage_annuity_high)} "
            f"| {_fmt(summary.median_asset_effect_household)} | {_fmt(summary.median_asset_effect_general_only)} |"
            for summary in report.household.summaries
        )
    if any(summary.undefined_ratio_cohorts > 0 for summary in report.summaries):
        lines.append(f"note: {_UNDEFINED_RATIO_NOTE}")
    fully_undefined_ids = sorted(
        {profile_id for summary in report.summaries for profile_id in summary.fully_undefined_profiles}
    )
    if fully_undefined_ids:
        lines.append(f"fully_undefined_profiles: {', '.join(fully_undefined_ids)}")
    lines.extend(
        [
            "",
            "| arm | horizon | cohorts | undefined | independent | median | worst | annuity low | annuity high | median XIRR | drawdown |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )

    lines.extend(
        f"| {summary.arm_id} | {summary.horizon_months} | {summary.cohort_count} "
        f"| {summary.undefined_ratio_cohorts} | {summary.independent_window_count} | {_fmt(summary.median_wealth_ratio)} "
        f"| {_fmt(summary.worst_wealth_ratio)} | {_fmt(summary.median_annuity_low_ratio)} "
        f"| {_fmt(summary.median_annuity_high_ratio)} | "
        f"{summary.median_cashflow_normalized_rate if summary.median_cashflow_normalized_rate is not None else 'N/A'} "
        f"| {_fmt(summary.median_max_drawdown)} |"
        for summary in report.summaries
    )
    try:
        ref = write_result(
            settings,
            experiment=report.name,
            kind=ResultKind.PENSION,
            run_id=experiment_id,
            payload=payload,
            markdown="\n".join(lines) + "\n",
        )
    except OSError as exc:
        raise OSError(f"pension campaign report unwritable: {exc}") from exc
    return ref.json_path
