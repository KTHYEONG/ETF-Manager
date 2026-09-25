"""Pension selection evidence writer: deterministic JSON verdict and Markdown records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from src.data.settings import DataSettings

if TYPE_CHECKING:
    from src.analytics.pension_selection import DcaTailStats, GrowthRegretTable
    from src.validation.pension_selection import PensionArmVerdict, PensionSelectionReport

__all__ = [
    "write_pension_selection_report",
]

def _growth_payload(table: GrowthRegretTable) -> dict[str, object]:
    return {
        "deltas": list(table.deltas),
        "baseline_arm_id": table.baseline_arm_id,
        "rows": [
            {
                "arm_id": row.arm_id,
                "volatility_annual": row.volatility_annual,
                "growth_by_delta": list(row.growth_by_delta),
                "max_regret": row.max_regret,
                "mean_regret": row.mean_regret,
            }
            for row in table.rows
        ],
        "breakeven_vs_baseline": dict(table.breakeven_vs_baseline),
    }


def _tail_payload(rows: Sequence[DcaTailStats]) -> list[dict[str, object]]:
    return [
        {
            "arm_id": row.arm_id,
            "delta": row.delta,
            "horizon_months": row.horizon_months,
            "n_paths": row.n_paths,
            "quantile": row.quantile,
            "low_quantile_terminal_multiple": row.low_quantile_terminal_multiple,
            "median_terminal_multiple": row.median_terminal_multiple,
            "high_quantile_pre_retirement_drawdown": row.high_quantile_pre_retirement_drawdown,
            "prob_below_principal": row.prob_below_principal,
        }
        for row in rows
    ]


def _verdict_payload(verdicts: Sequence[PensionArmVerdict]) -> list[dict[str, object]]:
    return [
        {
            "arm_id": verdict.arm_id,
            "ticker_count": verdict.ticker_count,
            "max_regret": verdict.max_regret,
            "stress_low_quantile_terminal_multiple": verdict.stress_low_quantile_terminal_multiple,
            "stress_high_quantile_pre_retirement_drawdown": verdict.stress_high_quantile_pre_retirement_drawdown,
            "historical_worst_ratio": verdict.historical_worst_ratio,
            "regret_pass": verdict.regret_pass,
            "tail_pass": verdict.tail_pass,
            "historical_pass": verdict.historical_pass,
            "reasons": list(verdict.reasons),
        }
        for verdict in verdicts
    ]


def _report_payload(report: PensionSelectionReport, provenance: Mapping[str, str]) -> dict[str, object]:
    delta = report.delta_estimate
    return {
        "name": report.name,
        "panel_start": report.panel_start.isoformat(),
        "panel_end": report.panel_end.isoformat(),
        "panel_months": report.panel_months,
        "delta_estimate": {
            "anchor_ticker": delta.anchor_ticker,
            "n_months": delta.n_months,
            "sample_alpha_annual": delta.sample_alpha_annual,
            "sample_alpha_se_annual": delta.sample_alpha_se_annual,
            "prior_mean_annual": delta.prior_mean_annual,
            "prior_sd_annual": delta.prior_sd_annual,
            "posterior_mean_annual": delta.posterior_mean_annual,
            "posterior_sd_annual": delta.posterior_sd_annual,
        },
        "growth_table": _growth_payload(report.growth_table),
        "stress_delta": report.stress_delta,
        "stress_tail": _tail_payload(report.stress_tail),
        "central_tail": _tail_payload(report.central_tail),
        "historical_worst_ratios": dict(report.historical_worst_ratios),
        "verdicts": _verdict_payload(report.verdicts),
        "status": report.status,
        "selected_arm_id": report.selected_arm_id,
        "selected_kr_targets": dict(report.selected_kr_targets),
        "fx_provenance": dict(report.fx_provenance),
        "manifest_hashes": dict(report.manifest_hashes),
        "provenance": dict(provenance),
    }


def _markdown(report: PensionSelectionReport) -> str:
    delta = report.delta_estimate
    lines = [
        f"# {report.name} — 연금 ETF 선택 판정",
        "",
        "## Decision",
        "",
        f"- status: `{report.status}`",
        f"- selected_arm_id: `{report.selected_arm_id or 'NONE'}`",
        f"- selected_kr_targets: `{dict(report.selected_kr_targets)}`",
        f"- panel: `{report.panel_start.isoformat()}` ~ `{report.panel_end.isoformat()}` ({report.panel_months}개월)",
        "",
        "## Delta Estimate",
        "",
        f"- anchor: `{delta.anchor_ticker}`",
        f"- n_months: `{delta.n_months}`",
        f"- prior_mean_annual: `{delta.prior_mean_annual:.8f}`",
        f"- prior_sd_annual: `{delta.prior_sd_annual:.8f}`",
        f"- sample_alpha_annual: `{delta.sample_alpha_annual:.8f}`",
        f"- sample_alpha_se_annual: `{delta.sample_alpha_se_annual:.8f}`",
        f"- posterior_mean_annual: `{delta.posterior_mean_annual:.8f}`",
        f"- posterior_sd_annual: `{delta.posterior_sd_annual:.8f}`",
        f"- delta_grid: `{list(report.growth_table.deltas)}`",
        "",
        "## Growth and Regret",
        "",
        "| arm_id | volatility_annual | max_regret | mean_regret | growth_by_delta |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report.growth_table.rows:
        growth = ", ".join(
            f"δ={scenario:.6f}: {value:.6f}"
            for scenario, value in zip(report.growth_table.deltas, row.growth_by_delta, strict=True)
        )
        lines.append(
            f"| {row.arm_id} | {row.volatility_annual:.6f} | {row.max_regret:.6f} | "
            f"{row.mean_regret:.6f} | {growth} |"
        )
    lines.extend(["", "### Breakeven vs Baseline", "", "| arm_id | breakeven_delta |", "|---|---:|"])
    for arm_id, value in report.growth_table.breakeven_vs_baseline.items():
        rendered = "N/A" if value is None else f"{value:.8f}"
        lines.append(f"| {arm_id} | {rendered} |")

    def _tail_section(title: str, rows: Sequence[DcaTailStats]) -> list[str]:
        section = [
            "",
            f"## {title}",
            "",
            (
                f"- horizon_months: `{rows[0].horizon_months}` · n_paths: `{rows[0].n_paths}` · "
                f"quantile: `{rows[0].quantile}`"
            ),
            "",
            "| arm_id | delta | low_terminal_multiple | median_terminal_multiple | high_drawdown | prob_below_principal |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        section.extend(
            f"| {row.arm_id} | {row.delta:.8f} | {row.low_quantile_terminal_multiple:.6f} | "
            f"{row.median_terminal_multiple:.6f} | {row.high_quantile_pre_retirement_drawdown:.6f} | "
            f"{row.prob_below_principal:.6f} |"
            for row in rows
        )
        return section

    lines.extend(_tail_section("Stress Tail", report.stress_tail))
    lines.extend(_tail_section("Central Tail", report.central_tail))
    lines.extend(
        [
            "",
            "## Historical After-tax Gate",
            "",
            "| arm_id | worst_ratio |",
            "|---|---:|",
        ]
    )
    for arm_id, value in report.historical_worst_ratios.items():
        lines.append(f"| {arm_id} | {'N/A' if value is None else f'{value:.6f}'} |")
    lines.extend(
        [
            "",
            "## Verdicts",
            "",
            "| arm_id | max_regret | stress_terminal | stress_drawdown | historical | regret_pass | tail_pass | historical_pass | reasons |",
            "|---|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for verdict in report.verdicts:
        historical = "N/A" if verdict.historical_worst_ratio is None else f"{verdict.historical_worst_ratio:.6f}"
        lines.append(
            f"| {verdict.arm_id} | {verdict.max_regret:.6f} | "
            f"{verdict.stress_low_quantile_terminal_multiple:.6f} | "
            f"{verdict.stress_high_quantile_pre_retirement_drawdown:.6f} | {historical} | "
            f"{verdict.regret_pass} | {verdict.tail_pass} | {verdict.historical_pass} | "
            f"{','.join(verdict.reasons) or 'NONE'} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- SOXX의 2021-06-21 이전 관측은 서로 다른 지수 레짐을 반영하며 break flag를 유지한다.",
            "- CAPM 시나리오는 beta에 보상이 있다는 가정과 표본 불확실성을 노출한 분석이다.",
            "- 본 결과는 시뮬레이션이며 투자 권유가 아니다.",
            "",
        ]
    )
    return "\n".join(lines)


def write_pension_selection_report(
    report: PensionSelectionReport,
    settings: DataSettings,
    *,
    experiment_id: str,
    provenance: Mapping[str, str],
) -> Path:
    """Write the existing selection evidence and verdict from completed calculations.

    Args:
        report: Fixed gate outcomes and candidate evidence.
        settings: Result output root.
        experiment_id: Existing run identity.
        provenance: Exact input identities used for the calculation.

    Returns:
        JSON report path.

    Raises:
        OSError: If result publication fails.
    """
    from src.data.result_store import ResultKind, write_result

    reference = write_result(
        settings,
        experiment=report.name,
        kind=ResultKind.SELECTION,
        run_id=experiment_id,
        payload=_report_payload(report, provenance),
        markdown=_markdown(report),
    )
    return reference.json_path
