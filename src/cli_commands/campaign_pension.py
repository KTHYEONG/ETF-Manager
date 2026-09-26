# ruff: noqa: S110,SIM102,SIM108,F541,I001,UP035
"""Pension decision and ISA-household CLI runners (split from campaign.py)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.cli_commands.parser import _resolve_git_commit
from src.data.schema import Dataset
from src.data.settings import DataSettings
from src.data.storage import UntrustedDatasetError
from src.analytics.metrics import XirrError
from src.policy.targets import PolicyError
from src.policy.thesis import ThesisError
from src.sim.allocation import AllocationDataError
from src.sim.baseline import BaselineDataError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.validation.pension_decision import PensionDecisionReport

_ERRORS = (AllocationDataError, BaselineDataError, PolicyError, ThesisError, UntrustedDatasetError, XirrError, ValueError, OSError)


def _read_incumbent_id(record_path: str) -> str:
    """Read the held candidate id from a frozen decision record file."""
    import json

    try:
        document = json.loads(Path(record_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"pension incumbent record is unreadable: {record_path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"pension incumbent record lacks an incumbent_id string: {record_path}")
    raw_id = document.get("incumbent_id")
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise ValueError(f"pension incumbent record lacks an incumbent_id string: {record_path}")
    return raw_id.strip()


def _read_record_id(record_path: str) -> str | None:
    """Read the frozen record id from a decision record file, if present."""
    import json

    try:
        document = json.loads(Path(record_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"pension incumbent record is unreadable: {record_path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"pension incumbent record lacks a record_id string: {record_path}")
    raw_id = document.get("record_id")
    if raw_id is None:
        return None
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise ValueError(f"pension incumbent record lacks a record_id string: {record_path}")
    return raw_id.strip()


def _decision_markdown(report: PensionDecisionReport, trial_count: int, dominance_reference_id: str | None) -> str:
    """Render a Korean human summary of the pension decision verdict."""
    lines = [
        f"# 연금 결정 {report.name}",
        "",
        f"- 상태: {report.status}",
        f"- 선택: {report.selected_id or '없음'}",
        f"- 동등 후보: {', '.join(report.equivalent_ids) if report.equivalent_ids else '없음'}",
        f"- 사유: {', '.join(report.reasons) if report.reasons else '없음'}",
        f"- 시행 횟수: {trial_count}",
        "",
        "## 후보 강건 점수",
        "",
        "| 후보 | 강건 점수 | 부트스트랩 승률 |",
        "| --- | --- | --- |",
    ]
    for candidate_id, score in sorted(report.robust_scores.items()):
        share = report.bootstrap_win_share.get(candidate_id)
        share_text = f"{share:.3f}" if share is not None else "없음"
        lines.append(f"| {candidate_id} | {score:.6f} | {share_text} |")
    lines += ["", f"세후 순위 일치: {'예' if report.tax_rank_agreement else '아니오'}"]
    lines += [
        "",
        "## 도미넌스 가드",
        "",
        f"- 기준 후보: {dominance_reference_id or '없음'}",
        f"- 제외 후보: {', '.join(report.guard_excluded_ids) if report.guard_excluded_ids else '없음'}",
        "",
    ]
    guard_tiers = [tier for tier in ("modern", "century", "realized") if tier in report.dominance_min_ratio_by_tier]
    lines += [
        "| 후보 | 기준 대비 최소 비율" + "".join(f" | {tier}" for tier in guard_tiers) + " |",
        "| --- | --- |" + "".join(" --- |" for _ in guard_tiers),
    ]
    for candidate_id in sorted(report.dominance_min_ratio):
        row = f"| {candidate_id} | {report.dominance_min_ratio[candidate_id]:.6f} |"
        for tier in (tier for tier in ("modern", "century", "realized") if tier in report.dominance_min_ratio_by_tier):
            row += f" {report.dominance_min_ratio_by_tier[tier][candidate_id]:.6f} |"
        lines.append(row)
    lines += [
        "",
        "## 컨트롤",
        "",
        "| 컨트롤 | 현대 점수 | 기준 대비 |",
        "| --- | --- | --- |",
    ]
    for control_id in sorted(report.control_scores):
        versus = report.control_vs_reference.get(control_id)
        versus_text = f"{versus:.6f}" if versus is not None else "없음"
        lines.append(f"| {control_id} | {report.control_scores[control_id]:.6f} | {versus_text} |")
    return "\n".join(lines) + "\n"


def run_pension_decision_command(*, config_path: str, settings: DataSettings, seed: int, incumbent_record: str | None = None, freeze: bool = False) -> int:
    """Run the pre-registered robust pension holding decision and persist its evidence.

    Returns: Zero after a reproducible report is written, nonzero on invalid data or policy.
    Raises: No domain exception escapes the CLI boundary; failure is logged and returned.
    """
    from dataclasses import replace

    import polars as pl

    from src.data.catalog import load_snapshot_visible, resolve_snapshot
    from src.data.result_store import ResultKind, write_result
    from src.sim.pension_engine import PensionDataError
    from src.sim.pension_monthly import panel_from_research
    from src.sim.pension_splice import realized_panel_from_prices, realized_window_start, splice_modern_panel
    from src.validation.pension_campaign import load_pension_campaign_spec, run_pension_campaign
    from src.validation.pension_decision import assert_tax_rank_neutrality, evaluate_pension_decision
    from src.validation.pension_decision_config import load_pension_decision_spec

    try:
        import hashlib
        from datetime import UTC, datetime

        spec = load_pension_decision_spec(config_path)
        config_bytes = Path(config_path).read_bytes()
        incumbent_id = _read_incumbent_id(incumbent_record) if incumbent_record else None
        snapshot = resolve_snapshot(settings, (Dataset.PRICES, Dataset.RESEARCH_MONTHLY))
        modern_as_of = datetime(spec.modern_end.year, spec.modern_end.month, spec.modern_end.day, 23, 59, tzinfo=UTC)
        # 두 근거층 모두 같은 결정 시점(modern_end)에 공개된 행만 읽는다; 이후 공개분은 미래 정보다.
        modern_frame = load_snapshot_visible(snapshot, Dataset.PRICES, modern_as_of)
        research_frame_at_as_of = load_snapshot_visible(snapshot, Dataset.RESEARCH_MONTHLY, modern_as_of)
        century_frame = research_frame_at_as_of.filter(
            pl.col("period_end").is_between(spec.century_start, spec.century_end)
        )
        sleeves = sorted(
            {sleeve for schedule in spec.candidates.values() for sleeve in schedule.start_weights}
            | {sleeve for schedule in spec.controls.values() for sleeve in schedule.start_weights}
        )
        candidate_sleeves = {
            sleeve for schedule in spec.candidates.values() for sleeve in schedule.start_weights
        }
        modern, splice_records = splice_modern_panel(
            modern_frame,
            research_frame_at_as_of,
            sleeves,
            spec.modern_splices,
            modern_as_of,
            spec.modern_start,
            spec.modern_end,
        )
        if spec.realized_horizons_years:
            realized_start = realized_window_start(
                spec.modern_splices, candidate_sleeves, spec.modern_start
            )
            realized = realized_panel_from_prices(
                modern_frame,
                sorted(candidate_sleeves),
                modern_as_of,
                realized_start,
                spec.modern_end,
            )
            realized_window: dict[str, object] | None = {
                "start": realized_start.isoformat(),
                "end": spec.modern_end.isoformat(),
                "months": len(realized.months),
            }
        else:
            realized = None
            realized_window = None
        century = panel_from_research(century_frame, spec.century_series, modern_as_of)
        if century.months[0] != spec.century_start or century.months[-1] != spec.century_end:
            raise ValueError(
                f"pension century panel spans {century.months[0].isoformat()}..{century.months[-1].isoformat()}, "
                f"config requires {spec.century_start.isoformat()}..{spec.century_end.isoformat()} "
                f"visible at {modern_as_of.isoformat()}"
            )
        report = evaluate_pension_decision(spec, modern, century, seed=seed, incumbent_id=incumbent_id, realized=realized)
        consumed = {
            str(dataset): Path(snapshot.artifacts[dataset].manifest_path).stem
            for dataset in (Dataset.PRICES, Dataset.RESEARCH_MONTHLY)
        }
        report = replace(report, manifest_hashes=consumed)
        campaign_spec = load_pension_campaign_spec(spec.tax_crosscheck_campaign_path)
        campaign_report = run_pension_campaign(campaign_spec, settings, seed=seed)
        summaries = [
            {
                "arm_id": summary.arm_id,
                "horizon_months": summary.horizon_months,
                "median_wealth_ratio": summary.median_wealth_ratio,
            }
            for summary in campaign_report.summaries
        ]
        agreement = assert_tax_rank_neutrality(report, summaries, spec.tax_crosscheck_arm_map)
        if agreement:
            report = replace(report, tax_rank_agreement=True)
        else:
            logger.warning(
                "[PORTFOLIO] event=pension_decision_tax_disagreement status=%s selected=%s",
                report.status,
                report.selected_id or "NONE",
            )
            report = replace(
                report,
                status="NO_DECISION",
                selected_id=None,
                reasons=(*report.reasons, "TAX_RANK_DISAGREEMENT"),
                tax_rank_agreement=False,
            )
        campaign_bytes = Path(spec.tax_crosscheck_campaign_path).read_bytes()
        git_commit = _resolve_git_commit()
        digest = hashlib.sha256(
            config_bytes + git_commit.encode() + "".join(consumed.values()).encode()
            + campaign_bytes + str(seed).encode()
        ).hexdigest()[:16]
        provenance: dict[str, str] = {
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "tax_crosscheck_config_sha256": hashlib.sha256(campaign_bytes).hexdigest(),
            "git_commit": git_commit,
            "seed": str(seed),
            "freeze": str(freeze),
        }
        if incumbent_record:
            provenance["incumbent_record"] = incumbent_record
        payload: dict[str, object] = {
            "name": report.name,
            "status": report.status,
            "selected_id": report.selected_id,
            "equivalent_ids": list(report.equivalent_ids),
            "reasons": list(report.reasons),
            "trial_count": report.trial_count,
            "robust_scores": dict(report.robust_scores),
            "sensitivity_robust_scores": {
                str(gamma): dict(scores) for gamma, scores in report.sensitivity_robust_scores.items()
            },
            "bootstrap_win_share": dict(report.bootstrap_win_share),
            "tax_rank_agreement": report.tax_rank_agreement,
            "manifest_hashes": dict(report.manifest_hashes),
            "dominance_reference_id": spec.dominance_reference_id,
            "dominance_min_ratio": dict(report.dominance_min_ratio),
            "dominance_min_ratio_by_tier": {
                tier: dict(per_tier) for tier, per_tier in report.dominance_min_ratio_by_tier.items()
            },
            "realized_window": realized_window,
            "guard_excluded_ids": list(report.guard_excluded_ids),
            "control_scores": dict(report.control_scores),
            "control_vs_reference": dict(report.control_vs_reference),
            "modern_splices": [
                {
                    "sleeve": record.sleeve,
                    "proxy_first_month": record.proxy_first_month.isoformat(),
                    "proxy_last_month": record.proxy_last_month.isoformat(),
                    "etf_first_month": record.etf_first_month.isoformat(),
                    "proxy_weights": dict(record.proxy_weights),
                }
                for record in splice_records
            ],
            "sleeve_products": {
                sleeve: {
                    "krx_code": product.krx_code,
                    "name": product.name,
                    "listing_date": product.listing_date.isoformat(),
                    "total_expense_ratio": product.total_expense_ratio,
                    "currency_hedged": product.currency_hedged,
                    "source_url": product.source_url,
                    "source_checked_date": product.source_checked_date.isoformat(),
                }
                for sleeve, product in spec.sleeve_products.items()
            },
            "scores": [
                {
                    "candidate_id": score.candidate_id,
                    "tier": score.tier,
                    "horizon_years": score.horizon_years,
                    "gamma": score.gamma,
                    "ce_ratio": score.ce_ratio,
                    "median_ratio": score.median_ratio,
                    "worst_ratio": score.worst_ratio,
                    "cohort_count": score.cohort_count,
                    "median_pre_retirement_drawdown": score.median_pre_retirement_drawdown,
                }
                for score in report.scores
            ],
        }
        ref = write_result(
            settings,
            experiment=spec.name,
            kind=ResultKind.PENSION_DECISION,
            run_id=digest,
            payload=payload,
            markdown=_decision_markdown(report, report.trial_count, spec.dominance_reference_id),
        )
        if freeze and report.status != "NO_DECISION":
            from src.data.paths import resolve_repository_paths
            from src.validation.pension_decision_record import freeze_pension_decision

            previous_id: str | None = None
            if incumbent_record:
                previous_id = _read_record_id(incumbent_record)
            record_path = freeze_pension_decision(
                report,
                spec,
                output_dir=resolve_repository_paths(settings).frozen / "pension",
                frozen_at=datetime.now(UTC),
                git_commit=git_commit,
                config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                previous_record_id=previous_id,
            )
            logger.info(
                "[PORTFOLIO] event=pension_decision_frozen record=%s report=%s",
                record_path.as_posix(),
                ref.json_path.as_posix(),
            )
    except (*_ERRORS, PensionDataError, UntrustedDatasetError) as exc:
        logger.error(
            "[PORTFOLIO] event=pension_decision_cli_failed reason_type=%s reason=%s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return 1
    logger.info(
        "[PORTFOLIO] event=pension_decision_cli_done experiment=%s experiment_id=%s status=%s selected=%s freeze=%s report=%s",
        spec.name,
        digest,
        report.status,
        report.selected_id or "NONE",
        freeze,
        ref.json_path.as_posix(),
    )
    return 0


def run_isa_household_command(*, config_path: str, settings: DataSettings, seed: int, freeze: bool = False) -> int:
    """Run the pre-registered ISA + pension household operating-policy decision and persist its evidence.

    Returns: Zero after a reproducible report is written, nonzero on invalid data or policy.
    Raises: No domain exception escapes the CLI boundary; failure is logged and returned.
    """
    import polars as pl

    from src.data.catalog import load_snapshot_visible, resolve_snapshot
    from src.data.result_store import ResultKind, write_result
    from src.sim.isa_tax import load_isa_tax_regime
    from src.sim.pension_engine import PensionDataError
    from src.sim.pension_monthly import panel_from_research
    from src.sim.pension_splice import splice_modern_panel
    from src.sim.pension_tax import load_pension_tax_regime
    from src.sim.tax import load_tax_regime
    from src.validation.isa_household_config import load_isa_household_spec
    from src.validation.isa_household_decision import (
        evaluate_isa_household,
        freeze_isa_household_decision,
        isa_household_markdown,
    )
    from src.validation.pension_decision_config import load_pension_decision_spec
    from src.validation.pension_decision_record import load_pension_decision_record

    try:
        import hashlib
        from datetime import UTC, datetime

        spec = load_isa_household_spec(config_path)
        config_bytes = Path(config_path).read_bytes()
        decision_spec = load_pension_decision_spec(spec.pension_decision_config_path)
        record = load_pension_decision_record(spec.pension_record_path)
        isa_regime = load_isa_tax_regime(spec.isa_tax_regime_path)
        pension_regime = load_pension_tax_regime(spec.pension_tax_regime_path)
        overseas_regime = load_tax_regime(spec.overseas_tax_regime_path)
        snapshot = resolve_snapshot(settings, (Dataset.PRICES, Dataset.RESEARCH_MONTHLY))
        modern_as_of = datetime(
            decision_spec.modern_end.year, decision_spec.modern_end.month, decision_spec.modern_end.day, 23, 59, tzinfo=UTC
        )
        modern_frame = load_snapshot_visible(snapshot, Dataset.PRICES, modern_as_of)
        research_frame_at_as_of = load_snapshot_visible(snapshot, Dataset.RESEARCH_MONTHLY, modern_as_of)
        century_frame = research_frame_at_as_of.filter(
            pl.col("period_end").is_between(decision_spec.century_start, decision_spec.century_end)
        )
        sleeves = sorted(
            {sleeve for schedule in decision_spec.candidates.values() for sleeve in schedule.start_weights}
            | {sleeve for schedule in decision_spec.controls.values() for sleeve in schedule.start_weights}
        )
        modern, _splice_records = splice_modern_panel(
            modern_frame,
            research_frame_at_as_of,
            sleeves,
            decision_spec.modern_splices,
            modern_as_of,
            decision_spec.modern_start,
            decision_spec.modern_end,
        )
        century = panel_from_research(century_frame, decision_spec.century_series, modern_as_of)
        if century.months[0] != decision_spec.century_start or century.months[-1] != decision_spec.century_end:
            raise ValueError(
                f"pension century panel spans {century.months[0].isoformat()}..{century.months[-1].isoformat()}, "
                f"config requires {decision_spec.century_start.isoformat()}..{decision_spec.century_end.isoformat()} "
                f"visible at {modern_as_of.isoformat()}"
            )
        report = evaluate_isa_household(
            spec,
            decision_spec,
            record,
            modern,
            century,
            isa_regime=isa_regime,
            pension_regime=pension_regime,
            overseas_regime=overseas_regime,
            seed=seed,
        )
        consumed = {
            str(dataset): Path(snapshot.artifacts[dataset].manifest_path).stem
            for dataset in (Dataset.PRICES, Dataset.RESEARCH_MONTHLY)
        }
        pension_record_bytes = Path(spec.pension_record_path).read_bytes()
        pension_record_id = _read_record_id(spec.pension_record_path) or Path(spec.pension_record_path).stem
        git_commit = _resolve_git_commit()
        digest = hashlib.sha256(
            config_bytes + git_commit.encode() + "".join(consumed.values()).encode()
            + pension_record_bytes + str(seed).encode()
        ).hexdigest()[:16]
        config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        payload: dict[str, object] = {
            "name": report.name,
            "incumbent_id": report.incumbent_id,
            "pension_record_id": pension_record_id,
            "trial_count": report.trial_count,
            "lineage": {
                "related_trial_count": spec.lineage.related_trial_count,
                "related_trials": list(spec.lineage.related_trials),
                "first_test_date": spec.lineage.first_test_date.isoformat(),
                "post_hoc_disclosure": spec.lineage.post_hoc_disclosure,
            },
            "manifest_hashes": dict(consumed),
            "config_sha256": config_sha256,
            "git_commit": git_commit,
            "seed": seed,
            "decisions": [
                {
                    "budget_krw": decision.budget_krw,
                    "status": decision.status,
                    "selected_arm_id": decision.selected_arm_id,
                    "equivalent_arm_ids": list(decision.equivalent_arm_ids),
                    "robust_scores": dict(decision.robust_scores),
                    "per_profile_best": dict(decision.per_profile_best),
                    "bootstrap_win_share": decision.bootstrap_win_share,
                    "sensitivity_selected": {str(year): arm for year, arm in decision.sensitivity_selected.items()},
                    "holding_scores": dict(decision.holding_scores),
                    "holding_consistent": decision.holding_consistent,
                    "reasons": list(decision.reasons),
                }
                for decision in report.decisions
            ],
            "cells": [
                {
                    "budget_krw": cell.budget_krw,
                    "arm_id": cell.arm_id,
                    "profile_id": cell.profile_id,
                    "tier": cell.tier,
                    "horizon_years": cell.horizon_years,
                    "ce_ratio": cell.ce_ratio,
                    "median_ratio": cell.median_ratio,
                    "worst_ratio": cell.worst_ratio,
                    "scenario_count": cell.scenario_count,
                    "median_household_net_krw": cell.median_household_net_krw,
                    "median_pension_locked_share": cell.median_pension_locked_share,
                }
                for cell in report.cells
            ],
            "notes": spec.notes,
        }
        ref = write_result(
            settings,
            experiment=spec.name,
            kind=ResultKind.ISA_HOUSEHOLD,
            run_id=digest,
            payload=payload,
            markdown=isa_household_markdown(report),
        )
        if freeze:
            if all(decision.status in ("ADOPT_ARM", "KEEP_BASELINE") for decision in report.decisions):
                from src.data.paths import resolve_repository_paths

                record_path = freeze_isa_household_decision(
                    report,
                    output_dir=resolve_repository_paths(settings).frozen / "isa",
                    frozen_at=datetime.now(UTC),
                    git_commit=git_commit,
                    config_sha256=config_sha256,
                    pension_record_id=pension_record_id,
                    run_id=digest,
                )
                logger.info("[PORTFOLIO] event=isa_household_frozen record=%s", record_path.as_posix())
            else:
                logger.warning(
                    "[PORTFOLIO] event=isa_household_freeze_skipped statuses=%s",
                    ",".join(decision.status for decision in report.decisions),
                )
    except (*_ERRORS, PensionDataError, UntrustedDatasetError) as exc:
        logger.error(
            "[PORTFOLIO] event=isa_household_cli_failed reason_type=%s reason=%s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return 1
    logger.info(
        "[PORTFOLIO] event=isa_household_cli_done experiment=%s experiment_id=%s budgets=%d report=%s",
        spec.name,
        digest,
        len(report.decisions),
        ref.json_path.as_posix(),
    )
    return 0


def run_pension_campaign_command(*, config_path: str, settings: DataSettings, seed: int) -> int:
    """Run a reporting-only pension campaign from trusted, source-labeled inputs.

    Returns: Zero after a reproducible report is written, nonzero on invalid data or policy.
    Raises: No domain exception escapes the CLI boundary; failure is logged and returned.
    """
    from src.sim.pension_engine import PensionDataError, PensionMarketMode
    from src.validation.pension_campaign import (
        load_pension_campaign_spec, run_pension_campaign, write_pension_campaign_report,
    )

    try:
        import hashlib

        spec = load_pension_campaign_spec(config_path)
        regime_bytes = Path(spec.tax_regime_path).read_bytes()
        config_bytes = Path(config_path).read_bytes()
        general_regime_bytes = b""
        general_regime_sha: str | None = None
        if spec.household is not None:
            general_regime_bytes = Path(spec.household.general_tax_regime_path).read_bytes()
            general_regime_sha = hashlib.sha256(general_regime_bytes).hexdigest()
        report = run_pension_campaign(spec, settings, seed=seed)
        consumed = report.manifest_hashes
        if spec.market_mode is PensionMarketMode.KR_LIVE:
            manifest_hashes = [str(consumed[str(Dataset.KR_ETF_PRICES)]), str(consumed.get(str(Dataset.CPI)) or "NO_TRUSTED_CPI")]
        else:
            manifest_hashes = [str(consumed[str(Dataset.PRICES)]), str(consumed[str(Dataset.FX_KRW_BASE)]), str(consumed.get(str(Dataset.FX)) or "NO_FX_FALLBACK"), str(consumed.get(str(Dataset.CPI)) or "NO_TRUSTED_CPI")]
        git_commit = _resolve_git_commit()
        digest = hashlib.sha256(
            config_bytes + git_commit.encode() + "".join(manifest_hashes).encode()
            + regime_bytes + general_regime_bytes + str(seed).encode()
        ).hexdigest()[:16]
        provenance: dict[str, str] = {
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "tax_regime_sha256": hashlib.sha256(regime_bytes).hexdigest(),
            "git_commit": git_commit,
            "seed": str(seed),
        }
        if general_regime_sha is not None:
            provenance["general_tax_regime_sha256"] = general_regime_sha
        report_path = write_pension_campaign_report(report, settings, experiment_id=digest, provenance=provenance)
    except (*_ERRORS, PensionDataError, UntrustedDatasetError) as exc:
        logger.error("[DATA] event=pension_campaign_cli_failed reason_type=%s reason=%s", type(exc).__name__, exc)
        return 1
    logger.info("[DATA] event=pension_campaign_cli_done experiment=%s experiment_id=%s arms=%d report=%s", spec.name, digest, len(report.summaries), report_path)
    return 0


def run_pension_selection_command(*, config_path: str, settings: DataSettings, seed: int) -> int:
    """Run the pre-registered standalone pension ETF selection and persist its evidence."""
    from src.analytics.pension_selection import PensionSelectionDataError
    from src.sim.pension_engine import PensionDataError
    from src.validation.pension_selection import (
        load_pension_selection_spec, run_pension_selection, write_pension_selection_report,
    )

    try:
        import hashlib

        spec = load_pension_selection_spec(config_path)
        config_bytes = Path(config_path).read_bytes()
        tax_bytes = Path(spec.tax_regime_path).read_bytes()
        identity_bytes = Path(spec.etf_identity_path).read_bytes()
        campaign_bytes = [Path(path).read_bytes() for path in spec.historical.campaign_config_paths]
        report = run_pension_selection(spec, settings, seed=seed)
        # 식별자는 실행이 고정한 스냅샷에서만 가져온다; 라벨은 게시된 적 없는(부재) 선택 입력에만 쓰인다.
        consumed = report.manifest_hashes
        manifest_hashes = [
            str(consumed[str(Dataset.PRICES)]),
            str(consumed[str(Dataset.FX_KRW_BASE)]),
            consumed.get(str(Dataset.FX)) or "NO_FX_FALLBACK",
            consumed.get(str(Dataset.CPI)) or "NO_TRUSTED_CPI",
        ]
        campaign_hashes = [hashlib.sha256(value).hexdigest() for value in campaign_bytes]
        git_commit = _resolve_git_commit()
        digest = hashlib.sha256(
            config_bytes + git_commit.encode() + "".join(manifest_hashes).encode()
            + tax_bytes + identity_bytes + b"".join(campaign_bytes) + str(seed).encode()
        ).hexdigest()[:16]
        provenance: dict[str, str] = {
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "tax_regime_sha256": hashlib.sha256(tax_bytes).hexdigest(),
            "etf_identity_sha256": hashlib.sha256(identity_bytes).hexdigest(),
            "campaign_config_sha256": ",".join(campaign_hashes),
            "git_commit": git_commit,
            "seed": str(seed),
        }
        report_path = write_pension_selection_report(report, settings, experiment_id=digest, provenance=provenance)
    except (*_ERRORS, PensionDataError, PensionSelectionDataError, UntrustedDatasetError) as exc:
        logger.error(
            "[PORTFOLIO] event=pension_selection_cli_failed reason_type=%s reason=%s",
            type(exc).__name__, exc, exc_info=True,
        )
        return 1
    logger.info(
        "[PORTFOLIO] event=pension_selection_cli_done experiment=%s experiment_id=%s status=%s selected=%s report=%s",
        spec.name, digest, report.status, report.selected_arm_id or "NONE", report_path,
    )
    return 0
