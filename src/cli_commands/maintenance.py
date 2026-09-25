"""Maintain-data command: doctor inspection, repair, and safe pruning."""

from __future__ import annotations

import argparse
import logging

from src.cli_commands.parser import _UsageError
from src.data.doctor import (
    BronzeFetcher,
    DataHealthReport,
    DataMaintenancePlan,
    SilverCondition,
    apply_data_maintenance,
    plan_data_maintenance,
)
from src.data.retention import PrunePlan, apply_prune, plan_prune
from src.data.settings import DataSettings
from src.data.storage import UntrustedDatasetError

logger = logging.getLogger(__name__)

BRONZE_FETCHER: BronzeFetcher | None = None


def run_maintain_prune_command(args: argparse.Namespace, settings: DataSettings) -> int:
    """Prune stale partitions and mirrors; dry-run by default, mutate only with apply."""
    plan = plan_prune(
        settings,
        keep_latest_only=bool(getattr(args, "keep_latest_only", True)),
        drop_nport_zip_mirrors=bool(getattr(args, "drop_nport_zip_mirrors", True)),
        migrate_results_layout=bool(getattr(args, "migrate_results_layout", True)),
    )
    dry = not bool(getattr(args, "apply", False))
    report = apply_prune(plan, dry_run=dry)
    logger.info(
        "[DATA] event=prune target=%s dry_run=%s to_delete=%d to_migrate=%d deleted=%d migrated=%d",
        "prune",
        dry,
        len(plan.to_delete),
        len(plan.to_migrate),
        len(report.deleted),
        len(report.migrated),
    )
    return 0


def run_maintain_recover_command(args: argparse.Namespace, settings: DataSettings) -> int:
    """Recover keys lost by a truncated latest partition; dry-run by default."""
    from src.data.merge import apply_history_recovery, plan_history_recovery
    from src.data.schema import Dataset

    recover_name = str(getattr(args, "dataset", ""))
    try:
        recover_dataset = Dataset(recover_name)
    except ValueError:
        raise _UsageError(f"unknown dataset {recover_name!r}") from None
    recovery_plan = plan_history_recovery(settings, recover_dataset)
    if recovery_plan is None:
        logger.info("[DATA] event=recover_none dataset=%s", str(recover_dataset))
        return 0
    if not bool(getattr(args, "apply", False)):
        logger.info(
            "[DATA] event=recover_plan dataset=%s latest_rows=%d recovered_rows=%d sources=%s",
            str(recover_dataset),
            recovery_plan.latest_rows,
            recovery_plan.recovered_rows,
            ",".join(recovery_plan.source_manifest_sha256s),
        )
        return 0
    artifact = apply_history_recovery(recovery_plan, settings)
    logger.info(
        "[DATA] event=recover_applied dataset=%s manifest=%s rows=%d",
        str(recover_dataset),
        artifact.manifest_path.stem,
        artifact.manifest.row_count,
    )
    return 0


def run_maintain_data_command(args: argparse.Namespace, settings: DataSettings) -> int:
    """Show one concise health summary and optionally apply the verified repair plan.

    Args:
        args: Parsed maintenance options.
        settings: Data root to inspect.

    Returns:
        Zero only when the requested inspection or repair completed safely.

    Raises:
        UntrustedDatasetError: If a repair source or protected partition cannot be trusted.
    """
    plan = plan_data_maintenance(settings)
    dry_run = not bool(getattr(args, "apply", False))
    try:
        retention = plan_prune(settings)
    except UntrustedDatasetError:
        retention = None
    _log_summary(plan, dry_run, retention)
    if dry_run:
        return 0
    report = apply_data_maintenance(plan, settings, fetcher=BRONZE_FETCHER)
    _log_report(report, len(plan.repairs))
    return 0


def _log_summary(plan: DataMaintenancePlan, dry_run: bool, retention: PrunePlan | None) -> None:
    """Emit the one-screen health, repair, and retention counts for a dry plan."""
    findings = plan.findings
    healthy = sum(1 for item in findings.datasets if item.condition is SilverCondition.HEALTHY)
    missing = sum(1 for item in findings.datasets if item.condition is SilverCondition.MISSING_BRONZE)
    damaged = sum(1 for item in findings.datasets if item.condition is SilverCondition.DAMAGED_SILVER)
    if retention is None:
        logger.info(
            "[DATA] event=maintain_data dry_run=%s healthy=%d missing_bronze=%d damaged_silver=%d repairs=%d retention=unknown",
            dry_run,
            healthy,
            missing,
            damaged,
            len(plan.repairs),
        )
        return
    logger.info(
        "[DATA] event=maintain_data dry_run=%s healthy=%d missing_bronze=%d damaged_silver=%d repairs=%d protected=%d prunable=%d",
        dry_run,
        healthy,
        missing,
        damaged,
        len(plan.repairs),
        len(retention.retained_manifests) + len(retention.retained_parquets),
        len(retention.to_delete),
    )


def _log_report(report: DataHealthReport, repair_count: int) -> None:
    """Emit the post-repair health state after apply completes."""
    healthy = sum(1 for item in report.datasets if item.condition is SilverCondition.HEALTHY)
    missing = sum(1 for item in report.datasets if item.condition is SilverCondition.MISSING_BRONZE)
    damaged = sum(1 for item in report.datasets if item.condition is SilverCondition.DAMAGED_SILVER)
    logger.info(
        "[DATA] event=maintain_data_applied healthy=%d missing_bronze=%d damaged_silver=%d repairs=%d",
        healthy,
        missing,
        damaged,
        repair_count,
    )
