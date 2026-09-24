# ruff: noqa: T201
"""Maintain results command runners (stdout output is the command contract)."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from src.data.result_maintenance import (
    apply_result_migration,
    apply_result_prune,
    list_results,
    plan_result_migration,
    plan_result_prune,
    promote_result,
)
from src.data.result_store import ResultKind
from src.data.settings import DataSettings

logger = logging.getLogger(__name__)


def _run_list(args: argparse.Namespace, settings: DataSettings) -> int:
    _ = args
    for summary in list_results(settings):
        rel = os.path.relpath(summary.json_path, Path.cwd())
        print(
            "\t".join(
                (
                    summary.experiment,
                    summary.kind.value,
                    summary.run_id,
                    summary.last_written_at.isoformat(),
                    str(summary.run_count),
                    rel,
                )
            )
        )
    return 0


def _run_promote(args: argparse.Namespace, settings: DataSettings) -> int:
    paths = promote_result(
        settings,
        experiment=str(args.experiment),
        kind=ResultKind(str(args.kind)),
        run_id=str(args.run_id),
        dest_root=Path(str(args.dest_root)),
    )
    for path in paths:
        print(path.as_posix())
    return 0


def _run_prune(args: argparse.Namespace, settings: DataSettings) -> int:
    plan = plan_result_prune(settings, keep=int(args.keep))
    dry_run = not bool(args.apply)
    deleted = apply_result_prune(settings, plan, dry_run=dry_run)
    for path in deleted:
        print(path.as_posix())
    print(f"keep={plan.keep} dry_run={dry_run} count={len(deleted)}")
    return 0


def _run_migrate(args: argparse.Namespace, settings: DataSettings) -> int:
    plan = plan_result_migration(settings)
    count = apply_result_migration(settings, plan, dry_run=not bool(args.apply))
    for src, dest in plan.moves:
        print(f"{src.as_posix()} -> {dest.as_posix()}")
    print(f"dry_run={not bool(args.apply)} count={count}")
    return 0


def run_results_command(args: argparse.Namespace, settings: DataSettings) -> int:
    """Dispatch ``maintain results <action>``.

    Returns 0 on success, 1 on handled failure (``ValueError``, ``OSError`` incl.
    ``FileExistsError``/``FileNotFoundError``) logged as ``[DATA] event=results_<action>_failed``.
    """
    action = str(getattr(args, "results_action", None))
    try:
        if action == "list":
            return _run_list(args, settings)
        if action == "promote":
            return _run_promote(args, settings)
        if action == "prune":
            return _run_prune(args, settings)
        if action == "migrate":
            return _run_migrate(args, settings)
        raise ValueError(f"unknown results action {action!r}")
    except (ValueError, OSError) as exc:
        logger.error("[DATA] event=results_%s_failed reason=%s", action, exc)
        return 1
