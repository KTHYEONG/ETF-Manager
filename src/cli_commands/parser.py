# ruff: noqa: S110,SIM102,SIM108,F541,I001,UP035
"""CLI parser construction."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import date
from typing import NoReturn

from src.policy.targets import OPERATIONAL_POLICY_ID, POLICY_ALIASES, PolicyId
from src.policy.tilt import TILT_FACTORS
from src.sim.baseline import BASELINE_ALIASES


class _UsageError(Exception):
    """Parse or requirement failure surfaced as exit code 2."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


def _iso_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date {text!r}") from exc


def _resolve_git_commit() -> str:
    """Current HEAD commit hash; an experiment record without lineage is useless."""
    git_path = shutil.which("git")
    if git_path is None:
        raise ValueError("git executable unavailable")
    completed = subprocess.run(  # noqa: S603
        [git_path, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or not commit:
        raise ValueError("git commit hash unavailable")
    return commit


def _build_parser() -> _Parser:
    parser = _Parser(prog="etf-manager", description="ETF research ingest CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest", help="Fetch and persist one vendor dataset")
    ingest.add_argument("dataset", choices=("prices", "fx", "macro", "cpi", "factors", "research-returns", "smoke", "history", "static-dca", "nport", "thesis-panel", "thesis-fundamentals"))
    ingest.add_argument("--tickers", nargs="+", default=None, help="Price tickers (prices/smoke only)")
    ingest.add_argument("--provider", choices=("fred", "ecos"), default=None, help="FX vendor (fx/smoke only)")
    ingest.add_argument("--series-id", default=None, help="FRED series identifier (macro only)")
    ingest.add_argument("--filing-quarter", default=None, help="N-PORT filing quarter like 2019q4 (nport only)")
    ingest.add_argument("--start", type=_iso_date, default=None, help="ISO start date (required except smoke)")
    ingest.add_argument("--end", type=_iso_date, default=None, help="ISO end date (required except smoke)")
    ingest.add_argument(
        "--production-data",
        action="store_true",
        help="Write smoke ingest to data/ instead of scratch/smoke_data (not recommended)",
    )
    run_parser = subparsers.add_parser("run", help="Run a stored-data simulation")
    run_targets = run_parser.add_subparsers(dest="target", required=True)
    baseline = run_targets.add_parser("baseline", help="Run a B0/B1 DCA baseline on catalog partitions")
    baseline.add_argument(
        "--id",
        choices=tuple(BASELINE_ALIASES),
        required=True,
        help="Baseline id (canonical dca_global/dca_us or legacy b0_global/b1_us)",
    )
    baseline.add_argument("--ticker", required=True)
    baseline.add_argument("--start", required=True, type=_iso_date)
    baseline.add_argument("--end", required=True, type=_iso_date)
    baseline.add_argument("--contribution-krw", required=True, type=float)
    policy = run_targets.add_parser("policy", help="Run an S-policy strategic allocation on catalog partitions")
    policy.add_argument(
        "--id",
        choices=tuple(POLICY_ALIASES),
        required=True,
        help=f"Policy id (operational default: {OPERATIONAL_POLICY_ID.value} with locked adaptive contribution)",
    )
    policy.add_argument("--start", required=True, type=_iso_date)
    policy.add_argument("--end", required=True, type=_iso_date)
    policy.add_argument("--contribution-krw", required=True, type=float)
    policy.add_argument(
        "--tilt-factor",
        choices=TILT_FACTORS,
        default=None,
        help="Factor to tilt (requires --tilt-intensity)",
    )
    policy.add_argument(
        "--tilt-intensity",
        type=float,
        default=None,
        help="Tilt strength in (0, 0.25] (requires --tilt-factor)",
    )
    policy.add_argument(
        "--rebalance-band",
        type=float,
        default=None,
        help="Buy-only rebalance band in [0, 1); omit for Phase 3 mix",
    )
    policy.add_argument(
        "--overlay-max-tilt",
        "--overlay-max-shift",
        dest="overlay_max_shift",
        type=float,
        default=None,
        help="Bounded overlay max tilt in (0, 0.10]; omit to disable overlay",
    )
    policy.add_argument(
        "--vix-threshold",
        type=float,
        default=None,
        help="Optional VIX de-risk threshold (requires --overlay-max-shift)",
    )
    policy.add_argument(
        "--reserve-withhold-cap",
        "--reserve-max-withhold",
        dest="reserve_max_withhold",
        type=float,
        default=None,
        help="Reserve ledger withhold cap in (0, 0.10]; omit to disable the reserve",
    )
    policy.add_argument(
        "--fx-max-defer",
        type=float,
        default=None,
        help="Max KRW defer fraction in (0, 1]; omit to disable FX defer",
    )
    policy.add_argument(
        "--fx-expensive-percentile",
        type=float,
        default=None,
        help="Optional expensive-USD percentile in (0, 1) (requires --fx-max-defer)",
    )
    policy.add_argument(
        "--map-etf",
        action="store_true",
        help="Map economic sleeves to implementation ETFs with incumbent hysteresis",
    )
    policy.add_argument(
        "--map-min-improvement",
        type=float,
        default=None,
        help="Optional hysteresis min improvement in (0, 1] (requires --map-etf)",
    )
    validate = run_targets.add_parser("validate", help="Cohort CE gate versus B0 on catalog partitions")
    validate.add_argument("--id", choices=tuple(str(member) for member in PolicyId), required=True)
    validate.add_argument("--start", required=True, type=_iso_date)
    validate.add_argument("--end", required=True, type=_iso_date)
    validate.add_argument("--contribution-krw", required=True, type=float)
    validate.add_argument(
        "--hurdle",
        "--delta0",
        dest="delta0",
        type=float,
        default=0.02,
        help="Per-module complexity margin (hurdle)",
    )
    validate.add_argument(
        "--extra-rules",
        "--modules",
        dest="modules",
        type=int,
        default=0,
        help="Count of added signal/sleeve modules (extra rules)",
    )
    validate.add_argument("--horizon-months", type=int, default=36, help="Cohort horizon in calendar months")
    validate.add_argument(
        "--cohort-step-months",
        type=int,
        default=12,
        help="Months between cohort start dates",
    )
    validate.add_argument(
        "--bootstrap-paths",
        type=int,
        default=0,
        help="Moving-block bootstrap paths on cohort wealths; 0 disables",
    )
    validate.add_argument("--seed", type=int, default=None, help="Required when --bootstrap-paths > 0")
    paper = run_targets.add_parser("paper", help="Replay a stored-data policy as paper buy orders")
    paper.add_argument("--id", choices=tuple(str(member) for member in PolicyId), required=True)
    paper.add_argument("--start", required=True, type=_iso_date)
    paper.add_argument("--end", required=True, type=_iso_date)
    paper.add_argument("--contribution-krw", required=True, type=float)
    ablation = run_targets.add_parser(
        "ablation",
        help="Identical-cashflow adoption ablation from an experiment JSON",
    )
    ablation.add_argument(
        "--config",
        required=True,
        help="Path to the experiment JSON (baseline plus candidates)",
    )
    walk_forward = run_targets.add_parser(
        "walk-forward",
        help="Walk-forward adoption campaign from an experiment JSON",
    )
    walk_forward.add_argument(
        "--config",
        required=True,
        help="Path to the experiment JSON (single candidate with train/test months)",
    )
    walk_forward_costs = run_targets.add_parser(
        "walk-forward-costs",
        help="Walk-forward adoption grid over fixed cost scenarios from an experiment JSON",
    )
    walk_forward_costs.add_argument(
        "--config",
        required=True,
        help="Path to the experiment JSON (single candidate with train/test months)",
    )
    walk_forward_proxy = run_targets.add_parser(
        "walk-forward-proxy",
        help="Wave C research-proxy versus ETF-baseline adoption campaign from an experiment JSON",
    )
    walk_forward_proxy.add_argument(
        "--config",
        required=True,
        help="Path to the experiment JSON (single us_ff research-proxy candidate with train/test months)",
    )
    cadence_robustness = run_targets.add_parser(
        "cadence-robustness",
        help="Growth-first cadence robustness gate (cost grid, worst cohort, bootstrap tail)",
    )
    cadence_robustness.add_argument(
        "--config",
        required=True,
        help="Path to the experiment JSON (single growth_first cadence candidate with train/test months)",
    )
    cadence_robustness.add_argument("--seed", type=int, required=True, help="Bootstrap RNG seed")
    cadence_robustness.add_argument(
        "--bootstrap-paths",
        type=int,
        default=1000,
        help="Moving-block bootstrap paths on cohort wealth ratios (must be >= 1)",
    )
    diagnose_us = run_targets.add_parser(
        "diagnose-us-vehicles",
        help="Popular US vehicle diagnostics on identical cashflows; reporting only, never an adoption gate",
    )
    diagnose_us.add_argument("--start", required=True, type=_iso_date)
    diagnose_us.add_argument("--end", required=True, type=_iso_date)
    diagnose_us.add_argument("--contribution-krw", required=True, type=float)
    diagnose_qqq = run_targets.add_parser(
        "diagnose-qqq-regimes",
        help="QQQ versus VTI regime-window ratios on identical cashflows; reporting only, never an adoption gate",
    )
    diagnose_qqq.add_argument("--contribution-krw", required=True, type=float)
    diagnose_qqq_blends = run_targets.add_parser(
        "diagnose-qqq-blends",
        help="QQQ drawdown-blend recipe ratios versus QQQ/VTI anchors on identical cashflows; reporting only, never an adoption gate",
    )
    diagnose_qqq_blends.add_argument("--contribution-krw", required=True, type=float)
    diagnose_qqq_reserve = run_targets.add_parser(
        "diagnose-qqq-reserve",
        help="QQQ reserve-versus-plain ratios and reserve usage per regime window; reporting only, never an adoption gate",
    )
    diagnose_qqq_reserve.add_argument("--contribution-krw", required=True, type=float)
    diagnose_qqq_reserve.add_argument(
        "--reserve-schedule", choices=("v1", "v2", "v3", "v4"), default="v1"
    )
    diagnose_qqq_cadence = run_targets.add_parser(
        "diagnose-qqq-cadence",
        help="QQQ month-open-cadence ratios versus the default monthly cadence per regime window; reporting only, never an adoption gate",
    )
    diagnose_qqq_cadence.add_argument("--contribution-krw", required=True, type=float)
    diagnose_qqq_accumulation = run_targets.add_parser(
        "diagnose-qqq-accumulation-alpha",
        help="QQQ buy-cadence accumulation-screen ratios versus month-end; reporting only, never an adoption gate",
    )
    diagnose_qqq_accumulation.add_argument("--contribution-krw", required=True, type=float)
    diagnose_qqq_kafi = run_targets.add_parser(
        "diagnose-qqq-kafi",
        help="KAFI path plus shaped-versus-flat DCA real TW ratios per regime window; reporting only, never an adoption gate",
    )
    diagnose_qqq_kafi.add_argument("--contribution-krw", required=True, type=float)
    diagnose_qqq_adaptive_hp = run_targets.add_parser(
        "diagnose-qqq-adaptive-hp",
        help="Adaptive contribution HP neighbourhood screen versus operational v5; reporting only, never an adoption gate",
    )
    diagnose_qqq_adaptive_hp.add_argument("--contribution-krw", required=True, type=float)
    diagnose_compound_dca = run_targets.add_parser(
        "diagnose-compound-dca",
        help="Compound DCA tournament QQQ vs QQQ90/SOXX10 flat vs adaptive; reporting only, never an adoption gate",
    )
    diagnose_compound_dca.add_argument("--contribution-krw", required=True, type=float)
    accumulation_cohort = run_targets.add_parser(
        "accumulation-cohort",
        help="Rolling 120M accumulation cohort report (reporting-only, never an adoption gate)",
    )
    accumulation_cohort.add_argument(
        "--config",
        required=True,
        help="Path to the experiment JSON (baseline plus candidate)",
    )
    accumulation_cohort.add_argument(
        "--horizon-months",
        type=int,
        default=120,
        help="Cohort horizon in calendar months (default 120)",
    )
    accumulation_cohort.add_argument(
        "--cohort-step-months",
        type=int,
        default=12,
        help="Months between cohort starts; must be one of 1, 12, 36",
    )
    accumulation_cohort.add_argument(
        "--bootstrap-paths",
        type=int,
        default=4000,
        help="Moving-block bootstrap paths on cohort wealth ratios (must be >= 1)",
    )
    accumulation_cohort.add_argument("--seed", type=int, default=None, help="Bootstrap RNG seed")
    audit_feasibility = run_targets.add_parser(
        "audit-feasibility",
        help="Static DCA feasibility window audit (reporting only)",
    )
    audit_feasibility.add_argument("--config", required=True, help="Path to the experiment JSON")
    audit_feasibility.add_argument("--write-report", action="store_true", help="Persist audit JSON under audits/")
    thesis = run_targets.add_parser(
        "thesis",
        help="Inspect thesis registry (reporting only, never an adoption gate)",
    )
    thesis.add_argument("--id", dest="thesis_id", default=None, help="Thesis id to inspect (omit to list)")
    thesis.add_argument("--config-dir", default="configs/theses", help="Thesis registry directory")
    thesis.add_argument("--compute-evidence", action="store_true", help="Compute evidence vector for thesis via compute_evidence_vector")
    thesis_report = run_targets.add_parser(
        "thesis-report",
        help="Build thesis report (evidence + long-horizon + prospective)",
    )
    thesis_report.add_argument("--id", dest="thesis_id", required=True, help="Thesis id for report")
    thesis_report.add_argument("--as-of", dest="as_of", default=None, help="ISO datetime for report as-of (default now)")
    thesis_report.add_argument("--experiment", dest="experiment_path", default=None, help="Optional experiment JSON path")
    diagnose_overlap = run_targets.add_parser(
        "diagnose-overlap",
        help="Holdings overlap between two vehicles at PIT as-of (reporting only, never an adoption gate)",
    )
    diagnose_overlap.add_argument("--vehicle", required=True, help="Primary vehicle ticker (e.g. SOXX)")
    diagnose_overlap.add_argument("--baseline", required=True, help="Baseline vehicle ticker (e.g. QQQ)")
    diagnose_overlap.add_argument("--as-of", dest="as_of", default=None, help="ISO datetime for PIT as-of (default now)")
    thesis_wave = run_targets.add_parser(
        "thesis-wave",
        help="Run thesis wave (all theses) and write combined wave JSON and markdown",
    )
    thesis_wave.add_argument("--as-of", dest="as_of", default=None, help="ISO datetime for wave as-of (default now)")
    thesis_wave.add_argument("--allow-stale", action="store_true", help="Allow stale panel without hard-stop")
    thesis_incremental = run_targets.add_parser(
        "thesis-incremental",
        help="Run Track H incremental portfolio (QQQ95/90/85 vs QQQ100) with attribution and path bootstrap",
    )
    thesis_incremental.add_argument("--thesis-id", dest="thesis_id", default="ai_compute", help="Thesis id (default ai_compute)")
    thesis_incremental.add_argument("--as-of", dest="as_of", default=None, help="ISO datetime for as-of (default panel_as_of)")
    thesis_incremental.add_argument("--allow-stale", action="store_true", help="Allow stale panel without hard-stop")
    thesis_incremental.add_argument("--seed", type=int, default=7, help="Bootstrap RNG seed")
    thesis_incremental.add_argument("--bootstrap-paths", type=int, default=400, help="Bootstrap paths for path bootstrap")
    thesis_incremental.add_argument("--contribution-krw", type=float, default=1_000_000, help="Monthly contribution KRW")
    thesis_pipeline = run_targets.add_parser(
        "thesis-pipeline",
        help="Run thesis pipeline (wave + incremental + Wave D exit) and write Wave D exit markdown",
    )
    thesis_pipeline.add_argument("--thesis-id", dest="thesis_id", default="ai_compute", help="Thesis id (default ai_compute)")
    thesis_pipeline.add_argument("--as-of", dest="as_of", default=None, help="ISO datetime for as-of (default panel_as_of)")
    thesis_pipeline.add_argument("--allow-stale", action="store_true", help="Allow stale panel without hard-stop")
    thesis_pipeline.add_argument("--seed", type=int, default=7, help="Bootstrap RNG seed")
    thesis_pipeline.add_argument("--bootstrap-paths", type=int, default=400, help="Bootstrap paths")
    # wiring: run_thesis_pipeline_command invocation
    from src.analytics.wave_d_exit import run_thesis_pipeline_command as _run_thesis_pipeline_command

    _ = _run_thesis_pipeline_command
    maintain = subparsers.add_parser("maintain", help="Maintenance utilities")
    maintain_targets = maintain.add_subparsers(dest="target", required=True)
    prune = maintain_targets.add_parser("prune", help="Prune stale partitions and mirrors (dry-run by default)")
    prune.add_argument("--apply", action="store_true", help="Apply deletions/migrations; omit for dry-run")
    prune.add_argument("--keep-latest-only", action="store_true", default=True, help="Retain only latest partition per dataset")
    prune.add_argument("--no-keep-latest-only", dest="keep_latest_only", action="store_false", help="Disable keep-latest pruning")
    prune.add_argument("--drop-nport-zip-mirrors", action="store_true", default=True, help="Drop N-PORT ZIP mirrors")
    prune.add_argument("--no-drop-nport-zip-mirrors", dest="drop_nport_zip_mirrors", action="store_false", help="Keep N-PORT ZIP mirrors")
    prune.add_argument("--migrate-results-layout", action="store_true", default=True, help="Migrate results layout")
    prune.add_argument("--no-migrate-results-layout", dest="migrate_results_layout", action="store_false", help="Skip results migration")
    return parser
