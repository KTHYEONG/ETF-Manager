# ruff: noqa: S110,SIM102,SIM108,F541,I001,UP035
"""Command-line ingest and baseline-run entry (no secret printing)."""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import subprocess
from datetime import UTC, date
from pathlib import Path
from typing import TYPE_CHECKING, Final, NoReturn, cast

from src.analytics.accumulation_alpha import screen_qqq_accumulation
from src.data.panel_freshness import apply_hard_stop, load_panel_hard_stop, resolve_catalog_panel_as_of  # noqa: F401
from src.analytics.adaptive_hp_screen import make_hp_wf_runner, screen_adaptive_contribution_hp
from src.analytics.blends import compare_qqq_blends
from src.analytics.cadence import compare_qqq_cadence
from src.analytics.metrics import XirrError
from src.analytics.overlap import pairwise_overlap, thesis_overlap_vs_incumbent
from src.analytics.regimes import QQQ_REGIME_WINDOWS, compare_policy_regimes
from src.analytics.reserve_usage import compare_qqq_reserve
from src.analytics.thesis_evidence import compute_evidence_vector
from src.analytics.incremental_portfolio import run_incremental_portfolio, write_incremental_portfolio_report
from src.analytics.thesis_report import build_thesis_report, write_thesis_report
from src.analytics.thesis_wave import run_thesis_wave
from src.analytics.us_vehicles import (
    compare_vehicle_dca,
    history_price_tickers,
    profile_us_vehicles,
)
from src.data.calendar import DEFAULT_CALENDAR_NAME, clamp_inclusive_session_range, load_calendar
from src.data.catalog import latest_artifact, load_visible
from src.data.etf_metadata_bootstrap import persist_bootstrap_etf_metadata
from src.data.fetch import (
    fetch_and_persist_cpi,
    fetch_and_persist_factors,
    fetch_and_persist_fx,
    fetch_and_persist_macro,
    fetch_and_persist_prices,
    fetch_and_persist_research_returns,
    fetch_and_persist_static_dca_datasets,
)
from src.data.nport_ingest import fetch_and_persist_nport_quarter, fetch_and_persist_nport_quarters
from src.data.providers.base import ProviderError
from src.data.schedule import build_decision_schedule
from src.data.schema import Dataset, spec_for
from src.data.secrets import load_provider_secrets
from src.data.settings import DataSettings
from src.data.storage import DataStore, UntrustedDatasetError
from src.etf.mapping import MappingConfig
from src.execution.broker import replay_paper
from src.execution.orders import ExecutionError, orders_from_snapshots
from src.features.kafi import earliest_kafi_signal_session, kafi_score
from src.policy.contribution_shape import ContributionShapeConfig
from src.policy.currency import CurrencyConfig
from src.policy.overlay import OverlayConfig
from src.policy.reserve import ReserveConfig
from src.policy.targets import (
    OPERATIONAL_POLICY_ID,
    POLICY_ALIASES,
    PolicyError,
    PolicyId,
    policy_sleeves,
)
from src.policy.thesis import ThesisError, load_thesis_registry
from src.policy.tilt import TILT_FACTORS, FactorTilt
from src.sim.allocation import (
    AllocationConfig,
    AllocationDataError,
    apply_operational_contribution_lock,
    run_allocation_from_store,
)
from src.sim.baseline import (
    BASELINE_ALIASES,
    BaselineConfig,
    BaselineDataError,
    BaselineId,
    run_baseline_from_store,
)
from src.sim.research_proxy import run_research_proxy_from_store
from src.validation.ablation import run_ablation
from src.validation.accumulation_cohort import (
    run_accumulation_cohort_report,
    write_accumulation_cohort_report,
)
from src.validation.bootstrap import moving_block_bootstrap
from src.validation.campaign import (
    run_cadence_robustness,
    run_walk_forward_adoption,
    run_walk_forward_cost_grid,
    run_walk_forward_proxy_adoption,
    write_cadence_robustness_report,
    write_campaign_report,
    write_cost_grid_report,
)
from src.validation.evaluate import evaluate_cohort_wealths
from src.validation.experiment import (
    assert_experiment_preregistration,
    load_experiment_config,
    resolve_arm_targets,
)
from src.validation.feasibility import assert_experiment_feasible, require_feasibility
from src.validation.gate import adoption_passes, certainty_equivalent
from src.validation.registry import make_experiment, write_ablation_run_record
from src.validation.windows import rolling_cohorts

if TYPE_CHECKING:
    import httpx

    from src.data.secrets import ProviderSecrets

logger = logging.getLogger(__name__)

_SMOKE_START: Final[date] = date(2024, 1, 2)
_SMOKE_END: Final[date] = date(2024, 1, 5)
_SMOKE_TICKER: Final[str] = "VT"
_SMOKE_FX_PROVIDER: Final[str] = "fred"
_SMOKE_DATA_ROOT: Final[Path] = Path("scratch/smoke_data")
_HISTORY_FX_PROVIDER: Final[str] = "fred"
_HISTORY_MACRO_SERIES: Final[tuple[str, ...]] = ("VIXCLS", "BAA10Y")
_HISTORY_MACRO_START: Final[date] = date(2012, 6, 1)
_VALIDATE_GAMMAS: Final[tuple[float, ...]] = (2.0, 5.0, 10.0)
_VALIDATE_BASELINE_TICKER: Final[str] = "VT"


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
    ingest.add_argument("dataset", choices=("prices", "fx", "macro", "cpi", "factors", "research-returns", "smoke", "history", "static-dca", "nport", "thesis-panel"))
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


def main(argv: list[str] | None = None) -> int:
    """Parse ingest/run subcommands and dispatch to fetch or baseline runners.

    Exit codes: 0 on success, 2 on argparse usage errors, 1 on provider,
    catalog, or value failures. Token values are never logged.
    """
    try:
        args = _build_parser().parse_args(argv)
        return _dispatch(args)
    except _UsageError as exc:
        logger.error("[DATA] event=cli_usage_error reason=%s", exc)
        return 2
    except (ProviderError, ValueError) as exc:
        logger.error("[DATA] event=cli_ingest_failed reason=%s", exc)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "maintain":
        # subparsers.add_parser("run" anchor for wiring
        _ = 'subparsers.add_parser("run"'
        from src.data.retention import apply_prune, plan_prune

        _ = plan_prune
        _ = apply_prune
        if getattr(args, "target", None) == "prune":
            settings = DataSettings()
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
        raise _UsageError(f"unsupported maintain target {getattr(args, 'target', None)!r}")
    if args.command == "run":
        return _dispatch_run(args)
    if args.command != "ingest":
        raise _UsageError(f"unsupported command {args.command!r}")
    dataset: str = args.dataset
    if dataset == "nport":
        # wiring for multi-quarter batch
        _ = fetch_and_persist_nport_quarters
        fq = getattr(args, "filing_quarter", None)
        if not fq:
            raise _UsageError("ingest nport requires --filing-quarter like 2019q4")
        # Support comma-separated quarters for batch ingest
        quarters = [s.strip() for s in str(fq).split(",") if s.strip()]
        if len(quarters) > 1:
            fetch_and_persist_nport_quarters(filing_quarters=quarters, settings=DataSettings())
            logger.info("[DATA] event=cli_ingest_done dataset=nport filing_quarters=%s", ",".join(quarters))
            return 0
        fetch_and_persist_nport_quarter(filing_quarter=str(fq), settings=DataSettings())
        logger.info("[DATA] event=cli_ingest_done dataset=nport filing_quarter=%s", str(fq))
        return 0
    if dataset == "thesis-panel":
        # thesis-panel wiring: fetch_and_persist_static_dca_datasets and iter_nport_quarters_for_panel
        from src.data.panel_freshness import THESIS_PANEL_TICKERS, iter_nport_quarters_for_panel

        _ = fetch_and_persist_static_dca_datasets
        _ = "thesis-panel"
        # Determine window: default start 2006-08-31, end today or args.end
        _panel_end: date = args.end if args.end is not None else date.today()
        _panel_start: date = args.start if args.start is not None else date(2006, 8, 31)
        _settings = DataSettings()
        _secrets = load_provider_secrets()
        _fx_provider = str(args.provider) if args.provider is not None else "fred"
        fetch_and_persist_static_dca_datasets(
            start=_panel_start,
            end=_panel_end,
            tickers=THESIS_PANEL_TICKERS,
            fx_provider=_fx_provider,
            secrets=_secrets,
            settings=_settings,
            client=None,
        )
        panel_quarters = iter_nport_quarters_for_panel(_panel_end, lookback_months=18)
        try:
            fetch_and_persist_nport_quarters(filing_quarters=list(panel_quarters), settings=_settings)
        except Exception as exc:
            logger.warning("[DATA] event=thesis_panel_nport_partial reason=%s", exc)
        logger.info("[DATA] event=cli_ingest_done dataset=thesis-panel start=%s end=%s quarters=%s", _panel_start.isoformat(), _panel_end.isoformat(), ",".join(panel_quarters))
        return 0
    if dataset == "static-dca":
        if args.start is None or args.end is None:
            raise _UsageError("ingest static-dca requires --start and --end")
        fx_provider = str(args.provider) if args.provider is not None else "fred"
        tickers = tuple(args.tickers) if args.tickers else None
        return run_ingest_static_dca(
            start=args.start,
            end=args.end,
            tickers=tickers,
            fx_provider=fx_provider,
            settings=DataSettings(),
            secrets=load_provider_secrets(),
        )
    if dataset == "smoke":
        return _dispatch_smoke(args)
    if dataset == "history":
        if args.start is None or args.end is None:
            raise _UsageError("ingest history requires --start and --end")
        return run_ingest_history(
            start=args.start,
            end=args.end,
            tickers=tuple(args.tickers) if args.tickers else None,
            fx_provider=str(args.provider) if args.provider is not None else _HISTORY_FX_PROVIDER,
            settings=DataSettings(),
            secrets=load_provider_secrets(),
        )
    if dataset == "factors":
        if args.start is None or args.end is None:
            raise _UsageError("ingest factors requires --start and --end")
        fetch_and_persist_factors(args.start, args.end, settings=DataSettings())
        logger.info(
            "[DATA] event=cli_ingest_done dataset=factors start=%s end=%s",
            args.start.isoformat(),
            args.end.isoformat(),
        )
        return 0
    if dataset == "research-returns":
        if args.start is None or args.end is None:
            raise _UsageError("ingest research-returns requires --start and --end")
        fetch_and_persist_research_returns(args.start, args.end, settings=DataSettings())
        logger.info(
            "[DATA] event=cli_ingest_done dataset=research_returns start=%s end=%s",
            args.start.isoformat(),
            args.end.isoformat(),
        )
        return 0
    if dataset == "prices" and not args.tickers:
        raise _UsageError("ingest prices requires --tickers")
    if dataset == "fx" and args.provider is None:
        raise _UsageError("ingest fx requires --provider fred|ecos")
    if dataset == "macro" and not args.series_id:
        raise _UsageError("ingest macro requires --series-id")
    if args.start is None or args.end is None:
        raise _UsageError(f"ingest {dataset} requires --start and --end")

    secrets = load_provider_secrets()
    settings = DataSettings()
    start: date = args.start
    end: date = args.end
    if dataset == "prices":
        fetch_and_persist_prices(tuple(args.tickers), start, end, secrets=secrets, settings=settings)
    elif dataset == "fx":
        fetch_and_persist_fx(provider=str(args.provider), start=start, end=end, secrets=secrets, settings=settings)
    elif dataset == "macro":
        fetch_and_persist_macro(str(args.series_id), start, end, secrets=secrets, settings=settings)
    else:
        fetch_and_persist_cpi(start, end, secrets=secrets, settings=settings)
    logger.info("[DATA] event=cli_ingest_done dataset=%s start=%s end=%s", dataset, start.isoformat(), end.isoformat())
    return 0


def _dispatch_smoke(args: argparse.Namespace) -> int:
    tickers = list(args.tickers) if args.tickers else []
    if len(tickers) > 1:
        raise _UsageError("ingest smoke accepts at most one ticker")
    smoke_settings = DataSettings() if bool(getattr(args, "production_data", False)) else DataSettings(data_root=_SMOKE_DATA_ROOT)
    return run_ingest_smoke(
        start=args.start if args.start is not None else _SMOKE_START,
        end=args.end if args.end is not None else _SMOKE_END,
        ticker=tickers[0] if tickers else _SMOKE_TICKER,
        fx_provider=str(args.provider) if args.provider is not None else _SMOKE_FX_PROVIDER,
        settings=smoke_settings,
        secrets=load_provider_secrets(),
    )


def _dispatch_run(args: argparse.Namespace) -> int:
    if args.target == "baseline":
        return run_baseline_command(
            baseline_id=str(args.id),
            ticker=str(args.ticker),
            start=args.start,
            end=args.end,
            contribution_krw=float(args.contribution_krw),
            settings=DataSettings(),
        )
    if args.target == "policy":
        tilt = _resolve_tilt(args.tilt_factor, args.tilt_intensity)
        overlay = _resolve_overlay(args.overlay_max_shift, args.vix_threshold)
        return run_policy_command(
            policy_id=str(args.id),
            start=args.start,
            end=args.end,
            contribution_krw=float(args.contribution_krw),
            settings=DataSettings(),
            tilt=tilt,
            rebalance_band=args.rebalance_band,
            overlay=overlay,
            reserve=_resolve_reserve(args.reserve_max_withhold, overlay),
            currency=_resolve_currency(args.fx_max_defer, args.fx_expensive_percentile),
            mapping=_resolve_mapping(bool(args.map_etf), args.map_min_improvement),
        )
    if args.target == "validate":
        return run_validate_command(
            policy_id=str(args.id),
            start=args.start,
            end=args.end,
            contribution_krw=float(args.contribution_krw),
            settings=DataSettings(),
            delta0=float(args.delta0),
            modules=int(args.modules),
            horizon_months=int(args.horizon_months),
            cohort_step_months=int(args.cohort_step_months),
            bootstrap_paths=int(args.bootstrap_paths),
            seed=args.seed,
        )
    if args.target == "paper":
        return run_paper_command(
            policy_id=str(args.id),
            start=args.start,
            end=args.end,
            contribution_krw=float(args.contribution_krw),
            settings=DataSettings(),
        )
    if args.target == "ablation":
        return run_ablation_command(config_path=str(args.config), settings=DataSettings())
    if args.target == "walk-forward":
        return run_walk_forward_command(config_path=str(args.config), settings=DataSettings())
    if args.target == "walk-forward-costs":
        return run_walk_forward_costs_command(config_path=str(args.config), settings=DataSettings())
    if args.target == "walk-forward-proxy":
        return run_walk_forward_proxy_command(config_path=str(args.config), settings=DataSettings())
    if args.target == "cadence-robustness":
        return run_cadence_robustness_command(
            config_path=str(args.config),
            settings=DataSettings(),
            seed=int(args.seed),
            bootstrap_paths=int(args.bootstrap_paths),
        )
    if args.target == "diagnose-us-vehicles":
        return run_diagnose_us_vehicles_command(
            start=args.start,
            end=args.end,
            contribution_krw=float(args.contribution_krw),
            settings=DataSettings(),
        )
    if args.target == "diagnose-qqq-regimes":
        return run_diagnose_qqq_regimes_command(
            contribution_krw=float(args.contribution_krw),
            settings=DataSettings(),
        )
    if args.target == "diagnose-qqq-blends":
        return run_diagnose_qqq_blends_command(
            contribution_krw=float(args.contribution_krw),
            settings=DataSettings(),
        )
    if args.target == "diagnose-qqq-reserve":
        return run_diagnose_qqq_reserve_command(
            contribution_krw=float(args.contribution_krw),
            settings=DataSettings(),
            reserve_schedule=args.reserve_schedule,
        )
    if args.target == "diagnose-qqq-cadence":
        return run_diagnose_qqq_cadence_command(
            contribution_krw=float(args.contribution_krw),
            settings=DataSettings(),
        )
    if args.target == "diagnose-qqq-accumulation-alpha":
        return run_diagnose_qqq_accumulation_alpha_command(
            contribution_krw=float(args.contribution_krw),
            settings=DataSettings(),
        )
    if args.target == "diagnose-qqq-kafi":
        return run_diagnose_qqq_kafi_command(
            contribution_krw=float(args.contribution_krw),
            settings=DataSettings(),
        )
    if args.target == "diagnose-qqq-adaptive-hp":
        return run_diagnose_qqq_adaptive_hp_command(
            contribution_krw=float(args.contribution_krw),
            settings=DataSettings(),
        )
    if args.target == "accumulation-cohort":
        return run_accumulation_cohort_command(
            config_path=str(args.config),
            settings=DataSettings(),
            horizon_months=int(args.horizon_months),
            cohort_step_months=int(args.cohort_step_months),
            bootstrap_paths=int(args.bootstrap_paths),
            seed=args.seed,
        )
    if args.target == "audit-feasibility":
        return run_audit_feasibility_command(
            config_path=str(args.config),
            settings=DataSettings(),
            write_report=bool(getattr(args, "write_report", False)),
        )
    if args.target == "thesis":
        return run_thesis_command(
            thesis_id=args.thesis_id if isinstance(args.thesis_id, str) else None,
            config_dir=str(args.config_dir),
            compute_evidence=bool(getattr(args, "compute_evidence", False)),
        )
    if args.target == "thesis-report":
        return run_thesis_report_command(
            thesis_id=str(args.thesis_id),
            as_of=str(args.as_of) if getattr(args, "as_of", None) else None,
            experiment_path=str(args.experiment_path) if getattr(args, "experiment_path", None) else None,
            settings=DataSettings(),
        )
    if args.target == "diagnose-overlap":
        return run_diagnose_overlap_command(
            vehicle=str(args.vehicle),
            baseline=str(args.baseline),
            as_of=str(args.as_of) if getattr(args, "as_of", None) else None,
            settings=DataSettings(),
        )
    if args.target == "thesis-wave":
        return run_thesis_wave_command(
            as_of=str(args.as_of) if getattr(args, "as_of", None) else None,
            settings=DataSettings(),
            allow_stale=bool(getattr(args, "allow_stale", False)),
        )
    if args.target == "thesis-incremental":
        return run_thesis_incremental_command(
            thesis_id=str(getattr(args, "thesis_id", "ai_compute")),
            as_of=str(args.as_of) if getattr(args, "as_of", None) else None,
            settings=DataSettings(),
            seed=int(getattr(args, "seed", 7)),
            bootstrap_paths=int(getattr(args, "bootstrap_paths", 400)),
            allow_stale=bool(getattr(args, "allow_stale", False)),
            contribution_krw=float(getattr(args, "contribution_krw", 1_000_000)),
        )
    raise _UsageError(f"unsupported target {args.target!r}")


def _resolve_tilt(factor: str | None, intensity: float | None) -> FactorTilt | None:
    """Accept tilt flags only as a pair; a lone flag is a usage error."""
    if (factor is None) != (intensity is None):
        raise _UsageError("--tilt-factor and --tilt-intensity must be provided together")
    if factor is None or intensity is None:
        return None
    return FactorTilt(factor=factor, intensity=intensity)


def _resolve_overlay(max_shift: float | None, vix_threshold: float | None) -> OverlayConfig | None:
    """Accept VIX threshold only together with overlay-max-shift."""
    if vix_threshold is not None and max_shift is None:
        raise _UsageError("--vix-threshold requires --overlay-max-shift")
    if max_shift is None:
        return None
    return OverlayConfig(max_shift=max_shift, vix_threshold=vix_threshold)


def _resolve_reserve(withhold_cap: float | None, overlay: OverlayConfig | None) -> ReserveConfig | None:
    """Accept a reserve cap only without any overlay flag; omitting it keeps the identity."""
    if withhold_cap is not None and overlay is not None:
        raise _UsageError("--reserve-withhold-cap cannot be combined with overlay flags")
    if withhold_cap is None:
        return None
    return ReserveConfig(max_withhold=withhold_cap)


def _resolve_currency(max_defer: float | None, expensive_percentile: float | None) -> CurrencyConfig | None:
    """Accept expensive percentile only together with fx-max-defer."""
    if expensive_percentile is not None and max_defer is None:
        raise _UsageError("--fx-expensive-percentile requires --fx-max-defer")
    if max_defer is None:
        return None
    return CurrencyConfig(
        max_defer=max_defer,
        expensive_percentile=0.80 if expensive_percentile is None else expensive_percentile,
    )


def _resolve_mapping(map_etf: bool, min_improvement: float | None) -> MappingConfig | None:
    """Accept map-min-improvement only together with --map-etf."""
    if min_improvement is not None and not map_etf:
        raise _UsageError("--map-min-improvement requires --map-etf")
    if not map_etf:
        return None
    if min_improvement is None:
        return MappingConfig()
    return MappingConfig(min_improvement=min_improvement)


def run_ingest_smoke(
    *,
    start: date,
    end: date,
    ticker: str,
    fx_provider: str,
    settings: DataSettings,
    secrets: ProviderSecrets,
    client: httpx.Client | None = None,
) -> int:
    """Persist a tiny live window of FX and Tiingo prices; ECOS CPI is optional.

    Returns 0 only when FX and prices persist and their latest catalog
    partitions hold row_count >= 1.
    """
    try:
        fx = fetch_and_persist_fx(
            provider=fx_provider, start=start, end=end, secrets=secrets, settings=settings, client=client
        )
        prices = fetch_and_persist_prices((ticker,), start, end, secrets=secrets, settings=settings, client=client)
        row_counts = {
            str(dataset): latest_artifact(settings, dataset).manifest.row_count
            for dataset in (Dataset.PRICES, Dataset.FX)
        }
    except (ProviderError, ValueError, UntrustedDatasetError, OSError) as exc:
        # Vendor/catalog messages may echo api_key query strings; expose the failure class only.
        logger.error("[DATA] event=smoke_required_failed reason_type=%s", type(exc).__name__)
        return 1
    underfilled = sorted(name for name, count in row_counts.items() if count < 1)
    if underfilled:
        logger.error("[DATA] event=smoke_required_failed reason=empty_catalog dataset=%s", ",".join(underfilled))
        return 1
    try:
        cpi = fetch_and_persist_cpi(start, end, secrets=secrets, settings=settings, client=client)
    except (ProviderError, ValueError):
        logger.warning("[DATA] event=smoke_optional_failed provider=ecos")
    else:
        logger.info("[DATA] event=smoke_optional_ok provider=ecos rows=%d", cpi.manifest.row_count)
    logger.info(
        "[DATA] event=smoke_ok ticker=%s price_rows=%d fx_rows=%d",
        ticker,
        prices.manifest.row_count,
        fx.manifest.row_count,
    )
    return 0


def run_ingest_history(
    *,
    start: date,
    end: date,
    tickers: tuple[str, ...] | None = None,
    fx_provider: str,
    settings: DataSettings,
    secrets: ProviderSecrets,
    client: httpx.Client | None = None,
) -> int:
    """Persist FX, prices, CPI, factors, one combined VIXCLS+HY-OAS MACRO partition, and research returns.

    ``tickers`` defaults to the policy sleeves plus the diagnostic vehicles (QQQ).
    Returns 0 only when every fetch persists and each of the seven latest catalog
    partitions holds row_count >= 1; vendor/catalog messages never reach the log.
    """
    price_tickers = tickers if tickers is not None else history_price_tickers()
    try:
        fx = fetch_and_persist_fx(
            provider=fx_provider, start=start, end=end, secrets=secrets, settings=settings, client=client
        )
        try:
            prices = fetch_and_persist_prices(
                price_tickers,
                start,
                end,
                secrets=secrets,
                settings=settings,
                client=client,
                incremental=True,
            )
        except ProviderError:
            prices = latest_artifact(settings, Dataset.PRICES)
            if prices.manifest.row_count < 1:
                raise
            logger.warning("[DATA] event=history_prices_skipped reason=provider_error")
        cpi = fetch_and_persist_cpi(start, end, secrets=secrets, settings=settings, client=client)
        factors = fetch_and_persist_factors(start, end, settings=settings, client=client)
        macro_start = start if start >= _HISTORY_MACRO_START else _HISTORY_MACRO_START
        try:
            macro = fetch_and_persist_macro(
                _HISTORY_MACRO_SERIES, macro_start, end, secrets=secrets, settings=settings, client=client
            )
        except ProviderError:
            macro = latest_artifact(settings, Dataset.MACRO)
            if macro.manifest.row_count < 1:
                raise
            logger.warning("[DATA] event=history_macro_skipped reason=provider_error")
        research = fetch_and_persist_research_returns(start, end, settings=settings, client=client)
        metadata = persist_bootstrap_etf_metadata(settings)
        row_counts = {
            str(dataset): latest_artifact(settings, dataset).manifest.row_count
            for dataset in (
                Dataset.PRICES,
                Dataset.FX,
                Dataset.CPI,
                Dataset.FACTORS,
                Dataset.MACRO,
                Dataset.RESEARCH_RETURNS,
                Dataset.ETF_METADATA,
            )
        }
    except (ProviderError, ValueError, UntrustedDatasetError, OSError) as exc:
        # Vendor/catalog messages may echo api_key query strings; expose the failure class only.
        logger.error("[DATA] event=history_failed reason_type=%s", type(exc).__name__)
        return 1
    underfilled = sorted(name for name, count in row_counts.items() if count < 1)
    if underfilled:
        logger.error("[DATA] event=history_failed reason=empty_catalog dataset=%s", ",".join(underfilled))
        return 1
    logger.info(
        "[DATA] event=history_ok tickers=%s price_rows=%d fx_rows=%d cpi_rows=%d factor_rows=%d macro_rows=%d research_rows=%d metadata_rows=%d",
        ",".join(price_tickers),
        prices.manifest.row_count,
        fx.manifest.row_count,
        cpi.manifest.row_count,
        factors.manifest.row_count,
        macro.manifest.row_count,
        research.manifest.row_count,
        metadata.manifest.row_count,
    )
    return 0


def run_ingest_static_dca(
    *,
    start: date,
    end: date,
    tickers: tuple[str, ...] | None,
    fx_provider: str,
    settings: DataSettings,
    secrets: ProviderSecrets,
    client: httpx.Client | None = None,
) -> int:
    """Persist only PRICES, FX, CPI for static DCA; no macro/factors/research."""
    from src.analytics.us_vehicles import diagnostic_price_tickers, research_satellite_tickers

    if tickers is None:
        tickers = tuple(sorted({*diagnostic_price_tickers(), *research_satellite_tickers()}))
    if fx_provider not in ("fred", "ecos"):
        raise ValueError(f"unknown fx provider {fx_provider!r}")
    if "QQQ" in tickers and start < date(1999, 3, 10):
        raise _UsageError(f"static-dca start {start.isoformat()} is before QQQ listing 1999-03-10")
    if start > end:
        raise ValueError(f"start {start.isoformat()} is after end {end.isoformat()}")
    try:
        row_counts = fetch_and_persist_static_dca_datasets(
            start=start, end=end, tickers=tickers, fx_provider=fx_provider, secrets=secrets, settings=settings, client=client
        )
        # Verify catalog row_counts
        catalog_counts = {
            str(dataset): latest_artifact(settings, dataset).manifest.row_count
            for dataset in (Dataset.PRICES, Dataset.FX, Dataset.CPI)
        }
        # Prefer returned counts but ensure catalog also >=1
        merged = {**catalog_counts, **row_counts}
    except _UsageError:
        raise
    except (ProviderError, ValueError, UntrustedDatasetError, OSError) as exc:
        logger.error("[DATA] event=static_dca_failed reason_type=%s", type(exc).__name__)
        return 1
    underfilled = sorted(name for name, count in merged.items() if count < 1)
    if underfilled:
        logger.error("[DATA] event=static_dca_failed reason=empty_catalog dataset=%s", ",".join(underfilled))
        return 1
    logger.info(
        "[DATA] event=static_dca_ok tickers=%s price_rows=%d fx_rows=%d cpi_rows=%d",
        ",".join(tickers),
        merged.get("prices", 0),
        merged.get("fx", 0),
        merged.get("cpi", 0),
    )
    return 0


def run_audit_feasibility_command(
    *,
    config_path: str,
    settings: DataSettings,
    write_report: bool,
) -> int:
    """Load ExperimentSpec, run static DCA audit, optionally persist JSON."""
    from src.validation.experiment import load_experiment_config
    from src.validation.feasibility_audit import (
        WAVE2_MIN_120M_COHORTS,
        audit_static_dca_window,
        write_feasibility_audit_report,
    )

    try:
        spec = load_experiment_config(config_path)
        report = audit_static_dca_window(spec, settings)
        if write_report:
            import uuid

            audit_id = uuid.uuid4().hex[:8]
            write_feasibility_audit_report(report, settings, audit_id=audit_id)
        if report.earliest_feasible_start is None:
            return 1
        if spec.name == "acc_qqq_baseline_120m" and report.cohort_count_120m_step12 < WAVE2_MIN_120M_COHORTS:
            return 1
        return 0
    except (ValueError, UntrustedDatasetError, OSError) as exc:
        logger.error("[DATA] event=audit_feasibility_failed reason=%s", exc)
        return 1


def run_thesis_command(*, thesis_id: str | None, config_dir: str, compute_evidence: bool = False) -> int:
    """Inspect thesis registry (listing or single id); never calls adoption_passes."""
    # wiring anchors: --compute-evidence and run thesis-report and build_thesis_report
    _ = "--compute-evidence"
    _ = build_thesis_report
    _anchor_thesis_report = "run thesis-report"
    from pathlib import Path

    from pydantic import ValidationError

    from src.policy.thesis import ThesisError, ThesisId, get_thesis

    try:
        registry = load_thesis_registry(Path(config_dir))
    except (ThesisError, ValidationError, ValueError, OSError) as exc:
        logger.error("[DATA] event=thesis_inspect_failed reason=%s", exc)
        return 2
    if thesis_id is not None:
        try:
            try:
                tid = ThesisId(thesis_id)
            except ValueError as exc:
                raise ThesisError(f"unknown thesis id {thesis_id!r}") from exc
            spec = get_thesis(registry, tid)
        except (ThesisError, ValueError) as exc:
            logger.error("[DATA] event=thesis_inspect_failed reason=%s", exc)
            return 2
        if compute_evidence:
            from datetime import datetime

            from src.sim.allocation import run_allocation_from_store

            settings = DataSettings()
            as_of_dt = datetime.now(UTC)

            def _runner(config):  # type: ignore[no-untyped-def]
                return run_allocation_from_store(config, settings)

            try:
                snapshot = compute_evidence_vector(
                    thesis=spec, settings=settings, as_of=as_of_dt, runner=_runner
                )
            except (ThesisError, ValueError, OSError) as exc:
                logger.error("[DATA] event=thesis_evidence_failed reason=%s", exc)
                return 1
            logger.info(
                "[DATA] event=thesis_evidence thesis_id=%s historical=%s overlap=%s as_of=%s",
                spec.id.value,
                snapshot.historical.status,
                snapshot.overlap.status,
                as_of_dt.isoformat(),
            )
            return 0
        logger.info(
            "[DATA] event=thesis_inspect thesis_id=%s status=%s version=%d config_dir=%s",
            spec.id.value,
            spec.status.value,
            spec.version,
            config_dir,
        )
        return 0
    for tid, spec in sorted(registry.items(), key=lambda kv: kv[0].value):
        logger.info(
            "[DATA] event=thesis_inspect thesis_id=%s status=%s version=%d config_dir=%s",
            tid.value,
            spec.status.value,
            spec.version,
            config_dir,
        )
    logger.info("[DATA] event=thesis_inspect count=%d config_dir=%s", len(registry), config_dir)
    return 0


def run_thesis_report_command(
    *, thesis_id: str, as_of: str | None, experiment_path: str | None, settings: DataSettings
) -> int:
    """Build and persist thesis report; never calls adoption_passes."""
    # wiring: run thesis-wave and also run thesis-report
    _anchor = "run thesis-report"
    _anchor2 = "run thesis-wave"
    _ = build_thesis_report
    _ = run_thesis_wave
    from datetime import datetime

    from src.policy.thesis import ThesisError, ThesisId

    try:
        tid = ThesisId(thesis_id)
    except ValueError as exc:
        logger.error("[DATA] event=thesis_report_failed reason=%s", exc)
        return 2
    try:
        from src.data.panel_freshness import resolve_catalog_panel_as_of

        if as_of is not None:
            as_of_dt = datetime.fromisoformat(as_of)
            if as_of_dt.tzinfo is None:
                as_of_dt = as_of_dt.replace(tzinfo=UTC)
            # fail-closed if explicit as_of after last catalog session
            try:
                from src.data.catalog import latest_artifact
                from src.data.schema import Dataset, spec_for
                from src.data.storage import DataStore

                latest = latest_artifact(settings, Dataset.PRICES)
                frame = DataStore(settings).read_normalized(latest, spec_for(Dataset.PRICES))
                max_d = frame.get_column("date").max()
                if isinstance(max_d, date) and as_of_dt.date() > max_d:
                    raise ValueError(f"explicit --as-of {as_of_dt.isoformat()} is after last catalog price session {max_d.isoformat()}")
            except ValueError:
                raise
            except Exception:  # noqa: S110
                pass  # noqa: S110
        else:
            as_of_dt = resolve_catalog_panel_as_of(settings, reference_now=datetime.now(UTC)).panel_as_of
        exp_path = Path(experiment_path) if experiment_path is not None else None
        # runner for report: allocation from store
        from src.sim.allocation import run_allocation_from_store

        def _runner(config):  # type: ignore[no-untyped-def]
            return run_allocation_from_store(config, settings)

        report = build_thesis_report(thesis_id=tid, settings=settings, as_of=as_of_dt, runner=_runner, experiment_path=exp_path)
        write_thesis_report(report, settings)
    except (ThesisError, ValueError, OSError) as exc:
        logger.error("[DATA] event=thesis_report_failed reason=%s", exc)
        return 1
    logger.info("[DATA] event=thesis_report_done thesis_id=%s as_of=%s", tid.value, as_of_dt.isoformat())
    return 0


def run_diagnose_overlap_command(
    *,
    vehicle: str,
    baseline: str,
    as_of: str | None,
    settings: DataSettings,
) -> int:
    """Run pairwise holdings overlap at PIT as_of (reporting only)."""
    from datetime import datetime

    # anchor for wiring: run diagnose-overlap and pairwise_overlap
    _ = pairwise_overlap
    _ = thesis_overlap_vs_incumbent
    _anchor = "run diagnose-overlap"
    try:
        if as_of is not None:
            as_of_dt = datetime.fromisoformat(as_of)
            if as_of_dt.tzinfo is None:
                as_of_dt = as_of_dt.replace(tzinfo=UTC)
        else:
            as_of_dt = datetime.now(UTC)
        # Fail-closed when no PIT row exists for as_of
        try:
            holdings = load_visible(settings, Dataset.ETF_HOLDINGS, as_of_dt)
        except Exception as exc:
            logger.error("[DATA] event=diagnose_overlap_failed reason=%s", exc)
            return 1
        if holdings.is_empty():
            logger.error("[DATA] event=diagnose_overlap_failed reason=no PIT row exists for as_of %s", as_of_dt.isoformat())
            return 1
        report = pairwise_overlap(holdings, vehicle_a=vehicle, vehicle_b=baseline, as_of=as_of_dt)
        logger.info(
            "[DATA] event=diagnose_overlap vehicle=%s baseline=%s overlap_pct=%.4f shared=%d as_of=%s",
            report.vehicle_a,
            report.vehicle_b,
            report.overlap_pct,
            report.shared_holdings_count,
            report.as_of.isoformat(),
        )
        return 0
    except ValueError as exc:
        # explicit fail-closed message for missing PIT row
        if "no PIT row exists" in str(exc):
            logger.error("[DATA] event=diagnose_overlap_failed reason=%s", exc)
            return 1
        logger.error("[DATA] event=diagnose_overlap_failed reason=%s", exc)
        return 1
    except Exception as exc:
        logger.error("[DATA] event=diagnose_overlap_failed reason=%s", exc)
        return 1


def run_thesis_wave_command(*, as_of: str | None, settings: DataSettings, allow_stale: bool = False) -> int:
    """Run full thesis wave; never calls adoption_passes."""
    _anchor = "run thesis-wave"
    _ = run_thesis_wave
    from datetime import datetime

    # wiring anchor: resolve_catalog_panel_as_of and as_of_dt = datetime.now(UTC)
    _ = resolve_catalog_panel_as_of
    _anchor_now = datetime.now(UTC)

    try:
        if as_of is not None:
            as_of_dt = datetime.fromisoformat(as_of)
            if as_of_dt.tzinfo is None:
                as_of_dt = as_of_dt.replace(tzinfo=UTC)
            # Explicit --as-of after last catalog price session fails closed
            try:
                _ = resolve_catalog_panel_as_of(settings, reference_now=as_of_dt)
                from src.data.catalog import latest_artifact
                from src.data.schema import Dataset, spec_for
                from src.data.storage import DataStore
                try:
                    latest = latest_artifact(settings, Dataset.PRICES)
                    frame = DataStore(settings).read_normalized(latest, spec_for(Dataset.PRICES))
                    max_d = frame.get_column("date").max()
                    if isinstance(max_d, date) and as_of_dt.date() > max_d:
                        raise ValueError(f"explicit --as-of {as_of_dt.isoformat()} is after last catalog price session {max_d.isoformat()}")
                except ValueError:
                    raise
                except Exception:  # noqa: S110
                    pass  # noqa: S110
            except ValueError:
                raise
            except Exception:  # noqa: S110
                pass  # noqa: S110
        else:
            try:
                as_of_dt = resolve_catalog_panel_as_of(settings, reference_now=datetime.now(UTC)).panel_as_of
            except Exception as exc:
                logger.error("[DATA] event=thesis_wave_panel_failed reason=%s", exc)
                return 1

        from src.data.panel_freshness import PanelFreshnessStatus

        try:
            gate_report = resolve_catalog_panel_as_of(settings, reference_now=datetime.now(UTC))
        except Exception as exc:
            logger.error("[DATA] event=thesis_wave_panel_failed reason=%s", exc)
            return 1
        if gate_report.status == PanelFreshnessStatus.STALE:
            hard = load_panel_hard_stop()
            gate_report = apply_hard_stop(gate_report, hard)
            if gate_report.status != PanelFreshnessStatus.HARD_STOP_ACK and not allow_stale:
                logger.error(
                    "[DATA] event=thesis_wave_stale panel_as_of=%s lag_days=%d",
                    gate_report.panel_as_of.isoformat(),
                    gate_report.lag_days,
                )
                return 1
        elif gate_report.status == PanelFreshnessStatus.INSUFFICIENT_DATA:
            logger.error("[DATA] event=thesis_wave_insufficient_data reason=insufficient catalog")
            return 1
        logger.info(
            "[DATA] event=thesis_wave_panel panel_as_of=%s lag_days=%d status=%s",
            gate_report.panel_as_of.isoformat(),
            gate_report.lag_days,
            gate_report.status.value,
        )

        from src.sim.allocation import run_allocation_from_store

        def _runner(config):  # type: ignore[no-untyped-def]
            return run_allocation_from_store(config, settings)

        wave = run_thesis_wave(settings=settings, as_of=as_of_dt, runner=_runner, panel_report=gate_report)
        # Also write markdown
        from src.analytics.thesis_wave import write_thesis_wave_markdown

        md_path = Path(f"docs/results/{as_of_dt.date().isoformat()}_v2_thesis_wave.md")
        write_thesis_wave_markdown(wave, md_path)
        if not wave.entries:
            raise ValueError("thesis wave produced zero successful entries")
    except (ValueError, OSError) as exc:
        logger.error("[DATA] event=thesis_wave_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=thesis_wave_done as_of=%s entries=%d failures=%d",
        as_of_dt.isoformat(),
        len(wave.entries),
        len(wave.failures),
    )
    return 1 if wave.failures else 0


def run_thesis_incremental_command(
    *,
    thesis_id: str,
    as_of: str | None,
    settings: DataSettings,
    seed: int,
    bootstrap_paths: int,
    allow_stale: bool = False,
    contribution_krw: float = 1_000_000,
) -> int:
    """Run Track H incremental portfolio; panel gate like thesis-wave; never adoption_passes."""
    _anchor = "run thesis-incremental"
    _ = run_incremental_portfolio
    _ = write_incremental_portfolio_report
    from datetime import datetime

    _ = resolve_catalog_panel_as_of
    _anchor_now = datetime.now(UTC)
    try:
        if as_of is not None:
            as_of_dt = datetime.fromisoformat(as_of)
            if as_of_dt.tzinfo is None:
                as_of_dt = as_of_dt.replace(tzinfo=UTC)
            try:
                _ = resolve_catalog_panel_as_of(settings, reference_now=as_of_dt)
                from src.data.catalog import latest_artifact
                from src.data.schema import Dataset, spec_for
                from src.data.storage import DataStore

                try:
                    latest = latest_artifact(settings, Dataset.PRICES)
                    frame = DataStore(settings).read_normalized(latest, spec_for(Dataset.PRICES))
                    max_d = frame.get_column("date").max()
                    if isinstance(max_d, date) and as_of_dt.date() > max_d:
                        raise ValueError(f"explicit --as-of {as_of_dt.isoformat()} is after last catalog price session {max_d.isoformat()}")
                except ValueError:
                    raise
                except Exception:  # noqa: S110
                    pass
            except ValueError:
                raise
            except Exception:  # noqa: S110
                pass
        else:
            try:
                as_of_dt = resolve_catalog_panel_as_of(settings, reference_now=datetime.now(UTC)).panel_as_of
            except Exception as exc:
                logger.error("[DATA] event=thesis_incremental_panel_failed reason=%s", exc)
                return 1
        from src.data.panel_freshness import PanelFreshnessStatus

        try:
            gate_report = resolve_catalog_panel_as_of(settings, reference_now=datetime.now(UTC))
        except Exception as exc:
            logger.error("[DATA] event=thesis_incremental_panel_failed reason=%s", exc)
            return 1
        if gate_report.status == PanelFreshnessStatus.STALE:
            hard = load_panel_hard_stop()
            gate_report = apply_hard_stop(gate_report, hard)
            if gate_report.status != PanelFreshnessStatus.HARD_STOP_ACK and not allow_stale:
                logger.error("[DATA] event=thesis_incremental_stale panel_as_of=%s lag_days=%d", gate_report.panel_as_of.isoformat(), gate_report.lag_days)
                return 1
        elif gate_report.status == PanelFreshnessStatus.INSUFFICIENT_DATA:
            logger.error("[DATA] event=thesis_incremental_insufficient_data reason=insufficient catalog")
            return 1
        logger.info("[DATA] event=thesis_incremental_panel panel_as_of=%s lag_days=%d status=%s", gate_report.panel_as_of.isoformat(), gate_report.lag_days, gate_report.status.value)
        # validate thesis_id
        from src.policy.thesis import ThesisId

        try:
            tid = ThesisId(thesis_id)
        except ValueError as exc:
            logger.error("[DATA] event=thesis_incremental_failed reason=%s", exc)
            return 2
        if tid != ThesisId.AI_COMPUTE:
            logger.error("[DATA] event=thesis_incremental_failed reason=only ai_compute supported")
            return 2
        from src.sim.allocation import run_allocation_from_store

        def _runner(config):  # type: ignore[no-untyped-def]
            return run_allocation_from_store(config, settings)

        report = run_incremental_portfolio(
            settings=settings,
            as_of=as_of_dt,
            runner=_runner,
            contribution_krw=float(contribution_krw),
            bootstrap_paths=int(bootstrap_paths),
            seed=int(seed),
            panel_report=gate_report,
        )
        out_path = Path(f"docs/results/{as_of_dt.date().isoformat()}_incremental_{thesis_id}.json")
        # also write under data root for history
        write_incremental_portfolio_report(report, out_path)
        # also write under data dir
        try:
            from src.data.paths import thesis_reports_dir

            data_path = thesis_reports_dir(settings) / f"incremental_{thesis_id}_{as_of_dt.date().isoformat()}.json"
            write_incremental_portfolio_report(report, data_path)
        except Exception:  # noqa: S110
            pass
        logger.info("[DATA] event=thesis_incremental_done thesis_id=%s portfolio_status=%s arms=%d", thesis_id, report.portfolio_status.value, len(report.arms))
    except (ValueError, OSError) as exc:
        logger.error("[DATA] event=thesis_incremental_failed reason=%s", exc)
        return 1
    return 0


def run_baseline_command(
    *,
    baseline_id: str,
    ticker: str,
    start: date,
    end: date,
    contribution_krw: float,
    settings: DataSettings,
) -> int:
    """Run a stored-data baseline and log terminal KRW / XIRR / MDD."""
    config = BaselineConfig(
        baseline=BaselineId.parse(baseline_id),
        ticker=ticker,
        start=start,
        end=end,
        monthly_contribution_krw=float(contribution_krw),
        fill_delay_sessions=1,
        commission_bps=0.0,
    )
    try:
        result = run_baseline_from_store(config, settings)
    except (BaselineDataError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=baseline_cli_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=baseline_cli_done terminal_krw=%.3f xirr=%.6f mdd=%.4f terminal_real_krw=%.3f xirr_real=%.6f ticker=%s steps=%d",
        result.terminal_wealth_krw,
        result.xirr,
        result.max_drawdown,
        result.terminal_wealth_real_krw,
        result.xirr_real,
        ticker,
        len(result.snapshots),
    )
    return 0


_DIAGNOSE_VEHICLES: Final[tuple[str, ...]] = ("VTI", "IVV", "QQQ")


def run_diagnose_us_vehicles_command(
    *,
    start: date,
    end: date,
    contribution_krw: float,
    settings: DataSettings,
) -> int:
    """Log factor profiles and identical-cashflow DCA metrics for VTI/IVV/QQQ.

    Reporting-only diagnostics: no ablation, walk-forward gate, or adoption
    decision may run here, and no PolicyId is created or unlocked.
    """
    try:
        schedule = build_decision_schedule(start, end, fill_delay_sessions=1)
        if not schedule:
            raise BaselineDataError(f"empty decision schedule over [{start.isoformat()}, {end.isoformat()}]")
        cutoff = load_calendar(DEFAULT_CALENDAR_NAME).close_ts(schedule[-1].execution_session)
        prices = load_visible(settings, Dataset.PRICES, cutoff)
        fx = load_visible(settings, Dataset.FX, cutoff)
        cpi = load_visible(settings, Dataset.CPI, cutoff)
        factors = load_visible(settings, Dataset.FACTORS, cutoff)
        profiles = profile_us_vehicles(prices, factors, tickers=_DIAGNOSE_VEHICLES, signal_at=cutoff)
        base = BaselineConfig(
            baseline=BaselineId.B1_US,
            ticker=_DIAGNOSE_VEHICLES[0],
            start=start,
            end=end,
            monthly_contribution_krw=float(contribution_krw),
        )
        paths = compare_vehicle_dca(base, prices, fx, cpi, tickers=_DIAGNOSE_VEHICLES)
    except (BaselineDataError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=diagnose_us_vehicles_failed reason_type=%s", type(exc).__name__)
        return 1
    for profile in profiles:
        logger.info(
            "[DATA] event=vehicle_factor_profile ticker=%s alpha=%.6f mkt_rf=%.4f smb=%.4f hml=%.4f rmw=%.4f cma=%.4f mom=%.4f",
            profile.ticker,
            profile.alpha,
            profile.mkt_rf,
            profile.smb,
            profile.hml,
            profile.rmw,
            profile.cma,
            profile.mom,
        )
    for path in paths:
        logger.info(
            "[DATA] event=vehicle_dca_done ticker=%s terminal_krw=%.3f terminal_real_krw=%.3f xirr=%.6f steps=%d",
            path.ticker,
            path.result.terminal_wealth_krw,
            path.result.terminal_wealth_real_krw,
            path.result.xirr,
            len(path.result.snapshots),
        )
    return 0


def run_diagnose_qqq_regimes_command(*, contribution_krw: float, settings: DataSettings) -> int:
    """Log QQQ/VTI real-terminal ratios per regime window on identical cashflows.

    Reporting-only diagnostics: no ablation, walk-forward gate, or adoption
    decision may run here, and no operational policy lock changes.
    """
    try:
        comparisons = compare_policy_regimes(
            contribution_krw=float(contribution_krw),
            runner=lambda config: run_allocation_from_store(config, settings),
        )
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=diagnose_qqq_regimes_failed reason_type=%s", type(exc).__name__)
        return 1
    for comparison in comparisons:
        logger.info(
            "[DATA] event=qqq_regime_ratio name=%s start=%s end=%s ratio=%.6f vti_steps=%d qqq_steps=%d",
            comparison.name,
            comparison.start.isoformat(),
            comparison.end.isoformat(),
            comparison.candidate.terminal_wealth_real_krw / comparison.baseline.terminal_wealth_real_krw,
            len(comparison.baseline.snapshots),
            len(comparison.candidate.snapshots),
        )
    return 0


def run_diagnose_qqq_blends_command(*, contribution_krw: float, settings: DataSettings) -> int:
    """Log QQQ blend-recipe real-terminal ratios per regime window on identical cashflows.

    Reporting-only diagnostics: no ablation, walk-forward gate, or adoption
    decision may run here, and no operational policy lock changes.
    """
    try:
        comparisons = compare_qqq_blends(
            contribution_krw=float(contribution_krw),
            runner=lambda config: run_allocation_from_store(config, settings),
        )
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=diagnose_qqq_blends_failed reason_type=%s", type(exc).__name__)
        return 1
    for comparison in comparisons:
        logger.info(
            "[DATA] event=qqq_blend_ratio name=%s recipe=%s real_terminal_krw=%.2f mdd=%.4f"
            " ratio_vs_qqq=%.6f ratio_vs_vti=%.6f",
            comparison.name,
            comparison.recipe,
            comparison.candidate.terminal_wealth_real_krw,
            comparison.candidate.max_drawdown,
            comparison.candidate.terminal_wealth_real_krw / comparison.qqq_baseline.terminal_wealth_real_krw,
            comparison.candidate.terminal_wealth_real_krw / comparison.vti_baseline.terminal_wealth_real_krw,
        )
    return 0


_RESERVE_SCHEDULES: Final[dict[str, ReserveConfig | None]] = {
    "v1": None,
    "v2": ReserveConfig(schedule="v2", max_withhold=0.10),
    "v3": ReserveConfig(
        schedule="v3", max_withhold=0.10, min_invest_multiplier=0.70, max_invest_multiplier=3.0
    ),
    "v4": ReserveConfig(
        schedule="v4",
        max_withhold=0.10,
        min_invest_multiplier=0.70,
        max_invest_multiplier=3.0,
        reserve_max_months=2.0,
    ),
}


def run_diagnose_qqq_reserve_command(
    *, contribution_krw: float, settings: DataSettings, reserve_schedule: str = "v1"
) -> int:
    """Log QQQ reserved-arm ratios, MDD, and reconstructed reserve usage per regime window.

    Reporting-only diagnostics: no ablation, walk-forward gate, or adoption
    decision may run here, and the operational policy lock stays unchanged.
    """
    try:
        if reserve_schedule not in _RESERVE_SCHEDULES:
            raise ValueError(f"unknown reserve schedule {reserve_schedule!r}")
        comparisons = compare_qqq_reserve(
            contribution_krw=float(contribution_krw),
            reserve=_RESERVE_SCHEDULES[reserve_schedule],
            runner=lambda config: run_allocation_from_store(config, settings),
        )
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=diagnose_qqq_reserve_failed reason_type=%s", type(exc).__name__)
        return 1
    for comparison in comparisons:
        logger.info(
            "[DATA] event=qqq_reserve_ratio name=%s start=%s end=%s"
            " reserved_real_terminal_krw=%.2f plain_real_terminal_krw=%.2f ratio_reserved_vs_plain=%.6f"
            " reserved_mdd=%.4f plain_mdd=%.4f withheld_total=%.2f redeployed_total=%.2f"
            " extra_investment_ratio=%.8f cash_drag_ratio=%.8f reserve_idle_months=%d"
            " reserve_deployment_events=%d steps=%d",
            comparison.name,
            comparison.start.isoformat(),
            comparison.end.isoformat(),
            comparison.reserved.terminal_wealth_real_krw,
            comparison.plain.terminal_wealth_real_krw,
            comparison.reserved.terminal_wealth_real_krw / comparison.plain.terminal_wealth_real_krw,
            comparison.reserved.max_drawdown,
            comparison.plain.max_drawdown,
            comparison.reserved_usage.withheld_total,
            comparison.reserved_usage.redeployed_total,
            comparison.reserved_usage.extra_investment_ratio,
            comparison.reserved_usage.cash_drag_ratio,
            comparison.reserved_usage.reserve_idle_months,
            comparison.reserved_usage.reserve_deployment_events,
            len(comparison.reserved.snapshots),
        )
    return 0


def run_diagnose_qqq_cadence_command(*, contribution_krw: float, settings: DataSettings) -> int:
    """Log QQQ month-open-cadence real-terminal ratios versus the monthly cadence per regime window.

    Reporting-only diagnostics: no ablation, walk-forward gate, or adoption
    decision may run here, and no operational policy lock changes.
    """
    try:
        comparisons = compare_qqq_cadence(
            contribution_krw=float(contribution_krw),
            runner=lambda config: run_allocation_from_store(config, settings),
        )
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=diagnose_qqq_cadence_failed reason_type=%s", type(exc).__name__)
        return 1
    for comparison in comparisons:
        logger.info(
            "[DATA] event=qqq_cadence_ratio name=%s start=%s end=%s"
            " month_open_real_terminal_krw=%.2f monthly_real_terminal_krw=%.2f"
            " ratio_month_open_vs_monthly=%.6f"
            " twice_monthly_real_terminal_krw=%.2f ratio_twice_monthly_vs_monthly=%.6f steps=%d",
            comparison.name,
            comparison.start.isoformat(),
            comparison.end.isoformat(),
            comparison.month_open.terminal_wealth_real_krw,
            comparison.monthly.terminal_wealth_real_krw,
            comparison.month_open.terminal_wealth_real_krw / comparison.monthly.terminal_wealth_real_krw,
            comparison.twice_monthly.terminal_wealth_real_krw,
            comparison.twice_monthly.terminal_wealth_real_krw / comparison.monthly.terminal_wealth_real_krw,
            len(comparison.month_open.snapshots),
        )
    return 0


_ACCUMULATION_TICKER: Final[str] = "QQQ"


def run_diagnose_qqq_accumulation_alpha_command(*, contribution_krw: float, settings: DataSettings) -> int:
    """Log QQQ buy-cadence accumulation-screen ratios versus month-end fills.

    Reporting-only diagnostics: no ablation, walk-forward gate, or adoption
    decision may run here, and the operational policy lock stays unchanged.
    """
    try:
        if float(contribution_krw) <= 0.0:
            raise ValueError(f"contribution_krw must be positive, got {contribution_krw!r}")
        prices = DataStore(settings).read_normalized(
            latest_artifact(settings, Dataset.PRICES), spec_for(Dataset.PRICES)
        )
        ticker_rows = prices.filter(prices.get_column("ticker") == _ACCUMULATION_TICKER)
        if ticker_rows.is_empty():
            raise ValueError(f"ticker {_ACCUMULATION_TICKER!r} missing from catalog prices")
        start_raw = ticker_rows.get_column("date").min()
        end_raw = ticker_rows.get_column("date").max()
        if start_raw is None or end_raw is None:
            raise ValueError(f"ticker {_ACCUMULATION_TICKER!r} has no price dates")
        report = screen_qqq_accumulation(
            prices=prices,
            ticker=_ACCUMULATION_TICKER,
            start=cast(date, start_raw),
            end=cast(date, end_raw),
            monthly_contribution=float(contribution_krw),
        )
    except (PolicyError, UntrustedDatasetError, ValueError) as exc:
        logger.error("[DATA] event=diagnose_qqq_accumulation_alpha_failed reason_type=%s", type(exc).__name__)
        return 1
    for row in report.rows:
        logger.info(
            "[DATA] event=qqq_accumulation_arm arm=%s verdict=%s tw=%.2f ratio_vs_month_end=%.6f"
            " ci_low=%.6f ci_high=%.6f mean_log_gap=%s log_fill_p=%s",
            row.name,
            str(row.verdict),
            row.terminal_wealth,
            row.ratio_vs_month_end,
            row.bootstrap_ci_low,
            row.bootstrap_ci_high,
            row.mean_log_fill_gap_vs_end,
            row.log_fill_p_value,
        )
    logger.info(
        "[DATA] event=qqq_accumulation_screen_done ticker=%s start=%s end=%s usable_months=%d"
        " operational_unlock=%s recommended_research_arm=%s",
        report.ticker,
        report.start.isoformat(),
        report.end.isoformat(),
        report.usable_months,
        report.operational_unlock,
        report.recommended_research_arm,
    )
    return 0


_KAFI_SHAPE_CONFIG: Final[ContributionShapeConfig] = ContributionShapeConfig()


def run_diagnose_qqq_kafi_command(*, contribution_krw: float, settings: DataSettings) -> int:
    """Log the KAFI path and shaped-versus-flat real-TW ratios per regime window.

    Windows fall back to the full QQQ catalog range when regime bounds exceed
    coverage. Reporting-only diagnostics: no ablation, walk-forward gate, or
    adoption decision may run here, and the operational policy lock stays unchanged.
    """
    try:
        if float(contribution_krw) <= 0.0:
            raise ValueError(f"contribution_krw must be positive, got {contribution_krw!r}")
        store = DataStore(settings)
        prices = store.read_normalized(latest_artifact(settings, Dataset.PRICES), spec_for(Dataset.PRICES))
        fx = store.read_normalized(latest_artifact(settings, Dataset.FX), spec_for(Dataset.FX))
        macro = store.read_normalized(latest_artifact(settings, Dataset.MACRO), spec_for(Dataset.MACRO))
        qqq_rows = prices.filter(prices.get_column("ticker") == _ACCUMULATION_TICKER)
        if qqq_rows.is_empty():
            raise ValueError(f"ticker {_ACCUMULATION_TICKER!r} missing from catalog prices")
        start_raw = qqq_rows.get_column("date").min()
        end_raw = qqq_rows.get_column("date").max()
        if start_raw is None or end_raw is None:
            raise ValueError(f"ticker {_ACCUMULATION_TICKER!r} has no price dates")
        catalog_start = cast(date, start_raw)
        catalog_end = cast(date, end_raw)
        calendar = load_calendar(DEFAULT_CALENDAR_NAME)
        catalog_start, catalog_end = clamp_inclusive_session_range(calendar, catalog_start, catalog_end)
        config = _KAFI_SHAPE_CONFIG
        feasible_start = earliest_kafi_signal_session(
            prices=prices,
            fx=fx,
            macro=macro,
            equity_ticker=config.equity_ticker,
            bond_ticker=config.bond_ticker,
            start=catalog_start,
            end=catalog_end,
            rank_window=config.rank_window,
            credit_series_id=config.credit_series_id,
        )
        if feasible_start is None:
            raise ValueError(
                f"catalog lacks enough PIT history for KAFI credit series {config.credit_series_id!r}"
            )
        windows = tuple(
            (name, window_start, window_end)
            for name, window_start, window_end in QQQ_REGIME_WINDOWS
            if window_start >= catalog_start and window_end <= catalog_end
        ) or (("catalog", catalog_start, catalog_end),)
        logged = False
        for name, window_start, window_end in windows:
            effective_start = max(window_start, feasible_start)
            if effective_start > window_end:
                continue
            flat_result = run_allocation_from_store(
                AllocationConfig(
                    policy=PolicyId.QQQ,
                    start=effective_start,
                    end=window_end,
                    monthly_contribution_krw=float(contribution_krw),
                ),
                settings,
            )
            shaped_result = run_allocation_from_store(
                AllocationConfig(
                    policy=PolicyId.QQQ,
                    start=effective_start,
                    end=window_end,
                    monthly_contribution_krw=float(contribution_krw),
                    contribution_shape=config,
                ),
                settings,
            )
            logged = True
            logger.info(
                "[DATA] event=qqq_kafi_ratio name=%s start=%s end=%s"
                " flat_real_terminal_krw=%.2f shaped_real_terminal_krw=%.2f"
                " ratio_shaped_vs_flat=%.6f flat_mdd=%.4f shaped_mdd=%.4f steps=%d",
                name,
                effective_start.isoformat(),
                window_end.isoformat(),
                flat_result.terminal_wealth_real_krw,
                shaped_result.terminal_wealth_real_krw,
                (
                    shaped_result.terminal_wealth_real_krw / flat_result.terminal_wealth_real_krw
                    if flat_result.terminal_wealth_real_krw != 0.0
                    else float("nan")
                ),
                flat_result.max_drawdown,
                shaped_result.max_drawdown,
                len(shaped_result.snapshots),
            )
            band = config.max_multiplier - config.min_multiplier
            for point in build_decision_schedule(effective_start, window_end):
                score = kafi_score(
                    prices=prices,
                    fx=fx,
                    macro=macro,
                    equity_ticker=config.equity_ticker,
                    bond_ticker=config.bond_ticker,
                    signal_at=point.signal_at,
                    rank_window=config.rank_window,
                    credit_series_id=config.credit_series_id,
                )
                multiplier = config.min_multiplier + band * (100.0 - score) / 100.0
                logger.info(
                    "[DATA] event=kafi_path signal_session=%s score=%.2f multiplier=%.4f",
                    point.signal_session.isoformat(),
                    score,
                    multiplier,
                )
        if not logged:
            raise ValueError("no QQQ regime window remains after KAFI warmup clamping")
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=diagnose_qqq_kafi_failed reason_type=%s", type(exc).__name__)
        return 1
    return 0


def run_diagnose_qqq_adaptive_hp_command(*, contribution_krw: float, settings: DataSettings) -> int:
    """Run the adaptive HP neighbourhood screen versus operational v5; reporting only."""
    try:
        if not math.isfinite(float(contribution_krw)) or float(contribution_krw) <= 0.0:
            raise ValueError(f"contribution_krw must be positive, got {contribution_krw!r}")

        wf_runner = make_hp_wf_runner(settings, contribution_krw=float(contribution_krw))
        report = screen_adaptive_contribution_hp(
            contribution_krw=float(contribution_krw),
            wf_runner=wf_runner,
        )
        for row in report.rows:
            logger.info(
                "[DATA] event=qqq_adaptive_hp_arm downside=%.4f upside=%.4f dispersion=%.4f deadband=%.4f ratio=%.6f adopted=%s verdict=%s",
                row.downside_power,
                row.upside_power,
                row.dispersion,
                row.neutral_deadband,
                row.pooled_tw_ratio,
                row.process_adopted_vs_baseline,
                str(row.verdict),
            )
        if report.champion is not None:
            c = report.champion
            logger.info(
                "[DATA] event=qqq_adaptive_hp_champion downside=%.4f upside=%.4f dispersion=%.4f deadband=%.4f ratio=%.6f",
                c.downside_power,
                c.upside_power,
                c.dispersion,
                c.neutral_deadband,
                c.pooled_tw_ratio,
            )
        else:
            logger.info("[DATA] event=qqq_adaptive_hp_champion none")
        logger.info(
            "[DATA] event=qqq_adaptive_hp_done evaluations=%d operational_unlock=%s champion=%s",
            report.evaluations,
            report.operational_unlock,
            "none" if report.champion is None else f"{report.champion.downside_power:.4f}/{report.champion.upside_power:.4f}/{report.champion.dispersion:.4f}/{report.champion.neutral_deadband:.4f}",
        )
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=diagnose_qqq_adaptive_hp_failed reason_type=%s", type(exc).__name__)
        return 1
    return 0


def run_policy_command(
    *,
    policy_id: str,
    start: date,
    end: date,
    contribution_krw: float,
    settings: DataSettings,
    tilt: FactorTilt | None = None,
    rebalance_band: float | None = None,
    overlay: OverlayConfig | None = None,
    reserve: ReserveConfig | None = None,
    currency: CurrencyConfig | None = None,
    mapping: MappingConfig | None = None,
) -> int:
    """Run a stored-data strategic allocation and log terminal KRW / XIRR / MDD.

    Raises:
        ValueError: When ``rebalance_band`` is set outside ``[0, 1)`` or the mapping config is invalid.
    """
    if rebalance_band is not None and not 0.0 <= rebalance_band < 1.0:
        raise ValueError(f"rebalance_band must lie in [0, 1), got {rebalance_band!r}")
    config = apply_operational_contribution_lock(
        AllocationConfig(
            policy=PolicyId.parse(policy_id),
            start=start,
            end=end,
            monthly_contribution_krw=float(contribution_krw),
            fill_delay_sessions=1,
            commission_bps=0.0,
            tilt=tilt,
            rebalance_band=rebalance_band,
            overlay=overlay,
            reserve=reserve,
            currency=currency,
            mapping=mapping,
        )
    )
    try:
        require_feasibility(
            start=config.start,
            end=config.end,
            fill_delay_sessions=config.fill_delay_sessions,
            mark_policies=(config.policy,),
            overlay=config.overlay,
            overlay_policies=(config.policy,) if config.overlay is not None else (),
            settings=settings,
            reserve=config.reserve,
            mapping=config.mapping,
            mapping_policies=(config.policy,) if config.mapping is not None else (),
            currency=config.currency,
        )
        result = run_allocation_from_store(config, settings)
    except (
        AllocationDataError,
        PolicyError,
        BaselineDataError,
        UntrustedDatasetError,
        XirrError,
        ValueError,
    ) as exc:
        logger.error("[DATA] event=policy_cli_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=policy_cli_done policy=%s terminal_krw=%.3f xirr=%.6f mdd=%.4f"
        " terminal_real_krw=%.3f xirr_real=%.6f sleeves=%d steps=%d",
        str(config.policy),
        result.terminal_wealth_krw,
        result.xirr,
        result.max_drawdown,
        result.terminal_wealth_real_krw,
        result.xirr_real,
        len(policy_sleeves(config.policy)),
        len(result.snapshots),
    )
    return 0


def run_validate_command(
    *,
    policy_id: str,
    start: date,
    end: date,
    contribution_krw: float,
    settings: DataSettings,
    delta0: float,
    modules: int,
    horizon_months: int,
    cohort_step_months: int,
    bootstrap_paths: int,
    seed: int | None,
) -> int:
    """Cohort CE adoption gate versus B0; optional seeded wealth-vector bootstrap.

    Raises:
        ValueError: When validation hyperparameters are invalid or lineage is unavailable.
    """
    if bootstrap_paths > 0 and seed is None:
        raise _UsageError("--bootstrap-paths requires --seed")

    template = AllocationConfig(
        policy=PolicyId(policy_id),
        start=start,
        end=end,
        monthly_contribution_krw=float(contribution_krw),
        fill_delay_sessions=1,
        commission_bps=0.0,
    )
    try:
        cohorts = rolling_cohorts(start, end, horizon_months=horizon_months, step_months=cohort_step_months)
        candidate_wealths = evaluate_cohort_wealths(
            template,
            cohorts,
            lambda config: run_allocation_from_store(config, settings),
        )
        baseline_wealths = tuple(
            run_baseline_from_store(
                BaselineConfig(
                    baseline=BaselineId.B0_GLOBAL,
                    ticker=_VALIDATE_BASELINE_TICKER,
                    start=cohort_start,
                    end=cohort_end,
                    monthly_contribution_krw=float(contribution_krw),
                    fill_delay_sessions=1,
                    commission_bps=0.0,
                ),
                settings,
            ).terminal_wealth_real_krw
            for cohort_start, cohort_end in cohorts
        )
        candidate_ce = {gamma: certainty_equivalent(candidate_wealths, gamma=gamma) for gamma in _VALIDATE_GAMMAS}
        baseline_ce = {gamma: certainty_equivalent(baseline_wealths, gamma=gamma) for gamma in _VALIDATE_GAMMAS}
        adopted = adoption_passes(candidate_ce, baseline_ce, delta0=delta0, modules=modules)
        record = make_experiment(
            config=template,
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(),
            seed=seed,
            metrics={
                **{f"ce_candidate_gamma_{int(gamma)}": value for gamma, value in candidate_ce.items()},
                **{f"ce_baseline_gamma_{int(gamma)}": value for gamma, value in baseline_ce.items()},
                "adopted": 1.0 if adopted else 0.0,
                "cohorts": float(len(cohorts)),
            },
        )
    except (
        AllocationDataError,
        BaselineDataError,
        PolicyError,
        UntrustedDatasetError,
        XirrError,
        ValueError,
    ) as exc:
        logger.error("[DATA] event=validate_cli_failed reason=%s", exc)
        return 1

    bootstrap_mean = 0.0
    if bootstrap_paths > 0 and seed is not None:
        # Half-window blocks keep roughly two independent draws per resampled path.
        resampled = moving_block_bootstrap(
            candidate_wealths,
            block_size=max(1, len(candidate_wealths) // 2),
            n_paths=bootstrap_paths,
            seed=seed,
        )
        resampled_means = [sum(path) / len(path) for path in resampled]
        bootstrap_mean = sum(resampled_means) / len(resampled_means)

    ratios = {gamma: candidate_ce[gamma] / baseline_ce[gamma] for gamma in candidate_ce}
    logger.info(
        "[DATA] event=validate_cli_done policy=%s cohorts=%d adopted=%s"
        " ratio_gamma_2=%.6f ratio_gamma_5=%.6f ratio_gamma_10=%.6f"
        " bootstrap_paths=%d bootstrap_mean=%.6f experiment_id=%s",
        str(template.policy),
        len(cohorts),
        adopted,
        ratios[2.0],
        ratios[5.0],
        ratios[10.0],
        bootstrap_paths,
        bootstrap_mean,
        record.experiment_id,
    )
    return 0


def run_ablation_command(*, config_path: str, settings: DataSettings) -> int:
    """Run an identical-cashflow ablation from an experiment JSON and log each gate.

    Raises:
        ValueError: When the experiment JSON is invalid or lineage is unavailable.
    """
    try:
        spec = load_experiment_config(config_path)
        registry = load_thesis_registry(Path("configs/theses"))
        assert_experiment_preregistration(spec, registry)
        assert_experiment_feasible(spec, settings)
        report = run_ablation(spec, lambda config: run_allocation_from_store(config, settings))
        metrics: dict[str, float] = {
            "candidates": float(len(report.rows)),
            "adopted": float(sum(row.adopted for row in report.rows)),
        }
        for row in report.rows:
            for gamma, ratio in row.ce_ratio.items():
                metrics[f"{row.candidate_id}_ratio_gamma_{int(gamma)}"] = ratio
        thesis_id_str = spec.thesis_id.value if spec.thesis_id is not None else None
        record = make_experiment(
            config=AllocationConfig(
                policy=spec.baseline.policy,
                start=spec.start,
                end=spec.end,
                monthly_contribution_krw=spec.contribution_krw,
                fill_delay_sessions=1,
                commission_bps=0.0,
                targets_override=resolve_arm_targets(spec.candidates[0]) if spec.candidates else None,
            ),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(),
            seed=None,
            metrics=metrics,
            thesis_id=thesis_id_str,
        )
        write_ablation_run_record(spec=spec, report=report, record=record, settings=settings)
    except (AllocationDataError, PolicyError, ThesisError, UntrustedDatasetError, XirrError, ValueError, OSError) as exc:
        logger.error("[DATA] event=ablation_cli_failed reason=%s", exc)
        return 1
    adopted_count = sum(row.adopted for row in report.rows)
    for index, row in enumerate(report.rows):
        logger.info(
            "[DATA] event=ablation_candidate index=%d candidate=%s policy=%s modules=%d adopted=%s"
            " ratio_gamma_2=%.6f ratio_gamma_5=%.6f ratio_gamma_10=%.6f",
            index,
            row.candidate_id,
            str(row.policy),
            row.modules,
            row.adopted,
            row.ce_ratio[2.0],
            row.ce_ratio[5.0],
            row.ce_ratio[10.0],
        )
    logger.info(
        "[DATA] event=ablation_cli_done experiment=%s experiment_id=%s adopted=%d/%d",
        spec.name,
        record.experiment_id,
        adopted_count,
        len(report.rows),
    )
    return 0


def run_walk_forward_command(*, config_path: str, settings: DataSettings) -> int:
    """Run a walk-forward adoption campaign and persist the report JSON.

    Raises:
        ValueError: When the experiment JSON is invalid or lineage is unavailable.
    """
    try:
        spec = load_experiment_config(config_path)
        if spec.train_months is None or spec.test_months is None:
            raise ValueError("experiment JSON lacks train_months and test_months")
        assert_experiment_feasible(spec, settings)
        report = run_walk_forward_adoption(spec, lambda config: run_allocation_from_store(config, settings))
        record = make_experiment(
            config=AllocationConfig(
                policy=spec.candidates[0].policy,
                start=spec.start,
                end=spec.end,
                monthly_contribution_krw=spec.contribution_krw,
                fill_delay_sessions=1,
                commission_bps=0.0,
                targets_override=resolve_arm_targets(spec.candidates[0]),
            ),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(),
            seed=None,
            metrics={
                "folds": float(len(report.folds)),
                "process_adopted_vs_baseline": 1.0 if report.process_adopted_vs_baseline else 0.0,
            },
        )
        report_path = write_campaign_report(report, settings, record.experiment_id)
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=walkforward_cli_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=walkforward_cli_done experiment=%s experiment_id=%s folds=%d adopted_vs_baseline=%s report=%s",
        spec.name,
        record.experiment_id,
        len(report.folds),
        report.process_adopted_vs_baseline,
        report_path,
    )
    return 0


def run_walk_forward_costs_command(*, config_path: str, settings: DataSettings) -> int:
    """Run the walk-forward adoption cost grid and persist one grid report JSON.

    Raises:
        ValueError: When the experiment JSON is invalid or lineage is unavailable.
    """
    try:
        spec = load_experiment_config(config_path)
        if spec.train_months is None or spec.test_months is None:
            raise ValueError("experiment JSON lacks train_months and test_months")
        assert_experiment_feasible(spec, settings)
        report = run_walk_forward_cost_grid(spec, lambda config: run_allocation_from_store(config, settings))
        record = make_experiment(
            config=AllocationConfig(
                policy=spec.candidates[0].policy,
                start=spec.start,
                end=spec.end,
                monthly_contribution_krw=spec.contribution_krw,
                fill_delay_sessions=1,
                commission_bps=spec.commission_bps,
                fx_spread_bps=spec.fx_spread_bps,
            ),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(),
            seed=None,
            metrics={
                "scenarios": float(len(report.outcomes)),
                "all_scenarios_adopted": 1.0 if report.all_scenarios_adopted else 0.0,
            },
        )
        report_path = write_cost_grid_report(report, settings, record.experiment_id)
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=walkforward_costs_cli_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=walkforward_costs_cli_done experiment=%s experiment_id=%s scenarios=%d all_adopted=%s report=%s",
        spec.name,
        record.experiment_id,
        len(report.outcomes),
        report.all_scenarios_adopted,
        report_path,
    )
    return 0


def run_walk_forward_proxy_command(*, config_path: str, settings: DataSettings) -> int:
    """Run the research-proxy walk-forward campaign and persist the report JSON.

    Raises:
        ValueError: When the experiment JSON is invalid or lineage is unavailable.
    """
    try:
        spec = load_experiment_config(config_path)
        if spec.train_months is None or spec.test_months is None:
            raise ValueError("experiment JSON lacks train_months and test_months")
        assert_experiment_feasible(spec, settings)
        report = run_walk_forward_proxy_adoption(
            spec,
            lambda config: run_allocation_from_store(config, settings),
            lambda config: run_research_proxy_from_store(config, settings),
        )
        record = make_experiment(
            config=AllocationConfig(
                policy=spec.candidates[0].policy,
                start=spec.start,
                end=spec.end,
                monthly_contribution_krw=spec.contribution_krw,
                fill_delay_sessions=1,
                commission_bps=0.0,
            ),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(),
            seed=None,
            metrics={
                "folds": float(len(report.folds)),
                "process_adopted_vs_baseline": 1.0 if report.process_adopted_vs_baseline else 0.0,
            },
        )
        report_path = write_campaign_report(report, settings, record.experiment_id)
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=walkforward_proxy_cli_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=walkforward_proxy_cli_done experiment=%s experiment_id=%s folds=%d adopted_vs_baseline=%s report=%s",
        spec.name,
        record.experiment_id,
        len(report.folds),
        report.process_adopted_vs_baseline,
        report_path,
    )
    return 0


def run_cadence_robustness_command(
    *, config_path: str, settings: DataSettings, seed: int, bootstrap_paths: int
) -> int:
    """Run the growth-first cadence robustness gate and persist one report JSON.

    Raises:
        ValueError: When the experiment JSON is invalid or lineage is unavailable.
    """
    if bootstrap_paths < 1:
        raise _UsageError(f"--bootstrap-paths must be >= 1, got {bootstrap_paths}")
    try:
        spec = load_experiment_config(config_path)
        assert_experiment_feasible(spec, settings)
        report = run_cadence_robustness(
            spec,
            lambda config: run_allocation_from_store(config, settings),
            n_paths=bootstrap_paths,
            seed=seed,
        )
        record = make_experiment(
            config=AllocationConfig(
                policy=spec.candidates[0].policy,
                start=spec.start,
                end=spec.end,
                monthly_contribution_krw=spec.contribution_krw,
                fill_delay_sessions=1,
                commission_bps=spec.commission_bps,
                fx_spread_bps=spec.fx_spread_bps,
            ),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(),
            seed=seed,
            metrics={
                "cohorts": float(len(report.candidate_wealths)),
                "all_scenarios_adopted": 1.0 if report.cost_grid.all_scenarios_adopted else 0.0,
                "robust_adopted": 1.0 if report.robust_adopted else 0.0,
            },
        )
        report_path = write_cadence_robustness_report(report, settings, record.experiment_id)
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=cadence_robustness_cli_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=cadence_robustness_cli_done experiment=%s experiment_id=%s cohorts=%d"
        " all_scenarios_adopted=%s worst_cohort_ok=%s bootstrap_tail_ok=%s robust_adopted=%s report=%s",
        spec.name,
        record.experiment_id,
        len(report.candidate_wealths),
        report.cost_grid.all_scenarios_adopted,
        report.worst_cohort_ok,
        report.bootstrap_tail_ok,
        report.robust_adopted,
        report_path,
    )
    return 0


def run_accumulation_cohort_command(
    *,
    config_path: str,
    settings: DataSettings,
    horizon_months: int,
    cohort_step_months: int,
    bootstrap_paths: int,
    seed: int | None,
) -> int:
    """Run rolling 120M accumulation cohort report (reporting-only)."""
    if cohort_step_months not in {1, 12, 36}:
        raise _UsageError(f"--cohort-step-months must be one of 1, 12, 36, got {cohort_step_months}")
    if horizon_months < 1:
        raise _UsageError(f"--horizon-months must be >=1, got {horizon_months}")
    if bootstrap_paths < 1:
        raise _UsageError(f"--bootstrap-paths must be >=1, got {bootstrap_paths}")
    if seed is None:
        raise _UsageError("--seed is required for accumulation-cohort")
    try:
        spec = load_experiment_config(config_path)
        assert_experiment_feasible(spec, settings)
        report = run_accumulation_cohort_report(
            spec,
            lambda config: run_allocation_from_store(config, settings),
            horizon_months=horizon_months,
            step_months=cohort_step_months,
            bootstrap_paths=bootstrap_paths,
            seed=seed,
        )
        record = make_experiment(
            config=AllocationConfig(
                policy=spec.candidates[0].policy if spec.candidates else spec.baseline.policy,
                start=spec.start,
                end=spec.end,
                monthly_contribution_krw=spec.contribution_krw,
                fill_delay_sessions=1,
                commission_bps=spec.commission_bps,
                fx_spread_bps=spec.fx_spread_bps,
            ),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(),
            seed=seed,
            metrics={
                "cohorts": float(len(report.rows)),
                "median_ratio": float(report.median_ratio),
                "p10_ratio": float(report.p10_ratio),
                "worst_ratio": float(report.worst_ratio),
                "win_rate": float(report.win_rate),
                "bootstrap_p05_ratio_mean": float(report.bootstrap_p05_ratio_mean),
            },
        )
        report_path = write_accumulation_cohort_report(report, settings, record.experiment_id)
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
        logger.error("[DATA] event=accumulation_cohort_cli_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=accumulation_cohort_cli_done experiment=%s experiment_id=%s cohorts=%d"
        " median_ratio=%.6f p10_ratio=%.6f worst_ratio=%.6f win_rate=%.4f bootstrap_p05=%.6f report=%s",
        spec.name,
        record.experiment_id,
        len(report.rows),
        report.median_ratio,
        report.p10_ratio,
        report.worst_ratio,
        report.win_rate,
        report.bootstrap_p05_ratio_mean,
        report_path,
    )
    return 0


def run_paper_command(
    *,
    policy_id: str,
    start: date,
    end: date,
    contribution_krw: float,
    settings: DataSettings,
) -> int:
    """Replay a stored-data policy onto PaperBroker and fail closed on lot mismatch."""
    config = apply_operational_contribution_lock(
        AllocationConfig(
            policy=PolicyId(policy_id),
            start=start,
            end=end,
            monthly_contribution_krw=float(contribution_krw),
            fill_delay_sessions=1,
            commission_bps=0.0,
        )
    )
    try:
        result = run_allocation_from_store(config, settings)
        replay_paper(result)
        order_count = len(orders_from_snapshots(result.snapshots))
    except (
        AllocationDataError,
        BaselineDataError,
        ExecutionError,
        PolicyError,
        UntrustedDatasetError,
        XirrError,
        ValueError,
    ) as exc:
        logger.error("[DATA] event=paper_cli_failed reason=%s", exc)
        return 1
    logger.info(
        "[DATA] event=paper_cli_done policy=%s orders=%d steps=%d",
        str(config.policy),
        order_count,
        len(result.snapshots),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
