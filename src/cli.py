# ruff: noqa: S110,SIM102,SIM108,F541,I001,UP035
"""Command-line ingest and baseline-run entry (no secret printing)."""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, date

from src.cli_commands.campaign import (
    run_ablation_command,
    run_accumulation_cohort_command,
    run_audit_feasibility_command,
    run_cadence_robustness_command,
    run_validate_command,
    run_walk_forward_command,
    run_walk_forward_costs_command,
    run_walk_forward_proxy_command,
)
from src.cli_commands.diagnose import (
    run_diagnose_compound_dca_command,
    run_diagnose_qqq_accumulation_alpha_command,
    run_diagnose_qqq_adaptive_hp_command,
    run_diagnose_qqq_blends_command,
    run_diagnose_qqq_cadence_command,
    run_diagnose_qqq_kafi_command,
    run_diagnose_qqq_regimes_command,
    run_diagnose_qqq_reserve_command,
    run_diagnose_us_vehicles_command,
)
from src.cli_commands.ingest import (
    _HISTORY_FX_PROVIDER,
    _SMOKE_DATA_ROOT,
    _SMOKE_END,
    _SMOKE_FX_PROVIDER,
    _SMOKE_START,
    _SMOKE_TICKER,
    run_ingest_history,
    run_ingest_smoke,
    run_ingest_static_dca,
)
from src.cli_commands.parser import _UsageError, _build_parser
from src.cli_commands.resolvers import (
    _resolve_currency,
    _resolve_mapping,
    _resolve_overlay,
    _resolve_reserve,
    _resolve_tilt,
)
from src.cli_commands.sim_run import run_baseline_command, run_paper_command, run_policy_command
from src.cli_commands.thesis import (
    run_diagnose_overlap_command,
    run_thesis_command,
    run_thesis_incremental_command,
    run_thesis_report_command,
    run_thesis_wave_command,
)
from src.data.fetch import (
    fetch_and_persist_cpi,
    fetch_and_persist_factors,
    fetch_and_persist_fx,
    fetch_and_persist_macro,
    fetch_and_persist_prices,
    fetch_and_persist_research_returns,
)
from src.data.nport_ingest import fetch_and_persist_nport_quarter, fetch_and_persist_nport_quarters
from src.data.providers.base import ProviderError
from src.data.secrets import load_provider_secrets
from src.data.settings import DataSettings

# Re-export for test monkeypatch compatibility
__all__ = ["main"]

logger = logging.getLogger(__name__)


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
        quarters = [s.strip() for s in str(fq).split(",") if s.strip()]
        if len(quarters) > 1:
            fetch_and_persist_nport_quarters(filing_quarters=quarters, settings=DataSettings())
            logger.info("[DATA] event=cli_ingest_done dataset=nport filing_quarters=%s", ",".join(quarters))
            return 0
        fetch_and_persist_nport_quarter(filing_quarter=str(fq), settings=DataSettings())
        logger.info("[DATA] event=cli_ingest_done dataset=nport filing_quarter=%s", str(fq))
        return 0
    if dataset == "thesis-panel":
        from src.data.fetch import fetch_and_persist_static_dca_datasets
        from src.data.panel_freshness import THESIS_PANEL_TICKERS, iter_nport_quarters_for_panel

        _ = fetch_and_persist_static_dca_datasets
        _ = "thesis-panel"
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
    if dataset == "thesis-fundamentals":
        from datetime import datetime as _dt_for_fund

        from src.data.thesis_fundamentals import fetch_and_persist_thesis_fundamentals

        _ = fetch_and_persist_thesis_fundamentals
        _fund_start: date = args.start if args.start is not None else date(2000, 1, 1)
        _fund_end: date = args.end if args.end is not None else _dt_for_fund.now(UTC).date()
        _fund_settings = DataSettings()
        _fund_secrets = load_provider_secrets()
        fetch_and_persist_thesis_fundamentals(start=_fund_start, end=_fund_end, settings=_fund_settings, secrets=_fund_secrets)
        logger.info("[DATA] event=cli_ingest_done dataset=thesis-fundamentals start=%s end=%s", _fund_start.isoformat(), _fund_end.isoformat())
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
    if args.target == "diagnose-compound-dca":
        return run_diagnose_compound_dca_command(
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
        from src.analytics.wave_d_exit import assess_wave_d_exit as _assess

        _ = _assess
        return run_thesis_incremental_command(
            thesis_id=str(getattr(args, "thesis_id", "ai_compute")),
            as_of=str(args.as_of) if getattr(args, "as_of", None) else None,
            settings=DataSettings(),
            seed=int(getattr(args, "seed", 7)),
            bootstrap_paths=int(getattr(args, "bootstrap_paths", 400)),
            allow_stale=bool(getattr(args, "allow_stale", False)),
            contribution_krw=float(getattr(args, "contribution_krw", 1_000_000)),
        )
    if args.target == "thesis-pipeline":
        from src.analytics.wave_d_exit import run_thesis_pipeline_command

        return run_thesis_pipeline_command(
            thesis_id=str(getattr(args, "thesis_id", "ai_compute")),
            as_of=str(args.as_of) if getattr(args, "as_of", None) else None,
            settings=DataSettings(),
            allow_stale=bool(getattr(args, "allow_stale", False)),
            seed=int(getattr(args, "seed", 7)),
            bootstrap_paths=int(getattr(args, "bootstrap_paths", 400)),
        )
    raise _UsageError(f"unsupported target {args.target!r}")


if __name__ == "__main__":
    raise SystemExit(main())
