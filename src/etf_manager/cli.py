"""Command-line ingest and baseline-run entry (no secret printing)."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from datetime import date
from typing import TYPE_CHECKING, Final, NoReturn

from src.etf_manager.analytics.metrics import XirrError
from src.etf_manager.data.catalog import latest_artifact
from src.etf_manager.data.fetch import (
    fetch_and_persist_cpi,
    fetch_and_persist_factors,
    fetch_and_persist_fx,
    fetch_and_persist_macro,
    fetch_and_persist_prices,
)
from src.etf_manager.data.providers.base import ProviderError
from src.etf_manager.data.schema import Dataset
from src.etf_manager.data.secrets import load_provider_secrets
from src.etf_manager.data.settings import DataSettings
from src.etf_manager.data.storage import UntrustedDatasetError
from src.etf_manager.etf.mapping import MappingConfig
from src.etf_manager.execution.broker import replay_paper
from src.etf_manager.execution.orders import ExecutionError, orders_from_snapshots
from src.etf_manager.policy.currency import CurrencyConfig
from src.etf_manager.policy.overlay import OverlayConfig
from src.etf_manager.policy.targets import (
    PolicyError,
    PolicyId,
    all_policy_tickers,
    policy_sleeves,
)
from src.etf_manager.policy.tilt import TILT_FACTORS, FactorTilt
from src.etf_manager.sim.allocation import (
    AllocationConfig,
    AllocationDataError,
    run_allocation_from_store,
)
from src.etf_manager.sim.baseline import (
    BaselineConfig,
    BaselineDataError,
    BaselineId,
    run_baseline_from_store,
)
from src.etf_manager.validation.ablation import run_ablation
from src.etf_manager.validation.bootstrap import moving_block_bootstrap
from src.etf_manager.validation.evaluate import evaluate_cohort_wealths
from src.etf_manager.validation.experiment import load_experiment_config
from src.etf_manager.validation.gate import adoption_passes, certainty_equivalent
from src.etf_manager.validation.registry import make_experiment
from src.etf_manager.validation.windows import rolling_cohorts

if TYPE_CHECKING:
    import httpx

    from src.etf_manager.data.secrets import ProviderSecrets

logger = logging.getLogger(__name__)

_SMOKE_START: Final[date] = date(2024, 1, 2)
_SMOKE_END: Final[date] = date(2024, 1, 5)
_SMOKE_TICKER: Final[str] = "VT"
_SMOKE_FX_PROVIDER: Final[str] = "fred"
_HISTORY_FX_PROVIDER: Final[str] = "fred"
_HISTORY_MACRO_SERIES: Final[str] = "VIXCLS"
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
    ingest.add_argument("dataset", choices=("prices", "fx", "macro", "cpi", "factors", "smoke", "history"))
    ingest.add_argument("--tickers", nargs="+", default=None, help="Price tickers (prices/smoke only)")
    ingest.add_argument("--provider", choices=("fred", "ecos"), default=None, help="FX vendor (fx/smoke only)")
    ingest.add_argument("--series-id", default=None, help="FRED series identifier (macro only)")
    ingest.add_argument("--start", type=_iso_date, default=None, help="ISO start date (required except smoke)")
    ingest.add_argument("--end", type=_iso_date, default=None, help="ISO end date (required except smoke)")
    run_parser = subparsers.add_parser("run", help="Run a stored-data simulation")
    run_targets = run_parser.add_subparsers(dest="target", required=True)
    baseline = run_targets.add_parser("baseline", help="Run a B0/B1 DCA baseline on catalog partitions")
    baseline.add_argument("--id", choices=("b0_global", "b1_us"), required=True)
    baseline.add_argument("--ticker", required=True)
    baseline.add_argument("--start", required=True, type=_iso_date)
    baseline.add_argument("--end", required=True, type=_iso_date)
    baseline.add_argument("--contribution-krw", required=True, type=float)
    policy = run_targets.add_parser("policy", help="Run an S-policy strategic allocation on catalog partitions")
    policy.add_argument("--id", choices=tuple(str(member) for member in PolicyId), required=True)
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
        "--overlay-max-shift",
        type=float,
        default=None,
        help="Bounded overlay max_shift in (0, 0.10]; omit to disable overlay",
    )
    policy.add_argument(
        "--vix-threshold",
        type=float,
        default=None,
        help="Optional VIX de-risk threshold (requires --overlay-max-shift)",
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
    validate.add_argument("--delta0", type=float, default=0.02, help="Per-module complexity margin")
    validate.add_argument("--modules", type=int, default=0, help="Count of added signal/sleeve modules")
    validate.add_argument("--horizon-months", type=int, default=12, help="Cohort horizon in calendar months")
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
    if args.command == "run":
        return _dispatch_run(args)
    if args.command != "ingest":
        raise _UsageError(f"unsupported command {args.command!r}")
    dataset: str = args.dataset
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
    return run_ingest_smoke(
        start=args.start if args.start is not None else _SMOKE_START,
        end=args.end if args.end is not None else _SMOKE_END,
        ticker=tickers[0] if tickers else _SMOKE_TICKER,
        fx_provider=str(args.provider) if args.provider is not None else _SMOKE_FX_PROVIDER,
        settings=DataSettings(),
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
        return run_policy_command(
            policy_id=str(args.id),
            start=args.start,
            end=args.end,
            contribution_krw=float(args.contribution_krw),
            settings=DataSettings(),
            tilt=tilt,
            rebalance_band=args.rebalance_band,
            overlay=_resolve_overlay(args.overlay_max_shift, args.vix_threshold),
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
    """Persist FX, prices, CPI, factors, and VIXCLS macro over a long window.

    ``tickers`` defaults to the full policy sleeve universe. Returns 0 only when
    every fetch persists and each of the five latest catalog partitions holds
    row_count >= 1; vendor/catalog messages never reach the log.
    """
    price_tickers = tickers if tickers is not None else all_policy_tickers()
    try:
        fx = fetch_and_persist_fx(
            provider=fx_provider, start=start, end=end, secrets=secrets, settings=settings, client=client
        )
        prices = fetch_and_persist_prices(price_tickers, start, end, secrets=secrets, settings=settings, client=client)
        cpi = fetch_and_persist_cpi(start, end, secrets=secrets, settings=settings, client=client)
        factors = fetch_and_persist_factors(start, end, settings=settings, client=client)
        macro = fetch_and_persist_macro(
            _HISTORY_MACRO_SERIES, start, end, secrets=secrets, settings=settings, client=client
        )
        row_counts = {
            str(dataset): latest_artifact(settings, dataset).manifest.row_count
            for dataset in (Dataset.PRICES, Dataset.FX, Dataset.CPI, Dataset.FACTORS, Dataset.MACRO)
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
        "[DATA] event=history_ok tickers=%s price_rows=%d fx_rows=%d cpi_rows=%d factor_rows=%d macro_rows=%d",
        ",".join(price_tickers),
        prices.manifest.row_count,
        fx.manifest.row_count,
        cpi.manifest.row_count,
        factors.manifest.row_count,
        macro.manifest.row_count,
    )
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
        baseline=BaselineId(baseline_id),
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
    currency: CurrencyConfig | None = None,
    mapping: MappingConfig | None = None,
) -> int:
    """Run a stored-data strategic allocation and log terminal KRW / XIRR / MDD.

    Raises:
        ValueError: When ``rebalance_band`` is set outside ``[0, 1)`` or the mapping config is invalid.
    """
    if rebalance_band is not None and not 0.0 <= rebalance_band < 1.0:
        raise ValueError(f"rebalance_band must lie in [0, 1), got {rebalance_band!r}")
    config = AllocationConfig(
        policy=PolicyId(policy_id),
        start=start,
        end=end,
        monthly_contribution_krw=float(contribution_krw),
        fill_delay_sessions=1,
        commission_bps=0.0,
        tilt=tilt,
        rebalance_band=rebalance_band,
        overlay=overlay,
        currency=currency,
        mapping=mapping,
    )
    try:
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
        report = run_ablation(spec, lambda config: run_allocation_from_store(config, settings))
        metrics: dict[str, float] = {
            "candidates": float(len(report.rows)),
            "adopted": float(sum(row.adopted for row in report.rows)),
        }
        for row in report.rows:
            for gamma, ratio in row.ce_ratio.items():
                metrics[f"{row.candidate_id}_ratio_gamma_{int(gamma)}"] = ratio
        record = make_experiment(
            config=AllocationConfig(
                policy=spec.baseline.policy,
                start=spec.start,
                end=spec.end,
                monthly_contribution_krw=spec.contribution_krw,
                fill_delay_sessions=1,
                commission_bps=0.0,
            ),
            manifest_hash=latest_artifact(settings, Dataset.PRICES).manifest.normalized_sha256,
            git_commit=_resolve_git_commit(),
            seed=None,
            metrics=metrics,
        )
    except (AllocationDataError, PolicyError, UntrustedDatasetError, XirrError, ValueError) as exc:
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


def run_paper_command(
    *,
    policy_id: str,
    start: date,
    end: date,
    contribution_krw: float,
    settings: DataSettings,
) -> int:
    """Replay a stored-data policy onto PaperBroker and fail closed on lot mismatch."""
    config = AllocationConfig(
        policy=PolicyId(policy_id),
        start=start,
        end=end,
        monthly_contribution_krw=float(contribution_krw),
        fill_delay_sessions=1,
        commission_bps=0.0,
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
