"""Command-line ingest and baseline-run entry (no secret printing)."""

from __future__ import annotations

import argparse
import logging
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
from src.etf_manager.policy.targets import PolicyError, PolicyId, policy_sleeves
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

if TYPE_CHECKING:
    import httpx

    from src.etf_manager.data.secrets import ProviderSecrets

logger = logging.getLogger(__name__)

_SMOKE_START: Final[date] = date(2024, 1, 2)
_SMOKE_END: Final[date] = date(2024, 1, 5)
_SMOKE_TICKER: Final[str] = "VT"
_SMOKE_FX_PROVIDER: Final[str] = "fred"
_HISTORY_TICKERS: Final[tuple[str, ...]] = ("VT", "VTI")
_HISTORY_FX_PROVIDER: Final[str] = "fred"


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
        tickers = tuple(args.tickers) if args.tickers else _HISTORY_TICKERS
        return run_ingest_history(
            start=args.start,
            end=args.end,
            tickers=tickers,
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
        )
    raise _UsageError(f"unsupported target {args.target!r}")


def _resolve_tilt(factor: str | None, intensity: float | None) -> FactorTilt | None:
    """Accept tilt flags only as a pair; a lone flag is a usage error."""
    if (factor is None) != (intensity is None):
        raise _UsageError("--tilt-factor and --tilt-intensity must be provided together")
    if factor is None or intensity is None:
        return None
    return FactorTilt(factor=factor, intensity=intensity)


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
    tickers: tuple[str, ...],
    fx_provider: str,
    settings: DataSettings,
    secrets: ProviderSecrets,
    client: httpx.Client | None = None,
) -> int:
    """Persist FX, prices, and CPI over a long window; all three datasets are required.

    Returns 0 only when every fetch persists and each latest catalog partition
    holds row_count >= 1; vendor/catalog messages never reach the log.
    """
    try:
        fx = fetch_and_persist_fx(
            provider=fx_provider, start=start, end=end, secrets=secrets, settings=settings, client=client
        )
        prices = fetch_and_persist_prices(tickers, start, end, secrets=secrets, settings=settings, client=client)
        cpi = fetch_and_persist_cpi(start, end, secrets=secrets, settings=settings, client=client)
        row_counts = {
            str(dataset): latest_artifact(settings, dataset).manifest.row_count
            for dataset in (Dataset.PRICES, Dataset.FX, Dataset.CPI)
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
        "[DATA] event=history_ok tickers=%s price_rows=%d fx_rows=%d cpi_rows=%d",
        ",".join(tickers),
        prices.manifest.row_count,
        fx.manifest.row_count,
        cpi.manifest.row_count,
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
) -> int:
    """Run a stored-data strategic allocation and log terminal KRW / XIRR / MDD."""
    config = AllocationConfig(
        policy=PolicyId(policy_id),
        start=start,
        end=end,
        monthly_contribution_krw=float(contribution_krw),
        fill_delay_sessions=1,
        commission_bps=0.0,
        tilt=tilt,
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


if __name__ == "__main__":
    raise SystemExit(main())
