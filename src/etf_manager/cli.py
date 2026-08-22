"""Command-line ingest entry (no secret printing)."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from typing import TYPE_CHECKING, NoReturn

from src.etf_manager.data.fetch import (
    fetch_and_persist_cpi,
    fetch_and_persist_fx,
    fetch_and_persist_macro,
    fetch_and_persist_prices,
)
from src.etf_manager.data.providers.base import ProviderError
from src.etf_manager.data.secrets import load_provider_secrets
from src.etf_manager.data.settings import DataSettings

if TYPE_CHECKING:
    import httpx

    from src.etf_manager.data.secrets import ProviderSecrets

logger = logging.getLogger(__name__)


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
    ingest.add_argument("dataset", choices=("prices", "fx", "macro", "cpi"))
    ingest.add_argument("--tickers", nargs="+", default=None, help="Price tickers (prices only)")
    ingest.add_argument("--provider", choices=("fred", "ecos"), default=None, help="FX vendor (fx only)")
    ingest.add_argument("--series-id", default=None, help="FRED series identifier (macro only)")
    ingest.add_argument("--start", required=True, type=_iso_date)
    ingest.add_argument("--end", required=True, type=_iso_date)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ingest subcommands and dispatch to fetch_and_persist functions.

    Exit codes: 0 on success, 2 on argparse usage errors, 1 on provider or
    value failures. Token values are never logged.
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
    if args.command != "ingest":
        raise _UsageError(f"unsupported command {args.command!r}")
    dataset: str = args.dataset
    if dataset == "prices" and not args.tickers:
        raise _UsageError("ingest prices requires --tickers")
    if dataset == "fx" and args.provider is None:
        raise _UsageError("ingest fx requires --provider fred|ecos")
    if dataset == "macro" and not args.series_id:
        raise _UsageError("ingest macro requires --series-id")

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


if __name__ == "__main__":
    raise SystemExit(main())


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
    """Persist a tiny live window of FRED FX and Tiingo prices; ECOS is optional."""
    raise NotImplementedError


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
    raise NotImplementedError
