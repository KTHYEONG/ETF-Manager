"""Validated point-in-time listing lifetimes for tradable instruments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import cast
from urllib.parse import urlparse

import polars as pl

from src.data.paths import UNIVERSE_MEMBERSHIP_PATH, resolve_input_path

__all__ = [
    "MembershipEntry",
    "UniverseMembership",
    "assert_rows_within_lifetime",
    "eligible_at",
    "load_universe_membership",
]


@dataclass(frozen=True, slots=True)
class MembershipEntry:
    """Listing lifetime of one tradable instrument.

    Attributes:
        ticker: PRICES ticker.
        listing_date: First tradable session.
        last_trading_date: Final session on which the instrument trades, or None while listed.
        evidence_url: http(s) source for the dates.
    """

    ticker: str
    listing_date: date
    last_trading_date: date | None
    evidence_url: str


@dataclass(frozen=True, slots=True)
class UniverseMembership:
    """Validated membership table with the SHA-256 of its source file."""

    entries: Mapping[str, MembershipEntry]
    sha256: str


def _parse_date(value: object, field: str, ticker: str, source: Path) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(
            f"membership entry {ticker!r} in {source.as_posix()!r} has malformed {field!r} {value!r}"
        ) from exc


def _parse_entry(entry: object, index: int, source: Path) -> MembershipEntry:
    if not isinstance(entry, dict):
        raise ValueError(f"membership entry {index} in {source.as_posix()!r} is not a JSON object")
    required = {"ticker", "listing_date", "last_trading_date", "evidence_url"}
    missing = required.difference(entry)
    if missing:
        raise ValueError(
            f"membership entry {index} in {source.as_posix()!r} is missing field(s) {sorted(missing)!r}"
        )

    ticker = str(entry["ticker"]).strip()
    if not ticker:
        raise ValueError(f"membership entry {index} in {source.as_posix()!r} has an empty ticker")
    listing_date = _parse_date(entry["listing_date"], "listing_date", ticker, source)
    raw_last_trading_date = entry["last_trading_date"]
    last_trading_date = (
        None
        if raw_last_trading_date is None
        else _parse_date(raw_last_trading_date, "last_trading_date", ticker, source)
    )
    if last_trading_date is not None and last_trading_date < listing_date:
        raise ValueError(
            f"membership entry {ticker!r} has last_trading_date before listing_date"
        )
    evidence_url = str(entry["evidence_url"]).strip()
    parsed_url = urlparse(evidence_url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        raise ValueError(f"membership entry {ticker!r} has non-http evidence_url {evidence_url!r}")
    return MembershipEntry(
        ticker=ticker,
        listing_date=listing_date,
        last_trading_date=last_trading_date,
        evidence_url=evidence_url,
    )


def load_universe_membership(path: str | Path = UNIVERSE_MEMBERSHIP_PATH) -> UniverseMembership:
    """Load the versioned membership table.

    Raises:
        ValueError: On duplicate tickers, delisting not after listing, non-http evidence, or malformed JSON.
    """
    source = resolve_input_path(path)
    raw = source.read_bytes()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"membership file {source.as_posix()!r} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"membership file {source.as_posix()!r} is not a JSON object")
    raw_entries = document.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError(f"membership file {source.as_posix()!r} declares no 'entries' list")

    entries: dict[str, MembershipEntry] = {}
    for index, raw_entry in enumerate(raw_entries):
        entry = _parse_entry(raw_entry, index, source)
        if entry.ticker in entries:
            raise ValueError(f"duplicate membership ticker {entry.ticker!r}")
        entries[entry.ticker] = entry
    ordered = MappingProxyType(dict(sorted(entries.items())))
    return UniverseMembership(entries=ordered, sha256=hashlib.sha256(raw).hexdigest())


def eligible_at(
    membership: UniverseMembership, tickers: Sequence[str], day: date
) -> tuple[str, ...]:
    """Tickers tradable on `day`, preserving input order.

    Eligibility uses only listing and delisting events on or before `day`, so a
    selection at time T never learns that a member will later delist.

    Raises:
        ValueError: If a requested ticker has no membership entry.
    """
    eligible: list[str] = []
    for ticker in tickers:
        entry = membership.entries.get(ticker)
        if entry is None:
            raise ValueError(f"unknown membership ticker {ticker!r}")
        if entry.listing_date <= day and (
            entry.last_trading_date is None or day <= entry.last_trading_date
        ):
            eligible.append(ticker)
    return tuple(eligible)


def assert_rows_within_lifetime(frame: pl.DataFrame, membership: UniverseMembership) -> None:
    """Reject PRICES rows that fall outside their ticker's listed lifetime.

    Vendors reuse the tickers of closed funds for later, unrelated products (a
    reused symbol would otherwise splice two different funds into one price
    series), and some keep rows after the final trading date.

    Args:
        frame: PRICES rows with `ticker` and `date`.
        membership: Validated membership table.

    Raises:
        ValueError: If a row precedes `listing_date` or follows `last_trading_date`,
            or its ticker has no membership entry; the message names ticker, date, and count.
    """
    tickers = sorted(frame.get_column("ticker").unique().to_list())
    for ticker in tickers:
        ticker_rows = frame.filter(pl.col("ticker") == ticker)
        entry = membership.entries.get(ticker)
        if entry is None:
            first_date = cast(date, ticker_rows.get_column("date").min())
            raise ValueError(
                f"ticker={ticker!r} has no universe membership entry; "
                f"date={first_date} count={ticker_rows.height}"
            )
        within_lifetime = pl.col("date") >= entry.listing_date
        if entry.last_trading_date is not None:
            within_lifetime = within_lifetime & (pl.col("date") <= entry.last_trading_date)
        offending = ticker_rows.filter(~within_lifetime)
        if not offending.is_empty():
            first_date = cast(date, offending.get_column("date").min())
            raise ValueError(
                f"ticker={ticker!r} has PRICES rows outside its membership lifetime; "
                f"date={first_date} count={offending.height}"
            )
