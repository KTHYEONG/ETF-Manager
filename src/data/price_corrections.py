"""Operator-curated replacement bars for vendor prints proven wrong by an official source.

Corrections are research inputs, never inferred values: every entry carries an
evidence URL, and a price set that fails validation is rejected rather than repaired.
The applied set is applied to the candidate frame immediately before Silver
persistence, so the corrected bar and the file digest share one manifest.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import polars as pl

from src.data.paths import PRICE_CORRECTIONS_PATH, resolve_input_path

logger = logging.getLogger(__name__)

_CORRECTED_PRICE_FIELDS: Final[tuple[str, ...]] = ("open", "high", "low", "close")


@dataclass(frozen=True, slots=True)
class PriceCorrection:
    """One replacement bar for a vendor print proven wrong by an official source.

    Attributes:
        ticker: PRICES ticker.
        session: Exchange session of the bad bar.
        open: Corrected open.
        high: Corrected high.
        low: Corrected low.
        close: Corrected unadjusted close.
        evidence_url: http(s) source proving the corrected values.
        reason: Short operator rationale.
    """

    ticker: str
    session: date
    open: float
    high: float
    low: float
    close: float
    evidence_url: str
    reason: str


@dataclass(frozen=True, slots=True)
class PriceCorrectionSet:
    """Validated corrections plus the SHA-256 of the file bytes they came from.

    Attributes:
        corrections: Validated replacement bars ordered by (ticker, session).
        sha256: Hex digest of the correction file bytes, recorded in the manifest.
    """

    corrections: tuple[PriceCorrection, ...]
    sha256: str


def _finite_price(value: object, field: str, ticker: str) -> float:
    """Coerce one JSON price into a finite positive float, naming the offending field.

    Raises:
        ValueError: If the value is not numeric, is not finite, or is not positive.
    """
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"correction {ticker!r} has a non-numeric {field!r} {value!r}")
    price = float(value)
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"correction {ticker!r} has a non-positive or non-finite {field!r} {value!r}")
    return price


def _parse_correction(entry: object, index: int, source: Path) -> PriceCorrection:
    """Validate one JSON entry into a typed correction, naming the offending field.

    Raises:
        ValueError: If a required field is absent or a value violates the OHLC contract.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"correction {index} of {source.as_posix()!r} is not a JSON object")
    ticker = str(entry.get("ticker", "")).strip()
    if not ticker:
        raise ValueError(f"correction {index} of {source.as_posix()!r} declares no 'ticker'")
    raw_session = entry.get("session")
    try:
        session = date.fromisoformat(str(raw_session))
    except ValueError as exc:
        raise ValueError(f"correction {ticker!r} has a malformed 'session' {raw_session!r}") from exc
    key = f"{ticker}@{session.isoformat()}"
    prices = {field: _finite_price(entry.get(field), field, ticker) for field in _CORRECTED_PRICE_FIELDS}
    evidence_url = str(entry.get("evidence_url", "")).strip()
    if urlparse(evidence_url).scheme not in ("http", "https"):
        raise ValueError(f"correction {key} has a non-http 'evidence_url' {evidence_url!r}")
    reason = str(entry.get("reason", "")).strip()
    if not reason:
        raise ValueError(f"correction {key} declares no 'reason'")
    if prices["high"] < max(prices["open"], prices["close"]):
        raise ValueError(f"correction {key} has 'high' {prices['high']!r} below open/close")
    if prices["low"] > min(prices["open"], prices["close"]):
        raise ValueError(f"correction {key} has 'low' {prices['low']!r} above open/close")
    return PriceCorrection(
        ticker=ticker,
        session=session,
        open=prices["open"],
        high=prices["high"],
        low=prices["low"],
        close=prices["close"],
        evidence_url=evidence_url,
        reason=reason,
    )


def load_price_corrections(path: str | Path = PRICE_CORRECTIONS_PATH) -> PriceCorrectionSet:
    """Load and validate the versioned correction file.

    A relative path is anchored to the repository root, so every caller reads the one
    git-tracked correction file regardless of the process working directory.

    Args:
        path: JSON document with a `corrections` list.

    Returns:
        Corrections ordered by (ticker, session) with the file digest.

    Raises:
        ValueError: If the document is malformed, a key is duplicated, a price is
            non-finite or non-positive, high/low do not bound open/close, or the
            evidence URL is not http(s).
    """
    source = resolve_input_path(path)
    raw = source.read_bytes()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"correction file {source.as_posix()!r} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"correction file {source.as_posix()!r} is not a JSON object")
    entries = document.get("corrections")
    if not isinstance(entries, list):
        raise ValueError(f"correction file {source.as_posix()!r} declares no 'corrections' list")
    corrections: list[PriceCorrection] = []
    seen: set[tuple[str, date]] = set()
    for index, entry in enumerate(entries):
        correction = _parse_correction(entry, index, source)
        identity = (correction.ticker, correction.session)
        if identity in seen:
            raise ValueError(f"duplicate correction for {correction.ticker}@{correction.session.isoformat()}")
        seen.add(identity)
        corrections.append(correction)
    corrections.sort(key=lambda correction: (correction.ticker, correction.session))
    return PriceCorrectionSet(corrections=tuple(corrections), sha256=hashlib.sha256(raw).hexdigest())


def apply_price_corrections(frame: pl.DataFrame, corrections: PriceCorrectionSet) -> pl.DataFrame:
    """Replace corrected bars while preserving each row's vendor adjustment factor.

    The adjusted close of a corrected row keeps the row's original
    adjusted_close/close ratio, because that ratio is the cumulative split and
    distribution factor and is independent of the bad print.

    Args:
        frame: PRICES candidate frame before persistence.
        corrections: Validated correction set.

    Returns:
        A frame with identical keys, row count, and column order.

    Raises:
        ValueError: If a corrected row's original close is non-positive.
    """
    applied = 0
    skipped = 0
    patched = frame
    for correction in corrections.corrections:
        target = (pl.col("ticker") == correction.ticker) & (pl.col("date") == correction.session)
        matched = patched.filter(target)
        if matched.is_empty():
            skipped += 1
            continue
        if (matched.get_column("close") <= 0).any():
            raise ValueError(
                f"correction for {correction.ticker}@{correction.session.isoformat()} "
                "targets a row whose close is non-positive"
            )
        adjustment = pl.col("adjusted_close") / pl.col("close")
        patched = patched.with_columns(
            pl.when(target).then(pl.lit(correction.open)).otherwise(pl.col("open")).alias("open"),
            pl.when(target).then(pl.lit(correction.high)).otherwise(pl.col("high")).alias("high"),
            pl.when(target).then(pl.lit(correction.low)).otherwise(pl.col("low")).alias("low"),
            pl.when(target).then(pl.lit(correction.close)).otherwise(pl.col("close")).alias("close"),
            pl.when(target)
            .then(pl.lit(correction.close) * adjustment)
            .otherwise(pl.col("adjusted_close"))
            .alias("adjusted_close"),
        )
        applied += 1
    logger.info(
        "[DATA] event=price_corrections_applied applied=%d skipped=%d sha=%s",
        applied,
        skipped,
        corrections.sha256,
    )
    return patched
