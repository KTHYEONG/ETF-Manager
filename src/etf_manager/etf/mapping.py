"""Bounded ETF implementation mapping with incumbent hysteresis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import polars as pl

from src.etf_manager.etf.score import etf_score, latest_metadata_row, passes_hard_filters
from src.etf_manager.policy.targets import PolicyError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

_WEIGHT_SUM_TOLERANCE: Final[float] = 1e-6

DEFAULT_CANDIDATES: Final[dict[str, tuple[str, ...]]] = {
    "VT": ("VT",),
    "VTI": ("VTI", "ITOT"),
    "VEA": ("VEA", "SCHF"),
    "VWO": ("VWO", "IEMG"),
    "BND": ("BND",),
    "IEF": ("IEF",),
    "TLT": ("TLT",),
}

__all__ = ["DEFAULT_CANDIDATES", "MappingConfig", "apply_etf_mapping", "mapping_implementation_tickers"]


@dataclass(frozen=True, slots=True)
class MappingConfig:
    """Implementation catalog and hysteresis; scores never chase trailing return."""

    min_improvement: float = 0.02
    min_aum_usd: float = 1e8
    min_dollar_volume: float = 1e6
    min_track_record_days: int = 365
    fit_window: int = 63
    td_window: int = 126
    expense_weight: float = 1.0
    td_weight: float = 1.0
    spread_weight: float = 1.0
    candidates: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(DEFAULT_CANDIDATES))

    def __post_init__(self) -> None:
        if not 0.0 < self.min_improvement <= 1.0:
            raise ValueError(f"min_improvement must lie in (0, 1], got {self.min_improvement!r}")


def mapping_implementation_tickers(
    candidates: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    """Sorted union of every implementation ticker in a sleeve catalog."""
    catalog = DEFAULT_CANDIDATES if candidates is None else candidates
    return tuple(sorted({ticker for implementations in catalog.values() for ticker in implementations}))


def _argmax_lexicographic(scores: Mapping[str, float]) -> str:
    """Highest score; ties resolved by the lexicographically smallest ticker."""
    best: str | None = None
    for ticker in sorted(scores):
        if best is None or scores[ticker] > scores[best]:
            best = ticker
    assert best is not None
    return best


def _select_implementation(
    prices: pl.DataFrame,
    metadata: pl.DataFrame,
    *,
    sleeve: str,
    signal_at: datetime,
    mapping: MappingConfig,
    incumbent: str | None,
) -> str:
    """Argmax-score passing candidate, sticky to the incumbent within ``min_improvement``."""
    candidates = mapping.candidates.get(sleeve, ())
    if not candidates:
        raise PolicyError(f"no candidate catalog for economic sleeve {sleeve!r}")
    scores: dict[str, float] = {}
    for candidate in candidates:
        row = latest_metadata_row(metadata, ticker=candidate, signal_at=signal_at)
        if not passes_hard_filters(row, sleeve=sleeve, signal_at=signal_at, mapping=mapping):
            continue
        scores[candidate] = etf_score(
            prices, metadata, ticker=candidate, sleeve=sleeve, signal_at=signal_at, mapping=mapping
        )
    if not scores:
        raise PolicyError(f"no candidate passes hard filters for sleeve {sleeve!r} at the signal instant")
    if incumbent is not None and incumbent in scores:
        challengers = {ticker: score for ticker, score in scores.items() if ticker != incumbent}
        if challengers:
            challenger = _argmax_lexicographic(challengers)
            if scores[challenger] >= scores[incumbent] + mapping.min_improvement:
                return challenger
        return incumbent
    # Absent or filtered-out incumbent: forced switch to the current argmax.
    return _argmax_lexicographic(scores)


def apply_etf_mapping(
    weights: dict[str, float],
    prices: pl.DataFrame,
    metadata: pl.DataFrame,
    signal_at: datetime,
    mapping: MappingConfig,
    incumbents: dict[str, str],
) -> tuple[dict[str, float], dict[str, str]]:
    """Map economic sleeve weights to implementation tickers; sticky vs incumbent.

    Each economic weight is remapped onto its chosen implementation ticker;
    colliding implementation keys are summed and weights are never renormalized
    nor sold down, so existing lots stay untouched. The returned incumbents map
    persists across schedule steps.

    Raises:
        ValueError: When ``signal_at`` is naive or ``weights`` violate the simplex invariants.
        PolicyError: When a sleeve has no passing candidate or a score window fails closed.
    """
    if signal_at.tzinfo is None:
        raise ValueError(f"signal_at must be timezone-aware, got naive datetime {signal_at!r}")
    total = sum(weights.values())
    if any(weight < 0.0 for weight in weights.values()) or total > 1.0 + _WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"weights must be nonnegative with sum <= 1, got sum {total!r}")
    mapped: dict[str, float] = {}
    updated_incumbents = dict(incumbents)
    for sleeve in sorted(weights):
        choice = _select_implementation(
            prices, metadata, sleeve=sleeve, signal_at=signal_at, mapping=mapping,
            incumbent=updated_incumbents.get(sleeve),
        )
        mapped[choice] = mapped.get(choice, 0.0) + weights[sleeve]
        updated_incumbents[sleeve] = choice
    return mapped, updated_incumbents
