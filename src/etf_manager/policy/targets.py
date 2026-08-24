"""Named strategic target weights."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from src.etf_manager.features.returns import session_returns
from src.etf_manager.features.risk import trailing_simple_vol

if TYPE_CHECKING:
    from datetime import datetime

    import polars as pl

__all__ = [
    "BASELINE_ALIASES",
    "POLICY_ALIASES",
    "UNIVERSE_VEHICLE",
    "BaselineId",
    "PolicyError",
    "PolicyId",
    "UsEquityUniverse",
    "all_policy_tickers",
    "policy_sleeves",
    "resolve_targets",
]


class PolicyId(StrEnum):
    """Named strategic policies; static maps plus one inverse-vol rule."""

    S0_GLOBAL = "global"
    S1_US = "us"
    S2_REGIONAL = "regional"
    S3_GLOBAL_BOND = "global_bond"
    S4_DEFENSIVE = "defensive"
    S5_INVVOL = "inv_vol"
    S6_US_CORE_VALUE = "us_value"
    S7_US_LARGE_CAP = "us_large"
    # Campaign identity only: never resolvable into ETF sleeve targets.
    R1_US_MKT_FF = "us_ff"

    @classmethod
    def parse(cls, value: object) -> PolicyId:
        """Resolve a canonical or legacy policy string to its member; unknown fails closed."""
        key = value if isinstance(value, PolicyId) else str(value)
        if key in POLICY_ALIASES:
            return POLICY_ALIASES[key]
        raise ValueError(f"unknown policy {value!r}")


POLICY_ALIASES: Final[Mapping[str, PolicyId]] = {
    member.value: member for member in PolicyId
} | {
    "s0_global": PolicyId.S0_GLOBAL,
    "s1_us": PolicyId.S1_US,
    "s2_regional": PolicyId.S2_REGIONAL,
    "s3_global_bond": PolicyId.S3_GLOBAL_BOND,
    "s4_defensive": PolicyId.S4_DEFENSIVE,
    "s5_invvol": PolicyId.S5_INVVOL,
    "s6_us_core_value": PolicyId.S6_US_CORE_VALUE,
    "s7_us_large_cap": PolicyId.S7_US_LARGE_CAP,
    "r1_us_mkt_ff": PolicyId.R1_US_MKT_FF,
}


class BaselineId(StrEnum):
    """Named one-ticker accumulation policies; weights live only in config."""

    B0_GLOBAL = "dca_global"
    B1_US = "dca_us"

    @classmethod
    def parse(cls, value: object) -> BaselineId:
        """Resolve a canonical or legacy baseline id to its member; unknown fails closed."""
        key = value if isinstance(value, BaselineId) else str(value)
        if key in BASELINE_ALIASES:
            return BASELINE_ALIASES[key]
        raise ValueError(f"unknown baseline id {value!r}")


BASELINE_ALIASES: Final[Mapping[str, BaselineId]] = {
    member.value: member for member in BaselineId
} | {
    "b0_global": BaselineId.B0_GLOBAL,
    "b1_us": BaselineId.B1_US,
}


class PolicyError(RuntimeError):
    """Target-weight resolution failed closed (invalid weights or unusable sleeve vols)."""


class UsEquityUniverse(StrEnum):
    """US equity universe buckets, each mapped to a single listed vehicle."""

    TOTAL_MARKET = "us_total_market"
    LARGE_CAP = "us_large_cap"
    # Diagnostic-only bucket: resolvable to a ticker through UNIVERSE_VEHICLE,
    # never through resolve_targets (no PolicyId owns a Nasdaq-100 sleeve).
    NASDAQ_100 = "us_nasdaq_100"


UNIVERSE_VEHICLE: Final[dict[UsEquityUniverse, str]] = {
    UsEquityUniverse.TOTAL_MARKET: "VTI",
    UsEquityUniverse.LARGE_CAP: "IVV",
    UsEquityUniverse.NASDAQ_100: "QQQ",
}

_STATIC_TARGETS: Final[dict[PolicyId, dict[str, float]]] = {
    PolicyId.S0_GLOBAL: {"VT": 1.0},
    PolicyId.S1_US: {UNIVERSE_VEHICLE[UsEquityUniverse.TOTAL_MARKET]: 1.0},
    PolicyId.S2_REGIONAL: {"VTI": 0.5, "VEA": 0.3, "VWO": 0.2},
    PolicyId.S3_GLOBAL_BOND: {"VT": 0.7, "BND": 0.3},
    PolicyId.S4_DEFENSIVE: {"VT": 0.6, "IEF": 0.2, "TLT": 0.2},
    PolicyId.S6_US_CORE_VALUE: {"VTI": 0.8, "VTV": 0.2},
    PolicyId.S7_US_LARGE_CAP: {UNIVERSE_VEHICLE[UsEquityUniverse.LARGE_CAP]: 1.0},
}
_INVVOL_SLEEVES: Final[tuple[str, ...]] = ("VTI", "VEA", "VWO")
_WEIGHT_SUM_TOLERANCE: Final[float] = 1e-6


def resolve_targets(policy: PolicyId, prices: pl.DataFrame, signal_at: datetime) -> dict[str, float]:
    """Resolve sleeve weights for ``policy`` using only data visible at ``signal_at``.

    Static ids ignore ``prices`` entirely; ``S5_INVVOL`` weighs each regional sleeve
    by inverse trailing 63-session vol from PIT session returns at ``signal_at``.
    Every returned map is nonnegative and sums to one within 1e-6.

    Raises:
        ValueError: When ``signal_at`` is naive.
        PolicyError: When the requested policy is a research_proxy identity, or any
            sleeve's vol window fails closed, or the resolved weights violate the
            simplex invariants.
    """
    if policy is PolicyId.R1_US_MKT_FF:
        raise PolicyError("R1_US_MKT_FF is a research_proxy identity; it has no ETF target weights")
    if signal_at.tzinfo is None:
        raise ValueError(f"signal_at must be timezone-aware, got naive datetime {signal_at!r}")
    targets = _invvol_targets(prices, signal_at) if policy is PolicyId.S5_INVVOL else dict(_STATIC_TARGETS[policy])
    total = sum(targets.values())
    if any(weight < 0.0 for weight in targets.values()) or abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise PolicyError(f"{policy!s} resolved invalid target weights: sum={total!r}")
    return targets


def policy_sleeves(policy: PolicyId) -> tuple[str, ...]:
    """Ordered sleeve tickers owned by ``policy`` (CLI logging seam).

    The research_proxy policy owns no tickers, so the ingest universe never grows.
    """
    if policy is PolicyId.R1_US_MKT_FF:
        return ()
    if policy is PolicyId.S5_INVVOL:
        return _INVVOL_SLEEVES
    return tuple(_STATIC_TARGETS[policy])


def all_policy_tickers() -> tuple[str, ...]:
    """Sorted union of every policy's sleeve tickers; the history ingest universe."""
    union: set[str] = set(_INVVOL_SLEEVES)
    for targets in _STATIC_TARGETS.values():
        union.update(targets)
    return tuple(sorted(union))


def _invvol_targets(prices: pl.DataFrame, signal_at: datetime) -> dict[str, float]:
    """Inverse-vol weights over regional sleeves; fails closed on any vol failure."""
    inverse_vols: list[float] = []
    for sleeve in _INVVOL_SLEEVES:
        try:
            sigma = trailing_simple_vol(session_returns(prices, ticker=sleeve), as_of_ts=signal_at)
        except ValueError as exc:
            raise PolicyError(f"inverse-vol sleeve {sleeve!r} failed closed at {signal_at.isoformat()}: {exc}") from exc
        inverse_vols.append(1.0 / sigma)
    total_inverse = sum(inverse_vols)
    return {sleeve: inverse / total_inverse for sleeve, inverse in zip(_INVVOL_SLEEVES, inverse_vols, strict=True)}
