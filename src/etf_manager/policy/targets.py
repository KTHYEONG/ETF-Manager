"""Named strategic target weights."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Final

from src.etf_manager.features.returns import session_returns
from src.etf_manager.features.risk import trailing_simple_vol

if TYPE_CHECKING:
    from datetime import datetime

    import polars as pl

__all__ = ["PolicyError", "PolicyId", "policy_sleeves", "resolve_targets"]


class PolicyId(StrEnum):
    """Named strategic policies; static maps plus one inverse-vol rule."""

    S0_GLOBAL = "s0_global"
    S1_US = "s1_us"
    S2_REGIONAL = "s2_regional"
    S3_GLOBAL_BOND = "s3_global_bond"
    S4_DEFENSIVE = "s4_defensive"
    S5_INVVOL = "s5_invvol"


class PolicyError(RuntimeError):
    """Target-weight resolution failed closed (invalid weights or unusable sleeve vols)."""


_STATIC_TARGETS: Final[dict[PolicyId, dict[str, float]]] = {
    PolicyId.S0_GLOBAL: {"VT": 1.0},
    PolicyId.S1_US: {"VTI": 1.0},
    PolicyId.S2_REGIONAL: {"VTI": 0.5, "VEA": 0.3, "VWO": 0.2},
    PolicyId.S3_GLOBAL_BOND: {"VT": 0.7, "BND": 0.3},
    PolicyId.S4_DEFENSIVE: {"VT": 0.6, "IEF": 0.2, "TLT": 0.2},
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
        PolicyError: When any sleeve's vol window fails closed or the resolved
            weights violate the simplex invariants.
    """
    if signal_at.tzinfo is None:
        raise ValueError(f"signal_at must be timezone-aware, got naive datetime {signal_at!r}")
    targets = _invvol_targets(prices, signal_at) if policy is PolicyId.S5_INVVOL else dict(_STATIC_TARGETS[policy])
    total = sum(targets.values())
    if any(weight < 0.0 for weight in targets.values()) or abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise PolicyError(f"{policy!s} resolved invalid target weights: sum={total!r}")
    return targets


def policy_sleeves(policy: PolicyId) -> tuple[str, ...]:
    """Ordered sleeve tickers owned by ``policy`` (CLI logging seam)."""
    if policy is PolicyId.S5_INVVOL:
        return _INVVOL_SLEEVES
    return tuple(_STATIC_TARGETS[policy])


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
