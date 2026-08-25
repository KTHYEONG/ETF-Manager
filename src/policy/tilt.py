"""Fixed long-only factor tilt over Phase 3 strategic weights."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from src.features.factors import estimate_factor_loadings
from src.policy.targets import PolicyError, PolicyId, resolve_targets

if TYPE_CHECKING:
    from datetime import datetime

    import polars as pl

TILT_FACTORS: Final[tuple[str, ...]] = ("smb", "hml", "rmw", "cma", "mom")
_TILT_FACTOR_SET: Final[frozenset[str]] = frozenset(TILT_FACTORS)
_MAX_INTENSITY: Final[float] = 0.25
_WEIGHT_SUM_TOLERANCE: Final[float] = 1e-6

__all__ = ["TILT_FACTORS", "FactorTilt", "apply_fixed_tilt", "resolve_tilted_targets"]


@dataclass(frozen=True, slots=True)
class FactorTilt:
    """Bounded fixed tilt on one non-market factor; market beta is never tilted."""

    factor: str
    intensity: float

    def __post_init__(self) -> None:
        if self.factor not in _TILT_FACTOR_SET:
            raise ValueError(f"tilt factor must be one of {sorted(_TILT_FACTOR_SET)}, got {self.factor!r}")
        if not 0.0 < self.intensity <= _MAX_INTENSITY:
            raise ValueError(f"tilt intensity must lie in (0, {_MAX_INTENSITY}], got {self.intensity!r}")


def apply_fixed_tilt(
    weights: dict[str, float],
    loadings: dict[str, dict[str, float]],
    tilt: FactorTilt,
) -> dict[str, float]:
    """Overlay ``intensity * z / sum(|z|)`` per sleeve, clip at zero, renormalize.

    Z-scores use the sample mean/stdev of the chosen factor loading across the
    sleeves present in ``weights``.

    Returns:
        Long-only weights summing to one within 1e-6.

    Raises:
        PolicyError: When loadings are missing, degenerate (zero dispersion), or
            the tilted map violates the simplex invariants.
    """
    try:
        values = [float(loadings[sleeve][tilt.factor]) for sleeve in weights]
    except KeyError as exc:
        raise PolicyError(f"missing {tilt.factor!r} loading for tilt: {exc}") from exc
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1) if len(values) > 1 else 0.0
    stdev = math.sqrt(variance)
    if stdev <= 0.0:
        raise PolicyError(f"factor tilt requires nonzero loading dispersion for {tilt.factor!r}")
    z_scores = [(value - mean) / stdev for value in values]
    z_abs_sum = sum(abs(z) for z in z_scores)
    if z_abs_sum == 0.0:
        raise PolicyError(f"factor tilt requires nonzero loading dispersion for {tilt.factor!r}")
    clipped = {
        sleeve: max(0.0, weight + tilt.intensity * z / z_abs_sum)
        for sleeve, weight, z in zip(weights, weights.values(), z_scores, strict=True)
    }
    total = sum(clipped.values())
    if total <= 0.0:
        raise PolicyError("factor tilt collapsed every sleeve weight to zero")
    tilted = {sleeve: weight / total for sleeve, weight in clipped.items()}
    resolved_total = sum(tilted.values())
    if any(weight < 0.0 for weight in tilted.values()) or abs(resolved_total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise PolicyError(f"tilted weights violate the simplex: sum={resolved_total!r}")
    return tilted


def resolve_tilted_targets(
    policy: PolicyId,
    prices: pl.DataFrame,
    factors: pl.DataFrame,
    signal_at: datetime,
    tilt: FactorTilt | None,
) -> dict[str, float]:
    """Base policy targets at ``signal_at``, overlaid with ``tilt`` when present.

    A ``None`` tilt is the exact Phase 3 identity path.
    """
    targets = resolve_targets(policy, prices, signal_at)
    if tilt is None:
        return targets
    loadings = {
        sleeve: estimate_factor_loadings(prices, factors, ticker=sleeve, signal_at=signal_at)
        for sleeve in targets
    }
    return apply_fixed_tilt(targets, loadings, tilt)
