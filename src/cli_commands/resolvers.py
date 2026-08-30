"""Resolver helpers for policy flags."""

from __future__ import annotations

from src.cli_commands.parser import _UsageError
from src.etf.mapping import MappingConfig
from src.policy.currency import CurrencyConfig
from src.policy.overlay import OverlayConfig
from src.policy.reserve import ReserveConfig
from src.policy.tilt import FactorTilt


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


def _resolve_reserve(withhold_cap: float | None, overlay: OverlayConfig | None) -> ReserveConfig | None:
    """Accept a reserve cap only without any overlay flag; omitting it keeps the identity."""
    if withhold_cap is not None and overlay is not None:
        raise _UsageError("--reserve-withhold-cap cannot be combined with overlay flags")
    if withhold_cap is None:
        return None
    return ReserveConfig(max_withhold=withhold_cap)


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
