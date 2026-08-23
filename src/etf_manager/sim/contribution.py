"""Buy-only contribution mixer: band + cost-aware spend fractions."""

from __future__ import annotations

import math
from typing import Final

__all__ = ["allocate_contribution"]

_BPS: Final[float] = 10_000.0
_WEIGHT_SUM_TOLERANCE: Final[float] = 1e-6


def allocate_contribution(
    *,
    targets: dict[str, float],
    marks_krw: dict[str, float],
    nav_krw: float,
    commission_bps: float,
    rebalance_band: float | None,
) -> dict[str, float]:
    """Split one month's investable KRW into buy-only spend fractions per sleeve.

    ``rebalance_band is None`` returns the targets unchanged (Phase 3 identity).
    Otherwise eligible sleeves are those underweight beyond the band
    (``w_cur < w_tgt - band``, missing marks count as 0), falling back to any
    underweight sleeve, then to the plain target mix when nothing is underweight.
    Eligible sleeves score proportionally to their KRW deficit divided by the
    gross-cost multiplier ``1 + commission_bps / 1e4``; zero total deficit falls
    back to the targets.

    Raises:
        ValueError: When ``targets`` is not a nonnegative simplex, ``nav_krw``
            is not positive, any mark is negative or non-finite, or
            ``commission_bps`` is negative, or ``rebalance_band`` is set and
            lies outside ``[0, 1)``.
    """
    if not math.isfinite(nav_krw) or nav_krw <= 0.0:
        raise ValueError(f"nav_krw must be positive, got {nav_krw!r}")
    if any(not math.isfinite(mark) or mark < 0.0 for mark in marks_krw.values()):
        raise ValueError("marks_krw entries must be finite and nonnegative")
    target_total = sum(targets.values())
    if any(not math.isfinite(weight) or weight < 0.0 for weight in targets.values()) or abs(target_total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"targets must form a nonnegative simplex, got sum={target_total!r}")
    if commission_bps < 0.0:
        raise ValueError(f"commission_bps must be nonnegative, got {commission_bps!r}")
    if rebalance_band is None:
        return dict(targets)
    band = float(rebalance_band)
    if not math.isfinite(band) or not 0.0 <= band < 1.0:
        raise ValueError(f"rebalance_band must lie in [0, 1), got {rebalance_band!r}")

    def current_weight(ticker: str) -> float:
        return marks_krw.get(ticker, 0.0) / nav_krw

    eligible = [ticker for ticker in targets if current_weight(ticker) < targets[ticker] - band]
    if not eligible:
        eligible = [ticker for ticker in targets if current_weight(ticker) < targets[ticker]]
    if not eligible:
        return dict(targets)

    cost_multiplier = 1.0 + commission_bps / _BPS
    scores = {
        ticker: max(0.0, targets[ticker] * nav_krw - marks_krw.get(ticker, 0.0)) / cost_multiplier
        for ticker in eligible
    }
    score_total = sum(scores.values())
    if score_total <= 0.0:
        return dict(targets)
    fractions = dict.fromkeys(targets, 0.0)
    for ticker, score in scores.items():
        fractions[ticker] = score / score_total
    return fractions
