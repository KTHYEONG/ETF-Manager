"""QQQ core / SOXX satellite risk-budget mixing."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import polars as pl

from src.features.returns import session_returns
from src.features.risk import trailing_simple_corr, trailing_simple_vol
from src.policy.targets import PolicyError

__all__ = [
    "OPERATIONAL_MIX_RISK_BUDGET",
    "MixRiskBudgetConfig",
    "resolve_mix_risk_budget_targets",
    "satellite_risk_budget_weight",
]


@dataclass(frozen=True, slots=True)
class MixRiskBudgetConfig:
    core_ticker: str = "QQQ"
    satellite_ticker: str = "SOXX"
    satellite_risk_budget: float = 0.10
    satellite_weight_floor: float = 0.0
    satellite_weight_cap: float = 0.15
    vol_window: int = 63


OPERATIONAL_MIX_RISK_BUDGET: Final[MixRiskBudgetConfig] = MixRiskBudgetConfig(
    core_ticker="QQQ",
    satellite_ticker="SOXX",
    satellite_risk_budget=0.10,
    satellite_weight_floor=0.0,
    satellite_weight_cap=0.15,
    vol_window=63,
)


def satellite_risk_budget_weight(
    *, sigma_core: float, sigma_satellite: float, rho: float, theta: float
) -> float:
    """Solve risk-budget weight for the satellite.

    Unique root in (0,1) of ``A*w^2+B*w+C=0`` with the coefficients
    described in the spec. Uses linear fallback when ``|A|<1e-18``.
    """
    sc = float(sigma_core)
    ss = float(sigma_satellite)
    r = float(rho)
    th = float(theta)
    if not math.isfinite(sc) or sc <= 0.0:
        raise PolicyError(f"sigma_core must be finite positive, got {sigma_core!r}")
    if not math.isfinite(ss) or ss <= 0.0:
        raise PolicyError(f"sigma_satellite must be finite positive, got {sigma_satellite!r}")
    if not math.isfinite(r):
        raise PolicyError(f"rho must be finite, got {rho!r}")
    if not math.isfinite(th):
        raise PolicyError(f"theta must be finite, got {theta!r}")
    # also rho outside [-1,1] could be clipping overflow but non-finite already handled
    # allow rho slightly beyond due to overflow but treat as error if far?
    # Spec says rho clipped only for overflow, so we accept values but algorithm will still work.
    # However degenerate rho nan already caught; infinite already.
    a = ss * ss
    b = sc * sc
    c = r * sc * ss
    A = (a - c) - th * (a + b - 2.0 * c)  # noqa: N806
    B = c + 2.0 * th * b - 2.0 * th * c  # noqa: N806
    C = -th * b  # noqa: N806
    if abs(A) < 1e-18:
        if not math.isfinite(B) or abs(B) < 1e-18:
            raise PolicyError(f"degenerate linear coefficient B={B!r} for A={A!r}")
        w = -C / B
        if not math.isfinite(w):
            raise PolicyError(f"linear solve non-finite w={w!r}")
        if not (0.0 < w < 1.0):
            raise PolicyError(f"linear solve w={w!r} not in (0,1)")
        return float(w)
    disc = B * B - 4.0 * A * C
    if not math.isfinite(disc):
        raise PolicyError(f"discriminant non-finite {disc!r}")
    if disc < 0.0:
        # allow tiny negative due to float error
        if disc > -1e-18:
            disc = 0.0
        else:
            raise PolicyError(f"discriminant negative {disc!r}")
    sqrt_disc = math.sqrt(disc)
    r1 = (-B + sqrt_disc) / (2.0 * A)
    r2 = (-B - sqrt_disc) / (2.0 * A)
    candidates = [float(w) for w in (r1, r2) if math.isfinite(w) and 0.0 < w < 1.0]
    if len(candidates) != 1:
        raise PolicyError(f"expected unique root in (0,1), got {candidates!r} from roots {(r1, r2)!r}")
    return float(candidates[0])


def resolve_mix_risk_budget_targets(
    prices: pl.DataFrame, signal_at: datetime, config: MixRiskBudgetConfig
) -> dict[str, float]:
    """Resolve core/satellite simplex targets at ``signal_at`` via trailing risk budget."""
    if signal_at.tzinfo is None:
        raise ValueError(f"signal_at must be timezone-aware, got naive datetime {signal_at!r}")
    try:
        rets_core = session_returns(prices, ticker=config.core_ticker)
        rets_sat = session_returns(prices, ticker=config.satellite_ticker)
    except ValueError as exc:
        raise PolicyError(f"risk budget session_returns failed: {exc}") from exc
    try:
        sigma_core = trailing_simple_vol(rets_core, as_of_ts=signal_at, window=config.vol_window)
    except ValueError as exc:
        raise PolicyError(f"core vol failed closed: {exc}") from exc
    try:
        sigma_sat = trailing_simple_vol(rets_sat, as_of_ts=signal_at, window=config.vol_window)
    except ValueError as exc:
        raise PolicyError(f"satellite vol failed closed: {exc}") from exc
    try:
        rho = trailing_simple_corr(rets_core, rets_sat, as_of_ts=signal_at, window=config.vol_window)
    except ValueError as exc:
        raise PolicyError(f"corr failed closed: {exc}") from exc
    w_sat = satellite_risk_budget_weight(
        sigma_core=sigma_core, sigma_satellite=sigma_sat, rho=rho, theta=config.satellite_risk_budget
    )
    w_sat = float(min(config.satellite_weight_cap, max(config.satellite_weight_floor, w_sat)))
    w_core = 1.0 - w_sat
    # simplex validation
    if not math.isfinite(w_sat) or not math.isfinite(w_core):
        raise PolicyError(f"non-finite weights core={w_core!r} sat={w_sat!r}")
    if w_sat < -1e-12 or w_core < -1e-12:
        raise PolicyError(f"negative weights core={w_core!r} sat={w_sat!r}")
    # clip tiny negatives
    if w_sat < 0.0:
        w_sat = 0.0
    if w_core < 0.0:
        w_core = 0.0
    total = w_core + w_sat
    if abs(total - 1.0) > 1e-6:
        raise PolicyError(f"weights sum {total!r} not within 1e-6 of 1")
    return {config.core_ticker: float(w_core), config.satellite_ticker: float(w_sat)}
