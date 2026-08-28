"""Popular US vehicle diagnostics versus locked S1 sleeves (reporting only)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from src.etf.mapping import mapping_implementation_tickers
from src.features.factors import estimate_factor_loadings
from src.policy.targets import all_policy_tickers
from src.sim.baseline import BaselineConfig, BaselineResult, run_baseline

if TYPE_CHECKING:
    from datetime import datetime

    import polars as pl

__all__ = [
    "VehicleDcaPath",
    "VehicleFactorProfile",
    "compare_vehicle_dca",
    "diagnostic_price_tickers",
    "history_price_tickers",
    "profile_us_vehicles",
    "research_satellite_tickers",
]


@dataclass(frozen=True, slots=True)
class VehicleFactorProfile:
    """Trailing factor loadings of one vehicle (decimals); never an adoption input."""

    ticker: str
    alpha: float
    mkt_rf: float
    smb: float
    hml: float
    rmw: float
    cma: float
    mom: float


@dataclass(frozen=True, slots=True)
class VehicleDcaPath:
    """Reporting-only DCA outcome of one vehicle on identical external cashflows."""

    ticker: str
    result: BaselineResult


def diagnostic_price_tickers() -> tuple[str, ...]:
    """Diagnostic-only price tickers kept in history ingest alongside policy sleeves."""
    return ("QQQ",)


def research_satellite_tickers() -> tuple[str, ...]:
    """Future-industry satellite sleeve tickers for static-mix research."""
    return ("BOTZ", "GRID", "IBB", "ITA", "IWF", "SOXX", "XLI")


def history_price_tickers() -> tuple[str, ...]:
    """Sorted union of policy sleeves, mapping implementations, diagnostics, and research satellites."""
    return tuple(
        sorted(
            {
                *all_policy_tickers(),
                *diagnostic_price_tickers(),
                *mapping_implementation_tickers(),
                *research_satellite_tickers(),
            }
        )
    )


def profile_us_vehicles(
    prices: pl.DataFrame,
    factors: pl.DataFrame,
    *,
    tickers: tuple[str, ...],
    signal_at: datetime,
    window: int = 36,
) -> tuple[VehicleFactorProfile, ...]:
    """PIT trailing factor profile per ticker in request order; fail-closed per sleeve.

    Raises:
        ValueError: On a naive ``signal_at`` or any sleeve failing its OLS window.
    """
    if signal_at.tzinfo is None:
        raise ValueError(f"signal_at must be timezone-aware, got naive datetime {signal_at!r}")
    profiles: list[VehicleFactorProfile] = []
    for ticker in tickers:
        loadings = estimate_factor_loadings(prices, factors, ticker=ticker, signal_at=signal_at, window=window)
        profiles.append(
            VehicleFactorProfile(
                ticker=ticker,
                alpha=loadings["alpha"],
                mkt_rf=loadings["mkt_rf"],
                smb=loadings["smb"],
                hml=loadings["hml"],
                rmw=loadings["rmw"],
                cma=loadings["cma"],
                mom=loadings["mom"],
            )
        )
    return tuple(profiles)


def compare_vehicle_dca(
    base: BaselineConfig,
    prices: pl.DataFrame,
    fx: pl.DataFrame,
    cpi: pl.DataFrame,
    tickers: tuple[str, ...],
) -> tuple[VehicleDcaPath, ...]:
    """Run the same cashflow DCA per vehicle; wealth plus XIRR are reporting only.

    Raises:
        ValueError: On a non-positive contribution or diverging contribution ledgers.
    """
    paths: list[VehicleDcaPath] = []
    contributions: list[tuple[float, ...]] = []
    for ticker in tickers:
        config = replace(base, ticker=ticker)
        result = run_baseline(config, prices, fx, cpi)
        contributions.append(tuple(snapshot.contribution_krw for snapshot in result.snapshots))
        paths.append(VehicleDcaPath(ticker=ticker, result=result))
    if contributions and any(ledger != contributions[0] for ledger in contributions[1:]):
        raise ValueError("vehicle DCA snapshots diverge on contribution_krw; cashflows must be identical")
    return tuple(paths)
