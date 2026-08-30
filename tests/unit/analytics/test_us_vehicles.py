"""Unit tests for popular US vehicle diagnostics (reporting only; no adoption gate)."""

from __future__ import annotations

import calendar as _calendar
import math
from datetime import UTC, date, datetime, timedelta
from typing import Final

import polars as pl
import pytest

from src.analytics.us_vehicles import (
    compare_vehicle_dca,
    diagnostic_price_tickers,
    history_price_tickers,
    profile_us_vehicles,
    research_satellite_tickers,
)
from src.data.calendar import load_calendar
from src.data.pipeline import ingest
from src.data.pit import AVAILABLE_AT
from src.data.schema import Dataset, spec_for
from src.etf.mapping import mapping_implementation_tickers
from src.features.factors import FACTOR_COLUMNS
from src.policy.targets import all_policy_tickers
from src.sim.baseline import BaselineConfig, BaselineId

_CALENDAR = load_calendar("XNYS")
_RETRIEVED_AT = datetime(2024, 4, 1, 5, 0, tzinfo=UTC)
_VEHICLE_TICKERS: Final[tuple[str, ...]] = ("IVV", "QQQ", "VTI")
_SMB_BETAS: Final[dict[str, float]] = {"VTI": 0.20, "IVV": 0.00, "QQQ": -0.30}


def _month_ends(count: int, start_year: int = 2019, start_month: int = 1) -> list[date]:
    ends: list[date] = []
    year, month = start_year, start_month
    for _ in range(count):
        ends.append(date(year, month, _calendar.monthrange(year, month)[1]))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return ends


def _close_ts(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 21, tzinfo=UTC)


@pytest.mark.parametrize("scenario_id", ["VEH-E-history-union"])
def test_veh_e_history_union(scenario_id: str) -> None:
    """VEH-E-history-union"""
    assert diagnostic_price_tickers() == ("QQQ",)
    assert history_price_tickers() == (
        "BND",
        "BOTZ",
        "GRID",
        "IBB",
        "IEF",
        "IEMG",
        "ITA",
        "ITOT",
        "IVV",
        "IWF",
        "PAVE",
        "QQQ",
        "ROBO",
        "SCHF",
        "SOXX",
        "TLT",
        "VEA",
        "VT",
        "VTI",
        "VTV",
        "VWO",
        "XLI",
    )
    assert set(history_price_tickers()) - set(all_policy_tickers()) == {
        "BOTZ",
        "GRID",
        "IBB",
        "IEMG",
        "ITA",
        "ITOT",
        "IWF",
        "PAVE",
        "ROBO",
        "SCHF",
        "SOXX",
        "XLI",
    }


@pytest.mark.parametrize("scenario_id", ["VEH-J-history-includes-itot"])
def test_veh_j_history_includes_itot(scenario_id: str) -> None:
    """VEH-J-history-includes-itot"""
    implementations = mapping_implementation_tickers()
    assert isinstance(implementations, tuple)
    assert implementations == tuple(sorted(implementations))
    assert "ITOT" in implementations

    history = history_price_tickers()
    assert "ITOT" in history
    assert "VTI" in history
    assert set(implementations) <= set(history)


@pytest.mark.parametrize("scenario_id", ["VEH-MIX-satellite-ingest"])
def test_veh_mix_satellite_ingest(scenario_id: str) -> None:
    """VEH-MIX-satellite-ingest"""
    satellites = research_satellite_tickers()
    assert satellites == ("BOTZ", "GRID", "IBB", "ITA", "IWF", "PAVE", "ROBO", "SOXX", "XLI")
    history = history_price_tickers()
    assert set(satellites) <= set(history)
    assert set(satellites).isdisjoint(set(all_policy_tickers()))


@pytest.mark.parametrize("scenario_id", ["test_thesis_panel_tickers_include_pave"])
def test_thesis_panel_tickers_include_pave(scenario_id: str) -> None:
    """test_thesis_panel_tickers_include_pave"""
    from src.data.panel_freshness import THESIS_PANEL_TICKERS

    assert "PAVE" in THESIS_PANEL_TICKERS
    assert "GRID" in THESIS_PANEL_TICKERS
    assert THESIS_PANEL_TICKERS == ("BOTZ", "GRID", "PAVE", "QQQ", "ROBO", "SOXX")
    satellites = research_satellite_tickers()
    assert "PAVE" in satellites
    assert "PAVE" in history_price_tickers()


@pytest.mark.parametrize("scenario_id", ["test_thesis_panel_tickers_include_robo"])
def test_thesis_panel_tickers_include_robo(scenario_id: str) -> None:
    """test_thesis_panel_tickers_include_robo"""
    from src.data.panel_freshness import THESIS_PANEL_TICKERS

    assert "ROBO" in THESIS_PANEL_TICKERS
    assert "ROBO" in research_satellite_tickers()
    assert "ROBO" in history_price_tickers()
    assert "BOTZ" in THESIS_PANEL_TICKERS


def _smb_factor_frame(months: list[date]) -> pl.DataFrame:
    """Only mkt_rf and smb vary; structurally zero columns must yield zero betas."""
    count = len(months)
    data: dict[str, list[object]] = {"period_end": months}
    for name in FACTOR_COLUMNS:
        if name == "mkt_rf":
            data[name] = [0.010 + 0.001 * math.sin(index) for index in range(count)]
        elif name == "smb":
            data[name] = [0.002 * math.cos(index / 2.0) for index in range(count)]
        else:
            data[name] = [0.0] * count
    data["rf"] = [0.0005] * count
    data[AVAILABLE_AT] = [_close_ts(month) + timedelta(days=1) for month in months]
    return pl.DataFrame(data)


def _vehicle_price_frame(months: list[date], tickers: tuple[str, ...]) -> pl.DataFrame:
    """Month-end prices whose monthly excess return is exactly mkt_rf + beta_smb * smb."""
    factors = _smb_factor_frame(months)
    rf = factors.get_column("rf").to_list()
    mkt_rf = factors.get_column("mkt_rf").to_list()
    smb = factors.get_column("smb").to_list()
    ticker_column: list[str] = []
    dates: list[date] = []
    closes: list[float] = []
    for ticker in tickers:
        price = 100.0
        ticker_column.append(ticker)
        dates.append(months[0])
        closes.append(price)
        for index in range(1, len(months)):
            gross_return = rf[index] + mkt_rf[index] + _SMB_BETAS[ticker] * smb[index]
            price *= 1.0 + gross_return
            ticker_column.append(ticker)
            dates.append(months[index])
            closes.append(price)
    return pl.DataFrame(
        {
            "ticker": ticker_column,
            "date": dates,
            "adjusted_close": closes,
            AVAILABLE_AT: [_close_ts(month) + timedelta(hours=1) for month in dates],
        }
    )


@pytest.mark.parametrize("scenario_id", ["VEH-E-smb-order"])
def test_veh_e_smb_order(scenario_id: str) -> None:
    """VEH-E-smb-order"""
    months = _month_ends(38)
    prices = _vehicle_price_frame(months, _VEHICLE_TICKERS)
    factors = _smb_factor_frame(months)
    signal_at = factors.get_column(AVAILABLE_AT)[-1] + timedelta(hours=1)

    profiles = profile_us_vehicles(prices, factors, tickers=_VEHICLE_TICKERS, signal_at=signal_at, window=36)

    assert tuple(profile.ticker for profile in profiles) == _VEHICLE_TICKERS
    betas = {profile.ticker: profile.smb for profile in profiles}
    assert betas["QQQ"] < betas["IVV"] < betas["VTI"]
    for profile in profiles:
        assert abs(profile.smb - _SMB_BETAS[profile.ticker]) <= 0.05
        assert profile.alpha == pytest.approx(0.0, abs=1e-9)
        assert profile.mkt_rf == pytest.approx(1.0, abs=1e-9)
        for name in ("hml", "rmw", "cma", "mom"):
            assert getattr(profile, name) == pytest.approx(0.0, abs=1e-12)

    with pytest.raises(ValueError, match="timezone-aware"):
        profile_us_vehicles(
            prices, factors, tickers=_VEHICLE_TICKERS, signal_at=signal_at.replace(tzinfo=None), window=36
        )


_PANEL_START: Final[date] = date(2024, 1, 10)
_PANEL_END: Final[date] = date(2024, 2, 28)


def _dca_prices_panel(days: tuple[date, ...], tickers: tuple[str, ...]) -> pl.DataFrame:
    spec = spec_for(Dataset.PRICES)
    ticker_column: list[str] = []
    dates: list[date] = []
    closes: list[float] = []
    for ticker in tickers:
        price = 100.0
        for day in days:
            ticker_column.append(ticker)
            dates.append(day)
            closes.append(price)
            price *= 1.001
    n = len(dates)
    return ingest(
        pl.DataFrame(
            {
                "ticker": ticker_column,
                "date": dates,
                "open": [value * 0.98 for value in closes],
                "high": [value * 1.02 for value in closes],
                "low": [value * 0.97 for value in closes],
                "close": closes,
                "volume": [10_000] * n,
                "adjusted_close": closes,
                "dividend": [0.0] * n,
                "split_factor": [1.0] * n,
                "source": ["synthetic"] * n,
                "retrieved_at": [_RETRIEVED_AT] * n,
            },
            schema=dict(spec.columns),
        ),
        Dataset.PRICES,
    )


def _dca_fx_panel(days: tuple[date, ...]) -> pl.DataFrame:
    spec = spec_for(Dataset.FX)
    ordered = list(days)
    return ingest(
        pl.DataFrame(
            {
                "date": ordered,
                "usdkrw": [1300.0] * len(ordered),
                "source": ["synthetic"] * len(ordered),
                "retrieved_at": [_RETRIEVED_AT] * len(ordered),
            },
            schema=dict(spec.columns),
        ),
        Dataset.FX,
    )


def _dca_cpi_panel() -> pl.DataFrame:
    spec = spec_for(Dataset.CPI)
    return ingest(
        pl.DataFrame(
            {
                "period_end": [date(2023, 12, 1)],
                "value": [100.0],
                "source": ["synthetic"],
                "retrieved_at": [_RETRIEVED_AT],
            },
            schema=dict(spec.columns),
        ),
        Dataset.CPI,
    )


@pytest.mark.parametrize("scenario_id", ["VEH-E-dca-identical-cashflow"])
def test_veh_e_dca_identical_cashflow(scenario_id: str) -> None:
    """VEH-E-dca-identical-cashflow"""
    days = _CALENDAR.sessions(_PANEL_START, _PANEL_END)
    prices = _dca_prices_panel(days, ("VTI", "IVV"))
    fx = _dca_fx_panel(days)
    cpi = _dca_cpi_panel()
    base = BaselineConfig(
        baseline=BaselineId.B1_US,
        ticker="VTI",
        start=date(2024, 1, 15),
        end=date(2024, 2, 26),
        monthly_contribution_krw=1_000_000.0,
    )

    paths = compare_vehicle_dca(base, prices, fx, cpi, tickers=("VTI", "IVV"))

    assert tuple(path.ticker for path in paths) == ("VTI", "IVV")
    contribution_sequences = [
        tuple(snapshot.contribution_krw for snapshot in path.result.snapshots) for path in paths
    ]
    assert contribution_sequences[0] == contribution_sequences[1]
    assert all(value == base.monthly_contribution_krw for value in contribution_sequences[0])

    with pytest.raises(ValueError, match="monthly_contribution_krw"):
        compare_vehicle_dca(
            BaselineConfig(
                baseline=base.baseline,
                ticker="VTI",
                start=base.start,
                end=base.end,
                monthly_contribution_krw=-1.0,
            ),
            prices,
            fx,
            cpi,
            tickers=("VTI",),
        )
