"""Unit tests for the bounded dynamic overlay."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from src.data.calendar import load_calendar
from src.data.pit import AVAILABLE_AT
from src.policy.overlay import OverlayConfig, apply_bounded_overlay
from src.policy.targets import PolicyError

_CALENDAR = load_calendar("XNYS")
_SLEEVES = ("VTI", "VEA", "VWO")
_PANEL_DAYS = _CALENDAR.sessions(date(2023, 3, 1), date(2024, 3, 28))[:260]
_SIGNAL_AT = _CALENDAR.close_ts(_PANEL_DAYS[-1])


def _rising_panel() -> pl.DataFrame:
    """Identical monotonic 0.1%/session closes per sleeve: positive trend, equal vols, no drawdown."""
    tickers: list[str] = []
    dates: list[date] = []
    closes: list[float] = []
    for ticker in _SLEEVES:
        for index, day in enumerate(_PANEL_DAYS):
            tickers.append(ticker)
            dates.append(day)
            closes.append(100.0 * 1.001**index)
    return pl.DataFrame(
        {
            "ticker": tickers,
            "date": dates,
            "adjusted_close": closes,
            AVAILABLE_AT: [_CALENDAR.close_ts(day) for day in dates],
        },
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "adjusted_close": pl.Float64,
            AVAILABLE_AT: pl.Datetime("us", "UTC"),
        },
    )


def _vix_frame(series_id: str = "VIXCLS", value: float = 25.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "series_id": [series_id],
            "observation_date": [date(2024, 2, 1)],
            "value": [value],
            AVAILABLE_AT: [_SIGNAL_AT - timedelta(days=1)],
        },
        schema={
            "series_id": pl.String,
            "observation_date": pl.Date,
            "value": pl.Float64,
            AVAILABLE_AT: pl.Datetime("us", "UTC"),
        },
    )


@pytest.mark.parametrize("scenario_id", ["POL-J02-overlay-bounds"])
def test_pol_j02_overlay_bounds(scenario_id: str) -> None:
    """POL-J02-overlay-bounds"""
    panel = _rising_panel()
    overlay = OverlayConfig(max_shift=0.10)

    boosted = apply_bounded_overlay({"VTI": 0.4, "VEA": 0.3, "VWO": 0.2}, panel, _SIGNAL_AT, overlay)

    # Benign signals give u_i = +0.5 (trend only), so each sleeve scales by 1 + 0.10*0.5.
    assert boosted["VTI"] == pytest.approx(0.42, rel=1e-9)
    assert boosted["VEA"] == pytest.approx(0.315, rel=1e-9)
    assert boosted["VWO"] == pytest.approx(0.21, rel=1e-9)
    assert sum(boosted.values()) <= 1.0

    renormalized = apply_bounded_overlay({"VTI": 0.6, "VEA": 0.4}, panel, _SIGNAL_AT, overlay)

    # Sum above one must be scaled back to exactly one, preserving sleeve proportions.
    assert sum(renormalized.values()) == pytest.approx(1.0, rel=1e-9)
    assert renormalized["VTI"] / renormalized["VEA"] == pytest.approx(1.5, rel=1e-9)
    assert all(weight >= 0.0 for weight in (*boosted.values(), *renormalized.values()))
    with pytest.raises(ValueError, match="max_shift"):
        OverlayConfig(max_shift=0.2)


@pytest.mark.parametrize("scenario_id", ["POL-J03-vix-derisk"])
def test_pol_j03_vix_derisk(scenario_id: str) -> None:
    """POL-J03-vix-derisk"""
    panel = _rising_panel()
    weights = {"VTI": 0.3, "VEA": 0.3}
    plain = OverlayConfig(max_shift=0.10)
    gated = OverlayConfig(max_shift=0.10, vix_threshold=20.0)

    without_vix = apply_bounded_overlay(weights, panel, _SIGNAL_AT, plain)
    with_vix = apply_bounded_overlay(weights, panel, _SIGNAL_AT, gated, macro=_vix_frame(value=25.0))

    assert sum(with_vix.values()) < sum(without_vix.values())
    with pytest.raises(PolicyError):
        apply_bounded_overlay(weights, panel, _SIGNAL_AT, gated)
    with pytest.raises(PolicyError):
        apply_bounded_overlay(weights, panel, _SIGNAL_AT, gated, macro=_vix_frame(series_id="OTHER"))


@pytest.mark.parametrize("scenario_id", ["POL-J03-vix-derisk"])
def test_pol_j03_fail_closed_guards(scenario_id: str) -> None:
    """POL-J03-vix-derisk"""
    panel = _rising_panel()
    overlay = OverlayConfig(max_shift=0.10)
    naive_signal = datetime(2024, 3, 28, 21)
    with pytest.raises(ValueError, match="timezone-aware"):
        apply_bounded_overlay({"VTI": 1.0}, panel, naive_signal, overlay)
    short_panel = panel.filter(pl.col("date") >= date(2024, 3, 1))
    with pytest.raises(PolicyError):
        apply_bounded_overlay({"VTI": 1.0}, short_panel, _SIGNAL_AT, overlay)
