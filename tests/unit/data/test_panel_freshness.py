"""Unit tests for panel freshness."""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.data.panel_freshness import (
    THESIS_PANEL_TICKERS,
    CatalogPanelReport,
    PanelFreshnessStatus,
    PanelHardStop,
    apply_hard_stop,
    effective_thesis_end,
    iter_nport_quarters_for_panel,
    resolve_catalog_panel_as_of,
)
from src.data.settings import DataSettings


def _synthetic_frames(panel_end: date, tickers: tuple[str, ...] = THESIS_PANEL_TICKERS) -> dict:
    """Build minimal synthetic frames that will pass boundary check for panel_end."""
    from src.data.calendar import load_calendar
    from src.data.pit import AVAILABLE_AT, TS_DTYPE
    import polars as pl
    from datetime import datetime, UTC

    calendar = load_calendar("XNYS")
    # Ensure panel_end is a session; if not, use next session earlier? For synthetic we force pass via available_at early.
    try:
        close_ts = calendar.close_ts(panel_end)
    except Exception:
        close_ts = datetime(panel_end.year, panel_end.month, panel_end.day, 20, 0, tzinfo=UTC)

    # Prices: one row per ticker at panel_end
    prices_rows = [
        {
            "ticker": t,
            "date": panel_end,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000,
            "adjusted_close": 100.0,
            "dividend": 0.0,
            "split_factor": 1.0,
            "source": "tiingo",
            "retrieved_at": datetime(2026, 1, 1, tzinfo=UTC),
            AVAILABLE_AT: close_ts,
        }
        for t in tickers
    ]
    prices = pl.DataFrame(prices_rows, schema={
        "ticker": pl.String, "date": pl.Date, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
        "close": pl.Float64, "volume": pl.Int64, "adjusted_close": pl.Float64, "dividend": pl.Float64,
        "split_factor": pl.Float64, "source": pl.String, "retrieved_at": TS_DTYPE, AVAILABLE_AT: TS_DTYPE,
    })
    fx = pl.DataFrame([{
        "date": panel_end,
        "usdkrw": 1300.0,
        "source": "fred",
        "retrieved_at": datetime(2026, 1, 1, tzinfo=UTC),
        AVAILABLE_AT: close_ts,
    }], schema={"date": pl.Date, "usdkrw": pl.Float64, "source": pl.String, "retrieved_at": TS_DTYPE, AVAILABLE_AT: TS_DTYPE})
    cpi = pl.DataFrame([{
        "period_end": panel_end,
        "value": 300.0,
        "source": "ecos",
        "retrieved_at": datetime(2026, 1, 1, tzinfo=UTC),
        AVAILABLE_AT: close_ts,
    }], schema={"period_end": pl.Date, "value": pl.Float64, "source": pl.String, "retrieved_at": TS_DTYPE, AVAILABLE_AT: TS_DTYPE})
    return { "prices": prices, "fx": fx, "cpi": cpi, "close_ts": close_ts }


@pytest.mark.parametrize("scenario_id", ["test_panel_a_fresh_within_lag"])
def test_panel_a_fresh_within_lag(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = DataSettings(data_root=tmp_path / "data")
    panel_end = date(2026, 6, 30)
    frames = _synthetic_frames(panel_end)
    # Patch loader to return synthetic frames
    def fake_load(settings_inner):
        from src.data.schema import Dataset
        return {
            Dataset.PRICES: frames["prices"],
            Dataset.FX: frames["fx"],
            Dataset.CPI: frames["cpi"],
        }
    # Also need holdings optional; patch holdings loader to return empty
    monkeypatch.setattr("src.data.panel_freshness._load_catalog_frames", fake_load)
    # Also patch month_end helper to ensure 2026-06-30 considered
    reference_now = datetime(2026, 7, 15, 0, 0, tzinfo=UTC)
    report = resolve_catalog_panel_as_of(settings, reference_now=reference_now)
    assert report.panel_as_of.date() == date(2026, 6, 30)
    assert report.status == PanelFreshnessStatus.FRESH
    assert report.lag_days <= 62
    assert report.lag_days == (reference_now.date() - date(2026, 6, 30)).days


@pytest.mark.parametrize("scenario_id", ["test_panel_b_stale_over_62_days"])
def test_panel_b_stale_over_62_days(scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = DataSettings(data_root=tmp_path / "data")
    panel_end = date(2025, 4, 30)
    frames = _synthetic_frames(panel_end)
    def fake_load(settings_inner):
        from src.data.schema import Dataset
        return {Dataset.PRICES: frames["prices"], Dataset.FX: frames["fx"], Dataset.CPI: frames["cpi"]}
    monkeypatch.setattr("src.data.panel_freshness._load_catalog_frames", fake_load)
    reference_now = datetime(2026, 8, 29, 0, 0, tzinfo=UTC)
    report = resolve_catalog_panel_as_of(settings, reference_now=reference_now)
    assert report.lag_days > 62
    assert report.status == PanelFreshnessStatus.STALE


@pytest.mark.parametrize("scenario_id", ["test_panel_c_hard_stop_ack"])
def test_panel_c_hard_stop_ack(scenario_id: str) -> None:
    panel_as_of = datetime(2025, 4, 30, 20, 0, tzinfo=UTC)
    report = CatalogPanelReport(
        panel_as_of=panel_as_of,
        lag_days=100,
        status=PanelFreshnessStatus.STALE,
        ticker_last_session={t: date(2025,4,30) for t in THESIS_PANEL_TICKERS},
        cpi_last_observation=date(2025,4,30),
        fx_last_observation=date(2025,4,30),
        holdings_last_filing=None,
        hard_stop_reason=None,
    )
    hard_stop = PanelHardStop(reason="tiingo free-tier cap", max_panel_as_of=date(2025, 6, 30))
    acked = apply_hard_stop(report, hard_stop)
    assert acked.status == PanelFreshnessStatus.HARD_STOP_ACK
    assert acked.hard_stop_reason == "tiingo free-tier cap"


@pytest.mark.parametrize("scenario_id", ["test_panel_d_nport_quarter_window"])
def test_panel_d_nport_quarter_window(scenario_id: str) -> None:
    result = iter_nport_quarters_for_panel(date(2026, 8, 29), lookback_months=18)
    assert "2025q1" in result
    assert "2026q2" in result
    assert len(result) >= 6
    # no label after 2026q3 (i.e., 2026q4 should not be included)
    assert "2026q4" not in result
    assert result == tuple(sorted(result))
    # ensure ascending and within lookback window
    assert result[0] <= result[-1]


@pytest.mark.parametrize("scenario_id", ["test_panel_e_effective_end"])
def test_panel_e_effective_end(scenario_id: str) -> None:
    dt = datetime(2026, 6, 30, 20, 0, tzinfo=UTC)
    assert effective_thesis_end(dt) == date(2026, 6, 30)


@pytest.mark.parametrize("scenario_id", ["test_panel_f_naive_reference_rejected"])
def test_panel_f_naive_reference_rejected(scenario_id: str, tmp_path: Path) -> None:
    settings = DataSettings(data_root=tmp_path / "data")
    naive = datetime(2026, 7, 15, 0, 0)  # no tzinfo
    with pytest.raises(ValueError, match=r"tz|timezone"):
        resolve_catalog_panel_as_of(settings, reference_now=naive)
