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
    load_panel_hard_stop,
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


def test_load_panel_hard_stop_default_path_without_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default registry path resolves against cwd; absent file yields None."""
    monkeypatch.chdir(tmp_path)
    assert load_panel_hard_stop() is None


def _persist_panel_frames(
    settings: DataSettings,
    sessions: tuple[date, ...],
    tickers: tuple[str, ...] = THESIS_PANEL_TICKERS,
    *,
    close: float = 100.0,
    fx_rate: float = 1300.0,
    retrieved_at: datetime = datetime(2024, 1, 5, 5, 0, tzinfo=UTC),
) -> None:
    import polars as pl

    from src.data.pipeline import persist_ingest
    from src.data.schema import Dataset, spec_for
    from src.data.storage import RawPayload

    def _payload() -> RawPayload:
        return RawPayload(
            provider="synthetic",
            endpoint="probe",
            request_params={},
            retrieved_at=retrieved_at,
            extension="json",
            content=b"{}",
        )

    spec_prices = spec_for(Dataset.PRICES)
    rows = [(ticker, day) for ticker in tickers for day in sessions]
    persist_ingest(
        pl.DataFrame(
            {
                "ticker": [ticker for ticker, _ in rows],
                "date": [day for _, day in rows],
                "open": [close * 0.98] * len(rows),
                "high": [close * 1.02] * len(rows),
                "low": [close * 0.97] * len(rows),
                "close": [close] * len(rows),
                "volume": [10_000] * len(rows),
                "adjusted_close": [close] * len(rows),
                "dividend": [0.0] * len(rows),
                "split_factor": [1.0] * len(rows),
                "source": ["synthetic"] * len(rows),
                "retrieved_at": [retrieved_at] * len(rows),
            },
            schema=dict(spec_prices.columns),
        ),
        Dataset.PRICES,
        _payload(),
        settings,
    )
    spec_fx = spec_for(Dataset.FX)
    persist_ingest(
        pl.DataFrame(
            {
                "date": list(sessions),
                "usdkrw": [fx_rate] * len(sessions),
                "source": ["synthetic"] * len(sessions),
                "retrieved_at": [retrieved_at] * len(sessions),
            },
            schema=dict(spec_fx.columns),
        ),
        Dataset.FX,
        _payload(),
        settings,
    )
    spec_cpi = spec_for(Dataset.CPI)
    persist_ingest(
        pl.DataFrame(
            {
                "period_end": [date(2023, 11, 1)],
                "value": [300.0],
                "source": ["synthetic"],
                "retrieved_at": [retrieved_at],
            },
            schema=dict(spec_cpi.columns),
        ),
        Dataset.CPI,
        _payload(),
        settings,
    )


def test_resolve_catalog_panel_coverage_and_read_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coverage and pinned reads name the same manifest set across a later ingest."""
    from pathlib import Path as _Path

    import polars as _pl

    from src.data.calendar import load_calendar
    from src.data.catalog import load_snapshot_visible, resolve_snapshot
    from src.data.schema import Dataset

    monkeypatch.chdir(tmp_path)
    settings = DataSettings(data_root="data")
    sessions = load_calendar("XNYS").sessions(date(2024, 1, 2), date(2024, 1, 31))
    _persist_panel_frames(settings, sessions)
    reference_now = datetime(2024, 2, 15, tzinfo=UTC)
    report = resolve_catalog_panel_as_of(settings, reference_now=reference_now)
    assert report.status in (PanelFreshnessStatus.FRESH, PanelFreshnessStatus.STALE)
    snapshot = resolve_snapshot(settings, (Dataset.PRICES, Dataset.FX, Dataset.CPI))
    before = {dataset: _Path(artifact.manifest_path).stem for dataset, artifact in snapshot.artifacts.items()}

    _persist_panel_frames(
        settings,
        sessions,
        close=999.0,
        fx_rate=9999.0,
        retrieved_at=datetime(2024, 2, 2, 5, 0, tzinfo=UTC),
    )

    pinned = load_snapshot_visible(snapshot, Dataset.PRICES, report.panel_as_of)
    assert pinned.filter(_pl.col("adjusted_close") == 999.0).is_empty()
    assert before == {
        dataset: _Path(artifact.manifest_path).stem for dataset, artifact in snapshot.artifacts.items()
    }


def test_resolve_catalog_panel_damaged_source_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corrupt required Silver raises instead of returning healthy panel coverage."""
    from pathlib import Path as _Path

    from src.data.calendar import load_calendar
    from src.data.catalog import latest_artifact
    from src.data.schema import Dataset
    from src.data.storage import UntrustedDatasetError

    monkeypatch.chdir(tmp_path)
    settings = DataSettings(data_root="data")
    sessions = load_calendar("XNYS").sessions(date(2024, 1, 2), date(2024, 1, 31))
    _persist_panel_frames(settings, sessions)
    _Path(latest_artifact(settings, Dataset.FX).normalized_path).unlink()
    with pytest.raises(UntrustedDatasetError, match=r"missing|unreadable|mismatch|manifest|parquet|required"):
        resolve_catalog_panel_as_of(settings, reference_now=datetime(2024, 2, 15, tzinfo=UTC))


def test_resolve_catalog_panel_absent_source_is_insufficient(tmp_path: Path) -> None:
    """A catalog with no manifests reports INSUFFICIENT_DATA instead of a trust failure."""
    settings = DataSettings(data_root=tmp_path / "data")
    report = resolve_catalog_panel_as_of(settings, reference_now=datetime(2024, 2, 15, tzinfo=UTC))
    assert report.status is PanelFreshnessStatus.INSUFFICIENT_DATA
    assert report.ticker_last_session == {}
