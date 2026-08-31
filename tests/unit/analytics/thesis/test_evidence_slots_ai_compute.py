"""Tests for thesis evidence vector."""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.policy.thesis import ThesisId, ThesisSpec, Horizon
from src.sim.allocation import AllocationConfig, AllocationResult



def test_ev_structural_not_unknown_ai_compute(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.analytics.thesis_evidence import compute_evidence_vector

    thesis = ThesisSpec(
        id=ThesisId.AI_COMPUTE,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["f1"],
        candidate_sleeves=["ai_semiconductor"],
        historical_proxies=["SOXX"],
    )
    settings = DataSettings(data_root=Path("data"))
    as_of = datetime(2025, 4, 30, tzinfo=UTC)
    release = datetime(2025, 4, 29, 12, 0, tzinfo=UTC)
    obs_dates = [
        date(2015, 3, 31),
        date(2015, 6, 30),
        date(2015, 9, 30),
        date(2015, 12, 31),
        date(2016, 3, 31),
        date(2016, 6, 30),
        date(2016, 9, 30),
        date(2016, 12, 31),
        date(2017, 3, 31),
        date(2017, 6, 30),
        date(2017, 9, 30),
        date(2017, 12, 31),
    ]
    macro = pl.DataFrame(
        {
            "series_id": ["PNFI"] * 12,
            "observation_date": obs_dates,
            "release_date": [release] * 12,
            "value": [3000.0 + i * 50.0 for i in range(12)],
        }
    )

    def fake_catalog_load(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:
        if dataset == Dataset.ETF_HOLDINGS:
            raise ValueError("no holdings")
        return pl.DataFrame()

    def fake_struct_load(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:
        if dataset == Dataset.MACRO:
            return macro
        raise ValueError("no holdings")

    def fake_val_load(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:
        raise ValueError("no prices")

    def fake_crd_load(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:
        raise ValueError("no holdings")

    monkeypatch.setattr("src.data.catalog.load_visible", fake_catalog_load)
    monkeypatch.setattr("src.analytics.structural_evidence.load_visible", fake_struct_load)
    monkeypatch.setattr("src.analytics.valuation_evidence.load_visible", fake_val_load)
    monkeypatch.setattr("src.analytics.crowding_evidence.load_visible", fake_crd_load)

    def runner(config: AllocationConfig) -> AllocationResult:
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=100.0,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=100.0,
            xirr_real=0.0,
        )

    snapshot = compute_evidence_vector(thesis=thesis, settings=settings, as_of=as_of, runner=runner)
    assert snapshot.structural.status in ("computed", "insufficient_data")
    assert snapshot.structural.status != "unknown"
    assert snapshot.valuation.status in ("insufficient_data", "computed")
    assert snapshot.crowding.status in ("insufficient_data", "computed")


def test_ev_market_regime_not_structural(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.analytics.thesis_evidence import compute_evidence_vector
    from src.analytics.thesis_evidence import EvidenceSlot

    thesis = ThesisSpec(
        id=ThesisId.AI_COMPUTE,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["f1"],
        candidate_sleeves=["ai_semiconductor"],
        historical_proxies=["SOXX"],
    )
    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2025, 4, 30, tzinfo=UTC)

    def fake_regime(*args, **kwargs):
        return EvidenceSlot(status="computed", summary="regime proxy computed", metrics={"windows_tested": 3})

    monkeypatch.setattr("src.analytics.regime_proxy.compute_regime_proxy_slot", fake_regime)

    def fake_catalog(settings, dataset, decision_ts):
        if dataset == Dataset.ETF_HOLDINGS:
            raise ValueError("no holdings")
        return pl.DataFrame()

    def fake_val(settings, dataset, decision_ts):
        raise ValueError("no prices")

    def fake_crd(settings, dataset, decision_ts):
        raise ValueError("no holdings")

    def fake_struct(settings, dataset, decision_ts):
        raise ValueError("no macro")

    monkeypatch.setattr("src.data.catalog.load_visible", fake_catalog)
    monkeypatch.setattr("src.analytics.valuation_evidence.load_visible", fake_val)
    monkeypatch.setattr("src.analytics.crowding_evidence.load_visible", fake_crd)
    monkeypatch.setattr("src.analytics.structural_evidence.load_visible", fake_struct)

    def runner(config: AllocationConfig) -> AllocationResult:
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=100.0,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=100.0,
            xirr_real=0.0,
        )

    snapshot = compute_evidence_vector(thesis=thesis, settings=settings, as_of=as_of, runner=runner, include_regime=True)
    assert snapshot.market_regime.status in ("computed", "insufficient_data")
    assert snapshot.structural.status in ("computed", "insufficient_data")
    assert snapshot.structural.status != "unknown"
    assert snapshot.valuation.status in ("insufficient_data", "computed")
    assert snapshot.crowding.status in ("insufficient_data", "computed")
    if snapshot.structural.status == "computed":
        assert snapshot.structural.summary.startswith("fundamental:")
    else:
        assert "error" in snapshot.structural.metrics


def test_ev_valuation_crowding_ai_compute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.analytics.thesis_evidence import compute_evidence_vector

    thesis = ThesisSpec(
        id=ThesisId.AI_COMPUTE,
        version=1,
        title="test",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["f1"],
        candidate_sleeves=["ai_semiconductor"],
        historical_proxies=["SOXX"],
    )
    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2025, 4, 30, tzinfo=UTC)
    # Build PRICES: SOXX and QQQ 300 sessions
    from src.data.schema import Dataset
    from src.data.calendar import load_calendar
    from src.data.pit import stamp_availability

    spec_prices = spec_for(Dataset.PRICES)
    spec_holdings = spec_for(Dataset.ETF_HOLDINGS)
    n = 300
    start = date(2020, 1, 1)
    # For calibration to avoid NotSessionError, use valid session dates
    cal = load_calendar("XNYS")
    sessions = cal.sessions(date(2020, 1, 1), date(2021, 12, 31))
    assert len(sessions) >= n
    price_dates = list(sessions[:n])
    # SOXX rising, QQQ flat
    soxx_prices = [100 * (1.001 ** i) for i in range(n)]
    qqq_prices = [100.0] * n
    price_rows = []
    retrieved = datetime(2021, 12, 30, tzinfo=UTC)
    for i, d in enumerate(price_dates):
        for ticker, px in [("SOXX", soxx_prices[i]), ("QQQ", qqq_prices[i])]:
            price_rows.append(
                {
                    "ticker": ticker,
                    "date": d,
                    "open": px,
                    "high": px,
                    "low": px,
                    "close": px,
                    "volume": 1000,
                    "adjusted_close": px,
                    "dividend": 0.0,
                    "split_factor": 1.0,
                    "source": "test",
                    "retrieved_at": retrieved,
                }
            )
    prices_df = pl.DataFrame(price_rows).cast(pl.Schema(dict(spec_prices.columns)))
    prices_stamped = stamp_availability(prices_df, spec_prices, cal)

    # ETF_HOLDINGS: concentrated SOXX snapshot + QQQ for overlap
    report_date = price_dates[-1]
    filing = datetime(2021, 12, 15, tzinfo=UTC)
    weights_soxx = [35.0, 17.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]
    weights_qqq = [20.0, 20.0, 20.0, 20.0, 20.0]
    hold_rows = []
    for i, w in enumerate(weights_soxx):
        hold_rows.append(
            {
                "etf_ticker": "SOXX",
                "report_date": report_date,
                "filing_date": filing,
                "holding_id": f"S{i}",
                "issuer_name": f"S Issuer {i}",
                "cusip": f"CUSIP_S{i}",
                "isin": None,
                "lei": None,
                "weight_pct": w,
                "value_usd": w * 10,
                "source": "sec_nport",
                "retrieved_at": retrieved,
            }
        )
    for i, w in enumerate(weights_qqq):
        hold_rows.append(
            {
                "etf_ticker": "QQQ",
                "report_date": report_date,
                "filing_date": filing,
                "holding_id": f"Q{i}",
                "issuer_name": f"Q Issuer {i}",
                "cusip": f"CUSIP_Q{i}",
                "isin": None,
                "lei": None,
                "weight_pct": w,
                "value_usd": w * 10,
                "source": "sec_nport",
                "retrieved_at": retrieved,
            }
        )
    holdings_df = pl.DataFrame(hold_rows).cast(pl.Schema(dict(spec_holdings.columns)))
    holdings_stamped = stamp_availability(holdings_df, spec_holdings)

    # MACRO for structural
    release = datetime(2025, 4, 29, 12, 0, tzinfo=UTC)
    obs_dates = [date(2015, 3, 31), date(2015, 6, 30), date(2015, 9, 30), date(2015, 12, 31), date(2016, 3, 31), date(2016, 6, 30), date(2016, 9, 30), date(2016, 12, 31), date(2017, 3, 31), date(2017, 6, 30), date(2017, 9, 30), date(2017, 12, 31)]
    macro = pl.DataFrame({"series_id": ["PNFI"] * 12, "observation_date": obs_dates, "release_date": [release] * 12, "value": [3000.0 + i * 50.0 for i in range(12)]})

    def fake_catalog(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:
        if dataset == Dataset.PRICES:
            return prices_stamped
        if dataset == Dataset.ETF_HOLDINGS:
            return holdings_stamped
        if dataset == Dataset.MACRO:
            return macro
        return pl.DataFrame()

    def fake_struct_load(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:
        if dataset == Dataset.MACRO:
            return macro
        raise ValueError("unexpected")

    def fake_val_load(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:
        if dataset == Dataset.PRICES:
            return prices_stamped
        raise ValueError("unexpected")

    def fake_crd_load(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:
        if dataset == Dataset.ETF_HOLDINGS:
            return holdings_stamped
        raise ValueError("unexpected")

    monkeypatch.setattr("src.data.catalog.load_visible", fake_catalog)
    monkeypatch.setattr("src.analytics.structural_evidence.load_visible", fake_struct_load)
    monkeypatch.setattr("src.analytics.valuation_evidence.load_visible", fake_val_load)
    monkeypatch.setattr("src.analytics.crowding_evidence.load_visible", fake_crd_load)

    def runner(config) -> AllocationResult:
        return AllocationResult(config=config, snapshots=(), terminal_wealth_krw=100.0, xirr=0.0, max_drawdown=0.0, terminal_wealth_real_krw=100.0, xirr_real=0.0)

    snapshot = compute_evidence_vector(thesis=thesis, settings=settings, as_of=as_of, runner=runner)
    assert snapshot.valuation.status in ("computed", "insufficient_data")
    assert snapshot.crowding.status in ("computed", "insufficient_data")
    assert snapshot.valuation.status != "unknown"
    assert snapshot.crowding.status != "unknown"
    assert snapshot.structural.status != "unknown"


def test_thesis_evidence_ai_power_structural_not_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.analytics.thesis_evidence import compute_evidence_vector
    from src.data.settings import DataSettings

    thesis = ThesisSpec(
        id=ThesisId.AI_POWER_BOTTLENECK,
        version=1,
        title="test power",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["backlog_normalization"],
        candidate_sleeves=["ai_power_equipment"],
        historical_proxies=["GRID"],
    )
    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2025, 4, 30, tzinfo=UTC)
    release = datetime(2025, 4, 29, 12, 0, tzinfo=UTC)
    obs_dates = [date(2023, 1, 15), date(2023, 2, 15), date(2023, 3, 15), date(2023, 4, 15), date(2023, 5, 15), date(2023, 6, 15), date(2023, 7, 15), date(2023, 8, 15), date(2023, 9, 15), date(2023, 10, 15), date(2023, 11, 15), date(2023, 12, 15), date(2024, 1, 15), date(2024, 2, 15), date(2024, 3, 15), date(2024, 4, 15), date(2024, 5, 15), date(2024, 6, 15), date(2024, 7, 15), date(2024, 8, 15)]
    values = [100.0 + i * 5.0 for i in range(len(obs_dates))]
    macro = pl.DataFrame(
        {
            "series_id": ["A35SNO"] * len(obs_dates),
            "observation_date": obs_dates,
            "release_date": [release] * len(obs_dates),
            "value": values,
        }
    )

    def fake_struct_load(settings, dataset, decision_ts):
        if dataset == Dataset.MACRO:
            return macro
        raise ValueError("no holdings")

    def fake_catalog(settings, dataset, decision_ts):
        if dataset == Dataset.MACRO:
            return macro
        if dataset == Dataset.ETF_HOLDINGS:
            raise ValueError("no holdings")
        return pl.DataFrame()

    def fake_val(settings, dataset, decision_ts):
        raise ValueError("no prices")

    def fake_crd(settings, dataset, decision_ts):
        raise ValueError("no holdings")

    monkeypatch.setattr("src.analytics.structural_evidence.load_visible", fake_struct_load)
    monkeypatch.setattr("src.data.catalog.load_visible", fake_catalog)
    monkeypatch.setattr("src.analytics.valuation_evidence.load_visible", fake_val)
    monkeypatch.setattr("src.analytics.crowding_evidence.load_visible", fake_crd)

    def runner(config: AllocationConfig) -> AllocationResult:
        return AllocationResult(config=config, snapshots=(), terminal_wealth_krw=100.0, xirr=0.0, max_drawdown=0.0, terminal_wealth_real_krw=100.0, xirr_real=0.0)

    snapshot = compute_evidence_vector(thesis=thesis, settings=settings, as_of=as_of, runner=runner)
    assert snapshot.structural.status == "computed"
    assert snapshot.structural.status != "unknown"
    assert snapshot.valuation.status == "unknown" or snapshot.valuation.status in ("insufficient_data", "unknown")
    # valuation remains unknown until slice 2 registry blocks exist, but allow insufficient_data as well; main check is structural computed
    assert snapshot.structural.summary.startswith("fundamental:")


def test_thesis_evidence_ai_power_valuation_crowding_not_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.analytics.thesis_evidence import compute_evidence_vector
    from src.data.calendar import load_calendar
    from src.data.pit import stamp_availability
    from src.data.schema import Dataset

    thesis = ThesisSpec(
        id=ThesisId.AI_POWER_BOTTLENECK,
        version=1,
        title="test power",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["backlog_normalization"],
        candidate_sleeves=["ai_power_equipment"],
        historical_proxies=["GRID"],
    )
    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2025, 4, 30, tzinfo=UTC)

    cal = load_calendar("XNYS")
    sessions = cal.sessions(date(2020, 1, 1), date(2025, 4, 30))
    n = 300
    assert len(sessions) >= n
    price_dates = list(sessions[:n])
    grid_prices = [100.0 * (1.001 ** i) for i in range(n)]
    qqq_prices = [100.0] * n
    price_retrieved = datetime(2025, 4, 29, tzinfo=UTC)
    spec_prices = spec_for(Dataset.PRICES)
    price_rows: list[dict[str, object]] = []
    for i, d in enumerate(price_dates):
        for ticker, px in [("GRID", grid_prices[i]), ("QQQ", qqq_prices[i])]:
            price_rows.append(
                {
                    "ticker": ticker,
                    "date": d,
                    "open": px,
                    "high": px,
                    "low": px,
                    "close": px,
                    "volume": 1000,
                    "adjusted_close": px,
                    "dividend": 0.0,
                    "split_factor": 1.0,
                    "source": "test",
                    "retrieved_at": price_retrieved,
                }
            )
    prices_df = pl.DataFrame(price_rows).cast(pl.Schema(dict(spec_prices.columns)))
    prices_stamped = stamp_availability(prices_df, spec_prices, cal)

    spec_holdings = spec_for(Dataset.ETF_HOLDINGS)
    holdings_retrieved = datetime(2025, 4, 29, tzinfo=UTC)
    filing = datetime(2025, 4, 15, tzinfo=UTC)
    report_date = price_dates[-1]
    weights = [35.0, 17.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]
    hold_rows: list[dict[str, object]] = []
    for i, w in enumerate(weights):
        hold_rows.append(
            {
                "etf_ticker": "GRID",
                "report_date": report_date,
                "filing_date": filing,
                "holding_id": f"G{i}",
                "issuer_name": f"Issuer{i}",
                "cusip": f"CUSIP{i}",
                "isin": None,
                "lei": None,
                "weight_pct": w,
                "value_usd": w * 10,
                "source": "sec_nport",
                "retrieved_at": holdings_retrieved,
            }
        )
    holdings_df = pl.DataFrame(hold_rows).cast(pl.Schema(dict(spec_holdings.columns)))
    holdings_stamped = stamp_availability(holdings_df, spec_holdings)

    release = datetime(2025, 4, 29, 12, 0, tzinfo=UTC)
    obs_dates = [
        date(2023, 1, 15),
        date(2023, 2, 15),
        date(2023, 3, 15),
        date(2023, 4, 15),
        date(2023, 5, 15),
        date(2023, 6, 15),
        date(2023, 7, 15),
        date(2023, 8, 15),
        date(2023, 9, 15),
        date(2023, 10, 15),
        date(2023, 11, 15),
        date(2023, 12, 15),
        date(2024, 1, 15),
        date(2024, 2, 15),
        date(2024, 3, 15),
        date(2024, 4, 15),
        date(2024, 5, 15),
        date(2024, 6, 15),
        date(2024, 7, 15),
        date(2024, 8, 15),
    ]
    values = [100.0 + i * 5.0 for i in range(len(obs_dates))]
    macro = pl.DataFrame(
        {
            "series_id": ["A35SNO"] * len(obs_dates),
            "observation_date": obs_dates,
            "release_date": [release] * len(obs_dates),
            "value": values,
        }
    )

    def fake_catalog(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:  # type: ignore[no-untyped-def]
        if dataset == Dataset.PRICES:
            return prices_stamped
        if dataset == Dataset.ETF_HOLDINGS:
            return holdings_stamped
        if dataset == Dataset.MACRO:
            return macro
        return pl.DataFrame()

    def fake_struct_load(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:  # type: ignore[no-untyped-def]
        if dataset == Dataset.MACRO:
            return macro
        raise ValueError("unexpected")

    def fake_val_load(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:  # type: ignore[no-untyped-def]
        if dataset == Dataset.PRICES:
            return prices_stamped
        raise ValueError("unexpected")

    def fake_crd_load(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:  # type: ignore[no-untyped-def]
        if dataset == Dataset.ETF_HOLDINGS:
            return holdings_stamped
        raise ValueError("unexpected")

    monkeypatch.setattr("src.data.catalog.load_visible", fake_catalog)
    monkeypatch.setattr("src.analytics.structural_evidence.load_visible", fake_struct_load)
    monkeypatch.setattr("src.analytics.valuation_evidence.load_visible", fake_val_load)
    monkeypatch.setattr("src.analytics.crowding_evidence.load_visible", fake_crd_load)

    def runner(config: AllocationConfig) -> AllocationResult:  # type: ignore[no-untyped-def]
        return AllocationResult(
            config=config,
            snapshots=(),
            terminal_wealth_krw=100.0,
            xirr=0.0,
            max_drawdown=0.0,
            terminal_wealth_real_krw=100.0,
            xirr_real=0.0,
        )

    snapshot = compute_evidence_vector(thesis=thesis, settings=settings, as_of=as_of, runner=runner)
    assert snapshot.structural.status == "computed"
    assert snapshot.valuation.status != "unknown"
    assert snapshot.crowding.status != "unknown"
    assert snapshot.valuation.status in {"computed", "insufficient_data"}
    assert snapshot.crowding.status in {"computed", "insufficient_data"}
    if snapshot.valuation.status == "computed":
        assert snapshot.valuation.summary.startswith("valuation:")
    if snapshot.crowding.status == "computed":
        assert snapshot.crowding.summary.startswith("crowding:")


