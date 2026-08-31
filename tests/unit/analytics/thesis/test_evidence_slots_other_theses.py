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



def test_overlap_slot_uses_purity_when_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.analytics.thesis_evidence import EvidenceSlot, _overlap_slot
    from src.data.schema import Dataset
    from src.data.pit import stamp_availability

    # AI_POWER_BOTTLENECK with mocked purity slot
    from src.policy.thesis import Horizon, ThesisId, ThesisSpec

    thesis_power = ThesisSpec(
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
    as_of = datetime(2020, 1, 15, tzinfo=UTC)
    fake_purity = EvidenceSlot(status="computed", summary="purity: mixed aligned 50.0%", metrics={"purity_label": "mixed"})
    monkeypatch.setattr("src.analytics.purity_evidence.compute_purity_slot", lambda **kwargs: fake_purity)
    slot = _overlap_slot(thesis_power, settings, as_of)
    assert slot.summary.startswith("purity:")
    assert slot.metrics["purity_label"] == "mixed"

    # AI_COMPUTE with purity None -> classic overlap
    thesis_compute = ThesisSpec(
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
    monkeypatch.setattr("src.analytics.purity_evidence.compute_purity_slot", lambda **kwargs: None)
    spec = spec_for(Dataset.ETF_HOLDINGS)
    retrieved = datetime(2020, 1, 1, tzinfo=UTC)
    filing = datetime(2019, 12, 15, tzinfo=UTC)
    report_date = date(2019, 12, 31)
    rows = [
        {"etf_ticker": "SOXX", "report_date": report_date, "filing_date": filing, "holding_id": "X", "issuer_name": "X Inc", "cusip": "X-cusip", "isin": None, "lei": None, "weight_pct": 60.0, "value_usd": 60, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "SOXX", "report_date": report_date, "filing_date": filing, "holding_id": "Y", "issuer_name": "Y Inc", "cusip": "Y-cusip", "isin": None, "lei": None, "weight_pct": 40.0, "value_usd": 40, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "QQQ", "report_date": report_date, "filing_date": filing, "holding_id": "X", "issuer_name": "X Inc", "cusip": "X-cusip", "isin": None, "lei": None, "weight_pct": 50.0, "value_usd": 50, "source": "sec_nport", "retrieved_at": retrieved},
        {"etf_ticker": "QQQ", "report_date": report_date, "filing_date": filing, "holding_id": "Z", "issuer_name": "Z Inc", "cusip": "Z-cusip", "isin": None, "lei": None, "weight_pct": 50.0, "value_usd": 50, "source": "sec_nport", "retrieved_at": retrieved},
    ]
    df = pl.DataFrame(rows).cast(pl.Schema(dict(spec.columns)))
    stamped = stamp_availability(df, spec)

    def fake_load_visible(settings, dataset, decision_ts):
        if dataset == Dataset.ETF_HOLDINGS:
            return stamped
        raise ValueError("unexpected dataset")

    monkeypatch.setattr("src.data.catalog.load_visible", fake_load_visible)
    slot2 = _overlap_slot(thesis_compute, settings, as_of)
    assert slot2.summary.startswith("overlap")
    assert "overlap_pct" in slot2.metrics


@pytest.mark.parametrize("scenario_id", ["test_thesis_evidence_physical_automation_overlap_uses_purity"])
def test_thesis_evidence_physical_automation_overlap_uses_purity(
    scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """test_thesis_evidence_physical_automation_overlap_uses_purity"""
    from src.analytics.thesis_evidence import EvidenceSlot, _overlap_slot
    from src.data.schema import Dataset
    from src.data.pit import stamp_availability

    thesis_physical = ThesisSpec(
        id=ThesisId.PHYSICAL_AUTOMATION,
        version=1,
        title="test physical",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["commercialization_lag"],
        candidate_sleeves=["physical_automation"],
        historical_proxies=["BOTZ"],
    )
    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2020, 1, 15, tzinfo=UTC)
    fake_purity = EvidenceSlot(
        status="computed",
        summary="purity: mixed aligned 51.0%",
        metrics={
            "purity_label": "mixed",
            "industrial_weight_pct": 45.0,
            "humanoid_weight_pct": 6.0,
        },
    )
    monkeypatch.setattr("src.analytics.purity_evidence.compute_purity_slot", lambda **kwargs: fake_purity)
    slot = _overlap_slot(thesis_physical, settings, as_of)
    assert slot.summary.startswith("purity:")
    assert slot.metrics["purity_label"] == "mixed"
    assert slot.metrics["industrial_weight_pct"] == 45.0
    assert slot.metrics["humanoid_weight_pct"] == 6.0

    thesis_compute = ThesisSpec(
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
    monkeypatch.setattr("src.analytics.purity_evidence.compute_purity_slot", lambda **kwargs: None)
    spec = spec_for(Dataset.ETF_HOLDINGS)
    retrieved = datetime(2020, 1, 1, tzinfo=UTC)
    filing = datetime(2019, 12, 15, tzinfo=UTC)
    report_date = date(2019, 12, 31)
    rows = [
        {
            "etf_ticker": "SOXX",
            "report_date": report_date,
            "filing_date": filing,
            "holding_id": "X",
            "issuer_name": "X Inc",
            "cusip": "X-cusip",
            "isin": None,
            "lei": None,
            "weight_pct": 60.0,
            "value_usd": 60,
            "source": "sec_nport",
            "retrieved_at": retrieved,
        },
        {
            "etf_ticker": "SOXX",
            "report_date": report_date,
            "filing_date": filing,
            "holding_id": "Y",
            "issuer_name": "Y Inc",
            "cusip": "Y-cusip",
            "isin": None,
            "lei": None,
            "weight_pct": 40.0,
            "value_usd": 40,
            "source": "sec_nport",
            "retrieved_at": retrieved,
        },
        {
            "etf_ticker": "QQQ",
            "report_date": report_date,
            "filing_date": filing,
            "holding_id": "X",
            "issuer_name": "X Inc",
            "cusip": "X-cusip",
            "isin": None,
            "lei": None,
            "weight_pct": 50.0,
            "value_usd": 50,
            "source": "sec_nport",
            "retrieved_at": retrieved,
        },
        {
            "etf_ticker": "QQQ",
            "report_date": report_date,
            "filing_date": filing,
            "holding_id": "Z",
            "issuer_name": "Z Inc",
            "cusip": "Z-cusip",
            "isin": None,
            "lei": None,
            "weight_pct": 50.0,
            "value_usd": 50,
            "source": "sec_nport",
            "retrieved_at": retrieved,
        },
    ]
    df = pl.DataFrame(rows).cast(pl.Schema(dict(spec.columns)))
    stamped = stamp_availability(df, spec)

    def fake_load_visible(settings, dataset, decision_ts):
        if dataset == Dataset.ETF_HOLDINGS:
            return stamped
        raise ValueError("unexpected dataset")

    monkeypatch.setattr("src.data.catalog.load_visible", fake_load_visible)
    slot2 = _overlap_slot(thesis_compute, settings, as_of)
    assert slot2.summary.startswith("overlap")
    assert "overlap_pct" in slot2.metrics


@pytest.mark.parametrize("scenario_id", ["test_thesis_evidence_physical_automation_valuation_crowding_not_unknown"])
def test_thesis_evidence_physical_automation_valuation_crowding_not_unknown(
    scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """test_thesis_evidence_physical_automation_valuation_crowding_not_unknown"""
    from src.analytics.thesis_evidence import compute_evidence_vector
    from src.data.calendar import load_calendar
    from src.data.pit import stamp_availability
    from src.data.schema import Dataset

    thesis = ThesisSpec(
        id=ThesisId.PHYSICAL_AUTOMATION,
        version=1,
        title="test physical",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["commercialization_lag"],
        candidate_sleeves=["physical_automation"],
        historical_proxies=["BOTZ"],
    )
    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2025, 4, 30, tzinfo=UTC)

    cal = load_calendar("XNYS")
    sessions = cal.sessions(date(2020, 1, 1), date(2025, 4, 30))
    n = 300
    assert len(sessions) >= n
    price_dates = list(sessions[:n])
    botz_prices = [100.0 * (1.001 ** i) for i in range(n)]
    qqq_prices = [100.0] * n
    price_retrieved = datetime(2025, 4, 29, tzinfo=UTC)
    spec_prices = spec_for(Dataset.PRICES)
    price_rows: list[dict[str, object]] = []
    for i, d in enumerate(price_dates):
        for ticker, px in [("BOTZ", botz_prices[i]), ("QQQ", qqq_prices[i])]:
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
                "etf_ticker": "BOTZ",
                "report_date": report_date,
                "filing_date": filing,
                "holding_id": f"B{i}",
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
            "series_id": ["NEWORDER"] * len(obs_dates),
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
    assert snapshot.structural.status == "computed"
    assert snapshot.structural.summary.startswith("fundamental:")
    assert snapshot.valuation.status != "unknown"
    assert snapshot.valuation.status in {"computed", "insufficient_data"}
    assert snapshot.crowding.status != "unknown"
    assert snapshot.crowding.status in {"computed", "insufficient_data"}
    if snapshot.valuation.status == "computed":
        assert snapshot.valuation.summary.startswith("valuation:")
        assert snapshot.valuation.metrics["vehicle_ticker"] == "BOTZ"
    if snapshot.crowding.status == "computed":
        assert snapshot.crowding.summary.startswith("crowding:")
        assert snapshot.crowding.metrics["vehicle_ticker"] == "BOTZ"


@pytest.mark.parametrize("scenario_id", ["test_thesis_evidence_physical_automation_structural_not_unknown"])
def test_thesis_evidence_physical_automation_structural_not_unknown(
    scenario_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """test_thesis_evidence_physical_automation_structural_not_unknown"""
    from src.analytics.thesis_evidence import compute_evidence_vector

    thesis = ThesisSpec(
        id=ThesisId.PHYSICAL_AUTOMATION,
        version=1,
        title="test physical",
        status="research",
        horizon=Horizon(min_years=5, target_years=10),
        causal_chain=["a"],
        falsifiers=["commercialization_lag"],
        candidate_sleeves=["physical_automation"],
        historical_proxies=["BOTZ"],
    )
    settings = DataSettings(data_root=tmp_path / "data")
    as_of = datetime(2025, 4, 30, tzinfo=UTC)
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
            "series_id": ["NEWORDER"] * len(obs_dates),
            "observation_date": obs_dates,
            "release_date": [release] * len(obs_dates),
            "value": values,
        }
    )

    def fake_struct_load(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:
        if dataset == Dataset.MACRO:
            return macro
        raise ValueError("unexpected dataset")

    def fake_catalog(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:
        if dataset == Dataset.MACRO:
            return macro
        if dataset == Dataset.ETF_HOLDINGS:
            raise ValueError("no holdings")
        return pl.DataFrame()

    def fake_val(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:
        raise ValueError("no prices")

    def fake_crd(settings: DataSettings, dataset: Dataset, decision_ts: datetime) -> pl.DataFrame:
        raise ValueError("no holdings")

    monkeypatch.setattr("src.analytics.structural_evidence.load_visible", fake_struct_load)
    monkeypatch.setattr("src.data.catalog.load_visible", fake_catalog)
    monkeypatch.setattr("src.analytics.valuation_evidence.load_visible", fake_val)
    monkeypatch.setattr("src.analytics.crowding_evidence.load_visible", fake_crd)

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
    assert snapshot.structural.status == "computed"
    assert snapshot.structural.summary.startswith("fundamental:")
    assert snapshot.valuation.status != "unknown"
    assert snapshot.crowding.status != "unknown"
    assert snapshot.valuation.status in {"computed", "insufficient_data"}
    assert snapshot.crowding.status in {"computed", "insufficient_data"}
