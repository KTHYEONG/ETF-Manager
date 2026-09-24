"""Schema and session guards for the Korean ETF price dataset."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.data.calendar import load_calendar
from src.data.pension_market import import_pension_etf_history, load_pension_etf_identities
from src.data.pipeline import persist_ingest
from src.data.quality import DataQualityError
from src.data.query import load_as_of
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import RawPayload

_RETRIEVED_AT = datetime(2024, 2, 14, 5, 0, tzinfo=UTC)
_SOURCE_URLS = ("https://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd?vsView=Y",)
_IDENTITY_PATH = str(Path(__file__).resolve().parents[3] / "configs" / "data" / "pension_etfs_2026.json")


def _kr_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    spec = spec_for(Dataset.KR_ETF_PRICES)
    data = {
        "ticker": [row["ticker"] for row in rows],
        "date": [row["date"] for row in rows],
        "close_krw": [row["close_krw"] for row in rows],
        "nav_krw": [row["nav_krw"] for row in rows],
        "distribution_krw": [row["distribution_krw"] for row in rows],
        "distribution_pay_date": [row.get("distribution_pay_date") for row in rows],
        "split_factor": [row["split_factor"] for row in rows],
        "volume": [row["volume"] for row in rows],
        "source": [_SOURCE_URLS[0]] * len(rows),
        "retrieved_at": [_RETRIEVED_AT] * len(rows),
    }
    return pl.DataFrame(data, schema=dict(spec.columns))


def _payload() -> RawPayload:
    return RawPayload(
        provider="krx",
        endpoint=_SOURCE_URLS[0],
        request_params={"source_urls": list(_SOURCE_URLS)},
        retrieved_at=_RETRIEVED_AT,
        extension="csv",
        content=b"ticker,date\n",
    )


def _isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataSettings:
    root = tmp_path / "data_root"
    root.mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    return DataSettings(data_root="data_root")


def test_korean_session_close_visibility(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A row on the session after an XKRX holiday is visible only from the Korean close."""
    settings = _isolated_settings(tmp_path, monkeypatch)
    calendar = load_calendar("XKRX")
    assert calendar.is_session(date(2024, 2, 9)) is False
    assert calendar.is_session(date(2024, 2, 13)) is True
    frame = _kr_frame(
        [
            {
                "ticker": "379800",
                "date": date(2024, 2, 13),
                "close_krw": 10000.0,
                "nav_krw": 10010.0,
                "distribution_krw": 0.0,
                "distribution_pay_date": None,
                "split_factor": 1.0,
                "volume": 1000,
            }
        ]
    )
    artifact = persist_ingest(frame, Dataset.KR_ETF_PRICES, _payload(), settings, calendar_name="XKRX")
    expected_close = calendar.close_ts(date(2024, 2, 13))
    stored = pl.read_parquet(artifact.normalized_path)
    assert stored.get_column("available_at").to_list() == [expected_close]
    assert load_as_of(stored, Dataset.KR_ETF_PRICES, expected_close - timedelta(seconds=1)).is_empty()
    assert load_as_of(stored, Dataset.KR_ETF_PRICES, expected_close).height == 1


def test_missing_distribution_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An uncertified distribution amount is never accepted as a normalized row."""
    settings = _isolated_settings(tmp_path, monkeypatch)
    frame = _kr_frame(
        [
            {
                "ticker": "379800",
                "date": date(2024, 1, 30),
                "close_krw": 10000.0,
                "nav_krw": 10010.0,
                "distribution_krw": None,
                "distribution_pay_date": None,
                "split_factor": 1.0,
                "volume": 1000,
            }
        ]
    )
    with pytest.raises(DataQualityError):
        persist_ingest(frame, Dataset.KR_ETF_PRICES, _payload(), settings, calendar_name="XKRX")
    assert not (tmp_path / "data_root" / "normalized").exists()


def _bad_row_case(kind: str) -> list[dict[str, object]]:
    base = {
        "ticker": "379800",
        "date": date(2024, 1, 30),
        "close_krw": 10000.0,
        "nav_krw": 10010.0,
        "distribution_krw": 0.0,
        "distribution_pay_date": None,
        "split_factor": 1.0,
        "volume": 1000,
    }
    if kind == "duplicate":
        second = dict(base, date=date(2024, 1, 30))
        return [base, second]
    if kind == "nonpositive_nav":
        return [dict(base, nav_krw=0.0)]
    if kind == "negative_distribution":
        return [dict(base, distribution_krw=-5.0)]
    if kind == "zero_dist_with_pay":
        return [dict(base, distribution_pay_date=date(2024, 2, 1))]
    if kind == "paid_without_pay_date":
        return [dict(base, distribution_krw=50.0, distribution_pay_date=None)]
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["duplicate", "nonpositive_nav", "negative_distribution", "zero_dist_with_pay", "paid_without_pay_date"])
def test_bad_market_rows_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    """Duplicates, nonpositive NAV, and negative distributions each fail closed."""
    settings = _isolated_settings(tmp_path, monkeypatch)
    frame = _kr_frame(_bad_row_case(kind))
    with pytest.raises(DataQualityError):
        persist_ingest(frame, Dataset.KR_ETF_PRICES, _payload(), settings, calendar_name="XKRX")


def test_prelisting_row_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pre-listing observation is rejected at the identity boundary."""
    settings = _isolated_settings(tmp_path, monkeypatch)
    identities = load_pension_etf_identities(_IDENTITY_PATH)
    listing = next(item for item in identities if item.ticker == "379800").listing_date
    early = (listing - timedelta(days=1)).isoformat()
    csv_path = tmp_path / "early.csv"
    csv_path.write_text(
        "ticker,date,close_krw,nav_krw,distribution_krw,distribution_pay_date,split_factor,volume\n"
        f"379800,{early},10000,10010,0,,1.0,1000\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="precedes listing"):
        import_pension_etf_history(
            csv_path, settings, source_urls=_SOURCE_URLS, retrieved_at=_RETRIEVED_AT, identities=identities
        )
