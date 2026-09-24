"""Invariant guards for pension ETF identities and canonical history import."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.data.catalog import latest_artifact
from src.data.pension_market import import_pension_etf_history, load_pension_etf_identities
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import DataStore

_SOURCE_URLS = ("https://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd?vsView=Y",)
_RETRIEVED_AT = datetime(2024, 2, 14, 5, 0, tzinfo=UTC)
_IDENTITY_PATH = str(Path(__file__).resolve().parents[3] / "configs" / "data" / "pension_etfs_2026.json")


def _write_csv(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    header = "ticker,date,close_krw,nav_krw,distribution_krw,distribution_pay_date,split_factor,volume"
    lines = [header]
    for row in rows:
        pay = row.get("distribution_pay_date")
        lines.append(
            ",".join(
                [
                    str(row["ticker"]),
                    str(row["date"]),
                    str(row["close_krw"]),
                    str(row["nav_krw"]),
                    str(row["distribution_krw"]),
                    "" if pay is None else str(pay),
                    str(row["split_factor"]),
                    str(row["volume"]),
                ]
            )
        )
    target = tmp_path / "krx_export.csv"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataSettings:
    root = tmp_path / "data_root"
    root.mkdir()
    monkeypatch.chdir(tmp_path)
    return DataSettings(data_root="data_root")


def test_load_identities_preserve_listing_boundary() -> None:
    """A 2023-listed fund keeps its own listing date; proxies never backdate it."""
    identities = load_pension_etf_identities(_IDENTITY_PATH)
    assert len(identities) == 3
    by_ticker = {identity.ticker: identity for identity in identities}
    assert set(by_ticker) == {"379800", "379810", "469060"}
    assert by_ticker["379800"].proxy_ticker == "SPY"
    assert by_ticker["379810"].proxy_ticker == "QQQ"
    assert by_ticker["469060"].proxy_ticker == "SOXX"
    assert all(identity.pension_eligible for identity in identities)
    assert all(identity.currency_hedged is False for identity in identities)
    assert by_ticker["469060"].listing_date.year == 2023
    assert by_ticker["379800"].listing_date < by_ticker["469060"].listing_date


def test_import_preserves_official_lineage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Raw bytes, normalized hash, source, and capture time stay recoverable."""
    settings = _isolated_settings(tmp_path, monkeypatch)
    identities = load_pension_etf_identities(_IDENTITY_PATH)
    csv_path = _write_csv(
        tmp_path,
        [
            {
                "ticker": "379800",
                "date": "2024-01-30",
                "close_krw": 10000.0,
                "nav_krw": 10010.0,
                "distribution_krw": 0.0,
                "distribution_pay_date": None,
                "split_factor": 1.0,
                "volume": 1000,
            },
            {
                "ticker": "379800",
                "date": "2024-01-31",
                "close_krw": 10100.0,
                "nav_krw": 10105.0,
                "distribution_krw": 0.0,
                "distribution_pay_date": None,
                "split_factor": 1.0,
                "volume": 1100,
            },
        ],
    )
    artifact = import_pension_etf_history(
        csv_path, settings, source_urls=_SOURCE_URLS, retrieved_at=_RETRIEVED_AT, identities=identities
    )
    assert artifact.manifest.raw_artifact.sha256 == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert artifact.manifest.retrieved_at == _RETRIEVED_AT
    assert artifact.manifest.request_params["source_urls"] == list(_SOURCE_URLS)
    frame = DataStore(settings).read_normalized(artifact, spec_for(Dataset.KR_ETF_PRICES))
    assert frame.get_column("source").to_list() == [f"{_SOURCE_URLS[0]}"] * 2
    assert frame.get_column("retrieved_at").to_list() == [_RETRIEVED_AT] * 2
    assert artifact.manifest.normalized_sha256 is not None


def test_import_rejects_prelisting_proxy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-listing 469060 observations fail instead of masquerading as fund returns."""
    settings = _isolated_settings(tmp_path, monkeypatch)
    identities = load_pension_etf_identities(_IDENTITY_PATH)
    listing = next(item for item in identities if item.ticker == "469060").listing_date
    early = (listing - timedelta(days=30)).isoformat()
    csv_path = _write_csv(
        tmp_path,
        [
            {
                "ticker": "469060",
                "date": early,
                "close_krw": 9000.0,
                "nav_krw": 9010.0,
                "distribution_krw": 0.0,
                "distribution_pay_date": None,
                "split_factor": 1.0,
                "volume": 500,
            }
        ],
    )
    with pytest.raises(ValueError, match="precedes listing"):
        import_pension_etf_history(
            csv_path, settings, source_urls=_SOURCE_URLS, retrieved_at=_RETRIEVED_AT, identities=identities
        )


def test_import_keeps_distribution_timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Observation date, pay date, and cash amount remain distinct fields."""
    settings = _isolated_settings(tmp_path, monkeypatch)
    identities = load_pension_etf_identities(_IDENTITY_PATH)
    csv_path = _write_csv(
        tmp_path,
        [
            {
                "ticker": "379810",
                "date": "2024-01-30",
                "close_krw": 12000.0,
                "nav_krw": 12005.0,
                "distribution_krw": 50.0,
                "distribution_pay_date": "2024-02-15",
                "split_factor": 1.0,
                "volume": 900,
            },
            {
                "ticker": "379810",
                "date": "2024-01-31",
                "close_krw": 12100.0,
                "nav_krw": 12105.0,
                "distribution_krw": 0.0,
                "distribution_pay_date": None,
                "split_factor": 1.0,
                "volume": 950,
            },
        ],
    )
    artifact = import_pension_etf_history(
        csv_path, settings, source_urls=_SOURCE_URLS, retrieved_at=_RETRIEVED_AT, identities=identities
    )
    frame = DataStore(settings).read_normalized(artifact, spec_for(Dataset.KR_ETF_PRICES))
    paid = frame.filter(pl.col("distribution_krw") > 0).to_dicts()[0]
    assert paid["date"] == date(2024, 1, 30)
    assert paid["distribution_pay_date"] == date(2024, 2, 15)
    assert paid["distribution_krw"] == 50.0


def test_import_rejects_conflicting_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A conflicting same-key row never silently replaces trusted history."""
    settings = _isolated_settings(tmp_path, monkeypatch)
    identities = load_pension_etf_identities(_IDENTITY_PATH)
    first = _write_csv(
        tmp_path,
        [
            {
                "ticker": "379800",
                "date": "2024-01-30",
                "close_krw": 10000.0,
                "nav_krw": 10010.0,
                "distribution_krw": 0.0,
                "distribution_pay_date": None,
                "split_factor": 1.0,
                "volume": 1000,
            },
            {
                "ticker": "379800",
                "date": "2024-01-31",
                "close_krw": 10100.0,
                "nav_krw": 10105.0,
                "distribution_krw": 0.0,
                "distribution_pay_date": None,
                "split_factor": 1.0,
                "volume": 1100,
            },
        ],
    )
    baseline = import_pension_etf_history(
        first, settings, source_urls=_SOURCE_URLS, retrieved_at=_RETRIEVED_AT, identities=identities
    )
    conflict = _write_csv(
        tmp_path,
        [
            {
                "ticker": "379800",
                "date": "2024-01-30",
                "close_krw": 9999.0,
                "nav_krw": 10010.0,
                "distribution_krw": 0.0,
                "distribution_pay_date": None,
                "split_factor": 1.0,
                "volume": 1000,
            },
            {
                "ticker": "379800",
                "date": "2024-01-31",
                "close_krw": 10100.0,
                "nav_krw": 10105.0,
                "distribution_krw": 0.0,
                "distribution_pay_date": None,
                "split_factor": 1.0,
                "volume": 1100,
            },
        ],
    )
    with pytest.raises(ValueError, match="conflicting revision"):
        import_pension_etf_history(
            conflict, settings, source_urls=_SOURCE_URLS, retrieved_at=_RETRIEVED_AT, identities=identities
        )
    current = latest_artifact(settings, Dataset.KR_ETF_PRICES)
    assert current.manifest.normalized_sha256 == baseline.manifest.normalized_sha256


def _base_identity_doc() -> dict[str, object]:
    return json.loads(Path(_IDENTITY_PATH).read_text(encoding="utf-8"))


def _write_identity(tmp_path: Path, doc: object) -> str:
    target = tmp_path / "identities.json"
    if isinstance(doc, bytes):
        target.write_bytes(doc)
    elif isinstance(doc, str):
        target.write_text(doc, encoding="utf-8")
    else:
        target.write_text(json.dumps(doc), encoding="utf-8")
    return str(target)


def test_identity_records_fail_closed(tmp_path: Path) -> None:
    """Unsupported, hedged, undocumented, or inconsistent records never load."""
    base = _base_identity_doc()
    assert isinstance(base, dict)
    funds = list(base["funds"])  # type: ignore[union-attr]
    caveats = base["index_caveats"]

    def fund_override(index: int, **changes: object) -> list[object]:
        mutated = [dict(item) for item in funds]  # type: ignore[union-attr]
        mutated[index].update(changes)
        return mutated

    bad_docs: list[tuple[object, str]] = [
        ([1, 2, 3], "must be an object"),
        ([dict(f) for f in funds[:2]] + [{**dict(funds[0]), "ticker": "379800", "extra": 1}], "exactly keys"),
        (fund_override(0, ticker=""), "non-empty string"),
        (fund_override(0, ticker="000000", proxy_ticker="SPY", benchmark_id="SP500"), "unsupported"),
        (fund_override(0, proxy_ticker="QQQ"), "inconsistent proxy"),
        (fund_override(0, benchmark_id="NASDAQ100"), "inconsistent benchmark"),
        (fund_override(0, pension_eligible=False), "pension eligible"),
        (fund_override(0, currency_hedged=True), "unhedged"),
        (fund_override(0, listing_date="not-a-date"), "invalid listing_date"),
        (fund_override(0, source_checked_date="not-a-date"), "invalid source_checked_date"),
        (fund_override(0, source_url="ftp://example.com/x"), "undocumented"),
        ({"no_funds": []}, "must carry a 'funds' array"),
        ({"funds": funds}, "index-break caveat"),
        ({"funds": funds[:2], "index_caveats": caveats}, "exactly three"),
        ({"funds": [dict(funds[0]), dict(funds[0]), dict(funds[2])], "index_caveats": caveats}, "duplicated"),
    ]
    for doc, pattern in bad_docs:
        with pytest.raises(ValueError, match=pattern):
            load_pension_etf_identities(_write_identity(tmp_path, doc))
    with pytest.raises(ValueError, match="unreadable"):
        load_pension_etf_identities(str(tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="not valid JSON"):
        load_pension_etf_identities(_write_identity(tmp_path, "{broken"))
    with pytest.raises(ValueError, match="must be an object or array"):
        load_pension_etf_identities(_write_identity(tmp_path, 42))


def _csv_bytes(header: str, body: str) -> bytes:
    return (header + "\n" + body + "\n").encode("utf-8")


_VALID_HEADER = "ticker,date,close_krw,nav_krw,distribution_krw,distribution_pay_date,split_factor,volume"


def test_import_format_and_lineage_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed CSV bytes, lineage gaps, and bad market values never import."""
    settings = _isolated_settings(tmp_path, monkeypatch)
    identities = load_pension_etf_identities(_IDENTITY_PATH)
    naive = datetime(2024, 2, 14, 5, 0)
    good_body = "379800,2024-01-30,10000,10010,0,,1.0,1000\n379800,2024-01-31,10100,10105,0,,1.0,1100"

    def run_case(content: bytes, **kwargs: object) -> None:
        target = tmp_path / "case.csv"
        target.write_bytes(content)
        pattern = str(kwargs.pop("pattern", "CSV|source|retrieved|identit|empty|UTF|column|market|ticker|distribution|pay|listing"))
        call_kwargs: dict[str, object] = {
            "source_urls": _SOURCE_URLS,
            "retrieved_at": _RETRIEVED_AT,
            "identities": identities,
        }
        call_kwargs.update(kwargs)
        with pytest.raises(ValueError, match=pattern):
            import_pension_etf_history(target, settings, **call_kwargs)  # type: ignore[arg-type]

    run_case(b"\xff\xfe broken", pattern="UTF-8")
    run_case(b"a\x00b", pattern="null bytes")
    run_case(_csv_bytes("ticker,date", "379800,2024-01-30"), pattern="exactly columns")
    run_case(_csv_bytes(_VALID_HEADER, ""), pattern="non-empty|unreadable|malformed|empty|unknown")
    run_case(_VALID_HEADER.encode(), pattern="non-empty")
    run_case(_csv_bytes(_VALID_HEADER, "379800,not-a-date,10000,10010,0,,1.0,1000"), pattern="malformed")
    run_case(_csv_bytes(_VALID_HEADER, "999999,2024-01-30,10000,10010,0,,1.0,1000"), pattern="unknown tickers")
    run_case(_csv_bytes(_VALID_HEADER, "379800,2024-01-30,10000,10010,,,1.0,1000"), pattern="missing distribution")
    run_case(
        _csv_bytes(_VALID_HEADER, "379800,2024-01-30,10000,10010,50,,1.0,1000"),
        pattern="needs a pay date",
    )
    run_case(
        _csv_bytes(_VALID_HEADER, "379800,2024-01-31,10000,10010,50,2024-01-30,1.0,1000"),
        pattern="precedes observation",
    )
    run_case(
        _csv_bytes(_VALID_HEADER, "379800,2024-01-30,10000,10010,0,2024-02-01,1.0,1000"),
        pattern="null pay date",
    )
    run_case(
        _csv_bytes(_VALID_HEADER, "379800,2024-01-30,10000,10010,-5,,1.0,1000"),
        pattern="negative distribution",
    )
    target = tmp_path / "ok.csv"
    target.write_bytes(_csv_bytes(_VALID_HEADER, good_body))
    with pytest.raises(ValueError, match="at least one official origin"):
        import_pension_etf_history(target, settings, source_urls=(), retrieved_at=_RETRIEVED_AT, identities=identities)
    with pytest.raises(ValueError, match="http"):
        import_pension_etf_history(
            target, settings, source_urls=("ftp://x",), retrieved_at=_RETRIEVED_AT, identities=identities
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        import_pension_etf_history(target, settings, source_urls=_SOURCE_URLS, retrieved_at=naive, identities=identities)
    with pytest.raises(ValueError, match="identities"):
        import_pension_etf_history(target, settings, source_urls=_SOURCE_URLS, retrieved_at=_RETRIEVED_AT, identities=())
    with pytest.raises(ValueError, match="unreadable"):
        import_pension_etf_history(
            tmp_path / "gone.csv", settings, source_urls=_SOURCE_URLS, retrieved_at=_RETRIEVED_AT, identities=identities
        )
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        import_pension_etf_history(
            empty, settings, source_urls=_SOURCE_URLS, retrieved_at=_RETRIEVED_AT, identities=identities
        )


def test_import_rejects_history_shrink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial re-import that drops trusted rows is rejected."""
    settings = _isolated_settings(tmp_path, monkeypatch)
    identities = load_pension_etf_identities(_IDENTITY_PATH)
    full = tmp_path / "full.csv"
    full.write_text(
        _VALID_HEADER + "\n379800,2024-01-30,10000,10010,0,,1.0,1000\n379800,2024-01-31,10100,10105,0,,1.0,1100\n",
        encoding="utf-8",
    )
    import_pension_etf_history(full, settings, source_urls=_SOURCE_URLS, retrieved_at=_RETRIEVED_AT, identities=identities)
    partial = tmp_path / "partial.csv"
    partial.write_text(_VALID_HEADER + "\n379800,2024-01-31,10100,10105,0,,1.0,1100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="shrink"):
        import_pension_etf_history(
            partial, settings, source_urls=_SOURCE_URLS, retrieved_at=_RETRIEVED_AT, identities=identities
        )
