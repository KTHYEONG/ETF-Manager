"""NPORT holdings ingest spec tests."""

from __future__ import annotations

import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from src.data.catalog import load_visible
from src.data.pipeline import persist_ingest
from src.data.providers.sec_nport import SecNportClient
from src.data.schema import Dataset, spec_for
from src.data.settings import DataSettings
from src.data.storage import DataStore, RawPayload


@pytest.mark.parametrize("scenario_id", ["NPORT-A-schema-register"])
def test_nport_a_schema_register(scenario_id: str) -> None:
    spec = spec_for(Dataset.ETF_HOLDINGS)
    assert spec.key == ("etf_ticker", "report_date", "filing_date", "holding_id")
    assert spec.availability.kind.value == "release_column"
    assert spec.availability.release_column == "filing_date"
    assert spec.missing_policy.value == "fail"
    assert spec.revisable is True


@pytest.mark.parametrize("scenario_id", ["NPORT-B-fixture-parse"])
def test_nport_b_fixture_parse(scenario_id: str) -> None:
    fixture = Path("tests/fixtures/nport/minimal_2019q4.zip")
    assert fixture.is_file(), "fixture zip missing"
    content = fixture.read_bytes()
    df = SecNportClient.parse_quarter_zip(content, filing_quarter="2019q4")
    assert df.height >= 1
    # SOXX mapped
    soxx = df.filter(pl.col("etf_ticker") == "SOXX")
    assert soxx.height >= 1
    assert soxx.get_column("weight_pct").is_finite().all()
    assert (soxx.get_column("weight_pct") >= 0).all()
    assert (soxx.get_column("weight_pct") <= 100).all()
    assert soxx.get_column("filing_date").dtype == pl.Datetime("us", "UTC")
    # tz-aware check: dtype tz is UTC
    assert soxx.get_column("filing_date").dtype.time_zone == "UTC"


@pytest.mark.parametrize("scenario_id", ["NPORT-C-pit-amendment"])
def test_nport_c_pit_amendment(scenario_id: str) -> None:
    spec = spec_for(Dataset.ETF_HOLDINGS)
    retrieved = datetime.now(UTC)
    report = date(2019, 12, 31)
    filing1 = datetime(2020, 3, 15, 12, 0, tzinfo=UTC)
    filing2 = datetime(2020, 4, 20, 12, 0, tzinfo=UTC)
    rows = [
        {
            "etf_ticker": "SOXX",
            "report_date": report,
            "filing_date": filing1,
            "holding_id": "H1",
            "issuer_name": "NVIDIA",
            "cusip": "023135106",
            "isin": None,
            "lei": None,
            "weight_pct": 5.0,
            "value_usd": 1e6,
            "source": "sec_nport",
            "retrieved_at": retrieved,
        },
        {
            "etf_ticker": "SOXX",
            "report_date": report,
            "filing_date": filing2,
            "holding_id": "H1",
            "issuer_name": "NVIDIA",
            "cusip": "023135106",
            "isin": None,
            "lei": None,
            "weight_pct": 6.0,
            "value_usd": 1.2e6,
            "source": "sec_nport",
            "retrieved_at": retrieved,
        },
    ]
    df = pl.DataFrame(rows).cast(pl.Schema(dict(spec.columns)))
    from src.data.pit import stamp_availability

    stamped = stamp_availability(df, spec)
    t_between = datetime(2020, 4, 1, tzinfo=UTC)
    t_after = datetime(2020, 5, 1, tzinfo=UTC)
    _ = load_visible  # keep import used
    from src.data.query import load_as_of

    # Use stamped frame directly via load_as_of
    assert float(load_as_of(stamped, Dataset.ETF_HOLDINGS, t_between).get_column("weight_pct")[0]) == 5.0
    assert float(load_as_of(stamped, Dataset.ETF_HOLDINGS, t_after).get_column("weight_pct")[0]) == 6.0


@pytest.mark.parametrize("scenario_id", ["NPORT-D-persist-roundtrip"])
def test_nport_d_persist_roundtrip(scenario_id: str) -> None:
    spec = spec_for(Dataset.ETF_HOLDINGS)
    retrieved = datetime.now(UTC)
    filing = datetime(2020, 3, 15, 12, 0, tzinfo=UTC)
    report = date(2019, 12, 31)
    rows = [
        {
            "etf_ticker": "SOXX",
            "report_date": report,
            "filing_date": filing,
            "holding_id": "H1",
            "issuer_name": "NVIDIA",
            "cusip": "023135106",
            "isin": None,
            "lei": None,
            "weight_pct": 5.0,
            "value_usd": 1e6,
            "source": "sec_nport",
            "retrieved_at": retrieved,
        }
    ]
    df = pl.DataFrame(rows).cast(pl.Schema(dict(spec.columns)))
    with tempfile.TemporaryDirectory() as tmp:
        settings = DataSettings(data_root=Path(tmp))
        payload = RawPayload(
            provider="sec",
            endpoint="test",
            request_params={},
            retrieved_at=retrieved,
            extension="zip",
            content=b"testzipcontent",
        )
        art = persist_ingest(df, Dataset.ETF_HOLDINGS, payload, settings)
        assert art.manifest.normalized_sha256
        assert len(art.manifest.normalized_sha256) == 64
        assert art.manifest.row_count == df.height
        reloaded = DataStore(settings).read_normalized(art, spec)
        assert reloaded.height == art.manifest.row_count


def test_nport_ingest_writes_pointer_not_second_zip(tmp_path: Path) -> None:
    from src.data.nport_ingest import fetch_and_persist_nport_quarter

    fixture = Path("tests/fixtures/nport/minimal_2019q4.zip")
    assert fixture.is_file()
    zip_bytes = fixture.read_bytes()
    payload_sha = __import__("hashlib").sha256(zip_bytes).hexdigest()

    class _FakeResponse:
        status_code = 200
        content = zip_bytes

    class _FakeClient:
        def get(self, url: str, **kwargs: object) -> _FakeResponse:
            _ = url, kwargs
            return _FakeResponse()

    settings = DataSettings(data_root=tmp_path / "data")
    fetch_and_persist_nport_quarter(
        filing_quarter="2019q4",
        settings=settings,
        client=_FakeClient(),
    )

    data_root = settings.resolved_data_root()
    content_addressed = data_root / "raw" / "sec" / "etf_holdings" / payload_sha / "payload.zip"
    zip_mirror = data_root / "raw" / "sec" / "nport" / "2019q4.zip"
    pointer = data_root / "raw" / "sec" / "nport" / "2019q4.json"

    assert content_addressed.is_file()
    assert not zip_mirror.exists()
    if pointer.exists():
        assert pointer.stat().st_size < 8192
        assert payload_sha in pointer.read_text(encoding="utf-8")


def test_nport_reuses_pointer_zip_without_http(tmp_path: Path) -> None:
    """test_nport_reuses_pointer_zip_without_http"""
    import hashlib
    import json

    import httpx

    from src.data.nport_ingest import _load_nport_zip_bytes

    fixture = Path("tests/fixtures/nport/minimal_2019q4.zip")
    assert fixture.is_file()
    zip_bytes = fixture.read_bytes()
    payload_sha = hashlib.sha256(zip_bytes).hexdigest()
    settings = DataSettings(data_root=tmp_path / "data")
    data_root = settings.resolved_data_root()
    rel = f"raw/sec/etf_holdings/{payload_sha}/payload.zip"
    target = data_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(zip_bytes)
    pointer_path = data_root / "raw/sec/nport/2019q4.json"
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(
        json.dumps({"sha256": payload_sha, "relative_path": rel, "filing_quarter": "2019q4"}),
        encoding="utf-8",
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500, content=b"boom")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        loaded, from_cache = _load_nport_zip_bytes("2019q4", settings, client)

    assert from_cache is True
    assert loaded == zip_bytes
    assert len(requests) == 0


def _holdings_frame(rows: list[tuple[str, date, str]]) -> pl.DataFrame:
    spec = spec_for(Dataset.ETF_HOLDINGS)
    retrieved = datetime.now(UTC)
    filing = datetime(2020, 3, 15, 12, 0, tzinfo=UTC)
    return pl.DataFrame(
        {
            "etf_ticker": [r[0] for r in rows],
            "report_date": [r[1] for r in rows],
            "filing_date": [filing] * len(rows),
            "holding_id": [r[2] for r in rows],
            "issuer_name": ["Issuer"] * len(rows),
            "cusip": ["123456789"] * len(rows),
            "isin": [None] * len(rows),
            "lei": [None] * len(rows),
            "weight_pct": [1.0] * len(rows),
            "value_usd": [1e6] * len(rows),
            "source": ["sec_nport"] * len(rows),
            "retrieved_at": [retrieved] * len(rows),
        }
    ).cast(pl.Schema(dict(spec.columns)))


def test_nport_unreadable_prior_aborts_and_writes_nothing(tmp_path: Path) -> None:
    """2026-09-25 incident replay: missing Bronze aborts ingest without new manifests."""
    from src.data.merge import PriorPartitionUntrustedError
    from src.data.nport_ingest import fetch_and_persist_nport_quarter

    settings = DataSettings(data_root=tmp_path / "data")
    seed = _holdings_frame(
        [
            ("SOXX", date(2019, 3, 31), "H1"),
            ("SOXX", date(2019, 6, 30), "H1"),
            ("SOXX", date(2019, 9, 30), "H1"),
            ("SOXX", date(2019, 12, 31), "H1"),
        ]
    )
    payload = RawPayload(
        provider="sec",
        endpoint="seed",
        request_params={},
        retrieved_at=datetime.now(UTC),
        extension="zip",
        content=b"seed-zip-bytes",
    )
    persist_ingest(seed, Dataset.ETF_HOLDINGS, payload, settings)
    manifests_dir = settings.resolved_data_root() / "manifests" / str(Dataset.ETF_HOLDINGS)
    before = sorted(manifests_dir.glob("*.json"))
    assert len(before) == 1

    from src.data.catalog import latest_artifact

    trusted = latest_artifact(settings, Dataset.ETF_HOLDINGS)
    raw_path = settings.resolved_data_root().joinpath(*trusted.manifest.raw_artifact.relative_path.parts)
    raw_path.unlink()

    from src.data.catalog import clear_catalog_frame_cache

    clear_catalog_frame_cache()
    fixture = Path("tests/fixtures/nport/minimal_2019q4.zip")
    zip_bytes = fixture.read_bytes()

    class _FakeResponse:
        status_code = 200
        content = zip_bytes

    class _FakeClient:
        def get(self, url: str, **kwargs: object) -> _FakeResponse:
            _ = url, kwargs
            return _FakeResponse()

    with pytest.raises(PriorPartitionUntrustedError):
        fetch_and_persist_nport_quarter(filing_quarter="2020q1", settings=settings, client=_FakeClient())
    assert sorted(manifests_dir.glob("*.json")) == before


def test_nport_quarter_append_keeps_history(tmp_path: Path) -> None:
    """A new quarter partition keeps Q1-Q2 keys and records the prior manifest hash."""
    from src.data.nport_ingest import fetch_and_persist_nport_quarter

    settings = DataSettings(data_root=tmp_path / "data")
    seed = _holdings_frame([("SOXX", date(2019, 3, 31), "H1"), ("SOXX", date(2019, 6, 30), "H1")])
    payload = RawPayload(
        provider="sec",
        endpoint="seed",
        request_params={},
        retrieved_at=datetime.now(UTC),
        extension="zip",
        content=b"seed-zip-bytes",
    )
    first = persist_ingest(seed, Dataset.ETF_HOLDINGS, payload, settings)

    fixture = Path("tests/fixtures/nport/minimal_2019q4.zip")
    zip_bytes = fixture.read_bytes()

    class _FakeResponse:
        status_code = 200
        content = zip_bytes

    class _FakeClient:
        def get(self, url: str, **kwargs: object) -> _FakeResponse:
            _ = url, kwargs
            return _FakeResponse()

    artifact = fetch_and_persist_nport_quarter(
        filing_quarter="2019q4", settings=settings, client=_FakeClient()
    )
    assert artifact.manifest.prior_manifest_sha256 == first.manifest_path.stem
    stored = DataStore(settings).read_normalized(artifact, spec_for(Dataset.ETF_HOLDINGS))
    pairs = set(zip(stored.get_column("etf_ticker").to_list(), stored.get_column("report_date").to_list(), strict=True))
    assert ("SOXX", date(2019, 3, 31)) in pairs
    assert ("SOXX", date(2019, 6, 30)) in pairs
    assert artifact.manifest.row_count > first.manifest.row_count


def test_nport_reingest_same_quarter_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ingesting the same quarter twice keeps row count and normalized hash."""
    from datetime import datetime as _real_datetime

    from src.data.nport_ingest import fetch_and_persist_nport_quarter

    fixed = _real_datetime(2020, 4, 1, 12, 0, tzinfo=UTC)

    class _FrozenDatetime(_real_datetime):
        @classmethod
        def now(cls, tz: object = None) -> _real_datetime:
            return fixed

    monkeypatch.setattr("src.data.nport_ingest.datetime", _FrozenDatetime)

    settings = DataSettings(data_root=tmp_path / "data")
    fixture = Path("tests/fixtures/nport/minimal_2019q4.zip")
    zip_bytes = fixture.read_bytes()

    class _FakeResponse:
        status_code = 200
        content = zip_bytes

    class _FakeClient:
        def get(self, url: str, **kwargs: object) -> _FakeResponse:
            _ = url, kwargs
            return _FakeResponse()

    first = fetch_and_persist_nport_quarter(
        filing_quarter="2019q4", settings=settings, client=_FakeClient()
    )
    second = fetch_and_persist_nport_quarter(
        filing_quarter="2019q4", settings=settings, client=_FakeClient()
    )
    assert second.manifest.row_count == first.manifest.row_count
    assert second.manifest.normalized_sha256 == first.manifest.normalized_sha256
