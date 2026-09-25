"""Unit tests for evidence-backed vendor price corrections applied before Silver."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from src.data.calendar import TradingCalendar, load_calendar
from src.data.fetch import fetch_and_persist_prices
from src.data.pit import stamp_availability
from src.data.price_corrections import (
    PriceCorrection,
    PriceCorrectionSet,
    apply_price_corrections,
    load_price_corrections,
)
from src.data.quality import validate_frame
from src.data.schema import Dataset, DatasetSpec, spec_for
from src.data.secrets import ProviderSecrets
from src.data.settings import DataSettings
from src.data.storage import DataStore

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHIPPED_CORRECTIONS = _REPO_ROOT / "configs" / "data" / "price_corrections.json"
_RETRIEVED_AT = datetime(2024, 2, 1, 5, 0, tzinfo=UTC)
_SECRETS = ProviderSecrets(tiingo_api="wire-tiingo-token", fred_api="wire-fred-key", ecos_api="wire-ecos-key")
_ASH_CRASH_SESSION = date(2010, 5, 6)
_PRICE_COLUMNS = ("open", "high", "low", "close", "adjusted_close")
_CORRECTION = PriceCorrection(
    ticker="AAA",
    session=date(2024, 1, 30),
    open=49.0,
    high=51.0,
    low=48.0,
    close=50.0,
    evidence_url="https://example.com/aaa/2024-01-30",
    reason="vendor flat print",
)
_CORRECTIONS = PriceCorrectionSet(corrections=(_CORRECTION,), sha256="a" * 64)


def _prices_bars(bars: list[tuple[str, date, float, float, bool]]) -> pl.DataFrame:
    """Build a PRICES frame from (ticker, session, close, adjusted_close, is_flat_print) bars."""
    spec = spec_for(Dataset.PRICES)
    n = len(bars)
    return pl.DataFrame(
        {
            "ticker": [ticker for ticker, _, _, _, _ in bars],
            "date": [session for _, session, _, _, _ in bars],
            "open": [close if flat else close * 0.98 for _, _, close, _, flat in bars],
            "high": [close if flat else close * 1.02 for _, _, close, _, flat in bars],
            "low": [close if flat else close * 0.97 for _, _, close, _, flat in bars],
            "close": [close for _, _, close, _, _ in bars],
            "volume": [10_000] * n,
            "adjusted_close": [adjusted for _, _, _, adjusted, _ in bars],
            "dividend": [0.0] * n,
            "split_factor": [1.0] * n,
            "source": ["tiingo"] * n,
            "retrieved_at": [_RETRIEVED_AT] * n,
        },
        schema=dict(spec.columns),
    )


def _flash_crash_bars() -> list[tuple[str, date, float, float, bool]]:
    """Vendor bars around the 2010-05-06 flash crash, where the middle bar is a flat print."""
    return [
        ("ITA", date(2010, 5, 5), 58.0, 29.0, False),
        ("ITA", _ASH_CRASH_SESSION, 40.79, 20.395, True),
        ("ITA", date(2010, 5, 7), 56.5, 28.25, False),
    ]


def _reverting_print_findings(
    frame: pl.DataFrame, spec: DatasetSpec, calendar: TradingCalendar
) -> tuple[str, ...]:
    report = validate_frame(stamp_availability(frame, spec, calendar), spec, calendar)
    return tuple(finding.code for finding in report.findings if finding.code == "REVERTING_FLAT_PRINT")


def _corrected_row(frame: pl.DataFrame) -> dict[str, object]:
    return frame.filter(pl.col("date") == _ASH_CRASH_SESSION).row(0, named=True)


def _client_serving(body: bytes) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _fresh_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> DataSettings:
    monkeypatch.chdir(tmp_path)
    return DataSettings(data_root="data")


def _tiingo_flash_crash_body() -> bytes:
    return json.dumps(
        [
            {
                "date": "2010-05-05T00:00:00.000Z",
                "open": 57.9,
                "high": 58.4,
                "low": 57.6,
                "close": 58.0,
                "volume": 1_000_000,
                "adjClose": 29.0,
                "divCash": 0.0,
                "splitFactor": 1.0,
            },
            {
                "date": "2010-05-06T00:00:00.000Z",
                "open": 40.79,
                "high": 40.79,
                "low": 40.79,
                "close": 40.79,
                "volume": 1_178_600,
                "adjClose": 20.395,
                "divCash": 0.0,
                "splitFactor": 1.0,
            },
            {
                "date": "2010-05-07T00:00:00.000Z",
                "open": 56.4,
                "high": 56.8,
                "low": 56.2,
                "close": 56.5,
                "volume": 900_000,
                "adjClose": 28.25,
                "divCash": 0.0,
                "splitFactor": 1.0,
            },
        ]
    ).encode()


def test_apply_price_corrections_replaces_bar_and_keeps_adjustment_factor() -> None:
    """A replaced bar keeps its row's adjusted_close/close split and distribution factor."""
    frame = _prices_bars([("AAA", date(2024, 1, 30), 40.0, 8.0, False)])
    patched = apply_price_corrections(frame, _CORRECTIONS)
    row = patched.row(0, named=True)
    assert (row["open"], row["high"], row["low"], row["close"]) == (49.0, 51.0, 48.0, 50.0)
    assert row["adjusted_close"] == pytest.approx(10.0)
    untouched = [name for name in frame.columns if name not in _PRICE_COLUMNS]
    assert patched.select(untouched).equals(frame.select(untouched))


def test_apply_price_corrections_conserves_keys_and_column_order() -> None:
    """Applying one correction adds, drops, and reorders nothing."""
    frame = _prices_bars(
        [
            ("AAA", date(2024, 1, 29), 100.0, 100.0, False),
            ("AAA", date(2024, 1, 30), 40.0, 8.0, False),
            ("BBB", date(2024, 1, 29), 55.0, 55.0, False),
            ("BBB", date(2024, 1, 30), 56.0, 56.0, False),
        ]
    )
    patched = apply_price_corrections(frame, _CORRECTIONS)
    assert patched.columns == frame.columns
    assert patched.height == frame.height
    assert patched.select("ticker", "date").rows() == frame.select("ticker", "date").rows()
    assert patched.filter(pl.col("ticker") == "BBB").equals(frame.filter(pl.col("ticker") == "BBB"))


def test_apply_price_corrections_skips_absent_key(caplog: pytest.LogCaptureFixture) -> None:
    """A correction whose ticker is absent leaves the frame untouched and is counted as skipped."""
    frame = _prices_bars([("BBB", date(2024, 1, 30), 40.0, 8.0, False)])
    with caplog.at_level(logging.INFO, logger="src.data.price_corrections"):
        patched = apply_price_corrections(frame, _CORRECTIONS)
    assert patched.equals(frame)
    assert f"applied=0 skipped=1 sha={'a' * 64}" in caplog.text


def test_apply_price_corrections_is_idempotent() -> None:
    """Applying the same set twice equals applying it once."""
    frame = _prices_bars([("AAA", date(2024, 1, 30), 40.0, 8.0, False)])
    once = apply_price_corrections(frame, _CORRECTIONS)
    assert apply_price_corrections(once, _CORRECTIONS).equals(once)


def test_apply_price_corrections_rejects_nonpositive_original_close() -> None:
    """A correction onto a non-positive close is refused instead of dividing by it."""
    frame = _prices_bars([("AAA", date(2024, 1, 30), 0.0, 0.0, False)])
    with pytest.raises(ValueError, match="non-positive"):
        apply_price_corrections(frame, _CORRECTIONS)


def _entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "ticker": "AAA",
        "session": "2024-01-30",
        "open": 49.0,
        "high": 51.0,
        "low": 48.0,
        "close": 50.0,
        "evidence_url": "https://example.com/aaa/2024-01-30",
        "reason": "vendor flat print",
    }
    entry.update(overrides)
    return entry


def _write_document(tmp_path: Path, document: object) -> Path:
    target = tmp_path / "price_corrections.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    return target


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        pytest.param({"corrections": [_entry(), _entry()]}, "duplicate correction", id="duplicate-key"),
        pytest.param({"corrections": [_entry(close=0.0)]}, "non-positive or non-finite 'close'", id="non-positive"),
        pytest.param(
            {"corrections": [_entry(close=float("nan"))]},
            "non-positive or non-finite 'close'",
            id="non-finite",
        ),
        pytest.param({"corrections": [_entry(close=None)]}, "non-numeric 'close'", id="non-numeric"),
        pytest.param({"corrections": [_entry(high=49.5)]}, "'high'", id="high-below-close"),
        pytest.param({"corrections": [_entry(low=49.5)]}, "'low'", id="low-above-open"),
        pytest.param(
            {"corrections": [_entry(evidence_url="ftp://example.com/bar")]},
            "non-http 'evidence_url'",
            id="non-http-evidence",
        ),
        pytest.param({"corrections": [_entry(evidence_url="example.com/bar")]}, "'evidence_url'", id="schemeless-url"),
        pytest.param({"corrections": [_entry(session="30-01-2024")]}, "malformed 'session'", id="malformed-session"),
        pytest.param({"corrections": [_entry(ticker="  ")]}, "declares no 'ticker'", id="missing-ticker"),
        pytest.param({"corrections": [_entry(reason="")]}, "declares no 'reason'", id="missing-reason"),
        pytest.param({"corrections": ["not-an-object"]}, "is not a JSON object", id="entry-not-object"),
        pytest.param({}, "declares no 'corrections' list", id="missing-list"),
        pytest.param([], "is not a JSON object", id="document-not-object"),
    ],
)
def test_load_price_corrections_rejects_invalid_documents(
    tmp_path: Path, document: object, expected: str
) -> None:
    """Every invalid correction is refused with a message naming the offending field."""
    with pytest.raises(ValueError, match=expected):
        load_price_corrections(_write_document(tmp_path, document))


def test_load_price_corrections_rejects_malformed_json(tmp_path: Path) -> None:
    """A document that is not JSON fails closed instead of yielding an empty set."""
    target = tmp_path / "price_corrections.json"
    target.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="is not valid JSON"):
        load_price_corrections(target)


def test_shipped_correction_file_is_loaded_unchanged_and_ordered() -> None:
    """The git-tracked file holds the three researched 2010-05-06 bars, keyed and digested."""
    corrections = load_price_corrections()
    assert [(item.ticker, item.session) for item in corrections.corrections] == [
        ("ITA", _ASH_CRASH_SESSION),
        ("IWF", _ASH_CRASH_SESSION),
        ("VTV", _ASH_CRASH_SESSION),
    ]
    assert corrections.sha256 == hashlib.sha256(_SHIPPED_CORRECTIONS.read_bytes()).hexdigest()
    assert all(item.evidence_url.startswith("https://") and item.reason for item in corrections.corrections)


def test_corrected_flat_print_is_no_longer_flagged() -> None:
    """The shipped ITA correction clears REVERTING_FLAT_PRINT and lands the cited close."""
    calendar = load_calendar("XNYS")
    spec = spec_for(Dataset.PRICES)
    frame = _prices_bars(_flash_crash_bars())
    assert _reverting_print_findings(frame, spec, calendar) == ("REVERTING_FLAT_PRINT",)
    corrected = apply_price_corrections(frame, load_price_corrections())
    assert _reverting_print_findings(corrected, spec, calendar) == ()
    row = _corrected_row(corrected)
    assert (row["open"], row["high"], row["low"], row["close"]) == (57.61, 58.16, 25.91, 56.06)
    assert row["adjusted_close"] == pytest.approx(28.03)


def test_fetch_and_persist_prices_records_correction_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The manifest request params carry the file digest and the Silver row the corrected close."""
    settings = _fresh_settings(monkeypatch, tmp_path)
    with _client_serving(_tiingo_flash_crash_body()) as http:
        artifact = fetch_and_persist_prices(
            ("ITA",),
            date(2010, 5, 5),
            date(2010, 5, 7),
            secrets=_SECRETS,
            settings=settings,
            client=http,
        )
    expected_digest = hashlib.sha256(_SHIPPED_CORRECTIONS.read_bytes()).hexdigest()
    assert artifact.manifest.request_params["price_corrections_sha256"] == expected_digest
    assert all(finding.code != "REVERTING_FLAT_PRINT" for finding in artifact.manifest.quality_findings)
    row = _corrected_row(DataStore(settings).read_normalized(artifact, spec_for(Dataset.PRICES)))
    assert row["close"] == 56.06
    assert row["adjusted_close"] == pytest.approx(28.03)


def test_incremental_republish_applies_corrections_without_shrinking_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Re-running ingest with no new sessions republishes the prior partition corrected."""
    import src.data.fetch as fetch_module

    settings = _fresh_settings(monkeypatch, tmp_path)
    spec = spec_for(Dataset.PRICES)
    shipped_loader = fetch_module.load_price_corrections
    with _client_serving(_tiingo_flash_crash_body()) as http:
        monkeypatch.setattr(
            fetch_module, "load_price_corrections", lambda: PriceCorrectionSet((), "0" * 64)
        )
        first = fetch_and_persist_prices(
            ("ITA",), date(2010, 5, 5), date(2010, 5, 7), secrets=_SECRETS, settings=settings, client=http
        )
        monkeypatch.setattr(fetch_module, "load_price_corrections", shipped_loader)
        second = fetch_and_persist_prices(
            ("ITA",),
            date(2010, 5, 5),
            date(2010, 5, 7),
            secrets=_SECRETS,
            settings=settings,
            client=http,
            incremental=True,
        )
    assert _corrected_row(DataStore(settings).read_normalized(first, spec))["close"] == 40.79
    assert second.manifest.row_count == first.manifest.row_count
    assert second.manifest.prior_manifest_sha256 is not None
    assert second.manifest.normalized_sha256 != first.manifest.normalized_sha256
    assert second.manifest.request_params["price_corrections_sha256"] == hashlib.sha256(
        _SHIPPED_CORRECTIONS.read_bytes()
    ).hexdigest()
    assert _corrected_row(DataStore(settings).read_normalized(second, spec))["close"] == 56.06
