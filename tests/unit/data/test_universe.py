"""Invariant tests for point-in-time universe membership."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from src.data.universe import (
    MembershipEntry,
    UniverseMembership,
    assert_rows_within_lifetime,
    eligible_at,
    load_universe_membership,
)


def _entry(
    ticker: str,
    listing_date: date,
    last_trading_date: date | None = None,
    *,
    evidence_url: str = "https://example.com/evidence",
) -> MembershipEntry:
    return MembershipEntry(ticker, listing_date, last_trading_date, evidence_url)


def _membership(*entries: MembershipEntry) -> UniverseMembership:
    return UniverseMembership(entries={entry.ticker: entry for entry in entries}, sha256="test")


def _document(
    *entries: dict[str, object],
) -> dict[str, object]:
    return {"entries": list(entries)}


def _valid_raw(ticker: str = "TEST") -> dict[str, object]:
    return {
        "ticker": ticker,
        "listing_date": "2010-01-04",
        "last_trading_date": None,
        "evidence_url": "https://example.com/evidence",
    }


def _write_document(tmp_path: Path, document: object, name: str = "membership.json") -> Path:
    path = tmp_path / name
    if isinstance(document, str):
        path.write_text(document, encoding="utf-8")
    else:
        path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _prices(rows: tuple[tuple[str, date], ...]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": [ticker for ticker, _day in rows],
            "date": [day for _ticker, day in rows],
        },
        schema={"ticker": pl.String, "date": pl.Date},
    )


def test_listing_boundary_is_inclusive() -> None:
    membership = _membership(_entry("TEST", date(2010, 1, 4)))

    assert eligible_at(membership, ("TEST",), date(2010, 1, 1)) == ()
    assert eligible_at(membership, ("TEST",), date(2010, 1, 4)) == ("TEST",)


def test_last_trading_date_is_inclusive() -> None:
    membership = _membership(_entry("TEST", date(2010, 1, 4), date(2015, 6, 1)))

    assert eligible_at(membership, ("TEST",), date(2015, 6, 1)) == ("TEST",)
    assert eligible_at(membership, ("TEST",), date(2015, 6, 2)) == ()


def test_eligible_at_preserves_input_order() -> None:
    membership = _membership(
        _entry("CCC", date(2010, 1, 1)),
        _entry("AAA", date(2010, 1, 1)),
        _entry("BBB", date(2010, 1, 1)),
    )

    assert eligible_at(membership, ("BBB", "CCC", "AAA"), date(2020, 1, 1)) == (
        "BBB",
        "CCC",
        "AAA",
    )


def test_unknown_ticker_fails_closed() -> None:
    membership = _membership(_entry("TEST", date(2010, 1, 1)))

    with pytest.raises(ValueError, match="unknown membership ticker"):
        eligible_at(membership, ("MISSING",), date(2020, 1, 1))


@pytest.mark.parametrize(
    ("document", "match"),
    [
        pytest.param(
            _document(_valid_raw("TEST"), _valid_raw("TEST")),
            "duplicate membership ticker",
            id="duplicate",
        ),
        pytest.param(
            _document(
                {
                    **_valid_raw(),
                    "listing_date": "2015-01-02",
                    "last_trading_date": "2015-01-01",
                }
            ),
            "before listing_date",
            id="delisting-before-listing",
        ),
        pytest.param(
            _document({**_valid_raw(), "evidence_url": "ftp://example.com/evidence"}),
            "non-http evidence_url",
            id="non-http-evidence",
        ),
    ],
)
def test_invalid_membership_table_rejected(
    tmp_path: Path, document: dict[str, object], match: str
) -> None:
    path = _write_document(tmp_path, document)

    with pytest.raises(ValueError, match=match):
        load_universe_membership(path)


@pytest.mark.parametrize(
    ("document", "match"),
    [
        pytest.param("{not-json", "not valid JSON", id="malformed-json"),
        pytest.param([], "not a JSON object", id="non-object-root"),
        pytest.param({"entries": {}}, "no 'entries' list", id="non-list-entries"),
        pytest.param(_document("not-an-object"), "not a JSON object", id="non-object-entry"),
        pytest.param(_document({"ticker": "TEST"}), "missing field", id="missing-fields"),
        pytest.param(_document({**_valid_raw(), "ticker": " "}), "empty ticker", id="empty-ticker"),
        pytest.param(
            _document({**_valid_raw(), "listing_date": "not-a-date"}),
            "malformed 'listing_date'",
            id="bad-listing-date",
        ),
        pytest.param(
            _document({**_valid_raw(), "last_trading_date": "not-a-date"}),
            "malformed 'last_trading_date'",
            id="bad-last-date",
        ),
    ],
)
def test_malformed_membership_document_rejected(
    tmp_path: Path, document: Any, match: str
) -> None:
    path = _write_document(tmp_path, document)

    with pytest.raises(ValueError, match=match):
        load_universe_membership(path)


def test_loader_returns_sorted_entries_and_source_hash(tmp_path: Path) -> None:
    path = _write_document(
        tmp_path,
        _document(_valid_raw("ZZZ"), _valid_raw("AAA")),
    )

    membership = load_universe_membership(path)

    assert tuple(membership.entries) == ("AAA", "ZZZ")
    assert len(membership.sha256) == 64
    assert membership.entries["AAA"].evidence_url == "https://example.com/evidence"


def test_shipped_membership_covers_current_price_universe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    membership = load_universe_membership()

    assert len(membership.entries) == 26
    assert membership.entries["QQQ"].listing_date == date(1999, 3, 10)
    assert all(entry.last_trading_date is None for entry in membership.entries.values())
    assert len(membership.sha256) == 64


def test_rows_inside_lifetime_pass() -> None:
    membership = _membership(_entry("TEST", date(2010, 1, 4), date(2015, 6, 1)))
    frame = _prices((("TEST", date(2010, 1, 4)), ("TEST", date(2015, 6, 1))))

    assert_rows_within_lifetime(frame, membership)


@pytest.mark.parametrize(
    ("rows", "expected_date", "expected_count"),
    [
        pytest.param(
            (("TEST", date(2010, 1, 1)), ("TEST", date(2010, 1, 2))),
            "2010-01-01",
            2,
            id="before-listing",
        ),
        pytest.param(
            (("TEST", date(2015, 6, 2)), ("TEST", date(2015, 6, 3))),
            "2015-06-02",
            2,
            id="after-delisting",
        ),
    ],
)
def test_reused_ticker_rows_are_rejected(
    rows: tuple[tuple[str, date], ...], expected_date: str, expected_count: int
) -> None:
    membership = _membership(_entry("TEST", date(2010, 1, 4), date(2015, 6, 1)))

    with pytest.raises(ValueError, match="outside its membership lifetime") as exc_info:
        assert_rows_within_lifetime(_prices(rows), membership)

    message = str(exc_info.value)
    assert "ticker='TEST'" in message
    assert f"date={expected_date}" in message
    assert f"count={expected_count}" in message


def test_unknown_price_ticker_fails_closed_with_count() -> None:
    frame = _prices((("MISSING", date(2020, 1, 2)), ("MISSING", date(2020, 1, 3))))

    with pytest.raises(ValueError, match="has no universe membership entry") as exc_info:
        assert_rows_within_lifetime(frame, _membership(_entry("TEST", date(2010, 1, 1))))

    message = str(exc_info.value)
    assert "ticker='MISSING'" in message
    assert "date=2020-01-02" in message
    assert "count=2" in message
