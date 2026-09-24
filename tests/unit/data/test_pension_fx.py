"""Invariant guards for the merged USD/KRW fallback series."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from src.data.pension_fx import FxFallbackStatus, build_krw_fx_series

_TS = datetime(2024, 1, 1, 5, 0, tzinfo=UTC)


def _primary(rows: list[tuple[date, float | None, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {"date": [r[0] for r in rows], "usdkrw": [r[1] for r in rows], "source": [r[2] for r in rows]},
        schema={"date": pl.Date, "usdkrw": pl.Float64, "source": pl.String},
    )


def _fallback(rows: list[tuple[date, float | None, str]], late: set[date] | None = None) -> pl.DataFrame:
    late = late or set()
    stamps = []
    for day, _, _ in rows:
        if day in late:
            stamps.append(datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=1))
        else:
            stamps.append(datetime(day.year, day.month, day.day, 18, 0, tzinfo=UTC))
    return pl.DataFrame(
        {
            "date": [r[0] for r in rows],
            "usdkrw": [r[1] for r in rows],
            "source": [r[2] for r in rows],
            "available_at": stamps,
        },
        schema={"date": pl.Date, "usdkrw": pl.Float64, "source": pl.String, "available_at": pl.Datetime("us", "UTC")},
    )


def test_fills_interior_gap() -> None:
    """Fallback fills the Chuseok-style interior gap with its own labels."""
    primary = _primary([(date(2017, 9, 29), 1100.0, "ecos"), (date(2017, 10, 10), 1110.0, "ecos")])
    gap = [date(2017, 10, 2), date(2017, 10, 3), date(2017, 10, 4), date(2017, 10, 5), date(2017, 10, 6)]
    fallback = _fallback([(d, 1105.0, "fred") for d in gap])
    series = build_krw_fx_series(primary, fallback)
    assert series.status is FxFallbackStatus.APPLIED
    assert series.fallback_row_count == 5
    assert series.frame.get_column("date").to_list() == sorted([date(2017, 9, 29), date(2017, 10, 10), *gap])
    by_date = dict(zip(series.frame.get_column("date").to_list(), series.frame.get_column("source").to_list(), strict=True))
    assert all(by_date[d] == "fred" for d in gap)
    assert by_date[date(2017, 9, 29)] == "ecos"


def test_primary_wins_overlap() -> None:
    """A primary quote always beats a fallback quote on the same date."""
    day = date(2017, 9, 29)
    primary = _primary([(day, 1100.0, "ecos"), (date(2017, 10, 10), 1110.0, "ecos")])
    fallback = _fallback([(day, 1200.0, "fred")])
    series = build_krw_fx_series(primary, fallback)
    row = series.frame.filter(pl.col("date") == day).to_dicts()[0]
    assert row["usdkrw"] == pytest.approx(1100.0)
    assert row["source"] == "ecos"
    assert series.status is FxFallbackStatus.NOT_NEEDED


def test_fallback_never_extends_past_edges() -> None:
    """Fallback rows outside the primary range are ignored."""
    primary = _primary([(date(2017, 9, 29), 1100.0, "ecos"), (date(2017, 10, 10), 1110.0, "ecos")])
    fallback = _fallback(
        [(date(2017, 9, 28), 1090.0, "fred"), (date(2017, 10, 11), 1120.0, "fred")]
    )
    series = build_krw_fx_series(primary, fallback)
    assert series.fallback_row_count == 0
    assert series.frame.height == 2
    assert series.status is FxFallbackStatus.NOT_NEEDED


def test_vendor_holiday_null_is_not_a_quote() -> None:
    """A null fallback quote leaves the date absent without an error."""
    primary = _primary([(date(2017, 9, 29), 1100.0, "ecos"), (date(2017, 10, 10), 1110.0, "ecos")])
    fallback = _fallback([(date(2017, 10, 3), None, "fred")])
    series = build_krw_fx_series(primary, fallback)
    assert date(2017, 10, 3) not in series.frame.get_column("date").to_list()
    assert series.status is FxFallbackStatus.NOT_NEEDED


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_quote_rejected(bad: float) -> None:
    """Non-positive or non-finite quotes fail closed on either frame."""
    good_day, other = date(2017, 9, 29), date(2017, 10, 10)
    with pytest.raises(ValueError, match="non-positive or non-finite"):
        build_krw_fx_series(_primary([(good_day, bad, "ecos"), (other, 1110.0, "ecos")]), None)
    with pytest.raises(ValueError, match="non-positive or non-finite"):
        build_krw_fx_series(
            _primary([(good_day, 1100.0, "ecos"), (other, 1110.0, "ecos")]),
            _fallback([(date(2017, 10, 3), bad, "fred")]),
        )


def test_duplicate_dates_rejected() -> None:
    """Duplicate quote dates fail closed on either frame."""
    with pytest.raises(ValueError, match="duplicate"):
        build_krw_fx_series(
            _primary([(date(2017, 9, 29), 1100.0, "ecos"), (date(2017, 9, 29), 1101.0, "ecos")]), None
        )
    with pytest.raises(ValueError, match="duplicate"):
        build_krw_fx_series(
            _primary([(date(2017, 9, 29), 1100.0, "ecos"), (date(2017, 10, 10), 1110.0, "ecos")]),
            _fallback([(date(2017, 10, 3), 1105.0, "fred"), (date(2017, 10, 3), 1106.0, "fred")]),
        )


def test_missing_column_rejected() -> None:
    """A fallback without available_at is a contract violation."""
    primary = _primary([(date(2017, 9, 29), 1100.0, "ecos"), (date(2017, 10, 10), 1110.0, "ecos")])
    bad = pl.DataFrame(
        {"date": [date(2017, 10, 3)], "usdkrw": [1105.0], "source": ["fred"]},
        schema={"date": pl.Date, "usdkrw": pl.Float64, "source": pl.String},
    )
    with pytest.raises(ValueError, match="available_at"):
        build_krw_fx_series(primary, bad)


def test_empty_primary_rejected() -> None:
    """A primary with only null quotes carries no market information."""
    with pytest.raises(ValueError, match="no quote"):
        build_krw_fx_series(_primary([(date(2017, 9, 29), None, "ecos")]), None)


def test_identical_source_labels_rejected() -> None:
    """Sharing a source label across frames would hide provenance."""
    primary = _primary([(date(2017, 9, 29), 1100.0, "ecos"), (date(2017, 10, 10), 1110.0, "ecos")])
    with pytest.raises(ValueError, match="source label"):
        build_krw_fx_series(primary, _fallback([(date(2017, 10, 3), 1105.0, "ecos")]))


def test_late_published_fallback_rejected_and_counted() -> None:
    """A quote published the next day is excluded from fills and basis."""
    primary = _primary([(date(2017, 9, 29), 1000.0, "ecos"), (date(2017, 10, 10), 1000.0, "ecos")])
    day = date(2017, 10, 3)
    fallback = _fallback([(day, 1005.0, "fred")], late={day})
    series = build_krw_fx_series(primary, fallback)
    assert day not in series.frame.get_column("date").to_list()
    assert series.rejected_late_count == 1
    assert series.basis is None


def test_basis_measured_on_overlap() -> None:
    """Overlapping quotes disclose the same-day source basis."""
    primary = _primary([(date(2017, 9, 29), 1000.0, "ecos"), (date(2017, 9, 30), 1000.0, "ecos")])
    fallback = _fallback([(date(2017, 9, 29), 1001.0, "fred"), (date(2017, 9, 30), 999.0, "fred")])
    series = build_krw_fx_series(primary, fallback)
    assert series.basis is not None
    assert series.basis.overlap_count == 2
    assert series.basis.mean_abs_bps == pytest.approx(10.0)
    assert series.basis.max_abs_bps == pytest.approx(10.0)


def test_no_overlap_yields_no_basis() -> None:
    """Disjoint coverage leaves nothing to compare."""
    primary = _primary([(date(2017, 9, 29), 1100.0, "ecos"), (date(2017, 10, 10), 1110.0, "ecos")])
    fallback = _fallback([(date(2017, 10, 3), 1105.0, "fred")])
    series = build_krw_fx_series(primary, fallback)
    assert series.basis is None
    provenance = series.provenance([date(2017, 10, 3)])
    assert provenance["basis_overlap_count"] == 0
    assert provenance["basis_mean_abs_bps"] is None


def test_unavailable_fallback() -> None:
    """Missing fallback data reproduces the previous primary-only behavior."""
    primary = _primary([(date(2017, 9, 29), 1100.0, "ecos"), (date(2017, 10, 10), 1110.0, "ecos")])
    series = build_krw_fx_series(primary, None)
    assert series.status is FxFallbackStatus.UNAVAILABLE
    assert series.fallback_row_count == 0
    assert series.frame.get_column("date").to_list() == [date(2017, 9, 29), date(2017, 10, 10)]


def test_not_needed_when_gap_free() -> None:
    """An unused fallback dataset is disclosed, not hidden."""
    primary = _primary([(date(2017, 9, 29), 1100.0, "ecos"), (date(2017, 9, 30), 1101.0, "ecos")])
    fallback = _fallback([(date(2017, 9, 29), 1100.5, "fred")])
    series = build_krw_fx_series(primary, fallback)
    assert series.status is FxFallbackStatus.NOT_NEEDED
    assert series.fallback_row_count == 0


def test_session_share_counts_carried_quotes() -> None:
    """Sessions inherit the latest quote, so a gap fill covers following sessions."""
    thu, fri, mon, tue = date(2017, 9, 28), date(2017, 9, 29), date(2017, 10, 2), date(2017, 10, 3)
    primary = _primary([(thu, 1100.0, "ecos"), (tue, 1110.0, "ecos")])
    fallback = _fallback([(fri, 1105.0, "fred")])
    series = build_krw_fx_series(primary, fallback)
    assert series.fallback_session_dates([fri, mon, tue]) == (fri, mon)
    provenance = series.provenance([fri, mon, tue])
    assert provenance["fallback_session_count"] == 2
    assert provenance["fallback_session_share"] == pytest.approx(2 / 3)


def test_empty_sessions() -> None:
    """No sessions means zero share without an error."""
    primary = _primary([(date(2017, 9, 29), 1100.0, "ecos"), (date(2017, 10, 10), 1110.0, "ecos")])
    series = build_krw_fx_series(primary, None)
    assert series.fallback_session_dates([]) == ()
    assert series.provenance([])["fallback_session_share"] == 0.0


def test_order_independence() -> None:
    """Shuffled input rows produce identical output."""
    primary = _primary([(date(2017, 9, 29), 1100.0, "ecos"), (date(2017, 10, 10), 1110.0, "ecos")])
    rows = [(date(2017, 10, 3), 1105.0, "fred"), (date(2017, 10, 4), 1106.0, "fred")]
    first = build_krw_fx_series(primary, _fallback(rows))
    second = build_krw_fx_series(primary, _fallback(list(reversed(rows))))
    assert first.frame.equals(second.frame)
    assert first.provenance([date(2017, 10, 3)]) == second.provenance([date(2017, 10, 3)])


def test_future_perturbation_invariance() -> None:
    """Fallback rows after a cutoff never change earlier output rows."""
    primary = _primary([(date(2017, 9, 29), 1100.0, "ecos"), (date(2017, 10, 10), 1110.0, "ecos")])
    early = [(date(2017, 10, 2), 1105.0, "fred")]
    late_a = [(date(2017, 10, 5), 1106.0, "fred")]
    late_b = [(date(2017, 10, 5), 9999.0, "fred")]
    left = build_krw_fx_series(primary, _fallback(early + late_a))
    right = build_krw_fx_series(primary, _fallback(early + late_b))
    cutoff = date(2017, 10, 3)
    assert left.frame.filter(pl.col("date") <= cutoff).equals(right.frame.filter(pl.col("date") <= cutoff))


def test_provenance_is_json_serializable() -> None:
    """The disclosure mapping must survive a JSON round-trip."""
    primary = _primary([(date(2017, 9, 29), 1100.0, "ecos"), (date(2017, 10, 10), 1110.0, "ecos")])
    fallback = _fallback([(date(2017, 10, 3), 1105.0, "fred")])
    series = build_krw_fx_series(primary, fallback)
    json.dumps(series.provenance([date(2017, 9, 29), date(2017, 10, 3)]))


def test_session_before_first_quote_is_not_fallback() -> None:
    """Sessions predating the series rely on the engine guard, not the fallback."""
    primary = _primary([(date(2017, 9, 29), 1100.0, "ecos"), (date(2017, 10, 10), 1110.0, "ecos")])
    fallback = _fallback([(date(2017, 10, 3), 1105.0, "fred")])
    series = build_krw_fx_series(primary, fallback)
    assert series.fallback_session_dates([date(2017, 9, 28)]) == ()


def test_non_date_value_rejected() -> None:
    """A non-date observation key fails closed."""
    bad = pl.DataFrame(
        {"date": ["2017-09-29"], "usdkrw": [1100.0], "source": ["ecos"]},
        schema={"date": pl.String, "usdkrw": pl.Float64, "source": pl.String},
    )
    with pytest.raises(ValueError, match="non-date"):
        build_krw_fx_series(bad, None)


def test_missing_source_rejected() -> None:
    """A valid quote without a source label cannot disclose provenance."""
    bad = pl.DataFrame(
        {"date": [date(2017, 9, 29), date(2017, 10, 10)], "usdkrw": [1100.0, 1110.0], "source": ["ecos", None]},
        schema={"date": pl.Date, "usdkrw": pl.Float64, "source": pl.String},
    )
    with pytest.raises(ValueError, match="missing source"):
        build_krw_fx_series(bad, None)


def test_naive_available_at_rejected() -> None:
    """A tz-naive fallback stamp cannot prove point-in-time publication."""
    from datetime import datetime as _dt

    primary = _primary([(date(2017, 9, 29), 1100.0, "ecos"), (date(2017, 10, 10), 1110.0, "ecos")])
    bad = pl.DataFrame(
        {"date": [date(2017, 10, 3)], "usdkrw": [1105.0], "source": ["fred"],
         "available_at": [_dt(2017, 10, 3, 18, 0)]},
        schema={"date": pl.Date, "usdkrw": pl.Float64, "source": pl.String, "available_at": pl.Datetime("us")},
    )
    with pytest.raises(ValueError, match="tz-aware"):
        build_krw_fx_series(primary, bad)


def test_missing_available_at_rejected() -> None:
    """A fallback quote without a publication stamp cannot be admitted."""
    primary = _primary([(date(2017, 9, 29), 1100.0, "ecos"), (date(2017, 10, 10), 1110.0, "ecos")])
    bad = pl.DataFrame(
        {"date": [date(2017, 10, 3)], "usdkrw": [1105.0], "source": ["fred"], "available_at": [None]},
        schema={"date": pl.Date, "usdkrw": pl.Float64, "source": pl.String, "available_at": pl.Datetime("us", "UTC")},
    )
    with pytest.raises(ValueError, match="missing available_at"):
        build_krw_fx_series(primary, bad)
