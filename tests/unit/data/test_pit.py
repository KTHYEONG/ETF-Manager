"""Unit and property tests for the point-in-time temporal core."""

from __future__ import annotations

from datetime import date, datetime, timedelta, UTC
from pathlib import Path

import polars as pl
from hypothesis import given, settings, strategies as st

from src.data.calendar import load_calendar
from src.data.pit import AVAILABLE_AT, LookAheadError, as_of, assert_no_lookahead, stamp_availability
from src.data.schema import (
    AvailabilityKind,
    AvailabilityRule,
    Dataset,
    DatasetSpec,
    MissingPolicy,
    TotalReturnSource,
    spec_for,
)

UTC = UTC
TS_DTYPE = pl.Datetime("us", "UTC")


def _prices_frame(dates: list[date]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": dates,
            "close": [100.0 + i for i in range(len(dates))],
        },
        schema={"date": pl.Date, "close": pl.Float64},
    )


def _fixed_lag_spec(lag: timedelta) -> DatasetSpec:
    return DatasetSpec(
        dataset=Dataset.FACTORS,
        columns={"period_end": pl.Date, "mkt_rf": pl.Float64},
        key=("period_end",),
        observation_column="period_end",
        availability=AvailabilityRule(kind=AvailabilityKind.FIXED_LAG, lag=lag),
        missing_policy=MissingPolicy.FAIL,
        revisable=False,
        total_return_source=TotalReturnSource.NOT_APPLICABLE,
        schema_version="1",
    )


def _release_spec(revisable: bool) -> DatasetSpec:
    return DatasetSpec(
        dataset=Dataset.MACRO,
        columns={
            "observation_date": pl.Date,
            "release_date": TS_DTYPE,
            "value": pl.Float64,
        },
        key=("observation_date",),
        observation_column="observation_date",
        availability=AvailabilityRule(kind=AvailabilityKind.RELEASE_COLUMN, release_column="release_date"),
        missing_policy=MissingPolicy.FAIL,
        revisable=revisable,
        total_return_source=TotalReturnSource.NOT_APPLICABLE,
        schema_version="1",
    )


def test_pit_a02_session_close_availability() -> None:
    """PIT-A02-session-close-availability"""
    calendar = load_calendar("XNYS")
    frame = _prices_frame([date(2024, 1, 31)])
    stamped = stamp_availability(frame, spec_for(Dataset.PRICES), calendar)
    close_ts = calendar.close_ts(date(2024, 1, 31))
    assert stamped.get_column(AVAILABLE_AT)[0] == close_ts
    assert as_of(stamped, spec_for(Dataset.PRICES), close_ts - timedelta(microseconds=1)).height == 0
    assert as_of(stamped, spec_for(Dataset.PRICES), close_ts).height == 1


def test_pit_a03_fixed_lag_availability() -> None:
    """PIT-A03-fixed-lag-availability"""
    spec = _fixed_lag_spec(timedelta(days=60))
    frame = pl.DataFrame(
        {"period_end": [date(2020, 1, 31)], "mkt_rf": [0.5]},
        schema={"period_end": pl.Date, "mkt_rf": pl.Float64},
    )
    stamped = stamp_availability(frame, spec)
    expected = datetime(2020, 3, 31, tzinfo=UTC)
    assert stamped.get_column(AVAILABLE_AT)[0] == expected
    assert as_of(stamped, spec, datetime(2020, 3, 30, 23, 59, 59, tzinfo=UTC)).height == 0
    assert as_of(stamped, spec, expected).height == 1


def test_pit_a04_vintage_resolution() -> None:
    """PIT-A04-vintage-resolution"""
    revisable_spec = _release_spec(revisable=True)
    frame = pl.DataFrame(
        {
            "observation_date": [date(2020, 1, 1), date(2020, 1, 1)],
            "release_date": [
                datetime(2020, 2, 10, tzinfo=UTC),
                datetime(2020, 3, 12, tzinfo=UTC),
            ],
            "value": [1.0, 1.2],
        },
        schema={"observation_date": pl.Date, "release_date": TS_DTYPE, "value": pl.Float64},
    )
    stamped_frame = stamp_availability(frame, revisable_spec)
    early = as_of(stamped_frame, revisable_spec, datetime(2020, 3, 1, tzinfo=UTC))
    assert early.height == 1
    assert early.get_column("value")[0] == 1.0
    late = as_of(stamped_frame, revisable_spec, datetime(2020, 3, 31, tzinfo=UTC))
    assert late.height == 1
    assert late.get_column("value")[0] == 1.2

    append_only_spec = _release_spec(revisable=False)
    assert as_of(stamp_availability(frame, append_only_spec), append_only_spec, datetime(2020, 3, 31, tzinfo=UTC)).height == 2


def test_pit_a05_lookahead_guard() -> None:
    """PIT-A05-lookahead-guard"""
    decision_ts = datetime(2024, 2, 1, tzinfo=UTC)
    original = pl.DataFrame(
        {
            "date": [date(2024, 1, 31), date(2024, 2, 1)],
            "close": [100.0, 101.0],
            AVAILABLE_AT: [
                datetime(2024, 1, 31, tzinfo=UTC),
                decision_ts + timedelta(microseconds=1),
            ],
        },
        schema={"date": pl.Date, "close": pl.Float64, AVAILABLE_AT: TS_DTYPE},
    )

    try:
        assert_no_lookahead(original, decision_ts)
        raised = False
    except LookAheadError:
        raised = True
    assert raised is True
    assert original.equals(original.clone())

    ok_frame = original.head(1)
    assert assert_no_lookahead(ok_frame, decision_ts) is None

    missing_column = original.drop(AVAILABLE_AT)
    try:
        assert_no_lookahead(missing_column, decision_ts)
        value_error_raised = False
    except ValueError:
        value_error_raised = True
    assert value_error_raised is True

    naive_decision = decision_ts.replace(tzinfo=None)
    try:
        assert_no_lookahead(ok_frame, naive_decision)
        naive_raised = False
    except ValueError:
        naive_raised = True
    assert naive_raised is True

    assert original.equals(
        pl.DataFrame(
            {
                "date": [date(2024, 1, 31), date(2024, 2, 1)],
                "close": [100.0, 101.0],
                AVAILABLE_AT: [
                    datetime(2024, 1, 31, tzinfo=UTC),
                    decision_ts + timedelta(microseconds=1),
                ],
            },
            schema={"date": pl.Date, "close": pl.Float64, AVAILABLE_AT: TS_DTYPE},
        )
    )


_AVS_STRATEGY = st.lists(
    st.datetimes(
        min_value=datetime(2000, 1, 1),
        max_value=datetime(2030, 1, 1),
        timezones=st.just(UTC),
    ),
    min_size=1,
    max_size=12,
)


@given(_AVS_STRATEGY)
@settings(max_examples=50, deadline=None)
def test_pit_a06_asof_invariant_property(availabilities: list[datetime]) -> None:
    """PIT-A06-asof-invariant-property"""
    spec = _release_spec(revisable=True)
    n = len(availabilities)
    keys = [i % 3 for i in range(n)]
    frame = pl.DataFrame(
        {
            "observation_date": [date(2020, 1, 1) + timedelta(days=k) for k in keys],
            "release_date": availabilities,
            "value": [float(i) for i in range(n)],
        },
        schema={"observation_date": pl.Date, "release_date": TS_DTYPE, "value": pl.Float64},
    )
    decision_ts = max(availabilities) + timedelta(days=30)
    stamped = stamp_availability(frame, spec)
    result = as_of(stamped, spec, decision_ts)

    assert result.height <= frame.height
    assert all(ts <= decision_ts for ts in result.get_column(AVAILABLE_AT).to_list())
    assert_no_lookahead(result, decision_ts)
    assert set(result.columns) == set(stamped.columns)
    assert result.get_column("observation_date").n_unique() == result.height


def test_pit_a07_no_imputation() -> None:
    """PIT-A07-no-imputation"""
    calendar = load_calendar("XNYS")
    frame = _prices_frame([date(2024, 1, 29), date(2024, 1, 31)])
    stamped = stamp_availability(frame, spec_for(Dataset.PRICES), calendar)
    assert stamped.height == 2
    availability = stamped.get_column(AVAILABLE_AT)
    assert availability.null_count() == 0
    assert stamped.get_column("date").to_list() == [date(2024, 1, 29), date(2024, 1, 31)]

    source = Path("src/data/pit.py").read_text(encoding="utf-8")
    for banned in ("fill_null", "forward_fill", "interpolate", "upsample"):
        assert banned not in source


def _macro_two_release_frame() -> pl.DataFrame:
    spec = spec_for(Dataset.MACRO)
    return pl.DataFrame(
        {
            "series_id": ["CPI", "CPI"],
            "observation_date": [date(2024, 1, 1), date(2024, 1, 1)],
            "release_date": [
                datetime(2024, 2, 10, tzinfo=UTC),
                datetime(2024, 3, 12, tzinfo=UTC),
            ],
            "value": [1.0, 1.2],
        },
        schema=dict(spec.columns),
    )


def test_registered_macro_vintage_selection() -> None:
    """Two macro releases resolve to the then-latest visible vintage."""
    spec = spec_for(Dataset.MACRO)
    stamped = stamp_availability(_macro_two_release_frame(), spec)
    early = as_of(stamped, spec, datetime(2024, 3, 1, tzinfo=UTC))
    assert early.height == 1
    assert early.get_column("value")[0] == 1.0
    late = as_of(stamped, spec, datetime(2024, 3, 31, tzinfo=UTC))
    assert late.height == 1
    assert late.get_column("value")[0] == 1.2


def test_future_revision_independence() -> None:
    """Appending a later revision leaves an earlier decision result unchanged."""
    spec = spec_for(Dataset.MACRO)
    decision_ts = datetime(2024, 3, 1, tzinfo=UTC)
    before = as_of(stamp_availability(_macro_two_release_frame().head(1), spec), spec, decision_ts)
    after = as_of(stamp_availability(_macro_two_release_frame(), spec), spec, decision_ts)
    assert before.height == 1
    assert after.equals(before)


def _holdings_two_filing_frame() -> pl.DataFrame:
    spec = spec_for(Dataset.ETF_HOLDINGS)
    return pl.DataFrame(
        {
            "etf_ticker": ["FOO", "FOO"],
            "report_date": [date(2024, 3, 31), date(2024, 3, 31)],
            "filing_date": [
                datetime(2024, 5, 1, tzinfo=UTC),
                datetime(2024, 5, 30, tzinfo=UTC),
            ],
            "holding_id": ["H1", "H1"],
            "issuer_name": ["Acme", "Acme"],
            "cusip": ["111", "111"],
            "isin": ["US111", "US111"],
            "lei": ["LEI1", "LEI1"],
            "weight_pct": [10.0, 12.0],
            "value_usd": [100.0, 120.0],
            "source": ["sec", "sec"],
            "retrieved_at": [
                datetime(2024, 5, 1, tzinfo=UTC),
                datetime(2024, 5, 30, tzinfo=UTC),
            ],
        },
        schema=dict(spec.columns),
    )


def test_holdings_amendment_selection() -> None:
    """Two filings for one holding resolve to the later visible filing."""
    spec = spec_for(Dataset.ETF_HOLDINGS)
    stamped = stamp_availability(_holdings_two_filing_frame(), spec)
    result = as_of(stamped, spec, datetime(2024, 6, 30, tzinfo=UTC))
    assert result.height == 1
    assert result.get_column("weight_pct")[0] == 12.0
    assert result.get_column("filing_date")[0] == datetime(2024, 5, 30, tzinfo=UTC)


def test_equal_time_ambiguity_fails() -> None:
    """Distinct revisions sharing observation key and availability fail closed."""
    spec = DatasetSpec(
        dataset=Dataset.MACRO,
        columns={
            "observation_date": pl.Date,
            "release_date": TS_DTYPE,
            "value": pl.Float64,
        },
        key=("observation_date", "release_date"),
        observation_column="observation_date",
        availability=AvailabilityRule(kind=AvailabilityKind.RELEASE_COLUMN, release_column="release_date"),
        missing_policy=MissingPolicy.FAIL,
        revisable=True,
        total_return_source=TotalReturnSource.NOT_APPLICABLE,
        schema_version="1",
        observation_key=("observation_date",),
    )
    frame = pl.DataFrame(
        {
            "observation_date": [date(2024, 1, 1), date(2024, 1, 1)],
            "release_date": [
                datetime(2024, 2, 10, tzinfo=UTC),
                datetime(2024, 3, 12, tzinfo=UTC),
            ],
            "value": [1.0, 1.2],
            AVAILABLE_AT: [
                datetime(2024, 4, 1, tzinfo=UTC),
                datetime(2024, 4, 1, tzinfo=UTC),
            ],
        },
        schema={
            "observation_date": pl.Date,
            "release_date": TS_DTYPE,
            "value": pl.Float64,
            AVAILABLE_AT: TS_DTYPE,
        },
    )
    try:
        as_of(frame, spec, datetime(2024, 4, 2, tzinfo=UTC))
        raised = False
    except ValueError:
        raised = True
    assert raised is True


def test_storage_versions_remain_distinct() -> None:
    """Two macro releases are distinct stored rows under quality checks."""
    from src.data.quality import validate_frame

    spec = spec_for(Dataset.MACRO)
    stamped = stamp_availability(_macro_two_release_frame(), spec)
    report = validate_frame(stamped, spec)
    assert not any(finding.code == "KEY_DUPLICATE" for finding in report.findings)


def test_as_of_missing_availability_column_fails() -> None:
    """A frame without the availability stamp fails closed."""
    import pytest

    spec = spec_for(Dataset.MACRO)
    frame = _macro_two_release_frame()
    assert AVAILABLE_AT not in frame.columns
    with pytest.raises(ValueError, match=r"available_at"):
        as_of(frame, spec, datetime(2024, 6, 30, tzinfo=UTC))


def test_as_of_missing_key_column_fails() -> None:
    """A revisable frame missing an observation key column fails closed."""
    import pytest

    spec = spec_for(Dataset.MACRO)
    stamped = stamp_availability(_macro_two_release_frame(), spec).drop("series_id")
    with pytest.raises(ValueError, match=r"key column"):
        as_of(stamped, spec, datetime(2024, 6, 30, tzinfo=UTC))
