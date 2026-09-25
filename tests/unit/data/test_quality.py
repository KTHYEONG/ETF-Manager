"""Unit tests for the pure, fail-closed data-quality gate."""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest

from src.data.calendar import TradingCalendar, load_calendar
from src.data.pit import stamp_availability
from src.data.quality import (
    DataQualityError,
    FindingSeverity,
    QualityFinding,
    QualityReport,
    enforce,
    validate_frame,
)
from src.data.schema import Dataset, DatasetSpec, spec_for

_RETRIEVED_AT = datetime(2024, 2, 1, 5, 0, tzinfo=UTC)


def _prices_frame(dates: list[date], closes: list[float], ticker: str = "AAA") -> pl.DataFrame:
    spec = spec_for(Dataset.PRICES)
    n = len(dates)
    return pl.DataFrame(
        {
            "ticker": [ticker] * n,
            "date": list(dates),
            "open": [value * 0.98 for value in closes],
            "high": [value * 1.02 for value in closes],
            "low": [value * 0.97 for value in closes],
            "close": list(closes),
            "volume": [10_000] * n,
            "adjusted_close": list(closes),
            "dividend": [0.0] * n,
            "split_factor": [1.0] * n,
            "source": ["synthetic"] * n,
            "retrieved_at": [_RETRIEVED_AT] * n,
        },
        schema=dict(spec.columns),
    )


def _stamp(frame: pl.DataFrame, spec: DatasetSpec, calendar: TradingCalendar | None) -> pl.DataFrame:
    return stamp_availability(frame, spec, calendar)


def _errors(report: QualityReport) -> tuple[QualityFinding, ...]:
    return tuple(finding for finding in report.findings if finding.severity is FindingSeverity.ERROR)


@pytest.mark.parametrize("scenario_id", ["QL-B01-schema-exactness"])
def test_schema_exactness(scenario_id: str) -> None:
    """QL-B01-schema-exactness"""
    calendar = load_calendar("XNYS")
    spec = spec_for(Dataset.PRICES)
    base = _prices_frame([date(2024, 1, 30)], [100.0])
    stamped = _stamp(base, spec, calendar)
    snapshot = stamped.clone()

    clean_report = validate_frame(stamped, spec, calendar)
    assert _errors(clean_report) == ()

    missing_column_frame = stamped.drop("close")
    missing_report = validate_frame(missing_column_frame, spec, calendar)
    columns_findings = [f for f in missing_report.findings if f.code == "SCHEMA_COLUMNS"]
    assert len(columns_findings) == 1
    assert columns_findings[0].severity is FindingSeverity.ERROR
    assert columns_findings[0].row_count == 0

    extra_column_frame = stamped.with_columns(pl.lit(1, dtype=pl.Int64).alias("unexpected_column"))
    extra_report = validate_frame(extra_column_frame, spec, calendar)
    extra_findings = [f for f in extra_report.findings if f.code == "SCHEMA_COLUMNS"]
    assert len(extra_findings) == 1
    assert len(_errors(extra_report)) == 1

    wrong_dtype_frame = stamped.with_columns(pl.col("volume").cast(pl.Float64))
    dtype_report = validate_frame(wrong_dtype_frame, spec, calendar)
    dtype_findings = [f for f in dtype_report.findings if f.code == "SCHEMA_DTYPE"]
    assert len(dtype_findings) == 1
    assert dtype_findings[0].severity is FindingSeverity.ERROR

    assert stamped.equals(snapshot)


@pytest.mark.parametrize("scenario_id", ["QL-B02-keys-and-nulls"])
def test_keys_and_nulls(scenario_id: str) -> None:
    """QL-B02-keys-and-nulls"""
    calendar = load_calendar("XNYS")
    spec = spec_for(Dataset.PRICES)
    single = _prices_frame([date(2024, 1, 30)], [100.0])

    duplicated = pl.concat([single, single])
    duplicated_snapshot = duplicated.clone()
    duplicate_report = validate_frame(_stamp(duplicated, spec, calendar), spec, calendar)
    duplicate_findings = [f for f in duplicate_report.findings if f.code == "KEY_DUPLICATE"]
    assert len(duplicate_findings) == 1
    assert duplicate_findings[0].severity is FindingSeverity.ERROR
    assert duplicate_findings[0].row_count == 2
    assert duplicated.equals(duplicated_snapshot)

    nulled = single.with_columns(
        pl.when(pl.col("date") == date(2024, 1, 30))
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    nulled_snapshot = nulled.clone()
    null_report = validate_frame(_stamp(nulled, spec, calendar), spec, calendar)
    null_findings = [f for f in null_report.findings if f.code == "REQUIRED_NULL"]
    assert len(null_findings) == 1
    assert null_findings[0].severity is FindingSeverity.ERROR
    assert null_findings[0].row_count == 1
    assert nulled.equals(nulled_snapshot)


@pytest.mark.parametrize("scenario_id", ["QL-B03-market-consistency"])
def test_market_consistency(scenario_id: str) -> None:
    """QL-B03-market-consistency"""
    calendar = load_calendar("XNYS")
    spec = spec_for(Dataset.PRICES)

    low_too_high = _prices_frame([date(2024, 1, 30)], [100.0]).with_columns(pl.lit(101.0).alias("low"))
    high_too_low = _prices_frame([date(2024, 1, 30)], [100.0]).with_columns(pl.lit(99.0).alias("high"))
    non_positive_open = _prices_frame([date(2024, 1, 30)], [100.0]).with_columns(pl.lit(-1.0).alias("open"))

    for broken in (low_too_high, high_too_low, non_positive_open):
        report = validate_frame(_stamp(broken, spec, calendar), spec, calendar)
        ohlc_findings = [f for f in report.findings if f.code == "OHLC_INVALID"]
        assert len(ohlc_findings) == 1
        assert ohlc_findings[0].severity is FindingSeverity.ERROR
        assert ohlc_findings[0].row_count == 1

    outlier_pair = _prices_frame([date(2024, 1, 30), date(2024, 1, 31)], [100.0, 250.0])
    outlier_stamped = _stamp(outlier_pair, spec, calendar)
    outlier_snapshot = outlier_stamped.clone()
    outlier_report = validate_frame(outlier_stamped, spec, calendar)
    outlier_findings = [f for f in outlier_report.findings if f.code == "RETURN_OUTLIER"]
    assert len(outlier_findings) == 1
    assert outlier_findings[0].severity is FindingSeverity.WARN
    assert outlier_report.has_errors is False
    assert outlier_stamped.equals(outlier_snapshot)


@pytest.mark.parametrize("scenario_id", ["QL-B04-session-and-gap-policy"])
def test_session_and_gap_policy(scenario_id: str) -> None:
    """QL-B04-session-and-gap-policy"""
    calendar = load_calendar("XNYS")
    spec = spec_for(Dataset.PRICES)
    gap_panel = _prices_frame([date(2024, 1, 29), date(2024, 1, 31)], [100.0, 101.0])
    gap_report = validate_frame(_stamp(gap_panel, spec, calendar), spec, calendar)
    session_findings = [f for f in gap_report.findings if f.code == "SESSION_MISSING"]
    assert len(session_findings) == 1
    assert session_findings[0].severity is FindingSeverity.ERROR
    assert session_findings[0].row_count == 1
    assert "2024-01-30" in session_findings[0].message

    macro_spec = spec_for(Dataset.MACRO)
    macro_frame = pl.DataFrame(
        {
            "series_id": ["VIXCLS"],
            "observation_date": [date(2024, 1, 1)],
            "release_date": [datetime(2024, 2, 14, tzinfo=UTC)],
            "value": [None],
        },
        schema=dict(macro_spec.columns),
    )
    macro_stamped = stamp_availability(macro_frame, macro_spec)
    macro_report = validate_frame(macro_stamped, macro_spec, None)
    assert all(finding.code != "REQUIRED_NULL" for finding in macro_report.findings)
    assert macro_report.has_errors is False


@pytest.mark.parametrize("scenario_id", ["QL-B05-enforcement-boundary"])
def test_enforcement_boundary(scenario_id: str) -> None:
    """QL-B05-enforcement-boundary"""
    calendar = load_calendar("XNYS")
    spec = spec_for(Dataset.PRICES)
    single = _prices_frame([date(2024, 1, 30)], [100.0])

    duplicated = pl.concat([single, single])
    error_report = validate_frame(_stamp(duplicated, spec, calendar), spec, calendar)
    assert error_report.has_errors is True
    with pytest.raises(DataQualityError) as excinfo:
        enforce(error_report)
    assert excinfo.value.report is error_report

    outlier_pair = _prices_frame([date(2024, 1, 30), date(2024, 1, 31)], [100.0, 250.0])
    warn_only_report = validate_frame(_stamp(outlier_pair, spec, calendar), spec, calendar)
    assert warn_only_report.has_errors is False
    assert any(f.severity is FindingSeverity.WARN for f in warn_only_report.findings)
    assert enforce(warn_only_report) is None

    with pytest.raises(ValueError, match="requires a calendar"):
        validate_frame(_stamp(single, spec, calendar), spec, None)


def _fx_frame(usdkrw_values: list[float | None], dates: list[date]) -> pl.DataFrame:
    spec = spec_for(Dataset.FX)
    return pl.DataFrame(
        {
            "date": list(dates),
            "usdkrw": list(usdkrw_values),
            "source": ["synthetic"] * len(dates),
            "retrieved_at": [_RETRIEVED_AT] * len(dates),
        },
        schema=dict(spec.columns),
    )


def _cpi_frame(values: list[float | None], period_end: date = date(2024, 1, 31)) -> pl.DataFrame:
    spec = spec_for(Dataset.CPI)
    return pl.DataFrame(
        {
            "period_end": [period_end],
            "value": list(values),
            "source": ["synthetic"],
            "retrieved_at": [_RETRIEVED_AT],
        },
        schema=dict(spec.columns),
    )


def _macro_frame(values: list[float | None]) -> pl.DataFrame:
    spec = spec_for(Dataset.MACRO)
    return pl.DataFrame(
        {
            "series_id": ["VIXCLS"] * len(values),
            "observation_date": [date(2024, 1, 1)] * len(values),
            "release_date": [datetime(2024, 2, 14, tzinfo=UTC)] * len(values),
            "value": list(values),
        },
        schema=dict(spec.columns),
    )


def test_validate_frame_rejects_nonfinite_adjusted_close() -> None:
    """NaN adjusted close is ERROR and blocks enforcement."""
    calendar = load_calendar("XNYS")
    spec = spec_for(Dataset.PRICES)
    broken = _prices_frame([date(2024, 1, 30)], [100.0]).with_columns(
        pl.lit(float("nan")).alias("adjusted_close")
    )
    snapshot = broken.clone()
    report = validate_frame(_stamp(broken, spec, calendar), spec, calendar)
    findings = [f for f in report.findings if f.code == "NUMERIC_NONFINITE"]
    assert len(findings) == 1
    assert findings[0].severity is FindingSeverity.ERROR
    assert findings[0].row_count == 1
    with pytest.raises(DataQualityError):
        enforce(report)
    assert broken.equals(snapshot)


def test_validate_frame_retains_nullable_fx_gap() -> None:
    """Null FX level under EXPLICIT_GAP stays and raises no numeric finding."""
    calendar = load_calendar("XNYS")
    spec = spec_for(Dataset.FX)
    assert spec.missing_policy.name == "EXPLICIT_GAP"
    frame = _fx_frame([None], [date(2024, 1, 30)])
    stamped = _stamp(frame, spec, calendar)
    report = validate_frame(stamped, spec, calendar)
    assert all(finding.code != "NUMERIC_NONFINITE" for finding in report.findings)
    assert report.has_errors is False
    assert stamped.get_column("usdkrw").to_list() == [None]


def test_validate_frame_rejects_infinite_nullable_macro_value() -> None:
    """Infinite macro value is ERROR even though the column is nullable."""
    spec = spec_for(Dataset.MACRO)
    assert "value" in spec.nullable_columns
    stamped = stamp_availability(_macro_frame([float("inf")]), spec)
    report = validate_frame(stamped, spec, None)
    findings = [f for f in report.findings if f.code == "NUMERIC_NONFINITE"]
    assert len(findings) == 1
    assert findings[0].severity is FindingSeverity.ERROR
    assert findings[0].row_count == 1
    with pytest.raises(DataQualityError):
        enforce(report)


def test_validate_frame_rejects_invalid_price_magnitudes() -> None:
    """Zero adjusted/split or negative dividend/volume is ERROR counted once per row."""
    calendar = load_calendar("XNYS")
    spec = spec_for(Dataset.PRICES)
    base = _prices_frame([date(2024, 1, 30)], [100.0])
    cases = [
        base.with_columns(pl.lit(0.0).alias("adjusted_close")),
        base.with_columns(pl.lit(0.0).alias("split_factor")),
        base.with_columns(pl.lit(-0.5).alias("dividend")),
        base.with_columns(pl.lit(-5, dtype=pl.Int64).alias("volume")),
    ]
    for broken in cases:
        report = validate_frame(_stamp(broken, spec, calendar), spec, calendar)
        findings = [f for f in report.findings if f.code == "PRICE_FIELD_INVALID"]
        assert len(findings) == 1
        assert findings[0].severity is FindingSeverity.ERROR
        assert findings[0].row_count == 1
    combined = base.with_columns(
        pl.lit(0.0).alias("adjusted_close"),
        pl.lit(0.0).alias("split_factor"),
        pl.lit(-0.5).alias("dividend"),
        pl.lit(-5, dtype=pl.Int64).alias("volume"),
    )
    combined_report = validate_frame(_stamp(combined, spec, calendar), spec, calendar)
    combined_findings = [f for f in combined_report.findings if f.code == "PRICE_FIELD_INVALID"]
    assert len(combined_findings) == 1
    assert combined_findings[0].row_count == 1


def test_validate_frame_rejects_nonpositive_currency_and_cpi() -> None:
    """Zero FX level and negative CPI level are ERROR."""
    calendar = load_calendar("XNYS")
    fx_spec = spec_for(Dataset.FX)
    fx_report = validate_frame(
        _stamp(_fx_frame([0.0], [date(2024, 1, 30)]), fx_spec, calendar), fx_spec, calendar
    )
    fx_findings = [f for f in fx_report.findings if f.code == "LEVEL_NONPOSITIVE"]
    assert len(fx_findings) == 1
    assert fx_findings[0].severity is FindingSeverity.ERROR
    assert fx_findings[0].row_count == 1

    cpi_spec = spec_for(Dataset.CPI)
    cpi_report = validate_frame(stamp_availability(_cpi_frame([-1.0]), cpi_spec), cpi_spec, None)
    cpi_findings = [f for f in cpi_report.findings if f.code == "LEVEL_NONPOSITIVE"]
    assert len(cpi_findings) == 1
    assert cpi_findings[0].severity is FindingSeverity.ERROR
    assert cpi_findings[0].row_count == 1


def test_validate_frame_keeps_outlier_warning_without_blocking() -> None:
    """A finite large move stays WARN and passes enforcement."""
    calendar = load_calendar("XNYS")
    spec = spec_for(Dataset.PRICES)
    pair = _prices_frame([date(2024, 1, 30), date(2024, 1, 31)], [100.0, 250.0])
    report = validate_frame(_stamp(pair, spec, calendar), spec, calendar)
    outlier = [f for f in report.findings if f.code == "RETURN_OUTLIER"]
    assert len(outlier) == 1
    assert outlier[0].severity is FindingSeverity.WARN
    assert report.has_errors is False
    assert enforce(report) is None
