"""Invariant tests for century-length Ken French research returns."""

from __future__ import annotations

import io
import zipfile
from datetime import date

import httpx
import polars as pl
import pytest

from src.data.providers.base import ProviderError
from src.data.providers.french import FrenchClient
from src.data.storage import RawPayload

_WINDOW: tuple[date, date] = (date(1926, 7, 31), date(1926, 12, 31))

_FACTOR_TEXT = """\
This file was created using a test CRSP database.

,Mkt-RF,SMB,HML,RF
192607,   1.20,  -0.10,   0.20,   0.30
192608,   2.00,  -0.20,   0.40,   0.30
192609,   3.00,  -0.30,   0.60,   0.30
192610,   4.00,  -0.40,   0.80,   0.30
192611,   5.00,  -0.50,   1.00,   0.30
192612,   6.00,  -0.60,   1.20,   0.30

Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
1926,  21.20,  -2.10,   4.20,   3.60
"""

_INDUSTRY_TEXT = """\
This file was created using a test CRSP database.

Average Value Weighted Returns -- Monthly
,NoDur,HiTec,Other

192607,   9.00,   1.10,  11.00
192608,   9.00,   2.20,  11.00
192609,   9.00,   3.30,  11.00
192610,   9.00,   4.40,  11.00
192611,   9.00,   5.50,  11.00
192612,   9.00,   6.60,  11.00

Average Equal Weighted Returns -- Monthly
,NoDur,HiTec,Other
192607,  19.00,  11.10,  21.00
192608,  19.00,  12.20,  21.00
192609,  19.00,  13.30,  21.00
192610,  19.00,  14.40,  21.00
192611,  19.00,  15.50,  21.00
192612,  19.00,  16.60,  21.00

Average Value Weighted Returns -- Annual
,NoDur,HiTec,Other
1926,  99.00, 111.00, 199.00

Number of Firms in Portfolios
,NoDur,HiTec,Other
192607,  100, 111, 199
"""

_DIVIDEND_TEXT = """\
This file was created using a test CRSP database.

Value Weight Returns -- Monthly
,Lo 30,Med 40,Hi 30
192607,   1.00,   2.00,   0.70
192608,   1.00,   2.00,   0.80
192609,   1.00,   2.00,   0.90
192610,   1.00,   2.00,   1.00
192611,   1.00,   2.00,   1.10
192612,   1.00,   2.00,   1.20

Equal Weight Returns -- Monthly
,Lo 30,Med 40,Hi 30
192607,   9.00,   9.00,   9.70
192608,   9.00,   9.00,   9.80
192609,   9.00,   9.00,   9.90
192610,   9.00,   9.00,  19.00
192611,   9.00,   9.00,  19.10
192612,   9.00,   9.00,  19.20
"""

_DEV_TEXT = """\
This file was created using a test database.

,Mkt-RF,SMB,HML,RF
192607,   1.00,   0.10,   0.20,   0.30
192608,   1.10,   0.10,   0.20,   0.30
192609,   1.20,   0.10,   0.20,   0.30
192610,   1.30,   0.10,   0.20,   0.30
192611,   1.40,   0.10,   0.20,   0.30
192612,   1.50,   0.10,   0.20,   0.30

Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
1926,   7.50,   0.60,   1.20,   1.80
"""

_EMERGING_TEXT = """\
This file was created using a test database.

,Mkt-RF,SMB,HML,RMW,CMA,RF
192607,   1.00,   0.10,   0.20,   0.30,   0.40,   0.30
192608,   1.10,   0.10,   0.20,   0.30,   0.40,   0.30
192609,   1.20,   0.10,   0.20,   0.30,   0.40,   0.30
192610,   1.30,   0.10,   0.20,   0.30,   0.40,   0.30
192611,   1.40,   0.10,   0.20,   0.30,   0.40,   0.30
192612,   1.50,   0.10,   0.20,   0.30,   0.40,   0.30

Annual Factors: January-December
,Mkt-RF,SMB,HML,RMW,CMA,RF
1926,   7.50,   0.60,   1.20,   1.80,   2.40,   1.80
"""


def _zip_bytes(member: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text)
    return buffer.getvalue()


def _client_serving(
    factor: bytes, industry: bytes, dividend: bytes, dev: bytes, emerging: bytes
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "10_Industry_Portfolios" in url:
            return httpx.Response(200, content=industry)
        if "Portfolios_Formed_on_D-P" in url:
            return httpx.Response(200, content=dividend)
        if "Developed_ex_US" in url:
            return httpx.Response(200, content=dev)
        if "Emerging_5" in url:
            return httpx.Response(200, content=emerging)
        return httpx.Response(200, content=factor)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _fetch(
    factor_text: str = _FACTOR_TEXT,
    industry_text: str = _INDUSTRY_TEXT,
    dividend_text: str = _DIVIDEND_TEXT,
    dev_text: str = _DEV_TEXT,
    emerging_text: str = _EMERGING_TEXT,
    *,
    window: tuple[date, date] = _WINDOW,
) -> tuple[RawPayload, pl.DataFrame]:
    factor = _zip_bytes("F-F_Research_Data_Factors.csv", factor_text)
    industry = _zip_bytes("10_Industry_Portfolios.csv", industry_text)
    dividend = _zip_bytes("Portfolios_Formed_on_D-P.csv", dividend_text)
    dev = _zip_bytes("Developed_ex_US_3_Factors.csv", dev_text)
    emerging = _zip_bytes("Emerging_5_Factors.csv", emerging_text)
    with _client_serving(factor, industry, dividend, dev, emerging) as client:
        return FrenchClient(client).fetch_monthly_research_returns(*window)


def test_fetch_monthly_research_returns_uses_value_weighted_hitec_only() -> None:
    """Value-weighted monthly HiTec is used; later industry blocks are ignored."""
    factor = _zip_bytes("F-F_Research_Data_Factors.csv", _FACTOR_TEXT)
    industry = _zip_bytes("10_Industry_Portfolios.csv", _INDUSTRY_TEXT)
    dividend = _zip_bytes("Portfolios_Formed_on_D-P.csv", _DIVIDEND_TEXT)
    dev = _zip_bytes("Developed_ex_US_3_Factors.csv", _DEV_TEXT)
    emerging = _zip_bytes("Emerging_5_Factors.csv", _EMERGING_TEXT)

    with _client_serving(factor, industry, dividend, dev, emerging) as client:
        payload, frame = FrenchClient(client).fetch_monthly_research_returns(*_WINDOW)

    hitec = frame.filter(pl.col("series_id") == "ff_hitec_monthly").sort("period_end")
    assert hitec.get_column("simple_return").to_list() == pytest.approx([0.011, 0.022, 0.033, 0.044, 0.055, 0.066])
    assert hitec.get_column("label").unique().to_list() == ["research_proxy"]
    assert hitec.get_column("source").unique().to_list() == ["ken_french"]
    assert payload.content == factor + b"\n" + industry + b"\n" + dividend + b"\n" + dev + b"\n" + emerging
    assert payload.request_params == {
        "factor_file": "F-F_Research_Data_Factors_CSV.zip",
        "industry_file": "10_Industry_Portfolios_CSV.zip",
        "dividend_file": "Portfolios_Formed_on_D-P_CSV.zip",
        "dev_ex_us_file": "Developed_ex_US_3_Factors_CSV.zip",
        "emerging_file": "Emerging_5_Factors_CSV.zip",
        "start": "1926-07-31",
        "end": "1926-12-31",
    }


def test_fetch_monthly_research_returns_market_is_excess_plus_risk_free() -> None:
    """Monthly market return is Mkt-RF plus RF in decimal units."""
    _, frame = _fetch()

    market = frame.filter(
        (pl.col("series_id") == "ff_mkt_monthly") & (pl.col("period_end") == date(1926, 7, 31))
    ).row(0, named=True)
    assert market["simple_return"] == pytest.approx(0.015)


def test_fetch_monthly_research_returns_window_filter_is_inclusive() -> None:
    """The requested start and end month-ends are both retained."""
    window = (date(1926, 8, 31), date(1926, 11, 30))

    _, frame = _fetch(window=window)

    expected_months = [
        date(1926, 8, 31),
        date(1926, 9, 30),
        date(1926, 10, 31),
        date(1926, 11, 30),
    ]
    assert frame.height == 20
    for series_id in (
        "ff_mkt_monthly",
        "ff_hitec_monthly",
        "ff_dp_hi30_monthly",
        "ff_dev_ex_us_mkt_monthly",
        "ff_em_mkt_monthly",
    ):
        months = frame.filter(pl.col("series_id") == series_id).get_column("period_end").to_list()
        assert months == expected_months


def test_fetch_monthly_research_returns_sentinel_fails_closed() -> None:
    """A sentinel HiTec percentage is reported with its exact cell."""
    industry = _INDUSTRY_TEXT.replace("192608,   9.00,   2.20", "192608,   9.00, -99.99")

    with pytest.raises(ProviderError) as exc_info:
        _fetch(industry_text=industry)

    assert "HiTec" in str(exc_info.value)
    assert "-99.99" in str(exc_info.value)


def test_fetch_monthly_research_returns_month_sets_must_match() -> None:
    """A month present in only one source archive fails closed."""
    factor = _FACTOR_TEXT.replace("192609,   3.00,  -0.30,   0.60,   0.30\n", "")

    with pytest.raises(ProviderError, match="month sets differ"):
        _fetch(factor_text=factor)


def test_fetch_monthly_research_returns_missing_hitec_fails_closed() -> None:
    """A first-block header without HiTec fails before later blocks are considered."""
    industry = _INDUSTRY_TEXT.replace(",NoDur,HiTec,Other", ",NoDur,Telcm,Other", 1)

    with pytest.raises(ProviderError, match="HiTec"):
        _fetch(industry_text=industry)


def test_fetch_monthly_research_returns_empty_block_fails_closed() -> None:
    """A header-only monthly block is not a valid parse."""
    factor = "\n,Mkt-RF,SMB,HML,RF\n\nAnnual Factors: January-December\n"

    with pytest.raises(ProviderError, match="no monthly rows"):
        _fetch(factor_text=factor)


def test_fetch_monthly_research_returns_empty_window_fails_closed() -> None:
    """A valid source with no rows in the requested window fails closed."""
    window = (date(1800, 1, 1), date(1800, 12, 31))

    with pytest.raises(ProviderError, match="no monthly rows in the requested window"):
        _fetch(window=window)


def test_fetch_monthly_research_returns_missing_section_fails_closed() -> None:
    """The value-weighted monthly section is required explicitly."""
    industry = _INDUSTRY_TEXT.replace(
        "Average Value Weighted Returns -- Monthly",
        "Average Value Weighted Returns -- Quarterly",
        1,
    ).replace(",NoDur,HiTec,Other", ",NoDur,Telcm,Other", 1)

    with pytest.raises(ProviderError, match="missing section"):
        _fetch(industry_text=industry)


def test_fetch_monthly_research_returns_emits_five_series_with_independent_coverage() -> None:
    """Extra series keep their own start months while market/HiTec share one set."""
    factor_text = """\
,Mkt-RF,SMB,HML,RF
192706,   1.00,   0.10,   0.20,   0.30
192707,   1.10,   0.10,   0.20,   0.30
192708,   1.20,   0.10,   0.20,   0.30
192709,   1.30,   0.10,   0.20,   0.30

Annual Factors: January-December
"""
    industry_text = """\
Average Value Weighted Returns -- Monthly
,NoDur,HiTec,Other
192706,   9.00,   1.00,  11.00
192707,   9.00,   1.10,  11.00
192708,   9.00,   1.20,  11.00
192709,   9.00,   1.30,  11.00

Average Equal Weighted Returns -- Monthly
"""
    dividend_text = """\
Value Weight Returns -- Monthly
,Lo 30,Med 40,Hi 30
192707,   1.00,   2.00,   0.70
192708,   1.00,   2.00,   0.80
192709,   1.00,   2.00,   0.90

Equal Weight Returns -- Monthly
"""
    dev_text = """\
,Mkt-RF,SMB,HML,RF
192708,   1.00,   0.10,   0.20,   0.30
192709,   1.10,   0.10,   0.20,   0.30

Annual Factors: January-December
"""
    emerging_text = """\
,Mkt-RF,SMB,HML,RMW,CMA,RF
192708,   1.00,   0.10,   0.20,   0.30,   0.40,   0.30
192709,   1.10,   0.10,   0.20,   0.30,   0.40,   0.30

Annual Factors: January-December
"""
    window = (date(1927, 6, 30), date(1927, 9, 30))

    _, frame = _fetch(
        factor_text, industry_text, dividend_text, dev_text, emerging_text, window=window
    )

    ids = sorted(frame.get_column("series_id").unique().to_list())
    assert ids == [
        "ff_dev_ex_us_mkt_monthly",
        "ff_dp_hi30_monthly",
        "ff_em_mkt_monthly",
        "ff_hitec_monthly",
        "ff_mkt_monthly",
    ]
    by_id = {sid: frame.filter(pl.col("series_id") == sid).get_column("period_end").to_list() for sid in ids}
    assert by_id["ff_mkt_monthly"][0] == date(1927, 6, 30)
    assert by_id["ff_hitec_monthly"][0] == date(1927, 6, 30)
    assert by_id["ff_dp_hi30_monthly"][0] == date(1927, 7, 31)
    assert by_id["ff_dev_ex_us_mkt_monthly"][0] == date(1927, 8, 31)
    assert by_id["ff_em_mkt_monthly"][0] == date(1927, 8, 31)


def test_fetch_monthly_research_returns_global_market_adds_risk_free() -> None:
    """Developed ex-US market return is Mkt-RF plus RF in decimal units."""
    dev_text = """\
,Mkt-RF,SMB,HML,RF
192607,   1.99,   0.10,   0.20,   0.68
192608,   1.10,   0.10,   0.20,   0.30
192609,   1.20,   0.10,   0.20,   0.30
192610,   1.30,   0.10,   0.20,   0.30
192611,   1.40,   0.10,   0.20,   0.30
192612,   1.50,   0.10,   0.20,   0.30

Annual Factors: January-December
"""

    _, frame = _fetch(dev_text=dev_text)

    row = frame.filter(
        (pl.col("series_id") == "ff_dev_ex_us_mkt_monthly") & (pl.col("period_end") == date(1926, 7, 31))
    ).row(0, named=True)
    assert row["simple_return"] == pytest.approx(0.0267, abs=1e-12)


def test_fetch_monthly_research_returns_unused_emerging_sentinel_is_ignored() -> None:
    """A sentinel in an unread emerging column does not fail the fetch."""
    emerging = _EMERGING_TEXT.replace(
        "192607,   1.00,   0.10,   0.20,   0.30,   0.40,   0.30",
        "192607,   1.00,   0.10,   0.20, -99.99,   0.40,   0.30",
    )

    _, frame = _fetch(emerging_text=emerging)

    row = frame.filter(
        (pl.col("series_id") == "ff_em_mkt_monthly") & (pl.col("period_end") == date(1926, 7, 31))
    ).row(0, named=True)
    assert row["simple_return"] == pytest.approx(0.013)


def test_fetch_monthly_research_returns_dividend_sentinel_fails_closed() -> None:
    """A sentinel in the dividend Hi 30 column fails with the column name."""
    dividend = _DIVIDEND_TEXT.replace(
        "192608,   1.00,   2.00,   0.80", "192608,   1.00,   2.00, -99.99"
    )

    with pytest.raises(ProviderError) as exc_info:
        _fetch(dividend_text=dividend)

    assert "Hi 30" in str(exc_info.value)


def test_fetch_monthly_research_returns_gap_in_extra_series_fails_closed() -> None:
    """A missing month inside an extra series names the series and month."""
    dev_text = """\
,Mkt-RF,SMB,HML,RF
192708,   1.00,   0.10,   0.20,   0.30
192710,   1.10,   0.10,   0.20,   0.30

Annual Factors: January-December
"""
    factor_text = """\
,Mkt-RF,SMB,HML,RF
192708,   1.00,   0.10,   0.20,   0.30
192709,   1.10,   0.10,   0.20,   0.30
192710,   1.20,   0.10,   0.20,   0.30

Annual Factors: January-December
"""
    industry_text = """\
Average Value Weighted Returns -- Monthly
,NoDur,HiTec,Other
192708,   9.00,   1.00,  11.00
192709,   9.00,   1.10,  11.00
192710,   9.00,   1.20,  11.00

Average Equal Weighted Returns -- Monthly
"""
    dividend_text = """\
Value Weight Returns -- Monthly
,Lo 30,Med 40,Hi 30
192708,   1.00,   2.00,   0.70
192709,   1.00,   2.00,   0.80
192710,   1.00,   2.00,   0.90

Equal Weight Returns -- Monthly
"""
    emerging_text = """\
,Mkt-RF,SMB,HML,RMW,CMA,RF
192708,   1.00,   0.10,   0.20,   0.30,   0.40,   0.30
192709,   1.10,   0.10,   0.20,   0.30,   0.40,   0.30
192710,   1.20,   0.10,   0.20,   0.30,   0.40,   0.30

Annual Factors: January-December
"""
    window = (date(1927, 8, 31), date(1927, 10, 31))

    with pytest.raises(ProviderError) as exc_info:
        _fetch(factor_text, industry_text, dividend_text, dev_text, emerging_text, window=window)

    assert "ff_dev_ex_us_mkt_monthly" in str(exc_info.value)
    assert "1927-09" in str(exc_info.value)


def test_fetch_monthly_research_returns_dividend_uses_value_weight_only() -> None:
    """Distinct equal-weight Hi 30 values never leak into the dividend series."""
    _, frame = _fetch()

    dividend = frame.filter(pl.col("series_id") == "ff_dp_hi30_monthly").sort("period_end")
    assert dividend.get_column("simple_return").to_list() == pytest.approx(
        [0.007, 0.008, 0.009, 0.010, 0.011, 0.012]
    )


def test_fetch_monthly_research_returns_extra_series_empty_in_window_fails_closed() -> None:
    """A window ending before the dividend start names the dividend series."""
    narrow = (date(1926, 6, 30), date(1926, 6, 30))

    with pytest.raises(ProviderError) as exc_info:
        _fetch(window=narrow)

    assert "ff_dp_hi30_monthly" in str(exc_info.value)
