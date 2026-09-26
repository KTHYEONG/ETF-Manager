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


def _zip_bytes(member: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text)
    return buffer.getvalue()


def _client_serving(factor: bytes, industry: bytes) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        body = industry if "10_Industry_Portfolios" in str(request.url) else factor
        return httpx.Response(200, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _fetch(
    factor_text: str = _FACTOR_TEXT,
    industry_text: str = _INDUSTRY_TEXT,
    *,
    window: tuple[date, date] = _WINDOW,
) -> tuple[RawPayload, pl.DataFrame]:
    factor = _zip_bytes("F-F_Research_Data_Factors.csv", factor_text)
    industry = _zip_bytes("10_Industry_Portfolios.csv", industry_text)
    with _client_serving(factor, industry) as client:
        return FrenchClient(client).fetch_monthly_research_returns(*window)


def test_fetch_monthly_research_returns_uses_value_weighted_hitec_only() -> None:
    """Value-weighted monthly HiTec is used; later industry blocks are ignored."""
    factor = _zip_bytes("F-F_Research_Data_Factors.csv", _FACTOR_TEXT)
    industry = _zip_bytes("10_Industry_Portfolios.csv", _INDUSTRY_TEXT)

    with _client_serving(factor, industry) as client:
        payload, frame = FrenchClient(client).fetch_monthly_research_returns(*_WINDOW)

    hitec = frame.filter(pl.col("series_id") == "ff_hitec_monthly").sort("period_end")
    assert hitec.get_column("simple_return").to_list() == pytest.approx([0.011, 0.022, 0.033, 0.044, 0.055, 0.066])
    assert hitec.get_column("label").unique().to_list() == ["research_proxy"]
    assert hitec.get_column("source").unique().to_list() == ["ken_french"]
    assert payload.content == factor + b"\n" + industry
    assert payload.request_params == {
        "factor_file": "F-F_Research_Data_Factors_CSV.zip",
        "industry_file": "10_Industry_Portfolios_CSV.zip",
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
    assert frame.height == 8
    for series_id in ("ff_mkt_monthly", "ff_hitec_monthly"):
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
