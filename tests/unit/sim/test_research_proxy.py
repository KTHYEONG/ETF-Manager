"""Unit tests for the research-proxy index DCA simulator."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Final

import polars as pl
import pytest

from src.data.calendar import load_calendar
from src.data.pipeline import ingest
from src.data.schema import Dataset, spec_for
from src.policy.targets import PolicyId, all_policy_tickers
from src.sim.allocation import AllocationConfig, AllocationDataError
from src.sim.research_proxy import run_research_proxy

_CALENDAR = load_calendar("XNYS")
_RETRIEVED_AT = datetime(2024, 4, 1, 5, 0, tzinfo=UTC)
_CONTRIBUTION_KRW = 1_300_000.0
_PANEL_START: Final[date] = date(2024, 1, 10)
_PANEL_END: Final[date] = date(2024, 2, 28)


def _returns_frame(series_id: str, rets: dict[date, float]) -> pl.DataFrame:
    """Availability-stamped RESEARCH_RETURNS frame over the given session dates."""
    spec = spec_for(Dataset.RESEARCH_RETURNS)
    days = sorted(rets)
    raw = pl.DataFrame(
        {
            "series_id": [series_id] * len(days),
            "date": days,
            "simple_return": [rets[day] for day in days],
            "label": ["research_proxy"] * len(days),
            "source": ["synthetic"] * len(days),
            "retrieved_at": [_RETRIEVED_AT] * len(days),
        },
        schema=dict(spec.columns),
    )
    return ingest(raw, Dataset.RESEARCH_RETURNS)


def _fx_panel() -> pl.DataFrame:
    days = _CALENDAR.sessions(_PANEL_START, _PANEL_END)
    spec = spec_for(Dataset.FX)
    return ingest(
        pl.DataFrame(
            {
                "date": list(days),
                "usdkrw": [1300.0] * len(days),
                "source": ["synthetic"] * len(days),
                "retrieved_at": [_RETRIEVED_AT] * len(days),
            },
            schema=dict(spec.columns),
        ),
        Dataset.FX,
    )


def _constant_cpi() -> pl.DataFrame:
    """FIXED_LAG 45d stamping makes the level visible at every 2024 execution close."""
    spec = spec_for(Dataset.CPI)
    return ingest(
        pl.DataFrame(
            {
                "period_end": [date(2023, 12, 1)],
                "value": [100.0],
                "source": ["synthetic"],
                "retrieved_at": [_RETRIEVED_AT],
            },
            schema=dict(spec.columns),
        ),
        Dataset.CPI,
    )


def _config(policy: PolicyId = PolicyId.FF_PROXY) -> AllocationConfig:
    # Month-end signals land on 2024-01-31 and 2024-02-01, filling on the two
    # consecutive sessions 2024-02-01 / 2024-02-02 covered by the return frame.
    return AllocationConfig(
        policy=policy,
        start=date(2024, 1, 15),
        end=date(2024, 2, 1),
        monthly_contribution_krw=_CONTRIBUTION_KRW,
    )


@pytest.mark.parametrize("scenario_id", ["I9-C-no-price-splice"])
def test_i9_c_no_price_splice(scenario_id: str) -> None:
    """I9-C-no-price-splice"""
    fx = _fx_panel()
    cpi = _constant_cpi()

    for banned_series_id in all_policy_tickers():
        with pytest.raises(ValueError, match=r"research_proxy|I9"):
            run_research_proxy(
                _config(),
                _returns_frame(banned_series_id, {date(2024, 2, 1): 0.01, date(2024, 2, 2): -0.005}),
                fx,
                cpi,
            )

    result = run_research_proxy(
        _config(),
        _returns_frame("us_mkt_ff_daily", {date(2024, 2, 1): 0.01, date(2024, 2, 2): -0.005}),
        fx,
        cpi,
    )

    assert result.terminal_wealth_real_krw > 0.0
    # Lot 1 compounds through both returns; lot 2 buys at the final post-return close.
    index_first = 1.0 * (1.0 + 0.01)
    index_final = index_first * (1.0 - 0.005)
    expected_nominal = _CONTRIBUTION_KRW * (index_final / index_first + 1.0)
    assert result.terminal_wealth_krw == pytest.approx(expected_nominal, rel=1e-9)
    # Constant CPI leaves real wealth equal to nominal.
    assert result.terminal_wealth_real_krw == pytest.approx(expected_nominal, rel=1e-9)


@pytest.mark.parametrize("scenario_id", ["I9-C-no-price-splice"])
def test_i9_c_missing_or_wipeout_return_fails_closed(scenario_id: str) -> None:
    """I9-C-no-price-splice"""
    fx = _fx_panel()
    cpi = _constant_cpi()
    partial_window = {date(2024, 2, 1): 0.01}
    with pytest.raises(AllocationDataError, match="research_proxy"):
        run_research_proxy(_config(), _returns_frame("us_mkt_ff_daily", partial_window), fx, cpi)

    with pytest.raises(AllocationDataError, match="research_proxy"):
        run_research_proxy(
            _config(),
            _returns_frame("us_mkt_ff_daily", {date(2024, 2, 1): 0.01, date(2024, 2, 2): -1.0}),
            fx,
            cpi,
        )


@pytest.mark.parametrize("scenario_id", ["I9-C-no-price-splice"])
def test_i9_c_proxy_rejects_non_r1_policy(scenario_id: str) -> None:
    """I9-C-no-price-splice"""
    fx = _fx_panel()
    cpi = _constant_cpi()
    rets = _returns_frame("us_mkt_ff_daily", {date(2024, 2, 1): 0.01, date(2024, 2, 2): -0.005})
    with pytest.raises(ValueError, match=r"research_proxy|FF_PROXY"):
        run_research_proxy(_config(PolicyId.VT), rets, fx, cpi)
