"""Unit tests for named strategic target-weight policies."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Final

import polars as pl
import pytest

from src.data.calendar import load_calendar
from src.policy.adaptive_contribution import OPERATIONAL_ADAPTIVE_CONTRIBUTION
from src.data.pipeline import ingest
from src.data.pit import TS_DTYPE
from src.data.schema import Dataset, spec_for
from src.policy.targets import (
    POLICY_ALIASES,
    UNIVERSE_VEHICLE,
    OPERATIONAL_POLICY_ID,
    PolicyError,
    PolicyId,
    UsEquityUniverse,
    all_policy_tickers,
    policy_sleeves,
    resolve_targets,
)

_CALENDAR = load_calendar("XNYS")
_RETRIEVED_AT = datetime(2024, 4, 1, 5, 0, tzinfo=UTC)
_SIGNAL_AT = datetime(2024, 1, 31, 21, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("scenario_id", "policy_value"),
    [
        ("POL-G03-static-sum-one", "vt"),
        ("POL-G03-static-sum-one", "vti"),
        ("POL-G03-static-sum-one", "world_split"),
        ("POL-G03-static-sum-one", "vt_bnd"),
        ("POL-G03-static-sum-one", "vt_treas"),
        ("POL-G03-static-sum-one", "vti_vtv"),
    ],
)
def test_pol_g03_static_sum_one(scenario_id: str, policy_value: str) -> None:
    """POL-G03-static-sum-one"""
    targets = resolve_targets(PolicyId.parse(policy_value), pl.DataFrame(), _SIGNAL_AT)

    weights = list(targets.values())
    assert abs(sum(weights) - 1.0) <= 1e-6
    assert min(weights) >= 0.0
    if policy_value == "world_split":
        assert targets == {"VTI": 0.5, "VEA": 0.3, "VWO": 0.2}
    if policy_value == "vti_vtv":
        assert targets == {"VTI": 0.8, "VTV": 0.2}


@pytest.mark.parametrize("scenario_id", ["POL-D-s7-large-cap-ivv"])
def test_pol_d_s7_large_cap_ivv(scenario_id: str) -> None:
    """POL-D-s7-large-cap-ivv"""
    assert UNIVERSE_VEHICLE[UsEquityUniverse.LARGE_CAP] == "IVV"
    assert UNIVERSE_VEHICLE[UsEquityUniverse.TOTAL_MARKET] == "VTI"

    s7 = resolve_targets(PolicyId.IVV, pl.DataFrame(), _SIGNAL_AT)
    s1 = resolve_targets(PolicyId.VTI, pl.DataFrame(), _SIGNAL_AT)

    assert s7 == {"IVV": 1.0}
    assert s1 == {"VTI": 1.0}
    for weights in (list(s7.values()), list(s1.values())):
        assert min(weights) >= 0.0
        assert abs(sum(weights) - 1.0) <= 1e-6

    banned = {"s9_voo_qqq", "qqqm"}
    assert banned.isdisjoint(member.value for member in PolicyId)


@pytest.mark.parametrize("scenario_id", ["POL-O-operational-qqq"])
def test_pol_o_operational_qqq(scenario_id: str) -> None:
    """POL-O-operational-qqq"""
    assert OPERATIONAL_POLICY_ID is PolicyId.QQQ
    assert resolve_targets(OPERATIONAL_POLICY_ID, pl.DataFrame(), _SIGNAL_AT) == {"QQQ": 1.0}


@pytest.mark.parametrize("scenario_id", ["POL-AF-operational-adaptive"])
def test_pol_af_operational_adaptive(scenario_id: str) -> None:
    """POL-AF-operational-adaptive"""
    assert OPERATIONAL_ADAPTIVE_CONTRIBUTION.rank_window == 126
    assert OPERATIONAL_ADAPTIVE_CONTRIBUTION.downside_power == pytest.approx(4.0)
    assert OPERATIONAL_ADAPTIVE_CONTRIBUTION.upside_power == pytest.approx(0.25)
    assert OPERATIONAL_ADAPTIVE_CONTRIBUTION.neutral_deadband == pytest.approx(5.0)
    assert OPERATIONAL_ADAPTIVE_CONTRIBUTION.include_vol_dampener is False
    assert OPERATIONAL_ADAPTIVE_CONTRIBUTION.min_multiplier == pytest.approx(0.0)
    assert OPERATIONAL_ADAPTIVE_CONTRIBUTION.max_multiplier == pytest.approx(2.0)


@pytest.mark.parametrize("scenario_id", ["POL-N-qqq-nasdaq"])
def test_pol_n_qqq_nasdaq(scenario_id: str) -> None:
    """POL-N-qqq-nasdaq"""
    assert UNIVERSE_VEHICLE[UsEquityUniverse.NASDAQ_100] == "QQQ"

    targets = resolve_targets(PolicyId.QQQ, pl.DataFrame(), _SIGNAL_AT)

    assert targets == {"QQQ": 1.0}
    assert min(targets.values()) >= 0.0
    assert abs(sum(targets.values()) - 1.0) <= 1e-6
    assert PolicyId.parse("s8_us_nasdaq") is PolicyId.QQQ
    assert "QQQ" in all_policy_tickers()

    banned = {"s9_voo_qqq", "qqqm"}
    assert banned.isdisjoint(member.value for member in PolicyId)


@pytest.mark.parametrize("scenario_id", ["TGT-W1-sleeve-universe"])
def test_tgt_w1_sleeve_universe(scenario_id: str) -> None:
    """TGT-W1-sleeve-universe"""
    assert all_policy_tickers() == ("BND", "IEF", "IVV", "QQQ", "TLT", "VEA", "VT", "VTI", "VTV", "VWO")
    union: set[str] = set()
    for member in PolicyId:
        union.update(policy_sleeves(member))
    assert all_policy_tickers() == tuple(sorted(union))


@pytest.mark.parametrize("scenario_id", ["POL-M2-s6-weights"])
def test_pol_m2_s6_weights(scenario_id: str) -> None:
    """POL-M2-s6-weights"""
    targets = resolve_targets(PolicyId.VTI_VTV, pl.DataFrame(), _SIGNAL_AT)

    assert targets == {"VTI": 0.8, "VTV": 0.2}
    weights = list(targets.values())
    assert min(weights) >= 0.0
    assert abs(sum(weights) - 1.0) <= 1e-6


@pytest.mark.parametrize("scenario_id", ["POL-G03-static-sum-one"])
def test_pol_g03_rejects_naive_signal_at(scenario_id: str) -> None:
    """Static policies ignore prices but must still type-check signal_at."""
    with pytest.raises(ValueError, match="timezone-aware"):
        resolve_targets(PolicyId.WORLD_SPLIT, pl.DataFrame(), _SIGNAL_AT.replace(tzinfo=None))


@pytest.mark.parametrize("scenario_id", ["I9-C-resolve-targets-rejects-r1"])
def test_i9_c_resolve_targets_rejects_r1(scenario_id: str) -> None:
    """I9-C-resolve-targets-rejects-r1"""
    with pytest.raises(PolicyError, match=r"FF_PROXY|research_proxy"):
        resolve_targets(PolicyId.FF_PROXY, pl.DataFrame(), _SIGNAL_AT)

    # R1 is a campaign identity, not a sleeve map: the ingest universe stays frozen.
    union: set[str] = set()
    for member in PolicyId:
        union.update(policy_sleeves(member))
    expected = ("BND", "IEF", "IVV", "QQQ", "TLT", "VEA", "VT", "VTI", "VTV", "VWO")
    assert all_policy_tickers() == tuple(sorted(union)) == expected


_INVVOL_BARS: Final = 64


def _invvol_days() -> tuple[date, ...]:
    return _CALENDAR.sessions(date(2023, 10, 2), date(2024, 1, 31))[:_INVVOL_BARS]


def _invvol_panel() -> pl.DataFrame:
    """VTI vol is twice VEA/VWO vol via alternating +/- amplitude returns."""
    amplitudes = {"VEA": 0.01, "VWO": 0.01, "VTI": 0.02}
    days = _invvol_days()
    spec = spec_for(Dataset.PRICES)
    tickers: list[str] = []
    dates: list[date] = []
    closes: list[float] = []
    for ticker in sorted(amplitudes):
        price = 100.0
        for index, day in enumerate(days):
            sign = 1.0 if index % 2 == 0 else -1.0
            tickers.append(ticker)
            dates.append(day)
            closes.append(price)
            price *= 1.0 + sign * amplitudes[ticker]
    n = len(dates)
    raw = pl.DataFrame(
        {
            "ticker": tickers,
            "date": dates,
            "open": [value * 0.98 for value in closes],
            "high": [value * 1.02 for value in closes],
            "low": [value * 0.97 for value in closes],
            "close": closes,
            "volume": [10_000] * n,
            "adjusted_close": closes,
            "dividend": [0.0] * n,
            "split_factor": [1.0] * n,
            "source": ["synthetic"] * n,
            "retrieved_at": [_RETRIEVED_AT] * n,
        },
        schema=dict(spec.columns),
    )
    return ingest(raw, Dataset.PRICES)


def _delay_last_vti_bar(prices: pl.DataFrame) -> pl.DataFrame:
    last_day = _invvol_days()[-1]
    later = _CALENDAR.close_ts(_CALENDAR.next_session(last_day))
    return prices.with_columns(
        pl.when((pl.col("ticker") == "VTI") & (pl.col("date") == last_day))
        .then(pl.lit(later, dtype=TS_DTYPE))
        .otherwise(pl.col("available_at"))
        .alias("available_at")
    )


@pytest.mark.parametrize("scenario_id", ["POL-G04-invvol-pit"])
def test_pol_g04_invvol_pit(scenario_id: str) -> None:
    """POL-G04-invvol-pit"""
    signal_at = _CALENDAR.close_ts(_invvol_days()[-1])
    prices = _invvol_panel()

    targets = resolve_targets(PolicyId.INV_VOL, prices, signal_at)

    assert abs(sum(targets.values()) - 1.0) <= 1e-6
    assert targets["VEA"] == pytest.approx(targets["VWO"])
    assert targets["VTI"] == pytest.approx(0.5 * targets["VEA"])

    with pytest.raises(PolicyError):
        resolve_targets(PolicyId.INV_VOL, _delay_last_vti_bar(prices), signal_at)


@pytest.mark.parametrize("scenario_id", ["NAM-A-policy-qqq"])
def test_nam_a_policy_qqq(scenario_id: str) -> None:
    """NAM-A-policy-qqq"""
    assert PolicyId.parse("qqq") is PolicyId.QQQ
    assert PolicyId.parse("nasdaq") is PolicyId.QQQ
    assert PolicyId.parse("s8_us_nasdaq") is PolicyId.QQQ
    assert PolicyId.QQQ.value == "qqq"
    assert OPERATIONAL_POLICY_ID is PolicyId.QQQ
    assert PolicyId.parse("vti") is PolicyId.VTI
    assert PolicyId.parse("s1_us") is PolicyId.VTI
    with pytest.raises(ValueError, match="unknown policy"):
        PolicyId.parse("not_a_policy")
    assert hasattr(PolicyId, "S8_US_NASDAQ") is False


@pytest.mark.parametrize("scenario_id", ["PKG-A-no-nested-package"])
def test_pkg_a_no_nested_package(scenario_id: str) -> None:
    """PKG-A-no-nested-package"""
    import importlib

    targets = importlib.import_module("src.policy.targets")
    assert hasattr(targets, "PolicyId")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.etf_manager.policy.targets")


@pytest.mark.parametrize("scenario_id", ["NAM-A01-policy-aliases"])
def test_nam_a01_policy_aliases(scenario_id: str) -> None:
    """NAM-A01-policy-aliases"""
    assert {member.value for member in PolicyId} == {
        "vt",
        "vti",
        "world_split",
        "vt_bnd",
        "vt_treas",
        "inv_vol",
        "vti_vtv",
        "ivv",
        "qqq",
        "ff_proxy",
    }
    assert PolicyId.parse("vti") is PolicyId.parse("s1_us") is PolicyId.VTI
    assert PolicyId.parse("s1_us").value == "vti"
    assert POLICY_ALIASES["s1_us"] is POLICY_ALIASES["us"] is POLICY_ALIASES["vti"] is PolicyId.VTI
    with pytest.raises(ValueError, match="unknown policy"):
        PolicyId.parse("not_a_policy")

    assert resolve_targets(PolicyId.VTI, pl.DataFrame(), _SIGNAL_AT)["VTI"] == 1.0
