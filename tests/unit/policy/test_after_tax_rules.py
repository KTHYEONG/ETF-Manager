"""Invariant guard tests for the after-tax weight-rule family."""

from __future__ import annotations

import calendar
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from src.features.pit_market import PitMarket
from src.policy.after_tax_rules import AfterTaxRuleId, AfterTaxRuleSpec, build_weight_rule, parse_after_tax_rule_spec
from src.policy.weight_rule import CASH_SLEEVE


def _month_dates(start: date, count: int) -> list[date]:
    dates: list[date] = []
    year, month = start.year, start.month
    for _ in range(count):
        last_day = calendar.monthrange(year, month)[1]
        dates.append(date(year, month, last_day))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return dates


def _prices_frame(entries: list[tuple[str, date, float]], signal_at: datetime) -> pl.DataFrame:
    rows = [
        {
            "ticker": ticker,
            "date": day,
            "adjusted_close": price,
            "available_at": signal_at - timedelta(days=1),
        }
        for ticker, day, price in entries
    ]
    return pl.DataFrame(rows)


def _rates_frame(levels: list[tuple[date, float]], signal_at: datetime, series: str = "DTB3") -> pl.DataFrame:
    rows = [
        {
            "series_id": series,
            "observation_date": day,
            "value": value,
            "available_at": signal_at - timedelta(days=1),
        }
        for day, value in levels
    ]
    return pl.DataFrame(rows)


def _market(
    price_entries: list[tuple[str, date, float]],
    rate_levels: list[tuple[date, float]],
    signal_at: datetime,
) -> PitMarket:
    return PitMarket(_prices_frame(price_entries, signal_at), _rates_frame(rate_levels, signal_at))


SIGNAL_AT = datetime(2021, 8, 31, 20, 0, tzinfo=UTC)
MONTHS = _month_dates(date(2020, 8, 31), 13)


def _rate_levels(value: float = 5.0) -> list[tuple[date, float]]:
    return [(day, value) for day in MONTHS]


def test_dual_exit_risk_on_keeps_core() -> None:
    """Dual exit risk-on keeps core."""
    closes = [100.0 + i for i in range(13)]
    entries = [("QQQ", day, price) for day, price in zip(MONTHS, closes, strict=True)]
    entries += [("SOXX", day, 50.0) for day in MONTHS]
    market = _market(entries, _rate_levels(), SIGNAL_AT)
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "dual_momentum_exit",
            "core_targets": {"QQQ": 0.9, "SOXX": 0.1},
            "safe_asset": "CASH",
            "signal_ticker": "QQQ",
            "sma_months": 10,
            "momentum_months": [12],
        }
    )
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    assert dict(rule(SIGNAL_AT, market)) == {"QQQ": 0.9, "SOXX": 0.1}


def test_dual_exit_needs_both_signals_off() -> None:
    """Dual exit with SMA off but momentum on stays risk-on, both off goes safe."""
    # Recent dip below SMA but 12-month gain still beats a 5% hurdle.
    closes = [100.0, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 115.0]
    entries = [("QQQ", day, price) for day, price in zip(MONTHS, closes, strict=True)]
    market = _market(entries, _rate_levels(), SIGNAL_AT)
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "dual_momentum_exit",
            "core_targets": {"QQQ": 0.9, "SOXX": 0.1},
            "safe_asset": "CASH",
            "signal_ticker": "QQQ",
            "sma_months": 10,
            "momentum_months": [12],
        }
    )
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    assert dict(rule(SIGNAL_AT, market)) == {"QQQ": 0.9, "SOXX": 0.1}
    falling = [130.0 - i for i in range(13)]
    entries2 = [("QQQ", day, price) for day, price in zip(MONTHS, falling, strict=True)]
    market2 = _market(entries2, _rate_levels(), SIGNAL_AT)
    assert dict(rule(SIGNAL_AT, market2)) == {"CASH": 1.0}


def test_trend_partial_scales_risk() -> None:
    """Trend partial scales risk when below the SMA."""
    falling = [130.0 - i for i in range(13)]
    entries = [("QQQ", day, price) for day, price in zip(MONTHS, falling, strict=True)]
    market = _market(entries, _rate_levels(), SIGNAL_AT)
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "trend_partial",
            "core_targets": {"QQQ": 0.9, "SOXX": 0.1},
            "safe_asset": "CASH",
            "signal_ticker": "QQQ",
            "sma_months": 10,
            "risk_on_fraction_when_off": 0.5,
        }
    )
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    weights = dict(rule(SIGNAL_AT, market))
    assert weights["QQQ"] == pytest.approx(0.45)
    assert weights["SOXX"] == pytest.approx(0.05)
    assert weights[CASH_SLEEVE] == pytest.approx(0.5)


def test_vol_target_caps_at_one() -> None:
    """Vol target scales down high vol and caps low vol at one."""
    import math

    signal_day = date(2021, 8, 31)
    days = [signal_day - timedelta(days=i) for i in range(63, -1, -1)]

    def _closes(amplitude: float) -> list[float]:
        closes = [100.0]
        for i in range(64 - 1):
            ret = amplitude if i % 2 == 0 else -amplitude
            closes.append(closes[-1] * (1.0 + ret))
        return closes

    amp_high = 0.40 / math.sqrt(252.0)
    amp_low = 0.10 / math.sqrt(252.0)
    high_entries = [("QQQ", day, price) for day, price in zip(days, _closes(amp_high), strict=True)]
    low_entries = [("QQQ", day, price) for day, price in zip(days, _closes(amp_low), strict=True)]
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "vol_target",
            "core_targets": {"QQQ": 0.9, "SOXX": 0.1},
            "safe_asset": "CASH",
            "signal_ticker": "QQQ",
            "vol_target_annual": 0.20,
            "vol_window_sessions": 63,
        }
    )
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    high_market = _market(high_entries, _rate_levels(), SIGNAL_AT)
    high = dict(rule(SIGNAL_AT, high_market))
    assert high["QQQ"] == pytest.approx(0.45, abs=0.05)
    assert high[CASH_SLEEVE] == pytest.approx(0.5, abs=0.05)
    low_market = _market(low_entries, _rate_levels(), SIGNAL_AT)
    low = dict(rule(SIGNAL_AT, low_market))
    assert low["QQQ"] == pytest.approx(0.9, abs=0.02)
    assert low["SOXX"] == pytest.approx(0.1, abs=0.02)


def test_taa_replaces_failing_pick() -> None:
    """TAA top-2 replaces a hurdle-failing pick with the safe asset."""
    qqq = [100.0 + 2 * i for i in range(13)]
    spy = [100.0 - 0.5 * i for i in range(13)]
    efa = [100.0 - i for i in range(13)]
    entries: list[tuple[str, date, float]] = []
    for day, v in zip(MONTHS, qqq, strict=True):
        entries.append(("QQQ", day, v))
    for day, v in zip(MONTHS, spy, strict=True):
        entries.append(("SPY", day, v))
    for day, v in zip(MONTHS, efa, strict=True):
        entries.append(("EFA", day, v))
    market = _market(entries, _rate_levels(), SIGNAL_AT)
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "taa_top_n",
            "universe": ["QQQ", "SPY", "EFA"],
            "momentum_months": [12],
            "top_n": 2,
            "safe_asset": "IEF",
        }
    )
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    assert dict(rule(SIGNAL_AT, market)) == {"QQQ": 0.5, "IEF": 0.5}


def test_gem_hurdle_to_safety() -> None:
    """GEM below-hurdle signal moves fully to safety."""
    spy = [130.0 - i for i in range(13)]
    efa = [100.0 + i for i in range(13)]
    entries = [("SPY", day, v) for day, v in zip(MONTHS, spy, strict=True)]
    entries += [("EFA", day, v) for day, v in zip(MONTHS, efa, strict=True)]
    market = _market(entries, _rate_levels(), SIGNAL_AT)
    spec = parse_after_tax_rule_spec(
        {"rule_id": "gem", "signal_ticker": "SPY", "universe": ["SPY", "EFA"], "safe_asset": "IEF"}
    )
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    assert dict(rule(SIGNAL_AT, market)) == {"IEF": 1.0}


def test_glide_path_switches_at_boundary() -> None:
    """Glide path holds core before the boundary and safety after it."""
    market = _market([("QQQ", date(2021, 8, 31), 100.0)], _rate_levels(), SIGNAL_AT)
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "glide_path",
            "core_targets": {"QQQ": 0.9, "SOXX": 0.1},
            "safe_asset": "IEF",
            "glide_months": 60,
        }
    )
    rule = build_weight_rule(spec, horizon_end=date(2026, 7, 31))
    before = datetime(2021, 7, 30, 20, 0, tzinfo=UTC)
    after = datetime(2021, 8, 31, 20, 0, tzinfo=UTC)
    assert dict(rule(before, market)) == {"QQQ": 0.9, "SOXX": 0.1}
    assert dict(rule(after, market)) == {"IEF": 1.0}


def test_spec_validation_rejects_missing_fraction() -> None:
    """TREND_PARTIAL without its fraction is rejected."""
    with pytest.raises(ValueError, match="risk_on_fraction_when_off"):
        parse_after_tax_rule_spec(
            {
                "rule_id": "trend_partial",
                "core_targets": {"QQQ": 1.0},
                "safe_asset": "CASH",
                "signal_ticker": "QQQ",
                "sma_months": 10,
            }
        )


def test_rule_look_ahead_invariance() -> None:
    """Perturbing rows after the signal instant leaves weights unchanged."""
    closes = [100.0 + i for i in range(13)]
    entries = [("QQQ", day, price) for day, price in zip(MONTHS, closes, strict=True)]
    base = _market(entries, _rate_levels(), SIGNAL_AT)
    future_day = date(2021, 9, 30)
    base_prices = _prices_frame(entries, SIGNAL_AT)
    extended = pl.concat(
        [
            base_prices,
            pl.DataFrame(
                [
                    {
                        "ticker": "QQQ",
                        "date": future_day,
                        "adjusted_close": 9999.0,
                        "available_at": SIGNAL_AT + timedelta(days=5),
                    }
                ]
            ),
        ]
    )
    rates = _rates_frame(_rate_levels(), SIGNAL_AT)
    extended_market = PitMarket(extended, rates)
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "dual_momentum_exit",
            "core_targets": {"QQQ": 0.9, "SOXX": 0.1},
            "safe_asset": "CASH",
            "signal_ticker": "QQQ",
            "sma_months": 10,
            "momentum_months": [12],
        }
    )
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    assert dict(rule(SIGNAL_AT, base)) == dict(rule(SIGNAL_AT, extended_market))


def test_static_rule_returns_core_and_metadata() -> None:
    """STATIC always returns core with correct tickers and no cash-rate need."""
    market = _market([("QQQ", date(2021, 8, 31), 100.0)], _rate_levels(), SIGNAL_AT)
    spec = parse_after_tax_rule_spec({"rule_id": "static", "core_targets": {"QQQ": 0.9, "SOXX": 0.1}})
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    assert dict(rule(SIGNAL_AT, market)) == {"QQQ": 0.9, "SOXX": 0.1}
    assert rule.tickers == frozenset({"QQQ", "SOXX"})
    assert rule.requires_cash_rate is False


def test_gem_picks_higher_momentum_member() -> None:
    """GEM above-hurdle signal buys the stronger universe member."""
    spy = [100.0 + 2 * i for i in range(13)]
    efa = [100.0 + 0.5 * i for i in range(13)]
    entries = [("SPY", day, v) for day, v in zip(MONTHS, spy, strict=True)]
    entries += [("EFA", day, v) for day, v in zip(MONTHS, efa, strict=True)]
    market = _market(entries, _rate_levels(), SIGNAL_AT)
    spec = parse_after_tax_rule_spec(
        {"rule_id": "gem", "signal_ticker": "SPY", "universe": ["SPY", "EFA"], "safe_asset": "IEF"}
    )
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    assert dict(rule(SIGNAL_AT, market)) == {"SPY": 1.0}
    assert rule.requires_cash_rate is True


def test_core_satellite_blends_taa_weights() -> None:
    """CORE_SATELLITE blends scaled core with the satellite TAA sleeve."""
    qqq = [100.0 + 2 * i for i in range(13)]
    spy = [100.0 - 0.5 * i for i in range(13)]
    efa = [100.0 - i for i in range(13)]
    entries: list[tuple[str, date, float]] = []
    for day, v in zip(MONTHS, qqq, strict=True):
        entries.append(("QQQ", day, v))
    for day, v in zip(MONTHS, spy, strict=True):
        entries.append(("SPY", day, v))
    for day, v in zip(MONTHS, efa, strict=True):
        entries.append(("EFA", day, v))
    market = _market(entries, _rate_levels(), SIGNAL_AT)
    spec = parse_after_tax_rule_spec(
        {
            "rule_id": "core_satellite_taa",
            "core_targets": {"QQQ": 0.9, "SOXX": 0.1},
            "universe": ["QQQ", "SPY", "EFA"],
            "momentum_months": [12],
            "top_n": 2,
            "satellite_weight": 0.3,
            "safe_asset": "IEF",
        }
    )
    rule = build_weight_rule(spec, horizon_end=date(2031, 8, 30))
    weights = dict(rule(SIGNAL_AT, market))
    assert weights["QQQ"] == pytest.approx(0.63 + 0.15)
    assert weights["SOXX"] == pytest.approx(0.07)
    assert weights["IEF"] == pytest.approx(0.15)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_parse_rejects_unknown_and_misplaced_fields() -> None:
    """Unknown ids, unknown fields, and misplaced params fail closed."""
    with pytest.raises(ValueError, match="unknown rule id"):
        parse_after_tax_rule_spec({"rule_id": "nope", "core_targets": {"QQQ": 1.0}})
    with pytest.raises(ValueError, match="unknown rule fields"):
        parse_after_tax_rule_spec({"rule_id": "static", "core_targets": {"QQQ": 1.0}, "bogus": 1})
    with pytest.raises(ValueError, match="does not apply"):
        parse_after_tax_rule_spec(
            {"rule_id": "static", "core_targets": {"QQQ": 1.0}, "safe_asset": "CASH"}
        )
    with pytest.raises(ValueError, match="top_n <= len"):
        parse_after_tax_rule_spec(
            {
                "rule_id": "taa_top_n",
                "universe": ["QQQ", "SPY"],
                "momentum_months": [12],
                "top_n": 3,
                "safe_asset": "IEF",
            }
        )
    with pytest.raises(ValueError, match="requires core_targets"):
        build_weight_rule(
            AfterTaxRuleSpec(rule_id=AfterTaxRuleId.STATIC, core_targets={}),
            horizon_end=date(2031, 8, 30),
        )
